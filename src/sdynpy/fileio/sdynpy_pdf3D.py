# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:44:09 2025

@author: dprohe
"""

from ..core.sdynpy_geometry import (Geometry,CoordinateSystemArray,IGNORE_PLOTS,
                                    global_coord,_beam_elem_types,_face_element_types,
                                    _solid_element_types, _vtk_element_map,
                                    _element_types,split_list,GeometryPlotter)
from ..core.sdynpy_shape import ShapeArray
from ..core.sdynpy_colors import colormap

import datetime
import warnings
import numpy as np
import pyvista as pv

try:
    from vtk import vtkU3DExporter
except ImportError:
    vtkU3DExporter = None

def _create_u3d_for_animation(output_name, geometry,node_size : int = 5, line_width: int = 1, opacity: float = 1.0,
                             show_edges: bool =False):
    if IGNORE_PLOTS:
        raise ValueError('Cannot Create VTK Objects, No GPU Found')
    if vtkU3DExporter is None:
        raise ValueError('Cannot Import vtkU3DExporter.  It must first be installed with `pip install vtk-u3dexporter`')
    
    # Get part information
    nodes = geometry.node.flatten()
    css = geometry.coordinate_system.flatten()
    elems = geometry.element.flatten()
    tls = geometry.traceline.flatten()

    coordinate_system = CoordinateSystemArray(nodes.shape)
    for key, node in nodes.ndenumerate():
        coordinate_system[key] = css[np.where(geometry.coordinate_system.id == node.def_cs)]
    global_node_positions = global_coord(coordinate_system, nodes.coordinate)
    node_index_dict = {node.id[()]: index[0] for index, node in nodes.ndenumerate()}
    node_index_map = np.vectorize(node_index_dict.__getitem__)

    # Now go through and get the element and line connectivity
    face_element_connectivity = []
    face_element_colors = []
    node_colors = []
    line_connectivity = []
    line_colors = []
    for index, node in nodes.ndenumerate():
        # element_connectivity.append(1)
        # element_connectivity.append(index[0])
        # element_colors.append(node.color)
        node_colors.append(node.color)
    for index, element in elems.ndenumerate():
        # Check which type of element it is
        if element.type in _beam_elem_types:  # Beamlike element, use a line
            try:
                line_connectivity.append(node_index_map(element.connectivity))
            except KeyError:
                raise KeyError(
                    'Element {:} contains a node id not found in the node array'.format(element.id))
            line_colors.append(element.color)
        elif element.type in _face_element_types:
            try:
                if len(element.connectivity) == 3:
                    face_element_connectivity.append(node_index_map(element.connectivity))
                    face_element_colors.append(element.color)
                else:
                    face_element_connectivity.append(node_index_map(element.connectivity[[0,1,3]]))
                    face_element_colors.append(element.color)
                    face_element_connectivity.append(node_index_map(element.connectivity[[1,2,3]]))
                    face_element_colors.append(element.color)
            except KeyError:
                raise KeyError(
                    'Element {:} contains a node id not found in the node array'.format(element.id))
        elif element.type in _solid_element_types:
            warnings.warn('Solid Elements are currently not supported and will be skipped!')
        else:
            raise ValueError('Unknown element type {:}'.format(element.type))

    for index, tl in tls.ndenumerate():
        for conn_group in split_list(tl.connectivity, 0):
            if len(conn_group) == 0:
                continue
            try:
                mapped_conn_group = node_index_map(conn_group)
            except KeyError:
                raise KeyError(
                    'Traceline {:} contains a node id not found in the node array'.format(tl.id))
            for indices in zip(mapped_conn_group[:-1],mapped_conn_group[1:]):
                line_connectivity.append(indices)
                line_colors.append(tl.color)

    # Now we start to plot
    plotter = GeometryPlotter(editor=True)
    face_map = []
    for conn,color in zip(face_element_connectivity,face_element_colors):
        node_indices,inverse = np.unique(conn,return_inverse=True)
        node_positions = global_node_positions[node_indices][inverse]
        node_connectivity = np.arange(node_positions.shape[0])
        mesh = pv.PolyData(node_positions,faces=[len(node_connectivity)]+[c for c in node_connectivity])
        mesh.cell_data['color'] = color
        mesh.SetObjectName('Elem: {:} {:} {:}'.format(*geometry.node.id[node_indices[inverse]]))
        plotter.add_mesh(mesh,scalars='color',cmap=colormap, clim = [0,15],
                         show_edges=show_edges, show_scalar_bar = False,
                         line_width=line_width,opacity=opacity,label='Elem: {:} {:} {:}'.format(*geometry.node.id[node_indices[inverse]]))
        face_map.append(geometry.node.id[node_indices[inverse]])

    line_map = []
    for conn,color in zip(line_connectivity,line_colors):
        node_indices,inverse = np.unique(conn,return_inverse=True)
        node_positions = global_node_positions[node_indices][inverse]
        node_connectivity = np.arange(node_positions.shape[0])
        mesh = pv.PolyData(node_positions,lines=[len(node_connectivity)]+[c for c in node_connectivity])
        mesh.cell_data['color'] = color
        mesh.SetObjectName('Line: {:} {:}'.format(*geometry.node.id[node_indices[inverse]]))
        plotter.add_mesh(mesh,scalars='color',cmap=colormap, clim = [0,15],
                         show_edges=show_edges, show_scalar_bar = False,
                         line_width=line_width,opacity=opacity,label='Line: {:} {:}'.format(*geometry.node.id[node_indices[inverse]]))
        line_map.append(geometry.node.id[node_indices[inverse]])
    
    point_map = []
    for node,position,color in zip(geometry.node.id,global_node_positions,geometry.node.color):
        if node_size > 0:
            mesh = pv.PolyData(position)
            mesh.cell_data['color'] = color
            plotter.add_mesh(mesh, scalars = color, cmap=colormap, clim=[0,15],
                             show_edges=show_edges, show_scalar_bar=False, point_size=node_size,
                             opacity=opacity)
            point_map.append(node)
        
    exporter = vtkU3DExporter.vtkU3DExporter()
    exporter.SetFileName(output_name)
    exporter.SetInput(plotter.render_window)
    exporter.Write()
    
    plotter.close()
    
    return face_map, line_map, point_map

def create_animated_modeshape_content(geometry,shapes = None,u3d_name='geometry',
                                      js_name = 'mode_shapes.js',one_js=True,
                                      node_size = 5, line_width = 2,
                                      opacity = 1, show_edges = False,
                                      debug_js = False, displacement_scale = 1.0):
    """
    Creates files that can be embedded into a PDF to create interactive content
    
    This function produces a Universal3D model containing Geometry information.
    This function also produces one or more JavaScript programs that are used
    by the PDF to animate the model data.  The model and programs can be
    embedded into a PDF either by using Adobe Acrobat Professional, or by using
    LaTeX with the media9 package to build a PDF document.

    Parameters
    ----------
    geometry : Geometry
        A SDynPy Geometry object that will be exported to Universal3D format.
    shapes : ShapeArray, optional
        A SDynPy ShapeArray object that contains the mode shapes that will be
        used to animate the model in the final PDF.  If not specified, only the
        geometry will be exported.
    u3d_name : str, optional
        The file path and name of the Universal3D file that will be created
        from the SDynPy Geometry.  Note that the .u3d extension will be
        automatically appended to the file path. The default is 'geometry'.
    js_name : str, optional
        The file path and name of the JavaScript file(s) that will be created
        from the geometry and shape information.  The final file name will be
        constructed from the code `js_name.format(i+1)` where `i` is the shape
        index.  This allows for a different file name to be produced for each
        mode if `one_js` is `False`. The default is 'mode_shapes.js'.
    one_js : bool, optional
        If True, only one JavaScript file will be produced containing all mode
        shape information.  If False, one JavaScript file will be produced per
        mode in the `shapes` argument, with each script having that shape as
        shape that is plotted first.  Even if `one_js` is False, all JavaScript
        programs will have all modes in them. The default is True.
    node_size : int, optional
        The size of the rendered node in pixels. The default is 5.
    line_width : int, optional
        The width of lines rendered in the model. The default is 2.  Note that
        lines with a width of 1 may not show their color well in the final PDF.
    opacity : float, optional
        The opacity of the model, from 0 (transparent) to 1 (opaque). The
        default is 1.
    show_edges : bool, optional
        If True, edges will be drawn on element faces. The default is False.
    debug_js : bool or str, optional
        If True, the JavaScript programs will have various print statements
        enabled to help users debug.  If False, these will be commented out.
        String values of 'map', 'point', 'line', or 'face' can also be specified
        to only print debug text for those operations. The default is False.
    displacement_scale : float, optional
        A scale factor applied to the shape displacements to achieve a
        reasonable deflection size. The default is 1.0.

    Notes
    -----
    There is no support currently for Solid elements (e.g. tets or hexes).
    Only surface elements (tris and quads), lines (beam elements or tracelines),
    and points (nodes) are currently supported.

    Returns
    -------
    None.

    """
    face_map, line_map, point_map = _create_u3d_for_animation(u3d_name,
                                                              geometry, 
                                                              node_size,
                                                              line_width,
                                                              opacity,
                                                              show_edges)
    
    if shapes is None:
        return
    
    # Need to get the shape information in the global coordinate system
    node_displacement_dict = {}
    for node_id in geometry.node.id:
        node_displacement_dict[node_id] = np.zeros((shapes.size,3))
        for shape_index,shape in enumerate(shapes.flatten()):
            coords = abs(shape.coordinate[(shape.coordinate.node == node_id) & np.in1d(abs(shape.coordinate.direction),[1,2,3])])
            shape_data = shape[coords]
            coord_directions = geometry.global_deflection(coords)
            global_deflection = np.sum(shape_data[:,np.newaxis]*coord_directions,axis=0)
            node_displacement_dict[node_id][shape_index] = global_deflection
    
    node_position_dict = {node_id:position for node_id,position in zip(geometry.node.id,geometry.global_node_coordinate())}
    
    if one_js:
        js_indices = [0]
    else:
        js_indices = np.arange(shapes.size)
    
    for i in js_indices:
        js_output_name = js_name.format(i+1)
        with open(js_output_name,'w') as f:
            f.write(
    """//Written by sdynpy_pdf3D.py on {date:}
    
    function pause(){{if(speed)lastspeed=speed; speed=0;}}
    function play(){{speed=lastspeed;}}
    function scaleSpeed(s){{lastspeed*=s; if(speed) speed=lastspeed;}}
    function scale(s){{scale*=s;}}
    
    var omega0=Math.PI; // init. angular frequency (half turn per second)
    var speed=1;        // speed multiplier
    var lastspeed=1;
    var theta=0;
    var scale={displacement_scale:};
    var shape={shape:};
    var num_shapes = {num_shapes:};
    var panel = 1
    var node_positions = {{}};
    var node_displacements = {{}};
    
    // Node and Shape Information
    """.format(date=datetime.datetime.now(),
           shape=i+1,
           num_shapes=shapes.size,
           displacement_scale=displacement_scale))
            for node_id in node_position_dict:
                f.write('\nnode_positions[{:}] = Vector3({:},{:},{:});\n'.format(node_id,*node_position_dict[node_id]))
                f.write('node_displacements[{:}] = [\n'.format(node_id))
                for shape_vector in node_displacement_dict[node_id]:
                    f.write('    Vector3({:},{:},{:}),\n'.format(*shape_vector))
                f.write('  ];\n')
            f.write('\n// Frequency Information\n')
            f.write('var mode_frequencies = [];\n')
            for i,shape in enumerate(shapes.flatten()):
                f.write('mode_frequencies[{:}] = {:0.2f};\n'.format(i,shape.frequency))
            # Now we need to go through and collect the node information and
            # Map it back to node id numbers
            f.write('\n// Connectivity Maps\n')
            f.write('face_map = [\n')
            for face in face_map:
                f.write('   {:},\n'.format(str(list(face))))
            f.write('    ];\n')
            f.write('line_map = [\n')
            for line in line_map:
                f.write('   {:},\n'.format(str(list(line))))
            f.write('    ];\n')
            f.write('point_map = [\n')
            for point in point_map:
                f.write('   {:},\n'.format(point))
            f.write('    ];\n')
            f.write("""
    var scene = this.scene;
    var face_nodes = [];
    var line_nodes = [];
    var point_nodes = [];
    for (i=0; i < scene.nodes.count; i++){{
        name = scene.nodes.getByIndex(i).name
        if (name.slice(0,4) === "Mesh"){{
            {debug:}host.console.println("Mesh found at node "+i+" "+ name)
            face_nodes.push(i);
        }} else if (name.slice(0,4) === "Line"){{
            {debug:}host.console.println("Line found at node "+i+" "+ name)
            line_nodes.push(i);
        }} else if (name.slice(0,4) === "Poin"){{
            {debug:}host.console.println("Point found at node "+i+" "+ name)
            point_nodes.push(i);
        }} else {{
            {debug:}host.console.println("Unknown node "+i+" "+name)
        }}
    }}
    """.format(debug='' if debug_js is True or debug_js=='map' else '// '))

# Now we need to go through and apply displacements
            f.write('''
    // Translate Nodes
    for (i=0; i < point_nodes.length; i++){{
        geometry_node_id = point_map[i];
        model_node = scene.nodes.getByIndex(point_nodes[i]);
        {debug:}host.console.println("Point Index "+i+" corresponds to model node "+model_node.name+" and geometry node id "+geometry_node_id);
        model_node.transform.setIdentity();
        model_node.transform.translateInPlace(node_displacements[geometry_node_id][shape-1].scale(scale));
        {debug:}host.console.println(model_node.transform.toString()+"\\n")
    }}'''.format(debug='' if debug_js is True or debug_js=='point' else '// '))
            f.write('''
    // Translate Lines
    
    // Computation variables
    var p1;    // position of node 1
    var p2;    // position of node 2
    var d1;    // displacement of node 1
    var d2;    // displacement of node 2
    var v1;    // traceline vector from p1 to p2 (p2-p1)
    var v2;    // traceline vector after modification ((d2+p2)-(d1+p1))
    var vc1;   // traceline center ((p2+p1)/2)
    var vc2;   // traceline center after modification ((p2+d2+p1+d1)/2)
    var axis;  // axis of rotation v1 x v2 unit vector
    var angle; // angle of rotation cos(angle) = v1.v2/(|v1||v2|)
    var c;     // scale factor |v2|/|v1|
    var t;     // translate vc2 - c*vc1
    
    // Go through traceline nodes
    for (i=0; i < line_nodes.length; i++){{
        model_node = scene.nodes.getByIndex(line_nodes[i]);
        geometry_node_ids = line_map[i];
        {debug:}host.console.println("Line Index "+i+" corresponds to model node "+model_node.name+" and geometry node ids "+geometry_node_ids);
        model_node.transform.setIdentity();
        p1 = node_positions[geometry_node_ids[0]];
        p2 = node_positions[geometry_node_ids[1]];
        d1 = node_displacements[geometry_node_ids[0]][shape-1].scale(scale);
        d2 = node_displacements[geometry_node_ids[1]][shape-1].scale(scale);
        v1 = p2.add(p1.scale(-1));
        v2 = p2.add(d2).add(p1.add(d1).scale(-1));
        vc1 = p2.add(p1).scale(0.5);
        vc2 = p2.add(d2).add(p1).add(d1).scale(0.5);
        axis = v1.cross(v2);
        angle = Math.acos(v1.dot(v2)/(Math.sqrt(v1.dot(v1))*Math.sqrt(v2.dot(v2))));
        c = Math.sqrt(v2.dot(v2)/v1.dot(v1));
        t = vc2.add(vc1.scale(-c));
        {debug:}host.console.println("Transforming TL: "+geometry_node_ids);
        {debug:}host.console.println("  p1: "+p1)
        {debug:}host.console.println("  p2: "+p2)
        {debug:}host.console.println("  d1: "+d1)
        {debug:}host.console.println("  d2: "+d2)
        {debug:}host.console.println("  v1: "+v1)
        {debug:}host.console.println("  v2: "+v2)
        {debug:}host.console.println("  vc1: "+vc1)
        {debug:}host.console.println("  vc2: "+vc2)
        {debug:}host.console.println("  axis: "+axis)
        {debug:}host.console.println("  angle: "+angle)
        {debug:}host.console.println("  c: "+c)
        {debug:}host.console.println("  t: "+t)
        {debug:}host.console.println("\\n  Transform prior to anything:")
        {debug:}host.console.println(model_node.transform.toString())
        //model_node.transform.rotateAboutLineInPlace(angle,vc1,vc1.add(axis)); //There seems to be a bug with this command...
        model_node.transform.translateInPlace(vc1.scale(-1));
        model_node.transform.rotateAboutVectorInPlace(angle,axis);
        model_node.transform.translateInPlace(vc1);
        {debug:}host.console.println("\\n  Transform after rotation:")
        {debug:}host.console.println(model_node.transform.toString())
        model_node.transform.scaleInPlace(c,c,c);
        {debug:}host.console.println("\\n  Transform after scaling:")
        {debug:}host.console.println(model_node.transform.toString())
        model_node.transform.translateInPlace(t);
        {debug:}host.console.println("\\n  Transform after translation:")
        {debug:}host.console.println(model_node.transform.toString())
    }}'''.format(debug='' if debug_js is True or debug_js=='line' else '// '))

            f.write('''
    // We want to compute the SVD of the transformation, so we will define a number of functions here
    var _gamma = 5.828427124;
    var _cstar = 0.923879532;
    var _sstar = 0.3826834323;
    var EPSILON = 1e-6;
    
    function condSwap(c, X, Y)
    {
        // used in step 2
        // var Z = X;
        // X = c ? Y : X;
        // Y = c ? Z : Y;
        return (c ? [Y,X] : [X,Y])
    }
    
    function condNegSwap(c, X, Y)
    {
        // used in step 2 and 3
        // var Z = -X;
        // X = c ? Y : X;
        // Y = c ? Z : Y;
        return (c ? [Y,-X] : [X,Y])
    }
    
    // matrix multiplication M = A * B
    function multAB(a11, a12, a13,a21, a22, a23, a31, a32, a33, b11, b12, b13, b21, b22, b23, b31, b32, b33)
    {
        var m11=a11*b11 + a12*b21 + a13*b31; var m12=a11*b12 + a12*b22 + a13*b32; var m13=a11*b13 + a12*b23 + a13*b33;
        var m21=a21*b11 + a22*b21 + a23*b31; var m22=a21*b12 + a22*b22 + a23*b32; var m23=a21*b13 + a22*b23 + a23*b33;
        var m31=a31*b11 + a32*b21 + a33*b31; var m32=a31*b12 + a32*b22 + a33*b32; var m33=a31*b13 + a32*b23 + a33*b33;
        return [m11,m12,m13,m21,m22,m23,m31,m32,m33];
    }
    
    // matrix multiplication M = Transpose[A] * B
    function multAtB(a11, a12, a13,
                        a21, a22, a23,
                        a31, a32, a33,
                        b11, b12, b13,
                        b21, b22, b23,
                        b31, b32, b33)
    {
      var m11=a11*b11 + a21*b21 + a31*b31; var m12=a11*b12 + a21*b22 + a31*b32; var m13=a11*b13 + a21*b23 + a31*b33;
      var m21=a12*b11 + a22*b21 + a32*b31; var m22=a12*b12 + a22*b22 + a32*b32; var m23=a12*b13 + a22*b23 + a32*b33;
      var m31=a13*b11 + a23*b21 + a33*b31; var m32=a13*b12 + a23*b22 + a33*b32; var m33=a13*b13 + a23*b23 + a33*b33;
      return [m11,m12,m13,m21,m22,m23,m31,m32,m33];
    }
    
    function quatToMat3(x,y,z,w)
    {
    	var qxx = x*x;
    	var qyy = y*y;
    	var qzz = z*z;
    	var qxz = x*z;
    	var qxy = x*y;
    	var qyz = y*z;
    	var qwx = w*x;
    	var qwy = w*y;
    	var qwz = w*z;
    
    	var m11=1 - 2*(qyy + qzz); var m12=2*(qxy - qwz); var m13=2*(qxz + qwy);
        var m21=2*(qxy + qwz); var m22=1 - 2*(qxx + qzz); var m23=2*(qyz - qwx);
        var m31=2*(qxz - qwy); var m32=2*(qyz + qwx); var m33=1 - 2*(qxx + qyy);
        return [m11,m12,m13,m21,m22,m23,m31,m32,m33];
    }
    
    function approximateGivensQuaternion(a11, a12, a22)
    {
    /*
         * Given givens angle computed by approximateGivensAngles,
         * compute the corresponding rotation quaternion.
         */
        var ch = 2*(a11-a22);
        var sh = a12;
        var b = _gamma*sh*sh < ch*ch;
        var w = 1/Math.sqrt(ch*ch+sh*sh);
        ch=b?w*ch:_cstar;
        sh=b?w*sh:_sstar;
        return [ch,sh]
    }
    
    function jacobiConjugation(x, y, z, s11, s21, s22, s31, s32, s33, qV)
    {
        var return_val = approximateGivensQuaternion(s11,s21,s22);
        var ch = return_val[0];
        var sh = return_val[1];
    
    	var  scale = ch*ch+sh*sh;
        var a = (ch*ch-sh*sh)/scale;
        var b = (2*sh*ch)/scale;
    
        // make temp copy of S
        var _s11 = s11;
    	var _s21 = s21; var _s22 = s22;
    	var _s31 = s31; var _s32 = s32; var _s33 = s33;
    
    	// perform conjugation S = Q''*S*Q
    	// Q already implicitly solved from a, b
        s11 = a*(a*_s11 + b*_s21) + b*(a*_s21 + b*_s22);
        s21 = a*(-b*_s11 + a*_s21) + b*(-b*_s21 + a*_s22);	s22=-b*(-b*_s11 + a*_s21) + a*(-b*_s21 + a*_s22);
        s31 = a*_s31 + b*_s32;								s32=-b*_s31 + a*_s32; s33=_s33;
    
    	// update cumulative rotation qV
        var tmp = [];
        tmp[0]=qV[0]*sh;
        tmp[1]=qV[1]*sh;
        tmp[2]=qV[2]*sh;
        sh *= qV[3];
    
        qV[0] *= ch;
        qV[1] *= ch;
        qV[2] *= ch;
        qV[3] *= ch;
    
        // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
        // for (p,q) = ((0,1),(1,2),(0,2))
        qV[z] += sh;
        qV[3] -= tmp[z]; // w
        qV[x] += tmp[y];
        qV[y] -= tmp[x];
    
        // re-arrange matrix for next iteration
        _s11 = s22;
    	_s21 = s32; _s22 = s33;
    	_s31 = s21; _s32 = s31; _s33 = s11;
    	s11 = _s11;
    	s21 = _s21; s22 = _s22;
    	s31 = _s31; s32 = _s32; s33 = _s33;
        
        return [s11,s21,s22,s31,s32,s33,qV]
    
    }
    
    function dist2(x, y, z)
    {
        return x*x+y*y+z*z;
    }
    
    function jacobiEigenanlysis(s11,s21,s22,s31,s32,s33)
    {
        var return_val;
        var qV = []
        qV[3]=1; qV[0]=0;qV[1]=0;qV[2]=0; // follow same indexing convention as GLM
        for (var i=0;i<4;i++)
    	{
    		// we wish to eliminate the maximum off-diagonal element
            // on every iteration, but cycling over all 3 possible rotations
            // in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
            //  asymptotic convergence
            return_val = jacobiConjugation(0,1,2,s11,s21,s22,s31,s32,s33,qV); // p,q = 0,1
            s11 = return_val[0]; s21 = return_val[1]; s22 = return_val[2]; s31 = return_val[3];
            s32 = return_val[4]; s33 = return_val[5]; qV = return_val[6];
            return_val = jacobiConjugation(1,2,0,s11,s21,s22,s31,s32,s33,qV); // p,q = 1,2
            s11 = return_val[0]; s21 = return_val[1]; s22 = return_val[2]; s31 = return_val[3];
            s32 = return_val[4]; s33 = return_val[5]; qV = return_val[6];
            return_val = jacobiConjugation(2,0,1,s11,s21,s22,s31,s32,s33,qV); // p,q = 0,2
            s11 = return_val[0]; s21 = return_val[1]; s22 = return_val[2]; s31 = return_val[3];
            s32 = return_val[4]; s33 = return_val[5]; qV = return_val[6];
    	}
        return [s11,s21,s22,s31,s32,s33,qV]
    }
    
    function sortSingularValues(b11, b12, b13,
    							b21, b22, b23,
    							b31, b32, b33,
    							v11, v12, v13,
    							v21, v22, v23,
                                v31, v32, v33)
    {
        var return_val;
        var rho1 = dist2(b11,b21,b31);
        var rho2 = dist2(b12,b22,b32);
        var rho3 = dist2(b13,b23,b33);
    	var c;
        c = rho1 < rho2;
        return_val = condNegSwap(c,b11,b12);
        b11 = return_val[0]; b12 = return_val[1];
        return_val = condNegSwap(c,v11,v12);
        v11 = return_val[0]; v12 = return_val[1];
        return_val = condNegSwap(c,b21,b22); 
        b21 = return_val[0]; b22 = return_val[1];
        return_val = condNegSwap(c,v21,v22); 
        v21 = return_val[0]; v22 = return_val[1];
        return_val = condNegSwap(c,b31,b32); 
        b31 = return_val[0]; b32 = return_val[1];
        return_val = condNegSwap(c,v31,v32); 
        v31 = return_val[0]; v32 = return_val[1];
        return_val = condSwap(c,rho1,rho2); 
        rho1 = return_val[0]; rho2 = return_val[1];
        c = rho1 < rho3;
        return_val = condNegSwap(c,b11,b13); 
        b11 = return_val[0]; b13 = return_val[1]; 
        return_val = condNegSwap(c,v11,v13); 
        v11 = return_val[0]; v13 = return_val[1];
        return_val = condNegSwap(c,b21,b23); 
        b21 = return_val[0]; b23 = return_val[1]; 
        return_val = condNegSwap(c,v21,v23); 
        v21 = return_val[0]; v23 = return_val[1];
        return_val = condNegSwap(c,b31,b33); 
        b31 = return_val[0]; b33 = return_val[1]; 
        return_val = condNegSwap(c,v31,v33); 
        v31 = return_val[0]; v33 = return_val[1];
        return_val = condSwap(c,rho1,rho3); 
        rho1 = return_val[0]; rho3 = return_val[1];
        c = rho2 < rho3;
        return_val = condNegSwap(c,b12,b13); 
        b12 = return_val[0]; b13 = return_val[1]; 
        return_val = condNegSwap(c,v12,v13); 
        v12 = return_val[0]; v13 = return_val[1];
        return_val = condNegSwap(c,b22,b23); 
        b22 = return_val[0]; b23 = return_val[1]; 
        return_val = condNegSwap(c,v22,v23); 
        v22 = return_val[0]; v23 = return_val[1];
        return_val = condNegSwap(c,b32,b33); 
        b32 = return_val[0]; b33 = return_val[1]; 
        return_val = condNegSwap(c,v32,v33); 
        v32 = return_val[0]; v33 = return_val[1];
        return [b11, b12, b13, b21, b22, b23, b31, b32, b33, v11, v12, v13, v21, v22, v23, v31, v32, v33];
    }
    
    function QRGivensQuaternion(a1, a2)
    {
        // a1 = pivot point on diagonal
        // a2 = lower triangular entry we want to annihilate
        var rho = Math.sqrt(a1*a1 + a2*a2);
    
        var sh = rho > EPSILON ? a2 : 0;
        var ch = Math.abs(a1) + Math.max(rho,EPSILON);
        var b = a1 < 0;
        var return_val = condSwap(b,sh,ch);
        sh = return_val[0]; ch = return_val[1];
        var w = 1/Math.sqrt(ch*ch+sh*sh);
        ch *= w;
        sh *= w;
        return [ch,sh];
    }
    
    function QRDecomposition(b11, b12, b13,
    							b21, b22, b23,
    							b31, b32, b33)
    {    
        var ch1,sh1,ch2,sh2,ch3,sh3;
        var a,b;
        var return_val;
        
        // first givens rotation (ch,0,0,sh)
        return_val = QRGivensQuaternion(b11,b21);
        ch1 = return_val[0]; sh1 = return_val[1]
        a=1-2*sh1*sh1;
        b=2*ch1*sh1;
        // apply B = Q'' * B
        var r11=a*b11+b*b21;  var r12=a*b12+b*b22;  var r13=a*b13+b*b23;
        var r21=-b*b11+a*b21; var r22=-b*b12+a*b22; var r23=-b*b13+a*b23;
        var r31=b31;          var r32=b32;          var r33=b33;
      
        // second givens rotation (ch,0,-sh,0)
        return_val = QRGivensQuaternion(r11,r31);
        ch2 = return_val[0]; sh2 = return_val[1];
        a=1-2*sh2*sh2;
        b=2*ch2*sh2;
        // apply B = Q'' * B;
        b11=a*r11+b*r31;  b12=a*r12+b*r32;  b13=a*r13+b*r33;
        b21=r21;           b22=r22;           b23=r23;
        b31=-b*r11+a*r31; b32=-b*r12+a*r32; b33=-b*r13+a*r33;
    
        // third givens rotation (ch,sh,0,0)
        return_val = QRGivensQuaternion(b22,b32);
        ch3 = return_val[0]; sh3 = return_val[1];
        a=1-2*sh3*sh3;
        b=2*ch3*sh3;
        // R is now set to desired value
        r11=b11;             r12=b12;           r13=b13;
        r21=a*b21+b*b31;     r22=a*b22+b*b32;   r23=a*b23+b*b33;
        r31=-b*b21+a*b31;    r32=-b*b22+a*b32;  r33=-b*b23+a*b33;
    
        // construct the cumulative rotation Q=Q1 * Q2 * Q3
        // the number of floating point operations for three quaternion multiplications
        // is more or less comparable to the explicit form of the joined matrix.
        // certainly more memory-efficient!
        var sh12=sh1*sh1;
        var sh22=sh2*sh2;
        var sh32=sh3*sh3;
    
        var q11=(-1+2*sh12)*(-1+2*sh22); 
        var q12=4*ch2*ch3*(-1+2*sh12)*sh2*sh3+2*ch1*sh1*(-1+2*sh32); 
        var q13=4*ch1*ch3*sh1*sh3-2*ch2*(-1+2*sh12)*sh2*(-1+2*sh32);
    
        var q21=2*ch1*sh1*(1-2*sh22); 
        var q22=-8*ch1*ch2*ch3*sh1*sh2*sh3+(-1+2*sh12)*(-1+2*sh32); 
        var q23=-2*ch3*sh3+4*sh1*(ch3*sh1*sh3+ch1*ch2*sh2*(-1+2*sh32));
        
        var q31=2*ch2*sh2; 
        var q32=2*ch3*(1-2*sh22)*sh3; 
        var q33=(-1+2*sh22)*(-1+2*sh32);
        
        return [q11,q12,q13,q21,q22,q23,q31,q32,q33,r11,r12,r13,r21,r22,r23,r31,r32,r33];
    }
    
    function svd(a11, a12, a13,
    		 a21, a22, a23,
    		 a31, a32, a33)
    {
        var ata_return_val;
        var jacobi_return_val;
        var quat2mat_return_val;
        var b_return_vals;
        var ssv_return_vals;
        
    	// normal equations matrix
    	var ATA11, ATA12, ATA13;
    	var ATA21, ATA22, ATA23;
    	var ATA31, ATA32, ATA33;
        // host.console.println("Multiplying Matrices ATA ");
    	ata_return_val = multAtB(a11,a12,a13,a21,a22,a23,a31,a32,a33,
                                 a11,a12,a13,a21,a22,a23,a31,a32,a33);
        // host.console.println("result = " + ata_return_val);
        ATA11=ata_return_val[0];
        ATA12=ata_return_val[1];
        ATA13=ata_return_val[2];
        ATA21=ata_return_val[3];
        ATA22=ata_return_val[4];
        ATA23=ata_return_val[5];
        ATA31=ata_return_val[6];
        ATA32=ata_return_val[7];
        ATA33=ata_return_val[8];
    
    	// symmetric eigenalysis
        // host.console.println("Jacobi Eigenanalysis");
        jacobi_return_val = jacobiEigenanlysis( ATA11,ATA21,ATA22, ATA31,ATA32,ATA33);
        // host.console.println("result = " + jacobi_return_val);
        ATA11 = jacobi_return_val[0];
        ATA21 = jacobi_return_val[1];
        ATA22 = jacobi_return_val[2];
        ATA31 = jacobi_return_val[3];
        ATA32 = jacobi_return_val[4];
        ATA33 = jacobi_return_val[5];
        var qV = jacobi_return_val[6];
        
        // host.console.println("Quat2Mat3");
    	quat2mat_return_val = quatToMat3(qV[0],qV[1],qV[2],qV[3]);
        // host.console.println("result = " + jacobi_return_val);
        v11=quat2mat_return_val[0];
        v12=quat2mat_return_val[1];
        v13=quat2mat_return_val[2];
        v21=quat2mat_return_val[3];
        v22=quat2mat_return_val[4];
        v23=quat2mat_return_val[5];
        v31=quat2mat_return_val[6];
        v32=quat2mat_return_val[7];
        v33=quat2mat_return_val[8];
        
    	var b11, b12, b13;
    	var b21, b22, b23;
    	var b31, b32, b33;
    	b_return_vals = multAB(a11,a12,a13,a21,a22,a23,a31,a32,a33,
    		v11,v12,v13,v21,v22,v23,v31,v32,v33);
        b11=b_return_vals[0]; 
        b12=b_return_vals[1]; 
        b13=b_return_vals[2]; 
        b21=b_return_vals[3]; 
        b22=b_return_vals[4]; 
        b23=b_return_vals[5]; 
        b31=b_return_vals[6]; 
        b32=b_return_vals[7]; 
        b33=b_return_vals[8];
    
    	// sort singular values and find V
    	ssv_return_vals = sortSingularValues(b11, b12, b13, b21, b22, b23, b31, b32, b33,
                                             v11,v12,v13,v21,v22,v23,v31,v32,v33);
        b11=ssv_return_vals[0];
        b12=ssv_return_vals[1];
        b13=ssv_return_vals[2];
        b21=ssv_return_vals[3];
        b22=ssv_return_vals[4];
        b23=ssv_return_vals[5];
        b31=ssv_return_vals[6];
        b32=ssv_return_vals[7];
        b33=ssv_return_vals[8];
        v11=ssv_return_vals[9];
        v12=ssv_return_vals[10];
        v13=ssv_return_vals[11];
        v21=ssv_return_vals[12];
        v22=ssv_return_vals[13];
        v23=ssv_return_vals[14];
        v31=ssv_return_vals[15];
        v32=ssv_return_vals[16];
        v33=ssv_return_vals[17];
        
    	// QR decomposition
    	qr_return_vals = QRDecomposition(b11, b12, b13, b21, b22, b23, b31, b32, b33);
        u11=qr_return_vals[0];
        u12=qr_return_vals[1];
        u13=qr_return_vals[2];
        u21=qr_return_vals[3];
        u22=qr_return_vals[4];
        u23=qr_return_vals[5];
        u31=qr_return_vals[6];
        u32=qr_return_vals[7];
        u33=qr_return_vals[8];
        s11=qr_return_vals[9];
        s12=qr_return_vals[10];
        s13=qr_return_vals[11];
        s21=qr_return_vals[12];
        s22=qr_return_vals[13];
        s23=qr_return_vals[14];
        s31=qr_return_vals[15];
        s32=qr_return_vals[16];
        s33=qr_return_vals[17];
        
        return [u11, u12, u13,
                u21, u22, u23,
                u31, u32, u33,
                s11, s12, s13,
                s21, s22, s23,
                s31, s32, s33,
                v11, v12, v13,
                v21, v22, v23,
                v31, v32, v33];
    }
    
    function mat_inv(n00,n01,n02,n10,n11,n12,n20,n21,n22)
    {
        var v1 = (n11 * n22 - n21 * n12);
        var v2 = (n10 * n22 - n12 * n20);
        var v3 = (n10 * n21 - n11 * n20);
        
        var det = (n00 * v1) - (n01 * v2) + (n02 * v3);
        
        var invdet = 1/det;
        
        var ninv00 = v1 * invdet;
        var ninv01 = (n02 * n21 - n01 * n22) * invdet;
        var ninv02 = (n01 * n12 - n02 * n11) * invdet;
        var ninv10 = -v2 * invdet;
        var ninv11 = (n00 * n22 - n02 * n20) * invdet;
        var ninv12 = (n10 * n02 - n00 * n12) * invdet;
        var ninv20 = v3 * invdet;
        var ninv21 = (n20 * n01 - n00 * n21) * invdet;
        var ninv22 = (n00 * n11 - n10 * n01) * invdet;
        
        return [ninv00,ninv01,ninv02,ninv10,ninv11,ninv12,ninv20,ninv21,ninv22]
    }
    
    function AmultAT3x4(a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34)
    {
        var AAT11 = a11*a11 + a12*a12 + a13*a13 + a14*a14;
        var AAT12 = a11*a21 + a12*a22 + a13*a23 + a14*a24;
        var AAT13 = a11*a31 + a12*a32 + a13*a33 + a14*a34;
        var AAT21 = a11*a21 + a12*a22 + a13*a23 + a14*a24;
        var AAT22 = a21*a21 + a22*a22 + a23*a23 + a24*a24;
        var AAT23 = a21*a31 + a22*a32 + a23*a33 + a24*a34;
        var AAT31 = a11*a31 + a12*a32 + a13*a33 + a14*a34;
        var AAT32 = a21*a31 + a22*a32 + a23*a33 + a24*a34;
        var AAT33 = a31*a31 + a32*a32 + a33*a33 + a34*a34;
        
        return [AAT11,AAT12,AAT13,AAT21,AAT22,AAT23,AAT31,AAT32,AAT33];
    }
    
    function matmul4x3x3(a11,a12,a13,a21,a22,a23,a31,a32,a33,a41,a42,a43,
                         b11,b12,b13,b21,b22,b23,b31,b32,b33)
    {
        var C11 = a11*b11 + a12*b21 + a13*b31;
        var C12 = a11*b12 + a12*b22 + a13*b32;
        var C13 = a11*b13 + a12*b23 + a13*b33;
        var C21 = a21*b11 + a22*b21 + a23*b31;
        var C22 = a21*b12 + a22*b22 + a23*b32;
        var C23 = a21*b13 + a22*b23 + a23*b33;
        var C31 = a31*b11 + a32*b21 + a33*b31;
        var C32 = a31*b12 + a32*b22 + a33*b32;
        var C33 = a31*b13 + a32*b23 + a33*b33;
        var C41 = a41*b11 + a42*b21 + a43*b31;
        var C42 = a41*b12 + a42*b22 + a43*b32;
        var C43 = a41*b13 + a42*b23 + a43*b33;
        
        return [C11,C12,C13,C21,C22,C23,C31,C32,C33,C41,C42,C43];
    }
    
    function matrix_axis_angle(a,b,c,d,e,f,g,h,i)
    {
        var axis = Vector3(h-f,c-g,d-b);
        var angle = Math.atan2(axis.length,(a+e+i-1));
        axis = axis.scale(1/axis.length);
        return [axis,angle]
    }
    ''')
    
            f.write('''
    // Go through element nodes
    for (i=0; i < face_nodes.length; i++){{
        model_node = scene.nodes.getByIndex(face_nodes[i]);
        geometry_node_ids = face_map[i];
        {debug:}host.console.println("Element Index "+i+" corresponds to model node "+model_node.name+" and geometry node ids "+geometry_node_ids);
        model_node.transform.setIdentity();
        p1 = node_positions[geometry_node_ids[0]];
        p2 = node_positions[geometry_node_ids[1]];
        p3 = node_positions[geometry_node_ids[2]];
        centroid = p1.add(p2.add(p3)).scale(.3333333333333333);
        {debug:}host.console.println("\\n  Initial Transformation:");
        {debug:}host.console.println(model_node.transform.toString());
        model_node.transform.translateInPlace(centroid.scale(-1));
        {debug:}host.console.println("\\n  Transform after removing centroid "+centroid+":");
        {debug:}host.console.println(model_node.transform.toString());
        p1 = p1.subtract(centroid)
        p2 = p2.subtract(centroid)
        p3 = p3.subtract(centroid)
        d1 = node_displacements[geometry_node_ids[0]][shape-1].scale(scale);
        d2 = node_displacements[geometry_node_ids[1]][shape-1].scale(scale);
        d3 = node_displacements[geometry_node_ids[2]][shape-1].scale(scale);
        pd1 = p1.add(d1);
        pd2 = p2.add(d2);
        pd3 = p3.add(d3);
        {debug:}host.console.println("Transforming Element: "+geometry_node_ids);
        {debug:}host.console.println("  p1: "+p1)
        {debug:}host.console.println("  p2: "+p2)
        {debug:}host.console.println("  p3: "+p3)
        {debug:}host.console.println("  pd1: "+pd1)
        {debug:}host.console.println("  pd2: "+pd2)
        {debug:}host.console.println("  pd3: "+pd3)
        var AAT = AmultAT3x4(p1.x,p1.y,p1.z,1,p2.x,p2.y,p2.z,1,p3.x,p3.y,p3.z,1);
        {debug:}host.console.println("  AAT: "+AAT)
        var pinvA = matmul4x3x3.apply(this,[p1.x,p2.x,p3.x,p1.y,p2.y,p3.y,p1.z,p2.z,p3.z,1,1,1].concat(mat_inv.apply(this,AAT)));
        {debug:}host.console.println("  pinvA: "+pinvA)
        var b = [pd1.x,pd1.y,pd1.z,pd2.x,pd2.y,pd2.z,pd3.x,pd3.y,pd3.z];
        {debug:}host.console.println("  b: "+b)
        var x = matmul4x3x3.apply(this,pinvA.concat(b));
        {debug:}host.console.println("  x: "+x)
        T = [x[0],x[3],x[6],x[1],x[4],x[7],x[2],x[5],x[8]];
        t = Vector3(x[9],x[10],x[11]);
        {debug:}host.console.println("  T: "+T)
        {debug:}host.console.println("  t: "+t)
        svd_res = svd.apply(this,T);
        U = svd_res.slice(0,9);
        S = svd_res.slice(9,18);
        V = svd_res.slice(18);
        {debug:}host.console.println("  U: "+U);
        {debug:}host.console.println("  S: "+S);
        {debug:}host.console.println("  V: "+V);
        uvar = matrix_axis_angle.apply(this,U);
        uaxis = uvar[0]; uangle = uvar[1];
        vvar = matrix_axis_angle.apply(this,V);
        vaxis = vvar[0]; vangle = vvar[1];
        {debug:}host.console.println("  U axis: "+uaxis);
        {debug:}host.console.println("  U angle: "+uangle);
        {debug:}host.console.println("  V axis: "+vaxis);
        {debug:}host.console.println("  V angle: "+vangle);
        model_node.transform.rotateAboutVectorInPlace(-vangle,vaxis);
        {debug:}host.console.println("\\n  Transform after rotation about "+vaxis+" by "+(-vangle)+":");
        {debug:}host.console.println(model_node.transform.toString());
        model_node.transform.scaleInPlace(S[0],S[4],1);
        {debug:}host.console.println("\\n  Transform after scaling by "+[S[0],S[4],1]+":");
        {debug:}host.console.println(model_node.transform.toString());
        model_node.transform.rotateAboutVectorInPlace(uangle,uaxis);
        {debug:}host.console.println("\\n  Transform after rotation about "+uaxis+" by "+(-uangle)+":");
        {debug:}host.console.println(model_node.transform.toString());
        model_node.transform.translateInPlace(t.add(centroid));
        {debug:}host.console.println("\\n  Transform after translation by t and centroid "+t.add(centroid)+":");
        {debug:}host.console.println(model_node.transform.toString());
        
    }}
    '''.format(debug='' if debug_js is True or debug_js=='face' else '// '))

# Note: there should be some way to build the transformation matrix without an
# SVD.  However, it results in one of the dimensions of the transformation
# getting squashed to zero, which doesn't have any affect on the resulting
# geometry, but it does mess with the shading due to the surface normal being
# kind of screwed up.  I'm not sure exactly the solution apart from computing an
# SVD and making all singular values positive (which is exactly what I do now)
# function matmul4x4x4(
#     a00, a01, a02, a03,
#     a10, a11, a12, a13,
#     a20, a21, a22, a23,
#     a30, a31, a32, a33,
#     b00, b01, b02, b03,
#     b10, b11, b12, b13,
#     b20, b21, b22, b23,
#     b30, b31, b32, b33
# ) {
#     const result = [
#             a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30,
#             a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31,
#             a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32,
#             a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33,
#             a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30,
#             a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31,
#             a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32,
#             a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33,
#             a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30,
#             a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31,
#             a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32,
#             a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33,
#             a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30,
#             a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31,
#             a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32,
#             a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33,
#     ];

#     return result;
# }
# // Try the same thing by just building the transformation without SVDs
# T_raw = [1,0,0,0, //indices 0-3
#          0,1,0,0, //indices 4-7
#          0,0,1,0, //indices 8-11
#          0,0,0,1]; //indices 12-15
# {debug:}host.console.println("\\n  Initial Transformation:");
# {debug:}host.console.println(T_raw);
# T_raw[12] -= centroid.x;
# T_raw[13] -= centroid.y;
# T_raw[14] -= centroid.z;
# {debug:}host.console.println("\\n  With Removal of Centroid:");
# {debug:}host.console.println(T_raw);
# X = [x[0],x[1],x[2],0,
#      x[3],x[4],x[5],0,
#      x[6],x[7],x[8],0,
#      x[9],x[10],x[11],1];
# T_raw = matmul4x4x4.apply(this,T_raw.concat(X));
# {debug:}host.console.println("\\n  After Rotation, Scaling, and Translation:");
# {debug:}host.console.println(T_raw);
# T_raw[12] += centroid.x;
# T_raw[13] += centroid.y;
# T_raw[14] += centroid.z;
# {debug:}host.console.println("\\n  After Addition of Centroid:");
# {debug:}host.console.println(T_raw);
# model_node.transform.set(T_raw);
# break;

            f.write('''
    scene.update();
    
    
    timeEvHnd=new TimeEventHandler();
    timeEvHnd.onEvent=function(event) {
        var dtheta=speed*omega0*event.deltaTime;
        if (dtheta!=0){
    ''')
    
            f.write('''
            // Translate Nodes
            for (i=0; i < point_nodes.length; i++){{
                geometry_node_id = point_map[i];
                model_node = scene.nodes.getByIndex(point_nodes[i]);
                model_node.transform.setIdentity();
                model_node.transform.translateInPlace(node_displacements[geometry_node_id][shape-1].scale(scale*Math.cos(theta)));
            }}'''.format())
            f.write('''
            // Translate and Rotate Lines
            for (i=0; i < line_nodes.length; i++){{
                model_node = scene.nodes.getByIndex(line_nodes[i]);
                geometry_node_ids = line_map[i];
                model_node.transform.setIdentity();
                p1 = node_positions[geometry_node_ids[0]];
                p2 = node_positions[geometry_node_ids[1]];
                d1 = node_displacements[geometry_node_ids[0]][shape-1].scale(scale*Math.cos(theta));
                d2 = node_displacements[geometry_node_ids[1]][shape-1].scale(scale*Math.cos(theta));
                v1 = p2.add(p1.scale(-1));
                v2 = p2.add(d2).add(p1.add(d1).scale(-1));
                vc1 = p2.add(p1).scale(0.5);
                vc2 = p2.add(d2).add(p1).add(d1).scale(0.5);
                axis = v1.cross(v2);
                angle = Math.acos(v1.dot(v2)/(Math.sqrt(v1.dot(v1))*Math.sqrt(v2.dot(v2))));
                c = Math.sqrt(v2.dot(v2)/v1.dot(v1));
                t = vc2.add(vc1.scale(-c));
                //model_node.transform.rotateAboutLineInPlace(angle,vc1,vc1.add(axis)); //There seems to be a bug with this command...
                if (angle > 0.00175) {{
                    model_node.transform.translateInPlace(vc1.scale(-1));
                    model_node.transform.rotateAboutVectorInPlace(angle,axis);
                    model_node.transform.translateInPlace(vc1);
                }}
                model_node.transform.scaleInPlace(c,c,c);
                model_node.transform.translateInPlace(t);
            }}'''.format())
            f.write('''
            // Go through element nodes
            for (i=0; i < face_nodes.length; i++){{
                model_node = scene.nodes.getByIndex(face_nodes[i]);
                geometry_node_ids = face_map[i];
                model_node.transform.setIdentity();
                p1 = node_positions[geometry_node_ids[0]];
                p2 = node_positions[geometry_node_ids[1]];
                p3 = node_positions[geometry_node_ids[2]];
                centroid = p1.add(p2.add(p3)).scale(.3333333333333333);
                model_node.transform.translateInPlace(centroid.scale(-1));
                p1 = p1.subtract(centroid)
                p2 = p2.subtract(centroid)
                p3 = p3.subtract(centroid)
                d1 = node_displacements[geometry_node_ids[0]][shape-1].scale(scale*Math.cos(theta));
                d2 = node_displacements[geometry_node_ids[1]][shape-1].scale(scale*Math.cos(theta));
                d3 = node_displacements[geometry_node_ids[2]][shape-1].scale(scale*Math.cos(theta));
                pd1 = p1.add(d1);
                pd2 = p2.add(d2);
                pd3 = p3.add(d3);
                var AAT = AmultAT3x4(p1.x,p1.y,p1.z,1,p2.x,p2.y,p2.z,1,p3.x,p3.y,p3.z,1);
                var pinvA = matmul4x3x3.apply(this,[p1.x,p2.x,p3.x,p1.y,p2.y,p3.y,p1.z,p2.z,p3.z,1,1,1].concat(mat_inv.apply(this,AAT)));
                var b = [pd1.x,pd1.y,pd1.z,pd2.x,pd2.y,pd2.z,pd3.x,pd3.y,pd3.z];
                var x = matmul4x3x3.apply(this,pinvA.concat(b));
                T = [x[0],x[3],x[6],x[1],x[4],x[7],x[2],x[5],x[8]];
                t = Vector3(x[9],x[10],x[11]);
                svd_res = svd.apply(this,T);
                U = svd_res.slice(0,9);
                S = svd_res.slice(9,18);
                V = svd_res.slice(18);
                uvar = matrix_axis_angle.apply(this,U);
                uaxis = uvar[0]; uangle = uvar[1];
                vvar = matrix_axis_angle.apply(this,V);
                vaxis = vvar[0]; vangle = vvar[1];
                model_node.transform.rotateAboutVectorInPlace(-vangle,vaxis);
                model_node.transform.scaleInPlace(S[0],S[4],1);
                model_node.transform.rotateAboutVectorInPlace(uangle,uaxis);
                model_node.transform.translateInPlace(t.add(centroid));
                
            }}
            '''.format())

            f.write(
    '''        
            theta+=dtheta+2*Math.PI;
            theta %= 2*Math.PI;
            scene.update();
    
        }
    }
    
    
    
    menuEvHnd = new MenuEventHandler();
    menuEvHnd.onEvent=function(event) {
        if(event.menuItemName === "inc_mode"){
            shape += 1;
            if (shape > num_shapes) {
                shape = num_shapes;
            }
            host.console.show();
            host.console.println("Now displaying Mode "+shape+" at "+mode_frequencies[shape-1]+" Hz in panel "+panel);
        }
        if(event.menuItemName === "dec_mode"){
            shape -= 1;
            if (shape < 1) {
                shape = 1;
            }
            host.console.show();
            host.console.println("Now displaying Mode "+shape+" at "+mode_frequencies[shape-1]+" Hz in panel "+panel);
        }
        if(event.menuItemName === "scale_up"){
            scale *= 1.25;
        }
        if(event.menuItemName === "scale_down"){
            scale *= 0.8;
        }
        if(event.menuItemName === "speed_up"){
            speed *= 1.25;
        }
        if(event.menuItemName === "speed_down"){
            speed *= 0.8;
        }
    }
    
    keyEvHnd = new KeyEventHandler();
    keyEvHnd.onEvent = function(event) {
        //host.console.show();
        //host.console.println("Key pressed: Code = "+event.characterCode);
        if (event.characterCode === 44) {
            shape -= 1;
            if (shape < 1) {
                shape = 1;
            }
            host.console.show();
            host.console.println("Now displaying Mode "+shape+" at "+mode_frequencies[shape-1]+" Hz in panel "+panel);
        }
        if (event.characterCode === 46) {
            shape += 1;
            if (shape > num_shapes) {
                shape = num_shapes;
            }
            host.console.show();
            host.console.println("Now displaying Mode "+shape+" at "+mode_frequencies[shape-1]+" Hz in panel "+panel);
        }
        if(event.characterCode === 43){
            scale *= 1.25;
        }
        if(event.characterCode === 45){
            scale *= 0.8;
        }
        if(event.characterCode === 102){
            speed *= 1.25;
        }
        if(event.characterCode === 115){
            speed *= 0.8;
        }
    }
    runtime.addEventHandler(timeEvHnd);
    runtime.addCustomMenuItem("inc_mode","Next Mode                  .","default",false);
    runtime.addCustomMenuItem("dec_mode","Previous Mode           ,","default",false);
    runtime.addCustomMenuItem("scale_up","Scale 1.25x                  +","default",false);
    runtime.addCustomMenuItem("scale_down","Scale 0.8x                    -","default",false);
    runtime.addCustomMenuItem("speed_up","Speed 1.25x                 f","default",false);
    runtime.addCustomMenuItem("speed_down","Speed 0.8x                   s","default",false);
    runtime.addEventHandler(menuEvHnd);
    runtime.addEventHandler(keyEvHnd);''')
    
    
def get_view_parameters_from_plotter(plotter):
    c = plotter.camera
    camera_position = np.array(c.position)
    focus = np.array(c.focal_point)
    view_angle = c.view_angle
    c2c = camera_position - focus # Center of orbit to camera
    c2c /= np.linalg.norm(c2c)
    view_axis = -c2c
    view_up = np.array(c.up) # This is the general up direction and may not be the actual up direction of the camera
    view_up -= (np.dot(view_axis,view_up)*view_axis)
    view_up /= np.linalg.norm(view_up)
    yaw = np.arctan2(view_axis[1],view_axis[0])
    pitch = -np.arcsin(view_axis[2])
    roll = np.arcsin(view_up[0]*np.sin(yaw)-view_up[1]*np.cos(yaw))
    print('Yaw: {:}\nPitch: {:}\nRoll: {:}'.format(yaw*180/np.pi,
                                                   pitch*180/np.pi,
                                                   roll*180/np.pi))
    bounding_box = np.array(plotter.bounds).reshape(3,2)
    bounding_distance = np.linalg.norm(bounding_box[:,0]-bounding_box[:,1])
    media9 = {}
    media9['3Dortho'] = str(1/bounding_distance)
    media9['3Droll'] = str(roll*180/np.pi)
    media9['3Dc2c'] = ' '.join([str(v) for v in c2c])
    media9['3Dcoo'] = ' '.join([str(v) for v in focus]) # Center of orbit of virtual camera
    media9['3Droo'] = str(np.linalg.norm(camera_position - focus))
    media9['3Daac'] = str(view_angle)
    return media9
    