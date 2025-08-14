# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:47:58 2025

@author: dprohe
"""

from ..core.sdynpy_geometry import (Geometry,NodeArray,CoordinateSystemArray,
                                    coordinate_system_array,node_array,
                                    global_coord,_element_types,_exodus_elem_type_map)
from ..core.sdynpy_shape import ShapeArray,shape_array
from ..core.sdynpy_data import data_array,FunctionTypes,NDDataArray
from ..core.sdynpy_colors import color_list
from ..core.sdynpy_coordinate import coordinate_array

from qtpy.QtWidgets import (QMainWindow)

import numpy as np

try:
    import escdf
except ImportError:
    escdf = None
    
def to_geometry(geometry_dataset):
    if not geometry_dataset.istype('geometry'):
        raise ValueError('geometry_dataset must have be an ESCDF `geometry` dataset.')
    element_type_string = None
    nodes = []
    css = [coordinate_system_array()]
    for id,position,x,y,z in zip(
            geometry_dataset.node_id,
            geometry_dataset.node_position,
            geometry_dataset.node_x_direction,
            geometry_dataset.node_y_direction,
            geometry_dataset.node_z_direction):
        nodes.append(node_array(id,position,def_cs=1,disp_cs=id+1,color=0))
        css.append(coordinate_system_array(id+1,'Node {:} Disp CS'.format(id),
                                           matrix=np.array((x,y,z,np.zeros(3)))))
    nodes = np.array(nodes).view(NodeArray)
    css = np.array(css).view(CoordinateSystemArray)

    geometry = Geometry(nodes,css)
    try:
        for i,connection in enumerate(geometry_dataset.line_connection):
            try:
                color = geometry_dataset.line_color[i,:]
                color_index = np.argmin(np.linalg.norm(color/255 - color_list,axis=-1))
            except TypeError:
                color_index = 0
            geometry.add_traceline(connection,color=color_index)
    except TypeError:
        # No tracelines
        pass
    
    try:
        for i,connection in enumerate(geometry_dataset.element_connection):
            try:
                color = geometry_dataset.element_color[i,:]
                color_index = np.argmin(np.linalg.norm(color/255 - color_list,axis=-1))
            except TypeError:
                color_index = 0
            type_name = geometry_dataset.element_type[i][()]
            try:
                element_type = _exodus_elem_type_map[type_name.lower()]
            except KeyError:
                if element_type_string is None:
                    element_type_string = {val.lower():key for key,val in _element_types.items()}
                try:
                    element_type = element_type_string[type_name.lower()]
                except KeyError:
                    print('Unknown Element Type at index {:}, {:}.  Skipping...'.format(i,type_name))
                    continue
            geometry.add_element(element_type, connection, color=color_index)
    except TypeError:
        # No elements
        pass
    return geometry

def from_geometry(dataset_name, geometry : Geometry, descriptive_name = ''):
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    esgeo = escdf.classes.geometry(dataset_name,descriptive_name = descriptive_name)
    esgeo.node_id = geometry.node.id
    esgeo.node_position = geometry.global_node_coordinate(geometry.node.id)
    esgeo.node_x_direction = geometry.global_deflection(
        coordinate_array(geometry.node.id,'X+'))
    esgeo.node_y_direction = geometry.global_deflection(
        coordinate_array(geometry.node.id,'Y+'))
    esgeo.node_z_direction = geometry.global_deflection(
        coordinate_array(geometry.node.id,'Z+'))
    if geometry.traceline.size > 0:
        esgeo.line_color = [np.array(color_list[i])*255 for i in geometry.traceline.color]
        esgeo.line_connection = geometry.traceline.connectivity
    if geometry.element.size > 0:
        esgeo.element_connection = geometry.element.connectivity
        esgeo.element_color = [np.array(color_list[i])*255 for i in geometry.element.color]
        esgeo.element_type = [_element_types[t] for t in geometry.element.type]
    return esgeo
    
def to_shape(shape_dataset):
    if not shape_dataset.istype('mode'):
        raise ValueError('shape_dataset must be an ESCDF `mode` dataset.')
    coordinate = coordinate_array(string_array=shape_dataset.dof_name[...])
    shape_matrix = shape_dataset.shape[...].T
    frequency = shape_dataset.frequency[...]
    damping = shape_dataset.damping_ratio[...]
    try:
        modal_mass = shape_dataset.modal_mass[...]
    except TypeError:
        modal_mass = 1
    try:
        description = shape_dataset.description[...]
    except TypeError:
        description = ''
    return shape_array(coordinate,shape_matrix,frequency,damping,modal_mass,
                       description)

def from_shape(dataset_name, shape : ShapeArray, descriptive_name = ''):
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    shape = shape.flatten()
    modes = escdf.classes.mode(dataset_name,descriptive_name = descriptive_name)
    modes.damping_ratio = shape.damping
    modes.frequency = shape.frequency
    dofs = np.unique(shape.coordinate)
    modes.shape = shape[dofs].T
    modes.dof_name = dofs.string_array()
    modes.modal_mass = shape.modal_mass
    if not np.all(shape.comment1 == ''):
        modes.description = shape.comment1.astype(object)
    return modes
    
datatype_enum_names = {
    FunctionTypes.TIME_RESPONSE:'time response',
    FunctionTypes.CROSSSPECTRUM:'power spectrum',
    FunctionTypes.FREQUENCY_RESPONSE_FUNCTION:'frequency response function',
    FunctionTypes.TRANSMISIBILITY:'transmissibility',
    FunctionTypes.COHERENCE:'coherence',
    FunctionTypes.CROSSCORRELATION:'correlation',
    FunctionTypes.POWER_SPECTRAL_DENSITY:'power spectral density',
    FunctionTypes.SPECTRUM:'spectrum',
    FunctionTypes.MODE_INDICATOR_FUNCTION:'mode indicator function',
    FunctionTypes.PARTIAL_COHERENCE:'partial coherence',
    FunctionTypes.SHOCK_RESPONSE_SPECTRUM:'shock response spectrum',
    FunctionTypes.IMPULSE_RESPONSE_FUNCTION:'impulse response function',
    FunctionTypes.MULTIPLE_COHERENCE:'multiple coherence'}
   
datatype_names = {
    FunctionTypes.GENERAL:
        ['general','none'],
    FunctionTypes.TIME_RESPONSE:
        ['time history','time response', 'th', 'time data', 'time',
         'time histories','time responses', 'ths', 'times',],
    FunctionTypes.AUTOSPECTRUM:
        ['autospectrum','as','autospectra'],
    FunctionTypes.CROSSSPECTRUM:
        ['crossspectrum','cs','crossspectra'],
    FunctionTypes.FREQUENCY_RESPONSE_FUNCTION:
        ['frf','frequency response','frequency response function',
         'frfs','frequency responses', 'frequency response functions',
         'transfer function','transfer functions', 'tf'],
    FunctionTypes.TRANSMISIBILITY:
        ['transmisibility','transmisibility function',
         'transmisibilities','transmisibility functions'],
    FunctionTypes.COHERENCE:['coherence','coh'],
    FunctionTypes.AUTOCORRELATION:
        ['corr','correlation','autocorrelation','corrs','correlations',
         'autocorrelations'],
    FunctionTypes.CROSSCORRELATION:
        ['ccorr','crosscorrelation''ccorrs',
         'crosscorrelations'],
    FunctionTypes.POWER_SPECTRAL_DENSITY:
        ['power spectral density','power spectral densities','psd','psds',
         'apsd','apsds','cpsd','cpsds','autopower spectral density',
         'autopower spectral densities','crosspower spectral density',
         'crosspower spectral densities'],
    FunctionTypes.ENERGY_SPECTRAL_DENSITY:
        ['esd','energy spectral density','energy spectral densities','esds'],
    FunctionTypes.PROBABILITY_DENSITY_FUNCTION:
        ['pdf','pdfs','probability density','probability densities',
         'probability density function','probability density functions'],
    FunctionTypes.SPECTRUM:
        ['fft','ffts','spectrum','spectra'],
    FunctionTypes.CUMULATIVE_FREQUENCY_DISTRIBUTION:
        ['cfd','cfds','cumulative frequency distribution',
         'cumulative frequency distributions'],
    FunctionTypes.PEAKS_VALLEY:
        ['peaks valley'],
    FunctionTypes.STRESS_PER_CYCLE:
        ['stress per cycle','stress/cycle'],
    FunctionTypes.STRAIN_PER_CYCLE:
        ['strain per cycle','strain/cycle'],
    FunctionTypes.ORBIT:['orbit','orbits'],
    FunctionTypes.MODE_INDICATOR_FUNCTION:
        ['mif','cmif','mmif','nmif','mode indicator function','mode indicator functions',
         'complex mode indicator function','complex mode indicator functions',
         'multimode indicator function','multimode indicator functions',
         'normal mode indicator function normal mode indicator functions'],
    FunctionTypes.FORCE_PATTERN:['force_pattern'],
    FunctionTypes.PARTIAL_POWER:['partial power'],
    FunctionTypes.PARTIAL_COHERENCE:['partial coherence','partial coh','pcoh'],
    FunctionTypes.EIGENVALUE:['eigenvalue','eigenvalues'],
    FunctionTypes.EIGENVECTOR:['eigenvector','eigenvectors'],
    FunctionTypes.SHOCK_RESPONSE_SPECTRUM:
        ['shock response spectrum','shock response spectra','srs','srss'],
    FunctionTypes.FINITE_IMPULSE_RESPONSE_FILTER:
        ['finite impulse response function','firf'],
    FunctionTypes.MULTIPLE_COHERENCE:
        ['mcoh','multiple coherence'],
    FunctionTypes.ORDER_FUNCTION:['order','order function','order functions'],
    FunctionTypes.PHASE_COMPENSATION:['phase comp','phase compensation'],
    FunctionTypes.IMPULSE_RESPONSE_FUNCTION:
        ['irf','irfs','impulse response function','impulse response functions',
         'impuse response','impulse responses']
    }
 
datatype_map = {}
for key,name_list in datatype_names.items():
    for name in name_list:
        datatype_map[name.replace(' ','').replace('_','').replace('-','').lower()] = key

def to_data(data_dataset):
    if not data_dataset.istype('data'):
        raise ValueError('data_dataset must be an ESCDF `data` dataset.')
    try:
        function_type = datatype_map[
            data_dataset.data_type[...][()].replace(' ','').replace('_','').replace('-','').lower()]
    except KeyError:
        raise ValueError('Unknown data type {:}'.format(data_dataset.data_type[...][()]))
    coordinate = coordinate_array(string_array=data_dataset.channel[...])
    ordinate_unit = np.char.add('Ordinate Unit: ', data_dataset.ordinate_unit[...])
    abscissa_unit = np.char.add(' -- Abscissa Unit: ', data_dataset.abscissa_unit[...])
    comment1 = np.char.add(ordinate_unit,abscissa_unit)
    try:
        abscissa = np.arange(data_dataset.ordinate.shape[-1])*data_dataset.abscissa_step[...] + data_dataset.abscissa_start[...]
    except TypeError:
        abscissa = data_dataset.abscissa[...]
    data = data_array(function_type, abscissa, data_dataset.ordinate[...],
                      coordinate, comment1)
    return data

def from_data(dataset_name, data : NDDataArray,descriptive_name=''):
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    data = data.flatten()
    try:
        abscissa_spacing = data.abscissa_spacing
    except ValueError:
        abscissa_spacing = None
    starting_abscissa = data.abscissa[...,0]
    if not np.all(starting_abscissa == starting_abscissa[0]):
        starting_abscissa = None
    else:
        starting_abscissa = starting_abscissa[0]
    dataset = escdf.classes.data(dataset_name,descriptive_name = descriptive_name)
    if starting_abscissa is None or abscissa_spacing is None:
        if np.all(data.abscissa == data.abscissa[0]):
            abscissa = data.abscissa[0]
        else:
            abscissa = data.abscissa
        dataset.abscissa = abscissa
    else:
        dataset.abscissa_step = abscissa_spacing
        dataset.abscissa_start = starting_abscissa
    dataset.ordinate = data.ordinate
    dataset.data_type = datatype_enum_names[data.function_type]
    dataset.channel = data.coordinate.string_array().astype(object)
    return dataset

class ESCDFVisualizer(QMainWindow):
    """An interactive window allowing users to explore an ESCDF file"""
    
    def __init__(self, escdf_file = None):
        """Create an ESCDF Visualizer Window to explore an ESCDF file.
        
        A filename or ESCDF object can be passed as an argment, or it can be
        loaded at a later point.
        
        Parameters
        ----------
        escdf_file : str or ESCDF, optional
            The file or ESCDF object to explore.  If not passed, one can be
            loaded through the graphical user interface.

        Returns
        -------
        None.

        """
        pass