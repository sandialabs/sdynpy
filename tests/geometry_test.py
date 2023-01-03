# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:55:18 2022

@author: dprohe
"""

import sys
sys.path.insert(0,'./src')
import sdynpy as sdpy
import pytest
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np

# Create some geometry objects
@pytest.fixture
def cartesian_geometry():
    coordinates = np.linspace(-1,1,5)
    all_coords = np.array(np.meshgrid(coordinates,coordinates,coordinates,indexing='ij')).reshape(3,-1).T
    # Scale so they are different lengths
    all_coords *= np.array((1.1,1.0,0.9))
    node_ids = np.arange(all_coords.shape[0])+1
    rotation = sdpy.rotation.R(2,20,degrees=True)@sdpy.rotation.R(1,-30,degrees=True)@sdpy.rotation.R(0,-45,degrees=True)
    translation = np.array(((0.0,2.0,1.0),))
    rotmat = np.concatenate((rotation,translation),axis=0)
    definition_coordinate_system = sdpy.coordinate_system_array(2,matrix=rotmat)[np.newaxis]
    nodes = sdpy.node_array(node_ids,all_coords,def_cs=2,disp_cs=2)
    geometry = sdpy.Geometry(nodes,definition_coordinate_system)
    return geometry
    
@pytest.fixture
def cylindrical_geometry():
    all_coords = np.array(np.meshgrid(np.linspace(-1,1,5),np.linspace(0,360,10),np.linspace(-1,1,5),indexing='ij')).reshape(3,-1).T
    node_ids = np.arange(all_coords.shape[0])+1
    rotation = sdpy.rotation.R(2,20,degrees=True)@sdpy.rotation.R(1,-30,degrees=True)@sdpy.rotation.R(0,-45,degrees=True)
    translation = np.array(((0.0,2.0,1.0),))
    rotmat = np.concatenate((rotation,translation),axis=0)
    definition_coordinate_system = sdpy.coordinate_system_array(2,matrix=rotmat,cs_type=1)[np.newaxis]
    nodes = sdpy.node_array(node_ids,all_coords,def_cs=2,disp_cs=2)
    geometry = sdpy.Geometry(nodes,definition_coordinate_system)
    return geometry

@pytest.fixture
def spherical_geometry():
    all_coords = np.array(np.meshgrid(np.linspace(-1,1,5),np.linspace(0,180,10),np.linspace(0,360,20),indexing='ij')).reshape(3,-1).T
    node_ids = np.arange(all_coords.shape[0])+1
    rotation = sdpy.rotation.R(2,20,degrees=True)@sdpy.rotation.R(1,-30,degrees=True)@sdpy.rotation.R(0,-45,degrees=True)
    translation = np.array(((0.0,2.0,1.0),))
    rotmat = np.concatenate((rotation,translation),axis=0)
    definition_coordinate_system = sdpy.coordinate_system_array(2,matrix=rotmat,cs_type=2)[np.newaxis]
    nodes = sdpy.node_array(node_ids,all_coords,def_cs=2,disp_cs=2)
    geometry = sdpy.Geometry(nodes,definition_coordinate_system)
    return geometry

@pytest.mark.parametrize('geometry_fixture',
                         ['cartesian_geometry','cylindrical_geometry','spherical_geometry'])
def test_geometry_construction(geometry_fixture,request):
    geometry = request.getfixturevalue(geometry_fixture)
    straight_geometry = geometry.copy()
    straight_geometry.coordinate_system.matrix[...,:3,:3] = np.eye(3)
    straight_geometry.coordinate_system.matrix[...,3:,:3] = 0
    rotated_geometry = geometry.copy()
    rotated_geometry.coordinate_system.matrix[...,3:,:3] = 0
    assert isinstance(geometry,sdpy.Geometry)
    straight_coords = straight_geometry.global_node_coordinate()
    rotated_coords = rotated_geometry.global_node_coordinate()
    final_coords = geometry.global_node_coordinate()
    rotation = geometry.coordinate_system.matrix[0,:3,:3]
    translation = geometry.coordinate_system.matrix[0,3,:3]
    cs_type = geometry.coordinate_system.cs_type[0]
    assert np.allclose(straight_coords@rotation,rotated_coords)
    assert np.allclose(rotated_coords+translation,final_coords)
    if cs_type == 0: # Cartesian
        assert np.allclose(geometry.node.coordinate,straight_coords)
    if cs_type == 1: # Cylindrical
        check_coords = geometry.node.coordinate.copy()
        check_coords[:,0],check_coords[:,1] = (check_coords[:,0]*np.cos(check_coords[:,1]*np.pi/180),
        check_coords[:,0]*np.sin(check_coords[:,1]*np.pi/180))
        assert np.allclose(check_coords,straight_coords)
    if cs_type == 2: # Spherical
        check_coords = geometry.node.coordinate.copy()
        check_coords[:,0],check_coords[:,1],check_coords[:,2] = (
            check_coords[:,0]*np.sin(check_coords[:,1]*np.pi/180)*np.cos(check_coords[:,2]*np.pi/180),
            check_coords[:,0]*np.sin(check_coords[:,1]*np.pi/180)*np.sin(check_coords[:,2]*np.pi/180),
            check_coords[:,0]*np.cos(check_coords[:,1]*np.pi/180))
        assert np.allclose(check_coords,straight_coords)

def test_node_indexing(cartesian_geometry):
    indices = np.array([[0,10],
                        [40,20]])
    scalar_index = 5
    ids = cartesian_geometry.node.id[indices]
    scalar_id = cartesian_geometry.node.id[scalar_index]
    # Index via id number
    nodes = cartesian_geometry.node(ids)
    scalar_node = cartesian_geometry.node(scalar_id)
    check_nodes = cartesian_geometry.node[indices]
    check_scalar_node = cartesian_geometry.node[scalar_index]
    assert np.all(nodes == check_nodes)
    assert np.all(check_scalar_node == scalar_node)
    