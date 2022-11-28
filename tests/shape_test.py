# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:17:18 2022

@author: dprohe
"""

import sys
sys.path.insert(0,'./src')
import sdynpy as sdpy
import pytest
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np

@pytest.fixture
def real_shape_information():
    n_shapes = 30
    n_nodes = 20
    frequencies = np.arange(n_shapes)*0.1
    dampings = np.arange(n_shapes)*0.01
    coordinates = sdpy.coordinate.from_nodelist(np.arange(n_nodes)+1)
    shapes = np.arange(n_shapes*coordinates.size).reshape(n_shapes,coordinates.size)*1.0
    modal_masses = ((np.arange(n_shapes) % 2)+1)*0.25
    return {'frequency':frequencies,
            'damping':dampings,
            'coordinate':coordinates,
            'shape_matrix':shapes,
            'modal_mass':modal_masses}

def test_construct_shape(real_shape_information):
    shape = sdpy.shape_array(**real_shape_information)
    assert np.all(shape.frequency == real_shape_information['frequency'])
    assert np.all(shape.damping == real_shape_information['damping'])
    assert np.all(shape.modal_mass == real_shape_information['modal_mass'])
    assert np.all(shape.shape_matrix == real_shape_information['shape_matrix'])
    assert shape.is_complex() is False
    assert shape.ndof == real_shape_information['coordinate'].size

def test_shape_reduction(real_shape_information):
    shape = sdpy.shape_array(**real_shape_information)
    indices = [1,2,5,10]
    coordinates = real_shape_information['coordinate'][indices]
    shape_matrix_truth = real_shape_information['shape_matrix'][...,indices]
    reduced_shape = shape.reduce(coordinates)
    assert np.allclose(shape_matrix_truth, reduced_shape.shape_matrix)
    
def test_shape_coordinate_indexing(real_shape_information):
    shape = sdpy.shape_array(**real_shape_information).reshape(2,3,5)
    node_indices = np.array([2,4,3,5])
    coordinates = sdpy.coordinate.from_nodelist(node_indices+1,flatten=False)
    shape_matrix = shape[coordinates]
    negative_shape_matrix = shape[-coordinates]
    truth_matrix = real_shape_information['shape_matrix'].reshape(2,3,5,-1,3)
    truth_matrix = truth_matrix[...,node_indices,:]
    assert shape_matrix.shape == shape.shape+coordinates.shape
    assert np.allclose(shape_matrix, truth_matrix)
    assert np.allclose(negative_shape_matrix,-truth_matrix)