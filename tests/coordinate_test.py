# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 14:07:56 2022

@author: dprohe
"""

import sys
sys.path.insert(0,'./src')
import sdynpy as sdpy
import pytest
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np

# Test parsing coordinate strings
@pytest.mark.parametrize("string,node,direction",
                         [('102X+',102,1),
                          ('999999999Y-',999999999,-2),
                          ('2',2,0),
                          ('56RZ-',56,-6)])
def test_parse_coordinate_string(string,node,direction):
    assert sdpy.coordinate.parse_coordinate_string(string) == (node,direction)
    
def test_parse_coordinate_string_invalid_node():
    with pytest.raises(ValueError):
        sdpy.coordinate.parse_coordinate_string('X+')

def test_parse_coordinate_string_invalid_direction():
    with pytest.raises(KeyError):
        sdpy.coordinate.parse_coordinate_string('201Q')
    
def test_parse_coordinate_string_vectorized():
    string_array = [['101X+','101X-','101Z+'],
                    ['232101Y-','301RZ+','0']]
    output = sdpy.coordinate.parse_coordinate_string_array(string_array)
    assert np.all(output[0] == np.array([[   101,    101,    101],
                                         [232101,    301,      0]]))
    assert np.all(output[1] == np.array([[ 1, -1,  3],
                                         [-2,  6,  0]]))
    
# Test coordinate string generation

nodes = np.array((0,2,10,99,302,102938495,421,21256,23234324,5323,12,520,201,15)).reshape(7,2)
directions = np.array((1,2,3,4,5,6,-1,-2,-3,-4,-5,-6,0,-4)).reshape(7,2)
outputs = np.array(['0X+', '2Y+', '10Z+', '99RX+', '302RY+', '102938495RZ+', '421X-',
       '21256Y-', '23234324Z-', '5323RX-', '12RY-', '520RZ-', '201','15RX-']).reshape(7,2)

@pytest.mark.parametrize('node,direction,string',
                         [(node,direction,string) for node,direction,string
                          in zip(nodes.flatten(),
                                 directions.flatten(),
                                 outputs.flatten())])
def test_create_coordinate_string(node,direction,string):
    assert sdpy.coordinate.create_coordinate_string(node,direction) == string
    
def test_create_coordinate_string_invalid_direction():
    with pytest.raises(KeyError):
        sdpy.coordinate.create_coordinate_string(10, -7)

def test_create_coordinate_string_array():
    assert np.all(sdpy.coordinate.create_coordinate_string_array(
        nodes,
        directions) == np.array(outputs))
    
def test_coordinate_array_class_new():
    coord_array = sdpy.CoordinateArray((3,4,6))
    assert coord_array.shape == (3,4,6)
    assert isinstance(coord_array,sdpy.CoordinateArray)
    assert coord_array.node.dtype == np.dtype('uint64')
    assert coord_array.direction.dtype == np.dtype('int8')
    

def test_coordinate_array_function_node_direction():
    coord_array = sdpy.coordinate_array([[1,2],[3,4]],[[0,1],[-2,-5]])
    assert np.all(coord_array.node == np.array([[1,2],[3,4]]))
    assert np.all(coord_array.direction == np.array([[0,1],[-2,-5]]))

def test_coordinate_array_function_node_direction_broadcast():
    nodes = (np.arange(10)+1)[:,np.newaxis]
    directions = np.arange(3)+1
    coord_array = sdpy.coordinate_array(nodes,directions)
    broad_node,broad_dir = np.broadcast_arrays(nodes,directions)
    assert np.all(coord_array.node == broad_node)
    assert np.all(coord_array.direction == broad_dir)

def test_coordinate_array_function_structured_array():
    struct_array = np.array([(0,1),(1,2)],dtype=[('node',int),('direction',int)])
    coord_array = sdpy.coordinate_array(structured_array = struct_array)
    assert np.all(coord_array.node == struct_array['node'])
    assert np.all(coord_array.direction == struct_array['direction'])
    
def test_coordinate_array_function_string_array():
    coord_array = sdpy.coordinate_array(string_array=outputs)
    assert np.all(coord_array.node == nodes)
    assert np.all(coord_array.direction == directions)
    
@pytest.fixture
def equality_checks():
    nodes = np.array(([1,2],[3,4]))
    directions = np.array(((1,-1),(0,-5)))
    coord_array_1 = sdpy.coordinate_array(nodes,directions)
    coord_array_2 = sdpy.coordinate_array(nodes,-directions)
    coord_array_3 = sdpy.coordinate_array(nodes,abs(directions))
    coord_array_4 = sdpy.coordinate_array(nodes.copy(),directions.copy())
    return coord_array_1,coord_array_2,coord_array_3,coord_array_4

def test_coordinate_array_equality_checks(equality_checks):
    coord_array_1,coord_array_2,coord_array_3,coord_array_4 = equality_checks
    assert np.all(coord_array_1 == coord_array_4)
    assert np.any(coord_array_1 != coord_array_2)
    assert np.all(coord_array_1 == -coord_array_2)
    assert np.all(abs(coord_array_1) == coord_array_3)
    assert np.all(coord_array_1.abs() == coord_array_3)
    assert np.all(coord_array_1 == +coord_array_1)
    assert np.all(coord_array_1.sign() == np.array(((1,-1),(1,-1))))

def test_coordinate_array_local_direction():
    nodes = np.arange(1,3)*1000
    directions = np.arange(-6,7)
    coord_array = sdpy.coordinate_array(nodes[:,np.newaxis],directions)
    assert np.all(coord_array.local_direction() == np.array([[[ 0.,  0., -1.],
                                                              [ 0., -1.,  0.],
                                                              [-1.,  0.,  0.],
                                                              [ 0.,  0., -1.],
                                                              [ 0., -1.,  0.],
                                                              [-1.,  0.,  0.],
                                                              [ 0.,  0.,  0.],
                                                              [ 1.,  0.,  0.],
                                                              [ 0.,  1.,  0.],
                                                              [ 0.,  0.,  1.],
                                                              [ 1.,  0.,  0.],
                                                              [ 0.,  1.,  0.],
                                                              [ 0.,  0.,  1.]],
                                                             
                                                             [[ 0.,  0., -1.],
                                                              [ 0., -1.,  0.],    
                                                              [-1.,  0.,  0.],    
                                                              [ 0.,  0., -1.],
                                                              [ 0., -1.,  0.],
                                                              [-1.,  0.,  0.],
                                                              [ 0.,  0.,  0.],
                                                              [ 1.,  0.,  0.],
                                                              [ 0.,  1.,  0.],
                                                              [ 0.,  0.,  1.],
                                                              [ 1.,  0.,  0.],
                                                              [ 0.,  1.,  0.],
                                                              [ 0.,  0.,  1.]]]))
    assert np.all(coord_array[0,0].local_direction() == np.array([0.,0.,-1.]))
    assert np.all(coord_array[0,5].local_direction() == np.array([-1.,0.,0.]))
    
def test_coordinate_array_string_array():
    nodes = np.arange(13)
    directions = np.arange(13)-6
    coord_array = sdpy.coordinate_array(nodes,directions)
    assert np.all(coord_array.string_array() == ['0RZ-', '1RY-', '2RX-', '3Z-',
                                                 '4Y-', '5X-', '6', '7X+', '8Y+',
                                                 '9Z+', '10RX+', '11RY+', '12RZ+'])

def test_coordinate_array_str():
    nodes = np.arange(13)
    directions = np.arange(13)-6
    coord_array = sdpy.coordinate_array(nodes,directions)
    assert np.all(str(coord_array).replace('\n','') == 
                  "['0RZ-' '1RY-' '2RX-' '3Z-' '4Y-' '5X-' '6' '7X+' '8Y+' '9Z+' '10RX+' '11RY+' '12RZ+']")
    
def test_coordinate_array_repr():
    nodes = np.arange(13)
    directions = np.arange(13)-6
    coord_array = sdpy.coordinate_array(nodes,directions)
    assert np.all(repr(coord_array).replace('\n','').replace(' ','') == 
                  "coordinate_array(string_array=array(['0RZ-','1RY-','2RX-','3Z-','4Y-','5X-','6','7X+','8Y+','9Z+','10RX+','11RY+','12RZ+'],dtype='<U5'))")
    
def test_coordinate_equality():
    nodes = [1,200,3021]
    directions = [-1,0,4]
    coord_array = sdpy.coordinate_array(nodes,directions)
    assert np.all((coord_array == '1X-') == np.array([True,False,False]))
    assert np.all((coord_array == ['1X-','200','RX+3021']))
    assert np.all((coord_array == np.array([nodes,directions]).T))
    assert np.all((coord_array == sdpy.coordinate_array(nodes[1],directions[1]))==np.array([False,True,False]))
    assert np.all((coord_array != '1X-') == ~np.array([True,False,False]))
    assert np.all(~(coord_array != ['1X-','200','RX+3021']))
    assert np.all(~(coord_array != np.array([nodes,directions]).T))
    assert np.all((coord_array != sdpy.coordinate_array(nodes[1],directions[1]))== ~np.array([False,True,False]))
    
def test_matlab_cellstr():
    from scipy.io import loadmat
    matdict = loadmat(os.path.join(THIS_DIR,'test_data/cellstr_test.mat'))
    coord_array_0 = sdpy.coordinate.from_matlab_cellstr(matdict['cellstr'])
    coord_array_1 = sdpy.coordinate.from_matlab_cellstr(matdict['cellstr_t'])
    assert coord_array_0.shape == (1,5)
    assert coord_array_1.shape == (5,1)
    assert np.all(coord_array_0 == np.array(['104X-','4','302Y+','34RZ-','5X+'])[np.newaxis])
    assert np.all(coord_array_1 == np.array(['104X-','4','302Y+','34RZ-','5X+'])[:,np.newaxis])
    
def test_save_load():
    from numpy import load
    nodes = np.array(np.arange(10)+1).reshape(2,5)
    directions = np.array(np.arange(5))[np.newaxis,:]
    coordinate_array = sdpy.coordinate_array(nodes,directions)
    filename = os.path.join(THIS_DIR,'test_data/coordinate_save_load_test.npy')
    coordinate_array.save(filename)
    # Load it in to see if it looks like it should
    array = load(filename)
    assert np.all(array['node']==nodes)
    assert np.all(array['direction']==directions)
    coordinate_array_test = sdpy.coordinate.load(filename)
    assert np.all(coordinate_array == coordinate_array_test)
    
def test_raise_invalid_coordinate_array():
    with pytest.raises(ValueError):
        sdpy.coordinate_array(structured_array=
          np.zeros((3,3),dtype=[('nodes','uint'),('directions','int8')]))
    with pytest.raises(ValueError):
        sdpy.coordinate_array(structured_array=3)
        
def test_bad_broadcast():
    nodes = np.arange(13)
    directions = np.arange(6)
    with pytest.raises(ValueError):
        sdpy.coordinate_array(node=nodes,direction=directions)
    
def test_string_to_direction():
    nodes = [1,2,3,4]
    directions = ['X+','-Y','-RZ','']
    coord_array = sdpy.coordinate_array(node=nodes,direction=directions)
    assert np.all(coord_array.direction == np.array((1,-2,-6,0)))
    
def test_outer_product():
    coord_array_1 = sdpy.coordinate_array(node=(1,2,3),
                                          direction=(1,2,3))
    coord_array_2 = sdpy.coordinate_array(node=(3,2,1),
                                          direction=(3,2,1))
    assert np.all(sdpy.coordinate.outer_product(coord_array_1,coord_array_2) == sdpy.coordinate_array(string_array=
        np.array([[['1X+', '3Z+'],
                   ['1X+', '2Y+'],
                   ['1X+', '1X+']],
                  
                  [['2Y+', '3Z+'],
                   ['2Y+', '2Y+'],
                   ['2Y+', '1X+']],
                  
                  [['3Z+', '3Z+'],
                   ['3Z+', '2Y+'],
                   ['3Z+', '1X+']]], dtype='<U3')))