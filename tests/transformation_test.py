"""
Created Wed May 28 2025

@author: spcarte
"""
import numpy as np
import sdynpy as sdpy
from scipy.signal import chirp
import pytest

@pytest.fixture
def beam():
    beam_length = 1 # m
    box_cross_section_size = 0.05 # m
    number_nodes = 4

    youngs_modulus = 69.8e9 # pa
    density = 2700 # kg/m^3
    poissons_ratio = 0.33

    beam_k, beam_m = sdpy.beam.beamkm_2d(beam_length, box_cross_section_size, box_cross_section_size, 
                                        number_nodes, youngs_modulus, density, poissons_ratio)
    return beam_k, beam_m

@pytest.fixture
def external_node():
    external_node_m = np.zeros((3, 3), dtype=float)
    external_node_m[0,0] = 0.01 # kg
    external_node_m[1,1] = 0.01 # kg
    external_node_m[2,2] = 0.01 # kg*m^2

    external_node_k = np.zeros((3, 3), dtype=float)
    external_node_k[0,0] = 10e6 # kg
    external_node_k[1,1] = 10e6 # kg
    external_node_k[2,2] = 10e3 # N*m/rad
    return external_node_k, external_node_m

@pytest.fixture
def full_system(beam, external_node):
    beam_k, beam_m = beam
    external_node_k, external_node_m = external_node

    physical_m = np.zeros((beam_m.shape[0]+external_node_m.shape[0], beam_m.shape[1]+external_node_m.shape[1]), dtype=float)
    physical_m[:3, :3] = external_node_m
    physical_m[3:, 3:] = beam_m

    physical_k = np.zeros((beam_k.shape[0]+3, beam_k.shape[1]+3), dtype=float)
    physical_k[:3, :3] = external_node_k
    physical_k[3:, 3:] = beam_k
    
    transformation_array = np.array([[1, 0, 0, 1,   0,   0, 1,   0,     0, 1,    0,     0, 1,    0,   0], 
                                     [0, 1, 0, 0,   1,   0, 0,   1,     0, 0,    1,     0, 0,    1,   0], 
                                     [0, 0, 1, 0.1, 0.5, 0, 0.1, 0.165, 0, 0.1, -0.165, 0, 0.1, -0.5, 0]])

    transformed_m = transformation_array@physical_m@transformation_array.T
    transformed_k = transformation_array@physical_k@transformation_array.T
    transformed_c = 0.0001*transformed_m + 0.00001*transformed_k

    system_coordinate = sdpy.coordinate_array(node=np.array([101,201,202,203,204])[...,np.newaxis], direction=[1,3,5]).flatten()
    return sdpy.System(system_coordinate, transformed_m, transformed_k, transformed_c, transformation=transformation_array.T)

@pytest.fixture
def full_system_geometry():
    node_locations = np.array([[ 0,     0, 0],
                               [-0.5,   0, 0.1],
                               [-0.165, 0, 0.1],
                               [ 0.165, 0, 0.1],
                               [ 0.5,   0, 0.1]])

    node_array = sdpy.node_array([101,201,202,203,204], node_locations)

    traceline_array = sdpy.traceline_array(id=1, connectivity=[201,202,203,204,0,101,201,0,101,202,0,101,203,0,101,204])

    return sdpy.Geometry(node_array, sdpy.coordinate_system_array(id=1), traceline_array)

@pytest.fixture
def full_system_time_response(full_system):
    time = np.arange(5001)/5000

    external_node_coordinate = sdpy.coordinate_array(node=101, direction=[1,3,5]).flatten()

    chirp_excitation_ordinate = np.array([chirp(time, 20, 1, 500), chirp(time, 20, 1, 500), chirp(time, 20, 1, 500)])

    chirp_excitation = sdpy.data_array(sdpy.data.FunctionTypes.TIME_RESPONSE, time, chirp_excitation_ordinate, external_node_coordinate[...,np.newaxis])
    time_response = full_system.time_integrate(chirp_excitation,{2:full_system.coordinate})

    return time_response


@pytest.fixture
def response_transformation(full_system_geometry):
    beam_transform_coordinate = sdpy.coordinate_array(node=np.array([201,202,203,204])[...,np.newaxis], direction=[1,3]).flatten()
    return full_system_geometry.response_kinematic_transformation(response_coordinate = beam_transform_coordinate, 
                                                                  virtual_point_node_number = 101,
                                                                  virtual_point_location = [0,0,0])

@pytest.fixture
def reference_transformation(full_system_geometry):
    beam_transform_coordinate = sdpy.coordinate_array(node=np.array([201,202,203,204])[...,np.newaxis], direction=[1,3]).flatten()
    return full_system_geometry.force_kinematic_transformation(force_coordinate = beam_transform_coordinate, 
                                                               virtual_point_node_number = 101,
                                                               virtual_point_location = [0,0,0])

@pytest.fixture
def frequency_response_function(full_system):
    frequency = np.arange(601)
    return full_system.frequency_response(frequency)

def test_time_transformation(full_system_time_response, response_transformation):
    transformed_time_response = full_system_time_response[response_transformation.column_coordinate[...,np.newaxis]].apply_transformation(response_transformation)
    external_node_coordinate = sdpy.coordinate_array(node=101, direction=[1,3,5]).flatten()
    assert np.allclose(transformed_time_response[external_node_coordinate[...,np.newaxis]].ordinate, full_system_time_response[external_node_coordinate[...,np.newaxis]].ordinate)

def test_spectrum_transformation(full_system_time_response, response_transformation):
    external_node_coordinate = sdpy.coordinate_array(node=101, direction=[1,3,5]).flatten()
    spectrum_response = full_system_time_response.fft()
    transformed_spectrum_response = spectrum_response[response_transformation.column_coordinate[...,np.newaxis]].apply_transformation(response_transformation)
    assert np.allclose(transformed_spectrum_response[external_node_coordinate[...,np.newaxis]].ordinate, spectrum_response[external_node_coordinate[...,np.newaxis]].ordinate)

def test_cpsd_transformation(full_system_time_response, response_transformation):
    external_node_coordinate = sdpy.coordinate_array(node=101, direction=[1,3,5]).flatten()
    cpsd_response = full_system_time_response.cpsd(samples_per_frame=full_system_time_response.abscissa.shape[-1], overlap=0, window='boxcar')

    beam_cpsd_coordinate = sdpy.coordinate.outer_product(response_transformation.column_coordinate, response_transformation.column_coordinate)
    transformed_cpsd_response = cpsd_response[beam_cpsd_coordinate].apply_transformation(response_transformation)

    external_node_cpsd_coordinate = sdpy.coordinate.outer_product(external_node_coordinate, external_node_coordinate)
    assert np.allclose(transformed_cpsd_response[external_node_cpsd_coordinate].ordinate, cpsd_response[external_node_cpsd_coordinate].ordinate)

def test_frf_transformation(frequency_response_function, response_transformation, reference_transformation):
    external_node_coordinate = sdpy.coordinate_array(node=101, direction=[1,3,5]).flatten()
    external_node_frf_coordinate = sdpy.coordinate.outer_product(external_node_coordinate, external_node_coordinate)
    external_node_frf = frequency_response_function[external_node_frf_coordinate]

    transform_frf_coordinate = sdpy.coordinate.outer_product(response_transformation.column_coordinate, reference_transformation.column_coordinate)
    beam_frf = frequency_response_function[transform_frf_coordinate]
    transformed_frf = beam_frf.apply_transformation(response_transformation = response_transformation, 
                                                reference_transformation = reference_transformation)[external_node_frf_coordinate]
    assert np.allclose(external_node_frf.ordinate.diagonal(axis1=0, axis2=1), transformed_frf.ordinate.diagonal(axis1=0, axis2=1))

