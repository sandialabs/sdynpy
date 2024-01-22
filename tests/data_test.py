# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 01:06:03 2023

@author: dprohe
"""

import sys
sys.path.insert(0,'./src')
import sdynpy as sdpy
import sdynpy.demo as demo
import pytest
import os
import numpy as np
from scipy.signal.windows import get_window
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def sine_time_history():
    abscissa = np.arange(100)*0.01
    return sdpy.data_array(sdpy.data.FunctionTypes.TIME_RESPONSE,
                           abscissa,
                           np.array([np.sin(2*np.pi*10*abscissa),
                            np.cos(2*np.pi*10*abscissa)]),
                           sdpy.coordinate_array(string_array=['101X+','102X+'])
                           )

@pytest.fixture
def flat_time_history():
    abscissa = np.arange(100)*0.01
    ordinate = np.ones((2,3,100))
    return sdpy.data_array(sdpy.data.FunctionTypes.TIME_RESPONSE,
                           abscissa,
                           ordinate,
                           sdpy.coordinate_array(np.array((100,200))[:,np.newaxis],[1,2,3])[...,np.newaxis]
                           )

@pytest.fixture
def plate_transfer_function():
    modes = demo.beam_plate.system.eigensolution()
    modes.damping = 0.01
    modes[:6].frequency = 0.5
    frequencies = np.arange(501)
    transfer_function = modes.compute_frf(
        frequencies,
        references = sdpy.coordinate_array(
            string_array=['1Z+','25Z+']),
        responses=sdpy.coordinate_array(demo.beam_plate.geometry.node.id,'Z+'))
    return transfer_function

@pytest.fixture
def impulse_response_medium():
    num_samples = 1000
    ordinate = np.zeros(num_samples)
    ordinate[0] = 1.0
    return sdpy.data_array(sdpy.data.FunctionTypes.TIME_RESPONSE,
                           np.arange(num_samples)/num_samples,
                           ordinate,
                           coordinate = sdpy.coordinate_array(string_array=['1X+']))

@pytest.fixture
def impulse_response_short():
    num_samples = 334
    sample_rate = 0.001
    ordinate = np.zeros(num_samples)
    ordinate[0] = 1.0
    return sdpy.data_array(sdpy.data.FunctionTypes.TIME_RESPONSE,
                           np.arange(num_samples)*sample_rate,
                           ordinate,
                           coordinate = sdpy.coordinate_array(string_array=['1X+']))

@pytest.fixture
def impulse_response_long():
    num_samples = 4096
    sample_rate = 0.001
    ordinate = np.zeros(num_samples)
    ordinate[0] = 1.0
    return sdpy.data_array(sdpy.data.FunctionTypes.TIME_RESPONSE,
                           np.arange(num_samples)*sample_rate,
                           ordinate,
                           coordinate = sdpy.coordinate_array(string_array=['1X+']))

def test_zero_pad_left(sine_time_history):
    num_pad = 11
    padded = sine_time_history.zero_pad(num_pad,left=True,right=False)
    # Check the left side is zeros
    assert np.all(padded.ordinate[...,:num_pad] == 0)
    # Check the right side is equal to what is was before
    assert np.all(padded.ordinate[...,num_pad:] == sine_time_history.ordinate)
    # Check that the abscissa are modified correctly
    assert np.all(padded.abscissa[...,num_pad:] == sine_time_history.abscissa)
    # Even the "negative abscissa"
    assert np.allclose(padded.abscissa_spacing,sine_time_history.abscissa_spacing)
    
def test_zero_pad_right(sine_time_history):
    num_pad = 21
    padded = sine_time_history.zero_pad(num_pad,left=False,right=True)
    # Check the right side is zeros
    assert np.all(padded.ordinate[...,-num_pad:] == 0)
    # Check the right side is equal to what is was before
    assert np.all(padded.ordinate[...,:-num_pad] == sine_time_history.ordinate)
    # Check that the abscissa are modified correctly
    assert np.all(padded.abscissa[...,:-num_pad] == sine_time_history.abscissa)
    # Even the "extra abscissa"
    assert np.allclose(padded.abscissa_spacing, sine_time_history.abscissa_spacing)
    
def test_zero_pad_both(sine_time_history):
    num_pad = 7
    padded = sine_time_history.zero_pad(num_pad*2,left=True,right=True)
    # Check the left side is zeros
    assert np.all(padded.ordinate[...,:num_pad] == 0)
    # Check the right side is zeros
    assert np.all(padded.ordinate[...,-num_pad:] == 0)
    # Check the middle side is equal to what is was before
    assert np.all(padded.ordinate[...,num_pad:-num_pad] == sine_time_history.ordinate)
    # Check that the abscissa are modified correctly
    assert np.all(padded.abscissa[...,num_pad:-num_pad] == sine_time_history.abscissa)
    # Even the "extra abscissa"
    assert np.allclose(padded.abscissa_spacing, sine_time_history.abscissa_spacing)
    
def test_zero_pad_no_abscissa_update(sine_time_history):
    num_pad = 13
    padded = sine_time_history.zero_pad(num_pad*2,update_abscissa=False,left=True,right=True)
    # Check the left side is zeros
    assert np.all(padded.ordinate[...,:num_pad] == 0)
    # Check the right side is zeros
    assert np.all(padded.ordinate[...,-num_pad:] == 0)
    # Check the left side is zeros
    assert np.all(padded.abscissa[...,:num_pad] == 0)
    # Check the right side is zeros
    assert np.all(padded.abscissa[...,-num_pad:] == 0)
    # Check the middle side is equal to what is was before
    assert np.all(padded.ordinate[...,num_pad:-num_pad] == sine_time_history.ordinate)
    # Check that the abscissa are modified correctly
    assert np.all(padded.abscissa[...,num_pad:-num_pad] == sine_time_history.abscissa)
    # The abscissa should no longer be equally spaced
    with pytest.raises(ValueError):
        padded.abscissa_spacing
        
def test_segment_frames(sine_time_history):
    frame_length = 0.2
    overlap = 0.25
    correct_frame_length = 20
    correct_number_of_frames = 6
    correct_stride = 15
    segmented_time_history = sine_time_history.split_into_frames(
        frame_length = frame_length,
        overlap = overlap)
    assert segmented_time_history.num_elements == correct_frame_length
    assert segmented_time_history.shape[1:] == sine_time_history.shape
    assert segmented_time_history.shape[0] == correct_number_of_frames
    # Check each frame contains the right information
    for i in range(correct_number_of_frames):
        indices = slice(i*correct_stride,i*correct_stride+correct_frame_length)
        assert np.allclose(sine_time_history.ordinate[...,indices], segmented_time_history[i].ordinate)

def test_partial_segment_frames(sine_time_history):
    samples_per_frame = 20
    overlap_samples = 13
    correct_number_of_frames = 17
    correct_stride = 7
    segmented_time_history = sine_time_history.split_into_frames(
        samples_per_frame = samples_per_frame,
        overlap_samples = overlap_samples,
        allow_fractional_frames=True)
    assert segmented_time_history.num_elements == samples_per_frame
    assert segmented_time_history.shape[1:] == sine_time_history.shape
    assert segmented_time_history.shape[0] == correct_number_of_frames
    # Check each frame contains the right information
    for i in range(correct_number_of_frames):
        start_index = overlap_samples - i*correct_stride
        if start_index < 0:
            start_index = 0
        check2 = segmented_time_history[i].ordinate[...,start_index:]
        start_index = -overlap_samples + i*correct_stride
        end_index = start_index + samples_per_frame
        if start_index < 0:
            start_index = 0
        check1 = sine_time_history.ordinate[...,start_index:end_index]
        check2 = check2[...,:check1.shape[-1]]
        assert np.allclose(check1, check2)

def test_window_segment_frames(flat_time_history):
    samples_per_frame = 20
    overlap_samples = 15
    window = 'hann'
    correct_number_of_frames = 17
    segmented_time_history = flat_time_history.split_into_frames(
        samples_per_frame = samples_per_frame,
        overlap_samples = overlap_samples,
        window=window,check_cola=True)
    assert segmented_time_history.num_elements == samples_per_frame
    assert segmented_time_history.shape[1:] == flat_time_history.shape
    assert segmented_time_history.shape[0] == correct_number_of_frames
    assert np.allclose(segmented_time_history.ordinate,get_window('hann',samples_per_frame))
    
def test_segment_frames_bad_arguments(flat_time_history):
    # No frame length specified
    with pytest.raises(ValueError):
        segmented_time_history = flat_time_history.split_into_frames()
    # Redundant frame length specified
    with pytest.raises(ValueError):
        segmented_time_history = flat_time_history.split_into_frames(
            frame_length = 10, samples_per_frame = 10)
    # Redundant overlap specified
    with pytest.raises(ValueError):
        segmented_time_history = flat_time_history.split_into_frames(
            frame_length = 10, overlap = 0.5, overlap_samples = 5)
    # Odd number of samples per frame specified
    with pytest.raises(ValueError):
        segmented_time_history = flat_time_history.split_into_frames(
            samples_per_frame = 11, overlap_samples = 5)
    # No overlap specified
    segmented_time_history = flat_time_history.split_into_frames(
        samples_per_frame = 10)
    assert segmented_time_history.shape[0] == 10
    # Bad window size specified
    with pytest.raises(ValueError):
        segmented_time_history = flat_time_history.split_into_frames(
            samples_per_frame = 10, overlap_samples = 5,
            window = np.arange(6))
    # COLA Check Failure
    with pytest.raises(ValueError):
        segmented_time_history = flat_time_history.split_into_frames(
            samples_per_frame = 10,
            overlap_samples = 3,
            window='hann',check_cola=True)

def test_frf_irf_compatibility(plate_transfer_function):
    frf = plate_transfer_function
    irf = plate_transfer_function.ifft()
    # Shape is preserved
    assert irf.shape == frf.shape
    # Coordinates are preserved
    assert np.all(irf.coordinate == frf.coordinate)
    # Number of elements is correct
    assert (frf.num_elements-1)*2 == irf.num_elements
    # Abscissa spacing is correct
    assert np.allclose(frf.abscissa_spacing, irf.abscissa_spacing*irf.num_elements)
    # Round-trip checks
    frf_check = irf.fft()
    assert frf.shape == frf_check.shape
    assert np.all(frf.coordinate == frf_check.coordinate)
    assert frf.num_elements == frf_check.num_elements
    assert np.allclose(frf.abscissa_spacing, frf_check.abscissa_spacing)
    # Make sure you get the same thing back when you do a round-trip calculation
    assert np.allclose(frf.ordinate[...,:-1],frf_check.ordinate[...,:-1])
    
def test_mimo_forward(plate_transfer_function,impulse_response_medium):
    frf = plate_transfer_function
    impulse = impulse_response_medium
    # Just get one frf
    frf = frf[:,0]
    impulse.coordinate = np.unique(frf.reference_coordinate)
    output = impulse.mimo_forward(frf)
    irf = frf.ifft().squeeze()
    assert np.allclose(output.ordinate.squeeze(),irf.ordinate.squeeze())

def test_interpolate(impulse_response_medium):
    test_fn = np.concatenate((impulse_response_medium[np.newaxis],
                              2*impulse_response_medium[np.newaxis]))
    abscissa_to_check = [i*test_fn.abscissa_spacing/2 for i in range(10)]
    interpolated_fn = test_fn.interpolate(abscissa_to_check)
    # Check that the first indices are the same
    assert np.allclose(interpolated_fn.ordinate[:,0], test_fn.ordinate[:,0])
    # Check that the second interpolated value is half the first
    assert np.allclose(interpolated_fn.ordinate[:,1], test_fn.ordinate[:,0]/2)
    # Check that the third is the same
    assert np.allclose(interpolated_fn.ordinate[:,2], test_fn.ordinate[:,1])
    # Now adjust the abscissa to see that it works with the for-loop approach
    test_fn.abscissa[1,:] -= 0.25*test_fn.abscissa_spacing
    interpolated_fn = test_fn.interpolate(abscissa_to_check)
    # Check that the first one is still the same
    assert np.allclose(interpolated_fn.ordinate[0,0], test_fn.ordinate[0,0])
    # Check that the second signal was interpolated differently
    assert np.allclose(interpolated_fn.ordinate[1,0], test_fn.ordinate[1,0]*3/4)
    