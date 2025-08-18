# -*- coding: utf-8 -*-
"""
This module handles reading data from output files of T-shaker, which is a
Labview-based Vibration software in use at Sandia National Laboratories.

T-shaker writes TDMS files natively, but can also output .mat files as well.
There are .mat file formats for shaker shock, random vibration, and time
histories.

Copyright 2022 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
try:
    import nptdms as tdms
except ImportError:
    tdms = None

from ..core.sdynpy_coordinate import (coordinate_array, outer_product,
                                      CoordinateArray, _string_map)
from ..core.sdynpy_data import data_array, FunctionTypes
from glob import glob
import os
from scipy.io import loadmat


def read_tdms(file, coordinate_property="Ch_Location",
              coordinate_string_map=None, invalid_coordinate_value='999999',
              min_time=None, max_time=None, channel_indices=None):
    if tdms is None:
        raise ValueError('Must have npTDMS installed to import T-shaker files')
    # Read in the file if we've passed a string.
    if not isinstance(file, tdms.TdmsFile):
        file = tdms.TdmsFile.read(file)

    # Set up the coordinate_string_map if necessary
    if coordinate_string_map is None:
        coordinate_string_map = {}

    # Go through and get the metadata
    dt = 1/file['TH Data Info'].properties['Sample Rate']
    run_number = file['Test Run Info'].properties['Run Number']
    run_description = file['Test Run Info'].properties['Run Description']
    test_axis = file['Test Run Info'].properties['Test Axis']
    test_type = file['Test Run Info'].properties['Test Type']
    test_title = file['Test Series Info'].properties['Test Series Title']

    # Set up abscissa limits
    if min_time is None:
        min_index = 0
    else:
        min_index = int(np.ceil(min_time / dt))
    if max_time is None:
        max_index = None
    else:
        max_index = int(np.floor(max_time / dt))+1
    abscissa_slice = slice(min_index, max_index)

    # Now let's go through the channels
    data = []
    for i, channel in enumerate(file['TH Data'].channels()):
        if channel_indices is not None and i not in channel_indices:
            continue
        print('Reading {:}'.format(channel.name))
        dsa_device = channel.properties['DSA_Device']
        dsa_sn = channel.properties['DSA_S/N']
        dsa_channel = channel.properties['DSA_Channel']
        trans_type = channel.properties['Trans_Type']
        ch_eu = channel.properties['Ch_EU']
        trans_model = channel.properties['Trans_Model']
        trans_axis = channel.properties['Trans_Axis']
        trans_sn = channel.properties['Trans_S/N']
        coordinate = channel.properties[coordinate_property]
        if coordinate in coordinate_string_map:
            coordinate = coordinate_string_map[coordinate]
        try:
            coordinate = coordinate_array(string_array=coordinate)
        except (KeyError, ValueError):
            coordinate = coordinate_array(string_array=invalid_coordinate_value)
        time_data = channel[abscissa_slice]
        num_elements = time_data.size
        abscissa = np.arange(num_elements)*dt + min_index*dt
        # Comment 1 will be Trans_Type :: Ch_EU
        # Comment 2 will be DSA_Device DSA_S/N :: DSA_Channel
        # Comment 3 will be Test Series Title :: Run Number :: Run Description, Test Axis :: Test Type
        # Comment 5 will be the coordinate value as a string, just in case we can't convert it to a coordinate
        comment1 = 'Type: {:}; Unit: {:}'.format(trans_type, ch_eu)
        comment2 = 'Device: {:}; S/N: {:}; Ch: {:}'.format(dsa_device, dsa_sn, dsa_channel)
        comment3 = '{:} Run {:}: {:} {:} {:}'.format(test_title, run_number, run_description, test_axis, test_type)
        comment4 = 'Sensor: {:} {:} {:}'.format(trans_model, trans_sn, trans_axis)
        comment5 = '{:}'.format(channel.properties[coordinate_property])
        # Now create the data object
        data.append(data_array(FunctionTypes.TIME_RESPONSE, abscissa, time_data, coordinate,
                    comment1, comment2, comment3, comment4, comment5)[np.newaxis])
    data = np.concatenate(data)
    return data


def read_mat_time_history(data_directory, coordinate_property="Ch_ID",
                          coordinate_string_map=None, invalid_coordinate_value='999999',
                          min_time=None, max_time=None, file_pattern='Ch*_THData.mat',
                          file_sort_key=lambda x: int(os.path.split(x)[-1].split('_')[0].replace('Ch', ''))):
    # Set up the coordinate_string_map if necessary
    if coordinate_string_map is None:
        coordinate_string_map = {}

    # Find files in the directory
    mat_files = glob(os.path.join(data_directory, file_pattern))
    # Sort the files by number
    mat_files = sorted(mat_files, key=file_sort_key)
    data_arrays = []
    for i, data_file in enumerate(mat_files):
        print('Reading File {:}'.format(data_file))
        data = loadmat(data_file)
        # Get sample rate
        dt = data['dt'].squeeze()
        # Set up abscissa limits
        if min_time is None:
            min_index = 0
        else:
            min_index = int(np.ceil(min_time / dt))
        if max_time is None:
            max_index = None
        else:
            max_index = int(np.floor(max_time / dt))+1
        abscissa_slice = slice(min_index, max_index)
        # Get metadata
        run_number = str(data['Test_RunNum'].squeeze())
        run_description = str(data['Test_Description'].squeeze())
        test_axis = str(data['Test_Axis'].squeeze())
        dsa_channel = str(data['Daq_Ch'].squeeze())
        trans_type = str(data['Data_Description'].squeeze())
        ch_eu = str(data['Ch_Unit'].squeeze())
        trans_model = str(data['Trans_Model'].squeeze())
        trans_sn = str(data['Trans_SN'].squeeze())
        coordinate = str(data[coordinate_property].squeeze())
        if coordinate in coordinate_string_map:
            coordinate = coordinate_string_map[coordinate]
        try:
            coordinate = coordinate_array(string_array=coordinate)
        except (KeyError, ValueError):
            coordinate = coordinate_array(string_array=invalid_coordinate_value)
        # Get the data
        time_data = data['THData'].squeeze()[abscissa_slice]
        num_elements = time_data.size
        abscissa = np.arange(num_elements)*dt + min_index*dt
        comment1 = 'Type: {:}; Unit: {:}'.format(trans_type, ch_eu)
        comment2 = 'Channel: {:}'.format(dsa_channel)
        comment3 = 'Run {:}: {:} {:}'.format(run_number, run_description, test_axis)
        comment4 = 'Sensor: {:} {:}'.format(trans_model, trans_sn)
        comment5 = '{:}'.format(str(data[coordinate_property].squeeze()))
        # Now create the data object
        data_arrays.append(data_array(FunctionTypes.TIME_RESPONSE, abscissa, time_data, coordinate,
                                      comment1, comment2, comment3, comment4, comment5)[np.newaxis])
    data_arrays = np.concatenate(data_arrays)
    return data_arrays


def read_mat_shock(data_file, coordinate_property="Ch_Info",
                   coordinate_string_map=None, invalid_coordinate_value='999999',
                   min_time=None, max_time=None, read_filtered_time_data=True):
    # Set up the coordinate_string_map if necessary
    if coordinate_string_map is None:
        coordinate_string_map = {}
    data = loadmat(data_file)
    # Get sample rate
    dt = data['dt'].squeeze()
    # Set up abscissa limits
    if min_time is None:
        min_index = 0
    else:
        min_index = int(np.ceil(min_time / dt))
    if max_time is None:
        max_index = None
    else:
        max_index = int(np.floor(max_time / dt))+1
    abscissa_slice = slice(min_index, max_index)
    # Get metadata
    run_number = str(data['Test_RunNum'].squeeze())
    run_description = str(data['Test_Description'].squeeze())
    test_axis = str(data['Test_Axis'].squeeze())
    dsa_channel = [str(ch) for ch in data['Daq_Ch'].squeeze()]
    trans_type = [str(ch) for ch in data['Data_Description'].squeeze()]
    ch_eu = [str(ch) for ch in data['Ch_Unit'].squeeze()]
    coordinates_raw = data[coordinate_property]
    coordinates = []
    for coordinate in coordinates_raw:
        coordinate = str(coordinate).strip()
        if coordinate in coordinate_string_map:
            coordinate = coordinate_string_map[coordinate]
        try:
            coordinate = coordinate_array(string_array=coordinate)
        except (KeyError, ValueError):
            coordinate = coordinate_array(string_array=invalid_coordinate_value)
        coordinates.append(coordinate[np.newaxis])
    coordinates = np.concatenate(coordinates)[:, np.newaxis]
    time_data = data['shk' if read_filtered_time_data else 'unfilshk'][abscissa_slice].T
    num_elements = time_data.shape[-1]
    abscissa = np.arange(num_elements)*dt + min_index*dt
    if len(trans_type) == 0:
        comment1 = ['Unit: {:}'.format(ch) for ch in ch_eu]
    else:
        comment1 = ['Type: {:}; Unit: {:}'.format(t, ch) for t, ch in zip(trans_type, ch_eu)]
    comment2 = ['Channel: {:}'.format(ch) for ch in dsa_channel]
    comment3 = 'Run {:}: {:} {:}'.format(run_number, run_description, test_axis)
    comment5 = data[coordinate_property]
    time_data = data_array(FunctionTypes.TIME_RESPONSE, abscissa, time_data,
                           coordinates, comment1, comment2, comment3,
                           comment5=comment5)
    return time_data


def read_mat_random(data_file, coordinate_property="Ch_Info",
                    reference_coordinate_property="FRF_RefID",
                    coordinate_string_map=None, invalid_coordinate_value='999999'):
    # Set up the coordinate_string_map if necessary
    if coordinate_string_map is None:
        coordinate_string_map = {}
    data = loadmat(data_file)
    abscissa = data['Freq'].squeeze()
    coherence = data['Coh'].T
    frf = data['FRF'].T
    psd = data['PSD'].T
    coordinates_raw = data[coordinate_property]
    coordinates = []
    for coordinate in coordinates_raw:
        coordinate = str(coordinate).strip()
        if coordinate in coordinate_string_map:
            coordinate = coordinate_string_map[coordinate]
        try:
            coordinate = coordinate_array(string_array=coordinate)
        except (KeyError, ValueError):
            coordinate = coordinate_array(string_array=invalid_coordinate_value)
        coordinates.append(coordinate[np.newaxis])
    coordinates = np.concatenate(coordinates)[:, np.newaxis]
    reference_coordinates_raw = data[reference_coordinate_property]
    reference_coordinates = []
    for coordinate in reference_coordinates_raw:
        coordinate = str(coordinate).strip()
        if coordinate in coordinate_string_map:
            coordinate = coordinate_string_map[coordinate]
        try:
            coordinate = coordinate_array(string_array=coordinate)
        except (KeyError, ValueError):
            coordinate = coordinate_array(string_array=invalid_coordinate_value)
        reference_coordinates.append(coordinate[np.newaxis])
    reference_coordinates = np.concatenate(reference_coordinates)
    # Get metadata
    run_number = str(data['Test_RunNum'].squeeze())
    run_description = str(data['Test_Description'].squeeze())
    test_axis = str(data['Test_Axis'].squeeze())
    dsa_channel = [str(ch).strip() for ch in data['Daq_Ch'].squeeze()]
    trans_type = [str(ch).strip() for ch in data['Ch_Type'].squeeze()]
    ch_eu = [str(ch).strip() for ch in data['Ch_Unit'].squeeze()]
    if len(trans_type) == 0:
        psd_comment1 = ['Unit: {:}^2/Hz'.format(ch) for ch in ch_eu]
        coh_comment1 = ['' for ch in ch_eu]
        frf_comment1 = ['Unit: {:}'.format(ch) for ch in ch_eu]
    else:
        psd_comment1 = ['Type: {:}; Unit: {:}^2/Hz'.format(t, ch) for t, ch in zip(trans_type, ch_eu)]
        coh_comment1 = ['Type: {:};'.format(t) for t, ch in zip(trans_type, ch_eu)]
        frf_comment1 = ['Type: {:}; Unit: {:}'.format(t, ch) for t, ch in zip(trans_type, ch_eu)]
    comment2 = ['Channel: {:}'.format(ch) for ch in dsa_channel]
    comment3 = 'Run {:}: {:} {:}'.format(run_number, run_description, test_axis)
    comment5 = data[coordinate_property]
    psd_data = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                          abscissa, psd, np.concatenate((coordinates, coordinates), axis=-1),
                          psd_comment1, comment2, comment3, comment5=comment5)
    coh_data = data_array(FunctionTypes.COHERENCE,
                          abscissa, coherence, outer_product(coordinates.flatten(), reference_coordinates.flatten())[:, 0, :],
                          coh_comment1, comment2, comment3, comment5=comment5)
    frf_data = data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                          abscissa, frf, outer_product(coordinates.flatten(), reference_coordinates.flatten())[:, 0, :],
                          frf_comment1, comment2, comment3, comment5=comment5)
    return psd_data, coh_data, frf_data
