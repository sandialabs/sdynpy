# -*- coding: utf-8 -*-
"""
Load in time data from Rattlesnake runs

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

import netCDF4 as nc4
import numpy as np
from ..core.sdynpy_coordinate import (coordinate_array, outer_product,
                                      CoordinateArray, _string_map)
from ..core.sdynpy_data import data_array, FunctionTypes
from ..core.sdynpy_system import System
import pandas as pd
import sys
import openpyxl as opxl
import os
import warnings


def read_rattlesnake_output(file, coordinate_override_column=None, read_only_indices=None,
                            read_variable='time_data'):
    """
    Reads in a Rattlesnake data file and returns the time history array as well
    as the channel table

    Parameters
    ----------
    file : str or nc4.Dataset
        Path to the file to read in or an already open

    Returns
    -------
    data_array : TimeHistoryArray
        Time history data in the Rattlesnake output file
    channel_table : DataFrame
        Pandas Dataframe containing the channel table information

    """
    if isinstance(file, str):
        ds = nc4.Dataset(file, 'r')
    elif isinstance(file, nc4.Dataset):
        ds = file
    if read_only_indices is None:
        read_only_indices = slice(None)
    output_data = np.array(ds[read_variable][...][read_only_indices])
    abscissa = np.arange(output_data.shape[-1]) / ds.sample_rate
    if coordinate_override_column is None:
        nodes = [int(''.join(char for char in node if char in '0123456789'))
                 for node in ds['channels']['node_number'][...][read_only_indices]]
        directions = np.array(ds['channels']['node_direction'][...][read_only_indices], dtype='<U3')
        coordinates = coordinate_array(nodes, directions)[:, np.newaxis]
    else:
        coordinates = coordinate_array(string_array=ds['channels'][coordinate_override_column][read_only_indices])[
            :, np.newaxis]
    array = {name: np.array(variable[:]) for name, variable in ds['channels'].variables.items()}
    channel_table = pd.DataFrame(array)
    comment1 = np.char.add(np.char.add(np.array(ds['channels']['channel_type'][...][read_only_indices], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['unit'][...][read_only_indices], dtype='<U80'))
    comment2 = np.char.add(np.char.add(np.array(ds['channels']['physical_device'][...][read_only_indices], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['physical_channel'][...][read_only_indices], dtype='<U80'))
    comment3 = np.char.add(np.char.add(np.array(ds['channels']['feedback_device'][...][read_only_indices], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['feedback_channel'][...][read_only_indices], dtype='<U80'))
    comment4 = np.array(ds['channels']['comment'][...][read_only_indices], dtype='<U80')
    comment5 = np.array(ds['channels']['make'][...][read_only_indices], dtype='<U80')
    for key in ('model', 'serial_number', 'triax_dof'):
        comment5 = np.char.add(comment5, np.array(' '))
        comment5 = np.char.add(comment5, np.array(ds['channels'][key][...][read_only_indices], dtype='<U80'))
    time_data = data_array(FunctionTypes.TIME_RESPONSE,
                           abscissa,
                           output_data,
                           coordinates,
                           comment1,
                           comment2,
                           comment3,
                           comment4,
                           comment5)
    if isinstance(file, str):
        ds.close()
    return time_data, channel_table


def read_system_id_data(file):
    if isinstance(file, str):
        file = np.load(file)
    df = file['sysid_frequency_spacing']
    if np.isnan(file['response_transformation_matrix']):
        try:
            response_dofs = coordinate_array(
                [int(v) for v in file['channel_node_number'][file['response_indices']]],
                file['channel_node_direction'][file['response_indices']])
        except Exception:
            response_dofs = coordinate_array(file['response_indices']+1, 0)
    else:
        response_dofs = coordinate_array(np.arange(file['response_transformation_matrix'].shape[0])+1, 0)
    if np.isnan(file['reference_transformation_matrix']):
        try:
            reference_dofs = coordinate_array(
                [int(v) for v in file['channel_node_number'][file['reference_indices']]],
                file['channel_node_direction'][file['reference_indices']])
        except Exception:
            reference_dofs = coordinate_array(file['reference_indices']+1, 0)
    else:
        reference_dofs = coordinate_array(np.arange(file['reference_transformation_matrix'].shape[0])+1, 0)
    ordinate = np.moveaxis(file['frf_data'], 0, -1)
    frfs = data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                      df*np.arange(ordinate.shape[-1]), ordinate,
                      outer_product(response_dofs, reference_dofs))
    ordinate = np.moveaxis(file['response_cpsd'], 0, -1)
    response_cpsd = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                               df*np.arange(ordinate.shape[-1]), ordinate,
                               outer_product(response_dofs, response_dofs))
    ordinate = np.moveaxis(file['reference_cpsd'], 0, -1)
    reference_cpsd = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                                df*np.arange(ordinate.shape[-1]), ordinate,
                                outer_product(reference_dofs, reference_dofs))
    ordinate = np.moveaxis(file['response_noise_cpsd'], 0, -1)
    response_noise_cpsd = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                                     df*np.arange(ordinate.shape[-1]), ordinate,
                                     outer_product(response_dofs, response_dofs))
    ordinate = np.moveaxis(file['reference_noise_cpsd'], 0, -1)
    reference_noise_cpsd = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                                      df*np.arange(ordinate.shape[-1]), ordinate,
                                      outer_product(reference_dofs, reference_dofs))
    ordinate = np.moveaxis(file['coherence'], 0, -1)
    coherence = data_array(FunctionTypes.MULTIPLE_COHERENCE,
                           df*np.arange(ordinate.shape[-1]), ordinate,
                           outer_product(response_dofs))
    return frfs, response_cpsd, reference_cpsd, response_noise_cpsd, reference_noise_cpsd, coherence


def read_random_spectral_data(file, coordinate_override_column=None):
    if isinstance(file, str):
        ds = nc4.Dataset(file, 'r')
    elif isinstance(file, nc4.Dataset):
        ds = file
    coordinate_override_column = None

    environment = [group for group in ds.groups if not group == 'channels'][0]

    # Get the channels in the group
    if coordinate_override_column is None:
        nodes = [int(''.join(char for char in node if char in '0123456789'))
                 for node in ds['channels']['node_number']]
        directions = np.array(ds['channels']['node_direction'][:], dtype='<U3')
        coordinates = coordinate_array(nodes, directions)
    else:
        coordinates = coordinate_array(string_array=ds['channels'][coordinate_override_column])
    drives = ds['channels']['feedback_device'][:] != ''

    # Cull down to just those in the environment
    environment_index = np.where(ds['environment_names'][:] == environment)[0][0]
    environment_channels = ds['environment_active_channels'][:, environment_index].astype(bool)

    drives = drives[environment_channels]
    coordinates = coordinates[environment_channels]

    control_indices = ds[environment]['control_channel_indices'][:]

    control_coordinates = coordinates[control_indices]

    drive_coordinates = coordinates[drives]

    # Load the spectral data
    frequencies = np.array(ds[environment]['specification_frequency_lines'][:])

    spec_cpsd = np.moveaxis(
        np.array(ds[environment]['specification_cpsd_matrix_real'][:]
                 + 1j*ds[environment]['specification_cpsd_matrix_imag'][:]),
        0, -1)

    response_cpsd = np.moveaxis(
        np.array(ds[environment]['response_cpsd_real'][:]
                 + 1j*ds[environment]['response_cpsd_imag'][:]),
        0, -1)

    drive_cpsd = np.moveaxis(
        np.array(ds[environment]['drive_cpsd_real'][:]
                 + 1j*ds[environment]['drive_cpsd_imag'][:]),
        0, -1)

    response_coordinates = outer_product(control_coordinates, control_coordinates)
    drive_coordinates = outer_product(drive_coordinates, drive_coordinates)

    comment1 = np.char.add(np.char.add(np.array(ds['channels']['channel_type'][:], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['unit'][:], dtype='<U80'))
    comment2 = np.char.add(np.char.add(np.array(ds['channels']['physical_device'][:], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['physical_channel'][:], dtype='<U80'))
    comment3 = np.char.add(np.char.add(np.array(ds['channels']['feedback_device'][:], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['feedback_channel'][:], dtype='<U80'))
    comment4 = np.array(ds['channels']['comment'][:], dtype='<U80')
    comment5 = np.array(ds['channels']['make'][:], dtype='<U80')
    for key in ('model', 'serial_number', 'triax_dof'):
        comment5 = np.char.add(comment5, np.array(' '))
        comment5 = np.char.add(comment5, np.array(ds['channels'][key][:], dtype='<U80'))

    comment1 = comment1[environment_channels]
    comment2 = comment2[environment_channels]
    comment3 = comment3[environment_channels]
    comment4 = comment4[environment_channels]
    comment5 = comment5[environment_channels]

    comment1_response = np.empty((response_coordinates.shape[0], response_coordinates.shape[1]), dtype=comment1.dtype)
    comment2_response = np.empty((response_coordinates.shape[0], response_coordinates.shape[1]), dtype=comment1.dtype)
    comment3_response = np.empty((response_coordinates.shape[0], response_coordinates.shape[1]), dtype=comment1.dtype)
    comment4_response = np.empty((response_coordinates.shape[0], response_coordinates.shape[1]), dtype=comment1.dtype)
    comment5_response = np.empty((response_coordinates.shape[0], response_coordinates.shape[1]), dtype=comment1.dtype)
    for i, idx in enumerate(control_indices):
        for j, jdx in enumerate(control_indices):
            comment1_response[i, j] = comment1[idx] + ' // ' + comment1[jdx]
            comment2_response[i, j] = comment2[idx] + ' // ' + comment2[jdx]
            comment3_response[i, j] = comment3[idx] + ' // ' + comment3[jdx]
            comment4_response[i, j] = comment4[idx] + ' // ' + comment4[jdx]
            comment5_response[i, j] = comment5[idx] + ' // ' + comment5[jdx]

    comment1_drive = np.empty((drive_coordinates.shape[0], drive_coordinates.shape[1]), dtype=comment1.dtype)
    comment2_drive = np.empty((drive_coordinates.shape[0], drive_coordinates.shape[1]), dtype=comment1.dtype)
    comment3_drive = np.empty((drive_coordinates.shape[0], drive_coordinates.shape[1]), dtype=comment1.dtype)
    comment4_drive = np.empty((drive_coordinates.shape[0], drive_coordinates.shape[1]), dtype=comment1.dtype)
    comment5_drive = np.empty((drive_coordinates.shape[0], drive_coordinates.shape[1]), dtype=comment1.dtype)
    drive_indices = np.where(drives)[0]
    for i, idx in enumerate(drive_indices):
        for j, jdx in enumerate(drive_indices):
            comment1_drive[i, j] = comment1[idx] + ' // ' + comment1[jdx]
            comment2_drive[i, j] = comment2[idx] + ' // ' + comment2[jdx]
            comment3_drive[i, j] = comment3[idx] + ' // ' + comment3[jdx]
            comment4_drive[i, j] = comment4[idx] + ' // ' + comment4[jdx]
            comment5_drive[i, j] = comment5[idx] + ' // ' + comment5[jdx]

    # Save the data to SDynpy objects
    response_cpsd = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                               frequencies, response_cpsd, response_coordinates,
                               comment1_response, comment2_response, comment3_response,
                               comment4_response, comment5_response)
    spec_cpsd = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                           frequencies, spec_cpsd, response_coordinates,
                           comment1_response, comment2_response, comment3_response,
                           comment4_response, comment5_response)
    drive_cpsd = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                            frequencies, drive_cpsd, drive_coordinates,
                            comment1_drive, comment2_drive, comment3_drive,
                            comment4_drive, comment5_drive)
    return response_cpsd, spec_cpsd, drive_cpsd


def read_modal_data(file, coordinate_override_column=None, read_only_indices=None):
    if isinstance(file, str):
        ds = nc4.Dataset(file, 'r')
    elif isinstance(file, nc4.Dataset):
        ds = file
    if read_only_indices is None:
        read_only_indices = slice(None)
    # Get parameters
    num_channels = ds.groups['channels'].variables['physical_device'].size
    group_key = [g for g in ds.groups if not g == 'channels'][0]
    group = ds.groups[group_key]
    sample_rate = ds.sample_rate
    samples_per_frame = group.samples_per_frame
    num_averages = group.num_averages
    # Load in the time data
    try:
        output_data = np.array(ds['time_data'][...][read_only_indices]).reshape(num_channels, num_averages, samples_per_frame).transpose(1, 0, 2)
    except ValueError:
        warnings.warn('Number of averages in the time data does not match the number of averages specified in the test settings.  Your test may be incomplete.')
        output_data = np.array(ds['time_data'][...][read_only_indices]).reshape(num_channels, -1, samples_per_frame).transpose(1, 0, 2)
    abscissa = np.arange(samples_per_frame) / sample_rate
    if coordinate_override_column is None:
        nodes = [int(''.join(char for char in node if char in '0123456789'))
                 for node in ds['channels']['node_number'][...][read_only_indices]]
        directions = np.array(ds['channels']['node_direction'][...][read_only_indices], dtype='<U3')
        coordinates = coordinate_array(nodes, directions)[:, np.newaxis]
    else:
        coordinates = coordinate_array(string_array=ds['channels'][coordinate_override_column][read_only_indices])[
            :, np.newaxis]
    array = {name: np.array(variable[:]) for name, variable in ds['channels'].variables.items()}
    channel_table = pd.DataFrame(array)
    comment1 = np.char.add(np.char.add(np.array(ds['channels']['channel_type'][...][read_only_indices], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['unit'][...][read_only_indices], dtype='<U80'))
    comment2 = np.char.add(np.char.add(np.array(ds['channels']['physical_device'][...][read_only_indices], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['physical_channel'][...][read_only_indices], dtype='<U80'))
    comment3 = np.char.add(np.char.add(np.array(ds['channels']['feedback_device'][...][read_only_indices], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['feedback_channel'][...][read_only_indices], dtype='<U80'))
    comment4 = np.array(ds['channels']['comment'][...][read_only_indices], dtype='<U80')
    comment5 = np.array(ds['channels']['make'][...][read_only_indices], dtype='<U80')
    for key in ('model', 'serial_number', 'triax_dof'):
        comment5 = np.char.add(comment5, np.array(' '))
        comment5 = np.char.add(comment5, np.array(ds['channels'][key][...][read_only_indices], dtype='<U80'))
    time_data = data_array(FunctionTypes.TIME_RESPONSE,
                           abscissa,
                           output_data,
                           coordinates,
                           comment1,
                           comment2,
                           comment3,
                           comment4,
                           comment5)
    # Response and Reference Indices
    kept_indices = np.arange(num_channels)[read_only_indices]
    reference_indices = np.array(group.variables['reference_channel_indices'][:])
    response_indices = np.array(group.variables['response_channel_indices'][:])
    keep_response_indices = np.array([i for i, index in enumerate(response_indices) if index in kept_indices])
    keep_reference_indices = np.array([i for i, index in enumerate(reference_indices) if index in kept_indices])
    frequency_lines = np.arange(group.dimensions['fft_lines'].size)*sample_rate/samples_per_frame
    coherence_data = np.array(group['coherence'][:, keep_response_indices]).T
    comment1 = np.char.add(np.char.add(np.array(ds['channels']['channel_type'][...][response_indices[keep_response_indices]], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['unit'][...][response_indices[keep_response_indices]], dtype='<U80'))
    comment2 = np.char.add(np.char.add(np.array(ds['channels']['physical_device'][...][response_indices[keep_response_indices]], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['physical_channel'][...][response_indices[keep_response_indices]], dtype='<U80'))
    comment3 = np.char.add(np.char.add(np.array(ds['channels']['feedback_device'][...][response_indices[keep_response_indices]], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['feedback_channel'][...][response_indices[keep_response_indices]], dtype='<U80'))
    comment4 = np.array(ds['channels']['comment'][...][response_indices[keep_response_indices]], dtype='<U80')
    comment5 = np.array(ds['channels']['make'][...][response_indices[keep_response_indices]], dtype='<U80')
    for key in ('model', 'serial_number', 'triax_dof'):
        comment5 = np.char.add(comment5, np.array(' '))
        comment5 = np.char.add(comment5, np.array(ds['channels'][key][...][response_indices[keep_response_indices]], dtype='<U80'))
    coherence_data = data_array(FunctionTypes.MULTIPLE_COHERENCE,
                                frequency_lines,
                                coherence_data,
                                coordinates[response_indices[keep_response_indices]],
                                comment1,
                                comment2,
                                comment3,
                                comment4,
                                comment5)
    # Frequency Response Functions
    frf_data = np.moveaxis(np.array(group['frf_data_real'])[:, keep_response_indices[:, np.newaxis], keep_reference_indices]
                           + np.array(group['frf_data_imag'])[:, keep_response_indices[:, np.newaxis], keep_reference_indices]*1j, 0, -1)
    frf_coordinate = outer_product(coordinates[response_indices[keep_response_indices], 0],
                                   coordinates[reference_indices[keep_reference_indices], 0])
    # print(response_indices[keep_response_indices])
    # print(reference_indices[keep_reference_indices])
    response_comment1 = np.char.add(np.char.add(np.array(ds['channels']['channel_type'][...][response_indices[keep_response_indices]], dtype='<U80'),
                                                np.array(' :: ')),
                                    np.array(ds['channels']['unit'][...][response_indices[keep_response_indices]], dtype='<U80'))
    response_comment2 = np.char.add(np.char.add(np.array(ds['channels']['physical_device'][...][response_indices[keep_response_indices]], dtype='<U80'),
                                                np.array(' :: ')),
                                    np.array(ds['channels']['physical_channel'][...][response_indices[keep_response_indices]], dtype='<U80'))
    response_comment3 = np.char.add(np.char.add(np.array(ds['channels']['feedback_device'][...][response_indices[keep_response_indices]], dtype='<U80'),
                                                np.array(' :: ')),
                                    np.array(ds['channels']['feedback_channel'][...][response_indices[keep_response_indices]], dtype='<U80'))
    response_comment4 = np.array(ds['channels']['comment'][...][response_indices[keep_response_indices]], dtype='<U80')
    response_comment5 = np.array(ds['channels']['make'][...][response_indices[keep_response_indices]], dtype='<U80')
    for key in ('model', 'serial_number', 'triax_dof'):
        response_comment5 = np.char.add(response_comment5, np.array(' '))
        response_comment5 = np.char.add(response_comment5, np.array(ds['channels'][key][...][response_indices[keep_response_indices]], dtype='<U80'))
    reference_comment1 = np.char.add(np.char.add(np.array(ds['channels']['channel_type'][...][reference_indices[keep_reference_indices]], dtype='<U80'),
                                                 np.array(' :: ')),
                                     np.array(ds['channels']['unit'][...][reference_indices[keep_reference_indices]], dtype='<U80'))
    reference_comment2 = np.char.add(np.char.add(np.array(ds['channels']['physical_device'][...][reference_indices[keep_reference_indices]], dtype='<U80'),
                                                 np.array(' :: ')),
                                     np.array(ds['channels']['physical_channel'][...][reference_indices[keep_reference_indices]], dtype='<U80'))
    reference_comment3 = np.char.add(np.char.add(np.array(ds['channels']['feedback_device'][...][reference_indices[keep_reference_indices]], dtype='<U80'),
                                                 np.array(' :: ')),
                                     np.array(ds['channels']['feedback_channel'][...][reference_indices[keep_reference_indices]], dtype='<U80'))
    reference_comment4 = np.array(ds['channels']['comment'][...][reference_indices[keep_reference_indices]], dtype='<U80')
    reference_comment5 = np.array(ds['channels']['make'][...][reference_indices[keep_reference_indices]], dtype='<U80')
    for key in ('model', 'serial_number', 'triax_dof'):
        reference_comment5 = np.char.add(reference_comment5, np.array(' '))
        reference_comment5 = np.char.add(reference_comment5, np.array(ds['channels'][key][...][reference_indices[keep_reference_indices]], dtype='<U80'))
    response_comment1, reference_comment1 = np.broadcast_arrays(response_comment1[:, np.newaxis], reference_comment1)
    comment1 = np.char.add(np.char.add(response_comment1, np.array(' / ')), reference_comment1)
    response_comment2, reference_comment2 = np.broadcast_arrays(response_comment2[:, np.newaxis], reference_comment2)
    comment2 = np.char.add(np.char.add(response_comment2, np.array(' / ')), reference_comment2)
    response_comment3, reference_comment3 = np.broadcast_arrays(response_comment3[:, np.newaxis], reference_comment3)
    comment3 = np.char.add(np.char.add(response_comment3, np.array(' / ')), reference_comment3)
    response_comment4, reference_comment4 = np.broadcast_arrays(response_comment4[:, np.newaxis], reference_comment4)
    comment4 = np.char.add(np.char.add(response_comment4, np.array(' / ')), reference_comment4)
    response_comment5, reference_comment5 = np.broadcast_arrays(response_comment5[:, np.newaxis], reference_comment5)
    comment5 = np.char.add(np.char.add(response_comment5, np.array(' / ')), reference_comment5)
    frf_data = data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                          frequency_lines,
                          frf_data,
                          frf_coordinate,
                          comment1,
                          comment2,
                          comment3,
                          comment4,
                          comment5)
    return time_data, frf_data, coherence_data, channel_table


def read_transient_control_data(file, coordinate_override_column=None):
    if isinstance(file, str):
        ds = nc4.Dataset(file, 'r')
    elif isinstance(file, nc4.Dataset):
        ds = file
    coordinate_override_column = None

    environment = [group for group in ds.groups if not group == 'channels'][0]

    # Get the channels in the group
    if coordinate_override_column is None:
        nodes = [int(''.join(char for char in node if char in '0123456789'))
                 for node in ds['channels']['node_number']]
        directions = np.array(ds['channels']['node_direction'][:], dtype='<U3')
        coordinates = coordinate_array(nodes, directions)
    else:
        coordinates = coordinate_array(string_array=ds['channels'][coordinate_override_column])
    drives = ds['channels']['feedback_device'][:] != ''

    # Cull down to just those in the environment
    environment_index = np.where(ds['environment_names'][:] == environment)[0][0]
    environment_channels = ds['environment_active_channels'][:, environment_index].astype(bool)

    drives = drives[environment_channels]
    coordinates = coordinates[environment_channels]

    control_indices = ds[environment]['control_channel_indices'][:]

    control_coordinates = coordinates[control_indices]

    drive_coordinates = coordinates[drives]

    # Load the time data
    timesteps = np.arange(ds[environment].dimensions['signal_samples'].size)/ds.sample_rate

    spec_signal = np.array(ds[environment]['control_signal'][...])

    response_signal = np.array(ds[environment]['control_response'][...])

    drive_signal = np.array(ds[environment]['control_drives'][...])

    response_coordinates = control_coordinates[:, np.newaxis]
    drive_coordinates = drive_coordinates[:, np.newaxis]

    comment1 = np.char.add(np.char.add(np.array(ds['channels']['channel_type'][:], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['unit'][:], dtype='<U80'))
    comment2 = np.char.add(np.char.add(np.array(ds['channels']['physical_device'][:], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['physical_channel'][:], dtype='<U80'))
    comment3 = np.char.add(np.char.add(np.array(ds['channels']['feedback_device'][:], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['feedback_channel'][:], dtype='<U80'))
    comment4 = np.array(ds['channels']['comment'][:], dtype='<U80')
    comment5 = np.array(ds['channels']['make'][:], dtype='<U80')
    for key in ('model', 'serial_number', 'triax_dof'):
        comment5 = np.char.add(comment5, np.array(' '))
        comment5 = np.char.add(comment5, np.array(ds['channels'][key][:], dtype='<U80'))

    comment1 = comment1[environment_channels]
    comment2 = comment2[environment_channels]
    comment3 = comment3[environment_channels]
    comment4 = comment4[environment_channels]
    comment5 = comment5[environment_channels]

    # Save the data to SDynpy objects
    response_signal = data_array(FunctionTypes.TIME_RESPONSE,
                                 timesteps, response_signal, response_coordinates,
                                 comment1[control_indices], comment2[control_indices], comment3[control_indices],
                                 comment4[control_indices], comment5[control_indices])
    spec_signal = data_array(FunctionTypes.TIME_RESPONSE,
                             timesteps, spec_signal, response_coordinates,
                             comment1[control_indices], comment2[control_indices], comment3[control_indices],
                             comment4[control_indices], comment5[control_indices])
    drive_signal = data_array(FunctionTypes.TIME_RESPONSE,
                              timesteps, drive_signal, drive_coordinates,
                              comment1[drives], comment2[drives], comment3[drives],
                              comment4[drives], comment5[drives])
    return response_signal, spec_signal, drive_signal


def create_synthetic_test(spreadsheet_file_name: str,
                          system_filename: str, system: System,
                          excitation_coordinates: CoordinateArray,
                          response_coordinates: CoordinateArray,
                          rattlesnake_directory: str,
                          displacement_derivative=2,
                          sample_rate: int = None,
                          time_per_read: float = None,
                          time_per_write: float = None,
                          integration_oversample: int = 10,
                          environments: list = [],
                          channel_comment_data: list = None,
                          channel_serial_number_data: list = None,
                          channel_triax_dof_data: list = None,
                          channel_engineering_unit_data: list = None,
                          channel_warning_level_data: list = None,
                          channel_abort_level_data: list = None,
                          channel_active_in_environment_data: dict = None
                          ):
    A, B, C, D = system.to_state_space(displacement_derivative == 0,
                                       displacement_derivative == 1,
                                       displacement_derivative == 2,
                                       True,
                                       response_coordinates,
                                       excitation_coordinates)
    np.savez(system_filename, A=A, B=B, C=C, D=D)
    # Load in Rattlesnake to create a template for the test
    sys.path.insert(0, rattlesnake_directory)
    import components as rs
    environment_data = []
    for environment_type, environment_name in environments:
        # Find the identifier
        environment_type = rs.environments.ControlTypes[environment_type.upper()]
        environment_data.append((environment_type, environment_name))
    rs.ui_utilities.save_combined_environments_profile_template(spreadsheet_file_name, environment_data)
    sys.path.pop(0)
    # Populate the channel table
    workbook = opxl.load_workbook(spreadsheet_file_name)
    worksheet = workbook.get_sheet_by_name('Channel Table')
    index = 3
    for i, channel in enumerate(response_coordinates):
        worksheet.cell(index, 1, i+1)
        worksheet.cell(index, 2, channel.node)
        worksheet.cell(index, 3, _string_map[channel.direction])
        worksheet.cell(index, 12, 'Virtual')
        worksheet.cell(index, 14, 'Accel')
        index += 1
    for i, channel in enumerate(excitation_coordinates):
        worksheet.cell(index, 1, len(response_coordinates)+i+1)
        worksheet.cell(index, 2, channel.node)
        worksheet.cell(index, 3, _string_map[channel.direction])
        worksheet.cell(index, 12, 'Virtual')
        worksheet.cell(index, 14, 'Force')
        worksheet.cell(index, 20, 'Shaker')
        index += 1
    # Go through the various channel table data that could have been optionally
    # provided
    for column, data in [(4, channel_comment_data),
                         (5, channel_serial_number_data),
                         (6, channel_triax_dof_data),
                         (8, channel_engineering_unit_data),
                         (22, channel_warning_level_data),
                         (23, channel_abort_level_data)]:
        if data is None:
            continue
        for row_index, value in enumerate(data):
            worksheet.cell(3+row_index, column, value)
    # Now fill out the environment table
    if channel_active_in_environment_data is not None:
        for environment_index, (environment_type, environment_name) in enumerate(environment_data):
            for row_index, value in enumerate(channel_active_in_environment_data[environment_name]):
                if value:
                    worksheet.cell(3+row_index, 24+environment_index, 'X')
    else:
        for environment_index, (environment_type, environment_name) in enumerate(environment_data):
            for row_index in range(response_coordinates.size + excitation_coordinates.size):
                worksheet.cell(3+row_index, 24+environment_index, 'X')
    worksheet = workbook.get_sheet_by_name('Hardware')
    worksheet.cell(1, 2, 5)
    worksheet.cell(2, 2, os.path.abspath(system_filename))
    if sample_rate is not None:
        worksheet.cell(3, 2, sample_rate)
    if time_per_read is not None:
        worksheet.cell(4, 2, time_per_read)
    if time_per_write is not None:
        worksheet.cell(5, 2, time_per_write)
    worksheet.cell(6, 2, 1)
    worksheet.cell(7, 2, integration_oversample)
    workbook.save(spreadsheet_file_name)
