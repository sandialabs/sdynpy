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
from ..core.sdynpy_coordinate import coordinate_array,outer_product
from ..core.sdynpy_data import data_array, FunctionTypes
import pandas as pd


def read_rattlesnake_output(file, coordinate_override_column=None, read_only_indices = None):
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
    output_data = np.array(ds['time_data'][read_only_indices])
    abscissa = np.arange(output_data.shape[-1]) / ds.sample_rate
    if coordinate_override_column is None:
        nodes = [int(''.join(char for char in node if char in '0123456789'))
                 for node in ds['channels']['node_number'][read_only_indices]]
        directions = np.array(ds['channels']['node_direction'][read_only_indices], dtype='<U3')
        coordinates = coordinate_array(nodes, directions)[:, np.newaxis]
    else:
        coordinates = coordinate_array(string_array=ds['channels'][coordinate_override_column][read_only_indices])[
            :, np.newaxis]
    array = {name: np.array(variable[:]) for name, variable in ds['channels'].variables.items()}
    channel_table = pd.DataFrame(array)
    comment1 = np.char.add(np.char.add(np.array(ds['channels']['channel_type'][read_only_indices], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['unit'][read_only_indices], dtype='<U80'))
    comment2 = np.char.add(np.char.add(np.array(ds['channels']['physical_device'][read_only_indices], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['physical_channel'][read_only_indices], dtype='<U80'))
    comment3 = np.char.add(np.char.add(np.array(ds['channels']['feedback_device'][read_only_indices], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['feedback_channel'][read_only_indices], dtype='<U80'))
    comment4 = np.array(ds['channels']['comment'][read_only_indices], dtype='<U80')
    comment5 = np.array(ds['channels']['make'][read_only_indices], dtype='<U80')
    for key in ('model', 'serial_number', 'triax_dof'):
        comment5 = np.char.add(comment5, np.array(' '))
        comment5 = np.char.add(comment5, np.array(ds['channels'][key][read_only_indices], dtype='<U80'))
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
    if isinstance(file,str):
        file = np.load(file)
    df = file['sysid_frequency_spacing']
    if np.isnan(file['response_transformation_matrix']):
        try:
            response_dofs = coordinate_array(
                [int(v) for v in file['channel_node_number'][file['response_indices']]],
                 file['channel_node_direction'][file['response_indices']])
        except Exception:
            response_dofs = coordinate_array(file['response_indices']+1,0)
    else:
        response_dofs = coordinate_array(np.arange(file['response_transformation_matrix'].shape[0])+1,0)
    if np.isnan(file['reference_transformation_matrix']):
        try:
            reference_dofs = coordinate_array(
                [int(v) for v in file['channel_node_number'][file['reference_indices']]],
                 file['channel_node_direction'][file['reference_indices']])
        except Exception:
            reference_dofs = coordinate_array(file['reference_indices']+1,0)
    else:
        reference_dofs = coordinate_array(np.arange(file['reference_transformation_matrix'].shape[0])+1,0)
    ordinate = np.moveaxis(file['frf_data'],0,-1)
    frfs = data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                      df*np.arange(ordinate.shape[-1]),ordinate,
                      outer_product(response_dofs,reference_dofs))
    ordinate = np.moveaxis(file['response_cpsd'],0,-1)
    response_cpsd = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                      df*np.arange(ordinate.shape[-1]),ordinate,
                      outer_product(response_dofs,response_dofs))
    ordinate = np.moveaxis(file['reference_cpsd'],0,-1)
    reference_cpsd = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                      df*np.arange(ordinate.shape[-1]),ordinate,
                      outer_product(reference_dofs,reference_dofs))
    ordinate = np.moveaxis(file['response_noise_cpsd'],0,-1)
    response_noise_cpsd = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                      df*np.arange(ordinate.shape[-1]),ordinate,
                      outer_product(response_dofs,response_dofs))
    ordinate = np.moveaxis(file['reference_noise_cpsd'],0,-1)
    reference_noise_cpsd = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                      df*np.arange(ordinate.shape[-1]),ordinate,
                      outer_product(reference_dofs,reference_dofs))
    ordinate = np.moveaxis(file['coherence'],0,-1)
    coherence = data_array(FunctionTypes.MULTIPLE_COHERENCE,
                      df*np.arange(ordinate.shape[-1]),ordinate,
                      outer_product(response_dofs))
    return frfs,response_cpsd,reference_cpsd,response_noise_cpsd,reference_noise_cpsd,coherence
    

def read_modal_data(file, coordinate_override_column=None, read_only_indices = None):
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
    output_data = np.array(ds['time_data'][read_only_indices]).reshape(num_averages,num_channels,samples_per_frame)
    abscissa = np.arange(samples_per_frame) / sample_rate
    if coordinate_override_column is None:
        nodes = [int(''.join(char for char in node if char in '0123456789'))
                 for node in ds['channels']['node_number'][read_only_indices]]
        directions = np.array(ds['channels']['node_direction'][read_only_indices], dtype='<U3')
        coordinates = coordinate_array(nodes, directions)[:, np.newaxis]
    else:
        coordinates = coordinate_array(string_array=ds['channels'][coordinate_override_column][read_only_indices])[
            :, np.newaxis]
    array = {name: np.array(variable[:]) for name, variable in ds['channels'].variables.items()}
    channel_table = pd.DataFrame(array)
    comment1 = np.char.add(np.char.add(np.array(ds['channels']['channel_type'][read_only_indices], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['unit'][read_only_indices], dtype='<U80'))
    comment2 = np.char.add(np.char.add(np.array(ds['channels']['physical_device'][read_only_indices], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['physical_channel'][read_only_indices], dtype='<U80'))
    comment3 = np.char.add(np.char.add(np.array(ds['channels']['feedback_device'][read_only_indices], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['feedback_channel'][read_only_indices], dtype='<U80'))
    comment4 = np.array(ds['channels']['comment'][read_only_indices], dtype='<U80')
    comment5 = np.array(ds['channels']['make'][read_only_indices], dtype='<U80')
    for key in ('model', 'serial_number', 'triax_dof'):
        comment5 = np.char.add(comment5, np.array(' '))
        comment5 = np.char.add(comment5, np.array(ds['channels'][key][read_only_indices], dtype='<U80'))
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
    reference_indices = np.array(ds.groups['Modal'].variables['reference_channel_indices'][:])
    response_indices = np.array([i for i in range(num_channels) if not i in reference_indices])
    keep_response_indices = np.array([i for i,index in enumerate(response_indices) if index in kept_indices])
    keep_reference_indices = np.array([i for i,index in enumerate(reference_indices) if index in kept_indices])
    frequency_lines = np.arange(ds.dimensions['fft_lines'].size)*sample_rate/samples_per_frame
    coherence_data = np.array(ds['coherence'][:,keep_response_indices]).T
    comment1 = np.char.add(np.char.add(np.array(ds['channels']['channel_type'][response_indices[keep_response_indices]], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['unit'][response_indices[keep_response_indices]], dtype='<U80'))
    comment2 = np.char.add(np.char.add(np.array(ds['channels']['physical_device'][response_indices[keep_response_indices]], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['physical_channel'][response_indices[keep_response_indices]], dtype='<U80'))
    comment3 = np.char.add(np.char.add(np.array(ds['channels']['feedback_device'][response_indices[keep_response_indices]], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['feedback_channel'][response_indices[keep_response_indices]], dtype='<U80'))
    comment4 = np.array(ds['channels']['comment'][response_indices[keep_response_indices]], dtype='<U80')
    comment5 = np.array(ds['channels']['make'][response_indices[keep_response_indices]], dtype='<U80')
    for key in ('model', 'serial_number', 'triax_dof'):
        comment5 = np.char.add(comment5, np.array(' '))
        comment5 = np.char.add(comment5, np.array(ds['channels'][key][response_indices[keep_response_indices]], dtype='<U80'))
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
    frf_data = np.moveaxis(np.array(ds['frf_data_real'])[:,keep_response_indices[:,np.newaxis],keep_reference_indices]
                           +np.array(ds['frf_data_imag'])[:,keep_response_indices[:,np.newaxis],keep_reference_indices]*1j,0,-1)
    frf_coordinate = outer_product(coordinates[response_indices[keep_response_indices]].squeeze(),
                                   coordinates[reference_indices[keep_reference_indices]].squeeze())
    response_comment1 = np.char.add(np.char.add(np.array(ds['channels']['channel_type'][response_indices[keep_response_indices]], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['unit'][response_indices[keep_response_indices]], dtype='<U80'))
    response_comment2 = np.char.add(np.char.add(np.array(ds['channels']['physical_device'][response_indices[keep_response_indices]], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['physical_channel'][response_indices[keep_response_indices]], dtype='<U80'))
    response_comment3 = np.char.add(np.char.add(np.array(ds['channels']['feedback_device'][response_indices[keep_response_indices]], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['feedback_channel'][response_indices[keep_response_indices]], dtype='<U80'))
    response_comment4 = np.array(ds['channels']['comment'][response_indices[keep_response_indices]], dtype='<U80')
    response_comment5 = np.array(ds['channels']['make'][response_indices[keep_response_indices]], dtype='<U80')
    for key in ('model', 'serial_number', 'triax_dof'):
        response_comment5 = np.char.add(response_comment5, np.array(' '))
        response_comment5 = np.char.add(response_comment5, np.array(ds['channels'][key][response_indices[keep_response_indices]], dtype='<U80'))
    reference_comment1 = np.char.add(np.char.add(np.array(ds['channels']['channel_type'][reference_indices[keep_reference_indices]], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['unit'][reference_indices[keep_reference_indices]], dtype='<U80'))
    reference_comment2 = np.char.add(np.char.add(np.array(ds['channels']['physical_device'][reference_indices[keep_reference_indices]], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['physical_channel'][reference_indices[keep_reference_indices]], dtype='<U80'))
    reference_comment3 = np.char.add(np.char.add(np.array(ds['channels']['feedback_device'][reference_indices[keep_reference_indices]], dtype='<U80'),
                                       np.array(' :: ')),
                           np.array(ds['channels']['feedback_channel'][reference_indices[keep_reference_indices]], dtype='<U80'))
    reference_comment4 = np.array(ds['channels']['comment'][reference_indices[keep_reference_indices]], dtype='<U80')
    reference_comment5 = np.array(ds['channels']['make'][reference_indices[keep_reference_indices]], dtype='<U80')
    for key in ('model', 'serial_number', 'triax_dof'):
        reference_comment5 = np.char.add(reference_comment5, np.array(' '))
        reference_comment5 = np.char.add(reference_comment5, np.array(ds['channels'][key][reference_indices[keep_reference_indices]], dtype='<U80'))
    response_comment1,reference_comment1 = np.broadcast_arrays(response_comment1[:,np.newaxis],reference_comment1)
    comment1 = np.char.add(np.char.add(response_comment1,np.array(' / ')),reference_comment1)
    response_comment2,reference_comment2 = np.broadcast_arrays(response_comment2[:,np.newaxis],reference_comment2)
    comment2 = np.char.add(np.char.add(response_comment2,np.array(' / ')),reference_comment2)
    response_comment3,reference_comment3 = np.broadcast_arrays(response_comment3[:,np.newaxis],reference_comment3)
    comment3 = np.char.add(np.char.add(response_comment3,np.array(' / ')),reference_comment3)
    response_comment4,reference_comment4 = np.broadcast_arrays(response_comment4[:,np.newaxis],reference_comment4)
    comment4 = np.char.add(np.char.add(response_comment4,np.array(' / ')),reference_comment4)
    response_comment5,reference_comment5 = np.broadcast_arrays(response_comment5[:,np.newaxis],reference_comment5)
    comment5 = np.char.add(np.char.add(response_comment5,np.array(' / ')),reference_comment5)
    frf_data = data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                          frequency_lines,
                          frf_data,
                          frf_coordinate,
                          comment1,
                          comment2,
                          comment3,
                          comment4,
                          comment5)
    return time_data,frf_data,coherence_data,channel_table