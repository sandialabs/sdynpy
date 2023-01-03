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


def read_rattlesnake_output(file, coordinate_override_column=None):
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
    output_data = np.array(ds['time_data'][:])
    abscissa = np.arange(output_data.shape[-1]) / ds.sample_rate
    if coordinate_override_column is None:
        nodes = [int(''.join(char for char in node if char in '0123456789'))
                 for node in ds['channels']['node_number'][:]]
        directions = np.array(ds['channels']['node_direction'][:], dtype='<U3')
        coordinates = coordinate_array(nodes, directions)[:, np.newaxis]
    else:
        coordinates = coordinate_array(string_array=ds['channels'][coordinate_override_column][:])[
            :, np.newaxis]
    array = {name: np.array(variable[:]) for name, variable in ds['channels'].variables.items()}
    channel_table = pd.DataFrame(array)
    comment1 = np.array(ds['channels']['channel_type'][:], dtype='<U80')
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
    