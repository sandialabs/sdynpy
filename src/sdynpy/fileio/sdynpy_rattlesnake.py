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
from ..core.sdynpy_coordinate import coordinate_array
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
