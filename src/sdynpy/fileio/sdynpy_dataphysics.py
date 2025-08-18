# -*- coding: utf-8 -*-
"""
Load in time data from dataphysics runs.

Using the functions in this module, one can read mat v7.3 files written from DataPhysics.

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

import h5py
import numpy as np
import os
from ..core.sdynpy_data import coordinate_array, TimeHistoryArray, CoordinateArray, FunctionTypes, data_array
from scipy.io import loadmat

def read_dataphysics_output(file: os.PathLike | h5py.File | dict, coordinate: CoordinateArray | None = None):
    """
    Reads a Data Physics .mat or .uff file and converts it into a SDYNPY timehistory array object.

    Parameters
    ----------
    file : os.PathLike | h5py.File | dict
        Path to a .mat file (can be legacy .mat or v7.3) or a .uff file
        Alternatively, can be an `h5py.File` object or a `dict` created using `scipy.io.loadmat`

    coordinate : CoordinateArray, optional
        sdynpy coordinate array. If not provided, will attempt to determine coordinate 
        nodes/directions using `ChanName` from the .mat file or `comment4` from the .uff/.unv file.

    Returns
    -------
    TimeHistoryArray
        Time history array object.

    """
    UFF = False
    MAT = False
    MAT73 = False
    ext = None
    if isinstance(file, str):
        # Get File Extension
        _, ext = os.path.splitext(file)
        ext = ext.lower()
    elif isinstance(file, dict):
        data = file
        MAT = True
    elif isinstance(file, h5py.File):
        MAT = True
        MAT73 = True

    if ext == '.mat':
        try:
            data = loadmat(file)
            MAT = True
        except NotImplementedError:            
            file = h5py.File(file,'r')
            MAT = True
            MAT73 = True
    elif ext == '.uff':
        UFF = True
    
    
    def dec_to_strings(array):
        """
        Convert numpy array of decimal characters into a numpy array of strings.

        Parameters
        ----------
        array : ndarray
            Numpy array of decimal character codes

        Returns
        -------
        array : ndarray
            Numpy array of strings with one string for each row of the original numpy array
        """
        
        # Convert Decimal Codes into Strings
        strings = np.array([''.join(map(chr, col)) for col in array.T])
        return strings
    
    
    if MAT:
        if MAT73:
            # Create Dictionary Structure
            data = {
                'ChanName':dec_to_strings(np.array(file['ChanName'])), # Numpy Array (# Channels)
                'EUType':np.array(file['EUType']).T, # Numpy Array (# Channels x 1)
                'InputChanNums':np.array(file['InputChanNums']).T, # Numpy Array (# Channels x 1)
                'Sensitivity':np.array(file['Sensitivity']).T, # Numpy Array (# Channels x 1)
                'Unit':dec_to_strings(np.array(file['Unit'])), # Numpy Array (# Channels)
                'hDelta':np.array(file['hDelta']), # Numpy Array (1 x 1)
                'TimeData':np.array(file['TimeData']).T, # Numpy Array (# Abscissa x # Channels)
                }
        
        if coordinate is None:
            coordinate = coordinate_array(string_array=data['ChanName'])
        ordinate = np.array(data['TimeData']).swapaxes(0,1)
        abscissa_spacing = data['hDelta'][0,0]
        abscissa = np.linspace(0, ordinate.shape[-1]*abscissa_spacing, num=ordinate.shape[-1])
        comment1 = 'Data Physics Corporation'
        comment2 = ''
        comment3 = ''
        comment4 = data['ChanName']
        comment5 = data['Sensitivity'][:,0]
        return data_array(FunctionTypes.TIME_RESPONSE, abscissa, ordinate, coordinate[:,np.newaxis],comment1=comment1, comment2=comment2, comment3=comment3, comment4=comment4, comment5=comment5)
        
    elif UFF:
        data = TimeHistoryArray.load(file)
        if coordinate is None:
            coordinate = coordinate_array(string_array=data.comment4)
        data.coordinate[:, 0] = coordinate
        return data
    
    else:
        raise ValueError('Invalid file type')