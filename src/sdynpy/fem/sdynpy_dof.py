# -*- coding: utf-8 -*-
"""
Functions for optimally down-selecting degrees of freedom from a candidate set.

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
import time

UPDATE_TIME = 5


def by_condition_number(sensors_to_keep, shape_matrix, return_condition_numbers=False):
    '''Get the best set of degrees of freedom by mode shape condition number

    This function accepts a shape matrix and returns the set of degrees of
    freedom that corresponds to the lowest condition number.

    Parameters
    ----------
    sensors_to_keep : int
        The number of sensors to keep.
    shape_matrix : np.ndarray
        A 2D or 3D numpy array.  If it is a 2D array, the row indices should
        correspond to the degree of freedom and the column indices should
        correspond to the mode.  For a 3D array, the first index corresponds to
        a "bundle" of channels (e.g. a triaxial accelerometer) that must be
        kept or removed as one unit.
    return_condition_numbers : bool (default False)
        If True, return a second value that is the condition number at each
        iteration of the technique.

    Returns
    -------
    indices : np.array
        A 1d array corresponding to the indices to keep in the first dimension
        of the shape_matrix array (e.g. new_shape_matrix = 
        shape_matrix[incies,...])
    returned_condition_numbers : list
        The condition number at each iteration.  Returned only if
        return_condition_numbers is True.
    '''
    shape_matrix = shape_matrix.copy()
    keep_indices = np.arange(shape_matrix.shape[0])
    if return_condition_numbers:
        returned_condition_numbers = [np.linalg.cond(
            shape_matrix.reshape(-1, shape_matrix.shape[-1]))]
    start_time = time.time()
    while shape_matrix.shape[0] > sensors_to_keep:
        condition_numbers = [np.linalg.cond(shape_matrix[np.arange(shape_matrix.shape[0]) != removed_dof_index, ...].reshape(
            -1, shape_matrix.shape[-1])) for removed_dof_index in range(shape_matrix.shape[0])]
        dof_to_remove = np.argmin(condition_numbers)
#        print('Condition Numbers {:}'.format(condition_numbers))
#        print('Removing DOF {:}'.format(dof_to_remove))
        shape_matrix = np.delete(shape_matrix.copy(), dof_to_remove, axis=0)
        keep_indices = np.delete(keep_indices, dof_to_remove, axis=0)
        if return_condition_numbers:
            returned_condition_numbers.append(condition_numbers[dof_to_remove])
        new_time = time.time()
        if new_time - start_time > UPDATE_TIME:
            print('{:} DoFs Remaining'.format(shape_matrix.shape[0]))
            start_time += UPDATE_TIME
    if return_condition_numbers:
        return keep_indices, returned_condition_numbers
    else:
        return keep_indices


def by_effective_independence(sensors_to_keep, shape_matrix, return_efi=False):
    '''Get the best set of degrees of freedom by mode shape effective independence

    This function accepts a shape matrix and returns the set of degrees of
    freedom that corresponds to the maximum effective independence.

    Parameters
    ----------
    sensors_to_keep : int
        The number of sensors to keep.
    shape_matrix : np.ndarray
        A 2D or 3D numpy array.  If it is a 2D array, the row indices should
        correspond to the degree of freedom and the column indices should
        correspond to the mode.  For a 3D array, the first index corresponds to
        a "bundle" of channels (e.g. a triaxial accelerometer) that must be
        kept or removed as one unit.
    return_efi : bool (default False)
        If True, return a second value that is the effective independence at
        each iteration of the technique.

    Returns
    -------
    indices : np.array
        A 1d array corresponding to the indices to keep in the first dimension
        of the shape_matrix array (e.g. new_shape_matrix = 
        shape_matrix[incies,...])
    returned_efi : list
        The effective independence at each iteration.  Returned only if
        return_efi is set to True.
    '''
    shape_matrix = shape_matrix.copy()
    keep_indices = np.arange(shape_matrix.shape[0])
    if return_efi:
        returned_efi = []
    start_time = time.time()
    while shape_matrix.shape[0] > sensors_to_keep:
        Q = shape_matrix.reshape(-1, shape_matrix.shape[-1]
                                 ).T @ shape_matrix.reshape(-1, shape_matrix.shape[-1])
        if return_efi:
            returned_efi.append(np.linalg.det(Q))
        Qinv = np.linalg.inv(Q)
        if shape_matrix.ndim == 2:
            EfIs = np.diag(shape_matrix @ Qinv @ shape_matrix.T)
        else:
            EfIs = 1 - np.linalg.det(np.eye(3) - np.einsum('ijk,kl,iml->ijm',
                                     shape_matrix, Qinv, shape_matrix))
        dof_to_remove = np.argmin(EfIs)
#        print('Effective Independences {:}'.format(EfI3s))
#        print('Removing DOF {:}'.format(dof_to_remove))
        shape_matrix = np.delete(shape_matrix.copy(), dof_to_remove, axis=0)
        keep_indices = np.delete(keep_indices, dof_to_remove, axis=0)
        new_time = time.time()
        if new_time - start_time > UPDATE_TIME:
            print('{:} DoFs Remaining'.format(shape_matrix.shape[0]))
            start_time += UPDATE_TIME
    if return_efi:
        return keep_indices, returned_efi
    else:
        return keep_indices
