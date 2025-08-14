# -*- coding: utf-8 -*-
"""
Functions for dealing with complex numbers

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
import matplotlib.pyplot as plt


def collapse_complex_to_real(vector, axis=-1, preserve_magnitude=False, plot=False,
                             force_angle=None):
    x = np.real(vector)
    y = np.imag(vector)
    slope = np.sum(x * y, axis=axis, keepdims=True) / np.sum(x * x, axis=axis, keepdims=True)
    angle = np.arctan(slope)
    if force_angle is not None:
        angle[...] = force_angle
    rotated_vector = vector * np.exp(-1j * angle)
    if plot:
        plt.figure('Complex Rotation')
        shape = list(vector.shape)
        shape[axis] = 1
        for key in np.ndindex(*shape):
            print(key)
            index = list(key)
            index[axis] = slice(None)
            this_vector = vector[index]
            this_rotated_vector = rotated_vector[index]
            plt.plot(np.real(this_vector), np.imag(this_vector), 'x')
            plt.plot(np.real(this_rotated_vector), np.imag(this_rotated_vector), 'o')
    if preserve_magnitude:
        return np.sign(np.real(rotated_vector)) * np.abs(rotated_vector)
    else:
        return np.real(rotated_vector)
    
def fit_complex_angle(vector,axis=-1):
    x = np.real(vector)
    y = np.imag(vector)
    slope = np.sum(x * y, axis=axis, keepdims=True) / np.sum(x * x, axis=axis, keepdims=True)
    angle = np.arctan(slope)
    return angle
    
def rotate_vector(vector,angle):
    return vector * np.exp(1j * angle)
