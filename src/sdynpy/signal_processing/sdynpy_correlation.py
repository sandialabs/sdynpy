# -*- coding: utf-8 -*-
"""
Functions to compute correlation metrics between datasets

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
import matplotlib.ticker as ticker
import copy

# Look at FRAC and TRAC values

def mac(phi_1, phi_2=None):
    if phi_2 is None:
        phi_2 = phi_1
    mac = np.zeros([phi_1.shape[-1], phi_2.shape[-1]])
    for i, shape_1 in enumerate(phi_1.T):
        for j, shape_2 in enumerate(phi_2.T):
            mac[i, j] = np.abs(shape_1.T @ shape_2.conj())**2 / \
                ((shape_1.T @ shape_1.conj()) * (shape_2.T @ shape_2.conj()))
    return mac


def frac(fft_1, fft_2=None):
    if fft_2 is None:
        fft_2 = fft_1
    fft_1_original_shape = fft_1.shape
    fft_1_flattened = fft_1.reshape(-1, fft_1.shape[-1])
    fft_2_flattened = fft_2.reshape(-1, fft_2.shape[-1])
    frac = np.abs(np.sum(fft_1_flattened * fft_2_flattened.conj(), axis=-1))**2 / ((np.sum(fft_1_flattened *
                                                                                           fft_1_flattened.conj(), axis=-1)) * np.sum(fft_2_flattened * fft_2_flattened.conj(), axis=-1))
    return frac.reshape(fft_1_original_shape[:-1])


def trac(th_1, th_2=None):
    if th_2 is None:
        th_2 = th_1
    th_1_original_shape = th_1.shape
    th_1_flattened = th_1.reshape(-1, th_1.shape[-1])
    th_2_flattened = th_2.reshape(-1, th_2.shape[-1])
    trac = np.abs(np.sum(th_1_flattened * th_2_flattened.conj(), axis=-1))**2 / ((np.sum(th_1_flattened *
                                                                                         th_1_flattened.conj(), axis=-1)) * np.sum(th_2_flattened * th_2_flattened.conj(), axis=-1))
    return trac.reshape(th_1_original_shape[:-1])

def msf(shapes,reference_shapes = None):
    if reference_shapes is None:
        reference_shapes = shapes
    output = (np.einsum('...ij,...ij->...j',shapes,reference_shapes.conj())/
              np.einsum('...ij,...ij->...j',reference_shapes,reference_shapes.conj()))
    return output

def orthog(shapes_1,mass_matrix,shapes_2 = None,scaling = None):
    if not scaling in ['none','unity',None]:
        raise ValueError('Invalid scaling, should be one of "none", "unity", or None')
    if shapes_2 is None:
        shapes_2 = shapes_1
    mat = np.moveaxis(shapes_1,-2,-1) @ mass_matrix @ shapes_2
    if scaling == 'unity':
        diagonal = np.einsum('...ii->...i',mat)
        scaling = 1/np.sqrt(diagonal)
        scaling_matrix = np.zeros(mat.shape)
        scaling_matrix[...,
                       np.arange(scaling_matrix.shape[-2]),
                       np.arange(scaling_matrix.shape[-1])] = scaling
        mat = scaling_matrix @ mat @ scaling_matrix
    return mat
        
def matrix_plot(shape_matrix, ax=None, display_values=(0.1, 1.1), text_size=12, vmin=0, vmax=1,
                boundaries=None):
    if boundaries is None:
        # Display number not index
        @ticker.FuncFormatter
        def major_formatter(x, pos):
            return '{:0.0f}'.format(x + 1)
        cm = plt.get_cmap()
    else:
        # Add boundaries to the shape matrix
        boundaries = np.array(boundaries)
        shape_matrix_original = shape_matrix.copy()
        n_shapes = shape_matrix_original.shape[0]
        shape_matrix = np.nan * np.empty([v + len(boundaries) for v in shape_matrix_original.shape])
        index_map = {i + np.sum(boundaries <= i): i for i in np.arange(n_shapes)}
        inverse_index_map = {i: i + np.sum(boundaries <= i) for i in np.arange(n_shapes)}
        outputs = np.arange(n_shapes)
        inputs = np.array([inverse_index_map[i] for i in outputs])
        shape_matrix[
            inputs[:, np.newaxis], inputs
        ] = shape_matrix_original[
            outputs[:, np.newaxis], outputs]
        cm = copy.copy(plt.get_cmap())
        cm.set_bad(color='w')

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            x = int(np.round(x))
            if not x in index_map:
                return ''
            else:
                x = index_map[x]
                level = np.sum(x >= boundaries)
                if level == 0:
                    shape_index = x
                else:
                    shape_index = x - boundaries[level - 1]
                return '{:},{:}'.format(level + 1, shape_index + 1)
    if ax is None:
        fig, ax = plt.subplots()
    out = ax.imshow(shape_matrix, vmin=vmin, vmax=vmax, cmap=cm)
    plt.colorbar(out, ax=ax)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(major_formatter)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel('Shape Number')
    ax.set_ylabel('Shape Number')
    if not display_values is None:
        for key, val in np.ndenumerate(shape_matrix):
            if ((True if display_values[0] is None else (val > display_values[0]))
                and
                    (True if display_values[1] is None else (val <= display_values[1]))):
                ax.text(key[1], key[0], '{:0.0f}'.format(val * 100),
                        fontdict={'size': text_size}, ha='center', va='center')
    return ax
