# -*- coding: utf-8 -*-
"""
Defines a matrix that has helpful tools for bookkeeping.

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
from .sdynpy_array import SdynpyArray
from .sdynpy_coordinate import CoordinateArray, outer_product


class Matrix(SdynpyArray):
    """Matrix with degrees of freedom stored for better bookkeeping.

    Use the matrix helper function to create the object.
    """

    @staticmethod
    def data_dtype(rows, columns, is_complex=False):
        """
        Data type of the underlying numpy structured array for real shapes

        Parameters
        ----------
        rows : int
            Number of rows in the matrix
        columns : int
            Number of columns in the matrix

        Returns
        -------
        list
            Numpy dtype that can be passed into any of the numpy array
            constructors

        """
        return [
            ('matrix', 'complex128' if is_complex else 'float64', (rows, columns)),
            ('row_coordinate', CoordinateArray.data_dtype, (rows,)),
            ('column_coordinate', CoordinateArray.data_dtype, (columns,)),
        ]

    """Datatype for the underlying numpy structured array"""

    def __new__(subtype, shape, nrows, ncols, is_complex=False, buffer=None, offset=0,
                strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super(Matrix, subtype).__new__(subtype, shape,
                                             Matrix.data_dtype(nrows, ncols, is_complex),
                                             buffer, offset, strides,
                                             order)
        # Finally, we must return the newly created object:
        return obj

    @property
    def coordinate(self):
        """
        Returns the full coordinate array for the matrix
        """
        return outer_product(self.row_coordinate, self.column_coordinate)

    @coordinate.setter
    def coordinate(self, value):
        raise RuntimeError('Cannot set coordinate directly.  Set row_coordinate or column_coordinate instead.')

    @property
    def num_coordinate_rows(self):
        """
        Returns the number of coordinate rows
        """
        return self.matrix.shape[-2]

    @property
    def num_coordinate_columns(self):
        """
        Returns the number of coordinate columns
        """
        return self.matrix.shape[-1]

    def __setitem__(self, key, value):
        if type(key) is tuple and len(key) == 2 and any([(type(val) is CoordinateArray) for val in key]):
            # Get indices corresponding to the coordinates
            row_request, column_request = key
            if row_request is None or (isinstance(row_request, slice) and row_request == slice(None)) or isinstance(row_request, type(Ellipsis)):
                matrix_row_indices = np.arange(self.num_coordinate_rows)
                request_row_indices = np.arange(self.num_coordinate_rows)
                row_multiplications = np.ones(self.num_coordinate_rows)
            else:
                row_request = np.atleast_1d(row_request)
                column_request = np.atleast_1d(column_request)
                # Start with rows
                coordinate_array = self.row_coordinate
                single_matrix_coordinate_array = coordinate_array[
                    (0,) * (coordinate_array.ndim - 1) + (slice(None),)]
                # Now check if the coordinates are consistent across the arrays
                if not np.all((coordinate_array[..., :] == single_matrix_coordinate_array)):
                    # If they aren't, raise a value error
                    raise ValueError(
                        'Matrix must have equivalent row coordinates for all matrices to index by coordinate')
                consistent_row_coordinates, matrix_row_indices, request_row_indices = np.intersect1d(
                    abs(single_matrix_coordinate_array), abs(row_request), assume_unique=False, return_indices=True)
                if consistent_row_coordinates.size != row_request.size:
                    extra_keys = np.setdiff1d(abs(row_request), abs(single_matrix_coordinate_array))
                    if extra_keys.size == 0:
                        raise ValueError(
                            'Duplicate coordinate values requested.  Please ensure coordinate indices are unique.')
                    raise ValueError(
                        'Not all indices in requested coordinate array exist in the shape\n{:}'.format(str(extra_keys)))
                # Handle sign flipping
                row_multiplications = row_request.flatten()[request_row_indices].sign(
                    ) * single_matrix_coordinate_array[matrix_row_indices].sign()
                # # Invert the indices to return the dofs in the correct order as specified in keys
                # inverse_row_indices = np.zeros(matrix_row_indices.shape, dtype=int)
                # inverse_row_indices[matrix_row_indices] = np.arange(len(matrix_row_indices))
            if column_request is None or (isinstance(column_request, slice) and column_request == slice(None)) or isinstance(column_request, type(Ellipsis)):
                matrix_col_indices = np.arange(self.num_coordinate_columns)
                request_col_indices = np.arange(self.num_coordinate_columns)
                col_multiplications = np.ones(self.num_coordinate_columns)
            else:
                # Now columns
                coordinate_array = self.column_coordinate
                single_matrix_coordinate_array = coordinate_array[
                    (0,) * (coordinate_array.ndim - 1) + (slice(None),)]
                # Now check if the coordinates are consistent across the arrays
                if not np.all((coordinate_array[..., :] == single_matrix_coordinate_array)):
                    # If they aren't, raise a value error
                    raise ValueError(
                        'Matrix must have equivalent column coordinates for all matrices to index by coordinate')
                consistent_col_coordinates, matrix_col_indices, request_col_indices = np.intersect1d(
                    abs(single_matrix_coordinate_array), abs(column_request), assume_unique=False, return_indices=True)
                if consistent_col_coordinates.size != column_request.size:
                    extra_keys = np.setdiff1d(abs(column_request), abs(single_matrix_coordinate_array))
                    if extra_keys.size == 0:
                        raise ValueError(
                            'Duplicate coordinate values requested.  Please ensure coordinate indices are unique.')
                    raise ValueError(
                        'Not all indices in requested coordinate array exist in the shape\n{:}'.format(str(extra_keys)))
                # Handle sign flipping
                col_multiplications = column_request.flatten()[request_col_indices].sign(
                    ) * single_matrix_coordinate_array[matrix_col_indices].sign()
                # # Invert the indices to return the dofs in the correct order as specified in keys
                # inverse_col_indices = np.zeros(matrix_col_indices.shape, dtype=int)
                # inverse_col_indices[matrix_col_indices] = np.arange(len(matrix_col_indices))
            value = np.array(value)
            value = np.broadcast_to(value,
                                    value.shape[:-2] + (len(consistent_row_coordinates),
                                                        len(consistent_col_coordinates)))
            self.matrix[..., matrix_row_indices[:, np.newaxis], matrix_col_indices] = (
                value[..., request_row_indices[:, np.newaxis], request_col_indices]
                * row_multiplications[:, np.newaxis] * col_multiplications)
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        if type(key) is tuple and len(key) == 2 and any([(type(val) is CoordinateArray) for val in key]):
            # Get indices corresponding to the coordinates
            row_request, column_request = key
            if row_request is None or (isinstance(row_request, slice) and row_request == slice(None)) or isinstance(row_request, type(Ellipsis)):
                matrix_row_indices = np.arange(self.num_coordinate_rows)
                inverse_row_indices = np.arange(self.num_coordinate_rows)
                row_multiplications = np.ones(self.num_coordinate_rows)
            else:
                row_request = np.atleast_1d(row_request)
                column_request = np.atleast_1d(column_request)
                # Start with rows
                coordinate_array = self.row_coordinate
                single_matrix_coordinate_array = coordinate_array[
                    (0,) * (coordinate_array.ndim - 1) + (slice(None),)]
                # Now check if the coordinates are consistent across the arrays
                if not np.all((coordinate_array[..., :] == single_matrix_coordinate_array)):
                    # If they aren't, raise a value error
                    raise ValueError(
                        'Matrix must have equivalent row coordinates for all matrices to index by coordinate')
                consistent_row_coordinates, matrix_row_indices, request_row_indices = np.intersect1d(
                    abs(single_matrix_coordinate_array), abs(row_request), assume_unique=False, return_indices=True)
                if consistent_row_coordinates.size != row_request.size:
                    extra_keys = np.setdiff1d(abs(row_request), abs(single_matrix_coordinate_array))
                    if extra_keys.size == 0:
                        raise ValueError(
                            'Duplicate coordinate values requested.  Please ensure coordinate indices are unique.')
                    raise ValueError(
                        'Not all indices in requested coordinate array exist in the shape\n{:}'.format(str(extra_keys)))
                # Handle sign flipping
                row_multiplications = row_request.flatten()[request_row_indices].sign(
                    ) * single_matrix_coordinate_array[matrix_row_indices].sign()
                # Invert the indices to return the dofs in the correct order as specified in keys
                inverse_row_indices = np.zeros(request_row_indices.shape, dtype=int)
                inverse_row_indices[request_row_indices] = np.arange(len(request_row_indices))
            if column_request is None or (isinstance(column_request, slice) and column_request == slice(None)) or isinstance(column_request, type(Ellipsis)):
                matrix_col_indices = np.arange(self.num_coordinate_columns)
                inverse_col_indices = np.arange(self.num_coordinate_columns)
                col_multiplications = np.ones(self.num_coordinate_columns)
            else:
                # Now columns
                coordinate_array = self.column_coordinate
                single_matrix_coordinate_array = coordinate_array[
                    (0,) * (coordinate_array.ndim - 1) + (slice(None),)]
                # Now check if the coordinates are consistent across the arrays
                if not np.all((coordinate_array[..., :] == single_matrix_coordinate_array)):
                    # If they aren't, raise a value error
                    raise ValueError(
                        'Matrix must have equivalent column coordinates for all matrices to index by coordinate')
                consistent_col_coordinates, matrix_col_indices, request_col_indices = np.intersect1d(
                    abs(single_matrix_coordinate_array), abs(column_request), assume_unique=False, return_indices=True)
                if consistent_col_coordinates.size != column_request.size:
                    extra_keys = np.setdiff1d(abs(column_request), abs(single_matrix_coordinate_array))
                    if extra_keys.size == 0:
                        raise ValueError(
                            'Duplicate coordinate values requested.  Please ensure coordinate indices are unique.')
                    raise ValueError(
                        'Not all indices in requested coordinate array exist in the shape\n{:}'.format(str(extra_keys)))
                # Handle sign flipping
                col_multiplications = column_request.flatten()[request_col_indices].sign(
                    ) * single_matrix_coordinate_array[matrix_col_indices].sign()
                # Invert the indices to return the dofs in the correct order as specified in keys
                inverse_col_indices = np.zeros(request_col_indices.shape, dtype=int)
                inverse_col_indices[request_col_indices] = np.arange(len(request_col_indices))
            return_value = self.matrix[..., matrix_row_indices[:, np.newaxis], matrix_col_indices] * row_multiplications[:, np.newaxis] * col_multiplications
            return_value = return_value[..., inverse_row_indices[:, np.newaxis], inverse_col_indices]
            return return_value
        else:
            output = super().__getitem__(key)
            if isinstance(key, str) and key in ['row_coordinate', 'column_coordinate']:
                return output.view(CoordinateArray)
            else:
                return output

    def __repr__(self):
        return '\n\n'.join(['matrix = \n'+repr(self.matrix),
                            'row coordinates = \n'+repr(self.row_coordinate),
                            'column coordinates = \n'+repr(self.column_coordinate)])

    def argsort_coordinate(self):
        """
        Returns indices used to sort the coordinates on the rows and columns

        Returns
        -------
        row_indices
            Indices used to sort the row coordinates.
        column_indices
            Indices used to sort the column coordinates

        """
        row_indices = np.empty(self.row_coordinate.shape, dtype=int)
        column_indices = np.empty(self.column_coordinate.shape, dtype=int)
        for key, matrix in self.ndenumerate():
            row_indices[key] = np.argsort(self.row_coordinate[key])
            column_indices[key] = np.argsort(self.column_coordinate[key])
        return row_indices[..., :, np.newaxis], column_indices[..., np.newaxis, :]

    def sort_coordinate(self):
        """
        Returns a copy of the Matrix with coordinate sorted

        Returns
        -------
        return_val : Matrix
            Matrix with row and column coordinates sorted.

        """
        sorting = self.argsort_coordinate()
        return_val = self.copy()
        for key, matrix in self.ndenumerate():
            return_val.matrix[key] = self.matrix[key][..., sorting[0][key], sorting[1][key]]
            return_val.row_coordinate = self.row_coordinate[..., sorting[0][key][:, 0]]
            return_val.column_coordinate = self.column_coordinate[..., sorting[1][key][0, :]]
        return return_val

    def plot(self, ax=None, show_colorbar=True, **imshow_args):
        """
        Plots the matrix with coordinates labeled

        Parameters
        ----------
        ax : axis, optional
            Axis on which the matrix will be plotted.  If not specified, a new
            figure will be created.
        show_colorbar : bool, optional
            If true, show a colorbar.  Default is true.
        **imshow_args : various
            Additional arguments passed to imshow

        Raises
        ------
        ValueError
            Raised if matrix is multi-dimensional.

        """
        if self.size > 1:
            raise ValueError('Cannot plot more than one Matrix object simultaneously')
        if ax is None:
            fig, ax = plt.subplots()

        @ticker.FuncFormatter
        def row_formatter(x, pos):
            if int(x) < 0 or int(x) >= len(self.row_coordinate):
                return ''
            try:
                return str(self.row_coordinate[int(x)])
            except IndexError:
                return ''

        @ticker.FuncFormatter
        def col_formatter(x, pos):
            if int(x) < 0 or int(x) >= len(self.column_coordinate):
                return ''
            try:
                return str(self.column_coordinate[int(x)])
            except IndexError:
                return ''
        out = ax.imshow(self.matrix, **imshow_args)
        if show_colorbar:
            plt.colorbar(out, ax=ax)
        ax.xaxis.set_major_formatter(col_formatter)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(row_formatter)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    @classmethod
    def eye(cls, row_coordinate, column_coordinate=None):
        """
        Creates an identity matrix with the specified coordinates

        Parameters
        ----------
        row_coordinate : CoordinateArray
            Coordinate array to use as the row coordinates
        column_coordinate : CoordinateArray, optional
            Coordinate array to use as the column coordinates.  If not specified,
            the row coordinates are used.

        Returns
        -------
        Matrix
            Diagonal matrix with the specified coordinates

        """
        if column_coordinate is None:
            column_coordinate = row_coordinate
        return matrix(np.eye(row_coordinate.shape[-1], column_coordinate.shape[-1]),
                      row_coordinate, column_coordinate)
    
    def pinv(self, **pinv_params):
        """
        Creates the pseudoinverse of the matrix

        Parameters
        ----------
        **pinv_params : various
            Extra keyword arguments are passed directly to np.linalg.pinv

        Returns
        -------
        Matrix
            A matrix consisting of the pseudoinverse of the original matrix

        """
        mat = np.linalg.pinv(self.matrix,**pinv_params)
        return matrix(matrix=mat,row_coordinate = self.column_coordinate,
                      column_coordinate = self.row_coordinate)
    
    def __matmul__(self,other):
        if not isinstance(other,Matrix):
            # If it is not another matrix, rely on the other data to define what
            # matrix multiplication means
            return NotImplemented
        return super().__matmul__(other)


def matrix(matrix, row_coordinate, column_coordinate, buffer=None, offset=0,
           strides=None, order=None):
    """
    Create a matrix object

    Parameters
    ----------
    matrix : ndarray
        The values in the matrix object
    row_coordinate : CoordinateArray
        Coordinates to assign to the rows of the matrix
    column_coordinate : CoordinateArray
        Coordinates to assign to the columns of the matrix

    Returns
    -------
    matrix : Matrix
        Matrix object

    """
    shape = matrix.shape[:-2]
    nrow, ncol = matrix.shape[-2:]
    m = Matrix(shape, nrow, ncol, np.iscomplexobj(matrix),
               buffer=buffer, offset=offset,
               strides=strides, order=order)
    m.row_coordinate = row_coordinate
    m.column_coordinate = column_coordinate
    m.matrix = matrix
    return m
