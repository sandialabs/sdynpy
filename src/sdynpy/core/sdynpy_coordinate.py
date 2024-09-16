"""
Defines the CoordinateArray, which specifies arrays of node numbers and directions.

Coordinates in SDynPy are used to define degrees of freedom.  These consist of
a node number (which corresponds to a node in a SDynPy Geometry object) and a
direction (which corresponds to the local displacement coordinate system of
that node in the SDynPy Geometry object).  Directions are the translations or
rotations about the principal axis, and can be positive or negative.  The
direction can also be empty for non-directional data.

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
from .sdynpy_array import SdynpyArray
import warnings

# This maps direction integers to their string counterparts
_string_map = {1: 'X+', 2: 'Y+', 3: 'Z+', 4: 'RX+', 5: 'RY+', 6: 'RZ+',
               -1: 'X-', -2: 'Y-', -3: 'Z-', -4: 'RX-', -5: 'RY-', -6: 'RZ-',
               0: ''}

# This maps direction strings to their integer counterparts
_direction_map = {'X+': 1, 'X': 1, '+X': 1,
                  'Y+': 2, 'Y': 2, '+Y': 2,
                  'Z+': 3, 'Z': 3, '+Z': 3,
                  'RX+': 4, 'RX': 4, '+RX': 4,
                  'RY+': 5, 'RY': 5, '+RY': 5,
                  'RZ+': 6, 'RZ': 6, '+RZ': 6,
                  'X-': -1, '-X': -1,
                  'Y-': -2, '-Y': -2,
                  'Z-': -3, '-Z': -3,
                  'RX-': -4, '-RX': -4,
                  'RY-': -5, '-RY': -5,
                  'RZ-': -6, '-RZ': -6,
                  '': 0}

_map_direction_string_array = np.vectorize(_string_map.get)
_map_direction_array = np.vectorize(_direction_map.get)


def parse_coordinate_string(coordinate: str):
    """
    Parse coordinate string into node and direction integers.

    Parameters
    ----------
    coordinate : str
        String representation of a coordinate, e.g. '101X+'

    Returns
    -------
    node : int
        Integer representing the node number
    direction : int
        Integer representing the direction, 'X+' = 1, 'Y+' = 2, 'Z+' = 3,
        'X-' = -1, 'Y-' = -2, 'Z-' = -3

    """
    try:
        node = int(''.join(v for v in coordinate if v in '0123456789'))
    except ValueError:
        warnings.warn('Node ID {:} is not Valid.  Defaulting to 0.'.format(''.join(v for v in coordinate if v in '0123456789')))
        node = 0
    try:
        direction = _direction_map[''.join(v for v in coordinate if not v in '0123456789')]
    except KeyError:
        warnings.warn('Direction {:} is not Valid.  Defaulting to no direction.'.format(''.join(v for v in coordinate if not v in '0123456789')))
        direction = 0
    return node, direction


parse_coordinate_string_array = np.vectorize(parse_coordinate_string, otypes=('int64', 'int64'))


def create_coordinate_string(node: int, direction: int):
    """
    Create a string from node and directions integers

    Parameters
    ----------
    node : int
        Node number
    direction : int
        Integer representing the direction, 'X+' = 1, 'Y+' = 2, 'Z+' = 3,
        'X-' = -1, 'Y-' = -2, 'Z-' = -3

    Returns
    -------
    str
        String representation of the coordinate, e.g. '101X+'

    """
    return str(node) + _string_map[direction]


create_coordinate_string_array = np.vectorize(create_coordinate_string, otypes=['<U4'])


class CoordinateArray(SdynpyArray):
    """Coordinate information specifying Degrees of Freedom (e.g. 101X+).

    Use the coordinate_array helper function to create the array.
    """

    data_dtype = [('node', 'uint64'), ('direction', 'int8')]
    """Datatype for the underlying numpy structured array"""

    def __new__(subtype, shape, buffer=None, offset=0,
                strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super(CoordinateArray, subtype).__new__(subtype, shape, CoordinateArray.data_dtype,
                                                      buffer, offset, strides,
                                                      order)
        # Finally, we must return the newly created object:
        return obj

    def string_array(self):
        """
        Returns a string array representation of the coordinate array

        Returns
        -------
        np.ndarray
            ndarray of strings representing the CoordinateArray
        """
        return create_coordinate_string_array(self.node, self.direction)

    def direction_string_array(self):
        """
        Returns a string array representation of the direction

        Returns
        -------
        np.ndarray
            ndarray of direction strings representing the CoordinateArray
        """
        return _map_direction_string_array(self.direction)

    def __repr__(self):
        return 'coordinate_array(string_array=\n' + repr(self.string_array()) + ')'

    def __str__(self):
        return str(self.string_array())

    def __eq__(self, value):
        value = np.array(value)
#        # A string
        if np.issubdtype(value.dtype, np.character):
            value = coordinate_array(string_array=value)
        if value.dtype.names is None:
            node_logical = self.node == value[..., 0]
            direction_logical = self.direction == value[..., 1]
        else:
            node_logical = self.node == value['node']
            direction_logical = self.direction == value['direction']
        return np.logical_and(node_logical, direction_logical)

    def __ne__(self, value):
        return ~self.__eq__(value)

    def __abs__(self):
        abs_coord = self.copy()
        abs_coord.direction = abs(abs_coord.direction)
        return abs_coord

    def __neg__(self):
        neg_coord = self.copy()
        neg_coord.direction = -neg_coord.direction
        return neg_coord

    def __pos__(self):
        pos_coord = self.copy()
        pos_coord.direction = +pos_coord.direction
        return pos_coord

    def abs(self):
        """Returns a coordinate array with direction signs flipped positive"""
        return self.__abs__()

    def sign(self):
        """Returns the sign on the directions of the CoordinateArray"""
        out = np.ones(self.shape)
        out[self.direction < 0] = -1
        return out

    def local_direction(self):
        """
        Returns a local direction array

        Returns
        -------
        local_direction_array : np.ndarray
            Returns a (...,3) array where ... is the dimension of the
            CoordinateArray.  The (...,0), (...,1), and (...,2) indices
            represent the x,y,z direction of the local coordinate direction.
            For example, a CoordinateArray with direction X- would return
            [-1,0,0].

        """
        output = np.zeros(self.shape + (3,))
        signs = self.sign()
        indices = abs(self.direction) - 1
        if self.ndim > 0:
            indices[indices > 2] -= 3
        else:
            if indices > 2:
                indices -= 3
        for key, index in np.ndenumerate(indices):
            if index != -1:
                output[key + (index,)] = signs[key]
        return output

    def offset_node_ids(self, offset_value):
        """
        Returns a copy of the CoordinateArray with the node IDs offset

        Parameters
        ----------
        offset_value : int
            The value to offset the node IDs by.

        Returns
        -------
        CoordinateArray

        """
        output = self.copy()
        output.node += offset_value
        return output

    @classmethod
    def from_matlab_cellstr(cls, cellstr_data):
        """
        Creates a CoordinateArray from a matlab cellstring object loaded from
        scipy.io.loadmat

        Parameters
        ----------
        cellstr_data : np.ndarray
            Dictionary entry corresponding to a cell string variable in a mat
            file loaded from scipy.io.loadmat

        Returns
        -------
        CoordinateArray
            CoordinateArray built from the provided cell string array

        """
        str_array = np.empty(cellstr_data.shape, dtype=object)
        for key, val in np.ndenumerate(cellstr_data):
            str_array[key] = val[0]
        return coordinate_array(string_array=str_array)

    @classmethod
    def from_nodelist(cls, nodes, directions=[1, 2, 3], flatten=True):
        """
        Returns a coordinate array with a set of nodes with a set of directions

        Parameters
        ----------
        nodes : iterable
            A list of nodes to create degrees of freedom at
        directions : iterable, optional
            A list of directions to create for each node. The default is [1,2,3],
            which provides the three positive translations (X+, Y+, Z+).
        flatten : bool, optional
            Specifies that the array should be flattened prior to output. The
            default is True.  If False, the output will have a dimension one
            larger than  the input node list due to the added direction
            dimension.

        Returns
        -------
        coordinate_array : CoordinateArray
            Array of coordinates with each specified direction defined at each
            node.  If flatten is false, this array will have shape
            nodes.shape + directions.shape.  Otherwise, this array will have
            shape (nodes.size*directions.size,)

        """
        ca = coordinate_array(np.array(nodes)[..., np.newaxis], np.array(directions))
        if flatten:
            return ca.flatten()
        else:
            return ca


def coordinate_array(node=None, direction=None,
                     structured_array=None,
                     string_array=None, force_broadcast=False):
    """
    Creates a coordinate array that specify degrees of freedom.

    Creates an array of coordinates that specify degrees of freedom in a test
    or analysis.  Coordinate arrays can be created using a numpy structured
    array or two arrays for node and direction.  Multidimensional arrays can
    be used.

    Parameters
    ----------
    node : ndarray
        Integer array corresponding to the node ids of the coordinates.  Input
        will be cast to an integer (i.e. 2.0 -> 2, 1.9 -> 1)
    direction : ndarray
        Direction corresponding to the coordinate.  If a string is passed, it
        must consist of a direction (RX, RY, RZ, X, Y, Z) and whether or not it
        is positive or negative (+ or -).  If no positive or negative value is
        given, then positive will be assumed.
    structured_array : ndarray (structured)
        Alternatively to node and direction, a single numpy structured array
        can be passed, which should have names ['node','direction']
    string_array : ndarray
        Alternatively to node and direction, a single numpy string array can
        be passed into the function, which will be parsed to create the
        data.
    force_broadcast : bool, optional
        Return all combinations of nodes and directions regardless of their
        shapes.  This will return a flattened array

    Returns
    -------
    coordinate_array : CoordinateArray

    """
    if structured_array is not None:
        try:
            node = structured_array['node']
            direction = structured_array['direction']
        except (ValueError, TypeError):
            raise ValueError(
                'structured_array must be numpy.ndarray with dtype names "node" and "direction"')
    elif string_array is not None:
        string_array = np.array(string_array)
        node, direction = parse_coordinate_string_array(string_array)
    else:
        node = np.array(node)
        direction = np.array(direction)
    if force_broadcast:
        node = np.unique(node)
        direction = np.unique(direction)
        bc_direction = np.tile(direction, node.size)
        bc_node = np.repeat(node, direction.size)
    else:
        try:
            bc_node, bc_direction = np.broadcast_arrays(node, direction)
        except ValueError:
            raise ValueError('node and direction should be broadcastable to the same shape (node: {:}, direction: {:})'.format(
                node.shape, direction.shape))

    # Create the coordinate array
    coord_array = CoordinateArray(bc_node.shape)
    coord_array.node = bc_node
    if not np.issubdtype(direction.dtype.type, np.integer):
        bc_direction = _map_direction_array(bc_direction)
    coord_array.direction = bc_direction

    return coord_array


def outer_product(*args):
    """
    Returns a CoordinateArray consisting of all combinations of the provided
    CoordinateArrays

    Parameters
    ----------
    *args : CoordinateArray
        CoordinateArrays to combine into a single CoordinateArray

    Returns
    -------
    CoordinateArray
        CoordinateArray consisting of combinations of provided CoordinateArrays

    """
    ndims = len(args) + 1
    expanded_coord_array = []
    for i, array in enumerate(args):
        index = tuple([Ellipsis]+[slice(None) if i == j else np.newaxis for j in range(ndims)])
        expanded_coord_array.append(array[index])
    return np.concatenate(np.broadcast_arrays(*expanded_coord_array), axis=-1)


def from_matlab_cellstr(cellstr_data):
    """
    Creates a CoordinateArray from a matlab cellstring object loaded from
    scipy.io.loadmat

    Parameters
    ----------
    cellstr_data : np.ndarray
        Dictionary entry corresponding to a cell string variable in a mat
        file loaded from scipy.io.loadmat

    Returns
    -------
    CoordinateArray
        CoordinateArray built from the provided cell string array

    """
    return CoordinateArray.from_matlab_cellstr(cellstr_data)


load = CoordinateArray.load
from_nodelist = CoordinateArray.from_nodelist
