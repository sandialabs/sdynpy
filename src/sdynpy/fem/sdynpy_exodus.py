# -*- coding: utf-8 -*-
"""
Import data from and save data to Exodus files.

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

import netCDF4
import numpy as np
from types import SimpleNamespace
import datetime
import copy
from ..core.sdynpy_geometry import global_coord, local_coord, global_deflection, _exodus_elem_type_map

_inverse_exodus_elem_type_map = {val: key for key, val in _exodus_elem_type_map.items()}

__version__ = '0.99'


def face_connectivity(reduced_element_type, keep_midside_nodes):
    if reduced_element_type.lower() == 'tetra10' and keep_midside_nodes:
        return (('tri6', np.array([[0, 1, 3, 4, 8, 7],
                                   [1, 2, 3, 5, 9, 8],
                                   [2, 0, 3, 6, 7, 9],
                                   [0, 2, 1, 6, 5, 4]])),)
    elif reduced_element_type.lower() == 'tetra4' or (reduced_element_type == 'tetra10' and not keep_midside_nodes):
        return (('tri3', np.array([[0, 1, 3],
                                   [1, 2, 3],
                                   [2, 0, 3],
                                   [0, 2, 1]])),)
    elif reduced_element_type.lower() == 'hex20' and keep_midside_nodes:
        return (('quad8', np.array([[0, 1, 5, 4, 8, 13, 16, 12],
                                   [1, 2, 6, 5, 9, 14, 17, 13],
                                   [2, 3, 7, 6, 10, 15, 18, 14],
                                   [3, 0, 4, 7, 11, 12, 19, 15],
                                   [3, 2, 1, 0, 11, 8, 9, 10],
                                   [4, 5, 6, 7, 16, 17, 18, 19]])),)
    elif reduced_element_type.lower() == 'hex8' or reduced_element_type.lower() == 'hex' or (reduced_element_type == 'hex20' and not keep_midside_nodes):
        return (('quad4', np.array([[0, 1, 5, 4],
                                    [1, 2, 6, 5],
                                    [2, 3, 7, 6],
                                    [3, 0, 4, 7],
                                    [3, 2, 1, 0],
                                    [4, 5, 6, 7]])),)
    else:
        return ((None, None),)


def mesh_triangulation_array(element_type):
    if element_type == 'quad4':
        triangles = np.array([[0, 1, 2],
                              [0, 2, 3]])
    elif element_type == 'quad8':
        triangles = np.array([[0, 4, 7],
                              [4, 1, 5],
                              [5, 2, 6],
                              [7, 6, 3],
                              [4, 6, 7],
                              [4, 5, 6]])
    else:
        triangles = None
    return triangles


class ExodusError(Exception):
    pass


class Exodus:
    '''Read or write exodus files.

    This class creates functionality to read or write exodus files using the
    netCDF4 python module.

    Parameters
    ----------
    filename : str
        The path string to the file that will be opened
    mode : str
        Mode with which the file is opened, 'r' - Read, 'w' - Write,
        'a' - Append.  Default is 'r'.
    title : str
        Title of the exodus file, only required if mode='w'.
    num_dims : int
        Number of dimensions in the exodus file, only required if mode='w'.
    num_nodes : int
        Number of nodes in the exodus file, only required if mode='w'.
    num_elem : int
        Number of elements in the exodus file, only required if mode='w'.
    num_blocks : int
        Number of blocks in the exodus file, only required if mode='w'.
    num_node_sets : int
        Number of node sets in the exodus file, only required if mode = 'w'.
    num_side_sets : int
        Number of side sets in the exodus file, only required if mode='w'.
    '''

    def __init__(self, filename, mode='r',
                 title=None, num_dims=None, num_nodes=None, num_elem=None,
                 num_blocks=None, num_node_sets=None, num_side_sets=None,
                 clobber=False):
        self.filename = filename
        self.mode_character = mode
        if mode in ['r', 'a']:
            # Here we read in a file that already exists
            self._ncdf_handle = netCDF4.Dataset(filename, mode)
        else:
            # Here we have to create one first
            self._ncdf_handle = netCDF4.Dataset(filename, mode, clobber=clobber,
                                                format='NETCDF3_64BIT_OFFSET')
            # Assign attributes
            self._ncdf_handle.api_version = 5.22
            self._ncdf_handle.version = 5.22
            self._ncdf_handle.floating_point_word_size = 8
            self._ncdf_handle.file_size = 1
            self._ncdf_handle.int64_status = 0
            if title is None:
                title = ''
            self._ncdf_handle.title = title

            # Assign Dimensions
            self._ncdf_handle.createDimension('len_string', 33)
            self._ncdf_handle.createDimension('len_name', 33)
            self._ncdf_handle.createDimension('len_line', 81)
            self._ncdf_handle.createDimension('four', 4)
            if num_dims is None:
                raise ValueError("num_dims must be assigned for mode 'w'")
            self._ncdf_handle.createDimension('num_dim', num_dims)
            if num_nodes is None:
                raise ValueError("num_nodes must be assigned for mode 'w'")
            self._ncdf_handle.createDimension('num_nodes', num_nodes)
            if num_elem is None:
                raise ValueError("num_elem must be assigned for mode 'w'")
            self._ncdf_handle.createDimension('num_elem', num_elem)
            if num_blocks is None:
                raise ValueError("num_blocks must be assigned for mode 'w'")
            self._ncdf_handle.createDimension('num_el_blk', num_blocks)
            if num_node_sets is None:
                raise ValueError("num_node_sets must be assigned for mode 'w'")
            if num_node_sets > 0:
                self._ncdf_handle.createDimension('num_node_sets', num_node_sets)
            if num_side_sets is None:
                raise ValueError("num_side_sets must be assigned for mode 'w'")
            if num_side_sets > 0:
                self._ncdf_handle.createDimension('num_side_sets', num_side_sets)

            # Create convenience variables
            if self._ncdf_handle.floating_point_word_size == 8:
                self.floating_point_type = 'f8'
            else:
                self.floating_point_type = 'f4'

            # Initialize time
            # The time dimension is expandable in the Exodus format so we will
            # set the size of the dimension to none, which means unlimited.
            self._ncdf_handle.createDimension('time_step', None)
            # Initialize the time variable
            self._ncdf_handle.createVariable('time_whole', self.floating_point_type,
                                             ('time_step',))

    @property
    def title(self):
        '''Get the title of the exodus file'''
        return self._ncdf_handle.title

    def get_qa_records(self):
        '''Get the quality assurance records in the exodus file

        Returns
        -------
        qa_records : tuple
            Returns a nested tuple of strings where the first index is the
            record number, and the second is the line in the record.

        '''
        try:
            raw_records = self._ncdf_handle.variables['qa_records'][:]
        except KeyError:
            raise ExodusError('QA Records are not yet defined!')
        qa_records = tuple(tuple(''.join(value.decode() for value in line)
                           for line in record) for record in raw_records.data)
        return qa_records

    def put_qa_records(self, records):
        '''Puts the quality assurance records in the exodus file

        Parameters
        ----------
        qa_records : sequence of sequence of strings
            A nested sequence (list/tuple/etc.) containing the quality assurance
            records.


        Notes
        -----
        Each index in qa_records should consist of a length-4 tuple of strings:

        1. Analysis Code Name
        2. Analysis Code Version
        3. Analysis Date
        4. Analysis Time
        '''
        if 'num_qa_rec' in self._ncdf_handle.dimensions:
            raise ExodusError('QA Records have already been put to the exodus file')
        # Make sure that it is the correct shape
        num_qa_rec = len(records)
        len_string = self._ncdf_handle.dimensions['len_string'].size
        array = np.zeros((num_qa_rec, 4, len_string), dtype='|S1')
        for i, record in enumerate(records):
            if len(record) > 4:
                raise ValueError('QA Records can only have 4 entries per record!')
            for j, line in enumerate(record):
                if len(line) > len_string:
                    print('Warning:When setting QA records, entry "{}" (record {} entry {}) will be truncated'.format(
                        line, i, j))
                    line = line[:len_string]
                listed_string = [val for i, val in enumerate(line)]
                array[i, j, :len(listed_string)] = listed_string
        # Create the dimension
        self._ncdf_handle.createDimension('num_qa_rec', num_qa_rec)
        # Create the variable
        self._ncdf_handle.createVariable('qa_records', 'S1', ('num_qa_rec', 'four', 'len_string'))
        # Assign to it
        self._ncdf_handle.variables['qa_records'][:] = array[:]

    def get_info_records(self):
        '''Get the information records in the exodus file

        Returns
        -------
        info_records : tuple
            Returns a tuple of strings where the index is the
            record number.
        '''
        try:
            raw_records = self._ncdf_handle.variables['info_records'][:]
        except KeyError:
            raise ExodusError('Information Records are not yet defined!')
        info_records = tuple(''.join(value.decode() for value in line) for line in raw_records.data)
        return info_records

    def put_info_records(self, records):
        '''Puts the information records in the exodus file

        Parameters
        ----------
        info_records : sequence of strings
            A sequence (list/tuple/etc.) containing the information records.
        '''
        if 'num_info' in self._ncdf_handle.dimensions:
            raise ExodusError('Information Records have already been put to the exodus file')
        # Make sure that it is the correct shape
        num_info = len(records)
        len_line = self._ncdf_handle.dimensions['len_line'].size
        array = np.zeros((num_info, len_line), dtype='|S1')
        for i, line in enumerate(records):
            if len(line) > len_line:
                print('Warning:When setting info records, entry "{}" (record {}) will be truncated'.format(line, i))
                line = line[:len_line]
            listed_string = [val for i, val in enumerate(line)]
            array[i, :len(listed_string)] = listed_string
        # Create the dimension
        self._ncdf_handle.createDimension('num_info', num_info)
        # Create the variable
        self._ncdf_handle.createVariable('info_records', 'S1', ('num_info', 'len_line'))
        # Assign to it
        self._ncdf_handle.variables['info_records'][:] = array[:]

    @property
    def num_times(self):
        return self._ncdf_handle.dimensions['time_step'].size

    def get_times(self, indices=None):
        '''Gets the time values from the exodus file

        Returns
        -------
        time_array : np.masked_array
            A masked_array containing the time values.
        '''
        if indices is None:
            indices = slice(None)
        elif np.size(indices) == 0:
            return np.ma.zeros((0,))
        return self._ncdf_handle.variables['time_whole'][indices]

    def set_time(self, step, value):
        '''Sets the time value of a given step in the exodus file

        Parameters
        ----------
        step : int
            The index in the time vector to set to value
        value : float
            A real number to set the value of the specified index in the time
            vector to.

        Notes
        -----
        If step is not a valid index for the time vector, the time vector will
        be expanded so that it is.
        '''
        self._ncdf_handle.variables['time_whole'][step] = value

    def set_times(self, values):
        '''Sets the time vector for the exodus file

        Parameters
        ----------
        values : array-like
            A 1-dimensional array that has the time step values as it's entries

        Notes
        -----
        If the vector is longer than the current time step vector, it will be
        expanded.  If the vector is shorter than the current time step vector,
        only the first len(values) entries of the time vector will be assigned.
        '''
        self._ncdf_handle.variables['time_whole'][:] = values

    @property
    def num_dimensions(self):
        return self._ncdf_handle.dimensions['num_dim'].size

    r'''
     _   _           _
    | \ | | ___   __| | ___  ___
    |  \| |/ _ \ / _` |/ _ \/ __|
    | |\  | (_) | (_| |  __/\__ \
    |_| \_|\___/ \__,_|\___||___/
    '''

    @property
    def num_nodes(self):
        return self._ncdf_handle.dimensions['num_nodes'].size

    def get_coord_names(self):
        '''Retrieve the coordinate names in the exodus file.

        Returns
        -------
        coord_names : tuple
            Returns a tuple of strings containing the coordinate names in the
            exodus file.
        '''
        try:
            raw_records = self._ncdf_handle.variables['coor_names']
        except KeyError:
            raise ExodusError('Coordinate Names are not yet defined!')
        coord_names = tuple(''.join(value.decode() for value in line if not isinstance(
            value, np.ma.core.MaskedConstant)) for line in raw_records)

        return coord_names

    def put_coord_names(self, coord_names):
        '''Puts the coordinate names into the exodus file

        Parameters
        ----------
        coord_names : sequence of strings
            A sequence (list/tuple/etc.) containing the coordinate names.
        '''
        if 'coor_names' in self._ncdf_handle.variables:
            raise ExodusError('Coordinate Names have already been put to the exodus file')
        # Make sure that it is the correct shape
        num_dim = self.num_dimensions  # self._ncdf_handle.dimensions['num_dim'].size
        len_name = self._ncdf_handle.dimensions['len_name'].size
        array = np.zeros((num_dim, len_name), dtype='|S1')
        for i, line in enumerate(coord_names):
            if len(line) > len_name:
                print('Warning:When setting info records, entry "{}" (record {}) will be truncated'.format(line, i))
                line = line[:len_name]
            listed_string = [val for i, val in enumerate(line)]
            # Create the coordinate variables
            array[i, :len(listed_string)] = listed_string
        # Create the variable
        self._ncdf_handle.createVariable('coor_names', 'S1', ('num_dim', 'len_name'))
        # Assign to it
        self._ncdf_handle.variables['coor_names'][:] = array[:]

    def get_coords(self):
        '''Retrieve the coordinates of the nodes in the exodus file.

        Returns
        -------
        coords : np.array
            Returns a 2D array with size num_dims x num_nodes
        '''
        # TODO Add error checking
        coord_names = ('coordx', 'coordy', 'coordz')[:self.num_dimensions]
        raw_list = [self._ncdf_handle.variables[name][:] for name in coord_names]
        coords = np.array(raw_list)
        return coords

    def get_coord(self, index):
        '''Retrieve the coordinates of the specified node in the exodus file.

        Parameters
        ----------
        index : int
            The global node index (not node number) of the node of which the
            coordinates are desired

        Returns
        -------
        coords : np.array
            Returns an array with size num_dims
        '''
        # TODO Add Error Checking
        coord_names = ('coordx', 'coordy', 'coordz')[:self.num_dimensions]
        raw_list = [self._ncdf_handle.variables[name][index] for name in coord_names]
        coords = np.array(raw_list)
        return coords

    def put_coords(self, coords):
        '''Puts the coordinate values into the exodus file

        Parameters
        ----------
        coords : np.ndarray
            A 2d array containing coordinate values.
        '''
        coord_names = ('coordx', 'coordy', 'coordz')[:self._ncdf_handle.dimensions['num_dim'].size]
        if coord_names[0] in self._ncdf_handle.variables:
            raise ExodusError('Coordinates have already been put to the exodus file')
        coords = np.array(coords)
        if coords.shape != (self.num_dimensions, self.num_nodes):
            raise ExodusError('coords.shape should be (self.num_dimensions,self.num_nodes)')
        for i, name in enumerate(coord_names):
            self._ncdf_handle.createVariable(name, self.floating_point_type, ('num_nodes',))
            self._ncdf_handle.variables[name][:] = coords[i]

    def get_node_num_map(self):
        '''Retrieve the list of local node IDs from the exodus file.

        Returns
        -------
        node_num_map : np.array
            Returns a 1D array with size num_nodes, denoting the node number
            for the node in each index

        Notes
        -----
        If there is no node_num_map in the exodus file, this function simply
        returns an array from 1 to self.num_nodes
        '''
        if 'node_num_map' in self._ncdf_handle.variables:
            return self._ncdf_handle.variables['node_num_map'][:]
        else:
            return np.ma.MaskedArray(np.arange(self.num_nodes) + 1)

    def put_node_num_map(self, node_num_map):
        '''Puts a list of local node IDs into the exodus file.

        Parameters
        ----------
        node_num_map : np.array
            A 1D array with size num_nodes, denoting the node number
            for the node in each index

        '''
        if 'node_num_map' in self._ncdf_handle.variables:
            raise ExodusError('node_num_map has already been put to the exodus file.')
        # Create the variable
        self._ncdf_handle.createVariable('node_num_map', 'int32', ('num_nodes',))
        self._ncdf_handle.variables['node_num_map'][:] = node_num_map

    def get_node_variable_names(self):
        '''Gets a tuple of nodal variable names from the exodus file

        Returns
        -------
        node_var_names : tuple of strings
            Returns a tuple containing the names of the nodal variables in the
            exodus file.
        '''
        try:
            raw_records = self._ncdf_handle.variables['name_nod_var']
        except KeyError:
            raise ExodusError('Node Variable Names are not defined!')
        node_var_names = tuple(''.join(value.decode() for value in line if not isinstance(
            value, np.ma.core.MaskedConstant)) for line in raw_records)
        return node_var_names

    def put_node_variable_names(self, node_var_names):
        '''Puts the specified variable names in the exodus file

        Parameters
        ----------
        node_var_names : tuple of strings
            A tuple containing the names of the nodal variables in the model
        '''
        if 'name_nod_var' in self._ncdf_handle.variables:
            raise ExodusError('Nodal Variable Names have already been put to the exodus file')
        # Make sure that it is the correct shape
        num_names = len(node_var_names)
        string_length = self._ncdf_handle.dimensions['len_name'].size
        array = np.zeros((num_names, string_length), dtype='|S1')
        for i, line in enumerate(node_var_names):
            if len(line) > string_length:
                print('Warning:When setting nodal variable names, entry "{}" (record {}) will be truncated'.format(
                    line, i))
                line = line[:string_length]
            listed_string = [val for i, val in enumerate(line)]
            # Create the coordinate variables
            array[i, :len(listed_string)] = listed_string
            self._ncdf_handle.createVariable('vals_nod_var{:d}'.format(
                i + 1), self.floating_point_type, ('time_step', 'num_nodes'))
        # Create the dimension
        self._ncdf_handle.createDimension('num_nod_var', num_names)
        # Create the variable
        self._ncdf_handle.createVariable('name_nod_var', 'S1', ('num_nod_var', 'len_name'))
        # Assign to it
        self._ncdf_handle.variables['name_nod_var'][:] = array[:]

    @property
    def num_node_variables(self):
        try:
            return self._ncdf_handle.dimensions['num_nod_var'].size
        except KeyError:
            raise ExodusError('Number of Node Variables is not defined!')

    def get_node_variable_values(self, name_or_index, step=None):
        '''Gets the node variable values for the specified timestep

        Parameters
        ----------
        name_or_index : str or int
            Name or Index of the nodal variable that is desired.  If
            type(name_or_index) == str, then it is assumed to be the name.  If
            type(name_or_index) == int, then it is assumed to be the index.
        step : int
            Time step at which to recover the nodal variable

        Returns
        -------
        node_variable_values : maskedarray
            A 1d array consisting of the variable values for each node at the
            specified time step.
        '''
        if isinstance(name_or_index, (int, np.integer)):
            index = name_or_index
        elif isinstance(name_or_index, (str, np.character)):
            try:
                index = self.get_node_variable_names().index(name_or_index)
            except ValueError:
                raise ExodusError('Name {} not found in self.get_node_variable_names().  Options are {}'.format(
                    name_or_index, self.get_node_variable_names()))
        vals_nod_var_name = 'vals_nod_var{:d}'.format(index + 1)
        if step is not None:
            if step >= self.num_times:
                raise ExodusError('Invalid Time Step')
            return self._ncdf_handle.variables[vals_nod_var_name][step, :]
        else:
            return self._ncdf_handle.variables[vals_nod_var_name][:]

    def get_node_variable_value(self, name_or_index, node_index, step=None):
        '''Gets a node variable value for the specified timestep

        Parameters
        ----------
        name_or_index : str or int
            Name or Index of the nodal variable that is desired.  If
            type(name_or_index) == str, then it is assumed to be the name.  If
            type(name_or_index) == int, then it is assumed to be the index.
        node_index : int
            Node index at which to recover the nodal variable
        step : int
            Time step at which to recover the nodal variable
        Returns
        -------
        node_variable_value : float
            The variable values for the specified node at the specified time step.
        '''
        if isinstance(name_or_index, (int, np.integer)):
            index = name_or_index
        elif isinstance(name_or_index, (str, np.character)):
            try:
                index = self.get_node_variable_names().index(name_or_index)
            except ValueError:
                raise ExodusError('Name {} not found in self.get_node_variable_names().  Options are {}'.format(
                    name_or_index, self.get_node_variable_names()))
        vals_nod_var_name = 'vals_nod_var{:d}'.format(index + 1)
        if step is not None:
            if step >= self.num_times:
                raise ExodusError('Invalid Time Step')
            return self._ncdf_handle.variables[vals_nod_var_name][step, node_index]
        else:
            return self._ncdf_handle.variables[vals_nod_var_name][:, node_index]

    def set_node_variable_values(self, name_or_index, step, values):
        '''Sets the node variable values for the specified timestep

        Parameters
        ----------
        name_or_index : str or int
            Name or Index of the nodal variable that is desired.  If
            type(name_or_index) == str, then it is assumed to be the name.  If
            type(name_or_index) == int, then it is assumed to be the index.
        step : int
            Time step at which to recover the nodal variable
        values : array-like
            A 1d array consisting of the variable values for each node at the
            specified time step.

        Notes
        -----
        If step is not a valid index for the time vector, the time vector will
        be expanded so that it is.
        '''
        if isinstance(name_or_index, (int, np.integer)):
            index = name_or_index
        elif isinstance(name_or_index, (str, np.character)):
            try:
                index = self.get_node_variable_names().index(name_or_index)
            except ValueError:
                raise ValueError('Name {} not found in self.get_node_variable_names().  Options are {}'.format(
                    name_or_index, self.get_node_variable_names()))
        vals_nod_var_name = 'vals_nod_var{:d}'.format(index + 1)
        self._ncdf_handle.variables[vals_nod_var_name][step, :] = values

    def set_node_variable_value(self, name_or_index, node_index, step, value):
        '''Sets the node variable values for the specified timestep

        Parameters
        ----------
        name_or_index : str or int
            Name or Index of the nodal variable that is desired.  If
            type(name_or_index) == str, then it is assumed to be the name.  If
            type(name_or_index) == int, then it is assumed to be the index.
        node_index : int
            Node index at which to recover the nodal variable
        step : int
            Time step at which to recover the nodal variable
        value : float
            The variable value for the specified node at the specified time step.

        Notes
        -----
        If step is not a valid index for the time vector, the time vector will
        be expanded so that it is.
        '''
        if isinstance(name_or_index, (int, np.integer)):
            index = name_or_index
        elif isinstance(name_or_index, (str, np.character)):
            try:
                index = self.get_node_variable_names().index(name_or_index)
            except ValueError:
                raise ValueError('Name {} not found in self.get_node_variable_names().  Options are {}'.format(
                    name_or_index, self.get_node_variable_names()))
        vals_nod_var_name = 'vals_nod_var{:d}'.format(index + 1)
        self._ncdf_handle.variables[vals_nod_var_name][step, node_index] = value

    def get_displacements(self, displacement_name='Disp', capital_coordinates=True):
        return np.array([self.get_node_variable_values(displacement_name + (val.upper() if capital_coordinates else val.lower()))
                         for val in 'xyz'])

    r'''
      _____ _                           _
     | ____| | ___ _ __ ___   ___ _ __ | |_ ___
     |  _| | |/ _ \ '_ ` _ \ / _ \ '_ \| __/ __|
     | |___| |  __/ | | | | |  __/ | | | |_\__ \
     |_____|_|\___|_| |_| |_|\___|_| |_|\__|___/
    '''

    @property
    def num_elems(self):
        if 'num_elem' in self._ncdf_handle.dimensions:
            return self._ncdf_handle.dimensions['num_elem'].size
        else:
            return 0

    def get_elem_num_map(self):
        '''Retrieve the list of local element IDs from the exodus file.

        Returns
        -------
        elem_num_map : np.array
            Returns a 1D array with size num_elems, denoting the element number
            for the element in each index

        Notes
        -----
        If there is no elem_num_map in the exodus file, this function simply
        returns an array from 1 to self.num_elems
        '''
        if 'elem_num_map' in self._ncdf_handle.variables:
            return self._ncdf_handle.variables['elem_num_map'][:]
        else:
            return np.ma.MaskedArray(np.arange(self.num_elems) + 1)

    def put_elem_num_map(self, elem_num_map):
        '''Puts a list of local element IDs into the exodus file.

        Parameters
        ----------
        elem_num_map : np.array
            A 1D array with size num_elems, denoting the element number
            for the element in each index

        '''
        if 'elem_num_map' in self._ncdf_handle.variables:
            raise ExodusError('elem_num_map has already been put to the exodus file.')
        # Create the variable
        self._ncdf_handle.createVariable('elem_num_map', 'int32', ('num_elem',))
        self._ncdf_handle.variables['elem_num_map'][:] = elem_num_map

    @property
    def num_blks(self):
        return self._ncdf_handle.dimensions['num_el_blk'].size

    def get_elem_blk_ids(self):
        ''' Gets a list of the element block ID numbers in the exodus file

        Returns
        -------
        block_ids : tuple
            A tuple containing the block ID numbers
        '''
        try:
            return tuple(self._ncdf_handle.variables['eb_prop1'][:])
        except KeyError:
            raise ExodusError('Element Block IDs are not defined!')

    def put_elem_blk_ids(self, block_ids):
        '''Puts a list of the element block ID numbers in the exodus file

        Parameters
        ----------
        block_ids : array-like
            A sequency of integers specifying the block id numbers
        '''
        if 'eb_prop1' in self._ncdf_handle.variables:
            raise ExodusError('Element Block IDs have already been put to the exodus file.')
        block_ids = np.array(block_ids).flatten()
        if len(block_ids) != self._ncdf_handle.dimensions['num_el_blk'].size:
            raise ExodusError('Invalid Number of Block IDs Specified')
        self._ncdf_handle.createVariable('eb_prop1', 'int32', ('num_el_blk',))
        self._ncdf_handle.variables['eb_prop1'][:] = block_ids
        self._ncdf_handle.variables['eb_prop1'].setncattr('name', 'ID')
        # Create block names too
        block_names = ['block_{:d}'.format(id) for id in block_ids]
        num_blk, len_name = (self._ncdf_handle.dimensions['num_el_blk'].size,
                             self._ncdf_handle.dimensions['len_name'].size)
        array = np.zeros((num_blk, len_name), dtype='|S1')
        for i, line in enumerate(block_names):
            if len(line) > len_name:
                print('Warning:When setting block names, entry "{}" (record {}) will be truncated'.format(line, i))
                line = line[:len_name]
            listed_string = [val for i, val in enumerate(line)]
            # Create the coordinate variables
            array[i, :len(listed_string)] = listed_string
        # Create the variable
        self._ncdf_handle.createVariable('eb_names', 'S1', ('num_el_blk', 'len_name'))
        # Assign to it
        self._ncdf_handle.variables['eb_names'][:] = array[:]
        # Create eb_status as well
        self._ncdf_handle.createVariable('eb_status', 'int32', ('num_el_blk',))
        self._ncdf_handle.variables['eb_status'][:] = 0

    def get_elem_blk_info(self, id):
        ''' Gets the element block information for the specified element block ID

        Parameters
        ----------
        id : int
            Element Block ID number

        Returns
        -------
        element_type : str
            Name of the Element Type
        elements_in_block : int
            Number of elements in the block
        nodes_per_element : int
            Number of nodes per element in the block
        attributes_per_element : int
            Number of attributes per element
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid Block ID')
        try:
            num_el_in_blk = self._ncdf_handle.dimensions['num_el_in_blk{:d}'.format(
                block_index)].size
            num_nod_per_el = self._ncdf_handle.dimensions['num_nod_per_el{:d}'.format(
                block_index)].size
            element_type = self._ncdf_handle.variables['connect{:d}'.format(
                block_index)].getncattr('elem_type')
            if 'num_att_in_blk{:d}'.format(block_index) in self._ncdf_handle.dimensions:
                attr_per_element = self._ncdf_handle.dimensions['num_att_in_blk{:d}'.format(
                    block_index)].size
            else:
                attr_per_element = 0
        except KeyError:
            raise ExodusError('Block {:d} not initialized correctly')
        return element_type, num_el_in_blk, num_nod_per_el, attr_per_element

    def put_elem_blk_info(self, id, elem_type, num_elements, num_nodes_per_element, num_attrs_per_elem=0):
        '''Puts the element block information for an element block ID into the exodus file

        Parameters
        ----------
        id : int
            The block ID (not index) that the information is being specified for
        elem_type : str
            The element type ('SHELL4','HEX8', etc.) in the block
        num_elements : int
            The number of elements in the block
        num_nodes_per_element : int
            The number of nodes per element in the block
        num_attrs_per_element : int
            The number of attributes per element in the block
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid Block ID')
        num_el_in_blk_name = 'num_el_in_blk{:d}'.format(block_index)
        num_nod_per_el_name = 'num_nod_per_el{:d}'.format(block_index)
        if num_el_in_blk_name in self._ncdf_handle.dimensions:
            raise ExodusError(
                'Information for block {:d} has already been put to the exodus file.'.format(id))
        connect_name = 'connect{:d}'.format(block_index)
        # Create the dimensions
        self._ncdf_handle.createDimension(num_el_in_blk_name, num_elements)
        self._ncdf_handle.createDimension(num_nod_per_el_name, num_nodes_per_element)
        # Create the connectivity matrix
        self._ncdf_handle.createVariable(
            connect_name, 'int32', (num_el_in_blk_name, num_nod_per_el_name))
        self._ncdf_handle.variables[connect_name].setncattr('elem_type', elem_type)
        # Create attributes if necessary
        if num_attrs_per_elem > 0:
            print('Warning:Element attribute functionality has not been thoroughly checked out, use with caution.')
            num_att_in_blk_name = 'num_att_in_blk{:d}'.format(block_index)
            attrib_name = 'attrib{:d}'.format(block_index)
            self._ncdf_handle.createDimension(num_att_in_blk_name, num_attrs_per_elem)
            self._ncdf_handle.createVariable(
                attrib_name, self.floating_point_type, (num_el_in_blk_name, num_att_in_blk_name))
        # Set the eb_status to 1 when it is defined (I'm not sure this is
        # actually what it should be, just every exodus file I've looked at has
        # the eb_status as all ones...)
        self._ncdf_handle.variables['eb_status'][block_index - 1] = 1

    def get_elem_connectivity(self, id):
        '''Gets the element connectivity matrix for a given block

        Parameters
        ----------
        id : int
            The block id for which the element connectivity matrix is desired.

        Returns
        -------
        connectivity : np.masked_array
            The 2d connectivity matrix for the block with dimensions num_elem x
            num_nodes_per_element.  The returned value is converted from the
            1-based Exodus indexing to 0-based Python/NumPy indexing.
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid Block ID')
        conn_name = 'connect{:d}'.format(block_index)
        if not self.num_elems == 0:
            try:
                return self._ncdf_handle.variables[conn_name][:] - 1
            except KeyError:
                raise ExodusError('Element Block {:d} has not been defined yet')

    def set_elem_connectivity(self, id, connectivity):
        '''Sets the element connectivity matrix for a given block

        Parameters
        ----------
        id : int
            The block id for which the element connectivity is being assigned
        connectivity : 2D array-like
            A 2D array of dimension num_elements x num_nodes_per_element
            defining the element connectivity.  Note that the connectivity
            matrix should be 0-based (first node index is zero) per Python
            conventions.  It will be converted to one-based when it is written
            to the Exodus file.

        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid Block ID')
        conn_name = 'connect{:d}'.format(block_index)
        if conn_name not in self._ncdf_handle.variables:
            raise ExodusError('Element Block {:d} has not been defined yet')
        connectivity = np.array(connectivity)
        if self._ncdf_handle.variables[conn_name][:].shape != connectivity.shape:
            raise ExodusError('Shape of connectivity matrix should be {} (recieved {})'.format(self._ncdf_handle.variables[conn_name][:].shape,
                                                                                               connectivity.shape))
        self._ncdf_handle.variables[conn_name][:] = connectivity + 1

    def num_attr(self, id):
        '''Gets the number of attributes per element for an element block

        Parameters
        ----------
        d : int
            The block id for which the number of attributes is desired

        Returns
        -------
        attr_per_element : int
            The number of attributes per element in a block
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid Block ID')
        if 'num_att_in_blk{:d}'.format(block_index) in self._ncdf_handle.dimensions:
            attr_per_element = self._ncdf_handle.dimensions['num_att_in_blk{:d}'.format(
                block_index)].size
        else:
            attr_per_element = 0
        return attr_per_element

    def num_elems_in_blk(self, id):
        '''Gets the number of elements in an element block

        Parameters
        ----------
        d : int
            The block id for which the number of attributes is desired

        Returns
        -------
        elem_per_element : int
            The number of elements in the block
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid Block ID')
        try:
            return self._ncdf_handle.dimensions['num_el_in_blk{:d}'.format(block_index)].size
        except KeyError:
            raise ExodusError('Block {:d} not initialized correctly'.format(id))

    def num_nodes_per_elem(self, id):
        '''Gets the number of nodes per element in an element block

        Parameters
        ----------
        d : int
            The block id for which the number of attributes is desired

        Returns
        -------
        nodes_per_elem : int
            The number of nodes per element in the block
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid Block ID')
        try:
            return self._ncdf_handle.dimensions['num_nod_per_el{:d}'.format(block_index)].size
        except KeyError:
            raise ExodusError('Block {:d} not initialized correctly'.format(id))

    def get_elem_attr(self, id):
        '''Gets the element attributes for a given block

        Parameters
        ----------
        id : int
            The block id for which the element connectivity matrix is desired.

        Returns
        -------
        attributes : np.masked_array
            The 2d attribute matrix for the block with dimensions num_elem x
            num_attributes
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid Block ID')
        try:
            return self._ncdf_handle.variables['attrib{:d}'.format(block_index)][:]
        except KeyError:
            raise ExodusError('No attributes defined for Block {:d}'.format(id))

    def set_elem_attr(self, id, attributes):
        '''Sets the element attributes for a given block

        Parameters
        ----------
        id : int
            The block id for which the element connectivity matrix is desired.
        attributes : 2D array-like
            The 2d attribute matrix for the block with dimensions num_elem x
            num_attributes
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid Block ID')
        attr_name = 'attrib{:d}'.format(block_index)
        if attr_name not in self._ncdf_handle.variables:
            raise ExodusError('Element Block {:d} has no defined attributes')
        attributes = np.array(attributes)
        if self._ncdf_handle.variables[attr_name][:].shape != attributes.shape:
            raise ExodusError('Shape of attributes matrix should be {} (recieved {})'.format(self._ncdf_handle.variables[attr_name][:].shape,
                                                                                             attributes.shape))
        self._ncdf_handle.variables[attr_name][:] = attributes

    def get_elem_type(self, id):
        '''Gets the element type for a given block

        Parameters
        ----------
        id : int
            The block id for which the element connectivity matrix is desired.

        Returns
        -------
        type : str
            The element type of the block
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid Block ID')
        try:
            return self._ncdf_handle.variables['connect{:d}'.format(block_index)].getncattr('elem_type')
        except KeyError:
            raise ExodusError('Element Block {:d} has not been defined yet')

    def get_elem_variable_names(self):
        '''Gets a tuple of element variable names from the exodus file

        Returns
        -------
        elem_var_names : tuple of strings
            Returns a tuple containing the names of the element variables in the
            exodus file.
        '''
        try:
            raw_records = self._ncdf_handle.variables['name_elem_var']
        except KeyError:
            raise ExodusError('Element Variable Names are not defined!')
        elem_var_names = tuple(''.join(value.decode() for value in line if not isinstance(
            value, np.ma.core.MaskedConstant)) for line in raw_records)
        return elem_var_names

    def put_elem_variable_names(self, elem_var_names, elem_var_table=None):
        '''Puts the specified variable names in the exodus file

        Parameters
        ----------
        elem_var_names : tuple of strings
            A tuple containing the names of the element variables in the exodus
            file
        elem_var_table : 2d array-like
            A 2d array of shape num_el_blk x num_elem_var defining which
            element variables are defined for which element blocks
            elem_var_table[i,j] should be True if the j-th element variable is
            defined for the i-th block (index, not id number) in the model.
            If not specified, it is assumed that all variables are defined for
            all blocks.
        '''
        if 'name_elem_var' in self._ncdf_handle.variables:
            raise ExodusError('Element Variable Names have already been put to the exodus file')
        # Make sure that it is the correct shape
        num_names = len(elem_var_names)
        if elem_var_table is None:
            elem_var_table = np.ones((self.num_blks, num_names), dtype=bool)
        elem_var_table = np.array(elem_var_table)
        if elem_var_table.shape != (self.num_blks, num_names):
            raise ExodusError('Shape of elem_var_table matrix should be {} (recieved {})'.format((self.num_blks, num_names),
                                                                                                 elem_var_table.shape))
        string_length = self._ncdf_handle.dimensions['len_name'].size
        array = np.zeros((num_names, string_length), dtype='|S1')
        for i, line in enumerate(elem_var_names):
            if len(line) > string_length:
                print('Warning:When setting nodal variable names, entry "{}" (record {}) will be truncated'.format(
                    line, i))
                line = line[:string_length]
            listed_string = [val for i, val in enumerate(line)]
            # Create the coordinate variables
            array[i, :len(listed_string)] = listed_string
            # Create the element variables if they exist
            for j, blkid in enumerate(self.get_elem_blk_ids()):
                if elem_var_table[j, i]:
                    self._ncdf_handle.createVariable('vals_elem_var{:d}eb{:d}'.format(
                        i + 1, j + 1), self.floating_point_type, ('time_step', 'num_el_in_blk{:d}'.format(j + 1)))
        # Create the dimension
        self._ncdf_handle.createDimension('num_elem_var', num_names)
        # Create the variable
        self._ncdf_handle.createVariable('name_elem_var', 'S1', ('num_elem_var', 'len_name'))
        self._ncdf_handle.createVariable('elem_var_tab', 'int32', ('num_el_blk', 'num_elem_var'))
        # Assign to it
        self._ncdf_handle.variables['name_elem_var'][:] = array[:]
        self._ncdf_handle.variables['elem_var_tab'][:] = elem_var_table

    def get_elem_variable_table(self):
        '''Gets the element variable table

        Gets the element variable table showing which elements are defined for
        which blocks.

        Returns
        -------
        elem_var_table : np.masked_array
            A 2D array with dimension num_blocks x num_element_variables.  If
            the jth element variable is defined for the ith block,
            elem_var_table[i,j] == True
        '''
        try:
            return self._ncdf_handle.variables['elem_var_tab'][:]
        except KeyError:
            raise ExodusError('Element Variable Table has not been defined in the exodus file')

    @property
    def num_elem_variables(self):
        try:
            return self._ncdf_handle.dimensions['num_elem_var'].size
        except KeyError:
            raise ExodusError('Number of element variables is not defined')

    def get_elem_variable_values(self, block_id, name_or_index, step=None):
        '''Gets a block's element variable values for the specified timestep

        Parameters
        ----------
        block_id : int
            Block id number for the block from which element variable values
            are desired.
        name_or_index : str or int
            Name or Index of the element variable that is desired.  If
            type(name_or_index) == str, then it is assumed to be the name.  If
            type(name_or_index) == int, then it is assumed to be the index.
        step : int
            Time step at which to recover the element variable

        Returns
        -------
        elem_variable_values : maskedarray
            A 1d array consisting of the variable values for each element in
            the specified block at the specified time step.
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(block_id)
        except ValueError:
            raise ExodusError('Invalid Block ID')
        var_names = self.get_elem_variable_names()
        if isinstance(name_or_index, (int, np.integer)):
            var_index = name_or_index
            var_name = var_names[var_index]
        elif isinstance(name_or_index, (str, np.character)):
            try:
                var_name = name_or_index
                var_index = var_names.index(var_name)
            except ValueError:
                raise ExodusError(
                    'Name {} not found in self.get_elem_variable_names().  Options are {}'.format(var_name, var_names))
        if not self.get_elem_variable_table()[block_index, var_index]:
            raise ExodusError('Variable {} is not defined for block {}'.format(var_name, block_id))
        variable_value_name = 'vals_elem_var{:d}eb{:d}'.format(var_index + 1, block_index + 1)
        if step is not None:
            if step >= self.num_times:
                raise ExodusError('Invalid Timestep')
            return self._ncdf_handle.variables[variable_value_name][step, :]
        else:
            return self._ncdf_handle.variables[variable_value_name][:, :]

    def get_elem_variable_value(self, block_id, name_or_index, element_index, step=None):
        '''Gets an element's variable value for the specified timestep

        Parameters
        ----------
        block_id : int
            Block id number for the block from which element variable value
            is desired.
        name_or_index : str or int
            Name or Index of the element variable that is desired.  If
            type(name_or_index) == str, then it is assumed to be the name.  If
            type(name_or_index) == int, then it is assumed to be the index.
        element_index : int
            element index at which to recover the nodal variable
        step : int
            Time step at which to recover the nodal variable
        Returns
        -------
        elem_variable_value : float
            The variable values for the specified element at the specified time
            step.
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(block_id)
        except ValueError:
            raise ExodusError('Invalid Block ID')
        var_names = self.get_elem_variable_names()
        if isinstance(name_or_index, (int, np.integer)):
            var_index = name_or_index
            var_name = var_names[var_index]
        elif isinstance(name_or_index, (str, np.character)):
            try:
                var_name = name_or_index
                var_index = var_names.index(var_name)
            except ValueError:
                raise ExodusError(
                    'Name {} not found in self.get_elem_variable_names().  Options are {}'.format(var_name, var_names))
        if not self.get_elem_variable_table()[block_index, var_index]:
            raise ExodusError('Variable {} is not defined for block {}'.format(var_name, block_id))
        variable_value_name = 'vals_elem_var{:d}eb{:d}'.format(var_index + 1, block_index + 1)
        if step is not None:
            if step >= self.num_times:
                raise ExodusError('Invalid Timestep')
            return self._ncdf_handle.variables[variable_value_name][step, element_index]
        else:
            return self._ncdf_handle.variables[variable_value_name][:, element_index]

    def set_elem_variable_values(self, block_id, name_or_index, step, values):
        '''Sets a block's element variable values for the specified timestep

        Parameters
        ----------
        block_id : int
            Block id number for the block from which element variable values
            are desired.
        name_or_index : str or int
            Name or Index of the element variable that is desired.  If
            type(name_or_index) == str, then it is assumed to be the name.  If
            type(name_or_index) == int, then it is assumed to be the index.
        step : int
            Time step at which to recover the element variable
        values : array-like
            A 1d array consisting of the variable values for each element in
            the specified block at the specified time step.

        Notes
        -----
        If step is not a valid index for the time vector, the time vector will
        be expanded so that it is.
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(block_id)
        except ValueError:
            raise ExodusError('Invalid Block ID')
        var_names = self.get_elem_variable_names()
        if isinstance(name_or_index, (int, np.integer)):
            var_index = name_or_index
            var_name = var_names[var_index]
        elif isinstance(name_or_index, (str, np.character)):
            try:
                var_name = name_or_index
                var_index = var_names.index(var_name)
            except ValueError:
                raise ExodusError(
                    'Name {} not found in self.get_elem_variable_names().  Options are {}'.format(var_name, var_names))
        if not self.get_elem_variable_table()[block_index, var_index]:
            raise ExodusError('Variable {} is not defined for block {}'.format(var_name, block_id))
        variable_value_name = 'vals_elem_var{:d}eb{:d}'.format(var_index + 1, block_index + 1)
#        values = np.array(values).flatten()
#        if values.shape != (self.num_elems_in_blk(block_id),):
#            raise ExodusError('values should have {:d} elements'.format(self.num_elems_in_blk(block_id)))
        self._ncdf_handle.variables[variable_value_name][step, :] = values

    def set_elem_variable_value(self, block_id, name_or_index, element_index,
                                step, value):
        '''Sets an element variable value for the specified timestep

        Parameters
        ----------
        block_id : int
            Block id number for the block from which element variable values
            are desired.
        name_or_index : str or int
            Name or Index of the element variable that is desired.  If
            type(name_or_index) == str, then it is assumed to be the name.  If
            type(name_or_index) == int, then it is assumed to be the index.
        step : int
            Time step at which to recover the element variable
        value : float
            The variable values for the specified element in the specified
            block at the specified time step.

        Notes
        -----
        If step is not a valid index for the time vector, the time vector will
        be expanded so that it is.
        '''
        block_ids = self.get_elem_blk_ids()
        try:
            block_index = block_ids.index(block_id)
        except ValueError:
            raise ExodusError('Invalid Block ID')
        var_names = self.get_elem_variable_names()
        if isinstance(name_or_index, (int, np.integer)):
            var_index = name_or_index
            var_name = var_names[var_index]
        elif isinstance(name_or_index, (str, np.character)):
            try:
                var_name = name_or_index
                var_index = var_names.index(var_name)
            except ValueError:
                raise ExodusError(
                    'Name {} not found in self.get_elem_variable_names().  Options are {}'.format(var_name, var_names))
        if not self.get_elem_variable_table()[block_index, var_index]:
            raise ExodusError('Variable {} is not defined for block {}'.format(var_name, block_id))
        variable_value_name = 'vals_elem_var{:d}eb{:d}'.format(var_index + 1, block_index + 1)
        self._ncdf_handle.variables[variable_value_name][step, element_index] = value

    def get_element_property_names(self):
        raise NotImplementedError

    def get_element_property_value(self):
        raise NotImplementedError

    def put_element_property_names(self):
        raise NotImplementedError

    def put_element_property_value(self):
        raise NotImplementedError
    r'''
     _   _           _                _
    | \ | | ___   __| | ___  ___  ___| |_ ___
    |  \| |/ _ \ / _` |/ _ \/ __|/ _ \ __/ __|
    | |\  | (_) | (_| |  __/\__ \  __/ |_\__ \
    |_| \_|\___/ \__,_|\___||___/\___|\__|___/
    '''

    @property
    def num_node_sets(self):
        return self._ncdf_handle.dimensions['num_node_sets'].size

    def get_node_set_names(self):
        '''Retrieve the node set names in the exodus file.

        Returns
        -------
        ns_names : tuple
            Returns a tuple of strings containing the node set names in the
            exodus file.
        '''
        try:
            raw_records = self._ncdf_handle.variables['ns_names']
        except KeyError:
            raise ExodusError('Node set names are not yet defined!')
        ns_names = tuple(''.join(value.decode() for value in line if not isinstance(
            value, np.ma.core.MaskedConstant)) for line in raw_records)
        return ns_names

    def put_node_set_names(self, ns_names):
        '''Puts the node set names into the exodus file

        Parameters
        ----------
        ns_names : sequence of strings
            A sequence (list/tuple/etc.) containing the node set names.
        '''
        if 'ns_names' in self._ncdf_handle.variables:
            raise ExodusError('Node set names have already been put to the exodus file')
        # Make sure that it is the correct shape
        num_ns = self.num_node_sets  # self._ncdf_handle.dimensions['num_dim'].size
        if num_ns != len(ns_names):
            raise ExodusError('Length of ns_names should be same as self.num_node_sets')
        len_name = self._ncdf_handle.dimensions['len_name'].size
        array = np.zeros((num_ns, len_name), dtype='|S1')
        for i, line in enumerate(ns_names):
            if len(line) > len_name:
                print('Warning:When setting info records, entry "{}" (record {}) will be truncated'.format(line, i))
                line = line[:len_name]
            listed_string = [val for i, val in enumerate(line)]
            # Create the coordinate variables
            array[i, :len(listed_string)] = listed_string
        # Create the variable
        self._ncdf_handle.createVariable('ns_names', 'S1', ('num_node_sets', 'len_name'))
        # Assign to it
        self._ncdf_handle.variables['ns_names'][:] = array[:]

    def get_node_set_ids(self):
        ''' Gets a list of the node set ID numbers in the exodus file

        Returns
        -------
        ns_ids : tuple
            A tuple containing the node set ID numbers
        '''
        try:
            return tuple(self._ncdf_handle.variables['ns_prop1'][:])
        except KeyError:
            raise ExodusError('Node set IDs are not defined!')

    def put_node_set_ids(self, ns_ids):
        '''Puts a list of the node set ID numbers in the exodus file

        Parameters
        ----------
        ns_ids : array-like
            A sequency of integers specifying the node set id numbers
        '''
        if 'ns_prop1' in self._ncdf_handle.variables:
            raise ExodusError('Node set IDs have already been put to the exodus file.')
        ns_ids = np.array(ns_ids).flatten()
        if len(ns_ids) != self._ncdf_handle.dimensions['num_node_sets'].size:
            raise ExodusError('Invalid number of Node set IDs specified')
        self._ncdf_handle.createVariable('ns_prop1', 'int32', ('num_node_sets',))
        self._ncdf_handle.variables['ns_prop1'][:] = ns_ids
        self._ncdf_handle.variables['ns_prop1'].setncattr('name', 'ID')
        # Create ns_status as well
        self._ncdf_handle.createVariable('ns_status', 'int32', ('num_node_sets',))
        self._ncdf_handle.variables['ns_status'][:] = 0

    def get_node_set_num_nodes(self, id):
        '''Get the number of nodes in the specified node set

        Parameters
        ----------
        id : int
            Node set ID (not index)

        Returns
        -------
        num_nodes : int
            The number of nodes in the node set.
        '''
        ns_ids = self.get_node_set_ids()
        try:
            ns_index = ns_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid node set ID')
        try:
            return self._ncdf_handle.dimensions['num_nod_ns{:d}'.format(ns_index)].size
        except KeyError:
            raise ExodusError('Node set {:d} not initialized correctly'.format(id))

    def get_node_set_dist_factors(self, id):
        '''Get the distribution factors of the specified node set

        Parameters
        ----------
        id : int
            Node set ID (not index)

        Returns
        -------
        dist_factor : np.array
            The distribution factors in the node set
        '''
        ns_ids = self.get_node_set_ids()
        try:
            ns_index = ns_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid node set ID')
        try:
            return self._ncdf_handle.variables['dist_fact_ns{:d}'.format(ns_index)][:]
        except KeyError:
            raise ExodusError('No distribution factors defined for node set {:d}'.format(id))

    def get_node_set_nodes(self, id):
        '''Get the nodes in the specified node set

        Parameters
        ----------
        id : int
            Node set ID (not index)

        Returns
        -------
        nodes : np.array
            The node indices of nodes in the node set.  Note that while the
            Exodus file format uses 1-based indexing, the returned nodes array
            is converted so it is 0-based.
        '''
        ns_ids = self.get_node_set_ids()
        try:
            ns_index = ns_ids.index(id) + 1
#            print(ns_index)
        except ValueError:
            raise ExodusError('Invalid node set ID')
        try:
            return self._ncdf_handle.variables['node_ns{:d}'.format(ns_index)][:] - 1
        except KeyError:
            raise ExodusError('Node set {:d} not initialized correctly'.format(id))

    def put_node_set_info(self, id, nodes, dist_fact=None):
        '''Puts the node set information for a node set ID into the exodus file

        Parameters
        ----------
        id : int
            The node set ID (not index)
        nodes : array-like
            A 1d array containing the node indices.  Note the array should be
            zero-based, it will be converted to one based when written to the
            exodus file.
        dist_fact : array-like
            A 1d array containing the node set distribution factors.  If not
            specified, the distribution factors will be assumed to be 1.
        '''
        ns_ids = self.get_node_set_ids()
        try:
            ns_index = ns_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid node set ID')
        num_node_in_ns_name = 'num_nod_ns{:d}'.format(ns_index)
        if num_node_in_ns_name in self._ncdf_handle.dimensions:
            raise ExodusError(
                'Information for node set {:d} has already been put to the exodus file.'.format(id))
        nodes = np.array(nodes).flatten()
        if dist_fact is not None:
            dist_fact = np.array(dist_fact).flatten()
            if nodes.shape != dist_fact.shape:
                raise ExodusError('dist_fact must be the same size as nodes')
        # Create dimension
        self._ncdf_handle.createDimension(num_node_in_ns_name, len(nodes))
        # Create variables
        node_name = 'node_ns{:d}'.format(ns_index)
        self._ncdf_handle.createVariable(node_name, 'int32', (num_node_in_ns_name,))
        self._ncdf_handle.variables[node_name][:] = nodes + 1
        if dist_fact is not None:
            distfact_name = 'dist_fact_ns{:d}'.format(ns_index)
            self._ncdf_handle.createVariable(
                distfact_name, self.floating_point_type, (num_node_in_ns_name,))
            self._ncdf_handle.variables[distfact_name][:] = dist_fact
        # Set the ns_status to 1 when it is defined (I'm not sure this is
        # actually what it should be, just every exodus file I've looked at has
        # the eb_status as all ones...)
        self._ncdf_handle.variables['ns_status'][ns_index - 1] = 1
    r'''
       _____ _     _                _
      / ____(_)   | |              | |
     | (___  _  __| | ___  ___  ___| |_ ___
      \___ \| |/ _` |/ _ \/ __|/ _ \ __/ __|
      ____) | | (_| |  __/\__ \  __/ |_\__ \
     |_____/|_|\__,_|\___||___/\___|\__|___/
     '''

    @property
    def num_side_sets(self):
        return self._ncdf_handle.dimensions['num_side_sets'].size

    def get_side_set_names(self):
        '''Retrieve the side set names in the exouds file.

        Returns
        -------
        ss_names : tuple
            Returns a tuple of strings containing the side set names in the
            exodus file
        '''
        try:
            raw_records = self._ncdf_handle.variables['ss_names']
        except KeyError:
            raise ExodusError('Side set names are not yet defined!')
        ss_names = tuple(''.join(value.decode() for value in line if not isinstance(
            value, np.ma.core.MaskedConstant)) for line in raw_records)
        return ss_names

    def put_side_set_names(self, ss_names):
        '''Puts the side set names into the exodus file

        Parameters
        ----------
        ss_names : sequence of strings
            A sequence (list/tuple/etc.) containing the side set names.
        '''
        if 'xs_names' in self._ncdf_handle.variables:
            raise ExodusError('Side set names have already been put to the exodus file')
        # Make sure that it is the correct shape
        num_ss = self.num_side_sets
        if num_ss != len(ss_names):
            raise ExodusError('Length of ss_names should be same as self.num_side_sets')
        len_name = self._ncdf_handle.dimensions['len_name'].size
        array = np.zeros((num_ss, len_name), dtype='|S1')
        for i, line in enumerate(ss_names):
            if len(line) > len_name:
                print('Warning:When setting info records, entry "{}" (record {}) will be truncated'.format(line, i))
                line = line[:len_name]
            listed_string = [val for i, val in enumerate(line)]
            # Create the coordinate variables
            array[i, :len(listed_string)] = listed_string
        # Create the variable
        self._ncdf_handle.createVariable('ss_names', 'S1', ('num_side_sets', 'len_name'))
        # Assign to it
        self._ncdf_handle.variables['ss_names'][:] = array[:]

    def get_side_set_ids(self):
        ''' Gets a list of the side set ID numbers in the exodus file

        Returns
        -------
        ss_ids : tuple
            A tuple containing the side set ID numbers
        '''
        try:
            return tuple(self._ncdf_handle.variables['ss_prop1'][:])
        except KeyError:
            raise ExodusError('Side set IDs are not defined!')

    def put_side_set_ids(self, ss_ids):
        '''Puts a list of the side set ID numbers in the exodus file

        Parameters
        ----------
        ss_ids : array-like
            A sequency of integers specifying the side set id numbers
        '''
        if 'ss_prop1' in self._ncdf_handle.variables:
            raise ExodusError('Side set IDs have already been put to the exodus file.')
        ss_ids = np.array(ss_ids).flatten()
        if len(ss_ids) != self._ncdf_handle.dimensions['num_side_sets'].size:
            raise ExodusError('Invalid number of side set IDs specified')
        self._ncdf_handle.createVariable('ss_prop1', 'int32', ('num_side_sets',))
        self._ncdf_handle.variables['ss_prop1'][:] = ss_ids
        self._ncdf_handle.variables['ss_prop1'].setncattr('name', 'ID')
        # Create ns_status as well
        self._ncdf_handle.createVariable('ss_status', 'int32', ('num_side_sets',))
        self._ncdf_handle.variables['ss_status'][:] = 0

    def get_side_set_num_faces(self, id):
        '''Get the number of faces in the specified side set

        Parameters
        ----------
        id : int
            Side set ID (not index)

        Returns
        -------
        num_faces : int
            The number of faces in the side set.
        '''
        ss_ids = self.get_side_set_ids()
        try:
            ss_index = ss_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid side set ID')
        try:
            return self._ncdf_handle.dimensions['num_side_ss{:d}'.format(ss_index)].size
        except KeyError:
            raise ExodusError('Side set {:d} not initialized correctly'.format(id))

    def get_side_set_dist_factors(self, id):
        '''Get the distribution factors of the specified side set

        Parameters
        ----------
        id : int
            Side set ID (not index)

        Returns
        -------
        dist_factor : np.array
            The distribution factors in the node set
        '''
        ss_ids = self.get_side_set_ids()
        try:
            ss_index = ss_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid side set ID')
        try:
            return self._ncdf_handle.variables['dist_fact_ss{:d}'.format(ss_index)][:]
        except KeyError:
            raise ExodusError('No distribution factors defined for side set {:d}'.format(id))

    def get_side_set_faces(self, id):
        '''Get the faces in the specified side set

        Parameters
        ----------
        id : int
            Side set ID (not index)

        Returns
        -------
        elements : np.array
            The element of each face in the sideset (converted from 1-based
            exodus indexing to 0-based python indexing)
        sides : np.array
            The element side of each face in the sideset (converted from
            1-based exodus indexing to 0-based python indexing)
        '''
        ss_ids = self.get_side_set_ids()
        try:
            ss_index = ss_ids.index(id) + 1
#            print(ss_index)
        except ValueError:
            raise ExodusError('Invalid side set ID')
        try:
            return (self._ncdf_handle.variables['elem_ss{:d}'.format(ss_index)][:] - 1,
                    self._ncdf_handle.variables['side_ss{:d}'.format(ss_index)][:] - 1)
        except KeyError:
            raise ExodusError('Side set {:d} not initialized correctly'.format(id))

    def put_side_set_info(self, id, elements, sides, dist_fact=None):
        '''Puts the side set information for a side set ID into the exodus file

        Parameters
        ----------
        id : int
            The side set ID (not index)
        elements : np.array
            The element of each face in the sideset (converted from 1-based
            exodus indexing to 0-based python indexing)
        sides : np.array
            The element side of each face in the sideset (converted from
            1-based exodus indexing to 0-based python indexing)
        dist_fact : array-like
            A 1d array containing the node set distribution factors.  If not
            specified, the distribution factors will be assumed to be 1.
        '''
        ss_ids = self.get_side_set_ids()
        try:
            ss_index = ss_ids.index(id) + 1
        except ValueError:
            raise ExodusError('Invalid side set ID')
        num_side_in_ss_name = 'num_side_ss{:d}'.format(ss_index)
        if num_side_in_ss_name in self._ncdf_handle.dimensions:
            raise ExodusError(
                'Information for side set {:d} has already been put to the exodus file.'.format(id))
        elements = np.array(elements).flatten()
        sides = np.array(sides).flatten()
        if dist_fact is not None:
            print('Warning:Distribution factors for sidesets have not been thoroughly checked! Use with Caution!')
            dist_fact = np.array(dist_fact).flatten()
            # TODO: Check if the size of dist_fact is correct
#            if nodes.shape != dist_fact.shape:
#                raise ExodusError('dist_fact must be the same size as nodes')
        # Create dimensions
        self._ncdf_handle.createDimension(num_side_in_ss_name, len(elements))
        # Create variables
        elem_name = 'elem_ss{:d}'.format(ss_index)
        side_name = 'side_ss{:d}'.format(ss_index)
        self._ncdf_handle.createVariable(elem_name, 'int32', (num_side_in_ss_name,))
        self._ncdf_handle.createVariable(side_name, 'int32', (num_side_in_ss_name,))
        self._ncdf_handle.variables[elem_name][:] = elements + 1
        self._ncdf_handle.variables[side_name][:] = sides + 1
        if dist_fact is not None:
            distfact_name = 'dist_fact_ss{:d}'.format(ss_index)
            distfact_dim_name = 'num_df_ss{:d}'.format(ss_index)
            self._ncdf_handle.createDimension(distfact_dim_name, len(dist_fact))
            self._ncdf_handle.createVariable(
                distfact_name, self.floating_point_type, (distfact_dim_name,))
            self._ncdf_handle.variables[distfact_name][:] = dist_fact
        # Set the ns_status to 1 when it is defined (I'm not sure this is
        # actually what it should be, just every exodus file I've looked at has
        # the eb_status as all ones...)
        self._ncdf_handle.variables['ss_status'][ss_index - 1] = 1

    def get_side_set_node_list(self, id):
        raise NotImplementedError

    r'''
       _____ _       _           _  __      __        _       _     _
      / ____| |     | |         | | \ \    / /       (_)     | |   | |
     | |  __| | ___ | |__   __ _| |  \ \  / /_ _ _ __ _  __ _| |__ | | ___  ___
     | | |_ | |/ _ \| '_ \ / _` | |   \ \/ / _` | '__| |/ _` | '_ \| |/ _ \/ __|
     | |__| | | (_) | |_) | (_| | |    \  / (_| | |  | | (_| | |_) | |  __/\__ \
      \_____|_|\___/|_.__/ \__,_|_|     \/ \__,_|_|  |_|\__,_|_.__/|_|\___||___/
    '''

    @property
    def num_global_variables(self):
        try:
            return self._ncdf_handle.dimensions['num_glo_var'].size
        except KeyError:
            raise ExodusError('Number of global variables is not defined')

    def put_global_variable_names(self, global_var_names):
        '''Puts the specified global variable names in the exodus file

        Parameters
        ----------
        global_var_names : tuple of strings
            A tuple containing the names of the global variables in the model
        '''
        if 'name_glo_var' in self._ncdf_handle.variables:
            raise ExodusError('Global Variable Names have already been put to the exodus file')
        # Make sure that it is the correct shape
        num_names = len(global_var_names)
        string_length = self._ncdf_handle.dimensions['len_name'].size
        array = np.zeros((num_names, string_length), dtype='|S1')
        for i, line in enumerate(global_var_names):
            if len(line) > string_length:
                print('Warning:When setting global variable names, entry "{}" (record {}) will be truncated'.format(
                    line, i))
                line = line[:string_length]
            listed_string = [val for i, val in enumerate(line)]
            # Create the coordinate variables
            array[i, :len(listed_string)] = listed_string
        # Create the dimension
        self._ncdf_handle.createDimension('num_glo_var', num_names)
        # Create the variable
        self._ncdf_handle.createVariable('name_glo_var', 'S1', ('num_glo_var', 'len_name'))
        self._ncdf_handle.createVariable(
            'vals_glo_var', self.floating_point_type, ('time_step', 'num_glo_var'))
        # Assign to it
        self._ncdf_handle.variables['name_glo_var'][:] = array[:]

    def get_global_variable_names(self):
        '''Gets a tuple of global variable names from the exodus file

        Returns
        -------
        global_var_names : tuple of strings
            Returns a tuple containing the names of the global variables in the
            exodus file.
        '''
        try:
            raw_records = self._ncdf_handle.variables['name_glo_var']
        except KeyError:
            raise ExodusError('Global Variable Names are not defined!')
        global_var_names = tuple(''.join(value.decode() for value in line if not isinstance(
            value, np.ma.core.MaskedConstant)) for line in raw_records)
        return global_var_names

    def get_global_variable_values(self, name_or_index, step=None):
        '''Gets the global variable value for the specified timesteps

        Parameters
        ----------
        name_or_index : str or int
            Name or Index of the global variable that is desired.  If
            type(name_or_index) == str, then it is assumed to be the name.  If
            type(name_or_index) == int, then it is assumed to be the index.
        step : int
            Time step at which to recover the nodal variable

        Returns
        -------
        global_variable_values : maskedarray or float
            A 1d array or floating point number consisting of the variable
            values at the specified time steps.
        '''
        if isinstance(name_or_index, (int, np.integer)):
            index = name_or_index
        elif isinstance(name_or_index, (str, np.character)):
            try:
                index = self.get_global_variable_names().index(name_or_index)
            except ValueError:
                raise ExodusError('Name {} not found in self.get_global_variable_names().  Options are {}'.format(
                    name_or_index, self.get_node_variable_names()))
        vals_glo_var_name = 'vals_glo_var'
        if step is not None:
            if step >= self.num_times:
                raise ExodusError('Invalid Time Step')
            return self._ncdf_handle.variables[vals_glo_var_name][step, index]
        else:
            return self._ncdf_handle.variables[vals_glo_var_name][:, index]

    def set_global_variable_values(self, name_or_index, step, value):
        '''Sets the global variable values for the specified timestep

        Parameters
        ----------
        name_or_index : str or int
            Name or Index of the nodal variable that is desired.  If
            type(name_or_index) == str, then it is assumed to be the name.  If
            type(name_or_index) == int, then it is assumed to be the index.
        step : int
            Time step at which to recover the nodal variable
        value : array-like
            A 1d array consisting of the variable values for each node at the
            specified time step.

        Notes
        -----
        If step is not a valid index for the time vector, the time vector will
        be expanded so that it is.
        '''
        if isinstance(name_or_index, (int, np.integer)):
            index = name_or_index
        elif isinstance(name_or_index, (str, np.character)):
            try:
                index = self.get_global_variable_names().index(name_or_index)
            except ValueError:
                raise ExodusError('Name {} not found in self.get_global_variable_names().  Options are {}'.format(
                    name_or_index, self.get_node_variable_names()))
        vals_glo_var_name = 'vals_glo_var'
        self._ncdf_handle.variables[vals_glo_var_name][step, index] = value

    # TODO: Still need to do Nodeset and Sideset Variables;
    # Coordinate Frames (see ExodusIO.m), Element Block Attribute Names

    def close(self):
        self._ncdf_handle.close()

    def load_into_memory(self, close=True, variables=None, timesteps=None, blocks=None):
        '''Loads the exodus file into an ExodusInMemory object

        This function loads the exodus file into memory in an ExodusInMemory
        format.  Not for use with large files.

        Parameters
        ----------
        close : bool
            Close the netcdf file upon loading into memory.  Optional argument,
            default is true.
        variables : iterable
            A list of variable names that are loaded into memory.  Default is
            to load all variables
        timesteps : iterable
            A list of timestep indices that are loaded into memory.  Default is
            to load all timesteps.
        blocks : iterable
            A list of block ids that are loaded into memory.  Default is to
            load all blocks.

        Returns
        -------
        fexo : ExodusInMemory
            The exodus file in an ExodusInMemory format

        '''
        fexo = ExodusInMemory(self, variables=variables, timesteps=timesteps, blocks=blocks)
        if close:
            self.close()
        return fexo

    def get_block_surface(self, block_id, keep_midside_nodes=False, warn=True):
        '''Gets the node indices and element connectivity of surface elements

        This function "skins" the element block, returning a list of node
        indices and a surface connectivity matrix.

        Parameters
        ----------
        block_id : int
            The ID number of the block that will be skinned.
        keep_midside_nodes : bool
            Specifies whether or not to keep midside nodes in the surface mesh.
            Default is False.
        warn : bool
            Specifies whether or not to warn the user if the block ID doesn't
            have a skinning method defined for its element type.  Default is
            True.

        Returns
        -------
        element_block_information : list
            A list of tuples of element information.  These data are
            element_type, node_indices, block_surface_connectivity, and
            block_surface_original_elements.  The element_type is a string
            representing the new block element type ('quad4','tri3',etc.).  The
            node_indices can be used as an index into the coordinate or nodal
            variable arrays to select nodes corresponding to this block. The
            block_surface_connectivity represents the connectivity array of the
            surface faces of the block.  Values in this array correspond to
            indices into the node_indices array.  To recover the connectivity
            array in the original node indices of the exodus file, it can be
            passed through the node_indices as
            node_indices[block_surface_connectivity].  The
            block_surface_original_elements array shows the original element
            indices of the block that each surface came from.  This can be
            used to map element variables to the new surface mesh.  This list
            will normally be length 1 unless an element type is processed that
            has two different surface elements in it (e.g. wedges have
            tris and quads)
        '''
        elem_type = self.get_elem_type(block_id)
        connectivity = self.get_elem_connectivity(block_id)
        face_connectivity_array = face_connectivity(elem_type, keep_midside_nodes)
        element_block_information = []
        for reduced_elem_type, reduced_connectivity in face_connectivity_array:
            if reduced_elem_type is None:
                if warn:
                    print('Warning: Element Type {:} has no defined face reduction.  Passing connectivity unchanged.'.format(
                        elem_type))
                block_face_connectivity = connectivity
                block_elem_type = elem_type
                block_face_original_elements = np.arange(connectivity.shape[0])
            else:
                block_face_connectivity = connectivity[:,
                                                       reduced_connectivity].reshape(-1, reduced_connectivity.shape[-1])
                block_face_original_elements = np.repeat(np.arange(connectivity.shape[0]),
                                                         reduced_connectivity.shape[0])
                block_elem_type = reduced_elem_type
                # Remove faces that are duplicates
                (unique_rows, unique_row_indices, unique_inverse, unique_counts) = np.unique(np.sort(
                    block_face_connectivity, axis=1), return_index=True, return_inverse=True, return_counts=True, axis=0)
                original_unique_counts = unique_counts[unique_inverse]
                nondupe_faces = original_unique_counts == 1
                block_face_connectivity = block_face_connectivity[nondupe_faces, :]
                block_face_original_elements = block_face_original_elements[nondupe_faces]

            node_indices = np.unique(block_face_connectivity)
            node_map = np.zeros(self.num_nodes, dtype=int)
            node_map[node_indices] = np.arange(len(node_indices))
            block_face_connectivity = node_map[block_face_connectivity]
            element_block_information.append((block_elem_type,
                                              node_indices,
                                              block_face_connectivity,
                                              block_face_original_elements))
        return element_block_information

    def triangulate_surface_mesh(self):
        '''Triangulate a surface mesh for plotting patches

        This function generates a triangle mesh for each block in the model if
        it can.  If there are more than 3 nodes per element in a block, and the
        triangulation scheme hasn't been defined in
        pyexodus.mesh_triangulation_array, it will be skipped.

        Parameters
        ----------
        None

        Returns
        -------
        triangulated_mesh_info : list
            A list of tuples containing block id, node_indices, triangulated
            connectivity, and original block elements
        '''
        triangulated_mesh_info = []
        for block_id in self.get_elem_blk_ids():
            surface_mesh_info = self.get_block_surface(block_id, warn=False)
            for elem_type, node_indices, connectivity, original_elements in surface_mesh_info:
                triangulation_scheme = mesh_triangulation_array(elem_type)
                if triangulation_scheme is None:
                    if connectivity.shape[-1] > 3:
                        print('Warning: More than 3 ({:}) nodes per element in block {:}, this block will not be triangulated'.format(
                            connectivity.shape[-1], block_id))
                    else:
                        triangulated_mesh_info.append(
                            (block_id, node_indices, connectivity, original_elements))
                else:
                    triangulated_mesh_info.append(
                        (block_id,
                         node_indices,
                         connectivity[:,
                                      triangulation_scheme].reshape(-1, triangulation_scheme.shape[-1]),
                         np.repeat(original_elements, triangulation_scheme.shape[0])))
        return triangulated_mesh_info

    def reduce_to_surfaces(self, *args, **kwargs):
        return reduce_exodus_to_surfaces(self, *args, **kwargs)

    def extract_sharp_edges(self, *args, **kwargs):
        return extract_sharp_edges(self, *args, **kwargs)

    def __repr__(self):
        return_string = 'Exodus File at {:}'.format(self.filename)
        try:
            return_string += '\n  {:} Timesteps'.format(self.num_times)
        except Exception:
            pass
        try:
            return_string += '\n  {:} Nodes'.format(self.num_nodes)
        except Exception:
            pass
        try:
            return_string += '\n  {:} Elements'.format(self.num_elems)
        except Exception:
            pass
        try:
            return_string += '\n  Blocks: {:}'.format(', '.join(str(v)
                                                      for v in self.get_elem_blk_ids()))
        except Exception:
            pass
        try:
            return_string += '\n  Node Variables: {:}'.format(
                ', '.join(self.get_node_variable_names()))
        except Exception:
            pass
        try:
            return_string += '\n  Element Variables: {:}'.format(
                ', '.join(self.get_elem_variable_names()))
        except Exception:
            pass
        try:
            return_string += '\n  Global Variables: {:}'.format(
                ', '.join(self.get_global_variable_names()))
        except Exception:
            pass
        return return_string

    # def plot_mesh(self,surface_kwargs={'color':(0,0,1)},
    #              bar_kwargs = {'color':(0,1,0),'tube_radius':None},
    #              point_kwargs = {'color':(1,0,0),'scale_factor':0.1},
    #              plot_surfaces = True, plot_bars = True, plot_points = True,
    #              plot_edges = False):
    #     '''Skins, triangulates, and plots a 3D representation of the mesh.

    #     Parameters
    #     ----------
    #     mesh_kwargs : dict
    #         A dictionary of keyword arguments for the rendering of surface
    #         patches.
    #     bar_kwargs : dict
    #         A dictionary of keyword arguments for the rendering of lines (bar,
    #         beam elements)
    #     point_kwargs : dict
    #         A dictionary of keyword arguments for the rendering of points
    #         (sphere elements)
    #     show : bool
    #         A flag to specify whether or not to show the window.

    #     Returns
    #     -------
    #     window : GLViewWidget
    #         A reference to the view that the 3D content is plotted in.
    #     meshes : list
    #         A list of the mesh objects that have been added to the view.

    #     '''
    #     if mlab is None:
    #         raise ModuleNotFoundError('Mayavi not installed!')
    #     triangulated_mesh_info = self.triangulate_surface_mesh()
    #     meshes = []
    #     for block,node_indices,faces,original_elements in triangulated_mesh_info:
    #         if faces.shape[1] == 3:
    #             if plot_surfaces:
    #                 meshes.append(mlab.triangular_mesh(
    #                         *(self.get_coords()[:,node_indices]),faces,
    #                               **surface_kwargs))
    #                 if plot_edges:
    #                     meshes.append(mlab.triangular_mesh(
    #                         *(self.get_coords()[:,node_indices]),faces,
    #                               representation='wireframe',color=(0,0,0)))
    #         elif faces.shape[1] == 2:
    #             if plot_bars:
    #                 line_items = []
    #                 xs,ys,zs = self.get_coords()[:,node_indices][:,faces]
    #                 for x,y,z in zip(xs,ys,zs):
    #                     line_items.append(mlab.plot3d(x,y,z,**bar_kwargs))
    #                 meshes.append(line_items)
    #         elif faces.shape[1] == 1:
    #             if plot_points:
    #                 meshes.append(mlab.points3d(*(self.get_coords()[:,node_indices]),
    #                                             **point_kwargs))
    #     return meshes


class subfield(SimpleNamespace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return_string = 'Field with subfields: {:}'.format(', '.join(['{:}'.format(key) if type(
            val) is np.ndarray else '{:}={:}'.format(key, val) for key, val in self.__dict__.items()]))
        return return_string

    def __str__(self):
        return repr(self)


class ExodusInMemory:
    '''Read or write exodus files loaded into memory

    This is a convenience class wrapped around the exodus class to enable
    easier manipulation of exodus files that fit entirely into memory

    Parameters
    ----------
    exo : exodus
        An exodus object, if you want to load an exodus file into memory, or
        None if you want to create an empty exodus file.
    variables : iterable
        A list of variable names that are loaded into memory.  Default is
        to load all variables
    timesteps : iterable
        A list of timestep indices that are loaded into memory.  Default is
        to load all timesteps.
    blocks : iterable
        A list of block ids that are loaded into memory.  Default is to load
        all blocks.
    '''

    def __init__(self, exo=None, variables=None, timesteps=None, blocks=None):
        if exo is not None:
            self.load_from_exodus(exo, variables=variables, timesteps=timesteps, blocks=blocks)
        else:
            # Title
            self.title = ''
            # Attributes
            attribute_dict = {}
            attribute_dict['api_version'] = 5.22
            attribute_dict['version'] = 5.22
            attribute_dict['floating_point_word_size'] = 8
            attribute_dict['file_size'] = 1
            attribute_dict['int64_status'] = 0
            self.attributes = subfield(**attribute_dict)
            # QA Records
            self.qa_records = []
            # Info records
            self.info_records = [""]
            # Nodes
            node_dict = {}
            node_dict['coordinates'] = np.empty(0, dtype=float)
            node_dict['node_num_map'] = None
            node_dict['names'] = ('x', 'y', 'z')
            self.nodes = subfield(**node_dict)
            # TODO Coordinate Frames
            # Blocks
            self.blocks = []
            # Element Map
            self.element_map = None
            # Nodesets
            self.nodesets = None
            # Sidesets
            self.sidesets = None
            # Global Variables
            self.global_vars = None
            # Element Variables
            self.elem_vars = None
            # Nodal Variables
            self.nodal_vars = None
            # Time Steps
            self.time = []

    def load_from_exodus(self, exo, variables=None, timesteps=None, blocks=None):
        # Title
        self.title = exo.title
        # Attributes
        attribute_dict = {}
        attribute_dict['api_version'] = exo._ncdf_handle.api_version
        attribute_dict['version'] = exo._ncdf_handle.version
        attribute_dict['floating_point_word_size'] = exo._ncdf_handle.floating_point_word_size
        attribute_dict['file_size'] = exo._ncdf_handle.file_size
        try:
            attribute_dict['int64_status'] = exo._ncdf_handle.int64_status
        except AttributeError:
            attribute_dict['int64_status'] = 0
        self.attributes = subfield(**attribute_dict)

        # QA Records
        try:
            self.qa_records = exo.get_qa_records()
        except ExodusError:
            self.qa_records = []
        # Info Records
        try:
            self.info_records = exo.get_info_records()
        except ExodusError:
            self.info_records = [""]
        # Get Nodes
        node_dict = {}
        node_dict['coordinates'] = exo.get_coords().T
        node_dict['node_num_map'] = exo.get_node_num_map().data
        node_dict['names'] = exo.get_coord_names()
        self.nodes = subfield(**node_dict)
        # TODO Read Coordinate Frames
        # Blocks
        if blocks is None:
            block_ids = exo.get_elem_blk_ids()
            self.element_map = None
        else:
            block_ids = [block_id for block_id in exo.get_elem_blk_ids() if block_id in blocks]
        if timesteps is None:
            time_indices = np.arange(exo.get_times().size)
        else:
            time_indices = list(timesteps)
        self.blocks = []
        for index, blk_id in enumerate(block_ids):
            blk_dict = {}
            blk_dict['id'] = blk_id
            if not exo.num_elems == 0:
                blk_dict['connectivity'] = exo.get_elem_connectivity(blk_id).data
                blk_dict['elem_type'] = exo.get_elem_type(blk_id)
            blk_dict['name'] = ''.join(s.decode() for s in exo._ncdf_handle.variables['eb_names']
                                       [index, :] if not isinstance(s, np.ma.core.MaskedConstant))
            blk_dict['status'] = exo._ncdf_handle.variables['eb_status'][:].data[index]
            try:
                blk_dict['attributes'] = exo.get_elem_attr(blk_id)
            except ExodusError:
                blk_dict['attributes'] = None
            # TODO: Add attribute names once implemented
            try:
                blk_dict['attributes_name'] = None  # exo.get_elem_attr_names(blk_id)
            except ExodusError:
                blk_dict['attributes_name'] = None
            self.blocks.append(subfield(**blk_dict))
        # Element Map
        self.element_map = exo.get_elem_num_map().data

        # Get Nodesets
        try:
            nodeset_ids = exo.get_node_set_ids()
            self.nodesets = []
            for index, ns_id in enumerate(nodeset_ids):
                ns_dict = {}
                ns_dict['id'] = ns_id
                ns_dict['nodes'] = exo.get_node_set_nodes(ns_id)
                try:
                    ns_dict['dist_factors'] = exo.get_node_set_dist_factors(ns_id)
                except ExodusError:
                    ns_dict['dist_factors'] = None
                try:
                    ns_dict['name'] = exo.get_node_set_names()[index]
                except ExodusError:
                    ns_dict['name'] = None
                ns_dict['status'] = exo._ncdf_handle.variables['ns_status'][:].data[index]
                self.nodesets.append(subfield(**ns_dict))
        except ExodusError:
            self.nodesets = None

        # Get Sidesets
        try:
            sideset_ids = exo.get_side_set_ids()
            self.sidesets = []
            for index, ss_id in enumerate(sideset_ids):
                ss_dict = {}
                ss_dict['id'] = ss_id
                ss_dict['elements'], ss_dict['sides'] = exo.get_side_set_faces(ss_id)
                try:
                    ss_dict['dist_factors'] = exo.get_side_set_dist_factors(ss_id)
                except ExodusError:
                    ss_dict['dist_factors'] = None
                ss_dict['status'] = exo._ncdf_handle.variables['ss_status'][:].data[index]
                try:
                    ss_dict['name'] = exo.get_side_set_names()[index]
                except ExodusError:
                    ss_dict['name'] = None
                self.sidesets.append(subfield(**ss_dict))
        except ExodusError:
            self.sidesets = None

        # Get Global Variables
        try:
            num_glob_vars = exo.num_global_variables
            if num_glob_vars == 0:
                self.global_vars = None
            else:
                self.global_vars = []
                global_var_names = exo.get_global_variable_names()
                if variables is not None:
                    global_var_names = [name for name in global_var_names if name in variables]
                    if len(global_var_names) == 0:
                        raise ExodusError('No Global Variables to Load')
                for index, name in enumerate(global_var_names):
                    global_var_dict = {}
                    global_var_dict['name'] = name
                    global_var_dict['data'] = exo.get_global_variable_values(name)[time_indices]
                    self.global_vars.append(subfield(**global_var_dict))
        except ExodusError:
            self.global_vars = None

        # Get Nodal Variables
        try:
            num_node_vars = exo.num_node_variables
            if num_node_vars == 0:
                self.nodal_vars = None
            else:
                self.nodal_vars = []
                node_var_names = exo.get_node_variable_names()
                if variables is not None:
                    node_var_names = [name for name in node_var_names if name in variables]
                    if len(node_var_names) == 0:
                        raise ExodusError('No Nodal Variables to Load')
                for index, name in enumerate(node_var_names):
                    node_var_dict = {}
                    node_var_dict['name'] = name
                    node_var_dict['data'] = exo.get_node_variable_values(name).data[time_indices, :]
                    self.nodal_vars.append(subfield(**node_var_dict))
        except ExodusError:
            self.nodal_vars = None

        # Get Element Variables
        try:
            num_elem_vars = exo.num_elem_variables
            if num_elem_vars == 0:
                self.elem_vars = None
            else:
                self.elem_vars = []
                elem_var_names = exo.get_elem_variable_names()
                variable_truth_table = exo.get_elem_variable_table().data
                for blk_index, blk_id in enumerate(exo.get_elem_blk_ids()):
                    if blk_id not in block_ids:
                        continue
                    block_dict = {}
                    block_dict['id'] = blk_id
                    block_dict['elem_var_data'] = []
                    for elem_index, elem_name in enumerate(elem_var_names):
                        if (variables is not None) and (elem_name not in variables):
                            continue
                        if variable_truth_table[blk_index, elem_index] == 0:
                            block_dict['elem_var_data'].append(None)
                        else:
                            elem_var_dict = {}
                            elem_var_dict['name'] = elem_name
                            elem_var_dict['data'] = exo.get_elem_variable_values(
                                blk_id, elem_index).data[time_indices, :]
                            block_dict['elem_var_data'].append(subfield(**elem_var_dict))
                    if len(block_dict['elem_var_data']) == 0:
                        raise ExodusError('No Element Variables to Load')
                    self.elem_vars.append(subfield(**block_dict))
        except ExodusError:
            self.elem_vars = None

        # Get Time Steps
        self.time = exo.get_times(time_indices).data

    @staticmethod
    def from_sdynpy(geometry, displacement_data=None):
        fexo = ExodusInMemory()

        # Put in the node positions
        node_data = np.sort(geometry.node)
        definition_coordinate_systems = geometry.coordinate_system(node_data.def_cs)
        global_positions = global_coord(definition_coordinate_systems, node_data.coordinate)
        fexo.nodes.coordinates = global_positions
        fexo.nodes.node_num_map = node_data.id
        node_map_dict = {node_id: node_index for node_index, node_id in enumerate(node_data.id)}
        node_map_dict[0] = -1 # For tracelines, map 0 (pick up pen) to -1 (something easy to throw out)
        node_index_map = np.vectorize(
            node_map_dict.__getitem__)
        # Now go through and add in the elements
        element_data = geometry.element
        element_types = np.unique(element_data.type)
        elem_colors = []
        for elem_type in element_types:
            blk_dict = {}
            blk_dict['id'] = elem_type
            blk_dict['name'] = 'block_{:}'.format(elem_type)
            elements_this_type = element_data[element_data.type == elem_type]
            connectivity = np.stack(elements_this_type.connectivity)
            blk_dict['connectivity'] = node_index_map(connectivity)
            blk_dict['elem_type'] = _inverse_exodus_elem_type_map[elem_type]
            blk_dict['status'] = 1
            blk_dict['attributes'] = None
            blk_dict['attributes_name'] = None
            elem_colors.append(elements_this_type.color)
            fexo.blocks.append(subfield(**blk_dict))

        # Now do the same for the tracelines
        traceline_data = geometry.traceline
        for i, (key, traceline) in enumerate(traceline_data.ndenumerate()):
            connectivity_1d = node_index_map(np.stack(traceline.connectivity))
            connectivity = np.concatenate((connectivity_1d[:-1, np.newaxis],
                                           connectivity_1d[1:, np.newaxis]), axis=-1)
            connectivity = connectivity[~np.any(connectivity == -1, axis=1)] # We mapped 0s to -1, so we want to throw those out
            blk_dict = {}
            blk_dict['id'] = i + 500
            blk_dict['name'] = traceline.description
            blk_dict['connectivity'] = connectivity
            blk_dict['elem_type'] = 'bar'
            blk_dict['status'] = 1
            blk_dict['attributes'] = None
            blk_dict['attributes_name'] = None
            elem_colors.append(np.ones(connectivity.shape[0]) * traceline.color)
            fexo.blocks.append(subfield(**blk_dict))

        from ..core.sdynpy_shape import ShapeArray
        from ..core.sdynpy_data import NDDataArray, TimeHistoryArray
        if isinstance(displacement_data, ShapeArray):
            shapes = displacement_data.flatten()
            fexo.time = shapes.frequency
            # Add global variables frequency and damping
            for name in ['frequency', 'damping']:
                global_var_dict = {}
                global_var_dict['name'] = name
                global_var_dict['data'] = shapes[name]
            # Now add displacement data for the shapes
            coordinates = shapes[0].coordinate.copy()
            indices = np.in1d(coordinates.node, node_data.id)
            coordinates = coordinates[indices]
            # Keep only displacements
            coordinates = coordinates[(abs(coordinates.direction) <= 3) &
                                      (abs(coordinates.direction) >= 1)]
            local_deformation_directions = coordinates.local_direction()
            coordinate_nodes = node_data(coordinates.node)
            displacement_coordinate_systems = geometry.coordinate_system(coordinate_nodes.disp_cs)
            shape_displacements = shapes[coordinates]
            coordinates.node = node_index_map(coordinates.node)
            points = global_positions[coordinates.node]
            global_deformation_directions = global_deflection(displacement_coordinate_systems,
                                                              local_deformation_directions,
                                                              points)
            node_displacements = np.zeros((shape_displacements.shape[0],)
                                          + node_data.shape
                                          + (3,), dtype=shape_displacements.dtype)

            for coordinate, direction, shape_coefficients in zip(coordinates, global_deformation_directions, shape_displacements.T):
                node_index = coordinate.node
                displacements = shape_coefficients[:, np.newaxis] * direction
                node_displacements[:, node_index, :] += displacements

            # Now add the variables
            fexo.nodal_vars = []
            for i, name in enumerate(('DispX', 'DispY', 'DispZ')):
                node_var_dict = {}
                node_var_dict['name'] = name
                node_var_dict['data'] = node_displacements[..., i]
                fexo.nodal_vars.append(subfield(**node_var_dict))
        elif isinstance(displacement_data, NDDataArray):
            if isinstance(displacement_data, TimeHistoryArray):
                data = displacement_data.flatten()
                fexo.time = data[0].abscissa
                # Now add displacement data for the shapes
                coordinates = data.response_coordinate
                indices = np.in1d(coordinates.node, node_data.id)
                coordinates = coordinates[indices]
                # Keep only displacements
                coordinates = coordinates[(abs(coordinates.direction) <= 3)
                                          & (abs(coordinates.direction) >= 1)]
                local_deformation_directions = coordinates.local_direction()
                coordinate_nodes = node_data(coordinates.node)
                displacement_coordinate_systems = geometry.coordinate_system(
                    coordinate_nodes.disp_cs)
                data_displacements = data[coordinates[:, np.newaxis]].ordinate.T
                coordinates.node = node_index_map(coordinates.node)
                points = global_positions[coordinates.node]
                global_deformation_directions = global_deflection(displacement_coordinate_systems,
                                                                  local_deformation_directions,
                                                                  points)
                node_displacements = np.zeros((data_displacements.shape[0],)
                                              + node_data.shape
                                              + (3,), dtype=data_displacements.dtype)
                for coordinate, direction, shape_coefficients in zip(coordinates, global_deformation_directions, data_displacements.T):
                    node_index = coordinate.node
                    displacements = shape_coefficients[:, np.newaxis] * direction
                    node_displacements[:, node_index, :] += displacements

                # Now add the variables
                fexo.nodal_vars = []
                for i, name in enumerate(('DispX', 'DispY', 'DispZ')):
                    node_var_dict = {}
                    node_var_dict['name'] = name
                    node_var_dict['data'] = node_displacements[..., i]
                    fexo.nodal_vars.append(subfield(**node_var_dict))
            else:
                raise NotImplementedError(
                    'Saving Functions with type {:} to Exodus Data is Not Implemented Yet'.format(type(displacement_data)))
        else:
            fexo.time = np.array([0])
            fexo.nodal_vars = []

        # Now add color information
        node_var_dict = {}
        node_var_dict['name'] = 'NodeColor'
        node_var_dict['data'] = np.tile(node_data.color[np.newaxis, :], (len(fexo.time), 1))
        fexo.nodal_vars.append(subfield(**node_var_dict))

        fexo.elem_vars = []
        for block, color in zip(fexo.blocks, elem_colors):
            block_dict = {}
            block_dict['id'] = block.id
            elem_var_dict = {}
            elem_var_dict['name'] = 'ElemColor'
            elem_var_dict['data'] = color * np.ones((len(fexo.time), 1))

            block_dict['elem_var_data'] = [subfield(**elem_var_dict)]
            fexo.elem_vars.append(subfield(**block_dict))

        return fexo

    def write_to_file(self, filename, clobber=False):
        num_nodes, num_dims = self.nodes.coordinates.shape
        num_elem = sum([block.connectivity.shape[0] for block in self.blocks])
        num_blocks = len(self.blocks)
        if self.nodesets is None:
            num_node_sets = 0
        else:
            num_node_sets = len(self.nodesets)
        if self.sidesets is None:
            num_side_sets = 0
        else:
            num_side_sets = len(self.sidesets)
        exo_out = Exodus(filename, 'w', self.title,
                         num_dims, num_nodes, num_elem, num_blocks, num_node_sets,
                         num_side_sets, clobber=clobber)

        # Write time steps
        exo_out.set_times(self.time)

        # Initialize Information
        exo_out.put_info_records(self.info_records)
        qa_records = list(self.qa_records)
        qa_records.append(('pyexodus', __version__, str(
            datetime.datetime.now().date()), str(datetime.datetime.now().time())))
        exo_out.put_qa_records(qa_records)

        # Write the nodes
        exo_out.put_coord_names(self.nodes.names)
        exo_out.put_coords(self.nodes.coordinates.T)
        if self.nodes.node_num_map is not None:
            exo_out.put_node_num_map(self.nodes.node_num_map)

        # Initialize blocks
        exo_out.put_elem_blk_ids([block.id for block in self.blocks])
        for block in self.blocks:
            if block.attributes is None:
                nattr_per_elem = 0
            else:
                nattr_per_elem = block.attributes.shape[1]
            exo_out.put_elem_blk_info(block.id,
                                      block.elem_type,
                                      block.connectivity.shape[0],
                                      block.connectivity.shape[1],
                                      nattr_per_elem)
            exo_out.set_elem_connectivity(block.id, block.connectivity)
            if nattr_per_elem > 0:
                exo_out.set_elem_attr(block.id, block.attributes)

        # Element Map
        if self.element_map is not None:
            exo_out.put_elem_num_map(self.element_map)

        # Nodesets
        if self.nodesets is not None:
            exo_out.put_node_set_ids([ns.id for ns in self.nodesets])
            exo_out.put_node_set_names(
                ['' if nodeset.name is None else nodeset.name for nodeset in self.nodesets])
            for nodeset in self.nodesets:
                exo_out.put_node_set_info(nodeset.id, nodeset.nodes, nodeset.dist_factors)

        # Sidesets
        if self.sidesets is not None:
            exo_out.put_side_set_ids([ns.id for ns in self.sidesets])
            exo_out.put_side_set_names(
                ['' if sideset.name is None else sideset.name for sideset in self.sidesets])
            for sideset in self.sidesets:
                exo_out.put_side_set_info(sideset.id, sideset.elements,
                                          sideset.sides, sideset.dist_factors)

        # Global Variables #TODO: Make it so I can write all timesteps at once.
        if self.global_vars is not None:
            exo_out.put_global_variable_names([var.name for var in self.global_vars])
            for i, var in enumerate(self.global_vars):
                for j, time in enumerate(self.time):
                    exo_out.set_global_variable_values(i, j, self.global_vars[i].data[j])

        # Nodal Variables #TODO: Make it so I can write all timesteps at once.
        if self.nodal_vars is not None:
            exo_out.put_node_variable_names([var.name for var in self.nodal_vars])
            for i, var in enumerate(self.nodal_vars):
                for j, time in enumerate(self.time):
                    exo_out.set_node_variable_values(i, j, self.nodal_vars[i].data[j, :])

        # Element Variables # TODO: Make it so I can write all timesteps at once
        if self.elem_vars is not None:
            nblocks = len(self.elem_vars)
            nelemvars = len(self.elem_vars[0].elem_var_data)
            elem_table = np.zeros((nblocks, nelemvars), dtype=int)
            for i in range(nblocks):
                for j in range(nelemvars):
                    elem_table[i, j] = not self.elem_vars[i].elem_var_data[j] is None
            # Detect if any variables are not used ever
            variable_names = []
            keep_booleans = []
            for j in range(nelemvars):
                indices = np.nonzero(elem_table[:, j])[0]
                if len(indices) == 0:
                    variable_names.append(None)
                    keep_booleans.append(False)
                else:
                    variable_names.append(self.elem_vars[indices[0]].elem_var_data[j].name)
                    keep_booleans.append(True)
            elem_table = elem_table[:, keep_booleans]
            variable_names = [name for name, boolean in zip(
                variable_names, keep_booleans) if boolean]
            keep_indices = np.nonzero(keep_booleans)[0]
            exo_out.put_elem_variable_names(variable_names, elem_table)
            for i, block in enumerate(self.blocks):
                for j, name in enumerate(variable_names):
                    if elem_table[i, j]:
                        for k, time in enumerate(self.time):
                            exo_out.set_elem_variable_values(block.id, j, k,
                                                             self.elem_vars[i].elem_var_data[keep_indices[j]].data[k, :])

        exo_out.close()

    def repack(self, q, modes=None):
        '''Repackages an exodus file as a linear combination of itself'''
        if modes is None:
            mode_indices = np.arange(len(self.time))
        else:
            mode_indices = modes

        if q.shape[0] != len(mode_indices):
            raise ValueError(
                'Number of rows in q must be equivalent to number of time steps in exodus file')

        out_fexo = copy.deepcopy(self)

        out_fexo.time = np.zeros((q.shape[1]), dtype='float')

        if self.global_vars is not None:
            for i, global_var in enumerate(self.global_vars):
                out_fexo.global_vars[i].data = (
                    global_var.data[np.newaxis, mode_indices] @ q).squeeze()

        if self.nodal_vars is not None:
            for i, nodal_var in enumerate(self.nodal_vars):
                out_fexo.nodal_vars[i].data = (nodal_var.data.T[:, mode_indices] @ q).T

        if self.elem_vars is not None:
            for i, block in enumerate(self.elem_vars):
                for j, elem_vars in enumerate(block.elem_var_data):
                    if elem_vars is not None:
                        out_fexo.elem_vars[i].elem_var_data[j].data = (
                            elem_vars.data.T[:, mode_indices] @ q).T

        return out_fexo

    def get_block_surface(self, block_id, keep_midside_nodes=False, warn=True):
        '''Gets the node indices and element connectivity of surface elements

        This function "skins" the element block, returning a list of node
        indices and a surface connectivity matrix.

        Parameters
        ----------
        block_id : int
            The ID number of the block that will be skinned.
        keep_midside_nodes : bool
            Specifies whether or not to keep midside nodes in the surface mesh.
            Default is False.
        warn : bool
            Specifies whether or not to warn the user if the block ID doesn't
            have a skinning method defined for its element type.  Default is
            True.

        Returns
        -------
        element_block_information : list
            A list of tuples of element information.  These data are
            element_type, node_indices, block_surface_connectivity, and
            block_surface_original_elements.  The element_type is a string
            representing the new block element type ('quad4','tri3',etc.).  The
            node_indices can be used as an index into the coordinate or nodal
            variable arrays to select nodes corresponding to this block. The
            block_surface_connectivity represents the connectivity array of the
            surface faces of the block.  Values in this array correspond to
            indices into the node_indices array.  To recover the connectivity
            array in the original node indices of the exodus file, it can be
            passed through the node_indices as
            node_indices[block_surface_connectivity].  The
            block_surface_original_elements array shows the original element
            indices of the block that each surface came from.  This can be
            used to map element variables to the new surface mesh.  This list
            will normally be length 1 unless an element type is processed that
            has two different surface elements in it (e.g. wedges have
            tris and quads)
        '''
        block_index = [block.id for block in self.blocks].index(block_id)
        elem_type = self.blocks[block_index].elem_type
        connectivity = self.blocks[block_index].connectivity
        face_connectivity_array = face_connectivity(elem_type, keep_midside_nodes)
        element_block_information = []
        for reduced_elem_type, reduced_connectivity in face_connectivity_array:
            if reduced_elem_type is None:
                if warn:
                    print('Warning: Element Type {:} has no defined face reduction.  Passing connectivity unchanged.'.format(
                        elem_type))
                block_face_connectivity = connectivity
                block_elem_type = elem_type
                block_face_original_elements = np.arange(connectivity.shape[0])
            else:
                block_face_connectivity = connectivity[:,
                                                       reduced_connectivity].reshape(-1, reduced_connectivity.shape[-1])
                block_face_original_elements = np.repeat(np.arange(connectivity.shape[0]),
                                                         reduced_connectivity.shape[0])
                block_elem_type = reduced_elem_type
                # Remove faces that are duplicates
                (unique_rows, unique_row_indices, unique_inverse, unique_counts) = np.unique(np.sort(
                    block_face_connectivity, axis=1), return_index=True, return_inverse=True, return_counts=True, axis=0)
                original_unique_counts = unique_counts[unique_inverse]
                nondupe_faces = original_unique_counts == 1
                block_face_connectivity = block_face_connectivity[nondupe_faces, :]
                block_face_original_elements = block_face_original_elements[nondupe_faces]

            node_indices = np.unique(block_face_connectivity)
            node_map = np.zeros(self.nodes.coordinates.shape[0], dtype=int)
            node_map[node_indices] = np.arange(len(node_indices))
            block_face_connectivity = node_map[block_face_connectivity]
            element_block_information.append((block_elem_type,
                                              node_indices,
                                              block_face_connectivity,
                                              block_face_original_elements))
        return element_block_information

    def triangulate_surface_mesh(self):
        '''Triangulate a surface mesh for plotting patches

        This function generates a triangle mesh for each block in the model if
        it can.  If there are more than 3 nodes per element in a block, and the
        triangulation scheme hasn't been defined in
        pyexodus.mesh_triangulation_array, it will be skipped.

        Parameters
        ----------
        None

        Returns
        -------
        triangulated_mesh_info : list
            A list of tuples containing block id, node_indices, triangulated
            connectivity, and original block elements
        '''
        triangulated_mesh_info = []
        for block in self.blocks:
            surface_mesh_info = self.get_block_surface(block.id, warn=False)
            for elem_type, node_indices, connectivity, original_elements in surface_mesh_info:
                triangulation_scheme = mesh_triangulation_array(elem_type)
                if triangulation_scheme is None:
                    if connectivity.shape[-1] > 3:
                        print('Warning: More than 3 ({:}) nodes per element in block {:}, this block will not be triangulated'.format(
                            connectivity.shape[-1], block.id))
                    else:
                        triangulated_mesh_info.append(
                            (block.id, node_indices, connectivity, original_elements))
                else:
                    triangulated_mesh_info.append(
                        (block.id,
                         node_indices,
                         connectivity[:,
                                      triangulation_scheme].reshape(-1, triangulation_scheme.shape[-1]),
                         np.repeat(original_elements, triangulation_scheme.shape[0])))
        return triangulated_mesh_info

    # def plot_mesh(self,surface_kwargs={'color':(0,0,1)},
    #              bar_kwargs = {'color':(0,1,0),'tube_radius':None},
    #              point_kwargs = {'color':(1,0,0),'scale_factor':0.1},
    #              plot_surfaces = True, plot_bars = True, plot_points = True,
    #              plot_edges = False):
    #     '''Skins, triangulates, and plots a 3D representation of the mesh.

    #     Parameters
    #     ----------
    #     mesh_kwargs : dict
    #         A dictionary of keyword arguments for the rendering of surface
    #         patches.
    #     bar_kwargs : dict
    #         A dictionary of keyword arguments for the rendering of lines (bar,
    #         beam elements)
    #     point_kwargs : dict
    #         A dictionary of keyword arguments for the rendering of points
    #         (sphere elements)
    #     show : bool
    #         A flag to specify whether or not to show the window.

    #     Returns
    #     -------
    #     window : GLViewWidget
    #         A reference to the view that the 3D content is plotted in.
    #     meshes : list
    #         A list of the mesh objects that have been added to the view.

    #     '''
    #     if mlab is None:
    #         raise ModuleNotFoundError('Mayavi not installed!')
    #     triangulated_mesh_info = self.triangulate_surface_mesh()
    #     meshes = []
    #     for block,node_indices,faces,original_elements in triangulated_mesh_info:
    #         if faces.shape[1] == 3:
    #             if plot_surfaces:
    #                 meshes.append(mlab.triangular_mesh(
    #                         *(self.nodes.coordinates.T[:,node_indices]),faces,
    #                               **surface_kwargs))
    #                 if plot_edges:
    #                     meshes.append(mlab.triangular_mesh(
    #                         *(self.nodes.coordinates.T[:,node_indices]),faces,
    #                               representation='wireframe',color=(0,0,0)))
    #         elif faces.shape[1] == 2:
    #             if plot_bars:
    #                 line_items = []
    #                 xs,ys,zs = self.nodes.coordinates.T[:,node_indices][:,faces]
    #                 for x,y,z in zip(xs,ys,zs):
    #                     line_items.append(mlab.plot3d(x,y,z,**bar_kwargs))
    #                 meshes.append(line_items)
    #         elif faces.shape[1] == 1:
    #             if plot_points:
    #                 meshes.append(mlab.points3d(*(self.nodes.coordinates.T[:,node_indices]),
    #                                             **point_kwargs))
    #     return meshes

    def reduce_to_surfaces(self,*args,**kwargs):
        return reduce_exodus_to_surfaces(self,*args,**kwargs)

    def extract_sharp_edges(self,*args,**kwargs):
        return extract_sharp_edges(self, *args,**kwargs)


def reduce_exodus_to_surfaces(fexo, blocks_to_transform=None, variables_to_transform=None, keep_midside_nodes=False, verbose=False):
    '''Convert exodus finite element models to surface elements

    This function converts specified volume meshes in an exodus file to surface
    meshes to ease computational complexity.

    Parameters
    ----------
    fexo : ExodusInMemory object or Exodus object
        fexo should be an in-memory representation of the finite element mesh
    blocks_to_transform : iterable
        blocks_to_transform includes all of the blocks that will be included
        in the output model
    variables_to_transform : iterable
        variables_to_transform is a case-insensitive list of all variable names
        that will be kept in the final model
    keep_midside_nodes : bool
        keep_midside_nodes specifies whether or not to transform quadratic
        elements into linear elements, discarding any non-corner nodes.

    Returns
    -------
    fexo_out : ExodusInMemory object
        An equivalent fexo reduced to the surface geometry.

    Notes
    -----
    TO ADD MORE ELEMENT TYPES, we need to define a volume element name that
    we will find in the exodus file, for example 'tetra4' or 'hex20', the
    number of nodes per face on the element, and a connectivity matrix that
    specifies how the faces are made from the nodes of the element.  For
    example a hex8 element has the following structure::

              8--------------7
             /|             /|
            / |            / |
           /  |           /  |
          5--------------6   |
          |   4----------|---3
          |  /           |  /
          | /            | /
          |/             |/
          1--------------2

    So the 6 faces are as follows::

       1,2,6,5
       2,3,7,6
       3,4,8,7
       4,1,5,8
       4,3,2,1
       5,6,7,8

    We create the element face connectivity by simply mashing these together
    one after another like so: 1,2,6,5,2,3,7,6,3,4,8,7,4,1,5,8,4,3,2,1,5,6,7,8
    We lastly need to specify what we are turning the element into, in this
    case, quad4.
    So we add an entry to the reduce_element_types,
    reduce_element_nodes_per_face, reduce_element_face, and
    reduce_element_substitute_type variables containing this information
    hex8::

        reduce_element_types{4} = 'hex8'
        reduce_element_nodes_per_face{4} = 4
        reduce_element_face{4} = [1,2,6,5,2,3,7,6,3,4,8,7,4,1,5,8,4,3,2,1,5,6,7,8]
        reduce_element_substitute_type{4} = 'quad4'

    To see what other elements look like, see page 18 of the exodusII manual:
    http://prod.sandia.gov/techlib/access-control.cgi/1992/922137.pdf
    '''

    if isinstance(fexo, Exodus):
        exo_in_mem = False
    else:
        exo_in_mem = True

    # If blocks_to_transform is not specified, keep all blocks
    if blocks_to_transform is None:
        if exo_in_mem:
            blocks_to_transform = [block.id for block in fexo.blocks]
        else:
            blocks_to_transform = fexo.get_elem_blk_ids()

    # if variables_to_keep is not specified, keep all variables
    if variables_to_transform is None:
        if exo_in_mem:
            try:
                node_variables_to_transform = np.arange(len(fexo.nodal_vars))
            except TypeError:
                node_variables_to_transform = []
            try:
                elem_variables_to_transform = np.arange(len(fexo.elem_vars[0].elem_var_data))
            except TypeError:
                elem_variables_to_transform = []
        else:
            try:
                node_variables_to_transform = np.arange(fexo.num_node_variables)
            except ExodusError:
                node_variables_to_transform = []
            try:
                elem_variables_to_transform = np.arange(fexo.num_elem_variables)
            except ExodusError:
                elem_variables_to_transform = []
    else:
        if exo_in_mem:
            try:
                node_variables_to_transform = np.array([i for i, nodal_variable in enumerate(
                    fexo.nodal_vars) if nodal_variable.lower() in [var.lower() for var in variables_to_transform]])
            except TypeError:
                node_variables_to_transform = []
            # Need to do extra processing for element variable names because
            # not all variable names are in all blocks (might be none)
            try:
                elem_var_names = []
                for i in range(len(fexo.elem_vars[0].elem_var_data)):
                    var_name = None
                    for elem_var in fexo.elem_vars:
                        try:
                            var_name = elem_var.elem_var_data[i].name
                            break
                        except AttributeError:
                            continue
                    elem_var_names.append(var_name)
                elem_variables_to_transform = np.array([i for i, name in enumerate(
                    elem_var_names) if name.lower() in [var.lower() for var in variables_to_transform]])
            except TypeError:
                elem_variables_to_transform = []
        else:
            try:
                node_variables_to_transform = np.array([i for i, name in enumerate(
                    fexo.get_node_variable_names()) if name.lower() in [var.lower() for var in variables_to_transform]])
            except ExodusError:
                node_variables_to_transform = []
            try:
                elem_variables_to_transform = np.array([i for i, name in enumerate(
                    fexo.get_elem_variable_names()) if name.lower() in [var.lower() for var in variables_to_transform]])
            except ExodusError:
                elem_variables_to_transform = []
        if verbose:
            print('Keeping {:} nodal variables and {:} element variables'.format(
                len(node_variables_to_transform), len(elem_variables_to_transform)))

    # Define element types and what we will do to reduce them
    reduced_element_types = []
    reduced_element_nodes_per_face = []
    reduced_element_face_connectivity = []
    reduced_element_substitute_type = []

    # 10-node tetrahedral
    reduced_element_types.append('tetra10')
    if keep_midside_nodes:
        reduced_element_nodes_per_face.append(6)
        reduced_element_face_connectivity.append([0, 1, 3, 4, 8, 7,
                                                  1, 2, 3, 5, 9, 8,
                                                  2, 0, 3, 6, 7, 9,
                                                  0, 2, 1, 6, 5, 4])
        reduced_element_substitute_type.append('tri6')
    else:
        reduced_element_nodes_per_face.append(3)
        reduced_element_face_connectivity.append([0, 1, 3,
                                                  1, 2, 3,
                                                  2, 0, 3,
                                                  0, 2, 1])
        reduced_element_substitute_type.append('tri3')
    # 4-node tetrahedral
    reduced_element_types.append('tetra4')
    reduced_element_nodes_per_face.append(3)
    reduced_element_face_connectivity.append([0, 1, 3,
                                              1, 2, 3,
                                              2, 0, 3,
                                              0, 2, 1])
    reduced_element_substitute_type.append('tri3')
    # 20-node hexahedral
    reduced_element_types.append('hex20')
    if keep_midside_nodes:
        reduced_element_nodes_per_face.append(8)
        reduced_element_face_connectivity.append([0, 1, 5, 4, 8, 13, 16, 12,
                                                  1, 2, 6, 5, 9, 14, 17, 13,
                                                  2, 3, 7, 6, 10, 15, 18, 14,
                                                  3, 0, 4, 7, 11, 12, 19, 15,
                                                  3, 2, 1, 0, 11, 8, 9, 10,
                                                  4, 5, 6, 7, 16, 17, 18, 19])
        reduced_element_substitute_type.append('quad8')
    else:
        reduced_element_nodes_per_face.append(4)
        reduced_element_face_connectivity.append([0, 1, 5, 4,
                                                  1, 2, 6, 5,
                                                  2, 3, 7, 6,
                                                  3, 0, 4, 7,
                                                  3, 2, 1, 0,
                                                  4, 5, 6, 7])
        reduced_element_substitute_type.append('quad4')
    # 8-node hexahedral
    reduced_element_types.append('hex8')
    reduced_element_nodes_per_face.append(4)
    reduced_element_face_connectivity.append([0, 1, 5, 4,
                                              1, 2, 6, 5,
                                              2, 3, 7, 6,
                                              3, 0, 4, 7,
                                              3, 2, 1, 0,
                                              4, 5, 6, 7])
    reduced_element_substitute_type.append('quad4')
    reduced_element_types.append('hex')
    reduced_element_nodes_per_face.append(4)
    reduced_element_face_connectivity.append([0, 1, 5, 4,
                                              1, 2, 6, 5,
                                              2, 3, 7, 6,
                                              3, 0, 4, 7,
                                              3, 2, 1, 0,
                                              4, 5, 6, 7])
    reduced_element_substitute_type.append('quad4')

    # Keep track of which nodes are used in the final model
    if exo_in_mem:
        keep_nodes = np.zeros(fexo.nodes.coordinates.shape[0], dtype=bool)
    else:
        keep_nodes = np.zeros(fexo.num_nodes, dtype=bool)

    # Create the output variable
    fexo_out = ExodusInMemory()
    # Set the same times
    if exo_in_mem:
        fexo_out.time = copy.deepcopy(fexo.time)
    else:
        fexo_out.time = fexo.get_times().data

    # Since we are just visualizing, eliminate exodus features that we don't
    # care about
    fexo_out.element_map = None
    fexo_out.sidesets = None
    fexo_out.nodesets = None

    if exo_in_mem:
        block_inds_to_transform = [(block_index, block.id) for block_index, block in enumerate(
            fexo.blocks) if block.id in blocks_to_transform]
    else:
        block_inds_to_transform = [(block_index, block_id) for block_index, block_id in enumerate(
            fexo.get_elem_blk_ids()) if block_id in blocks_to_transform]

    # If the exodus file has element variables, we are only keeping the ones
    # corresponding to the blocks we are keeping
    if exo_in_mem:
        has_elem_vars = (fexo.elem_vars is not None) and len(node_variables_to_transform) > 0
    else:
        try:
            has_elem_vars = fexo.num_elem_variables > 0 and len(elem_variables_to_transform) > 0
        except ExodusError:
            has_elem_vars = False
    if has_elem_vars:
        fexo_out.elem_vars = []

    # Now we loop through each block to reduce the elements to faces
    for block_index, block_id in block_inds_to_transform:
        # Determine the element type
        if exo_in_mem:
            elem_type = copy.deepcopy(fexo.blocks[block_index].elem_type.lower())
            connectivity = copy.deepcopy(fexo.blocks[block_index].connectivity)
        else:
            elem_type = copy.deepcopy(fexo.get_elem_blk_info(block_id)[0].lower())
            connectivity = copy.deepcopy(fexo.get_elem_connectivity(block_id).data)
        if verbose:
            print('Analyzing Block {}, type {}'.format(block_id, elem_type))
        if elem_type in reduced_element_types:
            block_type_ind = reduced_element_types.index(elem_type)
            # If the element type IS in our list of types to reduce, then we
            # will reduce to only surface nodes and elements.
            # Here we are computing all the faces in the block and creating a
            # face connectivity matrix (nfaces x nnodes per face) from an
            # element connectivity matrix (nelems x nnodes per element).  It is
            # a complicated one-liner but boy is it fast.  The basic steps are:
            # 1. Pass the element reduce_element_face vector in as the column
            # indices of the element connectivity matrix, which selects the face
            # nodes of all faces, making each row of the block connectivity
            #  matrix contain all faces
            # block.connectivity[:,reduced_element_face_connectivity[block_type_ind]]
            # 2. Since each row now contains a number of faces, we want to
            # split it out into each face using the reshape command with the
            # number of nodes per face.  After we do this, we have a connectivity
            # matrix where each row is a face of the element and the columns are
            # the nodes in that face.
            face_connectivity = connectivity[:, reduced_element_face_connectivity[block_type_ind]].reshape(
                (-1, reduced_element_nodes_per_face[block_type_ind]))
            # We will create a list that ties each face back to the original
            # element.  This is used to map the element variables that were
            # defined for the volume elements to the newly created surface
            # elements.  Basically this vector will be increasing by 1, with
            # each number repeated for the number of faces in the element.  For
            # example if the elements are tet4s with 4 faces per element, the
            # vector will look like
            # [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, ... ]
            face_original_elements = np.repeat(np.arange(connectivity.shape[0]), len(
                reduced_element_face_connectivity[block_type_ind]) / reduced_element_nodes_per_face[block_type_ind])
            # To find EXTERIOR faces, we want to look for faces that only exist
            # once.  If two elements have the same face, then that face is
            # necessarily on the interior of the block.  Because the ordering
            # of the nodes is somewhat arbitrary, we first sort the nodes so
            # they are in ascending order for each face, then we can directly
            # compare the rows because two rows consisting of the same nodes
            # will look the same if they are sorted.  This gives us the unique
            # rows (but we don't really care about them because they are
            # sorted, so we ditch the first output argument), the unique row
            # indices, which is a list of the first time that index appeared
            # for each unique entry, and a unique row mapping, which is the
            # location each row in the original face connectivity matrix ends
            # up in the unique set of rows (see help unique for more info)
            (unique_rows, unique_row_indices, unique_inverse, unique_counts) = np.unique(np.sort(
                face_connectivity, axis=1), return_index=True, return_inverse=True, return_counts=True, axis=0)
            # We can compute the values that are duplicates by using the optional
            # output argumenst of unique.  The counts variable will enable us to
            # see which values were repeated, and the unique_inverse will allow us
            # to reconstruct the original array from the unique array.
            original_unique_counts = unique_counts[unique_inverse]
            # The non-duplicated faces are then the ones that only exist once in
            # the array
            nondupe_faces = original_unique_counts == 1
            # We then reduce the face connectivity matrix, keeping only the faces
            # that are not duplicated.
            face_connectivity = face_connectivity[nondupe_faces, :]
            # We also reduce our vector that ties the faces back to the original
            # values to keep it consistent with the face connectivity matrix
            face_original_elements = face_original_elements[nondupe_faces]
            # Now we need to also reduce the element variables in the same way,
            # mapping the original elements variables to the faces that correspond
            # to that element.
            new_elem_type = reduced_element_substitute_type[block_type_ind]
        else:
            # If we don't reduce,
            if verbose:
                print('  No Reduction')
            face_connectivity = connectivity
            face_original_elements = np.arange(connectivity.shape[0])
            new_elem_type = elem_type
        # We then update the connectivity in the block to make it match our new
        # face connectivity
        blk_dict = {}
        blk_dict['id'] = block_id
        blk_dict['connectivity'] = face_connectivity
        blk_dict['elem_type'] = new_elem_type
        blk_dict['name'] = 'block_{:}'.format(block_id)
        blk_dict['status'] = 1
        blk_dict['attributes'] = None
        fexo_out.blocks.append(subfield(**blk_dict))
        if verbose:
            print('  Converted to {}'.format(blk_dict['elem_type']))
        # The nodes in the block are now all the unique nodes in the face
        # connectivity matrix, so we make sure to keep those nodes in the final
        # set of nodes.
        block_nodes = np.unique(face_connectivity)
        keep_nodes[block_nodes] = True

        # Update element variables if necessary
        if has_elem_vars:
            var_names = []
            var_data = []
            if exo_in_mem:
                # Select the correct variable
                face_elem_vars = fexo.elem_vars[[var.id for var in fexo.elem_vars].index(block_id)]
                for elem_index, elem_var_data in enumerate(face_elem_vars.elem_var_data):
                    # Check if we need to skip it
                    if elem_index not in elem_variables_to_transform:
                        print('  Skipping Element Variable {:}'.format(elem_index))
                        continue
                    # If the variable doesn't exist it will be None (for example a
                    # volume quantity on a surface element).  We only operate if
                    # the variable exists (is not None)
                    if elem_var_data is not None:
                        # We use the face's original elements vector to map the
                        # element data to the face connectivity matrix
                        if verbose:
                            print('  Processing Element Variable {:} for block {:}'.format(
                                elem_var_data.name, block_id))
                        var_data.append(elem_var_data.data[:, face_original_elements])
                        var_names.append(elem_var_data.name)
                    else:
                        if verbose:
                            print('  Element Variable {:} not defined for block {:}'.format(
                                elem_index, block_id))
                        var_data.append(None)
                        var_names.append(None)
            else:
                # Get the element truth table
                elem_truth_table = fexo.get_elem_variable_table()
                # Get the element variable names
                elem_var_names = fexo.get_elem_variable_names()
                # Loop through and pick element variables that exist in the truth table
                for elem_index, elem_name in enumerate(elem_var_names):
                    # Check if we need to skip it
                    if elem_index not in elem_variables_to_transform:
                        if verbose:
                            print('  Skipping Element Variable {:}'.format(elem_name))
                        continue
                    if elem_truth_table[block_index, elem_index] == 0:
                        if verbose:
                            print('  Element Variable {:} not defined for block {:}'.format(
                                elem_name, block_id))
                        var_data.append(None)
                        var_names.append(None)
                    else:
                        # We use the face's original elements vector to map the
                        # element data to the face connectivity matrix
                        if verbose:
                            print('  Processing Element Variable {:} for block {:}'.format(
                                elem_name, block_id))
                        var_data.append(fexo.get_elem_variable_values(
                            block_id, elem_index).data[:, face_original_elements])
                        var_names.append(elem_name)
            # Now add the element variable
            if verbose:
                print('  Adding Element Variables to output exodus file')
            blk_dict = {}
            blk_dict['id'] = block_id
            blk_dict['elem_var_data'] = []
            for name, data in zip(var_names, var_data):
                if name is None or data is None:
                    blk_dict['elem_var_data'].append(None)
                else:
                    elem_var_dict = {}
                    elem_var_dict['name'] = name
                    elem_var_dict['data'] = data
                    blk_dict['elem_var_data'].append(subfield(**elem_var_dict))
            fexo_out.elem_vars.append(subfield(**blk_dict))

    # We have reduced the face connectivity, so now we need to reduce the nodes
    # as well.  We create a node map vector that maps the initial node indices
    # to the new node indices
    # i.e. the new_node_index = node_map[old_node_index]
    # We will need to do this because once we reduce the nodes, we will need to
    # re-update the element connectivity matrices with the updated node numbers
    # since the element connectivity matrix refers to the node index, not a
    # node number.  By eliminating nodes, we are effectively renumbering the
    # nodes.
    if verbose:
        print('Analyzing Nodes')
    if exo_in_mem:
        original_nodes = np.arange(len(keep_nodes)) + \
            1 if fexo.nodes.node_num_map is None else fexo.nodes.node_num_map
    else:
        original_nodes = fexo.get_node_num_map()
    fexo_out.nodes.node_num_map = original_nodes[keep_nodes]
    node_map = np.zeros(keep_nodes.shape, dtype=int)
    node_map[keep_nodes] = np.arange(sum(keep_nodes))
    # Reduce the nodes, keeping only the ones where keep_nodes is true
    if verbose:
        print('  Reducing Coordinates')
    if exo_in_mem:
        fexo_out.nodes.coordinates = copy.deepcopy(fexo.nodes.coordinates[keep_nodes, :])
    else:
        fexo_out.nodes.coordinates = fexo.get_coords()[:, keep_nodes].T
    # We do the same thing to the nodal variables if they exist
    if verbose:
        print('  Reducing Nodal Variables')
    if exo_in_mem:
        if len(node_variables_to_transform) > 0:
            fexo_out.nodal_vars = []
            for node_var_index, node_var in enumerate(fexo.nodal_vars):
                if node_var_index not in node_variables_to_transform:
                    if verbose:
                        print('  Skipping Nodal Variable {:}'.format(node_var.name))
                    continue
                node_dict = {}
                node_dict['name'] = node_var.name
                node_dict['data'] = copy.deepcopy(node_var.data[:, keep_nodes])
                fexo_out.nodal_vars.append(subfield(**node_dict))
    else:
        try:
            if len(node_variables_to_transform) > 0:
                fexo_out.nodal_vars = []
                node_vars = fexo.get_node_variable_names()
                for node_var_index, node_var_name in enumerate(node_vars):
                    if node_var_index not in node_variables_to_transform:
                        if verbose:
                            print('  Skipping Nodal Variable {:}'.format(node_var_name))
                        continue
                    if verbose:
                        print('  Variable {:}: {:}'.format(node_var_index, node_var_name))
                    node_dict = {}
                    node_dict['name'] = node_var_name
                    node_dict['data'] = np.empty((len(fexo_out.time), sum(keep_nodes)))
                    for i, time in enumerate(fexo_out.time):
                        if verbose:
                            print('    Step {:}'.format(i))
                        node_dict['data'][i, :] = fexo.get_node_variable_values(
                            node_var_index, step=i).data[keep_nodes]
                    fexo_out.nodal_vars.append(subfield(**node_dict))
        except ExodusError:
            fexo_out.nodal_vars = None

    # Finally we go through each element block and map the old node indices to
    # the new node indices
    for block in fexo_out.blocks:
        block.connectivity = node_map[block.connectivity]

    # Return our new fexo
    return fexo_out


def extract_sharp_edges(exo, edge_threshold=60, **kwargs):
    dot_threshold = np.cos(edge_threshold * np.pi / 180)
    surface_fexo = reduce_exodus_to_surfaces(exo, **kwargs)
    edge_fexo = copy.deepcopy(surface_fexo)
    keep_nodes = np.zeros(surface_fexo.nodes.coordinates.shape[0], dtype=bool)
    for block_ind, block in enumerate(surface_fexo.blocks):
        connectivity = block.connectivity
        if connectivity.shape[-1] < 3 or 'bar' in block.elem_type.lower() or 'beam' in block.elem_type.lower():
            edge_fexo.blocks[block_ind].id = None
            continue
        print('Reducing Block {:}'.format(block.id))
        edge_connectivity = np.concatenate((connectivity[..., np.newaxis],
                                            np.roll(connectivity, -1, -1)[..., np.newaxis]), -1).reshape(-1, 2)
        edge_original_elements = np.repeat(np.arange(connectivity.shape[0]), connectivity.shape[-1])
        (unique_rows, unique_row_indices, unique_inverse, unique_counts) = np.unique(np.sort(
            edge_connectivity, axis=1), return_index=True, return_inverse=True, return_counts=True, axis=0)
        # Now we go through each edge and compute the angle between the two original faces
        keep_edges = np.zeros(unique_rows.shape[0], dtype=bool)
        for i, (edge, count) in enumerate(zip(unique_rows, unique_counts)):
            if count != 2:
                keep_edges[i] = True
            else:
                # Find faces that the edges are in
                edge_indices = np.nonzero(unique_inverse == i)[0]
                face_indices = edge_original_elements[edge_indices]
                face_connectivities = connectivity[face_indices]
                node_coords = surface_fexo.nodes.coordinates[face_connectivities[:, :3], :]
                vectors_a = node_coords[:, 2] - node_coords[:, 0]
                vectors_b = node_coords[:, 1] - node_coords[:, 0]
                normals = np.cross(vectors_a, vectors_b)
                normals /= np.linalg.norm(normals, axis=-1)[:, np.newaxis]
                if np.dot(*normals) < dot_threshold:
                    keep_edges[i] = True
        print('  Kept {:} of {:} edges'.format(keep_edges.sum(), keep_edges.size))
        edge_connectivity = unique_rows[keep_edges]
        edge_fexo.blocks[block_ind].connectivity = edge_connectivity
        edge_fexo.blocks[block_ind].elem_type = 'bar'
        block_nodes = np.unique(edge_connectivity)
        keep_nodes[block_nodes] = True
    node_map = np.zeros(keep_nodes.shape, dtype=int)
    node_map[keep_nodes] = np.arange(sum(keep_nodes))
    edge_fexo.nodes.coordinates = edge_fexo.nodes.coordinates[keep_nodes, :]
    for node_var_index, node_var in enumerate(edge_fexo.nodal_vars):
        node_var.data = node_var.data[:, keep_nodes]
    edge_fexo.elem_vars = None
    edge_fexo.blocks = [block for block in edge_fexo.blocks if block.id is not None]
    for block in edge_fexo.blocks:
        block.connectivity = node_map[block.connectivity]
    edge_fexo.nodes.node_num_map = surface_fexo.nodes.node_num_map[keep_nodes]
    return edge_fexo
