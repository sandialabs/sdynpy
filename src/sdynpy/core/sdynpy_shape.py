# -*- coding: utf-8 -*-
"""
Objects and procedures to handle operations on test or model shapes

Shapes are generally defined as a set of coordinates or degrees of freedom and
the respective displacements at each of those degrees of freedom.

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
import scipy.optimize as opt
from . import sdynpy_colors
from . import sdynpy_coordinate
from . import sdynpy_array
from . import sdynpy_data
from . import sdynpy_system
from ..signal_processing.sdynpy_integration import integrate_MCK
from ..signal_processing.sdynpy_correlation import mac as mac_corr, matrix_plot
from ..signal_processing.sdynpy_complex import collapse_complex_to_real
from ..signal_processing.sdynpy_rotation import unit_magnitude_constraint, quaternion_to_rotation_matrix
from ..fem.sdynpy_exodus import Exodus
from ..fem.sdynpy_dof import by_condition_number, by_effective_independence
from ..core.sdynpy_matrix import matrix
from copy import deepcopy
import pandas as pd
from qtpy.QtWidgets import QDialog, QTableWidget, QDialogButtonBox, QVBoxLayout, QTableWidgetItem, QAbstractItemView
from qtpy.QtCore import Qt
import time


class ShapeArray(sdynpy_array.SdynpyArray):
    """Shape information specifying displacements at nodes.

    Use the shape_array helper function to create the array.
        """
    @staticmethod
    def real_data_dtype(ndof):
        """
        Data type of the underlying numpy structured array for real shapes

        Parameters
        ----------
        ndof : int
            Number of degrees of freedom in the shape array

        Returns
        -------
        list
            Numpy dtype that can be passed into any of the numpy array
            constructors

        """
        return [
            ('frequency', 'float64'),
            ('damping', 'float64'),
            ('coordinate', sdynpy_coordinate.CoordinateArray.data_dtype, (ndof,)),
            ('shape_matrix', 'float64', (ndof,)),
            ('modal_mass', 'float64'),
            ('comment1', '<U80'),
            ('comment2', '<U80'),
            ('comment3', '<U80'),
            ('comment4', '<U80'),
            ('comment5', '<U80'),
        ]

    @staticmethod
    def complex_data_dtype(ndof):
        """
        Data type of the underlying numpy structured array for complex shapes

        Parameters
        ----------
        ndof : int
            Number of degrees of freedom in the shape array

        Returns
        -------
        list
            Numpy dtype that can be passed into any of the numpy array
            constructors

        """
        return [
            ('frequency', 'float64'),
            ('damping', 'float64'),
            ('coordinate', sdynpy_coordinate.CoordinateArray.data_dtype, (ndof,)),
            ('shape_matrix', 'complex128', (ndof,)),
            ('modal_mass', 'complex128'),
            ('comment1', '<U80'),
            ('comment2', '<U80'),
            ('comment3', '<U80'),
            ('comment4', '<U80'),
            ('comment5', '<U80'),
        ]

    def __new__(subtype, shape, ndof, shape_type='real', buffer=None, offset=0,
                strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        # Data Types
        #    'coordinates', - This will actually be an attribute
        #    'shape_matrix',
        #    'frequency',
        #    'damping',
        #    'modal_mass',
        #    'comment1',
        #    'comment2',
        #    'comment3',
        #    'comment4',
        #    'comment5',
        #    'numerator_type',
        #    'denominator_type',
        #    'load_case_number',
        #    'mode_number',
        #    'reference_coordinate',
        #    'response_coordinate',
        if shape_type == 'real':
            data_dtype = ShapeArray.real_data_dtype(ndof)
        elif shape_type == 'complex' or shape_type == 'imaginary' or shape_type == 'imag':
            data_dtype = ShapeArray.complex_data_dtype(ndof)
        obj = super(ShapeArray, subtype).__new__(subtype, shape, data_dtype,
                                                 buffer, offset, strides,
                                                 order)
        # Finally, we must return the newly created object:
        return obj

    def is_complex(self):
        """
        Returns true if the shape is a complex shape, False if shape is real

        Returns
        -------
        bool
            True if the shape is complex

        """
        return np.iscomplexobj(self.shape_matrix)

    def __repr__(self):
        string_out = '{:>8s}, {:>10s}, {:>10s}, {:>10s}\n'.format(
            'Index', 'Frequency', 'Damping', '# DoFs')
        if self.size == 0:
            string_out += '----------- Empty -------------\n'
        for key, val in self.ndenumerate():
            string_out += '{:>8s}, {:>10.4f}, {:>9.4f}%, {:>10d}\n'.format(
                str(key), val.frequency, val.damping * 100, np.prod(val.dtype['coordinate'].shape))
        return string_out

    # def __getattribute__(self,attr):
    #     return_obj = np.recarray.__getattribute__(self,attr)
    #     if attr == 'coordinate':
    #         return return_obj.view(sdynpy_coordinate.CoordinateArray)
    #     else:
    #         return return_obj

    def __getitem__(self, key):
        if type(key) is sdynpy_coordinate.CoordinateArray:
            # Get the coordinate array
            coordinate_array = self.coordinate
            single_shape_coordinate_array = coordinate_array[(
                0,) * (coordinate_array.ndim - 1) + (slice(None),)]
            # Now check if the coordinates are consistent across the arrays
            if not np.all((coordinate_array[..., :] == single_shape_coordinate_array)):
                # If they aren't, raise a value error
                raise ValueError(
                    'Shape array must have equivalent coordinates for all shapes to index by coordinate')
            # Otherwise we will do an intersection
            consistent_arrays, shape_indices, request_indices = np.intersect1d(
                abs(single_shape_coordinate_array), abs(key), assume_unique=False, return_indices=True)
            # Make sure that all of the keys are actually in the consistent array matrix
            if consistent_arrays.size != key.size:
                extra_keys = np.setdiff1d(abs(key), abs(single_shape_coordinate_array))
                if extra_keys.size == 0:
                    raise ValueError(
                        'Duplicate coordinate values requested.  Please ensure coordinate indices are unique.')
                raise ValueError(
                    'Not all indices in requested coordinate array exist in the shape\n{:}'.format(str(extra_keys)))
            # Handle sign flipping
            multiplications = key.flatten()[request_indices].sign(
            ) * single_shape_coordinate_array[shape_indices].sign()
            return_value = self.shape_matrix[..., shape_indices] * multiplications
            # Invert the indices to return the dofs in the correct order as specified in keys
            inverse_indices = np.zeros(request_indices.shape, dtype=int)
            inverse_indices[request_indices] = np.arange(len(request_indices))
            return_value = return_value[..., inverse_indices]
            # Reshape to the original coordinate array shape
            return_value = return_value.reshape(*return_value.shape[:-1], *key.shape)
            return return_value
        else:
            output = super().__getitem__(key)
            if isinstance(key, str) and key == 'coordinate':
                return output.view(sdynpy_coordinate.CoordinateArray)
            else:
                return output

    def __mul__(self, val):
        this = deepcopy(self)
        if isinstance(val, ShapeArray):
            val = val.shape_matrix
        this.shape_matrix *= val
        return this

    @property
    def ndof(self):
        """The number of degrees of freedom in the shape"""
        return self.dtype['coordinate'].shape[0]

    # @property
    # def U(self):
    #     if self.shape_matrix.ndim > 1:
    #         return self.shape_matrix.swapaxes(-2,-1)
    #     else:
    #         return self.shape_matrix[:,np.newaxis]

    # @U.setter
    # def U(self,value):
    #     raise AttributeError('Cannot set the U attribute directly.  Set the shape_matrix parameter instead')

    @property
    def modeshape(self):
        """The mode shape matrix with degrees of freedom as second to last axis"""
        if self.shape_matrix.ndim > 1:
            return self.shape_matrix.swapaxes(-2, -1)
        else:
            return self.shape_matrix[:, np.newaxis]

    @modeshape.setter
    def modeshape(self, value):
        value = np.array(value)
        if value.ndim > 1:
            self.shape_matrix = value.swapaxes(-2, -1)
        else:
            self.shape_matrix = value

    @staticmethod
    def from_unv(unv_data_dict, combine=True):
        """
        Load ShapeArrays from universal file data from read_unv

        Parameters
        ----------
        unv_data_dict : dict
            Dictionary containing data from read_unv
        combine : bool, optional
            If True, return as a single ShapeArray

        Raises
        ------
        ValueError
            Raised if an unknown data characteristic is provided

        Returns
        -------
        output_arrays : ShapeArray
            Shapes read from unv

        """
        try:
            datasets = unv_data_dict[55]
        except KeyError:
            if combine:
                return ShapeArray(0, 0)
            else:
                return []
        output_arrays = []
        for dataset in datasets:
            nodes = np.array([val for val in dataset.node_data_dictionary.keys()])
            if dataset.data_characteristic == 2:
                coordinates = sdynpy_coordinate.coordinate_array(
                    nodes[:, np.newaxis], np.array([1, 2, 3])).flatten()
            elif dataset.data_characteristic == 3:
                coordinates = sdynpy_coordinate.coordinate_array(
                    nodes[:, np.newaxis], np.array([1, 2, 3, 4, 5, 6])).flatten()
            else:
                raise ValueError('Cannot handle shapes besides 3Dof and 6Dof')
            data_matrix = np.array([dataset.node_data_dictionary[key] for key in nodes]).flatten()
            if dataset.analysis_type == 2:
                this_shape = shape_array(
                    coordinate=coordinates,
                    shape_matrix=data_matrix,
                    frequency=dataset.frequency,
                    damping=dataset.modal_viscous_damping,
                    modal_mass=dataset.modal_mass,
                    comment1=dataset.idline1,
                    comment2=dataset.idline2,
                    comment3=dataset.idline3,
                    comment4=dataset.idline4,
                    comment5=dataset.idline5,
                )
            elif dataset.analysis_type == 3:
                this_shape = shape_array(
                    coordinate=coordinates,
                    shape_matrix=data_matrix,
                    frequency=np.abs(dataset.eigenvalue)/(2*np.pi),
                    damping=-np.real(dataset.eigenvalue)/np.abs(dataset.eigenvalue),
                    modal_mass=dataset.modal_a,
                    comment1=dataset.idline1,
                    comment2=dataset.idline2,
                    comment3=dataset.idline3,
                    comment4=dataset.idline4,
                    comment5=dataset.idline5,
                )
            else:
                raise NotImplementedError('Analysis Type {:} for UFF Dataset 55 not implemented yet'.format(dataset.analysis_type))
            output_arrays.append(this_shape)
        if combine:
            output_arrays = np.concatenate([val[np.newaxis] for val in output_arrays])
        return output_arrays

    from_uff = from_unv

    # @classmethod
    # def from_exodus(cls,exo,x_disp = 'DispX',y_disp = 'DispY',z_disp = 'DispZ',timesteps = None):
    #     if isinstance(exo,Exodus):
    #         exo = exo.load_into_memory(close=False,variables = [x_disp,y_disp,z_disp],timesteps = None, blocks=[])
    #     node_ids = np.arange(exo.nodes.coordinates.shape[0])+1 if exo.nodes.node_num_map is None else exo.nodes.node_num_map
    #     x_var = [var for var in exo.nodal_vars if var.name.lower() == x_disp.lower()][0].data[slice(timesteps) if timesteps is None else timesteps]
    #     y_var = [var for var in exo.nodal_vars if var.name.lower() == y_disp.lower()][0].data[slice(timesteps) if timesteps is None else timesteps]
    #     z_var = [var for var in exo.nodal_vars if var.name.lower() == z_disp.lower()][0].data[slice(timesteps) if timesteps is None else timesteps]
    #     frequencies = exo.time[slice(timesteps) if timesteps is None else timesteps]
    #     shape_matrix = np.concatenate((x_var,y_var,z_var),axis=-1)
    #     coordinates = sdynpy_coordinate.coordinate_array(node_ids,np.array((1,2,3))[:,np.newaxis]).flatten()
    #     return shape_array(coordinates,shape_matrix,frequencies)

    @classmethod
    def from_exodus(cls, exo, x_disp='DispX', y_disp='DispY', z_disp='DispZ', x_rot=None, y_rot=None, z_rot=None, timesteps=None):
        """
        Reads shape data from displacements in an Exodus file

        Parameters
        ----------
        exo : Exodus or ExodusInMemory
            The exodus data from which shapes will be created.
        x_disp : str, optional
            String denoting the nodal variable in the exodus file from which
            the X-direction displacement should be read. The default is 'DispX'.
            Specify `None` if no x_disp is to be read.
        y_disp : str, optional
            String denoting the nodal variable in the exodus file from which
            the Y-direction displacement should be read. The default is 'DispY'.
            Specify `None` if no y_disp is to be read.
        z_disp : str, optional
            String denoting the nodal variable in the exodus file from which
            the Z-direction displacement should be read. The default is 'DispZ'.
            Specify `None` if no z_disp is to be read.
        x_rot : str, optional
            String denoting the nodal variable in the exodus file from which
            the X-direction rotation should be read. The default is `None` which
            results in the X-direction rotation not being read. Typically this
            would be set to 'RotX' if rotational values are desired.
        y_rot : str, optional
            String denoting the nodal variable in the exodus file from which
            the Y-direction rotation should be read. The default is `None` which
            results in the Y-direction rotation not being read. Typically this
            would be set to 'RotY' if rotational values are desired.
        z_rot : str, optional
            String denoting the nodal variable in the exodus file from which
            the Z-direction rotation should be read. The default is `None` which
            results in the Z-direction rotation not being read. Typically this
            would be set to 'RotZ' if rotational values are desired.
        timesteps : iterable, optional
            A list of timesteps from which data should be read. The default is
            `None`, which reads all timesteps.

        Returns
        -------
        ShapeArray
            Shape data from the exodus file

        """
        if isinstance(exo, Exodus):
            variables = [v for v in [x_disp, y_disp, z_disp, x_rot, y_rot, z_rot] if v is not None]
            exo = exo.load_into_memory(close=False, variables=variables, timesteps=None, blocks=[])
        node_ids = np.arange(
            exo.nodes.coordinates.shape[0]) + 1 if exo.nodes.node_num_map is None else exo.nodes.node_num_map

        shape_matrix = np.empty(
            [exo.time[slice(timesteps) if timesteps is None else timesteps].shape[0], 0])
        coord_nums = np.empty([0], dtype=int)

        variables = [x_disp, y_disp, z_disp, x_rot, y_rot, z_rot]
        for counter, variable in enumerate(variables):
            if variable is not None:
                var = [var for var in exo.nodal_vars if var.name.lower() == variable.lower(
                )][0].data[slice(timesteps) if timesteps is None else timesteps]
                shape_matrix = np.append(shape_matrix, var, axis=-1)
                coord_nums = np.append(coord_nums, [counter + 1])

        frequencies = exo.time[slice(timesteps) if timesteps is None else timesteps]
        coordinates = sdynpy_coordinate.coordinate_array(
            node_ids, coord_nums[:, np.newaxis]).flatten()

        return shape_array(coordinates, shape_matrix, frequencies)

    @classmethod
    def from_imat_struct(cls, imat_shp_struct):
        """
        Constructs a ShapeArray from an imat_shp class saved to a Matlab structure

        In IMAT, a structure can be created from an `imat_shp` by using the get()
        function.  This can then be saved to a .mat file and loaded using
        `scipy.io.loadmat`.  The output from loadmat can be passed into this function

        Parameters
        ----------
        imat_fem_struct : np.ndarray
            structure from loadmat containing data from an imat_shp

        Returns
        -------
        ShapeArray
            ShapeArray constructed from the data in the imat structure

        """
        frequency = imat_shp_struct['Frequency'][0, 0]
        nodes = imat_shp_struct['Node'][0, 0]
        nodes = nodes.reshape(nodes.shape[0], 1, *frequency.shape)
        doftype = np.array(imat_shp_struct['DOFType'][0, 0].tolist())
        doftype = doftype.reshape(*doftype.shape[:-1])
        directions = (np.array([1, 2, 3]) if np.all(
            doftype == '3DOF') else np.array(1, 2, 3, 4, 5, 6))
        directions = directions.reshape(1, -1, *([1] * (nodes.ndim - 2)))
        coordinates = np.moveaxis(sdynpy_coordinate.coordinate_array(nodes,
                                                                     directions
                                                                     ).reshape(-1, *frequency.shape), 0, -1)
        shape_matrix = np.moveaxis(imat_shp_struct['Shape'][0, 0].reshape(
            coordinates.shape[-1], *frequency.shape), 0, -1)
        comment_1 = np.array(imat_shp_struct['IDLine1'][0, 0].tolist())
        if comment_1.size > 0:
            comment_1 = comment_1.reshape(*comment_1.shape[:-1])
        else:
            comment_1 = np.zeros(comment_1.shape[:-1], dtype='<U1')
        comment_2 = np.array(imat_shp_struct['IDLine2'][0, 0].tolist())
        if comment_2.size > 0:
            comment_2 = comment_1.reshape(*comment_2.shape[:-1])
        else:
            comment_2 = np.zeros(comment_2.shape[:-1], dtype='<U1')
        comment_3 = np.array(imat_shp_struct['IDLine3'][0, 0].tolist())
        if comment_3.size > 0:
            comment_3 = comment_3.reshape(*comment_3.shape[:-1])
        else:
            comment_3 = np.zeros(comment_3.shape[:-1], dtype='<U1')
        comment_4 = np.array(imat_shp_struct['IDLine4'][0, 0].tolist())
        if comment_4.size > 0:
            comment_4 = comment_4.reshape(*comment_4.shape[:-1])
        else:
            comment_4 = np.zeros(comment_4.shape[:-1], dtype='<U1')
        comment_5 = np.array(imat_shp_struct['IDLine5'][0, 0].tolist())
        if comment_5.size > 0:
            comment_5 = comment_5.reshape(*comment_5.shape[:-1])
        else:
            comment_5 = np.zeros(comment_5.shape[:-1], dtype='<U1')
        modal_mass = imat_shp_struct['ModalMassReal'][0, 0] + \
            (0 if np.isrealobj(shape_matrix) else 1j * imat_shp_struct['ModalMassImag'][0, 0])
        damping = imat_shp_struct['Damping'][0, 0] if np.isrealobj(shape_matrix) else (
            imat_shp_struct['ModalDampingReal'][0, 0] + 1j * imat_shp_struct['ModalDampingImag'][0, 0])
        return shape_array(coordinates, shape_matrix, frequency, damping, modal_mass,
                           comment_1, comment_2, comment_3, comment_4, comment_5)

    def compute_frf(self, frequencies, responses=None, references=None, displacement_derivative=2):
        """
        Computes FRFs from shape data

        Parameters
        ----------
        frequencies : iterable
            A list of frequencies to compute the FRF at.
        responses : CoordinateArray, optional
            Degrees of freedom to use as responses. The default is to compute
            FRFs at all degrees of freedom in the shape.
        references : CoordinateArray, optional
            Degrees of freedom to use as references.  The default is to compute
            FRFs using the response degrees of freedom also as references.
        displacement_derivative : int, optional
            The derivative to use when computing the FRFs.  0 corresponds to
            displacement FRFs, 1 corresponds to velocity, and 2 corresponds to
            acceleration. The default is 2.

        Returns
        -------
        output_data : TransferFunctionArray
            A transfer function array containing the specified references and
            responses.

        """
        flat_self = self.flatten()
        if responses is None:
            all_coords = flat_self.coordinate
            dim = all_coords.ndim
            index = (0,) * (dim - 1)
            responses = all_coords[index]
        else:
            
            responses = responses.flatten()
        if references is None:
            references = responses
        else:
            references = references.flatten()
        damping_ratios = flat_self.damping
        angular_natural_frequencies = flat_self.frequency*2*np.pi
        angular_frequencies = frequencies*2*np.pi
        response_shape_matrix = flat_self[responses]  # nm x no
        reference_shape_matrix = flat_self[references]  # nm x ni
        modal_mass = flat_self.modal_mass
        if self.is_complex():
            poles = -damping_ratios*angular_natural_frequencies + 1j*np.sqrt(1-damping_ratios**2)*angular_natural_frequencies
            denominator = 1/(modal_mass*(1j*angular_frequencies[:, np.newaxis] - poles))  # nf x nm
            denominator_conj = 1/(modal_mass*(1j*angular_frequencies[:, np.newaxis] - poles.conjugate()))  # nf x nm
            frf_ordinate = (np.einsum('mo,mi,fm->oif', response_shape_matrix, reference_shape_matrix, denominator) +
                            np.einsum('mo,mi,fm->oif', response_shape_matrix.conjugate(), reference_shape_matrix.conjugate(), denominator_conj))
        else:
            denominator = 1/(modal_mass *
                             (-angular_frequencies[:, np.newaxis]**2
                              + angular_natural_frequencies**2
                              + 2j*damping_ratios*angular_frequencies[:, np.newaxis]*angular_natural_frequencies))

            frf_ordinate = np.einsum('mo,mi,fm->oif',
                                     response_shape_matrix,
                                     reference_shape_matrix,
                                     denominator,
                                     )
        # Modify for data type
        if displacement_derivative > 0:
            frf_ordinate *= (1j*angular_frequencies)**displacement_derivative
        # Now package into a sdynpy array
        output_data = sdynpy_data.data_array(sdynpy_data.FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                                             frequencies, frf_ordinate,
                                             sdynpy_coordinate.outer_product(responses, references))

        return output_data

    def reduce(self, nodelist_or_coordinate_array):
        """
        Reduces the shape to the degrees of freedom specified

        Parameters
        ----------
        nodelist_or_coordinate_array : iterable or CoordinateArray
            Consists of either a list of node ids or a CoordinateArray.  The
            ShapeArray will be reduced to only the degrees of freedom specified

        Returns
        -------
        ShapeArray
            A reduced ShapeArray

        """
        if isinstance(nodelist_or_coordinate_array, sdynpy_coordinate.CoordinateArray):
            coord_array = nodelist_or_coordinate_array
        else:
            coord_array = np.unique(self.coordinate)
            coord_array = coord_array[np.in1d(coord_array.node, nodelist_or_coordinate_array)]
        shape_matrix = self[coord_array.flatten()]
        return shape_array(coord_array.flatten(), shape_matrix, self.frequency, self.damping,
                           self.modal_mass, self.comment1, self.comment2, self.comment3,
                           self.comment4, self.comment5)

    def repack(self, coefficients):
        """
        Creates new shapes by linearly combining existing shapes

        Parameters
        ----------
        coefficients : np.ndarray
            coefficient matrix that will be multiplied by the shapes.  This
            should have number of rows equal to the number of shapes that are
            going to be combined.  The number of columns will specify the
            number of shapes that will result

        Returns
        -------
        ShapeArray
            ShapeArray consisting of linear combinations of the original
            ShapeArray

        """
        coordinates = np.unique(self.coordinate)
        new_shape = self.flatten()[coordinates].T @ coefficients
        return shape_array(coordinates, new_shape.T)

    def expand(self, initial_geometry, expansion_geometry, expansion_shapes,
               node_id_map=None, expansion_coordinates=None, return_coefficients=False):
        """
        Perform SEREP expansion on shape data

        Parameters
        ----------
        initial_geometry : Geometry
            The initial or "Test" Geometry, corresponding to the initial
            shapes (self)
        expansion_geometry : Geometry
            The expanded or "FEM" Geometry, corresponding to the shapes that
            will be expanded
        expansion_shapes : ShapeArray
            Shapes defined on the expanded geometry, which will be used
            to expand the initial shapes
        node_id_map : id_map, optional
            If the initial and expanded geometry or shapes do not have common
            node ids, an id_map can be specified to map the finite element
            node ids to test node ids.  The default is None, which means no
            mapping will occur, and the shapes have common id numbers.
        expansion_coordinates : CoordinateArray, optional
            Degrees of freedom in the test shapes to use in the expansion.  The
            default is None, which results in all degrees of freedom being used
            for expansion
        return_coefficients : bool, optional
            If True, the coefficients used in the expansion will be returned
            along with the expanded shapes.  The default is False.

        Returns
        -------
        ShapeArray
            The original shapes expanded to the expansion_geometry
        np.ndarray
            The coefficients used to perform the expansion, only returned if
            return_coefficients is True

        """
        expansion_shape_in_original_basis = expansion_shapes.transform_coordinate_system(
            expansion_geometry, initial_geometry, node_id_map)
        if expansion_coordinates is None:
            expansion_coordinates = np.intersect1d(
                self.coordinate, expansion_shape_in_original_basis.coordinate)
        expansion_shape_matrix = expansion_shape_in_original_basis[expansion_coordinates].T
        original_shape_matrix = self[expansion_coordinates].T
        coefficients = np.linalg.lstsq(expansion_shape_matrix, original_shape_matrix)[0]
        expanded_shapes = expansion_shapes.repack(coefficients)
        expanded_shapes.frequency = self.frequency
        expanded_shapes.damping = self.damping
        expanded_shapes.modal_mass = self.modal_mass
        expanded_shapes.comment1 = self.comment1
        expanded_shapes.comment2 = self.comment2
        expanded_shapes.comment3 = self.comment3
        expanded_shapes.comment4 = self.comment4
        expanded_shapes.comment5 = self.comment5
        if return_coefficients:
            return expanded_shapes, coefficients
        else:
            return expanded_shapes

    def transform_coordinate_system(self, original_geometry, new_geometry, node_id_map=None, rotations=False,
                                    missing_dofs_are_zero=False):
        """
        Performs coordinate system transformations on the shape

        Parameters
        ----------
        original_geometry : Geometry
            The Geometry in which the shapes are currently defined
        new_geometry : Geometry
            The Geometry in which the shapes are desired to be defined
        node_id_map : id_map, optional
            If the original and new geometries do not have common
            node ids, an id_map can be specified to map the original geometry
            node ids to new geometry node ids.  The default is None, which means no
            mapping will occur, and the geometries have common id numbers.
        rotations : bool, optional
            If True, also transform rotational degrees of freedom. The default
            is False.
        missing_dofs_are_zero : bool, optional
            If False, any degree of freedom required for the transformation that
            is not provided will result in a ValueError.  If True, these missing
            degrees of freedom will simply be appended to the original shape
            matrix as zeros.  Default is False.

        Returns
        -------
        ShapeArray
            A ShapeArray that can now be plotted with the new geometry

        """
        if node_id_map is not None:
            original_geometry = original_geometry.reduce(node_id_map.from_ids)
            original_geometry.node.id = node_id_map(original_geometry.node.id)
            self = self.reduce(node_id_map.from_ids)
            self.coordinate.node = node_id_map(self.coordinate.node)
        common_nodes = np.intersect1d(np.intersect1d(original_geometry.node.id, new_geometry.node.id),
                                      np.unique(self.coordinate.node))
        coordinates = sdynpy_coordinate.coordinate_array(
            common_nodes[:, np.newaxis], [1, 2, 3, 4, 5, 6] if rotations else [1, 2, 3])
        transform_from_original = original_geometry.global_deflection(coordinates)
        transform_to_new = new_geometry.global_deflection(coordinates)
        if missing_dofs_are_zero:
            # Find any coordinates that are not in shapes
            shape_coords = np.unique(abs(self.coordinate))
            coords_not_in_shape = coordinates[~np.isin(abs(coordinates), shape_coords)]
            # Create a new shape array and set the coefficients to zero
            append_shape_matrix = np.zeros(self.shape+(coords_not_in_shape.size,),
                                           dtype=self.shape_matrix.dtype)
            # Create a shape
            append_shape = shape_array(coords_not_in_shape, append_shape_matrix)
            # Append it
            self = ShapeArray.concatenate_dofs((self, append_shape))
        shape_matrix = self[coordinates].reshape(*self.shape, *coordinates.shape)
        new_shape_matrix = np.einsum('nij,nkj,...nk->...ni', transform_to_new,
                                     transform_from_original, shape_matrix)
        return shape_array(coordinates.flatten(), new_shape_matrix.reshape(*self.shape, -1), self.frequency, self.damping, self.modal_mass,
                           self.comment1, self.comment2, self.comment3, self.comment4, self.comment5)

    def reduce_for_comparison(self, comparison_shape, node_id_map=None):
        if node_id_map is not None:
            self = self.reduce(node_id_map.from_ids)
            self.coordinate.node = node_id_map(self.coordinate.node)
        common_dofs = np.intersect1d(abs(self.coordinate), abs(comparison_shape.coordinate))
        reduced_self = self.reduce(common_dofs)
        reduced_comparison = comparison_shape.reduce(common_dofs)
        return reduced_self, reduced_comparison

    def plot_frequency(self, interp_abscissa, interp_ordinate, ax=None, plot_kwargs={'color': 'k', 'marker': 'x', 'linestyle': "None"}):
        """
        Plots the frequencies of the shapes on curves of a 2D plot

        Parameters
        ----------
        interp_abscissa : np.ndarray
            The abscissa used to interpolate the Y-position of the frequency
            mark
        interp_ordinate : np.ndarray
            The ordinate used to interpolate the Y-postiion of the frequency
            mark
        ax : matplotlib axes, optional
            Axes on which to plot the frequency marks. The default is None,
            which creates a new window
        plot_kwargs : dict, optional
            Additional arguments to pass to the matplotlib plot command. The
            default is {'color':'k','marker':'x','linestyle':"None"}.

        Returns
        -------
        ax : matplotlib axes, optional
            Axes on which the frequency marks were plotted

        """
        if ax is None:
            fig, ax = plt.subplots()
        ys = np.interp(self.frequency, interp_abscissa, interp_ordinate)
        ax.plot(self.frequency.flatten(), ys.flatten(), **plot_kwargs)
        return ax

    def to_real(self, force_angle=-np.pi/4, **kwargs):
        """
        Creates real shapes from complex shapes by collapsing the complexity

        Parameters
        ----------
        force_angle : float
            Angle to force the complex collapsing to real, by default it is
            -pi/4 (-45 degrees).  To allow other angles, use None for this
            argument.
        **kwargs : various
            Extra arguments to pass to collapse_complex_to_real

        Returns
        -------
        ShapeArray
            Real shapes created from the original complex shapes

        """
        if not self.is_complex():
            return self.copy()
        matrix = collapse_complex_to_real(self.shape_matrix,
                                          force_angle=force_angle,
                                          **kwargs)
        damped_natural_frequency = (self.frequency*2*np.pi*np.sqrt(1-self.damping**2)).real[..., np.newaxis]
        matrix *= np.sqrt(2*damped_natural_frequency)
        return shape_array(self.coordinate, matrix.real, self.frequency, self.damping.real,
                           self.modal_mass.real, self.comment1, self.comment2, self.comment3,
                           self.comment4, self.comment5)
    
    def to_complex(self):
        """
        Creates complex shapes from real shapes

        Returns
        -------
        ShapeArray
            Complex shapes compute from the real shape coefficients

        """
        if self.is_complex():
            return self.copy()
        matrix = self.shape_matrix-1j*self.shape_matrix
        damped_natural_frequency = (self.frequency*2*np.pi*np.sqrt(1-self.damping**2)).real[..., np.newaxis]
        matrix /= 2*np.sqrt(damped_natural_frequency)
        return shape_array(self.coordinate, matrix, self.frequency, self.damping.real,
                           self.modal_mass.real, self.comment1, self.comment2, self.comment3,
                           self.comment4, self.comment5)
    
    def normalize(self,system_or_matrix, return_modal_matrix = False):
        """
        Computes A-normalized or mass-normalized shapes

        Parameters
        ----------
        system_or_matrix : System or np.ndarray
            A System object or a mass matrix for real modes or A-matrix for
            complex modes.
        return_modal_matrix : bool, optional
            If true, it will return the modal mass or modal-A matrix computed
            from the normalized mode shapes.  The default is False.

        Returns
        -------
        ShapeArray
            A copy of the original shape array with normalized shape coefficients

        """
        if self.is_complex():
            if isinstance(system_or_matrix,sdynpy_system.System):
                Z = np.zeros(system_or_matrix.M.shape)
                A = np.block([[                 Z, system_or_matrix.M],
                              [system_or_matrix.M, system_or_matrix.C]])
                shapes = self[system_or_matrix.coordinate].T
            else:
                A = system_or_matrix
                shapes = self.modeshape
            omega_r = self.frequency*2*np.pi
            zeta_r = self.damping
            lam = -omega_r*zeta_r + 1j*omega_r*np.sqrt(1-zeta_r**2)
            E = np.concatenate((lam*shapes,shapes),axis=-2)
            scale_factors = 1/np.sqrt(np.einsum('ji,jk,ki->i',E,A,E))
            if return_modal_matrix:
                modal_matrix = np.einsum('ji,jk,kl->il',E*scale_factors,A,E*scale_factors)
        else:
            if isinstance(system_or_matrix,sdynpy_system.System):
                M = system_or_matrix.M
                shapes = self[system_or_matrix.coordinate].T
            else:
                M = system_or_matrix
                shapes = self.modeshape
            scale_factors = 1/np.sqrt(np.einsum('ji,jk,ki->i',shapes,M,shapes))
            if return_modal_matrix:
                modal_matrix = np.einsum('ji,jk,kl->il',shapes*scale_factors,M,shapes*scale_factors)
        output_shape = self.copy()
        output_shape *= scale_factors
        if return_modal_matrix:
            return output_shape, modal_matrix
        else:
            return output_shape

    def write_to_unv(self, filename, specific_data_type=12, load_case_number=0
                     ):
        """
        Writes shape data to a unverisal file

        Parameters
        ----------
        filename : str
            Filename to which the geometry will be written.  If None,
            unv data will be returned instead, similar to that
            obtained from the readunv function in sdynpy
        specific_data_type : int, optional
            Integer specifying the type of data in the shape. The default is 12,
            which denotes acceleration shapes.
        load_case_number : int, optional
            The load case number. The default is 0.

        Raises
        ------
        NotImplementedError
            Raised if complex numbers are used as they are not implemented yet

        Returns
        -------
        shape_unv : List
            A list of Sdynpy_UFF_Dataset_55 objects, only if filename is None

        """
        from ..fileio.sdynpy_uff import dataset_55
        # First check if any rotations are in the shape
        is_six_dof = np.any(np.abs(self.coordinate.direction) > 3)
        is_complex = np.iscomplexobj(self.shape_matrix)
        nodes = np.unique(self.coordinate.node)
        directions = np.arange(6) + 1 if is_six_dof else np.arange(3) + 1
        full_coord_array = sdynpy_coordinate.coordinate_array(nodes[:, np.newaxis], directions)
        flat_self = self.flatten()
        shape_matrix = np.zeros(full_coord_array.shape + flat_self.shape,
                                dtype=self.shape_matrix.dtype)
        for index, coord in full_coord_array.ndenumerate():
            try:
                shape_matrix[index] = flat_self[coord].flatten()
            except ValueError:
                pass  # This means it's a uniaxial sensor and no triaxial dofs exist so we'll leave it at zero
        # Go through shapes and create the unv structures
        shape_unv = []
        for index, shape in flat_self.ndenumerate():
            # Create node dictionary
            node_dict = {node_coords[0].node: node_shape_matrix[..., index[0]]
                         for node_coords, node_shape_matrix in zip(full_coord_array, shape_matrix)}
            # Create the integer data
            ints = [load_case_number, index[0] + 1]
            # Create the real data
            if not is_complex:
                reals = [shape.frequency, shape.modal_mass, shape.damping, 0.0]
            else:
                if shape.modal_mass.real*1e-8 < shape.modal_mass.imag:
                    raise NotImplementedError("I don't know what modal mass is for a complex mode")
                reals = [(-shape.damping * shape.frequency * 2 * np.pi).real,  # Real part of eigenvalue
                         # Imaginary part of eigenvalue
                         (shape.frequency * 2 * np.pi * np.sqrt(1 - shape.damping**2)).real,
                         shape.modal_mass.real,  # Real modal mass
                         shape.modal_mass.imag,  # Imaginary modal mass
                         0.0,
                         0.0,
                         ]  # I don't actually know what modal a and modal b are...
            shape_unv.append(dataset_55.Sdynpy_UFF_Dataset_55(
                shape.comment1, shape.comment2, shape.comment3, shape.comment4, shape.comment5,
                1, 3 if is_complex else 2, 3 if is_six_dof else 2, specific_data_type, 5 if is_complex else 2,
                ints, reals, node_dict))
        if filename is None:
            return shape_unv
        else:
            with open(filename, 'w') as f:
                for dataset in shape_unv:
                    f.write('    -1\n')
                    f.write('    55\n')
                    f.write(dataset.write_string())
                    f.write('    -1\n')

    @staticmethod
    def concatenate_dofs(shape_arrays):
        """
        Combines the degrees of freedom from multiple shapes into one set of
        shapes

        Parameters
        ----------
        shape_arrays : list of ShapeArray
            List of ShapeArray objects to combine in to one set of shapes

        Returns
        -------
        ShapeArray
            ShapeArray object containing degrees of freedom from all input shapes

        """
        dof_lists = [np.unique(shape.coordinate) for shape in shape_arrays]
        shape_matrices = [shape_arrays[i][dofs] for i, dofs in enumerate(dof_lists)]

        all_dofs = np.concatenate(dof_lists, axis=-1)
        all_shape_matrices = np.concatenate(shape_matrices, axis=-1)

        # Create new shapes
        return shape_array(all_dofs, all_shape_matrices, shape_arrays[0].frequency,
                           shape_arrays[0].damping, shape_arrays[0].modal_mass,
                           shape_arrays[0].comment1, shape_arrays[0].comment2,
                           shape_arrays[0].comment3, shape_arrays[0].comment4,
                           shape_arrays[0].comment5)

    @staticmethod
    def overlay_shapes(geometries, shapes, color_override=None):
        """
        Combines several shapes and geometries for comparitive visualization


        Parameters
        ----------
        geometries : Iterable of Geometry
            A list of Geometry objects that will be combined into a single
            Geometry
        shapes : Iterable of ShapeArray
            A list of ShapeArray objects that will be combined into a single
            ShapeArray
        color_override : iterable, optional
            An iterble of integers specifying colors, which will override the
            existing geometry colors.  This should have the same length as the
            `geometries` input.  The default is None, which keeps the original
            geometry colors.

        Returns
        -------
        new_geometry : Geometry
            A geometry consisting of a combination of the specified geometries
        new_shape : ShapeArray
            A ShapeArray consisting of a combination of the specified ShapeArrays

        """
        from .sdynpy_geometry import Geometry
        new_geometry, node_offset = Geometry.overlay_geometries(
            geometries, color_override, True)

        new_shapes = [shape.copy() for shape in shapes]
        for i in range(len(new_shapes)):
            new_shapes[i].coordinate.node += node_offset * (i + 1)
        new_shape = ShapeArray.concatenate_dofs(new_shapes)
        return new_geometry, new_shape

    def time_integrate(self, forces, dt, responses=None, references=None,
                       displacement_derivative=2):
        """
        Integrate equations of motion created from shapes

        Parameters
        ----------
        forces : np.ndarray
            Input force provided to the shapes, with number of rows equal to
            the number of references, and number of columns equal to the number
            of time steps in the simulation.
        dt : float
            Time increment for the integration.  For accuracy, this is generally
            about 10x higher than the bandwidth of interest
        responses : CoordinateArray, optional
            Degrees of freedom to get response measurements. The default is None,
            which returns modal responses.
        references : CoordinateArray, optional
            Degrees of freedom at which to apply forces. The default is None,
            which means the input forces are treated as modal forces
        displacement_derivative : int, optional
            The derivative to use for the responses.  0 corresponds to
            displacement, 1 corresponds to velocity, and 2 corresponds to
            acceleration. The default is 2.

        Raises
        ------
        NotImplementedError
            Complex shapes are not currently implemented

        Returns
        -------
        response_array : TimeHistoryArray
            Responses assembled into a TimeHistoryArray.
        reference_array : TimeHistoryArray
            Input forces assembled into a TimeHistoryArray.

        """
        if self.is_complex():
            raise NotImplementedError('Complex Modes not Implemented')
        else:
            flat_self = self.flatten()
            M = np.diag(flat_self.modal_mass)
            K = np.diag(flat_self.modal_mass * (flat_self.frequency * 2 * np.pi)**2)
            C = np.diag(flat_self.modal_mass * (2 * 2 * np.pi *
                        flat_self.frequency * flat_self.damping))
            if responses is None:
                phi_out = np.eye(flat_self.size)
            else:
                phi_out = flat_self[responses].T
            if references is None:
                phi_in = np.eye(flat_self.size)
            else:
                phi_in = flat_self[references].T
            modal_forces = phi_in.T @ forces.reshape(-1, forces.shape[-1])
            times = np.arange(forces.shape[-1]) * dt
            modal_response = integrate_MCK(M, C, K, times, modal_forces.T)[
                displacement_derivative].T
            response = phi_out @ modal_response
            response_array = sdynpy_data.data_array(sdynpy_data.FunctionTypes.TIME_RESPONSE,
                                                    times, response,
                                                    sdynpy_coordinate.coordinate_array(np.arange(flat_self.size) + 1, 0)[:, np.newaxis]
                                                    if responses is None else np.atleast_1d(responses)[:, np.newaxis])
            reference_array = sdynpy_data.data_array(sdynpy_data.FunctionTypes.TIME_RESPONSE,
                                                     times, forces.reshape(-1, forces.shape[-1]),
                                                     sdynpy_coordinate.coordinate_array(np.arange(flat_self.size) + 1, 0)[:, np.newaxis]
                                                     if references is None else np.atleast_1d(references)[:, np.newaxis])
            return response_array, reference_array

    def optimize_degrees_of_freedom(self, sensors_to_keep,
                                    group_by_node=False, method='ei'):
        """
        Creates a reduced set of shapes using optimal degrees of freedom

        Parameters
        ----------
        sensors_to_keep : int
            Number of sensors to keep
        group_by_node : bool, optional
            If True, group shape degrees of freedom a the same node as one
            sensor, like a triaxial accelerometer. The default is False.
        method : str, optional
            'ei' for effective independence or 'cond' for condition number.
            The default is 'ei'.

        Returns
        -------
        ShapeArray
            The set of shapes with a reduced set of degrees of freedom that are
            optimally chosen.

        """
        if group_by_node:
            nodes = np.unique(self.coordinate.node)
            directions = np.unique(abs(self.coordinate).direction)
            coordinate_array = sdynpy_coordinate.coordinate_array(nodes[:, np.newaxis], directions)
        else:
            coordinate_array = np.unique(abs(self.coordinate))
        shape_matrix = self[coordinate_array]
        shape_matrix = np.moveaxis(shape_matrix, 0, -1)
        if method == 'ei':
            indices = by_effective_independence(sensors_to_keep, shape_matrix)
        elif method == 'cond':
            indices = by_condition_number(sensors_to_keep, shape_matrix)
        else:
            raise ValueError('Invalid `method`, must be one of "ei" or "cond"')
        coordinate_array = coordinate_array[indices]
        return self.reduce(coordinate_array)

    def system(self):
        """
        Create system matrices from the shapes

        This will create a System object with modal mass, stiffness, and
        damping matrices, with the mode shape matrix as the transformation
        to physical coordinates

        Raises
        ------
        NotImplementedError
            Raised if complex modes are used

        Returns
        -------
        System
            A System object containing modal mass, stiffness, and damping
            matrices, with the mode shape matrix as the transformation
            to physical coordinates

        """
        if self.is_complex():
            raise NotImplementedError('Complex Modes Not Implemented Yet')
        else:
            coordinates = np.unique(self.coordinate)
            return sdynpy_system.System(coordinates, np.diag(self.flatten().modal_mass),
                                        np.diag((2 * np.pi * self.frequency.flatten())
                                                ** 2 * self.flatten().modal_mass),
                                        np.diag(2 * (2 * np.pi * self.frequency) *
                                                self.damping * self.flatten().modal_mass),
                                        self[coordinates].T)

    @staticmethod
    def shape_alignment(shape_1, shape_2, node_id_map=None):
        """
        Computes if the shapes are aligned, or if one needs to be flipped

        Parameters
        ----------
        shape_1 : ShapeArray
            Shape to compare.
        shape_2 : ShapeArray
            Shape to compare.

        Returns
        -------
        np.ndarray
            An array denoting if one of the shapes need to be flipped (-1)
            to be equivalent to the other, or if they are already aligned (1)

        """
        if node_id_map is not None:
            shape_2 = shape_2.copy()
            shape_2.coordinate.node = node_id_map(shape_2.coordinate.node)
        common_coordinates = np.intersect1d(
            np.unique(shape_1.coordinate), np.unique(shape_2.coordinate))
        return np.sign(np.einsum('ij,ij->i', shape_1[common_coordinates], shape_2[common_coordinates]))

    def mode_table(self, table_format='csv',
                   frequency_format='{:0.2f}',
                   damping_format='{:0.2f}%'):
        """
        Generates a table of modal information including frequency and damping

        Parameters
        ----------
        table_format : str, optional
            The type of table to generate.  Can be 'csv', 'rst', 'markdown',
            'latex', 'pandas', or 'ascii'. The default is 'csv'.
        frequency_format : str, optional
            Format specifier for frequency. The default is '{:0.2f}'.
        damping_format : str, optional
            Format specifier for damping percent. The default is '{:0.2f}%'.

        Raises
        ------
        ValueError
            Raised if a invalid `table_format` is specified

        Returns
        -------
        table : str
            String representation of the mode table

        """
        available_formats = ['csv', 'rst', 'markdown', 'latex', 'ascii','pandas']
        if not table_format.lower() in available_formats:
            raise ValueError('`table_format` must be one of {:}'.format(available_formats))
        if table_format.lower() == 'ascii':
            return repr(self)
        elif table_format.lower() in ['csv', 'latex','pandas']:
            # Create a Pandas dataframe
            sorted_flat_self = np.sort(self.flatten())
            data_dict = {'Mode': np.arange(sorted_flat_self.size) + 1,
                         'Frequency': [frequency_format.format(v) for v in sorted_flat_self.frequency],
                         'Damping': [damping_format.format(v * (100 if '%' in damping_format else 1)) for v in sorted_flat_self.damping]}
            for field in ['comment1', 'comment2', 'comment3', 'comment4', 'comment5']:
                if np.all(sorted_flat_self[field] == ''):
                    continue
                data_dict[field.title()] = sorted_flat_self[field]
            df = pd.DataFrame(data_dict)
            if table_format.lower() == 'csv':
                return df.to_csv(index=False)
            elif table_format.lower() == 'latex':
                return df.to_latex(index=False)
            elif table_format.lower() == 'pandas':
                return df
            else:
                raise ValueError('Unknown Table Format: {:}'.format(table_format))
            
    def edit_comments(self, geometry = None):
        """
        Opens up a table where the shape comments can be edited
        
        If a geometry is also passed, it will also open up a mode shape plotter
        window where you can visualize the modes you are looking at.
        
        Edited comments will be stored back into the ShapeArray object when the
        OK button is pressed.  Comments will not be stored if the Cancel button
        is pressed.

        Parameters
        ----------
        geometry : Geometry, optional
            A geometry on which the shapes will be plotted.  If not specified,
            a table will just open up.

        Returns
        -------
        ShapeCommentTable
            A ShapeCommentTable displaying the modal information where comments
            can be edited.
            
        Notes
        -----
        Due to how Python handles garbage collection, the table may be
        immediately closed if not assigned to a variable, as Python things it
        is no longer in use.

        """
        if geometry is not None:
            plotter = geometry.plot_shape(self)
        return ShapeCommentTable(self, plotter)
    
    def transformation_matrix(self, physical_coordinates, inversion=True, normalized = True):
        """
        Creates a transformation matrix that describes a transformation from a physical coordinate array into modal space using the provided mode shapes.

        Parameters
        ----------
        coordinates : CoordinateArray
            The "physical" force coordinates for the transformation.
            
        inversion : bool, optional
            If True, the pseudo inverse of the mode shape matrix will be performed. If False, the mode shape matrix will not be inverted.

        normalized : bool, optional
            If True, the rows of the transformation matrix will be normalized to unit magnitude.

        Returns
        -------
        transformation : Matrix
            The transformation matrix as a matrix object. It is organized with
            the modal CoordinateArray on the rows and the physical 
            force CoordinateArray on the columns.
            
        Notes
        -----
        The transformation automatically handles polarity differences in the geometry 
        and force_coordinate.
        
        The returned transformation matrix is intended to be used as an output transformation matrix for MIMO vibration control.

        """
        # Generate Modal Coordinates
        modal_coordinates = sdynpy_coordinate.coordinate_array(node=np.arange(len(self)), direction=0)

        # Truncate Shapes to Physical Coordinates
        self = self.reduce(physical_coordinates)
        
        # Invert Matrix (if desired)
        if inversion:
            transformation_matrix = np.linalg.pinv(self.shape_matrix.T)
        else:
            transformation_matrix = self.shape_matrix
            
        # Normalize Matrix (if desired)
        if normalized:
            transformation_matrix = (transformation_matrix.T / np.abs(transformation_matrix).max(axis=1)).T

        return matrix(transformation_matrix, modal_coordinates, physical_coordinates)
        
class ShapeCommentTable(QDialog):
    def __init__(self, shapes, plotter = None, parent = None):
        """
        Creates a table window that allows editing of comments on the mode
        shapes.

        Parameters
        ----------
        shapes : ShapeArray
            The shapes for which the comments need to be modified.
        plotter : ShapePlotter, optional
            A shape plotter that is to be linked to the table.  It should have
            the same modes used for the table plotted on it.  The plotter will
            automatically update the mode being displayed as different rows of
            the table are selected. If not specified, there will be no mode
            shape display linked to the table.
        parent : QWidget, optional
            Parent widget for the window. The default is No parent.

        Returns
        -------
        None.

        """
        super().__init__(parent)
        # Add a table widget
        self.setWindowTitle('Shape Comment Editor')
        
        self.plotter = plotter
        self.shapes = shapes
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        self.layout = QVBoxLayout()
        # Columns will be key, frequency, damping, comment1, comment2, ... comment5
        self.mode_table = QTableWidget(shapes.size, 8)
        self.mode_table.currentItemChanged.connect(self.update_mode)
        self.mode_table.setHorizontalHeaderLabels(['Index','Frequency','Damping','Comment1','Comment2','Comment3','Comment4','Comment5'])
        self.mode_table.setSelectionMode(QAbstractItemView.SingleSelection)
        for row,(shape_index,shape) in enumerate(shapes.ndenumerate()):
            for column,value in enumerate([shape_index,shape.frequency,
                                           shape.damping,
                                           shape.comment1,
                                           shape.comment2,
                                           shape.comment3,
                                           shape.comment4,
                                           shape.comment5]):
                if column == 1:
                    item = QTableWidgetItem('{:0.2f}'.format(value))
                elif column == 2:
                    item = QTableWidgetItem('{:0.2f}%'.format(value*100))
                else:
                    item = QTableWidgetItem(str(value))
                if column <= 2:
                    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.mode_table.setItem(row,column,item)
        
        self.layout.addWidget(self.mode_table)
        self.layout.addWidget(self.button_box)
        self.setLayout(self.layout)
        
        self.show()
        
    def update_mode(self):
        row = self.mode_table.currentRow()
        if self.plotter is not None:
            self.plotter.current_shape = row
            self.plotter.reset_shape()
        
    def accept(self):
        for index,(shape_index,shape) in enumerate(self.shapes.ndenumerate()):
            shape.comment1 = self.mode_table.item(index,3).text()
            shape.comment2 = self.mode_table.item(index,4).text()
            shape.comment3 = self.mode_table.item(index,5).text()
            shape.comment4 = self.mode_table.item(index,6).text()
            shape.comment5 = self.mode_table.item(index,7).text()
        super().accept()
    
#    def string_array(self):
#        return create_coordinate_string_array(self.node,self.direction)
#
#    def __repr__(self):
#        return 'coordinate_array(string_array=\n'+repr(self.string_array())+')'
#
#    def __str__(self):
#        return str(self.string_array())
#
#    def __eq__(self,value):
#        value = np.array(value)
#        # A string
#        if np.issubdtype(value.dtype,np.character):
#            return self.string_array() == value
#        else:
#            if value.dtype.names is None:
#                node_logical = self.node == value[...,0]
#                direction_logical = self.direction == value[...,1]
#            else:
#                node_logical = self.node == value['node']
#                direction_logical = self.direction == value['direction']
#            return np.logical_and(node_logical,direction_logical)
#
#    def __ne__(self,value):
#        return ~self.__eq__(value)
#
#    def __abs__(self):
#        abs_coord = self.copy()
#        abs_coord.direction = abs(abs_coord.direction)
#        return abs_coord
#
#    def __neg__(self):
#        neg_coord = self.copy()
#        neg_coord.direction = -neg_coord.direction
#        return neg_coord
#
#    def __pos__(self):
#        pos_coord = self.copy()
#        pos_coord.direction = +pos_coord.direction
#        return pos_coord
#
#    def abs(self):
#        return self.__abs__()
#

def shape_array(coordinate=None, shape_matrix=None, frequency=1.0, damping=0.0, modal_mass=1.0,
                comment1='', comment2='', comment3='', comment4='', comment5='',
                structured_array=None):
    """
    Creates a coordinate array that specify degrees of freedom.

    Creates an array of coordinates that specify degrees of freedom in a test
    or analysis.  Coordinate arrays can be created using a numpy structured
    array or two arrays for node and direction.  Multidimensional arrays can
    be used.

    Parameters
    ----------
    coordinate : CoordinateArray
        Array of coordinates corresponding to the last dimension of the
        shape_matrix
    shape_matrix : ndarray
        Matrix of shape coefficients.  If complex, then the shape will be made
        into a complex shape.  Otherwise it will be real.  The last dimension
        should be the "coordinate" dimension, and the first dimension(s) should
        be the shape dimension(s).  Note that this is transposed from the
        typical modal approach, but it makes for better itegration with numpy.
    frequency : ndarray
        Natural Frequencies for each shape of the array
    damping : ndarray
        Fraction of Critical Damping (as proportion not percentage) for each
        shape of the array
    modal_mass : ndarray
        Modal mass for each shape of the array
    comment1 - comment5 : ndarray
        Comments for the universal file for each shape.  Note that comment5 will
        not be stored to the universal file format.
    structured_array : ndarray (structured)
        Alternatively to the above, a structured array can be passed with
        identical parameters to the above.

    Returns
    -------
    shape_array : ShapeArray

    """
    keys = [
        'coordinate',
        'shape_matrix',
        'frequency',
        'damping',
        'modal_mass',
        'comment1',
        'comment2',
        'comment3',
        'comment4',
        'comment5',
    ]
    data = {}
    if structured_array is not None:
        for key in keys:
            try:
                data[key] = structured_array[key]
            except (ValueError, TypeError):
                raise ValueError(
                    'structured_array must be numpy.ndarray with dtype names {:}'.format(keys))
    else:
        data['coordinate'] = np.array(coordinate).view(sdynpy_coordinate.CoordinateArray)
        data['shape_matrix'] = np.array(shape_matrix)
        data['frequency'] = np.array(frequency)
        data['damping'] = np.array(damping)
        data['modal_mass'] = np.array(modal_mass)
        data['comment1'] = np.array(comment1, dtype='<U80')
        data['comment2'] = np.array(comment2, dtype='<U80')
        data['comment3'] = np.array(comment3, dtype='<U80')
        data['comment4'] = np.array(comment4, dtype='<U80')
        data['comment5'] = np.array(comment5, dtype='<U80')

    # Create the coordinate array
    shp_array = ShapeArray(data['shape_matrix'].shape[:-1], data['shape_matrix'].shape[-1],
                           'real' if np.isrealobj(data['shape_matrix']) else 'complex')
    shp_array.coordinate = data['coordinate']
    shp_array.shape_matrix = data['shape_matrix']
    shp_array.frequency = data['frequency']
    shp_array.damping = data['damping']
    shp_array.modal_mass = data['modal_mass']
    shp_array.comment1 = data['comment1']
    shp_array.comment2 = data['comment2']
    shp_array.comment3 = data['comment3']
    shp_array.comment4 = data['comment4']
    shp_array.comment5 = data['comment5']

    return shp_array


from_imat_struct = ShapeArray.from_imat_struct
from_exodus = ShapeArray.from_exodus
from_unv = ShapeArray.from_unv
load = ShapeArray.load
concatenate_dofs = ShapeArray.concatenate_dofs
overlay_shapes = ShapeArray.overlay_shapes
shape_alignment = ShapeArray.shape_alignment


def rigid_body_error(geometry, rigid_shapes, **rigid_shape_kwargs):
    """
    Computes rigid shape error based on geometries

    Analytic rigid body shapes are computed from the geometry.  The supplied
    rigid_shapes are then projected through these analytic shapes and a
    residual is computed by subtracting the projected shapes from the original
    shapes.  This residual is a measure of how "non-rigid" the provided shapes
    were.

    Parameters
    ----------
    geometry : Geometry
        Geometry from which analytic rigid body shapes will be created
    rigid_shapes : ShapeArray
        ShapeArray consisting of nominally rigid shapes from which errors are
        to be computed
    **rigid_shape_kwargs : various
        Additional keywords that can be passed to the Geometry.rigid_body_shapes
        method.

    Returns
    -------
    coordinates : CoordinateArray
        The coordinates corresponding to the output residual array
    residual : np.ndarray
        Residuals computed by subtracting the provided shapes from those same
        shapes projected through the analytical rigid body shapes.

    """
    coordinates = np.unique(rigid_shapes.coordinate)
    true_rigid_shapes = geometry.rigid_body_shapes(coordinates, **rigid_shape_kwargs)
    shape_matrix_exp = rigid_shapes[coordinates].T
    shape_matrix_true = true_rigid_shapes[coordinates].T
    projection = shape_matrix_true @ np.linalg.lstsq(shape_matrix_true, shape_matrix_exp)[0]
    residual = np.abs(shape_matrix_exp - projection)
    return coordinates, residual


def rigid_body_check(geometry, rigid_shapes, distance_number=5, residuals_to_label=5,
                     return_shape_diagnostics=False, plot=True, return_figures = False, **rigid_shape_kwargs):
    """
    Performs rigid body checks, both looking at the complex plane and residuals

    Parameters
    ----------
    geometry : Geometry
        Geometry from which analytic rigid body shapes will be created
    rigid_shapes : ShapeArray
        ShapeArray consisting of nominally rigid shapes from which errors are
        to be computed
    distance_number : int, optional
        Threshold for number of neighbors to find outliers. The default is 5.
    residuals_to_label : int, optional
        The top `residuals_to_label` residuals will be highlighted in the
        residual plots. The default is 5.
    return_shape_diagnostics : True, optional
        If True, additional outputs are returned to help diagnose issues. The
        default is False.
    plot : bool, optional
        Whether or not to create plots of the results. The default is True.
    **rigid_shape_kwargs : various
        Additional keywords that can be passed to the Geometry.rigid_body_shapes
        method.

    Returns
    -------
    suspicious_channels : CoordinateArray
        A set of suspicous channels that should be investigated
    analytic_rigid_shapes : ShapeArray
        Rigid body shapes created from the geometry.  Only returned if
        return_shape_diagnostics is True
    residual : np.ndarray
        Values of the residuals at coordinates np.unique(rigid_shapes.coordinate).
        Only returned if return_shape_diagnostics is True
    shape_matrix_exp : np.ndarray
        The shape matrix of the supplied rigid shapes.  Only returned if
        return_shape_diagnostics is True

    """
    coordinates = np.unique(rigid_shapes.coordinate)
    true_rigid_shapes = geometry.rigid_body_shapes(coordinates, **rigid_shape_kwargs)
    shape_matrix_exp = rigid_shapes[coordinates].T
    shape_matrix_true = true_rigid_shapes[coordinates].T
    projection = shape_matrix_true @ np.linalg.lstsq(shape_matrix_true, shape_matrix_exp)[0]
    residual = np.abs(shape_matrix_exp - projection)/np.max(np.abs(projection),axis=0) # Normalize to the largest dof in each shape
    suspicious_channels = []
    # Plot the complex plane for each shape
    figs = []
    if plot:
        for i, (shape, data) in enumerate(zip(rigid_shapes, shape_matrix_exp.T)):
            fig, ax = plt.subplots(num='Shape {:} Complex Plane'.format(
                i if shape.comment1 == '' else shape.comment1))
            ax.plot(data.real, data.imag, 'bx')
            ax.set_xlabel('Real Part')
            ax.set_ylabel('Imag Part')
            ax.set_title('Shape {:}\nComplex Plane'.format(
                i+1 if shape.comment1 == '' else shape.comment1))
            x = np.real(data)
            y = np.imag(data)
            # Fit line through the origin (y = mx)
            m, residuals, rank, singular_values = np.linalg.lstsq(x[:, np.newaxis], y)
            m = m[0]  # Get the slope
            point_distances = np.abs(m * x - y) / np.sqrt(m**2 + 1)
            # Plot the line fit on the plot
            ax.axline((0,0),slope=m,color='k',linestyle='--')
            outliers = np.argsort(point_distances)[-5:]
            for outlier in outliers:
                ax.text(data[outlier].real, data[outlier].imag, str(coordinates[outlier]))
                suspicious_channels.append(coordinates[outlier])
            figs.append(fig)
    # Now plot the residuals
    residual_order = np.argsort(np.max(residual, axis=-1))
    if plot:
        fig, ax = plt.subplots(num='Projection Residuals')
        ax.plot(residual, 'x')
        figs.append(fig)
    for i in range(residuals_to_label):
        index = residual_order[-1 - i]
        if plot:
            ax.text(index, np.max(residual[index]), str(coordinates[index]))
        suspicious_channels.append(coordinates[index])
    if plot:
        # ax.set_yscale('log')
        ax.set_ylabel('Residual')
        ax.set_xlabel('Degree of Freedom')
    return_object = (np.array(suspicious_channels).view(sdynpy_coordinate.CoordinateArray),)
    if return_shape_diagnostics:
        return_object = return_object + (true_rigid_shapes, residual, shape_matrix_exp)
    if return_figures:
        return_object = return_object + tuple(figs)
    if len(return_object) == 1:
        return return_object[0]
    else:
        return return_object


def rigid_body_fix_node_orientation(geometry, rigid_shapes, suspect_nodes, new_cs_ids=None, gtol=1e-8, xtol=1e-8):
    """
    Solves for the best sensor orientation in the geometry to minimize the residual

    This function uses a nonlinear optimizer to solve for the orientation of
    the specified nodes that provide the best estimate of a rigid body shape.
    This can be used to try to figure out the true orientation of a sensor in
    a test.

    Parameters
    ----------
    geometry : Geometry
        Geometry from which analytic rigid body shapes will be created
    rigid_shapes : ShapeArray
        ShapeArray consisting of nominally rigid shapes from which errors are
        to be corrected
    suspect_nodes : np.ndarray
        A set of node ids to investigate.  The coordinate system of each of
        these will be modified.
    new_cs_ids : np.ndarray, optional
        Id numbers to give the newly added coordinate systems.   The default is
        None, which simply increments the maximum coordinate system for each
        suspect node
    gtol : float, optional
        Global tolerance for convergence for the nonlinear optimizer. The
        default is 1e-8.
    xtol : TYPE, optional
        Relative tolerance for convergence for the nonlinear optimizer. The
        default is 1e-8.

    Returns
    -------
    corrected_geometry : Geometry
        A geometry with coordinate systems modified to be consistent with the
        rigid body shapes supplied.

    """
    new_nodes = []
    new_css = []
    # Loop through all suspect nodes
    for i, node in enumerate(suspect_nodes):
        keep_nodes = [val for val in np.unique(geometry.node.id) if (
            val not in suspect_nodes) or (val == node)]
        error_geometry = geometry.reduce(keep_nodes)
        error_shapes = rigid_shapes.reduce(keep_nodes)

        def objective_function(quaternion):
            r = quaternion_to_rotation_matrix(quaternion)
            # Create a new geometry with the rotation applied
            updated_geometry = error_geometry.copy()
            new_cs = updated_geometry.coordinate_system(updated_geometry.node(node).disp_cs)
            new_cs.id = updated_geometry.coordinate_system.id.max(
            ) + 1 if new_cs_ids is None else new_cs_ids[i]
            new_cs.matrix[:3, :3] = r @ new_cs.matrix[:3, :3]
            updated_geometry.coordinate_system = np.concatenate(
                [updated_geometry.coordinate_system, new_cs[np.newaxis]])
            updated_geometry.node.disp_cs[updated_geometry.node.id == node] = new_cs.id
            residual = np.linalg.norm(rigid_body_error(updated_geometry, error_shapes)[1])
            return residual
        print('Optimizing Orientation of Node {:}'.format(node))
        output = opt.minimize(objective_function, [1, 0, 0, 0], method='trust-constr', constraints=[
                              unit_magnitude_constraint], options={'gtol': gtol, 'xtol': xtol})
        r = quaternion_to_rotation_matrix(output.x)
        new_cs = error_geometry.coordinate_system(error_geometry.node(node).disp_cs)
        new_cs.id = error_geometry.coordinate_system.id.max(
        ) + 1 + i if new_cs_ids is None else new_cs_ids[i]
        new_cs.matrix[:3, :3] = r @ new_cs.matrix[:3, :3]
        new_node = error_geometry.node[error_geometry.node.id == node].copy()
        new_node.disp_cs = new_cs.id
        new_nodes.append(new_node)
        new_css.append(new_cs)
    # Now add the nodes and coordinate systems to the geometry
    output_geometry = geometry.copy()
    for node in new_nodes:
        output_geometry.node[output_geometry.node.id == node.id] = node
    output_geometry.coordinate_system = np.concatenate([output_geometry.coordinate_system, new_css])
    return output_geometry


def mac(shape_1, shape_2=None, node_id_map=None):
    """
    Computes the modal assurance critera between two sets of shapes

    Parameters
    ----------
    shape_1 : ShapeArray
        A set of shapes to compute the MAC
    shape_2 : ShapeArray, optional
        A second set of shapes to compute the MAC.  If not specified, the
        AutoMAC of shape_1 is computed

    Returns
    -------
    mac_array : np.ndarray
        A numpy ndarray consisting of the MAC

    """
    if shape_2 is None:
        shape_2 = shape_1
    if node_id_map is not None:
        shape_2 = shape_2.copy()
        shape_2.coordinate.node = node_id_map(shape_2.coordinate.node)
    common_coordinates = np.intersect1d(
        np.unique(shape_1.coordinate), np.unique(shape_2.coordinate))
    phi_1 = shape_1[common_coordinates].T
    phi_2 = shape_2[common_coordinates].T
    return mac_corr(phi_1, phi_2)


def shape_comparison_table(shape_1, shape_2, frequency_format='{:0.2f}',
                           damping_format='{:.02f}%', mac_format='{:0.0f}',
                           percent_error_format='{:0.1f}%', spacing=2,
                           table_format='text', node_id_map=None):
    """
    Generates a shape comparison table between two sets of shapes

    Parameters
    ----------
    shape_1 : ShapeArray
        The first shape set to compare
    shape_2 : ShapeArray
        The second shape set to compare
    frequency_format : str, optional
        Format specifier for frequency. The default is '{:0.2f}'.
    damping_format : TYPE, optional
        Format specifier for damping. The default is '{:.02f}%'.
    mac_format : TYPE, optional
        Format specifier for MAC. The default is '{:0.0f}'.
    percent_error_format : TYPE, optional
        Format specifier for percent error. The default is '{:0.1f}%'.
    spacing : TYPE, optional
        Spacing added between columns. The default is 2.
    table_format : str, optional
        Table format to return.  Must be 'text', 'markdown', or 'latex'. The
        default is 'text'.
    node_id_map : id_map, optional
        An ID map to use if shapes do not have identical node ids.  The default
        is to assume that the shapes have common node ids.

    Raises
    ------
    ValueError
        Raised if an invalid `table_format` is specified

    Returns
    -------
    output_string : str
        String representation of the output table

    """
    shape_1 = shape_1.flatten()
    shape_2 = shape_2.flatten()
    frequency_strings_1 = ['Freq 1 (Hz)'] + [frequency_format.format(shape.frequency)
                                             for shape in shape_1]
    frequency_strings_2 = ['Freq 2 (Hz)'] + [frequency_format.format(shape.frequency)
                                             for shape in shape_2]
    index_strings = ['Mode'] + ['{:}'.format(v) for v in np.arange(shape_1.size) + 1]
    mac_strings = ['MAC'] + [mac_format.format(m * 100)
                             for m in np.einsum('ii->i', mac(shape_1, shape_2, node_id_map))]
    freq_error_strings = ['Freq Error'] + [percent_error_format.format(
        (s1.frequency - s2.frequency) * 100 / s2.frequency) for s1, s2 in zip(shape_1, shape_2)]
    index_size = max([len(s) for s in index_strings])
    freq_size = max([len(s) for s in frequency_strings_1 + frequency_strings_2])
    mac_size = max([len(s) for s in mac_strings])
    freqerr_size = max([len(s) for s in freq_error_strings])
    index_table_format = '{{:>{:}}}'.format(index_size + spacing)
    freq_table_format = '{{:>{:}}}'.format(freq_size + spacing)
    mac_table_format = '{{:>{:}}}'.format(mac_size + spacing)
    freqerr_table_format = '{{:>{:}}}'.format(freqerr_size + spacing)
    if damping_format is not None:
        damping_strings_1 = ['Damp 1'] + [damping_format.format(shape.damping * 100)
                                          for shape in shape_1]
        damping_strings_2 = ['Damp 2'] + [damping_format.format(shape.damping * 100)
                                          for shape in shape_2]
        damp_error_strings = ['Damp Error'] + [percent_error_format.format(
            (s1.damping - s2.damping) * 100 / s2.damping) for s1, s2 in zip(shape_1, shape_2)]
        damp_size = max([len(s) for s in damping_strings_1 + damping_strings_2])
        damperr_size = max([len(s) for s in damp_error_strings])
        damp_table_format = '{{:>{:}}}'.format(damp_size + spacing)
        damperr_table_format = '{{:>{:}}}'.format(damperr_size + spacing)
    if table_format.lower() == 'text':
        table_begin_string = ''
        table_end_string = ''
        separator_string = ''
        lineend_string = '\n'
        linebegin_string = ''
        header_separator_string = '\n'
    elif table_format.lower() == 'markdown':
        table_begin_string = ''
        table_end_string = ''
        separator_string = '|'
        lineend_string = '|\n'
        linebegin_string = '|'
        if damping_format is not None:
            header_separator_string = lineend_string + linebegin_string + separator_string.join(['-' * (length + spacing) for length in [
                index_size, freq_size, freq_size, freqerr_size,
                damp_size, damp_size, damperr_size, mac_size]]) + lineend_string
        else:
            header_separator_string = lineend_string + linebegin_string + separator_string.join(['-' * (length + spacing) for length in [
                index_size, freq_size, freq_size, freqerr_size, mac_size]]) + lineend_string
    elif table_format.lower() == 'latex':
        if damping_format is not None:
            table_begin_string = '\\begin{tabular}{rrrrrrrr}\n'
        else:
            table_begin_string = '\\begin{tabular}{rrrrr}\n'
        table_end_string = r'\end{tabular}'
        separator_string = ' & '
        lineend_string = '\\\\\n'
        linebegin_string = '    '
        header_separator_string = '\\\\ \\hline\n'
    else:
        raise ValueError("Invalid table_format.  Must be one of ['text','markdown','latex'].")
    output_string = table_begin_string
    if damping_format is not None:
        for i, (ind, f1, f2, d1, d2, fe, de, m) in enumerate(zip(
                index_strings, frequency_strings_1, frequency_strings_2,
                damping_strings_1, damping_strings_2, freq_error_strings,
                damp_error_strings, mac_strings)):
            output_string += (linebegin_string + index_table_format + separator_string
                              + freq_table_format + separator_string
                              + freq_table_format + separator_string
                              + freqerr_table_format + separator_string
                              + damp_table_format + separator_string
                              + damp_table_format + separator_string
                              + damperr_table_format + separator_string
                              + mac_table_format
                              + (header_separator_string if i == 0 else lineend_string)).format(
                                  ind, f1, f2, fe, d1, d2, de, m)
    else:
        for i, (ind, f1, f2, fe, m) in enumerate(zip(
                index_strings, frequency_strings_1, frequency_strings_2,
                freq_error_strings,
                mac_strings)):
            output_string += (linebegin_string + index_table_format + separator_string
                              + freq_table_format + separator_string
                              + freq_table_format + separator_string
                              + freqerr_table_format + separator_string
                              + mac_table_format
                              + (header_separator_string if i == 0 else lineend_string)).format(
                                  ind, f1, f2, fe, m)
    output_string += table_end_string
    if table_format.lower() == 'latex':
        output_string = output_string.replace('%', '\\%')
    return output_string
