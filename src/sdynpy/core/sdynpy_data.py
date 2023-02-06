# -*- coding: utf-8 -*-
"""
Defines the NDDataArray, which defines function data such as time histories.

This module also defines several subclasses of NDDataArray, which contain
function-type-specific capabilities.  Several Enumerations are also defined
that connect data fields from the universal file format to the NDDataArray
subclasses.

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
from . import sdynpy_coordinate
from .sdynpy_array import SdynpyArray
from .sdynpy_coordinate import outer_product
from .sdynpy_matrix import Matrix
from ..signal_processing.sdynpy_correlation import mac
from ..signal_processing.sdynpy_frf import timedata2frf
from ..signal_processing.sdynpy_cpsd import (cpsd as sp_cpsd,
                                             cpsd_coherence as sp_coherence,
                                             cpsd_to_time_history,
                                             cpsd_from_coh_phs)
from ..fem.sdynpy_exodus import Exodus
from scipy.linalg import eigh
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from copy import copy, deepcopy
from datetime import datetime
from qtpy import QtWidgets, uic, QtGui
from qtpy.QtGui import QIcon, QFont
from qtpy.QtCore import Qt, QCoreApplication, QRect
from qtpy.QtWidgets import (QToolTip, QLabel, QPushButton, QApplication,
                             QGroupBox, QWidget, QMessageBox, QHBoxLayout,
                             QVBoxLayout, QSizePolicy, QMainWindow,
                             QFileDialog, QErrorMessage, QListWidget, QLineEdit,
                             QDockWidget, QGridLayout, QButtonGroup, QDialog,
                             QCheckBox, QRadioButton, QMenuBar, QMenu)
try:
    from qtpy.QtGui import QAction
except ImportError:
    from qtpy.QtWidgets import QAction
import pyqtgraph
pyqtgraph.setConfigOption('background', 'w')
pyqtgraph.setConfigOption('foreground', 'k')
import os
import scipy.signal as sig


class SpecificDataType(Enum):
    """Enumeration containing the types of data in universal files"""
    UNKNOWN = 0
    GENERAL = 1
    STRESS = 2
    STRAIN = 3
    TEMPERATURE = 5
    HEAT_FLUX = 6
    DISPLACEMENT = 8
    REACTION_FORCE = 9
    VELOCITY = 11
    ACCELERATION = 12
    EXCITATION_FORCE = 13
    PRESSURE = 15
    MASS = 16
    TIME = 17
    FREQUENCY = 18
    RPM = 19
    ORDER = 20
    SOUND_PRESSURE = 21
    SOUND_INTENSITY = 22
    SOUND_POWER = 23


_specific_data_names = {val: val.name.replace('_', ' ').title() for val in SpecificDataType}

_specific_data_names_vectorized = np.vectorize(_specific_data_names.__getitem__)

#                 Table - Unit Exponents
# -------------------------------------------------------
#  Specific                   Direction
#           ---------------------------------------------
#    Data       Translational            Rotational
#           ---------------------------------------------
#    Type    Length  Force  Temp    Length  Force  Temp
# -------------------------------------------------------
#     0        0       0      0       0       0      0
#     1             (requires input to fields 2,3,4)
#     2       -2       1      0      -1       1      0
#     3        0       0      0       0       0      0
#     5        0       0      1       0       0      1
#     6        1       1      0       1       1      0
#     8        1       0      0       0       0      0
#     9        0       1      0       1       1      0
#    11        1       0      0       0       0      0
#    12        1       0      0       0       0      0
#    13        0       1      0       1       1      0
#    15       -2       1      0      -1       1      0
#    16       -1       1      0       1       1      0
#    17        0       0      0       0       0      0
#    18        0       0      0       0       0      0
#    19        0       0      0       0       0      0
# --------------------------------------------------------

_exponent_table = {
    SpecificDataType.UNKNOWN: (0, 0, 0, 0, 0, 0, 0, 0),
    SpecificDataType.GENERAL: (0, 0, 0, 0, 0, 0, 0, 0),
    SpecificDataType.STRESS: (-2, 1, 0, 0, -1, 1, 0, 0),
    SpecificDataType.STRAIN: (0, 0, 0, 0, 0, 0, 0, 0),
    SpecificDataType.TEMPERATURE: (0, 0, 1, 0, 0, 0, 1, 0),
    SpecificDataType.HEAT_FLUX: (1, 1, 0, 0, 1, 1, 0, 0),
    SpecificDataType.DISPLACEMENT: (1, 0, 0, 0, 0, 0, 0, 0),
    SpecificDataType.REACTION_FORCE: (0, 1, 0, 0, 1, 1, 0, 0),
    SpecificDataType.VELOCITY: (1, 0, 0, -1, 0, 0, 0, -1),
    SpecificDataType.ACCELERATION: (1, 0, 0, -2, 0, 0, 0, -2),
    SpecificDataType.EXCITATION_FORCE: (0, 1, 0, 0, 1, 1, 0, 0),
    SpecificDataType.PRESSURE: (-2, 1, 0, 0, -1, 1, 0, 0),
    SpecificDataType.MASS: (-1, 1, 0, 2, 1, 1, 0, 2),
    SpecificDataType.TIME: (0, 0, 0, 1, 0, 0, 0, 1),
    SpecificDataType.FREQUENCY: (0, 0, 0, -1, 0, 0, 0, -1),
    SpecificDataType.RPM: (0, 0, 0, -1, 0, 0, 0, -1)
}

_exponent_table_vectorized = np.vectorize(_exponent_table.__getitem__)


class TypeQual(Enum):
    """Enumeration containing the quantity type (Rotation or Translation)"""
    TRANSLATION = 0
    ROTATION = 1


_type_qual_names = {val: val.name.replace('_', ' ').title() for val in TypeQual}
_type_qual_names_vectorized = np.vectorize(_type_qual_names.__getitem__)


class FunctionTypes(Enum):
    """Enumeration containing types of functions found in universal files"""
    GENERAL = 0
    TIME_RESPONSE = 1
    AUTOSPECTRUM = 2
    CROSSSPECTRUM = 3
    FREQUENCY_RESPONSE_FUNCTION = 4
    TRANSMISIBILITY = 5
    COHERENCE = 6
    AUTOCORRELATION = 7
    CROSSCORRELATION = 8
    POWER_SPECTRAL_DENSITY = 9
    ENERGY_SPECTRAL_DENSITY = 10
    PROBABILITY_DENSITY_FUNCTION = 11
    SPECTRUM = 12
    CUMULATIVE_FREQUENCY_DISTRIBUTION = 13
    PEAKS_VALLEY = 14
    STRESS_PER_CYCLE = 15
    STRAIN_PER_CYCLE = 16
    ORBIT = 17
    MODE_INDICATOR_FUNCTION = 18
    FORCE_PATTERN = 19
    PARTIAL_POWER = 20
    PARTIAL_COHERENCE = 21
    EIGENVALUE = 22
    EIGENVECTOR = 23
    SHOCK_RESPONSE_SPECTRUM = 24
    FINITE_IMPULSE_RESPONSE_FILTER = 25
    MULTIPLE_COHERENCE = 26
    ORDER_FUNCTION = 27
    PHASE_COMPENSATION = 28


_imat_function_type_map = {'General': FunctionTypes.GENERAL,
                           'Time Response': FunctionTypes.TIME_RESPONSE,
                           'Auto Spectrum': FunctionTypes.AUTOSPECTRUM,
                           'Cross Spectrum': FunctionTypes.CROSSSPECTRUM,
                           'Frequency Response Function': FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                           'Transmissibility': FunctionTypes.TRANSMISIBILITY,
                           'Coherence': FunctionTypes.COHERENCE,
                           'Auto Correlation': FunctionTypes.AUTOCORRELATION,
                           'Cross Correlation': FunctionTypes.CROSSCORRELATION,
                           'Power Spectral Density': FunctionTypes.POWER_SPECTRAL_DENSITY,
                           'Energy Spectral Density': FunctionTypes.ENERGY_SPECTRAL_DENSITY,
                           'Probability Density Function': FunctionTypes.PROBABILITY_DENSITY_FUNCTION,
                           'Spectrum': FunctionTypes.SPECTRUM,
                           'Cumulative Frequency Distribution': FunctionTypes.CUMULATIVE_FREQUENCY_DISTRIBUTION,
                           'Peaks Valley': FunctionTypes.PEAKS_VALLEY,
                           'Stress/Cycles': FunctionTypes.STRESS_PER_CYCLE,
                           'Strain/Cycles': FunctionTypes.STRAIN_PER_CYCLE,
                           'Orbit': FunctionTypes.ORBIT,
                           'Mode Indicator Function': FunctionTypes.MODE_INDICATOR_FUNCTION,
                           'Force Pattern': FunctionTypes.FORCE_PATTERN,
                           'Partial Power': FunctionTypes.PARTIAL_POWER,
                           'Partial Coherence': FunctionTypes.PARTIAL_COHERENCE,
                           'Eigenvalue': FunctionTypes.EIGENVALUE,
                           'Eigenvector': FunctionTypes.EIGENVECTOR,
                           'Shock Response Spectrum': FunctionTypes.SHOCK_RESPONSE_SPECTRUM,
                           'Finite Impulse Response Filter': FunctionTypes.FINITE_IMPULSE_RESPONSE_FILTER,
                           'Multiple Coherence': FunctionTypes.MULTIPLE_COHERENCE,
                           'Order Function': FunctionTypes.ORDER_FUNCTION,
                           'Phase Compensation': FunctionTypes.PHASE_COMPENSATION,
                           }

_imat_function_type_inverse_map = {val: key for key, val in _imat_function_type_map.items()}


class NDDataArray(SdynpyArray):
    """Generic N-Dimensional data structure

    This data structure can contain real or complex data.  More specific
    SDynPy data arrays inherit from this superclass.
        """

    def __new__(subtype, shape, nelements, data_dimension, ordinate_dtype='float64',
                buffer=None, offset=0,
                strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to __array_finalize__
        data_dtype = [
            ('abscissa', 'float64', nelements),
            ('ordinate', ordinate_dtype, nelements),
            ('comment1', '<U80'),
            ('comment2', '<U80'),
            ('comment3', '<U80'),
            ('comment4', '<U80'),
            ('comment5', '<U80'),
            ('coordinate', sdynpy_coordinate.CoordinateArray.data_dtype,
             () if data_dimension is None else (data_dimension,))
        ]
        obj = super(NDDataArray, subtype).__new__(subtype, shape,
                                                  data_dtype, buffer, offset, strides, order)
        # Finally, we must return the newly created object:
        return obj

    @property
    def function_type(self):
        """
        Returns the function type of the data array as a FunctionTypes Enum
        """
        return FunctionTypes.GENERAL

    @property
    def response_coordinate(self):
        """CoordinateArray corresponding to the response coordinates"""
        return self.coordinate[..., 0]

    @response_coordinate.setter
    def response_coordinate(self, value):
        """Set the response coordinate of the data array"""
        self.coordinate[..., 0] = value

    @property
    def reference_coordinate(self):
        """CoordinateArray corresponding to the response coordinates"""
        if self.dtype['coordinate'].shape[0] == 1:
            raise AttributeError('{:} has no reference coordinate'.format(self.__class__.__name__))
        return self.coordinate[..., 1]

    @reference_coordinate.setter
    def reference_coordinate(self, value):
        """Set the reference coordinate of the data array"""
        if self.dtype['coordinate'].shape[0] == 1:
            raise AttributeError('{:} has no reference coordinate'.format(self.__class__.__name__))
        self.coordinate[..., 1] = value

    @property
    def num_elements(self):
        """Number of elements in each data array"""
        return self.dtype['ordinate'].shape[0]

    @property
    def num_coordinates(self):
        """Number of coordinates defining the data array"""
        return self.dtype['coordinate'].shape[0]

    @property
    def data_dimension(self):
        """Number of dimensions to the data"""
        return self.dtype['coordinate'].shape[-1]

    def plot(self, one_axis: bool = True, subplots_kwargs: dict = {},
             plot_kwargs: dict = {}):
        """
        Plot the data array

        Parameters
        ----------
        one_axis : bool, optional
            Set to True to plot all data on one axis.  Set to False to plot
            data on multiple subplots.  one_axis can also be set to a 
            matplotlib axis to plot data on an existing axis.  The default is
            True.
        subplots_kwargs : dict, optional
            Keywords passed to the matplotlib subplots function to create the
            figure and axes. The default is {}.
        plot_kwargs : dict, optional
            Keywords passed to the matplotlib plot function. The default is {}.

        Returns
        -------
        axis : matplotlib axis or array of axes
             On which the data were plotted

        """
        if one_axis is True:
            figure, axis = plt.subplots(**subplots_kwargs)
            axis.plot(self.flatten().abscissa.T, self.flatten().ordinate.T, **plot_kwargs)
        elif one_axis is False:
            ncols = int(np.floor(np.sqrt(self.size)))
            nrows = int(np.ceil(self.size / ncols))
            figure, axis = plt.subplots(nrows, ncols, **subplots_kwargs)
            for i, (ax, (index, function)) in enumerate(zip(axis.flatten(), self.ndenumerate())):
                ax.plot(function.abscissa.T, function.ordinate.T, **plot_kwargs)
                ax.set_ylabel('/'.join([str(v) for i, v in function.coordinate.ndenumerate()]))
            for ax in axis.flatten()[i + 1:]:
                ax.remove()
        else:
            axis = one_axis
            axis.plot(self.abscissa.T, self.ordinate.T, **plot_kwargs)
        return axis

    def reshape_to_matrix(self):
        """
        Reshapes a data array to a matrix with response coordinates along the
        rows and reference coordinates along the columns

        Returns
        -------
        output_array : Data Aarray
            2D Array of NDDataArray

        """
        flattened_functions = self.flatten()
        response_coords = np.unique(self.response_coordinate)
        reference_coords = np.unique(self.reference_coordinate)
        output_array = self.__class__(
            (response_coords.size, reference_coords.size), self.num_elements)
        for row_index, response_coord in response_coords.ndenumerate():
            for col_index, reference_coord in reference_coords.ndenumerate():
                current_function = flattened_functions[
                    (flattened_functions.response_coordinate == response_coord)
                    &
                    (flattened_functions.reference_coordinate == reference_coord)]
                if current_function.size == 0:
                    raise ValueError('No function exists with reference coordinate {:} and response coordinate {:}'.format(
                        str(reference_coord), str(response_coord)))
                if current_function.size > 1:
                    raise ValueError('Multiple functions exist ({:}) with reference coordinate {:} and response coordinate {:}'.format(
                        current_function.size, str(reference_coord), str(response_coord)))
                output_array[row_index[0], col_index[0]] = current_function
        return output_array

    def extract_elements(self, indices):
        """
        Parses elements from the data array specified by the passed indices

        Parameters
        ----------
        indices : 
            Any type of indices into a np.ndarray to select the elements to keep

        Returns
        -------
        NDDataArray
            Array reduced to specified elements

        """
        new_ordinate = self.ordinate[..., indices]
        new_abscissa = self.abscissa[..., indices]
        return data_array(self.function_type, new_abscissa, new_ordinate, self.coordinate,
                          self.comment1, self.comment2, self.comment3, self.comment4, self.comment5)

    def extract_elements_by_abscissa(self, min_abscissa, max_abscissa):
        abscissa_indices = (self.abscissa >= min_abscissa) & (self.abscissa <= max_abscissa)
        indices = np.all(abscissa_indices, axis=tuple(np.arange(abscissa_indices.ndim - 1)))
        new_ordinate = self.ordinate[..., indices]
        new_abscissa = self.abscissa[..., indices]
        return data_array(self.function_type, new_abscissa, new_ordinate, self.coordinate,
                          self.comment1, self.comment2, self.comment3, self.comment4, self.comment5)

    def downsample(self, factor):
        """
        Downsample a signal by keeping only every n-th abscissa/ordinate pair.

        Parameters
        ----------
        factor : int
            Downsample factor.  Only the factor-th abcissa will be kept.

        Returns
        -------
        NDDataArray
            The downsampled data object

        """
        new_ordinate = self.ordinate[..., ::factor]
        new_abscissa = self.abscissa[..., ::factor]
        return data_array(self.function_type, new_abscissa, new_ordinate, self.coordinate,
                          self.comment1, self.comment2, self.comment3, self.comment4, self.comment5)

    def validate_common_abscissa(self, **allclose_kwargs):
        """
        Returns True if all functions have the same abscissa

        Parameters
        ----------
        **allclose_kwargs : various
            Arguments to np.allclose to specify tolerances

        Returns
        -------
        bool
            True if all functions have the same abscissa

        """
        return np.allclose(self.flatten()[0].abscissa, self.abscissa, **allclose_kwargs)

    def transform_coordinate_system(self, original_geometry, new_geometry, node_id_map=None, rotations=False):
        """
        Performs coordinate system transformations on the data

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

        Returns
        -------
        NDDataArray or Subclass
            A NDDataArray that can now be plotted with the new geometry

        """
        if self.data_dimension == 1:
            if not node_id_map is None:
                original_geometry = original_geometry.reduce(node_id_map.from_ids)
                original_geometry.node.id = node_id_map(original_geometry.node.id)
                self = self.copy()[np.in1d(self.coordinate.node, node_id_map.from_ids)]
                self.coordinate.node = node_id_map(self.coordinate.node)
            common_nodes = np.intersect1d(np.intersect1d(original_geometry.node.id, new_geometry.node.id),
                                          np.unique(self.coordinate.node))
            original_coordinate_systems = original_geometry.coordinate_system(
                original_geometry.node(common_nodes).disp_cs)
            new_coordinate_systems = new_geometry.coordinate_system(
                new_geometry.node(common_nodes).disp_cs)
            coordinates = sdynpy_coordinate.coordinate_array(
                common_nodes[:, np.newaxis], [1, 2, 3, 4, 5, 6] if rotations else [1, 2, 3])
            new_data_array = self[coordinates[..., np.newaxis]].copy()
            shape_matrix = new_data_array.ordinate
            transform_from_original = original_coordinate_systems.matrix[..., :3, :3]
            transform_to_new = new_coordinate_systems.matrix[..., :3, :3]
            if rotations:
                # If we are doing rotations, we need to do a block diagonal for translations and rotations
                transform_from_original = np.concatenate((
                    np.concatenate((transform_from_original, np.zeros(
                        transform_from_original.shape)), axis=-1),
                    np.concatenate((np.zeros(transform_from_original.shape),
                                   transform_from_original), axis=-1),
                ), axis=-2)
                transform_to_new = np.concatenate((
                    np.concatenate((transform_to_new, np.zeros(transform_to_new.shape)), axis=-1),
                    np.concatenate((np.zeros(transform_to_new.shape), transform_to_new), axis=-1),
                ), axis=-2)
            new_shape_matrix = np.einsum('nij,nkj,nkl->nil', transform_to_new,
                                         transform_from_original, shape_matrix)
            new_data_array.ordinate = new_shape_matrix
            return new_data_array.flatten()
        else:
            raise NotImplementedError('2D Data not Implemented Yet')

    def __getitem__(self, key):
        """
        Selects specific data items by index or by coordinate

        Parameters
        ----------
        key : CoordinateArray or indices
            If key is a CoordinateArray, the returned NDDataArray will have the
            specified coordinates.  Otherwise, any form of indices can be passed
            to select specific data arrays.

        Returns
        -------
        NDDataArray
            Data Array partitioned to the selected arrays.
        """
        if isinstance(key, sdynpy_coordinate.CoordinateArray):
            coordinate_dim = self.dtype['coordinate'].ndim
            output_shape = key.shape[:-coordinate_dim]
            flat_self = self.flatten()
            index_array = np.empty(output_shape, dtype=int)
            positive_coordinates = abs(flat_self.coordinate)
            for index in np.ndindex(output_shape):
                positive_key = abs(key[index])
                index_array[index] = np.where(
                    np.all(positive_coordinates == positive_key, axis=-1))[0][0]
            return_shape = flat_self[index_array].copy()
            if self.function_type in [FunctionTypes.COHERENCE,FunctionTypes.MULTIPLE_COHERENCE]:
                ordinate_multiplication_array = np.array(1)
            else:
                ordinate_multiplication_array = np.prod(
                    np.sign(return_shape.coordinate.direction) * np.sign(key.direction), axis=-1)
            return_shape.coordinate = key
            return_shape.ordinate *= ordinate_multiplication_array[..., np.newaxis]
            return return_shape
        else:
            output = super().__getitem__(key)
            if key == 'coordinate':
                return output.view(sdynpy_coordinate.CoordinateArray)
            else:
                return output

    def __repr__(self):
        return '{:} with shape {:} and {:} elements per function'.format(self.__class__.__name__, ' x '.join(str(v) for v in self.shape), self.num_elements)

    def __add__(self, val):
        this = deepcopy(self)
        if isinstance(val, NDDataArray):
            # Check if abscissa are equivalent
            if not np.all(this.abscissa == val.abscissa):
                raise ValueError(
                    'Binary operations on NDDataArrays require equivalent or broadcastably equivalent abscissa')
            this.ordinate += val.ordinate
        else:
            this.ordinate += val
        return this

    def __sub__(self, val):
        this = deepcopy(self)
        if isinstance(val, NDDataArray):
            # Check if abscissa are equivalent
            if not np.all(this.abscissa == val.abscissa):
                raise ValueError(
                    'Binary operations on NDDataArrays require equivalent or broadcastably equivalent abscissa')
            this.ordinate -= val.ordinate
        else:
            this.ordinate -= val
        return this

    def __mul__(self, val):
        this = deepcopy(self)
        if isinstance(val, NDDataArray):
            # Check if abscissa are equivalent
            if not np.all(this.abscissa == val.abscissa):
                raise ValueError(
                    'Binary operations on NDDataArrays require equivalent or broadcastably equivalent abscissa')
            this.ordinate *= val.ordinate
        else:
            this.ordinate *= val
        return this

    def __truediv__(self, val):
        this = deepcopy(self)
        if isinstance(val, NDDataArray):
            # Check if abscissa are equivalent
            if not np.all(this.abscissa == val.abscissa):
                raise ValueError(
                    'Binary operations on NDDataArrays require equivalent or broadcastably equivalent abscissa')
            this.ordinate /= val.ordinate
        else:
            this.ordinate /= val
        return this

    def __pow__(self, val):
        this = deepcopy(self)
        if isinstance(val, NDDataArray):
            # Check if abscissa are equivalent
            if not np.all(this.abscissa == val.abscissa):
                raise ValueError(
                    'Binary operations on NDDataArrays require equivalent or broadcastably equivalent abscissa')
            this.ordinate **= val.ordinate
        else:
            this.ordinate **= val
        return this

    def __radd__(self, val):
        this = deepcopy(self)
        if isinstance(val, NDDataArray):
            # Check if abscissa are equivalent
            if not np.all(this.abscissa == val.abscissa):
                raise ValueError(
                    'Binary operations on NDDataArrays require equivalent or broadcastably equivalent abscissa')
            this.ordinate += val.ordinate
        else:
            this.ordinate += val
        return this

    def __rsub__(self, val):
        this = deepcopy(self)
        if isinstance(val, NDDataArray):
            # Check if abscissa are equivalent
            if not np.all(this.abscissa == val.abscissa):
                raise ValueError(
                    'Binary operations on NDDataArrays require equivalent or broadcastably equivalent abscissa')
            this.ordinate = val.ordinate - this.ordinate
        else:
            this.ordinate = val - this.ordinate
        return this

    def __rmul__(self, val):
        this = deepcopy(self)
        if isinstance(val, NDDataArray):
            # Check if abscissa are equivalent
            if not np.all(this.abscissa == val.abscissa):
                raise ValueError(
                    'Binary operations on NDDataArrays require equivalent or broadcastably equivalent abscissa')
            this.ordinate *= val.ordinate
        else:
            this.ordinate *= val
        return this

    def __rtruediv__(self, val):
        this = deepcopy(self)
        if isinstance(val, NDDataArray):
            # Check if abscissa are equivalent
            if not np.all(this.abscissa == val.abscissa):
                raise ValueError(
                    'Binary operations on NDDataArrays require equivalent or broadcastably equivalent abscissa')
            this.ordinate = val.ordinate / this.ordinate
        else:
            this.ordinate = val / this.ordinate
        return this

    def __rpow__(self, val):
        this = deepcopy(self)
        if isinstance(val, NDDataArray):
            # Check if abscissa are equivalent
            if not np.all(this.abscissa == val.abscissa):
                raise ValueError(
                    'Binary operations on NDDataArrays require equivalent or broadcastably equivalent abscissa')
            this.ordinate = val.ordinate**this.ordinate
        else:
            this.ordinate = val**this.ordinate
        return this

    def __neg__(self):
        this = deepcopy(self)
        this.ordinate *= -1
        return this

    def __abs__(self):
        this = deepcopy(self)
        this.ordinate = np.abs(this.ordinate)
        return this

    def min(self, reduction=None, *min_args, **min_kwargs):
        """
        Returns the minimum ordinate in the data array

        Parameters
        ----------
        reduction : function, optional
            Optional function to modify the data, e.g. to select minimum of the
            absolute value. The default is None.
        *min_args : various
            Additional arguments passed to np.min
        **min_kwargs : various
            Additional keyword arguments passed to np.min

        Returns
        -------
        Value
            Minimum value in the ordinate.

        """
        if reduction is None:
            return np.min(self.ordinate)
        else:
            return np.min(reduction(self.ordinate))

    def max(self, reduction=None, *max_args, **max_kwargs):
        """
        Returns the maximum ordinate in the data array

        Parameters
        ----------
        reduction : function, optional
            Optional function to modify the data, e.g. to select maximum of the
            absolute value. The default is None.
        *max_args : various
            Additional arguments passed to np.max
        **max_kwargs : various
            Additional keyword arguments passed to np.max

        Returns
        -------
        Value
            Maximum value in the ordinate.
        """
        if reduction is None:
            return np.max(self.ordinate, *max_args, **max_kwargs)
        else:
            return np.max(reduction(self.ordinate), *max_args, **max_kwargs)

    def argmin(self, reduction=None, *argmin_args, **argmin_kwargs):
        """
        Returns the index of the minimum ordinate in the data array

        Parameters
        ----------
        reduction : function, optional
            Optional function to modify the data, e.g. to select minimum of the
            absolute value. The default is None.
        *argmin_args : various
            Additional arguments passed to np.argmax
        **argmin_kwargs : various
            Additional keyword arguments passed to np.argmax

        Returns
        -------
        int
            Index of the minimum of the flattened ordinate.  Use
            np.unravel_index with self.ordinate.shape to get the unflattened
            index.
        """
        if reduction is None:
            return np.argmin(self.ordinate, *argmin_args, **argmin_kwargs)
        else:
            return np.argmin(reduction(self.ordinate), *argmin_args, **argmin_kwargs)

    def argmax(self, reduction=None, *argmax_args, **argmax_kwargs):
        """
        Returns the index of the maximum ordinate in the data array

        Parameters
        ----------
        reduction : function, optional
            Optional function to modify the data, e.g. to select maximum of the
            absolute value. The default is None.
        *argmax_args : various
            Additional arguments passed to np.argmax
        **argmax_kwargs : various
            Additional keyword arguments passed to np.argmax


        Returns
        -------
        int
            Index of the maximum of the flattened ordinate.  Use
            np.unravel_index with self.ordinate.shape to get the unflattened
            index.
        """
        if reduction is None:
            return np.argmax(self.ordinate, *argmax_args, **argmax_kwargs)
        else:
            return np.argmax(reduction(self.ordinate), *argmax_args, **argmax_kwargs)

    def to_imat_struct_array(self, Version=1, SetRecord=0, CreateDate: datetime = None, ModifyDate: datetime = None,
                             OwnerName='', AbscissaDataType=SpecificDataType.UNKNOWN, AbscissaTypeQual=TypeQual.TRANSLATION,
                             AbscissaAxisLab='', AbscissaUnitsLab='',
                             OrdNumDataType=SpecificDataType.UNKNOWN, OrdNumTypeQual=TypeQual.TRANSLATION,
                             OrdDenDataType=SpecificDataType.UNKNOWN, OrdDenTypeQual=TypeQual.TRANSLATION,
                             OrdinateAxisLab='', OrdinateUnitsLab='',
                             ZAxisDataType=SpecificDataType.UNKNOWN, ZAxisTypeQual=TypeQual.TRANSLATION,
                             ZGeneralValue=0, ZRPMValue=0, ZOrderValue=0, ZTimeValue=0,
                             UserValue1=0, UserValue2=0, UserValue3=0, UserValue4=0,
                             SamplingType='Dynamic', WeightingType='None', WindowType='None',
                             AmplitudeUnits='Unknown', Normalization='Unknown', OctaveFormat=0,
                             OctaveAvgType='None', ExpDampingFact=0, PulsesPerRev=0, MeasurementRun=0,
                             LoadCase=0, IRIGTime='', verbose=False
                             ):
        """
        Creates a Matlab structure that can be read the IMAT toolbox.

        This structure can be read by the IMAT toolbox in Matlab to create an
        imat_fn object.  Note this is generally a slower function than
        to_imat_struct.

        Parameters
        ----------
        Version : int, optional
            The version number of the function. The default is 1.
        SetRecord : int, optional
            The set record of the function. The default is 0.
        CreateDate : datetime, optional
            The date that the function was created. The default is Now.
        ModifyDate : datetime, optional
            The date that the function was modified. The default is Now.
        OwnerName : str, optional
            The owner of the dataset. The default is ''.
        AbscissaDataType : SpecificDataType, optional
            The type of data associated with the Abscissa of the function.
            The default is SpecificDataType.UNKNOWN.
        AbscissaTypeQual : TypeQual, optional
            The qualifier associated with the abscissa of the function. The
            default is TypeQual.TRANSLATION.
        AbscissaAxisLab : str, optional
            String used to label the abscissa axis. The default is ''.
        AbscissaUnitsLab : str, optional
            String used to label the units on the abscissa axis. The default is ''.
        OrdNumDataType : SpecificDataType, optional
            The type of data associated with the numerator of the ordinate of
            the function. The default is SpecificDataType.UNKNOWN.
        OrdNumTypeQual : TypeQual, optional
            The qualifier associated with the numerator of the ordinate of the
            function. The default is TypeQual.TRANSLATION.
        OrdDenDataType : SpecificDataType, optional
            The type of data associated with the denominator of the ordinate of
            the function. The default is SpecificDataType.UNKNOWN.
        OrdDenTypeQual : TypeQual, optional
            The qualifier associated with the denominator of the ordinate of the
            function. The default is TypeQual.TRANSLATION.
        OrdinateAxisLab : str, optional
            String used to label the ordinate axis. The default is ''.
        OrdinateUnitsLab : TYPE, optional
            String used to label the units on the ordinate axis. The default is ''.
        ZAxisDataType : TYPE, optional
            DESCRIPTION. The default is SpecificDataType.UNKNOWN.
        ZAxisTypeQual : TYPE, optional
            DESCRIPTION. The default is TypeQual.TRANSLATION.
        ZGeneralValue : TYPE, optional
            DESCRIPTION. The default is 0.
        ZRPMValue : TYPE, optional
            DESCRIPTION. The default is 0.
        ZOrderValue : TYPE, optional
            DESCRIPTION. The default is 0.
        ZTimeValue : TYPE, optional
            DESCRIPTION. The default is 0.
        UserValue1 : TYPE, optional
            DESCRIPTION. The default is 0.
        UserValue2 : TYPE, optional
            DESCRIPTION. The default is 0.
        UserValue3 : TYPE, optional
            DESCRIPTION. The default is 0.
        UserValue4 : TYPE, optional
            DESCRIPTION. The default is 0.
        SamplingType : TYPE, optional
            DESCRIPTION. The default is 'Dynamic'.
        WeightingType : TYPE, optional
            DESCRIPTION. The default is 'None'.
        WindowType : TYPE, optional
            DESCRIPTION. The default is 'None'.
        AmplitudeUnits : TYPE, optional
            DESCRIPTION. The default is 'Unknown'.
        Normalization : TYPE, optional
            DESCRIPTION. The default is 'Unknown'.
        OctaveFormat : TYPE, optional
            DESCRIPTION. The default is 0.
        OctaveAvgType : TYPE, optional
            DESCRIPTION. The default is 'None'.
        ExpDampingFact : TYPE, optional
            DESCRIPTION. The default is 0.
        PulsesPerRev : TYPE, optional
            DESCRIPTION. The default is 0.
        MeasurementRun : TYPE, optional
            DESCRIPTION. The default is 0.
        LoadCase : TYPE, optional
            DESCRIPTION. The default is 0.
        IRIGTime : TYPE, optional
            DESCRIPTION. The default is ''.
        verbose : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        output_struct : np.ndarray
            A numpy structured array that can be saved to a mat file using
            scipy.io.savemat.

        """
        flat_self = self.flatten()
        dtype = [('FunctionType', object),
                 ('Version', 'int'),
                 ('SetRecord', 'int'),
                 ('ResponseCoord', object),
                 ('ReferenceCoord', object),
                 ('IDLine1', object),
                 ('IDLine2', object),
                 ('IDLine3', object),
                 ('IDLine4', object),
                 ('CreateDate', object),
                 ('ModifyDate', object),
                 ('OwnerName', object),
                 ('Abscissa', 'float', (self.num_elements,)),
                 ('Ordinate', self.ordinate.dtype, (self.num_elements)),
                 ('AbscissaDataType', object),
                 ('AbscissaTypeQual', object),
                 ('AbscissaExpLength', float),
                 ('AbscissaExpForce', float),
                 ('AbscissaExpTemp', float),
                 ('AbscissaExpTime', float),
                 ('AbscissaAxisLab', object),
                 ('AbscissaUnitsLab', object),
                 ('OrdinateType', object),
                 ('OrdNumDataType', object),
                 ('OrdNumTypeQual', object),
                 ('OrdNumExpLength', float),
                 ('OrdNumExpForce', float),
                 ('OrdNumExpTemp', float),
                 ('OrdNumExpTime', float),
                 ('OrdDenDataType', object),
                 ('OrdDenTypeQual', object),
                 ('OrdDenExpLength', float),
                 ('OrdDenExpForce', float),
                 ('OrdDenExpTemp', float),
                 ('OrdDenExpTime', float),
                 ('OrdinateAxisLab', object),
                 ('OrdinateUnitsLab', object),
                 ('ZAxisDataType', object),
                 ('ZAxisTypeQual', object),
                 ('ZAxisExpLength', float),
                 ('ZAxisExpForce', float),
                 ('ZAxisExpTemp', float),
                 ('ZAxisExpTime', float),
                 ('ZGeneralValue', float),
                 ('ZRPMValue', float),
                 ('ZTimeValue', float),
                 ('ZOrderValue', float),
                 ('UserValue1', float),
                 ('UserValue2', float),
                 ('UserValue3', float),
                 ('UserValue4', float),
                 ('SamplingType', object),
                 ('WeightingType', object),
                 ('WindowType', object),
                 ('AmplitudeUnits', object),
                 ('Normalization', object),
                 ('OctaveFormat', float),
                 ('OctaveAvgType', object),
                 ('ExpDampingFact', float),
                 ('PulsesPerRev', float),
                 ('MeasurementRun', int),
                 ('LoadCase', int),
                 ('IRIGTime', object)
                 ]
        output_struct = np.empty(flat_self.shape,
                                 dtype=dtype)
        if verbose:
            print('Looping through functions for initial data')
        for i, fn in enumerate(flat_self):
            if fn.coordinate.shape == ():
                output_struct[i]['ResponseCoord'] = str(fn.coordinate)
                output_struct[i]['ReferenceCoord'] = ''
            else:
                output_struct[i]['ResponseCoord'] = str(fn.coordinate[0])
                if fn.coordinate.shape[0] > 1:
                    output_struct[i]['ReferenceCoord'] = str(fn.coordinate[1])
                else:
                    output_struct[i]['ReferenceCoord'] = ''
            output_struct[i]['Abscissa'] = fn.abscissa
            output_struct[i]['Ordinate'] = fn.ordinate
            output_struct[i]['IDLine1'] = fn.comment1
            output_struct[i]['IDLine2'] = fn.comment2
            output_struct[i]['IDLine3'] = fn.comment3
            output_struct[i]['IDLine4'] = fn.comment4
            output_struct[i]['FunctionType'] = _imat_function_type_inverse_map[self.function_type]

        if verbose:
            print('Assigning Version')
        output_struct['Version'] = np.broadcast_to(Version, flat_self.shape)
        if verbose:
            print('Assigning SetRecord')
        output_struct['SetRecord'] = np.broadcast_to(SetRecord, flat_self.shape)
        if CreateDate is None:
            CreateDate = datetime.now().strftime('%d-%b-%y    %H:%M:%S')
        if verbose:
            print('Assigning CreateDate')
        output_struct['CreateDate'] = np.broadcast_to(CreateDate, flat_self.shape)
        if ModifyDate is None:
            ModifyDate = datetime.now().strftime('%d-%b-%y    %H:%M:%S')
        if verbose:
            print('Assigning Modify Date')
        output_struct['ModifyDate'] = np.broadcast_to(ModifyDate, flat_self.shape)
        if verbose:
            print('Assigning OwnerName')
        output_struct['OwnerName'] = np.broadcast_to(OwnerName, flat_self.shape)
        # Abscissa
        if verbose:
            print('Assigning Abscissa Data')
        output_struct['AbscissaDataType'] = _specific_data_names_vectorized(
            np.broadcast_to(AbscissaDataType, flat_self.shape))
        output_struct['AbscissaTypeQual'] = _type_qual_names_vectorized(
            np.broadcast_to(AbscissaTypeQual, flat_self.shape))
        for i, (qual, datatype) in enumerate(zip(np.broadcast_to(AbscissaTypeQual, flat_self.shape), np.broadcast_to(AbscissaDataType, flat_self.shape))):
            exponents = _exponent_table[datatype]
            (output_struct['AbscissaExpLength'], output_struct['AbscissaExpForce'],
             output_struct['AbscissaExpTemp'], output_struct['AbscissaExpTime']
             ) = exponents[4:] if qual == TypeQual.ROTATION else exponents[:4]
        output_struct['AbscissaAxisLab'] = np.broadcast_to(AbscissaAxisLab, flat_self.shape)
        output_struct['AbscissaUnitsLab'] = np.broadcast_to(AbscissaUnitsLab, flat_self.shape)
        if verbose:
            print('Assigning Ordinate Numerator Data')
        # Ordinate Numerator
        output_struct['OrdNumDataType'] = _specific_data_names_vectorized(
            np.broadcast_to(OrdNumDataType, flat_self.shape))
        output_struct['OrdNumTypeQual'] = _type_qual_names_vectorized(
            np.broadcast_to(OrdNumTypeQual, flat_self.shape))
        for i, (qual, datatype) in enumerate(zip(np.broadcast_to(AbscissaTypeQual, flat_self.shape), np.broadcast_to(AbscissaDataType, flat_self.shape))):
            exponents = _exponent_table[datatype]
            (output_struct['OrdNumExpLength'], output_struct['OrdNumExpForce'],
             output_struct['OrdNumExpTemp'], output_struct['OrdNumExpTime']
             ) = exponents[4:] if qual == TypeQual.ROTATION else exponents[:4]
        # Ordinate Denominator
        if verbose:
            print('Assigning Ordinate Denominator Data')
        output_struct['OrdDenDataType'] = _specific_data_names_vectorized(
            np.broadcast_to(OrdDenDataType, flat_self.shape))
        output_struct['OrdDenTypeQual'] = _type_qual_names_vectorized(
            np.broadcast_to(OrdDenTypeQual, flat_self.shape))
        for i, (qual, datatype) in enumerate(zip(np.broadcast_to(AbscissaTypeQual, flat_self.shape), np.broadcast_to(AbscissaDataType, flat_self.shape))):
            exponents = _exponent_table[datatype]
            (output_struct['OrdDenExpLength'], output_struct['OrdDenExpForce'],
             output_struct['OrdDenExpTemp'], output_struct['OrdDenExpTime']
             ) = exponents[4:] if qual == TypeQual.ROTATION else exponents[:4]
        output_struct['OrdinateAxisLab'] = np.broadcast_to(OrdinateAxisLab, flat_self.shape)
        output_struct['OrdinateUnitsLab'] = np.broadcast_to(OrdinateUnitsLab, flat_self.shape)
        if self.ordinate.dtype == 'complex128':
            output_struct['OrdinateType'] = np.broadcast_to('Complex Double', flat_self.shape)
        elif self.ordinate.dtype == 'complex64':
            output_struct['OrdinateType'] = np.broadcast_to('Complex Single', flat_self.shape)
        if self.ordinate.dtype == 'float64':
            output_struct['OrdinateType'] = np.broadcast_to('Real Double', flat_self.shape)
        elif self.ordinate.dtype == 'float32':
            output_struct['OrdinateType'] = np.broadcast_to('Real Single', flat_self.shape)
        # Z Axis
        if verbose:
            print('Assigning ZAxis Data')
        output_struct['ZAxisDataType'] = _specific_data_names_vectorized(
            np.broadcast_to(ZAxisDataType, flat_self.shape))
        output_struct['ZAxisTypeQual'] = _type_qual_names_vectorized(
            np.broadcast_to(ZAxisTypeQual, flat_self.shape))
        for i, (qual, datatype) in enumerate(zip(np.broadcast_to(AbscissaTypeQual, flat_self.shape), np.broadcast_to(AbscissaDataType, flat_self.shape))):
            exponents = _exponent_table[datatype]
            (output_struct['ZAxisExpLength'], output_struct['ZAxisExpForce'],
             output_struct['ZAxisExpTemp'], output_struct['ZAxisExpTime']
             ) = exponents[4:] if qual == TypeQual.ROTATION else exponents[:4]
        output_struct['ZGeneralValue'] = np.broadcast_to(ZGeneralValue, flat_self.shape)
        output_struct['ZRPMValue'] = np.broadcast_to(ZRPMValue, flat_self.shape)
        output_struct['ZOrderValue'] = np.broadcast_to(ZOrderValue, flat_self.shape)
        output_struct['ZTimeValue'] = np.broadcast_to(ZTimeValue, flat_self.shape)
        if verbose:
            print('Assigning User Values')
        output_struct['UserValue1'] = np.broadcast_to(UserValue1, flat_self.shape)
        output_struct['UserValue2'] = np.broadcast_to(UserValue2, flat_self.shape)
        output_struct['UserValue3'] = np.broadcast_to(UserValue3, flat_self.shape)
        output_struct['UserValue4'] = np.broadcast_to(UserValue4, flat_self.shape)
        if verbose:
            print('Assigning Misc Data')
        output_struct['SamplingType'] = np.broadcast_to(SamplingType, flat_self.shape)
        output_struct['WeightingType'] = np.broadcast_to(WeightingType, flat_self.shape)
        output_struct['WindowType'] = np.broadcast_to(WindowType, flat_self.shape)
        output_struct['AmplitudeUnits'] = np.broadcast_to(AmplitudeUnits, flat_self.shape)
        output_struct['Normalization'] = np.broadcast_to(Normalization, flat_self.shape)
        output_struct['OctaveFormat'] = np.broadcast_to(OctaveFormat, flat_self.shape)
        output_struct['OctaveAvgType'] = np.broadcast_to(OctaveAvgType, flat_self.shape)
        output_struct['ExpDampingFact'] = np.broadcast_to(ExpDampingFact, flat_self.shape)
        output_struct['PulsesPerRev'] = np.broadcast_to(PulsesPerRev, flat_self.shape)
        output_struct['MeasurementRun'] = np.broadcast_to(MeasurementRun, flat_self.shape)
        output_struct['LoadCase'] = np.broadcast_to(LoadCase, flat_self.shape)
        output_struct['IRIGTime'] = np.broadcast_to(IRIGTime, flat_self.shape)

        return output_struct
    
    def save(self,filename):
        """
        Save the array to a numpy file

        Parameters
        ----------
        filename : str
            Filename that the array will be saved to.  Will be appended with
            .npz if not specified in the filename

        """
        np.savez(filename, data = self.view(np.ndarray),
                 function_type=self.function_type.value)
        
    @classmethod
    def load(cls,filename):
        """
        Load in the specified file into a SDynPy array object

        Parameters
        ----------
        filename : str
            Filename specifying the file to load.  If the filename has 
            extension .unv or .uff, it will be loaded as a universal file.
            Otherwise, it will be loaded as a NumPy file.

        Raises
        ------
        AttributeError
            Raised if a unv file is loaded from a class that does not have a
            from_unv attribute defined.

        Returns
        -------
        cls
            SDynpy array of the appropriate type from the loaded file.

        """
        if filename[-4:].lower() in ['.unv', '.uff']:
            try:
                from ..fileio.sdynpy_uff import readunv
                unv_dict = readunv(filename)
                return cls.from_unv(unv_dict)
            except AttributeError:
                raise AttributeError('Class {:} has no from_unv attribute defined'.format(cls))
        else:
            fn_data = np.load(filename, allow_pickle=True)
            return fn_data['data'].view(
                _function_type_class_map[FunctionTypes(fn_data['function_type'])])

    def to_imat_struct(self, Version=None, SetRecord=None, CreateDate: datetime = None, ModifyDate: datetime = None,
                       OwnerName=None, AbscissaDataType=None, AbscissaTypeQual=None,
                       AbscissaAxisLab=None, AbscissaUnitsLab=None,
                       OrdNumDataType=None, OrdNumTypeQual=None,
                       OrdDenDataType=None, OrdDenTypeQual=None,
                       OrdinateAxisLab=None, OrdinateUnitsLab=None,
                       ZAxisDataType=None, ZAxisTypeQual=None,
                       ZGeneralValue=None, ZRPMValue=None, ZOrderValue=None, ZTimeValue=None,
                       UserValue1=None, UserValue2=None, UserValue3=None, UserValue4=None,
                       SamplingType=None, WeightingType=None, WindowType=None,
                       AmplitudeUnits=None, Normalization=None, OctaveFormat=None,
                       OctaveAvgType=None, ExpDampingFact=None, PulsesPerRev=None, MeasurementRun=None,
                       LoadCase=None, IRIGTime=None
                       ):
        """
        Creates a Matlab structure that can be read the IMAT toolbox.

        This structure can be read by the IMAT toolbox in Matlab to create an
        imat_fn object.  Note this is generally a faster function than
        to_imat_struct_array.


        Parameters
        ----------
        Version : TYPE, optional
            DESCRIPTION. The default is None.
        SetRecord : TYPE, optional
            DESCRIPTION. The default is None.
        CreateDate : datetime, optional
            DESCRIPTION. The default is None.
        ModifyDate : datetime, optional
            DESCRIPTION. The default is None.
        OwnerName : TYPE, optional
            DESCRIPTION. The default is None.
        AbscissaDataType : TYPE, optional
            DESCRIPTION. The default is None.
        AbscissaTypeQual : TYPE, optional
            DESCRIPTION. The default is None.
        AbscissaAxisLab : TYPE, optional
            DESCRIPTION. The default is None.
        AbscissaUnitsLab : TYPE, optional
            DESCRIPTION. The default is None.
        OrdNumDataType : TYPE, optional
            DESCRIPTION. The default is None.
        OrdNumTypeQual : TYPE, optional
            DESCRIPTION. The default is None.
        OrdDenDataType : TYPE, optional
            DESCRIPTION. The default is None.
        OrdDenTypeQual : TYPE, optional
            DESCRIPTION. The default is None.
        OrdinateAxisLab : TYPE, optional
            DESCRIPTION. The default is None.
        OrdinateUnitsLab : TYPE, optional
            DESCRIPTION. The default is None.
        ZAxisDataType : TYPE, optional
            DESCRIPTION. The default is None.
        ZAxisTypeQual : TYPE, optional
            DESCRIPTION. The default is None.
        ZGeneralValue : TYPE, optional
            DESCRIPTION. The default is None.
        ZRPMValue : TYPE, optional
            DESCRIPTION. The default is None.
        ZOrderValue : TYPE, optional
            DESCRIPTION. The default is None.
        ZTimeValue : TYPE, optional
            DESCRIPTION. The default is None.
        UserValue1 : TYPE, optional
            DESCRIPTION. The default is None.
        UserValue2 : TYPE, optional
            DESCRIPTION. The default is None.
        UserValue3 : TYPE, optional
            DESCRIPTION. The default is None.
        UserValue4 : TYPE, optional
            DESCRIPTION. The default is None.
        SamplingType : TYPE, optional
            DESCRIPTION. The default is None.
        WeightingType : TYPE, optional
            DESCRIPTION. The default is None.
        WindowType : TYPE, optional
            DESCRIPTION. The default is None.
        AmplitudeUnits : TYPE, optional
            DESCRIPTION. The default is None.
        Normalization : TYPE, optional
            DESCRIPTION. The default is None.
        OctaveFormat : TYPE, optional
            DESCRIPTION. The default is None.
        OctaveAvgType : TYPE, optional
            DESCRIPTION. The default is None.
        ExpDampingFact : TYPE, optional
            DESCRIPTION. The default is None.
        PulsesPerRev : TYPE, optional
            DESCRIPTION. The default is None.
        MeasurementRun : TYPE, optional
            DESCRIPTION. The default is None.
        LoadCase : TYPE, optional
            DESCRIPTION. The default is None.
        IRIGTime : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        data_dict : TYPE
            DESCRIPTION.

        """
        arguments = {key: val for key, val in locals().items() if not key ==
                     'self'}  # Get all the local variables that have been defined
        data_dict = {}
        data_dict['Ordinate'] = np.moveaxis(self.ordinate, -1, 0)
        data_dict['Abscissa'] = np.moveaxis(self.abscissa, -1, 0)
        data_dict['IDLine1'] = self.comment1
        data_dict['IDLine2'] = self.comment2
        data_dict['IDLine3'] = self.comment3
        data_dict['IDLine4'] = self.comment4
        data_dict['ResponseCoord'] = np.array(self.response_coordinate.string_array(), dtype=object)
        try:
            data_dict['ReferenceCoord'] = np.array(
                self.reference_coordinate.string_array(), dtype=object)
        except AttributeError:
            pass
        data_dict['FunctionType'] = _imat_function_type_inverse_map[self.function_type]
        for argument, data in arguments.items():
            if not data is None:
                if isinstance(data, datetime):
                    data = data.strftime('%d-%b-%y    %H:%M:%S')
                if isinstance(data, SpecificDataType):
                    data = data.name.replace('_', ' ')
                data_dict[argument] = data
        return data_dict

    # def to_unv(self,function_id=1,version_number=1,load_case = 0,
    #            response_entity_name = None,reference_entity_name = None,
    #            abscissa_data_type = None,abscissa_length_exponent = None,
    #            abscissa_force_exponent = None, abscissa_temp_exponent = None,
    #            abscissa_axis_label = None,abscissa_units_label = None,
    #            ordinate_num_data_type = None,ordinate_num_length_exponent = None,
    #            ordinate_num_force_exponent = None,ordinate_num_temp_exponent = None,
    #            ordinate_num_axis_label = None,ordinate_num_units_label = None,
    #            ordinate_den_data_type = None,ordinate_den_length_exponent = None,
    #            ordinate_den_force_exponent = None,ordinate_den_temp_exponent = None,
    #            ordinate_den_axis_label = None,ordinate_den_units_label = None,
    #            zaxis_data_type = None,zaxis_length_exponent = None,
    #            zaxis_force_exponent = None,zaxis_temp_exponent = None,
    #            zaxis_axis_label = None,zaxis_units_label = None,zaxis_value= None):
    #     unv_data_dict = {58:[]}
    #     for key,func in self.ndenumerate():
    #         dataset_58 = sdpy.unv.dataset_58.Sdynpy_UFF_Dataset_58(
    #             func.comment1,func.comment2,func.comment3,func.comment4,
    #             func.comment5,func.function_type.value,)

    @staticmethod
    def from_unv(unv_data_dict, squeeze=True):
        """
        Create a data array from a unv dictionary from read_unv

        Parameters
        ----------
        unv_data_dict : dict
            Dictionary containing data from read_unv
        squeeze : bool, optional
            Automatically reduce dimension of the read data if possible.
            The default is True.

        Returns
        -------
        return_functions : NDDataArray
            Data read from unv

        """
        fn_datasets = unv_data_dict[58]
        fn_types = [dataset.function_type for dataset in fn_datasets]
        function_type_dict = {}
        for fn_dataset, fn_type in zip(fn_datasets, fn_types):
            fn_type_enum = FunctionTypes(fn_type)
            if not fn_type_enum in function_type_dict:
                function_type_dict[fn_type_enum] = []
            function_type_dict[fn_type_enum].append(fn_dataset)
        return_functions = []
        for key, function_list in function_type_dict.items():
            abscissa = []
            ordinate = []
            coordinate = []
            comment1 = []
            comment2 = []
            comment3 = []
            comment4 = []
            comment5 = []
            for function in function_list:
                abscissa.append(function.abscissa)
                ordinate.append(function.ordinate)
                coordinate.append((sdynpy_coordinate.coordinate_array(function.response_node, function.response_direction),
                                   sdynpy_coordinate.coordinate_array(function.reference_node, function.reference_direction)))
                comment1.append(function.idline1)
                comment2.append(function.idline2)
                comment3.append(function.idline3)
                comment4.append(function.idline4)
                comment5.append(function.idline5)
            return_functions.append(
                data_array(key, np.array(abscissa), np.array(ordinate),
                           np.array(coordinate).view(sdynpy_coordinate.CoordinateArray),
                           comment1,
                           comment2,
                           comment3,
                           comment4,
                           comment5)
            )
        if len(return_functions) == 1 and squeeze:
            return_functions = return_functions[0]
        return return_functions

    from_uff = from_unv


class TimeHistoryArray(NDDataArray):
    """Data array used to store time history data"""
    def __new__(subtype, shape, nelements, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(subtype, shape, nelements, 1, 'float64', buffer, offset, strides, order)
        return obj

    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.TIME_RESPONSE

    @classmethod
    def from_exodus(cls, exo, x_disp='DispX', y_disp='DispY', z_disp='DispZ', x_rot=None, y_rot=None, z_rot=None, timesteps=None):
        """
        Reads time data from displacements in an Exodus file

        Parameters
        ----------
        exo : Exodus or ExodusInMemory
            The exodus data from which geometry will be created.
        x_disp : str, optional
            String denoting the nodal variable in the exodus file from which
            the X-direction displacement should be read. The default is 'DispX'.
        y_disp : str, optional
            String denoting the nodal variable in the exodus file from which
            the Y-direction displacement should be read. The default is 'DispY'.
        z_disp : str, optional
            String denoting the nodal variable in the exodus file from which
            the Z-direction displacement should be read. The default is 'DispZ'.
        timesteps : iterable, optional
            A list of timesteps from which data should be read. The default is
            None, which reads all timesteps.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if isinstance(exo, Exodus):
            node_ids = exo.get_node_num_map()
            variables = [(i + 1, v) for i, v in enumerate([x_disp, y_disp,
                                                           z_disp, x_rot, y_rot, z_rot]) if not v is None]
            abscissa = exo.get_times()
            data = [data_array(FunctionTypes.TIME_RESPONSE, abscissa,
                               exo.get_node_variable_values(variable, timesteps).T,
                               sdynpy_coordinate.coordinate_array(node_ids, index)[:, np.newaxis]) for index, variable in variables]
            return np.concatenate(data)
        else:
            # TODO need to add in rotations
            node_ids = np.arange(
                exo.nodes.coordinates.shape[0]) + 1 if exo.nodes.node_num_map is None else exo.nodes.node_num_map
            x_var = [var for var in exo.nodal_vars if var.name.lower() == x_disp.lower(
            )][0].data[slice(timesteps) if timesteps is None else timesteps]
            y_var = [var for var in exo.nodal_vars if var.name.lower() == y_disp.lower(
            )][0].data[slice(timesteps) if timesteps is None else timesteps]
            z_var = [var for var in exo.nodal_vars if var.name.lower() == z_disp.lower(
            )][0].data[slice(timesteps) if timesteps is None else timesteps]
            abscissa = exo.time[slice(timesteps) if timesteps is None else timesteps]
            ordinate = np.concatenate((x_var, y_var, z_var), axis=-1).T
            coordinates = sdynpy_coordinate.coordinate_array(
                node_ids, np.array((1, 2, 3))[:, np.newaxis]).flatten()
            return data_array(FunctionTypes.TIME_RESPONSE, abscissa, ordinate, coordinates[:, np.newaxis])

    def fft(self, samples_per_frame=None, scaling=None, rtol=1, atol=1e-8):
        """
        Computes the frequency spectra of the time signal

        Parameters
        ----------
        samples_per_frame : int, optional
            Number of samples per measurement frame.  If this is specified, then
            the signal will be split up into frames and averaged together.  Be
            aware that if the time signal is not periodic, averaging it may have
            the effect of zeroing out the spectrum (because the average time
            signal is zero). The default is no averaging, the frame size is the
            length of the signal.
        scaling : str, optional
            The type of scaling applied to the output spectra.  This is not
            implemented yet. The default is None.
        rtol : float, optional
            Relative tolerance used in the abcsissa spacing check.
            The default is 1e-5.
        atol : float, optional
            Relative tolerance used in the abscissa spacing check.
            The default is 1e-8.

        Raises
        ------
        ValueError
            Raised if the time signal passed to this function does not have
            equally spaced abscissa.
        NotImplementedError
            Raised if the user specifies scaling.

        Returns
        -------
        SpectrumArray
            The frequency spectra of the TimeHistoryArray.

        """
        diffs = np.diff(self.abscissa, axis=-1).flatten()
        if not np.allclose(diffs, diffs[0], rtol, atol):
            raise ValueError('Abscissa must have identical spacing to perform the FFT')
        ordinate = self.ordinate
        if not samples_per_frame is None:
            frame_indices = np.arange(samples_per_frame) + np.arange(ordinate.size //
                                                                     samples_per_frame)[:, np.newaxis] * samples_per_frame
            ordinate = ordinate[..., frame_indices]
        dt = np.mean(diffs)
        n = ordinate.shape[-1]
        frequencies = np.fft.rfftfreq(n, dt)
        ordinate = np.fft.rfft(ordinate, axis=-1)
        if not scaling is None:
            raise NotImplementedError('Scaling is not implemented yet.')
        if not samples_per_frame is None:
            ordinate = np.mean(ordinate, axis=-2)
        # Create the output signal
        return data_array(FunctionTypes.SPECTRUM, frequencies, ordinate, self.coordinate,
                          self.comment1, self.comment2, self.comment3, self.comment4, self.comment5)

    def cpsd(self, samples_per_frame: int, overlap: float, window: str,
             averages_to_keep: int = None,
             only_asds=False, rtol=1, atol=1e-8):
        """
        Computes a CPSD matrix from the time histories

        Parameters
        ----------
        samples_per_frame : int
            Number of samples per frame
        overlap : float
            Overlap fraction (not percent, e.g. 0.5 not 50)
        window : str
            Name of a window function in scipy.signal.windows
        averages_to_keep : int, optional
            Optional number of averages to use, otherwise as many as possible
            will be used.
        only_asds : bool, optional
            If True, only compute autospectral densities (diagonal of the
            CPSD matrix)
        rtol : float, optional
            Tolerance used to check abscissa spacing. The default is 1.
        atol : float, optional
            Tolerance used to check abscissa spacing. The default is 1e-8.

        Raises
        ------
        ValueError
            If time history abscissa are not equally spaced.

        Returns
        -------
        cpsd_array : PowerSpectralDensityArray
            Cross Power Spectral Density Array.

        """
        diffs = np.diff(self.abscissa, axis=-1).flatten()
        if not np.allclose(diffs, diffs[0], rtol, atol):
            raise ValueError('Abscissa must have identical spacing to perform the cpsd')
        flat_self = self.flatten()
        coords = flat_self.response_coordinate
        ordinate = flat_self.ordinate
        df, cpsd_matrix = sp_cpsd(ordinate, 1 / np.mean(diffs), samples_per_frame,
                                  overlap, window, averages_to_keep, only_asds)
        #  Construct the spectrum array
        abscissa = np.arange(cpsd_matrix.shape[0]) * df
        cpsd_array = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, abscissa,
                                np.moveaxis(cpsd_matrix, 0, -1),
                                np.tile(coords[:, np.newaxis], [1, 2]) if only_asds
                                else outer_product(coords, coords))
        return cpsd_array

    def rms(self):
        return np.sqrt(np.mean(self.ordinate**2, axis=-1))


# def time_history_array(abscissa,ordinate,coordinate,comment1='',comment2='',comment3='',comment4='',comment5=''):
#     pass

class SpectrumArray(NDDataArray):
    """Data array used to store linear spectra (for example scaled FFT results)"""
    def __new__(subtype, shape, nelements, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(subtype, shape, nelements, 1, 'complex128', buffer, offset, strides, order)
        return obj

    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.SPECTRUM

    def ifft(self, scaling=None, rtol=1, atol=1e-8):
        """
        Computes a time signal from the frequency spectrum

        Parameters
        ----------
        scaling : str, optional
            The type of scaling applied to the output spectra.  This is not
            implemented yet. The default is None.
        rtol : float, optional
            Relative tolerance used in the abcsissa spacing check.
            The default is 1e-5.
        atol : float, optional
            Relative tolerance used in the abscissa spacing check.
            The default is 1e-8.

        Raises
        ------
        ValueError
            Raised if the spectra passed to this function do not have
            equally spaced abscissa.
        NotImplementedError
            Raised if the user specifies scaling.

        Returns
        -------
        TimeHistoryArray
            The time history of the SpectrumArray.

        """
        diffs = np.diff(self.abscissa, axis=-1).flatten()
        if not np.allclose(diffs, diffs[0], rtol, atol):
            raise ValueError('Abscissa must have identical spacing to perform the FFT')
        ordinate = self.ordinate
        if not scaling is None:
            raise NotImplementedError('Scaling is not implemented yet.')
        ordinate = np.fft.irfft(ordinate, axis=-1)
        dt = 1 / (self.abscissa.max() * 2)
        abscissa = np.arange(ordinate.shape[-1]) * dt
        return data_array(FunctionTypes.TIME_RESPONSE, abscissa, ordinate, self.coordinate,
                          self.comment1, self.comment2, self.comment3, self.comment4, self.comment5)

    def plot(self, one_axis=True, subplots_kwargs={}, plot_kwargs={}):
        """
        Plot the transfer functions

        Parameters
        ----------
        one_axis : bool, optional
            Set to True to plot all data on one axis.  Set to False to plot
            data on multiple subplots.  one_axis can also be set to a 
            matplotlib axis to plot data on an existing axis.  The default is
            True.
        subplots_kwargs : dict, optional
            Keywords passed to the matplotlib subplots function to create the
            figure and axes. The default is {}.
        plot_kwargs : dict, optional
            Keywords passed to the matplotlib plot function. The default is {}.

        Returns
        -------
        axis : matplotlib axis or array of axes
             On which the data were plotted

        """
        if one_axis is True:
            figure, axis = plt.subplots(2, 1, **subplots_kwargs)
            axis[0].plot(self.flatten().abscissa.T, np.angle(
                self.flatten().ordinate.T), **plot_kwargs)
            axis[1].plot(self.flatten().abscissa.T, np.abs(
                self.flatten().ordinate.T), **plot_kwargs)
            axis[1].set_yscale('log')
            axis[0].set_ylabel('Phase')
            axis[1].set_ylabel('Amplitude')
        elif one_axis is False:
            ncols = int(np.floor(np.sqrt(self.size)))
            nrows = int(np.ceil(self.size / ncols))
            figure, axis = plt.subplots(nrows, ncols, **subplots_kwargs)
            for i, (ax, (index, function)) in enumerate(zip(axis.flatten(), self.ndenumerate())):
                ax.plot(function.abscissa.T, np.abs(function.ordinate.T), **plot_kwargs)
                ax.set_ylabel('/'.join([str(v) for i, v in function.coordinate.ndenumerate()]))
                ax.set_yscale('log')
            for ax in axis.flatten()[i + 1:]:
                ax.remove()
        else:
            axis = one_axis
            axis.plot(self.flatten().abscissa.T, np.abs(self.flatten().ordinate.T), **plot_kwargs)
        return axis

# def spectrum_array(abscissa,ordinate,coordinate,comment1='',comment2='',comment3='',comment4='',comment5=''):
#     pass


class PowerSpectralDensityArray(NDDataArray):
    """Data array used to store power spectral density arrays"""
    def __new__(subtype, shape, nelements, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(subtype, shape, nelements, 2, 'complex128', buffer, offset, strides, order)
        return obj

    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.POWER_SPECTRAL_DENSITY

    def generate_time_history(self, time_length=None, output_oversample=1):
        """
        Generates a time history from a CPSD matrix

        Parameters
        ----------
        time_length : float, optional
            The length (in time, not samples) of the signal.  If not specified,
            the signal length will be based on the frequency spacing and
            nyquist frequency of the CPSD matrix.  If specified, a signal will
            be constructed using constant overlap and add techniques.  A whole
            number of realizations will be constructed, so the output signal
            can be longer than the `time_length` specified.
        output_oversample : int, optional
            Oversample factor applied to the output signal. The default is 1.

        Raises
        ------
        ValueError
            If the entries in the CPSD matrix do not have consistent abscissa
            or equally spaced frequency bins.

        Returns
        -------
        time_history : TimeHistoryArray
            A time history satisfying the properties of the CPSD matrix.

        """
        matrix_format = self.reshape_to_matrix()
        coordinates = matrix_format[:, 0].response_coordinate
        cpsd_matrix = np.moveaxis(matrix_format.ordinate, -1, 0)
        if not self.validate_common_abscissa(rtol=1, atol=1e-8):
            raise ValueError('All functions in CPSD matrix must have the same abscissa')
        abs_diff = np.diff(matrix_format.abscissa)
        df = np.mean(abs_diff)
        if not np.allclose(abs_diff, df):
            raise ValueError('Abscissa must have constant frequency spacing.  Max {:}, Min {:}'.format(
                abs_diff.max(), abs_diff.min()))
        sample_rate = 2 * matrix_format.abscissa.max()
        realization_length = 1 / df
        if time_length is None:
            realizations = 1
        else:
            realizations = int(np.ceil(time_length / realization_length * 2 - 1))
        # Do constant overlap and add
        final_signals = np.zeros((coordinates.size, (realizations+1) *
                                 (self.num_elements - 1)*output_oversample))
        window_function = sig.hann(2 * (self.num_elements - 1)*output_oversample, sym=False)**0.5
        window_first_half = window_function.copy()
        window_first_half[window_first_half.size // 2:] = 1
        window_second_half = window_function.copy()
        window_second_half[:window_second_half.size // 2] = 1
        for i in range(realizations):
            indices = slice(i * (self.num_elements - 1)*output_oversample, (i + 2) * (self.num_elements - 1)*output_oversample)
            realization = cpsd_to_time_history(cpsd_matrix, sample_rate, df, output_oversample)
            if i > 0:
                realization *= window_first_half
            if i < realizations - 1:
                realization *= window_second_half
            final_signals[:, indices] += realization
        # Create time history array
        abscissa = np.arange(final_signals.shape[-1]) / sample_rate / output_oversample
        time_history = data_array(FunctionTypes.TIME_RESPONSE, abscissa,
                                  final_signals, coordinates[:, np.newaxis])
        return time_history

    def mimo_forward(self, transfer_function):
        """
        Compute the forward MIMO problem Gxx = Hxv@Gvv@Hxv*

        Parameters
        ----------
        transfer_function : TransferFunctionArray
            Transfer function used to transform the input matrix to the
            response matrix

        Raises
        ------
        ValueError
            If abscissa do not match between self and transfer function

        Returns
        -------
        PowerSpectralDensityArray
            Response CPSD matrix

        """
        # Check consistent abscissa
        abscissa = self.flatten()[0].abscissa
        if not np.allclose(abscissa,transfer_function.abscissa):
            raise ValueError('Transfer Function Abscissa do not match CPSD')
        if not np.allclose(abscissa,self.abscissa):
            raise ValueError('All CPSD abscissa must be identical')
        # First do bookkeeping, we want to get the coordinates of the response
        # of the FRF corresponding to the specification matrix
        transfer_function = transfer_function.reshape_to_matrix()
        response_dofs = transfer_function[:,0].response_coordinate
        reference_dofs = transfer_function[0,:].reference_coordinate
        cpsd_dofs = outer_product(reference_dofs,reference_dofs)
        output_dofs= outer_product(response_dofs,response_dofs)
        frf_matrix = np.moveaxis(transfer_function.ordinate,-1,0)
        cpsd_matrix = np.moveaxis(self[cpsd_dofs].ordinate,-1,0)
        output_matrix = frf_matrix @ cpsd_matrix @ np.moveaxis(frf_matrix.conj(),-1,-2)
        return data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                          abscissa, np.moveaxis(output_matrix, 0, -1), output_dofs)
    
    def mimo_inverse(self, transfer_function,
                     response_weighting_matrix = None,
                     excitation_weighting_matrix = None,
                     regularization_parameter = None,
                     svd_regularization = None):
        """
        Computes input estimation for MIMO random vibration problems

        Parameters
        ----------
        transfer_function : TransferFunctionArray
            System transfer functions used to estimate the input from the given
            response matrix
        response_weighting_matrix : sdpy.Matrix or np.ndarray, optional
            Matrix used to weight response degrees of freedom.
        excitation_weighting_matrix : sdpy.Matrix or np.ndarray, optional
            Matrix used to weight input degrees of freedom
        regularization_parameter : float or np.ndarray, optional
            Scaling parameter used on the excitation weighting matrix
        svd_regularization : float, optional
            Condition number used for SVD truncation.  Can be used alternatively
            to the weighting matrices


        Raises
        ------
        ValueError
            If Abscissa are not consistent

        Returns
        -------
        PowerSpectralDensityArray
            Input CPSD matrix
            
        Notes
        -----
        This function solves the MIMO problem Gxx = Hxv@Gvv@Hxv^* using the
        pseudoinverse.  Gvv = Hxv^+@Gxx@Hxv^+^*.  We compute the pseudoinverse
        Hxv^+ = (Hxv^T@W^T@W@Hxv + l*Z)^-1@Hxv^T@W^T
        where W is the response_weighting_matrix, Z is the excitation_weighting_matrix,
        and l is the regularization_parameter.  If these are not specified,
        the SVD regularization is used, where the svd_regularization parameter
        is passed as the rcond argument to np.linalg.pinv.

        """
        # Check consistent abscissa
        abscissa = self.flatten()[0].abscissa
        if not np.allclose(abscissa,transfer_function.abscissa):
            raise ValueError('Transfer Function Abscissa do not match CPSD')
        if not np.allclose(abscissa,self.abscissa):
            raise ValueError('All CPSD abscissa must be identical')
        # First do bookkeeping, we want to get the coordinates of the response
        # of the FRF corresponding to the specification matrix
        transfer_function = transfer_function.reshape_to_matrix()
        response_dofs = transfer_function[:,0].response_coordinate
        reference_dofs = transfer_function[0,:].reference_coordinate
        cpsd_dofs = outer_product(response_dofs,response_dofs)
        output_dofs= outer_product(reference_dofs,reference_dofs)
        frf_matrix = np.moveaxis(transfer_function.ordinate,-1,0)
        cpsd_matrix = np.moveaxis(self[cpsd_dofs].ordinate.copy(),-1,0)
        # Perform the generalized inversion
        if (response_weighting_matrix is None and
            excitation_weighting_matrix is None and
            regularization_parameter is None):
            frf_pinv = np.linalg.pinv(frf_matrix,rcond=1e-15 if svd_regularization is None else svd_regularization)
        else:
            if response_weighting_matrix is not None:
                if isinstance(response_weighting_matrix,Matrix):
                    response_weighting_matrix = response_weighting_matrix[response_dofs,response_dofs]
                frf_matrix = response_weighting_matrix@frf_matrix
                cpsd_matrix = response_weighting_matrix @ cpsd_matrix @ np.moveaxis(response_weighting_matrix.conj(),-1,-2)
            if regularization_parameter is None:
                regularization_parameter = 0
            if excitation_weighting_matrix is None:
                excitation_weighting_matrix = 0
            elif isinstance(excitation_weighting_matrix,Matrix):
                excitation_weighting_matrix = excitation_weighting_matrix[reference_dofs,reference_dofs]
            frf_matrix_H = np.moveaxis(frf_matrix.conj(),-1,-2)
            frf_pinv = np.linalg.solve((frf_matrix_H@frf_matrix+regularization_parameter*excitation_weighting_matrix),frf_matrix_H)
        output_matrix = frf_pinv @ cpsd_matrix @ np.moveaxis(frf_pinv.conj(),-1,-2)
        return data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                          abscissa, np.moveaxis(output_matrix, 0, -1), output_dofs)
    
    def error_summary(self,figure_kwargs={}, linewidth=1, plot_kwargs={},**cpsd_matrices):
        """
        Plots an error summary compared to the current array

        Parameters
        ----------
        figure_kwargs : dict, optional
            Arguments to use when creating the figure. The default is {}.
        linewidth : float, optional
            Widths of the lines on the plot. The default is 1.
        plot_kwargs : dict, optional
            Arguments to use when plotting the lines. The default is {}.
        **cpsd_matrices : PowerSpectralDensityArray
            Data to compare against the current CPSD matrix.  The keys will be
            used as labels with _ replaced with a space.

        Raises
        ------
        ValueError
            If CPSD abscissa do not match

        Returns
        -------
        Error Metrics
            A tuple of dictionaries of error metrics

        """
        def rms(x, axis=None):
            return np.sqrt(np.mean(x**2, axis=axis))
        def dB_pow(x): return 10 * np.log10(x)
        frequencies = self.flatten()[0].abscissa
        for legend,cpsd in cpsd_matrices.items():
            if not np.allclose(frequencies,cpsd.abscissa):
                raise ValueError('Compared CPSD abscissa do not match')
        if not np.allclose(frequencies,self.abscissa):
            raise ValueError('All CPSD abscissa must be identical')
        # Get ASDs
        responses = np.unique(abs(self.coordinate))
        response_dofs = np.tile(responses[:,np.newaxis],2)
        channel_names = responses.string_array()
        spec_asd = np.real(self[response_dofs].ordinate)
        data_asd = {legend:np.real(data[response_dofs].ordinate) for legend,data in cpsd_matrices.items()}
        num_channels = spec_asd.shape[0]
        ncols = int(np.floor(np.sqrt(num_channels)))
        nrows = int(np.ceil(num_channels / ncols))
        if len(cpsd_matrices) > 1:
            total_rows = nrows + 2
        elif len(cpsd_matrices) == 1:
            total_rows = nrows + 1
        else:
            total_rows = nrows
        fig = plt.figure(**figure_kwargs)
        grid_spec = plt.GridSpec(total_rows, ncols, figure=fig)
        for i in range(num_channels):
            this_row = i // ncols
            this_col = i % ncols
            if i == 0:
                ax = fig.add_subplot(grid_spec[this_row, this_col])
                original_ax = ax
            else:
                ax = fig.add_subplot(grid_spec[this_row, this_col], sharex=original_ax,
                                     sharey=original_ax)
            ax.plot(frequencies, spec_asd[i], linewidth=linewidth * 2, color='k', **plot_kwargs)
            for legend, data in data_asd.items():
                ax.plot(frequencies, data[i], linewidth=linewidth)
            ax.set_ylabel(channel_names[i])
            if i == 0:
                ax.set_yscale('log')
            if this_row == nrows - 1:
                ax.set_xlabel('Frequency (Hz)')
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            if this_col != 0:
                plt.setp(ax.get_yticklabels(), visible=False)
        return_data = None
        if len(cpsd_matrices) > 0:
            spec_sum_asd = np.sum(spec_asd, axis=0)
            data_sum_asd = {legend: np.sum(data, axis=0) for legend, data in data_asd.items()}
            db_error = {legend: rms(dB_pow(data) - dB_pow(spec_asd), axis=0)
                        for legend, data in data_asd.items()}
            plot_width = ncols // 2
            ax = fig.add_subplot(grid_spec[nrows, 0:plot_width])
            ax.plot(frequencies, spec_sum_asd, linewidth=2 * linewidth, color='k')
            for legend, data in data_sum_asd.items():
                ax.plot(frequencies, data, linewidth=linewidth)
            ax.set_yscale('log')
            ax.set_ylabel('Sum ASDs')
            ax = fig.add_subplot(grid_spec[nrows, -plot_width:])
            for legend, data in db_error.items():
                ax.plot(frequencies, data, linewidth=linewidth)
            ax.set_ylabel('dB Error')
        if len(cpsd_matrices) > 1:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color'] * 10
            db_error_sum_asd = {legend: rms(dB_pow(sum_asd) - dB_pow(spec_sum_asd))
                                for legend, sum_asd in data_sum_asd.items()}
            db_error_rms = {legend: rms(data) for legend, data in db_error.items()}
            return_data = (db_error_sum_asd, db_error_rms)
            ax = fig.add_subplot(grid_spec[nrows + 1, 0:plot_width])
            for i, (legend, data) in enumerate(db_error_sum_asd.items()):
                ax.bar(i, data, color=colors[i])
                ax.text(i, 0, '{:.2f}'.format(data),
                        horizontalalignment='center', verticalalignment='bottom')
            ax.set_xticks(np.arange(i + 1))
            ax.set_xticklabels([legend.replace('_', ' ')
                               for legend in db_error_sum_asd], rotation=20, horizontalalignment='right')
            ax.set_ylabel('Sum RMS dB Error')
            ax = fig.add_subplot(grid_spec[nrows + 1, -plot_width:])
            for i, (legend, data) in enumerate(db_error_rms.items()):
                ax.bar(i, data, color=colors[i])
                ax.text(i, 0, '{:.2f}'.format(data),
                        horizontalalignment='center', verticalalignment='bottom')
            ax.set_xticks(np.arange(i + 1))
            ax.set_xticklabels([legend.replace('_', ' ')
                               for legend in db_error_rms], rotation=20, horizontalalignment='right')
            ax.set_ylabel('RMS dB Error')
        fig.tight_layout()
        return return_data

    def get_asd(self):
        """
        Get functions where the response coordinate is equal to the reference coordinate

        Returns
        -------
        PowerSpectralDensityArray
            PowerSpectralDensityArrays where the response is equal to the reference

        """
        indices = np.where(abs(self.coordinate[...,0]) == abs(self.coordinate[...,1]))
        return self[indices]

    def rms(self):
        """
        Compute RMSs of the PSDs using the diagonals

        Returns
        -------
        ndarray
            RMS values for the ASDS

        """
        asd = self.get_asd()
        return np.sqrt(np.sum(asd.ordinate.real,axis=-1)*np.mean(np.diff(asd.abscissa,axis=-1),axis=-1))

    @staticmethod
    def plot_asds(figure_kwargs={}, linewidth=1,**cpsd_matrices):
        """
        Plot the diagonals of the CPSD matrix, as well as the level

        Parameters
        ----------
        figure_kwargs : dict, optional
            Optional arguments to use when creating the figure. The default is {}.
        linewidth : float, optional
            Width of plotted lines. The default is 1.
        **cpsd_matrices : PowerSpectralDensityArray
            PSDs to plot.  Only gets plotted if response and reference are
            identical.  The key will be used as the label with _ replaced by a
            space.

        Raises
        ------
        ValueError
            If degrees of freedom are not consistent between PSDs

        Returns
        -------
        None.

        """
        asds = {legend:cpsd.get_asd() for legend,cpsd in cpsd_matrices.items()}
        for i,(legend,asd) in enumerate(asds.items()):
            this_dofs = np.unique(asd.coordinate)
            this_abscissa = asd.abscissa
            if i == 0:
                dofs = this_dofs
            if not np.all(this_dofs == dofs):
                raise ValueError('CPSDs must have identical dofs')
        # Sort the dofs correctly
        asds = {legend:asd[np.tile(dofs[:,np.newaxis],2)] for legend,asd in asds.items()}
        num_channels = len(this_dofs)
        ncols = int(np.floor(np.sqrt(num_channels)))
        nrows = int(np.ceil(num_channels / ncols))
        total_rows = nrows + 1
        fig = plt.figure(**figure_kwargs)
        grid_spec = plt.GridSpec(total_rows, ncols, figure=fig)
        for i in range(num_channels):
            this_row = i // ncols
            this_col = i % ncols
            if i == 0:
                ax = fig.add_subplot(grid_spec[this_row, this_col])
                original_ax = ax
            else:
                ax = fig.add_subplot(grid_spec[this_row, this_col], sharex=original_ax,
                                     sharey=original_ax)
            for legend, data in asds.items():
                ax.plot(data[i].abscissa, np.real(data[i].ordinate), linewidth=linewidth)
            ax.set_ylabel(str(dofs[i]))
            if i == 0:
                ax.set_yscale('log')
            if this_row == nrows - 1:
                ax.set_xlabel('Frequency (Hz)')
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            if this_col != 0:
                plt.setp(ax.get_yticklabels(), visible=False)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color'] * 10
        ax = fig.add_subplot(grid_spec[total_rows-1, 0:ncols])
        legend_handles = []
        legend_strings = []
        for i,(legend,asd) in enumerate(asds.items()):
            for j,fn in enumerate(asd):
                rms = np.sqrt(np.sum(fn.ordinate.real)*np.mean(np.diff(fn.abscissa)))
                x = i+(len(asds)+1)*j
                a = ax.bar(x, rms, color=colors[i])
                if j == 0:
                    legend_handles.append(a)
                    legend_strings.append(legend.replace('_',' '))
                ax.text(x, 0, ' {:.2f}'.format(rms),
                        horizontalalignment='center', verticalalignment='bottom',rotation=90)
        # Set XTicks
        xticks = np.mean(np.arange(len(asds))) + np.arange(len(dofs))*(len(asds)+1)
        xticklabels = [str(dof) for dof in dofs]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel('RMS Levels')
        fig.tight_layout()
        l = ax.legend(legend_handles,legend_strings,bbox_to_anchor=(1,1))
        fig.canvas.draw()
        legend_width = l.get_window_extent().width
        figure_width = fig.bbox.width
        figure_fraction = legend_width/figure_width
        ax_position = ax.get_position()
        ax_position.x1 -= figure_fraction
        ax.set_position(ax_position)
        
    def coherence(self):
        """
        Computes the coherence of a PSD matrix

        Raises
        ------
        ValueError
            If abscissa are not consistent.

        Returns
        -------
        CoherenceArray
            CoherenceArray containing the values of coherence for each function.

        """
        reshaped_array = self.reshape_to_matrix()
        abscissa = reshaped_array[0,0].abscissa
        if not np.allclose(reshaped_array.abscissa,abscissa):
            raise ValueError('All functions must have identical abscissa')
        cpsd_matrix = np.moveaxis(reshaped_array.ordinate,-1,0)
        coherence_matrix = np.moveaxis(sp_coherence(cpsd_matrix),0,-1)
        coherence_array = data_array(
            FunctionTypes.COHERENCE,abscissa = reshaped_array.abscissa,
            ordinate = coherence_matrix, coordinate = reshaped_array.coordinate)
        return coherence_array[self.coordinate]
    
    def angle(self):
        """
        Computes the angle of a PSD matrix

        Returns
        -------
        NDDataArray
            Data array consisting of the angle of each function at each
            frequency line

        """
        return data_array(FunctionTypes.GENERAL,self.abscissa,np.angle(self.ordinate),
                          self.coordinate)
    
    def set_coherence_phase(self,coherence_array,angle_array):
        """
        Sets the coherence and phase of a PSD matrix while maintaining the ASDs

        Parameters
        ----------
        coherence_array : CoherenceArray
            Coherence to which the PSD will be set
        angle_array : NDDataArray
            Phase to which the PSD will be set

        Returns
        -------
        output : PowerSpectralDensityArray
            PSD with coherence and phase matching that of the input argument

        """
        asds = self.get_asd()
        dofs = outer_product(asds.response_coordinate,asds.reference_coordinate)
        reshaped_coherence = coherence_array[dofs]
        reshaped_angle = angle_array[dofs]
        asd_matrix = np.moveaxis(asds.ordinate,-1,0)
        coherence_matrix = np.moveaxis(reshaped_coherence.ordinate,-1,0)
        phase_matrix = np.moveaxis(reshaped_angle.ordinate,-1,0)
        cpsd_matrix = cpsd_from_coh_phs(asd_matrix,coherence_matrix,phase_matrix)
        output = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                            asds[0].abscissa,np.moveaxis(cpsd_matrix,0,-1),
                            dofs)[self.coordinate]
        return output

    @classmethod
    def eye(cls,frequencies,coordinates,rms=None,full_matrix=False,
            breakpoint_frequencies = None, breakpoint_levels = None,
            breakpoint_interpolation = 'lin',
            min_frequency=0.0, max_frequency=None):
        """
        Computes a diagonal CPSD matrix

        Parameters
        ----------
        frequencies : ndarray
            Frequencies at which the CPSD should be constructed
        coordinates : CoordinateArray
            CoordinateArray to use to set the CPSD values
        rms : ndarray, optional
            Value to scale the RMS of each CPSD to
        full_matrix : bool, optional
            If True, a full, square CPSD matrix will be computed.  If False, only
            the ASDs will be computed. The default is False.
        breakpoint_frequencies : iterable, optional
            A list of frequencies that breakpoints are defined at.
        breakpoint_levels : iterable, optional
            A list of levels that breakpoints are defined at
        breakpoint_interpolation : str, optional
            'lin' or 'log' to specify the type of interpolation. The default is
            'lin'.
        min_frequency : float, optional
            Low frequency cutoff for the CPSD.  Frequency lines below this value
            will be set to zero.
        max_frequency : float, optional
            High frequency cutoff for the CPSD.  Frequency lines above this value
            will be set to zero.

        Raises
        ------
        ValueError
            If invalid interpolation is specified, or if RMS is specified with
            inconsistent frequency spacing.

        Returns
        -------
        PowerSpectralDensityArray
            A set of PSDs.

        """
        if breakpoint_frequencies is None or breakpoint_levels is None:
            cpsd = np.ones(frequencies.shape)
        else:
            if breakpoint_interpolation in ['log', 'logarithmic']:
                cpsd = np.interp(np.log(frequencies), np.log(breakpoint_frequencies),
                                 breakpoint_levels, left=0, right=0)
            elif breakpoint_interpolation in ['lin', 'linear']:
                cpsd = np.interp(frequencies, breakpoint_frequencies, breakpoint_levels)
            else:
                raise ValueError('Invalid Interpolation, should be "lin" or "log"')
        
        # Truncate to the minimum frequency
        cpsd[frequencies < min_frequency] = 0
        if not max_frequency is None:
            cpsd[frequencies > max_frequency] = 0

        if not rms is None:
            frequency_spacing = np.mean(np.diff(frequencies))
            if not np.allclose(frequency_spacing,np.diff(frequencies)):
                raise ValueError('In order to specify RMS, the spacing of frequencies must be constant')
            cpsd_rms = np.sqrt(np.sum(cpsd) * frequency_spacing)
            cpsd *= (rms / cpsd_rms)**2
            
        num_channels = coordinates.size
        if full_matrix:
            full_cpsd = np.zeros((num_channels,num_channels,frequencies.size))
            full_cpsd[np.arange(num_channels), np.arange(num_channels),:] = cpsd
            cpsd_coordinates = outer_product(coordinates,coordinates)
        else:
            full_cpsd = np.tile(cpsd,(num_channels,1))
            cpsd_coordinates = np.tile(coordinates[:,np.newaxis],(1,2))
        
        return data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,frequencies,
                          full_cpsd,cpsd_coordinates)
        
    def plot_magnitude_coherence_phase(self,compare_data = None,plot_axes=False,
                                       sharex=True,sharey=True,logx=False,
                                       logy=True,
                                       magnitude_plot_kwargs = {},
                                       coherence_plot_kwargs = {},
                                       angle_plot_kwargs = {},
                                       figure_kwargs = {}):
        """
        Plots the magnitude, coherence, and phase of a CPSD matrix.
        
        Coherence is plotted on the upper triangle, phase on the lower triangle,
        and magnitude on the diagonal.

        Parameters
        ----------
        compare_data : PowerSpectralDensityArray, optional
            An optional dataset to compare against. The default is None.
        plot_axes : bool, optional
            If True, axes tick labels will be plotted.  If false, the plots will
            be pushed right against one another without room for labels.
            The default is False.
        sharex : bool, optional
            If True, all plots will share the same range on the X axis. The
            default is True.
        sharey : bool, optional
            If true, all plots of the same type will share the same range on the
            Y axis. The default is True.
        logx : bool, optional
            If true, the x-axis will be logarithmic. The default is False.
        logy : bool, optional
            If true, the y-axis on magnitude plots will be logrithmic. The
            default is True.
        magnitude_plot_kwargs : dict, optional
            Optional keywards to use when plotting magnitude. The default is {}.
        coherence_plot_kwargs : dict, optional
            Optional keywards to use when plotting coherence. The default is {}.
        angle_plot_kwargs : dict, optional
            Optional keywards to use when plotting phase. The default is {}.
        figure_kwargs : dict, optional
            Optional keywards to use when creating the figure. The default is {}.

        Returns
        -------
        None.

        """
        fig = plt.figure(**figure_kwargs)
        reshaped_array = self.reshape_to_matrix()
        coherence = reshaped_array.coherence()
        phase = reshaped_array.angle()
        if compare_data is not None:
            reshaped_compare_data = compare_data.reshape_to_matrix()
            compare_coherence = reshaped_compare_data.coherence()
            phase_compare = reshaped_compare_data.angle()
        ax = {}
        gs = GridSpec(*reshaped_array.shape,fig,
                      wspace=None if plot_axes else 0,
                      hspace = None if plot_axes else 0)
        for (i,j),function in reshaped_array.ndenumerate():
            if i == j:
                if ((not sharex) and (not sharey)) or i == 0:
                    ax[i,j] = fig.add_subplot(gs[i,j])
                elif (not sharex) and sharey and (i > 0):
                    ax[i,j] = fig.add_subplot(gs[i,j],sharey=ax[0,0])
                elif sharex and (not sharey) and (i > 0):
                    ax[i,j] = fig.add_subplot(gs[i,j],sharex=ax[0,0])
                else:
                    ax[i,j] = fig.add_subplot(gs[i,j],sharex=ax[0,0],
                                              sharey=ax[0,0])
                ax[i,j].plot(function.abscissa,np.abs(function.ordinate),
                             'r',**magnitude_plot_kwargs)
                if compare_data is not None:
                    ax[i,j].plot(reshaped_compare_data[i,j].abscissa,
                                 np.abs(reshaped_compare_data[i,j].ordinate),
                                 color=[1.0,.5,.5],**magnitude_plot_kwargs)
                if logy:
                    ax[i,j].set_yscale('log')
            if i > j:
                ax[i,j] = fig.add_subplot(
                    gs[i,j],
                    sharex=ax[0,0] if (i > 0) and sharex else None,
                    sharey=ax[1,0] if (i > 1) and sharey else None)
                ax[i,j].plot(function.abscissa,phase[i,j].ordinate,
                              'g',**angle_plot_kwargs)
                if compare_data is not None:
                    ax[i,j].plot(phase_compare[i,j].abscissa,
                                 phase_compare[i,j].ordinate,
                                 color=[0,1,0],**angle_plot_kwargs)
                ax[i,j].set_ylim(-np.pi,np.pi)
            if i < j:
                ax[i,j] = fig.add_subplot(
                    gs[i,j],
                    sharex=ax[0,0] if (j > 0) and sharex else None,
                    sharey=ax[0,1] if (j > 1) and sharey else None)
                ax[i,j].plot(function.abscissa,coherence[i,j].ordinate,
                              'b',**coherence_plot_kwargs)
                if compare_data is not None:
                    ax[i,j].plot(compare_coherence[i,j].abscissa,
                                 compare_coherence[i,j].ordinate,
                                 color=[0.5,0.5,1.0],**coherence_plot_kwargs)
                ax[i,j].set_ylim(0,1)
            if logx:
                ax[i,j].set_xscale('log')
            if j == 0:
                ax[i,j].set_ylabel(str(function.response_coordinate))
            if i == reshaped_array.shape[0]-1:
                ax[i,j].set_xlabel(str(function.reference_coordinate))
            if not plot_axes:
                ax[i,j].set_yticklabels([])
                ax[i,j].set_xticklabels([])
                ax[i,j].tick_params(axis='x',direction='in')
                ax[i,j].tick_params(axis='y',direction='in')
                
    def to_rattlesnake_specification(self,filename,coordinate_order = None,
                                     min_frequency = None,
                                     max_frequency = None):
        if coordinate_order is not None:
            coordinate_array = outer_product(coordinate_order)
            reshaped_data = self[coordinate_array]
        else:
            if self.ndim != 2:
                raise ValueError('CPSD Matrix must be 2D to transform to rattlesnake specification')
            if self.shape[0] != self.shape[1]:
                raise ValueError('CPSD Matrix must be square')
            if not np.all(self.coordinate[...,0] == self.coordinate[...,1].T):
                raise ValueError('Row and column coordinates of the CPSD matrix are not ordered identically')
            reshaped_data = self
        if min_frequency is not None or max_frequency is not None:
            if min_frequency is None:
                min_frequency = -np.inf
            if max_frequency is None:
                max_frequency = np.inf
            reshaped_data = reshaped_data.extract_elements_by_abscissa(min_frequency,max_frequency)
        np.savez(filename,
                 f = reshaped_data[0,0].abscissa,
                 cpsd = np.moveaxis(reshaped_data.ordinate,-1,0))
    
class PowerSpectrumArray(NDDataArray):
    """Data array used to store power spectra arrays"""
    def __new__(subtype, shape, nelements, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(subtype, shape, nelements, 2, 'complex128', buffer, offset, strides, order)
        return obj

    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.AUTOSPECTRUM
# def power_spectrum_array(abscissa,ordinate,coordinate,comment1='',comment2='',comment3='',comment4='',comment5=''):
#     pass


class TransferFunctionArray(NDDataArray):
    """Data array used to store transfer functions (for example FRFs)"""
    def __new__(subtype, shape, nelements, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(subtype, shape, nelements, 2, 'complex128', buffer, offset, strides, order)
        return obj

    @staticmethod
    def from_time_data(reference_data: TimeHistoryArray,
                       response_data: TimeHistoryArray,
                       samples_per_average: int = None,
                       overlap: float = 0.0, method: str = 'H1',
                       window=np.array((1.0,)), **timedata2frf_kwargs):
        """
        Computes a transfer function from reference and response time histories

        Parameters
        ----------
        reference_data : TimeHistoryArray
            Time data to be used as a reference
        response_data : TimeHistoryArray
            Time data to be used as responses
        samples_per_average : int, optional
            Number of samples used to split up the signals into averages.  The
            default is None, meaning the data is treated as a single measurement
            frame.
        overlap : float, optional
            The overlap as a fraction of the frame (e.g. 0.5 specifies 50% overlap).
            The default is 0.0, meaning no overlap is used.
        method : str, optional
            The method for creating the frequency response function. 'H1' is 
            default if not specified.
        window : np.ndarray or str, optional
            A 1D ndarray with length samples_per_average that specifies the
            coefficients of the window.  No window is applied if not specified.
            If a string is specified, then the window will be obtained from scipy.
        **timedata2frf_kwargs : various
            Additional keyword arguments that may be passed into the
            timedata2frf function in sdynpy.frf.

        Raises
        ------
        ValueError
            Raised if reference and response functions do not have consistent
            abscissa

        Returns
        -------
        TransferFunctionArray
            A transfer function array computed from the specified references and
            responses.

        """
        ref_data = reference_data.flatten()
        res_data = response_data.flatten()
        ref_ord = ref_data.ordinate
        res_ord = res_data.ordinate
        if not np.allclose(ref_data[0].abscissa,
                           res_data[0].abscissa):
            raise ValueError('Reference and Response Data should have identical abscissa spacing!')
        dt = np.mean(np.diff(ref_data[0].abscissa))
        freq, frf = timedata2frf(ref_ord, res_ord, dt, samples_per_average, overlap,
                                 method, window, **timedata2frf_kwargs)
        # Now construct the transfer function array
        coordinate = outer_product(res_data.coordinate.flatten(),
                                   ref_data.coordinate.flatten())
        return data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                          freq, np.moveaxis(frf, 0, -1), coordinate)

    def compute_mif(self, mif_type, *mif_args, **mif_kwargs):
        """
        Compute a mode indicator functions from the transfer functions

        Parameters
        ----------
        mif_type : str
            Mode indicator function type, one of 'cmif','nmif', or 'mmif'
        *mif_args : list
            Arguments passed to the compute_*mif function
        **mif_kwargs : dict
            Keyword arguments passed to the compute_*mif function

        Raises
        ------
        ValueError
            If an invalid mif name is provided.

        Returns
        -------
        ModeIndicatorFunctionArray
            Mode indicator function

        """
        if mif_type.lower() == 'cmif':
            return self.compute_cmif(*mif_args, **mif_kwargs)
        elif mif_type.lower() == 'mmif':
            return self.compute_mmif(*mif_args, **mif_kwargs)
        elif mif_type.lower() == 'nmif':
            return self.compute_nmif(*mif_args, **mif_kwargs)
        else:
            raise ValueError('Invalid MIF type, must be one of cmif, mmif, or nmif')

    def compute_cmif(self, part='both', tracking=None):
        """
        Computes a complex mode indicator function from the
        TransferFunctionArray

        Parameters
        ----------
        part : str, optional
            Specifies which part(s) of the transfer functions are used to
            compute the CMIF.  Can be 'real', 'imag', or 'both'.  The default
            is 'both'.
        tracking : str or None, optional
            Specifies if any singular value tracking should be used.  Can be
            'left' or 'right'. The default is None.

        Raises
        ------
        ValueError
            Raised if an invalid tracking is specified

        Returns
        -------
        output_array : ModeIndicatorFunctionArray
            Complex Mode Indicator Function
        """
        matrix_form = self.reshape_to_matrix()
        ordinate = np.moveaxis(matrix_form.ordinate, -1, 0)
        if part.lower() == 'imag' or part.lower() == 'imaginary':
            ordinate = ordinate.imag
        if part.lower() == 'real':
            ordinate = ordinate.real
        u, s, vh = np.linalg.svd(ordinate, full_matrices=False, compute_uv=True)
        v = vh.conjugate().transpose(0, 2, 1)
        if not tracking is None:
            u_unshuffled = np.zeros(u.shape, dtype=u.dtype)
            s_unshuffled = np.zeros(s.shape, dtype=s.dtype)
            v_unshuffled = np.zeros(v.shape, dtype=v.dtype)
            if tracking == 'left':
                u_unshuffled[0] = u[0]
                s_unshuffled[0] = s[0]
                v_unshuffled[0] = v[0]
                for line in range(s.shape[0]):
                    if line == 0:
                        continue
                    previous_u = u_unshuffled[line - 1]
                    this_u = u[line]
                    # np.linalg.norm(previous_u-this_u,axis=0)
                    comparison_matrix = mac(previous_u, this_u)
                    current_sorting = np.argmax(comparison_matrix, axis=1)
                    u_unshuffled[line] = u[line][:, current_sorting]
                    s_unshuffled[line] = s[line][current_sorting]
                    v_unshuffled[line] = v[line][:, current_sorting]
            elif tracking == 'right':
                raise NotImplementedError(
                    'Tracking by right singular vector is not yet implemented')
            else:
                raise ValueError('tracking must be None or "left" or "right"')
            u = u_unshuffled
            v = v_unshuffled
            s = s_unshuffled
        output_array = ModeIndicatorFunctionArray((s.shape[1],), s.shape[0])
        output_array.response_coordinate = sdynpy_coordinate.coordinate_array(
            np.arange(s.shape[1]) + 1, 0)
        output_array.ordinate = s.T
        output_array.abscissa = self.abscissa[(
            0,) * (self.abscissa.ndim - 1) + (slice(None, None, None),)]
        return output_array

    def compute_nmif(self, part='real'):
        """
        Computes a normal mode indicator function from the
        TransferFunctionArray

        Parameters
        ----------
        part : str, optional
            Specifies which part(s) of the transfer functions are used to
            compute the NMIF.  Can be 'real' or 'imag'.  The default
            is 'real'.

        Raises
        ------
        ValueError
            Raised if an invalid part is specified

        Returns
        -------
        output_array : ModeIndicatorFunctionArray
            Normal Mode Indicator Function
        """
        ordinate = self.flatten().ordinate
        if part == 'real':
            nmif = np.sum(np.abs(ordinate) * np.abs(np.real(ordinate)),
                          axis=0) / np.sum(np.abs(ordinate)**2, axis=0)
        elif part == 'imag':
            nmif = np.sum(np.abs(ordinate) * np.abs(np.imag(ordinate)),
                          axis=0) / np.sum(np.abs(ordinate)**2, axis=0)
        else:
            raise ValueError('part must be "real" or "imag"')
        output_array = ModeIndicatorFunctionArray((), nmif.size)
        output_array.abscissa = self.abscissa[(
            0,) * (self.abscissa.ndim - 1) + (slice(None, None, None),)]
        output_array.ordinate = nmif
        output_array.response_coordinate = sdynpy_coordinate.coordinate_array(1, 0)
        return output_array

    def compute_mmif(self, part='real', mass_matrix=None):
        """
        Computes a Multi Mode indicator function from the
        TransferFunctionArray

        Parameters
        ----------
        part : str, optional
            Specifies which part(s) of the transfer functions are used to
            compute the NMIF.  Can be 'real' or 'imag'.  The default
            is 'real'.
        mass_matrix : np.ndarray, optional
            Matrix used to compute the MMIF, Identity is used if not specified

        Raises
        ------
        ValueError
            Raised if an invalid part is specified

        Returns
        -------
        output_array : ModeIndicatorFunctionArray
            Multi Mode Indicator Function
        """
        rect_frf = self.reshape_to_matrix()
        ordinate = rect_frf.ordinate.transpose(2, 0, 1)
        real = np.real(ordinate)
        imag = np.imag(ordinate)
        if mass_matrix is None:
            mass_matrix = np.eye(ordinate.shape[1])
        A = real.transpose(0, 2, 1) @ mass_matrix @ real
        B = imag.transpose(0, 2, 1) @ mass_matrix @ imag
        mif_ordinate = np.zeros((ordinate.shape[0], ordinate.shape[-1]))
        if part == 'real':
            for index, (this_a, this_b) in enumerate(zip(A, B)):
                evalue = eigh(this_a, (this_a + this_b), eigvals_only=True)
                mif_ordinate[index] = evalue
        elif part == 'imag':
            for index, (this_a, this_b) in enumerate(zip(A, B)):
                evalue = eigh(this_b, (this_a + this_b), eigenvals_only=True)
                mif_ordinate[index] = evalue
        else:
            raise ValueError('part must be "real" or "imag"')
        output_array = ModeIndicatorFunctionArray(mif_ordinate.shape[-1], mif_ordinate.shape[0])
        output_array.abscissa = self.abscissa[(
            0,) * (self.abscissa.ndim - 1) + (slice(None, None, None),)]
        output_array.ordinate = mif_ordinate.T
        output_array.response_coordinate = sdynpy_coordinate.coordinate_array(
            np.arange(mif_ordinate.shape[1]) + 1, 0)
        return output_array

    def plot(self, one_axis=True, subplots_kwargs={}, plot_kwargs={}):
        """
        Plot the transfer functions

        Parameters
        ----------
        one_axis : bool, optional
            Set to True to plot all data on one axis.  Set to False to plot
            data on multiple subplots.  one_axis can also be set to a 
            matplotlib axis to plot data on an existing axis.  The default is
            True.
        subplots_kwargs : dict, optional
            Keywords passed to the matplotlib subplots function to create the
            figure and axes. The default is {}.
        plot_kwargs : dict, optional
            Keywords passed to the matplotlib plot function. The default is {}.

        Returns
        -------
        axis : matplotlib axis or array of axes
             On which the data were plotted

        """
        if one_axis is True:
            figure, axis = plt.subplots(2, 1, **subplots_kwargs)
            axis[0].plot(self.flatten().abscissa.T, np.angle(
                self.flatten().ordinate.T), **plot_kwargs)
            axis[1].plot(self.flatten().abscissa.T, np.abs(
                self.flatten().ordinate.T), **plot_kwargs)
            axis[1].set_yscale('log')
            axis[0].set_ylabel('Phase')
            axis[1].set_ylabel('Amplitude')
        elif one_axis is False:
            ncols = int(np.floor(np.sqrt(self.size)))
            nrows = int(np.ceil(self.size / ncols))
            figure, axis = plt.subplots(nrows, ncols, **subplots_kwargs)
            for i, (ax, (index, function)) in enumerate(zip(axis.flatten(), self.ndenumerate())):
                ax.plot(function.abscissa.T, np.abs(function.ordinate.T), **plot_kwargs)
                ax.set_ylabel('/'.join([str(v) for i, v in function.coordinate.ndenumerate()]))
                ax.set_yscale('log')
            for ax in axis.flatten()[i + 1:]:
                ax.remove()
        else:
            axis = one_axis
            axis[0].plot(self.flatten().abscissa.T, np.angle(
                self.flatten().ordinate.T), **plot_kwargs)
            axis[1].plot(self.flatten().abscissa.T, np.abs(
                self.flatten().ordinate.T), **plot_kwargs)
        return axis

    def delay_response(self, dt):
        """
        Adjusts the FRF phases as if the response had been shifted `dt` in time

        Parameters
        ----------
        dt : float
            Time shift to apply to the responses

        Returns
        -------
        shifted_transfer_function : TransferFunctionArray
            A copy of the transfer function with the phase shifted
        """
        shifted_transfer_function = self.copy()
        omegas = shifted_transfer_function.abscissa * 2 * np.pi
        shifted_transfer_function.ordinate *= np.exp(-1j * omegas * dt)
        return shifted_transfer_function

    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.FREQUENCY_RESPONSE_FUNCTION

    def substructure_by_constraint_matrix(self, dofs, constraint_matrix):
        """
        Performs frequency based substructuring using the 

        Parameters
        ----------
        dofs : CoordinateArray
            Coordinates to use in the constraints
        constraint_matrix : np.ndarray
            Constraints to apply to the frequency response functions

        Raises
        ------
        ValueError
            If listed degrees of freedom are not found in the function.

        Returns
        -------
        constrained_frfs : TransferFunctionArray
            Constrained Frequency Response Functions

        """
        # Create a rectangular FRF array
        rect_frfs = self.reshape_to_matrix().copy()
        # Extract the reference and response dofs
        response_dofs = rect_frfs[:, 0].response_coordinate
        reference_dofs = rect_frfs[0, :].reference_coordinate
        # Now find the indices of each of the dofs
        reference_indices = np.searchsorted(abs(reference_dofs),
                                            abs(dofs))
        if np.any(abs(reference_dofs[reference_indices]) !=
                  abs(dofs)):
            raise ValueError('Not all constraint degrees of freedom are found in the references')
        response_indices = np.searchsorted(abs(response_dofs),
                                           abs(dofs))
        if np.any(abs(response_dofs[response_indices]) !=
                  abs(dofs)):
            raise ValueError('Not all constraint degrees of freedom are found in the responses')
        # Handle sign flipping
        flip_sign_references = reference_dofs[reference_indices].sign() * dofs.sign()
        flip_sign_responses = response_dofs[response_indices].sign() * dofs.sign()
        # Put together the constraint matrix
        constraint_matrix_responses = np.zeros((constraint_matrix.shape[0], response_dofs.size))
        constraint_matrix_references = np.zeros((constraint_matrix.shape[0], reference_dofs.size))
        constraint_matrix_responses[:, response_indices] = flip_sign_responses * constraint_matrix
        constraint_matrix_references[:,
                                     reference_indices] = flip_sign_references * constraint_matrix
        # Perform the constraint
        H = np.moveaxis(rect_frfs.ordinate, -1, 0)
        H_constrained = H - H @ constraint_matrix_references.T @ np.linalg.solve(
            constraint_matrix_responses @ H @ constraint_matrix_references.T, constraint_matrix_responses @ H)
        rect_frfs.ordinate = np.moveaxis(H_constrained, 0, -1)
        return rect_frfs

    def substructure_by_coordinate(self, dof_pairs):
        """
        Performs frequency based substructuring by constraining pairs of degrees
        of freedom

        Parameters
        ----------
        dof_pairs : CoordinateArray or None
            Pairs of coordinates to constrain together.  To constain a coordinate
            to ground (i.e. fix it so it cannot move), the coordinate should be
            paired with None.

        Returns
        -------
        TransferFunctionArray
            Constrained frequency response functions

        """
        dof_list = []
        constraint_matrix_values = []
        for constraint_index, dof_pair in enumerate(dof_pairs):
            for sign, dof in zip([1, -1], dof_pair):
                if dof is None:
                    continue
                try:
                    index = dof_list.index(abs(dof))
                except ValueError:
                    dof_list.append(abs(dof))
                    index = len(dof_list) - 1
                flip_sign = dof.sign()
                constraint_matrix_values.append((constraint_index, index,
                                                 flip_sign * sign))
        # Now create the final matrix and fill it
        constraint_matrix = np.zeros((len(dof_pairs),
                                      len(dof_list)))
        for row, column, value in constraint_matrix_values:
            constraint_matrix[row, column] = value
        # Apply Constraints
        dof_list = np.array(dof_list).view(sdynpy_coordinate.CoordinateArray)
        return self.substructure_by_constraint_matrix(dof_list, constraint_matrix)

# def transfer_function_array(abscissa,ordinate,coordinate,comment1='',comment2='',comment3='',comment4='',comment5=''):
#     pass


class TransmissibilityArray(NDDataArray):
    """Data array used to store transmissibility data"""
    def __new__(subtype, shape, nelements, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(subtype, shape, nelements, 2, 'complex128', buffer, offset, strides, order)
        return obj

    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.TRANSMISIBILITY
# def transmisibility_array(abscissa,ordinate,coordinate,comment1='',comment2='',comment3='',comment4='',comment5=''):
#     pass


class CoherenceArray(NDDataArray):
    """Data array used to store coherence data"""
    def __new__(subtype, shape, nelements, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(subtype, shape, nelements, 2, 'float64', buffer, offset, strides, order)
        return obj

    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.COHERENCE


class MultipleCoherenceArray(NDDataArray):
    """Data array used to store coherence data"""
    def __new__(subtype, shape, nelements, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(subtype, shape, nelements, 1, 'float64', buffer, offset, strides, order)
        return obj

    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.MULTIPLE_COHERENCE

# def coherence_array(abscissa,ordinate,coordinate,comment1='',comment2='',comment3='',comment4='',comment5=''):
#     pass


class CorrelationArray(NDDataArray):
    """Data array used to store correlation data"""
    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.CROSSCORRELATION

# def correlation_array(abscissa,ordinate,coordinate,comment1='',comment2='',comment3='',comment4='',comment5=''):
#     pass


class ModeIndicatorFunctionArray(NDDataArray):
    """Mode indicator function (CMIF, NMIF, or NMIF)"""
    def __new__(subtype, shape, nelements, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(subtype, shape, nelements, 1, 'float64', buffer, offset, strides, order)
        return obj

    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.MODE_INDICATOR_FUNCTION

# def mode_indicator_function_array(abscissa,ordinate,coordinate,comment1='',comment2='',comment3='',comment4='',comment5=''):
#     pass


_function_type_class_map = {FunctionTypes.GENERAL: NDDataArray,
                            FunctionTypes.TIME_RESPONSE: TimeHistoryArray,
                            FunctionTypes.AUTOSPECTRUM: PowerSpectrumArray,
                            FunctionTypes.CROSSSPECTRUM: PowerSpectrumArray,
                            FunctionTypes.FREQUENCY_RESPONSE_FUNCTION: TransferFunctionArray,
                            FunctionTypes.TRANSMISIBILITY: TransmissibilityArray,
                            FunctionTypes.COHERENCE: CoherenceArray,
                            FunctionTypes.AUTOCORRELATION: CorrelationArray,
                            FunctionTypes.CROSSCORRELATION: CorrelationArray,
                            FunctionTypes.POWER_SPECTRAL_DENSITY: PowerSpectralDensityArray,
                            FunctionTypes.ENERGY_SPECTRAL_DENSITY: PowerSpectralDensityArray,
                            # FunctionTypes.PROBABILITY_DENSITY_FUNCTION, : ,
                            FunctionTypes.SPECTRUM: SpectrumArray,
                            # FunctionTypes.CUMULATIVE_FREQUENCY_DISTRIBUTION, : ,
                            # FunctionTypes.PEAKS_VALLEY, : ,
                            # FunctionTypes.STRESS_PER_CYCLE, : ,
                            # FunctionTypes.STRAIN_PER_CYCLE, : ,
                            # FunctionTypes.ORBIT, : ,
                            FunctionTypes.MODE_INDICATOR_FUNCTION: ModeIndicatorFunctionArray,
                            # FunctionTypes.FORCE_PATTERN, : ,
                            # FunctionTypes.PARTIAL_POWER, : ,
                            # FunctionTypes.PARTIAL_COHERENCE, : ,
                            # FunctionTypes.EIGENVALUE, : ,
                            # FunctionTypes.EIGENVECTOR, : ,
                            # FunctionTypes.SHOCK_RESPONSE_SPECTRUM, : ,
                            # FunctionTypes.FINITE_IMPULSE_RESPONSE_FILTER, : ,
                            FunctionTypes.MULTIPLE_COHERENCE: MultipleCoherenceArray,
                            # FunctionTypes.ORDER_FUNCTION, : ,
                            # FunctionTypes.PHASE_COMPENSATION,  : ,
                            }


def data_array(data_type, abscissa, ordinate, coordinate, comment1='', comment2='', comment3='', comment4='', comment5=''):
    """
    Helper function to create a data array object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the NDDataArray object.

    Parameters
    ----------
    data_type : FunctionTypes
        Type of data array that will be created
    abscissa : np.ndarray
        Numpy array specifying the abscissa of the function
    ordinate : np.ndarray
        Numpy array specifying the ordinate of the function
    coordinate : CoordinateArray
        Coordinate for each data in the data array
    comment1 : np.ndarray, optional
        Comment used to describe the data in the data array. The default is ''.
    comment2 : np.ndarray, optional
        Comment used to describe the data in the data array. The default is ''.
    comment3 : np.ndarray, optional
        Comment used to describe the data in the data array. The default is ''.
    comment4 : np.ndarray, optional
        Comment used to describe the data in the data array. The default is ''.
    comment5 : np.ndarray, optional
        Comment used to describe the data in the data array. The default is ''.

    Returns
    -------
    obj : NDDataArray or subclass
        The constructed NDDataArray (or subclass) object
    """
    cls = _function_type_class_map[data_type]
    nelem = ordinate.shape[-1]
    shape = ordinate.shape[:-1]
    if data_type is FunctionTypes.GENERAL:
        obj = cls(shape,nelem,coordinate.shape[-1])
    else:
        obj = cls(shape, nelem)
    obj.ordinate = ordinate
    obj.abscissa = abscissa
    obj.comment1 = comment1
    obj.comment2 = comment2
    obj.comment3 = comment3
    obj.comment4 = comment4
    obj.comment5 = comment5
    num_coords = obj.dtype['coordinate'].shape[0]
    if coordinate.ndim == 0:
        obj.coordinate[:] = coordinate
    else:
        obj.coordinate[:] = coordinate[..., :num_coords]
    return obj


from_unv = NDDataArray.from_unv
from_uff = from_unv
load = NDDataArray.load

def from_imat_struct(imat_fn_struct, squeeze=True):
    """
    Constructs a NDDataArray from an imat_fn class saved to a Matlab structure

    In IMAT, a structure can be created from an `imat_fn` by using the get()
    function.  This can then be saved to a .mat file and loaded using
    `scipy.io.loadmat`.  The output from loadmat can be passed into this function

    Parameters
    ----------
    imat_fn_struct : np.ndarray
        structure from loadmat containing data from an imat_fn
    squeeze : bool, optional
        If True, return a single NDDataArray object or subclass if only one
        function type exists in the data.  Otherwise, it will return a list of
        length one.  If more than one function type exists, a list of
        NDDataArray objects will be returned regardless of the value of
        `squeeze`.  Default is True.

    Returns
    -------
    return_functions : NDDataArray or list of NDDataArray
        Returns a list of NDDataArray objects if `squeeze` is false, or a single
        NDDataArray object if `squeeze` is True, unless there are multiple
        function types stored in the structure.

    """

    # Get function types
    fn_types = np.array(imat_fn_struct['FunctionType'][0, 0].tolist())
    fn_types = fn_types.reshape(*fn_types.shape[:-1])
    # Separate into the different types of functions
    function_type_dict = {}
    for i, fn_type in enumerate(fn_types.flatten()):
        fn_type_enum = _imat_function_type_map[fn_type]
        if not fn_type_enum in function_type_dict:
            function_type_dict[fn_type_enum] = []
        function_type_dict[fn_type_enum].append(i)
    return_functions = []
    abscissa = imat_fn_struct['Abscissa'][0, 0]
    abscissa = abscissa.reshape(abscissa.shape[0], -1)
    ordinate = imat_fn_struct['Ordinate'][0, 0]
    ordinate = ordinate.reshape(ordinate.shape[0], -1)
    reference_coords = np.array(imat_fn_struct['ReferenceCoord'][0, 0].tolist())
    if reference_coords.size > 0:
        reference_coords = reference_coords.reshape(*reference_coords.shape[:-1]).flatten()
    else:
        reference_coords = np.zeros(reference_coords.shape[:-1], dtype='<U1').flatten()
    response_coords = np.array(imat_fn_struct['ResponseCoord'][0, 0].tolist())
    if response_coords.size > 0:
        response_coords = response_coords.reshape(*response_coords.shape[:-1]).flatten()
    else:
        response_coords = np.zeros(response_coords.shape[:-1], dtype='<U1').flatten()
    comment_1 = np.array(imat_fn_struct['IDLine1'][0, 0].tolist())
    if comment_1.size > 0:
        comment_1 = comment_1.reshape(*comment_1.shape[:-1]).flatten()
    else:
        comment_1 = np.zeros(comment_1.shape[:-1], dtype='<U1').flatten()
    comment_2 = np.array(imat_fn_struct['IDLine2'][0, 0].tolist())
    if comment_2.size > 0:
        comment_2 = comment_1.reshape(*comment_2.shape[:-1]).flatten()
    else:
        comment_2 = np.zeros(comment_2.shape[:-1], dtype='<U1').flatten()
    comment_3 = np.array(imat_fn_struct['IDLine3'][0, 0].tolist())
    if comment_3.size > 0:
        comment_3 = comment_3.reshape(*comment_3.shape[:-1]).flatten()
    else:
        comment_3 = np.zeros(comment_3.shape[:-1], dtype='<U1').flatten()
    comment_4 = np.array(imat_fn_struct['IDLine4'][0, 0].tolist())
    if comment_4.size > 0:
        comment_4 = comment_4.reshape(*comment_4.shape[:-1]).flatten()
    else:
        comment_4 = np.zeros(comment_4.shape[:-1], dtype='<U1').flatten()
    comment_5 = np.zeros(comment_4.shape, dtype='<U1')
    all_coords = sdynpy_coordinate.coordinate_array(string_array=np.concatenate(
        (response_coords[:, np.newaxis], reference_coords[:, np.newaxis]), axis=-1))
    for fn_type, indices in function_type_dict.items():
        return_functions.append(
            data_array(fn_type, abscissa[:, indices].T, ordinate[:, indices].T,
                       all_coords[indices], comment_1[indices], comment_2[indices],
                       comment_3[indices], comment_4[indices], comment_5[indices])
        )
    if len(return_functions) == 1 and squeeze:
        return_functions = return_functions[0]
    return return_functions

class ComplexType(Enum):
    """Enumeration containing the various ways to plot complex data"""
    REAL = 0
    IMAGINARY = 1
    MAGNITUDE = 2
    PHASE = 3
    REALIMAG = 4
    MAGPHASE = 5

class GUIPlot(QMainWindow):
    """An iteractive plot window allowing users to visualize data"""

    def __init__(self, data_array: NDDataArray, compare_data=None):
        """
        Create a GUIPlot window to visualize data.

        Two datasets can be overlaid by providing a second identically sized
        dataset as the `compare_data` argument.

        Parameters
        ----------
        data_array : NDDataArray
            Data to visualize
        compare_data : NDDataArray, optional
            Data to compare.  Default is None, which results in no comparison
            data plotted.

        Returns
        -------
        None.

        """
        super(GUIPlot, self).__init__()
        uic.loadUi(os.path.join(os.path.abspath(os.path.dirname(
            os.path.abspath(__file__))), 'GUIPlot.ui'), self)
        for index, fn in data_array.ndenumerate():
            this_row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(this_row)
            try:
                self.tableWidget.item(this_row, 0).setText(str(index))
            except AttributeError:
                item = QtWidgets.QTableWidgetItem(str(index))
                self.tableWidget.setItem(this_row, 0, item)
            try:
                self.tableWidget.item(this_row, 1).setText(str(fn.response_coordinate))
            except AttributeError:
                item = QtWidgets.QTableWidgetItem(str(fn.response_coordinate))
                self.tableWidget.setItem(this_row, 1, item)
            if fn.dtype['coordinate'].shape[0] > 1:
                try:
                    self.tableWidget.item(this_row, 2).setText(str(fn.reference_coordinate))
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(fn.reference_coordinate))
                    self.tableWidget.setItem(this_row, 2, item)
            try:
                self.tableWidget.item(this_row, 3).setText(
                    str(_imat_function_type_inverse_map[fn.function_type]))
            except AttributeError:
                item = QtWidgets.QTableWidgetItem(
                    str(_imat_function_type_inverse_map[fn.function_type]))
                self.tableWidget.setItem(this_row, 3, item)
        self.tableWidget.resizeColumnsToContents()
        # Adjust the default plotting
        self.data = data_array.flatten()
        self.data_original_shape = data_array.shape
        if compare_data is None:
            self.compare_data = None
            self.cm = cm.Dark2
            self.cm_mod = 8
        else:
            self.compare_data = compare_data.flatten()
            self.cm = cm.Paired
            self.cm_mod = 12
        if self.data[0].function_type in [FunctionTypes.GENERAL, FunctionTypes.TIME_RESPONSE,
                                          FunctionTypes.COHERENCE, FunctionTypes.AUTOCORRELATION,
                                          FunctionTypes.CROSSCORRELATION, FunctionTypes.MODE_INDICATOR_FUNCTION,
                                          FunctionTypes.MULTIPLE_COHERENCE]:
            complex_type = ComplexType.REAL
            self.abscissa_log = False
            self.ordinate_log = False
        elif self.data[0].function_type in [FunctionTypes.AUTOSPECTRUM, FunctionTypes.POWER_SPECTRAL_DENSITY,
                                            FunctionTypes.SHOCK_RESPONSE_SPECTRUM, FunctionTypes.ENERGY_SPECTRAL_DENSITY]:
            complex_type = ComplexType.MAGNITUDE
            self.abscissa_log = False
            self.ordinate_log = True
        elif self.data[0].function_type in [FunctionTypes.CROSSSPECTRUM, FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                                            FunctionTypes.TRANSMISIBILITY, FunctionTypes.SPECTRUM]:
            complex_type = ComplexType.MAGPHASE
            self.abscissa_log = False
            self.ordinate_log = True
        else:
            # print('Unknown Function Type {:}'.format(self.data[0].function_type))
            complex_type = ComplexType.REAL
            self.abscissa_log = False
            self.ordinate_log = False
        self.complex_types = {ComplexType.IMAGINARY: self.actionImaginary,
                              ComplexType.MAGNITUDE: self.actionMagnitude,
                              ComplexType.MAGPHASE: self.actionMagnitude_Phase,
                              ComplexType.PHASE: self.actionPhase,
                              ComplexType.REAL: self.actionReal,
                              ComplexType.REALIMAG: self.actionReal_Imag}
        self.complex_function_maps = {ComplexType.IMAGINARY: (np.imag,),
                                      ComplexType.MAGNITUDE: (np.abs,),
                                      ComplexType.MAGPHASE: (np.angle, np.abs),
                                      ComplexType.PHASE: (np.angle,),
                                      ComplexType.REAL: (np.real,),
                                      ComplexType.REALIMAG: (np.real, np.imag)}
        self.complex_labels = {np.imag: 'Imag',
                               np.abs: 'Mag',
                               np.angle: 'Angle',
                               np.real: 'Real'}
        for ct, action in self.complex_types.items():
            action.setChecked(False)
        self.complex_types[complex_type].setChecked(True)
        self.actionAbscissa_Log.setChecked(self.abscissa_log)
        self.actionOrdinate_Log.setChecked(self.ordinate_log)
        self.menuShare_Axes.setEnabled(False)
        self.pushButton.setEnabled(False)
        self.actionOverlay.setEnabled(False)  # TODO Remove when you implement non-overlapping plots
        self.connect_callbacks()
        # Set the first plot
        self.tableWidget.selectRow(0)
        self.setWindowTitle('GUIPlot')
        self.show()

    def connect_callbacks(self):
        """
        Connects the callback functions to events

        Returns
        -------
        None.

        """
        self.tableWidget.itemSelectionChanged.connect(self.selection_changed)
        self.pushButton.clicked.connect(self.update)
        self.actionImaginary.triggered.connect(self.set_imaginary)
        self.actionMagnitude.triggered.connect(self.set_magnitude)
        self.actionMagnitude_Phase.triggered.connect(self.set_magnitude_phase)
        self.actionPhase.triggered.connect(self.set_phase)
        self.actionReal.triggered.connect(self.set_real)
        self.actionReal_Imag.triggered.connect(self.set_real_imag)
        self.actionOrdinate_Log.triggered.connect(self.update_ordinate_log)
        self.actionAbscissa_Log.triggered.connect(self.update_abscissa_log)
        self.checkBox.clicked.connect(self.update_checkbox)

    def update(self):
        """
        Updates the figure in the GUIPlot

        Returns
        -------
        None.

        """
        # print('Updating')
        select = self.tableWidget.selectionModel()
        # print('Has Selection {:}'.format(select.hasSelection()))
        row_indices = [val.row() for val in select.selectedRows()]
        # print('Rows Selected {:}'.format(row_indices))
        # Check if we want to plot them all on one plot
        # Get existing xaxis to keep it consistent
        try:
            xrange = self.graphicsLayoutWidget.getItem(0, 0).getViewBox().viewRange()[0]
        except AttributeError:
            xrange = None
        # print(xrange)
        self.graphicsLayoutWidget.clear()
        if self.actionOverlay.isChecked():
            # print('Single Plot')
            # Figure out which is checked
            checked_complex_type = [key for key,
                                    val in self.complex_types.items() if val.isChecked()][0]
            plots = []
            for i, complex_fn in enumerate(self.complex_function_maps[checked_complex_type]):
                plot = self.graphicsLayoutWidget.addPlot(
                    i, 0, labels={'left': self.complex_labels[complex_fn]})
                if i > 0:
                    plot.setXLink(plots[0])
                else:
                    plot.addLegend()
                for j, index in enumerate(row_indices):
                    data_entry = self.data[index]
                    original_index = np.unravel_index(index, self.data_original_shape)
                    legend_entry = '{:} {:}'.format(
                        original_index, data_entry.coordinate).replace("'", '')
                    pen = pyqtgraph.mkPen(
                        color=[int(255 * v) for v in self.cm(j * (1 if self.compare_data is None else 2) % self.cm_mod)])
                    plot.plot(x=data_entry.abscissa, y=complex_fn(data_entry.ordinate) *
                              (180 / np.pi if complex_fn is np.angle else 1), name=legend_entry, pen=pen)
                    if not self.compare_data is None:
                        compare_data_entry = self.compare_data[index]
                        pen = pyqtgraph.mkPen(color=[int(255 * v)
                                              for v in self.cm((2 * j + 1) % self.cm_mod)])
                        plot.plot(x=compare_data_entry.abscissa, y=complex_fn(compare_data_entry.ordinate) * (
                            180 / np.pi if complex_fn is np.angle else 1), name=legend_entry + ' Comparison', pen=pen)
                plot.setLogMode(self.abscissa_log,
                                False if complex_fn is np.angle else self.ordinate_log)
                if not xrange is None:
                    plot.setXRange(*xrange, padding=0.0)
                plots.append(plot)

    def update_data(self, new_data, new_compare_data=None):
        original_coordinate = self.data.coordinate.reshape(*self.data_original_shape, -1)
        new_coordinate = new_data.coordinate
        if ((original_coordinate.shape != new_coordinate.shape) or
                (np.any(original_coordinate != new_coordinate))):
            # Redo the table
            self.tableWidget.blockSignals(True)
            self.tableWidget.clear()
            self.tableWidget.setRowCount(0)
            for index, fn in new_data.ndenumerate():
                this_row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(this_row)
                try:
                    self.tableWidget.item(this_row, 0).setText(str(index))
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(index))
                    self.tableWidget.setItem(this_row, 0, item)
                try:
                    self.tableWidget.item(this_row, 1).setText(str(fn.response_coordinate))
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(str(fn.response_coordinate))
                    self.tableWidget.setItem(this_row, 1, item)
                if fn.dtype['coordinate'].shape[0] > 1:
                    try:
                        self.tableWidget.item(this_row, 2).setText(str(fn.reference_coordinate))
                    except AttributeError:
                        item = QtWidgets.QTableWidgetItem(str(fn.reference_coordinate))
                        self.tableWidget.setItem(this_row, 2, item)
                try:
                    self.tableWidget.item(this_row, 3).setText(
                        str(_imat_function_type_inverse_map[fn.function_type]))
                except AttributeError:
                    item = QtWidgets.QTableWidgetItem(
                        str(_imat_function_type_inverse_map[fn.function_type]))
                    self.tableWidget.setItem(this_row, 3, item)
            self.tableWidget.resizeColumnsToContents()
            self.tableWidget.blockSignals(False)

        self.data = new_data.flatten()
        self.data_original_shape = new_data.shape

        if new_compare_data is None:
            self.compare_data = None
            self.cm = cm.Dark2
            self.cm_mod = 8
        else:
            self.compare_data = new_compare_data.flatten()
            self.cm = cm.Paired
            self.cm_mod = 12

        self.update()

    def selection_changed(self):
        """Called when the selected functions is changed"""
        if self.checkBox.isChecked():
            self.update()

    def deselect_all_complex_types_except(self, complex_type):
        """
        Deselects all complex types except the specified type.

        Makes the checkboxes in the menu act like radiobuttons

        Parameters
        ----------
        complex_type : ComplexType
            Enumeration specifying which complex plot type is selected

        Returns
        -------
        None.

        """
        for ct, action in self.complex_types.items():
            action.blockSignals(True)
            if ct is complex_type:
                action.setChecked(True)
            else:
                action.setChecked(False)
            action.blockSignals(False)
        if self.checkBox.isChecked():
            self.update()

    def set_imaginary(self):
        """Sets the complex type to imaginary"""
        self.deselect_all_complex_types_except(ComplexType.IMAGINARY)

    def set_real(self):
        """Sets the complex type to real"""
        self.deselect_all_complex_types_except(ComplexType.REAL)

    def set_magnitude(self):
        """Sets the complex type to magnitude"""
        self.deselect_all_complex_types_except(ComplexType.MAGNITUDE)

    def set_phase(self):
        """Sets the complex type to phase"""
        self.deselect_all_complex_types_except(ComplexType.PHASE)

    def set_magnitude_phase(self):
        """Sets the complex type to magnitude and phase"""
        self.deselect_all_complex_types_except(ComplexType.MAGPHASE)

    def set_real_imag(self):
        """Sets the complex type to real and imaginary"""
        self.deselect_all_complex_types_except(ComplexType.REALIMAG)

    def update_abscissa_log(self):
        """Updates whether the abscissa should be plotted as log scale"""
        self.abscissa_log = self.actionAbscissa_Log.isChecked()
        if self.checkBox.isChecked():
            self.update()

    def update_ordinate_log(self):
        """Updates whether the ordinate should be plotted as log scale"""
        self.ordinate_log = self.actionOrdinate_Log.isChecked()
        if self.checkBox.isChecked():
            self.update()

    def update_checkbox(self):
        """Disables the update button if set to auto-update"""
        self.pushButton.setEnabled(not self.checkBox.isChecked())


class MPLCanvas(FigureCanvas):
    # This is a custom widget that can be used to put plots into a GUI window
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axis = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class MPLMultiCanvas(FigureCanvas):
    # This is a custom widget that can be used to put plots into a GUI window
    def __init__(self, parent=None, width=5, height=4, dpi=100, subplots=(1, 1), ignore_subplots=[]):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = np.empty(subplots, dtype=object)
        for i in range(subplots[0]):
            for j in range(subplots[1]):
                index = i * subplots[1] + j + 1
                if index in ignore_subplots:
                    self.axes[i, j] = None
                else:
                    self.axes[i, j] = self.fig.add_subplot(subplots[0], subplots[1], index)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class CPSDPlot(QMainWindow):

    class DataType(Enum):
        MAGNITUDE = 1
        COHERENCE = 2
        PHASE = 4
        REAL = 8
        IMAGINARY = 16

    def __init__(self, data: PowerSpectralDensityArray,
                 compare_data: PowerSpectralDensityArray = None):
        if not data.validate_common_abscissa(rtol=1, atol=1e-8):
            raise ValueError('Data must have common abscissa')
        if not compare_data is None:
            if data.ordinate.shape != compare_data.ordinate.shape:
                raise ValueError('compare_data.ordinate must have the same size as data.ordinate ({:} != {:})'.format(
                    compare_data.ordinate.shape, data.ordinate.shape))
        super().__init__()
        self.abscissa = data.flatten()[0].abscissa
        self.matrix = np.moveaxis(data.reshape_to_matrix().ordinate, -1, 0)
        self.compare_matrix = None if compare_data is None else np.moveaxis(
            compare_data.reshape_to_matrix().ordinate, -1, 0)
        self.cm = cm.Dark2 if compare_data is None else cm.Paired
        self.coh_matrix = sp_coherence(self.matrix)
        self.compare_coh_matrix = None if compare_data is None else sp_coherence(
            self.compare_matrix)
        self.selected_function = None
        self.plotted_function = None
        self.plot_type_bits = None
        self.select_start = None
        self.select_rectangles = None
        self.plot_rectangles = None
        self.initUI()
        self.connectUI()
        self.show()
        self.init_matrix_plot()

    def initUI(self):
        # Main Widget
        self.main_widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.main_widget)
        # Figure Control Groupbox
        self.fig_control_groupbox = QGroupBox(self.main_widget)
        self.fig_control_groupbox_layout = QVBoxLayout(self.fig_control_groupbox)
        # Figure Selection Group Box
        self.function_select_groupbox = QGroupBox(self.fig_control_groupbox)
        self.function_select_groupbox_layout = QVBoxLayout()
        self.function_select_groupbox.setLayout(self.function_select_groupbox_layout)
        self.fig_control_groupbox_layout.addWidget(self.function_select_groupbox)
        self.selection_options_groupbox = QGroupBox(self.fig_control_groupbox)
        self.selection_options_groupbox_layout = QVBoxLayout(self.selection_options_groupbox)
        # Grid Layout for Selection Buttons
        self.selection_options_button_layout = QGridLayout()
        self.diagonal_select_button = QPushButton(self.selection_options_groupbox)
        self.selection_options_button_layout.addWidget(self.diagonal_select_button, 0, 0, 1, 1)
        self.clear_selection_button = QPushButton(self.selection_options_groupbox)
        self.selection_options_button_layout.addWidget(self.clear_selection_button, 2, 2, 1, 1)
        self.upper_tri_select_button = QPushButton(self.selection_options_groupbox)
        self.selection_options_button_layout.addWidget(self.upper_tri_select_button, 0, 1, 1, 1)
        self.invert_select_button = QPushButton(self.selection_options_groupbox)
        self.selection_options_button_layout.addWidget(self.invert_select_button, 2, 1, 1, 1)
        self.plotted_select_button = QPushButton(self.selection_options_groupbox)
        self.selection_options_button_layout.addWidget(self.plotted_select_button, 2, 0, 1, 1)
        self.lower_tri_select_button = QPushButton(self.selection_options_groupbox)
        self.selection_options_button_layout.addWidget(self.lower_tri_select_button, 0, 2, 1, 1)
        self.diagonal_deselect_button = QPushButton(self.selection_options_groupbox)
        self.selection_options_button_layout.addWidget(self.diagonal_deselect_button, 1, 0, 1, 1)
        self.upper_tri_deselect_button = QPushButton(self.selection_options_groupbox)
        self.selection_options_button_layout.addWidget(self.upper_tri_deselect_button, 1, 1, 1, 1)
        self.lower_tri_deselect_button = QPushButton(self.selection_options_groupbox)
        self.selection_options_button_layout.addWidget(self.lower_tri_deselect_button, 1, 2, 1, 1)
        self.selection_options_groupbox_layout.addLayout(self.selection_options_button_layout)
        self.matrix_select_checkbox = QCheckBox(self.selection_options_groupbox)
        self.selection_options_groupbox_layout.addWidget(self.matrix_select_checkbox)
        self.fig_control_groupbox_layout.addWidget(self.selection_options_groupbox)
        # Plotting Options Groupbox
        self.plotting_options_groupbox = QGroupBox(self.fig_control_groupbox)
        self.plotting_options_groupbox_layout = QVBoxLayout(self.plotting_options_groupbox)
        # Grid layout for plot buttons
        self.plotting_layout = QGridLayout()
        self.plot_selected_button = QPushButton(self.plotting_options_groupbox)
        self.plotting_layout.addWidget(self.plot_selected_button, 0, 0, 1, 1)
        self.plotting_mode_layout = QVBoxLayout()
        # Radio Buttons for plotting method
        self.matrix_mode_button = QRadioButton(self.plotting_options_groupbox)
        self.plotting_mode_layout.addWidget(self.matrix_mode_button)
        self.matrix_mode_button.setChecked(True)
        self.sequential_mode_button = QRadioButton(self.plotting_options_groupbox)
        self.plotting_mode_layout.addWidget(self.sequential_mode_button)
        self.plotting_layout.addLayout(self.plotting_mode_layout, 0, 1, 1, 1)
        self.plotting_options_groupbox_layout.addLayout(self.plotting_layout)
        # Grid Layout for the function types
        self.function_type_layout = QGridLayout()
        self.phase_checkbox = QCheckBox(self.plotting_options_groupbox)
        self.function_type_layout.addWidget(self.phase_checkbox, 0, 2, 1, 1)
        self.magnitude_checkbox = QCheckBox(self.plotting_options_groupbox)
        self.function_type_layout.addWidget(self.magnitude_checkbox, 0, 0, 1, 1)
        self.coherence_checkbox = QCheckBox(self.plotting_options_groupbox)
        self.function_type_layout.addWidget(self.coherence_checkbox, 0, 1, 1, 1)
        self.real_checkbox = QCheckBox(self.plotting_options_groupbox)
        self.function_type_layout.addWidget(self.real_checkbox, 1, 0, 1, 1)
        self.imaginary_checkbox = QCheckBox(self.plotting_options_groupbox)
        self.function_type_layout.addWidget(self.imaginary_checkbox, 1, 1, 1, 1)
        self.function_type_checkboxes = [self.magnitude_checkbox,
                                         self.coherence_checkbox,
                                         self.phase_checkbox,
                                         self.real_checkbox,
                                         self.imaginary_checkbox]
        # Set them so they are tristate
        for box in [self.phase_checkbox, self.magnitude_checkbox, self.coherence_checkbox,
                    self.real_checkbox, self.imaginary_checkbox]:
            box.setTristate(True)
        self.plotting_options_groupbox_layout.addLayout(self.function_type_layout)
        self.fig_control_groupbox_layout.addWidget(self.plotting_options_groupbox)
        # Finish setting up the main window
        self.main_layout.addWidget(self.fig_control_groupbox)
        self.setCentralWidget(self.main_widget)
        # Set up the menu bar
        # self.menubar = QMenuBar(self)
        # self.menubar.setGeometry(QRect(0, 0, 742, 21))
        # self.menubar.setAccessibleName("")
        # self.menuFile = QMenu(self.menubar)
        # self.menuFigure = QMenu(self.menubar)
        # self.setMenuBar(self.menubar)
        self.functions_dock = QDockWidget(self)
        self.functions_dock.setFeatures(QDockWidget.DockWidgetFloatable |
                                        QDockWidget.DockWidgetMovable)
        self.functions_dock_main = QWidget()
        self.functions_dock_main_layout = QVBoxLayout(self.functions_dock_main)
        self.functions_groupbox = QGroupBox(self.functions_dock_main)
        self.functions_groupbox_layout = QVBoxLayout()
        self.functions_groupbox.setLayout(self.functions_groupbox_layout)
        self.functions_dock_main_layout.addWidget(self.functions_groupbox)
        self.functions_dock.setWidget(self.functions_dock_main)
        self.addDockWidget(Qt.DockWidgetArea(1), self.functions_dock)

        # self.actionLoad_Matrix = QAction(self)
        # self.actionLoad_Matrix.setObjectName("actionLoad_Matrix")
        # self.actionExit = QAction(self)
        # self.actionExit.setObjectName("actionExit")
        # self.actionSave_Figure = QAction(self)
        # self.actionSave_Figure.setObjectName("actionSave_Figure")
        # self.menuFile.addAction(self.actionLoad_Matrix)
        # self.menuFile.addSeparator()
        # self.menuFile.addAction(self.actionExit)
        # self.menuFigure.addAction(self.actionSave_Figure)
        # self.menubar.addAction(self.menuFile.menuAction())
        # self.menubar.addAction(self.menuFigure.menuAction())

        self.settext()

    def settext(self):
        self.setWindowTitle('Cross-power Spectral Density Matrix Viewer')
        self.fig_control_groupbox.setTitle("Figure Control")
        self.function_select_groupbox.setTitle("Function Select")
        self.selection_options_groupbox.setTitle("Selection Options")
        self.diagonal_select_button.setText("Select Diagonal")
        self.clear_selection_button.setText("Clear Selection")
        self.upper_tri_select_button.setText("Select Upper Triangle")
        self.invert_select_button.setText("Invert Selection")
        self.plotted_select_button.setText("Select Plotted")
        self.lower_tri_select_button.setText("Select Lower Triangle")
        self.diagonal_deselect_button.setText("Deselect Diagonal")
        self.upper_tri_deselect_button.setText("Deselect Upper Triangle")
        self.lower_tri_deselect_button.setText("Deselect Lower Triangle")
        self.matrix_select_checkbox.setText("Matrix Select Mode")
        self.plotting_options_groupbox.setTitle("Plotting Options")
        self.plot_selected_button.setText("Plot Selected")
        self.matrix_mode_button.setText("Matrix Mode")
        self.sequential_mode_button.setText("Sequential Mode")
        self.phase_checkbox.setText("Phase")
        self.magnitude_checkbox.setText("Magnitude")
        self.coherence_checkbox.setText("Coherence")
        self.real_checkbox.setText("Real")
        self.imaginary_checkbox.setText("Imaginary")
        # self.menuFile.setTitle("File")
        # self.menuFigure.setTitle("Figure")
        self.functions_groupbox.setTitle("Functions")
        # self.actionLoad_Matrix.setText("Load Matrix...")
        # self.actionExit.setText("Exit")
        # self.actionSave_Figure.setText("Save Figure...")

    def connectUI(self):
        # self.actionLoad_Matrix.triggered.connect(self.load)
        # self.actionExit.triggered.connect(self.quit)
        # Selection Method checkbox
        self.matrix_select_checkbox.stateChanged.connect(self.state_changed)
        # Plot type checkboxes
        self.magnitude_checkbox.stateChanged.connect(self.magnitude_state)
        self.coherence_checkbox.stateChanged.connect(self.coherence_state)
        self.phase_checkbox.stateChanged.connect(self.phase_state)
        self.real_checkbox.stateChanged.connect(self.real_state)
        self.imaginary_checkbox.stateChanged.connect(self.imaginary_state)
        # Selection Buttons
        self.diagonal_select_button.clicked.connect(self.select_diagonal)
        self.diagonal_deselect_button.clicked.connect(self.deselect_diagonal)
        self.upper_tri_select_button.clicked.connect(self.select_upper_triangular)
        self.upper_tri_deselect_button.clicked.connect(self.deselect_upper_triangular)
        self.lower_tri_select_button.clicked.connect(self.select_lower_triangular)
        self.lower_tri_deselect_button.clicked.connect(self.deselect_lower_triangular)
        self.clear_selection_button.clicked.connect(self.clear_selection)
        self.invert_select_button.clicked.connect(self.invert_selection)
        self.plotted_select_button.clicked.connect(self.select_plotted)
        # Plot Button
        self.plot_selected_button.clicked.connect(self.plot_selected_function)

    def init_matrix_plot(self):
        # Create the matrix
        shape = self.matrix.shape
        max_freq_lines = 50
        freq_decimate_factor = int(np.ceil(shape[0] / max_freq_lines))
        indices = np.zeros(shape[0], dtype=bool)
        indices[freq_decimate_factor // 2::freq_decimate_factor] = True
        self.selector = MPLCanvas(self, width=2, height=2, dpi=100)
        self.selector.setFocusPolicy(Qt.ClickFocus)
        self.selector.setFocus()
        self.selector.mpl_connect('button_press_event', self.selector_click)
        self.selector.mpl_connect('button_release_event', self.selector_unclick)
        self.selector.axis.plot([shape[2], shape[1]], [0, shape[1]], 'k', linewidth=0.25)
        self.selector.axis.plot([0, shape[2]], [shape[1], shape[1]], 'k', linewidth=0.25)
        self.selector.axis.plot([0, shape[2]], [0, shape[1]], 'k', linewidth=0.25)
        self.select_rectangles = np.empty(shape[1:], dtype=object)
        self.plot_rectangles = np.empty(shape[1:], dtype=object)
        self.function_select_groupbox_layout.addWidget(self.selector)
        for i in range(shape[1]):
            self.selector.axis.plot([0, shape[2]], [i, i], 'k', linewidth=0.25)
            for j in range(shape[2]):
                if i == 0:
                    self.selector.axis.plot([j, j], [0, shape[1]], 'k', linewidth=0.25)
                data = np.log10(abs(self.matrix[:, i, j]))
                data -= np.mean(data)
                data /= -2.5 * np.max(data)
                abscissa = np.linspace(j, j + 1, shape[0])
                logical_selector = np.logical_and(np.logical_and(data > -.5, data < .5), indices)
                self.selector.axis.plot(abscissa[logical_selector],
                                        data[logical_selector] + i + .5, 'b', linewidth=.25)
                # Create the rectangle
                self.select_rectangles[i, j] = self.selector.axis.add_patch(
                    Rectangle((j, i), 1, 1, alpha=0.5, color='k', visible=False))
                self.plot_rectangles[i, j] = self.selector.axis.add_patch(
                    Rectangle((j, i), 1, 1, alpha=0.5, color='r', visible=False))

        self.selector.axis.set_position([0, 0, 1, 1])
        self.selector.axis.set_xlim(0, shape[2])
        self.selector.axis.set_ylim(0, shape[1])
        self.selector.axis.set_xticks([])
        self.selector.axis.set_yticks([])
        self.selector.axis.invert_yaxis()
        self.selected_function = np.zeros(shape[1:], dtype=bool)
        self.plotted_function = np.zeros(shape[1:], dtype=bool)
        self.plot_type_bits = np.zeros(shape[1:], dtype='int8')
#        # Get the current widget size and resize if necessary
#        self.selector.updateGeometry()
#        w = self.selector.width()
#        h = self.selector.height()
#        print((w,h))
#        self.selector.resize(shape[2]*10 if w < shape[2]*10 else w,
#                             shape[1]*10 if h < shape[1]*10 else h)
#        self.selector.updateGeometry()

    def selector_click(self, event):
        # print('{:} Click: button={:}, x={:}, y={:}, xdata={:}, ydata={:}'.format(
        #         'double' if event.dblclick else 'single',
        #         event.button, event.x,event.y,event.xdata,event.ydata))
        j = int(np.floor(event.xdata))
        i = int(np.floor(event.ydata))
        self.select_start = (i, j)

    def selector_unclick(self, event):
        modifiers = QApplication.keyboardModifiers()
        # print('{:} Release: button={:}, x={:}, y={:}, xdata={:}, ydata={:}'.format(
        #         'double' if event.dblclick else 'single',
        #         event.button, event.x,event.y,event.xdata,event.ydata))
        j = int(np.floor(event.xdata))
        i = int(np.floor(event.ydata))
        try:
            ii, ji = self.select_start
        except TypeError:
            self.select_start = None
            return
        # See if we need to switch them
        if i < ii:
            i, ii = ii, i
        if j < ji:
            j, ji = ji, j
        if modifiers == Qt.ShiftModifier:
            self.selected_function[ii:i + 1, ji:j + 1] = True
        elif modifiers == Qt.ControlModifier and ii == i and ji == j:
            if self.matrix_select_checkbox.isChecked() and self.selected_function[i, j]:
                self.selected_function[i, :] = False
                self.selected_function[:, j] = False
            else:
                self.selected_function[i, j] = not self.selected_function[i, j]
        else:
            self.selected_function[:] = False
            self.selected_function[ii:i + 1, ji:j + 1] = True
        if self.matrix_select_checkbox.isChecked():
            self.extend_selection_matrix()
        self.update_selection()

    def select_upper_triangular(self, event):
        self.matrix_select_checkbox.setChecked(False)
        for (i, j), val in np.ndenumerate(self.selected_function):
            if i < j:
                self.selected_function[i, j] = True
        self.update_selection()

    def select_lower_triangular(self, event):
        self.matrix_select_checkbox.setChecked(False)
        for (i, j), val in np.ndenumerate(self.selected_function):
            if i > j:
                self.selected_function[i, j] = True
        self.update_selection()

    def select_diagonal(self, event):
        self.matrix_select_checkbox.setChecked(False)
        for (i, j), val in np.ndenumerate(self.selected_function):
            if i == j:
                self.selected_function[i, j] = True
        self.update_selection()

    def deselect_upper_triangular(self, event):
        self.matrix_select_checkbox.setChecked(False)
        for (i, j), val in np.ndenumerate(self.selected_function):
            if i < j:
                self.selected_function[i, j] = False
        self.update_selection()

    def deselect_lower_triangular(self, event):
        self.matrix_select_checkbox.setChecked(False)
        for (i, j), val in np.ndenumerate(self.selected_function):
            if i > j:
                self.selected_function[i, j] = False
        self.update_selection()

    def deselect_diagonal(self, event):
        self.matrix_select_checkbox.setChecked(False)
        for (i, j), val in np.ndenumerate(self.selected_function):
            if i == j:
                self.selected_function[i, j] = False
        self.update_selection()

    def clear_selection(self, event):
        self.selected_function[:] = False
        self.update_selection()

    def invert_selection(self, event):
        self.matrix_select_checkbox.setChecked(False)
        for (i, j), val in np.ndenumerate(self.selected_function):
            self.selected_function[i, j] = not self.selected_function[i, j]
        self.update_selection()

    def select_plotted(self, event):
        self.matrix_select_checkbox.setChecked(False)
        self.selected_function[:] = self.plotted_function[:]
        self.update_selection()

    def magnitude_state(self, event):
        # print('Magnitude Checkbox Fired')
        # print(self.magnitude_checkbox.checkState())
        if self.magnitude_checkbox.checkState() > 0:
            self.plot_type_bits[self.selected_function] |= self.DataType.MAGNITUDE.value
            self.magnitude_checkbox.blockSignals(True)
            self.magnitude_checkbox.setCheckState(2)
            self.magnitude_checkbox.blockSignals(False)
        else:
            self.plot_type_bits[self.selected_function] &= ~self.DataType.MAGNITUDE.value

    def coherence_state(self, event):
        # print('Coherence Checkbox Fired')
        if self.coherence_checkbox.checkState() > 0:
            self.plot_type_bits[self.selected_function] |= self.DataType.COHERENCE.value
            self.coherence_checkbox.blockSignals(True)
            self.coherence_checkbox.setCheckState(2)
            self.coherence_checkbox.blockSignals(False)
        else:
            self.plot_type_bits[self.selected_function] &= ~self.DataType.COHERENCE.value

    def phase_state(self, event):
        # print('Phase Checkbox Fired')
        if self.phase_checkbox.checkState() > 0:
            self.plot_type_bits[self.selected_function] |= self.DataType.PHASE.value
            self.phase_checkbox.blockSignals(True)
            self.phase_checkbox.setCheckState(2)
            self.phase_checkbox.blockSignals(False)
        else:
            self.plot_type_bits[self.selected_function] &= ~self.DataType.PHASE.value

    def real_state(self, event):
        # print('Real Checkbox Fired')
        if self.real_checkbox.checkState() > 0:
            self.plot_type_bits[self.selected_function] |= self.DataType.REAL.value
            self.real_checkbox.blockSignals(True)
            self.real_checkbox.setCheckState(2)
            self.real_checkbox.blockSignals(False)
        else:
            self.plot_type_bits[self.selected_function] &= ~self.DataType.REAL.value

    def imaginary_state(self, event):
        # print('Imaginary Checkbox Fired')
        if self.imaginary_checkbox.checkState() > 0:
            self.plot_type_bits[self.selected_function] |= self.DataType.IMAGINARY.value
            self.imaginary_checkbox.blockSignals(True)
            self.imaginary_checkbox.setCheckState(2)
            self.imaginary_checkbox.blockSignals(False)
        else:
            self.plot_type_bits[self.selected_function] &= ~self.DataType.IMAGINARY.value

    def plot_selected_function(self, event):
        self.plotted_function[:] = False
        self.plotted_function[self.selected_function] = True
        for key, val in np.ndenumerate(self.plotted_function):
            #            print((key,val))
            if val:
                self.plot_rectangles[key].set_visible(True)
            else:
                self.plot_rectangles[key].set_visible(False)
        self.selector.draw()
        self.plot()

    def update_selection(self):
        # print('updating selection')
        # Update the function type check boxes
        out = self.find_function_types()
        for val, checkbox in zip(out, self.function_type_checkboxes):
            checkbox.blockSignals(True)
            checkbox.setCheckState(int(val))
            checkbox.blockSignals(False)
        for key, val in np.ndenumerate(self.selected_function):
            #            print((key,val))
            if val:
                self.select_rectangles[key].set_visible(True)
            else:
                self.select_rectangles[key].set_visible(False)
        # print('drawing')
        self.selector.draw()

    def extend_selection_matrix(self):
        rows = np.any(self.selected_function, axis=1)
        cols = np.any(self.selected_function, axis=0)
        self.selected_function = rows[:, np.newaxis] * cols[np.newaxis, :]

    def state_changed(self, event):
        if self.matrix_select_checkbox.isChecked():
            self.extend_selection_matrix()
            self.update_selection()
        self.selector.setFocus()

    def find_function_types(self):
        # print(self.plot_type_bits.shape)
        # print(self.selected_function.shape)
        mat = self.plot_type_bits[self.selected_function]
        # print(mat)
        any_functions = [1 if np.any(mat & bit) else 0 for bit in [self.DataType.MAGNITUDE.value, self.DataType.COHERENCE.value,
                                                                   self.DataType.PHASE.value, self.DataType.REAL.value, self.DataType.IMAGINARY.value]]
        all_functions = [2 if np.all(mat & bit) else 0 for bit in [self.DataType.MAGNITUDE.value, self.DataType.COHERENCE.value,
                                                                   self.DataType.PHASE.value, self.DataType.REAL.value, self.DataType.IMAGINARY.value]]
        return np.max([any_functions, all_functions], axis=0)

    def plot(self):
        # print('plotting')
        # Create the figure and put it in the dock
        functions = self.matrix[:, self.plotted_function]
        coherence_functions = self.coh_matrix[:, self.plotted_function]
        if not self.compare_matrix is None:
            compare_functions = self.compare_matrix[:, self.plotted_function]
            compare_coherence_functions = self.compare_coh_matrix[:, self.plotted_function]
            cm_increment = 2
        else:
            cm_increment = 1
        plot_bits = self.plot_type_bits[self.plotted_function]
        if self.matrix_mode_button.isChecked():
            rows = np.nonzero(np.any(self.plotted_function, axis=1))[0]
            cols = np.nonzero(np.any(self.plotted_function, axis=0))[0]
            indices_to_skip = []
            index = 1
            for row in rows:
                for col in cols:
                    # print((row,col))
                    # print(self.plotted_function.shape)
                    # print(self.plotted_function[row,col])
                    if not self.plotted_function[row, col]:
                        indices_to_skip.append(index)
                    index += 1
            nrows = len(rows)
            ncols = len(cols)
        else:
            square_array_size = np.ceil(np.sqrt(functions.shape[1]))
            nrows = int(square_array_size)
            ncols = int(np.ceil(functions.shape[1] / nrows))
            indices_to_skip = list(range(functions.shape[1] + 1, nrows * ncols + 1))
        # Create the canvas
        try:
            # Delete the canvas if it exists
            self.functions_groupbox_layout.removeWidget(self.plot_canvas)
            self.plot_canvas.deleteLater()
        except AttributeError:
            # print('No Widget to Remove')
            pass
        self.plot_canvas = MPLMultiCanvas(
            self, subplots=[nrows, ncols], ignore_subplots=indices_to_skip)
        self.functions_groupbox_layout.addWidget(self.plot_canvas)
        index = 0
        for key, axis in np.ndenumerate(self.plot_canvas.axes):
            if axis is None:
                continue
            function = functions[:, index]
            coh_function = coherence_functions[:, index]
            if not self.compare_matrix is None:
                compare_function = compare_functions[:, index]
                compare_coh_function = compare_coherence_functions[:, index]
            else:
                compare_function = np.nan
                compare_coh_function = np.nan
            bits = [val for val in [self.DataType.MAGNITUDE.value, self.DataType.COHERENCE.value, self.DataType.PHASE.value,
                                    self.DataType.REAL.value, self.DataType.IMAGINARY.value] if bool(plot_bits[index] & val)]
            plots = []
            for bit in bits:
                if bit == self.DataType.MAGNITUDE.value:
                    plots.append((abs(function), abs(compare_function), 0, 'log'))
                elif bit == self.DataType.COHERENCE.value:
                    plots.append((coh_function, compare_coh_function, 1, 'linear'))
                elif bit == self.DataType.PHASE.value:
                    plots.append((np.angle(function), np.angle(compare_function), 2, 'linear'))
                elif bit == self.DataType.REAL.value:
                    plots.append((np.real(function), np.real(compare_function), 3, 'linear'))
                elif bit == self.DataType.IMAGINARY.value:
                    plots.append((np.imag(function), np.imag(compare_function), 4, 'linear'))
            # Add a y axis for each plot
            this_ax = [axis] + [axis.twinx() for p in plots[1:]]
            for (fn, compare_fn, color_index, scale), ax in zip(plots, this_ax):
                color = self.cm(color_index * cm_increment)
                ax.plot(self.abscissa, fn, color=color)
                if not self.compare_matrix is None:
                    compare_color = self.cm(color_index * cm_increment + 1)
                    ax.plot(self.abscissa, compare_fn, color=compare_color)
                ax.set_yscale(scale)
                ax.tick_params(axis='y', colors=color)
            index += 1
#        self.plot_canvas.fig.tight_layout()



frf_from_time_data = TransferFunctionArray.from_time_data
