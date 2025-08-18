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

import itertools
import numpy as np
from .sdynpy_array import SdynpyArray
from .sdynpy_coordinate import outer_product, CoordinateArray, coordinate_array
from .sdynpy_matrix import Matrix, matrix
from ..signal_processing.sdynpy_correlation import mac
from ..signal_processing.sdynpy_frf import timedata2frf
from ..signal_processing.sdynpy_cpsd import (cpsd as sp_cpsd,
                                             cpsd_coherence as sp_coherence,
                                             cpsd_to_time_history,
                                             cpsd_from_coh_phs,
                                             db2scale)
from ..signal_processing.sdynpy_srs import (srs as sp_srs,
                                            octspace,
                                            sum_decayed_sines as sp_sds,
                                            sum_decayed_sines_reconstruction,
                                            sum_decayed_sines_displacement_velocity)
from ..signal_processing.sdynpy_rotation import lstsq_rigid_transform
from ..signal_processing.sdynpy_generator import (
    pseudorandom, sine, ramp_envelope, chirp, pulse, sine_sweep)
from ..signal_processing.sdynpy_frf_inverse import (frf_inverse,
                                                    compute_tikhonov_modified_singular_values)

from ..fem.sdynpy_exodus import Exodus
from scipy.linalg import eigh
from enum import Enum
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
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
import os
import scipy.signal as sig
import warnings
import scipy.fft as scipyfft
from scipy.signal.windows import exponential, get_window
from scipy.signal import oaconvolve, convolve
from scipy.interpolate import interp1d
pyqtgraph.setConfigOption('background', 'w')
pyqtgraph.setConfigOption('foreground', 'k')


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
    IMPULSE_RESPONSE_FUNCTION = 29


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
                           'Impulse Response Function': FunctionTypes.IMPULSE_RESPONSE_FUNCTION
                           }

_imat_function_type_inverse_map = {val: key for key, val in _imat_function_type_map.items()}


def _flat_frequency_shape(freq):
    return 1


class AbscissaIndexExtractor:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, key):
        return self.parent.extract_elements(key)

    def __call__(self, key):
        return self.parent.extract_elements(key)


class AbscissaValueExtractor:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, key):
        return self.parent.extract_elements_by_abscissa(key[0], key[1])

    def __call__(self, key):
        return self.parent.extract_elements_by_abscissa(key[0], key[1])


def _update_annotations_to_axes_bottom(axes):
    annotations = [child for child in axes.get_children() if isinstance(child, matplotlib.text.Annotation)]
    for annotation in annotations:
        new_position = (annotation.xy[0],axes.get_ylim()[0])
        annotation.xy = new_position

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
            ('abscissa', 'float64', (nelements,)),
            ('ordinate', ordinate_dtype, (nelements,)),
            ('comment1', '<U80'),
            ('comment2', '<U80'),
            ('comment3', '<U80'),
            ('comment4', '<U80'),
            ('comment5', '<U80'),
            ('coordinate', CoordinateArray.data_dtype,
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

    @property
    def idx_by_el(self):
        """
        AbscissaIndexExtractor that can be indexed to extract specific elements
        """
        return AbscissaIndexExtractor(self)

    @property
    def idx_by_ab(self):
        """
        AbscissaValueExtractor that can be indexed to extract an abscissa range
        """
        return AbscissaValueExtractor(self)

    @property
    def abscissa_spacing(self):
        """The spacing of the abscissa in the function.  Returns ValueError if
        abscissa are not evenly spaced."""
        # Look at the spacing between abscissa
        spacing = np.diff(self.abscissa, axis=-1)
        mean_spacing = np.mean(spacing)
        if not np.allclose(spacing, mean_spacing):
            raise ValueError('{:} do not have evenly spaced abscissa'.format(self.__class__.__name__))
        return mean_spacing

    def plot(self, one_axis: bool = True, subplots_kwargs: dict = {},
             plot_kwargs: dict = {}, abscissa_markers = None, 
             abscissa_marker_labels = None, abscissa_marker_type = 'vline',
             abscissa_marker_plot_kwargs = {}):
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
        abscissa_markers : ndarray, optional
            Array containing abscissa values to mark on the plot to denote
            significant events.
        abscissa_marker_labels : str or ndarray
            Array of strings to label the abscissa_markers with, or
            alternatively a format string that accepts index and abscissa
            inputs (e.g. '{index:}: {abscissa:0.2f}').  By default no label
            will be applied.
        abscissa_marker_type : str
            The type of marker to use.  This can either be the string 'vline'
            or a valid matplotlib symbol specifier (e.g. 'o', 'x', '.').
        abscissa_marker_plot_kwargs : dict
            Additional keyword arguments used when plotting the abscissa label
            markers.

        Returns
        -------
        axis : matplotlib axis or array of axes
             On which the data were plotted

        """
        if abscissa_markers is not None:
            if abscissa_marker_labels is None:
                abscissa_marker_labels = ['' for value in abscissa_markers]
            elif isinstance(abscissa_marker_labels,str):
                abscissa_marker_labels = [abscissa_marker_labels.format(
                    index = i, abscissa = v) for i,v in enumerate(abscissa_markers)]
        
        if one_axis is True:
            figure, axis = plt.subplots(**subplots_kwargs)
            lines = axis.plot(self.flatten().abscissa.T, self.flatten().ordinate.T.real, **plot_kwargs)
            if abscissa_markers is not None:
                if abscissa_marker_type == 'vline':
                    kwargs = {'color':'k'}
                    kwargs.update(abscissa_marker_plot_kwargs)
                    for value,label in zip(abscissa_markers,abscissa_marker_labels):
                        axis.axvline(value, **kwargs)
                        axis.annotate(label, xy = (value, axis.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                    axis.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                else:
                    for line in lines:
                        x = line.get_xdata()
                        y = line.get_ydata()
                        marker_y = np.interp(abscissa_markers, x, y)
                        kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        axis.plot(abscissa_markers,marker_y,**kwargs)
                        for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                            axis.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
        elif one_axis is False:
            ncols = int(np.floor(np.sqrt(self.size)))
            nrows = int(np.ceil(self.size / ncols))
            figure, axis = plt.subplots(nrows, ncols, **subplots_kwargs)
            for i, (ax, (index, function)) in enumerate(zip(axis.flatten(), self.ndenumerate())):
                lines = ax.plot(function.abscissa.T, function.ordinate.T.real, **plot_kwargs)
                ax.set_ylabel('/'.join([str(v) for i, v in function.coordinate.ndenumerate()]))
                if abscissa_markers is not None:
                    if abscissa_marker_type == 'vline':
                        kwargs = {'color':'k'}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        for value,label in zip(abscissa_markers,abscissa_marker_labels):
                            ax.axvline(value, **kwargs)
                            ax.annotate(label, xy = (value, ax.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                        ax.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                    else:
                        for line in lines:
                            x = line.get_xdata()
                            y = line.get_ydata()
                            marker_y = np.interp(abscissa_markers, x, y)
                            kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                            kwargs.update(abscissa_marker_plot_kwargs)
                            ax.plot(abscissa_markers,marker_y,**kwargs)
                            for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                                ax.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
            for ax in axis.flatten()[i + 1:]:
                ax.remove()
        else:
            axis = one_axis
            lines = axis.plot(self.abscissa.T, self.ordinate.T.real, **plot_kwargs)
            if abscissa_markers is not None:
                if abscissa_marker_type == 'vline':
                    kwargs = {'color':'k'}
                    kwargs.update(abscissa_marker_plot_kwargs)
                    for value,label in zip(abscissa_markers,abscissa_marker_labels):
                        axis.axvline(value, **kwargs)
                        axis.annotate(label, xy = (value, axis.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                    axis.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                else:
                    for line in lines:
                        x = line.get_xdata()
                        y = line.get_ydata()
                        marker_y = np.interp(abscissa_markers, x, y)
                        kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        axis.plot(abscissa_markers,marker_y,**kwargs)
                        for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                            axis.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
        return axis

    def gui_plot(self,abscissa_markers = None,abscissa_marker_labels = None,
                 abscissa_marker_type = None, legend_label = None):
        """
        Create a GUIPlot window to visualize data.

        Parameters
        ----------
        abscissa_markers : np.ndarray
            Abscissa values at which markers will be placed.  If not specified,
            no markers will be added.  Markers will be added to all plotted
            curves if this argument is passed.
        abscissa_marker_labels : str or iterable
            Labels that will be applied to the markers.  If not specified, no
            label will be applied.  If a single string is passed, it will be
            passed to the `.format` method with keyword arguments `index` and
            `abscissa`.  Otherwise there should be one string for each marker.
        abscissa_marker_type : str:
            The type of marker that will be applied.  Can be 'vline' for a
            vertical line across the axis, or it can be a pyqtgraph symbol specifier
            (e.g. 'x', 'o', 'star', etc.) which will be placed on the plotted curves.
            If not specified, a vertical line will be used.

        Returns
        -------
        GUIPlot

        """
        args = []
        kwargs = {}
        if legend_label is not None:
            kwargs[legend_label] = self
        else:
            args.append(self)
        if abscissa_markers is not None:
            kwargs['abscissa_markers'] = abscissa_markers
        if abscissa_marker_labels is not None:
            kwargs['abscissa_marker_labels'] = abscissa_marker_labels
        if abscissa_marker_type is not None:
            kwargs['abscissa_marker_type'] = abscissa_marker_type
        return GUIPlot(*args,**kwargs)
        

    def plot_image(self,ax = None, reduction_function = None, colorbar_scale = 'linear',
                   colorbar_min = None, colorbar_max = None):
        image_data = self.flatten().ordinate
        if colorbar_scale == 'log':
            image_data = np.log10(image_data)
            if colorbar_min is not None:
                colorbar_min = np.log10(colorbar_min)
            if colorbar_max is not None:
                colorbar_max = np.log10(colorbar_max)
        def dof_formatter(x, pos):
            if x < 0:
                return ''
            elif x >= self.size:
                return ''
            else:
                return '/'.join([str(v) for v in self.flatten()[int(x)].coordinate])
        if ax is None:
            fig, ax = plt.subplots(1,1, layout='constrained')
        im_data = ax.imshow(image_data,vmin=colorbar_min,vmax=colorbar_max,aspect='auto',
                            interpolation='none',extent=[
                                self.flatten()[0].abscissa[0]-0.5*self.abscissa_spacing,
                                self.flatten()[0].abscissa[-1]+0.5*self.abscissa_spacing,
                                self.size-0.5,-0.5])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(min_n_ticks=1,integer=True))
        ax.yaxis.set_major_formatter(dof_formatter)
        if colorbar_scale == 'log':
            plt.colorbar(
                im_data,ax=ax,
                format=lambda x,pos: '$10^{:}$'.format(x))
        else:
            plt.colorbar(
                im_data,ax=ax)
        return ax

    def reshape_to_matrix(self, error_if_missing = True):
        """
        Reshapes a data array to a matrix with response coordinates along the
        rows and reference coordinates along the columns

        Parameters
        ----------
        error_if_missing : bool
            If True, an error will be thrown if there are missing data objects
            when trying to make a matrix of functions (i.e. if a response
            degree of freedom is missing from one reference).  If False,
            response coordinates will simply be discarded if they do not exist
            for all references.  Default is True.

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
        if not error_if_missing:
            keep_response_indices = np.ones(response_coords.shape,dtype=bool)
        for row_index, response_coord in response_coords.ndenumerate():
            for col_index, reference_coord in reference_coords.ndenumerate():
                current_function = flattened_functions[
                    (flattened_functions.response_coordinate == response_coord)
                    &
                    (flattened_functions.reference_coordinate == reference_coord)]
                if current_function.size == 0:
                    if error_if_missing:
                        raise ValueError('No function exists with reference coordinate {:} and response coordinate {:}'.format(
                            str(reference_coord), str(response_coord)))
                    else:
                        keep_response_indices[row_index] = False
                        continue
                if current_function.size > 1:
                    raise ValueError('Multiple functions exist ({:}) with reference coordinate {:} and response coordinate {:}'.format(
                        current_function.size, str(reference_coord), str(response_coord)))
                output_array[row_index[0], col_index[0]] = current_function
        if not error_if_missing:
            output_array = output_array[keep_response_indices,:]
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
        """
        Extracts elements with abscissa values within the specified range

        Parameters
        ----------
        min_abscissa : float
            Minimum abscissa value to keep
        max_abscissa : float
            Maximum abscissa value to keep.

        Returns
        -------
        NDDataArray
            Array reduced to specified elements.

        """
        abscissa_indices = (self.abscissa >= min_abscissa) & (self.abscissa <= max_abscissa)
        indices = np.all(abscissa_indices, axis=tuple(np.arange(abscissa_indices.ndim - 1)))
        new_ordinate = self.ordinate[..., indices]
        new_abscissa = self.abscissa[..., indices]
        return data_array(self.function_type, new_abscissa, new_ordinate, self.coordinate,
                          self.comment1, self.comment2, self.comment3, self.comment4, self.comment5)

    @classmethod
    def join(cls, data_arrays, increment_abscissa=True):
        """
        Joins several data arrays together by concatenating their ordinates

        Parameters
        ----------
        data_arrays : NDDataArray
            Arrays to concatenate
        increment_abscissa : bool, optional
            Determines how the abscissa concatenation is handled.  If False,
            the abscissa is left as it was in the original functions.  If True,
            it will be incremented so it is continuous.

        Returns
        -------
        NDDataArray subclass
        """
        func_type = data_arrays[0].function_type
        # Verify that coordinates are consistent
        all_coordinate = np.array([array.coordinate for array in data_arrays]).view(CoordinateArray)
        if not np.all(all_coordinate[:1] == all_coordinate):
            raise ValueError('Signals do not have equivalent coordinates')
        coordinate = data_arrays[0].coordinate
        ordinate = np.concatenate([array.ordinate for array in data_arrays], axis=-1)
        if increment_abscissa:
            delta_abscissa = data_arrays[0].abscissa_spacing
            abscissa = np.arange(ordinate.shape[-1])*delta_abscissa
        else:
            abscissa = np.concatenate([array[coordinate].abscissa for array in data_arrays], axis=-1)
        return data_array(func_type, abscissa, ordinate, coordinate)

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
            if node_id_map is not None:
                original_geometry = original_geometry.reduce(node_id_map.from_ids)
                original_geometry.node.id = node_id_map(original_geometry.node.id)
                self = self.copy()[np.in1d(self.coordinate.node, node_id_map.from_ids)]
                self.coordinate.node = node_id_map(self.coordinate.node)
            common_nodes = np.intersect1d(np.intersect1d(original_geometry.node.id, new_geometry.node.id),
                                          np.unique(self.coordinate.node))
            coordinates = coordinate_array(
                common_nodes[:, np.newaxis], [1, 2, 3, 4, 5, 6] if rotations else [1, 2, 3])
            transform_from_original = original_geometry.global_deflection(coordinates)
            transform_to_new = new_geometry.global_deflection(coordinates)
            new_data_array = self[coordinates[..., np.newaxis]].copy()
            shape_matrix = new_data_array.ordinate
            new_shape_matrix = np.einsum('nij,nkj,nkl->nil', transform_to_new,
                                         transform_from_original, shape_matrix)
            new_data_array.ordinate = new_shape_matrix
            return new_data_array.flatten()
        else:
            raise NotImplementedError('2D Data not Implemented Yet')

    def to_shape_array(self, abscissa_values=None):
        """
        Converts an NDDataArray to a ShapeArray

        Parameters
        ----------
        abscissa_values : ndarray, optional
            Abscissa values at which the shapes will be created. The default is
            to create shapes at all abscissa values.  If an entry in
            abscissa_values does not match a value in abscissa, the closest
            abscissa value will be selected

        Raises
        ------
        ValueError
            If the data does not have common abscissa across all functions or if
            duplicate response coordinates occur in the NDDataArray

        Returns
        -------
        ShapeArray
            ShapeArray containing the NDDataArray's ordinate as its shape_matrix

        """
        flat_self = self.flatten()
        if not self.validate_common_abscissa():
            raise ValueError('Data must have common abscissa to transform to `ShapeArray`')
        if abscissa_values is None:
            abscissa_indices = slice(None)
        else:
            abscissa_indices = np.argmin(abs(flat_self[0].abscissa - np.atleast_1d(abscissa_values)[:, np.newaxis]), axis=-1)
        # Check if there are repeated responses
        coordinates = flat_self.response_coordinate
        if coordinates.size != np.unique(coordinates).size:
            raise ValueError('Data has duplicate response coordinates.  Please ensure that there is only one of each response coordinate in the data.')
        # Extract the shape matrix
        shape_matrix = flat_self.ordinate[:, abscissa_indices].T
        # Create the new shape
        from .sdynpy_shape import shape_array
        return shape_array(coordinates, shape_matrix, flat_self[0].abscissa[abscissa_indices])

    def zero_pad(self, num_samples=0, update_abscissa=True,
                 left=False, right=True,
                 use_next_fast_len=False):
        """
        Add zeros to the beginning or end of a signal

        Parameters
        ----------
        num_samples : int, optional
            Number of zeros to add to the function.  If not specified, no zeros
            are added unless `use_next_fast_len` is `True`
        update_abscissa : bool, optional
            If True, modify the abscissa to keep the same abscissa spacing.
            The function must have equally spaced abscissa for this to work.
            If False, the added abscissa will have a value of zero.
            The default is True.
        left : bool, optional
            Add zeros to the left side (beginning) of the function. The default
            is False.  If both `left` and `right` are specified, the zeros will
            be split half on the left and half on the right.
        right : bool, optional
            Add zeros to the right side (end) of the function. The default is
            True.  If both `left` and `right` are specified, the zeros will be
            split half on the left and half on the right
        use_next_fast_len : bool, optional
            If True, potentially add additional zeros to the value specified by
            `num_samples` to allow the total length of the final signal to reach
            fast values for FFT as specified by `scipy.fft.next_fast_len`.

        Returns
        -------
        NDDataArray subclass
            The zero-padded version of the function

        """
        if use_next_fast_len:
            total_samples = scipyfft.next_fast_len(self.num_elements + num_samples)
            num_samples = total_samples - self.num_elements
        # Create the additional zeros vectors
        if left and (not right):
            left_samples = num_samples
            right_samples = 0
        elif (not left) and right:
            right_samples = num_samples
            left_samples = 0
        elif left and right:
            left_samples = num_samples//2
            right_samples = num_samples - left_samples
        else:
            left_samples = 0
            right_samples = 0
        added_zeros_left = np.zeros(self.shape+(left_samples,))
        added_zeros_right = np.zeros(self.shape+(right_samples,))

        new_ordinate = np.concatenate((added_zeros_left, self.ordinate, added_zeros_right), axis=-1)

        if update_abscissa:
            new_abscissa = self.abscissa_spacing*(np.arange(new_ordinate.shape[-1])-added_zeros_left.shape[-1]) + self.abscissa[..., 0, np.newaxis]
        else:
            new_abscissa = np.concatenate((added_zeros_left, self.abscissa, added_zeros_right), axis=-1)

        return data_array(self.function_type, new_abscissa, new_ordinate,
                          self.coordinate, self.comment1, self.comment2,
                          self.comment3, self.comment4, self.comment5)

    def interpolate(self, interpolated_abscissa, kind='linear', **kwargs):
        """
        Interpolates the NDDataArray using SciPy's interp1d.

        Parameters
        ----------
        interpolated_abscissa : ndarray
            Abscissa values at which to interpolate the function.  If
            multi-dimensional, it will be flattened.
        kind : str or int, optional
            Specifies the kind of interpolation as a string or as an integer
            specifying the order of the spline interpolator to use. The string
            has to be one of 'linear', 'nearest', 'nearest-up', 'zero',
            'slinear', 'quadratic', 'cubic', 'previous', or 'next'.
            'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of zeroth, first, second or third order; 'previous'
            and 'next' simply return the previous or next value of the point;
            'nearest-up' and 'nearest' differ when interpolating half-integers
            (e.g. 0.5, 1.5) in that 'nearest-up' rounds up and 'nearest' rounds
            down. 'logx', 'logy', and 'loglog' use linear interpolation on 
            the values converted to log scale.  Default is 'linear'.
        **kwargs :
            Additional arguments to scipy.interpolate.interp1d.

        Returns
        -------
        NDDataArray :
            Array with interpolated arguments
        """
        # Flatten the abscissa
        interpolated_abscissa = np.reshape(interpolated_abscissa, -1)
        # Create the output class
        output = self.__class__(self.shape, interpolated_abscissa.size)
        output.coordinate = self.coordinate
        output.comment1 = self.comment1
        output.comment2 = self.comment2
        output.comment3 = self.comment3
        output.comment4 = self.comment4
        output.comment5 = self.comment5
        output.abscissa = interpolated_abscissa
        if kind == 'logx':
            logx = True
            logy = False
            kind = 'linear'
        elif kind == 'logy':
            logx = False
            logy = True
            kind = 'linear'
        elif kind == 'loglog':
            logx=True
            logy=True
            kind = 'linear'
        else:
            logx = False
            logy = False
        if self.validate_common_abscissa():
            x = np.log(self.flatten()[0].abscissa) if logx else self.flatten()[0].abscissa
            y = np.log(self.ordinate) if logy else self.ordinate
            interp = interp1d(x, y, kind=kind, axis=-1, **kwargs)
            interpolated_ordinate = interp(np.log(interpolated_abscissa) if logx else interpolated_abscissa)
            output.ordinate = np.exp(interpolated_ordinate) if logy else interpolated_ordinate
        else:
            for key, function in self.ndenumerate():
                x = np.log(function.abscissa) if logx else function.abscissa
                y = np.log(function.ordinate) if logy else function.ordinate
                interp = interp1d(x, y, kind=kind, axis=-1, **kwargs)
                interpolated_ordinate = interp(np.log(interpolated_abscissa) if logx else interpolated_abscissa)
                output[key].ordinate = np.exp(interpolated_ordinate) if logy else interpolated_ordinate
        return output

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
        if isinstance(key, CoordinateArray):
            coordinate_dim = self.dtype['coordinate'].ndim
            output_shape = key.shape[:-coordinate_dim]
            flat_self = self.flatten()
            index_array = np.empty(output_shape, dtype=int)
            positive_coordinates = abs(flat_self.coordinate)
            for index in np.ndindex(output_shape):
                positive_key = abs(key[index])
                try:
                    index_array[index] = np.where(
                        np.all(positive_coordinates == positive_key, axis=-1))[0][0]
                except IndexError:
                    raise ValueError('Coordinate {:} not found in data array'.format(str(key[index])))
            return_shape = flat_self[index_array].copy()
            if self.function_type in [FunctionTypes.COHERENCE, FunctionTypes.MULTIPLE_COHERENCE]:
                ordinate_multiplication_array = np.array(1)
            else:
                ordinate_multiplication_array = np.prod(
                    np.sign(return_shape.coordinate.direction) * np.sign(key.direction), axis=-1)
            # Set up for broadcasting
            ordinate_multiplication_array = ordinate_multiplication_array[..., np.newaxis]
            # Remove zeros and replace with 1s because we don't flip signs if
            # there is no direction associated with the coordinate
            ordinate_multiplication_array[ordinate_multiplication_array == 0] = 1
            return_shape.coordinate = key
            return_shape.ordinate *= ordinate_multiplication_array
            return return_shape
        else:
            output = super().__getitem__(key)
            if isinstance(key, str) and key == 'coordinate':
                return output.view(CoordinateArray)
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
            return np.min(self.ordinate, *min_args, **min_kwargs)
        else:
            return np.min(reduction(self.ordinate), *min_args, **min_kwargs)

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

    def save(self, filename):
        """
        Save the array to a numpy file

        Parameters
        ----------
        filename : str
            Filename that the array will be saved to.  Will be appended with
            .npz if not specified in the filename

        """
        np.savez(filename, data=self.view(np.ndarray),
                 function_type=self.function_type.value)

    @classmethod
    def load(cls, filename):
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
            if data is not None:
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
        try:
            fn_datasets = unv_data_dict[58]
        except KeyError:
            return NDDataArray((0,),nelements=1,data_dimension=1)
        fn_types = [dataset.function_type for dataset in fn_datasets]
        function_type_dict = {}
        for fn_dataset, fn_type in zip(fn_datasets, fn_types):
            fn_type_enum = FunctionTypes(fn_type)
            if fn_type_enum not in function_type_dict:
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
                coordinate.append((coordinate_array(function.response_node, function.response_direction),
                                   coordinate_array(function.reference_node, function.reference_direction)))
                comment1.append(function.idline1)
                comment2.append(function.idline2)
                comment3.append(function.idline3)
                comment4.append(function.idline4)
                comment5.append(function.idline5)
            return_functions.append(
                data_array(key, np.array(abscissa), np.array(ordinate),
                           np.array(coordinate).view(CoordinateArray),
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
    
    def get_reciprocal_data(self,return_indices = False):
        """
        Gets reciprocal pairs of data from an NDDataArray.

        Parameters
        ----------
        return_indices : bool, optional
            If True, it will return a set of indices into the original array
            that extract the reciprocal functions. If False, then the
            reciprocal functions are returned directly.  The default is False.

        Raises
        ------
        ValueError
            If the data does not have reference and response coordinates, the
            method will raise a ValueError.

        Returns
        -------
        np.ndarray or NDDataArray subclass
            If return_indices is True, this will return the indices into the
            original array that extract the reciprocal data.  If return_indices
            is False, this will return the reciprocal NDDataArrays directly.

        """
        # Check if there are references and responses
        if self.dtype['coordinate'].shape != (2,):
            raise ValueError('Cannot compute reciprocal data of functions with only one coordinate')
        # Get all of the degrees of freedom that are both in references and
        # responses
        references = np.unique(abs(self.reference_coordinate))
        responses = np.unique(abs(self.response_coordinate))
        # Get common references and responses
        common_dofs = np.intersect1d(references,responses)
        # Get pairs of those degrees of freedom
        dof_combos = np.array([combo for combo in itertools.combinations(common_dofs,2)])
        # Now we need to select the reciprocal degrees of freedom
        reciprocal_dofs = np.array((dof_combos,dof_combos[...,::-1])).view(CoordinateArray)
        reciprocal_slice = tuple([Ellipsis]+self.ndim*[np.newaxis]+[slice(None)])
        equal_logical = np.all(abs(self.coordinate) == reciprocal_dofs[reciprocal_slice],axis=-1)
        equal_indices = np.where(equal_logical)
        equal_indices =  tuple([inds.reshape(2,-1) for inds in equal_indices[2:]])
        if return_indices:
            return equal_indices
        else:
            return self[equal_indices]
        
    def get_drive_points(self,return_indices=False):
        """
        Returns data arrays where the reference is equal to the response

        Parameters
        ----------
        return_indices : bool, optional
            If True, it will return a set of indices into the original array
            that extract the drive point functions. If False, then the
            drive point functions are returned directly.  The default is False.

        Raises
        ------
        ValueError
            If the data does not have reference and response coordinates, the
            method will raise a ValueError.

        Returns
        -------
        np.ndarray or NDDataArray subclass
            If return_indices is True, this will return the indices into the
            original array that extract the drive point data.  If return_indices
            is False, this will return the drive point NDDataArrays directly.

        """
        # Check if there are references and responses
        if self.dtype['coordinate'].shape != (2,):
            raise ValueError('Cannot compute drive point data of functions with only one coordinate')
        equal_logical = abs(self.response_coordinate) == abs(self.reference_coordinate)
        if return_indices:
            equal_indices = np.where(equal_logical)
            return equal_indices
        else:
            return self[equal_logical]
        
    def shape_filter(self,shape, filter_responses = True, filter_references = False,
                     rcond=None):
        """
        Spatially filters the data using the specified ShapeArray.

        Parameters
        ----------
        shape : ShapeArray
            A set of shapes used to filter the data
        filter_responses : bool, optional
            If True, will filter the response degrees of freedom. The default
            is True.
        filter_references : bool, optional
            If True, will filter the reference degrees of freedom.  The default
            is False.
        rcond : float, optional
            Condition number threshold used in the pseudoinverse to compute the
            inverse of the specified shape arrays. The default is None.

        Raises
        ------
        ValueError
            Raised if the abscissa are not consistent across all data.

        Returns
        -------
        filtered_data : NDDataArray or subclass
            The type of the output will be the same type as the input, but
            filtered such that the degrees of freedom correspond to the shapes
            in the provided ShapeArray

        """
        if not self.validate_common_abscissa():
            raise ValueError('Abscissa must be consistent between data to filter.')
        # Reshape data into matrix form if multidimensional
        if self.coordinate.shape[-1] > 1:
            response_coords = np.unique(self.response_coordinate)
            reference_coords = np.unique(self.reference_coordinate)
            coords = outer_product(response_coords,reference_coords)
            data = self[coords]
        else:
            response_coords = np.unique(self.response_coordinate)
            reference_coords = None
            coords = response_coords[:,np.newaxis]
            data = self[coords]
        data_type=data.function_type
        
        if filter_responses:
            response_shape_matrix = np.linalg.pinv(shape[response_coords].T,rcond=rcond)
            output_response_coords = coordinate_array(np.arange(response_shape_matrix.shape[0])+1,0)
        else:
            response_shape_matrix = None
            output_response_coords = response_coords
        if reference_coords is not None and filter_references:
            reference_shape_matrix = np.linalg.pinv(shape[reference_coords].T,rcond=rcond)
            output_reference_coords = coordinate_array(np.arange(reference_shape_matrix.shape[0])+1,0)
        else:
            reference_shape_matrix = None
            output_reference_coords = reference_coords
            
        if output_reference_coords is not None:
            output_coords = outer_product(output_response_coords,output_reference_coords)
        else:
            output_coords = output_response_coords[:,np.newaxis]
            
        ordinate = data.ordinate
        if response_shape_matrix is not None:
            ordinate = np.einsum('mi,i...s->m...s',response_shape_matrix,ordinate)
        if reference_shape_matrix is not None:
            ordinate = np.einsum('mj,...js->...ms',reference_shape_matrix,ordinate)
        
        filtered_data = data_array(data_type=data_type,
                                   abscissa=data.reshape(-1)[0].abscissa,
                                   ordinate=ordinate,
                                   coordinate=output_coords)

        return filtered_data

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
                                                           z_disp, x_rot, y_rot, z_rot]) if v is not None]
            abscissa = exo.get_times()
            data = [data_array(FunctionTypes.TIME_RESPONSE, abscissa,
                               exo.get_node_variable_values(variable, timesteps).T,
                               coordinate_array(node_ids, index)[:, np.newaxis]) for index, variable in variables]
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
            coordinates = coordinate_array(
                node_ids, np.array((1, 2, 3))[:, np.newaxis]).flatten()
            return data_array(FunctionTypes.TIME_RESPONSE, abscissa, ordinate, coordinates[:, np.newaxis])

    def fft(self, samples_per_frame=None, norm="backward", rtol=1, atol=1e-8,
            **scipy_rfft_kwargs):
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
        norm : str, optional
            The type of normalization applied to the fft computation.
            The default is "backward".
        rtol : float, optional
            Relative tolerance used in the abcsissa spacing check.
            The default is 1e-5.
        atol : float, optional
            Relative tolerance used in the abscissa spacing check.
            The default is 1e-8.
        scipy_rfft_kwargs :
            Additional keywords that will be passed to SciPy's rfft function.

        Raises
        ------
        ValueError
            Raised if the time signal passed to this function does not have
            equally spaced abscissa.

        Returns
        -------
        SpectrumArray
            The frequency spectra of the TimeHistoryArray.

        """
        diffs = np.diff(self.abscissa, axis=-1).flatten()
        if not np.allclose(diffs, diffs[0], rtol, atol):
            raise ValueError('Abscissa must have identical spacing to perform the FFT')
        ordinate = self.ordinate
        if samples_per_frame is not None:
            frame_indices = np.arange(samples_per_frame) + np.arange(ordinate.size //
                                                                     samples_per_frame)[:, np.newaxis] * samples_per_frame
            ordinate = ordinate[..., frame_indices]
        dt = np.mean(diffs)
        n = ordinate.shape[-1]
        # frequencies = scipyfft.rfftfreq(n, dt)
        frequencies = scipyfft.rfftfreq(n, dt)
        # ordinate = scipyfft.rfft(ordinate, axis=-1)
        ordinate = scipyfft.rfft(ordinate, axis=-1, norm=norm,
                                 **scipy_rfft_kwargs)
        if samples_per_frame is not None:
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

    def srs(self, min_frequency=None, max_frequency=None, frequencies=None,
            damping=0.03, num_points=None, points_per_octave=12,
            srs_type='MMAA'):
        """
        Compute a shock response spectrum (SRS) from the time history

        Parameters
        ----------
        min_frequency : float, optional
            Minimum frequency to compute the SRS. Either `frequencies` or
            `min_frequency` and `max_frequency` must be specified.
        max_frequency : float, optional
            Maximum frequency to compute the SRS. Either `frequencies` or
            `min_frequency` and `max_frequency` must be specified.
        frequencies : np.ndarray, optional
            Frequency lines at which to compute the SRS. Either `frequencies` or
            `min_frequency` and `max_frequency` must be specified.
        damping : float, optional
            Fraction of critical damping to use in the SRS calculation (e.g. you
            should specify 0.03 to represent 3%, not 3). The default is 0.03.
        num_points : int, optional
            Number of frequency lines to compute from `min_frequency` to
            `max_frequency`, log spaced between these two values.  If
            `min_frequency` and `max_frequency` are specified, then either
            `num_points` or `points_per_octave` must be specified.  If
            `frequencies` is specified, this argument is ignored.
        points_per_octave : float, optional
            Number of frequency lines per octave to compute from `min_frequency`
            to `max_frequency`.  If `min_frequency` and `max_frequency` are
            specified, then either `num_points` or `points_per_octave` must be
            specified.  If `frequencies` is specified, this argument is ignored.
            The default is 12.
        srs_type : str, optional
            A string encoding for the type of SRS to be computed.  See notes for
            more information.

        Returns
        -------
        ShockResponseSpectrumArray
            SRSs representing the current time histories.  If `srs_type` is
            `'all'`, then an extra dimension of 9 will be added to the front of
            the array, and the indices in that dimension will be different SRS
            types.

        Notes
        -----
        The `srs_type` argument takes a 4 character string that specifies how
        the SRS is computed.
        """
        # Compute default parameters
        try:
            srs_type_val = ShockResponseSpectrumArray._srs_type_map[srs_type.lower()]
        except KeyError:
            raise ValueError('Invalid `srs_type` specified, should be one of {:} (case insensitive)'.format(
                [k for k in ShockResponseSpectrumArray._srs_type_map]))

        if frequencies is None:
            if min_frequency is None or max_frequency is None:
                raise ValueError('`min_frequency` and `max_frequency` must be provided if `frequencies` is not')
            if num_points is None:
                frequencies = octspace(min_frequency, max_frequency, points_per_octave)
            else:
                frequencies = np.logspace(np.log10(min_frequency), np.log10(max_frequency), num_points)

        srss, f = sp_srs(self.ordinate, self.abscissa_spacing, frequencies, damping, srs_type_val)
        if abs(srs_type_val) == 10:
            np.moveaxis(srss, -2, 0)

        # Now construct the output object
        srs = data_array(FunctionTypes.SHOCK_RESPONSE_SPECTRUM, frequencies.copy(),
                         srss, self.coordinate.copy(),
                         self.comment1.copy(), self.comment2.copy(), self.comment3.copy(), self.comment4.copy(),
                         self.comment5.copy())
        return srs

    def filter(self,filter_order, frequency, filter_type = 'low',
               filter_method = 'filtfilt', filter_kwargs = None):
        """
        Filter the signal using a butterworth filter of the specified order

        Parameters
        ----------
        filter_order : int
            Order of the butterworth filter
        frequency : array_like
            The critical frequency or frequencies.  For lowpass and highpass
            filters, this is a scalar; for bandpass and bandstop filters, this
            is a length-2 sequence (low,high).
        filter_type : str, optional
            Type of filter.  Must be one of 'low','high','bandpass', or
            'bandstop'.  The default is 'low'.
        filter_method : str, optional
            Method of filter application. Must be one of 'filtfilt' or 'lfilter'.
            The default is 'filtfilt'.
        filter_kwargs: dict, optional
            Optional keyword arguments to be passed to filtfilt or lfilter.

        Returns
        -------
        TimeHistoryArray
            The filtered time history array.

        """
        fs = 1/self.abscissa_spacing
        b,a = sig.butter(filter_order,frequency,filter_type,fs=fs)
        if filter_kwargs is None:
            filter_kwargs = {}
        if filter_method == 'filtfilt':
            filtered_ordinate = sig.filtfilt(b,a,self.ordinate,axis=-1,
                                             **filter_kwargs)
        elif filter_method == 'lfilter':
            filtered_ordinate = sig.lfilter(b,a,self.ordinate,axis=-1,
                                            **filter_kwargs)
        else:
            raise ValueError('filter_type must be one of filtfilt or lfilter')
        return_val = self.copy()
        return_val.ordinate = filtered_ordinate
        return return_val

    def split_into_frames(self, samples_per_frame=None, frame_length=None,
                          overlap=None, overlap_samples=None, window=None,
                          check_cola=False, allow_fractional_frames=False):
        """
        Splits a time history into measurement frames with a given overlap and
        window function applied.

        Parameters
        ----------
        samples_per_frame : int, optional
            Number of samples in each measurement frame. Either this argument
            or `frame_length` must be specified.  If both or neither are
            specified, a `ValueError` is raised.
        frame_length : float, optional
            Length of each measurement frame in the same units as the `abscissa`
            field (`samples_per_frame` = `frame_length`/`self.abscissa_spacing`).
            Either this argument or `samples_per_frame` must be specified.  If
            both or neither are specified, a `ValueError` is raised.
        overlap : float, optional
            Fraction of the measurement frame to overlap (i.e. 0.25 not 25 to
            overlap a quarter of the frame). Either this argument or
            `overlap_samples` must be specified.  If both are
            specified, a `ValueError` is raised.  If neither are specified, no
            overlap is used.
        overlap_samples : int, optional
            Number of samples in the measurement frame to overlap. Either this
            argument or `overlap_samples` must be specified.  If both
            are specified, a `ValueError` is raised.  If neither are specified,
            no overlap is used.
        window : str or tuple or array_like, optional
            Desired window to use. If window is a string or tuple, it is passed
            to `scipy.signal.get_window` to generate the window values, which
            are DFT-even by default. See `get_window` for a list of windows and
            required parameters. If window is array_like it will be used
            directly as the window and its length must be `samples_per_frame`.
            If not specified, no window will be applied.
        check_cola : bool, optional
            If `True`, raise a `ValueError` if the specified overlap and window
            function are not compatible with COLA. The default is False.
        allow_fractional_frames : bool, optional
            If `False` (default), the signal will be split into a number of
            full frames, and any remaining fractional frame will be discarded.
            This will not allow COLA to be satisfied.
            If `True`, fractional frames will be retained and zero padded to
            create a full frames.

        Returns
        -------
        TimeHistoryArray
            Returns a new TimeHistoryArray with shape [num_frames,...] where
            ... is the shape of the original array.

        """
        # Check to see that the arguments were specified correctly
        if samples_per_frame is None and frame_length is None:
            raise ValueError('One of `samples_per_frame` or `frame_length` must be specified')
        elif samples_per_frame is not None and frame_length is not None:
            raise ValueError('`samples_per_frame` can not be specified along with `frame_length`')
        if overlap is None and overlap_samples is None:
            overlap_samples = 0
        elif overlap is not None and overlap_samples is not None:
            raise ValueError('`overlap` can not be specified along with `overlap_samples`')
        # Compute samples_per_frame and overlap_samples
        if samples_per_frame is None:
            samples_per_frame = int(np.round(frame_length/self.abscissa_spacing))
        # Make sure that we have an even number of samples per frame.
        if samples_per_frame % 2 == 1:
            raise ValueError('`samples_per_frame` must be an even number')
        if overlap_samples is None:
            overlap_samples = int(np.round(samples_per_frame*overlap))

        # If partial frames, then we will zero pad to make it the right length
        if allow_fractional_frames:
            self = self.zero_pad(samples_per_frame*2, left=True, right=True)
        num_frames = int(np.floor((self.num_elements-overlap_samples)/(samples_per_frame - overlap_samples)))
        frame_indices = np.arange(samples_per_frame) + np.arange(num_frames)[:, np.newaxis]*(samples_per_frame-overlap_samples)
        # See if we need to truncate empty frames
        if allow_fractional_frames:
            # Get rid of the first frame which is all zeros
            frame_indices = frame_indices[1:]
            # See if we need to get rid of the last frame, which could be all
            # zeros
            if frame_indices[-1, -1] + 1 == self.num_elements:
                frame_indices = frame_indices[:-1]

        # Put the "frame" axis at the front of the new array so all other parameters
        # can broadcast out across measurement frames
        new_abscissa = np.moveaxis(self.abscissa[..., frame_indices], -2, 0)
        new_ordinate = np.moveaxis(self.ordinate[..., frame_indices], -2, 0)

        # Now apply the window
        if window is None:
            window = 'boxcar'
        if isinstance(window, str) or isinstance(window, tuple):
            window = get_window(window, samples_per_frame)
        try:
            new_ordinate *= window
        except ValueError:
            raise ValueError('Could Not Multiply Window Function (shape {:}) by Ordinate (shape {:})'.format(window.shape, new_ordinate.shape))
        if check_cola:
            if not sig.check_COLA(window, samples_per_frame, overlap_samples):
                raise ValueError('COLA Check Failed: specified window and overlap do not result in a constant overlap-add condition, see scipy.check_COLA for more information')
                
        return data_array(FunctionTypes.TIME_RESPONSE, new_abscissa, new_ordinate,
                          self.coordinate, self.comment1, self.comment2, self.comment3,
                          self.comment4, self.comment5)
    

    def mimo_forward(self, transfer_function):
        """
        Performs the forward mimo calculation via convolution.

        Parameters
        ----------
        transfer_function : TransferFunctionArray or ImpulseResponseFunctionArray
            This is the FRFs that will be used in the forward problem. A matrix of IRFs
            is prefered, but FRFs can also be used, although the FRFs will be immediately
            converted to IRFs.

        Raises
        ------
        ValueError
            If the sampling rates for the data and IRFs/FRFs don't match.
        ValueError
            If the references in the IRFs/FRFs don't match the supplied input
            data.

        Returns
        -------
        TimeHistoryArray
            Response time histories

        """
        # Converting FRFs to IRFs, if required
        if isinstance(transfer_function, TransferFunctionArray):
            transfer_function = transfer_function.ifft()

        # Some initial organization
        transfer_function = transfer_function.reshape_to_matrix()
        reference_dofs = transfer_function.reference_coordinate[0, :]
        response_dofs = transfer_function.response_coordinate[:, 0]
        self = self[reference_dofs[..., np.newaxis]]
        irfs = np.moveaxis(transfer_function.ordinate, -2, 0)
        num_references, number_responses, model_order = irfs.shape
        signal_length = self.num_elements

        # Checking to see if the sampling rates are the same for both data sets
        if not np.isclose(self.abscissa_spacing, transfer_function.abscissa_spacing):
            raise ValueError('The transfer function sampling rate {:} does not match the time data {:}.'.format(
                1/transfer_function.abscissa_spacing,1/self.abscissa_spacing
            ))

        # Setting up and doing the convolution
        convolved_response = np.zeros((number_responses, signal_length), dtype=np.float64)
        for reference_irfs, inputs in zip(irfs, self.ordinate):
            convolved_response += oaconvolve(reference_irfs, inputs[np.newaxis, :])[:, :signal_length]

        return data_array(FunctionTypes.TIME_RESPONSE, self.abscissa[0], convolved_response, response_dofs[..., np.newaxis])

    def mimo_inverse(self, transfer_function,
                     time_method='single_frame',
                     cola_frame_length=None,
                     cola_window='hann',
                     cola_overlap=None,
                     zero_pad_length=None,
                     inverse_method='standard',
                     response_weighting_matrix=None,
                     reference_weighting_matrix=None,
                     regularization_weighting_matrix=None,
                     regularization_parameter=None,
                     cond_num_threshold=None,
                     num_retained_values=None,
                     transfer_function_odd_samples = False):
        """
        Performs the inverse source estimation for time domain (transient) problems
        using Fourier deconvolution. The response nodes used in the inverse source
        estimation are the ones contained in the supplied FRF matrix.

        Parameters
        ----------
        transer_function : TransferFunctionArray or ImpulseResponseFunctionArray
            This is the FRFs that will be used in the inverse source estimation
        time_method : str, optional
            The method to used to handle the time data for the inverse source
            estimation. The available options are:
                - single_frame - this method performs the Fourier deconvolution
                  via an FFT on a single frame that encompases the entire time
                  signal.
                - COLA - this method performs the Fourier deconvolution via a
                  series of FFTs on relatively small frames of the time signal
                  using a "constant overlap and add" method. This method may be
                  faster than the single_frame method.
        cola_frame_length : float, optional
            The frame length (in samples) if the COLA method is being used. The
            default frame length is Fs/df from the transfer function.
        cola_window : str, optional
            The desired window for the COLA procedure, must exist in the scipy
            window library. The default is a hann window.
        cola_overlap : int, optional
            The number of overlapping samples between measurement frames in the
            COLA procedure.  If not specified, a default value of half the
            cola_frame_length is used.
        zero_pad_length : int, optional
            The number of zeros used to pre and post pad the response data, to
            avoid convolution wrap-around error. The default is to use the
            "determine_zero_pad_length" function to determine the zero_pad_length.
        inverse_method : str, optional
            The method to be used for the FRF matrix inversions. The available
            methods are:
                - standard - basic pseudo-inverse via numpy.linalg.pinv with the
                  default rcond parameter, this is the default method
                - threshold - pseudo-inverse via numpy.linalg.pinv with a specified
                  condition number threshold
                - tikhonov - pseudo-inverse using the Tikhonov regularization method
                - truncation - pseudo-inverse where a fixed number of singular values
                  are retained for the inverse
        response_weighting_matrix : sdpy.Matrix or np.ndarray, optional
            Not currently implemented
        reference_weighting_matrix : sdpy.Matrix or np.ndarray, optional
            Not currently implemented
        regularization_weighting_matrix : sdpy.Matrix, optional
            Matrix used to weight input degrees of freedom via Tikhonov regularization.
        regularization_parameter : float or np.ndarray, optional
            Scaling parameter used on the regularization weighting matrix when the tikhonov
            method is chosen. A vector of regularization parameters can be provided so the
            regularization is different at each frequency line. The vector must match the
            length of the FRF abscissa in this case (either be size [num_lines,] or [num_lines, 1]).
        cond_num_threshold : float or np.ndarray, optional
            Condition number used for SVD truncation when the threshold method is chosen.
            A vector of condition numbers can be provided so it varies as a function of
            frequency. The vector must match the length of the FRF abscissa in this case
            (either be size [num_lines,] or [num_lines, 1]).
        num_retained_values : float or np.ndarray, optional
            Number of singular values to retain in the pseudo-inverse when the truncation
            method is chosen. A vector of can be provided so the number of retained values
            can change as a function of frequency. The vector must match the length of the
            FRF abscissa in this case (either be size [num_lines,] or [num_lines, 1]).
        transfer_function_odd_samples : bool, optional
            If True, then it is assumed that the spectrum has been constructed
            from a signal with an odd number of samples.  Note that this
            function uses the rfft function from scipy to compute the
            inverse fast fourier transform.  The irfft function is not round-trip
            equivalent for odd functions, because by default it assumes an even
            signal length.  For an odd signal length, the user must either specify
            transfer_function_odd_samples = True to make it round-trip equivalent.

        Raises
        ------
        NotImplementedError
            If a response weighting matrix is supplied
        NotImplementedError
            If a reference weighting matrix is supplied
        ValueError
            If the sampling rates for the data and FRFs don't match.
        ValueError
            If the number of responses in the FRFs don't match the supplied response
            data.

        Returns
        -------
        TimeHistoryArray
            Time history array of the estimated sources

        Notes
        -----
        This function computes the time domain inputs required to match the target time traces
        using Fourier deconvolution, which is essentially a frequency domain problem. The general
        method is to compute the frequency spectrum of the target time traces, then solve the
        inverse problem in the time domain using the supplied FRFs (H^+ * X). The inverse of the
        FRF matrix is found using the same methods as the mimo_inverse function for the
        PowerSpectralDensityArray class. The input spectrum is then converted back to the time
        domain via a inverse fourier transform.

        The 0 Hz component is explicitly rejected from the FRFs, so the estimated forces cannot
        include a 0 Hz component. 
        
        References
        ----------
        .. [1] Wikipedia, "Moore-Penrose inverse".
               https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
        .. [2] A.N. Tithe, D.J. Thompson, The quantification of structure-borne transmission pathsby inverse methods. Part 2: Use of regularization techniques,
               Journal of Sound and Vibration, Volume 264, Issue 2, 2003, Pages 433-451, ISSN 0022-460X,
               https://doi.org/10.1016/S0022-460X(02)01203-8.
        .. [3] Wikipedia, "Ridge regression".
               https://en.wikipedia.org/wiki/Ridge_regression
        .. [4] Wikipedia, "Overlap-add Method".
               https://en.wikipedia.org/wiki/Overlap-add_method
        """
        # Converting IRFs to FRFs, if required
        if isinstance(transfer_function, ImpulseResponseFunctionArray):
            transfer_function = transfer_function.fft()

        # Initial orginization of the data
        indexed_transfer_function = transfer_function.reshape_to_matrix()
        response_dofs = indexed_transfer_function[:, 0].response_coordinate
        reference_dofs = indexed_transfer_function[0, :].reference_coordinate

        indexed_response_data = self[response_dofs[..., np.newaxis]]

        indexed_irf = indexed_transfer_function.ifft()
        model_order = indexed_irf.num_elements

        # Checking to see if the sampling rates are the same for both data sets
        fs_inputs = 1/indexed_response_data.abscissa_spacing
        fs_frf = indexed_transfer_function.flatten()[0].abscissa[-1]*2
        if not np.isclose(fs_frf, fs_inputs):
            raise ValueError('The transfer function sampling rate does not match the time data.')

        # Preparing the response data and FRFs for the source estimation
        if time_method == 'single_frame':
            # Zero pad for convolution wrap-around
            if zero_pad_length is None:
                padded_response = indexed_response_data.zero_pad(2*model_order, left=True, right = True,
                                                                 use_next_fast_len = True)
            else:
                padded_response = indexed_response_data.zero_pad(zero_pad_length, left=True, right=True)
            actual_zero_pad = padded_response.num_elements - indexed_response_data.num_elements
            # Now make the FRFs the same size
            modified_frfs = indexed_transfer_function.interpolate_by_zero_pad(padded_response.num_elements,odd_num_samples = transfer_function_odd_samples)
            modified_frfs.ordinate[...,0] = 0
            padded_frequency_domain_data = padded_response.fft(norm='backward')
            irfft_num_samples = padded_response.num_elements
        elif time_method == 'cola':
            if cola_frame_length is None:
                cola_frame_length = int(model_order + model_order % 2) # This is a slightly strange operation to gaurantee an even frame length
            if cola_overlap is None:
                cola_overlap = cola_frame_length//2
            # Split into measurement frames
            segmented_data = indexed_response_data.split_into_frames(
                samples_per_frame=cola_frame_length,
                overlap_samples=cola_overlap,
                window=cola_window,
                check_cola=True,
                allow_fractional_frames=True)
            # Zero pad
            if zero_pad_length is None:
                zero_padded_data = segmented_data.zero_pad(
                    2*model_order, left=True, right=True, use_next_fast_len=True)
            else:
                zero_padded_data = segmented_data.zero_pad(
                    zero_pad_length, left=True, right=True)
            actual_zero_pad = zero_padded_data.num_elements - segmented_data.num_elements
            modified_frfs = indexed_transfer_function.interpolate_by_zero_pad(zero_padded_data.num_elements,odd_num_samples = transfer_function_odd_samples)
            modified_frfs.ordinate[...,0] = 0
            padded_frequency_domain_data = zero_padded_data.fft(norm='backward')
            irfft_num_samples = zero_padded_data.num_elements
        else:
            raise NameError('The selected time method is not available')

        # Need to interpolate the conditioning parameters to match the length of the padded
        # FRFs
        if cond_num_threshold is not None:
            cond_num_threshold = np.asarray(cond_num_threshold, dtype=np.float64)
            if cond_num_threshold.size > 1:
                cond_num_threshold = interp1d(
                    indexed_transfer_function[0, 0].abscissa,
                    cond_num_threshold,
                    'linear',
                    bounds_error=False,
                    fill_value=(cond_num_threshold[0], cond_num_threshold[-1]),
                    assume_sorted=True)(modified_frfs[0, 0].abscissa)
        if num_retained_values is not None:
            num_retained_values = np.asarray(num_retained_values, dtype=np.intc)
            if num_retained_values.size > 1:
                num_retained_values = interp1d(
                    indexed_transfer_function[0, 0].abscissa,
                    num_retained_values,
                    'previous',
                    bounds_error=False,
                    fill_value=(num_retained_values[0], num_retained_values[-1]),
                    assume_sorted=True)(modified_frfs[0, 0].abscissa)
        if regularization_parameter is not None:
            regularization_parameter = np.asarray(regularization_parameter, dtype=np.float64)
            if regularization_parameter.size > 1:
                regularization_parameter = interp1d(
                    indexed_transfer_function[0, 0].abscissa,
                    regularization_parameter,
                    'linear',
                    bounds_error=False,
                    fill_value=(regularization_parameter[0], regularization_parameter[-1]),
                    assume_sorted=True)(modified_frfs[0, 0].abscissa)

        # Set up weighting matrices
        if response_weighting_matrix is not None:
            raise NotImplementedError('Response weighting has not been implemented yet')
        if reference_weighting_matrix is not None:
            raise NotImplementedError('Reference weighting has not been implemented yet')
        if regularization_weighting_matrix is not None:
            regularization_weighting_matrix = regularization_weighting_matrix[reference_dofs, reference_dofs].matrix

        # Now solve the inverse problem
        frf_pinv = frf_inverse(np.moveaxis(modified_frfs.ordinate, -1, 0),
                               method=inverse_method,
                               response_weighting_matrix=response_weighting_matrix,
                               reference_weighting_matrix=reference_weighting_matrix,
                               regularization_weighting_matrix=regularization_weighting_matrix,
                               regularization_parameter=regularization_parameter,
                               cond_num_threshold=cond_num_threshold,
                               num_retained_values=num_retained_values)
        method_statement_start = 'The FRFs are being inverted using the '
        method_statement_end = ' method'
        print(method_statement_start+inverse_method+method_statement_end)

        # Get the first modified frequency line above the original starting point of the FRF
        inverse_start_index = np.argmax(modified_frfs[0, 0].abscissa >= indexed_transfer_function[0, 0].abscissa[0])

        # Doing the source estimation
        padded_frequency_domain_data = np.moveaxis(padded_frequency_domain_data.ordinate, -1, -2)[..., np.newaxis]
        forces_frequency_domain = frf_pinv@padded_frequency_domain_data
        forces_frequency_domain[..., :inverse_start_index, :, 0] = 0
        forces_time_domain_with_padding = scipyfft.irfft(forces_frequency_domain[..., 0], n = irfft_num_samples, axis=-2, norm='backward')
        # Compute the zero padding used
        pre_pad_length = actual_zero_pad//2
        post_pad_length = actual_zero_pad - pre_pad_length
        if time_method == 'single_frame':
            forces_time_domain = forces_time_domain_with_padding[pre_pad_length:-post_pad_length, :]
            return_val = data_array(FunctionTypes.TIME_RESPONSE, indexed_response_data[0].abscissa, np.moveaxis(forces_time_domain, 0, -1),
                                    reference_dofs[..., np.newaxis])
        elif time_method == 'cola':
            forces_time_domain_with_padding = data_array(
                FunctionTypes.TIME_RESPONSE,
                zero_padded_data.abscissa[..., :1, :],
                np.moveaxis(forces_time_domain_with_padding, -1, -2),
                reference_dofs[:, np.newaxis])

            # Assemble the COLA
            forces_time_domain_with_padding = TimeHistoryArray.overlap_and_add(
                forces_time_domain_with_padding,
                overlap_samples=actual_zero_pad + cola_overlap)
            start_index = cola_frame_length - cola_overlap + pre_pad_length
            end_index = start_index + indexed_response_data.num_elements
            return_val = forces_time_domain_with_padding.idx_by_el[start_index:end_index]

            # Compute COLA weighting
            window_fn = get_window(cola_window, cola_frame_length)
            step = cola_frame_length - cola_overlap
            weighting = np.median(sum(window_fn[ii*step:(ii+1)*step] for ii in range(cola_frame_length//step)))
            return_val = return_val / weighting

        return return_val

    def rms(self):
        return np.sqrt(np.mean(self.ordinate**2, axis=-1))

    def to_rattlesnake_specification(self, filename, coordinate_order=None,
                                     min_time=None,
                                     max_time=None):
        if coordinate_order is not None:
            if coordinate_order.ndim == 1:
                coordinate_order = coordinate_order[:, np.newaxis]
            reshaped_data = self[coordinate_order]
        else:
            reshaped_data = self
        if min_time is not None or max_time is not None:
            if min_time is None:
                min_time = -np.inf
            if max_time is None:
                max_time = np.inf
            reshaped_data = reshaped_data.extract_elements_by_abscissa(min_time, max_time)
        np.savez(filename,
                 t=reshaped_data[0].abscissa - reshaped_data[0].abscissa[0],
                 signal=reshaped_data.ordinate)

    def find_signal_shift(self, other_signal,
                          compute_subsample_shift=True,
                          good_line_threshold=0.01):
        """
        Computes the shift between two sets of time signals

        This is the amount that `other_signal` leads `self`.  If the time shift
        is positive, it means that features in `other_signal` occur earlier in
        time compared to `self`.  If the time shift is negative, it means that
        features in `other_signal` occur later in time compared to `self`.

        To align two signals, you can take the time shift from this function and
        pass it into the `shift_signal` method of `other_signal`.

        Parameters
        ----------
        other_signal : TimeHistoryArray
            The signal against which this signal should be compared in time.
            It should have the same coordinate ordering and the same number of
            abscissa as this signal.
        compute_subsample_shift : bool, optional
            If False, this function will simply align to the nearest sample.
            If True, this function will attempt to use FFT phases to compute a
            subsample shift between the signals.  Default is True.
        good_line_threshold : float, optional
            Threshold to use to compute "good" frequency lines.  This function
            uses phase to compute subsample shifts.  If there are frequency
            lines without content, they should be ignored.  Frequency lines less
            than `good_line_threshold` times the maximum of the spectra are
            ignored. The default is 0.01.

        Returns
        -------
        time_shift : float
            The time difference between the two signals.

        """
        this_fft = self.fft()

        this_ordinate = this_fft.ordinate
        other_ordinate = other_signal.fft().ordinate

        correlation = scipyfft.irfft(this_ordinate*other_ordinate.conj())
        time_shift_indices = int(np.mean(np.argmax(correlation, axis=-1)))

        # Roll the arrays to get them to align
        shifted_signal = other_signal.copy()
        shifted_signal.ordinate = np.roll(other_signal.ordinate, time_shift_indices, axis=-1)

        dt = np.mean(np.diff(self.abscissa, axis=-1))
        time_shift = dt*time_shift_indices

        if compute_subsample_shift:
            # Now compute the subsample shift
            shifted_ordinate = shifted_signal.fft().ordinate

            # Only compute at frequency lines where there's signal
            good_lines = np.abs(shifted_ordinate)/np.max(np.abs(shifted_ordinate), axis=-1, keepdims=True) > good_line_threshold
            good_lines[..., 0] = False

            phase_difference = np.angle(this_ordinate/shifted_ordinate)

            phase_slope = np.median(phase_difference[good_lines]/this_fft.abscissa[good_lines])

            time_shift -= phase_slope/(2*np.pi)

        # Wrap so it's negative if that's the smaller distance
        if time_shift > dt*self.num_elements/2:
            time_shift -= dt*self.num_elements

        return time_shift

    def shift_signal(self, time_shift):
        """
        Shift a signal in time by a specified amount.

        Utilizes the FFT shift theorem to move a signal in time.

        Parameters
        ----------
        time_shift : float
            The time shift to apply to the signal.  A negative value will cause
            features to occur earlier in time.  A positive value will cause
            features to occur later in time.

        Returns
        -------
        shifted_signal : TimeHistoryArray
            A shifted version of the original signal.

        """
        phase_shift_slope = -time_shift*2*np.pi
        signal_fft = self.fft()

        signal_fft.ordinate *= np.exp(1j*phase_shift_slope*signal_fft.flatten()[0].abscissa)

        shifted_signal = signal_fft.ifft()

        return shifted_signal

    @staticmethod
    def overlap_and_add(functions_to_overlap, overlap_samples):
        """
        Creates a time history by overlapping and adding other time histories.

        Parameters
        ----------
        functions_to_overlap : TimeHistoryArray or list of TimeHistoryArray
            A set of TimeHistoryArrays to overlap and add together.  If a single
            TimeHistoryArray is specified, then the first dimension will be used
            to split the signal into segments.  All TimeHistoryArrays must have
            the same shape and metadata, but need not have the same number of
            elements.
        overlap_samples : int
            Number of samples to overlap the segments as they are added together


        Returns
        -------
        TimeHistoryArray
            A TimeHistoryArray consisting of the signals overlapped and added
            together.

        Notes
        -----
        All metadata is taken from the first signal.  No checks are performed to
        make sure that the subsequent functions have common coordinates or
        abscissa spacing.
        """
        # First compute the final length of the signal
        num_samples = functions_to_overlap[0].num_elements
        for signal in functions_to_overlap[1:]:
            num_samples += signal.num_elements-overlap_samples
        # Set up the ordinate
        ordinate = np.zeros(functions_to_overlap[0].shape+(num_samples,))
        # Go through each frame and add it to the function
        starting_index = 0
        for signal in functions_to_overlap:
            ordinate[..., starting_index:starting_index+signal.num_elements] += signal.ordinate
            starting_index += signal.num_elements - overlap_samples
        # Now set up the rest of the metadata
        abscissa = functions_to_overlap[0].abscissa_spacing*np.arange(num_samples) + functions_to_overlap[0].abscissa.min()
        return data_array(FunctionTypes.TIME_RESPONSE, abscissa, ordinate,
                          functions_to_overlap[0].coordinate)

    def remove_rigid_body_motion(self, geometry):
        """
        Removes rigid body displacements from time data.

        This function assumes the current TimeHistoryArray is a displacement
        signal and adds it to the geometry to create node positions over time,
        then it fits a rigid coordinate transformation to each time step and
        subtracts off that portion of the motion from the displacement signal.

        Parameters
        ----------
        geometry : Geometry
            Geometry with which the node positions are computed

        Returns
        -------
        TimeHistoryArray
            A TimeHistoryArray with the rigid body component of motion removed

        """
        nodes = np.unique(self.coordinate.node)
        dofs = coordinate_array(nodes[:, np.newaxis], [1, 2, 3])
        sorted_self = self[dofs[..., np.newaxis]]
        displacements = sorted_self.ordinate
        abscissa = sorted_self.abscissa
        starting_positions = geometry.node(nodes).coordinate[..., np.newaxis]
        positions_over_time = displacements + starting_positions
        # Rearrange indices to match the rigid transformation code
        y = np.transpose(positions_over_time, [2, 1, 0])
        x = np.transpose(starting_positions, [2, 1, 0])
        R, t = lstsq_rigid_transform(x, y)
        y_rigid = R@x+t
        nonrigid_displacements = data_array(
            FunctionTypes.TIME_RESPONSE,
            abscissa,
            np.transpose(y - y_rigid, [2, 1, 0]),
            dofs[..., np.newaxis],
            sorted_self.comment1,
            sorted_self.comment2,
            sorted_self.comment3,
            sorted_self.comment4,
            sorted_self.comment5)
        return nonrigid_displacements[self.coordinate]

    def stft(self, samples_per_frame=None, frame_length=None,
             overlap=None, overlap_samples=None, window=None,
             check_cola=False, allow_fractional_frames=False,
             norm='backward'):
        """
        Computes a Short-Time Fourier Transform (STFT)

        The time history is split up into frames with specified length and
        computes the spectra for each frame.

        Parameters
        ----------
        samples_per_frame : int, optional
            Number of samples in each measurement frame. Either this argument
            or `frame_length` must be specified.  If both or neither are
            specified, a `ValueError` is raised.
        frame_length : float, optional
            Length of each measurement frame in the same units as the `abscissa`
            field (`samples_per_frame` = `frame_length`/`self.abscissa_spacing`).
            Either this argument or `samples_per_frame` must be specified.  If
            both or neither are specified, a `ValueError` is raised.
        overlap : float, optional
            Fraction of the measurement frame to overlap (i.e. 0.25 not 25 to
            overlap a quarter of the frame). Either this argument or
            `overlap_samples` must be specified.  If both are
            specified, a `ValueError` is raised.  If neither are specified, no
            overlap is used.
        overlap_samples : int, optional
            Number of samples in the measurement frame to overlap. Either this
            argument or `overlap_samples` must be specified.  If both
            are specified, a `ValueError` is raised.  If neither are specified,
            no overlap is used.
        window : str or tuple or array_like, optional
            Desired window to use. If window is a string or tuple, it is passed
            to `scipy.signal.get_window` to generate the window values, which
            are DFT-even by default. See `get_window` for a list of windows and
            required parameters. If window is array_like it will be used
            directly as the window and its length must be `samples_per_frame`.
            If not specified, no window will be applied.
        check_cola : bool, optional
            If `True`, raise a `ValueError` if the specified overlap and window
            function are not compatible with COLA. The default is False.
        allow_fractional_frames : bool, optional
            If `False` (default), the signal will be split into a number of
            full frames, and any remaining fractional frame will be discarded.
            This will not allow COLA to be satisfied.
            If `True`, fractional frames will be retained and zero padded to
            create a full frames.

        Returns
        -------
        frame_abscissa : np.ndarray
             The abscissa values at the center of each of the STFT frames
        stft : SpectrumArray
            A spectrum array with the first axis corresponding to the time
            values in `frame_abscissa`.
        """
        split_frames = self.split_into_frames(
            samples_per_frame, frame_length,
            overlap, overlap_samples, window,
            check_cola, allow_fractional_frames)
        frame_abscissa = np.median(split_frames.abscissa, axis=-1)
        stft = split_frames.fft(norm=norm)
        return frame_abscissa, stft

    def upsample(self, factor):
        """
        Upsamples a time history using frequency domain zero padding.

        Parameters
        ----------
        factor : int
            The upsample factor.

        Returns
        -------
        TimeHistoryArray
            A time history with a sample rate that is factor larger than the
            original signal

        """
        fft = self.fft()
        fft_zp = fft.zero_pad(fft.num_elements*(factor-1))
        return fft_zp.ifft()*factor

    def apply_transformation(self, transformation, invert_transformation=False):
        """
        Applies a transformations to the time traces.

        Parameters
        ----------
        transformation : Matrix
            The transformation to apply to the time traces. It should be a 
            SDynPy matrix object with the "transformed" coordinates on the 
            rows and the "physical" coordinates on the columns. The matrix 
            can only be be 2D.
        invert_reference_transformation : bool, optional
            Whether or not to invert the transformation when applying it to 
            the time traces. The default is False, which is standard practice. 
            The row/column ordering in the transformation should be flipped 
            if this is set to true.

        Raises
        ------
        ValueError
            If the transformation array has more than two dimensions.
        ValueError
            If the physical degrees of freedom in the transformation does not 
            match the spectra.
        
        Returns
        -------
        transformed_data : TimeHistoryArray
            The time traces with the transformations applied.
        """
        if not self.validate_common_abscissa():
            raise ValueError('The abscissa must be consistent accross all functions in the NDDataArray')

        physical_coordinate = np.unique(self.response_coordinate)
        original_data_ordinate = np.moveaxis(self[physical_coordinate[...,np.newaxis]].ordinate, -1, 0)[..., np.newaxis]
        
        if invert_transformation:
            if not np.all(np.unique(transformation.row_coordinate) == physical_coordinate):
                raise ValueError('The physical coordinates in the transformation do no match the spectra')
            transformed_coordinate = np.unique(transformation.column_coordinate)
            transformation_matrix = np.linalg.pinv(transformation[physical_coordinate, transformed_coordinate])
        elif not invert_transformation:
            if not np.all(np.unique(transformation.column_coordinate) == physical_coordinate):
                raise ValueError('The physical coordinates in the transformation do no match the spectra')
            transformed_coordinate = np.unique(transformation.row_coordinate)
            transformation_matrix = transformation[transformed_coordinate, physical_coordinate]
        
        if transformation_matrix.ndim != 2:
            raise ValueError('The transformation array must be two dimensional')

        transformed_data_ordinate = (transformation_matrix @ original_data_ordinate)[...,0]

        return data_array(FunctionTypes.TIME_RESPONSE, self.ravel().abscissa[0], np.moveaxis(transformed_data_ordinate, 0, -1), 
                          transformed_coordinate[...,np.newaxis])

    @classmethod
    def pseudorandom_signal(cls, dt, signal_length, coordinates,
                            min_frequency=None, max_frequency=None,
                            signal_rms=1, frames=1, frequency_shape=None,
                            different_realizations=False,
                            comment1='', comment2='', comment3='',
                            comment4='', comment5=''):
        """
        Generates a pseudorandom signal at the specified coordinates

        Parameters
        ----------
        dt : float
            Abscissa spacing in the final signal.
        signal_length : int
            Number of samples in the signal
        coordinates : CoordinateArray
            Coordinate array used to generate the signal.  If the last dimension
            of coordinates is not shape 1, then a new axis will be added to make
            it shape 1.  The shape of the resulting TimeHistoryArray will be
            determined by the shape of the input coordinates.
        min_frequency : float, optional
            Minimum frequency content in the signal. The default is the lowest
            nonzero frequency line.
        max_frequency : float, optional
            Maximum frequency content in the signal. The default is the highest
            frequency content in the signal, e.g. the Nyquist frequency.
        signal_rms : float or np.ndarray, optional
            RMS value for the generated signals. The default is 1.  The shape of
            this value should be broadcastable with the size of the
            generated TimeHistoryArray if different RMS values are desired for
            each signal.
        frames : int, optional
            Number of frames to generate.  These will essentially be repeats of
            the first frame for the number of frames specified. The default is 1.
        frequency_shape : function, optional
            An optional function that should accept a frequency value and return
            an amplitude at that frequency. The default is constant scaling
            across all frequency lines.
        different_realizations : bool
            An optional argument that specifies whether or not different
            functions should have different realizations of the pseudorandom
            signal, or if they should all be identical.
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
        TimeHistoryArray :
            A time history containing the specified pseudorandom signal

        """
        # Compute signal processing parameters
        total_frame_length = dt*signal_length
        df = 1/total_frame_length
        f_nyquist = 1/(2*dt)
        fft_lines = f_nyquist/df
        if frequency_shape is None:
            frequency_shape = _flat_frequency_shape
        if np.array(coordinates).dtype.type is np.str_:
            coordinates = coordinate_array(string_array=coordinates)
        # Get coordinate size
        if coordinates.ndim == 0 or coordinates.shape[-1] != 1:
            coordinates = coordinates[..., np.newaxis]
        ordinate = np.empty(coordinates.shape[:-1]+(signal_length*frames,))
        if different_realizations:
            for index in np.ndindex(coordinates.shape[:-1]):
                ordinate[index] = pseudorandom(fft_lines, f_nyquist,
                                               min_freq=min_frequency,
                                               max_freq=max_frequency,
                                               averages=frames,
                                               shape_function=frequency_shape)[1]
        else:
            ordinate[...] = pseudorandom(fft_lines, f_nyquist,
                                         min_freq=min_frequency,
                                         max_freq=max_frequency,
                                         averages=frames,
                                         shape_function=frequency_shape)[1]
        # Apply the RMS
        current_rms = np.sqrt(np.mean(ordinate**2, axis=-1))
        ordinate *= np.array(signal_rms)[..., np.newaxis]/current_rms[..., np.newaxis]
        # Now create the object
        abscissa = dt * np.arange(signal_length*frames)
        time_history = data_array(FunctionTypes.TIME_RESPONSE,
                                  abscissa,
                                  ordinate,
                                  coordinates,
                                  comment1, comment2, comment3, comment4, comment5)
        return time_history

    @classmethod
    def random_signal(cls, dt, signal_length, coordinates,
                      min_frequency=None, max_frequency=None,
                      signal_rms=1, frames=1, frequency_shape=None,
                      comment1='', comment2='', comment3='',
                      comment4='', comment5=''):
        """
        Generates a random signal with the specified parameters

        Parameters
        ----------
        dt : float
            Abscissa spacing in the final signal.
        signal_length : int
            Number of samples in the signal
        coordinates : CoordinateArray
            Coordinate array used to generate the signal.  If the last dimension
            of coordinates is not shape 1, then a new axis will be added to make
            it shape 1.  The shape of the resulting TimeHistoryArray will be
            determined by the shape of the input coordinates.
        min_frequency : float, optional
            Minimum frequency content in the signal. The default is the lowest
            nonzero frequency line.
        max_frequency : float, optional
            Maximum frequency content in the signal. The default is the highest
            frequency content in the signal, e.g. the Nyquist frequency.
        signal_rms : float or np.ndarray, optional
            RMS value for the generated signals. The default is 1.  The shape of
            this value should be broadcastable with the size of the
            generated TimeHistoryArray if different RMS values are desired for
            each signal.
        frames : int, optional
            Number of frames to generate.  These will essentially be repeats of
            the first frame for the number of frames specified. The default is 1.
        frequency_shape : function, optional
            An optional function that should accept a frequency value and return
            an amplitude at that frequency. The default is constant scaling
            across all frequency lines.
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
        TimeHistoryArray :
            A time history containing the specified random signal
        """
        if np.array(coordinates).dtype.type is np.str_:
            coordinates = coordinate_array(string_array=coordinates)
        # Get coordinate size
        if coordinates.ndim == 0 or coordinates.shape[-1] != 1:
            coordinates = coordinates[..., np.newaxis]
        ordinate = np.random.randn(*coordinates.shape[:-1], signal_length*frames)
        if (min_frequency is not None or max_frequency is not None or frequency_shape is not None):
            if frequency_shape is None:
                frequency_shape = _flat_frequency_shape
            frequencies = scipyfft.rfftfreq(signal_length*frames, dt)
            fft = scipyfft.rfft(ordinate, axis=-1)
            for index, frequency in enumerate(frequencies):
                if min_frequency is not None and frequency < min_frequency:
                    fft[..., index] = 0
                elif max_frequency is not None and frequency > max_frequency:
                    fft[..., index] = 0
                else:
                    fft[..., index] *= frequency_shape(frequency)
            ordinate = scipyfft.irfft(fft, n = signal_length*frames, axis=-1)
        # Now set RMS
        current_rms = np.sqrt(np.mean(ordinate**2, axis=-1))
        ordinate *= np.array(signal_rms)[..., np.newaxis]/current_rms[..., np.newaxis]
        # Now create the object
        abscissa = dt * np.arange(signal_length*frames)
        time_history = data_array(FunctionTypes.TIME_RESPONSE,
                                  abscissa,
                                  ordinate,
                                  coordinates,
                                  comment1, comment2, comment3, comment4, comment5)
        return time_history

    @classmethod
    def sine_signal(cls, dt, signal_length, coordinates,
                    frequency, amplitude=1, phase=0,
                    comment1='', comment2='', comment3='',
                    comment4='', comment5=''):
        """
        Creates a sinusoidal signal with the specified parameters

        Parameters
        ----------
        dt : float
            Abscissa spacing in the final signal.
        signal_length : int
            Number of samples in the signal
        coordinates : CoordinateArray
            Coordinate array used to generate the signal.  If the last dimension
            of coordinates is not shape 1, then a new axis will be added to make
            it shape 1.  The shape of the resulting TimeHistoryArray will be
            determined by the shape of the input coordinates.
        frequency : float or np.ndarray
            Frequency of signal that will be generated.  If multiple frequencies
            are specified, they must broadcast with the final size of the
            TimeHistoryArray.
        amplitude : TYPE, optional
            Amplitude of signal that will be generated.  If multiple amplitudes
            are specified, they must broadcast with the final size of the
            TimeHistoryArray. The default is 1.
        phase : TYPE, optional
            Phase of signal that will be generated.  If multiple phases
            are specified, they must broadcast with the final size of the
            TimeHistoryArray.. The default is 0.
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
        TimeHistoryArray :
            A time history containing the specified sine signal

        """
        if np.array(coordinates).dtype.type is np.str_:
            coordinates = coordinate_array(string_array=coordinates)
        # Get coordinate size
        if coordinates.ndim == 0 or coordinates.shape[-1] != 1:
            coordinates = coordinates[..., np.newaxis]
        ordinate = np.empty(coordinates.shape[:-1]+(signal_length,))
        ordinate[...] = sine(frequency, dt, signal_length, amplitude, phase)
        # Now create the object
        abscissa = dt * np.arange(signal_length)
        time_history = data_array(FunctionTypes.TIME_RESPONSE,
                                  abscissa,
                                  ordinate,
                                  coordinates,
                                  comment1, comment2, comment3, comment4, comment5)
        return time_history

    @classmethod
    def burst_random_signal(cls, dt, signal_length, coordinates,
                            min_frequency=None, max_frequency=None,
                            signal_rms=1, frames=1, frequency_shape=None,
                            on_fraction=0.5, delay_fraction=0.0,
                            ramp_fraction=0.05,
                            comment1='', comment2='', comment3='',
                            comment4='', comment5=''):
        """
        Generates a burst random signal with the specified parameters

        Parameters
        ----------
        dt : float
            Abscissa spacing in the final signal.
        signal_length : int
            Number of samples in the signal
        coordinates : CoordinateArray
            Coordinate array used to generate the signal.  If the last dimension
            of coordinates is not shape 1, then a new axis will be added to make
            it shape 1.  The shape of the resulting TimeHistoryArray will be
            determined by the shape of the input coordinates.
        min_frequency : float, optional
            Minimum frequency content in the signal. The default is the lowest
            nonzero frequency line.
        max_frequency : float, optional
            Maximum frequency content in the signal. The default is the highest
            frequency content in the signal, e.g. the Nyquist frequency.
        signal_rms : float or np.ndarray, optional
            RMS value for the generated signals. The default is 1.  The shape of
            this value should be broadcastable with the size of the
            generated TimeHistoryArray if different RMS values are desired for
            each signal.  Note that the RMS will be computed for the "burst"
            part of the signal and not include the zero portion of the signal.
        frames : int, optional
            Number of frames to generate.  These will essentially be repeats of
            the first frame for the number of frames specified. The default is 1.
        frequency_shape : function, optional
            An optional function that should accept a frequency value and return
            an amplitude at that frequency. The default is constant scaling
            across all frequency lines.
        on_fraction : float, optional
            The fraction of the frame that the signal is active, default is 0.5.
            This portion includes the ramp_fraction, so an on_fraction of 0.5 with
            a ramp_fraction of 0.05 will be at full level for 0.5-2*0.05 = 0.4
            fraction of the full measurement frame.
        delay_fraction : float, optional
            The fraction of the frame that is empty before the signal starts,
            default is 0.0
        ramp_fraction : float, optional
            The fraction of the frame that is used to ramp between the off
            and active signal
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
        TimeHistoryArray :
            A time history containing the specified burst random signal
        """
        if np.array(coordinates).dtype.type is np.str_:
            coordinates = coordinate_array(string_array=coordinates)
        # Get coordinate size
        if coordinates.ndim == 0 or coordinates.shape[-1] != 1:
            coordinates = coordinates[..., np.newaxis]
        ordinate = np.random.randn(*coordinates.shape[:-1], signal_length*frames)
        if (min_frequency is not None or max_frequency is not None or frequency_shape is not None):
            if frequency_shape is None:
                frequency_shape = _flat_frequency_shape
            frequencies = scipyfft.rfftfreq(signal_length*frames, dt)
            fft = scipyfft.rfft(ordinate, axis=-1)
            for index, frequency in enumerate(frequencies):
                if min_frequency is not None and frequency < min_frequency:
                    fft[..., index] = 0
                elif max_frequency is not None and frequency > max_frequency:
                    fft[..., index] = 0
                else:
                    fft[..., index] *= frequency_shape(frequency)
            ordinate = scipyfft.irfft(fft, n = signal_length*frames, axis=-1)
        # Now set RMS
        current_rms = np.sqrt(np.mean(ordinate**2, axis=-1))
        ordinate *= np.array(signal_rms)[..., np.newaxis]/current_rms[..., np.newaxis]
        # Apply the window
        delay_samples = int(delay_fraction * signal_length)
        ramp_samples = int(ramp_fraction * signal_length)
        on_samples = int(on_fraction * signal_length)
        burst_window = np.zeros(coordinates.shape[:-1]+(signal_length,))
        burst_window[..., delay_samples:delay_samples +
                     on_samples] = ramp_envelope(on_samples, ramp_samples)
        burst_window = np.tile(burst_window, frames)
        ordinate *= burst_window
        # Now create the object
        abscissa = dt * np.arange(signal_length*frames)
        time_history = data_array(FunctionTypes.TIME_RESPONSE,
                                  abscissa,
                                  ordinate,
                                  coordinates,
                                  comment1, comment2, comment3, comment4, comment5)
        return time_history

    @classmethod
    def chirp_signal(cls, dt, signal_length, coordinates,
                     start_frequency=None, end_frequency=None,
                     frames=1, amplitude_function=None,
                     force_integer_cycles=True,
                     comment1='', comment2='', comment3='',
                     comment4='', comment5=''):
        """
        Creates a chirp (sine sweep) signal with the specified parameters

        Parameters
        ----------
        dt : float
            Abscissa spacing in the final signal.
        signal_length : int
            Number of samples in the signal
        coordinates : CoordinateArray
            Coordinate array used to generate the signal.  If the last dimension
            of coordinates is not shape 1, then a new axis will be added to make
            it shape 1.  The shape of the resulting TimeHistoryArray will be
            determined by the shape of the input coordinates.
        start_frequency : TYPE, optional
            Starting frequency content in the signal. The default is the lowest
            nonzero frequency line.
        end_frequency : TYPE, optional
            Stopping frequency content in the signal. The default is the highest
            non-nyquist frequency line.
        frames : int, optional
            Number of frames to generate.  These will essentially be repeats of
            the first frame for the number of frames specified. The default is 1.
        amplitude_function : function, optional
            An optional function that should accept a frequency value and return
            an amplitude at that frequency. The default is constant scaling
            across all frequencies.  Multiple amplitudes can be returned as long
            as they broadcast with the shape of the final TimeHistoryArray.
        force_integer_cycles : bool, optional
            If True, it will force an integer number of cycles, which will
            adjust the maximum frequency of the signal.  This will ensure the
            signal is continuous if repeated.  If False, the
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
        TimeHistoryArray :
            A time history containing the specified chirp signal

        """
        if np.array(coordinates).dtype.type is np.str_:
            coordinates = coordinate_array(string_array=coordinates)
        # Get coordinate size
        if coordinates.ndim == 0 or coordinates.shape[-1] != 1:
            coordinates = coordinates[..., np.newaxis]
        # Create the chirp
        signal_length_in_time = dt*signal_length
        df = 1/signal_length_in_time
        if start_frequency is None:
            start_frequency = df
        if end_frequency is None:
            end_frequency = 1/dt/2-df
        ordinate = np.empty(coordinates.shape[:-1]+(signal_length,))
        ordinate[...] = chirp(start_frequency, end_frequency, signal_length_in_time,
                              dt, force_integer_cycles)
        if amplitude_function is not None:
            if force_integer_cycles:
                n_cycles = np.ceil(end_frequency * signal_length_in_time)
                end_frequency = n_cycles / signal_length_in_time
            frequency_slope = (end_frequency - start_frequency) / signal_length
            frequency_over_time = start_frequency + frequency_slope*np.arange(signal_length)
            amplitude_over_time = np.array([amplitude_function(f) for f in frequency_over_time])
            ordinate *= amplitude_over_time
        # Create the measurement frames
        ordinate = np.tile(ordinate, frames)
        # Now create the object
        abscissa = dt * np.arange(signal_length*frames)
        time_history = data_array(FunctionTypes.TIME_RESPONSE,
                                  abscissa,
                                  ordinate,
                                  coordinates,
                                  comment1, comment2, comment3, comment4, comment5)
        return time_history

    @classmethod
    def pulse_signal(cls, dt, signal_length, coordinates,
                     pulse_width=None, pulse_time=None, pulse_peak=1,
                     sine_exponent=1, frames=1,
                     comment1='', comment2='', comment3='',
                     comment4='', comment5=''):
        """
        Creates a pulse using a cosine function raised to a specified exponent

        Parameters
        ----------
        dt : float
            Abscissa spacing in the final signal.
        signal_length : int
            Number of samples in the signal
        coordinates : CoordinateArray
            Coordinate array used to generate the signal.  If the last dimension
            of coordinates is not shape 1, then a new axis will be added to make
            it shape 1.  The shape of the resulting TimeHistoryArray will be
            determined by the shape of the input coordinates.
        pulse_width : float, optional
            With of the pulse in the same units as `dt`. The default is 5*dt.
        pulse_time : float, optional
            The time of the pulse's occurance in the same units as `dt`.
            The default is 5*dt.
        pulse_peak : float, optional
            The peak amplitude of the pulse. The default is 1.
        sine_exponent : float, optional
            The exponent that the cosine function is raised to. The default is 1.
        frames : int, optional
            Number of frames to generate.  These will essentially be repeats of
            the first frame for the number of frames specified. The default is 1.
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
        TimeHistoryArray :
            A time history containing the specified pulse signal

        """
        if np.array(coordinates).dtype.type is np.str_:
            coordinates = coordinate_array(string_array=coordinates)
        # Get coordinate size
        if coordinates.ndim == 0 or coordinates.shape[-1] != 1:
            coordinates = coordinates[..., np.newaxis]
        ordinate = np.empty(coordinates.shape[:-1]+(signal_length,))
        if pulse_time is None:
            pulse_time = dt*5
        if pulse_width is None:
            pulse_width = dt*5
        ordinate[...] = pulse(signal_length, pulse_time, pulse_width, pulse_peak,
                              dt, sine_exponent)
        # Create the measurement frames
        ordinate = np.tile(ordinate, frames)
        # Now create the object
        abscissa = dt * np.arange(signal_length*frames)
        time_history = data_array(FunctionTypes.TIME_RESPONSE,
                                  abscissa,
                                  ordinate,
                                  coordinates,
                                  comment1, comment2, comment3, comment4, comment5)
        return time_history

    @classmethod
    def haversine_signal(cls, dt, signal_length, coordinates,
                         pulse_width=None, pulse_time=None, pulse_peak=1,
                         frames=1,
                         comment1='', comment2='', comment3='',
                         comment4='', comment5=''):
        """
        Creates a haversine pulse with the specified parameters

        Parameters
        ----------
        dt : float
            Abscissa spacing in the final signal.
        signal_length : int
            Number of samples in the signal
        coordinates : CoordinateArray, optional
            Coordinate array used to generate the signal.  If the last dimension
            of coordinates is not shape 1, then a new axis will be added to make
            it shape 1.  The shape of the resulting TimeHistoryArray will be
            determined by the shape of the input coordinates.
        pulse_width : float, optional
            With of the pulse in the same units as `dt`. The default is 5*dt.
        pulse_time : float, optional
            The time of the pulse's peak occurance in the same units as `dt`.
            The default is 5*dt.
        pulse_peak : float, optional
            The peak amplitude of the pulse. The default is 1.
        frames : int, optional
            Number of frames to generate.  These will essentially be repeats of
            the first frame for the number of frames specified. The default is 1.
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
        TimeHistoryArray :
            A time history containing the specified haversine pulse signal

        """
        if np.array(coordinates).dtype.type is np.str_:
            coordinates = coordinate_array(string_array=coordinates)
        # Get coordinate size
        if coordinates.ndim == 0 or coordinates.shape[-1] != 1:
            coordinates = coordinates[..., np.newaxis]
        abscissa_frame = np.arange(signal_length) * dt
        ordinate = np.zeros(coordinates.shape[:-1]+(signal_length,))
        if pulse_time is None:
            pulse_time = dt*5
        if pulse_width is None:
            pulse_width = dt*5
        pulse_time, pulse_width, pulse_peak = np.broadcast_arrays(pulse_time, pulse_width, pulse_peak)
        for time, width, peak in zip(pulse_time.flatten(), pulse_width.flatten(), pulse_peak.flatten()):
            period = width
            argument = 2 * np.pi / period * (abscissa_frame - time)
            ordinate += peak/2 * (1+np.cos(argument)) * (np.abs(argument) <= (np.pi))
        # Create the measurement frames
        ordinate = np.tile(ordinate, frames)
        abscissa = dt * np.arange(signal_length*frames)
        # Now create the object
        time_history = data_array(FunctionTypes.TIME_RESPONSE,
                                  abscissa,
                                  ordinate,
                                  coordinates,
                                  comment1, comment2, comment3, comment4, comment5)
        return time_history

    @classmethod
    def sine_sweep_signal(cls, dt, coordinates,
                         frequency_breakpoints, sweep_types, sweep_rates,
                         amplitudes = 1, phases = 1,
                         comment1='', comment2='', comment3='',
                         comment4='', comment5=''):
        frequency_breakpoints = np.array(frequency_breakpoints)
        broadcast_to_shape = coordinates.shape + (frequency_breakpoints.size,)
        full_amplitudes = np.broadcast_to(amplitudes,broadcast_to_shape)
        full_phases = np.broadcast_to(phases,broadcast_to_shape)
        full_sweep_types = np.broadcast_to(sweep_types, frequency_breakpoints.size-1)
        full_sweep_rates = np.broadcast_to(sweep_rates, frequency_breakpoints.size-1)
        output_signals = []
        for key,coordinate in coordinates.ndenumerate():
            output_signals.append(
                sine_sweep(
                    dt, frequency_breakpoints, full_sweep_rates, full_sweep_types,
                    full_amplitudes[key],full_phases[key]))
        output_signals = np.array(output_signals)
        abscissa = dt*np.arange(output_signals.shape[-1])
        # Create a time history array
        time_history = data_array(FunctionTypes.TIME_RESPONSE,
                                  abscissa,output_signals,coordinates.flatten()[:,np.newaxis],
                                  comment1, comment2, comment3, comment4, comment5)
        return time_history.reshape(coordinates.shape)
        
        

def time_history_array(abscissa,ordinate,coordinate,comment1='',comment2='',comment3='',comment4='',comment5=''):
    """
    Helper function to create a TimeHistoryArray object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the TimeHistoryArray object.

    Parameters
    ----------
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
    obj : TimeHistoryArray
        The constructed TimeHistoryArray object
    """
    return data_array(FunctionTypes.TIME_RESPONSE,abscissa,ordinate,coordinate,
                      comment1,comment2,comment3,comment4,comment5)


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

    def ifft(self, norm="backward", rtol=1, atol=1e-8, odd_num_samples = False,
             **scipy_irfft_kwargs):
        """
        Computes a time signal from the frequency spectrum

        Parameters
        ----------
        norm : str, optional
            The type of normalization applied to the fft computation.
            The default is "backward".
        rtol : float, optional
            Relative tolerance used in the abcsissa spacing check.
            The default is 1e-5.
        atol : float, optional
            Relative tolerance used in the abscissa spacing check.
            The default is 1e-8.
        odd_num_samples : bool, optional
            If True, then it is assumed that the output signal has an odd
            number of samples, meaning the signal will have a length of 
            2*(m-1)+1 where m is the number of frequency lines.  Otherwise, the
            default value of 2*(m-1) is used, assuming an even signal.  This is
            ignored if num_samples is specified.
        scipy_irfft_kwargs :
            Additional keywords that will be passed to SciPy's irfft function.

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


        Notes
        -----
        Note that the ifft uses the rfft function from scipy to compute the
        inverse fast fourier transform.  This function is not round-trip
        equivalent for odd functions, because by default it assumes an even
        signal length.  For an odd signal length, the user must either specify
        odd_num_samples = True or set num_samples to the correct number of
        samples.
        """
        
        df = self.abscissa_spacing
        min_freq = self.abscissa.min()
        if min_freq % df > 0.01*df:
            raise ValueError('Frequency bins do not line up with zero.  Cannot compute rfft bins.')
        first_frequency_bin = int(np.round(self.abscissa.min()/df))
        padding = np.zeros(self.ordinate.shape[:-1]+(first_frequency_bin,),self.ordinate.dtype)
        if padding.shape[-1] > 0:
            warnings.warn(
                'The FRFs are missing some low frequency data'
                + ' and it is assumed that this is due to some high pass cut-off.'
                + ' The data is being zero padded at low frequencies.')
        num_elements = first_frequency_bin+self.num_elements
        
        if odd_num_samples:
            num_samples = 2*(num_elements-1)+1
        else:
            num_samples = 2*(num_elements-1)

        # Organizing the FRFs for the ifft, this handles the zero padding if low frequency
        # data is missing
        ordinate = np.concatenate((padding,self.ordinate),axis=-1)
        irfft = scipyfft.irfft(ordinate, axis=-1, n=num_samples, norm=norm,
                               **scipy_irfft_kwargs)

        # Building the time vectors
        dt = 1 / (self.abscissa.max()*num_samples/np.floor(num_samples/2))
        time_vector = dt * np.arange(num_samples)
        
        return data_array(FunctionTypes.TIME_RESPONSE, time_vector, irfft, self.coordinate,
                          self.comment1, self.comment2, self.comment3, self.comment4, self.comment5)

    def interpolate_by_zero_pad(self, time_response_padded_length,
                                return_time_response=False,
                                odd_num_samples = False):
        """
        Interpolates a spectrum by zero padding or truncating its
        time response

        Parameters
        ----------
        time_response_padded_length : int
            Length of the final zero-padded time response
        return_time_response : bool, optional
            If True, the zero-padded impulse response function will be returned.
            If False, it will be transformed back to a transfer function prior
            to being returned.
        odd_num_samples : bool, optional
            If True, then it is assumed that the spectrum has been constructed
            from a signal with an odd number of samples.  Note that this
            function uses the rfft function from scipy to compute the
            inverse fast fourier transform.  The irfft function is not round-trip
            equivalent for odd functions, because by default it assumes an even
            signal length.  For an odd signal length, the user must either specify
            odd_num_samples = True to make it round-trip equivalent.

        Returns
        -------
        SpectrumArray or TimeHistoryArray:
            Spectrum array with appropriately spaced abscissa

        Notes
        -----
        This function will automatically set the last frequency line of the
        SpectrumArray to zero because it won't be accurate anyway.
        If `time_response_padded_length` is less than the current function's
        `num_elements`, then it will be truncated instead of zero-padded.
        """
        time_response = self.ifft(odd_num_samples=odd_num_samples)
        if time_response_padded_length < time_response.num_elements:
            time_response = time_response.idx_by_el[:time_response_padded_length]
        else:
            time_response = time_response.zero_pad(
                time_response_padded_length - time_response.num_elements)
        if return_time_response:
            return time_response
        else:
            spectrum = time_response.fft()
            if time_response_padded_length % 2 == 0:
                spectrum.ordinate[..., -1] = 0
            return spectrum

    def apply_transformation(self, transformation, invert_transformation=False):
        """
        Applies response transformations spectra.

        Parameters
        ----------
        transformation : Matrix
            The transformation to apply to the spectra. It should be a 
            SDynPy matrix object with the "transformed" coordinates on the 
            rows and the "physical" coordinates on the columns. The matrix 
            can be either 2D or 3D (for a frequency dependent transform).
        invert_reference_transformation : bool, optional
            Whether or not to invert the transformation when applying it to 
            the spectra. The default is False, which is standard practice. 
            The row/column ordering in the transformation should be flipped 
            if this is set to true.

        Raises
        ------
        ValueError
            If the physical degrees of freedom in the transformation does not 
            match the spectra.
        
        Returns
        -------
        transformed_spectra : SpectrumArray
            The spectra with the transformation applied.
        """
        if not self.validate_common_abscissa():
            raise ValueError('The abscissa must be consistent accross all functions in the NDDataArray')

        physical_coordinate = np.unique(self.response_coordinate)
        original_spectra_ordinate = np.moveaxis(self[physical_coordinate[...,np.newaxis]].ordinate, -1, 0)[..., np.newaxis]

        if invert_transformation:
            if not np.all(np.unique(transformation.row_coordinate) == physical_coordinate):
                raise ValueError('The physical coordinates in the transformation do no match the spectra')
            transformed_coordinate = np.unique(transformation.column_coordinate)
            transformation_matrix = np.linalg.pinv(transformation[physical_coordinate, transformed_coordinate])
        elif not invert_transformation:
            if not np.all(np.unique(transformation.column_coordinate) == physical_coordinate):
                raise ValueError('The physical coordinates in the transformation do no match the spectra')
            transformed_coordinate = np.unique(transformation.row_coordinate)
            transformation_matrix = transformation[transformed_coordinate, physical_coordinate]
                
        transformed_spectra_ordinate = (transformation_matrix @ original_spectra_ordinate)[...,0]

        return data_array(FunctionTypes.SPECTRUM, self.ravel().abscissa[0], np.moveaxis(transformed_spectra_ordinate, 0, -1), 
                          transformed_coordinate[...,np.newaxis])

    def plot(self, one_axis=True, subplots_kwargs={}, plot_kwargs={},
             abscissa_markers = None, 
             abscissa_marker_labels = None, abscissa_marker_type = 'vline',
             abscissa_marker_plot_kwargs = {}):
        """
        Plot the spectra

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
        abscissa_markers : ndarray, optional
            Array containing abscissa values to mark on the plot to denote
            significant events.
        abscissa_marker_labels : str or ndarray
            Array of strings to label the abscissa_markers with, or
            alternatively a format string that accepts index and abscissa
            inputs (e.g. '{index:}: {abscissa:0.2f}').  By default no label
            will be applied.
        abscissa_marker_type : str
            The type of marker to use.  This can either be the string 'vline'
            or a valid matplotlib symbol specifier (e.g. 'o', 'x', '.').
        abscissa_marker_plot_kwargs : dict
            Additional keyword arguments used when plotting the abscissa label
            markers.

        Returns
        -------
        axis : matplotlib axis or array of axes
             On which the data were plotted

        """
        if abscissa_markers is not None:
            if abscissa_marker_labels is None:
                abscissa_marker_labels = ['' for value in abscissa_markers]
            elif isinstance(abscissa_marker_labels,str):
                abscissa_marker_labels = [abscissa_marker_labels.format(
                    index = i, abscissa = v) for i,v in enumerate(abscissa_markers)]
                
        if one_axis is True:
            figure, axis = plt.subplots(2, 1, **subplots_kwargs)
            lines = axis[0].plot(self.flatten().abscissa.T, np.angle(
                self.flatten().ordinate.T), **plot_kwargs)
            axis[0].set_ylabel('Phase')
            if abscissa_markers is not None:
                if abscissa_marker_type == 'vline':
                    kwargs = {'color':'k'}
                    kwargs.update(abscissa_marker_plot_kwargs)
                    for value,label in zip(abscissa_markers,abscissa_marker_labels):
                        axis[0].axvline(value, **kwargs)
                        axis[0].annotate(label, xy = (value, axis[0].get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                    axis.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                else:
                    for line in lines:
                        x = line.get_xdata()
                        y = line.get_ydata()
                        marker_y = np.interp(abscissa_markers, x, y)
                        kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        axis[0].plot(abscissa_markers,marker_y,**kwargs)
                        for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                            axis[0].annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
            lines = axis[1].plot(self.flatten().abscissa.T, np.abs(
                self.flatten().ordinate.T), **plot_kwargs)
            axis[1].set_yscale('log')
            axis[1].set_ylabel('Amplitude')
            if abscissa_markers is not None:
                if abscissa_marker_type == 'vline':
                    kwargs = {'color':'k'}
                    kwargs.update(abscissa_marker_plot_kwargs)
                    for value,label in zip(abscissa_markers,abscissa_marker_labels):
                        axis[1].axvline(value, **kwargs)
                        axis[1].annotate(label, xy = (value, axis[1].get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                    axis[1].callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                else:
                    for line in lines:
                        x = line.get_xdata()
                        y = line.get_ydata()
                        marker_y = np.interp(abscissa_markers, x, y)
                        kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        axis[1].plot(abscissa_markers,marker_y,**kwargs)
                        for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                            axis[1].annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
        elif one_axis is False:
            ncols = int(np.floor(np.sqrt(self.size)))
            nrows = int(np.ceil(self.size / ncols))
            figure, axis = plt.subplots(nrows, ncols, **subplots_kwargs)
            for i, (ax, (index, function)) in enumerate(zip(axis.flatten(), self.ndenumerate())):
                lines = ax.plot(function.abscissa.T, np.abs(function.ordinate.T), **plot_kwargs)
                ax.set_ylabel('/'.join([str(v) for i, v in function.coordinate.ndenumerate()]))
                ax.set_yscale('log')
                if abscissa_markers is not None:
                    if abscissa_marker_type == 'vline':
                        kwargs = {'color':'k'}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        for value,label in zip(abscissa_markers,abscissa_marker_labels):
                            ax.axvline(value, **kwargs)
                            ax.annotate(label, xy = (value, ax.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                        ax.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                    else:
                        for line in lines:
                            x = line.get_xdata()
                            y = line.get_ydata()
                            marker_y = np.interp(abscissa_markers, x, y)
                            kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                            kwargs.update(abscissa_marker_plot_kwargs)
                            ax.plot(abscissa_markers,marker_y,**kwargs)
                            for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                                ax.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
            for ax in axis.flatten()[i + 1:]:
                ax.remove()
        else:
            axis = one_axis
            lines = axis.plot(self.flatten().abscissa.T, np.abs(self.flatten().ordinate.T), **plot_kwargs)
            if abscissa_markers is not None:
                if abscissa_marker_type == 'vline':
                    kwargs = {'color':'k'}
                    kwargs.update(abscissa_marker_plot_kwargs)
                    for value,label in zip(abscissa_markers,abscissa_marker_labels):
                        axis.axvline(value, **kwargs)
                        axis.annotate(label, xy = (value, axis.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                    axis.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                else:
                    for line in lines:
                        x = line.get_xdata()
                        y = line.get_ydata()
                        marker_y = np.interp(abscissa_markers, x, y)
                        kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        axis.plot(abscissa_markers,marker_y,**kwargs)
                        for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                            axis.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
        return axis

    def plot_spectrogram(self, abscissa=None, axis=None,
                         subplots_kwargs={},
                         pcolormesh_kwargs={'shading': 'auto'},
                         log_scale=True):
        """
        Plots a spectrogram

        Parameters
        ----------
        abscissa : np.ndarray
            Optional argument to specify as the abscissa values.  If not
            specified, this will be the index of the flattened SpectrumArray.
        axis : matplotlib.axis, optional
            An optional argument that specifies the axis to plot the spectrogram
            on
        subplots_kwargs : dict, optional
            Optional keywords to specify to the subplots function that creates
            a new figure if `axis` is not specified.
        pcolormesh_kwargs : dict, optional
            Optional arguments to pass to the pcolormesh function
        log_scale : bool
            If True, the colormap will be applied logarithmically

        Returns
        -------
        ax : matplotlib.axis
            The axis on which the spectrogram was plotted
        """
        # Make sure the abscissa are common
        self.validate_common_abscissa()
        flat_self = self.flatten()
        data = abs(self.ordinate)
        if log_scale:
            data = np.log(data)
        y_coords = flat_self[0].abscissa
        if abscissa is None:
            abscissa = np.arange(self.size)
        if axis is None:
            fig, axis = plt.subplots(**subplots_kwargs)
        axis.pcolormesh(abscissa, y_coords, data.T, **pcolormesh_kwargs)
        return axis


def spectrum_array(abscissa,ordinate,coordinate,comment1='',comment2='',
                   comment3='',comment4='',comment5=''):
    """
    Helper function to create a SpectrumArray object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the SpectrumArray object.

    Parameters
    ----------
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
    obj : SpectrumArray
        The constructed SpectrumArray object
    """
    return data_array(FunctionTypes.SPECTRUM,abscissa,ordinate,coordinate,
                      comment1,comment2,comment3,comment4,comment5)


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

    @staticmethod
    def from_time_data(response_data: TimeHistoryArray,
                       samples_per_average: int = None,
                       overlap: float = 0.0,
                       window=np.array((1.0,)), 
                       reference_data: TimeHistoryArray = None,
                       only_asds = False):
        """
        Computes a PSD matrix from reference and response time histories

        Parameters
        ----------
        response_data : TimeHistoryArray
            Time data to be used as responses
        samples_per_average : int, optional
            Number of samples used to split up the signals into averages.  The
            default is None, meaning the data is treated as a single measurement
            frame.
        overlap : float, optional
            The overlap as a fraction of the frame (e.g. 0.5 specifies 50% overlap).
            The default is 0.0, meaning no overlap is used.
        window : np.ndarray or str, optional
            A 1D ndarray with length samples_per_average that specifies the
            coefficients of the window.  A Hann window is applied if not specified.
            If a string is specified, then the window will be obtained from scipy.
        reference_data : TimeHistoryArray
            Time data to be used as reference.  If not specified, the response
            data will be used as references, resulting in a square CPSD matrix.

        Raises
        ------
        ValueError
            Raised if reference and response functions do not have consistent
            abscissa

        Returns
        -------
        PowerSpectralDensityArray
            A PSD array computed from the specified reference and
            response signals.

        """
        if reference_data is None:
            reference_data = response_data
            ref_ord = None
            ref_data = reference_data.flatten()
        elif only_asds:
            raise ValueError('`only_asds` cannot be true when reference data is '
                             'specified')
        else:
            ref_data = reference_data.flatten()
            ref_ord = ref_data.ordinate
        res_data = response_data.flatten()
        res_ord = res_data.ordinate
        if ((not np.allclose(ref_data[0].abscissa,
                           res_data[0].abscissa))
            or (not np.allclose(ref_data.abscissa_spacing,res_data.abscissa_spacing))):
            raise ValueError('Reference and Response Data should have identical abscissa!')
        dt = res_data.abscissa_spacing
        df, cpsd = sp_cpsd(res_ord, 1/dt, samples_per_average, overlap,
                           window, reference_signals = ref_ord,only_asds = only_asds)
        freq = np.arange(cpsd.shape[0])*df
        # Now construct the transfer function array
        if only_asds:
            coordinate = np.concatenate((res_data.coordinate,
                                         ref_data.coordinate),axis=-1)
        else:
            coordinate = outer_product(res_data.coordinate.flatten(),
                                       ref_data.coordinate.flatten())
        return data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                          freq, np.moveaxis(cpsd, 0, -1), coordinate)

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
        window_function = sig.windows.hann(2 * (self.num_elements - 1)*output_oversample, sym=False)**0.5
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
        if not np.allclose(abscissa, transfer_function.abscissa):
            raise ValueError('Transfer Function Abscissa do not match CPSD')
        if not np.allclose(abscissa, self.abscissa):
            raise ValueError('All CPSD abscissa must be identical')
        # First do bookkeeping, we want to get the coordinates of the response
        # of the FRF corresponding to the specification matrix
        transfer_function = transfer_function.reshape_to_matrix()
        response_dofs = transfer_function[:, 0].response_coordinate
        reference_dofs = transfer_function[0, :].reference_coordinate
        cpsd_dofs = outer_product(reference_dofs, reference_dofs)
        output_dofs = outer_product(response_dofs, response_dofs)
        frf_matrix = np.moveaxis(transfer_function.ordinate, -1, 0)
        cpsd_matrix = np.moveaxis(self[cpsd_dofs].ordinate, -1, 0)
        output_matrix = frf_matrix @ cpsd_matrix @ np.moveaxis(frf_matrix.conj(), -1, -2)
        return data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                          abscissa, np.moveaxis(output_matrix, 0, -1), output_dofs)

    def mimo_inverse(self, transfer_function,
                     method='standard',
                     response_weighting_matrix=None,
                     reference_weighting_matrix=None,
                     regularization_weighting_matrix=None,
                     regularization_parameter=None,
                     cond_num_threshold=None,
                     num_retained_values=None):
        """
        Computes input estimation for MIMO random vibration problems

        Parameters
        ----------
        transfer_function : TransferFunctionArray
            System transfer functions used to estimate the input from the given
            response matrix
        method : str, optional
            The method to be used for the FRF matrix inversions. The available
            methods are:
                - standard - basic pseudo-inverse via numpy.linalg.pinv with the
                  default rcond parameter, this is the default method
                - threshold - pseudo-inverse via numpy.linalg.pinv with a specified
                  condition number threshold
                - tikhonov - pseudo-inverse using the Tikhonov regularization method
                - truncation - pseudo-inverse where a fixed number of singular values
                  are retained for the inverse
        response_weighting_matrix : sdpy.Matrix, optional
            Diagonal matrix used to weight response degrees of freedom (to solve the
            problem as a weight least squares) by multiplying the rows of the FRF
            matrix by a scalar weights. This matrix can also be a 3D matrix such that
            the the weights are different for each frequency line. The matrix should
            be sized [number of lines, number of references, number of references],
            where the number of lines either be one (the same weights at all frequencies)
            or the length of the abscissa (for the case where a 3D matrix is supplied).
        reference_weighting_matrix : sdpy.Matrix, optional
            Diagonal matrix used to weight reference degrees of freedom (generally for
            normalization) by multiplying the columns of the FRF matrix by a scalar weights.
            This matrix can also be a 3D matrix such that the the weights are different
            for each frequency line. The matrix should be sized
            [number of lines, number of references, number of references], where the number
            of lines either be one (the same weights at all frequencies) or the length
            of the abscissa (for the case where a 3D matrix is supplied).
        regularization_weighting_matrix : sdpy.Matrix, optional
            Matrix used to weight input degrees of freedom via Tikhonov regularization.
            This matrix can also be a 3D matrix such that the the weights are different
            for each frequency line. The matrix should be sized
            [number of lines, number of references, number of references], where the number
            of lines either be one (the same weights at all frequencies) or the length
            of the abscissa (for the case where a 3D matrix is supplied).
        regularization_parameter : float or np.ndarray, optional
            Scaling parameter used on the regularization weighting matrix when the tikhonov
            method is chosen. A vector of regularization parameters can be provided so the
            regularization is different at each frequency line. The vector must match the
            length of the abscissa in this case (either be size [num_lines,] or [num_lines, 1]).
        cond_num_threshold : float or np.ndarray, optional
            Condition number used for SVD truncation when the threshold method is chosen.
            A vector of condition numbers can be provided so it varies as a function of
            frequency. The vector must match the length of the abscissa in this case.
        num_retained_values : float or np.ndarray, optional
            Number of singular values to retain in the pseudo-inverse when the truncation
            method is chosen. A vector of can be provided so the number of retained values
            can change as a function of frequency. The vector must match the length of the
            abscissa in this case.

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
        This function solves the MIMO problem Gxx = Hxv@Gvv@Hxv^* using the pseudoinverse.
        Gvv = Hxv^+@Gxx@Hxv^+^*, where Gvv is the source.

        References
        ----------
        .. [1] Wikipedia, "Moore-Penrose inverse".
               https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
        .. [2] A.N. Tithe, D.J. Thompson, The quantification of structure-borne transmission pathsby inverse methods. Part 2: Use of regularization techniques,
               Journal of Sound and Vibration, Volume 264, Issue 2, 2003, Pages 433-451, ISSN 0022-460X,
               https://doi.org/10.1016/S0022-460X(02)01203-8.
        .. [3] Wikipedia, "Ridge regression".
               https://en.wikipedia.org/wiki/Ridge_regression
        """
        # Check consistent abscissa
        abscissa = self.flatten()[0].abscissa
        if not np.allclose(abscissa, transfer_function.abscissa):
            raise ValueError('Transfer Function Abscissa do not match CPSD')
        if not np.allclose(abscissa, self.abscissa):
            raise ValueError('All CPSD abscissa must be identical')
        # First do bookkeeping, we want to get the coordinates of the response
        # of the FRF corresponding to the specification matrix
        transfer_function = transfer_function.reshape_to_matrix()
        response_dofs = transfer_function[:, 0].response_coordinate
        reference_dofs = transfer_function[0, :].reference_coordinate
        cpsd_dofs = outer_product(response_dofs, response_dofs)
        output_dofs = outer_product(reference_dofs, reference_dofs)
        frf_matrix = np.moveaxis(transfer_function.ordinate, -1, 0)
        cpsd_matrix = np.moveaxis(self[cpsd_dofs].ordinate.copy(), -1, 0)
        # Perform the generalized inversion
        if response_weighting_matrix is not None:
            if isinstance(response_weighting_matrix, Matrix):
                response_weighting_matrix = response_weighting_matrix[response_dofs, response_dofs]
            cpsd_matrix = response_weighting_matrix @ cpsd_matrix @ np.moveaxis(response_weighting_matrix.conj(), -1, -2)
        if reference_weighting_matrix is not None:
            if isinstance(reference_weighting_matrix, Matrix):
                reference_weighting_matrix = reference_weighting_matrix[reference_dofs, reference_dofs]
        if isinstance(regularization_weighting_matrix, Matrix):
            regularization_weighting_matrix = regularization_weighting_matrix[reference_dofs, reference_dofs]
        frf_pinv = frf_inverse(frf_matrix,
                               method=method,
                               response_weighting_matrix=response_weighting_matrix,
                               reference_weighting_matrix=reference_weighting_matrix,
                               regularization_weighting_matrix=regularization_weighting_matrix,
                               regularization_parameter=regularization_parameter,
                               cond_num_threshold=cond_num_threshold,
                               num_retained_values=num_retained_values)
        method_statement_start = 'The inputs are being computed using the '
        method_statement_end = ' method'
        print(method_statement_start+method+method_statement_end)
        output_matrix = frf_pinv @ cpsd_matrix @ np.moveaxis(frf_pinv.conj(), -1, -2)
        if reference_weighting_matrix is not None:
            output_matrix = reference_weighting_matrix@output_matrix@reference_weighting_matrix
        return data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                          abscissa, np.moveaxis(output_matrix, 0, -1), output_dofs)

    def error_summary(self, figure_kwargs={}, linewidth=1, plot_kwargs={}, **cpsd_matrices):
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
        for legend, cpsd in cpsd_matrices.items():
            if not np.allclose(frequencies, cpsd.abscissa):
                raise ValueError('Compared CPSD abscissa do not match')
        if not np.allclose(frequencies, self.abscissa):
            raise ValueError('All CPSD abscissa must be identical')
        # Get ASDs
        responses = np.unique(abs(self.coordinate))
        response_dofs = np.tile(responses[:, np.newaxis], 2)
        channel_names = responses.string_array()
        spec_asd = np.real(self[response_dofs].ordinate)
        data_asd = {legend: np.real(data[response_dofs].ordinate) for legend, data in cpsd_matrices.items()}
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
            colors = prop_cycle.by_key()['color'] * 100
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

    def svd(self, full_matrices=True, compute_uv=True, as_matrix=True):
        """
        Compute the SVD of the provided CPSD matrix

        Parameters
        ----------
        full_matrices : bool, optional
            This is an optional input for np.linalg.svd
        compute_uv : bool, optional
            This is an optional input for np.linalg.svd
        as_matrix : bool, optional
            If True, matrices are returned as a SDynPy Matrix class with named
            rows and columns.  Otherwise, a simple numpy array is returned

        Returns
        -------
        u : ndarray
            Left hand singular vectors, sized [..., num_responses, num_responses].
            Only returned when compute_uv is True.
        s : ndarray
            Singular values, sized [..., num_references]
        vh : ndarray
            Right hand singular vectors, sized [..., num_references, num_references].
            Only returned when compute_uv is True.
        """
        cpsd = self.reshape_to_matrix()
        cpsdOrd = np.moveaxis(cpsd.ordinate, -1, 0)
        if compute_uv:
            u, s, vh = np.linalg.svd(cpsdOrd, full_matrices, compute_uv)
            if as_matrix:
                u = matrix(u, cpsd[:, 0].response_coordinate,
                           coordinate_array(np.arange(u.shape[-1])+1, 0))
                s = matrix(s[:, np.newaxis]*np.eye(s.shape[-1]), coordinate_array(np.arange(s.shape[-1])+1, 0),
                           coordinate_array(np.arange(s.shape[-1])+1, 0))
                vh = matrix(vh, coordinate_array(np.arange(vh.shape[-2])+1, 0),
                            cpsd[0, :].reference_coordinate,
                            )
            return u, s, vh
        else:
            s = np.linalg.svd(cpsdOrd, full_matrices, compute_uv)
            if as_matrix:
                s = matrix(s[:, np.newaxis]*np.eye(s.shape[-1]), coordinate_array(np.arange(s.shape[-1])+1, 0),
                           coordinate_array(np.arange(s.shape[-1])+1, 0))
            return s

    def get_asd(self):
        """
        Get functions where the response coordinate is equal to the reference coordinate

        Returns
        -------
        PowerSpectralDensityArray
            PowerSpectralDensityArrays where the response is equal to the reference

        """
        indices = np.where(abs(self.coordinate[..., 0]) == abs(self.coordinate[..., 1]))
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
        abscissa_spacing = self.abscissa_spacing
        return np.sqrt(np.sum(asd.ordinate.real, axis=-1)*abscissa_spacing)

    def plot_asds(self, figure_kwargs={}, linewidth=1):
        asds = self.get_asd()
        try:
            rms = asds.rms()
        except ValueError:
            rms = None
        ax = asds.plot(one_axis=False, subplots_kwargs=figure_kwargs, plot_kwargs={'linewidth': linewidth})
        for i, (a,asd) in enumerate(zip(ax.flatten(),asds)):
            if rms is not None:
                a.set_ylabel(a.get_ylabel()+'\nRMS: {:0.4f}'.format(rms[i]))
            a.set_yscale('log')
        return ax

    @staticmethod
    def compare_asds(figure_kwargs={}, linewidth=1, **cpsd_matrices):
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
        asds = {legend: cpsd.get_asd() for legend, cpsd in cpsd_matrices.items()}
        for i, (legend, asd) in enumerate(asds.items()):
            this_dofs = np.unique(abs(asd.coordinate))
            this_abscissa = asd.abscissa
            if i == 0:
                dofs = this_dofs
            if not np.all(this_dofs == dofs):
                raise ValueError('CPSDs must have identical dofs')
        # Sort the dofs correctly
        asds = {legend: asd[np.tile(dofs[:, np.newaxis], 2)] for legend, asd in asds.items()}
        num_channels = len(dofs)
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
        for i, (legend, asd) in enumerate(asds.items()):
            for j, fn in enumerate(asd):
                rms = np.sqrt(np.sum(fn.ordinate.real)*np.mean(np.diff(fn.abscissa)))
                x = i+(len(asds)+1)*j
                a = ax.bar(x, rms, color=colors[i])
                if j == 0:
                    legend_handles.append(a)
                    legend_strings.append(legend.replace('_', ' '))
                ax.text(x, 0, ' {:.2f}'.format(rms),
                        horizontalalignment='center', verticalalignment='bottom', rotation=90)
        # Set XTicks
        xticks = np.mean(np.arange(len(asds))) + np.arange(len(dofs))*(len(asds)+1)
        xticklabels = [str(dof) for dof in dofs]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel('RMS Levels')
        fig.tight_layout()
        legend = ax.legend(legend_handles, legend_strings, bbox_to_anchor=(1, 1))
        fig.canvas.draw()
        legend_width = legend.get_window_extent().width
        figure_width = fig.bbox.width
        figure_fraction = legend_width/figure_width
        ax_position = ax.get_position()
        ax_position.x1 -= figure_fraction
        ax.set_position(ax_position)

    def plot_singular_values(self, rcond=None, min_freqency=None, max_frequency=None):
        """
        Plot the singular values of an FRF matrix with a visualization of the rcond tolerance

        Parameters
        ----------
        rcond : value of float, optional
            Cutoff for small singular values. Implemented such that the cutoff is rcond*
            largest_singular_value (the same as np.linalg.pinv). This is to visualize the
            effect of rcond and is used for display purposes only.
        min_frequency : float, optional
            Minimum frequency to plot
        max_frequency : float, optional
            Maximum frequency to plot

        """
        freq = self.flatten().abscissa[0, :]
        s_cpsd = self.svd(compute_uv=False, as_matrix=False)
        plt.figure()
        plt.semilogy(freq, s_cpsd)
        if rcond is not None:
            cutoff = s_cpsd[:, 0] * rcond
            plt.semilogy(freq, cutoff, color='k', linestyle='dashed', linewidth=3)
        plt.grid()
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Singular Values')
        plt.title('Singular Values of CPSD Matrix')
        if min_freqency is not None:
            plt.xlim(left=min_freqency)
        if max_frequency is not None:
            plt.xlim(right=max_frequency)

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
        abscissa = reshaped_array[0, 0].abscissa
        if not np.allclose(reshaped_array.abscissa, abscissa):
            raise ValueError('All functions must have identical abscissa')
        cpsd_matrix = np.moveaxis(reshaped_array.ordinate, -1, 0)
        coherence_matrix = np.moveaxis(sp_coherence(cpsd_matrix), 0, -1)
        coherence_array = data_array(
            FunctionTypes.COHERENCE, abscissa=reshaped_array.abscissa,
            ordinate=coherence_matrix, coordinate=reshaped_array.coordinate)
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
        return data_array(FunctionTypes.GENERAL, self.abscissa, np.angle(self.ordinate),
                          self.coordinate)

    def set_coherence_phase(self, coherence_array, angle_array):
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
        dofs = outer_product(asds.response_coordinate, asds.reference_coordinate)
        reshaped_coherence = coherence_array[dofs]
        reshaped_angle = angle_array[dofs]
        asd_matrix = np.moveaxis(asds.ordinate, -1, 0)
        coherence_matrix = np.moveaxis(reshaped_coherence.ordinate, -1, 0)
        phase_matrix = np.moveaxis(reshaped_angle.ordinate, -1, 0)
        cpsd_matrix = cpsd_from_coh_phs(asd_matrix, coherence_matrix, phase_matrix)
        output = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                            asds[0].abscissa, np.moveaxis(cpsd_matrix, 0, -1),
                            dofs)[dofs]
        return output
    
    def get_cpsd_from_asds(self):
        """
        Transforms ASDs to a full CPSD matrix with zeros on the off-diagonals

        Returns
        -------
        output : PowerSpectralDensityArray
            CPSD matrix with the inputs on the diagonals and the off-diagonals
            as zeros.

        """
        asds = self.get_asd()
        dofs = outer_product(asds.response_coordinate, asds.reference_coordinate)
        asd_matrix = np.moveaxis(asds.ordinate, -1, 0)
        coherence_matrix = np.tile(np.eye(dofs.shape[0]),(asd_matrix.shape[0],1,1))
        phase_matrix = 0
        cpsd_matrix = cpsd_from_coh_phs(asd_matrix, coherence_matrix, phase_matrix)
        output = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,
                            asds[0].abscissa, np.moveaxis(cpsd_matrix, 0, -1),
                            dofs)[dofs]
        return output

    @classmethod
    def eye(cls, frequencies, coordinates, rms=None, full_matrix=False,
            breakpoint_frequencies=None, breakpoint_levels=None,
            breakpoint_interpolation='lin',
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
        if max_frequency is not None:
            cpsd[frequencies > max_frequency] = 0

        if rms is not None:
            frequency_spacing = np.mean(np.diff(frequencies))
            if not np.allclose(frequency_spacing, np.diff(frequencies)):
                raise ValueError('In order to specify RMS, the spacing of frequencies must be constant')
            cpsd_rms = np.sqrt(np.sum(cpsd) * frequency_spacing)
            cpsd *= (rms / cpsd_rms)**2

        num_channels = coordinates.size
        if full_matrix:
            full_cpsd = np.zeros((num_channels, num_channels, frequencies.size))
            full_cpsd[np.arange(num_channels), np.arange(num_channels), :] = cpsd
            cpsd_coordinates = outer_product(coordinates, coordinates)
        else:
            full_cpsd = np.tile(cpsd, (num_channels, 1))
            cpsd_coordinates = np.tile(coordinates[:, np.newaxis], (1, 2))

        return data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, frequencies,
                          full_cpsd, cpsd_coordinates)

    def apply_transformation(self, transformation, invert_transformation=False):
        """
        Applies a transformation to a cross power spectral density matrix.

        Parameters
        ----------
        transformation : Matrix
            The transformation to apply to the spectra. It should be a 
            SDynPy matrix object with the "transformed" coordinates on the 
            rows and the "physical" coordinates on the columns. The matrix 
            can be either 2D or 3D (for a frequency dependent transform).
        invert_reference_transformation : bool, optional
            Whether or not to invert the transformation when applying it to 
            the spectra. The default is False, which is standard practice. 
            The row/column ordering in the transformation should be flipped 
            if this is set to true.

        Raises
        ------
        ValueError
            If the cross power spectral density matrix is not square.
        ValueError
            If the physical degrees of freedom in the transformation does not 
            match the spectra.
        
        Returns
        -------
        transformed_spectra : PowerSpectralDensityArray
            The cross power spectral density with the transformation applied.
        """
        if not self.validate_common_abscissa():
            raise ValueError('The abscissa must be consistent accross all functions in the NDDataArray')

        if self.ordinate.size != (np.unique(self.response_coordinate).shape[0]*np.unique(self.reference_coordinate).shape[0]*np.unique(self.abscissa).shape[0]):
            raise ValueError('The supplied array must be a full cross power spectral density matrix')

        physical_coordinate = np.unique(self.response_coordinate)
        original_spectra_ordinate = np.moveaxis(self[outer_product(physical_coordinate, physical_coordinate)].ordinate, -1, 0)

        if invert_transformation:
            if not np.all(np.unique(transformation.row_coordinate) == physical_coordinate):
                raise ValueError('The physical coordinates in the transformation do no match the spectra')
            transformed_coordinate = np.unique(transformation.column_coordinate)
            transformation_matrix = np.linalg.pinv(transformation[physical_coordinate, transformed_coordinate])
        elif not invert_transformation:
            if not np.all(np.unique(transformation.column_coordinate) == physical_coordinate):
                raise ValueError('The physical coordinates in the transformation do no match the spectra')
            transformed_coordinate = np.unique(transformation.row_coordinate)
            transformation_matrix = transformation[transformed_coordinate, physical_coordinate]
        
        if transformation_matrix.ndim == 2:
            transformation_matrix = transformation_matrix[np.newaxis,...] # this ensures the transpose in the next line works

        transformed_spectra_ordinate = transformation_matrix @ original_spectra_ordinate @ np.transpose(transformation_matrix.conj(), (0, 2, 1))

        return data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, self.ravel().abscissa[0], np.moveaxis(transformed_spectra_ordinate, 0, -1), 
                          outer_product(transformed_coordinate, transformed_coordinate))

    def plot_magnitude_coherence_phase(self, compare_data=None, plot_axes=False,
                                       sharex=True, sharey=True, logx=False,
                                       logy=True,
                                       magnitude_plot_kwargs={},
                                       coherence_plot_kwargs={},
                                       angle_plot_kwargs={},
                                       figure_kwargs={}):
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
        gs = GridSpec(*reshaped_array.shape, fig,
                      wspace=None if plot_axes else 0,
                      hspace=None if plot_axes else 0)
        for (i, j), function in reshaped_array.ndenumerate():
            if i == j:
                if ((not sharex) and (not sharey)) or i == 0:
                    ax[i, j] = fig.add_subplot(gs[i, j])
                elif (not sharex) and sharey and (i > 0):
                    ax[i, j] = fig.add_subplot(gs[i, j], sharey=ax[0, 0])
                elif sharex and (not sharey) and (i > 0):
                    ax[i, j] = fig.add_subplot(gs[i, j], sharex=ax[0, 0])
                else:
                    ax[i, j] = fig.add_subplot(gs[i, j], sharex=ax[0, 0],
                                               sharey=ax[0, 0])
                ax[i, j].plot(function.abscissa, np.abs(function.ordinate),
                              'r', **magnitude_plot_kwargs)
                if compare_data is not None:
                    ax[i, j].plot(reshaped_compare_data[i, j].abscissa,
                                  np.abs(reshaped_compare_data[i, j].ordinate),
                                  color=[1.0, .5, .5], **magnitude_plot_kwargs)
                if logy:
                    ax[i, j].set_yscale('log')
            if i > j:
                ax[i, j] = fig.add_subplot(
                    gs[i, j],
                    sharex=ax[0, 0] if (i > 0) and sharex else None,
                    sharey=ax[1, 0] if (i > 1) and sharey else None)
                ax[i, j].plot(function.abscissa, phase[i, j].ordinate,
                              'g', **angle_plot_kwargs)
                if compare_data is not None:
                    ax[i, j].plot(phase_compare[i, j].abscissa,
                                  phase_compare[i, j].ordinate,
                                  color=[0, 1, 0], **angle_plot_kwargs)
                ax[i, j].set_ylim(-np.pi, np.pi)
            if i < j:
                ax[i, j] = fig.add_subplot(
                    gs[i, j],
                    sharex=ax[0, 0] if (j > 0) and sharex else None,
                    sharey=ax[0, 1] if (j > 1) and sharey else None)
                ax[i, j].plot(function.abscissa, coherence[i, j].ordinate,
                              'b', **coherence_plot_kwargs)
                if compare_data is not None:
                    ax[i, j].plot(compare_coherence[i, j].abscissa,
                                  compare_coherence[i, j].ordinate,
                                  color=[0.5, 0.5, 1.0], **coherence_plot_kwargs)
                ax[i, j].set_ylim(0, 1)
            if logx:
                ax[i, j].set_xscale('log')
            if j == 0:
                ax[i, j].set_ylabel(str(function.response_coordinate))
            if i == reshaped_array.shape[0]-1:
                ax[i, j].set_xlabel(str(function.reference_coordinate))
            if not plot_axes:
                ax[i, j].set_yticklabels([])
                ax[i, j].set_xticklabels([])
                ax[i, j].tick_params(axis='x', direction='in')
                ax[i, j].tick_params(axis='y', direction='in')

    def to_rattlesnake_specification(self, filename=None,
                                     coordinate_order=None,
                                     min_frequency=None,
                                     max_frequency=None,
                                     upper_warning_db=None,
                                     lower_warning_db=None,
                                     upper_abort_db=None,
                                     lower_abort_db=None,
                                     upper_warning_psd=None,
                                     lower_warning_psd=None,
                                     upper_abort_psd=None,
                                     lower_abort_psd=None):
        if coordinate_order is not None:
            coordinate_array = outer_product(coordinate_order)
            reshaped_data = self[coordinate_array]
        else:
            if self.ndim != 2:
                raise ValueError('CPSD Matrix must be 2D to transform to rattlesnake specification')
            if self.shape[0] != self.shape[1]:
                raise ValueError('CPSD Matrix must be square')
            if not np.all(self.coordinate[..., 0] == self.coordinate[..., 1].T):
                raise ValueError('Row and column coordinates of the CPSD matrix are not ordered identically')
            reshaped_data = self
            coordinate_array = reshaped_data.coordinate
        if min_frequency is not None or max_frequency is not None:
            if min_frequency is None:
                min_frequency = -np.inf
            if max_frequency is None:
                max_frequency = np.inf
            reshaped_data = reshaped_data.extract_elements_by_abscissa(min_frequency, max_frequency)
        out_dict = dict(
            f=reshaped_data[0, 0].abscissa,
            cpsd=np.moveaxis(reshaped_data.ordinate, -1, 0))
        if upper_warning_db is not None:
            out_dict['warning_upper'] = np.einsum('ijj->ij', out_dict['cpsd']*db2scale(upper_warning_db)**2).real
        if lower_warning_db is not None:
            out_dict['warning_lower'] = np.einsum('ijj->ij', out_dict['cpsd']*db2scale(lower_warning_db)**2).real
        if upper_abort_db is not None:
            out_dict['abort_upper'] = np.einsum('ijj->ij', out_dict['cpsd']*db2scale(upper_abort_db)**2).real
        if lower_abort_db is not None:
            out_dict['abort_lower'] = np.einsum('ijj->ij', out_dict['cpsd']*db2scale(lower_abort_db)**2).real
        if upper_warning_psd is not None:
            signal = upper_warning_psd
            reshaped_signal = signal[np.einsum('iij->ij', coordinate_array)]
            if min_frequency is not None or max_frequency is not None:
                reshaped_signal = reshaped_signal.extract_elements_by_abscissa(min_frequency, max_frequency)
            if np.any(np.einsum('jji->ji', reshaped_data.abscissa) != reshaped_signal.abscissa):
                raise ValueError('Abscissa specified by upper warning signal is not equal to the specification signal')
            out_dict['warning_upper'] = np.moveaxis(reshaped_signal.ordinate, -1, 0).real
        if lower_warning_psd is not None:
            signal = lower_warning_psd
            reshaped_signal = signal[np.einsum('iij->ij', coordinate_array)]
            if min_frequency is not None or max_frequency is not None:
                reshaped_signal = reshaped_signal.extract_elements_by_abscissa(min_frequency, max_frequency)
            if np.any(np.einsum('jji->ji', reshaped_data.abscissa) != reshaped_signal.abscissa):
                raise ValueError('Abscissa specified by lower warning signal is not equal to the specification signal')
            out_dict['warning_lower'] = np.moveaxis(reshaped_signal.ordinate, -1, 0).real
        if upper_abort_psd is not None:
            signal = upper_abort_psd
            reshaped_signal = signal[np.einsum('iij->ij', coordinate_array)]
            if min_frequency is not None or max_frequency is not None:
                reshaped_signal = reshaped_signal.extract_elements_by_abscissa(min_frequency, max_frequency)
            if np.any(np.einsum('jji->ji', reshaped_data.abscissa) != reshaped_signal.abscissa):
                raise ValueError('Abscissa specified by upper abort signal is not equal to the specification signal')
            out_dict['abort_upper'] = np.moveaxis(reshaped_signal.ordinate, -1, 0).real
        if lower_abort_psd is not None:
            signal = lower_abort_psd
            reshaped_signal = signal[np.einsum('iij->ij', coordinate_array)]
            if min_frequency is not None or max_frequency is not None:
                reshaped_signal = reshaped_signal.extract_elements_by_abscissa(min_frequency, max_frequency)
            if np.any(np.einsum('jji->ji', reshaped_data.abscissa) != reshaped_signal.abscissa):
                raise ValueError('Abscissa specified by lower abort signal is not equal to the specification signal')
            out_dict['abort_lower'] = np.moveaxis(reshaped_signal.ordinate, -1, 0).real
        if filename is not None:
            np.savez(filename, **out_dict)
        return out_dict
    
    def bandwidth_average(self,band_lb,band_ub):
        """
        Integrates the PSD over frequency to get the power spectrum for each 
        frequency bin (line)

        Parameters
        ----------
        band_lb : ndarray
            (n_bands,1) array of bandwidth lower bounds
        band_ub : ndarray
            (n_bands,1) array of bandwidth upper bounds

        Returns
        -------
        PowerSpectralDensityArray with abscissa given by the mean of band_lb
        and band_ub
        
        Notes
        -------
        Determines which freq bins (lines) contribute to each band. Contribute
        means the freq bin is at least partially within the band limits
        
        The portion of the bin which contributes to the band is computed based
        multiplied by the fraction of the contributing frequency to get how
        much bin PS adds to the band PS
        """
        # Process inputs
        if self.ordinate.ndim ==2:
            freq = self.abscissa[0,:]
            ein_str = 'jk,lk->lj'
        else:
            freq = self.abscissa[0,0,:]
            ein_str = 'jk,lmk->lmj'
            
        band_lb, band_ub = [band_lb.flatten(),band_ub.flatten()]
        band_lb,band_ub = [ band_lb[:,np.newaxis] , band_ub[:,np.newaxis] ]
        df = freq[2] - freq[1]
        hlf_bin = df/2
        if np.abs(np.diff(freq)-df).max() > 1e-12:
            ValueError('Frequencies are not evenly spaced')
        
        # Determine matrix A s.t. A_jk PSD_lmk = PSDav_lmj
        bandwidths = band_ub-band_lb
        bin_map_lb = np.maximum(freq-hlf_bin,band_lb) # LB of overlap for each bin/band combo
        bin_map_ub = np.minimum(freq+hlf_bin,band_ub) # UB of overlap for each bin/band combo
        
        bin_to_band = (bin_map_ub-bin_map_lb)/df
        bin_to_band = np.maximum(bin_to_band,np.zeros(bin_to_band.shape))
        
        # Get PSD
        psd_ave = np.einsum(ein_str,bin_to_band,df*self.ordinate)
            
        psd_ave = psd_ave/(band_ub[:,0]-band_lb[:,0])
        freqs = np.concatenate( (band_lb,band_ub) ,1).mean(1)
        
        return data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,freqs,
                         psd_ave,self.coordinate,self.comment1,self.comment2,self.comment3,
                         self.comment4,self.comment5)


def power_spectral_density_array(abscissa,ordinate,coordinate,
                                 comment1='',comment2='',
                                 comment3='',comment4='',comment5=''):
    """
    Helper function to create a PowerSpectralDensityArray object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the PowerSpectralDensityArray object.

    Parameters
    ----------
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
    obj : PowerSpectralDensityArray
        The constructed PowerSpectralDensityArray object
    """
    return data_array(FunctionTypes.POWER_SPECTRAL_DENSITY,abscissa,ordinate,
                      coordinate, comment1,comment2,comment3,comment4,comment5)

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


def power_spectrum_array(abscissa,ordinate,coordinate,
                         comment1='',comment2='',
                         comment3='',comment4='',comment5=''):
    """
    Helper function to create a PowerSpectrumArray object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the PowerSpectrumArray object.

    Parameters
    ----------
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
    obj : PowerSpectrumArray
        The constructed PowerSpectrumArray object
    """
    return data_array(FunctionTypes.AUTOSPECTRUM,abscissa,ordinate,
                      coordinate, comment1,comment2,comment3,comment4,comment5)

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
                       window=np.array((1.0,)), return_model_data = False,
                       **timedata2frf_kwargs):
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
            default if not specified. samples_per_average, overlap, and window
            are not used if method=='LRM'.
        window : np.ndarray or str, optional
            A 1D ndarray with length samples_per_average that specifies the
            coefficients of the window.  No window is applied if not specified.
            If a string is specified, then the window will be obtained from scipy.
        **timedata2frf_kwargs : various
            Additional keyword arguments that may be passed into the
            timedata2frf function in sdynpy.frf.  If method=='LRM', see also
            frf_local_model in sdynpy.lrm for more options.


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
        if return_model_data:
            freq,frf,model_data = timedata2frf(ref_ord, res_ord, dt, samples_per_average,
                                            overlap, method, window, return_model_data=True,
                                            **timedata2frf_kwargs)
        else:
            freq,frf = timedata2frf(ref_ord, res_ord, dt, samples_per_average,
                                    overlap, method, window, return_model_data=False,
                                    **timedata2frf_kwargs)
        # Now construct the transfer function array
        coordinate = outer_product(res_data.coordinate.flatten(),
                                   ref_data.coordinate.flatten())
        if return_model_data:
            model_data = (model_data['model_selected']==len(model_data['modelset'])-1).mean()
            model_data = 'Highest order model selected in ' + str(round(model_data*100,1)) + '% of bins.'
            return data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                              freq, np.moveaxis(frf, 0, -1), coordinate,
                              comment1=model_data)
        else:
            return data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                              freq, np.moveaxis(frf, 0, -1), coordinate)


    def ifft(self, norm="backward",  odd_num_samples = False,
             **scipy_irfft_kwargs):
        """
        Converts frequency response functions to impulse response functions via an
        inverse fourier transform.

        Paramters
        ---------
        norm : str, optional
            The type of normalization applied to the fft computation.
        odd_num_samples : bool, optional
            If True, then it is assumed that the output signal has an odd
            number of samples, meaning the signal will have a length of 
            2*(m-1)+1 where m is the number of frequency lines.  Otherwise, the
            default value of 2*(m-1) is used, assuming an even signal.  This is
            ignored if num_samples is specified.
        scipy_irfft_kwargs :
            Additional keywords that will be passed to SciPy's irfft function.

        Raises
        ------
        Warning
            Raised if the transfer function array does not have evenly spaced
            frequency data in the 0-maximum frequency range, but appears to have been
            high pass filtered.
        ValueError
            Raised if the transfer function array does not have evenly spaced
            frequency data in the 0-maximum frequency range and it does not appear
            to have been high pass filtered.

        Returns
        -------
        ImpulseResponseFunctionArray
            The impulse response function array computed from the transfer function
            array.
        """

        df = self.abscissa_spacing
        min_freq = self.abscissa.min()
        if min_freq % df > 0.01*df:
            raise ValueError('Frequency bins do not line up with zero.  Cannot compute rfft bins.')
        first_frequency_bin = int(np.round(self.abscissa.min()/df))
        padding = np.zeros(self.ordinate.shape[:-1]+(first_frequency_bin,),self.ordinate.dtype)
        if padding.shape[-1] > 0:
            warnings.warn(
                'The FRFs are missing some low frequency data'
                + ' and it is assumed that this is due to some high pass cut-off.'
                + ' The data is being zero padded at low frequencies.')
        num_elements = first_frequency_bin+self.num_elements
        
        if odd_num_samples:
            num_samples = 2*(num_elements-1)+1
        else:
            num_samples = 2*(num_elements-1)

        # Organizing the FRFs for the ifft, this handles the zero padding if low frequency
        # data is missing
        ordinate = np.concatenate((padding,self.ordinate),axis=-1)
        irfft = scipyfft.irfft(ordinate, axis=-1, n=num_samples, norm=norm,
                               **scipy_irfft_kwargs)

        # Building the time vectors
        dt = 1 / (self.abscissa.max()*num_samples/np.floor(num_samples/2))
        time_vector = dt * np.arange(num_samples)

        return data_array(FunctionTypes.IMPULSE_RESPONSE_FUNCTION, time_vector, irfft, self.coordinate,
                          self.comment1, self.comment2, self.comment3, self.comment4, self.comment5)

    def enforce_causality(self, method='exponential_taper',
                          window_parameter=None,
                          end_of_ringdown=None):
        """
        Enforces causality on the frequency response function via a conversion
        to a impulse response function, applying a cutoff window, then converting
        back to a frequency response function.

        Parameters
        ----------
        method : str
            The window type that is applied to the data to enforce causality.
            Note that these options are not necessarily traditional windows
            (used for data processing). The current options are:
                - exponential_taper (default) - this applies a exponential taper
                  to the end of a boxcar window on the IRF.
                - boxcar - this applies a boxcar (uniform) window to the IRF
                  with the cuttoff at a specified sample.
                - exponential - this applies an exponential window to the IRF
                  with the 40 dB down point (of the window) at a specified sample.
                  Care should be taken when using this window type, since it can
                  lead to erratic behavior.
        window_parameter : int, optional
            This is a parameter that defines the window for the causality
            enforcement. Methods exist to define this parameter automatically
            if it isn't provided. The behaviors for the options are:
                - boxcar - the window_paramter is the sample after which the
                  IRF is set to zero. It is the same as the end_of_ringdown
                  parameter for this window type.
                - exponential - the window_parameter is where the 40 dB down
                  point is for the window. It is the same as the end_of_ringdown
                  parameter for this window type.
                - exponential_taper - the window_parameter is where the end point
                  of the window (where the amplitude is 0.001), as defined by the
                  number of samples after the uniform section of the window.
        end_of_ringdown : int, optional
            This is a parameter that defines the end of the uniform section of
            the exponetional_taper window. It is not used for either the boxcar
            or exponential window. Methods exist to define this parameter
            automatically if it isn't provided.

        Returns
        -------
        TransferFunctionArray
            The FRF with causality enforced.

        Notes
        -----
        This is a wrapper around the method in the impulse response function class
        and it may be wiser to use that function instead.

        Although optional, it is best practice for the user to supply a parameter
        for the end_of_ringdown variable if the "exponential_taper" method is
        being used or a window_parameter if the "exponential" or "boxcar" methods
        are being used. The code will attempt to find the end of the ring-down in
        the IRF and use use that as the end_of_ringdown parameter for the
        "exponential_taper" window or the window_parameter for the exponential and
        boxcar windows.

        It is not suggested that the user provide a window_paramter if the
        "exponential_taper" method is being used, since the default is likely the
        most logical choice.

        References
        ----------
        .. [1] Zvonkin, M. (2015). Methods for checking and enforcing physical quality of linear electrical network models
               [Masters Theses, Missouri University of Science and Technology], Missouri S&T Scholars' Mine, https://scholarsmine.mst.edu/masters_theses/7490/

        """
        irfs = self.ifft()
        causal_irfs = irfs.enforce_causality(method=method,
                                             window_parameter=window_parameter,
                                             end_of_ringdown=end_of_ringdown)
        return causal_irfs.fft()

    def svd(self, full_matrices=True, compute_uv=True, as_matrix=True):
        """
        Compute the SVD of the provided FRF matrix

        Parameters
        ----------
        full_matrices : bool, optional
            This is an optional input for np.linalg.svd, the default for this
            function is true (which differs from  the np.linalg.svd function).
        compute_uv : bool, optional
            This is an optional input for np.linalg.svd, the default for this
            function is true (which differs from the np.linalg.svd function).
        as_matrix : bool, optional
            If True, matrices are returned as a SDynPy Matrix class with named
            rows and columns.  Otherwise, a simple numpy array is returned

        Returns
        -------
        u : ndarray or Matrix
            Left hand singular vectors, sized [..., num_responses, num_responses].
            Only returned when compute_uv is True.
        s : ndarray or Matrix
            Singular values, sized [..., num_references]
        vh : ndarray or Matrix
            Right hand singular vectors, sized [..., num_references, num_references].
            Only returned when compute_uv is True.
        """
        frf = self.reshape_to_matrix()
        frfOrd = np.moveaxis(frf.ordinate, -1, 0)
        if compute_uv:
            u, s, vh = np.linalg.svd(frfOrd, full_matrices, compute_uv)
            if as_matrix:
                u = matrix(u, frf[:, 0].response_coordinate,
                           coordinate_array(np.arange(u.shape[-1])+1, 0))
                s = matrix(s[:, np.newaxis]*np.eye(s.shape[-1]), coordinate_array(np.arange(s.shape[-1])+1, 0),
                           coordinate_array(np.arange(s.shape[-1])+1, 0))
                vh = matrix(vh, coordinate_array(np.arange(vh.shape[-2])+1, 0),
                            frf[0, :].reference_coordinate,
                            )
            return u, s, vh
        else:
            s = np.linalg.svd(frfOrd, full_matrices, compute_uv)
            if as_matrix:
                s = matrix(s[:, np.newaxis]*np.eye(s.shape[-1]), coordinate_array(np.arange(s.shape[-1])+1, 0),
                           coordinate_array(np.arange(s.shape[-1])+1, 0))
            return s

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
        if tracking is not None:
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
        output_array.response_coordinate = coordinate_array(
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
        output_array.response_coordinate = coordinate_array(1, 0)
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
        output_array.response_coordinate = coordinate_array(
            np.arange(mif_ordinate.shape[1]) + 1, 0)
        return output_array

    def plot_cond_num(self, number_retained_values=None, min_frequency=None, max_frequency=None):
        """
        Plots the condition number of the FRF matrix

        Parameters
        ----------
        min_freqency : float, optional
            Minimum frequency to plot. The default is None.
        max_frequency : float, optional
            Maximum frequency to plot. The default is None.

        Returns
        -------
        None.

        """
        freq = self.flatten().abscissa[0, :]
        s_frf = self.svd(full_matrices=False, compute_uv=False, as_matrix=False)
        cond_num = s_frf[..., 0]/s_frf[..., -1]
        figure, axis = plt.subplots(1, 1)
        axis.plot(freq, cond_num, label='Unmodified Condition Number')
        if number_retained_values is not None:
            cutoff = np.zeros_like(s_frf[:, 1])
            number_retained_values = np.asarray(number_retained_values, dtype=np.intc)
            if number_retained_values.size == 1:
                cutoff = s_frf[:, number_retained_values-1]
            else:
                for ii in range(len(number_retained_values)):
                    cutoff[ii] = s_frf[ii, number_retained_values[ii]-1]
            axis.plot(freq, s_frf[..., 0]/cutoff, label='Modified Condition Number')
            axis.legend()
        axis.grid()
        axis.set_xlabel('Frequency (Hz)')
        axis.set_ylabel('Condition Number')
        axis.set_title('Condition Number of FRF Matrix')
        if min_frequency is not None:
            axis.set_xlim(left=min_frequency)
            cond_num = cond_num[np.argmin(np.abs(freq-min_frequency)):]
        if max_frequency is not None:
            axis.set_xlim(right=max_frequency)
            cond_num = cond_num[:np.argmin(np.abs(freq-max_frequency))]
        axis.set_ylim(bottom=0, top=cond_num.max()*1.01)

        return figure, axis

    def plot_singular_values(self, rcond=None,
                             condition_number=None,
                             number_retained_values=None,
                             regularization_parameter=None,
                             min_frequency=None,
                             max_frequency=None):
        """
        Plot the singular values of an FRF matrix with a visualization of the rcond tolerance

        Parameters
        ----------
        rcond : float or ndarray, optional
            Cutoff for small singular values. Implemented such that the cutoff is rcond*
            largest_singular_value (the same as np.linalg.pinv). This is to visualize the
            effect of rcond and is used for display purposes only.
        condition_number : float or ndarray, optional
            Condition number threshold for small singular values. The condition number
            is the reciprocal of rcond. This is to visualize the effect of condition
            number threshold and is used for display purposes only.
        number_retained_values : float or ndarray, optional
            Cutoff for small singular values a an integer value of number of values
            to retain. This is to visualize the effect of singular value truncation
            and is used for display purposes only.
        regularization_parameter: float or ndarray, optional
            Regularization parameter to compute the modified singular values. This is
            to visualize the effect of Tikhonov regularization and is used for display
            purposes only.
        min_frequency : float, optional
            Minimum frequency to plot
        max_frequency : float, optional
            Maximum frequency to plot
        """
        freq = self.flatten().abscissa[0, :]
        s_frf = self.svd(compute_uv=False, as_matrix=False)
        figure, axis = plt.subplots(1, 1)
        axis.semilogy(freq, s_frf)
        if rcond is not None:
            cutoff = s_frf[:, 0] * rcond
            axis.semilogy(freq, cutoff, color='k', linestyle='dashed', linewidth=3)
        if condition_number is not None:
            threshold = s_frf[:, 0] / condition_number
            cutoff = np.zeros_like(s_frf)
            for ii in range(threshold.size):
                number_values_above_cutoff = (s_frf[ii, :] > threshold[ii]).sum()
                cutoff[ii, number_values_above_cutoff:] = s_frf[ii, number_values_above_cutoff:]
            cutoff[cutoff == 0] = np.nan
            axis.semilogy(freq, cutoff, color='k', linestyle='dashed', linewidth=3)
        if number_retained_values is not None:
            cutoff = np.zeros_like(s_frf[:, 1])
            number_retained_values = np.asarray(number_retained_values, dtype=np.intc)
            if number_retained_values.size == 1:
                cutoff = s_frf[:, number_retained_values:]
            else:
                cutoff = np.zeros_like(s_frf)
                for ii in range(len(number_retained_values)):
                    cutoff[ii, number_retained_values[ii]:] = s_frf[ii, number_retained_values[ii]:]
            cutoff[cutoff == 0] = np.nan
            axis.semilogy(freq, cutoff, color='k', linestyle='dashed', linewidth=3)
        if regularization_parameter is not None:
            s_modified = compute_tikhonov_modified_singular_values(s_frf, regularization_parameter)
            figure.gca().set_prop_cycle(None)
            axis.semilogy(freq, s_modified, linestyle='dotted', linewidth=4)
        axis.grid()
        axis.set_xlabel('Frequency (Hz)')
        axis.set_ylabel('Singular Values')
        axis.set_title('Singular Values of FRF Matrix')
        if min_frequency is not None:
            axis.set_xlim(left=min_frequency)
            s_frf = s_frf[np.argmin(np.abs(freq-min_frequency)):, :]
        if max_frequency is not None:
            axis.set_xlim(right=max_frequency)
            s_frf = s_frf[:np.argmin(np.abs(freq-max_frequency)), :]
        axis.set_ylim(bottom=s_frf.min()*0.9, top=s_frf.max()*1.1)

        return figure, axis

    def plot(self, one_axis=True, part = None, subplots_kwargs={},
             plot_kwargs={}, abscissa_markers = None, 
             abscissa_marker_labels = None, abscissa_marker_type = 'vline',
             abscissa_marker_plot_kwargs = {}):
        """
        Plot the transfer functions

        Parameters
        ----------
        one_axis : bool, optional
            Set to True to plot all data on one axis.  Set to False to plot
            data on multiple subplots.  one_axis can also be set to a
            matplotlib axis to plot data on an existing axis.  The default is
            True.
        part : str, optional
            The part of the FRF to plot.  This can be, 'real', 'imag' or
            'imaginary', 'mag' or 'magnitude', or 'phase'.  If not specified,
            magnitude and phase will be plotted if `one_axis` is True, and 
            magnitude will be plotted if `one_axis` is False.
        subplots_kwargs : dict, optional
            Keywords passed to the matplotlib subplots function to create the
            figure and axes. The default is {}.
        plot_kwargs : dict, optional
            Keywords passed to the matplotlib plot function. The default is {}.
        abscissa_markers : ndarray, optional
            Array containing abscissa values to mark on the plot to denote
            significant events.
        abscissa_marker_labels : str or ndarray
            Array of strings to label the abscissa_markers with, or
            alternatively a format string that accepts index and abscissa
            inputs (e.g. '{index:}: {abscissa:0.2f}').  By default no label
            will be applied.
        abscissa_marker_type : str
            The type of marker to use.  This can either be the string 'vline'
            or a valid matplotlib symbol specifier (e.g. 'o', 'x', '.').
        abscissa_marker_plot_kwargs : dict
            Additional keyword arguments used when plotting the abscissa label
            markers.

        Returns
        -------
        axis : matplotlib axis or array of axes
             On which the data were plotted

        """
        if abscissa_markers is not None:
            if abscissa_marker_labels is None:
                abscissa_marker_labels = ['' for value in abscissa_markers]
            elif isinstance(abscissa_marker_labels,str):
                abscissa_marker_labels = [abscissa_marker_labels.format(
                    index = i, abscissa = v) for i,v in enumerate(abscissa_markers)]
                
        part_fns = {'imag':np.imag,
                    'real':np.real,
                    'mag':np.abs,
                    'magnitude':np.abs,
                    'phase':np.angle,
                    'imaginary':np.imag}
        part_labels = {'imag':'Imaginary',
                       'real':'Real',
                       'mag':'Magnitude',
                       'magnitude':'Magnitude',
                       'phase':'Phase',
                       'imaginary':'Imaginary'}
        part_yscale = {'imag':'linear',
                       'real':'linear',
                       'mag':'log',
                       'magnitude':'log',
                       'phase':'linear',
                       'imaginary':'linear'}
        if one_axis is True:
            if part is None:
                figure, axis = plt.subplots(2, 1, **subplots_kwargs)
                lines = axis[0].plot(self.flatten().abscissa.T, np.angle(
                    self.flatten().ordinate.T), **plot_kwargs)
                if abscissa_markers is not None:
                    if abscissa_marker_type == 'vline':
                        kwargs = {'color':'k'}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        for value,label in zip(abscissa_markers,abscissa_marker_labels):
                            axis[0].axvline(value, **kwargs)
                            axis[0].annotate(label, xy = (value, axis[0].get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                        axis[0].callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                    else:
                        for line in lines:
                            x = line.get_xdata()
                            y = line.get_ydata()
                            marker_y = np.interp(abscissa_markers, x, y)
                            kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                            kwargs.update(abscissa_marker_plot_kwargs)
                            axis[0].plot(abscissa_markers,marker_y,**kwargs)
                            for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                                axis[0].annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
                lines = axis[1].plot(self.flatten().abscissa.T, np.abs(
                    self.flatten().ordinate.T), **plot_kwargs)
                axis[1].set_yscale('log')
                if abscissa_markers is not None:
                    if abscissa_marker_type == 'vline':
                        kwargs = {'color':'k'}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        for value,label in zip(abscissa_markers,abscissa_marker_labels):
                            axis[1].axvline(value, **kwargs)
                            axis[1].annotate(label, xy = (value, axis[1].get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                        axis[1].callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                    else:
                        for line in lines:
                            x = line.get_xdata()
                            y = line.get_ydata()
                            marker_y = np.interp(abscissa_markers, x, y)
                            kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                            kwargs.update(abscissa_marker_plot_kwargs)
                            axis[1].plot(abscissa_markers,marker_y,**kwargs)
                            for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                                axis[1].annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
                axis[0].set_ylabel('Phase')
                axis[1].set_ylabel('Amplitude')
                axis[1].set_xlabel('Frequency')
                
            else:
                figure, axis = plt.subplots(1, 1, **subplots_kwargs)
                lines = axis.plot(self.flatten().abscissa.T, part_fns[part](
                    self.flatten().ordinate.T), **plot_kwargs)
                axis.set_yscale(part_yscale[part])
                axis.set_ylabel(part_labels[part])
                axis.set_xlabel('Frequency')
                if abscissa_markers is not None:
                    if abscissa_marker_type == 'vline':
                        kwargs = {'color':'k'}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        for value,label in zip(abscissa_markers,abscissa_marker_labels):
                            axis.axvline(value, **kwargs)
                            axis.annotate(label, xy = (value, axis.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                        axis.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                    else:
                        for line in lines:
                            x = line.get_xdata()
                            y = line.get_ydata()
                            marker_y = np.interp(abscissa_markers, x, y)
                            kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                            kwargs.update(abscissa_marker_plot_kwargs)
                            axis.plot(abscissa_markers,marker_y,**kwargs)
                            for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                                axis.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
        elif one_axis is False:
            ncols = int(np.floor(np.sqrt(self.size)))
            nrows = int(np.ceil(self.size / ncols))
            figure, axis = plt.subplots(nrows, ncols, **subplots_kwargs)
            if part is None:
                for i, (ax, (index, function)) in enumerate(zip(axis.flatten(), self.ndenumerate())):
                    lines = ax.plot(function.abscissa.T, np.abs(function.ordinate.T), **plot_kwargs)
                    ax.set_ylabel('/'.join([str(v) for i, v in function.coordinate.ndenumerate()]))
                    ax.set_yscale('log')
                    if abscissa_markers is not None:
                        if abscissa_marker_type == 'vline':
                            kwargs = {'color':'k'}
                            kwargs.update(abscissa_marker_plot_kwargs)
                            for value,label in zip(abscissa_markers,abscissa_marker_labels):
                                ax.axvline(value, **kwargs)
                                ax.annotate(label, xy = (value, ax.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                            ax.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                        else:
                            for line in lines:
                                x = line.get_xdata()
                                y = line.get_ydata()
                                marker_y = np.interp(abscissa_markers, x, y)
                                kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                                kwargs.update(abscissa_marker_plot_kwargs)
                                ax.plot(abscissa_markers,marker_y,**kwargs)
                                for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                                    ax.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
            else:
                for i, (ax, (index, function)) in enumerate(zip(axis.flatten(), self.ndenumerate())):
                    lines = ax.plot(function.abscissa.T, part_fns[part](function.ordinate.T), **plot_kwargs)
                    ax.set_ylabel('/'.join([str(v) for i, v in function.coordinate.ndenumerate()]))
                    ax.set_yscale(part_yscale[part])
                    if abscissa_markers is not None:
                        if abscissa_marker_type == 'vline':
                            kwargs = {'color':'k'}
                            kwargs.update(abscissa_marker_plot_kwargs)
                            for value,label in zip(abscissa_markers,abscissa_marker_labels):
                                ax.axvline(value, **kwargs)
                                ax.annotate(label, xy = (value, ax.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                            ax.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                        else:
                            for line in lines:
                                x = line.get_xdata()
                                y = line.get_ydata()
                                marker_y = np.interp(abscissa_markers, x, y)
                                kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                                kwargs.update(abscissa_marker_plot_kwargs)
                                ax.plot(abscissa_markers,marker_y,**kwargs)
                                for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                                    ax.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
            for ax in axis.flatten()[i + 1:]:
                ax.remove()
        else:
            axis = one_axis
            if part is None:
                try:
                    lines = axis[0].plot(self.flatten().abscissa.T, np.angle(
                        self.flatten().ordinate.T), **plot_kwargs)
                    if abscissa_markers is not None:
                        if abscissa_marker_type == 'vline':
                            kwargs = {'color':'k'}
                            kwargs.update(abscissa_marker_plot_kwargs)
                            for value,label in zip(abscissa_markers,abscissa_marker_labels):
                                axis[0].axvline(value, **kwargs)
                                axis[0].annotate(label, xy = (value, axis[0].get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                            axis[0].callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                        else:
                            for line in lines:
                                x = line.get_xdata()
                                y = line.get_ydata()
                                marker_y = np.interp(abscissa_markers, x, y)
                                kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                                kwargs.update(abscissa_marker_plot_kwargs)
                                axis[0].plot(abscissa_markers,marker_y,**kwargs)
                                for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                                    axis[0].annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
                    lines = axis[1].plot(self.flatten().abscissa.T, np.abs(
                        self.flatten().ordinate.T), **plot_kwargs)
                    if abscissa_markers is not None:
                        if abscissa_marker_type == 'vline':
                            kwargs = {'color':'k'}
                            kwargs.update(abscissa_marker_plot_kwargs)
                            for value,label in zip(abscissa_markers,abscissa_marker_labels):
                                axis[1].axvline(value, **kwargs)
                                axis[1].annotate(label, xy = (value, axis[1].get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                            axis[1].callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                        else:
                            for line in lines:
                                x = line.get_xdata()
                                y = line.get_ydata()
                                marker_y = np.interp(abscissa_markers, x, y)
                                kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                                kwargs.update(abscissa_marker_plot_kwargs)
                                axis[1].plot(abscissa_markers,marker_y,**kwargs)
                                for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                                    axis[1].annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
                except TypeError:
                    lines = axis.plot(self.flatten().abscissa.T, np.abs(
                        self.flatten().ordinate.T), **plot_kwargs)
                    if abscissa_markers is not None:
                        if abscissa_marker_type == 'vline':
                            kwargs = {'color':'k'}
                            kwargs.update(abscissa_marker_plot_kwargs)
                            for value,label in zip(abscissa_markers,abscissa_marker_labels):
                                axis.axvline(value, **kwargs)
                                axis.annotate(label, xy = (value, axis.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                            axis.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                        else:
                            for line in lines:
                                x = line.get_xdata()
                                y = line.get_ydata()
                                marker_y = np.interp(abscissa_markers, x, y)
                                kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                                kwargs.update(abscissa_marker_plot_kwargs)
                                axis.plot(abscissa_markers,marker_y,**kwargs)
                                for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                                    axis.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
            else:
                lines = axis.plot(self.flatten().abscissa.T, part_fns[part](
                    self.flatten().ordinate.T), **plot_kwargs)
                if abscissa_markers is not None:
                    if abscissa_marker_type == 'vline':
                        kwargs = {'color':'k'}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        for value,label in zip(abscissa_markers,abscissa_marker_labels):
                            axis.axvline(value, **kwargs)
                            axis.annotate(label, xy = (value, axis.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                        axis.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                    else:
                        for line in lines:
                            x = line.get_xdata()
                            y = line.get_ydata()
                            marker_y = np.interp(abscissa_markers, x, y)
                            kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                            kwargs.update(abscissa_marker_plot_kwargs)
                            axis.plot(abscissa_markers,marker_y,**kwargs)
                            for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                                axis.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
        return axis

    def plot_with_coherence(self, coherence, part = None, subplots_kwargs={}, plot_kwargs={}):
        axes = self.plot(one_axis=False,part=part,subplots_kwargs = subplots_kwargs,
                         plot_kwargs = plot_kwargs)
        # Get the corresponding coherences
        if isinstance(coherence,CoherenceArray):
            coherence = coherence[self.coordinate]
        elif isinstance(coherence,MultipleCoherenceArray):
            coherence = coherence[self.coordinate[...,:1]]
        coh_axes = []
        for ax, coh in zip(axes.flatten(),coherence.flatten()):
            coh_ax = ax.twinx()
            coh_ax.plot(coh.abscissa,coh.ordinate,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],**plot_kwargs)
            coh_ax.set_yscale('linear')
            coh_ax.set_ylim([-0.05,1.05])
            coh_axes.append(coh_ax)
        return axes,np.array(coh_axes).reshape(axes.shape)

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

    def apply_transformation(self, response_transformation=None, reference_transformation=None, 
                             invert_response_transformation=False, invert_reference_transformation=True):
        """
        Applies reference and response transformations to the transfer 
        functions.

        Parameters
        ----------
        response_transformation : Matrix, optional
            The response transformation to apply to the (rows of the)
            transfer functions. It should be a SDynPy matrix object with 
            the "transformed" coordinates on the rows and the "physical" 
            coordinates on the columns. The matrix can be either 2D or 3D
            (for a frequency dependent transform).
        reference_transformation : Matrix, optional
            The reference transformation to apply to the (columns of the)
            transfer functions. It should be a SDynPy matrix object with 
            the "transformed" coordinates on the rows and the "physical" 
            coordinates on the columns. The matrix can be either 2D or 3D
            (for a frequency dependent transform).
        invert_response_transformation : bool, optional
            Whether or not to invert the response transformation when 
            applying it to the transfer functions. The default is false, 
            which is standard practice. The row/column ordering in the 
            reference transformation should be flipped if this is set to
            true.
        invert_reference_transformation : bool, optional
            Whether or not to invert the reference transformation when 
            applying it to the transfer functions. The default is true, 
            which is standard practice. The row/column ordering in the 
            reference transformation should be flipped if this is set to
            false.

        Raises
        ------
        ValueError
            If the physical degrees of freedom in the transformations don't
            match the transfer functions
        
        Returns
        -------
        transformed_transfer_function : TransferFunctionArray
            The transfer functions with the transformations applied.

        Notes
        -----
        This method can be used with just a response transformation, just a reference
        transformation, or both a response and reference transformation. The 
        transformation will be set to identity if it is not supplied. 

        References
        ----------
        .. [1] M. Van der Seijs, D. van den Bosch, D. Rixen, and D. Klerk, "An improved 
               methodology for the virtual point transformation of measured frequency 
               response functions in dynamic substructuring," in Proceedings of the 4th 
               International Conference on Computational Methods in Structural Dynamics 
               and Earthquake Engineering, Kos Island, 2013, pp. 4334-4347, 
               doi: 10.7712/120113.4816.C1539. 
        """
        if not self.validate_common_abscissa():
            raise ValueError('The abscissa must be consistent accross all functions in the NDDataArray')

        physical_response_coordinate = np.unique(self.response_coordinate)
        physical_reference_coordinate = np.unique(self.reference_coordinate)
        original_frf_ordinate = np.moveaxis(self[outer_product(physical_response_coordinate, physical_reference_coordinate)].ordinate, -1, 0)

        if reference_transformation is None:
            transformed_reference_coordinate = physical_reference_coordinate
            reference_transformation_matrix = np.eye(physical_reference_coordinate.shape[0])
        else:
            if invert_reference_transformation:
                if not np.all(np.unique(reference_transformation.column_coordinate) == physical_reference_coordinate):
                    raise ValueError('The physical coordinates in the reference transformation do no match the transfer functions')
                transformed_reference_coordinate = np.unique(reference_transformation.row_coordinate)
                reference_transformation_matrix = np.linalg.pinv(reference_transformation[transformed_reference_coordinate, physical_reference_coordinate])
            elif not invert_reference_transformation:
                if not np.all(np.unique(reference_transformation.row_coordinate) == physical_reference_coordinate):
                    raise ValueError('The physical coordinates in the reference transformation do no match the transfer functions')
                transformed_reference_coordinate = np.unique(reference_transformation.column_coordinate)
                reference_transformation_matrix = reference_transformation[physical_reference_coordinate, transformed_reference_coordinate]

        if response_transformation is None:
            transformed_response_coordinate = physical_response_coordinate
            response_transformation_matrix = np.eye(physical_response_coordinate.shape[0])
        else:
            if invert_response_transformation:
                if not np.all(np.unique(response_transformation.row_coordinate) == physical_response_coordinate):
                    raise ValueError('The physical coordinates in the response transformation do no match the transfer functions')
                transformed_response_coordinate = np.unique(response_transformation.row_coordinate)
                response_transformation_matrix = np.linalg.pinv(response_transformation[transformed_response_coordinate, physical_response_coordinate])
            elif not invert_response_transformation:
                if not np.all(np.unique(response_transformation.column_coordinate) == physical_response_coordinate):
                    raise ValueError('The physical coordinates in the response transformation do no match the transfer functions')
                transformed_response_coordinate = np.unique(response_transformation.row_coordinate)
                response_transformation_matrix = response_transformation[transformed_response_coordinate, physical_response_coordinate]
                
        transformed_frf_ordinate = response_transformation_matrix @ original_frf_ordinate @ reference_transformation_matrix

        return data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION, self.ravel().abscissa[0], np.moveaxis(transformed_frf_ordinate, 0, -1), 
                          outer_product(transformed_response_coordinate, transformed_reference_coordinate))

    def interpolate_by_zero_pad(self, irf_padded_length, return_irf=False,
                                odd_num_samples = False):
        """
        Interpolates a transfer function by zero padding or truncating its
        impulse response

        Parameters
        ----------
        irf_padded_length : int
            Length of the final zero-padded impulse response function
        return_irf : bool, optional
            If True, the zero-padded impulse response function will be returned.
            If False, it will be transformed back to a transfer function prior
            to being returned.
        odd_num_samples : bool, optional
            If True, then it is assumed that the spectrum has been constructed
            from a signal with an odd number of samples.  Note that this
            function uses the rfft function from scipy to compute the
            inverse fast fourier transform.  The irfft function is not round-trip
            equivalent for odd functions, because by default it assumes an even
            signal length.  For an odd signal length, the user must either specify
            odd_num_samples = True to make it round-trip equivalent.

        Returns
        -------
        TransferFunctionArray or ImpulseResponseFunctionArray:
            Transfer function array with appropriately spaced abscissa

        Notes
        -----
        This function will automatically set the last frequency line of the
        TransferFunctionArray to zero because it won't be accurate anyway.
        If `irf_padded_length` is less than the current function's `num_elements`,
        then it will be truncated instead of zero-padded.
        """
        irf = self.ifft(odd_num_samples=odd_num_samples)
        if irf_padded_length < irf.num_elements:
            irf = irf.idx_by_el[:irf_padded_length]
        else:
            irf = irf.zero_pad(irf_padded_length - irf.num_elements,left=False,right=True)
        if return_irf:
            return irf
        else:
            frf = irf.fft(norm='backward')
            if irf_padded_length % 2 == 0:
                frf.ordinate[...,-1] = 0
            return frf

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
        dof_list = np.array(dof_list).view(CoordinateArray)
        return self.substructure_by_constraint_matrix(dof_list, constraint_matrix)

def transfer_function_array(abscissa,ordinate,coordinate,
                            comment1='',comment2='',
                            comment3='',comment4='',comment5=''):
    """
    Helper function to create a TransferFunctionArray object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the TransferFunctionArray object.

    Parameters
    ----------
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
    obj : TransferFunctionArray
        The constructed TransferFunctionArray object
    """
    return data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,abscissa,ordinate,
                      coordinate, comment1,comment2,comment3,comment4,comment5)


class ImpulseResponseFunctionArray(NDDataArray):
    """Data array used to store impulse response functions"""
    def __new__(subtype, shape, nelements, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(subtype, shape, nelements, 2, 'float64', buffer, offset, strides, order)
        return obj

    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.IMPULSE_RESPONSE_FUNCTION

    def fft(self, norm='backward', **scipy_rfft_kwargs):
        """
        Converts the impulse response function to a frequency response function
        using the fft function.

        Paramters
        ---------
        norm : str, optional
            The type of normalization applied to the fft computation.
        scipy_rfft_kwargs :
            Additional keywords that will be passed to SciPy's rfft function.

        Returns
        -------
        TransferFunctionArray
            The transfer function array computed from the impusle response function
            array.
        """
        # Some initial organization
        irfs = self.reshape_to_matrix()
        irf_ordinate = np.moveaxis(irfs.ordinate, -1, 0)

        # Getting sampling parameters for the fft
        number_samples = irf_ordinate.shape[0]
        dt = irfs[0, 0].abscissa[1] - irfs[0, 0].abscissa[0]

        # Doing the fft
        frf_ordinate = scipyfft.rfft(irf_ordinate, axis=0, norm=norm, **scipy_rfft_kwargs)
        freq_vector = scipyfft.rfftfreq(number_samples, dt)

        # Broadcasting the frequency vector to the correct size
        abscissa = np.broadcast_to(freq_vector[..., np.newaxis, np.newaxis], frf_ordinate.shape)

        return data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION, np.moveaxis(abscissa, 0, -1),
                          np.moveaxis(frf_ordinate, 0, -1), irfs.coordinate)

    def find_end_of_ringdown(self):
        """
        Finds the end of the ringdown in a impulse response function (IRF).

        It does this by smoothing the IRF via a moving average filter, then finding
        the index of the minimum of the smoothed IRF (for each response/reference
        pair). The "end of ringdown" is defined as the median of the possible
        indices.

        Returns
        -------
        end_of_ringdow : int
            Index that represents the end of the ringdown for the supplied IRFs.
        """
        irf_ordinate = np.moveaxis(self.ordinate, -1, 0)

        # Start off by performing a moving average on the IRF to make
        # it easier to find the end of the ring down
        moving_average_kernel_length = int(irf_ordinate.shape[0]/5)
        moving_average_kernel = np.broadcast_to(np.ones(moving_average_kernel_length)[..., np.newaxis, np.newaxis],
                                                (moving_average_kernel_length, irf_ordinate.shape[1], irf_ordinate.shape[2]))
        irf_moving_average = convolve(np.absolute(irf_ordinate), moving_average_kernel, mode='same')

        # Find where the smoothed version of the IRF is at a minimum and
        # then using the median index (by channel) as the window_parameter
        min_index = np.argmin(irf_moving_average, axis=0)
        end_of_ringdown = int(np.median(min_index))

        return end_of_ringdown

    def enforce_causality(self, method='exponential_taper',
                          window_parameter=None,
                          end_of_ringdown=None):
        """
        Enforces causality on the impulse response function via a cutoff
        of some sort.

        Parameters
        ----------
        method : str
            The window type that is applied to the data to enforce causality.
            Note that these options are not necessarily traditional windows
            (used for data processing). The current options are:
                - exponential_taper (default) - this applies a exponential taper
                to the end of a boxcar window on the IRF.
                - boxcar - this applies a boxcar (uniform) window to the IRF
                with the cuttoff at a specified sample.
                - exponential - this applies an exponential window to the IRF
                with the 40 dB down point (of the window) at a specified sample.
                Care should be taken when using this window type, since it can
                lead to erratic behavior.
        window_parameter : int, optional
            This is a parameter that defines the window for the causality
            enforcement. Methods exist to define this parameter automatically
            if it isn't provided. The behaviors for the options are:
                - boxcar - the window_paramter is the sample after which the
                IRF is set to zero. It is the same as the end_of_ringdown
                parameter for this window type.
                - exponential - the window_parameter is where the 40 dB down
                point is for the window. It is the same as the end_of_ringdown
                parameter for this window type.
                - exponential_taper - the window_parameter is where the end point
                of the window (where the amplitude is 0.001), as defined by the
                number of samples after the uniform section of the window.
        end_of_ringdown : int, optional
            This is a parameter that defines the end of the uniform section of
            the exponetional_taper window. It is not used for either the boxcar
            or exponential window. Methods exist to define this parameter
            automatically if it isn't provided.

        Returns
        -------
        ImpulseResponseFunctionArray
            The IRF with causality enforced.

        Notes
        -----
        Although optional, it is best practice for the user to supply a parameter
        for the end_of_ringdown variable if the "exponential_taper" method is
        being used or a window_parameter if the "exponential" or "boxcar" methods
        are being used. The code will attempt to find the end of the ring-down in
        the IRF and use use that as the end_of_ringdown parameter for the
        "exponential_taper" window or the window_parameter for the exponential and
        boxcar windows.

        It is not suggested that the user provide a window_paramter if the
        "exponential_taper" method is being used, since the default is likely the
        most logical choice.

        References
        ----------
        .. [1] Zvonkin, M. (2015). Methods for checking and enforcing physical quality of linear electrical network models
               [Masters Theses, Missouri University of Science and Technology], Missouri S&T Scholars' Mine, https://scholarsmine.mst.edu/masters_theses/7490/
        """
        # Organizing the IRFs and pulling the ordinate out
        irfs = self.reshape_to_matrix()
        irf_ord = np.moveaxis(irfs.ordinate, -1, 0)

        if method == 'exponential_taper' and end_of_ringdown is None:
            end_of_ringdown = self.find_end_of_ringdown()

        if method == 'boxcar' and window_parameter is not None and window_parameter >= irf_ord.shape[0]:
            raise ValueError('window parameter is greater than the IRF block size and creates an illogical window for the data')

        if method in ['exponential', 'boxcar'] and window_parameter is None:
            window_parameter = self.find_end_of_ringdown()

        # Generating the desired window, based on the seleted method
        if method == 'exponential_taper':
            window = np.ones(irf_ord.shape[0])
            window[end_of_ringdown:] = np.zeros(irf_ord.shape[0] - end_of_ringdown)
            window_length = irf_ord.shape[0] - end_of_ringdown
            if window_parameter is None:
                window_parameter = window_length
            window_tau = -(window_parameter-1)/np.log(0.001)
            window[end_of_ringdown+np.arange(window_length)] = exponential(window_length, center=0, tau=window_tau, sym=False)
        elif method == 'exponential':
            end_of_ringdown = window_parameter
            window_tau = window_parameter*8.69/20
            window = exponential(irf_ord.shape[0], center=0, tau=window_tau, sym=False)
        elif method == 'boxcar':
            end_of_ringdown = window_parameter
            window = np.ones(irf_ord.shape[0])
            window[window_parameter:] = 0

        # Setting up the time reversal window so the non-causal portion of the IRF can be added back to the IRFs
        time_reversal_window = np.ones(irf_ord.shape[0]) - window
        # time_reversal_window[:window_parameter] = 0, might need this if using an exponential window

        method_statement_start = 'Causality is being enforced using '
        method_statement_middle = ' method with a end_of_ringdown of '
        print(method_statement_start+method+method_statement_middle+str(end_of_ringdown))

        # Applying the window to the data for the causality enforcement
        irf_causal_ord = irf_ord * window[..., np.newaxis, np.newaxis]
        irf_noncausal_ord = irf_ord * time_reversal_window[..., np.newaxis, np.newaxis]

        # Generating the new impulse response function array
        irfs_causal = irfs
        irfs_causal.ordinate = np.moveaxis(irf_causal_ord+np.flip(irf_noncausal_ord, 0), 0, -1)

        return irfs_causal

def impulse_response_function_array(abscissa,ordinate,coordinate,
                                    comment1='',comment2='',
                                    comment3='',comment4='',comment5=''):
    """
    Helper function to create a ImpulseResponseFunctionArray object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the ImpulseResponseFunctionArray object.

    Parameters
    ----------
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
    obj : ImpulseResponseFunctionArray
        The constructed ImpulseResponseFunctionArray object
    """
    return data_array(FunctionTypes.IMPULSE_RESPONSE_FUNCTION,abscissa,ordinate,
                      coordinate, comment1,comment2,comment3,comment4,comment5)

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


def transmissibility_array(abscissa,ordinate,coordinate,
                           comment1='',comment2='',
                           comment3='',comment4='',comment5=''):
    """
    Helper function to create a TransmissibilityArray object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the TransmissibilityArray object.

    Parameters
    ----------
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
    obj : TransmissibilityArray
        The constructed TransmissibilityArray object
    """
    return data_array(FunctionTypes.TRANSMISIBILITY,abscissa,ordinate,
                      coordinate, comment1,comment2,comment3,comment4,comment5)


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

    @staticmethod
    def from_time_data(response_data: TimeHistoryArray,
                       samples_per_average: int = None,
                       overlap: float = 0.0,
                       window=np.array((1.0,)),
                       reference_data: TimeHistoryArray = None):
        """
        Computes coherence from reference and response time histories

        Parameters
        ----------
        response_data : TimeHistoryArray
            Time data to be used as responses
        samples_per_average : int, optional
            Number of samples used to split up the signals into averages.  The
            default is None, meaning the data is treated as a single measurement
            frame.
        overlap : float, optional
            The overlap as a fraction of the frame (e.g. 0.5 specifies 50% overlap).
            The default is 0.0, meaning no overlap is used.
        window : np.ndarray or str, optional
            A 1D ndarray with length samples_per_average that specifies the
            coefficients of the window.  A Hann window is applied if not specified.
            If a string is specified, then the window will be obtained from scipy.
        reference_data : TimeHistoryArray
            Time data to be used as reference.  If not specified, the response
            data will be used as references, resulting in a square coherence matrix.

        Raises
        ------
        ValueError
            Raised if reference and response functions do not have consistent
            abscissa

        Returns
        -------
        PowerSpectralDensityArray
            A PSD array computed from the specified reference and
            response signals.

        """
        if reference_data is None:
            reference_data = response_data
        ref_data = reference_data.flatten()
        res_data = response_data.flatten()
        ref_ord = ref_data.ordinate
        res_ord = res_data.ordinate
        if ((not np.allclose(ref_data[0].abscissa,
                           res_data[0].abscissa))
            or (not np.allclose(ref_data.abscissa_spacing,res_data.abscissa_spacing))):
            raise ValueError('Reference and Response Data should have identical abscissa!')
        dt = res_data.abscissa_spacing
        df, cpsd = sp_cpsd(res_ord, 1/dt, samples_per_average, overlap,
                           window, reference_signals = ref_ord)
        df, res_asds = sp_cpsd(res_ord, 1/dt, samples_per_average, overlap,
                           window, only_asds = True)
        df, ref_asds = sp_cpsd(ref_ord, 1/dt, samples_per_average, overlap,
                               window, only_asds = True)
        num = np.abs(cpsd)**2
        den = res_asds[...,np.newaxis]*ref_asds[:,np.newaxis,:]
        den[den == 0.0] = 1  # Set to 1 if denominator is zero
        coh = np.real(num/den)
        freq = np.arange(cpsd.shape[0])*df
        # Now construct the transfer function array
        coordinate = outer_product(res_data.coordinate.flatten(),
                                   ref_data.coordinate.flatten())
        return data_array(FunctionTypes.COHERENCE,
                          freq, np.moveaxis(coh, 0, -1), coordinate)


def coherence_array(abscissa,ordinate,coordinate,
                    comment1='',comment2='',
                    comment3='',comment4='',comment5=''):
    """
    Helper function to create a CoherenceArray object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the CoherenceArray object.

    Parameters
    ----------
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
    obj : CoherenceArray
        The constructed CoherenceArray object
    """
    return data_array(FunctionTypes.COHERENCE,abscissa,ordinate,
                      coordinate, comment1,comment2,comment3,comment4,comment5)


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
    
    @staticmethod
    def from_time_data(response_data: TimeHistoryArray,
                       samples_per_average: int = None,
                       overlap: float = 0.0,
                       window=np.array((1.0,)),
                       reference_data: TimeHistoryArray = None):
        """
        Computes coherence from reference and response time histories

        Parameters
        ----------
        response_data : TimeHistoryArray
            Time data to be used as responses
        samples_per_average : int, optional
            Number of samples used to split up the signals into averages.  The
            default is None, meaning the data is treated as a single measurement
            frame.
        overlap : float, optional
            The overlap as a fraction of the frame (e.g. 0.5 specifies 50% overlap).
            The default is 0.0, meaning no overlap is used.
        window : np.ndarray or str, optional
            A 1D ndarray with length samples_per_average that specifies the
            coefficients of the window.  A Hann window is applied if not specified.
            If a string is specified, then the window will be obtained from scipy.
        reference_data : TimeHistoryArray
            Time data to be used as reference.  If not specified, the response
            data will be used as references, resulting in a square coherence matrix.

        Raises
        ------
        ValueError
            Raised if reference and response functions do not have consistent
            abscissa

        Returns
        -------
        PowerSpectralDensityArray
            A PSD array computed from the specified reference and
            response signals.

        """
        if reference_data is None:
            reference_data = response_data
        ref_data = reference_data.flatten()
        res_data = response_data.flatten()
        ref_ord = ref_data.ordinate
        res_ord = res_data.ordinate
        if ((not np.allclose(ref_data[0].abscissa,
                           res_data[0].abscissa))
            or (not np.allclose(ref_data.abscissa_spacing,res_data.abscissa_spacing))):
            raise ValueError('Reference and Response Data should have identical abscissa!')
        dt = res_data.abscissa_spacing
        df, cross_cpsd = sp_cpsd(res_ord, 1/dt, samples_per_average, overlap,
                           window, reference_signals = ref_ord)
        df, res_apsd = sp_cpsd(res_ord, 1/dt, samples_per_average, overlap,
                              window,only_asds=True)
        df, ref_cpsd = sp_cpsd(ref_ord, 1/dt, samples_per_average, overlap,
                               window)
        
        num = np.einsum('fij,fji->fi', cross_cpsd, np.linalg.solve(ref_cpsd,np.moveaxis(cross_cpsd.conj(),-2,-1)))
        den = res_apsd
        den[den == 0.0] = 1  # Set to 1 if denominator is zero
        mcoh = np.real(num/den)
        freq = np.arange(num.shape[0])*df
        return data_array(FunctionTypes.MULTIPLE_COHERENCE,
                          freq, np.moveaxis(mcoh, 0, -1), res_data.coordinate)

def multiple_coherence_array(abscissa,ordinate,coordinate,
                    comment1='',comment2='',
                    comment3='',comment4='',comment5=''):
    """
    Helper function to create a MultipleCoherenceArray object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the MultipleCoherenceArray object.

    Parameters
    ----------
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
    obj : MultipleCoherenceArray
        The constructed MultipleCoherenceArray object
    """
    return data_array(FunctionTypes.MULTIPLE_COHERENCE,abscissa,ordinate,
                      coordinate, comment1,comment2,comment3,comment4,comment5)


class CorrelationArray(NDDataArray):
    """Data array used to store correlation data"""
    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.CROSSCORRELATION

def correlation_array(abscissa,ordinate,coordinate,
                    comment1='',comment2='',
                    comment3='',comment4='',comment5=''):
    """
    Helper function to create a CorrelationArray object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the CorrelationArray object.

    Parameters
    ----------
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
    obj : CorrelationArray
        The constructed CorrelationArray object
    """
    return data_array(FunctionTypes.CROSSCORRELATION,abscissa,ordinate,
                      coordinate, comment1,comment2,comment3,comment4,comment5)


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

def mode_indicator_function_array(abscissa,ordinate,coordinate,
                                  comment1='',comment2='',
                                  comment3='',comment4='',comment5=''):
    """
    Helper function to create a ModeIndicatorFunctionArray object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the ModeIndicatorFunctionArray object.

    Parameters
    ----------
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
    obj : ModeIndicatorFunctionArray
        The constructed ModeIndicatorFunctionArray object
    """
    return data_array(FunctionTypes.MODE_INDICATOR_FUNCTION,abscissa,ordinate,
                      coordinate, comment1,comment2,comment3,comment4,comment5)

class ShockResponseSpectrumArray(NDDataArray):

    _srs_type_map = {'ppaa': 1,
                     'pnaa': 2,
                     'pmaa': 3,
                     'rpaa': 4,
                     'rnaa': 5,
                     'rmaa': 6,
                     'mpaa': 7,
                     'mnaa': 8,
                     'mmaa': 9,
                     'alaa': 10,
                     'pprd': -1,
                     'pnrd': -2,
                     'pmrd': -3,
                     'rprd': -4,
                     'rnrd': -5,
                     'rmrd': -6,
                     'mprd': -7,
                     'mnrd': -8,
                     'mmrd': -9,
                     'alrd': -10}

    """Shock Response Spectrum (SRS)"""
    def __new__(subtype, shape, nelements, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(subtype, shape, nelements, 1, 'float64', buffer, offset, strides, order)
        return obj

    @property
    def function_type(self):
        """
        Returns the function type of the data array
        """
        return FunctionTypes.SHOCK_RESPONSE_SPECTRUM

    def sum_decayed_sines(self, sample_rate, block_size,
                          sine_frequencies=None, sine_tone_range=None, sine_tone_per_octave=None,
                          sine_amplitudes=None, sine_decays=None, sine_delays=None,
                          srs_damping=0.03, srs_type="MMAA",
                          compensation_frequency=None, compensation_decay=0.95,
                          # Paramters for the iteration
                          number_of_iterations=3, convergence=0.8,
                          error_tolerance=0.05,
                          tau=None, num_time_constants=None, decay_resolution=None,
                          scale_factor=1.02,
                          acceleration_factor=1.0,
                          plot_results=False, srs_frequencies=None,
                          return_velocity=False, return_displacement=False,
                          return_srs=False, return_sine_table=False,
                          verbose=False):
        """Generate a Sum of Decayed Sines signal given an SRS.

        Note that there are many approaches to do this, with many optional arguments
        so please read the documentation carefully to understand which arguments
        must be passed to the function.

        Parameters
        ----------
        sample_rate : float
            The sample rate of the generated signal.
        block_size : int
            The number of samples in the generated signal.
        sine_frequencies : np.ndarray, optional
            The frequencies of the sine tones.  If this argument is not specified
            and the `sine_tone_range` argument is not specified, then the
            `sine_tone_range` will be set to the maximum and minimum abscissa
            value for this `ShockResponseSpectrumArray`.
        sine_tone_range : np.ndarray, optional
            A length-2 array containing the minimum and maximum sine tone to
            generate.  If this argument is not specified
            and the `sine_frequencies` argument is not specified, then the
            `sine_tone_range` will be set to the maximum and minimum abscissa
            value for this `ShockResponseSpectrumArray`.
        sine_tone_per_octave : int, optional
            The number of sine tones per octave. If not specified along with
            `sine_tone_range`, then a default value of 4 will be used if the
            `srs_damping` is >= 0.05.  Otherwise, the formula of
            `sine_tone_per_octave = 9 - srs_damping*100` will be used.
        sine_amplitudes : np.ndarray, optional
            The initial amplitude of the sine tones used in the optimization.  If
            not specified, they will be set to the value of the SRS at each frequency
            divided by the quality factor of the SRS.
        sine_decays : np.ndarray, optional
            An array of decay value time constants (often represented by variable
            tau).  Tau is the time for the amplitude of motion to decay 63% defined
            by the equation `1/(2*np.pi*freq*zeta)` where `freq` is the frequency
            of the sine tone and `zeta` is the fraction of critical damping.
            If not specified, then either the `tau` or `num_time_constants`
            arguments must be specified instead.
        sine_delays : np.ndarray, optional
            An array of delay values for the sine components. If not specified,
            all tones will have zero delay.
        srs_damping : float, optional
            Fraction of critical damping to use in the SRS calculation (e.g. you
            should specify 0.03 to represent 3%, not 3). If not defined, a
            default of 0.03 will be used.
        srs_type : int or str
            The type of spectrum desired: This can be an integer or a string.
            If `srs_type` is an integer:
            if `srs_type` > 0 (pos) then the SRS will be a base
            acceleration-absolute acceleration model
            If `srs_type` < 0 (neg) then the SRS will be a base acceleration-relative
            displacement model (expressed in equivalent static acceleration units).
            If abs(`srs_type`) is:
                1--positive primary,  2--negative primary,  3--absolute maximum primary
                4--positive residual, 5--negative residual, 6--absolute maximum residual
                7--largest of 1&4, maximum positive, 8--largest of 2&5, maximum negative
                9 -- maximax, the largest absolute value of 1-8
               10 -- returns a matrix s(9,length(fn)) with all the types 1-9.
        compensation_frequency : float
            The frequency of the compensation pulse.  If not specified, it will be
            set to 1/3 of the lowest sine tone
        compensation_decay : float
            The decay value for the compensation pulse.  If not specified, it will
            be set to 0.95.
        number_of_iterations : int, optional
            The number of iterations to perform. At least two iterations should be
            performed.  3 iterations is preferred, and will be used if this argument
            is not specified.
        convergence : float, optional
            The fraction of the error corrected each iteration. The default is 0.8.
        error_tolerance : float, optional
            Allowable relative error in the SRS. The default is 0.05.
        tau : float, optional
            If a floating point number is passed, then this will be used for the
            `sine_decay` values.  Alternatively, a dictionary can be passed with
            the keys containing a length-2 tuple specifying the minimum and maximum
            frequency range, and the value specifying the value of `tau` within that
            frequency range.  If this latter approach is used, all `sine_frequencies`
            must be contained within a frequency range. If this argument is not
            specified, then either `sine_decays` or `num_time_constants` must be
            specified instead.
        num_time_constants : int, optional
            If an integer is passed, then this will be used to set the `sine_decay`
            values by ensuring the specified number of time constants occur in the
            `block_size`.  Alternatively, a dictionary can be passed with the keys
            containing a length-2 tuple specifying the minimum and maximum
            frequency range, and the value specifying the value of
            `num_time_constants` over that frequency range. If this latter approach
            is used, all `sine_frequencies` must be contained within a frequency
            range. If this argument is not specified, then either `sine_decays` or
            `tau` must be specified instead.
        decay_resolution : float, optional
            A scalar identifying the resolution of the fractional decay rate
            (often known by the variable `zeta`).  The decay parameters will be
            rounded to this value.  The default is to not round.
        scale_factor : float, optional
            A scaling applied to the sine tone amplitudes so the achieved SRS better
            fits the specified SRS, rather than just touching it. The default is 1.02.
        acceleration_factor : float, optional
            Optional scale factor to convert acceleration into velocity and
            displacement.  For example, if sine amplitudes are in G and displacement
            is desired in inches, the acceleration factor should be set to 386.089.
            If sine amplitudes are in G and displacement is desired in meters, the
            acceleration factor should be set to 9.80665.  The default is 1, which
            assumes consistent units (e.g. acceleration in m/s^2, velocity in m/s,
            displacement in m).
        plot_results : bool, optional
            If True, a figure will be plotted showing the acceleration, velocity,
            and displacement signals, as well as the desired and achieved SRS.
        srs_frequencies : np.ndarray, optional
            If specified, these frequencies will be used to compute the SRS that
            will be plotted when the `plot_results` value is `True`.
        return_velocity : bool, optional
            If specified, a velocity signal will also be returned.  Default is
            False
        return_displacement : bool, optional
            If True, a displacement signal will also be returned.  Default is
            False
        return_srs : bool, optional
            If True, the SRS of the generated signal will also be returned
        return_sine_table : bool, optional
            If True, a sine table will also be returned
        verbose : True, optional
            If True, additional diagnostics will be printed to the console.

        Returns
        -------
        acceleration : TimeHistoryArray
            A TimeHistoryArray object containing an acceleration response that
            satisfies the SRS
        velocity : TimeHistoryArray
            A TimeHistoryArray object containing the velocity corresponding to
            `acceleration`.  Only returned if `return_velocity` is True.
        displacement : TimeHistoryArray
            A TimeHistoryArray object containing the displacement corresponding
            to `acceleration`.  Only returned if `return_displacement` is True.
        srs : TimeHistoryArray
            A `ShockResponseSpectrumArray` containing the SRS of `acceleration`.
            This can be used to check against the original signal to identify
            how good the match is.  Only returned if `return_srs` is True.
        sine_table : DecayedSineTable
            A `DecayedSineTable` object containing the frequency, amplitude,
            delay, and decay parameters that are used to generate `acceleration`.
        """
        try:
            if isinstance(srs_type, str):
                srs_type = ShockResponseSpectrumArray._srs_type_map[srs_type.lower()]
        except KeyError:
            raise ValueError('Invalid `srs_type` specified, should be one of {:} (case insensitive)'.format(
                [k for k in ShockResponseSpectrumArray._srs_type_map]))

        acceleration = np.empty(self.shape+(block_size,))
        if return_displacement:
            displacement = np.empty(self.shape+(block_size,))
        if return_velocity:
            velocity = np.empty(self.shape+(block_size,))
        if return_srs:
            srs = None
        if return_sine_table:
            sine_table = None

        for index, srs_fn in self.ndenumerate():

            if sine_frequencies is None and sine_tone_range is None:
                this_sine_tone_range = [srs_fn.abscissa.min(), srs_fn.abscissa.max()]
            else:
                this_sine_tone_range = sine_tone_range

            srs_breakpoints = np.array((srs_fn.abscissa, srs_fn.ordinate)).T

            (acceleration_signal, velocity_signal, displacement_signal,
             all_frequencies, all_amplitudes, all_decays, all_delays,
             *plot_stuff) = sp_sds(
                 sample_rate, block_size, sine_frequencies,
                 this_sine_tone_range, sine_tone_per_octave, sine_amplitudes,
                 sine_decays, sine_delays, None, srs_breakpoints,
                 srs_damping, srs_type, compensation_frequency,
                 compensation_decay, number_of_iterations, convergence,
                 error_tolerance, tau, num_time_constants, decay_resolution,
                 scale_factor, acceleration_factor, plot_results,
                 srs_frequencies, verbose)

            acceleration[index] = acceleration_signal
            if return_displacement:
                displacement[index] = displacement_signal
            if return_velocity:
                velocity[index] = velocity_signal
            if return_srs:
                # Compute the SRS
                if srs_frequencies is None:
                    this_srs_frequencies = all_frequencies[:-1]
                else:
                    this_srs_frequencies = srs_frequencies
                this_srs, freq = sp_srs(acceleration_signal, 1/sample_rate,
                                        this_srs_frequencies, srs_damping, srs_type)
                if srs is None:
                    srs = np.empty(self.shape+(this_srs.size,))
                srs[index] = this_srs
            if return_sine_table:
                if sine_table is None:
                    sine_table = DecayedSineTable(self.shape, all_frequencies.size)
                sine_table[index] = decayed_sine_table(all_frequencies, all_amplitudes, all_decays, all_delays, srs_fn.coordinate)

        # Now convert to objects
        times = np.arange(block_size)/sample_rate
        acceleration = data_array(FunctionTypes.TIME_RESPONSE,
                                  times, acceleration, self.coordinate,
                                  self.comment1, self.comment2, self.comment3,
                                  self.comment4, self.comment5)
        return_values = (acceleration,)
        if return_velocity:
            velocity = data_array(FunctionTypes.TIME_RESPONSE,
                                  times, displacement, self.coordinate,
                                  self.comment1, self.comment2, self.comment3,
                                  self.comment4, self.comment5)
            return_values += (velocity,)
        if return_displacement:
            displacement = data_array(FunctionTypes.TIME_RESPONSE,
                                      times, displacement, self.coordinate,
                                      self.comment1, self.comment2, self.comment3,
                                      self.comment4, self.comment5)
            return_values += (displacement,)
        if return_srs:
            srs = data_array(FunctionTypes.SHOCK_RESPONSE_SPECTRUM,
                             this_srs_frequencies, srs, self.coordinate,
                             self.comment1, self.comment2, self.comment3,
                             self.comment4, self.comment5)
            return_values += (srs,)
        if return_sine_table:
            return_values += (sine_table,)

        if len(return_values) == 1:
            return_values = return_values[0]

        return return_values

    def plot(self, one_axis: bool = True, subplots_kwargs: dict = {},
             plot_kwargs: dict = {}, abscissa_markers = None, 
             abscissa_marker_labels = None, abscissa_marker_type = 'vline',
             abscissa_marker_plot_kwargs = {}):
        """
        Plot the shock response spectrum

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
        abscissa_markers : ndarray, optional
            Array containing abscissa values to mark on the plot to denote
            significant events.
        abscissa_marker_labels : str or ndarray
            Array of strings to label the abscissa_markers with, or
            alternatively a format string that accepts index and abscissa
            inputs (e.g. '{index:}: {abscissa:0.2f}').  By default no label
            will be applied.
        abscissa_marker_type : str
            The type of marker to use.  This can either be the string 'vline'
            or a valid matplotlib symbol specifier (e.g. 'o', 'x', '.').
        abscissa_marker_plot_kwargs : dict
            Additional keyword arguments used when plotting the abscissa label
            markers.

        Returns
        -------
        axis : matplotlib axis or array of axes
             On which the data were plotted

        """
        if abscissa_markers is not None:
            if abscissa_marker_labels is None:
                abscissa_marker_labels = ['' for value in abscissa_markers]
            elif isinstance(abscissa_marker_labels,str):
                abscissa_marker_labels = [abscissa_marker_labels.format(
                    index = i, abscissa = v) for i,v in enumerate(abscissa_markers)]
                
        if one_axis is True:
            figure, axis = plt.subplots(**subplots_kwargs)
            lines = axis.plot(self.flatten().abscissa.T, self.flatten().ordinate.T, **plot_kwargs)
            axis.set_yscale('log')
            axis.set_xscale('log')
            if abscissa_markers is not None:
                if abscissa_marker_type == 'vline':
                    kwargs = {'color':'k'}
                    kwargs.update(abscissa_marker_plot_kwargs)
                    for value,label in zip(abscissa_markers,abscissa_marker_labels):
                        axis.axvline(value, **kwargs)
                        axis.annotate(label, xy = (value, axis.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                    axis.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                else:
                    for line in lines:
                        x = line.get_xdata()
                        y = line.get_ydata()
                        marker_y = np.interp(abscissa_markers, x, y)
                        kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        axis.plot(abscissa_markers,marker_y,**kwargs)
                        for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                            axis.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
        elif one_axis is False:
            ncols = int(np.floor(np.sqrt(self.size)))
            nrows = int(np.ceil(self.size / ncols))
            figure, axis = plt.subplots(nrows, ncols, **subplots_kwargs)
            for i, (ax, (index, function)) in enumerate(zip(axis.flatten(), self.ndenumerate())):
                lines = ax.plot(function.abscissa.T, function.ordinate.T, **plot_kwargs)
                ax.set_ylabel('/'.join([str(v) for i, v in function.coordinate.ndenumerate()]))
                ax.set_yscale('log')
                ax.set_xscale('log')
                if abscissa_markers is not None:
                    if abscissa_marker_type == 'vline':
                        kwargs = {'color':'k'}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        for value,label in zip(abscissa_markers,abscissa_marker_labels):
                            ax.axvline(value, **kwargs)
                            ax.annotate(label, xy = (value, ax.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                        ax.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                    else:
                        for line in lines:
                            x = line.get_xdata()
                            y = line.get_ydata()
                            marker_y = np.interp(abscissa_markers, x, y)
                            kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                            kwargs.update(abscissa_marker_plot_kwargs)
                            ax.plot(abscissa_markers,marker_y,**kwargs)
                            for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                                ax.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
            for ax in axis.flatten()[i + 1:]:
                ax.remove()
        else:
            axis = one_axis
            lines = axis.plot(self.abscissa.T, self.ordinate.T, **plot_kwargs)
            if abscissa_markers is not None:
                if abscissa_marker_type == 'vline':
                    kwargs = {'color':'k'}
                    kwargs.update(abscissa_marker_plot_kwargs)
                    for value,label in zip(abscissa_markers,abscissa_marker_labels):
                        axis.axvline(value, **kwargs)
                        axis.annotate(label, xy = (value, axis.get_ylim()[0]), rotation = 90, xytext=(4,4),textcoords='offset pixels',ha='left',va='bottom')
                    axis.callbacks.connect('ylim_changed',_update_annotations_to_axes_bottom)
                else:
                    for line in lines:
                        x = line.get_xdata()
                        y = line.get_ydata()
                        marker_y = np.interp(abscissa_markers, x, y)
                        kwargs = {'color':line.get_color(),'marker':abscissa_marker_type,'linewidth':0}
                        kwargs.update(abscissa_marker_plot_kwargs)
                        axis.plot(abscissa_markers,marker_y,**kwargs)
                        for label,mx,my in zip(abscissa_marker_labels,abscissa_markers,marker_y):
                            axis.annotate(label, xy=(mx,my), textcoords='offset pixels', xytext=(4,4), ha='left', va='bottom')
        return axis

def shock_response_spectrum_array(abscissa,ordinate,coordinate,
                                  comment1='',comment2='',
                                  comment3='',comment4='',comment5=''):
    """
    Helper function to create a ShockResponseSpectrumArray object.

    All input arguments to this function are allowed to broadcast to create the
    final data in the ShockResponseSpectrumArray object.

    Parameters
    ----------
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
    obj : ShockResponseSpectrumArray
        The constructed ShockResponseSpectrumArray object
    """
    return data_array(FunctionTypes.SHOCK_RESPONSE_SPECTRUM,abscissa,ordinate,
                      coordinate, comment1,comment2,comment3,comment4,comment5)


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
                            FunctionTypes.SHOCK_RESPONSE_SPECTRUM: ShockResponseSpectrumArray,
                            # FunctionTypes.FINITE_IMPULSE_RESPONSE_FILTER, : ,
                            FunctionTypes.MULTIPLE_COHERENCE: MultipleCoherenceArray,
                            # FunctionTypes.ORDER_FUNCTION, : ,
                            # FunctionTypes.PHASE_COMPENSATION,  : ,
                            FunctionTypes.IMPULSE_RESPONSE_FUNCTION: ImpulseResponseFunctionArray,
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
        obj = cls(shape, nelem, coordinate.shape[-1])
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
        if fn_type_enum not in function_type_dict:
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
    all_coords = coordinate_array(string_array=np.concatenate(
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


class DecayedSineTable(SdynpyArray):
    """Structure for storing sum-of-decayed-sines information
        """

    def __new__(subtype, shape, num_elements, buffer=None, offset=0, strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to __array_finalize__
        data_dtype = [
            ('frequency', 'float64', num_elements),
            ('amplitude', 'float64', num_elements),
            ('decay', 'float64', num_elements),
            ('delay', 'float64', num_elements),
            ('comment1', '<U80'),
            ('comment2', '<U80'),
            ('comment3', '<U80'),
            ('comment4', '<U80'),
            ('comment5', '<U80'),
            ('coordinate', CoordinateArray.data_dtype, (1,)),
        ]
        obj = super(SdynpyArray, subtype).__new__(subtype, shape,
                                                  data_dtype, buffer, offset, strides, order)
        # Finally, we must return the newly created object:
        return obj

    def __getitem__(self, key):
        output = super().__getitem__(key)
        if isinstance(key, str) and key == 'coordinate':
            return output.view(CoordinateArray)
        else:
            return output

    def construct_signal(self, sample_rate, block_size):
        output_abscissa = np.arange(block_size)/sample_rate
        output_ordinate = np.empty(self.shape+(block_size,))
        for index, table in self.ndenumerate():
            signal = sum_decayed_sines_reconstruction(
                table.frequency, table.amplitude, table.decay, table.delay,
                sample_rate, block_size)
            output_ordinate[index] = signal
        return data_array(FunctionTypes.TIME_RESPONSE, output_abscissa,
                          output_ordinate, self.coordinate, self.comment1,
                          self.comment2, self.comment3, self.comment4,
                          self.comment5)

    def construct_velocity(self, sample_rate, block_size, acceleration_factor=1):
        output_abscissa = np.arange(block_size)/sample_rate
        output_ordinate = np.empty(self.shape+(block_size,))
        for index, table in self.ndenumerate():
            signal = sum_decayed_sines_displacement_velocity(
                table.frequency, table.amplitude, table.decay, table.delay,
                sample_rate, block_size, acceleration_factor)[0]
            output_ordinate[index] = signal
        return data_array(FunctionTypes.TIME_RESPONSE, output_abscissa,
                          output_ordinate, self.coordinate, self.comment1,
                          self.comment2, self.comment3, self.comment4,
                          self.comment5)

    def construct_displacement(self, sample_rate, block_size, acceleration_factor=1):
        output_abscissa = np.arange(block_size)/sample_rate
        output_ordinate = np.empty(self.shape+(block_size,))
        for index, table in self.ndenumerate():
            signal = sum_decayed_sines_displacement_velocity(
                table.frequency, table.amplitude, table.decay, table.delay,
                sample_rate, block_size, acceleration_factor)[1]
            output_ordinate[index] = signal
        return data_array(FunctionTypes.TIME_RESPONSE, output_abscissa,
                          output_ordinate, self.coordinate, self.comment1,
                          self.comment2, self.comment3, self.comment4,
                          self.comment5)


def decayed_sine_table(frequency, amplitude, decay, delay, coordinate, comment1='', comment2='', comment3='', comment4='', comment5=''):
    """
    Helper function to create a DecayedSineTable object.

    Parameters
    ----------
    frequency : np.ndarray
        Frequencies of the decaying sine waves
    amplitude : np.ndarray
        Amplitudes of the decaying sine waves.
    decay : np.ndarray
        Damping values of the decaying sine waves.
    delay : np.ndarray
        Delay values of the decaying sine waves.
    coordinate : np.ndarray
        Coordinate information for each of the decaying sine waves.  Must match
        the coordinate shape of a TimeHistoryArray, which means it must have
        shape (...,1)
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
    SineTable :
        A SineTable object containing the specified information

    """
    coordinate = np.atleast_1d(coordinate)
    if coordinate.shape[-1] != 1:
        raise ValueError('`coordinate` must have shape (...,1)')
    *shape, num_elements = frequency.shape
    st = DecayedSineTable(shape, num_elements)
    st.frequency = frequency
    st.amplitude = amplitude
    st.decay = decay
    st.delay = delay
    st.coordinate = coordinate
    st.comment1 = comment1
    st.comment2 = comment2
    st.comment3 = comment3
    st.comment4 = comment4
    st.comment5 = comment5
    return st


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

    def __init__(self, *data_to_plot, **labeled_data_to_plot):
        """
        Create a GUIPlot window to visualize data.

        Multiple datasets can be overlaid by providing additional datasets as
        arguments.  Position arguments will be labelled generically in the
        legend.  Keyword arguments will be labelled with their keywords with
        `_` replaced by ` `.

        Parameters
        ----------
        *data_to_plot : NDDataArray
            Data to visualize.  Data passed by positional argument will be
            labeled generically in the legend
        **labeled_data_to_plot : NDDataArray
            Data to visualize.  Data passed by keyword argument will be
            labeled with its keyword
        abscissa_markers : np.ndarray
            Abscissa values at which markers will be placed.  If not specified,
            no markers will be added.  Markers will be added to all plotted
            curves if this argument is passed.  To add markers to just a
            specific plotted data, pass the argument `abscissa_markers_*` where
            `*` is replaced with either the index of the data that was passed
            via a positional argument, or the keyword of the data that was
            passed via a keyword argument.  Must be passed as a keyword argument.
        abscissa_marker_labels : iterable
            Labels that will be applied to the markers.  If not specified, no
            label will be applied.  If a single string is passed, it will be
            passed to the `.format` method with keyword arguments `index` and
            `abscissa`.  This marker label will be used for all plotted
            curves if this argument is passed.  To add markers to just a
            specific plotted data, pass the argument `abscissa_marker_labels_*` where
            `*` is replaced with either the index of the data that was passed
            via a positional argument, or the keyword of the data that was
            passed via a keyword argument.  Must be passed as a keyword argument.
        abscissa_marker_type : str:
            The type of marker that will be applied.  Can be 'vline' for a
            vertical line across the axis, or it can be a pyqtgraph symbol specifier
            (e.g. 'x', 'o', 'star', etc.) which will be placed on the plotted curves.
            If not specified, a vertical line will be used.
            This marker type will be used for all plotted
            curves if this argument is passed.  To add markers to just a
            specific plotted data, pass the argument `abscissa_marker_type_*` where
            `*` is replaced with either the index of the data that was passed
            via a positional argument, or the keyword of the data that was
            passed via a keyword argument.  Must be passed as a keyword argument.

        Returns
        -------
        None.

        """
        # Parse the dataset arguments
        self._parse_arguments(data_to_plot,labeled_data_to_plot)
        # Now go through and reshape the data so it's all the same size
        first_data = [v for v in self.data_dictionary.values()][0]
        self.data_original_shape = first_data.shape
        self.coordinates = first_data.coordinate
        function_type = first_data.function_type
        for key in self.data_dictionary:
            if not np.all(self.coordinates.flatten() == self.data_dictionary[key].coordinate.flatten()):
                print('Warning: Coordinates not consistent for dataset {:}'.format(key))
            self.data_dictionary[key] = self.data_dictionary[key].flatten()
        super(GUIPlot, self).__init__()
        uic.loadUi(os.path.join(os.path.abspath(os.path.dirname(
            os.path.abspath(__file__))), 'GUIPlot.ui'), self)
        # Set up the table
        for index, fn in first_data.ndenumerate():
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
        # Set up color map
        if self.number_of_datasets == 2:
            self.cm = cm.tab20
            self.cm_mod = 20
        elif self.number_of_datasets == 3:
            # Combine tab20b and tab20c
            tab_b = cm.get_cmap('tab20b', 15)(np.linspace(0, 1, 15))
            tab_c = cm.get_cmap('tab20c', 15)(np.linspace(0, 1, 15))
            self.cm = ListedColormap(np.concatenate((tab_c, tab_b), axis=0))
            self.cm_mod = 30
        elif self.number_of_datasets == 4:
            # Combine tab20b and tab20c
            tab_b = cm.get_cmap('tab20b', 20)(np.linspace(0, 1, 20))
            tab_c = cm.get_cmap('tab20c', 20)(np.linspace(0, 1, 20))
            self.cm = ListedColormap(np.concatenate((tab_c, tab_b), axis=0))
            self.cm_mod = 40
        else:
            self.cm = cm.tab10
            self.cm_mod = 10
        # Adjust the default plotting
        if function_type in [FunctionTypes.GENERAL, FunctionTypes.TIME_RESPONSE,
                             FunctionTypes.COHERENCE, FunctionTypes.AUTOCORRELATION,
                             FunctionTypes.CROSSCORRELATION, FunctionTypes.MODE_INDICATOR_FUNCTION,
                             FunctionTypes.MULTIPLE_COHERENCE]:
            complex_type = ComplexType.REAL
            self.abscissa_log = False
            self.ordinate_log = False
        elif function_type in [FunctionTypes.AUTOSPECTRUM, FunctionTypes.POWER_SPECTRAL_DENSITY,
                               FunctionTypes.ENERGY_SPECTRAL_DENSITY]:
            complex_type = ComplexType.MAGNITUDE
            self.abscissa_log = False
            self.ordinate_log = True
        elif function_type in [FunctionTypes.CROSSSPECTRUM, FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                               FunctionTypes.TRANSMISIBILITY, FunctionTypes.SPECTRUM]:
            complex_type = ComplexType.MAGPHASE
            self.abscissa_log = False
            self.ordinate_log = True
        elif function_type in [FunctionTypes.SHOCK_RESPONSE_SPECTRUM]:
            complex_type = ComplexType.MAGNITUDE
            self.abscissa_log = True
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
        self.update_button.setEnabled(False)
        self.actionOverlay.setEnabled(False)  # TODO Remove when you implement non-overlapping plots
        self.connect_callbacks()
        # Set the first plot
        self.tableWidget.selectRow(0)
        self.setWindowTitle('GUIPlot')
        self.show()

    def _parse_arguments(self,data_to_plot,labeled_data_to_plot):
        self.data_dictionary = {}
        self.marker_data = {}
        for i, dataset in enumerate(data_to_plot):
            self.data_dictionary['Dataset {:}'.format(i+1)] = dataset
        for key, dataset in labeled_data_to_plot.items():
            if key == 'abscissa_markers':
                if not None in self.marker_data:
                    self.marker_data[None] = [None,None,None]
                self.marker_data[None][0] = dataset
            elif key == 'abscissa_marker_labels':
                if not None in self.marker_data:
                    self.marker_data[None] = [None,None,None]
                self.marker_data[None][1] = dataset
            elif key == 'abscissa_marker_type':
                if not None in self.marker_data:
                    self.marker_data[None] = [None,None,None]
                self.marker_data[None][2] = dataset
            elif key[:17] == 'abscissa_markers_':
                marker_key = key.replace('abscissa_markers_','')
                try:
                    marker_key = int(marker_key)
                    marker_key = 'Dataset {:}'.format(marker_key+1)
                except ValueError:
                    marker_key = marker_key.replace('_',' ')
                if not marker_key in self.marker_data:
                    self.marker_data[marker_key] = [None,None,None]
                self.marker_data[marker_key][0] = dataset
            elif key[:23] == 'abscissa_marker_labels_':
                marker_key = key.replace('abscissa_marker_labels_','')
                try:
                    marker_key = int(marker_key)
                    marker_key = 'Dataset {:}'.format(marker_key+1)
                except ValueError:
                    marker_key = marker_key.replace('_',' ')
                if not marker_key in self.marker_data:
                    self.marker_data[marker_key] = [None,None,None]
                self.marker_data[marker_key][1] = dataset
            elif key[:21] == 'abscissa_marker_type_':
                marker_key = key.replace('abscissa_marker_type_','')
                try:
                    marker_key = int(marker_key)
                    marker_key = 'Dataset {:}'.format(marker_key+1)
                except ValueError:
                    marker_key = marker_key.replace('_',' ')
                if not marker_key in self.marker_data:
                    self.marker_data[marker_key] = [None,None,None]
                self.marker_data[marker_key][2] = dataset
            else:
                self.data_dictionary[key.replace('_', ' ')] = dataset
        self.number_of_datasets = len(self.data_dictionary)
        if self.number_of_datasets == 0:
            raise ValueError('At least one dataset must be provided!')

    def connect_callbacks(self):
        """
        Connects the callback functions to events

        Returns
        -------
        None.

        """
        self.tableWidget.itemSelectionChanged.connect(self.selection_changed)
        self.update_button.clicked.connect(self.update)
        self.actionImaginary.triggered.connect(self.set_imaginary)
        self.actionMagnitude.triggered.connect(self.set_magnitude)
        self.actionMagnitude_Phase.triggered.connect(self.set_magnitude_phase)
        self.actionPhase.triggered.connect(self.set_phase)
        self.actionReal.triggered.connect(self.set_real)
        self.actionReal_Imag.triggered.connect(self.set_real_imag)
        self.actionOrdinate_Log.triggered.connect(self.update_ordinate_log)
        self.actionAbscissa_Log.triggered.connect(self.update_abscissa_log)
        self.autoupdate_checkbox.clicked.connect(self.update_checkbox)
        self.linewidth_selector.valueChanged.connect(self.update)
        self.symbolsize_selector.valueChanged.connect(self.update)

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
                    original_index = np.unravel_index(index, self.data_original_shape)
                    for k, (label, dataset) in enumerate(self.data_dictionary.items()):
                        data_entry = dataset[index]
                        legend_entry = '{:} {:} {:}'.format(
                            tuple([str(v) for v in original_index]),
                            data_entry.coordinate, label).replace("'", '')
                        pen = pyqtgraph.mkPen(
                            color=[int(255 * v) for v in self.cm((j * (self.number_of_datasets) + k) % self.cm_mod)],
                            width=self.linewidth_selector.value())
                        plot.plot(x=data_entry.abscissa, y=complex_fn(data_entry.ordinate) *
                                  (180 / np.pi if complex_fn is np.angle else 1), name=legend_entry, pen=pen)
                        # Now handle the markers
                        # First see if there is a matching marker set
                        if label in self.marker_data:
                            marker_abscissa, marker_labels, marker_type = self.marker_data[label]
                        else:
                            marker_abscissa, marker_labels, marker_type = [None,None,None]
                        # Now handle if things are missing
                        if marker_abscissa is None and None in self.marker_data:
                            marker_abscissa = self.marker_data[None][0]
                        if marker_labels is None and None in self.marker_data:
                            marker_labels = self.marker_data[None][1]
                        if marker_type is None and None in self.marker_data:
                            marker_type = self.marker_data[None][2]
                        # Now finally go back to the defaults if they still are
                        # not defined
                        if marker_abscissa is not None: # If marker_abscissa is not defined, we don't plot markers
                            # The default marker is a vertical line
                            if marker_type is None:
                                marker_type = 'vline'
                            # Parse out special values of the marker_labels
                            if isinstance(marker_labels,str):
                                marker_labels = [marker_labels.format(index=i,abscissa=a) for i,a in enumerate(marker_abscissa)]
                            # Now that we have all of the data we can plot it.
                            if marker_type == 'vline':
                                if marker_labels is None:
                                    marker_labels = [None]*len(marker_abscissa)
                                for value,marker_label in zip(marker_abscissa,marker_labels):
                                    vlinepen = pyqtgraph.mkPen(
                                        color=[int(255 * v) for v in self.cm((j * (self.number_of_datasets) + k) % self.cm_mod)],
                                        width=1)
                                    vline = pyqtgraph.InfiniteLine(
                                        value,pen=vlinepen,hoverPen=pen,
                                        label=marker_label,
                                        labelOpts={'position':0.0,'rotateAxis':(1,0),'anchors':[(0,0),(0,1)],
                                                   'color':(0,0,0)})
                                    plot.addItem(vline)
                            else:
                                x=data_entry.abscissa
                                y=complex_fn(data_entry.ordinate) * (180 / np.pi if complex_fn is np.angle else 1)
                                marker_ordinate = np.interp(marker_abscissa, x, y)
                                brush = pyqtgraph.mkBrush(
                                    color=[int(255 * v) for v in self.cm((j * (self.number_of_datasets) + k) % self.cm_mod)])
                                plot.plot(x=marker_abscissa,y=marker_ordinate,pen=None,
                                          symbol=marker_type, symbolSize=self.symbolsize_selector.value(), 
                                          symbolPen=pen,symbolBrush=brush)
                                if marker_labels is not None:
                                    for text_abscissa,text_ordinate,text_label in zip(marker_abscissa,marker_ordinate,marker_labels):
                                        ti = pyqtgraph.TextItem(text_label,anchor=(0,1),color=(0,0,0))
                                        plot.addItem(ti)
                                        # TODO: Remove the log10s when pyqtgraph issue 2166 is fixed
                                        # https://github.com/pyqtgraph/pyqtgraph/issues/2166
                                        ti.setPos(np.log10(text_abscissa) if self.abscissa_log else text_abscissa,
                                                  np.log10(text_ordinate) if self.ordinate_log else text_ordinate)
                        
                plot.setLogMode(self.abscissa_log,
                                False if complex_fn is np.angle else self.ordinate_log)
                if xrange is not None:
                    plot.setXRange(*xrange, padding=0.0)
                plots.append(plot)

    def update_data(self, *data_to_plot, **labeled_data_to_plot):
        # Parse the dataset arguments
        self._parse_arguments(data_to_plot, labeled_data_to_plot)
        # Now go through and reshape the data so it's all the same size
        first_data = [v for v in self.data_dictionary.values()][0]
        self.data_original_shape = first_data.shape
        self.coordinates = first_data.coordinate
        for key in self.data_dictionary:
            if not np.all(self.coordinates.flatten() == self.data_dictionary[key].coordinate.flatten()):
                print('Warning: Coordinates not consistent for dataset {:}'.format(key))
            self.data_dictionary[key] = self.data_dictionary[key].flatten()
        new_coordinate = first_data.coordinate
        if ((self.coordinates.shape != new_coordinate.shape) or
                (np.any(self.coordinates != new_coordinate))):
            # Redo the table
            self.tableWidget.blockSignals(True)
            self.tableWidget.clear()
            self.tableWidget.setRowCount(0)
            for index, fn in first_data.ndenumerate():
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

        # Set up color map
        if self.number_of_datasets == 2:
            self.cm = cm.tab20
            self.cm_mod = 20
        elif self.number_of_datasets == 3:
            # Combine tab20b and tab20c
            tab_b = cm.get_cmap('tab20b', 15)(np.linspace(0, 1, 15))
            tab_c = cm.get_cmap('tab20c', 15)(np.linspace(0, 1, 15))
            self.cm = ListedColormap(np.concatenate((tab_c, tab_b), axis=0))
            self.cm_mod = 30
        elif self.number_of_datsets == 4:
            # Combine tab20b and tab20c
            tab_b = cm.get_cmap('tab20b', 20)(np.linspace(0, 1, 20))
            tab_c = cm.get_cmap('tab20c', 20)(np.linspace(0, 1, 20))
            self.cm = ListedColormap(np.concatenate((tab_c, tab_b), axis=0))
            self.cm_mod = 40
        else:
            self.cm = cm.tab10
            self.cm_mod = 10

        self.update()

    def selection_changed(self):
        """Called when the selected functions is changed"""
        if self.autoupdate_checkbox.isChecked():
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
        if self.autoupdate_checkbox.isChecked():
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
        if self.autoupdate_checkbox.isChecked():
            self.update()

    def update_ordinate_log(self):
        """Updates whether the ordinate should be plotted as log scale"""
        self.ordinate_log = self.actionOrdinate_Log.isChecked()
        if self.autoupdate_checkbox.isChecked():
            self.update()

    def update_checkbox(self):
        """Disables the update button if set to auto-update"""
        self.pushButton.setEnabled(not self.autoupdate_checkbox.isChecked())


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
        if compare_data is not None:
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
        if self.compare_matrix is not None:
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
            if self.compare_matrix is not None:
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
                if self.compare_matrix is not None:
                    compare_color = self.cm(color_index * cm_increment + 1)
                    ax.plot(self.abscissa, compare_fn, color=compare_color)
                ax.set_yscale(scale)
                ax.tick_params(axis='y', colors=color)
            index += 1
#        self.plot_canvas.fig.tight_layout()


frf_from_time_data = TransferFunctionArray.from_time_data
join = NDDataArray.join
