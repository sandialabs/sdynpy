# -*- coding: utf-8 -*-
"""
Data container and automated plotting tools for random vibration tests.

Public API
----------
RandomVibTest
    Container for a random vibration test data set.  Holds time history data,
    cross-power spectral densities, and specification PSDs.  Provides methods
    to compute CPSDs/FRFs and produce a standard suite of diagnostic figures.
optimal_subset
    Identify the time window with the most stable RMS amplitude.
make_multi_figures
    Create paginated grids of subplots for multi-channel data.
dynamic_barh
    Horizontal bar chart with automatic multi-column layout.
response_outside_mask
    Boolean mask of frequency lines where a response exceeds a limit.

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

from ..core.sdynpy_data import (TransferFunctionArray, CoherenceArray, MultipleCoherenceArray, PowerSpectralDensityArray, TimeHistoryArray, NDDataArray)
from ..core.sdynpy_geometry import (Geometry,GeometryPlotter,ShapePlotter)
from ..core.sdynpy_coordinate import CoordinateArray, coordinate_array as sd_coordinate_array
from ..modal.sdynpy_signal_processing_gui import SignalProcessingGUI
from ..fileio.sdynpy_rattlesnake import RattlesnakeData, RattlesnakeRandomEnvironmentData
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator
import os
import netCDF4 as nc4
import numpy as np
import pandas as pd
import sdynpy as sdpy
import warnings
import math
from scipy.stats import kurtosis

# Define Functions
def optimal_subset(time_data: TimeHistoryArray,num_subset_samples: int, amplitude_factor: float=0.5):
    """Find the start index of the time subset with most stable amplitude.

    Searches for a contiguous window of ``num_subset_samples`` samples that
    minimises the standard deviation of RMS amplitude across channels while
    remaining above a minimum amplitude threshold.

    Parameters
    ----------
    time_data : TimeHistoryArray
        Time-domain data whose ordinate has shape ``(n_channels, n_samples)``.
    num_subset_samples : int
        Length of the sliding window in samples.
    amplitude_factor : float, optional
        Fraction of the peak-to-trough mean range used as the minimum
        acceptable window mean amplitude.  Windows below this threshold are
        penalised.  Default is 0.5.

    Returns
    -------
    int
        Sample index where the optimal subset begins.
    """
    ordinate_rms = np.sqrt(np.mean(time_data.ordinate**2,axis=0))

    # Compute Cumulative Sums
    cumsum1 = np.concatenate([[0],np.cumsum(ordinate_rms)])
    cumsum2 = np.concatenate([[0],np.cumsum(ordinate_rms**2)])

    # Compute Variance and Standard Deviation
    window_mean = (cumsum1[num_subset_samples:] - cumsum1[:-num_subset_samples]) / num_subset_samples
    window_variance = (cumsum2[num_subset_samples:] - cumsum2[:-num_subset_samples]) / num_subset_samples - window_mean**2
    window_std = np.sqrt(window_variance)

    # Add threashold for mean data
    threshold = (max(window_mean) - min(window_mean))*amplitude_factor + min(window_mean)
    score = np.where(window_mean > threshold, window_std, np.inf)

    # Define Window Start
    start_index = np.argmin(score)

    return start_index

def make_multi_figures(total_plots, max_cols=3, max_subplots_per_fig=18,xlabel='Frequency (Hz)',ylabel='|PSD| (EU$^2$/Hz)',fig_size_x=8.5,fig_size_y=11):
    """Create one or more figures, each containing a grid of subplots.

    Parameters
    ----------
    total_plots : int
        Total number of subplots to create across all figures.
    max_cols : int, optional
        Maximum number of subplot columns per figure.  Default is 3.
    max_subplots_per_fig : int, optional
        Maximum subplots per figure before a new figure is started.  Default
        is 18.
    xlabel : str, optional
        Label for the x-axis of each subplot.  Default is ``'Frequency (Hz)'``.
    ylabel : str, optional
        Label for the y-axis of each subplot.  Default is ``'|PSD| (EU$^2$/Hz)'``.
    fig_size_x : float, optional
        Figure width in inches.  Default is 8.5.
    fig_size_y : float, optional
        Figure height in inches.  Default is 11.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        List of created figures.
    all_axes : list of list of matplotlib.axes.Axes
        Nested list where ``all_axes[i]`` contains the axes for ``figs[i]``.
    """
    figs = []
    all_axes = []
    
    start_idx = 0

    while start_idx < total_plots:
        # Determine how many plots in the figure
        end_idx = min(start_idx + max_subplots_per_fig,total_plots)
        n_plots = end_idx - start_idx
        
        cols = min(max_cols,n_plots)
        rows = math.ceil(n_plots/cols)
        max_rows = math.ceil(max_subplots_per_fig/max_cols)

        fig = plt.figure(figsize=(fig_size_x,fig_size_y))
        gs = fig.add_gridspec(max_rows, max_cols)

        axes = []
        global_head = None

        # Create subplots into the GridSpec
        for i in range(n_plots):
            r = i // cols
            c = i % cols

            # First Subplot becomes the global head
            if global_head is None:
                ax = fig.add_subplot(gs[r,c])
                global_head = ax
            else:
                ax = fig.add_subplot(gs[r,c], sharex=global_head, sharey=global_head)

            axes.append(ax)
            
            # Set global axis labels on bottom left subplot
            if c == 0:
                ax.set_ylabel(ylabel)
                ax.tick_params(labelleft=True)
            else:
                ax.tick_params(labelleft=False)
                # ax.set_yticklabels([])

            if r == rows - 1:
                ax.set_xlabel(xlabel)
                ax.tick_params(labelbottom=True)
            else:
                ax.tick_params(labelbottom=False)

        for i in range(n_plots, max_rows * max_cols):
            fig.add_subplot(gs[i // max_cols, i % max_cols]).set_visible(False)

        figs.append(fig)
        all_axes.append(axes)

        start_idx += n_plots

    return figs, all_axes

def dynamic_barh(values, labels=None, max_per_col=35):
    """Create a horizontal bar chart with automatic multi-column layout.

    Parameters
    ----------
    values : array_like
        Numeric values to plot as horizontal bars.
    labels : CoordinateArray or None, optional
        Labels for the y-axis ticks.  If a ``CoordinateArray`` is provided the
        tick labels are set accordingly.  Default is ``None``.
    max_per_col : int, optional
        Maximum number of bars per column before a new column is added.
        Default is 35.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : list of matplotlib.axes.Axes
        List of axes, one per column.
    """
    n = len(values)

    # ---- Determine number of columns ----
    cols = math.ceil(n / max_per_col)
    rows = 1

    # ---- Split values into chunks for each column ----
    chunks = [
        values[i * max_per_col : (i + 1) * max_per_col]
        for i in range(cols)
    ]

    if isinstance(labels,CoordinateArray):
        label_chunks = [
            labels[i * max_per_col : (i + 1) * max_per_col]
            for i in range(cols)
        ]
    else:
        label_chunks = [None] * cols

    # ---- Automatic figure height based on the tallest column ----
    h_per_bar = 0.35
    base_height = 1.0
    tallest_col = max(len(chunk) for chunk in chunks)
    fig_height = base_height + h_per_bar * tallest_col

    # Wider figure for more columns
    fig_width = 6 * cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_width, fig_height),
        sharex=True
    )

    if cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # ---- Plot each column ----
    for ax, chunk, lbl_chunk in zip(axes, chunks, label_chunks):
        y = range(len(chunk))
        ax.barh(y, chunk,color='deepskyblue',edgecolor='black', linewidth=1)

        if isinstance(lbl_chunk,CoordinateArray):
            ax.set_yticks(y)
            ax.set_yticklabels(lbl_chunk)
        else:
            ax.set_yticks(y)

        ax.invert_yaxis()
        ax.margins(y=0)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True)

        # Apply shared y-limits
        ax.set_ylim(axes[0].get_ylim())

    # Remove unused axes (shouldn't happen but safe)
    for ax in axes[len(chunks):]:
        ax.remove()

    fig.tight_layout()
    return fig, axes

def response_outside_mask(response, response_limit, mask_type: str = 'above'):
    """Compute a boolean mask of where response is outside the specified limit.

    Parameters
    ----------
    response : NDDataArray
        The measured response data array.
    response_limit : NDDataArray
        The limit data array to compare against.
    mask_type : str, optional
        Direction of comparison: 'above' to flag where response exceeds the
        limit, 'below' to flag where response falls below the limit.
        Default is 'above'.

    Returns
    -------
    np.ndarray
        Boolean array that is True where the response is outside the limit.

    Raises
    ------
    ValueError
        If mask_type is not 'above' or 'below'.
    """
    response_in_limit_mask = np.isin(response.abscissa, response_limit.abscissa)
    limit_in_response_mask = np.isin(response_limit.abscissa, response.abscissa)
    response_exsists_mask = np.invert(np.isnan(response.ordinate))
    limit_exists_mask = np.invert(np.isnan(response_limit.ordinate))

    if mask_type == 'above':
        outside_mask = np.logical_and(np.where(response_in_limit_mask, np.abs(np.abs(response.ordinate)), 0) > np.where(limit_in_response_mask, np.abs(np.abs(response_limit.ordinate)), 0), np.where(response_in_limit_mask, np.abs(response.ordinate), 0) > np.where(limit_in_response_mask, np.abs(response_limit.ordinate), 0))
    elif mask_type == 'below':
        outside_mask = np.logical_and(np.where(response_in_limit_mask, np.abs(np.abs(response.ordinate)), 0) < np.where(limit_in_response_mask, np.abs(np.abs(response_limit.ordinate)), 0), np.where(response_in_limit_mask, np.abs(response.ordinate), 0) < np.where(limit_in_response_mask, np.abs(response_limit.ordinate), 0))
    else:
        raise ValueError(f"mask_type must be 'above' or 'below', got '{mask_type}'")

    outside_mask = np.logical_and.reduce((outside_mask, response_in_limit_mask, response_exsists_mask, limit_exists_mask))

    return outside_mask

def _shade_out_of_limit(ax, freqs, mask, limit_ordinate, ylim_bound, color, alpha, label):
    """Shade from the abort limit curve to the plot edge for out-of-limit regions.

    For each contiguous run of True values in *mask*, builds an x array that
    extends half the local frequency spacing beyond each edge so that even a
    single isolated spectral line produces a visible shaded band.  The y extent
    runs from the abort limit ordinate (extended at constant value to the padded
    edges) to *ylim_bound*, preserving the original fill-between appearance.
    Works correctly for non-uniform (e.g. octave-band) frequency spacing.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the shading.
    freqs : np.ndarray, shape (N,)
        Frequency abscissa values corresponding to *mask*.
    mask : np.ndarray of bool, shape (N,)
        True at each frequency line that is outside the limit.
    limit_ordinate : np.ndarray, shape (N,)
        Abort limit ordinate values at each frequency in *freqs*.
    ylim_bound : float
        The plot-edge y value to shade toward (``min(ylim)`` for below,
        ``max(ylim)`` for above).
    color : str
        Patch fill color.
    alpha : float
        Patch transparency.
    label : str
        Legend label applied to the first patch; subsequent patches use
        ``'_nolegend_'`` to avoid duplicate legend entries.
    """
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return

    # Split into contiguous groups
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    groups = np.split(indices, breaks)

    labeled = False
    for group in groups:
        i0, i1 = int(group[0]), int(group[-1])

        # Left edge: midpoint to the previous line; use right-side spacing at boundary
        if i0 > 0:
            x0 = freqs[i0] - (freqs[i0] - freqs[i0 - 1]) / 2
        else:
            x0 = freqs[i0] - (freqs[i0 + 1] - freqs[i0]) / 2 if len(freqs) > 1 else freqs[i0]

        # Right edge: midpoint to the next line; use left-side spacing at boundary
        if i1 < len(freqs) - 1:
            x1 = freqs[i1] + (freqs[i1 + 1] - freqs[i1]) / 2
        else:
            x1 = freqs[i1] + (freqs[i1] - freqs[i1 - 1]) / 2 if len(freqs) > 1 else freqs[i1]

        # Build x array with padded endpoints; extend limit ordinate at constant value
        x_vals = np.concatenate([[x0], freqs[i0:i1 + 1], [x1]])
        y_limit = np.abs(np.concatenate([[limit_ordinate[i0]], limit_ordinate[i0:i1 + 1], [limit_ordinate[i1]]]))

        ax.fill_between(x_vals, y_limit, ylim_bound, color=color, alpha=alpha,
                        edgecolor='none', label=label if not labeled else '_nolegend_')
        labeled = True

# class TransientVibTest:

# class RandomVibTest:
class RandomVibTest:
    """Container for a random vibration test data set with automated plotting.

    Holds time history data, cross-power spectral densities, specifications,
    and data for a random vibration test environment.  Provides methods to
    compute CPSDs/FRFs and produce a standard suite of diagnostic plots.

    Parameters
    ----------
    source_data : RattlesnakeRandomEnvironmentData, optional
        Parsed Rattlesnake environment data for the test environment.
    geometry : str or Geometry, optional
        Path to a geometry file or a ``Geometry`` object for 3-D visualization.
    coordinate : CoordinateArray, optional
        Full channel coordinate array (all channels).
    control_coordinate : CoordinateArray, optional
        Subset of coordinates for control/reference channels.
    response_coordinate : CoordinateArray, optional
        Subset of coordinates for response channels.
    excitation_coordinate : CoordinateArray, optional
        Subset of coordinates for excitation (drive) channels.
    time_data : list of TimeHistoryArray, optional
        List of time history arrays, one per data set.
    cpsd : list of PowerSpectralDensityArray or None, optional
        Pre-computed cross-power spectral densities.  Pass ``None`` (default)
        to have them computed on demand.
    specification_cpsd : PowerSpectralDensityArray, optional
        Target PSD specification.
    specification_warning_psd : PowerSpectralDensityArray, optional
        Warning-level tolerance bands around the specification.
    specification_abort_psd : PowerSpectralDensityArray, optional
        Abort-level tolerance bands around the specification.
    averages : int, optional
        Number of spectral averages used when computing CPSDs.
    overlap : float, optional
        Fractional overlap between frames (0 to 1).
    fft_lines : int, optional
        Number of FFT lines.
    window : str, optional
        Name of the windowing function (e.g. ``'hann'``).
    samples_per_frame : int, optional
        Number of samples per FFT frame.
    start_time : float or None, optional
        Start time (seconds) for the analysis window within each data set.
        ``None`` triggers automatic selection via ``optimal_subset``.
    units : np.ndarray, optional
        String array of engineering-unit labels, one per channel.
    frf : list of TransferFunctionArray or None, optional
        Pre-computed frequency response functions.
    oct_order : int, optional
        Octave band order for octave-averaged results.
    """
    def __init__(self,
                 source_data: RattlesnakeRandomEnvironmentData = None,
                 geometry: str | Geometry = None,
                 coordinate: CoordinateArray = None,
                 control_coordinate: CoordinateArray = None,
                 response_coordinate: CoordinateArray = None,
                 excitation_coordinate: CoordinateArray = None,
                 time_data: list[TimeHistoryArray] = None,
                 cpsd: list[PowerSpectralDensityArray] = None,
                 specification_cpsd: PowerSpectralDensityArray = None,
                 specification_warning_psd: PowerSpectralDensityArray = None,
                 specification_abort_psd: PowerSpectralDensityArray = None,
                 averages: int = None,
                 overlap: float = None,
                 fft_lines: int = None,
                 window: str = None,
                 samples_per_frame: int = None,
                 start_time: float = None,
                 units: np.ndarray = None,
                 frf: list[TransferFunctionArray] = None,
                 oct_order: int = None,
                 ):
        """Store all test data attributes.

        Every named parameter is assigned directly to ``self``.  See the class
        docstring for parameter descriptions.  ``cpsd`` and ``frf`` default to
        empty lists when ``None`` is passed, and the specification CPSD is
        sanitised so that zero values are replaced with ``NaN``.
        """
        self.source_data = source_data
        self._geometry = None
        self.geometry = geometry
        self.coordinate = coordinate
        self.control_coordinate = control_coordinate
        self.response_coordinate = response_coordinate
        self.excitation_coordinate = excitation_coordinate
        self._time_data_truncated = None
        self._time_data = None
        self._start_time = None
        self.time_data = time_data
        self.start_time = start_time
        self.cpsd = cpsd if cpsd is not None else []
        self.specification_cpsd = specification_cpsd
        self.specification_warning_psd = specification_warning_psd
        self.specification_abort_psd = specification_abort_psd
        self.averages = averages
        self.overlap = overlap
        self.fft_lines = fft_lines
        self.window = window
        self.samples_per_frame = samples_per_frame
        self.units = units
        self.frf = frf if frf is not None else []
        self.oct_order = oct_order

        # Sanitize the specification so 0 values go to NaN
        if self.specification_cpsd is not None:
            zero_mask = np.abs(self.specification_cpsd.ordinate)==0
            self.specification_cpsd.ordinate[zero_mask] = np.nan

    @property
    def geometry(self):
        """Geometry object for 3-D visualisation.

        Returns
        -------
        Geometry or None
            The loaded geometry object, or ``None`` if none has been set.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, arg):
        """Set the geometry from a file path, ``Geometry`` object, or ``None``.

        Parameters
        ----------
        arg : str, Geometry, or None
            If a string, the geometry is loaded from that file path.  If a
            ``Geometry`` instance it is stored directly.  Any other value
            (including ``None``) is stored as-is.
        """
        if isinstance(arg, str):
            self._geometry = Geometry.load(arg)
        elif isinstance(arg, Geometry):
            self._geometry = arg
        else:
            self._geometry = arg

    @property
    def time_data(self):
        """List of full-length ``TimeHistoryArray`` objects, one per data set.

        Returns
        -------
        list of TimeHistoryArray or None
        """
        return self._time_data

    @time_data.setter
    def time_data(self, value):
        """Set time data and invalidate any cached truncated arrays.

        Parameters
        ----------
        value : list of TimeHistoryArray or None
            New time data list.
        """
        self._time_data = value
        self._time_data_truncated = None

    @property
    def start_time(self):
        """List of analysis-window start times in seconds, one per data set.

        A value of ``None`` for any element triggers automatic selection via
        :func:`optimal_subset` when the truncated data is first requested.

        Returns
        -------
        list of float or None
        """
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        """Set start times and invalidate any cached truncated arrays.

        Parameters
        ----------
        value : list of float or None
            New start-time list.
        """
        self._start_time = value
        self._time_data_truncated = None

    @property
    def time_data_truncated(self):
        """Time data windowed to the analysis subset used for CPSD computation.

        Lazily computed on first access and cached.  Each element is extracted
        from the corresponding full ``TimeHistoryArray`` beginning at
        ``start_time`` and covering exactly
        ``samples_per_frame * averages - overlap_samples * (averages - 1)``
        samples.  When ``start_time`` is ``None`` for a data set, the optimal
        start index is found via :func:`optimal_subset` and the corresponding
        ``start_time`` entry is updated in-place.

        Returns
        -------
        list of TimeHistoryArray
            One truncated ``TimeHistoryArray`` per data set.
        """
        if self._time_data_truncated is None:
            result = []
            if self.start_time is None:
                self.start_time = [None]*len(self.time_data)
            for index, (start_time, time_data) in enumerate(zip(self.start_time, self.time_data)):
                avg_dt_per_channel = np.mean(np.diff(time_data.abscissa, axis=1), axis=1)
                dt = np.mean(avg_dt_per_channel)
                samples_overlap = int(self.samples_per_frame * self.overlap)
                samples_total = self.samples_per_frame * self.averages - samples_overlap * (self.averages - 1)
                if start_time is None:
                    start_index = optimal_subset(time_data, samples_total)
                    self._start_time[index] = start_index * dt
                result.append(time_data.extract_elements_by_abscissa(self._start_time[index], self._start_time[index] + samples_total * dt))
            self._time_data_truncated = result
        return self._time_data_truncated

    def set_tolerance_limit_psd(self, dB=6):
        """Set the abort-tolerance PSD from the specification using a dB tolerance.

        Computes lower and upper bounds as
        ``specification_asd / 10^(dB/10)`` and
        ``specification_asd * 10^(dB/10)`` respectively, then stores them
        as ``self.specification_abort_psd``.  Only the abort band is updated;
        ``specification_warning_psd`` is not modified.

        Parameters
        ----------
        dB : float, optional
            Tolerance in decibels above and below the specification.
            Default is 6 dB.

        Raises
        ------
        TypeError
            If ``specification_cpsd`` is not a ``PowerSpectralDensityArray``.
        """
        if isinstance(self.specification_cpsd, PowerSpectralDensityArray):
            lower_limit = self.specification_cpsd.get_asd() / 10**(dB / 10)
            upper_limit = self.specification_cpsd.get_asd() * 10**(dB / 10)
            self.specification_abort_psd = np.concatenate((lower_limit[np.newaxis, :], upper_limit[np.newaxis, :]))
        else:
            raise TypeError("specification_cpsd must be a PowerSpectralDensityArray to set tolerance limits")

    def compute_cpsd(self):
        """Compute cross-power spectral densities for each time data set.

        Computes ASDs for each truncated time data segment and appends them to
        ``self.cpsd``.  Already-computed entries (``PowerSpectralDensityArray``
        instances) are left unchanged; only ``None`` or non-array entries are
        recomputed.
        """
        # Ensure the list is long enough
        while len(self.cpsd) < len(self.time_data):
            self.cpsd.append(None)

        for index, time_data_truncated in enumerate(self.time_data_truncated):
            if not isinstance(self.cpsd[index], PowerSpectralDensityArray):
                self.cpsd[index] = time_data_truncated.cpsd(
                    samples_per_frame=self.samples_per_frame,
                    overlap=self.overlap,
                    window=self.window.lower(),
                    averages_to_keep=self.averages,
                    only_asds=True
                )

    def plot_cpsd_time_subset(self, save_path=None):
        """Plot the selected time subset used for CPSD computation.

        Parameters
        ----------
        save_path : str or None, optional
            If provided, the figure is saved to this path. If there are
            multiple time data sets the data set number is appended to the
            filename stem.  Default is ``None`` (figure is not saved).
        """
        for index in range(len(self.time_data)):
            with plt.rc_context({'path.simplify': True, 'path.simplify_threshold': 1.0,
                                 'agg.path.chunksize': 10000}):
                fig, ax = plt.subplots(figsize=(12, 6))
                ordinate_rms = np.sqrt(np.mean(self.time_data[index].ordinate**2, axis=0))
                ordinate_rms_truncated = np.sqrt(np.mean(self.time_data_truncated[index].ordinate**2, axis=0))
                ax.plot(self.time_data[index][0].abscissa, ordinate_rms, label='Full Time Set', color='darkgrey')
                ax.plot(self.time_data_truncated[index][0].abscissa, ordinate_rms_truncated, label='Selected Time Subset', color='deepskyblue')
                ax.set_title('Subsection of Time Data')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('RMS Amplitude Across All Channels')
                ax.legend()
                plt.tight_layout()

                if save_path is not None:
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    if len(self.time_data) == 1:
                        save_name = save_path
                    else:
                        save_name = os.path.splitext(save_path)[0] + ' (Data Set ' + str(index + 1) + ')' + os.path.splitext(save_path)[-1]
                    fig.savefig(save_name, dpi=600, bbox_inches='tight')

    def compute_frf(self):
        """Compute FRFs for each time data set and append them to ``self.frf``.

        For each truncated time data segment, computes H1 frequency response
        functions between the excitation (reference) channels and all response
        channels using the same windowing parameters as CPSD computation.
        Each resulting ``TransferFunctionArray`` is appended to ``self.frf``.

        Excitation channels are identified by a non-empty ``feedback_channel``
        entry in the source channel table, matching the logic used in
        ``RattlesnakeData.excitation_coordinate``.
        """
        for index, time_data_truncated in enumerate(self.time_data_truncated):
            reference_time_data = time_data_truncated[self.excitation_coordinate[:, np.newaxis]]
            response_time_data = time_data_truncated[self.response_coordinate[:, np.newaxis]]
            self.frf.append(sdpy.data.frf_from_time_data(
                reference_data=reference_time_data,
                response_data=response_time_data,
                samples_per_average=self.samples_per_frame,
                overlap=self.overlap,
                method='H1',
                window=self.window.lower()
            ))

    def plot_control_vs_spec(self, save_path: None|str = None):
        """Plot measured control PSD against the specification for each control channel.

        For each data set, creates a grid of subplots — one per control channel
        — showing the abort-limit band (filled grey), warning-limit band
        (light grey), specification (dashed black), and measured response
        (blue).  Out-of-limit regions are shaded and a percentage-out-of-limit
        annotation is added to each subplot.  The in-band RMS level is shown
        as a text annotation.

        Parameters
        ----------
        save_path : str or None, optional
            Base file path for saving figures.  The figure index and (when
            multiple data sets exist) data set index are appended to the stem.
            Default is ``None`` (figures are not saved).

        Returns
        -------
        figures : list of list of matplotlib.figure.Figure
            ``figures[i]`` is the list of figures for data set *i*.
        axes : list of list of list of matplotlib.axes.Axes
            ``axes[i]`` is the list of per-figure axes groups for data set *i*.
            ``axes[i][j]`` is the list of ``Axes`` objects on figure *j*.
        """
        figures = [None]*len(self.time_data)
        axes = [None]*len(self.time_data)

        # If the CPSD is not defined, compute it automatically
        if any(not isinstance(cpsd, PowerSpectralDensityArray) for cpsd in self.cpsd):
            self.compute_cpsd()

        control_coordinates = np.concatenate((self.control_coordinate[:,np.newaxis],self.control_coordinate[:,np.newaxis]),axis=1)
        for index,cpsd in enumerate(self.cpsd):
            xlim = NDDataArray.get_abscissa_limits([cpsd[control_coordinates],self.specification_cpsd.get_drive_points(),self.specification_warning_psd,self.specification_abort_psd])
            ylim = NDDataArray.get_ordinate_limits([cpsd[control_coordinates],self.specification_cpsd.get_drive_points(),self.specification_warning_psd,self.specification_abort_psd],xlim)

            np.max(np.abs(cpsd[control_coordinates].extract_elements_by_abscissa(min(xlim),max(xlim)).ordinate))

            # Plot CPSD Results for Control Channels
            all_figs, all_axes = make_multi_figures(len(self.control_coordinate))

            coordinate_index = 0
            figure_index = 0
            for fig, axes_group in zip(all_figs,all_axes):
                for ax in axes_group:
                    coordinate = self.control_coordinate[coordinate_index]
                    handles = []
                    labels = []

                    # Plot Abort Limit
                    if isinstance(self.specification_abort_psd,PowerSpectralDensityArray) and np.any(~np.isnan(self.specification_abort_psd.ordinate)):
                        abscissa = self.specification_abort_psd[0][coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).abscissa
                        ordinate1 = self.specification_abort_psd[-1][coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).ordinate
                        ordinate2 = self.specification_abort_psd[-0][coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).ordinate
                        ax.fill_between(abscissa,np.abs(ordinate1),np.abs(ordinate2), alpha=1, facecolor='lightgrey',edgecolor='darkgrey',linewidth=1)
                        ax.collections[-1].set_label('Abort Limit')

                    # Plot Warning Limit
                    if isinstance(self.specification_warning_psd,PowerSpectralDensityArray) and np.any(~np.isnan(self.specification_warning_psd.ordinate)):
                        abscissa = self.specification_warning_psd[0][coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).abscissa
                        ordinate1 = self.specification_warning_psd[-1][coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).ordinate
                        ordinate2 = self.specification_warning_psd[-0][coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).ordinate
                        ax.fill_between(abscissa,np.abs(ordinate1),np.abs(ordinate2), alpha=0.2, facecolor='lightgrey',edgecolor='darkgrey',linewidth=1)
                        ax.collections[-1].set_label('Warning Limit')
                    
                    # Plot Specification
                    if isinstance(self.specification_cpsd,PowerSpectralDensityArray) and np.any(~np.isnan(self.specification_cpsd.ordinate)):
                        abscissa = self.specification_cpsd[coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).abscissa
                        ordinate = self.specification_cpsd[coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).ordinate
                        ax.plot(abscissa,np.abs(ordinate),color='black',linestyle='--',linewidth=1)
                        ax.get_lines()[-1].set_label('Specification')

                    # Plot Control
                    if isinstance(cpsd,PowerSpectralDensityArray):
                        abscissa = cpsd[coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).abscissa
                        ordinate = cpsd[coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).ordinate
                        ax.plot(abscissa,np.abs(ordinate),color='deepskyblue',linewidth=1)
                        ax.get_lines()[-1].set_label('Control')

                    # Shade Area Outside the Abort Limit if Response Exceeds the Abort Limit
                    if isinstance(cpsd,PowerSpectralDensityArray) and isinstance(self.specification_abort_psd,PowerSpectralDensityArray) and np.any(~np.isnan(self.specification_abort_psd.ordinate)):
                        response_too_low_mask = response_outside_mask(cpsd[coordinate],self.specification_abort_psd[0][coordinate],mask_type='below')
                        response_too_high_mask = response_outside_mask(cpsd[coordinate],self.specification_abort_psd[1][coordinate],mask_type='above')

                        _shade_out_of_limit(ax, cpsd[coordinate].abscissa, response_too_low_mask,
                                            self.specification_abort_psd[0][coordinate].ordinate, min(ylim),
                                            color='blue', alpha=0.2, label='Response Below Tolerance')
                        _shade_out_of_limit(ax, cpsd[coordinate].abscissa, response_too_high_mask,
                                            self.specification_abort_psd[-1][coordinate].ordinate, max(ylim),
                                            color='red', alpha=0.2, label='Response Above Tolerance')

                        response_inside_bandwidth_mask = np.logical_and(cpsd[coordinate].abscissa>=min(xlim),cpsd[coordinate].abscissa<=max(xlim))
                        response_outside_tolerance_mask = np.logical_or(response_too_low_mask,response_too_high_mask)
                        response_inside_bandwith_and_outside_tolerance = np.logical_and(response_inside_bandwidth_mask,response_outside_tolerance_mask)
                        percent_out = sum(response_inside_bandwith_and_outside_tolerance)/sum(response_inside_bandwidth_mask)*100

                        ax.text(0.99, 0.01, 'Out: ' + str(percent_out.round(2)) + '%',transform=ax.transAxes,ha="right", va="bottom", fontsize=8)

                    # Compute RMS Level of Response within the x-limits
                    rms_level = cpsd[coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).rms(self.oct_order)

                    # Set Plot Properties
                    ax.set_yscale('log')

                    ax.set_title(str(coordinate))
                    units = self.units[np.intersect1d(self.coordinate,coordinate,return_indices=True)[1]][0]
                    ax.text(0.01, 0.01, 'RMS: ' + str(rms_level.round(2)) + ' ' + units,transform=ax.transAxes,ha="left", va="bottom", fontsize=8)
                    ax.set_xlim(min(xlim),max(xlim))
                    ax.set_ylim(min(ylim),max(ylim))
                    ax.set_axisbelow(True)

                    # ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=12))
                    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=12))
                    ax.grid(True,which='major',ls='-',color='grey',zorder=0, linewidth=0.2)
                    ax.grid(True,which='minor',ls=':',color='grey',zorder=0, linewidth=0.2)

                    coordinate_index += 1
                
                    h, l = ax.get_legend_handles_labels()
                    handles.extend(h)
                    labels.extend(l)
                
                unique = dict(zip(labels,handles))

                fig.legend(unique.values(),unique.keys(),loc='lower center',ncol=min([len(unique),3]))
                fig.suptitle('Control vs. Specification PSD')

                # Save Picture
                if save_path is not None:
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    if len(self.time_data) == 1:
                        save_name = os.path.splitext(save_path)[0] + ' (Figure ' + str(figure_index + 1) + ')' + os.path.splitext(save_path)[-1]
                    else:
                        save_name = os.path.splitext(save_path)[0] + ' (Data Set ' + str(index + 1) + ', Figure ' + str(figure_index + 1) + ')' + os.path.splitext(save_path)[-1]
                    fig.savefig(save_name,dpi=600 ,bbox_inches='tight')

                figure_index += 1
            figures[index] = all_figs
            axes[index] = all_axes
        return figures, axes

    def plot_response(self, save_path: None|str = None, one_figure=False):
        """Plot the measured response PSD for each response channel.

        For each data set, creates a grid of subplots — one per response
        channel — showing the measured ASD (blue) on a log scale.  The
        in-band RMS level is annotated on each subplot.

        Parameters
        ----------
        save_path : str or None, optional
            Base file path for saving figures.  The figure index and (when
            multiple data sets exist) data set index are appended to the stem.
            Default is ``None`` (figures are not saved).

        one_figure : bool, optional
            Whether successive datasets should be overlayed on the same grid of 
            figures, or new grids should be made for each dataset
            Default is `False`

        Returns
        -------
        figures : list of list of matplotlib.figure.Figure
            ``figures[i]`` is the list of figures for data set *i*.
        axes : list of list of list of matplotlib.axes.Axes
            ``axes[i]`` is the nested list of axis groups for data set *i*.
        """
        figures = [None]*len(self.time_data)
        axes = [None]*len(self.time_data)

        # If the CPSD is not defined, compute it automatically
        if all(not isinstance(cpsd, PowerSpectralDensityArray) for cpsd in self.cpsd):
            self.compute_cpsd()

        response_coordinates = np.concatenate((self.response_coordinate[:,np.newaxis],self.response_coordinate[:,np.newaxis]),axis=1)
        for index,cpsd in enumerate(self.cpsd):
            xlim = NDDataArray.get_abscissa_limits(cpsd[response_coordinates])
            ylim = NDDataArray.get_ordinate_limits(cpsd[response_coordinates],xlim)

            # Plot CPSD Results for Response Channels
            if (index == 0 and one_figure) or (not one_figure):
                all_figs, all_axes = make_multi_figures(len(self.response_coordinate))

            coordinate_index = 0
            figure_index = 0
            for fig, axes_group in zip(all_figs,all_axes):
                for ax in axes_group:
                    coordinate = self.response_coordinate[coordinate_index]
                    # for fig in all_figs:
                    handles = []
                    labels = []

                    # Plot Response
                    if isinstance(cpsd,PowerSpectralDensityArray):
                        abscissa = cpsd[coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).abscissa
                        ordinate = cpsd[coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).ordinate
                        ax.plot(abscissa,np.abs(ordinate),color=None if one_figure else 'deepskyblue',linewidth=1)

                    # Compute RMS Level of Response within the x-limits
                    rms_level = cpsd[coordinate].extract_elements_by_abscissa(min(xlim),max(xlim)).rms(self.oct_order)

                    # Set Plot Properties
                    ax.set_yscale('log')
                    ax.set_title(str(coordinate))
                    units = self.units[np.intersect1d(self.coordinate,coordinate,return_indices=True)[1]][0]
                    if not one_figure:
                        ax.text(0.01, 0.01, 'RMS: ' + str(rms_level.round(2)) + ' ' + units,transform=ax.transAxes,ha="left", va="bottom")

                    ax.set_xlim(min(xlim),max(xlim))
                    ax.set_ylim(min(ylim),max(ylim))
                    ax.set_axisbelow(True)
                    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=12))
                    ax.grid(True,which='major',ls='-',color='grey',zorder=0, linewidth=0.2)
                    ax.grid(True,which='minor',ls=':',color='grey',zorder=0, linewidth=0.2)
                
                    coordinate_index += 1
                
                fig.suptitle('Response PSD')

                # Save Picture
                if save_path is not None:
                    if (one_figure and index==len(self.cpsd)-1) or (not one_figure):
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        if len(self.time_data) == 1 or one_figure:
                            save_name = os.path.splitext(save_path)[0] + ' (Figure ' + str(figure_index + 1) + ')' + os.path.splitext(save_path)[-1]
                        else:
                            save_name = os.path.splitext(save_path)[0] + ' (Data Set ' + str(index + 1) + ', Figure ' + str(figure_index + 1) + ')' + os.path.splitext(save_path)[-1]
                        fig.savefig(save_name,dpi=600 ,bbox_inches='tight')

                figure_index += 1
            figures[index] = all_figs
            axes[index] = all_axes
        return figures, axes

    def plot_percent_lines_out(self, save_path: None|str = None):
        """Plot the percentage of spectral lines out of tolerance for each control channel.

        For each data set, computes the fraction of frequency lines within the
        analysis bandwidth where the measured response falls outside the abort
        tolerance limits, then displays the result as a horizontal bar chart.
        Bars exceeding a 10 % threshold are coloured red.  The fraction of
        channels that exceed the threshold is shown in the figure title.

        Parameters
        ----------
        save_path : str or None, optional
            Base file path for saving figures.  The data set index is appended
            to the stem when multiple data sets exist.  Default is ``None``
            (figures are not saved).

        Returns
        -------
        figures : list of matplotlib.figure.Figure
            One figure per data set.
        axes : list of list of matplotlib.axes.Axes
            One list of axes per data set.
        """
        figures = [None]*len(self.time_data)
        axes = [None]*len(self.time_data)

        # If the CPSD is not defined, compute it automatically
        if all(not isinstance(cpsd, PowerSpectralDensityArray) for cpsd in self.cpsd):
            self.compute_cpsd()

        control_coordinates = np.concatenate((self.control_coordinate[:,np.newaxis],self.control_coordinate[:,np.newaxis]),axis=1)

        for index,cpsd in enumerate(self.cpsd):
            xlim = NDDataArray.get_abscissa_limits([cpsd[control_coordinates],self.specification_cpsd.get_drive_points(),self.specification_warning_psd,self.specification_abort_psd])

            # Plot Percent Lines Out
            response_too_low_mask = response_outside_mask(cpsd[control_coordinates],self.specification_abort_psd[0][control_coordinates],mask_type='below')
            response_too_high_mask = response_outside_mask(cpsd[control_coordinates],self.specification_abort_psd[1][control_coordinates],mask_type='above')

            response_inside_bandwidth_mask = np.logical_and(cpsd[control_coordinates].abscissa>=min(xlim),cpsd[control_coordinates].abscissa<=max(xlim))
            response_outside_tolerance_mask = np.logical_or(response_too_low_mask,response_too_high_mask)
            response_inside_bandwith_and_outside_tolerance = np.logical_and(response_inside_bandwidth_mask,response_outside_tolerance_mask)
            percent_out = sum(response_inside_bandwith_and_outside_tolerance.T)/sum(response_inside_bandwidth_mask.T)*100

            threshold = 10

            percent_channels_out = sum(percent_out>threshold)/len(percent_out)*100

            fig,axes_group = dynamic_barh(values=percent_out,labels=self.control_coordinate)
            for ax in axes_group:
                bars = ax.patches
                for bar in bars:
                    if bar.get_width() > threshold:
                        bar.set_facecolor('red')
                ax.axvspan(0, threshold, alpha=0.6, facecolor='lightgrey',edgecolor='darkgrey',linewidth=1,zorder=0)
                ax.set_xlabel('Percent Lines Out')
            fig.suptitle('Percent Lines Out (' + str(percent_channels_out.round(2)) + r'% of Channels)')
            fig.tight_layout()

            # Save Picture
            if save_path is not None:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                if len(self.time_data) == 1:
                    save_name = save_path
                else:
                    save_name = os.path.splitext(save_path)[0] + ' (Data Set ' + str(index + 1) + ')' + os.path.splitext(save_path)[-1]
                fig.savefig(save_name,dpi=600 ,bbox_inches='tight')
            
            figures[index] = fig
            axes[index] = axes_group
        return figures, axes

    def plot_rms_level(self, save_path: None|str = None):
        """Plot the broadband RMS level for each response channel.

        RMS values are computed from the CPSD restricted to the analysis
        bandwidth defined by the specification, warning, and abort PSDs.
        Results are displayed as a horizontal bar chart.

        Parameters
        ----------
        save_path : str or None, optional
            Base file path for saving figures.  The data set index is appended
            to the stem when multiple data sets exist.  Default is ``None``
            (figures are not saved).

        Returns
        -------
        figures : list of matplotlib.figure.Figure
            One figure per data set.
        axes : list of list of matplotlib.axes.Axes
            One list of axes per data set.
        """
        figures = [None]*len(self.time_data)
        axes = [None]*len(self.time_data)

        # If the CPSD is not defined, compute it automatically
        if all(not isinstance(cpsd, PowerSpectralDensityArray) for cpsd in self.cpsd):
            self.compute_cpsd()

        response_coordinates = np.concatenate((self.response_coordinate[:,np.newaxis],self.response_coordinate[:,np.newaxis]),axis=1)
        for index,cpsd in enumerate(self.cpsd):
            xlim = NDDataArray.get_abscissa_limits([cpsd,self.specification_cpsd.get_drive_points(),self.specification_warning_psd,self.specification_abort_psd])

            # Get Response Response Level
            rms_response = cpsd[response_coordinates].extract_elements_by_abscissa(min(xlim),max(xlim)).rms(oct_order=self.oct_order)

            # Plot RMS Response Level
            fig,axis = dynamic_barh(values=rms_response,labels=self.response_coordinate)

            for ax in axis:
                ax.set_xlabel('RMS Level (EU)')
            fig.suptitle('RMS Response Level')
            fig.tight_layout()

            # Save Picture
            if save_path is not None:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                if len(self.time_data) == 1:
                    save_name = save_path
                else:
                    save_name = os.path.splitext(save_path)[0] + ' (Data Set ' + str(index + 1) + ')' + os.path.splitext(save_path)[-1]
                fig.savefig(save_name,dpi=600 ,bbox_inches='tight')

            figures[index] = fig
            axes[index] = axis
        return figures,axes

    def plot_rms_error(self, save_path: None|str = None):
        """Plot the broadband RMS error between response and specification for each control channel.

        The RMS error is computed as the difference between the measured RMS
        level and the specification RMS level, both restricted to the analysis
        bandwidth.  Results are displayed as a horizontal bar chart.

        Parameters
        ----------
        save_path : str or None, optional
            Base file path for saving figures.  The data set index is appended
            to the stem when multiple data sets exist.  Default is ``None``
            (figures are not saved).

        Returns
        -------
        figures : list of matplotlib.figure.Figure
            One figure per data set.
        axes : list of list of matplotlib.axes.Axes
            One list of axes per data set.
        """
        figures = [None]*len(self.time_data)
        axes = [None]*len(self.time_data)

        control_coordinates = np.concatenate((self.control_coordinate[:,np.newaxis],self.control_coordinate[:,np.newaxis]),axis=1)

        # If the CPSD is not defined, compute it automatically
        if all(not isinstance(cpsd, PowerSpectralDensityArray) for cpsd in self.cpsd):
            self.compute_cpsd()

        for index,cpsd in enumerate(self.cpsd):
            xlim = NDDataArray.get_abscissa_limits([cpsd[control_coordinates],self.specification_cpsd.get_drive_points(),self.specification_warning_psd,self.specification_abort_psd])

            # Get Response PSDs
            response_psds = cpsd[control_coordinates].extract_elements_by_abscissa(min(xlim),max(xlim))
            specification_psds = self.specification_cpsd[control_coordinates].extract_elements_by_abscissa(min(xlim),max(xlim))

            # Integrate the PSDs over the frequency band to get power in EU
            rms_response = response_psds.rms(self.oct_order)
            rms_spec = specification_psds.rms(self.oct_order)

            # Compute the RMS error between the two signals
            rms_error = rms_response - rms_spec

            # Plot RMS Error
            fig,axis = dynamic_barh(values=rms_error,labels=self.control_coordinate)

            for ax in axis:
                ax.set_xlabel('RMS Error (EU)')
            fig.suptitle('RMS Error Per Channel')
            fig.tight_layout()

            # Save Picture
            if save_path is not None:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                if len(self.time_data) == 1:
                    save_name = save_path
                else:
                    save_name = os.path.splitext(save_path)[0] + ' (Data Set ' + str(index + 1) + ')' + os.path.splitext(save_path)[-1]
                fig.savefig(save_name,dpi=600 ,bbox_inches='tight')
            
            figures[index] = fig
            axes[index] = axis
        return figures,axes

    def plot_kurtosis(self, save_path: None|str = None):
        """Plot the kurtosis of each response channel time history.

        Kurtosis is computed on the truncated (analysis-window) portion of
        each data set using Fisher's definition disabled (Pearson kurtosis,
        where Gaussian has kurtosis = 3).  Bars that fall outside the range
        ``[3 - 1, 3 + 1]`` are coloured red.

        Parameters
        ----------
        save_path : str or None, optional
            Base file path for saving figures.  The data set index is appended
            to the stem when multiple data sets exist.  Default is ``None``
            (figures are not saved).

        Returns
        -------
        figures : list of matplotlib.figure.Figure
            One figure per data set.
        axes : list of list of matplotlib.axes.Axes
            One list of axes per data set.
        """
        figures = [None]*len(self.time_data)
        axes = [None]*len(self.time_data)
        for index,time_data_truncated in enumerate(self.time_data_truncated):            
            # Remove Excitation Coordinate Data to leave just response Data
            control_indices = time_data_truncated.response_coordinate.find_indices(self.excitation_coordinate)[0][0]
            time_data_truncated_indices = np.arange(len(time_data_truncated.response_coordinate))
            response_time_data = time_data_truncated[np.setdiff1d(time_data_truncated_indices,control_indices)]

            # Plot Kurtosis
            response_kurtosis = kurtosis(response_time_data.ordinate.T,fisher=False,bias=True)

            fig,axis = dynamic_barh(values=response_kurtosis,labels=self.response_coordinate)

            figures[index] = fig
            axes[index] = axis

            tolerance = 1

            for ax in axis:
                bars = ax.patches
                for bar in bars:
                    if bar.get_width() > 3+tolerance or bar.get_width() < 3-tolerance:
                        bar.set_facecolor('red')
                ax.axvspan(3-tolerance, 3+tolerance, alpha=0.6, facecolor='lightgrey',edgecolor='darkgrey',linewidth=1,zorder=0)
                ax.set_xlabel('Kurtosis')
                ax.set_xlim([0,3+tolerance])
            fig.suptitle('Kurtosis')
            fig.tight_layout()

            # Save Picture
            if save_path is not None:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                if len(self.time_data) == 1:
                    save_name = save_path
                else:
                    save_name = os.path.splitext(save_path)[0] + ' (Data Set ' + str(index + 1) + ')' + os.path.splitext(save_path)[-1]
                fig.savefig(save_name,dpi=600 ,bbox_inches='tight')

        return figures,axes

    def plot_signal_to_noise(self, save_path: None|str = None):
        """Plot the signal-to-noise ratio for a two-dataset (noise + signal) test.

        Requires exactly two data sets: the first is treated as the noise
        measurement and the second as the signal measurement.  SNR is computed
        as ``10 * log10(signal_power / noise_power)`` where power is the
        integral of the ASD over the full frequency range.  Bars with SNR
        below 10 dB are coloured red.

        Parameters
        ----------
        save_path : str or None, optional
            File path for saving the figure.  Default is ``None`` (figure is
            not saved).

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The figure, or ``None`` if ``time_data`` does not contain exactly
            two data sets.
        axis : list of matplotlib.axes.Axes or None
            The axes list, or ``None`` as above.
        """
        if len(self.time_data) == 2:
            # If the CPSD is not defined, compute it automatically
            if all(not isinstance(cpsd, PowerSpectralDensityArray) for cpsd in self.cpsd):
                self.compute_cpsd()
            
            # Calculate spacing along the frequency axis (last axis)
            diffs = np.diff(self.cpsd[-1].abscissa, axis=-1) # Shape: (M, N-1)
            
            # Create the inner widths (averages of adjacent gaps)
            # Shape: (M, N-2)
            inner_df = (diffs[:, :-1] + diffs[:, 1:]) / 2
            
            # Concatenate the boundary conditions
            # First col: same as first diff
            # Last col: same as last diff
            df = np.column_stack([
                diffs[:, [0]],   # Twice half-distance to next
                inner_df,        # Centered inner points
                diffs[:, [-1]]   # Twice half-distance from previous
            ])

            # Calculate Signal Power
            signal_power = np.sum(self.cpsd[-1].ordinate.real*df,axis=1)
            noise_power = np.sum(self.cpsd[0].ordinate.real*df,axis=1)

            # Calculate Signal to Noise Ratio and Convert to dB
            signal_to_noise_ratios = 10*np.log10(signal_power / noise_power)

            fig,axis = dynamic_barh(values=signal_to_noise_ratios,labels=self.coordinate)

            for ax in axis:
                bars = ax.patches
                for bar in bars:
                    if bar.get_width() < 10:
                        bar.set_facecolor('red')
                ax.axvspan(10, max(signal_to_noise_ratios), alpha=0.6, facecolor='lightgrey',edgecolor='darkgrey',linewidth=1,zorder=0)
                ax.set_xlabel('Signal to Noise Ratio (dB)')
                ax.set_xlim(0,max(signal_to_noise_ratios))
            fig.suptitle('Signal to Noise Ratio')
            fig.tight_layout()

            # Save Picture
            if save_path is not None:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                save_name = save_path
                fig.savefig(save_name,dpi=600 ,bbox_inches='tight')

            return fig,axis

    def plot_time_histories(self, save_path: None|str = None):
        """Plot the truncated time history for each channel.

        For each data set, creates a grid of subplots — one per channel in
        the truncated ``TimeHistoryArray`` — showing the time-domain signal.
        Path simplification is enabled for performance on large records.

        Parameters
        ----------
        save_path : str or None, optional
            Base file path for saving figures.  The figure index and (when
            multiple data sets exist) data set index are appended to the stem.
            Default is ``None`` (figures are not saved).

        Returns
        -------
        figures : list of list of matplotlib.figure.Figure
            ``figures[i]`` is the list of figures for data set *i*.
        axes : list of list of list of matplotlib.axes.Axes
            ``axes[i]`` is the nested list of axis groups for data set *i*.
        """
        figures = [None]*len(self.time_data)
        axes = [None]*len(self.time_data)

        for index,time_data_truncated in enumerate(self.time_data_truncated):
            xlim = NDDataArray.get_abscissa_limits(time_data_truncated)
            ylim = NDDataArray.get_ordinate_limits(time_data_truncated,xlim)

            # Plot Time Data for Response Channels — enable path simplification for large signals
            with plt.rc_context({'path.simplify': True, 'path.simplify_threshold': 1.0,
                                 'agg.path.chunksize': 10000}):
                all_figs, all_axes = make_multi_figures(len(time_data_truncated),xlabel='Time (s)',ylabel='EU')

                coordinate_index = 0
                figure_index = 0
                for fig, axes_group in zip(all_figs,all_axes):
                    for ax in axes_group:
                        coordinate = np.squeeze(time_data_truncated.coordinate)[coordinate_index]
                        handles = []
                        labels = []

                        # Plot Response
                        ax.plot(time_data_truncated[coordinate].abscissa,time_data_truncated[coordinate].ordinate,color='deepskyblue',linewidth=1)

                        units = self.units[np.intersect1d(self.coordinate,coordinate,return_indices=True)[1]][0]
                        ax.text(0.99, 0.01, 'EU: ' + units,transform=ax.transAxes,ha="right", va="bottom")

                        # Set Plot Properties
                        ax.set_title(str(coordinate))
                        ax.set_xlim(min(xlim),max(xlim))
                        ax.set_ylim(-max(np.abs(ylim)),max(np.abs(ylim)))

                        coordinate_index += 1

                    fig.suptitle('Time Histories')

                    # Save Picture
                    if save_path is not None:
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        if len(self.time_data) == 1:
                            save_name = os.path.splitext(save_path)[0] + ' (Figure ' + str(figure_index + 1) + ')' + os.path.splitext(save_path)[-1]
                        else:
                            save_name = os.path.splitext(save_path)[0] + ' (Data Set ' + str(index + 1) + ', Figure ' + str(figure_index + 1) + ')' + os.path.splitext(save_path)[-1]
                        fig.savefig(save_name,dpi=600 ,bbox_inches='tight')

                    figure_index += 1
            figures[index] = all_figs
            axes[index] = all_axes
        return figures, axes

    def plot_multiple_coherence(self, save_path: None|str = None):
        """Plot multiple coherence for each response channel within the analysis bandwidth.

        For each data set, computes the multiple coherence between every
        response channel and all excitation channels (using the same windowing
        parameters as CPSD computation), truncates to the specification
        bandwidth, then produces a grid of subplots with one panel per
        response channel.  The y-axis is fixed to ``[0, 1]``.

        Parameters
        ----------
        save_path : str or None, optional
            Base file path for saving figures.  The figure index and (when
            multiple data sets exist) data set index are appended to the stem.
            Default is ``None`` (figures are not saved).

        Returns
        -------
        figures : list of list of matplotlib.figure.Figure
            ``figures[i]`` is the list of figures for data set *i*.
        axes : list of list of list of matplotlib.axes.Axes
            ``axes[i]`` is the nested list of axis groups for data set *i*.
        """

        # If the CPSD is not defined, compute it automatically
        if all(not isinstance(cpsd, PowerSpectralDensityArray) for cpsd in self.cpsd):
            self.compute_cpsd()

        figures = [None]*len(self.time_data)
        axes = [None]*len(self.time_data)

        for index,(time_data_truncated,cpsd) in enumerate(zip(self.time_data_truncated,self.cpsd)):
            if self.specification_cpsd is not None:
                xlim = NDDataArray.get_abscissa_limits([self.specification_cpsd.get_drive_points(),self.specification_warning_psd,self.specification_abort_psd])
            else:
                xlim = NDDataArray.get_abscissa_limits(self.cpsd)

            # Get Time Data
            time_data_response = time_data_truncated[self.response_coordinate[:,np.newaxis]]
            time_data_reference = time_data_truncated[self.excitation_coordinate[:,np.newaxis]]

            # Compute Coherence
            multiple_coherence = sdpy.data.MultipleCoherenceArray.from_time_data(response_data=time_data_response,samples_per_average=self.samples_per_frame,overlap=self.overlap,window=self.window.lower(),reference_data=time_data_reference)

            # Truncate Frequency to Bandwith Analyzed
            if not any(lim is None for lim in xlim):
                multiple_coherence = multiple_coherence.extract_elements_by_abscissa(min(xlim),max(xlim))

            # Plot Multiple Coherence Results for Response Channels
            all_figs, all_axes = make_multi_figures(len(self.response_coordinate),ylabel='Coherence')

            coordinate_index = 0
            figure_index = 0
            for fig, axes_group in zip(all_figs,all_axes):
                for ax in axes_group:
                    coordinate = self.response_coordinate[coordinate_index]

                    # Plot Multiple Coherence
                    abscissa = multiple_coherence[coordinate].abscissa
                    ordinate = multiple_coherence[coordinate].ordinate
                    ax.plot(abscissa,ordinate,color='deepskyblue',linewidth=1)

                    # Set Plot Properties
                    ax.set_title(str(coordinate))
                    if any(lim is None for lim in xlim):
                        ax.set_xlim(min(abscissa),max(abscissa))
                    else:
                        ax.set_xlim(min(xlim),max(xlim))
                    ax.set_ylim(0,1)
                
                    coordinate_index += 1
                
                fig.suptitle('Multiple Coherence')

                # Save Picture
                if save_path is not None:
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    if len(self.time_data) == 1:
                        save_name = os.path.splitext(save_path)[0] + ' (Figure ' + str(figure_index + 1) + ')' + os.path.splitext(save_path)[-1]
                    else:
                        save_name = os.path.splitext(save_path)[0] + ' (Data Set ' + str(index + 1) + ', Figure ' + str(figure_index + 1) + ')' + os.path.splitext(save_path)[-1]
                    fig.savefig(save_name,dpi=600 ,bbox_inches='tight')

                figure_index += 1
            figures[index] = all_figs
            axes[index] = all_axes
        return figures, axes

    def create_all_plots(self, figure_root_path='Figures'):
        """Generate and save all standard diagnostic plots to disk.

        Computes CPSDs if they have not already been computed, saves a time-
        subset plot, then iterates over all standard plot methods, saving each
        figure to a labelled subdirectory under *figure_root_path*.  All
        matplotlib figures are closed after each method to free memory.

        The plot methods called (and their subfolder names) are:

        - ``plot_control_vs_spec`` -> ``Control vs Specification PSD Plot``
        - ``plot_response`` -> ``Response PSD Plot``
        - ``plot_percent_lines_out`` -> ``Percent Lines Out Plot``
        - ``plot_rms_level`` -> ``RMS Plot``
        - ``plot_rms_error`` -> ``RMS Error Plot``
        - ``plot_kurtosis`` -> ``Kurtosis Plot``
        - ``plot_time_histories`` -> ``Time Histories Plot``
        - ``plot_multiple_coherence`` -> ``Multiple Coherence Plot``
        - ``plot_signal_to_noise`` -> ``Signal to Noise Plot``

        Parameters
        ----------
        figure_root_path : str, optional
            Root directory under which all figure subdirectories are created.
            Default is ``'Figures'``.
        """
        # If the CPSD is not defined, compute it automatically
        if all(not isinstance(cpsd, PowerSpectralDensityArray) for cpsd in self.cpsd):
            self.compute_cpsd()
            self.plot_cpsd_time_subset(save_path=figure_root_path + os.sep + 'Analyzed Section of Time' + os.sep + 'Analyzed Section of Time Plot')
        
        # Specify Folder Names for Each Set of Plots
        # Keys are Method Names
        # Values are a sub-dictionary with keys 'plot_folder' and 'plot_name'
        path_and_plot_names = {
            'plot_control_vs_spec':{'plot_folder':'Control vs Specification PSD Plot','plot_name':'Control vs Specification PSD Plot'},
            'plot_response':{'plot_folder':'Response PSD Plot','plot_name':'Response PSD Plot'},
            'plot_percent_lines_out':{'plot_folder':'Percent Lines Out Plot','plot_name':'Percent Lines Out Plot'},
            'plot_rms_level':{'plot_folder':'RMS Plot','plot_name':'RMS Plot'},
            'plot_rms_error':{'plot_folder':'RMS Error Plot','plot_name':'RMS Error Plot'},
            'plot_kurtosis':{'plot_folder':'Kurtosis Plot','plot_name':'Kurtosis Plot'},
            'plot_time_histories':{'plot_folder':'Time Histories Plot','plot_name':'Time Histories Plot'},
            'plot_multiple_coherence':{'plot_folder':'Multiple Coherence Plot','plot_name':'Multiple Coherence Plot'},
            'plot_signal_to_noise':{'plot_folder':'Signal to Noise Plot','plot_name':'Signal to Noise Plot'},
            }

        # Create Plots for Each Plot Method
        for method_name in path_and_plot_names:
            method = getattr(self,method_name)
            method(save_path = figure_root_path + os.sep + path_and_plot_names[method_name]['plot_folder'] + os.sep + path_and_plot_names[method_name]['plot_name'])
            plt.close('all')

    def convert_to_octave(self, oct_order = 6):
        """Convert all stored PSDs to octave-band averages.

        Restricts the analysis to the specification bandwidth, computes nominal
        octave-band centres for the given *oct_order*, then replaces the stored
        CPSDs, specification CPSD, warning PSD, and abort PSD with their
        octave-band equivalents using ``bandwidth_average``.  Sets
        ``self.oct_order`` so that subsequent RMS computations use octave
        integration.

        Parameters
        ----------
        oct_order : int, optional
            Octave-band order (e.g. 3 for third-octave, 6 for sixth-octave).
            Default is 6.
        """
        self.oct_order = oct_order

        # Compute CPSDs if they don't already exsist
        if all(not isinstance(cpsd, PowerSpectralDensityArray) for cpsd in self.cpsd):
                self.compute_cpsd()

        # Get Analysis Bandwidth
        xlim = NDDataArray.get_abscissa_limits([self.specification_cpsd.get_drive_points(),self.specification_warning_psd,self.specification_abort_psd])

        # Get Frequency Spacing
        nominal_band_centers,band_lb,band_ub,band_centers = sdpy.cpsd.nth_octave_freqs(freq=xlim,oct_order=oct_order)

        # Convert to Octave Space
        for index,cpsd in enumerate(self.cpsd):
            self.cpsd[index] = cpsd.extract_elements_by_abscissa(min(xlim),max(xlim)).bandwidth_average(band_lb,band_ub)

        self.specification_cpsd = self.specification_cpsd.extract_elements_by_abscissa(min(xlim),max(xlim)).bandwidth_average(band_lb,band_ub)
        self.specification_warning_psd = self.specification_warning_psd.extract_elements_by_abscissa(min(xlim),max(xlim)).bandwidth_average(band_lb,band_ub)
        self.specification_abort_psd = self.specification_abort_psd.extract_elements_by_abscissa(min(xlim),max(xlim)).bandwidth_average(band_lb,band_ub)

    @classmethod
    def load_rattlesnake_streaming_data(cls, file: os.PathLike | RattlesnakeData,
                                        environment_name: str = None,
                                        # time_array_index: int = 0,
                                        geometry: os.PathLike | Geometry = None):
        """Construct a ``RandomVibTest`` from a Rattlesnake streaming nc4 file.

        Reads (or reuses) a ``RattlesnakeData`` object, selects the
        requested environment, and populates a new ``RandomVibTest`` with all
        time data, coordinate arrays, specification PSDs, and spectral
        processing parameters drawn from the file.

        Parameters
        ----------
        file : os.PathLike or RattlesnakeData
            Path to a Rattlesnake nc4 file, or an already-parsed
            ``RattlesnakeData`` object.
        environment_name : str, optional
            Name of the environment group to load.  When ``None`` (default)
            the first environment in the file is used.  A warning is emitted
            when the file contains more than one environment and none is
            specified.
        geometry : os.PathLike or Geometry, optional
            Path to a geometry file or a ``Geometry`` object for 3-D
            visualisation.  Default is ``None``.

        Returns
        -------
        RandomVibTest
            Fully initialised ``RandomVibTest`` instance ready for CPSD
            computation and plotting.

        Warns
        -----
        UserWarning
            If the file contains multiple environments and *environment_name*
            is not specified.
        """
        if not isinstance(file, RattlesnakeData):
            data = RattlesnakeData.read_rattlesnake_nc4(file)
        else:
            data = file

        if environment_name is None:
            environment_name = next(iter(data.environments))
            if len(data.environments) > 1:
                warnings.warn('There are multiple environments in the Rattlesnake data and none were specified. Using the first environmnet named: ' + environment_name + ' .')

        return cls(
            source_data = data.environments[environment_name],
            geometry = geometry,
            coordinate = data.get_coordinate(env_name=environment_name),
            control_coordinate = data.environments[environment_name].control_coordinate,
            response_coordinate = data.response_coordinate,
            excitation_coordinate = data.excitation_coordinate,
            time_data = data.get_time_data(env_name=environment_name),
            start_time = [None]*len(data.get_time_data(env_name=environment_name)),
            specification_cpsd = data.environments[environment_name].specification_cpsd,
            specification_warning_psd = data.environments[environment_name].specification_warning_psd,
            specification_abort_psd = data.environments[environment_name].specification_abort_psd,
            averages = data.environments[environment_name].sysid_averages,
            overlap = data.environments[environment_name].cpsd_overlap,
            fft_lines = data.environments[environment_name].fft_lines,
            window = data.environments[environment_name].cpsd_window,
            samples_per_frame = data.environments[environment_name].samples_per_frame,
            cpsd = [],
            units = data.units(env_name=environment_name),
            oct_order = None,
        )