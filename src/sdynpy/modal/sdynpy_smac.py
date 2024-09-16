# -*- coding: utf-8 -*-
"""
Implementation of the Synthesize Modes and Correlate (SMAC) curve fitter for Python

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
from ..core.sdynpy_data import TransferFunctionArray, GUIPlot
from ..core.sdynpy_geometry import Geometry
from ..core.sdynpy_shape import ShapeArray
from ..signal_processing.sdynpy_correlation import mac, matrix_plot
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter
from .sdynpy_modeshape import compute_residues as modeshape_compute_residues, compute_shapes as modeshape_compute_shapes, ShapeSelection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from enum import Enum

from qtpy import QtWidgets, uic, QtGui
from qtpy.QtWidgets import QApplication, QMainWindow
import pyqtgraph
pyqtgraph.setConfigOption('background', 'w')
pyqtgraph.setConfigOption('foreground', 'k')


def correlation_coefficient(x, y, axis=-1):
    mx = np.mean(x, axis=axis, keepdims=True)
    my = np.mean(y, axis=axis, keepdims=True)
    xn = x - mx
    yn = y - my
    return np.sum(xn * yn, axis=axis) / np.sqrt(np.sum(xn**2, axis=axis) * np.sum(yn**2, axis=axis))


class ConvergenceException(Exception):
    pass


class AutoFitTypes(Enum):
    PARABOLOID = 0
    ALTERNATE = 1


class SMAC:
    def __init__(self, frfs: TransferFunctionArray, min_frequency=None, max_frequency=None, complex_modes=False,
                 displacement_derivative=2):
        self.frfs = frfs.reshape_to_matrix()
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        abscissa_indices = np.ones(self.frequencies.shape, dtype=bool)
        if min_frequency is not None:
            abscissa_indices &= (self.frequencies >= min_frequency)
        if max_frequency is not None:
            abscissa_indices &= (self.frequencies <= max_frequency)
        abscissa = self.frequencies[abscissa_indices]
        freq_range = np.array((np.min(abscissa), np.max(abscissa)))
        index_range = np.argmin(np.abs(self.frequencies - freq_range[:, np.newaxis]), axis=1)
        self.frequency_slice = slice(index_range[0], index_range[1] + 1)
        self.displacement_derivative = displacement_derivative
        # Other information
        self.pinv_frfs = None
        self.complex_modes = complex_modes
        self.initial_rootlist = None
        self.rootlist = None
        self.residuals = None
        self.initial_correlation = None

    def save(self, filename):
        np.savez(filename, frfs=self.frfs,
                 frequency_slice=self.frequency_slice,
                 pinv_frfs=self.pinv_frfs,
                 complex_modes=self.complex_modes,
                 initial_rootlist=self.initial_rootlist,
                 rootlist=self.rootlist)

    @property
    def frequencies(self):
        return self.frfs[0, 0].abscissa

    @property
    def angular_frequencies(self):
        return 2 * np.pi * self.frequencies

    @property
    def reference_coordinates(self):
        return self.frfs[0, :].reference_coordinate

    @property
    def frequency_spacing(self):
        return np.mean(np.diff(self.frequencies))

    def __repr__(self):
        if self.pinv_frfs is None:
            next_step = "Compute pseudoinverse with compute_pseudoinverse"
        elif self.initial_rootlist is None:
            next_step = "Compute initial rootlist with compute_initial_rootlist"
        elif self.rootlist is None:
            next_step = "Iterate on the initial rootlist with autofit_roots"
        else:
            next_step = 'Analysis Finished'
        return 'SMAC analysis of {:}.  Next step: {:}'.format(self.frfs.__repr__(), next_step)

    def compute_pseudoinverse(self):
        ordinate_for_pinv = self.frfs.ordinate[..., self.frequency_slice].transpose(1, 2, 0)
        if self.complex_modes:
            self.pinv_frfs = np.linalg.pinv(ordinate_for_pinv)
        else:
            self.pinv_frfs = np.linalg.pinv(np.concatenate(
                (ordinate_for_pinv.real, ordinate_for_pinv.imag), axis=1))

    def compute_correlation_matrix(self, low_frequency=None, high_frequency=None,
                                   frequency_samples=None, frequency_resolution=None,
                                   low_damping=0.0025, high_damping=0.05, damping_samples=21,
                                   frequency_lines_for_correlation=20, plot=False):
        if self.pinv_frfs is None:
            raise RuntimeError(
                'Pseudoinverse must be calculated first by calling compute_pseudoinverse')
        if low_frequency is None:
            low_frequency = self.frequencies[self.frequency_slice].min()
        if high_frequency is None:
            high_frequency = self.frequencies[self.frequency_slice].max()
        if frequency_samples is None:
            if frequency_resolution is None:
                frequency_samples = int(
                    np.round((high_frequency - low_frequency) / self.frequency_spacing)) + 1
            else:
                frequency_samples = int(
                    np.round((high_frequency - low_frequency) / frequency_resolution)) + 1
        correlation_frequencies = np.linspace(low_frequency, high_frequency, frequency_samples)
        correlation_dampings = np.linspace(low_damping, high_damping, damping_samples)
        # Set up for a big broadcast
        omega_frequency_line = self.angular_frequencies[self.frequency_slice, np.newaxis]
        omega_correlation_frequencies = 2 * np.pi * \
            correlation_frequencies[:, np.newaxis, np.newaxis, np.newaxis]
        zeta = correlation_dampings[:, np.newaxis, np.newaxis]
        frfs_sdof = ((1j * omega_frequency_line)**self.displacement_derivative /
                     (omega_correlation_frequencies**2
                      + 1j * 2 * zeta * omega_frequency_line * omega_correlation_frequencies
                      - omega_frequency_line**2))
        if self.complex_modes:
            psi = self.pinv_frfs[:, np.newaxis, np.newaxis] @ frfs_sdof
        else:
            psi = self.pinv_frfs[:, np.newaxis,
                                 np.newaxis] @ np.concatenate((frfs_sdof.real, frfs_sdof.imag), axis=2)
        frfs_sdof_reprojected = self.frfs.ordinate.transpose(
            1, 2, 0)[:, np.newaxis, np.newaxis, self.frequency_slice, :] @ psi
        correlation_matrix = np.zeros(frfs_sdof_reprojected.shape[:3])
        for i, frequency in enumerate(correlation_frequencies):
            closest_index = np.argmin(np.abs(self.frequencies[self.frequency_slice] - frequency))
            abscissa_slice = slice(max([0, closest_index - frequency_lines_for_correlation]),
                                   closest_index + frequency_lines_for_correlation)  # Note that this will automatically stop at the end of the array
            correlation_matrix[:, i] = correlation_coefficient(np.abs(frfs_sdof[i, :, abscissa_slice, 0]),
                                                               np.abs(frfs_sdof_reprojected[:, i, :, abscissa_slice, 0]))
        if plot:
            X, Y = np.meshgrid(correlation_dampings, correlation_frequencies)
            for i, (matrix, coordinate) in enumerate(zip(correlation_matrix, self.frfs[0, :].reference_coordinate)):
                if np.all([shape > 1 for shape in matrix.shape]):
                    fig, ax = plt.subplots(2, 1, num='Reference {:} Correlation'.format(
                        str(coordinate)), gridspec_kw={'height_ratios': [3, 1]})
                    mat2plot = np.log10(1 - matrix)
                    surf = ax[0].contourf(X, Y, -mat2plot, -np.linspace(-1, mat2plot.min(), 20))
                    c = fig.colorbar(surf, ax=ax)  # , shrink=0.5, aspect=5)
                    c.ax.set_yticklabels(['{:0.4f}'.format(1 - 10**(-x))
                                         for x in c.ax.get_yticks()])
                    ax[1].plot(correlation_frequencies, np.max(matrix, axis=1))
                    ax[1].set_ylim(0, 1)
                else:
                    fig, ax = plt.subplots(
                        1, 1, num='Reference {:} Correlation'.format(str(coordinate)))
                    ax.plot(correlation_frequencies, np.max(matrix, axis=1))
                    ax.set_ylim(0, 1)

        return correlation_frequencies, correlation_dampings, correlation_matrix

    def find_peaks(self, correlation_matrix, size=3, threshold=0.9):
        return [np.where((maximum_filter(matrix, size=size) == matrix) & (matrix > threshold)) for matrix in correlation_matrix]

    def compute_initial_rootlist(self,
                                 frequency_samples=None, frequency_resolution=None,
                                 low_damping=0.0025, high_damping=0.05, damping_samples=21,
                                 frequency_lines_for_correlation=20, peak_finder_filter_size=3,
                                 correlation_threshold=0.9, num_roots_mif='cmif',
                                 num_roots_frequency_threshold=0.005, plot_correlation=False):
        self.initial_correlation_frequencies, self.initial_correlation_dampings, self.initial_correlation_matrix = self.compute_correlation_matrix(
            None, None,
            frequency_samples, frequency_resolution,
            low_damping, high_damping, damping_samples,
            frequency_lines_for_correlation, plot=plot_correlation
        )
        peaklists = self.find_peaks(self.initial_correlation_matrix,
                                    peak_finder_filter_size, correlation_threshold)
        # Collapse identical roots
        root_index_list = {}
        for reference_index, (matrix, peaklist) in enumerate(zip(self.initial_correlation_matrix, peaklists)):
            for key in zip(*peaklist):
                value = matrix[key]
                if key not in root_index_list or root_index_list[key][0] < value:
                    root_index_list[key] = (value, reference_index)
        root_list = np.ndarray(len(root_index_list), dtype=[('frequency', 'float64'),
                                                            ('damping', 'float64'),
                                                            ('correlation', 'float64'),
                                                            ('reference_index', 'int'),
                                                            ('num_roots', 'int')])
        for index, (frequency_index, damping_index) in enumerate(sorted(root_index_list)):
            frequency = self.initial_correlation_frequencies[frequency_index]
            damping = self.initial_correlation_dampings[damping_index]
            root_list[index] = (frequency, damping,) + \
                root_index_list[frequency_index, damping_index] + (0,)
        root_list['num_roots'] = self.get_num_roots(
            root_list['frequency'], num_roots_mif, num_roots_frequency_threshold)
        root_list = root_list[root_list['num_roots'] > 0]
        self.initial_rootlist = root_list

    def get_num_roots(self, frequencies, mif_type, frequency_threshold=0.005, plot=False):
        # Find the closest index for each frequency in the data
        abscissa = self.frequencies
        differences = np.abs(abscissa - frequencies[:, np.newaxis])
        frequency_indices = np.argmin(differences, axis=-1)
        nroots = np.zeros(len(frequencies), dtype=int)
        # MMIF
        if mif_type.lower() == 'mmif':
            mmif = self.frfs.compute_mmif()
            mmif_inv = 1 / mmif
            peaks = [find_peaks(mmif_inv.ordinate[i])[0]
                     for i in range(self.reference_coordinates.size)]
            # Now see if we are within the tolerance of a peak
            for i in range(self.reference_coordinates.size):
                for j, (freq, index) in enumerate(zip(frequencies, frequency_indices)):
                    argument = np.abs(abscissa[peaks[i]] - freq)
                    peak_index = np.argmin(argument)
                    diff = argument[peak_index]
                    if diff / freq < frequency_threshold or diff / self.frequency_spacing < 1.25:  # Gives it the ability to be one frequency line off
                        # Check if it is repeated
                        if i > 0:  # check if we are on our second mif line or greater
                            # To be repeated, the higher MIF must drop to at least 60% of the principal mif value
                            # The MMIF value should also be below 0.9
                            # The primary mif must also have a peak -- I don't think this is true
                            if ((1 - mmif[i].ordinate[peaks[i][peak_index]]) / (1 - mmif[i - 1].ordinate[peaks[i][peak_index]]) >= 0.6
                                    and mmif[i].ordinate[peaks[i][peak_index]] < 0.9
                                    # and nroots[j] > 0
                                    ):  # noqa: E124
                                nroots[j] += 1
                        else:
                            nroots[j] += 1
            if plot:
                ax = mmif.plot()
                for nroot, freq_ind in zip(nroots, frequency_indices):
                    ax.plot(abscissa[freq_ind], mmif.ordinate[0, freq_ind], 'r*')
                    ax.text(abscissa[freq_ind], mmif.ordinate[0, freq_ind], '{:}'.format(nroot))
        # CMIF
        elif mif_type.lower() == 'cmif':
            cmif = self.frfs.compute_cmif()
            peaks = [find_peaks(cmif.ordinate[i])[0]
                     for i in range(self.reference_coordinates.size)]
            # Now see if we are within the tolerance of a peak
            for i in range(self.reference_coordinates.size):
                for j, (freq, index) in enumerate(zip(frequencies, frequency_indices)):
                    argument = np.abs(abscissa[peaks[i]] - freq)
                    peak_index = np.argmin(argument)
                    diff = argument[peak_index]
                    if diff / freq < frequency_threshold or diff / self.frequency_spacing < 1.25:  # Gives it the ability to be one frequency line off
                        # Check if it is repeated
                        if i > 0:  # check if we are on our second mif line or greater
                            # To be repeated, the lower MIF must be at least 10% of the primary
                            # The primary mif must also have a peak - I don't think this is true
                            if ((cmif[i].ordinate[peaks[i][peak_index]]) / (cmif[i - 1].ordinate[peaks[i][peak_index]]) >= 0.1
                                    # and nroots[j] > 0
                                    ):  # noqa: E124
                                nroots[j] += 1
                        else:
                            nroots[j] += 1
            if plot:
                ax = cmif.plot()
                for nroot, freq_ind in zip(nroots, frequency_indices):
                    ax.plot(abscissa[freq_ind], cmif.ordinate[0, freq_ind], 'r*')
                    ax.text(abscissa[freq_ind], cmif.ordinate[0, freq_ind], '{:}'.format(nroot))
        return nroots

    def autofit_roots(self, frequency_range=0.01, frequency_points=21, frequency_convergence=0.00025,
                      damping_low=0.0025, damping_high=0.05, damping_points=21, damping_convergence=0.02,
                      frequency_lines_for_correlation=20, max_iter=200,
                      zoom_rate=0.75, plot_convergence=False,
                      autofit_type=AutoFitTypes.ALTERNATE):
        rootlist_dtype = [('frequency', 'float64'),
                          ('damping', 'float64'),
                          ('correlation', 'float64'),
                          ('initial_frequency', 'float64'),
                          ('initial_damping', 'float64'),
                          ('initial_correlation', 'float64'),
                          ('shape_reference', 'int'),
                          ('drive_point_coefficient', 'int')]
        root_list = np.ndarray((0,), rootlist_dtype)
        for i, (initial_frequency,
                initial_damping,
                initial_correlation,
                initial_reference_index,
                initial_num_roots) in enumerate(self.initial_rootlist):
            print('Fitting peak {:}: {:0.2f} Hz {:0.2f} % Damping'.format(
                i, initial_frequency, initial_damping * 100))
            if initial_num_roots > 1:
                print('  Multiple roots detected.  Splitting up peak into multiple seeds.')
                initial_frequencies = initial_frequency * \
                    (1 + frequency_range / 2 * np.array((-1, 0, 1)))
                initial_dampings = np.ones(3) * np.mean([damping_low, damping_high])
                for initial_frequency, initial_damping in zip(initial_frequencies, initial_dampings):
                    try:
                        if autofit_type == AutoFitTypes.ALTERNATE:
                            converged_frequency, converged_damping, converged_correlation = self.autofit_root_alternate(
                                initial_frequency, initial_damping, frequency_range,
                                frequency_points, frequency_convergence, damping_low,
                                damping_high, damping_points, damping_convergence,
                                frequency_lines_for_correlation, max_iter, zoom_rate,
                                plot_convergence)
                        elif autofit_type == AutoFitTypes.PARABOLOID:
                            converged_frequency, converged_damping, converged_correlation = self.autofit_root_paraboloid(
                                initial_frequency, initial_damping, frequency_range,
                                frequency_points, frequency_convergence, damping_low,
                                damping_high, damping_points, damping_convergence,
                                frequency_lines_for_correlation, max_iter, zoom_rate,
                                plot_convergence)
                    except ConvergenceException:
                        print('  Root diverged. Skipping.')
                        continue
                    # Check if the root is already in there
                    same_roots = np.where(((np.abs(root_list['frequency'] - converged_frequency) / converged_frequency < frequency_convergence)
                                           & (np.abs(root_list['damping'] - converged_damping) / converged_damping < damping_convergence)))[0]
                    if same_roots.size > 0:
                        print('  Root already exists.')
                        # Just grab the first one?
                        same_roots = same_roots[0]
                        previous_correlation = root_list['correlation'][same_roots]
                        if converged_correlation > previous_correlation:
                            # Replace the previous one
                            print('    Replacing previous root')
                            root_list[same_roots] = (converged_frequency,
                                                     converged_damping,
                                                     converged_correlation,
                                                     initial_frequency,
                                                     initial_damping,
                                                     initial_correlation,
                                                     0, 0)
                        # Otherwise we don't add it
                        else:
                            print('    Not adding this root.')
                    else:
                        print('  Appending to root list')
                        root_list = np.append(root_list, np.array((converged_frequency,
                                                                  converged_damping,
                                                                  converged_correlation,
                                                                  initial_frequency,
                                                                  initial_damping,
                                                                  initial_correlation,
                                                                  0, 0), rootlist_dtype))
            else:
                try:
                    if autofit_type == AutoFitTypes.ALTERNATE:
                        converged_frequency, converged_damping, converged_correlation = self.autofit_root_alternate(
                            initial_frequency, initial_damping, frequency_range,
                            frequency_points, frequency_convergence, damping_low,
                            damping_high, damping_points, damping_convergence,
                            frequency_lines_for_correlation, max_iter, zoom_rate,
                            plot_convergence)
                    elif autofit_type == AutoFitTypes.PARABOLOID:
                        converged_frequency, converged_damping, converged_correlation = self.autofit_root_paraboloid(
                            initial_frequency, initial_damping, frequency_range,
                            frequency_points, frequency_convergence, damping_low,
                            damping_high, damping_points, damping_convergence,
                            frequency_lines_for_correlation, max_iter, zoom_rate,
                            plot_convergence)
                except ConvergenceException:
                    print('  Root diverged. Skipping.')
                    continue
                # Check if the root is already in there
                same_roots = np.where(((np.abs(root_list['frequency'] - converged_frequency) / converged_frequency < frequency_convergence)
                                       & (np.abs(root_list['damping'] - converged_damping) / converged_damping < damping_convergence)))[0]
                if same_roots.size > 0:
                    print('  Root already exists.')
                    # Just grab the first one?
                    same_roots = same_roots[0]
                    previous_correlation = root_list['correlation'][same_roots]
                    if converged_correlation > previous_correlation:
                        # Replace the previous one
                        print('    Replacing previous root')
                        root_list[same_roots] = (converged_frequency,
                                                 converged_damping,
                                                 converged_correlation,
                                                 initial_frequency,
                                                 initial_damping,
                                                 initial_correlation,
                                                 0, 0)
                    # Otherwise we don't add it
                    else:
                        print('    Not adding this root.')
                else:
                    print('  Appending to root list')
                    root_list = np.append(root_list, np.array((converged_frequency,
                                                              converged_damping,
                                                              converged_correlation,
                                                              initial_frequency,
                                                              initial_damping,
                                                              initial_correlation,
                                                              0, 0), rootlist_dtype))
        self.rootlist = np.sort(root_list)

    def autofit_root_paraboloid(self, initial_frequency, initial_damping,
                                frequency_range=0.01, frequency_points=21,
                                frequency_convergence=0.00025,
                                damping_low=0.0025, damping_high=0.05,
                                damping_points=21, damping_convergence=0.02,
                                frequency_lines_for_correlation=20, max_iter=200,
                                zoom_rate=0.75, plot_convergence=False):
        niter = 0
        frequency_bounds = initial_frequency * (1 + frequency_range / 2 * np.array([-1, 1]))
        damping_bounds = np.array([damping_low, damping_high])
        last_frequency = initial_frequency
        last_damping = initial_damping
        frequency_range = frequency_bounds[1] - frequency_bounds[0]
        damping_range = damping_bounds[1] - damping_bounds[0]
        if plot_convergence:
            fig = plt.figure('Root {:.2f} Hz, {:.2f} % Convergence'.format(initial_frequency, 100 * initial_damping),
                             figsize=plt.figaspect(1 / 3))
            fig.clf()
            ax_3d = fig.add_subplot(1, 3, 1, projection='3d')
            ax_3d.set_title('Correlation Coefficient')
            ax_3d.set_xlabel('Damping')
            ax_3d.set_ylabel('Frequency')
            ax_freq = fig.add_subplot(1, 3, 2)
            ax_freq.set_title('Frequency Convergence')
            ax_freq.set_ylabel('Frequency')
            ax_damp = fig.add_subplot(1, 3, 3)
            ax_damp.set_title('Damping Convergence')
            ax_damp.set_ylabel('Damping')
            ax_freq.plot(niter, last_frequency, 'b.')
            ax_damp.plot(niter, last_damping, 'b.')
        convergence_count = 0
        while niter < max_iter:
            # Compute the correlation matrix
            freq, damp, matrix = self.compute_correlation_matrix(
                low_frequency=frequency_bounds[0],
                high_frequency=frequency_bounds[1],
                frequency_samples=frequency_points,
                low_damping=damping_bounds[0],
                high_damping=damping_bounds[1],
                damping_samples=damping_points,
                frequency_lines_for_correlation=frequency_lines_for_correlation
            )
            x, y = np.meshgrid(damp, freq)
            z = matrix
            peak_matrix, peak_frequency, peak_damping = np.unravel_index(np.argmax(z), z.shape)
            peak_corr = z[peak_matrix, peak_frequency, peak_damping]
            peak_damping = damp[peak_damping]
            peak_frequency = freq[peak_frequency]
            (next_damping, next_frequency, corr_coef), point_fits = self.fit_paraboloid(x, y, matrix)
            best_correlation_index = np.argmax(corr_coef)
            next_damping = next_damping[best_correlation_index]
            next_frequency = next_frequency[best_correlation_index]
            next_correlation = corr_coef[best_correlation_index]
            if plot_convergence:
                ax_3d.cla()
                ax_3d.plot_surface(x, y, z[best_correlation_index],
                                   color='b', linewidth=0, alpha=0.5)
                ax_3d.plot_surface(point_fits[0].reshape(x.shape),
                                   point_fits[1].reshape(y.shape),
                                   point_fits[2].reshape(z.shape)[best_correlation_index],
                                   color='r', linewidth=0, alpha=0.5)
                ax_3d.plot(next_damping, next_frequency, corr_coef[best_correlation_index], 'k.')
            # Check if it is within the bounds
            if (next_damping > damping_bounds[1] or next_damping < damping_bounds[0]):
                next_damping = peak_damping
            else:
                damping_range *= zoom_rate
            if (next_frequency > frequency_bounds[1] or next_frequency < frequency_bounds[0]):
                next_frequency = peak_frequency
            else:
                frequency_range *= zoom_rate
            frequency_bounds = np.array((next_frequency - frequency_range / 2,
                                         next_frequency + frequency_range / 2))
            damping_bounds = np.array((next_damping - damping_range / 2,
                                       next_damping + damping_range / 2))
            niter += 1
            if plot_convergence:
                ax_freq.plot(niter, next_frequency, 'b.')
                ax_damp.plot(niter, next_damping, 'b.')
                plt.pause(0.001)
            if (np.abs(next_damping - last_damping) / last_damping < damping_convergence and
                    np.abs(next_frequency - last_frequency) / last_frequency < frequency_convergence):
                convergence_count += 1
                if convergence_count > 4:
                    break
            else:
                convergence_count = 0
            if (  # np.abs(next_damping - initial_damping)/initial_damping > 10 or
                np.abs(next_frequency - initial_frequency) / initial_frequency > 0.1 or
                next_damping < 0 or
                    next_frequency < 0):
                print('Diverged: Damping Change {:}\nFrequency Change {:}'.format(
                    np.abs(next_damping - initial_damping) / initial_damping,
                    np.abs(next_frequency - initial_frequency) / initial_frequency))
                raise ConvergenceException('Peak Diverged!')
            last_damping = next_damping
            last_frequency = next_frequency
            # break
        return next_frequency, next_damping, next_correlation

    def autofit_root_alternate(self, initial_frequency, initial_damping,
                               frequency_range=0.01, frequency_points=21,
                               frequency_convergence=0.00025,
                               damping_low=0.0025, damping_high=0.05,
                               damping_points=21, damping_convergence=0.02,
                               frequency_lines_for_correlation=20, max_iter=200,
                               zoom_rate=0.75, plot_convergence=False):
        niter = 0
        frequency_bounds = initial_frequency * (1 + frequency_range / 2 * np.array([-1, 1]))
        damping_bounds = np.array([damping_low, damping_high])
        last_frequency = initial_frequency
        last_damping = initial_damping
        frequency_range = frequency_bounds[1] - frequency_bounds[0]
        damping_range = damping_bounds[1] - damping_bounds[0]
        if plot_convergence:
            fig = plt.figure('Root {:.2f} Hz, {:.2f} % Convergence'.format(initial_frequency, 100 * initial_damping),
                             figsize=plt.figaspect(1 / 3))
            fig.clf()
            ax_freq_fit = fig.add_subplot(2, 2, 1)
            ax_freq_fit.set_title('Frequency Correlation')
            ax_freq_fit.set_xlabel('Frequency')
            ax_freq_fit.set_ylabel('Correlation')
            ax_damp_fit = fig.add_subplot(2, 2, 2)
            ax_damp_fit.set_title('Damping Correlation')
            ax_damp_fit.set_xlabel('Damping')
            ax_damp_fit.set_ylabel('Correlation')
            ax_freq = fig.add_subplot(2, 2, 3)
            ax_freq.set_title('Frequency Convergence')
            ax_freq.set_ylabel('Frequency')
            ax_damp = fig.add_subplot(2, 2, 4)
            ax_damp.set_title('Damping Convergence')
            ax_damp.set_ylabel('Damping')
            ax_freq.plot(niter, last_frequency, 'b.')
            ax_damp.plot(niter, last_damping, 'b.')
        convergence_count = 0
        while niter < max_iter:
            next_frequency, freqs_checked, freq_matrix = self.fit_frequency(*frequency_bounds, last_damping,
                                                                            frequency_points, frequency_lines_for_correlation)
            next_damping, damps_checked, damp_matrix = self.fit_damping(*damping_bounds, last_frequency,
                                                                        damping_points, frequency_lines_for_correlation)
            next_correlation = np.max([np.max(freq_matrix), np.max(damp_matrix)])
            if plot_convergence:
                ax_freq_fit.cla()
                ax_freq_fit.plot(freqs_checked, freq_matrix.T)
                ax_damp_fit.cla()
                ax_damp_fit.plot(damps_checked, damp_matrix.T)
            # Check if it is within the bounds
            if (next_damping <= damping_bounds[1] and next_damping >= damping_bounds[0]):
                damping_range *= zoom_rate
            if (next_frequency <= frequency_bounds[1] or next_frequency >= frequency_bounds[0]):
                frequency_range *= zoom_rate
            frequency_bounds = np.array((next_frequency - frequency_range / 2,
                                         next_frequency + frequency_range / 2))
            damping_bounds = np.array((next_damping - damping_range / 2,
                                       next_damping + damping_range / 2))
            niter += 1
            if plot_convergence:
                ax_freq.plot(niter, next_frequency, 'b.')
                ax_damp.plot(niter, next_damping, 'b.')
                plt.pause(0.001)
            if (np.abs(next_damping - last_damping) / last_damping < damping_convergence and
                    np.abs(next_frequency - last_frequency) / last_frequency < frequency_convergence):
                convergence_count += 1
                if convergence_count > 4:
                    break
            else:
                convergence_count = 0
            if (  # np.abs(next_damping - initial_damping)/initial_damping > 10 or
                np.abs(next_frequency - initial_frequency) / initial_frequency > 0.1 or
                next_damping < 0 or
                    next_frequency < 0):
                print('Diverged: Damping Change {:}\nFrequency Change {:}'.format(
                    np.abs(next_damping - initial_damping) / initial_damping,
                    np.abs(next_frequency - initial_frequency) / initial_frequency))
                raise ConvergenceException('Peak Diverged!')
            last_damping = next_damping
            last_frequency = next_frequency
        return next_frequency, next_damping, next_correlation

    def fit_frequency(self, min_freq, max_freq, damping, frequency_points=21,
                      frequency_lines_for_correlation=20):
        freq, damp, matrix = self.compute_correlation_matrix(
            low_frequency=min_freq, high_frequency=max_freq,
            frequency_samples=frequency_points,
            low_damping=damping, high_damping=damping, damping_samples=1,
            frequency_lines_for_correlation=frequency_lines_for_correlation, plot=False)
        # Find the maximum frequency
        max_freq_ind = np.argmax(np.max(matrix[..., 0], 0))
        max_freq = freq[max_freq_ind]
        return max_freq, freq, matrix[..., 0]

    def fit_damping(self, min_damp, max_damp, frequency, damping_points=21,
                    frequency_lines_for_correlation=20):
        freq, damp, matrix = self.compute_correlation_matrix(
            low_frequency=frequency, high_frequency=frequency,
            frequency_samples=1,
            low_damping=min_damp, high_damping=max_damp, damping_samples=damping_points,
            frequency_lines_for_correlation=frequency_lines_for_correlation, plot=False)
        # Find the maximum frequency
        max_damping_ind = np.argmax(np.max(matrix[..., 0, :], 0))
        max_damp = damp[max_damping_ind]
        return max_damp, damp, matrix[..., 0, :]

    def compute_residues(self, roots,
                         residuals=True, weighting='magnitude'):
        self.natural_frequencies = roots['frequency']
        self.damping_ratios = roots['damping']
        self.residues, self.frfs_synth_residue, self.frfs_synth_residual = (
            modeshape_compute_residues(
                self.frfs, self.natural_frequencies, self.damping_ratios,
                not self.complex_modes, residuals, self.min_frequency, self.max_frequency,
                weighting, self.displacement_derivative))

    def compute_shapes(self):
        self.shapes, negative_drive_points = modeshape_compute_shapes(
            self.natural_frequencies, self.damping_ratios,
            self.frfs.coordinate, self.residues, ShapeSelection.DRIVE_POINT_COEFFICIENT)
        self.negative_drive_points = negative_drive_points

    def frf_sdof_real(self, frequencies, root_frequencies, root_dampings):
        omega = 2 * np.pi * frequencies
        omega_n = 2 * np.pi * root_frequencies
        zeta = root_dampings
        return -omega[:, np.newaxis]**2 / (-omega[:, np.newaxis]**2 + omega_n**2 + 2 * 1j * zeta * omega_n * omega[:, np.newaxis])

    def frf_sdof_complex(self, frequencies, root_frequencies, root_dampings):
        raise NotImplementedError('Complex Shapes are not implemented yet!')

    def fit_paraboloid(self, x, y, z):
        x = np.array(x)
        original_x_shape = x.shape
        x = x.flatten()
        y = np.array(y).flatten()
        z = np.array(z)
        new_z_shape = z.shape[:-len(original_x_shape)] + (np.prod(x.shape), 1)
        z = z.reshape(*new_z_shape)
        # Now do some normalization
        x_range = x.max() - x.min()
        x_mean = (x.max() + x.min()) / 2
        y_range = y.max() - y.min()
        y_mean = (y.max() + y.min()) / 2
        # z_range = z.max() - z.min()
        # z_mean = (z.max() + z.min())/2
        # x_range = 1
        # x_mean = 0
        # y_range = 1
        # y_mean = 0
        z_range = 1
        z_mean = 0
        x = (x - x_mean) / x_range
        y = (y - y_mean) / y_range
        z = (z - z_mean) / z_range
        A = np.array((
            x**2,
            y**2,
            x * y,
            x,
            y,
            np.ones(x.shape)
        )).T
        a, b, c, d, e, f = (np.linalg.pinv(A) @ z)[..., 0].T
        x_c = (-2 * b * d + c * e) / (4 * a * b - c**2)
        y_c = (-2 * a * e + c * d) / (4 * a * b - c**2)
        z_c = (a * x_c**2 + b * y_c**2 + c * x_c * y_c + d * x_c + e * y_c + f)
        x_c = x_c * x_range + x_mean
        y_c = y_c * y_range + y_mean
        z_c = z_c * z_range + z_mean
        x_fit = x * x_range + x_mean
        y_fit = y * y_range + y_mean
        z_fit = (a[:, np.newaxis] * x**2 + b[:, np.newaxis] * y**2 + c[:, np.newaxis] * x *
                 y + d[:, np.newaxis] * x + e[:, np.newaxis] * y + f[:, np.newaxis])  # *z_range+z_mean
        return (x_c, y_c, z_c), (x_fit, y_fit, z_fit)


class AddRootDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super(QtWidgets.QDialog, self).__init__(parent)
        uic.loadUi(os.path.join(os.path.abspath(os.path.dirname(
            os.path.abspath(__file__))), 'smac_add_root.ui'), self)
        self.setWindowTitle('PySMAC: Add Root')

        self.low_damping.setValue(parent.minimum_damping_autofit_selector.value())
        self.high_damping.setValue(parent.maximum_damping_autofit_selector.value())
        self.low_frequency.setValue(parent.low_frequency)
        self.high_frequency.setValue(parent.high_frequency)
        for widget in [self.low_frequency, self.high_frequency]:
            widget.setMinimum(parent.low_frequency)
            widget.setMaximum(parent.high_frequency)
        self.frequency_samples.setValue(parent.frequency_autofit_values_in_range.value())
        self.damping_samples.setValue(parent.damping_autofit_number_samples_selector.value())

        self.correlation_plot = self.correlation_view.getPlotItem()
        self.correlation_plot.setLimits(xMin=parent.low_frequency,
                                        xMax=parent.high_frequency,
                                        yMin=0.001,
                                        yMax=99.99)
        self.correlation_plot.setRange(xRange=(parent.low_frequency, parent.high_frequency),
                                       yRange=(parent.minimum_damping_autofit_selector.value(),
                                               parent.maximum_damping_autofit_selector.value()), padding=0.0)
        self.correlation_plot.getAxis('left').setLabel('Damping (%)')
        self.correlation_plot.getAxis('bottom').setLabel('Frequency (Hz)')
        self.SMAC_GUI = parent

        # Compute initial correlation
        self.image = None
        self.image_log_scale = False
        self.recompute_correlation()

        self.max_correlation = None
        self.max_freq = None
        self.max_damp = None

        self.connect_callbacks()
        print('Completed init')

    def connect_callbacks(self):
        self.recompute_correlation_button.clicked.connect(self.recompute_correlation)
        self.linear_plot.toggled.connect(self.switch_log_plot)
        self.correlation_plot.sigRangeChanged.connect(self.update_range_selectors)
        self.low_frequency.valueChanged.connect(self.update_plot_range)
        self.high_frequency.valueChanged.connect(self.update_plot_range)
        self.low_damping.valueChanged.connect(self.update_plot_range)
        self.high_damping.valueChanged.connect(self.update_plot_range)

    def update_range_selectors(self):
        self.low_frequency.blockSignals(True)
        self.high_frequency.blockSignals(True)
        self.low_damping.blockSignals(True)
        self.high_damping.blockSignals(True)
        ((xmin, xmax), (ymin, ymax)) = self.correlation_plot.vb.viewRange()
        self.low_frequency.setValue(xmin)
        self.high_frequency.setValue(xmax)
        self.low_damping.setValue(ymin)
        self.high_damping.setValue(ymax)
        self.low_frequency.blockSignals(False)
        self.high_frequency.blockSignals(False)
        self.low_damping.blockSignals(False)
        self.high_damping.blockSignals(False)

    def update_plot_range(self):
        self.correlation_plot.blockSignals(True)
        lf = self.low_frequency.value()
        hf = self.high_frequency.value()
        ld = self.low_damping.value()
        hd = self.high_damping.value()
        self.correlation_plot.setRange(xRange=(lf, hf),
                                       yRange=(ld, hd), padding=0.0)
        self.correlation_plot.blockSignals(False)

    def switch_log_plot(self):
        if self.image_log_scale and self.linear_plot.isChecked():
            # Undo log scale back to linear
            self.image.setImage((-(10**(-self.image.image))) + 1)
            self.image_log_scale = False
        if not self.image_log_scale and self.log_plot.isChecked():
            # put into log scale
            self.image.setImage(-np.log10(1 - self.image.image))
            self.image_log_scale = True

    def recompute_correlation(self):
        self.correlation_plot.clear()
        lf = self.low_frequency.value()
        hf = self.high_frequency.value()
        ld = self.low_damping.value()
        hd = self.high_damping.value()
        fs = self.frequency_samples.value()
        ds = self.damping_samples.value()
        df = (hf - lf) / (fs - 1)
        dd = (hd - ld) / (ds - 1)
        freq, damp, matrix = self.SMAC_GUI.smac.compute_correlation_matrix(
            low_frequency=lf, high_frequency=hf,
            frequency_samples=fs,
            low_damping=ld / 100,
            high_damping=hd / 100,
            damping_samples=ds,
            frequency_lines_for_correlation=self.SMAC_GUI.frequency_autofit_lines_to_use_in_correlation.value())
        matrix = np.max(matrix, axis=0)
        freq_ind, damp_ind = np.unravel_index(matrix.argmax(), matrix.shape)
        self.max_correlation = matrix[freq_ind, damp_ind]
        self.max_freq = freq[freq_ind]
        self.max_damp = damp[damp_ind]
        self.correlation_description.setText('Correlation:\n{:0.5f}\nFrequency:\n{:0.3f}\nDamping:\n{:0.3f}%'.format(
            self.max_correlation, self.max_freq, self.max_damp * 100))
        if self.linear_plot.isChecked():
            self.image_log_scale = False
        else:
            self.image_log_scale = True
            matrix = -np.log10(1 - matrix)
        self.image = pyqtgraph.ImageItem(
            matrix, rect=[lf - (df / 2), ld - (dd / 2), hf - lf + df, hd - ld + dd])
        self.correlation_plot.addItem(self.image)

    @staticmethod
    def add_root(parent):
        dialog = AddRootDialog(parent)
        parent.add_root_dialog = dialog
        result = (dialog.exec_() == QtWidgets.QDialog.Accepted)
        print('Continued: {:}'.format(result))
        if result:
            return dialog.max_freq, dialog.max_damp, dialog.max_correlation
        else:
            return None, None, None
        parent.add_root_dialog = None


class SMAC_GUI(QMainWindow):
    def __init__(self, frf_data: TransferFunctionArray):
        super(SMAC_GUI, self).__init__()
        uic.loadUi(os.path.join(os.path.abspath(
            os.path.dirname(os.path.abspath(__file__))), 'smac.ui'), self)

        # Store FRFs
        # TODO Make it so we can load date if none is passed
        self.frfs = frf_data.reshape_to_matrix()

        # Set up colormaps
        self.cm = cm.Dark2
        self.cm_mod = 8
        self.compare_cm = cm.Paired
        self.compare_cm_mod = 12

        # Initialize Plots
        self.mif_plot = self.mif_plot_widget.getPlotItem()
        self.correlation_plot = self.corrrelation_coefficient_plot_widget.getPlotItem()

        self.mif_plot.getAxis('left').setLabel('Mode Indicator Function')
        self.correlation_plot.getAxis('left').setLabel('Correlation Coefficient')
        self.mif_plot.setXLink(self.correlation_plot)
        self.correlation_coefficient_selector = pyqtgraph.LinearRegionItem(values=(0, 0.9),
                                                                           orientation='horizontal',
                                                                           bounds=(0, 1))

        self.connect_callbacks()

        # Set defaults
        self.high_frequency = self.frfs.abscissa.max()
        self.low_frequency = self.frfs.abscissa.min()
        self.low_frequency_selector.setValue(self.low_frequency)
        self.high_frequency_selector.setValue(self.high_frequency)
        self.frequency_spacing_selector.setValue(np.mean(np.diff(self.frfs[0, 0].abscissa)))
        self.num_responses_label.setText('{:} Responses'.format(self.frfs.shape[0]))
        self.num_references_label.setText('{:} References'.format(self.frfs.shape[1]))
        self.num_frequencies_label.setText('{:} Frequencies'.format(self.frfs.num_elements))
        self.smac = None
        self.initial_roots = []
        self.initial_roots_for_autofit = []
        self.initial_correlation = None
        self.root_markers = []
        self.crosshair_v_mif = pyqtgraph.InfiniteLine(angle=90, movable=False)
        self.crosshair_v_corr = pyqtgraph.InfiniteLine(angle=90, movable=False)
        self.mif_plot.addItem(self.crosshair_v_mif)
        self.correlation_plot.addItem(self.correlation_coefficient_selector)
        self.correlation_plot.addItem(self.crosshair_v_corr)
        self.rootlist = []
        self.guiplots = []
        self.shapes = None
        self.geometry = None
        self.resynthesized_frfs = None
        self.add_root_dialog = None
        self.setWindowTitle('PySMAC')

        self.show()

    def connect_callbacks(self):
        self.frequency_spacing_selector.valueChanged.connect(self.update_frequency_line_label)
        self.compute_pseudoinverse_button.clicked.connect(self.compute_pseudoinverse)
        self.compute_correlation_matrix_button.clicked.connect(self.compute_correlation_matrix)
        self.correlation_plot.scene().sigMouseMoved.connect(self.update_crosshairs)
        self.correlation_plot.scene().sigMouseClicked.connect(self.add_initial_root)
        self.roots_delete_button.clicked.connect(self.delete_initial_roots)
        self.rootlist_tablewidget.itemSelectionChanged.connect(self.paint_markers)
        self.minimum_coefficient_entry.valueChanged.connect(self.update_selector_and_refind)
        self.correlation_coefficient_selector.sigRegionChanged.connect(
            self.update_coefficient_and_refind)
        self.confirm_initial_rootlist_button.clicked.connect(self.confirm_initial_roots_for_autofit)
        self.autofit_roots_button.clicked.connect(self.autofit_roots)
        self.delete_button.clicked.connect(self.delete_roots)
        self.add_button.clicked.connect(self.add_root)
        self.resynthesize_button.clicked.connect(self.compute_shapes)
        self.load_geometry_button.clicked.connect(self.load_geometry)
        self.plot_shapes_button.clicked.connect(self.plot_shapes)
        self.plot_mac_button.clicked.connect(self.plot_mac)
        self.save_shapes_button.clicked.connect(self.save_shapes)
        self.export_mode_table_button.clicked.connect(self.export_mode_table)
        self.merge_button.clicked.connect(self.load_shapes)

    def plot_mac(self):
        mac_matrix = mac(self.shapes.shape_matrix.T)
        matrix_plot(mac_matrix)

    def save_shapes(self):
        if self.shapes is None:
            QtWidgets.QMessageBox.warning(self, 'PySMAC', 'Shapes not created!')
            return
        filename, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Shapes', filter='Numpy Files (*.npy)')
        if filename == '':
            return
        self.shapes.save(filename)

    def load_shapes(self):
        filename, file_filter = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open Shapes', filter='Numpy Files (*.npy);;Universal Files (*.unv *.uff)')
        if filename == '':
            return
        new_shapes = ShapeArray.load(filename).flatten()
        new_root = np.zeros((new_shapes.size,), dtype=self.rootlist.dtype)
        new_root['frequency'] = new_shapes.frequency
        new_root['damping'] = new_shapes.damping
        new_root['correlation'] = 0
        new_root['shape_reference'] = -1
        self.rootlist = np.sort(np.concatenate((self.rootlist, new_root)))
        self.update_rootlist_table()

    def export_mode_table(self):
        QtWidgets.QMessageBox.warning(self, 'PySMAC', 'Not Implemented')

    def load_geometry(self):
        filename, file_filter = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open Geometry', filter='Numpy Files (*.npz);;Universal Files (*.unv *.uff)')
        if filename == '':
            return
        self.geometry = Geometry.load(filename)

    def plot_shapes(self):
        if self.geometry is None:
            self.load_geometry()
            if self.geometry is None:
                QtWidgets.QMessageBox.warning(self, 'PySMAC', 'No Geometry Loaded!')
                return
        if self.shapes is None:
            QtWidgets.QMessageBox.warning(self, 'PySMAC', 'Shapes not created!')
            return
        self.geometry.plot_shape(self.shapes)

    def compute_shapes(self):
        try:
            row_indices = [i for i in range(self.root_table.rowCount())
                           if self.root_table.cellWidget(i, 0).isChecked()]
            logical_index = np.zeros(self.rootlist.shape, dtype=bool)
            logical_index[row_indices] = True
            rootlist = self.rootlist[logical_index]
            if rootlist.size == 0:
                QtWidgets.QMessageBox.warning(
                    self, 'PySMAC', 'No Roots Selected!\n\nPlease select roots to use in resynthesis!')
                return
            self.smac.compute_residues(rootlist, residuals=self.use_residuals_checkbox.isChecked())
            self.smac.compute_shapes()
            self.resynthesized_frfs = (self.smac.shapes.compute_frf(self.frfs[0, 0].abscissa[self.smac.frequency_slice],
                                                                    np.unique(
                                                                        self.frfs.response_coordinate),
                                                                    np.unique(
                                                                        self.frfs.reference_coordinate),
                                                                    self.datatype_selector.currentIndex())
                                       + self.smac.frfs_synth_residual)
            if self.collapse_to_real_checkbox.isChecked():
                self.shapes = self.smac.shapes.to_real()
            else:
                self.shapes = self.smac.shapes
            for row_index in np.arange(self.root_table.rowCount()):
                item = QtWidgets.QTableWidgetItem('')
                self.root_table.setItem(row_index, 6, item)
            for row_index, shape in zip(np.sort(row_indices), self.shapes):
                item = QtWidgets.QTableWidgetItem(str(shape.comment1))
                self.root_table.setItem(row_index, 6, item)
            # Now plot the shapes requested
            self.guiplots = []
            if self.nmif_checkbox.isChecked():
                self.guiplots.append(
                    GUIPlot(self.frfs.compute_nmif(),
                            self.resynthesized_frfs.compute_nmif())
                )
            if self.mmif_checkbox.isChecked():
                self.guiplots.append(
                    GUIPlot(self.frfs.compute_mmif(),
                            self.resynthesized_frfs.compute_mmif())
                )
            if self.cmif_checkbox.isChecked():
                self.guiplots.append(
                    GUIPlot(self.frfs.compute_cmif(),
                            self.resynthesized_frfs.compute_cmif())
                )
            if self.frf_checkbox.isChecked():
                self.guiplots.append(
                    GUIPlot(self.frfs,
                            self.resynthesized_frfs)
                )
        except Exception as e:
            print('Error: {:}'.format(str(e)))

    def add_root(self):
        freq, damp, corr = AddRootDialog.add_root(self)
        if freq is not None:
            # add root to the table
            new_root = np.zeros((1,), dtype=self.rootlist.dtype)
            new_root['frequency'] = freq
            new_root['damping'] = damp
            new_root['correlation'] = corr
            new_root['shape_reference'] = -1
            self.rootlist = np.sort(np.concatenate((self.rootlist, new_root)))
            self.update_rootlist_table()

    def delete_roots(self):
        select = self.root_table.selectionModel()
        row_indices = [val.row() for val in select.selectedRows()]
        logical_index = np.ones(self.rootlist.shape, dtype=bool)
        logical_index[row_indices] = False
        self.rootlist = self.rootlist[logical_index]
        self.update_rootlist_table()

    def autofit_roots(self):
        self.smac.autofit_roots(
            self.frequency_autofit_range_selector.value() / 100,
            self.frequency_autofit_values_in_range.value(),
            self.frequency_autofit_convergence_percentage_selector.value() / 100,
            self.minimum_damping_autofit_selector.value() / 100,
            self.maximum_damping_autofit_selector.value() / 100,
            self.damping_autofit_number_samples_selector.value(),
            self.damping_convergence_percentage_selector.value() / 100,
            self.frequency_autofit_lines_to_use_in_correlation.value(),
            self.maximum_iterations_selector.value(),
            self.zoom_rate_selector.value(),
            self.display_convergence_graphs_checkbox.isChecked())
        self.rootlist = self.smac.rootlist
        self.rootlist['shape_reference'] = -1
        # Initialize table
        self.update_rootlist_table()
        self.smac_tabs.setCurrentIndex(4)

    def update_rootlist_table(self):
        self.shapes = None
        self.resynthesized_frfs = None
        self.root_table.clearContents()
        self.root_table.setRowCount(self.rootlist.size)
        for i, root in enumerate(self.rootlist):
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(True)
            self.root_table.setCellWidget(i, 0, checkbox)
            item = QtWidgets.QTableWidgetItem('{:0.2f}'.format(root['frequency']))
            self.root_table.setItem(i, 1, item)
            item = QtWidgets.QTableWidgetItem('{:0.3f}'.format(root['damping'] * 100))
            self.root_table.setItem(i, 2, item)
            item = QtWidgets.QTableWidgetItem('{:0.4f}'.format(root['correlation']))
            self.root_table.setItem(i, 3, item)
            item = QtWidgets.QTableWidgetItem('{:0.2f}'.format(root['initial_frequency']))
            self.root_table.setItem(i, 4, item)
            item = QtWidgets.QTableWidgetItem('{:0.3f}'.format(root['initial_damping'] * 100))
            self.root_table.setItem(i, 5, item)
            item = QtWidgets.QTableWidgetItem('{:0.4f}'.format(root['initial_correlation']))
            self.root_table.setItem(i, 6, item)
            if root['shape_reference'] >= 0:
                item = QtWidgets.QTableWidgetItem(
                    str(np.unique(self.frfs.reference_coordinate)[root['shape_reference']]))
                self.root_table.setItem(i, 7, item)
                item = QtWidgets.QTableWidgetItem('{:0.2f}'.format(root['drive_point_coefficient']))
                self.root_table.setItem(i, 8, item)

    def confirm_initial_roots_for_autofit(self):
        self.initial_roots_for_autofit = self.initial_roots.copy()
        self.smac.initial_rootlist = self.initial_roots_for_autofit
        self.smac_tabs.setCurrentIndex(3)

    def update_coefficient_and_refind(self):
        self.minimum_coefficient_entry.blockSignals(True)
        self.minimum_coefficient_entry.setValue(
            self.correlation_coefficient_selector.getRegion()[1])
        self.minimum_coefficient_entry.blockSignals(False)
        self.refind_peaks()

    def update_selector_and_refind(self):
        self.correlation_coefficient_selector.blockSignals(True)
        self.correlation_coefficient_selector.setRegion((0, self.minimum_coefficient_entry.value()))
        self.correlation_coefficient_selector.blockSignals(False)
        self.refind_peaks()

    def refind_peaks(self):
        num_roots_mif = 'cmif'
        num_roots_frequency_threshold = 0.005
        peaklists = self.smac.find_peaks(self.initial_correlation[..., np.newaxis], size=3,
                                         threshold=self.minimum_coefficient_entry.value())
        # Collapse identical roots
        root_index_list = {}
        for reference_index, (matrix, peaklist) in enumerate(zip(self.initial_correlation[..., np.newaxis], peaklists)):
            for key in zip(*peaklist):
                value = matrix[key]
                if key not in root_index_list or root_index_list[key][0] < value:
                    root_index_list[key] = (value, reference_index)
        root_list = np.ndarray(len(root_index_list), dtype=[('frequency', 'float64'),
                                                            ('damping', 'float64'),
                                                            ('correlation', 'float64'),
                                                            ('reference_index', 'int'),
                                                            ('num_roots', 'int')])
        for index, (frequency_index, damping_index) in enumerate(sorted(root_index_list)):
            frequency = self.initial_correlation_frequencies[frequency_index]
            damping = self.initial_correlation_damping
            root_list[index] = (frequency, damping,) + root_index_list[frequency_index, 0] + (0,)
        root_list['num_roots'] = self.smac.get_num_roots(
            root_list['frequency'], num_roots_mif, num_roots_frequency_threshold)
        root_list = root_list[root_list['num_roots'] > 0]
        self.initial_roots = root_list
        self.update_initial_rootlist_tab(no_reset_axes=True)

    def paint_markers(self):
        select = self.rootlist_tablewidget.selectionModel()
        row_indices = [val.row() for val in select.selectedRows()]
        logical_index = np.zeros(self.initial_roots.shape, dtype=bool)
        logical_index[row_indices] = True
        for markers, selected in zip(self.root_markers, logical_index):
            if selected:
                for marker in markers:
                    marker.setBrush((0, 0, 255))
            else:
                for marker in markers:
                    marker.setBrush((0, 0, 0, 0))

    def delete_initial_roots(self):
        select = self.rootlist_tablewidget.selectionModel()
        row_indices = [val.row() for val in select.selectedRows()]
        logical_index = np.ones(self.initial_roots.shape, dtype=bool)
        logical_index[row_indices] = False
        self.initial_roots = self.initial_roots[logical_index]
        self.update_initial_rootlist_tab(no_reset_axes=True)

    def compute_correlation_matrix(self):
        self.smac.compute_initial_rootlist(
            frequency_resolution=self.frequency_spacing_selector.value(),
            low_damping=self.initial_damping_selector.value() / 100,
            damping_samples=1,
            frequency_lines_for_correlation=self.correlation_lines_selector.value())
        self.initial_roots = self.smac.initial_rootlist
        self.initial_correlation = self.smac.initial_correlation_matrix[..., 0]
        self.initial_correlation_frequencies = self.smac.initial_correlation_frequencies
        self.initial_correlation_damping = self.smac.initial_correlation_dampings[0]
        self.update_initial_rootlist_tab()
        self.smac_tabs.setCurrentIndex(2)

    def update_initial_rootlist_tab(self, no_reset_axes=False):
        if no_reset_axes:
            mif_range = self.mif_plot.vb.viewRange()
            corr_range = self.correlation_plot.vb.viewRange()
        self.mif_plot.clear()
        self.mif_plot.addItem(self.crosshair_v_mif)
        self.correlation_plot.clear()
        self.correlation_plot.addItem(self.correlation_coefficient_selector)
        self.correlation_plot.addItem(self.crosshair_v_corr)
        if self.cmif_radio_button.isChecked():
            self.mif_plot.setLogMode(False, True)
            mif = self.frfs.compute_cmif()
            for i, curve in enumerate(mif):
                pen = pyqtgraph.mkPen(color=[int(255 * v) for v in self.cm(i % self.cm_mod)])
                self.mif_plot.plot(curve.abscissa, curve.ordinate, pen=pen)
        elif self.mmif_radio_button.isChecked():
            self.mif_plot.setLogMode(False, False)
            mif = self.frfs.compute_mmif()
            for i, curve in enumerate(mif):
                pen = pyqtgraph.mkPen(color=[int(255 * v) for v in self.cm(i % self.cm_mod)])
                self.mif_plot.plot(curve.abscissa, curve.ordinate, pen=pen)
        elif self.nmif_radio_button.isChecked():
            self.mif_plot.setLogMode(False, False)
            mif = self.frfs.compute_nmif()
            pen = pyqtgraph.mkPen(color=[int(255 * v) for v in self.cm(0 % self.cm_mod)])
            self.mif_plot.plot(mif.abscissa, mif.ordinate)
        # Now plot the correlation coefficient
        if self.initial_correlation is not None:
            for i, curve in enumerate(self.initial_correlation):
                pen = pyqtgraph.mkPen(color=[int(255 * v) for v in self.cm(i % self.cm_mod)])
                self.correlation_plot.plot(self.initial_correlation_frequencies, curve, pen=pen)
            self.correlation_plot.setLimits(yMin=0, yMax=1.1)
            if no_reset_axes:
                self.correlation_plot.setRange(
                    xRange=corr_range[0], yRange=corr_range[1], padding=0)
                self.mif_plot.setRange(xRange=mif_range[0], yRange=mif_range[1], padding=0)
            else:
                self.correlation_plot.setRange(xRange=(self.low_frequency, self.high_frequency),
                                               yRange=(0, 1))
            self.rootlist_tablewidget.setRowCount(len(self.initial_roots))
            self.root_markers = []
            try:
                mif = mif[0]
            except IndexError:
                pass
            root_mif_ordinates = np.interp(
                self.initial_roots['frequency'], mif.abscissa, mif.ordinate)
            root_corr_ordinates = np.interp(
                self.initial_roots['frequency'], self.initial_correlation_frequencies, np.max(self.initial_correlation, axis=0))
            for index, (root, mif_ord, corr_ord) in enumerate(zip(self.initial_roots, root_mif_ordinates, root_corr_ordinates)):
                frequency = root['frequency']
                item = QtWidgets.QTableWidgetItem('{:0.2f}'.format(frequency))
                self.rootlist_tablewidget.setItem(index, 0, item)
                item = QtWidgets.QTableWidgetItem('{:0.4f}'.format(root['correlation']))
                self.rootlist_tablewidget.setItem(index, 1, item)
                item = QtWidgets.QTableWidgetItem('{:}'.format(root['num_roots']))
                self.rootlist_tablewidget.setItem(index, 2, item)
                mif_item = pyqtgraph.ScatterPlotItem([frequency], [np.log10(
                    mif_ord)], symbol='o', symbolPen=(0, 0, 255), brush=(0, 0, 0, 0))
                self.mif_plot.addItem(mif_item)
                corr_item = pyqtgraph.ScatterPlotItem(
                    [frequency], [corr_ord], symbol='o', symbolPen=(0, 0, 255), brush=(0, 0, 0, 0))
                self.correlation_plot.addItem(corr_item)
                self.root_markers.append((mif_item, corr_item))

    def update_crosshairs(self, position):
        if self.correlation_plot.vb.sceneBoundingRect().contains(position):
            mouse_point = self.correlation_plot.vb.mapSceneToView(position)
            self.crosshair_v_corr.setPos(mouse_point.x())
            self.crosshair_v_mif.setPos(mouse_point.x())

    def compute_pseudoinverse(self):
        self.smac = SMAC(self.frfs, self.low_frequency_selector.value(),
                         self.high_frequency_selector.value(), self.complex_modes_selector.isChecked(),
                         self.datatype_selector.currentIndex())
        self.smac.compute_pseudoinverse()
        self.low_frequency = self.low_frequency_selector.value()
        self.high_frequency = self.high_frequency_selector.value()
        self.update_frequency_line_label()
        self.smac_tabs.setCurrentIndex(1)

    def update_frequency_line_label(self):
        num_samples = int(np.round((self.high_frequency - self.low_frequency) /
                          self.frequency_spacing_selector.value())) + 1
        self.frequency_spacing_label.setText(
            ' Hz Frequency Spacing ({:} Frequency Samples)'.format(num_samples))

    def add_initial_root(self, event):
        position = event.scenePos()
        if self.correlation_plot.vb.sceneBoundingRect().contains(position):
            mouse_point = self.correlation_plot.vb.mapSceneToView(position)
            self.crosshair_v_corr.setPos(mouse_point.x())
            self.crosshair_v_mif.setPos(mouse_point.x())
            frequency_index = np.argmin(
                np.abs(self.initial_correlation_frequencies - mouse_point.x()))
            new_root = np.ndarray((1,), dtype=self.initial_roots.dtype)
            new_root['frequency'] = self.initial_correlation_frequencies[frequency_index]
            new_root['reference_index'] = np.argmax(self.initial_correlation[:, frequency_index])
            new_root['correlation'] = self.initial_correlation[new_root['reference_index'], frequency_index]
            new_root['damping'] = self.initial_correlation_damping
            new_root['num_roots'] = self.smac.get_num_roots(
                np.array([new_root['frequency']]), 'cmif')[0]
            self.initial_roots = np.sort(np.concatenate((self.initial_roots, new_root)))
            self.update_initial_rootlist_tab(no_reset_axes=True)
