# -*- coding: utf-8 -*-
"""
Implementation of a multi-reference polynomial curve fitter for Python

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

import os

import numpy as np
from ..core.sdynpy_data import TransferFunctionArray, GUIPlot
from ..signal_processing.sdynpy_complex import collapse_complex_to_real
from ..core.sdynpy_geometry import Geometry
from ..core.sdynpy_shape import ShapeArray
from .sdynpy_modeshape import compute_residues as modeshape_compute_residues, compute_shapes as modeshape_compute_shapes, ShapeSelection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import traceback

from qtpy import QtWidgets, uic
from qtpy.QtWidgets import QMainWindow, QTableWidgetItem, QWidget
import pyqtgraph
pyqtgraph.setConfigOption('background', 'w')
pyqtgraph.setConfigOption('foreground', 'k')


class PolyPy:

    def __init__(self, frfs: TransferFunctionArray, min_frequency=None,
                 max_frequency=None, displacement_derivative=2):
        self.frfs = frfs.reshape_to_matrix()
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.displacement_derivative = displacement_derivative
        abscissa_indices = np.ones(self.frequencies.shape, dtype=bool)
        if not min_frequency is None:
            abscissa_indices &= (self.frequencies >= min_frequency)
        if not max_frequency is None:
            abscissa_indices &= (self.frequencies <= max_frequency)
        abscissa = self.frequencies[abscissa_indices]
        freq_range = np.array((np.min(abscissa), np.max(abscissa)))
        index_range = np.argmin(np.abs(self.frequencies - freq_range[:, np.newaxis]), axis=1)
        self.frequency_slice = slice(index_range[0], index_range[1] + 1)

    def compute_poles(self, polynomial_orders, weighting=None,
                      frequency_stability_threshold=0.01,
                      damping_stability_threshold=0.1,
                      modal_participation_threshold=0.15):
        H = self.frfs.ordinate[..., self.frequency_slice]
        num_output, num_input, num_freq = H.shape
        num_poles = int(polynomial_orders[-1])
        omegas = self.angular_frequencies[self.frequency_slice]

        # Convert frf to acceleration for pole fitting
        H *= (1j * omegas)**(2 - self.displacement_derivative)

        omega0 = omegas[0]
        sample_time = np.pi / (omegas[-1] - omega0)
        omegas = omegas - omega0
        if weighting is None:
            weighting = np.ones((num_freq, num_output))
        else:
            weighting = weighting[self.frequency_slice]

        if isinstance(polynomial_orders, (int, np.integer)):
            polynomial_orders = (np.arange(np.floor(polynomial_orders /
                                 num_input).astype(int)) + 1)[1:] * num_input

        self.polynomial_orders = np.flip(polynomial_orders)
        self.weighting = weighting
        self.frequency_stability_threshold = frequency_stability_threshold
        self.damping_stability_threshold = damping_stability_threshold
        self.modal_participation_threshold = modal_participation_threshold

        pole_dtype = np.dtype([('omega', float), ('zeta', float), ('Lr_real', float, num_input),
                               ('Lr_complex', complex, num_input), ('freq_stable',
                                                                    bool), ('damp_stable', bool),
                               ('part_stable', bool)])

        self.pole_list = []

        Omega = np.exp(-1j * omegas[:, np.newaxis] * sample_time * np.arange(num_poles + 1))

        RTot = np.zeros([num_poles + 1, num_poles + 1, num_output])
        STot = np.zeros([num_poles + 1, num_input * (num_poles + 1), num_output])
        TTot = np.zeros([num_input * (num_poles + 1), num_input * (num_poles + 1), num_output])

        for j_output in range(num_output):
            print('Accumulating Data: Progress = {:0.2f}%'.format(j_output / num_output * 100.0))
            Xj = weighting[:, j_output][..., np.newaxis] * Omega
            Hj = H[j_output, ...]
            Yj = np.array([-np.kron(Xk, Hk) for Xk, Hk in zip(Xj, Hj.transpose())])
            Rj = np.real(Xj.conjugate().transpose() @ Xj)
            Sj = np.real(Xj.conjugate().transpose() @ Yj)
            Tj = np.real(Yj.conjugate().transpose() @ Yj)
            RTot[:, :, j_output] = Rj
            STot[:, :, j_output] = Sj
            TTot[:, :, j_output] = Tj

        for i_order, order in enumerate(self.polynomial_orders):

            print('Solving for {:} roots ({:} of {:})'.format(
                order, i_order + 1, len(polynomial_orders)))

            RTot = RTot[0:order + 1, 0:order + 1, :]
            STot = STot[0:order + 1, 0:num_input * (order + 1), :]
            TTot = TTot[0:num_input * (order + 1), 0:num_input * (order + 1), :]

            M = np.zeros(np.shape(TTot)[0:2])
            for Rj, Sj, Tj in zip(RTot.transpose([2, 0, 1]), STot.transpose([2, 0, 1]), TTot.transpose([2, 0, 1])):
                M = M + Tj - Sj.transpose() @ np.linalg.solve(Rj, Sj)

            A = M[:order * num_input, :order * num_input]
            B = -M[:order * num_input, order * num_input:]

            alpha = np.linalg.solve(A, B)
            Ac_top_left = np.zeros([(order - 1) * num_input, num_input])
            Ac_top_right = np.eye((order - 1) * num_input)
            Ac_bottom = -alpha.transpose()

            Ac = np.concatenate((np.concatenate((Ac_top_left, Ac_top_right), axis=1),
                                 Ac_bottom), axis=0)

            zpoles, V = np.linalg.eig(Ac)
            spoles = -1 / sample_time * np.log(zpoles)
            Lr = V[-num_input:]
            keep_poles = (np.imag(spoles) > 0) & (np.real(spoles) < 0)
            poles = spoles[keep_poles]
            Lr = Lr[:, keep_poles]
            omgr = abs(poles) + omega0
            zetar = -np.real(poles) / omgr
            isort = np.argsort(omgr)
            omgr = omgr[isort]
            zetar = zetar[isort]
            Lr = Lr[:, isort]
            Lr_real = collapse_complex_to_real(Lr, axis=0, preserve_magnitude=True)

            # Now check stability
            pole_list_i = np.zeros(omgr.size, pole_dtype)
            pole_list_i['omega'] = omgr
            pole_list_i['zeta'] = zetar
            if np.shape(Lr)[0] == 1:
                pole_list_i['Lr_complex'] = Lr.flatten()
                pole_list_i['Lr_real'] = Lr_real.flatten()
            else:
                pole_list_i['Lr_complex'] = (Lr.T)
                pole_list_i['Lr_real'] = (Lr_real.T)

            self.pole_list.append(pole_list_i)

        # Now analyze stability
        last_poles = None
        for i_order, (order, poles) in enumerate(zip(self.polynomial_orders[::-1], self.pole_list[::-1])):
            if i_order > 0:
                for i_pole, pole in enumerate(poles):
                    previous_omegas = last_poles['omega']
                    if previous_omegas.size == 0:
                        continue
                    # Check frequency
                    frequency_differences = np.abs(
                        pole['omega'] - previous_omegas) / previous_omegas
                    closest_freq_index = np.argmin(frequency_differences)
                    if frequency_differences[closest_freq_index] < frequency_stability_threshold:
                        poles['freq_stable'][i_pole] = True
                        # Now check damping
                        previous_zeta = last_poles['zeta'][closest_freq_index]
                        if abs(pole['zeta'] - previous_zeta) / previous_zeta < damping_stability_threshold:
                            poles['damp_stable'][i_pole] = True
                            # Now check participation factor
                            previous_Lr = last_poles['Lr_complex'][closest_freq_index]
                            if np.mean(np.abs(np.abs(pole['Lr_complex']) - np.abs(previous_Lr)) / np.abs(previous_Lr)) < modal_participation_threshold:
                                poles['part_stable'][i_pole] = True
            last_poles = poles.copy()

    def plot_stability(self, no_converge_marger='rx',
                       freq_converge_marker='b^',
                       damp_converge_marker='bs',
                       full_converge_marker='go',
                       label_poles=False, order_range=None,
                       mif_type='cmif', *mif_args, **mif_kwargs):
        cmif = self.frfs.compute_mif(mif_type, *mif_args, **mif_kwargs)

        ax = cmif.plot()

        ax.set_yscale('log')
        ax_poles = ax.twinx()

        for j, (order, poles) in enumerate(zip(self.polynomial_orders, self.pole_list)):
            if not order_range is None:
                if order < order_range[0]:
                    continue
                if order > order_range[1]:
                    continue
            frequencies = poles['omega'] / (2 * np.pi)
            freq_convergence = poles['freq_stable']
            damp_convergence = poles['damp_stable']
            part_convergence = poles['part_stable']
            for i, (freq, fc, dc, pc) in enumerate(zip(frequencies, freq_convergence, damp_convergence, part_convergence)):
                if pc:
                    marker = full_converge_marker
                elif dc:
                    marker = damp_converge_marker
                elif fc:
                    marker = freq_converge_marker
                else:
                    marker = no_converge_marger
                ax_poles.plot(freq, order, marker, markerfacecolor='none')
                if label_poles:
                    ax_poles.text(freq, order, '{:},{:}'.format(j, i))
        freq = self.frequencies[self.frequency_slice]
        ax.set_xlim(freq.min(), freq.max())
        return ax, ax_poles

    def pole_list_from_indices(self, indices):
        index_list = [self.pole_list[order_index][pole_index, np.newaxis]
                      for order_index, pole_index in indices]
        if len(index_list) == 0:
            return np.zeros((0,), dtype=self.pole_list[0].dtype)
        return np.concatenate(index_list)

    def analyze_pole_convergence(self, pole_or_index,
                                 frequency_stability_threshold=0.01,
                                 damping_stability_threshold=0.1,
                                 modal_participation_threshold=0.15,
                                 subplots_kwargs={},
                                 label_order=True,
                                 no_converge_marger='rx',
                                 freq_converge_marker='b^',
                                 damp_converge_marker='bs',
                                 full_converge_marker='go'):
        try:
            omega = pole_or_index['omega']
            zeta = pole_or_index['zeta']
            part = pole_or_index['Lr_complex']
        except (IndexError, TypeError):
            pole_or_index = self.pole_list[pole_or_index[0]][pole_or_index[1]]
            omega = pole_or_index['omega']
            zeta = pole_or_index['zeta']
            part = pole_or_index['Lr_complex']
        fig, ax = plt.subplots(2, 2, **subplots_kwargs)
        ax = ax.flatten()
        for poles, order in zip(self.pole_list, self.polynomial_orders):
            if len(poles) < 1:
                continue
            frequency_differences = np.abs(poles['omega'] - omega) / omega
            closest_freq_index = np.argmin(frequency_differences)
            pole = poles[closest_freq_index]
            if pole['part_stable']:
                marker = full_converge_marker
            elif pole['damp_stable']:
                marker = damp_converge_marker
            elif pole['freq_stable']:
                marker = freq_converge_marker
            else:
                marker = no_converge_marger
            for a, index in zip(ax[:3], ['omega', 'zeta', 'Lr_real']):
                a.plot(order * np.ones(pole[index].shape),
                       pole[index], marker, markerfacecolor='none')
                if label_order:
                    try:
                        for val in pole[index]:
                            a.text(order, val, '{:}'.format(order))
                    except TypeError:
                        a.text(order, pole[index], '{:}'.format(order))
            ax[-1].plot(pole['Lr_complex'].real, pole['Lr_complex'].imag,
                        marker, markerfacecolor='none')
            if label_order:
                for val in pole['Lr_complex']:
                    ax[-1].text(val.real, val.imag, '{:}'.format(order))
        for a, title in zip(ax, ['Angular Frequency', 'Damping', 'Participation Factor (real)', 'Participation Factor (complex)']):
            a.set_title(title)

    def compute_residues(self, poles,
                         residuals=True, real_modes=False, weighting='magnitude'):
        self.natural_frequencies = poles['omega'] / (2 * np.pi)
        self.damping_ratios = poles['zeta']
        self.participation_factors = poles['Lr_complex']
        self.residues, self.frfs_synth_residue, self.frfs_synth_residual = (
            modeshape_compute_residues(
                self.frfs, self.natural_frequencies, self.damping_ratios,
                real_modes, residuals, self.min_frequency, self.max_frequency,
                weighting, self.displacement_derivative))

    def compute_shapes(self, selection_criteria=ShapeSelection.DRIVE_POINT_COEFFICIENT):
        self.shapes, negative_drive_points = modeshape_compute_shapes(
            self.natural_frequencies, self.damping_ratios,
            self.frfs.coordinate, self.residues, selection_criteria,
            self.participation_factors)
        self.negative_drive_points = negative_drive_points

    @property
    def frequencies(self):
        return self.frfs[0, 0].abscissa

    @property
    def angular_frequencies(self):
        return 2 * np.pi * self.frequencies

    @property
    def frequency_spacing(self):
        return np.mean(np.diff(self.frequencies))


class PolyPy_Stability(QWidget):
    def __init__(self, polypy_gui, polypy_tabwidget,
                 frfs, frequency_region, poles, displacement_derivative):
        # Load in the GUI
        super(PolyPy_Stability, self).__init__(polypy_tabwidget)
        uic.loadUi(os.path.join(os.path.abspath(os.path.dirname(
            os.path.abspath(__file__))), 'polypy_stability_diagram.ui'), self)
        self.polypy_gui = polypy_gui
        self.stability_tabwidget = polypy_tabwidget
        self.tabwidget_index = self.stability_tabwidget.count()
        self.stability_tabwidget.addTab(self, "{:0.2f}--{:0.2f} Hz".format(*frequency_region))

        # Set up colormaps
        self.cm = cm.Dark2
        self.cm_mod = 8
        self.compare_cm = cm.Paired
        self.compare_cm_mod = 12

        self.frfs = frfs
        self.pole_markers = []
        self.pole_positions = np.zeros((0, 2), dtype=float)
        self.pole_indices = np.zeros((0, 2), dtype=int)
        self.previous_closest_marker_index = None
        self.selected_poles = set([])

        # Set up the second axis for the stabilizaton plot
        self.pole_plot = self.stabilization_diagram.getPlotItem()
        self.pole_plot.setLabels(left='Polynomial Order')
        self.stabilization_plot = pyqtgraph.ViewBox()
        self.pole_plot.setAxisItems(
            {'right': pyqtgraph.AxisItem(orientation='right', showValues=False)})
        self.pole_plot.showAxis('right')
        self.pole_plot.scene().addItem(self.stabilization_plot)
        self.pole_plot.getAxis('right').linkToView(self.stabilization_plot)
        self.stabilization_plot.setXLink(self.pole_plot)
        self.pole_plot.getAxis('right').setLabel('Mode Indicator Function')
        self.update_stability_plot_views()
        self.update_stabilization_plot()

        self.connect_callbacks()

        self.polypy = PolyPy(frfs, *frequency_region, displacement_derivative)
        self.polypy.compute_poles(poles)

        self.plot_poles()

    def connect_callbacks(self):
        self.stabilization_cmif_selection.clicked.connect(self.update_stabilization_plot)
        self.stabilization_qmif_selection.clicked.connect(self.update_stabilization_plot)
        self.stabilization_mmif_selection.clicked.connect(self.update_stabilization_plot)
        self.stabilization_nmif_selection.clicked.connect(self.update_stabilization_plot)
        self.pole_plot.vb.sigResized.connect(self.update_stability_plot_views)
        self.pole_plot.scene().sigMouseMoved.connect(self.mouseMoved)
        self.pole_plot.scene().sigMouseClicked.connect(self.mouseClicked)
        self.discard_button.clicked.connect(self.discard)

    def discard(self):
        self.polypy_gui.stability_diagrams.pop(self.tabwidget_index)
        for i in range(len(self.polypy_gui.stability_diagrams)):
            self.polypy_gui.stability_diagrams[i].tabwidget_index = i
        self.stability_tabwidget.removeTab(self.tabwidget_index)
        self.polypy_gui.pole_selection_changed()

    def plot_poles(self):
        self.pole_markers = []
        self.pole_positions = []
        self.pole_indices = []
        for j, (order, poles) in enumerate(zip(self.polypy.polynomial_orders, self.polypy.pole_list)):
            frequencies = poles['omega'] / (2 * np.pi)
            freq_convergence = poles['freq_stable']
            damp_convergence = poles['damp_stable']
            part_convergence = poles['part_stable']
            for i, (freq, fc, dc, pc) in enumerate(zip(frequencies, freq_convergence, damp_convergence, part_convergence)):
                if pc:
                    pen = pyqtgraph.mkPen(color=(0, 128, 0))
                    symbol = 'o'
                elif dc:
                    pen = pyqtgraph.mkPen(color='b')
                    symbol = 's'
                elif fc:
                    pen = pyqtgraph.mkPen(color='b')
                    symbol = 't1'
                else:
                    pen = pyqtgraph.mkPen(color='r')
                    symbol = 'x'
                item = pyqtgraph.ScatterPlotItem(
                    [freq], [order], pen=pen, symbol=symbol, symbolPen=pen, brush=(0, 0, 0, 0))
                self.pole_markers.append(item)
                self.pole_indices.append((j, i))
                self.pole_positions.append((freq, order))
                self.pole_plot.addItem(item)
        freq = self.polypy.frequencies[self.polypy.frequency_slice]
        self.pole_plot.setRange(xRange=(freq.min(), freq.max()),
                                yRange=(self.polypy.polynomial_orders.min(),
                                        self.polypy.polynomial_orders.max()), padding=0.1)
        self.pole_indices = np.array(self.pole_indices)
        self.pole_positions = np.array(self.pole_positions)

    def update_stabilization_plot(self):
        self.stabilization_plot.clear()
        if self.stabilization_cmif_selection.isChecked():
            cmif = self.frfs.compute_cmif()
            for i, curve in enumerate(cmif):
                pen = pyqtgraph.mkPen(color=[int(255 * v) for v in self.cm(i % self.cm_mod)])
                self.stabilization_plot.addItem(pyqtgraph.PlotCurveItem(
                    curve.abscissa, np.log10(curve.ordinate), pen=pen))
        elif self.stabilization_qmif_selection.isChecked():
            qmif = self.frfs.compute_cmif(
                part='real' if self.polypy_gui.datatype_selector.currentIndex() == 1 else 'imag')
            for i, curve in enumerate(qmif):
                pen = pyqtgraph.mkPen(color=[int(255 * v) for v in self.cm(i % self.cm_mod)])
                self.stabilization_plot.addItem(pyqtgraph.PlotCurveItem(
                    curve.abscissa, np.log10(curve.ordinate), pen=pen))
        elif self.stabilization_mmif_selection.isChecked():
            mmif = self.frfs.compute_mmif()
            for i, curve in enumerate(mmif):
                pen = pyqtgraph.mkPen(color=[int(255 * v) for v in self.cm(i % self.cm_mod)])
                self.stabilization_plot.addItem(pyqtgraph.PlotCurveItem(
                    curve.abscissa, curve.ordinate, pen=pen))
        elif self.stabilization_nmif_selection.isChecked():
            nmif = self.frfs.compute_nmif()
            pen = pyqtgraph.mkPen(color=[int(255 * v) for v in self.cm(0 % self.cm_mod)])
            self.stabilization_plot.addItem(pyqtgraph.PlotCurveItem(
                nmif.abscissa, nmif.ordinate), pen=pen)

    def update_stability_plot_views(self):
        self.stabilization_plot.setGeometry(self.pole_plot.vb.sceneBoundingRect())
        self.stabilization_plot.linkedViewChanged(self.pole_plot.vb, self.stabilization_plot.XAxis)

    def mouseMoved(self, position):
        if self.pole_plot.vb.sceneBoundingRect().contains(position):
            # Get aspect ratio of the box
            ar = np.array([self.pole_plot.vb.getAspectRatio(), 1])
            # Get the point in scene coordinates
            mouse_point = self.pole_plot.vb.mapSceneToView(position)
            # print('Mouse Point: {:}'.format((mouse_point.x(), mouse_point.y())))
            # Find closest point to the mouse point
            try:
                closest_marker = np.argmin(np.linalg.norm(
                    (self.pole_positions - np.array([mouse_point.x(), mouse_point.y()])) * ar, axis=-1))
                # print('Closest Marker {:}'.format(self.pole_positions[closest_marker]))
            except ValueError:
                return
            if not self.previous_closest_marker_index is None:
                if self.previous_closest_marker_index in self.selected_poles:
                    order_index, pole_index = self.pole_indices[self.previous_closest_marker_index]
                    pole = self.polypy.pole_list[order_index][pole_index]
                    if pole['part_stable']:
                        brush = (0, 128, 0)
                    elif pole['damp_stable'] or pole['freq_stable']:
                        brush = 'b'
                    else:
                        brush = 'r'
                    self.pole_markers[self.previous_closest_marker_index].setBrush(brush)
                else:
                    self.pole_markers[self.previous_closest_marker_index].setBrush((0, 0, 0, 0))
            self.previous_closest_marker_index = closest_marker
            order_index, pole_index = self.pole_indices[self.previous_closest_marker_index]
            pole = self.polypy.pole_list[order_index][pole_index]
            self.selected_pole_display.setText('Highlighted Pole: {:0.2f} Hz, {:0.3f}% Damping'.format(
                pole['omega'] / (2 * np.pi), pole['zeta'] * 100))
            self.pole_markers[self.previous_closest_marker_index].setBrush((0, 0, 0, 255))

    def mouseClicked(self, event):
        view_position = self.pole_plot.vb.mapSceneToView(event.scenePos())
        clickx = view_position.x()
        clicky = view_position.y()
        [[xmin, xmax], [ymin, ymax]] = self.pole_plot.vb.viewRange()
        if (clickx < xmin) or (clickx > xmax) or (clicky > ymax) or (clicky < ymin):
            return
        if self.previous_closest_marker_index is None:
            return
        if self.previous_closest_marker_index in self.selected_poles:
            self.selected_poles.remove(self.previous_closest_marker_index)
            self.pole_markers[self.previous_closest_marker_index].setBrush((0, 0, 0, 0))
        else:
            order_index, pole_index = self.pole_indices[self.previous_closest_marker_index]
            pole = self.polypy.pole_list[order_index][pole_index]
            if pole['part_stable']:
                brush = (0, 128, 0)
            elif pole['damp_stable'] or pole['freq_stable']:
                brush = 'b'
            else:
                brush = 'r'
            self.selected_poles.add(self.previous_closest_marker_index)
            self.pole_markers[self.previous_closest_marker_index].setBrush(brush)
        self.previous_closest_marker_index = None
        self.selected_pole_display.setText('Highlighted Pole:')
        # Now update the pole list
        self.pole_table.clearContents()
        self.pole_table.setRowCount(len(self.selected_poles))
        selected_pole_list = []
        for index in self.selected_poles:
            order_index, pole_index = self.pole_indices[index]
            pole = self.polypy.pole_list[order_index][pole_index]
            if pole['part_stable']:
                stable = 'All'
            elif pole['damp_stable']:
                stable = 'Damp'
            elif pole['freq_stable']:
                stable = 'Freq'
            else:
                stable = 'None'
            selected_pole_list.append((pole['omega'] / (2 * np.pi),
                                       pole['zeta'], stable))
        for i, (frequency, damping, stable) in enumerate(sorted(selected_pole_list)):
            freq_label = QTableWidgetItem('{:0.2f}'.format(frequency))
            damp_label = QTableWidgetItem('{:0.2f}%'.format(100 * damping))
            stab_label = QTableWidgetItem(stable)
            self.pole_table.setItem(i, 0, freq_label)
            self.pole_table.setItem(i, 1, damp_label)
            self.pole_table.setItem(i, 2, stab_label)
        # Send a notification to the main gui that the pole selection has been
        # updated
        self.polypy_gui.pole_selection_changed()


class PolyPy_GUI(QMainWindow):

    def __init__(self, frf_data: TransferFunctionArray):
        super(PolyPy_GUI, self).__init__()
        uic.loadUi(os.path.join(os.path.abspath(os.path.dirname(
            os.path.abspath(__file__))), 'polypy.ui'), self)
        # Store the FRF
        # TODO Make it so we can load data if none is passed in
        self.frfs = frf_data.reshape_to_matrix()
        self.resynthesis_plots = []
        self.resynthesized_frfs = self.frfs.copy()
        self.resynthesized_frfs.ordinate = 0
        self.frfs_synth_residual = self.resynthesized_frfs.copy()
        self.shapes = ShapeArray((0,), self.frfs.shape[0])
        self.negative_drive_points = np.zeros((0,), dtype=int)
        self.first_stability = True

        # Set up colormaps
        self.cm = cm.Dark2
        self.cm_mod = 8
        self.compare_cm = cm.Paired
        self.compare_cm_mod = 12

        # Initialize the plots
        self.data_plot = self.data_display.getPlotItem()

        # Create the frequency selector
        freq_range = (self.frfs.abscissa.min(),
                      self.frfs.abscissa.max())
        self.frequency_selector = pyqtgraph.LinearRegionItem(values=freq_range,
                                                             orientation='vertical',
                                                             bounds=freq_range,
                                                             )
        self.update_data_plot()

        # Update max and mins of selectors
        self.min_frequency_selector.setRange(*freq_range)
        self.min_frequency_selector.setValue(freq_range[0])
        self.max_frequency_selector.setRange(*freq_range)
        self.max_frequency_selector.setValue(freq_range[1])

        self.lines_at_resonance_spinbox.setMaximum(self.frfs.num_elements)
        self.lines_at_residuals_spinbox.setMaximum(self.frfs.num_elements)
        self.lines_at_residuals_spinbox.setValue(self.frfs.num_elements // 10)

        self.connect_callbacks()
        self.num_responses_label.setText('{:} Responses'.format(self.frfs.shape[0]))
        self.num_references_label.setText('{:} References'.format(self.frfs.shape[1]))
        self.num_frequencies_label.setText('{:} Frequencies'.format(self.frfs.num_elements))
        self.by_order_spinbox.setValue(self.frfs.shape[1])
        self.stability_diagrams = []
        self.geometry = None
        self.shapes = None
        self.setWindowTitle('PolyPy -- Multi-reference Polynomial Curve Fitter')
        self.show()

    @property
    def min_frequency(self):
        min_freq = np.min([sd.polypy.min_frequency for sd in self.stability_diagrams])
        if min_freq == 0:
            min_freq = self.frequencies[1]
        return min_freq

    @property
    def max_frequency(self):
        return np.max([sd.polypy.max_frequency for sd in self.stability_diagrams])

    @property
    def frequencies(self):
        return self.frfs[0, 0].abscissa

    @property
    def frequency_slice(self):
        abscissa_indices = np.ones(self.frequencies.shape, dtype=bool)
        if not self.min_frequency is None:
            abscissa_indices &= (self.frequencies >= self.min_frequency)
        if not self.max_frequency is None:
            abscissa_indices &= (self.frequencies <= self.max_frequency)
        abscissa = self.frequencies[abscissa_indices]
        if abscissa[0] == 0:
            abscissa = abscissa[1:]
        freq_range = np.array((np.min(abscissa), np.max(abscissa)))
        index_range = np.argmin(np.abs(self.frequencies - freq_range[:, np.newaxis]), axis=1)
        return slice(index_range[0], index_range[1] + 1)

    def connect_callbacks(self):
        self.setup_cmif_selection.clicked.connect(self.update_data_plot)
        self.setup_qmif_selection.clicked.connect(self.update_data_plot)
        self.setup_mmif_selection.clicked.connect(self.update_data_plot)
        self.setup_nmif_selection.clicked.connect(self.update_data_plot)
        self.frequency_selector.sigRegionChanged.connect(self.update_frequency_from_region)
        self.min_frequency_selector.valueChanged.connect(self.update_min_frequency)
        self.max_frequency_selector.valueChanged.connect(self.update_max_frequency)
        self.compute_stabilization_button.clicked.connect(self.compute_stabilization)
        self.compute_shapes_button.clicked.connect(self.compute_shapes)
        self.load_geometry_button.clicked.connect(self.load_geometry)
        self.plot_shapes_button.clicked.connect(self.plot_shapes)
        self.export_shapes_button.clicked.connect(self.save_shapes)
        self.create_frf_window_button.clicked.connect(self.create_frf_window)
        self.create_cmif_window_button.clicked.connect(self.create_cmif_window)
        self.create_qmif_window_button.clicked.connect(self.create_qmif_window)
        self.create_mmif_window_button.clicked.connect(self.create_mmif_window)
        self.create_nmif_window_button.clicked.connect(self.create_nmif_window)
        self.complex_modes_checkbox.stateChanged.connect(self.pole_selection_changed)
        self.autoresynthesize_checkbox.stateChanged.connect(self.pole_selection_changed)
        self.weighting_selector.currentIndexChanged.connect(self.pole_selection_changed)
        self.all_frequency_lines_checkbox.stateChanged.connect(self.pole_selection_changed)
        self.lines_at_resonance_spinbox.valueChanged.connect(self.pole_selection_changed)
        self.lines_at_residuals_spinbox.valueChanged.connect(self.pole_selection_changed)
        self.use_residuals_checkbox.stateChanged.connect(self.pole_selection_changed)
        self.export_fit_data_button.clicked.connect(self.export_fit_data)
        self.all_frequency_lines_checkbox.stateChanged.connect(self.show_line_selectors)

    def show_line_selectors(self):
        if self.all_frequency_lines_checkbox.isChecked():
            for widget in [self.lines_at_resonance_label, self.lines_at_resonance_spinbox,
                           self.lines_at_residuals_label, self.lines_at_residuals_spinbox]:
                widget.hide()
        else:
            for widget in [self.lines_at_resonance_label, self.lines_at_resonance_spinbox,
                           self.lines_at_residuals_label, self.lines_at_residuals_spinbox]:
                widget.show()

    def load_geometry(self):
        filename, file_filter = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open Geometry', filter='Numpy Files (*.npz);;Universal Files (*.unv *.uff)')
        if filename == '':
            return
        self.geometry = Geometry.load(filename)

    def set_geometry(self, geometry):
        self.geometry = geometry

    def save_shapes(self):
        if self.shapes is None:
            QtWidgets.QMessageBox.warning(self, 'PyPoly', 'Shapes not created!')
            return
        filename, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Shapes', filter='Numpy Files (*.npy)')
        if filename == '':
            return
        np.sort(self.shapes).save(filename)

    def export_fit_data(self):
        if self.shapes is None:
            QtWidgets.QMessageBox.warning(self, 'PyPoly', 'Shapes not created!')
            return
        filename, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Fit Data', filter='Numpy Files (*.npz)')
        if filename == '':
            return
        shapes = np.sort(self.shapes)
        np.savez(filename, shapes=shapes.view(np.ndarray),
                 frfs=self.frfs.view(np.ndarray),
                 frfs_resynth=self.resynthesized_frfs.view(np.ndarray),
                 frfs_residual=self.frfs_synth_residual.view(np.ndarray))

    def plot_shapes(self):
        if self.geometry is None:
            self.load_geometry()
            if self.geometry is None:
                QtWidgets.QMessageBox.warning(self, 'PyPoly', 'No Geometry Loaded!')
                return
        if self.shapes is None:
            QtWidgets.QMessageBox.warning(self, 'PyPoly', 'Shapes not created!')
            return
        self.geometry.plot_shape(np.sort(self.shapes))

    def pole_selection_changed(self):
        if self.autoresynthesize_checkbox.isChecked():
            self.compute_shapes()

    def create_frf_window(self):
        self.resynthesis_plots.append(
            ('frf', GUIPlot(self.frfs, self.resynthesized_frfs)))
        self.update_resynthesis()

    def create_cmif_window(self):
        self.resynthesis_plots.append(
            ('cmif', GUIPlot(self.frfs.compute_cmif(), self.resynthesized_frfs.compute_cmif())))
        self.resynthesis_plots[-1][-1].ordinate_log = True
        self.resynthesis_plots[-1][-1].actionOrdinate_Log.setChecked(True)
        self.update_resynthesis()

    def create_qmif_window(self):
        self.resynthesis_plots.append(
            ('qmif', GUIPlot(self.frfs.compute_cmif(part='real' if self.datatype_selector.currentIndex() == 1 else 'imag'), self.resynthesized_frfs.compute_cmif(part='imag'))))
        self.resynthesis_plots[-1][-1].ordinate_log = True
        self.resynthesis_plots[-1][-1].actionOrdinate_Log.setChecked(True)
        self.update_resynthesis()

    def create_mmif_window(self):
        self.resynthesis_plots.append(
            ('mmif', GUIPlot(self.frfs.compute_mmif(), self.resynthesized_frfs.compute_mmif())))
        self.update_resynthesis()

    def create_nmif_window(self):
        self.resynthesis_plots.append(
            ('nmif', GUIPlot(self.frfs.compute_nmif(), self.resynthesized_frfs.compute_nmif())))
        self.update_resynthesis()

    def update_resynthesis(self):
        # First go through and remove any closed windows
        self.resynthesis_plots = [(function_type, window) for function_type,
                                  window in self.resynthesis_plots if window.isVisible()]
        # Now go and update the resynthesized FRFs
        data_computed = {'frf': self.resynthesized_frfs.flatten()}
        for function_type, window in self.resynthesis_plots:
            if not function_type in data_computed:
                if function_type == 'cmif':
                    data_computed[function_type] = self.resynthesized_frfs.compute_cmif().flatten()
                elif function_type == 'qmif':
                    data_computed[function_type] = self.resynthesized_frfs.compute_cmif(
                        part='real' if self.datatype_selector.currentIndex() == 1 else 'imag').flatten()
                elif function_type == 'mmif':
                    data_computed[function_type] = self.resynthesized_frfs.compute_mmif().flatten()
                elif function_type == 'nmif':
                    data_computed[function_type] = self.resynthesized_frfs.compute_nmif().flatten()
            window.compare_data = data_computed[function_type]
            window.update()

    def compute_shapes(self):
        try:
            # Assemble pole list
            poles = []
            for sd in self.stability_diagrams:
                pole_indices = [sd.pole_indices[val] for val in sd.selected_poles]
                poles.append(sd.polypy.pole_list_from_indices(pole_indices))
            poles = np.concatenate(poles)
            if poles.size == 0:
                self.resynthesized_frfs = self.frfs.copy()
                self.resynthesized_frfs.ordinate = 0
                self.frfs_synth_residual = self.resynthesized_frfs.copy()
                self.shapes = ShapeArray((0,), self.frfs.shape[0])
                self.negative_drive_points = np.zeros((0,), dtype=int)
                return
            natural_frequencies = poles['omega'] / (2 * np.pi)
            damping_ratios = poles['zeta']
            participation_factors = poles['Lr_complex']
            H = self.frfs.ordinate[..., self.frequency_slice]
            num_outputs, num_inputs, num_elements = H.shape
            residues, frfs_synth_residue, self.frfs_synth_residual = (
                modeshape_compute_residues(
                    self.frfs, natural_frequencies, damping_ratios,
                    not self.complex_modes_checkbox.isChecked(
                    ), self.use_residuals_checkbox.isChecked(), self.min_frequency, self.max_frequency,
                    'magnitude' if self.weighting_selector.currentIndex() == 0 else 'uniform',
                    self.datatype_selector.currentIndex(),
                    None if self.all_frequency_lines_checkbox.isChecked() else self.lines_at_resonance_spinbox.value(),
                    self.lines_at_residuals_spinbox.value()))
            self.shapes, self.negative_drive_points = modeshape_compute_shapes(
                natural_frequencies, damping_ratios,
                self.frfs.coordinate, residues, ShapeSelection.DRIVE_POINT_COEFFICIENT,
                participation_factors)
            self.resynthesized_frfs = (self.shapes.compute_frf(self.frfs[0, 0].abscissa[self.frequency_slice],
                                                               np.unique(
                                                                   self.frfs.response_coordinate),
                                                               np.unique(
                                                                   self.frfs.reference_coordinate),
                                                               self.datatype_selector.currentIndex())
                                       + self.frfs_synth_residual)
            self.update_resynthesis()
        except Exception:
            print(traceback.format_exc())

    def compute_stabilization(self):
        self.compute_stabilization_button.setEnabled(False)
        poles = np.arange(self.min_order_spinbox.value(),
                          self.max_order_spinbox.value() + 1,
                          self.by_order_spinbox.value())
        self.stability_diagrams.append(
            PolyPy_Stability(self, self.stability_tab_widget,
                             self.frfs, self.frequency_selector.getRegion(), poles, self.datatype_selector.currentIndex()))
        self.compute_stabilization_button.setEnabled(True)
        if self.first_stability:
            self.first_stability = False
            num_lines = np.sum((self.frequencies >= self.min_frequency) &
                               (self.frequencies <= self.max_frequency))
            self.lines_at_residuals_spinbox.setValue(num_lines // 10)
        self.stability_tab_widget.setCurrentIndex(len(self.stability_diagrams) - 1)
        self.polypy_tab_widget.setCurrentIndex(1)

    def update_frequency_from_region(self):
        self.min_frequency_selector.blockSignals(True)
        self.max_frequency_selector.blockSignals(True)
        region = self.frequency_selector.getRegion()
        self.min_frequency_selector.setValue(region[0])
        self.max_frequency_selector.setValue(region[1])
        self.min_frequency_selector.blockSignals(False)
        self.max_frequency_selector.blockSignals(False)

    def update_min_frequency(self):
        self.frequency_selector.blockSignals(True)
        self.max_frequency_selector.blockSignals(True)
        if self.min_frequency_selector.value() > self.max_frequency_selector.value():
            self.max_frequency_selector.setValue(self.min_frequency_selector.value())
        self.frequency_selector.setRegion((self.min_frequency_selector.value(),
                                           self.max_frequency_selector.value()))
        self.frequency_selector.blockSignals(False)
        self.max_frequency_selector.blockSignals(False)

    def update_max_frequency(self):
        self.frequency_selector.blockSignals(True)
        self.min_frequency_selector.blockSignals(True)
        if self.min_frequency_selector.value() > self.max_frequency_selector.value():
            self.min_frequency_selector.setValue(self.max_frequency_selector.value())
        self.frequency_selector.setRegion((self.min_frequency_selector.value(),
                                           self.max_frequency_selector.value()))
        self.frequency_selector.blockSignals(False)
        self.min_frequency_selector.blockSignals(False)

    def update_data_plot(self):
        self.data_plot.clear()
        self.data_plot.addItem(self.frequency_selector)
        if self.setup_cmif_selection.isChecked():
            self.data_plot.setLogMode(False, True)
            cmif = self.frfs.compute_cmif()
            for i, curve in enumerate(cmif):
                pen = pyqtgraph.mkPen(color=[int(255 * v) for v in self.cm(i % self.cm_mod)])
                self.data_plot.plot(curve.abscissa, curve.ordinate, pen=pen)
        elif self.setup_qmif_selection.isChecked():
            self.data_plot.setLogMode(False, True)
            qmif = self.frfs.compute_cmif(
                part='real' if self.datatype_selector.currentIndex() == 1 else 'imag')
            for i, curve in enumerate(qmif):
                pen = pyqtgraph.mkPen(color=[int(255 * v) for v in self.cm(i % self.cm_mod)])
                self.data_plot.plot(curve.abscissa, curve.ordinate, pen=pen)
        elif self.setup_mmif_selection.isChecked():
            self.data_plot.setLogMode(False, False)
            mmif = self.frfs.compute_mmif()
            for i, curve in enumerate(mmif):
                pen = pyqtgraph.mkPen(color=[int(255 * v) for v in self.cm(i % self.cm_mod)])
                self.data_plot.plot(curve.abscissa, curve.ordinate, pen=pen)
        elif self.setup_nmif_selection.isChecked():
            self.data_plot.setLogMode(False, False)
            nmif = self.frfs.compute_nmif()
            pen = pyqtgraph.mkPen(color=[int(255 * v) for v in self.cm(0 % self.cm_mod)])
            self.data_plot.plot(nmif.abscissa, nmif.ordinate)
