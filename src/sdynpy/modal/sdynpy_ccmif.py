# -*- coding: utf-8 -*-
"""
Graphical tool for selecting final mode sets from single-reference data

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
from typing import List

import numpy as np
import matplotlib.cm as cm
import pyqtgraph as pqtg

from ..core.sdynpy_data import (TransferFunctionArray, ModeIndicatorFunctionArray,
                                data_array, FunctionTypes, GUIPlot)
from ..core.sdynpy_coordinate import CoordinateArray, outer_product
from ..core.sdynpy_geometry import Geometry
from ..core.sdynpy_shape import ShapeArray, mac, matrix_plot

from qtpy import QtWidgets, uic, QtGui
from qtpy.QtGui import QIcon, QFont
from qtpy.QtCore import Qt, QCoreApplication, QRect
from qtpy.QtWidgets import (QToolTip, QLabel, QPushButton, QApplication,
                            QGroupBox, QWidget, QMessageBox, QHBoxLayout,
                            QVBoxLayout, QSizePolicy, QMainWindow,
                            QFileDialog, QErrorMessage, QListWidget, QListWidgetItem,
                            QLineEdit,
                            QDockWidget, QGridLayout, QButtonGroup, QDialog,
                            QCheckBox, QRadioButton, QMenuBar, QMenu)
try:
    from qtpy.QtGui import QAction
except ImportError:
    from qtpy.QtWidgets import QAction
import traceback


class PropertiesDialog(QDialog):
    def __init__(self, shape, *args, **kwargs):
        super(QDialog, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(os.path.abspath(os.path.dirname(
            os.path.abspath(__file__))), 'mode_properties.ui'), self)
        self.frequencyDoubleSpinBox.setValue(shape.frequency)
        self.dampingDoubleSpinBox.setValue(shape.damping * 100)
        self.comment1LineEdit.setText(shape.comment1)
        self.comment2LineEdit.setText(shape.comment2)
        self.comment3LineEdit.setText(shape.comment3)
        self.comment4LineEdit.setText(shape.comment4)
        self.comment5LineEdit.setText(shape.comment5)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    @staticmethod
    def show(shape, parent=None):
        dialog = PropertiesDialog(shape, parent)
        result = dialog.exec_() == QtWidgets.QDialog.Accepted
        frequency = dialog.frequencyDoubleSpinBox.value()
        damping = dialog.dampingDoubleSpinBox.value()
        comment1 = dialog.comment1LineEdit.text()
        comment2 = dialog.comment2LineEdit.text()
        comment3 = dialog.comment3LineEdit.text()
        comment4 = dialog.comment4LineEdit.text()
        comment5 = dialog.comment5LineEdit.text()
        return (result, frequency, damping, comment1, comment2, comment3, comment4, comment5)


class ColoredCMIF(QMainWindow):
    """An Interactive Window for Selecting Shapes from Single Reference Data"""

    def __init__(self, frfs: List[TransferFunctionArray] = None,
                 shapes: List[ShapeArray] = None):
        super(ColoredCMIF, self).__init__()
        uic.loadUi(os.path.join(os.path.abspath(os.path.dirname(
            os.path.abspath(__file__))), 'colored_cmif.ui'), self)
        if ((not (frfs is None and shapes is None)) and
            (((frfs is None and shapes is not None) or
              (frfs is not None and shapes is None) or
                (len(frfs) != len(shapes))))):
            raise ValueError('frfs and shapes must both be specified and the ' +
                             'same size, or neither must be specified')
        self.ccmif_data = None
        self.ccmif_curves = []
        self.mode_scatter_plots = []
        self.selection_plots = {}
        self.shape_array = [] if shapes is None else [shape.flatten() for shape in shapes]
        self.shape_points_on_plot = None
        self.file_names = []
        self.references = []
        self.cm = cm.Dark2
        self.cmif_plot.setLogMode(False, True)
        self.mac_plot.getPlotItem().vb.setAspectLocked(True)
        self.frf_resynth = None
        self.external_plots = {}
        self.geometry = None
        self.enabled_files = []
        if frfs is not None:
            frf_array = []
            for index, frf in enumerate(frfs):
                this_frf = frf.reshape_to_matrix()
                this_shape = this_frf.shape
                if this_shape[-1] > 1:
                    raise ValueError(
                        'FRF index {:} has more than one reference.  Inputs to ColoredCMIF should be single reference FRFs.'.format(index))
                if index > 0 and frf_array[-1].shape[0] != this_shape[0]:
                    raise ValueError(
                        'FRF index {:} does not have the number of responses as previous indices'.format(index))
                if index > 0 and not np.all(frf_array[-1].response_coordinate == this_frf.response_coordinate):
                    raise ValueError(
                        'FRF index {:} does not have the same responses as previous indices'.format(index))
                frf_array.append(this_frf)
            self.frf_array = np.concatenate(frf_array, axis=-1)
            self.references = self.frf_array[0].reference_coordinate
            for reference in self.references:
                self.file_selector.insertItem(self.file_selector.count() - 1, str(reference))
            self.file_selector.setCurrentIndex(0)
            self.file_names = [str(ref) for ref in self.references]
            self.selected_modes = [[False for shape in shape_array]
                                   for shape_array in self.shape_array]
            self.enabled_files = [True for i in range(self.frf_array.shape[-1])]
            self.compute_ccmif()
            self.plot_ccmif()
            for shape, reference in zip(self.shape_array, self.references):
                shape.comment1 = str(reference)
                # print(shape.comment1)
            self.update_shape_list(no_load=True)
        else:
            self.frf_array = None
            self.selected_modes = []

        self.connect_callbacks()
        self.show()

    def connect_callbacks(self):
        self.line_width_selector.valueChanged.connect(self.update_line_width)
        self.mode_selector.itemSelectionChanged.connect(self.update_selection)
        self.file_selector.activated.connect(self.update_shape_list)
        self.cmif_plot.scene().sigMouseClicked.connect(self.clicked_point)
        self.resynthesize_button.clicked.connect(self.resynthesize)
        self.plot_frfs_button.clicked.connect(self.plot_frfs)
        self.plot_cmifs_button.clicked.connect(self.plot_cmifs)
        self.load_geometry_button.clicked.connect(self.load_geometry)
        self.plot_shapes_button.clicked.connect(self.plot_shapes)
        self.save_shapes_button.clicked.connect(self.save_shapes)
        self.save_progress_button.clicked.connect(self.save_progress)
        self.load_progress_button.clicked.connect(self.load_progress)
        self.export_mode_table_button.clicked.connect(self.export_mode_table)
        self.send_to_new_figure_button.clicked.connect(self.export_figure)
        self.cluster_modes_button.clicked.connect(self.cluster_modes)
        self.mark_modes_checkbox.stateChanged.connect(self.plot_ccmif)
        self.plot_vertical_lines_checkbox.stateChanged.connect(self.plot_ccmif)
        self.label_selector.currentIndexChanged.connect(self.plot_ccmif)
        self.part_selector.currentIndexChanged.connect(self.compute_and_plot_ccmif)
        self.enable_button.clicked.connect(self.enable_file)
        self.disable_button.clicked.connect(self.disable_file)
        self.remove_file_button.clicked.connect(self.remove_file)
        self.replace_file_button.clicked.connect(self.replace_file)
        self.mode_selector.itemDoubleClicked.connect(self.set_mode_properties)
        self.cmif_plot.getPlotItem().ctrl.logXCheck.toggled.connect(self.plot_ccmif)
        self.cmif_plot.getPlotItem().ctrl.logYCheck.toggled.connect(self.plot_ccmif)
        self.autoresynth_checkbox.stateChanged.connect(self.toggle_auto_resynth)

    def toggle_auto_resynth(self):
        self.resynthesize_button.setEnabled(not self.autoresynth_checkbox.isChecked())

    def set_mode_properties(self, item):
        try:
            shape_index = [self.mode_selector.item(i) for i in range(
                self.mode_selector.count())].index(item)
            file_index = self.file_selector.currentIndex()
            item.setSelected(not item.isSelected())
            self.set_properties(file_index, shape_index)
            self.update_shape_list(no_load=True)
        except Exception:
            print(traceback.format_exc())

    def set_properties(self, file_index, shape_index):
        shape = self.shape_array[file_index][shape_index]
        # if not self.geometry is None:
        #     plotter = self.geometry.plot_shape(shape)
        dialog = PropertiesDialog(shape, self)
        dialog.comment1LineEdit.setReadOnly(True)
        result = dialog.exec_() == QtWidgets.QDialog.Accepted
        if result:
            self.shape_array[file_index][shape_index].comment2 = dialog.comment2LineEdit.text()
            self.shape_array[file_index][shape_index].comment3 = dialog.comment3LineEdit.text()
            self.shape_array[file_index][shape_index].comment4 = dialog.comment4LineEdit.text()
            self.shape_array[file_index][shape_index].comment5 = dialog.comment5LineEdit.text()
        # if not self.geometry is None:
        #     plotter.close()

    def compute_and_plot_ccmif(self):
        self.compute_ccmif()
        self.plot_ccmif()

    def compute_ccmif(self):
        try:
            self.frequencies = self.frf_array[0, 0].abscissa
            # Compute the SVD
            if self.part_selector.currentIndex() == 0:
                H = np.imag(self.frf_array.ordinate)
            elif self.part_selector.currentIndex() == 1:
                H = np.real(self.frf_array.ordinate)
            elif self.part_selector.currentIndex() == 2:
                H = self.frf_array.ordinate
            H = H[:, self.enabled_files, :]
            U, S, Vh = np.linalg.svd(np.moveaxis(H, -1, 0))
            Vh = np.abs(Vh)
            best_references = np.argmax(Vh, axis=-1)
            self.ccmif_data = np.empty(S.shape[1:] + S.shape)
            self.ccmif_data[:] = np.nan
            for i in range(S.shape[-1]):
                logical_indices = best_references == i
                # Extend each side by 1
                logical_indices[1:] += logical_indices[:-1]
                logical_indices[:-1] += logical_indices[1:]
                keep_indices = np.where(logical_indices)
                self.ccmif_data[(i,) + keep_indices] = S[keep_indices]
            # Find the shape points on the plot
            self.shape_points_on_plot = []
            for shape_array, ccmif_data in zip(self.enabled_shape_array, self.ccmif_data):
                point_locations = np.array(
                    [np.interp(shape_array.frequency, self.frequencies, data) for data in ccmif_data.T])
                self.shape_points_on_plot.append(point_locations)
        except Exception:
            print(traceback.format_exc())

    @property
    def enabled_shape_array(self):
        return [shape for i, shape in enumerate(self.shape_array) if self.enabled_files[i]]

    @property
    def enabled_selected_modes(self):
        return [selection for i, selection in enumerate(self.selected_modes) if self.enabled_files[i]]

    def update_shape_list(self, event=None, no_load=False):
        # print(event,no_load)
        if self.file_selector.currentIndex() == self.file_selector.count() - 1 and (not no_load):
            self.load_file()
            return
        self.mode_selector.blockSignals(True)
        shape_array = self.shape_array[self.file_selector.currentIndex()]
        self.mode_selector.clear()
        for i, shape in enumerate(shape_array):
            self.mode_selector.addItem('{:}: {:0.2f} Hz, {:0.2f}%, {:}'.format(
                i + 1, shape.frequency, shape.damping * 100, shape.comment2))
        selected_modes = self.selected_modes[self.file_selector.currentIndex()]
        # print(selected_modes)
        for i, selected in enumerate(selected_modes):
            if selected:
                self.mode_selector.item(i).setSelected(True)
        if self.enabled_files[self.file_selector.currentIndex()]:
            self.disable_button.setEnabled(True)
            self.enable_button.setEnabled(False)
            self.mode_selector.setEnabled(True)
        else:
            self.disable_button.setEnabled(False)
            self.enable_button.setEnabled(True)
            self.mode_selector.setEnabled(False)
        self.mode_selector.blockSignals(False)

    def load_file(self):
        filename, file_filter = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open Fit Data', filter='Numpy Files (*.npz)')
        if filename == '':
            return False
        else:
            try:
                file_data = np.load(filename)
                frf = file_data['frfs'].view(TransferFunctionArray).reshape_to_matrix()
                shapes = file_data['shapes'].view(ShapeArray).flatten()
                # Check sizes of things
                if (self.frf_array is not None) and (self.frf_array.size > 0):
                    if (np.setdiff1d(frf.response_coordinate, self.frf_array.response_coordinate).size > 0 or
                            np.setdiff1d(self.frf_array.response_coordinate, frf.response_coordinate).size > 0):
                        QMessageBox.critical(self, 'Bad Response Data',
                                             'Loaded FRF does not have consistent response coordinates with existing data')
                        return
                if (not len(self.shape_array) == 0):
                    if (np.setdiff1d(shapes.coordinate, self.shape_array[0].coordinate).size > 0 or
                            np.setdiff1d(self.shape_array[0].coordinate, shapes.coordinate).size > 0):
                        QMessageBox.critical(self, 'Bad Response Data',
                                             'Loaded Shape does not have consistent response coordinates with existing data')
                        return
                if frf.shape[-1] > 1:
                    QMessageBox.critical(self, 'Bad Response Data',
                                         'Loaded FRF has multiple references.  Inputs to ColoredCMIF should be single reference FRFs')
                    return
                if shapes.is_complex():
                    QMessageBox.critical(self, 'Complex Shape',
                                         'ColoredCMIF currently does not allow complex shapes')
                    return
                # Otherwise everything looks good
                self.shape_array.append(shapes.flatten())
                self.enabled_files.append(True)
                self.selected_modes.append([False for shape in shapes])
                self.file_names.append(filename)
                if (self.frf_array is None) or (self.frf_array.size == 0):
                    self.frf_array = frf
                    self.references = np.unique(frf.reference_coordinate)
                else:
                    self.frf_array = np.concatenate((self.frf_array, frf), axis=-1)
                    self.references = np.concatenate(
                        (self.references, np.unique(frf.reference_coordinate)))
                self.file_selector.blockSignals(True)
                self.file_selector.insertItem(self.file_selector.count(
                ) - 1, str(self.references[-1]) + ': {:}'.format(filename))
                self.file_selector.setCurrentIndex(self.file_selector.count() - 2)
                self.file_selector.blockSignals(False)
                self.update_shape_list()
                self.compute_and_plot_ccmif()
            except Exception:
                print(traceback.format_exc())

    def plot_ccmif(self):
        try:
            # print('Plotting CCMIF')
            if self.ccmif_data is None:
                QMessageBox.critical(self, 'CCMIF Not Yet Computed',
                                     'CCMIF must be computed before it can be plotted')
            xlog = self.cmif_plot.getPlotItem().ctrl.logXCheck.isChecked()
            ylog = self.cmif_plot.getPlotItem().ctrl.logYCheck.isChecked()
            self.clear_plot()
            self.cmif_plot.addLegend()
            non_disabled_indices = np.arange(len(self.enabled_files))[self.enabled_files]
            for i, data in enumerate(self.ccmif_data):
                pen = pqtg.mkPen(
                    color=[int(255 * v) for v in self.cm(non_disabled_indices[i])], width=self.line_width_selector.value())
                for j, singular_value in enumerate(data.T):
                    if j == 0:
                        self.ccmif_curves.append(
                            self.cmif_plot.plot(self.frequencies, singular_value,
                                                pen=pen, name=str(self.references[non_disabled_indices[i]])))
                    else:
                        self.ccmif_curves.append(
                            self.cmif_plot.plot(self.frequencies, singular_value,
                                                pen=pen))
            if self.mark_modes_checkbox.isChecked():
                for i, (shape_array, point_locations) in enumerate(zip(self.enabled_shape_array, self.shape_points_on_plot)):
                    pen = pqtg.mkPen(color=[int(255 * v) for v in self.cm(i)])
                    brush = pqtg.mkBrush([int(255 * v) for v in self.cm(i)])
                    abscissa = shape_array.frequency
                    for j, ordinate in enumerate(point_locations):
                        if np.all(np.isnan(ordinate)):
                            self.mode_scatter_plots.append(None)
                        else:
                            self.mode_scatter_plots.append(
                                pqtg.PlotDataItem(abscissa, ordinate, pen=None, symbolBrush=brush, symbol='x', symbolPen=pen))
                            self.cmif_plot.addItem(self.mode_scatter_plots[-1])
                            if self.label_selector.currentIndex() == 0:
                                for k, (o, a) in enumerate(zip(ordinate, abscissa)):
                                    if np.isnan(o):
                                        continue
                                    ti = pqtg.TextItem('{:}: {:}'.format(
                                        non_disabled_indices[i] + 1, k + 1), color=[0, 0, 0])
                                    self.cmif_plot.addItem(ti, ignoreBounds=True)
                                    ti.setPos(np.log10(a) if xlog else a,
                                              np.log10(o) if ylog else o)
                            elif self.label_selector.currentIndex() == 1:
                                for o, a in zip(ordinate, abscissa):
                                    if np.isnan(o):
                                        continue
                                    ti = pqtg.TextItem('{:}: {:0.1f}'.format(
                                        non_disabled_indices[i] + 1, a), color=[0, 0, 0])
                                    self.cmif_plot.addItem(ti, ignoreBounds=True)
                                    ti.setPos(np.log10(a) if xlog else a,
                                              np.log10(o) if ylog else o)
            for i, (selections, shapes, point_locations) in enumerate(zip(self.enabled_selected_modes,
                                                                          self.enabled_shape_array,
                                                                          self.shape_points_on_plot)):
                for j, (selection, shape, points) in enumerate(zip(selections, shapes, point_locations.T)):
                    if selection:
                        # print('Drawing {:}'.format((i,j)))
                        for k, point in enumerate(points):
                            if np.isnan(point):
                                continue
                            self.selection_plots[(non_disabled_indices[i], j, k)] = pqtg.PlotDataItem(
                                [shape.frequency], [point], pen=None, symbolBrush=pqtg.mkBrush([0, 0, 0, 0]),
                                symbol='o', symbolPen=pqtg.mkPen(color=[0, 0, 0, 255]))
                            self.selection_plots[(non_disabled_indices[i], j, k)].setZValue(0)
                            self.cmif_plot.addItem(
                                self.selection_plots[(non_disabled_indices[i], j, k)])
            self.mac_plot.clear()
            shapes = self.collect_shapes()
            if shapes.size > 0:
                mac_matrix = mac(shapes)
                self.mac_plot.addItem(pqtg.ImageItem(mac_matrix))
            # print('Finished Plotting CCMIF')
        except Exception:
            print(traceback.format_exc())

    def clear_plot(self):
        # for curve in self.ccmif_curves:
        #     self.cmif_plot.removeItem(curve)
        #     del(curve)
        self.ccmif_curves = []
        # for plot in self.mode_scatter_plots:
        #     if plot is None:
        #         continue
        #     self.cmif_plot.removeItem(plot)
        #     plot.sigPointsClicked.disconnect()
        #     del(plot)
        self.mode_scatter_plots = []
        # for key,value in self.selection_plots.items():
        #     self.cmif_plot.removeItem(value)
        #     del(value)
        self.selection_plots = {}
        self.cmif_plot.clear()

    def clicked_point(self, mouse_event):
        # print('Clicked {:}'.format(repr(mouse_event)))
        # print('Position: {:}'.format(mouse_event.pos()))
        # print('Scene Position: {:}'.format(mouse_event.scenePos()))
        # print('View Position: {:}'.format(self.cmif_plot.getPlotItem().vb.mapSceneToView(mouse_event.scenePos())))
        # print('Button: {:}'.format(repr(mouse_event.button())))
        if mouse_event.button() != Qt.LeftButton:
            return
        xlog = self.cmif_plot.getPlotItem().ctrl.logXCheck.isChecked()
        ylog = self.cmif_plot.getPlotItem().ctrl.logYCheck.isChecked()
        ar = self.cmif_plot.getPlotItem().vb.getAspectRatio()
        # print('Xlog {:}'.format(xlog))
        # print('YLog {:}'.format(ylog))
        # Go through and find the closest point to the click
        min_dist = float('inf')
        min_point = None
        min_point_xy = None
        view_position = self.cmif_plot.getPlotItem().vb.mapSceneToView(mouse_event.scenePos())
        clickx = view_position.x()
        clicky = view_position.y()
        # Check if the click is outside the range
        [[xmin, xmax], [ymin, ymax]] = self.cmif_plot.getPlotItem().vb.viewRange()
        if (clickx < xmin) or (clickx > xmax) or (clicky > ymax) or (clicky < ymin):
            return
        for i, (shapes, point_locations) in enumerate(zip(self.shape_array,
                                                          self.shape_points_on_plot)):
            for j, (shape, points) in enumerate(zip(shapes, point_locations.T)):
                if np.all(np.isnan(points)):
                    continue
                x = np.log10(shape.frequency) if xlog else shape.frequency
                y = np.log10(points) if ylog else points
                dist = np.nanmin(((clickx - x) * ar)**2 + (clicky - y)**2)
                if dist < min_dist:
                    min_dist = dist
                    min_point = (i, j)
                    min_point_xy = (x, y)
        # print('Min Point XY: {:}'.format(min_point_xy))
        # print('Min Point {:}'.format(min_point))
        # print('Minimum Point {:}, {:}'.format(min_point,min_dist))
        file_index, mode_index = min_point
        # Toggle Selected
        self.selected_modes[file_index][mode_index] = not self.selected_modes[file_index][mode_index]
        self.file_selector.setCurrentIndex(file_index)
        self.update_shape_list()
        self.plot_ccmif()
        if self.autoresynth_checkbox.isChecked():
            self.resynthesize()

    def update_line_width(self):
        for i, curve in enumerate(self.ccmif_curves):
            pen = pqtg.mkPen(color=[int(255 * v) for v in self.cm(i // self.frf_array.shape[-1])],
                             width=self.line_width_selector.value())
            curve.setPen(pen)

    def update_selection(self):
        self.selected_modes[self.file_selector.currentIndex()] = [
            self.mode_selector.item(i).isSelected() for i in range(self.mode_selector.count())]
        self.plot_ccmif()
        if self.autoresynth_checkbox.isChecked():
            self.resynthesize()
        # print(self.selected_modes)

    def collect_shapes(self):
        all_shapes = []
        for shapes, keep, enabled, reference in zip(self.shape_array, self.selected_modes, self.enabled_files, self.references):
            if not enabled:
                continue
            this_shapes = shapes[keep].copy()
            this_shapes.comment1 = str(reference)
            all_shapes.append(this_shapes)
        all_shapes = np.concatenate(all_shapes)
        isort = np.argsort(all_shapes.frequency)
        all_shapes = all_shapes[isort]
        return all_shapes

    def resynthesize(self):
        shapes = self.collect_shapes()
        if shapes.size == 0:
            QMessageBox.critical(self, 'No Shapes Selected',
                                 'Please Select Shapes Before Resynthesizing')
            return
        enabled_frf_array = self.frf_array[:, self.enabled_files]
        self.frf_resynth = shapes.compute_frf(self.frequencies,
                                              enabled_frf_array[:, 0].response_coordinate,
                                              enabled_frf_array[0].reference_coordinate,
                                              displacement_derivative=self.data_type_selector.currentIndex())
        if 'cmif' in self.external_plots:
            if self.part_selector.currentIndex() == 0:
                part = 'imag'
            if self.part_selector.currentIndex() == 1:
                part = 'real'
            if self.part_selector.currentIndex() == 2:
                part = 'both'
            exp_cmif = np.concatenate([enabled_frf_array.compute_cmif(
                part)] + [this_frf.compute_cmif(part) for this_frf in enabled_frf_array.T])
            ana_cmif = np.concatenate([self.frf_resynth.compute_cmif(
                part)] + [this_frf.compute_cmif(part) for this_frf in self.frf_resynth.T])
            self.external_plots['cmif'].update_data(exp_cmif, ana_cmif)

        if 'frf' in self.external_plots:
            self.external_plots['frf'].update_data(enabled_frf_array, self.frf_resynth)

    def plot_frfs(self):
        enabled_frf_array = self.frf_array[:, self.enabled_files]
        self.external_plots['frf'] = GUIPlot(enabled_frf_array, self.frf_resynth)

    def plot_cmifs(self):
        enabled_frf_array = self.frf_array[:, self.enabled_files]
        if self.part_selector.currentIndex() == 0:
            part = 'imag'
        if self.part_selector.currentIndex() == 1:
            part = 'real'
        if self.part_selector.currentIndex() == 2:
            part = 'both'
        exp_cmif = np.concatenate([enabled_frf_array.compute_cmif(part)] +
                                  [this_frf.compute_cmif(part) for this_frf in enabled_frf_array.T])
        ana_cmif = np.concatenate([self.frf_resynth.compute_cmif(part)] +
                                  [this_frf.compute_cmif(part) for this_frf in self.frf_resynth.T])
        self.external_plots['cmif'] = GUIPlot(exp_cmif, ana_cmif)
        self.external_plots['cmif'].ordinate_log = True
        self.external_plots['cmif'].actionOrdinate_Log.setChecked(True)
        self.external_plots['cmif'].update()

    def load_geometry(self):
        filename, file_filter = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open Geometry', filter='Numpy Files (*.npz);;Universal Files (*.unv *.uff)')
        if filename == '':
            return
        self.geometry = Geometry.load(filename)

    def plot_shapes(self):
        if self.geometry is None:
            QMessageBox.critical(self, 'No Geometry Loaded',
                                 'Please load geometry prior to plotting shapes')
            return
        shapes = self.collect_shapes()
        self.external_plots['geometry'] = self.geometry.plot_shape(shapes)

    def save_shapes(self):
        filename, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Shapes', filter='Numpy Files (*.npy)')
        if filename == '':
            return
        self.collect_shapes().save(filename)

    def save_progress(self):
        filename, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save CCMIF', filter='CCMIF (*.ccm)')
        if filename == '':
            return

    def load_progress(self):
        filename, file_filter = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load CCMIF', filter='CCMIF (*.ccm)')
        if filename == '':
            return

    def export_mode_table(self):
        filename, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Mode Table', filter='CSV (*.csv);;reStructured Text (*.rst);;Markdown (*.md);;LaTeX (*.tex)')
        if filename == '':
            return
        shapes = self.collect_shapes()
        if file_filter == 'CSV (*.csv)':
            with open(filename, 'w') as f:
                f.write(shapes.mode_table('csv'))
        elif file_filter == 'reStructured Text (*.rst)':
            with open(filename, 'w') as f:
                f.write(shapes.mode_table('rst'))
        elif file_filter == 'Markdown (*.md)':
            with open(filename, 'w') as f:
                f.write(shapes.mode_table('markdown'))
        elif file_filter == 'LaTeX (*.tex)':
            with open(filename, 'w') as f:
                f.write(shapes.mode_table('latex'))

    def export_figure(self):
        pass

    def cluster_modes(self):
        pass

    def enable_file(self):
        self.enabled_files[self.file_selector.currentIndex()] = True
        self.update_shape_list()
        self.compute_and_plot_ccmif()

    def disable_file(self):
        self.enabled_files[self.file_selector.currentIndex()] = False
        self.update_shape_list()
        self.compute_and_plot_ccmif()

    def remove_file(self):
        index_to_remove = self.file_selector.currentIndex()
        self.shape_array.pop(index_to_remove)
        self.enabled_files.pop(index_to_remove)
        self.selected_modes.pop(index_to_remove)
        self.file_names.pop(index_to_remove)
        index_array = [False if i == index_to_remove else True for i in range(
            self.frf_array.shape[-1])]
        self.frf_array = self.frf_array[:, index_array]
        if self.frf_array.shape[-1] == 0:
            self.frf_array = None
        self.references = self.references[index_array]
        # Remove entry from the combobox
        self.file_selector.blockSignals(True)
        self.file_selector.removeItem(index_to_remove)
        self.file_selector.setCurrentIndex(0)
        self.file_selector.blockSignals(False)
        self.update_shape_list(no_load=True)
        self.compute_and_plot_ccmif()

    def replace_file(self):
        pass
