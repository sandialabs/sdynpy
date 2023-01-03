# -*- coding: utf-8 -*-
"""
Graphical Signal Processing tool for computing FRFs and CPSDs

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

from ..core.sdynpy_data import TimeHistoryArray, data_array, FunctionTypes, GUIPlot
from ..core.sdynpy_coordinate import CoordinateArray, outer_product
from ..fileio.sdynpy_rattlesnake import read_rattlesnake_output
from ..core.sdynpy_geometry import Geometry
from .sdynpy_smac import SMAC_GUI
from .sdynpy_polypy import PolyPy_GUI
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QCoreApplication, QRect
from PyQt5.QtWidgets import (QToolTip, QLabel, QPushButton, QApplication,
                             QGroupBox, QWidget, QMessageBox, QHBoxLayout,
                             QVBoxLayout, QSizePolicy, QMainWindow,
                             QFileDialog, QErrorMessage, QListWidget, QListWidgetItem,
                             QLineEdit,
                             QDockWidget, QGridLayout, QButtonGroup, QDialog,
                             QCheckBox, QRadioButton, QMenuBar, QMenu, QAction)
import numpy as np
import pyqtgraph as pqtg
import matplotlib.cm as cm
from scipy.signal import get_window
import traceback

class SignalProcessingGUI(QMainWindow):
    """An iteractive window allowing users to compute FRFs"""

    def __init__(self, time_history_array: TimeHistoryArray = None,
                 geometry=None):
        """
        Create a Signal Processing window to compute FRF and other spectral data.

        A TimeHistoryArray can be passed as an argument, or data can be loaded
        from a file.

        Parameters
        ----------
        data_array : TimeHistoryArray
            Time history data to use to compute FRF and other spectral data.
        geometry : Geometry
            Geometry data used to plot transients or deflection shapes

        Returns
        -------
        None.

        """
        self.sample_rate = 1.0
        self.reference_indices = []
        self.window_parameters = {
            'rectangle': (),
            'hann': (),
            'hamming': (),
            'flattop': (),
            'tukey': ('Cosine Fraction:'),
            'blackmanharris': (),
            'exponential': ('Center:', 'tau:'),
            'exponential+force': ('Pulse End:', 'Center:', 'tau:'), }
        self.cm = cm.Dark2
        self.cm_mod = 8
        self.time_selector_references = pqtg.LinearRegionItem(values=[0, 1],
                                                              orientation='vertical',
                                                              bounds=[0, 1],
                                                              )
        self.time_selector_responses = pqtg.LinearRegionItem(values=[0, 1],
                                                             orientation='vertical',
                                                             bounds=[0, 1],
                                                             )
        self.trigger_level_selector_references = pqtg.InfiniteLine(pos=0,
                                                                   angle=0,
                                                                   movable=True,
                                                                   label='Trigger                 ',
                                                                   pen='b',
                                                                   labelOpts={
                                                                       'position': 1.0, 'color': 'b'}
                                                                   )
        self.hysteresis_level_selector_references = pqtg.InfiniteLine(pos=0,
                                                                      angle=0,
                                                                      movable=True,
                                                                      label='Hysteresis                       ',
                                                                      pen='r',
                                                                      labelOpts={
                                                                          'position': 1.0, 'color': 'r'}
                                                                      )
        self.trigger_level_selector_responses = pqtg.InfiniteLine(pos=0,
                                                                  angle=0,
                                                                  movable=True,
                                                                  label='Trigger                 ',
                                                                  pen='b',
                                                                  labelOpts={
                                                                      'position': 1.0, 'color': 'b'}
                                                                  )
        self.hysteresis_level_selector_responses = pqtg.InfiniteLine(pos=0,
                                                                     angle=0,
                                                                     movable=True,
                                                                     label='Hysteresis                       ',
                                                                     pen='r',
                                                                     labelOpts={
                                                                         'position': 1.0, 'color': 'r'}
                                                                     )
        self.frame_start_indices = []
        self.ignore_frames = []
        self.response_rois = []
        self.reference_rois = []
        self.windowed_time_data = None
        self.autospectra_data = None
        self.crossspectra_data = None
        self.frf_data = None
        self.coherence_data = None
        self.geometry = geometry
        self.plots = {}
        self.deflection_actions = []
        self.active_plots = []
        super(SignalProcessingGUI, self).__init__()
        uic.loadUi(os.path.join(os.path.abspath(os.path.dirname(
            os.path.abspath(__file__))), 'signal_processing.ui'), self)
        self.responsesPlot.addItem(self.trigger_level_selector_responses)
        self.referencesPlot.addItem(self.trigger_level_selector_references)
        self.responsesPlot.addItem(self.hysteresis_level_selector_responses)
        self.referencesPlot.addItem(self.hysteresis_level_selector_references)
        self.responsesPlot.addItem(self.time_selector_responses)
        self.referencesPlot.addItem(self.time_selector_references)
        for widget in [self.trigger_level_selector_responses,
                       self.trigger_level_selector_references,
                       self.hysteresis_level_selector_responses,
                       self.hysteresis_level_selector_references]:
            widget.setVisible(False)
        self.connect_callbacks()
        if not time_history_array is None:
            self.time_history_data = time_history_array.flatten()
            self.initialize_ui()
        self.setWindowTitle('Graphical Signal Processing Tool')
        self.show()

    def connect_callbacks(self):
        self.overlapDoubleSpinBox.valueChanged.connect(self.overlapChanged)
        self.overlapSamplesSpinBox.valueChanged.connect(self.overlapSamplesChanged)
        self.framesSpinBox.valueChanged.connect(self.framesChanged)
        self.startTimeDoubleSpinBox.valueChanged.connect(self.startTimeChanged)
        self.endTimeDoubleSpinBox.valueChanged.connect(self.endTimeChanged)
        self.frameSizeSpinBox.valueChanged.connect(self.frameSizeChanged)
        self.frequencyLinesSpinBox.valueChanged.connect(self.frequencyLinesChanged)
        self.frameTimeDoubleSpinBox.valueChanged.connect(self.frameTimeChanged)
        self.frequencySpacingDoubleSpinBox.valueChanged.connect(self.frequencySpacingChanged)
        self.windowComboBox.currentIndexChanged.connect(self.windowChanged)
        self.typeComboBox.currentIndexChanged.connect(self.typeChanged)
        self.referencesSelector.itemSelectionChanged.connect(self.referencesChanged)
        self.responsesSelector.itemSelectionChanged.connect(self.responsesChanged)
        self.referencesSelector.itemDoubleClicked.connect(self.sendToResponse)
        self.responsesSelector.itemDoubleClicked.connect(self.sendToReference)
        self.time_selector_references.sigRegionChanged.connect(self.updateTimeFromReference)
        self.time_selector_responses.sigRegionChanged.connect(self.updateTimeFromResponse)
        self.trigger_level_selector_references.sigPositionChanged.connect(
            self.updateTriggerFromReference)
        self.trigger_level_selector_responses.sigPositionChanged.connect(
            self.updateTriggerFromResponse)
        self.hysteresis_level_selector_references.sigPositionChanged.connect(
            self.updateHysteresisFromReference)
        self.hysteresis_level_selector_responses.sigPositionChanged.connect(
            self.updateHysteresisFromResponse)
        self.levelDoubleSpinBox.valueChanged.connect(self.levelChanged)
        self.hysteresisLevelDoubleSpinBox.valueChanged.connect(self.hysteresisChanged)
        self.pretriggerDoubleSpinBox.valueChanged.connect(self.pretriggerChanged)
        self.acceptanceComboBox.currentIndexChanged.connect(self.acceptanceChanged)
        self.responsesPlot.getViewBox().sigYRangeChanged.connect(self.responseViewChanged)
        self.referencesPlot.getViewBox().sigYRangeChanged.connect(self.referenceViewChanged)
        self.computeButton.clicked.connect(self.compute)
        self.plotWindowedTimeHistoryButton.clicked.connect(self.showWindowedTimeHistory)
        self.saveWindowedTimeHistoryButton.clicked.connect(self.saveWindowedTimeHistory)
        self.plotFRFButton.clicked.connect(self.showFRF)
        self.saveFRFButton.clicked.connect(self.saveFRF)
        self.plotCoherenceButton.clicked.connect(self.showCoherence)
        self.saveCoherenceButton.clicked.connect(self.saveCoherence)
        self.plotAutospectraButton.clicked.connect(self.showAutospectra)
        self.saveAutospectraButton.clicked.connect(self.saveAutospectra)
        self.plotCrossspectraButton.clicked.connect(self.showCrossspectra)
        self.saveCrossspectraButton.clicked.connect(self.saveCrossspectra)
        self.actionSend_to_SMAC.triggered.connect(self.analyzeSMAC)
        self.actionSend_to_PolyPy.triggered.connect(self.analyzePolyPy)
        self.actionLoad_Geometry.triggered.connect(self.loadGeometry)
        self.actionVisualize_with_TransientPlotter.triggered.connect(self.plotTransient)
        self.actionLoad_Data.triggered.connect(self.loadData)

    def block_averaging_signals(self, block: bool):
        for widget in [self.overlapDoubleSpinBox,
                       self.framesSpinBox,
                       self.overlapSamplesSpinBox,
                       self.channelComboBox,
                       self.slopeComboBox,
                       self.levelDoubleSpinBox,
                       self.hysteresisLevelDoubleSpinBox,
                       self.pretriggerDoubleSpinBox]:
            widget.blockSignals(block)

    def block_data_range_signals(self, block: bool):
        for widget in [self.startTimeDoubleSpinBox,
                       self.endTimeDoubleSpinBox,
                       self.time_selector_references,
                       self.time_selector_responses]:
            widget.blockSignals(block)

    def block_sampling_signals(self, block: bool):
        for widget in [self.frameSizeSpinBox,
                       self.frequencyLinesSpinBox,
                       self.frameTimeDoubleSpinBox,
                       self.frequencySpacingDoubleSpinBox]:
            widget.blockSignals(block)

    def reset_ui(self):
        self.sample_rate = 1.0
        self.reference_indices = []
        self.signalsSpinBox.setValue(0)
        self.referencesSpinBox.setValue(0)
        self.responsesSpinBox.setValue(0)
        self.samplesSpinBox.setValue(0)
        self.sampleRateDoubleSpinBox.setValue(0)
        self.durationDoubleSpinBox.setValue(0)
        self.startTimeDoubleSpinBox.setalue(0)
        self.endTimeDoubleSpinBox.setValue(0)
        self.frameSizeSpinBox.setValue(0)
        self.windowComboBox.setCurrentIndex(0)
        self.typeComboBox.setCurrentIndex(0)
        self.channelComboBox.clear()

    def initialize_ui(self):
        # Set the information
        if not self.time_history_data.validate_common_abscissa():
            QMessageBox.critical(self, 'Invalid Abscissa',
                                 'Time histories must have identical abscissa')
            self.reset_ui()
            return
        dt = np.diff(self.time_history_data.abscissa)
        if not np.allclose(dt, dt[0]):
            QMessageBox.critical(self, 'Invalid Abscissa',
                                 'Time histories must have equally spaced timesteps')
            self.reset_ui()
            return
        self.block_averaging_signals(True)
        self.block_data_range_signals(True)
        self.block_sampling_signals(True)
        dt = np.mean(dt)
        self.sample_rate = 1 / dt
        self.signalsSpinBox.setValue(self.time_history_data.size)
        self.referencesSpinBox.setValue(0)
        self.responsesSpinBox.setValue(self.time_history_data.size)
        self.samplesSpinBox.setValue(self.time_history_data.num_elements)
        self.sampleRateDoubleSpinBox.setValue(self.sample_rate)
        self.durationDoubleSpinBox.setValue(dt * self.time_history_data.num_elements)
        self.time_selector_references.setBounds((0, dt * self.time_history_data.num_elements))
        self.time_selector_responses.setBounds((0, dt * self.time_history_data.num_elements))
        self.startTimeDoubleSpinBox.setMaximum(dt * self.time_history_data.num_elements)
        self.startTimeDoubleSpinBox.setValue(0)
        self.startTimeDoubleSpinBox.setSingleStep(dt)
        self.startTimeDoubleSpinBox.setDecimals(len(str(int(self.sample_rate))) + 2)
        self.endTimeDoubleSpinBox.setMaximum(dt * self.time_history_data.num_elements)
        self.endTimeDoubleSpinBox.setValue(dt * self.time_history_data.num_elements)
        self.endTimeDoubleSpinBox.setSingleStep(dt)
        self.endTimeDoubleSpinBox.setDecimals(len(str(int(self.sample_rate))) + 2)
        self.frameSizeSpinBox.setValue(int(self.sample_rate))
        self.frameTimeDoubleSpinBox.setSingleStep(dt)
        self.frameTimeDoubleSpinBox.setDecimals(len(str(int(self.sample_rate))) + 2)
        self.frameSizeSpinBox.setMaximum(self.time_history_data.num_elements)
        self.frameTimeDoubleSpinBox.setMaximum(dt * self.time_history_data.num_elements)
        self.frequencyLinesSpinBox.setMaximum(self.time_history_data.num_elements // 2 + 1)
        self.frequencySpacingDoubleSpinBox.setMinimum(
            1 / (dt * self.time_history_data.num_elements))
        self.windowComboBox.setCurrentIndex(0)
        self.windowParameter1DoubleSpinBox.setVisible(False)
        self.windowParameter2DoubleSpinBox.setVisible(False)
        self.windowParameter3DoubleSpinBox.setVisible(False)
        self.windowParameter1Label.setVisible(False)
        self.windowParameter2Label.setVisible(False)
        self.windowParameter3Label.setVisible(False)
        self.typeComboBox.setCurrentIndex(0)
        self.typeChanged()
        self.overlapDoubleSpinBox.setValue(0)
        # Set up the channel numbers
        self.channelComboBox.clear()
        for i, coordinate in enumerate(self.time_history_data.coordinate[:, 0]):
            self.channelComboBox.addItem('{:}: {:}'.format(i + 1, str(coordinate)), i)
        self.responsesSelector.clear()
        for i, coordinate in enumerate(self.time_history_data.coordinate[:, 0]):
            list_item = QListWidgetItem('{:}: {:}'.format(i + 1, str(coordinate)))
            list_item.setData(Qt.UserRole, i)
            self.responsesSelector.addItem(list_item)
        self.referencesSelector.clear()
        self.referencesPlot.clear()
        self.responsesPlot.clear()
        self.time_selector_references.setRegion((0, dt * self.time_history_data.num_elements))
        self.time_selector_responses.setRegion((0, dt * self.time_history_data.num_elements))
        self.referencesPlot.setXRange(0, dt * self.time_history_data.num_elements)
        self.responsesPlot.setXRange(0, dt * self.time_history_data.num_elements)
        self.block_averaging_signals(False)
        self.block_data_range_signals(False)
        self.block_sampling_signals(False)
        self.frameSizeChanged()
        self.overlapSamplesChanged()

    def create_rois(self):
        xsize = self.frameSizeSpinBox.value() / self.sample_rate
        for roi in self.response_rois:
            self.responsesPlot.removeItem(roi)
        for roi in self.reference_rois:
            self.referencesPlot.removeItem(roi)
        self.response_rois.clear()
        self.reference_rois.clear()
        for roi_list, plot in zip([self.response_rois, self.reference_rois],
                                  [self.responsesPlot, self.referencesPlot]):
            yrange = plot.getViewBox().viewRange()[1]
            ysize = (yrange[1] - yrange[0]) * .9
            ystart = yrange[0] + 0.05 * (yrange[1] - yrange[0])
            for frame_start in self.frame_start_indices:
                xstart = frame_start / self.sample_rate
                roi = pqtg.ROI((xstart, ystart), (xsize, ysize),
                               movable=False, rotatable=False,
                               resizable=False, removable=False,
                               pen=pqtg.mkPen(color=(0, 125, 0), width=2.0),
                               hoverPen=pqtg.mkPen(color=(0, 255, 0), width=4.0))
                roi.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
                roi.sigClicked.connect(self.toggleROI)
                roi_list.append(roi)
                plot.addItem(roi, ignoreBounds=True)

    def toggleROI(self, roi):
        print('Clicked ROI {:}'.format(roi))
        if self.acceptanceComboBox.currentIndex() != 0:
            # Find the current index
            for index, (response_roi, reference_roi) in enumerate(zip(self.response_rois,
                                                                      self.reference_rois)):
                if roi is response_roi or roi is reference_roi:
                    print('ROI found at index {:}'.format(index))
                    # Check if it is already ignored
                    if index in self.ignore_frames:
                        self.ignore_frames.pop(self.ignore_frames.index(index))
                        response_roi.setPen(pqtg.mkPen(color=(0, 125, 0), width=2.0))
                        reference_roi.setPen(pqtg.mkPen(color=(0, 125, 0), width=2.0))
                        self.block_averaging_signals(True)
                        self.framesSpinBox.setValue(self.framesSpinBox.value() + 1)
                        self.block_averaging_signals(False)
                    else:
                        self.ignore_frames.append(index)
                        response_roi.setPen(pqtg.mkPen(color=(125, 0, 0), width=2.0))
                        reference_roi.setPen(pqtg.mkPen(color=(125, 0, 0), width=2.0))
                        self.block_averaging_signals(True)
                        self.framesSpinBox.setValue(self.framesSpinBox.value() - 1)
                        self.block_averaging_signals(False)

    def acceptanceChanged(self):
        print('Acceptance Changed')
        if self.acceptanceComboBox.currentIndex() == 0:
            self.ignore_frames.clear()
            for index, (response_roi, reference_roi) in enumerate(zip(self.response_rois,
                                                                      self.reference_rois)):
                response_roi.setPen(pqtg.mkPen(color=(0, 125, 0), width=2.0))
                reference_roi.setPen(pqtg.mkPen(color=(0, 125, 0), width=2.0))

    def referenceViewChanged(self):
        print('Reference View Changed')
        yrange = self.referencesPlot.getViewBox().viewRange()[1]
        print('YRange: {:}'.format(yrange))
        ysize = (yrange[1] - yrange[0]) * .9
        ystart = yrange[0] + 0.05 * (yrange[1] - yrange[0])
        print('YStart: {:}'.format(ystart))
        print('YSize: {:}'.format(ysize))
        print('YEnd: {:}'.format(ystart + ysize))
        for roi in self.reference_rois:
            currentX, currentY = roi.pos()
            currentxsize, currentysize = roi.size()
            roi.setPos(currentX, ystart)
            roi.setSize((currentxsize, ysize))

    def responseViewChanged(self):
        print('Response View Changed')
        yrange = self.responsesPlot.getViewBox().viewRange()[1]
        ysize = (yrange[1] - yrange[0]) * .9
        ystart = yrange[0] + 0.05 * (yrange[1] - yrange[0])
        for roi in self.response_rois:
            currentX, currentY = roi.pos()
            currentxsize, currentysize = roi.size()
            roi.setPos(currentX, ystart)
            roi.setSize((currentxsize, ysize))

    def get_abscissa_index_range(self):
        return (int(np.round(self.startTimeDoubleSpinBox.value() * self.sample_rate)),
                int(np.round(self.endTimeDoubleSpinBox.value() * self.sample_rate)))

    def overlapChanged(self):
        print('Overlap Changed')
        self.block_averaging_signals(True)
        frame_size = self.frameSizeSpinBox.value()
        overlap_samples = int(np.round(self.overlapDoubleSpinBox.value() / 100 * frame_size))
        self.overlapSamplesSpinBox.setValue(overlap_samples)
        self.overlapDoubleSpinBox.setValue(100 * overlap_samples / frame_size)
        # Compute the number of frames available
        min_index, max_index = self.get_abscissa_index_range()
        total_samples = max_index - min_index + 1
        self.framesSpinBox.setValue(
            int((total_samples - overlap_samples) / (frame_size - overlap_samples)))
        self.block_averaging_signals(False)
        self.frame_start_indices = [min_index + i * (frame_size - overlap_samples)
                                    for i in range(self.framesSpinBox.value())]
        self.ignore_frames = []
        self.create_rois()

    def overlapSamplesChanged(self):
        print('Overlap Samples Changed')
        self.block_averaging_signals(True)
        frame_size = self.frameSizeSpinBox.value()
        overlap_samples = self.overlapSamplesSpinBox.value()
        self.overlapDoubleSpinBox.setValue(100 * overlap_samples / frame_size)
        # Compute the number of frames available
        min_index, max_index = self.get_abscissa_index_range()
        total_samples = max_index - min_index + 1
        self.framesSpinBox.setValue(
            int((total_samples - overlap_samples) / (frame_size - overlap_samples)))
        self.block_averaging_signals(False)
        self.frame_start_indices = [min_index + i * (frame_size - overlap_samples)
                                    for i in range(self.framesSpinBox.value())]
        self.ignore_frames = []
        self.create_rois()

    def framesChanged(self):
        print('Number Frames Changed')
        self.block_averaging_signals(True)
        frame_size = self.frameSizeSpinBox.value()
        frames = self.framesSpinBox.value()
        min_index, max_index = self.get_abscissa_index_range()
        total_samples = max_index - min_index + 1
        try:
            overlap_samples = (frames * frame_size - total_samples) / (frames - 1)
        except ZeroDivisionError:
            overlap_samples = 0
        print('Overlap Samples {:}'.format(overlap_samples))
        if overlap_samples < 0:
            overlap_samples = 0
        self.overlapSamplesSpinBox.setValue(overlap_samples)
        self.overlapDoubleSpinBox.setValue(100 * overlap_samples / frame_size)
        self.block_averaging_signals(False)
        self.frame_start_indices = [min_index + i * (frame_size - overlap_samples)
                                    for i in range(self.framesSpinBox.value())]
        self.ignore_frames = []
        self.create_rois()

    def startTimeChanged(self):
        print('Start Time Changed')
        self.block_data_range_signals(True)
        samples = np.round(self.startTimeDoubleSpinBox.value() * self.sample_rate)
        self.startTimeDoubleSpinBox.setValue(samples / self.sample_rate)
        if self.startTimeDoubleSpinBox.value() > self.endTimeDoubleSpinBox.value():
            self.startTimeDoubleSpinBox.setValue(self.endTimeDoubleSpinBox.value())
        self.time_selector_references.setRegion((self.startTimeDoubleSpinBox.value(),
                                                 self.endTimeDoubleSpinBox.value()))
        self.time_selector_responses.setRegion((self.startTimeDoubleSpinBox.value(),
                                                self.endTimeDoubleSpinBox.value()))
        self.block_data_range_signals(False)
        # Now adjust averaging information
        if self.typeComboBox.currentIndex() == 0:
            self.overlapChanged()
        else:
            self.compute_triggers()
        # Adjust limits on the frame size
        dt = 1 / self.sample_rate
        min_index, max_index = self.get_abscissa_index_range()
        total_samples = max_index - min_index + 1
        self.frameSizeSpinBox.setMaximum(total_samples)
        self.frameTimeDoubleSpinBox.setMaximum(dt * total_samples)
        self.frequencyLinesSpinBox.setMaximum(total_samples // 2 + 1)
        self.frequencySpacingDoubleSpinBox.setMinimum(1 / (dt * total_samples))

    def endTimeChanged(self):
        print('End Time Changed')
        self.block_data_range_signals(True)
        samples = np.round(self.endTimeDoubleSpinBox.value() * self.sample_rate)
        self.endTimeDoubleSpinBox.setValue(samples / self.sample_rate)
        if self.startTimeDoubleSpinBox.value() > self.endTimeDoubleSpinBox.value():
            self.endTimeDoubleSpinBox.setValue(self.startTimeDoubleSpinBox.value())
        self.time_selector_references.setRegion((self.startTimeDoubleSpinBox.value(),
                                                 self.endTimeDoubleSpinBox.value()))
        self.time_selector_responses.setRegion((self.startTimeDoubleSpinBox.value(),
                                                self.endTimeDoubleSpinBox.value()))
        self.block_data_range_signals(False)
        # Now adjust averaging information
        if self.typeComboBox.currentIndex() == 0:
            self.overlapChanged()
        else:
            self.compute_triggers()
        # Adjust limits on the frame size
        dt = 1 / self.sample_rate
        min_index, max_index = self.get_abscissa_index_range()
        total_samples = max_index - min_index + 1
        self.frameSizeSpinBox.setMaximum(total_samples)
        self.frameTimeDoubleSpinBox.setMaximum(dt * total_samples)
        self.frequencyLinesSpinBox.setMaximum(total_samples // 2 + 1)
        self.frequencySpacingDoubleSpinBox.setMinimum(1 / (dt * total_samples))

    def updateTimeFromReference(self):
        print('Time Changed from References')
        self.block_data_range_signals(True)
        min_time, max_time = self.time_selector_references.getRegion()
        samples_min = np.round(min_time * self.sample_rate)
        samples_max = np.round(max_time * self.sample_rate)
        min_time = samples_min / self.sample_rate
        max_time = samples_max / self.sample_rate
        self.startTimeDoubleSpinBox.setValue(min_time)
        self.endTimeDoubleSpinBox.setValue(max_time)
        self.time_selector_references.setRegion((self.startTimeDoubleSpinBox.value(),
                                                 self.endTimeDoubleSpinBox.value()))
        self.time_selector_responses.setRegion((self.startTimeDoubleSpinBox.value(),
                                                self.endTimeDoubleSpinBox.value()))
        self.block_data_range_signals(False)
        # Now adjust averaging information
        if self.typeComboBox.currentIndex() == 0:
            self.overlapChanged()
        else:
            self.compute_triggers()
        # Adjust limits on the frame size
        dt = 1 / self.sample_rate
        min_index, max_index = self.get_abscissa_index_range()
        total_samples = max_index - min_index + 1
        self.frameSizeSpinBox.setMaximum(total_samples)
        self.frameTimeDoubleSpinBox.setMaximum(dt * total_samples)
        self.frequencyLinesSpinBox.setMaximum(total_samples // 2 + 1)
        self.frequencySpacingDoubleSpinBox.setMinimum(1 / (dt * total_samples))

    def updateTimeFromResponse(self):
        print('Time Changed from Responses')
        self.block_data_range_signals(True)
        min_time, max_time = self.time_selector_responses.getRegion()
        samples_min = np.round(min_time * self.sample_rate)
        samples_max = np.round(max_time * self.sample_rate)
        min_time = samples_min / self.sample_rate
        max_time = samples_max / self.sample_rate
        self.startTimeDoubleSpinBox.setValue(min_time)
        self.endTimeDoubleSpinBox.setValue(max_time)
        self.time_selector_references.setRegion((self.startTimeDoubleSpinBox.value(),
                                                 self.endTimeDoubleSpinBox.value()))
        self.time_selector_responses.setRegion((self.startTimeDoubleSpinBox.value(),
                                                self.endTimeDoubleSpinBox.value()))
        self.block_data_range_signals(False)
        # Now adjust averaging information
        if self.typeComboBox.currentIndex() == 0:
            self.overlapChanged()
        else:
            self.compute_triggers()
        # Adjust limits on the frame size
        dt = 1 / self.sample_rate
        min_index, max_index = self.get_abscissa_index_range()
        total_samples = max_index - min_index + 1
        self.frameSizeSpinBox.setMaximum(total_samples)
        self.frameTimeDoubleSpinBox.setMaximum(dt * total_samples)
        self.frequencyLinesSpinBox.setMaximum(total_samples // 2 + 1)
        self.frequencySpacingDoubleSpinBox.setMinimum(1 / (dt * total_samples))

    def frameSizeChanged(self):
        print('Frame Size Changed')
        self.block_sampling_signals(True)
        frame_size = self.frameSizeSpinBox.value()
        # Make sure it stays as a factor of 2
        if frame_size % 2 == 1:
            self.frameSizeSpinBox.setValue(frame_size + 1)
            frame_size += 1
        self.frequencyLinesSpinBox.setValue(frame_size // 2 + 1)
        self.frameTimeDoubleSpinBox.setValue(frame_size / self.sample_rate)
        self.frequencySpacingDoubleSpinBox.setValue(self.sample_rate / frame_size)
        self.block_sampling_signals(False)
        if self.typeComboBox.currentIndex() == 0:
            self.overlapChanged()
        else:
            self.compute_triggers()

    def frequencyLinesChanged(self):
        print('Frequency Lines Changed')
        self.block_sampling_signals(True)
        frame_size = (self.frequencyLinesSpinBox - 1) * 2
        self.frameSizeSpinBox.setValue(frame_size)
        self.frameTimeDoubleSpinBox.setValue(frame_size / self.sample_rate)
        self.frequencySpacingDoubleSpinBox.setValue(self.sample_rate / frame_size)
        self.block_sampling_signals(False)
        self.overlapChanged()

    def frameTimeChanged(self):
        print('Frame Time Changed')
        frame_size = int(np.round(self.frameTimeDoubleSpinBox.value() * self.sample_rate))
        self.frameSizeSpinBox.setValue(frame_size)

    def frequencySpacingChanged(self):
        print('Frequency Spacing Changed')
        frame_time = 1 / self.frequencySpacingDoubleSpinBox.value()
        self.frameTimeDoubleSpinBox.setValue(frame_time)

    def windowChanged(self):
        print('Window Changed')
        window_text = self.windowComboBox.currentText().lower()
        self.windowParameter1DoubleSpinBox.setVisible(False)
        self.windowParameter2DoubleSpinBox.setVisible(False)
        self.windowParameter3DoubleSpinBox.setVisible(False)
        self.windowParameter1Label.setVisible(False)
        self.windowParameter2Label.setVisible(False)
        self.windowParameter3Label.setVisible(False)
        for parameter, label, spinbox in zip(self.window_parameters[window_text],
                                             [self.windowParameter1Label,
                                             self.windowParameter2Label,
                                             self.windowParameter3Label],
                                             [self.windowParameter1DoubleSpinBox,
                                             self.windowParameter2DoubleSpinBox,
                                             self.windowParameter3DoubleSpinBox]):
            label.setText(parameter)
            label.setVisible(True)
            spinbox.setVisible(True)

    def typeChanged(self):
        print('Type Changed')
        try:
            trigger_run = self.typeComboBox.currentIndex() != 0
            self.framesSpinBox.setReadOnly(trigger_run)
            self.framesSpinBox.setButtonSymbols(
                self.framesSpinBox.NoButtons if trigger_run else self.framesSpinBox.UpDownArrows)
            for widget in [self.channelComboBox,
                           self.slopeComboBox,
                           self.levelDoubleSpinBox,
                           self.pretriggerDoubleSpinBox,
                           self.channelLabel,
                           self.slopeLabel,
                           self.levelLabel,
                           self.pretriggerLabel,
                           self.hysteresisLevelDoubleSpinBox,
                           self.hysteresisLevelLabel
                           ]:
                widget.setVisible(trigger_run)
            for widget in [self.overlapDoubleSpinBox,
                           self.overlapSamplesSpinBox,
                           self.overlapLabel,
                           self.overlapSamplesLabel]:
                widget.setVisible(not trigger_run)
            if self.typeComboBox.currentIndex() == 0:
                self.overlapChanged()
            else:
                self.compute_triggers()
            self.referencesChanged()
            self.responsesChanged()
        except Exception as e:
            print(e)

    def referencesChanged(self):
        print('References Changed')
        indices = [(item.data(Qt.UserRole), item.text())
                   for item in self.referencesSelector.selectedItems()]
        try:
            xrange = self.referencesPlot.getViewBox().viewRange()[0]
        except AttributeError:
            xrange = None
        self.referencesPlot.clear()
        for j, (index, text) in enumerate(indices):
            data_entry = self.time_history_data[index]
            pen = pqtg.mkPen(color=[int(255 * v) for v in self.cm(j % self.cm_mod)])
            self.referencesPlot.plot(x=data_entry.abscissa,
                                     y=data_entry.ordinate, name=text, pen=pen)
        if not xrange is None:
            self.referencesPlot.setXRange(*xrange, padding=0.0)
        self.referencesPlot.addItem(self.time_selector_references)
        self.referencesPlot.addItem(self.trigger_level_selector_references)
        self.referencesPlot.addItem(self.hysteresis_level_selector_references)
        for roi in self.reference_rois:
            self.referencesPlot.addItem(roi, ignoreBounds=True)
        if self.typeComboBox.currentIndex() == 1 and self.channelComboBox.currentIndex() in [index[0] for index in indices]:
            for widget in [self.trigger_level_selector_references,
                           self.hysteresis_level_selector_references]:
                widget.setVisible(True)
        else:
            for widget in [self.trigger_level_selector_references,
                           self.hysteresis_level_selector_references]:
                widget.setVisible(False)

    def responsesChanged(self):
        print('Responses Changed')
        indices = [(item.data(Qt.UserRole), item.text())
                   for item in self.responsesSelector.selectedItems()]
        # print(indices)
        try:
            xrange = self.responsesPlot.getViewBox().viewRange()[0]
        except AttributeError:
            xrange = None
        self.responsesPlot.clear()
        for j, (index, text) in enumerate(indices):
            data_entry = self.time_history_data[index]
            # print(data_entry.abscissa)
            # print(data_entry.ordinate)
            pen = pqtg.mkPen(color=[int(255 * v) for v in self.cm(j % self.cm_mod)])
            # print(pen)
            self.responsesPlot.plot(x=data_entry.abscissa,
                                    y=data_entry.ordinate, name=text, pen=pen)
        if not xrange is None:
            self.responsesPlot.setXRange(*xrange, padding=0.0)
        self.responsesPlot.addItem(self.time_selector_responses)
        self.responsesPlot.addItem(self.trigger_level_selector_responses)
        self.responsesPlot.addItem(self.hysteresis_level_selector_responses)
        for roi in self.response_rois:
            self.responsesPlot.addItem(roi, ignoreBounds=True)
        if self.typeComboBox.currentIndex() == 1 and self.channelComboBox.currentIndex() in [index[0] for index in indices]:
            for widget in [self.trigger_level_selector_responses,
                           self.hysteresis_level_selector_responses]:
                widget.setVisible(True)
        else:
            for widget in [self.trigger_level_selector_responses,
                           self.hysteresis_level_selector_responses]:
                widget.setVisible(False)

    def sendToResponse(self):
        print('Sent to Response')
        try:
            self.responsesSelector.clearSelection()
            for item in self.referencesSelector.selectedItems():
                # Get the indices
                moved_item = self.referencesSelector.takeItem(self.referencesSelector.row(item))
                # Get the responses data
                indices = np.array([self.responsesSelector.item(index).data(Qt.UserRole)
                                   for index in range(self.responsesSelector.count())])
                # Find the row we want to put it in
                index_checks = indices > moved_item.data(Qt.UserRole)
                if len(index_checks) == 0:
                    row = 0
                elif index_checks.sum() == 0:
                    row = self.referencesSelector.count()
                else:
                    row = np.argmax(index_checks)
                self.responsesSelector.insertItem(row, moved_item)
                item.setSelected(True)
            self.referencesSpinBox.setValue(self.referencesSelector.count())
            self.responsesSpinBox.setValue(self.responsesSelector.count())
        except Exception as e:
            print(e)

    def sendToReference(self):
        print('Sent to Reference')
        try:
            self.referencesSelector.clearSelection()
            for item in self.responsesSelector.selectedItems():
                # Get the indices
                moved_item = self.responsesSelector.takeItem(self.responsesSelector.row(item))
                # Get the responses data
                indices = np.array([self.referencesSelector.item(index).data(Qt.UserRole)
                                   for index in range(self.referencesSelector.count())])
                # Find the row we want to put it in
                index_checks = indices > moved_item.data(Qt.UserRole)
                if len(index_checks) == 0:
                    row = 0
                elif index_checks.sum() == 0:
                    row = self.referencesSelector.count()
                else:
                    row = np.argmax(index_checks)
                self.referencesSelector.insertItem(row, moved_item)
                item.setSelected(True)
            self.referencesSpinBox.setValue(self.referencesSelector.count())
            self.responsesSpinBox.setValue(self.responsesSelector.count())
        except Exception as e:
            print(e)

    def updateTriggerFromReference(self):
        print('Trigger Level Changed from Reference')
        self.block_averaging_signals(True)
        value = self.trigger_level_selector_references.value()
        hysteresis_value = self.hysteresis_level_selector_references.value()
        print('Value: {:}'.format(value))
        slope_positive = self.slopeComboBox.currentIndex() == 0
        if slope_positive and value < hysteresis_value:
            value = hysteresis_value
        if (not slope_positive) and (value > hysteresis_value):
            value = hysteresis_value
        print('Value: {:}'.format(value))
        self.levelDoubleSpinBox.setValue(value)
        self.trigger_level_selector_responses.setValue(value)
        self.trigger_level_selector_references.setValue(value)
        self.block_averaging_signals(False)
        self.compute_triggers()

    def updateTriggerFromResponse(self):
        print('Trigger Level Changed from Response')
        self.block_averaging_signals(True)
        value = self.trigger_level_selector_responses.value()
        hysteresis_value = self.hysteresis_level_selector_responses.value()
        print('Value: {:}'.format(value))
        slope_positive = self.slopeComboBox.currentIndex() == 0
        if slope_positive and value < hysteresis_value:
            value = hysteresis_value
        if (not slope_positive) and (value > hysteresis_value):
            value = hysteresis_value
        print('Value: {:}'.format(value))
        self.levelDoubleSpinBox.setValue(value)
        self.trigger_level_selector_references.setValue(value)
        self.trigger_level_selector_responses.setValue(value)
        self.block_averaging_signals(False)
        self.compute_triggers()

    def updateHysteresisFromReference(self):
        print('Hysteresis Level Changed from Reference')
        self.block_averaging_signals(True)
        value = self.trigger_level_selector_references.value()
        hysteresis_value = self.hysteresis_level_selector_references.value()
        print('Hysteresis Value: {:}'.format(hysteresis_value))
        slope_positive = self.slopeComboBox.currentIndex() == 0
        if slope_positive and value < hysteresis_value:
            hysteresis_value = value
        if (not slope_positive) and (value > hysteresis_value):
            hysteresis_value = value
        print('Hysteresis Value: {:}'.format(hysteresis_value))
        self.hysteresisLevelDoubleSpinBox.setValue(hysteresis_value)
        self.hysteresis_level_selector_responses.setValue(hysteresis_value)
        self.hysteresis_level_selector_references.setValue(hysteresis_value)
        self.block_averaging_signals(False)
        self.compute_triggers()

    def updateHysteresisFromResponse(self):
        print('Hysteresis Level Changed from Response')
        self.block_averaging_signals(True)
        value = self.trigger_level_selector_responses.value()
        hysteresis_value = self.hysteresis_level_selector_responses.value()
        print('Hysteresis Value: {:}'.format(hysteresis_value))
        slope_positive = self.slopeComboBox.currentIndex() == 0
        if slope_positive and value < hysteresis_value:
            hysteresis_value = value
        if (not slope_positive) and (value > hysteresis_value):
            hysteresis_value = value
        print('Hysteresis Value: {:}'.format(hysteresis_value))
        self.hysteresisLevelDoubleSpinBox.setValue(hysteresis_value)
        self.hysteresis_level_selector_responses.setValue(hysteresis_value)
        self.hysteresis_level_selector_references.setValue(hysteresis_value)
        self.block_averaging_signals(False)
        self.compute_triggers()

    def levelChanged(self):
        print('Trigger Level Changed')
        self.block_averaging_signals(True)
        slope_positive = self.slopeComboBox.currentIndex() == 0
        if slope_positive and self.levelDoubleSpinBox.value() < self.hysteresisLevelDoubleSpinBox.value():
            self.levelDoubleSpinBox.setValue(self.hysteresisLevelDoubleSpinBox.value())
        if (not slope_positive) and (self.levelDoubleSpinBox.value() > self.hysteresisLevelDoubleSpinBox.value()):
            self.levelDoubleSpinBox.setValue(self.hysteresisLevelDoubleSpinBox.value())
        self.trigger_level_selector_references.setValue(self.levelDoubleSpinBox.value())
        self.trigger_level_selector_responses.setValue(self.levelDoubleSpinBox.value())
        self.block_averaging_signals(False)
        self.compute_triggers()

    def hysteresisChanged(self):
        print('Hysteresis Level Changed')
        self.block_averaging_signals(True)
        slope_positive = self.slopeComboBox.currentIndex() == 0
        if slope_positive and self.levelDoubleSpinBox.value() < self.hysteresisLevelDoubleSpinBox.value():
            self.hysteresisLevelDoubleSpinBox.setValue(self.levelDoubleSpinBox.value())
        if (not slope_positive) and (self.levelDoubleSpinBox.value() > self.hysteresisLevelDoubleSpinBox.value()):
            self.hysteresisLevelDoubleSpinBox.setValue(self.levelDoubleSpinBox.value())
        self.hysteresis_level_selector_references.setValue(
            self.hysteresisLevelDoubleSpinBox.value())
        self.hysteresis_level_selector_responses.setValue(self.hysteresisLevelDoubleSpinBox.value())
        self.block_averaging_signals(False)
        self.compute_triggers()

    def pretriggerChanged(self):
        print('Pretrigger Changed')
        self.compute_triggers()

    def compute_triggers(self):
        try:
            print('Computing Triggers')
            level = self.levelDoubleSpinBox.value()
            hysteresis = self.hysteresisLevelDoubleSpinBox.value()
            positive_slope = self.slopeComboBox.currentIndex() == 0
            channel_index = self.channelComboBox.currentIndex()
            pretrigger_samples = int(self.pretriggerDoubleSpinBox.value() /
                                     100 * self.frameSizeSpinBox.value())
            min_abscissa_index, max_abscissa_index = self.get_abscissa_index_range()
            frame_size = self.frameSizeSpinBox.value()
            data = self.time_history_data[channel_index].ordinate
            if positive_slope:
                potential_triggers = np.where((data[:-1] <= level) & (data[1:] > level))[0]
                trigger_resets = data[:-1] < hysteresis
            else:
                potential_triggers = np.where((data[:-1] >= level) & (data[1:] < level))[0]
                trigger_resets = data[:-1] > hysteresis
            triggers = []
            for trigger in potential_triggers:
                last_trigger = triggers[-1] if len(triggers) > 0 else 0
                if (np.any(trigger_resets[last_trigger:trigger])
                    and (trigger - pretrigger_samples >= min_abscissa_index)
                    and (trigger - pretrigger_samples + frame_size < max_abscissa_index)
                        and (len(triggers) == 0 or (trigger - pretrigger_samples - triggers[-1] >= frame_size))):
                    triggers.append(trigger - pretrigger_samples)
            self.block_averaging_signals(True)
            self.framesSpinBox.setValue(len(triggers))
            self.block_averaging_signals(False)
            self.ignore_frames = []
            self.frame_start_indices = triggers
            self.create_rois()
        except Exception as e:
            print(e)

    def compute(self):
        # Split up the measurement into frames

        try:
            frequency_spacing = self.frequencySpacingDoubleSpinBox.value()
            frame_size = self.frameSizeSpinBox.value()
            frame_indices = np.array([start_index for frame_number, start_index in enumerate(self.frame_start_indices)
                                      if not frame_number in self.ignore_frames])[:, np.newaxis] + np.arange(frame_size)
            if frame_indices.shape[0] == 0:
                QMessageBox.critical(self, 'Invalid Frames',
                                     'At least one measurement frame must be selected')
                return
            reference_indices = np.array([self.referencesSelector.item(i).data(
                Qt.UserRole) for i in range(self.referencesSelector.count())])
            if reference_indices.size == 0:
                QMessageBox.critical(self, 'Invalid References',
                                     'At least one reference must be selected')
                return
            response_indices = np.array([self.responsesSelector.item(i).data(
                Qt.UserRole) for i in range(self.responsesSelector.count())])
            if response_indices.size == 0:
                QMessageBox.critical(self, 'Invalid Responses',
                                     'At least one response must be selected')
                return
            time_by_frames = self.time_history_data.ordinate[..., frame_indices]
            reference_data = time_by_frames[reference_indices].copy()
            response_data = time_by_frames[response_indices].copy()
            # print(reference_data.shape)
            # print(response_data.shape)
            # Get the window function
            window_function_string = self.windowComboBox.currentText().lower()
            num_parameters = len(self.window_parameters[window_function_string])
            window_parameters = [widget.value() for widget in [self.windowParameter1DoubleSpinBox,
                                                               self.windowParameter2DoubleSpinBox,
                                                               self.windowParameter3DoubleSpinBox][:num_parameters]]
            if window_function_string == 'rectangle':
                window_function_string = 'boxcar'
            if window_function_string == 'exponential+force':
                window_function_string = 'exponential'
                response_window = get_window(
                    tuple([window_function_string] + window_parameters[1:]), frame_size, fftbins=True)
                non_pulse_samples = np.arange(frame_size) / self.sample_rate > window_parameters[0]
                reference_window = response_window.copy()
                reference_window[non_pulse_samples] = 0
                # Need to adjust the reference signals so we don't create jumps when we apply the force window
                dc_offsets = np.mean(reference_data[..., non_pulse_samples], axis=-1, keepdims=True)
                reference_data -= dc_offsets
            else:
                response_window = get_window(
                    tuple([window_function_string] + window_parameters), frame_size, fftbins=True)
                reference_window = response_window
            window_correction = 1 / np.mean(response_window**2)
            response_data *= response_window
            reference_data *= reference_window
            reference_coordinates = self.time_history_data[reference_indices].coordinate.flatten()
            response_coordinates = self.time_history_data[response_indices].coordinate.flatten()
            # Create time history array
            self.windowed_time_data = data_array(FunctionTypes.TIME_RESPONSE,
                                                 np.arange(frame_size) / self.sample_rate,
                                                 np.concatenate(
                                                     (reference_data, response_data), axis=0),
                                                 self.time_history_data.coordinate[np.concatenate((reference_indices, response_indices)), np.newaxis])
            self.plotWindowedTimeHistoryButton.setEnabled(True)
            self.saveWindowedTimeHistoryButton.setEnabled(True)
            response_fft = np.fft.rfft(response_data, axis=-1)
            reference_fft = np.fft.rfft(reference_data, axis=-1)
            # print(reference_fft.shape)
            # print(response_fft.shape)
            freq = np.fft.rfftfreq(frame_size, 1 / self.sample_rate)
            # Compute FRFs
            if self.frfCheckBox.isChecked():
                # Check the type of FRF
                frf_coordinate = outer_product(response_coordinates,
                                               reference_coordinates)
                success = False
                if self.frfComboBox.currentIndex() == 0:  # H1
                    # We want to compute X*F^H = [X1;X2;X3][F1^H F2^H F3^H]
                    Gxf = np.sum(np.einsum('iaf,jaf->afij', response_fft,
                                 np.conj(reference_fft)), axis=0)
                    Gff = np.sum(np.einsum('iaf,jaf->afij', reference_fft,
                                 np.conj(reference_fft)), axis=0)
                    # Add small values to any matrices that are singular
                    singular_matrices = np.abs(np.linalg.det(Gff)) < 2 * np.finfo(Gff.dtype).eps
                    Gff[singular_matrices] += np.eye(Gff.shape[-1]) * np.finfo(Gff.dtype).eps
                    H = np.linalg.solve(Gff.transpose(0, 2, 1),
                                        Gxf.transpose(0, 2, 1)).transpose(0, 2, 1)
                    # Create TransferFunctionArray
                    self.frf_data = data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                                               freq, np.moveaxis(H, 0, -1), frf_coordinate)
                    success = True
                elif self.frfComboBox.currentIndex() == 1: # H2
                    if (response_fft.shape != reference_fft.shape):
                        QMessageBox.critical(self, 'Bad FRF Shape',
                                             'For H2, Number of inputs must equal number of outputs')
                        success = False
                    else:
                        Gxx = np.einsum('iaf,jaf->fij', response_fft, np.conj(response_fft))
                        Gfx = np.einsum('iaf,jaf->fij', reference_fft, np.conj(response_fft))
                        singular_matrices = np.abs(np.linalg.det(Gfx)) < 2 * np.finfo(Gfx.dtype).eps
                        Gfx[singular_matrices] += np.eye(Gfx.shape[-1]) * np.finfo(Gfx.dtype).eps
                        H = np.moveaxis(np.linalg.solve(np.moveaxis(Gfx, -2, -1), np.moveaxis(Gxx, -2, -1)), -2, -1)
                        # Create TransferFunctionArray
                        self.frf_data = data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                                                   freq, np.moveaxis(H, 0, -1), frf_coordinate)
                        success = True
                elif self.frfComboBox.currentIndex() == 2: # H3
                    if (response_fft.shape != reference_fft.shape):
                        QMessageBox.critical(self, 'Bad FRF Shape',
                                             'For H3, Number of inputs must equal number of outputs')
                        success = False
                    else:
                        Gxf = np.einsum('iaf,jaf->fij', response_fft, np.conj(reference_fft))
                        Gff = np.einsum('iaf,jaf->fij', reference_fft, np.conj(reference_fft))
                        # Add small values to any matrices that are singular
                        singular_matrices = np.abs(np.linalg.det(Gff)) < 2 * np.finfo(Gff.dtype).eps
                        Gff[singular_matrices] += np.eye(Gff.shape[-1]) * np.finfo(Gff.dtype).eps
                        Gxx = np.einsum('iaf,jaf->fij', response_fft, np.conj(response_fft))
                        Gfx = np.einsum('iaf,jaf->fij', reference_fft, np.conj(response_fft))
                        singular_matrices = np.abs(np.linalg.det(Gfx)) < 2 * np.finfo(Gfx.dtype).eps
                        Gfx[singular_matrices] += np.eye(Gfx.shape[-1]) * np.finfo(Gfx.dtype).eps
                        H = (np.moveaxis(np.linalg.solve(np.moveaxis(Gfx, -2, -1), np.moveaxis(Gxx, -2, -1)), -2, -1) +
                             np.moveaxis(np.linalg.solve(np.moveaxis(Gff, -2, -1), np.moveaxis(Gxf, -2, -1)), -2, -1)) / 2
                        self.frf_data = data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                                                   freq, np.moveaxis(H, 0, -1), frf_coordinate)
                        success = True
                elif self.frfComboBox.currentIndex() == 3: # Hv
                    Gxx = np.einsum('...iaf,...iaf->...if', response_fft,
                                    np.conj(response_fft))[..., np.newaxis, np.newaxis]
                    Gxf = np.einsum('...iaf,...jaf->...ifj', response_fft,
                                    np.conj(reference_fft))[..., np.newaxis, :]
                    Gff = np.einsum('...iaf,...jaf->...fij', reference_fft,
                                    np.conj(reference_fft))[..., np.newaxis, :, :, :]
                    # print(Gxx.shape)
                    # print(Gxf.shape)
                    # print(Gff.shape)
                    # Broadcast over all responses
                    Gff = np.broadcast_to(Gff,Gxx.shape[:-2]+Gff.shape[-2:])
                    Gffx = np.block([[Gff, np.conj(np.moveaxis(Gxf, -2, -1))],
                                     [Gxf, Gxx]])
                    # Compute eigenvalues
                    lam, evect = np.linalg.eigh(np.moveaxis(Gffx, -2, -1))
                    # Get the evect corresponding to the minimum eigenvalue
                    evect = evect[..., 0]  # Assumes evals are sorted ascending
                    H = np.moveaxis(-evect[..., :-1] / evect[..., -1:],  # Scale so last value is -1
                                    -3, -2)
                    self.frf_data = data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION,
                                               freq, np.moveaxis(H, 0, -1), frf_coordinate)
                    success = True
                else:
                    QMessageBox.critical(self, 'FRF Technique Not Implemented',
                                         'FRF Technique {:} has not been implemented yet, sorry!'.format(self.frfComboBox.currentText()))
                    success = False
                if success:
                    self.plotFRFButton.setEnabled(True)
                    self.saveFRFButton.setEnabled(True)
                    self.menuVisualize_with_DeflectionShapePlotter.setEnabled(True)
                    self.actionSend_to_SMAC.setEnabled(True)
                    self.actionSend_to_PolyPy.setEnabled(True)
                    # Add actions to a menu
                    self.menuVisualize_with_DeflectionShapePlotter.clear()
                    self.deflection_actions.clear()
                    for reference_coord in reference_coordinates:
                        action = QAction('Reference {:}'.format(str(reference_coord)))
                        action.triggered.connect(self.plotDeflection)
                        self.menuVisualize_with_DeflectionShapePlotter.addAction(action)
                        self.deflection_actions.append(action)
            if self.autospectraCheckBox.isChecked():
                # Compute Autospectra
                full_fft = np.concatenate((reference_fft, response_fft), axis=0)
                full_coords = np.concatenate((reference_coordinates, response_coordinates))
                spectral_matrix = np.einsum('iaf,iaf->fi', full_fft,
                                            np.conj(full_fft)) / full_fft.shape[1]
                spectral_matrix *= (frequency_spacing * window_correction /
                                    self.sample_rate**2)
                spectral_matrix[1:-1] *= 2
                self.autospectra_data = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, freq,
                                                   np.moveaxis(spectral_matrix, 0, -1),
                                                   full_coords[:, np.newaxis])
                self.plotAutospectraButton.setEnabled(True)
                self.saveAutospectraButton.setEnabled(True)
            if self.crossspectraCheckBox.isChecked():
                # Compute cross-spectra
                full_fft = np.concatenate((reference_fft, response_fft), axis=0)
                full_coords = np.concatenate((reference_coordinates, response_coordinates))
                spectral_matrix = np.einsum('iaf,jaf->fij', full_fft,
                                            np.conj(full_fft)) / full_fft.shape[1]
                spectral_matrix *= (frequency_spacing * window_correction /
                                    self.sample_rate**2)
                spectral_matrix[1:-1] *= 2
                self.crossspectra_data = data_array(FunctionTypes.POWER_SPECTRAL_DENSITY, freq,
                                                    np.moveaxis(spectral_matrix, 0, -1),
                                                    outer_product(full_coords, full_coords))
                self.plotCrossspectraButton.setEnabled(True)
                self.saveCrossspectraButton.setEnabled(True)
            if self.coherenceCheckBox.isChecked():
                if reference_fft.shape[0] == 1:
                    # Ordinary Coherence
                    Gxf = np.einsum('iaf,jaf->fij', response_fft,
                                    np.conj(reference_fft)) / frame_indices.shape[0]
                    Gxx = np.einsum('iaf,iaf->fi', response_fft,
                                    np.conj(response_fft)) / frame_indices.shape[0]
                    Gff = np.einsum('iaf,iaf->fi', reference_fft,
                                    np.conj(reference_fft)) / frame_indices.shape[0]
                    coh = np.abs(Gxf)**2 / (Gxx[:, :, np.newaxis] * Gff[:, np.newaxis, :])
                    self.coherence_data = data_array(FunctionTypes.COHERENCE, freq,
                                                     np.moveaxis(coh, 0, -1),
                                                     outer_product(response_coordinates,
                                                                   reference_coordinates))
                else:
                    # Multiple Coherence
                    Gxf = np.einsum('iaf,jaf->fij', response_fft,
                                    np.conj(reference_fft)) / frame_indices.shape[0]
                    Gxx = np.einsum('iaf,iaf->fi', response_fft,
                                    np.conj(response_fft)) / frame_indices.shape[0]
                    Gff = np.einsum('iaf,jaf->fij', reference_fft,
                                    np.conj(reference_fft)) / frame_indices.shape[0]
                    Mcoh = (np.einsum('fij,fjk,fik->fi', Gxf,
                            np.linalg.inv(Gff), Gxf.conj()) / Gxx).real
                    self.coherence_data = data_array(FunctionTypes.MULTIPLE_COHERENCE, freq,
                                                     np.moveaxis(Mcoh, 0, -1), response_coordinates[:, np.newaxis])
                self.plotCoherenceButton.setEnabled(True)
                self.saveCoherenceButton.setEnabled(True)
        except Exception:
            print(traceback.format_exc())

    def showWindowedTimeHistory(self):
        self.plots['windowed'] = GUIPlot(self.windowed_time_data)

    def saveWindowedTimeHistory(self):
        filename, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Select File to Save Windowed Time Data', filter='Numpy File (*.npz)')
        if filename == '':
            return
        self.windowed_time_data.save(filename)

    def showFRF(self):
        self.plots['frf'] = GUIPlot(self.frf_data)

    def saveFRF(self):
        filename, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Select File to Save FRF Data', filter='Numpy File (*.npz)')
        if filename == '':
            return
        self.frf_data.save(filename)

    def showCoherence(self):
        self.plots['coherence'] = GUIPlot(self.coherence_data)

    def saveCoherence(self):
        filename, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Select File to Save Coherence Data', filter='Numpy File (*.npz)')
        if filename == '':
            return
        self.coherence_data.save(filename)

    def showAutospectra(self):
        self.plots['autospectra'] = GUIPlot(self.autospectra_data)

    def saveAutospectra(self):
        filename, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Select File to Save Autospectra Data', filter='Numpy File (*.npz)')
        if filename == '':
            return
        self.autospectra_data.save(filename)

    def showCrossspectra(self):
        self.plots['crossspectra'] = GUIPlot(self.crossspectra_data)

    def saveCrossspectra(self):
        filename, file_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Select File to Save Crosspectra Data', filter='Numpy File (*.npz)')
        if filename == '':
            return
        self.crossspectra_data.save(filename)

    def analyzeSMAC(self):
        self.plots['smac'] = SMAC_GUI(self.frf_data)
        self.plots['smac'].geometry = self.geometry

    def analyzePolyPy(self):
        self.plots['polypy'] = PolyPy_GUI(self.frf_data)
        self.plots['polypy'].geometry = self.geometry

    def loadGeometry(self):
        filename, file_filter = QFileDialog.getOpenFileName(
            self, 'Select Geometry File', filter='Numpyz (*.npz);;Universal File Format (*.uff *.unv)')
        if filename == '':
            return
        self.geometry = Geometry.load(filename)

    def plotTransient(self):
        response_indices = np.array([self.responsesSelector.item(i).data(
            Qt.UserRole) for i in range(self.responsesSelector.count())])
        if response_indices.size == 0:
            QMessageBox.critical(self, 'Invalid Responses',
                                 'At least one response must be selected')
            return
        if self.geometry is None:
            self.loadGeometry()
            if self.geometry is None:
                return
        self.plots['transient'] = self.geometry.plot_transient(
            self.time_history_data[response_indices].extract_elements(slice(*self.get_abscissa_index_range())))

    def plotDeflection(self):
        try:
            print('Plotting deflection:')
            print('Action: {:}'.format(self.sender()))
            for index, action in enumerate(self.menuVisualize_with_DeflectionShapePlotter.actions()):
                if self.sender() is action:
                    print('Index {:}'.format(index))
                    break
            if self.geometry is None:
                self.loadGeometry()
                if self.geometry is None:
                    return
            self.plots['deflection_shape'] = self.geometry.plot_deflection_shape(
                self.frf_data[:, index])
        except Exception:
            print(traceback.format_exc())

    def loadData(self):
        try:
            filename, file_filter = QFileDialog.getOpenFileName(
                self, 'Select Time History File', filter='Numpy (*.npz);;Rattlesnake (*.nc4)')
            if filename == '':
                return
            if file_filter == 'Numpy (*.npz)':
                self.time_history_data = TimeHistoryArray.load(filename)
                self.initialize_ui()
            elif file_filter == 'Rattlesnake (*.nc4)':
                self.time_history_data, channel_table = read_rattlesnake_output(filename)
                self.initialize_ui()
                # TODO Automatically select references and responses
            else:
                QMessageBox.critical(self, 'Invalid Data File Type',
                                     'File must be a Rattlesnake or SDynPy Time History File')
        except Exception:
            print(traceback.format_exc())
