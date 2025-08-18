# -*- coding: utf-8 -*-
"""
Class defining the typical data for a modal test used to automatically create
reports and plots

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
import tempfile
import numpy as np
import io
from ..core.sdynpy_shape import ShapeArray, mac, matrix_plot, rigid_body_check,load as shape_load
from ..core.sdynpy_data import (TransferFunctionArray, CoherenceArray,
                                MultipleCoherenceArray, PowerSpectralDensityArray,
                                join as data_join)
from ..core.sdynpy_geometry import (Geometry,GeometryPlotter,ShapePlotter)
from ..core.sdynpy_coordinate import CoordinateArray, coordinate_array as sd_coordinate_array
from .sdynpy_signal_processing_gui import SignalProcessingGUI
from .sdynpy_polypy import PolyPy_GUI
from .sdynpy_smac import SMAC_GUI
from ..fileio.sdynpy_rattlesnake import read_modal_data
from ..doc.sdynpy_latex import (
    figure as latex_figure, table as latex_table,
    create_data_quality_summary, create_geometry_overview,
    create_mode_fitting_summary, create_mode_shape_figures, create_rigid_body_analysis)
from qtpy.QtCore import Qt
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import netCDF4 as nc4

def read_modal_fit_data(modal_fit_data):
    """
    Reads Modal Fit Data from PolyPy_GUI which contains modes and FRF data

    Parameters
    ----------
    modal_fit_data : str or NpzFile
        Filename to or data loaded from a Modal Fit Data .npz file that is
        saved from SDynPy's mode fitters.

    Returns
    -------
    shapes : ShapeArray
        The modes fit to the structure
    experimental_frfs : TransferFunctionArray
        The FRFs to which the modes were fit
    resynthesized_frfs : TransferFunctionArray
        FRFs resynthesized from the fit modes
    residual_frfs : TransferFunctionArray
        FRF contribution from the residual terms in the modal fit.

    """
    if isinstance(modal_fit_data,str):
        modal_fit_data = np.load(modal_fit_data)
    shapes = modal_fit_data['shapes'].view(ShapeArray)
    experimental_frfs = modal_fit_data['frfs'].view(TransferFunctionArray)
    resynthesized_frfs = modal_fit_data['frfs_resynth'].view(TransferFunctionArray)
    residual_frfs = modal_fit_data['frfs_residual'].view(TransferFunctionArray)
    return shapes, experimental_frfs, resynthesized_frfs, residual_frfs

class ModalTest:
    
    def __init__(self,
                 geometry = None,
                 time_histories = None,
                 autopower_spectra = None,
                 frfs = None,
                 coherence = None,
                 fit_modes = None,
                 resynthesized_frfs = None,
                 response_unit = None,
                 reference_unit = None,
                 rigid_body_shapes = None,
                 channel_table = None
                 ):
        self.response_unit = response_unit
        self.reference_unit = reference_unit
        self.geometry = geometry
        self.time_histories = time_histories
        self.autopower_spectra = autopower_spectra
        self.frfs = frfs
        self.coherence = coherence
        self.fit_modes = fit_modes
        self.resynthesized_frfs = resynthesized_frfs
        self.rigid_body_shapes = rigid_body_shapes
        self.channel_table = channel_table
        # Handles for GUIs that exist
        self.spgui = None
        self.ppgui = None
        self.smacgui = None
        self.modeshape_plotter = None
        self.deflectionshape_plotter = None
        # Quantities of interest for computing spectral quantities
        self.reference_indices = None
        self.autopower_spectra_reference_indices = None
        if time_histories is not None:
            self.sample_rate = 1/time_histories.abscissa_spacing
        else:
            self.sample_rate = None
        self.num_samples_per_frame = None
        self.num_averages = None
        self.start_time = None
        self.end_time = None
        self.trigger = None
        self.trigger_channel_index = None
        self.trigger_slope = None
        self.trigger_level = None
        self.pretrigger = None
        self.overlap = None
        self.window = None
        self.frf_estimator = None
        # Quantities of interest from Curve Fitters
        self.fit_modes_information = None
        # Excitation Information
        self.excitation_information = None
        # Figures to save to documentation
        self.documentation_figures = {}
        
    @classmethod
    def from_rattlesnake_modal_data(cls, input_file, geometry = None,
                                    fit_modes = None, resynthesized_frfs = None,
                                    rigid_body_shapes = None):
        if isinstance(input_file,str):
            input_file = nc4.Dataset(input_file)

        environment = input_file.groups[input_file['environment_names'][0]]
            
        time_histories, frfs, coherence, channel_table = read_modal_data(input_file)
        time_histories = data_join(time_histories)

        rename_dict = {key:key.replace('_',' ').title() for key in channel_table.columns}
        channel_table = channel_table.rename(columns=rename_dict)

        out = cls(geometry, time_histories, frfs = frfs, coherence = coherence,
                  fit_modes = fit_modes, resynthesized_frfs = resynthesized_frfs,
                  rigid_body_shapes = rigid_body_shapes, channel_table = channel_table)

        # Get spectral processing parameters
        out.define_spectral_processing_parameters(
            reference_indices = np.array(environment['reference_channel_indices'][:]),
            num_samples_per_frame = environment.samples_per_frame,
            num_averages = environment.num_averages,
            start_time = None,
            end_time = None,
            trigger = environment.trigger_type,
            trigger_channel_index = environment.trigger_channel,
            trigger_slope = 'Positive' if environment.trigger_slope_positive else 'Negative',
            trigger_level = environment.trigger_level,
            pretrigger = environment.pretrigger,
            overlap = environment.overlap,
            window = environment.frf_window,
            frf_estimator = environment.frf_technique,
            sample_rate = input_file.sample_rate)

        out.set_autopower_spectra(out.time_histories.cpsd(
            out.num_samples_per_frame, overlap = 0.0, window = 'boxcar' if out.window == 'rectangle' else out.window,
            averages_to_keep = out.num_averages, only_asds = True))

        reference_units = channel_table['Unit'][out.reference_indices].to_numpy()
        if np.all(reference_units == reference_units[0]):
            reference_units = reference_units[0]

        response_units = channel_table['Unit'][environment['response_channel_indices'][...]].to_numpy()
        if np.all(response_units == response_units[0]):
            response_units = response_units[0]
            
        out.set_units(response_units, reference_units)

        if environment.signal_generator_type == 'burst':
            out.excitation_information = {'text':
            f'Excitation for this test used a burst random signal from {environment.signal_generator_min_frequency} '
            f'to {environment.signal_generator_max_frequency:} Hz.  The signal level was {environment.signal_generator_level:} '
            f'V RMS and was on for {environment.signal_generator_on_fraction*100:}\\% of the '
            'measurement frame.'}
        elif environment.signal_generator_type == 'random':
            out.excitation_information = {'text':
            f'Excitation for this test used a random signal from {environment.signal_generator_min_frequency} '
            f'to {environment.signal_generator_max_frequency:} Hz.  The signal level was {environment.signal_generator_level:} '
            f'V RMS.'}
        elif environment.signal_generator_type == 'pseudorandom':
            out.excitation_information = {'text':
            f'Excitation for this test used a pseudorandom signal from {environment.signal_generator_min_frequency} '
            f'to {environment.signal_generator_max_frequency:} Hz.  The signal level was {environment.signal_generator_level:} '
            f'V RMS.'}
        elif environment.signal_generator_type == 'chirp':
            out.excitation_information = {'text':
            f'Excitation for this test used a chirp signal from {environment.signal_generator_min_frequency} '
            f'to {environment.signal_generator_max_frequency:} Hz.  The signal level was {environment.signal_generator_level:} '
            f'V peak amplitude.'}
        elif environment.signal_generator_type == 'sine':
            out.excitation_information = {'text':
            f'Excitation for this test used a sine signal at {environment.signal_generator_min_frequency}'
            f' Hz.  The signal level was {environment.signal_generator_level:} V peak amplitude.'}
        elif environment.signal_generator_type == 'square':
            out.excitation_information = {'text':
            f'Excitation for this test used a square wave signal at {environment.signal_generator_min_frequency}'
            f' Hz.  The signal level was {environment.signal_generator_level:} V peak amplitude.'}

        return out
    
    def set_rigid_body_shapes(self,rigid_body_shapes):
        self.rigid_body_shapes = rigid_body_shapes
    
    def set_units(self,response_unit, reference_unit):
        self.response_unit = response_unit
        self.reference_unit = reference_unit
        
    def set_geometry(self, geometry):
        self.geometry = geometry
    
    def set_time_histories(self, time_histories):
        self.time_histories = time_histories
        self.sample_rate = 1/time_histories.abscissa
    
    def set_autopower_spectra(self, autopower_spectra):
        self.autopower_spectra = autopower_spectra
    
    def set_frfs(self, frfs):
        self.frfs = frfs
    
    def set_coherence(self, coherence):
        self.coherence = coherence
    
    def set_fit_modes(self, fit_modes):
        self.fit_modes = fit_modes
        
    def set_resynthesized_frfs(self, resynthesized_frfs):
        self.resynthesized_frfs = resynthesized_frfs
    
    def set_channel_table(self,channel_table):
        self.channel_table = channel_table
    
    def compute_spectral_quantities_SignalProcessingGUI(self):
        if self.time_histories is None:
            raise ValueError('Time Histories must be defined in order to compute spectral quantities')
        self.spgui = SignalProcessingGUI(self.time_histories)
        if self.geometry is not None:
            self.spgui.geometry = self.geometry
    
    @property
    def response_indices(self):
        response_indices = np.arange(self.time_histories.size)[
            ~np.in1d(np.arange(self.time_histories.size),self.reference_indices)]
        return response_indices
    
    def retrieve_spectral_quantities_SignalProcessingGUI(self):
        self.reference_indices = np.array([self.spgui.referencesSelector.item(i).data(
            Qt.UserRole) for i in range(self.spgui.referencesSelector.count())])
        self.num_samples_per_frame = self.spgui.frameSizeSpinBox.value()
        self.num_averages = self.spgui.framesSpinBox.value()
        self.start_time = self.spgui.startTimeDoubleSpinBox.value()
        self.end_time = self.spgui.endTimeDoubleSpinBox.value()
        self.trigger = self.spgui.typeComboBox.currentIndex() == 1
        self.trigger_channel_index = self.spgui.channelComboBox.currentIndex()
        self.trigger_slope = self.spgui.slopeComboBox.currentIndex() == 0
        self.trigger_level = self.spgui.levelDoubleSpinBox.value()
        self.pretrigger = self.spgui.pretriggerDoubleSpinBox.value()
        self.overlap = self.spgui.overlapDoubleSpinBox.value()/100
        self.window = self.spgui.windowComboBox.currentText().lower()
        self.frf_estimator = self.spgui.frfComboBox.currentText().lower()
        self.spgui.frfCheckBox.setChecked(True)
        self.spgui.autospectraCheckBox.setChecked(True)
        self.spgui.coherenceCheckBox.setChecked(True)
        self.spgui.compute()
        self.autopower_spectra = self.spgui.autospectra_data
        self.autopower_spectra_reference_indices = np.arange(len(self.reference_indices))
        self.frfs = self.spgui.frf_data
        self.coherence = self.spgui.coherence_data
        
    def compute_spectral_quantities(
            self, reference_indices, start_time, end_time, num_samples_per_frame,
            overlap, window, frf_estimator):
        if self.time_histories is None:
            raise ValueError('Time Histories must be defined in order to compute spectral quantities')
        self.reference_indices = reference_indices
        self.autopower_spectra_reference_indices = reference_indices
        self.num_samples_per_frame = num_samples_per_frame
        self.start_time = start_time
        self.end_time = end_time
        self.trigger = False
        self.trigger_channel_index = None
        self.trigger_slope = None
        self.trigger_level = None
        self.pretrigger = None
        self.overlap = overlap
        self.window = window
        self.frf_estimator = frf_estimator
        
        # Separate into references and responses
        time_data = self.time_histories.extract_elements_by_abscissa(start_time, end_time)
        references = time_data[self.reference_indices]
        
        responses = time_data[self.response_indices]
        
        self.num_averages = int(time_data.num_elements - (1-overlap)*num_samples_per_frame)//num_samples_per_frame + 1
        
        # Compute FRFs, Coherence, and Autospectra
        self.frfs = TransferFunctionArray.from_time_data(
            references, responses, num_samples_per_frame, overlap, frf_estimator,
            window)
        
        self.autopower_spectra = PowerSpectralDensityArray.from_time_data(
            time_data, num_samples_per_frame, overlap, window, only_asds = True)
        
        if len(self.reference_indices) > 1:
            self.coherence = MultipleCoherenceArray.from_time_data(
                responses, num_samples_per_frame, overlap, window, 
                references)
        else:
            self.coherence = CoherenceArray.from_time_data(
                responses, num_samples_per_frame, overlap, window, references)
        
    def define_spectral_processing_parameters(
            self, reference_indices, num_samples_per_frame, num_averages, 
            start_time, end_time, trigger, trigger_channel_index,
            trigger_slope, trigger_level, pretrigger, overlap, window,
            frf_estimator, sample_rate):
        self.reference_indices = reference_indices
        self.autopower_spectra_reference_indices = reference_indices
        self.num_samples_per_frame = num_samples_per_frame
        self.num_averages = num_averages
        self.start_time = start_time
        self.end_time = end_time
        self.trigger = trigger
        self.trigger_channel_index = trigger_channel_index
        self.trigger_slope = trigger_slope
        self.trigger_level = trigger_level
        self.pretrigger = pretrigger
        self.overlap = overlap
        self.window = window
        self.frf_estimator = frf_estimator
        self.sample_rate = sample_rate
    
    def fit_modes_PolyPy(self):
        if self.frfs is None:
            raise ValueError('FRFs must be defined in order to fit modes')
        self.ppgui = PolyPy_GUI(self.frfs)
    
    def retrieve_modes_PolyPy(self):
        self.ppgui.compute_shapes()
        self.fit_modes = self.ppgui.shapes
        self.resynthesized_frfs = self.resynthesized_frfs
        
        self.fit_modes_information = {'text':[
            'Modes were fit to the data using the PolyPy curve fitter implemented in SDynPy in {:} bands.'.format(len(self.ppgui.stability_diagrams))]}
        figure_index = 0
        # Now go through and get polynomial data from each frequency range
        for stability_diagram in self.ppgui.stability_diagrams:
            # First get data about the stabilization diagram
            min_frequency = stability_diagram.polypy.min_frequency
            max_frequency = stability_diagram.polypy.max_frequency
            min_order = np.min(stability_diagram.polypy.polynomial_orders)
            max_order = np.max(stability_diagram.polypy.polynomial_orders)
            num_selected_poles = len(stability_diagram.selected_poles)
            self.fit_modes_information['text'].append(
                ('The frequency band from {:0.2f} to {:0.2f} was analyzed with polynomials from order {:} to {:}.  '+
                 '{:} poles were selected from this band.  The stabilization diagram is shown in Figure {{figure{:}ref:}}.').format(
                    min_frequency, max_frequency,min_order,max_order,num_selected_poles,figure_index))
            # Go through and save out a figure for each stabilization diagram
            tempdir = tempfile.mkdtemp()
            filename = os.path.join(tempdir,'stability_diagram.png')
            # Turn off last highlighted closest mode
            if stability_diagram.previous_closest_marker_index is not None:
                if stability_diagram.previous_closest_marker_index in stability_diagram.selected_poles:
                    order_index, pole_index = stability_diagram.pole_indices[stability_diagram.previous_closest_marker_index]
                    pole = stability_diagram.polypy.pole_list[order_index][pole_index]
                    if pole['part_stable']:
                        brush = (0, 128, 0)
                    elif pole['damp_stable'] or pole['freq_stable']:
                        brush = 'b'
                    else:
                        brush = 'r'
                    stability_diagram.pole_markers[stability_diagram.previous_closest_marker_index].setBrush(brush)
                else:
                    stability_diagram.pole_markers[stability_diagram.previous_closest_marker_index].setBrush((0, 0, 0, 0))
            stability_diagram.pole_plot.writeImage(filename)
            with open(filename,'rb') as f:
                image_bytes = f.read()
                self.fit_modes_information['figure'+str(figure_index)] = image_bytes
                self.fit_modes_information['figure'+str(figure_index)+'caption'] = (
                    'Stabilization Diagram from {:0.2f} to {:0.2f} Hz.  '.format(min_frequency,max_frequency) + 
                    'Red Xs represent unstable poles.  ' + 
                    'Blue Triangles represent that the frequency has stablized.  ' + 
                    'Blue Squares represent that the frequency and damping have stablized.  ' + 
                    'Green circles represent that the frequency, damping, and shape have stablized.  ' + 
                    'Solid markers are poles that were selected in the final mode set.')
            figure_index += 1
            os.remove(filename)
        os.removedirs(tempdir)
        self.fit_modes_information['text'].append(
            ('Complex Modes' if self.ppgui.complex_modes_checkbox.isChecked() else 'Normal Modes') + ' were fit to the data.  ' + 
            ('Residuals' if self.ppgui.use_residuals_checkbox.isChecked() else 'No residuals') + ' were used when fitting mode shapes.  ' +
            ('All frequency lines were used to fit mode shapes.' if self.ppgui.all_frequency_lines_checkbox.isChecked() else
            'Mode shapes were fit using {:} frequency lines around each resonance, and {:} frequency lines were used to fit residuals.'.format(
                self.ppgui.lines_at_resonance_spinbox.value(),self.ppgui.lines_at_residuals_spinbox.value())))
    
    def fit_modes_SMAC(self):
        raise NotImplementedError('SMAC has not been implemented yet')
    
    def retrieve_modes_SMAC(self):
        raise NotImplementedError('SMAC has not been implemented yet')
    
    def fit_modes_opoly(self):
        raise NotImplementedError('Opoly has not been implemented yet')
        
    def retrieve_modes_opoly(self,fit_modes,
                             opoly_progress = None,
                             opoly_shape_info = None,
                             opoly_mif_override = None,
                             stabilization_subplots_kwargs = None,
                             stabilization_plot_kwargs = None):
        stabilization_axis = None
        if stabilization_subplots_kwargs is None:
            stabilization_subplots_kwargs = {}
        if stabilization_plot_kwargs is None:
            stabilization_plot_kwargs = {}

        if isinstance(fit_modes,str):
            if opoly_shape_info is None:
                opoly_shape_info = fit_modes+'.info.csv'
                if not os.path.exists(opoly_shape_info):
                    opoly_shape_info = None
            fit_modes = shape_load(fit_modes)
            self.fit_modes = fit_modes

        categories = ['Poly Sieve',
                      'Poly Model',
                      'Poly Range',
                      'Stability',
                      'Autonomous',
                      'Shapes Sieve',
                      'Shapes Model',
                      'Shapes Range',
                      'Pole List'
                      ]

        opoly_settings = {category:{} for category in categories}

        if opoly_progress is not None:
            if isinstance(opoly_progress,str):
                opoly_progress = loadmat(opoly_progress)
            # Pull out all of the settings
            opoly_settings['OPoly Version'] = str(opoly_progress['OPOLY_PROGRESS_001']['APPINFO'][0,0]['Version'][0,0][0,0])+'.'+str(int(
                opoly_progress['OPOLY_PROGRESS_001']['APPINFO'][0,0]['Version'][0,0][0,1]))
            opoly_settings['OPoly Version'] = str(opoly_progress['OPOLY_PROGRESS_001']['APPINFO'][0,0]['Version'][0,0][0,0])
            opoly_settings['Poly Sieve']['References'] = opoly_progress['OPOLY_PROGRESS_001']['IMATDATA'][0,0]['Sieve'][0,0]['Poles'][0,0]['References'][0,0].flatten()
            opoly_settings['Poly Sieve']['Responses'] = opoly_progress['OPOLY_PROGRESS_001']['IMATDATA'][0,0]['Sieve'][0,0]['Poles'][0,0]['Responses'][0,0].flatten()
            opoly_settings['Poly Model']['Method'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['Method'][0,0][0])
            opoly_settings['Poly Model']['Min Order'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['MinOrder'][0,0][0,0])
            opoly_settings['Poly Model']['Step Order'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['StepOrder'][0,0][0,0])
            opoly_settings['Poly Model']['Max Order'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['MaxOrder'][0,0][0,0])
            opoly_settings['Poly Model']['Clear Ege Lines'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['ClearEdges'][0,0][0,0])
            opoly_settings['Poly Model']['Solver Function'] = 'OPoly M-File'
            opoly_settings['Poly Model']['Frequency Basis'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['Params'][0,0]['FreqSided'][0,0][0])+'-sided'
            opoly_settings['Poly Model']['Identity Order'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['Params'][0,0]['IdentOrder'][0,0][0])
            opoly_settings['Poly Model']['Residuals'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['Params'][0,0]['Residuals'][0,0][0,0])
            opoly_settings['Poly Model']['Fixed Numerator'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['Params'][0,0]['FixedNumer'][0,0][0])
            opoly_settings['Poly Model']['Weighting'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['Params'][0,0]['Weighting'][0,0][0])
            opoly_settings['Poly Model']['Real Participations'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['Params'][0,0]['RealMPF'][0,0][0])
            opoly_settings['Poly Model']['Keep All Poles'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['Params'][0,0]['KeepAll'][0,0][0])
            opoly_settings['Poly Model']['Overdetermination'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['PolyModel'][0,0]['Params'][0,0]['Overdeterm'][0,0][0,0])
            opoly_settings['Poly Range']['Freq Range (Hz)'] = [str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['FreqRange'][0,0]['Lower'][0,0][0,0]),str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['FreqRange'][0,0]['Upper'][0,0][0,0])]
            opoly_settings['Poly Range']['Spectral Lines'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Setup'][0,0]['FreqRange'][0,0]['Lines'][0,0][0,0])
            opoly_settings['Stability']['Frequency (%)'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Poles'][0,0]['Stability'][0,0]['Frequency'][0,0][0,0]*100)
            opoly_settings['Stability']['Damping (%)'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Poles'][0,0]['Stability'][0,0]['Damping'][0,0][0,0]*100)
            opoly_settings['Stability']['Vector (%)'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Poles'][0,0]['Stability'][0,0]['Vector'][0,0][0,0]*100)
            opoly_settings['Autonomous']['Minimum Pole Density'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Poles'][0,0]['Autonomous'][0,0]['PoleDensity'][0,0][0,0])
            opoly_settings['Autonomous']['Pole Weighted Vector Order'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Poles'][0,0]['Autonomous'][0,0]['WeightedOrder'][0,0][0,0])
            opoly_settings['Autonomous']['Pole Weighted Vector MAC'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Poles'][0,0]['Autonomous'][0,0]['WeightedMAC'][0,0][0,0])
            opoly_settings['Autonomous']['Minimum Cluster Size'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Poles'][0,0]['Autonomous'][0,0]['ClusterSize'][0,0][0,0])
            opoly_settings['Autonomous']['Cluster Inclusion Threshold'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Poles'][0,0]['Autonomous'][0,0]['ClusterInclusion'][0,0][0,0])
            opoly_settings['Shapes Sieve']['References'] = opoly_progress['OPOLY_PROGRESS_001']['IMATDATA'][0,0]['Sieve'][0,0]['Shapes'][0,0]['References'][0,0].flatten()
            opoly_settings['Shapes Sieve']['Responses'] = opoly_progress['OPOLY_PROGRESS_001']['IMATDATA'][0,0]['Sieve'][0,0]['Shapes'][0,0]['Responses'][0,0].flatten()
            opoly_settings['Shapes Model']['Method'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Shapes'][0,0]['ResiModel'][0,0]['Method'][0,0][0])
            opoly_settings['Shapes Model']['Type'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Shapes'][0,0]['ResiModel'][0,0]['Type'][0,0][0])
            opoly_settings['Shapes Model']['FRF Parts'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Shapes'][0,0]['ResiModel'][0,0]['FrfParts'][0,0][0])
            opoly_settings['Shapes Model']['Residuals'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Shapes'][0,0]['ResiModel'][0,0]['Residuals'][0,0][0])
            opoly_settings['Shapes Model']['Solver Function'] = 'OPoly M-File'
            opoly_settings['Shapes Model']['Frequency Basis'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Shapes'][0,0]['ResiModel'][0,0]['Params'][0,0]['FreqSided'][0,0][0])+'-sided'
            opoly_settings['Shapes Model']['Refit Freq'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Shapes'][0,0]['ResiModel'][0,0]['RefitFreq'][0,0][0])
            opoly_settings['Shapes Range']['Freq Range (Hz)'] = [str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Shapes'][0,0]['Refit'][0,0]['FreqRange'][0,0]['Lower'][0,0][0,0]),str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Shapes'][0,0]['Refit'][0,0]['FreqRange'][0,0]['Upper'][0,0][0,0])]
            opoly_settings['Shapes Range']['Spectral Lines'] = str(opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Shapes'][0,0]['Refit'][0,0]['FreqRange'][0,0]['Lines'][0,0][0,0])
            keep = np.array([v[0,0] for v in opoly_progress['OPOLY_PROGRESS_001']['POLELIST'][0,0]['Checked'].flatten()])
            opoly_settings['Pole List']['Model Order'] = np.array([v[0,0] for v in opoly_progress['OPOLY_PROGRESS_001']['POLELIST'][0,0]['ModelOrder'].flatten()])[keep.astype(bool)]
            opoly_settings['Pole List']['Pole Index'] = np.array([v[0,0] for v in opoly_progress['OPOLY_PROGRESS_001']['POLELIST'][0,0]['PoleIndex'].flatten()])[keep.astype(bool)]
            # Recreate the stabilization diagram in OPoly
            if opoly_mif_override is None:
                mif_type = opoly_progress['OPOLY_PROGRESS_001']['STATE'][0,0]['Poles'][0,0]['Plot'][0,0]['Function'][0,0][0,0][0]
            else:
                mif_type = opoly_mif_override
            if mif_type.lower() == 'cmif':
                mif_log = True
                mif = self.frfs.compute_cmif()
            elif mif_type.lower() == 'qmif':
                mif_log = True
                mif = self.frfs.compute_cmif(part='imag')
            elif mif_type.lower() == 'mmif':
                mif_log = False
                mif = self.frfs.compute_mmif()
            elif mif_type.lower() == 'nmif':
                mif_log = False
                mif = self.frfs.compute_nmif()
            elif mif_type.lower() == 'psmif':
                mif_log = True
                mif = self.frfs.compute_cmif()[0]
            else:
                raise ValueError('Unknown mode indicator function {:}'.format(mif_type))
            poles = [v.flatten() for v in opoly_progress['OPOLY_PROGRESS_001'][0,0]['POLYDATA']['Poles'][0,:]]
            stabilities = [[str(v[0]) for v in row.flatten()] for row in opoly_progress['OPOLY_PROGRESS_001'][0,0]['POLYDATA']['Stability'][0,:]]
            orders = [str(row[0,0]) for row in opoly_progress['OPOLY_PROGRESS_001'][0,0]['POLYDATA']['ModelOrder'][0,:]]
        else:
            poles = None
            mif = self.frfs.compute_cmif()
            mif_log = True

        if opoly_shape_info is not None:    
            with open(opoly_shape_info,'r') as f:
                for line in f:
                    if line[:5] == 'Notes':
                        break
                    parts = [v.strip() for v in line.split(',')]
                    if parts[0] in categories:
                        category,field,*data = parts
                        opoly_settings[category][field] = data[0] if len(data) == 1 else data
                    else:
                        opoly_settings[parts[0]] = parts[1] if len(parts[1:]) == 1 else parts[1:]

        try:
            fmin,fmax = opoly_settings['Shapes Range']['Freq Range (Hz)']
            mif = mif.extract_elements_by_abscissa(float(fmin),float(fmax))
        except KeyError:
            pass

        stabilization_axis = mif.plot(True,stabilization_subplots_kwargs,stabilization_plot_kwargs)
        if mif_log:
            stabilization_axis.set_yscale('log')
        stabilization_axis.set_xlabel('Frequency (Hz)')

        if poles is not None:
            ax_poles = stabilization_axis.twinx()
            legend_data = {}
            legend_choices = {'xf':'No Stability',
                              'xt':'No Stability\nSelected For Final Mode Set',
                              'sf':'Pole Stability',
                              'st':'Pole Stability\nSelected For Final Mode Set',
                              'of':'Vector Stability',
                              'ot':'Vector Stability\nSelected For Final Mode Set',
                              'af':'Autonomous Selection',
                              'at':'Autonomous Selection\nSelected For Final Mode Set',
                              }
            picked_model_orders = []
            picked_pole_indices = []
            for order,index in zip(opoly_settings['Pole List']['Model Order'],
                                   opoly_settings['Pole List']['Pole Index']):
                try:
                    picked_model_orders.append(int(order))
                except (ValueError,OverflowError):
                    picked_model_orders.append('Auto')
                picked_pole_indices.append(int(index)-1)
            picked_pole_indices = np.array(picked_pole_indices)
            for order,pole,stability in zip(orders,poles,stabilities):
                try:
                    order = int(order)
                except ValueError:
                    order = 'Auto'
                if order == 0:
                    continue
                for index,(freq,stab) in enumerate(zip(np.abs(pole),stability)):
                    if order == 'Auto':
                        kwargs = {'marker':'*','markeredgecolor':'y','markerfacecolor':'y','color':'none','markersize':8}
                        l = 'a'
                    elif stab == 'none':
                        continue
                        kwargs = {'marker':'x','markeredgecolor':'r','markerfacecolor':'r','color':'none','markersize':5}
                        l = 'x'
                    elif stab == 'freq':
                        kwargs = {'marker':'s','markeredgecolor':'b','markerfacecolor':'b','color':'none','markersize':5}
                        l = 's'
                    elif stab == 'damp':
                        kwargs = {'marker':'s','markeredgecolor':'b','markerfacecolor':'b','color':'none','markersize':5}
                        l = 's'
                    elif stab == 'vect':
                        kwargs = {'marker':'o','markeredgecolor':'g','markerfacecolor':'g','color':'none','markersize':5}
                        l = 'o'
                    elif stab == '[]':
                        continue
                        kwargs = {'marker':'x','markeredgecolor':'r','markerfacecolor':'r','color':'none','markersize':5}
                        l = 'x'
                    # Check to see if the index is in the range
                    same_orders = [order == o for o in picked_model_orders]
                    is_picked = any(picked_pole_indices[same_orders] == index)
                    if is_picked:
                        s = 't'
                        kwargs['markeredgecolor'] = 'k'
                    else:
                        s = 'f'
                        kwargs['markerfacecolor'] = 'none'
                        kwargs['markersize'] = 3
                    legend_data[l+s] = ax_poles.plot(freq,(int(orders[-2])*2-int(orders[-3])) if order == 'Auto' else order,**kwargs)[0]
            # Create the legend
            legend_strings = []
            legend_handles = []
            for choice,string in legend_choices.items():
                if choice in legend_data:
                    legend_strings.append(string)
                    legend_handles.append(legend_data[choice])
            ax_poles.legend(legend_handles,legend_strings)
            ax_poles.set_ylabel('Polynomial Model Order')
            fig = ax_poles.figure
            # bio = io.BytesIO()
            # fig.savefig(bio,format='png')
            # bio.seek(0)
            # fig_bytes = bio.read()
            self.fit_modes_information = {'text':[
                'Modes were fit to the data using the OPoly (version {:}) curve fitter implemented in the IMAT Matlab toolbox.'.format(opoly_settings['OPoly Version'])]}
            
            self.fit_modes_information['text'].append((
                'The frequency band from {:0.2f} to {:0.2f} was analyzed with polynomials from order {:} to {:} using the {:} method.  '+
                '{:} poles were selected from this band.  The stabilization diagram is shown in Figure {{figure1ref:}}.'
                ).format(*[float(v) for v in opoly_settings['Poly Range']['Freq Range (Hz)']],
                         opoly_settings['Poly Model']['Min Order'],opoly_settings['Poly Model']['Max Order'],
                         opoly_settings['Poly Model']['Method'].upper(),
                         len(opoly_settings['Pole List']['Pole Index'])))
            self.fit_modes_information['figure1'] = fig
            self.fit_modes_information['figure1caption'] = 'Stabilization Diagram from OPoly showing stable poles and those selected in the final mode set.'
            self.fit_modes_information['text'].append((
                '{:} modes were used to fit the data.  {:} residuals were used when fitting the mode shapes.').format(
                    opoly_settings['Shapes Model']['Type'].title(),
                    opoly_settings['Shapes Model']['Residuals'].replace('+',' and ').title()))
    
    def edit_mode_comments(self,mif = 'cmif'):
        if self.fit_modes is None:
            raise ValueError('Modes have not yet been fit or assigned.')
        getattr(self,'plot_{:}'.format(mif))(measured=True,resynthesized=True,mark_modes=True)
        return self.fit_modes.edit_comments(self.geometry)
    
    def compute_resynthesized_frfs(self):
        self.resynthesized_frfs = self.fit_modes.compute_frf(self.frfs.flatten()[0].abscissa,
                                                             np.unique(self.frfs.response_coordinate),
                                                             np.unique(self.frfs.reference_coordinate),
                                                             )[self.frfs.coordinate]
    
    def plot_reference_autospectra(self, plot_kwargs = {}, subplots_kwargs = {}):
        if self.autopower_spectra is None:
            raise ValueError('Autopower Spectra have not yet been computed or assigned')
        reference_apsd = self.autopower_spectra[self.autopower_spectra_reference_indices]
        ax = reference_apsd.plot(one_axis=False, plot_kwargs=plot_kwargs, subplots_kwargs = subplots_kwargs)
        for a in ax.flatten():
            if self.reference_unit is not None:
                a.set_ylabel(a.get_ylabel()+'\n({:}$^2$/Hz)'.format(self.reference_unit))
            a.set_xlabel('Frequency (Hz)')
            a.set_yscale('log')
        return ax.flatten()[0].figure, ax
    
    def plot_drive_point_frfs(self, part='imag', plot_kwargs = {}, subplots_kwargs = {}):
        if self.frfs is None:
            raise ValueError('FRFs have not yet been computed or assigned')
        ax = self.frfs.get_drive_points().plot(one_axis=False,part=part, plot_kwargs=plot_kwargs, subplots_kwargs = subplots_kwargs)
        for a in ax.flatten():
            if self.reference_unit is not None and self.response_unit is not None:
                a.set_ylabel(a.get_ylabel()+'\n({:}/{:})'.format(self.response_unit, self.reference_unit))
            a.set_xlabel('Frequency (Hz)')
        return ax.flatten()[0].figure, ax
    
    def plot_reciprocal_frfs(self, plot_kwargs = {}, subplots_kwargs = {}):
        if self.frfs is None:
            raise ValueError('FRFs have not yet been computed or assigned')
        reciprocal_frfs = self.frfs.get_reciprocal_data()
        axes = reciprocal_frfs[0].plot(one_axis=False, plot_kwargs=plot_kwargs, subplots_kwargs = subplots_kwargs)
        for ax, original_frf, reciprocal_frf in zip(axes.flatten(),*reciprocal_frfs):
            reciprocal_frf.plot(ax, **plot_kwargs)
            ax.legend([
                '/'.join([str(coord) for coord in original_frf.coordinate]),
                '/'.join([str(coord) for coord in reciprocal_frf.coordinate])])
            ax.set_xlabel('Frequency (Hz)')
            if self.reference_unit is not None and self.response_unit is not None:
                ax.set_ylabel(ax.get_ylabel()+'\n({:}/{:})'.format(self.response_unit, self.reference_unit))
        return axes.flatten()[0].figure,axes
    
    def plot_coherence_image(self):
        if self.coherence is None:
            raise ValueError('Coherence has not yet been computed or assigned')
        ax = self.coherence.plot_image(colorbar_min = 0, colorbar_max = 1)
        ax.set_ylabel('Degree of Freedom')
        ax.set_xlabel('Frequency (Hz)')
        return ax.figure, ax
        
    def plot_drive_point_frf_coherence(self, plot_kwargs = {}, subplots_kwargs = {}):
        if self.frfs is None:
            raise ValueError('FRFs have not yet been computed or assigned')
        if self.coherence is None:
            raise ValueError('Coherence has not yet been computed or assigned')
        frf_ax, coh_ax = self.frfs.get_drive_points().plot_with_coherence(self.coherence, plot_kwargs = plot_kwargs, subplots_kwargs = subplots_kwargs)
        for a in frf_ax.flatten():
            if self.reference_unit is not None and self.response_unit is not None:
                a.set_ylabel(a.get_ylabel()+'\n({:}/{:})'.format(self.response_unit, self.reference_unit))
            a.set_xlabel('Frequency (Hz)')
        return frf_ax.flatten()[0].figure, frf_ax, coh_ax
    
    def plot_cmif(self, measured = True, resynthesized = False, mark_modes = False,
                  measured_plot_kwargs = {}, resynthesized_plot_kwargs = {},
                  subplots_kwargs = {}):
        """
        Plots the complex mode indicator function

        Parameters
        ----------
        measured : bool, optional
            If True, plots the measured MIF. The default is True.
        resynthesized : bool, optional
            If True, plots resynthesized MIF. The default is False.
        mark_modes : bool, optional
            If True, plots a vertical line at the frequency of each mode. The
            default is False.
        measured_plot_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the measured
            data is plotted. The default is {}.
        resynthesized_plot_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the 
            resynthesized data is plotted. The default is {}.
        subplots_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the figure
            and axes are created.  This is passed to the plt.subplots function.
            The default is {}.

        Raises
        ------
        ValueError
            Raised if a required data has not been computed or assigned yet.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A reference to the figure on which the plot is plotted.
        ax : matplotlib.axes.Axes
            A reference to the axes on which the plot is plotted.

        """
        fig, ax = plt.subplots(1,1,**subplots_kwargs)
        if measured and resynthesized:
            if self.frfs is None:
                raise ValueError('FRFs have not yet been computed or assigned')
            cmif = self.frfs.compute_cmif()
            cmif.plot(ax, plot_kwargs = measured_plot_kwargs)
            ax.set_yscale('log')
            ylim = ax.get_ylim()
        elif measured and not resynthesized:
            if self.frfs is None:
                raise ValueError('FRFs have not yet been computed or assigned')
            cmif = self.frfs.compute_cmif()
            cmif.plot(ax, plot_kwargs = measured_plot_kwargs,
                      abscissa_markers = self.fit_modes.frequency
                      if mark_modes else None, abscissa_marker_plot_kwargs={'linewidth':0.5,
                                                                            'linestyle':'--',
                                                                            'alpha':0.5,
                                                                            'color':'k'},
                      abscissa_marker_labels = '{abscissa:0.1f}')
            ax.set_yscale('log')
            ylim = ax.get_ylim()
        else:
            ylim = None
        if resynthesized:
            if self.resynthesized_frfs is None:
                raise ValueError('Resynthesized FRFs have not yet been computed or assigned')
            cmif = self.resynthesized_frfs[self.frfs.coordinate].compute_cmif()
            cmif.plot(ax, plot_kwargs = resynthesized_plot_kwargs,
                      abscissa_markers = self.fit_modes.frequency
                      if mark_modes else None, abscissa_marker_plot_kwargs={'linewidth':0.5,
                                                                            'linestyle':'--',
                                                                            'alpha':0.5,
                                                                            'color':'k'},
                      abscissa_marker_labels = '{abscissa:0.1f}')
            ax.set_yscale('log')
            if ylim is not None:
                ax.set_ylim(ylim)
        if measured and resynthesized:
            ax.legend([ax.lines[0],ax.lines[cmif.size]],['Measured','Resynthesized'],loc='upper right')
        ax.set_xlabel('Frequency (Hz)')
        if self.reference_unit is not None and self.response_unit is not None:
            ax.set_ylabel('Complex Mode Indicator Function ({:}/{:})'.format(self.response_unit,self.reference_unit))
        else:
            ax.set_ylabel('Complex Mode Indicator Function')
        return fig, ax
    
    def plot_qmif(self, measured = True, resynthesized = False, mark_modes = False,
                  measured_plot_kwargs = {}, resynthesized_plot_kwargs = {},
                  subplots_kwargs = {}):
        """
        Plots the complex mode indicator function computed from the imaginary
        part of the FRFs

        Parameters
        ----------
        measured : bool, optional
            If True, plots the measured MIF. The default is True.
        resynthesized : bool, optional
            If True, plots resynthesized MIF. The default is False.
        mark_modes : bool, optional
            If True, plots a vertical line at the frequency of each mode. The
            default is False.
        measured_plot_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the measured
            data is plotted. The default is {}.
        resynthesized_plot_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the 
            resynthesized data is plotted. The default is {}.
        subplots_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the figure
            and axes are created.  This is passed to the plt.subplots function.
            The default is {}.

        Raises
        ------
        ValueError
            Raised if a required data has not been computed or assigned yet.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A reference to the figure on which the plot is plotted.
        ax : matplotlib.axes.Axes
            A reference to the axes on which the plot is plotted.

        """
        fig, ax = plt.subplots(1,1,**subplots_kwargs)
        if measured:
            if self.frfs is None:
                raise ValueError('FRFs have not yet been computed or assigned')
            cmif = self.frfs.compute_cmif(part='imag')
            cmif.plot(ax, plot_kwargs = measured_plot_kwargs)
            ax.set_yscale('log')
            ylim = ax.get_ylim()
        else:
            ylim = None
        if resynthesized:
            if self.resynthesized_frfs is None:
                raise ValueError('Resynthesized FRFs have not yet been computed or assigned')
            cmif = self.resynthesized_frfs[self.frfs.coordinate].compute_cmif(part='imag')
            cmif.plot(ax, plot_kwargs = resynthesized_plot_kwargs,
                      abscissa_markers = self.fit_modes.frequency
                      if mark_modes else None, abscissa_marker_plot_kwargs={'linewidth':0.5,
                                                                            'linestyle':'--',
                                                                            'alpha':0.5,
                                                                            'color':'k'},
                      abscissa_marker_labels = '{abscissa:0.1f}')
            ax.set_yscale('log')
            if ylim is not None:
                ax.set_ylim(ylim)
        if measured and resynthesized:
            ax.legend([ax.lines[0],ax.lines[cmif.size]],['Measured','Resynthesized'],loc='upper right')
        ax.set_xlabel('Frequency (Hz)')
        if self.reference_unit is not None and self.response_unit is not None:
            ax.set_ylabel('Quadrature Mode Indicator Function ({:}/{:})'.format(self.response_unit,self.reference_unit))
        else:
            ax.set_ylabel('Quadrature Mode Indicator Function')
        return fig, ax
    
    def plot_psmif(self, measured = True, resynthesized = False, mark_modes = False,
                   measured_plot_kwargs = {}, resynthesized_plot_kwargs = {},
                   subplots_kwargs = {}):
        """
        Plots the first singular value of the complex mode indicator function

        Parameters
        ----------
        measured : bool, optional
            If True, plots the measured MIF. The default is True.
        resynthesized : bool, optional
            If True, plots resynthesized MIF. The default is False.
        mark_modes : bool, optional
            If True, plots a vertical line at the frequency of each mode. The
            default is False.
        measured_plot_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the measured
            data is plotted. The default is {}.
        resynthesized_plot_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the 
            resynthesized data is plotted. The default is {}.
        subplots_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the figure
            and axes are created.  This is passed to the plt.subplots function.
            The default is {}.

        Raises
        ------
        ValueError
            Raised if a required data has not been computed or assigned yet.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A reference to the figure on which the plot is plotted.
        ax : matplotlib.axes.Axes
            A reference to the axes on which the plot is plotted.

        """
        fig, ax = plt.subplots(1,1,**subplots_kwargs)
        if measured:
            if self.frfs is None:
                raise ValueError('FRFs have not yet been computed or assigned')
            cmif = self.frfs.compute_cmif()[:1]
            cmif.plot(ax, plot_kwargs = measured_plot_kwargs)
            ax.set_yscale('log')
            ylim = ax.get_ylim()
        else:
            ylim = None
        if resynthesized:
            if self.resynthesized_frfs is None:
                raise ValueError('Resynthesized FRFs have not yet been computed or assigned')
            cmif = self.resynthesized_frfs[self.frfs.coordinate].compute_cmif()[:1]
            cmif.plot(ax, plot_kwargs = resynthesized_plot_kwargs,
                      abscissa_markers = self.fit_modes.frequency
                      if mark_modes else None, abscissa_marker_plot_kwargs={'linewidth':0.5,
                                                                            'linestyle':'--',
                                                                            'alpha':0.5,
                                                                            'color':'k'},
                      abscissa_marker_labels = '{abscissa:0.1f}')
            ax.set_yscale('log')
            if ylim is not None:
                ax.set_ylim(ylim)
        if measured and resynthesized:
            ax.legend([ax.lines[0],ax.lines[cmif.size]],['Measured','Resynthesized'],loc='upper right')
        ax.set_xlabel('Frequency (Hz)')
        if self.reference_unit is not None and self.response_unit is not None:
            ax.set_ylabel('Principal Singular Value\nMode Indicator Function ({:}/{:})'.format(self.response_unit,self.reference_unit))
        else:
            ax.set_ylabel('Principal Singular Value\nMode Indicator Function')
        return fig, ax
    
    def plot_nmif(self, measured = True, resynthesized = False, mark_modes = False,
                  measured_plot_kwargs = {}, resynthesized_plot_kwargs = {},
                  subplots_kwargs = {}):
        """
        Plots the normal mode indicator function

        Parameters
        ----------
        measured : bool, optional
            If True, plots the measured MIF. The default is True.
        resynthesized : bool, optional
            If True, plots resynthesized MIF. The default is False.
        mark_modes : bool, optional
            If True, plots a vertical line at the frequency of each mode. The
            default is False.
        measured_plot_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the measured
            data is plotted. The default is {}.
        resynthesized_plot_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the 
            resynthesized data is plotted. The default is {}.
        subplots_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the figure
            and axes are created.  This is passed to the plt.subplots function.
            The default is {}.

        Raises
        ------
        ValueError
            Raised if a required data has not been computed or assigned yet.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A reference to the figure on which the plot is plotted.
        ax : matplotlib.axes.Axes
            A reference to the axes on which the plot is plotted.

        """
        fig, ax = plt.subplots(1,1,**subplots_kwargs)
        if measured:
            if self.frfs is None:
                raise ValueError('FRFs have not yet been computed or assigned')
            cmif = self.frfs.compute_nmif()
            cmif.plot(ax, plot_kwargs = measured_plot_kwargs)
            ylim = ax.get_ylim()
        else:
            ylim = None
        if resynthesized:
            if self.resynthesized_frfs is None:
                raise ValueError('Resynthesized FRFs have not yet been computed or assigned')
            cmif = self.resynthesized_frfs[self.frfs.coordinate].compute_nmif()
            cmif.plot(ax, plot_kwargs = resynthesized_plot_kwargs,
                      abscissa_markers = self.fit_modes.frequency
                      if mark_modes else None, abscissa_marker_plot_kwargs={'linewidth':0.5,
                                                                            'linestyle':'--',
                                                                            'alpha':0.5,
                                                                            'color':'k'},
                      abscissa_marker_labels = '{abscissa:0.1f}')
            if ylim is not None:
                ax.set_ylim(ylim)
        if measured and resynthesized:
            ax.legend([ax.lines[0],ax.lines[cmif.size]],['Measured','Resynthesized'],loc='upper right')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normal Mode Indicator Function')
        return fig, ax

    def plot_mmif(self, measured = True, resynthesized = False, mark_modes = False,
                  measured_plot_kwargs = {}, resynthesized_plot_kwargs = {},
                  subplots_kwargs = {}):
        """
        Plots the multi mode indicator function

        Parameters
        ----------
        measured : bool, optional
            If True, plots the measured MIF. The default is True.
        resynthesized : bool, optional
            If True, plots resynthesized MIF. The default is False.
        mark_modes : bool, optional
            If True, plots a vertical line at the frequency of each mode. The
            default is False.
        measured_plot_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the measured
            data is plotted. The default is {}.
        resynthesized_plot_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the 
            resynthesized data is plotted. The default is {}.
        subplots_kwargs : dict, optional
            Dictionary containing keyword arguments to specify how the figure
            and axes are created.  This is passed to the plt.subplots function.
            The default is {}.

        Raises
        ------
        ValueError
            Raised if a required data has not been computed or assigned yet.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A reference to the figure on which the plot is plotted.
        ax : matplotlib.axes.Axes
            A reference to the axes on which the plot is plotted.

        """
        fig, ax = plt.subplots(1,1,**subplots_kwargs)
        if measured:
            if self.frfs is None:
                raise ValueError('FRFs have not yet been computed or assigned')
            cmif = self.frfs.compute_mmif()
            cmif.plot(ax, plot_kwargs = measured_plot_kwargs)
            ylim = ax.get_ylim()
        else:
            ylim = None
        if resynthesized:
            if self.resynthesized_frfs is None:
                raise ValueError('Resynthesized FRFs have not yet been computed or assigned')
            cmif = self.resynthesized_frfs[self.frfs.coordinate].compute_mmif()
            cmif.plot(ax, plot_kwargs = resynthesized_plot_kwargs,
                      abscissa_markers = self.fit_modes.frequency
                      if mark_modes else None, abscissa_marker_plot_kwargs={'linewidth':0.5,
                                                                            'linestyle':'--',
                                                                            'alpha':0.5,
                                                                            'color':'k'},
                      abscissa_marker_labels = '{abscissa:0.1f}')
            if ylim is not None:
                ax.set_ylim(ylim)
        if measured and resynthesized:
            ax.legend([ax.lines[0],ax.lines[cmif.size]],['Measured','Resynthesized'],loc='upper right')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Multi Mode Indicator Function')
        return fig, ax
    
    def plot_deflection_shapes(self):
        if self.frfs is None:
            raise ValueError('FRFs have not yet been computed or assigned')
        if self.geometry is None:
            raise ValueError('Geometry must be assigned to plot deflection shapes')
        frfs = self.frfs.reshape_to_matrix()
        plotters = []
        for frf in frfs.T:
            plotters.append(self.geometry.plot_deflection_shape(frf))
        return plotters
    
    def plot_mac(self, *matrix_plot_args, **matrix_plot_kwargs):
        if self.fit_modes is None:
            raise ValueError('Modes have not yet been fit or assigned.')
        mac_matrix = mac(self.fit_modes)
        ax = matrix_plot(mac_matrix, *matrix_plot_args, **matrix_plot_kwargs)
        return ax.figure, ax
    
    def plot_modeshape(self):
        if self.fit_modes is None:
            raise ValueError('Modes have not yet been fit or assigned.')
        if self.geometry is None:
            raise ValueError('Geometry must be assigned to plot deflection shapes')
        return self.geometry.plot_shape(self.fit_modes)
    
    
    def plot_figures_for_documentation(self,
                                       plot_geometry = True,
                                       geometry_kwargs = {},
                                       plot_coordinate = True,
                                       coordinate_kwargs = {},
                                       plot_rigid_body_checks = True,
                                       rigid_body_checks_kwargs = {},
                                       plot_reference_autospectra = True,
                                       reference_autospectra_kwargs = {},
                                       plot_drive_point_frfs = True, 
                                       drive_point_frfs_kwargs = {},
                                       plot_reciprocal_frfs = True,
                                       reciprocal_frfs_kwargs = {},
                                       plot_frf_coherence = True,
                                       frf_coherence_kwargs = {},
                                       plot_coherence_image = True,
                                       coherence_image_kwargs = {},
                                       plot_cmif = True,
                                       cmif_kwargs = {},
                                       plot_qmif = False,
                                       qmif_kwargs = {},
                                       plot_nmif = False,
                                       nmif_kwargs = {},
                                       plot_mmif = False,
                                       mmif_kwargs = {},
                                       plot_modeshapes = True,
                                       modeshape_kwargs = {},
                                       plot_mac = True, 
                                       mac_kwargs = {},
                                       ):
        self.documentation_figures = {}
        if plot_geometry:
            if self.geometry is None:
                print('Warning: Could not plot geometry; geometry is undefined.')
            else:
                plotter = self.geometry.plot(**geometry_kwargs)[0]
                self.documentation_figures['geometry'] = plotter
        if plot_coordinate:
            if self.geometry is None:
                print('Warning: Could not plot coordinates; geometry is undefined.')
            elif self.time_histories is None:
                print('Warning: Could not plot coordinates; time_histories is undefined')
            else:
                if not 'plot_kwargs' in coordinate_kwargs:
                    coordinate_kwargs['plot_kwargs'] = geometry_kwargs
                # Check and see if the references are defined
                if self.reference_indices is not None:
                    plotter = self.geometry.plot_coordinate(
                        self.time_histories[self.response_indices].coordinate.flatten(),
                        **coordinate_kwargs)
                    self.documentation_figures['response_coordinate'] = plotter
                    plotter = self.geometry.plot_coordinate(
                        self.time_histories[self.reference_indices].coordinate.flatten(),
                        **coordinate_kwargs)
                    self.documentation_figures['reference_coordinate'] = plotter
                else:
                    plotter = self.geometry.plot_coordinate(
                        self.time_histories.coordinate.flatten(),
                        **coordinate_kwargs)
                    self.documentation_figures['coordinate'] = plotter
        if plot_rigid_body_checks:
            if self.geometry is None:
                print('Warning: Could not plot rigid body checks; geometry is undefined.')
            elif self.rigid_body_shapes is None:
                print('Warning: Could not plot rigid body checks; rigid_body_shapes is undefined.')
            else:
                common_nodes = np.intersect1d(self.geometry.node.id,np.unique(self.rigid_body_shapes.coordinate.node))
                geometry = self.geometry.reduce(common_nodes)
                rigid_shapes = self.rigid_body_shapes.reduce(common_nodes)
                supicious_channels, *figures = rigid_body_check(geometry, rigid_shapes,
                                                                return_figures = True, **rigid_body_checks_kwargs)
                num_figs = self.rigid_body_shapes.size+1
                figures = figures[-num_figs:]
                for i in range(num_figs-1):
                    self.documentation_figures['rigid_body_complex_plane_{:}'.format(i)] = figures[i]
                self.documentation_figures['rigid_body_residuals'] = figures[-1]
                plotter = self.geometry.plot_shape(
                    self.rigid_body_shapes,
                    plot_kwargs = geometry_kwargs)
                self.documentation_figures['rigid_body_shapes'] = plotter
        if plot_reference_autospectra:
            if self.autopower_spectra is None:
                print('Warning: Could not plot reference autospectra; autopower_spectra is undefined')
            else:
                fig,ax = self.plot_reference_autospectra(**reference_autospectra_kwargs)
                self.documentation_figures['reference_autospectra'] = fig
        if plot_drive_point_frfs:
            if self.frfs is None:
                print('Warning: Could not plot drive point FRFs; frfs is undefined')
            else:
                kwargs = {'part':'imag'}
                kwargs.update(drive_point_frfs_kwargs)
                fig,ax = self.plot_drive_point_frfs(**kwargs)
                self.documentation_figures['drive_point_frf'] = fig
        if plot_reciprocal_frfs:
            if self.frfs is None:
                print('Warning: Could not plot drive point FRFs; frfs is undefined')
            else:
                kwargs = {}
                kwargs.update(reciprocal_frfs_kwargs)
                fig,ax = self.plot_reciprocal_frfs(**kwargs)
                self.documentation_figures['reciprocal_frfs'] = fig
        if plot_frf_coherence:
            if self.frfs is None:
                print('Warning: Could not plot FRF coherence; frfs is undefined')
            elif self.coherence is None:
                print('Warning: Could not plot FRF coherence; coherence is undefined')
            else:
                fig, ax, cax = self.plot_drive_point_frf_coherence(**frf_coherence_kwargs)
                self.documentation_figures['frf_coherence'] = fig
        if plot_coherence_image:
            if self.coherence is None:
                print('Warning: Could not plot coherence; coherence is undefined')
            else:
                fig,ax = self.plot_coherence_image(**coherence_image_kwargs)
                self.documentation_figures['coherence'] = fig
        if plot_cmif:
            if self.frfs is None:
                print('Warning: Could not plot CMIF; frfs is undefined')
            else:
                fig,ax = self.plot_cmif(True,self.resynthesized_frfs is not None,
                                        self.fit_modes is not None, cmif_kwargs, cmif_kwargs)
                self.documentation_figures['cmif'] = fig
        if plot_qmif:
            if self.frfs is None:
                print('Warning: Could not plot QMIF; frfs is undefined')
            else:
                fig,ax = self.plot_qmif(True,self.resynthesized_frfs is not None,
                                        self.fit_modes is not None, qmif_kwargs, qmif_kwargs)
                self.documentation_figures['qmif'] = fig
        if plot_nmif:
            if self.frfs is None:
                print('Warning: Could not plot NMIF; frfs is undefined')
            else:
                fig,ax = self.plot_nmif(True,self.resynthesized_frfs is not None,
                                        self.fit_modes is not None, nmif_kwargs, nmif_kwargs)
                self.documentation_figures['nmif'] = fig
        if plot_mmif:
            if self.frfs is None:
                print('Warning: Could not plot MMIF; frfs is undefined')
            else:
                fig,ax = self.plot_mmif(True,self.resynthesized_frfs is not None,
                                        self.fit_modes is not None, mmif_kwargs, mmif_kwargs)
                self.documentation_figures['mmif'] = fig
        if plot_modeshapes:
            if self.fit_modes is None:
                print('Warning: Cannot plot modeshapes, fit_modes is undefined.')
            elif self.geometry is None:
                print('Warning: Cannot plot modeshapes, geometry is undefined.')
            else:
                plotter = self.geometry.plot_shape(
                    self.fit_modes,
                    plot_kwargs = geometry_kwargs,**modeshape_kwargs)
                self.documentation_figures['mode_shapes'] = plotter
        if plot_mac:
            if self.fit_modes is None:
                print('Warning: Cannot plot MAC, fit_modes is undefined.')
            else:
                fig,ax = self.plot_mac(**mac_kwargs)
                self.documentation_figures['mac'] = fig
                

    def create_documentation_latex(
            self,
            # Important stuff
            coordinate_array='local',
            fit_modes_table = None,
            resynthesis_comparison='cmif',
            resynthesis_figure = None,
            one_file = True,
            # Animation options
            global_animation_style = '2d', 
            geometry_animation_frames=200, geometry_animation_frame_rate=20,
            shape_animation_frames=20, shape_animation_frame_rate=20, 
            animation_style_geometry=None, 
            animation_style_rigid_body=None,
            animation_style_mode_shape=None,
            # Global Paths
            latex_root=r'', figure_root=None,
            # Figure names
            geometry_figure_save_name=None,
            coordinate_figure_save_name=None,
            rigid_body_figure_save_names=None,
            complex_plane_figure_save_names=None,
            residual_figure_save_names=None,
            reference_autospectra_figure_save_names=None,
            drive_point_frfs_figure_save_names=None,
            reciprocal_frfs_figure_save_names=None,
            frf_coherence_figure_save_names=None,
            coherence_figure_save_names=None,
            fit_mode_information_save_names=None,
            mac_plot_save_name=None,
            resynthesis_plot_save_name=None,
            mode_shape_save_names = None,
            # Function KWARGS
            plot_geometry_kwargs={},
            plot_shape_kwargs = {},
            plot_coordinate_kwargs = {},
            rigid_body_check_kwargs={},
            resynthesis_plot_kwargs=None,
            fit_mode_table_kwargs={},
            mac_plot_kwargs=None,
            # Include names
            include_name_geometry=None,
            include_name_signal_processing=None,
            include_name_rigid_body=None,
            include_name_data_quality=None,
            include_name_mode_fitting=None,
            include_name_mode_shape=None,
            include_name_channel_table=None,
            # Arguments for create_geometry_overview
            geometry_figure_label='fig:geometry',
            geometry_figure_caption='Geometry',
            geometry_graphics_options=r'width=0.7\linewidth',
            geometry_animate_graphics_options=r'width=0.7\linewidth,loop',
            geometry_figure_placement='[h]',
            coordinate_figure_label='fig:coordinate',
            coordinate_figure_caption='Local Coordinate Directions (Red: X+, Green: Y+, Blue: Z+)',
            coordinate_graphics_options=r'width=0.7\linewidth',
            coordinate_animate_graphics_options=r'width=0.7\linewidth,loop',
            coordinate_figure_placement='[h]',
            # Arguments for create_rigid_body_analysis
            figure_label_rigid_body='fig:rigid_shapes',
            complex_plane_figure_label='fig:complex_plane',
            residual_figure_label='fig:rigid_shape_residual',
            figure_caption_rigid_body='Rigid body shapes extracted from test data.',
            complex_plane_caption='Complex Plane of the extracted shapes.',
            residual_caption='Rigid body residual showing non-rigid portions of the shapes.',
            graphics_options_rigid_body=r'width=\linewidth',
            complex_plane_graphics_options=r'width=\linewidth',
            residual_graphics_options=r'width=0.7\linewidth',
            animate_graphics_options_rigid_body=r'width=\linewidth,loop',
            figure_placement_rigid_body='',
            complex_plane_figure_placement='',
            residual_figure_placement='',
            subfigure_options_rigid_body=r'[t]{0.45\linewidth}',
            subfigure_labels_rigid_body=None,
            subfigure_captions_rigid_body=None,
            complex_plane_subfigure_options=r'[t]{0.45\linewidth}',
            complex_plane_subfigure_labels=None,
            max_subfigures_per_page_rigid_body=None,
            max_subfigures_first_page_rigid_body=None,
            # Arguments for create_data_quality_summary
            reference_autospectra_figure_label='fig:reference_autospectra',
            reference_autospectra_figure_caption='Autospectra of the reference channels',
            reference_autospectra_graphics_options=r'width=0.7\linewidth',
            reference_autospectra_figure_placement='',
            reference_autospectra_subfigure_options=r'[t]{0.45\linewidth}',
            reference_autospectra_subfigure_labels=None,
            reference_autospectra_subfigure_captions=None,
            drive_point_frfs_figure_label='fig:drive_point_frf',
            drive_point_frfs_figure_caption='Drive point frequency response functions',
            drive_point_frfs_graphics_options=r'width=\linewidth',
            drive_point_frfs_figure_placement='',
            drive_point_frfs_subfigure_options=r'[t]{0.45\linewidth}',
            drive_point_frfs_subfigure_labels=None,
            drive_point_frfs_subfigure_captions=None,
            reciprocal_frfs_figure_label='fig:reciprocal_frfs',
            reciprocal_frfs_figure_caption='Reciprocal frequency response functions.',
            reciprocal_frfs_graphics_options=r'width=0.7\linewidth',
            reciprocal_frfs_figure_placement='',
            reciprocal_frfs_subfigure_options=r'[t]{0.45\linewidth}',
            reciprocal_frfs_subfigure_labels=None,
            reciprocal_frfs_subfigure_captions=None,
            frf_coherence_figure_label='fig:frf_coherence',
            frf_coherence_figure_caption='Drive point frequency response functions with coherence overlaid',
            frf_coherence_graphics_options=r'width=\linewidth',
            frf_coherence_figure_placement='',
            frf_coherence_subfigure_options=r'[t]{0.45\linewidth}',
            frf_coherence_subfigure_labels=None,
            frf_coherence_subfigure_captions=None,
            coherence_figure_label='fig:coherence',
            coherence_figure_caption='Coherence of all channels in the test.',
            coherence_graphics_options=r'width=0.7\linewidth',
            coherence_figure_placement='',
            coherence_subfigure_options=r'[t]{0.45\linewidth}',
            coherence_subfigure_labels=None,
            coherence_subfigure_captions=None,
            max_subfigures_per_page=None,
            max_subfigures_first_page=None,
            # Arguments for create_mode_fitting_summary
            fit_modes_information_table_justification_string=None,
            fit_modes_information_table_longtable=True,
            fit_modes_information_table_header=True,
            fit_modes_information_table_horizontal_lines=False,
            fit_modes_information_table_placement='',
            fit_modes_information_figure_graphics_options=r'width=0.7\linewidth',
            fit_modes_information_figure_placement='',
            fit_modes_table_justification_string=None,
            fit_modes_table_label='tab:mode_fits',
            fit_modes_table_caption='Modal parameters fit to the test data.',
            fit_modes_table_longtable=True,
            fit_modes_table_header=True,
            fit_modes_table_horizontal_lines=False,
            fit_modes_table_placement='',
            fit_modes_table_header_override=None,
            mac_plot_figure_label='fig:mac',
            mac_plot_figure_caption='Modal Assurance Criterion Matrix from Fit Modes', mac_plot_graphics_options=r'width=0.7\linewidth',
            mac_plot_figure_placement='',
            resynthesis_plot_figure_label='fig:resynthesis',
            resynthesis_plot_figure_caption='Test data compared to equivalent data computed from modal fits.',
            resynthesis_plot_graphics_options=r'width=0.7\linewidth',
            resynthesis_plot_figure_placement='',
            # Arguments for create_mode_shape_figures
            figure_label_mode_shape='fig:modeshapes',
            figure_caption_mode_shape='Mode shapes extracted from test data.',
            graphics_options_mode_shape=r'width=\linewidth',
            animate_graphics_options_mode_shape=r'width=\linewidth,loop',
            figure_placement_mode_shape='',
            subfigure_options_mode_shape=r'[t]{0.45\linewidth}',
            subfigure_labels_mode_shape=None,
            subfigure_captions_mode_shape=None,
            max_subfigures_per_page_mode_shape=None, max_subfigures_first_page_mode_shape=None,
        ):
        
        if one_file:
            all_strings = []
        
        if len(self.documentation_figures) == 0:
            print('Warning, you may need to create documentation figures by calling create_figures_for_documentation prior to calling this function!')
        # Set up the files
        if figure_root is None:
            figure_root = os.path.join(latex_root,'figures')
            
        Path(figure_root).mkdir(parents=True,exist_ok=True)
        Path(latex_root).mkdir(parents=True,exist_ok=True)
        
        if animation_style_geometry is None:
            animation_style_geometry = global_animation_style
        if 'geometry' in self.documentation_figures and (animation_style_geometry is None or animation_style_geometry.lower() != '3d'):
            geometry = self.documentation_figures['geometry']
        else:
            geometry = self.geometry
            
        if isinstance(geometry,GeometryPlotter) and not isinstance(coordinate_array,GeometryPlotter):
            if isinstance(coordinate_array,CoordinateArray):
                coordinate_array = self.geometry.plot_coordinate(coordinate_array,**plot_coordinate_kwargs)
            elif coordinate_array == 'local':
                css_to_plot = self.geometry.coordinate_system.id[[not np.allclose(matrix,np.eye(3)) for matrix in self.geometry.coordinate_system.matrix[...,:3,:3]]]
                nodes_to_plot = self.geometry.node.id[np.in1d(self.geometry.node.disp_cs,css_to_plot)]
                coordinate_array = sd_coordinate_array(nodes_to_plot,[1,2,3],force_broadcast=True)
                coordinate_array = self.geometry.plot_coordinate(coordinate_array,**plot_coordinate_kwargs)
            else:
                coordinate_array = None

        print('Creating Geometry Overview')
        geometry_string = create_geometry_overview(
            geometry, plot_geometry_kwargs, coordinate_array, plot_coordinate_kwargs,
            animation_style_geometry, geometry_animation_frames, geometry_animation_frame_rate,
            geometry_figure_label, geometry_figure_caption, geometry_graphics_options, geometry_animate_graphics_options,
            geometry_figure_placement, geometry_figure_save_name, coordinate_figure_label, coordinate_figure_caption,
            coordinate_graphics_options, coordinate_animate_graphics_options, coordinate_figure_placement,
            coordinate_figure_save_name, latex_root, figure_root,
            None if one_file else os.path.join(latex_root,'geometry.tex') if include_name_geometry is None else include_name_geometry
        )
        
        if one_file:
            all_strings.append(geometry_string)

        if 'reference_autospectra' in self.documentation_figures:
            reference_autospectra_figure = self.documentation_figures['reference_autospectra']
        else:
            reference_autospectra_figure = None
        if 'drive_point_frf' in self.documentation_figures:
            drive_point_frfs_figure = self.documentation_figures['drive_point_frf']
        else:
            drive_point_frfs_figure = None
        if 'reciprocal_frfs' in self.documentation_figures:
            reciprocal_frfs_figure = self.documentation_figures['reciprocal_frfs']
        else:
            reciprocal_frfs_figure = None
        if 'frf_coherence' in self.documentation_figures:
            frf_coherence_figure = self.documentation_figures['frf_coherence']
        else:
            frf_coherence_figure = None
        if 'coherence' in self.documentation_figures:
            coherence_figure = self.documentation_figures['coherence']
        else:
            coherence_figure = None
        
        if animation_style_rigid_body is None:
            animation_style_rigid_body = global_animation_style
        
        if 'rigid_body_shapes' in self.documentation_figures and (animation_style_rigid_body is None or animation_style_rigid_body.lower() != '3d'):
            rigid_shapes = self.documentation_figures['rigid_body_shapes']
        else:
            rigid_shapes = self.rigid_body_shapes
        
        if 'rigid_body_complex_plane_0' in self.documentation_figures:
            complex_plane_figures = []
            i = 0
            while True:
                try:
                    complex_plane_figures.append(self.documentation_figures['rigid_body_complex_plane_{:}'.format(i)])
                    i += 1
                except KeyError:
                    break
        else:
            complex_plane_figures = None
        if 'rigid_body_residuals' in self.documentation_figures:
            residual_figure = self.documentation_figures['rigid_body_residuals']
        else:
            residual_figure = None
        
        print('Creating Test Parameters Section')
        test_parameters_string = (
                         'The data acquisition system was set to acquire {sample_rate:} '
                         'samples per second.  To compute spectra, the time signals were '
                         'split into measurement frames with {samples_per_frame:} samples '
                         'per measurement frame.  Measurement channels were split into {num_ref:} '
                         'references and {num_resp} responses.  {num_avg:} frames were acquired with '
                         '{overlap:}\\% overlap and averaged to compute frequency response functions via the {estimator:} '
                         'method.').format(sample_rate = self.sample_rate,
                                           samples_per_frame = self.num_samples_per_frame,
                                           num_ref = len(np.unique(self.frfs.reference_coordinate)),
                                           num_resp = len(np.unique(self.frfs.response_coordinate)),
                                           num_avg = self.num_averages,
                                           overlap = float(self.overlap)*100,
                                           estimator = self.frf_estimator,
                                           )
        if (self.window is not None and self.window.lower() != 'rectangle' and 
            self.window.lower() != 'boxcar' and self.window.lower() != 'none' and
            self.window.lower() != 'rectangular'):
            test_parameters_string += '  A {window:} window was applied to each frame.'
        if (self.trigger is not None and self.trigger.lower() != 'free run' and
            self.trigger != 'none'):
            test_parameters_string += (
                '  A trigger used to start the measurement of {frame:} frame. '
                'Channel {index:} was used to trigger the measurement with a {slope:} '
                'and at a level of {level:}% and a pretrigger of {pretrigger:}%.').format(
                    frame = 'every' if 'every' in self.trigger.lower() else 'first',
                    index = self.trigger_channel_index+1,
                    slope = self.trigger_slope.lower() + ' slope',
                    level = float(self.trigger_level)*100,
                    pretrigger = float(self.pretrigger)*100)
        if not one_file:
            with open(os.path.join(latex_root,'signal_processing.tex') if include_name_signal_processing is None else include_name_signal_processing,'w') as f:
                f.write(test_parameters_string)
        else:
            all_strings.append(test_parameters_string)
            
        print('Creating Rigid Body Analysis')
        rigid_body_string = create_rigid_body_analysis(
            self.geometry, rigid_shapes, complex_plane_figures, residual_figure, figure_label_rigid_body,
            complex_plane_figure_label, residual_figure_label, figure_caption_rigid_body, complex_plane_caption, residual_caption,
            graphics_options_rigid_body, complex_plane_graphics_options, residual_graphics_options, animate_graphics_options_rigid_body,
            figure_placement_rigid_body, complex_plane_figure_placement, residual_figure_placement, subfigure_options_rigid_body,
            subfigure_labels_rigid_body, subfigure_captions_rigid_body, complex_plane_subfigure_options, complex_plane_subfigure_labels,
            max_subfigures_per_page_rigid_body, max_subfigures_first_page_rigid_body, rigid_body_figure_save_names,
            complex_plane_figure_save_names, residual_figure_save_names, latex_root, figure_root,
            animation_style_rigid_body, shape_animation_frames, shape_animation_frame_rate, plot_shape_kwargs,
            rigid_body_check_kwargs,
            None if one_file else os.path.join(latex_root,'rigid_body.tex') if include_name_rigid_body is None else include_name_rigid_body
        )
        
        if one_file:
            all_strings.append(rigid_body_string)
        
        print('Creating Data Quality Summary')
        data_quality_string = create_data_quality_summary(
            reference_autospectra_figure, drive_point_frfs_figure, reciprocal_frfs_figure, frf_coherence_figure, coherence_figure,
            reference_autospectra_figure_label, reference_autospectra_figure_caption, reference_autospectra_graphics_options,
            reference_autospectra_figure_placement, reference_autospectra_subfigure_options, reference_autospectra_subfigure_labels,
            reference_autospectra_subfigure_captions, drive_point_frfs_figure_label, drive_point_frfs_figure_caption,
            drive_point_frfs_graphics_options, drive_point_frfs_figure_placement, drive_point_frfs_subfigure_options,
            drive_point_frfs_subfigure_labels, drive_point_frfs_subfigure_captions, reciprocal_frfs_figure_label,
            reciprocal_frfs_figure_caption, reciprocal_frfs_graphics_options, reciprocal_frfs_figure_placement,
            reciprocal_frfs_subfigure_options, reciprocal_frfs_subfigure_labels, reciprocal_frfs_subfigure_captions,
            frf_coherence_figure_label, frf_coherence_figure_caption, frf_coherence_graphics_options, frf_coherence_figure_placement,
            frf_coherence_subfigure_options, frf_coherence_subfigure_labels, frf_coherence_subfigure_captions, coherence_figure_label,
            coherence_figure_caption, coherence_graphics_options, coherence_figure_placement, coherence_subfigure_options,
            coherence_subfigure_labels, coherence_subfigure_captions, max_subfigures_per_page, max_subfigures_first_page,
            latex_root, figure_root,
            None if one_file else os.path.join(latex_root,'data_quality.tex') if include_name_data_quality is None else include_name_data_quality,
            reference_autospectra_figure_save_names,
            drive_point_frfs_figure_save_names, reciprocal_frfs_figure_save_names, frf_coherence_figure_save_names, coherence_figure_save_names
        )
        
        if one_file:
            all_strings.append(data_quality_string)

        if 'mac' in self.documentation_figures:
            mac_figure = self.documentation_figures['mac']
        else:
            mac_figure = None

        print('Creating Mode Fitting Summary')
        mode_fitting_string = create_mode_fitting_summary(
            self.fit_modes_information, self.fit_modes, fit_modes_table, fit_mode_table_kwargs, mac_figure, mac_plot_kwargs,
            self.frfs, self.resynthesized_frfs, resynthesis_comparison, resynthesis_figure, resynthesis_plot_kwargs,
            latex_root, figure_root, fit_mode_information_save_names, mac_plot_save_name,
            resynthesis_plot_save_name,
            None if one_file else os.path.join(latex_root,'mode_fitting.tex') if include_name_mode_fitting is None else include_name_mode_fitting,
            fit_modes_information_table_justification_string,
            fit_modes_information_table_longtable, fit_modes_information_table_header, fit_modes_information_table_horizontal_lines,
            fit_modes_information_table_placement, fit_modes_information_figure_graphics_options, fit_modes_information_figure_placement,
            fit_modes_table_justification_string, fit_modes_table_label, fit_modes_table_caption, fit_modes_table_longtable,
            fit_modes_table_header, fit_modes_table_horizontal_lines, fit_modes_table_placement, fit_modes_table_header_override,
            mac_plot_figure_label, mac_plot_figure_caption, mac_plot_graphics_options, mac_plot_figure_placement,
            resynthesis_plot_figure_label, resynthesis_plot_figure_caption, resynthesis_plot_graphics_options, resynthesis_plot_figure_placement
        )
        
        if one_file:
            all_strings.append(mode_fitting_string)

        if animation_style_mode_shape is None:
            animation_style_mode_shape = global_animation_style
        
        if 'mode_shapes' in self.documentation_figures and (animation_style_mode_shape is None or animation_style_mode_shape.lower() not in ['3d','one3d']):
            shapes = self.documentation_figures['mode_shapes']
        else:
            shapes = self.fit_modes

        print('Creating Mode Shape Figure')
        mode_shape_string = create_mode_shape_figures(
            self.geometry, shapes, figure_label_mode_shape, figure_caption_mode_shape, graphics_options_mode_shape,
            animate_graphics_options_mode_shape, figure_placement_mode_shape, subfigure_options_mode_shape, subfigure_labels_mode_shape,
            subfigure_captions_mode_shape, max_subfigures_per_page_mode_shape, max_subfigures_first_page_mode_shape,
            mode_shape_save_names, latex_root, figure_root, animation_style_mode_shape,
            shape_animation_frames, shape_animation_frame_rate, plot_shape_kwargs,
            None if one_file else os.path.join(latex_root,'mode_shape.tex') if include_name_mode_shape is None else include_name_mode_shape
        )
        
        if one_file:
            all_strings.append(mode_shape_string)
        
        
        if self.channel_table is not None:
            print('Creating the Channel Table')
            channel_table_string = latex_table(self.channel_table,table_label = 'tab:channel_table',
                                               table_caption = 'Channel Table', longtable=True, header=True)
            if not one_file:
                with open(os.path.join(latex_root,'channel_table.tex') if include_name_channel_table is None else include_name_channel_table,'w') as f:
                    f.write(channel_table_string)
            else:
                all_strings.append(channel_table_string)
                
        if one_file:
            final_string = '\n\n\n'.join(all_strings)
            if one_file is True:
                with open(os.path.join(latex_root,'document.tex'),'w') as f:
                    f.write(final_string)
            elif isinstance(one_file,str):
                with open(one_file,'w') as f:
                    f.write(final_string)
            return final_string

    
    def create_documentation_word(self):
        raise NotImplementedError('Not Implemented Yet')
    
    def create_documentation_pptx(self):
        raise NotImplementedError('Not Implemented Yet')
    
    