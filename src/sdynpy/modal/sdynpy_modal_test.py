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
from ..core.sdynpy_shape import ShapeArray, mac, matrix_plot, rigid_body_check
from ..core.sdynpy_data import (TransferFunctionArray, CoherenceArray,
                                MultipleCoherenceArray, PowerSpectralDensityArray)
from .sdynpy_signal_processing_gui import SignalProcessingGUI
from .sdynpy_polypy import PolyPy_GUI
from .sdynpy_smac import SMAC_GUI
from ..doc.sdynpy_latex import figure as latex_figure, table as latex_table
from qtpy.QtCore import Qt
import matplotlib.pyplot as plt

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
        # Figures to save to documentation
        self.documentation_figures = {}
        
    def set_rigid_body_shapes(self,rigid_body_shapes):
        self.rigid_body_shapes = rigid_body_shapes
    
    def set_units(self,response_unit, reference_unit):
        self.response_unit = response_unit
        self.reference_unit = reference_unit
        
    def set_geometry(self, geometry):
        self.geometry = geometry
    
    def set_time_histories(self, time_histories):
        self.time_histories = time_histories
    
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
            frf_estimator):
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
    
    def fit_modes_PolyPy(self):
        if self.frfs is None:
            raise ValueError('FRFs must be defined in order to fit modes')
        self.ppgui = PolyPy_GUI(self.frfs)
    
    def retrieve_modes_PolyPy(self):
        self.ppgui.compute_shapes()
        self.fit_modes = self.ppgui.shapes
        self.resynthesized_frfs = self.resynthesized_frfs
        
        self.fit_modes_information = {'info':[
            'Modes were fit to the data in {:} bands.'.format(len(self.ppgui.stability_diagrams))]}
        figure_index = 0
        # Now go through and get polynomial data from each frequency range
        for stability_diagram in self.ppgui.stability_diagrams:
            # First get data about the stabilization diagram
            min_frequency = stability_diagram.polypy.min_frequency
            max_frequency = stability_diagram.polypy.max_frequency
            min_order = np.min(stability_diagram.polypy.polynomial_orders)
            max_order = np.max(stability_diagram.polypy.polynomial_orders)
            num_selected_poles = len(stability_diagram.selected_poles)
            self.fit_modes_information['info'].append(
                ('The frequency band from {:0.2f} to {:0.2f} was analyzed with polynomials from order {:} to {:}.  '+
                 '{:} poles were selected from this band.  The stabilization diagram is shown in {{figure{:}ref:}}.').format(
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
        self.fit_modes_information['info'].append(
            ('Complex Modes' if self.ppgui.complex_modes_checkbox.isChecked() else 'Normal Modes') + ' were fit to the data.  ' + 
            ('Residuals' if self.ppgui.use_residuals_checkbox.isChecked() else 'No residuals') + ' were used when fitting mode shapes.  ' +
            ('All frequency lines were used to fit mode shapes.' if self.ppgui.all_frequency_lines_checkbox.isChecked() else
            'Mode shapes were fit using {:} frequency lines around each resonance, and {:} frequency lines were used to fit residuals.'.format(
                self.ppgui.lines_at_resonance_spinbox.value(),self.ppgui.lines_at_residuals_spinbox.value())))
    
    def fit_modes_SMAC(self):
        raise NotImplementedError('SMAC has not been implemented yet')
    
    def retrieve_modes_SMAC(self):
        raise NotImplementedError('SMAC has not been implemented yet')
    
    def edit_mode_comments(self):
        if self.fit_modes is None:
            raise ValueError('Modes have not yet been fit or assigned.')
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
        if measured:
            if self.frfs is None:
                raise ValueError('FRFs have not yet been computed or assigned')
            cmif = self.frfs.compute_cmif()
            cmif.plot(ax, plot_kwargs = measured_plot_kwargs)
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
                supicious_channels, *figures = rigid_body_check(self.geometry, self.rigid_body_shapes, 
                                                                return_figures = True, **rigid_body_checks_kwargs)
                num_figs = self.rigid_body_shapes.size+1
                figures = figures[-num_figs:]
                for i in range(num_figs-1):
                    self.documentation_figures['rigid_body_complex_plane_{:}'.format(i)] = figures[i]
                self.documentation_figures['rigid_body_residuals'] = figures[-1]
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
        if plot_frf_coherence:
            if self.frfs is None:
                print('Warning: Could not plot FRF coherence; frfs is undefined')
            elif self.coherence is None:
                print('Warning: Could not plot FRF coherence; coherence is undefined')
            else:
                pass

    def create_documentation_latex(self, mif_plots = ['cmif'], 
                                   latex_root = '', figure_folder = 'figures',
                                   document_filename = 'include.tex',
                                   appendix_filename = 'appendix.tex',
                                   frequency_format = '{:0.1f}',
                                   damping_format = '{:0.2f}\\%',
                                   comment1_header = 'Description',
                                   comment2_header = None,
                                   comment3_header = None,
                                   comment4_header = None,
                                   comment5_header = None
                                   
                                   ):
        raise NotImplementedError('Not Implemented Yet')
    
    def create_documentation_word(self):
        raise NotImplementedError('Not Implemented Yet')
    
    def create_documentation_pptx(self):
        raise NotImplementedError('Not Implemented Yet')
    
    