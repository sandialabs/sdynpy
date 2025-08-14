# -*- coding: utf-8 -*-
"""
Functions for creating a LaTeX report from SDynPy objects.

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
import os
import pyqtgraph as pqtg
import PIL
from ..signal_processing.sdynpy_correlation import mac, matrix_plot
from ..core.sdynpy_geometry import Geometry,GeometryPlotter, ShapePlotter
from ..core.sdynpy_coordinate import CoordinateArray,coordinate_array as sd_coordinate_array
from ..core.sdynpy_shape import ShapeArray, mac as shape_mac,rigid_body_check
from ..fileio.sdynpy_pdf3D import create_animated_modeshape_content,get_view_parameters_from_plotter
from shutil import copy
from qtpy.QtWidgets import QApplication
import pandas as pd
from io import BytesIO

try:
    from vtk import vtkU3DExporter
except ImportError:
    vtkU3DExporter = None

def create_latex_summary(figure_basename, geometry, shapes, frfs,
                         output_file=None, figure_basename_relative_to_latex=None,
                         max_shapes=None, max_frequency=None,
                         frequency_format='{:0.1f}', damping_format='{:0.2f}\\%',
                         cmif_kwargs={'part': 'imag', 'tracking': None},
                         cmif_subplots_kwargs={},
                         mac_subplots_kwargs={}, mac_plot_kwargs={},
                         geometry_plot_kwargs={},
                         shape_plot_kwargs={},
                         save_animation_kwargs={'frames': 20},
                         latex_cmif_graphics_options=r'width=0.7\linewidth',
                         latex_mac_graphics_options=r'width=0.5\linewidth',
                         latex_shape_graphics_options=r'width=\linewidth,loop',
                         latex_shape_subplot_options=r'[t]{0.45\linewidth}',
                         latex_max_figures_per_page=6,
                         latex_max_figures_first_page=None,
                         latex_cmif_caption='Complex Mode Indicator Function showing experimental data compared to modal fitting.',
                         latex_cmif_label='fig:cmif',
                         latex_mac_caption='Auto Modal Assurance Criterion Plot showing independence of fit mode shapes.',
                         latex_mac_label='fig:mac',
                         latex_shape_subcaption='Shape {number:} at {frequency:} Hz, {damping:}\\ damping',
                         latex_shape_sublabel='fig:shape{:}',
                         latex_shape_caption='Mode shapes extracted from test data.',
                         latex_shape_label='fig:modeshapes',
                         latex_shape_table_columns='lllp{3.5in}',
                         latex_shape_table_caption=(
                             'List of modes extracted from the test data.  Modal parameters are shown along with a brief description of the mode shape.'),
                         latex_shape_table_label='tab:modelist'):

    if figure_basename_relative_to_latex is None:
        figure_basename_relative_to_latex = figure_basename.replace('\\', '/')

    if latex_max_figures_first_page is None:
        latex_max_figures_first_page = latex_max_figures_per_page

    # Get the figure names
    figure_base_path, figure_base_filename = os.path.split(figure_basename)
    figure_base_filename, figure_base_ext = os.path.splitext(figure_base_filename)
    latex_figure_base_path, latex_figure_base_filename = os.path.split(
        figure_basename_relative_to_latex)
    latex_figure_base_filename, latex_figure_base_ext = os.path.splitext(latex_figure_base_filename)

    cmif_file_name = os.path.join(figure_base_path, figure_base_filename +
                                  '_cmif_comparison' + figure_base_ext)
    mac_file_name = os.path.join(figure_base_path, figure_base_filename + '_mac' + figure_base_ext)
    shape_file_name = os.path.join(
        figure_base_path, figure_base_filename + '_shape_{:}' + figure_base_ext)

    cmif_latex_file_name = (latex_figure_base_path + '/' +
                            figure_base_filename + '_cmif_comparison').replace('\\', '/')
    mac_latex_file_name = (latex_figure_base_path + '/' +
                           figure_base_filename + '_mac').replace('\\', '/')
    shape_latex_file_name = (latex_figure_base_path + '/' + figure_base_filename + '_shape_{:}-')

    # Go through and save out all the files
    experimental_cmif = None if frfs is None else frfs.compute_cmif(**cmif_kwargs)
    frequencies = None if experimental_cmif is None else experimental_cmif[0].abscissa

    analytic_frfs = None if (shapes is None or frfs is None) else shapes.compute_frf(frequencies, np.unique(frfs.coordinate[..., 0]),
                                                                                     np.unique(frfs.coordinate[..., 1]))
    analytic_cmif = analytic_frfs.compute_cmif(**cmif_kwargs)

    # Compute CMIF

    fig, ax = plt.subplots(num=figure_basename + ' CMIF', **cmif_subplots_kwargs)
    experimental_cmif[0].plot(ax, plot_kwargs={'color': 'b', 'linewidth': 1})
    analytic_cmif[0].plot(ax, plot_kwargs={'color': 'r', 'linewidth': 1})
    experimental_cmif[1:].plot(ax, plot_kwargs={'color': 'b', 'linewidth': 0.25})
    analytic_cmif[1:].plot(ax, plot_kwargs={'color': 'r', 'linewidth': 0.25})
    shapes.plot_frequency(experimental_cmif[0].abscissa, experimental_cmif[0].ordinate, ax)
    ax.legend(['Experiment', 'Fit'])
    ax.set_yscale('log')
    ax.set_ylim(experimental_cmif.min(abs) / 2, experimental_cmif.max(abs) * 2)
    ax.set_ylabel('CMIF (m/s^2/N)')
    ax.set_xlabel('Frequency (Hz)')
    fig.tight_layout()
    fig.savefig(cmif_file_name)

    # Compute MAC
    mac_matrix = mac(shapes.flatten().shape_matrix.T)
    fig, ax = plt.subplots(num=figure_basename + ' MAC')
    matrix_plot(mac_matrix, ax, **mac_plot_kwargs)
    fig.tight_layout()
    fig.savefig(mac_file_name)

    # Now go through and save the shapes
    plotter = geometry.plot_shape(shapes, plot_kwargs=geometry_plot_kwargs, **shape_plot_kwargs)
    plotter.save_animation_all_shapes(
        shape_file_name, individual_images=True, **save_animation_kwargs)

    # Go through and create the latex document
    output_string = ''

    # Add the CMIF plot
    output_string += r'''\begin{{figure}}
    \centering
    \includegraphics[{:}]{{{:}}}
    \caption{{{:}}}
    \label{{{:}}}
\end{{figure}}'''.format(latex_cmif_graphics_options,
                         cmif_latex_file_name,
                         latex_cmif_caption,
                         latex_cmif_label)

    output_string += r'''

\begin{{figure}}
    \centering
    \includegraphics[{:}]{{{:}}}
    \caption{{{:}}}
    \label{{{:}}}
\end{{figure}}'''.format(latex_mac_graphics_options,
                         mac_latex_file_name,
                         latex_mac_caption,
                         latex_mac_label)

    # Create a table of natural frequencies, damping values, and comments
    output_string += r'''

\begin{{table}}
    \centering
    \caption{{{:}}}
    \label{{{:}}}
    %\resizebox{{\linewidth}}{{!}}{{
    \begin{{tabular}}{{{:}}}
        Mode & Freq (Hz) & Damping & Description \\ \hline'''.format(
        latex_shape_table_caption, latex_shape_table_label, latex_shape_table_columns)
    for i, shape in enumerate(shapes.flatten()):
        output_string += r'''
        {:} & {:} & {:} & {:} \\'''.format(i + 1, frequency_format.format(shape.frequency),
                                           damping_format.format(shape.damping * 100), shape.comment1)
    output_string += r'''
    \end{tabular}
    %}
\end{table}'''

    # Now lets create the modeshape figure
    output_string += r'''
\begin{figure}[h]
    \centering'''
    for index, shape in enumerate(shapes.flatten()):
        if index == latex_max_figures_first_page or ((index - latex_max_figures_first_page) % latex_max_figures_per_page == 0 and index != 0):
            output_string += r'''
\end{figure}
\begin{figure}[h]
    \ContinuedFloat
    \centering'''
        output_string += r'''
    \begin{{subfigure}}{subfigure_options:}
        \centering
        \animategraphics[{graphics_options:}]{{{num_frames:}}}{{{base_name:}}}{{0}}{{{end_frame:}}}
        \caption{{{caption:}}}
        \label{{{label:}}}
    \end{{subfigure}}'''.format(graphics_options=latex_shape_graphics_options, num_frames=save_animation_kwargs['frames'],
                                base_name=shape_latex_file_name.format(index + 1), end_frame=save_animation_kwargs['frames'] - 1,
                                caption=latex_shape_subcaption.format(
            number=index + 1,
            frequency=frequency_format.format(shape.frequency),
            damping=damping_format.format(shape.damping * 100)),
            label=latex_shape_sublabel.format(index + 1),
            subfigure_options=latex_shape_subplot_options)
    output_string += r'''
    \caption{{{:}}}
    \label{{{:}}}
\end{{figure}}
'''.format(latex_shape_caption, latex_shape_label)
    if isinstance(output_file, str):
        close = True
        output_file = open(output_file, 'w')
    else:
        close = False
    try:
        output_file.write(output_string)
    except AttributeError:
        print('Error writing to output file {:}'.format(output_file))
    if close:
        output_file.close()
    return output_string

def create_geometry_overview(geometry, plot_kwargs = {}, 
                             coordinate_array = None, plot_coordinate_kwargs = {},
                             animation_style = '3d',
                             animation_frames = 200,
                             animation_frame_rate = 20,
                             geometry_figure_label = 'fig:geometry',
                             geometry_figure_caption = 'Geometry',
                             geometry_graphics_options = r'width=0.7\linewidth',
                             geometry_animate_graphics_options = r'width=0.7\linewidth,loop',
                             geometry_figure_placement = '[h]',
                             geometry_figure_save_name = None,
                             coordinate_figure_label = 'fig:coordinate',
                             coordinate_figure_caption = 'Local Coordinate Directions (Red: X+, Green: Y+, Blue: Z+)',
                             coordinate_graphics_options = r'width=0.7\linewidth',
                             coordinate_animate_graphics_options = r'width=0.7\linewidth,loop',
                             coordinate_figure_placement = '[h]',
                             coordinate_figure_save_name = None,
                             latex_root = r'',
                             figure_root = None,
                             include_name = None,
                             ):
    
    if geometry_figure_save_name is None:
        if figure_root is None:
            geometry_figure_save_name = os.path.join(latex_root,'geometry')
        else:
            geometry_figure_save_name = os.path.join(figure_root,'geometry')
    
    if coordinate_figure_save_name is None:
        if figure_root is None:
            coordinate_figure_save_name = os.path.join(latex_root,'coordinate')
        else:
            coordinate_figure_save_name = os.path.join(figure_root,'coordinate')
    
    plot_local_coords = False
    if isinstance(geometry,Geometry):
        geom_plotter = geometry.plot(**plot_kwargs,plot_individual_items=True)[0]
    elif isinstance(geometry,GeometryPlotter):
        geom_plotter = geometry
        geometry = None
    else:
        raise ValueError('`geometry` should be a `Geometry` or `GeometryPlotter` object')
    if len(plot_coordinate_kwargs) == 0 and len(plot_kwargs) > 0:
        plot_coordinate_kwargs['plot_kwargs'] = plot_kwargs
    if isinstance(coordinate_array,CoordinateArray):
        if geometry is None:
            raise ValueError('If `coordinate_array` is a `CoordinateArray` object, then `geometry` must be a `Geometry` object.')
        coord_plotter = geometry.plot_coordinate(coordinate_array,**plot_coordinate_kwargs,plot_individual_items=True)
    elif isinstance(coordinate_array,GeometryPlotter):
        coord_plotter = coordinate_array
        coordinate_array = None
    elif coordinate_array == 'local':
        if geometry is None:
            raise ValueError('If `coordinate_array` is local, then `geometry` must be a `Geometry` object.')
        css_to_plot = geometry.coordinate_system.id[[not np.allclose(matrix,np.eye(3)) for matrix in geometry.coordinate_system.matrix[...,:3,:3]]]
        nodes_to_plot = geometry.node.id[np.in1d(geometry.node.disp_cs,css_to_plot)]
        coordinate_array = sd_coordinate_array(nodes_to_plot,[1,2,3],force_broadcast=True)
        coord_plotter = geometry.plot_coordinate(coordinate_array,**plot_coordinate_kwargs,plot_individual_items=True)
        plot_local_coords = True
    elif coordinate_array is None:
        coord_plotter = None
    else:
        raise ValueError('`coordinate_array` should be a `CoordinateArray` or `GeometryPlotter` object or `None`')
    
    latex_string = [
"""To describe the data acquired in this activity, a geometry is constructed
consisting of the measurement positions and orientations, as well as lines and
elements to aid in visualization of the geometry.  The geometry for this
activity is shown in Figure \\ref{{{geometry_reference:}}}.""".format(
    geometry_reference = geometry_figure_label)]
    
    latex_string.append(figure([geom_plotter],geometry_figure_label,
                               geometry_figure_caption,
                               geometry_graphics_options,
                               geometry_animate_graphics_options,
                               geometry_figure_placement,
                               figure_save_names = [geometry_figure_save_name],
                               latex_root = latex_root,
                               animation_style = animation_style,
                               animation_frames = animation_frames,
                               animation_frame_rate = animation_frame_rate))
    
    if coord_plotter is not None:
    
        if plot_local_coords:
            latex_string.append(
"""To describe orientations of measurements, coordinate systems are used to define
local directions.  Figure \\ref{{{coordinate_reference:}}} shows the local
coordinate systems defined in the test.""".format(coordinate_reference = coordinate_figure_label)
            )
        
        latex_string.append(figure([coord_plotter],coordinate_figure_label,
                                   coordinate_figure_caption,
                                   coordinate_graphics_options,
                                   coordinate_animate_graphics_options,
                                   coordinate_figure_placement,
                                   figure_save_names = [coordinate_figure_save_name],
                                   latex_root = latex_root,
                                   animation_style = animation_style,
                                   animation_frames = animation_frames,
                                   animation_frame_rate = animation_frame_rate))
    
    if include_name is not None:
        with open(include_name,'w') as f:
            f.write('\n\n'.join(latex_string))
    
    return latex_string

def create_data_quality_summary(
        reference_autospectra_figure = None,
        drive_point_frfs_figure = None,
        reciprocal_frfs_figure = None,
        frf_coherence_figure = None,
        coherence_figure = None,
        reference_autospectra_figure_label = 'fig:reference_autospectra',
        reference_autospectra_figure_caption = 'Autospectra of the reference channels',
        reference_autospectra_graphics_options = r'width=0.7\linewidth',
        reference_autospectra_figure_placement = '',
        reference_autospectra_subfigure_options = r'[t]{0.45\linewidth}',
        reference_autospectra_subfigure_labels = None,
        reference_autospectra_subfigure_captions = None,
        drive_point_frfs_figure_label = 'fig:drive_point_frf',
        drive_point_frfs_figure_caption = 'Drive point frequency response functions',
        drive_point_frfs_graphics_options = r'width=0.7\linewidth',
        drive_point_frfs_figure_placement = '',
        drive_point_frfs_subfigure_options = r'[t]{0.45\linewidth}',
        drive_point_frfs_subfigure_labels = None,
        drive_point_frfs_subfigure_captions = None,
        reciprocal_frfs_figure_label = 'fig:reciprocal_frfs',
        reciprocal_frfs_figure_caption = 'Reciprocal frequency response functions.',
        reciprocal_frfs_graphics_options = r'width=0.7\linewidth',
        reciprocal_frfs_figure_placement = '',
        reciprocal_frfs_subfigure_options = r'[t]{0.45\linewidth}',
        reciprocal_frfs_subfigure_labels = None,
        reciprocal_frfs_subfigure_captions = None,
        frf_coherence_figure_label = 'fig:frf_coherence',
        frf_coherence_figure_caption = 'Drive point frequency response functions with coherence overlaid',
        frf_coherence_graphics_options = r'width=0.7\linewidth',
        frf_coherence_figure_placement = '',
        frf_coherence_subfigure_options = r'[t]{0.45\linewidth}',
        frf_coherence_subfigure_labels = None,
        frf_coherence_subfigure_captions = None,
        coherence_figure_label = 'fig:coherence',
        coherence_figure_caption = 'Coherence of all channels in the test.',
        coherence_graphics_options = r'width=0.7\linewidth',
        coherence_figure_placement = '',
        coherence_subfigure_options = r'[t]{0.45\linewidth}',
        coherence_subfigure_labels = None,
        coherence_subfigure_captions = None,
        max_subfigures_per_page = None,
        max_subfigures_first_page = None,
        latex_root = r'',
        figure_root = None,
        include_name = None,
        reference_autospectra_figure_save_names = None,
        drive_point_frfs_figure_save_names = None,
        reciprocal_frfs_figure_save_names = None,
        frf_coherence_figure_save_names = None,
        coherence_figure_save_names = None,
        ):
    
    latex_string = []
    
    if reference_autospectra_figure is not None:
        latex_string.append(
    f'Figure \\ref{{{reference_autospectra_figure_label:}}} shows the autospectra of the '
    'reference channels in the test.  ')
        
        if reference_autospectra_figure_save_names is None:
            if figure_root is None:
                reference_autospectra_figure_save_names = os.path.join(latex_root,'reference_autospectra_{:}')
            else:
                reference_autospectra_figure_save_names = os.path.join(figure_root,'reference_autospectra_{:}')
        
        latex_string.append(figure(
            figures = [reference_autospectra_figure],
            figure_label = reference_autospectra_figure_label,
            figure_caption = reference_autospectra_figure_caption,
            graphics_options = reference_autospectra_graphics_options,
            figure_placement = reference_autospectra_figure_placement,
            subfigure_options = reference_autospectra_subfigure_options,
            subfigure_labels = reference_autospectra_subfigure_labels,
            subfigure_captions = reference_autospectra_subfigure_captions,
            max_subfigures_per_page = max_subfigures_per_page,
            max_subfigures_first_page = max_subfigures_first_page,
            figure_save_names = reference_autospectra_figure_save_names,
            latex_root = latex_root))
        
    if drive_point_frfs_figure is not None:
        latex_string.append(
    f'Figure \\ref{{{drive_point_frfs_figure_label:}}} shows the imaginary part '
    'of the drive point frequency response functions.')
        
        if drive_point_frfs_figure_save_names is None:
            if figure_root is None:
                drive_point_frfs_figure_save_names = os.path.join(latex_root,'drive_point_frf_{:}')
            else:
                drive_point_frfs_figure_save_names = os.path.join(figure_root,'drive_point_frf_{:}')
        
        latex_string.append(figure(
            figures = [drive_point_frfs_figure],
            figure_label = drive_point_frfs_figure_label,
            figure_caption = drive_point_frfs_figure_caption,
            graphics_options = drive_point_frfs_graphics_options,
            figure_placement = drive_point_frfs_figure_placement,
            subfigure_options = drive_point_frfs_subfigure_options,
            subfigure_labels = drive_point_frfs_subfigure_labels,
            subfigure_captions = drive_point_frfs_subfigure_captions,
            max_subfigures_per_page = max_subfigures_per_page,
            max_subfigures_first_page = max_subfigures_first_page,
            figure_save_names = drive_point_frfs_figure_save_names,
            latex_root = latex_root))
    
    if reciprocal_frfs_figure is not None:
        latex_string.append(
    f'Figure \\ref{{{reciprocal_frfs_figure_label:}}} shows the reciprocal frequency response functions in the test.')
        
        if reciprocal_frfs_figure_save_names is None:
            if figure_root is None:
                reciprocal_frfs_figure_save_names = os.path.join(latex_root,'reciprocal_frfs_{:}')
            else:
                reciprocal_frfs_figure_save_names = os.path.join(figure_root,'reciprocal_frfs_{:}')
        
        latex_string.append(figure(
            figures = [reciprocal_frfs_figure],
            figure_label = reciprocal_frfs_figure_label,
            figure_caption = reciprocal_frfs_figure_caption,
            graphics_options = reciprocal_frfs_graphics_options,
            figure_placement = reciprocal_frfs_figure_placement,
            subfigure_options = reciprocal_frfs_subfigure_options,
            subfigure_labels = reciprocal_frfs_subfigure_labels,
            subfigure_captions = reciprocal_frfs_subfigure_captions,
            max_subfigures_per_page = max_subfigures_per_page,
            max_subfigures_first_page = max_subfigures_first_page,
            figure_save_names = reciprocal_frfs_figure_save_names,
            latex_root = latex_root))

    if frf_coherence_figure is not None:
        latex_string.append(
    f'Figure \\ref{{{frf_coherence_figure_label:}}} shows the coherence overlaying the drive point frequency response functions.')
        
        if frf_coherence_figure_save_names is None:
            if figure_root is None:
                frf_coherence_figure_save_names = os.path.join(latex_root,'frf_coherence_{:}')
            else:
                frf_coherence_figure_save_names = os.path.join(figure_root,'frf_coherence_{:}')
        
        latex_string.append(figure(
            figures = [frf_coherence_figure],
            figure_label = frf_coherence_figure_label,
            figure_caption = frf_coherence_figure_caption,
            graphics_options = frf_coherence_graphics_options,
            figure_placement = frf_coherence_figure_placement,
            subfigure_options = frf_coherence_subfigure_options,
            subfigure_labels = frf_coherence_subfigure_labels,
            subfigure_captions = frf_coherence_subfigure_captions,
            max_subfigures_per_page = max_subfigures_per_page,
            max_subfigures_first_page = max_subfigures_first_page,
            figure_save_names = frf_coherence_figure_save_names,
            latex_root = latex_root))
        
    if coherence_figure is not None:
        latex_string.append(
    f'Figure \\ref{{{coherence_figure_label:}}} shows the coherence of the '
    'response channels in the test.  ')
        
        if coherence_figure_save_names is None:
            if figure_root is None:
                coherence_figure_save_names = os.path.join(latex_root,'coherence_{:}')
            else:
                coherence_figure_save_names = os.path.join(figure_root,'coherence_{:}')
        
        latex_string.append(figure(
            figures = [coherence_figure],
            figure_label = coherence_figure_label,
            figure_caption = coherence_figure_caption,
            graphics_options = coherence_graphics_options,
            figure_placement = coherence_figure_placement,
            subfigure_options = coherence_subfigure_options,
            subfigure_labels = coherence_subfigure_labels,
            subfigure_captions = coherence_subfigure_captions,
            max_subfigures_per_page = max_subfigures_per_page,
            max_subfigures_first_page = max_subfigures_first_page,
            figure_save_names = coherence_figure_save_names,
            latex_root = latex_root))
        
    if include_name is not None:
        with open(include_name,'w') as f:
            f.write('\n\n'.join(latex_string))
            
    return latex_string

def create_mode_fitting_summary(
        # General information from the curve fitter
        fit_modes_information = None,
        # Information about the modes
        fit_modes = None,
        fit_modes_table = None,
        fit_mode_table_kwargs = {},
        mac_figure = None,
        mac_plot_kwargs = None,
        # Information to create resynthesis plots
        experimental_frfs = None,
        resynthesized_frfs = None,
        resynthesis_comparison = 'cmif',
        resynthesis_figure = None,
        resynthesis_plot_kwargs = None,
        # Path options
        latex_root = r'',
        figure_root = None,
        fit_mode_information_save_names = None,
        mac_plot_save_name = None,
        resynthesis_plot_save_name = None,
        include_name = None,
        # Latex formatting information
        fit_modes_information_table_justification_string = None,
        fit_modes_information_table_longtable = True,
        fit_modes_information_table_header = True,
        fit_modes_information_table_horizontal_lines = False,
        fit_modes_information_table_placement = '',
        fit_modes_information_figure_graphics_options = r'width=0.7\linewidth',
        fit_modes_information_figure_placement = '',
        fit_modes_table_justification_string = None,
        fit_modes_table_label = 'tab:mode_fits',
        fit_modes_table_caption = 'Modal parameters fit to the test data.',
        fit_modes_table_longtable = True,
        fit_modes_table_header = True,
        fit_modes_table_horizontal_lines = False,
        fit_modes_table_placement = '',
        fit_modes_table_header_override = None,
        mac_plot_figure_label = 'fig:mac',
        mac_plot_figure_caption = 'Modal Assurance Criterion Matrix from Fit Modes',
        mac_plot_graphics_options = r'width=0.7\linewidth',
        mac_plot_figure_placement = '',
        resynthesis_plot_figure_label = 'fig:resynthesis',
        resynthesis_plot_figure_caption = 'Test data compared to equivalent data computed from modal fits.',
        resynthesis_plot_graphics_options = r'width=0.7\linewidth',
        resynthesis_plot_figure_placement = '',
            ):
    latex_string = []
    if not fit_modes_information is None:
        figure_keys = [key for key in fit_modes_information.keys() if (key[:6].lower() == 'figure') and (key[-7:].lower() != 'caption')]
        table_keys = [key for key in fit_modes_information.keys() if key[:5].lower() == 'table' and key[-7:].lower() != 'caption']
        fig_format_kwargs = {key+'ref':'\\ref{{fig:mode_fitting_{:}}}'.format(i) for i,key in enumerate(figure_keys)}
        table_format_kwargs = {key+'ref':'\\ref{{tab:mode_fitting_{:}}}'.format(i) for i,key in enumerate(table_keys)}
        text = '\n\n'.join([v for v in fit_modes_information['text']]).format(**fig_format_kwargs, **table_format_kwargs)
        latex_string.append(text)
        for i,key in enumerate(figure_keys):
            reference = 'fig:mode_fitting_{:}'.format(i)
            if isinstance(fit_modes_information_figure_graphics_options,dict):
                graphics_options = fit_modes_information_figure_graphics_options[key]
            else:
                graphics_options = fit_modes_information_figure_graphics_options
            if fit_mode_information_save_names is None:
                if figure_root is None:
                    save_name = os.path.join(latex_root,'fit_mode_figure_{:}'.format(i))
                else:
                    save_name = os.path.join(figure_root,'fit_mode_figure_{:}'.format(i))
            else:
                save_name = fit_mode_information_save_names.format(i)
            fig = figure([fit_modes_information[key]],
                         reference,
                         fit_modes_information[key+'caption'],
                         graphics_options,
                         figure_save_names=[save_name],
                         latex_root = latex_root,
                         )
            latex_string.append(fig)
    
    if fit_modes is not None:
        latex_string.append(f'Table \\ref{{{fit_modes_table_label:}}} shows the modal parameters fit to the test data.')
        
        if fit_modes_table is None:
            fit_modes_table = fit_modes.mode_table('pandas')
        
        if fit_modes_table_header_override is not None:
            if isinstance(fit_modes_table,pd.DataFrame):
                fit_modes_table = fit_modes_table.rename(columns=fit_modes_table_header_override)
            else:
                fit_modes_table[0] = [val if not val in fit_modes_table_header_override else 
                                      fit_modes_table_header_override[val] for val in fit_modes_table[0]]
        
        latex_string.append(table(fit_modes_table,
                                  fit_modes_table_justification_string,
                                  fit_modes_table_label,
                                  fit_modes_table_caption,
                                  fit_modes_table_longtable,
                                  fit_modes_table_header,
                                  fit_modes_table_horizontal_lines,
                                  fit_modes_table_placement))
    
    if (mac_figure is None) and (fit_modes is not None):
        # Create the MAC from fit modes
        mac = shape_mac(fit_modes)
        if mac_plot_kwargs is None:
            mac_plot_kwargs = {}
        ax = matrix_plot(mac,**mac_plot_kwargs)
        mac_figure = ax.figure
    
    if mac_figure is not None:
        latex_string.append(
f'Figure \\ref{{{mac_plot_figure_label:}}} shows the Modal Assurance Criterion Matrix,'
'which is a measure of how similar each mode shape looks to all the other mode shapes.')
        
        if mac_plot_save_name is None:
            if figure_root is None:
                mac_plot_save_name = os.path.join(latex_root,'mac')
            else:
                mac_plot_save_name = os.path.join(figure_root,'mac')
        
        latex_string.append(figure([mac_figure],
                                   mac_plot_figure_label,
                                   'Modal Assurance Criterion Matrix of the fit mode shapes.',
                                   mac_plot_graphics_options,
                                   figure_placement = mac_plot_figure_placement,
                                   figure_save_names = [mac_plot_save_name],
                                   latex_root = latex_root))
    
    if (resynthesis_figure is None) and (experimental_frfs is not None) and (resynthesized_frfs is not None):
        if resynthesis_plot_kwargs is None:
            resynthesis_plot_kwargs = {}
        max_abscissa = np.min([np.max(experimental_frfs.abscissa),
                               np.max(resynthesized_frfs.abscissa)])
        min_abscissa = np.max([np.min(experimental_frfs.abscissa),
                               np.min(resynthesized_frfs.abscissa)])
        abscissa_range = max_abscissa - min_abscissa
        max_abscissa += abscissa_range/20
        min_abscissa -= abscissa_range/20
        if resynthesis_comparison == 'cmif':
            ds1 = experimental_frfs.compute_cmif()
            ds2 = resynthesized_frfs.compute_cmif()
            resynthesis_figure, ax = plt.subplots()
            kwargs1 = resynthesis_plot_kwargs.copy()
            kwargs2 = resynthesis_plot_kwargs.copy()
            kwargs1['color'] = 'b'
            kwargs2['color'] = 'r'
            ds1.plot(ax, plot_kwargs = kwargs1)
            ds2.plot(ax, plot_kwargs = kwargs2)
            ax.set_yscale('log')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('CMIF')
            ax.set_xlim([min_abscissa,max_abscissa])
        elif resynthesis_comparison == 'qmif':
            ds1 = experimental_frfs.compute_cmif(part='imag')
            ds2 = resynthesized_frfs.compute_cmif(part='imag')
            resynthesis_figure, ax = plt.subplots()
            kwargs1 = resynthesis_plot_kwargs.copy()
            kwargs2 = resynthesis_plot_kwargs.copy()
            kwargs1['color'] = 'b'
            kwargs2['color'] = 'r'
            ds1.plot(ax, plot_kwargs = kwargs1)
            ds2.plot(ax, plot_kwargs = kwargs2)
            ax.set_yscale('log')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('QMIF')
            ax.set_xlim([min_abscissa,max_abscissa])
        elif resynthesis_comparison == 'mmif':
            ds1 = experimental_frfs.compute_mmif()
            ds2 = resynthesized_frfs.compute_mmif()
            resynthesis_figure, ax = plt.subplots()
            kwargs1 = resynthesis_plot_kwargs.copy()
            kwargs2 = resynthesis_plot_kwargs.copy()
            kwargs1['color'] = 'b'
            kwargs2['color'] = 'r'
            ds1.plot(ax, plot_kwargs = kwargs1)
            ds2.plot(ax, plot_kwargs = kwargs2)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('MMIF')
            ax.set_xlim([min_abscissa,max_abscissa])
        elif resynthesis_comparison == 'nmif':
            ds1 = experimental_frfs.compute_nmif()
            ds2 = resynthesized_frfs.compute_nmif()
            resynthesis_figure, ax = plt.subplots()
            kwargs1 = resynthesis_plot_kwargs.copy()
            kwargs2 = resynthesis_plot_kwargs.copy()
            kwargs1['color'] = 'b'
            kwargs2['color'] = 'r'
            ds1.plot(ax, plot_kwargs = kwargs1)
            ds2.plot(ax, plot_kwargs = kwargs2)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('MMIF')
            ax.set_xlim([min_abscissa,max_abscissa])
        elif resynthesis_comparison == 'frf':
            ds1 = experimental_frfs
            ds2 = resynthesized_frfs
            kwargs1 = resynthesis_plot_kwargs.copy()
            kwargs2 = resynthesis_plot_kwargs.copy()
            kwargs1['color'] = 'b'
            kwargs2['color'] = 'r'
            ax = ds1.plot(False, plot_kwargs = kwargs1)
            for a,frf in zip(ax.flaten(),ds2[ds1.coordinate].flatten()):
                frf.plot(a, plot_kwargs = kwargs2)
                a.set_xlim([min_abscissa,max_abscissa])
            resynthesis_figure = ax.flatten()[0].figure
    
    if resynthesis_figure is not None:
        latex_string.append(
f'To judge the adequacy of the fit modes, Figure \\ref{{{resynthesis_plot_figure_label:}}} shows the data resynthesized from the '
'fit modes compared to the equivalent experimental data.')
        
        if resynthesis_plot_save_name is None:
            if figure_root is None:
                resynthesis_plot_save_name = os.path.join(latex_root,'resynthesis')
            else:
                resynthesis_plot_save_name = os.path.join(figure_root,'resynthesis')
        
        latex_string.append(figure([resynthesis_figure],
                                   resynthesis_plot_figure_label,
                                   'Experimental data compared to that resynthesized from the fit modes.',
                                   resynthesis_plot_graphics_options,
                                   figure_placement = resynthesis_plot_figure_placement,
                                   figure_save_names = [resynthesis_plot_save_name],
                                   latex_root = latex_root))
    
    if include_name is not None:
        with open(include_name,'w') as f:
            f.write('\n\n'.join(latex_string))
    
    return latex_string

def create_mode_shape_figures(
        geometry, shapes, figure_label = 'fig:modeshapes',
        figure_caption = 'Mode shapes extracted from test data.',
        graphics_options = r'width=0.7\linewidth', 
        animate_graphics_options = r'width=0.7\linewidth,loop',
        figure_placement = '',
        subfigure_options = r'[t]{0.45\linewidth}', subfigure_labels = None, 
        subfigure_captions = None, max_subfigures_per_page = None,
        max_subfigures_first_page = None, figure_save_names = None,
        latex_root = r'',
        figure_root = None,
        animation_style = None,
        animation_frames = 20,
        animation_frame_rate = 20,
        geometry_plot_shape_kwargs = {},
        include_name = None):
    latex_string = ['Figure \\ref{{{:}}} shows the mode shapes extracted from the test.'.format(figure_label)]
    
    if subfigure_captions is None:
        if isinstance(shapes,ShapePlotter):
            subfigure_captions = ['Mode {:} at {:0.2f} Hz with {:0.2f}\\% damping'.format(
                i+1,mode.frequency,mode.damping*100) for i,mode in enumerate(shapes.shapes)]
        else:
            subfigure_captions = ['Mode {:} at {:0.2f} Hz with {:0.2f}\\% damping'.format(
                i+1,mode.frequency,mode.damping*100) for i,mode in enumerate(shapes)]
    
    if figure_save_names is None:
        if figure_root is None:
            figure_save_names = os.path.join(latex_root, 'modeshape_{:}')
        else:
            figure_save_names = os.path.join(figure_root, 'modeshape_{:}')
    
    if animation_style == 'one3d':
        animation_style = '3d'
        shapes = [shapes]
    
    latex_string.append(
        figure(shapes,figure_label,figure_caption,graphics_options,
               animate_graphics_options, figure_placement, subfigure_options,
               subfigure_labels, subfigure_captions, max_subfigures_per_page,
               max_subfigures_first_page,
               figure_save_names, latex_root, animation_style, animation_frames,
               animation_frame_rate, geometry, geometry_plot_shape_kwargs))
    
    if include_name is not None:
        with open(include_name,'w') as f:
            f.write('\n\n'.join(latex_string))
    
    return latex_string

def create_rigid_body_analysis(
        geometry, rigid_shapes,
        complex_plane_figures = None,
        residual_figure = None,
        figure_label = 'fig:rigid_shapes',
        complex_plane_figure_label = 'fig:complex_plane',
        residual_figure_label = 'fig:rigid_shape_residual',
        figure_caption = 'Rigid body shapes extracted from test data.',
        complex_plane_caption = 'Complex Plane of the extracted shapes.',
        residual_caption = 'Rigid body residual showing non-rigid portions of the shapes.',
        graphics_options = r'width=0.7\linewidth', 
        complex_plane_graphics_options = r'width=0.7\linewidth', 
        residual_graphics_options = r'width=0.7\linewidth', 
        animate_graphics_options = r'width=0.7\linewidth,loop',
        figure_placement = '',
        complex_plane_figure_placement = '',
        residual_figure_placement = '',
        subfigure_options = r'[t]{0.45\linewidth}', subfigure_labels = None, 
        subfigure_captions = None, 
        complex_plane_subfigure_options = r'[t]{0.45\linewidth}',
        complex_plane_subfigure_labels = None, 
        max_subfigures_per_page = None,
        max_subfigures_first_page = None,
        figure_save_names = None,
        complex_plane_figure_save_names = None,
        residual_figure_save_names = None,
        latex_root = r'',
        figure_root = None,
        animation_style = None,
        animation_frames = 20,
        animation_frame_rate = 20,
        geometry_plot_shape_kwargs = {},
        rigid_body_check_kwargs = {},
        include_name = None
        ):
    
    latex_string = [
        f'Figure \\ref{{{figure_label:}}} shows the rigid shapes extracted from the '
        'rigid body analysis.  This analysis is performed to ensure all '
        'sensors are installed and documented correctly in the channel table. '
        'If a sensor had the wrong sensitivity, had its polarity flipped, or '
        'if cables were plugged in incorrectly, that sensor would not be '
        'moving rigidly with the rest of the test article.']
    
    if subfigure_captions is None:
        if isinstance(rigid_shapes,ShapePlotter):
            subfigure_captions = ['Rigid body shape {:}'.format(
                i+1) for i,mode in enumerate(rigid_shapes.shapes)]
        else:
            subfigure_captions = ['Rigid body shape {:}'.format(
                i+1) for i,mode in enumerate(rigid_shapes)]
    
    if figure_save_names is None:
        if figure_root is None:
            figure_save_names = os.path.join(latex_root, 'rigid_shape_{:}')
        else:
            figure_save_names = os.path.join(figure_root, 'rigid_shape_{:}')

    latex_string.append(
        figure(rigid_shapes,figure_label,figure_caption,graphics_options,
               animate_graphics_options, figure_placement, subfigure_options,
               subfigure_labels, subfigure_captions, max_subfigures_per_page,
               max_subfigures_first_page,
               figure_save_names, latex_root, animation_style, animation_frames,
               animation_frame_rate, geometry, geometry_plot_shape_kwargs))
    
    if complex_plane_figure_save_names is None:
        if figure_root is None:
            complex_plane_figure_save_names = os.path.join(latex_root, 'rigid_complex_plane_{:}')
        else:
            complex_plane_figure_save_names = os.path.join(figure_root, 'rigid_complex_plane_{:}')
    
    if residual_figure_save_names is None:
        if figure_root is None:
            residual_figure_save_names = os.path.join(latex_root, 'rigid_residual')
        else:
            residual_figure_save_names = os.path.join(figure_root, 'rigid_residual')
    
    if complex_plane_figures is None or residual_figure is None:
        rigid_body_kwargs = rigid_body_check_kwargs.copy()
        rigid_body_kwargs['return_figures'] = True
        out = rigid_body_check(geometry, rigid_shapes, **rigid_body_kwargs)
        if complex_plane_figures is None:
            complex_plane_figures = out[-(len(rigid_shapes)+1):-1]
        if residual_figure is None:
            residual_figure = out[-1]
    
    latex_string.append((
        'A more quantitiative analysis of the rigid body shapes is shown in '
        f'Figure \\ref{{{complex_plane_figure_label:}}} and \\ref{{{residual_figure_label:}}}. '
        f'Figure \\ref{{{complex_plane_figure_label:}}} shows the complex plane '
        'of the shape, which should look like a line through the origin.  '
        f'Figure \\ref{{{residual_figure_label:}}} shows the shape residuals, '
        'which are the remaining shape coefficient when the rigid portion of '
        'the motion is subtracted away.  The residual is then the remaining '
        'non-rigid portion of the motion, so large residual suggest issues with '
        'that channel.'))
    
    latex_string.append(figure(
        list(complex_plane_figures),complex_plane_figure_label,
        complex_plane_caption, complex_plane_graphics_options,
        animate_graphics_options, complex_plane_figure_placement,
        complex_plane_subfigure_options,
        complex_plane_subfigure_labels,
        rigid_shapes.shapes.comment1 if isinstance(rigid_shapes,ShapePlotter) else rigid_shapes.comment1,
        max_subfigures_per_page,
        max_subfigures_first_page,
        complex_plane_figure_save_names, latex_root))
    
    latex_string.append(figure(
        [residual_figure],residual_figure_label,
        residual_caption, residual_graphics_options,
        animate_graphics_options, residual_figure_placement,
        figure_save_names = residual_figure_save_names,
        latex_root = latex_root))
    
    if include_name is not None:
        with open(include_name,'w') as f:
            f.write('\n\n'.join(latex_string))
            
    return latex_string

def figure(figures, figure_label = None, figure_caption = None,
           graphics_options = r'width=0.7\linewidth', 
           animate_graphics_options = r'width=0.7\linewidth,loop',
           figure_placement = '',
           subfigure_options = r'[t]{0.45\linewidth}', subfigure_labels = None, 
           subfigure_captions = None, max_subfigures_per_page = None,
           max_subfigures_first_page = None, figure_save_names = None,
           latex_root = r'',
           animation_style = None,
           animation_frames = 20,
           animation_frame_rate = 20,
           geometry = None,
           geometry_plot_shape_kwargs = {}
           ):
    r"""
    Adds figures, subfigures, and animations to a running latex document.

    Parameters
    ----------
    figures : list
        Figure or figures that can be inserted into a latex document.  See 
        note for various figure types and configurations that can be used.
    figure_label : str, optional
        The label that will be used for the figure in the latex document.
        If not specified, the figure will not be labeled.
    figure_caption : str, optional
        The caption that will be used for the figure in the latex document.
        If not specified, the figure will only be captioned with the figure
        number.
    graphics_options : str, optional
        Graphics options that will be used for the figure.  If not specified
        this will be r'width=0.7\linewidth'.
    animate_graphics_options : str, optional
        Graphics options that will be used for an animation.  If not specified
        this will be r'width=0.7\linewidth,loop'.
    figure_placement : str, optional
        Specify the placement of the figure with strings such as '[t]', '[b]',
        or '[h]'.  If not specified, the figure will have no placement
        specified.
    subfigure_options : str, optional
        The options that will be applied to each subfigure in the figure, if
        subfigures are specified.  By default, this will be r'[t]{0.45\linewidth}'
    subfigure_labels : str, optional
        Labels to apply to the subfigure.  This can either be a list of strings
        the same size as the list of figures, or a string with a format specifier
        accepting the subfigure index.  If not specified, the subfigures will
        not be labeled.
    subfigure_captions : list, optional
        A list of strings the same length as the list of figures to use as
        captions for the subfigures.  If not specified, the subfigures will
        only be captioned with the subfigure number.
    max_subfigures_per_page : int, optional
        The maximum number of subfigures on a page.  Longer figures will be
        broken up into multiple figures using \ContinuedFloat.  If not specified,
        a single figure environment will be generated.
    max_subfigures_first_page : int, optional
        The maximum number of subfigures on the first page.  Longer figures will be
        broken up into multiple figures using \ContinuedFloat.  If not specified,
        the max_subfigures_per_page value will be used if specified, otherwise
        a single figure environment will be generated.
    figure_save_names : str or list of str, optional
        File names to save the figures as.  This can be specified as a string
        with a format specifier in it that will accept the figure index, or
        a list of strings the same length as the list of figures.
        If not specified, files will be specified as 'figure_0',
        'figure_1', etc.  If file names are not present, then the file name
        will be automatically selected for the type of figure given.
    latex_root : str, optional
        Directory in which the latex .tex file will be constructed.  This is
        used to create relative paths to the save_figure_names within the latex
        document.  If not specified, then the current directory will be assumed.
    animation_style : str, optional
        If a GeometryPlotter or ShapePlotter object is passed, this argument
        will determine what is saved from it.  To save just a screen shot of
        the plotter, use `animation_style = None` or `animation_style = 'none'.
        To save an animated 2D figure, use `animation_style= '2d'`.  To save
        an animated 3D figure, use `animation_style = '3d'`.  If not specified,
        a screenshot will be saved.
    animation_frames : int
        If a GeometryPlotter or ShapePlotter object is passed with a 2D
        `animation_style`, this argument will determine how many frames are
        rendered.
    animation_frame_rate : int
        This is the frame rate used in the animation.
    geometry : Geometry, optional
        If a ShapeArray is passed as a figure type, then a geometry must also
        be specified to define how the shape should be plotted.
    geometry_plot_shape_kwargs : dict, optional
        If a ShapeArray and Geometry are passed, then this is a dictionary of
        keyword arguments into the Geometry.plot_shape function in
        sdynpy.pdf3D.
        
     
    Returns
    -------
    latex_string : str
        The latex source code to insert the figures into the document.

    Notes
    -----
    The `figures` argument must be a list of figures.  If only one entry
    is present in the list, a figure will be made in the latex document.  If
    multiple entries are present, a figure will be made and subfigures will be
    made for each entry in the list.  If an entry in the list is also a list,
    then that figure or subfigure will be made into an animation.
    
    The list of figures can contain many types of objects that a figure will be
    made from, including:
        - A 2D numpy array
        - A Matplotlib figure
        - A pyqtgraph plotitem
        - A bytes object that represents an image
        - A string to a file name
        - A GeometryPlotter containing a geometry
        - A ShapePlotter containing a mode shape
        - A ShapeArray object containing a mode
    """
    
    
    if ((isinstance(figures,ShapePlotter) and len(figures.shapes) == 1) or 
        (not isinstance(figures,ShapePlotter) and len(figures) == 1)):
        subfigures = False
    else:
        subfigures = True
    
    # If it's a shapeplotter, we need to break it out into the multiple shapes
    # but keep track of the iterable
    if isinstance(figures,ShapePlotter):
        shapeplotter = figures
        figures = [i for i in shapeplotter.shapes]
    else:
        shapeplotter = None
        
    if max_subfigures_first_page is None:
        max_subfigures_first_page = max_subfigures_per_page
        
    if figure_save_names is None:
        figure_save_names = ['figure_{:}'.format(i) for i in range(len(figures))]
    elif isinstance(figure_save_names,str):
        figure_save_names = [figure_save_names.format(i) for i in range(len(figures))]
        
    latex_string = r'\begin{figure}'+figure_placement+'\n    \\centering'
    # Go through and save all of the files out to disk
    for i,figure in enumerate(figures):
        # Check the type of figure.  If it's a list of figures, then it's an
        # animation.
        if isinstance(figure, list):
            animate = True
            num_frames = len(list)
        # Otherwise it's just a figure, but we turn it into a list anyways to
        # make it so we only have to program this once.
        else:
            animate = False
            num_frames = 1
            figure = [figure]
        # We need to get the extension of the file name to figure out what
        # type of file to save the image to.
        base,ext = os.path.splitext(figure_save_names[i])
        # We also want to get the directory so we can get the relative path to
        # the file
        relpath = os.path.relpath(base,latex_root).replace('\\','/')
        for j,this_figure in enumerate(figure):
            pdf3d = False
            if animate:
                this_filename = base+'_{:}'.format(j)+ext
                relpath += '_'
            else:
                this_filename = figure_save_names[i]
            # Matplotlib Figure
            if isinstance(this_figure,plt.Figure):
                if ext == '':
                    this_filename += '.pdf'
                this_figure.savefig(this_filename)
            # Pyqtgraph PlotItem
            elif isinstance(this_figure,pqtg.PlotItem):
                if ext == '':
                    this_filename += '.png'
                this_figure.writeImage(this_filename)
            # ShapePlotter
            elif isinstance(this_figure, ShapePlotter):
                if ext == '':
                    ext = '.png'
                    this_filename += ext
                if animation_style is None or animation_style.lower() == 'none':
                    PIL.Image.fromarray(this_figure.screenshot()).save(this_filename)
                elif animation_style.lower() == '2d':
                    this_figure.save_animation(this_filename,frames=animation_frames,
                                               frame_rate = animation_frame_rate,
                                               individual_images = True)
                    animate = True
                    relpath += '-'
                    num_frames = animation_frames
                elif animation_style.lower() == '3d':
                    raise ValueError('3D Animation is not supported for ShapePlotter. Pass ShapeArray for interactive mode shape plots.')
                else:
                    raise ValueError('Invalid animation_style.  Must be one of None, "2d", or "3d".')
            # ShapeArray
            elif isinstance(this_figure,ShapeArray):
                if geometry is None:
                    raise ValueError('If a ShapeArray is passed as a figure, a Geometry must also be passed to the geometry argument.')
                if ext == '':
                    ext = '.png'
                    this_filename += ext
                plotter = geometry.plot_shape(this_figure,**geometry_plot_shape_kwargs)
                if animation_style is None or animation_style.lower() == 'none':
                    PIL.Image.fromarray(plotter.screenshot()).save(this_filename)
                elif animation_style.lower() == '2d':
                    plotter.save_animation(this_filename,frames=animation_frames,
                                           frame_rate = animation_frame_rate,
                                           individual_images = True)
                    animate = True
                    relpath += '-'
                    num_frames = animation_frames
                elif animation_style.lower() == '3d':
                    if vtkU3DExporter is None:
                        raise ValueError('Cannot Import vtkU3DExporter.  It must first be installed with `pip install vtk-u3dexporter`')
                    PIL.Image.fromarray(plotter.screenshot()).save(this_filename)
                    u3d_filename = this_filename.replace(ext,'.u3d')
                    js_filename = this_filename.replace(ext,'.js')
                    rel_path_u3d = os.path.relpath(u3d_filename,latex_root).replace('\\','/')
                    rel_path_js = os.path.relpath(js_filename,latex_root).replace('\\','/')
                    # Pick out appropriate arguments
                    kwargs = {}
                    try:
                        kwargs['node_size'] = geometry_plot_shape_kwargs['plot_kwargs']['node_size']
                    except KeyError:
                        pass
                    try:
                        kwargs['line_width'] = geometry_plot_shape_kwargs['plot_kwargs']['line_width']
                    except KeyError:
                        pass
                    try:
                        kwargs['opacity'] = geometry_plot_shape_kwargs['deformed_opacity']
                    except KeyError:
                        pass
                    try:
                        kwargs['show_edges'] = geometry_plot_shape_kwargs['plot_kwargs']['show_edges']
                    except KeyError:
                        pass
                    try:
                        kwargs['displacement_scale'] = geometry_plot_shape_kwargs['starting_scale']
                    except KeyError:
                        pass
                    create_animated_modeshape_content(geometry,this_figure,
                                                      u3d_name=u3d_filename.replace('.u3d',''),
                                                      js_name = js_filename, one_js = True,
                                                      **kwargs)
                    pdf3d = True
                    pdf3d_parameters = ', '.join([key+'='+val for key,val in get_view_parameters_from_plotter(plotter).items()])
                    pdf3d_parameters += ', '+graphics_options+', add3Djscript='+rel_path_js
                else:
                    plotter.close()
                    raise ValueError('Invalid animation_style.  Must be one of None, "2d", or "3d".')
                plotter.close()
            # GeometryPlotter
            elif isinstance(this_figure, GeometryPlotter):
                if ext == '':
                    ext = '.png'
                    this_filename += ext
                if animation_style is None or animation_style.lower() == 'none':
                    PIL.Image.fromarray(this_figure.screenshot()).save(this_filename)
                elif animation_style.lower() == '2d':
                    this_figure.save_rotation_animation(this_filename,frames=animation_frames,
                                                        frame_rate = animation_frame_rate,
                                                        individual_images = True)
                    animate = True
                    relpath += '-'
                    num_frames = animation_frames
                elif animation_style.lower() == '3d':
                    if vtkU3DExporter is None:
                        raise ValueError('Cannot Import vtkU3DExporter.  It must first be installed with `pip install vtk-u3dexporter`')
                    PIL.Image.fromarray(this_figure.screenshot()).save(this_filename)
                    u3d_filename = this_filename.replace(ext,'.u3d')
                    rel_path_u3d = os.path.relpath(u3d_filename,latex_root).replace('\\','/')
                    exporter = vtkU3DExporter.vtkU3DExporter()
                    exporter.SetFileName(u3d_filename.replace('.u3d',''))
                    exporter.SetInput(this_figure.render_window)
                    exporter.Write()
                    
                    pdf3d = True
                    pdf3d_parameters = ', '.join([key+'='+val for key,val in get_view_parameters_from_plotter(this_figure).items()])
                    pdf3d_parameters += ', '+graphics_options
                else:
                    raise ValueError('Invalid animation_style.  Must be one of None, "2d", or "3d".')
            elif isinstance(this_figure, int):
                if shapeplotter is not None:
                    shapeplotter.current_shape = j
                    shapeplotter.compute_displacements()
                    shapeplotter.update_shape_mode(0)
                    shapeplotter.show_comment()
                    QApplication.processEvents()
                    if animation_style is None or animation_style.lower() == 'none':
                        PIL.Image.fromarray(shapeplotter.screenshot()).save(this_filename)
                    elif animation_style.lower() == '2d':
                        shapeplotter.save_animation(this_filename,frames=animation_frames,
                                               frame_rate = animation_frame_rate,
                                               individual_images = True)
                        animate = True
                        relpath += '-'
                        num_frames = animation_frames
                else:
                    raise ValueError('Bad type with integer figure.')
            # Bytes object
            elif isinstance(this_figure, bytes):
                if ext == '':
                    this_filename += '.png'
                PIL.Image.open(BytesIO(this_figure)).save(this_filename)
            # String to file name
            elif isinstance(this_figure, str):
                if ext == '':
                    this_filename += os.path.splitext(this_figure)[-1]
                copy(this_figure,this_filename)
            # 2D NumpyArray
            elif isinstance(this_figure, np.ndarray):
                if ext == '':
                    this_filename += '.png'
                PIL.Image.fromarray(this_figure).save(this_filename)
            # Otherwise
            else:
                raise ValueError('Unknown Figure Type: {:}'.format(type(this_figure)))
        # Now we end the figure and create a new one if we are at the right
        # subfigure number
        if (subfigures and max_subfigures_per_page is not None and 
            ((i-max_subfigures_first_page)%max_subfigures_per_page == 0
              and i > 0)):
            latex_string += r"""
\end{figure}
\begin{figure}[h]
    \ContinuedFloat
    \centering"""
        # If we have subfigures we need to stick in the subfigure environment
        if subfigures:
            latex_string += r"""
    \begin{subfigure}"""+subfigure_options+r"""
        \centering"""
        # Now we have to insert the includegraphics or animategraphics command
        if animate:
            latex_string += r"""
        \animategraphics[{graphics_options:}]{{{num_frames:}}}{{{base_name:}}}{{0}}{{{end_frame:}}}""".format(
            graphics_options=animate_graphics_options,
            num_frames=num_frames,
            base_name=relpath, end_frame=num_frames - 1)
        elif pdf3d:
            latex_string += r"""
        \includemedia[{graphics_options_3D:}]{{\includegraphics[{graphics_options:}]{{{base_name:}}}}}{{{base_name_u3d:}}}""".format(
            graphics_options_3D = pdf3d_parameters,
            graphics_options = graphics_options,
            base_name = relpath,
            base_name_u3d = rel_path_u3d)
        else:
            latex_string += r"""
        \includegraphics[{:}]{{{:}}}""".format(
        graphics_options,relpath)
        # Now add captions and labels if they exist
        if subfigures:
            latex_string += r"""
        \caption{{{:}}}""".format('' if subfigure_captions is None else subfigure_captions[i])
            if subfigure_labels is not None:
                if isinstance(subfigure_labels,str):
                    label = subfigure_labels.format(i)
                else:
                    label = subfigure_labels[i]
                latex_string += r"""
        \label{{{:}}}""".format(label)
            latex_string += r"""
    \end{subfigure}"""
    # Add the figure caption and label
    latex_string += r"""
    \caption{{{:}}}""".format('' if figure_caption is None else figure_caption)
    if figure_label is not None:
        latex_string += r"""
    \label{{{:}}}""".format(figure_label)
    latex_string += r"""
\end{figure}
    """
    return latex_string

def table(table, justification_string = None, 
          table_label = None, table_caption = None, longtable = False,
          header = True, horizontal_lines = False, table_placement = ''):
    if isinstance(table,pd.DataFrame):
        table_as_list = table.to_numpy().tolist()
        if header:
            table_as_list.insert(0,table.columns.tolist())
        table = table_as_list
    nrows = len(table)
    ncols = len(table[0])
    if justification_string is None:
        justification_string = 'c'*ncols
    if longtable:
        latex_string = r'\begin{{longtable}}{{{:}}}'.format(justification_string)+r'''
    \caption{{{:}}}'''.format('' if table_caption is None else table_caption)
        if table_label is not None:
            latex_string += r'''
    \label{{{:}}}'''.format(table_label)
        latex_string += '\\\\'
    else:
        latex_string = r'''\begin{{table}}{:}
    \centering
    \caption{{{:}}}'''.format(table_placement,'' if table_caption is None else table_caption)
        if table_label is not None:
            latex_string += r'''
    \label{{{:}}}'''.format(table_label)
        latex_string += r'''
    \begin{{tabular}}{{{:}}}'''.format(justification_string)
    # Now create the meat of the table
    if horizontal_lines:
        latex_string += r'''
        \hline'''
    for i in range(nrows):
        row = '        '+' & '.join([str(table[i][j]).replace('%','\\%') for j in range(ncols)]) + '\\\\'
        if header and i == 0:
            row += r'\hline'
            if longtable:
                row += '\n        \\endhead'
        latex_string += '\n'+row
        if horizontal_lines:
            latex_string += r'''
        \hline'''
    if longtable:
        latex_string += r'''
\end{longtable}'''
    else:
        latex_string += r'''
    \end{tabular}
\end{table}'''
    return latex_string