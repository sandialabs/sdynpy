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
from ..signal_processing.sdynpy_correlation import mac, matrix_plot


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
                         latex_shape_table_caption='List of modes extracted from the test data.  Modal parameters are shown along with a brief description of the mode shape.',
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
    output_string += '''
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
