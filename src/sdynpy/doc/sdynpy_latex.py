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
from ..core.sdynpy_geometry import GeometryPlotter, ShapePlotter
from shutil import copy


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

def figure(figures, figure_label = None, figure_caption = None,
           graphics_options = r'width=0.7\linewidth', 
           animate_graphics_options = r'width=0.7\linewidth,loop',
           figure_placement = '',
           subfigure_options = r'[t]{0.45\linewidth}', subfigure_labels = None, 
           subfigure_captions = None, max_subfigures_per_page = None,
           max_subfigures_first_page = None, figure_save_names = None,
           latex_root = r''
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
    """
    if len(figures) == 1:
        subfigures = False
    else:
        subfigures = True
        
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
            # 2D NumpyArray
            elif isinstance(this_figure, np.ndarray):
                if ext == '':
                    this_filename += '.png'
                PIL.Image.fromarray(this_figure).save(this_filename)
            # ShapePlotter
            elif isinstance(this_figure, ShapePlotter):
                raise NotImplementedError('ShapePlotter is not implemented yet')
            # GeometryPlotter
            elif isinstance(this_figure, GeometryPlotter):
                raise NotImplementedError('GeometryPlotter is not implemented yet')
            # Bytes object
            elif isinstance(this_figure, bytes):
                if ext == '':
                    this_filename += '.png'
                PIL.Image.frombytes(this_figure).save(this_filename)
            # String to file name
            elif isinstance(this_figure, str):
                if ext == '':
                    this_filename += os.path.splitext(this_figure)[-1]
                copy(this_figure,this_filename)
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
          header = True, horizontal_lines = False):
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
    else:
        latex_string = r'''\begin{{table}}
    \centering
    \caption{{{:}}}'''.format('' if table_caption is None else table_caption)
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
        row = '        '+' & '.join([str(table[i][j]) for j in range(ncols)]) + '\\\\'
        if header and i == 0:
            row += r'\hline'
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