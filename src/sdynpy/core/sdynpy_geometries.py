"""
Objects and procedures to handle operations on multiple test or model geometries

This module defines plotter objects for plotting multiple geometries and shapes
simultaneously.

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

import copy
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import pyqtgraph as pg
import vtk
from qtpy import uic, QtCore
from qtpy.QtGui import QKeySequence
try:
    from qtpy.QtGui import QAction
except ImportError:
    from qtpy.QtWidgets import QAction
from qtpy.QtWidgets import QApplication, QMainWindow, QSizePolicy, QFileDialog, QMenu
import time
import os
from PIL import Image

from .sdynpy_geometry import GeometryPlotter, global_coord, global_deflection
from .sdynpy_shape import ShapeCommentTable
from .sdynpy_colors import color_list


class MultipleShapePlotter(GeometryPlotter):
    """Class used to plot animated shapes on multiple geometries"""

    def __init__(self, geometries, shapes_list, plot_kwargs_list=None,
                 background_plotter_kwargs={'auto_update': 0.01},
                 undeformed_opacity=0.25, starting_scale=1.0,
                 deformed_opacity=1.0, shape_name='Mode',
                 show_damping=True):
        """
        Create a MultipleShapePlotter object to plot shapes on multiple geometries.

        Parameters
        ----------
        geometries : list of Geometry
            Geometries on which the shapes will be plotted.
        shapes_list : list of ShapeArray
            Shapes to plot on the geometries.
        plot_kwargs_list : list of dict, optional
            Keyword arguments passed to the Geometry.plot function for each geometry.
        background_plotter_kwargs : dict, optional
            Keyword arguments passed to the BackgroundPlotter constructor.
        undeformed_opacity : float, optional
            Opacity of the undeformed geometry.
        starting_scale : float, optional
            Starting scale of the shapes on the plot.
        deformed_opacity : float, optional
            Opacity of the deformed geometry.
        shape_name : str, optional
            Name used to describe the shape.
        show_damping : bool, optional
            Specifies whether damping should be displayed.
        """
        self.shape_select_toolbar: pvqt.plotting.QToolBar = None
        self.animation_control_toolbar: pvqt.plotting.QToolBar = None
        self.signed_amplitude_action = None
        self.complex_action = None
        self.real_action = None
        self.imag_action = None
        self.loop_action = None
        self.plot_comments_action = None
        self.save_animation_action = None
        self.display_undeformed = None
        
        self.geometries = geometries
        self.shapes_list = [s.ravel() for s in shapes_list]
        if plot_kwargs_list is None:
            self.plot_kwargs_list = [{} for _ in geometries]
        else:
            self.plot_kwargs_list = plot_kwargs_list

        self.node_displacements_list = []
        self.current_shape = 0
        self.displacement_scale = 1.0
        self.displacement_scale_modification = starting_scale
        self.animation_speed = 1.0
        self.last_time = time.time()
        self.phase = 0.0
        self.timer: pvqt.plotting.QTimer = None
        self.shape_name = shape_name
        self.show_damping = show_damping
        self.shape_comment_table = None

        super().__init__(**background_plotter_kwargs)

        self.text = self.add_text('', font_size=10)

        self.complex_action.setChecked(True)
        self.loop_action.setChecked(True)
        self.plot_comments_action.setChecked(True)

        self.face_mesh_undeformed_list = []
        self.point_mesh_undeformed_list = []
        self.solid_mesh_undeformed_list = []
        self.face_mesh_deformed_list = []
        self.point_mesh_deformed_list = []
        self.solid_mesh_deformed_list = []

        # Assign colors automatically if not provided
        for i, plot_kwargs in enumerate(self.plot_kwargs_list):
            if 'color' not in plot_kwargs:
                plot_kwargs['color'] = color_list[i % len(color_list)]

        for geometry, plot_kwargs in zip(self.geometries, self.plot_kwargs_list):
            _, face_mesh_u, point_mesh_u, solid_mesh_u = geometry.plot_actors(
                plotter=self, opacity=undeformed_opacity, **plot_kwargs)
            self.face_mesh_undeformed_list.append(face_mesh_u)
            self.point_mesh_undeformed_list.append(point_mesh_u)
            self.solid_mesh_undeformed_list.append(solid_mesh_u)

            _, face_mesh_d, point_mesh_d, solid_mesh_d = geometry.plot_actors(
                plotter=self, opacity=deformed_opacity, **plot_kwargs)
            self.face_mesh_deformed_list.append(face_mesh_d)
            self.point_mesh_deformed_list.append(point_mesh_d)
            self.solid_mesh_deformed_list.append(solid_mesh_d)

        self.indices_list = []
        self.coordinate_node_index_map_list = []
        self.global_deflections_list = []

        for geometry, shapes in zip(self.geometries, self.shapes_list):
            coordinates = shapes[self.current_shape].coordinate
            nodes = coordinates.node
            indices = np.in1d(coordinates.node, geometry.node.id)
            self.indices_list.append(indices)
            coordinates = coordinates[indices]
            nodes = nodes[indices]
            direction = coordinates.direction
            
            coordinate_node_index_map = {}
            for coordinate_index, node_id in np.ndenumerate(nodes):
                if direction[coordinate_index] == 0 or abs(direction[coordinate_index]) > 3:
                    continue
                if node_id not in coordinate_node_index_map:
                    coordinate_node_index_map[node_id] = []
                coordinate_node_index_map[node_id].append(coordinate_index[0])
            self.coordinate_node_index_map_list.append(coordinate_node_index_map)

            local_deformations = coordinates.local_direction()
            ordered_nodes = geometry.node(nodes)
            coordinate_systems = geometry.coordinate_system(ordered_nodes.disp_cs)
            points = global_coord(geometry.coordinate_system(
                ordered_nodes.def_cs), ordered_nodes.coordinate)
            global_deflections = global_deflection(coordinate_systems, local_deformations, points)
            self.global_deflections_list.append(global_deflections)

        self.compute_displacements()
        self.update_shape()
        self.show_comment()

    def compute_displacements(self, compute_scale=True):
        self.node_displacements_list = []
        scales = []

        for i, (geometry, shapes) in enumerate(zip(self.geometries, self.shapes_list)):
            indices = self.indices_list[i]
            coordinate_node_index_map = self.coordinate_node_index_map_list[i]
            global_deflections = self.global_deflections_list[i]

            shape_displacements = shapes[self.current_shape].shape_matrix[..., indices]
            node_displacements = np.zeros(geometry.node.shape + (3,), 
                                          dtype=shape_displacements.dtype)
            for node_index, node in geometry.node.ndenumerate():
                try:
                    node_indices = coordinate_node_index_map[node.id]
                except KeyError:
                    node_displacements[node_index] = 0
                    continue
                node_deflection_scales = shape_displacements[node_indices]
                node_deflection_directions = global_deflections[node_indices]
                node_displacements[node_index] += np.sum(node_deflection_scales[:, np.newaxis] *
                                                         node_deflection_directions, axis=0)
            
            node_displacements = node_displacements.reshape(-1, 3)
            self.node_displacements_list.append(node_displacements)

            if compute_scale:
                max_displacement = np.max(np.linalg.norm(node_displacements, axis=-1))
                if max_displacement > 1e-9:
                    global_coords = geometry.global_node_coordinate()
                    bbox_diagonal = np.linalg.norm(
                        np.max(global_coords, axis=0) - np.min(global_coords, axis=0))
                    scales.append(0.05 * bbox_diagonal / max_displacement)

        if compute_scale and scales:
            self.displacement_scale = min(scales)

    def update_shape_mode(self, phase=None):
        if phase is None:
            now = time.time()
            time_elapsed = now - self.last_time
            phase_change = 2 * np.pi / self.animation_speed * time_elapsed
            self.last_time = now
            self.phase += phase_change
        else:
            self.phase = phase
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi
            if not self.loop_action.isChecked():
                self.stop_animation()

        for i in range(len(self.geometries)):
            node_displacements = self.node_displacements_list[i]
            if self.complex_action.isChecked():
                deformation = np.real(np.exp(self.phase * 1j) * node_displacements)
            elif self.real_action.isChecked():
                deformation = np.cos(self.phase) * np.real(node_displacements)
            elif self.imag_action.isChecked():
                deformation = np.cos(self.phase) * np.imag(node_displacements)
            elif self.signed_amplitude_action.isChecked():
                imag = np.imag(node_displacements).flatten()[:, np.newaxis]
                real = np.real(node_displacements).flatten()[:, np.newaxis]
                best_fit = np.linalg.lstsq(real, imag, rcond=None)[0][0, 0]
                best_fit_phase = np.arctan(best_fit)
                rectified_displacements = node_displacements / np.exp(1j * best_fit_phase)
                deformation = np.cos(self.phase) * np.real(rectified_displacements)

            for mesh_deformed, mesh_undeformed in (
                    (self.face_mesh_deformed_list[i], self.face_mesh_undeformed_list[i]),
                    (self.point_mesh_deformed_list[i], self.point_mesh_undeformed_list[i]),
                    (self.solid_mesh_deformed_list[i], self.solid_mesh_undeformed_list[i])
            ):
                if mesh_deformed is not None and mesh_undeformed is not None:
                    mesh_deformed.mapper.dataset.points = (mesh_undeformed.mapper.dataset.points +
                                                            deformation * self.displacement_scale * self.displacement_scale_modification)
        self.render()

    def update_shape(self):
        self.update_shape_mode()

    def save_animation_from_action(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Select File to Save Animation', filter='GIF (*.gif)')
        if filename:
            self.save_animation(filename)

    def save_animation(self, filename=None, frames=20, frame_rate=20,
                       individual_images=False):
        self.stop_animation()
        imgs = []
        phases = np.linspace(0, 2 * np.pi, frames, endpoint=False)
        for phase in phases:
            self.update_shape_mode(phase)
            QApplication.processEvents()
            imgs.append(self.screenshot())
        if filename is None:
            return imgs
        else:
            if individual_images:
                for i, img in enumerate(imgs):
                    fname, ext = os.path.splitext(filename)
                    Image.fromarray(img).save(f"{fname}-{i}{ext}")
            else:
                imgs = [Image.fromarray(img).convert('P', palette=Image.ADAPTIVE, colors=256) for img in imgs]
                imgs[0].save(fp=filename, format='GIF', append_images=imgs[1:],
                             save_all=True, duration=int(1 / frame_rate * 1000), loop=0)

    def save_animation_all_shapes(self, filename_base='Shape_{:}.gif',
                                  frames=20, frame_rate=20,
                                  individual_images=False):
        num_shapes = min(s.size for s in self.shapes_list)
        for shape_idx in range(num_shapes):
            self.current_shape = shape_idx
            self.compute_displacements()
            self.show_comment()
            self.save_animation(filename_base.format(shape_idx + 1),
                                frames, frame_rate, individual_images)

    def add_menu_bar(self):
        super().add_menu_bar()
        file_menu = [child for child in self.main_menu.children()
                     if child.__class__.__name__ == 'QMenu'
                     and child.title() == 'File'][0]
        if file_menu:
            self.save_animation_action = QAction('Save Animation')
            self.save_animation_action.triggered.connect(self.save_animation_from_action)
            file_menu.insertAction(file_menu.actions()[1], self.save_animation_action)

        shape_menu = self.main_menu.addMenu("Shape")
        complex_display_menu = shape_menu.addMenu("Complex Display")
        self.signed_amplitude_action = QAction('Signed Amplitude', checkable=True)
        self.signed_amplitude_action.triggered.connect(self.select_complex)
        complex_display_menu.addAction(self.signed_amplitude_action)
        self.complex_action = QAction('Complex', checkable=True)
        self.complex_action.triggered.connect(self.select_complex)
        complex_display_menu.addAction(self.complex_action)
        self.real_action = QAction('Real', checkable=True)
        self.real_action.triggered.connect(self.select_complex)
        complex_display_menu.addAction(self.real_action)
        self.imag_action = QAction('Imaginary', checkable=True)
        self.imag_action.triggered.connect(self.select_complex)
        complex_display_menu.addAction(self.imag_action)
        
        scaling_menu = shape_menu.addMenu('Scaling')
        scaling_menu.addAction('1/4x', self.select_scaling_0p25)
        scaling_menu.addAction('1/2x', self.select_scaling_0p5)
        self.scale_0p8_action = QAction('0.8x')
        self.scale_0p8_action.triggered.connect(self.select_scaling_0p8)
        self.scale_0p8_action.setShortcut(QKeySequence(','))
        scaling_menu.addAction(self.scale_0p8_action)
        self.scale_reset_action = QAction('Reset')
        self.scale_reset_action.triggered.connect(self.select_scaling_1)
        self.scale_reset_action.setShortcut(QKeySequence('/'))
        scaling_menu.addAction(self.scale_reset_action)
        self.scale_1p25_action = QAction('1.25x')
        self.scale_1p25_action.triggered.connect(self.select_scaling_1p25)
        self.scale_1p25_action.setShortcut(QKeySequence('.'))
        scaling_menu.addAction(self.scale_1p25_action)
        scaling_menu.addAction('2x', self.select_scaling_2p0)
        scaling_menu.addAction('4x', self.select_scaling_4p0)
        speed_menu = shape_menu.addMenu('Animation Speed')
        self.speed_0p8_action = QAction('0.8x')
        self.speed_0p8_action.triggered.connect(self.select_speed_0p8)
        self.speed_0p8_action.setShortcut(QKeySequence(';'))
        speed_menu.addAction(self.speed_0p8_action)
        speed_menu.addAction('Reset', self.select_speed_1)
        self.speed_1p25_action = QAction('1.25x')
        self.speed_1p25_action.triggered.connect(self.select_speed_1p25)
        self.speed_1p25_action.setShortcut(QKeySequence("'"))
        speed_menu.addAction(self.speed_1p25_action)
        shape_menu.addSeparator()
        self.plot_comments_action = QAction('Show Comments', checkable=True)
        self.plot_comments_action.triggered.connect(self.show_comment)
        shape_menu.addAction(self.plot_comments_action)
        self.loop_action = QAction('Loop Animation', checkable=True)
        self.loop_action.triggered.connect(self.select_loop)
        shape_menu.addAction(self.loop_action)

        display_menu = self.main_menu.addMenu('Display')
        self.display_undeformed = QAction('Show Undeformed', checkable=True)
        self.display_undeformed.setChecked(True)
        self.display_undeformed.triggered.connect(self.toggle_undeformed)
        display_menu.addAction(self.display_undeformed)

    def add_toolbars(self):
        super().add_toolbars()
        self.shape_select_toolbar = self.app_window.addToolBar("Shape Selector")
        self._add_action(self.shape_select_toolbar, '<<', self.prev_shape)
        self._add_action(self.shape_select_toolbar, '?', self.select_shape)
        self._add_action(self.shape_select_toolbar, '>>', self.next_shape)
        self.animation_control_toolbar = self.app_window.addToolBar("Animation Control")
        self._add_action(self.animation_control_toolbar, 'Play', self.play_animation)
        self._add_action(self.animation_control_toolbar, 'Stop', self.stop_animation)

    def play_animation(self):
        self.timer = pvqt.plotting.QTimer()
        self.timer.timeout.connect(self.update_shape)
        self.timer.start(10)
        self.phase = 0.0
        self.last_time = time.time()

    def stop_animation(self):
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

    def toggle_undeformed(self):
        is_visible = self.display_undeformed.isChecked()
        for mesh in self.face_mesh_undeformed_list + self.point_mesh_undeformed_list + self.solid_mesh_undeformed_list:
            if mesh is not None:
                mesh.SetVisibility(is_visible)
        self.render()

    def select_scaling_0p25(self): self.displacement_scale_modification *= 0.25; self.update_shape()
    def select_scaling_0p5(self): self.displacement_scale_modification *= 0.5; self.update_shape()
    def select_scaling_0p8(self): self.displacement_scale_modification *= 0.8; self.update_shape()
    def select_scaling_1(self): self.displacement_scale_modification = 1.0; self.update_shape()
    def select_scaling_1p25(self): self.displacement_scale_modification *= 1.25; self.update_shape()
    def select_scaling_2p0(self): self.displacement_scale_modification *= 2.0; self.update_shape()
    def select_scaling_4p0(self): self.displacement_scale_modification *= 4.0; self.update_shape()

    def select_speed_1(self): self.animation_speed = 1.0
    def select_speed_0p8(self): self.animation_speed /= 0.8
    def select_speed_1p25(self): self.animation_speed /= 1.25

    def select_complex(self):
        actions = [self.signed_amplitude_action, self.complex_action, self.real_action, self.imag_action]
        sender = self.sender()
        for action in actions:
            if action is not sender:
                action.setChecked(False)
        if not sender.isChecked():
            sender.setChecked(True)
        self.phase = 0.0
        self.last_time = time.time()
        self.update_shape()

    def select_loop(self): pass

    def next_shape(self):
        self.current_shape += 1
        num_shapes = min(s.size for s in self.shapes_list) if self.shapes_list else 0
        if self.current_shape >= num_shapes:
            self.current_shape = 0
        self.compute_displacements()
        self.phase = 0.0
        self.last_time = time.time()
        self.update_shape()
        self.show_comment()

    def prev_shape(self):
        self.current_shape -= 1
        if self.current_shape < 0:
            num_shapes = min(s.size for s in self.shapes_list) if self.shapes_list else 0
            self.current_shape = num_shapes - 1
        self.compute_displacements()
        self.phase = 0.0
        self.last_time = time.time()
        self.update_shape()
        self.show_comment()

    def reset_shape(self):
        self.compute_displacements()
        self.phase = 0.0
        self.last_time = time.time()
        self.update_shape()
        self.show_comment()

    def select_shape(self):
        if self.shapes_list:
            self.shape_comment_table = ShapeCommentTable(self.shapes_list[0], self)

    def show_comment(self):
        if self.plot_comments_action.isChecked() and self.shapes_list:
            shapes = self.shapes_list[0]
            if self.current_shape < len(shapes):
                shape = shapes[self.current_shape]
                comment = [
                    f'{self.shape_name} {self.current_shape + 1}',
                    f'  Frequency: {shape.frequency:0.2f}'
                ]
                if self.show_damping:
                    comment.append(f'  Damping: {shape.damping * 100:.2f}%')
                for i in range(1, 6):
                    comment.append(f'  {getattr(shape, f"comment{i}")}')
                self.text.SetText(2, '\n'.join(comment))
        else:
            self.text.SetText(2, '')

    def _close(self):
        self.stop_animation()
        super()._close()


class MultipleDeflectionShapePlotter(MultipleShapePlotter):
    """Class used to plot animated deflection shapes from spectra on multiple geometries"""

    def __init__(self, geometries, deflection_shape_data_list, plot_kwargs_list=None,
                 background_plotter_kwargs={'auto_update': 0.01},
                 undeformed_opacity=0.25, starting_scale=1.0,
                 deformed_opacity=1.0, num_curves=50):
        """
        Create a MultipleDeflectionShapePlotter object.

        Parameters
        ----------
        geometries : list of Geometry
            Geometries on which the shapes will be plotted.
        deflection_shape_data_list : list of NDDataArray
            Data arrays containing the deflection shapes to plot.
        plot_kwargs_list : list of dict, optional
            Keyword arguments for plotting each geometry.
        background_plotter_kwargs : dict, optional
            Keyword arguments for the BackgroundPlotter.
        undeformed_opacity : float, optional
            Opacity of the undeformed geometry.
        starting_scale : float, optional
            Starting scale of the shapes.
        deformed_opacity : float, optional
            Opacity of the deformed geometry.
        num_curves : int, optional
            Maximum number of curves to plot on the frequency selector.
        """
        from .sdynpy_shape import shape_array

        shapes_list = []
        for deflection_shape_data in deflection_shape_data_list:
            flattened_data = deflection_shape_data.flatten()
            response_coordinates = flattened_data.response_coordinate
            if not np.unique(response_coordinates).size == response_coordinates.size:
                raise ValueError('all response coordinates for operating data must be unique')
            shapes = shape_array(response_coordinates, flattened_data.ordinate.T,
                                 flattened_data[0].abscissa)
            shapes_list.append(shapes)

        super().__init__(geometries, shapes_list, plot_kwargs_list,
                         background_plotter_kwargs, undeformed_opacity,
                         starting_scale, deformed_opacity, shape_name='Shape',
                         show_damping=False)

        layout = self.frame.layout()
        self.frequency_selector_plot = pg.PlotWidget()
        layout.addWidget(self.frequency_selector_plot)
        sp = self.frequency_selector_plot.sizePolicy()
        sp.setVerticalPolicy(QSizePolicy.Minimum)
        self.frequency_selector_plot.setSizePolicy(sp)
        self.frequency_selector_plot.sizeHint = lambda: QtCore.QSize(1006, 120)

        all_max_envelopes = []
        min_abscissa = float('inf')
        max_abscissa = float('-inf')

        for deflection_shape_data in deflection_shape_data_list:
            flattened_data = deflection_shape_data.flatten()
            if num_curves < flattened_data.size:
                indices = np.linspace(0, flattened_data.size - 1, num_curves).astype(int)
            else:
                indices = np.arange(flattened_data.size)
            
            for data in flattened_data[indices]:
                self.frequency_selector_plot.plot(data.abscissa, np.abs(data.ordinate))
            
            max_envelope = np.linalg.svd(
                flattened_data.ordinate.T[..., np.newaxis], compute_uv=False).squeeze()
            self.frequency_selector_plot.plot(
                flattened_data[0].abscissa, max_envelope, pen=pg.mkPen(color='k', width=2))
            
            all_max_envelopes.append(max_envelope)
            min_abscissa = min(min_abscissa, flattened_data.abscissa.min())
            max_abscissa = max(max_abscissa, flattened_data.abscissa.max())

        self.frequency_selector = pg.InfiniteLine(
            0, movable=True, bounds=[min_abscissa, max_abscissa])
        self.frequency_selector_plot.addItem(self.frequency_selector)
        self.frequency_selector.sigPositionChanged.connect(self.modify_abscissa)

        self.frequency_selector_plot.setLogMode(False, True)
        if all_max_envelopes:
            full_envelope = np.concatenate(all_max_envelopes)
            if full_envelope.size > 0:
                min_val = full_envelope[full_envelope > 0].min()
                max_val = full_envelope.max()
                if min_val > 0 and max_val > 0:
                    self.frequency_selector_plot.setRange(
                        yRange=(np.log10(min_val), np.log10(max_val)))

    def modify_abscissa(self):
        frequency = self.frequency_selector.value()
        if self.shapes_list:
            self.current_shape = np.argmin(np.abs(self.shapes_list[0].frequency - frequency))
            self.compute_displacements()
            self.update_shape()
            self.show_comment()

    def save_multiline_animation(self, filename=None, frame_rate=20,
                                 phase_change_per_frame=0.087,
                                 frequency_change_per_frame=0.5,
                                 start_phase=0.0,
                                 start_frequency=None, end_frequency=None,
                                 individual_images=False):
        self.stop_animation()
        imgs = []
        
        if not self.shapes_list:
            return

        ref_frequencies = self.shapes_list[0].frequency
        
        if start_frequency is None:
            start_frequency = ref_frequencies[0]
        if end_frequency is None:
            end_frequency = ref_frequencies[-1]
            
        frequency_frames = np.arange(start_frequency, end_frequency, frequency_change_per_frame)
        phase_frames = start_phase + (np.arange(frequency_frames.size) * phase_change_per_frame)
        
        for frequency, phase in zip(frequency_frames, phase_frames):
            self.current_shape = np.argmin(np.abs(ref_frequencies - frequency))
            self.compute_displacements()
            self.update_shape_mode(phase)
            self.show_comment()
            QApplication.processEvents()
            imgs.append(self.screenshot())
            
        if filename is None:
            return imgs
        else:
            if individual_images:
                for i, img in enumerate(imgs):
                    fname, ext = os.path.splitext(filename)
                    Image.fromarray(img).save(f"{fname}-{i}{ext}")
            else:
                imgs = [Image.fromarray(img).convert('P', palette=Image.ADAPTIVE, colors=256) for img in imgs]
                imgs[0].save(fp=filename, format='GIF', append_images=imgs[1:],
                             save_all=True, duration=int(1 / frame_rate * 1000), loop=0)
