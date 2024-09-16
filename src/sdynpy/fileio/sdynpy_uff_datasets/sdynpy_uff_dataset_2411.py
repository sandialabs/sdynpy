# -*- coding: utf-8 -*-
"""
Nodes

This dataset defines the node positions, colors, and coordinate systems in a
geometry.

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

from ..sdynpy_uff import UFFReadError, parse_uff_line, parse_uff_lines, write_uff_line
import numpy as np


class Sdynpy_UFF_Dataset_2411:
    def __init__(self, node_labels=np.array((1,)), export_coordinate_systems=np.array((1,)),
                 displacement_coordinate_systems=np.array((1,)), colors=np.array((1,)),
                 coordinates=np.array([(0, 0, 0)])):
        self.node_labels = node_labels
        self.export_coordinate_systems = export_coordinate_systems
        self.displacement_coordinate_systems = displacement_coordinate_systems
        self.colors = colors
        self.coordinates = coordinates

    @property
    def dataset_number(self):
        return 2411

    @classmethod
    def from_uff_data_array(cls, data):
        # Transform from binary to ascii
        data = [line.decode() for line in data]
        nnodes = len(data) // 2
        node_labels = np.empty(nnodes, dtype=int)
        export_coordinate_systems = np.empty(nnodes, dtype=int)
        displacement_coordinate_systems = np.empty(nnodes, dtype=int)
        coordinates = np.empty((nnodes, 3), dtype=float)
        colors = np.empty(nnodes, dtype=int)
        for i in range(nnodes):
            #            Record 1:        FORMAT(4I10)
            #                 Field 1       -- node label
            #                 Field 2       -- export coordinate system number
            #                 Field 3       -- displacement coordinate system number
            #                 Field 4       -- color
            node_labels[i], export_coordinate_systems[i], displacement_coordinate_systems[i], colors[i] = (
                parse_uff_line(data[2 * i], 4 * ['I10']))
# Record 2:        FORMAT(1P3D25.16)
#                 Fields 1-3    -- node coordinates in the part coordinate
#                                  system
            coordinates[i, :] = parse_uff_line(data[2 * i + 1], 3 * ['D25.16'])
        # Make sure coordinates don't end up being 2D
        ds_2411 = cls(node_labels, export_coordinate_systems, displacement_coordinate_systems,
                      colors, coordinates)
        return ds_2411

    def __repr__(self):
        return 'Sdynpy_UFF_Dataset_2411<{} node(s)>'.format(len(self.node_labels))

    def write_string(self):
        return_string = ''
        for node_label, export_coordinate_system, displacement_coordinate_system, color, coordinates in zip(
                self.node_labels, self.export_coordinate_systems, self.displacement_coordinate_systems, self.colors, self.coordinates):
            return_string += write_uff_line([node_label, export_coordinate_system, displacement_coordinate_system,
                                            color] + coordinates.tolist(), 4 * ['I10'] + ['\n'] + 3 * ['D25.16'])
        return return_string

    def __str__(self):
        lines = self.write_string().split('\n')
        if len(lines) > 8:
            return 'Dataset 2411: Nodes\n  ' + '\n  '.join(lines[0:5] + ['.', '.', '.'])
        else:
            return 'Dataset 2411: Nodes\n  ' + '\n  '.join(lines)


def read(data):
    return Sdynpy_UFF_Dataset_2411.from_uff_data_array(data)
