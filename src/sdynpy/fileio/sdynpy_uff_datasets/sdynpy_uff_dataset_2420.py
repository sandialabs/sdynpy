# -*- coding: utf-8 -*-
"""
Coordinate Systems

This dataset defines the coordinate systems in a test geometry

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
from ...core import sdynpy_colors
import numpy as np


class Sdynpy_UFF_Dataset_2420:
    def __init__(self, part_uid=1, part_name='',
                 cs_labels=[], cs_types=[], cs_colors=[],
                 cs_names=[], cs_matrices=np.zeros((0, 4, 3))):
        self.part_uid = part_uid
        self.part_name = part_name
        self.cs_labels = cs_labels
        self.cs_types = cs_types
        self.cs_colors = cs_colors
        self.cs_names = cs_names
        self.cs_matrices = cs_matrices

    @property
    def dataset_number(self):
        return 2420

    @classmethod
    def from_uff_data_array(cls, data):
        # Transform from binary to ascii
        data = [line.decode() for line in data]
        #       Record 1:        FORMAT (1I10)
        #                 Field 1       -- Part UID
        part_uid, = parse_uff_line(data[0], ['I10'])
#        Record 2:        FORMAT (40A2)
#                 Field 1       -- Part Name
        part_name, = parse_uff_line(data[1], ['A80'])
        cs_labels = []
        cs_types = []
        cs_colors = []
        cs_matrices = []
        cs_names = []
        index = 2
        while index < len(data):
            #           Record 3:        FORMAT (3I10)
            #                 Field 1       -- Coordinate System Label
            #                 Field 2       -- Coordinate System Type
            #                                  = 0, Cartesian
            #                                  = 1, Cylindrical
            #                                  = 2, Spherical
            #                 Field 3       -- Coordinate System Color
            cs_label, cs_type, cs_color = parse_uff_line(data[index], 3 * ['I10'])
#            Record 4:        FORMAT (40A2)
#                 Field 1       -- Coordinate System Name
            cs_name, = parse_uff_line(data[index + 1], ['A80'])
#           Record 5:        FORMAT (1P3D25.16)
#                 Field 1-3     -- Transformation Matrix Row 1
#
#           Record 6:        FORMAT (1P3D25.16)
#                 Field 1-3     -- Transformation Matrix Row 2
#
#           Record 7:        FORMAT (1P3D25.16)
#                 Field 1-3     -- Transformation Matrix Row 3
#
#           Record 8:        FORMAT (1P3D25.16)
#                 Field 1-3     -- Transformation Matrix Row 4
            cs_matrix, read_lines = parse_uff_lines(data[index + 2:], 3 * ['D25.16'], 12)
            cs_matrix = np.array(cs_matrix).reshape(4, 3)
            index += 2 + read_lines
            cs_labels.append(cs_label)
            cs_types.append(cs_type)
            cs_colors.append(cs_color)
            cs_matrices.append(cs_matrix)
            cs_names.append(cs_name)
        # Make sure matrices are the right size
        matrix_array = np.array(cs_matrices)
        ds_2420 = cls(part_uid, part_name, cs_labels, cs_types, cs_colors, cs_names,
                      matrix_array)
        return ds_2420

    def __repr__(self):
        return 'Sdynpy_UFF_Dataset_2420<{} coordinate systems(s)>'.format(len(self.cs_labels))

    def write_string(self):
        return_string = write_uff_line([self.part_uid, self.part_name], ['I10', '\n', 'A80'])

        for cs_label, cs_type, cs_color, cs_name, cs_matrix in zip(self.cs_labels, self.cs_types,
                                                                   self.cs_colors, self.cs_names,
                                                                   self.cs_matrices):
            return_string += (write_uff_line([cs_label, cs_type, cs_color], 3 * ['I10']) +
                              write_uff_line([cs_name], ['A80']) +
                              write_uff_line(cs_matrix.flatten(), 3 * ['D25.16']))
        return return_string

    def __str__(self):
        lines = self.write_string().split('\n')
        if len(lines) > 8:
            return 'Dataset 2420: Coordinate Systems\n  ' + '\n  '.join(lines[0:5] + ['.', '.', '.'])
        else:
            return 'Dataset 2420: Coordinate Systems\n  ' + '\n  '.join(lines)


def read(data):
    return Sdynpy_UFF_Dataset_2420.from_uff_data_array(data)
