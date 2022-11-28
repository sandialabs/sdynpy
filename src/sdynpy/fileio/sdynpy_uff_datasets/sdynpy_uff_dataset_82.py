# -*- coding: utf-8 -*-
"""
Tracelines

This dataset defines tracelines, which are used to improve visibility of a
test geometry.

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


class Sdynpy_UFF_Dataset_82:
    def __init__(self,
                 traceline_number=1,
                 color=1,
                 identification='',
                 nodes=[1]):
        self.traceline_number = traceline_number
        self.color = color
        self.identification = identification
        self.nodes = nodes

    @property
    def dataset_number(self):
        return 82

    @classmethod
    def from_uff_data_array(cls, data):
        ids = []
        nodes = []
        colors = []
        descriptions = []
        index = 0
#            Record 1: FORMAT(3I10)
#                       Field 1 -    trace line number
#                       Field 2 -    number of nodes defining trace line
#                                    (maximum of 250)
#                       Field 3 -    color
        traceline_number, number_nodes, color = parse_uff_line(data[0], 3 * ['I10'])
#            Record 2: FORMAT(80A1)
#                       Field 1 -    Identification line
        identification, = parse_uff_line(data[1], ['A80'])
#            Record 3: FORMAT(8I10)
#                       Field 1 -    nodes defining trace line
#                               =    > 0 draw line to node
#                               =    0 move to node (a move to the first
#                                    node is implied)
        nodes, read_lines = parse_uff_lines(data[2:], 8 * ['I10'], number_nodes)
        # Make sure connectivities doesn't end up being 2D
        ds_82 = cls(traceline_number=traceline_number,
                    nodes=nodes,
                    color=color,
                    identification=identification)
        return ds_82

    def __repr__(self):
        return 'Sdynpy_UFF_Dataset_82<traceline {:}>'.format(self.traceline_number)

    def write_string(self):
        return_string = ''
        return_string += write_uff_line([self.traceline_number,
                                         len(self.nodes),
                                         self.color], 3 * ['I10'])
        return_string += write_uff_line([self.identification], ['A80'])
        return_string += write_uff_line(self.nodes, 8 * ['I10'])
        return return_string

    def __str__(self):
        lines = self.write_string().split('\n')
        if len(lines) > 8:
            return 'Dataset 82: Tracelines\n  ' + '\n  '.join(lines[0:5] + ['.', '.', '.'])
        else:
            return 'Dataset 82: Tracelines\n  ' + '\n  '.join(lines)


def read(data):
    return Sdynpy_UFF_Dataset_82.from_uff_data_array(data)
