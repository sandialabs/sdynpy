# -*- coding: utf-8 -*-
"""
Elements

This dataset defines the elements in a test geometry

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
from ...core.sdynpy_geometry import _beam_elem_types
from ...core import sdynpy_colors
import numpy as np


class Sdynpy_UFF_Dataset_2412:
    def __init__(self, element_labels=[1],
                 fe_descriptor_ids=[161],
                 physical_property_table_numbers=None, # [1],
                 material_property_table_numbers=None, # [1],
                 colors=[1],
                 connectivities=[[1]],
                 beam_orientations=None, # [None],
                 beam_fore_cross_section_numbers=None, #[None],
                 beam_aft_cross_section_numbers=None): #[None]):
        self.element_labels = element_labels
        self.fe_descriptor_ids = fe_descriptor_ids
        self.physical_property_table_numbers = physical_property_table_numbers if physical_property_table_numbers is not None else [1 for conn in connectivities]
        self.material_property_table_numbers = material_property_table_numbers if material_property_table_numbers is not None else [1 for conn in connectivities]
        self.colors = colors
        self.connectivities = connectivities
        self.beam_orientations = beam_orientations if beam_orientations is not None else [None for conn in connectivities]
        self.beam_fore_cross_section_numbers = beam_fore_cross_section_numbers if beam_fore_cross_section_numbers is not None else [None for conn in connectivities]
        self.beam_aft_cross_section_numbers = beam_aft_cross_section_numbers if beam_aft_cross_section_numbers is not None else [None for conn in connectivities]

    @property
    def dataset_number(self):
        return 2412

    @classmethod
    def from_uff_data_array(cls, data):
        # Transform from binary to ascii
        data = [line.decode() for line in data]
        element_labels = []
        fe_descriptor_ids = []
        physical_property_table_numbers = []
        material_property_table_numbers = []
        colors = []
        connectivities = []
        beam_orientations = []
        beam_fore_cross_section_numbers = []
        beam_aft_cross_section_numbers = []
        index = 0
        while index < len(data):
            #           Record 1:        FORMAT(6I10)
            #                 Field 1       -- element label
            #                 Field 2       -- fe descriptor id
            #                 Field 3       -- physical property table number
            #                 Field 4       -- material property table number
            #                 Field 5       -- color
            #                 Field 6       -- number of nodes on element
            element_label, fe_descriptor_id, physical_property_table_number, material_property_table_number, color, number_nodes = (
                parse_uff_line(data[index], 6 * ['I10']))
            if fe_descriptor_id in _beam_elem_types:
                #               Record 2:  *** FOR BEAM ELEMENTS ONLY ***
                #                 FORMAT(3I10)
                #                 Field 1       -- beam orientation node number
                #                 Field 2       -- beam fore-end cross section number
                #                 Field 3       -- beam  aft-end cross section number
                beam_orientation, beam_fore_cross_section_number, beam_aft_cross_section_number = parse_uff_line(
                    data[index + 1], 3 * ['I10'])
#               Record 3:  *** FOR BEAM ELEMENTS ONLY ***
#                 FORMAT(8I10)
#                 Fields 1-n    -- node labels defining element
                connectivity, read_lines = parse_uff_lines(
                    data[index + 2:], 8 * ['I10'], number_nodes)
                index += read_lines + 2
            else:
                #               Record 2:  *** FOR NON-BEAM ELEMENTS ***
                #                 FORMAT(8I10)
                #                 Fields 1-n    -- node labels defining element
                beam_orientation = 0
                beam_fore_cross_section_number = 0
                beam_aft_cross_section_number = 0
                connectivity, read_lines = parse_uff_lines(
                    data[index + 1:], 8 * ['I10'], number_nodes)
                index += read_lines + 1
            element_labels.append(element_label)
            fe_descriptor_ids.append(fe_descriptor_id)
            physical_property_table_numbers.append(physical_property_table_number)
            material_property_table_numbers.append(material_property_table_number)
            colors.append(color)
            connectivities.append(connectivity)
            beam_orientations.append(beam_orientation)
            beam_fore_cross_section_numbers.append(beam_fore_cross_section_number)
            beam_aft_cross_section_numbers.append(beam_aft_cross_section_number)
        # Make sure beam_props and connectivity don't end up being 2D
        ds_2412 = cls(element_labels,
                      fe_descriptor_ids,
                      physical_property_table_numbers,
                      material_property_table_numbers,
                      colors,
                      connectivities,
                      beam_orientations,
                      beam_fore_cross_section_numbers,
                      beam_aft_cross_section_numbers)
        return ds_2412

    def __repr__(self):
        return 'Sdynpy_UFF_Dataset_2414<{} element(s)>'.format(len(self.element_labels))

    def write_string(self):
        return_string = ''
        for (element_label, fe_descriptor_id, physical_property_table_number,
             material_property_table_number, color, connectivity, beam_orientation,
             beam_fore_cross_section_number, beam_aft_cross_section_number) in zip(
                self.element_labels,
                self.fe_descriptor_ids,
                self.physical_property_table_numbers,
                self.material_property_table_numbers,
                self.colors,
                self.connectivities,
                self.beam_orientations,
                self.beam_fore_cross_section_numbers,
                self.beam_aft_cross_section_numbers):
            return_string += write_uff_line([element_label, fe_descriptor_id,
                                             physical_property_table_number,
                                             material_property_table_number,
                                             color, len(connectivity)],
                                            6 * ['I10'])
            if fe_descriptor_id in _beam_elem_types:
                return_string += write_uff_line(
                    [beam_orientation, beam_fore_cross_section_number, beam_aft_cross_section_number], 3 * ['I10'])
            return_string += write_uff_line(connectivity, 8 * ['I10'])
        return return_string

    def __str__(self):
        lines = self.write_string().split('\n')
        if len(lines) > 8:
            return 'Dataset 2412: Elements\n  ' + '\n  '.join(lines[0:5] + ['.', '.', '.'])
        else:
            return 'Dataset 2412: Elements\n  ' + '\n  '.join(lines)


def read(data):
    return Sdynpy_UFF_Dataset_2412.from_uff_data_array(data)
