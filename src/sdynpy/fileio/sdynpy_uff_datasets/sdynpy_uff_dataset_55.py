# -*- coding: utf-8 -*-
"""
Data at Nodes (Shapes)

This dataset defines mode shapes, which are defined as data at nodes.

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


class Sdynpy_UFF_Dataset_55:
    def __init__(self, idline1, idline2, idline3, idline4, idline5, model_type,
                 analysis_type, data_characteristic, specific_data_type, data_type,
                 integer_data, real_data, node_data_dictionary):
        self.idline1 = idline1
        self.idline2 = idline2
        self.idline3 = idline3
        self.idline4 = idline4
        self.idline5 = idline5
        self.model_type = model_type
        self.analysis_type = analysis_type
        self.data_characteristic = data_characteristic
        self.specific_data_type = specific_data_type
        self.data_type = data_type
        self.integer_data = integer_data
        self.real_data = real_data
        self.node_data_dictionary = node_data_dictionary

    @property
    def dataset_number(self):
        return 55

    @property
    def id_number(self):
        return self.integer_data[0]

    @id_number.setter
    def id_number(self, value):
        index = 0
        try:
            self.integer_data[index] = value
        except IndexError:
            print('Integer data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.integer_data[index]), index, index + 1))
            self.integer_data += ((index + 1) - (len(self.integer_data))) * [0]
            self.integer_data[index] = value

    @property
    def load_case_number(self):
        return self.integer_data[0]

    @load_case_number.setter
    def load_case_number(self, value):
        index = 0
        try:
            self.integer_data[index] = value
        except IndexError:
            print('Integer data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.integer_data[index]), index, index + 1))
            self.integer_data += ((index + 1) - (len(self.integer_data))) * [0]
            self.integer_data[index] = value

    @property
    def mode_number(self):
        return self.integer_data[1]

    @mode_number.setter
    def mode_number(self, value):
        index = 1
        try:
            self.integer_data[index] = value
        except IndexError:
            print('Integer data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.integer_data[index]), index, index + 1))
            self.integer_data += ((index + 1) - (len(self.integer_data))) * [0]
            self.integer_data[index] = value

    @property
    def time_step_number(self):
        return self.integer_data[1]

    @time_step_number.setter
    def time_step_number(self, value):
        index = 1
        try:
            self.integer_data[index] = value
        except IndexError:
            print('Integer data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.integer_data[index]), index, index + 1))
            self.integer_data += ((index + 1) - (len(self.integer_data))) * [0]
            self.integer_data[index] = value

    @property
    def frequency_step_number(self):
        return self.integer_data[1]

    @frequency_step_number.setter
    def frequency_step_number(self, value):
        index = 1
        try:
            self.integer_data[index] = value
        except IndexError:
            print('Integer data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.integer_data[index]), index, index + 1))
            self.integer_data += ((index + 1) - (len(self.integer_data))) * [0]
            self.integer_data[index] = value

    @property
    def frequency(self):
        return self.real_data[0]

    @frequency.setter
    def frequency(self, value):
        index = 0
        try:
            self.real_data[index] = value
        except IndexError:
            print('Real data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.real_data[index]), index, index + 1))
            self.real_data += ((index + 1) - (len(self.real_data))) * [0]
            self.real_data[index] = value

    @property
    def modal_mass(self):
        return self.real_data[1]

    @modal_mass.setter
    def modal_mass(self, value):
        index = 1
        try:
            self.real_data[index] = value
        except IndexError:
            print('Real data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.real_data[index]), index, index + 1))
            self.real_data += ((index + 1) - (len(self.real_data))) * [0]
            self.real_data[index] = value

    @property
    def modal_viscous_damping(self):
        return self.real_data[2]

    @modal_viscous_damping.setter
    def modal_viscous_damping(self, value):
        index = 2
        try:
            self.real_data[index] = value
        except IndexError:
            print('Real data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.real_data[index]), index, index + 1))
            self.real_data += ((index + 1) - (len(self.real_data))) * [0]
            self.real_data[index] = value

    @property
    def modal_hysteretic_damping(self):
        return self.real_data[3]

    @modal_hysteretic_damping.setter
    def modal_hysteretic_damping(self, value):
        index = 3
        try:
            self.real_data[index] = value
        except IndexError:
            print('Real data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.real_data[index]), index, index + 1))
            self.real_data += ((index + 1) - (len(self.real_data))) * [0]
            self.real_data[index] = value

    @property
    def real_eigenvalue(self):
        return self.real_data[0]

    @real_eigenvalue.setter
    def real_eigenvalue(self, value):
        index = 0
        try:
            self.real_data[index] = value
        except IndexError:
            print('Real data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.real_data[index]), index, index + 1))
            self.real_data += ((index + 1) - (len(self.real_data))) * [0]
            self.real_data[index] = value

    @property
    def imaginary_eigenvalue(self):
        return self.real_data[1]

    @imaginary_eigenvalue.setter
    def imaginary_eigenvalue(self, value):
        index = 1
        try:
            self.real_data[index] = value
        except IndexError:
            print('Real data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.real_data[index]), index, index + 1))
            self.real_data += ((index + 1) - (len(self.real_data))) * [0]
            self.real_data[index] = value

    @property
    def real_modal_a(self):
        return self.real_data[2]

    @real_modal_a.setter
    def real_modal_a(self, value):
        index = 2
        try:
            self.real_data[index] = value
        except IndexError:
            print('Real data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.real_data[index]), index, index + 1))
            self.real_data += ((index + 1) - (len(self.real_data))) * [0]
            self.real_data[index] = value

    @property
    def imaginary_modal_a(self):
        return self.real_data[3]

    @imaginary_modal_a.setter
    def imaginary_modal_a(self, value):
        index = 3
        try:
            self.real_data[index] = value
        except IndexError:
            print('Real data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.real_data[index]), index, index + 1))
            self.real_data += ((index + 1) - (len(self.real_data))) * [0]
            self.real_data[index] = value

    @property
    def real_modal_b(self):
        return self.real_data[4]

    @real_modal_b.setter
    def real_modal_b(self, value):
        index = 4
        try:
            self.real_data[index] = value
        except IndexError:
            print('Real data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.real_data[index]), index, index + 1))
            self.real_data += ((index + 1) - (len(self.real_data))) * [0]
            self.real_data[index] = value

    @property
    def imaginary_modal_b(self):
        return self.real_data[5]

    @imaginary_modal_b.setter
    def imaginary_modal_b(self, value):
        index = 5
        try:
            self.real_data[index] = value
        except IndexError:
            print('Real data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.real_data[index]), index, index + 1))
            self.real_data += ((index + 1) - (len(self.real_data))) * [0]
            self.real_data[index] = value

    @property
    def time(self):
        return self.real_data[0]

    @time.setter
    def time(self, value):
        index = 0
        try:
            self.real_data[index] = value
        except IndexError:
            print('Real data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.real_data[index]), index, index + 1))
            self.real_data += ((index + 1) - (len(self.real_data))) * [0]
            self.real_data[index] = value

    @property
    def buckling_eigenvalue(self):
        return self.real_data[0]

    @buckling_eigenvalue.setter
    def buckling_eigenvalue(self, value):
        index = 0
        try:
            self.real_data[index] = value
        except IndexError:
            print('Real data currently only has {:} indices, tried to index {:}, expanding to {:}'.format(
                len(self.real_data[index]), index, index + 1))
            self.real_data += ((index + 1) - (len(self.real_data))) * [0]
            self.real_data[index] = value

    @classmethod
    def from_uff_data_array(cls, data):
        #        RECORD 1:      Format (40A2)
        #               FIELD 1:          ID Line 1
        idline1, = parse_uff_line(data[0], ['A80'])
#          RECORD 2:      Format (40A2)
#               FIELD 1:          ID Line 2
        idline2, = parse_uff_line(data[1], ['A80'])
#          RECORD 3:      Format (40A2)
#
#               FIELD 1:          ID Line 3
        idline3, = parse_uff_line(data[2], ['A80'])
#          RECORD 4:      Format (40A2)
#               FIELD 1:          ID Line 4
        idline4, = parse_uff_line(data[3], ['A80'])
#          RECORD 5:      Format (40A2)
#               FIELD 1:          ID Line 5
        idline5, = parse_uff_line(data[4], ['A80'])
#          RECORD 6:      Format (6I10)
#
#          Data Definition Parameters
#
#               FIELD 1: Model Type
#                           0:   Unknown
#                           1:   Structural
#                           2:   Heat Transfer
#                           3:   Fluid Flow
#
#               FIELD 2:  Analysis Type
#                           0:   Unknown
#                           1:   Static
#                           2:   Normal Mode
#                           3:   Complex eigenvalue first order
#                           4:   Transient
#                           5:   Frequency Response
#                           6:   Buckling
#                           7:   Complex eigenvalue second order
#
#               FIELD 3:  Data Characteristic
#                           0:   Unknown
#                           1:   Scalar
#                           2:   3 DOF Global Translation
#                                Vector
#                           3:   6 DOF Global Translation
#                                & Rotation Vector
#                           4:   Symmetric Global Tensor
#                           5:   General Global Tensor
#
#
#
#               FIELD 4: Specific Data Type
#                           0:   Unknown
#                           1:   General
#                           2:   Stress
#                           3:   Strain (Engineering)
#                           4:   Element Force
#                           5:   Temperature
#                           6:   Heat Flux
#                           7:   Strain Energy
#                           8:   Displacement
#                           9:   Reaction Force
#                           10:   Kinetic Energy
#                           11:   Velocity
#                           12:   Acceleration
#                           13:   Strain Energy Density
#                           14:   Kinetic Energy Density
#                           15:   Hydro-Static Pressure
#                           16:   Heat Gradient
#                           17:   Code Checking Value
#                           18:   Coefficient Of Pressure
#
#               FIELD 5:  Data Type
#                           2:   Real
#                           5:   Complex
#
#               FIELD 6:  Number Of Data Values Per Node (NDV)
        model_type, analysis_type, data_characteristic, specific_data_type, data_type, num_data_per_node = (
            parse_uff_line(data[5], 6 * ['I10']))
        data_is_complex = data_type == 5
#          Records 7 And 8 Are Analysis Type Specific
#
#          General Form
#
#          RECORD 7:      Format (8I10)
#
#               FIELD 1:          Number Of Integer Data Values
#                           1 < Or = Nint < Or = 10
#               FIELD 2:          Number Of Real Data Values
#                           1 < Or = Nrval < Or = 12
#               FIELDS 3-N:       Type Specific Integer Parameters
#
#
#          RECORD 8:      Format (6E13.5)
#               FIELDS 1-N:       Type Specific Real Parameters
        # Initially just read the number of data that we will get
        nints, nreals = parse_uff_line(data[6], 2 * ['I10'])
        # Now read the whole of record 7
        int_data, lines_read = parse_uff_lines(data[6:], 8 * ['I10'], 2 + nints)
        next_line = 6 + lines_read
        integer_data = int_data[2:]
        real_data, lines_read = parse_uff_lines(data[next_line:], 6 * ['E13.5'], nreals)
        next_line += lines_read
#          RECORD 9:      Format (I10)
#
#               FIELD 1:          Node Number
#
#          RECORD 10:     Format (6E13.5)
#               FIELDS 1-N:       Data At This Node (NDV Real Or
#                         Complex Values)
#
#          Records 9 And 10 Are Repeated For Each Node.
        node_data_dictionary = {}
        while next_line < len(data):
            node, = parse_uff_line(data[next_line], ['I10'])
            next_line += 1
            node_data, read_lines = parse_uff_lines(
                data[next_line:], 6 * ['E13.5'], num_data_per_node * (2 if data_is_complex else 1))
            next_line += read_lines
            if data_is_complex:
                node_data_dictionary[node] = [real + 1j *
                                              imag for real, imag in zip(data[::2], data[1::2])]
            else:
                node_data_dictionary[node] = node_data

        ds_55 = cls(idline1, idline2, idline3, idline4, idline5, model_type,
                    analysis_type, data_characteristic, specific_data_type, data_type,
                    integer_data, real_data, node_data_dictionary)
        return ds_55

    def __repr__(self):
        return 'Sdynpy_UFF_Dataset_55<Shape with {:} Nodes>'.format(len(self.node_data_dictionary))

    def write_string(self):
        return_string = ''
        # See how many values are in each node
        nodes = [key for key in self.node_data_dictionary.keys()]
        if self.data_type == 5:
            num_data = int(len(self.node_data_dictionary[nodes[0]]) // 2)
        else:
            num_data = int(len(self.node_data_dictionary[nodes[0]]))
        return_string += write_uff_line(['NONE' if self.idline1 == '' else self.idline1,
                                         'NONE' if self.idline2 == '' else self.idline2,
                                         'NONE' if self.idline3 == '' else self.idline3,
                                         'NONE' if self.idline4 == '' else self.idline4,
                                         'NONE' if self.idline5 == '' else self.idline5,
                                         ], ['A80'])
        return_string += write_uff_line([self.model_type,
                                         self.analysis_type,
                                         self.data_characteristic,
                                         self.specific_data_type,
                                         self.data_type,
                                         num_data], 6 * ['I10'])
        nint = len(self.integer_data)
        nfloat = len(self.real_data)
        return_string += write_uff_line([nint, nfloat] + self.integer_data,
                                        8 * ['I10'])
        return_string += write_uff_line(self.real_data, 6 * ['E13.5'])
        for node in sorted(self.node_data_dictionary.keys()):
            return_string += write_uff_line([node], ['I10'])
            return_string += write_uff_line(
                [fn(val) for val in self.node_data_dictionary[node] for fn in (np.real,np.imag)] 
                if np.iscomplexobj(self.node_data_dictionary[node])
                else self.node_data_dictionary[node], 6 * ['E13.5'])
        return return_string

    def __str__(self):
        lines = self.write_string().split('\n')
        if len(lines) > 8:
            return 'Dataset 55: Shapes\n  ' + '\n  '.join(lines[0:5] + ['.', '.', '.'])
        else:
            return 'Dataset 55: Shapes\n  ' + '\n  '.join(lines)


def read(data):
    return Sdynpy_UFF_Dataset_55.from_uff_data_array(data)
