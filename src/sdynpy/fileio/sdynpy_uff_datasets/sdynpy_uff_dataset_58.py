# -*- coding: utf-8 -*-
"""
Functions at Nodal Degrees of Freedom

Defines several types of functions from time histories to spectra, spectral
densities, and transfer functions.

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

from ..sdynpy_uff import parse_uff_line, parse_uff_lines, write_uff_line
from ...core.sdynpy_coordinate import parse_coordinate_string
import numpy as np
import struct
import warnings

def is_abscissa_even(abscissa):
    abscissa_inc = np.mean(np.diff(abscissa))
    abscissa_start = abscissa[0]
    return np.allclose(abscissa, abscissa_start + abscissa_inc * np.arange(abscissa.size))


class Sdynpy_UFF_Dataset_58:
    def __init__(self, idline1, idline2, idline3, idline4, idline5,
                 function_type, function_id, version_number, load_case,
                 response_entity_name, response_node, response_direction,
                 reference_entity_name, reference_node, reference_direction,
                 abscissa_data_type, abscissa_length_exponent,
                 abscissa_force_exponent, abscissa_temp_exponent,
                 abscissa_axis_label, abscissa_units_label,
                 ordinate_num_data_type, ordinate_num_length_exponent,
                 ordinate_num_force_exponent, ordinate_num_temp_exponent,
                 ordinate_num_axis_label, ordinate_num_units_label,
                 ordinate_den_data_type, ordinate_den_length_exponent,
                 ordinate_den_force_exponent, ordinate_den_temp_exponent,
                 ordinate_den_axis_label, ordinate_den_units_label,
                 zaxis_data_type, zaxis_length_exponent,
                 zaxis_force_exponent, zaxis_temp_exponent,
                 zaxis_axis_label, zaxis_units_label, zaxis_value,
                 abscissa, ordinate
                 ):
        self.idline1 = idline1
        self.idline2 = idline2
        self.idline3 = idline3
        self.idline4 = idline4
        self.idline5 = idline5
        self.function_type = function_type
        self.function_id = function_id
        self.version_number = version_number
        self.load_case = load_case
        self.response_entity_name = response_entity_name
        self.response_node = response_node
        self.response_direction = response_direction
        self.reference_entity_name = reference_entity_name
        self.reference_node = reference_node
        self.reference_direction = reference_direction
        self.abscissa_data_type = abscissa_data_type
        self.abscissa_length_exponent = abscissa_length_exponent
        self.abscissa_force_exponent = abscissa_force_exponent
        self.abscissa_temp_exponent = abscissa_temp_exponent
        self.abscissa_axis_label = abscissa_axis_label
        self.abscissa_units_label = abscissa_units_label
        self.ordinate_num_data_type = ordinate_num_data_type
        self.ordinate_num_length_exponent = ordinate_num_length_exponent
        self.ordinate_num_force_exponent = ordinate_num_force_exponent
        self.ordinate_num_temp_exponent = ordinate_num_temp_exponent
        self.ordinate_num_axis_label = ordinate_num_axis_label
        self.ordinate_num_units_label = ordinate_num_units_label
        self.ordinate_den_data_type = ordinate_den_data_type
        self.ordinate_den_length_exponent = ordinate_den_length_exponent
        self.ordinate_den_force_exponent = ordinate_den_force_exponent
        self.ordinate_den_temp_exponent = ordinate_den_temp_exponent
        self.ordinate_den_axis_label = ordinate_den_axis_label
        self.ordinate_den_units_label = ordinate_den_units_label
        self.zaxis_data_type = zaxis_data_type
        self.zaxis_length_exponent = zaxis_length_exponent
        self.zaxis_force_exponent = zaxis_force_exponent
        self.zaxis_temp_exponent = zaxis_temp_exponent
        self.zaxis_axis_label = zaxis_axis_label
        self.zaxis_units_label = zaxis_units_label
        self.zaxis_value = zaxis_value
        self.abscissa = abscissa
        self.ordinate = ordinate

    @property
    def dataset_number(self):
        return 58

    @classmethod
    def from_uff_data_array(cls, block_lines, is_binary, byte_ordering,
                            floating_point_format, num_ascii_lines_following,
                            num_bytes_following):
        """
        Extract function at nodal DOF - data-set 58.

        Returns
        -------
        ds_58
            Object with attributes as the dataset fields and values containing the
            data from the universal file in those dataset fields.
        """
        if num_ascii_lines_following is None:
            num_ascii_lines_following = 11
        split_header = [line.decode('utf-8') for line in block_lines[:num_ascii_lines_following]]
        #        Record 1:     Format(80A1)
        #                       Field 1    - ID Line 1
        #
        #                                                 NOTE
        #
        #                           ID Line 1 is generally  used  for  the  function
        #                           description.
        idline1, = parse_uff_line(split_header[0], ['A80'])
#        Record 2:     Format(80A1)
#                       Field 1    - ID Line 2
        idline2, = parse_uff_line(split_header[1], ['A80'])
#        Record 3:     Format(80A1)
#                       Field 1    - ID Line 3
#
#                                                 NOTE
#
#                           ID Line 3 is generally used to identify when the
#                           function  was  created.  The date is in the form
#                           DD-MMM-YY, and the time is in the form HH:MM:SS,
#                           with a general Format(9A1,1X,8A1).
        idline3, = parse_uff_line(split_header[2], ['A80'])
#        Record 4:     Format(80A1)
#                       Field 1    - ID Line 4
        idline4, = parse_uff_line(split_header[3], ['A80'])
#         Record 5:     Format(80A1)
#                       Field 1    - ID Line 5
        idline5, = parse_uff_line(split_header[4], ['A80'])
#        Record 6:     Format(2(I5,I10),2(1X,10A1,I10,I4))
#                                  DOF Identification
#                       Field 1    - Function Type
#                                    0 - General or Unknown
#                                    1 - Time Response
#                                    2 - Auto Spectrum
#                                    3 - Cross Spectrum
#                                    4 - Frequency Response Function
#                                    5 - Transmissibility
#                                    6 - Coherence
#                                    7 - Auto Correlation
#                                    8 - Cross Correlation
#                                    9 - Power Spectral Density (PSD)
#                                    10 - Energy Spectral Density (ESD)
#                                    11 - Probability Density Function
#                                    12 - Spectrum
#                                    13 - Cumulative Frequency Distribution
#                                    14 - Peaks Valley
#                                    15 - Stress/Cycles
#                                    16 - Strain/Cycles
#                                    17 - Orbit
#                                    18 - Mode Indicator Function
#                                    19 - Force Pattern
#                                    20 - Partial Power
#                                    21 - Partial Coherence
#                                    22 - Eigenvalue
#                                    23 - Eigenvector
#                                    24 - Shock Response Spectrum
#                                    25 - Finite Impulse Response Filter
#                                    26 - Multiple Coherence
#                                    27 - Order Function
#                                    28 - Phase Compensation
#                       Field 2    - Function Identification Number
#                       Field 3    - Version Number, or sequence number
#                       Field 4    - Load Case Identification Number
#                                    0 - Single Point Excitation
#                       Field 5    - Response Entity Name ("NONE" if unused)
#                       Field 6    - Response Node
#                       Field 7    - Response Direction
#                                     0 - Scalar
#                                     1 - +X Translation       4 - +X Rotation
#                                    -1 - -X Translation      -4 - -X Rotation
#                                     2 - +Y Translation       5 - +Y Rotation
#                                    -2 - -Y Translation      -5 - -Y Rotation
#                                     3 - +Z Translation       6 - +Z Rotation
#                                    -3 - -Z Translation      -6 - -Z Rotation
#                       Field 8    - Reference Entity Name ("NONE" if unused)
#                       Field 9    - Reference Node
#                       Field 10   - Reference Direction  (same as field 7)
#
#                                                 NOTE
#
#                           Fields 8, 9, and 10 are only relevant if field 4
#                           is zero.
        (function_type, function_id, version_number, load_case,
         response_entity_name, response_node, response_direction,
         reference_entity_name, reference_node, reference_direction) = (
            parse_uff_line(split_header[5], 2 * ['I5', 'I10'] + 2 * ['X1', 'A10', 'I10', 'I4']))

#        Record 7:     Format(3I10,3E13.5)
#                                  Data Form
#                       Field 1    - Ordinate Data Type
#                                    2 - real, single precision
#                                    4 - real, double precision
#                                    5 - complex, single precision
#                                    6 - complex, double precision
#                       Field 2    - Number of data pairs for uneven  abscissa
#                                    spacing, or number of data values for even
#                                    abscissa spacing
#                       Field 3    - Abscissa Spacing
#                                    0 - uneven
#                                    1 - even (no abscissa values stored)
#                       Field 4    - Abscissa minimum (0.0 if spacing uneven)
#                       Field 5    - Abscissa increment (0.0 if spacing uneven)
#                       Field 6    - Z-axis value (0.0 if unused)
        (ordinate_data_type, num_data, abscissa_spacing, abscissa_minimum,
         abscissa_increment, zaxis_value) = (
            parse_uff_line(split_header[6], 3 * ['I10'] + 3 * ['E13.5']))
#        Record 8:     Format(I10,3I5,2(1X,20A1))
#                                  Abscissa Data Characteristics
#                       Field 1    - Specific Data Type
#                                    0 - unknown
#                                    1 - general
#                                    2 - stress
#                                    3 - strain
#                                    5 - temperature
#                                    6 - heat flux
#                                    8 - displacement
#                                    9 - reaction force
#                                    11 - velocity
#                                    12 - acceleration
#                                    13 - excitation force
#                                    15 - pressure
#                                    16 - mass
#                                    17 - time
#                                    18 - frequency
#                                    19 - rpm
#                                    20 - order
#                                    21 - sound pressure
#                                    22 - sound intensity
#                                    23 - sound power
#                       Field 2    - Length units exponent
#                       Field 3    - Force units exponent
#                       Field 4    - Temperature units exponent
#
#                                                 NOTE
#
#                           Fields 2, 3 and  4  are  relevant  only  if  the
#                           Specific Data Type is General, or in the case of
#                           ordinates, the response/reference direction is a
#                           scalar, or the functions are being used for
#                           nonlinear connectors in System Dynamics Analysis.
#                           See Addendum 'A' for the units exponent table.
#
#                       Field 5    - Axis label ("NONE" if not used)
#                       Field 6    - Axis units label ("NONE" if not used)
#
#                                                 NOTE
#
#                           If fields  5  and  6  are  supplied,  they  take
#                           precendence  over  program  generated labels and
#                           units.
        (abscissa_data_type, abscissa_length_exponent,
         abscissa_force_exponent, abscissa_temp_exponent,
         abscissa_axis_label, abscissa_units_label) = (
            parse_uff_line(split_header[7], ['I10'] + 3 * ['I5'] + 2 * ['X1', 'A20']))
#         Record 9:     Format(I10,3I5,2(1X,20A1))
#                       Ordinate (or ordinate numerator) Data Characteristics
        (ordinate_num_data_type, ordinate_num_length_exponent,
         ordinate_num_force_exponent, ordinate_num_temp_exponent,
         ordinate_num_axis_label, ordinate_num_units_label) = (
            parse_uff_line(split_header[8], ['I10'] + 3 * ['I5'] + 2 * ['X1', 'A20']))
#         Record 10:    Format(I10,3I5,2(1X,20A1))
#                       Ordinate Denominator Data Characteristics
        (ordinate_den_data_type, ordinate_den_length_exponent,
         ordinate_den_force_exponent, ordinate_den_temp_exponent,
         ordinate_den_axis_label, ordinate_den_units_label) = (
            parse_uff_line(split_header[9], ['I10'] + 3 * ['I5'] + 2 * ['X1', 'A20']))
#         Record 11:    Format(I10,3I5,2(1X,20A1))
#                       Z-axis Data Characteristics
        (zaxis_data_type, zaxis_length_exponent,
         zaxis_force_exponent, zaxis_temp_exponent,
         zaxis_axis_label, zaxis_units_label) = (
            parse_uff_line(split_header[10], ['I10'] + 3 * ['I5'] + 2 * ['X1', 'A20']))
#         Record 12:
#                                   Data Values
#
#                         Ordinate            Abscissa
#             Case     Type     Precision     Spacing       Format
#           -------------------------------------------------------------
#               1      real      single        even         6E13.5
#               2      real      single       uneven        6E13.5
#               3     complex    single        even         6E13.5
#               4     complex    single       uneven        6E13.5
#               5      real      double        even         4E20.12
#               6      real      double       uneven     2(E13.5,E20.12)
#               7     complex    double        even         4E20.12
#               8     complex    double       uneven      E13.5,2E20.12
#           --------------------------------------------------------------
        
        # Number of data points to read per item
        read_multiplication_factor = 1  # Base value is just one value
        if abscissa_spacing == 0:       # If abscissa spacing is uneven
            read_multiplication_factor += 1  # Will need to read that too
        if ordinate_data_type in [5, 6]:  # If complex,
            read_multiplication_factor += 1  # Will need to read 2 ordinates

        # reading Record 12 if encoded in binary
        if is_binary:
            split_data = b''.join(block_lines[num_ascii_lines_following:])
            if byte_ordering == 1:
                bo = '<'
            elif byte_ordering == 2:
                bo = '>'
            else:
                warnings.warn('UFF file does not contain byte_ordering parameter, assuming Little Endian (DEC VMS & ULTRIX, WIN NT)', EncodingWarning, stacklevel=5)
                bo = '<'
            if (ordinate_data_type in [2, 5]):
                # single precision - 4 bytes
                ordinate = np.asarray(struct.unpack('%c%sf' % (bo, int(len(split_data) / 4)), split_data), 'd').reshape(-1,read_multiplication_factor)
            else:
                # double precision - 8 bytes
                ordinate = np.asarray(struct.unpack('%c%sd' % (bo, int(len(split_data) / 8)), split_data), 'd').reshape(-1,read_multiplication_factor)

        # reading Record 12 if encoded as ascii
        else:
            if (ordinate_data_type in [2, 5]):   # Single, regardless of abscissa or complexity
                read_format = 6 * ['E13.5']
                ordinate_data_dtype = 'float32'
            elif (ordinate_data_type in [4, 6]   # Double precision
                and abscissa_spacing == 1):   # Even spacing
                read_format = 4 * ['E20.12']
                ordinate_data_dtype = 'float64'
            elif (ordinate_data_type == 4       # Real double precision
                and abscissa_spacing == 0):   # Uneven spacing
                read_format = 2 * ['E13.5', 'E20.12']
                ordinate_data_dtype = 'float64'
            elif (ordinate_data_type == 6       # Complex double precision
                and abscissa_spacing == 0):   # Uneven spacing
                read_format = ['E13.5'] + 2 * ['E20.12']
                ordinate_data_dtype = 'float64'
            total_number_of_entries = read_multiplication_factor * num_data
            # Now read the data
            ordinate = np.array(
                parse_uff_lines([line.decode('utf-8') for line in block_lines[num_ascii_lines_following:]], read_format, total_number_of_entries)[0],
                dtype=ordinate_data_dtype).reshape(-1, read_multiplication_factor)
        # Now parse the data into the right format
        if abscissa_spacing == 0:  # If abscissa spacing is uneven
            abscissa = ordinate[:, 0].astype('float32')
            ordinate = ordinate[:, 1:]
        else:
            abscissa = (abscissa_minimum + abscissa_increment *
                        np.arange(ordinate.shape[0])).astype('float32')
        if ordinate_data_type in [5, 6]:  # If complex
            ordinate = ordinate[:, 0] + 1j * ordinate[:, 1]
        else:
            ordinate = ordinate[:, 0]
        # Now create the dataset
        ds_58 = cls(idline1, idline2, idline3, idline4, idline5,
                    function_type, function_id, version_number, load_case,
                    response_entity_name, response_node, response_direction,
                    reference_entity_name, reference_node, reference_direction,
                    abscissa_data_type, abscissa_length_exponent,
                    abscissa_force_exponent, abscissa_temp_exponent,
                    abscissa_axis_label, abscissa_units_label,
                    ordinate_num_data_type, ordinate_num_length_exponent,
                    ordinate_num_force_exponent, ordinate_num_temp_exponent,
                    ordinate_num_axis_label, ordinate_num_units_label,
                    ordinate_den_data_type, ordinate_den_length_exponent,
                    ordinate_den_force_exponent, ordinate_den_temp_exponent,
                    ordinate_den_axis_label, ordinate_den_units_label,
                    zaxis_data_type, zaxis_length_exponent,
                    zaxis_force_exponent, zaxis_temp_exponent,
                    zaxis_axis_label, zaxis_units_label, zaxis_value,
                    abscissa, ordinate)
        return ds_58

    def __repr__(self):
        return 'Sdynpy_UFF_Dataset_58<{} Elements in Function>'.format(self.ordinate.size)

    def write_string(self):
        return_string = ''
        return_string += write_uff_line([self.idline1,
                                         self.idline2,
                                         self.idline3,
                                         self.idline4,
                                         self.idline5,
                                         ], ['A80'])
        return_string += write_uff_line([self.function_type,
                                         self.function_id,
                                         self.version_number,
                                         self.load_case,
                                         self.response_entity_name,
                                         self.response_node,
                                         self.response_direction,
                                         self.reference_entity_name,
                                         self.reference_node,
                                         self.reference_direction],
                                        2 * ['I5', 'I10'] + 2 * ['X1', 'A10', 'I10', 'I4'])
        # Check what type of data
        type_size = self.ordinate.dtype.itemsize
        if np.iscomplexobj(self.ordinate):  # Complex
            if type_size == 16:  # Double Precision
                ordinate_data_type = 6
            elif type_size == 8:  # Single Precision
                ordinate_data_type = 5
            else:
                raise ValueError(
                    'Unknown data type to write to UFF file!  Should be one of "float32", "float64", "complex64", "complex128".')
            ordinate = np.concatenate((self.ordinate.real.reshape(-1, 1),
                                      self.ordinate.imag.reshape(-1, 1)), axis=1)
        elif np.isrealobj(self.ordinate):  # Real
            if type_size == 8:  # Double precision
                ordinate_data_type = 4
            elif type_size == 4:  # Single Precision
                ordinate_data_type = 2
            else:
                raise ValueError(
                    'Unknown data type to write to UFF file!  Should be one of "float32", "float64", "complex64", "complex128".')
            ordinate = self.ordinate.reshape(-1, 1)
        else:
            raise ValueError(
                'Unknown data type to write to UFF file!  Should be one of "float32", "float64", "complex64", "complex128".')
        # Check if abscissa is even
        if is_abscissa_even(self.abscissa):
            abscissa_spacing = 1
            abscissa_minimum = self.abscissa[0]
            abscissa_increment = np.mean(np.diff(self.abscissa))
        else:
            abscissa_spacing = 0
            abscissa_minimum = 0
            abscissa_increment = 0
            ordinate = np.concatenate((self.abscissa.reshape(-1, 1), ordinate), axis=1)
        number_data = ordinate.shape[0]
        return_string += write_uff_line([ordinate_data_type, number_data, abscissa_spacing,
                                         abscissa_minimum, abscissa_increment, self.zaxis_value], 3 * ['I10'] + 3 * ['E13.5'])
        return_string += write_uff_line([self.abscissa_data_type,
                                         self.abscissa_length_exponent,
                                         self.abscissa_force_exponent,
                                         self.abscissa_temp_exponent,
                                         self.abscissa_axis_label,
                                         self.abscissa_units_label], ['I10'] + 3 * ['I5'] + 2 * ['X1', 'A20'])
        return_string += write_uff_line([self.ordinate_num_data_type,
                                         self.ordinate_num_length_exponent,
                                         self.ordinate_num_force_exponent,
                                         self.ordinate_num_temp_exponent,
                                         self.ordinate_num_axis_label,
                                         self.ordinate_num_units_label], ['I10'] + 3 * ['I5'] + 2 * ['X1', 'A20'])
        return_string += write_uff_line([self.ordinate_den_data_type,
                                         self.ordinate_den_length_exponent,
                                         self.ordinate_den_force_exponent,
                                         self.ordinate_den_temp_exponent,
                                         self.ordinate_den_axis_label,
                                         self.ordinate_den_units_label], ['I10'] + 3 * ['I5'] + 2 * ['X1', 'A20'])
        return_string += write_uff_line([self.zaxis_data_type,
                                         self.zaxis_length_exponent,
                                         self.zaxis_force_exponent,
                                         self.zaxis_temp_exponent,
                                         self.zaxis_axis_label,
                                         self.zaxis_units_label], ['I10'] + 3 * ['I5'] + 2 * ['X1', 'A20'])
        # Get write format
        if (ordinate_data_type in [2, 5]):   # Single, regardless of abscissa or complexity
            write_format = 6 * ['E13.5']
        elif (ordinate_data_type in [4, 6]   # Double precision
              and abscissa_spacing == 1):   # Even spacing
            write_format = 4 * ['E20.12']
        elif (ordinate_data_type == 4       # Real double precision
              and abscissa_spacing == 0):   # Uneven spacing
            write_format = 2 * ['E13.5', 'E20.12']
        elif (ordinate_data_type == 6       # Complex double precision
              and abscissa_spacing == 0):   # Uneven spacing
            write_format = ['E13.5'] + 2 * ['E20.12']
        return_string += write_uff_line(ordinate.flatten(), write_format)
        return return_string

    def __str__(self):
        lines = self.write_string().split('\n')
        if len(lines) > 8:
            return 'Dataset 58: Data\n  ' + '\n  '.join(lines[0:5] + ['.', '.', '.'])
        else:
            return 'Dataset 58: Data\n  ' + '\n  '.join(lines)


def read(data,is_binary=False, byte_ordering = None, floating_point_format = None,
         num_ascii_lines_following = None, num_bytes_following = None):
    return Sdynpy_UFF_Dataset_58.from_uff_data_array(
        data, is_binary, byte_ordering, floating_point_format,
        num_ascii_lines_following, num_bytes_following)
