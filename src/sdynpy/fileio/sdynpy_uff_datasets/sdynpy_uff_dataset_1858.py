# -*- coding: utf-8 -*-
"""
Dataset 58 Qualifiers

This dataset defines additional function information that is not included in
dataset 58.

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
import numpy as np


def is_abscissa_even(abscissa):
    abscissa_inc = np.mean(np.diff(abscissa))
    abscissa_start = abscissa[0]
    return np.allclose(abscissa, abscissa_start + abscissa_inc * np.arange(abscissa.size))


class Sdynpy_UFF_Dataset_1858:
    def __init__(self, set_record_number, octave_format, measurement_run,
                 weighting_type, window_type, amplitude_units, normalization_method,
                 abscissa_data_qualifier, ordinate_num_data_qualifier,
                 ordinate_den_data_qualifier, zaxis_data_qualifier,
                 sampling_type, time_average, zrpm, ztime, zorder, number_samples,
                 uservalue1, uservalue2, uservalue3, uservalue4,
                 exponential_window_damping_factor, overall_rms, weighted_rms,
                 response_direction, reference_direction):
        self.set_record_number = set_record_number
        self.octave_format = octave_format
        self.measurement_run = measurement_run
        self.weighting_type = weighting_type
        self.window_type = window_type
        self.amplitude_units = amplitude_units
        self.normalization_method = normalization_method
        self.abscissa_data_qualifier = abscissa_data_qualifier
        self.ordinate_num_data_qualifier = ordinate_num_data_qualifier
        self.ordinate_den_data_qualifier = ordinate_den_data_qualifier
        self.zaxis_data_qualifier = zaxis_data_qualifier
        self.sampling_type = sampling_type
        self.time_average = time_average
        self.zrpm = zrpm
        self.ztime = ztime
        self.zorder = zorder
        self.number_samples = number_samples
        self.uservalue1 = uservalue1
        self.uservalue2 = uservalue2
        self.uservalue3 = uservalue3
        self.uservalue4 = uservalue4
        self.exponential_window_damping_factor = exponential_window_damping_factor
        self.overall_rms = overall_rms
        self.weighted_rms = weighted_rms
        self.response_direction = response_direction
        self.reference_direction = reference_direction

    @property
    def dataset_number(self):
        return 1858

    @classmethod
    def from_uff_data_array(cls, data):
        # Transform from binary to ascii
        data = [line.decode() for line in data]
        #       Record 1:     FORMAT(6I12)
        #              Field 1       - Set record number
        #              Field 2       - Octave format
        #                              0 - not in octave format (default)
        #                              1 - octave
        #                              3 - one third octave
        #                              n - 1/n octave
        #              Field 3       - Measurement run number
        #              Fields 4-6    - Not used (fill with zeros)
        set_record_number, octave_format, measurement_run, not_used, not_used, not_used = parse_uff_line(
            data[0], 6 * ['I12'])
#        Record 2:     FORMAT(12I6)
#              Field 1       - Weighting Type
#                              0 - No weighting or Unknown (default)
#                              1 - A weighting
#                              2 - B weighting
#                              3 - C weighting
#                              4 - D weighting (not yet implemented)
#              Field 2       - Window Type
#                              0 - No window or unknown (default)
#                              1 - Hanning Narrow
#                              2 - Hanning Broad
#                              3 - Flattop
#                              4 - Exponential
#                              5 - Impact
#                              6 - Impact and Exponential
#              Field 3       - Amplitude units
#                              0 - unknown (default)
#                              1 - Half-peak scale
#                              2 - Peak scale
#                              3 - RMS
#              Field 4       - Normalization Method
#                              0 - unknown (default)
#                              1 - Units squared
#                              2 - Units squared per Hz (PSD)
#                              3 - Units squared seconds per Hz (ESD)
#              Field 5       - Abscissa Data Type Qualifier
#                              0 - Translation
#                              1 - Rotation
#                              2 - Translation Squared
#                              3 - Rotation Squared
#              Field 6       - Ordinate Numerator Data Type Qualifier
#                              0 - Translation
#                              1 - Rotation
#                              2 - Translation Squared
#                              3 - Rotation Squared
#              Field 7       - Ordinate Denominator Data Type Qualifier
#                              0 - Translation
#                              1 - Rotation
#                              2 - Translation Squared
#                              3 - Rotation Squared
#              Field 8       - Z-axis Data Type Qualifier
#                              0 - Translation
#                              1 - Rotation
#                              2 - Translation Squared
#                              3 - Rotation Squared
#              Field 9       - Sampling Type
#                              0 - Dynamic
#                              1 - Static
#                              2 - RPM from Tach
#                              3 - Frequency from tach
#                              4 - Octave
#              Field 10      - Time Average
#                              0 - None
#                              1 - Fast
#                              2 - Slow
#                              3 - Impulse
#                              4 - Linear
#              Fields 11-12  - not used (fill with zeros)
        (weighting_type, window_type, amplitude_units, normalization_method,
         abscissa_data_qualifier, ordinate_num_data_qualifier,
         ordinate_den_data_qualifier, zaxis_data_qualifier,
         sampling_type, time_average, not_used, not_used) = parse_uff_line(data[1], 12 * ['I6'])
#       Record 3:     FORMAT  (1P5E15.7)
#              Field 1       - Z RPM value
#              Field 2       - Z Time value
#              Field 3       - Z Order value
#              Field 4       - Number of samples
#              Field 5       - Not used
        zrpm, ztime, zorder, number_samples, not_used = parse_uff_line(data[2], 5 * ['E15.7'])
#       Record 4:     FORMAT  (1P5E15.7)
#              Field 1       - User value 1
#              Field 2       - User value 2
#              Field 3       - User value 3
#              Field 4       - User value 4
#              Field 5       - Exponential window damping factor
        (uservalue1, uservalue2, uservalue3, uservalue4,
         exponential_window_damping_factor) = parse_uff_line(data[3], 5 * ['E15.7'])
#       Record 5:     FORMAT  (1P5E15.7)
#              Field 1       - Overall RMS
#              Field 2       - Weighted RMS
#              Fields 3-5    - not used (fill with zeros)
        overall_rms, weighted_rms, not_used, not_used, not_used = parse_uff_line(
            data[4], 5 * ['E15.7'])
#       Record 6:     FORMAT  (2A2,2X,2A2)
#              Field 1       - Response direction
#              Field 2       - Reference direction
        response_direction, reference_direction = parse_uff_line(data[5], ['A4', 'X2', 'A4'])
#       Record 7:     FORMAT  (40A2)
#              Field 1       - not used
        # Now create the dataset
        ds_1858 = cls(set_record_number, octave_format, measurement_run,
                      weighting_type, window_type, amplitude_units, normalization_method,
                      abscissa_data_qualifier, ordinate_num_data_qualifier,
                      ordinate_den_data_qualifier, zaxis_data_qualifier,
                      sampling_type, time_average, zrpm, ztime, zorder, number_samples,
                      uservalue1, uservalue2, uservalue3, uservalue4,
                      exponential_window_damping_factor, overall_rms, weighted_rms,
                      response_direction, reference_direction)
        return ds_1858

    def __repr__(self):
        return 'Sdynpy_UFF_Dataset_1858'

    def write_string(self):
        return_string = ''
        return_string += write_uff_line([self.set_record_number,
                                         self.octave_format,
                                         self.measurement_run,
                                         0, 0, 0], 6 * ['I12'])
        return_string += write_uff_line([self.weighting_type, self.window_type, self.amplitude_units, self.normalization_method,
                                         self.abscissa_data_qualifier, self.ordinate_num_data_qualifier,
                                         self.ordinate_den_data_qualifier, self.zaxis_data_qualifier,
                                         self.sampling_type, self.time_average, 0, 0], 12 * ['I6'])
        return_string += write_uff_line([self.zrpm, self.ztime, self.zorder, self.number_samples, 0],
                                        5 * ['E15.7'])
        return_string += write_uff_line([self.uservalue1, self.uservalue2, self.uservalue3, self.uservalue4,
                                         self.exponential_window_damping_factor],
                                        5 * ['E15.7'])
        return_string += write_uff_line([self.overall_rms, self.weighted_rms, 0, 0, 0],
                                        5 * ['E15.7'])
        return_string += write_uff_line([self.response_direction,
                                        self.reference_direction], ['A4', 'X2', 'A4'])
        return_string += write_uff_line(['NONE'], ['A80'])
        return return_string

    def __str__(self):
        lines = self.write_string().split('\n')
        if len(lines) > 8:
            return 'Dataset 1858: Data\n  ' + '\n  '.join(lines[0:5] + ['.', '.', '.'])
        else:
            return 'Dataset 1858: Data\n  ' + '\n  '.join(lines)


def read(data):
    return Sdynpy_UFF_Dataset_1858.from_uff_data_array(data)
