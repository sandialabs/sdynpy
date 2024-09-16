# -*- coding: utf-8 -*-
"""
Units

This dataset defines the unit system for the universal file

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

from ..sdynpy_uff import UFFReadError


class Sdynpy_UFF_Dataset_164:
    def __init__(self):
        self.units_code = 0
        self.units_description = ''
        self.temperature_mode = 1
        self.length_conv = 1
        self.force_conv = 1
        self.temp_conv = 1
        self.temp_offset = 1

    @property
    def dataset_number(self):
        return 164

    @classmethod
    def from_uff_data_array(cls, data):
        # Transform from binary to ascii
        data = [line.decode() for line in data]
        ds_164 = cls()
        # Record 1:       FORMAT(I10,20A1,I10)
        # Field 1      -- units code
        #                = 1 - SI: Meter (newton)
        #                = 2 - BG: Foot (pound f)
        #                = 3 - MG: Meter (kilogram f)
        #                = 4 - BA: Foot (poundal)
        #                = 5 - MM: mm (milli newton)
        #                = 6 - CM: cm (centi newton)
        #                = 7 - IN: Inch (pound f)
        #                = 8 - GM: mm (kilogram f)
        #                = 9 - US: USER_DEFINED
        #                = 10- MN: mm (newton)
        try:
            ds_164.units_code = int(data[0][0:10])
        except ValueError:
            return UFFReadError('Units Code (record 1 field 1) must be an integer!')
        # Field 2      -- units description (used for
        #                documentation only)
        ds_164.units_description = data[0][10:30].strip()
        # Field 3      -- temperature mode
        #                = 1 - absolute
        #                = 2 - relative
        try:
            ds_164.temperature_mode = int(data[0][30:40])
        except ValueError:
            return UFFReadError('Temperature Mode (record 1 field 3) must be an integer!')
        # Record 2:       FORMAT(3D25.17)
        #                Unit factors for converting universal file units to SI.
        #                To convert from universal file units to SI divide by
        #                the appropriate factor listed below.
        # Field 1      -- length
        try:
            ds_164.length_conv = float(data[1][0:25])
        except ValueError:
            return UFFReadError('Length Conversion Factor (record 2 field 1) must be a floating point number')
        # Field 2      -- force
        try:
            ds_164.force_conv = float(data[1][25:50])
        except ValueError:
            return UFFReadError('Force Conversion Factor (record 2 field 2) must be a floating point number')
        # Field 3      -- temperature
        try:
            ds_164.temp_conv = float(data[1][50:75])
        except ValueError:
            return UFFReadError('Temperature Conversion Factor (record 2 field 3) must be a floating point number')
        # Field 4      -- temperature offset <-- Actually Record 3 field 1!
        try:
            ds_164.temp_offset = float(data[2][0:25])
        except ValueError:
            return UFFReadError('Temperature offset (record 4 field 1) must be a floating point number')
        return ds_164

    def __repr__(self):
        return 'Sdynpy_UFF_Dataset_164<{}>'.format(self.units_description)

    def write_string(self):
        return ('{:<10}{:<20}{:<10}\n'.format(self.units_code,
                                              self.units_description,
                                              self.temperature_mode) +
                '{:>25.17E}{:>25.17E}{:>25.17E}\n{:>25.17E}\n'.format(self.length_conv,
                                                                      self.force_conv,
                                                                      self.temp_conv,
                                                                      self.temp_offset))

    def __str__(self):
        lines = self.write_string().split('\n')
        if len(lines) > 8:
            return 'Dataset 164: Units\n  ' + '\n  '.join(lines[0:5] + ['.', '.', '.'])
        else:
            return 'Dataset 164: Units\n  ' + '\n  '.join(lines)


def read(data):
    return Sdynpy_UFF_Dataset_164.from_uff_data_array(data)
