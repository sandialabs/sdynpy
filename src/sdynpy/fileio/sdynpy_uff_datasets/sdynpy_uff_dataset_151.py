# -*- coding: utf-8 -*-
"""
Header

This dataset defines the header for a universal file

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

import datetime
from ..sdynpy_uff import UFFReadError, parse_uff_line, write_uff_line


class Sdynpy_UFF_Dataset_151:
    def __init__(self, model_file_name='', model_file_description='',
                 db_program='sdynpy', db_date_created=None, db_version=0,
                 db_subversion=0, file_type=0, date_db_last_saved=None,
                 writing_program='sdynpy.fileio.sdynpy_uff.py',
                 date_written=None, release=0, version_number=0,
                 host_id=0, test_id=0, release_counter_per_host=0):
        now = datetime.datetime.now()
        self.model_file_name = model_file_name
        self.model_file_description = model_file_description
        self.db_program = db_program
        if db_date_created is None:
            self.db_date_created = now
        else:
            if not isinstance(db_date_created, datetime.datetime):
                raise ValueError('db_date_created should be a datetime.datetime object or None')
            self.db_date_created = db_date_created
        self.db_version = db_version
        self.db_subversion = db_subversion
        self.file_type = file_type
        if date_db_last_saved is None:
            self.date_db_last_saved = now
        else:
            if not isinstance(date_db_last_saved, datetime.datetime):
                raise ValueError('date_db_last_saved should be a datetime.datetime object or None')
            self.date_db_last_saved = date_db_last_saved
        self.writing_program = writing_program
        if date_written is None:
            self.date_written = now
        else:
            if not isinstance(date_written, datetime.datetime):
                raise ValueError('date_written should be a datetime.datetime object or None')
            self.date_written = date_written
        self.release = release
        self.version_number = version_number
        self.host_id = host_id
        self.test_id = test_id
        self.release_counter_per_host = release_counter_per_host

    @property
    def dataset_number(self):
        return 151

    @classmethod
    def from_uff_data_array(cls, data):
        # Transform from binary to ascii
        data = [line.decode() for line in data]
        ds_151 = cls()
        # Record 1:       FORMAT(80A1)
        # Field 1      -- model file name
        ds_151.model_file_name, = parse_uff_line(data[0], ['A80'])
        # Record 2:       FORMAT(80A1)
        # Field 1      -- model file description
        ds_151.model_file_description, = parse_uff_line(data[1], ['A80'])
        # Record 3:       FORMAT(80A1)
        # Field 1      -- program which created DB
        ds_151.db_program, = parse_uff_line(data[2], ['A80'])
        # Record 4:       FORMAT(10A1,10A1,3I10)
        # Field 1      -- date database created (DD-MMM-YY)
        # Field 2      -- time database created (HH:MM:SS)
        # Field 3      -- Version from database
        # Field 4      -- Subversion from database
        # Field 5      -- File type
        #                =0  Universal
        #                =1  Archive
        #                =2  Other
        # Note that this script reads the date and time together as one field
        (date_time, ds_151.db_version, ds_151.db_subversion, ds_151.file_type) = (
            parse_uff_line(data[3], ['A20'] + 3 * ['I10']))
        try:
            ds_151.db_date_created = datetime.datetime.strptime(
                ''.join(date_time.split()), '%d-%b-%y%H:%M:%S')
        except ValueError:
            ds_151.db_date_created = date_time
        # Record 5:       FORMAT(10A1,10A1)
        # Field 1      -- date database last saved (DD-MMM-YY)
        # Field 2      -- time database last saved (HH:MM:SS)
        # Note that this script reads the date and time together as one field
        date_time, = parse_uff_line(data[4], ['A20'])
        try:
            ds_151.date_db_last_saved = datetime.datetime.strptime(
                ''.join(date_time.split()), '%d-%b-%y%H:%M:%S')
        except ValueError:
            ds_151.date_db_last_saved = date_time
        # Record 6:       FORMAT(80A1)
        # Field 1      -- program which created universal file
        ds_151.writing_program, = parse_uff_line(data[5], ['A80'])
        # Record 7:       FORMAT(10A1,10A1,4I5) <-- Is the 4I5 wrong?  There are 5 fields of integers?
        # Field 1      -- date universal file written (DD-MMM-YY)
        # Field 2      -- time universal file written (HH:MM:SS)
        # Field 3      -- Release which wrote universal file
        # Field 4      -- Version number
        # Field 5      -- Host ID
        #                MS1.  1-Vax/VMS 2-SGI     3-HP7xx,HP-UX
        #                      4-RS/6000 5-Alp/VMS 6-Sun 7-Sony
        #                      8-NEC     9-Alp/OSF
        # Field 6      -- Test ID
        # Field 7      -- Release counter per host
        (date_time, ds_151.release, ds_151.version_number,
         ds_151.host_id, ds_151.test_id, ds_151.release_counter_per_host) = (
            parse_uff_line(data[6], ['A20'] + 5 * ['I5']))
        try:
            ds_151.date_written = datetime.datetime.strptime(
                ''.join(date_time.split()), '%d-%b-%y%H:%M:%S')
        except ValueError:
            ds_151.date_written = date_time
        return ds_151

    def __repr__(self):
        return 'Sdynpy_UFF_Dataset_151<{}>'.format(self.model_file_name)

    def write_string(self):
        try:
            date_db_created = ('{:<10}'.format(self.db_date_created.strftime('%d-%b-%y')) +
                               '{:>10}'.format(self.db_date_created.strftime('%H:%M:%S')))
        except AttributeError:
            date_db_created = self.db_date_created
        try:
            date_db_last_saved = ('{:<10}'.format(self.date_db_last_saved.strftime('%d-%b-%y')) +
                                  '{:<10}'.format(self.date_db_last_saved.strftime('%H:%M:%S')))
        except AttributeError:
            date_db_last_saved = self.date_db_last_saved
        try:
            date_written = ('{:<10}'.format(self.date_written.strftime('%d-%b-%y')) +
                            '{:<10}'.format(self.date_written.strftime('%H:%M:%S')))
        except AttributeError:
            date_written = self.date_written

        return (write_uff_line([self.model_file_name], ['A80']) +
                write_uff_line([self.model_file_description], ['A80']) +
                write_uff_line([self.db_program], ['A80']) +
                write_uff_line([date_db_created,
                                self.db_version,
                                self.db_subversion,
                                self.file_type], ['A20'] + 3 * ['I10']) +
                write_uff_line([date_db_last_saved], ['A20']) +
                write_uff_line([self.writing_program], ['A80']) +
                write_uff_line([date_written,
                                self.release,
                                self.version_number,
                                self.host_id,
                                self.test_id,
                                self.release_counter_per_host], ['A20'] + 5 * ['I5']))

    def __str__(self):
        lines = self.write_string().split('\n')
        if len(lines) > 8:
            return 'Dataset 151: Header\n  ' + '\n  '.join(lines[0:5] + ['.', '.', '.'])
        else:
            return 'Dataset 151: Header\n  ' + '\n  '.join(lines)


def read(data):
    return Sdynpy_UFF_Dataset_151.from_uff_data_array(data)
