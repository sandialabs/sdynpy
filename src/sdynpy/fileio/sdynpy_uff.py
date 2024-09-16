# -*- coding: utf-8 -*-
"""
Interface to the universal file format (UFF).

Using the functions in this module, one can read and write unv files.

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

import numpy as np

class UFFReadError(Exception):
    """Exception to be used when there is an error reading a UNV file"""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


read_type_functions = {'A': str,
                       'I': int,
                       'X': str,
                       'E': float,
                       'D': float}

write_type_format_operations = {'A': '{{:<{:}s}}',
                                'I': '{{:>{:}d}}',
                                'X': '{{:>{:}s}}',
                                'E': '{{:>{:}E}}',
                                'D': '{{:>{:}e}}'
                                }


def parse_uff_line(line, format_specs, read_number=None):
    """
    Parses a line from a universal file format

    Parameters
    ----------
    line : str
        A line from a unv file
    format_specs : iterable
        The format specifiers for the line that determines how the string is
        transformed into data
    read_number : int, optional
        The number of entries to read.  Will repeat the format specifiers if
        necessary

    Raises
    ------
    UFFReadError
        Raised if an error occurs reading the file.

    Returns
    -------
    output : list
        Data from this line of the unv file.

    """
    if read_number is None:
        read_number = len(format_specs)
    else:
        copies = int((read_number - 1) // len(format_specs) + 1)
        format_specs = (copies * format_specs)[:read_number]
    position = 0
    output = []
    for spec in format_specs:
        spec_type = spec[0].upper()
        spec_length = int(spec[1:].split('.')[0])
        if spec_type == 'X':
            position += spec_length
            continue
        try:
            type_function = read_type_functions[spec_type]
        except KeyError:
            raise UFFReadError('Invalid Type {:}, should be one of {:}'.format(
                spec_type, [key for key in read_type_functions.keys()]))
        position_string = line[position:position + spec_length].rstrip()
        if position_string == '':
            output.append(None)
        else:
            try:
                output.append(type_function(position_string))
            except ValueError:
                raise UFFReadError('Line "{:}"\n  characters "{:}" cannot be transformed to type {:}'.format(
                    line, position_string, spec_type))
        position += spec_length
    return output


def parse_uff_lines(lines, line_format_spec, read_number):
    """
    Reads multiple lines from a universal file

    Parameters
    ----------
    lines : iterable
        List of lines to read
    line_format_spec : iterable
        The format specifiers for the line that determines how the string is
        transformed into data
    read_number : int
        The number of entries to read

    Returns
    -------
    output
        Data from the universal file over the specified lines.
    lines_read : int
        Number of lines read from the universal file

    """
    full_lines = read_number // len(line_format_spec)
    remainder = read_number % len(line_format_spec)
    output = []
    for i, line in zip(range(full_lines), lines):
        output += parse_uff_line(line, line_format_spec)
    if remainder > 0:
        output += parse_uff_line(lines[full_lines], line_format_spec, remainder)
    return output, full_lines + (1 if remainder > 0 else 0)


def write_uff_line(data, format_specs, fill_line=True):
    """
    Write data to universal file format

    Parameters
    ----------
    data : iterable
        The data to write to the universal file.
    format_specs : iterable
        The format specification for each value in data
    fill_line : bool, optional
        Fill the line completely. The default is True.

    Returns
    -------
    line : str
        A string representation of the data in the universal file format

    """
    write_number = len(data)
    non_X_format_specs = [
        format_spec for format_spec in format_specs if 'X' not in format_spec.upper()]
    copies = int((write_number - 1) // len(non_X_format_specs) + 1)
    format_specs = (copies * (format_specs + ['\n']))
    line = ''
    data_index = 0
    for spec in format_specs:
        if spec == '\n':
            line += '\n'
            continue
        spec_type = spec[0].upper()
        spec_format = spec[1:]
        spec_length = int(spec_format.split('.')[0])
        if spec_type == 'X':
            new_data = (write_type_format_operations[spec_type].format(spec_format)).format('')
        elif data[data_index] is None:
            new_data = (write_type_format_operations['X'].format(spec_length)).format('')
            data_index += 1
        else:
            new_data = (write_type_format_operations[spec_type].format(
                spec_format)).format(data[data_index])
            data_index += 1
        if len(new_data) > spec_length:
            print('Data to write {:} longer than specification length of {:}.  Truncating!'.format(
                new_data, spec_length))
            new_data = new_data[:spec_length]
        line += new_data
        if data_index == len(data):
            break
    if fill_line:
        lines = line.split('\n')
        line = '\n'.join('{:<80s}'.format(this_line) for this_line in lines) + '\n'
    return line


# To add a data set for reading, you must write it in a file defining a read
# and write command, import it here, then add it to the dataset dictionary
# using the dataset number as the key.

from .sdynpy_uff_datasets import sdynpy_uff_dataset_55 as dataset_55  # noqa: E402
from .sdynpy_uff_datasets import sdynpy_uff_dataset_58 as dataset_58  # noqa: E402
from .sdynpy_uff_datasets import sdynpy_uff_dataset_82 as dataset_82  # noqa: E402
from .sdynpy_uff_datasets import sdynpy_uff_dataset_151 as dataset_151  # noqa: E402
from .sdynpy_uff_datasets import sdynpy_uff_dataset_164 as dataset_164  # noqa: E402
from .sdynpy_uff_datasets import sdynpy_uff_dataset_1858 as dataset_1858  # noqa: E402
from .sdynpy_uff_datasets import sdynpy_uff_dataset_2400 as dataset_2400  # noqa: E402
from .sdynpy_uff_datasets import sdynpy_uff_dataset_2411 as dataset_2411  # noqa: E402
from .sdynpy_uff_datasets import sdynpy_uff_dataset_2412 as dataset_2412  # noqa: E402
from .sdynpy_uff_datasets import sdynpy_uff_dataset_2420 as dataset_2420  # noqa: E402

dataset_dict = {55: dataset_55,
                58: dataset_58,
                82: dataset_82,
                151: dataset_151,
                164: dataset_164,
                1858: dataset_1858,
                2400: dataset_2400,
                2411: dataset_2411,
                2412: dataset_2412,
                2420: dataset_2420}



def readuff(filename, datasets=None, verbose=False):
    """
    Read a universal file

    Parameters
    ----------
    filename : str
        Path to the file that should be read.
    datasets : iterable, optional
        List of dataset id numbers to read. The default is None.
    verbose : bool, optional
        Output extra information when reading the file. The default is False.

    Raises
    ------
    UFFReadError
        Raised if errors are found when reading the file.

    Returns
    -------
    dict
        Dictionary with keys as the dataset id numbers and values containing the
        data from the universal file in those datasets.

    """
    return_dict = {}
    with open(filename, 'rb') as f:
        line = b'\n'
        line_num = 0
        dataset_line_num = 0
        # Loop through the file until it is at its end
        while line != b'':
            # Find the first delimiter line
            # Here we want to find a line that has -1 in the 5th and 6th column,
            # and make sure that -1 isn't the only thing in the line to make sure
            # that any comments at the start of the file don't accidentally line up
            while not line[4:6] == b'-1' and not line.strip() == b'-1':
                line = f.readline()
                line_num += 1
                if line == b'':
                    break
            if line == b'':
                break
            dataset_line_num = line_num
            # Load in the dataset specifier
            line = f.readline()
            line_num += 1
            try:
                # Make sure that we can convert it to an integer
                (dataset, b, byte_ordering, floating_point_format,
                 num_ascii_lines_following, num_bytes_following, *not_used) = parse_uff_line(
                     line.decode(), ['I6', 'A1', 'I6', 'I6', 'I12', 'I12', 'I6', 'I6', 'I12', 'I12'])
                is_binary = b is not None
                if is_binary:
                    if byte_ordering is None:
                        byte_ordering = 1
                    if num_ascii_lines_following is None:
                        num_ascii_lines_following = 11
            except UFFReadError:
                raise UFFReadError(
                    'Improperly formatted dataset specification at line {}, {}'.format(line_num, line))
            if verbose:
                print('Reading Dataset {:} at line {:}'.format(dataset, dataset_line_num))
            # Load in the dataset information
            line = f.readline()
            line_num += 1
            data = []
            # Loop through the file until we find the delimiter
            while (not (not is_binary and line[4:6] == b'-1' and line.strip() == b'-1')
                   and
                   not (is_binary and line.rstrip()[-6:] == b'    -1' and line_num - dataset_line_num - 1 > num_ascii_lines_following)):
                data.append(line)
                line = f.readline()
                line_num += 1
                if line == '':
                    raise UFFReadError(
                        'File ended before dataset starting at line {} was ended.'.format(dataset_line_num))
                if is_binary and line.rstrip()[-6:] == b'    -1' and line_num - dataset_line_num - 1 > num_ascii_lines_following:
                    data.append(line.rstrip()[:-6])
            try:
                read_fn = dataset_dict[dataset].read
            except KeyError:
                print('Dataset {} at line {} is not implemented, skipping...'.format(
                    dataset, dataset_line_num))
                # Read the next line in preparation for the next loop of the script
                line = f.readline()
                line_num += 1
                continue
            except AttributeError:
                print('Dataset {} at line {} read function is not implemented, skipping...'.format(
                    dataset, dataset_line_num))
                # Read the next line in preparation for the next loop of the script
                line = f.readline()
                line_num += 1
                continue
            if datasets is not None and dataset not in datasets:
                print('Skipping dataset {} at line {} due to it not being specified in the `datasets` input argument'.format(
                    dataset, dataset_line_num))
                line = f.readline()
                line_num += 1
                continue
            if is_binary:
                dataset_obj = read_fn(data, is_binary, byte_ordering,
                                      floating_point_format, num_ascii_lines_following,
                                      num_bytes_following)
            else:
                dataset_obj = read_fn(data)
            if isinstance(dataset_obj, UFFReadError):
                raise UFFReadError('In dataset starting at line {}, {}'.format(
                    dataset_line_num, dataset_obj.value))
            if dataset in return_dict:
                return_dict[dataset].append(dataset_obj)
            else:
                return_dict[dataset] = [dataset_obj]
            # Read the next line in preparation for the next loop of the script
            line = f.readline()
            line_num += 1

    return return_dict


readunv = readuff