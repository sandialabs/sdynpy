# -*- coding: utf-8 -*-
"""
Base class for all SDynPy object arrays.

SDynPy object arrays are subclasses of numpy's ndarray.  SDynPy uses structured
arrays to store the underlying data objects, resulting in potentially complex
data types while still achieving the efficiency and flexibility of numpy arrays.

This module defines the SdynpyArray, which is a subclass of numpy ndarray.  The
core SDynPy objects inherit from this class.  The main contribution of this
array is allowing users to access the underlying structured array fields using
attribute notation rather than the index notation used by numpy
(e.g. object.field rather than object["field"]).

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
from scipy.io import savemat as scipy_savemat


class SdynpyArray(np.ndarray):
    """Superclass of the core SDynPy objects

    The core SDynPy object arrays inherit from this class.  The class is a
    subclass of numpy's ndarray.  The underlying data structure is stored as a
    structured array, but the class's implementation allows accessing the array
    fields as if they were attributes.
    """
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super(SdynpyArray, subtype).__new__(subtype, shape, dtype,
                                                  buffer, offset, strides,
                                                  order)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        # if obj is None:
        #     return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        # We do not need to return anything
        pass

    # this method is called whenever you use a ufunc
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        f = {
            "reduce": ufunc.reduce,
            "accumulate": ufunc.accumulate,
            "reduceat": ufunc.reduceat,
            "outer": ufunc.outer,
            "at": ufunc.at,
            "__call__": ufunc,
        }
        # print('In ufunc\n  ufunc: {:}\n  method: {:}\n  inputs: {:}\n  kwargs: {:}\n'.format(
        # ufunc,method,inputs,kwargs))
        # convert the inputs to np.ndarray to prevent recursion, call the function, then cast it back as CoordinateArray
        output = f[method](*(i.view(np.ndarray) for i in inputs), **kwargs).view(self.__class__)
        return output

    def __array_function__(self, func, types, args, kwargs):
        # print('In Array Function\n  func: {:}\n  types: {:}\n  args: {:}\n  kwargs: {:}'.format(
        # func,types,args,kwargs))
        output = super().__array_function__(func, types, args, kwargs)
        # print('  Output Type: {:}'.format(type(output)))
        # if isinstance(output,tuple):
        #     print('Tuple Types: {:}'.format(type(val) for val in output))
        # print('Output: {:}'.format(output))
        if output is NotImplemented:
            return NotImplemented
        else:
            if isinstance(output, np.ndarray) and (self.dtype == output.dtype):
                return output.view(self.__class__)
            elif isinstance(output, np.ndarray):
                return output.view(np.ndarray)
            else:
                output_values = []
                for val in output:
                    # print('  Tuple Output Type: {:}'.format(type(val)))
                    if isinstance(val, np.ndarray):
                        # print('    Output dtypes {:},{:}'.format(self.dtype,val.dtype))
                        if self.dtype == val.dtype:
                            # print('    Appending {:}'.format(self.__class__))
                            output_values.append(val.view(self.__class__))
                        else:
                            # print('    Appending {:}'.format(np.ndarray))
                            output_values.append(val.view(np.ndarray))
                    else:
                        output_values.append(val)
                return output_values

    def __getattr__(self, attr):
        try:
            return self[attr]
        except ValueError:
            raise AttributeError("'{:}' object has no attribute '{:}'".format(self.__class__, attr))

    def __setattr__(self, attr, value):
        try:
            self[attr] = value
        except (ValueError, IndexError) as e:
            # # Check and make sure you don't have an attribute already with that
            # # name
            if attr in self.dtype.fields:
                # print('ERROR: Assignment to item failed, attempting to assign item to attribute!')
                raise e
            super().__setattr__(attr, value)

    def __getitem__(self, key):
        # print('Key is type {:}'.format(type(key)))
        # print('Key is {:}'.format(key))
        return_val = super().__getitem__(key)
        try:
            if isinstance(key, str) and (not isinstance(return_val, np.void)) and key in self.dtype.names:
                return_val = return_val.view(np.ndarray)
                if return_val.ndim == 0:
                    return_val = return_val[()]
        except TypeError:
            pass
        if isinstance(return_val, np.void):
            return_val = np.asarray(return_val).view(self.__class__)
        # print('Returning a {:}'.format(type(return_val)))
        return return_val

    def __setitem__(self, key, value):
        try:
            if key in self.dtype.fields:
                self[key][...] = value
            else:
                super().__setitem__(key, value)
        except TypeError:
            super().__setitem__(key, value)

    def ndenumerate(self):
        """
        Enumerates over all entries in the array

        Yields
        ------
        tuple
            indices corresponding to each entry in the array
        array
            entry in the array corresponding to the index
        """
        for key, val in np.ndenumerate(self):
            yield (key, np.asarray(val).view(self.__class__))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'Shape {:} {:} with fields {:}'.format(' x '.join(str(v) for v in self.shape), self.__class__.__name__, self.dtype.names)

    def save(self, filename: str):
        """
        Save the array to a numpy file

        Parameters
        ----------
        filename : str
            Filename that the array will be saved to.  Will be appended with
            .npy if not specified in the filename

        """
        np.save(filename, self.view(np.ndarray))

    def assemble_mat_dict(self):
        """
        Assembles a dictionary of fields

        Returns
        -------
        output_dict : dict
            A dictionary of contents of the file
        """
        output_dict = {}
        for field in self.fields:
            val = self[field]
            if isinstance(val, SdynpyArray):
                val = val.assemble_mat_dict()
            else:
                val = np.ascontiguousarray(val)
            output_dict[field] = val
        return output_dict

    def savemat(self, filename):
        """
        Save array to a Matlab `*.mat` file.

        Parameters
        ----------
        filename : str
            Name of the file in which the data will be saved

        Returns
        -------
        None.

        """
        scipy_savemat(filename, self.assemble_mat_dict())

    @classmethod
    def load(cls, filename):
        """
        Load in the specified file into a SDynPy array object

        Parameters
        ----------
        filename : str
            Filename specifying the file to load.  If the filename has
            extension .unv or .uff, it will be loaded as a universal file.
            Otherwise, it will be loaded as a NumPy file.

        Raises
        ------
        AttributeError
            Raised if a unv file is loaded from a class that does not have a
            from_unv attribute defined.

        Returns
        -------
        cls
            SDynpy array of the appropriate type from the loaded file.

        """
        if filename[-4:].lower() in ['.unv', '.uff']:
            try:
                from ..fileio.sdynpy_uff import readunv
                unv_dict = readunv(filename)
                return cls.from_unv(unv_dict)
            except AttributeError:
                raise AttributeError('Class {:} has no from_unv attribute defined'.format(cls))
        else:
            return np.load(filename, allow_pickle=True).view(cls)

    def __eq__(self, other):
        if not isinstance(self, other.__class__):
            return NotImplemented
        equal_array = []
        for field, (dtype, extra) in self.dtype.fields.items():
            if dtype.kind == 'O':
                self_data = self[field]
                other_data = other[field]
                if self.ndim == 0:
                    obj_arr = np.ndarray((), 'object')
                    obj_arr[()] = self_data
                    self_data = obj_arr
                if other.ndim == 0:
                    obj_arr = np.ndarray((), 'object')
                    obj_arr[()] = other_data
                    other_data = obj_arr
                self_data, other_data = np.broadcast_arrays(self_data, other_data)
                truth_array = np.zeros(self_data.shape, dtype=bool)
                for key in np.ndindex(truth_array.shape):
                    truth_array[key] = np.array_equal(self_data[key], other_data[key])
            else:
                truth_array = self[field] == other[field]
            if len(dtype.shape) != 0:
                truth_array = np.all(truth_array, axis=tuple(-1 - np.arange(len(dtype.shape))))
            equal_array.append(truth_array)
        return np.all(equal_array, axis=0)

    def __ne__(self, other):
        return ~self.__eq__(other)

    @property
    def fields(self):
        """Returns the fields of the structured array.

        These fields can be accessed through attribute syntax."""
        return self.dtype.names
