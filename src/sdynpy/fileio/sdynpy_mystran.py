# -*- coding: utf-8 -*-
"""
Neal Gordon  
2026-10-27

Readers for MYSTRAN input and output files.

more info on the solver can be found at  
https://mystran.com/

Reads the bulk data (.dat), the printed results (.F06), and the binary
OUTPUT4 matrices (.OU4) and converts them into SDynPy geometry and shape
objects.  Only numpy and scipy are required.

The bulk data parsing, coordinate systems and SDynPy conversion are shared
with the Nastran reader, see sdynpy_nastran_common.

"""

import re
import warnings

import numpy as np
from scipy.io import FortranEOFError, FortranFile

from ..core.sdynpy_coordinate import coordinate_array
from ..core.sdynpy_shape import shape_array
from .sdynpy_nastran_common import (FLOAT, eigenvalue_row, find_file, free_dofs,
                                    geometry_from_model, modes_from_tables,
                                    read_dat, shapes_from_modes)

shapes_from_f06 = shapes_from_modes


# ---------------------------------------------------------------------------
# F06
# ---------------------------------------------------------------------------

def read_f06(path):
    """
    Read eigenvalues and eigenvectors from a MYSTRAN F06 file.

    Parameters
    ----------
    path : str
        Path to the F06 file.

    Returns
    -------
    dict
        Dictionary with keys

        - ``eigenvalue`` : (nmodes,) eigenvalues (rad/s)^2
        - ``radians`` : (nmodes,) natural frequencies in rad/s
        - ``frequency`` : (nmodes,) natural frequencies in Hz
        - ``generalized_mass`` : (nmodes,)
        - ``generalized_stiffness`` : (nmodes,)
        - ``grid_id`` : (ngrids,) grid ids in the order printed
        - ``shape`` : (nmodes, ngrids, 6) eigenvectors (T1-T3, R1-R3)
    """
    row_vec = re.compile(r'^\s*(\d+)\s+(\d+)\s+((?:' + FLOAT + r'\s*){6})\s*$')
    eig_rows = []
    vectors = {}
    grid_ids = {}
    in_eig_table = False
    current = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if 'R E A L   E I G E N V A L U E S' in line:
                in_eig_table = True
                continue
            m = re.match(r'^\s*OUTPUT FOR EIGENVECTOR\s+(\d+)', line)
            if m:
                in_eig_table = False
                current = int(m.group(1))
                vectors[current] = []
                grid_ids[current] = []
                continue
            if in_eig_table:
                row = eigenvalue_row(line)
                if row:
                    eig_rows.append(row)
                elif eig_rows and line.strip() == '':
                    in_eig_table = False
                continue
            if current is not None:
                if '------' in line:
                    current = None
                    continue
                m = row_vec.match(line)
                if m:
                    grid_ids[current].append(int(m.group(1)))
                    vectors[current].append([float(v) for v in m.group(3).split()])
    return modes_from_tables(eig_rows, vectors, grid_ids, path)


# ---------------------------------------------------------------------------
# OU4
# ---------------------------------------------------------------------------

def read_ou4(path):
    """
    Read the matrices from a MYSTRAN OUTPUT4 binary (.OU4) file.

    The file is a sequence of Fortran unformatted records.  Each matrix has
    a 24 byte header (ncols, nrows, form, type, 8 character name) followed by
    one record per column holding (column, first row, number of 4 byte words)
    and the column values.

    Parameters
    ----------
    path : str
        Path to the OU4 file.

    Returns
    -------
    dict
        Matrix name (as stored, at most 8 characters, e.g. ``EIGEN_VA``)
        mapped to a dense 2D array.
    """
    matrices = {}
    with FortranFile(path, 'r') as f:
        while True:
            try:
                rec = f.read_record(np.uint8)
            except (EOFError, FortranEOFError):
                break
            if rec.size != 24:
                continue
            ncols, nrows, form, mtype = (int(v) for v in np.frombuffer(rec[:16], dtype=np.int32))
            name = rec[16:24].tobytes().decode(errors='ignore').strip()
            if ncols < 1 or nrows < 1 or name == '' or mtype not in (1, 2):
                continue
            dtype = np.float32 if mtype == 1 else np.float64
            mat = np.zeros((nrows, ncols), dtype=dtype)
            for _ in range(ncols):
                col_rec = f.read_record(np.uint8)
                col, first_row, _ = (int(v) for v in np.frombuffer(col_rec[:12], dtype=np.int32))
                values = np.frombuffer(col_rec[12:], dtype=dtype)
                mat[first_row - 1:first_row - 1 + values.size, col - 1] = values
            matrices[name] = mat
    return matrices


# ---------------------------------------------------------------------------
# SDynPy objects
# ---------------------------------------------------------------------------

def shapes_from_ou4(ou4, model, damping=0.0):
    """
    Create a ShapeArray from the matrices returned by :func:`read_ou4`.

    The EIGEN_VEC matrix is assumed to be in the G-set (6 DOFs per grid) if
    it has 6 times as many rows as grids, and in the L-set otherwise.  The
    OU4 file does not contain generalized masses, so a modal mass of 1 is
    assigned (MYSTRAN mass normalizes the vectors by default).
    """
    vec = ou4['EIGEN_VE'].T
    val = np.asarray(ou4['EIGEN_VA']).reshape(-1)
    frequency = np.sqrt(np.abs(val)) / (2 * np.pi)
    if vec.shape[1] == 6 * model['grid_id'].size:
        grid_nodes = np.repeat(model['grid_id'], 6)
        grid_dirs = np.tile(np.arange(1, 7), model['grid_id'].size)
        keep = [i for i, (n, d) in enumerate(zip(grid_nodes, grid_dirs))
                if d not in model['constraints'].get(int(n), set())]
        nodes, directions = grid_nodes[keep], grid_dirs[keep]
        vec = vec[:, keep]
    else:
        nodes, directions = free_dofs(model)
        if vec.shape[1] != nodes.size:
            raise ValueError('EIGEN_VEC has {:} DOFs but the model has {:} free DOFs'.format(
                vec.shape[1], nodes.size))
    return shape_array(coordinate_array(nodes, directions), vec, frequency, damping, 1.0)


def read_mystran(basename, source='f06', damping=0.0):
    """
    Read a MYSTRAN model and its modal results.

    Parameters
    ----------
    basename : str
        Path to the model without extension (or to the .dat file).  The
        .dat, .F06 and .OU4 files are expected next to each other.
    source : str, optional
        'f06' or 'ou4', which file to read the modes from.  The default is
        'f06'.  Use 'none', '' or None to read only the geometry.  If the
        results file is not found, only the geometry is read and a warning
        is issued.
    damping : float, optional
        Fraction of critical damping assigned to every mode.

    Returns
    -------
    geometry : Geometry
    shapes : ShapeArray or None
        None if no results were read.
    """
    model = read_dat(find_file(basename, '.dat'))
    geometry = geometry_from_model(model)
    src = (source or 'none').lower()
    if src == 'none':
        return geometry, None
    if src not in ('f06', 'ou4'):
        raise ValueError("source must be 'f06', 'ou4', 'none' or ''")
    try:
        path = find_file(basename, '.' + src)
    except FileNotFoundError as e:
        warnings.warn('Results file {:} not found, only the geometry was read'.format(e))
        return geometry, None
    if src == 'f06':
        shapes = shapes_from_modes(read_f06(path), model, damping)
    else:
        shapes = shapes_from_ou4(read_ou4(path), model, damping)
    return geometry, shapes
