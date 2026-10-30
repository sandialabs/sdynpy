# -*- coding: utf-8 -*-
"""
Neal Gordon  
2026-10-27

Readers for Nastran bulk data (.dat), printed results (.f06) and punch (.pch)
files, converted into SDynPy geometry and shape objects.

Real modes (SOL 103 / SEMODES) of a single subcase are supported.  See
sdynpy_nastran_common for the supported bulk data syntax and elements.
"""

import re
import warnings

import numpy as np

from .sdynpy_nastran_common import (FLOAT, eigenvalue_row, find_file,
                                    geometry_from_model, modes_from_tables,
                                    read_dat, shapes_from_modes)

_PCH_FLOAT = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?')


def read_f06(path):
    """
    Read real eigenvalues and eigenvectors from a Nastran F06 file.

    Returns a dict with the same keys as the MYSTRAN ``read_f06``: ``eigenvalue``,
    ``radians``, ``frequency``, ``generalized_mass``, ``generalized_stiffness``,
    ``grid_id`` and ``shape`` (nmodes, ngrids, 6).  Scalar points are ignored.
    """
    row_vec = re.compile(r'^\s*(\d+)\s+G\s+((?:' + FLOAT + r'\s*){6})\s*$')
    header = re.compile(r'R E A L   E I G E N V E C T O R   N O \.\s+(\d+)')
    eig_rows = []
    vectors = {}
    grid_ids = {}
    in_eig_table = False
    current = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if line[:2] == '1 ':
                # New page: each page of a mode repeats the eigenvector header
                current = None
            if 'R E A L   E I G E N V A L U E S' in line:
                in_eig_table = True
                continue
            m = header.search(line)
            if m:
                in_eig_table = False
                current = int(m.group(1))
                vectors.setdefault(current, [])
                grid_ids.setdefault(current, [])
                continue
            if in_eig_table:
                row = eigenvalue_row(line)
                if row:
                    eig_rows.append(row)
                elif eig_rows and line.strip() == '':
                    in_eig_table = False
                continue
            if current is not None:
                m = row_vec.match(line)
                if m:
                    grid_ids[current].append(int(m.group(1)))
                    vectors[current].append([float(v) for v in m.group(2).split()])
    return modes_from_tables(eig_rows, vectors, grid_ids, path)


def read_pch(path):
    """
    Read real eigenvectors from a Nastran punch file (``DISPLACEMENT(PUNCH)``).

    The punch file holds eigenvalues but no generalized mass, so a modal mass of
    1 is assigned (the EIGRL default is mass normalization).  Only the first
    subcase is read.  Returns the same dict as :func:`read_f06`.
    """
    eigenvalues = {}
    vectors = {}
    grid_ids = {}
    subcase = first_subcase = current = pending = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if line.startswith('$'):
                m = re.match(r'\$SUBCASE ID\s*=\s*(\d+)', line)
                if m:
                    subcase = int(m.group(1))
                m = re.match(r'\$EIGENVALUE\s*=\s*(\S+)\s+MODE\s*=\s*(\d+)', line)
                if m:
                    if first_subcase is None:
                        first_subcase = subcase
                    pending = None
                    current = int(m.group(2)) if subcase == first_subcase else None
                    if current is not None:
                        eigenvalues[current] = float(m.group(1))
                        vectors[current] = []
                        grid_ids[current] = []
                continue
            if current is None:
                continue
            if line.startswith('-CONT-'):
                if pending is not None:
                    values = _PCH_FLOAT.findall(line.split(None, 1)[1])[:3]
                    grid_ids[current].append(pending[0])
                    vectors[current].append(pending[1] + [float(v) for v in values])
                    pending = None
                continue
            tokens = line.split(None, 2)
            if len(tokens) < 3 or tokens[1] != 'G':
                pending = None
                continue
            pending = (int(tokens[0]), [float(v) for v in _PCH_FLOAT.findall(tokens[2])[:3]])
    rows = []
    for mode, value in eigenvalues.items():
        radians = np.sqrt(abs(value))
        rows.append([mode, value, radians, radians / (2 * np.pi), 1.0, value])
    return modes_from_tables(rows, vectors, grid_ids, path)


def read_nastran(basename, source='f06', damping=0.0):
    """
    Read a Nastran model and its modal results.

    Parameters
    ----------
    basename : str
        Path to the model without extension (or to the .dat file).
    source : str, optional
        'f06' or 'pch', which file to read the modes from.  The default is
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
    if src not in ('f06', 'pch'):
        raise ValueError("source must be 'f06', 'pch', 'none' or ''")
    try:
        path = find_file(basename, '.' + src)
    except FileNotFoundError as e:
        warnings.warn('Results file {:} not found, only the geometry was read'.format(e))
        return geometry, None
    modes = read_f06(path) if src == 'f06' else read_pch(path)
    return geometry, shapes_from_modes(modes, model, damping)
