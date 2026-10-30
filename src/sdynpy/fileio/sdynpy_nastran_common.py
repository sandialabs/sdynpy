# -*- coding: utf-8 -*-
"""
Neal Gordon  
2026-10-27

Shared bulk data parsing and SDynPy conversion for the MYSTRAN and Nastran readers.

Bulk data may be short field (8 character) or free field (comma separated).
Large field cards (``GRID*``) are not supported.  Only rod (CROD, CONROD),
bar (CBAR) and beam (CBEAM) elements are read, other element cards are
ignored.  Constraints come from SPC and SPC1 (SPCADD is not supported).
"""

import os
import re

import numpy as np

from ..core.sdynpy_coordinate import coordinate_array
from ..core.sdynpy_geometry import (Geometry, coordinate_system_array,
                                    element_array, node_array)
from ..core.sdynpy_shape import shape_array

FLOAT = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
_CS_TYPES = {'R': 0, 'C': 1, 'S': 2}
# SDynPy element types: 11 = rod, 21 = linear beam
_LINE_ELEMENTS = {'CROD': 11, 'CONROD': 11, 'CBAR': 21, 'CBEAM': 21}
_EIGENVALUE_ROW = re.compile(r'^\s*(\d+)\s+(\d+)\s+((?:' + FLOAT + r'\*?\s*){5})\s*$')


def find_file(basename, ext):
    """Locate ``basename`` + ``ext`` in any case, accepting a basename that already has an extension."""
    basename = str(basename)
    base, e = os.path.splitext(basename)
    if e.lower() not in ('.dat', '.f06', '.ou4', '.pch'):
        base = basename
    for candidate in (ext, ext.lower(), ext.upper()):
        if os.path.exists(base + candidate):
            return base + candidate
    raise FileNotFoundError(base + ext)


def eigenvalue_row(line):
    """Parse a row of the real eigenvalue table, or return None."""
    m = _EIGENVALUE_ROW.match(line)
    if not m:
        return None
    # A trailing '*' flags a suspect (e.g. negative) eigenvalue
    return [int(m.group(1))] + [float(v) for v in m.group(3).replace('*', '').split()]


def modes_from_tables(eig_rows, vectors, grid_ids, path):
    """Assemble the mode dictionary returned by the F06 and PCH readers."""
    if not eig_rows:
        raise ValueError('No eigenvalue table found in {:}'.format(path))
    eig = np.array(eig_rows)
    mode_numbers = eig[:, 0].astype(int)
    missing = [n for n in mode_numbers if n not in vectors]
    if missing:
        raise ValueError('No eigenvectors found in {:} for modes {:}'.format(path, missing))
    return {
        'eigenvalue': eig[:, 1],
        'radians': eig[:, 2],
        'frequency': eig[:, 3],
        'generalized_mass': eig[:, 4],
        'generalized_stiffness': eig[:, 5],
        'grid_id': np.array(grid_ids[mode_numbers[0]], dtype=int),
        'shape': np.array([vectors[n] for n in mode_numbers]),
    }


# ---------------------------------------------------------------------------
# Bulk data
# ---------------------------------------------------------------------------

def _to_float(s):
    s = s.strip()
    if s == '':
        return 0.0
    try:
        return float(s)
    except ValueError:
        # Nastran shorthand exponent, e.g. 1.0-3 -> 1.0e-3
        return float(re.sub(r'^([-+]?[\d.]+)([-+]\d+)$', r'\1e\2', s.replace('D', 'E').replace('d', 'e')))


def _to_int(s):
    s = s.strip()
    return int(s) if s != '' else 0


def _split_fields(line):
    """Return (name, fields) for a bulk data line, padded to 8 data fields."""
    if ',' in line:
        parts = [p.strip() for p in line.split(',')]
        name, fields = parts[0], parts[1:]
    else:
        name = line[:8].strip()
        fields = [line[i:i + 8].strip() for i in range(8, 72, 8)]
    fields = fields[:8] + [''] * (8 - len(fields))
    return name, fields


def _read_cards(path):
    """Return ([(name, fields), ...], case_control_lines) with continuations merged."""
    in_bulk = False
    case_control = []
    cards = []
    card = None
    with open(path, 'r', errors='replace') as f:
        for raw in f:
            line = raw.rstrip('\r\n').split('$', 1)[0].rstrip()
            if not in_bulk:
                if line.strip().upper().startswith('BEGIN') and 'BULK' in line.upper():
                    in_bulk = True
                else:
                    case_control.append(line)
                continue
            if line.strip() == '':
                continue
            if line.strip().upper().startswith('ENDDATA'):
                break
            if line[0] in ' +,':
                if card is None:
                    continue
                if ',' in line:
                    cont = [p.strip() for p in line.split(',')[1:]]
                else:
                    cont = [line[i:i + 8].strip() for i in range(8, 72, 8)]
                card[1].extend(cont[:8] + [''] * (8 - len(cont)))
                continue
            name, fields = _split_fields(line)
            if name.endswith('*') or line[0] == '*':
                raise NotImplementedError('Large field format cards are not supported: ' + name)
            card = (name.upper(), fields)
            cards.append(card)
    return cards, case_control


def read_dat(path):
    """
    Read grids, coordinate systems, rod/bar/beam elements and single point
    constraints from a MYSTRAN or Nastran bulk data file.

    Parameters
    ----------
    path : str
        Path to the .dat file.

    Returns
    -------
    dict
        - ``grid_id`` : (ngrids,) grid ids in sorted order
        - ``grid_coordinate`` : (ngrids, 3) positions in the CP system
        - ``grid_cp`` : (ngrids,) position coordinate system ids
        - ``grid_cd`` : (ngrids,) displacement (output) coordinate system ids
        - ``coordinate_systems`` : dict of coordinate system id to definition
          (CORD1x/CORD2x cards)
        - ``element_id``, ``element_type``, ``element_connectivity`` : ids,
          SDynPy element types (11 rod, 21 beam) and (n, 2) grid ids
        - ``constraints`` : dict mapping grid id to a set of constrained
          components (1-6), from the SPC/SPC1 set selected in the case control
          (all sets if none is selected) and the GRID PS field
    """
    cards, case_control = _read_cards(path)
    grids = {}
    elements = {}
    spc = []
    ps = {}
    cords = {}
    for name, f in cards:
        if name in ('CORD2R', 'CORD2C', 'CORD2S'):
            f = f + [''] * (11 - len(f))
            cords[_to_int(f[0])] = {
                'type': _CS_TYPES[name[-1]], 'rid': _to_int(f[1]),
                'points': [[_to_float(v) for v in f[i:i + 3]] for i in (2, 5, 8)]}
        elif name in ('CORD1R', 'CORD1C', 'CORD1S'):
            for k in (0, 4):
                if f[k] != '':
                    cords[_to_int(f[k])] = {
                        'type': _CS_TYPES[name[-1]],
                        'grids': [_to_int(v) for v in f[k + 1:k + 4]]}
        elif name == 'GRID':
            gid = _to_int(f[0])
            grids[gid] = (_to_int(f[1]), [_to_float(v) for v in f[2:5]], _to_int(f[5]))
            if f[6]:
                ps[gid] = {int(c) for c in f[6] if c in '123456'}
        elif name in _LINE_ELEMENTS:
            ga, gb = (f[1], f[2]) if name == 'CONROD' else (f[2], f[3])
            elements[_to_int(f[0])] = (_LINE_ELEMENTS[name], (_to_int(ga), _to_int(gb)))
        elif name == 'SPC1':
            spc.append((_to_int(f[0]), f[1], [v for v in f[2:] if v != '']))
        elif name == 'SPC':
            for g, c in ((1, 2), (4, 5)):
                if f[g] != '':
                    spc.append((_to_int(f[0]), f[c], [f[g]]))
    selected = None
    for line in case_control:
        m = re.match(r'^\s*SPC\s*=\s*(\d+)', line, re.IGNORECASE)
        if m:
            selected = int(m.group(1))
    grid_ids = np.array(sorted(grids), dtype=int)
    constraints = {}
    for sid, comps, nodes in spc:
        if selected is not None and sid != selected:
            continue
        if 'THRU' in [n.upper() for n in nodes]:
            i = [n.upper() for n in nodes].index('THRU')
            lo, hi = int(nodes[i - 1]), int(nodes[i + 1])
            nodes = nodes[:i - 1] + [str(g) for g in grid_ids if lo <= g <= hi] + nodes[i + 2:]
        for n in nodes:
            constraints.setdefault(int(n), set()).update(int(c) for c in comps if c in '123456')
    for gid, comps in ps.items():
        constraints.setdefault(gid, set()).update(comps)
    element_ids = np.array(sorted(elements), dtype=int)
    return {
        'grid_id': grid_ids,
        'grid_coordinate': np.array([grids[g][1] for g in grid_ids]).reshape(-1, 3),
        'grid_cp': np.array([grids[g][0] for g in grid_ids], dtype=int),
        'grid_cd': np.array([grids[g][2] for g in grid_ids], dtype=int),
        'coordinate_systems': cords,
        'element_id': element_ids,
        'element_type': np.array([elements[e][0] for e in element_ids], dtype=int),
        'element_connectivity': np.array([elements[e][1] for e in element_ids], dtype=int).reshape(-1, 2),
        'constraints': constraints,
    }


def free_dofs(model):
    """
    Return the unconstrained (L-set) degrees of freedom of a model.

    Returns
    -------
    nodes, directions : np.ndarray
        Grid ids and directions (1-6) in grid-major order.
    """
    nodes = []
    directions = []
    for g in model['grid_id']:
        fixed = model['constraints'].get(int(g), set())
        for d in range(1, 7):
            if d not in fixed:
                nodes.append(g)
                directions.append(d)
    return np.array(nodes, dtype=int), np.array(directions, dtype=int)


# ---------------------------------------------------------------------------
# SDynPy objects
# ---------------------------------------------------------------------------

def _cartesian(cs_type, p):
    """Convert a point in a rectangular/cylindrical/spherical system to rectangular."""
    p = np.asarray(p, dtype=float)
    if cs_type == 1:
        t = np.radians(p[1])
        return np.array([p[0] * np.cos(t), p[0] * np.sin(t), p[2]])
    if cs_type == 2:
        t, f = np.radians(p[1]), np.radians(p[2])
        return p[0] * np.array([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)])
    return p


def _resolve_coordinate_systems(model):
    """
    Express every CORD1x/CORD2x system relative to the basic system.

    Returns a dict mapping coordinate system id to (cs_type, matrix) where
    matrix is the (4, 3) SDynPy matrix: rows 0-2 are the local x, y, z axes
    and row 3 is the origin, all in basic coordinates.
    """
    cards = model['coordinate_systems']
    grid_index = {int(g): i for i, g in enumerate(model['grid_id'])}
    resolved = {}
    active = set()

    def to_basic(cid, p):
        if cid == 0:
            return np.asarray(p, dtype=float)
        cs_type, matrix = resolve(cid)
        return matrix[:3].T @ _cartesian(cs_type, p) + matrix[3]

    def grid_basic(gid):
        if gid not in grid_index:
            raise ValueError('Coordinate system references undefined grid {:}'.format(gid))
        i = grid_index[gid]
        return to_basic(int(model['grid_cp'][i]), model['grid_coordinate'][i])

    def resolve(cid):
        if cid in resolved:
            return resolved[cid]
        if cid not in cards:
            raise ValueError('Coordinate system {:} is referenced but not defined'.format(cid))
        if cid in active:
            raise ValueError('Circular coordinate system definition at {:}'.format(cid))
        active.add(cid)
        card = cards[cid]
        if 'grids' in card:
            a, b, c = (grid_basic(g) for g in card['grids'])
        else:
            a, b, c = (to_basic(card['rid'], p) for p in card['points'])
        z = b - a
        y = np.cross(z, c - a)
        if np.linalg.norm(z) == 0 or np.linalg.norm(y) == 0:
            raise ValueError('Coordinate system {:} is degenerate'.format(cid))
        z = z / np.linalg.norm(z)
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)
        active.discard(cid)
        resolved[cid] = (card['type'], np.array([x, y, z, a]))
        return resolved[cid]

    for cid in cards:
        resolve(cid)
    return resolved


def geometry_from_model(model):
    """
    Create an SDynPy Geometry from the output of :func:`read_dat`.

    Every CORD1x/CORD2x system becomes a CoordinateSystemArray entry with the
    same id.  The basic system (id 0) is given id 1, or the next unused id if
    the model defines its own system 1.  The GRID CP field becomes the node
    ``def_cs`` and the CD field becomes ``disp_cs``, the frame the results
    (and therefore the shape DOFs) are expressed in.  Rods become type 11 and
    bars/beams type 21.
    """
    resolved = _resolve_coordinate_systems(model)
    basic_id = 1 if 1 not in resolved else max(resolved) + 1
    id_map = {0: basic_id}
    id_map.update({cid: cid for cid in resolved})
    used = set(int(c) for c in model['grid_cp']) | set(int(c) for c in model['grid_cd'])
    missing = sorted(used - set(id_map))
    if missing:
        raise ValueError('GRID cards reference undefined coordinate systems {:}'.format(missing))
    order = sorted(resolved)
    cs = coordinate_system_array(
        np.array([basic_id] + order),
        np.array(['Global'] + ['CORD {:}'.format(c) for c in order]), 1,
        np.array([0] + [resolved[c][0] for c in order]),
        np.array([np.concatenate((np.eye(3), np.zeros((1, 3))))] + [resolved[c][1] for c in order]))
    nodes = node_array(model['grid_id'], model['grid_coordinate'], 1,
                       np.array([id_map[int(c)] for c in model['grid_cp']], dtype=int),
                       np.array([id_map[int(c)] for c in model['grid_cd']], dtype=int))
    if model['element_id'].size:
        elements = element_array(model['element_id'], model['element_type'], 1,
                                 model['element_connectivity'])
        return Geometry(nodes, cs, element=elements)
    return Geometry(nodes, cs)


def _select_dofs(grid_id, shape, nodes, directions):
    """Pick (nmodes, ndof) shape coefficients for the node/direction pairs."""
    index = {int(g): i for i, g in enumerate(grid_id)}
    cols = [index[int(n)] * 6 + int(d) - 1 for n, d in zip(nodes, directions)]
    return shape.reshape(shape.shape[0], -1)[:, cols]


def shapes_from_modes(modes, model=None, damping=0.0):
    """
    Create a ShapeArray from the dictionary returned by an F06 or PCH reader.

    If ``model`` (output of :func:`read_dat`) is given only unconstrained
    degrees of freedom are kept, otherwise all six DOFs of every grid are.
    """
    if model is None:
        nodes = np.repeat(modes['grid_id'], 6)
        directions = np.tile(np.arange(1, 7), modes['grid_id'].size)
    else:
        nodes, directions = free_dofs(model)
    matrix = _select_dofs(modes['grid_id'], modes['shape'], nodes, directions)
    return shape_array(coordinate_array(nodes, directions), matrix,
                       modes['frequency'], damping, modes['generalized_mass'])
