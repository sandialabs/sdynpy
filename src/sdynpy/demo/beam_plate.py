# -*- coding: utf-8 -*-
"""
Demonstration system for a plate-like structure made of beams

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

import numpy as _np
from ..fem.sdynpy_beam import beamkm as _beamkm
from ..core.sdynpy_geometry import (Geometry as _Geometry,
                                    node_array as _node_array,
                                    traceline_array as _traceline_array,
                                    coordinate_system_array as _coordinate_system_array)
from ..core.sdynpy_coordinate import from_nodelist as _from_nodelist
from ..core.sdynpy_system import (System as _System,
                                  substructure_by_position as _substructure_by_position)


def create_models(width=1, length=0.5, grid=0.125,
                  beam_height=0.02, beam_width=0.02,
                  E=69e9, nu=0.33, rho=2830):
    nodes_width = int(_np.round(width / grid)) + 1
    nodes_length = int(_np.round(length / grid)) + 1
    coords = _np.moveaxis(_np.meshgrid(_np.linspace(0, width, nodes_width),
                                       _np.linspace(0, length, nodes_length),
                                       0,
                                       indexing='ij'), 0, -1)[:, :, 0, :]
    indices = _np.arange(_np.prod(coords.shape[:-1])).reshape(coords.shape[:-1])

    conn = []
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            try:
                conn.append([indices[i, j], indices[i, j + 1]])
            except IndexError:
                pass
            try:
                conn.append([indices[i, j], indices[i + 1, j]])
            except IndexError:
                pass
    conn = _np.array(conn)

    bend_direction_1 = _np.array(
        [_np.zeros(conn.shape[0]),
         _np.zeros(conn.shape[0]),
         _np.ones(conn.shape[0])]
    ).T

    I1 = beam_width * beam_height**3 / 12
    I2 = beam_width**3 * beam_height / 12
    G = E / (2 * (1 - nu))
    J = I1 + I2
    Ixx_per_L = (1 / 12) * rho * beam_width * beam_height * (beam_width**2 + beam_height**2)

    # Create arguments for the beamkm function
    ae = beam_width * beam_height * E * _np.ones(conn.shape[0])
    jg = J * G * _np.ones(conn.shape[0])
    ei1 = E * I1 * _np.ones(conn.shape[0])
    ei2 = E * I2 * _np.ones(conn.shape[0])
    mass_per_length = rho * beam_width * beam_height * _np.ones(conn.shape[0])
    tmmi_per_length = Ixx_per_L * _np.ones(conn.shape[0])

    # Compute K and M via beamkm
    this_K, this_M = _beamkm(coords.reshape(-1, 3), conn, bend_direction_1,
                             ae, jg, ei1, ei2, mass_per_length, tmmi_per_length)

    geometry = _Geometry(
        _node_array(_np.arange(coords.reshape(-1, 3).shape[0]) + 1, coords.reshape(-1, 3), 1),
        _coordinate_system_array(),
        _traceline_array(_np.arange(conn.shape[0]) + 1, color=1, connectivity=conn + 1)
    )

    system = _System(
        _from_nodelist(geometry.node.id, [1, 2, 3, 4, 5, 6]),
        this_M, this_K
    )

    return system, geometry


system, geometry = create_models()
