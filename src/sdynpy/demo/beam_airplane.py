# -*- coding: utf-8 -*-
"""
Demonstration system for a simple airplane made of beams

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
from ..core.sdynpy_coordinate import (from_nodelist as _from_nodelist,
                                      coordinate_array as _coordinate_array)
from ..core.sdynpy_system import (System as _System,
                                  substructure_by_position as _substructure_by_position)


def create_models():
    """
    Creates models for the demonstration airplane.

    Returns
    -------
    full_system : System
        `System` containing all the airplane degrees of freedom
    full_geometry : Geometry
        `Geometry` containing the full airplane model
    transmission_system : System
        `System` containing a shorter wing that can be used as a transmission
        simulator
    transmission_geometry : Geometry
        `Geometry` containing a shorter wing that can be used as a transmission
        simulator
    all_systems : dict[System]
        A dictionary containing component `System` objects for use in
        substructuring
    all_geometries : dict[Geometry]
        A dictionary containing component `Geometry` objects for use in
        substructuring

    """
    # Fuselage coords
    fuselage_coords = _np.array([_np.linspace(0, 10, 21),
                                _np.zeros(21),
                                _np.zeros(21)]).T

    # Wing coords
    wing_coords = _np.array([[x, y, 0] for x in [2, 3] for y in _np.linspace(-10, 10, 41)])

    # Transmission simulator
    transmission_simulator_coords = _np.array(
        [[x, y, 0] for x in [2, 3] for y in _np.linspace(-2.5, 2.5, 11)])

    # Tail Coords
    tail_coords = _np.array([[9, 2, 2], [9, 1, 1], [9, 0, 0], [9, -1, 1],
                            [9, -2, 2], [10, 2, 2], [10, 1, 1], [10, 0, 0], [10, -1, 1], [10, -2, 2]])

    # Create element connectivity
    fuselage_connectivity = _np.array(
        [_np.arange(0, fuselage_coords.shape[0] - 1), _np.arange(1, fuselage_coords.shape[0])]).T
    wing_connectivity = _np.concatenate((
        _np.array([_np.arange(0, wing_coords.shape[0] // 2 - 1),
                  _np.arange(1, wing_coords.shape[0] // 2)]).T,  # Front stringer
        _np.array([_np.arange(wing_coords.shape[0] // 2, wing_coords.shape[0] - 1),
                  _np.arange(wing_coords.shape[0] // 2 + 1, wing_coords.shape[0])]).T,  # Back stringer
        _np.array([_np.arange(0, wing_coords.shape[0] // 2),
                  _np.arange(wing_coords.shape[0] // 2, wing_coords.shape[0])]).T  # Spars
    ), axis=0)
    transmission_simulator_connectivity = _np.concatenate((
        _np.array([_np.arange(0, transmission_simulator_coords.shape[0] // 2 - 1),
                  _np.arange(1, transmission_simulator_coords.shape[0] // 2)]).T,  # Front stringer
        _np.array([_np.arange(transmission_simulator_coords.shape[0] // 2, transmission_simulator_coords.shape[0] - 1),
                  _np.arange(transmission_simulator_coords.shape[0] // 2 + 1, transmission_simulator_coords.shape[0])]).T,  # Back stringer
        _np.array([_np.arange(0, transmission_simulator_coords.shape[0] // 2),
                  _np.arange(transmission_simulator_coords.shape[0] // 2, transmission_simulator_coords.shape[0])]).T  # Spars
    ), axis=0)
    tail_connectivity = _np.concatenate((
        _np.array([_np.arange(0, tail_coords.shape[0] // 2 - 1),
                  _np.arange(1, tail_coords.shape[0] // 2)]).T,  # Front stringer
        _np.array([_np.arange(tail_coords.shape[0] // 2, tail_coords.shape[0] - 1),
                  _np.arange(tail_coords.shape[0] // 2 + 1, tail_coords.shape[0])]).T,  # Back stringer
        _np.array([_np.arange(0, tail_coords.shape[0] // 2),
                  _np.arange(tail_coords.shape[0] // 2, tail_coords.shape[0])]).T  # Spars
    ), axis=0)

    # Create the bending directions, we will do positive z for all
    fuselage_bend_direction_1 = _np.array(
        [_np.zeros(fuselage_connectivity.shape[0]),
         _np.zeros(fuselage_connectivity.shape[0]),
         _np.ones(fuselage_connectivity.shape[0])]
    ).T
    wing_bend_direction_1 = _np.array(
        [_np.zeros(wing_connectivity.shape[0]),
         _np.zeros(wing_connectivity.shape[0]),
         _np.ones(wing_connectivity.shape[0])]
    ).T
    transmission_simulator_bend_direction_1 = _np.array(
        [_np.zeros(transmission_simulator_connectivity.shape[0]),
         _np.zeros(transmission_simulator_connectivity.shape[0]),
         _np.ones(transmission_simulator_connectivity.shape[0])]
    ).T
    tail_bend_direction_1 = _np.array(
        [_np.zeros(tail_connectivity.shape[0]),
         _np.zeros(tail_connectivity.shape[0]),
         _np.ones(tail_connectivity.shape[0])]
    ).T

    # Define beam parameters
    beam_width = 0.1
    beam_height = 0.15
    E = 69e9
    nu = 0.33
    rho = 2830
    I1 = beam_width * beam_height**3 / 12
    I2 = beam_width**3 * beam_height / 12
    G = E / (2 * (1 - nu))
    J = I1 + I2
    Ixx_per_L = (1 / 12) * rho * beam_width * beam_height * (beam_width**2 + beam_height**2)

    # Rather than doing all the math to create matrices 3 times, let's just do it once in a for loop.
    # Let's use dictionaries to store the matrices
    all_systems = {}
    all_geometries = {}
    names = ['fuselage', 'wing', 'tail', 'transmission_simulator']
    colors = [1, 7, 11, 13]

    for coord, conn, bend_direction, name, color in zip([fuselage_coords, wing_coords, tail_coords, transmission_simulator_coords],
                                                        [fuselage_connectivity, wing_connectivity,
                                                            tail_connectivity, transmission_simulator_connectivity],
                                                        [fuselage_bend_direction_1, wing_bend_direction_1,
                                                            tail_bend_direction_1, transmission_simulator_bend_direction_1],
                                                        names, colors):
        # Create arguments for the beamkm function
        ae = beam_width * beam_height * E * _np.ones(conn.shape[0])
        jg = J * G * _np.ones(conn.shape[0])
        ei1 = E * I1 * _np.ones(conn.shape[0])
        ei2 = E * I2 * _np.ones(conn.shape[0])
        mass_per_length = rho * beam_width * beam_height * _np.ones(conn.shape[0])
        tmmi_per_length = Ixx_per_L * _np.ones(conn.shape[0])

        # Compute K and M via beamkm
        this_K, this_M = _beamkm(coord, conn, bend_direction, ae, jg, ei1,
                                 ei2, mass_per_length, tmmi_per_length)

        # Create objects
        # Create geometry
        all_geometries[name] = _Geometry(
            _node_array(_np.arange(coord.shape[0]) + 1, coord, color),
            _coordinate_system_array(),
            _traceline_array(_np.arange(conn.shape[0]) + 1, color=color, connectivity=conn + 1)
        )
        # Create system
        all_systems[name] = _System(
            _from_nodelist(all_geometries[name].node.id, [1, 2, 3, 4, 5, 6]),
            this_M, this_K
        )

    # Now let's create the full airplane
    systems = [all_systems[name] for name in names if name not in ['transmission_simulator']]
    geometries = [all_geometries[name] for name in names if name not in ['transmission_simulator']]

    full_system, full_geometry = _substructure_by_position(systems, geometries)

    # Now let's create the stubby airplane
    systems = [all_systems[name] for name in names if name not in ['wing']]
    geometries = [all_geometries[name] for name in names if name not in ['wing']]

    transmission_system, transmission_geometry = _substructure_by_position(systems, geometries)

    return full_system, full_geometry, transmission_system, transmission_geometry, all_systems, all_geometries


(system, geometry, transmission_system, transmission_geometry,
 component_systems, component_geometries) = create_models()
