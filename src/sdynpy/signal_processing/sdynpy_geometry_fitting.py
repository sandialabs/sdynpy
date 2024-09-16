# -*- coding: utf-8 -*-
"""
Functions to fit geometry to point data

This module defines a Geometry object as well as all of the subcomponents of
a geometry object: nodes, elements, tracelines and coordinate system.  Geometry
plotting is also handled in this module.

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
import scipy.optimize as opt


def cone_fit(points, origin, direction, angle):
    a = origin
    p = points
    n = direction
    height = np.einsum('...i,i->...', (a - p), n)
    distance = np.linalg.norm((a-p) - (height[:, np.newaxis])*n, axis=1)
    expected_distance = -np.tan(angle)*height
    return np.linalg.norm((distance-expected_distance))


def create_cone(ncircum, naxial, height, angle, origin, direction):
    circum_angles = np.linspace(0, 2*np.pi, ncircum, endpoint=False)
    heights = np.linspace(0, height, naxial+1)[1:]
    points = []
    for i, cangle in enumerate(circum_angles):
        for j, h in enumerate(heights):
            radius = np.tan(angle)*h
            points.append([np.sin(cangle)*radius, np.cos(cangle)*radius, h])
    points = np.array(points)
    rotation_matrix_z = direction/np.linalg.norm(direction)
    rotation_matrix_y = np.cross(rotation_matrix_z, [1.0, 0.0, 0.0])
    if np.linalg.norm(rotation_matrix_y) < 1e-10:
        rotation_matrix_x = np.cross([0.0, 1.0, 0.0], rotation_matrix_z)
        rotation_matrix_x /= np.linalg.norm(rotation_matrix_x)
        rotation_matrix_y = np.cross(rotation_matrix_z, rotation_matrix_x)
        rotation_matrix_y /= np.linalg.norm(rotation_matrix_y)
    else:
        rotation_matrix_y /= np.linalg.norm(rotation_matrix_y)
        rotation_matrix_x = np.cross(rotation_matrix_y, rotation_matrix_z)
        rotation_matrix_x /= np.linalg.norm(rotation_matrix_x)
    R = np.array((rotation_matrix_x, rotation_matrix_y, rotation_matrix_z)).T
    new_points = ((R@points.T).T+origin)
    return new_points


def cone_error_fn_fixed_angle(points, angle):
    return lambda x: cone_fit(points, x[:3], x[3:], angle)


def cone_error_fn_free_angle(points):
    return lambda x: cone_fit(points, x[:3], x[3:6], x[6])


def fit_cone_fixed_angle(points, angle):

    def constraint_eqn(x):
        return np.linalg.norm(x[3:])

    opt_fn = cone_error_fn_fixed_angle(points, angle)
    origin_guess = np.mean(points, axis=0)
    u, s, vh = np.linalg.svd(points-origin_guess)
    direction_guess = vh[0]
    x0_1 = np.concatenate((origin_guess, direction_guess))
    x0_2 = np.concatenate((origin_guess, -direction_guess))
    # Constrain to a unit vector
    constraint = opt.NonlinearConstraint(constraint_eqn, 1.0, 1.0)
    result_1 = opt.minimize(opt_fn, x0_1, method='trust-constr', constraints=constraint)
    result_2 = opt.minimize(opt_fn, x0_2, method='trust-constr', constraints=constraint)
    # Figure out which is the better one
    result = result_1 if result_1.fun < result_2.fun else result_2
    cone_origin = result.x[:3]
    cone_direction = result.x[3:]
    return result.fun, cone_origin, cone_direction


def distance_point_line(points, origin, direction):
    return np.linalg.norm(np.cross((points-origin), direction), axis=-1)/np.linalg.norm(direction)


def distance_point_plane(point, plane_point, plane_direction):
    return abs(np.einsum('...i,...i->...', point-plane_point, plane_direction))/np.linalg.norm(plane_direction, axis=-1)


def cylinder_fit(points, origin, direction, radius):
    return np.linalg.norm(distance_point_line(points, origin, direction) - radius)


def fit_cylinder(points, origin_guess=None, direction_guess=None, radius_guess=None):
    def opt_fn(x):
        return cylinder_fit(points, x[:3], x[3:6], x[6:7])

    def constraint_eqn(x):
        return np.linalg.norm(x[3:6])

    if origin_guess is None:
        origin_guess = np.mean(points, axis=0)
    if direction_guess is None or direction_guess == 'largest':
        u, s, vh = np.linalg.svd(points-origin_guess)
        direction_guess = vh[0]
    elif direction_guess == 'smallest':
        u, s, vh = np.linalg.svd(points-origin_guess)
        direction_guess = vh[-1]
    if radius_guess is None:
        u, s, vh = np.linalg.svd(points-origin_guess)
        radius_guess = s[0]/2
    x0 = np.concatenate((origin_guess, direction_guess, [radius_guess]))
    # Constrain to a unit vector
    constraint = opt.NonlinearConstraint(constraint_eqn, 1.0, 1.0)
    result = opt.minimize(opt_fn, x0, method='trust-constr', constraints=constraint)
    origin = result.x[:3]
    direction = result.x[3:6]
    radius = result.x[6]
    return result.fun, origin, direction, radius


def fit_line_point_cloud(points):
    center = np.mean(points, axis=0)
    shifted_points = points-center
    cov = points.T@shifted_points
    evals, evects = np.linalg.eigh(cov)
    direction = evects[:, np.argmax(evals)]
    direction /= np.linalg.norm(direction)
    return lambda t: center+t*direction
