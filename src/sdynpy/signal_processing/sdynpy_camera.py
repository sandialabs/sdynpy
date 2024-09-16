# -*- coding: utf-8 -*-
"""
Functions for dealing with camera data

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
import scipy.linalg as la
from scipy.optimize import least_squares
from .sdynpy_rotation import matrix_to_rodrigues, rodrigues_to_matrix


def camera_matrix_from_image(image_points, spatial_points):
    '''Computes camera matrices from at least six points

    This function computes camera matrices K (intrinic parameters) and R|T
    (extrinsic parameters) from a set of points defined on the image (u,v) and
    in 3D space (x,y,z).  It sets up a system of homogeneous linear equations
    for the camera parameters and solves for them using the singular value
    decomposition to extract the left singular vector corresponding to the
    smallest singular value.  This vector is the closest vector to the null
    space of the system of equations.

    Parameters
    ----------
    image_points : ndarray
        A n x 2 array where n is the number of points being correlated.  The
        first column is the horizontal pixel index from the left side of the
        image, and the second column is the vertical pixel index from the top
        of the image.  Pixel locations can be extracted from software such as
        Paint or GIMP, or if the image is plotted using
        matplotlib.pyplot.imshow.
    spatial_points : ndarray
        A n x 3 array where n is the number of points being correlated.  The
        rows of spatial_points correpond to the rows of image_points: for a
        point spatial_points[i,:] in 3D space, its position in the image in
        pixels is image_points[i,:].

    Returns
    -------
    K : ndarray
        A 3 x 3 upper triangular array consisting of the camera intrinisc
        parameters.
    RT : ndarray
        A 3 x 4 array where the first three columns are an orthogonal matrix
        denoting a rotation matrix, and the last column is a translation.

    Notes
    -----
    The return matrix K is of the form::

            [f_u  s  u_0]
        K = [ 0  f_v v_0]
            [ 0   0   1 ]

    And the matrix RT is of the form::

                     [r_xx r_xy r_xz | t_x]
        RT = [R|T] = [r_yx r_yy r_yz | t_y]
                     [r_zx r_zy r_zz | t_z]

    And satisfy the equation::

        c*[u v 1].T = K@RT@[x y z 1].T

    '''
    # Construct the coefficient matrix
    n_points = len(image_points)
    coefficient_matrix = np.zeros((2 * n_points, 12))
    for i, ([u, v], [x, y, z]) in enumerate(zip(image_points, spatial_points)):
        coefficient_matrix[2 * i:2 * i + 2, :] = np.array([[x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u],
                                                           [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v]])
    # Solve the homogeneous linear equations
    [U, S, Vh] = np.linalg.svd(coefficient_matrix)
    V = np.conj(Vh.T)
    m = V[:, -1]

    # Reshape back from vectorized form
    M = m.reshape(3, 4)

    # Decompose the matrix into an orthonormal and upper triangular matrix
    A = M[:3, :3]

    [K, R] = la.rq(A)

    # Make sure that the intrinsic matrix is physical (positive focal lengths)
    correction = np.diag([1 if K[i, i] > 0 else -1 for i in range(K.shape[0])])
    if np.linalg.det(correction) < 0:
        correction *= -1

    K = K @ correction
    R = correction @ R

    # Assemble the final matrices
    t = np.linalg.solve(K, M[:, -1, np.newaxis])
    RT = np.concatenate((R, t), axis=-1)

    # Normalize K
    K = K / K[2, 2]

    # Return values
    return K, RT


def compute_pixel_position(K, RT, coords):
    '''Compute pixel coordinates from world coordinates

    This function computes pixel coordinates from world coordinates and camera
    matrices.

    Parameters
    ----------
    K : np.ndarray
        A 3x3 array of camera intrinsic parameters
    RT : np.ndarray
        A 3x4 array of camera extrinsic parameters
    coords : np.ndarray
        A 3xn array of coordinates

    Returns
    -------
    pixel_position : np.ndarray
        A (mx)2xn array of pixel positions

    '''
    coords_shape = np.array(coords.shape).copy()
    coords_shape[-2] = 1
    coords_homo = np.concatenate((coords, np.ones(coords_shape)), axis=-2)
    pixels = K @ RT @ coords_homo
    pixels = pixels[..., :2, :] / pixels[..., 2, np.newaxis, :]
    return pixels


def compute_pixel_displacement(K, RT, coords, disp):
    '''Compute pixel displacements from coordinate displacements

    This function computes pixel displacements from coordinate displacements

    Parameters
    ----------
    K : np.ndarray
        A 3x3 array of camera intrinsic parameters
    RT : np.ndarray
        A 3x4 array of camera extrinsic parameters
    coords : np.ndarray
        A 3xn array of coordinates
    disp : np.ndarray
        A (mx)3xn array of displacements

    Returns
    -------
    pixel_disp : np.ndarray
        A (mx)2xn array of pixel displacements

    '''
    initial_pixels = compute_pixel_position(K, RT, coords)
    displaced_pixels = compute_pixel_position(K, RT, disp + coords)
    return displaced_pixels - initial_pixels


def camera_derivative_matrix(K, RT, x, y, z):
    '''Returns a matrix of derivatives du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz

    Derived from derivatives of camera equation matrices.  See
    displacement_derivatives.py.
    '''
    return np.array(
        [[(K[0, 0] * RT[0, 0] + K[0, 1] * RT[1, 0] + K[0, 2] * RT[2, 0])
          / (x * K[2, 2] * RT[2, 0] + y * K[2, 2] * RT[2, 1] + z * K[2, 2] * RT[2, 2] + K[2, 2] * RT[2, 3])
          - (x * (K[0, 0] * RT[0, 0] + K[0, 1] * RT[1, 0] + K[0, 2] * RT[2, 0])
             + y * (K[0, 0] * RT[0, 1] + K[0, 1] * RT[1, 1] + K[0, 2] * RT[2, 1])
             + z * (K[0, 0] * RT[0, 2] + K[0, 1] * RT[1, 2] + K[0, 2] * RT[2, 2])
             + K[0, 0] * RT[0, 3] + K[0, 1] * RT[1, 3] + K[0, 2] * RT[2, 3]) * K[2, 2] * RT[2, 0]
          / (x * K[2, 2] * RT[2, 0] + y * K[2, 2] * RT[2, 1] + z * K[2, 2] * RT[2, 2] + K[2, 2] * RT[2, 3])**2,
          (K[0, 0] * RT[0, 1] + K[0, 1] * RT[1, 1] + K[0, 2] * RT[2, 1])
          / (x * K[2, 2] * RT[2, 0] + y * K[2, 2] * RT[2, 1] + z * K[2, 2] * RT[2, 2] + K[2, 2] * RT[2, 3])
          - (x * (K[0, 0] * RT[0, 0] + K[0, 1] * RT[1, 0] + K[0, 2] * RT[2, 0])
             + y * (K[0, 0] * RT[0, 1] + K[0, 1] * RT[1, 1] + K[0, 2] * RT[2, 1])
             + z * (K[0, 0] * RT[0, 2] + K[0, 1] * RT[1, 2] + K[0, 2] * RT[2, 2])
             + K[0, 0] * RT[0, 3] + K[0, 1] * RT[1, 3] + K[0, 2] * RT[2, 3]) * K[2, 2] * RT[2, 1]
          / (x * K[2, 2] * RT[2, 0] + y * K[2, 2] * RT[2, 1] + z * K[2, 2] * RT[2, 2] + K[2, 2] * RT[2, 3])**2,
          (K[0, 0] * RT[0, 2] + K[0, 1] * RT[1, 2] + K[0, 2] * RT[2, 2])
          / (x * K[2, 2] * RT[2, 0] + y * K[2, 2] * RT[2, 1] + z * K[2, 2] * RT[2, 2] + K[2, 2] * RT[2, 3])
          - (x * (K[0, 0] * RT[0, 0] + K[0, 1] * RT[1, 0] + K[0, 2] * RT[2, 0])
             + y * (K[0, 0] * RT[0, 1] + K[0, 1] * RT[1, 1] + K[0, 2] * RT[2, 1])
             + z * (K[0, 0] * RT[0, 2] + K[0, 1] * RT[1, 2] + K[0, 2] * RT[2, 2])
             + K[0, 0] * RT[0, 3] + K[0, 1] * RT[1, 3] + K[0, 2] * RT[2, 3]) * K[2, 2] * RT[2, 2]
          / (x * K[2, 2] * RT[2, 0] + y * K[2, 2] * RT[2, 1] + z * K[2, 2] * RT[2, 2] + K[2, 2] * RT[2, 3])**2
          ],
         [(K[1, 1] * RT[1, 0] + K[1, 2] * RT[2, 0])
          / (x * K[2, 2] * RT[2, 0] + y * K[2, 2] * RT[2, 1] + z * K[2, 2] * RT[2, 2] + K[2, 2] * RT[2, 3])
          - (x * (K[1, 1] * RT[1, 0] + K[1, 2] * RT[2, 0])
             + y * (K[1, 1] * RT[1, 1] + K[1, 2] * RT[2, 1])
             + z * (K[1, 1] * RT[1, 2] + K[1, 2] * RT[2, 2])
             + K[1, 1] * RT[1, 3] + K[1, 2] * RT[2, 3]) * K[2, 2] * RT[2, 0]
          / (x * K[2, 2] * RT[2, 0] + y * K[2, 2] * RT[2, 1] + z * K[2, 2] * RT[2, 2] + K[2, 2] * RT[2, 3])**2,
          (K[1, 1] * RT[1, 1] + K[1, 2] * RT[2, 1])
          / (x * K[2, 2] * RT[2, 0] + y * K[2, 2] * RT[2, 1] + z * K[2, 2] * RT[2, 2] + K[2, 2] * RT[2, 3])
          - (x * (K[1, 1] * RT[1, 0] + K[1, 2] * RT[2, 0])
             + y * (K[1, 1] * RT[1, 1] + K[1, 2] * RT[2, 1])
             + z * (K[1, 1] * RT[1, 2] + K[1, 2] * RT[2, 2])
             + K[1, 1] * RT[1, 3] + K[1, 2] * RT[2, 3]) * K[2, 2] * RT[2, 1]
          / (x * K[2, 2] * RT[2, 0] + y * K[2, 2] * RT[2, 1] + z * K[2, 2] * RT[2, 2] + K[2, 2] * RT[2, 3])**2,
          (K[1, 1] * RT[1, 2] + K[1, 2] * RT[2, 2])
          / (x * K[2, 2] * RT[2, 0] + y * K[2, 2] * RT[2, 1] + z * K[2, 2] * RT[2, 2] + K[2, 2] * RT[2, 3])
          - (x * (K[1, 1] * RT[1, 0] + K[1, 2] * RT[2, 0]) + y * (K[1, 1] * RT[1, 1] + K[1, 2] * RT[2, 1])
             + z * (K[1, 1] * RT[1, 2] + K[1, 2] * RT[2, 2]) + K[1, 1] * RT[1, 3] + K[1, 2] * RT[2, 3]) * K[2, 2] * RT[2, 2]
          / (x * K[2, 2] * RT[2, 0] + y * K[2, 2] * RT[2, 1] + z * K[2, 2] * RT[2, 2] + K[2, 2] * RT[2, 3])**2
          ]]
    )


def homogeneous_coordinates(array):
    shape = np.array(array.shape).copy()
    shape[-2] = 1
    return np.concatenate((array, np.ones(shape)), axis=-2)


def unhomogeneous_coordinates(array):
    return array[..., :-1, :] / array[..., -1:, :]


def point_on_pixel(K, RT, pixel, distance=1):
    """
    Compute the 3D position of a point on a pixel of a given camera

    Parameters
    ----------
    K : np.ndarray
        A 3x3 array of camera intrinsic parameters
    RT : np.ndarray
        A 3x4 array of camera extrinsic parameters
    pixel : ndarray
        A (...,2) shape ndarray where the last dimension contains pixel positions
    distance : float, optional
        The distance that the point will be positioned from the camera pinhole.
        The default is 1.

    Returns
    -------
    pixel_coordinates : ndarray
        An array of 3D coordinates with the same shape as the original pixel
        array, where the 2D pixel position has been replaced with a 3D X,Y,Z
        position of the point

    """
    pixel = np.array(pixel)
    pixel_center_positions = pixel - K[:2, 2]

    ray_positions = np.concatenate((
        np.linalg.solve(K[:2, :2], pixel_center_positions[..., np.newaxis]),
        np.ones(pixel.shape[:-1] + (1, 1))
    ), axis=-2)

    ray_positions_homogeneous = np.concatenate((
        ray_positions,
        np.ones(pixel.shape[:-1] + (1, 1))
    ), axis=-2)

    RT_homogeneous = np.concatenate((RT, np.array([[0, 0, 0, 1]])), axis=-2)

    ray_positions_3D_homogeneous = np.linalg.solve(RT_homogeneous, ray_positions_homogeneous)
    # Convert back to normal coordinates
    ray_positions_3D = ray_positions_3D_homogeneous[...,
                                                    :3, :] / ray_positions_3D_homogeneous[..., 3:, :]

    pinhole = -RT[:3, :3].T @ RT[:3, 3:]

    ray_unit_vectors = ray_positions_3D - pinhole
    ray_unit_vectors = ray_unit_vectors / np.linalg.norm(ray_unit_vectors, axis=-2, keepdims=True)

    pixel_coordinates = distance * ray_unit_vectors + pinhole
    return pixel_coordinates[..., 0]


def distort_points(image_points, K=None, k=np.zeros(6), p=np.zeros(2), s=np.zeros(4)):
    if K is None:
        normalized_image_points = image_points
    else:
        normalized_image_points = image_points - K[..., :2, 2]
        normalized_image_points = np.einsum(
            '...ij,...j->...i', np.linalg.inv(K[..., :2, :2]), normalized_image_points)
    r = np.linalg.norm(normalized_image_points, axis=-1)
    r2 = r**2
    r4 = r**4
    r6 = r**6
    xp = normalized_image_points[..., 0]
    yp = normalized_image_points[..., 1]
    xp2 = xp**2
    yp2 = yp**2
    xpp = (xp * (1 + k[..., 0] * r2 + k[..., 1] * r4 + k[..., 2] * r6) / (1 + k[..., 3] * r2 + k[..., 4] * r4 + k[..., 5] * r6)
           + 2 * p[..., 0] * xp * yp + p[..., 1] * (r2 + 2 * xp2)
           + s[..., 0] * r2 + s[..., 1] * r4)
    ypp = (yp * (1 + k[..., 0] * r2 + k[..., 1] * r4 + k[..., 2] * r6) / (1 + k[..., 3] * r2 + k[..., 4] * r4 + k[..., 5] * r6)
           + 2 * p[..., 1] * xp * yp + p[..., 0] * (r2 + 2 * yp2)
           + s[..., 2] * r2 + s[..., 3] * r4)
    distorted_normalized_points = np.concatenate(
        (xpp[..., np.newaxis], ypp[..., np.newaxis]), axis=-1)
    if K is None:
        return distorted_normalized_points
    else:
        return np.einsum('...ij,...j->...i', K[..., :2, :2], distorted_normalized_points) + K[..., :2, 2]


def calibration_linear_estimate(image_points, plane_points):
    """
    Computes a linear estimate of the camera parameters from point correspondences

    Uses pass in a set of points on images for multiple views of the same
    calibration object

    Parameters
    ----------
    image_points : ndarray
        A (n_camera x) n_image x n_point x 2 array of image points
    plane_points : ndarray
        A n_point x 2 array of 2D positions of the points on the calibration
        object

    Raises
    ------
    ValueError
        If the point correspondence matrices are not the correct sizes.

    Returns
    -------
    K_est
        A (n_camera x) 3 x 3 estimate of the camera intrinsic matrix
    RT_est
        A (n_camera x) n_image x 3 x 4 estimate of the camera extrinsic matrix
    """
    original_shape = image_points.shape[:-2]

    if image_points.ndim < 3:
        raise ValueError(
            'image_points must have shape (n_images x n_points x 2) or (n_cameras x n_images x n_points x 2)')
    if image_points.ndim == 3:
        image_points = image_points[np.newaxis, ...]

    n_cameras, n_images, n_points, _ = image_points.shape

    # First let's go through and find the mask where the points are not defined
    # or are off the image
    valid_points = ~np.any(np.isnan(image_points), axis=-1)

    # Now let's compute the homography
    plane_points_h = np.concatenate(
        (plane_points, np.ones(plane_points.shape[:-1] + (1,))), axis=-1).T
    zero = np.zeros(plane_points_h.shape)

    homography_guess_matrix = np.array([np.block([
        [plane_points_h.T, zero.T, -points[:, 0, np.newaxis] * plane_points_h.T],
        [zero.T, plane_points_h.T, -points[:, 1, np.newaxis] * plane_points_h.T]
    ]) for points in image_points.reshape(-1, n_points, 2)])

    # Go through and compute the homography transformation for each camera and each
    # image
    homography = []
    for index, (guess_matrix, mask) in enumerate(zip(homography_guess_matrix, valid_points.reshape(n_cameras * n_images, n_points))):
        guess_matrix = guess_matrix[np.concatenate((mask, mask))]
        if guess_matrix.shape[0] < 9:
            homography.append(np.nan * np.ones(9))
            continue
        column_preconditioner = np.diag(1 / np.sqrt(np.mean(guess_matrix**2, axis=-2)))
        row_preconditioner = np.diag(1 / np.sqrt(np.mean(guess_matrix**2, axis=-1)))
        homography_guess_matrix_normalized = row_preconditioner @ guess_matrix @ column_preconditioner
        _, _, Vh = np.linalg.svd(homography_guess_matrix_normalized)
        homography_guesses = column_preconditioner @ Vh[-1, :]
        # Now do the nonlinear updating
        valid_image_points = image_points[index // n_images, index % n_images, mask].T
        valid_plane_points = plane_points_h[:, mask]

        def optfunc(h_vect):
            return np.linalg.norm(valid_image_points - (h_vect[0:6].reshape(2, 3) @ valid_plane_points)
                                  / (h_vect[6:9].reshape(1, 3) @ valid_plane_points), axis=0)
        soln = least_squares(optfunc, homography_guesses, method='lm')
        homography.append(soln.x)

    homography = np.array(homography).reshape(n_cameras, n_images, 3, 3)

    # Set up constraint matrix (V in zhang)
    def intrinsic_constraint(i, j): return np.moveaxis(np.array([homography[..., 0, i] * homography[..., 0, j],
                                                                 homography[..., 0, i] * homography[..., 1, j] +
                                                                 homography[..., 1, i] *
                                                                 homography[..., 0, j],
                                                                 homography[..., 1, i] *
                                                                 homography[..., 1, j],
                                                                 homography[..., 2, i] * homography[..., 0, j] +
                                                                 homography[..., 0, i] *
                                                                 homography[..., 2, j],
                                                                 homography[..., 2, i] * homography[..., 1, j] +
                                                                 homography[..., 1, i] *
                                                                 homography[..., 2, j],
                                                                 homography[..., 2, i] * homography[..., 2, j]]), 0, -1)

    intrinsic_constraint_matrix = np.block([
        [intrinsic_constraint(0, 1)],
        [intrinsic_constraint(0, 0) - intrinsic_constraint(1, 1)]
    ])

    K_est = []
    RT_est = []

    for this_intrinsic_constraint_matrix, this_homography in zip(intrinsic_constraint_matrix, homography):
        good_image_rows = ~np.any(np.isnan(this_intrinsic_constraint_matrix), axis=1)
        U, S, Vh = np.linalg.svd(this_intrinsic_constraint_matrix[good_image_rows])
        # Intrinsic parameter vector (b in zhang)
        intrinsic_parameters = Vh[..., -1, :]

        [B11, B12, B22, B13, B23, B33] = intrinsic_parameters.T

        cy = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
        lam = B33 - (B13**2 + cy * (B12 * B13 - B11 * B23)) / B11
        fx = np.sqrt(lam / B11)
        fy = np.sqrt(lam * B11 / (B11 * B22 - B12**2))
        s = -B12 * fx**2 * fy / lam
        cx = s * cy / fx - B13 * fx**2 / lam

        K_est.append(np.array([
            [fx, s, cx],
            [0, fy, cy],
            [0, 0, 1]]))

        # extrinsics_estimate = np.linalg.solve(K_est[:,np.newaxis],homography)
        extrinsics_estimate = np.linalg.solve(K_est[-1], this_homography)
        this_RT = np.nan * np.ones((n_images, 3, 4))
        good_image_rows = np.where(~np.any(np.isnan(extrinsics_estimate), axis=(1, 2)))[0]
        extrinsics_estimate = extrinsics_estimate[good_image_rows]
        r1 = extrinsics_estimate[..., 0]
        r2 = extrinsics_estimate[..., 1]
        scale_factor_1 = np.linalg.norm(r1, axis=-1, keepdims=True)
        scale_factor_2 = np.linalg.norm(r2, axis=-1, keepdims=True)
        scale_factor = (scale_factor_1 + scale_factor_2) / 2
        r1 /= scale_factor
        r2 /= scale_factor
        r3 = np.cross(r1, r2)
        t = extrinsics_estimate[..., 2]
        t /= scale_factor
        U, S, Vh = np.linalg.svd(np.concatenate((r1[..., np.newaxis],
                                                 r2[..., np.newaxis],
                                                 r3[..., np.newaxis]), axis=-1))
        R_est = U @ Vh
        RT_est_0 = np.concatenate((R_est, t[..., np.newaxis]), axis=-1)

        r1 = -r1
        r2 = -r2
        t = -t
        r3 = np.cross(r1, r2)
        U, S, Vh = np.linalg.svd(np.concatenate((r1[..., np.newaxis],
                                                 r2[..., np.newaxis],
                                                 r3[..., np.newaxis]), axis=-1))
        R_est = U @ Vh
        RT_est_1 = np.concatenate((R_est, t[..., np.newaxis]), axis=-1)

        check_points = RT_est_0 @ np.concatenate((plane_points, np.zeros(
            (plane_points.shape[0], 1)), np.ones((plane_points.shape[0], 1))), axis=-1).T
        use_negative = np.any(check_points[..., 2, :] < 0, axis=-1)

        this_RT[good_image_rows[use_negative]] = RT_est_1[use_negative]
        this_RT[good_image_rows[~use_negative]] = RT_est_0[~use_negative]

        RT_est.append(this_RT)

    return np.array(K_est).reshape(*original_shape[:-1], 3, 3), np.array(RT_est).reshape(*original_shape, 3, 4)


def reconstruct_scene_from_calibration_parameters(parameter_array, n_cameras, n_images,
                                                  radial_distortions=0,
                                                  prismatic_distortions=0,
                                                  tangential_distortions=0,
                                                  radial_denominator_distortions=False,
                                                  use_K_for_distortions=True):
    index = 0
    # Intrinsic Parameters
    n_parameters = n_cameras * 5
    K = np.tile(np.eye(3), (n_cameras, 1, 1))
    K[:, [0, 1, 0, 0, 1], [0, 1, 1, 2, 2]] = parameter_array[index:index +
                                                             n_parameters].reshape(n_cameras, 5)
    index += n_parameters
    # Camera extrinsics
    RT_cameras = np.tile(np.eye(3, 4), (n_cameras, 1, 1))
    # Camera Rotations
    n_parameters = (n_cameras - 1) * 3
    rvecs = parameter_array[index:index + n_parameters].reshape((n_cameras - 1), 3)
    R = rodrigues_to_matrix(rvecs)
    RT_cameras[1:, :3, :3] = R
    index += n_parameters
    # Camera translations
    RT_cameras[1:, :, 3] = parameter_array[index:index + n_parameters].reshape(n_cameras - 1, 3)
    index += n_parameters
    # Image extrinsics
    RT_images = np.tile(np.eye(4, 4), (n_images, 1, 1))
    # Image rotations
    n_parameters = n_images * 3
    rvecs = parameter_array[index:index + n_parameters].reshape(n_images, 3)
    R = rodrigues_to_matrix(rvecs)
    RT_images[:, :3, :3] = R
    index += n_parameters
    # Image translations
    RT_images[:, :3, 3] = parameter_array[index:index + n_parameters].reshape(n_images, 3)
    index += n_parameters
    # Radial distortion parameters
    radial_distortion_parameters = np.zeros((n_cameras, 6))
    if radial_denominator_distortions:
        n_parameters = radial_distortions * 2 * n_cameras
        params = parameter_array[index:index +
                                 n_parameters].reshape(n_cameras, radial_distortions * 2)
        radial_distortion_parameters[:, :radial_distortions] = params[:, :params.shape[-1] // 2]
        radial_distortion_parameters[:, 3:3 +
                                     radial_distortions] = params[:, params.shape[-1] // 2:]
    else:
        n_parameters = radial_distortions * n_cameras
        params = parameter_array[index:index + n_parameters].reshape(n_cameras, radial_distortions)
        radial_distortion_parameters[:, :radial_distortions] = params
    index += n_parameters
    # Prismatic distortions
    prismatic_distortion_parameters = np.zeros((n_cameras, 4))
    n_parameters = prismatic_distortions * 2 * n_cameras
    prismatic_distortion_parameters[:, :prismatic_distortions *
                                    2] = parameter_array[index:index + n_parameters].reshape(n_cameras, prismatic_distortions * 2)
    if prismatic_distortions == 1:
        prismatic_distortion_parameters = prismatic_distortion_parameters[:, [0, 2, 1, 3]]
    index += n_parameters
    # Tangential Distortions
    tangential_distortion_parameters = np.zeros((n_cameras, 2))
    n_parameters = tangential_distortions * n_cameras
    tangential_distortion_parameters[:, :tangential_distortions] = parameter_array[index:index +
                                                                                   n_parameters].reshape(n_cameras, tangential_distortions)
    index += n_parameters
    # K matrix for distortions (if used)
    if use_K_for_distortions:
        K_distortion = None
    else:
        n_parameters = n_cameras * 5
        K_distortion = np.tile(np.eye(3), (n_cameras, 1, 1))
        K_distortion[:, [0, 1, 0, 0, 1], [0, 1, 1, 2, 2]
                     ] = parameter_array[index:index + n_parameters].reshape(n_cameras, 5)
        index += n_parameters
    return (K, RT_cameras, RT_images, radial_distortion_parameters,
            prismatic_distortion_parameters, tangential_distortion_parameters,
            K_distortion)


def optimize_calibration(image_points, plane_points, K_guess, RT_guess, radial_distortions=0,
                         prismatic_distortions=0, tangential_distortions=0,
                         radial_denominator_distortions=False,
                         K_guess_distortion=None, **least_squares_kwargs):
    n_cameras, n_images, n_points, _ = image_points.shape
    # Set up initial rotation guesses
    RT_guess_cameras = np.empty((n_cameras - 1, 3, 4))
    RT_guess_images = np.empty((n_images, 3, 4))
    valid_images = ~np.any(np.isnan(RT_guess), axis=(-1, -2))
    # Set up the initial camera transformations
    for i in range(n_cameras - 1):
        for j in range(n_images):
            if valid_images[i + 1, j] and valid_images[0, j]:
                global_transformation = np.linalg.inv(
                    np.concatenate((RT_guess[0, j], [[0, 0, 0, 1]])))
                RT_guess_cameras[i] = RT_guess[i + 1, j] @ global_transformation
    # Set up the initial image transformations
    for j in range(n_images):
        for i in range(n_cameras):
            if valid_images[i, j]:
                if i == 0:
                    RT_guess_images[j] = RT_guess[i, j]
                else:
                    RT_guess_images[j] = (np.linalg.inv(np.concatenate(
                        (RT_guess_cameras[i - 1], [[0, 0, 0, 1]]))) @ np.concatenate((RT_guess[i, j], [[0, 0, 0, 1]])))[:3]
    # # Check that this is right
    # RT_1 = np.concatenate((np.eye(3,4)[np.newaxis],RT_guess_cameras),axis=0)
    # RT_2 = np.concatenate((RT_guess_images,np.tile(np.array((0,0,0,1)),(15,1,1))),axis=-2)
    # RT_guess_check = RT_1[:,np.newaxis]@RT_2
    initial_guesses = np.concatenate([
        # Intrinsic parameters fx,fy,s,cx,cy
        K_guess[:, [0, 1, 0, 0, 1], [0, 1, 1, 2, 2]].flatten(),
        # Rotations between cameras coordinate systems
        matrix_to_rodrigues(RT_guess_cameras[..., :3, :3]).flatten(),
        RT_guess_cameras[..., :, 3].flatten(),  # Translations between camera coordinate systems
        # Rotations of plane at each image
        matrix_to_rodrigues(RT_guess_images[:, :3, :3]).flatten(),
        RT_guess_images[:, :3, 3].flatten(),  # Translations of plane at each image
        np.zeros(radial_distortions * (2 if radial_denominator_distortions else 1)
                 * n_cameras),  # Radial distortions
        np.zeros(prismatic_distortions * 2 * n_cameras),  # Prismatic distortions
        np.zeros(tangential_distortions * n_cameras),  # tangential distortions
        [] if K_guess_distortion is None else K_guess_distortion[:, [0, 1, 0, 0, 1], [0, 1, 1, 2, 2]].flatten()
    ])
    variable_scale = np.concatenate([
        1000 * np.ones(n_cameras * 5),  # Intrinsic parameters fx,fy,s,cx,cy, in the 1000s of pixels
        # Rotatons between cameras coordinate systems, generally between -pi and pi
        np.ones((n_cameras - 1) * 3),
        # Translations bewteen cameras, assume approximately 1 meter distances
        np.ones((n_cameras - 1) * 3),
        np.ones(n_images * 3),  # Rotations of the plane at each image, generally between -pi and pi
        # Translations of the plane at each image, assume 10 centimeters
        0.1 * np.ones(n_images * 3),
        0.1 * np.ones(radial_distortions * (2 if radial_denominator_distortions else 1)
                      * n_cameras),  # Radial distortions
        0.1 * np.ones(prismatic_distortions * 2 * n_cameras),  # Prismatic distortions
        0.1 * np.ones(tangential_distortions * n_cameras),  # tangential distortions
        # Intrinsic parameters used in distortion calculations
        [] if K_guess_distortion is None else 1000 * np.ones(n_cameras * 5)
    ])
    plane_points_h = np.concatenate((plane_points.T,
                                     np.zeros((1, plane_points.shape[0])),
                                     np.ones((1, plane_points.shape[0]))))
    print('Optimizing {:} degrees of freedom'.format(initial_guesses.size))

    def error_func(x):
        # Get scene parameters
        (K, RT_cameras, RT_images, radial_distortion_parameters,
         prismatic_distortion_parameters, tangential_distortion_parameters,
         K_distortion) = reconstruct_scene_from_calibration_parameters(
            x, n_cameras, n_images,
            radial_distortions,
            prismatic_distortions,
            tangential_distortions,
            radial_denominator_distortions,
            K_guess_distortion is None)
        # Project points through camera equations
        image_points_reproj_h = K[:, np.newaxis] @ RT_cameras[:,
                                                              np.newaxis] @ RT_images @ plane_points_h
        image_points_reproj = np.moveaxis(
            image_points_reproj_h[..., :2, :] / image_points_reproj_h[..., 2:, :], -2, -1)
        # Deform the points
        if np.any(radial_distortion_parameters != 0.0) or np.any(prismatic_distortion_parameters != 0.0) or np.any(prismatic_distortion_parameters != 0.0):
            image_points_reproj = distort_points(
                image_points_reproj,
                K[:, np.newaxis, np.newaxis] if K_distortion is None else K_distortion[:, np.newaxis, np.newaxis],
                radial_distortion_parameters[:, np.newaxis, np.newaxis],
                tangential_distortion_parameters[:, np.newaxis, np.newaxis],
                prismatic_distortion_parameters[:, np.newaxis, np.newaxis])
        # Compute the residual
        residual = image_points - image_points_reproj
        return residual[~np.isnan(residual)]

    # Now optimize
    soln = least_squares(error_func, initial_guesses, method='lm',
                         x_scale=variable_scale, **least_squares_kwargs)
    (K, RT_cameras, RT_images, radial_distortion_parameters,
     prismatic_distortion_parameters, tangential_distortion_parameters,
     K_distortion) = reconstruct_scene_from_calibration_parameters(
        soln.x, n_cameras, n_images,
        radial_distortions,
        prismatic_distortions,
        tangential_distortions,
        radial_denominator_distortions,
        K_guess_distortion is None)
    return (K, RT_cameras, RT_images, radial_distortion_parameters,
            prismatic_distortion_parameters, tangential_distortion_parameters,
            K_distortion)


def decomposeP(P):
    """
    Decomposes a projection matrix P into intrinsic and extrinsic matrices K, R, and t

    Parameters
    ----------
    P : np.ndarray
        3x4 camera projection matrix P = K@np.concatenate((R,t),axis=-1)

    Returns
    -------
    K : ndarray
        3x3 upper triangular intrinsic parameter matrix
    R : ndarray
        3x3 rotation matrix
    t : ndarray
        3x1 translation vector

    """
    M = P[0:3, 0:3]
    Q = np.eye(3)[:: -1]
    P_b = Q @ M @ M.T @ Q
    K_h = Q @ np.linalg.cholesky(P_b) @ Q
    K = K_h / K_h[2, 2]
    A = np.linalg .inv(K) @ M
    L = (1 / np.linalg.det(A))**(1 / 3)
    R = L * A
    t = L * np.linalg.inv(K) @ P[0:3, 3]
    return K, R, t
