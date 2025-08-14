# -*- coding: utf-8 -*-
"""
Functions for dealing with the geometry of rotations.

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
from scipy.optimize import NonlinearConstraint


def cross_mat(v):
    """
    Return matrix representation of the cross product of the given vector

    Parameters
    ----------
    v : np.ndarray
        3-vector that will be assembled into the matrix.

    Returns
    -------
    cross_matrix : np.ndarray
        Matrix representing the cross product.  np.cross(a,b) is equivalent to
        cross_mat(a)@b

    """
    v = np.array(v).flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def R(axis, angle, degrees=False):
    """
    Create a rotation matrix consisting of a rotation about an axis

    Parameters
    ----------
    axis : index or np.ndarray
        The axis of rotation.  Specifying 0, 1, or 2 will construct a rotation
        about the X, Y, or Z axes, respectively.  Alternatively, the axis of
        rotation can be specficied as a 3-vector.
    angle : float
        The angle of rotation
    degrees : bool, optional
        True if the angle value is specified in degrees, false if radians.
        The default is False.

    Raises
    ------
    ValueError
        If an improper axis is specified.

    Returns
    -------
    rotmat : np.ndarray
        A 3x3 rotation matrix.

    """
    rotmat = None
    if degrees:
        angle *= np.pi / 180
    s = np.sin(angle)
    c = np.cos(angle)
    try:
        if axis == 0:
            rotmat = np.array([[1, 0, 0],
                               [0, c, -s],
                               [0, s, c]])
        elif axis == 1:
            rotmat = np.array([[c, 0, s],
                               [0, 1, 0],
                               [-s, 0, c]])
        elif axis == 2:
            rotmat = np.array([[c, -s, 0],
                               [s, c, 0],
                               [0, 0, 1]])
        else:
            axis = np.reshape(axis, (3, 1)) / np.linalg.norm(axis)
            rotmat = np.eye(3) * c + s * cross_mat(axis) + (1 - c) * (axis @ axis.T)
    except ValueError:
        axis = np.reshape(axis, (3, 1)) / np.linalg.norm(axis)
        rotmat = np.eye(3) * c + s * cross_mat(axis) + (1 - c) * (axis @ axis.T)
    if rotmat is None:
        raise ValueError('Invalid Axis {:}'.format(axis))
    return rotmat


def quaternion_to_rotation_matrix(quat):
    output = np.zeros(quat.shape[:-1] + (3, 3))
    q0 = quat[..., 0]
    q1 = quat[..., 1]
    q2 = quat[..., 2]
    q3 = quat[..., 3]

    # From https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    # First row of the rotation matrix
    output[..., 0, 0] = 2 * (q0 * q0 + q1 * q1) - 1
    output[..., 0, 1] = 2 * (q1 * q2 - q0 * q3)
    output[..., 0, 2] = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    output[..., 1, 0] = 2 * (q1 * q2 + q0 * q3)
    output[..., 1, 1] = 2 * (q0 * q0 + q2 * q2) - 1
    output[..., 1, 2] = 2 * (q2 * q3 - q0 * q1)
    # Third row of the rotation matrix
    output[..., 2, 0] = 2 * (q1 * q3 - q0 * q2)
    output[..., 2, 1] = 2 * (q2 * q3 + q0 * q1)
    output[..., 2, 2] = 2 * (q0 * q0 + q3 * q3) - 1

    return output


unit_magnitude_constraint = NonlinearConstraint(np.linalg.norm, 1, 1)


def lstsq_rigid_transform(x, y, w=None):
    """ Computes the Transform between two point sets such that
        y = Rx + t

        This is a least squares methods as descibed in
        "Least-Squares Rigid Motion Using SVD", O. Sorkine-Hornung, M.
        Rabinovich, Department of Computer Science, ETH Zurich, Jan 16, 2017
        Note found online at https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

        INPUT
            x,y = the two point sets as [x,y,z] column vectors
                  sizes are (3,n) for n points
            w = weighting vector for each point, wi>0
                default is uniform weighting of wi=1
                size is (1,n) for n points

        OUTPUT
            R,t  = [(3,3),(3,1)] transformation parameters such that
            y = Rx + t
    """
    if w is None:
        w = np.ones((x.shape[-1], 1)).T
        # default weighting is uniform w=1

    W = np.diag(w[0])

    # Find the weighted centroids (for w=1, this is the mean)
    xbar = np.sum(w * x, axis=-1) / np.sum(w, axis=1)
    ybar = np.sum(w * y, axis=-1) / np.sum(w, axis=1)

    # Center the points
    X = x - xbar[..., np.newaxis]
    Y = y - ybar[..., np.newaxis]

    # Calculate the Covariance
    Cov = X @ W @ np.moveaxis(Y, -1, -2)

    # Take the SVD of the Covariance matrix
    U, S, VH = np.linalg.svd(Cov)
    V = np.moveaxis(VH, -1, -2)  # numpy's SVD gives you back V', not V like Matlab
    det = np.linalg.det(V @ np.moveaxis(U, -1, -2))
    D = np.broadcast_to(np.eye(3), det.shape+(3, 3)).copy()
    D[..., -1, -1] = det

    R = V @ D @ np.moveaxis(U, -1, -2)
    t = ybar[..., np.newaxis] - R @ xbar[..., np.newaxis]

    return R, t


def cross_mat_vectorized(rvec):
    zero = np.zeros(rvec.shape[:-1])
    return np.moveaxis(np.moveaxis(
        np.array([
            [zero, -rvec[..., 2], rvec[..., 1]],
            [rvec[..., 2], zero, -rvec[..., 0]],
            [-rvec[..., 1], rvec[..., 0], zero]
        ]), 0, -1), 0, -1)


def rodrigues_to_matrix(rvec, threshold=0.000001):
    theta = np.linalg.norm(rvec, axis=-1, keepdims=True)
    rvec = rvec / theta
    R = (
        np.cos(theta[..., np.newaxis]) * np.eye(3)
        + (1 - np.cos(theta[..., np.newaxis])) * rvec[..., np.newaxis] @ rvec[..., np.newaxis, :]
        + np.sin(theta[..., np.newaxis]) * cross_mat_vectorized(rvec))
    R[theta[..., 0] < threshold] = np.eye(3)
    return R


def matrix_to_rodrigues(R, threshold=0.000001):
    A = (R - np.moveaxis(R, -2, -1)) / 2
    rho = np.moveaxis(np.array((A[..., 2, 1], A[..., 0, 2], A[..., 1, 0])), 0, -1)
    s = np.linalg.norm(rho, axis=-1)
    c = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    u = np.zeros(rho.shape)
    u[s >= threshold] = rho[s >= threshold] / s[s >= threshold][:, np.newaxis]
    u[(s < threshold) & (c > 1 - threshold)] = 0
    theta = np.arctan2(s, c)
    rvec = u * theta[..., np.newaxis]
    # Now handle the cases where s = 0 and c = -1
    special_cases = (s < threshold) & (c < -1 + threshold)
    for index in zip(*np.where(special_cases)):
        thisR = R[index]
        RpI = thisR + np.eye(3)
        best_column = np.linalg.norm(RpI, axis=0).argmax()
        v = RpI[:, best_column]
        thisu = v / np.linalg.norm(v)
        if ((abs(thisu[0]) < threshold and abs(thisu[1]) < threshold and thisu[2] < 0) or
            (abs(thisu[0] < threshold and thisu[1] < 0)) or
                (thisu[0] < 0)):
            rvec[index] = -thisu * np.pi
        else:
            rvec[index] = thisu * np.pi
    return rvec
