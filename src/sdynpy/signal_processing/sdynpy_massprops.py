# -*- coding: utf-8 -*-
"""
Functions for working with mass properties
"""
"""
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


def rigid_body_shape_scaling(
    rbm_modes,
    trans_shapes,
    rot_shapes,
    center_of_rotation=None,
    return_mass_properties=False,
    rcond=None,
    symmetrize=True,
    check_rank=True,
):
    """
    Recover scaled/mass-normalized rigid-body shapes and optionally mass properties
    from a set of mass-normalized rigid-body modes and user-supplied pure rigid-body
    translation/rotation shapes.

    Parameters
    ----------
    rbm_modes : (n, 6) array_like
        Mass-normalized rigid-body mode shapes. Columns span the 6 rigid-body modes.
        Assumed to satisfy Phi.T @ M @ Phi = I in the underlying mass metric.
    trans_shapes : (n, 3) array_like
        Unscaled pure translation shapes corresponding to unit translations in x, y, z
        (or equivalent unit physical translations in the chosen measurement basis).
    rot_shapes : (n, 3) array_like
        Unscaled pure rotation shapes corresponding to unit rotations about x, y, z
        (1 radian), defined about some reference point.
    center_of_rotation : (3,) array_like, optional
        Reference point about which rot_shapes were constructed. If provided, the
        center of mass is also returned in global coordinates. If omitted, COM is
        returned only relative to the rotation reference point.
    return_mass_properties : bool, optional
        If True, also return recovered mass properties.
    rcond : float, optional
        Cutoff for pseudoinverse.
    symmetrize : bool, optional
        If True, symmetrize the recovered rigid-body mass matrix and inertia tensors.
    check_rank : bool, optional
        If True, check that the supplied rigid-body basis has rank 6.

    Returns
    -------
    scaled_shapes : (n, 6) ndarray
        Mass-normalized/scaled physical rigid-body shapes corresponding as closely
        as possible to the supplied pure translation/rotation basis.
        Columns correspond to [Tx, Ty, Tz, Rx, Ry, Rz].
    mass_props : dict, optional
        Returned only if return_mass_properties=True. Contains:
            - 'A' : (6, 6) transformation matrix such that B ≈ Phi @ A
            - 'G' : (6, 6) rigid-body generalized mass matrix in basis B
            - 'mass' : scalar total mass
            - 'com_relative' : (3,) center of mass relative to rotation reference point
            - 'com_global' : (3,) center of mass in global coordinates (if center_of_rotation provided)
            - 'inertia_about_reference' : (3, 3) inertia tensor about the rotation reference point
            - 'inertia_about_com' : (3, 3) inertia tensor about center of mass
            - 'principal_inertias' : (3,) eigenvalues of inertia_about_com
            - 'principal_axes' : (3, 3) eigenvectors of inertia_about_com
            - 'reconstruction_error' : relative Frobenius norm of B - Phi @ A
            - 'G_tt', 'G_tr', 'G_rt', 'G_rr' : 3x3 blocks of G

    Notes
    -----
    Let Phi = rbm_modes and B = [trans_shapes rot_shapes].
    Since both span the rigid-body subspace, B = Phi A for some A.
    Using mass normalization Phi.T M Phi = I, the rigid-body generalized mass matrix is

        G = B.T M B = A.T A

    The mass properties are extracted from:

        G = [[ m I,      -m [c]_x ],
             [ m [c]_x,   J_ref   ]]

    where c is COM relative to the reference point used to define the rotation shapes,
    and J_ref is the inertia tensor about that same reference point.

    Sign convention for c assumes the rotation shapes follow:
        u = omega x (x - O)
    """
    Phi = np.asarray(rbm_modes, dtype=float)
    T = np.asarray(trans_shapes, dtype=float)
    R = np.asarray(rot_shapes, dtype=float)

    if Phi.ndim != 2 or Phi.shape[1] != 6:
        raise ValueError("rbm_modes must have shape (n, 6)")
    if T.ndim != 2 or T.shape[1] != 3:
        raise ValueError("trans_shapes must have shape (n, 3)")
    if R.ndim != 2 or R.shape[1] != 3:
        raise ValueError("rot_shapes must have shape (n, 3)")
    if T.shape[0] != Phi.shape[0] or R.shape[0] != Phi.shape[0]:
        raise ValueError("rbm_modes, trans_shapes, and rot_shapes must have the same number of rows")

    B = np.hstack((T, R))

    if check_rank:
        rank_phi = np.linalg.matrix_rank(Phi)
        rank_B = np.linalg.matrix_rank(B)
        if rank_phi < 6:
            raise ValueError(f"rbm_modes has rank {rank_phi}, expected 6")
        if rank_B < 6:
            raise ValueError(f"[trans_shapes rot_shapes] has rank {rank_B}, expected 6")

    # Solve B ≈ Phi @ A
    Phi_pinv = np.linalg.pinv(Phi, rcond=rcond)
    A = Phi_pinv @ B

    # "Scaled" physical rigid-body shapes in the mass-normalized basis:
    # columns are the unit physical RB shapes normalized by their generalized masses.
    #
    # Since G = A.T @ A, one valid whitening transform is inv(cholesky(G)).
    # Then B_scaled = B @ inv(cholesky(G)) gives shapes whose mass metric is identity.
    G = A.T @ A
    if symmetrize:
        G = 0.5 * (G + G.T)

    # Cholesky can fail if numerically near-singular; use eig fallback
    try:
        L = np.linalg.cholesky(G)
        G_inv_sqrt = np.linalg.inv(L).T  # because G = L L^T, so G^{-1/2} can be taken as L^{-T}
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(G)
        tol = np.max(eigvals) * 1e-12 if np.max(eigvals) > 0 else 1e-12
        if np.any(eigvals <= tol):
            raise ValueError("Recovered generalized mass matrix is not positive definite enough to invert.")
        G_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    scaled_shapes = B @ G_inv_sqrt

    if not return_mass_properties:
        return scaled_shapes

    # Diagnostics
    recon = Phi @ A
    reconstruction_error = np.linalg.norm(B - recon, ord="fro") / max(np.linalg.norm(B, ord="fro"), 1e-16)

    # Partition G
    G_tt = G[:3, :3]
    G_tr = G[:3, 3:]
    G_rt = G[3:, :3]
    G_rr = G[3:, 3:]

    if symmetrize:
        G_tt = 0.5 * (G_tt + G_tt.T)
        G_rr = 0.5 * (G_rr + G_rr.T)
        # Ideally G_rt = G_tr.T, but keep both in case diagnostics are useful

    # Total mass
    mass = np.trace(G_tt) / 3.0

    # Recover COM relative to reference point from G_tr = -m [c]_x
    # For [c]_x = [[0, -cz, cy], [cz, 0, -cx], [-cy, cx, 0]]
    #
    # Then:
    # G_tr = -m[c]_x = [[0, m*cz, -m*cy],
    #                   [-m*cz, 0, m*cx],
    #                   [m*cy, -m*cx, 0]]
    #
    # Use antisymmetric average for robustness.
    G_tr_skew = 0.5 * (G_tr - G_rt.T)

    cx = G_tr_skew[1, 2] / mass
    cy = G_tr_skew[2, 0] / mass
    cz = G_tr_skew[0, 1] / mass
    com_relative = np.array([cx, cy, cz])

    inertia_about_reference = G_rr.copy()
    if symmetrize:
        inertia_about_reference = 0.5 * (inertia_about_reference + inertia_about_reference.T)

    # Parallel-axis theorem: J_com = J_ref - m (||c||^2 I - c c^T)
    c = com_relative
    c2 = np.dot(c, c)
    inertia_about_com = inertia_about_reference - mass * (c2 * np.eye(3) - np.outer(c, c))
    if symmetrize:
        inertia_about_com = 0.5 * (inertia_about_com + inertia_about_com.T)

    principal_inertias, principal_axes = np.linalg.eigh(inertia_about_com)

    mass_props = {
        "A": A,
        "G": G,
        "mass": mass,
        "com_relative": com_relative,
        "inertia_about_reference": inertia_about_reference,
        "inertia_about_com": inertia_about_com,
        "principal_inertias": principal_inertias,
        "principal_axes": principal_axes,
        "reconstruction_error": reconstruction_error,
        "G_tt": G_tt,
        "G_tr": G_tr,
        "G_rt": G_rt,
        "G_rr": G_rr,
    }

    if center_of_rotation is not None:
        O = np.asarray(center_of_rotation, dtype=float).reshape(3)
        mass_props["center_of_rotation"] = O
        mass_props["com_global"] = O + com_relative

    return scaled_shapes, mass_props