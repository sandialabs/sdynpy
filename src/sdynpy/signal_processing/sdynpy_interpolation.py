# -*- coding: utf-8 -*-
"""
Functions for interpolating signals
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


import numpy as np


def adaptive_linear_interpolation(
    y,
    x=None,
    rtol=1e-3,
    atol=0.0,
    scale="range",
    max_points=None,
    return_reconstruction=False,
    relative_mode="global",
    local_floor_fraction=1e-3,
):
    """
    Adaptive piecewise-linear approximation with shared breakpoints across channels.

    Parameters
    ----------
    y : array_like, shape (C, N) or (N,)
        Signal values.
    x : array_like, shape (N,), optional
        Sample locations. If None, uses np.arange(N).
    rtol : float or array_like, shape (C,)
        Relative tolerance(s).
    atol : float or array_like, shape (C,)
        Absolute tolerance(s).
    scale : {"range", "maxabs", "rms"} or array_like, shape (C,)
        Global scale per channel.
    max_points : int, optional
        Maximum number of breakpoints including endpoints.
    return_reconstruction : bool, default False
        If True, return reconstructed y_hat.
    relative_mode : {"global", "local", "hybrid"}, default "global"
        How relative error is normalized.
    local_floor_fraction : float, default 1e-3
        Minimum local scale as a fraction of global channel scale.

    Returns
    -------
    breakpoints, x_bp, y_bp[, y_hat]
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    elif y.ndim != 2:
        raise ValueError("y must be 1D or 2D")

    C, N = y.shape
    if N < 2:
        raise ValueError("Need at least two samples")

    if x is None:
        x = np.arange(N, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        if x.shape != (N,):
            raise ValueError("x must have shape (N,)")
        if not np.all(np.diff(x) > 0):
            raise ValueError("x must be strictly increasing")

    rtol = np.asarray(rtol, dtype=float)
    if rtol.ndim == 0:
        rtol = np.full(C, rtol)
    elif rtol.shape != (C,):
        raise ValueError("rtol must be scalar or shape (C,)")

    atol = np.asarray(atol, dtype=float)
    if atol.ndim == 0:
        atol = np.full(C, atol)
    elif atol.shape != (C,):
        raise ValueError("atol must be scalar or shape (C,)")

    if isinstance(scale, str):
        if scale == "range":
            global_scale = y.max(axis=1) - y.min(axis=1)
        elif scale == "maxabs":
            global_scale = np.max(np.abs(y), axis=1)
        elif scale == "rms":
            global_scale = np.sqrt(np.mean(y**2, axis=1))
        else:
            raise ValueError("scale must be 'range', 'maxabs', 'rms', or an array")
    else:
        global_scale = np.asarray(scale, dtype=float)
        if global_scale.shape != (C,):
            raise ValueError("scale array must have shape (C,)")

    eps = np.finfo(float).eps
    global_scale = np.maximum(global_scale, eps)
    local_floor = local_floor_fraction * global_scale

    def reconstruct(bp):
        y_hat = np.empty_like(y)
        for k in range(len(bp) - 1):
            i0 = bp[k]
            i1 = bp[k + 1]
            x0 = x[i0]
            x1 = x[i1]
            alpha = (x[i0:i1+1] - x0) / (x1 - x0)
            y_hat[:, i0:i1+1] = (
                y[:, [i0]] + (y[:, [i1]] - y[:, [i0]]) * alpha[np.newaxis, :]
            )
        return y_hat

    def allowed_error(y_true, y_fit):
        if relative_mode == "global":
            rel_scale = global_scale[:, None]
        elif relative_mode == "local":
            rel_scale = np.maximum(np.maximum(np.abs(y_true), np.abs(y_fit)),
                                   local_floor[:, None])
        elif relative_mode == "hybrid":
            local_scale = np.maximum(np.maximum(np.abs(y_true), np.abs(y_fit)),
                                     local_floor[:, None])
            rel_scale = np.maximum(0.1 * global_scale[:, None], local_scale)
        else:
            raise ValueError("relative_mode must be 'global', 'local', or 'hybrid'")

        return atol[:, None] + rtol[:, None] * rel_scale

    bp = [0, N - 1]

    while True:
        bp = sorted(set(bp))
        y_hat = reconstruct(bp)

        abs_err = np.abs(y - y_hat)
        allowed = allowed_error(y, y_hat)
        violation = abs_err / allowed

        max_violation = np.max(violation)
        if max_violation <= 1.0:
            break

        score = np.max(violation, axis=0)
        score[np.array(bp)] = -np.inf

        idx = int(np.argmax(score))
        if not np.isfinite(score[idx]):
            break

        bp.append(idx)

        if max_points is not None and len(set(bp)) >= max_points:
            bp = sorted(set(bp))
            y_hat = reconstruct(bp)
            break

    bp = np.array(sorted(set(bp)), dtype=int)
    x_bp = x[bp]
    y_bp = y[:, bp]

    if return_reconstruction:
        y_hat = reconstruct(bp)
        return bp, x_bp, y_bp, y_hat

    return bp, x_bp, y_bp