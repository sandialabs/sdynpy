# -*- coding: utf-8 -*-
"""
Tools for computing inverses of frequency response functions

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
from scipy.interpolate import interp1d
import warnings


def frf_inverse(frf_matrix,
                method='standard',
                response_weighting_matrix=None,
                reference_weighting_matrix=None,
                regularization_weighting_matrix=None,
                regularization_parameter=None,
                cond_num_threshold=None,
                num_retained_values=None):
    """
    Computes the inverse of an FRF matrix for source estimation problems.

    Parameters
    ----------
    frf_matrix : NDArray
        Transfer function as an np.ndarray, should be organized such that the
        size is [number of lines, number of responses, number of references]
    method : str, optional
            The method to be used for the FRF matrix inversions. The available
            methods are:
                - standard - basic pseudo-inverse via numpy.linalg.pinv with the
                  default rcond parameter, this is the default method
                - threshold - pseudo-inverse via numpy.linalg.pinv with a specified
                  condition number threshold
                - tikhonov - pseudo-inverse using the Tikhonov regularization method
                - truncation - pseudo-inverse where a fixed number of singular values
                  are retained for the inverse
        response_weighting_matrix : sdpy.Matrix or np.ndarray, optional
            Diagonal matrix used to weight response degrees of freedom (to solve the
            problem as a weight least squares) by multiplying the rows of the FRF
            matrix by a scalar weights. This matrix can also be a 3D matrix such that
            the the weights are different for each frequency line. The matrix should
            be sized [number of lines, number of references, number of references],
            where the number of lines either be one (the same weights at all frequencies)
            or the length of the abscissa (for the case where a 3D matrix is supplied).
        reference_weighting_matrix : sdpy.Matrix or np.ndarray, optional
            Diagonal matrix used to weight reference degrees of freedom (generally for
            normalization) by multiplying the columns of the FRF matrix by a scalar weights.
            This matrix can also be a 3D matrix such that the the weights are different
            for each frequency line. The matrix should be sized
            [number of lines, number of references, number of references], where the number
            of lines either be one (the same weights at all frequencies) or the length
            of the abscissa (for the case where a 3D matrix is supplied).
        regularization_weighting_matrix : sdpy.Matrix or np.ndarray, optional
            Matrix used to weight input degrees of freedom via Tikhonov regularization.
            This matrix can also be a 3D matrix such that the the weights are different
            for each frequency line. The matrix should be sized
            [number of lines, number of references, number of references], where the number
            of lines either be one (the same weights at all frequencies) or the length
            of the abscissa (for the case where a 3D matrix is supplied).
        regularization_parameter : float or np.ndarray, optional
            Scaling parameter used on the regularization weighting matrix when the tikhonov
            method is chosen. A vector of regularization parameters can be provided so the
            regularization is different at each frequency line. The vector must match the
            length of the abscissa in this case (either be size [num_lines,] or [num_lines, 1]).
        cond_num_threshold : float or np.ndarray, optional
            Condition number used for SVD truncation when the threshold method is chosen.
            A vector of condition numbers can be provided so it varies as a function of
            frequency. The vector must match the length of the abscissa in this case.
        num_retained_values : float or np.ndarray, optional
            Number of singular values to retain in the pseudo-inverse when the truncation
            method is chosen. A vector of can be provided so the number of retained values
            can change as a function of frequency. The vector must match the length of the
            abscissa in this case.

    Returns
    -------
    np.ndarray
        Inverse of the supplied FRF matrix

    Raises
    ------
    Warning
        If regularization_weighting_matrix is supplied without selecting the tikhonov method
    Exception
        If the threshold method is chosen but a condition number threshold isn't supplied
    Exception
        If the declared method is not one of the available options

    Notes
    -----
    This function solves the inverse problem for the supplied FRF matrix. All of the inverse
    methods use the SVD (or modified SVD) to compute the pseudo-inverse.

    References
    ----------
    .. [1] Wikipedia, "Moore-Penrose inverse".
           https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
    .. [2] A.N. Tithe, D.J. Thompson, The quantification of structure-borne transmission pathsby inverse methods. Part 2: Use of regularization techniques,
           Journal of Sound and Vibration, Volume 264, Issue 2, 2003, Pages 433-451, ISSN 0022-460X,
           https://doi.org/10.1016/S0022-460X(02)01203-8.
    .. [3] Wikipedia, "Ridge regression".
           https://en.wikipedia.org/wiki/Ridge_regression
    """
    if regularization_weighting_matrix is not None and method != 'tikhonov':
        warnings.warn("Regularization weighting matrix is only used during Tikhonov regularization and is not being used")

    if response_weighting_matrix is not None:
        frf_matrix = response_weighting_matrix@frf_matrix
    if reference_weighting_matrix is not None:
        frf_matrix = frf_matrix@reference_weighting_matrix

    if method == 'standard':
        frf_pinv = np.linalg.pinv(frf_matrix)
    elif method == 'threshold':
        if cond_num_threshold is None:
            raise ValueError('A condition number threshold must be supplied when using the threshold method')
        rcond_thresh = 1/cond_num_threshold
        frf_pinv = np.linalg.pinv(frf_matrix, rcond=rcond_thresh)
    elif method == 'tikhonov':
        frf_pinv = pinv_by_tikhonov(frf_matrix,
                                    regularization_weighting_matrix=regularization_weighting_matrix,
                                    regularization_parameter=regularization_parameter)
    elif method == 'truncation':
        frf_pinv = pinv_by_truncation(frf_matrix,
                                      num_retained_values=num_retained_values)
    else:
        error_string_start = 'Declared method "'
        error_string_end = '" is not an available option'
        raise NameError(error_string_start+method+error_string_end)

    return frf_pinv


def pinv_by_tikhonov(frf_matrix,
                     regularization_weighting_matrix=None,
                     regularization_parameter=None):
    """
    Computes the pseudo-inverse of an FRF matrix via Tikhonov regularization

    Parameters
    ----------
    frf_matrix : NDArray
        Transfer function as an np.ndarray, should be organized such that the
        frequency is the first axis, the responses are the second axis, and the
        references are the third axis
    regularization_weighting_matrix : sdpy.Matrix or np.ndarray, optional
        Matrix used to weight input degrees of freedom in the regularization, the
        default is identity. This can be a 3D matrix such that the the weights are
        different for each frequency line. The matrix should be sized
        [number of lines, number of references, number of references], where the number
        of lines either be one (the same weights at all frequencies) or the length
        of the abscissa (for the case where a 3D matrix is supplied).
    regularization_parameter : float or np.ndarray, optional
        Scaling parameter used on the regularization weighting matrix. A vector of
        regularization parameters can be provided so the regularization is different at
        each frequency line. The vector must match the length of the abscissa in this
        case (either be size [num_lines,] or [num_lines, 1]).

    Returns
    -------
    np.ndarray
        Inverse of the supplied FRF matrix

    Raises
    ------
    Exception
        If a regularization parameter isn't supplied
    Warning
        If a 2D regularization weighting matrix is supplied when a vector of regularization parameters is supplied

    Notes
    -----
    Tikhonov regularization is being done via the SVD method (details on Wikipedia or
    in the paper by Tithe and Thompson).

    References
    ----------
    .. [1] A.N. Tithe, D.J. Thompson, The quantification of structure-borne transmission pathsby inverse methods. Part 2: Use of regularization techniques,
           Journal of Sound and Vibration, Volume 264, Issue 2, 2003, Pages 433-451, ISSN 0022-460X,
           https://doi.org/10.1016/S0022-460X(02)01203-8.
    .. [2] Wikipedia, "Ridge regression".
           https://en.wikipedia.org/wiki/Ridge_regression
    """

    if regularization_parameter is None:
        raise ValueError('A regularization parameter must be supplied when using Tikhonov regularization')

    regularization_parameter = np.asarray(regularization_parameter, dtype=np.float64)

    u, s, vh = np.linalg.svd(frf_matrix, full_matrices=False)
    vh_conjT = np.swapaxes(vh.conj(), -2, -1)
    u_conjT = np.swapaxes(u.conj(), -2, -1)

    # Need to add a second axis to the regularization parameter if it's a vector
    # so it broadcasts appropriately.
    if regularization_parameter.size > 1 and regularization_parameter.ndim == 1:
        regularization_parameter = regularization_parameter[..., np.newaxis]

    if regularization_parameter.size > 1 and regularization_parameter.shape[1] > 1:
        raise ValueError('The regularization parameter is not a scalar or vector and cannot be interpreted')

    if regularization_parameter.size > 1 and regularization_parameter.size != frf_matrix.shape[0]:
        raise ValueError('The regularization parameter is a vector but the size'
                         + ' does not match the number of frequency lines and cannot be broadcasted to the FRFs')

    # This is adding a third dimension to the regularization weighting matrix (if
    # it doesn't already exist) in the case that the regularization parameter is a
    # vector, this is required so the weighting matrix and parameter will broadcast
    # together appropriately.
    if regularization_weighting_matrix is not None and regularization_parameter.size > 1 and regularization_weighting_matrix.ndim == 2:
        warnings.warn('The regularization weighting matrix is 2D while the regularization'
                      + ' parameter is a vector, the regularization weighting matrix is '
                      + 'being expanded to 3D. Consider adding a third dimension to the regularization weighting matrix as appropriate')
        desired_shape = [regularization_parameter.shape[0], regularization_weighting_matrix.shape[0], regularization_weighting_matrix.shape[1]]
        regularization_weighting_matrix = np.broadcast_to(regularization_weighting_matrix, desired_shape)

    # This is adding a third dimension to the regularization parameter vector when a
    # regularization weighting matrix is being used. This is required so the broadcasting
    # works correctly. I.E., if the regularization weighting matrix is sized
    # [num_lines, num_refs, num_refs] (which it always is), the regularization parameter
    # vector has to be sized [num_lines, 1, 1]. This is not required when the regularization
    # parameter is a scaler.
    if (regularization_weighting_matrix is not None
            and regularization_weighting_matrix.ndim == 3
            and regularization_parameter.size > 1 and regularization_parameter.ndim != 3):
        regularization_parameter = regularization_parameter[..., np.newaxis]

    if regularization_weighting_matrix is None:
        regularized_sInv = s/(s*s+regularization_parameter)
        if frf_matrix.ndim == 3:
            # This turns s into a 3D matrix with the singular values on the diagonal
            regularized_sInv = np.apply_along_axis(np.diag, 1, regularized_sInv)
        else:
            # This is for the case where a single frequency line is supplied for the FRF
            regularized_sInv = np.diag(regularized_sInv)
    else:
        if frf_matrix.ndim == 3:
            # This turns s into a 3D matrix with the singular values on the diagonal
            s = np.apply_along_axis(np.diag, 1, s)
            regularized_sInv = np.linalg.pinv(np.moveaxis(s, -1, -2)@s+regularization_weighting_matrix*regularization_parameter)@np.moveaxis(s, -1, -2)
        else:
            # This is for the case where a single frequency line is supplied for the FRF
            regularized_sInv = s/(s*s+regularization_parameter*regularization_weighting_matrix)

    frf_pinv = vh_conjT@regularized_sInv@u_conjT

    return frf_pinv


def pinv_by_truncation(frf_matrix,
                       num_retained_values):
    """
    Computes the pseudo-inverse of an FRF matrix where a fixed number of singular
    values are retained for the inverse

    Parameters
    ----------
    frf_matrix : NDArray
        Transfer function as an np.ndarray, should be organized such that the
        frequency is the first axis, the responses are the second axis, and the
        references are the third axis
    num_retained_values : float or np.ndarray, optional
        Number of singular values to retain in the pseudo-inverse when the truncation
        method is chosen. A vector of can be provided so the number of retained values
        can change as a function of frequency. The vector must match the length of the
        abscissa in this case.

    Returns
    -------
    np.ndarray
        Inverse of the supplied FRF matrix

    Raises
    ------
    Exception
        If a number of retained values isn't supplied
    Warning
        if the number of retained values is greater than the actual number of singular
        values
    """
    if num_retained_values is None:
        raise ValueError('The number of retained values must be supplied when using the truncation method')

    num_retained_values = np.asarray(num_retained_values, dtype=np.intc)

    # Need to add a second axis to the number of retained values if it's a vector
    # so it broadcasts appropriately.
    if num_retained_values.size > 1 and num_retained_values.ndim == 1:
        num_retained_values = num_retained_values[..., np.newaxis]

    if num_retained_values.size > 1 and num_retained_values.shape[1] > 1:
        raise ValueError('The number of retained values parameter is not a scalar or vector and cannot be interpreted')

    if num_retained_values.size > 1 and num_retained_values.size != frf_matrix.shape[0]:
        raise ValueError('The number of retained values parameter is a vector but '
                         + 'the size does not match the number of frequency lines and cannot be broadcasted to the FRFs')

    u, s, vh = np.linalg.svd(frf_matrix, full_matrices=False)
    vh_conjT = np.swapaxes(vh.conj(), -2, -1)
    u_conjT = np.swapaxes(u.conj(), -2, -1)

    # The weird looking logic is to handle if num_retained_values is either a string
    # or a scaler. It also changes the number of retained values in the case that
    # it is greater than the actual number of singular values (it changes to retain
    # all the singular values).
    if num_retained_values.size > 1 and any(num_retained_values > s.shape[1]) or num_retained_values.size == 1 and num_retained_values > s.shape[1]:
        warnings.warn('The requested number of retained singular values is greater than the '
                      + 'actual number of singular values (for at least one frequency line). All singular values are being retained.')
        num_retained_values[num_retained_values > s.shape[1]] = s.shape[1]

    s_truncated_inv = np.zeros((s.shape[0], s.shape[1], s.shape[1]))
    num_lines = s.shape[0]

    # Making num_retained_values a vector if an integer was supplied
    if not num_retained_values.size > 1:
        num_retained_values = (np.ones(num_lines) * num_retained_values)[..., np.newaxis]

    num_retained_values = num_retained_values.astype(int)

    for ii in range(num_lines):
        if np.any(s[ii, :num_retained_values[ii, 0]] == 0):
            continue
        else:
            s_truncated_inv[ii, :num_retained_values[ii, 0], :num_retained_values[ii, 0]] = np.diag(1/s[ii, :num_retained_values[ii, 0]])

    frf_pinv = vh_conjT@s_truncated_inv@u_conjT

    return frf_pinv


def compute_tikhonov_modified_singular_values(original_singular_values, regularization_parameter):
    """
    Computes the modified singular values that would be seen in Tikhonov
    regularization. This is only intended for visualization purposes
    only.

    Parameters
    ----------
    original_singular_values : np.ndarray
        The singular values from the oringinal FRF matrix. Should be organized
        [frequency lines, singular values.]
    regularization_parameter : float or np.ndarray
        The regularization parameter that would be used in an inverse
        via Tikhonov regularization.

    Returns
    -------
    modified_singular_values : np.ndarray
        The modified singular values from the Tikhonov regularization.

    Notes
    -----
    This is not a mathematically rigorous version of the modified singular
    values and should only be used for subjective evaluation of the effect of
    the regularization parameter.
    """
    regularization_parameter = np.asarray(regularization_parameter, dtype=np.float64)

    if regularization_parameter.size != 1 and regularization_parameter.ndim == 1:
        regularization_parameter = regularization_parameter[..., np.newaxis]

    modified_singular_values = (original_singular_values**2 + regularization_parameter)/original_singular_values

    return modified_singular_values
