# -*- coding: utf-8 -*-
"""
Functions for computing shapes after the poles are known

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

from enum import Enum
import numpy as np
from ..core.sdynpy_data import TransferFunctionArray
from ..core.sdynpy_shape import shape_array
from ..core.sdynpy_coordinate import CoordinateArray
from ..signal_processing.sdynpy_complex import collapse_complex_to_real
import warnings


class ShapeSelection(Enum):
    ALL = 0
    DRIVE_POINT_COEFFICIENT = 1
    PARTICIPATION_FACTOR = 2


def compute_residues(experimental_frf: TransferFunctionArray,
                     natural_frequencies: np.ndarray,
                     damping_ratios: np.ndarray,
                     real_modes: bool = False,
                     residuals: bool = True,
                     min_frequency: float = None,
                     max_frequency: float = None,
                     weighting: np.ndarray = 'uniform',
                     displacement_derivative: int = 0,
                     frequency_lines_at_resonance: int = None,
                     frequency_lines_for_residuals: int = None):
    """
    Fit residues to FRF data given frequency and damping values

    Parameters
    ----------
    experimental_frf : TransferFunctionArray
        Experimental FRF data to which modes will be fit
    natural_frequencies : np.ndarray
        Natural Frequencies (in Hz) at which modes will be fit
    damping_ratios : np.ndarray
        Damping Ratios at which modes will be fit
    real_modes : bool, optional
        If true, fit residues will be real-valued.  False allows complex modes.
        The default is False.
    residuals : bool, optional
        Use residuals in the FRF fit. The default is True.
    min_frequency : float, optional
        Minimum frequency to use in the shape fit. The default is the lowest
        frequency in the experimental FRF.
    max_frequency : float, optional
        Maximum frequency to use in the shape fit. The default is the highest
        frequency in the experimental FRF.
    weighting : np.ndarray or string, optional
        A weighting array to use to fit shapes better at specific frequencies.
        The default is weighted by the log magnitude of the FRF matrix.  Can be
        defined as 'magnitude','uniform', or an ndarray with shape identical
        to the ordinate of the experimental frfs
    displacement_derivative : int, optional
        Defines the type of data in the FRF based on the number of derivatives
        from displacement (0 - displacement, 1 - velocity, 2 - acceleration).
        The default is 0 (displacement).
    frequency_lines_at_resonance : int, optional
        Defines the number of frequency lines to look at around the specified
        natural frequencies for computing residues.  If not specified, all
        frequency lines are used for computing shapes.
    frequency_lines_for_residuals : int, optional
        Defines the number of frequency lines at the low and high frequency to
        use in computing shapes.  Only used if frequency_lines_at_resonance is
        specified.  If not specified, the lower 10% and upper 10% of frequency
        lines will be kept.

    Returns
    -------
    shape_residues : np.ndarray
        A (..., n_modes) shaped np.ndarray where ... is the shape of the input
        experimental_frf array.  There will be one residue for each experimental
        frf (reference and response) for each mode.
    synthesized_frf : TransferFunctionArray
        Transfer function array containing the analytical fits using the
        residues.
    residual_frf : TransferFunctionArray
        Transfer function array containing the residual data from the analytical
        fits.
    """
    flat_frf = experimental_frf.flatten()
    frequencies = flat_frf[0].abscissa.copy()
    if min_frequency is None:
        min_frequency = np.min(frequencies)
    if max_frequency is None:
        max_frequency = np.max(frequencies)
    abscissa_indices = np.ones(frequencies.shape, dtype=bool)
    abscissa_indices &= (frequencies >= min_frequency)
    abscissa_indices &= (frequencies <= max_frequency)
    frequencies = frequencies[abscissa_indices]
    frf_matrix = flat_frf.ordinate[:, abscissa_indices].T.copy()
    angular_frequencies = 2 * np.pi * frequencies[:, np.newaxis]
    angular_natural_frequencies = 2 * np.pi * np.array(natural_frequencies).flatten()
    damping_ratios = np.array(damping_ratios).flatten()

    # Reduce to the kept frequency lines
    if frequency_lines_at_resonance is not None:
        solve_indices = np.argmin(np.abs(angular_natural_frequencies - angular_frequencies), axis=0)
        # print(solve_indices)
        solve_indices = np.unique(
            solve_indices[:, np.newaxis] + np.arange(frequency_lines_at_resonance) - frequency_lines_at_resonance // 2)
        solve_indices = solve_indices[(solve_indices >= 0) & (
            solve_indices < angular_frequencies.size)]
        # Add the residual indices
        if residuals:
            if frequency_lines_for_residuals is None:
                low_freq_indices = np.arange(angular_frequencies.size // 10)
                high_freq_indices = angular_frequencies.size - \
                    np.arange(angular_frequencies.size // 10) - 1
            else:
                low_freq_indices = np.arange(frequency_lines_for_residuals)
                high_freq_indices = angular_frequencies.size - \
                    np.arange(frequency_lines_for_residuals) - 1
            solve_indices = np.unique(np.concatenate(
                (solve_indices, low_freq_indices, high_freq_indices)))
        kernel_indices = np.concatenate((solve_indices, solve_indices + angular_frequencies.size))
        # print(kernel_indices)

    # Set up the kernel to solve the least squares residue problem
    denominator = ((angular_natural_frequencies**2 - angular_frequencies**2)**2
                   + (2 * damping_ratios * angular_natural_frequencies * angular_frequencies)**2)
    kernel_rr = (angular_natural_frequencies**2 - angular_frequencies**2) / denominator
    kernel_ri = 2 * damping_ratios * angular_natural_frequencies * angular_frequencies / denominator
    kernel_ir = -2 * damping_ratios * angular_natural_frequencies * angular_frequencies / denominator
    kernel_ii = (angular_natural_frequencies**2 - angular_frequencies**2) / denominator
    low_frequency_residual = (-1 / angular_frequencies**2)
    high_frequency_residual = (np.ones(angular_frequencies.shape))
    zeros = np.zeros(angular_frequencies.shape)

    kernel = np.concatenate((np.concatenate((kernel_rr, kernel_ri, low_frequency_residual, zeros, high_frequency_residual, zeros), axis=-1),
                             np.concatenate((kernel_ir, kernel_ii, zeros, low_frequency_residual, zeros, high_frequency_residual), axis=-1)), axis=0)

    # print(kernel.shape)

    # Reduce kernel depending on whether or not we want real modes or complex
    # modes
    if real_modes:
        kernel[..., angular_natural_frequencies.size:2 * angular_natural_frequencies.size] = 0

    # Reduce kernel depending on whether or not we want residuals
    if not residuals:
        kernel[..., -4:] = 0

    # Perform the solution in acceleration space
    kernel *= np.tile(-angular_frequencies**2, (2, 1))
    frf_matrix *= (1j * angular_frequencies)**(2 - displacement_derivative)

    # print(kernel.shape)

    # Weighting matrix
    if isinstance(weighting, str):
        if weighting.lower() == 'uniform':
            weighting = np.ones(frf_matrix.shape)
        elif weighting.lower() == 'magnitude':
            max_frf = np.max(np.abs(frf_matrix), axis=0, keepdims=True)
            max_frf[max_frf == 0.0] = 1.0
            weighting = np.abs(frf_matrix) / max_frf
            min_weighting = np.min(weighting, axis=0, keepdims=True)
            min_weighting[min_weighting == 0] = 1
            weighting_ratio = weighting / min_weighting
            weighting_ratio[weighting_ratio == 0] = 1
            weighting = np.log10(weighting_ratio)
        else:
            raise ValueError(
                'If weighting is specified as a string, must be "uniform" or "magnitude"')

    # Now assemble the FRF matrix
    frf_matrix_to_fit = np.concatenate((frf_matrix.real,
                                        frf_matrix.imag), axis=0)

    # Now perform the weighting
    weighting = np.tile(weighting, (2, 1))
    weighted_kernel = weighting.T[..., np.newaxis] * kernel
    weighted_frf_matrix_to_fit = (weighting * frf_matrix_to_fit).T[..., np.newaxis]

    # print(weighted_kernel.shape)
    # print(weighted_frf_matrix_to_fit.shape)

    if frequency_lines_at_resonance is not None:
        weighted_kernel = weighted_kernel[:, kernel_indices]
        weighted_frf_matrix_to_fit = weighted_frf_matrix_to_fit[:, kernel_indices]

    # now solve
    residues = (np.linalg.pinv(weighted_kernel) @ weighted_frf_matrix_to_fit).squeeze().T

    frf_fit_matrix = kernel @ residues
    frf_fit_matrix = (frf_fit_matrix[:frf_fit_matrix.shape[0] // 2, :]
                      + 1j * frf_fit_matrix[frf_fit_matrix.shape[0] // 2:, :])
    output_frf = flat_frf.extract_elements(abscissa_indices)
    output_frf.ordinate = (frf_fit_matrix / (1j * angular_frequencies)
                           ** (2 - displacement_derivative)).T
    output_frf = output_frf.reshape(experimental_frf.shape)

    # Extract shape residues and residuals
    shape_residues = (residues[:angular_natural_frequencies.size]
                      + 1j * residues[angular_natural_frequencies.size:2 * angular_natural_frequencies.size])
    shape_residues = np.moveaxis(shape_residues.reshape(-1, *experimental_frf.shape), 0, -1)
    if real_modes:
        shape_residues = np.real(shape_residues)

    # Extract residuals
    residual_frf = flat_frf.extract_elements(abscissa_indices)
    residual_matrix = kernel[:, -4:] @ residues[-4:]
    residual_matrix = (residual_matrix[:residual_matrix.shape[0] // 2, :]
                       + 1j * residual_matrix[residual_matrix.shape[0] // 2:, :])
    residual_frf.ordinate = (residual_matrix / (1j * angular_frequencies)
                             ** (2 - displacement_derivative)).T
    residual_frf = residual_frf.reshape(experimental_frf.shape)

    return shape_residues, output_frf, residual_frf


def compute_shapes(natural_frequencies: np.ndarray,
                   damping_ratios: np.ndarray,
                   coordinates: CoordinateArray,
                   residue_matrix: np.ndarray,
                   shape_selection=ShapeSelection.ALL,
                   participation_factors: np.ndarray = None):
    abs_coordinates = abs(coordinates)
    sign_matrix = np.prod(coordinates.sign(), axis=-1)
    equality_matrix = np.all(abs_coordinates == abs_coordinates[..., 0, np.newaxis], axis=-1)
    drive_point_indices = np.argmax(equality_matrix, axis=0)
    residue_scales = np.array([sign_matrix[response_index, reference_index]
                               for reference_index, response_index in enumerate(drive_point_indices)])
    signed_residue_matrix = residue_matrix * residue_scales[:, np.newaxis]
    drive_point_residues = np.array([signed_residue_matrix[response_index, reference_index, :]
                                     for reference_index, response_index in enumerate(drive_point_indices)])
    negative_drive_points = np.where(drive_point_residues < 0)
    if np.isrealobj(residue_matrix) and np.any(drive_point_residues < 0):
        print('Negative Drive Point Residues Found!')
        for mode_index, reference_index in np.sort(np.array(negative_drive_points).T[..., ::-1], axis=0):
            print('  Mode {:} Reference {:}'.format(
                mode_index + 1, str(coordinates[0, reference_index, -1])))
    mode_shape_scaling = np.sqrt(np.abs(drive_point_residues) if np.isrealobj(
        residue_matrix) else drive_point_residues)
    mode_shape_matrix = signed_residue_matrix / mode_shape_scaling
    if isinstance(shape_selection, str):
        if shape_selection.lower() in ['all']:
            shape_selection = ShapeSelection.ALL
        elif shape_selection.lower() in ['drive', 'drive point', 'dp']:
            shape_selection = ShapeSelection.DRIVE_POINT_COEFFICIENT
        elif shape_selection.lower() in ['part', 'participation', 'participation factor']:
            shape_selection = ShapeSelection.PARTICIPATION_FACTOR
    if not shape_selection == ShapeSelection.ALL:
        if shape_selection == ShapeSelection.DRIVE_POINT_COEFFICIENT:
            shape_selection_indices = np.argmax(drive_point_residues, axis=0)
        elif shape_selection == ShapeSelection.PARTICIPATION_FACTOR:
            shape_selection_indices = np.argmax(np.abs(participation_factors), axis=-1)
        else:
            raise ValueError('Invalid Shape Selection Technique')
        mode_shape_matrix = np.array([mode_shape_matrix[:, reference_index, mode_index]
                                      for mode_index, reference_index in enumerate(shape_selection_indices)]).T
    else:
        shape_selection_indices = np.arange(mode_shape_matrix.shape[1])[
            :, np.newaxis] * np.ones(mode_shape_matrix.shape[2], dtype=int)
    ref_array = np.ndarray
    shapes = shape_array(coordinate=coordinates[..., 0, 0],
                         shape_matrix=np.moveaxis(mode_shape_matrix, 0, -1),
                         frequency=natural_frequencies, damping=damping_ratios,
                         comment1=coordinates[0, :, -1].string_array()[shape_selection_indices])
    return shapes, negative_drive_points


def compute_shapes_multireference(experimental_frf: TransferFunctionArray,
                                  natural_frequencies: np.ndarray,
                                  damping_ratios: np.ndarray,
                                  participation_factors: np.ndarray,
                                  real_modes: bool = False,
                                  lower_residuals: bool = True,
                                  upper_residuals: bool = True,
                                  min_frequency: float = None,
                                  max_frequency: float = None,
                                  displacement_derivative: int = 0,
                                  frequency_lines_at_resonance: int = None,
                                  frequency_lines_for_residuals: int = None):
    """
    Computes mode shapes from multireference datasets.

    Uses the modal participation factor as a constraint on the mode shapes to
    solve for the shapes in one pass, rather than solving for residues and
    subsequently solving for shapes.

    Parameters
    ----------
    experimental_frf : TransferFunctionArray
        Experimental FRF data to which modes will be fit
    natural_frequencies : np.ndarray
        Natural Frequencies (in Hz) at which modes will be fit
    damping_ratios : np.ndarray
        Damping Ratios at which modes will be fit
    participation_factors : np.ndarray
        Mode participation factors from which the shapes can be computed.
        Should have shape (n_modes x n_inputs)
    real_modes : bool, optional
        Specifies whether to solve for real modes or complex modes (default).
    lower_residuals : bool, optional
        Use lower residuals in the FRF fit. The default is True.
    upper_residuals : bool, optional
        Use upper residuals in the FRF fit. The default is True.
    min_frequency : float, optional
        Minimum frequency to use in the shape fit. The default is the lowest
        frequency in the experimental FRF.
    max_frequency : float, optional
        Maximum frequency to use in the shape fit. The default is the highest
        frequency in the experimental FRF.
    displacement_derivative : int, optional
        Defines the type of data in the FRF based on the number of derivatives
        from displacement (0 - displacement, 1 - velocity, 2 - acceleration).
        The default is 0 (displacement).
    frequency_lines_at_resonance : int, optional
        Defines the number of frequency lines to look at around the specified
        natural frequencies for computing residues.  If not specified, all
        frequency lines are used for computing shapes.
    frequency_lines_for_residuals : int, optional
        Defines the number of frequency lines at the low and high frequency to
        use in computing shapes.  Only used if frequency_lines_at_resonance is
        specified.  If not specified, the lower 10% and upper 10% of frequency
        lines will be kept for computing residuals.

    Raises
    ------
    ValueError
        If the FRF is not 2-dimensional with references on the columns and
        responses on the rows.

    Returns
    -------
    output_shape : ShapeArray
        ShapeArray containing the mode shapes of the system
    frfs_resynthesized : TransferFunctionArray
        FRFs resynthesized from the fit shapes and residuals
    residual_frfs : TransferFunctionArray
        FRFs resynthesized only from the residuals used in the calculation
    kernel_frfs: TransferFunctionArray
        FRFs synthesized by the kernel solution without taking into account
        construction of mode shapes and reciprocity with participation factors
    """
    original_coordinates = experimental_frf.coordinate
    if not experimental_frf.ndim == 2:
        raise ValueError('FRF must be shaped n_outputs x n_inputs')
    abs_coordinate = abs(original_coordinates)
    experimental_frf = experimental_frf[abs_coordinate]
    # Also need to adjust the participation factors
    participation_factors = participation_factors*np.sign(original_coordinates[0, :, 1].direction)
    if real_modes:
        participation_factors = collapse_complex_to_real(participation_factors)
    frequencies = experimental_frf[0, 0].abscissa.copy()
    if min_frequency is None:
        min_frequency = np.min(frequencies)
    if max_frequency is None:
        max_frequency = np.max(frequencies)
    abscissa_indices = np.ones(frequencies.shape, dtype=bool)
    abscissa_indices &= (frequencies >= min_frequency)
    abscissa_indices &= (frequencies <= max_frequency)
    frequencies = frequencies[abscissa_indices]
    frf_matrix = experimental_frf.ordinate[..., abscissa_indices].copy()
    angular_frequencies = 2 * np.pi * frequencies
    reconstruction_angular_frequencies = frequencies*2*np.pi
    angular_natural_frequencies = 2 * np.pi * np.array(natural_frequencies).flatten()
    damping_ratios = np.array(damping_ratios).flatten()

    # Reduce to the kept frequency lines
    if frequency_lines_at_resonance is not None:
        solve_indices = np.argmin(np.abs(angular_natural_frequencies - angular_frequencies[:, np.newaxis]), axis=0)
        # print(solve_indices)
        solve_indices = np.unique(
            solve_indices[:, np.newaxis] + np.arange(frequency_lines_at_resonance) - frequency_lines_at_resonance // 2)
        solve_indices = solve_indices[(solve_indices >= 0) & (
            solve_indices < angular_frequencies.size)]
        # Add the residual indices
        if lower_residuals:
            if frequency_lines_for_residuals is None:
                low_freq_indices = np.arange(angular_frequencies.size // 10)
            else:
                low_freq_indices = np.arange(frequency_lines_for_residuals)
        else:
            low_freq_indices = []
        if upper_residuals:
            if frequency_lines_for_residuals is None:
                high_freq_indices = angular_frequencies.size - \
                    np.arange(angular_frequencies.size // 10) - 1
            else:
                high_freq_indices = angular_frequencies.size - \
                    np.arange(frequency_lines_for_residuals) - 1
        else:
            high_freq_indices = []
        solve_indices = np.unique(np.concatenate(
            (solve_indices, low_freq_indices, high_freq_indices)))
        frf_matrix = frf_matrix[..., solve_indices]
        angular_frequencies = angular_frequencies[solve_indices]

    poles = -damping_ratios*angular_natural_frequencies + 1j*np.sqrt(1-damping_ratios**2)*angular_natural_frequencies

    if real_modes:
        kernel = generate_kernel_real(
            angular_frequencies, poles,
            participation_factors, lower_residuals, 
            upper_residuals, displacement_derivative)
        if angular_frequencies.size != reconstruction_angular_frequencies.size:
            reconstruction_kernel = generate_kernel_real(
                reconstruction_angular_frequencies, poles,
                participation_factors, lower_residuals, 
                upper_residuals, displacement_derivative)
        else:
            reconstruction_kernel = kernel
    else:
        kernel = generate_kernel_complex(
            angular_frequencies, poles, participation_factors, lower_residuals,
            upper_residuals, displacement_derivative)
        if angular_frequencies.size != reconstruction_angular_frequencies.size:
            reconstruction_kernel = generate_kernel_complex(
                reconstruction_angular_frequencies, poles, participation_factors, lower_residuals, 
                upper_residuals, displacement_derivative)
        else:
            reconstruction_kernel = kernel

    coefficients, (full_reconstruction, residual_reconstruction) = stack_and_lstsq(
        frf_matrix, kernel,
        return_reconstruction=True,
        reconstruction_partitions = [slice(None),
                                     slice(poles.size*(1 if real_modes else 2),None)],
        reconstruction_kernel = reconstruction_kernel)

    residual_frfs = experimental_frf.copy().extract_elements(abscissa_indices)
    residual_frfs.ordinate = residual_reconstruction
    kernel_frfs = experimental_frf.copy().extract_elements(abscissa_indices)
    kernel_frfs.ordinate = full_reconstruction

    # Extract the shapes and residues
    if real_modes:
        shapes = (coefficients[:poles.size]).T
    else:
        shapes = (coefficients[:poles.size] + 1j*coefficients[poles.size:2*poles.size]).T

    # Now we have to go in and find the scale factor to scale the shapes correctly
    drive_points = np.where(experimental_frf.response_coordinate == experimental_frf.reference_coordinate)
    if drive_points[0].size == 0:
        print('Warning: Drive Points Not Found in Dataset, Shapes are Unscaled.')
        output_shape = shape_array(abs_coordinate[:, 0, 0], shapes.T,
                                   angular_natural_frequencies/(2*np.pi),
                                   damping_ratios,
                                   1)
        frfs_resynthesized = kernel_frfs
    else:
        shapes_normalized = []
        drive_residues = shapes[drive_points[0]].T*participation_factors[:,drive_points[1]]
        for k, drive_residue in enumerate(drive_residues):
            if real_modes:
                # Throw away negative drive points
                bad_indices = drive_residue < 0
            else:
                # Throw away non-negative imaginary parts
                bad_indices_positive = drive_residue.imag > 0
                # Throw away small values compared to the average value (only considering negative imaginary parts)
                bad_indices_small = np.abs(drive_residue) < 0.01*np.mean(np.abs(drive_residue[~bad_indices_positive]))
                # Combine into a single criteria
                bad_indices = bad_indices_positive | bad_indices_small
            # Get the good indices that are remaining
            remaining_indices = np.where(~bad_indices)[0]
            if len(remaining_indices) == 0:
                print('Warning: Mode {:} had no valid drive points and is therefore unscaled.'.format(k+1))
                shapes_normalized.append(shapes[:,k])
                continue
            # We will then construct the least squares solution
            shape_coefficients = shapes[drive_points[0][remaining_indices],k][:,np.newaxis]
            residue_coefficients = np.sqrt(drive_residue[remaining_indices])[:,np.newaxis]
            # Before we compute the scale, we need to make sure that we have all of the signs the same way.
            # This is because the square root can give you +/- root where root**2 = complex number
            # This will mess up the least squares at it will try to find something between the
            # two vectors.
            scale_vector = (residue_coefficients/shape_coefficients).flatten()
            sign_vector = np.array((scale_vector.real,scale_vector.imag))
            # Get the signs
            signs = np.sign(np.dot(sign_vector[:,0],sign_vector))
            residue_coefficients = residue_coefficients*signs[:,np.newaxis]
            # Now compute the least-squares solution
            scale = np.linalg.lstsq(shape_coefficients,residue_coefficients)[0].squeeze()
            print('Scale for mode {:}: {:}'.format(k+1,scale))
            shapes_normalized.append(shapes[:,k]*scale)
        shapes_normalized = np.array(shapes_normalized).T

        output_shape = shape_array(abs_coordinate[:, 0, 0], shapes_normalized.T,
                                   angular_natural_frequencies/(2*np.pi),
                                   damping_ratios,
                                   1)

        frfs_resynthesized = (output_shape.compute_frf(
            frequencies,
            responses=experimental_frf.response_coordinate[:, 0],
            references=experimental_frf.reference_coordinate[0, :],
            displacement_derivative=displacement_derivative)
            + residual_frfs)

    frfs_resynthesized = frfs_resynthesized[original_coordinates]
    residual_frfs = residual_frfs[original_coordinates]
    kernel_frfs = kernel_frfs[original_coordinates]

    return output_shape, frfs_resynthesized, residual_frfs, kernel_frfs

def generate_kernel_complex(omegas, poles, participation_factors,
                            lower_residuals = False, upper_residuals = False,
                            displacement_derivative = 0):
    """
    

    Parameters
    ----------
    omegas : np.ndarray
        The angular frequencies (in radians/s) at which the kernel matrix will
        be computed.  This should be a 1D array with length num_freqs.
    poles : np.ndarray
        An array of poles corresponding to the modes of the structure.  This
        should be a 1D array with length num_modes.
    participation_factors : np.ndarray
        A 2D array of participation factors corresponding to the reference
        degrees of freedom.  This should have shape (num_modes, num_inputs).
    lower_residuals : bool, optional
        If True, construct the kernel matrix such that lower residuals will be
        computed in the least-squares operation.  The default is False.
    upper_residuals : bool, optional
        If True, construct the kernel matrix such that upper residuals will be
        computed in the least-squares operation.  The default is False.
    displacement_derivative : int, optional
        The derivative of displacement used to construct the frequency response
        functions.  Should be 0 for a receptance (displacement/force) frf, 1
        for a mobility (velocity/force) frf, or 2 for an accelerance
        (acceleration/force) frf.

    Returns
    -------
    kernel_matrix : np.ndarray
        A 3D matrix that represents the kernel that can be inverted to solve
        for mode shapes.  The size of the output array will be num_inputs x
        num_freq*2 x num_modes*2.  The top partition of the num_freq dimension
        corresponds to the real part of the frf, and the
        bottom portion corresponds to the imaginary part of the frf.  The top
        partition of the num_modes dimension corresponds to the real part of
        the mode shape matrix, and the bottom partition corresponds to the
        imaginary part.  If residuals are included, there will be an extra two
        entries along the num_modes dimension corresponding to real and
        imaginary parts of the residual matrix for each of the residuals
        included.
    """
    omegas = np.array(omegas)
    poles = np.array(poles)
    participation_factors = np.array(participation_factors)
    if omegas.ndim != 1:
        raise ValueError('`omegas` should be 1-dimensional')
    if poles.ndim != 1:
        raise ValueError('`poles` should be 1-dimensional')
    if participation_factors.ndim != 2:
        raise ValueError('`participation_factors` should be 2-dimensional (num_modes x num_inputs)')
    n_modes,n_inputs = participation_factors.shape
    if poles.size != n_modes:
        raise ValueError('number of poles ({:}) is not equal to the number of rows in the participation_factors array ({:})'.format(
            poles.size,n_modes))
    # Set up broadcasting
    # We want the output array to be n_input x n_freq*2 x n_modes*2
    # So let's adjust the terms so they have the right shapes
    # We want frequency lines to be the middle dimension
    omega = omegas[np.newaxis,:,np.newaxis]
    # We want inputs to be the first dimension and modes the
    # last dimension
    participation_factors = participation_factors.T[:,np.newaxis,:]
    # Split up terms into real and imaginary parts
    p_r = poles.real
    p_i = poles.imag
    l_r = participation_factors.real
    l_i = participation_factors.imag
    
    # Now go through the different derivatives and compute the kernel plus any
    # residuals
    if displacement_derivative == 0:
        kernel = np.array([
            [(-l_i*omega - l_i*p_i - l_r*p_r)/(omega**2 + 2*omega*p_i + p_i**2 + p_r**2) 
              + (l_i*omega - l_i*p_i - l_r*p_r)/(omega**2 - 2*omega*p_i + p_i**2 + p_r**2),
             (l_i*p_r - l_r*omega - l_r*p_i)/(omega**2 + 2*omega*p_i + p_i**2 + p_r**2)
              + (l_i*p_r + l_r*omega - l_r*p_i)/(omega**2 - 2*omega*p_i + p_i**2 + p_r**2)],
            [(-l_i*p_r - l_r*omega + l_r*p_i)/(omega**2 - 2*omega*p_i + p_i**2 + p_r**2)
              + (l_i*p_r - l_r*omega - l_r*p_i)/(omega**2 + 2*omega*p_i + p_i**2 + p_r**2),
             (l_i*omega - l_i*p_i - l_r*p_r)/(omega**2 - 2*omega*p_i + p_i**2 + p_r**2) 
              + (l_i*omega + l_i*p_i + l_r*p_r)/(omega**2 + 2*omega*p_i + p_i**2 + p_r**2)]])
        if lower_residuals:
            kernel_RL = np.concatenate([
                np.kron(-1/omegas[:,np.newaxis]**2,np.kron(np.eye(n_inputs)[:,np.newaxis],[1,0])),
                np.kron(-1/omegas[:,np.newaxis]**2,np.kron(np.eye(n_inputs)[:,np.newaxis],[0,1]))],axis=1)
        if upper_residuals:
            kernel_RU = np.concatenate([
                np.kron(np.ones((omegas.size,1)),np.kron(np.eye(n_inputs)[:,np.newaxis],[1,0])),
                np.kron(np.ones((omegas.size,1)),np.kron(np.eye(n_inputs)[:,np.newaxis],[0,1]))],axis=1)
    elif displacement_derivative == 1:
        kernel = np.array([
            [(-l_i*omega*p_r + l_r*omega**2 + l_r*omega*p_i)/(omega**2 + 2*omega*p_i + p_i**2 + p_r**2)
              + (l_i*omega*p_r + l_r*omega**2 - l_r*omega*p_i)/(omega**2 - 2*omega*p_i + p_i**2 + p_r**2),
             (-l_i*omega**2 - l_i*omega*p_i - l_r*omega*p_r)/(omega**2 + 2*omega*p_i + p_i**2 + p_r**2)
              + (-l_i*omega**2 + l_i*omega*p_i + l_r*omega*p_r)/(omega**2 - 2*omega*p_i + p_i**2 + p_r**2)],
            [(-l_i*omega**2 - l_i*omega*p_i - l_r*omega*p_r)/(omega**2 + 2*omega*p_i + p_i**2 + p_r**2)
              + (l_i*omega**2 - l_i*omega*p_i - l_r*omega*p_r)/(omega**2 - 2*omega*p_i + p_i**2 + p_r**2),
             (l_i*omega*p_r - l_r*omega**2 - l_r*omega*p_i)/(omega**2 + 2*omega*p_i + p_i**2 + p_r**2)
              + (l_i*omega*p_r + l_r*omega**2 - l_r*omega*p_i)/(omega**2 - 2*omega*p_i + p_i**2 + p_r**2)]])
        if lower_residuals:
            kernel_RL = np.concatenate([
                np.kron( 1/omegas[:,np.newaxis],np.kron(np.eye(n_inputs)[:,np.newaxis],[0,1])),
                np.kron(-1/omegas[:,np.newaxis],np.kron(np.eye(n_inputs)[:,np.newaxis],[1,0]))],axis=1)
        if upper_residuals:
            kernel_RU = np.concatenate([
                np.kron(-omegas[:,np.newaxis],np.kron(np.eye(n_inputs)[:,np.newaxis],[0,1])),
                np.kron( omegas[:,np.newaxis],np.kron(np.eye(n_inputs)[:,np.newaxis],[1,0]))],axis=1)
    elif displacement_derivative == 2:
        kernel = np.array([
            [(-l_i*omega**3 + l_i*omega**2*p_i + l_r*omega**2*p_r)/(omega**2 - 2*omega*p_i + p_i**2 + p_r**2)
              + (l_i*omega**3 + l_i*omega**2*p_i + l_r*omega**2*p_r)/(omega**2 + 2*omega*p_i + p_i**2 + p_r**2),
             (-l_i*omega**2*p_r - l_r*omega**3 + l_r*omega**2*p_i)/(omega**2 - 2*omega*p_i + p_i**2 + p_r**2)
              + (-l_i*omega**2*p_r + l_r*omega**3 + l_r*omega**2*p_i)/(omega**2 + 2*omega*p_i + p_i**2 + p_r**2)],
            [(-l_i*omega**2*p_r + l_r*omega**3 + l_r*omega**2*p_i)/(omega**2 + 2*omega*p_i + p_i**2 + p_r**2)
              + (l_i*omega**2*p_r + l_r*omega**3 - l_r*omega**2*p_i)/(omega**2 - 2*omega*p_i + p_i**2 + p_r**2),
             (-l_i*omega**3 - l_i*omega**2*p_i - l_r*omega**2*p_r)/(omega**2 + 2*omega*p_i + p_i**2 + p_r**2)
              + (-l_i*omega**3 + l_i*omega**2*p_i + l_r*omega**2*p_r)/(omega**2 - 2*omega*p_i + p_i**2 + p_r**2)]])
        if lower_residuals:
            kernel_RL = np.concatenate([
                np.kron(np.ones((omegas.size,1)),np.kron(np.eye(n_inputs)[:,np.newaxis],[1,0])),
                np.kron(np.ones((omegas.size,1)),np.kron(np.eye(n_inputs)[:,np.newaxis],[0,1]))],axis=1)
        if upper_residuals:
            kernel_RU = np.concatenate([
                np.kron(-omegas[:,np.newaxis]**2,np.kron(np.eye(n_inputs)[:,np.newaxis],[1,0])),
                np.kron(-omegas[:,np.newaxis]**2,np.kron(np.eye(n_inputs)[:,np.newaxis],[0,1]))],axis=1)
    kernel = np.block([[kernel[0,0],kernel[0,1]],
                       [kernel[1,0],kernel[1,1]]])
    if lower_residuals:
        kernel = np.concatenate((kernel,kernel_RL),axis=-1)
    if upper_residuals:
        kernel = np.concatenate((kernel,kernel_RU),axis=-1)
    
    return kernel

def generate_kernel_real(omegas, poles, participation_factors,
                         lower_residuals = False, upper_residuals = False,
                         displacement_derivative = 0):
    """
    

    Parameters
    ----------
    omegas : np.ndarray
        The angular frequencies (in radians/s) at which the kernel matrix will
        be computed.  This should be a 1D array with length num_freqs.
    poles : np.ndarray
        An array of poles corresponding to the modes of the structure.  This
        should be a 1D array with length num_modes.
    participation_factors : np.ndarray
        A 2D array of participation factors corresponding to the reference
        degrees of freedom.  This should have shape (num_modes, num_inputs).
    lower_residuals : bool, optional
        If True, construct the kernel matrix such that lower residuals will be
        computed in the least-squares operation.  The default is False.
    upper_residuals : bool, optional
        If True, construct the kernel matrix such that upper residuals will be
        computed in the least-squares operation.  The default is False.
    displacement_derivative : int, optional
        The derivative of displacement used to construct the frequency response
        functions.  Should be 0 for a receptance (displacement/force) frf, 1
        for a mobility (velocity/force) frf, or 2 for an accelerance
        (acceleration/force) frf.

    Returns
    -------
    kernel_matrix : np.ndarray
        A 3D matrix that represents the kernel that can be inverted to solve
        for mode shapes.  The size of the output array will be num_inputs x
        num_freq*2 x num_modes.  The top partition of the num_freq dimension
        corresponds to the real part of the frf, and the
        bottom portion corresponds to the imaginary part of the frf.
        If residuals are included, there will be an extra two
        entries along the num_modes dimension corresponding to real and
        imaginary parts of the residual matrix for each of the residuals
        included.
    """
    omegas = np.array(omegas)
    poles = np.array(poles)
    participation_factors = np.array(participation_factors)
    if omegas.ndim != 1:
        raise ValueError('`omegas` should be 1-dimensional')
    if poles.ndim != 1:
        raise ValueError('`poles` should be 1-dimensional')
    if participation_factors.ndim != 2:
        raise ValueError('`participation_factors` should be 2-dimensional (num_modes x num_inputs)')
    n_modes,n_inputs = participation_factors.shape
    if poles.size != n_modes:
        raise ValueError('number of poles ({:}) is not equal to the number of rows in the participation_factors array ({:})'.format(
            poles.size,n_modes))
    # Set up broadcasting
    # We want the output array to be n_input x n_freq*2 x n_modes*2
    # So let's adjust the terms so they have the right shapes
    # We want frequency lines to be the middle dimension
    omega = omegas[np.newaxis,:,np.newaxis]
    # We want inputs to be the first dimension and modes the
    # last dimension
    l_r = participation_factors.T[:,np.newaxis,:]
    # Split up terms into real and imaginary parts
    omega_r = np.abs(poles)
    zeta_r = -np.real(poles)/omega_r
    
    # Now go through the different derivatives and compute the kernel plus any
    # residuals
    if displacement_derivative == 0:
        kernel = np.array([[(-l_r*omega**2 + l_r*omega_r**2)/
                             (omega**4 + 4*omega**2*omega_r**2*zeta_r**2 - 2*omega**2*omega_r**2 + omega_r**4)],
                           [-2*l_r*omega*omega_r*zeta_r/
                             (omega**4 + 4*omega**2*omega_r**2*zeta_r**2 - 2*omega**2*omega_r**2 + omega_r**4)]])
        if lower_residuals:
            kernel_RL = np.concatenate([
                np.kron(-1/omegas[:,np.newaxis]**2,np.kron(np.eye(n_inputs)[:,np.newaxis],[1,0])),
                np.kron(-1/omegas[:,np.newaxis]**2,np.kron(np.eye(n_inputs)[:,np.newaxis],[0,1]))],axis=1)
        if upper_residuals:
            kernel_RU = np.concatenate([
                np.kron(np.ones((omegas.size,1)),np.kron(np.eye(n_inputs)[:,np.newaxis],[1,0])),
                np.kron(np.ones((omegas.size,1)),np.kron(np.eye(n_inputs)[:,np.newaxis],[0,1]))],axis=1)
    elif displacement_derivative == 1:
        kernel = np.array([[2*l_r*omega**2*omega_r*zeta_r/
                             (omega**4 + 4*omega**2*omega_r**2*zeta_r**2 - 2*omega**2*omega_r**2 + omega_r**4)],
                           [(-l_r*omega**3 + l_r*omega*omega_r**2)/
                             (omega**4 + 4*omega**2*omega_r**2*zeta_r**2 - 2*omega**2*omega_r**2 + omega_r**4)]])
        if lower_residuals:
            kernel_RL = np.concatenate([
                np.kron( 1/omegas[:,np.newaxis],np.kron(np.eye(n_inputs)[:,np.newaxis],[0,1])),
                np.kron(-1/omegas[:,np.newaxis],np.kron(np.eye(n_inputs)[:,np.newaxis],[1,0]))],axis=1)
        if upper_residuals:
            kernel_RU = np.concatenate([
                np.kron(-omegas[:,np.newaxis],np.kron(np.eye(n_inputs)[:,np.newaxis],[0,1])),
                np.kron( omegas[:,np.newaxis],np.kron(np.eye(n_inputs)[:,np.newaxis],[1,0]))],axis=1)
    elif displacement_derivative == 2:
        kernel = np.array([[(l_r*omega**4 - l_r*omega**2*omega_r**2)/
                             (omega**4 + 4*omega**2*omega_r**2*zeta_r**2 - 2*omega**2*omega_r**2 + omega_r**4)],
                           [2*l_r*omega**3*omega_r*zeta_r/
                             (omega**4 + 4*omega**2*omega_r**2*zeta_r**2 - 2*omega**2*omega_r**2 + omega_r**4)]])
        if lower_residuals:
            kernel_RL = np.concatenate([
                np.kron(np.ones((omegas.size,1)),np.kron(np.eye(n_inputs)[:,np.newaxis],[1,0])),
                np.kron(np.ones((omegas.size,1)),np.kron(np.eye(n_inputs)[:,np.newaxis],[0,1]))],axis=1)
        if upper_residuals:
            kernel_RU = np.concatenate([
                np.kron(-omegas[:,np.newaxis]**2,np.kron(np.eye(n_inputs)[:,np.newaxis],[1,0])),
                np.kron(-omegas[:,np.newaxis]**2,np.kron(np.eye(n_inputs)[:,np.newaxis],[0,1]))],axis=1)
    kernel = np.block([[kernel[0,0]],
                       [kernel[1,0]]])
    if lower_residuals:
        kernel = np.concatenate((kernel,kernel_RL),axis=-1)
    if upper_residuals:
        kernel = np.concatenate((kernel,kernel_RU),axis=-1)
    
    return kernel

def stack_and_lstsq(frf_matrix, kernel_matrix, return_reconstruction = False,
                    reconstruction_partitions = None,
                    reconstruction_kernel = None):
    """
    Stacks the frf and kernel matrices appropriately,
    then solves the least-squares problem

    Parameters
    ----------
    frf_matrix : np.ndarray
        A num_outputs x num_inputs x num_frequencies array representing the
        frequency response function matrix.
    kernel_matrix : np.ndarray
        A num_inputs x 2*num_freq x num_coefficients kernel matrix from generated
        from generate_kernel_complex or generate_kernel_real functions.
    return_reconstruction : bool
        If True, a reconstruction of the frequency response function matrix
        using the kernel and the computed shape coefficients.  Default is False.
    reconstruction_paritions : list
        If specified, a list of reconstructions will be returned.  For each
        entry in this list, a frf matrix will be reconstructed using the entry
        as indices into the coefficient matrix's rows and the kernel matrix's
        columns.  This allows, for example, reconstruction of the frf matrix
        using only the residual terms or only the shape terms.  If not
        specified when `return_reconstruction` is True, only one frf matrix will
        be returned using all shape coefficients.
    reconstruction_kernel : np.ndarray
        A separate kernel matrix used for reconstruction.  If not specified, the
        kernel matrix used to solve the least-squares problem will be used for
        the reconstruction.

    Returns
    -------
    coefficients : np.ndarray
        A num_coefficients x num_outputs array of solution values to the
        least-squares problem.
    reconstruction : np.ndarray or list of np.ndarray
        A frf matrix reconstructed from the kernel and the shape coefficients.
        If multiple `reconstruction_partitions` are specified in a list, then
        this return value will also be a list of reconstructions corresponding
        to those partitions.  This will only be returned if
        `return_reconstruction` is True.
        

    """
    H = frf_matrix.transpose(1,2,0) # input, freq, output
    H = np.concatenate((H.real,H.imag),axis=1) # Stack real/imag along the freq axis
    if H.shape[:2] != kernel_matrix.shape[:2]:
        raise ValueError('`frf_matrix` {:} does not have consistent dimensions with the kernel matrix {:}'.format(
            H.shape,kernel_matrix.shape))
    H = H.reshape(-1,H.shape[-1]) # Stack input and freq along the same dimension
    kernel = kernel_matrix.reshape(-1,kernel_matrix.shape[-1]) # Stack input and freq
    coefficients = np.linalg.lstsq(kernel,H)[0]
    if not return_reconstruction:
        return coefficients
    if reconstruction_kernel is None:
        reconstruction_kernel = kernel_matrix
    if reconstruction_partitions is None:
        H_reconstructed = reconstruction_kernel@coefficients
        frf_matrix_reconstructed = H_reconstructed.transpose(2,0,1) # output, input, freq
        frf_matrix_reconstructed = (frf_matrix_reconstructed[...,:frf_matrix_reconstructed.shape[-1]//2] + 
                                    1j*frf_matrix_reconstructed[...,frf_matrix_reconstructed.shape[-1]//2:])
    else:
        frf_matrix_reconstructed = []
        for partition in reconstruction_partitions:
            recon = (reconstruction_kernel[...,partition]@coefficients[partition]).transpose(2,0,1) # output, input, freq
            recon = recon[...,:recon.shape[-1]//2] + 1j*recon[...,recon.shape[-1]//2:]
            frf_matrix_reconstructed.append(recon)
    return coefficients, frf_matrix_reconstructed
