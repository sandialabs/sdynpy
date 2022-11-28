# -*- coding: utf-8 -*-
"""
Functions for integrating equations of motion in time.

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
import scipy.signal as signal


def MCK_to_StateSpace(M, C, K):

    ndofs = M.shape[0]

    # A = [[     0,     I],
    #      [M^-1*K,M^-1*C]]

    A_state = np.block([[np.zeros((ndofs, ndofs)), np.eye(ndofs)],
                        [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]])

    # B = [[     0,  M^-1]]

    B_state = np.block([[np.zeros((ndofs, ndofs))], [np.linalg.inv(M)]])

    # C = [[     I,     0],   # Displacements
    #      [     0,     I],   # Velocities
    #      [M^-1*K,M^-1*C],   # Accelerations
    #      [     0,     0]]   # Forces

    C_state = np.block([[np.eye(ndofs), np.zeros((ndofs, ndofs))],
                        [np.zeros((ndofs, ndofs)), np.eye(ndofs)],
                        [-np.linalg.solve(M, K), -np.linalg.solve(M, C)],
                        [np.zeros((ndofs, ndofs)), np.zeros((ndofs, ndofs))]])

    # D = [[     0],   # Displacements
    #      [     0],   # Velocities
    #      [  M^-1],   # Accelerations
    #      [     I]]   # Forces

    D_state = np.block([[np.zeros((ndofs, ndofs))],
                        [np.zeros((ndofs, ndofs))],
                        [np.linalg.inv(M)],
                        [np.eye(ndofs)]])

    return A_state, B_state, C_state, D_state


def MCK_incomplete_to_StateSpace(M, C, K, massless_dofs=[]):
    nomass = len(massless_dofs)
    mass = M.shape[0] - nomass
    mass_dofs = np.array([v for v in range(M.shape[0]) if not v in massless_dofs])
    massless_dofs = np.array(massless_dofs)

    Kmm = K[mass_dofs[:, np.newaxis], mass_dofs]
    Kmn = K[mass_dofs[:, np.newaxis], massless_dofs]
    Knm = K[massless_dofs[:, np.newaxis], mass_dofs]
    Knn = K[massless_dofs[:, np.newaxis], massless_dofs]

    Cmm = C[mass_dofs[:, np.newaxis], mass_dofs]
    Cnm = C[massless_dofs[:, np.newaxis], mass_dofs]
    Cnn = C[massless_dofs[:, np.newaxis], massless_dofs]

    Mmm = M[mass_dofs[:, np.newaxis], mass_dofs]

    A = np.block([
        [np.zeros((mass, mass)), np.eye(mass), np.zeros((mass, nomass))],
        [-np.linalg.solve(Mmm, Kmm), -np.linalg.solve(Mmm, Cmm), -np.linalg.solve(Mmm, Kmn)],
        [-np.linalg.solve(Cnn, Knm), -np.linalg.solve(Cnn, Cnm), -np.linalg.solve(Cnn, Knn)]
    ])

    B = np.block([
        [np.zeros((mass, mass)), np.zeros((mass, nomass))],
        [np.linalg.inv(Mmm), np.zeros((mass, nomass))],
        [np.zeros((nomass, mass)), np.linalg.inv(Cnn)]
    ])

    C = np.block([
        [np.eye(mass), np.zeros((mass, mass)), np.zeros((mass, nomass))],  # Massive displacements
        [np.zeros((mass, mass)), np.eye(mass), np.zeros((mass, nomass))],  # Massive velocities
        [-np.linalg.solve(Mmm, Kmm), -np.linalg.solve(Mmm, Cmm), - \
         np.linalg.solve(Mmm, Kmn)],  # Massive Accelerations
        [np.zeros((mass, mass)), np.zeros((mass, mass)),
         np.zeros((mass, nomass))],  # Massive Inputs
        [np.zeros((nomass, mass)), np.zeros((nomass, mass)),
         np.eye(nomass)],  # Nonmassive displacements
        [-np.linalg.solve(Cnn, Knm), -np.linalg.solve(Cnn, Cnm), - \
         np.linalg.solve(Cnn, Knn)],  # Nonmassive velocities
        [np.zeros((nomass, mass)), np.zeros((nomass, mass)),
         np.zeros((nomass, nomass))],  # Nonmassive inputs
    ])

    D = np.block([
        [np.zeros((mass, mass)), np.zeros((mass, nomass))],  # Massive displacements
        [np.zeros((mass, mass)), np.zeros((mass, nomass))],  # Massive velocities
        [np.linalg.inv(Mmm), np.zeros((mass, nomass))],  # Massive Accelerations
        [np.eye(mass), np.zeros((mass, nomass))],  # Massive Inputs
        [np.zeros((nomass, mass)), np.zeros((nomass, nomass))],  # Nonmassive displacements
        [np.zeros((nomass, mass)), np.linalg.inv(Cnn)],  # Nonmassive velocities
        [np.zeros((nomass, mass)), np.eye(nomass)],  # Nonmassive inputs
    ])

    doflist = {}
    doflist['state'] = [(dof, modifier) for modifier in ['', 'd']
                        for dof in mass_dofs] + [(dof, '') for dof in massless_dofs]
    doflist['input'] = [(dof, 'in') for dof in mass_dofs] + [(dof, 'in') for dof in massless_dofs]
    doflist['output'] = [(dof, modifier) for modifier in ['', 'd', 'dd', 'in'] for dof in mass_dofs] + [
        (dof, modifier) for modifier in ['', 'd', 'in'] for dof in massless_dofs]

    return A, B, C, D, doflist


def integrate_MCK(M, C, K, times, forces, x0=None):
    A, B, C, D = MCK_to_StateSpace(M, C, K)
    linear_system = signal.StateSpace(A, B, C, D)

    times_out, sys_out, x_out = signal.lsim(linear_system, forces, times, x0)

    sys_disps = sys_out[:, 0 * M.shape[0]:1 * M.shape[0]]
    sys_vels = sys_out[:, 1 * M.shape[0]:2 * M.shape[0]]
    sys_accels = sys_out[:, 2 * M.shape[0]:3 * M.shape[0]]
    sys_forces = sys_out[:, 3 * M.shape[0]:4 * M.shape[0]]

    return sys_disps, sys_vels, sys_accels, sys_forces


def modal_parameters_to_MCK(natural_frequencies, damping_ratios):
    nmodes = natural_frequencies.size
    M = np.eye(nmodes)
    K = np.diag((natural_frequencies * 2 * np.pi)**2)
    C = np.diag((2 * 2 * np.pi * natural_frequencies * damping_ratios))
    return M, C, K


def integrate_modes(natural_frequencies, damping_ratios, modal_forces, times, return_accel=False):
    M, C, K = modal_parameters_to_MCK(natural_frequencies, damping_ratios)
    q = integrate_MCK(M, C, K, times, modal_forces)
    if return_accel:
        return q[2]
    else:
        return q[0]


def frequency_domain_differentiation(abscissa, signal, order=1, axis=-1, fmin=-1, fmax=np.inf):
    dt = np.mean(np.diff(abscissa))
    n = signal.shape[axis]
    freqs = np.fft.fftfreq(n, dt)
    frequency_indices = np.logical_and(np.abs(freqs) > fmin, np.abs(freqs) < fmax)
    fft = np.fft.fft(signal, axis=axis)
    # Construct tuples for multidimensional analysis
    slice_array = tuple([frequency_indices if i == axis or (i - axis) ==
                        fft.ndim else np.newaxis for i in range(fft.ndim)])
    freq_slice_array = tuple([frequency_indices if i == axis or (
        i - axis) == fft.ndim else slice(None, None, None) for i in range(fft.ndim)])
    not_freq_slice_array = tuple([np.logical_not(frequency_indices) if i == axis or (
        i - axis) == fft.ndim else slice(None, None, None) for i in range(fft.ndim)])
    factor = (1j * 2 * np.pi * freqs[slice_array])**order
    # Perform integration/differentiation
    fft[freq_slice_array] *= factor
    fft[not_freq_slice_array] = 0 + 0j
    return np.real(np.fft.ifft(fft, axis=axis))
