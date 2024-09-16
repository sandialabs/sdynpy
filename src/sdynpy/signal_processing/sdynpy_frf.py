# -*- coding: utf-8 -*-
"""
Tools for computing frequency response functions

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
import scipy.signal as sig
import matplotlib.pyplot as plt
from .sdynpy_lrm import frf_local_model


def sysmat2frf(frequencies, M, C, K, frf_type='disp'):
    '''Compute Frequency Response Functions given system matrices M, C, and K

    This function computes frequency response functions (FRFs) given the system
    mass matrix, M, damping matrix, C, and stiffness matrix, K, in the equation
    of motion

    M x_dd + C x_d + K x = F

    This will return the frequency response function

    H = x/F

    as a function of frequency.

    Parameters
    ----------
    frequencies : ndarray
        frequencies should be a 1D np.ndarray consisting of the frequency lines
        at which to evaluate the frequency response function.  They should be
        in units of cycles per second or hertz (Hz), rather than in angular
        frequency of radians/s.
    M : ndarray
        M should be a 2D np.ndarray consisting of the system mass matrix.
    C : ndarray
        C should be a 2D np.ndarray consisting of the system damping matrix.
    K : ndarray
        K should be a 2D np.ndarray consisting of the system stiffness matrix.
    frf_type : str
        frf_type should be one of ['disp','vel','accel'] or ['displacement',
        'velocity','acceleration'] to specify which "type" of frequency
        response function to compute.  By default it computes a displacement or
        "receptance" FRF.  However, if an acceleration or "Accelerance" FRF is
        desired, specify 'accel' instead.  The displacement, velocity, and
        acceleration FRFs differ by a factor of 1j*omega where omega is the
        angular frequency at a given frequency line.

    Returns
    -------
    H : ndarray
        A 3D np array with shape (nf,no,ni), where nf is the number of
        frequency lines, no is the number of outputs, and ni is the number of
        inputs.  Since M, C, and K should be square, ni should equal no.
        Values in H are complex.

    Notes
    -----
    This performs a direct inversion of the system matrix, therefore it is not
    advisable to compute FRFs of large systems using this method.  An
    alternative would be to compute modes first then compute the FRFs from
    the modes.
    '''
    if not frf_type in ['displacement', 'velocity', 'acceleration', 'disp', 'vel',
                        'accel']:
        raise ValueError('frf_type must be one of {:}'.format(['displacement',
                         'velocity', 'acceleration', 'disp', 'vel', 'accel']))
    Z = (-(2 * np.pi * frequencies[:, np.newaxis, np.newaxis])**2 * M
         + 1j * (2 * np.pi * frequencies[:, np.newaxis, np.newaxis]) * C
         + K)
    H = np.linalg.inv(Z)
    if frf_type in ['vel', 'velocity']:
        H = 1j * (2 * np.pi * frequencies[:, np.newaxis, np.newaxis]) * H
    elif frf_type in ['accel', 'acceleration']:
        H = -(2 * np.pi * frequencies[:, np.newaxis, np.newaxis])**2 * H

    return H


def modes2frf(frequencies, natural_frequencies, damping_ratios, mode_shapes=None,
              input_mode_shapes=None, frf_type='disp'):
    '''Compute Frequency Response Functions given modal properties

    This function computes frequency responses given modal parameters.

    Parameters
    ----------
    frequencies : ndarray
        frequencies should be a 1D np.ndarray consisting of the frequency lines
        at which to evaluate the frequency response function.  They should be
        in units of cycles per second or hertz (Hz), rather than in angular
        frequency of radians/s.
    natural_frequencies : ndarray
        Natural frequencies of the structure in cycles per second or hertz (Hz)
        rather than in angular frequency of radians/s.
    damping_ratios : ndarray
        Critical damping ratios of the structure in ratio form rather than
        percentange, e.g. 2% damping would be specified as 0.02 rather than 2.
    mode_shapes : ndarray, optional
        A 2D mode shape matrix with shape (no,nm) where no is the number of
        output responses and nm is the number of modes.  If the optional
        argument input_mode_shapes is not specified, this mode shape matrix
        will also be used for the inputs, resulting in a square FRF matrix.
        If mode_shapes is not specified, the values of the modal FRF matrix
        are returned as 2D array.
    input_mode_shapes : ndarray, optional
        A 2D mode shape matrix with shape (ni,nm) where ni is the number of
        input forces and nm is the number of modes.  If the optional argument 
        input_mode_shapes is specified, it be used for the inputs, resulting in
        a potentially nonsquare FRF matrix.
    frf_type : str, optional
        frf_type should be one of ['disp','vel','accel'] or ['displacement',
        'velocity','acceleration'] to specify which "type" of frequency
        response function to compute.  By default it computes a displacement or
        "receptance" FRF.  However, if an acceleration or "Accelerance" FRF is
        desired, specify 'accel' instead.  The displacement, velocity, and
        acceleration FRFs differ by a factor of 1j*omega where omega is the
        angular frequency at a given frequency line.

    Returns
    -------
    H : ndarray
        A 3D (or 2D) np array depending on whether or not mode_shapes is 
        defined with shape (nf,no,ni) or (nf,nm), where nf is the number of
        frequency lines, no is the number of outputs, ni is the number of
        inputs, and nm is the number of modes.  Values in H are complex.

    Notes
    -----
    This function assumes mass normalized mode shapes.

    '''
    if not frf_type in ['displacement', 'velocity', 'acceleration', 'disp', 'vel',
                        'accel']:
        raise ValueError('frf_type must be one of {:}'.format(['displacement',
                         'velocity', 'acceleration', 'disp', 'vel', 'accel']))

    Z = (-(2 * np.pi * frequencies[:, np.newaxis])**2
         + 1j * (2 * np.pi * frequencies[:, np.newaxis]) * 2 *
         damping_ratios * (2 * np.pi * natural_frequencies)
         + (2 * np.pi * natural_frequencies)**2)
    if mode_shapes is not None:
        if input_mode_shapes is None:
            input_mode_shapes = mode_shapes
        H = np.einsum('ij,fj,lj->fil', mode_shapes, 1 / Z, input_mode_shapes)
    else:
        H = 1/Z

    if frf_type in ['vel', 'velocity']:
        if np.ndim(H) == 3:
            H = 1j * (2 * np.pi * frequencies[:, np.newaxis, np.newaxis]) * H
        elif np.ndim(H) == 2:
            H = 1j * (2 * np.pi * frequencies[:, np.newaxis]) * H
        # num = [1,0]
    elif frf_type in ['accel', 'acceleration']:
        if np.ndim(H) == 3:
            H = -(2 * np.pi * frequencies[:, np.newaxis, np.newaxis])**2 * H
        elif np.ndim(H) == 2:
            H = -(2 * np.pi * frequencies[:, np.newaxis])**2 * H
    return H

def timedata2frf(references, responses, dt=1, samples_per_average=None,
                 overlap=0.0, method='H1', window=np.array((1.0,)),
                 response_fft=lambda array: np.fft.rfft(array, axis=-1),
                 reference_fft=lambda array: np.fft.rfft(array, axis=-1),
                 response_fft_array=None, reference_fft_array=None,
                 return_model_data=False, **lrm_kwargs):
    # TODO Update DOCSTRING with changes after they are all made.
    '''Creates an FRF matrix given time histories of responses and references

    This function creates a nf x no x ni FRF matrix from the time histories
    provided.

    Parameters
    ----------
    references : ndarray
        A ni x nt or nt array where ni is the number of references and nt is
        the number of time steps in the signal.  If averaging is specified, nt
        should be divisible by the number of averages.
    responses : ndarray
        A no x nt or nt array where no is the number of responses and nt is the
        number of time steps in the signal.  If averaging is specified, nt
        should be divisible by the number of averages.
    dt : float
        The time between samples
    samples_per_average : int
        The number of time samples per average.  If not specified, it is set
        to the number of samples in the time signal, and no averaging is 
        performed
    overlap : float
        The overlap as a fraction of the frame (e.g. 0.5 specifies 50% overlap).
        If not specified, no overlap is used.
    method : str in ['H1','H2','Hv','Hs','LRM']
        The method for creating the frequency response function. 'H1' is 
        default if not specified.
    window : ndarray or str
        A 1D ndarray with length samples_per_average that specifies the
        coefficients of the window.  No window is applied if not specified.
        If a string is specified, then the window will be obtained from scipy
    fft : function
        FFT Function that should be used.  FFT must take the fft over axis -1.
    response_fft_array : np.ndarray
        Array to store the data into before taking the FFT.  Should be
        size number_of_responses, n_averages, samples_per_average
    reference_fft_array : np.ndarray
        Array to store the data into before taking the FFT.  Should be
        size number_of_references, n_averages, samples_per_average
    return_model_data : boolean
        Wheter to return selected model orders used in the 'LRM' method.
        default is False        
    **lrm_kwargs
        Additional keyword arguments specify parameters if method == 'LRM'.
        Possible arguments are: f_out (ndarray), bandwidth (float), transient
        (boolean), modelset (iterable of (3,1) ndarray), export_ratio (float), 
        max_parallel (int), print_time (bool). See sdynpy_lrm.frf_local_model.
        samples_per_average, overlap, and window are void if method=='LRM'.

    Returns
    -------
    frequencies : ndarray
        A nf array of the frequency values associated with H
    H : ndarray
        A nf x no x ni array where nf is the number of frequency lines, no is
        the number of outputs, and ni is the number of inputs.
    model_data: dict
        Contains None if method != 'LRM'. See sdynpy_lrm.frf_local_model.
        
    Notes
    -----
    There are requirements for the shapes of references and responses for some
    FRF computations.

    '''
    if references.shape[-1] != responses.shape[-1]:
        raise ValueError(
            'reference and responses must have the same number of time steps (last dimension)!')
    if not method in ['H1', 'H2', 'H3', 'Hs', 'Hv', 'Hcd','LRM']:
        raise ValueError('method parameter must be one of ["H1","H2","H3","Hv","Hcd"]')
    if references.ndim == 1:
        references = references[np.newaxis, :]
    if responses.ndim == 1:
        responses = responses[np.newaxis, :]
    # Set up time indices
    ntimes_total = references.shape[-1]
    if samples_per_average is None or method == 'LRM':
        samples_per_average = ntimes_total
    overlap_samples = int(samples_per_average * overlap)
    time_starts = np.arange(0, ntimes_total - samples_per_average +
                            1, samples_per_average - overlap_samples)
    time_indices_for_averages = time_starts[:, np.newaxis] + np.arange(samples_per_average)
    if method == 'LRM':
        window=np.array((1.0,)) # local modeling does not use windowing
    if isinstance(window, str):
        window = sig.windows.get_window(window.lower(), samples_per_average, fftbins=True)
    # Sort the references and responses into their time arrays
    if response_fft_array is None:
        response_fft_array = responses[..., time_indices_for_averages] * window
    else:
        response_fft_array[...] = responses[:, time_indices_for_averages] * window
    if reference_fft_array is None:
        reference_fft_array = references[:, time_indices_for_averages] * window
    else:
        reference_fft_array[...] = references[:, time_indices_for_averages] * window
    # Compute FFTs
    frequencies = np.fft.rfftfreq(samples_per_average, dt)
    response_ffts = response_fft(response_fft_array)
    reference_ffts = reference_fft(reference_fft_array)
    # Now start computing FRFs
    if method == 'H1':
        # We want to compute X*F^H = [X1;X2;X3][F1^H F2^H F3^H]
        Gxf = np.einsum('...iaf,...jaf->...fij', response_ffts, np.conj(reference_ffts))
        Gff = np.einsum('...iaf,...jaf->...fij', reference_ffts, np.conj(reference_ffts))
        # Add small values to any matrices that are singular
        singular_matrices = np.abs(np.linalg.det(Gff)) < 2 * np.finfo(Gff.dtype).eps
        Gff[singular_matrices] += np.eye(Gff.shape[-1]) * np.finfo(Gff.dtype).eps
        H = np.moveaxis(np.linalg.solve(np.moveaxis(Gff, -2, -1), np.moveaxis(Gxf, -2, -1)), -2, -1)
    elif method == 'H2':
        if (response_ffts.shape != reference_ffts.shape):
            raise ValueError('For H2, Number of inputs must equal number of outputs')
        Gxx = np.einsum('...iaf,...jaf->...fij', response_ffts, np.conj(response_ffts))
        Gfx = np.einsum('...iaf,...jaf->...fij', reference_ffts, np.conj(response_ffts))
        singular_matrices = np.abs(np.linalg.det(Gfx)) < 2 * np.finfo(Gfx.dtype).eps
        Gfx[singular_matrices] += np.eye(Gfx.shape[-1]) * np.finfo(Gfx.dtype).eps
        H = np.moveaxis(np.linalg.solve(np.moveaxis(Gfx, -2, -1), np.moveaxis(Gxx, -2, -1)), -2, -1)
    elif method == 'H3':
        if (response_ffts.shape != reference_ffts.shape):
            raise ValueError('For H3, Number of inputs must equal number of outputs')
        Gxf = np.einsum('...iaf,...jaf->...fij', response_ffts, np.conj(reference_ffts))
        Gff = np.einsum('...iaf,...jaf->...fij', reference_ffts, np.conj(reference_ffts))
        # Add small values to any matrices that are singular
        singular_matrices = np.abs(np.linalg.det(Gff)) < 2 * np.finfo(Gff.dtype).eps
        Gff[singular_matrices] += np.eye(Gff.shape[-1]) * np.finfo(Gff.dtype).eps
        Gxx = np.einsum('...iaf,...jaf->...fij', response_ffts, np.conj(response_ffts))
        Gfx = np.einsum('...iaf,...jaf->...fij', reference_ffts, np.conj(response_ffts))
        singular_matrices = np.abs(np.linalg.det(Gfx)) < 2 * np.finfo(Gfx.dtype).eps
        Gfx[singular_matrices] += np.eye(Gfx.shape[-1]) * np.finfo(Gfx.dtype).eps
        H = (np.moveaxis(np.linalg.solve(np.moveaxis(Gfx, -2, -1), np.moveaxis(Gxx, -2, -1)), -2, -1) +
             np.moveaxis(np.linalg.solve(np.moveaxis(Gff, -2, -1), np.moveaxis(Gxf, -2, -1)), -2, -1)) / 2
    elif method == 'Hcd':
        Gxf = np.einsum('...iaf,...jaf->...fij', response_ffts, np.conj(reference_ffts))
        Gff = np.einsum('...iaf,...jaf->...fij', reference_ffts, np.conj(reference_ffts))
        # Add small values to any matrices that are singular
        singular_matrices = np.abs(np.linalg.det(Gff)) < 2 * np.finfo(Gff.dtype).eps
        Gff[singular_matrices] += np.eye(Gff.shape[-1]) * np.finfo(Gff.dtype).eps
        Lfz = np.linalg.cholesky(Gff)
        Lzf = np.conj(np.moveaxis(Lfz, -2, -1))
        Gxz = np.moveaxis(np.linalg.solve(np.moveaxis(
            Lzf, -2, -1), np.moveaxis(Gxf, -2, -1)), -2, -1)
        H = np.moveaxis(np.linalg.solve(np.moveaxis(Lfz, -2, -1), np.moveaxis(Gxz, -2, -1)), -2, -1)
    elif method == 'Hv':
        Gxx = np.einsum('...iaf,...iaf->...if', response_ffts,
                        np.conj(response_ffts))[..., np.newaxis, np.newaxis]
        Gxf = np.einsum('...iaf,...jaf->...ifj', response_ffts,
                        np.conj(reference_ffts))[..., np.newaxis, :]
        Gff = np.einsum('...iaf,...jaf->...fij', reference_ffts,
                        np.conj(reference_ffts))[..., np.newaxis, :, :, :]
        # Broadcast over all responses
        Gff = np.broadcast_to(Gff,Gxx.shape[:-2]+Gff.shape[-2:])
        Gffx = np.block([[Gff, np.conj(np.moveaxis(Gxf, -2, -1))],
                         [Gxf, Gxx]])
        # Compute eigenvalues
        lam, evect = np.linalg.eigh(np.moveaxis(Gffx, -2, -1))
        # Get the evect corresponding to the minimum eigenvalue
        evect = evect[..., 0]  # Assumes evals are sorted ascending
        H = np.moveaxis(-evect[..., :-1] / evect[..., -1:],  # Scale so last value is -1
                        -3, -2)
    elif method == 'LRM':
        frequencies, H, model_data = frf_local_model(reference_ffts,response_ffts,
                                                         frequencies,**lrm_kwargs)
    else:
        raise NotImplementedError('Method {:} has not been implemented yet!'.format(method))
        
    if return_model_data:
        if method != 'LRM':
            model_data = {'info': None,'modelset': None,'model_selected': None}
        return frequencies, H, model_data
    else:
        return frequencies, H


def fft2frf(references, responses, method='H1',freqs_in=None,**kwargs):
    '''Creates an FRF matrix given ffts of responses and references

    This function creates a nf x no x ni FRF matrix from the ffts
    provided.

    Parameters
    ----------
    references : ndarray
        A ni x nf or nf array where ni is the number of references and nt is
        the number of frequencies in the fft.
    responses : ndarray
        A no x nf or nf array where no is the number of responses and nf is the
        number of time frequencies in the fft.
    method : str in ['H1','H2','Hv','Hs','LRM']
        The method for creating the frequency response function.
    freqs_in : ndarray
        Only used if method == 'LRM'. A nf or nf x 1 array of frquency values.
    **lrm_kwargs
        Additional keyword arguments specify parameters if method == 'LRM'.
        Possible arguments are: f_out (ndarray), bandwidth (float), transient
        (boolean), modelset (iterable of (3,1) ndarray), max_parallel (int), 
        export_ratio (float). See sdynpy_lrm.frf_local_model.

    Returns
    -------
    H : ndarray
        A nf x no x ni array where nf is the number of frequency lines, no is
        the number of outputs, and ni is the number of inputs.  The output
        frequency lines will correspond to the frequency lines in the ffts.
    freqs_out : None or ndarray
        None unless method == 'LRM'. See sdynpy_lrm.frf_local_model.
    model_data: None dict
        None unless method == 'LRM'. See sdynpy_lrm.frf_local_model.

    Notes
    -----
    There are requirements for the shapes of references and responses for some
    FRF computation methods.  No averaging is performed by this function.

    '''
    if references.ndim == 1:
        references = references[np.newaxis, :]
    elif references.ndim > 2:
        raise ValueError('references should be at maximum a 2 dimensional array')
    if responses.ndim == 1:
        responses = responses[np.newaxis, :]
    elif responses.ndim > 2:
        raise ValueError('responses should be at maximum a 2 dimensional array')
    if not method in ['H1', 'H2', 'Hs', 'Hv', 'LRM']:
        raise ValueError('method parameter must be one of ["H1","H2","Hs","Hv","LRM"]')
#    H = np.zeros((references.shape[1],responses.shape[0],references.shape[0]),dtype=complex)
#    for i in range(H.shape[1]):
#            for j in range(H.shape[2]):
#                if method == 'H1':
#                    H[:,i,j] = (responses[i,:]*references[j,:].conj())/(references[j,:]*references[j,:].conj())
#                else:
#                    raise NotImplementedError('Method {:} has not been implemented'.format(method))
    if method == 'H1':
        Gxf = np.einsum('if,jf->fij', responses, references.conj())
        Gff = np.einsum('if,jf->fij', references, references.conj())
        H = np.linalg.solve(Gff.transpose(0, 2, 1), Gxf.transpose((0, 2, 1))).transpose((0, 2, 1))
    elif method == 'LRM':
        if freqs_in is None:
            raise ValueError('if method == \'LRM\', freqs_in must be specified')
        import sdynpy.signal_processing.sdynpy_lrm as frflm
        freqs_out, H, model_data = frflm.frf_local_model(references, responses, 
                                                         freqs_in, **lrm_kwargs)
    else:
        raise NotImplementedError('Method {:} has not been implemented'.format(method))
    if method != 'LRM':
        freqs_out = None
        model_data = None
    return H, freqs_out, model_data


def plot(H, f, responses=None, references=None, real_imag=False):
    fig = plt.figure()
    phase_axis = fig.add_subplot(2, 1, 1)
    mag_axis = fig.add_subplot(2, 1, 2)

    if responses is None:
        responses = np.arange(H.shape[1])[:, np.newaxis]
    if references is None:
        references = np.arange(H.shape[2])

    H_to_plot = H[:, responses, references]
    H_to_plot = H_to_plot.transpose(*np.arange(1, H_to_plot.ndim), 0)

    for index in np.ndindex(H_to_plot.shape[:-1]):
        if real_imag:
            phase_axis.plot(f, np.imag(H_to_plot[index]))
            mag_axis.plot(f, np.real(H_to_plot[index]))
        else:
            phase_axis.plot(f, np.angle(H_to_plot[index]) * 180 / np.pi)
            mag_axis.plot(f, np.abs(H_to_plot[index]))

    if real_imag:
        phase_axis.set_ylabel('Imag(H)')
        mag_axis.set_ylabel('Real(H)')
    else:
        phase_axis.set_ylabel('Angle(H)')
        mag_axis.set_ylabel('Abs(H)')
        mag_axis.set_yscale('log')
    mag_axis.set_xlabel('Frequency')
    fig.tight_layout()


def delay_signal(times, signal, dt):
    '''Delay a time signal by the specified amount

    This function takes a signal and delays it by a specified amount of time
    that need not be an integer number of samples.  It does this by adjusting
    the phaes of the signal's FFT.

    Parameters
    ----------
    times : np.ndarray
        A signal specifying the time values at which the samples in signal
        occur.
    signal : np.ndarray
        A signal that is to be delayed (n_signals x len(times))
    dt : float
        The amount of time to delay the signal

    Returns
    -------
    updated_signal : np.ndarray
        The time-shifted signal.
    '''
    fft_omegas = np.fft.fftfreq(len(times), np.mean(np.diff(times))) * 2 * np.pi
    signal_fft = np.fft.fft(signal, axis=-1)
    signal_fft *= np.exp(-1j * fft_omegas * dt)
    return np.fft.ifft(signal_fft, axis=-1).real.astype(signal.dtype)
