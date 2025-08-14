# -*- coding: utf-8 -*-
"""
Functions for dealing with sinusoidal data

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
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.signal import lfilter, lfiltic, butter, windows
from scipy.optimize import minimize
from scipy.sparse import linalg
from scipy import sparse
from .sdynpy_buffer import CircularBufferWithOverlap


def harmonic_mag_phase(ts, xs, frequency, nharmonics=1, return_residual = False):
    A = np.zeros((ts.size, nharmonics * 2 + 1))
    for i in range(nharmonics):
        A[:, i] = np.sin(2 * np.pi * frequency * (i + 1) * ts)
        A[:, nharmonics + i] = np.cos(2 * np.pi * frequency * (i + 1) * ts)
    A[:, -1] = np.ones(ts.size)
    coefs = la.lstsq(A, xs[:, np.newaxis])[0]
    As = np.array(coefs[:nharmonics, :])
    Bs = np.array(coefs[nharmonics:nharmonics * 2, :])
    phases = np.arctan2(Bs, As)[:, 0]
    magnitudes = np.sqrt(As**2 + Bs**2)[:, 0]
    if return_residual:
        return magnitudes, phases, (A@coefs - xs[:,np.newaxis])
    else:
        return magnitudes, phases

def harmonic_mag_phase_fit(ts,xs, frequency_guess, nharmonics = 1,**minimize_options):
    
    def min_fun(frequency):
        return np.mean(harmonic_mag_phase(ts, xs, frequency[0], nharmonics, True)[2]**2)
    
    result = minimize(min_fun, [frequency_guess], **minimize_options)
    
    frequency_fit = result.x[0]
    mag, phs = harmonic_mag_phase(ts, xs, frequency_fit)
    return frequency_fit, mag, phs


def digital_tracking_filter(dt, xs, frequencies, arguments, cutoff_frequency_ratio = 0.15,
                            filter_order = 2, phase_estimate = None, amplitude_estimate = None,
                            block_size = None, plot_results = False,
                            ):
    """
    Computes amplitudes and phases using a digital tracking filter

    Parameters
    ----------
    dt : float
        The time step of the signal
    xs : iterable
        The signal that will have amplitude and phase extracted.
    frequencies : iterable
        The instantaneous frequency at each time step.
    arguments : iterable
        The instantaneous argument to the sine wave at each time step.
    cutoff_frequency_ratio : float
        The cutoff frequency of the low-pass filter compared to the lowest
        frequency sine tone in each block.  Default is 0.15.
    filter_order : float
        The filter order of the low-pass butterworth filter.  Default is 2.
    phase_estimate : float
        An estimate of the initial phase to seed the low-pass filter.
    amplitude_estimate : float
        An estimate of the initial amplitude to seed the low-pass filter.
    block_size : int
        Number of samples to use for each block.  If not specified, the entire
        signal is treated as a single block.
    plot_results : bool
        If True, will plot the data at multiple steps for diagnostics

    Returns
    -------
    amplitude : np.ndarray
        The amplitude at each time step
    phase : np.ndarray
        The phase at each time step
    """
    if block_size is None:
        block_size = xs.shape[-1]
    if plot_results:
        fig,ax = plt.subplots(2,2,sharex=True)
        ax[0,0].set_ylabel('Signal and Amplitude')
        ax[0,1].set_ylabel('Phase')
        ax[1,0].set_ylabel('Filtered COLA Signal (cos)')
        ax[1,1].set_ylabel('Filtered COLA Signal (sin)')
    if phase_estimate is None:
        phase_estimate = 0
    if amplitude_estimate is None:
        amplitude_estimate = 0
    
    xi_0_filt = None
    xi_90_filt = None
    xi_0 = None
    xi_90 = None
    frame_index = 0
    xs = np.array(xs)
    frequencies = np.array(frequencies)
    arguments = np.array(arguments)
    phases = []
    amplitudes = []
    while frame_index*block_size < xs.shape[-1]:
        idx = (Ellipsis,slice(frame_index*block_size,(frame_index+1)*block_size))
        fi = frequencies[idx]
        b,a = butter(filter_order, cutoff_frequency_ratio*np.min(fi),fs=1/dt)
        if xi_0_filt is None:
            # Set up some fake data to initialize the filter to a good value
            past_ts = np.arange(-filter_order*2-1,0)*dt
            past_xs = amplitude_estimate * np.cos(2*np.pi*fi[0]*past_ts + phase_estimate)
            xi_0 = np.cos(2*np.pi*fi[0]*past_ts)*past_xs
            xi_90 = -np.sin(2*np.pi*fi[0]*past_ts)*past_xs
            xi_0_filt = 0.5*amplitude_estimate*np.cos(phase_estimate)*np.ones(xi_0.shape)
            xi_90_filt = 0.5*amplitude_estimate*np.sin(phase_estimate)*np.ones(xi_90.shape)
            if plot_results:
                ax[1,0].plot(past_ts,xi_0,'r')
                ax[1,0].plot(past_ts,xi_0_filt,'m')
                ax[1,1].plot(past_ts,xi_90,'r')
                ax[1,1].plot(past_ts,xi_90_filt,'m')
        # Set up the filter initial states
        z0i = lfiltic(b,a,xi_0_filt[::-1],xi_0[::-1])
        z90i = lfiltic(b,a,xi_90_filt[::-1],xi_90[::-1])
        # Extract this portion of the signal
        xi = xs[idx]
        ti = np.arange(frame_index*block_size,(frame_index+1)*block_size)*dt
        ti = ti[...,:xi.shape[-1]]
        argsi = arguments[idx]
        # Now set up the tracking filter
        cola0 = np.cos(argsi)
        cola90 = -np.sin(argsi)
        xi_0 = cola0*xi
        xi_90 = cola90*xi
        xi_0_filt,z0i = lfilter(b,a,xi_0,zi=z0i)
        xi_90_filt,z90i = lfilter(b,a,xi_90,zi=z90i)
        phases.append(np.arctan2(xi_90_filt,xi_0_filt))
        amplitudes.append(2*np.sqrt(xi_0_filt**2 + xi_90_filt**2))
        frame_index += 1
        if plot_results:
            ax[0,0].plot(ti,xi,'b')
            ax[0,0].plot(ti,amplitudes[-1],'g')
            ax[0,1].plot(ti,phases[-1],'g')
            ax[1,0].plot(ti,xi_0,'b')
            ax[1,0].plot(ti,xi_0_filt,'g')
            ax[1,1].plot(ti,xi_90,'b')
            ax[1,1].plot(ti,xi_90_filt,'g')
    amplitude = np.concatenate(amplitudes,axis=-1)
    phase = np.concatenate(phases,axis=-1)
    if plot_results:
        fig.tight_layout()
    return amplitude,phase

def digital_tracking_filter_generator(
        dt, cutoff_frequency_ratio = 0.15,
        filter_order = 2, phase_estimate = None, amplitude_estimate = None,
        plot_results = False,
        ):
    """
    Computes amplitudes and phases using a digital tracking filter

    Parameters
    ----------
    dt : float
        The time step of the signal
    cutoff_frequency_ratio : float
        The cutoff frequency of the low-pass filter compared to the lowest
        frequency sine tone in each block.  Default is 0.15.
    filter_order : float
        The filter order of the low-pass butterworth filter.  Default is 2.
    phase_estimate : float
        An estimate of the initial phase to seed the low-pass filter.
    amplitude_estimate : float
        An estimate of the initial amplitude to seed the low-pass filter.
    plot_results : bool
        If True, will plot the data at multiple steps for diagnostics

    Sends
    ------
    xi : iterable
        The next block of the signal to be filtered
    fi : iterable
        The frequencies at the time steps in xi
    argsi : iterable
        The argument to a cosine function at the time steps in xi

    Yields
    ------
    amplitude : np.ndarray
        The amplitude at each time step
    phase : np.ndarray
        The phase at each time step
    """
    if plot_results:
        fig,ax = plt.subplots(2,2,sharex=True)
        ax[0,0].set_ylabel('Signal and Amplitude')
        ax[0,1].set_ylabel('Phase')
        ax[1,0].set_ylabel('Filtered COLA Signal (cos)')
        ax[1,1].set_ylabel('Filtered COLA Signal (sin)')
        sample_index = 0
        fig.tight_layout()
    if phase_estimate is None:
        phase_estimate = 0
    if amplitude_estimate is None:
        amplitude_estimate = 0
    
    xi_0_filt = None
    xi_90_filt = None
    xi_0 = None
    xi_90 = None
    amplitude = None
    phase = None
    while True:
        xi, fi, argsi = yield amplitude,phase
        xi = np.array(xi)
        fi = np.array(fi)
        argsi = np.array(argsi)
        b,a = butter(filter_order, cutoff_frequency_ratio*np.min(fi),fs=1/dt)
        if xi_0_filt is None:
            # Set up some fake data to initialize the filter to a good value
            past_ts = np.arange(-filter_order*2-1,0)*dt
            past_xs = amplitude_estimate * np.cos(2*np.pi*fi[0]*past_ts + phase_estimate)
            xi_0 = np.cos(2*np.pi*fi[0]*past_ts)*past_xs
            xi_90 = -np.sin(2*np.pi*fi[0]*past_ts)*past_xs
            xi_0_filt = 0.5*amplitude_estimate*np.cos(phase_estimate)*np.ones(xi_0.shape)
            xi_90_filt = 0.5*amplitude_estimate*np.sin(phase_estimate)*np.ones(xi_90.shape)
            if plot_results:
                ax[1,0].plot(past_ts,xi_0,'r')
                ax[1,0].plot(past_ts,xi_0_filt,'m')
                ax[1,1].plot(past_ts,xi_90,'r')
                ax[1,1].plot(past_ts,xi_90_filt,'m')
        # Set up the filter initial states
        z0i = lfiltic(b,a,xi_0_filt[::-1],xi_0[::-1])
        z90i = lfiltic(b,a,xi_90_filt[::-1],xi_90[::-1])
        # Now set up the tracking filter
        cola0 = np.cos(argsi)
        cola90 = -np.sin(argsi)
        xi_0 = cola0*xi
        xi_90 = cola90*xi
        xi_0_filt,z0i = lfilter(b,a,xi_0,zi=z0i)
        xi_90_filt,z90i = lfilter(b,a,xi_90,zi=z90i)
        phase = np.arctan2(xi_90_filt,xi_0_filt)
        amplitude = 2*np.sqrt(xi_0_filt**2 + xi_90_filt**2)
        if plot_results:
            ti = np.arange(sample_index,sample_index + xi.shape[-1])*dt
            ax[0,0].plot(ti,xi,'b')
            ax[0,0].plot(ti,amplitude,'g')
            ax[0,1].plot(ti,phase,'g')
            ax[1,0].plot(ti,xi_0,'b')
            ax[1,0].plot(ti,xi_0_filt,'g')
            ax[1,1].plot(ti,xi_90,'b')
            ax[1,1].plot(ti,xi_90_filt,'g')
            sample_index += xi.shape[-1]
            
def vold_kalman_filter(sample_rate, signal, arguments, filter_order = None,
                       bandwidth = None, method = None, return_amp_phs = False,
                       return_envelope = False, return_r = False,):
    """
    Extract sinusoidal components from a signal using the second generation
    Vold-Kalman filter.

    Parameters
    ----------
    sample_rate : float
        The sample rate of the signal in Hz.
    signal : ndarray
        A 1D signal containing sinusoidal components that need to be extracted
    arguments : ndarray
        A 2D array consisting of the arguments to the sinusoidal components of
        the form exp(1j*argument).  This is the integral over time of the
        angular frequency, which can be approximated as
        2*np.pi*scipy.integrate.cumulative_trapezoid(frequencies,timesteps,initial=0)
        if frequencies is the frequency at each time step in Hz timesteps is
        the vector of time steps in seconds.  This is a 2D array where the
        number of rows is the
        number of different sinusoidal components that are desired to be
        extracted, and the number of columns are the number of time steps in
        the `signal` argument.
    filter_order : int, optional
        The order of the VK filter, which should be 1, 2, or 3. The default is
        2.  The low-pass filter roll-off is approximately -40 dB per times the
        filter order.
    bandwidth : ndarray, optional
        The prescribed bandwidth of the filter. This is related to the filter
        selectivity parameter `r` in the literature.  This will be broadcast to
        the same shape as the `arguments` argument.  The default is the sample
        rate divided by 1000.
    method : str, optional
        Can be set to either 'single' or 'multi'.  In a 'single' solution, each
        sinusoidal component will be solved independently without any coupling.
        This can be more efficient, but will result in errors if the
        frequencies of the sine waves cross.  The 'multi' solution will solve
        for all sinusoidal components simultaneously, resulting in a better
        estimate of crossing frequencies. The default is 'multi'.
    return_amp_phs : bool
        Returns the amplitude and phase of the reconstructed signals at each
        time step.  Default is False
    return_envelope : bool
        Returns the complex envelope and phasors at each time step.  Default is
        False
    return_r : bool
        Returns the computed selectivity parameters for the filter.  Default is
        False

    Raises
    ------
    ValueError
        If arguments are not the correct size or values.

    Returns
    -------
    reconstructed_signals : ndarray
        Returns a time history the same size as `signal` for each of the
        sinusoidal components solved for.
    reconstructed_amplitudes : ndarray
        Returns the amplitude over time for each of the sinusoidal components
        solved for.  Only returned if return_amp_phs is True.
    reconstructed_phases : ndarray
        Returns the phase over time for each of the sinusoidal components
        solved for.  Only returned if return_amp_phs is True.
    reconstructed_envelope : ndarray
        Returns the complex envelope `x` over time for each of the sinusoidal
        components solved for.  Only returned if return_envelope is True.
    reconstructed_phasor : ndarray
        Returns the phasor `c` over time for each of the sinusoidal components
        solved for.  Only returned if return_envelope is True.
    r : ndarray
        Returns the selectivity `r` over time for each of the sinusoidal
        components solved for.  Only returned if return_r is True.

    """
    if filter_order is None:
        filter_order = 2
    
    if bandwidth is None:
        bandwidth = sample_rate/1000
    
    # Make sure input data are numpy arrays
    signal = np.array(signal)
    arguments = np.atleast_2d(arguments)
    bandwidth = np.atleast_2d(bandwidth)
    bandwidth = np.broadcast_to(bandwidth,arguments.shape)
    relative_bandwidth = bandwidth/sample_rate
    
    # Extract some sizes to make sure everything is correctly sized
    n_samples = signal.shape[-1]
    # n_orders_freq, n_freq = frequencies.shape
    # if n_freq != n_samples:
    #     raise ValueError('Frequency array must have identical number of columns as samples in signal')
    n_orders_arg, n_arg = arguments.shape
    if n_arg != n_samples:
        raise ValueError('Argument array must have identical number of columns as samples in signal')
    
    if method is None:
        if n_orders_arg > 1:
            method = 'multi'
        else:
            method = 'single'
    if method.lower() not in ['multi','single']:
        raise ValueError('`method` must be either "multi" or "single"')
    
    # Construct phasors to multiply the signals by
    phasor = np.exp(1j*arguments)
    
    # Construct the matrices for the least squares solution
    if filter_order == 1:
        coefs = np.array([1,-1])
        r = np.sqrt((np.sqrt(2)-1)/
                    (2*(1-np.cos(np.pi*relative_bandwidth))))
    elif filter_order == 2:
        coefs = np.array([1,-2,1])
        r = np.sqrt((np.sqrt(2)-1)/
                    (6-8*np.cos(np.pi*relative_bandwidth)+2*np.cos(2*np.pi*relative_bandwidth)))
    elif filter_order == 3:
        coefs = np.array([1,-3,3,-1])
        r = np.sqrt((np.sqrt(2)-1)/
                    (20-30*np.cos(np.pi*relative_bandwidth)+12*np.cos(2*np.pi*relative_bandwidth)-2*np.cos(3*np.pi*relative_bandwidth)))
    else:
        raise ValueError('filter order must be 1, 2, or 3')
    
    # Construct the solution matrices
    A = sparse.spdiags(np.tile(coefs,(n_samples,1)).T,
                       np.arange(filter_order+1),
                       n_samples-filter_order,
                       n_samples)
    B = []
    for rvec in r:
        R = sparse.spdiags(rvec,0,n_samples,n_samples)
        AR = A@R
        B.append((AR).T@(AR) + sparse.eye(n_samples))
    
    if method.lower() == 'multi':
        # This solves the multiple order approach, constructing a big matrix of
        # Bs on the diagonal and CHCs on the off-diagonals.  We can set up the
        # matrix as diagonals and upper diagonals then add the transpose to get the
        # lower diagonals
        B_multi_diagonal = sparse.block_diag(B)
        # There will be number of orders**2 B matrices, and number of orders
        # diagonals, so there will be n_orders**2-n_orders off diagonals, half on
        # on the upper triangle.  We need to fill in all of these values for all
        # time steps.
        num_off_diags = (n_orders_arg**2-n_orders_arg)//2
        row_indices = np.zeros((n_samples,num_off_diags),dtype=int)
        col_indices = np.zeros((n_samples,num_off_diags),dtype=int)
        CHC = np.zeros((n_samples,num_off_diags),dtype='c16')
        # Keep track of the off-diagonal index so we know which column to put the
        # data in
        off_diagonal_index = 0
        # Now we need to step through the off-diagonal blocks and create the arrays
        for row_index in range(n_orders_arg):
            # Since we need to stay on the upper triangle, column indices will start
            # after the diagonal entry
            for col_index in range(row_index+1,n_orders_arg):
                row_indices[:,off_diagonal_index] = np.arange(row_index*n_samples,(row_index+1)*n_samples)
                col_indices[:,off_diagonal_index] = np.arange(col_index*n_samples,(col_index+1)*n_samples)
                CHC[:,off_diagonal_index] = phasor[row_index].conj()*phasor[col_index]
                off_diagonal_index += 1
        # We set up the variables as multidimensional so we could store them easier,
        # but now we need to flatten them to put them into the sparse matrix.
        # We choose CSR because we can do math with it easier
        B_multi_utri = sparse.csr_matrix((CHC.flatten(),(row_indices.flatten(),col_indices.flatten())),shape=B_multi_diagonal.shape)
    
        # Now we can assemble the entire matrix by adding with the complex conjugate
        # of the upper triangle to get the lower triangle
        B_multi = B_multi_diagonal + B_multi_utri + B_multi_utri.getH()
    
        # We also need to construct the right hand side of the equation.  This
        # should be a multiplication of the phasor^H with the signal
        RHS = phasor.flatten().conj()*np.tile(signal,n_orders_arg)
    
        x_multi = linalg.spsolve(B_multi,RHS[:,np.newaxis])
        x = 2*x_multi.reshape(n_orders_arg,-1) # Multiply by 2 to account for missing negative frequency components
    else:
        # This solves the single order approach.  If the user has put in multiple
        # orders, it will solve them all independently instead of combining them
        # into a single larger solve.
        x = np.zeros((n_orders_arg,n_samples),dtype=np.complex128)
        for i,(phasor_i, B_i) in enumerate(zip(phasor,B)):
            # We already have the left side of the equation B, now we just need the
            # right side of the equation, which is the phasor hermetian
            # times the signal elementwise-multiplied
            RHS = phasor_i.conj()*signal
            x[i] = 2*linalg.spsolve(B_i,RHS)
    
    return_value = [np.real(x*phasor)]
    if return_amp_phs:
        return_value.extend([np.abs(x),np.angle(x)])
    if return_envelope:
        return_value.extend([x,phasor])
    if return_r:
        return_value.extend([r])
    if len(return_value) == 1:
        return return_value[0]
    else:
        return return_value
    
def vold_kalman_filter_generator(sample_rate, num_orders, block_size, overlap,
                                 filter_order = None,
                                 bandwidth = None, method = None,
                                 yield_amp_phs = False,
                                 yield_envelope = False, plot_results = False):
    """
    Extracts sinusoidal information using a Vold-Kalman Filter
    
    This uses an windowed-overlap-and-add process to solve for the signal while
    removing start and end effects of the filter.  Each time the generator is
    called, it will yield a further section of the results up until the overlap
    section.

    Parameters
    ----------
    sample_rate : float
        The sample rate of the signal in Hz.
    num_orders : int
        The number of orders that will be found in the signal
    block_size : int
        The size of the blocks used in the analysis.
    filter_order : int, optional
        The order of the VK filter, which should be 1, 2, or 3. The default is
        2.  The low-pass filter roll-off is approximately -40 dB per times the
        filter order.
    bandwidth : ndarray, optional
        The prescribed bandwidth of the filter. This is related to the filter
        selectivity parameter `r` in the literature.  This will be broadcast to
        the same shape as the `arguments` argument.  The default is the sample
        rate divided by 1000.
    method : str, optional
        Can be set to either 'single' or 'multi'.  In a 'single' solution, each
        sinusoidal component will be solved independently without any coupling.
        This can be more efficient, but will result in errors if the
        frequencies of the sine waves cross.  The 'multi' solution will solve
        for all sinusoidal components simultaneously, resulting in a better
        estimate of crossing frequencies. The default is 'multi'.
    overlap : float, optional
        Fraction of the block size to overlap when computing the results.  If
        not specified, it will default to 0.15.
    plot_results : bool
        If True, will plot the data at multiple steps for diagnostics

    Raises
    ------
    ValueError
        If arguments are not the correct size or values.
    ValueError
        If data is provided subsequently to specifying last_signal = True

    Sends
    -----
    xi : iterable
        The next block of the signal to be filtered.  This should be a 1D
        signal containing sinusoidal components that need to be extracted.
    argsi : iterable
        A 2D array consisting of the arguments to the sinusoidal components of
        the form exp(1j*argsi).  This is the integral over time of the
        angular frequency, which can be approximated as
        2*np.pi*scipy.integrate.cumulative_trapezoid(frequencies,timesteps,initial=0)
        if frequencies is the frequency at each time step in Hz timesteps is
        the vector of time steps in seconds.  This is a 2D array where the
        number of rows is the
        number of different sinusoidal components that are desired to be
        extracted, and the number of columns are the number of time steps in
        the `signal` argument.
    last_signal : bool
        If True, the remainder of the data will be returned and the
        overlap-and-add process will be finished.

    Yields
    -------
    reconstructed_signals : ndarray
        Returns a time history the same size as `signal` for each of the
        sinusoidal components solved for.
    reconstructed_amplitudes : ndarray
        Returns the amplitude over time for each of the sinusoidal components
        solved for.  Only returned if return_amp_phs is True.
    reconstructed_phases : ndarray
        Returns the phase over time for each of the sinusoidal components
        solved for.  Only returned if return_amp_phs is True.

    """
    if plot_results:
        fig,ax = plt.subplots(num_orders+1,3)
        signal_index = 0
        analysis_index = 0
    print('Startup')
    previous_envelope = None
    reconstructed_signals = None
    reconstructed_amplitudes = None
    reconstructed_phases = None
    overlap_samples = int(overlap*block_size)
    window = windows.hann(overlap_samples*2,False)
    start_window = window[:overlap_samples]
    end_window = window[overlap_samples:]
    buffer = CircularBufferWithOverlap(3*block_size, block_size, overlap_samples, data_shape=(num_orders+1,))
    first_output = True
    while True:
        print('Before Yield')
        xi,argsi,last_signal = yield reconstructed_signals,reconstructed_amplitudes, reconstructed_phases
        argsi = np.atleast_2d(argsi)
        if plot_results:
            timesteps_signal = np.arange(signal_index, signal_index + xi.shape[-1])/sample_rate
            ax[0,0].plot(timesteps_signal,xi,'k')
            signal_index += xi.shape[-1]
        print('After Yield')
        buffer_data = np.concatenate([xi[np.newaxis],argsi])
        buffer_output = buffer.write_get_data(buffer_data)
        if buffer_output is not None:
            if first_output:
                buffer_output = buffer_output[...,overlap_samples:]
                first_output = False
                signal = buffer_output[0]
                arguments = buffer_output[1:]
            else:
                signal = buffer_output[0]
                arguments = buffer_output[1:]
                signal[:overlap_samples] = signal[:overlap_samples]*start_window
            if not last_signal:
                signal[-overlap_samples:] = signal[-overlap_samples:]*end_window
            if plot_results:
                timesteps_analysis = np.arange(analysis_index, analysis_index + signal.shape[-1])/sample_rate
                ax[0,1].plot(timesteps_analysis,signal)
                analysis_index += signal.shape[-1] - overlap_samples
            # Do the VK Filtering
            vk_signal, vk_envelope, vk_phasor = vold_kalman_filter(
                sample_rate, signal, arguments, filter_order,
                bandwidth, method, return_envelope=True)
            if plot_results:
                for vks,vke,a in zip(vk_signal,vk_envelope,ax[1:]):
                    a[0].plot(timesteps_analysis,vks.copy())
                    a[1].plot(timesteps_analysis,np.abs(vke))
                    a[2].plot(timesteps_analysis,np.angle(vke))
            # If necessary, do the overlap
            if previous_envelope is not None:
                vk_envelope[...,:overlap_samples] = vk_envelope[...,:overlap_samples] + previous_envelope[...,-overlap_samples:]
            reconstructed_signals = np.real(vk_envelope[...,:-overlap_samples]*vk_phasor[...,:-overlap_samples])
            reconstructed_amplitudes = np.abs(vk_envelope[...,:-overlap_samples])
            reconstructed_phases = np.angle(vk_envelope[...,:-overlap_samples])
            if plot_results:
                for vks,vka,vkp,a in zip(reconstructed_signals,reconstructed_amplitudes,reconstructed_phases,ax[1:]):
                    a[0].plot(timesteps_analysis[:-overlap_samples],vks,'k')
                    a[1].plot(timesteps_analysis[:-overlap_samples],vka,'k')
                    a[2].plot(timesteps_analysis[:-overlap_samples],vkp,'k')
            previous_envelope = vk_envelope
        else:
            outputs = None
        
        
        
        