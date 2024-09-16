# -*- coding: utf-8 -*-
"""
Functions for dealing with CPSD Matrices

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


def cpsd_coherence(cpsd):
    num = np.abs(cpsd)**2
    den = (cpsd[:, np.newaxis, np.arange(cpsd.shape[1]), np.arange(cpsd.shape[2])] *
           cpsd[:, np.arange(cpsd.shape[1]), np.arange(cpsd.shape[2]), np.newaxis])
    den[den == 0.0] = 1  # Set to 1
    return np.real(num /
                   den)


def cpsd_phase(cpsd):
    return np.angle(cpsd)


def cpsd_from_coh_phs(asd, coh, phs):
    return np.exp(phs * 1j) * np.sqrt(coh * asd[:, :, np.newaxis] * asd[:, np.newaxis, :])


def cpsd_autospectra(cpsd):
    return np.einsum('ijj->ij', cpsd)


def trace(cpsd):
    return np.einsum('ijj->i', cpsd)


def match_coherence_phase(cpsd_original, cpsd_to_match):
    coh = cpsd_coherence(cpsd_to_match)
    phs = cpsd_phase(cpsd_to_match)
    asd = cpsd_autospectra(cpsd_original)
    return cpsd_from_coh_phs(asd, coh, phs)


def cpsd(signals: np.ndarray, sample_rate: int,
         samples_per_frame: int, overlap: float,
         window: str, averages_to_keep: int = None,
         only_asds: bool = False, reference_signals = None):
    """
    Compute cpsd from signals

    Parameters
    ----------
    signals : np.ndarray
        2D numpy array containing the signals, with each row containing a
        signal and each column containing a sample of each signal
    sample_rate : int
        Sample rate of the signal.
    samples_per_frame : int
        Number of samples per frame in the CPSD
    overlap : float
        Overlap as a fraction of the frame (e.g. 0.5 not 50).
    window : str
        Name of a window function in scipy.signal.windows, or a NumPy ndarray
        with the window coefficients
    averages_to_keep : int, optional
        Optional number of averages to use. The default is None.
    only_asds : bool, optional
        If True, only compute autospectral densities, otherwise compute the
        full CPSD matrix
    reference_signals : np.ndarray
        If specified, these signals will be used for the columns of the CPSD
        matrix.  If not specified, the signals passed to the `signals` argument
        will be used for both the rows and columns of the CPSD matrix.  Cannot
        be used simultaneously with the `only_asds` argument being set to True.

    Returns
    -------
    frequency_spacing : float
        The frequency spacing of the CPSD matrix
    response_spectral_matrix : np.ndarray
        A complex array with dimensions number of frequency lines by
        number of signals (by number of signals, if only_asds is False)

    """

    samples_per_acquire = int(samples_per_frame * (1 - overlap))
    if isinstance(window[0],str): # This will pass if it is a string or a container with a string as the first value
        window_array = sig.windows.get_window(window, samples_per_frame, fftbins=True)
    else:
        window_array = window
    window_correction = 1 / np.mean(window_array**2)
    frequency_spacing = sample_rate / samples_per_frame
    time_starts = np.arange(0, signals.shape[-1] - samples_per_frame + 1, samples_per_acquire)
    time_indices_for_averages = time_starts[:, np.newaxis] + np.arange(samples_per_frame)
    if averages_to_keep is not None:
        time_indices_for_averages = time_indices_for_averages[:averages_to_keep]
    response_fft_array = signals[:, time_indices_for_averages]
    response_fft = np.fft.rfft(response_fft_array[:, :, :] * window_array, axis=-1)
    if only_asds:
        if reference_signals is not None:
            raise ValueError('Cannot specify `only_asds` equal to True and simultaneously provide reference signals')
        response_spectral_matrix = np.einsum(
            'iaf,iaf->fi', response_fft, np.conj(response_fft)) / response_fft.shape[1]
    else:
        if reference_signals is None:
            reference_fft = response_fft
        else:
            reference_fft_array = reference_signals[:,time_indices_for_averages]
            reference_fft = np.fft.rfft(reference_fft_array[:, :, :] * window_array, axis=-1)
        response_spectral_matrix = np.einsum(
            'iaf,jaf->fij', response_fft, np.conj(reference_fft)) / response_fft.shape[1]
    # Normalize
    response_spectral_matrix *= (frequency_spacing * window_correction /
                                 sample_rate**2)
    response_spectral_matrix[1:-1] *= 2
    return frequency_spacing, response_spectral_matrix

def dB_pow(x): return 10 * np.log10(x)


def db2scale(dB):
    """ Converts a decibel value to a scale factor

    Parameters
    ----------
    dB : float :
        Value in decibels


    Returns
    -------
    scale : float :
        Value in linear

    """
    return 10**(dB/20)


def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))


def rms_csd(csd, df):
    """Computes RMS of a CPSD matrix

    Parameters
    ----------
    csd : np.ndarray :
        3D complex Numpy array where the first dimension is the frequency line
        and the second two dimensions are the rows and columns of the CPSD
        matrix.
    df : float :
        Frequency spacing of the CPSD matrix

    Returns
    -------
    rms : numpy scalar or numpy.ndarray
        The root-mean-square value of signals in the CPSD matrix

    """
    return np.sqrt(np.einsum('ijj->j', csd).real * df)


def plot_cpsd_error(frequencies, spec, channel_names=None, figure_kwargs={}, linewidth=1, plot_kwargs={}, **kwargs):
    spec_asd = np.real(np.einsum('ijj->ji', spec))
    data_asd = {legend: np.real(np.einsum('ijj->ji', data)) for legend, data in kwargs.items()}
    num_channels = spec_asd.shape[0]
    if channel_names is None:
        channel_names = ['Channel {:}'.format(i) for i in range(num_channels)]
    ncols = int(np.floor(np.sqrt(num_channels)))
    nrows = int(np.ceil(num_channels / ncols))
    if len(kwargs) > 1:
        total_rows = nrows + 2
    elif len(kwargs) == 1:
        total_rows = nrows + 1
    else:
        total_rows = nrows
    fig = plt.figure(**figure_kwargs)
    grid_spec = plt.GridSpec(total_rows, ncols, figure=fig)
    for i in range(num_channels):
        this_row = i // ncols
        this_col = i % ncols
        if i == 0:
            ax = fig.add_subplot(grid_spec[this_row, this_col])
            original_ax = ax
        else:
            ax = fig.add_subplot(grid_spec[this_row, this_col], sharex=original_ax,
                                 sharey=original_ax)
        ax.plot(frequencies, spec_asd[i], linewidth=linewidth * 2, color='k', **plot_kwargs)
        for legend, data in data_asd.items():
            ax.plot(frequencies, data[i], linewidth=linewidth)
        ax.set_ylabel(channel_names[i])
        if i == 0:
            ax.set_yscale('log')
        if this_row == nrows - 1:
            ax.set_xlabel('Frequency (Hz)')
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        if this_col != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
    return_data = None
    if len(kwargs) > 0:
        spec_sum_asd = np.sum(spec_asd, axis=0)
        data_sum_asd = {legend: np.sum(data, axis=0) for legend, data in data_asd.items()}
        db_error = {legend: rms(dB_pow(data) - dB_pow(spec_asd), axis=0)
                    for legend, data in data_asd.items()}
        plot_width = ncols // 2
        ax = fig.add_subplot(grid_spec[nrows, 0:plot_width])
        ax.plot(frequencies, spec_sum_asd, linewidth=2 * linewidth, color='k')
        for legend, data in data_sum_asd.items():
            ax.plot(frequencies, data, linewidth=linewidth)
        ax.set_yscale('log')
        ax.set_ylabel('Sum ASDs')
        ax = fig.add_subplot(grid_spec[nrows, -plot_width:])
        for legend, data in db_error.items():
            ax.plot(frequencies, data, linewidth=linewidth)
        ax.set_ylabel('dB Error')
    if len(kwargs) > 1:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color'] * 10
        db_error_sum_asd = {legend: rms(dB_pow(sum_asd) - dB_pow(spec_sum_asd))
                            for legend, sum_asd in data_sum_asd.items()}
        db_error_rms = {legend: rms(data) for legend, data in db_error.items()}
        return_data = (db_error_sum_asd, db_error_rms)
        ax = fig.add_subplot(grid_spec[nrows + 1, 0:plot_width])
        for i, (legend, data) in enumerate(db_error_sum_asd.items()):
            ax.bar(i, data, color=colors[i])
            ax.text(i, 0, '{:.2f}'.format(data),
                    horizontalalignment='center', verticalalignment='bottom')
        ax.set_xticks(np.arange(i + 1))
        ax.set_xticklabels([legend.replace('_', ' ')
                           for legend in db_error_sum_asd], rotation=20, horizontalalignment='right')
        ax.set_ylabel('Sum RMS dB Error')
        ax = fig.add_subplot(grid_spec[nrows + 1, -plot_width:])
        for i, (legend, data) in enumerate(db_error_rms.items()):
            ax.bar(i, data, color=colors[i])
            ax.text(i, 0, '{:.2f}'.format(data),
                    horizontalalignment='center', verticalalignment='bottom')
        ax.set_xticks(np.arange(i + 1))
        ax.set_xticklabels([legend.replace('_', ' ')
                           for legend in db_error_rms], rotation=20, horizontalalignment='right')
        ax.set_ylabel('RMS dB Error')
    fig.tight_layout()
    return return_data


def plot_asds(cpsd, freq=None, ax=None, subplots_kwargs={'sharex': True, 'sharey': True}, plot_kwargs={}):
    num_freq, num_channels, num_channels = cpsd.shape
    ncols = int(np.floor(np.sqrt(num_channels)))
    nrows = int(np.ceil(num_channels / ncols))
    if ax is None:
        f, ax = plt.subplots(nrows, ncols, **subplots_kwargs)
    diag = np.einsum('jii->ij', cpsd).real
    for asd, a in zip(diag, ax.flatten()):
        if freq is None:
            a.plot(asd, **plot_kwargs)
        else:
            a.plot(freq, asd, **plot_kwargs)
        a.set_yscale('log')
    return ax


def cpsd_to_time_history(cpsd_matrix, sample_rate, df, output_oversample=1):
    """Generates a time history realization from a CPSD matrix

    Parameters
    ----------
    cpsd_matrix : np.ndarray :
        A 3D complex np.ndarray representing a CPSD matrix where the first
        dimension is the frequency line and the second two dimensions are the
        rows and columns of the matrix at each frequency line.
    sample_rate: float :
        The sample rate of the controller in samples per second
    df : float :
        The frequency spacing of the cpsd matrix


    Returns
    -------
    output : np.ndarray :
        A numpy array containing the generated signals

    Notes
    -----
    Uses the process described in [1]_

    .. [1] R. Schultz and G. Nelson, "Input signal synthesis for open-loop
       multiple-input/multiple-output testing," Proceedings of the International
       Modal Analysis Conference, 2019.

    """
    # Compute SVD broadcasting over all frequency lines
    [U, S, Vh] = np.linalg.svd(cpsd_matrix, full_matrices=False)
    # Reform using the sqrt of the S matrix
    Lsvd = U * np.sqrt(S[:, np.newaxis, :]) @ Vh
    # Compute Random Process
    W = np.sqrt(0.5) * (np.random.randn(*
                                        cpsd_matrix.shape[:-1], 1) + 1j * np.random.randn(*cpsd_matrix.shape[:-1], 1))
    Xv = 1 / np.sqrt(df) * Lsvd @ W
    # Ensure that the signal is real by setting the nyquist and DC component to 0
    Xv[[0, -1], :, :] = 0
    # Compute the IFFT, using the real version makes it so you don't need negative frequencies
    zero_padding = np.zeros([((output_oversample - 1) * (Xv.shape[0] - 1))
                             ] + list(Xv.shape[1:]), dtype=Xv.dtype)
    xv = np.fft.irfft(np.concatenate((Xv, zero_padding), axis=0) /
                      np.sqrt(2), axis=0) * output_oversample * sample_rate
    output = xv[:, :, 0].T
    return output


def shaped_psd(frequency_spacing, bandwidth, num_channels=1, target_rms=None,
               min_frequency=0.0, max_frequency=None, breakpoint_frequencies=None,
               breakpoint_levels=None, breakpoint_interpolation='linear',
               squeeze=True):
    num_lines = int(bandwidth / frequency_spacing) + 1
    all_freqs = np.arange(num_lines) * frequency_spacing

    if breakpoint_frequencies is None or breakpoint_levels is None:
        cpsd = np.ones(all_freqs.shape)
    else:
        if breakpoint_interpolation in ['log', 'logarithmic']:
            cpsd = np.interp(np.log(all_freqs), np.log(breakpoint_frequencies),
                             breakpoint_levels, left=0, right=0)
        elif breakpoint_interpolation in ['lin', 'linear']:
            cpsd = np.interp(all_freqs, breakpoint_frequencies, breakpoint_levels)
        else:
            raise ValueError('Invalid Interpolation, should be "lin" or "log"')

    # Truncate to the minimum frequency
    cpsd[all_freqs < min_frequency] = 0
    if max_frequency is not None:
        cpsd[all_freqs > max_frequency] = 0

    if target_rms is not None:
        cpsd_rms = np.sqrt(np.sum(cpsd) * frequency_spacing)
        cpsd *= (target_rms / cpsd_rms)**2

    full_cpsd = np.zeros(cpsd.shape + (num_channels, num_channels), dtype='complex128')

    full_cpsd[:, np.arange(num_channels), np.arange(num_channels)] = cpsd[:, np.newaxis]

    if squeeze:
        full_cpsd = full_cpsd.squeeze()

    return full_cpsd

def nth_octave_freqs(freq,oct_order=1):
    """
    Get N-th octave band frequencies

    Parameters
    ----------
    freq : ndarray
        array of frequency values, either including all freqs or only min and max
    oct_order : int, optional
        octave type, 1/octave order. 3 represents 1/3 octave bands. default is 1

    Returns
    -------
    nominal_band_centers : ndarray
        rounded band center frequencies using ANSI standers. Used to report
        results, not to compute band limits
    band_lb : ndarray
        lower band frequencies in Hz
    band_ub : ndarray
        upper band frequencies in Hz
    band_centers : ndarray
        exact computed octave center band frequencies
        
    Notes
    -------
    Uses equations in ANSI S1.11-2014 "Electroacoustics – Octave-band and 
    Fractionaloctave-band Filters – Part 1: Specifications"
    
    Uses the min and max freq lines in the provided freq to determine the upper
    and lower bands
    
    Uses 1000 Hz as the reference frequency (as per the ANSI standard)
    
    Computes the 1/Nth band center frequencies, using different equations
    depending on the octave order. Orders with odd numbers (1, 1/3, etc.) 
    use ANSI eqn 2. Order with even numbers (1/6, 1/12, etc.) use eqn 3. 
    This causes any sub-bands to fit entirely within the full octave band
    
    Note: even-numbered sub-bands will not share center frequencies with 
    the full octave band, but will share the upper and lower limits.
    ANSI S1.11 Annex A gives instructions for rounding to make the nominal
    band center freqs. If left-most digit is 1-4, round to 3 significant
    digits. If left-most digit is 5-9, round to 2 significant digits. For
    example, 41.567 rounds to 41.6. 8785.2 rounds to 8800. 
    """
    
    # Process inputs
    if not isinstance(freq,np.ndarray):
        freq = np.array([freq]).flatten()
    
    # Setup
    oct_ratio = 10**0.3
    f_ref = 1000
    bands = np.arange(-200,201)
    
    # Compute exact center freqs and edges
    # equation is different if using odd octOrder (1, 3) vs. even octOrder
    # (6, 12). Note: bandNumbers = x and octOrder = b in the ANSI eqns
    pow = [(2*bands+1)/(2*oct_order), bands/oct_order]
    band_centers = f_ref * oct_ratio**pow[oct_order%2]
    band_lb = ( band_centers * oct_ratio**(-1/(2*oct_order)) ).round(12) # round slightly
    band_ub = ( band_centers * oct_ratio**(1/(2*oct_order)) ).round(12) # round slightly
    
    # ONLY KEEP BANDS IN THE FREQ RANGE OF INTEREST: 
    # lower limit of lowest band is above min freq
    # upper limit of highest band is below max freq
    # rationale: want to compute oct-ave using all the data, so include 
    # the band the includes the minimum frequency and the band that includes
    # the maximum frequency.
    # Note that this will result in an under-estimate of the first and last
    # band levels since some of the band is empty.
    inds = (max([freq.min(),10]) < band_ub) * (freq.max() > band_lb)
    band_centers,band_lb,band_ub = [band_centers[inds],band_lb[inds],band_ub[inds]]
    

    # GET THE NOMINAL BAND CENTER FREQUENCIES BY ROUNDING
    # These are just "nice" numbers to make it easier to identify the bands 
    # edges are still calculated from the exact center freqs
    # method used here is based on:
    # https://www.mathworks.com/matlabcentral/fileexchange/26212-round-with-significant-digits
    # freqBandCenters = [ 41.567, 8785.2 ] should give: [ 41.6, 8800]
    pow_10 = 10** np.floor(np.log10(band_centers))
    left_digit =np.floor(band_centers/pow_10)
    sig_digits = 3*(left_digit<=4) + 2*(left_digit>4)
    tmp = 10 ** np.floor(np.abs(np.log10(band_centers)) - sig_digits + 1)
    nominal_band_centers = np.round(band_centers/tmp)*tmp
    
    # Replace calculated nominal band centers with ANSI nominal band centers if
    # doing full or 1/3rd octaves:
    # ANSI only gives freqs from 25 to 20000, so don't correct outside that range
    # find nearest ANSI nominal center, then verify it is between the band
    # limits. If the ANSI nominal is between the band limits, replace the 
    # calculated nominal with the ANSI value.
    if oct_order == 1:
        ansi_nominal = np.array([31.5 , 63 , 125 , 250 , 500 , 1000 , 2000 , 4000 , 8000 , 16000])
    elif oct_order == 3:
        ansi_nominal = np.array([25 , 31.5 , 40 , 50 , 63 , 80 , 100 , 125 , 160 , 200 ,
                                   250 , 315 , 400 , 500 , 630 , 800 , 1000 , 1250 , 1600 , 2000 ,
                                   2500 , 3150 , 4000 , 5000 , 8000 , 10000 , 12500 , 16000 , 20000])
    
    if oct_order==1 or oct_order==3:
        dists = abs(ansi_nominal - band_centers[:,np.newaxis]) # dist from center freq to nominal
        contained = (ansi_nominal < band_ub[:,np.newaxis]) * (ansi_nominal > band_lb[:,np.newaxis]) # bands contain nominals
        contained_dists = contained*dists + 1e20*(~contained) # product is infty if not contained
        
        assign_inds = contained_dists.argmin(1) # nominal inds to assign    
        assign_bools = contained_dists.min(1) < 1e20 # center freqs inds to assign
        
        nominal_band_centers[assign_bools] = ansi_nominal[assign_inds[assign_bools]]
        
    return nominal_band_centers, band_lb, band_ub, band_centers