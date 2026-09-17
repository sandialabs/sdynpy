# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 09:23:59 2023

@author: dprohe
"""

import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from scipy.signal import oaconvolve, windows
from scipy.optimize import minimize, NonlinearConstraint, nnls
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


def srs(signal, dt, frequencies=None, damping=0.05, spectrum_type=9,
        b_filter_weights=None, a_filter_weights=None):
    """
    Computes shock response spectrum of a signal

    Computes the shock response spectrum using a ramp invariant digital filter
    simulation of the single degree-of-freedom system.

    Parameters
    ----------
    signal : np.ndarray
        A shape (..., num_samples) ndarray containing the sampled signals to
        compute SRSs from.
    dt : float
        The time between samples (1/sample rate)
    frequencies : np.ndarray
        An iterable of frequency lines at which the shock response spectrum
        will be calculated.  If not specified, it will be computed from the
        sample rate/10000 to the sample rate/4 over 50 lines.
    damping : float
        Fraction of critical damping to use in the SRS calculation (e.g. you
        should specify 0.03 to represent 3%, not 3). The default is 0.03.
    spectrum_type : int
        The type of spectrum desired:
        If `spectrum_type` > 0 (pos) then the SRS will be a base
        acceleration-absolute acceleration model
        If `spectrum_type` < 0 (neg) then the SRS will be a base acceleration-relative
        displacement model (expressed in equivalent static acceleration units).
        If abs(`spectrum_type`) is:
            1--positive primary,  2--negative primary,  3--absolute maximum primary
            4--positive residual, 5--negative residual, 6--absolute maximum residual
            7--largest of 1&4, maximum positive, 8--largest of 2&5, maximum negative
            9 -- maximax, the largest absolute value of 1-8
           10 -- returns a matrix s(9,length(fn)) with all the types 1-9.
    b_filter_weights : np.ndarray, optional
        Optional filter weights with shape (frequencies.size,3).
        The default is to automatically compute filter weights using
        `sdof_ramp_invariant_filter_weights`.
    a_filter_weights : np.ndarray, optional
        Optional filter weights with shape (frequencies.size,3).
        The default is to automatically compute filter weights using
        `sdof_ramp_invariant_filter_weights`.

    Returns
    -------
    srs : np.ndarray
        The shock response spectrum at the frequencies specified.  If
        `spectrum_type` == 10 or -10, then this will have shape
        (... x 9 x len(frequencies)) where each row is a different type of SRS
    frequencies : np.ndarray
        The frequencies at which the SRSs were computed.
    """
    # Compute default parameters
    sample_rate = 1/dt
    if frequencies is None:
        frequencies = np.logspace(np.log10(sample_rate/1e4),
                                  np.log10(sample_rate/4),
                                  50)
    else:
        frequencies = np.array(frequencies).flatten()
    if frequencies.size == 0:
        raise ValueError('Frequencies must have nonzero size')
    signal = np.array(signal)
    num_samples = signal.shape[-1]

    # Compute bad parameters
    abs_type = abs(spectrum_type)
    if abs_type > 10:
        raise ValueError("`spectrum_type` must be in the range [-10,10]")
    if abs_type == 0:
        raise ValueError('`spectrum_type` must not be 0')
    if sample_rate <= 0:
        raise ValueError('`dt` must be positive')
    if frequencies.min() < 0:
        raise ValueError('`frequencies` must all be positive')
    if damping < 0:
        raise ValueError('`damping` must be positive')
    if damping == 0:
        print('Warning: Damping is Zero in SRS calculation!')

    # Get the filter coefficients if not specified
    if b_filter_weights is None or a_filter_weights is None:
        b_filter_weights, a_filter_weights = sdof_ramp_invariant_filter_weights(
            frequencies, sample_rate, damping, spectrum_type)
    else:
        b_filter_weights = np.atleast_2d(np.array(b_filter_weights))
        a_filter_weights = np.atleast_2d(np.array(a_filter_weights))

    if b_filter_weights.shape != (frequencies.size, 3):
        raise ValueError('`b_filter_weights` and `a_filter_weights` must have shape (frequencies.size,3)')

    srs_output = np.zeros(signal.shape[:-1]
                          + ((9,) if np.abs(spectrum_type) == 10 else ())
                          + (frequencies.size,))
    for i_freq, (frequency, b, a) in enumerate(zip(
            frequencies, b_filter_weights, a_filter_weights)):
        # Append enough zeros to get 1/100 of the residual response
        zeros_to_append = int(np.max([2, np.ceil(0.01*sample_rate/frequency)]))
        signals_extended = np.concatenate((
            signal, np.zeros(signal.shape[:-1] + (zeros_to_append,))), axis=-1)

        # Filter the signal
        response = sdof_filter(b, a, signals_extended)

        # Now compute the SRS parameters
        # Primary response
        primary_maximum = np.array(np.max(response[..., :num_samples], axis=-1))
        primary_maximum[primary_maximum < 0] = 0
        primary_minimum = np.array(np.abs(np.min(response[..., :num_samples], axis=-1)))
        primary_minimum[primary_minimum < 0] = 0
        primary_abs = np.array(np.max([primary_maximum, primary_minimum], axis=0))

        # Now compute the residual response. we need time steps and
        # responses at two points
        residual_times = [0, (zeros_to_append-1)/sample_rate]
        residual_responses = np.concatenate([response[..., num_samples, np.newaxis],
                                             response[..., num_samples+zeros_to_append-1, np.newaxis]],
                                            axis=-1)

        # Compute the peak response after the structure has stopped responding
        peak_residual_responses = sdof_free_decay_peak_response(
            residual_responses, residual_times, frequency, damping)

        # Pull out the residual peaks
        residual_maximum = np.array(np.max(peak_residual_responses, axis=-1))
        residual_maximum[residual_maximum < 0] = 0
        residual_minimum = np.array(np.abs(np.min(peak_residual_responses, axis=-1)))
        residual_minimum[residual_minimum < 0] = 0
        residual_abs = np.array(np.max([residual_maximum, residual_minimum], axis=0))

        all_srs_this_frequency_line = np.array([
            primary_maximum, primary_minimum, primary_abs,
            residual_maximum, residual_minimum, residual_abs,
            np.max([primary_maximum, residual_maximum], axis=0),
            np.max([primary_minimum, residual_minimum], axis=0),
            np.max([primary_abs, residual_abs], axis=0)])

        if abs_type < 10:
            srs_output[..., i_freq] = all_srs_this_frequency_line[abs_type-1]
        else:
            srs_output[..., i_freq] = np.moveaxis(all_srs_this_frequency_line, 0, -1)
    return srs_output, frequencies


def sdof_ramp_invariant_filter_weights(frequencies, sample_rate, damping,
                                       spectrum_type):
    """
    Computes filter weights for SDOF resonators using a ramp-invariant filter.

    The weights are used in conjunction with the function
    `sdof_filter` and `srs` to calculate the shock response
    spectrum of an acceleration time history using a ramp
    invarient simulation.

    Parameters
    ----------
    frequencies : np.ndarray
        The frequencies at which filters should be computed
    sample_rate : float
        The sample rate of the measurement
    damping : float
        The fraction of critical damping for the system (e.g. 0.03, not 3 for 3%)
    spectrum_type : int
        See `spectrum_type` argument for compute_srs

    Returns
    -------
    b : np.ndarray
        Filter coefficients with shape (frequencies.shape,3)
    a : np.ndarray
        Filter coefficients with shape (frequencies.shape,3)
    """
    # Make sure it's a numpy array
    frequencies = np.array(frequencies)
    # Preallocate arrays
    a = np.ones(frequencies.shape+(3,))
    b = np.ones(frequencies.shape+(3,))

    normalized_frequencies = 2*np.pi*frequencies/sample_rate

    small_freq_indices = normalized_frequencies < 0.001

    # Small frequencies
    small_freqs = normalized_frequencies[small_freq_indices]

    # 2*z*w + w*w*(1-2*z*z) where z is damping and w is normalized frequencies
    a[small_freq_indices, 1] = (
          2*damping*small_freqs
          + small_freqs**2*(1-2*damping**2)
          )
    # -2*z*w + 2*z*z*w*w where z is damping and w is normalized frequencies
    a[small_freq_indices, 2] = (
          -2*damping*small_freqs
          + 2*damping**2*small_freqs**2
          )

    if spectrum_type > 0:
        # Absolute Acceleration Model
        # z*w + (w*w)*( (1/6) - 2*z*z/3 );
        b[small_freq_indices, 0] = (
              damping*small_freqs
              + small_freqs**2
              * (1/6 - 2*damping**2/3)
              )
        # 2*w*w*(1-z*z)/3
        b[small_freq_indices, 1] = (
            2*small_freqs**2*(1-damping**2)/3
            )
        # -z*w + w*w*( (1/6) - 4*z*z/3 );
        b[small_freq_indices, 2] = (
            - damping*small_freqs
            + small_freqs**2
            * (1/6 - 4*damping**2/3)
            )
    else:
        # Relative Displacement Model
        # -w*w/6;
        b[small_freq_indices, 0] = -small_freqs**2/6
        # -2*w*w/3;
        b[small_freq_indices, 1] = -2*small_freqs**2/3
        # -w*w/6;
        b[small_freq_indices, 2] = -small_freqs**2/6

    # Large frequencies
    large_freqs = normalized_frequencies[~small_freq_indices]

    # Define some helper variables
    sq = np.sqrt(1-damping**2)
    e = np.exp(-damping*large_freqs)
    wd = large_freqs*sq
    sp = e*np.sin(wd)
    fact = (2*damping**2 - 1)*sp/sq
    c = e*np.cos(wd)

    a[~small_freq_indices, 1] = 2*(1-c)
    a[~small_freq_indices, 2] = -1 + e**2

    if spectrum_type > 0:
        # Absolute Acceleration Model
        spwd = sp/wd
        b[~small_freq_indices, 0] = 1-spwd
        b[~small_freq_indices, 1] = 2*(spwd - c)
        b[~small_freq_indices, 2] = e**2 - spwd
    else:
        # Relative Displacement Model
        b[~small_freq_indices, 0] = -(2*damping*(c-1) + fact + large_freqs)/large_freqs
        b[~small_freq_indices, 1] = -(-2*c*large_freqs + 2*damping*(1-e**2) - 2*fact)/large_freqs
        b[~small_freq_indices, 2] = -(e**2*(large_freqs+2*damping) - 2*damping*c + fact)/large_freqs

    return b, a


def sdof_filter(b, a, signal, zi=None):
    """
    Applies a filter to simulate a single degree of freedom system

    Parameters
    ----------
    b : np.ndarray
        Size 3 array representing filter coefficients used by scipy.signal.lfilter
    a : np.array
        Size 3 array representing filter coefficients used by scipy.signal.lfilter
    signal : np.ndarray
        The signals that are to be filtered
    zi : np.ndarray, optional
        Optional initial state for the filters, having length
        `max(len(a), len(b)) - 1`.  If not specified, zero initial conditions
        are assumed.

    Returns
    -------
    filtered_signal : np.ndarray
        The filtered signal

    """
    aa = np.array([a[0], a[1]-2, a[2] + 1])
    return lfilter(b, aa, signal, axis=-1, zi=zi)


def sdof_free_decay_peak_response(responses, times_at_responses, frequency, damping):
    """
    Calculates peak response of a freely-decaying sdof system.

    The residual response is the peak response of the sdof system
    as it decays after the input has ended, i.e., the input is zero.
    The first two peaks in the decaying response will be the largest.
    One peak will be negative and one positive.  We don't know before
    the calculation if the first and largest peak in amplitude will be
    positive or negative.

    Parameters
    ----------
    responses : np.array
        A (...,2) shape array giving the response at two time steps
    times_at_responses : np.array
        A length-2 array giving the time values that the responses occur at
    frequency : float
        The frequency at which the computation is performed
    damping : float
        The fraction of critical damping for the system (e.g. 0.03, not 3 for 3%)

    Returns
    -------
    response_peaks : np.ndarray
        A shape (...,2) array giving the amplitude at the two peaks after the
        response has decayed

    Notes
    -----
    The response has the form
    a(t) = exp(-zeta wn t)[z[0]sin(wd t) + z[1]cos(wd t)].
    If I know a(t) at two values of time, t, I can calculate the constants
    z[0] and z[1].  The general form of the response can then be solved for
    the maximum by finding the time of the first maximum by setting the
    derivative to zero.  Then substituting the time of the maximum response
    back into the general equation to find the maximum response.
    The second peak will occur half a cycle later.

    """
    responses = np.array(responses)
    times_at_responses = np.array(times_at_responses)
    # Set up some initial helper variables
    fd = frequency*np.sqrt(1-damping*damping)
    wd = 2*np.pi*fd
    wn = 2*np.pi*frequency
    wdt = wd*times_at_responses
    e = np.exp(-wn*damping*times_at_responses)
    sd = np.sin(wdt)
    cd = np.cos(wdt)
    c = e[:, np.newaxis]*np.array((sd, cd)).T
    z = np.linalg.solve(c, responses[..., np.newaxis])[..., 0]

    # the response has the form
    # a(t) = exp(-zeta wn t)[z[0]sin(wd t) + z[1]cos(wd t)]
    # the first maximum response will occur at the time tmax
    tmax = np.array((1/wd) * np.arctan((wd*z[..., 0] - damping*wn*z[..., 1])/(wd*z[..., 1] - damping*wn*z[..., 0])))
    tmax[tmax < 0] = tmax[tmax < 0] + np.pi/wd
    tmax[tmax > np.pi/wd] = tmax[tmax > np.pi/wd] - np.pi/wd

    # We can now plug it into the equation to find the maximum
    response_peaks = []
    response_peaks.append(np.exp(-damping*wn*tmax)*(z[..., 0]*np.sin(wd*tmax) + z[..., 1]*np.cos(wd*tmax)))
    tmin = tmax + np.pi/wd
    response_peaks.append(np.exp(-damping*wn*tmin)*(z[..., 0]*np.sin(wd*tmin) + z[..., 1]*np.cos(wd*tmin)))
    return np.moveaxis(response_peaks, 0, -1)


def octspace(low, high, points_per_octave):
    """
    Constructs octave spacing between low and high values

    Parameters
    ----------
    low : float
        Starting value for the spacing
    high : float
        Upper value for the spacing
    points_per_octave : int
        Number of points per octave

    Returns
    -------
    octave_points : np.ndarray
        Octave-spaced points
    """
    num_octaves = np.log2(high/low)
    num_steps = np.ceil(num_octaves*points_per_octave)
    point_indices = np.arange(num_steps+1)
    log_points = np.log2(low) + num_octaves/num_steps*point_indices
    points = 2**log_points
    return points


def sum_decayed_sines(sample_rate, block_size,
                      sine_frequencies=None, sine_tone_range=None, sine_tone_per_octave=None,
                      sine_amplitudes=None, sine_decays=None, sine_delays=None,
                      required_srs=None, srs_breakpoints=None,
                      srs_damping=0.05, srs_type=9,
                      compensation_frequency=None, compensation_decay=0.95,
                      # Paramters for the iteration
                      number_of_iterations=3, convergence=0.8,
                      error_tolerance=0.05,
                      tau=None, num_time_constants=None, decay_resolution=None,
                      scale_factor=1.02,
                      acceleration_factor=1.0,
                      plot_results=False, srs_frequencies=None,
                      ignore_compensation_pulse = False,
                      verbose=False
                      ):
    """
    Generate a Sum of Decayed Sines signal given an SRS.

    Note that there are many approaches to do this, with many optional arguments
    so please read the documentation carefully to understand which arguments
    must be passed to the function.

    Parameters
    ----------
    sample_rate : float
        The sample rate of the generated signal.
    block_size : int
        The number of samples in the generated signal.
    sine_frequencies : np.ndarray, optional
        The frequencies of the sine tones.  If this argument is not specified,
        then the `sine_tone_range` argument must be specified.
    sine_tone_range : np.ndarray, optional
        A length-2 array containing the minimum and maximum sine tone to
        generate.  If this is argument is not specified, then the
        `sine_frequencies` argument must be specified instead.
    sine_tone_per_octave : int, optional
        The number of sine tones per octave. If not specified along with
        `sine_tone_range`, then a default value of 4 will be used if the
        `srs_damping` is >= 0.05.  Otherwise, the formula of
        `sine_tone_per_octave = 9 - srs_damping*100` will be used.
    sine_amplitudes : np.ndarray, optional
        The initial amplitude of the sine tones used in the optimization.  If
        not specified, they will be set to the value of the SRS at each frequency
        divided by the quality factor of the SRS.
    sine_decays : np.ndarray, optional
        An array of decay value time constants (often represented by variable
        tau).  Tau is the time for the amplitude of motion to decay 63% defined
        by the equation `1/(2*np.pi*freq*zeta)` where `freq` is the frequency
        of the sine tone and `zeta` is the fraction of critical damping.
        If not specified, then either the `tau` or `num_time_constants`
        arguments must be specified instead.
    sine_delays : np.ndarray, optional
        An array of delay values for the sine components. If not specified,
        all tones will have zero delay.
    required_srs : np.ndarray, optional
        The SRS to match defined at each of the `sine_frequencies`.  If this
        argument is not passed, then the `srs_breakpoints` argument must be
        passed instead.  The default is None.
    srs_breakpoints : np.ndarray, optional
        A numpy array with shape `(n,2)` where the first column is the
        frequencies at which each breakpoint occurs, and the second column is
        the value of the SRS at that breakpoint.  SRS values at the
        `sine_frequencies` will be interpolated in a log-log sense from this
        breakpoint array.  If this argument is not specified, then the
        `required_srs` must be passed instead.
    srs_damping : float, optional
        Fraction of critical damping to use in the SRS calculation (e.g. you
        should specify 0.03 to represent 3%, not 3). The default is 0.03.
    srs_type : int
        The type of spectrum desired:
        If `srs_type` > 0 (pos) then the SRS will be a base
        acceleration-absolute acceleration model
        If `srs_type` < 0 (neg) then the SRS will be a base acceleration-relative
        displacement model (expressed in equivalent static acceleration units).
        If abs(`srs_type`) is:
            1--positive primary,  2--negative primary,  3--absolute maximum primary
            4--positive residual, 5--negative residual, 6--absolute maximum residual
            7--largest of 1&4, maximum positive, 8--largest of 2&5, maximum negative
            9 -- maximax, the largest absolute value of 1-8
           10 -- returns a matrix s(9,length(fn)) with all the types 1-9.
    compensation_frequency : float
        The frequency of the compensation pulse.  If not specified, it will be
        set to 1/3 of the lowest sine tone
    compensation_decay : float
        The decay value for the compensation pulse.  If not specified, it will
        be set to 0.95.
    number_of_iterations : int, optional
        The number of iterations to perform. At least two iterations should be
        performed.  3 iterations is preferred, and will be used if this argument
        is not specified.
    convergence : float, optional
        The fraction of the error corrected each iteration. The default is 0.8.
    error_tolerance : float, optional
        Allowable relative error in the SRS. The default is 0.05.
    tau : float, optional
        If a floating point number is passed, then this will be used for the
        `sine_decay` values.  Alternatively, a dictionary can be passed with
        the keys containing a length-2 tuple specifying the minimum and maximum
        frequency range, and the value specifying the value of `tau` within that
        frequency range.  If this latter approach is used, all `sine_frequencies`
        must be contained within a frequency range. If this argument is not
        specified, then either `sine_decays` or `num_time_constants` must be
        specified instead.
    num_time_constants : int, optional
        If an integer is passed, then this will be used to set the `sine_decay`
        values by ensuring the specified number of time constants occur in the
        `block_size`.  Alternatively, a dictionary can be passed with the keys
        containing a length-2 tuple specifying the minimum and maximum
        frequency range, and the value specifying the value of
        `num_time_constants` over that frequency range. If this latter approach
        is used, all `sine_frequencies` must be contained within a frequency
        range. If this argument is not specified, then either `sine_decays` or
        `tau` must be specified instead.
    decay_resolution : float, optional
        A scalar identifying the resolution of the fractional decay rate
        (often known by the variable `zeta`).  The decay parameters will be
        rounded to this value.  The default is to not round.
    scale_factor : float, optional
        A scaling applied to the sine tone amplitudes so the achieved SRS better
        fits the specified SRS, rather than just touching it. The default is 1.02.
    acceleration_factor : float, optional
        Optional scale factor to convert acceleration into velocity and
        displacement.  For example, if sine amplitudes are in G and displacement
        is desired in inches, the acceleration factor should be set to 386.089.
        If sine amplitudes are in G and displacement is desired in meters, the
        acceleration factor should be set to 9.80665.  The default is 1, which
        assumes consistent units (e.g. acceleration in m/s^2, velocity in m/s,
        displacement in m).
    plot_results : bool, optional
        If True, a figure will be plotted showing the acceleration, velocity,
        and displacement signals, as well as the desired and achieved SRS.
    srs_frequencies : np.ndarray, optional
        If specified, these frequencies will be used to compute the SRS that
        will be plotted when the `plot_results` value is `True`.
    ignore_compensation_pulse : bool, optional
        If True, the compensation pulse will be ignored.
    verbose : True, optional
        If True, additional diagnostics will be printed to the console.

    Returns
    -------
    acceleration_signal : ndarray
        The acceleration signal that satisfies the SRS.
    velocity_signal : ndarray
        The velocity of the acceleration signal.
    displacement_signal : ndarray
        The displacement of the acceleration signal.
    sine_frequencies : ndarray
        An array of frequencies for each sine tone, including the compensation
        pulse
    sine_amplitudes : ndarray
        An array of amplitudes for each sine tone, including the compensation
        pulse
    sine_decays : ndarray
        An array of decay values for each sine tone, including the compensation
        pulse
    sine_delays : ndarray
        An array of delay values for each sine tone, including the compensation
        pulse
    fig : matplotlib.Figure
        A reference to the plotted figure.  Only returned if `plot_results` is
        `True`.
    ax : matplotlib.Axes
        A reference to the plotted axes.  Only returned if `plot_results` is
        `True`.

    """
    # Handle the sine tone frequencies
    if sine_frequencies is None and sine_tone_range is None:
        raise ValueError('Either `sine_frequencies` or `sine_tone_range` must be specified')
    if sine_frequencies is not None and sine_tone_range is not None:
        raise ValueError('`sine_frequencies` can not be specified simultaneously with `sine_tone_range`')
    if sine_frequencies is None:
        # Create sine tones
        if sine_tone_per_octave is None:
            sine_tone_per_octave = int(np.floor(9-srs_damping*100))
        sine_frequencies = octspace(sine_tone_range[0], sine_tone_range[1],
                                    sine_tone_per_octave)
    # Now set up the SRS
    if required_srs is None and srs_breakpoints is None:
        raise ValueError('Either `required_srs` or `srs_breakpoints` must be specified')
    if required_srs is not None and srs_breakpoints is not None:
        raise ValueError('`required_srs` can not be specified simultaneously with `srs_breakpoints`')
    if required_srs is None:
        required_srs = loginterp(sine_frequencies,
                                 srs_breakpoints[:, 0],
                                 srs_breakpoints[:, 1])
    if sine_amplitudes is None:
        srs_amplitudes = required_srs.copy()
        srs_amplitudes[np.arange(srs_amplitudes.size) % 2 == 0] *= -1
        quality = 1/(2*srs_damping)
        srs_amplitudes /= quality
        sine_amplitudes = srs_amplitudes

    if sine_delays is None:
        sine_delays = np.zeros(sine_frequencies.size)

    if compensation_frequency is None:
        compensation_frequency = np.min(sine_frequencies)/3

    # Set up decay terms
    decay_terms_specified = 0
    if sine_decays is not None:
        decay_terms_specified += 1
    if tau is not None:
        decay_terms_specified += 1
    if num_time_constants is not None:
        decay_terms_specified += 1
    if decay_terms_specified == 0:
        raise ValueError('One of `sine_decays`, `tau`, or `num_time_constants` must be specified')
    if decay_terms_specified > 1:
        raise ValueError('Only one of `sine_decays`, `tau`, or `num_time_constants` can be specified')

    # Now check and see which is defined
    if num_time_constants is not None:
        period = block_size / sample_rate
        if isinstance(num_time_constants, dict):
            tau = {}
            for freq_range, num_time_constant in num_time_constants.items():
                tau[freq_range] = period / num_time_constant
        else:
            tau = period/num_time_constants
    if tau is not None:
        sine_decays = []
        for freq in sine_frequencies:
            if isinstance(tau, dict):
                this_decay = None
                for freq_range, this_tau in tau.items():
                    if freq_range[0] <= freq <= freq_range[1]:
                        this_decay = 1/(2*np.pi*freq*this_tau)
                        break
                if this_decay is None:
                    raise ValueError('No frequency range matching frequency {:} was found in the specified decay parameters.'.format(freq))
                sine_decays.append(this_decay)
            else:
                sine_decays.append(1/(2*np.pi*freq*tau))
        sine_decays = np.array(sine_decays)
    # Otherwise we just keep the specified sine_decays

    # Now handle the minimum resolution on the decay
    if decay_resolution is not None:
        sine_decays = decay_resolution*np.round(sine_decays/decay_resolution)

    if compensation_frequency is None:
        compensation_frequency = np.min(sine_frequencies)/3

    # Now we can actually run the generation process
    (acceleration_signal, used_frequencies, used_amplitudes, used_decays,
     used_delays, used_comp_frequency, used_comp_amplitude, used_comp_decay,
     used_comp_delay) = _sum_decayed_sines(
         sine_frequencies, sine_amplitudes, sine_decays, sine_delays,
         scale_factor*required_srs, compensation_frequency, compensation_decay,
         sample_rate, block_size, srs_damping, srs_type, number_of_iterations,
         convergence, error_tolerance, ignore_compensation_pulse, verbose)

    all_frequencies = np.concatenate((used_frequencies,
                                      [used_comp_frequency]))
    all_amplitudes = np.concatenate((used_amplitudes,
                                     [used_comp_amplitude]))
    all_decays = np.concatenate((used_decays,
                                 [used_comp_decay]))
    all_delays = np.concatenate((used_delays,
                                 [used_comp_delay]))
    # Now compute displacement and velocity
    velocity_signal, displacement_signal = sum_decayed_sines_displacement_velocity(
        all_frequencies, all_amplitudes, all_decays, all_delays, sample_rate,
        block_size, acceleration_factor)

    return_vals = (acceleration_signal, velocity_signal, displacement_signal,
                   all_frequencies, all_amplitudes, all_decays, all_delays)

    # Plot the results
    if plot_results:
        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        times = np.arange(block_size)/sample_rate
        ax[0, 0].plot(times, acceleration_signal)
        ax[0, 0].set_ylabel('Acceleration')
        ax[0, 0].set_xlabel('Time (s)')
        ax[0, 1].plot(times, velocity_signal)
        ax[0, 1].set_ylabel('Velocity')
        ax[0, 1].set_xlabel('Time (s)')
        ax[1, 0].plot(times, displacement_signal)
        ax[1, 0].set_ylabel('Displacement')
        ax[1, 0].set_xlabel('Time (s)')
        # Compute SRS
        if srs_frequencies is None:
            srs_frequencies = sine_frequencies
        this_srs, this_frequencies = srs(
            acceleration_signal, 1/sample_rate, srs_frequencies,
            srs_damping, srs_type)
        if srs_breakpoints is None:
            srs_abscissa = sine_frequencies
            srs_ordinate = required_srs
        else:
            srs_abscissa, srs_ordinate = srs_breakpoints.T
        ax[1, 1].plot(srs_abscissa, srs_ordinate, 'k--')
        ax[1, 1].plot(this_frequencies, this_srs)
        ax[1, 1].set_ylabel('SRS ({:0.2f}% damping)'.format(srs_damping*100))
        ax[1, 1].set_xlabel('Frequency (Hz)')
        ax[1, 1].set_yscale('log')
        ax[1, 1].set_xscale('log')
        fig.tight_layout()
        return_vals += (fig, ax)
        ax[1, 1].legend(('Reference', 'Decayed Sine'))
    return return_vals


def _sum_decayed_sines(sine_frequencies, sine_amplitudes,
                       sine_decays, sine_delays,
                       required_srs,
                       compensation_frequency, compensation_decay,
                       sample_rate, block_size,
                       srs_damping, srs_type,
                       number_of_iterations=3, convergence=0.8,
                       error_tolerance=0.05, ignore_compensation_pulse = False,
                       verbose=False):
    """
    Optimizes the amplitudes of sums of decayed sines

    Parameters
    ----------
    sine_frequencies : ndarray
        An array of frequencies for each sine tone
    sine_amplitudes : ndarray
        An array of amplitudes for each sine tone
    sine_decays : ndarray
        An array of decay values for each sine tone
    sine_delays : ndarray
        An array of delay values for each sine tone
    required_srs : ndarray
        An array of SRS values for each frequency in `sine_frequencies`
    compensation_frequency : float
        The frequency of the compensation pulse
    compensation_decay : float
        The decay value for the compensation pulse
    sample_rate : float
        The sample rate of the signal
    block_size : int
        The number of samples in the signal
    srs_damping : float
        Fraction of critical damping to use in the SRS calculation (e.g. you
        should specify 0.03 to represent 3%, not 3). The default is 0.03.
    srs_type : int
        The type of spectrum desired:
        If `spectrum_type` > 0 (pos) then the SRS will be a base
        acceleration-absolute acceleration model
        If `spectrum_type` < 0 (neg) then the SRS will be a base acceleration-relative
        displacement model (expressed in equivalent static acceleration units).
        If abs(`spectrum_type`) is:
            1--positive primary,  2--negative primary,  3--absolute maximum primary
            4--positive residual, 5--negative residual, 6--absolute maximum residual
            7--largest of 1&4, maximum positive, 8--largest of 2&5, maximum negative
            9 -- maximax, the largest absolute value of 1-8
           10 -- returns a matrix s(9,length(fn)) with all the types 1-9.
    number_of_iterations : int
        The number of iterations that will be performed.  The default is 3.
    convergence : float, optional
        The fraction of the error corrected each iteration. The default is 0.8.
    error_tolerance : float, optional
        Allowable relative error in the SRS. The default is 0.05.
    ignore_compensation_pulse : bool, optional
        If True, ignores the compensation pulse.  Default is false.
    verbose : bool, optional
        If True, information on the interations will be provided. The default
        is False.

    Returns
    -------
    this_signal : np.ndarray
        A numpy array containing the generated signal
    sine_frequencies : ndarray
        An array of frequencies for each sine tone
    sine_amplitudes : ndarray
        An array of amplitudes for each sine tone
    sine_decays : ndarray
        An array of decay values for each sine tone
    sine_delays : ndarray
        An array of delay values for each sine tone
    this_compensation_frequency : float
        The frequency of the compensation term.
    this_compensation_amplitude : float
        The amplitude of the compensation term.
    this_compensation_decay : float
        The decay constant of the compensation term
    this_compensation_delay : float
        The delay constant of the compenstation term
    """
    for i in range(number_of_iterations):
        if verbose:
            print('Starting Iteration {:}'.format(i+1))
        # Compute updated sine amplitudes and compensation terms
        this_sine_amplitudes, this_compensation_amplitude, this_compensation_delay = (
            _sum_decayed_sines_single_iteration(
                sine_frequencies, sine_amplitudes, sine_decays, sine_delays,
                required_srs, compensation_frequency, compensation_decay,
                sample_rate, block_size, srs_damping, srs_type,
                number_of_iterations=10, convergence=convergence,
                error_tolerance=error_tolerance, ignore_compensation_pulse=ignore_compensation_pulse,
                verbose=verbose))
        # get the SRS error at each frequency term by first computing the signal
        (this_signal, this_compensation_frequency, this_compensation_amplitude,
         this_compensation_decay, this_compensation_delay) = sum_decayed_sines_reconstruction_with_compensation(
            sine_frequencies, this_sine_amplitudes, sine_decays, sine_delays,
            compensation_frequency, compensation_decay, sample_rate, block_size, ignore_compensation_pulse=ignore_compensation_pulse)
        # Then computing the SRS of the signal
        this_srs = srs(this_signal, 1/sample_rate, sine_frequencies, srs_damping,
                       srs_type)[0]
        srs_error = (this_srs - required_srs)/required_srs
        sine_amplitudes = this_sine_amplitudes
    if verbose:
        print('Sine Table after Iterations:')
        print('{:>8s}, {:>10s}, {:>10s}, {:>10s},  {:>10s}, {:>10s}'.format(
            'Sine', 'Frequency', 'Amplitude', 'SRS Val', 'SRS Req', 'Error'))
        for i, (freq, amp, srs_val, srs_req, srs_err) in enumerate(zip(
                sine_frequencies, sine_amplitudes, this_srs, required_srs,
                srs_error)):
            print('{:>8s}, {:>10.4f}, {:>10.4f}, {:>10.4f}, {:>10.4f}, {:>10.4f}'.format(
                str(i), freq, amp, srs_val, srs_req, srs_err))
        print('Compensating Pulse:')
        print('{:>10s}, {:>10s}, {:>10s}, {:>10s}'.format(
            'Frequency', 'Amplitude', 'Decay', 'Delay'))
        print('{:>10.4f}, {:>10.4f}, {:>10.4f}, {:>10.4f}'.format(
            this_compensation_frequency, this_compensation_amplitude,
            this_compensation_decay, this_compensation_delay))
    return (this_signal, sine_frequencies, sine_amplitudes, sine_decays,
            sine_delays, this_compensation_frequency, this_compensation_amplitude,
            this_compensation_decay, this_compensation_delay)


def _sum_decayed_sines_single_iteration(sine_frequencies, sine_amplitudes,
                                        sine_decays, sine_delays,
                                        required_srs, compensation_frequency,
                                        compensation_decay,
                                        sample_rate, block_size,
                                        damping_srs, srs_type,
                                        number_of_iterations=10, convergence=0.8,
                                        error_tolerance=0.05,
                                        ignore_compensation_pulse=False,
                                        verbose=False):
    """
    Iterates on amplitudes of decayed sine waves to match a prescribed SRS

    Parameters
    ----------
    sine_frequencies : ndarray
        An array of frequencies for each sine tone
    sine_amplitudes : ndarray
        An array of amplitudes for each sine tone
    sine_decays : ndarray
        An array of decay values for each sine tone
    sine_delays : ndarray
        An array of delay values for each sine tone
    required_srs : ndarray
        An array of SRS values for each frequency in `sine_frequencies`
    compensation_frequency : float
        The frequency of the compensation pulse
    compensation_decay : float
        The decay value for the compensation pulse
    sample_rate : float
        The sample rate of the signal
    block_size : int
        The number of samples in the signal
    damping_srs : float
        Fraction of critical damping to use in the SRS calculation (e.g. you
        should specify 0.03 to represent 3%, not 3). The default is 0.03.
    srs_type : int
        The type of spectrum desired:
        If `spectrum_type` > 0 (pos) then the SRS will be a base
        acceleration-absolute acceleration model
        If `spectrum_type` < 0 (neg) then the SRS will be a base acceleration-relative
        displacement model (expressed in equivalent static acceleration units).
        If abs(`spectrum_type`) is:
            1--positive primary,  2--negative primary,  3--absolute maximum primary
            4--positive residual, 5--negative residual, 6--absolute maximum residual
            7--largest of 1&4, maximum positive, 8--largest of 2&5, maximum negative
            9 -- maximax, the largest absolute value of 1-8
           10 -- returns a matrix s(9,length(fn)) with all the types 1-9.
    number_of_iterations : int
        Maximum number of iterations that can be performed at each frequency line.
        Default is 10.
    convergence : float, optional
        The fraction of the error corrected each iteration. The default is 0.8.
    error_tolerance : float, optional
        Allowable relative error in the SRS. The default is 0.05.
    ignore_compensation_pulse : bool, optional
        If True, do not use a compensation pulse.  Default is false.
    verbose : bool, optional
        If True, information on the interations will be provided. The default
        is False.

    Raises
    ------
    ValueError
        If compensation delay is required to be too long.

    Returns
    -------
    sine_amplitudes : np.ndarray
        An array the same size as the input `sine_amplitudes` array with updated
        amplitudes.
    compensation_amplitude : float
        Amplitude of the compensation pulse
    compensation_delay : float
        Delay of the compensation pulse.
    """
    # Define some helper variables
    Amax = np.max(np.abs(sine_amplitudes))
    # Reduce the error tolerance so there's a bit of room for round-off
    error_tolerance = error_tolerance*0.9
    # Get filter weights for SRS calculations
    b, a = sdof_ramp_invariant_filter_weights(sine_frequencies, sample_rate,
                                              damping_srs, srs_type)
    # Copy the arrays so we don't overwrite anything
    sine_frequencies = np.array(sine_frequencies).copy()
    sine_amplitudes = np.array(sine_amplitudes).copy()
    sine_decays = np.array(sine_decays).copy()
    sine_delays = np.array(sine_delays).copy()
    # Iterate one frequency at a time
    for i, (frequency, amplitude, decay, delay, desired_srs) in enumerate(zip(
            sine_frequencies, sine_amplitudes, sine_decays, sine_delays,
            required_srs)):
        # if i == 10:
        #     break
        srs_error = float('inf')
        iteration_count = 1
        increment = 0.1
        # Get the pulse without this frequency line
        other_sine_frequencies = np.concatenate((sine_frequencies[:i], sine_frequencies[i+1:]))
        other_sine_amplitudes = np.concatenate((sine_amplitudes[:i], sine_amplitudes[i+1:]))
        other_sine_decays = np.concatenate((sine_decays[:i], sine_decays[i+1:]))
        other_sine_delays = np.concatenate((sine_delays[:i], sine_delays[i+1:]))
        other_pulse = sum_decayed_sines_reconstruction(
            other_sine_frequencies, other_sine_amplitudes, other_sine_decays,
            other_sine_delays, sample_rate, block_size)
        # Now iterate at the single frequency until convergence
        while abs(srs_error) > error_tolerance:
            sine_amplitudes[i] = amplitude
            # Find the pulse at this frequency line
            this_pulse = sum_decayed_sines_reconstruction(
                frequency, amplitude, decay, delay, sample_rate, block_size)
            # Get the compensating pulse
            if not ignore_compensation_pulse:
                compensation_amplitude, compensation_delay = sum_decayed_sines_compensating_pulse_parameters(
                    sine_frequencies, sine_amplitudes, sine_decays, sine_delays, compensation_frequency, compensation_decay)
            else:
                compensation_amplitude = 0
                compensation_delay = 0
            # Find the number of samples to shift the waveform
            num_shift = -int(np.floor(compensation_delay*sample_rate))
            if abs(num_shift) >= block_size:
                raise ValueError('The number of samples in the compensation delay ({:}) is larger than the block_size ({:}).'.format(num_shift, block_size)
                                 + '  The entire pulse will consist of part of the compensation pulse.'
                                 + '  Please increase the block_size or compensation frequency.')
            compensation_delay_corrected = compensation_delay + num_shift/sample_rate
            # Find compensating time history
            compensation_pulse = sum_decayed_sines_reconstruction(
                compensation_frequency, compensation_amplitude, compensation_decay,
                compensation_delay_corrected, sample_rate, block_size)
            # Build the composite waveform, need to shift the signals to align
            if num_shift >= 0:
                composite_pulse = (
                      compensation_pulse
                      + np.concatenate((
                          np.zeros(num_shift),
                          (other_pulse + this_pulse)[:block_size-num_shift])))
            else:
                composite_pulse = (
                      other_pulse + this_pulse +
                      + np.concatenate((
                          np.zeros(-num_shift),
                          (compensation_pulse)[:block_size+num_shift])))
            # Find the SRS at the current frequency line
            srs_prediction = srs(composite_pulse, 1/sample_rate, frequency, damping_srs,
                                 srs_type, b[i], a[i])[0][0]  # only get the SRS and there should only be one value due to one frequency line
            srs_error = (srs_prediction - desired_srs)/desired_srs
            if verbose:
                print('Iteration at frequency {:}, {:0.4f}\n  Iteration Count: {:}, SRS Error: {:0.4f}'.format(i, frequency, iteration_count, srs_error))
            # Now we're going to compute the same thing again with a perturbed
            # amplitude, this will allow us to compute the slope change at the
            # current amplitude
            amplitude_change = np.sign(srs_error)*increment*np.sign(amplitude)*Amax
            # Now check and see if we need to modify the amplitude
            if abs(srs_error) > error_tolerance:
                if amplitude_change == 0:  # perturb it a bit
                    amplitude_change = np.sign(srs_error)*np.sign(sine_amplitudes[i])*Amax*increment/10
                new_amplitude = amplitude + amplitude_change
                # Can't allow the sign of the amplitude to change
                if np.sign(amplitude) != np.sign(new_amplitude):
                    new_amplitude *= -1
                if amplitude == new_amplitude:
                    new_amplitude = amplitude * 1.01
                sine_amplitudes[i] = new_amplitude
                # Find new component at this frequency
                this_pulse = sum_decayed_sines_reconstruction(
                    frequency, new_amplitude, decay, delay, sample_rate, block_size)
                # Get the compensating pulse
                if not ignore_compensation_pulse:
                    compensation_amplitude, compensation_delay = sum_decayed_sines_compensating_pulse_parameters(
                        sine_frequencies, sine_amplitudes, sine_decays, sine_delays, compensation_frequency, compensation_decay)
                else:
                    compensation_amplitude = 0
                    compensation_delay = 0
                # Find the number of samples to shift the waveform
                num_shift = -int(np.floor(compensation_delay*sample_rate))
                if abs(num_shift) >= block_size:
                    raise ValueError('The number of samples in the compensation delay ({:}) is larger than the block_size ({:}).'.format(num_shift, block_size)
                                     + '  The entire pulse will consist of part of the compensation pulse.  '
                                     + 'Please increase the block_size or compensation frequency.')
                compensation_delay_corrected = compensation_delay + num_shift/sample_rate
                # Find compensating time history
                compensation_pulse = sum_decayed_sines_reconstruction(
                    compensation_frequency, compensation_amplitude, compensation_decay,
                    compensation_delay_corrected, sample_rate, block_size)
                # Build the composite waveform, need to shift the signals to align
                if num_shift >= 0:
                    composite_pulse = (
                          compensation_pulse
                          + np.concatenate((
                              np.zeros(num_shift),
                              (other_pulse + this_pulse)[:block_size-num_shift])))
                else:
                    composite_pulse = (
                          other_pulse + this_pulse +
                          + np.concatenate((
                              np.zeros(-num_shift),
                              (compensation_pulse)[:block_size+num_shift])))
                # Find the SRS at the current frequency line
                srs_perturbed = srs(composite_pulse, 1/sample_rate, frequency, damping_srs,
                                    srs_type, b[i], a[i])[0][0]  # only get the SRS and there should only be one value due to one frequency line
                # Get slope of correction
                correction_slope = (abs(amplitude)-abs(new_amplitude))/(srs_prediction-srs_perturbed)
                if correction_slope > 0:
                    # this is what we want, it means we are changing in the right
                    # direction
                    new_amplitude = convergence*correction_slope*(desired_srs-srs_prediction) + abs(amplitude)
                    # Never let the change be too large
                    amplitude_check = max([abs(amplitude), Amax/10])
                    if new_amplitude > 2*amplitude_check:
                        new_amplitude = 2*amplitude_check
                    # Never let it be less than zero
                    if new_amplitude < 0:
                        new_amplitude = np.abs(amplitude)/10
                    # Move the previous amplitude into a different variable
                    old_amplitude = amplitude
                    # Never allow new amplitude to be zero
                    if new_amplitude == 0:
                        amplitude = 2*np.finfo(float).eps*np.sign(amplitude)
                    else:
                        amplitude = new_amplitude*np.sign(amplitude)
                    # Store it for the next iteration
                    sine_amplitudes[i] = amplitude
                    # Find new component at this frequency
                    this_pulse = sum_decayed_sines_reconstruction(
                        frequency, amplitude, decay, delay, sample_rate, block_size)
                    # Get the compensating pulse
                    if not ignore_compensation_pulse:
                        compensation_amplitude, compensation_delay = sum_decayed_sines_compensating_pulse_parameters(
                            sine_frequencies, sine_amplitudes, sine_decays, sine_delays, compensation_frequency, compensation_decay)
                    else:
                        compensation_amplitude = 0
                        compensation_delay = 0
                    # Find the number of samples to shift the waveform
                    num_shift = -int(np.floor(compensation_delay*sample_rate))
                    if abs(num_shift) >= block_size:
                        raise ValueError('The number of samples in the compensation delay'
                                         + ' ({:}) is larger than the block_size ({:}).'.format(num_shift, block_size)
                                         + '  The entire pulse will consist of part of the compensation pulse.  '
                                         + 'Please increase the block_size or compensation frequency.')
                    compensation_delay_corrected = compensation_delay + num_shift/sample_rate
                    # Find compensating time history
                    compensation_pulse = sum_decayed_sines_reconstruction(
                        compensation_frequency, compensation_amplitude, compensation_decay,
                        compensation_delay_corrected, sample_rate, block_size)
                    # Build the composite waveform, need to shift the signals to align
                    if num_shift >= 0:
                        composite_pulse = (
                              compensation_pulse
                              + np.concatenate((
                                  np.zeros(num_shift),
                                  (other_pulse + this_pulse)[:block_size-num_shift])))
                    else:
                        composite_pulse = (
                              other_pulse + this_pulse +
                              + np.concatenate((
                                  np.zeros(-num_shift),
                                  (compensation_pulse)[:block_size+num_shift])))
                    # Find the SRS at the current frequency line
                    srs_corrected = srs(composite_pulse, 1/sample_rate, frequency, damping_srs,
                                        srs_type, b[i], a[i])[0][0]  # only get the SRS and there should only be one value due to one frequency line
                    srs_error = (srs_corrected - desired_srs)/desired_srs
                    # If the SRS error is positive and the amplitude has already been reduced to zero, stop trying
                    if srs_error > 0 and amplitude == 0:
                        iteration_count = number_of_iterations + 1
                    if verbose:
                        print('  Old Amplitude: {:0.4f}, New Amplitude: {:0.4f}, Error: {:0.4f}'.format(old_amplitude, amplitude, srs_error))
                    # Don't allow amplitude to change more than a factor of 10 in an iteration
                    if np.abs(10*old_amplitude) < np.abs(amplitude) or np.abs(0.1*old_amplitude) > np.abs(amplitude):
                        iteration_count = number_of_iterations + 1
                else:
                    # The slope is negative, so we try a bigger increment
                    increment *= 2
                    amplitude_change = np.sign(srs_error)*increment*np.sign(amplitude)*Amax
                    if verbose:
                        print('Slope of correction was negative, trying a bigger increment')
                iteration_count += 1
                if iteration_count > number_of_iterations:
                    print('  Warning: SRS did not converge for frequency {:}: {:0.4f}'.format(i, frequency))
                    break
    if not ignore_compensation_pulse:
        compensation_amplitude, compensation_delay = sum_decayed_sines_compensating_pulse_parameters(
            sine_frequencies, sine_amplitudes, sine_decays, sine_delays, compensation_frequency, compensation_decay)
    else:
        compensation_amplitude = 0
        compensation_delay = 0
    return sine_amplitudes, compensation_amplitude, compensation_delay


def sum_decayed_sines_compensating_pulse_parameters(sine_frequencies, sine_amplitudes, sine_decays, sine_delays,
                                                    compensation_frequency, compensation_decay):
    omegas = sine_frequencies*2*np.pi
    omega_comp = compensation_frequency*2*np.pi
    var = -np.sum(sine_amplitudes/(sine_frequencies*(sine_decays**2+1)))
    compensation_amplitude = compensation_frequency*(compensation_decay**2+1)*var
    var0 = (np.sum(sine_amplitudes*sine_delays/(omegas*(sine_decays**2+1))))/compensation_amplitude
    # TODO Verify if this is a bug or not, I think this should be matlab code but graflab code had it as a matrix division.
    # var1 = (np.sum(2*sine_decays*sine_amplitudes/(omegas**2*(sine_decays**2 + 1)**2)))/compensation_amplitude
    var1 = (np.sum((2*sine_decays*sine_amplitudes)[np.newaxis, :]@np.linalg.pinv((omegas**2*(sine_decays**2 + 1)**2)[np.newaxis, :])))/compensation_amplitude
    var2 = 2*compensation_decay/(omega_comp*omega_comp*(compensation_decay**2+1)**2)
    var3 = omega_comp*(compensation_decay**2+1)
    compensation_delay = -var3*(var2 + var1 + var0)
    return compensation_amplitude, compensation_delay


def sum_decayed_sines_reconstruction(sine_frequencies, sine_amplitudes,
                                     sine_decays, sine_delays, sample_rate,
                                     block_size):
    """
    Computes a sum of decayed sines signal from a set of frequencies, amplitudes,
    decays, and delays.

    Parameters
    ----------
    sine_frequencies : ndarray
        An array of frequencies for each sine tone
    sine_amplitudes : ndarray
        An array of amplitudes for each sine tone
    sine_decays : ndarray
        An array of decay values for each sine tone
    sine_delays : ndarray
        An array of delay values for each sine tone
    sample_rate : float
        The sample rate of the signal
    block_size : int
        The number of samples in the signal

    Returns
    -------
    ndarray
        A signal containing the sum of decayed sinusoids.

    """
    times = np.arange(block_size)/sample_rate
    # See if any delays are below zero and if so then shift all delays forward
    if np.any(sine_delays < 0):
        sine_delays = sine_delays - np.min(sine_delays)
    # Now go through and compute the sine tones
    omegas = 2*np.pi*sine_frequencies
    this_times = times[:, np.newaxis] - sine_delays
    response = sine_amplitudes*np.exp(-sine_decays*omegas*this_times)*np.sin(omegas*this_times)
    response[this_times < 0] = 0
    return np.sum(response, axis=-1)


def sum_decayed_sines_reconstruction_with_compensation(
        sine_frequencies, sine_amplitudes,
        sine_decays, sine_delays, compensation_frequency, compensation_decay,
        sample_rate, block_size, ignore_compensation_pulse = False):
    """


    Parameters
    ----------
    sine_frequencies : ndarray
        An array of frequencies for each sine tone
    sine_amplitudes : ndarray
        An array of amplitudes for each sine tone
    sine_decays : ndarray
        An array of decay values for each sine tone
    sine_delays : ndarray
        An array of delay values for each sine tone
    compensation_frequency : float
        The frequency of the compensation pulse
    compensation_decay : float
        The decay value for the compensation pulse
    sample_rate : float
        The sample rate of the signal
    block_size : int
        The number of samples in the signal
    ignore_compensation_pulse : bool, optional
        If True, ignores the compensation pulse

    Returns
    -------
    signal : ndarray
        A signal containing the sum of decayed sinusoids.
    compensation_frequency : float
        The frequency of the compensation pulse
    compensation_amplitude : float
        The amplitude value for the compensation pulse
    compensation_decay : float
        The decay value for the compensation pulse
    compensation_delay : float
        The delay value for the compensation pulse
    """
    sine_frequencies = np.array(sine_frequencies).flatten()
    sine_amplitudes = np.array(sine_amplitudes).flatten()
    sine_delays = np.array(sine_delays).flatten()
    sine_decays = np.array(sine_decays).flatten()
    if not ignore_compensation_pulse:
        compensation_amplitude, compensation_delay = sum_decayed_sines_compensating_pulse_parameters(
            sine_frequencies, sine_amplitudes, sine_decays, sine_delays,
            compensation_frequency, compensation_decay)
    else:
        compensation_amplitude = 0
        compensation_delay = 0
    sine_frequencies = np.concatenate((sine_frequencies, [compensation_frequency]))
    sine_amplitudes = np.concatenate((sine_amplitudes, [compensation_amplitude]))
    sine_delays = np.concatenate((sine_delays, [compensation_delay]))
    sine_decays = np.concatenate((sine_decays, [compensation_decay]))
    signal = sum_decayed_sines_reconstruction(
        sine_frequencies, sine_amplitudes, sine_decays, sine_delays,
        sample_rate, block_size)
    return (signal, compensation_frequency, compensation_amplitude,
            compensation_decay, compensation_delay)


def sum_decayed_sines_displacement_velocity(
        sine_frequencies, sine_amplitudes,
        sine_decays, sine_delays, sample_rate, block_size,
        acceleration_factor=1):
    """
    Creates velocity and displacement signals from acceleration sinusoids.

    Parameters
    ----------
    sine_frequencies : ndarray
        An array of frequencies for each sine tone
    sine_amplitudes : ndarray
        An array of amplitudes for each sine tone
    sine_decays : ndarray
        An array of decay values for each sine tone
    sine_delays : ndarray
        An array of delay values for each sine tone
    sample_rate : float
        The sample rate of the signal
    block_size : int
        The number of samples in the signal
    acceleration_factor : float, optional
        Optional scale factor to convert acceleration into velocity and
        displacement.  For example, if sine amplitudes are in G and displacement
        is desired in inches, the acceleration factor should be set to 386.089.
        If sine amplitudes are in G and displacement is desired in meters, the
        acceleration factor should be set to 9.80665.  The default is 1, which
        assumes consistent units (e.g. acceleration in m/s^2, velocity in m/s,
        displacement in m).

    Returns
    -------
    v : ndarray
        The velocity of the signal.
    d : ndarray
        The displacement of the signal.

    """
    # Make sure everything is a numpy array
    sine_frequencies = np.array(sine_frequencies).flatten()
    sine_amplitudes = np.array(sine_amplitudes).flatten()
    sine_delays = np.array(sine_delays).flatten()
    sine_decays = np.array(sine_decays).flatten()
    # Transform units
    sine_amplitudes = sine_amplitudes * acceleration_factor
    # See if any delays are below zero and if so then shift all delays forward
    if np.any(sine_delays < 0):
        sine_delays = sine_delays - np.min(sine_delays)
    f = sine_frequencies
    A = sine_amplitudes
    z = sine_decays
    tau = sine_delays
    x = np.zeros(block_size)
    tmp1 = x.copy()
    tmp2 = x.copy()
    tmp3 = x.copy()
    x1 = np.ones(x.shape)
    v = x.copy()
    d = x.copy()
    w = 2*np.pi*f
    w2 = w*w
    zw = z*w
    zp1 = z*z + 1
    zm1 = z*z - 1
    t = np.arange(block_size)/sample_rate
    for k in range(len(sine_frequencies)):
        indices = t-tau[k] > 0   # index's where vel and disp are evaluated
        Awz1 = A[k]/(w[k]*zp1[k])
        tmp1[indices] = Awz1*np.exp(-zw[k]*(t[indices]-tau[k]))
        tmp2[indices] = z[k]*np.sin(w[k]*(t[indices]-tau[k])) + np.cos(w[k]*(t[indices]-tau[k]))
        v[indices] = v[indices] - tmp1[indices]*tmp2[indices] + Awz1*x1[indices]
        Awz2 = A[k]/(w2[k]*zp1[k]*zp1[k])
        tmp1[indices] = Awz2*np.exp(-zw[k]*(t[indices]-tau[k]))
        tmp2[indices] = zm1[k]*np.sin(w[k]*(t[indices]-tau[k])) + 2*z[k]*np.cos(w[k]*(t[indices]-tau[k]))
        tmp3[indices] = Awz1*(t[indices]-tau[k])
        tmp4 = 2*z[k]*Awz2
        d[indices] = d[indices] + tmp1[indices]*tmp2[indices] + tmp3[indices] - tmp4
    return v, d


def loginterp(x, xp, fp):
    return 10**np.interp(np.log10(x), np.log10(xp), np.log10(fp))


def optimization_error_function(
        amplitude_lin_scales, sample_rate, block_size,
        sine_frequencies, sine_amplitudes, sine_decays, sine_delays,
        compensation_frequency, compensation_decay, control_irfs, limit_irfs,
        srs_damping, srs_type, control_srs, control_weights, limit_srs,
        b=None, a=None, frequency_index=None):
    if frequency_index is None:
        frequency_index = slice(None)
    # Apply the scale factors
    new_sine_amplitudes = sine_amplitudes.copy()
    new_sine_amplitudes[frequency_index] *= amplitude_lin_scales
    # Compute a new time history signal
    (input_signal, new_compensation_frequency, new_compensation_amplitude,
     new_compensation_decay, new_compensation_delay) = sum_decayed_sines_reconstruction_with_compensation(
        sine_frequencies, new_sine_amplitudes, sine_decays, sine_delays,
        compensation_frequency, compensation_decay, sample_rate, block_size)
    # Transform the signal to the control degrees of freedom
    control_responses = np.array([oaconvolve(control_irf, input_signal) for control_irf in control_irfs])[..., :block_size]
    # Compute SRSs at all frequencies
    predicted_control_srs, frequencies = srs(control_responses, 1/sample_rate, sine_frequencies[frequency_index],
                                             srs_damping, srs_type,
                                             None if b is None else b[frequency_index],
                                             None if a is None else a[frequency_index])
    # Compute error
    mean_control_error = np.mean(
            (control_weights[:, np.newaxis]
             * ((predicted_control_srs - control_srs[..., frequency_index])
                / control_srs[..., frequency_index])))
    if limit_srs is not None:
        limit_responses = np.array([oaconvolve(limit_irf, input_signal) for limit_irf in limit_irfs])[..., :block_size]
        predicted_limit_srs, frequencies = srs(limit_responses, 1/sample_rate, sine_frequencies[frequency_index],
                                               srs_damping, srs_type,
                                               None if b is None else b[frequency_index],
                                               None if a is None else a[frequency_index])
        max_limit_error = np.max(
            ((predicted_limit_srs - limit_srs[..., frequency_index])
             / limit_srs[..., frequency_index]))
        if max_limit_error >= 0 and mean_control_error >= 0:
            srs_error = np.max((mean_control_error, max_limit_error))
        elif max_limit_error <= 0 and mean_control_error <= 0:
            srs_error = np.max((mean_control_error, max_limit_error))
        elif max_limit_error >= 0 and mean_control_error <= 0:
            srs_error = max_limit_error
        elif max_limit_error <= 0 and mean_control_error >= 0:
            srs_error = mean_control_error
    else:
        limit_responses = None
        predicted_limit_srs = None
        srs_error = mean_control_error

    return (np.abs(srs_error), input_signal, control_responses,
            predicted_control_srs, limit_responses, predicted_limit_srs,
            new_sine_amplitudes, new_compensation_frequency,
            new_compensation_amplitude, new_compensation_decay,
            new_compensation_delay)


def optimization_callback(intermediate_result, rms_error_threshold=0.02,
                          verbose=True):
    if verbose:
        print('Amplitude Scale: {:}, error is {:}'.format(intermediate_result.x.squeeze(),
                                                          intermediate_result.fun))
    if rms_error_threshold is not None:
        if intermediate_result.fun < rms_error_threshold:
            raise StopIteration


def sum_decayed_sines_minimize(sample_rate, block_size,
                               sine_frequencies=None, sine_tone_range=None, sine_tone_per_octave=None,
                               sine_amplitudes=None, sine_decays=None, sine_delays=None,
                               control_srs=None, control_breakpoints=None,
                               srs_damping=0.05, srs_type=9,
                               compensation_frequency=None, compensation_decay=0.95,
                               # Parameters for defining decays
                               tau=None, num_time_constants=None, decay_resolution=None,
                               scale_factor=1.02,
                               acceleration_factor=1.0,
                               # Parameters for imposing limits
                               limit_breakpoints=None, limit_transfer_functions=None,
                               control_transfer_functions=None, control_weights=None,
                               # Parameters for the optimizer
                               minimize_iterations=1, rms_error_threshold=None,
                               optimization_passes=3,
                               plot_results=False, verbose=False
                               ):
    # Handle the sine tone frequencies
    if sine_frequencies is None and sine_tone_range is None:
        raise ValueError('Either `sine_frequencies` or `sine_tone_range` must be specified')
    if sine_frequencies is not None and sine_tone_range is not None:
        raise ValueError('`sine_frequencies` can not be specified simultaneously with `sine_tone_range`')
    if sine_frequencies is None:
        # Create sine tones
        if sine_tone_per_octave is None:
            sine_tone_per_octave = int(np.floor(9-srs_damping*100))
        sine_frequencies = octspace(sine_tone_range[0], sine_tone_range[1],
                                    sine_tone_per_octave)
    # Now set up the SRS
    if control_srs is None and control_breakpoints is None:
        raise ValueError('Either `control_srs` or `control_breakpoints` must be specified')
    if control_srs is not None and control_breakpoints is not None:
        raise ValueError('`control_srs` can not be specified simultaneously with `control_breakpoints`')
    if control_srs is None:
        frequencies = control_breakpoints[:, 0]
        breakpoint_curves = control_breakpoints[:, 1:].T
        control_srs = np.array([loginterp(sine_frequencies,
                                          frequencies,
                                          breakpoint_curve) for breakpoint_curve in breakpoint_curves])
    else:
        control_srs = np.atleast_2d(control_srs)
    if control_transfer_functions is None:
        control_transfer_functions = np.ones(((control_srs.shape[0]-1)*2, block_size//2+1))
    tf_frequencies = np.fft.rfftfreq(control_transfer_functions.shape[-1]*2-1, 1/sample_rate)
    if sine_amplitudes is None:
        tf_at_frequencies = np.array([
            np.interp(sine_frequencies, tf_frequencies, np.abs(control_transfer_function))
            for control_transfer_function in control_transfer_functions])
        srs_amplitudes = np.array([nnls(ai[:, np.newaxis], bi)[0][0] for ai, bi in zip(tf_at_frequencies.T, control_srs.T)])
        srs_amplitudes[np.arange(srs_amplitudes.size) % 2 == 0] *= -1
        quality = 1/(2*srs_damping)
        srs_amplitudes /= quality
        sine_amplitudes = srs_amplitudes

    if sine_delays is None:
        sine_delays = np.zeros(sine_frequencies.size)

    if compensation_frequency is None:
        compensation_frequency = np.min(sine_frequencies)/3

    # Set up decay terms
    decay_terms_specified = 0
    if sine_decays is not None:
        decay_terms_specified += 1
    if tau is not None:
        decay_terms_specified += 1
    if num_time_constants is not None:
        decay_terms_specified += 1
    if decay_terms_specified == 0:
        raise ValueError('One of `sine_decays`, `tau`, or `num_time_constants` must be specified')
    if decay_terms_specified > 1:
        raise ValueError('Only one of `sine_decays`, `tau`, or `num_time_constants` can be specified')

    # Now check and see which is defined
    if num_time_constants is not None:
        period = block_size / sample_rate
        if isinstance(num_time_constants, dict):
            tau = {}
            for freq_range, num_time_constant in num_time_constants.items():
                tau[freq_range] = period / num_time_constant
        else:
            tau = period/num_time_constants
    if tau is not None:
        sine_decays = []
        for freq in sine_frequencies:
            if isinstance(tau, dict):
                this_decay = None
                for freq_range, this_tau in tau.items():
                    if freq_range[0] <= freq <= freq_range[1]:
                        this_decay = 1/(2*np.pi*freq*this_tau)
                        break
                if this_decay is None:
                    raise ValueError('No frequency range matching frequency {:} was found in the specified decay parameters.'.format(freq))
                sine_decays.append(this_decay)
            else:
                sine_decays.append(1/(2*np.pi*freq*tau))
        sine_decays = np.array(sine_decays)
    # Otherwise we just keep the specified sine_decays

    # Now handle the minimum resolution on the decay
    if decay_resolution is not None:
        sine_decays = decay_resolution*np.round(sine_decays/decay_resolution)

    if compensation_frequency is None:
        compensation_frequency = np.min(sine_frequencies)/3

    # Now set up limits and transfer functions
    if control_weights is None:
        control_weights = np.ones((control_srs.shape[0]))
    if limit_breakpoints is not None:
        frequencies = limit_breakpoints[:, 0]
        breakpoint_curves = limit_breakpoints[:, 1:].T
        limit_srs = np.array([loginterp(sine_frequencies,
                                        frequencies,
                                        breakpoint_curve) for breakpoint_curve in breakpoint_curves])
    else:
        limit_srs = None

    # Compute impulse responses
    control_irfs = np.fft.irfft(control_transfer_functions, axis=-1)
    if limit_transfer_functions is not None:
        limit_irfs = np.fft.irfft(limit_transfer_functions, axis=-1)
    else:
        limit_irfs = None
    b, a = sdof_ramp_invariant_filter_weights(sine_frequencies, sample_rate, srs_damping, srs_type)

    # Normalize control weights
    control_weights = control_weights/np.linalg.norm(control_weights)

    # Copy things so we don't overwrite
    sine_amplitudes = sine_amplitudes.copy()

    # Now we will iterate over all of the frequencies and compute the new
    # amplitudes
    for j in range(optimization_passes):
        for i in range(len(sine_frequencies)):
            if verbose:
                print('Pass {:}, Analyzing Frequency {:}: {:0.2f}'.format(j+1, i, sine_frequencies[i]))

            # We will now go through and optimize the frequency line
            def error_function(x):
                return optimization_error_function(
                    x, sample_rate, block_size,
                    sine_frequencies, sine_amplitudes, sine_decays, sine_delays,
                    compensation_frequency, compensation_decay, control_irfs, limit_irfs,
                    srs_damping, srs_type, control_srs, control_weights, limit_srs,
                    b, a, frequency_index=[i])[0]

            def callback(intermediate_result):
                return optimization_callback(
                    intermediate_result, rms_error_threshold, verbose)

            optimization_result = minimize(error_function,
                                           np.ones(1),
                                           method='Powell',
                                           bounds=[(0, np.inf)],
                                           callback=callback,
                                           options={'maxiter': minimize_iterations})

            # Populate the amplitudes with the updated result
            if verbose:
                print('Initial Amplitude: {:}, Updated Amplitude: {:}\n'.format(sine_amplitudes[i], sine_amplitudes[i]*optimization_result.x.squeeze()))
            sine_amplitudes[i] *= optimization_result.x.squeeze()

    (error, input_signal, control_responses, predicted_control_srs,
     limit_responses, predicted_limit_srs,
     sine_amplitudes, compensation_frequency,
     compensation_amplitude, compensation_decay,
     compensation_delay) = optimization_error_function(
         np.ones(sine_frequencies.size),
         sample_rate, block_size,
         sine_frequencies, sine_amplitudes, sine_decays, sine_delays,
         compensation_frequency, compensation_decay, control_irfs, limit_irfs,
         srs_damping, srs_type, control_srs, control_weights, limit_srs,
         b, a)

    all_frequencies = np.concatenate((sine_frequencies,
                                      [compensation_frequency]))
    all_amplitudes = np.concatenate((sine_amplitudes,
                                     [compensation_amplitude]))
    all_decays = np.concatenate((sine_decays,
                                 [compensation_decay]))
    all_delays = np.concatenate((sine_delays,
                                 [compensation_delay]))

    # Now compute displacement and velocity
    velocity_signal, displacement_signal = sum_decayed_sines_displacement_velocity(
        all_frequencies, all_amplitudes, all_decays, all_delays, sample_rate,
        block_size, acceleration_factor)

    return_vals = (input_signal, velocity_signal, displacement_signal,
                   control_responses, predicted_control_srs,
                   limit_responses, predicted_limit_srs,
                   all_frequencies, all_amplitudes, all_decays, all_delays,
                   )

    if plot_results:
        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        times = np.arange(block_size)/sample_rate
        ax[0, 0].plot(times, input_signal)
        ax[0, 0].set_ylabel('Acceleration')
        ax[0, 0].set_xlabel('Time (s)')
        ax[0, 1].plot(times, velocity_signal)
        ax[0, 1].set_ylabel('Velocity')
        ax[0, 1].set_xlabel('Time (s)')
        ax[1, 0].plot(times, displacement_signal)
        ax[1, 0].set_ylabel('Displacement')
        ax[1, 0].set_xlabel('Time (s)')
        # Compute SRS
        this_srs, this_frequencies = srs(
            input_signal, 1/sample_rate, sine_frequencies,
            srs_damping, srs_type)
        if control_breakpoints is None:
            srs_abscissa = sine_frequencies
            srs_ordinate = control_srs.T
        else:
            srs_abscissa = control_breakpoints[:, 0]
            srs_ordinate = control_breakpoints[:, 1:]
        ax[1, 1].plot(srs_abscissa, srs_ordinate, 'k--')
        ax[1, 1].plot(this_frequencies, this_srs)
        ax[1, 1].set_ylabel('SRS ({:0.2f}% damping)'.format(srs_damping*100))
        ax[1, 1].set_xlabel('Frequency (Hz)')
        ax[1, 1].set_yscale('log')
        ax[1, 1].set_xscale('log')
        ax[1, 1].legend(('Reference', 'Decayed Sine'))
        fig.tight_layout()
        return_vals += (fig, ax)
        # Now plot all control channels
        for i, (predicted, control, response) in enumerate(zip(predicted_control_srs, control_srs, control_responses)):
            fig, ax = plt.subplots(1, 2, figsize=(8, 3))
            ax[0].set_title('Control Signal {:}'.format(i))
            ax[0].plot(times, response)
            ax[0].set_label('Acceleration')
            ax[0].set_xlabel('Time (s)')
            ax[1].set_title('Control SRS {:}'.format(i))
            ax[1].plot(sine_frequencies, control, 'k--')
            ax[1].plot(sine_frequencies, predicted)
            ax[1].set_ylabel('SRS ({:0.2f}% damping)'.format(srs_damping*100))
            ax[1].set_xlabel('Frequency (Hz)')
            ax[1].set_yscale('log')
            ax[1].set_xscale('log')
            ax[1].legend(('Desired', 'Achieved'))
            fig.tight_layout()
            return_vals += (fig, ax)
        # Now plot all limit channels
        if limit_srs is not None:
            for i, (predicted, limit, response) in enumerate(zip(predicted_limit_srs, limit_srs, limit_responses)):
                fig, ax = plt.subplots(1, 2, figsize=(8, 3))
                ax[0].set_title('Limit Signal {:}'.format(i))
                ax[0].plot(times, response)
                ax[0].set_label('Acceleration')
                ax[0].set_xlabel('Time (s)')
                ax[1].set_title('Limit SRS {:}'.format(i))
                ax[1].plot(sine_frequencies, limit, 'k--')
                ax[1].plot(sine_frequencies, predicted)
                ax[1].set_ylabel('SRS ({:0.2f}% damping)'.format(srs_damping*100))
                ax[1].set_xlabel('Frequency (Hz)')
                ax[1].set_yscale('log')
                ax[1].set_xscale('log')
                ax[1].legend(('Desired', 'Achieved'))
                fig.tight_layout()
                return_vals += (fig, ax)

    return return_vals


@dataclass
class WindowedRandomMetrics:
    """
    Scalar metrics describing one generated realization.

    Parameters
    ----------
    lanl_rms_db_error : float
        LANL-style root-mean-square dB error between the achieved and target
        SRS over the matching frequencies.
    energy : float
        Signal energy computed from the acceleration time history.
    rea : float
        Root energy amplitude.
    peak_accel : float
        Maximum absolute acceleration in the compensated windowed history.
    peak_vel : float
        Maximum absolute velocity derived by integrating the compensated
        acceleration history.
    peak_disp : float
        Maximum absolute displacement derived by integrating the compensated
        acceleration history.

    Notes
    -----
    These metrics are derived from the compensated windowed realization and are
    intended for ranking or selecting among multiple random realizations.
    """
    lanl_rms_db_error: float
    energy: float
    rea: float
    peak_accel: float
    peak_vel: float
    peak_disp: float


@dataclass
class WindowedRandomResults:
    """
    Collection of generated windowed-random realizations and associated results.

    Parameters
    ----------
    time : ndarray
        Time vector with shape `(n_samples,)`.
    sample_rate : float
        Sampling rate in samples per second.
    gravity : float
        Acceleration-of-gravity conversion factor used when integrating
        acceleration to obtain velocity and displacement.
    window : ndarray
        Applied time-domain window with shape `(n_samples,)`.
    ref_srs : ndarray
        Reference SRS breakpoint array with shape `(n_breakpoints, 2)`. The
        first column contains frequency and the second contains spectral
        amplitude.
    srs_frequency : ndarray
        Frequency vector corresponding to `srs_matrix`, with shape `(n_freq,)`.
    srs_matrix : ndarray
        Computed SRS values for each realization, with shape
        `(n_freq, n_realizations)`.
    xc_matrix : ndarray
        Compensated windowed acceleration histories, with shape
        `(n_samples, n_realizations)`.
    rr_matrix : ndarray
        Stationary random acceleration histories prior to windowing, with shape
        `(n_samples, n_realizations)`.
    metrics : list of WindowedRandomMetrics
        Metrics for each realization.
    metadata : dict
        Dictionary of generation settings and other bookkeeping information.

    Notes
    -----
    Each column of `xc_matrix`, `rr_matrix`, and `srs_matrix` corresponds to
    one realization.
    """
    time: np.ndarray
    sample_rate: float
    gravity: float
    window: np.ndarray
    ref_srs: np.ndarray
    srs_frequency: np.ndarray
    srs_matrix: np.ndarray
    xc_matrix: np.ndarray
    rr_matrix: np.ndarray
    metrics: List[WindowedRandomMetrics]
    metadata: Dict[str, Any]

    @property
    def metrics_array(self) -> np.ndarray:
        """
        Return the realization metrics as a dense numeric array.

        Returns
        -------
        ndarray
            Array with shape `(n_realizations, 6)` containing columns:

            1. LANL RMS dB error
            2. energy
            3. REA
            4. peak acceleration
            5. peak velocity
            6. peak displacement
        """
        return np.array([
            [
                m.lanl_rms_db_error,
                m.energy,
                m.rea,
                m.peak_accel,
                m.peak_vel,
                m.peak_disp,
            ]
            for m in self.metrics
        ], dtype=float)


@dataclass
class SelectedRealization:
    """
    One realization selected from an ensemble.

    Parameters
    ----------
    index : int
        Zero-based index of the selected realization.
    time : ndarray
        Time vector with shape `(n_samples,)`.
    xc : ndarray
        Compensated windowed acceleration history with shape `(n_samples,)`.
    rr : ndarray
        Stationary random acceleration history with shape `(n_samples,)`.
    srs : ndarray
        Shock response spectrum of the selected realization with shape
        `(n_freq, 2)`, where the first column is frequency and the second is
        SRS amplitude.
    metrics : WindowedRandomMetrics
        Scalar metrics associated with the selected realization.
    """
    index: int
    time: np.ndarray
    xc: np.ndarray
    rr: np.ndarray
    srs: np.ndarray
    metrics: WindowedRandomMetrics


def breakpoint_spectrum(
    frequencies: np.ndarray,
    break_frequencies: np.ndarray,
    break_values: np.ndarray,
    floor: Optional[float] = None,
) -> np.ndarray:
    """
    Evaluate a breakpoint-defined spectrum using log-log interpolation.

    Parameters
    ----------
    frequencies : ndarray
        Frequencies at which the spectrum should be evaluated.
    break_frequencies : ndarray
        Monotonically increasing breakpoint frequencies.
    break_values : ndarray
        Spectrum amplitudes at `break_frequencies`.
    floor : float, optional
        Value used outside the breakpoint range. If not provided, the default
        is `min(break_values) / 1000`.

    Returns
    -------
    ndarray
        Spectrum amplitudes evaluated at `frequencies`.

    Raises
    ------
    ValueError
        If any breakpoint frequencies or breakpoint amplitudes are not
        positive.

    Notes
    -----
    This function reproduces the numerical intent of MATLAB ``bpspec``:
    straight lines on log-log axes between breakpoint pairs, with a constant
    floor outside the breakpoint range.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    break_frequencies = np.asarray(break_frequencies, dtype=float)
    break_values = np.asarray(break_values, dtype=float)

    if np.any(break_frequencies <= 0):
        raise ValueError("break_frequencies must be positive")
    if np.any(break_values <= 0):
        raise ValueError("break_values must be positive")

    if floor is None:
        floor = np.min(break_values) / 1000.0

    y = np.full_like(frequencies, floor, dtype=float)
    inside = (frequencies >= break_frequencies[0]) & (frequencies <= break_frequencies[-1])

    if np.any(inside):
        f = interp1d(
            np.log(break_frequencies),
            np.log(break_values),
            kind="linear",
            bounds_error=False,
            fill_value=np.log(floor),
        )
        y[inside] = np.exp(f(np.log(frequencies[inside])))

    return y


def temporal_moments(x: np.ndarray, dt: float) -> Dict[str, float]:
    """
    Compute temporal moments of a sampled signal.

    Parameters
    ----------
    x : ndarray
        Input signal amplitudes with shape `(n_samples,)`.
    dt : float
        Sampling interval.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"energy"`` : float
            Signal energy.
        - ``"tau"`` : float
            Time centroid.
        - ``"duration_rms"`` : float
            Root-mean-square duration.
        - ``"skewness_metric"`` : float
            Cube-root form of the third central temporal moment.
        - ``"kurtosis_metric"`` : float
            Fourth-root form of the fourth central temporal moment.
        - ``"rea"`` : float
            Root energy amplitude.

    Notes
    -----
    This function follows the intent of MATLAB ``gtempmom`` and is used to
    compute metrics for each generated acceleration history.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    t = dt * np.arange(len(x))
    z = x ** 2

    E = abs(dt * np.sum(z))
    tau = float(dt * np.dot(t, z) / E)

    t_centered = t - tau
    D2 = float((dt / E) * np.dot(t_centered ** 2, z))
    D = np.sqrt(max(D2, 0.0))

    S3 = float((dt / E) * np.dot(t_centered ** 3, z))
    K4 = float((dt / E) * np.dot(t_centered ** 4, z))

    S = np.sign(S3) * abs(S3) ** (1 / 3) if S3 != 0 else 0.0
    K = abs(K4) ** 0.25
    R = np.sqrt(E / D) if D > 0 else np.inf

    return {
        "energy": E,
        "tau": tau,
        "duration_rms": D,
        "skewness_metric": S,
        "kurtosis_metric": K,
        "rea": R,
    }


def lanl_rms_db_error(
    db_error: np.ndarray,
    frequency: np.ndarray,
    octave_resolution: float,
) -> Dict[str, np.ndarray | float]:
    """
    Compute LANL-style RMS dB error over octave-spaced frequency lines.

    Parameters
    ----------
    db_error : ndarray
        dB error values. May be one-dimensional with shape `(n_freq,)` or
        two-dimensional with shape `(n_freq, n_series)`.
    frequency : ndarray
        Frequency vector with shape `(n_freq,)`.
    octave_resolution : float
        Number of points per octave.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"rms"`` : float
            Average RMS dB error across series.
        - ``"std"`` : float
            Average standard deviation of dB error across series.
        - ``"avg"`` : ndarray
            Mean dB error across series at each frequency.

    Raises
    ------
    ValueError
        If the first dimension of `db_error` does not match the length of
        `frequency`.

    Notes
    -----
    This follows the octave-band weighting used in the MATLAB
    ``lanl_rmsdberror`` routine.
    """
    db_error = np.asarray(db_error, dtype=float)
    frequency = np.asarray(frequency, dtype=float).reshape(-1)

    if db_error.ndim == 1:
        db_error = db_error[:, None]

    if db_error.shape[0] != len(frequency):
        raise ValueError("frequency length must match first dimension of db_error")

    upper = (2 ** (1 / (2 * octave_resolution))) * frequency
    lower = np.concatenate([[frequency[0] / (2 ** (1 / (2 * octave_resolution)))], upper[:-1]])
    band = upper - lower
    frange = upper[-1] - lower[0]

    avg = np.mean(db_error, axis=1)

    rms_vals = []
    std_vals = []
    for j in range(db_error.shape[1]):
        rms_vals.append(np.sqrt((1.0 / frange) * np.sum((db_error[:, j] ** 2) * band)))
        std_vals.append(np.sqrt((1.0 / frange) * np.sum(((db_error[:, j] - avg) ** 2) * band)))

    return {
        "rms": float(np.mean(rms_vals)),
        "std": float(np.mean(std_vals)),
        "avg": avg,
    }


_NUTTALL_COEFFS = {
    0: (1.0, 0.0, 0.0, 0.0),
    1: (0.5, 0.5, 0.0, 0.0),
    2: (0.42, 0.50, 0.08, 0.0),
    3: (7938 / 18608, 9240 / 18608, 1430 / 18608, 0.0),
    4: (0.42323, 0.49755, 0.07922, 0.0),
    5: (0.44959, 0.49364, 0.05677, 0.0),
    6: (0.35875, 0.48829, 0.14128, 0.01168),
    7: (0.40217, 0.49703, 0.09892, 0.00188),
    8: (0.375, 0.5, 0.125, 0.0),
    9: (0.40897, 0.5, 0.09103, 0.0),
    10: (10 / 32, 15 / 32, 6 / 32, 1 / 32),
    11: (0.338936, 0.481973, 0.161054, 0.018027),
    12: (0.355768, 0.487396, 0.144232, 0.012604),
    13: (0.53836, 0.46164, 0.0, 0.0),
    14: (0.4243801, 0.4973406, 0.0782793, 0.0),
    15: (0.3635819, 0.4891775, 0.1365995, 0.0106411),
}


def nuttall_window(n: int, window_type: int = 12) -> np.ndarray:
    """
    Generate a Nuttall-family window.

    Parameters
    ----------
    n : int
        Number of samples in the window.
    window_type : int, default=12
        Window type identifier. Only the subset required by the compensation
        algorithm is supported here.

    Returns
    -------
    ndarray
        Window values with shape `(n,)`.

    Raises
    ------
    ValueError
        If `n` is not positive or if `window_type` is unsupported.

    Notes
    -----
    This is a minimal implementation derived from the MATLAB ``nuttall``
    routine and is included only to support waveform compensation.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    if window_type == 16:
        return windows.flattop(n, sym=False)
    if window_type == 17:
        if n % 2 == 0:
            t = np.arange(-n // 2, n // 2)
        else:
            m = (n - 1) / 2
            t = np.arange(-m - 0.5, m + 0.5, 1.0)
        return np.cos(np.pi * t / n)
    if window_type == 19:
        return windows.kaiser(n, beta=9.0, sym=False)

    if window_type not in _NUTTALL_COEFFS:
        raise ValueError(f"Unsupported Nuttall window type {window_type}")

    if n % 2 == 0:
        t = np.arange(-n // 2, n // 2)
    else:
        m = (n - 1) / 2
        t = np.arange(-m - 0.5, m + 0.5, 1.0)

    a0, a1, a2, a3 = _NUTTALL_COEFFS[window_type]
    return (
        a0
        + a1 * np.cos(2 * np.pi * t / n)
        + a2 * np.cos(4 * np.pi * t / n)
        + a3 * np.cos(6 * np.pi * t / n)
    )


def window_compensation(
    x: np.ndarray,
    window: np.ndarray | float,
    sample_rate: float,
    delay: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Apply a compensation waveform to enforce zero terminal cumulative moments.

    Parameters
    ----------
    x : ndarray
        Input waveform with shape `(n_samples,)`.
    window : ndarray or float
        If an array, it is used directly as the compensation window. If a
        scalar, it is interpreted as a compensation frequency and a Nuttall
        window is generated internally.
    sample_rate : float
        Sampling rate in samples per second.
    delay : float, default=0.0
        Delay applied to the compensating waveform, in seconds.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"comp"`` : ndarray
            The compensating waveform.
        - ``"xc"`` : ndarray
            The compensated waveform.
        - ``"coefficients"`` : ndarray
            The three coefficients multiplying the DC, sine, and cosine
            compensation basis terms.

    Notes
    -----
    This function follows the numerical approach of MATLAB ``sdwcomp``. The
    compensation is constructed from three basis terms: a windowed DC term,
    a windowed sine term, and a windowed cosine term. Their amplitudes are
    chosen so that the final cumulative sums corresponding to acceleration,
    velocity, and displacement-like integrals vanish.
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    if np.max(np.abs(x)) <= np.finfo(float).eps:
        return {
            "comp": np.zeros_like(x),
            "xc": np.zeros_like(x),
            "coefficients": np.zeros(3),
        }

    if np.isscalar(window):
        freq = float(window)
        lc = int(np.fix(sample_rate / freq))
        w = nuttall_window(lc, window_type=12)
    else:
        w = np.asarray(window, dtype=float).reshape(-1)
        lc = len(w)
        freq = sample_rate / lc

    t = np.arange(lc) / sample_rate
    acd = w
    acs = w * np.sin(2 * np.pi * freq * t)
    acc = w * np.cos(2 * np.pi * freq * t)

    ld = int(np.fix(delay * sample_rate))
    lx = len(x)

    if ld < 0:
        if lc + ld <= lx:
            l = -ld + lx
            xw = np.concatenate([np.zeros(l - lx), x])
            acd = np.concatenate([acd, np.zeros(l - lc)])
            acs = np.concatenate([acs, np.zeros(l - lc)])
            acc = np.concatenate([acc, np.zeros(l - lc)])
        else:
            l = lc
            xw = np.concatenate([np.zeros(-ld), x, np.zeros(l - lx + ld)])
    else:
        if ld + lc <= lx:
            l = lx
            xw = x.copy()
            acd = np.concatenate([np.zeros(ld), acd, np.zeros(lx - ld - lc)])
            acs = np.concatenate([np.zeros(ld), acs, np.zeros(lx - ld - lc)])
            acc = np.concatenate([np.zeros(ld), acc, np.zeros(lx - ld - lc)])
        else:
            l = ld + lc
            xw = np.concatenate([x, np.zeros(l - lx)])
            acd = np.concatenate([np.zeros(ld), acd])
            acs = np.concatenate([np.zeros(ld), acs])
            acc = np.concatenate([np.zeros(ld), acc])

    xv = np.cumsum(xw)
    xd = np.cumsum(xv)
    xe = np.cumsum(xd)

    vcd = np.cumsum(acd)
    vcs = np.cumsum(acs)
    vcc = np.cumsum(acc)

    dcd = np.cumsum(vcd)
    dcs = np.cumsum(vcs)
    dcc = np.cumsum(vcc)

    ecd = np.cumsum(dcd)
    ecs = np.cumsum(dcs)
    ecc = np.cumsum(dcc)

    rhs = -np.array([xv[-1], xd[-1], xe[-1]], dtype=float)
    mat = np.array([
        [vcd[-1], vcs[-1], vcc[-1]],
        [dcd[-1], dcs[-1], dcc[-1]],
        [ecd[-1], ecs[-1], ecc[-1]],
    ], dtype=float)

    coeff = np.linalg.solve(mat, rhs)
    comp = coeff[0] * acd + coeff[1] * acs + coeff[2] * acc
    xc = xw + comp

    return {
        "comp": comp,
        "xc": xc,
        "coefficients": coeff,
    }


def _inverse_real_spectrum(one_sided_spectrum: np.ndarray) -> np.ndarray:
    """
    Reconstruct a real time history from a one-sided complex spectrum.

    Parameters
    ----------
    one_sided_spectrum : ndarray
        One-sided complex spectrum of length `floor(N/2) + 1`.

    Returns
    -------
    ndarray
        Real-valued time history reconstructed by inverse FFT.

    Notes
    -----
    This matches the convention used by MATLAB ``rffti``: the omitted
    negative-frequency half of the spectrum is inferred by Hermitian
    symmetry.
    """
    X = np.asarray(one_sided_spectrum, dtype=complex).reshape(-1)
    n = len(X)

    if np.abs(np.imag(X[-1])) != 0:
        Y = np.concatenate([X, np.conj(X[n - 1:0:-1])])
    else:
        Y = np.concatenate([X, np.conj(X[n - 2:0:-1])])

    return np.fft.ifft(Y).real


def generate_single_windowed_random_realization(
    ref_srs: np.ndarray,
    window: np.ndarray,
    sample_rate: float,
    gravity: float = 386.0,
    damping: float = 0.03,
    srs_type: int = 9,
    match_points_per_octave: float = 4.0,
    output_points_per_octave: float = 12.0,
    amp_init: Optional[np.ndarray] = None,
    phase_init: Optional[np.ndarray] = None,
    iterations: int = 10,
    correction: float = 0.8,
    error_tolerance: float = 0.05,
    randomize_phase: bool = False,
    max_frequency: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Generate a single windowed random realization matched to a target SRS.

    Parameters
    ----------
    ref_srs : ndarray
        Target SRS breakpoint array with shape `(n_breakpoints, 2)`. The first
        column is frequency and the second is SRS amplitude.
    window : ndarray
        Time-domain window with shape `(n_samples,)`.
    sample_rate : float
        Sampling rate in samples per second.
    gravity : float, default=386.0
        Acceleration-of-gravity conversion factor used when integrating to
        velocity and displacement.
    damping : float, default=0.03
        Fraction of critical damping used for SRS calculations.
    srs_type : int, default=9
        SRS type passed to :func:`sdynpy_srs.srs`.
    match_points_per_octave : float, default=4.0
        Frequency resolution used during the iterative SRS matching process.
    output_points_per_octave : float, default=12.0
        Frequency resolution used for the returned SRS curve.
    amp_init : ndarray, optional
        Initial spectrum amplitude guess as a two-column array
        `[frequency, amplitude]`.
    phase_init : ndarray, optional
        Initial one-sided phase vector in radians.
    iterations : int, default=10
        Maximum number of matching iterations.
    correction : float, default=0.8
        Fraction of the SRS error corrected at each iteration.
    error_tolerance : float, default=0.05
        Relative SRS error tolerance for convergence.
    randomize_phase : bool, default=False
        If True, ignore `phase_init` and generate a random phase realization.
    max_frequency : float, optional
        Maximum frequency for nonzero Fourier amplitude. If not specified, the
        default is `min(sample_rate/4, ref_srs[-1, 0])`.
    rng : numpy.random.Generator, optional
        Random number generator used when random phase is required.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"time"`` : ndarray
            Time vector.
        - ``"xc"`` : ndarray
            Compensated windowed acceleration history.
        - ``"rr"`` : ndarray
            Stationary random acceleration history before windowing.
        - ``"srs_freq"`` : ndarray
            Frequency vector for the returned SRS.
        - ``"srs_values"`` : ndarray
            SRS values for the returned realization.
        - ``"srsc"`` : ndarray
            Matching-frequency SRS array with columns `[frequency, amplitude]`.
        - ``"metrics"`` : WindowedRandomMetrics
            Scalar metrics for the realization.
        - ``"amp"`` : ndarray
            Final matched spectral amplitudes at the matching frequencies.
        - ``"phase"`` : ndarray
            Final phase vector used for waveform synthesis.

    Raises
    ------
    ValueError
        If the FFT line spacing is too coarse to resolve the lowest matched SRS
        frequency, or if the internally defined frequency grid is invalid.

    Notes
    -----
    This function is the numerical core corresponding to one pass through the
    MATLAB ``wrcomp_tdh`` / ``gwinrand`` workflow.
    """
    ref_srs = np.asarray(ref_srs, dtype=float)
    window = np.asarray(window, dtype=float).reshape(-1)
    rng = np.random.default_rng() if rng is None else rng

    n = len(window)
    t = np.arange(n) / sample_rate

    fmin = ref_srs[0, 0]
    fmax = ref_srs[-1, 0]

    if max_frequency is None:
        max_frequency = min(sample_rate / 4.0, fmax)

    freq_match = octspace(fmin, fmax, match_points_per_octave).reshape(-1)

    if phase_init is None:
        block_size = int(2 ** np.ceil(np.log2(n)))
    else:
        block_size = 2 * (len(phase_init) - 1)

    delta_f = sample_rate / block_size
    if delta_f > freq_match[0]:
        raise ValueError(
            "FFT line spacing is too coarse to match the lowest SRS frequency. "
            f"delta_f={delta_f}, fmin={freq_match[0]}"
        )

    freq_line = np.arange(block_size // 2 + 1) * delta_f
    if len(freq_line) < 2:
        raise ValueError("Frequency line vector too short")
    freq_line[0] = freq_line[1] / 10.0

    if amp_init is not None and len(amp_init) > 0:
        amp_init = np.asarray(amp_init, dtype=float)
        amp_match = breakpoint_spectrum(freq_match, amp_init[:, 0], amp_init[:, 1])
    else:
        amp_match = None

    if amp_match is None:
        amp_line = 2 * np.pi * (1 / delta_f) * breakpoint_spectrum(
            freq_line, ref_srs[:, 0], ref_srs[:, 1]
        )
        amp_line[0] = 0.0
        amp_match = np.interp(freq_match, freq_line, amp_line, left=0.0, right=0.0) / freq_match
    else:
        amp_line = breakpoint_spectrum(freq_line, freq_match, amp_match)

    if randomize_phase or phase_init is None or len(phase_init) == 0:
        phase = 2 * np.pi * rng.random(len(amp_line))
        phase[0] = 0.0
        phase[-1] = 0.0
    else:
        phase = np.asarray(phase_init, dtype=float).reshape(-1)

    srs_required = breakpoint_spectrum(freq_match, ref_srs[:, 0], ref_srs[:, 1])

    spectrum = amp_line * np.exp(1j * phase)
    rr_full = _inverse_real_spectrum(spectrum)
    rr = rr_full[:n]
    xr = window * rr

    xc = window_compensation(xr, window, sample_rate, delay=0.0)["xc"][:n]

    srsc_values = srs(xc, 1.0 / sample_rate, freq_match, damping, srs_type)[0]
    srsc = np.column_stack([freq_match, srsc_values])

    maxline = np.where(freq_line > max_frequency)[0]
    amp_iter = amp_match.copy()

    for _ in range(1, iterations):
        amp_match = amp_iter + amp_iter * correction * (srs_required - srsc[:, 1]) / srsc[:, 1]
        amp_match = np.maximum(0.0, amp_match)
        amp_iter = amp_match.copy()

        f1, f2 = freq_match[0], freq_match[1]
        f3, f4 = freq_match[-2], freq_match[-1]
        freqq = np.concatenate([
            [0.0],
            [f1 + f1 / 2 - f2 / 2],
            freq_match,
            [f4 + f4 - f3],
            [sample_rate / 2.0],
        ])
        ampp = np.concatenate([
            [amp_match[0] / 1000.0],
            [amp_match[0]],
            amp_match,
            [amp_match[-1]],
            [amp_match[-1] / 1000.0],
        ])

        amp_line = np.interp(freq_line, freqq, ampp, left=0.0, right=0.0)
        amp_line[maxline] = 0.0

        nz = np.abs(amp_line[np.abs(amp_line) > 0])
        if len(nz):
            amp_line = np.maximum(0.01 * np.min(nz), amp_line)

        spectrum = amp_line * np.exp(1j * phase)
        rr_full = _inverse_real_spectrum(spectrum)
        rr = rr_full[:n]
        xr = window * rr

        xc = window_compensation(xr, window, sample_rate, delay=0.0)["xc"][:n]
        srsc[:, 1] = srs(xc, 1.0 / sample_rate, freq_match, damping, srs_type)[0]

        srs_error = (srsc[:, 1] - srs_required) / srs_required
        if np.max(np.abs(srs_error)) < error_tolerance:
            break

    srs_freq = octspace(fmin, fmax, output_points_per_octave)
    srs_values = srs(xc, 1.0 / sample_rate, srs_freq, damping, srs_type)[0]

    vc = (gravity / sample_rate) * cumulative_trapezoid(xc, initial=0.0)
    dc = (1.0 / sample_rate) * cumulative_trapezoid(vc, initial=0.0)

    sref_interp = breakpoint_spectrum(srsc[:, 0], ref_srs[:, 0], ref_srs[:, 1])
    db_error = 20.0 * np.log10(srsc[:, 1] / sref_interp)
    err = lanl_rms_db_error(db_error, srsc[:, 0], output_points_per_octave)
    tm = temporal_moments(xc, 1.0 / sample_rate)

    metrics = WindowedRandomMetrics(
        lanl_rms_db_error=err["rms"],
        energy=tm["energy"],
        rea=tm["rea"],
        peak_accel=float(np.max(np.abs(xc))),
        peak_vel=float(np.max(np.abs(vc))),
        peak_disp=float(np.max(np.abs(dc))),
    )

    return {
        "time": t,
        "xc": xc,
        "rr": rr,
        "srs_freq": srs_freq,
        "srs_values": srs_values,
        "srsc": srsc,
        "metrics": metrics,
        "amp": amp_match,
        "phase": phase,
    }


def generate_windowed_random(
    ref_srs: np.ndarray,
    window: np.ndarray,
    sample_rate: float,
    gravity: float = 386.0,
    damping: float = 0.03,
    srs_type: int = 9,
    match_points_per_octave: float = 4.0,
    output_points_per_octave: float = 12.0,
    n_realizations: int = 1,
    amp_init: Optional[np.ndarray] = None,
    phase_init: Optional[np.ndarray] = None,
    iterations: int = 10,
    correction: float = 0.8,
    error_tolerance: float = 0.05,
    randomize_phase: bool = False,
    max_frequency: Optional[float] = None,
    seed: Optional[int] = None,
) -> WindowedRandomResults:
    """
    Generate an ensemble of windowed random time histories matched to a target SRS.

    Parameters
    ----------
    ref_srs : ndarray
        Target SRS breakpoint array with shape `(n_breakpoints, 2)`. The first
        column is frequency and the second is amplitude.
    window : ndarray
        Time-domain window with shape `(n_samples,)`.
    sample_rate : float
        Sampling rate in samples per second.
    gravity : float, default=386.0
        Acceleration-of-gravity conversion factor used when integrating to
        velocity and displacement.
    damping : float, default=0.03
        Fraction of critical damping used for SRS calculations.
    srs_type : int, default=9
        SRS type passed to :func:`sdynpy_srs.srs`.
    match_points_per_octave : float, default=4.0
        Frequency resolution used for SRS matching during iteration.
    output_points_per_octave : float, default=12.0
        Frequency resolution used for the returned SRS curves.
    n_realizations : int, default=1
        Number of random realizations to generate.
    amp_init : ndarray, optional
        Initial spectrum amplitude guess as a two-column array
        `[frequency, amplitude]`.
    phase_init : ndarray, optional
        Initial one-sided phase vector in radians.
    iterations : int, default=10
        Maximum number of matching iterations per realization.
    correction : float, default=0.8
        Fraction of the SRS error corrected at each iteration.
    error_tolerance : float, default=0.05
        Relative SRS error tolerance for convergence.
    randomize_phase : bool, default=False
        If True, ignore `phase_init` and randomize the phase realization.
    max_frequency : float, optional
        Maximum frequency for nonzero Fourier amplitude. If not specified, the
        default is `min(sample_rate/4, ref_srs[-1, 0])`.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    WindowedRandomResults
        Ensemble result object containing all realizations, their SRS curves,
        and selection metrics.

    Notes
    -----
    Each realization is generated independently using the same configuration
    except for the random phase realization when randomization is enabled.
    """
    ref_srs = np.asarray(ref_srs, dtype=float)
    window = np.asarray(window, dtype=float).reshape(-1)

    rng = np.random.default_rng(seed)
    t = np.arange(len(window)) / sample_rate

    xc_list = []
    rr_list = []
    srs_list = []
    metrics = []
    srs_freq = None

    for _ in range(n_realizations):
        out = generate_single_windowed_random_realization(
            ref_srs=ref_srs,
            window=window,
            sample_rate=sample_rate,
            gravity=gravity,
            damping=damping,
            srs_type=srs_type,
            match_points_per_octave=match_points_per_octave,
            output_points_per_octave=output_points_per_octave,
            amp_init=amp_init,
            phase_init=phase_init,
            iterations=iterations,
            correction=correction,
            error_tolerance=error_tolerance,
            randomize_phase=randomize_phase,
            max_frequency=max_frequency,
            rng=rng,
        )

        xc_list.append(out["xc"])
        rr_list.append(out["rr"])
        srs_list.append(out["srs_values"])
        metrics.append(out["metrics"])

        if srs_freq is None:
            srs_freq = out["srs_freq"]

    return WindowedRandomResults(
        time=t,
        sample_rate=sample_rate,
        gravity=gravity,
        window=window,
        ref_srs=ref_srs,
        srs_frequency=srs_freq,
        srs_matrix=np.column_stack(srs_list),
        xc_matrix=np.column_stack(xc_list),
        rr_matrix=np.column_stack(rr_list),
        metrics=metrics,
        metadata={
            "damping": damping,
            "srs_type": srs_type,
            "match_points_per_octave": match_points_per_octave,
            "output_points_per_octave": output_points_per_octave,
            "iterations": iterations,
            "correction": correction,
            "error_tolerance": error_tolerance,
            "randomize_phase": randomize_phase,
            "max_frequency": max_frequency,
            "n_realizations": n_realizations,
        },
    )


def select_realization(
    results: WindowedRandomResults,
    criterion: str = "lanl_rms_db_error",
    index: Optional[int] = None,
) -> SelectedRealization:
    """
    Select a realization from an ensemble.

    Parameters
    ----------
    results : WindowedRandomResults
        Ensemble results returned by :func:`generate_windowed_random`.
    criterion : str, default="lanl_rms_db_error"
        Name of the metric field to minimize if `index` is not provided.
    index : int, optional
        Explicit zero-based realization index to select. If provided, this
        takes precedence over `criterion`.

    Returns
    -------
    SelectedRealization
        Selected realization containing the time history, SRS, and scalar
        metrics.

    Raises
    ------
    AttributeError
        If `criterion` is not a valid attribute of
        :class:`WindowedRandomMetrics`.

    Notes
    -----
    This function provides the deterministic equivalent of the MATLAB
    ``gwinrand_select`` helper without interactive point-picking.
    """
    if index is None:
        values = np.array([getattr(m, criterion) for m in results.metrics], dtype=float)
        index = int(np.argmin(values))

    return SelectedRealization(
        index=index,
        time=results.time,
        xc=results.xc_matrix[:, index],
        rr=results.rr_matrix[:, index],
        srs=np.column_stack([results.srs_frequency, results.srs_matrix[:, index]]),
        metrics=results.metrics[index],
    )