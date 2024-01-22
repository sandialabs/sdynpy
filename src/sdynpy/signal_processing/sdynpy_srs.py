# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 09:23:59 2023

@author: dprohe
"""

import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from scipy.signal import oaconvolve
from scipy.optimize import minimize, NonlinearConstraint, nnls

def srs(signal, dt, frequencies = None, damping = 0.05, spectrum_type = 9,
                b_filter_weights = None, a_filter_weights = None):
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
    
    if b_filter_weights.shape != (frequencies.size,3):
        raise ValueError('`b_filter_weights` and `a_filter_weights` must have shape (frequencies.size,3)')
    
    srs_output = np.zeros(  signal.shape[:-1]
                          + ((9,) if np.abs(spectrum_type) == 10 else ())
                          + (frequencies.size,))
    for i_freq,(frequency,b,a) in enumerate(zip(
            frequencies,b_filter_weights,a_filter_weights)):
        # Append enough zeros to get 1/100 of the residual response
        zeros_to_append = int(np.max([2,np.ceil(0.01*sample_rate/frequency)]))
        signals_extended = np.concatenate((
            signal, np.zeros(signal.shape[:-1] + (zeros_to_append,))),axis=-1)
        
        # Filter the signal
        response = sdof_filter(b,a,signals_extended)
        
        # Now compute the SRS parameters
        # Primary response
        primary_maximum = np.array(np.max(response[...,:num_samples], axis=-1))
        primary_maximum[primary_maximum < 0] = 0
        primary_minimum = np.array(np.abs(np.min(response[...,:num_samples], axis=-1)))
        primary_minimum[primary_minimum < 0] = 0
        primary_abs = np.array(np.max([primary_maximum,primary_minimum],axis=0))
        
        # Now compute the residual response. we need time steps and 
        # responses at two points
        residual_times = [0,(zeros_to_append-1)/sample_rate]
        residual_responses = np.concatenate([response[...,num_samples,np.newaxis],
                                             response[...,num_samples+zeros_to_append-1,np.newaxis]],
                                            axis=-1)
        
        # Compute the peak response after the structure has stopped responding
        peak_residual_responses = sdof_free_decay_peak_response(
            residual_responses, residual_times, frequency, damping)
        
        # Pull out the residual peaks
        residual_maximum = np.array(np.max(peak_residual_responses, axis=-1))
        residual_maximum[residual_maximum < 0] = 0
        residual_minimum = np.array(np.abs(np.min(peak_residual_responses, axis=-1)))
        residual_minimum[residual_minimum < 0] = 0
        residual_abs = np.array(np.max([residual_maximum,residual_minimum],axis=0))
        
        all_srs_this_frequency_line = np.array([
            primary_maximum, primary_minimum, primary_abs,
            residual_maximum, residual_minimum, residual_abs,
            np.max([primary_maximum,residual_maximum],axis=0),
            np.max([primary_minimum,residual_minimum],axis=0),
            np.max([primary_abs,residual_abs],axis=0)])

        if abs_type < 10:
            srs_output[...,i_freq] = all_srs_this_frequency_line[abs_type-1]
        else:
            srs_output[...,i_freq] = np.moveaxis(all_srs_this_frequency_line,0,-1)
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
    a[small_freq_indices,1] = (
          2*damping*small_freqs
        + small_freqs**2*(1-2*damping**2)
        )
    # -2*z*w + 2*z*z*w*w where z is damping and w is normalized frequencies
    a[small_freq_indices,2] = (
          -2*damping*small_freqs
        + 2*damping**2*small_freqs**2
        )
    
    if spectrum_type > 0:
        # Absolute Acceleration Model
        # z*w + (w*w)*( (1/6) - 2*z*z/3 );
        b[small_freq_indices,0] = (
              damping*small_freqs
            + small_freqs**2
            * (1/6 - 2*damping**2/3)
            )
        # 2*w*w*(1-z*z)/3
        b[small_freq_indices,1] = (
            2*small_freqs**2*(1-damping**2)/3
            )
        # -z*w + w*w*( (1/6) - 4*z*z/3 );
        b[small_freq_indices,2] = (
            - damping*small_freqs
            + small_freqs**2
            * (1/6 - 4*damping**2/3)
            )
    else:
        # Relative Displacement Model
        # -w*w/6;
        b[small_freq_indices,0] = -small_freqs**2/6
        # -2*w*w/3;
        b[small_freq_indices,1] = -2*small_freqs**2/3
        # -w*w/6;
        b[small_freq_indices,2] = -small_freqs**2/6
    
    # Large frequencies
    large_freqs = normalized_frequencies[~small_freq_indices]
    
    # Define some helper variables
    sq = np.sqrt(1-damping**2)
    e = np.exp(-damping*large_freqs)
    wd = large_freqs*sq
    sp = e*np.sin(wd)
    fact = (2*damping**2 - 1)*sp/sq
    c = e*np.cos(wd)
    
    a[~small_freq_indices,1] = 2*(1-c)
    a[~small_freq_indices,2] = -1 + e**2
    
    if spectrum_type > 0:
        # Absolute Acceleration Model
        spwd = sp/wd
        b[~small_freq_indices,0] = 1-spwd
        b[~small_freq_indices,1] = 2*(spwd - c)
        b[~small_freq_indices,2] = e**2 - spwd
    else:
        # Relative Displacement Model
        b[~small_freq_indices,0] = -(2*damping*(c-1) + fact + large_freqs)/large_freqs
        b[~small_freq_indices,1] = -( -2*c*large_freqs + 2*damping*(1-e**2) - 2*fact)/large_freqs
        b[~small_freq_indices,2] = -(e**2*(large_freqs+2*damping) - 2*damping*c + fact)/large_freqs
    
    return b,a

def sdof_filter(b,a,signal,zi = None):
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
    aa = np.array([a[0], a[1]-2, a[2] +1])
    return lfilter(b,aa,signal,axis=-1,zi = zi)

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
    c = e[:,np.newaxis]*np.array((sd,cd)).T
    z = np.linalg.solve(c,responses[...,np.newaxis])[...,0]
    
    # the response has the form
    # a(t) = exp(-zeta wn t)[z[0]sin(wd t) + z[1]cos(wd t)]
    # the first maximum response will occur at the time tmax
    tmax = np.array((1/wd) * np.arctan( (wd*z[...,0] - damping*wn*z[...,1])/(wd*z[...,1] - damping*wn*z[...,0]) ))
    tmax[tmax < 0] = tmax[tmax < 0] + np.pi/wd
    tmax[tmax > np.pi/wd] = tmax[tmax > np.pi/wd] - np.pi/wd
       
    # We can now plug it into the equation to find the maximum
    response_peaks = []
    response_peaks.append(np.exp(-damping*wn*tmax)*( z[...,0]*np.sin(wd*tmax) + z[...,1]*np.cos(wd*tmax)))
    tmin = tmax + np.pi/wd
    response_peaks.append(np.exp(-damping*wn*tmin)*( z[...,0]*np.sin(wd*tmin) + z[...,1]*np.cos(wd*tmin)))
    return np.moveaxis(response_peaks,0,-1)

def octspace(low,high,points_per_octave):
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
                      sine_frequencies = None, sine_tone_range = None, sine_tone_per_octave = None,
                      sine_amplitudes = None, sine_decays = None, sine_delays = None,
                      required_srs = None, srs_breakpoints = None,
                      srs_damping = 0.05, srs_type = 9,
                      compensation_frequency = None, compensation_decay = 0.95,
                      # Paramters for the iteration
                      number_of_iterations = 3, convergence = 0.8, 
                      error_tolerance = 0.05,
                      tau = None, num_time_constants = None, decay_resolution = None,
                      scale_factor = 1.02, 
                      acceleration_factor = 1.0,
                      plot_results = False, srs_frequencies = None,
                      verbose = False
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
        sine_frequencies = octspace(sine_tone_range[0],sine_tone_range[1],
                                    sine_tone_per_octave)
    # Now set up the SRS
    if required_srs is None and srs_breakpoints is None:
        raise ValueError('Either `required_srs` or `srs_breakpoints` must be specified')
    if required_srs is not None and srs_breakpoints is not None:
        raise ValueError('`required_srs` can not be specified simultaneously with `srs_breakpoints`')
    if required_srs is None:
        required_srs = loginterp(sine_frequencies,
                                 srs_breakpoints[:,0],
                                 srs_breakpoints[:,1])
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
        if isinstance(num_time_constants,dict):
            tau = {}
            for freq_range, num_time_constant in num_time_constants.items():
                tau[freq_range] = period / num_time_constant
        else:
            tau = period/num_time_constants
    if tau is not None:
        sine_decays = []
        for freq in sine_frequencies:
            if isinstance(tau,dict):
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
         convergence, error_tolerance, verbose)
    
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
        fig, ax = plt.subplots(2,2,figsize=(8,6))
        times = np.arange(block_size)/sample_rate
        ax[0,0].plot(times, acceleration_signal)
        ax[0,0].set_ylabel('Acceleration')
        ax[0,0].set_xlabel('Time (s)')
        ax[0,1].plot(times, velocity_signal)
        ax[0,1].set_ylabel('Velocity')
        ax[0,1].set_xlabel('Time (s)')
        ax[1,0].plot(times, displacement_signal)
        ax[1,0].set_ylabel('Displacement')
        ax[1,0].set_xlabel('Time (s)')
        # Compute SRS
        if srs_frequencies is None:
            srs_frequencies = sine_frequencies
        this_srs,this_frequencies = srs(
            acceleration_signal, 1/sample_rate, srs_frequencies,
            srs_damping, srs_type)
        if srs_breakpoints is None:
            srs_abscissa = sine_frequencies
            srs_ordinate = required_srs
        else:
            srs_abscissa, srs_ordinate = srs_breakpoints.T
        ax[1,1].plot(srs_abscissa, srs_ordinate, 'k--')
        ax[1,1].plot(this_frequencies, this_srs)
        ax[1,1].set_ylabel('SRS ({:0.2f}% damping)'.format(srs_damping*100))
        ax[1,1].set_xlabel('Frequency (Hz)')
        ax[1,1].set_yscale('log')
        ax[1,1].set_xscale('log')
        fig.tight_layout()
        return_vals += (fig,ax)
        ax[1,1].legend(('Reference','Decayed Sine'))
    return return_vals
    

def _sum_decayed_sines(sine_frequencies, sine_amplitudes, 
                      sine_decays, sine_delays,
                      required_srs, 
                      compensation_frequency, compensation_decay,
                      sample_rate, block_size, 
                      srs_damping, srs_type,
                      number_of_iterations = 3, convergence = 0.8,
                      error_tolerance = 0.05, verbose = False):
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
                number_of_iterations = 10, convergence = convergence,
                error_tolerance = error_tolerance, verbose = verbose))
        # get the SRS error at each frequency term by first computing the signal
        (this_signal, this_compensation_frequency, this_compensation_amplitude,
         this_compensation_decay, this_compensation_delay) = sum_decayed_sines_reconstruction_with_compensation(
            sine_frequencies, this_sine_amplitudes, sine_decays, sine_delays,
            compensation_frequency, compensation_decay, sample_rate, block_size)
        # Then computing the SRS of the signal
        this_srs = srs(this_signal, 1/sample_rate, sine_frequencies, srs_damping,
                       srs_type)[0]
        srs_error = (this_srs - required_srs)/required_srs
        sine_amplitudes = this_sine_amplitudes
    if verbose:
        print('Sine Table after Iterations:')
        print('{:>8s}, {:>10s}, {:>10s}, {:>10s},  {:>10s}, {:>10s}'.format(
            'Sine','Frequency','Amplitude','SRS Val','SRS Req','Error'))
        for i,(freq, amp, srs_val, srs_req, srs_err) in enumerate(zip(
                sine_frequencies, sine_amplitudes, this_srs, required_srs,
                srs_error)):
            print('{:>8s}, {:>10.4f}, {:>10.4f}, {:>10.4f}, {:>10.4f}, {:>10.4f}'.format(
                str(i),freq,amp,srs_val,srs_req,srs_err))
        print('Compensating Pulse:')
        print('{:>10s}, {:>10s}, {:>10s}, {:>10s}'.format(
            'Frequency','Amplitude','Decay','Delay'))
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
                                        number_of_iterations = 10, convergence = 0.8,
                                        error_tolerance = 0.05,
                                        verbose = False):
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
    b,a = sdof_ramp_invariant_filter_weights(sine_frequencies, sample_rate, 
                                             damping_srs, srs_type)
    # Copy the arrays so we don't overwrite anything
    sine_frequencies = np.array(sine_frequencies).copy()
    sine_amplitudes = np.array(sine_amplitudes).copy()
    sine_decays = np.array(sine_decays).copy()
    sine_delays = np.array(sine_delays).copy()
    # Iterate one frequency at a time
    for i,(frequency,amplitude,decay,delay,desired_srs) in enumerate(zip(
            sine_frequencies, sine_amplitudes, sine_decays, sine_delays,
            required_srs)):
        # if i == 10:
        #     break
        srs_error = float('inf')
        iteration_count = 1
        increment = 0.1
        # Get the pulse without this frequency line
        other_sine_frequencies = np.concatenate((sine_frequencies[:i],sine_frequencies[i+1:]))
        other_sine_amplitudes = np.concatenate((sine_amplitudes[:i],sine_amplitudes[i+1:]))
        other_sine_decays = np.concatenate((sine_decays[:i],sine_decays[i+1:]))
        other_sine_delays = np.concatenate((sine_delays[:i],sine_delays[i+1:]))
        other_pulse = sum_decayed_sines_reconstruction(
            other_sine_frequencies, other_sine_amplitudes, other_sine_decays,
            other_sine_delays, sample_rate, block_size)
        # Now iterate at the single frequency until convergence
        while abs(srs_error) > error_tolerance:
            sine_amplitudes[i] = amplitude
            # Find the pulse at this frequency line
            this_pulse = sum_decayed_sines_reconstruction(
                frequency,amplitude,decay,delay, sample_rate, block_size)
            # Get the compensating pulse
            compensation_amplitude, compensation_delay = sum_decayed_sines_compensating_pulse_parameters(
                sine_frequencies, sine_amplitudes, sine_decays, sine_delays, compensation_frequency, compensation_decay)
            # Find the number of samples to shift the waveform
            num_shift = -int(np.floor(compensation_delay*sample_rate))
            if abs(num_shift) >= block_size:
                raise ValueError('The number of samples in the compensation delay ({:}) is larger than the block_size ({:}).  The entire pulse will consist of part of the compensation pulse.  Plase increase the block_size or compensation frequency.'.format(num_shift, block_size))
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
                srs_type, b[i], a[i])[0][0] # only get the SRS and there should only be one value due to one frequency line
            srs_error = (srs_prediction - desired_srs)/desired_srs
            if verbose:
                print('Iteration at frequency {:}, {:0.4f}\n  Iteration Count: {:}, SRS Error: {:0.4f}'.format(i,frequency,iteration_count,srs_error))
            # Now we're going to compute the same thing again with a perturbed
            # amplitude, this will allow us to compute the slope change at the
            # current amplitude
            amplitude_change = np.sign(srs_error)*increment*np.sign(amplitude)*Amax
            # Now check and see if we need to modify the amplitude
            if abs(srs_error) > error_tolerance:
                if amplitude_change == 0: # perturb it a bit
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
                    frequency,new_amplitude, decay, delay, sample_rate, block_size)
                # Get the compensating pulse
                compensation_amplitude, compensation_delay = sum_decayed_sines_compensating_pulse_parameters(
                    sine_frequencies, sine_amplitudes, sine_decays, sine_delays, compensation_frequency, compensation_decay)
                # Find the number of samples to shift the waveform
                num_shift = -int(np.floor(compensation_delay*sample_rate))
                if abs(num_shift) >= block_size:
                    raise ValueError('The number of samples in the compensation delay ({:}) is larger than the block_size ({:}).  The entire pulse will consist of part of the compensation pulse.  Plase increase the block_size or compensation frequency.'.format(num_shift, block_size))
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
                    srs_type, b[i], a[i])[0][0] # only get the SRS and there should only be one value due to one frequency line
                # Get slope of correction
                correction_slope = (abs(amplitude)-abs(new_amplitude))/(srs_prediction-srs_perturbed)
                if correction_slope > 0:
                    # this is what we want, it means we are changing in the right
                    # direction
                    new_amplitude = convergence*correction_slope*(desired_srs-srs_prediction) + abs(amplitude)
                    # Never let the change be too large
                    amplitude_check = max([abs(amplitude),Amax/10])
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
                        frequency,amplitude, decay, delay, sample_rate, block_size)
                    # Get the compensating pulse
                    compensation_amplitude, compensation_delay = sum_decayed_sines_compensating_pulse_parameters(
                        sine_frequencies, sine_amplitudes, sine_decays, sine_delays, compensation_frequency, compensation_decay)
                    # Find the number of samples to shift the waveform
                    num_shift = -int(np.floor(compensation_delay*sample_rate))
                    if abs(num_shift) >= block_size:
                        raise ValueError('The number of samples in the compensation delay ({:}) is larger than the block_size ({:}).  The entire pulse will consist of part of the compensation pulse.  Plase increase the block_size or compensation frequency.'.format(num_shift, block_size))
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
                        srs_type, b[i], a[i])[0][0] # only get the SRS and there should only be one value due to one frequency line
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
                    print('  Warning: SRS did not converge for frequency {:}: {:0.4f}'.format(i,frequency))
                    break
    compensation_amplitude, compensation_delay = sum_decayed_sines_compensating_pulse_parameters(
        sine_frequencies, sine_amplitudes, sine_decays, sine_delays, compensation_frequency, compensation_decay)
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
    var1 = (np.sum((2*sine_decays*sine_amplitudes)[np.newaxis,:]@np.linalg.pinv((omegas**2*(sine_decays**2 + 1)**2)[np.newaxis,:])))/compensation_amplitude
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
    this_times = times[:,np.newaxis] - sine_delays
    response = sine_amplitudes*np.exp(-sine_decays*omegas*this_times)*np.sin(omegas*this_times)
    response[this_times < 0] = 0
    return np.sum(response,axis=-1)

def sum_decayed_sines_reconstruction_with_compensation(
        sine_frequencies, sine_amplitudes,
        sine_decays, sine_delays, compensation_frequency, compensation_decay,
        sample_rate,block_size):
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
    compensation_amplitude, compensation_delay = sum_decayed_sines_compensating_pulse_parameters(
        sine_frequencies, sine_amplitudes, sine_decays, sine_delays,
        compensation_frequency, compensation_decay)
    sine_frequencies = np.concatenate((sine_frequencies,[compensation_frequency]))
    sine_amplitudes = np.concatenate((sine_amplitudes,[compensation_amplitude]))
    sine_delays = np.concatenate((sine_delays,[compensation_delay]))
    sine_decays = np.concatenate((sine_decays,[compensation_decay]))
    signal = sum_decayed_sines_reconstruction(
        sine_frequencies, sine_amplitudes, sine_decays, sine_delays,
        sample_rate, block_size)
    return (signal, compensation_frequency, compensation_amplitude,
            compensation_decay, compensation_delay)

def sum_decayed_sines_displacement_velocity(
        sine_frequencies, sine_amplitudes,
        sine_decays, sine_delays, sample_rate,block_size,
        acceleration_factor = 1):
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
    tmp1=x.copy()
    tmp2=x.copy()
    tmp3=x.copy()
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
        indices = t-tau[k]>0   # index's where vel and disp are evaluated
        Awz1=A[k]/(w[k]*zp1[k])
        tmp1[indices] = Awz1*np.exp(-zw[k]*(t[indices]-tau[k]))
        tmp2[indices] = z[k]*np.sin(w[k]*(t[indices]-tau[k])) + np.cos(w[k]*(t[indices]-tau[k]))
        v[indices] = v[indices] - tmp1[indices]*tmp2[indices] +Awz1*x1[indices]
        Awz2 = A[k]/(w2[k]*zp1[k]*zp1[k])
        tmp1[indices] = Awz2*np.exp(-zw[k]*(t[indices]-tau[k]))
        tmp2[indices] = zm1[k]*np.sin(w[k]*(t[indices]-tau[k])) + 2*z[k]*np.cos(w[k]*(t[indices]-tau[k]))
        tmp3[indices] = Awz1*(t[indices]-tau[k])
        tmp4 = 2*z[k]*Awz2
        d[indices] = d[indices] + tmp1[indices]*tmp2[indices] + tmp3[indices] - tmp4
    return v,d

def loginterp(x,xp,fp):
    return 10**np.interp(np.log10(x),np.log10(xp),np.log10(fp))

def optimization_error_function(
        amplitude_lin_scales, sample_rate, block_size,
        sine_frequencies, sine_amplitudes, sine_decays, sine_delays, 
        compensation_frequency, compensation_decay, control_irfs, limit_irfs,
        srs_damping, srs_type, control_srs, control_weights, limit_srs, 
        b = None, a = None, frequency_index = None):
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
    control_responses = np.array([oaconvolve(control_irf, input_signal) for control_irf in control_irfs])[...,:block_size]
    # Compute SRSs at all frequencies
    predicted_control_srs,frequencies = srs(control_responses, 1/sample_rate, sine_frequencies[frequency_index], 
                                            srs_damping, srs_type, 
                                            None if b is None else b[frequency_index],
                                            None if a is None else a[frequency_index])
    # Compute error
    mean_control_error = np.mean(
            (control_weights[:,np.newaxis]
             *((predicted_control_srs - control_srs[...,frequency_index])
               /control_srs[...,frequency_index])))
    if limit_srs is not None:
        limit_responses = np.array([oaconvolve(limit_irf, input_signal) for limit_irf in limit_irfs])[...,:block_size]
        predicted_limit_srs,frequencies = srs(limit_responses, 1/sample_rate, sine_frequencies[frequency_index], 
                                              srs_damping, srs_type, 
                                              None if b is None else b[frequency_index], 
                                              None if a is None else a[frequency_index])
        max_limit_error = np.max(
            ((predicted_limit_srs - limit_srs[...,frequency_index])
             /limit_srs[...,frequency_index]))
        if max_limit_error >= 0 and mean_control_error >= 0:
            srs_error = np.max((mean_control_error,max_limit_error))
        elif max_limit_error <= 0 and mean_control_error <= 0:
            srs_error = np.max((mean_control_error,max_limit_error))
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

def optimization_callback(intermediate_result, rms_error_threshold = 0.02,
                          verbose = True):
    if verbose:
        print('Amplitude Scale: {:}, error is {:}'.format(intermediate_result.x.squeeze(),
                                                       intermediate_result.fun))
    if rms_error_threshold is not None:
        if intermediate_result.fun < rms_error_threshold:
            raise StopIteration

def sum_decayed_sines_minimize(sample_rate, block_size, 
                               sine_frequencies = None, sine_tone_range = None, sine_tone_per_octave = None,
                               sine_amplitudes = None, sine_decays = None, sine_delays = None,
                               control_srs = None, control_breakpoints = None,
                               srs_damping = 0.05, srs_type = 9,
                               compensation_frequency = None, compensation_decay = 0.95,
                               # Parameters for defining decays
                               tau = None, num_time_constants = None, decay_resolution = None,
                               scale_factor = 1.02, 
                               acceleration_factor = 1.0,
                               # Parameters for imposing limits
                               limit_breakpoints = None, limit_transfer_functions = None,
                               control_transfer_functions = None, control_weights = None,
                               # Parameters for the optimizer
                               minimize_iterations = 1, rms_error_threshold = None,
                               optimization_passes = 3,
                               plot_results = False, verbose = False
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
        sine_frequencies = octspace(sine_tone_range[0],sine_tone_range[1],
                                    sine_tone_per_octave)
    # Now set up the SRS
    if control_srs is None and control_breakpoints is None:
        raise ValueError('Either `control_srs` or `control_breakpoints` must be specified')
    if control_srs is not None and control_breakpoints is not None:
        raise ValueError('`control_srs` can not be specified simultaneously with `control_breakpoints`')
    if control_srs is None:
        frequencies = control_breakpoints[:,0]
        breakpoint_curves = control_breakpoints[:,1:].T
        control_srs = np.array([loginterp(sine_frequencies,
                                          frequencies,
                                          breakpoint_curve) for breakpoint_curve in breakpoint_curves])
    else:
        control_srs = np.atleast_2d(control_srs)
    if control_transfer_functions is None:
        control_transfer_functions = np.ones(((control_srs.shape[0]-1)*2,block_size//2+1))
    tf_frequencies = np.fft.rfftfreq(control_transfer_functions.shape[-1]*2-1,1/sample_rate)
    if sine_amplitudes is None:
        tf_at_frequencies = np.array([
            np.interp(sine_frequencies,tf_frequencies,np.abs(control_transfer_function))
            for control_transfer_function in control_transfer_functions])
        srs_amplitudes = np.array([nnls(ai[:,np.newaxis],bi)[0][0] for ai,bi in zip(tf_at_frequencies.T,control_srs.T)])
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
        if isinstance(num_time_constants,dict):
            tau = {}
            for freq_range, num_time_constant in num_time_constants.items():
                tau[freq_range] = period / num_time_constant
        else:
            tau = period/num_time_constants
    if tau is not None:
        sine_decays = []
        for freq in sine_frequencies:
            if isinstance(tau,dict):
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
        frequencies = limit_breakpoints[:,0]
        breakpoint_curves = limit_breakpoints[:,1:].T
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
    b,a = sdof_ramp_invariant_filter_weights(sine_frequencies, sample_rate, srs_damping, srs_type)
    
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
            error_function = lambda x: optimization_error_function(
                x, sample_rate, block_size,
                sine_frequencies, sine_amplitudes, sine_decays, sine_delays, 
                compensation_frequency, compensation_decay, control_irfs, limit_irfs,
                srs_damping, srs_type, control_srs, control_weights, limit_srs, 
                b, a, frequency_index = [i])[0]
        
            callback = lambda intermediate_result: optimization_callback(intermediate_result, rms_error_threshold, 
                                                       verbose)
            optimization_result = minimize(error_function, 
                                           np.ones(1),
                                           method = 'Powell',
                                            bounds = [(0,np.inf)],
                                           callback=callback,
                                           options = {'maxiter':minimize_iterations})
        
            # Populate the amplitudes with the updated result
            if verbose:
                print('Initial Amplitude: {:}, Updated Amplitude: {:}\n'.format(sine_amplitudes[i],sine_amplitudes[i]*optimization_result.x.squeeze()))
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
        fig, ax = plt.subplots(2,2, figsize=(8,6))
        times = np.arange(block_size)/sample_rate
        ax[0,0].plot(times, input_signal)
        ax[0,0].set_ylabel('Acceleration')
        ax[0,0].set_xlabel('Time (s)')
        ax[0,1].plot(times, velocity_signal)
        ax[0,1].set_ylabel('Velocity')
        ax[0,1].set_xlabel('Time (s)')
        ax[1,0].plot(times, displacement_signal)
        ax[1,0].set_ylabel('Displacement')
        ax[1,0].set_xlabel('Time (s)')
        # Compute SRS
        this_srs,this_frequencies = srs(
            input_signal, 1/sample_rate, sine_frequencies,
            srs_damping, srs_type)
        if control_breakpoints is None:
            srs_abscissa = sine_frequencies
            srs_ordinate = control_srs.T
        else:
            srs_abscissa = control_breakpoints[:,0]
            srs_ordinate = control_breakpoints[:,1:]
        ax[1,1].plot(srs_abscissa, srs_ordinate, 'k--')
        ax[1,1].plot(this_frequencies, this_srs)
        ax[1,1].set_ylabel('SRS ({:0.2f}% damping)'.format(srs_damping*100))
        ax[1,1].set_xlabel('Frequency (Hz)')
        ax[1,1].set_yscale('log')
        ax[1,1].set_xscale('log')
        ax[1,1].legend(('Reference','Decayed Sine'))
        fig.tight_layout()
        return_vals += (fig,ax)
        # Now plot all control channels
        for i,(predicted, control, response) in enumerate(zip(predicted_control_srs, control_srs, control_responses)):
            fig, ax = plt.subplots(1,2, figsize=(8,3))
            ax[0].set_title('Control Signal {:}'.format(i))
            ax[0].plot(times,response)
            ax[0].set_label('Acceleration')
            ax[0].set_xlabel('Time (s)')
            ax[1].set_title('Control SRS {:}'.format(i))
            ax[1].plot(sine_frequencies, control, 'k--')
            ax[1].plot(sine_frequencies, predicted)
            ax[1].set_ylabel('SRS ({:0.2f}% damping)'.format(srs_damping*100))
            ax[1].set_xlabel('Frequency (Hz)')
            ax[1].set_yscale('log')
            ax[1].set_xscale('log')
            ax[1].legend(('Desired','Achieved'))
            fig.tight_layout()
            return_vals += (fig,ax)
        # Now plot all limit channels
        if limit_srs is not None:
            for i,(predicted,limit, response) in enumerate(zip(predicted_limit_srs, limit_srs, limit_responses)):
                fig, ax = plt.subplots(1,2, figsize=(8,3))
                ax[0].set_title('Limit Signal {:}'.format(i))
                ax[0].plot(times,response)
                ax[0].set_label('Acceleration')
                ax[0].set_xlabel('Time (s)')
                ax[1].set_title('Limit SRS {:}'.format(i))
                ax[1].plot(sine_frequencies, limit, 'k--')
                ax[1].plot(sine_frequencies, predicted)
                ax[1].set_ylabel('SRS ({:0.2f}% damping)'.format(srs_damping*100))
                ax[1].set_xlabel('Frequency (Hz)')
                ax[1].set_yscale('log')
                ax[1].set_xscale('log')
                ax[1].legend(('Desired','Achieved'))
                fig.tight_layout()
                return_vals += (fig,ax)
                
    return return_vals