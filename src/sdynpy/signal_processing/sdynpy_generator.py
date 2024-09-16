# -*- coding: utf-8 -*-
"""
Functions for generating various excitation signals used in structural dynamics.

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


def pseudorandom(fft_lines, f_nyq, signal_rms=1, min_freq=None, max_freq=None, filter_oversample=2,
                 integration_oversample=1, averages=1, shape_function=lambda freq: 1):
    '''Creates a pseudorandom excitation given the specified signal parameters

    This function creates a pseudorandom excitation from a sum of sine waves
    with randomized phases.  The function can return shaped input if the
    optional shape_function argument is passed.

    Parameters
    ----------
    fft_lines : int
        The number of fft lines between 0 Hz and f_nyq, inclusive.
    f_nyq : float
        The maximum frequency.  The final sampling frequency will be f_nyq x
        filter_oversample x integration_oversample.
    signal_rms : float
        The RMS amplitude of the signal.  Default is 1.
    min_freq : float
        The minimum frequency of the signal.  If not specified, the minimum
        frequency will be the first frequency line (no 0 Hz component)
    max_freq : float
        The maximum frequency of the signal.  If not specified, the maximum
        frequency will be the nyquist frequency f_nyq.
    filter_oversample : float
        The oversample factor that is used by some data acquisition systems to
        provide bandwidth for antialiasing filters.  For example, in the I-DEAS
        data acquisition software, the true sampling frequency is 2.56x the
        nyquist frequency, rather than the 2x required by the sampling theorem.
        If not specified, the default factor is 2x.
    integration_oversample : int
        An oversampling factor that may be desired for integration of a signal.
        Default is 1
    averages : int
        The number of averages, or frames of data to create.  For a
        pseudorandom input, each frame will be a replica of the first.
    shape_function : function
        A shaping that can be applied to the signal per frequency line.  If
        specified, it should be a function that takes in one argument (the
        frequency in Hz) and returns a single argument (the amplitude of the
        sine wave at that frequency).

    Returns
    -------
    times : np.ndarray
        A 1D array containing the time of each sample
    signal : np.ndarray
        A 1D array containing the specified signal samples.
    '''
    sampling_frequency = f_nyq * filter_oversample
    oversample_frequency = sampling_frequency * integration_oversample
    df = f_nyq / fft_lines
    nbins = sampling_frequency / (2 * df)
    nsamples = 2 * nbins
    total_samples = nsamples * integration_oversample
    oversample_dt = 1 / oversample_frequency
    times = np.arange(total_samples) * oversample_dt
    if min_freq is None:
        min_freq = df
    if max_freq is None:
        max_freq = f_nyq
    frequencies = np.arange(fft_lines) * df
    max_freq_index = np.argmin(np.abs(frequencies - max_freq))
    min_freq_index = np.argmin(np.abs(frequencies - min_freq))
    frequencies = frequencies[min_freq_index:max_freq_index + 1]
    phases = 2 * np.pi * np.random.rand(*frequencies.shape)
    amplitudes = np.array([shape_function(frequency) for frequency in frequencies])
    # Create the sine signal
    signal = np.sum(amplitudes * np.sin(2 * np.pi * frequencies *
                    times[:, np.newaxis] + phases), -1)
    # Create the number of averages
    times = np.arange(total_samples * averages) * oversample_dt
    signal = np.tile(signal, averages)
    signal *= signal_rms / np.sqrt(np.mean(signal**2))
    return times, signal


def random(shape, n_samples, rms=1.0, dt=1.0,
           low_frequency_cutoff=None, high_frequency_cutoff=None):
    signal = np.random.randn(*shape, n_samples)
    if (low_frequency_cutoff is not None) or (high_frequency_cutoff is not None):
        freqs = np.fft.rfftfreq(n_samples, dt)
        signal_fft = np.fft.rfft(signal, axis=-1)
        if low_frequency_cutoff is not None:
            zero_indices = freqs < low_frequency_cutoff
            signal_fft[..., zero_indices] = 0
        if high_frequency_cutoff is not None:
            zero_indices = freqs > high_frequency_cutoff
            signal_fft[..., zero_indices] = 0
        signal = np.fft.irfft(signal_fft, axis=-1)
    # Now set RMSs
    current_rms = np.sqrt(np.mean(signal**2, axis=-1))
    scale = rms / current_rms
    signal *= scale[..., np.newaxis]
    return signal


def sine(frequencies, dt, num_samples, amplitudes=1, phases=0):
    times = np.arange(num_samples) * dt
    output = np.array(amplitudes)[..., np.newaxis] * np.sin(2 * np.pi *
                                                            np.array(frequencies)[..., np.newaxis] * times + np.array(phases)[..., np.newaxis])
    return output


def ramp_envelope(num_samples, ramp_samples, end_ramp_samples=None):
    output = np.ones(num_samples)
    output[:ramp_samples] = np.linspace(0, 1, ramp_samples)
    end_ramp_samples = ramp_samples if end_ramp_samples is None else end_ramp_samples
    output[-end_ramp_samples:] = np.linspace(1, 0, end_ramp_samples)
    return output


def burst_random(shape, n_samples, on_fraction, delay_fraction,
                 rms=1.0, dt=1.0, low_frequency_cutoff=None, high_frequency_cutoff=None,
                 ramp_fraction=0.05):
    random_signal = random(shape, n_samples, rms, dt, low_frequency_cutoff, high_frequency_cutoff)
    burst_window = np.zeros(random_signal.shape)
    delay_samples = int(delay_fraction * n_samples)
    ramp_samples = int(ramp_fraction * n_samples)
    on_samples = int(on_fraction * n_samples)
    burst_window[..., delay_samples:delay_samples +
                 on_samples] = ramp_envelope(on_samples, ramp_samples)
    return random_signal * burst_window


def chirp(frequency_min, frequency_max, signal_length, dt, force_integer_cycles=True):
    n_times = int(signal_length / dt)
    times = np.arange(n_times) * dt
    if force_integer_cycles:
        n_cycles = np.ceil(frequency_max * signal_length)
        frequency_max = n_cycles / signal_length
        # print(frequency_max)
    frequency_slope = (frequency_max - frequency_min) / signal_length
    argument = frequency_slope / 2 * times**2 + frequency_min * times
    signal = np.sin(2 * np.pi * argument)
    return signal


def pulse(signal_length, pulse_time, pulse_width, pulse_peak=1, dt=1, sine_exponent=1):
    signal = np.zeros(signal_length)
    abscissa = np.arange(signal_length) * dt
    pulse_time, pulse_width, pulse_peak = np.broadcast_arrays(pulse_time, pulse_width, pulse_peak)
    for time, width, peak in zip(pulse_time.flatten(), pulse_width.flatten(), pulse_peak.flatten()):
        period = width * 2
        argument = 2 * np.pi / period * (abscissa - time)
        signal += peak * np.cos(argument)**sine_exponent * (np.abs(argument) < np.pi / 2)
    return signal

def sine_sweep(dt, frequencies, sweep_rates, sweep_types, amplitudes = 1, phases = 0):
    """
    Generates a sweeping sine wave with linear or logarithmic sweep rate

    Parameters
    ----------
    dt : float
        The time step of the output signal
    frequencies : iterable
        A list of frequency breakpoints for the sweep.  Can be ascending or
        decending or both.  Frequencies are specified in Hz, not rad/s.
    sweep_rates : iterable
        A list of sweep rates between the breakpoints.  This array should have
        one fewer element than the `frequencies` array.  The ith element of this
        array specifies the sweep rate between `frequencies[i]` and
        `frequencies[i+1]`. For a linear sweep,
        the rate is in Hz/s.  For a logarithmic sweep, the rate is in octave/s.
    sweep_types : iterable or str
        The type of sweep to perform between each frequency breakpoint.  Can be
        'lin' or 'log'.  If a string is specified, it will be used for all
        breakpoints.  Otherwise it should be an array containing strings with
        one fewer element than that of the `frequencies` array.
    amplitudes : iterable or float, optional
        Amplitude of the sine wave at each of the frequency breakpoints.  Can
        be specified as a single floating point value, or as an array with a
        value specified for each breakpoint. The default is 1.
    phases : iterable or float, optional
        Phases of the sine wave at each of the frequency breakpoints.  Can
        be specified as a single floating point value, or as an array with a
        value specified for each breakpoint. Be aware that modifying the phase
        between breakpoints will effectively change the frequency of the signal,
        because the phase will change over time.  The default is 0.

    Raises
    ------
    ValueError
        If the sweep rate and start and end frequency would result in a negative
        sweep time, for example if the start frequency is above the end frequency
        and a positive sweep rate is specified.

    Returns
    -------
    ordinate : np.ndarray
        A numpy array consisting of the generated sine sweep signal.  The length
        of the signal will be determined by the frequency breakpoints and sweep
        rates.

    """
    last_phase = 0
    abscissa = []
    ordinate = []
    # Go through each section
    for i in range(len(frequencies)-1):
        # Extract the terms
        start_frequency = frequencies[i]
        end_frequency = frequencies[i+1]
        omega_start = start_frequency*2*np.pi
        try:
            sweep_rate = sweep_rates[i]
        except TypeError:
            sweep_rate = sweep_rates
        if isinstance(sweep_types,str):
            sweep_type = sweep_types
        else:
            sweep_type = sweep_types[i]
        try:
            start_amplitude = amplitudes[i]
            end_amplitude = amplitudes[i+1]
        except TypeError:
            start_amplitude = amplitudes
            end_amplitude = amplitudes
        try:
            start_phase = phases[i]*np.pi/180
            end_phase = phases[i-1]*np.pi/180
        except TypeError:
            start_phase = phases*np.pi/180
            end_phase = phases*np.pi/180
        # Compute the length of this portion of the signal
        if sweep_type.lower() in ['lin','linear']:
            sweep_time =+ (end_frequency-start_frequency)/sweep_rate
        elif sweep_type.lower() in ['log','logarithmic']:
            sweep_time = np.log(end_frequency/start_frequency)/(sweep_rate*np.log(2))
        if sweep_time < 0:
            raise ValueError('Sweep time for segment index {:} is negative.  Check sweep rate.'.format(i))
        sweep_samples = int(np.floor(sweep_time/dt))
        # Construct the abscissa
        this_abscissa = np.arange(sweep_samples+1)*dt
        # Compute the phase over time
        if sweep_type.lower() in ['lin','linear']:
            this_argument = (1/2)*(sweep_rate*2*np.pi)*this_abscissa**2 + omega_start*this_abscissa
            this_frequency = (sweep_rate)*this_abscissa + omega_start/(2*np.pi)
        elif sweep_type.lower() in ['log','logarithmic']:
            this_argument = 2**(sweep_rate*this_abscissa)*omega_start/(sweep_rate*np.log(2)) - omega_start/(sweep_rate*np.log(2))
            this_frequency = 2**(sweep_rate*this_abscissa)*omega_start/(2*np.pi)
        # Compute the phase at each time step
        if end_frequency > start_frequency:
            freq_interp = [start_frequency,end_frequency]
            phase_interp = [start_phase,end_phase]
            amp_interp = [start_amplitude,end_amplitude]
        else:
            freq_interp = [end_frequency, start_frequency]
            phase_interp = [end_phase,start_phase]
            amp_interp = [end_amplitude,start_amplitude]
        this_phases = np.interp(this_frequency,freq_interp,phase_interp)
        # Compute the amplitude at each time step
        this_amplitudes = np.interp(this_frequency,freq_interp,amp_interp)
        this_ordinate = this_amplitudes*np.sin(this_argument+this_phases+last_phase)
        last_phase += this_argument[-1]
        abscissa.append(this_abscissa[:-1])
        ordinate.append(this_ordinate[:-1])
    ordinate = np.concatenate(ordinate)
    return ordinate