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
    if (not low_frequency_cutoff is None) or (not high_frequency_cutoff is None):
        freqs = np.fft.rfftfreq(n_samples, dt)
        signal_fft = np.fft.rfft(signal, axis=-1)
        if not low_frequency_cutoff is None:
            zero_indices = freqs < low_frequency_cutoff
            signal_fft[..., zero_indices] = 0
        if not high_frequency_cutoff is None:
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
