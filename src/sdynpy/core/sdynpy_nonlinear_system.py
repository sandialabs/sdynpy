# -*- coding: utf-8 -*-
"""
Defines a nonlinear system using a generic equation of motion.

The SystemNL consists of a generic equation of motion that defines its behavior.
The SystemNL also contains a CoordinateArray defining the degrees of freedom of
the System.

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
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime
from scipy.linalg import eig
from scipy.signal import butter, filtfilt
from .sdynpy_coordinate import CoordinateArray,coordinate_array
from .sdynpy_system import System
from .sdynpy_data import TimeHistoryArray,time_history_array
from .sdynpy_shape import shape_array
from ..signal_processing import generator

class SystemNL:
    """Function representing a Structural Dynamics System"""

    def __init__(self, coordinate : CoordinateArray,
                 state_derivative_function,
                 state_size = None,
                 displacement_output_function = None,
                 velocity_output_function = None,
                 acceleration_output_function = None,
                 excitation_output_function = None):
        """Creates a nonlinear system object

        Parameters
        ----------
        coordinate : CoordinateArray
            The physical degrees of freedom associated with the system
        state_derivative_function : function
            A function with calling signature (time, state, input) that returns
            the state derivative.
        state_size : int, optional
            The number of state degrees of freedom in the system.  If not provided,
            it will be assumed to be the same size as coordinate.
        displacement_output_function : _type_, optional
            _description_, by default None
        velocity_output_function : _type_, optional
            _description_, by default None
        acceleration_output_function : _type_, optional
            _description_, by default None
        excitation_output_function : _type_, optional
            _description_, by default None
        """
        self.coordinate = coordinate.flatten()
        if state_size is None:
            self.state_size = self.coordinate.size*2
        else:
            self.state_size = state_size
        self.state_derivative_function = state_derivative_function
        if displacement_output_function is None:
            def default_disp_fn(sys_state, sys_input, response_coordinate = None):
                if response_coordinate is not None:
                    indices,signs = self.coordinate.find_indices(response_coordinate)
                else:
                    indices = slice(None)
                    signs = np.array([1])
                return signs[:,np.newaxis]*sys_state[:self.state_size//2][indices]
            self.displacement_output_function = default_disp_fn
        else:
            self.displacement_output_function = displacement_output_function
        if velocity_output_function is None:
            def default_vel_fn(sys_state, sys_input, response_coordinate = None):
                if response_coordinate is not None:
                    indices,signs = self.coordinate.find_indices(response_coordinate)
                else:
                    indices = slice(None)
                    signs = np.array([1])
                return signs[:,np.newaxis]*sys_state[self.state_size//2:][indices]
            self.velocity_output_function = default_vel_fn
        else:
            self.velocity_output_function = velocity_output_function
        if acceleration_output_function is None:
            def default_accel_fn(time, sys_state, sys_input, response_coordinate = None):
                if response_coordinate is not None:
                    indices,signs = self.coordinate.find_indices(response_coordinate)
                else:
                    indices = slice(None)
                    signs = np.array([1])
                return signs[:,np.newaxis]*self.state_derivative_function(time, sys_state,sys_input)[self.state_size//2:][indices]
            self.acceleration_output_function = default_accel_fn
        else:
            self.acceleration_output_function = acceleration_output_function
        if excitation_output_function is None:
            def default_ex_fn(sys_state, sys_input, excitation_coordinate = None):
                if excitation_coordinate is not None:
                    indices,signs = self.coordinate.find_indices(excitation_coordinate)
                else:
                    indices = slice(None)
                    signs = np.array([1])
                return signs[:,np.newaxis]*sys_input[indices]
            self.excitation_output_function = default_ex_fn
        else:
            self.excitation_output_function = excitation_output_function

    @property
    def num_dof(self):
        return self.coordinate.size

    @classmethod
    def from_linear_system(cls,system : System):
        if system.massless_dofs.sum() > 0:
            raise NotImplementedError('Linear systems with massless degrees of freedom are not yet implemented!')
        A,B,C,D = system.to_state_space()
        coordinates = system.coordinate
        nc = coordinates.shape[0]
        def state_derivative_function(time,sys_state,sys_input):
            return A@sys_state + B@sys_input
        def displacement_output_function(sys_state,sys_input, response_coordinate = None):
            if response_coordinate is not None:
                indices,signs = coordinates.find_indices(response_coordinate)
            else:
                indices = slice(None)
                signs = np.array([1])
            return (signs[:,np.newaxis]*C[0*nc:1*nc,:][indices])@sys_state + (signs[:,np.newaxis]*D[0*nc:1*nc,:][indices])@sys_input
        def velocity_output_function(sys_state,sys_input, response_coordinate = None):
            if response_coordinate is not None:
                indices,signs = coordinates.find_indices(response_coordinate)
            else:
                indices = slice(None)
                signs = np.array([1])
            return (signs[:,np.newaxis]*C[1*nc:2*nc,:][indices])@sys_state + (signs[:,np.newaxis]*D[1*nc:2*nc,:][indices])@sys_input
        def acceleration_output_function(time,sys_state,sys_input, response_coordinate = None):
            if response_coordinate is not None:
                indices,signs = coordinates.find_indices(response_coordinate)
            else:
                indices = slice(None)
                signs = np.array([1])
            return (signs[:,np.newaxis]*C[2*nc:3*nc,:][indices])@sys_state + (signs[:,np.newaxis]*D[2*nc:3*nc,:][indices])@sys_input
        def excitation_output_function(sys_state,sys_input, excitation_coordinate = None):
            if excitation_coordinate is not None:
                indices,signs = coordinates.find_indices(excitation_coordinate)
            else:
                indices = slice(None)
                signs = np.array([1])
            return (signs[:,np.newaxis]*C[3*nc:4*nc,:][indices])@sys_state + (signs[:,np.newaxis]*D[3*nc:4*nc,:][indices])@sys_input
        return cls(coordinates, state_derivative_function, 
                   A.shape[0], displacement_output_function,
                   velocity_output_function, acceleration_output_function,
                   excitation_output_function)
    
    @classmethod
    def duffing_oscillator(cls, delta, alpha, beta, coordinate = None):
        if coordinate is None:
            coordinate = coordinate_array([1],'X+')
        def state_derivative_function(time, sys_state, sys_input):
            return np.array([
                sys_state[1],
                - delta*sys_state[1] - alpha*sys_state[0] - beta*sys_state[0]**3 + sys_input[0]
            ])
        return cls(coordinate, state_derivative_function)

    @classmethod
    def elastic_pendulum(cls, mass, stiffness, unstretched_length, gravity=9.81, node = None):
        if node is None:
            node = 1
        coordinate = coordinate_array(node,['X+','Y+'])
        m = mass
        k = stiffness
        l_0 = unstretched_length
        g = gravity
        def state_derivative_function(time, sys_state, sys_input):
            # State vector x,y,xd,yd
            x = sys_state[0]
            y = sys_state[1]
            xd = sys_state[2]
            yd = sys_state[3]
            xdd = k*l_0*x/(m*np.sqrt((l_0 - y)**2 + x**2)) - k*x/m + sys_input[0]
            ydd = -g - k*l_0**2/(m*np.sqrt((l_0 - y)**2 + x**2)) + k*l_0/m + k*l_0*y/(m*np.sqrt((l_0 - y)**2 + x**2)) - k*y/m + sys_input[1]
            # Derivative vector xd, yd, xdd, ydd
            return np.array([xd,yd,xdd,ydd])
        return cls(coordinate, state_derivative_function)

    @classmethod
    def polynomial_stiffness_damping(cls, coordinate, mass, stiffnesses, dampings=None, transformation=None):
        if dampings is None:
            dampings = ()
        num_states = mass.shape[0]*2
        A_dd = np.concatenate(
            [-np.linalg.solve(mass,stiffness) for exponent,stiffness in stiffnesses.items()]
            + ([-np.linalg.solve(mass,damping) for exponent,damping in dampings.items()] if dampings is not None else []),axis=-1)
        B_dd = np.linalg.solve(mass,transformation.T)
        def state_derivative(t,sys_state,sys_input):
            # Assemble the state derivative from the different pieces
            x = sys_state[:sys_state.shape[0]//2]
            xd = sys_state[sys_state.shape[0]//2:]
            z = np.concatenate(
                [x**exponent for exponent in stiffnesses]
                + ([xd**exponent for exponent in dampings] if dampings is not None else []))
            xdd = A_dd@z+B_dd@sys_input
            return np.concatenate((xd,xdd))
        def displacement_output_function(sys_state,sys_input, response_coordinate = None):
            if response_coordinate is not None:
                indices,signs = coordinate.find_indices(response_coordinate)
            else:
                indices = slice(None)
                signs = np.array([1])
            return (signs[:,np.newaxis]*transformation[indices])@sys_state[:sys_state.shape[0]//2]
        def velocity_output_function(sys_state,sys_input, response_coordinate = None):
            if response_coordinate is not None:
                indices,signs = coordinate.find_indices(response_coordinate)
            else:
                indices = slice(None)
                signs = np.array([1])
            return (signs[:,np.newaxis]*transformation[indices])@sys_state[sys_state.shape[0]//2:]
        def acceleration_output_function(time,sys_state,sys_input, response_coordinate = None):
            if response_coordinate is not None:
                indices,signs = coordinate.find_indices(response_coordinate)
            else:
                indices = slice(None)
                signs = np.array([1])
            dstate = state_derivative(time,sys_state,sys_input)
            return (signs[:,np.newaxis]*transformation[indices])@dstate[dstate.shape[0]//2:]
        def excitation_output_function(sys_state,sys_input, excitation_coordinate = None):
            if excitation_coordinate is not None:
                indices,signs = coordinate.find_indices(excitation_coordinate)
            else:
                indices = slice(None)
                signs = np.array([1])
            return (signs[:,np.newaxis]*np.eye(sys_input.shape[0])[indices])@sys_input
        return cls(coordinate,state_derivative,num_states,displacement_output_function,
                   velocity_output_function,acceleration_output_function,
                   excitation_output_function)
        
        
    def time_integrate(self, forces, dt = None, responses = None, references = None,
                       displacement_derivative = 2, initial_state = None, method = 'RK45',
                       force_interpolation='linear', output_times = None, return_ivp_result = False, **solve_ivp_kwargs):
        # Set up the inputs and the functions
        if isinstance(forces, TimeHistoryArray):
            forces = forces.reshape(-1)
            dt = forces.abscissa_spacing
            references = forces.coordinate[...,0]
            forces = forces.ordinate
        else:
            forces = np.atleast_2d(forces)
            if dt is None:
                raise ValueError('`dt` must be specified if `forces` is not a `TimeHistoryArray`')
        # Construct the entire force vector over time
        if references is None:
            force_array = forces
        else:
            force_array = np.zeros((self.coordinate.shape[0],forces.shape[-1]))
            reference_indices, reference_signs = self.coordinate.find_indices(references)
            force_array[reference_indices] = reference_signs[:,np.newaxis]*forces
        # Create interpolator
        force_abscissa = np.arange(forces.shape[-1])*dt
        if output_times is None:
           output_times = force_abscissa
        force_interpolator = interp1d(force_abscissa,
                                      force_array,kind=force_interpolation,
                                      assume_sorted=True, fill_value=(force_array[:,0],force_array[:,-1]),
                                      bounds_error=False)
        def integration_function(t,y):
            return self.state_derivative_function(t, y, force_interpolator(t))
        if initial_state is None:
            initial_state = np.zeros(self.state_size)
        result = solve_ivp(integration_function, (np.min(output_times),np.max(output_times)),
                           initial_state, method, output_times, **solve_ivp_kwargs)
        # Package the forces into a time history array
        forces_ordinate = force_interpolator(output_times)
        if not result.success:
            print('Integration was not successful: {:}'.format(result.message))
            output_times = output_times[...,:result.y.shape[-1]]
            forces_ordinate = forces_ordinate[...,:result.y.shape[-1]]
        excitation_ordinate = self.excitation_output_function(result.y,forces_ordinate,references)
        excitation_signals = time_history_array(
            output_times,
            excitation_ordinate,
            self.coordinate[:,np.newaxis] if references is None else references[:,np.newaxis])
        if displacement_derivative == 0:
            fn = self.displacement_output_function
        elif displacement_derivative == 1:
            fn = self.velocity_output_function
        elif displacement_derivative == 2:
            fn = lambda state,force,response : self.acceleration_output_function(output_times,state,force,response)
        else:
            raise ValueError('Displacement Derivative must be one of 0, 1, or 2.')
        response_ordinate = fn(result.y,forces_ordinate,responses)
        response_signals = time_history_array(
            output_times,
            response_ordinate,
            self.coordinate[:,np.newaxis] if responses is None else responses[:,np.newaxis])
        return_vals = [response_signals,excitation_signals]
        if return_ivp_result:
            return_vals.append(result)
        return tuple(return_vals)

    def __repr__(self):
        return 'SystemNL with {:} Coordinates ({:} internal states)'.format(self.num_dof, self.state_size)

    def local_stiffness(self, state_point = None, delta_state = np.float64(1.4901161193847656e-08), time=0):
        if state_point is None:
            state_point = np.zeros(self.state_size)
        forces = np.zeros(self.coordinate.size)
        fn = lambda state: self.state_derivative_function(time,state,forces)
        return -approx_fprime(state_point,fn,delta_state)
    
    def eigensolution(self, state_point = None, delta_state = np.float64(1.4901161193847656e-08), time=0, group_conjugate_pairs = True):
        stiffness = self.local_stiffness(state_point, delta_state, time)
        evals, evects = eig(-stiffness)
        try:
            if group_conjugate_pairs:
                # print('Pairing Complex Conjugate Modes...')
                evals_reduced = []
                evects_reduced = []
                evals_handled = []
                for index,(evl,evc) in enumerate(zip(evals,evects.T)):
                    if index in evals_handled:
                        continue
                    matches = np.where(np.isclose(evals,evl.conj()) & (np.all(np.isclose(evects.T,evc.conj()),axis=-1) | np.all(np.isclose(-evects.T,evc.conj()),axis=-1)))[0]
                    matches = [match for match in matches if match!=index]
                    if len(matches) > 1:
                        raise ValueError(f'More than one conjugate eigenvalue was found for index {index}: {matches}')
                    if len(matches) == 0:
                        raise ValueError(f'No conjugate eigenvalue was found for index {index}')
                    # else:
                        # print(f'  Eigenvalue {index} {evl} matches {matches[0]}')
                    # Choose which to keep
                    if evl.imag > evals[matches[0]].imag:
                        evals_reduced.append(evl)
                        evects_reduced.append(evc)
                    else:
                        evals_reduced.append(evals[matches[0]])
                        evects_reduced.append(evects[:,matches[0]])
                    evals_handled.append(index)
                    evals_handled.append(matches[0])
                evals = np.array(evals_reduced)
                evects = np.array(evects_reduced).T
        except ValueError:
            print('Could not group conjugate pairs of modes... continuing.')
        # Compute natural frequency, damping, and mode shape
        angular_frequencies = np.abs(evals)
        frequencies = angular_frequencies/(2*np.pi)
        angular_frequencies[angular_frequencies==0] = 1
        dampings = -evals.real/angular_frequencies
        shape_matrix = self.displacement_output_function(evects,np.zeros((self.coordinate.size,evects.shape[-1]))).T
        shapes = shape_array(self.coordinate,shape_matrix,frequencies,dampings)
        shapes.sort()
        return shapes
    
    def simulate_test(
            self,  # The system itself
            bandwidth,
            frame_length,
            num_averages,
            excitation,
            references,
            responses=None,  # All Responses
            excitation_level=1.0,
            excitation_noise_level=0.0,
            response_noise_level=0.0,
            steady_state_time=0.0,
            excitation_min_frequency=None,
            excitation_max_frequency=None,
            signal_fraction=0.5,
            extra_time_between_frames=0.0,
            integration_oversample=10,
            displacement_derivative=2,
            antialias_filter_cutoff_factor=3,
            antialias_filter_order=4,
            multihammer_impact_spacing_factor = 4,
            time_integrate_kwargs = {},
            generator_kwargs = {}
    ):
        available_excitations = ['pseudorandom', 'random',
                                 'burst random', 'chirp', 'hammer', 'multi-hammer', 'sine']
        if not excitation.lower() in available_excitations:
            raise ValueError('Excitation must be one of {:}'.format(available_excitations))
        # Create the input signal
        num_signals = references.size
        sample_rate = bandwidth * 2 * integration_oversample
        dt = 1 / sample_rate
        frame_time = dt * frame_length * integration_oversample
        df = 1 / frame_time
        # Create the signals
        if excitation.lower() == 'pseudorandom':
            if num_signals > 1:
                print('Warning: Pseudorandom generally not recommended for multi-reference excitation.')
            kwargs = {'fft_lines': frame_length // 2,
                      'f_nyq': bandwidth,
                      'signal_rms': excitation_level,
                      'min_freq': excitation_min_frequency,
                      'max_freq': excitation_max_frequency,
                      'integration_oversample': integration_oversample,
                      'averages': num_averages + int(np.ceil(steady_state_time / frame_time))}
            kwargs.update(generator_kwargs)
            signals = np.array([
                generator.pseudorandom(**kwargs)[1]
                for i in range(num_signals)
            ])
        elif excitation.lower() == 'random':
            kwargs = {'shape': (num_signals,),
                      'n_samples': frame_length * integration_oversample * num_averages + int(steady_state_time * sample_rate),
                      'rms': excitation_level,
                      'dt': dt,
                      'low_frequency_cutoff': excitation_min_frequency,
                      'high_frequency_cutoff': bandwidth if excitation_max_frequency is None else excitation_max_frequency}
            kwargs.update(generator_kwargs)
            signals = generator.random(**kwargs)
        elif excitation.lower() == 'burst random':
            kwargs = {'shape': (num_signals,),
                      'n_samples': frame_length * integration_oversample,
                      'on_fraction': signal_fraction,
                      'delay_fraction': 0,
                      'rms': excitation_level,
                      'dt': dt,
                      'low_frequency_cutoff': excitation_min_frequency,
                      'high_frequency_cutoff': bandwidth if excitation_max_frequency is None else excitation_max_frequency}
            kwargs.update(generator_kwargs)
            signal_list = [generator.burst_random(**kwargs) for i in range(num_averages)]
            full_list = []
            for i, signal in enumerate(signal_list):
                full_list.append(
                    np.zeros((num_signals, int(extra_time_between_frames * sample_rate))))
                full_list.append(signal)
            signals = np.concatenate(full_list, axis=-1)
        elif excitation.lower() == 'chirp':
            if num_signals > 1:
                print('Warning: Chirp generally not recommended for multi-reference excitation.')
            kwargs = {'frequency_min': 0 if excitation_min_frequency is None else excitation_min_frequency,
                      'frequency_max': bandwidth if excitation_max_frequency is None else excitation_max_frequency,
                      'signal_length': frame_time,
                      'dt': dt}
            kwargs.update(generator_kwargs)
            signals = np.array([
                generator.chirp(**kwargs)
                for i in range(num_signals)
            ]) * excitation_level
            signals = np.tile(signals, [1, num_averages +
                              int(np.ceil(steady_state_time / frame_time))])
        elif excitation.lower() == 'hammer':
            if num_signals > 1:
                print(
                    'Warning: Hammer impact generally not recommended for multi-reference excitation, consider multi-hammer instead')
            pulse_width = 2 / (bandwidth if excitation_max_frequency is None else excitation_max_frequency)
            signal_length = int(frame_length * integration_oversample * num_averages + (num_averages + 1)
                                * extra_time_between_frames * sample_rate + 2 * pulse_width * sample_rate)
            pulse_times = np.arange(num_averages)[
                :, np.newaxis] * (frame_time + extra_time_between_frames) + pulse_width + extra_time_between_frames
            kwargs = {'signal_length': signal_length,
                      'pulse_time': pulse_times,
                      'pulse_width': pulse_width,
                      'pulse_peak': excitation_level,
                      'dt': dt,
                      'sine_exponent': 2}
            kwargs.update(generator_kwargs)
            signals = generator.pulse(**kwargs)
            signals = np.tile(signals, [num_signals, 1])
        elif excitation.lower() == 'multi-hammer':
            signal_length = frame_length * integration_oversample
            pulse_width = 2 / (bandwidth if excitation_max_frequency is None else excitation_max_frequency)
            signals = []
            for i in range(num_signals):
                signals.append([])
                for j in range(num_averages):
                    pulse_times = []
                    last_pulse = 0
                    while last_pulse < frame_time * signal_fraction:
                        next_pulse = last_pulse + pulse_width * (np.random.rand() * multihammer_impact_spacing_factor + 1)
                        pulse_times.append(next_pulse)
                        last_pulse = next_pulse
                    pulse_times = np.array(pulse_times)
                    pulse_times = pulse_times[pulse_times <
                                              frame_time * signal_fraction, np.newaxis]
                    kwargs = {'signal_length': signal_length,
                              'pulse_time': pulse_times,
                              'pulse_width': pulse_width,
                              'pulse_peak': excitation_level,
                              'dt': dt,
                              'sine_exponent': 2}
                    kwargs.update(generator_kwargs)
                    signal = generator.pulse(**kwargs)
                    signals[-1].append(np.zeros(int(extra_time_between_frames * sample_rate)))
                    signals[-1].append(signal)
                signals[-1].append(np.zeros(int(extra_time_between_frames * sample_rate)))
                signals[-1] = np.concatenate(signals[-1], axis=-1)
            signals = np.array(signals)
        elif excitation.lower() == 'sine':
            if num_signals > 1:
                print(
                    'Warning: Sine signal generally not recommended for multi-reference excitation')
            frequencies = excitation_max_frequency if excitation_min_frequency is None else excitation_min_frequency
            num_samples = frame_length * integration_oversample * num_averages + int(steady_state_time * sample_rate)
            kwargs = {'frequencies': frequencies,
                      'dt': dt,
                      'num_samples': num_samples,
                      'amplitudes': excitation_level}
            kwargs.update(generator_kwargs)
            signals = np.tile(generator.sine(**kwargs), (num_signals, 1))
        # Set up the integration
        responses, references = self.time_integrate(
            signals, dt, responses, references, displacement_derivative,**time_integrate_kwargs)
        # Now add noise
        responses.ordinate += response_noise_level * np.random.randn(*responses.ordinate.shape)
        references.ordinate += excitation_noise_level * np.random.randn(*references.ordinate.shape)
        # Filter with antialiasing filters, divide filter order by 2 because of filtfilt
        if antialias_filter_order > 0:
            lowpass_b, lowpass_a = butter(antialias_filter_order // 2,
                                          antialias_filter_cutoff_factor * bandwidth, fs=sample_rate)
            responses.ordinate = filtfilt(lowpass_b, lowpass_a, responses.ordinate)
            references.ordinate = filtfilt(lowpass_b, lowpass_a, references.ordinate)
        if integration_oversample > 1:
            responses = responses.downsample(integration_oversample)
            references = references.downsample(integration_oversample)
        responses = responses.extract_elements_by_abscissa(steady_state_time, np.inf)
        references = references.extract_elements_by_abscissa(steady_state_time, np.inf)
        return responses, references
    
    def copy(self):
        return SystemNL(self.coordinate.copy(),self.state_derivative_function,
                        self.state_size,self.displacement_output_function,self.velocity_output_function,
                        self.acceleration_output_function,self.excitation_output_function)