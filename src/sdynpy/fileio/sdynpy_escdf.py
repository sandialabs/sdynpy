# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:47:58 2025

@author: dprohe
"""

from ..core.sdynpy_geometry import (Geometry,NodeArray,CoordinateSystemArray,
                                    coordinate_system_array,node_array,
                                    global_coord,_element_types,_exodus_elem_type_map)
from ..core.sdynpy_shape import ShapeArray,shape_array
from ..core.sdynpy_data import data_array,FunctionTypes,NDDataArray,GUIPlot,join
from ..core.sdynpy_colors import color_list
from ..core.sdynpy_coordinate import coordinate_array
from .sdynpy_rattlesnake import (read_rattlesnake_output,read_system_id_nc4,
                                 read_modal_data,read_random_spectral_data,
                                 read_transient_control_data)

from qtpy.QtWidgets import (QMainWindow)
from qtpy import QtWidgets, uic, QtGui, QtCore
import netCDF4 as nc4
import os
import numpy as np

try:
    import escdf
except ImportError:
    escdf = None

def to_geometry(geometry_dataset):
    if not (geometry_dataset.istype("geometry") or geometry_dataset.istype("point_cloud")):
        raise ValueError(
            "geometry_dataset must have be an ESCDF `geometry` or `point_cloud` dataset."
        )
    if geometry_dataset.istype("geometry"):
        element_type_string = None
        nodes = []
        css = [coordinate_system_array()]
        for id, position, x, y, z in zip(
            geometry_dataset.node_id,
            geometry_dataset.node_position,
            geometry_dataset.node_x_direction,
            geometry_dataset.node_y_direction,
            geometry_dataset.node_z_direction,
        ):
            nodes.append(node_array(id, position, def_cs=1, disp_cs=id + 1, color=0))
            css.append(
                coordinate_system_array(
                    id + 1, "Node {:} Disp CS".format(id), matrix=np.array((x, y, z, np.zeros(3)))
                )
            )
        nodes = np.array(nodes).view(NodeArray)
        css = np.array(css).view(CoordinateSystemArray)

        geometry = Geometry(nodes, css)
        try:
            for i, connection in enumerate(geometry_dataset.line_connection):
                try:
                    color = geometry_dataset.line_color[i, :]
                    color_index = np.argmin(np.linalg.norm(color / 255 - color_list, axis=-1))
                except TypeError:
                    color_index = 0
                geometry.add_traceline(connection, color=color_index)
        except TypeError:
            # No tracelines
            pass

        try:
            for i, connection in enumerate(geometry_dataset.element_connection):
                try:
                    color = geometry_dataset.element_color[i, :]
                    color_index = np.argmin(np.linalg.norm(color / 255 - color_list, axis=-1))
                except TypeError:
                    color_index = 0
                type_name = geometry_dataset.element_type[i][()]
                try:
                    element_type = _exodus_elem_type_map[type_name.lower()]
                except KeyError:
                    if element_type_string is None:
                        element_type_string = {
                            val.lower(): key for key, val in _element_types.items()
                        }
                    try:
                        element_type = element_type_string[type_name.lower()]
                    except KeyError:
                        print(
                            "Unknown Element Type at index {:}, {:}.  Skipping...".format(
                                i, type_name
                            )
                        )
                        continue
                geometry.add_element(element_type, connection, color=color_index)
        except TypeError:
            # No elements
            pass
        return geometry
    elif geometry_dataset.istype("point_cloud"):
        element_type_string = None
        nodes = []
        css = [coordinate_system_array()]
        if geometry_dataset.node_id is None:
            node_id = np.arange(1, geometry_dataset.node_position.shape[0] + 1)
        else:
            node_id = geometry_dataset.node_id[...]
        for id, position in zip(node_id, geometry_dataset.node_position):
            nodes.append(node_array(id, position, def_cs=1, disp_cs=1, color=0))
        nodes = np.array(nodes).view(NodeArray)
        css = np.array(css).view(CoordinateSystemArray)

        geometry = Geometry(nodes, css)
        return geometry
    else:
        raise ValueError(
            "geometry_dataset must have be an ESCDF `geometry` or `point_cloud` dataset."
        )

def from_geometry(dataset_name, geometry : Geometry, descriptive_name = ''):
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    esgeo = escdf.classes.geometry(dataset_name,descriptive_name = descriptive_name)
    esgeo.node_id = geometry.node.id
    esgeo.node_position = geometry.global_node_coordinate(geometry.node.id)
    esgeo.node_x_direction = geometry.global_deflection(
        coordinate_array(geometry.node.id,'X+'))
    esgeo.node_y_direction = geometry.global_deflection(
        coordinate_array(geometry.node.id,'Y+'))
    esgeo.node_z_direction = geometry.global_deflection(
        coordinate_array(geometry.node.id,'Z+'))
    if geometry.traceline.size > 0:
        esgeo.line_color = [np.array(color_list[i])*255 for i in geometry.traceline.color]
        esgeo.line_connection = geometry.traceline.connectivity
    if geometry.element.size > 0:
        esgeo.element_connection = geometry.element.connectivity
        esgeo.element_color = [np.array(color_list[i])*255 for i in geometry.element.color]
        esgeo.element_type = [_element_types[t] for t in geometry.element.type]
    return esgeo

def to_shape(shape_dataset):
    if not shape_dataset.istype('mode'):
        raise ValueError('shape_dataset must be an ESCDF `mode` dataset.')
    coordinate = coordinate_array(string_array=shape_dataset.dof_name[...])
    shape_matrix = shape_dataset.shape[...].T
    frequency = shape_dataset.frequency[...]
    damping = shape_dataset.damping_ratio[...]
    try:
        modal_mass = shape_dataset.modal_mass[...]
    except TypeError:
        modal_mass = 1
    try:
        description = shape_dataset.description[...]
    except TypeError:
        description = ''
    return shape_array(coordinate,shape_matrix,frequency,damping,modal_mass,
                       description)

def from_shape(dataset_name, shape : ShapeArray, descriptive_name = ''):
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    shape = shape.flatten()
    modes = escdf.classes.mode(dataset_name,descriptive_name = descriptive_name)
    modes.damping_ratio = shape.damping
    modes.frequency = shape.frequency
    dofs = np.unique(shape.coordinate)
    modes.shape = shape[dofs].T
    modes.dof_name = dofs.string_array()
    modes.modal_mass = shape.modal_mass
    if not np.all(shape.comment1 == ''):
        modes.description = shape.comment1.astype(object)
    return modes

datatype_enum_names = {
    FunctionTypes.TIME_RESPONSE:'time response',
    FunctionTypes.CROSSSPECTRUM:'power spectrum',
    FunctionTypes.FREQUENCY_RESPONSE_FUNCTION:'frequency response function',
    FunctionTypes.TRANSMISIBILITY:'transmissibility',
    FunctionTypes.COHERENCE:'coherence',
    FunctionTypes.CROSSCORRELATION:'correlation',
    FunctionTypes.POWER_SPECTRAL_DENSITY:'power spectral density',
    FunctionTypes.SPECTRUM:'spectrum',
    FunctionTypes.MODE_INDICATOR_FUNCTION:'mode indicator function',
    FunctionTypes.PARTIAL_COHERENCE:'partial coherence',
    FunctionTypes.SHOCK_RESPONSE_SPECTRUM:'shock response spectrum',
    FunctionTypes.IMPULSE_RESPONSE_FUNCTION:'impulse response function',
    FunctionTypes.MULTIPLE_COHERENCE:'multiple coherence'}

datatype_names = {
    FunctionTypes.GENERAL:
        ['general','none'],
    FunctionTypes.TIME_RESPONSE:
        ['time history','time response', 'th', 'time data', 'time',
         'time histories','time responses', 'ths', 'times',],
    FunctionTypes.AUTOSPECTRUM:
        ['autospectrum','as','autospectra'],
    FunctionTypes.CROSSSPECTRUM:
        ['crossspectrum','cs','crossspectra'],
    FunctionTypes.FREQUENCY_RESPONSE_FUNCTION:
        ['frf','frequency response','frequency response function',
         'frfs','frequency responses', 'frequency response functions',
         'transfer function','transfer functions', 'tf'],
    FunctionTypes.TRANSMISIBILITY:
        ['transmisibility','transmisibility function',
         'transmisibilities','transmisibility functions'],
    FunctionTypes.COHERENCE:['coherence','coh'],
    FunctionTypes.AUTOCORRELATION:
        ['corr','correlation','autocorrelation','corrs','correlations',
         'autocorrelations'],
    FunctionTypes.CROSSCORRELATION:
        ['ccorr','crosscorrelation''ccorrs',
         'crosscorrelations'],
    FunctionTypes.POWER_SPECTRAL_DENSITY:
        ['power spectral density','power spectral densities','psd','psds',
         'apsd','apsds','cpsd','cpsds','autopower spectral density',
         'autopower spectral densities','crosspower spectral density',
         'crosspower spectral densities'],
    FunctionTypes.ENERGY_SPECTRAL_DENSITY:
        ['esd','energy spectral density','energy spectral densities','esds'],
    FunctionTypes.PROBABILITY_DENSITY_FUNCTION:
        ['pdf','pdfs','probability density','probability densities',
         'probability density function','probability density functions'],
    FunctionTypes.SPECTRUM:
        ['fft','ffts','spectrum','spectra'],
    FunctionTypes.CUMULATIVE_FREQUENCY_DISTRIBUTION:
        ['cfd','cfds','cumulative frequency distribution',
         'cumulative frequency distributions'],
    FunctionTypes.PEAKS_VALLEY:
        ['peaks valley'],
    FunctionTypes.STRESS_PER_CYCLE:
        ['stress per cycle','stress/cycle'],
    FunctionTypes.STRAIN_PER_CYCLE:
        ['strain per cycle','strain/cycle'],
    FunctionTypes.ORBIT:['orbit','orbits'],
    FunctionTypes.MODE_INDICATOR_FUNCTION:
        ['mif','cmif','mmif','nmif','mode indicator function','mode indicator functions',
         'complex mode indicator function','complex mode indicator functions',
         'multimode indicator function','multimode indicator functions',
         'normal mode indicator function normal mode indicator functions'],
    FunctionTypes.FORCE_PATTERN:['force_pattern'],
    FunctionTypes.PARTIAL_POWER:['partial power'],
    FunctionTypes.PARTIAL_COHERENCE:['partial coherence','partial coh','pcoh'],
    FunctionTypes.EIGENVALUE:['eigenvalue','eigenvalues'],
    FunctionTypes.EIGENVECTOR:['eigenvector','eigenvectors'],
    FunctionTypes.SHOCK_RESPONSE_SPECTRUM:
        ['shock response spectrum','shock response spectra','srs','srss'],
    FunctionTypes.FINITE_IMPULSE_RESPONSE_FILTER:
        ['finite impulse response function','firf'],
    FunctionTypes.MULTIPLE_COHERENCE:
        ['mcoh','multiple coherence'],
    FunctionTypes.ORDER_FUNCTION:['order','order function','order functions'],
    FunctionTypes.PHASE_COMPENSATION:['phase comp','phase compensation'],
    FunctionTypes.IMPULSE_RESPONSE_FUNCTION:
        ['irf','irfs','impulse response function','impulse response functions',
         'impuse response','impulse responses']
    }

datatype_map = {}
for key,name_list in datatype_names.items():
    for name in name_list:
        datatype_map[name.replace(' ','').replace('_','').replace('-','').lower()] = key

def to_data(data_dataset):
    if not data_dataset.istype('data'):
        raise ValueError('data_dataset must be an ESCDF `data` dataset.')
    try:
        function_type = datatype_map[
            data_dataset.data_type[...][()].replace(' ','').replace('_','').replace('-','').lower()]
    except KeyError:
        raise ValueError('Unknown data type {:}'.format(data_dataset.data_type[...][()]))
    coordinate = coordinate_array(string_array=data_dataset.channel[...])
    ordinate_unit = np.char.add('Ordinate Unit: ', data_dataset.ordinate_unit[...])
    abscissa_unit = np.char.add(' -- Abscissa Unit: ', data_dataset.abscissa_unit[...])
    comment1 = np.char.add(ordinate_unit,abscissa_unit)
    try:
        abscissa = np.arange(data_dataset.ordinate.shape[-1])*data_dataset.abscissa_step[...] + data_dataset.abscissa_start[...]
    except TypeError:
        abscissa = data_dataset.abscissa[...]
    data = data_array(function_type, abscissa, data_dataset.ordinate[...],
                      coordinate, comment1)
    return data

def from_data(dataset_name, data : NDDataArray,descriptive_name=''):
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    data = data.flatten()
    try:
        abscissa_spacing = data.abscissa_spacing
    except ValueError:
        abscissa_spacing = None
    starting_abscissa = data.abscissa[...,0]
    if not np.all(starting_abscissa == starting_abscissa[0]):
        starting_abscissa = None
    else:
        starting_abscissa = starting_abscissa[0]
    dataset = escdf.classes.data(dataset_name,descriptive_name = descriptive_name)
    if starting_abscissa is None or abscissa_spacing is None:
        if np.all(data.abscissa == data.abscissa[0]):
            abscissa = data.abscissa[0]
        else:
            abscissa = data.abscissa
        dataset.abscissa = abscissa
    else:
        dataset.abscissa_step = abscissa_spacing
        dataset.abscissa_start = starting_abscissa
    dataset.ordinate = data.ordinate
    dataset.data_type = datatype_enum_names[data.function_type]
    dataset.channel = data.coordinate.string_array().astype(object)
    return dataset

def from_rattlesnake_channel_info(dataset_name, channel_info, descriptive_name=''):
    """Generates a "vibration_channel_table" dataset from the channel information
    in a Rattlesnake netcdf4 file.

    Operates on channel tables like the one from 

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to be generated
    channel_info : pandas.DataFrame
        A dataframe returned from reading Rattlesnake netcdf4 output into SDynPy
    descriptive_name : str
        A description of the dataset to be generated

    Returns
    -------
    escdf.Dataset
        A dataset with the type 'vibration_channel_table' containing channel
        table information
    """
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')

    escdf_channels = escdf.Dataset(dataset_name,'vibration_channel_table',
                         descriptive_name)

    escdf_channels.coupling = [v.lower() for v in channel_info['coupling']]
    escdf_channels.daq = [daq+'/'+channel for daq,channel in zip(channel_info['physical_device'],channel_info['physical_channel'])]
    escdf_channels.data_type = [v.lower() for v in channel_info['channel_type']]
    escdf_channels.node_direction = channel_info['node_direction']
    escdf_channels.node_id = channel_info['node_number']
    escdf_channels.sensitivity = channel_info['sensitivity']
    escdf_channels.sensitivity_unit = channel_info['unit']
    escdf_channels.make = channel_info['make']
    escdf_channels.model = channel_info['model']
    escdf_channels.serial_number = [serial_number+triax_dof for serial_number,triax_dof in zip(channel_info['serial_number'],channel_info['triax_dof'])]

    return escdf_channels

def from_rattlesnake_modal_parameters(dataset_name, modal_dataset, environment_name = None, descriptive_name=''):
    """Generates a "modal_test_parameters" object from modal output

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to be generated
    modal_dataset : netCDF4.Dataset
        A dataset produced by loading the netcdf4 output file.
    environment_name : str, optional
        Name of the environment.  Must be specified if more than one environment
        exists in the file
    descriptive_name : str, optional
        A description of the generated dataset, by default ''.

    Returns
    -------
    escdf.Dataset
        A dataset with the type "modal_test_parameters" containing
        modal parameters
    """
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    
    environment_names = [name for name in modal_dataset.groups if name != 'channels']
    if environment_name is None:
        environment_names = [name for name in modal_dataset.groups if name != 'channels']
        if len(environment_names) != 1:
            raise ValueError('Found {:} environments in the file.  Could not select correctly.'.format(len(environment_names)))
        environment_name = environment_names[0]
    modal_parameters = escdf.Dataset(dataset_name, 'modal_test_parameters', descriptive_name)
    modal_parameters.sample_rate = modal_dataset.sample_rate
    modal_parameters.samples_per_frame = modal_dataset[environment_name].samples_per_frame
    modal_parameters.averaging_type = modal_dataset[environment_name].averaging_type.lower()
    modal_parameters.averages = modal_dataset[environment_name].num_averages
    if modal_dataset[environment_name].averaging_type.lower() == 'exponential':
        modal_parameters.averaging_coefficient = modal_dataset[environment_name].averaging_coefficient
    modal_parameters.overlap = modal_dataset[environment_name].overlap
    modal_parameters.window = modal_dataset[environment_name].frf_window.lower()
    if modal_dataset[environment_name].frf_window.lower() == 'exponential':
        modal_parameters.window_parameters = [modal_dataset[environment_name].exponential_window_value_at_frame_end]
        modal_parameters.window_parameter_names = ['Window Value at Frame End']
    modal_parameters.frf_technique = modal_dataset[environment_name].frf_technique.lower()
    trigger = modal_dataset[environment_name].trigger_type.lower()
    modal_parameters.trigger_type = trigger
    if trigger != 'free run':
        modal_parameters.trigger_slope = 'positive' if modal_dataset[environment_name].trigger_slope_positive else 'negative'
        modal_parameters.trigger_level = modal_dataset[environment_name].trigger_level
        modal_parameters.pretrigger = modal_dataset[environment_name].pretrigger
    signal_generator_type = modal_dataset[environment_name].signal_generator_type
    modal_parameters.signal_generator_type = signal_generator_type
    if signal_generator_type != 'none':
        modal_parameters.signal_generator_level = modal_dataset[environment_name].signal_generator_level
        modal_parameters.signal_generator_min_freq = modal_dataset[environment_name].signal_generator_min_frequency
        modal_parameters.signal_generator_max_freq = modal_dataset[environment_name].signal_generator_max_frequency
    if signal_generator_type == 'burst':
        modal_parameters.signal_generator_on_fraction = modal_dataset[environment_name].signal_generator_on_fraction
    environment_index = modal_dataset['environment_names'][...]==environment_name
    active_channel_indices = modal_dataset['environment_active_channels'][...,environment_index][:,0].astype(bool)
    channel_names = np.array([str(modal_dataset['channels']['node_number'][i])+str(modal_dataset['channels']['node_direction'][i]) for i in range(modal_dataset.dimensions['response_channels'].size)])[active_channel_indices]
    reference_channels = channel_names[modal_dataset[environment_name]['reference_channel_indices'][:]]
    modal_parameters.reference_channel = reference_channels
    excitation_indices = np.array([modal_dataset['channels']['feedback_device'][i] != '' for i in range(modal_dataset.dimensions['response_channels'].size)])
    excitation_channels = channel_names[excitation_indices]
    modal_parameters.excitation_channel = excitation_channels
    return modal_parameters

def from_rattlesnake_system_id_parameters(dataset_name, system_id_dataset, environment_name=None, descriptive_name=''):
    """Gets system identification parameters from a NetCDF4 streaming file

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to be generated
    system_id_dataset : netCDF4.Dataset
        A dataset created by loading a system identification streaming file
    environment_name : str, optional
        Name of the environment.  Must be specified if more than one environment
        exists in the file
    descriptive_name : str, optional
        A description for the dataset that will be generated, by default ''.

    Returns
    -------
    escdf.Dataset
        A dataset with the type "rattlesnake_sysid_parameters" containing
        system identification parameters.
    """
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    if environment_name is None:
        environment_names = [name for name in system_id_dataset.groups if name != 'channels']
        if len(environment_names) != 1:
            raise ValueError('Found {:} environments in the file.  Could not select correctly.'.format(len(environment_names)))
        environment_name = environment_names[0]
    sysid_params = escdf.classes.rattlesnake_sysid_parameters(dataset_name,
                                                              descriptive_name)
    sysid_params.sample_rate = system_id_dataset.sample_rate
    sysid_params.frame_size = system_id_dataset[environment_name].sysid_frame_size
    sysid_params.averaging_type = system_id_dataset[environment_name].sysid_averaging_type
    sysid_params.noise_averages = system_id_dataset[environment_name].sysid_noise_averages
    sysid_params.averages = system_id_dataset[environment_name].sysid_averages
    if system_id_dataset[environment_name].sysid_averaging_type != 'Linear':
        sysid_params.exponential_averaging_coefficient = system_id_dataset[environment_name].sysid_exponential_averaging_coefficient
    sysid_params.estimator = system_id_dataset[environment_name].sysid_estimator
    sysid_params.level = system_id_dataset[environment_name].sysid_level
    sysid_params.level_ramp_time = system_id_dataset[environment_name].sysid_level_ramp_time
    sysid_params.signal_type = system_id_dataset[environment_name].sysid_signal_type
    if system_id_dataset[environment_name].sysid_signal_type == 'Burst Random':
        sysid_params.burst_on = system_id_dataset[environment_name].sysid_burst_on
        sysid_params.pretrigger = system_id_dataset[environment_name].sysid_pretrigger
        sysid_params.burst_ramp_fraction = system_id_dataset[environment_name].burst_ramp_fraction
    sysid_params.window = system_id_dataset[environment_name].sysid_window
    sysid_params.overlap = system_id_dataset[environment_name].sysid_overlap
    environment_index = system_id_dataset['environment_names'][...]==environment_name
    active_channel_indices = system_id_dataset['environment_active_channels'][...,environment_index][:,0].astype(bool)
    channel_names = np.array([str(system_id_dataset['channels']['node_number'][i])+str(system_id_dataset['channels']['node_direction'][i]) for i in range(system_id_dataset.dimensions['response_channels'].size)])[active_channel_indices]
    control_channels = channel_names[system_id_dataset[environment_name]['control_channel_indices'][:]]
    sysid_params.control_channel = control_channels
    excitation_indices = np.array([system_id_dataset['channels']['feedback_device'][i] != '' for i in range(system_id_dataset.dimensions['response_channels'].size)])
    excitation_channels = channel_names[excitation_indices]
    sysid_params.excitation_channel = excitation_channels
    if 'response_transformation_matrix' in system_id_dataset[environment_name].variables:
        sysid_params.control_transformation_matrix = system_id_dataset[environment_name]['response_transformation_matrix'][...]
    if 'reference_transformation_matrix' in system_id_dataset[environment_name].variables:
        sysid_params.excitation_transformation_matrix = system_id_dataset[environment_name]['reference_transformation_matrix'][...]
    return sysid_params

def from_rattlesnake_random_parameters(dataset_name, random_dataset, environment_name=None, descriptive_name=''):
    """Gets random vibration parameters from a NetCDF4 streaming file

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to be generated
    random_dataset : netCDF4.Dataset
        A dataset created by loading the streaming file
    environment_name : str, optional
        Name of the environment.  Must be specified if more than one environment
        exists in the file
    descriptive_name : str, optional
        A description for the dataset that will be generated, by default ''.

    Returns
    -------
    escdf.Dataset
        A dataset with the type "rattlesnake_random_control_parameters" containing
        random vibration control parameters.
    """
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    if environment_name is None:
        environment_names = [name for name in random_dataset.groups if name != 'channels']
        if len(environment_names) != 1:
            raise ValueError('Found {:} environments in the file.  Could not select correctly.'.format(len(environment_names)))
        environment_name = environment_names[0]
    params = escdf.classes.rattlesnake_random_control_parameters(dataset_name,
                                                                 descriptive_name)
    params.sample_rate = random_dataset.sample_rate
    environment_index = random_dataset['environment_names'][...]==environment_name
    active_channel_indices = random_dataset['environment_active_channels'][...,environment_index][:,0].astype(bool)
    channel_names = np.array([str(random_dataset['channels']['node_number'][i])+str(random_dataset['channels']['node_direction'][i]) for i in range(random_dataset.dimensions['response_channels'].size)])[active_channel_indices]
    control_channels = channel_names[random_dataset[environment_name]['control_channel_indices'][:]]
    params.control_channel = control_channels
    excitation_indices = np.array([random_dataset['channels']['feedback_device'][i] != '' for i in range(random_dataset.dimensions['response_channels'].size)])
    excitation_channels = channel_names[excitation_indices]
    params.excitation_channel = excitation_channels
    params.samples_per_frame = random_dataset[environment_name].samples_per_frame
    params.control_frequency_lines = random_dataset[environment_name]['specification_frequency_lines'][...]
    params.test_level_ramp_time = random_dataset[environment_name].test_level_ramp_time
    params.cpsd_overlap = random_dataset[environment_name].cpsd_overlap
    params.cola_window = random_dataset[environment_name].cola_window
    params.update_tf_during_control = random_dataset[environment_name].update_tf_during_control
    params.cola_overlap = random_dataset[environment_name].cola_overlap
    params.cola_window_exponent = random_dataset[environment_name].cola_window_exponent
    params.frames_in_cpsd = random_dataset[environment_name].frames_in_cpsd
    params.cpsd_window = random_dataset[environment_name].cpsd_window
    params.control_python_script = random_dataset[environment_name].control_python_script
    params.control_python_function = random_dataset[environment_name].control_python_function
    params.control_python_function_type = random_dataset[environment_name].control_python_function_type
    params.control_python_function_parameters = random_dataset[environment_name].control_python_function_parameters
    params.allow_automatic_aborts = random_dataset[environment_name].allow_automatic_aborts
    units = [f'({unit})^2/Hz' for unit in np.array(
        [random_dataset['channels']['unit'][i] != '' for i in range(random_dataset.dimensions['response_channels'].size)]
        )[active_channel_indices][random_dataset[environment_name]['control_channel_indices'][:]]]
    params.specification_unit = units
    if not np.all(np.isnan(random_dataset[environment_name]['specification_warning_matrix'][...])):
        params.specification_warning_matrix = random_dataset[environment_name]['specification_warning_matrix'][...]
    if not np.all(np.isnan(random_dataset[environment_name]['specification_abort_matrix'][...])):
        params.specification_abort_matrix = random_dataset[environment_name]['specification_abort_matrix'][...]
    if 'response_transformation_matrix' in random_dataset[environment_name].variables:
        params.control_transformation_matrix = random_dataset[environment_name]['response_transformation_matrix'][...]
    if 'reference_transformation_matrix' in random_dataset[environment_name].variables:
        params.excitation_transformation_matrix = random_dataset[environment_name]['reference_transformation_matrix'][...]
    return params

def from_rattlesnake_transient_parameters(dataset_name, transient_dataset, environment_name=None, descriptive_name=''):
    """Gets transient vibration parameters from a NetCDF4 streaming file

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to be generated
    transient_dataset : netCDF4.Dataset
        A dataset created by loading the streaming file
    environment_name : str, optional
        Name of the environment.  Must be specified if more than one environment
        exists in the file
    descriptive_name : str, optional
        A description for the dataset that will be generated, by default ''.

    Returns
    -------
    escdf.Dataset
        A dataset with the type "rattlesnake_transient_control_parameters" containing
        random vibration control parameters.
    """
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    if environment_name is None:
        environment_names = [name for name in transient_dataset.groups if name != 'channels']
        if len(environment_names) != 1:
            raise ValueError('Found {:} environments in the file.  Could not select correctly.'.format(len(environment_names)))
        environment_name = environment_names[0]
    params = escdf.classes.rattlesnake_transient_control_parameters(dataset_name,
                                                                    descriptive_name)
    params.sample_rate = transient_dataset.sample_rate
    environment_index = transient_dataset['environment_names'][...]==environment_name
    active_channel_indices = transient_dataset['environment_active_channels'][...,environment_index][:,0].astype(bool)
    channel_names = np.array([str(transient_dataset['channels']['node_number'][i])+str(transient_dataset['channels']['node_direction'][i]) for i in range(transient_dataset.dimensions['response_channels'].size)])[active_channel_indices]
    control_channels = channel_names[transient_dataset[environment_name]['control_channel_indices'][:]]
    params.control_channel = control_channels
    excitation_indices = np.array([transient_dataset['channels']['feedback_device'][i] != '' for i in range(transient_dataset.dimensions['response_channels'].size)])
    excitation_channels = channel_names[excitation_indices]
    params.excitation_channel = excitation_channels
    params.test_level_ramp_time = transient_dataset[environment_name].test_level_ramp_time
    params.control_python_script = transient_dataset[environment_name].control_python_script
    params.control_python_function = transient_dataset[environment_name].control_python_function
    params.control_python_function_type = transient_dataset[environment_name].control_python_function_type
    params.control_python_function_parameters = transient_dataset[environment_name].control_python_function_parameters
    if 'response_transformation_matrix' in transient_dataset[environment_name].variables:
        params.control_transformation_matrix = transient_dataset[environment_name]['response_transformation_matrix'][...]
    if 'reference_transformation_matrix' in transient_dataset[environment_name].variables:
        params.excitation_transformation_matrix = transient_dataset[environment_name]['reference_transformation_matrix'][...]
    return params

def datasets_from_rattlesnake_system_identification(
        test_name_slug,
        sysid_streaming_data,
        sysid_results_data,
        verbose=False):
    """
    Converts system identification data from Rattlesnake into ESCDF Datasets

    Parameters
    ----------
    test_name_slug : str
        A string identifier that will be prepended to the dataset names
    sysid_streaming_data : str or netCDF4.Dataset
        A string pointing to a netCDF4 file or the netCDF4 file loaded as a
        netCDF4.Dataset object.  The file should contain streaming data from
        a System Identification phase of the Rattlesnake controller.
    sysid_results_data : str or netCDF4.Dataset
        A string pointing to a netCDF4 file or the netCDF4 file loaded as a
        netCDF4.Dataset object.  The file should contain spectral data from
        a System Identification phase of the Rattlesnake controller.
    verbose : bool, optional
        If True, progress will be printed. The default is False.

    Raises
    ------
    ImportError
        If the escdf package is not available.

    Returns
    -------
    ESCDF Datasets
        A nested Tuple of ESCDF datasets.  The first index is the metadata and
        the second index is the data.  Metadata includes system identification
        parameters and channel table.  Data includes FRFs, Noise and System ID
        Response CPSDs, Noise and System ID Drive CPSDs, and Multiple Coherence.

    """
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    if isinstance(sysid_streaming_data,str):
        sysid_streaming_data = nc4.Dataset(sysid_streaming_data)
    if isinstance(sysid_results_data,str):
        sysid_results_data = nc4.dataset(sysid_results_data)
    
    # Extract the data
    if verbose:
        print('Reading Time Data')
    noise_time_history, channel_table = read_rattlesnake_output(
        sysid_streaming_data,read_variable = 'time_data')
    sysid_time_history, _ = read_rattlesnake_output(
        sysid_streaming_data,read_variable = 'time_data_1')
    if verbose:
        print('Reading Spectral Data')
    (sysid_frfs, sysid_response_cpsd, sysid_drive_cpsd,
     sysid_response_noise_cpsd, sysid_drive_noise_cpsd, sysid_coherence) = (
         read_system_id_nc4(sysid_results_data))
    
    if verbose:
         print('Creating ESCDF Data Objects')
    # Create escdf data objects
    esnoise_time_history = from_data(
        f'{test_name_slug}_Noise_Time_History',noise_time_history)
    esnoise_time_history.ordinate_unit = [comment.split('::')[1].strip() for comment in noise_time_history.flatten().comment1]
    esnoise_time_history.abscissa_unit = 's'
    
    essysid_time_history = from_data(
        f'{test_name_slug}_SysID_Time_History',sysid_time_history)
    essysid_time_history.ordinate_unit = [comment.split('::')[1].strip() for comment in noise_time_history.flatten().comment1]
    essysid_time_history.abscissa_unit = 's'
    
    esspectral_data = []
    for data,joiner,appender,label in zip([sysid_frfs,sysid_response_cpsd,sysid_response_noise_cpsd,
                                  sysid_drive_cpsd,sysid_drive_noise_cpsd],
                                 ['/','*','*','*','*'],
                                 ['','/Hz','/Hz','/Hz','/Hz'],
                                 ['SysID_FRFs','SysID_Response_CPSD',
                                  'SysID_Response_Noise_CPSD',
                                  'SysID_Drive_CPSD',
                                  'SysID_Drive_Noise_CPSD']):
        data = data.flatten()
        esobj = from_data(f'{test_name_slug}_{label}',data)
        try:
            if np.all(data.comment1[0] == data.comment1): # All units are the same
                comment = data.comment1[0]
                esobj.ordinate_unit = joiner.join(['('+value.split('::')[1].strip()+')' for value in comment.split('//')])+appender
            else:
                esobj.ordinate_unit = [joiner.join(['('+value.split('::')[1].strip()+')' for value in comment.split('//')])+appender for comment in data.comment1]
        except IndexError:
            print(f'Could not automatically assign units to {esobj.name}')
        esobj.abscissa_unit = 'Hz'
        esspectral_data.append(esobj)
        
    essysid_coherence = from_data(
        f'{test_name_slug}_SysID_Coherence',sysid_coherence)
    essysid_coherence.ordinate_unit = '-'
    essysid_coherence.abscissa_unit = 'Hz'
        
    if verbose:
        print('Creating ESCDF Metadata Objects')
    # Channel table
    eschan = from_rattlesnake_channel_info(
        f'{test_name_slug}_Channel_Table', channel_table)
    
    # System Identification Parameters
    essysid = from_rattlesnake_system_id_parameters(
        f'{test_name_slug}_Params', sysid_streaming_data)
    
    return ((essysid, eschan),
            tuple([esnoise_time_history, essysid_time_history] + esspectral_data
                  + [essysid_coherence]))

def datasets_from_rattlesnake_random_vibration(test_name_slug,
                                               streaming_data,
                                               spectral_data,
                                               verbose=False):
    """
    Converts random vibration data from Rattlesnake into ESCDF Datasets

    Parameters
    ----------
    test_name_slug : str
        A string identifier that will be prepended to the dataset names
    streaming_data : str or netCDF4.Dataset
        A string pointing to a netCDF4 file or the netCDF4 file loaded as a
        netCDF4.Dataset object.  The file should contain streaming data from
        a random vibration phase of the Rattlesnake controller.
    spectral_data : str or netCDF4.Dataset
        A string pointing to a netCDF4 file or the netCDF4 file loaded as a
        netCDF4.Dataset object.  The file should contain spectral data from
        a random vibration phase of the Rattlesnake controller.
    verbose : bool, optional
        If True, progress will be printed. The default is False.

    Raises
    ------
    ImportError
        If the escdf package is not available.

    Returns
    -------
    ESCDF Datasets
        A nested tuple of ESCDF datasets.  The first index is the metadata and
        the second index is the data.  Metadata includes specification, random
        vibration parameters, and channel table.  Data includes Response CPSD
        and Drive CPSD.
    """
    
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    if isinstance(streaming_data,str):
        streaming_data = nc4.Dataset(streaming_data)
    if isinstance(spectral_data,str):
        spectral_data = nc4.dataset(spectral_data)
    
    # Extract the data
    time_dataset_names = [variable for variable in streaming_data.variables if 'time_data' in variable]
    if verbose:
        print('Reading Time Data')
    time_datasets = []
    for name in time_dataset_names:
        time_history, channel_table = read_rattlesnake_output(
            streaming_data,read_variable = name)
        time_datasets.append(time_history)
    
    if verbose:
        print('Reading Spectral Data')
    response_cpsd,spec_cpsd,drive_cpsd = read_random_spectral_data(spectral_data)
    
    if verbose:
         print('Creating ESCDF Data Objects')
    esdata = []
    for i,data in enumerate(time_datasets):
        this_esdata = from_data(
            f'{test_name_slug}_Time_History{"_"+str(i) if len(time_datasets)>1 else ""}',
            data)
        this_esdata.ordinate_unit = [comment.split('::')[1].strip() for comment in data.flatten().comment1]
        this_esdata.abscissa_unit = 's'
        esdata.append(this_esdata)
    
    esspec = []
    for i,(data, label) in enumerate(zip([spec_cpsd, response_cpsd, drive_cpsd],
                           ['Specification_CPSD','Response_CPSD','Drive_CPSD'])):
        data = data.flatten()
        this_esdata = from_data(f'{test_name_slug}_{label}',data)
        try:
            if np.all(data.comment1[0] == data.comment1): # All units are the same
                comment = data.comment1[0]
                this_esdata.ordinate_unit = '*'.join(['('+value.split('::')[1].strip()+')' for value in comment.split('//')])+'/Hz'
            else:
                this_esdata.ordinate_unit = ['*'.join(['('+value.split('::')[1].strip()+')' for value in comment.split('//')])+'/Hz' for comment in data.comment1]
        except IndexError:
            print(f'Could not automatically assign units to {this_esdata.name}')
        this_esdata.abscissa_unit = 'Hz'
        if i == 0:
            esspec.append(this_esdata)
        else:
            esdata.append(this_esdata)
    
    if verbose:
        print('Creating ESCDF Metadata Objects')
    
    # Channel table
    eschan = from_rattlesnake_channel_info(
        f'{test_name_slug}_Channel_Table', channel_table)
    # Random Vibration Parameters
    esrand = from_rattlesnake_random_parameters(
        f'{test_name_slug}_Params', streaming_data)
    
    # Pull the spec into metadata
    return ((esrand, eschan, esspec[0]), tuple(esdata))

def datasets_from_rattlesnake_transient_vibration(test_name_slug,
                                                  streaming_data,
                                                  control_data,
                                                  verbose=False):
    """
    Converts transient vibration data from Rattlesnake into ESCDF Datasets

    Parameters
    ----------
    test_name_slug : str
        A string identifier that will be prepended to the dataset names
    streaming_data : str or netCDF4.Dataset
        A string pointing to a netCDF4 file or the netCDF4 file loaded as a
        netCDF4.Dataset object.  The file should contain streaming data from
        a random vibration phase of the Rattlesnake controller.
    control_data : str or netCDF4.Dataset
        A string pointing to a netCDF4 file or the netCDF4 file loaded as a
        netCDF4.Dataset object.  The file should contain control data from
        a transient vibration phase of the Rattlesnake controller.
    verbose : bool, optional
        If True, progress will be printed. The default is False.

    Raises
    ------
    ImportError
        If the escdf package is not available.

    Returns
    -------
    ESCDF Datasets
        A nested tuple of ESCDF datasets.  The first index is the metadata and
        the second index is the data.  Metadata includes specification, transient
        vibration parameters, and channel table.  Data includes Response
        and Drive time histories.
    """

    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    if isinstance(streaming_data,str):
        streaming_data = nc4.Dataset(streaming_data)
    if isinstance(control_data,str):
        control_data = nc4.dataset(control_data)

    # Extract the data
    time_dataset_names = [variable for variable in streaming_data.variables if 'time_data' in variable]
    if verbose:
        print('Reading Time Data')
    time_datasets = []
    for name in time_dataset_names:
        time_history, channel_table = read_rattlesnake_output(
            streaming_data,read_variable = name)
        time_datasets.append(time_history)

    if verbose:
        print('Reading Control Data')
    response_th,spec_th,drive_th = read_transient_control_data(control_data)

    if verbose:
         print('Creating ESCDF Data Objects')
    esdata = []
    for i,data in enumerate(time_datasets):
        this_esdata = from_data(
            f'{test_name_slug}_Time_History{"_"+str(i) if len(time_datasets)>1 else ""}',
            data)
        this_esdata.ordinate_unit = [comment.split('::')[1].strip() for comment in data.flatten().comment1]
        this_esdata.abscissa_unit = 's'
        esdata.append(this_esdata)

    esspec = []
    for i,(data, label) in enumerate(zip([spec_th, response_th, drive_th],
                           ['Specification_Time_History','Control_Time_History','Drive_Time_History'])):
        data = data.flatten()
        this_esdata = from_data(f'{test_name_slug}_{label}',data)
        try:
            if np.all(data.comment1[0] == data.comment1): # All units are the same
                comment = data.comment1[0]
                this_esdata.ordinate_unit = '('+comment.split('::')[1].strip()+')'
            else:
                this_esdata.ordinate_unit = ['('+comment.split('::')[1].strip()+')' for comment in data.comment1]
        except IndexError:
            print(f'Could not automatically assign units to {this_esdata.name}')
        this_esdata.abscissa_unit = 's'
        if i == 0:
            esspec.append(this_esdata)
        else:
            esdata.append(this_esdata)

    if verbose:
        print('Creating ESCDF Metadata Objects')
    
    # Channel table
    eschan = from_rattlesnake_channel_info(
        f'{test_name_slug}_Channel_Table', channel_table)
    # Transient Vibration Parameters
    estrans = from_rattlesnake_transient_parameters(
        f'{test_name_slug}_Params', streaming_data)
    
    # Pull the spec into metadata
    return ((estrans, eschan, esspec[0]), tuple(esdata))

def datasets_from_rattlesnake_modal(
        test_name_slug,
        modal_results_data,
        mode_fits = None,
        resynthesized_frfs = None,
        verbose=False):
    """
    Converts modal data from Rattlesnake into ESCDF Datasets

    Parameters
    ----------
    test_name_slug : str
        A string identifier that will be prepended to the dataset names
    modal_results_data : str or netCDF4.Dataset
        A string pointing to a netCDF4 file or the netCDF4 file loaded as a
        netCDF4.Dataset object.  The file should contain modal results from a
        Rattlesnake modal environment.
    mode_fits : ShapeArray
        A ShapeArray object containing fit modes.  If not supplied, no mode
        dataset will be created.
    resynthesized_frfs : TransferFunctionArray
        A TransferFunctionArray object containing resynthesized FRFs.  If not
        supplied, no resynthesized frf dataset will be created.
    verbose : bool, optional
        If True, progress will be printed. The default is False.

    Raises
    ------
    ImportError
        If the escdf package is not available.

    Returns
    -------
    ESCDF Datasets
        A nested Tuple of ESCDF datasets.  The first index is the metadata and
        the second index is the data.  Metadata includes modal
        parameters and channel table.  Data includes FRFs, Coherence, Fit Modes
        (if supplied) and resynthesized FRFs (if supplied).

    """
    if escdf is None:
        raise ImportError('Could not import module `escdf`, unable to create dataset.')
    if isinstance(modal_results_data,str):
        modal_results_data = nc4.Dataset(modal_results_data)
    
    th, frf, mcoh, channel_table = read_modal_data(modal_results_data)
    th_joined = join(th)
    th_joined.comment1 = th[0].comment1
    frf = frf.flatten()
    
    
    if verbose:
         print('Creating ESCDF Data Objects')
    
    es_th = from_data(
        f'{test_name_slug}_Time_History',th)
    es_th.ordinate_unit = [comment.split('::')[1].strip() for comment in th.flatten().comment1]
    es_th.abscissa_unit = 's'
    
    es_frf = from_data(f'{test_name_slug}_FRF',frf)
    es_frf.abscissa_unit = 'Hz'
    try:
        if np.all(frf.comment1[0] == frf.comment1): # All units are the same
            comment = frf.comment1[0]
            es_frf.ordinate_unit = '/'.join(['('+value.split('::')[1].strip()+')' for value in comment.split('//')])
        else:
            es_frf.ordinate_unit = ['/'.join(['('+value.split('::')[1].strip()+')' for value in comment.split('//')]) for comment in frf.comment1]
    except IndexError:
        print(f'Could not automatically assign units to {es_frf.name}')
        
    es_coh = from_data(
        f'{test_name_slug}_Coherence',mcoh)
    es_coh.ordinate_unit = '-'
    es_coh.abscissa_unit = 'Hz'
    
    additional_data = []
    if mode_fits is not None:
        es_mode = from_shape(f'{test_name_slug}_Fit_Modes',mode_fits)
        additional_data.append(es_mode)
    
    if resynthesized_frfs is not None:
        resynthesized_frfs = resynthesized_frfs.flatten()
        es_frf_res = from_data(f'{test_name_slug}_Resynthesized_FRFs',resynthesized_frfs)
        es_frf_res.abscissa_unit = 'Hz'
        try:
            frf_match = frf[resynthesized_frfs.coordinate]
            if np.all(frf_match.comment1[0] == frf_match.comment1): # All units are the same
                comment = frf_match.comment1[0]
                es_frf_res.ordinate_unit = '/'.join(['('+value.split('::')[1].strip()+')' for value in comment.split('//')])
            else:
                es_frf_res.ordinate_unit = ['/'.join(['('+value.split('::')[1].strip()+')' for value in comment.split('//')]) for comment in frf_match.comment1]
        except (ValueError,IndexError):
            print('Could not automatically assign units to {es_frf_res.name}')
        additional_data.append(es_frf_res)
        
    if verbose:
        print('Creating ESCDF Metadata Objects')
    # Channel table
    eschan = from_rattlesnake_channel_info(
        f'{test_name_slug}_Channel_Table', channel_table)
    
    # System Identification Parameters
    esmod = from_rattlesnake_modal_parameters(
        f'{test_name_slug}_Params', modal_results_data)
    
    return ((esmod, eschan),
            tuple([es_th, es_frf, es_coh] + additional_data))

class ESCDFSettingDialog(QtWidgets.QDialog):
    def __init__(self, data_type, initial_value, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Settings")
        self.data_type = data_type
        self.value = None  # To store the final value if OK is pressed

        # Create layout
        layout = QtWidgets.QVBoxLayout()

        # Add message label
        self.message_label = QtWidgets.QLabel(message)
        layout.addWidget(self.message_label)

        # Create the appropriate input widget based on data type
        if data_type == int:
            self.input_widget = QtWidgets.QSpinBox()
            self.input_widget.setMaximum(999999999)
            self.input_widget.setMinimum(-999999999)
            self.input_widget.setValue(initial_value)
        elif data_type == float:
            self.input_widget = QtWidgets.QDoubleSpinBox()
            self.input_widget.setMaximum(999999999.0)
            self.input_widget.setMinimum(-999999999.0)
            self.input_widget.setValue(initial_value)
        elif data_type == str:
            self.input_widget = QtWidgets.QTextEdit()
            self.input_widget.setText(initial_value)
        elif data_type == list:
            self.input_widget = QtWidgets.QComboBox()
            self.input_widget.addItems(initial_value)
            self.input_widget.setCurrentIndex(0)  # Default to first item
        else:
            raise ValueError("Unsupported data type")

        layout.addWidget(self.input_widget)

        # Add OK and Cancel buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.ok_button = QtWidgets.QPushButton("OK")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Connect buttons to their respective slots
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        self.setLayout(layout)

    def get_input_value(self):
        """Retrieve the value from the input widget."""
        if self.data_type == int:
            return self.input_widget.value()
        elif self.data_type == float:
            return self.input_widget.value()
        elif self.data_type == str:
            return self.input_widget.toPlainText()
        elif self.data_type == list:
            return self.input_widget.currentIndex()
        else:
            return None

    @staticmethod
    def get_value(data_type, initial_value, message, parent=None):
        """
        Static method to create the dialog, show it, and return the value if OK is pressed.
        Returns None if Cancel is pressed.
        """
        dialog = ESCDFSettingDialog(data_type, initial_value, message, parent)
        result = dialog.exec_()  # Show the dialog and wait for user interaction
        if result == QtWidgets.QDialog.Accepted:
            return dialog.get_input_value()
        else:
            return None

# def trace_callback(func):
#     """
#     A decorator to trace the execution of callback functions.
#     Prints the function name before calling it.
#     """
#     def wrapper(*args, **kwargs):
#         print(f"Callback triggered: {func.__name__}")
#         try:
#             return func(*args, **kwargs)
#         except Exception as e:
#             print(f"Caught: {e}")
#     return wrapper

class ESCDFTableModel(QtCore.QAbstractTableModel):
    
    def __init__(self, escdf_property, indices, parent=None):
        super().__init__(parent)
        self.prop = escdf_property
        self.indices = indices
        self.table_dimensions = [dimension_size for index, dimension_size in zip(self.indices, self.prop.shape) if index < 0]
        if len(self.table_dimensions) > 2:
            raise ValueError('Only two dimensions can be scrollable')
        while len(self.table_dimensions) < 2:
            self.table_dimensions.append(1)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return self.table_dimensions[0]

    def columnCount(self, parent=QtCore.QModelIndex()):
        return self.table_dimensions[1]

    def get_slice(self, row, column):
        dynamic_variables = [row, column]
        output_slice = []
        for index in self.indices:
            if index < 0:
                output_slice.append(dynamic_variables.pop(0))
            else:
                output_slice.append(index)
        return tuple(output_slice)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            col = index.column()

            output_slice = self.get_slice(row, col)

            # Fetch the data lazily
            value = self.prop[output_slice]
            return str(value)
        return None
    
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            # Return 0-based counting for headers
            return str(section)
        return None

class DimensionSpinBox(QtWidgets.QSpinBox):
    def textFromValue(self, value):
        # Override to display ":" for -1
        if value == -1:
            return ":"
        return str(value)

    def valueFromText(self, text):
        # Override to interpret ":" as -1
        if text == ":":
            return -1
        return int(text)

    def validate(self, text, pos):
        # Allow ":" as valid input, along with numeric values
        if text == ":":
            return QtGui.QValidator.Acceptable, text, pos
        try:
            int(text)  # Check if the text can be converted to an integer
            return QtGui.QValidator.Acceptable, text, pos
        except ValueError:
            return QtGui.QValidator.Invalid, text, pos

class ESCDFVisualizer(QMainWindow):
    """An interactive window allowing users to explore an ESCDF file"""

    monofont = QtGui.QFont("monospace")
    monofont.setStyleHint(QtGui.QFont.TypeWriter)

    def __init__(self, escdf_file=None):
        """Create an ESCDF Visualizer Window to explore an ESCDF file.
        
        A filename or ESCDF object can be passed as an argment, or it can be
        loaded at a later point.
        
        Parameters
        ----------
        escdf_file : str or ESCDF, optional
            The file or ESCDF object to explore.  If not passed, one can be
            loaded through the graphical user interface.

        Returns
        -------
        None.

        """
        if escdf is None:
            raise ImportError('Could not import module `escdf`, unable to use Visualizer.')
        super().__init__()
        uic.loadUi(os.path.join(os.path.abspath(os.path.dirname(
            os.path.abspath(__file__))), 'escdf_visualizer.ui'), self)
        self.activity_selector : QtWidgets.QListWidget
        self.data_selector : QtWidgets.QListWidget
        self.metadata_selector : QtWidgets.QListWidget
        self.alldata_selector : QtWidgets.QListWidget
        self.allmetadata_selector : QtWidgets.QListWidget
        self.data_property_selector : QtWidgets.QListWidget
        self.metadata_property_selector : QtWidgets.QListWidget
        self.activity_property_string : QtWidgets.QLabel
        self.data_property_string : QtWidgets.QLabel
        self.export_mat_button : QtWidgets.QPushButton
        self.export_npz_button : QtWidgets.QPushButton
        self.plot_coordinate_button : QtWidgets.QPushButton
        self.plot_data_against_metadata_button : QtWidgets.QPushButton
        self.plot_data_button : QtWidgets.QPushButton
        self.plot_data_on_geometry_button : QtWidgets.QPushButton
        self.plot_geometry_button : QtWidgets.QPushButton
        self.plot_shape_button : QtWidgets.QPushButton
        self.tab_widget : QtWidgets.QTabWidget
        for widget in [self.activity_property_string, self.data_property_string]:
            widget.setFont(ESCDFVisualizer.monofont)
        self.setWindowTitle('ESCDF Visualizer')
        self.escdf : escdf.ESCDF
        self.escdf = None
        self.node_size : int
        self.node_size = 5
        self.line_width : int
        self.line_width = 2
        self.label_text_size : int
        self.label_text_size = 10
        self.arrow_size : float
        self.arrow_size = 0.1
        self.opacity : float
        self.opacity = 1.0
        self.undeformed_opacity : float
        self.undeformed_opacity = 0.1
        self.transient_start_time : float
        self.transient_start_time = -10000000.0
        self.transient_end_time : float
        self.transient_end_time = 10000000.0
        self.data_array = []
        self.metadata_array = []
        self.data_dimension_spinboxes = []
        self.metadata_dimension_spinboxes = []
        if escdf_file is not None:
            self.load(escdf_file)
        self.guiplots = []
        self.connect_callbacks()
        self.update_viz_buttons()
        self.show()

    # @trace_callback
    def connect_callbacks(self):
        self.actionLoad_File.triggered.connect(self.select_file)
        self.actionSet_Node_Size.triggered.connect(self.set_node_size)
        self.actionSet_Line_Width.triggered.connect(self.set_line_width)
        self.actionSet_Label_Text_Size.triggered.connect(self.set_label_text_size)
        self.actionSet_Arrow_Size.triggered.connect(self.set_arrow_size)
        self.actionSet_Opacity.triggered.connect(self.set_opacity)
        self.actionSet_Undeformed_Opacity.triggered.connect(self.set_undeformed_opacity)
        self.actionSet_Transient_Start_Time.triggered.connect(self.set_transient_start_time)
        self.actionSet_Transient_End_Time.triggered.connect(self.set_transient_end_time)
        self.activity_selector.currentItemChanged.connect(self.update_activity_data)
        self.data_selector.currentItemChanged.connect(self.update_data_data)
        self.data_selector.itemDoubleClicked.connect(self.go_to_data)
        self.metadata_selector.itemDoubleClicked.connect(self.go_to_metadata)
        self.metadata_selector.currentItemChanged.connect(self.update_metadata_data)
        self.data_property_selector.currentItemChanged.connect(self.update_data_property)
        self.metadata_property_selector.currentItemChanged.connect(self.update_metadata_property)
        self.plot_coordinate_button.clicked.connect(self.plot_coordinate)
        self.plot_data_against_metadata_button.clicked.connect(self.plot_data_against_metadata)
        self.plot_data_button.clicked.connect(self.plot_data)
        self.plot_data_on_geometry_button.clicked.connect(self.plot_data_on_geometry)
        self.plot_geometry_button.clicked.connect(self.plot_geometry)
        self.plot_shape_button.clicked.connect(self.plot_shape)
        self.tab_widget.currentChanged.connect(self.update_tab)
        self.alldata_selector.currentItemChanged.connect(self.update_data_properties)
        self.allmetadata_selector.currentItemChanged.connect(self.update_metadata_properties)

    # @trace_callback
    def update_tab(self, argument=None):
        self.update_viz_buttons()

    # @trace_callback
    def update_activity_data(self, current=None, previous = None):
        index = self.activity_selector.currentRow()
        if index < 0:
            self.activity_property_string.setText('Select Activity to See Properties')
            self.update_viz_buttons()
            return
        activity = self.escdf.activities[index]
        self.activity_property_string.setText('\n'.join(activity.repr().split('\n')[:-2]))
        self.data_selector.clear()
        for data in activity.data:
            name = data.name
            type = data.dataset_type
            self.data_selector.addItem(f'{name} ({type})')
        self.metadata_selector.clear()
        for name in activity.metadata_links:
            metadata = self.escdf.metadata[name]
            type = metadata.dataset_type
            self.metadata_selector.addItem(f'{name} ({type})')
        self.update_viz_buttons()

    # @trace_callback
    def go_to_data(self, item=None):
        found = False
        dataset = self.get_active_data()
        for index, compare_dataset in enumerate(self.data_array):
            if dataset is compare_dataset:
                # print('Found Dataset at {:}'.format(index))
                found = True
                break
        if found:
            self.tab_widget.setCurrentIndex(1)
            self.alldata_selector.setCurrentRow(index)
        else:
            # print('Did not find dataset!')
            raise ValueError("Did not find dataset!")

    # @trace_callback
    def go_to_metadata(self, item=None):
        found = False
        metadata_index = self.metadata_selector.currentRow()
        metadata_name = self.escdf.activities[self.activity_selector.currentRow()].metadata_links[metadata_index]
        for index, compare_dataset in enumerate(self.metadata_array):
            if compare_dataset.name == metadata_name:
                # print('Found Dataset at {:}'.format(index))
                found = True
                break
        if found:
            self.tab_widget.setCurrentIndex(2)
            self.allmetadata_selector.setCurrentRow(index)
        else:
            # print('Did not find dataset!')
            raise ValueError("Did not find dataset!")

    # @trace_callback
    def update_data_properties(self, current=None, previous=None):
        # print(self.alldata_selector.currentRow())
        data_index = self.alldata_selector.currentRow()
        if data_index < 0:
            self.data_property_selector.clear()
            return
        dataset = self.data_array[data_index]
        widget = self.data_property_selector
        self.populate_properties(dataset, widget)

    # @trace_callback
    def update_data_property(self, current=None, previous=None):
        data_index = self.alldata_selector.currentRow()
        data = self.data_array[data_index]
        property_index = self.data_property_selector.currentRow()
        if property_index < 0:
            return
        property_name = data.property_names[property_index]
        prop = getattr(data, property_name)
        # Set up spinboxes for the dimensions
        self.set_up_data_dimension_spinboxes(prop)
        self.update_data_dimension(prop=prop)

    # @trace_callback
    def set_up_data_dimension_spinboxes(self, prop):
        for spinbox in self.data_dimension_spinboxes:
            self.data_dimension_selector_layout.removeWidget(spinbox)
        self.data_dimension_spinboxes.clear()
        if prop is not None:
            for i, dimension in enumerate(prop.shape):
                spinbox = DimensionSpinBox()
                spinbox.blockSignals(True)
                spinbox.setMinimum(-1)
                spinbox.setMaximum(dimension-1)
                spinbox.valueChanged.connect(self.update_data_dimension)
                self.data_dimension_selector_layout.addWidget(spinbox)
                self.data_dimension_spinboxes.append(spinbox)
                if i < 2:
                    spinbox.setValue(-1)
                else:
                    spinbox.setValue(0)
            for spinbox in self.data_dimension_spinboxes:
                spinbox.blockSignals(False)

    # @trace_callback
    def update_data_dimension(self, ind=None, prop=None):
        if prop is None:
            data_index = self.alldata_selector.currentRow()
            data = self.data_array[data_index]
            property_index = self.data_property_selector.currentRow()
            if property_index < 0:
                return
            property_name = data.property_names[property_index]
            prop = getattr(data, property_name)
        if prop is None:
            self.data_table_view.setModel(None)
            return
        indices = [spinbox.value() for spinbox in self.data_dimension_spinboxes]
        model = ESCDFTableModel(prop, indices)
        self.data_table_view.setModel(model)

    # @trace_callback
    def update_metadata_properties(self, current = None, previous = None):
        # print(self.allmetadata_selector.currentRow())
        data_index = self.allmetadata_selector.currentRow()
        if data_index < 0:
            self.metadata_property_selector.clear()
            return
        dataset = self.metadata_array[data_index]
        widget = self.metadata_property_selector
        self.populate_properties(dataset, widget)

    # @trace_callback
    def update_metadata_property(self, current=None, previous=None):
        metadata_index = self.allmetadata_selector.currentRow()
        metadata = self.metadata_array[metadata_index]
        property_index = self.metadata_property_selector.currentRow()
        if property_index < 0:
            return
        property_name = metadata.property_names[property_index]
        prop = getattr(metadata, property_name)
        # Set up spinboxes for the dimensions
        self.set_up_metadata_dimension_spinboxes(prop)
        self.update_metadata_dimension(prop=prop)

    # @trace_callback
    def set_up_metadata_dimension_spinboxes(self, prop):
        for spinbox in self.metadata_dimension_spinboxes:
            self.metadata_dimension_selector_layout.removeWidget(spinbox)
        self.metadata_dimension_spinboxes.clear()
        if prop is not None:
            for i, dimension in enumerate(prop.shape):
                spinbox = DimensionSpinBox()
                spinbox.blockSignals(True)
                spinbox.setMinimum(-1)
                spinbox.setMaximum(dimension-1)
                spinbox.valueChanged.connect(self.update_metadata_dimension)
                self.metadata_dimension_selector_layout.addWidget(spinbox)
                self.metadata_dimension_spinboxes.append(spinbox)
                if i < 2:
                    spinbox.setValue(-1)
                else:
                    spinbox.setValue(0)
            for spinbox in self.metadata_dimension_spinboxes:
                spinbox.blockSignals(False)

    # @trace_callback
    def update_metadata_dimension(self, ind=None, prop=None):
        if prop is None:
            metadata_index = self.allmetadata_selector.currentRow()
            metadata = self.metadata_array[metadata_index]
            property_index = self.metadata_property_selector.currentRow()
            if property_index < 0:
                return
            property_name = metadata.property_names[property_index]
            prop = getattr(metadata, property_name)
        if prop is None:
            self.metadata_table_view.setModel(None)
            return
        indices = [spinbox.value() for spinbox in self.metadata_dimension_spinboxes]
        model = ESCDFTableModel(prop, indices)
        self.metadata_table_view.setModel(model)

    # @trace_callback
    def populate_properties(self, dataset, widget : QtWidgets.QListWidget):
        widget.clear()
        for property_name in dataset.property_names:
            prop = getattr(dataset, property_name)
            if prop is None:
                widget.addItem(f'{property_name}: []')
                continue
            if np.prod(prop.shape) == 1 and not prop.ragged:
                widget.addItem(f'{property_name}: {prop[...]}')
                continue
            widget.addItem(f'{property_name}: {prop.datatype}, {prop.shape}')

    # @trace_callback
    def get_active_data(self):
        activity_index = self.activity_selector.currentRow()
        if activity_index < 0:
            return
        index = self.data_selector.currentRow()
        if index < 0:
            return
        activity = self.escdf.activities[activity_index]
        data = activity.data[index]
        return data

    # @trace_callback
    def update_data_data(self, current = None, previous = None):
        data = self.get_active_data()
        if data is None:
            self.data_property_string.setText('Select Data to See Properties')
            self.update_viz_buttons()
            return
        self.data_property_string.setText(data.repr())
        self.update_viz_buttons()

    # @trace_callback
    def update_metadata_data(self, current = None, previous = None):
        self.update_viz_buttons()

    # @trace_callback
    def select_file(self, checked=False):
        filename, file_filter = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select ESCDF File', filter='ESCDF (*.h5 *.esf)')
        if filename == '':
            return
        self.load(filename)

    # @trace_callback
    def load(self,escdf_file):
        # Load the file
        self.escdf = escdf.ESCDF.load(escdf_file)

        # Update the activity selector
        self.activity_selector.clear()
        for name in self.escdf.activities.names:
            self.activity_selector.addItem(name)

        self.data_array = []
        self.metadata_array = []

        # Update the data selector
        self.alldata_selector.clear()
        for activity in self.escdf.activities:
            for data in activity.data:
                self.alldata_selector.addItem(f'{activity.name}: {data.name} ({data.dataset_type})')
                self.data_array.append(data)
        self.alldata_selector.setCurrentRow(0)
        self.update_data_properties()

        # Update the metadata selector
        self.allmetadata_selector.clear()
        for metadata in self.escdf.metadata:
            self.allmetadata_selector.addItem(f'{metadata.name} ({metadata.dataset_type})')
            self.metadata_array.append(metadata)
        self.allmetadata_selector.setCurrentRow(0)
        self.update_metadata_properties()

    # @trace_callback
    def get_activity_geometry(self):
        activity_index = self.activity_selector.currentRow()
        if activity_index < 0:
            return
        activity_metadata = [self.escdf.metadata[name] for name in self.escdf.activities[activity_index].metadata_links]
        geometry = [
            metadata
            for metadata in activity_metadata
            if metadata.istype("geometry") or metadata.istype("point_cloud")
        ]
        return geometry

    # @trace_callback
    def get_activity_metadata_data(self):
        activity_index = self.activity_selector.currentRow()
        activity_metadata = [self.escdf.metadata[name] for name in self.escdf.activities[activity_index].metadata_links]
        data = [metadata for metadata in activity_metadata if metadata.istype('data')]
        return data

    # @trace_callback
    def find_comparable_data(self, active_data):
        data_comparisions = self.get_activity_metadata_data()
        comparable_data = []
        for data in data_comparisions:
            if data.data_type[...] != active_data.data_type[...]:
                continue
            if data.channel.shape != active_data.channel.shape:
                continue
            if np.any(data.channel[...] != active_data.channel[...]):
                continue
            comparable_data.append(data)
        return comparable_data

    # @trace_callback
    def update_viz_buttons(self):
        if self.tab_widget.currentIndex() != 0:
            self.plot_shape_button.setVisible(False)
            self.plot_data_on_geometry_button.setVisible(False)
            self.plot_data_button.setVisible(False)
            self.plot_data_against_metadata_button.setVisible(False)
            self.plot_geometry_button.setVisible(False)
            self.plot_coordinate_button.setVisible(False)
            return
        geometries = self.get_activity_geometry()
        if geometries is None:
            self.plot_shape_button.setVisible(False)
            self.plot_data_on_geometry_button.setVisible(False)
            self.plot_data_button.setVisible(False)
            self.plot_data_against_metadata_button.setVisible(False)
            self.plot_geometry_button.setVisible(False)
            self.plot_coordinate_button.setVisible(False)
            return
        geom_active = len(geometries) > 0
        self.plot_geometry_button.setVisible(geom_active)
        self.plot_coordinate_button.setVisible(geom_active)
        active_data = self.get_active_data()
        if active_data is None:
            self.plot_shape_button.setVisible(False)
            self.plot_data_on_geometry_button.setVisible(False)
            self.plot_data_button.setVisible(False)
            self.plot_data_against_metadata_button.setVisible(False)
            return
        shape_active = geom_active and active_data.istype('mode')
        self.plot_shape_button.setVisible(shape_active)
        data_on_geom_active = geom_active and active_data.istype('data') and active_data.data_type[...] in [
            'time response', 'frequency response function', 'transmissibility', 'spectrum',
            'impulse_response_function']
        self.plot_data_on_geometry_button.setVisible(data_on_geom_active)
        data_active = active_data.istype('data')
        self.plot_data_button.setVisible(data_active)
        if data_active:
            comparable_data = self.find_comparable_data(active_data)
            data_comparable = len(comparable_data) > 0
        else:
            data_comparable = False
        self.plot_data_against_metadata_button.setVisible(data_comparable)

    # @trace_callback
    def set_node_size(self, checked=False):
        out = ESCDFSettingDialog.get_value(int, self.node_size, 'Set the Node Size', self)
        if out is not None:
            self.node_size = out

    # @trace_callback
    def set_line_width(self, checked=False):
        out = ESCDFSettingDialog.get_value(int, self.line_width, 'Set the Line Width', self)
        if out is not None:
            self.line_width = out

    # @trace_callback
    def set_label_text_size(self, checked=False):
        out = ESCDFSettingDialog.get_value(int, self.label_text_size, 'Set the Text Size for Labels', self)
        if out is not None:
            self.label_text_size = out

    # @trace_callback
    def set_arrow_size(self, checked=False):
        out = ESCDFSettingDialog.get_value(float, self.arrow_size, 'Set the Arrow Size for Degrees of Freedom', self)
        if out is not None:
            self.arrow_size = out

    # @trace_callback
    def set_opacity(self, checked=False):
        out = ESCDFSettingDialog.get_value(float, self.opacity, 'Set the Geometry Opacity', self)
        if out is not None:
            self.opacity = out

    # @trace_callback
    def set_undeformed_opacity(self, checked=False):
        out = ESCDFSettingDialog.get_value(float, self.undeformed_opacity, 'Set the Undeformed Geometry Opacity', self)
        if out is not None:
            self.undeformed_opacity = out

    # @trace_callback
    def set_transient_start_time(self, checked=False):
        out = ESCDFSettingDialog.get_value(float, self.transient_start_time, 'Set the Starting Time for Transient Plots', self)
        if out is not None:
            self.transient_start_time = out

    # @trace_callback
    def set_transient_end_time(self, checked=False):
        out = ESCDFSettingDialog.get_value(float, self.transient_end_time, 'Set the Ending Time for Transient Plots', self)
        if out is not None:
            self.transient_end_time = out

    # @trace_callback
    def plot_coordinate(self):
        geometries = self.get_activity_geometry()
        if len(geometries) > 1:
            index = ESCDFSettingDialog.get_value(list,[geo.name for geo in geometries],'Select Geometry')
            if index is None:
                return
        else:
            index = 0
        geometry = to_geometry(geometries[index])
        coordinates = coordinate_array(geometry.node.id, [1, 2, 3], force_broadcast=True)
        plot_kwargs = {'node_size': self.node_size,
                       'line_width': self.line_width,
                       'show_edges': self.actionPlot_Edges.isChecked(),
                       'label_nodes': self.actionLabel_Nodes.isChecked(),
                       'label_tracelines': self.actionLabel_Tracelines.isChecked(),
                       'label_elements': self.actionLabel_Elements.isChecked(),
                       'label_font_size': self.label_text_size}
        geometry.plot_coordinate(coordinates,
                                 arrow_scale=self.arrow_size,
                                 label_dofs=self.actionLabel_Degrees_of_Freedom.isChecked(),
                                 label_font_size=self.label_text_size,
                                 opacity=self.opacity,
                                 plot_kwargs=plot_kwargs
                                 )

    # @trace_callback
    def plot_data_against_metadata(self):
        data = self.get_active_data()
        compare_data = self.find_comparable_data(data)
        data = to_data(data)
        if len(compare_data) > 1:
            index = ESCDFSettingDialog.get_value(list, [d.name for d in compare_data], 'Select Comparison Data')
            if index is None:
                return
        else:
            index = 0
        compare_data = to_data(compare_data[index])
        if data.function_type == FunctionTypes.POWER_SPECTRAL_DENSITY and self.actionPlot_APSDs.isChecked():
            data = data.get_asd()
            compare_data = compare_data.get_asd()
        self.guiplots.append(GUIPlot(Active_Data=data, Comparison_Data=compare_data))

    # @trace_callback
    def plot_data(self):
        data = to_data(self.get_active_data())
        if data.function_type == FunctionTypes.POWER_SPECTRAL_DENSITY and self.actionPlot_APSDs.isChecked():
            data = data.get_asd()
        self.guiplots.append(data.gui_plot())

    # @trace_callback
    def plot_data_on_geometry(self):
        geometries = self.get_activity_geometry()
        if len(geometries) > 1:
            index = ESCDFSettingDialog.get_value(list,[geo.name for geo in geometries],'Select Geometry')
            if index is None:
                return
        else:
            index = 0
        geometry = to_geometry(geometries[index])
        data = to_data(self.get_active_data())
        # We need to reshape it to response x reference in order to plot it
        if data.coordinate.shape[-1] > 1:
            data = data.reshape_to_matrix()
            # Get the reference coordinates
            refs = data[0].reference_coordinate
            if len(refs) > 1:
                index = ESCDFSettingDialog.get_value(list, refs.string_array().tolist(), 'Select Reference to Plot')
                if index is None:
                    return
            else:
                index = 0
            data = data[:, index]
        plot_kwargs = {'node_size': self.node_size,
                       'line_width': self.line_width,
                       'show_edges': self.actionPlot_Edges.isChecked(),
                       'label_nodes': self.actionLabel_Nodes.isChecked(),
                       'label_tracelines': self.actionLabel_Tracelines.isChecked(),
                       'label_elements': self.actionLabel_Elements.isChecked(),
                       'label_font_size': self.label_text_size}
        if data.function_type in [FunctionTypes.TIME_RESPONSE, FunctionTypes.IMPULSE_RESPONSE_FUNCTION]:
            geometry.plot_transient(data.extract_elements_by_abscissa(
                                        self.transient_start_time,
                                        self.transient_end_time
                                    ), plot_kwargs=plot_kwargs,
                                    undeformed_opacity=self.undeformed_opacity,
                                    deformed_opacity=self.opacity)
        else:
            geometry.plot_deflection_shape(data, plot_kwargs=plot_kwargs,
                                           undeformed_opacity=self.undeformed_opacity,
                                           deformed_opacity=self.opacity)

    # @trace_callback
    def plot_geometry(self):
        geometries = self.get_activity_geometry()
        if len(geometries) > 1:
            index = ESCDFSettingDialog.get_value(list,[geo.name for geo in geometries],'Select Geometry')
            if index is None:
                return
        else:
            index = 0
        geometry = to_geometry(geometries[index])
        plot_kwargs = {'node_size': self.node_size,
                       'line_width': self.line_width,
                       'show_edges': self.actionPlot_Edges.isChecked(),
                       'label_nodes': self.actionLabel_Nodes.isChecked(),
                       'label_tracelines': self.actionLabel_Tracelines.isChecked(),
                       'label_elements': self.actionLabel_Elements.isChecked(),
                       'label_font_size': self.label_text_size,
                       'opacity': self.opacity}
        geometry.plot(**plot_kwargs)

    # @trace_callback
    def plot_shape(self):
        geometries = self.get_activity_geometry()
        if len(geometries) > 1:
            index = ESCDFSettingDialog.get_value(list,[geo.name for geo in geometries],'Select Geometry')
            if index is None:
                return
        else:
            index = 0
        geometry = to_geometry(geometries[index])
        shapes = to_shape(self.get_active_data())
        plot_kwargs = {'node_size': self.node_size,
                       'line_width': self.line_width,
                       'show_edges': self.actionPlot_Edges.isChecked(),
                       'label_nodes': self.actionLabel_Nodes.isChecked(),
                       'label_tracelines': self.actionLabel_Tracelines.isChecked(),
                       'label_elements': self.actionLabel_Elements.isChecked(),
                       'label_font_size': self.label_text_size}
        geometry.plot_shape(shapes, plot_kwargs=plot_kwargs,
                            undeformed_opacity=self.undeformed_opacity,
                            deformed_opacity=self.opacity)
