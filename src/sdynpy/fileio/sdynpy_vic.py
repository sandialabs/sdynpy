# -*- coding: utf-8 -*-
"""
Functions for working with camera output data from VIC3D

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

import zipfile
import xml.etree.ElementTree as et
import numpy as np
from scipy.io import loadmat
from ..core.sdynpy_coordinate import coordinate_array
from ..core.sdynpy_data import data_array, FunctionTypes
from ..core.sdynpy_geometry import Geometry, node_array, coordinate_system_array
from time import time

REPORT_TIME = 5


def extract_vic_cal_parameters(z3d_file):
    """
    Extracts camera calibration parameters from a VIC3D .z3d file.

    Parameters
    ----------
    z3d_file : str
        File path to the VIC3D .z3d file.

    Returns
    -------
    out_data : dict
        A dictionary with keys 'intrinsics_0', 'extrinsics_0', 'distortion_0',
        'intrinsics_1', 'extrinsics_1', and 'distortion_1', which contain the
        intrinsic, extrinsic, and distortion parameters for the left camera (0)
        and right camera (1).

    """
    with zipfile.ZipFile(z3d_file, 'r') as zf:
        with zf.open('project.xml') as pxml:
            xml = ''.join([line.decode() for line in pxml.readlines()])

    xml_data = et.fromstring(xml)
    calibration_info = xml_data.find('calibration')

    camera_infos = calibration_info.findall('camera')

    camera_0_info = [camera for camera in camera_infos if camera.attrib['id'] == '0'][0]
    camera_1_info = [camera for camera in camera_infos if camera.attrib['id'] == '1'][0]

    intrinsics_0 = [float(val) for val in camera_0_info.find('intrinsics').text.split(' ')]
    extrinsics_0 = [float(val) for val in camera_0_info.find('orientation').text.split(' ')]
    distortion_0 = [float(val) for val in camera_0_info.find('distortion').text.split(' ')]
    intrinsics_1 = [float(val) for val in camera_1_info.find('intrinsics').text.split(' ')]
    extrinsics_1 = [float(val) for val in camera_1_info.find('orientation').text.split(' ')]
    distortion_1 = [float(val) for val in camera_1_info.find('distortion').text.split(' ')]

    out_data = {'intrinsics_0': intrinsics_0,
                'extrinsics_0': extrinsics_0,
                'distortion_0': distortion_0,
                'intrinsics_1': intrinsics_1,
                'extrinsics_1': extrinsics_1,
                'distortion_1': distortion_1}

    return out_data


def vic_angles_from_matrix(R):
    '''Extract Bryant Angles from a rotation matrix.

    Extracts rotation angles r_x, r_y, r_z for the x,y,z rotation sequence.

    Parameters
    ----------
    R : np.ndarray
        A 3x3 array representing a rotation matrix.

    Returns
    -------
    rx : float
        Rotation angle about X in radians
    ry : float
        Rotation angle about Y in radians
    rz : float
        Rotation angle about Z in radians
    '''
    if abs(R[0, 2]) < 0.9999:
        ry = -np.arcsin(R[0, 2])
        rz = np.arctan2(R[0, 1] / np.cos(ry), R[0, 0] / np.cos(ry))
        rx = np.arctan2(R[1, 2] / np.cos(ry), R[2, 2] / np.cos(ry))
    else:
        if R[0, 2] > 0:
            ry = -np.pi / 2
            rz = 0.0
            rx = np.arctan2(-R[2, 1], R[1, 1])
        if R[0, 2] < 0:
            ry = np.pi / 2
            rz = 0.0
            rx = np.arctan2(-R[2, 1], R[1, 1])
    return rx, ry, rz


def get_vic_camera_parameters(K, RT):
    '''Computes VIC3D Camera Information for the project.xml file

    This function computes VIC3D Camera Calibration information from a camera
    intrinsic and extrinsic matrix.  This can be placed into a the project.xml
    file inside the unzipped Z3D project file.

    Parameters
    ----------
    K : ndarray
        A 3 x 3 upper triangular array consisting of the camera intrinisc
        parameters.
    RT : ndarray
        A 3 x 4 array where the first three columns are an orthogonal matrix
        denoting a rotation matrix, and the last column is a translation.

    Returns
    -------
    intrinsics : iterable
        A list of parameters that define the camera intrinsic properties.  They
        are ordered cx, cy, fx, fy, s
    extrinsics : iterable
        A list of parameters that define the camera extrinsic properties.  They
        are ordered rx, ry, rz, tx, ty, tz

    '''
    R = RT[:3, :3]
    T = RT[:3, 3, np.newaxis]
    Rw = R.T
    rx, ry, rz = vic_angles_from_matrix(Rw)
    tx, ty, tz = T.flatten()
    fx = K[0, 0]
    fy = K[1, 1]
    s = K[0, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    return [cx, cy, fx, fy, s], [rx * 180 / np.pi, ry * 180 / np.pi, rz * 180 / np.pi, tx * 1000, ty * 1000, tz * 1000]


def matrix_from_bryant_angles(rx, ry, rz):
    '''Computes a rotation matrix from Bryant Angles

    Parameters
    ----------
    rx : float
        Rotation angle about X in radians
    ry : float
        Rotation angle about Y in radians
    rz : float
        Rotation angle about Z in radians

    Returns
    -------
    R : np.ndarray
        A 3x3 array representing a rotation matrix.
    '''
    c1 = np.cos(rx)
    c2 = np.cos(ry)
    c3 = np.cos(rz)
    s1 = np.sin(rx)
    s2 = np.sin(ry)
    s3 = np.sin(rz)
    D = np.array([[1, 0, 0], [0, c1, s1], [0, -s1, c1]])
    C = np.array([[c2, 0, -s2], [0, 1, 0], [s2, 0, c2]])
    B = np.array([[c3, s3, 0], [-s3, c3, 0], [0, 0, 1]])

    return D @ C @ B


def camera_matrix_from_vic_parameters(intrinsics, extrinsics=None):
    '''Computes K and RT Camera matrices from information in the project.xml file

    This function takes VIC3D Camera Calibration information and makes a camera
    intrinsic and extrinsic matrix.  This can be extracted from the project.xml
    file inside the unzipped Z3D project file.

    Parameters
    ----------
    intrinsics : iterable
        A list of parameters that define the camera intrinsic properties.  They
        are ordered cx, cy, fx, fy, s
    extrinsics : iterable
        A list of parameters that define the camera extrinsic properties.  They
        are ordered rx, ry, rz, tx, ty, tz

    Returns
    -------
    K : ndarray
        A 3 x 3 upper triangular array consisting of the camera intrinisc
        parameters.
    RT : ndarray
        A 3 x 4 array where the first three columns are an orthogonal matrix
        denoting a rotation matrix, and the last column is a translation.
    '''
    K = np.array([[intrinsics[2], intrinsics[4], intrinsics[0]],
                  [0, intrinsics[3], intrinsics[1]],
                  [0, 0, 1]])
    if extrinsics is not None:
        rotations = [r * np.pi / 180 for r in extrinsics[:3]]
        T = np.array([t / 1000 for t in extrinsics[3:]])[:, np.newaxis]
        Rw = matrix_from_bryant_angles(*rotations)
        R = Rw.T
        RT = np.concatenate((R, T), axis=1)
        return K, RT
    else:
        return K


def read_vic3D_mat_files(files, read_3D=True, read_2D=False, read_quality=False,
                         sigma_tol=0.0, element_triangulation_condition=3.0, dt=1.0,
                         element_color_order=[1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14],
                         allow_dropouts=False):
    """
    Reads in data from Correlated Solutions' VIC3D

    Parameters
    ----------
    files : Iterable of str
        List of strings pointing to .mat files exported from VIC3D.
    read_3D : bool, optional
        Flag specifying whether or not to extract 3D information. The default
        is True.
    read_2D : bool, optional
        Flag specifying whether or not to extract 2D (pixel) information. The
        default is False.
    read_quality : bool, optional
        Flag specifying whether or not to output quality (sigma) information.
        Note that the quality value will be read regardless in order to discard
        bad subsets.  This flag only specifies whether the values are returned
        to the user.  The default is False.
    sigma_tol : float, optional
        Tolerance used to discard bad subsets. The default is 0.0.
    element_triangulation_condition : float, optional
        Maximum condition number for triangular elements generated from subset.
        positions.  The default is 3.0.
    dt : float, optional
        Time spacing to be used in the returned TimeHistoryArrays.
        The default is 1.0.
    element_color_order : Iterable, optional
        Specifies the color order used when creating elements from the various
        areas of interest in the VIC3D output.  The first area of interest will
        have color specified by the first entry in the array.  The array will
        loop around if more areas of interest exist than values in the array.
        The default is [1,3,5,7,9,11,13,15,2,4,6,8,10,12,14].
    allow_dropouts : bool, optional
        Specifies whether or not to allow data to drop out (True) or if the
        entire point is discarded if any dropouts are detected (False).
        Default is False.

    Returns
    -------
    geometry_3D : Geometry
        Geometry object consisting of the 3D node positions from the test.
        Only returned if read_3D is True
    time_data_3D : TimeHistoryArray
        3D Displacement data from the test.  Only returned if read_3D is True.
    geometry_2D : Geometry
        Geometry object consisting of the 2D (pixel) node positions from the
        test.  Only returned if read_2D is True
    time_data_2D : TimeHistoryArray
        2D Pixel Displacement data from the test.  Only returned if read_2D is
        True.
    time_data_2D_disparity : TimeHistoryArray
        2D Pixel Displacement data from the test for the second camera.  Adding
        this array to the original pixel positions will result in the pixel
        positions over time for the right image.  Only returned if read_2D is
        True.
    sigma_data : TimeHistoryArray
        Data quality metric for the test over time.  Only returned if
        read_quality is True

    """
    start_time = time()
    total_times = len(files)
    for ifile, file in enumerate(files):
        this_time = time()
        if this_time - start_time > REPORT_TIME:
            print('Reading File {:} of {:} ({:0.2f}%)'.format(
                ifile + 1, total_times, (ifile + 1) / total_times * 100))
            start_time += REPORT_TIME
        data = loadmat(file)
        # print('Reading Timestep {:}'.format(ifile))
        # Set up the initial measurement
        if ifile == 0:
            # print('  Setting up variables...')
            # Get the area of interest flags
            aois = sorted([key.replace('sigma', '') for key in data.keys() if 'sigma' in key])
            # Get the degrees of freedom numbers
            ndofs = [np.prod(data['sigma' + aoi].shape) for aoi in aois]
            total_dofs = np.sum(ndofs)
            boundaries = np.cumsum(ndofs)
            boundaries = np.concatenate(([0], boundaries))
            aoi_slices = [slice(boundaries[i], boundaries[i + 1]) for i in range(len(ndofs))]
            # Set up variables
            variable_array = {}
            variable_array['sigma'] = np.empty((total_dofs, total_times))
            variable_array['x'] = np.empty(total_dofs)
            variable_array['y'] = np.empty(total_dofs)
            if read_3D:
                variable_array['X'] = np.empty((total_dofs))
                variable_array['Y'] = np.empty((total_dofs))
                variable_array['Z'] = np.empty((total_dofs))
                variable_array['U'] = np.empty((total_dofs, total_times))
                variable_array['V'] = np.empty((total_dofs, total_times))
                variable_array['W'] = np.empty((total_dofs, total_times))
            if read_2D:
                variable_array['u'] = np.empty((total_dofs, total_times))
                variable_array['v'] = np.empty((total_dofs, total_times))
                variable_array['q'] = np.empty((total_dofs, total_times))
                variable_array['r'] = np.empty((total_dofs, total_times))
            # print('  ...done')
        # Now read in the data
        for name, array in variable_array.items():
            # print('  Reading variable {:}'.format(name))
            if array.ndim == 1 and ifile > 0:
                # print('    Skipping Variable {:} for timestep {:}'.format(name,ifile))
                continue
            for iaoi, aoi in enumerate(aois):
                # print('    Reading variable {:}{:} from AoI {:}'.format(name,aoi,iaoi))
                key_name = name + aoi
                key_data = data[key_name]
                array_index = (aoi_slices[iaoi],) + ((ifile,) if array.ndim > 1 else ())
                array[array_index] = key_data.flatten()
    # Now do some bookkeeping activities
    original_aoi_indices = np.empty(total_dofs)
    for i, aoi_slice in enumerate(aoi_slices):
        original_aoi_indices[aoi_slice] = i + 1
    # Reduce to just "good" dofs
    if allow_dropouts:
        # Only remove data if the first sample is bad
        good_dofs = variable_array['sigma'][..., 0] > sigma_tol
    else:
        # Remove data if any sample is bad
        good_dofs = np.all(variable_array['sigma'] > sigma_tol, axis=-1)
    for variable in variable_array:
        variable_array[variable] = variable_array[variable][good_dofs]
    original_aoi_indices = original_aoi_indices[good_dofs]
    boundaries = np.cumsum([np.count_nonzero(original_aoi_indices == i + 1)
                           for i in range(len(aois))])
    boundaries = np.concatenate(([0], boundaries))
    aoi_slices = [slice(boundaries[i], boundaries[i + 1]) for i in range(len(ndofs))]
    # Create node numbers for geometry
    num_nodes_per_aoi = np.diff(boundaries)
    node_length = max([len(str(num)) for num in num_nodes_per_aoi])
    node_numbers = np.empty(boundaries[-1])
    for i, (num_nodes, aoi_slice) in enumerate(zip(num_nodes_per_aoi, aoi_slices)):
        node_numbers[aoi_slice] = np.arange(num_nodes) + (i + 1) * 10**node_length + 1
    # Start accumulating output variables
    output_variables = []
    # Create the 2D geometry regardless, used for element triangulation
    node_positions = np.array([variable_array[val] for val in 'xy'])
    node_positions = np.concatenate(
        [node_positions, np.zeros((1, node_positions.shape[-1]))], axis=0)
    geometry_2D = Geometry(
        node_array(node_numbers, node_positions.T),
        coordinate_system_array(1))
    # Create triangulation
    element_arrays = [geometry_2D.node[aoi_slice].triangulate(
        geometry_2D.coordinate_system, condition_threshold=element_triangulation_condition) for aoi_slice in aoi_slices
        if geometry_2D.node[aoi_slice].size > 3]
    for i in range(len(element_arrays)):
        element_arrays[i].color = element_color_order[i % len(element_color_order)]
    geometry_2D.element = np.concatenate(element_arrays)
    if read_3D:
        # Create the geometry
        node_positions = np.array([variable_array[val] for val in 'XYZ'])
        geometry_3D = Geometry(
            node_array(node_numbers, node_positions.T),
            coordinate_system_array(1))
        geometry_3D.element = np.concatenate(element_arrays)
        output_variables.append(geometry_3D)
        # Create time history
        node_displacements = np.array([variable_array[val]
                                      for val in 'UVW']).reshape(-1, total_times)
        node_dofs = coordinate_array(node_numbers, np.array([1, 2, 3])[:, np.newaxis]).flatten()
        timesteps = np.arange(total_times) * dt
        time_data_3D = data_array(FunctionTypes.TIME_RESPONSE, timesteps, node_displacements,
                                  node_dofs[:, np.newaxis])
        output_variables.append(time_data_3D)
    if read_2D:
        output_variables.append(geometry_2D)
        # Create time history
        node_displacements = np.array([variable_array[val]
                                      for val in 'uv']).reshape(-1, total_times)
        node_dofs = coordinate_array(node_numbers, np.array([1, 2])[:, np.newaxis]).flatten()
        timesteps = np.arange(total_times) * dt
        time_data_2D = data_array(FunctionTypes.TIME_RESPONSE, timesteps, node_displacements,
                                  node_dofs[:, np.newaxis])
        output_variables.append(time_data_2D)
        # Create time history for other camera
        node_displacements = np.array([variable_array[val]
                                      for val in 'qr']).reshape(-1, total_times)
        node_dofs = coordinate_array(node_numbers, np.array([1, 2])[:, np.newaxis]).flatten()
        timesteps = np.arange(total_times) * dt
        time_data_2D = data_array(FunctionTypes.TIME_RESPONSE, timesteps, node_displacements,
                                  node_dofs[:, np.newaxis])
        output_variables.append(time_data_2D)
    if read_quality:
        sigma_data = variable_array['sigma']
        sigma_dofs = coordinate_array(node_numbers, 0)
        time_data = data_array(FunctionTypes.TIME_RESPONSE, timesteps, sigma_data,
                               sigma_dofs[:, np.newaxis])
        output_variables.append(time_data)
    return output_variables
