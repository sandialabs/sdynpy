# -*- coding: utf-8 -*-
"""
Functions for creating mass and stiffness matrices for beam structures.

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
import scipy.linalg as la


def beamkm(node_coords, element_connectivity, bend_direction_1,
           ae, jg, ei1, ei2, mass_per_length, tmmi_per_length):
    '''Compute mass and stiffness matrices for the given beam model

    This function computes the mass and stiffness matrices for the beam model
    defined by the given nodal coordinates, element connectivity, orientation
    parameters, and elastic properties.

    Parameters
    ----------
    node_coords : ndarray
        An nx3 array where node_coords[i,:] is the x,y,z coordinate of the i-th
        node.  The number of rows is equal to the number of nodes in the model.
    element_connectivity : ndarray
        An mx2 array where element_connectivity[i,:] are the row indices of
        node_coords corresponding to the nodes in the i-th element.  The number
        of rows is equal to the number of elements in the model
    bend_direction_1 : ndarray
        An mx3 array where bend_direction_1[i,:] is the x,y,z coordinates of a
        vector that defines the first bending axis of the beam.  The second
        bending axis is computed from the cross product of the beam axis and
        the first bending axis.  The number of rows is equal to the number of
        elements in the model
    ae : ndarray
        A length m array where ae[i] is the axial stiffness of element i.  This
        can be computed from the area of the element (A) times the elastic
        modulus (E).  The length of the array should be equal to the number of
        elements in the model.
    jg : ndarray
        A length m array where jg[i] is the torsional stiffness of element i.
        This can be computed from the torsional constant (J) times the shear
        modulus (G).  The length of the array should be equal to the number of
        elements in the model.
    ei1 : ndarray
        A length m array where ei1[i] is the bending stiffness about axis 1 of
        element i.  This can be computed from the second moment of area of the
        beam's cross section about bending axis 1 (I1) times the elastic
        modulus (E).  The length of the array should be equal to the number of
        elements in the model.
    ei2 : ndarray
        A length m array where ei2[i] is the bending stiffness about axis 2 of
        element i.  This can be computed from the second moment of area of the
        beam's cross section about bending axis 2 (I2) times the elastic
        modulus (E).  The length of the array should be equal to the number of
        elements in the model.
    mass_per_length : ndarray
        A length m array where mass_per_length[i] is the linear density of
        element i.  This can be computed from the density of the material times
        the cross sectional area of the beam.  The length of the array should
        be equal to the number of elements in the model.
    tmmi_per_length : ndarray
        A length m array where tmmi_per_length[i] is the torsional mass moment
        of inertia per unit length of element i.  This can be computed from the
        density of the material times the polar moment of inertia of the beam's
        cross-section.  The length of the array should be equal to the number
        of elements in the model.

    Returns
    -------
    K : ndarray
        The stiffness matrix of the beam model, which will have size 6nx6n
        where n is the number of nodes in the model.
    M : ndarray
        The mass matrix of the beam model, which will have size 6nx6n where n
        is the number of nodes in the model.

    Notes
    -----
    The degrees of freedom in the mass and stiffness matrices will be ordered
    as follows:
        [disp_x_0, disp_y_0, disp_z_0, rot_x_0, rot_y_0, rot_z_0,
        disp_x_1, disp_y_1, disp_z_1, rot_x_1, rot_y_1, rot_z_1,
        ...
        disp_x_n, disp_y_n, disp_z_n, rot_x_n, rot_y_n, rot_z_n]
    '''
    # Validate inputs
    try:
        number_of_nodes = node_coords.shape[0]
        if node_coords.ndim != 2:
            raise ValueError(
                'node_coords should be a 2D array with shape nx3 where n is the number of nodes in the model')
        if node_coords.shape[1] != 3:
            raise ValueError(
                'node_coords should have shape nx3 where n is the number of nodes in the model')
    except AttributeError:
        raise ValueError('node_coords should be a numpy.ndarray')
    try:
        number_of_elements = element_connectivity.shape[0]
        if element_connectivity.ndim != 2:
            raise ValueError(
                'element_connectivity should be a 2D array with shape nx2 where n is the number of elements in the model')
        if element_connectivity.shape[1] != 2:
            raise ValueError(
                'element_connectivity should have shape nx2 where n is the number of elements in the model')
    except AttributeError:
        raise ValueError('element_connectivity should be a numpy.ndarray')
    # Check that the element properties that have been passed are the correct
    # shape
    try:
        if bend_direction_1.shape != (number_of_elements, 3):
            raise ValueError(
                'bend_direction_1 should be a 2D array with shape (element_connectivity.shape[0],3)')
    except AttributeError:
        raise ValueError('bend_direction_1 should be a numpy.ndarray')
    for val in [ae, jg, ei1, ei2, mass_per_length, tmmi_per_length]:
        try:
            if val.shape != (number_of_elements,):
                raise ValueError(
                    'Element Properties (ae,jg,ei1,ei2,mass_per_length,tmmi_per_length) should be 1D arrays with length element_connectivity.shape[0]')
        except AttributeError:
            raise ValueError(
                'Element Properties (ae,jg,ei1,ei2,mass_per_length,tmmi_per_length) should be numpy.ndarray')
    # Initialize
    K = np.zeros((6 * number_of_nodes, 6 * number_of_nodes))
    M = np.zeros((6 * number_of_nodes, 6 * number_of_nodes))

    # Initialize Element Matrices
    PA = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    PT = np.array(
        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    PB1 = np.array(
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    PB2 = np.array(
        [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0]])
    # Loop through all elements and assemble mass and stiffness matrices
    for i, (node_indices, bd1, AE, JG, EI1, EI2, rho, rhoT) in enumerate(zip(element_connectivity,
                                                                             bend_direction_1, ae, jg, ei1, ei2,
                                                                             mass_per_length, tmmi_per_length)):
        # Get element length
        dx = node_coords[node_indices[1], :] - node_coords[node_indices[0], :]
        L = np.linalg.norm(dx)
        # Compute Element Matrices
        KelemA = AE / L * np.array([[1, -1],
                                    [-1, 1]])
        KelemT = JG / L * np.array([[1, -1],
                                    [-1, 1]])
        KelemB1 = EI1 / L**3 * np.array([[12, 6 * L, -12, 6 * L],
                                         [6 * L, 4 * L**2, -6 * L, 2 * L**2],
                                         [-12, -6 * L, 12, -6 * L],
                                         [6 * L, 2 * L**2, -6 * L, 4 * L**2]])
        KelemB2 = EI2 / L**3 * np.array([[12, 6 * L, -12, 6 * L],
                                         [6 * L, 4 * L**2, -6 * L, 2 * L**2],
                                         [-12, -6 * L, 12, -6 * L],
                                         [6 * L, 2 * L**2, -6 * L, 4 * L**2]])
        Kelem = PA.T @ KelemA @ PA + PT.T @ KelemT @ PT + PB1.T @ KelemB1 @ PB1 + PB2.T @ KelemB2 @ PB2

        MelemA = rho * L * np.array([[1 / 3, 1 / 6],
                                    [1 / 6, 1 / 3]])
        MelemT = rhoT * L * np.array([[1 / 3, 1 / 6],
                                     [1 / 6, 1 / 3]])
        MelemB1 = rho * L * np.array([[13 / 35, 11 / 210 * L, 9 / 70, -13 / 420 * L],
                                      [11 / 210 * L, 1 / 105 * L**2, 13 / 420 * L, -1 / 140 * L**2],
                                      [9 / 70, 13 / 420 * L, 13 / 35, -11 / 210 * L],
                                      [-13 / 420 * L, -1 / 140 * L**2, -11 / 210 * L, 1 / 105 * L**2]])
        MelemB2 = rho * L * np.array([[13 / 35, 11 / 210 * L, 9 / 70, -13 / 420 * L],
                                      [11 / 210 * L, 1 / 105 * L**2, 13 / 420 * L, -1 / 140 * L**2],
                                      [9 / 70, 13 / 420 * L, 13 / 35, -11 / 210 * L],
                                      [-13 / 420 * L, -1 / 140 * L**2, -11 / 210 * L, 1 / 105 * L**2]])
        Melem = PA.T @ MelemA @ PA + PT.T @ MelemT @ PT + PB1.T @ MelemB1 @ PB1 + PB2.T @ MelemB2 @ PB2
        # Compute Directions
        d0 = dx / np.linalg.norm(dx)
        d1 = bd1 / np.linalg.norm(bd1)
        d2 = np.cross(d0, d1)
        d2 = d2 / np.linalg.norm(d2)
        d1 = np.cross(d2, d0)
        d1 = d1 / np.linalg.norm(d1)
        C = np.array([d0, d1, d2])

        A = np.zeros((12, 12))
        A[0:3, 0:3] = C
        A[3:6, 3:6] = C
        A[6:9, 6:9] = C
        A[9:12, 9:12] = C

        i1 = np.arange(6 * node_indices[0], 6 * (node_indices[0] + 1))
        i2 = np.arange(6 * node_indices[1], 6 * (node_indices[1] + 1))

        K[np.concatenate((i1, i2))[:, np.newaxis], np.concatenate((i1, i2))] += A.T @ Kelem @ A
        M[np.concatenate((i1, i2))[:, np.newaxis], np.concatenate((i1, i2))] += A.T @ Melem @ A

    return K, M


'''
node_coords = np.array([np.linspace(0,1,10),
                         np.zeros(10),
                         np.zeros(10)]).T

element_connectivity = np.array([np.arange(0,9),np.arange(1,10)]).T

bend_direction_1 = np.array([np.zeros(element_connectivity.shape[0]),np.zeros(element_connectivity.shape[0]),np.ones(element_connectivity.shape[0])]).T

b = 0.01
h = 0.01
E = 69e9
nu = 0.33
rho = 2830
I1 = b*h**3/12
I2 = b**3*h/12
G = E/(2*(1-nu))
J = I1+I2
Ixx_per_L = (1/12)*rho*b*h*(b**2+h**2)

ae =  b*h*E * np.ones(element_connectivity.shape[0])
jg = J*G * np.ones(element_connectivity.shape[0])
ei1 = E*I1 * np.ones(element_connectivity.shape[0])
ei2 = E*I2 * np.ones(element_connectivity.shape[0])
mass_per_length = rho*b*h * np.ones(element_connectivity.shape[0])
tmmi_per_length = Ixx_per_L * np.ones(element_connectivity.shape[0])

K,M = beamkm(node_coords,element_connectivity,bend_direction_1,ae,jg,ei1,ei2,mass_per_length,tmmi_per_length)

# Perform an eigenanalysis
lam,phi = la.eigh(K,M)
lam[0:6] = 0
fn = np.sqrt(lam)/(2*np.pi)
'''


def rect_beam_props(E, rho, nu, b, h=None, nelem=None):
    '''Return beam keyword dictionary for a rectangular beam.

    Returns parameters that can be used in the beamkm function for a beam of
    uniform rectangular cross-section.

    Parameters
    ----------
    E : float
        Young's Modulus
    rho : float
        Density
    nu : float
        Poisson's Ratio
    b : float
        Thickness of the beam in the direction of bend_direction_1
    h : float
        Thickness of the beam in the direction of bend_direction_2.  If it is
        not specified, a square cross section is assumed.
    nelem : int
        Number of elements.  This will repeat the values in the dict nelem
        times, as is required by beamkm.  If not specified, only a single value
        will be output for each entry in the dict.

    Returns
    -------
    kwargs : dict
        A dictionary that can be used in beamkm(**kwargs).

    Notes
    -----
    Steel : SI
        E = 200e9 # [N/m^2],
        nu = 0.25 # [-],
        rho = 7850 # [kg/m^3]
    Aluminum : SI
        E = 69e9 # [N/m^2],
        nu = 0.33 # [-],
        rho = 2830 # [kg/m^3]
    '''
    if h is None:
        h = b
    A = b * h
    I1 = b * h**3 / 12
    I2 = b**3 * h / 12
    G = E / (2 * (1 - nu))
    J = I1 + I2
    Ixx_per_L = (1 / 12) * rho * b * h * (b**2 + h**2)
    return_dict = {}
    return_dict['ae'] = A * E
    return_dict['jg'] = J * G
    return_dict['ei1'] = E * I1
    return_dict['ei2'] = E * I2
    return_dict['mass_per_length'] = rho * A
    return_dict['tmmi_per_length'] = Ixx_per_L
    if nelem is not None:
        for key in return_dict:
            return_dict[key] = return_dict[key] * np.ones((nelem,), float)
    return return_dict


def beamkm_2d(length, width, height, nnodes, E, rho, nu, axial=True):
    '''A simplified 2D beam with uniform cross section and linear materials

    Parameters
    ----------
    length : float
        Length of the beam
    width : float
        Width of the beam, perpendicular to the bending plane
    height : float
        Height of the beam, in the bending direction.
    nnodes : int
        Number of nodes in the beam (number of elements + 1)
    E : float
        Elastic modulus
    rho : float
        Density
    nu : float
        Poisson's Ratio
    axial : bool
        Keep axial motions (default = True)

    Returns
    -------
    K : np.ndarray
        Stiffness matrix of the beam
    M : np.ndarray
        Mass matrix of the beam

    Notes
    -----
    Dof ordering is axial translation, bending translation, and bending
    rotation repeated for each node.  Axial is X, bending is Z, and rotation is
    about Y.  If axial == False, Dof ordering is bending translation, bending
    rotation.
    '''
    node_coords = np.linspace(0, length, nnodes)
    coords_3d = np.zeros((nnodes, 3))
    coords_3d[:, 0] = node_coords
    node_indices = np.arange(nnodes)
    elems = np.array((node_indices[:-1], node_indices[1:])).T
    bend_direction_1 = np.zeros((len(elems), 3))
    bend_direction_1[:, 2] = 1

    props = rect_beam_props(E, rho, nu, width, height, nelem=len(elems))

    K, M = beamkm(coords_3d, elems, bend_direction_1, **props)

    # Eliminate everything except in bend direction 1.  Keep translation in X, Z,
    # and rotation about y
    if axial:
        keep_dofs = ([0, 2, 4] + np.arange(nnodes)[:, np.newaxis] * 6).flatten()
    else:
        keep_dofs = ([2, 4] + np.arange(nnodes)[:, np.newaxis] * 6).flatten()

    K = K[keep_dofs[:, np.newaxis], keep_dofs[np.newaxis, :]]
    M = M[keep_dofs[:, np.newaxis], keep_dofs[np.newaxis, :]]
    return (K, M)
