# -*- coding: utf-8 -*-
"""
Tools for modeling shakers, and includes commercial shaker objects.

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
import matplotlib.pyplot as plt

# def shaker_4dof(m_armature,m_body,m_forcegauge,
#                 k_suspension,k_stinger,
#                 c_suspension,c_stinger,
#                 resistance,inductance,force_factor):
#     '''Creates a four degree of freedom electromechanical model of a shaker

#     Parameters
#     ----------
#     m_armature : float
#         Mass of the armaturesistance
#     m_body : float
#         Mass of the body and trunion of the shaker
#     m_forcegauge : float
#         Mass of the force gauge at the tip of the stinger
#     k_suspension : float
#         Stiffness of the spring between the body and armature
#     k_stinger : float
#         Stiffness of the stinger between the armature and force gauge
#     c_suspension : float
#         Damping of the spring between the body and armature
#     c_stinger : float
#         Damping of the stinger between the armature and force gauge
#     resistance : float
#         Coil resistance of the electronics portion of the shaker
#     inductance : float
#         Coil inductance of the electronics portion of the shaker
#     force_factor : float
#         Force factor BL of the magnet and coil

#     Returns
#     -------
#     M_shaker : ndarray
#         The mass matrix of the shaker
#     C_shaker : ndarray
#         The damping matrix of the shaker
#     K_shaker : ndarray
#         The stiffness matrix of the shaker

#     Notes
#     -----
#     The dof order of the responses are:
#     Displacement of Armature,
#     Displacement of Body,
#     Displacement of Force Gauge,
#     Shaker Current.

#     The dof order of the inputs are:
#     Force into Armature
#     Force into Body
#     Force into Force Gauge
#     Shaker Voltage.

#     The force imparted by the shaker can be computed as the stinger stiffness
#     times the difference in displacement of the armature and force gauge plus
#     the stinger damping times the difference in the velocity of the armature
#     and force gauge.
#     '''

#     # Cresistanceate the matrices
#     M_shaker = np.array([[m_armature,  0,  0,  0],
#                          [ 0, m_body,  0,  0],
#                          [ 0,  0, m_forcegauge,  0],
#                          [ 0,  0,  0,  0]])
#     C_shaker = np.array([[(c_suspension+c_stinger), -c_suspension, -c_stinger, 0],
#                          [     -c_suspension,  c_suspension,    0, 0],
#                          [     -c_stinger,    0,  c_stinger, 0],
#                          [       force_factor,  -force_factor, 0,   inductance]])
#     K_shaker = np.array([[(k_suspension+k_stinger), -k_suspension, -k_stinger, -force_factor],
#                          [     -k_suspension,  k_suspension,    0,  force_factor],
#                          [     -k_stinger,    0,  k_stinger,   0],
#                          [        0,    0,    0,  resistance]])

#     return M_shaker,C_shaker,K_shaker

# test_shaker_dakota = shaker_4dof( m_armature = 0.227, # [kg] ~ armaturesistance mass
#                                   m_body = 24.9, # [kg] ~ body mass
#                                   k_suspension = 2312, # [N/m] ~ suspension stiffness
#                                   c_suspension = 20, # [N/(m/s)] damping ...
#                                   k_stinger = 4e6, # [N/m] stiff stinger spring
#                                   m_forcegauge = 1e-2, # [kg] force gauge mass
#                                   c_stinger = 10, # [N/(m/s)] damping
#                                   resistance = 1.3, # [ohm] ~ coil resistance
#                                   inductance = 100e-6, # [Henry] ~ coil inductance
#                                   force_factor = 30 # [-] ~ Force Factor of magnet & coil
#                                 )


class Shaker4DoF:
    """A class defining a four degree-of-freedom model of a shaker

    """

    def __init__(self, m_armature, m_body, m_forcegauge,
                 k_suspension, k_stinger,
                 c_suspension, c_stinger,
                 resistance, inductance, force_factor):
        '''Creates a four degree of freedom electromechanical model of a shaker

        Parameters
        ----------
        m_armature : float
            Mass of the armature
        m_body : float
            Mass of the body and trunion of the shaker
        m_forcegauge : float
            Mass of the force gauge at the tip of the stinger
        k_suspension : float
            Stiffness of the spring between the body and armature
        k_stinger : float
            Stiffness of the stinger between the armature and force gauge
        c_suspension : float
            Damping of the spring between the body and armature
        c_stinger : float
            Damping of the stinger between the armature and force gauge
        resistance : float
            Coil resistance of the electronics portion of the shaker
        inductance : float
            Coil inductance of the electronics portion of the shaker
        force_factor : float
            Force factor BL of the magnet and coil
        '''
        self.m_armature = m_armature
        self.m_body = m_body
        self.m_forcegauge = m_forcegauge
        self.k_suspension = k_suspension
        self.k_stinger = k_stinger
        self.c_suspension = c_suspension
        self.c_stinger = c_stinger
        self.resistance = resistance
        self.inductance = inductance
        self.force_factor = force_factor

    def MCK(self):
        """
        Returns mass, damping, and stiffness matrices for the shakers.

        Returns
        -------
        M : np.ndarray
            The mass matrix of the shaker
        C : np.ndarray
            The damping matrix of the shaker
        K : np.ndarray
            The stiffness matrix of the shaker

        Notes
        -----
        The dof order of the responses are:
        Displacement of Armature,
        Displacement of Body,
        Displacement of Force Gauge,
        Shaker Current.

        The dof order of the inputs are:
        Force into Armature
        Force into Body
        Force into Force Gauge
        Shaker Voltage.

        The force imparted by the shaker can be computed as the stinger stiffness
        times the difference in displacement of the armature and force gauge plus
        the stinger damping times the difference in the velocity of the armature
        and force gauge.

        """
        M = np.array([[self.m_armature, 0, 0, 0], [0, self.m_body, 0, 0],
                     [0, 0, self.m_forcegauge, 0], [0, 0, 0, 0]])
        C = np.array([[(self.c_suspension + self.c_stinger), -self.c_suspension, -self.c_stinger, 0],
                      [-self.c_suspension, self.c_suspension, 0, 0],
                      [-self.c_stinger, 0, self.c_stinger, 0],
                      [self.force_factor, -self.force_factor, 0, self.inductance]])
        K = np.array([[(self.k_suspension + self.k_stinger), -self.k_suspension, -self.k_stinger, -self.force_factor],
                      [-self.k_suspension, self.k_suspension, 0, self.force_factor],
                      [-self.k_stinger, 0, self.k_stinger, 0],
                      [0, 0, 0, self.resistance]])
        return M, C, K

    def state_space(self):
        """
        Returns the state space formulation of the shaker model of the form

        x_dot = A@x + B@u
        y = C@x + D@u

        Returns
        -------
        A : np.ndarray
            State (system) Matrix.
        B : np.ndarray
            Input Matrix
        C : np.ndarray
            Output Matrix
        D : np.ndarray
            Feedthrough (feedforward) Matrix

        Notes
        -----
        The dof order of the state matrix is
          0. armature displacement
          1. body displacement
          2. forcegauge displacement,
          3. armature velocity
          4. body velocity
          5. forcegauge velocity
          6. current.

        The dof order of the input matrix is
          0. external force on armature
          1. external force on body
          2. external force on forcegauge
          3. shaker voltage

        The dof order for the output matries is
          0. Displacement of armature
          1. Velocity of armature
          2. Acceleration of armature
          3. External force on armature
          4. Displacement of body
          5. Velocity of body
          6. Acceleration of body
          7. External force on body
          8. Displacement of forcegauge
          9. Velocity of forcegauge
          10. Acceleration of forcegauge
          11. External force on forcegauge
          12. Shaker voltage
          13. Shaker current
          14. Current change per time
          15. Force on coil due to electronics
          16. Force in stinger
        """
        A = np.array([[0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [(-self.k_suspension - self.k_stinger) / self.m_armature,
                       self.k_suspension / self.m_armature,
                       self.k_stinger / self.m_armature,
                       (-self.c_suspension
                        - self.c_stinger) / self.m_armature,
                       self.c_suspension / self.m_armature,
                       self.c_stinger / self.m_armature,
                       self.force_factor / self.m_armature],
                      [self.k_suspension / self.m_body, -self.k_suspension / self.m_body, 0, self.c_suspension /
                          self.m_body, -self.c_suspension / self.m_body, 0, -self.force_factor / self.m_body],
                      [self.k_stinger / self.m_forcegauge, 0, -self.k_stinger / self.m_forcegauge,
                          self.c_stinger / self.m_forcegauge, 0, -self.c_stinger / self.m_forcegauge, 0],
                      [0, 0, 0, -self.force_factor / self.inductance, self.force_factor / self.inductance, 0, -self.resistance / self.inductance]])
        B = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [1 / self.m_armature, 0, 0, 0],
                      [0, 1 / self.m_body, 0, 0],
                      [0, 0, 1 / self.m_forcegauge, 0],
                      [0, 0, 0, 1 / self.inductance]])
        C = np.array([
            # States: x1, x2, x3, xd1, xd2, xd3, i
                     [1, 0, 0, 0, 0, 0, 0],  # Displacement of mass 1
                     [0, 0, 0, 1, 0, 0, 0],  # Velocity of mass 1
                     [(-self.k_suspension - self.k_stinger) / self.m_armature, self.k_suspension / self.m_armature,
                      self.k_stinger / self.m_armature, (-self.c_suspension - self.c_stinger) / \
                      self.m_armature, self.c_suspension / self.m_armature,
                      self.c_stinger / self.m_armature, self.force_factor / self.m_armature],  # Acceleration of mass 1
                     [0, 0, 0, 0, 0, 0, 0],  # External Force on mass 1
                     [0, 1, 0, 0, 0, 0, 0],  # Displacement of mass 2
                     [0, 0, 0, 0, 1, 0, 0],  # Velocity of mass 2
                     [self.k_suspension / self.m_body, -self.k_suspension / self.m_body, 0, self.c_suspension / self.m_body, - \
                         self.c_suspension / self.m_body, 0, -self.force_factor / self.m_body],  # Acceleration of mass 2
                     [0, 0, 0, 0, 0, 0, 0],  # External Force on mass 2
                     [0, 0, 1, 0, 0, 0, 0],  # Displacement of mass 3
                     [0, 0, 0, 0, 0, 1, 0],  # Velocity of mass 3
                     [self.k_stinger / self.m_forcegauge, 0, -self.k_stinger / self.m_forcegauge, self.c_stinger / \
                         self.m_forcegauge, 0, -self.c_stinger / self.m_forcegauge, 0],  # Acceleration of mass 3
                     [0, 0, 0, 0, 0, 0, 0],  # External Force on mass 3
                     [0, 0, 0, 0, 0, 0, 0],  # Voltage
                     [0, 0, 0, 0, 0, 0, 1],  # Current
                     [0, 0, 0, -self.force_factor / self.inductance, self.force_factor / self.inductance,
                         0, -self.resistance / self.inductance],  # Current change per time
                     [0, 0, 0, 0, 0, 0, self.force_factor],  # Force on coil
                     [self.k_stinger, 0, -self.k_stinger, self.c_stinger,
                         0, -self.c_stinger, 0],  # Force in stinger
                     ])

        D = np.array([
            # Inputs: f_armature, f_body, f_forcegauge, e
                     [0, 0, 0, 0],  # Displacement of mass 1
                     [0, 0, 0, 0],  # Velocity of mass 1
                     [1 / self.m_armature, 0, 0, 0],  # Acceleration of mass 1
                     [1, 0, 0, 0],  # External Force on mass 1
                     [0, 0, 0, 0],  # Displacement of mass 2
                     [0, 0, 0, 0],  # Velocity of mass 2
                     [0, 1 / self.m_body, 0, 0],  # Acceleration of mass 2
                     [0, 1, 0, 0],  # External Force on mass 2
                     [0, 0, 0, 0],  # Displacement of mass 3
                     [0, 0, 0, 0],  # Velocity of mass 3
                     [0, 0, 1 / self.m_forcegauge, 0],  # Acceleration of mass 3
                     [0, 0, 1, 0],  # External Force on mass 3
                     [0, 0, 0, 1],  # Voltage
                     [0, 0, 0, 0],  # Current
                     [0, 0, 0, 1 / self.inductance],  # Current change per time
                     [0, 0, 0, 0],  # Force on coil
                     [0, 0, 0, 0],  # Force in stinger
                     ])
        return A, B, C, D

    def transfer_function(self, frequencies):
        """
        Computes the shaker transfer function matrix

        Returns
        -------
        H : np.ndarray
            A transfer function matrix.

        """
        angular_frequencies = 2 * np.pi * frequencies[:, np.newaxis, np.newaxis]
        M, C, K = self.MCK()
        H = np.linalg.inv(-angular_frequencies**2 * M + 1j * angular_frequencies * C + K)
        return H

    def plot_electrical_impedance(self, frequencies=np.arange(4000) + 1, test_data=None):
        """
        Plots the electrical impedance voltage/current for the shaker model
        and compares to test data if supplied

        Parameters
        ----------
        frequencies : np.array, optional
            Frequency lines. The default is np.arange(4000).
        test_data : np.ndarray, optional
            Test data defined at frequencies. The default is None.


        """
        fig, ax = plt.subplots(2, 1, sharex=True)
        H = self.transfer_function(frequencies)[..., -1, -1]
        ax[0].plot(frequencies, np.real(1 / H))
        if test_data is not None:
            ax[0].plot(frequencies, np.real(test_data))
        ax[1].plot(frequencies, np.imag(1 / H))
        if test_data is not None:
            ax[1].plot(frequencies, np.real(test_data))

    @classmethod
    def modal_shop_100lbf(cls):
        """
        Returns a shaker model for a Modal Shop Model 2100E11 100lbf modal shaker.

        Returns
        -------
        shaker_model : Shaker4DoF
            A 4 degree of freedom model of the Modal Shop 2100E11 100lbf modal
            shaker.

        Notes
        -----
        The shaker uses the following parameters
         - m_armature = 0.44 kg
         - m_body = 15 kg
         - m_forcegauge = 1e-2 kg
         - k_suspension = 1.1e5 N/m
         - k_stinger = 9.63e6 N/m
         - c_suspension = 9.6 Ns/m
         - c_stinger = 0.42 Ns/m
         - resistance = 4 Ohm
         - inductance = 6e-4 Henry
         - force_factor = 36 (-)

        """
        return cls(m_armature=0.44,
                   m_body=15,
                   m_forcegauge=1e-2,
                   k_suspension=1.1e5,
                   k_stinger=9.63e6,
                   c_suspension=9.6,
                   c_stinger=0.42,
                   resistance=4,
                   inductance=6e-4,
                   force_factor=36)

# shaker = Shaker4DoF.modal_shop_100lbf()
# shaker.m_forcegauge = 2.6
# shaker.plot_electrical_impedance()
