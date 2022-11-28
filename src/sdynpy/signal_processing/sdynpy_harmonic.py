# -*- coding: utf-8 -*-
"""
Functions for dealing with sinusoidal data

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
import numpy.linalg as la


def harmonic_mag_phase(ts, xs, frequency, nharmonics=1):
    A = np.zeros((ts.size, nharmonics * 2 + 1))
    for i in range(nharmonics):
        A[:, i] = np.sin(2 * np.pi * frequency * (i + 1) * ts)
        A[:, nharmonics + i] = np.cos(2 * np.pi * frequency * (i + 1) * ts)
    A[:, -1] = np.ones(ts.size)
    coefs = la.lstsq(A, xs[:, np.newaxis])[0]
    As = np.array(coefs[:nharmonics, :])
    Bs = np.array(coefs[nharmonics:nharmonics * 2, :])
    phases = np.arctan2(Bs, As)[:, 0]
    magnitudes = np.sqrt(As**2 + Bs**2)[:, 0]
    return magnitudes, phases
