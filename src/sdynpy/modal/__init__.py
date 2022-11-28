"""Functions and classes for performing Experimental Modal Analysis

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
along with this program.  If not, see <https://www.gnu.org/licenses/>."""
from .sdynpy_polypy import PolyPy, PolyPy_GUI
from .sdynpy_smac import SMAC, SMAC_GUI
from .sdynpy_modeshape import compute_residues, compute_shapes
from .sdynpy_signal_processing_gui import SignalProcessingGUI
from .sdynpy_ccmif import ColoredCMIF
