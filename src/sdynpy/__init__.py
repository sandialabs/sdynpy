"""SDynPy: A Structural Dynamics Library for Python

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
from .core import coordinate, colors, array, geometry, shape, data, system, matrix_mod
from .fileio import unv, uff, rattlesnake, vic, tshaker
from .fem.sdynpy_exodus import Exodus, ExodusInMemory
from .fem import sdynpy_beam as beam
from .fem import sdynpy_shaker as shaker
from .fem import sdynpy_dof as dof
from .signal_processing import (frf, cpsd, integration, correlation, complex,
                                rotation, generator, camera, harmonic,
                                geometry_fitting, srs, lrm)
from .modal import (PolyPy, SMAC, PolyPy_GUI, SMAC_GUI, compute_residues,
                    compute_shapes, SignalProcessingGUI, ColoredCMIF,
                    read_modal_fit_data, ModalTest)
from . import doc

__version__ = "0.14.1"

# Pull things in for easier access
SdynpyArray = array.SdynpyArray
coordinate_array = coordinate.coordinate_array
CoordinateArray = coordinate.CoordinateArray
coordinate_system_array = geometry.coordinate_system_array
CoordinateSystemArray = geometry.CoordinateSystemArray
node_array = geometry.node_array
NodeArray = geometry.NodeArray
traceline_array = geometry.traceline_array
TracelineArray = geometry.TracelineArray
element_array = geometry.element_array
ElementArray = geometry.ElementArray
Geometry = geometry.Geometry
shape_array = shape.shape_array
ShapeArray = shape.ShapeArray
data_array = data.data_array
NDDataArray = data.NDDataArray
TimeHistoryArray = data.TimeHistoryArray
TransferFunctionArray = data.TransferFunctionArray
CoherenceArray = data.CoherenceArray
MultipleCoherenceArray = data.MultipleCoherenceArray
PowerSpectralDensityArray = data.PowerSpectralDensityArray
SpectrumArray = data.SpectrumArray
GUIPlot = data.GUIPlot
CPSDPlot = data.CPSDPlot
id_map = geometry.id_map
System = system.System
matrix_plot = correlation.matrix_plot
Matrix = matrix_mod.Matrix
matrix = matrix_mod.matrix
