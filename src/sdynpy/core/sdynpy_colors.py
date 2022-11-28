# -*- coding: utf-8 -*-
"""
Defines the color scheme used in SDynPy.  Matches the I-deas color scheme.

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

from matplotlib.colors import ListedColormap

colors = {0: [0.0, 0.0, 0.0],
          1: [0.0, 0.0, 1.0],
          2: [0.0, 0.329, 1.0],
          3: [0.0, 0.66, 1.0],
          4: [0.0, 1.0, 1.0],
          5: [0.0, 0.33, 0.0],
          6: [0.0, 0.66, 0.0],
          7: [0.0, 1.0, 0.0],
          8: [1.0, 1.0, 0.0],
          9: [1.0, 0.66, 0.0],
          10: [1.0, 0.33, 0.0],
          11: [1.0, 0.0, 0.0],
          12: [1.0, 0.0, 1.0],
          13: [1.0, 0.33, 1.0],
          14: [1.0, 0.66, 1.0],
          15: [1.0, 1.0, 1.0]}

color_list = [colors[i] for i in range(len(colors))]

colormap = ListedColormap(color_list)

coord_colormap = ListedColormap([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]])
