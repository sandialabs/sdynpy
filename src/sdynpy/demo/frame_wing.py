"""Generates a simplified model of the frame/wing structure from the SEM/IMAC Substructuring Focus
Group.  It's not a great model because it uses beam finite elements, but it should be able to be
used for example problems where the dynamics are somewhat similar to those in the real model.
"""

import numpy as np

# from ..core.sdynpy_coordinate import (from_nodelist as _from_nodelist,
#                                       coordinate_array as _coordinate_array)
# from ..fem.sdynpy_beam import beamkm as _beamkm, rect_beam_props as _rect_beam_props
# from ..core.sdynpy_geometry import (Geometry as _Geometry,
#                                     node_array as _node_array,
#                                     traceline_array as _traceline_array,
#                                     coordinate_system_array as _coordinate_system_array)
# from ..core.sdynpy_system import System as _System
from sdynpy.core.sdynpy_coordinate import from_nodelist as _from_nodelist
from sdynpy.fem.sdynpy_beam import beamkm as _beamkm, rect_beam_props as _rect_beam_props
from sdynpy.core.sdynpy_geometry import (
    Geometry as _Geometry,
    node_array as _node_array,
    traceline_array as _traceline_array,
    coordinate_system_array as _coordinate_system_array,
    element_array as _element_array,
)
from sdynpy.core.sdynpy_system import System as _System


def _build_models():
    # Define in inches, which is how the drawing is built, we will convert
    # to meters later.
    frame_thickness = 0.5  # inches
    frame_length = 7.75 * 2  # inches
    frame_width = 2.75 * 2  # inches
    frame_bays = 4
    elem_per_bay_length = 3
    elem_per_bay_width = 4
    wing_width = 22  # inches
    wing_thickness = 0.25 * 1.3  # inches
    wing_length = 4.385  # inches
    wing_attachment_bay = 2
    wing_attachment_offset = wing_length / 2 - 1.9376
    E = 69e9  # [N/m^2],
    nu = 0.33  # [-],
    rho = 2830  # [kg/m^3]
    IN2M = 0.0254

    # Create the frame
    node_x = np.linspace(-frame_length / 2, frame_length / 2, frame_bays * elem_per_bay_length + 1)
    node_y = np.linspace(-frame_width / 2, frame_width / 2, elem_per_bay_width + 1)

    node_indices = (
        [(i, 0) for i in range(len(node_x))]
        + [(i, len(node_y) - 1) for i in range(len(node_x))]
        + [
            (i * (elem_per_bay_length), j)
            for i in range(len(node_x) // elem_per_bay_length + 1)
            for j in range(len(node_y))
        ]
    )
    node_indices = sorted(list(set(node_indices)))

    node_ids = np.array([(i + 1) * 10 + j + 1 for i, j in node_indices])

    # Here we convert to meters
    frame_node_coordinates = (
        np.array([[node_x[i], node_y[j], 0] for i, j in node_indices]) * IN2M
    )  # Meters

    # Create connectivity arrays
    frame_connectivity = []
    for i in range(len(node_x)):
        for j in range(len(node_y)):
            if not (i, j) in node_indices:
                continue
            idx0 = node_indices.index((i, j))
            if (i + 1, j) in node_indices:
                idx1 = node_indices.index((i + 1, j))
                frame_connectivity.append((idx0, idx1))
            if (i, j + 1) in node_indices:
                idx1 = node_indices.index((i, j + 1))
                frame_connectivity.append((idx0, idx1))
    frame_connectivity = np.array(frame_connectivity)

    # Create the beam models
    frame_bend_direction_1 = np.array(
        (
            np.zeros(len(frame_connectivity)),
            np.zeros(len(frame_connectivity)),
            np.ones(len(frame_connectivity)),
        )
    ).T

    width = frame_thickness * IN2M  # Meters
    height = frame_thickness * IN2M  # Meters
    frame_mat_props = _rect_beam_props(E, rho, nu, width, height, len(frame_connectivity))
    K, M = _beamkm(
        frame_node_coordinates, frame_connectivity, frame_bend_direction_1, **frame_mat_props
    )
    coordinates = _from_nodelist(node_ids, directions=[1, 2, 3, 4, 5, 6])
    frame_system = _System(coordinates, M, K)

    frame_geometry = _Geometry(
        _node_array(node_ids, frame_node_coordinates),
        _coordinate_system_array(1),
        _traceline_array(
            np.arange(frame_connectivity.shape[0]) + 1,
            connectivity=[arr for arr in node_ids[frame_connectivity]],
        ),
    )
    frame_modes = frame_system.eigensolution(num_modes=50)
    frame_geometry.plot_shape(frame_modes)

    # Create the wing

    def structured_tri_mesh(
        width, height, dx, dy=None, origin=(0.0, 0.0), diag="alternating", return_edges=False
    ):
        """
        Structured triangular mesh on an axis-aligned rectangle using a rect grid + diagonal splits.

        Returns
          pts: (N,2) array
          tri: (M,3) array of indices into pts
          edges (optional): (E,2) array of unique undirected edges (i<j)
        """
        if dy is None:
            dy = dx
        x0, y0 = origin

        nx = int(np.ceil(width / dx))
        ny = int(np.ceil(height / dy))

        xs = x0 + np.linspace(0.0, width, nx + 1)
        ys = y0 + np.linspace(0.0, height, ny + 1)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        pts = np.column_stack([X.ravel(), Y.ravel()])

        def vid(i, j):  # i in [0..nx], j in [0..ny]
            return j * (nx + 1) + i

        tris = []
        edges = []
        for j in range(ny):
            for i in range(nx):
                v00 = vid(i, j)
                v10 = vid(i + 1, j)
                v01 = vid(i, j + 1)
                v11 = vid(i + 1, j + 1)

                edges.append([v00, v10])
                edges.append([v00, v11])
                edges.append([v00, v01])
                edges.append([v10, v11])
                edges.append([v10, v01])
                edges.append([v01, v11])

                if diag == "same" or diag == "x":
                    tris.append([v00, v10, v11])
                    tris.append([v00, v11, v01])
                elif diag == "y":
                    tris.append([v00, v10, v01])
                    tris.append([v10, v11, v01])
                elif diag == "alternating":
                    if (i + j) % 2 == 0:
                        tris.append([v00, v10, v11])
                        tris.append([v00, v11, v01])
                    else:
                        tris.append([v00, v10, v01])
                        tris.append([v10, v11, v01])
                else:
                    raise ValueError("diag must be one of: same, alternating, x, y")

        tri = np.asarray(tris, dtype=int)

        if not return_edges:
            return pts, tri

        # Build all edges from triangles, canonicalize (i<j), then unique
        all_edges = np.array(edges)
        all_edges = np.sort(all_edges, axis=1)  # undirected canonical form
        edges = np.unique(all_edges, axis=0)  # unique rows

        return pts, tri, edges

    wing_coordinates, wing_tris, wing_edges = structured_tri_mesh(
        wing_length * IN2M,
        wing_width * IN2M,
        dy=frame_width / elem_per_bay_width * IN2M,
        dx=frame_length / frame_bays / elem_per_bay_length * IN2M * 1.2,
        origin=np.array(
            (
                -frame_length / 2
                + frame_length / frame_bays * (wing_attachment_bay - 1)
                - wing_attachment_offset,
                -wing_width / 2,
            )
        )
        * IN2M,
        return_edges=True,
    )

    wing_coordinates = np.concatenate(
        (
            wing_coordinates,
            (frame_thickness / 2 + wing_thickness / 2)
            * IN2M
            * np.ones((wing_coordinates.shape[0], 1)),
        ),
        axis=-1,
    )
    wing_ids = np.arange(wing_coordinates.shape[0]) + 1001

    wing_geometry = _Geometry(
        _node_array(wing_ids, wing_coordinates),
        _coordinate_system_array(1),
        # _traceline_array(np.arange(wing_edges.shape[0])+1,
        #                  connectivity = [arr for arr in wing_ids[wing_edges]]),
        element=_element_array(
            np.arange(wing_tris.shape[0]) + 1,
            type=41,
            connectivity=[arr for arr in wing_ids[wing_tris]],
        ),
    )

    # Find the nodes in the second bay to make them coincident
    bay_length = frame_length / frame_bays
    bay_xmin = (
        -frame_length / 2
        + (wing_attachment_bay - 1) * bay_length
        - bay_length / (elem_per_bay_length * 2)
    )
    bay_xmax = (
        -frame_length / 2
        + (wing_attachment_bay) * bay_length
        + bay_length / (elem_per_bay_length * 2)
    )
    # Get nodes in this range
    nodes_in_bay = frame_geometry.node.id[
        (frame_geometry.node.coordinate[:, 0] > bay_xmin * IN2M)
        & (frame_geometry.node.coordinate[:, 0] < bay_xmax * IN2M)
    ]
    attachment_coordinates = frame_geometry.node(nodes_in_bay).coordinate
    equivalent_wing_indices = np.array(
        [
            np.where(
                wing_geometry.node.id
                == wing_geometry.node_by_global_position(attachment_coordinate).id
            )[0][0]
            for attachment_coordinate in attachment_coordinates
        ]
    )
    wing_geometry.node.coordinate[equivalent_wing_indices, :2] = attachment_coordinates[:, :2]

    # Now build complete mass and stiffness matrices
    wing_node_coordinates = wing_geometry.node.coordinate
    wing_connectivity = wing_edges
    wing_bend_direction_1 = np.array(
        (
            np.zeros(len(wing_connectivity)),
            np.zeros(len(wing_connectivity)),
            np.ones(len(wing_connectivity)),
        )
    ).T
    width = wing_thickness * IN2M  # Meters
    height = wing_thickness * IN2M  # Meters
    wing_mat_props = _rect_beam_props(E, rho, nu, width, height, len(wing_connectivity))
    K, M = _beamkm(
        wing_node_coordinates, wing_connectivity, wing_bend_direction_1, **wing_mat_props
    )
    coordinates = _from_nodelist(wing_geometry.node.id, directions=[1, 2, 3, 4, 5, 6])
    wing_system = _System(coordinates, M, K)

    wing_modes = wing_system.eigensolution(num_modes=50)
    wing_geometry.plot_shape(wing_modes)

    # Now let's create a coupled system
    equivalent_frame_indices = np.array(
        [
            np.where(
                frame_geometry.node.id
                == frame_geometry.node_by_global_position(attachment_coordinate).id
            )[0][0]
            for attachment_coordinate in attachment_coordinates
        ]
    )

    connection_connectivity = []
    for frame_index, wing_index in zip(equivalent_frame_indices, equivalent_wing_indices):
        connection_connectivity.append([frame_index, wing_index + frame_node_coordinates.shape[0]])

    connection_connectivity = np.array(connection_connectivity)
    connection_bend_direction_1 = np.array(
        (
            np.ones(len(connection_connectivity)),
            np.zeros(len(connection_connectivity)),
            np.zeros(len(connection_connectivity)),
        )
    ).T
    width = frame_thickness * IN2M  # Meters
    height = frame_thickness * IN2M  # Meters
    connection_mat_props = _rect_beam_props(E, rho, nu, width, height, len(connection_connectivity))

    full_connectivity = np.concatenate(
        (
            frame_connectivity,
            wing_connectivity + frame_node_coordinates.shape[0],
            connection_connectivity,
        )
    )
    full_coordinates = np.concatenate((frame_node_coordinates, wing_node_coordinates))
    full_bend_1_direction = np.concatenate(
        (frame_bend_direction_1, wing_bend_direction_1, connection_bend_direction_1)
    )
    full_mat_props = {}
    for mat_prop in (frame_mat_props, wing_mat_props, connection_mat_props):
        for key, value in mat_prop.items():
            if key not in full_mat_props:
                full_mat_props[key] = []
            full_mat_props[key] = np.concatenate((full_mat_props[key], value))
    K, M = _beamkm(full_coordinates, full_connectivity, full_bend_1_direction, **full_mat_props)
    coordinates = np.concatenate(
        (
            _from_nodelist(frame_geometry.node.id, directions=[1, 2, 3, 4, 5, 6]),
            _from_nodelist(wing_geometry.node.id, directions=[1, 2, 3, 4, 5, 6]),
        )
    )
    full_system = _System(coordinates, M, K)
    full_modes = full_system.eigensolution(num_modes=50)
    full_geometry = frame_geometry + wing_geometry

    full_geometry.plot_shape(full_modes)

    return full_geometry, full_system, frame_geometry, frame_system, wing_geometry, wing_system


geometry, system, frame_geometry, frame_system, wing_geometry, wing_system = _build_models()
