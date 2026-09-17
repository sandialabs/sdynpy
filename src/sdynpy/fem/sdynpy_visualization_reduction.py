"""
Functions to create visualization artifacts (e.g. tracelines) automatically such that a
reduced geometry consisting of only a point cloud of nodes can be visualized.

This module defines a Geometry object as well as all of the subcomponents of
a geometry object: nodes, elements, tracelines and coordinate system.  Geometry
plotting is also handled in this module.
"""

"""
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
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from ..core.sdynpy_geometry import Geometry, TracelineArray, traceline_array


def build_node_id_to_index(node_ids):
    node_ids = np.asarray(node_ids)
    return {nid: i for i, nid in enumerate(node_ids)}


def derive_node_block_sets(node_ids, elem_conn, elem_block_ids):
    """
    Derive, for each node index, the set of element blocks touching that node.

    Parameters
    ----------
    node_ids : (n,) array of node IDs
    elem_conn : sequence length m
        Each entry is an iterable of node IDs for one element.
    elem_block_ids : (m,) array-like
        Block ID for each element.

    Returns
    -------
    node_block_sets : list of sets, length n
        node_block_sets[i] is the set of block IDs incident on node i.
    """
    id_to_index = build_node_id_to_index(node_ids)
    n = len(node_ids)

    node_block_sets = [set() for _ in range(n)]

    for conn, blk in zip(elem_conn, elem_block_ids):
        for nid in conn:
            if nid in id_to_index:
                i = id_to_index[nid]
                node_block_sets[i].add(blk)

    return node_block_sets


def derive_node_adjacency_from_elements(node_ids, elem_conn):
    """
    Build a node adjacency graph from element connectivity.

    Here, any two nodes appearing in the same element are marked adjacent.
    This is a coarse notion of adjacency, but it is easy to compute and
    works even for mixed element sizes.

    Parameters
    ----------
    node_ids : (n,) array of node IDs
    elem_conn : sequence length m
        Each entry is an iterable of node IDs for one element.

    Returns
    -------
    adjacency : list of sets, length n
        adjacency[i] contains node indices adjacent to node i.
    """
    id_to_index = build_node_id_to_index(node_ids)
    n = len(node_ids)
    adjacency = [set() for _ in range(n)]

    for conn in elem_conn:
        idxs = [id_to_index[nid] for nid in conn if nid in id_to_index]

        # connect all pairs within the element
        # for visualization purposes this is acceptable as a coarse adjacency
        for a_pos in range(len(idxs)):
            i = idxs[a_pos]
            for b_pos in range(a_pos + 1, len(idxs)):
                j = idxs[b_pos]
                adjacency[i].add(j)
                adjacency[j].add(i)

    return adjacency


def block_overlap_penalty(block_set_i, block_set_j, penalty):
    """
    If block memberships overlap, no penalty.
    Otherwise apply multiplicative penalty.
    """
    if block_set_i & block_set_j:
        return 1.0
    return 1.0 + penalty


def build_visualization_graph_from_fe_data(
    node_ids,
    node_coords,
    elem_conn,
    elem_block_ids,
    retained_node_ids=None,
    k_candidates=12,
    k_extra=2,
    block_penalty=0.5,
    max_length_factor=3.0,
    use_node_adjacency=False,
):
    """
    Build a sparse visualization graph for retained FE nodes.

    Parameters
    ----------
    node_ids : (n,) array-like
        All node IDs.
    node_coords : (n,3) array-like
        Coordinates for all nodes.
    elem_conn : sequence length m
        Each element is an iterable of node IDs.
    elem_block_ids : (m,) array-like
        Block ID per element.
    retained_node_ids : array-like, optional
        Subset of node IDs to include. If None, all nodes are retained.
    k_candidates : int
        Number of Euclidean nearest neighbors to consider.
    k_extra : int
        Number of extra local edges per node beyond the MST.
    block_penalty : float
        Penalty applied when nodes share no block membership.
    max_length_factor : float
        Local edges longer than this times median nearest-neighbor distance
        are pruned. MST edges are always kept.
    use_node_adjacency : bool
        If True, require candidate edges to be between nodes that are adjacent
        in the coarse element-derived node adjacency graph.

    Returns
    -------
    retained_ids : (nr,) ndarray
        Retained node IDs, in returned local ordering.
    retained_coords : (nr,3) ndarray
        Retained node coordinates.
    edges : list of tuple(int, int)
        Undirected edges in retained-node local indexing.
    edge_lengths : ndarray
        Euclidean edge lengths.
    retained_block_sets : list of sets
        Block-membership sets for retained nodes.
    """

    node_ids = np.asarray(node_ids)
    node_coords = np.asarray(node_coords, dtype=float)
    elem_block_ids = np.asarray(elem_block_ids)

    if node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise ValueError("node_coords must have shape (n, 3)")

    if len(node_ids) != len(node_coords):
        raise ValueError("node_ids and node_coords must have same length")

    # Derive node block memberships on full node set
    full_node_block_sets = derive_node_block_sets(node_ids, elem_conn, elem_block_ids)

    # Optional full node adjacency
    full_adjacency = None
    if use_node_adjacency:
        full_adjacency = derive_node_adjacency_from_elements(node_ids, elem_conn)

    # Retained subset
    if retained_node_ids is None:
        retain_mask = np.ones(len(node_ids), dtype=bool)
    else:
        retained_node_ids = set(retained_node_ids)
        retain_mask = np.array([nid in retained_node_ids for nid in node_ids], dtype=bool)

    retained_ids = node_ids[retain_mask]
    retained_coords = node_coords[retain_mask]
    retained_block_sets = [full_node_block_sets[i] for i in np.where(retain_mask)[0]]

    nr = len(retained_ids)
    if nr < 2:
        return retained_ids, retained_coords, [], np.array([]), retained_block_sets

    # Map full-node index -> retained-node local index
    full_to_retained = {}
    retained_full_indices = np.where(retain_mask)[0]
    for local_i, full_i in enumerate(retained_full_indices):
        full_to_retained[full_i] = local_i

    # Build retained adjacency if requested
    retained_adjacency = None
    if use_node_adjacency:
        retained_adjacency = [set() for _ in range(nr)]
        for full_i in retained_full_indices:
            i_local = full_to_retained[full_i]
            for full_j in full_adjacency[full_i]:
                if full_j in full_to_retained:
                    j_local = full_to_retained[full_j]
                    retained_adjacency[i_local].add(j_local)

    # Geometric neighbor search on retained nodes
    tree = cKDTree(retained_coords)
    k_query = min(k_candidates + 1, nr)
    dists, nbrs = tree.query(retained_coords, k=k_query)

    if k_query == 1:
        dists = dists[:, None]
        nbrs = nbrs[:, None]

    nn_dists = []
    for i in range(nr):
        for jj in range(1, k_query):
            j = int(nbrs[i, jj])
            if i == j:
                continue
            if use_node_adjacency and j not in retained_adjacency[i]:
                continue
            nn_dists.append(dists[i, jj])
            break

    if not nn_dists:
        return retained_ids, retained_coords, [], np.array([]), retained_block_sets

    median_nn_dist = np.median(nn_dists)
    max_allowed_length = max_length_factor * median_nn_dist

    # Candidate edges
    edge_data = {}
    # edge_data[(i,j)] = (weighted_dist, euclidean_dist)

    for i in range(nr):
        for jj in range(1, k_query):
            j = int(nbrs[i, jj])
            if i == j:
                continue

            if use_node_adjacency and j not in retained_adjacency[i]:
                continue

            a, b = sorted((i, j))
            euclid = np.linalg.norm(retained_coords[a] - retained_coords[b])

            penalty_factor = block_overlap_penalty(
                retained_block_sets[a], retained_block_sets[b], block_penalty
            )
            weighted = euclid * penalty_factor

            key = (a, b)
            if key not in edge_data or weighted < edge_data[key][0]:
                edge_data[key] = (weighted, euclid)

    if not edge_data:
        return retained_ids, retained_coords, [], np.array([]), retained_block_sets

    # Build weighted sparse graph for MST
    rows, cols, data = [], [], []
    for (i, j), (weighted, _) in edge_data.items():
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([weighted, weighted])

    graph = coo_matrix((data, (rows, cols)), shape=(nr, nr)).tocsr()
    mst = minimum_spanning_tree(graph)

    mst_edges = set()
    mst_coo = mst.tocoo()
    for i, j in zip(mst_coo.row, mst_coo.col):
        a, b = sorted((int(i), int(j)))
        mst_edges.add((a, b))

    # Add extra local edges
    extra_edges = set()
    for i in range(nr):
        local_candidates = []

        for jj in range(1, k_query):
            j = int(nbrs[i, jj])
            if i == j:
                continue

            if use_node_adjacency and j not in retained_adjacency[i]:
                continue

            a, b = sorted((i, j))
            euclid = np.linalg.norm(retained_coords[a] - retained_coords[b])

            penalty_factor = block_overlap_penalty(
                retained_block_sets[a], retained_block_sets[b], block_penalty
            )
            weighted = euclid * penalty_factor

            local_candidates.append((weighted, euclid, a, b))

        local_candidates.sort(key=lambda x: x[0])

        added = 0
        for weighted, euclid, a, b in local_candidates:
            if euclid <= max_allowed_length:
                extra_edges.add((a, b))
                added += 1
            if added >= k_extra:
                break

    all_edges = mst_edges | extra_edges

    # Prune long edges, but keep MST
    final_edges = set()
    for a, b in all_edges:
        euclid = np.linalg.norm(retained_coords[a] - retained_coords[b])
        if (a, b) in mst_edges or euclid <= max_allowed_length:
            final_edges.add((a, b))

    final_edges = sorted(final_edges)
    edge_lengths = np.array(
        [np.linalg.norm(retained_coords[i] - retained_coords[j]) for i, j in final_edges]
    )

    return retained_ids, retained_coords, final_edges, edge_lengths, retained_block_sets


def create_visualization_tracelines(
    geometry: Geometry,
    retained_node_ids=None,
    k_candidates=12,
    k_extra=2,
    block_penalty=0.5,
    max_length_factor=3.0,
    use_node_adjacency=False,
):
    """Creates tracelines that can be used to improve visualization of a reduced geometry.

    Parameters
    ----------
    geometry : Geometry
        The geometry containing the original mesh with original elements and colors.
    retained_node_ids : np.ndarray,
        A numpy array consisting of the node IDs that will be retained.
    k_candidates : int, optional
        Number of nearest neighbors to consider, by default 12
    k_extra : int, optional
        Number of extra local edges per node beyond the MST, by default 2
    block_penalty : float, optional
        Penalty applied when nodes share no block membership, by default 0.5
    max_length_factor : float, optional
        Local edges longer than this times median nearest-neighbor distance
        are pruned. MST edges are always kept.  By default 3.0
    use_node_adjacency : bool, optional
        If True, require candidate edges to be between nodes that are adjacent
        in the coarse element-derived node adjacency graph.  By default, False.

    Returns
    -------
    tracelines : TracelineArray
        Returns a TracelineArray containing lines that can be used to visualize reduced geometry.
    """

    node_ids = geometry.node.id
    coords = geometry.global_node_coordinate()
    conn = geometry.element.connectivity
    block_ids = geometry.element.color

    ret_ids, ret_xyz, edges, lengths, block_sets = build_visualization_graph_from_fe_data(
        node_ids=node_ids,
        node_coords=coords,
        elem_conn=conn,
        elem_block_ids=block_ids,
        retained_node_ids=retained_node_ids,
        k_candidates=k_candidates,
        k_extra=k_extra,
        block_penalty=block_penalty,
        max_length_factor=max_length_factor,
        use_node_adjacency=use_node_adjacency,
    )
    connectivity = [ret_ids[np.array(edge)] for edge in edges]
    colors = []
    for edge in edges:
        set_1 = block_sets[edge[0]]
        set_2 = block_sets[edge[1]]
        full_set = set_1 | set_2
        # Pick the first item if there is more than 1
        color, *_ = full_set
        colors.append(color)
    return traceline_array(
        np.arange(len(connectivity)) + 1, color=colors, connectivity=connectivity
    )