"""Graph-based path planning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import math

import numpy as np
import networkx as nx

from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, Polygon
from shapely.ops import nearest_points

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .image_processing import _poly_to_linestring


@dataclass
class RoutePlan:
    segments: List[Tuple[bool, LineString]]
    component_paths: List[List[LineString]]


def plan_route_greedy(polylines, tolerance: float = 0.1) -> RoutePlan:
    """
    Plans a route by identifying "master" polylines (outermost) and then drawing
    their contents in an inside-out order based on a containment hierarchy.
    """
    if not polylines:
        return RoutePlan([], [])

    lines = [_poly_to_linestring(p) for p in polylines]
    lines = [ls for ls in lines if ls and not ls.is_empty and ls.length > 1e-4]

    if not lines:
        return RoutePlan([], [])

    # 1. Build a containment graph
    bounds = [line.bounds for line in lines]
    G = nx.DiGraph()
    for i in range(len(lines)):
        G.add_node(i)

    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue
            
            b_i = bounds[i]
            b_j = bounds[j]
            is_contained = (b_i[0] <= b_j[0] and b_i[1] <= b_j[1] and
                            b_i[2] >= b_j[2] and b_i[3] >= b_j[3])
            is_truly_larger = (b_i[2] - b_i[0] > b_j[2] - b_j[0] or
                               b_i[3] - b_i[1] > b_j[3] - b_j[1])

            if is_contained and is_truly_larger:
                # If i contains j, draw an edge i -> j
                G.add_edge(i, j)
    
    # 2. Identify master polylines (nodes with no incoming edges)
    masters = [i for i, in_degree in G.in_degree() if in_degree == 0]
    
    # 3. Plan the order of drawing masters (e.g., by proximity)
    master_order = []
    remaining_masters = set(masters)
    if not remaining_masters: # Handle cases where everything is nested (no single master)
        # Fallback: use topological sort of the whole graph
        try:
            drawing_order = list(reversed(list(nx.topological_sort(G))))
        except nx.NetworkXUnfeasible: # Cycle detected
            drawing_order = list(range(len(lines))) # Fallback to default order
        masters, master_order, remaining_masters = [], [], [] # skip master logic
    else:
        current_pos = Point(0,0)
        while remaining_masters:
            next_master = min(remaining_masters, key=lambda m: lines[m].distance(current_pos))
            master_order.append(next_master)
            remaining_masters.remove(next_master)
            current_pos = Point(lines[next_master].centroid.coords[0])

    # 4. Generate the final drawing order by processing each master's cluster inside-out
    drawing_order = []
    processed_nodes = set()
    for master_idx in master_order:
        if master_idx in processed_nodes:
            continue
        
        # Get all children in this master's cluster
        cluster_nodes = [master_idx] + list(nx.descendants(G, master_idx))
        
        # Create a subgraph for the cluster
        subgraph = G.subgraph(cluster_nodes)
        
        # Sort nodes inside the cluster from most nested to least (inside-out)
        try:
            cluster_order = list(reversed(list(nx.topological_sort(subgraph))))
        except nx.NetworkXUnfeasible:
            # If cycle, just use a default order for this cluster
            cluster_order = list(subgraph.nodes())

        for node in cluster_order:
            if node not in processed_nodes:
                drawing_order.append(node)
                processed_nodes.add(node)
                
    # Add any remaining nodes that weren't part of a master's cluster
    for i in range(len(lines)):
        if i not in processed_nodes:
            drawing_order.append(i)

    # 5. Build the final segments using the smart greedy execution
    segments: List[Tuple[bool, LineString]] = []
    component_paths: List[List[LineString]] = [[]]
    last_point: Optional[Point] = None

    for line_idx in drawing_order:
        target_line = lines[line_idx]
        
        start_point = Point(target_line.coords[0])
        end_point = Point(target_line.coords[-1])

        if last_point is None:
            connection_point = start_point
            line_to_draw = target_line
        else:
            dist_to_start = last_point.distance(start_point)
            dist_to_end = last_point.distance(end_point)

            if dist_to_start < dist_to_end:
                connection_point = start_point
                line_to_draw = target_line
            else:
                connection_point = end_point
                line_to_draw = LineString(list(target_line.coords)[::-1])

            if last_point.distance(connection_point) > 1e-6:
                segments.append((False, LineString([last_point, connection_point])))

        segments.append((True, line_to_draw))
        last_point = Point(line_to_draw.coords[-1])
        component_paths[-1].append(line_to_draw)

    return RoutePlan(segments=segments, component_paths=component_paths)


def build_graph(
    polylines: Sequence[Sequence[Tuple[float, float]]],
    tolerance: float = 0.1,
) -> nx.MultiGraph:
    G = nx.MultiGraph()
    node_map: Dict[Tuple[float, float], int] = {}
    node_counter = 0

    def get_or_create_node(point: Tuple[float, float]) -> int:
        nonlocal node_counter
        for existing_pt, node_id in node_map.items():
            if math.hypot(point[0] - existing_pt[0], point[1] - existing_pt[1]) < tolerance:
                return node_id
        node_id = node_counter
        node_map[point] = node_id
        G.add_node(node_id, pos=point)
        node_counter += 1
        return node_id

    # Track edges we've already added (by geometry) to prevent duplicates
    added_edges: List[LineString] = []

    for poly in polylines:
        if len(poly) < 2:
            continue
        ls = _poly_to_linestring(poly)
        if ls is None or ls.is_empty:
            continue

        # Skip zero-length or near-zero edges
        if ls.length < max(1e-6, 0.001):
            continue

        # Avoid adding duplicates extremely close to existing edges
        # Check both distance and reversed distance (for opposite direction edges)
        is_dup = False
        for existing in added_edges:
            try:
                # Check distance between lines
                if ls.distance(existing) < tolerance * 0.5:
                    is_dup = True
                    break
                # Also check if they're the same line but reversed
                # by checking if endpoints match (either way)
                ls_start, ls_end = ls.coords[0], ls.coords[-1]
                ex_start, ex_end = existing.coords[0], existing.coords[-1]
                
                # Check if same direction or reversed direction
                same_dir = (math.hypot(ls_start[0]-ex_start[0], ls_start[1]-ex_start[1]) < tolerance and
                           math.hypot(ls_end[0]-ex_end[0], ls_end[1]-ex_end[1]) < tolerance)
                reverse_dir = (math.hypot(ls_start[0]-ex_end[0], ls_start[1]-ex_end[1]) < tolerance and
                              math.hypot(ls_end[0]-ex_start[0], ls_end[1]-ex_start[1]) < tolerance)
                
                if same_dir or reverse_dir:
                    # Check if the lines are actually overlapping significantly
                    overlap_length = 0.0
                    try:
                        overlap = ls.intersection(existing)
                        if hasattr(overlap, 'length'):
                            overlap_length = overlap.length
                    except Exception:
                        pass
                    
                    # If overlap is significant (>50% of line length), it's a duplicate
                    if overlap_length > 0.5 * min(ls.length, existing.length):
                        is_dup = True
                        break
                        
            except Exception:
                continue
        if is_dup:
            continue

        start_node = get_or_create_node(poly[0])
        end_node = get_or_create_node(poly[-1])
        G.add_edge(start_node, end_node, weight=ls.length, geometry=ls, pen_down=True)
        added_edges.append(ls)

    return G


def _make_eulerian(subgraph: nx.MultiGraph):
    """Augment subgraph to be Eulerian using min-weight perfect matching of odd nodes."""
    odd_nodes = [n for n, d in subgraph.degree() if d % 2 != 0]
    if not odd_nodes:
        return

    # Build complete graph of odd nodes with shortest-path distances as weights
    K = nx.Graph()
    for i in range(len(odd_nodes)):
        for j in range(i + 1, len(odd_nodes)):
            u, v = odd_nodes[i], odd_nodes[j]
            try:
                dist = nx.shortest_path_length(subgraph, source=u, target=v, weight="weight")
            except Exception:
                continue
            K.add_edge(u, v, weight=dist)

    matching = nx.algorithms.matching.min_weight_matching(K)
    for u, v in matching:
        path = nx.shortest_path(subgraph, source=u, target=v, weight="weight")
        # Duplicate each edge along the shortest path
        for j in range(len(path) - 1):
            n1, n2 = path[j], path[j + 1]
            # Copy attributes from an existing edge between n1 and n2
            data_dict = subgraph.get_edge_data(n1, n2)
            if not data_dict:
                continue
            # Pick any one parallel edge as template
            template = next(iter(data_dict.values()))
            new_data = template.copy()
            new_data['pen_down'] = False  # Mark duplicated path as pen-up travel
            subgraph.add_edge(n1, n2, **new_data)


def _get_component_geometry(component: nx.MultiGraph) -> Polygon:
    """Computes the Convex Hull of a component's nodes."""
    points = [d["pos"] for _, d in component.nodes(data=True)]
    if not points:
        return Polygon()
    # MultiPoint.convex_hull returns a Polygon (or LineString/Point if degenerate)
    return MultiPoint(points).convex_hull


def _sort_components_cdg(components: List[nx.MultiGraph]) -> List[nx.MultiGraph]:
    """
    Sorts graph components based on topological containment (inside-out).
    This implements the Component Dependency Graph (CDG) algorithm.
    """
    if not components:
        return []

    # 1. Compute convex hulls for all components
    # Store as (index, geometry) tuples
    geoms = [_get_component_geometry(c) for c in components]

    # 2. Build dependency graph
    # Edge A -> B means A must be drawn BEFORE B.
    # Logic: If component B contains component A, we must draw A first.
    # So B depends on A. Edge A -> B.
    # This means in a topological sort, A comes before B.
    D = nx.DiGraph()
    D.add_nodes_from(range(len(components)))

    for i in range(len(components)):
        poly_i = geoms[i]
        if poly_i.is_empty:
            continue

        for j in range(len(components)):
            if i == j:
                continue
            
            poly_j = geoms[j]
            if poly_j.is_empty:
                continue

            # Optimization: Disjoint check
            if not poly_i.intersects(poly_j):
                continue

            # Check containment
            # If i contains j, then j is INSIDE i.
            # We want to draw j (inner) first, then i (outer).
            # So j -> i.
            try:
                # Use a small buffer for robustness against touching boundaries
                if poly_i.contains(poly_j) or poly_i.buffer(1e-6).contains(poly_j):
                    D.add_edge(j, i)
            except Exception:
                continue

    # 3. Topological Sort
    try:
        # Standard topological sort gives a valid ordering respecting dependencies
        order_indices = list(nx.topological_sort(D))
    except nx.NetworkXUnfeasible:
        # Fallback for cycles: Sort by Y-coordinate (original logic)
        return sorted(
            components,
            key=lambda g: max(d["pos"][1] for _, d in g.nodes(data=True)) if g.nodes else 0,
            reverse=True,
        )

    # Re-order the components list based on the sort
    return [components[i] for i in order_indices]


def plan_route_graph(polylines, hierarchy=None, tolerance: float = 0.1) -> RoutePlan:
    G = build_graph(polylines, tolerance)
    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    all_segments: List[Tuple[bool, LineString]] = []
    component_paths_for_plan: List[List[LineString]] = []

    # --- NEW: CDG Sorting Logic ---
    # Replaces the simple Y-axis sort with topological containment sort
    components = _sort_components_cdg(components)
    # ------------------------------

    last_pos = None

    for subG in components:
        odd_nodes = [n for n, d in subG.degree() if d % 2 != 0]
        path_edges = []

        if not subG.nodes():
            continue

        if len(odd_nodes) == 0:
            # Standard Eulerian circuit
            _make_eulerian(subG) # ensure connected even if no odd nodes
            start_node = next(iter(subG.nodes()))
            path_edges = list(nx.eulerian_circuit(subG, source=start_node, keys=True))
        elif len(odd_nodes) == 2:
            # Eulerian path exists, no need to duplicate edges
            start_node = odd_nodes[0]
            path_edges = list(nx.eulerian_path(subG, source=start_node, keys=True))
        else: # More than 2 odd nodes, requires edge duplication
            _make_eulerian(subG)
            start_node = odd_nodes[0] if odd_nodes else next(iter(subG.nodes()))
            path_edges = list(nx.eulerian_circuit(subG, source=start_node, keys=True))

        if not path_edges:
            continue
        
        component_start_pos = Point(subG.nodes[path_edges[0][0]]["pos"])
        path_segments_for_drawing: List[LineString] = []

        # Add pen-up travel from the end of the last component
        if last_pos and last_pos.distance(component_start_pos) > 1e-6:
            all_segments.append((False, LineString([last_pos, component_start_pos])))

        for u, v, k in path_edges:
            data = subG.get_edge_data(u, v, k)
            geom = data.get("geometry")
            if not geom or geom.is_empty:
                continue

            pen_down = data.get("pen_down", True)
            all_segments.append((pen_down, geom))
            if pen_down:
                path_segments_for_drawing.append(geom)

        if path_edges:
            last_node = path_edges[-1][1]
            last_pos = Point(subG.nodes[last_node]["pos"])
        
        if path_segments_for_drawing:
            component_paths_for_plan.append(path_segments_for_drawing)

    return RoutePlan(segments=all_segments, component_paths=component_paths_for_plan)


def visualize_polylines(polylines: Sequence[Sequence[Tuple[float, float]]], title: str = "Polylines"):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.3)

    segments = []
    for poly in polylines:
        if len(poly) >= 2:
            for i in range(len(poly) - 1):
                segments.append([poly[i], poly[i + 1]])

    if segments:
        coll = LineCollection(segments, colors="blue", linewidths=1.5)
        ax.add_collection(coll)
    ax.autoscale()
    plt.show()


def visualize_route(segments: Sequence[Tuple[bool, LineString]], title: str = "Planned Route"):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.3)

    draw_segments = []
    travel_segments = []
    for pen_down, ls in segments:
        coords = list(ls.coords)
        seg_array = [[coords[i], coords[i + 1]] for i in range(len(coords) - 1)]
        if pen_down:
            draw_segments.extend(seg_array)
        else:
            travel_segments.extend(seg_array)

    if travel_segments:
        travel_coll = LineCollection(travel_segments, colors="gray", linewidths=1.0, linestyles="dashed", label="Pen Up")
        ax.add_collection(travel_coll)
    if draw_segments:
        draw_coll = LineCollection(draw_segments, colors="red", linewidths=2.0, label="Pen Down")
        ax.add_collection(draw_coll)

    ax.legend(loc="upper right")
    ax.autoscale()
    plt.show()


__all__ = [
    "RoutePlan",
    "plan_route_graph",
    "visualize_polylines",
    "visualize_route",
    "build_graph",
]
