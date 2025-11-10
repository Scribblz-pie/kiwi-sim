"""Graph-based path planner with stage-by-stage visualization options.

Usage examples:
    python path_sim_graph_eulerian.py image.png --view image       # show image→mask→skeleton
    python path_sim_graph_eulerian.py image.png --view polylines   # show extracted polylines
    python path_sim_graph_eulerian.py image.png --view route       # show Eulerian route only
    python path_sim_graph_eulerian.py image.png --view full        # run full simulation (default)

The heavy lifting for image processing and graph planning now lives in
``path_pipeline.image_processing`` and ``path_pipeline.graph_planner`` so that
each stage can be inspected independently of the full simulator.
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.ops import unary_union, linemerge

from tqdm import tqdm

from path_pipeline.image_processing import (
    DEDUPLICATION_DISTANCE_TOLERANCE_DEFAULT,
    ImageStages,
    generate_image_stages,
    visualize_image_stages,
)
from path_pipeline.graph_planner import (
    RoutePlan,
    plan_route_graph,
    visualize_polylines,
    visualize_route,
    plan_route_greedy,
)


# =========================
# Parameters
# =========================
ROBOT_SIDE_LENGTH = 4.0
ANIMATION_FPS = 30
ANIMATION_INTERVAL_MS = 1000 // ANIMATION_FPS
STEP_SIZE = 0.75
ERASE_MARGIN = 0.0
TARGET_CANVAS_WIDTH = 50.0
CANVAS_PADDING = 2.0
PATH_SMOOTHING_FACTOR = 0.0
PATH_SMOOTHING_POINTS = 200
PATH_SIMPLIFICATION_EPSILON_FACTOR = 0.0
ROBOT_SPEED = 10.0
ROBOT_TURN_SPEED_DEG_PER_SEC = 180.0
ORIENTATION_NUM_CANDIDATES = 48
ORIENTATION_CONTACT_LENGTH_TOL = 0.01 * ROBOT_SIDE_LENGTH
ORIENTATION_AREA_TOL = 1e-8
ORIENTATION_AREA_WEIGHT = 1000.0


ACTIVE_ANIMATIONS: List[animation.FuncAnimation] = []


# =========================
# Geometry helpers
# =========================
def get_robot_body(pen_x: float, pen_y: float, theta: float, side_length: float) -> Polygon:
    sqrt3_over_2 = np.sqrt(3) / 2.0
    half_side = side_length / 2.0

    v2_local_x, v2_local_y = side_length, 0.0
    v3_local_x, v3_local_y = half_side, side_length * sqrt3_over_2

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    v2_rot_x = v2_local_x * cos_t - v2_local_y * sin_t
    v2_rot_y = v2_local_x * sin_t + v2_local_y * cos_t
    v3_rot_x = v3_local_x * cos_t - v3_local_y * sin_t
    v3_rot_y = v3_local_x * sin_t + v3_local_y * cos_t

    poly = Polygon(
        [
            (pen_x, pen_y),
            (pen_x + v2_rot_x, pen_y + v2_rot_y),
            (pen_x + v3_rot_x, pen_y + v3_rot_y),
        ]
    )

    if ERASE_MARGIN > 0:
        poly = poly.buffer(ERASE_MARGIN, cap_style=2, join_style=2)
    return poly


def segment_length(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])


def line_length(geom):
    if geom is None or geom.is_empty:
        return 0.0
    if isinstance(geom, LineString):
        return geom.length
    if isinstance(geom, MultiLineString):
        return sum(g.length for g in geom.geoms)
    try:
        return geom.length
    except Exception:
        return 0.0


def lerp_point(p, q, t):
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def normalize_angle(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def shortest_angle_diff(current: float, target: float) -> float:
    return normalize_angle(target - current)


def geometry_to_segments(geom):
    segments = []
    if geom is None or geom.is_empty:
        return segments
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        for i in range(len(coords) - 1):
            segments.append([coords[i], coords[i + 1]])
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                segments.append([coords[i], coords[i + 1]])
    return segments


# =========================
# Simulation
# =========================
def find_safe_orientation(x, y, ink, last_theta, num_angle_steps=ORIENTATION_NUM_CANDIDATES):
    if ink is None:
        return last_theta, 0.0

    best_theta, min_risk = last_theta, float("inf")
    candidate_thetas = np.linspace(0, 2 * np.pi, num_angle_steps, endpoint=False)
    safe_thetas, risky_options = [], []

    # Check intermediate positions during turn (8 steps per turn)
    num_turn_checks = 8

    for theta in candidate_thetas:
        # Calculate the turn path from last_theta to theta
        delta = shortest_angle_diff(last_theta, theta)
        total_turn_risk = 0.0
        final_position_risk = 0.0
        is_safe = True

        # Check intermediate angles during the turn
        for i in range(num_turn_checks + 1):
            t = i / num_turn_checks
            intermediate_theta = normalize_angle(last_theta + t * delta)
            poly = get_robot_body(x, y, intermediate_theta, ROBOT_SIDE_LENGTH)
            
            try:
                overlap = ink.intersection(poly)
            except Exception:
                overlap = None

            if overlap is None or overlap.is_empty:
                continue

            overlap_area = getattr(overlap, "area", 0.0)
            overlap_length = line_length(overlap)
            risk_metric = overlap_area * ORIENTATION_AREA_WEIGHT + overlap_length

            # Accumulate risk across the turn
            total_turn_risk += risk_metric
            
            # Track final position risk separately
            if i == num_turn_checks:
                final_position_risk = risk_metric

            if overlap_area > ORIENTATION_AREA_TOL or overlap_length > ORIENTATION_CONTACT_LENGTH_TOL:
                is_safe = False

        if is_safe:
            safe_thetas.append(theta)
        else:
            # Combined risk: total turn risk + final position risk (weighted more)
            combined_risk = total_turn_risk + final_position_risk * 2.0
            risky_options.append((combined_risk, theta))

    if safe_thetas:
        deltas = np.array([abs(shortest_angle_diff(last_theta, t)) for t in safe_thetas])
        return safe_thetas[np.argmin(deltas)], 0.0

    if risky_options:
        min_risk, best_theta = min(risky_options, key=lambda r: r[0])
        return best_theta, min_risk

    return last_theta, 0.0


def simulate_segments(segments, step_size):
    states: List[dict] = []
    ink_visible = None
    ink_planner = None
    erased_length_total, drawn_total, travel_up_total, turn_total_rad = 0.0, 0.0, 0.0, 0.0
    last_theta = 0.0
    turn_speed_rad = np.deg2rad(ROBOT_TURN_SPEED_DEG_PER_SEC)

    iterable = tqdm(segments, desc="Simulating segments")
    for (pen_down, ls) in iterable:
        coords = list(ls.coords)
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            seg_len = segment_length(p1, p2)
            n_steps = max(1, int(math.ceil(seg_len / step_size)))
            prev_back = p1

            for s in range(1, n_steps + 1):
                t = s / n_steps
                back = lerp_point(p1, p2, t)
                orientation_ink = ink_planner if ink_planner is not None else ink_visible
                target_theta, _ = find_safe_orientation(back[0], back[1], orientation_ink, last_theta)

                step_length = segment_length(prev_back, back)
                step_duration = step_length / ROBOT_SPEED if ROBOT_SPEED > 0 else 0.0
                delta_total = shortest_angle_diff(last_theta, target_theta)

                if turn_speed_rad > 0:
                    if step_duration > 0.0:
                        max_delta_per_move = max(turn_speed_rad * step_duration, turn_speed_rad / ANIMATION_FPS)
                    else:
                        max_delta_per_move = turn_speed_rad / ANIMATION_FPS
                else:
                    max_delta_per_move = abs(delta_total)

                remaining_delta = delta_total
                rotation_position = prev_back

                if turn_speed_rad > 0 and abs(remaining_delta) > 1e-6:
                    while abs(remaining_delta) > max_delta_per_move + 1e-6:
                        step_delta = math.copysign(max_delta_per_move, remaining_delta)
                        last_theta = normalize_angle(last_theta + step_delta)
                        turn_total_rad += abs(step_delta)
                        robot_poly = get_robot_body(rotation_position[0], rotation_position[1], last_theta, ROBOT_SIDE_LENGTH)
                        if ink_visible is not None:
                            length_before = line_length(ink_visible)
                            diff = ink_visible.difference(robot_poly)
                            ink_visible = diff if diff is not None and not diff.is_empty else None
                            length_after = line_length(ink_visible) if ink_visible is not None else 0.0
                            erased_length_total += (length_before - length_after)
                        states.append({"robot_poly": robot_poly, "ink": ink_visible})
                        remaining_delta = shortest_angle_diff(last_theta, target_theta)

                if turn_speed_rad > 0:
                    applied_delta = remaining_delta
                    if abs(applied_delta) > 1e-6:
                        applied_delta = math.copysign(min(abs(applied_delta), max_delta_per_move), applied_delta)
                        last_theta = normalize_angle(last_theta + applied_delta)
                        turn_total_rad += abs(applied_delta)
                else:
                    if abs(remaining_delta) > 1e-6:
                        last_theta = normalize_angle(last_theta + remaining_delta)
                        turn_total_rad += abs(remaining_delta)

                robot_poly = get_robot_body(back[0], back[1], last_theta, ROBOT_SIDE_LENGTH)
                if ink_visible is not None:
                    length_before = line_length(ink_visible)
                    diff = ink_visible.difference(robot_poly)
                    ink_visible = diff if diff is not None and not diff.is_empty else None
                    length_after = line_length(ink_visible) if ink_visible is not None else 0.0
                    erased_length_total += (length_before - length_after)

                if pen_down:
                    tiny = LineString([prev_back, back])
                    draw_len = tiny.length
                    drawn_total += draw_len
                    if ink_visible is not None:
                        ink_visible = ink_visible.difference(tiny)
                    ink_visible = tiny if ink_visible is None else unary_union([ink_visible, tiny])
                    ink_planner = tiny if ink_planner is None else unary_union([ink_planner, tiny])
                else:
                    travel_up_total += segment_length(prev_back, back)

                states.append({"robot_poly": robot_poly, "ink": ink_visible})
                prev_back = back

    turn_duration_sec = (
        turn_total_rad / np.deg2rad(ROBOT_TURN_SPEED_DEG_PER_SEC) if ROBOT_TURN_SPEED_DEG_PER_SEC > 0 else 0
    )
    linear_duration_sec = (drawn_total + travel_up_total) / ROBOT_SPEED if ROBOT_SPEED > 0 else float("inf")

    total_duration_sec = max(linear_duration_sec, turn_duration_sec)

    metrics = {
        "drawn_length": drawn_total,
        "penup_travel_length": travel_up_total,
        "erased_length": erased_length_total,
        "total_duration_sec": total_duration_sec,
    }
    return states, metrics


def animate_sim(target_geom_for_ref, sim_states, metrics, title_note="", canvas_dims=None):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.7)

    if target_geom_for_ref is not None and not target_geom_for_ref.is_empty:
        ref_segments = geometry_to_segments(target_geom_for_ref)
        ref_coll = LineCollection(ref_segments, colors="gray", linewidths=1.0, linestyles="dashed", label="Target")
        ax.add_collection(ref_coll)

    robot_patch = ax.fill([], [], alpha=0.7, fc="green", ec="black", label=f"Robot (Side:{ROBOT_SIDE_LENGTH})")[0]
    ink_collection = LineCollection([], colors="red", linewidths=2.5, label="Visible Ink")
    ax.add_collection(ink_collection)

    if canvas_dims:
        c_w, c_h = canvas_dims
        ax.set_xlim(-CANVAS_PADDING, c_w + CANVAS_PADDING)
        ax.set_ylim(-CANVAS_PADDING, c_h + CANVAS_PADDING)
    else:
        if target_geom_for_ref is not None and not target_geom_for_ref.is_empty:
            minx, miny, maxx, maxy = target_geom_for_ref.bounds
        else:
            minx = miny = 1e9
            maxx = maxy = -1e9
            for st in sim_states[:: max(1, len(sim_states) // 50)]:
                geom = st["ink"]
                if geom is not None and not geom.is_empty:
                    bx = geom.bounds
                    minx, miny = min(minx, bx[0]), min(miny, bx[1])
                    maxx, maxy = max(maxx, bx[2]), max(maxy, bx[3])
            if minx > maxx:
                minx, miny, maxx, maxy = -10, -10, 10, 10
        pad = max((maxx - minx), (maxy - miny)) * 0.1 + 1.0
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)

    legend = ax.legend(loc="upper right")

    def update(frame_idx):
        st = sim_states[frame_idx]
        poly = st["robot_poly"]
        x_poly, y_poly = poly.exterior.xy
        robot_patch.set_xy(list(zip(x_poly, y_poly)))
        ink_geom = st["ink"]
        segments = geometry_to_segments(ink_geom)
        ink_collection.set_segments(segments)
        ax.set_title(
            f"Robot Path (Frame {frame_idx+1}/{len(sim_states)}) {title_note}\n"
            f"Drawn: {metrics['drawn_length']:.1f} | Travel(up): {metrics['penup_travel_length']:.1f} | "
            f"Erased: {metrics.get('erased_length', 0.0):.1f} | Duration: {metrics['total_duration_sec']:.1f}s"
        )
        return robot_patch, ink_collection, legend

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(sim_states),
        interval=ANIMATION_INTERVAL_MS,
        blit=True,
        repeat=False,
    )
    ACTIVE_ANIMATIONS.append(ani)
    plt.show()
    return ani


# =========================
# Pipeline helpers
# =========================
def build_image_stages(image_path: str, dedup_tol: float, extractor: str, approx_tol: float) -> ImageStages:
    return generate_image_stages(
        image_path=image_path,
        target_width=TARGET_CANVAS_WIDTH,
        padding=CANVAS_PADDING,
        smooth_factor=PATH_SMOOTHING_FACTOR,
        smoothing_points=PATH_SMOOTHING_POINTS,
        simplification_epsilon_factor=PATH_SIMPLIFICATION_EPSILON_FACTOR,
        dedup_tolerance=dedup_tol,
        extractor=extractor,
        approx_tol=approx_tol,
    )


def run_stage(args, image_path: str):
    stages = build_image_stages(image_path, args.dedup_tol, args.extractor, args.approx_tol)

    if args.view == "image":
        visualize_image_stages(stages)
        return

    if args.view == "polylines":
        visualize_polylines(stages.rescaled_polylines, title="Rescaled Polylines")
        return

    route_plan = plan_route_greedy(stages.rescaled_polylines)

    if args.view == "route":
        visualize_route(route_plan.segments, title="Greedy Route")
        return

    if args.view == "components":
        for idx, component in enumerate(route_plan.component_paths, start=1):
            component_segments = [(True, seg) for seg in component]
            visualize_route(component_segments, title=f"Component {idx}")
        return

    # Full simulation flow
    target_geom = None
    for pen_down, ls in route_plan.segments:
        if pen_down:
            target_geom = ls if target_geom is None else unary_union([target_geom, ls])

    print("Simulating route...")
    sim_states, metrics = simulate_segments(route_plan.segments, STEP_SIZE)

    print("Simulation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")

    if not args.no_animate:
        print("Starting animation...")
        animate_sim(
            target_geom,
            sim_states,
            metrics,
            title_note="(Graph Eulerian)",
            canvas_dims=(TARGET_CANVAS_WIDTH, stages.canvas_height),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Graph-based path planner with stage visualization")
    parser.add_argument("image_path", nargs="?", help="Path to black-on-white line art image")
    parser.add_argument(
        "--view",
        choices=["image", "polylines", "route", "components", "full"],
        default="full",
        help="Which stage to visualize",
    )
    parser.add_argument("--no-animate", action="store_true", help="Skip animation even in full mode")
    parser.add_argument(
        "--dedup-tol",
        type=float,
        default=DEDUPLICATION_DISTANCE_TOLERANCE_DEFAULT,
        help="Tolerance (in canvas units) for merging duplicate polylines",
    )
    parser.add_argument(
        "--extractor",
        choices=["cv2", "skimage"],
        default="cv2",
        help="Polyline extractor: cv2 on skeleton (default) or skimage find_contours",
    )
    parser.add_argument(
        "--approx-tol",
        type=float,
        default=0.0,
        help="Polygon approximation tolerance for skimage extractor (0 disables)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = args.image_path
    if not image_path:
        image_path = input("Image path: ").strip()
    if not image_path:
        print("No image provided.")
        return

    run_stage(args, image_path)


if __name__ == "__main__":
    print("=" * 60)
    print("GRAPH-BASED EULERIAN PATH PLANNER")
    print("=" * 60)
    print(f"Robot: Triangle, Side={ROBOT_SIDE_LENGTH}, Pen at vertex")
    print(f"Canvas: {TARGET_CANVAS_WIDTH} units wide")
    print("-" * 60)
    main()

