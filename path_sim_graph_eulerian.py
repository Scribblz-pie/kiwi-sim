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

from path_pipeline.graph_planner import (
    RoutePlan,
    plan_route_graph,
    visualize_polylines,
    visualize_route,
    plan_route_greedy,
)
from path_pipeline.image_processing import (
    BodyTwist,
    RobotPose,
    WheelCommand,
    compute_kiwi_wheel_speeds,
    save_commands_to_json,
    _world_to_body_velocity,
    DEDUPLICATION_DISTANCE_TOLERANCE_DEFAULT,
    ImageStages,
    generate_image_stages,
    visualize_image_stages,
)
from dataclasses import dataclass


# =========================
# Parameters
# =========================
# --- Physical dimensions (in meters) ---
# To treat all units as meters, set TARGET_CANVAS_WIDTH to the real-world
# width of your drawing area. All other length parameters below should
# also be specified in meters.

ROBOT_SIDE_LENGTH = 0.1732  # Side length of the triangular robot chassis
ROBOT_WHEELBASE_L = ROBOT_SIDE_LENGTH / np.sqrt(3.0)
ROBOT_RADIUS = ROBOT_WHEELBASE_L
PEN_OFFSET_R = (ROBOT_SIDE_LENGTH + .0286) / np.sqrt(3.0)
WHEEL_RADIUS = 0.016  # Radius of the robot's wheels

# --- Canvas and Path Scaling ---
TARGET_CANVAS_WIDTH = .3  # The target width of the drawing area in meters
CANVAS_PADDING = 0.05      # Padding around the drawing on the canvas

# --- Simulation Parameters ---
ANIMATION_FPS = 30
ANIMATION_INTERVAL_MS = 1000 // ANIMATION_FPS
STEP_SIZE = 0.02  # Simulation step size in meters
ERASE_MARGIN = 0.0

# --- Path Processing Parameters ---
PATH_SMOOTHING_FACTOR = 0.0
PATH_SMOOTHING_POINTS = 200
PATH_SIMPLIFICATION_EPSILON_FACTOR = 0.0

# --- Robot Performance Parameters ---
ROBOT_SPEED = 0.245  # m/s
ROBOT_TURN_SPEED_DEG_PER_SEC = 110

# --- Collision Avoidance Parameters ---
ORIENTATION_NUM_CANDIDATES = 48
ORIENTATION_CONTACT_LENGTH_TOL = 0.01 * ROBOT_SIDE_LENGTH
ORIENTATION_AREA_TOL = 1e-8
ORIENTATION_AREA_WEIGHT = 1000.0


ACTIVE_ANIMATIONS: List[animation.FuncAnimation] = []


# =========================
# Geometry helpers
# =========================
def get_robot_body(pen_x: float, pen_y: float, theta: float, side_length: float) -> Polygon:
    """Calculates the robot's triangular polygon, with the pen at a vertex."""
    robot_radius = side_length / np.sqrt(3.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # The robot's center is offset from the pen tip.
    center_x = pen_x - robot_radius * cos_t
    center_y = pen_y - robot_radius * sin_t

    # Define vertices relative to the calculated center
    v1_local_x, v1_local_y = robot_radius, 0.0
    v2_local_x, v2_local_y = robot_radius * np.cos(2 * np.pi / 3), robot_radius * np.sin(2 * np.pi / 3)
    v3_local_x, v3_local_y = robot_radius * np.cos(4 * np.pi / 3), robot_radius * np.sin(4 * np.pi / 3)

    # Rotate vertices
    v1_rot_x = v1_local_x * cos_t - v1_local_y * sin_t
    v1_rot_y = v1_local_x * sin_t + v1_local_y * cos_t
    v2_rot_x = v2_local_x * cos_t - v2_local_y * sin_t
    v2_rot_y = v2_local_x * sin_t + v2_local_y * cos_t
    v3_rot_x = v3_local_x * cos_t - v3_local_y * sin_t
    v3_rot_y = v3_local_x * sin_t + v3_local_y * cos_t

    poly = Polygon(
        [
            (center_x + v1_rot_x, center_y + v1_rot_y),
            (center_x + v2_rot_x, center_y + v2_rot_y),
            (center_x + v3_rot_x, center_y + v3_rot_y),
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
def find_safe_orientation(
    x, y, ink, last_theta, num_angle_steps=ORIENTATION_NUM_CANDIDATES, search_radius=ROBOT_SIDE_LENGTH/4.0
):
    if ink is None:
        return x, y, last_theta, 0.0

    best_pos_x, best_pos_y, best_theta, min_risk = x, y, last_theta, float("inf")
    
    # Search grid for position: 5x5 grid around the current point
    pos_candidates = [(x, y)]
    for r_factor in [0.5, 1.0]:
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            px = x + search_radius * r_factor * np.cos(angle)
            py = y + search_radius * r_factor * np.sin(angle)
            pos_candidates.append((px, py))

    candidate_thetas = np.linspace(0, 2 * np.pi, num_angle_steps, endpoint=False)
    safe_options, risky_options = [], []

    num_turn_checks = 8

    for pos_x, pos_y in pos_candidates:
        for theta in candidate_thetas:
            delta_angle = shortest_angle_diff(last_theta, theta)
            total_turn_risk = 0.0
            final_position_risk = 0.0
            is_safe = True

            for i in range(num_turn_checks + 1):
                t = i / num_turn_checks
                intermediate_theta = normalize_angle(last_theta + t * delta_angle)
                
                # Interpolate position as well for a combined move-and-turn path
                interp_x = x + t * (pos_x - x)
                interp_y = y + t * (pos_y - y)
                
                poly = get_robot_body(interp_x, interp_y, intermediate_theta, ROBOT_SIDE_LENGTH)
                
                try:
                    overlap = ink.intersection(poly)
                except Exception:
                    overlap = None

                if overlap is None or overlap.is_empty:
                    continue

                overlap_area = getattr(overlap, "area", 0.0)
                overlap_length = line_length(overlap)
                risk_metric = overlap_area * ORIENTATION_AREA_WEIGHT + overlap_length

                total_turn_risk += risk_metric
                
                if i == num_turn_checks:
                    final_position_risk = risk_metric

                if overlap_area > ORIENTATION_AREA_TOL or overlap_length > ORIENTATION_CONTACT_LENGTH_TOL:
                    is_safe = False

            # Penalize distance from the original point to favor smaller adjustments
            dist_penalty = math.hypot(pos_x - x, pos_y - y)
            
            if is_safe:
                # Prioritize safe options with the smallest position and angle change
                total_penalty = dist_penalty + abs(delta_angle) * 0.1
                safe_options.append((total_penalty, pos_x, pos_y, theta))
            else:
                combined_risk = total_turn_risk + final_position_risk * 2.0 + dist_penalty
                risky_options.append((combined_risk, pos_x, pos_y, theta))

    if safe_options:
        # Find the best safe option (lowest penalty)
        _, best_pos_x, best_pos_y, best_theta = min(safe_options, key=lambda r: r[0])
        return best_pos_x, best_pos_y, best_theta, 0.0

    if risky_options:
        min_risk, best_pos_x, best_pos_y, best_theta = min(risky_options, key=lambda r: r[0])
        return best_pos_x, best_pos_y, best_theta, min_risk

    return x, y, last_theta, 0.0


@dataclass
class PenState:
    """A snapshot of the pen's state at a specific time."""
    timestamp: float
    x: float
    y: float
    theta: float
    pen_down: bool
    polyline_index: int
    segment_index: int


def simulate_segments(segments, step_size) -> Tuple[List[dict], List[PenState], dict]:
    """
    Simulates the pen's path, returning visualization states, a trajectory, and metrics.
    """
    sim_states: List[dict] = []
    trajectory: List[PenState] = []
    time_cursor = 0.0

    ink_visible = None
    ink_planner = None
    erased_length_total, drawn_total, travel_up_total, turn_total_rad = 0.0, 0.0, 0.0, 0.0
    last_theta = 0.0
    turn_speed_rad = np.deg2rad(ROBOT_TURN_SPEED_DEG_PER_SEC)

    # Add initial state
    trajectory.append(PenState(time_cursor, 0.0, 0.0, 0.0, False, -1, -1))

    iterable = tqdm(segments, desc="Simulating pen path")
    for seg_idx, (pen_down, ls) in enumerate(iterable):
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
                target_x, target_y, target_theta, _ = find_safe_orientation(
                    back[0], back[1], orientation_ink, last_theta
                )

                step_length = segment_length(prev_back, (target_x, target_y))
                step_duration_move = step_length / ROBOT_SPEED if ROBOT_SPEED > 0 else 0.0
                delta_total = shortest_angle_diff(last_theta, target_theta)

                # --- Rotation Phase ---
                if turn_speed_rad > 0 and abs(delta_total) > 1e-6:
                    turn_duration_total = abs(delta_total) / turn_speed_rad
                    # Break turn into smaller steps for smooth animation and trajectory
                    num_turn_steps = max(1, int(math.ceil(turn_duration_total * ANIMATION_FPS)))
                    
                    step_delta = delta_total / num_turn_steps
                    step_dt = turn_duration_total / num_turn_steps

                    for _ in range(num_turn_steps):
                        last_theta = normalize_angle(last_theta + step_delta)
                        time_cursor += step_dt
                        turn_total_rad += abs(step_delta)

                        # Add intermediate state for command generation
                        trajectory.append(PenState(time_cursor, prev_back[0], prev_back[1], last_theta, pen_down, seg_idx, i))

                        # Add intermediate state for visualization (robot turning in place)
                        robot_poly_turn = get_robot_body(prev_back[0], prev_back[1], last_theta, ROBOT_SIDE_LENGTH)
                        sim_states.append({"robot_poly": robot_poly_turn, "ink": ink_visible})
                else:
                    # If no rotation, ensure the state is captured before translation
                    if len(trajectory) == 0 or trajectory[-1].timestamp < time_cursor:
                         trajectory.append(PenState(time_cursor, prev_back[0], prev_back[1], last_theta, pen_down, seg_idx, i))

                # --- Translation Phase ---
                if step_duration_move > 1e-9:
                    time_cursor += step_duration_move
                    trajectory.append(PenState(time_cursor, target_x, target_y, last_theta, pen_down, seg_idx, i))

                robot_poly = get_robot_body(target_x, target_y, last_theta, ROBOT_SIDE_LENGTH)
                if ink_visible is not None:
                    length_before = line_length(ink_visible)
                    diff = ink_visible.difference(robot_poly)
                    ink_visible = diff if diff is not None and not diff.is_empty else None
                    length_after = line_length(ink_visible) if ink_visible is not None else 0.0
                    erased_length_total += (length_before - length_after)

                if pen_down:
                    tiny = LineString([prev_back, (target_x, target_y)])
                    draw_len = tiny.length
                    drawn_total += draw_len
                    if ink_visible is not None:
                        ink_visible = ink_visible.difference(tiny)
                    ink_visible = tiny if ink_visible is None else unary_union([ink_visible, tiny])
                    ink_planner = tiny if ink_planner is None else unary_union([ink_planner, tiny])
                else:
                    travel_up_total += segment_length(prev_back, (target_x, target_y))

                sim_states.append({"robot_poly": robot_poly, "ink": ink_visible})
                prev_back = (target_x, target_y)

    total_duration_sec = time_cursor
    metrics = {
        "drawn_length": drawn_total,
        "penup_travel_length": travel_up_total,
        "erased_length": erased_length_total,
        "total_duration_sec": total_duration_sec,
    }
    return sim_states, trajectory, metrics


def trajectory_to_commands(trajectory: List[PenState], pen_offset_r: float, wheelbase_l: float) -> List[WheelCommand]:
    """
    Converts a pen trajectory into physically-correct wheel commands for a center-pivoting robot.
    """
    commands: List[WheelCommand] = []
    if len(trajectory) < 2:
        return []

    for i in range(len(trajectory) - 1):
        start = trajectory[i]
        end = trajectory[i+1]

        dt = end.timestamp - start.timestamp
        if dt < 1e-9:
            continue

        # Calculate pen velocity in world frame
        v_px_world = (end.x - start.x) / dt
        v_py_world = (end.y - start.y) / dt
        
        # Calculate angular velocity (handle wraparound)
        delta_theta = shortest_angle_diff(start.theta, end.theta)
        omega = delta_theta / dt

        # Use the average orientation over the step for more stable calculations
        avg_theta = normalize_angle(start.theta + delta_theta / 2.0)
        
        # Calculate the required world-frame velocity of the ROBOT'S CENTER
        # to make the pen follow its path.
        v_cx_world = v_px_world + pen_offset_r * math.sin(avg_theta) * omega
        v_cy_world = v_py_world - pen_offset_r * math.cos(avg_theta) * omega

        vx_body, vy_body = _world_to_body_velocity(v_cx_world, v_cy_world, avg_theta)
        wheel_speeds = compute_kiwi_wheel_speeds(vx_body, vy_body, omega, wheelbase_l)

        # The command's "pose" is the pen's pose, which is what we are tracking.
        cmd_pose = RobotPose(x=start.x, y=start.y, yaw=start.theta)
        body_twist = BodyTwist(vx=vx_body, vy=vy_body, omega=omega, pen_down=start.pen_down)
        world_vel_center = (v_cx_world, v_cy_world)

        commands.append(WheelCommand(
            timestamp=start.timestamp,
            duration=dt,
            pose=cmd_pose,
            body_twist=body_twist,
            world_velocity=world_vel_center,
            wheel_speeds=tuple(float(w) for w in wheel_speeds),
            pen_down=start.pen_down,
            polyline_index=start.polyline_index,
            segment_index=start.segment_index,
        ))
        
    return commands


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
    sim_states, trajectory, metrics = simulate_segments(route_plan.segments, STEP_SIZE)
    commands = trajectory_to_commands(trajectory,
                                  pen_offset_r=PEN_OFFSET_R,
                                  wheelbase_l=ROBOT_WHEELBASE_L)

    print("Simulation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")

    if commands:
        metadata = {
            "robot_side_length": ROBOT_SIDE_LENGTH,
            "robot_radius_l": ROBOT_RADIUS,
            "robot_speed": ROBOT_SPEED,
            "robot_turn_speed_deg_per_sec": ROBOT_TURN_SPEED_DEG_PER_SEC,
        }
        output_path = "wheel_commands.json"
        save_commands_to_json(commands, output_path, metadata, wheel_radius=WHEEL_RADIUS)
        print(f"Saved wheel command schedule to {output_path}")

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
        "--emit-wheel-test",
        action="store_true",
        help="Generate staged wheel verification commands and exit",
    )
    parser.add_argument(
        "--test-linear-speed",
        type=float,
        default=0.2,
        help="Linear speed (m/s) used for staged wheel test translations",
    )
    parser.add_argument(
        "--test-angular-speed",
        type=float,
        default=0.5,
        help="Angular speed (rad/s) used for staged wheel test rotations",
    )
    parser.add_argument(
        "--test-translation-dist",
        type=float,
        default=1.0,
        help="Distance (m) each translation test should travel",
    )
    parser.add_argument(
        "--test-rotation-angle",
        type=float,
        default=2 * math.pi,
        help="Rotation angle (rad) for pure rotation/arc tests",
    )
    parser.add_argument(
        "--test-arc-radius",
        type=float,
        default=0.25,
        help="Radius (m) for the circular arc rotation test",
    )
    parser.add_argument(
        "--test-mixed-rotation",
        type=float,
        default=math.pi,
        help="Rotation angle (rad) to execute during the mixed translation+rotation test",
    )
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

    # Emit staged wheel test commands and exit
    if args.emit_wheel_test:
        print("Generating staged wheel verification commands...")
        stages = []
        commands: List[WheelCommand] = []
        time_cursor = 0.0

        def add_stage(label: str, duration: float, vx_b: float, vy_b: float, omega: float):
            nonlocal time_cursor, commands, stages
            wheel_speeds = compute_kiwi_wheel_speeds(vx_b, vy_b, omega, ROBOT_WHEELBASE_L)
            commands.append(
                WheelCommand(
                    timestamp=time_cursor,
                    duration=duration,
                    pose=RobotPose(0.0, 0.0, 0.0),
                    body_twist=BodyTwist(vx=vx_b, vy=vy_b, omega=omega, pen_down=True),
                    world_velocity=(vx_b, vy_b),
                    wheel_speeds=tuple(float(w) for w in wheel_speeds),
                    pen_down=True,
                    polyline_index=len(stages),
                    segment_index=0,
                )
            )
            stages.append(label)
            time_cursor += duration

        def distance_duration(speed_mag: float, distance: float) -> float:
            speed_mag = max(1e-6, abs(speed_mag))
            return abs(distance) / speed_mag

        # Speeds/distances for staged testing (configurable via CLI)
        V = max(1e-6, args.test_linear_speed)
        OMG = max(1e-6, args.test_angular_speed)
        TRANS_DIST = max(1e-6, args.test_translation_dist)
        ROT_ANGLE = max(1e-6, args.test_rotation_angle)
        ARC_RADIUS = max(1e-6, args.test_arc_radius)
        MIX_ROT_ANGLE = max(1e-6, args.test_mixed_rotation)

        # 1) Positive X (forward) — travel 1 meter
        add_stage("positive_x", distance_duration(V, TRANS_DIST), V, 0.0, 0.0)

        # 2) Positive Y (left) — travel 1 meter
        add_stage("positive_y", distance_duration(V, TRANS_DIST), 0.0, V, 0.0)

        # 3) 45 degrees (x=y) — travel 1 meter along diagonal
        diag_speed = V / math.sqrt(2.0)
        add_stage("diag_45_xy", distance_duration(V, TRANS_DIST), diag_speed, diag_speed, 0.0)

        # 4) Rotation in place CW (negative omega) — one full rotation
        add_stage("rotate_cw", ROT_ANGLE / OMG, 0.0, 0.0, -OMG)

        # 5) Rotation in place CCW (positive omega) — one full rotation
        add_stage("rotate_ccw", ROT_ANGLE / OMG, 0.0, 0.0, OMG)

        # 6) Rotation around front wheel (ICC at +x = wheelbase radius) — one rotation
        add_stage("rotate_about_front_wheel", ROT_ANGLE / OMG, 0.0, OMG * ROBOT_WHEELBASE_L, OMG)

        # 7) Rotation around an arc (ICC at 0, R) — one circle of radius R
        add_stage("arc_radius", ROT_ANGLE / OMG, -OMG * ARC_RADIUS, 0.0, OMG)

        # 8) Rotation and translation combined — travel configured distance while rotating MIX_ROT_ANGLE
        mix_vx, mix_vy = 0.12, 0.05
        mix_speed = math.hypot(mix_vx, mix_vy)
        mix_duration = distance_duration(mix_speed, TRANS_DIST)
        mix_omega = MIX_ROT_ANGLE / mix_duration
        add_stage("mixed_translation_rotation", mix_duration, mix_vx, mix_vy, mix_omega)

        metadata = {
            "stages": stages,
            "robot_side_length_m": ROBOT_SIDE_LENGTH,
            "robot_radius_l_m": ROBOT_WHEELBASE_L,
            "wheel_radius_m": WHEEL_RADIUS,
            "notes": (
                "Pen remains down for all stages. Linear stages cover 1 m; rotation stages cover 2π rad "
                "(mixed stage ≈ π rad). Wheel speeds converted to rad/s using wheel_radius."
            ),
        }
        out_path = "wheel_commands_test.json"
        save_commands_to_json(commands, out_path, metadata, wheel_radius=WHEEL_RADIUS)
        print(f"Saved staged wheel test commands to {out_path}")
        return

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

