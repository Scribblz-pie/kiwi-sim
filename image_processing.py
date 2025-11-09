"""Image processing utilities for the drawing robot pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import json
import math

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, snap

import os

# Optional dependencies
_HAS_SKIMAGE = False
_HAS_CV2 = False
_HAS_SCIPY = False
skio = None
rgb2gray = None
threshold_otsu = None
skeletonize = None
find_contours = None
approximate_polygon = None

DEDUPLICATION_DISTANCE_TOLERANCE_DEFAULT = 0.1  # Canvas units after rescale

try:  # scikit-image
    from skimage import io as skio
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu
    from skimage.morphology import skeletonize
    from skimage.measure import find_contours, approximate_polygon
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover - optional
    pass

try:  # OpenCV
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover - optional
    pass

try:  # SciPy
    from scipy.interpolate import splprep, splev
    from scipy.ndimage import binary_erosion
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - optional
    binary_erosion = None


@dataclass
class ImageStages:
    """Container holding intermediate outputs of the image-to-path pipeline."""

    image: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    polylines: List[List[Tuple[float, float]]]
    hierarchy: Optional[np.ndarray]
    rescaled_polylines: List[List[Tuple[float, float]]]
    bounds: Tuple[float, float, float, float]
    canvas_height: float


@dataclass
class RobotPose:
    """Robot pose in canvas coordinates."""

    x: float
    y: float
    yaw: float


@dataclass
class BodyTwist:
    """Body-frame velocity command."""

    vx: float
    vy: float
    omega: float


@dataclass
class WheelCommand:
    """Timed wheel command entry suitable for simulation or playback."""

    timestamp: float
    duration: float
    pose: RobotPose
    body_twist: BodyTwist
    world_velocity: Tuple[float, float]
    wheel_speeds: Tuple[float, float, float]
    polyline_index: Optional[int] = None
    segment_index: Optional[int] = None


@dataclass
class WheelCommandSchedule:
    """Collection of wheel commands plus metadata for downstream consumers."""

    commands: List[WheelCommand]
    wheel_names: Tuple[str, str, str] = ("wheel_1", "wheel_2", "wheel_3")
    metadata: Optional[Dict[str, float]] = None

    def to_jsonable(self) -> Dict[str, object]:
        meta = self.metadata or {}
        return {
            "wheel_names": list(self.wheel_names),
            "metadata": meta,
            "commands": [
                {
                    "t": cmd.timestamp,
                    "dt": cmd.duration,
                    "pose": {
                        "x": cmd.pose.x,
                        "y": cmd.pose.y,
                        "yaw": cmd.pose.yaw,
                    },
                    "body_twist": {
                        "vx": cmd.body_twist.vx,
                        "vy": cmd.body_twist.vy,
                        "omega": cmd.body_twist.omega,
                    },
                    "world_velocity": {
                        "vx": cmd.world_velocity[0],
                        "vy": cmd.world_velocity[1],
                    },
                    "wheel_speeds": list(cmd.wheel_speeds),
                    "polyline": cmd.polyline_index,
                    "segment": cmd.segment_index,
                }
                for cmd in self.commands
            ],
        }

    def save_json(self, filepath: str, *, indent: int = 2) -> None:
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(self.to_jsonable(), handle, indent=indent)


_KIWI_JACOBIAN_TEMPLATE = np.array(
    [
        [0.0, -np.sqrt(3.0) / 3.0, np.sqrt(3.0) / 3.0],
        [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
        [0.0, 0.0, 0.0],
    ]
)


def _world_to_body_velocity(vx: float, vy: float, yaw: float) -> Tuple[float, float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    bx = cos_yaw * vx + sin_yaw * vy
    by = -sin_yaw * vx + cos_yaw * vy
    return bx, by


def compute_kiwi_wheel_speeds(
    vx: float,
    vy: float,
    omega: float,
    robot_radius: float,
) -> np.ndarray:
    if robot_radius <= 0:
        raise ValueError("robot_radius must be positive for kiwi drive kinematics")

    j_mat = _KIWI_JACOBIAN_TEMPLATE.copy()
    j_mat[2, :] = 1.0 / (3.0 * robot_radius)
    twist = np.array([vx, vy, omega], dtype=float)
    return np.linalg.solve(j_mat, twist)


def _poly_to_linestring(poly: Sequence[Tuple[float, float]]) -> Optional[LineString]:
    try:
        return LineString(poly)
    except Exception:
        clean = [poly[0]]
        for p in poly[1:]:
            if p != clean[-1]:
                clean.append(p)
        if len(clean) >= 2:
            return LineString(clean)
        return None


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk and normalize to [0, 1]."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if _HAS_SKIMAGE and skio is not None:
        img = skio.imread(image_path).astype(float)
    else:  # pragma: no cover - fallback path
        import matplotlib.image as mpimg

        img = mpimg.imread(image_path).astype(float)
        if img.max() > 1.0:
            img = img / 255.0

    if img.ndim == 3 and img.shape[-1] == 4:
        # Drop alpha channel (assume premultiplied or opaque)
        img = img[..., :3]

    if img.max() > 1.0:
        img = img / img.max()

    return img


def binarize_image(img: np.ndarray) -> np.ndarray:
    """Return boolean mask where True represents ink pixels."""
    if img.ndim == 3:
        if _HAS_SKIMAGE:
            gray = rgb2gray(img)
        else:
            gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    else:
        gray = img.astype(float)
        gray = gray / (gray.max() if gray.max() else 1.0)

    if _HAS_SKIMAGE:
        t = threshold_otsu(gray)
        mask = gray < t
    else:
        mask = gray < 0.5
    return mask


def extract_polylines_from_mask(
    mask: np.ndarray,
    simplification_epsilon_factor: float = 0.0,
) -> Tuple[List[List[Tuple[float, float]]], Optional[np.ndarray]]:
    """Extract polylines from a thinned mask using OpenCV when available."""
    if not _HAS_CV2:
        return [], None

    m = (mask.astype(np.uint8) * 255)
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    polylines: List[List[Tuple[float, float]]] = []
    h, w = m.shape
    for cnt in contours:
        if simplification_epsilon_factor > 0.0 and len(cnt) > 10:
            epsilon = simplification_epsilon_factor * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

        pts = []
        for pt in cnt:
            x, y = int(pt[0][0]), int(pt[0][1])
            pts.append((float(x), float(h - 1 - y)))
        if len(pts) >= 2:
            polylines.append(pts)

    return polylines, hierarchy[0] if hierarchy is not None else None


def extract_polylines_skimage(
    img_or_mask: np.ndarray,
    level: float = 0.5,
    approx_tol: float = 0.0,
) -> Tuple[List[List[Tuple[float, float]]], Optional[np.ndarray]]:
    """
    Extract polylines using skimage.find_contours and approximate_polygon.
    - img_or_mask: float/uint mask or grayscale image in [0,1]
    - level: iso-value for contours (default 0.5 for binary masks)
    - approx_tol: polygon approximation tolerance (0 disables)
    Returns (polylines, None)
    """
    if not _HAS_SKIMAGE or find_contours is None:
        raise RuntimeError(
            "skimage extractor requested but scikit-image is not available. "
            "Install scikit-image or use --extractor cv2."
        )

    arr = img_or_mask.astype(float)
    if arr.max() > 1.0:
        arr = arr / arr.max()

    raw_contours = find_contours(arr, level)
    polylines: List[List[Tuple[float, float]]] = []
    h, w = arr.shape

    for c in raw_contours:
        # c is array of (row, col) with subpixel floats
        if approx_tol and approx_tol > 0.0:
            c = approximate_polygon(c, tolerance=approx_tol)
        pts: List[Tuple[float, float]] = []
        for rc in c:
            r, ccol = float(rc[0]), float(rc[1])
            # Convert to Cartesian: x = col, y = (h-1-r)
            pts.append((ccol, float(h - 1 - r)))
        if len(pts) >= 2:
            polylines.append(pts)

    return polylines, None


def dedupe_empty_polylines(polylines: Sequence[Sequence[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
    cleaned = []
    for poly in polylines:
        if len(poly) >= 2:
            cleaned.append(list(poly))
    return cleaned


def _canonical_poly_key(poly: Sequence[Tuple[float, float]], precision: int = 5) -> Optional[Tuple[Tuple[float, float], ...]]:
    if len(poly) < 2:
        return None
    rounded = tuple((round(x, precision), round(y, precision)) for x, y in poly)
    reversed_rounded = tuple(reversed(rounded))
    return rounded if rounded <= reversed_rounded else reversed_rounded


def deduplicate_polylines(
    polylines: Sequence[Sequence[Tuple[float, float]]],
    tolerance: float = DEDUPLICATION_DISTANCE_TOLERANCE_DEFAULT,
) -> List[List[Tuple[float, float]]]:
    """Remove duplicate and near-duplicate polylines."""

    cleaned: List[List[Tuple[float, float]]] = []
    seen_keys: set = set()

    # Always remove exact duplicates (including reversed) using canonical keys
    for poly in polylines:
        key = _canonical_poly_key(poly)
        if key is None:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        cleaned.append(list(poly))

    if tolerance <= 0:
        return cleaned

    unique_lines: List[List[Tuple[float, float]]] = []
    line_geoms: List[LineString] = []
    for poly in cleaned:
        ls = _poly_to_linestring(poly)
        if ls is None or ls.is_empty:
            continue

        is_duplicate = False
        for existing in line_geoms:
            try:
                if ls.distance(existing) < tolerance:
                    is_duplicate = True
                    break
            except Exception:
                continue

        if not is_duplicate:
            unique_lines.append(list(poly))
            line_geoms.append(ls)

    return unique_lines


def prune_micro_segments(polylines: Sequence[Sequence[Tuple[float, float]]], min_len: float = 0.25) -> List[List[Tuple[float, float]]]:
    """Remove extremely short segments from polylines."""
    cleaned: List[List[Tuple[float, float]]] = []
    for poly in polylines:
        if len(poly) < 2:
            continue
        new_poly: List[Tuple[float, float]] = [poly[0]]
        acc = 0.0
        for i in range(1, len(poly)):
            x0, y0 = new_poly[-1]
            x1, y1 = poly[i]
            seg_len = float(np.hypot(x1 - x0, y1 - y0))
            if seg_len < min_len:
                # Skip point to avoid tiny segment; merge at next longer step
                continue
            new_poly.append((x1, y1))
        if len(new_poly) >= 2:
            cleaned.append(new_poly)
    return cleaned


def rescale_polylines_to_canvas(
    polylines: Sequence[Sequence[Tuple[float, float]]],
    target_width: float,
    padding: float,
) -> Tuple[List[List[Tuple[float, float]]], Tuple[float, float, float, float], float]:
    if not polylines or all(len(p) == 0 for p in polylines):
        return dedupe_empty_polylines(polylines), (0, 0, target_width, target_width), target_width

    all_lines = []
    for p in polylines:
        ls = _poly_to_linestring(p)
        if ls:
            all_lines.append(ls)

    if not all_lines:
        return dedupe_empty_polylines(polylines), (0, 0, target_width, target_width), target_width

    all_geom = unary_union(all_lines)
    minx, miny, maxx, maxy = all_geom.bounds
    data_w = maxx - minx
    data_h = maxy - miny

    if data_w < 1e-6 or data_h < 1e-6:
        scale = 1.0
        target_height = target_width
    else:
        target_height = target_width * (data_h / data_w)
        scale = min(
            (target_width - 2 * padding) / data_w,
            (target_height - 2 * padding) / data_h,
        )

    new_w, new_h = data_w * scale, data_h * scale
    offset_x = padding + (target_width - 2 * padding - new_w) / 2.0 - (minx * scale)
    offset_y = padding + (target_height - 2 * padding - new_h) / 2.0 - (miny * scale)

    # Snap to a grid to reduce floating noise before dedup/graphing
    grid = max(1e-6, min((target_width - 2 * padding), (target_height - 2 * padding)) / 500.0)

    new_polylines: List[List[Tuple[float, float]]] = []
    for p in polylines:
        new_p = []
        for x, y in p:
            nx = x * scale + offset_x
            ny = y * scale + offset_y
            # snap to grid
            nx = round(nx / grid) * grid
            ny = round(ny / grid) * grid
            new_p.append((nx, ny))
        new_polylines.append(new_p)

    bounds = (minx * scale + offset_x, miny * scale + offset_y, maxx * scale + offset_x, maxy * scale + offset_y)
    return new_polylines, bounds, target_height


def smooth_polyline(polyline: Sequence[Tuple[float, float]], smooth_factor: float, num_points: int) -> List[Tuple[float, float]]:
    if not _HAS_SCIPY or smooth_factor == 0:
        return list(polyline)

    try:
        points = np.array(polyline)
        x, y = points[:, 0], points[:, 1]

        ok = np.where(np.diff(x) ** 2 + np.diff(y) ** 2 > 1e-10)[0]
        x = np.r_[x[ok], x[-1]]
        y = np.r_[y[ok], y[-1]]
        if len(x) < 4:
            return list(polyline)

        tck, u = splprep([x, y], s=smooth_factor, k=3)
        u_new = np.linspace(u.min(), u.max(), num_points)
        x_new, y_new = splev(u_new, tck)
        return list(zip(x_new, y_new))
    except Exception:
        return list(polyline)


def smooth_polylines(
    polylines: Sequence[Sequence[Tuple[float, float]]],
    smooth_factor: float,
    smoothing_points: int,
) -> List[List[Tuple[float, float]]]:
    smoothed = []
    for poly in polylines:
        ls = _poly_to_linestring(poly)
        if ls:
            num_points = max(50, int(ls.length * 2)) if smoothing_points <= 0 else smoothing_points
        else:
            num_points = smoothing_points if smoothing_points > 0 else 200
        smoothed.append(smooth_polyline(poly, smooth_factor, num_points))
    return smoothed


def generate_image_stages(
    image_path: str,
    target_width: float,
    padding: float,
    smooth_factor: float = 0.0,
    smoothing_points: int = 200,
    simplification_epsilon_factor: float = 0.0,
    dedup_tolerance: float = DEDUPLICATION_DISTANCE_TOLERANCE_DEFAULT,
    extractor: str = "cv2",
    approx_tol: float = 0.0,
) -> ImageStages:
    image = load_image(image_path)
    mask = binarize_image(image)
    
    # Skeletonization is removed; polylines are extracted directly from the mask contours.
    if extractor.lower() == "skimage":
        polylines, hierarchy = extract_polylines_skimage(mask.astype(float), level=0.5, approx_tol=approx_tol)
    else:
        polylines, hierarchy = extract_polylines_from_mask(mask.astype(np.uint8), simplification_epsilon_factor)

    polylines = dedupe_empty_polylines(polylines)
    polylines = deduplicate_polylines(polylines, tolerance=0.0)

    if smooth_factor > 0.0:
        smoothed = smooth_polylines(polylines, smooth_factor, smoothing_points)
    else:
        smoothed = polylines

    rescaled_polys, bounds, canvas_height = rescale_polylines_to_canvas(smoothed, target_width, padding)
    rescaled_polys = prune_micro_segments(rescaled_polys, min_len=max(0.25, target_width * 0.002))
    deduped_rescaled = deduplicate_polylines(rescaled_polys, tolerance=dedup_tolerance)

    return ImageStages(
        image=image,
        mask=mask,
        skeleton=mask,  # Pass mask in place of skeleton
        polylines=smoothed,
        hierarchy=hierarchy,
        rescaled_polylines=deduped_rescaled,
        bounds=bounds,
        canvas_height=canvas_height,
    )


def _plan_straight_segment(
    dx: float,
    dy: float,
    pose: RobotPose,
    nominal_speed: float,
    timestep: float,
    robot_radius: float,
    max_wheel_speed: Optional[float],
) -> Optional[Tuple[float, np.ndarray, Tuple[float, float], float, np.ndarray]]:
    seg_len = float(np.hypot(dx, dy))
    if seg_len < 1e-9:
        return None

    duration = max(seg_len / max(nominal_speed, 1e-9), timestep)
    vel_world = np.array([dx, dy], dtype=float) / duration
    vx_body, vy_body = _world_to_body_velocity(vel_world[0], vel_world[1], pose.yaw)
    omega = 0.0
    wheel_speeds = compute_kiwi_wheel_speeds(vy_body, vx_body, omega, robot_radius)

    if max_wheel_speed is not None and max_wheel_speed > 0:
        max_abs = float(np.max(np.abs(wheel_speeds)))
        if max_abs > max_wheel_speed + 1e-9:
            scale = max_abs / max_wheel_speed
            duration *= scale
            vel_world = np.array([dx, dy], dtype=float) / duration
            vx_body, vy_body = _world_to_body_velocity(vel_world[0], vel_world[1], pose.yaw)
            wheel_speeds = compute_kiwi_wheel_speeds(vx_body, vy_body, omega, robot_radius)
            max_abs = float(np.max(np.abs(wheel_speeds)))
            if max_abs > max_wheel_speed + 1e-6:
                raise ValueError(
                    "Segment requires wheel speed beyond max_wheel_speed even after scaling."
                )

    return duration, vel_world, (vx_body, vy_body), omega, wheel_speeds


def generate_wheel_command_schedule(
    polylines: Sequence[Sequence[Tuple[float, float]]],
    *,
    robot_radius: float,
    nominal_speed: float = 20.0,
    timestep: float = 0.05,
    max_wheel_speed: Optional[float] = None,
    initial_pose: Optional[RobotPose] = None,
    dwell_time: float = 0.0,
    wheel_names: Tuple[str, str, str] = ("wheel_1", "wheel_2", "wheel_3"),
    metadata: Optional[Dict[str, float]] = None,
) -> WheelCommandSchedule:
    if nominal_speed <= 0:
        raise ValueError("nominal_speed must be positive")
    if timestep <= 0:
        raise ValueError("timestep must be positive")
    if robot_radius <= 0:
        raise ValueError("robot_radius must be positive")

    usable_polys: List[List[Tuple[float, float]]] = [list(poly) for poly in polylines if len(poly) >= 1]
    if not usable_polys:
        sched_meta = metadata or {}
        sched_meta = {
            **sched_meta,
            "nominal_speed": float(nominal_speed),
            "timestep": float(timestep),
            "robot_radius": float(robot_radius),
        }
        return WheelCommandSchedule(commands=[], wheel_names=wheel_names, metadata=sched_meta)

    start_point = usable_polys[0][0]
    if initial_pose is None:
        pose = RobotPose(start_point[0], start_point[1], 0.0)
    else:
        pose = RobotPose(initial_pose.x, initial_pose.y, initial_pose.yaw)

    commands: List[WheelCommand] = []
    time_cursor = 0.0

    def append_segment(
        target: Tuple[float, float],
        poly_idx: Optional[int],
        seg_idx: Optional[int],
    ) -> None:
        nonlocal pose, time_cursor
        dx = target[0] - pose.x
        dy = target[1] - pose.y
        planned = _plan_straight_segment(
            dx,
            dy,
            pose,
            nominal_speed,
            timestep,
            robot_radius,
            max_wheel_speed,
        )
        if planned is None:
            pose = RobotPose(target[0], target[1], pose.yaw)
            return

        duration, vel_world, body_vel, omega, wheel_speeds = planned
        steps = max(1, int(math.ceil(duration / timestep)))
        step_dt = duration / steps
        wheel_tuple = tuple(float(w) for w in wheel_speeds)

        for _ in range(steps):
            commands.append(
                WheelCommand(
                    timestamp=time_cursor,
                    duration=step_dt,
                    pose=RobotPose(pose.x, pose.y, pose.yaw),
                    body_twist=BodyTwist(vx=body_vel[0], vy=body_vel[1], omega=omega),
                    world_velocity=(float(vel_world[0]), float(vel_world[1])),
                    wheel_speeds=wheel_tuple,
                    polyline_index=poly_idx,
                    segment_index=seg_idx,
                )
            )
            pose = RobotPose(
                pose.x + float(vel_world[0]) * step_dt,
                pose.y + float(vel_world[1]) * step_dt,
                pose.yaw,
            )
            time_cursor += step_dt

        pose = RobotPose(target[0], target[1], pose.yaw)

    # If initial pose is not at starting point, translate there first.
    if math.hypot(pose.x - start_point[0], pose.y - start_point[1]) > 1e-6:
        append_segment(start_point, None, None)

    for poly_idx, poly in enumerate(usable_polys):
        if len(poly) < 2:
            continue
        # Ensure pose is at start of current polyline
        if math.hypot(pose.x - poly[0][0], pose.y - poly[0][1]) > 1e-6:
            append_segment(poly[0], poly_idx, -1)

        for seg_idx in range(1, len(poly)):
            append_segment(poly[seg_idx], poly_idx, seg_idx - 1)

        if dwell_time > 0:
            commands.append(
                WheelCommand(
                    timestamp=time_cursor,
                    duration=dwell_time,
                    pose=RobotPose(pose.x, pose.y, pose.yaw),
                    body_twist=BodyTwist(0.0, 0.0, 0.0),
                    world_velocity=(0.0, 0.0),
                    wheel_speeds=(0.0, 0.0, 0.0),
                    polyline_index=poly_idx,
                    segment_index=None,
                )
            )
            time_cursor += dwell_time

    sched_meta = metadata or {}
    sched_meta = {
        **sched_meta,
        "nominal_speed": float(nominal_speed),
        "timestep": float(timestep),
        "robot_radius": float(robot_radius),
        "total_time": float(time_cursor),
    }

    return WheelCommandSchedule(commands=commands, wheel_names=wheel_names, metadata=sched_meta)


def export_wheel_schedule_to_json(
    schedule: WheelCommandSchedule,
    filepath: str,
    *,
    indent: int = 2,
) -> None:
    schedule.save_json(filepath, indent=indent)


def _plot_polylines(ax: plt.Axes, polylines: Sequence[Sequence[Tuple[float, float]]], title: str):
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.3)
    segments = []
    for poly in polylines:
        if len(poly) >= 2:
            for i in range(len(poly) - 1):
                segments.append([poly[i], poly[i + 1]])
    if segments:
        coll = LineCollection(segments, colors="red", linewidths=1.5)
        ax.add_collection(coll)
    ax.autoscale()


def visualize_image_stages(stages: ImageStages, show: bool = True):
    """Visualize the major stages of the image-to-path pipeline."""

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax_orig, ax_mask, ax_skel = axes[0]
    ax_poly, ax_rescaled, ax_empty = axes[1]

    ax_orig.imshow(stages.image, cmap="gray")
    ax_orig.set_title("Original Image")
    ax_orig.axis("off")

    ax_mask.imshow(stages.mask, cmap="gray")
    ax_mask.set_title("Binarized Mask")
    ax_mask.axis("off")

    ax_skel.imshow(stages.skeleton, cmap="gray")
    ax_skel.set_title("Skeletonized")
    ax_skel.axis("off")

    _plot_polylines(ax_poly, stages.polylines, "Extracted Polylines")
    _plot_polylines(ax_rescaled, stages.rescaled_polylines, "Rescaled Polylines")

    ax_empty.axis("off")
    ax_empty.text(0.5, 0.5, f"Bounds: {stages.bounds}\nCanvas H: {stages.canvas_height:.2f}",
                  ha="center", va="center")

    plt.tight_layout()
    if show:
        plt.show()


__all__ = [
    "ImageStages",
    "generate_image_stages",
    "visualize_image_stages",
    "rescale_polylines_to_canvas",
    "smooth_polylines",
    "load_image",
    "binarize_image",
    "skeletonize_mask",
    "deduplicate_polylines",
    "extract_polylines_skimage",
    "RobotPose",
    "BodyTwist",
    "WheelCommand",
    "WheelCommandSchedule",
    "compute_kiwi_wheel_speeds",
    "generate_wheel_command_schedule",
    "export_wheel_schedule_to_json",
]


