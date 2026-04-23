"""Render camera frustum wireframes ONLY (no point cloud) using Open3D.

Reads the viewpoint_config.json produced by render_points.py to use the
exact same camera setup, ensuring perfect alignment for compositing.

Output: one PNG per frame with camera frustums on a solid background.

Usage:
    python scripts/visualize/geometry/render/render_cameras.py \
        --pose_path data/datasets/kitti/07/mat_xyz.txt \
        --viewpoint_config data/output/kitti/07/progressive/viewpoint_config.json \
        --output_dir data/output/kitti/07/progressive/CAM2D
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from os.path import abspath, dirname
from contextlib import contextmanager

visualize_root = dirname(dirname(dirname(abspath(__file__))))
release_root = dirname(dirname(visualize_root))
if visualize_root not in sys.path: sys.path.insert(0, visualize_root)
if release_root not in sys.path: sys.path.insert(0, release_root)

from utils.color_utils import rainbow_colormap  # noqa: E402
from utils.camera_utils import prepare_poses, frustum_points, make_intrinsic, interpolate_viewpoints  # noqa: E402
from utils.render_utils import blend_rgb_to_bg, hex_to_float_rgb, find_image_files, load_texture_array, build_camera_image_quad, build_frustum_pointcloud, configure_offscreen_renderer, build_single_frustum_pointcloud, build_trajectory_segment_pointcloud  # noqa: E402


def normalize_vector(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.copy()
    return vec / norm


def build_lookat_pose(eye: np.ndarray, lookat: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build an OpenCV-style c2w pose from eye/lookat/up."""
    forward = normalize_vector(lookat - eye)
    down_ref = -normalize_vector(up)
    down = normalize_vector(down_ref - forward * np.dot(down_ref, forward))
    right = normalize_vector(np.cross(down, forward))
    down = normalize_vector(np.cross(forward, right))

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def quad_visible_in_camera(
    quad_points_world: np.ndarray,
    render_w2c: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    near_depth: float = 0.01,
    require_front_facing: bool = False,
    min_projected_area: float = 0.0,
) -> bool:
    """Cheap visibility test for a camera-image quad in the current render camera."""
    cam = quad_points_world @ render_w2c[:3, :3].T + render_w2c[:3, 3]
    z = cam[:, 2]
    valid = z > near_depth
    if not np.any(valid):
        return False

    uv = cam[valid, :2] / z[valid, None]
    u = intrinsic[0, 0] * uv[:, 0] + intrinsic[0, 2]
    v = intrinsic[1, 1] * uv[:, 1] + intrinsic[1, 2]

    inside = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    visible = bool(np.any(inside)) or bool(
        u.min() < width and u.max() >= 0.0 and v.min() < height and v.max() >= 0.0
    )
    if not visible:
        return False

    if require_front_facing:
        plane = quad_points_world[:4]
        normal = np.cross(plane[1] - plane[0], plane[2] - plane[0])
        center = plane.mean(axis=0)
        render_cam_center = -render_w2c[:3, :3].T @ render_w2c[:3, 3]
        view_dir = render_cam_center - center
        if float(np.dot(normal, view_dir)) <= 0.0:
            return False

    min_projected_area = float(max(min_projected_area, 0.0))
    if min_projected_area > 0.0:
        u0 = max(float(u.min()), 0.0)
        u1 = min(float(u.max()), float(width))
        v0 = max(float(v.min()), 0.0)
        v1 = min(float(v.max()), float(height))
        projected_area = max(u1 - u0, 0.0) * max(v1 - v0, 0.0)
        if projected_area < min_projected_area:
            return False

    return True


def compute_history_opacity(
    age: int, fade_window: int, min_opacity: float, mode: str = "forward"
) -> float:
    fade_window = max(int(fade_window), 0)
    min_opacity = float(np.clip(min_opacity, 0.0, 1.0))
    if fade_window <= 0:
        return 1.0 if age <= 0 else min_opacity
    t = min(max(age, 0), fade_window) / float(fade_window)
    if mode == "reverse":
        return float(min_opacity * (1.0 - t) + t)
    return float((1.0 - t) + t * min_opacity)


def build_point_material(rendering, point_size: float, opacity: float):
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = point_size
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    return material


def make_alpha_rgb(opacity: float) -> np.ndarray:
    alpha = float(np.clip(opacity, 0.0, 1.0))
    return np.array([alpha, alpha, alpha], dtype=np.float64)


def extract_alpha_from_render(alpha_img) -> np.ndarray:
    alpha_arr = np.asarray(alpha_img)
    if alpha_arr.ndim == 2:
        return alpha_arr.astype(np.uint8)
    return alpha_arr[..., :3].max(axis=-1).astype(np.uint8)


def write_rgba_image(rgb_img, alpha_img, out_path: str):
    rgb_arr = np.asarray(rgb_img)
    alpha_arr = extract_alpha_from_render(alpha_img)
    rgba_arr = np.dstack([rgb_arr[..., :3], alpha_arr]).astype(np.uint8)
    Image.fromarray(rgba_arr, mode="RGBA").save(out_path)


class StageProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self.stats: OrderedDict[str, dict[str, float | int]] = OrderedDict()
        self.t0 = time.perf_counter()

    @contextmanager
    def track(self, name: str):
        if not self.enabled:
            yield
            return
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t)

    def add(self, name: str, dt: float):
        if not self.enabled:
            return
        entry = self.stats.setdefault(name, {"seconds": 0.0, "calls": 0})
        entry["seconds"] += float(dt)
        entry["calls"] += 1

    def report(self, n_frames: int, n_saved_frames: int):
        total = time.perf_counter() - self.t0
        rows = []
        for name, entry in self.stats.items():
            seconds = float(entry["seconds"])
            calls = int(entry["calls"])
            rows.append(
                dict(
                    stage=name,
                    seconds=seconds,
                    calls=calls,
                    avg_ms_per_call=seconds * 1000.0 / max(calls, 1),
                    avg_ms_per_frame=seconds * 1000.0 / max(n_frames, 1),
                    pct_total=seconds * 100.0 / max(total, 1e-8),
                )
            )
        rows.sort(key=lambda x: x["seconds"], reverse=True)
        return dict(
            total_seconds=total,
            n_frames=int(n_frames),
            n_saved_frames=int(n_saved_frames),
            stages=rows,
        )

    def print(self, n_frames: int, n_saved_frames: int):
        report = self.report(n_frames, n_saved_frames)
        print("\n[render_cameras profile]")
        print(
            f"  total={report['total_seconds']:.3f}s, "
            f"frames={report['n_frames']}, saved={report['n_saved_frames']}"
        )
        for row in report["stages"]:
            print(
                "  "
                f"{row['stage']}: {row['seconds']:.3f}s "
                f"({row['pct_total']:.1f}%), calls={row['calls']}, "
                f"avg_call={row['avg_ms_per_call']:.2f}ms, "
                f"avg_frame={row['avg_ms_per_frame']:.2f}ms"
            )

    def save(self, path: str, n_frames: int, n_saved_frames: int):
        if not self.enabled or not path:
            return
        out_dir = dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(self.report(n_frames, n_saved_frames), indent=2))


def render_camera_frames(
    pose_path: str,
    viewpoint_config: str,
    output_dir: str,
    pred_pose_path: str = "",
    align_gt_to_whole: int = 0,
    normalize_first: int = -1,
    pose_in_xyz_frame: int = -1,
    bg_color: str = "#ffffff",
    frustum_size: float = 0.15,
    frustum_history: int = 0,
    frustum_history_stride: int = 1,
    history_size_step: float = 0.0,
    point_size: float = 3.5,
    point_samples: int = 128,
    frustum_aspect: float = -1.0,
    camera_image_dir: str = "",
    camera_image_max_dim: int = 256,
    render_camera_images: int = 0,
    camera_image_history_scope: str = "all",
    history_fade_window: int = 0,
    history_min_opacity: float = 0.2,
    history_fade_mode: str = "forward",
    save_from_frame: int = 0,
    start_from_frame: int = 0,
    end_at_frame: int = -1,
    camera_image_cull_to_view: int = 1,
    camera_image_cache_limit: int = 128,
    camera_image_require_front_facing: int = 0,
    camera_image_min_projected_area: float = 0.0,
    save_rgba: int = 1,
    profile_json: str = "",
    profile_print: int = 0,
    viewpoint_interp: int = 1,
    force_clean_exit: int = 0,
):
    import open3d as o3d
    import open3d.visualization.rendering as rendering  # type: ignore[import-unresolved]

    profiler = StageProfiler(enabled=bool(profile_json) or bool(profile_print))

    # Load config
    with profiler.track("load_viewpoint_config"):
        with open(viewpoint_config, "r") as f:
            config = json.load(f)

    width = config["width"]
    height = config["height"]
    fov = config.get("fov_y_deg", 60.0)
    if not bg_color:
        bg_color = config.get("bg_color", "#ffffff")
    view_mode = config.get("view_mode", "follow")
    config_viewpoint_interp = max(int(config.get("viewpoint_interp", 1)), 1)
    requested_viewpoint_interp = max(int(viewpoint_interp), 1)
    # Check if using extrinsic-based camera (firstperson mode)
    use_extrinsic_camera = view_mode == "firstperson" and "intrinsic" in config
    base_viewpoints = config["viewpoints"]

    if config_viewpoint_interp > 1:
        if requested_viewpoint_interp not in {1, config_viewpoint_interp}:
            raise ValueError(
                "viewpoint_config already contains interpolated viewpoints "
                f"(factor={config_viewpoint_interp}); refusing ambiguous "
                f"--viewpoint_interp={requested_viewpoint_interp}. Use 1 or the same factor."
            )
        viewpoints = base_viewpoints
        print(
            "Using interpolated render viewpoints from config: "
            f"factor={config_viewpoint_interp}, frames={len(viewpoints)}"
        )
    elif requested_viewpoint_interp > 1:
        viewpoints = interpolate_viewpoints(
            base_viewpoints,
            factor=requested_viewpoint_interp,
            use_extrinsic_camera=use_extrinsic_camera,
        )
        print(
            "Interpolating render viewpoints on the fly: "
            f"factor={requested_viewpoint_interp}, base={len(base_viewpoints)}, "
            f"dense={len(viewpoints)}"
        )
    else:
        viewpoints = base_viewpoints

    if use_extrinsic_camera:
        intrinsic = np.array(config["intrinsic"])
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        )
    else:
        intrinsic = make_intrinsic(fov, width, height)

    # Load poses
    print("Loading poses...")
    with profiler.track("load_poses"):
        poses, pose_meta = prepare_poses(
            pose_path,
            pred_pose_path=pred_pose_path,
            align_gt_to_whole=align_gt_to_whole,
            normalize_first=normalize_first,
            pose_in_xyz_frame=pose_in_xyz_frame,
        )
    print(f"  {len(poses)} poses loaded")
    if align_gt_to_whole:
        print("  GT poses aligned to `whole.ply` frame")
    if pose_meta.get("loaded_xyz_pose_cache"):
        print(f"  Reused cached xyz-frame poses: {pose_meta['xyz_pose_path']}")
    if pose_meta.get("saved_xyz_pose_cache"):
        print(f"  Saved cached xyz-frame poses: {pose_meta['xyz_pose_path']}")
    if pose_meta.get("normalized_first"):
        print("  Poses normalized by first frame")

    camera_image_files: list[str] = []
    if camera_image_dir:
        with profiler.track("scan_camera_images"):
            camera_image_files = find_image_files(camera_image_dir)
        if not camera_image_files:
            raise ValueError(f"No images found in camera_image_dir={camera_image_dir}")
        if len(camera_image_files) != len(poses):
            raise ValueError(
                f"Image/pose count mismatch: images={len(camera_image_files)} poses={len(poses)}"
            )
        if frustum_aspect <= 0:
            with profiler.track("probe_camera_image_aspect"):
                with Image.open(camera_image_files[0]) as img:
                    frustum_aspect = img.size[0] / max(img.size[1], 1)
        print(f"  Found {len(camera_image_files)} camera images in {camera_image_dir}")
    if frustum_aspect <= 0:
        frustum_aspect = width / max(height, 1)
    requested_frustum_history_stride = int(frustum_history_stride)
    history_fade_window = max(int(history_fade_window), 0)
    history_min_opacity = float(np.clip(history_min_opacity, 0.0, 1.0))
    if history_fade_mode not in {"forward", "reverse"}:
        raise ValueError(f"Unknown history_fade_mode: {history_fade_mode}")
    if history_fade_window > 0:
        print(
            "  Camera history fade enabled: "
            f"mode={history_fade_mode}, window={history_fade_window}, "
            f"min_opacity={history_min_opacity:.2f}"
        )
    camera_image_cull_to_view = bool(camera_image_cull_to_view)
    camera_image_cache_limit = max(int(camera_image_cache_limit), 0)
    camera_image_require_front_facing = bool(camera_image_require_front_facing)
    camera_image_min_projected_area = float(max(camera_image_min_projected_area, 0.0))
    save_rgba = bool(save_rgba)
    if camera_image_history_scope not in {"recent", "all"}:
        raise ValueError(f"Unknown camera_image_history_scope: {camera_image_history_scope}")
    if requested_frustum_history_stride <= 0:
        auto_history_span = max(int(frustum_history), 0) if frustum_history > 0 else len(poses)
        if camera_image_history_scope == "all":
            auto_target_visible_history = 3000
            frustum_history_stride = max(
                1, int(np.ceil(auto_history_span / max(auto_target_visible_history, 1)))
            )
        else:
            frustum_history_stride = 1
        print(
            "  Auto camera-history stride enabled: "
            f"requested={requested_frustum_history_stride}, resolved={frustum_history_stride}, "
            f"history_span={auto_history_span}, scope={camera_image_history_scope}"
        )
    else:
        frustum_history_stride = requested_frustum_history_stride
    if render_camera_images and camera_image_files:
        cache_desc = "unbounded" if camera_image_cache_limit <= 0 else str(camera_image_cache_limit)
        print(
            "  Camera image quad handling: "
            f"scope={camera_image_history_scope}, "
            f"cull_to_view={int(camera_image_cull_to_view)}, "
            f"front_facing={int(camera_image_require_front_facing)}, "
            f"min_area={camera_image_min_projected_area:.1f}, "
            f"cache_limit={cache_desc}"
        )

    # Setup renderer
    print("Setting up offscreen renderer...")
    with profiler.track("setup_renderer"):
        renderer = rendering.OffscreenRenderer(width, height)
        configure_offscreen_renderer(renderer, rendering, bg_color)
        alpha_renderer = None
        if save_rgba:
            alpha_renderer = rendering.OffscreenRenderer(width, height)
            configure_offscreen_renderer(alpha_renderer, rendering, "#000000")
    bg_rgb = np.array(hex_to_float_rgb(bg_color), dtype=np.float64)
    alpha_bg_rgb = np.zeros(3, dtype=np.float64)

    point_material_cache: dict[float, object] = {}
    camera_image_mesh_cache: OrderedDict[int, object] = OrderedDict()
    camera_image_texture_cache: OrderedDict[int, np.ndarray] = OrderedDict()
    camera_image_material_cache: OrderedDict[tuple[int, float], object] = OrderedDict()
    camera_image_alpha_mesh_cache: OrderedDict[int, object] = OrderedDict()
    camera_image_alpha_texture_cache: OrderedDict[int, np.ndarray] = OrderedDict()
    camera_image_alpha_material_cache: OrderedDict[tuple[int, float], object] = OrderedDict()
    camera_image_quad_cache: dict[int, np.ndarray] = {}
    colors_float = rainbow_colormap(np.linspace(0, 1, len(poses), endpoint=False))

    def get_point_material(opacity: float):
        key = round(float(opacity), 4)
        if key not in point_material_cache:
            point_material_cache[key] = build_point_material(rendering, point_size, key)
        return point_material_cache[key]

    image_geometry_added: set[int] = set()
    image_geometry_visible: dict[int, bool] = {}
    image_geometry_material_key: dict[int, tuple[int, float]] = {}
    image_geometry_alpha_material_key: dict[int, tuple[int, float]] = {}
    dynamic_geometry_names: list[str] = []
    older_history_chunk_geometry_names: dict[int, str] = {}
    older_history_chunk_xyz_parts: dict[int, list[np.ndarray]] = {}
    older_history_chunk_rgb_parts: dict[int, list[np.ndarray]] = {}
    older_history_chunk_alpha_rgb_parts: dict[int, list[np.ndarray]] = {}
    older_history_indices_cached: list[int] = []
    incremental_older_history_chunk_size = 64
    incremental_older_history = (
        history_fade_window > 0
        and frustum_history == 0
        and frustum_history_stride > 0
        and abs(history_size_step) < 1e-8
    )
    if incremental_older_history:
        print(
            "  Incremental older-frustum caching enabled "
            f"(chunk_size={incremental_older_history_chunk_size})"
        )

    def remove_geometry(geom_name: str):
        if renderer.scene.has_geometry(geom_name):
            renderer.scene.remove_geometry(geom_name)
        if alpha_renderer is not None and alpha_renderer.scene.has_geometry(geom_name):
            alpha_renderer.scene.remove_geometry(geom_name)

    def show_geometry(geom_name: str, visible: bool):
        if renderer.scene.has_geometry(geom_name):
            renderer.scene.show_geometry(geom_name, visible)
        if alpha_renderer is not None and alpha_renderer.scene.has_geometry(geom_name):
            alpha_renderer.scene.show_geometry(geom_name, visible)

    def get_camera_image_quad_points(vis_idx: int) -> np.ndarray:
        if vis_idx not in camera_image_quad_cache:
            plane = frustum_points(poses[vis_idx], size=frustum_size, aspect=frustum_aspect)[1:5]
            center = plane.mean(axis=0, keepdims=True)
            camera_image_quad_cache[vis_idx] = np.concatenate([plane, center], axis=0)
        return camera_image_quad_cache[vis_idx]

    def clear_incremental_older_history():
        for geom_name in list(older_history_chunk_geometry_names.values()):
            remove_geometry(geom_name)
        older_history_chunk_geometry_names.clear()
        older_history_chunk_xyz_parts.clear()
        older_history_chunk_rgb_parts.clear()
        older_history_chunk_alpha_rgb_parts.clear()
        older_history_indices_cached.clear()

    def rebuild_incremental_older_history_chunk(chunk_idx: int):
        geom_name = f"frustum_older_hist_chunk_{chunk_idx:06d}"
        remove_geometry(geom_name)

        xyz = np.concatenate(older_history_chunk_xyz_parts[chunk_idx], axis=0)
        rgb = np.concatenate(older_history_chunk_rgb_parts[chunk_idx], axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        renderer.scene.add_geometry(geom_name, pcd, get_point_material(history_min_opacity))
        if alpha_renderer is not None and chunk_idx in older_history_chunk_alpha_rgb_parts:
            alpha_rgb = np.concatenate(older_history_chunk_alpha_rgb_parts[chunk_idx], axis=0)
            alpha_pcd = o3d.geometry.PointCloud()
            alpha_pcd.points = o3d.utility.Vector3dVector(xyz)
            alpha_pcd.colors = o3d.utility.Vector3dVector(alpha_rgb)
            alpha_renderer.scene.add_geometry(
                geom_name, alpha_pcd, get_point_material(history_min_opacity)
            )
        older_history_chunk_geometry_names[chunk_idx] = geom_name

    def add_incremental_older_history(vis_idx: int, prev_idx: int | None, add_pos: int):
        frustum_color = blend_rgb_to_bg(colors_float[vis_idx], history_min_opacity, bg_rgb)
        frustum_pcd = build_single_frustum_pointcloud(
            poses[vis_idx],
            frustum_color,
            frustum_size=frustum_size,
            point_samples=point_samples,
            aspect=frustum_aspect,
            size_scale=1.0,
        )
        xyz_parts = [np.asarray(frustum_pcd.points)]
        rgb_parts = [np.asarray(frustum_pcd.colors)]
        alpha_rgb_parts = []
        if alpha_renderer is not None:
            alpha_rgb_parts.append(np.ones_like(rgb_parts[0]) * history_min_opacity)

        if prev_idx is not None:
            prev_color = blend_rgb_to_bg(colors_float[prev_idx], history_min_opacity, bg_rgb)
            seg_pcd = build_trajectory_segment_pointcloud(
                poses[prev_idx],
                poses[vis_idx],
                prev_color,
                frustum_color,
            )
            xyz_parts.append(np.asarray(seg_pcd.points))
            rgb_parts.append(np.asarray(seg_pcd.colors))
            if alpha_renderer is not None:
                alpha_rgb_parts.append(np.ones_like(rgb_parts[-1]) * history_min_opacity)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(xyz_parts, axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(rgb_parts, axis=0))

        chunk_idx = add_pos // incremental_older_history_chunk_size
        if chunk_idx not in older_history_chunk_xyz_parts:
            older_history_chunk_xyz_parts[chunk_idx] = []
            older_history_chunk_rgb_parts[chunk_idx] = []
            older_history_chunk_alpha_rgb_parts[chunk_idx] = []
        older_history_chunk_xyz_parts[chunk_idx].append(np.asarray(pcd.points))
        older_history_chunk_rgb_parts[chunk_idx].append(np.asarray(pcd.colors))
        if alpha_renderer is not None:
            older_history_chunk_alpha_rgb_parts[chunk_idx].append(
                np.concatenate(alpha_rgb_parts, axis=0)
            )
        rebuild_incremental_older_history_chunk(chunk_idx)

    def evict_camera_image_index(vis_idx: int):
        geom_name = f"cam_image_{vis_idx:06d}"
        remove_geometry(geom_name)
        image_geometry_added.discard(vis_idx)
        image_geometry_visible.pop(vis_idx, None)
        image_geometry_material_key.pop(vis_idx, None)
        image_geometry_alpha_material_key.pop(vis_idx, None)
        camera_image_mesh_cache.pop(vis_idx, None)
        camera_image_texture_cache.pop(vis_idx, None)
        camera_image_alpha_mesh_cache.pop(vis_idx, None)
        camera_image_alpha_texture_cache.pop(vis_idx, None)
        camera_image_quad_cache.pop(vis_idx, None)

        stale_material_keys = [key for key in camera_image_material_cache if key[0] == vis_idx]
        for key in stale_material_keys:
            camera_image_material_cache.pop(key, None)
        stale_alpha_material_keys = [
            key for key in camera_image_alpha_material_cache if key[0] == vis_idx
        ]
        for key in stale_alpha_material_keys:
            camera_image_alpha_material_cache.pop(key, None)

    def touch_camera_image_cache(vis_idx: int):
        if vis_idx in camera_image_texture_cache:
            camera_image_texture_cache.move_to_end(vis_idx)
        if vis_idx in camera_image_mesh_cache:
            camera_image_mesh_cache.move_to_end(vis_idx)
        if vis_idx in camera_image_alpha_texture_cache:
            camera_image_alpha_texture_cache.move_to_end(vis_idx)
        if vis_idx in camera_image_alpha_mesh_cache:
            camera_image_alpha_mesh_cache.move_to_end(vis_idx)

    def enforce_camera_image_cache_limit(protected_indices: set[int]):
        if camera_image_cache_limit <= 0:
            return
        while len(camera_image_texture_cache) > camera_image_cache_limit:
            evict_candidate = None
            for vis_idx in camera_image_texture_cache.keys():
                if vis_idx not in protected_indices:
                    evict_candidate = vis_idx
                    break
            if evict_candidate is None:
                break
            evict_camera_image_index(evict_candidate)

    def get_camera_image_material(vis_idx: int, opacity: float):
        material_key = (vis_idx, round(float(opacity), 4))
        if material_key not in camera_image_material_cache:
            material = build_camera_image_quad(
                poses[vis_idx],
                aspect=frustum_aspect,
                frustum_size=frustum_size,
                image_path=None,
                max_texture_dim=camera_image_max_dim,
                opacity=opacity,
                texture_array=camera_image_texture_cache[vis_idx],
                bg_rgb=bg_rgb,
            )[1]
            camera_image_material_cache[material_key] = material
        camera_image_material_cache.move_to_end(material_key)
        return material_key, camera_image_material_cache[material_key]

    def get_camera_image_alpha_material(vis_idx: int, opacity: float):
        material_key = (vis_idx, round(float(opacity), 4))
        if material_key not in camera_image_alpha_material_cache:
            material = build_camera_image_quad(
                poses[vis_idx],
                aspect=frustum_aspect,
                frustum_size=frustum_size,
                image_path=None,
                max_texture_dim=camera_image_max_dim,
                opacity=opacity,
                texture_array=camera_image_alpha_texture_cache[vis_idx],
                bg_rgb=alpha_bg_rgb,
            )[1]
            camera_image_alpha_material_cache[material_key] = material
        camera_image_alpha_material_cache.move_to_end(material_key)
        return material_key, camera_image_alpha_material_cache[material_key]

    # Render frames (camera frustums only)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Rendering {len(viewpoints)} camera frames...")
    save_from_frame = max(int(save_from_frame), 0)
    start_from_frame = max(int(start_from_frame), 0)
    end_at_frame = int(end_at_frame)
    if save_from_frame > 0:
        print(f"  Save only from frame index {save_from_frame}")
    if start_from_frame > 0:
        print(f"  Start rendering from frame index {start_from_frame}")
    if end_at_frame >= 0:
        print(f"  Stop rendering before frame index {end_at_frame}")

    frame_start = min(start_from_frame, len(viewpoints))
    if end_at_frame >= 0:
        frame_stop = min(max(end_at_frame, 0), len(viewpoints))
    else:
        frame_stop = len(viewpoints)
    frame_stop = max(frame_stop, frame_start)
    n_render_frames = max(frame_stop - frame_start, 0)
    n_saved_frames = max(frame_stop - max(save_from_frame, frame_start), 0)

    for frame_i in tqdm(range(frame_start, frame_stop)):
        vp = viewpoints[frame_i]
        with profiler.track("clear_dynamic_geometry"):
            for geom_name in dynamic_geometry_names:
                remove_geometry(geom_name)
        dynamic_geometry_names = []

        pose_idx = vp["pose_idx"]
        with profiler.track("prepare_history_indices"):
            if frustum_history > 0:
                start_idx = max(0, pose_idx - frustum_history + 1)
            else:
                start_idx = 0
            visible_indices = list(range(start_idx, pose_idx + 1, frustum_history_stride))
            if not visible_indices or visible_indices[-1] != pose_idx:
                visible_indices.append(pose_idx)
            order_lookup = {vis_idx: order for order, vis_idx in enumerate(visible_indices)}
            n_visible = len(visible_indices)
            size_scales = {
                vis_idx: 1.0 + history_size_step * max(n_visible - order_lookup[vis_idx] - 1, 0)
                for vis_idx in visible_indices
            }
            wire_opacity_map = {}
            image_opacity_map = {}
            for vis_idx in visible_indices:
                age = pose_idx - vis_idx
                if age == 0:
                    opacity = 1.0
                elif history_fade_window > 0 and age <= history_fade_window:
                    opacity = compute_history_opacity(
                        age,
                        history_fade_window,
                        history_min_opacity,
                        mode=history_fade_mode,
                    )
                else:
                    opacity = history_min_opacity
                wire_opacity_map[vis_idx] = opacity
                image_opacity_map[vis_idx] = opacity

            if history_fade_window > 0:
                older_indices = [
                    vis_idx
                    for vis_idx in visible_indices
                    if pose_idx - vis_idx > history_fade_window
                ]
                recent_indices = [
                    vis_idx
                    for vis_idx in visible_indices
                    if pose_idx - vis_idx <= history_fade_window
                ]
            else:
                older_indices = [vis_idx for vis_idx in visible_indices if vis_idx != pose_idx]
                recent_indices = [pose_idx] if pose_idx in order_lookup else []

        with profiler.track("older_wireframes"):
            if incremental_older_history:
                cached_prefix = older_indices[: len(older_history_indices_cached)]
                if older_history_indices_cached != cached_prefix:
                    clear_incremental_older_history()

                for add_pos in range(len(older_history_indices_cached), len(older_indices)):
                    vis_idx = older_indices[add_pos]
                    prev_idx = older_indices[add_pos - 1] if add_pos > 0 else None
                    add_incremental_older_history(vis_idx, prev_idx, add_pos)
                    older_history_indices_cached.append(vis_idx)
            else:
                if older_history_indices_cached:
                    clear_incremental_older_history()

                cam_pcd = build_frustum_pointcloud(
                    poses,
                    older_indices,
                    frustum_size,
                    point_samples,
                    history_size_step,
                    aspect=frustum_aspect,
                    size_scales=size_scales,
                )
                if cam_pcd is not None:
                    older_opacity = history_min_opacity
                    if older_opacity < 0.999:
                        older_colors = np.asarray(cam_pcd.colors)
                        cam_pcd.colors = o3d.utility.Vector3dVector(
                            blend_rgb_to_bg(older_colors, older_opacity, bg_rgb)
                        )
                    geom_name = "frustums_older"
                    renderer.scene.add_geometry(
                        geom_name, cam_pcd, get_point_material(older_opacity)
                    )
                    if alpha_renderer is not None:
                        alpha_pcd = build_frustum_pointcloud(
                            poses,
                            older_indices,
                            frustum_size,
                            point_samples,
                            history_size_step,
                            aspect=frustum_aspect,
                            size_scales=size_scales,
                        )
                        if alpha_pcd is not None:
                            alpha_colors = (
                                np.ones_like(np.asarray(alpha_pcd.colors)) * older_opacity
                            )
                            alpha_pcd.colors = o3d.utility.Vector3dVector(alpha_colors)
                            alpha_renderer.scene.add_geometry(
                                geom_name, alpha_pcd, get_point_material(older_opacity)
                            )
                    dynamic_geometry_names.append(geom_name)

        with profiler.track("recent_wireframes"):
            for vis_idx in recent_indices:
                recent_pcd = build_single_frustum_pointcloud(
                    poses[vis_idx],
                    blend_rgb_to_bg(colors_float[vis_idx], wire_opacity_map[vis_idx], bg_rgb),
                    frustum_size=frustum_size,
                    point_samples=point_samples,
                    aspect=frustum_aspect,
                    size_scale=size_scales[vis_idx],
                )
                geom_name = f"frustum_recent_{vis_idx:06d}"
                renderer.scene.add_geometry(
                    geom_name, recent_pcd, get_point_material(wire_opacity_map[vis_idx])
                )
                if alpha_renderer is not None:
                    alpha_recent_pcd = build_single_frustum_pointcloud(
                        poses[vis_idx],
                        make_alpha_rgb(wire_opacity_map[vis_idx]),
                        frustum_size=frustum_size,
                        point_samples=point_samples,
                        aspect=frustum_aspect,
                        size_scale=size_scales[vis_idx],
                    )
                    alpha_renderer.scene.add_geometry(
                        geom_name, alpha_recent_pcd, get_point_material(wire_opacity_map[vis_idx])
                    )
                dynamic_geometry_names.append(geom_name)

        with profiler.track("trajectory_segments"):
            if visible_indices:
                segment_start = max(len(older_indices) - 1, 0) if history_fade_window > 0 else 0
                for seg_order in range(segment_start, max(len(visible_indices) - 1, 0)):
                    idx0 = visible_indices[seg_order]
                    idx1 = visible_indices[seg_order + 1]
                    seg_pcd = build_trajectory_segment_pointcloud(
                        poses[idx0],
                        poses[idx1],
                        blend_rgb_to_bg(colors_float[idx0], wire_opacity_map[idx0], bg_rgb),
                        blend_rgb_to_bg(colors_float[idx1], wire_opacity_map[idx1], bg_rgb),
                    )
                    seg_opacity = 0.5 * (wire_opacity_map[idx0] + wire_opacity_map[idx1])
                    geom_name = f"traj_segment_{idx0:06d}_{idx1:06d}"
                    renderer.scene.add_geometry(geom_name, seg_pcd, get_point_material(seg_opacity))
                    if alpha_renderer is not None:
                        alpha_seg_pcd = build_trajectory_segment_pointcloud(
                            poses[idx0],
                            poses[idx1],
                            make_alpha_rgb(wire_opacity_map[idx0]),
                            make_alpha_rgb(wire_opacity_map[idx1]),
                        )
                        alpha_renderer.scene.add_geometry(
                            geom_name, alpha_seg_pcd, get_point_material(seg_opacity)
                        )
                    dynamic_geometry_names.append(geom_name)

        with profiler.track("prepare_render_camera"):
            if use_extrinsic_camera:
                render_extrinsic = np.array(vp["extrinsic"], dtype=np.float64)
            else:
                eye = np.array(vp["eye"], dtype=np.float64)
                lookat = np.array(vp["lookat"], dtype=np.float64)
                up = np.array(vp["up"], dtype=np.float64)
                render_extrinsic = np.linalg.inv(build_lookat_pose(eye, lookat, up))

        if render_camera_images and camera_image_files:
            if camera_image_history_scope == "all":
                image_candidate_indices = visible_indices
            else:
                image_candidate_indices = recent_indices
            image_visible_indices = image_candidate_indices
            with profiler.track("camera_images_cull"):
                if camera_image_cull_to_view:
                    image_visible_indices = [
                        vis_idx
                        for vis_idx in image_candidate_indices
                        if quad_visible_in_camera(
                            get_camera_image_quad_points(vis_idx),
                            render_extrinsic,
                            intrinsic,
                            width,
                            height,
                            require_front_facing=camera_image_require_front_facing,
                            min_projected_area=camera_image_min_projected_area,
                        )
                    ]

            candidate_set = set(image_candidate_indices)
            visible_set = set(image_visible_indices)
            cache_protected_set = (
                candidate_set if camera_image_history_scope == "recent" else visible_set
            )
            for vis_idx in image_visible_indices:
                if vis_idx not in camera_image_texture_cache:
                    with profiler.track("camera_images_texture_load"):
                        camera_image_texture_cache[vis_idx] = load_texture_array(
                            camera_image_files[vis_idx], camera_image_max_dim
                        )
                        if alpha_renderer is not None:
                            camera_image_alpha_texture_cache[vis_idx] = np.full_like(
                                camera_image_texture_cache[vis_idx], 255
                            )
                else:
                    camera_image_texture_cache.move_to_end(vis_idx)
                    if alpha_renderer is not None:
                        camera_image_alpha_texture_cache.move_to_end(vis_idx)
                if vis_idx not in camera_image_mesh_cache:
                    with profiler.track("camera_images_mesh_build"):
                        mesh, _ = build_camera_image_quad(
                            poses[vis_idx],
                            aspect=frustum_aspect,
                            frustum_size=frustum_size,
                            image_path=None,
                            max_texture_dim=camera_image_max_dim,
                            texture_array=camera_image_texture_cache[vis_idx],
                            bg_rgb=bg_rgb,
                        )
                        alpha_mesh = None
                        if alpha_renderer is not None:
                            alpha_mesh, _ = build_camera_image_quad(
                                poses[vis_idx],
                                aspect=frustum_aspect,
                                frustum_size=frustum_size,
                                image_path=None,
                                max_texture_dim=camera_image_max_dim,
                                texture_array=camera_image_alpha_texture_cache[vis_idx],
                                bg_rgb=alpha_bg_rgb,
                            )
                    camera_image_mesh_cache[vis_idx] = mesh
                    if alpha_renderer is not None and alpha_mesh is not None:
                        camera_image_alpha_mesh_cache[vis_idx] = alpha_mesh
                else:
                    camera_image_mesh_cache.move_to_end(vis_idx)
                    if alpha_renderer is not None:
                        camera_image_alpha_mesh_cache.move_to_end(vis_idx)
                with profiler.track("camera_images_material"):
                    material_key, material = get_camera_image_material(
                        vis_idx, image_opacity_map[vis_idx]
                    )
                    alpha_material_key = None
                    alpha_material = None
                    if alpha_renderer is not None:
                        alpha_material_key, alpha_material = get_camera_image_alpha_material(
                            vis_idx, image_opacity_map[vis_idx]
                        )
                mesh = camera_image_mesh_cache[vis_idx]
                geom_name = f"cam_image_{vis_idx:06d}"
                with profiler.track("camera_images_scene_update"):
                    if vis_idx not in image_geometry_added:
                        renderer.scene.add_geometry(geom_name, mesh, material)
                        if alpha_renderer is not None:
                            alpha_renderer.scene.add_geometry(
                                geom_name, camera_image_alpha_mesh_cache[vis_idx], alpha_material
                            )
                        image_geometry_added.add(vis_idx)
                        image_geometry_visible[vis_idx] = True
                        image_geometry_material_key[vis_idx] = material_key
                        if alpha_renderer is not None and alpha_material_key is not None:
                            image_geometry_alpha_material_key[vis_idx] = alpha_material_key
                    else:
                        if not image_geometry_visible.get(vis_idx, False):
                            show_geometry(geom_name, True)
                            image_geometry_visible[vis_idx] = True
                        if image_geometry_material_key.get(vis_idx) != material_key:
                            renderer.scene.modify_geometry_material(geom_name, material)
                            image_geometry_material_key[vis_idx] = material_key
                        if (
                            alpha_renderer is not None
                            and alpha_material_key is not None
                            and image_geometry_alpha_material_key.get(vis_idx) != alpha_material_key
                        ):
                            alpha_renderer.scene.modify_geometry_material(geom_name, alpha_material)
                            image_geometry_alpha_material_key[vis_idx] = alpha_material_key
                touch_camera_image_cache(vis_idx)

            with profiler.track("camera_images_cleanup"):
                for vis_idx in list(image_geometry_added):
                    if vis_idx in visible_set:
                        continue
                    if vis_idx not in candidate_set:
                        evict_camera_image_index(vis_idx)
                        continue
                    if image_geometry_visible.get(vis_idx, False):
                        show_geometry(f"cam_image_{vis_idx:06d}", False)
                        image_geometry_visible[vis_idx] = False

                enforce_camera_image_cache_limit(cache_protected_set)

        # Use exact same camera as render_points.py
        with profiler.track("setup_camera"):
            if use_extrinsic_camera:
                renderer.setup_camera(o3d_intrinsic, render_extrinsic)
                if alpha_renderer is not None:
                    alpha_renderer.setup_camera(o3d_intrinsic, render_extrinsic)
            else:
                eye = np.array(vp["eye"])
                lookat = np.array(vp["lookat"])
                up = np.array(vp["up"])
                renderer.setup_camera(fov, lookat, eye, up)
                if alpha_renderer is not None:
                    alpha_renderer.setup_camera(fov, lookat, eye, up)

        with profiler.track("render_to_image"):
            img = renderer.render_to_image()
            alpha_img = alpha_renderer.render_to_image() if alpha_renderer is not None else None
        if frame_i >= save_from_frame:
            out_path = os.path.join(output_dir, f"frame_{frame_i:05d}.png")
            with profiler.track("write_image"):
                if alpha_img is None:
                    o3d.io.write_image(out_path, img)
                else:
                    write_rgba_image(img, alpha_img, out_path)

    print(f"Done! {n_saved_frames} frames saved to {output_dir}")
    if profiler.enabled:
        profiler.save(profile_json, n_render_frames, n_saved_frames)
        if profile_print:
            profiler.print(n_render_frames, n_saved_frames)
    if force_clean_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="Render camera frustums only (no point cloud)")
    parser.add_argument("--pose_path", required=True)
    parser.add_argument("--pred_pose_path", default="")
    parser.add_argument("--align_gt_to_whole", type=int, default=0)
    parser.add_argument("--normalize_first", type=int, default=-1)
    parser.add_argument(
        "--pose_in_xyz_frame",
        type=int,
        default=-1,
        help="Whether pose_path is already in xyz/whole coordinates: -1 auto, 0 no, 1 yes",
    )
    parser.add_argument(
        "--viewpoint_config", required=True, help="viewpoint_config.json from render_points.py"
    )
    parser.add_argument(
        "--viewpoint_interp",
        type=int,
        default=1,
        help="Optional render-viewpoint interpolation factor for legacy viewpoint configs",
    )
    parser.add_argument("--output_dir", default="data/output/kitti/07/progressive/CAM2D")
    parser.add_argument("--bg_color", default="#ffffff")
    parser.add_argument("--frustum_size", type=float, default=0.15)
    parser.add_argument(
        "--frustum_history",
        type=int,
        default=0,
        help="Number of raw poses to keep; 0 means from the first pose to current pose",
    )
    parser.add_argument(
        "--frustum_history_stride",
        type=int,
        default=1,
        help="Subsample stride over raw poses when drawing camera history; <=0 enables auto mode",
    )
    parser.add_argument("--history_size_step", type=float, default=0.0)
    parser.add_argument("--point_size", type=float, default=3.5)
    parser.add_argument("--point_samples", type=int, default=128)
    parser.add_argument(
        "--frustum_aspect",
        type=float,
        default=-1.0,
        help="Frustum near-plane aspect ratio; <=0 defaults to viewpoint width/height",
    )
    parser.add_argument("--camera_image_dir", default="")
    parser.add_argument("--camera_image_max_dim", type=int, default=256)
    parser.add_argument("--render_camera_images", type=int, default=0)
    parser.add_argument(
        "--camera_image_history_scope",
        choices=["recent", "all"],
        default="all",
        help="Which history cameras can render original image quads: only recent fade-window frames or all visible history frames",
    )
    parser.add_argument(
        "--history_fade_window",
        type=int,
        default=0,
        help="Fade camera opacity linearly from current frame back this many frames; <=0 disables",
    )
    parser.add_argument(
        "--history_min_opacity",
        type=float,
        default=0.2,
        help="Minimum opacity for history cameras once they reach the fade window",
    )
    parser.add_argument(
        "--history_fade_mode",
        choices=["forward", "reverse"],
        default="forward",
        help="Opacity direction over the history window: current->oldest or reversed",
    )
    parser.add_argument(
        "--save_from_frame",
        type=int,
        default=0,
        help="Render all prior frames to rebuild progressive state, but only save frames >= this index",
    )
    parser.add_argument(
        "--start_from_frame",
        type=int,
        default=0,
        help="Start the render loop directly from this viewpoint index; camera-only output does not require earlier frames",
    )
    parser.add_argument(
        "--end_at_frame",
        type=int,
        default=-1,
        help="Stop rendering before this viewpoint index; <0 keeps rendering until the end",
    )
    parser.add_argument(
        "--camera_image_cull_to_view",
        type=int,
        default=1,
        help="Only render camera image quads that intersect the current render view",
    )
    parser.add_argument(
        "--camera_image_cache_limit",
        type=int,
        default=128,
        help="Maximum number of camera image quads/textures to keep cached; <=0 disables the limit",
    )
    parser.add_argument(
        "--camera_image_require_front_facing",
        type=int,
        default=0,
        help="Cull camera image quads whose front face is turned away from the render camera",
    )
    parser.add_argument(
        "--camera_image_min_projected_area",
        type=float,
        default=0.0,
        help="Cull camera image quads whose on-screen bbox area is smaller than this many pixels",
    )
    parser.add_argument(
        "--save_rgba",
        type=int,
        default=1,
        help="Save camera renders as RGBA PNGs using a matte pass for the alpha channel",
    )
    parser.add_argument(
        "--profile_json", default="", help="Optional JSON path for stage-level timing summary"
    )
    parser.add_argument(
        "--profile_print", type=int, default=0, help="Print stage-level timing summary at the end"
    )
    parser.add_argument(
        "--force_clean_exit",
        type=int,
        default=1,
        help="Exit the process immediately after successful rendering to avoid Open3D teardown crashes",
    )
    args = parser.parse_args()

    render_camera_frames(**vars(args))


if __name__ == "__main__":
    main()
