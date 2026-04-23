#!/usr/bin/env python3
"""Interactive Open3D viewer and camera-path editor for Scal3R point clouds.

Typical usage
-------------
python scripts/visualize/geometry/viewer/open3d_viewer.py \
    --ply_path data/datasets/kitti/07/xyz.ply \
    --pose_path data/datasets/kitti/07/mat_xyz.txt \
    --viewpoint_config data/output/kitti/07/viewpoint_config.json \
    --output_dir data/datasets/kitti/07/custom

Hotkeys
-------
H : print help
K : capture the current viewer pose as a keyframe
R : remove the last keyframe
C : clear all keyframes
P : preview the interpolated camera path
J : jump to previous captured keyframe
L : jump to next captured keyframe
O : export the interpolated path
"""

import os
import re
import sys
import time
import json
import argparse
import numpy as np
from dataclasses import dataclass
from os.path import abspath, dirname, join
from scipy.spatial.transform import Rotation, Slerp

visualize_root = dirname(dirname(dirname(abspath(__file__))))
release_root = dirname(dirname(visualize_root))
if visualize_root not in sys.path: sys.path.insert(0, visualize_root)
if release_root not in sys.path: sys.path.insert(0, release_root)

from utils.color_utils import rainbow_colormap  # noqa: E402
from utils.point_utils import load_pointcloud_arrays, random_downsample, voxel_downsample  # noqa: E402
from utils.camera_utils import FRUSTUM_EDGES, prepare_poses, make_intrinsic, frustum_points, opencv_c2w_to_o3d_extrinsic  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply_path", required=True)
    parser.add_argument("--pose_path", default="")
    parser.add_argument("--pred_pose_path", default="")
    parser.add_argument("--align_gt_to_whole", type=int, default=0)
    parser.add_argument("--normalize_first", type=int, default=-1)
    parser.add_argument(
        "--pose_in_xyz_frame",
        type=int,
        default=-1,
        help="Whether pose_path is already in xyz/whole coordinates: -1 auto, 0 no, 1 yes",
    )
    parser.add_argument("--viewpoint_config", default="")
    parser.add_argument("--intri_path", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--save_tag",
        default="custom",
        help="Default export folder/tag when output_dir/export_prefix are not explicitly set",
    )
    parser.add_argument("--window_width", type=int, default=-1)
    parser.add_argument("--window_height", type=int, default=-1)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--voxel_size", type=float, default=0.0)
    parser.add_argument(
        "--random_downsample_max_points",
        type=int,
        default=600000,
        help="Randomly cap the viewer point cloud to this many points; <=0 disables it",
    )
    parser.add_argument(
        "--random_downsample_seed",
        type=int,
        default=0,
        help="Seed used by the random viewer downsample",
    )
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--bg_color", default="#ffffff")
    parser.add_argument("--show_capture_poses", type=int, default=1)
    parser.add_argument("--pose_stride", type=int, default=5)
    parser.add_argument("--frustum_size", type=float, default=0.6)
    parser.add_argument("--frames_per_segment", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--export_prefix",
        default="",
        help="Optional filename prefix override; defaults to save_tag",
    )
    parser.add_argument("--export_evc", type=int, default=1)
    parser.add_argument("--preview_sleep", type=float, default=-1.0)
    parser.add_argument(
        "--init_pose_index",
        type=int,
        default=-1,
        help="Initial input pose index; negative values count from the end, default -1 means last frame",
    )
    return parser.parse_args()


def parse_hex_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)


def _parse_opencv_matrix(text: str, key: str) -> np.ndarray | None:
    pattern = (
        rf"{re.escape(key)}:\s*!!opencv-matrix\s*"
        rf"rows:\s*(\d+)\s*"
        rf"cols:\s*(\d+)\s*"
        rf"dt:\s*\w+\s*"
        rf"data:\s*\[(.*?)\]"
    )
    match = re.search(pattern, text, flags=re.S)
    if not match:
        return None
    rows = int(match.group(1))
    cols = int(match.group(2))
    data = [float(x.strip()) for x in match.group(3).replace("\n", " ").split(",") if x.strip()]
    return np.array(data, dtype=np.float64).reshape(rows, cols)


def _parse_scalar(text: str, key: str) -> float | None:
    match = re.search(rf"^{re.escape(key)}:\s*([^\n]+)\s*$", text, flags=re.M)
    if not match:
        return None
    return float(match.group(1).strip())


def load_first_intrinsics(intri_path: str) -> dict | None:
    if not intri_path:
        return None
    with open(intri_path, "r") as f:
        text = f.read()
    first_k = re.search(r"K_([0-9A-Za-z_]+):\s*!!opencv-matrix", text)
    if not first_k:
        return None
    name = first_k.group(1)
    K = _parse_opencv_matrix(text, f"K_{name}")
    H = _parse_scalar(text, f"H_{name}")
    W = _parse_scalar(text, f"W_{name}")
    if K is None or H is None or W is None:
        return None
    return dict(name=name, K=K, H=int(round(H)), W=int(round(W)))


def load_intrinsics_from_viewpoint_config(viewpoint_config: str) -> dict | None:
    if not viewpoint_config:
        return None

    with open(viewpoint_config, "r") as f:
        cfg = json.load(f)
    width = int(cfg["width"])
    height = int(cfg["height"])

    if "intrinsic" in cfg:
        K = np.array(cfg["intrinsic"], dtype=np.float64)
        return dict(name="viewpoint_config", K=K, H=height, W=width)

    fov_y_deg = cfg.get("fov_y_deg")
    if fov_y_deg is None:
        return None

    K = make_intrinsic(float(fov_y_deg), width, height)
    return dict(name="viewpoint_fov", K=K, H=height, W=width)


def load_import_intrinsics(intri_path: str = "", viewpoint_config: str = "") -> dict | None:
    intrinsics = load_first_intrinsics(intri_path) if intri_path else None
    if intrinsics is not None:
        return intrinsics
    return load_intrinsics_from_viewpoint_config(viewpoint_config) if viewpoint_config else None


def _yaml_matrix_block(key: str, array: np.ndarray) -> list[str]:
    flat = ", ".join(f"{float(x):.10f}" for x in array.reshape(-1))
    rows, cols = array.shape
    return [
        f"{key}: !!opencv-matrix",
        f"  rows: {rows}",
        f"  cols: {cols}",
        "  dt: d",
        f"  data: [{flat}]",
    ]


def write_easyvolcap_camera_files(
    output_dir: str,
    poses_src: np.ndarray,
    intrinsics: list[np.ndarray],
    widths: list[int],
    heights: list[int],
    frame_values: list[float],
    near_values: list[float],
    far_values: list[float],
    prefix: str = "",
):
    os.makedirs(output_dir, exist_ok=True)
    intri_path = join(output_dir, "intri.yml")
    extri_path = join(output_dir, "extri.yml")

    names = [f"{prefix}{i:06d}" for i in range(len(poses_src))]

    intri_lines = ["%YAML:1.0", "---", "names:"]
    extri_lines = ["%YAML:1.0", "---", "names:"]
    for name in names:
        intri_lines.append(f'  - "{name}"')
        extri_lines.append(f'  - "{name}"')

    for name, pose, K, W, H, t_val, n_val, f_val in zip(
        names, poses_src, intrinsics, widths, heights, frame_values, near_values, far_values
    ):
        w2c = np.linalg.inv(pose)
        R = w2c[:3, :3]
        T = w2c[:3, 3:4]

        intri_lines.extend(_yaml_matrix_block(f"K_{name}", K))
        intri_lines.append(f"H_{name}: {float(H):.10f}")
        intri_lines.append(f"W_{name}: {float(W):.10f}")
        intri_lines.extend(_yaml_matrix_block(f"D_{name}", np.zeros((5, 1), dtype=np.float64)))

        extri_lines.extend(_yaml_matrix_block(f"Rot_{name}", R))
        extri_lines.extend(_yaml_matrix_block(f"T_{name}", T))
        extri_lines.append(f"t_{name}: {float(t_val):.10f}")
        extri_lines.append(f"n_{name}: {float(n_val):.10f}")
        extri_lines.append(f"f_{name}: {float(f_val):.10f}")

    with open(intri_path, "w") as f:
        f.write("\n".join(intri_lines) + "\n")
    with open(extri_path, "w") as f:
        f.write("\n".join(extri_lines) + "\n")


def intrinsic_to_numpy(intrinsic) -> tuple[np.ndarray, int, int]:
    width = int(intrinsic.width)
    height = int(intrinsic.height)
    return np.array(intrinsic.intrinsic_matrix, dtype=np.float64), width, height


def make_pinhole_parameters(width: int, height: int, K: np.ndarray, extrinsic: np.ndarray):
    import open3d as o3d

    params = o3d.camera.PinholeCameraParameters()
    params.extrinsic = np.array(extrinsic, dtype=np.float64)
    params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(width),
        int(height),
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
    )
    return params


def build_capture_lineset(
    poses: np.ndarray,
    frustum_size: float,
    stride: int,
):
    import open3d as o3d

    if len(poses) == 0:
        return None

    stride = max(int(stride), 1)
    keep = list(range(0, len(poses), stride))
    colors = rainbow_colormap(np.linspace(0.0, 1.0, len(poses), endpoint=False))

    points = []
    lines = []
    line_colors = []

    for pose_idx in keep:
        corners = frustum_points(poses[pose_idx], size=frustum_size, aspect=16.0 / 9.0)
        base = len(points)
        points.extend(corners.tolist())
        for src, dst in FRUSTUM_EDGES:
            lines.append([base + src, base + dst])
            line_colors.append(colors[pose_idx].tolist())

    centers = poses[:, :3, 3]
    traj_base = len(points)
    points.extend(centers.tolist())
    for idx in range(len(centers) - 1):
        lines.append([traj_base + idx, traj_base + idx + 1])
        line_colors.append([0.85, 0.85, 0.85])

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    lineset.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    lineset.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=np.float64))
    return lineset


def build_keyframe_lineset(keyframes: list["CameraKeyframe"], frustum_size: float):
    import open3d as o3d

    if not keyframes:
        return None

    points = []
    lines = []
    line_colors = []
    frustum_color = np.array([1.0, 0.6, 0.1], dtype=np.float64)
    path_color = np.array([1.0, 0.2, 0.2], dtype=np.float64)

    poses = [np.linalg.inv(kf.extrinsic) for kf in keyframes]
    for pose in poses:
        corners = frustum_points(pose, size=frustum_size, aspect=16.0 / 9.0)
        base = len(points)
        points.extend(corners.tolist())
        for src, dst in FRUSTUM_EDGES:
            lines.append([base + src, base + dst])
            line_colors.append(frustum_color.tolist())

    traj_base = len(points)
    centers = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float64)
    points.extend(centers.tolist())
    for idx in range(len(centers) - 1):
        lines.append([traj_base + idx, traj_base + idx + 1])
        line_colors.append(path_color.tolist())

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    lineset.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    lineset.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=np.float64))
    return lineset


@dataclass
class CameraKeyframe:
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    width: int
    height: int

    @property
    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.extrinsic)


class Open3DViewer:
    def __init__(self, args):
        import open3d as o3d

        self.o3d = o3d
        self.args = args
        self.window_width = args.window_width
        self.window_height = args.window_height
        self.bg_color = parse_hex_rgb(args.bg_color)
        self.preview_sleep = 1.0 / args.fps if args.preview_sleep < 0 else args.preview_sleep
        self.keyframes: list[CameraKeyframe] = []
        self.preview_index = -1
        self.output_dir = self._resolve_output_dir()
        self.capture_geometry = None
        self.keyframe_geometry = None
        self.default_intrinsics = load_import_intrinsics(
            intri_path=args.intri_path,
            viewpoint_config=args.viewpoint_config,
        )

        xyz, rgb = load_pointcloud_arrays(args.ply_path)
        self.original_point_count = len(xyz)
        if args.voxel_size > 0:
            xyz, rgb = voxel_downsample(xyz, rgb, args.voxel_size)
        self.after_voxel_point_count = len(xyz)
        xyz, rgb = random_downsample(
            xyz,
            rgb,
            max_points=args.random_downsample_max_points,
            seed=args.random_downsample_seed,
        )
        self.scene_point_count = len(xyz)
        self.scene_pcd = o3d.geometry.PointCloud()
        self.scene_pcd.points = o3d.utility.Vector3dVector(xyz)
        self.scene_pcd.colors = o3d.utility.Vector3dVector(rgb)

        self.capture_poses = None
        self.pose_meta = {}
        if args.pose_path:
            self.capture_poses, self.pose_meta = prepare_poses(
                args.pose_path,
                pred_pose_path=args.pred_pose_path,
                align_gt_to_whole=args.align_gt_to_whole,
                normalize_first=args.normalize_first,
                pose_in_xyz_frame=args.pose_in_xyz_frame,
            )
            if args.show_capture_poses:
                self.capture_geometry = build_capture_lineset(
                    self.capture_poses,
                    frustum_size=args.frustum_size,
                    stride=args.pose_stride,
                )

        if self.default_intrinsics is not None:
            self.window_width = self.default_intrinsics["W"] if self.window_width <= 0 else self.window_width
            self.window_height = (
                self.default_intrinsics["H"] if self.window_height <= 0 else self.window_height
            )
        else:
            self.window_width = 1280 if self.window_width <= 0 else self.window_width
            self.window_height = 720 if self.window_height <= 0 else self.window_height
            K = make_intrinsic(args.fov, self.window_width, self.window_height)
            self.default_intrinsics = dict(name="fov", K=K, W=self.window_width, H=self.window_height)

        self.vis = None

    def _resolve_output_dir(self) -> str:
        if self.args.output_dir:
            return abspath(self.args.output_dir)
        return join(dirname(abspath(self.args.ply_path)), self.args.save_tag)

    def _resolve_export_tag(self) -> str:
        export_tag = self.args.export_prefix if self.args.export_prefix else self.args.save_tag
        return str(export_tag)

    def print_help(self):
        print("\n[Open3D Viewer]")
        print("  Mouse: orbit / pan / zoom with Open3D default controls")
        print("  H: print this help")
        print("  K: capture current view as a keyframe")
        print("  R: remove the last keyframe")
        print("  C: clear all keyframes")
        print("  J/L: jump to previous / next keyframe")
        print("  P: preview interpolated path")
        print("  O: export interpolated path")
        print(
            f"  points={self.scene_point_count:,}/{self.original_point_count:,}, "
            f"point_size={self.args.point_size}"
        )
        print(f"  frames_per_segment={self.args.frames_per_segment}, fps={self.args.fps}")
        print(f"  output_dir={self.output_dir}")
        print("")

    def _resolve_initial_pose_index(self) -> int | None:
        if self.capture_poses is None or len(self.capture_poses) == 0:
            return None

        pose_idx = int(self.args.init_pose_index)
        n_poses = len(self.capture_poses)

        if pose_idx < 0:
            pose_idx = n_poses + pose_idx

        pose_idx = min(max(pose_idx, 0), n_poses - 1)
        return pose_idx

    def create_window(self):
        self.vis = self.o3d.visualization.VisualizerWithKeyCallback()
        ok = self.vis.create_window(
            window_name="Scal3R Open3D Viewer",
            width=int(self.window_width),
            height=int(self.window_height),
        )
        if not ok:
            raise RuntimeError(
                "Open3D failed to create a GLFW window. "
                "In WSL/remote GUI environments, try `LIBGL_ALWAYS_SOFTWARE=1` first."
            )

        self.vis.add_geometry(self.scene_pcd)
        if self.capture_geometry is not None:
            self.vis.add_geometry(self.capture_geometry, reset_bounding_box=False)

        render_opt = self.vis.get_render_option()
        if render_opt is None:
            raise RuntimeError("Open3D window was created but render options are unavailable")
        render_opt.background_color = self.bg_color
        render_opt.point_size = float(self.args.point_size)
        render_opt.line_width = 1.0

        self.vis.register_key_callback(ord("H"), self._on_help)
        self.vis.register_key_callback(ord("K"), self._on_capture)
        self.vis.register_key_callback(ord("R"), self._on_remove_last)
        self.vis.register_key_callback(ord("C"), self._on_clear)
        self.vis.register_key_callback(ord("P"), self._on_preview)
        self.vis.register_key_callback(ord("J"), self._on_prev_keyframe)
        self.vis.register_key_callback(ord("L"), self._on_next_keyframe)
        self.vis.register_key_callback(ord("O"), self._on_export)

        self.print_help()

        pose_idx = self._resolve_initial_pose_index()
        if pose_idx is not None:
            K = self.default_intrinsics["K"]
            W = self.default_intrinsics["W"]
            H = self.default_intrinsics["H"]
            extrinsic = opencv_c2w_to_o3d_extrinsic(self.capture_poses[pose_idx])
            self._apply_view(self.vis, CameraKeyframe(extrinsic=extrinsic, intrinsic=K, width=W, height=H))
            print(f"Initialized viewer from input pose #{pose_idx}")

    def run(self):
        self.create_window()
        self.vis.run()
        self.vis.destroy_window()

    def _current_keyframe(self, vis) -> CameraKeyframe:
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        K, width, height = intrinsic_to_numpy(params.intrinsic)
        return CameraKeyframe(
            extrinsic=np.array(params.extrinsic, dtype=np.float64),
            intrinsic=K,
            width=width,
            height=height,
        )

    def _apply_view(self, vis, keyframe: CameraKeyframe):
        params = make_pinhole_parameters(
            keyframe.width,
            keyframe.height,
            keyframe.intrinsic,
            keyframe.extrinsic,
        )
        vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()

    def _rebuild_keyframe_geometry(self, vis):
        if self.keyframe_geometry is not None:
            vis.remove_geometry(self.keyframe_geometry, reset_bounding_box=False)
            self.keyframe_geometry = None
        self.keyframe_geometry = build_keyframe_lineset(
            self.keyframes,
            frustum_size=self.args.frustum_size * 0.7,
        )
        if self.keyframe_geometry is not None:
            vis.add_geometry(self.keyframe_geometry, reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()

    def _on_help(self, vis):
        self.print_help()
        return False

    def _on_capture(self, vis):
        keyframe = self._current_keyframe(vis)
        self.keyframes.append(keyframe)
        self.preview_index = len(self.keyframes) - 1
        self._rebuild_keyframe_geometry(vis)
        center = keyframe.c2w[:3, 3]
        print(
            f"[capture] keyframe #{len(self.keyframes) - 1}: "
            f"center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
        )
        return False

    def _on_remove_last(self, vis):
        if not self.keyframes:
            print("[remove] no keyframe to remove")
            return False
        self.keyframes.pop()
        self.preview_index = min(self.preview_index, len(self.keyframes) - 1)
        self._rebuild_keyframe_geometry(vis)
        print(f"[remove] remaining keyframes: {len(self.keyframes)}")
        return False

    def _on_clear(self, vis):
        self.keyframes.clear()
        self.preview_index = -1
        self._rebuild_keyframe_geometry(vis)
        print("[clear] all keyframes cleared")
        return False

    def _jump_to_keyframe(self, vis, step: int):
        if not self.keyframes:
            print("[jump] no keyframes captured")
            return False
        self.preview_index = (self.preview_index + step) % len(self.keyframes)
        self._apply_view(vis, self.keyframes[self.preview_index])
        print(f"[jump] keyframe #{self.preview_index}")
        return False

    def _on_prev_keyframe(self, vis):
        return self._jump_to_keyframe(vis, -1)

    def _on_next_keyframe(self, vis):
        return self._jump_to_keyframe(vis, 1)

    def _interpolate_segment(self, start: CameraKeyframe, end: CameraKeyframe, n_frames: int, include_last: bool):
        if n_frames <= 0:
            return []

        c2w0 = start.c2w
        c2w1 = end.c2w
        times = np.array([0.0, 1.0], dtype=np.float64)
        slerp = Slerp(times, Rotation.from_matrix(np.stack([c2w0[:3, :3], c2w1[:3, :3]], axis=0)))

        if include_last:
            alphas = np.linspace(0.0, 1.0, n_frames, endpoint=True)
        else:
            alphas = np.linspace(0.0, 1.0, n_frames, endpoint=False)

        keyframes = []
        for alpha in alphas:
            rot = slerp([alpha]).as_matrix()[0]
            trans = (1.0 - alpha) * c2w0[:3, 3] + alpha * c2w1[:3, 3]

            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = rot
            c2w[:3, 3] = trans

            K = (1.0 - alpha) * start.intrinsic + alpha * end.intrinsic
            width = int(round((1.0 - alpha) * start.width + alpha * end.width))
            height = int(round((1.0 - alpha) * start.height + alpha * end.height))
            keyframes.append(
                CameraKeyframe(
                    extrinsic=np.linalg.inv(c2w),
                    intrinsic=K,
                    width=width,
                    height=height,
                )
            )
        return keyframes

    def build_interpolated_path(self) -> list[CameraKeyframe]:
        if len(self.keyframes) < 2:
            return list(self.keyframes)

        frames_per_segment = max(int(self.args.frames_per_segment), 2)
        frames = []
        for idx in range(len(self.keyframes) - 1):
            segment = self._interpolate_segment(
                self.keyframes[idx],
                self.keyframes[idx + 1],
                n_frames=frames_per_segment,
                include_last=(idx == len(self.keyframes) - 2),
            )
            frames.extend(segment)
        return frames

    def _on_preview(self, vis):
        path = self.build_interpolated_path()
        if len(path) < 2:
            print("[preview] need at least 2 keyframes")
            return False
        print(
            f"[preview] {len(path)} frames, "
            f"{(len(path) / max(self.args.fps, 1e-6)):.2f}s at {self.args.fps:.2f} fps"
        )
        for keyframe in path:
            self._apply_view(vis, keyframe)
            time.sleep(self.preview_sleep)
        return False

    def export_path(self):
        os.makedirs(self.output_dir, exist_ok=True)
        path = self.build_interpolated_path()
        if len(path) < 2:
            raise ValueError("Need at least 2 keyframes to export")

        poses = np.stack([kf.c2w for kf in path], axis=0)
        intrinsics = [kf.intrinsic for kf in path]
        widths = [kf.width for kf in path]
        heights = [kf.height for kf in path]
        first_k = intrinsics[0]
        first_w = widths[0]
        first_h = heights[0]
        export_tag = self._resolve_export_tag()

        pose_path = join(self.output_dir, f"mat_{export_tag}.txt")
        keyframe_path = join(self.output_dir, f"keyframes_{export_tag}.json")
        viewpoint_path = join(self.output_dir, f"viewpoint_{export_tag}.json")
        evc_dir = join(self.output_dir, f"camera_path_{export_tag}")

        np.savetxt(pose_path, poses.reshape(len(poses), -1), fmt="%.8f")

        raw_keyframes = []
        for idx, kf in enumerate(self.keyframes):
            raw_keyframes.append(
                dict(
                    index=idx,
                    width=int(kf.width),
                    height=int(kf.height),
                    intrinsic=kf.intrinsic.tolist(),
                    extrinsic=kf.extrinsic.tolist(),
                    c2w=kf.c2w.tolist(),
                )
            )
        with open(keyframe_path, "w") as f:
            f.write(json.dumps(raw_keyframes, indent=2))

        cfg = dict(
            source="open3d_viewer",
            width=int(first_w),
            height=int(first_h),
            intrinsic=first_k.tolist(),
            bg_color=self.args.bg_color,
            fps=float(self.args.fps),
            frames_per_segment=int(self.args.frames_per_segment),
            n_keyframes=len(self.keyframes),
            n_frames=len(path),
            view_mode="firstperson",
            frame_step=1,
            viewpoints=[
                dict(
                    pose_idx=int(i),
                    extrinsic=kf.extrinsic.tolist(),
                )
                for i, kf in enumerate(path)
            ],
        )
        with open(viewpoint_path, "w") as f:
            f.write(json.dumps(cfg, indent=2))

        if self.args.export_evc:
            denom = max(len(path) - 1, 1)
            write_easyvolcap_camera_files(
                str(evc_dir),
                poses_src=poses,
                intrinsics=intrinsics,
                widths=widths,
                heights=heights,
                frame_values=[i / denom for i in range(len(path))],
                near_values=[0.01] * len(path),
                far_values=[1e6] * len(path),
                prefix="",
            )

        return dict(
            pose_path=pose_path,
            keyframe_path=keyframe_path,
            viewpoint_path=viewpoint_path,
            evc_dir=evc_dir if self.args.export_evc else None,
            n_keyframes=len(self.keyframes),
            n_frames=len(path),
        )

    def _on_export(self, vis):
        try:
            result = self.export_path()
        except Exception as err:
            print(f"[export] failed: {err}")
            return False

        print(
            f"[export] keyframes={result['n_keyframes']}, frames={result['n_frames']}\n"
            f"  poses: {result['pose_path']}\n"
            f"  keyframes: {result['keyframe_path']}\n"
            f"  viewpoint: {result['viewpoint_path']}"
        )
        if result["evc_dir"] is not None:
            print(f"  easyvolcap: {result['evc_dir']}")
        return False


def main():
    args = parse_args()
    viewer = Open3DViewer(args)

    if viewer.pose_meta.get("loaded_xyz_pose_cache"):
        print(f"Reused cached xyz-frame poses: {viewer.pose_meta['xyz_pose_path']}")
    if viewer.pose_meta.get("saved_xyz_pose_cache"):
        print(f"Saved cached xyz-frame poses: {viewer.pose_meta['xyz_pose_path']}")
    if viewer.pose_meta.get("normalized_first"):
        print("Poses normalized by first frame")
    print(
        f"Point cloud for viewer: {viewer.scene_point_count:,}/{viewer.original_point_count:,} points"
    )
    if args.voxel_size > 0:
        print(
            f"  after voxel downsample: {viewer.after_voxel_point_count:,} "
            f"(voxel_size={args.voxel_size})"
        )
    if (
        args.random_downsample_max_points > 0
        and viewer.scene_point_count < viewer.after_voxel_point_count
    ):
        print(
            f"  after random cap: {viewer.scene_point_count:,} "
            f"(max_points={args.random_downsample_max_points}, seed={args.random_downsample_seed})"
        )
    if viewer.default_intrinsics is not None:
        K = viewer.default_intrinsics["K"]
        print(
            f"Using intrinsics from `{viewer.default_intrinsics['name']}`: "
            f"{viewer.default_intrinsics['W']}x{viewer.default_intrinsics['H']}, "
            f"fx={K[0, 0]:.3f}, fy={K[1, 1]:.3f}, "
            f"cx={K[0, 2]:.3f}, cy={K[1, 2]:.3f}"
        )
    print(f"Output dir: {viewer.output_dir}")
    viewer.run()


if __name__ == "__main__":
    main()
