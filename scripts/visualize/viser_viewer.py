#!/usr/bin/env python3
"""Simple browser-based viser viewer for Scal3R reconstruction outputs."""

from __future__ import annotations

import argparse
import re
import time
from os.path import abspath, dirname, exists, join

import numpy as np
from scipy.spatial.transform import Rotation

visualize_root = dirname(abspath(__file__))
release_root = dirname(dirname(visualize_root))

import sys

if visualize_root not in sys.path:
    sys.path.insert(0, visualize_root)
if release_root not in sys.path:
    sys.path.insert(0, release_root)

from utils.point_utils import load_pointcloud_arrays, random_downsample  # noqa: E402


def rainbow_colormap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)

    def _channel(offset: float) -> np.ndarray:
        x = np.abs(((values + offset) * 6.0) % 6.0 - 3.0) - 1.0
        return np.clip(x, 0.0, 1.0)

    rgb = np.stack([_channel(0.0), _channel(2.0 / 6.0), _channel(4.0 / 6.0)], axis=-1)
    return rgb.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply_path", required=True, help="Path to the whole-scene PLY file.")
    parser.add_argument("--pose_path", required=True, help="Path to mat.txt containing c2w poses.")
    parser.add_argument(
        "--intri_path",
        default="",
        help="Optional intri.yml path. Defaults to a sibling intri.yml next to pose_path if present.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the viser server to.")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the viser server to.")
    parser.add_argument(
        "--max_points",
        type=int,
        default=600000,
        help="Randomly cap the point cloud to this many points; <=0 disables downsampling.",
    )
    parser.add_argument("--point_size", type=float, default=0.002, help="Viser point size.")
    parser.add_argument(
        "--camera_stride",
        type=int,
        default=5,
        help="Visualize every Nth camera frustum; 1 shows all cameras.",
    )
    parser.add_argument("--frustum_scale", type=float, default=0.15, help="Camera frustum scale.")
    parser.add_argument("--line_width", type=float, default=2.0, help="Camera frustum line width.")
    parser.add_argument("--show_trajectory", type=int, default=1, help="Whether to render the camera path.")
    parser.add_argument("--trajectory_line_width", type=float, default=1.5, help="Camera path line width.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random point downsampling.")
    return parser.parse_args()


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


def load_first_intrinsics(intri_path: str) -> tuple[np.ndarray, int, int] | None:
    if not intri_path or not exists(intri_path):
        return None
    with open(intri_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    first_key = re.search(r"K_([0-9A-Za-z_]+):\s*!!opencv-matrix", text)
    if not first_key:
        return None
    suffix = first_key.group(1)
    intrinsic = _parse_opencv_matrix(text, f"K_{suffix}")
    height = _parse_scalar(text, f"H_{suffix}")
    width = _parse_scalar(text, f"W_{suffix}")
    if intrinsic is None or height is None or width is None:
        return None
    return intrinsic, int(round(height)), int(round(width))


def load_poses(pose_path: str) -> np.ndarray:
    poses = np.loadtxt(pose_path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.shape[1] == 16:
        return poses.reshape(-1, 4, 4)
    if poses.shape[1] == 12:
        poses = poses.reshape(-1, 3, 4)
        padded = np.tile(np.eye(4, dtype=np.float64)[None], (poses.shape[0], 1, 1))
        padded[:, :3, :] = poses
        return padded
    raise ValueError(f"Unsupported pose shape from {pose_path}: {poses.shape}")


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return (float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2]))


def main() -> None:
    args = parse_args()

    try:
        import viser
    except ImportError as exc:
        raise SystemExit(
            "viser is not installed. Run `pip install -r scripts/visualize/requirements.txt` first."
        ) from exc

    ply_path = abspath(args.ply_path)
    pose_path = abspath(args.pose_path)
    intri_path = args.intri_path or join(dirname(pose_path), "intri.yml")
    intri_path = abspath(intri_path)

    print(f"Loading point cloud: {ply_path}")
    xyz, rgb = load_pointcloud_arrays(ply_path)
    total_points = len(xyz)
    xyz, rgb = random_downsample(xyz, rgb, max_points=args.max_points, seed=args.seed)
    print(f"Point cloud for viser: {len(xyz):,}/{total_points:,} points")

    poses = load_poses(pose_path)
    print(f"Loaded {len(poses)} camera poses from {pose_path}")

    intrinsic_bundle = load_first_intrinsics(intri_path)
    if intrinsic_bundle is None:
        width, height = 1280, 720
        fov = np.deg2rad(60.0)
        aspect = width / height
        print("No intri.yml found; using fallback intrinsics: 1280x720, 60 degree vertical FOV")
    else:
        intrinsic, height, width = intrinsic_bundle
        fy = float(intrinsic[1, 1])
        fov = 2.0 * np.arctan(height / (2.0 * fy))
        aspect = width / height
        print(
            f"Using intrinsics from {intri_path}: "
            f"{width}x{height}, fy={fy:.3f}, vertical_fov_deg={np.rad2deg(fov):.2f}"
        )

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("-y")
    server.scene.add_frame("/world", axes_length=0.5, axes_radius=0.01)
    server.scene.add_point_cloud(
        "/reconstruction/points",
        points=xyz.astype(np.float32),
        colors=np.clip(rgb, 0.0, 1.0).astype(np.float32),
        point_size=float(args.point_size),
        point_shape="circle",
    )

    stride = max(int(args.camera_stride), 1)
    shown_indices = list(range(0, len(poses), stride))
    camera_colors = rainbow_colormap(np.linspace(0.0, 1.0, len(poses), endpoint=False))
    for index in range(0, len(poses), stride):
        pose = poses[index]
        server.scene.add_camera_frustum(
            f"/reconstruction/cameras/{index:06d}",
            fov=float(fov),
            aspect=float(aspect),
            scale=float(args.frustum_scale),
            line_width=float(args.line_width),
            color=(camera_colors[index] * 255.0).astype(np.uint8),
            wxyz=rotation_matrix_to_wxyz(pose[:3, :3]),
            position=pose[:3, 3].astype(np.float32),
        )

    if int(args.show_trajectory):
        centers = poses[:, :3, 3].astype(np.float32)
        if len(centers) >= 2:
            segments = np.stack([centers[:-1], centers[1:]], axis=1)
            segment_colors = np.stack([camera_colors[:-1], camera_colors[1:]], axis=1)
            server.scene.add_line_segments(
                "/reconstruction/trajectory",
                points=segments,
                colors=(segment_colors * 255.0).astype(np.uint8),
                line_width=float(args.trajectory_line_width),
            )

    print(
        f"Displaying {len(xyz):,} points and {len(shown_indices)}/{len(poses)} camera frustums "
        f"(camera_stride={stride})"
    )
    print(f"Viser viewer ready at http://{args.host}:{args.port}")
    print("If bound to 0.0.0.0 on a remote server, use SSH port forwarding from your laptop.")

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
