#!/usr/bin/env python3
"""
Visualize camera poses as a colored frustum point cloud.

Relationship to `scripts/gfm/scal3r/inference_parallel.py`
----------------------------------------------------------
`inference_parallel.py` saves three relevant outputs:
- `mat.txt`: predicted cameras in the stitched prediction world frame
- `mat_gt.txt`: ground-truth cameras in the original GT world frame
- `mat_xyz.txt`: cached GT cameras already moved into the `whole.ply` frame
- `xyz.ply`: the exported `whole.ply`

`xyz.ply` is not in raw GT coordinates. Before saving `whole.ply`,
`inference_parallel.py` estimates a relative 7-DoF alignment between `mat.txt`
and `mat_gt.txt`, then applies that transform to the predicted world points.
This script reproduces that step so GT cameras can be moved into the same
frame as `xyz.ply`.

Typical steps
-------------
1. Pass `--pose_path data/datasets/kitti/07/mat_gt.txt`
2. Pass `--pred_pose_path data/datasets/kitti/07/mat.txt`
3. Enable `--align_gt_to_whole 1`
4. Pass `--ply_path data/datasets/kitti/07/xyz.ply`
5. Inspect `cam.ply` and `all.ply`

Keep `--normalize_first 0` when checking alignment with `xyz.ply`.
If you pass a subset copy of `mat_xyz.txt` under another filename, also set
`--pose_in_xyz_frame 1` so the script does not re-normalize it by mistake.
"""

import os
import sys
import argparse
import numpy as np
from os.path import abspath, dirname, exists

visualize_root = dirname(dirname(dirname(abspath(__file__))))
release_root = dirname(dirname(visualize_root))
if visualize_root not in sys.path:
    sys.path.insert(0, visualize_root)
if release_root not in sys.path:
    sys.path.insert(0, release_root)

from utils.color_utils import rainbow_colormap  # noqa: E402
from utils.camera_utils import frustum_points, prepare_poses  # noqa: E402
from utils.camera_utils import FRUSTUM_EDGES as frustum_edges  # noqa: E402


def make_dense_frustum_points(pose, size=2.0, aspect=16 / 9, samples=50):
    """Sample dense points along frustum edges for thicker visualization."""
    corners = frustum_points(pose, size=size, aspect=aspect)

    points = []
    for src, dst in frustum_edges:
        for frac in np.linspace(0, 1, samples):
            points.append(corners[src] * (1 - frac) + corners[dst] * frac)

    return np.asarray(points)


def save_pose_array(path: str, poses: np.ndarray):
    if not path:
        return

    out_dir = dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savetxt(path, poses.reshape(len(poses), -1), fmt="%.8f")
    print(f"Saved transformed poses to {path}")


def save_alignment(path: str, world: np.ndarray, scale: float, rigid: np.ndarray):
    if not path:
        return

    out_dir = dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(path, world_transform=world, align_scale=scale, align_transform=rigid)
    print(f"Saved alignment metadata to {path}")


def build_camera_cloud(poses: np.ndarray, size: float, step: int, samples: int):
    count = len(poses)
    vals = np.linspace(0, 1, count, endpoint=False)
    rgbs = rainbow_colormap(vals)

    ptss = []
    cols = []
    keep = list(range(0, count, step))
    print(f"Generating {len(keep)} frustums (every {step} poses)...")

    for idx in keep:
        pts = make_dense_frustum_points(poses[idx], size=size, samples=samples)
        col = np.tile(rgbs[idx], (len(pts), 1))
        ptss.append(pts)
        cols.append(col)

    cams = poses[:, :3, 3]
    traj_pts = []
    traj_rgb = []
    for idx in range(count - 1):
        for frac in np.linspace(0, 1, 10):
            traj_pts.append(cams[idx] * (1 - frac) + cams[idx + 1] * frac)
            traj_rgb.append(rgbs[idx] * (1 - frac) + rgbs[idx + 1] * frac)

    if traj_pts:
        ptss.append(np.asarray(traj_pts))
        cols.append(np.asarray(traj_rgb))

    return np.vstack(ptss), np.vstack(cols)


def save_point_cloud(path: str, xyz: np.ndarray, rgb: np.ndarray):
    import open3d as o3d

    out_dir = dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(path, cloud)
    print(f"Saved {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_path", required=True)
    parser.add_argument("--pred_pose_path", default="")
    parser.add_argument("--ply_path", default="", help="Point cloud PLY to merge")
    parser.add_argument("--output_cam", default="cam.ply")
    parser.add_argument("--output_all", default="all.ply")
    parser.add_argument("--output_pose", default="")
    parser.add_argument("--output_alignment", default="")
    parser.add_argument("--frustum_size", type=float, default=2.0)
    parser.add_argument("--frustum_step", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=80)
    parser.add_argument("--align_gt_to_whole", type=int, default=0)
    parser.add_argument("--normalize_first", type=int, default=-1)
    parser.add_argument(
        "--pose_in_xyz_frame",
        type=int,
        default=-1,
        help="Whether pose_path is already in xyz/whole coordinates: -1 auto, 0 no, 1 yes",
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Lazy import to avoid circular import
    import open3d as o3d

    poses, meta = prepare_poses(
        args.pose_path,
        pred_pose_path=args.pred_pose_path,
        align_gt_to_whole=args.align_gt_to_whole,
        normalize_first=args.normalize_first,
        pose_in_xyz_frame=args.pose_in_xyz_frame,
    )
    print(f"Loaded {len(poses)} poses")

    if args.align_gt_to_whole:
        world = meta["world_transform"]
        scale = meta["align_scale"]
        rigid = meta["align_transform"]
        print("Aligned GT poses to `whole.ply` frame")
        print(f"  align_scale: {scale:.6f}")
        print(f"  world_transform:\n{world}")
        save_alignment(args.output_alignment, world, scale, rigid)
    if meta.get("loaded_xyz_pose_cache"):
        print(f"Reused cached xyz-frame poses: {meta['xyz_pose_path']}")
    if meta.get("saved_xyz_pose_cache"):
        print(f"Saved cached xyz-frame poses: {meta['xyz_pose_path']}")

    if meta.get("normalized_first"):
        print("Normalized by first frame (pose[0] -> identity)")
    save_pose_array(args.output_pose, poses)

    cam_xyz, cam_rgb = build_camera_cloud(
        poses,
        size=args.frustum_size,
        step=args.frustum_step,
        samples=args.n_samples,
    )
    print(f"Camera PLY: {len(cam_xyz):,} points")
    save_point_cloud(args.output_cam, cam_xyz, cam_rgb)

    if args.ply_path and exists(args.ply_path):
        print(f"Loading scene point cloud: {args.ply_path}")
        scene = o3d.io.read_point_cloud(args.ply_path)
        xyz = np.asarray(scene.points)
        rgb = np.asarray(scene.colors)
        print(f"  Scene: {len(xyz):,} points (unchanged)")
        save_point_cloud(
            args.output_all,
            np.vstack([xyz, cam_xyz]),
            np.vstack([rgb, cam_rgb]),
        )
        print(f"Saved merged point cloud with {len(xyz) + len(cam_xyz):,} points")
    else:
        print("No scene PLY provided, skipping merge")


if __name__ == "__main__":
    main()
