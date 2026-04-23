#!/usr/bin/env python3
"""Align GT-side or pred-side poses into the `xyz` / `whole` frame.

This is a minimal wrapper around the same pose-alignment path used by
`render_points.py` and the other visualization scripts.

Usage:
    python scripts/visualize/geometry/utils/align_coordinate.py \
        data/datasets/kitti/07
"""

from __future__ import annotations

import argparse
import sys
from os.path import abspath, dirname, exists, isdir, join

visualize_root = dirname(dirname(dirname(abspath(__file__))))
release_root = dirname(dirname(visualize_root))
if visualize_root not in sys.path:
    sys.path.insert(0, visualize_root)
if release_root not in sys.path:
    sys.path.insert(0, release_root)

from utils.camera_utils import (  # noqa: E402
    align_gt_poses_to_whole,
    align_pred_poses_to_whole,
    load_poses,
    prepare_poses,
    save_poses,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing mat.txt and mat_gt.txt")
    parser.add_argument("--pred_name", default="mat.txt", help="Predicted pose filename")
    parser.add_argument("--gt_name", default="mat_gt.txt", help="Ground-truth pose filename")
    parser.add_argument(
        "--source",
        choices=["gt", "pred"],
        default="gt",
        help="Which pose family to move into the xyz/whole frame",
    )
    parser.add_argument(
        "--output_name",
        default="",
        help="Aligned output pose filename; defaults to mat_xyz.txt for gt or mat_pred_xyz.txt for pred",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        help="Recompute alignment even if the output cache already exists",
    )
    return parser.parse_args()


def require_file(path: str, name: str):
    if not exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")


def compute_aligned_poses(gt_path: str, pred_path: str, source: str):
    gt_poses = load_poses(gt_path)
    pred_poses = load_poses(pred_path)
    if len(gt_poses) != len(pred_poses):
        raise ValueError(f"Pose count mismatch: gt={len(gt_poses)} pred={len(pred_poses)}")
    if source == "gt":
        return align_gt_poses_to_whole(gt_poses, pred_poses)
    return align_pred_poses_to_whole(gt_poses, pred_poses)


def main():
    args = parse_args()

    data_dir = abspath(args.data_dir)
    if not isdir(data_dir):
        raise NotADirectoryError(f"Not a directory: {data_dir}")

    pred_path = join(data_dir, args.pred_name)
    gt_path = join(data_dir, args.gt_name)
    output_name = args.output_name or ("mat_xyz.txt" if args.source == "gt" else "mat_pred_xyz.txt")
    output_path = join(data_dir, output_name)

    require_file(pred_path, args.pred_name)
    require_file(gt_path, args.gt_name)

    if args.overwrite:
        poses, meta = compute_aligned_poses(gt_path, pred_path, args.source)
        cache_status = "recomputed"
    else:
        if args.source == "gt":
            poses, meta = prepare_poses(
                gt_path,
                pred_pose_path=pred_path,
                align_gt_to_whole=1,
                normalize_first=-1,
                pose_in_xyz_frame=0,
            )
            if meta.get("loaded_xyz_pose_cache"):
                cache_status = "reused"
            elif meta.get("saved_xyz_pose_cache"):
                cache_status = "computed"
            else:
                cache_status = "computed"
        else:
            if exists(output_path):
                poses = load_poses(output_path)
                meta = {"xyz_pose_path": output_path}
                cache_status = "reused"
            else:
                poses, meta = compute_aligned_poses(gt_path, pred_path, args.source)
                cache_status = "computed"

    save_poses(output_path, poses)

    print(f"Saved {len(poses)} aligned poses to {output_path}")
    print(f"  source:     {args.source}")
    print(f"  pred poses: {pred_path}")
    print(f"  gt poses:   {gt_path}")
    print(f"  status:     {cache_status}")
    if meta.get("xyz_pose_path"):
        print(f"  cache path: {meta['xyz_pose_path']}")
    if "align_scale" in meta:
        print(f"  align_scale: {meta['align_scale']:.6f}")
    if "world_transform" in meta:
        print(f"  world_transform:\n{meta['world_transform']}")


if __name__ == "__main__":
    main()
