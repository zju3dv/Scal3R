#!/usr/bin/env python3
"""Render a full static Scal3R scene from a custom viewpoint trajectory.

This script renders:
- the full `xyz.ply` point cloud
- the full input camera trajectory (`mat_xyz.txt`, `mat_gt.txt`, or similar)

Unlike `render_points.py`, this script does not progressively reveal points.
Unlike `render_cameras.py`, it does not animate camera history growth.
All scene content stays visible in every frame; only the render viewpoint moves.

Typical output layout:
- `data/output/kitti/07/global/COMBO` for rendered frames
- `data/output/kitti/07/global/global.mp4` for the encoded video
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from PIL import Image
from os.path import abspath, dirname

visualize_root = dirname(dirname(dirname(abspath(__file__))))
release_root = dirname(dirname(visualize_root))
if visualize_root not in sys.path: sys.path.insert(0, visualize_root)
if release_root not in sys.path: sys.path.insert(0, release_root)

from utils.camera_utils import prepare_poses
from utils.point_utils import adjust_point_colors, load_pointcloud_arrays, voxel_downsample
from utils.render_utils import build_camera_image_quad, build_frustum_pointcloud, configure_offscreen_renderer, find_image_files, hex_to_float_rgb


def load_viewpoint_config(path: str):
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def encode_video(frames_dir: str, output_video: str, fps: float):
    frame_pattern = os.path.join(frames_dir, "frame_%05d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "slow",
        output_video,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def subsample_viewpoints(
    viewpoints: list[dict],
    stride: int = 1,
    keep_last: int = 1,
) -> tuple[list[dict], list[int]]:
    stride = max(int(stride), 1)
    keep_last = int(keep_last)
    if stride <= 1 or len(viewpoints) <= 1:
        return viewpoints, list(range(len(viewpoints)))

    keep_indices = list(range(0, len(viewpoints), stride))
    if keep_last and keep_indices[-1] != len(viewpoints) - 1:
        keep_indices.append(len(viewpoints) - 1)

    sampled = [viewpoints[idx] for idx in keep_indices]
    return sampled, keep_indices


def render_scene_frames(
    ply_path: str,
    capture_pose_path: str,
    viewpoint_config: str,
    output_dir: str,
    output_video: str = "",
    pred_pose_path: str = "",
    align_gt_to_whole: int = 0,
    normalize_first: int = -1,
    pose_in_xyz_frame: int = -1,
    bg_color: str = "",
    voxel_size: float = 0.0,
    point_size: float = 2.5,
    frustum_size: float = 0.15,
    frustum_aspect: float = -1.0,
    frustum_pose_stride: int = 1,
    frustum_point_size: float = 3.5,
    frustum_point_samples: int = 128,
    camera_image_dir: str = "",
    camera_image_max_dim: int = 256,
    render_camera_images: int = 1,
    viewpoint_stride: int = 1,
    viewpoint_keep_last: int = 1,
    viewpoint_fps_mode: str = "keep_speed",
    color_gamma: float = 1.0,
    color_gain: float = 1.0,
    color_saturation: float = 1.0,
    color_contrast: float = 1.0,
):
    import open3d as o3d
    import open3d.visualization.rendering as rendering  # type: ignore[import-unresolved]
    from tqdm import tqdm

    cfg = load_viewpoint_config(viewpoint_config)
    viewpoints = cfg["viewpoints"]
    width = int(cfg["width"])
    height = int(cfg["height"])
    fps = float(cfg.get("fps", 30.0))
    viewpoint_stride = max(int(viewpoint_stride), 1)
    viewpoint_keep_last = int(viewpoint_keep_last)
    if viewpoint_fps_mode not in {"keep_speed", "keep_fps"}:
        raise ValueError(
            f"Unknown viewpoint_fps_mode={viewpoint_fps_mode}; expected keep_speed or keep_fps"
        )

    original_n_viewpoints = len(viewpoints)
    viewpoints, viewpoint_indices = subsample_viewpoints(
        viewpoints,
        stride=viewpoint_stride,
        keep_last=viewpoint_keep_last,
    )
    effective_fps = fps / viewpoint_stride if viewpoint_fps_mode == "keep_speed" else fps
    if viewpoint_stride > 1:
        print(
            "Subsampled render viewpoints: "
            f"{original_n_viewpoints} -> {len(viewpoints)} "
            f"(stride={viewpoint_stride}, keep_last={viewpoint_keep_last}, "
            f"fps_mode={viewpoint_fps_mode}, output_fps={effective_fps:.6g})"
        )

    if "intrinsic" not in cfg:
        raise ValueError("`viewpoint_config` must contain `intrinsic` for firstperson rendering")
    intrinsic = np.array(cfg["intrinsic"], dtype=np.float64)

    if not bg_color:
        bg_color = cfg.get("bg_color", "#ffffff")

    print("Loading point cloud...")
    xyz, rgb = load_pointcloud_arrays(ply_path)
    print(f"  {len(xyz):,} points loaded")
    if voxel_size > 0:
        xyz, rgb = voxel_downsample(xyz, rgb, voxel_size)
        print(f"  {len(xyz):,} points after voxel downsampling (voxel_size={voxel_size})")
    if (
        color_gamma != 1.0
        or color_gain != 1.0
        or color_saturation != 1.0
        or color_contrast != 1.0
    ):
        print(
            "Applying point color remap: "
            f"gamma={color_gamma:.3f}, gain={color_gain:.3f}, "
            f"saturation={color_saturation:.3f}, contrast={color_contrast:.3f}"
        )
        rgb = adjust_point_colors(
            rgb,
            gamma=color_gamma,
            gain=color_gain,
            saturation=color_saturation,
            contrast=color_contrast,
        )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb)

    print("Loading capture poses...")
    poses, pose_meta = prepare_poses(
        capture_pose_path,
        pred_pose_path=pred_pose_path,
        align_gt_to_whole=align_gt_to_whole,
        normalize_first=normalize_first,
        pose_in_xyz_frame=pose_in_xyz_frame,
    )
    print(f"  {len(poses)} poses loaded")
    if pose_meta.get("loaded_xyz_pose_cache"):
        print(f"  Reused cached xyz-frame poses: {pose_meta['xyz_pose_path']}")
    if pose_meta.get("saved_xyz_pose_cache"):
        print(f"  Saved cached xyz-frame poses: {pose_meta['xyz_pose_path']}")
    if pose_meta.get("normalized_first"):
        print("  Poses normalized by first frame")

    visible_indices = list(range(0, len(poses), max(int(frustum_pose_stride), 1)))

    camera_image_files: list[str] = []
    if camera_image_dir:
        camera_image_files = find_image_files(camera_image_dir)
        if not camera_image_files:
            raise ValueError(f"No images found in camera_image_dir={camera_image_dir}")
        if len(camera_image_files) != len(poses):
            raise ValueError(
                f"Image/pose count mismatch: images={len(camera_image_files)} poses={len(poses)}"
            )
        if frustum_aspect <= 0:
            with Image.open(camera_image_files[0]) as img:
                frustum_aspect = img.size[0] / max(img.size[1], 1)
        print(f"  Found {len(camera_image_files)} camera images in {camera_image_dir}")
    if frustum_aspect <= 0:
        frustum_aspect = width / max(height, 1)

    frustums = build_frustum_pointcloud(
        poses,
        visible_indices=visible_indices,
        frustum_size=frustum_size,
        point_samples=frustum_point_samples,
        history_size_step=0.0,
        aspect=frustum_aspect,
    )

    print("Setting up offscreen renderer...")
    renderer = rendering.OffscreenRenderer(width, height)
    configure_offscreen_renderer(renderer, rendering, bg_color)

    mat_points = rendering.MaterialRecord()
    mat_points.shader = "defaultUnlit"
    mat_points.point_size = point_size

    mat_cams = rendering.MaterialRecord()
    mat_cams.shader = "defaultUnlit"
    mat_cams.point_size = frustum_point_size

    renderer.scene.add_geometry("scene_points", point_cloud, mat_points)
    if frustums is not None:
        renderer.scene.add_geometry("capture_frustums", frustums, mat_cams)
    if render_camera_images and camera_image_files:
        print("Adding camera image quads...")
        for pose_idx in visible_indices:
            mesh, material = build_camera_image_quad(
                poses[pose_idx],
                aspect=frustum_aspect,
                frustum_size=frustum_size,
                image_path=camera_image_files[pose_idx],
                max_texture_dim=camera_image_max_dim,
            )
            renderer.scene.add_geometry(f"capture_image_{pose_idx:06d}", mesh, material)
        print(
            f"  Added {len(visible_indices)} textured camera quads "
            f"(max_texture_dim={camera_image_max_dim})"
        )

    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        intrinsic[0, 0],
        intrinsic[1, 1],
        intrinsic[0, 2],
        intrinsic[1, 2],
    )

    os.makedirs(output_dir, exist_ok=True)
    print(f"Rendering {len(viewpoints)} frames...")
    for frame_i, vp in enumerate(tqdm(viewpoints)):
        extrinsic = np.array(vp["extrinsic"], dtype=np.float64)
        renderer.setup_camera(o3d_intrinsic, extrinsic)
        img = renderer.render_to_image()
        out_path = os.path.join(output_dir, f"frame_{frame_i:05d}.png")
        o3d.io.write_image(out_path, img)

    if not output_video:
        output_root = os.path.basename(os.path.dirname(output_dir.rstrip(os.sep))) or "global"
        output_video = os.path.join(os.path.dirname(output_dir), f"{output_root}.mp4")

    print("Encoding MP4 with ffmpeg...")
    encode_video(output_dir, output_video, fps=effective_fps)
    print(f"Video saved to {output_video}")


def main():
    parser = argparse.ArgumentParser(description="Render full static point cloud + cameras from a custom path")
    parser.add_argument("--ply_path", required=True)
    parser.add_argument("--capture_pose_path", required=True)
    parser.add_argument("--viewpoint_config", required=True)
    parser.add_argument("--output_dir", default="data/output/kitti/07/global/COMBO")
    parser.add_argument("--output_video", default="")
    parser.add_argument("--pred_pose_path", default="")
    parser.add_argument("--align_gt_to_whole", type=int, default=0)
    parser.add_argument("--normalize_first", type=int, default=-1)
    parser.add_argument(
        "--pose_in_xyz_frame",
        type=int,
        default=-1,
        help="Whether capture_pose_path is already in xyz/whole coordinates: -1 auto, 0 no, 1 yes",
    )
    parser.add_argument("--bg_color", default="#ffffff")
    parser.add_argument("--voxel_size", type=float, default=0.0)
    parser.add_argument("--point_size", type=float, default=2.5)
    parser.add_argument(
        "--color_gamma",
        type=float,
        default=1.0,
        help="Point-color power remap; >1 darkens midtones, <1 brightens",
    )
    parser.add_argument(
        "--color_gain",
        type=float,
        default=1.0,
        help="Point-color brightness scale after other adjustments",
    )
    parser.add_argument(
        "--color_saturation",
        type=float,
        default=1.0,
        help="Point-color saturation scale; 1 keeps original, 0 becomes grayscale",
    )
    parser.add_argument(
        "--color_contrast",
        type=float,
        default=1.0,
        help="Point-color contrast scale around 0.5",
    )
    parser.add_argument("--frustum_size", type=float, default=0.15)
    parser.add_argument(
        "--frustum_aspect",
        type=float,
        default=-1.0,
        help="Frustum and image-plane aspect ratio; <=0 defaults to viewpoint width/height",
    )
    parser.add_argument("--frustum_pose_stride", type=int, default=1)
    parser.add_argument("--frustum_point_size", type=float, default=3.5)
    parser.add_argument("--frustum_point_samples", type=int, default=128)
    parser.add_argument("--camera_image_dir", default="")
    parser.add_argument("--camera_image_max_dim", type=int, default=256)
    parser.add_argument("--render_camera_images", type=int, default=1)
    parser.add_argument(
        "--viewpoint_stride",
        type=int,
        default=1,
        help="Subsample render viewpoints from viewpoint_config by this stride; 1 keeps all",
    )
    parser.add_argument(
        "--viewpoint_keep_last",
        type=int,
        default=1,
        help="When viewpoint_stride > 1, keep the final viewpoint even if it is off-stride",
    )
    parser.add_argument(
        "--viewpoint_fps_mode",
        choices=["keep_speed", "keep_fps"],
        default="keep_speed",
        help="keep_speed divides output fps by viewpoint_stride; keep_fps keeps fps unchanged",
    )
    args = parser.parse_args()

    render_scene_frames(**vars(args))


if __name__ == "__main__":
    main()
