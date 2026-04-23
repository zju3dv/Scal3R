"""Render point-cloud-only frames and save shared viewpoint metadata.

The default reveal mode uses cumulative frustum visibility: a point appears
once it falls inside one of the input camera frustums, then stays visible.
This is closer to a real "seen by camera" process than the older
trajectory-progress heuristic.

Usage:
    python scripts/visualize/geometry/render/render_points.py \
        --ply_path data/datasets/kitti/07/xyz.ply \
        --pose_path data/datasets/kitti/07/mat_xyz.txt \
        --output_dir data/output/kitti/07/progressive/POINT
"""

import os
import sys
import json
import argparse
import numpy as np
from os.path import abspath, dirname

visualize_root = dirname(dirname(dirname(abspath(__file__))))
release_root = dirname(dirname(visualize_root))
if visualize_root not in sys.path: sys.path.insert(0, visualize_root)
if release_root not in sys.path: sys.path.insert(0, release_root)

from utils.render_utils import configure_offscreen_renderer  # noqa: E402
from utils.point_utils import voxel_downsample, adjust_point_colors, get_progressive_mask, load_pointcloud_arrays, assign_points_to_poses, get_camera_frustum_mask  # noqa: E402
from utils.camera_utils import prepare_poses, make_intrinsic, get_camera_forward, get_camera_positions, interpolate_viewpoints, compute_follow_viewpoints, compute_bird_eye_viewpoint, opencv_c2w_to_o3d_extrinsic, compute_follow_viewpoints_from_masks, compute_follow_viewpoints_from_pointcloud  # noqa: E402


def normalize_vector(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.copy()
    return vec / norm


def describe_rgb_stats(tag: str, rgb: np.ndarray):
    if len(rgb) == 0:
        print(f"{tag}: empty")
        return

    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    sat = rgb.max(axis=1) - rgb.min(axis=1)
    mean_rgb = rgb.mean(axis=0)
    print(
        f"{tag}: mean_rgb=[{mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f}] "
        f"luma_mean={luma.mean():.3f} sat_mean={sat.mean():.3f}"
    )


def build_lookat_pose(
    eye: np.ndarray,
    lookat: np.ndarray,
    world_up: np.ndarray = np.array([0.0, -1.0, 0.0]),
) -> np.ndarray:
    """Build an OpenCV-style c2w pose from eye/lookat."""
    forward = normalize_vector(lookat - eye)
    down_ref = -world_up
    down = down_ref - forward * np.dot(down_ref, forward)
    down = normalize_vector(down)
    right = normalize_vector(np.cross(down, forward))
    down = normalize_vector(np.cross(forward, right))

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def build_render_poses(
    poses: np.ndarray,
    frame_indices,
    render_pose_lag: int = 0,
    render_back_offset: float = 0.0,
    render_side_offset: float = 0.0,
    render_up_offset: float = 0.0,
    render_lookahead: float = 3.0,
    render_lookdown_offset: float = 0.0,
):
    """Build camera poses used only for rendering the viewport."""
    render_poses = []
    render_indices = []
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    world_down = -world_up

    for pose_idx in frame_indices:
        render_idx = max(0, pose_idx - render_pose_lag)
        base_pose = poses[render_idx]
        forward = normalize_vector(base_pose[:3, 2])
        right = normalize_vector(base_pose[:3, 0])
        center = base_pose[:3, 3].copy()

        eye = (
            center
            - forward * render_back_offset
            + right * render_side_offset
            + world_up * render_up_offset
        )
        lookat = center + forward * render_lookahead + world_down * render_lookdown_offset
        render_pose = build_lookat_pose(eye, lookat, world_up=world_up)

        render_poses.append(render_pose)
        render_indices.append(render_idx)

    return np.stack(render_poses, axis=0), render_indices


def render_frames(
    ply_path: str,
    pose_path: str,
    output_dir: str,
    pred_pose_path: str = "",
    render_pose_path: str = "",
    align_gt_to_whole: int = 0,
    normalize_first: int = -1,
    pose_in_xyz_frame: int = -1,
    render_pose_in_xyz_frame: int = 1,
    width: int = 1920,
    height: int = 1080,
    bg_color: str = "#ffffff",
    frame_step: int = 1,
    voxel_size: float = 0.0,
    view_mode: str = "follow",
    offset_back: float = 3.0,
    offset_up: float = 0.4,
    lookahead: float = 5.0,
    smooth_window: int = 51,
    fov: float = 60.0,
    point_size: float = 3.0,
    fx: float = 0.0,
    fy: float = 0.0,
    cx: float = 0.0,
    cy: float = 0.0,
    reveal_mode: str = "accumulate_frustum",
    max_depth: float = 60.0,
    near_depth: float = 0.1,
    chunk_size: int = 262144,
    render_pose_lag: int = 5,
    render_back_offset: float = 1.25,
    render_side_offset: float = 0.4,
    render_up_offset: float = 1.0,
    render_lookahead: float = 5.0,
    render_lookdown_offset: float = 0.04,
    color_gamma: float = 1.0,
    color_gain: float = 1.0,
    color_saturation: float = 1.0,
    color_contrast: float = 1.0,
    viewpoint_interp: int = 1,
):
    import open3d as o3d
    import open3d.visualization.rendering as rendering  # type: ignore[import-unresolved]
    from tqdm import tqdm

    # Load data
    print("Loading point cloud...")
    xyz, rgb = load_pointcloud_arrays(ply_path)
    print(f"  {len(xyz):,} points loaded")
    describe_rgb_stats("  Raw point colors", rgb)

    if voxel_size > 0:
        print(f"Downsampling with voxel_size={voxel_size}...")
        xyz, rgb = voxel_downsample(xyz, rgb, voxel_size)
        print(f"  {len(xyz):,} points after downsampling")
        describe_rgb_stats("  Downsampled point colors", rgb)

    if color_gamma != 1.0 or color_gain != 1.0 or color_saturation != 1.0 or color_contrast != 1.0:
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
        describe_rgb_stats("  Adjusted point colors", rgb)

    print("Loading poses...")
    poses, pose_meta = prepare_poses(
        pose_path,
        pred_pose_path=pred_pose_path,
        align_gt_to_whole=align_gt_to_whole,
        normalize_first=normalize_first,
        pose_in_xyz_frame=pose_in_xyz_frame,
    )
    n_poses = len(poses)
    print(f"  {n_poses} poses loaded")
    if align_gt_to_whole:
        print("  GT poses aligned to `whole.ply` frame")
    if pose_meta.get("loaded_xyz_pose_cache"):
        print(f"  Reused cached xyz-frame poses: {pose_meta['xyz_pose_path']}")
    if pose_meta.get("saved_xyz_pose_cache"):
        print(f"  Saved cached xyz-frame poses: {pose_meta['xyz_pose_path']}")
    if pose_meta.get("normalized_first"):
        print("  Poses normalized by first frame")

    render_pose_meta = {}
    render_pose_source = None
    n_render_poses = 0
    if render_pose_path:
        if view_mode != "firstperson":
            raise ValueError("`render_pose_path` is only supported when `view_mode=firstperson`")
        print("Loading render poses...")
        render_pose_source, render_pose_meta = prepare_poses(
            render_pose_path,
            normalize_first=-1,
            pose_in_xyz_frame=render_pose_in_xyz_frame,
        )
        n_render_poses = len(render_pose_source)
        print(f"  {n_render_poses} render poses loaded")
        if render_pose_meta.get("loaded_xyz_pose_cache"):
            print(f"  Render poses reused xyz-frame cache: {render_pose_meta['xyz_pose_path']}")
        if render_pose_meta.get("saved_xyz_pose_cache"):
            print(f"  Render poses saved xyz-frame cache: {render_pose_meta['xyz_pose_path']}")
        if render_pose_meta.get("normalized_first"):
            print("  Render poses normalized by first frame")

    # Build frame list
    os.makedirs(output_dir, exist_ok=True)
    config_dir = os.path.dirname(output_dir)
    viewpoint_path = os.path.join(config_dir, "viewpoint_config.json")

    n_frame_poses = n_poses
    if render_pose_source is not None:
        n_frame_poses = min(n_poses, n_render_poses)
        if n_render_poses != n_poses:
            print(
                "  Capture/render pose count mismatch: "
                f"capture={n_poses}, render={n_render_poses}; "
                f"limiting progressive output to first {n_frame_poses} poses"
            )

    frame_indices = list(range(0, n_frame_poses, frame_step))
    if render_pose_source is not None:
        render_pose_indices = list(frame_indices)
        render_poses = render_pose_source[frame_indices].copy()
    else:
        render_poses, render_pose_indices = build_render_poses(
            poses,
            frame_indices,
            render_pose_lag=render_pose_lag,
            render_back_offset=render_back_offset,
            render_side_offset=render_side_offset,
            render_up_offset=render_up_offset,
            render_lookahead=render_lookahead,
            render_lookdown_offset=render_lookdown_offset,
        )

    # Build reveal camera intrinsic
    if fx > 0 and fy > 0:
        if cx <= 0:
            cx = width / 2.0
        if cy <= 0:
            cy = height / 2.0
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        intrinsic = make_intrinsic(fov, width, height)

    reveal_progress = None
    if reveal_mode == "trajectory":
        print("Assigning points to trajectory progress...")
        cam_positions = get_camera_positions(poses)
        cam_forwards = get_camera_forward(poses)
        reveal_progress = assign_points_to_poses(xyz, cam_positions, cam_forwards)
    elif reveal_mode == "all":
        print("Reveal mode: all points visible in every frame")
    else:
        print(f"Reveal mode: {reveal_mode} (near={near_depth:.2f}, max_depth={max_depth:.2f})")

    seen_mask = np.zeros(len(xyz), dtype=bool)

    def reset_reveal_state():
        if reveal_mode == "accumulate_frustum":
            seen_mask[:] = False

    def get_render_mask(pose_idx: int) -> np.ndarray:
        if reveal_mode == "all":
            return np.ones(len(xyz), dtype=bool)
        if reveal_mode == "trajectory":
            return get_progressive_mask(reveal_progress, pose_idx, n_poses)

        frustum_mask = get_camera_frustum_mask(
            xyz,
            poses[pose_idx],
            intrinsic,
            width,
            height,
            max_depth=max_depth,
            near_depth=near_depth,
            chunk_size=chunk_size,
        )
        if reveal_mode == "frustum":
            return frustum_mask

        seen_mask[:] |= frustum_mask
        return seen_mask.copy()

    # Setup renderer
    print("Setting up offscreen renderer...")
    renderer = rendering.OffscreenRenderer(width, height)
    configure_offscreen_renderer(renderer, rendering, bg_color)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size

    # Compute viewpoints
    use_extrinsic_camera = False

    if view_mode == "firstperson":
        use_extrinsic_camera = True
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        )
        extrinsics = opencv_c2w_to_o3d_extrinsic(render_poses)
        print("  Using firstperson mode")
        if render_pose_path:
            print(f"  Render viewpoints sourced directly from: {render_pose_path}")
        elif (
            render_pose_lag > 0
            or render_back_offset > 0
            or render_side_offset > 0
            or render_up_offset > 0
            or render_lookdown_offset > 0
        ):
            print(
                "  Render viewport "
                f"lag={render_pose_lag}, back={render_back_offset:.2f}, "
                f"side={render_side_offset:.2f}, up={render_up_offset:.2f}, "
                f"lookahead={render_lookahead:.2f}, "
                f"lookdown={render_lookdown_offset:.2f}"
            )
        print(
            f"  Intrinsic: fx={intrinsic[0, 0]:.1f}, fy={intrinsic[1, 1]:.1f}, cx={intrinsic[0, 2]:.1f}, cy={intrinsic[1, 2]:.1f}"
        )
    elif view_mode == "follow":
        eyes, lookats, ups = compute_follow_viewpoints(
            poses,
            offset_back=offset_back,
            offset_up=offset_up,
            lookahead=lookahead,
            smooth_window=smooth_window,
        )
    elif view_mode == "follow_pc":
        print("  Computing follow camera from point cloud trajectory...")
        if reveal_mode == "trajectory":
            eyes, lookats, ups = compute_follow_viewpoints_from_pointcloud(
                reveal_progress,
                xyz,
                n_poses,
                frame_indices,
                offset_back=offset_back,
                offset_up=offset_up,
                lookahead=lookahead,
                smooth_window=smooth_window,
            )
        else:
            reset_reveal_state()
            eyes, lookats, ups = compute_follow_viewpoints_from_masks(
                xyz,
                frame_indices,
                get_render_mask,
                offset_back=offset_back,
                offset_up=offset_up,
                lookahead=lookahead,
                smooth_window=smooth_window,
            )
            reset_reveal_state()
    elif view_mode == "bird_eye":
        eye, lookat, up = compute_bird_eye_viewpoint(poses)
        eyes = np.tile(eye, (n_poses, 1))
        lookats = np.tile(lookat, (n_poses, 1))
        ups = np.tile(up, (n_poses, 1))
    else:
        raise ValueError(f"Unknown view_mode: {view_mode}")

    # Save viewpoint config
    viewpoint_interp = max(int(viewpoint_interp), 1)
    base_viewpoints = []
    for fi, idx in enumerate(frame_indices):
        vp = {"pose_idx": int(idx)}
        if use_extrinsic_camera:
            vp["render_pose_idx"] = int(render_pose_indices[fi])
            vp["extrinsic"] = extrinsics[fi].tolist()
        elif view_mode == "follow_pc":
            # follow_pc has per-frame viewpoints (not per-pose)
            vp["eye"] = eyes[fi].tolist()
            vp["lookat"] = lookats[fi].tolist()
            vp["up"] = ups[fi].tolist()
        else:
            vp["eye"] = eyes[idx].tolist()
            vp["lookat"] = lookats[idx].tolist()
            vp["up"] = ups[idx].tolist()
        base_viewpoints.append(vp)

    viewpoints = interpolate_viewpoints(
        base_viewpoints,
        factor=viewpoint_interp,
        use_extrinsic_camera=use_extrinsic_camera,
    )
    if viewpoint_interp > 1:
        print(
            "Interpolating render viewpoints: "
            f"factor={viewpoint_interp}, base={len(base_viewpoints)}, dense={len(viewpoints)}"
        )

    config = {
        "width": width,
        "height": height,
        "fov_y_deg": fov,
        "bg_color": bg_color,
        "frame_step": frame_step,
        "n_poses": n_poses,
        "n_render_poses": n_render_poses if render_pose_source is not None else n_poses,
        "n_base_frames": len(base_viewpoints),
        "n_frames": len(viewpoints),
        "viewpoint_interp": viewpoint_interp,
        "view_mode": view_mode,
        "reveal_mode": reveal_mode,
        "max_depth": max_depth,
        "near_depth": near_depth,
        "render_pose_lag": render_pose_lag,
        "render_back_offset": render_back_offset,
        "render_side_offset": render_side_offset,
        "render_up_offset": render_up_offset,
        "render_lookahead": render_lookahead,
        "render_lookdown_offset": render_lookdown_offset,
        "color_gamma": color_gamma,
        "color_gain": color_gain,
        "color_saturation": color_saturation,
        "color_contrast": color_contrast,
        "render_pose_path": render_pose_path,
        "render_pose_in_xyz_frame": render_pose_in_xyz_frame,
        "viewpoints": viewpoints,
    }
    if use_extrinsic_camera:
        config["intrinsic"] = intrinsic.tolist()

    with open(viewpoint_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved viewpoint config to {viewpoint_path}")

    # Render frames (point cloud ONLY)
    print(f"Rendering {len(viewpoints)} point cloud frames...")
    reset_reveal_state()

    for frame_i, vp in enumerate(tqdm(viewpoints)):
        pose_idx = int(vp["pose_idx"])
        mask = get_render_mask(pose_idx)
        vis_xyz = xyz[mask]
        vis_rgb = rgb[mask]

        if len(vis_xyz) == 0:
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis_xyz)
        pcd.colors = o3d.utility.Vector3dVector(vis_rgb)

        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("pointcloud", pcd, mat)

        if use_extrinsic_camera:
            renderer.setup_camera(o3d_intrinsic, np.array(vp["extrinsic"], dtype=np.float64))
        else:
            renderer.setup_camera(
                fov,
                np.array(vp["lookat"], dtype=np.float64),
                np.array(vp["eye"], dtype=np.float64),
                np.array(vp["up"], dtype=np.float64),
            )

        img = renderer.render_to_image()
        out_path = os.path.join(output_dir, f"frame_{frame_i:05d}.png")
        o3d.io.write_image(out_path, img)

    print(f"Done! {len(viewpoints)} frames saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render point cloud frames (no cameras)")
    parser.add_argument("--ply_path", required=True)
    parser.add_argument("--pose_path", required=True)
    parser.add_argument("--output_dir", default="data/output/kitti/07/progressive/POINT")
    parser.add_argument("--pred_pose_path", default="")
    parser.add_argument(
        "--render_pose_path",
        default="",
        help="Optional render-only pose txt path; when set, only the viewport path changes",
    )
    parser.add_argument("--align_gt_to_whole", type=int, default=0)
    parser.add_argument("--normalize_first", type=int, default=-1)
    parser.add_argument(
        "--pose_in_xyz_frame",
        type=int,
        default=-1,
        help="Whether pose_path is already in xyz/whole coordinates: -1 auto, 0 no, 1 yes",
    )
    parser.add_argument(
        "--render_pose_in_xyz_frame",
        type=int,
        default=1,
        help="Whether render_pose_path is already in xyz/whole coordinates: -1 auto, 0 no, 1 yes",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--bg_color", default="#ffffff")
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--voxel_size", type=float, default=0.0)
    parser.add_argument(
        "--view_mode",
        choices=["firstperson", "follow", "follow_pc", "bird_eye"],
        default="follow_pc",
    )
    parser.add_argument("--offset_back", type=float, default=3.0)
    parser.add_argument("--offset_up", type=float, default=0.4)
    parser.add_argument("--lookahead", type=float, default=5.0)
    parser.add_argument("--smooth_window", type=int, default=51)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument(
        "--fx", type=float, default=0.0, help="Focal length x (overrides fov in firstperson mode)"
    )
    parser.add_argument(
        "--fy", type=float, default=0.0, help="Focal length y (overrides fov in firstperson mode)"
    )
    parser.add_argument(
        "--cx", type=float, default=0.0, help="Principal point x (default: width/2)"
    )
    parser.add_argument(
        "--cy", type=float, default=0.0, help="Principal point y (default: height/2)"
    )
    parser.add_argument(
        "--reveal_mode",
        choices=["accumulate_frustum", "frustum", "trajectory", "all"],
        default="accumulate_frustum",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=60.0,
        help="Maximum reveal depth in camera space; <=0 disables far clipping",
    )
    parser.add_argument(
        "--near_depth",
        type=float,
        default=0.1,
        help="Minimum reveal depth in camera space",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=262144,
        help="Point chunk size for frustum masking",
    )
    parser.add_argument(
        "--render_pose_lag",
        type=int,
        default=5,
        help="Lag render viewpoint behind the current input pose by N frames",
    )
    parser.add_argument(
        "--render_back_offset",
        type=float,
        default=1.25,
        help="Additional backward offset along the render pose forward axis",
    )
    parser.add_argument(
        "--render_side_offset",
        type=float,
        default=0.4,
        help="Lateral offset used to reveal more 3D frustum structure",
    )
    parser.add_argument(
        "--render_up_offset",
        type=float,
        default=1.0,
        help="Raise the render viewport above the lagged pose",
    )
    parser.add_argument(
        "--render_lookahead",
        type=float,
        default=5.0,
        help="Forward lookahead used to build the render viewport pose",
    )
    parser.add_argument(
        "--render_lookdown_offset",
        type=float,
        default=0.04,
        help="Downward look-at offset used to pitch the render viewport",
    )
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
    parser.add_argument(
        "--viewpoint_interp",
        type=int,
        default=1,
        help="Render-viewpoint interpolation factor; 2 inserts one extra render view between base viewpoints",
    )
    args = parser.parse_args()

    render_frames(**vars(args))


if __name__ == "__main__":
    main()
