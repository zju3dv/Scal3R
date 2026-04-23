"""Point utilities: loading, downsampling, and reveal-mask helpers."""

import numpy as np


def load_pointcloud_arrays(path: str):
    """Load a PLY point cloud and return xyz and rgb arrays.

    Uses open3d for PLY parsing.

    Returns:
        xyz: (N, 3) float64 array of point positions.
        rgb: (N, 3) float64 array of colors in [0, 1].
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    return xyz, rgb


def adjust_point_colors(
    rgb: np.ndarray,
    gamma: float = 1.0,
    gain: float = 1.0,
    saturation: float = 1.0,
    contrast: float = 1.0,
):
    """Apply a simple display-oriented color remap to point colors.

    Args:
        rgb: (N, 3) float colors in [0, 1].
        gamma: Power-law remap. Values > 1 darken midtones, < 1 brighten.
        gain: Final multiplicative brightness scale.
        saturation: 1 keeps original saturation, 0 becomes grayscale.
        contrast: 1 keeps original contrast, > 1 increases contrast around 0.5.

    Returns:
        Adjusted RGB array in [0, 1].
    """
    rgb_adj = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)

    if gamma > 0 and gamma != 1.0:
        rgb_adj = np.power(rgb_adj, float(gamma))

    if saturation != 1.0:
        luma = np.sum(
            rgb_adj * np.array([0.2126, 0.7152, 0.0722], dtype=rgb_adj.dtype),
            axis=1,
            keepdims=True,
        )
        rgb_adj = luma + float(saturation) * (rgb_adj - luma)

    if contrast != 1.0:
        rgb_adj = (rgb_adj - 0.5) * float(contrast) + 0.5

    if gain != 1.0:
        rgb_adj = rgb_adj * float(gain)

    return np.clip(rgb_adj, 0.0, 1.0)


def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel_size: float = 0.02):
    """Downsample point cloud with voxel grid.

    Returns:
        (xyz_ds, rgb_ds) downsampled arrays.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    pcd_ds = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_ds.points), np.asarray(pcd_ds.colors)


def random_downsample(xyz: np.ndarray, rgb: np.ndarray, max_points: int = 0, seed: int = 0):
    """Randomly keep at most ``max_points`` points without replacement.

    This is useful for interactive viewers where preserving the coarse scene
    layout matters more than keeping every dense point. Set ``max_points <= 0``
    to disable the cap.
    """
    n_points = len(xyz)
    if max_points <= 0 or n_points <= max_points:
        return xyz, rgb

    rng = np.random.default_rng(int(seed))
    keep = rng.choice(n_points, size=int(max_points), replace=False)
    keep.sort()

    xyz_ds = xyz[keep]
    rgb_ds = rgb[keep] if len(rgb) == n_points else rgb
    return xyz_ds, rgb_ds


def get_camera_frustum_mask(
    xyz: np.ndarray,
    pose: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    max_depth: float = 60.0,
    near_depth: float = 0.1,
    chunk_size: int = 262144,
) -> np.ndarray:
    """Return a visibility mask for points inside the current camera frustum.

    The input `pose` is camera-to-world in OpenCV convention. A point is kept
    when it lies in front of the camera, within the configured depth range, and
    projects into the image plane.
    """
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)

    w2c = np.linalg.inv(pose)
    rot = w2c[:3, :3]
    trans = w2c[:3, 3]

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    if max_depth <= 0:
        max_depth = np.inf

    mask = np.zeros(len(xyz), dtype=bool)

    for start in range(0, len(xyz), chunk_size):
        end = min(start + chunk_size, len(xyz))
        pts = xyz[start:end]

        cam = pts @ rot.T + trans
        z = cam[:, 2]

        depth_ok = (z > near_depth) & (z < max_depth)
        if not np.any(depth_ok):
            continue

        x = cam[depth_ok, 0] / z[depth_ok]
        y = cam[depth_ok, 1] / z[depth_ok]

        u = fx * x + cx
        v = fy * y + cy

        pixel_ok = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        chunk_mask = np.zeros(end - start, dtype=bool)
        chunk_mask[np.flatnonzero(depth_ok)[pixel_ok]] = True
        mask[start:end] = chunk_mask

    return mask


def assign_points_to_poses(
    xyz: np.ndarray,
    cam_positions: np.ndarray,
    cam_forwards: np.ndarray,
) -> np.ndarray:
    """Assign each point a reveal-time based on trajectory progress.

    For each point, compute the cumulative arc-length along the trajectory
    at the nearest camera, then offset by the signed distance from that
    camera (along the trajectory tangent). Points ahead of the camera
    get a later reveal time; points behind get an earlier one.

    This ensures that at any pose i, ALL points around and behind the
    camera are visible (no black early-frames).

    Args:
        xyz: (M, 3) point positions.
        cam_positions: (N, 3) camera positions.
        cam_forwards: (N, 3) camera forward directions (unit vectors).

    Returns:
        (M,) array of float "reveal progress" in [0, 1] for each point.
        A point with progress <= t is visible when the camera is at
        fraction t of the trajectory.
    """
    from scipy.spatial import cKDTree

    N = len(cam_positions)

    # Step 1: cumulative arc-length along the trajectory
    deltas = np.linalg.norm(np.diff(cam_positions, axis=0), axis=1)
    arc_length = np.zeros(N)
    arc_length[1:] = np.cumsum(deltas)
    total_length = arc_length[-1]

    # Step 2: for each point, find nearest camera
    tree = cKDTree(cam_positions)
    _, nearest_idx = tree.query(xyz)

    # Step 3: signed offset along camera forward direction
    # Positive = point is ahead of camera, negative = behind
    diff = xyz - cam_positions[nearest_idx]
    fwd = cam_forwards[nearest_idx]
    signed_offset = np.sum(diff * fwd, axis=1)

    # Step 4: reveal time = arc_length at nearest camera + signed offset
    # Clamp and normalize to [0, 1]
    reveal_dist = arc_length[nearest_idx] + signed_offset
    reveal_progress = np.clip(reveal_dist / total_length, 0.0, 1.0)

    return reveal_progress


def get_progressive_mask(
    reveal_progress: np.ndarray,
    current_pose_idx: int,
    n_poses: int,
) -> np.ndarray:
    """Get a boolean mask for points visible at current_pose_idx.

    Args:
        reveal_progress: (M,) array from assign_points_to_poses, in [0, 1].
        current_pose_idx: current pose index.
        n_poses: total number of poses.

    Returns:
        (M,) boolean mask.
    """
    # Current progress fraction along the trajectory
    # Add a small lookahead so camera always sees some content ahead
    t = (current_pose_idx + 1) / n_poses
    # Add 5% lookahead so points slightly ahead are also visible
    t = min(t + 0.05, 1.0)
    return reveal_progress <= t
