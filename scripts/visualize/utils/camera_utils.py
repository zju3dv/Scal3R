"""Camera utilities: pose loading, viewpoint calculation, frustum geometry."""

import os
import numpy as np
from copy import deepcopy
from os.path import basename, dirname, exists, join, splitext


def load_poses(path: str) -> np.ndarray:
    """Load camera poses from a text file.

    Each line contains either 16 floats (flattened 4x4) or 12 floats
    (flattened 3x4) camera-to-world matrix in row-major order.

    Returns:
        np.ndarray of shape (N, 4, 4).
    """
    raw = np.atleast_2d(np.loadtxt(path))
    n = raw.shape[0]
    if raw.shape[1] == 16:
        poses = raw.reshape(n, 4, 4)
    elif raw.shape[1] == 12:
        poses = raw.reshape(n, 3, 4)
        bottom = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=poses.dtype), (n, 1, 1))
        poses = np.concatenate([poses, bottom], axis=1)
    else:
        raise ValueError(f"Unsupported pose shape {raw.shape}, expected Nx16 or Nx12")
    return poses


def save_poses(path: str, poses: np.ndarray):
    dirpath = dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    np.savetxt(path, poses.reshape(len(poses), -1), fmt="%.8f")


def rigidify_c2w(pose: np.ndarray) -> np.ndarray:
    """Remove uniform Sim(3) scale from a c2w pose while preserving the center."""
    pose = np.array(pose, dtype=np.float64, copy=True)
    rot = pose[:3, :3]
    det = float(np.linalg.det(rot))
    scale = np.cbrt(abs(det)) if np.isfinite(det) else 1.0
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0

    rot = rot / scale
    U, _, Vt = np.linalg.svd(rot)
    rot = U @ Vt
    if np.linalg.det(rot) < 0:
        U[:, -1] *= -1.0
        rot = U @ Vt

    pose[:3, :3] = rot
    return pose


def rigidify_poses(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim == 2:
        return rigidify_c2w(poses)
    return np.stack([rigidify_c2w(pose) for pose in poses], axis=0)


def normalize_vector(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.copy()
    return vec / norm


def infer_xyz_pose_path(pose_path: str, pred_pose_path: str = "") -> str:
    pose_name = basename(pose_path)
    pose_dir = dirname(pose_path)
    stem, suffix = splitext(pose_name)
    if pose_name == "mat_xyz.txt":
        return pose_path
    if pose_name == "mat_pred_xyz.txt":
        return pose_path
    if pose_name == "mat_gt.txt":
        return join(pose_dir, "mat_xyz.txt")
    if pred_pose_path:
        return join(dirname(pred_pose_path), "mat_xyz.txt")
    return join(pose_dir, f"{stem}_xyz{suffix}")


def estimate_similarity_transform(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    with_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate a Sim(3) from source points to target points with Umeyama alignment."""
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    if source_xyz.shape != target_xyz.shape or source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError("source_xyz and target_xyz must both have shape (N, 3)")
    if len(source_xyz) < 3:
        raise ValueError(f"Need at least 3 points for similarity alignment, got {len(source_xyz)}")

    source_mean = source_xyz.mean(axis=0)
    target_mean = target_xyz.mean(axis=0)
    source_centered = source_xyz - source_mean[None]
    target_centered = target_xyz - target_mean[None]

    covariance = (target_centered.T @ source_centered) / float(len(source_xyz))
    u, singular_values, v_t = np.linalg.svd(covariance)
    sign = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(v_t) < 0.0:
        sign[-1, -1] = -1.0

    rotation = u @ sign @ v_t
    if with_scale:
        source_var = float(np.mean(np.sum(source_centered ** 2, axis=1)))
        scale = float(np.trace(np.diag(singular_values) @ sign) / max(source_var, 1e-12))
    else:
        scale = 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    return rotation, translation, scale


def estimate_pred_to_gt_sim3(gt_poses: np.ndarray, pred_poses: np.ndarray):
    """Estimate the Sim(3) that maps prediction-world points into `xyz.ply` / GT frame.

    The standalone release repo does not ship the full evaluator package, so
    this visualization helper estimates the 7DoF alignment directly from camera
    centers with Umeyama alignment.
    """
    gt_xyz = np.asarray(gt_poses[:, :3, 3], dtype=np.float64)
    pred_xyz = np.asarray(pred_poses[:, :3, 3], dtype=np.float64)
    rotation, translation, scale = estimate_similarity_transform(pred_xyz, gt_xyz, with_scale=True)

    rigid = np.eye(4, dtype=np.float64)
    rigid[:3, :3] = rotation
    rigid[:3, 3] = translation

    sim3 = np.eye(4, dtype=np.float64)
    sim3[:3, :3] = scale * rotation
    sim3[:3, 3] = translation
    meta = dict(
        align_scale=scale,
        align_transform=rigid,
        point_sim3=sim3,
    )
    return sim3, meta


def align_gt_poses_to_whole(gt_poses: np.ndarray, pred_poses: np.ndarray):
    """Move GT poses into the same world frame as `xyz.ply` / `whole.ply`."""
    sim3, meta = estimate_pred_to_gt_sim3(gt_poses, pred_poses)

    world = sim3 @ pred_poses[0] @ np.linalg.inv(sim3) @ np.linalg.inv(gt_poses[0])
    poses = world[None] @ gt_poses
    meta["world_transform"] = world
    return poses, meta


def align_pred_poses_to_whole(gt_poses: np.ndarray, pred_poses: np.ndarray):
    """Move predicted poses into the same world frame as `xyz.ply` / `whole.ply`."""
    _, meta = estimate_pred_to_gt_sim3(gt_poses, pred_poses)

    scale = float(meta["align_scale"])
    rigid = meta["align_transform"]

    poses = pred_poses.copy()
    poses[:, :3, :3] = rigid[:3, :3][None] @ pred_poses[:, :3, :3]
    poses[:, :3, 3] = (pred_poses[:, :3, 3] * scale) @ rigid[:3, :3].T + rigid[:3, 3][None]
    return poses, meta


def prepare_poses(
    pose_path: str,
    pred_pose_path: str = "",
    align_gt_to_whole: int = 0,
    normalize_first: int = -1,
    pose_in_xyz_frame: int = -1,
):
    """Load poses and optionally align GT poses into the `whole.ply` frame."""
    xyz_pose_path = infer_xyz_pose_path(pose_path, pred_pose_path)
    if pose_in_xyz_frame < 0:
        pose_is_xyz = basename(pose_path) in {"mat_xyz.txt", "mat_pred_xyz.txt"}
    else:
        pose_is_xyz = bool(pose_in_xyz_frame)
    if pose_is_xyz:
        xyz_pose_path = pose_path
    meta = dict(
        xyz_pose_path=xyz_pose_path,
        loaded_xyz_pose_cache=False,
        saved_xyz_pose_cache=False,
        pose_in_xyz_frame=pose_is_xyz,
    )

    if pose_is_xyz:
        poses = load_poses(pose_path)
        meta["loaded_xyz_pose_cache"] = True
    else:
        poses = load_poses(pose_path)

    if align_gt_to_whole and not pose_is_xyz:
        if exists(xyz_pose_path):
            poses = load_poses(xyz_pose_path)
            meta["loaded_xyz_pose_cache"] = True
        else:
            if not pred_pose_path:
                raise ValueError("`pred_pose_path` is required when `align_gt_to_whole` is enabled")

            pred_poses = load_poses(pred_pose_path)
            if len(pred_poses) != len(poses):
                raise ValueError(f"Pose count mismatch: gt={len(poses)} pred={len(pred_poses)}")

            poses, align_meta = align_gt_poses_to_whole(poses, pred_poses)
            meta.update(align_meta)
            save_poses(xyz_pose_path, poses)
            meta["saved_xyz_pose_cache"] = True

    if normalize_first < 0:
        normalize_first = 0 if (align_gt_to_whole or meta["loaded_xyz_pose_cache"]) else 1

    if normalize_first:
        poses = np.linalg.inv(poses[0]) @ poses
        meta["normalized_first"] = True
    else:
        meta["normalized_first"] = False

    return poses, meta


def get_camera_positions(poses: np.ndarray) -> np.ndarray:
    """Extract translation vectors from c2w poses. Shape (N, 3)."""
    return poses[:, :3, 3].copy()


def get_camera_forward(poses: np.ndarray) -> np.ndarray:
    """Extract forward (+z) direction from c2w poses. Shape (N, 3).

    Convention: camera looks along +z in camera space (COLMAP/KITTI convention),
    so forward in world space is R[:, 2] for each pose.
    """
    return rigidify_poses(poses)[:, :3, 2].copy()


def opencv_c2w_to_o3d_extrinsic(c2w: np.ndarray) -> np.ndarray:
    """Convert OpenCV c2w matrices to Open3D extrinsic (w2c) matrices.

    Open3D's OffscreenRenderer.setup_camera(intrinsic, extrinsic) expects
    the extrinsic in OpenCV convention (X-right, Y-down, Z-forward), so
    we just invert c2w to get w2c. No axis flip needed.

    Args:
        c2w: (4, 4) or (N, 4, 4) camera-to-world matrices in OpenCV convention.

    Returns:
        (4, 4) or (N, 4, 4) world-to-camera matrices (extrinsic).
    """
    c2w = rigidify_poses(c2w)
    single = c2w.ndim == 2
    if single:
        c2w = c2w[np.newaxis]

    extrinsics = np.linalg.inv(c2w)

    return extrinsics[0] if single else extrinsics


def rotation_matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    return normalize_vector(quat)


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = normalize_vector(quat)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def slerp_quaternion(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize_vector(q0)
    q1 = normalize_vector(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_vector((1.0 - t) * q0 + t * q1)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * float(t)
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = np.sin(theta_t) / sin_theta_0
    return normalize_vector(s0 * q0 + s1 * q1)


def interpolate_c2w_pose(pose0: np.ndarray, pose1: np.ndarray, t: float) -> np.ndarray:
    pose0 = rigidify_c2w(pose0)
    pose1 = rigidify_c2w(pose1)
    q0 = rotation_matrix_to_quaternion(pose0[:3, :3])
    q1 = rotation_matrix_to_quaternion(pose1[:3, :3])
    rot = quaternion_to_rotation_matrix(slerp_quaternion(q0, q1, t))
    trans = (1.0 - t) * pose0[:3, 3] + t * pose1[:3, 3]

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose


def interpolate_viewpoints(
    viewpoints: list[dict],
    factor: int = 1,
    use_extrinsic_camera: bool = False,
) -> list[dict]:
    factor = max(int(factor), 1)
    if factor <= 1 or len(viewpoints) <= 1:
        return [deepcopy(vp) for vp in viewpoints]

    dense_viewpoints: list[dict] = []
    for idx in range(len(viewpoints) - 1):
        vp0 = viewpoints[idx]
        vp1 = viewpoints[idx + 1]
        dense_viewpoints.append(deepcopy(vp0))

        for step in range(1, factor):
            t = step / float(factor)
            interp_vp = deepcopy(vp0)
            interp_vp["pose_idx"] = int(vp0["pose_idx"])
            interp_vp["interp_t"] = float(t)
            interp_vp["interp_next_pose_idx"] = int(vp1["pose_idx"])

            if "render_pose_idx" in vp0:
                interp_vp["render_pose_idx"] = int(vp0["render_pose_idx"])
            if "render_pose_idx" in vp1:
                interp_vp["interp_next_render_pose_idx"] = int(vp1["render_pose_idx"])

            if use_extrinsic_camera:
                c2w0 = np.linalg.inv(np.asarray(vp0["extrinsic"], dtype=np.float64))
                c2w1 = np.linalg.inv(np.asarray(vp1["extrinsic"], dtype=np.float64))
                interp_pose = interpolate_c2w_pose(c2w0, c2w1, t)
                interp_vp["extrinsic"] = opencv_c2w_to_o3d_extrinsic(interp_pose).tolist()
            else:
                eye0 = np.asarray(vp0["eye"], dtype=np.float64)
                eye1 = np.asarray(vp1["eye"], dtype=np.float64)
                lookat0 = np.asarray(vp0["lookat"], dtype=np.float64)
                lookat1 = np.asarray(vp1["lookat"], dtype=np.float64)
                up0 = np.asarray(vp0["up"], dtype=np.float64)
                up1 = np.asarray(vp1["up"], dtype=np.float64)

                interp_vp["eye"] = ((1.0 - t) * eye0 + t * eye1).tolist()
                interp_vp["lookat"] = ((1.0 - t) * lookat0 + t * lookat1).tolist()
                interp_vp["up"] = normalize_vector((1.0 - t) * up0 + t * up1).tolist()

            dense_viewpoints.append(interp_vp)

    dense_viewpoints.append(deepcopy(viewpoints[-1]))
    return dense_viewpoints


def make_intrinsic(fov_y_deg: float, width: int, height: int) -> np.ndarray:
    """Build a 3x3 camera intrinsic matrix from vertical FOV.

    Returns:
        (3, 3) intrinsic matrix with fx, fy, cx, cy.
    """
    fov_y_rad = np.radians(fov_y_deg)
    fy = height / (2.0 * np.tan(fov_y_rad / 2.0))
    fx = fy  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )
    return K


def get_camera_up(poses: np.ndarray) -> np.ndarray:
    """Extract up (+y) direction from c2w poses. Shape (N, 3)."""
    return rigidify_poses(poses)[:, :3, 1].copy()


def get_camera_right(poses: np.ndarray) -> np.ndarray:
    """Extract right (+x) direction from c2w poses. Shape (N, 3)."""
    return rigidify_poses(poses)[:, :3, 0].copy()


# Frustum geometry
def frustum_points(pose: np.ndarray, size: float = 0.06, aspect: float = 16.0 / 9.0) -> np.ndarray:
    """Return the 5 key points of a camera frustum in world coordinates.

    Points: [center, top-left, top-right, bottom-right, bottom-left]
    of the near plane.

    Args:
        pose: (4, 4) c2w matrix.
        size: half-height of the near plane rectangle.
        aspect: width / height ratio.

    Returns:
        (5, 3) array of world-space points.
    """
    pose = rigidify_c2w(pose)
    R = pose[:3, :3]
    t = pose[:3, 3]

    # Camera axes in world space
    right = R[:, 0]
    up = R[:, 1]
    forward = R[:, 2]  # camera looks along +z

    half_h = size
    half_w = size * aspect
    depth = size * 2.0  # distance from center to near plane

    center = t
    near_center = center + forward * depth

    tl = near_center + up * half_h - right * half_w
    tr = near_center + up * half_h + right * half_w
    br = near_center - up * half_h + right * half_w
    bl = near_center - up * half_h - right * half_w

    return np.stack([center, tl, tr, br, bl], axis=0)


# Edges of the frustum: (start_idx, end_idx) pairs for the 5-point frustum
FRUSTUM_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),  # center to 4 corners
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 1),  # near-plane rectangle
]


# Viewpoint computation
def smooth_trajectory(values: np.ndarray, window: int = 31) -> np.ndarray:
    """Apply a simple moving-average smoothing along axis 0.

    Args:
        values: (N, D) array.
        window: smoothing kernel size (must be odd).

    Returns:
        (N, D) smoothed array (same shape).
    """
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window) / window
    smoothed = np.empty_like(values)
    for d in range(values.shape[1]):
        smoothed[:, d] = np.convolve(values[:, d], kernel, mode="same")
    return smoothed


def compute_follow_viewpoints(
    poses: np.ndarray,
    offset_back: float = 2.0,
    offset_up: float = 1.5,
    lookahead: float = 3.0,
    smooth_window: int = 51,
) -> tuple:
    """Compute a smooth follow-camera trajectory.

    For each pose, the viewing camera is placed behind and above,
    looking ahead along the trajectory.

    Returns:
        (eyes, lookats, ups): each (N, 3) arrays.
    """
    positions = get_camera_positions(poses)
    forwards = get_camera_forward(poses)

    # Raw eye and lookat
    # KITTI convention: +Y points down (gravity), so "up" is -Y
    raw_eyes = positions - forwards * offset_back
    raw_eyes[:, 1] -= offset_up  # lift up = subtract Y (since +Y is down)

    raw_lookats = positions + forwards * lookahead

    # Smooth to prevent jitter
    eyes = smooth_trajectory(raw_eyes, window=smooth_window)
    lookats = smooth_trajectory(raw_lookats, window=smooth_window)
    ups = np.tile(np.array([0.0, -1.0, 0.0]), (len(poses), 1))  # -Y is up

    return eyes, lookats, ups


def compute_follow_viewpoints_from_pointcloud(
    reveal_progress: "np.ndarray",
    xyz: "np.ndarray",
    n_poses: int,
    frame_indices: list,
    offset_back: float = 5.0,
    offset_up: float = 3.0,
    lookahead: float = 10.0,
    smooth_window: int = 51,
) -> tuple:
    """Compute follow-camera trajectory from point cloud progressive reveal.

    Instead of following the camera poses (which may be misaligned with the
    point cloud), this computes the camera path from the centroid of visible
    points at each frame.

    Returns:
        (eyes, lookats, ups): each (N_frames, 3) arrays.
    """
    from scipy.ndimage import uniform_filter1d

    n_frames = len(frame_indices)
    centroids = np.zeros((n_frames, 3))
    forward_dirs = np.zeros((n_frames, 3))

    for fi, pose_idx in enumerate(frame_indices):
        # Points visible at this frame
        t = (pose_idx + 1) / n_poses
        t = min(t + 0.05, 1.0)
        mask = reveal_progress <= t

        visible_pts = xyz[mask]
        if len(visible_pts) == 0:
            centroids[fi] = centroids[max(0, fi - 1)]
            continue

        # Use the centroid of the NEWEST 20% of visible points as the "current position"
        # This tracks where the growth is happening
        t_prev = max(0, t - 0.15)
        recent_mask = (reveal_progress > t_prev) & (reveal_progress <= t)
        if recent_mask.sum() > 100:
            centroids[fi] = xyz[recent_mask].mean(axis=0)
        else:
            centroids[fi] = visible_pts.mean(axis=0)

    # Compute forward direction from centroid trajectory
    for fi in range(n_frames):
        look_idx = min(fi + max(1, n_frames // 20), n_frames - 1)
        forward_dirs[fi] = centroids[look_idx] - centroids[fi]
        norm = np.linalg.norm(forward_dirs[fi])
        if norm > 1e-6:
            forward_dirs[fi] /= norm
        else:
            forward_dirs[fi] = np.array([0, 0, 1])

    # Smooth centroids and forward directions
    if smooth_window > 1:
        for d in range(3):
            centroids[:, d] = uniform_filter1d(centroids[:, d], size=min(smooth_window, n_frames))
            forward_dirs[:, d] = uniform_filter1d(
                forward_dirs[:, d], size=min(smooth_window, n_frames)
            )
        # Re-normalize forward dirs
        norms = np.linalg.norm(forward_dirs, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        forward_dirs /= norms

    # Y-down convention (KITTI): up = -Y
    up_vec = np.array([0.0, -1.0, 0.0])

    eyes = centroids - forward_dirs * offset_back
    eyes[:, 1] -= offset_up  # lift camera (subtract Y since Y-down)

    lookats = centroids + forward_dirs * lookahead

    ups = np.tile(up_vec, (n_frames, 1))

    return eyes, lookats, ups


def compute_follow_viewpoints_from_masks(
    xyz: np.ndarray,
    frame_indices: list,
    mask_getter,
    offset_back: float = 5.0,
    offset_up: float = 3.0,
    lookahead: float = 10.0,
    smooth_window: int = 51,
) -> tuple:
    """Compute follow-camera viewpoints from per-frame visible-point masks."""
    from scipy.ndimage import uniform_filter1d

    n_frames = len(frame_indices)
    centroids = np.zeros((n_frames, 3))
    forward_dirs = np.zeros((n_frames, 3))
    prev_mask = None

    for fi, pose_idx in enumerate(frame_indices):
        mask = mask_getter(pose_idx)
        visible_pts = xyz[mask]
        if len(visible_pts) == 0:
            centroids[fi] = centroids[max(0, fi - 1)]
            prev_mask = mask.copy()
            continue

        if prev_mask is None:
            delta_mask = mask
        else:
            delta_mask = mask & ~prev_mask

        if delta_mask.sum() > 100:
            centroids[fi] = xyz[delta_mask].mean(axis=0)
        else:
            centroids[fi] = visible_pts.mean(axis=0)

        prev_mask = mask.copy()

    for fi in range(n_frames):
        look_idx = min(fi + max(1, n_frames // 20), n_frames - 1)
        forward_dirs[fi] = centroids[look_idx] - centroids[fi]
        norm = np.linalg.norm(forward_dirs[fi])
        if norm > 1e-6:
            forward_dirs[fi] /= norm
        else:
            forward_dirs[fi] = np.array([0.0, 0.0, 1.0])

    if smooth_window > 1:
        size = min(smooth_window, n_frames)
        for dim in range(3):
            centroids[:, dim] = uniform_filter1d(centroids[:, dim], size=size)
            forward_dirs[:, dim] = uniform_filter1d(forward_dirs[:, dim], size=size)

        norms = np.linalg.norm(forward_dirs, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        forward_dirs /= norms

    up_vec = np.array([0.0, -1.0, 0.0])

    eyes = centroids - forward_dirs * offset_back
    eyes[:, 1] -= offset_up
    lookats = centroids + forward_dirs * lookahead
    ups = np.tile(up_vec, (n_frames, 1))
    return eyes, lookats, ups


def compute_bird_eye_viewpoint(
    poses: np.ndarray,
    height: float = 15.0,
) -> tuple:
    """Compute a fixed bird's-eye viewpoint looking down at the trajectory center.

    Returns:
        (eye, lookat, up): each (3,) arrays.
    """
    positions = get_camera_positions(poses)
    center = positions.mean(axis=0)

    eye = center.copy()
    eye[1] -= height  # -Y is up in KITTI

    lookat = center.copy()
    up = np.array([0.0, 0.0, 1.0])  # Z-forward as "up" when looking down

    return eye, lookat, up


# 3D → 2D projection
def build_view_matrix(eye: np.ndarray, lookat: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a 4x4 view (world-to-camera) matrix (OpenGL convention).

    Returns:
        (4, 4) view matrix.
    """
    f = lookat - eye
    f = f / np.linalg.norm(f)

    s = np.cross(f, up)
    s = s / np.linalg.norm(s)

    u = np.cross(s, f)

    view = np.eye(4)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[0, 3] = -np.dot(s, eye)
    view[1, 3] = -np.dot(u, eye)
    view[2, 3] = np.dot(f, eye)

    return view


def build_projection_matrix(
    fov_y_deg: float, aspect: float, near: float = 0.01, far: float = 100.0
) -> np.ndarray:
    """Build a 4x4 perspective projection matrix (OpenGL convention).

    Returns:
        (4, 4) projection matrix.
    """
    fov_y = np.radians(fov_y_deg)
    t = np.tan(fov_y / 2.0)

    proj = np.zeros((4, 4))
    proj[0, 0] = 1.0 / (aspect * t)
    proj[1, 1] = 1.0 / t
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    return proj


def project_points_3d_to_2d(
    points_3d: np.ndarray,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Project 3D world points to 2D pixel coordinates.

    Args:
        points_3d: (N, 3) world coordinates.
        view_matrix: (4, 4) view matrix.
        proj_matrix: (4, 4) projection matrix.
        width, height: image dimensions.

    Returns:
        (N, 2) pixel coordinates (x, y). Points behind camera get NaN.
    """
    N = points_3d.shape[0]
    ones = np.ones((N, 1))
    pts_h = np.hstack([points_3d, ones])  # (N, 4)

    # World → camera
    pts_cam = (view_matrix @ pts_h.T).T  # (N, 4)

    # Camera → clip
    pts_clip = (proj_matrix @ pts_cam.T).T  # (N, 4)

    # Perspective divide
    w = pts_clip[:, 3:4]
    behind = w.ravel() <= 0
    w[w <= 0] = 1e-8  # avoid division by zero

    ndc = pts_clip[:, :3] / w  # (N, 3), range [-1, 1]

    # NDC → pixel
    px = (ndc[:, 0] * 0.5 + 0.5) * width
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height  # flip Y

    result = np.stack([px, py], axis=1)
    result[behind] = np.nan

    return result
