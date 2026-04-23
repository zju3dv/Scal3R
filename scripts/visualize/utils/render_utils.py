import os
import numpy as np
from PIL import Image

from utils.color_utils import rainbow_colormap
from utils.camera_utils import FRUSTUM_EDGES, frustum_points


def hex_to_float_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def configure_offscreen_renderer(renderer, rendering_module, bg_color: str):
    """Force a flat background in Open3D offscreen renders.

    Filament's post-processing can shift a nominal white clear color toward
    gray. Disable that path when the view API is available and fall back
    gracefully on older Open3D builds.
    """
    renderer.scene.set_background(np.array([*hex_to_float_rgb(bg_color), 1.0], dtype=np.float32))

    view = getattr(renderer.scene, "view", None)
    if view is None:
        return

    try:
        view.set_post_processing(False)
    except Exception:
        pass

    try:
        color_grading = rendering_module.ColorGrading(
            rendering_module.ColorGrading.Quality.ULTRA,
            rendering_module.ColorGrading.ToneMapping.LINEAR,
        )
        view.set_color_grading(color_grading)
    except Exception:
        pass


def make_dense_frustum_points(pose, size=2.0, aspect=16 / 9, samples=50):
    """Sample dense points along frustum edges for stable point rendering."""
    corners = frustum_points(pose, size=size, aspect=aspect)

    points = []
    for src, dst in FRUSTUM_EDGES:
        for frac in np.linspace(0, 1, samples):
            points.append(corners[src] * (1 - frac) + corners[dst] * frac)

    return np.asarray(points)


def build_frustum_pointcloud(
    poses,
    visible_indices,
    frustum_size,
    point_samples,
    history_size_step,
    aspect=16 / 9,
    size_scales=None,
):
    """Build a point-cloud visualization for the selected camera poses."""
    import open3d as o3d

    all_xyz = []
    all_rgb = []

    n_total = len(poses)
    colors_float = rainbow_colormap(np.linspace(0, 1, n_total, endpoint=False))

    n_visible = len(visible_indices)
    for order, vis_idx in enumerate(visible_indices):
        if size_scales is None:
            size_scale = 1.0 + history_size_step * max(n_visible - order - 1, 0)
        else:
            size_scale = size_scales[vis_idx]
        xyz = make_dense_frustum_points(
            poses[vis_idx],
            size=frustum_size * size_scale,
            aspect=aspect,
            samples=point_samples,
        )
        rgb = np.tile(colors_float[vis_idx], (len(xyz), 1))
        all_xyz.append(xyz)
        all_rgb.append(rgb)

    if len(visible_indices) >= 2:
        positions = poses[visible_indices, :3, 3]
        traj_xyz = []
        traj_rgb = []
        for idx in range(len(visible_indices) - 1):
            for frac in np.linspace(0, 1, 10):
                traj_xyz.append(positions[idx] * (1 - frac) + positions[idx + 1] * frac)
                traj_rgb.append(
                    colors_float[visible_indices[idx]] * (1 - frac)
                    + colors_float[visible_indices[idx + 1]] * frac
                )
        all_xyz.append(np.asarray(traj_xyz))
        all_rgb.append(np.asarray(traj_rgb))

    if not all_xyz:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_xyz))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_rgb))
    return pcd


def build_single_frustum_pointcloud(
    pose: np.ndarray,
    color: np.ndarray,
    frustum_size: float,
    point_samples: int,
    aspect: float = 16 / 9,
    size_scale: float = 1.0,
):
    import open3d as o3d

    xyz = make_dense_frustum_points(
        pose,
        size=frustum_size * size_scale,
        aspect=aspect,
        samples=point_samples,
    )
    rgb = np.tile(np.asarray(color, dtype=np.float64), (len(xyz), 1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def build_trajectory_segment_pointcloud(
    pose0: np.ndarray,
    pose1: np.ndarray,
    color0: np.ndarray,
    color1: np.ndarray,
    samples: int = 10,
):
    import open3d as o3d

    xyz = []
    rgb = []
    p0 = pose0[:3, 3]
    p1 = pose1[:3, 3]
    color0 = np.asarray(color0, dtype=np.float64)
    color1 = np.asarray(color1, dtype=np.float64)
    for frac in np.linspace(0.0, 1.0, samples):
        xyz.append(p0 * (1.0 - frac) + p1 * frac)
        rgb.append(color0 * (1.0 - frac) + color1 * frac)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb, dtype=np.float64))
    return pcd


def blend_rgb_to_bg(rgb: np.ndarray, opacity: float, bg_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    bg_rgb = np.asarray(bg_rgb, dtype=np.float64)
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity >= 0.999:
        return rgb.copy()
    return rgb * opacity + bg_rgb * (1.0 - opacity)


def find_image_files(image_dir: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    files = [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if os.path.isfile(os.path.join(image_dir, name))
        and os.path.splitext(name)[1].lower() in exts
    ]
    return files


def load_texture_array(image_path: str, max_dim: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    if max_dim > 0:
        scale = min(max_dim / max(width, 1), max_dim / max(height, 1), 1.0)
        if scale < 1.0:
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def make_texture_image(texture_array: np.ndarray, opacity: float, bg_rgb: np.ndarray):
    import open3d as o3d

    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity >= 0.999:
        arr = texture_array
    else:
        arr = texture_array.astype(np.float32)
        bg = (np.asarray(bg_rgb, dtype=np.float32) * 255.0)[None, None]
        arr = np.clip(np.round(arr * opacity + bg * (1.0 - opacity)), 0.0, 255.0).astype(np.uint8)
    # Open3D/Filament interprets image rows with the opposite vertical origin
    # from PIL/numpy images, so flip once here to keep camera images upright.
    arr = np.ascontiguousarray(np.flipud(arr))
    return o3d.geometry.Image(arr)


def build_camera_image_quad(
    pose: np.ndarray,
    aspect: float,
    frustum_size: float,
    image_path: str | None = None,
    max_texture_dim: int = 256,
    opacity: float = 1.0,
    texture_array: np.ndarray | None = None,
    bg_rgb: np.ndarray | None = None,
):
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    corners = frustum_points(pose, size=frustum_size, aspect=aspect)
    plane = corners[[1, 2, 3, 4]]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(plane.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [2, 1, 0],
                [3, 2, 0],
            ],
            dtype=np.int32,
        )
    )
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )

    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    if texture_array is None:
        if image_path is None:
            raise ValueError("Either `image_path` or `texture_array` must be provided")
        texture_array = load_texture_array(image_path, max_texture_dim)
    if bg_rgb is None:
        bg_rgb = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    material.albedo_img = make_texture_image(texture_array, opacity, bg_rgb)
    return mesh, material
