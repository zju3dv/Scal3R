#!/usr/bin/env python3
"""Composite point-cloud-only and camera-only frames, then export MP4.

The camera layer rendered by `render_cameras.py` already has image-quad
opacity baked into its RGB values. The default behavior here is therefore
to directly overwrite the point-cloud frame with the non-background camera
pixels, with a light mask feather to avoid overly hard edges, rather than
trying to reconstruct per-pixel alpha from textured RGB.

Usage:
    python scripts/visualize/geometry/utils/composite_frames.py \
        --pointcloud_dir data/output/kitti/07/progressive/POINT \
        --camera_dir data/output/kitti/07/progressive/CAM2D \
        --output_dir data/output/kitti/07/progressive/COMBO
"""

import os
import sys
import glob
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter


def estimate_background_rgb(cam_arr: np.ndarray, patch: int = 8) -> np.ndarray:
    height, width = cam_arr.shape[:2]
    patch_h = min(patch, height)
    patch_w = min(patch, width)
    corners = [
        cam_arr[:patch_h, :patch_w],
        cam_arr[:patch_h, -patch_w:],
        cam_arr[-patch_h:, :patch_w],
        cam_arr[-patch_h:, -patch_w:],
    ]
    samples = np.concatenate([corner.reshape(-1, 3) for corner in corners], axis=0)
    return np.median(samples, axis=0).astype(np.uint8)


def build_camera_mask(cam_arr: np.ndarray, bg_rgb: np.ndarray, threshold: int) -> np.ndarray:
    diff = np.abs(cam_arr.astype(np.int16) - bg_rgb.astype(np.int16))
    return diff.max(axis=-1) > threshold


def composite_premultiplied(
    pc_arr: np.ndarray,
    cam_arr: np.ndarray,
    bg_rgb: np.ndarray,
    threshold: int,
) -> np.ndarray:
    pc = pc_arr.astype(np.float32) / 255.0
    cam = cam_arr.astype(np.float32) / 255.0
    bg = bg_rgb.astype(np.float32) / 255.0

    alpha = np.clip((cam - bg) / np.clip(1.0 - bg, 1e-6, None), 0.0, 1.0).max(axis=-1)
    alpha[~build_camera_mask(cam_arr, bg_rgb, threshold)] = 0.0

    out = cam + (1.0 - alpha[..., None]) * (pc - bg)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def composite_binary_mask(
    pc_arr: np.ndarray,
    cam_arr: np.ndarray,
    bg_rgb: np.ndarray,
    threshold: int,
) -> np.ndarray:
    mask = build_camera_mask(cam_arr, bg_rgb, threshold)
    out = pc_arr.copy()
    out[mask] = cam_arr[mask]
    return out


def composite_feather_mask(
    pc_arr: np.ndarray,
    cam_arr: np.ndarray,
    bg_rgb: np.ndarray,
    threshold: int,
    feather_radius: float,
) -> np.ndarray:
    mask = build_camera_mask(cam_arr, bg_rgb, threshold).astype(np.uint8) * 255
    if feather_radius > 0:
        mask_img = Image.fromarray(mask, mode="L").filter(ImageFilter.GaussianBlur(feather_radius))
        alpha = np.asarray(mask_img, dtype=np.float32) / 255.0
    else:
        alpha = mask.astype(np.float32) / 255.0

    pc = pc_arr.astype(np.float32)
    cam = cam_arr.astype(np.float32)
    out = cam * alpha[..., None] + pc * (1.0 - alpha[..., None])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def composite_embedded_alpha_mask(
    pc_arr: np.ndarray,
    cam_rgb_arr: np.ndarray,
    cam_alpha_arr: np.ndarray,
    threshold: int,
) -> np.ndarray:
    mask = cam_alpha_arr.astype(np.uint8) > int(np.clip(threshold, 0, 255))
    out = pc_arr.copy()
    out[mask] = cam_rgb_arr[mask]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def resolve_local_max_filter_size(size: int) -> int:
    size = max(int(size), 1)
    if size % 2 == 0:
        size += 1
    return size


def erode_binary_mask(mask: np.ndarray, filter_size: int) -> np.ndarray:
    filter_size = resolve_local_max_filter_size(filter_size)
    if filter_size <= 1:
        return mask.copy()
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    eroded = np.asarray(mask_img.filter(ImageFilter.MinFilter(filter_size)), dtype=np.uint8)
    return eroded > 127


def estimate_local_opacity_and_coverage(
    cam_alpha_arr: np.ndarray,
    threshold: int,
    local_max_filter_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.clip(cam_alpha_arr.astype(np.float32) / 255.0, 0.0, 1.0)
    filter_size = resolve_local_max_filter_size(local_max_filter_size)
    if filter_size > 1:
        alpha_img = Image.fromarray(cam_alpha_arr.astype(np.uint8))
        local_max = (
            np.asarray(alpha_img.filter(ImageFilter.MaxFilter(filter_size)), dtype=np.float32) / 255.0
        )
    else:
        local_max = alpha.copy()

    local_max = np.maximum(local_max, alpha)
    threshold_alpha = float(np.clip(threshold, 0, 255)) / 255.0
    valid = local_max > max(threshold_alpha, 1e-6)

    coverage = np.zeros_like(alpha)
    coverage[valid] = np.clip(alpha[valid] / np.clip(local_max[valid], 1e-6, None), 0.0, 1.0)
    return alpha, local_max, coverage


def recover_black_bg_foreground(
    cam_rgb_arr: np.ndarray,
    coverage: np.ndarray,
    coverage_floor: float,
) -> np.ndarray:
    coverage_floor = float(np.clip(coverage_floor, 1e-4, 1.0))
    safe_coverage = np.maximum(coverage, coverage_floor)
    foreground = cam_rgb_arr.astype(np.float32) / safe_coverage[..., None]
    return np.clip(np.round(foreground), 0, 255).astype(np.uint8)


def composite_embedded_alpha_localmax_overwrite(
    pc_arr: np.ndarray,
    cam_rgb_arr: np.ndarray,
    cam_alpha_arr: np.ndarray,
    threshold: int,
    local_max_filter_size: int,
    coverage_floor: float,
) -> np.ndarray:
    alpha, _, coverage = estimate_local_opacity_and_coverage(
        cam_alpha_arr,
        threshold,
        local_max_filter_size,
    )
    mask = alpha > (float(np.clip(threshold, 0, 255)) / 255.0)
    foreground = recover_black_bg_foreground(cam_rgb_arr, coverage, coverage_floor)
    out = pc_arr.copy()
    out[mask] = foreground[mask]
    return out


def composite_embedded_alpha_localmax_blend(
    pc_arr: np.ndarray,
    cam_rgb_arr: np.ndarray,
    cam_alpha_arr: np.ndarray,
    threshold: int,
    local_max_filter_size: int,
    coverage_floor: float,
) -> np.ndarray:
    alpha, _, coverage = estimate_local_opacity_and_coverage(
        cam_alpha_arr,
        threshold,
        local_max_filter_size,
    )
    coverage = coverage.copy()
    coverage[alpha <= (float(np.clip(threshold, 0, 255)) / 255.0)] = 0.0
    foreground = recover_black_bg_foreground(cam_rgb_arr, coverage, coverage_floor).astype(np.float32)
    pc = pc_arr.astype(np.float32)
    out = foreground * coverage[..., None] + pc * (1.0 - coverage[..., None])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def composite_embedded_alpha_edge_blend(
    pc_arr: np.ndarray,
    cam_rgb_arr: np.ndarray,
    cam_alpha_arr: np.ndarray,
    threshold: int,
    local_max_filter_size: int,
    coverage_floor: float,
    edge_filter_size: int,
) -> np.ndarray:
    alpha, _, coverage = estimate_local_opacity_and_coverage(
        cam_alpha_arr,
        threshold,
        local_max_filter_size,
    )
    threshold_alpha = float(np.clip(threshold, 0, 255)) / 255.0
    mask = alpha > threshold_alpha
    eroded_mask = erode_binary_mask(mask, edge_filter_size)
    edge_band = mask & ~eroded_mask

    out = pc_arr.copy()
    out[mask] = cam_rgb_arr[mask]

    if np.any(edge_band):
        edge_coverage = coverage.copy()
        edge_coverage[~edge_band] = 0.0
        foreground = recover_black_bg_foreground(
            cam_rgb_arr, edge_coverage, coverage_floor
        ).astype(np.float32)
        pc = pc_arr.astype(np.float32)
        edge_result = foreground * edge_coverage[..., None] + pc * (1.0 - edge_coverage[..., None])
        out[edge_band] = np.clip(np.round(edge_result[edge_band]), 0, 255).astype(np.uint8)

    return out


def composite_camera_layer(
    pc_arr: np.ndarray,
    cam_arr: np.ndarray,
    bg_rgb: np.ndarray,
    threshold: int,
    mode: str,
    feather_radius: float,
) -> np.ndarray:
    if mode == "premultiplied":
        return composite_premultiplied(pc_arr, cam_arr, bg_rgb, threshold)
    if mode == "binary_mask":
        return composite_binary_mask(pc_arr, cam_arr, bg_rgb, threshold)
    if mode == "feather_mask":
        return composite_feather_mask(pc_arr, cam_arr, bg_rgb, threshold, feather_radius)
    raise ValueError(f"Unknown composite_mode={mode}")


def composite_single_frame(
    pc_frame_path: str,
    cam_frame_path: str,
    out_path: str,
    camera_mask_threshold: int,
    composite_mode: str,
    mask_feather_radius: float,
    rgba_composite_mode: str,
    rgba_local_max_filter_size: int,
    rgba_coverage_floor: float,
    rgba_edge_filter_size: int,
):
    pc_arr = np.array(Image.open(pc_frame_path).convert("RGB"))
    cam_arr = np.array(Image.open(cam_frame_path))

    if cam_arr.ndim == 2:
        cam_arr = np.repeat(cam_arr[..., None], 3, axis=-1)

    if cam_arr.ndim == 3 and cam_arr.shape[-1] == 4:
        if rgba_composite_mode == "mask_overwrite":
            result = composite_embedded_alpha_mask(
                pc_arr, cam_arr[..., :3], cam_arr[..., 3], camera_mask_threshold
            )
        elif rgba_composite_mode == "localmax_overwrite":
            result = composite_embedded_alpha_localmax_overwrite(
                pc_arr,
                cam_arr[..., :3],
                cam_arr[..., 3],
                camera_mask_threshold,
                rgba_local_max_filter_size,
                rgba_coverage_floor,
            )
        elif rgba_composite_mode == "localmax_blend":
            result = composite_embedded_alpha_localmax_blend(
                pc_arr,
                cam_arr[..., :3],
                cam_arr[..., 3],
                camera_mask_threshold,
                rgba_local_max_filter_size,
                rgba_coverage_floor,
            )
        elif rgba_composite_mode == "edge_blend":
            result = composite_embedded_alpha_edge_blend(
                pc_arr,
                cam_arr[..., :3],
                cam_arr[..., 3],
                camera_mask_threshold,
                rgba_local_max_filter_size,
                rgba_coverage_floor,
                rgba_edge_filter_size,
            )
        else:
            raise ValueError(f"Unknown rgba_composite_mode={rgba_composite_mode}")
    else:
        cam_rgb_arr = cam_arr[..., :3]
        bg_rgb = estimate_background_rgb(cam_rgb_arr)
        result = composite_camera_layer(
            pc_arr,
            cam_rgb_arr,
            bg_rgb,
            camera_mask_threshold,
            composite_mode,
            mask_feather_radius,
        )

    Image.fromarray(result.astype(np.uint8)).save(out_path)


def detect_camera_frame_layout(cam_frame_path: str) -> str:
    cam_arr = np.array(Image.open(cam_frame_path))
    if cam_arr.ndim == 3 and cam_arr.shape[-1] == 4:
        return "rgba"
    return "rgb"


def composite_frames(
    pointcloud_dir: str,
    camera_dir: str,
    output_dir: str,
    output_video: str = "",
    fps: int = 30,
    camera_mask_threshold: int = 12,
    composite_mode: str = "feather_mask",
    mask_feather_radius: float = 1.0,
    rgba_composite_mode: str = "edge_blend",
    rgba_local_max_filter_size: int = 9,
    rgba_coverage_floor: float = 0.08,
    rgba_edge_filter_size: int = 7,
    num_workers: int = 0,
):
    """Composite camera-only frames onto point-cloud-only frames."""

    os.makedirs(output_dir, exist_ok=True)

    pc_frames = sorted(glob.glob(os.path.join(pointcloud_dir, "frame_*.png")))
    cam_frames = sorted(glob.glob(os.path.join(camera_dir, "frame_*.png")))

    if not pc_frames:
        print(f"Error: No point cloud frames found in {pointcloud_dir}")
        sys.exit(1)
    if not cam_frames:
        print(f"Error: No camera frames found in {camera_dir}")
        sys.exit(1)

    n_frames = min(len(pc_frames), len(cam_frames))
    print(f"Compositing {n_frames} frames... (mode={composite_mode})")
    frame_layout = detect_camera_frame_layout(cam_frames[0])
    if frame_layout == "rgba":
        print(f"Detected RGBA camera frames; using rgba_composite_mode={rgba_composite_mode}")

    if num_workers <= 0:
        cpu_count = os.cpu_count() or 1
        num_workers = min(8, max(1, cpu_count // 2))
    num_workers = max(int(num_workers), 1)
    print(f"Using num_workers={num_workers}")

    tasks = []
    for i in range(n_frames):
        out_path = os.path.join(output_dir, f"frame_{i:05d}.png")
        tasks.append(
            (
                pc_frames[i],
                cam_frames[i],
                out_path,
                camera_mask_threshold,
                composite_mode,
                mask_feather_radius,
                rgba_composite_mode,
                rgba_local_max_filter_size,
                rgba_coverage_floor,
                rgba_edge_filter_size,
            )
        )

    if num_workers == 1:
        for task in tqdm(tasks):
            composite_single_frame(*task)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(composite_single_frame, *task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()

    print(f"Composited {n_frames} frames to {output_dir}")

    # ─── Export MP4 ───────────────────────────────────────────────────────
    if not output_video:
        output_root = os.path.basename(os.path.dirname(output_dir.rstrip(os.sep))) or "composite"
        output_video = os.path.join(os.path.dirname(output_dir), f"{output_root}.mp4")

    frame_pattern = os.path.join(output_dir, "frame_%05d.png")
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

    print(f"Running ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}")
        sys.exit(1)

    print(f"Video saved to {output_video}")


def main():
    parser = argparse.ArgumentParser(
        description="Composite point cloud + camera frames, export MP4"
    )
    parser.add_argument("--pointcloud_dir", default="data/output/kitti/07/progressive/POINT")
    parser.add_argument("--camera_dir", default="data/output/kitti/07/progressive/CAM2D")
    parser.add_argument("--output_dir", default="data/output/kitti/07/progressive/COMBO")
    parser.add_argument("--output_video", default="")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera_mask_threshold", type=int, default=12)
    parser.add_argument(
        "--composite_mode",
        choices=["premultiplied", "binary_mask", "feather_mask"],
        default="feather_mask",
    )
    parser.add_argument("--mask_feather_radius", type=float, default=1.0)
    parser.add_argument(
        "--rgba_composite_mode",
        choices=["mask_overwrite", "localmax_overwrite", "localmax_blend", "edge_blend"],
        default="edge_blend",
    )
    parser.add_argument("--rgba_local_max_filter_size", type=int, default=9)
    parser.add_argument("--rgba_coverage_floor", type=float, default=0.08)
    parser.add_argument("--rgba_edge_filter_size", type=int, default=7)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Frame compositor worker count; <=0 resolves to a conservative auto value",
    )
    args = parser.parse_args()

    composite_frames(**vars(args))


if __name__ == "__main__":
    main()
