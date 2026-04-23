# Visualization Tools

Geometry-related visualization scripts for [Scal3R](https://github.com/zju3dv/Scal3R). We release these scripts for public use, hope they are helpful for your research.

## Installation

Run the following commands from the repository root:

```bash
pip install -r scripts/visualize/requirements.txt
```

Extra system tools:

- `ffmpeg` for MP4 export

## Before You Start

The only question that really matters is this:

Are your point cloud and camera poses already in the same coordinate system?

- If yes, you can directly use the render and viewer scripts.
- If no, align them first, or use the alignment-related helper options.

The old `mat.txt`, `mat_gt.txt`, and `mat_xyz.txt` names are just historical [Scal3R](https://github.com/zju3dv/Scal3R) conventions. They are not a required public interface. If your own point cloud and pose file already share one world frame, just use them directly.

Pose files are plain-text OpenCV-style camera-to-world matrices:

- `N x 16` for flattened `4 x 4`
- `N x 12` for flattened `3 x 4`

## Choose A Script

| Need | Script |
| --- | --- |
| Render a progressive point-cloud flythrough | `geometry/render/render_points.py` |
| Render camera frustums with the exact same viewpoints | `geometry/render/render_cameras.py` |
| Render a full static scene from a saved viewpoint path | `geometry/render/render_scene_from_viewpoints.py` |
| Inspect the scene interactively and author a custom path | `geometry/viewer/open3d_viewer.py` |

Other helper scripts are kept in `geometry/utils` and `tools`. If you need alignment, frame compositing, PLY downsampling, or pose-only camera visualization, check those folders directly.

## Core Usage

#### `render_points.py`

Render point-cloud-only frames and save a shared `viewpoint_config.json`. This is usually the first stage of the pipeline.

```bash
python scripts/visualize/geometry/render/render_points.py \
    --ply_path data/result/custom/demo/xyz.ply \
    --pose_path data/result/custom/demo/mat.txt \
    --output_dir data/result/custom/demo/visualize/POINT
```

#### `render_cameras.py`

Render camera frustums only. It reuses the `viewpoint_config.json` from `render_points.py`, so the output frames are already aligned for compositing. The example below also enables textured camera-image quads, which is the usual demo-style setup when original RGB images are available.

```bash
python scripts/visualize/geometry/render/render_cameras.py \
    --pose_path data/result/custom/demo/mat.txt \
    --viewpoint_config data/result/custom/demo/visualize/POINT/viewpoint_config.json \
    --camera_image_dir data/result/custom/demo/images/00 \
    --render_camera_images 1 \
    --output_dir data/result/custom/demo/visualize/CAM2D
```

If you only want frustums without image quads, remove `--camera_image_dir` and `--render_camera_images 1`.

#### `render_scene_from_viewpoints.py`

Render a full static scene from an existing viewpoint path. Unlike `render_points.py`, it does not progressively reveal points.

```bash
python scripts/visualize/geometry/render/render_scene_from_viewpoints.py \
    --ply_path data/result/custom/demo/xyz.ply \
    --capture_pose_path data/result/custom/demo/mat.txt \
    --viewpoint_config data/result/custom/demo/visualize/POINT/viewpoint_config.json \
    --output_dir data/result/custom/demo/visualize/GLOBAL
```

#### `open3d_viewer.py`

Interactive Open3D viewer and camera-path editor for inspecting the scene, capturing keyframes, previewing a path, and exporting a new viewpoint path.

```bash
python scripts/visualize/geometry/viewer/open3d_viewer.py \
    --ply_path data/result/custom/demo/xyz.ply \
    --pose_path data/result/custom/demo/mat.txt \
    --output_dir data/result/custom/demo/visualize/custom_path
```

Hotkeys:

- `H`: print help
- `K`: capture current viewer pose as a keyframe
- `R`: remove the last keyframe
- `C`: clear keyframes
- `J` / `L`: jump to previous / next keyframe
- `P`: preview the interpolated path
- `O`: export the interpolated path

## Demo Pipeline

The demo-style geometry video is built in stages, not in one render pass:

1. Render points only with `render_points.py`.
2. Render cameras only with `render_cameras.py`.
3. Composite the two frame sequences with `geometry/utils/composite_frames.py`.
4. Encode the final frames with `tools/images_to_video.py`.

So the demo pipeline is exactly:

`render points` + `render cameras` + `alpha/mask composite` + `encode video`

End-to-end example:

```bash
python scripts/visualize/geometry/render/render_points.py \
    --ply_path data/result/custom/demo/xyz.ply \
    --pose_path data/result/custom/demo/mat.txt \
    --output_dir data/result/custom/demo/visualize/POINT

python scripts/visualize/geometry/render/render_cameras.py \
    --pose_path data/result/custom/demo/mat.txt \
    --viewpoint_config data/result/custom/demo/visualize/POINT/viewpoint_config.json \
    --output_dir data/result/custom/demo/visualize/CAM2D

python scripts/visualize/geometry/utils/composite_frames.py \
    --pointcloud_dir data/result/custom/demo/visualize/POINT \
    --camera_dir data/result/custom/demo/visualize/CAM2D \
    --output_dir data/result/custom/demo/visualize/COMBO

python scripts/visualize/tools/images_to_video.py \
    data/result/custom/demo/visualize/COMBO \
    20 \
    data/result/custom/demo/visualize/combo.mp4
```

## Notes

- `render_cameras.py` and `render_scene_from_viewpoints.py` expect a `viewpoint_config.json` produced by `render_points.py` or `open3d_viewer.py`.
- `render_cameras.py` can optionally render textured camera quads with `--camera_image_dir`.
