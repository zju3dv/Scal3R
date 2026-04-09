import os
import numpy as np
from os.path import join

from scal3r.utils.cam_utils import write_camera
from scal3r.utils.base_utils import DotDict as dotdict
from scal3r.utils.parallel_utils import parallel_execution
from scal3r.utils.data_utils import export_pts, save_image, write_json
from scal3r.utils.runtime_utils import StageRecorder, maybe_stop_after
from scal3r.utils.offload_utils import get_runtime_root, materialize_payload


def _collect_camera_results(processed, image_height: int, image_width: int, save_dpt: bool):
    n_frames = int(processed.output.c2w.shape[0])
    cameras = dotdict()
    scaled_dpt_map = processed.output.dpt_map.copy() if save_dpt else None

    for index in range(n_frames):
        c2w = processed.output.c2w[index].copy()
        scale = np.cbrt(np.linalg.det(c2w[:3, :3]))
        c2w[:3, :3] = c2w[:3, :3] / scale
        w2c = np.linalg.inv(c2w)
        cameras[f"{index:06d}"] = dotdict(
            H=image_height,
            W=image_width,
            K=processed.output.ixt[index],
            R=w2c[:3, :3],
            T=w2c[:3, 3],
        )
        if scaled_dpt_map is not None:
            scaled_dpt_map[index] = scaled_dpt_map[index] * scale

    return cameras, scaled_dpt_map, n_frames


def _save_depth_results(result_dir: str, scaled_dpt_map, render_height: int, render_width: int):
    dpt_paths = [
        join(result_dir, f"depths/{index:06d}.exr") for index in range(len(scaled_dpt_map))
    ]
    dpts = [dpt.reshape(render_height, render_width, 1) for dpt in scaled_dpt_map]
    parallel_execution(
        dpt_paths,
        dpts,
        action=save_image,
        sequential=False,
        num_workers=32,
        print_progress=True,
    )


def _save_block_points(result_dir: str, visualize, downsample_xyz_ratio: float):
    for index in range(len(visualize.block_xyz)):
        filename = join(result_dir, f"points/blocks/{index:03d}.ply")
        xyz = np.concatenate(visualize.block_xyz[index], axis=0)
        rgb = np.concatenate(visualize.block_rgb[index], axis=0)
        if len(xyz) == 0:
            continue
        sample_index = np.random.choice(
            len(xyz),
            size=max(1, int(len(xyz) * downsample_xyz_ratio)),
            replace=False,
        )
        export_pts(xyz[sample_index], rgb[sample_index], filename=filename)


def _save_world_masks(result_dir: str, visualize, n_frames: int):
    msk_paths = [join(result_dir, f"masks/{index:06d}.png") for index in range(n_frames)]
    parallel_execution(
        msk_paths,
        visualize.world_msk,
        action=save_image,
        sequential=False,
        num_workers=32,
        print_progress=True,
    )


def _save_world_points(result_dir: str, visualize, downsample_xyz_ratio: float):
    if not len(visualize.world_xyz):
        return

    filename = join(result_dir, "points/whole.ply")
    xyz = np.concatenate(visualize.world_xyz, axis=0)
    rgb = np.concatenate(visualize.world_rgb, axis=0)
    sample_index = np.random.choice(
        len(xyz),
        size=max(1, int(len(xyz) * downsample_xyz_ratio)),
        replace=False,
    )
    export_pts(xyz[sample_index], rgb[sample_index], filename=filename)
    np.save(
        join(result_dir, "points/whole_indices.npy"),
        np.array(sample_index, dtype=np.int32),
    )


def save_results(
    processed, batches, visualize, runtime, args, recorder: StageRecorder | None = None
):
    if recorder is not None:
        recorder.record(
            "save_results.begin",
            result_dir=args.result_dir,
            runtime_dir=args.runtime_dir,
        )

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(get_runtime_root(args), exist_ok=True)
    write_json(join(get_runtime_root(args), "runtime.json"), runtime)

    batch0 = materialize_payload(batches[0])
    image_height = batch0.H.item()
    image_width = batch0.W.item()
    render_height = batch0.meta.H[0].item()
    render_width = batch0.meta.W[0].item()
    del batch0

    cameras, scaled_dpt_map, n_frames = _collect_camera_results(
        processed, image_height, image_width, bool(args.save_dpt)
    )
    write_camera(cameras, args.result_dir)
    np.savetxt(
        join(args.result_dir, "mat.txt"), processed.output.c2w.reshape(n_frames, -1), fmt="%.6f"
    )

    if args.save_dpt and scaled_dpt_map is not None:
        _save_depth_results(args.result_dir, scaled_dpt_map, render_height, render_width)

    if args.save_xyz:
        _save_block_points(args.result_dir, visualize, args.downsample_xyz_ratio)
        _save_world_masks(args.result_dir, visualize, n_frames)
        _save_world_points(args.result_dir, visualize, args.downsample_xyz_ratio)

    if recorder is not None:
        recorder.record(
            "save_results.done",
            result_dir=args.result_dir,
            runtime_dir=args.runtime_dir,
            n_frames=int(n_frames),
            wrote_depths=bool(args.save_dpt),
            wrote_points=bool(args.save_xyz),
        )
        maybe_stop_after("save_results.done", args, recorder)
