import sys
import subprocess
from typing import Any
from os.path import join
from dataclasses import dataclass

from scal3r.engine.path import get_release_root, resolve_release_path
from scal3r.utils.data_utils import ensure_dir, write_json, write_lines
from scal3r.dataloaders.datasets.image_folder_dataset import ImageFolderDataset

backend_module = "scal3r.pipelines.backend"


@dataclass
class InferenceRequest:
    config_path: str
    input_dir: str
    output_dir: str
    runtime_dir: str | None = None
    checkpoint: str | None = None
    device: str | None = None
    max_images: int | None = None
    preprocess_workers: int | None = None
    block_size: int | None = None
    overlap_size: int | None = None
    use_loop: int | None = None
    use_xyz_align: int | None = None
    max_align_points_per_frame: int | None = None
    pgo_workers: int | None = None
    test_use_amp: bool = False
    save_dpt: int | None = None
    save_xyz: int | None = None
    streaming_state: int | None = None
    offload_batches: int | None = None
    offload_outputs: int | None = None
    cleanup_offload: int | None = None
    offload_dir: str | None = None
    probe_dir: str | None = None
    stop_after_stage: str | None = None
    dry_run: bool = False


def _get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def run_inference(config: dict[str, Any], request: InferenceRequest) -> dict[str, Any]:
    data_cfg = _get_nested(config, "data", default={}) or {}
    model_cfg = _get_nested(config, "model", default={}) or {}
    release_root = get_release_root()

    dataset = ImageFolderDataset(
        input_dir=request.input_dir,
        image_patterns=tuple(
            pattern.strip()
            for pattern in data_cfg.get("image_patterns", "*.png,*.jpg,*.jpeg,*.bmp").split(",")
            if pattern.strip()
        ),
        max_images=request.max_images,
    )
    image_paths = dataset.list_images()

    output_dir = ensure_dir(request.output_dir)
    runtime_dir = ensure_dir(request.runtime_dir or join(output_dir, "runtime"))
    manifest_path = join(runtime_dir, "image_manifest.txt")
    plan_path = join(runtime_dir, "run_plan.json")
    write_lines(manifest_path, [str(path) for path in image_paths])

    command = [
        sys.executable,
        "-m",
        backend_module,
        "--config",
        str(resolve_release_path(request.config_path)),
        "--input_dir",
        str(request.input_dir),
        "--result_dir",
        str(output_dir),
        "--runtime_dir",
        str(runtime_dir),
        "--image_patterns",
        str(data_cfg.get("image_patterns", "*.png,*.jpg,*.jpeg,*.bmp")),
        "--preprocess_workers",
        str(request.preprocess_workers or data_cfg.get("preprocess_workers", 32)),
        "--block_size",
        str(request.block_size or data_cfg.get("block_size", 60)),
        "--overlap_size",
        str(request.overlap_size or data_cfg.get("overlap_size", 30)),
        "--loop_size",
        str(data_cfg.get("loop_size", 20)),
        "--use_loop",
        str(request.use_loop if request.use_loop is not None else data_cfg.get("use_loop", 1)),
        "--use_xyz_align",
        str(request.use_xyz_align if request.use_xyz_align is not None else data_cfg.get("use_xyz_align", 0)),
        "--pgo_workers",
        str(request.pgo_workers if request.pgo_workers is not None else data_cfg.get("pgo_workers", 32)),
        "--save_dpt",
        str(request.save_dpt if request.save_dpt is not None else data_cfg.get("save_dpt", 1)),
        "--save_xyz",
        str(request.save_xyz if request.save_xyz is not None else data_cfg.get("save_xyz", 1)),
        "--streaming_state",
        str(request.streaming_state if request.streaming_state is not None else data_cfg.get("streaming_state", 0)),
        "--offload_batches",
        str(request.offload_batches if request.offload_batches is not None else data_cfg.get("offload_batches", 0)),
        "--offload_outputs",
        str(request.offload_outputs if request.offload_outputs is not None else data_cfg.get("offload_outputs", 0)),
        "--cleanup_offload",
        str(request.cleanup_offload if request.cleanup_offload is not None else data_cfg.get("cleanup_offload", 1)),
    ]
    max_align_points_per_frame = (
        request.max_align_points_per_frame
        if request.max_align_points_per_frame is not None
        else data_cfg.get("max_align_points_per_frame", None)
    )
    if max_align_points_per_frame is not None:
        command.extend(["--max_align_points_per_frame", str(max_align_points_per_frame)])
    if request.checkpoint:
        command.extend(["--checkpoint", str(request.checkpoint)])
    if request.device:
        command.extend(["--device", request.device])
    if request.max_images:
        command.extend(["--max_images", str(request.max_images)])
    if request.test_use_amp:
        command.append("--test_use_amp")
    if request.offload_dir:
        command.extend(["--offload_dir", str(request.offload_dir)])
    if request.probe_dir:
        command.extend(["--probe_dir", str(request.probe_dir)])
    if request.stop_after_stage:
        command.extend(["--stop_after_stage", request.stop_after_stage])
    if data_cfg.get("loop_ckpt"):
        command.extend(["--loop_ckpt", str(resolve_release_path(data_cfg["loop_ckpt"]))])

    payload = {
        "status": "ready_to_execute" if not request.dry_run else "dry_run",
        "note": "Inference results default to data/result/custom/<tag>, with runtime artifacts under <result_dir>/runtime.",
        "config_path": str(request.config_path),
        "input_dir": str(request.input_dir),
        "output_dir": str(output_dir),
        "runtime_dir": str(runtime_dir),
        "checkpoint": str(request.checkpoint) if request.checkpoint else str(model_cfg.get("checkpoint", "")),
        "model_name": model_cfg.get("name", "scal3r"),
        "image_count": len(image_paths),
        "first_image": str(image_paths[0]) if image_paths else "",
        "last_image": str(image_paths[-1]) if image_paths else "",
        "block_size": request.block_size or data_cfg.get("block_size", 60),
        "overlap_size": request.overlap_size or data_cfg.get("overlap_size", 30),
        "preprocess_workers": request.preprocess_workers or data_cfg.get("preprocess_workers", 32),
        "use_loop": request.use_loop if request.use_loop is not None else data_cfg.get("use_loop", 1),
        "use_xyz_align": request.use_xyz_align if request.use_xyz_align is not None else data_cfg.get("use_xyz_align", 0),
        "max_align_points_per_frame": max_align_points_per_frame,
        "pgo_workers": request.pgo_workers if request.pgo_workers is not None else data_cfg.get("pgo_workers", 32),
        "save_dpt": request.save_dpt if request.save_dpt is not None else data_cfg.get("save_dpt", 1),
        "save_xyz": request.save_xyz if request.save_xyz is not None else data_cfg.get("save_xyz", 1),
        "streaming_state": request.streaming_state if request.streaming_state is not None else data_cfg.get("streaming_state", 0),
        "offload_batches": request.offload_batches if request.offload_batches is not None else data_cfg.get("offload_batches", 0),
        "offload_outputs": request.offload_outputs if request.offload_outputs is not None else data_cfg.get("offload_outputs", 0),
        "cleanup_offload": request.cleanup_offload if request.cleanup_offload is not None else data_cfg.get("cleanup_offload", 1),
        "offload_dir": str(request.offload_dir or join(runtime_dir, "offload")),
        "probe_dir": str(request.probe_dir) if request.probe_dir else "",
        "stop_after_stage": request.stop_after_stage or "",
        "backend_module": backend_module,
        "manifest_path": str(manifest_path),
        "plan_path": str(plan_path),
        "command": command,
    }
    write_json(plan_path, payload)

    if not request.dry_run:
        subprocess.run(command, check=True, cwd=release_root)

    return {
        "plan_path": str(plan_path),
        "manifest_path": str(manifest_path),
        "runtime_dir": str(runtime_dir),
        "image_count": len(image_paths),
        "executed": not request.dry_run,
    }
