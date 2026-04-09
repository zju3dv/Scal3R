import gc
import os
import sys
import torch
import numpy as np
import time as wall_time
from os.path import join

from scal3r.utils.data_utils import ensure_dir, write_json


def _bytes_to_gb(value: float) -> float:
    return float(value) / (1024**3)


def _sanitize_stage_name(stage: str) -> str:
    return stage.replace("/", "_").replace(".", "_").replace(":", "_")


def scalarize(value):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item()
    return value


def get_process_memory_gb() -> float:
    try:
        import psutil  # type: ignore

        return _bytes_to_gb(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = float(rss) if sys.platform == "darwin" else float(rss) * 1024.0
        return _bytes_to_gb(rss_bytes)
    except Exception:
        return 0.0


def get_cuda_snapshot(device: str | torch.device | None) -> dict:
    if device is None:
        return {}

    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        return {}

    index = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    return {
        "device": str(device_obj),
        "allocated_gb": _bytes_to_gb(torch.cuda.memory_allocated(index)),
        "reserved_gb": _bytes_to_gb(torch.cuda.memory_reserved(index)),
        "max_allocated_gb": _bytes_to_gb(torch.cuda.max_memory_allocated(index)),
        "free_gb": _bytes_to_gb(free_bytes),
        "total_gb": _bytes_to_gb(total_bytes),
    }


class StageRecorder:
    def __init__(self, root: str | None, device: str | torch.device | None):
        self.root = ensure_dir(root) if root else None
        self.device = device
        self.counter = 0
        self.started_at = wall_time.time()

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def record(self, stage: str, **payload):
        if not self.enabled:
            return None

        self.counter += 1
        record = {
            "index": self.counter,
            "stage": stage,
            "elapsed_s": wall_time.time() - self.started_at,
            "rss_gb": get_process_memory_gb(),
            "cuda": get_cuda_snapshot(self.device),
        }
        record.update(payload)
        write_json(
            join(self.root, f"{self.counter:03d}_{_sanitize_stage_name(stage)}.json"),
            record
        )
        write_json(join(self.root, "latest.json"), record)
        return record


class StopAfterStage(RuntimeError):
    pass


def maybe_stop_after(stage: str, args, recorder: StageRecorder | None):
    if args.stop_after_stage and args.stop_after_stage == stage:
        if recorder is not None:
            recorder.record("stop_after_stage", target=stage)
        raise StopAfterStage(stage)


def release_memory(device: str | None = None):
    gc.collect()
    if device and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
