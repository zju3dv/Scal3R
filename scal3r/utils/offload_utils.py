import os
import torch
import shutil
from os.path import dirname, exists, isdir, join

from scal3r.utils.data_utils import ensure_dir
from scal3r.utils.base_utils import DotDict as dotdict


def _runtime_root_path(args) -> str:
    return ensure_dir(args.runtime_dir)


def _resolve_offload_root_path(args) -> str:
    if args.offload_dir:
        return args.offload_dir
    return join(_runtime_root_path(args), "offload")


def _offload_root_path(args) -> str:
    return ensure_dir(_resolve_offload_root_path(args))


def _category_root(args, category: str) -> str:
    return ensure_dir(join(_offload_root_path(args), category))


def _is_offloaded_payload(payload) -> bool:
    return isinstance(payload, dict) and "_offload_path" in payload


def _payload_path(payload) -> str:
    return payload._offload_path


def get_runtime_root(args) -> str:
    return str(_runtime_root_path(args))


def get_offload_root(args) -> str:
    return str(_offload_root_path(args))


def get_offload_path(args, category: str, index: int) -> str:
    return join(_category_root(args, category), f"{index:04d}.pt")


def get_state_root(args, category: str) -> str:
    return _category_root(args, category)


def get_agg_state_path(args, block_index: int) -> str:
    return join(get_state_root(args, "agg_state"), f"{block_index:04d}.pt")


def get_dpt_state_path(args, block_index: int, layer_index: int) -> str:
    directory = ensure_dir(join(get_state_root(args, "dpt_state"), f"block_{block_index:04d}"))
    return join(directory, f"layer_{layer_index:04d}.pt")


def offload_payload(payload, path: str, kind: str):
    ensure_dir(dirname(path))
    torch.save(payload, path)
    return dotdict(_offload_path=path, _offload_kind=kind)


def materialize_payload(payload):
    if _is_offloaded_payload(payload):
        return torch.load(_payload_path(payload), map_location="cpu", weights_only=False)
    return payload


def remove_payload(payload):
    if _is_offloaded_payload(payload):
        path = _payload_path(payload)
        if exists(path):
            os.remove(path)


def materialize_tensor_dict(payloads: dict) -> dict:
    tensors = {}
    cache = {}
    for key, value in payloads.items():
        if _is_offloaded_payload(value):
            path = _payload_path(value)
            cache_key = str(path)
            if cache_key not in cache:
                cache[cache_key] = torch.load(path, map_location="cpu")
            tensors[key] = cache[cache_key]
        else:
            tensors[key] = value
    return tensors


def offload_batch_block(batch: dotdict, args, block_index: int):
    return offload_payload(batch, get_offload_path(args, "batches", block_index), "batch")


def offload_output_block(block_output: dotdict, args, block_index: int):
    return offload_payload(block_output, get_offload_path(args, "outputs", block_index), "output")


def use_streaming_state(args) -> bool:
    return bool(getattr(args, "streaming_state", 0))


def should_release_runtime_state(args) -> bool:
    return bool(use_streaming_state(args) or args.offload_batches or args.offload_outputs)


def offload_agg_state(state: dotdict, args, block_index: int):
    return offload_payload(state, get_agg_state_path(args, block_index), "agg_state")


def offload_dpt_state_layer(tensor: torch.Tensor, args, block_index: int, layer_index: int):
    return offload_payload(tensor, get_dpt_state_path(args, block_index, layer_index), "dpt_state")


def store_agg_state(state: dotdict, args, block_index: int):
    if use_streaming_state(args):
        return offload_agg_state(state, args, block_index)
    return state


def store_dpt_state_layer(tensor: torch.Tensor, args, block_index: int, layer_index: int):
    if use_streaming_state(args):
        return offload_dpt_state_layer(tensor, args, block_index, layer_index)
    return tensor


def persist_dpt_state(temp_output: dict, dpt_state_refs: dict, args, block_index: int):
    for layer_index, tensor in temp_output.items():
        dpt_state_refs[layer_index] = store_dpt_state_layer(
            tensor, args, block_index, int(layer_index)
        )


def clear_dpt_state(dpt_state_refs: dict):
    seen = set()
    for payload in dpt_state_refs.values():
        if _is_offloaded_payload(payload):
            path = str(payload._offload_path)
            if path in seen:
                continue
            seen.add(path)
            remove_payload(payload)
    dpt_state_refs.clear()


def cleanup_offload_root(args):
    if not args.cleanup_offload:
        return
    root = _resolve_offload_root_path(args)
    if isdir(root):
        shutil.rmtree(root)
