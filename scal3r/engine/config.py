from typing import Any
from os import PathLike
from os.path import abspath, dirname, exists, expanduser, isabs, join

from . import io
from .path import check_file_exist, get_release_root
from scal3r.utils.base_utils import DotDict, to_dot_dict


def _resolve_config_path(config_ref: str | PathLike[str], parent_dir: str) -> str:
    path = expanduser(str(config_ref))
    if isabs(path) and exists(path):
        return abspath(path)

    for root in (parent_dir, get_release_root()):
        candidate = abspath(path if isabs(path) else join(root, path))
        if exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not resolve config path: {config_ref}")


def _merge_dicts(base: Any, update: Any) -> Any:
    if isinstance(base, dict) and isinstance(update, dict):
        merged = dict(base)
        for key, value in update.items():
            if isinstance(value, dict):
                value = dict(value)
                if value.pop("_delete_", False):
                    merged[key] = value
                    continue
            if key in merged:
                merged[key] = _merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged
    return update


def _load_yaml(path: str) -> dict[str, Any]:
    check_file_exist(path)
    data = io.load(path)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def load_config(config_path: str | PathLike[str]) -> DotDict:
    config_path = _resolve_config_path(config_path, abspath("."))
    current = _load_yaml(config_path)

    base_refs = current.pop("configs", [])
    if isinstance(base_refs, (str, PathLike)):
        base_refs = [base_refs]

    merged: dict[str, Any] = {}
    for base_ref in base_refs:
        base_cfg = load_config(_resolve_config_path(base_ref, dirname(config_path)))
        merged = _merge_dicts(merged, dict(base_cfg))

    merged = _merge_dicts(merged, current)
    merged["__config_path__"] = str(config_path)
    return to_dot_dict(merged)
