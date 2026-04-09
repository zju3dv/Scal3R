import json
import yaml
import pickle
from os import PathLike
from os.path import splitext
from io import BytesIO, StringIO
from typing import Optional, TextIO, Union

from .misc import is_list_of

FileLikeObject = Union[TextIO, StringIO, BytesIO]


def _infer_file_format(file, file_format: Optional[str] = None) -> str:
    if file_format is not None:
        return file_format
    if isinstance(file, (str, PathLike)):
        return splitext(str(file))[1].lstrip(".")
    raise ValueError("file_format must be specified for file-like objects")


def load(
    file: Union[str, PathLike[str], FileLikeObject],
    file_format: Optional[str] = None,
    **kwargs,
):
    file_format = _infer_file_format(file, file_format)
    if file_format == "json":
        if hasattr(file, "read"):
            return json.load(file, **kwargs)
        with open(str(file), "r", encoding="utf-8") as handle:
            return json.load(handle, **kwargs)
    if file_format in {"yaml", "yml"}:
        if hasattr(file, "read"):
            return yaml.safe_load(file)
        with open(str(file), "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    if file_format in {"pickle", "pkl"}:
        if hasattr(file, "read"):
            return pickle.load(file, **kwargs)
        with open(str(file), "rb") as handle:
            return pickle.load(handle, **kwargs)
    raise TypeError(f"Unsupported format: {file_format}")


def dump(
    obj,
    file: Optional[Union[str, PathLike[str], FileLikeObject]] = None,
    file_format: Optional[str] = None,
    **kwargs,
):
    file_format = _infer_file_format(file, file_format) if file is not None else file_format
    if file_format not in {"json", "yaml", "yml", "pickle", "pkl"}:
        raise TypeError(f"Unsupported format: {file_format}")

    if file is None:
        if file_format == "json":
            return json.dumps(obj, **kwargs)
        if file_format in {"yaml", "yml"}:
            return yaml.safe_dump(obj, **kwargs)
        return pickle.dumps(obj, **kwargs)

    if file_format == "json":
        if hasattr(file, "write"):
            json.dump(obj, file, **kwargs)
        else:
            with open(str(file), "w", encoding="utf-8") as handle:
                json.dump(obj, handle, **kwargs)
        return
    if file_format in {"yaml", "yml"}:
        dumped = yaml.safe_dump(obj, **kwargs)
        if hasattr(file, "write"):
            file.write(dumped)
        else:
            with open(str(file), "w", encoding="utf-8") as handle:
                handle.write(dumped)
        return
    if hasattr(file, "write"):
        pickle.dump(obj, file, **kwargs)
    else:
        with open(str(file), "wb") as handle:
            pickle.dump(obj, handle, **kwargs)


def register_handler(file_formats, **kwargs):
    if isinstance(file_formats, str):
        file_formats = [file_formats]
    if not is_list_of(file_formats, str):
        raise TypeError("file_formats must be a str or a list of str")

    def wrap(cls):
        return cls

    return wrap
