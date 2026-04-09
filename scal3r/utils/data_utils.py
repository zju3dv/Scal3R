import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Any, Iterable, Union
from os.path import dirname, expanduser, splitext

from scal3r.utils.base_utils import DotDict


def ensure_dir(path: str) -> str:
    path = expanduser(path)
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, payload: dict) -> None:
    ensure_dir(dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_lines(path: str, lines: Iterable[str]) -> None:
    ensure_dir(dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, DotDict):
        return {key: _to_numpy(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_numpy(item) for item in value]
    return value


def to_cuda(batch: Any, device: str = "cuda", ignore_list: bool = False) -> Any:
    if isinstance(batch, (tuple, list)):
        if ignore_list:
            return batch
        return [to_cuda(item, device, ignore_list) for item in batch]
    if isinstance(batch, dict):
        return DotDict(
            {key: (to_cuda(value, device, ignore_list) if key != "meta" else value) for key, value in batch.items()}
        )
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, np.ndarray):
        return torch.as_tensor(batch).to(device, non_blocking=True)
    return batch


def _normalize_image(image: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image)
    if tensor.ndim >= 3 and tensor.shape[-1] >= 3:
        tensor[..., :3] = tensor[..., [2, 1, 0]]
    if torch.is_floating_point(tensor):
        return tensor.float().numpy()
    tensor = tensor / torch.iinfo(tensor.dtype).max
    return tensor.float().numpy()


def load_image_from_bytes(
    buffer: np.ndarray | memoryview | bytes | BytesIO | torch.Tensor,
    ratio: float = 1.0,
    normalize: bool = False,
    decode_flag: int = cv2.IMREAD_UNCHANGED,
    imp: str | None = None,
) -> np.ndarray:
    if isinstance(buffer, BytesIO):
        buffer = buffer.getvalue()
    if isinstance(buffer, (memoryview, bytes)):
        buffer = np.frombuffer(buffer, np.uint8)
    if isinstance(buffer, torch.Tensor):
        buffer = buffer.detach().cpu().numpy()
    buffer = np.asarray(buffer, dtype=np.uint8)

    image = cv2.imdecode(buffer, decode_flag)
    if image is None:
        raise ValueError(f"Failed to decode image from bytes: {imp or '<unknown>'}")
    if image.ndim == 2:
        image = image[..., None]
    if normalize:
        image = _normalize_image(image)
    if ratio != 1.0:
        height, width = image.shape[:2]
        image = cv2.resize(
            image,
            (int(width * ratio), int(height * ratio)),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _load_image_file(img_path: str, ratio: float = 1.0) -> np.ndarray:
    suffix = splitext(img_path)[1].lower()
    if suffix in {".jpg", ".jpeg", ".webp"}:
        image = Image.open(img_path)
        width, height = image.width, image.height
        draft = image.draft("RGB", (int(width * ratio), int(height * ratio)))
        image = np.asarray(image).copy()
        if ratio != 1.0 and (
            draft is None
            or draft[1][2] != int(width * ratio)
            or draft[1][3] != int(height * ratio)
        ):
            image = cv2.resize(
                image,
                (int(width * ratio), int(height * ratio)),
                interpolation=cv2.INTER_AREA,
            )
    else:
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        if image.ndim >= 3 and image.shape[-1] >= 3:
            image[..., :3] = image[..., [2, 1, 0]]
        if ratio != 1.0:
            height, width = image.shape[:2]
            image = cv2.resize(
                image,
                (int(width * ratio), int(height * ratio)),
                interpolation=cv2.INTER_AREA,
            )

    if image.ndim == 2:
        image = image[..., None]
    return image


def load_image(path: Union[str, np.ndarray], ratio: float = 1.0) -> np.ndarray:
    if isinstance(path, str):
        return _load_image_file(path, ratio)
    if isinstance(path, np.ndarray):
        return load_image_from_bytes(path, ratio)
    raise NotImplementedError("Supported path types are str and np.ndarray")


def save_image(
    img_path: str,
    img: np.ndarray | torch.Tensor,
    jpeg_quality: int = 75,
    png_compression: int = 9,
    save_dtype=np.uint8,
) -> bool:
    img = _to_numpy(img)
    img = np.asarray(img)
    if img.ndim == 4:
        img = np.concatenate(img, axis=0)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[0] < img.shape[-1] and img.shape[0] in {1, 3, 4}:
        img = np.transpose(img, (1, 2, 0))

    if img.dtype == np.bool_:
        img = img.astype(np.float32)
    elif np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    else:
        img = img.astype(np.float32, copy=False)

    suffix = splitext(img_path)[1].lower()
    if img.shape[-1] >= 3 and suffix not in {".exr", ".hdr"}:
        if not img.flags.writeable:
            img = img.copy()
        img[..., :3] = img[..., [2, 1, 0]]

    ensure_dir(dirname(img_path))
    if suffix == ".png":
        max_value = np.iinfo(save_dtype).max
        img = (img * max_value).clip(0, max_value).round().astype(save_dtype)
    elif suffix in {".jpg", ".jpeg", ".webp"}:
        img = (img[..., :3] * 255).clip(0, 255).round().astype(np.uint8)
    elif suffix == ".hdr":
        img = img[..., :3]
    elif suffix == ".exr":
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    return cv2.imwrite(
        img_path,
        img,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            jpeg_quality,
            cv2.IMWRITE_PNG_COMPRESSION,
            png_compression,
            cv2.IMWRITE_EXR_COMPRESSION,
            cv2.IMWRITE_EXR_COMPRESSION_PIZ,
        ],
    )


def export_pts(
    pts: torch.Tensor | np.ndarray,
    color: torch.Tensor | np.ndarray | None = None,
    normal: torch.Tensor | np.ndarray | None = None,
    scalars: dict[str, Any] | DotDict | None = None,
    filename: str = "default.ply",
    skip_color: bool = False,
    **_: Any,
) -> None:
    points = np.asarray(_to_numpy(pts), dtype=np.float32).reshape(-1, 3)
    colors = None
    normals = None
    scalar_items: list[tuple[str, np.ndarray]] = []

    if color is not None:
        colors = np.asarray(_to_numpy(color)).reshape(-1, 3)
        if np.issubdtype(colors.dtype, np.floating):
            colors = (colors.clip(0.0, 1.0) * 255).round().astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
    elif not skip_color:
        colors = (points.clip(0.0, 1.0) * 255).round().astype(np.uint8)

    if normal is not None:
        normals = np.asarray(_to_numpy(normal), dtype=np.float32).reshape(-1, 3)
        denom = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-13
        normals = normals / denom

    if scalars:
        for key, value in _to_numpy(dict(scalars)).items():
            scalar_items.append((str(key), np.asarray(value, dtype=np.float32).reshape(-1)))

    ensure_dir(dirname(filename))
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if normals is not None:
            handle.write("property float nx\nproperty float ny\nproperty float nz\n")
        for key, _ in scalar_items:
            handle.write(f"property float {key}\n")
        handle.write("end_header\n")

        for index, point in enumerate(points):
            values = [f"{point[0]:.6f}", f"{point[1]:.6f}", f"{point[2]:.6f}"]
            if colors is not None:
                values.extend(str(int(v)) for v in colors[index])
            if normals is not None:
                values.extend(f"{v:.6f}" for v in normals[index])
            for _, scalar in scalar_items:
                values.append(f"{float(scalar[index]):.6f}")
            handle.write(" ".join(values) + "\n")
