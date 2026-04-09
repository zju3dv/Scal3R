import cv2
import torch
import numpy as np
from glob import glob
import math as py_math
import torch.nn.functional as F
from PIL import Image as PilImage
from typing import Iterable, List, Tuple

from scal3r.utils.console_utils import tqdm
from scal3r.utils.runtime_utils import scalarize
from scal3r.utils.base_utils import DotDict as dotdict
from scal3r.utils.data_utils import load_image_from_bytes
from scal3r.utils.parallel_utils import parallel_execution


def crop_nhwc_image(
    image: torch.Tensor,
    size: Iterable[int],
    center: bool = True,
    strict_center: bool = False,
    K: torch.Tensor | None = None,
    return_offset: bool = False,
):
    if center and strict_center:
        raise ValueError("center and strict_center cannot both be True")
    if strict_center and K is None:
        raise ValueError("K must be provided when strict_center is True")

    batch_shape = image.shape[:-3]
    image = image.reshape(-1, *image.shape[-3:])
    height, width = image.shape[-3:-1]
    target_h, target_w = [int(v) for v in size]

    if strict_center:
        start_h = int(round(K[1, 2].item())) - target_h // 2
        start_w = int(round(K[0, 2].item())) - target_w // 2
    elif center:
        start_h = (height - target_h) // 2
        start_w = (width - target_w) // 2
    else:
        start_h = 0
        start_w = 0

    end_h = start_h + target_h
    end_w = start_w + target_w
    offset_h = start_h
    offset_w = start_w

    if strict_center and (start_h < 0 or start_w < 0 or end_h > height or end_w > width):
        pad_top = max(0, -start_h)
        pad_bottom = max(0, end_h - height)
        pad_left = max(0, -start_w)
        pad_right = max(0, end_w - width)
        image = F.pad(image, (0, 0, pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        start_h += pad_top
        start_w += pad_left

    image = image[..., start_h:start_h + target_h, start_w:start_w + target_w, :]
    image = image.reshape(*batch_shape, *image.shape[-3:])
    if return_offset:
        return image, offset_h, offset_w
    return image


def rotate_image_rot90(image: torch.Tensor, clockwise: bool) -> torch.Tensor:
    if clockwise:
        return image.permute(1, 0, 2).flip(1).clone().contiguous()
    return image.permute(1, 0, 2).flip(0).clone().contiguous()


def adjust_intrinsic_rot90(
    K: torch.Tensor,
    width: int,
    height: int,
    clockwise: bool,
) -> torch.Tensor:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    rotated = torch.eye(3, dtype=K.dtype, device=K.device)
    rotated[0, 0], rotated[1, 1] = fy, fx
    if clockwise:
        rotated[0, 2], rotated[1, 2] = height - cy, cx
    else:
        rotated[0, 2], rotated[1, 2] = cy, width - cx
    return rotated


def adjust_extrinsic_rot90(w2c: torch.Tensor, clockwise: bool) -> torch.Tensor:
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    if clockwise:
        rot90 = torch.tensor(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            dtype=rotation.dtype,
            device=rotation.device,
        )
    else:
        rot90 = torch.tensor(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            dtype=rotation.dtype,
            device=rotation.device,
        )

    new_rotation = rot90 @ rotation
    new_translation = rot90 @ translation
    body = torch.cat([new_rotation, new_translation.unsqueeze(1)], dim=1)
    if w2c.shape[0] == 4:
        return torch.cat([body, w2c[3:4, :]], dim=0)
    return body


def rotate_90_degree(
    image: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    clockwise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = image.shape[:2]
    return (
        rotate_image_rot90(image, clockwise),
        adjust_intrinsic_rot90(K, width, height, clockwise),
        adjust_extrinsic_rot90(w2c, clockwise),
    )


def listify_patterns(patterns: str) -> List[str]:
    return [pattern.strip() for pattern in patterns.split(",") if pattern.strip()]


def collect_image_paths(input_dir: str, image_patterns: str) -> List[str]:
    image_paths = []
    for pattern in listify_patterns(image_patterns):
        image_paths.extend(glob(f"{input_dir}/{pattern}"))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir} with patterns: {image_patterns}")
    return image_paths


def build_dummy_ixt(height: int, width: int, focal_ratio: float = 1.0) -> torch.Tensor:
    focal = float(max(height, width)) * float(focal_ratio)
    return torch.tensor(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def load_rgb_from_path(path: str) -> torch.Tensor:
    with open(path, "rb") as file:
        image = load_image_from_bytes(file.read(), normalize=True)
    return torch.as_tensor(image).contiguous().to(torch.float32)


def resize_rgb(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    width, height = size
    return torch.from_numpy(
        np.array(
            PilImage.fromarray((image.numpy() * 255.0).astype(np.uint8)).resize(
                (width, height),
                PilImage.Resampling.BICUBIC,
            ),
            copy=True,
        )
    ).contiguous().to(torch.float32).div(255.0)


def resize_mask(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    width, height = size
    mask_np = cv2.resize(mask.numpy(), (width, height), interpolation=cv2.INTER_NEAREST)
    if mask_np.ndim == 2:
        mask_np = mask_np[..., None]
    return torch.from_numpy(mask_np).contiguous().to(torch.float32)


def apply_base_transforms(
    image: torch.Tensor,
    mask: torch.Tensor,
    ixt: torch.Tensor,
    w2c: torch.Tensor,
    dataset_cfg: dotdict,
):
    render_ratio = float(scalarize(dataset_cfg.get("render_ratio", 1.0)))
    if render_ratio != 1.0:
        height, width = image.shape[:2]
        target_h = max(1, int(height * render_ratio))
        target_w = max(1, int(width * render_ratio))
        ixt[0:1] *= target_w / width
        ixt[1:2] *= target_h / height
        image = resize_rgb(image, (target_w, target_h))
        mask = resize_mask(mask, (target_w, target_h))

    rot90 = dataset_cfg.get("rot90", None)
    if rot90 is not None:
        clockwise = rot90 == "clockwise"
        image, ixt, w2c = rotate_90_degree(image, ixt, w2c.clone(), clockwise=clockwise)
        mask, _, _ = rotate_90_degree(mask, ixt, w2c.clone(), clockwise=clockwise)

    return image, mask, ixt


def determine_target_size(height: int, width: int, dataset_cfg: dotdict) -> Tuple[int, int]:
    proc_max_size = int(scalarize(dataset_cfg.get("proc_max_size", -1)))
    proc_align_size = max(1, int(scalarize(dataset_cfg.get("proc_align_size", 1))))
    aspect_ratio = height / width

    if proc_max_size > 0:
        target_h = int(aspect_ratio * proc_max_size)
        target_w = proc_max_size
    else:
        target_h = int(height)
        target_w = int(width)

    target_h = max(proc_align_size, target_h // proc_align_size * proc_align_size)
    target_w = max(proc_align_size, target_w // proc_align_size * proc_align_size)
    return target_h, target_w


def resize_to_cover(
    image: torch.Tensor,
    mask: torch.Tensor,
    ixt: torch.Tensor,
    target_h: int,
    target_w: int,
    proc_align_size: int,
):
    height, width = image.shape[:2]
    ratio = max(target_h / max(height, 1), target_w / max(width, 1))
    resized_h = max(proc_align_size, round(height * ratio / proc_align_size) * proc_align_size)
    resized_w = max(proc_align_size, round(width * ratio / proc_align_size) * proc_align_size)

    if resized_h != height or resized_w != width:
        ixt[0:1] *= resized_w / width
        ixt[1:2] *= resized_h / height
        image = resize_rgb(image, (resized_w, resized_h))
        mask = resize_mask(mask, (resized_w, resized_h))

    return image, mask, ixt


def finalize_transforms(
    image: torch.Tensor,
    mask: torch.Tensor,
    ixt: torch.Tensor,
    dataset_cfg: dotdict,
    target_h: int,
    target_w: int,
):
    proc_align_size = max(1, int(scalarize(dataset_cfg.get("proc_align_size", 1))))
    center_crop = bool(dataset_cfg.get("center_crop", True))

    image, mask, ixt = resize_to_cover(image, mask, ixt, target_h, target_w, proc_align_size)
    image, crop_h, crop_w = crop_nhwc_image(
        image,
        size=(target_h, target_w),
        center=False,
        strict_center=center_crop,
        K=ixt,
        return_offset=True,
    )
    mask, _, _ = crop_nhwc_image(
        mask,
        size=(target_h, target_w),
        center=False,
        strict_center=center_crop,
        K=ixt,
        return_offset=True,
    )
    ixt[0, 2] -= crop_w
    ixt[1, 2] -= crop_h
    return image, mask, ixt


def build_preprocessed_frame(
    image: torch.Tensor,
    mask: torch.Tensor,
    ixt: torch.Tensor,
    path: str,
):
    return dotdict(
        rgb=image.reshape(-1, 3),
        msk=mask.reshape(-1, 1),
        ixt=ixt,
        path=path,
    )


def preprocess_image_path(path: str, dataset_cfg: dotdict, target_h: int, target_w: int):
    image = load_rgb_from_path(path)
    mask = torch.ones_like(image[..., :1])
    ixt = build_dummy_ixt(
        image.shape[0],
        image.shape[1],
        float(scalarize(dataset_cfg.get("focal_ratio", 1.0))),
    )
    w2c = torch.eye(4, dtype=torch.float32)
    image, mask, ixt = apply_base_transforms(image, mask, ixt, w2c, dataset_cfg)
    image, mask, ixt = finalize_transforms(image, mask, ixt, dataset_cfg, target_h, target_w)
    return build_preprocessed_frame(image, mask, ixt, path)


def load_and_preprocess_images(
    image_paths: List[str],
    dataset_cfg: dotdict,
    preprocess_workers: int = 1,
):
    first = load_rgb_from_path(image_paths[0])
    first_mask = torch.ones_like(first[..., :1])
    first_ixt = build_dummy_ixt(
        first.shape[0],
        first.shape[1],
        float(scalarize(dataset_cfg.get("focal_ratio", 1.0))),
    )
    first_w2c = torch.eye(4, dtype=torch.float32)
    first, first_mask, first_ixt = apply_base_transforms(first, first_mask, first_ixt, first_w2c, dataset_cfg)
    target_h, target_w = determine_target_size(first.shape[0], first.shape[1], dataset_cfg)
    first, first_mask, first_ixt = finalize_transforms(first, first_mask, first_ixt, dataset_cfg, target_h, target_w)

    sequence = [build_preprocessed_frame(first, first_mask, first_ixt, image_paths[0])]
    remaining_paths = image_paths[1:]
    if not remaining_paths:
        return sequence, target_h, target_w

    preprocess_workers = max(1, int(preprocess_workers))
    if preprocess_workers == 1:
        pbar = tqdm(total=len(remaining_paths), desc="Preprocessing images")
        for path in remaining_paths:
            sequence.append(preprocess_image_path(path, dataset_cfg, target_h, target_w))
            pbar.update()
        pbar.close()
    else:
        processed = parallel_execution(
            remaining_paths,
            action=preprocess_image_path,
            dataset_cfg=dataset_cfg,
            target_h=target_h,
            target_w=target_w,
            num_workers=preprocess_workers,
            print_progress=True,
            desc="Preprocessing images",
        )
        sequence.extend(processed)

    return sequence, target_h, target_w


def reorder_middle_reference(indices: List[int]) -> List[int]:
    if len(indices) <= 1:
        return indices
    mid = int(py_math.ceil(len(indices) / 2)) - 1
    return [indices[mid]] + indices[:mid] + indices[mid + 1 :]


def build_image_only_block(
    sequence,
    block_indices: List[int],
    height: int,
    width: int,
    overlap_size: int,
    dataset_cfg: dotdict,
):
    orig_src_inds = torch.as_tensor(block_indices, dtype=torch.long)
    src_inds = torch.as_tensor(reorder_middle_reference(block_indices), dtype=torch.long)

    rgbs = torch.stack([sequence[index].rgb for index in src_inds.tolist()], dim=0)
    msks = torch.stack([sequence[index].msk for index in src_inds.tolist()], dim=0) > 0.5

    return dotdict(
        H=torch.tensor(height, dtype=torch.long),
        W=torch.tensor(width, dtype=torch.long),
        src_inds=src_inds[None],
        orig_src_inds=orig_src_inds[None],
        msk=msks[None],
        meta=dotdict(
            rgb=rgbs[None],
            H=torch.tensor([height], dtype=torch.long),
            W=torch.tensor([width], dtype=torch.long),
            overlap_size=torch.tensor([overlap_size], dtype=torch.long),
            block_size=torch.tensor([len(block_indices)], dtype=torch.long),
            aspect_ratio=torch.tensor([float(height) / float(width)], dtype=torch.float32),
            cam_param_type=[dataset_cfg.get("cam_param_type", "abs_quat_fov")],
            use_world_coord=torch.tensor([bool(dataset_cfg.get("use_world_coord", True))]),
        ),
    )
