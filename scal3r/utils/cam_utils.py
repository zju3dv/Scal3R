import os
import cv2
import torch
import numpy as np
from os.path import join
from typing import Iterable
import torch.nn.functional as F

from scal3r.utils.base_utils import DotDict
from scal3r.utils.math_utils import affine_inverse


class _YamlFileStorageWriter:
    def __init__(self, filename: str):
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.handle = open(filename, "w", encoding="utf-8")
        self.handle.write("%YAML:1.0\r\n")
        self.handle.write("---\r\n")

    def write(self, key: str, value, value_type: str = "mat") -> None:
        if value_type == "mat":
            value = np.asarray(value, dtype=np.float64)
            self.handle.write(f"{key}: !!opencv-matrix\r\n")
            self.handle.write(f"  rows: {value.shape[0]}\r\n")
            self.handle.write(f"  cols: {value.shape[1]}\r\n")
            self.handle.write("  dt: d\r\n")
            flat = ", ".join(f"{item:.10f}" for item in value.reshape(-1))
            self.handle.write(f"  data: [{flat}]\r\n")
            return
        if value_type == "list":
            self.handle.write(f"{key}:\r\n")
            for item in value:
                self.handle.write(f'  - "{item}"\r\n')
            return
        if value_type == "real":
            if isinstance(value, np.ndarray):
                value = value.item()
            self.handle.write(f"{key}: {float(value):.10f}\r\n")
            return
        raise NotImplementedError(f"Unsupported YAML file storage type: {value_type}")

    def close(self) -> None:
        self.handle.close()


def compute_trajectory_length(c2ws: np.ndarray) -> float:
    centers = c2ws[:, :3, 3]
    deltas = centers[1:] - centers[:-1]
    return float(np.linalg.norm(deltas, axis=1).sum())


def _camera_to_w2c(camera) -> np.ndarray:
    if "R" in camera:
        rotation = np.asarray(camera.R, dtype=np.float64)
    else:
        rotation = cv2.Rodrigues(np.asarray(camera.Rvec, dtype=np.float64))[0]
    translation = np.asarray(camera.T, dtype=np.float64).reshape(3, 1)

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rotation
    w2c[:3, 3:] = translation
    return w2c


def camera_dict_to_c2ws(cameras: dict) -> np.ndarray:
    camera_items = [
        (raw_key.split(".")[0], value)
        for raw_key, value in DotDict(cameras).items()
        if raw_key != "basenames"
    ]
    camera_items.sort(key=lambda item: item[0])

    c2ws = np.zeros((len(camera_items), 4, 4), dtype=np.float64)
    for index, (_, camera) in enumerate(camera_items):
        c2ws[index] = np.linalg.inv(_camera_to_w2c(camera))
    return c2ws


def write_camera_mat(cameras: dict, path: str, filename: str = "mat.txt") -> None:
    os.makedirs(path, exist_ok=True)
    c2ws = camera_dict_to_c2ws(cameras)
    np.savetxt(join(path, filename), c2ws.reshape(c2ws.shape[0], 16), fmt="%.6f")


def write_camera(
    cameras: dict,
    path: str,
    intri_name: str = "",
    extri_name: str = "",
) -> None:
    os.makedirs(path, exist_ok=True)
    if not intri_name or not extri_name:
        intri_name = join(path, "intri.yml")
        extri_name = join(path, "extri.yml")

    intri = _YamlFileStorageWriter(intri_name)
    extri = _YamlFileStorageWriter(extri_name)
    cam_names = [key.split(".")[0] for key in cameras.keys()]
    intri.write("names", cam_names, "list")
    extri.write("names", cam_names, "list")

    cameras = DotDict(cameras)
    for raw_key, value in cameras.items():
        if raw_key == "basenames":
            continue
        key = raw_key.split(".")[0]
        intri.write(f"K_{key}", value.K)
        if "H" in value:
            intri.write(f"H_{key}", value.H, "real")
        if "W" in value:
            intri.write(f"W_{key}", value.W, "real")
        if "D" not in value:
            if "dist" in value:
                value.D = value.dist
            else:
                value.D = np.zeros((5, 1))
        value.D = np.asarray(value.D)
        if value.D.shape == (1, 4):
            value.D = np.concatenate([value.D.T, np.zeros_like(value.D.T[:1])], axis=0)
        intri.write(f"D_{key}", value.D.reshape(5, 1))

        if "R" not in value:
            value.R = cv2.Rodrigues(np.asarray(value.Rvec))[0]
        if "Rvec" not in value:
            value.Rvec = cv2.Rodrigues(np.asarray(value.R))[0]
        extri.write(f"R_{key}", value.Rvec)
        extri.write(f"Rot_{key}", value.R)
        translation = np.asarray(value.T).reshape(3, 1)
        extri.write(f"T_{key}", translation)

        if "t" in value:
            extri.write(f"t_{key}", value.t, "real")
        if "n" in value:
            extri.write(f"n_{key}", value.n, "real")
        if "f" in value:
            extri.write(f"f_{key}", value.f, "real")
        if "bounds" in value:
            extri.write(f"bounds_{key}", value.bounds)
        if "ccm" in value:
            intri.write(f"ccm_{key}", value.ccm)
        if "rdist" in value:
            intri.write(f"rdist_{key}", value.rdist)

    intri.close()
    extri.close()


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2.0 / (quaternions * quaternions).sum(dim=-1)
    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


def svd_orthogonalize(matrix: torch.Tensor) -> torch.Tensor:
    shape = matrix.shape
    matrix = matrix.reshape(-1, 3, 3)
    matrix = F.normalize(matrix, p=2, dim=-1).transpose(-1, -2)
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    v = vh.transpose(-1, -2)
    det = torch.det(v @ u.transpose(-1, -2)).view(-1, 1, 1)
    rotation = torch.matmul(
        torch.cat([v[:, :, :-1], v[:, :, -1:] * det], dim=-1),
        u.transpose(-1, -2),
    )
    return rotation.reshape(shape)


def decode_camera_params(
    cam: torch.Tensor,
    H: int | torch.Tensor | Iterable[int] | None = None,
    W: int | torch.Tensor | Iterable[int] | None = None,
    type: str = "abs_quat_fov",
    inverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    cam = torch.as_tensor(cam)
    height = torch.as_tensor(H, dtype=cam.dtype, device=cam.device)
    width = torch.as_tensor(W, dtype=cam.dtype, device=cam.device)

    if type == "abs_quat_fov":
        translation = cam[..., 0:3]
        rotation = quat_to_mat(cam[..., 3:7])
        fov = cam[..., 7:9]
    elif type == "abs_rotmat_fov":
        translation = cam[..., 0:3]
        rotation = svd_orthogonalize(cam[..., 3:12].reshape(cam.shape[:-1] + (3, 3)))
        fov = cam[..., 12:14]
    else:
        raise ValueError(f"Unknown camera parameter encoding type: {type}")

    ext = torch.cat([rotation, translation[..., None]], dim=-1)
    ixt = torch.zeros(cam.shape[:-1] + (3, 3), dtype=cam.dtype, device=cam.device)
    ixt[..., 0, 0] = width / 2 / torch.tan(fov[..., 1] / 2)
    ixt[..., 1, 1] = height / 2 / torch.tan(fov[..., 0] / 2)
    ixt[..., 0, 2] = width / 2
    ixt[..., 1, 2] = height / 2
    ixt[..., 2, 2] = 1.0
    if inverse:
        ext = affine_inverse(ext)
    return ext, ixt
