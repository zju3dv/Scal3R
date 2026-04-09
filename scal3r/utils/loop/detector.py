import os
from typing import List, Tuple
from os.path import join

import cv2
import torch
import numpy as np

from torch import nn
from PIL import Image

from scal3r.utils.console_utils import get_logger, tqdm
from scal3r.utils.loop.models.helper import get_aggregator, get_backbone

logger = get_logger("scal3r.utils.loop.detector")


class SALADLoopModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = get_backbone(
            "dinov2_vitb14",
            {
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
        )
        self.aggregator = get_aggregator(
            "SALAD",
            {
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            },
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.aggregator(self.backbone(inputs))


class LoopDetector:
    """Release-local wrapper of the original SALAD loop detector."""

    def __init__(
        self,
        image_list: List[str],
        result_dir: str,
        ckpt_path: str,
        image_size: Tuple[int, int] = (336, 336),
        batch_size: int = 32,
        similarity_threshold: float = 0.7,
        top_k: int = 5,
        use_nms: bool = True,
        nms_threshold: int = 25,
    ):
        self.image_list = image_list
        self.result_dir = result_dir

        self.ckpt_path = ckpt_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.use_nms = use_nms
        self.nms_threshold = nms_threshold

        self.model = None
        self.device = None
        self.descriptors = None
        self.loop_closures = None

    def _input_transform(self, image_size: Tuple[int, int] | None = None):
        import torchvision.transforms as transforms

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if image_size:
            return transforms.Compose(
                [
                    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def load_model(self):
        if not self.ckpt_path:
            raise ValueError("LoopDetector requires a SALAD checkpoint path.")

        self.model = SALADLoopModel()
        state_dict = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def extract_descriptor(self):
        if self.model is None or self.device is None:
            self.load_model()

        transform = self._input_transform(self.image_size)
        descriptors = []
        pbar = tqdm(range(0, len(self.image_list), self.batch_size), desc="Extracting loop features")
        for start in pbar:
            batch_paths = self.image_list[start:start + self.batch_size]
            batch_imgs = []
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    batch_imgs.append(transform(image))
                except Exception as exc:
                    logger.warning("failed to process loop image %s: %s", path, exc)
                    height, width = self.image_size
                    batch_imgs.append(torch.zeros(3, height, width))

            batch_tensor = torch.stack(batch_imgs).to(self.device)
            amp_enabled = self.device.type == "cuda"
            with torch.no_grad():
                with torch.autocast(
                    enabled=amp_enabled,
                    device_type=self.device.type,
                    dtype=torch.float16,
                ):
                    descriptors.append(self.model(batch_tensor).cpu())

        self.descriptors = torch.cat(descriptors, dim=0)

    def find_loop(self):
        if self.descriptors is None:
            self.extract_descriptor()

        descriptor = self.descriptors.numpy()
        try:
            import faiss

            index = faiss.IndexFlatIP(descriptor.shape[1])
            index.add(descriptor)
            similarity, indices = index.search(descriptor, self.top_k + 1)
        except Exception:
            similarity = descriptor @ descriptor.T
            indices = np.argsort(-similarity, axis=1)[:, :self.top_k + 1]
            similarity = np.take_along_axis(similarity, indices, axis=1)

        loop_closures = []
        for i in range(len(self.descriptors)):
            for j in range(1, self.top_k + 1):
                sim = float(similarity[i, j])
                idx = int(indices[i, j])
                if sim > self.similarity_threshold and abs(i - idx) > 10:
                    loop_closures.append((min(i, idx), max(i, idx), sim))

        loop_closures = sorted(set(loop_closures), key=lambda item: item[2], reverse=True)
        if self.use_nms and self.nms_threshold > 0:
            loop_closures = apply_loop_nms_filter(loop_closures, self.nms_threshold)

        self.loop_closures = [(max(i, j), min(i, j), sim) for i, j, sim in loop_closures]

    def save_result(self):
        if self.loop_closures is None:
            self.find_loop()

        os.makedirs(self.result_dir, exist_ok=True)
        save_path = join(self.result_dir, "loop_closures.txt")
        with open(save_path, "w", encoding="utf-8") as file:
            file.write("# Loop Detection Results (index1, index2, similarity)\n")
            if self.use_nms:
                file.write(f"# NMS filtering applied, threshold: {self.nms_threshold}\n")
            file.write("\n# Loop pairs:\n")
            for i, j, sim in self.loop_closures:
                file.write(f"{i}, {j}, {sim:.4f}\n")

        logger.info("Found %d loop pairs, results saved to %s", len(self.loop_closures), save_path)

    def get_loop(self):
        return [(idx1, idx2) for idx1, idx2, _ in self.loop_closures]

    def run(self):
        self.load_model()
        self.extract_descriptor()
        self.find_loop()
        self.save_result()
        return self.loop_closures


def apply_loop_nms_filter(loop_closures, nms_threshold: int):
    if not loop_closures or nms_threshold <= 0:
        return loop_closures

    sorted_loops = sorted(loop_closures, key=lambda item: item[2], reverse=True)
    filtered = []
    max_frame = max(max(idx1, idx2) for idx1, idx2, _ in loop_closures)
    suppressed = set()
    for idx1, idx2, sim in sorted_loops:
        if idx1 in suppressed or idx2 in suppressed:
            continue

        suppress_range = set()
        filtered.append((idx1, idx2, sim))
        sidx1 = max(0, idx1 - nms_threshold)
        eidx1 = min(idx1 + nms_threshold + 1, idx2)
        suppress_range.update(range(sidx1, eidx1))
        sidx2 = max(idx1 + 1, idx2 - nms_threshold)
        eidx2 = min(idx2 + nms_threshold + 1, max_frame + 1)
        suppress_range.update(range(sidx2, eidx2))
        suppressed.update(suppress_range)

    return filtered


def fallback_detect_loops(
    image_list: List[str],
    result_dir: str,
    image_size: Tuple[int, int] = (64, 64),
    top_k: int = 5,
    similarity_threshold: float = 0.92,
    nms_threshold: int = 25,
    min_frame_gap: int = 10,
):
    descriptors = []
    pbar = tqdm(total=len(image_list), desc="Fallback loop descriptors")
    for path in image_list:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to read image for fallback loop detection: {path}")
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        descriptor = image.astype(np.float32).reshape(-1) / 255.0
        descriptor = descriptor - descriptor.mean()
        descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-8)
        descriptors.append(descriptor.astype(np.float32))
        pbar.update()
    pbar.close()

    descriptors = np.stack(descriptors, axis=0)
    try:
        import faiss

        index = faiss.IndexFlatIP(descriptors.shape[1])
        index.add(descriptors)
        similarity, indices = index.search(descriptors, top_k + 1)
    except Exception:
        similarity = descriptors @ descriptors.T
        indices = np.argsort(-similarity, axis=1)[:, :top_k + 1]
        similarity = np.take_along_axis(similarity, indices, axis=1)

    loop_closures = []
    for i in range(len(descriptors)):
        for j in range(1, top_k + 1):
            sim = float(similarity[i, j])
            idx = int(indices[i, j])
            if sim > similarity_threshold and abs(i - idx) > min_frame_gap:
                loop_closures.append((min(i, idx), max(i, idx), sim))

    loop_closures = sorted(set(loop_closures), key=lambda item: item[2], reverse=True)
    loop_closures = apply_loop_nms_filter(loop_closures, nms_threshold)

    os.makedirs(result_dir, exist_ok=True)
    save_path = join(result_dir, "loop_closures.txt")
    with open(save_path, "w", encoding="utf-8") as file:
        file.write("# Fallback Loop Detection Results (index1, index2, similarity)\n")
        file.write(
            f"# image_size={image_size}, top_k={top_k}, "
            f"similarity_threshold={similarity_threshold}, min_frame_gap={min_frame_gap}\n"
        )
        file.write(f"# nms_threshold={nms_threshold}\n\n")
        for i, j, sim in loop_closures:
            file.write(f"{i}, {j}, {sim:.6f}\n")

    logger.info("Fallback loop detector found %d loop pairs", len(loop_closures))
    return [(idx1, idx2) for idx1, idx2, _ in loop_closures]


def detect_loops(
    image_list: List[str],
    result_dir: str,
    loop_ckpt: str = "",
    nms_threshold: int = 25,
    min_frame_gap: int = 10,
):
    if not loop_ckpt:
        return fallback_detect_loops(
            image_list,
            result_dir=result_dir,
            nms_threshold=nms_threshold,
            min_frame_gap=min_frame_gap,
        )

    try:
        loop_detector = LoopDetector(
            image_list=image_list,
            result_dir=result_dir,
            ckpt_path=loop_ckpt,
            nms_threshold=nms_threshold,
        )
        loop_detector.run()
        return loop_detector.get_loop()
    except Exception as exc:
        logger.warning(
            "loop detector failed (%s: %s), using fallback loop detector.",
            type(exc).__name__,
            exc,
        )
        return fallback_detect_loops(
            image_list,
            result_dir=result_dir,
            nms_threshold=nms_threshold,
            min_frame_gap=min_frame_gap,
        )
