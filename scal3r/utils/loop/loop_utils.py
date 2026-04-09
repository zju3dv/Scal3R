from typing import Callable, List
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from scal3r.utils.pgo.processor import MapProcessor
from scal3r.utils.loop.optimizer import Sim3LoopOptimizer

def build_map_processor(alignment: str, **kwargs):
    return MapProcessor(alignment, **kwargs)


def build_sim3_loop_optimizer():
    return Sim3LoopOptimizer()


def build_loop_batches(
    sequence,
    indices,
    loop_list,
    loop_size: int,
    height: int,
    width: int,
    dataset_cfg,
    build_block_fn: Callable,
    log_fn: Callable[[str], None],
):
    batches_loop = []
    indices_loop = []
    starts = [item[0] for item in indices]

    def locate_block(target: int):
        import bisect

        position = bisect.bisect_right(starts, target) - 1
        if position > len(starts) - 1 or position < 0 or target > indices[position][1] or target < indices[position][0]:
            raise ValueError(f"Cannot find the block for target index {target}.")
        return position

    def create_loop_block(block: int, target: int):
        start, end, num_samples = indices[block][0], indices[block][1], loop_size * 2
        if target - loop_size < start:
            end = min(end, start + num_samples)
        elif target + loop_size > end:
            start = max(start, end - num_samples)
        else:
            start, end = target - loop_size, target + loop_size
        return start, end

    def overlap(window_a: tuple, window_b: tuple):
        return max(window_a[0], window_b[0]) < min(window_a[1], window_b[1])

    detected_loop = {}
    for i, j in loop_list:
        try:
            block0 = locate_block(i)
            start0, end0 = create_loop_block(block0, i)
            sampler_indices = list(range(start0, end0))

            block1 = locate_block(j)
            start1, end1 = create_loop_block(block1, j)
            sampler_indices.extend(list(range(start1, end1)))

            if block0 == block1:
                log_fn(f"Loop pair ({i}, {j}) falls inside block {block0}, skipping pose-graph loop loading.")
                continue

            key = tuple(sorted((block0, block1)))
            windows = ((start0, end0), (start1, end1)) if key[0] == block0 else ((start1, end1), (start0, end0))
            if key not in detected_loop:
                detected_loop[key] = []
            if any(overlap(windows[0], prev[0]) and overlap(windows[1], prev[1]) for prev in detected_loop[key]):
                log_fn(f"Loop pair {key} with windows {windows} overlaps an existing loop block, skipping.")
                continue
            detected_loop[key].append(windows)

            batches_loop.append(
                build_block_fn(sequence, sampler_indices, height, width, loop_size, dataset_cfg)
            )
            indices_loop.append((block0, (start0, end0), block1, (start1, end1)))
        except ValueError as exc:
            log_fn(f"[WARNING] cannot find loop between frame {i} and {j}: {exc}, skipping.")

    log_fn(f"Found and loaded {len(batches_loop)} loop closure blocks.")
    log_fn(f"Loop closure blocks info: {indices_loop}")
    return batches_loop, indices_loop


def combine_transform(sa, ra, ta, sb, rb, tb):
    scale = sb / sa
    rotation = rb @ ra.T
    translation = tb - scale * (rotation @ ta)
    return scale, rotation, translation


def accumulate_transform(transforms) -> List[np.ndarray]:
    if len(transforms) == 0:
        return []

    def to_c2w(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        dtype = rotation.dtype
        c2w = np.eye(4, dtype=dtype)
        c2w[:3, :3] = scale * rotation
        c2w[:3, 3] = translation.reshape(3)
        return c2w

    c2w_accum = np.eye(4, dtype=np.float32)
    cumulative: List[np.ndarray] = [c2w_accum]
    for scale, rotation, translation in transforms:
        c2w_accum = c2w_accum @ to_c2w(scale, rotation, translation)
        cumulative.append(c2w_accum)
    return cumulative


def visualize_loop(ini_track, opt_track, loop_track, output_dir):
    def extract_xyz(c2ws):
        c2ws = c2ws.cpu().numpy()
        return c2ws[:, 0], c2ws[:, 1], c2ws[:, 2]

    x0, _, y0 = extract_xyz(ini_track)
    x1, _, y1 = extract_xyz(opt_track)

    plt.figure(figsize=(8, 6))
    plt.plot(x0, y0, "o--", alpha=0.45, label="Before Optimization")
    plt.plot(x1, y1, "o-", label="After Optimization")
    for i, j, _ in loop_track:
        plt.plot([x0[i], x0[j]], [y0[i], y0[j]], "r--", alpha=0.25, label="Loop (Before)" if i == 5 else "")
        plt.plot([x1[i], x1[j]], [y1[i], y1[j]], "g-", alpha=0.35, label="Loop (After)" if i == 5 else "")
    plt.gca().set_aspect("equal")
    plt.title("Sim3 Loop Closure Optimization")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    save_path = join(output_dir, "sim3_opt_result.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
