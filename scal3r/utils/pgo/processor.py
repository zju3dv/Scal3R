import os
import numpy as np
from typing import Union, List

from scal3r.utils.pgo.submap import Submap
from scal3r.utils.pgo.aligner import PointCloudAligner
from scal3r.utils.pgo.constraint import Sim3Constraint
from scal3r.utils.pgo.optimizer import PoseGraphOptimizer

class MapProcessor:
    """ Processes submaps, adds constraints, and optimizes the pose graph.
    This is the main interface for managing submaps and their relationships.
    """
    def __init__(
        self,
        align_mode: str = "sim3_wet",
        conf_percent: float | None = None,
        min_dpt_thres: float | None = None,
        max_dpt_thres: float | None = None,
        max_align_points_per_frame: int | None = None,
    ):
        # Initialize the aligner
        self.n_submaps = 0
        self.optimizer = PoseGraphOptimizer()

        # Alignment-related parameters
        self.align_mode = align_mode
        self.conf_percent = conf_percent
        self.min_dpt_thres = min_dpt_thres
        self.max_dpt_thres = max_dpt_thres
        self.max_align_points_per_frame = max_align_points_per_frame

    def add_constraint(
        self,
        prev_submap: Submap,
        curr_submap: Submap,
        update: bool = True,
    ):
        # Find the overlap points between the previous and current submaps
        prev_xyz, curr_xyz, prev_cnf, curr_cnf = prev_submap.find_overlap(
            curr_submap
        )

        # If no overlap points are found, skip adding the constraint
        if prev_xyz is None or curr_xyz is None:
            print(f"Warning: No overlap found between submap {prev_submap.submap_id} and submap {curr_submap.submap_id}. Skipping.")
            return

        # Create an aligner
        aligner = PointCloudAligner(prev_xyz, curr_xyz, prev_cnf, curr_cnf)
        # Align the point clouds using Sim3 transformation
        if self.align_mode == "se3":
            T_curr_to_prev, s, R, t = aligner.align_se3()  # (4, 4)
        elif self.align_mode == "sim3":
            T_curr_to_prev, s, R, t = aligner.align_sim3()  # (4, 4)
        elif self.align_mode == "sim3_wet":
            T_curr_to_prev, s, R, t = aligner.robust_weighted_align_sim3()  # (4, 4)
        else:
            raise ValueError(f"Invalid align mode: {self.align_mode}")

        # Add the Sim3 constraint to the optimizer
        # ? What is the use of this constraint here?
        self.optimizer.add_sim3_constraint(
            Sim3Constraint(
                prev_submap.submap_id,
                curr_submap.submap_id,
                T_curr_to_prev,
                prev_xyz,
                curr_xyz
            )
        )

        # If update is True, update the current submap's pose based on the previous one
        # NOTE: the global pose is c2w
        if update:
            curr_submap.global_pose = prev_submap.global_pose @ T_curr_to_prev

        return s, R, t

    def add_submap(
        self,
        xyz: np.ndarray,
        dpt: np.ndarray,
        cnf: np.ndarray,
        msk: Union[np.ndarray, List[str]],
        file_name: List[Union[str, int]],
        save_path: str = None,
        conf_percent: float | None = None,
        min_dpt_thres: float | None = None,
        max_dpt_thres: float | None = None,
        max_align_points_per_frame: int | None = None,
        update: bool = True,
        compute_constraint: bool = True,
    ):
        """ Adds a new submap to the optimizer and creates constraints with the previous submap,
        and optionally updates the current submap's pose based on the previous one.

        Args:
            xyz (np.ndarray): Point cloud data of this submap with shape (N, ..., 3)
            dpt (np.ndarray): Depth data of this submap with shape (N, ..., 1)
            cnf (np.ndarray): Confidence map of this submap with shape (N, ..., 1)
            msk (Union[np.ndarray, List[str]]): corresponding mask with shape (N, ..., 1) or list of file paths
            file_name (str): Identifier for this submap
            update (bool): Whether to update the current submap's pose based on the previous one
            save_path (str): Path to save submap-related data
            compute_constraint (bool): Whether to compute pairwise constraint with the previous submap.
                Set to False when using align_submaps_parallel() afterwards.
        Returns:
            None
        """
        conf_percent = self.conf_percent if conf_percent is None else conf_percent
        min_dpt_thres = self.min_dpt_thres if min_dpt_thres is None else min_dpt_thres
        max_dpt_thres = self.max_dpt_thres if max_dpt_thres is None else max_dpt_thres
        max_align_points_per_frame = (
            self.max_align_points_per_frame
            if max_align_points_per_frame is None
            else max_align_points_per_frame
        )

        # Create a new submap
        curr_submap = Submap(
            self.n_submaps,
            xyz,
            dpt,
            cnf,
            msk,
            file_name,
            save_path,
            conf_percent,
            min_dpt_thres,
            max_dpt_thres,
            max_align_points_per_frame,
        )
        # Add it to the optimizer
        self.optimizer.add_submap(curr_submap)

        # Add constraint with the previous submap if exists
        s, R, t = None, None, None
        if compute_constraint and self.n_submaps > 0:
            prev_submap = self.optimizer.get_submap(self.n_submaps - 1)
            # Add constraint between the previous submap and the current one
            s, R, t = self.add_constraint(prev_submap, curr_submap, update)

        # Increment the total submap count
        self.n_submaps += 1

        return s, R, t

    def _align_pair(self, pair_index: int):
        """Compute pairwise alignment between consecutive submaps (thread-safe)."""
        prev = self.optimizer.get_submap(pair_index)
        curr = self.optimizer.get_submap(pair_index + 1)

        prev_xyz, curr_xyz, prev_cnf, curr_cnf = prev.find_overlap(curr)
        if prev_xyz is None or curr_xyz is None:
            return pair_index, None, None, None, None, None, None

        aligner = PointCloudAligner(prev_xyz, curr_xyz, prev_cnf, curr_cnf)
        if self.align_mode == "se3":
            T, s, R, t = aligner.align_se3()
        elif self.align_mode == "sim3":
            T, s, R, t = aligner.align_sim3()
        elif self.align_mode == "sim3_wet":
            T, s, R, t = aligner.robust_weighted_align_sim3()
        else:
            raise ValueError(f"Invalid align mode: {self.align_mode}")
        return pair_index, T, s, R, t, prev_xyz, curr_xyz

    def align_submaps_parallel(self, max_workers: int | None = None):
        """Compute all pairwise alignments between consecutive submaps in parallel.

        Call this after adding all submaps with compute_constraint=False.
        Uses ThreadPoolExecutor — NumPy BLAS operations release the GIL,
        so threads achieve true parallelism for the heavy linear algebra.

        Returns:
            List of (s, R, t) tuples for each consecutive pair.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n = self.n_submaps
        if n <= 1:
            return []

        n_pairs = n - 1
        if max_workers is None:
            max_workers = min(n_pairs, os.cpu_count() or 4)

        results = [None] * n_pairs
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._align_pair, i): i
                for i in range(n_pairs)
            }
            for future in as_completed(futures):
                idx, T, s, R, t, prev_xyz, curr_xyz = future.result()
                results[idx] = (T, s, R, t, prev_xyz, curr_xyz)

        # Sequential: accumulate global poses and store constraints
        norm_track = []
        for i, (T, s, R, t, prev_xyz, curr_xyz) in enumerate(results):
            if T is not None:
                self.optimizer.add_sim3_constraint(
                    Sim3Constraint(i, i + 1, T, prev_xyz, curr_xyz)
                )
                prev = self.optimizer.get_submap(i)
                curr = self.optimizer.get_submap(i + 1)
                curr.global_pose = prev.global_pose @ T
            norm_track.append((s, R, t))

        return norm_track

    def add_loop_closure(self, prev_submap_id, curr_submap_id):
        # Retrieve the submaps by their IDs
        prev_submap = self.optimizer.get_submap(prev_submap_id)
        curr_submap = self.optimizer.get_submap(curr_submap_id)
        # Add loop closure constraint without updating poses
        # TODO: why not update?
        self.add_constraint(prev_submap, curr_submap, update=False)

    def run_optimization(self, with_gtsam: bool = False):
        if with_gtsam:
            raise NotImplementedError("The public release only keeps the PyPose optimizer path.")
        self.optimizer.optimize_pypose()

    def save_pointclouds(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self.optimizer.save_pointclouds(save_path)
