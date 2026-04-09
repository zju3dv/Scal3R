import cv2
import numpy as np
from typing import Union, List

from scal3r.utils.pgo.utils import SerializableMixin


class Submap(SerializableMixin):
    """ Represents a submap in the pose graph optimization process """
    def __init__(
        self,
        submap_id: int,
        xyz: np.ndarray,
        dpt: np.ndarray,
        cnf: np.ndarray,
        msk: Union[np.ndarray, List[str]],
        file_name: str,
        save_path: str,
        conf_percent: float | None = None,
        min_dpt_thres: float | None = None,
        max_dpt_thres: float | None = None,
        max_align_points_per_frame: int | None = None,
    ):
        # Submap identifier
        self.submap_id = submap_id

        # Essential data of the submap
        self.xyz = xyz  # (N, ..., 3)
        self.dpt = dpt  # (N, ..., 1)
        self.cnf = cnf  # (N, ..., 1)
        self.file_name = file_name
        self.save_path = save_path

        # Bookkeepings
        self.conf_percent = conf_percent
        self.min_dpt_thres = min_dpt_thres
        self.max_dpt_thres = max_dpt_thres
        self.max_align_points_per_frame = max_align_points_per_frame

        # Maintains the global pose of this submap
        self.global_pose = np.eye(4)  # c2w

        # Masking and filtering
        self.msk = self.get_mask(
            dpt,
            cnf,
            msk,
            conf_percent,
            min_dpt_thres,
            max_dpt_thres,
        )  # (N, ...)
        # Filter the local valid point cloud based on the mask
        self.xyz_local = self.get_local_xyz()  # (P, 3)

    def get_mask(
        self,
        dpt: np.ndarray,
        cnf: np.ndarray,
        msk: Union[np.ndarray, List[str]],
        conf_percent: float | None = None,
        min_dpt_thres: float | None = None,
        max_dpt_thres: float | None = None,
    ):
        msks = []
        # If msk is a list of file paths, we assume each path corresponds to a mask image
        # this is kind of a hacky way to handle business inference logic
        for i in range(self.xyz.shape[0]):
            msks.append(self.filter_by_conf_mask_depth(
                dpt[i, ..., 0] if dpt.shape[-1] == 1 else dpt[i],
                cnf[i, ..., 0] if cnf.shape[-1] == 1 else cnf[i],
                msk[i] if isinstance(msk, list) else msk[i, ..., 0] if msk.shape[-1] == 1 else msk[i],
                conf_percent=conf_percent,
                min_dpt_thres=min_dpt_thres,
                max_dpt_thres=max_dpt_thres,
            ))
        # Stack the masks along the first dimension
        return np.stack(msks, axis=0)  # (N, ...)

    def filter_by_conf_mask_depth(
        self,
        dpt: np.ndarray,
        cnf: np.ndarray,
        msk: Union[str, np.ndarray],
        conf_percent: float | None = None,
        min_dpt_thres: float | None = None,
        max_dpt_thres: float | None = None,
    ):
        """ Filter the point cloud based on confidence map, mask, and depth range """
        # Filter the point cloud based on confidence map
        if conf_percent is not None:
            top_k = int(cnf.size * conf_percent)  # int
            if top_k <= 0:
                cnf_msk = np.zeros_like(cnf, dtype=bool)  # (...,)
            elif top_k >= cnf.size:
                cnf_msk = np.ones_like(cnf, dtype=bool)  # (...,)
            else:
                thres = np.partition(cnf.flatten(), -top_k)[-top_k]  # scalar
                cnf_msk = cnf >= thres  # (...,)
        else:
            cnf_msk = np.ones_like(cnf, dtype=bool)  # (...,)

        # Maybe load the mask image if msk is a file path
        # Otherwise, we assume msk is already a numpy array
        if isinstance(msk, str):
            H, W = cnf.shape[:2]
            msk = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            smt_msk = ~(
                ((msk >= 9) & (msk <= 17)) | 
                (msk == 20) | (msk == 2) | (msk == 38)
            )  # (...,)
        else:
            smt_msk = msk > 0  # (...,)

        # Filter based on the depth range
        finite_dpt_msk = np.isfinite(dpt)
        positive_dpt_msk = dpt > 0
        if min_dpt_thres is not None and max_dpt_thres is not None:
            dpt_msk = finite_dpt_msk & (dpt >= min_dpt_thres) & (dpt <= max_dpt_thres)  # (...,)
        elif min_dpt_thres is not None:
            dpt_msk = finite_dpt_msk & (dpt >= min_dpt_thres)  # (...,)
        elif max_dpt_thres is not None:
            dpt_msk = finite_dpt_msk & (dpt <= max_dpt_thres)  # (...,)
        else:
            # The release image-folder path starts from an all-ones mask, so
            # at minimum we should ignore clearly invalid predicted depth.
            dpt_msk = finite_dpt_msk & positive_dpt_msk  # (...,)

        # Combine all masks
        xyz_msk = cnf_msk & smt_msk & dpt_msk  # (...,)
        return xyz_msk

    def get_local_xyz(self):
        """ Get all the valid points in the current submap based on the mask """
        xyzs = []
        for i in range(self.xyz.shape[0]):
            xyz = self.xyz[i][self.msk[i]]
            if len(xyz) > 0:
                xyzs.append(xyz)
        if not xyzs:
            return np.zeros((0, self.xyz.shape[-1]), dtype=self.xyz.dtype)
        return np.vstack(xyzs)  # (N, 3)

    def get_global_xyz(self):
        """ Get the global point cloud of the submap """
        xyz = np.hstack([
            self.xyz_local[:, :3],
            np.ones_like(self.xyz_local[:, :1])
        ])  # (P, 4)
        xyz = xyz @ self.global_pose.T  # (P, 4)
        # Sim3 trnsformation matrix sometimes has a non-unit scale,
        # like [R t/s; 0 1/s], so we normalize it
        xyz = xyz / xyz[:, 3:4]  # (P, 4)
        # FIXME: why concatenate xyz_local[:, 3:]?
        xyz_world = np.hstack([xyz[:, :3], self.xyz_local[:, 3:]])  # (P, 3)
        return xyz_world

    def find_overlap(self, submap):
        # Find the overlap frames using file names,
        # and record the indices of the overlapping frames in both submaps
        com_names, prev_inds, curr_inds = [], [], []
        for i, name in enumerate(submap.file_name):
            if name in self.file_name:
                com_names.append(name)
                prev_inds.append(self.file_name.index(name))
                curr_inds.append(i)

        # If no overlapping frames are found, return None
        if len(com_names) == 0:
            print("Warning: No overlapping frames found between submaps.")
            return None, None, None, None

        # Extract the valid points and the corresponding confidence
        # from both submaps based on the overlapping indices
        prev_xyzs, curr_xyzs, prev_cnfs, curr_cnfs = [], [], [], []
        for i, j in zip(prev_inds, curr_inds):
            msk = self.msk[i] & submap.msk[j]  # (...,)
            if not np.any(msk):
                continue

            prev_xyz = self.xyz[i][msk]
            curr_xyz = submap.xyz[j][msk]
            prev_cnf = self.cnf[i][msk]
            curr_cnf = submap.cnf[j][msk]

            candidates = [self.max_align_points_per_frame, submap.max_align_points_per_frame]
            valid_candidates = [candidate for candidate in candidates if candidate is not None]
            max_points = min(valid_candidates) if valid_candidates else None
            if max_points is not None and len(prev_xyz) > max_points:
                score = np.minimum(prev_cnf.reshape(-1), curr_cnf.reshape(-1))
                keep = np.argpartition(score, -max_points)[-max_points:]
                prev_xyz = prev_xyz[keep]
                curr_xyz = curr_xyz[keep]
                prev_cnf = prev_cnf[keep]
                curr_cnf = curr_cnf[keep]

            prev_xyzs.append(prev_xyz)
            curr_xyzs.append(curr_xyz)
            prev_cnfs.append(prev_cnf)
            curr_cnfs.append(curr_cnf)

        if len(prev_xyzs) == 0:
            print("Warning: No valid overlap points remained after filtering.")
            return None, None, None, None

        # Stack the valid points and the corresponding confidence
        # from both submaps
        prev_xyz = np.vstack(prev_xyzs)  # (N, 3)
        curr_xyz = np.vstack(curr_xyzs)  # (N, 3)
        prev_cnf = np.vstack(prev_cnfs)  # (N,)
        curr_cnf = np.vstack(curr_cnfs)  # (N,)

        # Return the valid points and the corresponding confidence
        # from both submaps
        return prev_xyz, curr_xyz, prev_cnf, curr_cnf
