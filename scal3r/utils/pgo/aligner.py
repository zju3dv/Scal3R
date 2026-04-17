import numpy as np


class PointCloudAligner:
    ''' Aligns two point clouds using SE(3) or Sim(3) transformations
    This PointCloudAligner only support aligning two point clouds with shape of (N, 3)
    TODO: Support batched shape (..., N, 3) aligning of point clouds
    '''
    def __init__(self, tar_xyz, src_xyz, tar_cnf=None, src_cnf=None):
        # Assertion
        assert src_xyz.shape == tar_xyz.shape, \
            "Point clouds must have the same shape."
        self.tar_xyz = tar_xyz  # (N, 3)
        self.src_xyz = src_xyz  # (N, 3)
        self.tar_cnf = tar_cnf  # (N,)
        self.src_cnf = src_cnf  # (N,)

    def align_se3(self):
        """ Return a 4x4 SE(3) matrix with R|t (no scaling)
        The objective function is ||R * src_xyz + t - tar_xyz||^2

        Args:
            src_xyz (np.ndarray): Source point cloud of shape (N, 3)
            tar_xyz (np.ndarray): Target point cloud of shape (N, 3)
        Returns:
            T (np.ndarray): Transformation matrix of shape (4, 4)
        """
        # Ensure we only use the first 3 columns
        src_xyz = self.src_xyz[:, :3]
        tar_xyz = self.tar_xyz[:, :3]

        # Compute centroids
        src_centroid = np.mean(src_xyz, axis=0)  # (3,)
        tar_centroid = np.mean(tar_xyz, axis=0)  # (3,)

        # Center the two set of points
        src_xyz_mu = src_xyz - src_centroid  # (N, 3)
        tar_xyz_mu = tar_xyz - tar_centroid  # (N, 3)

        # Compute rotation using SVD
        R, _, _ = _kabsch_rotation(
            src_xyz_mu, tar_xyz_mu, normalize=False
        )  # (3, 3)
        # Compute translation with equation t = target_centroid - R * source_centroid
        t = tar_centroid - R @ src_centroid  # (3,)

        # Construct the transformation matrix
        T = np.eye(4, dtype=src_xyz.dtype)
        T[:3, :3] = R
        T[:3,  3] = t
        return T, 1.0, R, t

    def align_sim3(self):
        """ Return a 4x4 Sim(3) Umeyama matrix with sR|t
        The objective function is ||s * R * src_xyz + t - tar_xyz||^2

        Args:
            src_xyz (np.ndarray): Source point cloud of shape (N, 3)
            tar_xyz (np.ndarray): Target point cloud of shape (N, 3)
        Returns:
            T (np.ndarray): Transformation matrix of shape (4, 4)
        """
        # Ensure we only use the first 3 columns
        src_xyz = self.src_xyz[:, :3]
        tar_xyz = self.tar_xyz[:, :3]

        # Compute centroids
        src_centroid = np.mean(src_xyz, axis=0)  # (3,)
        tar_centroid = np.mean(tar_xyz, axis=0)  # (3,)

        # Center the two set of points
        src_xyz_mu = src_xyz - src_centroid  # (N, 3)
        tar_xyz_mu = tar_xyz - tar_centroid  # (N, 3)

        # Compute rotation using SVD
        R, S, sign = _kabsch_rotation(
            src_xyz_mu, tar_xyz_mu, normalize=True
        )  # (3, 3)

        # Compute scale
        src_var = np.var(src_xyz, axis=0).sum()  # scalar
        scale = (S[0] + S[1] + sign * S[2]) / max(src_var, 1e-9)  # scalar

        # Compute translation with equation,
        # t = target_centroid - s * R * source_centroid
        t = tar_centroid - scale * R @ src_centroid  # (3,)

        # Construct the transformation matrix
        T = np.eye(4, dtype=src_xyz.dtype)
        T[:3, :3] = scale * R
        T[:3,  3] = t
        return T, scale, R, t

    def robust_weighted_align_sim3(
        self,
        conf_thres: float = 0.1,
        delta: float = 0.1,
        max_iters: int = 5,
        exit_thres: float = 1e-9,
    ):
        # Assert
        assert max_iters > 0, "max_iters must be greater than 0."
        assert exit_thres > 0, "exit_thres must be greater than 0."
        assert delta > 0, "delta must be greater than 0."

        # Compute the weights and mask the points
        src_xyz, tar_xyz, ini_wet = self.compute_weight(conf_thres)

        # Pre-convert to float32 once (avoid repeated conversion in inner loop)
        src_xyz = np.ascontiguousarray(src_xyz[:, :3], dtype=np.float32)
        tar_xyz = np.ascontiguousarray(tar_xyz[:, :3], dtype=np.float32)
        ini_wet = np.ascontiguousarray(ini_wet, dtype=np.float32)

        # Compute the initial transformation matrix
        s, R, t = weighted_align_sim3(
            src_xyz, tar_xyz, ini_wet
        )  # scalar, (3, 3), (3,)

        # Initialize the previous error
        prev_error = float('inf')

        # Compute the residuals iteratively
        for _ in range(max_iters):
            # Transform the source point cloud using the current transformation
            mid = apply_transformation(src_xyz, s, R, t)  # (N, 3)
            # Compute the residuals
            res = compute_residual(tar_xyz, mid)  # (N,)

            # Compute the huber weights
            hub_wet = compute_huber_weight(res, delta)  # (N,)
            # Update the weights
            cur_wet = ini_wet * hub_wet  # (N,)
            cur_wet = cur_wet / (np.sum(cur_wet) + 1e-12)  # (N,)

            # Update the transformation
            s_, R_, t_ = weighted_align_sim3(
                src_xyz, tar_xyz, cur_wet
            )  # scalar, (3, 3), (3,)

            # Compute the change of the transformation
            t_change = np.abs(s_ - s) + np.linalg.norm(t_ - t)  # scalar
            r_change = np.arccos(
                min(1.0, max(-1.0, (np.trace(R_ @ R.T) - 1) / 2))
            )  # scalar

            # Compute the current error
            curr_error = np.sum(
                huber_loss(res, delta) * ini_wet
            )  # scalar

            # Check if the change is small enough
            if (t_change < exit_thres and r_change < np.radians(0.1)) or \
               (abs(prev_error - curr_error) < exit_thres * prev_error):
                break

            # Update iteration
            prev_error = curr_error
            s, R, t = s_, R_, t_

        # Return the final transformation
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = s * R
        T[:3,  3] = t
        return T, s, R, t

    def compute_weight(self, conf_thres=0.1):
        # Assert
        assert self.tar_cnf.shape == self.src_cnf.shape, \
            "tar_cnf and src_cnf must have the same shape."

        # Deal with shapes
        if self.tar_cnf.shape[-1] == 1:
            self.tar_cnf = self.tar_cnf[..., 0]  # (N,)
        if self.src_cnf.shape[-1] == 1:
            self.src_cnf = self.src_cnf[..., 0]  # (N,)

        # Compute the confidence threshold
        conf_thres = min(
            np.median(self.tar_cnf),
            np.median(self.src_cnf)
        ) * conf_thres

        msk = (self.tar_cnf > conf_thres) \
            & (self.src_cnf > conf_thres)
        # Mask the points with confidence less than the threshold
        src_xyz = self.src_xyz[msk]
        tar_xyz = self.tar_xyz[msk]

        # Compute and return the weights
        weights = np.sqrt(self.tar_cnf[msk] * self.src_cnf[msk])
        return src_xyz, tar_xyz, weights


def _kabsch_rotation(src_xyz: np.ndarray, tar_xyz: np.ndarray, normalize: bool = False):
    """ Compute a proper rotation (det=+1) using Kabsch with reflection handling
    Args:
        src_xyz (np.ndarray): Source point cloud of shape (N, 3)
        tar_xyz (np.ndarray): Target point cloud of shape (N, 3)
        normalize (bool): Whether to normalize the cross-covariance matrix
    Returns:
        R (np.ndarray): Rotation matrix of shape (3, 3)
        S (np.ndarray): Singular values of the cross-covariance matrix
        sign (float): Sign of the determinant of the rotation matrix
    """
    # Cross-covariance; no 1/N here (SE3 case); Sim3 will build a normalized H separately
    H = src_xyz.T @ tar_xyz  # (3, 3)
    H = H.astype(np.float32)  # ensure float32
    # Maybe normalize the cross-covariance matrix
    if normalize:
        H /= max(src_xyz.shape[0], 1)
    U, S, Vt = np.linalg.svd(H)  # U (3,3), S (3,), Vt (3,3)
    V = Vt.T
    # Reflection handling via D = diag(1,1,sign)
    sign = np.sign(np.linalg.det(V @ U.T))
    if sign == 0:  # extremely degenerate; treat as +1
        sign = 1.0
    D = np.diag([1.0, 1.0, sign])
    R = V @ D @ U.T  # proper rotation
    return R, S, sign


def weighted_align_sim3(
    src_xyz: np.ndarray,
    tar_xyz: np.ndarray,
    weights: np.ndarray,
):
    """ Compute weighted Sim(3) transformation.

    Expects float32 inputs (caller should pre-convert).
    """
    s, src_centroid, tar_centroid, H = _weighted_align_sim3(
        src_xyz, tar_xyz, weights
    )
    if s < 0:
        raise ValueError("Total weight too small for meaningful estimation")

    U, _, Vt = np.linalg.svd(H.astype(np.float32))  # float32 SVD
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = tar_centroid - s * R @ src_centroid
    return s, R, t


def _weighted_align_sim3(
    src_xyz: np.ndarray,
    tar_xyz: np.ndarray,
    weights: np.ndarray,
):
    # Check if the weights are all zero
    wet_sum = np.sum(weights)
    if wet_sum < 1e-6:
        raise ValueError("Total weight too small for meaningful estimation")

    # Normalize the weights
    w = weights / wet_sum

    # Compute the weighted centroids
    src_centroid = np.sum(w[:, None] * src_xyz, axis=0)  # (3,)
    tar_centroid = np.sum(w[:, None] * tar_xyz, axis=0)  # (3,)

    # Center the two set of points
    src_centered = src_xyz - src_centroid  # (N, 3)
    tar_centered = tar_xyz - tar_centroid  # (N, 3)

    # Compute the scale
    src_scale = np.sqrt(
        np.sum(w * np.sum(src_centered**2, axis=1))
    )  # scalar
    tar_scale = np.sqrt(
        np.sum(w * np.sum(tar_centered**2, axis=1))
    )  # scalar
    s = tar_scale / src_scale  # scalar

    # Compute the weighted and scaled source and target point clouds
    sqrt_w = np.sqrt(w)[:, None]
    src_weighted = (s * src_centered) * sqrt_w
    tar_weighted = tar_centered * sqrt_w

    # Compute the weighted cross-covariance matrix
    H = src_weighted.T @ tar_weighted
    return s, src_centroid, tar_centroid, H


def apply_transformation(
    xyz: np.ndarray,
    s: float, R: np.ndarray, t: np.ndarray
):
    return s * (xyz @ R.T) + t  # (N, 3)


def compute_residual(tar_xyz: np.ndarray, src_xyz: np.ndarray):
    diff = tar_xyz - src_xyz
    return np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)  # (N,)


def compute_huber_weight(residual: np.ndarray, delta: float = 0.1):
    return np.where(
        residual > delta, np.float32(delta) / residual, np.float32(1.0)
    ).astype(np.float32)  # (N,)


def huber_loss(residual: np.ndarray, delta: float = 0.1):
    residual = residual.astype(np.float32)
    delta = np.float32(delta)
    absr = np.abs(residual)
    return np.where(
        absr <= delta,
        np.float32(0.5) * residual ** 2,
        delta * (absr - np.float32(0.5) * delta)
    ).astype(np.float32)  # (N,)
