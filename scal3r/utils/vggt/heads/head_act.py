# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F

from scal3r.utils.cam_utils import svd_orthogonalize


def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear", pose_encoding_type="absT_quaR_FoV"):
    """
    Activate pose parameters with specified activation functions.

    Args:
        pred_pose_enc: Tensor containing encoded pose parameters [translation, quaternion, focal length]
        trans_act: Activation type for translation component
        quat_act: Activation type for quaternion component
        fl_act: Activation type for focal length component

    Returns:
        Activated pose parameters tensor
    """
    if pose_encoding_type == "absT_quaR_FoV":
        # Expected shape: (B, N, 9) where N is the number of tokens
        # assert pred_pose_enc.shape[-1] == 9, f"Expected 9 pose parameters, got {pred_pose_enc.shape[-1]}"
        T = pred_pose_enc[..., :3]
        quat = pred_pose_enc[..., 3:7]
        fl = pred_pose_enc[..., 7:]  # or fov
    elif pose_encoding_type == "absT_matR_FoV":
        # Expected shape: (B, N, 14) where N is the number of tokens
        # assert pred_pose_enc.shape[-1] == 14, f"Expected 14 pose parameters, got {pred_pose_enc.shape[-1]}"
        T = pred_pose_enc[..., :3]
        quat = pred_pose_enc[..., 3:12]
        fl = pred_pose_enc[..., 12:]
    else:
        raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act)  # or fov

    pred_pose_enc = torch.cat([T, quat, fl], dim=-1)

    return pred_pose_enc


def base_pose_act(pose_enc, act_type="linear"):
    """
    Apply basic activation function to pose parameters.

    Args:
        pose_enc: Tensor containing encoded pose parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated pose parameters
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    elif act_type == "svd_orthogonalize":
        assert pose_enc.shape[-1] == 9, \
            f"Expected 9 pose parameters for SVD orthogonalization, got {pose_enc.shape[-1]}"
        # Reshape to (B, N, 3, 3) for SVD orthogonalization
        pose_enc = pose_enc.reshape(pose_enc.shape[:-1] + (3, 3))
        pose_enc = svd_orthogonalize(pose_enc)
        # Reshape back to (B, N, 9)
        pose_enc = pose_enc.reshape(pose_enc.shape[:-2] + (9,))
        return pose_enc
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


def activate_head(
    out,
    activation="norm_exp",
    exps_max=6.9,  # math.log(1000.)
    conf_activation="expp1",
    conf_exps_max=4.6,  # math.log(50.)
    conf_relu_max=100,
):
    """
    Process network output to extract 3D points and confidence values.

    Args:
        out: Network output tensor (B, C, H, W)
        activation: Activation type for 3D points
        conf_activation: Activation type for confidence values

    Returns:
        Tuple of (3D points tensor, confidence tensor)
    """
    # Move channels from last dim to the 4th dimension => (B, H, W, C)
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,C expected

    # Split into xyz (first C-1 channels) and confidence (last channel)
    xyz = fmap[:, :, :, :-1]
    conf = fmap[:, :, :, -1]

    # if activation in [
    #     'norm_exp', 'exp', 'inv_log', 'xy_inv_log'
    # ]:
    #     # Clamp xyz to avoid overflow
    #     xyz = torch.clamp(xyz, max=exps_max)

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == 'leaky_relu':
        pts3d = F.leaky_relu(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":
        xy, z = xyz.split([2, 1], dim=-1)
        z = inverse_log_transform(z)
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif activation == "xy_exp":
        xy, z = xyz.split([2, 1], dim=-1)
        z = torch.exp(z)
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif activation == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # # Clamp confidence values
    # # NOTE: you need to clamp the confidence values before the activation
    # # since you can still get a inf/nan gradient from the activation derivative itself
    # if conf_activation in ['expp1', 'expp0']:
    #     conf = torch.clamp(conf, max=conf_exps_max)
    # elif conf_activation in ['elup2', 'leaky_relu', 'symelup2']:
    #     conf = torch.clamp(conf, max=conf_relu_max)

    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    elif conf_activation == "elup2":
        conf_out = F.elu(conf) + 2
    elif conf_activation == "symelup2":
        conf_out = F.elu(conf)
        mask = conf > conf_relu_max
        conf_out[mask] = conf_relu_max - F.elu(conf_relu_max - conf[mask])
        conf_out = conf_out + 2
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out


def inverse_log_transform(y):
    """
    Apply inverse log transform: sign(y) * (exp(|y|) - 1)

    Args:
        y: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))
