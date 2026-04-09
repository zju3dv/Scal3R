# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import torch
from torch import nn, Tensor
from typing import Callable, List, Any, Tuple, Dict

from scal3r.utils.base_utils import dotdict
from scal3r.utils.vggt.layers.mlp import CamMlp, Mlp
from scal3r.utils.vggt.layers.drop_path import DropPath
from scal3r.utils.vggt.layers.attention import Attention
from scal3r.utils.vggt.layers.layer_scale import LayerScale
from scal3r.utils.ttt_utils import FastWeightGluMLPMultihead

xformers_available = False


def frame_modulate(x, shift, scale):
    # Per-batch modulation
    # x: (B, S*P, C), shift: (B, C), scale: (B, C)
    return x * (1 + scale[:, None]) + shift[:, None]


def global_modulate(x, shift, scale):
    # Global modulation
    # x: (B*S, P, C), shift: (B, C), scale: (B, C)
    sh = x.shape
    return (
        x.view(scale.shape[0], -1, scale.shape[1]) * (1 + scale[:, None]) + shift[:, None]
    ).view(sh)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        attn_param: Dict[str, Any] = {},
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        use_cam_emb: bool = False,
        cam_mlp: Callable[..., nn.Module] = None,
        use_ttt: bool = False,
        ttt_cfg: dotdict = dotdict(
            inter_multi=4,
            base_lr=0.01,
            muon_update_steps=5,
        ),
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
            **attn_param,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

        # Use camera embedding if specified
        if use_cam_emb:
            if cam_mlp is not None:
                self.cam_mlp = cam_mlp
            else:
                self.cam_mlp = CamMlp(
                    in_features=9,
                    hidden_features=mlp_hidden_dim,
                    out_features=dim*4,
                    act_layer=act_layer,
                    drop=drop,
                    bias=ffn_bias,
                )
        self.use_cam_emb = use_cam_emb

        # Maybe use TTT
        if use_ttt:
            self.ttt = FastWeightGluMLPMultihead(
                dim=dim,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                qk_norm=qk_norm,
                fused_attn=fused_attn,
                rope=rope,  # maybe use rope to be consistent with the attention module?
                **dict(ttt_cfg),
            )
        self.use_ttt = use_ttt

    def forward(
            self,
            x: Tensor, pos=None, cams=None, cam_drop=False,
            ttt_order=None, ttt_cache=None, ttt_fastw=None, ttt_steps=None, ttt_token=None, enable_ttt=True,
            B=None, S=None, P=None, C=None, patch_start_idx=None,
            output=None,
        ) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None, scale_msa=None, shift_msa=None) -> Tensor:
            if not self.use_cam_emb or cams is None:
                return self.ls1(self.attn(self.norm1(x), pos=pos))
            else:
                assert (scale_msa is not None) and (shift_msa is not None), "shift_msa, scale_msa must be provided"
                func = frame_modulate if scale_msa.shape[0] == x.shape[0] else global_modulate
                shift_msa, scale_msa = shift_msa * (1 - cam_drop), scale_msa * (1 - cam_drop)
                return self.ls1(self.attn(func(self.norm1(x), shift_msa, scale_msa), pos=pos))

        def ffn_residual_func(x: Tensor, scale_mlp=None, shift_mlp=None) -> Tensor:
            if not self.use_cam_emb or cams is None:
                return self.ls2(self.mlp(self.norm2(x)))
            else:
                assert (scale_mlp is not None) and (shift_mlp is not None), "shift_mlp, scale_mlp must be provided"
                func = frame_modulate if scale_mlp.shape[0] == x.shape[0] else global_modulate
                shift_mlp, scale_mlp = shift_mlp * (1 - cam_drop), scale_mlp * (1 - cam_drop)
                return self.ls2(self.mlp(func(self.norm2(x), shift_mlp, scale_mlp)))

        # Apply TTT before the attention layer if specified
        if self.use_ttt and enable_ttt and self.ttt.ttt_before_attn:
            with torch.amp.autocast('cuda', enabled=False):
                x = x + self.ttt(
                    x, pos=pos,
                    ttt_order=ttt_order, ttt_cache=ttt_cache,
                    ttt_fastw=ttt_fastw, ttt_steps=ttt_steps,
                    ttt_token=ttt_token,
                    batch_size=B, S=S, P=P, C=C, patch_start_idx=patch_start_idx,
                    output=output,
                )  # use residual addition

        # Get the camera embedding if specified
        if self.use_cam_emb and cams is not None:
            shift_msa, scale_msa, shift_mlp, scale_mlp = self.cam_mlp(cams).chunk(4, dim=1)
        else:
            shift_msa, scale_msa, shift_mlp, scale_mlp = None, None, None, None

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, scale_msa=scale_msa, shift_msa=shift_msa,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x, scale_mlp=scale_mlp, shift_mlp=shift_mlp,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos, scale_msa=scale_msa, shift_msa=shift_msa))
            x = x + self.drop_path2(ffn_residual_func(x, scale_mlp=scale_mlp, shift_mlp=shift_mlp))
        else:
            x = x + attn_residual_func(x, pos=pos, scale_msa=scale_msa, shift_msa=shift_msa)
            x = x + ffn_residual_func(x, scale_mlp=scale_mlp, shift_mlp=shift_mlp)

        # Apply TTT after the attention layer if specified
        if self.use_ttt and enable_ttt and not self.ttt.ttt_before_attn:
            with torch.amp.autocast('cuda', enabled=False):
                x = x + self.ttt(
                    x, pos=pos,
                    ttt_order=ttt_order, ttt_cache=ttt_cache,
                    ttt_fastw=ttt_fastw, ttt_steps=ttt_steps,
                    ttt_token=ttt_token,
                    batch_size=B, S=S, P=P, C=C, patch_start_idx=patch_start_idx,
                    output=output,
                )  # use residual addition
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
    pos=None,
    scale=None,
    shift=None,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos, scale=scale, shift=shift)
    else:
        residual = residual_func(x_subset, scale=scale, shift=shift)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    x_flat = x.flatten(1)
    residual = residual.to(dtype=x.dtype)
    if scaling_vector is not None:
        residual = residual * scaling_vector.view(1, 1, -1)
    residual = residual.flatten(1)
    return torch.index_add(x_flat, 0, brange, residual, alpha=residual_scale_factor)


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    if branges is not None:
        gathered = [x[brange] for x, brange in zip(x_list, branges)]
        cat_tensors = torch.cat(gathered, dim=0)
    else:
        cat_tensors = torch.cat(x_list, dim=0)
    return None, cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    outputs = []
    for x in x_list:
        brange, residual_scale_factor = get_branges_scales(x, sample_drop_ratio=sample_drop_ratio)
        residual = residual_func(x[brange], attn_bias=None)
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        return [super(NestedTensorBlock, self).forward(x) for x in x_list]

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
