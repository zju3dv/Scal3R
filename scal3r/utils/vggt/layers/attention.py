# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import torch

from torch import Tensor, nn
import torch.nn.functional as F

xformers_available = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        # TODO: maybe find a better way to check for the flash attention v3
        # https://github.com/Dao-AILab/flash-attention/blob/4d3d2ff2163ac011bce1b16a2eb2ca90a75f9628/flash_attn/flash_attn_interface.py#L1140-L1214
        try:
            import flash_attn_interface
            if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
                self.flash_attn_interface = flash_attn_interface
            else:
                self.flash_attn_interface = None
        except ImportError:
            self.flash_attn_interface = None
        # FIXME: Disable flash attention v3 for now, it seems to have some issues?
        self.flash_attn_interface = None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(
        self,
        x: Tensor,
        pos: Tensor = None,
        ttt_order: list = None,
    ) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.flash_attn_interface is not None:
            if self.attn_drop.p > 0.0 and self.training:
                raise AssertionError("Dropout is not supported with flash attention v3")
            # FA3 will return an additional logsumexp of each row of the matrix QK^T * scaling
            # https://github.com/Dao-AILab/flash-attention/blob/4d3d2ff2163ac011bce1b16a2eb2ca90a75f9628/hopper/flash_attn_interface.py#L495-L571
            x = self.flash_attn_interface.flash_attn_func(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
            )[0]
            x = x.to(q.dtype)
        elif self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if attn_bias is not None:
            raise AssertionError("xFormers is not packaged in the public SCAL3R release.")
        return super().forward(x)
