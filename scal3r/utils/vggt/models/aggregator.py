# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
import torch.nn as nn
from typing import Tuple, List

from scal3r.utils.base_utils import dotdict
from scal3r.utils.vggt.layers import PatchEmbed
from scal3r.utils.vggt.layers.mlp import CamMlp
from scal3r.utils.vggt.layers.block import Block
from scal3r.utils.vggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from scal3r.utils.vggt.layers.vision_transformer import vit_base, vit_giant2, vit_large, vit_small

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        permutation_equivariant=False,
        num_camera_tokens=1,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        use_checkpoint=False,
        use_reentrant=True,
        use_chunkwise_checkpoint=False,
        use_cam_emb=False,
        use_cam_token=False,
        cam_emb_layer=[-5, None, 1],
        cam_embed_cfg=False,
        intermediate_layer_idx=[4, 11, 17, 23],  # used by the DPT head
        num_global_tokens=0,
        frame_use_ttt=False,
        global_use_ttt=False,
        ttt_layer_idx=list(range(24)),
        num_block_tokens=0,
        ttt_cfg=dotdict(
            type="FastWeightGluMLPMultihead",
            inter_multi=4,
            base_lr=0.01,
            muon_update_steps=5,
        ),
    ):
        super().__init__()

        self.__build_patch_embed__(
            patch_embed,
            img_size,
            patch_size,
            num_register_tokens,
            embed_dim=embed_dim,
            use_checkpoint=use_checkpoint,
            use_reentrant=use_reentrant,
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # Gradient checkpointing
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant
        self.use_chunkwise_checkpoint = use_chunkwise_checkpoint

        # Select the layers to use camera embedding
        cam_emb_inds = range(depth)[cam_emb_layer[0]:cam_emb_layer[1]:cam_emb_layer[2]]
        cam_emb_list = [
            True if use_cam_emb and i in cam_emb_inds and not use_cam_token else False
                for i in range(depth)
        ]

        # Maybe create a camera MLP
        if use_cam_emb:
            cam_mlp_cfg = dict(cam_embed_cfg)
            cam_mlp_cfg.pop("type", None)
            self.cam_mlp = CamMlp(**cam_mlp_cfg)
        else:
            self.cam_mlp = None

        # Some bookkeepings
        self.use_cam_emb = use_cam_emb
        self.use_cam_token = use_cam_token
        self.cam_embed_cfg = cam_embed_cfg

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    use_cam_emb=cam_emb_list[i],
                    cam_mlp=self.cam_mlp,
                    use_ttt=frame_use_ttt and i in ttt_layer_idx,
                    ttt_cfg=ttt_cfg.update(dotdict(index=i)),
                )
                for i in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    use_cam_emb=cam_emb_list[i],
                    cam_mlp=self.cam_mlp,
                    use_ttt=global_use_ttt and i in ttt_layer_idx,
                    ttt_cfg=ttt_cfg.update(dotdict(index=i)),
                )
                for i in range(depth)
            ]
        )

        # TTT-related caches only keep the last fast weights for release inference
        if frame_use_ttt:
            self.frame_ttt_caches = [
                dotdict(
                    last_weights_test=[],
                )
                for _ in range(len(self.frame_blocks))
            ]
        if global_use_ttt:
            self.global_ttt_caches = [
                dotdict(
                    last_weights_test=[],
                )
                for _ in range(len(self.global_blocks))
            ]
        # Bookkeepings for TTT
        self.ttt_cfg = ttt_cfg
        self.frame_use_ttt = frame_use_ttt
        self.global_use_ttt = global_use_ttt
        self.ttt_layer_idx = ttt_layer_idx

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.intermediate_layer_idx = intermediate_layer_idx

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2 if not permutation_equivariant else 1, num_camera_tokens, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2 if not permutation_equivariant else 1, num_register_tokens, embed_dim))

        self.global_token = None
        # Add some global tokens for storing the global coordinate information
        if num_global_tokens > 0:
            # NOTE: We do not distinguish the global tokens for different frames, so a single leading dimension is enough
            self.global_token = nn.Parameter(torch.randn(1, 2 if not permutation_equivariant else 1, num_global_tokens, embed_dim))
            # Initialize the global tokens with a small value
            nn.init.normal_(self.global_token, std=1e-6)

        self.block_token = None
        if num_block_tokens > 0:
            self.block_token = nn.Parameter(torch.randn(1, 2, num_block_tokens, embed_dim))
            nn.init.normal_(self.block_token, std=1e-6)

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = num_camera_tokens + num_register_tokens + num_global_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Bookkeepings
        self.permutation_equivariant = permutation_equivariant
        self.num_camera_tokens = num_camera_tokens
        self.num_register_tokens = num_register_tokens

        # Additional special tokens?
        self.num_global_tokens = num_global_tokens
        self.num_block_tokens = num_block_tokens

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

    @staticmethod
    def _get_ttt_state(ttt_cache):
        if ttt_cache is None:
            return None, tuple(), 0
        return ttt_cache, tuple(ttt_cache.last_weights_test), 0

    def initialize_global_with_register(self):
        """
        Initialize the global tokens with the register tokens.
        This is useful when we want to use the global tokens to
        store the global coordinate information.
        """
        if self.global_token is not None:
            with torch.no_grad():
                if self.num_global_tokens <= self.num_register_tokens:
                    self.global_token.data.copy_(
                        self.register_token.data[..., :self.num_global_tokens, :]
                    )
                else:
                    # If we have more global tokens than register tokens,
                    # we copy a multiple times
                    repeat_times = self.num_global_tokens // self.num_register_tokens
                    self.global_token.data.copy_(
                        self.register_token.data.repeat(1, 1, repeat_times, 1)[..., :self.num_global_tokens, :]
                    )
            print("Initialized global tokens with register tokens")
        else:
            print("No global tokens to initialize")

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
        use_checkpoint=False,
        use_reentrant=True,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
                use_checkpoint=use_checkpoint,
                use_reentrant=use_reentrant,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        cameras: torch.Tensor = None, # dim 9 [quaternion, translation, fx, fy]
        camera_dropout: float = False,
        camera_token: torch.Tensor = None,
        register_token: torch.Tensor = None,
        global_token: torch.Tensor = None,
        ttt_order: list = None,
        is_reference: int = -1,
        output: dict = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        state = self._prepare_tokens(
            images,
            cameras=cameras,
            camera_dropout=camera_dropout,
            camera_token=camera_token,
            register_token=register_token,
            global_token=global_token,
            is_reference=is_reference,
        )
        tokens = state["tokens"]
        pos = state["pos"]
        B = state["B"]
        S = state["S"]
        P = state["P"]
        C = state["C"]

        output_dict = {}
        if not self.use_chunkwise_checkpoint:
            frame_idx = 0
            global_idx = 0
            for _ in range(self.aa_block_num):
                layer_idx = frame_idx
                for attn_type in self.aa_order:
                    if attn_type == "frame":
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos, cam=cameras, cam_drop=camera_dropout,
                        )
                    elif attn_type == "global":
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos, cam=cameras, cam_drop=camera_dropout, ttt_order=ttt_order, output=output,
                        )
                    else:
                        raise ValueError(f"Unknown attention type: {attn_type}")
            
                for i in range(len(frame_intermediates)):
                    # concat frame and global intermediates, [B x S x P x 2C]
                    if layer_idx + i in self.intermediate_layer_idx:
                        concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                        output_dict[layer_idx + i] = concat_inter
        else:
            def forward_layers_chunkwise(tokens, start_block_idx, end_block_idx):
                output_dict = {}
                for block_idx in range(start_block_idx, end_block_idx):
                    layer_idx = block_idx * self.aa_block_size
                    for attn_type in self.aa_order:
                        if attn_type == "frame":
                            tokens, _, frame_intermediates = self._process_frame_attention(
                                tokens, B, S, P, C, layer_idx, pos=pos, cam=cameras, cam_drop=camera_dropout,
                            )
                        elif attn_type == "global":
                            tokens, _, global_intermediates = self._process_global_attention(
                                tokens, B, S, P, C, layer_idx, pos=pos, cam=cameras, cam_drop=camera_dropout, ttt_order=ttt_order, output=output,
                            )
                    for j in range(len(frame_intermediates)):
                        if layer_idx + j in self.intermediate_layer_idx:
                            output_dict[layer_idx + j] = torch.cat([frame_intermediates[j], global_intermediates[j]], dim=-1)
                return tokens, output_dict

            chunk_block_num = 2
            assert self.aa_block_num % chunk_block_num == 0, \
                f"Please make sure the aa_block_num ({self.aa_block_num}) is divisible by chunk_block_num ({chunk_block_num})"
            for block_idx in range(0, self.aa_block_num, chunk_block_num):
                tokens, output_dict_idx = torch.utils.checkpoint.checkpoint(
                    forward_layers_chunkwise,
                    tokens,
                    block_idx, block_idx + chunk_block_num,
                    use_reentrant=self.use_reentrant,
                )
                output_dict.update(output_dict_idx)

        assert self.depth - 1 in output_dict, \
            f"Please make sure the last layer ({self.depth - 1}) is in the output_dict: {output_dict.keys()}"
        output_dict[-1] = output_dict[self.depth - 1]

        return output_dict, self.patch_start_idx

    def _prepare_tokens(
        self,
        images: torch.Tensor,
        cameras: torch.Tensor = None,
        camera_dropout: float = False,
        camera_token: torch.Tensor = None,
        register_token: torch.Tensor = None,
        global_token: torch.Tensor = None,
        is_reference: int = -1,
    ):
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        if camera_token is None:
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        if register_token is None:
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        if global_token is None and self.num_global_tokens > 0:
            global_token = slice_expand_and_flatten(self.global_token, B, S)

        if self.num_block_tokens > 0 and self.block_token is not None and is_reference >= 0:
            if is_reference:
                block_token = self.block_token[:, 0] + self.block_token[:, 1] * 0.0
            else:
                block_token = self.block_token[:, 1] + self.block_token[:, 0] * 0.0
            camera_token = block_token.expand(B * S, -1, -1).contiguous() + camera_token

        if self.use_cam_emb and self.use_cam_token and cameras is not None:
            camera_prior = self.cam_mlp(cameras).reshape(B * S, 1, -1) * (1 - camera_dropout)
            camera_token = camera_token + camera_prior

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        if global_token is not None:
            tokens = torch.cat([global_token, tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if pos is not None and self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape
        return dict(tokens=tokens, pos=pos, B=B, S=S, P=P, C=C)

    def _process_frame_attention(
        self, tokens, B, S, P, C, frame_idx,
        pos=None, cam=None, cam_drop=False,
        ttt_order=None, enable_ttt=True,
    ):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)
        
        if cam is not None and cam.shape != (B * S, 9):
            cam = cam.view(B, S, 9).view(B * S, 9)

        intermediates = []

        # Deal with TTT parameters
        if self.frame_use_ttt:
            ttt_cache, ttt_fastw, ttt_steps = self._get_ttt_state(self.frame_ttt_caches[frame_idx])
        else:
            ttt_cache = None
            ttt_fastw = tuple()
            ttt_steps = 0

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](
                tokens, pos, cam, cam_drop,
                ttt_order, ttt_cache, ttt_fastw, ttt_steps, self.block_token, enable_ttt, B, S, P, C, self.patch_start_idx,
            )
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self, tokens, B, S, P, C, global_idx,
        pos=None, cam=None, cam_drop=False,
        ttt_order=None, enable_ttt=True,
        output=None,
    ):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        if cam is not None and cam.shape != (B * S, 9):
            cam = cam.view(B, S, 9).view(B * S, 9)

        intermediates = []

        # Deal with TTT parameters
        if self.global_use_ttt:
            ttt_cache, ttt_fastw, ttt_steps = self._get_ttt_state(self.global_ttt_caches[global_idx])
        else:
            ttt_cache = None
            ttt_fastw = tuple()
            ttt_steps = 0

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.global_blocks[global_idx](
                tokens, pos, cam, cam_drop,
                ttt_order, ttt_cache, ttt_fastw, ttt_steps, self.block_token, enable_ttt, B, S, P, C, self.patch_start_idx, output=output,
            )
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def prepare(
        self,
        images: torch.Tensor,
        cameras: torch.Tensor = None, # dim 9 [quaternion, translation, fx, fy]
        camera_dropout: float = False,
        camera_token: torch.Tensor = None,
        register_token: torch.Tensor = None,
        global_token: torch.Tensor = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        state = self._prepare_tokens(
            images,
            cameras=cameras,
            camera_dropout=camera_dropout,
            camera_token=camera_token,
            register_token=register_token,
            global_token=global_token,
        )
        return dict(
            tokens=state["tokens"].cpu(),
            pos=state["pos"].cpu() if state["pos"] is not None else None,
            B=state["B"],
            S=state["S"],
            P=state["P"],
            C=state["C"],
        )

    def forward_layer(
        self,
        index: int,
        tokens: torch.Tensor,
        B: int, S: int, P: int, C: int,
        pos: torch.Tensor = None,
        cameras: torch.Tensor = None,
        camera_dropout: bool = False,
        output: dict = None,
    ):
        # Perform one layer of attention based on the aa_order
        # NOTE: only pure original transformer block here, no TTT is enabled
        for attn_type in self.aa_order:
            if attn_type == "frame":
                tokens, _, frame_intermediates = self._process_frame_attention(
                    tokens, B, S, P, C, index,
                    pos=pos, cam=cameras, cam_drop=camera_dropout,
                    enable_ttt=False,
                )
            elif attn_type == "global":
                tokens, _, global_intermediates = self._process_global_attention(
                    tokens, B, S, P, C, index,
                    pos=pos, cam=cameras, cam_drop=camera_dropout,
                    enable_ttt=False,
                    output=output,
                )

        # Maybe store the intermediate feature for later DPT head decode
        for i in range(len(frame_intermediates)):
            if index + i in self.intermediate_layer_idx:
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                if output is not None:
                    output[index + i] = concat_inter.cpu()
        return dict(tokens=tokens.cpu(), pos=pos.cpu(), B=B, S=S, P=P, C=C)

    def ttt_gradient(
        self,
        index: int,
        tokens: torch.Tensor,
        B: int, S: int, P: int, C: int,
        pos: torch.Tensor = None,
        ttt_order: List = None,
    ):
        """ Only global attention is used for TTT gradient computation """
        assert self.global_use_ttt, "Global TTT must be enabled for compute_ttt_gradient"
        assert not self.training, "Model must not be in training mode for compute_ttt_gradient"

        # Deal with shape things
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        # Deal with TTT parameters
        ttt_cache, ttt_fastw, ttt_steps = self._get_ttt_state(self.global_ttt_caches[index])

        # Compute TTT inner loop update gradients, no actual update
        w0_grad, w1_grad, w2_grad = self.global_blocks[index].ttt.gradient(
            tokens, pos, ttt_order, ttt_cache, ttt_fastw, ttt_steps, self.block_token,
            batch_size=B, S=S, P=P, C=C, patch_start_idx=self.patch_start_idx,
        )
        return w0_grad, w1_grad, w2_grad

    def ttt_update(
        self,
        index: int,
        tokens: torch.Tensor,
        w0_grad: torch.Tensor,
        w1_grad: torch.Tensor,
        w2_grad: torch.Tensor,
        ttt_order: List = None,
        w0_ready: torch.Tensor = None,
        w1_ready: torch.Tensor = None,
        w2_ready: torch.Tensor = None,
        lr: float = None,
    ):
        # Assertions
        assert self.global_use_ttt, "Global TTT must be enabled for compute_ttt_gradient"
        assert not self.training, "Model must not be in training mode for compute_ttt_gradient"

        # Deal with TTT parameters
        _, ttt_fastw, _ = self._get_ttt_state(self.global_ttt_caches[index])

        # Update the TTT weights with the input computed gradients
        w0, w1, w2 = self.global_blocks[index].ttt.update(
            tokens, w0_grad, w1_grad, w2_grad, ttt_order, ttt_fastw,
            w0_ready=w0_ready, w1_ready=w1_ready, w2_ready=w2_ready,
            lr=lr,
        )
        return w0, w1, w2

    def ttt_apply(
        self,
        index: int,
        tokens: torch.Tensor,
        B: int, S: int, P: int, C: int,
        pos: torch.Tensor = None,
        ttt_order: List = None,
        w0: torch.Tensor = None,
        w1: torch.Tensor = None,
        w2: torch.Tensor = None,
        output: dict = None,
    ):
        """ Only global attention is used for TTT gradient computation """
        assert self.global_use_ttt, "Global TTT must be enabled for compute_ttt_gradient"
        assert not self.training, "Model must not be in training mode for compute_ttt_gradient"

        # Deal with shape things
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        # Deal with TTT parameters
        ttt_cache, ttt_fastw, ttt_steps = self._get_ttt_state(self.global_ttt_caches[index])

        # Apply the updated TTT weights to the current layer
        tokens += self.global_blocks[index].ttt(
            tokens, pos, ttt_order, ttt_cache, ttt_fastw, ttt_steps, self.block_token,
            w0_cache=w0, w1_cache=w1, w2_cache=w2,
            batch_size=B, S=S, P=P, C=C, patch_start_idx=self.patch_start_idx,
        )

        # Maybe store the intermediate feature for later DPT head decode
        if index in self.intermediate_layer_idx:
            concat_inter = torch.cat([output[index][..., :C], tokens.view(B, S, P, C).cpu()], dim=-1)
            if output is not None:
                output[index] = concat_inter.cpu()
        return dict(tokens=tokens.cpu(), pos=pos.cpu(), B=B, S=S, P=P, C=C)

def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """
    if token_tensor.shape[1] == 2:
        # Slice out the "query" tokens => shape (1, 1, ...)
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
        # Slice out the "other" tokens => shape (1, S-1, ...)
        others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
        # Concatenate => shape (B, S, ...)
        combined = torch.cat([query, others], dim=1)
    else:
        combined = token_tensor.expand(B, S, *token_tensor.shape[2:])

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
