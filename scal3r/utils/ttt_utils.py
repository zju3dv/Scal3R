# This code implements a bidirectional LaCT SwiGLU attention
# Reference: https://github.com/a1600012888/LaCT/blob/main/minimal_implementations/bidirectional_lact_layer.py

import math
import torch
import collections
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from scal3r.utils.base_utils import dotdict
from scal3r.utils.dist_utils import context_parallelism


class CompatRMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | Tuple[int, ...],
        eps: float | None = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: Tensor):
        dims = tuple(range(-len(self.normalized_shape), 0))
        eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        output = x * torch.rsqrt(x.pow(2).mean(dims, keepdim=True) + eps)
        if self.weight is not None:
            output = output * self.weight
        return output


RMSNorm = getattr(nn, 'RMSNorm', CompatRMSNorm)


TTTOperator = collections.namedtuple(
    "TTTOperator",
    [
        "s",
        "e",
        "update",
        "apply",
        "use_cached",
        "cache_last",
    ]
)


def inv_softplus(x):
    """ Inverse of the softplus function """
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """ Backpropagation for the Swish (Silu) activation function

    Args:
        dy (torch.Tensor): (B, C, L), gradient of the outer loss wrt the y
        x (torch.Tensor): (B, C, L), input of the silu activation
    Returns:
        dx (torch.Tensor): (B, C, L), gradient of the outer loss wrt the x
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


# @torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """ Modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49
    Major change: G is (B, C, C) rather than (C, C).

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Args:
        G (torch.Tensor): (B, C, C) matrix to orthogonalize
        steps, (int): Number of iterations to perform
    Returns:
        X (torch.Tensor): (B, C, C) orthogonalized matrix
    """
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()  # FIXME: may cause 2% - 4% relative gradient error when using checkpoint through time
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


# @torch.compile
def fast_weight_swish_glu_weight_norm_mini_batch_apply(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
    use_ddp_allreduce: bool = False,
):
    """ Fast weight SwiGLU with weight normalization for mini-batch application.
    NOTE: the forward (or the output) is calculated via (silu(x @ w0) * (x @ w2)) @ w1

    Args:
        w0 (torch.Tensor): (B, C, Ch), weight for the first linear layer
        w1 (torch.Tensor): (B, Ch, C), weight for the second linear layer
        w2 (torch.Tensor): (B, C, Ch), weight for the third linear layer
        q, k, v (torch.Tensor): (B, L, C{q, k, v}), query, key, value tensors
        lr0, lr1, lr2 (torch.Tensor): (B, L, 1), learning rates for the three linear layers
    Returns:
        output (torch.Tensor): (B, L, C), the output of the attention layer
        w0, w1, w2 (torch.Tensor): updated weights after applying the learning rates
    """
    # Remember that the weight norm before the update is applied
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    # Mini-batch manner
    # NOTE: TTT layer operates in a mixed precision manner
    # NOTE: The learning rate lr0, lr1, lr2 are in fp32， the input q, k, v are in bf16
    output = []

    # Iterative TTT update and application for each batch
    for s, e, update, apply, *_ in ttt_ua_order:
        # Iterative variables
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            # NOTE: `s:None` is the same as `s:`
            ki, vi = k[:, s:e, :], v[:, s:e, :]  # (B, L, C), (B, L, C)
            lr0i = lr0[:, s:e, :]  # (B, L, 1)
            lr1i = lr1[:, s:e, :]  # (B, L, 1)
            lr2i = lr2[:, s:e, :]  # (B, L, 1)

            # Calculate hidden -> SiLU(W1x) ◦ (W3x)
            gate_before_act = ki @ w0_now  # (B, L, C) @ (B, C, Ch) -> (B, L, Ch)
            hide_before_mul = ki @ w2_now  # (B, L, C) @ (B, C, Ch) -> (B, L, Ch)
            hidden = F.silu(gate_before_act, inplace=False) * hide_before_mul  # (B, L, Ch)

            # Compute the middle local gradients
            dhidden = vi @ w1_now.transpose(-1, -2)  # (B, L, C) @ (B, C, Ch) -> (B, L, Ch)
            dhide_before_mul = dhidden * F.silu(gate_before_act, inplace=False)  # (B, L, Ch)
            dgate = dhidden * hide_before_mul  # (B, L, Ch)
            dgate_before_act = silu_backprop(dgate, gate_before_act)  # (B, L, Ch)

            # Calculate the gradients for w0, w1, w2
            w0_grad = (ki * lr0i).transpose(-1, -2) @ dgate_before_act  # (B, C, L) @ (B, L, Ch) -> (B, C, Ch)
            w1_grad = (hidden * lr1i).transpose(-1, -2) @ vi  # (B, Ch, L) @ (B, L, C) -> (B, Ch, C)
            w2_grad = (ki * lr2i).transpose(-1, -2) @ dhide_before_mul  # (B, C, L) @ (B, L, Ch) -> (B, C, Ch)

            # Maybe perform context parallelism here
            if use_ddp_allreduce:
                w0_grad, w1_grad, w2_grad = context_parallelism(
                    w0_grad, w1_grad, w2_grad,
                )

            # Apply the Muon update here
            w0_grad = zeropower_via_newtonschulz5(
                w0_grad, muon_update_steps
            )
            w1_grad = zeropower_via_newtonschulz5(
                w1_grad, muon_update_steps
            )
            w2_grad = zeropower_via_newtonschulz5(
                w2_grad, muon_update_steps
            )

            # Update the fast weights with the gradients
            w0_now = w0_now + w0_grad
            w1_now = w1_now + w1_grad
            w2_now = w2_now + w2_grad

            # Do weight normalization here
            w0_now = w0_now / (
                w0_now.norm(dim=1, keepdim=True) + 1e-5
            ) * w0_norm
            w1_now = w1_now / (
                w1_now.norm(dim=1, keepdim=True) + 1e-5
            ) * w1_norm
            w2_now = w2_now / (
                w2_now.norm(dim=1, keepdim=True) + 1e-5
            ) * w2_norm

            # Output assignment
            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            # Only in the last repeat # ? what does this mean?
            # o = fw(q) = W2 * (SiLU(W1x) ◦ (W3x))
            qi = q[:, s:e, :]
            oi = (F.silu(qi @ w0_now, inplace=False) * (qi @ w2_now)) @ w1_now  # (B, L, C)
            output.append(oi)

    # Concatenate the output along the sequence dimension,
    # output is a list of tensors, each of shape (B, Li, C)
    output = torch.cat(output, dim=1)  # (B, L, C)

    return output, w0, w1, w2


# @torch.compile
def fast_weight_swish_glu_weight_norm_mini_batch_gradient(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    ttt_ua_order: list,
):
    """ Fast weight SwiGLU with weight normalization for mini-batch application.
    NOTE: the forward (or the output) is calculated via (silu(x @ w0) * (x @ w2)) @ w1

    Args:
        w0 (torch.Tensor): (B, C, Ch), weight for the first linear layer
        w1 (torch.Tensor): (B, Ch, C), weight for the second linear layer
        w2 (torch.Tensor): (B, C, Ch), weight for the third linear layer
        q, k, v (torch.Tensor): (B, L, C{q, k, v}), query, key, value tensors
        lr0, lr1, lr2 (torch.Tensor): (B, L, 1), learning rates for the three linear layers
    Returns:
        output (torch.Tensor): (B, L, C), the output of the attention layer
        w0, w1, w2 (torch.Tensor): updated weights after applying the learning rates
    """
    # Only get the gradient
    s, e, update, apply, *_ = ttt_ua_order
    assert update, "Only support update operation for gradient computation."

    # NOTE: `s:None` is the same as `s:`
    ki, vi = k[:, s:e, :], v[:, s:e, :]  # (B, L, C), (B, L, C)
    lr0i = lr0[:, s:e, :]  # (B, L, 1)
    lr1i = lr1[:, s:e, :]  # (B, L, 1)
    lr2i = lr2[:, s:e, :]  # (B, L, 1)

    # Calculate hidden -> SiLU(W1x) ◦ (W3x)
    gate_before_act = ki @ w0  # (B, L, C) @ (B, C, Ch) -> (B, L, Ch)
    hide_before_mul = ki @ w2  # (B, L, C) @ (B, C, Ch) -> (B, L, Ch)
    hidden = F.silu(gate_before_act, inplace=False) * hide_before_mul  # (B, L, Ch)

    # Compute the middle local gradients
    dhidden = vi @ w1.transpose(-1, -2)  # (B, L, C) @ (B, C, Ch) -> (B, L, Ch)
    dhide_before_mul = dhidden * F.silu(gate_before_act, inplace=False)  # (B, L, Ch)
    dgate = dhidden * hide_before_mul  # (B, L, Ch)
    dgate_before_act = silu_backprop(dgate, gate_before_act)  # (B, L, Ch)

    # Calculate the gradients for w0, w1, w2
    w0_grad = (ki * lr0i).transpose(-1, -2) @ dgate_before_act  # (B, C, Ch)
    w1_grad = (hidden * lr1i).transpose(-1, -2) @ vi  # (B, Ch, C)
    w2_grad = (ki * lr2i).transpose(-1, -2) @ dhide_before_mul  # (B, C, Ch)

    # NOTE: we return the raw local gradients here, not the orthogonalized ones
    # NOTE: the muon update is applied in the update method
    return w0_grad, w1_grad, w2_grad


class FastWeightGluMLPMultihead(nn.Module):
    """ On the initialization of fast weight.

    Let's start with the magnitude of the value.
    - QKV projection is initialized with uniform distribution with range [-1.0/sqrt(d), 1.0/sqrt(d)]
    - x is layer normalizaed, so value is unit norm total (not per head, per head is 1.0/sqrt(num_head))
      during initialization, after silu, value is around norm of 2.7 per head
    NOTE: why? seems wired...

    - Then for the fast weight, assume initial lr = 0.
    - Then with l2 norm of q, k, input is unit normed, if w0 is initialized with kaiming,
      relu(w0 @ q) is unit normed.
    - Then w1 is initialized with kaiming, so w1 @ relu(w0 @ q) is of norm sqrt(2) per head
    - Since I compute total norm, it is sqrt(2) * sqrt(num_head), which is
      around 2.7 for dim=512, num_head=4.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,  # NOTE: should be small to make head dimension large, large head dimension is better for TTT
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope: nn.Module = None,
        bias: bool = False,
        base_lr: float = 0.01,
        inter_multi: int = 1,
        muon_update_steps: int = 5,
        ttt_before_attn: bool = False,
        use_ddp_allreduce: bool = False,
        use_modulation: bool = True,
        patch_size: int = 14,  # default patch size of DINOv2
        index: int = 0,  # debugging
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        # Initialize parameters
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        # Query, Key, Value linear layers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # Projection layers
        self.proj = nn.Linear(dim, dim if not use_modulation else dim * 3, bias=proj_bias)
        # # Zero-init the projection layer to make TTT no effect at the beginning of finetuning
        # nn.init.zeros_(self.proj.weight)
        # nn.init.zeros_(self.proj.bias)

        if use_modulation:
            # Adaptive layer normalization without affine parameters
            self.a_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self._init_modulation_params(dim, g_init=0.1)

        # Useless dropout layer, for compatibility
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Rotary positional encoding (optional)
        self.rope = rope
        self.muon_update_steps = muon_update_steps

        # TTT fast weight dimensions
        di = self.head_dim
        do = self.head_dim
        dh = self.head_dim * inter_multi

        # TTT fast weight parameters initialization
        self.w0 = nn.Parameter(
            torch.randn(
                self.num_heads, di, dh
            ) * (math.sqrt(2) / math.sqrt(di))
        )
        self.w1 = nn.Parameter(
            torch.randn(
                self.num_heads, dh, do
            ) * (math.sqrt(2) / math.sqrt(dh))
        )
        self.w2 = nn.Parameter(
            torch.randn(
                self.num_heads, di, dh
            ) * (math.sqrt(2) / math.sqrt(di))
        )

        # Learning rates for the fast weight
        dl = self.num_heads
        self.lrs = nn.Linear(dim, dl * 3)
        self.inv_base_lr = inv_softplus(base_lr)
        self.base_lr = base_lr

        # Output normalization
        # TODO: figure out what is difference between RMSNorm and LayerNorm
        self.o_norm = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

        self.patch_size = patch_size

        # Bookkeepings
        self.ttt_before_attn = ttt_before_attn
        self.use_modulation = use_modulation
        self.use_ddp_allreduce = use_ddp_allreduce
        self.index = index

    def _init_modulation_params(self, dim: int, g_init = 0.1):
        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self.proj.bias[2 * dim:] = g_init  # gate bias only

    def _flatten_patch_tokens(
        self,
        x: Tensor,
        pos: torch.Tensor | None,
        batch_size: int,
        S: int,
        P: int,
        C: int,
        patch_start_idx: int,
    ):
        x = x.reshape(batch_size, S, P, C)
        x = x[:, :, patch_start_idx:]
        x = x.reshape(batch_size, -1, C)
        if pos is None:
            return x, None

        pos = pos.reshape(batch_size, S, P, 2)
        pos = pos[:, :, patch_start_idx:]
        pos = pos.reshape(batch_size, -1, 2)
        return x, pos

    def _restore_patch_tokens(
        self,
        x: Tensor,
        batch_size: int,
        S: int,
        P: int,
        C: int,
        patch_start_idx: int,
    ) -> Tensor:
        x = x.reshape(batch_size, S, P - patch_start_idx, C)
        x = torch.cat([torch.zeros_like(x[:, :, :patch_start_idx]), x], dim=-2)
        return x.reshape(batch_size, -1, C)

    def _project_output(self, x: Tensor, batch_size: int):
        x = self.o_norm(x)
        x = rearrange(x, '(b h) l d -> b l (h d)', h=self.num_heads, b=batch_size)
        if not self.use_modulation:
            return self.proj(x)

        shift, scale, gate = self.proj(x).chunk(3, dim=-1)
        return gate * (self.a_norm(x) * (1 + scale) + shift)

    def forward(
        self,
        x: Tensor,
        pos: torch.Tensor = None,
        ttt_order: List[TTTOperator] = None,
        ttt_cache: Dict[str, List[torch.Tensor]] = None,
        ttt_fastw: Tuple[torch.Tensor] = None,  # ! must be tuple to avoid in-place modification
        ttt_steps: int = 0,
        ttt_token: torch.Tensor = None,
        w0_cache: torch.Tensor = None,  # only used for flexible inference
        w1_cache: torch.Tensor = None,
        w2_cache: torch.Tensor = None,
        batch_size: int = None, S: int = None, P: int = None, C: int = None, patch_start_idx: int = None,
        output: dotdict = None,
    ) -> Tensor:
        if self.training:
            raise RuntimeError("Release TTT only supports inference/eval mode.")
        assert isinstance(ttt_fastw, tuple) or ttt_fastw is None, \
            "ttt_fastw must be a tuple or None"
        assert batch_size is not None and S is not None and P is not None and C is not None and patch_start_idx is not None, \
            "batch_size, S, P, C, patch_start_idx must be provided for release inference"
        x, pos = self._flatten_patch_tokens(x, pos, batch_size, S, P, C, patch_start_idx)

        B, L, D = x.shape

        qkv = self.qkv(x)  # (B, L, 3 * D)
        q, k, v = rearrange(
            qkv, 'b l (qkv h d) -> qkv b h l d',
            qkv=3, h=self.num_heads
        )  # (B, num_heads, L, d)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)  # (B, num_heads, L, d)
            k = self.rope(k, pos)  # (B, num_heads, L, d)

        q = rearrange(q, 'b h l d -> (b h) l d')
        k = rearrange(k, 'b h l d -> (b h) l d')
        v = rearrange(v, 'b h l d -> (b h) l d')

        with torch.amp.autocast('cuda', enabled=False):
            lrs = self.lrs(x.float())  # (B, L, 3 * self.num_heads)
        lrs = F.softplus(lrs.float() + self.inv_base_lr)
        lr0, lr1, lr2 = rearrange(
            lrs, "b l (n h c) -> n (b h) l c",
            n=3, h=self.num_heads
        )  # (B * num_heads, L, D)

        if w0_cache is not None:
            w0 = w0_cache
            w1 = w1_cache
            w2 = w2_cache
        elif ttt_order[0].use_cached and len(ttt_fastw):
            w0 = ttt_fastw[0]
            w1 = ttt_fastw[1]
            w2 = ttt_fastw[2]
        else:
            w0 = self.w0.repeat(B, 1, 1)
            w1 = self.w1.repeat(B, 1, 1)
            w2 = self.w2.repeat(B, 1, 1)

        x, w0_, w1_, w2_ = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            q, k, v, w0, w1, w2, lr0, lr1, lr2,
            ttt_ua_order=ttt_order,
            muon_update_steps=self.muon_update_steps,
            use_ddp_allreduce=self.use_ddp_allreduce,
        )
        x = self._project_output(x, B)
        x = self._restore_patch_tokens(x, batch_size, S, P, C, patch_start_idx)
        if ttt_order[-1].cache_last:
            ttt_cache.last_weights_test = [w0_, w1_, w2_]
        else:
            ttt_cache.last_weights_test = []

        return x

    def gradient(
        self,
        x: Tensor,
        pos: torch.Tensor = None,
        ttt_order: List[TTTOperator] = None,
        ttt_cache: Dict[str, List[torch.Tensor]] = None,
        ttt_fastw: Tuple[torch.Tensor] = None,  # ! must be tuple to avoid in-place modification
        ttt_steps: int = 0,
        ttt_token: torch.Tensor = None,
        batch_size: int = None, S: int = None, P: int = None, C: int = None, patch_start_idx: int = None,
    ) -> Tensor:
        if self.training:
            raise RuntimeError("Release TTT only supports inference/eval mode.")
        assert isinstance(ttt_fastw, tuple) or ttt_fastw is None, \
            "ttt_fastw must be a tuple or None"
        assert batch_size is not None and S is not None and P is not None and C is not None and patch_start_idx is not None, \
            "batch_size, S, P, C, patch_start_idx must be provided for release inference"
        x, pos = self._flatten_patch_tokens(x, pos, batch_size, S, P, C, patch_start_idx)
        B, L, D = x.shape

        # Get query, key, value tensors
        qkv = self.qkv(x)  # (B, L, 3 * D)
        q, k, v = rearrange(
            qkv, 'b l (qkv h d) -> qkv b h l d',
            qkv=3, h=self.num_heads
        )  # (B, num_heads, L, d)

        # Maybe perform normalization on q and k
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply rotary positional encoding if provided
        if self.rope is not None:
            q = self.rope(q, pos)  # (B, num_heads, L, d)
            k = self.rope(k, pos)  # (B, num_heads, L, d)

        # Reshape to (B * num_heads, L, d)
        q = rearrange(q, 'b h l d -> (b h) l d')
        k = rearrange(k, 'b h l d -> (b h) l d')
        v = rearrange(v, 'b h l d -> (b h) l d')

        # TTT fast weight learning rates should be in fp32
        with torch.amp.autocast('cuda', enabled=False):
            lrs = self.lrs(x.float())  # (B, L, 3 * self.num_heads)
        lrs = F.softplus(lrs.float() + self.inv_base_lr)

        # Split learning rates for w0, w1, w2
        lr0, lr1, lr2 = rearrange(
            lrs, "b l (n h c) -> n (b h) l c",
            n=3, h=self.num_heads
        )  # (B * num_heads, L, D)

        if ttt_order[0].use_cached and len(ttt_fastw):
            w0 = ttt_fastw[0]
            w1 = ttt_fastw[1]
            w2 = ttt_fastw[2]
        else:
            w0 = self.w0.repeat(B, 1, 1)
            w1 = self.w1.repeat(B, 1, 1)
            w2 = self.w2.repeat(B, 1, 1)

        w0_grad, w1_grad, w2_grad = fast_weight_swish_glu_weight_norm_mini_batch_gradient(
            q, k, v, w0, w1, w2, lr0, lr1, lr2,
            ttt_ua_order=ttt_order[0],
        )

        return w0_grad, w1_grad, w2_grad

    def update(
        self,
        x: Tensor,
        w0_grad: torch.Tensor,
        w1_grad: torch.Tensor,
        w2_grad: torch.Tensor,
        ttt_order: List[TTTOperator] = None,
        ttt_fastw: Tuple[torch.Tensor] = None,  # ! must be tuple to avoid in-place modification
        w0_ready: torch.Tensor = None,
        w1_ready: torch.Tensor = None,
        w2_ready: torch.Tensor = None,
        lr: float = None,
    ):
        if self.training:
            raise RuntimeError("Release TTT only supports inference/eval mode.")
        if w0_ready is not None:
            w0, w1, w2 = w0_ready, w1_ready, w2_ready
        elif ttt_order[0].use_cached and len(ttt_fastw):
            w0, w1, w2 = ttt_fastw[0], ttt_fastw[1], ttt_fastw[2]
        else:
            w0 = self.w0.repeat(x.shape[0], 1, 1)
            w1 = self.w1.repeat(x.shape[0], 1, 1)
            w2 = self.w2.repeat(x.shape[0], 1, 1)

        # Remember that the weight norm before the update is applied
        w0_norm = w0.detach().norm(dim=1, keepdim=True)
        w1_norm = w1.detach().norm(dim=1, keepdim=True)
        w2_norm = w2.detach().norm(dim=1, keepdim=True)

        # Apply the Muon update here
        w0_grad = zeropower_via_newtonschulz5(
            w0_grad, self.muon_update_steps
        )
        w1_grad = zeropower_via_newtonschulz5(
            w1_grad, self.muon_update_steps
        )
        w2_grad = zeropower_via_newtonschulz5(
            w2_grad, self.muon_update_steps
        )

        # Maybe apply the learning rate
        if lr is not None:
            w0_grad = w0_grad * lr
            w1_grad = w1_grad * lr
            w2_grad = w2_grad * lr

        # Update the weights
        w0 = w0 + w0_grad
        w1 = w1 + w1_grad
        w2 = w2 + w2_grad

        # Do weight normalization here
        w0 = w0 / (
            w0.norm(dim=1, keepdim=True) + 1e-5
        ) * w0_norm
        w1 = w1 / (
            w1.norm(dim=1, keepdim=True) + 1e-5
        ) * w1_norm
        w2 = w2 / (
            w2.norm(dim=1, keepdim=True) + 1e-5
        ) * w2_norm

        return w0, w1, w2
