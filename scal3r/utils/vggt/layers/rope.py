# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import torch
import torch.nn as nn
from typing import Dict, Tuple
import torch.nn.functional as F

class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

        # Compute feature dimension for each spatial direction
        feature_dim = tokens.size(-1) // 2

        # Get frequency components
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)


class RankRotaryEmbedding1D(nn.Module):
    """Apply a 1D rotary embedding based on the GPU (rank) index.

    This module assumes that each DDP process sees tokens of shape (B, L, D),
    and we want to apply a rank-dependent RoPE feature of shape (1, 1, D)
    to all tokens on that rank.

    Args:
        base_frequency: Base frequency for rotary embedding.
        scaling_factor: Optional scaling factor for frequencies.
    """

    def __init__(self, base_frequency: float = 64.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = base_frequency
        self.scaling_factor = scaling_factor
        # Cache: (dim, seq_len, device, dtype) -> (cos, sin)
        self.freq_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_freq(
        self,
        dim: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Pre-compute cos/sin tables for ranks in [0, seq_len-1].

        Args:
            dim: Feature dimension (must be even).
            seq_len: Number of positions, here typically group_size.
        """
        key = (dim, seq_len, device, dtype)
        if key in self.freq_cache:
            return self.freq_cache[key]

        assert dim % 2 == 0, "Feature dimension for RoPE must be even"

        # Compute inverse frequencies for half dimension
        half_dim = dim // 2
        exponents = torch.arange(0, half_dim, 1, device=device, dtype=torch.float32)
        exponents = exponents / half_dim  # in [0, 1)
        inv_freq = (self.base_frequency ** (-exponents)) * self.scaling_factor  # (half_dim,)

        # Positions are the ranks: 0, 1, ..., seq_len-1
        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)  # (seq_len,)
        angles = torch.einsum("i,j->ij", positions, inv_freq)  # (seq_len, half_dim)

        # Duplicate for even/odd channels
        angles_full = torch.cat([angles, angles], dim=-1)  # (seq_len, dim)
        cos = angles_full.cos().to(dtype)
        sin = angles_full.sin().to(dtype)

        self.freq_cache[key] = (cos, sin)
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """ Rotate last dimension by splitting into pairs.

        x: (..., D), where D is even.
        """
        d = x.size(-1)
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        rank: int,
        group_size: int
    ) -> torch.Tensor:
        """ Apply rank-dependent RoPE to all tokens.

        Args:
            x: Input tensor, shape (B, L, D).
            rank: Local rank of this DDP process.
            group_size: Total number of ranks (DDP world size).

        Returns:
            Tensor with the same shape as x, but RoPE-rotated by the given rank.
        """
        assert x.dim() == 3, "Expected x of shape (B, L, D)"
        B, L, D = x.shape
        assert D % 2 == 0, "Last dimension D must be even for RoPE"

        device = x.device
        dtype = x.dtype

        # Precompute cos/sin for all ranks in [0, group_size-1]
        cos_table, sin_table = self._compute_freq(D, group_size, device, dtype)
        # Select the entry for the given rank
        rank_idx = torch.tensor(rank, device=device, dtype=torch.long)
        cos_rank = cos_table[rank_idx]  # (D,)
        sin_rank = sin_table[rank_idx]  # (D,)

        # Reshape to (1, 1, D) so it can broadcast over (B, L, D)
        cos_rank = cos_rank.view(1, 1, D)
        sin_rank = sin_rank.view(1, 1, D)

        # Apply RoPE along the feature dimension
        x_rot = self._rotate_half(x)
        return x * cos_rank + x_rot * sin_rank
