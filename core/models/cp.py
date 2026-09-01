"""Canonical polyadic tensor decomposition baseline."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from .base import BaseTensorInpaintingModel


class CPDecomposition(BaseTensorInpaintingModel):
    """Represent an RGB image as a sum of rank-one third-order tensors."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        initial_channel_mean: Sequence[float],
        rank: int = 12,
        init_scale: float = 0.2,
    ) -> None:
        super().__init__(image_shape, initial_channel_mean)
        height, width, channels = self.image_shape
        if rank < 1:
            raise ValueError("CP rank must be positive")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.rank = int(rank)
        self.height_factor = nn.Parameter(torch.empty(height, self.rank))
        self.width_factor = nn.Parameter(torch.empty(width, self.rank))
        self.channel_factor = nn.Parameter(torch.empty(channels, self.rank))
        nn.init.normal_(self.height_factor, mean=0.0, std=init_scale)
        nn.init.normal_(self.width_factor, mean=0.0, std=init_scale)
        nn.init.normal_(self.channel_factor, mean=0.0, std=init_scale)

    def forward(self) -> torch.Tensor:
        decomposition = torch.einsum(
            "ir,jr,kr->ijk",
            self.height_factor,
            self.width_factor,
            self.channel_factor,
        )
        return decomposition + self.channel_bias

    @classmethod
    def search_space(cls, image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        del image_shape
        return {"rank": [4, 8, 12, 16, 24], "init_scale": [0.1, 0.2, 0.3]}

