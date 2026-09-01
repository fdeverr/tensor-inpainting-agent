"""Tucker tensor decomposition baseline."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from .base import BaseTensorInpaintingModel


class TuckerDecomposition(BaseTensorInpaintingModel):
    """Use independent spatial/channel ranks and a learnable core tensor."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        initial_channel_mean: Sequence[float],
        rank_h: int = 16,
        rank_w: int = 16,
        rank_c: int = 3,
        init_scale: float = 0.15,
    ) -> None:
        super().__init__(image_shape, initial_channel_mean)
        height, width, channels = self.image_shape
        ranks = (int(rank_h), int(rank_w), int(rank_c))
        limits = (height, width, channels)
        if any(rank < 1 or rank > limit for rank, limit in zip(ranks, limits)):
            raise ValueError("Tucker ranks must be positive and not exceed image dimensions")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.ranks = ranks
        self.core = nn.Parameter(torch.empty(*ranks))
        self.height_factor = nn.Parameter(torch.empty(height, ranks[0]))
        self.width_factor = nn.Parameter(torch.empty(width, ranks[1]))
        self.channel_factor = nn.Parameter(torch.empty(channels, ranks[2]))
        nn.init.normal_(self.core, mean=0.0, std=init_scale)
        nn.init.normal_(self.height_factor, mean=0.0, std=init_scale)
        nn.init.normal_(self.width_factor, mean=0.0, std=init_scale)
        nn.init.normal_(self.channel_factor, mean=0.0, std=init_scale)

    def forward(self) -> torch.Tensor:
        decomposition = torch.einsum(
            "abc,ia,jb,kc->ijk",
            self.core,
            self.height_factor,
            self.width_factor,
            self.channel_factor,
        )
        return decomposition + self.channel_bias

    @classmethod
    def search_space(cls, image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        height, width, channels = image_shape
        rank_h = sorted({max(1, min(height, value)) for value in (4, 8, 16, 32)})
        rank_w = sorted({max(1, min(width, value)) for value in (4, 8, 16, 32)})
        rank_c = list(range(1, channels + 1))
        return {
            "rank_h": rank_h,
            "rank_w": rank_w,
            "rank_c": rank_c,
            "init_scale": [0.1, 0.15, 0.2],
        }

