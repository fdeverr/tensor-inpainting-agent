"""Low-rank matrix factorization baseline."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from .base import BaseTensorInpaintingModel


class MatrixFactorization(BaseTensorInpaintingModel):
    """Factorize the image after flattening width and channels together."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        initial_channel_mean: Sequence[float],
        rank: int = 16,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(image_shape, initial_channel_mean)
        height, width, channels = self.image_shape
        max_rank = min(height, width * channels)
        if not 1 <= rank <= max_rank:
            raise ValueError("matrix rank must be in [1, %d]" % max_rank)
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.rank = int(rank)
        factor_scale = init_scale / math.sqrt(self.rank)
        self.left_factor = nn.Parameter(torch.empty(height, self.rank))
        self.right_factor = nn.Parameter(torch.empty(self.rank, width * channels))
        nn.init.normal_(self.left_factor, mean=0.0, std=factor_scale)
        nn.init.normal_(self.right_factor, mean=0.0, std=init_scale)

    def forward(self) -> torch.Tensor:
        height, width, channels = self.image_shape
        low_rank_matrix = self.left_factor @ self.right_factor
        return low_rank_matrix.reshape(height, width, channels) + self.channel_bias

    @classmethod
    def search_space(cls, image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        max_rank = min(image_shape[0], image_shape[1] * image_shape[2])
        candidates = sorted({max(1, min(max_rank, value)) for value in (4, 8, 16, 32)})
        return {"rank": candidates, "init_scale": [0.05, 0.1, 0.2]}

