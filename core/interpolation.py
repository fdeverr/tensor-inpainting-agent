"""Deterministic interpolation baselines."""

from __future__ import annotations

from collections import deque

import numpy as np


def nearest_neighbor_fill(
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    """Fill every hole from its nearest observed pixel in Manhattan distance.

    A multi-source breadth-first search starts at all observed pixels.  The
    implementation is O(HW), deterministic, and never reads hidden ground
    truth values from the missing locations.
    """

    if observed_image.ndim != 3 or observed_image.shape[2] != 3:
        raise ValueError("observed_image must have shape [H, W, 3]")
    if observed_mask.shape != observed_image.shape[:2] or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be bool and match image height and width")
    if not observed_mask.any():
        raise ValueError("nearest-neighbor filling requires at least one observed pixel")
    if not np.isfinite(observed_image).all():
        raise ValueError("observed_image contains NaN or Inf")

    reconstructed = observed_image.copy()
    visited = observed_mask.copy()
    queue = deque(zip(*np.nonzero(observed_mask)))
    height, width = observed_mask.shape

    while queue:
        y, x = queue.popleft()
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not 0 <= next_y < height or not 0 <= next_x < width:
                continue
            if visited[next_y, next_x]:
                continue
            reconstructed[next_y, next_x] = reconstructed[y, x]
            visited[next_y, next_x] = True
            queue.append((next_y, next_x))

    return reconstructed

