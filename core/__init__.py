"""Deterministic image-inpainting experiment core.

This package deliberately has no dependency on Hello Agents.  Agent tools will
call into this package, never the other way around.
"""

from .data import apply_observation_mask, load_rgb_image, save_image, save_mask
from .interpolation import nearest_neighbor_fill
from .masks import generate_observation_mask
from .metrics import composite_ssim, missing_region_mse, missing_region_psnr
from .pipeline import run_day1_baseline

__all__ = [
    "apply_observation_mask",
    "composite_ssim",
    "generate_observation_mask",
    "load_rgb_image",
    "missing_region_mse",
    "missing_region_psnr",
    "nearest_neighbor_fill",
    "run_day1_baseline",
    "save_image",
    "save_mask",
]

