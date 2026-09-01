"""Image I/O and observation-mask application."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def _validate_rgb_array(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an RGB array with shape [H, W, 3]")
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError("expected a floating-point image array")
    if not np.isfinite(image).all():
        raise ValueError("image contains NaN or Inf")


def load_rgb_image(path: str, max_size: Optional[int] = None) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1].

    When ``max_size`` is given, the largest spatial dimension is resized to
    that value while preserving the aspect ratio.
    """

    source = Path(path)
    if not source.is_file():
        raise ValueError("image file does not exist: %s" % source)

    with Image.open(source) as pil_image:
        pil_image = pil_image.convert("RGB")
        if max_size is not None:
            width, height = pil_image.size
            scale = float(max_size) / float(max(width, height))
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            if (new_width, new_height) != (width, height):
                pil_image = pil_image.resize(
                    (new_width, new_height),
                    resample=Image.Resampling.LANCZOS,
                )
        array = np.asarray(pil_image, dtype=np.float32) / 255.0

    _validate_rgb_array(array)
    return array


def save_image(path: str, image: np.ndarray) -> None:
    """Save a float RGB image after clipping to [0, 1]."""

    _validate_rgb_array(image)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    uint8_image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(uint8_image).save(destination)


def save_mask(path: str, observed_mask: np.ndarray) -> None:
    """Save an observation mask with white=observed and black=missing."""

    if observed_mask.ndim != 2 or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be a bool array with shape [H, W]")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mask_image = observed_mask.astype(np.uint8) * 255
    Image.fromarray(mask_image).save(destination)


def apply_observation_mask(
    image: np.ndarray,
    observed_mask: np.ndarray,
    missing_fill_value: float = 0.0,
) -> np.ndarray:
    """Hide missing pixels without changing observed pixels."""

    _validate_rgb_array(image)
    if observed_mask.shape != image.shape[:2] or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be bool and match image height and width")
    if not 0.0 <= missing_fill_value <= 1.0:
        raise ValueError("missing_fill_value must be in [0, 1]")

    corrupted = image.copy()
    corrupted[~observed_mask] = missing_fill_value
    return corrupted
