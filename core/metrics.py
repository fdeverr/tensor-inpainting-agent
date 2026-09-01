"""Evaluation-only image reconstruction metrics."""

from __future__ import annotations

import math

import numpy as np


def _validate_metric_inputs(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
) -> None:
    if prediction.shape != ground_truth.shape:
        raise ValueError("prediction and ground_truth must have identical shapes")
    if prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError("images must have shape [H, W, 3]")
    if observed_mask.shape != prediction.shape[:2] or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be bool and match image height and width")
    if observed_mask.all():
        raise ValueError("at least one missing pixel is required for missing-region metrics")
    if not np.isfinite(prediction).all() or not np.isfinite(ground_truth).all():
        raise ValueError("metric inputs contain NaN or Inf")


def missing_region_mse(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    """Mean squared error over missing pixels and RGB channels only."""

    _validate_metric_inputs(prediction, ground_truth, observed_mask)
    difference = prediction[~observed_mask] - ground_truth[~observed_mask]
    return float(np.mean(np.square(difference, dtype=np.float64)))


def missing_region_psnr(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Peak signal-to-noise ratio calculated only in the missing region."""

    if data_range <= 0:
        raise ValueError("data_range must be positive")
    mse = missing_region_mse(prediction, ground_truth, observed_mask)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10((data_range * data_range) / mse))


def _gaussian_kernel(window_size: int, sigma: float) -> np.ndarray:
    coordinates = np.arange(window_size, dtype=np.float64) - window_size // 2
    kernel_1d = np.exp(-(coordinates ** 2) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    return np.outer(kernel_1d, kernel_1d)


def _filter2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padding = kernel.shape[0] // 2
    padded = np.pad(image, padding, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)


def _global_ssim(channel_x: np.ndarray, channel_y: np.ndarray, data_range: float) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mean_x = float(channel_x.mean())
    mean_y = float(channel_y.mean())
    variance_x = float(channel_x.var())
    variance_y = float(channel_y.var())
    covariance = float(np.mean((channel_x - mean_x) * (channel_y - mean_y)))
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
    denominator = (mean_x ** 2 + mean_y ** 2 + c1) * (
        variance_x + variance_y + c2
    )
    return numerator / denominator


def structural_similarity(
    image_x: np.ndarray,
    image_y: np.ndarray,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Compute mean RGB SSIM using a Gaussian local window."""

    if image_x.shape != image_y.shape or image_x.ndim != 3 or image_x.shape[2] != 3:
        raise ValueError("SSIM inputs must both have shape [H, W, 3]")
    if data_range <= 0 or sigma <= 0:
        raise ValueError("data_range and sigma must be positive")
    if not np.isfinite(image_x).all() or not np.isfinite(image_y).all():
        raise ValueError("SSIM inputs contain NaN or Inf")

    height, width, channels = image_x.shape
    effective_window = min(window_size, height, width)
    if effective_window % 2 == 0:
        effective_window -= 1

    channel_scores = []
    if effective_window < 3:
        for channel in range(channels):
            channel_scores.append(
                _global_ssim(image_x[..., channel], image_y[..., channel], data_range)
            )
        return float(np.mean(channel_scores))

    kernel = _gaussian_kernel(effective_window, sigma)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    for channel in range(channels):
        x = image_x[..., channel].astype(np.float64, copy=False)
        y = image_y[..., channel].astype(np.float64, copy=False)
        mean_x = _filter2d(x, kernel)
        mean_y = _filter2d(y, kernel)
        variance_x = np.maximum(0.0, _filter2d(x * x, kernel) - mean_x * mean_x)
        variance_y = np.maximum(0.0, _filter2d(y * y, kernel) - mean_y * mean_y)
        covariance = _filter2d(x * y, kernel) - mean_x * mean_y
        numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
        denominator = (mean_x * mean_x + mean_y * mean_y + c1) * (
            variance_x + variance_y + c2
        )
        channel_scores.append(float(np.mean(numerator / denominator)))

    return float(np.mean(channel_scores))


def composite_ssim(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    """Compute SSIM after restoring known pixels from ground truth.

    This is intentionally named *composite* SSIM.  It is not a standardized
    masked SSIM: known pixels are copied from the ground truth so all remaining
    error comes from the inpainted region, but known areas can still dilute the
    global average.
    """

    _validate_metric_inputs(prediction, ground_truth, observed_mask)
    composite = prediction.copy()
    composite[observed_mask] = ground_truth[observed_mask]
    return structural_similarity(composite, ground_truth)

