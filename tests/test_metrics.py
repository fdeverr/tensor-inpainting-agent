import math

import numpy as np

from inpainting_research_agent.core.metrics import (
    composite_ssim,
    missing_region_mse,
    missing_region_psnr,
    structural_similarity,
)


def test_perfect_reconstruction_metrics():
    ground_truth = np.full((16, 16, 3), 0.5, dtype=np.float32)
    mask = np.ones((16, 16), dtype=np.bool_)
    mask[4:12, 4:12] = False

    assert missing_region_mse(ground_truth, ground_truth, mask) == 0.0
    assert math.isinf(missing_region_psnr(ground_truth, ground_truth, mask))
    assert np.isclose(composite_ssim(ground_truth, ground_truth, mask), 1.0)


def test_missing_region_error_is_not_diluted_by_observed_pixels():
    ground_truth = np.zeros((20, 20, 3), dtype=np.float32)
    prediction = ground_truth.copy()
    mask = np.ones((20, 20), dtype=np.bool_)
    mask[8:12, 8:12] = False
    prediction[~mask] = 1.0

    assert np.isclose(missing_region_mse(prediction, ground_truth, mask), 1.0)
    assert np.isclose(missing_region_psnr(prediction, ground_truth, mask), 0.0)
    assert composite_ssim(prediction, ground_truth, mask) < 1.0


def test_structural_similarity_detects_change():
    first = np.zeros((16, 16, 3), dtype=np.float32)
    second = first.copy()
    second[4:12, 4:12] = 1.0

    assert structural_similarity(first, first) > structural_similarity(first, second)
