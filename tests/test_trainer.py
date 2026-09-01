import numpy as np

from inpainting_research_agent.core.data import apply_observation_mask
from inpainting_research_agent.core.masks import generate_observation_mask
from inpainting_research_agent.core.trainer import (
    fit_tensor_model_on_all_observations,
    train_tensor_model,
)
from inpainting_research_agent.schemas import TrainingConfig


def _smooth_rgb_image(height=12, width=10):
    y, x = np.indices((height, width), dtype=np.float32)
    return np.stack(
        (
            x / (width - 1),
            y / (height - 1),
            (x + y) / (width + height - 2),
        ),
        axis=-1,
    ).astype(np.float32)


def test_trainer_uses_disjoint_observed_split_and_is_reproducible():
    ground_truth = _smooth_rgb_image()
    observed_mask = generate_observation_mask(12, 10, 0.25, "random", seed=5)
    corrupted = apply_observation_mask(ground_truth, observed_mask)
    config = TrainingConfig(
        learning_rate=0.05,
        max_steps=60,
        validation_observed_ratio=0.15,
        validation_interval=5,
        early_stopping_patience=20,
        device="cpu",
    )
    hyperparameters = {"rank": 3, "init_scale": 0.1}

    first = train_tensor_model(
        "matrix", hyperparameters, corrupted, observed_mask, config, seed=17
    )
    second = train_tensor_model(
        "matrix", hyperparameters, corrupted, observed_mask, config, seed=17
    )

    assert first.reconstruction.shape == ground_truth.shape
    assert np.isfinite(first.reconstruction).all()
    assert np.array_equal(first.train_mask | first.validation_mask, observed_mask)
    assert not np.logical_and(first.train_mask, first.validation_mask).any()
    assert not first.train_mask[~observed_mask].any()
    assert not first.validation_mask[~observed_mask].any()
    assert first.best_validation_mse <= first.history[0]["validation_mse"]
    assert first.parameter_count > 0
    assert first.device == "cpu"
    assert np.array_equal(first.reconstruction, second.reconstruction)
    assert first.best_validation_mse == second.best_validation_mse


def test_final_refit_uses_every_observed_pixel_and_no_hidden_pixel():
    ground_truth = _smooth_rgb_image()
    observed_mask = generate_observation_mask(12, 10, 0.25, "block", seed=9)
    corrupted = apply_observation_mask(ground_truth, observed_mask)
    config = TrainingConfig(
        learning_rate=0.05,
        max_steps=40,
        validation_observed_ratio=0.15,
        validation_interval=5,
        early_stopping_patience=20,
        device="cpu",
    )

    result = fit_tensor_model_on_all_observations(
        model_name="matrix",
        model_hyperparameters={"rank": 3, "init_scale": 0.1},
        observed_image=corrupted,
        observed_mask=observed_mask,
        config=config,
        selected_steps=25,
        seed=17,
    )

    assert result.fitted_steps == 25
    assert np.array_equal(result.fit_mask, observed_mask)
    assert not result.fit_mask[~observed_mask].any()
    assert result.final_train_mse >= 0.0
    assert np.isfinite(result.reconstruction).all()
