import json
from pathlib import Path

import numpy as np
from PIL import Image

from inpainting_research_agent.core.day2_pipeline import run_day2_experiment
from inpainting_research_agent.schemas import Day2ExperimentConfig, TrainingConfig


def _write_small_image(path: Path) -> None:
    height, width = 12, 16
    y, x = np.indices((height, width), dtype=np.float32)
    image = np.stack(
        (
            x / (width - 1),
            y / (height - 1),
            0.25 + 0.5 * x / (width - 1),
        ),
        axis=-1,
    )
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path)


def test_day2_pipeline_writes_model_and_interpolation_results(tmp_path):
    image_path = tmp_path / "image.png"
    _write_small_image(image_path)
    config = Day2ExperimentConfig(
        image_path=str(image_path),
        model_name="matrix",
        model_hyperparameters={"rank": 3, "init_scale": 0.1},
        training=TrainingConfig(
            learning_rate=0.05,
            max_steps=50,
            validation_observed_ratio=0.15,
            validation_interval=5,
            early_stopping_patience=20,
            device="cpu",
        ),
        output_dir=str(tmp_path / "outputs"),
        mask_type="block",
        missing_rate=0.3,
        seed=23,
        image_size=None,
    )

    result = run_day2_experiment(config)

    assert result["status"] == "success"
    assert result["model_name"] == "matrix"
    assert result["selection"]["best_step"] >= 1
    assert result["selection"]["parameter_count"] > 0
    assert result["final_fit"]["fitted_steps"] == result["selection"]["best_step"]
    assert result["final_fit"]["observed_pixels_used"] == round(12 * 16 * 0.7)
    assert result["interpolation"]["missing_psnr"] is not None
    assert result["selection_model_diagnostic"]["missing_psnr"] is not None
    assert result["tensor_model"]["missing_psnr"] is not None
    for artifact_path in result["artifacts"].values():
        assert Path(artifact_path).is_file()

    with Path(result["artifacts"]["metrics"]).open(encoding="utf-8") as metrics_file:
        saved = json.load(metrics_file)
    assert saved["notes"]["ground_truth_usage"] == "final_evaluation_only"
    assert saved["notes"]["tuning_signal"] == "held_out_observed_pixels_only"
    assert "100% of observed pixels" in saved["notes"]["final_fit"]
