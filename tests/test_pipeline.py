import json
from pathlib import Path

import numpy as np
from PIL import Image

from inpainting_research_agent.core.pipeline import run_day1_baseline
from inpainting_research_agent.schemas import ExperimentConfig


def _write_gradient_image(path: Path) -> None:
    height, width = 18, 24
    y, x = np.indices((height, width), dtype=np.float32)
    image = np.stack(
        (
            x / (width - 1),
            y / (height - 1),
            (x + y) / (width + height - 2),
        ),
        axis=-1,
    )
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path)


def test_day1_pipeline_writes_reproducible_artifacts(tmp_path):
    image_path = tmp_path / "gradient.png"
    output_dir = tmp_path / "outputs"
    _write_gradient_image(image_path)
    config = ExperimentConfig(
        image_path=str(image_path),
        output_dir=str(output_dir),
        mask_type="block",
        missing_rate=0.4,
        seed=42,
        image_size=None,
    )

    result = run_day1_baseline(config)

    assert result.status == "success"
    assert abs(result.actual_missing_rate - 0.4) < 1.0 / (18 * 24)
    assert result.missing_mse > 0.0
    assert result.missing_psnr is not None
    assert 0.0 <= result.composite_ssim <= 1.0
    for artifact_path in result.artifacts.values():
        assert Path(artifact_path).is_file()

    with Path(result.artifacts["metrics"]).open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    assert metrics["notes"]["ground_truth_usage"] == "evaluation_only"


def test_day1_pipeline_repeats_mask_reconstruction_and_metrics(tmp_path):
    image_path = tmp_path / "gradient.png"
    _write_gradient_image(image_path)
    config = ExperimentConfig(
        image_path=str(image_path),
        output_dir=str(tmp_path / "outputs"),
        mask_type="random",
        missing_rate=0.35,
        seed=19,
        image_size=None,
    )

    first = run_day1_baseline(config)
    second = run_day1_baseline(config)

    assert first.run_id != second.run_id
    assert first.actual_missing_rate == second.actual_missing_rate
    assert first.missing_mse == second.missing_mse
    assert first.missing_psnr == second.missing_psnr
    assert first.composite_ssim == second.composite_ssim
    assert Path(first.artifacts["mask"]).read_bytes() == Path(second.artifacts["mask"]).read_bytes()
    assert Path(first.artifacts["interpolated"]).read_bytes() == Path(
        second.artifacts["interpolated"]
    ).read_bytes()
