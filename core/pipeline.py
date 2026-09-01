"""Day 1 deterministic interpolation experiment pipeline."""

from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime
from pathlib import Path

from ..schemas import ExperimentConfig, ExperimentResult
from .data import apply_observation_mask, load_rgb_image, save_image, save_mask
from .interpolation import nearest_neighbor_fill
from .masks import generate_observation_mask
from .metrics import composite_ssim, missing_region_mse, missing_region_psnr


def _make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return "day1-%s-%s" % (timestamp, uuid.uuid4().hex[:6])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, allow_nan=False)
        output_file.write("\n")


def run_day1_baseline(config: ExperimentConfig) -> ExperimentResult:
    """Run the reproducible Day 1 nearest-neighbor experiment.

    The interpolation function receives only ``corrupted`` and ``mask``.  The
    complete image is passed to metric functions only after reconstruction.
    """

    config.validate()
    started_at = time.perf_counter()
    run_id = _make_run_id()
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    ground_truth = load_rgb_image(config.image_path, max_size=config.image_size)
    height, width, _ = ground_truth.shape
    observed_mask = generate_observation_mask(
        height=height,
        width=width,
        missing_rate=config.missing_rate,
        mask_type=config.mask_type,
        seed=config.seed,
    )
    corrupted = apply_observation_mask(
        ground_truth,
        observed_mask,
        missing_fill_value=config.missing_fill_value,
    )

    # The baseline has no access to ground_truth after the mask is applied.
    reconstruction = nearest_neighbor_fill(corrupted, observed_mask)

    mse = missing_region_mse(reconstruction, ground_truth, observed_mask)
    psnr = missing_region_psnr(reconstruction, ground_truth, observed_mask)
    ssim = composite_ssim(reconstruction, ground_truth, observed_mask)

    artifact_paths = {
        "config": str(run_dir / "config.json"),
        "original": str(run_dir / "original.png"),
        "corrupted": str(run_dir / "corrupted.png"),
        "mask": str(run_dir / "mask.png"),
        "interpolated": str(run_dir / "interpolated.png"),
        "metrics": str(run_dir / "metrics.json"),
    }
    save_image(artifact_paths["original"], ground_truth)
    save_image(artifact_paths["corrupted"], corrupted)
    save_mask(artifact_paths["mask"], observed_mask)
    save_image(artifact_paths["interpolated"], reconstruction)

    result = ExperimentResult(
        run_id=run_id,
        algorithm_name="nearest_neighbor_manhattan",
        status="success",
        seed=config.seed,
        requested_missing_rate=config.missing_rate,
        actual_missing_rate=float((~observed_mask).mean()),
        missing_mse=mse,
        missing_psnr=psnr if math.isfinite(psnr) else None,
        perfect_reconstruction=not math.isfinite(psnr),
        composite_ssim=ssim,
        runtime_seconds=float(time.perf_counter() - started_at),
        image_shape=list(ground_truth.shape),
        artifacts=artifact_paths,
        notes={
            "mask_convention": "1/white=observed, 0/black=missing",
            "ground_truth_usage": "evaluation_only",
            "ssim_definition": (
                "Known pixels are replaced by ground truth before standard RGB SSIM; "
                "this is composite SSIM, not a standardized masked SSIM."
            ),
        },
    )

    config_payload = config.to_dict()
    config_payload["resolved_image_shape"] = list(ground_truth.shape)
    config_payload["run_id"] = run_id
    _write_json(Path(artifact_paths["config"]), config_payload)
    _write_json(Path(artifact_paths["metrics"]), result.to_dict())
    return result

