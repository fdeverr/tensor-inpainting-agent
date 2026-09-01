"""Day 2 tensor-model experiment pipeline."""

from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..schemas import Day2ExperimentConfig
from .data import apply_observation_mask, load_rgb_image, save_image, save_mask
from .interpolation import nearest_neighbor_fill
from .masks import generate_observation_mask
from .metrics import composite_ssim, missing_region_mse, missing_region_psnr
from .trainer import train_tensor_model


def _make_run_id(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return "day2-%s-%s-%s" % (model_name, timestamp, uuid.uuid4().hex[:6])


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, allow_nan=False)
        output_file.write("\n")


def _safe_psnr(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def _evaluate(
    reconstruction,
    ground_truth,
    observed_mask,
) -> Dict[str, Any]:
    psnr = missing_region_psnr(reconstruction, ground_truth, observed_mask)
    return {
        "missing_mse": missing_region_mse(reconstruction, ground_truth, observed_mask),
        "missing_psnr": _safe_psnr(psnr),
        "perfect_reconstruction": not math.isfinite(psnr),
        "composite_ssim": composite_ssim(reconstruction, ground_truth, observed_mask),
    }


def run_day2_experiment(config: Day2ExperimentConfig) -> Dict[str, Any]:
    """Compare one trainable tensor model with the Day 1 interpolation baseline."""

    config.validate()
    started_at = time.perf_counter()
    run_id = _make_run_id(config.model_name)
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

    interpolation = nearest_neighbor_fill(corrupted, observed_mask)
    training_output = train_tensor_model(
        model_name=config.model_name,
        model_hyperparameters=config.model_hyperparameters,
        observed_image=corrupted,
        observed_mask=observed_mask,
        config=config.training,
        seed=config.seed,
    )
    raw_reconstruction = training_output.reconstruction
    completed_reconstruction = raw_reconstruction.copy()
    completed_reconstruction[observed_mask] = corrupted[observed_mask]

    artifact_paths = {
        "config": str(run_dir / "config.json"),
        "original": str(run_dir / "original.png"),
        "corrupted": str(run_dir / "corrupted.png"),
        "mask": str(run_dir / "mask.png"),
        "train_mask": str(run_dir / "train_mask.png"),
        "validation_mask": str(run_dir / "validation_mask.png"),
        "interpolated": str(run_dir / "interpolated.png"),
        "model_raw": str(run_dir / "model_raw.png"),
        "model_completed": str(run_dir / "model_completed.png"),
        "training_history": str(run_dir / "training_history.json"),
        "checkpoint": str(run_dir / "best_model.pt"),
        "metrics": str(run_dir / "metrics.json"),
    }
    save_image(artifact_paths["original"], ground_truth)
    save_image(artifact_paths["corrupted"], corrupted)
    save_mask(artifact_paths["mask"], observed_mask)
    save_mask(artifact_paths["train_mask"], training_output.train_mask)
    save_mask(artifact_paths["validation_mask"], training_output.validation_mask)
    save_image(artifact_paths["interpolated"], interpolation)
    save_image(artifact_paths["model_raw"], raw_reconstruction)
    save_image(artifact_paths["model_completed"], completed_reconstruction)
    torch.save(
        {
            "model_name": config.model_name,
            "model_hyperparameters": config.model_hyperparameters,
            "image_shape": tuple(ground_truth.shape),
            "state_dict": training_output.state_dict,
            "best_step": training_output.best_step,
            "best_validation_mse": training_output.best_validation_mse,
        },
        artifact_paths["checkpoint"],
    )

    interpolation_metrics = _evaluate(interpolation, ground_truth, observed_mask)
    model_metrics = _evaluate(completed_reconstruction, ground_truth, observed_mask)
    psnr_delta = None
    if (
        interpolation_metrics["missing_psnr"] is not None
        and model_metrics["missing_psnr"] is not None
    ):
        psnr_delta = (
            model_metrics["missing_psnr"] - interpolation_metrics["missing_psnr"]
        )

    payload = {
        "run_id": run_id,
        "status": "success",
        "image_shape": list(ground_truth.shape),
        "requested_missing_rate": config.missing_rate,
        "actual_missing_rate": float((~observed_mask).mean()),
        "model_name": config.model_name,
        "model_hyperparameters": config.model_hyperparameters,
        "training": {
            "best_step": training_output.best_step,
            "best_validation_mse": training_output.best_validation_mse,
            "runtime_seconds": training_output.runtime_seconds,
            "parameter_count": training_output.parameter_count,
            "device": training_output.device,
            "stopped_early": training_output.stopped_early,
        },
        "interpolation": interpolation_metrics,
        "tensor_model": model_metrics,
        "comparison": {
            "psnr_delta_vs_interpolation": psnr_delta,
            "ssim_delta_vs_interpolation": (
                model_metrics["composite_ssim"]
                - interpolation_metrics["composite_ssim"]
            ),
        },
        "total_runtime_seconds": float(time.perf_counter() - started_at),
        "artifacts": artifact_paths,
        "notes": {
            "ground_truth_usage": "final_evaluation_only",
            "tuning_signal": "held_out_observed_pixels_only",
            "model_output": "Known pixels in model_completed.png are copied from observations.",
        },
    }

    config_payload = config.to_dict()
    config_payload["run_id"] = run_id
    config_payload["resolved_image_shape"] = list(ground_truth.shape)
    _write_json(Path(artifact_paths["config"]), config_payload)
    _write_json(
        Path(artifact_paths["training_history"]),
        {"history": training_output.history},
    )
    _write_json(Path(artifact_paths["metrics"]), payload)
    return payload
