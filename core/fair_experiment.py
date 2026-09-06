"""Paired, ground-truth-isolated tuning for baseline and candidate models."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..schemas import TrainingConfig
from .data import save_image
from .metrics import composite_ssim, missing_region_mse, missing_region_psnr
from .trainer import (
    ModelBuilder,
    fit_tensor_model_on_all_observations,
    train_tensor_model,
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_search_space(search_space: Dict[str, Any]) -> None:
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a non-empty dict")
    for name, values in search_space.items():
        if not isinstance(name, str) or not isinstance(values, list) or not values:
            raise ValueError("each search-space entry must be a non-empty list")


def _grid(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    _validate_search_space(search_space)
    names = sorted(search_space)
    return [
        dict(zip(names, values))
        for values in itertools.product(*(search_space[name] for name in names))
    ]


def paired_trial_configurations(
    base_search_space: Dict[str, List[Any]],
    candidate_search_space: Dict[str, List[Any]],
    trial_count: int,
    seed: int,
    anchor_base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Use identical base parameters and pair only candidate-specific options."""

    if trial_count < 1 or trial_count > 5:
        raise ValueError("trial_count must be in [1, 5]")
    base_grid = _grid(base_search_space)
    if len(base_grid) < trial_count:
        raise ValueError("base search space contains fewer configs than trial_count")
    rng = np.random.default_rng(seed)
    selected_base = []
    if anchor_base_config is not None:
        anchor = {
            key: anchor_base_config[key]
            for key in base_search_space
            if key in anchor_base_config
        }
        if len(anchor) == len(base_search_space) and all(
            anchor[key] in base_search_space[key] for key in base_search_space
        ):
            selected_base.append(anchor)
    remaining = [config for config in base_grid if config not in selected_base]
    order = rng.permutation(len(remaining)).tolist()
    selected_base.extend(remaining[index] for index in order)
    selected_base = selected_base[:trial_count]

    extra_space = {
        key: values
        for key, values in candidate_search_space.items()
        if key not in base_search_space
    }
    if not extra_space:
        candidate_configs = [dict(config) for config in selected_base]
    else:
        extra_grid = _grid(extra_space)
        extra_order = rng.permutation(len(extra_grid)).tolist()
        ordered_extra = [extra_grid[index] for index in extra_order]
        candidate_configs = []
        for index, base_config in enumerate(selected_base):
            combined = dict(base_config)
            combined.update(ordered_extra[index % len(ordered_extra)])
            candidate_configs.append(combined)
    return {
        "baseline": selected_base,
        "candidate": candidate_configs,
        "shared_parameter_names": sorted(base_search_space),
        "candidate_only_parameter_names": sorted(extra_space),
    }


def tune_model_on_observed_pixels(
    model_name: str,
    model_builder: Optional[ModelBuilder],
    configurations: List[Dict[str, Any]],
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
    training_config: TrainingConfig,
    seed: int,
) -> Dict[str, Any]:
    """Tune without any ground-truth argument or hidden-region metric."""

    if not configurations:
        raise ValueError("at least one configuration is required")
    trials = []
    for trial_index, hyperparameters in enumerate(configurations):
        output = train_tensor_model(
            model_name=model_name,
            model_hyperparameters=hyperparameters,
            observed_image=observed_image,
            observed_mask=observed_mask,
            config=training_config,
            seed=seed,
            model_builder=model_builder,
        )
        trials.append(
            {
                "trial_index": trial_index,
                "hyperparameters": hyperparameters,
                "best_step": output.best_step,
                "best_validation_mse": output.best_validation_mse,
                "runtime_seconds": output.runtime_seconds,
                "parameter_count": output.parameter_count,
                "stopped_early": output.stopped_early,
                "history": output.history,
            }
        )
    best = min(trials, key=lambda item: item["best_validation_mse"])
    return {
        "model_name": model_name,
        "selection_metric": "held_out_observed_mse",
        "ground_truth_used": False,
        "trial_count": len(trials),
        "trials": trials,
        "best": best,
    }


def final_fit_and_evaluate(
    model_name: str,
    model_builder: Optional[ModelBuilder],
    selected_trial: Dict[str, Any],
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
    ground_truth: np.ndarray,
    training_config: TrainingConfig,
    seed: int,
    output_dir: str,
) -> Dict[str, Any]:
    """Refit on all observations, then and only then evaluate hidden pixels."""

    output = fit_tensor_model_on_all_observations(
        model_name=model_name,
        model_hyperparameters=selected_trial["hyperparameters"],
        observed_image=observed_image,
        observed_mask=observed_mask,
        config=training_config,
        selected_steps=int(selected_trial["best_step"]),
        seed=seed,
        model_builder=model_builder,
    )
    completed = output.reconstruction.copy()
    completed[observed_mask] = observed_image[observed_mask]
    psnr = missing_region_psnr(completed, ground_truth, observed_mask)
    metrics = {
        "missing_mse": missing_region_mse(completed, ground_truth, observed_mask),
        "missing_psnr": psnr if math.isfinite(psnr) else None,
        "perfect_reconstruction": not math.isfinite(psnr),
        "composite_ssim": composite_ssim(completed, ground_truth, observed_mask),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "model_raw.png"
    completed_path = directory / "model_completed.png"
    checkpoint_path = directory / "model.pt"
    history_path = directory / "final_fit_history.json"
    metrics_path = directory / "metrics.json"
    save_image(str(raw_path), output.reconstruction)
    save_image(str(completed_path), completed)
    torch.save(
        {
            "model_name": model_name,
            "hyperparameters": selected_trial["hyperparameters"],
            "selected_steps": selected_trial["best_step"],
            "state_dict": output.state_dict,
            "training_phase": "refit_on_all_observations",
        },
        checkpoint_path,
    )
    _write_json(history_path, {"history": output.history})
    result = {
        "model_name": model_name,
        "hyperparameters": selected_trial["hyperparameters"],
        "selected_steps": selected_trial["best_step"],
        "selection_validation_mse": selected_trial["best_validation_mse"],
        "metrics": metrics,
        "runtime_seconds": output.runtime_seconds,
        "selection_runtime_seconds": selected_trial["runtime_seconds"],
        "total_selected_runtime_seconds": (
            output.runtime_seconds + selected_trial["runtime_seconds"]
        ),
        "parameter_count": output.parameter_count,
        "observed_pixels_used": int(output.fit_mask.sum()),
        "artifacts": {
            "raw_reconstruction": str(raw_path),
            "reconstruction": str(completed_path),
            "checkpoint": str(checkpoint_path),
            "history": str(history_path),
            "metrics": str(metrics_path),
        },
    }
    _write_json(metrics_path, result)
    return result
