"""Fixed Trainer shared by every tensor-decomposition model."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..schemas import TrainingConfig
from .masks import split_observed_mask
from .models import create_model


ModelBuilder = Callable[
    [Tuple[int, int, int], Sequence[float], Dict[str, Any]],
    torch.nn.Module,
]


def _build_model(
    model_name: str,
    image_shape: Tuple[int, int, int],
    initial_channel_mean: Sequence[float],
    model_hyperparameters: Dict[str, Any],
    model_builder: Optional[ModelBuilder],
) -> torch.nn.Module:
    if model_builder is not None:
        return model_builder(
            image_shape,
            initial_channel_mean,
            model_hyperparameters,
        )
    return create_model(
        model_name=model_name,
        image_shape=image_shape,
        initial_channel_mean=initial_channel_mean,
        hyperparameters=model_hyperparameters,
    )


@dataclass
class TrainingOutput:
    """Internal training result; reconstruction arrays are saved by a pipeline."""

    reconstruction: np.ndarray
    history: List[Dict[str, Any]]
    best_step: int
    best_validation_mse: float
    runtime_seconds: float
    parameter_count: int
    device: str
    stopped_early: bool
    train_mask: np.ndarray
    validation_mask: np.ndarray
    state_dict: Dict[str, torch.Tensor]


@dataclass
class FinalFitOutput:
    """Result of refitting a selected configuration on every observed pixel."""

    reconstruction: np.ndarray
    history: List[Dict[str, Any]]
    fitted_steps: int
    final_train_mse: float
    runtime_seconds: float
    parameter_count: int
    device: str
    fit_mask: np.ndarray
    state_dict: Dict[str, torch.Tensor]


def resolve_device(requested_device: str) -> torch.device:
    """Resolve ``auto`` to CUDA on Linux servers and CPU otherwise."""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested_device)


def set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def _validate_training_arrays(
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
) -> None:
    if observed_image.ndim != 3 or observed_image.shape[2] != 3:
        raise ValueError("observed_image must have shape [H, W, 3]")
    if not np.issubdtype(observed_image.dtype, np.floating):
        raise ValueError("observed_image must be floating point")
    if observed_mask.shape != observed_image.shape[:2] or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be bool and match image height and width")
    if not observed_mask.any():
        raise ValueError("at least one observed pixel is required")
    if not np.isfinite(observed_image).all():
        raise ValueError("observed_image contains NaN or Inf")


def _validation_mse(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    validation_mask: torch.Tensor,
) -> torch.Tensor:
    expanded_mask = validation_mask.unsqueeze(-1).expand_as(prediction)
    return torch.square(prediction - observed)[expanded_mask].mean()


def train_tensor_model(
    model_name: str,
    model_hyperparameters: Dict[str, Any],
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
    config: TrainingConfig,
    seed: int,
    model_builder: Optional[ModelBuilder] = None,
) -> TrainingOutput:
    """Fit one tensor model without access to artificially hidden ground truth."""

    config.validate()
    _validate_training_arrays(observed_image, observed_mask)
    set_reproducibility(seed, config.deterministic)
    device = resolve_device(config.device)
    train_mask_np, validation_mask_np = split_observed_mask(
        observed_mask,
        validation_ratio=config.validation_observed_ratio,
        seed=seed + 10_003,
    )

    # Initialization statistics use training pixels only, not validation pixels.
    initial_channel_mean = observed_image[train_mask_np].mean(axis=0)
    model = _build_model(
        model_name=model_name,
        image_shape=tuple(observed_image.shape),
        initial_channel_mean=initial_channel_mean,
        model_hyperparameters=model_hyperparameters,
        model_builder=model_builder,
    ).to(device)

    observed = torch.as_tensor(observed_image, dtype=torch.float32, device=device)
    train_mask = torch.as_tensor(train_mask_np, dtype=torch.bool, device=device)
    validation_mask = torch.as_tensor(
        validation_mask_np,
        dtype=torch.bool,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_validation_mse = float("inf")
    best_step = 0
    best_state = None
    checks_without_improvement = 0
    stopped_early = False
    history = []
    started_at = time.perf_counter()

    for step in range(1, config.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model()
        loss_terms = model.loss_terms(prediction, observed, train_mask)
        total_loss = sum(loss_terms.values())
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("training loss became NaN or Inf at step %d" % step)
        total_loss.backward()

        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("model gradient became NaN or Inf at step %d" % step)
        optimizer.step()

        should_validate = (
            step == 1
            or step % config.validation_interval == 0
            or step == config.max_steps
        )
        if not should_validate:
            continue

        model.eval()
        with torch.no_grad():
            current_prediction = model()
            validation_mse = float(
                _validation_mse(current_prediction, observed, validation_mask).item()
            )
        history.append(
            {
                "step": step,
                "total_train_loss": float(total_loss.detach().item()),
                "data_train_loss": float(loss_terms["data_loss"].detach().item()),
                "validation_mse": validation_mse,
            }
        )

        if validation_mse < best_validation_mse - config.early_stopping_min_delta:
            best_validation_mse = validation_mse
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1

        if checks_without_improvement >= config.early_stopping_patience:
            stopped_early = True
            break

    if best_state is None or not math.isfinite(best_validation_mse):
        raise RuntimeError("training did not produce a finite validation checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    checkpoint_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    with torch.no_grad():
        reconstruction = model().clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)

    return TrainingOutput(
        reconstruction=reconstruction,
        history=history,
        best_step=best_step,
        best_validation_mse=best_validation_mse,
        runtime_seconds=float(time.perf_counter() - started_at),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        device=str(device),
        stopped_early=stopped_early,
        train_mask=train_mask_np,
        validation_mask=validation_mask_np,
        state_dict=checkpoint_state,
    )


def fit_tensor_model_on_all_observations(
    model_name: str,
    model_hyperparameters: Dict[str, Any],
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
    config: TrainingConfig,
    selected_steps: int,
    seed: int,
    model_builder: Optional[ModelBuilder] = None,
) -> FinalFitOutput:
    """Refit a selected model using 100% of the genuinely observed pixels.

    This function performs no validation and receives no missing-region ground
    truth. ``selected_steps`` must be chosen before this call, normally by
    ``train_tensor_model`` on a temporary train/validation split.
    """

    config.validate()
    _validate_training_arrays(observed_image, observed_mask)
    if selected_steps < 1:
        raise ValueError("selected_steps must be positive")

    set_reproducibility(seed, config.deterministic)
    device = resolve_device(config.device)
    initial_channel_mean = observed_image[observed_mask].mean(axis=0)
    model = _build_model(
        model_name=model_name,
        image_shape=tuple(observed_image.shape),
        initial_channel_mean=initial_channel_mean,
        model_hyperparameters=model_hyperparameters,
        model_builder=model_builder,
    ).to(device)
    observed = torch.as_tensor(observed_image, dtype=torch.float32, device=device)
    fit_mask = torch.as_tensor(observed_mask, dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    started_at = time.perf_counter()

    for step in range(1, selected_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model()
        loss_terms = model.loss_terms(prediction, observed, fit_mask)
        total_loss = sum(loss_terms.values())
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("final-fit loss became NaN or Inf at step %d" % step)
        total_loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(
                    "final-fit gradient became NaN or Inf at step %d" % step
                )
        optimizer.step()

        should_record = (
            step == 1
            or step % config.validation_interval == 0
            or step == selected_steps
        )
        if not should_record:
            continue
        model.eval()
        with torch.no_grad():
            current_prediction = model()
            current_terms = model.loss_terms(current_prediction, observed, fit_mask)
        history.append(
            {
                "step": step,
                "total_train_loss": float(sum(current_terms.values()).item()),
                "data_train_loss": float(current_terms["data_loss"].item()),
            }
        )

    model.eval()
    with torch.no_grad():
        final_prediction = model()
        final_terms = model.loss_terms(final_prediction, observed, fit_mask)
        reconstruction = (
            final_prediction.clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        )
    checkpoint_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }

    return FinalFitOutput(
        reconstruction=reconstruction,
        history=history,
        fitted_steps=selected_steps,
        final_train_mse=float(final_terms["data_loss"].item()),
        runtime_seconds=float(time.perf_counter() - started_at),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        device=str(device),
        fit_mask=observed_mask.copy(),
        state_dict=checkpoint_state,
    )
