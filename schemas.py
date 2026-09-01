"""Day 1 experiment schemas.

Dataclasses keep Day 1 runnable in the repository's current minimal Python
environment.  These schemas can be migrated to Pydantic when the Agent-facing
structured-output layer is introduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


SUPPORTED_MASK_TYPES = {"random", "block"}


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a reproducible interpolation-baseline run."""

    image_path: str
    output_dir: str = "inpainting_research_agent/outputs"
    mask_type: str = "block"
    missing_rate: float = 0.4
    seed: int = 42
    image_size: Optional[int] = 128
    missing_fill_value: float = 0.0

    def validate(self) -> None:
        image_path = Path(self.image_path)
        if not image_path.is_file():
            raise ValueError("image_path does not point to a readable file: %s" % image_path)
        if self.mask_type not in SUPPORTED_MASK_TYPES:
            raise ValueError(
                "mask_type must be one of %s, got %r"
                % (sorted(SUPPORTED_MASK_TYPES), self.mask_type)
            )
        if not 0.0 < self.missing_rate < 1.0:
            raise ValueError("missing_rate must be strictly between 0 and 1")
        if self.image_size is not None and self.image_size < 8:
            raise ValueError("image_size must be at least 8, or None to keep the source size")
        if not 0.0 <= self.missing_fill_value <= 1.0:
            raise ValueError("missing_fill_value must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Serializable result returned by the Day 1 baseline pipeline."""

    run_id: str
    algorithm_name: str
    status: str
    seed: int
    requested_missing_rate: float
    actual_missing_rate: float
    missing_mse: float
    missing_psnr: Optional[float]
    perfect_reconstruction: bool
    composite_ssim: float
    runtime_seconds: float
    image_shape: list
    artifacts: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SUPPORTED_MODEL_NAMES = {"matrix", "cp", "tucker"}


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration shared by every trainable tensor model."""

    learning_rate: float = 0.03
    max_steps: int = 500
    validation_observed_ratio: float = 0.1
    validation_interval: int = 10
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-7
    device: str = "auto"
    deterministic: bool = True

    def validate(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 0.0 < self.validation_observed_ratio < 1.0:
            raise ValueError("validation_observed_ratio must be strictly between 0 and 1")
        if self.validation_interval < 1:
            raise ValueError("validation_interval must be positive")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Day2ExperimentConfig:
    """Configuration for one tensor-model experiment."""

    image_path: str
    model_name: str
    model_hyperparameters: Dict[str, Any]
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "inpainting_research_agent/outputs"
    mask_type: str = "block"
    missing_rate: float = 0.4
    seed: int = 42
    image_size: Optional[int] = 128
    missing_fill_value: float = 0.0

    def validate(self) -> None:
        ExperimentConfig(
            image_path=self.image_path,
            output_dir=self.output_dir,
            mask_type=self.mask_type,
            missing_rate=self.missing_rate,
            seed=self.seed,
            image_size=self.image_size,
            missing_fill_value=self.missing_fill_value,
        ).validate()
        if self.model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(
                "model_name must be one of %s, got %r"
                % (sorted(SUPPORTED_MODEL_NAMES), self.model_name)
            )
        self.training.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
