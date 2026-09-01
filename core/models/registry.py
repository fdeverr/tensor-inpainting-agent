"""Small deterministic registry for built-in tensor models."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from .base import BaseTensorInpaintingModel
from .cp import CPDecomposition
from .matrix_factorization import MatrixFactorization
from .tucker import TuckerDecomposition


MODEL_CLASSES = {
    "matrix": MatrixFactorization,
    "cp": CPDecomposition,
    "tucker": TuckerDecomposition,
}


def get_default_hyperparameters(model_name: str) -> Dict[str, Any]:
    defaults = {
        "matrix": {"rank": 16, "init_scale": 0.1},
        "cp": {"rank": 12, "init_scale": 0.2},
        "tucker": {"rank_h": 16, "rank_w": 16, "rank_c": 3, "init_scale": 0.15},
    }
    if model_name not in defaults:
        raise ValueError("unknown model_name %r" % model_name)
    return dict(defaults[model_name])


def create_model(
    model_name: str,
    image_shape: Tuple[int, int, int],
    initial_channel_mean: Sequence[float],
    hyperparameters: Dict[str, Any],
) -> BaseTensorInpaintingModel:
    if model_name not in MODEL_CLASSES:
        raise ValueError("unknown model_name %r" % model_name)
    model_class = MODEL_CLASSES[model_name]
    return model_class(
        image_shape=image_shape,
        initial_channel_mean=initial_channel_mean,
        **hyperparameters,
    )
