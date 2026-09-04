"""Isolated-process functional checks for one AST-approved candidate."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

    import torch

    from inpainting_research_agent.core.models.base import BaseTensorInpaintingModel
    from inpainting_research_agent.core.models.cp import CPDecomposition
    from inpainting_research_agent.core.models.matrix_factorization import (
        MatrixFactorization,
    )
    from inpainting_research_agent.core.models.tucker import TuckerDecomposition

    if len(sys.argv) != 2:
        raise ValueError("expected candidate model path")
    model_path = Path(sys.argv[1])
    source = model_path.read_text(encoding="utf-8")
    namespace = {
        "__name__": "validated_candidate",
        "torch": torch,
        "BaseTensorInpaintingModel": BaseTensorInpaintingModel,
        "MatrixFactorization": MatrixFactorization,
        "CPDecomposition": CPDecomposition,
        "TuckerDecomposition": TuckerDecomposition,
    }
    exec(compile(source, str(model_path), "exec"), namespace, namespace)
    candidate_class = namespace.get("CandidateTensorInpaintingModel")
    if not isinstance(candidate_class, type):
        raise TypeError("CandidateTensorInpaintingModel is not a class")
    if not issubclass(candidate_class, BaseTensorInpaintingModel):
        raise TypeError("candidate does not inherit BaseTensorInpaintingModel")

    torch.manual_seed(123)
    image_shape = (16, 16, 3)
    model = candidate_class(
        image_shape=image_shape,
        initial_channel_mean=(0.4, 0.5, 0.6),
    )
    model.train()
    prediction = model()
    if tuple(prediction.shape) != image_shape:
        raise ValueError(
            "forward shape mismatch: expected %s, got %s"
            % (image_shape, tuple(prediction.shape))
        )
    if not prediction.dtype.is_floating_point:
        raise TypeError("forward output must use a floating dtype")
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("forward output contains NaN or Inf")

    observed = torch.rand(image_shape, dtype=torch.float32)
    train_mask = torch.ones(image_shape[:2], dtype=torch.bool)
    train_mask[4:8, 5:10] = False
    loss_terms = model.loss_terms(prediction, observed, train_mask)
    if not isinstance(loss_terms, dict) or "data_loss" not in loss_terms:
        raise TypeError("loss_terms must return a dict containing data_loss")
    total_loss = sum(loss_terms.values())
    if total_loss.ndim != 0 or not bool(torch.isfinite(total_loss)):
        raise FloatingPointError("total loss must be a finite scalar")
    total_loss.backward()

    gradient_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
    ]
    if not gradient_parameters:
        raise RuntimeError("no trainable parameter received a finite gradient")
    before = [parameter.detach().clone() for parameter in gradient_parameters]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.step()
    if not any(
        not torch.equal(old_value, parameter.detach())
        for old_value, parameter in zip(before, gradient_parameters)
    ):
        raise RuntimeError("optimizer step did not update any parameter")

    search_space = candidate_class.search_space(image_shape)
    if not isinstance(search_space, dict) or not search_space:
        raise TypeError("search_space must return a non-empty dict")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "passed": True,
                "forward_shape": list(prediction.shape),
                "dtype": str(prediction.dtype),
                "loss_terms": sorted(loss_terms),
                "finite_gradient_parameter_count": len(gradient_parameters),
                "parameter_count": parameter_count,
                "search_space": search_space,
                "optimizer_step_changed_parameter": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except Exception as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
        )
        raise SystemExit(1)
