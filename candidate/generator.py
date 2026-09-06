"""Prompt construction and constrained fallback candidate generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ..method_selector import _json_from_text
from .schemas import CandidateProposal


PROMPT_VERSION = "day5-candidate-v1"
ALLOWED_CHANGES = [
    "factor initialization",
    "asymmetric rank",
    "factor regularization",
    "total variation regularization",
    "factor smoothness regularization",
    "rank or regularization scheduling",
    "multi-scale fitting while retaining a tensor-decomposition parameterization",
]
FORBIDDEN_CHANGES = [
    "pretrained diffusion models",
    "CNN or Transformer replacement of the tensor-decomposition core",
    "access to complete ground truth or hidden-region metrics during selection/training",
    "network, subprocess, shell, or external-program calls",
    "changes to evaluation code",
    "larger training or tuning budgets than the baseline",
]


BASE_CLASS_NAMES = {
    "matrix": "MatrixFactorization",
    "cp": "CPDecomposition",
    "tucker": "TuckerDecomposition",
}


@dataclass
class CandidateGenerationResult:
    proposal: CandidateProposal
    attempts: int
    raw_outputs: List[str]
    validation_errors: List[str]
    fallback_reason: Optional[str]
    prompt_version: str


def _tv_candidate_code(base_method: str, tv_weights: Optional[List[float]] = None) -> str:
    base_class = BASE_CLASS_NAMES[base_method]
    weights = tv_weights or [0.0, 0.0001, 0.0005, 0.001]
    return f'''import torch


class CandidateTensorInpaintingModel({base_class}):
    """Add an image-space TV prior while retaining the learned tensor factors."""

    def __init__(self, image_shape, initial_channel_mean, tv_weight=0.0001, **kwargs):
        super().__init__(
            image_shape=image_shape,
            initial_channel_mean=initial_channel_mean,
            **kwargs,
        )
        if tv_weight < 0.0:
            raise ValueError("tv_weight must be non-negative")
        self.tv_weight = float(tv_weight)

    def loss_terms(self, prediction, observed, train_mask):
        terms = super().loss_terms(prediction, observed, train_mask)
        vertical_tv = torch.abs(prediction[1:, :, :] - prediction[:-1, :, :]).mean()
        horizontal_tv = torch.abs(prediction[:, 1:, :] - prediction[:, :-1, :]).mean()
        terms["tv_regularization"] = self.tv_weight * (vertical_tv + horizontal_tv)
        return terms

    @classmethod
    def search_space(cls, image_shape):
        space = dict({base_class}.search_space(image_shape))
        space["tv_weight"] = {weights!r}
        return space
'''


def deterministic_candidate(
    base_method: str,
    previous_feedback: Optional[List[Dict[str, Any]]] = None,
) -> CandidateProposal:
    """Safe learning-project fallback used when no code LLM is configured."""

    previous_feedback = previous_feedback or []
    tv_weights = (
        [0.0, 0.00001, 0.00005, 0.0001]
        if previous_feedback
        else [0.0, 0.0001, 0.0005, 0.001]
    )
    return CandidateProposal(
        base_method=base_method,
        hypothesis=(
            "Adding a small differentiable total-variation penalty to the selected "
            "tensor decomposition may reduce the banding and abrupt spatial changes "
            "observed inside a contiguous missing region."
        ),
        proposed_changes=[
            "Keep every factor and core parameter from the selected baseline.",
            "Add horizontal and vertical image-space total variation to loss_terms().",
            "Expose tv_weight as a bounded tunable hyperparameter.",
        ],
        expected_effect=(
            "The candidate should favour spatially coherent reconstructions while "
            "retaining the global low-rank structure of the base method."
        ),
        risks=[
            "An excessive TV weight can oversmooth edges and textures.",
            "Observed-pixel validation may still be mismatched with a block hole.",
            "The extra forward regularity does not add semantic information.",
        ],
        search_space={"tv_weight": tv_weights},
        model_code=_tv_candidate_code(base_method, tv_weights=tv_weights),
        generation_mode="deterministic_template",
    )


def load_improver_context(base_run_dir: str) -> Dict[str, Any]:
    """Load only the evidence needed by the improver, never GT image contents."""

    run_dir = Path(base_run_dir)
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise ValueError("base run state does not exist: %s" % state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("stage") != "COMPLETED":
        raise ValueError("base run must be COMPLETED")
    base_method = state.get("selected_model")
    if base_method not in BASE_CLASS_NAMES:
        raise ValueError("base run has unsupported selected_model")

    source_paths = {
        "matrix": Path(__file__).resolve().parents[1]
        / "core/models/matrix_factorization.py",
        "cp": Path(__file__).resolve().parents[1] / "core/models/cp.py",
        "tucker": Path(__file__).resolve().parents[1] / "core/models/tucker.py",
    }
    history_path = Path(state["artifacts"]["tensor_history"])
    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = history_payload.get("history", [])
    history_summary = {
        "record_count": len(history),
        "first": history[0] if history else None,
        "last": history[-1] if history else None,
        "minimum_recorded_train_loss": (
            min(item["total_train_loss"] for item in history) if history else None
        ),
    }
    return {
        "base_run_id": state["run_id"],
        "base_method": base_method,
        "image_profile": state["results"]["image_profile"],
        "method_plan": state["results"].get("method_plan"),
        "base_best_config": state["results"]["training"],
        "training_curve_summary": history_summary,
        "base_metrics": state["results"]["tensor_metrics"],
        "interpolation_metrics": state["results"]["interpolation_metrics"],
        "comparison": state["results"]["comparison"],
        "base_model_source": source_paths[base_method].read_text(encoding="utf-8"),
        "base_class_name": BASE_CLASS_NAMES[base_method],
        "previous_failure_feedback": [],
    }


class CandidateGenerator:
    """Generate one schema-valid proposal with one optional repair attempt."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        self.llm = llm

    @staticmethod
    def _messages(context: Dict[str, Any]) -> List[Dict[str, str]]:
        prompt_payload = {
            "task": (
                "Propose one testable improvement to the selected tensor decomposition "
                "and return the complete CandidateProposal JSON."
            ),
            "context": context,
            "allowed_changes": ALLOWED_CHANGES,
            "forbidden_changes": FORBIDDEN_CHANGES,
            "fixed_contract": {
                "class_name": "CandidateTensorInpaintingModel",
                "base_class_available": context["base_class_name"],
                "forward_output": "float tensor [H, W, 3] with finite values",
                "required_method": "search_space(image_shape)",
                "imports": ["torch", "torch.nn", "torch.nn.functional", "math"],
                "ground_truth_available": False,
                "evaluation_code_editable": False,
            },
            "output_schema": CandidateProposal.model_json_schema(),
            "prompt_version": PROMPT_VERSION,
        }
        return [
            {
                "role": "system",
                "content": (
                    "You improve tensor-decomposition inpainting models under a strict "
                    "research contract. Return one JSON object only. The model must remain "
                    "a learned tensor decomposition and must not perform I/O, networking, "
                    "subprocess execution, dynamic code execution, or evaluation changes."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            },
        ]

    @staticmethod
    def _content(response: Any) -> str:
        if isinstance(response, str):
            return response
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            raise ValueError("LLM response does not contain string content")
        return content

    def generate(self, context: Dict[str, Any]) -> CandidateGenerationResult:
        messages = self._messages(context)
        raw_outputs = []
        errors = []
        if self.llm is not None:
            for attempt in range(2):
                current_messages = list(messages)
                if attempt == 1:
                    current_messages.extend(
                        [
                            {"role": "assistant", "content": raw_outputs[-1]},
                            {
                                "role": "user",
                                "content": (
                                    "The proposal was invalid: %s. Repair it once and "
                                    "return only a complete JSON object."
                                    % errors[-1]
                                ),
                            },
                        ]
                    )
                try:
                    raw = self._content(
                        self.llm.invoke(current_messages, temperature=0.0)
                    )
                    raw_outputs.append(raw)
                    proposal = CandidateProposal.model_validate(_json_from_text(raw))
                    if proposal.base_method != context["base_method"]:
                        raise ValueError("proposal base_method differs from selected model")
                    proposal.generation_mode = (
                        "llm" if attempt == 0 else "llm_repaired"
                    )
                    return CandidateGenerationResult(
                        proposal=proposal,
                        attempts=attempt + 1,
                        raw_outputs=raw_outputs,
                        validation_errors=errors,
                        fallback_reason=None,
                        prompt_version=PROMPT_VERSION,
                    )
                except Exception as error:
                    errors.append(str(error))

        fallback_reason = (
            "LLM is not configured"
            if self.llm is None
            else "LLM proposal remained invalid after one repair attempt"
        )
        return CandidateGenerationResult(
            proposal=deterministic_candidate(
                context["base_method"],
                previous_feedback=context.get("previous_failure_feedback"),
            ),
            attempts=len(raw_outputs),
            raw_outputs=raw_outputs,
            validation_errors=errors,
            fallback_reason=fallback_reason,
            prompt_version=PROMPT_VERSION,
        )
