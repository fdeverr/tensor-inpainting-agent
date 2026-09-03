"""Day 4 workflow with retrieval-augmented method selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .knowledge import LocalKnowledgeRetriever
from .method_selector import MethodSelector, llm_from_environment
from .workflow import Day3Workflow, Day3WorkflowConfig, _write_json


SELECTION_QUERY = (
    "Choose matrix, CP, or Tucker tensor decomposition for image inpainting "
    "using missing pattern, channel correlation, local smoothness, high frequency, "
    "spatial anisotropy, rank guidance, limitations, and failure modes."
)


@dataclass(frozen=True)
class Day4WorkflowConfig(Day3WorkflowConfig):
    """Day 4 adds selector controls while retaining the Day 3 experiment budget."""

    model_name: str = "auto"
    llm_mode: str = "auto"
    retrieval_top_k: int = 8

    def validate(self) -> None:
        if not Path(self.image_path).is_file():
            raise ValueError("image_path does not point to a file: %s" % self.image_path)
        if self.mask_type not in {"random", "block"}:
            raise ValueError("mask_type must be random or block")
        if not 0.0 < self.missing_rate < 1.0:
            raise ValueError("missing_rate must be strictly between 0 and 1")
        if self.image_size is not None and self.image_size < 8:
            raise ValueError("image_size must be at least 8 or None")
        if self.model_name != "auto":
            raise ValueError("Day 4 model_name must be auto")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be strictly between 0 and 1")
        if self.validation_interval < 1 or self.patience < 1:
            raise ValueError("validation_interval and patience must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.llm_mode not in {"auto", "off", "required"}:
            raise ValueError("llm_mode must be auto, off, or required")
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be positive")


def _positive_ints(value: Any, maximum: int = 64) -> List[int]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            integer = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= integer <= maximum and integer not in result:
            result.append(integer)
    return result[:4]


def _candidates_from_plan(
    plan: Dict[str, Any],
    profile: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Convert untrusted but schema-valid rank suggestions to a bounded grid."""

    method = plan["method"]
    suggestions = plan["suggested_hyperparameters"]
    learning_rate = float(suggestions.get("learning_rate", 0.03))
    if not 1e-5 <= learning_rate <= 1.0:
        learning_rate = 0.03
    if method in {"matrix", "cp"}:
        max_rank = (
            int(profile["image_shape"][0]) if method == "matrix" else 64
        )
        ranks = _positive_ints(
            suggestions.get("rank_candidates"),
            maximum=max_rank,
        )
        if not ranks:
            return None
        init_scale = float(
            suggestions.get("init_scale", 0.1 if method == "matrix" else 0.2)
        )
        if not 0.0 < init_scale <= 1.0:
            init_scale = 0.1 if method == "matrix" else 0.2
        return [
            {
                "hyperparameters": {"rank": rank, "init_scale": init_scale},
                "learning_rate": learning_rate,
            }
            for rank in ranks[:3]
        ]

    rank_h = _positive_ints(
        suggestions.get("rank_h_candidates"),
        maximum=int(profile["image_shape"][0]),
    )
    rank_w = _positive_ints(
        suggestions.get("rank_w_candidates"),
        maximum=int(profile["image_shape"][1]),
    )
    rank_c = _positive_ints(suggestions.get("rank_c_candidates"), maximum=3)
    if not rank_h or not rank_w or not rank_c:
        return None
    init_scale = float(suggestions.get("init_scale", 0.15))
    if not 0.0 < init_scale <= 1.0:
        init_scale = 0.15
    count = min(3, max(len(rank_h), len(rank_w), len(rank_c)))
    candidates = []
    for index in range(count):
        candidates.append(
            {
                "hyperparameters": {
                    "rank_h": rank_h[min(index, len(rank_h) - 1)],
                    "rank_w": rank_w[min(index, len(rank_w) - 1)],
                    "rank_c": rank_c[min(index, len(rank_c) - 1)],
                    "init_scale": init_scale,
                },
                "learning_rate": learning_rate,
            }
        )
    return candidates


class Day4Workflow(Day3Workflow):
    """Replace Day 3's fixed choice with retrieval and a validated selector."""

    run_prefix = "day4"
    workflow_name = "day4_retrieval_augmented_method_selection"

    def __init__(
        self,
        config: Day4WorkflowConfig,
        selector: Optional[MethodSelector] = None,
        retriever: Optional[LocalKnowledgeRetriever] = None,
    ) -> None:
        config.validate()
        self.retriever = retriever or LocalKnowledgeRetriever()
        self.selector = selector or MethodSelector(llm_from_environment(config.llm_mode))
        self.llm_used = self.selector.llm is not None
        super().__init__(config)

    def _select_method(self) -> Dict[str, Any]:
        profile = self.state["results"]["image_profile"]
        retrieval = self.retriever.retrieve(
            profile=profile,
            query=SELECTION_QUERY,
            top_k=self.config.retrieval_top_k,
        )
        selection = self.selector.select(profile=profile, retrieval=retrieval)
        plan = selection["plan"]
        plan_payload = plan.model_dump()

        retrieval_path = self.run_dir / "retrieval_result.json"
        method_plan_path = self.run_dir / "method_plan.json"
        _write_json(retrieval_path, retrieval)
        _write_json(
            method_plan_path,
            {
                **plan_payload,
                "selector_attempts": selection["attempts"],
                "fallback_reason": selection["fallback_reason"],
                "validation_errors": selection["validation_errors"],
                "raw_outputs": selection["raw_outputs"],
                "ground_truth_provided_to_selector": False,
                "final_metrics_provided_to_selector": False,
            },
        )
        self.state["artifacts"]["retrieval_result"] = str(retrieval_path)
        self.state["artifacts"]["method_plan"] = str(method_plan_path)
        self.state["results"]["retrieval_summary"] = {
            "active_rule_count": len(retrieval["active_rules"]),
            "evidence_count": len(retrieval["evidence"]),
            "evidence_sources": [item["source"] for item in retrieval["evidence"]],
        }
        self.state["results"]["selector_diagnostics"] = {
            "attempts": selection["attempts"],
            "fallback_reason": selection["fallback_reason"],
            "validation_errors": selection["validation_errors"],
            "llm_used": self.llm_used,
        }
        return plan_payload

    def _tuning_candidates(
        self,
        selected_model: str,
        method_plan: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        if self.config.candidates is not None:
            return self.config.candidates
        return _candidates_from_plan(
            method_plan,
            self.state["results"]["image_profile"],
        )


def run_day4_workflow(
    config: Day4WorkflowConfig,
    selector: Optional[MethodSelector] = None,
) -> Dict[str, Any]:
    return Day4Workflow(config=config, selector=selector).run()
