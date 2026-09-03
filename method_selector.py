"""Schema-constrained tensor-decomposition method selection."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .agent_tools.framework import HelloAgentsLLM


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=3)
    claim: str = Field(min_length=8)


class MethodPlan(BaseModel):
    """Only validated plans may influence the training workflow."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["matrix", "cp", "tucker"]
    reason: str = Field(min_length=20, max_length=1200)
    evidence: List[EvidenceReference] = Field(min_length=1, max_length=6)
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_hyperparameters: Dict[str, Any]
    risks: List[str] = Field(min_length=1, max_length=6)
    selection_mode: Literal["llm", "llm_repaired", "deterministic_fallback"] = "llm"

    @field_validator("suggested_hyperparameters")
    @classmethod
    def require_nonempty_hyperparameters(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not value:
            raise ValueError("suggested_hyperparameters cannot be empty")
        return value


def _json_from_text(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("MethodPlan output must be a JSON object")
    return payload


def _validate_sources(plan: MethodPlan, retrieval: Dict[str, Any]) -> None:
    allowed = {
        item["source"] for item in retrieval["evidence"]
    } | {rule["source"] for rule in retrieval["active_rules"]}
    unknown = [item.source for item in plan.evidence if item.source not in allowed]
    if unknown:
        raise ValueError("MethodPlan cited unknown evidence sources: %s" % unknown)


def _default_hyperparameters(method: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    if method == "matrix":
        return {"rank_candidates": [8, 16], "init_scale": 0.1}
    if method == "cp":
        return {"rank_candidates": [8, 12], "init_scale": 0.2}
    aspect_ratio = float(profile["image_aspect_ratio"])
    if aspect_ratio >= 1.6:
        return {
            "rank_h_candidates": [8, 12],
            "rank_w_candidates": [12, 16],
            "rank_c_candidates": [2, 3],
            "init_scale": 0.15,
        }
    return {
        "rank_h_candidates": [8, 16],
        "rank_w_candidates": [8, 16],
        "rank_c_candidates": [2, 3],
        "init_scale": 0.15,
    }


def deterministic_method_plan(
    profile: Dict[str, Any],
    retrieval: Dict[str, Any],
) -> MethodPlan:
    """Create a reproducible fallback from activated, weighted rules."""

    scores = {"matrix": 0.0, "cp": 0.0, "tucker": 0.0}
    for rule in retrieval["active_rules"]:
        scores[str(rule["prefer"])] += float(rule.get("weight", 1.0))
    method = max(("matrix", "cp", "tucker"), key=lambda name: (scores[name], name))
    supporting_rules = [
        rule for rule in retrieval["active_rules"] if rule["prefer"] == method
    ]
    if supporting_rules:
        evidence = [
            EvidenceReference(source=rule["source"], claim=str(rule["reason"]))
            for rule in supporting_rules[:3]
        ]
        reason = " ".join(str(rule["reason"]) for rule in supporting_rules[:3])
    else:
        first_evidence = next(
            item for item in retrieval["evidence"] if item["method"] == method
        )
        evidence = [
            EvidenceReference(
                source=first_evidence["source"],
                claim="This is the highest-scoring available method evidence.",
            )
        ]
        reason = "No profile rule dominated, so the highest-scoring local evidence was used."
    total_score = sum(scores.values())
    confidence = scores[method] / total_score if total_score > 0 else 1.0 / 3.0
    return MethodPlan(
        method=method,
        reason=(
            "%s Profile evidence: mask=%s, channel_correlation=%.3f, "
            "largest_hole_ratio=%.3f."
            % (
                reason,
                profile["mask_type"],
                profile["visible_mean_absolute_channel_correlation"],
                profile["largest_missing_component_image_ratio"],
            )
        ),
        evidence=evidence,
        confidence=float(min(1.0, max(0.0, confidence))),
        suggested_hyperparameters=_default_hyperparameters(method, profile),
        risks=[
            "Held-out observed pixels may not represent the artificial missing pattern.",
            "A global low-rank model may blur or stripe local structure.",
        ],
        selection_mode="deterministic_fallback",
    )


class MethodSelector:
    """Use an optional LLM, with one repair attempt and deterministic fallback."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        self.llm = llm

    @staticmethod
    def _prompt(profile: Dict[str, Any], retrieval: Dict[str, Any]) -> List[Dict[str, str]]:
        allowed_sources = [
            item["source"] for item in retrieval["evidence"]
        ] + [rule["source"] for rule in retrieval["active_rules"]]
        schema = MethodPlan.model_json_schema()
        user_payload = {
            "fixed_prompt": (
                "请根据当前图像的缺失模式、统计特征、插值结果和张量分解经验文档，"
                "选择一个合适的基础张量分解方法完成图像补全。"
            ),
            "image_profile": profile,
            "interpolation": {
                "available": True,
                "metrics_available_during_selection": False,
                "visual_description_available": False,
            },
            "active_rules": retrieval["active_rules"],
            "evidence_chunks": retrieval["evidence"],
            "allowed_evidence_sources": allowed_sources,
            "output_schema": schema,
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are a tensor-decomposition method selector. Return one JSON "
                    "object only. Choose exactly one of matrix, cp, or tucker. Every "
                    "reason must cite supplied profile values or allowed evidence sources. "
                    "Never request or infer missing-region ground truth or final metrics."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
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

    def select(
        self,
        profile: Dict[str, Any],
        retrieval: Dict[str, Any],
    ) -> Dict[str, Any]:
        messages = self._prompt(profile, retrieval)
        raw_outputs = []
        validation_errors = []
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
                                    "The previous JSON was invalid: %s. Return a corrected "
                                    "JSON object only; do not add new evidence sources."
                                    % validation_errors[-1]
                                ),
                            },
                        ]
                    )
                try:
                    raw = self._content(
                        self.llm.invoke(current_messages, temperature=0.0)
                    )
                    raw_outputs.append(raw)
                    plan = MethodPlan.model_validate(_json_from_text(raw))
                    _validate_sources(plan, retrieval)
                    plan.selection_mode = "llm" if attempt == 0 else "llm_repaired"
                    return {
                        "plan": plan,
                        "attempts": attempt + 1,
                        "raw_outputs": raw_outputs,
                        "validation_errors": validation_errors,
                        "fallback_reason": None,
                        "messages": messages,
                    }
                except Exception as error:
                    validation_errors.append(str(error))

        fallback_reason = (
            "LLM is not configured"
            if self.llm is None
            else "LLM output remained invalid after one repair attempt"
        )
        return {
            "plan": deterministic_method_plan(profile, retrieval),
            "attempts": len(raw_outputs),
            "raw_outputs": raw_outputs,
            "validation_errors": validation_errors,
            "fallback_reason": fallback_reason,
            "messages": messages,
        }


def llm_from_environment(mode: str = "auto") -> Optional[HelloAgentsLLM]:
    """Build the Hello Agents LLM only when configuration is complete."""

    if mode not in {"auto", "off", "required"}:
        raise ValueError("llm mode must be auto, off, or required")
    if mode == "off":
        return None
    required_names = ("LLM_MODEL_ID", "LLM_API_KEY", "LLM_BASE_URL")
    missing = [name for name in required_names if not os.getenv(name)]
    if missing:
        if mode == "required":
            raise ValueError("missing required LLM environment variables: %s" % missing)
        return None
    return HelloAgentsLLM(temperature=0.0)
