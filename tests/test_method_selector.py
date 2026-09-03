import json

from inpainting_research_agent.knowledge import LocalKnowledgeRetriever
from inpainting_research_agent.method_selector import MethodSelector


def _profile():
    return {
        "image_shape": [64, 128, 3],
        "mask_type": "block",
        "actual_missing_rate": 0.4,
        "missing_component_count": 1,
        "largest_missing_component_image_ratio": 0.4,
        "image_aspect_ratio": 2.0,
        "visible_mean_absolute_channel_correlation": 0.82,
        "visible_local_smoothness_score": 0.76,
        "visible_high_frequency_energy_ratio": 0.04,
    }


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.outputs.pop(0)


def _valid_output(source):
    return json.dumps(
        {
            "method": "tucker",
            "reason": (
                "The block mask, high channel correlation, and anisotropic image "
                "support separate Tucker spatial and channel ranks."
            ),
            "evidence": [
                {
                    "source": source,
                    "claim": "The retrieved evidence supports separate spatial ranks.",
                }
            ],
            "confidence": 0.81,
            "suggested_hyperparameters": {
                "rank_h_candidates": [8, 12],
                "rank_w_candidates": [12, 16],
                "rank_c_candidates": [2, 3],
                "init_scale": 0.15,
            },
            "risks": ["The contiguous hole may still show global banding."],
            "selection_mode": "llm",
        }
    )


def _retrieval():
    return LocalKnowledgeRetriever().retrieve(
        profile=_profile(),
        query="block channel correlation spatial rank failure modes",
        top_k=6,
    )


def test_local_retrieval_returns_rules_and_sourced_chunks():
    retrieval = _retrieval()
    assert retrieval["active_rules"]
    assert retrieval["evidence"]
    assert all("#" in item["source"] for item in retrieval["evidence"])
    assert any(rule["prefer"] == "tucker" for rule in retrieval["active_rules"])


def test_valid_llm_method_plan_is_accepted_stably():
    retrieval = _retrieval()
    source = retrieval["evidence"][0]["source"]
    llm = FakeLLM([_valid_output(source)])
    result = MethodSelector(llm).select(_profile(), retrieval)
    assert result["plan"].method == "tucker"
    assert result["plan"].selection_mode == "llm"
    assert result["attempts"] == 1
    assert llm.calls[0]["kwargs"]["temperature"] == 0.0


def test_invalid_json_is_repaired_once():
    retrieval = _retrieval()
    source = retrieval["evidence"][0]["source"]
    llm = FakeLLM(["not JSON", _valid_output(source)])
    result = MethodSelector(llm).select(_profile(), retrieval)
    assert result["plan"].selection_mode == "llm_repaired"
    assert result["attempts"] == 2
    assert len(result["validation_errors"]) == 1
    assert "previous JSON was invalid" in llm.calls[1]["messages"][-1]["content"]


def test_two_invalid_outputs_use_deterministic_fallback():
    retrieval = _retrieval()
    llm = FakeLLM(["bad", '{"method": "not-supported"}'])
    result = MethodSelector(llm).select(_profile(), retrieval)
    assert result["plan"].method in {"matrix", "cp", "tucker"}
    assert result["plan"].selection_mode == "deterministic_fallback"
    assert result["fallback_reason"] is not None
    assert result["attempts"] == 2


def test_same_profile_has_stable_deterministic_plan():
    retrieval = _retrieval()
    first = MethodSelector(None).select(_profile(), retrieval)["plan"]
    second = MethodSelector(None).select(_profile(), retrieval)["plan"]
    assert first.model_dump() == second.model_dump()
