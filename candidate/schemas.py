"""Strict schemas for an LLM-proposed tensor-model improvement."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CandidateProposal(BaseModel):
    """Research hypothesis and complete candidate source code."""

    model_config = ConfigDict(extra="forbid")

    base_method: Literal["matrix", "cp", "tucker"]
    hypothesis: str = Field(min_length=30, max_length=1500)
    proposed_changes: List[str] = Field(min_length=1, max_length=8)
    expected_effect: str = Field(min_length=20, max_length=1000)
    risks: List[str] = Field(min_length=1, max_length=8)
    search_space: Dict[str, List[Any]]
    model_code: str = Field(min_length=100, max_length=50_000)
    generation_mode: Literal[
        "llm",
        "llm_repaired",
        "deterministic_template",
    ] = "llm"

    @field_validator("search_space")
    @classmethod
    def validate_search_space(
        cls,
        value: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        if not value:
            raise ValueError("search_space cannot be empty")
        if any(not isinstance(options, list) or not options for options in value.values()):
            raise ValueError("every search_space entry must be a non-empty list")
        return value
