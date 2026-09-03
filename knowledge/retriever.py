"""Small heading/keyword/rule retriever; no vector database required."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


METHOD_FILE_NAMES = {
    "matrix": "matrix_factorization.md",
    "cp": "cp_decomposition.md",
    "tucker": "tucker_decomposition.md",
}


def _condition_is_active(condition: str, profile: Dict[str, Any]) -> bool:
    correlation = float(profile["visible_mean_absolute_channel_correlation"])
    largest_hole = float(profile["largest_missing_component_image_ratio"])
    component_count = int(profile["missing_component_count"])
    condition_map = {
        "block_missing": profile["mask_type"] == "block",
        "random_missing": profile["mask_type"] == "random",
        "channel_correlation_high": correlation >= 0.75,
        "channel_correlation_low": correlation < 0.35,
        "large_contiguous_hole": largest_hole >= 0.20,
        "many_small_holes": component_count >= 8 and largest_hole < 0.10,
        "image_aspect_ratio_high": float(profile["image_aspect_ratio"]) >= 1.6,
        "local_smoothness_high": float(
            profile["visible_local_smoothness_score"]
        ) >= 0.75,
        "high_frequency_high": float(
            profile["visible_high_frequency_energy_ratio"]
        ) >= 0.08,
    }
    if condition not in condition_map:
        raise ValueError("unknown selection condition %r" % condition)
    return condition_map[condition]


def _markdown_chunks(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    chunks = []
    heading = "Introduction"
    body_lines = []
    for line in text.splitlines():
        if line.startswith("#"):
            if body_lines:
                chunks.append(
                    {
                        "heading": heading,
                        "content": "\n".join(body_lines).strip(),
                    }
                )
            heading = line.lstrip("#").strip()
            body_lines = []
        else:
            body_lines.append(line)
    if body_lines:
        chunks.append(
            {"heading": heading, "content": "\n".join(body_lines).strip()}
        )
    return [chunk for chunk in chunks if chunk["content"]]


class LocalKnowledgeRetriever:
    """Retrieve auditable evidence from local Markdown and YAML rules."""

    def __init__(
        self,
        knowledge_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.knowledge_dir = (
            Path(knowledge_dir)
            if knowledge_dir is not None
            else Path(__file__).resolve().parent
        )
        self.methods_dir = self.knowledge_dir / "methods"
        self.rules_path = self.knowledge_dir / "selection_rules.yaml"
        if not self.rules_path.is_file():
            raise ValueError("selection rules do not exist: %s" % self.rules_path)

    def active_rules(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        payload = yaml.safe_load(self.rules_path.read_text(encoding="utf-8"))
        rules = payload.get("rules", []) if isinstance(payload, dict) else []
        active = []
        for index, rule in enumerate(rules):
            if _condition_is_active(str(rule["condition"]), profile):
                active.append(
                    {
                        **rule,
                        "source": "selection_rules.yaml#rule-%d" % index,
                    }
                )
        return active

    def retrieve(
        self,
        profile: Dict[str, Any],
        query: str,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        active_rules = self.active_rules(profile)
        query_terms = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        for rule in active_rules:
            query_terms.add(str(rule["prefer"]).lower())
            for keyword in rule.get("keywords", []):
                query_terms.update(
                    re.findall(r"[a-zA-Z0-9_]+", str(keyword).lower())
                )

        evidence = []
        preferred_methods = [str(rule["prefer"]) for rule in active_rules]
        for method, filename in METHOD_FILE_NAMES.items():
            path = self.methods_dir / filename
            for chunk in _markdown_chunks(path):
                searchable = (chunk["heading"] + "\n" + chunk["content"]).lower()
                score = sum(1.0 for term in query_terms if term in searchable)
                score += 2.0 * preferred_methods.count(method)
                if score <= 0.0:
                    continue
                evidence.append(
                    {
                        "source": "%s#%s" % (filename, chunk["heading"]),
                        "method": method,
                        "heading": chunk["heading"],
                        "score": score,
                        "content": chunk["content"],
                    }
                )
        evidence.sort(key=lambda item: (-item["score"], item["source"]))
        return {
            "query": query,
            "active_rules": active_rules,
            "evidence": evidence[:top_k],
        }
