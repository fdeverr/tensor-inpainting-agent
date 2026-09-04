"""Day 5 candidate generation and validation workflow."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .agent_tools.framework import TraceLogger
from .candidate import CandidateGenerator, CandidateValidator
from .candidate.generator import load_improver_context
from .method_selector import llm_from_environment


def _identifier(prefix: str) -> str:
    return "%s-%s-%s" % (
        prefix,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        uuid.uuid4().hex[:6],
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class Day5WorkflowConfig:
    base_run_dir: str
    candidate_root: str = "inpainting_research_agent/algorithms/candidates"
    output_dir: str = "inpainting_research_agent/outputs"
    llm_mode: str = "auto"
    smoke_timeout_seconds: float = 10.0

    def validate(self) -> None:
        state_path = Path(self.base_run_dir) / "state.json"
        if not state_path.is_file():
            raise ValueError("base_run_dir must contain state.json")
        if self.llm_mode not in {"auto", "off", "required"}:
            raise ValueError("llm_mode must be auto, off, or required")
        if self.smoke_timeout_seconds <= 0:
            raise ValueError("smoke_timeout_seconds must be positive")


class Day5Workflow:
    """Produce an auditable candidate and reject it unless both checks pass."""

    def __init__(
        self,
        config: Day5WorkflowConfig,
        generator: Optional[CandidateGenerator] = None,
        validator: Optional[CandidateValidator] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.workflow_id = _identifier("day5")
        self.run_dir = Path(config.output_dir) / self.workflow_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.state_path = self.run_dir / "state.json"
        self.generator = generator or CandidateGenerator(
            llm_from_environment(config.llm_mode)
        )
        self.validator = validator or CandidateValidator(
            timeout_seconds=config.smoke_timeout_seconds
        )
        self.trace = TraceLogger(output_dir=str(self.run_dir / "traces"), sanitize=True)
        self.state: Dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "stage": "CREATED",
            "config": asdict(config),
            "candidate_id": None,
            "base_run_id": None,
            "artifacts": {
                "state": str(self.state_path),
                "trace_jsonl": str(self.trace.jsonl_path),
                "trace_html": str(self.trace.html_path),
            },
            "generation": None,
            "validation": None,
            "updated_at": datetime.now().isoformat(),
        }
        self._save()

    def _save(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat()
        _write_json(self.state_path, self.state)

    def run(self) -> Dict[str, Any]:
        self.trace.log_event(
            "session_start",
            {
                "workflow": "day5_candidate_generation_and_validation",
                "workflow_id": self.workflow_id,
                "llm_used": self.generator.llm is not None,
            },
        )
        try:
            context = load_improver_context(self.config.base_run_dir)
            self.state["base_run_id"] = context["base_run_id"]
            self.trace.log_event(
                "model_improver_input",
                {
                    "base_run_id": context["base_run_id"],
                    "base_method": context["base_method"],
                    "profile": context["image_profile"],
                    "method_plan": context["method_plan"],
                    "base_best_config": context["base_best_config"],
                    "training_curve_summary": context["training_curve_summary"],
                    "base_metrics": context["base_metrics"],
                    "comparison": context["comparison"],
                    "ground_truth_provided": False,
                    "base_source_character_count": len(context["base_model_source"]),
                },
                step=1,
            )
            generation = self.generator.generate(context)
            proposal = generation.proposal
            candidate_id = _identifier("candidate")
            candidate_dir = Path(self.config.candidate_root) / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=False)
            idea_path = candidate_dir / "idea.json"
            model_path = candidate_dir / "model.py"
            manifest_path = candidate_dir / "manifest.json"
            validation_path = candidate_dir / "validation.json"
            model_path.write_text(proposal.model_code, encoding="utf-8")
            code_hash = hashlib.sha256(
                proposal.model_code.encode("utf-8")
            ).hexdigest()
            idea_payload = proposal.model_dump(exclude={"model_code"})
            idea_payload.update(
                {
                    "candidate_id": candidate_id,
                    "prompt_version": generation.prompt_version,
                    "generation_attempts": generation.attempts,
                    "fallback_reason": generation.fallback_reason,
                    "schema_validation_errors": generation.validation_errors,
                    "raw_outputs": generation.raw_outputs,
                }
            )
            _write_json(idea_path, idea_payload)
            llm_model = getattr(self.generator.llm, "model", None)
            manifest = {
                "candidate_id": candidate_id,
                "base_run_id": context["base_run_id"],
                "base_method": proposal.base_method,
                "created_at": datetime.now().isoformat(),
                "generation_mode": proposal.generation_mode,
                "llm_model": llm_model,
                "prompt_version": generation.prompt_version,
                "code_sha256": code_hash,
                "validation_status": "pending",
                "allowed_search_space": proposal.search_space,
                "eligible_for_training": False,
            }
            _write_json(manifest_path, manifest)
            self.state.update(
                {
                    "stage": "GENERATED",
                    "candidate_id": candidate_id,
                    "generation": {
                        "mode": proposal.generation_mode,
                        "attempts": generation.attempts,
                        "fallback_reason": generation.fallback_reason,
                        "hypothesis": proposal.hypothesis,
                    },
                }
            )
            self.state["artifacts"].update(
                {
                    "candidate_dir": str(candidate_dir),
                    "idea": str(idea_path),
                    "model": str(model_path),
                    "manifest": str(manifest_path),
                    "validation": str(validation_path),
                }
            )
            self._save()
            self.trace.log_event(
                "candidate_generated",
                {
                    "candidate_id": candidate_id,
                    "base_method": proposal.base_method,
                    "hypothesis": proposal.hypothesis,
                    "proposed_changes": proposal.proposed_changes,
                    "search_space": proposal.search_space,
                    "generation_mode": proposal.generation_mode,
                    "code_sha256": code_hash,
                },
                step=2,
            )

            validation = self.validator.validate(
                model_path=str(model_path),
                output_path=str(validation_path),
            )
            manifest["validation_status"] = validation["status"]
            manifest["eligible_for_training"] = bool(validation["passed"])
            manifest["validated_at"] = datetime.now().isoformat()
            _write_json(manifest_path, manifest)
            self.state["stage"] = "VALIDATED" if validation["passed"] else "REJECTED"
            self.state["validation"] = {
                "passed": validation["passed"],
                "status": validation["status"],
                "feedback": validation["feedback"],
                "runtime_seconds": validation["runtime_seconds"],
                "eligible_for_training": validation["passed"],
            }
            self._save()
            self.trace.log_event(
                "candidate_validation",
                {
                    "candidate_id": candidate_id,
                    "passed": validation["passed"],
                    "status": validation["status"],
                    "static_validation": validation["static_validation"],
                    "smoke_test": validation["smoke_test"],
                    "feedback": validation["feedback"],
                },
                step=3,
            )
            self.trace.log_event(
                "session_end",
                {
                    "workflow_id": self.workflow_id,
                    "stage": self.state["stage"],
                    "candidate_id": candidate_id,
                },
            )
            return self.state
        except Exception as error:
            self.state["stage"] = "FAILED"
            self.state["validation"] = {
                "passed": False,
                "status": "workflow_error",
                "feedback": [str(error)],
                "eligible_for_training": False,
            }
            self._save()
            self.trace.log_event(
                "error",
                {"error_type": type(error).__name__, "message": str(error)},
            )
            self.trace.log_event(
                "session_end",
                {"workflow_id": self.workflow_id, "stage": "FAILED"},
            )
            raise
        finally:
            self.trace.finalize()


def run_day5_workflow(
    config: Day5WorkflowConfig,
    generator: Optional[CandidateGenerator] = None,
) -> Dict[str, Any]:
    return Day5Workflow(config=config, generator=generator).run()
