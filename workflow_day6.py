"""Day 6 fair evaluation, bounded improvement loop, and algorithm promotion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .agent_tools.framework import TraceLogger
from .candidate import (
    CandidateGenerator,
    CandidateValidator,
    candidate_builder,
    load_validated_candidate,
    promote_candidate,
)
from .candidate.generator import load_improver_context
from .core.data import load_observation_mask, load_rgb_image
from .core.experiment_judge import judge_candidate
from .core.fair_experiment import (
    final_fit_and_evaluate,
    paired_trial_configurations,
    tune_model_on_observed_pixels,
)
from .core.models.registry import MODEL_CLASSES
from .method_selector import llm_from_environment
from .schemas import TrainingConfig
from .workflow_day5 import _identifier, _write_json


@dataclass(frozen=True)
class Day6WorkflowConfig:
    base_run_dir: str
    initial_candidate_dir: str
    candidate_root: str = "inpainting_research_agent/algorithms/candidates"
    approved_root: str = "inpainting_research_agent/algorithms/approved"
    output_dir: str = "inpainting_research_agent/outputs"
    llm_mode: str = "auto"
    tuning_trials: int = 4
    max_steps: int = 200
    max_improvement_rounds: int = 2
    validation_ratio: float = 0.1
    validation_interval: int = 10
    patience: int = 20
    device: str = "auto"
    minimum_psnr_delta: float = 0.2
    ssim_tolerance: float = 0.002
    smoke_timeout_seconds: float = 10.0

    def validate(self) -> None:
        if not (Path(self.base_run_dir) / "state.json").is_file():
            raise ValueError("base_run_dir must contain state.json")
        if not (Path(self.initial_candidate_dir) / "manifest.json").is_file():
            raise ValueError("initial_candidate_dir must contain manifest.json")
        if self.llm_mode not in {"auto", "off", "required"}:
            raise ValueError("llm_mode must be auto, off, or required")
        if not 1 <= self.tuning_trials <= 5:
            raise ValueError("tuning_trials must be in [1, 5]")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 1 <= self.max_improvement_rounds <= 2:
            raise ValueError("max_improvement_rounds must be in [1, 2]")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be in (0, 1)")
        if self.validation_interval < 1 or self.patience < 1:
            raise ValueError("validation_interval and patience must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.minimum_psnr_delta < 0.0 or self.ssim_tolerance < 0.0:
            raise ValueError("judge thresholds must be non-negative")


class Day6Workflow:
    """Evaluate candidate hypotheses under a fixed, leak-free experiment protocol."""

    def __init__(
        self,
        config: Day6WorkflowConfig,
        generator: Optional[CandidateGenerator] = None,
        validator: Optional[CandidateValidator] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.workflow_id = _identifier("day6")
        self.run_dir = Path(config.output_dir) / self.workflow_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.state_path = self.run_dir / "state.json"
        self.generator = generator or CandidateGenerator(llm_from_environment(config.llm_mode))
        self.validator = validator or CandidateValidator(config.smoke_timeout_seconds)
        self.trace = TraceLogger(output_dir=str(self.run_dir / "traces"), sanitize=True)
        self.base_state = json.loads(
            (Path(config.base_run_dir) / "state.json").read_text(encoding="utf-8")
        )
        if self.base_state.get("stage") != "COMPLETED":
            raise ValueError("base run must be COMPLETED")
        self.state: Dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "stage": "CREATED",
            "config": asdict(config),
            "base_run_id": self.base_state["run_id"],
            "rounds": [],
            "feedback_history": [],
            "accepted": False,
            "stop_reason": None,
            "best_available": None,
            "promotion": None,
            "overall_comparison": None,
            "artifacts": {
                "run_dir": str(self.run_dir),
                "state": str(self.state_path),
                "trace_jsonl": str(self.trace.jsonl_path),
                "trace_html": str(self.trace.html_path),
            },
            "updated_at": datetime.now().isoformat(),
        }
        self._save()

    def _save(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat()
        _write_json(self.state_path, self.state)

    @staticmethod
    def _validate_candidate_search_contract(
        base_search_space: Dict[str, Any],
        candidate_search_space: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> None:
        """Ensure executable code did not widen the proposal's declared space."""

        executable_extra = {
            key: values
            for key, values in candidate_search_space.items()
            if key not in base_search_space
        }
        declared = {
            key: values
            for key, values in manifest.get("allowed_search_space", {}).items()
            if key not in base_search_space
        }
        if executable_extra != declared:
            raise ValueError(
                "candidate executable search space differs from its validated manifest"
            )

    def _make_next_candidate(self, context: Dict[str, Any], round_index: int) -> str:
        generation = self.generator.generate(context)
        proposal = generation.proposal
        candidate_id = _identifier("candidate")
        directory = Path(self.config.candidate_root) / candidate_id
        directory.mkdir(parents=True, exist_ok=False)
        model_path = directory / "model.py"
        idea_path = directory / "idea.json"
        manifest_path = directory / "manifest.json"
        validation_path = directory / "validation.json"
        model_path.write_text(proposal.model_code, encoding="utf-8")
        code_hash = hashlib.sha256(proposal.model_code.encode("utf-8")).hexdigest()
        idea = proposal.model_dump(exclude={"model_code"})
        idea.update(
            {
                "candidate_id": candidate_id,
                "improvement_round": round_index,
                "previous_failure_feedback": context["previous_failure_feedback"],
                "prompt_version": generation.prompt_version,
                "generation_attempts": generation.attempts,
                "fallback_reason": generation.fallback_reason,
                "schema_validation_errors": generation.validation_errors,
                "raw_outputs": generation.raw_outputs,
            }
        )
        _write_json(idea_path, idea)
        manifest = {
            "candidate_id": candidate_id,
            "base_run_id": context["base_run_id"],
            "base_method": proposal.base_method,
            "created_at": datetime.now().isoformat(),
            "improvement_round": round_index,
            "generation_mode": proposal.generation_mode,
            "llm_model": getattr(self.generator.llm, "model", None),
            "prompt_version": generation.prompt_version,
            "code_sha256": code_hash,
            "validation_status": "pending",
            "allowed_search_space": proposal.search_space,
            "eligible_for_training": False,
        }
        _write_json(manifest_path, manifest)
        validation = self.validator.validate(str(model_path), str(validation_path))
        manifest.update(
            {
                "validation_status": validation["status"],
                "eligible_for_training": bool(validation["passed"]),
                "validated_at": datetime.now().isoformat(),
            }
        )
        _write_json(manifest_path, manifest)
        self.trace.log_event(
            "candidate_regenerated",
            {
                "round": round_index,
                "candidate_id": candidate_id,
                "generation_mode": proposal.generation_mode,
                "validation_passed": validation["passed"],
                "feedback_count": len(context["previous_failure_feedback"]),
            },
            step=round_index,
        )
        if not validation["passed"]:
            raise RuntimeError("regenerated candidate failed validation: %s" % validation["feedback"])
        return str(directory)

    def run(self) -> Dict[str, Any]:
        self.trace.log_event(
            "session_start",
            {
                "workflow": "day6_fair_evaluation_and_bounded_improvement",
                "workflow_id": self.workflow_id,
                "maximum_rounds": self.config.max_improvement_rounds,
            },
        )
        try:
            artifacts = self.base_state["artifacts"]
            observed = load_rgb_image(artifacts["corrupted"], max_size=None)
            observed_mask = load_observation_mask(artifacts["mask"])
            ground_truth_path = Path(self.config.base_run_dir) / "evaluation_ground_truth.png"
            ground_truth = load_rgb_image(str(ground_truth_path), max_size=None)
            if observed.shape != ground_truth.shape or observed_mask.shape != observed.shape[:2]:
                raise ValueError("base-run image artifacts have inconsistent shapes")

            base_method = self.base_state["selected_model"]
            base_class = MODEL_CLASSES[base_method]
            anchor = self.base_state["results"]["selected_trial"]["hyperparameters"]
            learning_rate = float(self.base_state["results"]["training"]["learning_rate"])
            seed = int(self.base_state["config"]["seed"])
            training = TrainingConfig(
                learning_rate=learning_rate,
                max_steps=self.config.max_steps,
                validation_observed_ratio=self.config.validation_ratio,
                validation_interval=self.config.validation_interval,
                early_stopping_patience=self.config.patience,
                device=self.config.device,
            )
            improver_context = load_improver_context(self.config.base_run_dir)
            candidate_dir = self.config.initial_candidate_dir

            for round_index in range(1, self.config.max_improvement_rounds + 1):
                candidate_class, candidate_manifest = load_validated_candidate(candidate_dir)
                base_search_space = base_class.search_space(tuple(observed.shape))
                candidate_search_space = candidate_class.search_space(
                    tuple(observed.shape)
                )
                self._validate_candidate_search_contract(
                    base_search_space,
                    candidate_search_space,
                    candidate_manifest,
                )
                paired = paired_trial_configurations(
                    base_search_space=base_search_space,
                    candidate_search_space=candidate_search_space,
                    trial_count=self.config.tuning_trials,
                    seed=seed,
                    anchor_base_config=anchor,
                )
                round_dir = self.run_dir / ("round_%d" % round_index)
                baseline_tuning = tune_model_on_observed_pixels(
                    model_name=base_method,
                    model_builder=None,
                    configurations=paired["baseline"],
                    observed_image=observed,
                    observed_mask=observed_mask,
                    training_config=training,
                    seed=seed,
                )
                candidate_tuning = tune_model_on_observed_pixels(
                    model_name=candidate_manifest["candidate_id"],
                    model_builder=candidate_builder(candidate_class),
                    configurations=paired["candidate"],
                    observed_image=observed,
                    observed_mask=observed_mask,
                    training_config=training,
                    seed=seed,
                )
                _write_json(round_dir / "paired_configurations.json", paired)
                _write_json(round_dir / "baseline_tuning.json", baseline_tuning)
                _write_json(round_dir / "candidate_tuning.json", candidate_tuning)

                # Hidden-region ground truth first enters the workflow here, after tuning.
                baseline_final = final_fit_and_evaluate(
                    base_method,
                    None,
                    baseline_tuning["best"],
                    observed,
                    observed_mask,
                    ground_truth,
                    training,
                    seed,
                    str(round_dir / "baseline_final"),
                )
                candidate_final = final_fit_and_evaluate(
                    candidate_manifest["candidate_id"],
                    candidate_builder(candidate_class),
                    candidate_tuning["best"],
                    observed,
                    observed_mask,
                    ground_truth,
                    training,
                    seed,
                    str(round_dir / "candidate_final"),
                )
                judgment = judge_candidate(
                    baseline_tuning,
                    baseline_final,
                    candidate_tuning,
                    candidate_final,
                    self.config.minimum_psnr_delta,
                    self.config.ssim_tolerance,
                )
                _write_json(round_dir / "judgment.json", judgment)
                round_record = {
                    "round": round_index,
                    "candidate_id": candidate_manifest["candidate_id"],
                    "candidate_dir": candidate_dir,
                    "paired_configurations": paired,
                    "baseline_tuning": baseline_tuning,
                    "candidate_tuning": candidate_tuning,
                    "baseline_final": baseline_final,
                    "candidate_final": candidate_final,
                    "judgment": judgment,
                    "artifacts": {
                        "round_dir": str(round_dir),
                        "judgment": str(round_dir / "judgment.json"),
                    },
                }
                self.state["rounds"].append(round_record)
                self.state["feedback_history"].append(
                    {
                        "round": round_index,
                        "decision": judgment["decision"],
                        "psnr_delta": judgment["psnr_delta"],
                        "ssim_delta": judgment["ssim_delta"],
                        "runtime_ratio": judgment["runtime_ratio"],
                        "training_behavior": judgment["training_behavior"],
                        "suspected_causes": judgment["suspected_causes"],
                        "next_round_constraints": judgment["next_round_constraints"],
                    }
                )
                self.trace.log_event(
                    "experiment_judgment",
                    {
                        "round": round_index,
                        "candidate_id": candidate_manifest["candidate_id"],
                        "decision": judgment["decision"],
                        "psnr_delta": judgment["psnr_delta"],
                        "ssim_delta": judgment["ssim_delta"],
                        "equal_trial_counts": (
                            baseline_tuning["trial_count"] == candidate_tuning["trial_count"]
                        ),
                        "tuning_ground_truth_used": False,
                    },
                    step=round_index,
                )
                self._save()

                if judgment["accepted"]:
                    algorithm_name = "%s_tv_regularized" % base_method
                    self.state["promotion"] = promote_candidate(
                        candidate_dir=candidate_dir,
                        approved_root=self.config.approved_root,
                        algorithm_name=algorithm_name,
                        source_run_id=self.workflow_id,
                        best_config={
                            "hyperparameters": candidate_tuning["best"]["hyperparameters"],
                            "learning_rate": learning_rate,
                            "selected_steps": candidate_tuning["best"]["best_step"],
                        },
                        comparison=judgment,
                        conditions={
                            "image_shape": list(observed.shape),
                            "mask_type": self.base_state["config"]["mask_type"],
                            "actual_missing_rate": self.base_state["results"]["image_profile"][
                                "actual_missing_rate"
                            ],
                            "seed": seed,
                        },
                    )
                    self.state["accepted"] = True
                    self.state["stop_reason"] = "candidate_accepted"
                    break

                if round_index < self.config.max_improvement_rounds:
                    improver_context["previous_failure_feedback"] = list(
                        self.state["feedback_history"]
                    )
                    candidate_dir = self._make_next_candidate(
                        improver_context, round_index + 1
                    )

            final_round = self.state["rounds"][-1]
            if not self.state["accepted"]:
                self.state["stop_reason"] = "maximum_improvement_rounds_reached"
            eligible_results = [
                {
                    "algorithm": "nearest_neighbor_manhattan",
                    "role": "interpolation_baseline",
                    "reconstruction": artifacts["interpolation"],
                    "metrics": self.base_state["results"]["interpolation_metrics"],
                },
                {
                    "algorithm": base_method,
                    "role": "tensor_baseline",
                    "reconstruction": final_round["baseline_final"]["artifacts"][
                        "reconstruction"
                    ],
                    "metrics": final_round["baseline_final"]["metrics"],
                },
            ]
            if self.state["accepted"]:
                eligible_results.append(
                    {
                        "algorithm": self.state["promotion"]["algorithm_name"],
                        "role": "accepted_candidate",
                        "reconstruction": final_round["candidate_final"]["artifacts"][
                            "reconstruction"
                        ],
                        "metrics": final_round["candidate_final"]["metrics"],
                    }
                )
            winner = max(
                eligible_results,
                key=lambda item: (
                    float("inf")
                    if item["metrics"]["missing_psnr"] is None
                    else item["metrics"]["missing_psnr"]
                ),
            )
            self.state["overall_comparison"] = {
                "selection_rule": (
                    "highest missing-region PSNR among baselines and accepted candidates; "
                    "SSIM is reported but the fixed promotion gate remains separate"
                ),
                "eligible_results": eligible_results,
                "winner": winner["algorithm"],
            }
            self.state["best_available"] = winner
            self.state["stage"] = "COMPLETED"
            self._save()
            self.trace.log_event(
                "session_end",
                {
                    "workflow_id": self.workflow_id,
                    "stage": "COMPLETED",
                    "accepted": self.state["accepted"],
                    "round_count": len(self.state["rounds"]),
                    "stop_reason": self.state["stop_reason"],
                },
            )
            return self.state
        except Exception as error:
            self.state["stage"] = "FAILED"
            self.state["stop_reason"] = "workflow_error"
            self.state["last_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            self._save()
            self.trace.log_event(
                "error", {"error_type": type(error).__name__, "message": str(error)}
            )
            raise
        finally:
            self.trace.finalize()


def run_day6_workflow(
    config: Day6WorkflowConfig,
    generator: Optional[CandidateGenerator] = None,
) -> Dict[str, Any]:
    return Day6Workflow(config=config, generator=generator).run()
