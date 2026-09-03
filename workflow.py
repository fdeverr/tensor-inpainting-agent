"""Day 3 deterministic state machine built on Hello Agents tools."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_tools import build_research_tool_registry
from .agent_tools.framework import ToolStatus, TraceLogger


STAGES = (
    "CREATED",
    "ANALYZED",
    "INTERPOLATED",
    "TUNED",
    "TRAINED",
    "EVALUATED",
    "COMPLETED",
)


def _make_run_id(prefix: str = "day3") -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return "%s-%s-%s" % (prefix, timestamp, uuid.uuid4().hex[:6])


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, allow_nan=False)
        output_file.write("\n")


@dataclass(frozen=True)
class Day3WorkflowConfig:
    """Inputs for the deterministic Day 3 research workflow."""

    image_path: str
    output_dir: str = "inpainting_research_agent/outputs"
    mask_type: str = "block"
    missing_rate: float = 0.4
    seed: int = 42
    image_size: Optional[int] = 128
    model_name: str = "tucker"
    max_steps: int = 200
    validation_ratio: float = 0.1
    validation_interval: int = 10
    patience: int = 20
    device: str = "auto"
    candidates: Optional[List[Dict[str, Any]]] = field(default=None)

    def validate(self) -> None:
        if not Path(self.image_path).is_file():
            raise ValueError("image_path does not point to a file: %s" % self.image_path)
        if self.mask_type not in {"random", "block"}:
            raise ValueError("mask_type must be random or block")
        if not 0.0 < self.missing_rate < 1.0:
            raise ValueError("missing_rate must be strictly between 0 and 1")
        if self.image_size is not None and self.image_size < 8:
            raise ValueError("image_size must be at least 8 or None")
        if self.model_name != "tucker":
            raise ValueError("Day 3 workflow intentionally fixes model_name to tucker")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


class WorkflowExecutionError(RuntimeError):
    """Raised after a structured tool failure has been persisted."""


class Day3Workflow:
    """Call registered tools in a fixed, auditable order without an LLM."""

    run_prefix = "day3"
    workflow_name = "day3_deterministic_research_workflow"
    llm_used = False

    def __init__(self, config: Day3WorkflowConfig) -> None:
        config.validate()
        self.config = config
        self.run_id = _make_run_id(self.run_prefix)
        self.run_dir = Path(config.output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.state_path = self.run_dir / "state.json"
        self.registry = build_research_tool_registry()
        self.trace = TraceLogger(
            output_dir=str(self.run_dir / "traces"),
            sanitize=True,
        )
        self.step = 0
        self.state: Dict[str, Any] = {
            "run_id": self.run_id,
            "stage": "CREATED",
            "last_successful_stage": "CREATED",
            "selected_model": config.model_name,
            "config": asdict(config),
            "artifacts": {
                "run_dir": str(self.run_dir),
                "state": str(self.state_path),
                "trace_jsonl": str(self.trace.jsonl_path),
                "trace_html": str(self.trace.html_path),
            },
            "results": {},
            "last_error": None,
            "updated_at": datetime.now().isoformat(),
        }
        self._save_state()

    def _save_state(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat()
        _write_json(self.state_path, self.state)

    def _assert_stage(self, expected: str) -> None:
        actual = self.state["stage"]
        if actual != expected:
            raise WorkflowExecutionError(
                "workflow expected stage %s, found %s" % (expected, actual)
            )

    @staticmethod
    def _trace_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Keep paths/configs in Trace while excluding evaluation-only GT."""

        return {
            key: value
            for key, value in parameters.items()
            if key != "ground_truth_path"
        }

    @staticmethod
    def _response_summary(response) -> Dict[str, Any]:
        data = response.data or {}
        return {
            "status": response.status.value,
            "text": response.text,
            "run_id": data.get("run_id"),
            "artifacts": data.get("artifacts", {}),
            "winner": data.get("winner"),
            "error": response.error_info,
            "time_ms": (response.stats or {}).get("time_ms"),
        }

    def _call_tool(self, tool_name: str, parameters: Dict[str, Any]):
        self.step += 1
        self.trace.log_event(
            "tool_call",
            {
                "tool_name": tool_name,
                "workflow_stage": self.state["stage"],
                "parameters": self._trace_parameters(parameters),
            },
            step=self.step,
        )
        response = self.registry.execute_tool(tool_name, parameters)
        self.trace.log_event(
            "tool_result",
            {
                "tool_name": tool_name,
                **self._response_summary(response),
            },
            step=self.step,
        )
        if response.status == ToolStatus.ERROR:
            self.state["last_error"] = {
                "tool_name": tool_name,
                "stage": self.state["stage"],
                "error": response.error_info,
            }
            self._save_state()
            self.trace.log_event(
                "error",
                {
                    "error_type": (response.error_info or {}).get("code"),
                    "message": response.text,
                    "tool_name": tool_name,
                    "last_successful_stage": self.state["last_successful_stage"],
                },
                step=self.step,
            )
            raise WorkflowExecutionError(
                "%s failed: %s" % (tool_name, response.text)
            )
        return response

    def _transition(self, expected: str, target: str) -> None:
        self._assert_stage(expected)
        if target not in STAGES:
            raise ValueError("unknown workflow stage %s" % target)
        self.state["stage"] = target
        self.state["last_successful_stage"] = target
        self.state["last_error"] = None
        self._save_state()
        self.trace.log_event(
            "state_transition",
            {"from": expected, "to": target},
            step=self.step,
        )

    def _select_method(self) -> Dict[str, Any]:
        """Day 3 hook: use the deliberately fixed Tucker decision."""

        return {
            "method": self.config.model_name,
            "reason": "Day 3 fixes the method so the tool workflow can be tested without an LLM.",
            "evidence": [],
            "confidence": 1.0,
            "suggested_hyperparameters": {},
            "risks": ["This is a workflow test, not a data-dependent method decision."],
            "selection_mode": "fixed_day3",
        }

    def _tuning_candidates(
        self,
        selected_model: str,
        method_plan: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """Hook used by Day 4 to turn a validated plan into a small grid."""

        return self.config.candidates

    def run(self) -> Dict[str, Any]:
        """Execute all Day 3 stages and return the persisted final state."""

        self.trace.log_event(
            "session_start",
            {
                "run_id": self.run_id,
                "workflow": self.workflow_name,
                "llm_used": self.llm_used,
            },
        )
        try:
            self._assert_stage("CREATED")
            analysis = self._call_tool(
                "analyze_image",
                {
                    "run_id": self.run_id,
                    "image_path": self.config.image_path,
                    "run_dir": str(self.run_dir),
                    "mask_type": self.config.mask_type,
                    "missing_rate": self.config.missing_rate,
                    "seed": self.config.seed,
                    "image_size": self.config.image_size,
                },
            )
            self.state["artifacts"].update(analysis.data["artifacts"])
            self.state["results"]["image_profile"] = analysis.data["profile"]
            # This path is never returned by AnalyzeImageTool and therefore is
            # unavailable to a future LLM controller.  Only the deterministic
            # evaluator branch knows the fixed internal artifact name.
            ground_truth_path = str(self.run_dir / "evaluation_ground_truth.png")
            self._transition("CREATED", "ANALYZED")

            interpolation_path = self.run_dir / "interpolated.png"
            interpolation = self._call_tool(
                "run_interpolation",
                {
                    "run_id": self.run_id,
                    "corrupted_path": self.state["artifacts"]["corrupted"],
                    "mask_path": self.state["artifacts"]["mask"],
                    "output_path": str(interpolation_path),
                },
            )
            self.state["artifacts"]["interpolation"] = interpolation.data[
                "artifacts"
            ]["reconstruction"]
            self._transition("ANALYZED", "INTERPOLATED")

            method_plan = self._select_method()
            selected_model = method_plan["method"]
            if selected_model not in {"matrix", "cp", "tucker"}:
                raise WorkflowExecutionError(
                    "method selector returned unsupported model %r" % selected_model
                )
            self.state["selected_model"] = selected_model
            self.state["results"]["method_plan"] = method_plan
            self._save_state()
            self.trace.log_event(
                "method_selection",
                {
                    "method": selected_model,
                    "reason": method_plan["reason"],
                    "evidence": method_plan["evidence"],
                    "confidence": method_plan["confidence"],
                    "selection_mode": method_plan["selection_mode"],
                },
                step=self.step,
            )

            tuning_path = self.run_dir / "tuning_result.json"
            tuning_parameters = {
                "run_id": self.run_id,
                "corrupted_path": self.state["artifacts"]["corrupted"],
                "mask_path": self.state["artifacts"]["mask"],
                "output_path": str(tuning_path),
                "model_name": selected_model,
                "seed": self.config.seed,
                "max_steps": self.config.max_steps,
                "validation_ratio": self.config.validation_ratio,
                "validation_interval": self.config.validation_interval,
                "patience": self.config.patience,
                "device": self.config.device,
            }
            tuning_candidates = self._tuning_candidates(
                selected_model,
                method_plan,
            )
            if tuning_candidates is not None:
                tuning_parameters["candidates"] = tuning_candidates
            tuning = self._call_tool("tune_tensor_model", tuning_parameters)
            self.state["artifacts"]["tuning_result"] = tuning.data["artifacts"][
                "tuning_result"
            ]
            self.state["results"]["selected_trial"] = tuning.data["best"]
            self._transition("INTERPOLATED", "TUNED")

            best = tuning.data["best"]
            training = self._call_tool(
                "train_tensor_model",
                {
                    "run_id": self.run_id,
                    "corrupted_path": self.state["artifacts"]["corrupted"],
                    "mask_path": self.state["artifacts"]["mask"],
                    "output_dir": str(self.run_dir / "tensor_model"),
                    "model_name": selected_model,
                    "hyperparameters": best["hyperparameters"],
                    "learning_rate": best["learning_rate"],
                    "selected_steps": best["best_step"],
                    "seed": self.config.seed,
                    "validation_interval": self.config.validation_interval,
                    "device": self.config.device,
                },
            )
            self.state["artifacts"].update(
                {
                    "tensor_%s" % key: value
                    for key, value in training.data["artifacts"].items()
                }
            )
            self.state["results"]["training"] = {
                key: training.data[key]
                for key in (
                    "model_name",
                    "hyperparameters",
                    "learning_rate",
                    "fitted_steps",
                    "final_train_mse",
                    "observed_pixels_used",
                    "parameter_count",
                    "runtime_seconds",
                    "device",
                )
            }
            self._transition("TUNED", "TRAINED")

            baseline_metrics_path = self.run_dir / "interpolation_metrics.json"
            tensor_metrics_path = self.run_dir / "tensor_metrics.json"
            baseline_evaluation = self._call_tool(
                "evaluate_reconstruction",
                {
                    "run_id": self.run_id,
                    "algorithm_name": "nearest_neighbor_manhattan",
                    "reconstruction_path": self.state["artifacts"]["interpolation"],
                    "ground_truth_path": ground_truth_path,
                    "mask_path": self.state["artifacts"]["mask"],
                    "output_path": str(baseline_metrics_path),
                },
            )
            tensor_evaluation = self._call_tool(
                "evaluate_reconstruction",
                {
                    "run_id": self.run_id,
                    "algorithm_name": "%s_tensor_decomposition"
                    % selected_model,
                    "reconstruction_path": self.state["artifacts"][
                        "tensor_reconstruction"
                    ],
                    "ground_truth_path": ground_truth_path,
                    "mask_path": self.state["artifacts"]["mask"],
                    "output_path": str(tensor_metrics_path),
                },
            )
            self.state["artifacts"]["interpolation_metrics"] = str(
                baseline_metrics_path
            )
            self.state["artifacts"]["tensor_metrics"] = str(tensor_metrics_path)
            metric_keys = (
                "missing_mse",
                "missing_psnr",
                "perfect_reconstruction",
                "composite_ssim",
            )
            self.state["results"]["interpolation_metrics"] = {
                key: baseline_evaluation.data[key] for key in metric_keys
            }
            self.state["results"]["tensor_metrics"] = {
                key: tensor_evaluation.data[key] for key in metric_keys
            }
            self._transition("TRAINED", "EVALUATED")

            comparison_path = self.run_dir / "comparison.json"
            comparison = self._call_tool(
                "compare_experiments",
                {
                    "run_id": self.run_id,
                    "baseline_metrics_path": str(baseline_metrics_path),
                    "candidate_metrics_path": str(tensor_metrics_path),
                    "output_path": str(comparison_path),
                },
            )
            self.state["artifacts"]["comparison"] = str(comparison_path)
            self.state["results"]["comparison"] = {
                key: value
                for key, value in comparison.data.items()
                if key not in {"artifacts", "run_id"}
            }
            self._transition("EVALUATED", "COMPLETED")
            self.trace.log_event(
                "session_end",
                {
                    "run_id": self.run_id,
                    "stage": self.state["stage"],
                    "winner": self.state["results"]["comparison"]["winner"],
                },
            )
            return self.state
        except Exception:
            self.trace.log_event(
                "session_end",
                {
                    "run_id": self.run_id,
                    "stage": self.state["stage"],
                    "completed": False,
                },
            )
            raise
        finally:
            self.trace.finalize()


def run_day3_workflow(config: Day3WorkflowConfig) -> Dict[str, Any]:
    """Convenience API used by the CLI and tests."""

    return Day3Workflow(config).run()
