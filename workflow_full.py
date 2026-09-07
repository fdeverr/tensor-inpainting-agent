"""One-command orchestration of the complete seven-day research workflow."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .agent_tools.framework import TraceLogger
from .reporting import build_method_results, write_research_report
from .workflow import _make_run_id, _write_json
from .workflow_day4 import Day4WorkflowConfig, run_day4_workflow
from .workflow_day5 import Day5WorkflowConfig, run_day5_workflow
from .workflow_day6 import Day6WorkflowConfig, run_day6_workflow


DEFAULT_RESEARCH_PROMPT = "请分析图像和缺失模式，选择合适的张量分解，并在公平实验下提出、验证和改进一个补全算法。"


@dataclass(frozen=True)
class FullWorkflowConfig:
    image_path: str
    prompt: str = DEFAULT_RESEARCH_PROMPT
    output_dir: str = "inpainting_research_agent/outputs"
    candidate_root: str = "inpainting_research_agent/algorithms/candidates"
    approved_root: str = "inpainting_research_agent/algorithms/approved"
    mask_type: str = "block"
    missing_rate: float = 0.4
    seed: int = 42
    image_size: Optional[int] = 128
    method_max_steps: int = 200
    fair_max_steps: int = 200
    tuning_trials: int = 4
    max_improvement_rounds: int = 2
    validation_ratio: float = 0.1
    validation_interval: int = 10
    patience: int = 20
    device: str = "auto"
    llm_mode: str = "auto"
    retrieval_top_k: int = 8
    minimum_psnr_delta: float = 0.2
    ssim_tolerance: float = 0.002
    smoke_timeout_seconds: float = 10.0

    def validate(self) -> None:
        if not Path(self.image_path).is_file():
            raise ValueError("image_path does not point to a file: %s" % self.image_path)
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if self.mask_type not in {"random", "block"}:
            raise ValueError("mask_type must be random or block")
        if not 0.0 < self.missing_rate < 1.0:
            raise ValueError("missing_rate must be in (0, 1)")
        if self.image_size is not None and self.image_size < 8:
            raise ValueError("image_size must be at least 8 or None")
        if min(self.method_max_steps, self.fair_max_steps) < 1:
            raise ValueError("training steps must be positive")
        if not 1 <= self.tuning_trials <= 5:
            raise ValueError("tuning_trials must be in [1, 5]")
        if not 1 <= self.max_improvement_rounds <= 2:
            raise ValueError("max_improvement_rounds must be in [1, 2]")


def _score(metrics: Dict[str, Any]) -> float:
    return float("inf") if metrics["missing_psnr"] is None else metrics["missing_psnr"]


class FullResearchWorkflow:
    """Compose stable child workflows while retaining their independent traces."""

    def __init__(self, config: FullWorkflowConfig) -> None:
        config.validate()
        self.config = config
        self.run_id = _make_run_id("research-agent")
        self.run_dir = Path(config.output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.state_path = self.run_dir / "state.json"
        self.report_path = self.run_dir / "report.md"
        self.trace = TraceLogger(output_dir=str(self.run_dir / "traces"), sanitize=True)
        self.state: Dict[str, Any] = {
            "run_id": self.run_id,
            "stage": "CREATED",
            "prompt": config.prompt,
            "config": asdict(config),
            "child_runs": {},
            "method_results": [],
            "best_available": None,
            "candidate_accepted": False,
            "artifacts": {
                "run_dir": str(self.run_dir),
                "state": str(self.state_path),
                "report": str(self.report_path),
                "trace_jsonl": str(self.trace.jsonl_path),
                "trace_html": str(self.trace.html_path),
            },
            "last_error": None,
            "updated_at": datetime.now().isoformat(),
        }
        self._save()

    def _save(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat()
        _write_json(self.state_path, self.state)

    def _child_completed(self, day: str, state: Dict[str, Any]) -> None:
        state_path = state["artifacts"]["state"]
        self.state["child_runs"][day] = {
            "id": state.get("run_id", state.get("workflow_id")),
            "stage": state["stage"],
            "state": state_path,
        }
        self.state["artifacts"]["%s_state" % day] = state_path
        self._save()
        self.trace.log_event(
            "child_workflow_completed",
            {
                "day": day,
                "child_id": self.state["child_runs"][day]["id"],
                "stage": state["stage"],
            },
        )

    def run(self) -> Dict[str, Any]:
        self.trace.log_event(
            "session_start",
            {
                "workflow": "complete_inpainting_research_agent",
                "run_id": self.run_id,
                "prompt": self.config.prompt,
            },
        )
        day4: Dict[str, Any]
        day5: Dict[str, Any]
        day6: Optional[Dict[str, Any]] = None
        try:
            # ── 阶段 1：方法选择（Day 4 工作流）──────────────────────────
            # 输入：图片、mask 类型/缺失率、训练预算（Day4WorkflowConfig）
            # 作用：分析图像 → 最近邻插值 → 检索知识库选择 Matrix/CP/Tucker → 训练基础张量模型
            # 输出：day4 state（selected_model、image_profile、插值/张量指标、run 目录路径）
            self.state["stage"] = "METHOD_SELECTION"
            self._save()
            day4 = run_day4_workflow(
                Day4WorkflowConfig(
                    image_path=self.config.image_path,
                    output_dir=self.config.output_dir,
                    mask_type=self.config.mask_type,
                    missing_rate=self.config.missing_rate,
                    seed=self.config.seed,
                    image_size=self.config.image_size,
                    max_steps=self.config.method_max_steps,
                    validation_ratio=self.config.validation_ratio,
                    validation_interval=self.config.validation_interval,
                    patience=self.config.patience,
                    device=self.config.device,
                    llm_mode=self.config.llm_mode,
                    retrieval_top_k=self.config.retrieval_top_k,
                )
            )
            self._child_completed("day4", day4)

            # ── 阶段 2：候选生成与验证（Day 5 工作流）────────────────────
            # 输入：day4 的 run 目录（base_run_dir，作为改进起点）
            # 作用：读 Day4 上下文 → 生成候选改进（LLM/确定性模板）→ Schema + AST + smoke 三级验证
            # 输出：day5 state（candidate_dir、validation 状态、eligible_for_training 标志）
            self.state["stage"] = "CANDIDATE_GENERATION"
            self._save()
            day5 = run_day5_workflow(
                Day5WorkflowConfig(
                    base_run_dir=day4["artifacts"]["run_dir"],
                    candidate_root=self.config.candidate_root,
                    output_dir=self.config.output_dir,
                    llm_mode=self.config.llm_mode,
                    smoke_timeout_seconds=self.config.smoke_timeout_seconds,
                )
            )
            self._child_completed("day5", day5)

            # ── 阶段 3：公平实验与晋升（Day 6 工作流，条件执行）──────────
            # 输入：day4 的 run 目录（基线）+ day5 的候选目录
            # 作用：成对调参 → GT 隔离的公平评估 → 确定性 Judge → 接受则晋升 / 拒绝则带反馈改进
            # 输出：day6 state（rounds、accepted、promotion）；候选验证失败时跳过，day6 保持 None
            if day5["validation"]["eligible_for_training"]:
                self.state["stage"] = "FAIR_EVALUATION"
                self._save()
                day6 = run_day6_workflow(
                    Day6WorkflowConfig(
                        base_run_dir=day4["artifacts"]["run_dir"],
                        initial_candidate_dir=day5["artifacts"]["candidate_dir"],
                        candidate_root=self.config.candidate_root,
                        approved_root=self.config.approved_root,
                        output_dir=self.config.output_dir,
                        llm_mode=self.config.llm_mode,
                        tuning_trials=self.config.tuning_trials,
                        max_steps=self.config.fair_max_steps,
                        max_improvement_rounds=self.config.max_improvement_rounds,
                        validation_ratio=self.config.validation_ratio,
                        validation_interval=self.config.validation_interval,
                        patience=self.config.patience,
                        device=self.config.device,
                        minimum_psnr_delta=self.config.minimum_psnr_delta,
                        ssim_tolerance=self.config.ssim_tolerance,
                        smoke_timeout_seconds=self.config.smoke_timeout_seconds,
                    )
                )
                self._child_completed("day6", day6)

            # ── 阶段 4：汇总 → 选冠军 → 出报告 ──────────────────────────
            # 作用：标准化所有方法结果 → 在 eligible 方法中选 Missing PSNR 最高者 → 复制冠军图 → 生成 report.md
            self.state["method_results"] = build_method_results(day4, day6)
            eligible = [
                item
                for item in self.state["method_results"]
                if item["eligible_for_final_output"]
            ]
            winner = max(eligible, key=lambda item: _score(item["metrics"]))
            best_path = self.run_dir / "best_completion.png"
            shutil.copy2(winner["reconstruction"], best_path) #文件复制
            self.state["best_available"] = {
                **winner,
                "source_reconstruction": winner["reconstruction"],
                "reconstruction": str(best_path),
            }
            self.state["candidate_accepted"] = bool(day6 and day6["accepted"])
            self.state["artifacts"]["best_completion"] = str(best_path)
            self.state["stage"] = "COMPLETED"
            self._save()
            write_research_report(
                str(self.report_path), self.state, day4, day5, day6
            )
            self.trace.log_event(
                "session_end",
                {
                    "run_id": self.run_id,
                    "stage": "COMPLETED",
                    "candidate_accepted": self.state["candidate_accepted"],
                    "best_algorithm": winner["algorithm"],
                },
            )
            return self.state
        except Exception as error:
            self.state["stage"] = "FAILED"
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


def run_full_workflow(config: FullWorkflowConfig) -> Dict[str, Any]:
    return FullResearchWorkflow(config).run()
