"""Small reproducible benchmark driver for complete research-agent cases."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .workflow import _make_run_id, _write_json
from .workflow_full import FullWorkflowConfig, run_full_workflow


@dataclass(frozen=True)
class BenchmarkConfig:
    image_paths: Sequence[str]
    mask_types: Sequence[str] = ("random", "block")
    missing_rates: Sequence[float] = (0.4,)
    output_dir: str = "inpainting_research_agent/outputs"
    candidate_root: str = "inpainting_research_agent/algorithms/candidates"
    approved_root: str = "inpainting_research_agent/algorithms/approved"
    seed: int = 42
    image_size: int = 64
    method_max_steps: int = 50
    fair_max_steps: int = 50
    tuning_trials: int = 2
    max_improvement_rounds: int = 1
    device: str = "auto"
    llm_mode: str = "off"

    def validate(self) -> None:
        if not self.image_paths:
            raise ValueError("at least one image is required")
        missing = [path for path in self.image_paths if not Path(path).is_file()]
        if missing:
            raise ValueError("benchmark images do not exist: %s" % missing)
        if not self.mask_types or any(item not in {"random", "block"} for item in self.mask_types):
            raise ValueError("mask_types may contain only random and block")
        if not self.missing_rates or any(not 0.0 < rate < 1.0 for rate in self.missing_rates):
            raise ValueError("missing_rates must be in (0, 1)")


def aggregate_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate successful normalized method rows by semantic role."""

    successful = [case for case in cases if case["status"] == "completed"]
    roles: Dict[str, List[Dict[str, Any]]] = {}
    for case in successful:
        for method in case["method_results"]:
            roles.setdefault(method["role"], []).append(method)
    methods = {}
    for role, rows in roles.items():
        psnr = [
            row["metrics"]["missing_psnr"]
            for row in rows
            if row["metrics"]["missing_psnr"] is not None
        ]
        ssim = [row["metrics"]["composite_ssim"] for row in rows]
        runtimes = [
            row["runtime_seconds"]
            for row in rows
            if row["runtime_seconds"] is not None
        ]
        parameters = [row["parameter_count"] for row in rows]
        methods[role] = {
            "evaluated_cases": len(rows),
            "unavailable_or_failed_cases": len(cases) - len(rows),
            "mean_missing_psnr": statistics.fmean(psnr) if psnr else None,
            "std_missing_psnr": (
                statistics.pstdev(psnr) if len(psnr) > 1 else 0.0 if psnr else None
            ),
            "perfect_reconstruction_cases": sum(
                row["metrics"]["missing_psnr"] is None for row in rows
            ),
            "mean_composite_ssim": statistics.fmean(ssim),
            "std_composite_ssim": statistics.pstdev(ssim) if len(ssim) > 1 else 0.0,
            "mean_runtime_seconds": statistics.fmean(runtimes) if runtimes else None,
            "mean_parameter_count": statistics.fmean(parameters),
        }
    return {
        "total_cases": len(cases),
        "successful_cases": len(successful),
        "failed_cases": len(cases) - len(successful),
        "candidate_acceptance_count": sum(
            bool(case.get("candidate_accepted")) for case in successful
        ),
        "methods_by_role": methods,
    }


def _benchmark_markdown(state: Dict[str, Any]) -> str:
    lines = [
        "# Inpainting Research Agent Benchmark",
        "",
        "## 范围",
        "",
        "- Cases：%d" % state["aggregate"]["total_cases"],
        "- 成功：%d" % state["aggregate"]["successful_cases"],
        "- 失败：%d" % state["aggregate"]["failed_cases"],
        "- 候选晋升次数：%d" % state["aggregate"]["candidate_acceptance_count"],
        "",
        "## 聚合结果",
        "",
        "| 角色 | 完成 | 未评估/失败 | PSNR mean ± std | SSIM mean ± std | 平均训练时间 | 平均参数量 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role, item in state["aggregate"]["methods_by_role"].items():
        runtime = (
            "N/A"
            if item["mean_runtime_seconds"] is None
            else "%.4f s" % item["mean_runtime_seconds"]
        )
        psnr_text = (
            "perfect"
            if item["mean_missing_psnr"] is None
            else "%.4f ± %.4f"
            % (item["mean_missing_psnr"], item["std_missing_psnr"])
        )
        lines.append(
            "| %s | %d | %d | %s | %.4f ± %.4f | %s | %.1f |"
            % (
                role,
                item["evaluated_cases"],
                item["unavailable_or_failed_cases"],
                psnr_text,
                item["mean_composite_ssim"],
                item["std_composite_ssim"],
                runtime,
                item["mean_parameter_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Case 明细",
            "",
            "| Image | Mask | Missing rate | 状态 | 最终算法 | 总时间 |",
            "|---|---|---:|---|---|---:|",
        ]
    )
    for case in state["cases"]:
        lines.append(
            "| %s | %s | %.2f | %s | %s | %.3f s |"
            % (
                case["image_path"],
                case["mask_type"],
                case["missing_rate"],
                case["status"],
                case.get("best_algorithm", "N/A"),
                case["wall_runtime_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "> 注意：少量图片上的 quick benchmark 只用于端到端回归，不足以支持泛化结论。",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    config.validate()
    benchmark_id = _make_run_id("benchmark")
    directory = Path(config.output_dir) / benchmark_id
    directory.mkdir(parents=True, exist_ok=False)
    cases = []
    case_index = 0
    for image_path in config.image_paths:
        for mask_type in config.mask_types:
            for missing_rate in config.missing_rates:
                started_at = time.perf_counter()
                case_seed = config.seed + case_index
                case_index += 1
                try:
                    result = run_full_workflow(
                        FullWorkflowConfig(
                            image_path=image_path,
                            output_dir=config.output_dir,
                            candidate_root=config.candidate_root,
                            approved_root=config.approved_root,
                            mask_type=mask_type,
                            missing_rate=missing_rate,
                            seed=case_seed,
                            image_size=config.image_size,
                            method_max_steps=config.method_max_steps,
                            fair_max_steps=config.fair_max_steps,
                            tuning_trials=config.tuning_trials,
                            max_improvement_rounds=config.max_improvement_rounds,
                            device=config.device,
                            llm_mode=config.llm_mode,
                        )
                    )
                    cases.append(
                        {
                            "image_path": image_path,
                            "mask_type": mask_type,
                            "missing_rate": missing_rate,
                            "seed": case_seed,
                            "status": "completed",
                            "run_id": result["run_id"],
                            "state_path": result["artifacts"]["state"],
                            "best_algorithm": result["best_available"]["algorithm"],
                            "candidate_accepted": result["candidate_accepted"],
                            "method_results": result["method_results"],
                            "wall_runtime_seconds": time.perf_counter() - started_at,
                        }
                    )
                except Exception as error:
                    cases.append(
                        {
                            "image_path": image_path,
                            "mask_type": mask_type,
                            "missing_rate": missing_rate,
                            "seed": case_seed,
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "wall_runtime_seconds": time.perf_counter() - started_at,
                        }
                    )
    state = {
        "benchmark_id": benchmark_id,
        "created_at": datetime.now().isoformat(),
        "config": asdict(config),
        "cases": cases,
        "aggregate": aggregate_cases(cases),
        "artifacts": {
            "state": str(directory / "benchmark.json"),
            "report": str(directory / "benchmark.md"),
        },
    }
    _write_json(directory / "benchmark.json", state)
    (directory / "benchmark.md").write_text(
        _benchmark_markdown(state), encoding="utf-8"
    )
    return state
