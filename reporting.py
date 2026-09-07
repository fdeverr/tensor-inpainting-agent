"""Markdown reporting for a complete, auditable research run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def _number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return ("%%.%df" % digits) % float(value)


def build_method_results(
    day4: Dict[str, Any],
    day6: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize all evaluated methods into one report-friendly table."""

    results = [
        {
            "algorithm": "nearest_neighbor_manhattan",
            "role": "interpolation_baseline",
            "metrics": day4["results"]["interpolation_metrics"],
            "runtime_seconds": None,
            "parameter_count": 0,
            "eligible_for_final_output": True,
            "reconstruction": day4["artifacts"]["interpolation"],
        }
    ]
    if day6 is None:
        results.append(
            {
                "algorithm": day4["selected_model"],
                "role": "tensor_baseline",
                "metrics": day4["results"]["tensor_metrics"],
                "runtime_seconds": day4["results"]["training"]["runtime_seconds"],
                "parameter_count": day4["results"]["training"]["parameter_count"],
                "eligible_for_final_output": True,
                "reconstruction": day4["artifacts"]["tensor_reconstruction"],
            }
        )
        return results

    last_round = day6["rounds"][-1]
    baseline = last_round["baseline_final"]
    candidate = last_round["candidate_final"]
    accepted = last_round["judgment"]["accepted"]
    results.extend(
        [
            {
                "algorithm": day4["selected_model"],
                "role": "tensor_baseline",
                "metrics": baseline["metrics"],
                "runtime_seconds": baseline["runtime_seconds"],
                "parameter_count": baseline["parameter_count"],
                "eligible_for_final_output": True,
                "reconstruction": baseline["artifacts"]["reconstruction"],
            },
            {
                "algorithm": (
                    day6["promotion"]["algorithm_name"]
                    if accepted
                    else last_round["candidate_id"]
                ),
                "role": "candidate",
                "metrics": candidate["metrics"],
                "runtime_seconds": candidate["runtime_seconds"],
                "parameter_count": candidate["parameter_count"],
                "eligible_for_final_output": accepted,
                "reconstruction": candidate["artifacts"]["reconstruction"],
            },
        ]
    )
    return results


def render_research_report(
    final_state: Dict[str, Any],
    day4: Dict[str, Any],
    day5: Dict[str, Any],
    day6: Optional[Dict[str, Any]],
) -> str:
    """Render facts from persisted states; do not invent experimental claims."""

    profile = day4["results"]["image_profile"]
    plan = day4["results"]["method_plan"]
    lines = [
        "# Inpainting Research Agent 实验报告",
        "",
        "## 结论摘要",
        "",
        "- Run ID：`%s`" % final_state["run_id"],
        "- 输入任务：%s" % final_state["prompt"],
        "- 选择的张量分解：`%s`" % day4["selected_model"],
        "- 候选代码验证：`%s`" % day5["validation"]["status"],
        "- 候选实验结论：`%s`"
        % (day6["rounds"][-1]["judgment"]["decision"] if day6 else "not_evaluated"),
        "- 最终输出算法：`%s`" % final_state["best_available"]["algorithm"],
        "- 最终补全图：`%s`" % final_state["artifacts"]["best_completion"],
        "",
        "## 数据与缺失模式",
        "",
        "| 项目 | 值 |",
        "|---|---:|",
        "| 图像尺寸 | `%s` |" % profile["image_shape"],
        "| Mask 类型 | `%s` |" % profile["mask_type"],
        "| 实际缺失率 | %s |" % _number(profile["actual_missing_rate"]),
        "| 缺失连通区域数 | %d |" % profile["missing_component_count"],
        "| 可见像素局部平滑度 | %s |"
        % _number(profile["visible_local_smoothness_score"]),
        "",
        "## 方法选择",
        "",
        plan["reason"],
        "",
        "证据：",
        "",
    ]
    lines.extend(
        "- `%s`：%s" % (evidence["source"], evidence["claim"])
        for evidence in plan["evidence"]
    )
    lines.extend(
        [
            "",
            "## 候选研究假设",
            "",
            day5["generation"]["hypothesis"],
            "",
            "代码必须先通过 AST 策略检查，以及独立进程中的 forward、backward 和一步优化检查。",
            "",
            "## 最终指标",
            "",
            "| 方法 | 角色 | Missing PSNR ↑ | Composite SSIM ↑ | 最终拟合时间 | 参数量 | 可作为最终输出 |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in final_state["method_results"]:
        metrics = item["metrics"]
        runtime = (
            "N/A"
            if item["runtime_seconds"] is None
            else _number(item["runtime_seconds"]) + " s"
        )
        lines.append(
            "| %s | %s | %s | %s | %s | %d | %s |"
            % (
                item["algorithm"],
                item["role"],
                _number(metrics["missing_psnr"]),
                _number(metrics["composite_ssim"]),
                runtime,
                item["parameter_count"],
                "是" if item["eligible_for_final_output"] else "否",
            )
        )
    if day6:
        judgment = day6["rounds"][-1]["judgment"]
        lines.extend(
            [
                "",
                "## 公平实验与 Judge",
                "",
                "- 基础模型 trials：%d" % judgment["budget_audit"]["baseline_trial_count"],
                "- 候选模型 trials：%d" % judgment["budget_audit"]["candidate_trial_count"],
                "- Missing PSNR 差值：%s dB" % _number(judgment["psnr_delta"]),
                "- Composite SSIM 差值：%s" % _number(judgment["ssim_delta"]),
                "- 总运行时间比：%s" % _number(judgment["runtime_ratio"]),
                "- 决策：`%s`" % judgment["decision"],
                "- 停止原因：`%s`" % day6["stop_reason"],
                "",
                "调参函数不接收缺失区域 Ground Truth。超参数和训练步数确定后，模型才在全部观测像素上重新拟合并执行最终隐藏区域评估。",
            ]
        )
    lines.extend(
        [
            "",
            "## 可审计产物",
            "",
            "- 方法选择状态：`%s`" % final_state["artifacts"]["day4_state"],
            "- 候选生成状态：`%s`" % final_state["artifacts"]["day5_state"],
            "- 公平实验状态：`%s`"
            % final_state["artifacts"].get("day6_state", "not generated"),
            "- 顶层 Trace：`%s`" % final_state["artifacts"]["trace_jsonl"],
            "",
            "## 证据边界",
            "",
            "本报告只证明该工作流在当前图片、mask、缺失率、随机种子和预算下得到上述结果。单张图片上的晋升不能推出跨数据集优势，也不构成 SOTA 声明。",
            "",
        ]
    )
    return "\n".join(lines)


def write_research_report(path: str, *args: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_research_report(*args), encoding="utf-8")
