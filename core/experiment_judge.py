"""Deterministic acceptance rules and structured research feedback."""

from __future__ import annotations

from typing import Any, Dict, List


def _total_runtime(tuning: Dict[str, Any], final: Dict[str, Any]) -> float:
    return float(sum(item["runtime_seconds"] for item in tuning["trials"])) + float(
        final["runtime_seconds"]
    )


def judge_candidate(
    baseline_tuning: Dict[str, Any],
    baseline_final: Dict[str, Any],
    candidate_tuning: Dict[str, Any],
    candidate_final: Dict[str, Any],
    minimum_psnr_delta: float = 0.2,
    ssim_tolerance: float = 0.002,
) -> Dict[str, Any]:
    """Apply the fixed promotion gate; the LLM never decides acceptance."""

    baseline_psnr = baseline_final["metrics"]["missing_psnr"]
    candidate_psnr = candidate_final["metrics"]["missing_psnr"]
    psnr_delta = (
        None
        if baseline_psnr is None or candidate_psnr is None
        else float(candidate_psnr - baseline_psnr)
    )
    ssim_delta = float(
        candidate_final["metrics"]["composite_ssim"]
        - baseline_final["metrics"]["composite_ssim"]
    )
    baseline_runtime = _total_runtime(baseline_tuning, baseline_final)
    candidate_runtime = _total_runtime(candidate_tuning, candidate_final)
    runtime_ratio = candidate_runtime / baseline_runtime if baseline_runtime > 0.0 else None

    failures: List[str] = []
    if psnr_delta is None or psnr_delta < minimum_psnr_delta:
        failures.append("missing-region PSNR improvement is below the promotion threshold")
    if ssim_delta < -ssim_tolerance:
        failures.append("composite SSIM regression exceeds the allowed tolerance")
    if baseline_tuning["trial_count"] != candidate_tuning["trial_count"]:
        failures.append("baseline and candidate received unequal tuning trial counts")

    accepted = not failures
    selected_candidate = candidate_tuning["best"]["hyperparameters"]
    selected_weight = selected_candidate.get("tv_weight")
    suspected_causes = []
    constraints = []
    if not accepted:
        constraints.extend(
            [
                "Keep the same trial count, max steps, seed, split, device, and judge thresholds.",
                "Do not use missing-region ground truth in generation, tuning, or checkpoint selection.",
            ]
        )
    if not accepted and selected_weight == 0.0:
        suspected_causes.append(
            "Validation selected tv_weight=0, so the proposed regularizer added no useful signal."
        )
        constraints.append("Try a smaller non-zero regularization range or a different bounded prior.")
    elif not accepted and psnr_delta is not None and psnr_delta < 0.0:
        suspected_causes.append(
            "The selected spatial prior may oversmooth edges or optimize visible pixels without extrapolating into the hole."
        )
        constraints.append("Reduce the regularization scale and preserve the tensor-decomposition core.")
    if not accepted and ssim_delta < 0.0:
        suspected_causes.append(
            "The candidate reduced structural similarity, consistent with excessive smoothing or rank mismatch."
        )
    if not accepted and not suspected_causes:
        suspected_causes.append(
            "The gain was positive but too small to distinguish it from a practically negligible change."
        )

    return {
        "decision": "accept" if accepted else "reject",
        "accepted": accepted,
        "psnr_delta": psnr_delta,
        "ssim_delta": ssim_delta,
        "runtime_ratio": runtime_ratio,
        "thresholds": {
            "minimum_psnr_delta": minimum_psnr_delta,
            "maximum_ssim_drop": ssim_tolerance,
        },
        "gate_failures": failures,
        "training_behavior": {
            "baseline_best_validation_mse": baseline_tuning["best"]["best_validation_mse"],
            "candidate_best_validation_mse": candidate_tuning["best"]["best_validation_mse"],
            "baseline_best_step": baseline_tuning["best"]["best_step"],
            "candidate_best_step": candidate_tuning["best"]["best_step"],
            "candidate_selected_hyperparameters": selected_candidate,
        },
        "suspected_causes": suspected_causes,
        "next_round_constraints": constraints,
        "budget_audit": {
            "baseline_trial_count": baseline_tuning["trial_count"],
            "candidate_trial_count": candidate_tuning["trial_count"],
            "baseline_total_runtime_seconds": baseline_runtime,
            "candidate_total_runtime_seconds": candidate_runtime,
        },
    }
