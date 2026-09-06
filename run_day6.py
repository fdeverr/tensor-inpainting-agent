"""Command-line entry point for Day 6 fair evaluation and improvement."""

from __future__ import annotations

import argparse

from .workflow_day6 import Day6WorkflowConfig, run_day6_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fairly evaluate a candidate and run at most two improvement rounds."
    )
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--candidate-root", default="inpainting_research_agent/algorithms/candidates")
    parser.add_argument("--approved-root", default="inpainting_research_agent/algorithms/approved")
    parser.add_argument("--output-dir", default="inpainting_research_agent/outputs")
    parser.add_argument("--llm-mode", choices=("auto", "off", "required"), default="auto")
    parser.add_argument("--tuning-trials", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-improvement-rounds", type=int, default=2)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--minimum-psnr-delta", type=float, default=0.2)
    parser.add_argument("--ssim-tolerance", type=float, default=0.002)
    parser.add_argument("--smoke-timeout", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = run_day6_workflow(
        Day6WorkflowConfig(
            base_run_dir=args.base_run_dir,
            initial_candidate_dir=args.candidate_dir,
            candidate_root=args.candidate_root,
            approved_root=args.approved_root,
            output_dir=args.output_dir,
            llm_mode=args.llm_mode,
            tuning_trials=args.tuning_trials,
            max_steps=args.max_steps,
            max_improvement_rounds=args.max_improvement_rounds,
            validation_ratio=args.validation_ratio,
            validation_interval=args.validation_interval,
            patience=args.patience,
            device=args.device,
            minimum_psnr_delta=args.minimum_psnr_delta,
            ssim_tolerance=args.ssim_tolerance,
            smoke_timeout_seconds=args.smoke_timeout,
        )
    )
    print("Day 6 fair experiment completed")
    print("  workflow_id: %s" % state["workflow_id"])
    print("  rounds: %d" % len(state["rounds"]))
    print("  accepted: %s" % state["accepted"])
    print("  stop_reason: %s" % state["stop_reason"])
    print("  best_algorithm: %s" % state["best_available"]["algorithm"])
    print("  reconstruction: %s" % state["best_available"]["reconstruction"])


if __name__ == "__main__":
    main()
