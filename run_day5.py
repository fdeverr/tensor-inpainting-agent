"""Command-line entry point for Day 5 candidate generation and validation."""

from __future__ import annotations

import argparse

from .workflow_day5 import Day5WorkflowConfig, run_day5_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and validate one constrained tensor-model candidate."
    )
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument(
        "--candidate-root",
        default="inpainting_research_agent/algorithms/candidates",
    )
    parser.add_argument("--output-dir", default="inpainting_research_agent/outputs")
    parser.add_argument(
        "--llm-mode",
        choices=("auto", "off", "required"),
        default="auto",
    )
    parser.add_argument("--smoke-timeout", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = run_day5_workflow(
        Day5WorkflowConfig(
            base_run_dir=args.base_run_dir,
            candidate_root=args.candidate_root,
            output_dir=args.output_dir,
            llm_mode=args.llm_mode,
            smoke_timeout_seconds=args.smoke_timeout,
        )
    )
    print("Day 5 candidate workflow completed")
    print("  workflow_id: %s" % state["workflow_id"])
    print("  candidate_id: %s" % state["candidate_id"])
    print("  stage: %s" % state["stage"])
    print("  generation_mode: %s" % state["generation"]["mode"])
    print("  validation_passed: %s" % state["validation"]["passed"])
    print("  eligible_for_training: %s" % state["validation"]["eligible_for_training"])
    print("  candidate_dir: %s" % state["artifacts"]["candidate_dir"])
    print("  validation: %s" % state["artifacts"]["validation"])


if __name__ == "__main__":
    main()
