"""Command-line benchmark entry point."""

from __future__ import annotations

import argparse

from .benchmark import BenchmarkConfig, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small end-to-end benchmark grid.")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--mask-types", nargs="+", choices=("random", "block"), default=["random", "block"])
    parser.add_argument("--missing-rates", nargs="+", type=float, default=[0.4])
    parser.add_argument("--output-dir", default="inpainting_research_agent/outputs")
    parser.add_argument("--candidate-root", default="inpainting_research_agent/algorithms/candidates")
    parser.add_argument("--approved-root", default="inpainting_research_agent/algorithms/approved")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--method-max-steps", type=int, default=50)
    parser.add_argument("--fair-max-steps", type=int, default=50)
    parser.add_argument("--tuning-trials", type=int, default=2)
    parser.add_argument("--max-improvement-rounds", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--llm-mode", choices=("auto", "off", "required"), default="off")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = run_benchmark(
        BenchmarkConfig(
            image_paths=args.images,
            mask_types=args.mask_types,
            missing_rates=args.missing_rates,
            output_dir=args.output_dir,
            candidate_root=args.candidate_root,
            approved_root=args.approved_root,
            seed=args.seed,
            image_size=args.image_size,
            method_max_steps=args.method_max_steps,
            fair_max_steps=args.fair_max_steps,
            tuning_trials=args.tuning_trials,
            max_improvement_rounds=args.max_improvement_rounds,
            device=args.device,
            llm_mode=args.llm_mode,
        )
    )
    aggregate = state["aggregate"]
    print("Benchmark completed")
    print("  benchmark_id: %s" % state["benchmark_id"])
    print("  successful_cases: %d/%d" % (aggregate["successful_cases"], aggregate["total_cases"]))
    print("  candidate_acceptance_count: %d" % aggregate["candidate_acceptance_count"])
    print("  report: %s" % state["artifacts"]["report"])


if __name__ == "__main__":
    main()
