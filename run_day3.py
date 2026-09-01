"""Command-line entry point for the deterministic Day 3 Agent workflow."""

from __future__ import annotations

import argparse

from .workflow import Day3WorkflowConfig, run_day3_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the tool-based, no-LLM Day 3 research workflow."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="inpainting_research_agent/outputs")
    parser.add_argument("--mask-type", choices=("random", "block"), default="block")
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = run_day3_workflow(
        Day3WorkflowConfig(
            image_path=args.image,
            output_dir=args.output_dir,
            mask_type=args.mask_type,
            missing_rate=args.missing_rate,
            seed=args.seed,
            image_size=args.image_size or None,
            max_steps=args.max_steps,
            validation_ratio=args.validation_ratio,
            validation_interval=args.validation_interval,
            patience=args.patience,
            device=args.device,
        )
    )
    comparison = state["results"]["comparison"]
    interpolation = state["results"]["interpolation_metrics"]
    tensor = state["results"]["tensor_metrics"]
    print("Day 3 tool workflow completed")
    print("  run_id: %s" % state["run_id"])
    print("  stage: %s" % state["stage"])
    print("  selected_model: %s" % state["selected_model"])
    interpolation_psnr = interpolation["missing_psnr"]
    tensor_psnr = tensor["missing_psnr"]
    print(
        "  interpolation_psnr: %s"
        % (
            "infinite (perfect)"
            if interpolation_psnr is None
            else "%.4f dB" % interpolation_psnr
        )
    )
    print(
        "  tensor_psnr: %s"
        % (
            "infinite (perfect)"
            if tensor_psnr is None
            else "%.4f dB" % tensor_psnr
        )
    )
    print("  winner: %s" % comparison["winner"])
    print("  state: %s" % state["artifacts"]["state"])
    print("  trace: %s" % state["artifacts"]["trace_jsonl"])


if __name__ == "__main__":
    main()
