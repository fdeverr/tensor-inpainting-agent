"""Command-line entry point for Day 4 method selection."""

from __future__ import annotations

import argparse

from .workflow_day4 import Day4WorkflowConfig, run_day4_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve tensor knowledge, select a method, and run it."
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
    parser.add_argument(
        "--llm-mode",
        choices=("auto", "off", "required"),
        default="auto",
        help="auto falls back to deterministic rules when LLM env vars are absent",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    return parser


def _metric_text(value):
    return "infinite (perfect)" if value is None else "%.4f dB" % value


def main() -> None:
    args = build_parser().parse_args()
    state = run_day4_workflow(
        Day4WorkflowConfig(
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
            llm_mode=args.llm_mode,
            retrieval_top_k=args.retrieval_top_k,
        )
    )
    plan = state["results"]["method_plan"]
    diagnostics = state["results"]["selector_diagnostics"]
    interpolation = state["results"]["interpolation_metrics"]
    tensor = state["results"]["tensor_metrics"]
    comparison = state["results"]["comparison"]
    print("Day 4 method-selection workflow completed")
    print("  run_id: %s" % state["run_id"])
    print("  selected_model: %s" % plan["method"])
    print("  selection_mode: %s" % plan["selection_mode"])
    print("  confidence: %.4f" % plan["confidence"])
    print("  fallback_reason: %s" % diagnostics["fallback_reason"])
    print("  interpolation_psnr: %s" % _metric_text(interpolation["missing_psnr"]))
    print("  tensor_psnr: %s" % _metric_text(tensor["missing_psnr"]))
    print("  winner: %s" % comparison["winner"])
    print("  method_plan: %s" % state["artifacts"]["method_plan"])
    print("  state: %s" % state["artifacts"]["state"])


if __name__ == "__main__":
    main()
