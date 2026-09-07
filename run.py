"""Public one-command entry point for the complete Inpainting Research Agent."""

from __future__ import annotations

import argparse

from .workflow_full import (
    DEFAULT_RESEARCH_PROMPT,
    FullWorkflowConfig,
    run_full_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run method selection, candidate generation, fair evaluation, and reporting."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default=DEFAULT_RESEARCH_PROMPT)
    parser.add_argument("--output-dir", default="inpainting_research_agent/outputs")
    parser.add_argument("--candidate-root", default="inpainting_research_agent/algorithms/candidates")
    parser.add_argument("--approved-root", default="inpainting_research_agent/algorithms/approved")
    parser.add_argument("--mask-type", choices=("random", "block"), default="block")
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--method-max-steps", type=int, default=200)
    parser.add_argument("--fair-max-steps", type=int, default=200)
    parser.add_argument("--tuning-trials", type=int, default=4)
    parser.add_argument("--max-improvement-rounds", type=int, default=2)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--llm-mode", choices=("auto", "off", "required"), default="auto")
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    parser.add_argument("--minimum-psnr-delta", type=float, default=0.2)
    parser.add_argument("--ssim-tolerance", type=float, default=0.002)
    parser.add_argument("--smoke-timeout", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = run_full_workflow(
        FullWorkflowConfig(
            image_path=args.image,
            prompt=args.prompt,
            output_dir=args.output_dir,
            candidate_root=args.candidate_root,
            approved_root=args.approved_root,
            mask_type=args.mask_type,
            missing_rate=args.missing_rate,
            seed=args.seed,
            image_size=args.image_size or None,
            method_max_steps=args.method_max_steps,
            fair_max_steps=args.fair_max_steps,
            tuning_trials=args.tuning_trials,
            max_improvement_rounds=args.max_improvement_rounds,
            validation_ratio=args.validation_ratio,
            validation_interval=args.validation_interval,
            patience=args.patience,
            device=args.device,
            llm_mode=args.llm_mode,
            retrieval_top_k=args.retrieval_top_k,
            minimum_psnr_delta=args.minimum_psnr_delta,
            ssim_tolerance=args.ssim_tolerance,
            smoke_timeout_seconds=args.smoke_timeout,
        )
    )
    print("Inpainting Research Agent completed")
    print("  run_id: %s" % state["run_id"])
    print("  candidate_accepted: %s" % state["candidate_accepted"])
    print("  best_algorithm: %s" % state["best_available"]["algorithm"])
    best_psnr = state["best_available"]["metrics"]["missing_psnr"]
    print(
        "  best_psnr: %s"
        % ("infinite (perfect)" if best_psnr is None else "%.4f dB" % best_psnr)
    )
    print("  best_ssim: %.4f" % state["best_available"]["metrics"]["composite_ssim"])
    print("  completion: %s" % state["artifacts"]["best_completion"])
    print("  report: %s" % state["artifacts"]["report"])
    print("  state: %s" % state["artifacts"]["state"])


if __name__ == "__main__":
    main()
