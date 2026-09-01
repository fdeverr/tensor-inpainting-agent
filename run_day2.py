"""Command-line entry point for Day 2 tensor-model experiments."""

from __future__ import annotations

import argparse

from .core.day2_pipeline import run_day2_experiment
from .core.models import get_default_hyperparameters
from .schemas import Day2ExperimentConfig, TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a basic learnable tensor decomposition."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", choices=("matrix", "cp", "tucker"), required=True)
    parser.add_argument("--output-dir", default="inpainting_research_agent/outputs")
    parser.add_argument("--mask-type", choices=("random", "block"), default="block")
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--rank-h", type=int)
    parser.add_argument("--rank-w", type=int)
    parser.add_argument("--rank-c", type=int)
    parser.add_argument("--init-scale", type=float)
    return parser


def _model_hyperparameters(args: argparse.Namespace):
    hyperparameters = get_default_hyperparameters(args.model)
    if args.model in {"matrix", "cp"} and args.rank is not None:
        hyperparameters["rank"] = args.rank
    if args.model == "tucker":
        if args.rank_h is not None:
            hyperparameters["rank_h"] = args.rank_h
        if args.rank_w is not None:
            hyperparameters["rank_w"] = args.rank_w
        if args.rank_c is not None:
            hyperparameters["rank_c"] = args.rank_c
    if args.init_scale is not None:
        hyperparameters["init_scale"] = args.init_scale
    return hyperparameters


def main() -> None:
    args = build_parser().parse_args()
    training = TrainingConfig(
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        validation_observed_ratio=args.validation_ratio,
        validation_interval=args.validation_interval,
        early_stopping_patience=args.patience,
        device=args.device,
    )
    config = Day2ExperimentConfig(
        image_path=args.image,
        model_name=args.model,
        model_hyperparameters=_model_hyperparameters(args),
        training=training,
        output_dir=args.output_dir,
        mask_type=args.mask_type,
        missing_rate=args.missing_rate,
        seed=args.seed,
        image_size=args.image_size or None,
    )
    result = run_day2_experiment(config)
    interpolation = result["interpolation"]
    tensor_model = result["tensor_model"]
    interpolation_psnr = interpolation["missing_psnr"]
    tensor_psnr = tensor_model["missing_psnr"]
    interpolation_psnr_text = (
        "infinite (perfect)"
        if interpolation_psnr is None
        else "%.4f dB" % interpolation_psnr
    )
    tensor_psnr_text = (
        "infinite (perfect)" if tensor_psnr is None else "%.4f dB" % tensor_psnr
    )

    print("Day 2 tensor experiment completed")
    print("  run_id: %s" % result["run_id"])
    print("  model: %s" % result["model_name"])
    print("  device: %s" % result["training"]["device"])
    print("  best_step: %d" % result["training"]["best_step"])
    print("  interpolation_psnr: %s" % interpolation_psnr_text)
    print("  tensor_model_psnr: %s" % tensor_psnr_text)
    print("  tensor_model_ssim: %.6f" % tensor_model["composite_ssim"])
    print("  metrics: %s" % result["artifacts"]["metrics"])


if __name__ == "__main__":
    main()
