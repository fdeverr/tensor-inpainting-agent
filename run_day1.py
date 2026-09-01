"""Command-line entry point for the Day 1 baseline."""

from __future__ import annotations

import argparse

from .core.pipeline import run_day1_baseline
from .schemas import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the reproducible Day 1 interpolation baseline."
    )
    parser.add_argument("--image", required=True, help="Path to a complete benchmark image")
    parser.add_argument(
        "--output-dir",
        default="inpainting_research_agent/outputs",
        help="Directory that will contain one subdirectory per run",
    )
    parser.add_argument(
        "--mask-type",
        choices=("random", "block"),
        default="block",
    )
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Resize the largest side to this value; use 0 to keep source size",
    )
    parser.add_argument("--missing-fill-value", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        image_path=args.image,
        output_dir=args.output_dir,
        mask_type=args.mask_type,
        missing_rate=args.missing_rate,
        seed=args.seed,
        image_size=args.image_size or None,
        missing_fill_value=args.missing_fill_value,
    )
    result = run_day1_baseline(config)

    psnr_text = "infinite (perfect)" if result.missing_psnr is None else "%.4f dB" % result.missing_psnr
    print("Day 1 interpolation baseline completed")
    print("  run_id: %s" % result.run_id)
    print("  actual_missing_rate: %.4f" % result.actual_missing_rate)
    print("  missing_region_psnr: %s" % psnr_text)
    print("  composite_ssim: %.6f" % result.composite_ssim)
    print("  artifacts: %s" % result.artifacts["metrics"])


if __name__ == "__main__":
    main()
