# Inpainting Research Agent

This directory contains the incremental implementation of the seven-day
Inpainting Research Agent project.

## Day 1: trustworthy interpolation baseline

Day 1 deliberately contains no LLM and no Agent loop. It establishes the
deterministic experiment contract that later tools must obey:

- mask convention: `True/white = observed`, `False/black = missing`;
- the reconstruction algorithm sees only the corrupted image and mask;
- ground truth is used only by final metric functions;
- every stochastic operation takes an explicit seed;
- every run writes its config, images, mask, metrics, and run ID.

The accompanying Chinese learning notes are in [`DAY1_LEARNING.md`](DAY1_LEARNING.md).

Run it with any complete RGB image:

```bash
python3 -m inpainting_research_agent.run_day1 \
  --image /absolute/path/to/image.png \
  --mask-type block \
  --missing-rate 0.4 \
  --seed 42
```

Outputs are written to `outputs/<run_id>/`.

### Metrics

`missing_region_psnr` is calculated only on synthetically hidden pixels.

`composite_ssim` first replaces observed pixels in the prediction with ground
truth, then computes Gaussian-window RGB SSIM. It is useful for this project,
but it is not a standardized masked SSIM, and observed regions can still dilute
the score.

## Day 2: learnable tensor baselines

Day 2 adds Matrix Factorization, CP, and Tucker models whose factors are learned
as `nn.Parameter` objects through one fixed Trainer. Ground truth from the
artificially missing region is reserved for final evaluation only.

```bash
python3 -m inpainting_research_agent.run_day2 \
  --image inpainting_research_agent/assets/example.png \
  --model tucker \
  --rank-h 16 \
  --rank-w 16 \
  --rank-c 3 \
  --device auto
```

See [`DAY2_LEARNING.md`](DAY2_LEARNING.md) for the model equations, experiment
protocol, current baseline results, and exercises.
