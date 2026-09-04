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
artificially missing region is reserved for final evaluation only. Training is
two-stage: a train/validation split selects `best_step`, then the model is reset
and fitted for that many steps using every observed pixel.

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

## Day 3: Hello Agents tools and deterministic workflow

Day 3 wraps the experiment core in six native Hello Agents `Tool` classes and
executes them through `ToolRegistry`. A persisted state machine fixes the legal
order, while `TraceLogger` writes JSONL and HTML audit trails. No LLM is used
yet, and only the final evaluation tool receives ground truth.

```bash
python3 -m inpainting_research_agent.run_day3 \
  --image inpainting_research_agent/assets/example.png \
  --mask-type block \
  --missing-rate 0.4 \
  --max-steps 200 \
  --device auto
```

See [`DAY3_LEARNING.md`](DAY3_LEARNING.md) for the tool contracts, state
transitions, trace format, real run results, and exercises.

## Day 4: retrieval-augmented method selection

Day 4 enriches the visible-only image profile, retrieves sourced evidence from
local Matrix/CP/Tucker knowledge documents, and emits a Pydantic-validated
`MethodPlan`. An optional Hello Agents LLM gets one JSON repair attempt; missing
or repeatedly invalid LLM output falls back to deterministic weighted rules.

```bash
python3 -m inpainting_research_agent.run_day4 \
  --image inpainting_research_agent/assets/example.png \
  --mask-type block \
  --missing-rate 0.4 \
  --llm-mode auto \
  --device auto
```

See [`DAY4_LEARNING.md`](DAY4_LEARNING.md) for ImageProfile definitions,
retrieval logic, MethodPlan validation, LLM configuration, failure recovery,
real results, and exercises.

## Day 5: constrained candidate generation and validation

Day 5 feeds a completed Day 4 run into a schema-constrained Model Improver. It
stores a hypothesis, bounded search space, complete candidate class, manifest,
and structured validation. AST policy checks run before an isolated-process
forward/backward/optimizer smoke test. Only candidates passing both checks are
marked `eligible_for_training`.

```bash
python3 -m inpainting_research_agent.run_day5 \
  --base-run-dir inpainting_research_agent/outputs/<day4-run-id> \
  --llm-mode auto \
  --smoke-timeout 10
```

See [`DAY5_LEARNING.md`](DAY5_LEARNING.md) for the proposal contract, TV
candidate hypothesis, AST policy, smoke-test stages, safety limitations,
manifest gate, real validation result, and exercises.
