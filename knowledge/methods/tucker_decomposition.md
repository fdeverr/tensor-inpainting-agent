# Tucker Decomposition

## Mathematical form

Learn `X ≈ G ×₁ A ×₂ B ×₃ C`, where `G` is a core tensor and `A`, `B`, `C`
are height, width and channel factor matrices. The core, factors and channel
bias are all `nn.Parameter` objects.

## Parameter characteristics

The dominant count is `H R_h + W R_w + C R_c + R_h R_w R_c`. Tucker is less
compact than CP but permits different ranks for each mode. For RGB images,
`R_c` is normally 1, 2 or 3.

## Strengths

- Separate height, width and channel ranks give flexible capacity allocation.
- A small channel rank directly exploits correlated RGB channels.
- The core models interactions that CP's matched rank-one components cannot.
- Often the most expressive baseline for anisotropic or globally correlated images.

## Limitations

- More parameters and a larger tuning space than CP.
- Core size grows multiplicatively with the selected ranks.
- Still a global low-rank model without explicit local smoothness or edge priors.
- Can overfit held-out random pixels while extrapolating poorly into a large hole.

## Missing-pattern experience

Tucker is a reasonable first choice for a large contiguous block when global
spatial and colour correlations are strong, but it is not guaranteed to beat
local interpolation. For random missingness, simpler Matrix or CP may achieve a
similar result at lower cost.

## Spatial and channel experience

Prefer Tucker when RGB channels are strongly correlated, image aspect ratio is
large, or horizontal and vertical complexity differ. Use `R_h != R_w` for clear
anisotropy. Smooth global structure supports small spatial ranks; high-frequency
texture requires higher ranks or a later spatial regularization idea.

## Recommended ranks

For 64–256 pixel development images, begin with spatial ranks in `{4, 8, 16,
24}` and channel rank in `{1, 2, 3}`. A safe initial grid is `(8,8,2)`,
`(8,8,3)`, `(16,16,3)`, with asymmetric variants when the aspect ratio is high.

## Common failure modes

- Core over-parameterization reduces validation performance.
- Block interiors contain global bands instead of local edges.
- High spatial ranks fit observed texture but do not improve hole extrapolation.
- Validation based on random observed pixels selects a configuration unsuitable
  for a contiguous missing component.
