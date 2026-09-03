# CP Decomposition

## Mathematical form

Represent the RGB tensor as a sum of rank-one components:
`X[i,j,c] ≈ Σ_r A[i,r] B[j,r] C[c,r]`. The height, width and channel factors,
plus a channel bias, are learned as `nn.Parameter` objects.

## Parameter characteristics

The dominant parameter count is `R(H + W + C)`, normally the smallest of the
three baselines. CP therefore provides a useful parameter-efficient option when
memory, tuning time or deployment size is constrained.

## Strengths

- Preserves the three image modes explicitly.
- Parameter-efficient and suitable for tight budgets.
- Shared rank-one components can capture repeated global patterns.
- Works well when the tensor is close to a genuinely low CP rank.

## Limitations

- A single shared rank controls all modes and can be too restrictive.
- Factor scaling is non-identifiable and optimization can be sensitive.
- RGB has only three channels, so a large nominal CP rank does not guarantee
  useful additional channel structure.
- Local edges and irregular textures are not explicit priors.

## Missing-pattern experience

CP is often competitive for random missingness and repeated separable patterns.
For one large block, every rank-one component must extrapolate into the same
unobserved area; this may create smooth colour fields or periodic banding.

## Spatial and channel experience

Prefer CP when parameter budget is tight and spatial structure appears
separable. Moderate channel correlation is acceptable. If RGB correlation is
very high but height and width need different capacities, Tucker is usually
more flexible because it can use a small channel rank and unequal spatial ranks.

## Recommended ranks

Start with `R ∈ {4, 8, 12, 16}`. Expand only when held-out observed MSE improves
under the same training budget. CP ranks should not be compared to Tucker ranks
as though they implied the same parameter count.

## Common failure modes

- Factors grow or shrink reciprocally while reconstruction changes little.
- Repeated stripes caused by separable rank-one components.
- Low validation error near observations but weak extrapolation into block centres.
- Increasing rank adds optimization difficulty without improving hidden regions.
