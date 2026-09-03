# Matrix Factorization

## Mathematical form

Flatten an RGB image `X ∈ R^(H×W×C)` into `X_flat ∈ R^(H×WC)` and learn
`X_flat ≈ U V`, where `U ∈ R^(H×R)` and `V ∈ R^(R×WC)`. Both factors and the
RGB channel bias are `nn.Parameter` objects.

## Parameter characteristics

The dominant parameter count is `R(H + WC)`. It grows directly with image
width and couples width with colour. It is easy to implement and debug, but is
usually less parameter-efficient than CP at comparable nominal rank.

## Strengths

- Strong, transparent baseline with only two factor matrices.
- Useful when the unfolded image has clear global low-rank row/column structure.
- Often stable on smooth images and random missing pixels.
- A good diagnostic choice when simplicity and interpretability matter most.

## Limitations

- Flattening width and channel loses an explicit three-mode inductive bias.
- Cannot choose separate height, width and channel ranks.
- Large contiguous holes may expose stripes aligned with the unfolding.
- High-frequency textures and local edges are poorly represented at low rank.

## Missing-pattern experience

Random missing pixels are usually easier because each row and unfolded column
retains observations. A large block removes coordinated information and makes
the matrix completion assumption harder to satisfy. Prefer Matrix for random
or dispersed missingness; use caution when one component dominates the image.

## Spatial and channel experience

Matrix is reasonable for a nearly square, globally smooth image with moderate
channel correlation. Very high RGB correlation can be represented, but Tucker
expresses it more directly with a small channel rank. Highly asymmetric spatial
structure can also favour Tucker's separate spatial ranks.

## Recommended ranks

Start with `R ∈ {4, 8, 16, 32}` and cap the rank below `min(H, WC)`. Use smaller
ranks for very smooth images and increase cautiously when validation error
underfits. Rank is selected only on held-out observed pixels.

## Common failure modes

- Horizontal or vertical banding in a block hole.
- Rank too small: colour averages are plausible but structure disappears.
- Rank too large: validation error worsens while training error keeps falling.
- A learning rate that is too high produces unstable factor scale.
