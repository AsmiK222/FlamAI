# Parametric Curve Parameter Recovery

Recover the unknown parameters **θ, M, X** of the parametric curve

```
x(t) = t*cos(θ) - e^(M|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t*sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```

given a cloud of `(x, y)` sample points taken from the curve for `t ∈ (6, 60)`.

**Given ranges:**
- `0° < θ < 50°`
- `-0.05 < M < 0.05`
- `0 < X < 100`

## Result

| Parameter | Value |
|---|---|
| θ | **30°** |
| M | **0.03** |
| X | **55** |

Fitted curve (Desmos):
```
(t*cos(30)-e^{0.03|t|}·sin(0.3t)sin(30)+55, 42+t*sin(30)+e^{0.03|t|}·sin(0.3t)cos(30))
```
domain `6 ≤ t ≤ 60` — view it live: https://www.desmos.com/calculator/rfj91yrxob

**Validation:** sampling the fitted curve densely (2000 points over `t ∈ [6, 60]`) and
measuring the distance from every original data point to its nearest point on the curve:

- mean distance: **0.008**
- max distance: **0.024**

For reference, the data spans roughly 50 units in `x` and 24 units in `y`, so this is
an essentially exact fit.

## Approach

The 1500 points in `data/xy_data.csv` are **not ordered by `t`** — checking
`np.diff(x)` / `np.diff(y)` on consecutive rows shows no smooth progression, just
jumps back and forth. That rules out the naive shortcut of assuming
`t = linspace(6, 60, 1500)` in row order and fitting `(θ, M, X)` directly against
that guessed `t` (I tried this first — it gets stuck in a bad local optimum with `M`
pinned against its bound, since a wrong `t` assignment "explains" real error as noise).

Instead, `t` is treated as a **latent variable, unknown per point**, and the fit is
done as an **Orthogonal Distance Regression** via alternating minimization
(same idea as Iterative Closest Point):

1. **Freeze `(θ, M, X)`.** For every data point, find the `t ∈ [6, 60]` that puts the
   model curve closest to that point — coarse grid search over 2000 points, then a
   bounded 1-D refinement (`scipy.optimize.minimize_scalar`) around the best grid index.
2. **Freeze those per-point `t` estimates.** Re-fit `(θ, M, X)` against all 1500 points
   simultaneously via nonlinear least squares (`scipy.optimize.least_squares`), bounded
   to the given parameter ranges.
3. **Repeat** until the parameters stop changing (~30-40 iterations from a generic
   starting guess; a handful of iterations if started near the true values).

This converges to essentially exact, round-number parameters (θ=30°, M=0.03, X=55),
which is a good sign the recovery is correct rather than an overfit.

## Usage

```bash
pip install -r requirements.txt
python solve.py data/xy_data.csv
```

Outputs the per-iteration convergence trace, final `(θ, M, X)`, validation error, and
a ready-to-paste Desmos/LaTeX expression.

## Files

- `solve.py` — fitting script (see docstring for full method details)
- `data/xy_data.csv` — provided sample points
- `requirements.txt` — Python dependencies
