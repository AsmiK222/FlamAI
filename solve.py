"""
Recover the unknown parameters (theta, M, X) of the parametric curve

    x(t) = t*cos(theta) - e^(M*|t|) * sin(0.3t) * sin(theta) + X
    y(t) = 42 + t*sin(theta) + e^(M*|t|) * sin(0.3t) * cos(theta)

given an unordered cloud of (x, y) sample points taken from the curve
for t in (6, 60).

Approach
--------
The provided data points are NOT ordered by t (consecutive rows in the
CSV do not correspond to consecutive t values -- verified by checking
that np.diff(x), np.diff(y) jump around with no smooth pattern). That
rules out a naive "assume t = linspace(6, 60, N)" fit.

Instead this script treats t as a *latent* (unknown, per-point)
variable and performs an Orthogonal Distance Regression via alternating
minimization, similar in spirit to ICP (Iterative Closest Point):

  1. Freeze (theta, M, X). For every data point, find the t in [6, 60]
     that puts the model curve closest to that point (coarse grid
     search followed by a bounded 1-D refinement).
  2. Freeze those per-point t estimates. Re-fit (theta, M, X) via
     nonlinear least squares (scipy.optimize.least_squares) against all
     points simultaneously.
  3. Repeat until the parameters stop changing.

This converges quickly (a few iterations) to an essentially exact fit.

Usage
-----
    python solve.py [path/to/xy_data.csv]

Defaults to data/xy_data.csv if no path is given.
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

T_MIN, T_MAX = 6.0, 60.0
THETA_BOUNDS = (0.0, 50.0)     # degrees
M_BOUNDS = (-0.05, 0.05)
X_BOUNDS = (0.0, 100.0)


def curve_xy(t, theta_deg, M, X):
    """Evaluate the parametric curve at parameter value(s) t."""
    theta = np.radians(theta_deg)
    envelope = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - envelope * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + envelope * np.cos(theta)
    return x, y


def best_t_for_point(xi, yi, theta_deg, M, X, t_grid):
    """Find the t that minimizes distance from curve(t) to (xi, yi)."""
    xg, yg = curve_xy(t_grid, theta_deg, M, X)
    d2 = (xg - xi) ** 2 + (yg - yi) ** 2
    idx = np.argmin(d2)
    lo = t_grid[max(0, idx - 2)]
    hi = t_grid[min(len(t_grid) - 1, idx + 2)]

    def obj(t):
        xt, yt = curve_xy(t, theta_deg, M, X)
        return (xt - xi) ** 2 + (yt - yi) ** 2

    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
    return res.x


def assign_all_t(x_data, y_data, theta_deg, M, X, t_grid):
    n = len(x_data)
    ts = np.empty(n)
    for i in range(n):
        ts[i] = best_t_for_point(x_data[i], y_data[i], theta_deg, M, X, t_grid)
    return ts


def residuals_given_t(params, ts, x_data, y_data):
    theta_deg, M, X = params
    xp, yp = curve_xy(ts, theta_deg, M, X)
    return np.concatenate([xp - x_data, yp - y_data])


def fit(x_data, y_data, init=(25.0, 0.0, 50.0), n_iters=40, tol=1e-6, grid_size=2000):
    t_grid = np.linspace(T_MIN, T_MAX, grid_size)
    params = np.array(init, dtype=float)

    lower = [THETA_BOUNDS[0], M_BOUNDS[0], X_BOUNDS[0]]
    upper = [THETA_BOUNDS[1], M_BOUNDS[1], X_BOUNDS[1]]

    for it in range(n_iters):
        ts = assign_all_t(x_data, y_data, *params, t_grid)
        res = least_squares(
            residuals_given_t, x0=params, args=(ts, x_data, y_data),
            bounds=(lower, upper),
        )
        new_params = res.x
        change = np.abs(new_params - params).sum()
        params = new_params
        print(f"iter {it:2d}: theta={params[0]:.4f}  M={params[1]:.5f}  "
              f"X={params[2]:.4f}  cost={res.cost:.6f}  change={change:.6f}")
        if change < tol:
            break

    return params


def validate(params, x_data, y_data, grid_size=2000):
    theta_deg, M, X = params
    t_grid = np.linspace(T_MIN, T_MAX, grid_size)
    xp, yp = curve_xy(t_grid, theta_deg, M, X)
    dists = [np.min(np.hypot(xp - xi, yp - yi)) for xi, yi in zip(x_data, y_data)]
    return np.mean(dists), np.max(dists)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/xy_data.csv"
    df = pd.read_csv(csv_path)
    x_data = df["x"].values
    y_data = df["y"].values
    print(f"Loaded {len(x_data)} points from {csv_path}\n")

    params = fit(x_data, y_data)
    theta_deg, M, X = params

    mean_d, max_d = validate(params, x_data, y_data)

    print("\n=== RESULT ===")
    print(f"theta = {theta_deg:.4f} deg")
    print(f"M     = {M:.5f}")
    print(f"X     = {X:.4f}")
    print(f"\nValidation (nearest-point distance to fitted curve):")
    print(f"  mean = {mean_d:.5f}")
    print(f"  max  = {max_d:.5f}")

    # NOTE: Desmos' cos()/sin() take radians, not degrees, so theta is
    # converted to radians before being embedded in the LaTeX expression.
    theta_rad = np.radians(theta_deg)
    latex = (
        f"\\left(t*\\cos({theta_rad:.6g})-e^{{{M:.4g}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\sin({theta_rad:.6g})+{X:.4g},"
        f"42+t*\\sin({theta_rad:.6g})+e^{{{M:.4g}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\cos({theta_rad:.6g})\\right)"
    )
    print(f"\nDesmos/LaTeX, radians (domain 6 <= t <= 60):\n{latex}")
    print(f"(theta = {theta_deg:.4g} deg = {theta_rad:.6g} rad)")


if __name__ == "__main__":
    main()
