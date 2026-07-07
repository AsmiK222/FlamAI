# FlamAI
Parametric Curve Parameter Recovery
Problem: Given 1500 sample points from a parametric curve x(t), y(t) with three unknown parameters (θ, M, X) constrained to known ranges, recover their values using only the point cloud — no direct access to t.
Key challenge: The data wasn't ordered by t (consecutive CSV rows jumped around rather than progressing smoothly), so a naive fit assuming t = linspace(6, 60, 1500) in row order failed, converging to a wrong, boundary-pinned solution.
Method: Treated t as a hidden per-point variable and used an Orthogonal Distance Regression via alternating minimization (ICP-style):

Fix (θ, M, X) → find each point's best-matching t via grid search + refinement.
Fix those t values → re-fit (θ, M, X) via nonlinear least squares.
Repeat to convergence.

Result: θ = 30°, M = 0.03, X = 55 — mean fit error of 0.008 units against a curve spanning ~50 units (essentially exact).
Deliverables: solve.py (reproducible script), README with methodology and Desmos link, and the raw data — packaged as a git-ready repo.
