"""
This source code file aims at implementing a second method to 
compute spline approximation of price/volume kinematics. This
methods leverages on a unique set of knots to precompute a matrix
H used to compute the spline approximation over each window by doing 
a simple matrix multiplication H @ y, y being the data points.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import BSpline
from tqdm import tqdm

KINEMATIC_SUFFIXES = ("pos", "vel", "acc", "jrk")


class GCVOptimizationResult(NamedTuple):
    smoothing_lambda: float
    effective_df: float
    gcv_score: float

def make_clamped_uniform_knots(n_basis: int, degree: int) -> np.ndarray:
    """
    returns an array of the knots to be used when computing the clamped B-spline. 
    """
    if n_basis <= degree:
        raise ValueError("n_basis must be > degree.")

    n_internal = n_basis - degree - 1 #n_internal=number of internal knots (excluding the extremity points)
    if n_internal > 0:
        internal = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    else:
        internal = np.array([], dtype=float)

    return np.concatenate(
        [
            np.zeros(degree + 1), #we repeat (degree+1) times the extremity nodes for clamped B-splines
            internal,
            np.ones(degree + 1), #we repeat (degree+1) times the extremity nodes for clamped B-splines
        ]
    ) 

def bspline_design_matrix(
    x: np.ndarray,
    knots: np.ndarray,
    degree: int,
    n_basis: int,
    derivative_order: int = 0,
) -> np.ndarray:
    """
    computes the design matrix evaluated at x. 
    This is done iterating over j, constructing
    each basis B-spline iteratively up to the n_basis spline.
    derivative_order allows us to compute the second order 
    derivative of the spline, useful to compute the roughness penalty matrix. 
    """
    out = np.empty((len(x), n_basis), dtype=np.float64)

    for j in range(n_basis):
        coeff = np.zeros(n_basis, dtype=np.float64)
        coeff[j] = 1.0
        spline = BSpline(knots, coeff, degree, extrapolate=False)

        if derivative_order > 0:
            spline = spline.derivative(derivative_order)

        values = spline(x)
        out[:, j] = np.nan_to_num(values, nan=0.0)

    return out

def integrated_roughness_penalty(
knots: np.ndarray,
degree: int,
n_basis: int,
derivative_order: int = 2,
grid_size: int = 512,
) -> np.ndarray:
    """
    Computes the roughness penalty matrix omega from the 
    optimization problem. The grid_size is the number of points
    we use to discretize the integral into a sum.
    """
    grid = np.linspace(0.0, 1.0, grid_size)
    basis_derivative = bspline_design_matrix(
        grid,
        knots,
        degree,
        n_basis,
        derivative_order=derivative_order,
    )

    weights = np.ones(grid_size, dtype=np.float64) # from the trapeze integration method 
    weights[0] = 0.5 # because the 1st point is used only in 1 trapeze
    weights[-1] = 0.5 # because the last point is used only in 1 trapeze
    weights /= grid_size - 1

    return basis_derivative.T @ (basis_derivative * weights[:, None])


def _bspline_fit_system(window: int, n_basis: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
    if window < 2:
        raise ValueError("window must be >= 2.")
    tau = np.linspace(0.0, 1.0, window)
    knots = make_clamped_uniform_knots(n_basis=n_basis, degree=degree)
    B = bspline_design_matrix(
        tau,
        knots,
        degree,
        n_basis,
        derivative_order=0,
    )
    omega = integrated_roughness_penalty(
        knots,
        degree,
        n_basis,
        derivative_order=2,
    )
    dt = float(window - 1)
    # Express Omega in per-tick curvature units, matching derivative outputs.
    omega = omega * dt**(-4)
    return B, omega


def _regularized_lhs(
    design_matrix: np.ndarray,
    roughness_penalty: np.ndarray,
    smoothing_lambda: float,
    ridge: float,
) -> np.ndarray:
    return (
        design_matrix.T @ design_matrix
        + smoothing_lambda * roughness_penalty
        + ridge * np.eye(design_matrix.shape[1])
    )


def bspline_smoother_matrix(
    window: int,
    n_basis: int,
    smoothing_lambda: float,
    degree: int = 3,
    ridge: float = 1e-8,
) -> np.ndarray:
    """
    Returns S = B (B.T B + lambda Omega)^-1 B.T.

    S maps an observed window y to fitted spline values at the original
    window coordinates.
    """
    if smoothing_lambda < 0:
        raise ValueError("smoothing_lambda must be >= 0.")

    design_matrix, roughness_penalty = _bspline_fit_system(window, n_basis, degree)
    lhs = _regularized_lhs(design_matrix, roughness_penalty, smoothing_lambda, ridge)
    return design_matrix @ np.linalg.solve(lhs, design_matrix.T)


def _effective_degrees_of_freedom_from_system(
    design_matrix: np.ndarray,
    roughness_penalty: np.ndarray,
    smoothing_lambda: float,
    ridge: float,
) -> float:
    lhs = _regularized_lhs(design_matrix, roughness_penalty, smoothing_lambda, ridge)
    hat_basis = np.linalg.solve(lhs, design_matrix.T @ design_matrix)
    return float(np.trace(hat_basis))


def effective_degrees_of_freedom(
    window: int,
    n_basis: int,
    smoothing_lambda: float,
    degree: int = 3,
    ridge: float = 1e-8,
) -> float:
    """
    Computes trace(S_lambda), where S_lambda is the spline smoother matrix.
    """
    if smoothing_lambda < 0:
        raise ValueError("smoothing_lambda must be >= 0.")

    design_matrix, roughness_penalty = _bspline_fit_system(window, n_basis, degree)
    return _effective_degrees_of_freedom_from_system(
        design_matrix,
        roughness_penalty,
        smoothing_lambda,
        ridge,
    )


def lambda_for_effective_degrees_of_freedom(
    target_df: float,
    window: int,
    n_basis: int,
    degree: int = 3,
    ridge: float = 1e-8,
    tolerance: float = 1e-6,
    max_iterations: int = 80,
) -> float:
    """
    Finds lambda by bisection so trace(S_lambda) is close to target_df.
    """
    if target_df <= 0:
        raise ValueError("target_df must be > 0.")

    design_matrix, roughness_penalty = _bspline_fit_system(window, n_basis, degree)

    df_at_zero = _effective_degrees_of_freedom_from_system(design_matrix, roughness_penalty, 0.0, ridge)
    if target_df >= df_at_zero:
        return 0.0

    low = 0.0
    high = 1.0
    df_high = _effective_degrees_of_freedom_from_system(design_matrix, roughness_penalty, high, ridge)
    while df_high > target_df and high < 1e16:
        high *= 2.0
        df_high = _effective_degrees_of_freedom_from_system(design_matrix, roughness_penalty, high, ridge)

    if df_high > target_df:
        return high

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        df_mid = _effective_degrees_of_freedom_from_system(design_matrix, roughness_penalty, mid, ridge)
        if abs(df_mid - target_df) <= tolerance:
            return float(mid)
        if df_mid > target_df:
            low = mid
        else:
            high = mid

    return float(0.5 * (low + high))


def mean_gcv_score(
    values_by_day: list[np.ndarray],
    window: int,
    n_basis: int,
    smoothing_lambda: float,
    degree: int = 3,
    ridge: float = 1e-8,
    chunk_size: int = 4096,
    centers_by_day: list[np.ndarray] | None = None,
    scale: float = 1.0,
) -> tuple[float, float]:
    """
    Computes the mean per-window GCV score over all train windows.

    GCV(window) = (||y - Sy||^2 / W) / (1 - df / W)^2.
    Multiple features are averaged as independent spline windows.
    If centers_by_day is provided, each window is centered by its start
    center and divided by scale before fitting/scoring.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    if scale <= 0:
        raise ValueError("scale must be > 0.")
    if centers_by_day is not None and len(centers_by_day) != len(values_by_day):
        raise ValueError("centers_by_day must match values_by_day length.")

    smoother = bspline_smoother_matrix(
        window=window,
        n_basis=n_basis,
        smoothing_lambda=smoothing_lambda,
        degree=degree,
        ridge=ridge,
    )
    df = float(np.trace(smoother))
    denominator = (1.0 - df / float(window)) ** 2
    if denominator <= 0:
        raise ValueError(
            f"GCV denominator must be positive; got df={df:.6g} for window={window}."
        )

    total_score = 0.0
    total_count = 0
    for day_index, values in enumerate(values_by_day):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"Expected train values with shape [N, F], got {values.shape}.")
        n_rows, n_features = values.shape
        n_windows = n_rows - window + 1
        if n_windows <= 0 or n_features == 0:
            continue
        centers = None
        if centers_by_day is not None:
            centers = np.asarray(centers_by_day[day_index], dtype=np.float64)
            if len(centers) < n_windows:
                raise ValueError(
                    f"Expected at least {n_windows} centers for train day {day_index}, got {len(centers)}."
                )

        for start in range(0, n_windows, chunk_size):
            stop = min(start + chunk_size, n_windows)
            block = values[start : stop + window - 1]
            windows = sliding_windows_2d(block, window)
            if centers is not None:
                windows = (windows - centers[start:stop, None, None]) / scale
            fitted = np.einsum("ij,mjf->mif", smoother, windows, optimize=True)
            rss = np.sum((windows - fitted) ** 2, axis=1)
            scores = (rss / float(window)) / denominator
            total_score += float(np.sum(scores))
            total_count += scores.size

    if total_count == 0:
        raise ValueError("No train windows available for GCV smoothing-lambda optimization.")

    return total_score / total_count, df


def optimize_smoothing_lambda_gcv(
    values_by_day: list[np.ndarray],
    window: int,
    n_basis: int,
    max_df: float,
    degree: int = 3,
    ridge: float = 1e-8,
    chunk_size: int = 4096,
    n_df_candidates: int = 25,
    centers_by_day: list[np.ndarray] | None = None,
    scale: float = 1.0,
) -> GCVOptimizationResult:
    """
    Selects lambda by minimizing mean train GCV under an effective-df budget.

    The configured max_df is an upper bound on trace(S_lambda). Candidate df
    values are converted to lambda by bisection because df(lambda) decreases
    monotonically with lambda.
    """
    if n_df_candidates <= 0:
        raise ValueError("n_df_candidates must be > 0.")
    if max_df <= 0:
        raise ValueError("max_df must be > 0.")

    df_at_zero = effective_degrees_of_freedom(
        window=window,
        n_basis=n_basis,
        smoothing_lambda=0.0,
        degree=degree,
        ridge=ridge,
    )
    df_cap = min(float(max_df), df_at_zero)

    df_floor = effective_degrees_of_freedom(
        window=window,
        n_basis=n_basis,
        smoothing_lambda=1e12,
        degree=degree,
        ridge=ridge,
    )
    if df_cap <= df_floor:
        candidate_dfs = np.array([df_cap], dtype=np.float64)
    else:
        candidate_dfs = np.linspace(df_floor, df_cap, n_df_candidates, dtype=np.float64)

    best: GCVOptimizationResult | None = None
    progress = tqdm(
        candidate_dfs,
        desc="Optimizing smoothing lambda by train GCV",
        unit="df",
        mininterval=5,
    )
    for candidate_df in progress:
        smoothing_lambda = lambda_for_effective_degrees_of_freedom(
            target_df=float(candidate_df),
            window=window,
            n_basis=n_basis,
            degree=degree,
            ridge=ridge,
        )
        gcv_score, effective_df = mean_gcv_score(
            values_by_day=values_by_day,
            window=window,
            n_basis=n_basis,
            smoothing_lambda=smoothing_lambda,
            degree=degree,
            ridge=ridge,
            chunk_size=chunk_size,
            centers_by_day=centers_by_day,
            scale=scale,
        )
        result = GCVOptimizationResult(
            smoothing_lambda=float(smoothing_lambda),
            effective_df=float(effective_df),
            gcv_score=float(gcv_score),
        )
        if best is None or result.gcv_score < best.gcv_score:
            best = result
            progress.set_postfix(
                {
                    "best_lambda": f"{best.smoothing_lambda:.3g}",
                    "best_df": f"{best.effective_df:.3g}",
                    "best_gcv": f"{best.gcv_score:.3g}",
                }
            )

    if best is None:
        raise ValueError("Unable to select a smoothing lambda from GCV candidates.")
    return best


def endpoint_derivative_matrix(
    knots: np.ndarray,
    degree: int,
    n_basis: int,
    window: int,
    eval_at: float,
) -> np.ndarray:
    """
    returns R such that:

        [f(eval_at), df/dtick, d2f/dtick2, d3f/dtick3] = R @ c

    The BSpline derivatives are initially with respect to tau in [0, 1].
    We rescale them to derivatives per event/tick index.
    """
    x = np.array([eval_at], dtype=np.float64)
    dt = float(window - 1)

    rows = [] #each row embeds a derivative order of the spline at eval_at: multiplied by y, we get the derivatives. 
    for derivative_order in range(4):
        if derivative_order > degree:
            row = np.zeros((1, n_basis), dtype=np.float64)
        else:
            row = bspline_design_matrix(
                x,
                knots,
                degree,
                n_basis,
                derivative_order=derivative_order,
            )
            row = row / (dt ** derivative_order) #rescaling following chain rule derivation

        rows.append(row[0])

    return np.stack(rows, axis=0)

def penalized_bspline_filter_matrix(
    window: int,
    n_basis: int,
    smoothing_lambda: float,
    eval_at: float,
    degree: int = 3,
    ridge: float = 1e-8, #for stability, to ease the matrix inversion 
) -> np.ndarray:
    """
    Returns H with shape [4, window] such that tokens = H @ y_window.

    Objective:
        min_c ||B c - y||² + smoothing_lambda * ∫(f''(t))²dt
    """
    if window < 2:
        raise ValueError("window must be >= 2.")
    if not 0.0 <= eval_at <= 1.0:
        raise ValueError("eval_at must be in [0, 1]")

    knots = make_clamped_uniform_knots(n_basis=n_basis, degree=degree)

    B, P = _bspline_fit_system(window, n_basis, degree)
    R = endpoint_derivative_matrix(
        knots,
        degree,
        n_basis,
        window=window,
        eval_at=eval_at,
    )

    lhs = B.T @ B + smoothing_lambda * P + ridge * np.eye(n_basis)
    coef_from_y = np.linalg.solve(lhs, B.T)

    return R @ coef_from_y

def sliding_windows_2d(values: np.ndarray, window: int) -> np.ndarray:
    """
    values: [N, F]
    returns: [N - window + 1, window, F]
    """
    raw = sliding_window_view(values, window_shape=window, axis=0)
    return np.moveaxis(raw, -1, 1)

@dataclass(slots=True)
class PenalizedBSplineKinematicTokenizer:
    window: int
    n_basis: int
    smoothing_lambda: float
    eval_at: float
    chunk_size: int
    degree: int = 3
    dtype: np.dtype = np.float32
    H: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")
        self.H = penalized_bspline_filter_matrix(
            window=self.window,
            n_basis=self.n_basis,
            degree=self.degree,
            smoothing_lambda=self.smoothing_lambda,
            eval_at=self.eval_at,
        ).astype(self.dtype)

    def transform_values(self, values: np.ndarray, progress_desc: str | None = None) -> np.ndarray:
        """
        values: [N, F]
        returns: [N - window + 1, F, 4]
        """
        values = np.asarray(values, dtype=self.dtype)

        if values.ndim != 2:
            raise ValueError(f"Expected values with shape [N, F], got {values.shape}.")

        n_rows, n_features = values.shape
        n_windows = n_rows - self.window + 1
        if n_windows <= 0:
            raise ValueError(f"Need at least {self.window} rows, got {n_rows}.")

        output = np.empty((n_windows, n_features, 4), dtype=self.dtype)
        progress = tqdm(total=n_windows, desc=progress_desc, unit="rows", mininterval=5) if progress_desc else None

        try:
            for start in range(0, n_windows, self.chunk_size):
                stop = min(start + self.chunk_size, n_windows)
                block = values[start : stop + self.window - 1]
                windows = sliding_windows_2d(block, self.window)

                output[start:stop] = np.einsum(
                    "dw,mwf->mfd",
                    self.H,
                    windows,
                    optimize=True,
                )

                if progress is not None:
                    progress.update(stop - start)
        finally:
            if progress is not None:
                progress.close()

        return output
