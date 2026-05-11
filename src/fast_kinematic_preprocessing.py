"""
This source code aims at implementing a second method to 
compute spline approximation of price/volume kinematics. This
methods leverages on a unique set of knots to precompute a matrix
H used to compute the spline approximation over each window by doing 
a simple matrix multiplication H @ y, y being the data points.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import BSpline
from tqdm import tqdm

KINEMATIC_SUFFIXES = ("pos", "vel", "acc", "jrk")

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

    tau = np.linspace(0.0, 1.0, window)
    knots = make_clamped_uniform_knots(n_basis=n_basis, degree=degree)

    B = bspline_design_matrix(
        tau,
        knots,
        degree,
        n_basis,
        derivative_order=0,
    )
    roughness_derivative_order = 2
    P = integrated_roughness_penalty(
        knots,
        degree,
        n_basis,
        derivative_order=roughness_derivative_order,
    )
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
