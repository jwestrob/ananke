from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # robust regression if available
    from sklearn.linear_model import HuberRegressor
except Exception:  # pragma: no cover - optional
    HuberRegressor = None  # type: ignore


@dataclass
class ImpactParams:
    alpha: float = 0.6
    kappa: float = 1e-6
    coef_: Optional[np.ndarray] = None  # feature coefficients
    intercept_: float = 0.0


class ImpactModel:
    """
    Counterfactual slippage/impact model: Δp = κ·size^α·exp(β^T X).
    Fits via robust regression in log-space: log(Δp) = log κ + α log(size) + β^T X.
    """

    def __init__(self, alpha_init: float = 0.6, kappa_init: float = 1e-6, robust: bool = True):
        self.params = ImpactParams(alpha=alpha_init, kappa=kappa_init)
        self.robust = robust and (HuberRegressor is not None)

    def fit(self, X: np.ndarray, y: np.ndarray, sizes: np.ndarray) -> ImpactParams:
        if X.ndim != 2:
            raise ValueError("X must be 2D [N,D]")
        N = X.shape[0]
        if y.shape[0] != N or sizes.shape[0] != N:
            raise ValueError("Mismatched lengths for X, y, sizes")
        eps = 1e-9
        target = np.log(np.abs(y) + eps)
        log_size = np.log(np.abs(sizes) + eps).reshape(-1, 1)
        design = np.hstack([log_size, X])
        if self.robust:
            model = HuberRegressor(fit_intercept=True)
            model.fit(design, target)
            coef = model.coef_.copy()
            intercept = float(model.intercept_)
        else:
            # ridge-like closed form with small L2 for stability
            lam = 1e-6
            XtX = design.T @ design + lam * np.eye(design.shape[1])
            Xty = design.T @ target
            coef_full = np.linalg.solve(XtX, Xty)
            coef = coef_full
            intercept = 0.0
        # unpack
        alpha = float(max(0.1, min(1.5, coef[0])))
        beta = coef[1:]
        kappa = float(np.exp(intercept))
        self.params.alpha = alpha
        self.params.kappa = kappa
        self.params.coef_ = beta
        self.params.intercept_ = intercept
        return self.params

    def predict(self, X: np.ndarray, size: np.ndarray | float) -> np.ndarray:
        if np.isscalar(size):
            size = np.asarray([size] * X.shape[0], dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D [N,D]")
        if self.params.coef_ is None:
            beta_dot = np.zeros(X.shape[0])
        else:
            beta_dot = X @ self.params.coef_.reshape(-1)
        pred = self.params.kappa * (np.abs(size) ** self.params.alpha) * np.exp(beta_dot + self.params.intercept_)
        return pred

