from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class FQEResult:
    v_mean: float
    ci_low: float
    ci_high: float


class FQE:
    """Fitted Q Evaluation for discrete actions; minimal placeholder implementation."""

    def __init__(self, n_actions: int = 3):
        self.n_actions = n_actions

    def fit(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, sp: np.ndarray, done: np.ndarray, gamma: float = 0.997):
        # simple tabular-like averaging over returns; placeholder
        self.v = r.mean()
        return self

    def evaluate(self, s0: np.ndarray, bootstrap: int = 200) -> FQEResult:
        vals = np.random.normal(loc=self.v, scale=0.01, size=bootstrap)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return FQEResult(v_mean=float(self.v), ci_low=float(lo), ci_high=float(hi))

