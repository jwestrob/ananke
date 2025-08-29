from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist


def _mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float = None) -> float:
    if gamma is None:
        # median heuristic
        xy = np.vstack([x, y])
        dists = cdist(xy, xy, metric="sqeuclidean")
        med = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * med) if med > 0 else 1.0
    k_xx = np.exp(-gamma * cdist(x, x, metric="sqeuclidean"))
    k_yy = np.exp(-gamma * cdist(y, y, metric="sqeuclidean"))
    k_xy = np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))
    return float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())


def _energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    d_xx = cdist(x, x).mean()
    d_yy = cdist(y, y).mean()
    d_xy = cdist(x, y).mean()
    return float(2 * d_xy - d_xx - d_yy)


@dataclass
class ShiftResult:
    mmd: float
    energy: float
    alarm: bool


class ShiftDetector:
    """Embedding shift via MMD and energy distance."""

    def __init__(self, mmd_thresh: float = 0.05, energy_thresh: float = 0.05):
        self.mmd_thresh = mmd_thresh
        self.energy_thresh = energy_thresh

    def compare(self, ref: np.ndarray, cur: np.ndarray) -> ShiftResult:
        mmd = _mmd_rbf(ref, cur)
        energy = _energy_distance(ref, cur)
        alarm = (mmd > self.mmd_thresh) or (energy > self.energy_thresh)
        return ShiftResult(mmd=mmd, energy=energy, alarm=alarm)


class SPRTCalibrationGuard:
    """
    Sequential Probability Ratio Test on calibration error rates.

    Tracks failures of selective prediction (e.g., reliability below target).
    """

    def __init__(self, p0: float = 0.1, p1: float = 0.2, alpha: float = 0.05, beta: float = 0.05):
        self.p0, self.p1, self.alpha, self.beta = p0, p1, alpha, beta
        self.A = (1 - beta) / alpha
        self.B = beta / (1 - alpha)
        self.llr = 0.0

    def update(self, error: int) -> Tuple[bool, float]:
        # Bernoulli LLR increment
        self.llr += error * np.log(self.p1 / self.p0) + (1 - error) * np.log((1 - self.p1) / (1 - self.p0))
        # Decision thresholds in log space
        if np.exp(self.llr) >= self.A:
            return True, self.llr
        if np.exp(self.llr) <= self.B:
            return False, self.llr
        return False, self.llr

