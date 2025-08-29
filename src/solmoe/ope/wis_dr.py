from __future__ import annotations

import numpy as np


def wis(weights: np.ndarray, rewards: np.ndarray, clip: float = 10.0) -> float:
    w = np.clip(weights, 0.0, clip)
    if w.sum() == 0:
        return 0.0
    return float((w * rewards).sum() / w.sum())


def doubly_robust(weights: np.ndarray, rewards: np.ndarray, q_hat: np.ndarray, v_hat: np.ndarray, clip: float = 10.0) -> float:
    w = np.clip(weights, 0.0, clip)
    dr = v_hat + w * (rewards - q_hat)
    return float(dr.mean())

