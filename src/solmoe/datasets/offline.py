from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from solmoe.risk.slippage import ImpactModel


@dataclass
class RewardSpec:
    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 0.0
    drawdown_penalty: float = 0.0
    turnover_penalty: float = 0.0


def compute_reward(ret: np.ndarray, action: np.ndarray, fees_bps: float, slippage_bps: np.ndarray) -> np.ndarray:
    """Compute cost-aware reward per step for discrete actions {-1,0,1}."""
    gross = action[:-1] * ret  # position applies to next-step return
    # fees taken on change of position
    turn = np.abs(np.diff(action))
    fees = (turn > 0).astype(float) * (fees_bps / 10_000.0)
    slippage = slippage_bps / 10_000.0
    net = gross - fees[: gross.shape[0]] - slippage[: gross.shape[0]]
    return net


def build_offline_buffer(feat_dir: str, out_path: str, horizons: List[str]):
    """
    Build (s,a,r,s',done,info) buffer from features + behavior policies.
    This function expects user-provided features in feat_dir.
    """
    feats_path = os.path.join(feat_dir, "features.parquet")
    if not os.path.exists(feats_path):
        raise FileNotFoundError("features.parquet not found; run feature pipeline with real data.")
    df = pd.read_parquet(feats_path)
    if "close" not in df.columns:
        raise ValueError("features.parquet must include 'close' column")
    # Behavior actions placeholder: zero (FLAT). Real behavior policies should be provided externally.
    # Actions represent positions at each step, applying to next-step returns.
    # Use length N+1 so that a[:-1] aligns with ret of length N.
    a = np.zeros(len(df) + 1, dtype=int)
    ret = df["close"].pct_change().fillna(0.0).values
    # Simple slippage proxy based on volatility
    vol = df["close"].pct_change().rolling(10).std().fillna(0.0).values
    # Use non-robust closed-form by default here to avoid sklearn warnings on huge datasets
    im = ImpactModel(robust=False)
    X = vol.reshape(-1, 1)
    sizes = np.ones(len(df)) * 1.0
    y = np.abs(vol) * 100.0  # bps proxy for fitting only
    # Fit impact model; subsample deterministically if very large for efficiency
    N = X.shape[0]
    if N > 200_000:
        step = max(1, N // 200_000)
        idx = np.arange(10, N, step)  # start after warmup
        im.fit(X[idx], y[idx], sizes[idx])
    else:
        im.fit(X[10:], y[10:], sizes[10:])
    slip_bps = im.predict(X, size=1.0)
    r = compute_reward(ret, a, fees_bps=5.0, slippage_bps=slip_bps)
    # Build buffer arrays
    s = df.drop(columns=[c for c in ["open", "high", "low", "close"] if c in df.columns]).values
    sp = np.roll(s, -1, axis=0)
    done = np.zeros(len(df), dtype=bool)
    done[-1] = True
    buffer = {
        "s": s.tolist(),
        "a": a.tolist(),
        "r": r.tolist(),
        "sp": sp.tolist(),
        "done": done.tolist(),
        "horizons": horizons,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.Series(buffer).to_pickle(out_path)
