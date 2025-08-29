from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import pandas as pd


def run_vectorized(signal_path: str, prices_path: str, out_path: str) -> None:
    """Vectorized cost-aware replay using signals and prices."""
    if not os.path.exists(signal_path) or not os.path.exists(prices_path):
        raise FileNotFoundError("Provide existing signal_path and prices_path")
    sigs = pd.read_json(signal_path, lines=True)
    prices = pd.read_parquet(prices_path)
    if "ts" not in sigs.columns or "action" not in sigs.columns:
        raise ValueError("Signals must include 'ts' and 'action'")
    sigs["ts"] = pd.to_datetime(sigs["ts"], utc=True)
    prices.index = pd.to_datetime(prices.index, utc=True)
    df = sigs.set_index("ts").sort_index().join(prices[["close"]], how="inner").dropna()

    action_map: Dict[str, int] = {"LONG": 1, "FLAT": 0, "SHORT": -1}
    a = df["action"].map(action_map).astype(int).values
    px = df["close"].values
    # Returns aligned to position at time t (applies over [t, t+1])
    ret = np.diff(px) / px[:-1]
    a_t = a[:-1]
    pnl = a_t * ret
    # Trading fee assessed when position changes between t and t+1
    turn = np.abs(np.diff(a))
    fee_bp = 5.0
    fees = (turn > 0).astype(float) * (fee_bp / 10_000.0)
    # Ensure alignment: fees length equals pnl length
    net = pnl - fees
    res = {
        "n_trades": int((turn > 0).sum()),
        "ret": float(net.sum()),
        "sharpe": float(net.mean() / (net.std() + 1e-9) * np.sqrt(365 * 24 * 60)),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f)
