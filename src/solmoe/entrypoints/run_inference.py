from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd
import typer
import numpy as np


def _load_features(path_or_dir: str) -> pd.DataFrame:
    # Accept directory containing features.parquet or a direct file path
    p = path_or_dir
    if os.path.isdir(p):
        cand = os.path.join(p, "features.parquet")
    else:
        cand = p
    if not os.path.exists(cand):
        raise FileNotFoundError(f"Could not find features at: {cand}")
    df = pd.read_parquet(cand)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts" in df.columns:
            df.index = pd.to_datetime(df["ts"], utc=True)
        else:
            raise ValueError("features must have DatetimeIndex or a 'ts' column")
    return df.sort_index()


def main(
    expert_dir: str = typer.Option(..., help="Unused for baseline; reserved for model-based inference."),
    gate_ckpt: str = typer.Option(..., help="Unused for baseline; reserved for model-based inference."),
    feat_stream_dir: str = typer.Option(..., help="Directory or file path to features."),
    out_signal_path: str = typer.Option(..., help="Output JSONL path for signals."),
    mode: str = typer.Option("baseline_ma", help="Inference mode: 'baseline_ma' only implemented."),
    fast_window: int = typer.Option(30, help="Fast window for MA baseline (bars)."),
    slow_window: int = typer.Option(180, help="Slow window for MA baseline (bars)."),
    band_bp: float = typer.Option(10.0, help="Entry threshold in basis points for |fast/slow-1|."),
    exit_bp: float = typer.Option(5.0, help="Exit threshold in basis points for hysteresis."),
    min_hold: int = typer.Option(60, help="Minimum bars to hold a position before flipping."),
    long_only: bool = typer.Option(True, help="If True, never emit SHORT; use FLAT instead."),
    vol_window: int = typer.Option(0, help="Rolling volatility window; 0 disables vol filter."),
    vol_thr_bp: float = typer.Option(0.0, help="Minimum rolling vol (bps) required to trade; 0 disables."),
    session: str = typer.Option("all", help="Session preset: all,day,us,asia."),
):
    if mode != "baseline_ma":
        raise NotImplementedError("Only 'baseline_ma' mode is implemented in this scaffold.")

    df = _load_features(feat_stream_dir)
    # Compute rolling means on close to control windows explicitly
    if "close" not in df.columns:
        raise ValueError("features must include 'close' column")
    close = df["close"].astype(float)
    fast = close.rolling(fast_window, min_periods=1).mean()
    slow = close.rolling(slow_window, min_periods=1).mean()

    # Normalized difference and hysteresis thresholds (in fraction terms)
    diff_norm = ((fast / (slow.replace(0, pd.NA))) - 1.0).fillna(0.0)
    enter = band_bp / 10_000.0
    exit_thr = exit_bp / 10_000.0

    # Generate stateful actions with hysteresis and min-hold
    actions = []
    pos = 0  # -1,0,1
    last_change_idx = -10**9
    for i, x in enumerate(diff_norm.values):
        if pos == 0:
            if x > enter:
                pos = 1
                last_change_idx = i
            elif x < -enter:
                pos = -1
                last_change_idx = i
        elif pos == 1:
            if i - last_change_idx >= min_hold:
                if x < exit_thr:
                    pos = 0
                    last_change_idx = i
                if x < -enter:  # flip only after min_hold and strong signal
                    pos = -1
                    last_change_idx = i
        elif pos == -1:
            if i - last_change_idx >= min_hold:
                if x > -exit_thr:
                    pos = 0
                    last_change_idx = i
                if x > enter:
                    pos = 1
                    last_change_idx = i
        if long_only and pos == -1:
            actions.append("FLAT")
        else:
            actions.append("LONG" if pos == 1 else ("SHORT" if pos == -1 else "FLAT"))

    # Apply session and volatility filters (convert to numeric and mask)
    acts_num = np.array([1 if a == "LONG" else (-1 if a == "SHORT" else 0) for a in actions], dtype=int)
    # Session mask
    if session != "all":
        hrs = df.index.hour.values
        if session == "day":
            mask_sess = (hrs >= 7) & (hrs < 20)
        elif session == "us":
            mask_sess = (hrs >= 13) & (hrs < 22)
        elif session == "asia":
            mask_sess = (hrs >= 0) & (hrs < 9)
        else:
            mask_sess = np.ones_like(hrs, dtype=bool)
        acts_num = np.where(mask_sess, acts_num, 0)
    # Volatility mask
    if vol_window > 0 and vol_thr_bp > 0:
        ret = close.pct_change().fillna(0.0)
        vol_bp = (ret.rolling(vol_window, min_periods=1).std() * 10_000.0).values
        acts_num = np.where(vol_bp >= vol_thr_bp, acts_num, 0)

    # Convert back to strings
    actions = ["LONG" if a == 1 else ("SHORT" if a == -1 else "FLAT") for a in acts_num]

    os.makedirs(os.path.dirname(out_signal_path), exist_ok=True)
    with open(out_signal_path, "w") as f:
        for ts, a in zip(df.index, actions):
            rec = {"ts": ts.isoformat(), "action": a}
            f.write(json.dumps(rec) + "\n")

    typer.echo(f"Wrote signals to {out_signal_path}")
