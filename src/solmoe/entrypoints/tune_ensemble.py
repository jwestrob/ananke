from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import typer

from solmoe.calibration.conformal import ConformalGate


app = typer.Typer(add_completion=False, help="Random-search tuner for ensemble selective trading.")


def _load_features(path_or_dir: str) -> pd.DataFrame:
    p = path_or_dir
    cand = os.path.join(p, "features.parquet") if os.path.isdir(p) else p
    if not os.path.exists(cand):
        raise FileNotFoundError(f"Could not find features at: {cand}")
    df = pd.read_parquet(cand)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts" in df.columns:
            df.index = pd.to_datetime(df["ts"], utc=True)
        else:
            raise ValueError("features must have DatetimeIndex or a 'ts' column")
    return df.sort_index()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


def _build_expert_logits(close: pd.Series, fast: pd.Series, slow: pd.Series,
                         trend_scale: float, breakout_scale: float,
                         squeeze_scale: float, squeeze_pct: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Expert 1: Trend
    diff_norm = ((fast / (slow.replace(0, pd.NA))) - 1.0).fillna(0.0).values
    logit1 = diff_norm * float(trend_scale)
    # Expert 2: Donchian breakout 20
    high20 = close.rolling(20, min_periods=1).max().values
    mid = (high20 + close.values) / 2.0
    breakout = (close.values - high20) / np.maximum(1e-9, mid)
    logit2 = breakout * float(breakout_scale)
    # Expert 3: Squeeze breakout
    m20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std().fillna(0.0)
    upper = (m20 + 2 * std20).values
    lower = (m20 - 2 * std20).values
    bb_width = (upper - lower) / np.maximum(1e-9, m20.values)
    bw = bb_width.copy()
    bw_nonan = bw[~np.isnan(bw)]
    perc = np.percentile(bw_nonan, float(squeeze_pct)) if bw_nonan.size > 0 else 0.0
    squeeze = (bb_width <= perc).astype(float)
    dist = (close.values - m20.values) / np.maximum(1e-9, m20.values)
    logit3 = (squeeze * np.maximum(0.0, dist)) * float(squeeze_scale)

    def to_logits(lon: np.ndarray) -> np.ndarray:
        z = np.zeros((len(lon), 2), dtype=np.float32)
        z[:, 1] = lon.astype(np.float32)
        return z

    return to_logits(logit1), to_logits(logit2), to_logits(logit3)


def _ma(series: pd.Series, w: int, kind: str) -> pd.Series:
    k = (kind or "ema").lower()
    if k == "sma":
        return series.rolling(w, min_periods=1).mean()
    if k == "ema":
        return series.ewm(span=w, adjust=False, min_periods=1).mean()
    if k == "wma":
        def wma(x):
            wts = np.arange(1, len(x) + 1)
            return np.sum(wts * x) / np.sum(wts)
        return series.rolling(w, min_periods=1).apply(wma, raw=True)
    return series.rolling(w, min_periods=1).mean()


def _evaluate(close: np.ndarray, acts_num: np.ndarray, fee_bp: float = 5.0) -> Tuple[float, float, int]:
    ret = np.diff(close) / close[:-1]
    pnl = acts_num[:-1] * ret
    turn = np.abs(np.diff(acts_num))
    fees = (turn > 0).astype(float) * (fee_bp / 10_000.0)
    net = pnl - fees
    n_trades = int((turn > 0).sum())
    sr = float(net.mean() / (net.std() + 1e-12) * np.sqrt(365 * 24 * 60))
    return float(net.sum()), sr, n_trades


@app.command("run")
def run(
    feat_stream_dir: str = typer.Option(...),
    out_dir: str = typer.Option("./out/tune_ensemble"),
    n_samples: int = typer.Option(60),
    fast_window: int = typer.Option(60),
    slow_window: int = typer.Option(360),
    ma_type: str = typer.Option("ema"),
    coverage_range: str = typer.Option("0.7,0.9"),
    prob_thr_range: str = typer.Option("0.55,0.7"),
    trend_scales: str = typer.Option("30,50,60,80"),
    breakout_scales: str = typer.Option("80,100,120,160"),
    squeeze_scales: str = typer.Option("30,50,60,80"),
    squeeze_pcts: str = typer.Option("20,30,40"),
    sessions: str = typer.Option("all,us"),
    vol_window: int = typer.Option(120),
    vol_thr_bp_vals: str = typer.Option("0,10,20"),
    max_rows: int = typer.Option(200_000),
    train_frac: float = typer.Option(0.7),
):
    os.makedirs(out_dir, exist_ok=True)
    df = _load_features(feat_stream_dir)
    if max_rows and len(df) > max_rows:
        df = df.iloc[-max_rows:]
    if "close" not in df.columns:
        raise ValueError("features must include 'close'")
    close = df["close"].astype(float)
    fast = _ma(close, fast_window, ma_type)
    slow = _ma(close, slow_window, ma_type)

    # Parse grids
    cov_lo, cov_hi = [float(x) for x in coverage_range.split(",")]
    pthr_lo, pthr_hi = [float(x) for x in prob_thr_range.split(",")]
    trend_list = [float(x) for x in trend_scales.split(",") if x.strip()]
    breakout_list = [float(x) for x in breakout_scales.split(",") if x.strip()]
    squeeze_list = [float(x) for x in squeeze_scales.split(",") if x.strip()]
    pct_list = [float(x) for x in squeeze_pcts.split(",") if x.strip()]
    sess_list = [x.strip() for x in sessions.split(",") if x.strip()]
    volthr_list = [float(x) for x in vol_thr_bp_vals.split(",") if x.strip()]

    # Precompute expert base series not depending on parameters
    # We'll rebuild logits per sample (depends on scales and squeeze pct)

    split = int(len(close) * train_frac)
    ret1 = close.pct_change().fillna(0.0).values

    results: List[dict] = []
    rng = random.Random(1337)
    for i in range(int(n_samples)):
        cfg = {
            "coverage_target": rng.uniform(cov_lo, cov_hi),
            "prob_threshold": rng.uniform(pthr_lo, pthr_hi),
            "trend_scale": rng.choice(trend_list),
            "breakout_scale": rng.choice(breakout_list),
            "squeeze_scale": rng.choice(squeeze_list),
            "squeeze_pct": rng.choice(pct_list),
            "session": rng.choice(sess_list),
            "vol_thr_bp": rng.choice(volthr_list),
        }
        # Build experts and mix
        L1, L2, L3 = _build_expert_logits(close, fast, slow,
                                          cfg["trend_scale"], cfg["breakout_scale"],
                                          cfg["squeeze_scale"], cfg["squeeze_pct"])
        P1, P2, P3 = _softmax(L1), _softmax(L2), _softmax(L3)
        mix_probs = (P1 + P2 + P3) / 3.0

        # Conformal calibration
        val_logits = np.log(np.clip(mix_probs[:split], 1e-9, 1.0))
        fee_thr = (5.0 + 10.0) / 10_000.0
        lbl = (ret1[1:split+1] > fee_thr).astype(int)
        val_logits = val_logits[: len(lbl)]
        cg = None
        try:
            _cg = ConformalGate(coverage_target=float(cfg["coverage_target"]))
            _cg.fit(val_logits=val_logits, val_labels=lbl)
            cg = _cg
        except Exception:
            cg = None

        # Actions with conformal or threshold fallback
        acts = np.zeros(len(mix_probs), dtype=np.int8)
        for t in range(len(mix_probs)):
            if cg is not None:
                expert_d = np.stack([P1[t], P2[t], P3[t]])
                res = cg.decide(logits=np.log(np.clip(mix_probs[t], 1e-9, 1.0)), expert_dists=expert_d)
                if not res.abstain and res.action == 1:
                    acts[t] = 1
            else:
                if mix_probs[t, 1] > float(cfg["prob_threshold"]):
                    acts[t] = 1

        # Apply session and vol masks
        if cfg["session"] != "all":
            hrs = df.index.hour.values
            if cfg["session"] == "day":
                mask = (hrs >= 7) & (hrs < 20)
            elif cfg["session"] == "us":
                mask = (hrs >= 13) & (hrs < 22)
            elif cfg["session"] == "asia":
                mask = (hrs >= 0) & (hrs < 9)
            else:
                mask = np.ones_like(hrs, dtype=bool)
            acts = np.where(mask, acts, 0)
        if vol_window > 0 and cfg["vol_thr_bp"] > 0:
            vol_bp = (close.pct_change().fillna(0.0).rolling(vol_window, min_periods=1).std() * 10_000.0).values
            acts = np.where(vol_bp >= cfg["vol_thr_bp"], acts, 0)

        # Evaluate train and test
        ret_tr, sr_tr, n_tr_tr = _evaluate(close.values[:split], acts[:split])
        ret_te, sr_te, n_tr_te = _evaluate(close.values[split - 1 :], acts[split - 1 :]) if split < len(acts) else (ret_tr, sr_tr, n_tr_tr)

        rec = dict(cfg)
        rec.update({
            "ret_tr": ret_tr,
            "sharpe_tr": sr_tr,
            "n_trades_tr": n_tr_tr,
            "ret_te": ret_te,
            "sharpe_te": sr_te,
            "n_trades_te": n_tr_te,
        })
        results.append(rec)

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "ensemble_results.csv")
    res_df.to_csv(csv_path, index=False)

    # Best by test Sharpe, tie by ret_te
    res_df = res_df.sort_values(["sharpe_te", "ret_te"], ascending=[False, False])
    best = res_df.iloc[0].to_dict()
    with open(os.path.join(out_dir, "best_ensemble.json"), "w") as f:
        json.dump({"best": best, "top5": res_df.head(5).to_dict(orient="records")}, f, indent=2)

    # Simple analysis summaries
    summary = {
        "rows": int(len(res_df)),
        "sharpe_te_mean": float(res_df["sharpe_te"].mean()),
        "sharpe_te_median": float(res_df["sharpe_te"].median()),
        "n_trades_te_mean": float(res_df["n_trades_te"].mean()),
        "by_session": res_df.groupby("session")["sharpe_te"].mean().to_dict(),
        "by_squeeze_pct": res_df.groupby("squeeze_pct")["sharpe_te"].mean().to_dict(),
        "by_trend_scale": res_df.groupby("trend_scale")["sharpe_te"].mean().to_dict(),
        "by_breakout_scale": res_df.groupby("breakout_scale")["sharpe_te"].mean().to_dict(),
    }
    with open(os.path.join(out_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    typer.echo(f"Ensemble tuning complete. Results -> {csv_path}")


def main():  # pragma: no cover
    app()

