from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import typer


app = typer.Typer(add_completion=False, help="Grid search tuner for the MA baseline.")


@dataclass
class BaselineParams:
    fast: int
    slow: int
    band_bp: float
    exit_bp: float
    min_hold: int
    long_only: bool
    vol_window: int
    vol_thr_bp: float
    session: str


def _load_features(path_or_dir: str) -> pd.DataFrame:
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
    if "close" not in df.columns:
        raise ValueError("features must include 'close' column")
    return df.sort_index()


def _precompute_means(close: pd.Series, windows: Iterable[int]) -> dict[int, pd.Series]:
    out: dict[int, pd.Series] = {}
    for w in sorted(set(int(x) for x in windows)):
        out[w] = close.rolling(w, min_periods=1).mean()
    return out


def _actions_hysteresis(diff_norm: np.ndarray, band_bp: float, exit_bp: float, min_hold: int, long_only: bool) -> np.ndarray:
    enter = band_bp / 10_000.0
    exit_thr = exit_bp / 10_000.0
    pos = 0
    last_change = -10**9
    N = diff_norm.shape[0]
    acts = np.empty(N, dtype=np.int8)
    for i in range(N):
        x = diff_norm[i]
        if pos == 0:
            if x > enter:
                pos = 1
                last_change = i
            elif x < -enter and not long_only:
                pos = -1
                last_change = i
        elif pos == 1:
            if i - last_change >= min_hold:
                if x < exit_thr:
                    pos = 0
                    last_change = i
                if x < -enter and not long_only:
                    pos = -1
                    last_change = i
        elif pos == -1:
            if i - last_change >= min_hold:
                if x > -exit_thr:
                    pos = 0
                    last_change = i
                if x > enter:
                    pos = 1
                    last_change = i
        acts[i] = pos
    # Map to {-1,0,1} or {0,1} depending on long_only
    if long_only:
        acts = np.where(acts == -1, 0, acts)
    return acts


def _evaluate(close: np.ndarray, actions: np.ndarray, fee_bp: float = 5.0) -> tuple[float, float, int]:
    # Compute minute returns
    ret = np.diff(close) / close[:-1]
    a_t = actions[:-1]
    pnl = a_t * ret
    # Fees on changes between t and t+1
    turn = np.abs(np.diff(actions))
    fees = (turn > 0).astype(float) * (fee_bp / 10_000.0)
    net = pnl - fees
    # Metrics
    n_trades = int((turn > 0).sum())
    total_ret = float(net.sum())
    sharpe = float(net.mean() / (net.std() + 1e-12) * math.sqrt(365 * 24 * 60))
    return total_ret, sharpe, n_trades


def _precompute_vol(close: pd.Series, windows: Iterable[int]) -> dict[int, pd.Series]:
    ret = close.pct_change().fillna(0.0)
    vols: dict[int, pd.Series] = {}
    for w in sorted(set(int(x) for x in windows)):
        if w <= 0:
            continue
        vols[w] = (ret.rolling(w, min_periods=1).std() * 10_000.0).fillna(0.0)
    return vols


def _session_mask(index: pd.DatetimeIndex, preset: str) -> np.ndarray:
    if preset == "all":
        return np.ones(len(index), dtype=bool)
    hrs = index.hour.values
    if preset == "day":
        # 07:00–19:59 UTC
        return (hrs >= 7) & (hrs < 20)
    if preset == "us":
        # 13:00–21:59 UTC (US cash focus)
        return (hrs >= 13) & (hrs < 22)
    if preset == "asia":
        # 00:00–08:59 UTC
        return (hrs >= 0) & (hrs < 9)
    # Unknown preset -> treat as all
    return np.ones(len(index), dtype=bool)


@app.command("run")
def run(
    feat_stream_dir: str = typer.Option(..., help="Directory or file path to features."),
    out_dir: str = typer.Option("./out/tune_baseline"),
    fast_grid: str = typer.Option("20,30,60"),
    slow_grid: str = typer.Option("180,360,720"),
    band_bp_grid: str = typer.Option("10,20,50,100"),
    exit_bp_grid: str = typer.Option("5,10,20"),
    min_hold_grid: str = typer.Option("60,120,240"),
    include_short: bool = typer.Option(False, help="Allow short positions in search."),
    max_rows: int = typer.Option(200_000, help="Evaluate only last N rows for speed."),
    fee_bp: float = typer.Option(5.0, help="Per-trade fee in basis points."),
    csv_only: bool = typer.Option(False, help="Only write CSV of results; skip best signals output."),
    vol_window_grid: str = typer.Option("0,60,120", help="Rolling vol window sizes; 0 disables filter."),
    vol_thr_bp_grid: str = typer.Option("0,5,10,20", help="Volatility threshold in bps; 0 disables filter."),
    session_grid: str = typer.Option("all,day,us", help="Session presets to allow: all,day,us,asia."),
):
    os.makedirs(out_dir, exist_ok=True)
    df = _load_features(feat_stream_dir)
    if max_rows and len(df) > max_rows:
        df = df.iloc[-max_rows:]
    close = df["close"].astype(float)

    fast_list = [int(x) for x in fast_grid.split(",") if x.strip()]
    slow_list = [int(x) for x in slow_grid.split(",") if x.strip()]
    band_list = [float(x) for x in band_bp_grid.split(",") if x.strip()]
    exit_list = [float(x) for x in exit_bp_grid.split(",") if x.strip()]
    hold_list = [int(x) for x in min_hold_grid.split(",") if x.strip()]
    volw_list = [int(x) for x in vol_window_grid.split(",") if x.strip()]
    volthr_list = [float(x) for x in vol_thr_bp_grid.split(",") if x.strip()]
    sess_list = [x.strip() for x in session_grid.split(",") if x.strip()]

    # Precompute all rolling means needed
    means = _precompute_means(close, list(fast_list) + list(slow_list))
    vols = _precompute_vol(close, volw_list)
    hours = df.index

    results: list[dict] = []
    best = {"sharpe": -1e9, "ret": -1e9}
    best_cfg: BaselineParams | None = None
    best_actions: np.ndarray | None = None

    for f, s in product(fast_list, slow_list):
        if f >= s:
            continue
        fast = means[f].values
        slow = means[s].values
        diff_norm = (fast / (np.where(slow == 0, np.nan, slow)) - 1.0)
        diff_norm = np.nan_to_num(diff_norm, nan=0.0, posinf=0.0, neginf=0.0)
        for band, exit_bp, hold in product(band_list, exit_list, hold_list):
            if exit_bp > band:
                continue
            for long_only in ([True, False] if include_short else [True]):
                for vw, vthr, sess in product(volw_list, volthr_list, sess_list):
                    acts = _actions_hysteresis(
                        diff_norm, band_bp=band, exit_bp=exit_bp, min_hold=hold, long_only=long_only
                    )
                    # Apply session mask
                    if sess != "all":
                        mask_sess = _session_mask(hours, sess)
                        acts = acts.copy()
                        acts[~mask_sess] = 0
                    # Apply volatility filter if enabled
                    if vw > 0 and vthr > 0:
                        vol_series = vols.get(vw)
                        if vol_series is not None:
                            mask_vol = vol_series.values >= vthr
                            acts = acts.copy()
                            acts[~mask_vol] = 0
                    total_ret, sharpe, n_trades = _evaluate(close.values, acts, fee_bp=fee_bp)
                    rec = {
                        "fast": f,
                        "slow": s,
                        "band_bp": band,
                        "exit_bp": exit_bp,
                        "min_hold": hold,
                        "long_only": long_only,
                        "vol_window": vw,
                        "vol_thr_bp": vthr,
                        "session": sess,
                        "n_trades": n_trades,
                        "ret": total_ret,
                        "sharpe": sharpe,
                    }
                    results.append(rec)
                    # Select best by Sharpe, tie-break by ret
                    if sharpe > best["sharpe"] or (abs(sharpe - best["sharpe"]) < 1e-9 and total_ret > best["ret"]):
                        best.update({"sharpe": sharpe, "ret": total_ret})
                        best_cfg = BaselineParams(
                            fast=f,
                            slow=s,
                            band_bp=band,
                            exit_bp=exit_bp,
                            min_hold=hold,
                            long_only=long_only,
                            vol_window=vw,
                            vol_thr_bp=vthr,
                            session=sess,
                        )
                        best_actions = acts

    # Save results
    res_df = pd.DataFrame(results).sort_values(["sharpe", "ret"], ascending=[False, False])
    csv_path = os.path.join(out_dir, "tune_results.csv")
    res_df.to_csv(csv_path, index=False)

    # Write best summary and signals
    summary = {
        "best": best_cfg.__dict__ if best_cfg else None,
        "metrics": best,
        "top5": res_df.head(5).to_dict(orient="records"),
        "rows": len(res_df),
    }
    with open(os.path.join(out_dir, "best_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if not csv_only and best_cfg is not None and best_actions is not None:
        sig_path = os.path.join(out_dir, "best_signals.jsonl")
        with open(sig_path, "w") as f:
            for ts, a in zip(df.index, best_actions):
                if best_cfg.long_only and a == -1:
                    a_str = "FLAT"
                else:
                    a_str = "LONG" if a == 1 else ("SHORT" if a == -1 else "FLAT")
                f.write(json.dumps({"ts": ts.isoformat(), "action": a_str}) + "\n")

        # Quick backtest of best signals (same as our internal evaluator) for consistency
        total_ret, sharpe, n_trades = _evaluate(close.values, best_actions, fee_bp=fee_bp)
        with open(os.path.join(out_dir, "best_backtest.json"), "w") as f:
            json.dump({"n_trades": n_trades, "ret": total_ret, "sharpe": sharpe}, f)

    typer.echo(f"Grid search complete. Results -> {csv_path}")


def main():  # pragma: no cover - CLI entry
    app()
