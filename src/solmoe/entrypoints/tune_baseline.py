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
    ma_type: str
    enter_confirm: int
    slope_window: int
    thr_vol_window: int
    band_k: float
    exit_k: float
    atr_window: int
    atr_mult: float
    max_flips_day: int


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


def _compute_ma(series: pd.Series, window: int, ma_type: str) -> pd.Series:
    ma_type = (ma_type or "sma").lower()
    if ma_type == "sma":
        return series.rolling(window, min_periods=1).mean()
    if ma_type == "ema":
        return series.ewm(span=window, adjust=False, min_periods=1).mean()
    if ma_type == "wma":
        # Linear weights 1..n
        def wma(x):
            w = np.arange(1, len(x) + 1)
            return np.sum(w * x) / np.sum(w)
        return series.rolling(window, min_periods=1).apply(wma, raw=True)
    # default to sma
    return series.rolling(window, min_periods=1).mean()


def _precompute_ma(series: pd.Series, windows: Iterable[int], ma_type: str) -> dict[int, pd.Series]:
    return {w: _compute_ma(series, w, ma_type) for w in sorted(set(int(x) for x in windows))}


def _precompute_atr(high: pd.Series, low: pd.Series, close: pd.Series, windows: Iterable[int]) -> dict[int, pd.Series]:
    atrs: dict[int, pd.Series] = {}
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    for w in sorted(set(int(x) for x in windows)):
        if w <= 0:
            continue
        atrs[w] = tr.rolling(w, min_periods=1).mean()
    return atrs


def _actions_policy(
    diff_norm: np.ndarray,
    fast: np.ndarray,
    close: np.ndarray,
    enter_thr: np.ndarray,
    exit_thr: np.ndarray,
    min_hold: int,
    long_only: bool,
    enter_confirm: int,
    slope_window: int,
    atr: np.ndarray | None,
    atr_mult: float,
    max_flips_day: int,
    dates: np.ndarray,
) -> np.ndarray:
    N = diff_norm.shape[0]
    pos = 0
    last_change = -10**9
    confirm = 0
    trail = -np.inf
    flips_today = 0
    cur_day = dates[0]
    acts = np.zeros(N, dtype=np.int8)
    for i in range(N):
        # reset turnover cap per day
        if dates[i] != cur_day:
            cur_day = dates[i]
            flips_today = 0

        # compute slope gate
        allow_entry = True
        if slope_window > 0 and i - slope_window >= 0:
            slope = fast[i] - fast[i - slope_window]
            allow_entry = slope > 0
        # threshold for this bar
        ent = enter_thr[i]
        ex = exit_thr[i]
        x = diff_norm[i]
        # update confirmation counter
        if x > ent and allow_entry:
            confirm = min(enter_confirm, confirm + 1)
        else:
            confirm = 0

        # trailing stop update if long
        if pos == 1 and atr is not None and atr_mult > 0:
            trail = max(trail, close[i] - atr_mult * atr[i]) if np.isfinite(trail) else close[i] - atr_mult * atr[i]

        desired = pos
        if pos == 0:
            if confirm >= enter_confirm and allow_entry:
                desired = 1
        elif pos == 1:
            # exit on hysteresis or trailing stop after min_hold
            cond_exit = (x < ex) or (atr is not None and atr_mult > 0 and trail != -np.inf and close[i] < trail)
            if (i - last_change >= min_hold) and cond_exit:
                desired = 0
        elif pos == -1:
            # Not used when long_only true; keep symmetrical logic
            if i - last_change >= min_hold and x > -ex:
                desired = 0

        # turnover cap check
        will_flip = desired != pos
        if will_flip and max_flips_day > 0 and flips_today >= max_flips_day:
            desired = pos  # block flip

        if desired != pos:
            last_change = i
            flips_today += 1
            pos = desired
            # reset trail on entry
            if pos == 1:
                trail = close[i] - (atr_mult * atr[i] if (atr is not None and atr_mult > 0) else 0.0)
            else:
                trail = -np.inf

        acts[i] = pos

    if long_only:
        acts = np.where(acts == -1, 0, acts)
    return acts


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
    ma_type_grid: str = typer.Option("sma,ema", help="MA types for fast/slow: sma,ema,wma."),
    enter_confirm_grid: str = typer.Option("1,2", help="Consecutive bars above threshold to enter."),
    slope_window_grid: str = typer.Option("0,30", help="Slope window bars; 0 disables slope gating."),
    thr_vol_window_grid: str = typer.Option("0,120", help="Vol window for adaptive thresholds; 0 disables."),
    band_k_grid: str = typer.Option("0,1.0", help="Enter threshold multiplier on vol; 0 disables."),
    exit_k_grid: str = typer.Option("0,1.0", help="Exit threshold multiplier on vol; 0 disables."),
    atr_window_grid: str = typer.Option("0,14", help="ATR window; 0 disables trailing exit."),
    atr_mult_grid: str = typer.Option("0,1.0,2.0", help="ATR multiple for trailing stop; 0 disables."),
    max_flips_day_grid: str = typer.Option("0,20", help="Max position changes per UTC day; 0 = unlimited."),
    train_frac: float = typer.Option(1.0, help="Fraction of data to use for selection; rest for holdout reporting."),
    select_by: str = typer.Option("train_sharpe", help="Criterion: train_sharpe|test_sharpe|train_ret."),
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
    ma_types = [x.strip() for x in ma_type_grid.split(",") if x.strip()]
    confirm_list = [int(x) for x in enter_confirm_grid.split(",") if x.strip()]
    slopew_list = [int(x) for x in slope_window_grid.split(",") if x.strip()]
    thrvolw_list = [int(x) for x in thr_vol_window_grid.split(",") if x.strip()]
    bandk_list = [float(x) for x in band_k_grid.split(",") if x.strip()]
    exitk_list = [float(x) for x in exit_k_grid.split(",") if x.strip()]
    atrw_list = [int(x) for x in atr_window_grid.split(",") if x.strip()]
    atrm_list = [float(x) for x in atr_mult_grid.split(",") if x.strip()]
    flips_list = [int(x) for x in max_flips_day_grid.split(",") if x.strip()]

    # Precompute all rolling means needed
    vols = _precompute_vol(close, set(volw_list) | set(thrvolw_list))
    hours = df.index
    dates = df.index.date
    # Precompute ATR candidates if OHLC present
    atrs = _precompute_atr(df["high"], df["low"], close, atrw_list) if {"high", "low"}.issubset(df.columns) else {}

    results: list[dict] = []
    best = {"sharpe": -1e9, "ret": -1e9}
    best_cfg: BaselineParams | None = None
    best_actions: np.ndarray | None = None

    for ma_type in ma_types:
      means = _precompute_ma(close, list(fast_list) + list(slow_list), ma_type)
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
                    for thrw, bk, ek, conf, slopew, atrw, atrm, flips in product(
                        thrvolw_list, bandk_list, exitk_list, confirm_list, slopew_list, atrw_list, atrm_list, flips_list
                    ):
                        # Build per-bar thresholds in fraction
                        vol_thr = vols.get(thrw)
                        vol_frac = (vol_thr.values / 10_000.0) if (vol_thr is not None and thrw > 0) else None
                        enter_thr = np.full(len(diff_norm), band / 10_000.0, dtype=float)
                        exit_thr = np.full(len(diff_norm), exit_bp / 10_000.0, dtype=float)
                        if vol_frac is not None and bk > 0:
                            enter_thr = np.maximum(enter_thr, bk * vol_frac)
                        if vol_frac is not None and ek > 0:
                            exit_thr = np.maximum(exit_thr, ek * vol_frac)

                        # Base actions with policy enhancements
                        atr_arr = atrs.get(atrw).values if (atrw > 0 and atrs.get(atrw) is not None) else None
                        acts = _actions_policy(
                            diff_norm=diff_norm,
                            fast=fast,
                            close=close.values,
                            enter_thr=enter_thr,
                            exit_thr=exit_thr,
                            min_hold=hold,
                            long_only=long_only,
                            enter_confirm=conf,
                            slope_window=slopew,
                            atr=atr_arr,
                            atr_mult=atrm,
                            max_flips_day=flips,
                            dates=dates,
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
                    # Train/Test split
                    split = int(len(acts) * float(train_frac)) if 0 < train_frac < 1.0 else len(acts)
                    tr_slice = slice(0, split)
                    te_slice = slice(split, None)
                    tr_metrics = _evaluate(close.values[0:split], acts[0:split], fee_bp=fee_bp)
                    te_metrics = _evaluate(close.values[split - 1 :], acts[split - 1 :], fee_bp=fee_bp) if split < len(acts) else tr_metrics
                    total_ret, sharpe, n_trades = tr_metrics
                    total_ret_te, sharpe_te, n_trades_te = te_metrics
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
                        "ma_type": ma_type,
                        "enter_confirm": conf,
                        "slope_window": slopew,
                        "thr_vol_window": thrw,
                        "band_k": bk,
                        "exit_k": ek,
                        "atr_window": atrw,
                        "atr_mult": atrm,
                        "max_flips_day": flips,
                        "n_trades": n_trades,
                        "ret": total_ret,
                        "sharpe": sharpe,
                        "n_trades_te": n_trades_te,
                        "ret_te": total_ret_te,
                        "sharpe_te": sharpe_te,
                    }
                    results.append(rec)
                    # Selection
                    key = select_by.lower()
                    sel_val = sharpe if key == "train_sharpe" else (total_ret if key == "train_ret" else sharpe_te)
                    best_val = best["sharpe"] if key != "train_ret" else best["ret"]
                    if sel_val > best_val or (
                        abs(sel_val - best_val) < 1e-9 and (total_ret if key != "train_ret" else sharpe) > (best["ret"] if key != "train_ret" else best["sharpe"])
                    ):
                        best.update({"sharpe": sharpe_te if key == "test_sharpe" else sharpe, "ret": total_ret_te if key == "test_sharpe" else total_ret})
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
                            ma_type=ma_type,
                            enter_confirm=conf,
                            slope_window=slopew,
                            thr_vol_window=thrw,
                            band_k=bk,
                            exit_k=ek,
                            atr_window=atrw,
                            atr_mult=atrm,
                            max_flips_day=flips,
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
