from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd
import typer
import numpy as np
from numba import njit
from solmoe.calibration.conformal import ConformalGate


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
    mode: str = typer.Option("baseline_ma", help="Inference mode: 'baseline_ma' or 'ensemble'."),
    fast_window: int = typer.Option(30, help="Fast window for MA baseline (bars)."),
    slow_window: int = typer.Option(180, help="Slow window for MA baseline (bars)."),
    band_bp: float = typer.Option(10.0, help="Entry threshold in basis points for |fast/slow-1|."),
    exit_bp: float = typer.Option(5.0, help="Exit threshold in basis points for hysteresis."),
    min_hold: int = typer.Option(60, help="Minimum bars to hold a position before flipping."),
    long_only: bool = typer.Option(True, help="If True, never emit SHORT; use FLAT instead."),
    vol_window: int = typer.Option(0, help="Rolling volatility window; 0 disables vol filter."),
    vol_thr_bp: float = typer.Option(0.0, help="Minimum rolling vol (bps) required to trade; 0 disables."),
    session: str = typer.Option("all", help="Session preset: all,day,us,asia."),
    ma_type: str = typer.Option("ema", help="MA type for fast/slow: sma,ema,wma."),
    enter_confirm: int = typer.Option(1, help="Bars above threshold required to enter."),
    slope_window: int = typer.Option(30, help="Slope window bars; 0 disables slope gating."),
    thr_vol_window: int = typer.Option(120, help="Vol window for adaptive thresholds; 0 disables."),
    band_k: float = typer.Option(1.0, help="Enter threshold multiplier on vol; 0 disables."),
    exit_k: float = typer.Option(1.0, help="Exit threshold multiplier on vol; 0 disables."),
    atr_window: int = typer.Option(14, help="ATR window; 0 disables trailing exit."),
    atr_mult: float = typer.Option(1.0, help="ATR multiple for trailing exit; 0 disables."),
    max_flips_day: int = typer.Option(20, help="Max flips per UTC day; 0 = unlimited."),
    max_rows: int = typer.Option(0, help="If >0, evaluate only last N rows for speed."),
    # Ensemble-specific knobs
    coverage_target: float = typer.Option(0.8, help="Conformal coverage target for ensemble mode."),
    prob_threshold: float = typer.Option(0.6, help="Fallback probability threshold for LONG in ensemble mode."),
    trend_scale: float = typer.Option(50.0, help="Expert1 trend logit scale."),
    breakout_scale: float = typer.Option(100.0, help="Expert2 breakout logit scale."),
    squeeze_scale: float = typer.Option(50.0, help="Expert3 squeeze logit scale."),
    squeeze_pct: float = typer.Option(30.0, help="Squeeze percentile for BB width."),
):
    df = _load_features(feat_stream_dir)
    if max_rows and len(df) > max_rows:
        df = df.iloc[-max_rows:]
    # Compute rolling means on close to control windows explicitly
    if "close" not in df.columns:
        raise ValueError("features must include 'close' column")
    close = df["close"].astype(float)
    # MA helpers
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
    fast = _ma(close, fast_window, ma_type)
    slow = _ma(close, slow_window, ma_type)

    # Normalized difference and hysteresis thresholds (in fraction terms)
    diff_norm = ((fast / (slow.replace(0, pd.NA))) - 1.0).fillna(0.0)
    # If ensemble mode, bypass baseline state-machine and use conformal gating
    if mode == "ensemble":
        # Expert 1: Trend (diff_norm)
        dn = diff_norm.values
        logit1 = dn * float(trend_scale)
        # Expert 2: Donchian breakout 20
        high20 = close.rolling(20, min_periods=1).max().values
        mid = (high20 + close.values) / 2.0
        breakout = (close.values - high20) / np.maximum(1e-9, mid)
        logit2 = breakout * float(breakout_scale)
        # Expert 3: Squeeze breakout (low BB width + above mean)
        m20 = close.rolling(20, min_periods=1).mean()
        std20 = close.rolling(20, min_periods=1).std().fillna(0.0)
        upper = (m20 + 2 * std20).values
        lower = (m20 - 2 * std20).values
        bb_width = (upper - lower) / np.maximum(1e-9, m20.values)
        # Percentile threshold for "squeeze"
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

        L1 = to_logits(logit1)
        L2 = to_logits(logit2)
        L3 = to_logits(logit3)
        # Equal blend of probabilities
        def softmax(x):
            x = x - np.max(x, axis=1, keepdims=True)
            e = np.exp(x)
            return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)
        P1 = softmax(L1)
        P2 = softmax(L2)
        P3 = softmax(L3)
        mix_probs = (P1 + P2 + P3) / 3.0

        # Calibration/Conformal on 70% train
        split = int(len(mix_probs) * 0.7)
        val_logits = np.log(np.clip(mix_probs[:split], 1e-9, 1.0))
        # Binary label from next-step return after fee threshold
        ret1 = close.pct_change().fillna(0.0).values
        fee_thr = (5.0 + 10.0) / 10_000.0
        lbl = (ret1[1:split+1] > fee_thr).astype(int)
        val_logits = val_logits[:len(lbl)]

        cg = None
        try:
            _cg = ConformalGate(coverage_target=float(coverage_target))
            _cg.fit(val_logits=val_logits, val_labels=lbl)
            cg = _cg
        except Exception:
            cg = None

        actions = []
        for i in range(len(mix_probs)):
            if cg is not None:
                expert_d = np.stack([P1[i], P2[i], P3[i]])  # [K,A]
                res = cg.decide(logits=np.log(np.clip(mix_probs[i], 1e-9, 1.0)), expert_dists=expert_d)
                if res.abstain:
                    actions.append("FLAT")
                else:
                    actions.append("LONG" if res.action == 1 else "FLAT")
            else:
                # fallback: probability threshold
                actions.append("LONG" if mix_probs[i, 1] > float(prob_threshold) else "FLAT")

        # Session/volatility masks
        acts_num = np.array([1 if a == "LONG" else 0 for a in actions], dtype=int)
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
        if vol_window > 0 and vol_thr_bp > 0:
            ret = close.pct_change().fillna(0.0)
            vol_bp = (ret.rolling(vol_window, min_periods=1).std() * 10_000.0).values
            acts_num = np.where(vol_bp >= vol_thr_bp, acts_num, 0)

        actions = np.where(acts_num == 1, "LONG", "FLAT")
        os.makedirs(os.path.dirname(out_signal_path), exist_ok=True)
        sig_df = pd.DataFrame({"ts": df.index, "action": actions})
        sig_df.to_json(out_signal_path, orient="records", lines=True, date_format="iso")
        typer.echo(f"Wrote signals to {out_signal_path}")
        return

    # Baseline MA path below
    # Adaptive thresholds using rolling vol (fraction units)
    vol_frac = None
    if thr_vol_window > 0 and (band_k > 0 or exit_k > 0):
        ret = close.pct_change().fillna(0.0)
        vol_bp = (ret.rolling(thr_vol_window, min_periods=1).std() * 10_000.0)
        vol_frac = (vol_bp / 10_000.0).values
    enter_arr = np.full(len(close), band_bp / 10_000.0)
    exit_arr = np.full(len(close), exit_bp / 10_000.0)
    if vol_frac is not None and band_k > 0:
        enter_arr = np.maximum(enter_arr, band_k * vol_frac)
    if vol_frac is not None and exit_k > 0:
        exit_arr = np.maximum(exit_arr, exit_k * vol_frac)

    # ATR for trailing exit
    atr_arr = None
    if atr_window > 0 and atr_mult > 0 and {"high", "low"}.issubset(df.columns):
        prev_close = close.shift(1).fillna(close.iloc[0])
        tr = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_arr = tr.rolling(atr_window, min_periods=1).mean().values

    @njit(cache=True)
    def _compute_actions(diff_norm: np.ndarray, enter_arr: np.ndarray, exit_arr: np.ndarray, min_hold: int,
                         long_only_flag: int, slope_window: int, fast: np.ndarray,
                         use_atr: int, atr: np.ndarray, atr_mult: float, max_flips_day: int,
                         day_ord: np.ndarray, close_arr: np.ndarray, enter_confirm: int) -> np.ndarray:
        N = diff_norm.shape[0]
        acts = np.zeros(N, dtype=np.int8)
        pos = 0
        last_change = -10**9
        trail = -1e18
        flips_today = 0
        cur_day = day_ord[0]
        confirm = 0
        for i in range(N):
            if day_ord[i] != cur_day:
                cur_day = day_ord[i]
                flips_today = 0
            allow_entry = True
            if slope_window > 0 and i - slope_window >= 0:
                slope = fast[i] - fast[i - slope_window]
                allow_entry = slope > 0.0
            desired = pos
            x = diff_norm[i]
            if pos == 0:
                if x > enter_arr[i] and allow_entry:
                    confirm += 1
                else:
                    confirm = 0
                if confirm >= enter_confirm and allow_entry:
                    desired = 1
            elif pos == 1:
                if use_atr == 1:
                    if trail < -1e17:
                        trail = close_arr[i] - atr_mult * atr[i]
                    else:
                        t = close_arr[i] - atr_mult * atr[i]
                        if t > trail:
                            trail = t
                if i - last_change >= min_hold:
                    cond_exit = x < exit_arr[i]
                    if use_atr == 1 and trail > -1e17 and close_arr[i] < trail:
                        cond_exit = True
                    if cond_exit:
                        desired = 0
            else:
                if i - last_change >= min_hold:
                    if x > -exit_arr[i]:
                        desired = 0
                    if x > enter_arr[i]:
                        desired = 1
            if desired != pos and max_flips_day > 0 and flips_today >= max_flips_day:
                desired = pos
            if desired != pos:
                pos = desired
                last_change = i
                flips_today += 1
                if pos == 1 and use_atr == 1:
                    trail = close_arr[i] - atr_mult * atr[i]
                else:
                    trail = -1e18
            acts[i] = pos
        if long_only_flag == 1:
            for i in range(N):
                if acts[i] == -1:
                    acts[i] = 0
        return acts

    day_ord = (df.index.normalize().asi8 // 86_400_000_000_000).astype(np.int64)
    use_atr_flag = 1 if atr_arr is not None else 0
    acts_num = _compute_actions(
        diff_norm.values.astype(np.float64),
        enter_arr.astype(np.float64),
        exit_arr.astype(np.float64),
        int(min_hold),
        1 if long_only else 0,
        int(slope_window),
        fast.values.astype(np.float64),
        use_atr_flag,
        (atr_arr.astype(np.float64) if atr_arr is not None else np.zeros(len(diff_norm), dtype=np.float64)),
        float(atr_mult),
        int(max_flips_day),
        day_ord.astype(np.int64),
        close.values.astype(np.float64),
        int(enter_confirm),
    )

    # Apply session and volatility filters (convert to numeric and mask)
    # acts_num already computed; now apply filters
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
    actions = np.where(acts_num == 1, "LONG", np.where(acts_num == -1, "SHORT", "FLAT"))

    os.makedirs(os.path.dirname(out_signal_path), exist_ok=True)
    sig_df = pd.DataFrame({"ts": df.index, "action": actions})
    sig_df.to_json(out_signal_path, orient="records", lines=True, date_format="iso")

    typer.echo(f"Wrote signals to {out_signal_path}")
