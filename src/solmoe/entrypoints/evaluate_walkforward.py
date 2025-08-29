from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from solmoe.calibration.conformal import ConformalGate
from solmoe.risk.slippage import ImpactModel
import torch


app = typer.Typer(add_completion=False, help="Walk-forward evaluation for selective classifier with edge gating.")


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


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1).fillna(df["close"].iloc[0])
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()


def _future_rolling(series: pd.Series, window: int, fn: str) -> pd.Series:
    rev = series[::-1]
    if fn == "max":
        out = rev.rolling(window, min_periods=1).max()[::-1]
    else:
        out = rev.rolling(window, min_periods=1).min()[::-1]
    return out


def build_triple_barrier_labels(df: pd.DataFrame, horizon: int = 240, atr_mult_tp: float = 1.5, atr_mult_sl: float = 1.0, atr_window: int = 14) -> pd.Series:
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("features must include 'high','low','close' for triple-barrier labels")
    atr = _atr(df, window=atr_window)
    up = df["close"] + atr_mult_tp * atr
    dn = df["close"] - atr_mult_sl * atr
    fmax = _future_rolling(df["high"], horizon, "max")
    fmin = _future_rolling(df["low"], horizon, "min")
    # Approximate: label positive if future max crosses up and future min does not cross down
    pos = (fmax >= up) & (fmin > dn)
    neg = (fmin <= dn) & (fmax < up)
    # ambiguous cases (both or neither): fallback to horizon return sign
    ret_h = df["close"].shift(-horizon) / df["close"] - 1.0
    y = np.where(pos, 1, np.where(neg, 0, (ret_h > 0).astype(int)))
    y = pd.Series(y, index=df.index).fillna(0).astype(int)
    return y


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    mom1 = close.pct_change(1).fillna(0.0)
    sma60 = close.rolling(60, min_periods=1).mean()
    sma360 = close.rolling(360, min_periods=1).mean()
    z_mom_fast = (close / sma60 - 1.0).fillna(0.0)
    z_mom_slow = (sma60 / sma360 - 1.0).fillna(0.0)
    vol20 = (close.pct_change().rolling(20, min_periods=1).std()).fillna(0.0)
    atr14 = _atr(df, 14)
    atr_norm = (atr14 / close).fillna(0.0)
    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std().fillna(0.0)
    bbw = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / (bb_mid.replace(0, np.nan))
    bbw = bbw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    donch_max = df["high"].rolling(20, min_periods=1).max()
    donch_pos = ((close - donch_max) / close).fillna(0.0)
    # Cyclical time features
    hrs = df.index.hour.values
    hod_sin = np.sin(2 * np.pi * hrs / 24.0)
    hod_cos = np.cos(2 * np.pi * hrs / 24.0)
    X = pd.DataFrame({
        "mom1": mom1,
        "z_mom_fast": z_mom_fast,
        "z_mom_slow": z_mom_slow,
        "vol20": vol20,
        "atr_norm": atr_norm,
        "bbw": bbw,
        "donch_pos": donch_pos,
        "hod_sin": hod_sin,
        "hod_cos": hod_cos,
    }, index=df.index)
    return X.fillna(0.0)


def evaluate_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, horizon: int, atr_tp: float, atr_sl: float,
                  coverage: float, fee_bp: float, vol_slip_window: int = 20,
                  p_min: float = 0.55, slip_floor_bp: float = 2.0,
                  session: str = "all", edge_margin: float = 1.0) -> Dict:
    # Labels on train
    y = build_triple_barrier_labels(train_df, horizon=horizon, atr_mult_tp=atr_tp, atr_mult_sl=atr_sl)
    X = build_features(train_df)
    # Align
    y = y.loc[X.index]
    # Split train into fit/calib (80/20)
    split = int(len(X) * 0.8)
    X_fit, y_fit = X.iloc[:split], y.iloc[:split]
    X_cal, y_cal = X.iloc[split:], y.iloc[split:]
    # Optional session filter on train
    if session != "all":
        hrs_tr = train_df.index.hour.values
        if session == "day":
            mask_tr = (hrs_tr >= 7) & (hrs_tr < 20)
        elif session == "us":
            mask_tr = (hrs_tr >= 13) & (hrs_tr < 22)
        elif session == "asia":
            mask_tr = (hrs_tr >= 0) & (hrs_tr < 9)
        else:
            mask_tr = np.ones_like(hrs_tr, dtype=bool)
        X = X.loc[mask_tr]
        y = y.loc[X.index]

    # Classifier
    clf = Pipeline([
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ])
    clf.fit(X_fit.values, y_fit.values)
    # Calibration
    val_probs_np = clf.predict_proba(X_cal.values)
    cg = ConformalGate(coverage_target=coverage)
    # Use logits = log(prob) for conformal calibration
    val_logits_t = torch.log(torch.clamp(torch.tensor(val_probs_np, dtype=torch.float32), min=1e-9))
    val_labels_t = torch.tensor(y_cal.values.astype(int), dtype=torch.long)
    cg.fit(val_logits=val_logits_t, val_labels=val_labels_t)
    # Expected return model: linear fit of horizon return vs prob_long on calib
    ret_h = train_df["close"].shift(-horizon) / train_df["close"] - 1.0
    ret_cal = ret_h.loc[X_cal.index].fillna(0.0).values
    p_long_cal = val_probs_np[:, 1]
    lin = LinearRegression().fit(p_long_cal.reshape(-1, 1), ret_cal.reshape(-1, 1))
    # Slippage model on train using vol -> slip bps
    vol_feat = train_df["close"].pct_change().rolling(vol_slip_window, min_periods=1).std().fillna(0.0).values.reshape(-1, 1)
    y_bps = np.abs(train_df["close"].pct_change().fillna(0.0).values) * 10_000.0
    sizes = np.ones_like(y_bps)
    im = ImpactModel(robust=False)
    N = len(vol_feat)
    idx = np.arange(10, N, max(1, N // 200_000))
    im.fit(vol_feat[idx], y_bps[idx], sizes[idx])

    # Inference on test
    X_te = build_features(test_df)
    # Optional session filter on test
    if session != "all":
        hrs_te = test_df.index.hour.values
        if session == "day":
            mask_te = (hrs_te >= 7) & (hrs_te < 20)
        elif session == "us":
            mask_te = (hrs_te >= 13) & (hrs_te < 22)
        elif session == "asia":
            mask_te = (hrs_te >= 0) & (hrs_te < 9)
        else:
            mask_te = np.ones_like(hrs_te, dtype=bool)
        X_te = X_te.loc[mask_te]
        test_df = test_df.loc[X_te.index]
    probs = clf.predict_proba(X_te.values)
    actions = []
    close_te = test_df["close"].values
    # cost thresholds
    fee = fee_bp / 10_000.0
    slip_bps = im.predict(test_df["close"].pct_change().rolling(vol_slip_window, min_periods=1).std().fillna(0.0).values.reshape(-1, 1), 1.0)
    slip = np.maximum(slip_bps, slip_floor_bp) / 10_000.0
    for i in range(len(X_te)):
        # Conformal decision on logits
        res = cg.decide(logits=np.log(np.clip(probs[i], 1e-9, 1.0)))
        if res.abstain:
            actions.append(0)
            continue
        p_long = probs[i, 1]
        if p_long < p_min:
            actions.append(0)
            continue
        exp_ret = float(lin.predict(np.array([[p_long]]))[0, 0])
        cost = float(fee + (slip[i] if i < len(slip) else 0.0))
        actions.append(1 if (exp_ret - cost) > 0 and (exp_ret >= edge_margin * cost) else 0)

    acts = np.array(actions, dtype=np.int8)
    # Evaluate
    ret = np.diff(close_te) / close_te[:-1]
    pnl = acts[:-1] * ret
    turn = np.abs(np.diff(acts))
    fees = (turn > 0).astype(float) * fee
    # entries metric: count 0->1 transitions plus initial entry
    entries = int((acts[0] == 1)) + int(((acts[1:] - acts[:-1]) == 1).sum()) if len(acts) > 1 else int(acts[0] == 1)
    net = pnl - fees
    sharpe = float(net.mean() / (net.std() + 1e-12) * np.sqrt(365 * 24 * 60))
    res = {
        "n_trades": int((turn > 0).sum()),
        "ret": float(net.sum()),
        "sharpe": sharpe,
        "coverage": float((acts != 0).mean()),
        "calib_ECE": cg.reliability_diagram(val_logits_t, val_labels_t, bins=10)["ECE"],
        "entries": int(entries),
    }
    return res


@app.command("run")
def run(
    feat_stream_dir: str = typer.Option(...),
    out_dir: str = typer.Option("./out/walk"),
    max_rows: int = typer.Option(200_000),
    train_frac: float = typer.Option(0.7),
    horizon: int = typer.Option(240, help="Horizon in bars for triple-barrier and return mapping."),
    atr_tp: float = typer.Option(1.5),
    atr_sl: float = typer.Option(1.0),
    coverage: float = typer.Option(0.8),
    fee_bp: float = typer.Option(5.0),
    n_folds: int = typer.Option(2, help="Number of walk-forward folds over the selected window."),
    p_min: float = typer.Option(0.55, help="Minimum probability threshold to consider LONG after conformal."),
    slip_floor_bp: float = typer.Option(2.0, help="Minimum slippage floor in bps."),
    session: str = typer.Option("all", help="Session filter: all,day,us,asia."),
    edge_margin: float = typer.Option(1.0, help="Require expected edge >= edge_margin * cost."),
):
    os.makedirs(out_dir, exist_ok=True)
    df = _load_features(feat_stream_dir)
    if max_rows and len(df) > max_rows:
        df = df.iloc[-max_rows:]
    L = len(df)
    test_len = int(L * (1.0 - train_frac))
    train_len = L - test_len
    # Build folds by sliding test window backward
    fold_results: List[Dict] = []
    for k in range(n_folds):
        te_end = L - k * test_len
        te_start = max(train_len - k * test_len, 0)
        tr_start = 0
        tr_end = te_start
        if te_end - te_start < horizon + 10 or tr_end - tr_start < horizon + 50:
            continue
        tr_df = df.iloc[tr_start:tr_end]
        te_df = df.iloc[te_start:te_end]
        res = evaluate_fold(
            tr_df,
            te_df,
            horizon=horizon,
            atr_tp=atr_tp,
            atr_sl=atr_sl,
            coverage=coverage,
            fee_bp=fee_bp,
            p_min=p_min,
            slip_floor_bp=slip_floor_bp,
            session=session,
            edge_margin=edge_margin,
        )
        res.update({
            "fold": k,
            "train_span": f"{tr_df.index[0]} -> {tr_df.index[-1]}",
            "test_span": f"{te_df.index[0]} -> {te_df.index[-1]}",
        })
        fold_results.append(res)

    # Aggregate
    if not fold_results:
        raise RuntimeError("No valid folds produced. Try adjusting max_rows or horizon.")
    agg = {
        "folds": fold_results,
        "avg_sharpe": float(np.mean([r["sharpe"] for r in fold_results])),
        "avg_ret": float(np.mean([r["ret"] for r in fold_results])),
        "avg_trades": float(np.mean([r["n_trades"] for r in fold_results])),
        "avg_coverage": float(np.mean([r["coverage"] for r in fold_results])),
    }
    with open(os.path.join(out_dir, "walk_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    typer.echo(f"Walk-forward complete. Summary -> {os.path.join(out_dir, 'walk_summary.json')}")


def main():  # pragma: no cover
    app()
