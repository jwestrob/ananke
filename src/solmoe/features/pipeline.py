from __future__ import annotations

import os
from typing import Optional
import fnmatch

import numpy as np
import pandas as pd


def _shift_to_causal(df: pd.DataFrame, cols: list[str], bar: str) -> pd.DataFrame:
    """Shift exogenous cols so they are only available at the first fully observed bar."""
    shifted = df.copy()
    for c in cols:
        shifted[c] = shifted[c].shift(1)
    return shifted


def _resample_ohlc(df: pd.DataFrame, bar: str) -> pd.DataFrame:
    """Resample an OHLCV dataframe to the provided bar size."""
    if "ts" not in df.columns:
        if "time" in df.columns:
            df = df.copy()
            df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
        else:
            raise ValueError("Expected a 'ts' or 'time' column for timestamps.")
    df = df.set_index(pd.to_datetime(df["ts"], utc=True)).sort_index()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    res = df.resample(bar).agg(agg)
    return res.dropna(how="all")


def build_features(
    raw_dir: str,
    out_dir: str,
    bar: str,
    use_onchain: bool,
    use_derivs: bool,
    tokenizer: Optional[object] = None,
    file_glob: Optional[str] = None,
    symbol: Optional[str] = None,
) -> None:
    """
    Canonical bar resampling; dual-timescale summaries; Kronos tokenization;
    joins exogenous features with causal shift to avoid leakage.
    
    Now supports:
    - Trades with columns ['ts','price'] (Parquet)
    - OHLC(V) with ['ts','open','high','low','close'(,'volume')] (Parquet)
    - Kraken OHLCVT CSV dumps with schema: time,open,high,low,close,volume,count
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")
    os.makedirs(out_dir, exist_ok=True)

    files = [p for p in os.listdir(raw_dir) if p.endswith((".parquet", ".csv"))]
    if file_glob:
        files = [p for p in files if fnmatch.fnmatch(p, file_glob)]
    if symbol:
        sym = symbol.replace("-", "")
        sym_upper = sym.upper()
        sym_lower = sym.lower()
        filt: list[str] = []
        for p in files:
            name = p
            if p.endswith(".csv") and name.upper().startswith(f"{sym_upper}_"):
                filt.append(p)
            elif p.endswith(".parquet") and (f"_{sym_lower}_" in name.lower()):
                filt.append(p)
        files = filt
    if not files:
        raise FileNotFoundError("No raw files (.parquet or .csv) in raw_dir. Provide real data.")

    dfs = []
    for p in files:
        path = os.path.join(raw_dir, p)
        try:
            if p.endswith(".parquet"):
                df = pd.read_parquet(path)
                # trades schema
                if {"ts", "price"}.issubset(df.columns):
                    s = df.copy()
                    s["ts"] = pd.to_datetime(s["ts"], utc=True)
                    s = s.set_index("ts").sort_index()["price"].resample(bar).ohlc()
                    dfs.append(s)
                    continue
                # OHLC schema
                if {"open", "high", "low", "close"}.issubset(df.columns) and ("ts" in df.columns or "time" in df.columns):
                    dfs.append(_resample_ohlc(df, bar))
                    continue
                # unknown parquet schema; skip
                continue
            else:  # CSV (Kraken OHLCVT)
                # Kraken dumps have no header; 7 columns: time,open,high,low,close,volume,count
                df = pd.read_csv(
                    path,
                    header=None,
                    names=["time", "open", "high", "low", "close", "volume", "count"],
                )
                df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
                dfs.append(_resample_ohlc(df, bar))
        except Exception as e:
            # Skip files that do not match expected schemas
            # (Intentionally silent to allow mixed dirs; users should scope raw_dir when possible.)
            continue

    if not dfs:
        raise ValueError(
            "No compatible raw files found. Expect Parquet with ['ts','price'] or OHLC, or Kraken OHLCVT CSVs."
        )

    prices = pd.concat(dfs, axis=0).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    # Ensure required column 'close' exists for downstream buffer builder
    if "close" not in prices.columns:
        raise ValueError("Resampled data must include 'close' column")

    # Dual timescales (simple rolling stats on close)
    close = prices["close"].astype(float)
    fast = close.rolling(window=5, min_periods=1).agg(["mean", "std"]).rename(
        columns={"mean": "fast_mean", "std": "fast_std"}
    )
    slow = close.rolling(window=60, min_periods=1).agg(["mean", "std"]).rename(
        columns={"mean": "slow_mean", "std": "slow_std"}
    )
    feats = pd.concat([prices, fast, slow], axis=1).ffill()

    # Exogenous joins (derivs/onchain) are expected as separate Parquet files with timestamps
    exo_cols: list[str] = []
    if use_derivs:
        derivs_path = os.path.join(raw_dir, "derivs.parquet")
        if not os.path.exists(derivs_path):
            raise FileNotFoundError("use_derivs=True but derivs.parquet not found")
        d = pd.read_parquet(derivs_path).set_index("ts")
        d.index = pd.to_datetime(d.index, utc=True)
        feats = feats.join(d, how="left")
        exo_cols += [c for c in d.columns]
    if use_onchain:
        onchain_path = os.path.join(raw_dir, "onchain.parquet")
        if not os.path.exists(onchain_path):
            raise FileNotFoundError("use_onchain=True but onchain.parquet not found")
        o = pd.read_parquet(onchain_path).set_index("ts")
        o.index = pd.to_datetime(o.index, utc=True)
        feats = feats.join(o, how="left")
        exo_cols += [c for c in o.columns]

    # Causal shift of exogenous
    if exo_cols:
        feats = _shift_to_causal(feats, exo_cols, bar)

    # Optional Kronos tokenization if tokenizer provided
    if tokenizer is not None:
        if not hasattr(tokenizer, "encode"):
            raise ValueError("Provided tokenizer lacks 'encode' method")
        tokens = feats[["open", "high", "low", "close"]].dropna().values.astype(np.float32)
        _ = tokenizer.encode(tokens)  # noqa: F841

    feats.to_parquet(os.path.join(out_dir, "features.parquet"))
