from __future__ import annotations

import pandas as pd


def load_events(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ts" not in df.columns:
        raise ValueError("events CSV must have 'ts' column")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def features_for(ts):
    # Placeholder: expose a causal feature hook for events aligned at t0
    return {"event_count": 0}

