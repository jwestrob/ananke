from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class BookLevel:
    price: float
    size: float
    ahead: float = 0.0  # queue volume ahead of us at this level


class FillEngine:
    """
    Deterministic event-driven fill engine with queue position and partial fills.
    """

    def __init__(self, latency_ms: int = 50):
        self.latency_ms = latency_ms

    @staticmethod
    def _fill_at_level(avail: float, ahead: float, order_remaining: float) -> Tuple[float, float]:
        """Return (filled, remaining) at a single level given queue ahead."""
        # fraction available after queue
        eff = max(0.0, avail - ahead)
        fill = min(order_remaining, eff)
        return fill, order_remaining - fill

    def compute_fill(self, side: str, size: float, levels: list[BookLevel]) -> float:
        remaining = size
        filled = 0.0
        for lvl in levels:
            if remaining <= 0:
                break
            f, remaining = self._fill_at_level(lvl.size, lvl.ahead, remaining)
            filled += f
        return filled


def simulate(signal_path: str, lob_path: str, out_path: str, impact_model_path: str | None = None):
    """
    Event-driven LOB simulator stub. Requires user-provided signals and LOB snapshots.
    Validates I/O and runs a deterministic placeholder computation for now.
    """
    if not os.path.exists(signal_path) or not os.path.exists(lob_path):
        raise FileNotFoundError("Provide existing signal_path and lob_path")
    # Minimal I/O validation: ensure JSONL signals can be read
    with open(signal_path, "r") as f:
        _ = f.readline()
    # Basic result stub
    res: Dict[str, float | int] = {"orders": 0, "fills": 0, "pnl": 0.0}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f)

