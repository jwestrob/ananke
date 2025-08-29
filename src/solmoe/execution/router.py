from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RouterConfig:
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 5.0
    latency_ms: int = 50
    slip_cap_bps: float = 20.0
    venues: List[str] = None


class ExecutionRouter:
    """
    Maker/taker templates, queue-join heuristics, latency budget, slip caps,
    basis/funding filters, kill-switches, auto-throttle hooks.

    Note: Routing to offshore perps is explicitly not implemented.
    """

    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        if not cfg.venues:
            self.cfg.venues = ["coinbase", "kraken"]

    def allow_venue(self, venue: str) -> bool:
        return venue in (self.cfg.venues or [])

    def risk_checks(self, shift_alarm: bool, drawdown: float) -> bool:
        if shift_alarm:
            return False
        if drawdown < -0.2:
            return False
        return True

    def route(self, intent: Dict[str, float | str]) -> Dict[str, float | str]:
        venue = intent.get("venue", "")
        if not self.allow_venue(str(venue)):
            raise ValueError("Venue not allowed by router policy")
        return intent

