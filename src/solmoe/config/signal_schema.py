from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, Literal

Action = Literal["LONG", "FLAT", "SHORT"]
Horizon = Literal["5m", "30m", "2h", "1d"]


class Signal(BaseModel):
    ts: str
    action: Action
    horizon: Horizon
    p_action: float = Field(ge=0.0, le=1.0)
    rtg: float  # expected net bps
    q: Dict[str, float]  # quantiles {"p05":..., "p50":..., "p95":...}
    risk_gate: Dict[str, float | bool]  # {"vol":..., "cvar":..., "abstain": bool}
    explain: Dict[str, object]  # {"regime": str, "drivers": list[str], "experts": Dict[str,float]}

