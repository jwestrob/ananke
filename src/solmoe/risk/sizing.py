from __future__ import annotations

def kelly_fraction(mu_bp: float, q05_bp: float, vol_bp: float, cap: float = 0.2) -> float:
    """
    Kelly-capped position sizing using conservative quantiles.

    Args:
        mu_bp: expected edge in basis points
        q05_bp: conservative 5th percentile edge in bps (risk-averse)
        vol_bp: volatility proxy in bps (std dev)
        cap: hard cap on Kelly fraction
    Returns:
        clipped Kelly fraction in [0, cap]
    """
    # Convert bps to fraction returns
    mu = mu_bp / 10_000.0
    q05 = q05_bp / 10_000.0
    vol = max(1e-8, vol_bp / 10_000.0)
    conservative_mu = min(mu, q05)
    kelly = conservative_mu / (vol**2)
    return float(max(0.0, min(cap, kelly)))

