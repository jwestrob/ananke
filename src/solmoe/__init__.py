"""
SOL-MoE-RL: SOL-first, MoE + bilevel offline RL trading research stack.

This package provides ingestion, feature engineering, modeling, RL training,
calibration, memory retrieval, risk & execution, monitoring, backtesting,
simulation, OPE, and governance utilities.

All networked ingestion requires user-provided credentials; no synthetic data
is generated. See README and .env.example for required environment variables.
"""

__all__ = [
    "cli",
]

