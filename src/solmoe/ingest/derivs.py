from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os
import pandas as pd


@dataclass
class DerivsProvider:
    """
    Provider-agnostic interface for OI, funding, liquidations.
    Requires DERIVS_API_KEY; does not execute or route orders.
    """

    api_key: str | None = None

    def __post_init__(self):
        key = os.getenv("DERIVS_API_KEY")
        if not key:
            raise EnvironmentError("DERIVS_API_KEY not set; provide a real key.")
        self.api_key = key

    def fetch_oi(self, start, end) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_funding(self, start, end) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_liquidations(self, start, end) -> pd.DataFrame:
        raise NotImplementedError

