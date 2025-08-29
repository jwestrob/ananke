from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class OnchainProvider:
    """
    Helius/Flipside/Jupiter interfaces (DAS/transactions/DEX flows).
    Env-guarded; errors if keys missing.
    """

    helius_key: str | None = None
    flipside_key: str | None = None
    jupiter_key: str | None = None

    def __post_init__(self):
        hk = os.getenv("HELIUS_API_KEY")
        fk = os.getenv("FLIPSIDE_API_KEY")
        jk = os.getenv("JUPITER_API_KEY")
        if not (hk and fk and jk):
            raise EnvironmentError(
                "Missing one of HELIUS_API_KEY, FLIPSIDE_API_KEY, JUPITER_API_KEY. Provide real keys."
            )
        self.helius_key, self.flipside_key, self.jupiter_key = hk, fk, jk

