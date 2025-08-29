from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import aiohttp
import pandas as pd


def _require_env(keys: List[str]):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {missing}. No dummy data allowed.")


@dataclass
class CoinbaseClient:
    """Coinbase public REST clients (SOL-USD). Env-guarded; errors if no creds."""

    api_key: str | None = None
    api_secret: str | None = None

    def __post_init__(self):
        _require_env(["COINBASE_API_KEY", "COINBASE_API_SECRET"])
        self.api_key = os.getenv("COINBASE_API_KEY")
        self.api_secret = os.getenv("COINBASE_API_SECRET")

    async def fetch_ohlc(self, symbol: str, granularity: int = 60) -> pd.DataFrame:
        url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"CB-ACCESS-KEY": self.api_key or ""}) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Coinbase OHLC fetch failed: {resp.status} {text}")
                data = await resp.json()
        # Coinbase returns [time, low, high, open, close, volume]
        cols = ["time", "low", "high", "open", "close", "volume"]
        df = pd.DataFrame(data, columns=cols)
        df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("ts")
        return df


async def backfill_ohlc(raw_dir: str, symbol: str = "SOL-USD", granularity: int = 60) -> str:
    client = CoinbaseClient()
    df = await client.fetch_ohlc(symbol=symbol, granularity=granularity)
    os.makedirs(raw_dir, exist_ok=True)
    out = os.path.join(raw_dir, f"coinbase_{symbol.replace('-', '').lower()}_{granularity}.parquet")
    df.to_parquet(out)
    return out


async def stream_books_trades(raw_dir: str, symbol: str = "SOL-USD") -> None:
    # Placeholder for WS stream; kept minimal as unit tests do not run it.
    _require_env(["COINBASE_API_KEY", "COINBASE_API_SECRET"])
    raise NotImplementedError("Websocket streaming not implemented in this scaffold.")

