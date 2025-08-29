import asyncio
import os
import typer

from solmoe.ingest.coinbase import backfill_ohlc as cb_backfill
from solmoe.ingest.kraken import KrakenClient


def main(
    raw_dir: str = typer.Option(...),
    venues: str = typer.Option("coinbase,kraken"),
    symbol: str = typer.Option("SOL-USD", help="Trading pair for spot data (Coinbase 'BASE-QUOTE', Kraken 'BASEQUOTE')."),
    granularity: int = typer.Option(60, help="Coinbase candle granularity in seconds (e.g., 60 for 1m)."),
):
    v = [x.strip() for x in venues.split(",") if x.strip()]
    if "coinbase" in v:
        asyncio.run(cb_backfill(raw_dir, symbol=symbol, granularity=granularity))
    if "kraken" in v:
        # Optional fetch preview to confirm credentials/symbol work; writing left to users or features pipeline via CSV.
        # To keep the scaffold minimal/safe, we do not write Kraken here.
        # You can point build-features at the Kraken CSV dump instead.
        kr_symbol = symbol.replace("-", "")
        _ = KrakenClient()  # validates env
        # Uncomment to quickly check connectivity:
        # import asyncio
        # df = asyncio.run(_.fetch_ohlc(symbol=kr_symbol, interval=1))
        raise NotImplementedError("Kraken backfill writing not implemented; use Kraken CSV or extend this path.")
