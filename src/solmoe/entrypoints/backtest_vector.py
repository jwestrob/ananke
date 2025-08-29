import typer

from solmoe.backtest.vectorized import run_vectorized


def main(
    signal_path: str = typer.Option(...),
    prices_path: str = typer.Option(...),
    out_path: str = typer.Option(...),
):
    run_vectorized(signal_path=signal_path, prices_path=prices_path, out_path=out_path)

