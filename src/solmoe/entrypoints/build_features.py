import typer

from solmoe.features.pipeline import build_features as _build


def main(
    raw_dir: str = typer.Option(...),
    out_dir: str = typer.Option(...),
    bar: str = typer.Option("1min"),
    use_onchain: bool = typer.Option(False),
    use_derivs: bool = typer.Option(False),
    file_glob: str | None = typer.Option(None, help="Optional filename glob to filter inputs, e.g. 'SOLUSD_*.csv'"),
    symbol: str | None = typer.Option(None, help="Optional trading pair symbol to filter (e.g., SOLUSD)."),
):
    _build(
        raw_dir=raw_dir,
        out_dir=out_dir,
        bar=bar,
        use_onchain=use_onchain,
        use_derivs=use_derivs,
        file_glob=file_glob,
        symbol=symbol,
    )
