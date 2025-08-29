import typer

from solmoe.datasets.offline import build_offline_buffer


def main(
    feat_dir: str = typer.Option(...),
    out_path: str = typer.Option(...),
    horizons: str = typer.Option("5m,30m,2h,1d"),
):
    hrs = [h.strip() for h in horizons.split(",") if h.strip()]
    build_offline_buffer(feat_dir=feat_dir, out_path=out_path, horizons=hrs)

