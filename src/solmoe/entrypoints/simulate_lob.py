import typer

from solmoe.sim.lob import simulate


def main(
    signal_path: str = typer.Option(...),
    lob_path: str = typer.Option(...),
    out_path: str = typer.Option(...),
    impact_model_path: str = typer.Option(None),
):
    simulate(signal_path=signal_path, lob_path=lob_path, out_path=out_path, impact_model_path=impact_model_path)

