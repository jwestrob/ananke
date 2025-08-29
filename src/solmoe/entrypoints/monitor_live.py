import typer


def main(
    signal_stream: str = typer.Option(...),
    calib_window: int = typer.Option(2000),
    throttle_cfg: str = typer.Option(...),
):
    raise NotImplementedError("Live monitor not implemented in this scaffold.")

