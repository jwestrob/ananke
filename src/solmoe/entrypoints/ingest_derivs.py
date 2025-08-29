import typer

from solmoe.ingest.derivs import DerivsProvider


def main(raw_dir: str = typer.Option(...)):
    _ = DerivsProvider()  # will validate env and raise if missing
    raise NotImplementedError("Derivatives analytics ingestion not implemented in this scaffold.")

