import typer

from solmoe.ingest.onchain import OnchainProvider


def main(raw_dir: str = typer.Option(...)):
    _ = OnchainProvider()  # validates keys
    raise NotImplementedError("On-chain ingestion not implemented in this scaffold.")

