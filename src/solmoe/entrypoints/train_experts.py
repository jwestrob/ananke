import typer

from solmoe.rl.iql_expert import IQLConfig, IQLExpertTrainer


def main(
    buffer_path: str = typer.Option(...),
    kronos_ckpt: str = typer.Option(...),
    out_dir: str = typer.Option(...),
    k: int = typer.Option(3),
):
    _ = IQLExpertTrainer(IQLConfig())
    raise NotImplementedError("Expert training not implemented in this scaffold.")

