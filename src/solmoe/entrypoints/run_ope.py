import typer


def main(
    buffer_path: str = typer.Option(...),
    policy_ckpt: str = typer.Option(...),
    out_path: str = typer.Option(...),
):
    raise NotImplementedError("OPE runner not implemented in this scaffold.")

