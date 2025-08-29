import typer


def main(
    expert_dir: str = typer.Option(...),
    buffer_path: str = typer.Option(...),
    mixture_ckpt: str = typer.Option(...),
    out_dir: str = typer.Option(...),
    kl_eps: float = typer.Option(0.1),
):
    raise NotImplementedError("AWAC gate training not implemented in this scaffold.")

