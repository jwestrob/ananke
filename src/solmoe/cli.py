import typer
from solmoe.entrypoints import (
    ingest_spot,
    ingest_derivs,
    ingest_onchain,
    build_features,
    build_offline,
    train_experts,
    train_gate_bandit,
    train_gate_awac,
    train_mixture_critic,
    run_inference,
    backtest_vector,
    simulate_lob,
    run_ope,
    monitor_live,
    tune_baseline,
    tune_ensemble,
    evaluate_walkforward,
)

app = typer.Typer(add_completion=False, help="SOL-MoE-RL CLI entrypoints")

# Register commands with explicit names to match documentation/Makefile
app.command(name="ingest-spot")(ingest_spot.main)
app.command(name="ingest-derivs")(ingest_derivs.main)
app.command(name="ingest-onchain")(ingest_onchain.main)
app.command(name="build-features")(build_features.main)
app.command(name="build-offline")(build_offline.main)
app.command(name="train-experts")(train_experts.main)
app.command(name="train-gate-bandit")(train_gate_bandit.main)
app.command(name="train-gate-awac")(train_gate_awac.main)
app.command(name="train-mixture-critic")(train_mixture_critic.main)
app.command(name="run-inference")(run_inference.main)
app.command(name="backtest-vector")(backtest_vector.main)
app.command(name="simulate-lob")(simulate_lob.main)
app.command(name="run-ope")(run_ope.main)
app.command(name="monitor-live")(monitor_live.main)
app.command(name="tune-baseline")(tune_baseline.run)
app.command(name="tune-ensemble")(tune_ensemble.run)
app.command(name="evaluate-walk")(evaluate_walkforward.run)

if __name__ == "__main__":
    app()
