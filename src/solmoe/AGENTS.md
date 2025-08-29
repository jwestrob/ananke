# SOL‑MoE‑RL Agent Guide

This document orients code agents (and humans) to the repository structure, conventions, and extension points. It emphasizes strict time alignment, reproducibility, and “no dummy data”.

## Purpose
SOL‑first trading research stack combining:
- Ingestion: spot (Coinbase/Kraken), optional derivatives analytics, optional Solana on‑chain/DEX.
- Features: canonical bar resampling (default 1m), dual‑timescale summaries, Kronos‑compatible K‑line tokenization.
- Modeling: small‑K MoE (trend/mean‑revert/squeeze/crash) with shared frozen encoder + LoRA adapters; top‑2 gating, tempered blending, disagreement‑aware abstention.
- RL: bilevel offline RL (IQL experts → bandit/AWAC gate with conservative mixture critic).
- Calibration: temperature scaling + conformal sets + selective prediction.
- Memory: motif retrieval with FAISS/sklearn fused as logit prior.
- Invariance: domain‑adversarial GRL over {venue, session, month}.
- Risk & Execution: Kelly‑capped sizing, slippage/impact, throttles, kill‑switches.
- Evaluation: vectorized replay, event‑driven LOB sim, OPE (FQE + WIS + DR).
- Governance: vault with dataset/reward/simulator hashing, canary path, Hydra configs.

## Key Guarantees
- No synthetic data paths: ingestion and pipelines error without real credentials/files.
- Strict causality: exogenous sources (funding/OI/liqs/on‑chain) are shifted to first fully observable bar.
- Reproducibility: Hydra configs, typed APIs, deterministic unit tests.

## Layout
- `src/solmoe/cli.py`: Typer app wiring all entrypoints.
- `src/solmoe/config/`: schemas and config utilities.
- `src/solmoe/ingest/`: env‑guarded Coinbase/Kraken/Derivs/On‑chain interfaces.
- `src/solmoe/features/`: canonical bar + dual timescale + tokenizer hooks, causal joins.
- `src/solmoe/models/`: Kronos‑like encoder wrapper, gating utilities, GRL.
- `src/solmoe/rl/`: IQL experts, conservative mixture critic (CQL‑discrete), AWAC gate (APIs scaffolded).
- `src/solmoe/calibration/`: temperature scaling + conformal predictive sets with abstention.
- `src/solmoe/memory/`: FAISS/sklearn motif memory with nearest‑neighbor retrieval.
- `src/solmoe/risk/`: impact/slippage model and Kelly‑capped sizing.
- `src/solmoe/monitoring/`: shift detectors (MMD/energy) + SPRT calibration guard.
- `src/solmoe/backtest/`: vectorized replay (cost‑aware).
- `src/solmoe/sim/`: event‑driven LOB simulator (queueing, partial fills core included).
- `src/solmoe/ope/`: OPE estimators (FQE + clipped WIS + DR stubs).
- `src/solmoe/execution/`: router with venue allow‑listing, risk checks, slip caps.
- `src/solmoe/governance/`: policy vault with artifact hashing.
- `src/solmoe/events/`: causal event loaders.
- `configs/default.yaml`: default Hydra‑style configuration.
- `tests/`: focused unit tests for core logic (blending, conformal API, reward, slippage, LOB fills).

## Conventions
- Python ≥3.10, strong typing, docstrings on public APIs.
- Typer CLI; Hydra‑style YAML config structure (no global state).
- GPU‑friendly training assumptions (bf16, grad checkpointing, LoRA via PEFT). Training loops are scaffolded and meant to be filled.
- Encoders must be real: `KronosLikeEncoder` constructor enforces an external checkpoint/tokenizer path.
- Ingestion is provider‑backed only, guarded by environment variables (see `.env.example`).

## Entry Points (Typer)
Run via `python -m solmoe.cli`:
- `ingest-spot`, `ingest-derivs`, `ingest-onchain`
- `build-features`, `build-offline`
- `train-experts`, `train-gate-bandit`, `train-gate-awac`, `train-mixture-critic`
- `run-inference`, `backtest-vector`, `simulate-lob`, `run-ope`, `monitor-live`

Many heavy paths intentionally raise `NotImplementedError` in the scaffold; wire your research code where indicated.

## Data & Time Alignment
- Raw spot/LOB data must be supplied by the user. Derivatives/on‑chain data must be env‑backed and causally shifted.
- Feature joins must not leak future information; exogenous columns are shifted one bar by default.

## Testing
- `pytest` with fast, deterministic unit tests. Critical invariants covered:
  - Tempered blending + top‑2 gating shapes.
  - Conformal API + abstention contract.
  - Reward construction costs (fees, slippage).
  - Slippage model monotonicity in order size.
  - LOB fill edge cases (queueing, partial fills).

## Extension Points
- Replace placeholder encoder projection with your Kronos tokenizer/encoder.
- Implement IQL/CQL/AWAC trainers in `src/solmoe/rl/*` respecting the existing APIs.
- Extend vectorized backtest and LOB simulator for full fee/latency/queue dynamics.
- Add FAISS index building and retrieval‑prior fusion in inference.

## Operational Notes
- Credentials: set in environment (see `.env.example`).
- Safety: No routing to offshore perps; derivs integrations are analytics only.
- Governance: Use `PolicyVault` to register artifacts with dataset/reward/simulator hashes; use canary params from config for deployments.

## Agent Hand‑Off
- `CLAUDE.md` is a symlink to this guide to support swapping between agent UIs.
- Keep changes minimal and surgical; prefer adding tests for new logic.
- Do not add synthetic data or uncontrolled randomness.

## Baseline (MA) Tuning
We include a simple moving‑average baseline with hysteresis, min‑hold, and optional long‑only, session, and volatility filters.

Entrypoints:
- Generate signals: `python -m solmoe.cli run-inference --help`
- Grid search tuner: `python -m solmoe.cli tune-baseline --help`

Reference best config (from a recent SOLUSD 1‑min run on Kraken CSV features; long‑only):
- fast_window: 60
- slow_window: 360
- band_bp: 50.0
- exit_bp: 50.0
- min_hold: 480
- long_only: true
- vol_window: 120
- vol_thr_bp: 20.0
- session: all

Notes:
- These parameters were selected by in‑sample grid search; re‑tune per dataset/period and validate on holdout.
- Use `--file-glob 'SOLUSD_1.csv'` (and/or `--symbol SOLUSD`) in `build-features` to avoid mixing intervals.
