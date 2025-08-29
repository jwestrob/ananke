.PHONY: help install test
help:
	@echo "Install:        pip install -e ."
	@echo "Ingest spot:    python -m solmoe.cli ingest-spot --raw-dir ./data/raw"
	@echo "Ingest derivs:  python -m solmoe.cli ingest-derivs --raw-dir ./data/raw"
	@echo "On-chain:       python -m solmoe.cli ingest-onchain --raw-dir ./data/raw"
	@echo "Features:       python -m solmoe.cli build-features --raw-dir ./data/raw --out-dir ./data/features"
	@echo "Offline buf:    python -m solmoe.cli build-offline --feat-dir ./data/features --out-path ./data/offline/sol.buffer"
	@echo "Train experts:  python -m solmoe.cli train-experts --buffer-path ./data/offline/sol.buffer --kronos-ckpt /path/ckpt --out-dir ./out/experts"
	@echo "Gate (bandit):  python -m solmoe.cli train-gate-bandit --expert-dir ./out/experts --buffer-path ./data/offline/sol.buffer --out-dir ./out/gate_bandit"
	@echo "Mixture critic: python -m solmoe.cli train-mixture-critic --buffer-path ./data/offline/sol.buffer --out-dir ./out/mixcritic"
	@echo "Gate (AWAC):    python -m solmoe.cli train-gate-awac --expert-dir ./out/experts --buffer-path ./data/offline/sol.buffer --mixture-ckpt ./out/mixcritic --out-dir ./out/gate_awac"
	@echo "Inference:      python -m solmoe.cli run-inference --expert-dir ./out/experts --gate-ckpt ./out/gate_awac --feat-stream-dir ./data/features/stream --out-signal-path ./out/signals.jsonl"
	@echo "Backtest vec:   python -m solmoe.cli backtest-vector --signal-path ./out/signals.jsonl --prices-path ./data/features/prices.parquet --out-path ./out/backtests/vec.json"
	@echo "LOB sim:        python -m solmoe.cli simulate-lob --signal-path ./out/signals.jsonl --lob-path ./data/lob/coinbase_sol.parquet --out-path ./out/backtests/lob.json"
	@echo "OPE:            python -m solmoe.cli run-ope --buffer-path ./data/offline/sol.buffer --policy-ckpt ./out/gate_awac --out-path ./out/ope.json"
	@echo "Monitor:        python -m solmoe.cli monitor-live --signal-stream ./out/signals.jsonl --throttle-cfg ./configs/default.yaml"
install:
	pip install -e .
test:
	pytest -q

