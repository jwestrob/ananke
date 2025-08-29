from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class IQLConfig:
    expectile: float = 0.8
    gamma: float = 0.997
    tau: float = 0.005
    lr: float = 3e-4
    behavior_kl: float = 1e-3
    batch_size: int = 512
    steps: int = 200_000
    device: str = "cuda"
    horizons: Optional[List[str]] = None


class IQLExpertTrainer:
    """
    IQL for discrete actions; frozen encoder + LoRA adapters per expert; behavior-KL regularization.
    Expectile value, advantage-weighted policy updates, EMA targets; distributional heads optional.

    This is a placeholder interface; full training implementation is out of scope
    for unit tests but the API is provided for integration.
    """

    def __init__(self, cfg: IQLConfig):
        self.cfg = cfg

    def fit(self, buffer_path: str, encoder_ckpt: str, out_dir: str) -> None:
        raise NotImplementedError(
            "IQLExpertTrainer.fit is a research training loop not exercised by unit tests."
        )

