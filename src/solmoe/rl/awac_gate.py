class AWACGateTrainer:
    """
    Advantage-Weighted Actor Critic to train the gate over the blended policy,
    with KL trust-region to a baseline. Placeholder for API compatibility.
    """

    def __init__(self, kl_eps: float = 0.1):
        self.kl_eps = kl_eps

    def fit(self, expert_dir: str, buffer_path: str, mixture_ckpt: str, out_dir: str) -> None:
        raise NotImplementedError("AWACGateTrainer.fit not implemented in this scaffold.")

