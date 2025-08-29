from __future__ import annotations

import os
import torch
import torch.nn as nn


class KronosLikeEncoder(nn.Module):
    """
    Wrap an external Kronos checkpoint/tokenizer.
    Expose: forward(tokens)->[B,T,D], embed(tokens)->[B,D].
    Do NOT fabricate tokens: tokenizer must be provided or raise.
    """

    def __init__(self, ckpt_path: str, freeze: bool = True):
        super().__init__()
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                "KronosLikeEncoder requires a valid checkpoint/tokenizer path. Provided: %s" % ckpt_path
            )
        # Placeholder for integration hook
        # In a real implementation, load tokenizer and encoder weights here.
        # For now, define a simple linear projection as a placeholder encoder that
        # still enforces presence of an external checkpoint.
        self.proj = nn.Linear(32, 64, bias=False)
        if freeze:
            for p in self.proj.parameters():
                p.requires_grad = False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.size(-1) != 32:
            raise ValueError("Expected tokens with shape [B,T,32] for placeholder encoder")
        return self.proj(tokens)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.forward(tokens)
        return x.mean(dim=1)

