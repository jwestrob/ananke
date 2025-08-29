from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Top2Gate(nn.Module):
    def __init__(self, d_in: int, k: int, temperature: float = 1.0):
        super().__init__()
        self.w = nn.Linear(d_in, k)
        self.temperature = temperature

    def forward(self, x: torch.Tensor):
        logits = self.w(x) / self.temperature
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1)
        mask = torch.zeros_like(probs).scatter(-1, top2.indices, 1.0)
        top2_probs = probs * mask
        top2_probs = top2_probs / (top2_probs.sum(-1, keepdim=True) + 1e-9)
        entropy = -(top2_probs * (top2_probs.clamp_min(1e-12).log())).sum(-1)
        return top2_probs, entropy, logits


def tempered_blend(logits_list: List[torch.Tensor], weights: torch.Tensor, temps: torch.Tensor):
    """Blend expert action logits: sum_i w_i * (logits_i / T_i).

    Args:
        logits_list: list of [B,A]
        weights: [B,K]
        temps: [K]
    Returns:
        blended logits [B,A]
    """
    assert len(logits_list) == weights.shape[-1] == temps.shape[-1]
    stacked = torch.stack([li / t for li, t in zip(logits_list, temps)], dim=-1)  # [B,A,K]
    return (stacked * weights.unsqueeze(1)).sum(-1)  # [B,A]

