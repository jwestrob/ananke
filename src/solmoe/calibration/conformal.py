from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


class TemperatureScaler:
    """
    Learn a single temperature parameter to calibrate softmax logits by
    minimizing negative log-likelihood on a validation set.

    Usage:
        ts = TemperatureScaler().fit(val_logits, val_labels)
        probs = ts.predict_proba(logits)
    """

    def __init__(self, init_temperature: float = 1.0, device: Optional[str] = None):
        self._log_t = torch.nn.Parameter(torch.tensor(math.log(init_temperature), dtype=torch.float32))
        self._fitted = False
        self._device = device

    @property
    def temperature(self) -> float:
        return float(self._log_t.detach().exp().cpu().item())

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.05, steps: int = 200) -> "TemperatureScaler":
        if not isinstance(val_logits, torch.Tensor):
            val_logits = torch.tensor(val_logits, dtype=torch.float32)
        if not isinstance(val_labels, torch.Tensor):
            val_labels = torch.tensor(val_labels, dtype=torch.long)
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._log_t = torch.nn.Parameter(self._log_t.detach().to(device))
        val_logits = val_logits.to(device)
        val_labels = val_labels.to(device)
        opt = torch.optim.LBFGS([self._log_t], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            T = self._log_t.exp().clamp(min=1e-3, max=100.0)
            loss = F.cross_entropy(val_logits / T, val_labels)
            loss.backward()
            return loss

        opt.step(closure)
        self._fitted = True
        return self

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)
        T = self._log_t.detach().exp().clamp(min=1e-3)
        return F.softmax(logits / T, dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen-Shannon divergence between distributions p and q along last dim."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (p.log() - m.log()), dim=-1)
    kl_qm = torch.sum(q * (q.log() - m.log()), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


@dataclass
class ConformalResult:
    abstain: bool
    p_action: float
    action: int
    set_size: int
    threshold: float
    probs: np.ndarray
    reliability: Dict[str, float]


class ConformalGate:
    """
    Temperature scaling + conformal predictive sets (classification).

    - Calibrates temperature on validation set.
    - Computes a quantile threshold over nonconformity scores (1 - p_true) to
      target desired coverage.
    - At inference, constructs a set S = {a: 1 - p_a <= qhat}. Abstain if the
      set is non-singleton or if disagreement across experts (JSD) exceeds a
      threshold, or if gate entropy is high (optional signal).
    """

    def __init__(self, coverage_target: float = 0.8):
        if not (0.0 < coverage_target < 1.0):
            raise ValueError("coverage_target must be in (0,1)")
        self.coverage_target = coverage_target
        self.scaler = TemperatureScaler()
        self.qhat: Optional[float] = None

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor) -> "ConformalGate":
        self.scaler.fit(val_logits, val_labels)
        with torch.no_grad():
            probs = self.scaler.predict_proba(val_logits)
            true_prob = probs.gather(-1, val_labels.long().unsqueeze(-1)).squeeze(-1)
            nc = 1.0 - true_prob  # nonconformity
            q = float(torch.quantile(nc, torch.tensor(1.0 - self.coverage_target)))
            self.qhat = max(1e-6, min(1.0, q))
        return self

    def reliability_diagram(self, logits: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> Dict[str, float]:
        probs = self.scaler.predict_proba(logits).detach().cpu().numpy()
        conf = probs.max(axis=-1)
        preds = probs.argmax(axis=-1)
        acc = (preds == labels.detach().cpu().numpy()).astype(float)
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        for i in range(bins):
            m = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1] if i < bins - 1 else conf <= bin_edges[i + 1])
            if m.any():
                ece += np.abs(acc[m].mean() - conf[m].mean()) * m.mean()
        return {"ECE": float(ece), "avg_conf": float(conf.mean()), "acc": float(acc.mean())}

    def decide(
        self,
        logits: torch.Tensor,
        expert_dists: Optional[torch.Tensor] = None,
        gate_entropy: Optional[torch.Tensor] = None,
        jsd_threshold: float = 0.2,
    ) -> ConformalResult:
        """
        Return ConformalResult with abstain flag and action probabilities.

        Arguments:
          - logits: [A] or [B,A]
          - expert_dists: optional [K,A] or [B,K,A] distributions per expert
          - gate_entropy: optional scalar or [B]
        """
        if self.qhat is None:
            raise RuntimeError("ConformalGate not fit. Call fit(val_logits, val_labels) first.")
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)
        single = logits.ndim == 1
        if single:
            logits = logits.unsqueeze(0)
        probs = self.scaler.predict_proba(logits)
        # Build conformal set per item
        nc_full = 1.0 - probs  # [B,A]
        set_mask = (nc_full <= self.qhat + 1e-12)
        set_size = set_mask.sum(dim=-1)

        # disagreement via average JSD across experts if provided
        abstain_jsd = torch.zeros(logits.shape[0], dtype=torch.float32)
        if expert_dists is not None:
            if not isinstance(expert_dists, torch.Tensor):
                expert_dists = torch.tensor(expert_dists, dtype=torch.float32)
            if expert_dists.ndim == 2:  # [K,A]
                expert_dists = expert_dists.unsqueeze(0)
            B, K, A = expert_dists.shape
            if K > 1:
                mix = expert_dists.mean(dim=1)
                js_list = []
                for k in range(K):
                    p = expert_dists[:, k, :]
                    q = mix
                    js_list.append(js_divergence(p, q))
                abstain_jsd = torch.stack(js_list, dim=1).mean(dim=1)

        ent_gate = torch.zeros(logits.shape[0], dtype=torch.float32)
        if gate_entropy is not None:
            if not isinstance(gate_entropy, torch.Tensor):
                gate_entropy = torch.tensor(gate_entropy, dtype=torch.float32)
            ent_gate = gate_entropy.float()

        abstain_mask = (set_size > 1) | (abstain_jsd > jsd_threshold) | (ent_gate > 1.0)
        top_p, top_idx = probs.max(dim=-1)
        result = ConformalResult(
            abstain=bool(abstain_mask[0].item()),
            p_action=float(top_p[0].item()),
            action=int(top_idx[0].item()),
            set_size=int(set_size[0].item()),
            threshold=float(self.qhat),
            probs=probs[0].detach().cpu().numpy(),
            reliability={
                "entropy": float((-probs[0] * probs[0].clamp_min(1e-12).log()).sum().item()),
                "jsd": float(abstain_jsd[0].item()),
            },
        )
        return result

