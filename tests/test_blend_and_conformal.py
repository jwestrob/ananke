import numpy as np
import torch

from solmoe.models.gating import tempered_blend, Top2Gate
from solmoe.calibration.conformal import ConformalGate


def test_tempered_blend_shapes():
    B, A, K = 8, 5, 3
    logits = [torch.randn(B, A) for _ in range(K)]
    w = torch.softmax(torch.randn(B, K), -1)
    T = torch.ones(K) * 2.0
    out = tempered_blend(logits, w, T)
    assert out.shape == (B, A)


def test_top2_gate_basic():
    B, D, K = 4, 7, 3
    x = torch.randn(B, D)
    g = Top2Gate(d_in=D, k=K)
    w, ent, logits = g(x)
    assert w.shape == (B, K)
    assert ent.shape == (B,)
    assert logits.shape == (B, K)


def test_conformal_api():
    # create a simple separable validation set
    torch.manual_seed(0)
    V, A = 64, 3
    val_logits = torch.randn(V, A)
    val_labels = torch.randint(0, A, (V,))
    cg = ConformalGate(coverage_target=0.8)
    cg.fit(val_logits, val_labels)
    assert hasattr(cg, "fit") and hasattr(cg, "decide")
    res = cg.decide(val_logits[0])
    assert isinstance(res.abstain, bool)

