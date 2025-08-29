import numpy as np

from solmoe.risk.slippage import ImpactModel
from solmoe.risk.sizing import kelly_fraction


def test_impact_model_monotonic_size():
    np.random.seed(0)
    N, D = 200, 2
    X = np.random.randn(N, D)
    true_alpha = 0.8
    kappa = 1e-5
    sizes = np.random.uniform(0.1, 5.0, size=N)
    # generate positive impact in bps
    y = kappa * (sizes ** true_alpha) * np.exp(0.1 * X[:, 0] - 0.05 * X[:, 1]) * 1e4
    im = ImpactModel(alpha_init=0.6, kappa_init=1e-6)
    im.fit(X, y, sizes)
    x0 = X[:10]
    s_small, s_big = np.array([0.5] * 10), np.array([2.0] * 10)
    pred_small = im.predict(x0, s_small)
    pred_big = im.predict(x0, s_big)
    assert (pred_big > pred_small).all()


def test_kelly_fraction_caps():
    k = kelly_fraction(mu_bp=10.0, q05_bp=5.0, vol_bp=50.0, cap=0.2)
    assert 0.0 <= k <= 0.2

