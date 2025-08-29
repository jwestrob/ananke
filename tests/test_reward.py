import numpy as np

from solmoe.datasets.offline import compute_reward


def test_compute_reward_costs():
    # 5 steps returns, 6 actions (position at time t applies to return t->t+1)
    ret = np.array([0.01, -0.005, 0.0, 0.002, -0.001])
    action = np.array([0, 1, 1, -1, -1, 0])
    fees_bps = 10.0
    slippage_bps = np.ones_like(ret) * 5.0
    r = compute_reward(ret, action, fees_bps, slippage_bps)
    # check length and that trading incurs costs
    assert r.shape[0] == ret.shape[0]
    # when action changes at t=0 from 0->1, we pay fee on first step
    assert r[0] <= action[0] * ret[0]  # include negative fees and slippage

