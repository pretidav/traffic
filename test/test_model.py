"""Checks of the model. Run with ``python test_model.py`` or ``pytest`` (a few seconds)."""
import copy

import numpy as np

from model import SingleLane


def reference_step(x, v, L, vmax, p, rnd):
    """The four rules of the paper written independently of the class (periodic road, x sorted)."""
    d = (np.roll(x, -1) - x) % L
    d[d == 0] = L
    v = np.where((v < vmax) & (d > v + 1), v + 1, v)  # acceleration
    v = np.minimum(v, d - 1)  # slowing down
    v = np.where(rnd < p, np.maximum(v - 1, 0), v)  # randomization
    x = (x + v) % L  # car motion
    order = np.argsort(x)
    return x[order], v[order]


def test_rules_against_reference():
    k = 0
    for L in (20, 50):
        for vmax in (1, 3, 5):
            for p in (0.0, 0.3, 0.5):
                for N in (1, 2, L // 4, L // 2, L - 1):
                    lane = SingleLane(L=L, N=N, p=p, vmax=vmax, seed=k)
                    k += 1
                    rng = copy.deepcopy(lane.rng)  # the same random numbers for the two
                    x, v = lane.x.copy(), lane.v.copy()
                    for _ in range(100):
                        lane.update()
                        x, v = reference_step(x, v, L, vmax, p, rng.random(N))
                        assert np.array_equal(lane.x, x) and np.array_equal(lane.v, v)


def test_no_collisions_and_speed_limits():
    lane = SingleLane(L=60, N=25, p=0.3, vmax=5, seed=3)
    for _ in range(2000):
        lane.update()
        assert len(np.unique(lane.x)) == lane.N
        assert lane.v.min() >= 0 and lane.v.max() <= 5


def test_seed_makes_runs_reproducible():
    a, b, c = (SingleLane(L=100, N=20, seed=s) for s in (7, 7, 8))
    for _ in range(300):
        a.update(), b.update(), c.update()
    assert np.array_equal(a.x, b.x) and np.array_equal(a.v, b.v)
    assert not np.array_equal(a.x, c.x)


def test_flow_through_a_bond_is_rho_times_mean_speed():
    lane = SingleLane(L=200, N=30, p=0.5, seed=1)
    lane.relax()
    start, T, sum_v = lane.flow, 100000, 0.0
    for _ in range(T):
        lane.update()
        sum_v += lane.v.sum()
    assert abs((lane.flow - start) / T - sum_v / (T * 200)) < 1e-3


def test_deterministic_limit():
    """p = 0: q = min(vmax rho, 1 - rho)."""
    for rho in (0.05, 0.1, 0.2, 0.4):
        m = SingleLane(L=300, N=int(300 * rho), p=0.0, vmax=5, seed=1).measure(T=5000)
        assert abs(m['q'] - min(5 * m['rho'], 1 - m['rho'])) < 1e-3


def test_vmax_one_exact_result():
    """vmax = 1, parallel update: q = (1 - sqrt(1 - 4 (1-p) rho (1-rho))) / 2."""
    p = 0.3
    for rho in (0.1, 0.5, 0.7):
        m = SingleLane(L=400, N=int(400 * rho), p=p, vmax=1, seed=1).measure(T=30000)
        r = m['rho']
        assert abs(m['q'] - (1 - np.sqrt(1 - 4 * (1 - p) * r * (1 - r))) / 2) < 1e-3


def test_open_road_conserves_cars():
    lane = SingleLane(L=200, N=0, p=0.5, bc='open', seed=2)
    for _ in range(3000):
        n = lane.N
        lane.update()
        assert lane.N == n + lane.last_inflow - lane.last_flow
        assert len(np.unique(lane.x)) == lane.N
        assert lane.N == 0 or lane.x.max() < 200 - 6  # nobody stays in the exit zone


def test_bottleneck_of_the_paper():
    """Sect. 4 of the paper: flow 0.304 +- 0.001 and density 0.069 +- 0.002 (vmax = 5, p = 0.5)."""
    lane = SingleLane(L=1000, N=0, p=0.5, bc='open', seed=0)
    m = lane.measure(T=300000, profile=True)
    assert abs(m['q_out'] - 0.304) < 0.003
    assert abs(m['profile'][250:750].mean() - 0.069) < 0.004


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('ok  ', name)
