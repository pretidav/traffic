"""Checks of the two-lane model. Run with ``python test_model_multiple.py`` or ``pytest`` (some seconds)."""
import copy

import numpy as np

from model import SingleLane
from model_multiple import LEFT, RIGHT, TwoLane
from test_model import reference_step


def set_state(sim, right, left):
    """right, left: lists of (position, speed)."""
    for lane, cars in ((RIGHT, right), (LEFT, left)):
        cars = sorted(cars)
        sim.x[lane] = np.array([c[0] for c in cars], dtype=int)
        sim.v[lane] = np.array([c[1] for c in cars], dtype=int)


def reference_lane_change(x, v, L, vmax, lookback, pchange, rnd):
    """The rules of the paper written car by car with loops, independently of the class.

    x, v: lists of two dicts-free arrays (right, left). rnd: two arrays of random numbers.
    Returns the lists of (position, speed) of the two lanes after the lane change."""
    occ = [set(x[0]), set(x[1])]
    new = [[], []]
    for lane in (0, 1):
        other = 1 - lane
        for i, (xi, vi) in enumerate(zip(x[lane], v[lane])):
            gap = next(d - 1 for d in range(1, L + 1) if (xi + d) % L in occ[lane])  # alone: L - 1
            go = next((d - 1 for d in range(0, L) if (xi + d) % L in occ[other]), L)  # from the site itself
            gb = next((d - 1 for d in range(0, L) if (xi - d) % L in occ[other]), L)
            ok = go > vi + 1 and gb > lookback and rnd[lane][i] < pchange
            if lane == 0:
                ok = ok and gap < vi + 1
            new[other if ok else lane].append((xi, vi))
    return [sorted(c) for c in new]


def test_lane_change_against_reference():
    k = 0
    for L in (30, 60):
        for N in (2, L // 3, L, 2 * L - 5):
            for pchange in (1.0, 0.5):
                sim = TwoLane(L=L, N=N, p=0.5, pchange=pchange, seed=k)
                k += 1
                rng = copy.deepcopy(sim.rng)
                for _ in range(60):
                    x = [sim.x[0].copy(), sim.x[1].copy()]
                    v = [sim.v[0].copy(), sim.v[1].copy()]
                    # the random numbers drawn by lane_change: right lane first, then left lane
                    rnd = [rng.random(len(x[0])), rng.random(len(x[1]))]
                    expected = reference_lane_change(x, v, L, 5, 5, pchange, rnd)
                    sim.lane_change()
                    for lane in (0, 1):
                        got = list(zip(sim.x[lane].tolist(), sim.v[lane].tolist()))
                        assert got == [(int(a), int(b)) for a, b in expected[lane]]
                    # then the single lane update, from the same generator
                    for lane in (0, 1):
                        xs, vs = reference_step(sim.x[lane].copy(), sim.v[lane].copy(), L, 5, 0.5,
                                                rng.random(len(sim.x[lane])))
                        sim._advance(lane)
                        assert np.array_equal(sim.x[lane], xs) and np.array_equal(sim.v[lane], vs)


def test_the_rules_on_hand_made_situations():
    L = 100
    # 1. left lane, nobody around: the car goes back to the right lane (no T1 needed)
    sim = TwoLane(L=L, N=1, vmax=5)
    set_state(sim, [], [(10, 3)])
    sim.lane_change()
    assert sim.x[RIGHT].tolist() == [10] and sim.x[LEFT].tolist() == []
    # 2. right lane, free road ahead: it stays (T1 fails)
    set_state(sim, [(10, 3)], [])
    sim.lane_change()
    assert sim.x[RIGHT].tolist() == [10] and sim.x[LEFT].tolist() == []
    # 3. right lane, car ahead at gap 3 < v + 1 = 5: it moves to the free left lane (T1, T2, T3)
    set_state(sim, [(10, 4), (14, 0)], [])
    sim.lane_change()
    assert sim.x[LEFT].tolist() == [10] and sim.x[RIGHT].tolist() == [14]
    # 4. the same, but somebody on the left lane right beside: gap_o = -1, it stays
    set_state(sim, [(10, 4), (14, 0)], [(10, 5)])
    sim.lane_change()
    assert sim.x[RIGHT].tolist() == [10, 14] and sim.x[LEFT].tolist() == [10]
    # 5. the left lane is free ahead but too close behind (gap_back = 5, not > 5): it stays (T3)
    set_state(sim, [(10, 4), (14, 0)], [(4, 5)])
    sim.lane_change()
    assert sim.x[RIGHT].tolist() == [10, 14]
    #    one site farther behind (gap_back = 6 > 5): it goes
    set_state(sim, [(10, 4), (14, 0)], [(3, 5)])
    sim.lane_change()
    assert sim.x[LEFT].tolist() == [3, 10]
    # 6. car ahead on the left lane too close (gap_o = 4 = v + 1, not > v + 1): it stays (T2)
    set_state(sim, [(10, 4), (14, 0)], [(15, 5)])
    sim.lane_change()
    assert sim.x[RIGHT].tolist() == [10, 14]
    # 7. periodic: lookup across the end of the road
    set_state(sim, [(98, 4), (2, 0)], [])  # gap 3 across the end of the road, left lane empty
    sim.lane_change()
    assert sim.x[LEFT].tolist() == [98] and sim.x[RIGHT].tolist() == [2]
    # 8. left lane car with a right-lane car just beside: it stays on the left
    set_state(sim, [(10, 0)], [(10, 2)])
    sim.lane_change()
    assert sim.x[LEFT].tolist() == [10]


def test_conservation_no_collisions_and_limits():
    sim = TwoLane(L=80, N=60, p=0.3, seed=3)
    for _ in range(3000):
        sim.update()
        assert sim.N == 60
        for lane in (0, 1):
            assert len(np.unique(sim.x[lane])) == len(sim.x[lane])
            assert np.all(np.diff(sim.x[lane]) > 0)
            assert sim.v[lane].size == 0 or (sim.v[lane].min() >= 0 and sim.v[lane].max() <= 5)


def test_seed_makes_runs_reproducible():
    a, b, c = (TwoLane(L=100, N=30, seed=s) for s in (7, 7, 8))
    for _ in range(300):
        a.update(), b.update(), c.update()
    assert all(np.array_equal(a.x[i], b.x[i]) and np.array_equal(a.v[i], b.v[i]) for i in (0, 1))
    assert not all(np.array_equal(a.x[i], c.x[i]) for i in (0, 1))


def test_without_lane_changes_the_lanes_are_independent_single_lanes():
    """pchange = 0: every lane is a SingleLane, and the flow is the single-lane flow."""
    rho, L = 0.1, 400
    sim = TwoLane(L=L, N=int(2 * rho * L), pchange=0.0, seed=1)
    n_right = len(sim.x[RIGHT])
    m = sim.measure(T=20000)
    assert m['c'] == 0
    single = SingleLane(L=L, N=n_right, p=0.5, seed=1).measure(T=20000)
    assert abs(m['q_right'] - single['q']) < 0.01  # same density, independent random numbers


def test_flow_is_density_times_speed():
    m = TwoLane(L=300, N=60, seed=2).measure(T=5000)
    assert abs(m['q'] - m['rho'] * m['v']) < 2e-3
    assert abs(m['q'] - (m['q_right'] + m['q_left']) / 2) < 1e-12
    assert abs(m['rho'] - (m['rho_right'] + m['rho_left']) / 2) < 1e-12


def test_two_lanes_carry_more_than_one_at_the_maximum():
    """Paper, Sect. 4.2 and Fig. 4: the flow per lane at rho = 0.08 is above that of the single lane."""
    L, rho = 1000, 0.08
    two = TwoLane(L=L, N=int(2 * rho * L), seed=1).measure(T=30000)
    one = SingleLane(L=L, N=int(rho * L), seed=1).measure(T=30000)
    assert two['q'] > one['q'] and 0.32 < two['q'] < 0.36


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('ok  ', name)
