"""Two-lane Nagel-Schreckenberg model with the asymmetric lane-change rules.

M. Rickert, K. Nagel, M. Schreckenberg, A. Latour, "Two lane traffic simulations using cellular automata",
Physica A 231 (1996) 534, arXiv:cond-mat/9512119.

Two parallel lanes of L sites each, periodic boundary conditions, N cars in total. Lane 0 is the right lane
(the one cars return to), lane 1 is the left lane. One time step has two sub-steps (Sect. 3 of the paper):

  1. Lane change. Cars are moved sideways only, they do not advance. All the cars decide in parallel, from the
     configuration at the beginning of the time step.
  2. Single-lane update (the four rules of ``model.py``) of each lane, with the configuration left by sub-step 1.

A car i at position x with speed v changes to the other lane if ALL these conditions hold (Sect. 3 and 4):

    (T1) gap(i) < l                 only for right -> left. In the asymmetric model a car on the left lane does
                                    not need this: it always tries to return to the right lane.
    (T2) gap_o(i) > l_o             enough room ahead on the other lane
    (T3) gap_o,back(i) > l_o,back   enough room behind on the other lane (not to get in somebody's way)
    (T4) rand() < p_change

with the parameters of Sect. 4:  l = v + 1,  l_o = l = v + 1,  l_o,back = vmax,  p_change = 1.

gap(i)         empty sites ahead of the car on its own lane.
gap_o(i)       empty sites ahead on the other lane, counted from the position of the car itself.
gap_o,back(i)  empty sites behind on the other lane, counted from the position of the car itself.
               If the other lane is occupied at the same position of the car both gaps are -1.

Flows are per lane: j = (sum of the speeds of the cars) / (number of sites), as in the paper's fundamental
diagrams. The density is the mean density per lane, rho = N / (2 L).
"""
import numpy as np

RIGHT, LEFT = 0, 1


class TwoLane:
    def __init__(self, L: int, N: int, p: float = 0.5, pchange: float = 1.0, vmax: int = 5,
                 lookback: int = None, seed: int = 0):
        """L sites per lane, N cars on the two lanes (random sites, random lane, v = 0)."""
        if not 1 <= N <= 2 * L:
            raise ValueError("1 <= N <= 2 L")
        if not (0.0 <= p <= 1.0 and 0.0 <= pchange <= 1.0):
            raise ValueError("p and pchange must be in [0, 1]")
        if vmax < 1:
            raise ValueError("vmax must be at least 1")
        self.L, self.p, self.pchange, self.vmax, self.seed = L, p, pchange, vmax, seed
        self.lookback = vmax if lookback is None else lookback  # l_o,back = 5 = vmax in the paper
        self.rng = np.random.default_rng(seed)

        cells = np.sort(self.rng.choice(2 * L, size=N, replace=False))  # cell = lane * L + site
        self.x = [cells[cells < L], cells[cells >= L] - L]  # sorted, per lane
        self.v = [np.zeros(len(self.x[0]), dtype=int), np.zeros(len(self.x[1]), dtype=int)]

        self.t = 0
        self.last_changes = (0, 0)  # right->left and left->right changes in the last update

    # ------------------------------------------------------------------ state
    @property
    def N(self) -> int:
        return len(self.x[0]) + len(self.x[1])

    @property
    def density(self) -> float:
        """Mean density per lane."""
        return self.N / (2 * self.L)

    def lane_density(self, lane: int) -> float:
        return len(self.x[lane]) / self.L

    # ------------------------------------------------------------------- gaps
    def gap(self, lane: int) -> np.ndarray:
        """Empty sites ahead of each car of the lane, on the same lane (alone on the lane: L - 1)."""
        x = self.x[lane]
        d = np.empty(len(x), dtype=int)
        if len(x):
            d[:-1] = x[1:] - x[:-1]
            d[-1] = x[0] + self.L - x[-1]
        return d - 1

    def gaps_other(self, lane: int):
        """gap_o and gap_o,back of each car of ``lane``, measured on the other lane."""
        x, xo, L = self.x[lane], self.x[1 - lane], self.L
        if len(xo) == 0:
            full = np.full(len(x), L, dtype=int)  # nobody there: no limit
            return full, full.copy()
        ahead = xo[np.searchsorted(xo, x, side='left') % len(xo)]  # first car of the other lane at x' >= x
        behind = xo[np.searchsorted(xo, x, side='right') - 1]  # last car of the other lane at x' <= x
        return (ahead - x) % L - 1, (x - behind) % L - 1

    # ----------------------------------------------------------- lane changes
    def lane_change(self):
        """Sub-step 1: the asymmetric lane-change rule, a strict parallel update."""
        moves = []
        for lane in (RIGHT, LEFT):
            x, v = self.x[lane], self.v[lane]
            g, go, gb = self.gap(lane), *self.gaps_other(lane)
            ok = (go > v + 1) & (gb > self.lookback)  # T2, T3
            if lane == RIGHT:
                ok &= g < v + 1  # T1, only to go left
            ok &= self.rng.random(len(x)) < self.pchange  # T4
            moves.append(ok)
        self.last_changes = (int(moves[RIGHT].sum()), int(moves[LEFT].sum()))
        new_x, new_v = [], []
        for lane in (RIGHT, LEFT):
            stay, come = ~moves[lane], moves[1 - lane]
            x = np.concatenate((self.x[lane][stay], self.x[1 - lane][come]))
            v = np.concatenate((self.v[lane][stay], self.v[1 - lane][come]))
            order = np.argsort(x)
            new_x.append(x[order])
            new_v.append(v[order])
        self.x, self.v = new_x, new_v

    # ----------------------------------------------------------------- update
    def _advance(self, lane: int):
        """Sub-step 2 on one lane: the single-lane rules, as in ``model.SingleLane.update``."""
        x, v = self.x[lane], self.v[lane]
        if len(x) == 0:
            return
        d = self.gap(lane) + 1
        v = np.where((v < self.vmax) & (d > v + 1), v + 1, v)  # 1) acceleration
        v = np.minimum(v, d - 1)  # 2) slowing down
        slow = self.rng.random(len(x)) < self.p  # 3) randomization
        v = np.where(slow & (v > 0), v - 1, v)
        x = (x + v) % self.L  # 4) car motion
        order = np.argsort(x)  # no overtaking inside a lane: only a rotation of the arrays
        self.x[lane], self.v[lane] = x[order], v[order]

    def update(self):
        """One time step: lane changes, then the single-lane update of both lanes."""
        self.lane_change()
        self._advance(RIGHT)
        self._advance(LEFT)
        self.t += 1

    def relax(self, steps: int = None):
        """Discard the transient (10 L steps, like the single lane)."""
        for _ in range(10 * self.L if steps is None else steps):
            self.update()

    # ------------------------------------------------------------- measurements
    def run(self, T: int) -> dict:
        n = np.empty((T, 2))  # cars per lane in the motion step (after the lane changes)
        sumv = np.empty((T, 2))  # sum of the speeds used in the motion
        ch = np.empty((T, 2))  # right->left, left->right
        stopped = np.empty((T, 2))
        for t in range(T):
            self.update()
            n[t] = len(self.x[RIGHT]), len(self.x[LEFT])  # the lanes in which the cars moved
            sumv[t] = self.v[RIGHT].sum(), self.v[LEFT].sum()
            stopped[t] = (self.v[RIGHT] == 0).sum(), (self.v[LEFT] == 0).sum()
            ch[t] = self.last_changes
        return dict(n=n, sumv=sumv, changes=ch, stopped=stopped)

    def measure(self, T: int = 20000, relax: int = None, blocks: int = 20) -> dict:
        """Relax, then average over T steps. Errors from the scatter of ``blocks`` block averages.

        rho          mean density per lane, N / (2 L)
        q            flow per lane (both lanes together), (sum of all the speeds) / (2 L)
        q_right/left flow of each lane, (sum of the speeds on the lane) / L
        rho_right/left  density of each lane
        v            mean speed of the cars
        c            lane changes per site and time step (both directions), changes / L
        c_car        lane changes per car and time step
        """
        self.relax(relax)
        r = self.run(T)

        def mean_and_error(series):
            m = np.array([c.mean() for c in np.array_split(series, blocks)])
            return float(series.mean()), float(m.std(ddof=1) / np.sqrt(blocks))

        out = {}
        for name, series in dict(
                rho=r['n'].sum(1) / (2 * self.L), q=r['sumv'].sum(1) / (2 * self.L),
                q_right=r['sumv'][:, RIGHT] / self.L, q_left=r['sumv'][:, LEFT] / self.L,
                rho_right=r['n'][:, RIGHT] / self.L, rho_left=r['n'][:, LEFT] / self.L,
                v=r['sumv'].sum(1) / r['n'].sum(1), c=r['changes'].sum(1) / self.L,
                c_car=r['changes'].sum(1) / r['n'].sum(1),
                stopped=r['stopped'].sum(1) / r['n'].sum(1)).items():
            out[name], out['d' + name] = mean_and_error(series)
        return out

    # ------------------------------------------------------------------ text
    def row(self, lane: int, what: str = 'v') -> str:
        cells = np.full(self.L, '.', dtype='<U1')
        cells[self.x[lane]] = 'x' if what == 'x' else np.array([str(k) for k in self.v[lane]])
        return '|' + ''.join(cells) + '|'

    def __repr__(self) -> str:
        return (f"TwoLane(L={self.L}, N={self.N}, p={self.p}, pchange={self.pchange}, vmax={self.vmax}, "
                f"lookback={self.lookback}, seed={self.seed})")


if __name__ == '__main__':
    # a text version of the space-time diagram: left lane above, right lane below, time goes down
    sim = TwoLane(L=80, N=14, p=0.5)
    sim.relax()
    for _ in range(25):
        sim.update()
        print(sim.row(LEFT))
        print(sim.row(RIGHT))
        print()
