"""Nagel-Schreckenberg cellular automaton for single-lane freeway traffic.

K. Nagel, M. Schreckenberg, "A cellular automaton model for freeway traffic",
J. Phys. I France 2 (1992) 2221-2229.

The road has L sites, each empty or occupied by one car with an integer speed 0 <= v <= vmax.
One update is made of four steps, performed in parallel for all the cars:

    1) acceleration:    if v < vmax and the gap d to the car ahead is larger than v + 1:  v -> v + 1
    2) slowing down:    if d <= v:                                                         v -> d - 1
    3) randomization:   with probability p, if v > 0:                                      v -> v - 1
    4) car motion:      every car advances by v sites.

Here d is the difference between the positions of the car ahead and of the car.

Boundary conditions
    periodic  the road is a ring (closed system, N cars, constant density rho = N / L).
    open      the bottleneck of the paper (Sect. 4): when the leftmost site is empty a car with v = 0 is
              put there, and the cars that reach the rightmost vmax + 1 sites are removed.

Measurements are made with ``measure``: relax (10 L steps as in the paper), then average.
The flow is q = rho <v>: the number of cars crossing a bond per time step, averaged over the L bonds.
"""
import numpy as np


class SingleLane:
    def __init__(self, L: int, N: int = 0, p: float = 0.5, bc: str = 'periodic', seed: int = 0, vmax: int = 5):
        """L sites, N cars at random sites with v = 0, slowing-down probability p (the paper uses p = 0.5).

        For bc='open' N is the initial number of cars (0 is fine), for bc='periodic' it is the constant
        number of cars.
        """
        self.allowed_bc = ['periodic', 'open']
        if bc not in self.allowed_bc:
            raise KeyError(f"boundary condition bc: '{bc}' not in {self.allowed_bc}")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        if vmax < 1:
            raise ValueError("vmax must be at least 1")
        self.L, self.p, self.bc, self.seed, self.vmax = L, p, bc, seed, vmax
        self.rng = np.random.default_rng(seed)  # one generator, so that the seed makes the run reproducible

        if bc == 'periodic':
            if not 1 <= N <= L:
                raise ValueError("periodic road: 1 <= N <= L")
            self.x = np.sort(self.rng.choice(L, size=N, replace=False))
        else:
            if not 0 <= N <= L - (vmax + 1):
                raise ValueError("open road: 0 <= N <= L - (vmax + 1)")
            self.x = np.sort(self.rng.choice(L - (vmax + 1), size=N, replace=False))
        self.v = np.zeros(N, dtype=int)

        self.t = 0  # number of updates done
        self.flow = 0  # cars that went through the right end: the bond L-1 -> 0 (periodic) or the exit (open)
        self.last_flow = 0  # the same, in the last update only
        self.last_inflow = 0  # cars injected in the last update (open road)

    # ------------------------------------------------------------------ state
    @property
    def N(self) -> int:
        return len(self.x)

    @property
    def density(self) -> float:
        return self.N / self.L

    def gaps(self) -> np.ndarray:
        """Distance d (in sites) from every car to the car ahead. x is sorted."""
        d = np.empty(self.N, dtype=int)
        if self.N == 0:
            return d
        d[:-1] = self.x[1:] - self.x[:-1]
        if self.bc == 'periodic':
            d[-1] = self.x[0] + self.L - self.x[-1]  # the first car is ahead of the last one; alone: d = L
        else:
            d[-1] = self.L + self.vmax + 1  # nobody ahead of the last car
        return d

    distances = gaps  # old name

    # ----------------------------------------------------------------- dynamics
    def update(self):
        """One parallel update of the whole road."""
        n = self.N
        v = self.v
        if n > 0:
            d = self.gaps()
            v = np.where((v < self.vmax) & (d > v + 1), v + 1, v)  # 1) acceleration
            v = np.minimum(v, d - 1)  # 2) slowing down
            slow = self.rng.random(n) < self.p  # 3) randomization
            v = np.where(slow & (v > 0), v - 1, v)
        self.v = v
        x = self.x + v  # 4) car motion

        if self.bc == 'periodic':
            self.last_flow = int(np.sum(x >= self.L))  # cars that cross the bond L-1 -> 0
            x = x % self.L
            order = np.argsort(x)  # the cars never overtake: this is only a rotation of the arrays
            self.x, self.v = x[order], v[order]
        else:
            gone = x >= self.L - (self.vmax + 1)  # absorbing exit: the rightmost vmax + 1 sites
            self.last_flow = int(np.sum(gone))
            x, v = x[~gone], v[~gone]
            self.last_inflow = 0
            if len(x) == 0 or x[0] > 0:  # the leftmost site is empty: a new car, with v = 0
                x, v = np.concatenate(([0], x)), np.concatenate(([0], v))
                self.last_inflow = 1
            self.x, self.v = x, v
        self.flow += self.last_flow
        self.t += 1

    def relax(self, steps: int = None):
        """Discard the transient. The paper uses 10 L steps."""
        for _ in range(10 * self.L if steps is None else steps):
            self.update()

    # ------------------------------------------------------------- measurements
    def run(self, T: int, profile: bool = False) -> dict:
        """T updates, recording the time series of what is needed for the averages."""
        n_t = np.empty(T)  # cars on the road, when the update starts
        sumv_t = np.empty(T)  # sum of the speeds used in the update
        out_t = np.empty(T)  # cars through the right end
        in_t = np.empty(T)  # cars injected
        occ = np.zeros(self.L) if profile else None
        for t in range(T):
            n_t[t] = self.N
            self.update()
            sumv_t[t] = self.v.sum()
            out_t[t], in_t[t] = self.last_flow, self.last_inflow
            if profile:
                occ[self.x] += 1
        return dict(n=n_t, sumv=sumv_t, out=out_t, inflow=in_t, profile=None if occ is None else occ / T)

    def measure(self, T: int = 20000, relax: int = None, blocks: int = 20, profile: bool = False) -> dict:
        """Relax, then average over T steps. The errors come from the scatter of ``blocks`` block averages
        (successive time steps are strongly correlated, so sigma / sqrt(T) would be wrong).

        Returns the density rho, the mean speed v (cars and time), the flow q = rho <v>, and for an open
        road the rates of cars through the exit and injected.
        """
        self.relax(relax)
        r = self.run(T, profile=profile)

        def mean_and_error(series):
            b = np.array_split(series, blocks)
            m = np.array([c.mean() for c in b])
            return float(series.mean()), float(m.std(ddof=1) / np.sqrt(blocks))

        q, dq = mean_and_error(r['sumv'] / self.L)  # cars per bond per step, averaged over the bonds
        rho, drho = mean_and_error(r['n'] / self.L)
        ok = r['n'] > 0
        v, dv = mean_and_error((r['sumv'][ok] / r['n'][ok]))
        out = dict(rho=rho, drho=drho, v=v, dv=dv, q=q, dq=dq)
        if self.bc == 'open':
            out['q_out'], out['dq_out'] = mean_and_error(r['out'])
            out['q_in'], out['dq_in'] = mean_and_error(r['inflow'])
            out['profile'] = r['profile']
        return out

    # ------------------------------------------------------------------ text
    def row(self, what: str = 'x') -> str:
        """The road as a string: 'x' marks the cars, 'v' shows their speed (like Figs. 1 and 2 of the paper)."""
        cells = np.full(self.L, '.', dtype='<U1')
        cells[self.x] = 'x' if what == 'x' else np.array([str(k) for k in self.v])
        return '|' + ''.join(cells) + '|'

    @property
    def stringx(self) -> str:
        return self.row('x')

    @property
    def stringv(self) -> str:
        return self.row('v')

    plot = lambda self, xORv='x': self.row(xORv)  # old name

    def __repr__(self) -> str:
        return f"SingleLane(L={self.L}, N={self.N}, p={self.p}, bc='{self.bc}', seed={self.seed}, vmax={self.vmax})"


if __name__ == '__main__':
    # a text version of Fig. 2 of the paper: density 0.1, time goes down, the jams move backwards
    lane = SingleLane(L=70, N=7, p=0.5)
    lane.relax()
    for _ in range(33):
        lane.update()
        print(lane.stringv)
