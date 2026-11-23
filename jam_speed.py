"""Speed of the jams (start-stop waves), measured from the space-time pattern of the stopped cars.

A jam is a region of stopped cars (v = 0) that moves backwards. If the pattern of stopped cars at time t+tau is the
pattern at time t shifted by k sites, then the jams move at v_J = k / tau. The shift is found with the circular
cross-correlation of the two patterns (all the jams of the ring move together, so they all contribute), averaged
over time, for several lags tau. v_J comes from the fit k = v_J tau through the origin. Negative = backwards.

The pattern of the two-lane model is the sum of the stopped cars of the two lanes.

    python jam_speed.py                  # the three figures (about 15 minutes)
    python jam_speed.py rho | p | vmax   # one of them
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model import SingleLane
from model_multiple import TwoLane
from plots import bottleneck
from style import ACCENT, INK, PAPER, new_figure, style_axes, style_legend

LAGS = (50, 100, 150, 200)


def stopped_pattern(sim, T, two_lanes):
    """T x L array: number of stopped cars at every site, at every time step (after the relaxation)."""
    L = sim.L
    F = np.zeros((T, L), dtype=np.float32)
    for t in range(T):
        sim.update()
        if two_lanes:
            for lane in (0, 1):
                F[t, sim.x[lane][sim.v[lane] == 0]] += 1
        else:
            F[t, sim.x[sim.v == 0]] = 1
    return F


def pattern_velocity(F, lags=LAGS):
    """(v_J, r): the velocity of the pattern and the height of the correlation peak (1 = identical pattern)."""
    T, L = F.shape
    F = F - F.mean()
    var = (F ** 2).mean()
    if var < 1e-12:  # no stopped car at all: there is no jam
        return float('nan'), 0.0
    shifts, heights = [], []
    for tau in lags:
        A, B = np.fft.rfft(F[:-tau], axis=1), np.fft.rfft(F[tau:], axis=1)
        c = np.fft.irfft((np.conj(A) * B).mean(0), n=L) / (L * var)
        k = int(np.argmax(c))
        y0, y1, y2 = c[k - 1], c[k], c[(k + 1) % L]
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)  # parabola through the three points around the maximum
        shifts.append(k if k < L / 2 else k - L)
        heights.append(c.max())  # the normalized correlation at the maximum
    lags = np.asarray(lags, dtype=float)
    shifts = np.asarray(shifts)
    return float((shifts * lags).sum() / (lags ** 2).sum()), float(np.mean(heights))


def jam_speed(rho, p=0.5, vmax=5, two_lanes=False, L=2000, T=3000, seeds=(1, 2, 3), relax=None, pchange=1.0):
    """Mean and standard error of v_J over independent runs, and the stopped fraction of the cars."""
    v, stopped = [], []
    for seed in seeds:
        if two_lanes:
            sim = TwoLane(L=L, N=round(2 * rho * L), p=p, vmax=vmax, pchange=pchange, seed=seed)
        else:
            sim = SingleLane(L=L, N=round(rho * L), p=p, vmax=vmax, seed=seed)
        sim.relax(10 * L if relax is None else relax)
        F = stopped_pattern(sim, T, two_lanes)
        v.append(pattern_velocity(F)[0])
        stopped.append(F.sum(1).mean() / sim.N)
    v = np.array(v)
    v = v[~np.isnan(v)]
    if len(v) < 2:  # no jam in (almost) all the runs
        return float('nan'), 0.0, float(np.mean(stopped))
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), float(np.mean(stopped))


def predicted(p=0.5, vmax=5, L=1000, T=100000):
    """Rankine-Hugoniot at the front of a packed jam (q = 0, rho = 1), with the outflow state of the bottleneck."""
    rho_out, q_out, _, _ = bottleneck(L=L, p=p, vmax=vmax, T=T)
    return -q_out / (1 - rho_out)


def _plot(x, single, two, pred, xlabel, title, out, xlim=None, ylim=(-0.9, 0.05)):
    with open(out.replace(".png", ".json"), "w") as f:  # keep the numbers
        json.dump(dict(x=[float(v) for v in x], single=single, two=two, predicted=None if pred is None else [float(v) for v in pred]), f)
    fig = new_figure()
    ax = fig.add_subplot(111)
    style_axes(ax)
    ax.axhline(0, color=INK, lw=1)
    kw = dict(ecolor=INK, elinewidth=1.2)
    if pred is not None:
        ax.plot(x, pred, color=INK, lw=1.5, ls=(0, (4, 3)), zorder=2, label=r"BOTTLENECK:  $-q / (1 - \rho)$")
    ax.errorbar(x, [s[0] for s in single], yerr=[s[1] for s in single], fmt="s", ms=7, color=INK, zorder=4,
                label="SINGLE LANE", **kw)
    ax.errorbar(x, [s[0] for s in two], yerr=[s[1] for s in two], fmt="s", ms=10, mfc="none", mec=ACCENT, mew=2.5,
                zorder=5, label="TWO LANES, ASYMMETRIC", **kw)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontweight="bold", labelpad=8)
    ax.set_ylabel("JAM SPEED  (SITES PER STEP)", fontweight="bold", labelpad=8)
    ax.set_title(title, fontweight="bold", loc="left", pad=12, fontsize=10)
    style_legend(ax.legend(loc="lower right", fancybox=False, framealpha=1, edgecolor=INK, facecolor=PAPER,
                           borderpad=0.8, fontsize=8))
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=PAPER)
    plt.close(fig)
    print("->", out)


def vs_density(rhos=None, p=0.5, vmax=5, L=2000, out=None):
    rhos = np.arange(0.06, 0.61, 0.04) if rhos is None else rhos
    single = [jam_speed(r, p, vmax, False, L) for r in tqdm(rhos, desc="single lane")]
    two = [jam_speed(r, p, vmax, True, L) for r in tqdm(rhos, desc="two lanes")]
    _plot(rhos, single, two, [predicted(p, vmax)] * len(rhos), r"DENSITY PER LANE  $\rho$",
          f"[ JAM SPEED ]  p={p}  vmax={vmax}  L={L}", out or f"jam_speed_vs_density_p{p}_vmax{vmax}.png",
          xlim=(0, 0.65), ylim=(-0.5, -0.2))


def vs_p(ps=None, rho=0.4, vmax=5, L=2000, out=None):
    ps = np.arange(0.1, 0.91, 0.1) if ps is None else ps
    single = [jam_speed(rho, p, vmax, False, L) for p in tqdm(ps, desc="single lane")]
    two = [jam_speed(rho, p, vmax, True, L) for p in tqdm(ps, desc="two lanes")]
    pred = [predicted(p, vmax) for p in tqdm(ps, desc="bottleneck")]
    _plot(ps, single, two, pred, r"SLOWDOWN PROBABILITY  $p$", f"[ JAM SPEED ]  rho={rho}  vmax={vmax}  L={L}",
          out or f"jam_speed_vs_p_rho{rho}_vmax{vmax}.png", xlim=(0, 1), ylim=(-0.9, 0.0))


def vs_vmax(vmaxs=None, rho=0.4, p=0.5, L=2000, out=None):
    vmaxs = np.arange(2, 11) if vmaxs is None else vmaxs
    single = [jam_speed(rho, p, int(v), False, L) for v in tqdm(vmaxs, desc="single lane")]
    two = [jam_speed(rho, p, int(v), True, L) for v in tqdm(vmaxs, desc="two lanes")]
    pred = [predicted(p, int(v)) for v in tqdm(vmaxs, desc="bottleneck")]
    _plot(vmaxs, single, two, pred, r"MAXIMUM SPEED  $v_{max}$", f"[ JAM SPEED ]  rho={rho}  p={p}  L={L}",
          out or f"jam_speed_vs_vmax_rho{rho}_p{p}.png", xlim=(1, 11), ylim=(-0.5, -0.2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", nargs="?", default="all", choices=["all", "rho", "p", "vmax"])
    a = parser.parse_args()
    if a.what in ("all", "rho"):
        vs_density()
    if a.what in ("all", "p"):
        vs_p()
    if a.what in ("all", "vmax"):
        vs_vmax()
