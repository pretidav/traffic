"""Figures of the two-lane model (asymmetric lane changes), in the style of the single-lane ones.

    python plots_multiple.py                   # all the figures, with the parameters of the paper
    python plots_multiple.py fd                # flow of the two lanes against the density
    python plots_multiple.py compare           # zoom around the maximum, two lanes against one lane
    python plots_multiple.py changes           # frequency of the lane changes
    python plots_multiple.py spacetime         # space-time diagram of the two lanes

Parameters: vmax = 5, p = 0.5, l = v + 1, l_o = v + 1, l_o,back = vmax, p_change = 1 (Sect. 4 of the paper).
The density rho is the density per lane, N / (2 L); the flows are per lane.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from tqdm import tqdm

from model import SingleLane
from model_multiple import LEFT, RIGHT, TwoLane
from style import ACCENT, INK, PAPER, new_figure, style_axes, style_legend


def scan(rhos, L=1000, p=0.5, vmax=5, pchange=1.0, T=50000, seed=0, desc="two lanes"):
    """measure() of the two-lane model at every density, returned as a dict of arrays."""
    res = [TwoLane(L=L, N=max(1, round(2 * r * L)), p=p, vmax=vmax, pchange=pchange, seed=seed + i).measure(T=T)
           for i, r in enumerate(tqdm(rhos, desc=desc))]
    return {k: np.array([r[k] for r in res]) for k in res[0]}


def _legend(ax, loc="upper right"):
    style_legend(ax.legend(loc=loc, fancybox=False, framealpha=1, edgecolor=INK, facecolor=PAPER,
                           borderpad=0.8, fontsize=8))


def _title(L, p, vmax, pchange, what="TWO LANES, ASYMMETRIC"):
    return f"[ {what} ]  L={L}  p={p}  vmax={vmax}  pchange={pchange}"


def fundamental_diagram(data, L=1000, p=0.5, vmax=5, pchange=1.0, out=None):
    """Flow per lane against the density per lane: both lanes together, right lane, left lane."""
    fig = new_figure()
    ax = fig.add_subplot(111)
    style_axes(ax)
    rr = np.linspace(0, 1, 400)
    ax.plot(rr, np.minimum(vmax * rr, 1 - rr), color=INK, lw=1.5, ls=(0, (4, 3)), zorder=2,
            label=r"$p=0$:  min($v_{max}\,\rho$, 1 - $\rho$)")
    kw = dict(ecolor=INK, elinewidth=1.2)
    ax.errorbar(data["rho"], data["q_right"], yerr=data["dq_right"], fmt="s", ms=6, color=INK, zorder=4,
                label="RIGHT LANE", **kw)
    ax.errorbar(data["rho"], data["q_left"], yerr=data["dq_left"], fmt="s", ms=6, mfc=PAPER, mec=INK, mew=1.8,
                zorder=4, label="LEFT LANE", **kw)
    ax.errorbar(data["rho"], data["q"], yerr=data["dq"], fmt="s", ms=8, mfc=PAPER, mec=ACCENT, mew=2.5,
                zorder=5, label="BOTH LANES  (PER LANE)", **kw)
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"DENSITY PER LANE  $\rho = N / 2L$", fontweight="bold", labelpad=8)
    ax.set_ylabel("FLOW (CARS PER STEP, PER LANE)", fontweight="bold", labelpad=8)
    ax.set_title(_title(L, p, vmax, pchange), fontweight="bold", loc="left", pad=12, fontsize=10)
    _legend(ax)
    fig.tight_layout()
    out = out or f"fundamental_diagram_2lanes_L{L}_p{p}_vmax{vmax}.png"
    fig.savefig(out, dpi=150, facecolor=PAPER)
    plt.close(fig)
    print("->", out)


def compare(L=2000, p=0.5, vmax=5, pchange=1.0, T=50000, seed=0, rhos=None, out=None):
    """Zoom around the maximum of the flow: two lanes (per lane) against the single lane (Fig. 4 of the paper)."""
    rhos = np.arange(0.05, 0.1501, 0.005) if rhos is None else rhos
    two = scan(rhos, L=L, p=p, vmax=vmax, pchange=pchange, T=T, seed=seed)
    one = [SingleLane(L=L, N=round(r * L), p=p, vmax=vmax, seed=seed + i).measure(T=T)
           for i, r in enumerate(tqdm(rhos, desc="single lane"))]
    rho1 = np.array([r["rho"] for r in one])
    q1, dq1 = np.array([r["q"] for r in one]), np.array([r["dq"] for r in one])

    fig = new_figure()
    ax = fig.add_subplot(111)
    style_axes(ax)
    kw = dict(ecolor=INK, elinewidth=1.2)
    ax.errorbar(rho1, q1, yerr=dq1, fmt="s", ms=6, color=INK, zorder=4, label="SINGLE LANE", **kw)
    ax.errorbar(two["rho"], two["q"], yerr=two["dq"], fmt="s", ms=8, mfc=PAPER, mec=ACCENT, mew=2.5, zorder=5,
                label="TWO LANES, ASYMMETRIC  (PER LANE)", **kw)
    ax.set_xlabel(r"DENSITY PER LANE  $\rho$", fontweight="bold", labelpad=8)
    ax.set_ylabel("FLOW (CARS PER STEP, PER LANE)", fontweight="bold", labelpad=8)
    ax.set_title(_title(L, p, vmax, pchange, "NEAR THE MAXIMUM"), fontweight="bold", loc="left", pad=12, fontsize=10)
    _legend(ax)
    fig.tight_layout()
    out = out or f"comparison_1lane_2lanes_L{L}_p{p}_vmax{vmax}.png"
    fig.savefig(out, dpi=150, facecolor=PAPER)
    plt.close(fig)
    print("->", out)


def lane_changes(data, L=1000, p=0.5, vmax=5, pchange=1.0, out=None):
    """Lane changes per site and time step, and per car and time step (Figs. 5 and 6 of the paper)."""
    fig = new_figure((10, 4.6))
    ax1, ax2 = fig.subplots(1, 2)
    for ax, key, ylabel in ((ax1, "c", "PER SITE AND TIME STEP"), (ax2, "c_car", "PER CAR AND TIME STEP")):
        style_axes(ax)
        ax.errorbar(data["rho"], data[key], yerr=data["d" + key], fmt="s", ms=7, mfc=PAPER, mec=ACCENT, mew=2.5,
                    ecolor=INK, elinewidth=1.2, zorder=5)
        ax.set_xlabel(r"DENSITY PER LANE  $\rho$", fontweight="bold", labelpad=8)
        ax.set_ylabel("LANE CHANGES " + ylabel, fontweight="bold", labelpad=8, fontsize=8)
        ax.set_xlim(0, 0.9)
        ax.set_ylim(0, None)
    fig.suptitle(_title(L, p, vmax, pchange, "LANE CHANGES"), x=0.02, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out or f"lane_changes_L{L}_p{p}_vmax{vmax}.png"
    fig.savefig(out, dpi=150, facecolor=PAPER)
    plt.close(fig)
    print("->", out)


def spacetime(L=2000, rho=0.09, sites=400, rows=300, p=0.5, vmax=5, pchange=1.0, seed=1, out=None):
    """Space-time diagrams of the two lanes (Fig. 2 of the paper): time goes down. Orange: stopped cars."""
    sim = TwoLane(L=L, N=round(2 * rho * L), p=p, vmax=vmax, pchange=pchange, seed=seed)
    sim.relax()
    imgs = {lane: np.tile(to_rgb(PAPER), (rows, sites, 1)) for lane in (RIGHT, LEFT)}
    for t in range(rows):
        sim.update()
        for lane in (RIGHT, LEFT):
            keep = sim.x[lane] < sites
            x, v = sim.x[lane][keep], sim.v[lane][keep]
            imgs[lane][t, x] = np.where((v == 0)[:, None], to_rgb(ACCENT), to_rgb(INK))

    fig = new_figure((10, 6.4))
    axes = fig.subplots(1, 2)
    for ax, lane, name in ((axes[0], LEFT, "LEFT LANE"), (axes[1], RIGHT, "RIGHT LANE")):
        ax.imshow(imgs[lane], interpolation="nearest", aspect="auto")
        style_axes(ax, grid=False)
        ax.set_xlabel("SITE", fontweight="bold", labelpad=6)
        ax.set_title(f"[ {name} ]", fontweight="bold", loc="left", pad=8, fontsize=10)
    axes[0].set_ylabel("TIME STEP", fontweight="bold", labelpad=6)
    handles = [Patch(facecolor=INK, edgecolor=INK, label="MOVING CAR"),
               Patch(facecolor=ACCENT, edgecolor=INK, label="STOPPED CAR  (v = 0)")]
    style_legend(fig.legend(handles=handles, loc="lower center", ncol=2, fancybox=False, framealpha=1,
                            edgecolor=INK, facecolor=PAPER, fontsize=8, borderpad=0.7))
    fig.suptitle(f"[ SPACE-TIME, TWO LANES ]  DENSITY PER LANE {sim.density:.2f}  p={p}  vmax={vmax}"
                 f"  pchange={pchange}", x=0.02, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = out or f"spacetime_2lanes_p{p}_vmax{vmax}.png"
    fig.savefig(out, dpi=150, facecolor=PAPER)
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", nargs="?", default="all", choices=["all", "fd", "compare", "changes", "spacetime"])
    parser.add_argument("--L", type=int, default=1000, help="sites per lane")
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--vmax", type=int, default=5)
    parser.add_argument("--pchange", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=50000, help="steps of every average")
    a = parser.parse_args()
    kw = dict(p=a.p, vmax=a.vmax, pchange=a.pchange)

    if a.what in ("all", "fd", "changes"):
        data = scan(np.arange(0.02, 0.9 + 1e-9, 0.02), L=a.L, T=a.T, **kw)
        if a.what in ("all", "fd"):
            fundamental_diagram(data, L=a.L, **kw)
        if a.what in ("all", "changes"):
            lane_changes(data, L=a.L, **kw)
    if a.what in ("all", "compare"):
        compare(T=a.T, **kw)
    if a.what in ("all", "spacetime"):
        spacetime(**kw)
