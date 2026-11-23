"""Figures of the model, in the style of the blog.

    python plots.py                  # fundamental diagram and space-time diagram, with the parameters of the paper
    python plots.py fd --L 1000      # only the fundamental diagram
    python plots.py spacetime
    python plots.py bottleneck       # the open road of Sect. 4 of the paper, prints the numbers
    python plots.py scan             # flow of the open road against p, and against vmax (two figures)

The paper uses vmax = 5 and p = 0.5.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from tqdm import tqdm

from model import SingleLane
from style import ACCENT, INK, PAPER, new_figure, style_axes, style_legend


def bottleneck(L=1000, p=0.5, vmax=5, T=200000, seed=0):
    """Open road: cars injected at the left, removed at the right (Sect. 4 of the paper).

    Returns the density of the bulk (the middle half of the road) and the flow through the exit.
    The paper finds rho = 0.069 +- 0.002 and q = 0.304 +- 0.001.
    """
    lane = SingleLane(L=L, N=0, p=p, bc='open', vmax=vmax, seed=seed)
    r = lane.measure(T=T, profile=True)
    bulk = r['profile'][L // 4: 3 * L // 4]
    return float(bulk.mean()), r['q_out'], r['dq_out'], r


def fundamental_diagram(L=1000, p=0.5, vmax=5, T=50000, step=0.02, seed=0, out=None, with_bottleneck=True):
    """Flow q and mean speed v / vmax against the density, on a ring. One point per density."""
    rhos = np.arange(step, 0.9 + 1e-9, step)
    res = [SingleLane(L=L, N=max(1, round(r * L)), p=p, vmax=vmax, seed=seed + i).measure(T=T)
           for i, r in enumerate(tqdm(rhos, desc="densities"))]
    rho = np.array([r['rho'] for r in res])
    q, dq = np.array([r['q'] for r in res]), np.array([r['dq'] for r in res])
    v, dv = np.array([r['v'] for r in res]), np.array([r['dv'] for r in res])

    fig = new_figure()
    ax = fig.add_subplot(111)
    style_axes(ax)
    rr = np.linspace(0, 1, 400)
    ax.plot(rr, np.minimum(vmax * rr, 1 - rr), color=INK, lw=1.5, ls=(0, (4, 3)), zorder=2,
            label=r"RING ROAD, $p=0$:  min($v_{max}\,\rho$, 1 - $\rho$)")
    ax.errorbar(rho, v / vmax, yerr=dv / vmax, fmt="s", ms=6, color=INK, ecolor=INK, elinewidth=1.2,
                zorder=4, label=r"RING ROAD:  MEAN SPEED  $\langle v \rangle / v_{max}$")
    ax.errorbar(rho, q, yerr=dq, fmt="s", ms=8, mfc=PAPER, mec=ACCENT, mew=2.5, ecolor=INK, elinewidth=1.2,
                zorder=5, label=r"RING ROAD:  FLOW  $q = \rho \langle v \rangle$")
    if with_bottleneck:
        rho_c, q_c, dq_c, _ = bottleneck(L=L, p=p, vmax=vmax, seed=seed)
        ax.plot([rho_c], [q_c], marker="X", ms=13, mfc=ACCENT, mec=INK, mew=2, ls="none", zorder=6,
                label="OPEN ROAD:  BOTTLENECK")
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"DENSITY  $\rho$  (CARS PER SITE)", fontweight="bold", labelpad=8)
    ax.set_ylabel("FLOW (CARS PER STEP)  /  SPEED", fontweight="bold", labelpad=8)
    ax.set_title(f"[ NAGEL-SCHRECKENBERG ]  L={L}  p={p}  vmax={vmax}", fontweight="bold", loc="left", pad=12)
    style_legend(ax.legend(loc="upper right", fancybox=False, framealpha=1, edgecolor=INK, facecolor=PAPER,
                           borderpad=0.8, fontsize=8))
    fig.tight_layout()
    out = out or f"fundamental_diagram_L{L}_p{p}_vmax{vmax}.png"
    fig.savefig(out, dpi=150, facecolor=PAPER)
    plt.close(fig)
    print("->", out)


def spacetime(L=200, rows=100, rhos=(0.03, 0.10), p=0.5, vmax=5, seed=1, out=None):
    """Space-time diagrams like Figs. 1 and 2 of the paper: time goes down. Orange: stopped cars (v = 0)."""
    fig = new_figure((10, 5.8))
    axes = fig.subplots(1, len(rhos))
    for ax, rho in zip(np.atleast_1d(axes), rhos):
        lane = SingleLane(L=L, N=max(1, round(rho * L)), p=p, vmax=vmax, seed=seed)
        lane.relax()
        img = np.tile(to_rgb(PAPER), (rows, L, 1))
        for t in range(rows):
            lane.update()
            img[t, lane.x] = np.where((lane.v == 0)[:, None], to_rgb(ACCENT), to_rgb(INK))
        ax.imshow(img, interpolation="nearest", aspect="auto")
        style_axes(ax, grid=False)
        ax.set_xlabel("SITE", fontweight="bold", labelpad=6)
        ax.set_title(f"[ DENSITY {lane.density:.2f} ]", fontweight="bold", loc="left", pad=8, fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("TIME STEP", fontweight="bold", labelpad=6)
    handles = [Patch(facecolor=INK, edgecolor=INK, label="MOVING CAR"),
               Patch(facecolor=ACCENT, edgecolor=INK, label="STOPPED CAR  (v = 0)")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=2, fancybox=False, framealpha=1, edgecolor=INK,
                     facecolor=PAPER, fontsize=8, borderpad=0.7)
    style_legend(leg)
    fig.suptitle(f"[ SPACE-TIME ]  p={p}  vmax={vmax}", x=0.02, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = out or f"spacetime_p{p}_vmax{vmax}.png"
    fig.savefig(out, dpi=150, facecolor=PAPER)
    plt.close(fig)
    print("->", out)


def bottleneck_scan(what, values=None, L=1000, p=0.5, vmax=5, T=100000, seed=0, out=None):
    """Flow through the exit (and density of the bulk) of the open road, against p or against vmax.

    what = 'p'    : p varies, vmax fixed.        what = 'vmax' : vmax varies, p fixed.
    The point of the paper (p = 0.5, vmax = 5) is marked with a cross.
    """
    if values is None:
        values = np.arange(0.05, 0.951, 0.05) if what == "p" else np.arange(1, 11)
    res = []
    for i, x in enumerate(tqdm(values, desc=f"open road, {what}")):
        kw = dict(p=x, vmax=vmax) if what == "p" else dict(p=p, vmax=int(x))
        rho, q, dq, _ = bottleneck(L=L, T=T, seed=seed + i, **kw)
        res.append((rho, q, dq))
    rho, q, dq = map(np.array, zip(*res))

    fig = new_figure()
    ax = fig.add_subplot(111)
    style_axes(ax)
    ax.errorbar(values, rho, fmt="s", ms=6, color=INK, zorder=4, label=r"BULK DENSITY  $\rho$")
    ax.errorbar(values, q, yerr=dq, fmt="s", ms=8, mfc=PAPER, mec=ACCENT, mew=2.5, ecolor=INK, elinewidth=1.2,
                zorder=5, label="FLOW THROUGH THE EXIT  $q$")
    ref = p if what == "p" else vmax
    k = int(np.argmin(np.abs(np.asarray(values, dtype=float) - ref)))
    if abs(values[k] - ref) < 1e-9:
        ax.plot([values[k]], [q[k]], marker="X", ms=13, mfc=ACCENT, mec=INK, mew=2, ls="none", zorder=6,
                label=f"PAPER:  p={p}  vmax={vmax}")
    ax.set_ylim(0, max(q.max(), rho.max()) * 1.15)
    ax.set_xlabel(r"SLOWDOWN PROBABILITY  $p$" if what == "p" else r"MAXIMUM SPEED  $v_{max}$",
                  fontweight="bold", labelpad=8)
    ax.set_ylabel("FLOW (CARS PER STEP)  /  DENSITY", fontweight="bold", labelpad=8)
    fixed = f"vmax={vmax}" if what == "p" else f"p={p}"
    ax.set_title(f"[ OPEN ROAD ]  L={L}  {fixed}", fontweight="bold", loc="left", pad=12)
    style_legend(ax.legend(loc="upper right", fancybox=False, framealpha=1, edgecolor=INK, facecolor=PAPER,
                           borderpad=0.8, fontsize=8))
    fig.tight_layout()
    out = out or f"bottleneck_vs_{what}_L{L}_{fixed.replace('=', '')}.png"
    fig.savefig(out, dpi=150, facecolor=PAPER)
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", nargs="?", default="all", choices=["all", "fd", "spacetime", "bottleneck", "scan"])
    parser.add_argument("--L", type=int, default=1000)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--vmax", type=int, default=5)
    parser.add_argument("--T", type=int, default=50000, help="steps of every average")
    args = parser.parse_args()

    if args.what in ("all", "fd"):
        fundamental_diagram(L=args.L, p=args.p, vmax=args.vmax, T=args.T)
    if args.what in ("all", "spacetime"):
        spacetime(p=args.p, vmax=args.vmax)
    if args.what in ("all", "scan"):
        bottleneck_scan("p", L=args.L, p=args.p, vmax=args.vmax)
        bottleneck_scan("vmax", L=args.L, p=args.p, vmax=args.vmax)
    if args.what == "bottleneck":
        rho_c, q_c, dq_c, r = bottleneck(L=args.L, p=args.p, vmax=args.vmax)
        print(f"bulk density {rho_c:.4f}   flow through the exit {q_c:.4f} +- {dq_c:.4f}   "
              f"(paper: 0.069 +- 0.002 and 0.304 +- 0.001)")
