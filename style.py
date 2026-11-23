"""Plot style: retro / digital-brutalist, the same as the other figures of the blog.

Monospace font, hard black box, square markers, dotted grid, one accent colour.
"""
import matplotlib.pyplot as plt

PAPER, INK, ACCENT = "#f2efe6", "#000000", "#ff3b00"

RC = {
    "font.family": ["Space Mono", "Monaco", "Courier New", "monospace"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Space Mono",
    "mathtext.it": "Space Mono:italic",
    "mathtext.bf": "Space Mono:bold",
}


def style_axes(ax, grid=True):
    ax.set_facecolor(PAPER)
    if grid:
        ax.grid(True, which="major", color=INK, lw=0.5, ls=(0, (1, 3)))
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", length=6, width=2, color=INK)
    for sp in ax.spines.values():  # closed box
        sp.set_visible(True)
        sp.set_linewidth(3)
        sp.set_color(INK)
        sp.set_capstyle("projecting")


def style_legend(leg):
    leg.get_frame().set_linewidth(2)
    return leg


def new_figure(figsize=(8, 5)):
    plt.rcParams.update(RC)
    return plt.figure(figsize=figsize, facecolor=PAPER)
