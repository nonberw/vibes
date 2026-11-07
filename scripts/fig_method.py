"""Schematic of the measurement pipeline and nested inversion (paper/fig_method.pdf)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 7.5})

BLUE, RED, GREY, GREEN = "#2a7f9e", "#c8553d", "#777777", "#4a8f4a"


def box(ax, x, y, w, h, text, fc="white", ec="black", lw=0.9, fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None)


def arrow(ax, x0, y0, x1, y1, color="black", ls="-", lw=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, ls=ls))


def pipeline(ax, title, opt_z=False, opt_h=False, steer=False):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.text(0.1, 4.0, title, fontsize=8, fontweight="bold", va="top")
    # frames
    for i in range(4):
        ax.add_patch(Rectangle((0.3 + 0.28 * i, 2.0 + 0.1 * i), 0.9, 0.9, fc="#f3f3f3", ec="#999999", lw=0.6))
    ax.text(1.0, 1.75, "10 context\nframes", ha="center", va="top", fontsize=6.5)
    arrow(ax, 1.7, 2.55, 2.4, 2.55)
    box(ax, 2.4, 2.1, 1.5, 0.9, "frozen\nencoder")
    arrow(ax, 3.9, 2.55, 4.6, 2.55)
    zc = BLUE if opt_z else "black"
    box(ax, 4.6, 2.1, 1.3, 0.9, "latents\n$z_{1..10}$", ec=zc, lw=1.6 if opt_z else 0.9)
    arrow(ax, 5.9, 2.55, 6.6, 2.55)
    box(ax, 6.6, 2.1, 1.5, 0.9, "belief\nGRU")
    arrow(ax, 8.1, 2.55, 8.8, 2.55)
    hc = RED if opt_h else ("#7b5ea7" if steer else "black")
    box(ax, 8.8, 2.1, 1.0, 0.9, "$h_C$", ec=hc, lw=1.6 if (opt_h or steer) else 0.9, bold=True)
    arrow(ax, 9.3, 2.1, 9.3, 1.5)
    box(ax, 7.0, 0.6, 2.9, 0.9, "decoder: 20 future latents\n$\\rightarrow$ positions, $g_{\\rm eff}$, $\\lambda$")
    if opt_z:
        ax.text(5.25, 3.35, "optimize", color=BLUE, ha="center", fontsize=7, fontweight="bold")
    if opt_h:
        ax.text(9.3, 3.35, "optimize", color=RED, ha="center", fontsize=7, fontweight="bold")
    if steer:
        ax.text(9.3, 3.35, "$+\\,\\delta\\, u_g$", color="#7b5ea7", ha="center", fontsize=8, fontweight="bold")
    if opt_z or opt_h:
        ax.text(3.4, 0.9, "loss: first 4 predicted frames\nmetrics: held-out frames 4--13", ha="center", va="center",
                fontsize=6.5, color=GREY, style="italic")
        arrow(ax, 5.3, 1.05, 7.0, 1.05, color=GREY, ls=":", lw=0.8)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.0))
    pipeline(axes[0, 0], "(a) natural: where does the model's own belief take it?")
    pipeline(axes[0, 1], "(b) linear write: steer $h_C$ along the matched-pair gravity direction", steer=True)
    pipeline(axes[1, 0], "(c) invert context latents (reachability): frozen GRU and decoder", opt_z=True)
    pipeline(axes[1, 1], "(d) invert belief (execution): is there any $h_C$ that continues the law?", opt_h=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(ROOT / "paper" / "fig_method.pdf")
    fig.savefig(ROOT / "results" / "v2" / "fig_method.png", dpi=200)
    print("saved")


if __name__ == "__main__":
    main()
