"""Figure: example episodes (ground truth) and, if trained runs exist, open-loop rollouts.

    python scripts/fig_world.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes.analysis import C, Episodes, encode_context, load_run  # noqa: E402
from vibes.data import load  # noqa: E402
from vibes.render import render_np  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = load(ROOT / "data" / "test.npz")
    eps = [3, 11, 25]
    frames = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
    rows = [("ground truth", None)]
    for fam in ("rec", "jepa"):
        rd = ROOT / "runs" / f"{fam}_M_16k_s0"
        if (rd / "done").exists():
            rows.append((fam.upper(), rd))
    fig, ax = plt.subplots(len(eps) * len(rows), len(frames),
                           figsize=(len(frames) * 0.9, len(eps) * len(rows) * 0.95))
    ax = np.atleast_2d(ax)
    gt = render_np(p["states"][eps], p["kind"][eps], p["size"][eps], p["color"][eps], p["active"][eps])
    preds = {}
    for name, rd in rows[1:]:
        m, _ = load_run(rd)
        ep = Episodes.from_split(p, np.array(eps))
        with torch.no_grad():
            h = encode_context(m, ep)
            zr, _ = m.imagine(h, 40 - C)
            img = m.decode(zr).float().clamp(0, 1).permute(0, 1, 3, 4, 2).cpu().numpy()
        preds[name] = img
    r = 0
    for i, e in enumerate(eps):
        for name, _ in rows:
            for j, t in enumerate(frames):
                a = ax[r, j]
                if name == "ground truth":
                    a.imshow(gt[i, t])
                elif t >= C:
                    a.imshow(preds[name][i, t - C])
                else:
                    a.imshow(gt[i, t])
                    a.patch.set_alpha(0.0)
                a.set_xticks([])
                a.set_yticks([])
                if t < C and name != "ground truth":
                    for s in a.spines.values():
                        s.set_edgecolor("#bbbbbb")
                if r == 0:
                    a.set_title(f"t={t}" + (" (ctx)" if t < C else ""), fontsize=6)
            ax[r, 0].set_ylabel(f"ep {e}\n{name}", fontsize=6)
            r += 1
    fig.tight_layout(pad=0.2)
    fig.savefig(ROOT / "results" / "fig_world.png", dpi=180)
    fig.savefig(ROOT / "paper" / "fig_world.pdf")
    print("saved")


if __name__ == "__main__":
    main()
