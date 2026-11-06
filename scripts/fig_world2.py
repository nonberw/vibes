"""Figure: ground truth vs REC/JEPA predictions (v2 chunk models, decoded through the post-hoc
readout decoder).   python scripts/fig_world2.py"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes import world  # noqa: E402
from vibes.analysis2 import WM, Meta  # noqa: E402
from vibes.render import material_bg, render  # noqa: E402
from vibes.world import C, R  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = world.load("test")
    eps = np.array([3, 11, 25])
    frames = [0, 5, 9, 11, 14, 17, 20, 23, 26, 29]
    meta = Meta(p, eps)
    kw = dict(kind=meta.t("kind", torch.int64), size=meta.t("size"), color_id=meta.t("color", torch.int64),
              active=meta.t("active", torch.bool), bg=material_bg(meta.t("e"), meta.t("mu")))
    with torch.no_grad():
        gt = render(torch.as_tensor(p["states"][eps, :, :, :2], device="cuda"), **kw).permute(0, 1, 3, 4, 2).cpu().numpy()
    rows = [("ground truth", gt)]
    for fam in ("rec", "jepa"):
        rd = ROOT / "runs" / "v2" / f"dyn_{fam}_M_gru_train_100k_s0"
        if not (rd / "done").exists():
            continue
        wm = WM(rd)
        with torch.no_grad():
            out = wm.run(p["states"][eps, :C, :, :2], meta)
            z = out["zn"][:, -1]
            pr = wm.predict_latents(out["h"], z) * wm.dyn.sd + wm.dyn.mu
            img = wm.readout(pr.flatten(0, 1)).view(len(eps), R, 3, 64, 64).clamp(0, 1).permute(0, 1, 3, 4, 2).cpu().numpy()
        rows.append((fam.upper(), img))
    fig, ax = plt.subplots(len(eps) * len(rows), len(frames), figsize=(len(frames) * 0.9, len(eps) * len(rows) * 0.95))
    r = 0
    for i, e in enumerate(eps):
        for name, img in rows:
            for j, t in enumerate(frames):
                a = ax[r, j]
                if name == "ground truth" or t < C:
                    a.imshow(gt[i, t])
                else:
                    a.imshow(img[i, t - C])
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
    (ROOT / "results" / "v2").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "v2" / "fig_world.png", dpi=180)
    fig.savefig(ROOT / "paper" / "fig_world.pdf")
    print("saved", [n for n, _ in rows])


if __name__ == "__main__":
    main()
