"""Exploratory E8: does a frozen, large, natural-video JEPA encoder (V-JEPA 2 ViT-L) carry an
out-of-range gravity estimate in its features, and is its position readout stable OOD?

Protocol (frozen features only; no fine-tuning):
  1. Render the 10 context frames of val (in range) and gsweep episodes, upscale 64 -> 256.
  2. Encode with V-JEPA 2 (HF `facebook/vjepa2-vitl-fpc64-256`), mean-pool tokens per tubelet,
     concatenate the 5 tubelet means (5 x 1024 features).
  3. Ridge probe val features -> g; apply to gsweep: in-range slope and rho_hi/lo of the reading.
  4. Ridge probe val features (per frame) -> object positions by colour; readout error in range
     vs out of range on gsweep (perception check).

    python -P scripts/vjepa2_probe.py [--device cpu] [--n_val 2000]
Writes results/v2/vjepa2_probe.json and .npz.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes import world  # noqa: E402
from vibes.analysis2 import Meta, pos_err_px  # noqa: E402
from vibes.render import PALETTE, material_bg, render  # noqa: E402
from vibes.world import C  # noqa: E402

MODEL = "facebook/vjepa2-vitl-fpc64-256"


def clips(p, idx, device):
    meta = Meta(p, idx)
    xy = torch.as_tensor(p["states"][idx, :C, :, :2], dtype=torch.float32, device=device)
    x = render(xy, meta.t("kind", torch.int64).to(device), meta.t("size").to(device),
               meta.t("color", torch.int64).to(device), meta.t("active", torch.bool).to(device),
               bg=material_bg(meta.t("e"), meta.t("mu")).to(device))                 # (B, C, 3, 64, 64)
    x = F.interpolate(x.flatten(0, 1), size=(256, 256), mode="bilinear", align_corners=False)
    return x.view(len(idx), C, 3, 256, 256)


@torch.no_grad()
def encode(model, x, mean, std):
    """x (B, T, 3, H, W) in [0, 1] -> pooled features (B, T//2 * D)."""
    x = (x - mean) / std
    out = model(pixel_values_videos=x).last_hidden_state                       # (B, N, D)
    B, N, D = out.shape
    T2 = x.shape[1] // 2
    per_frame = out.view(B, T2, N // T2, D).mean(2)                            # (B, T2, D)
    return per_frame.flatten(1).float().cpu().numpy(), per_frame.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_gsweep", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import VJEPA2Model
    model = VJEPA2Model.from_pretrained(MODEL, torch_dtype=torch.float16 if a.device == "cuda" else torch.float32)
    model = model.to(a.device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=a.device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=a.device).view(1, 1, 3, 1, 1)
    out = {}
    t0 = time.time()
    feats, frames = {}, {}
    for split, n in (("val", a.n_val), ("gsweep", a.n_gsweep)):
        p = world.load(split)
        N = len(p["g"]) if n is None else min(n, len(p["g"]))
        fs, fr = [], []
        for i in range(0, N, a.batch):
            idx = np.arange(i, min(i + a.batch, N))
            x = clips(p, idx, a.device)
            if a.device == "cuda":
                x = x.half()
            f, pf = encode(model, x, mean, std)
            fs.append(f)
            fr.append(pf)
            if i % (a.batch * 25) == 0:
                print(f"{split} {i}/{N} {time.time() - t0:.0f}s", flush=True)
        feats[split] = np.concatenate(fs)
        frames[split] = np.concatenate(fr)
    v, g = world.load("val"), world.load("gsweep")
    Nv = len(feats["val"])
    Ng = len(feats["gsweep"])
    g = {k: np.asarray(vv)[:Ng] for k, vv in g.items()}
    # gravity probe
    probe = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(feats["val"], v["g"][:Nv])
    r2_val = probe.score(feats["val"], v["g"][:Nv])  # in-sample; see cv below
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(RidgeCV(alphas=np.logspace(-2, 5, 15)), feats["val"], v["g"][:Nv], cv=5).mean()
    gp = probe.predict(feats["gsweep"])
    gg = g["g"]
    grid = np.round(np.arange(1, 12.01, 0.5), 2)
    curve = [float(gp[gg == x].mean()) for x in grid]
    inr = (gg >= 4) & (gg <= 8)
    s = np.polyfit(gg[inr], gp[inr], 1)[0]
    gm = lambda x1, x2: np.mean([gp[gg == x1].mean(), gp[gg == x2].mean()])
    rho_hi = (gm(11.5, 12.0) - gm(7.5, 8.0)) / (4 * s)
    rho_lo = (gm(4.0, 4.5) - gm(1.0, 1.5)) / (3 * s)
    out.update(model=MODEL, n_val=int(Nv), r2_g_val_cv=float(cv), r2_g_val_insample=float(r2_val),
               probe_slope_in_range=float(s), probe_rho_hi=float(rho_hi), probe_rho_lo=float(rho_lo),
               probe_curve_grid=grid.tolist(), probe_curve=curve)
    # perception check: per-tubelet features -> positions of the object of each colour
    # (target: colour-indexed (cos, sin, y) at the tubelet's second frame)
    def targets(p, N):
        from vibes.stage2 import colour_targets
        xy = torch.as_tensor(p["states"][:N, 1:C:2, :, :2])                 # frames 1,3,5,7,9
        tgt, mask = colour_targets(xy, torch.as_tensor(p["color"][:N]), torch.as_tensor(p["active"][:N]))
        return tgt.numpy(), mask.numpy()
    tv, mv = targets(v, Nv)
    tg, mg = targets(g, len(gg))
    Xv = frames["val"].reshape(-1, frames["val"].shape[-1])                    # (Nv*5, D)
    Xg = frames["gsweep"].reshape(-1, frames["gsweep"].shape[-1])
    errs = np.full((len(gg), 5), np.nan)
    for c in range(len(PALETTE)):
        rows_v = np.repeat(mv[:, c], 5)
        if rows_v.sum() < 50:
            continue
        Yv = tv[:, :, c, :].reshape(-1, 3)[rows_v]
        reg = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Xv[rows_v], Yv)
        rows_g = np.repeat(mg[:, c], 5)
        pred = reg.predict(Xg[rows_g]).reshape(-1, 3)
        Yg = tg[:, :, c, :].reshape(-1, 3)[rows_g]
        x_p = (np.arctan2(pred[:, 1], pred[:, 0]) / (2 * np.pi)) % 1.0
        x_t = (np.arctan2(Yg[:, 1], Yg[:, 0]) / (2 * np.pi)) % 1.0
        dx = x_p - x_t
        dx -= np.round(dx)
        e = np.hypot(dx, pred[:, 2] - Yg[:, 2]) * 64
        ep = np.repeat(np.arange(len(gg)), 5)[rows_g]
        fr = np.tile(np.arange(5), len(gg))[rows_g]
        for ei, fi, ee in zip(ep, fr, e):
            errs[ei, fi] = ee if np.isnan(errs[ei, fi]) else (errs[ei, fi] + ee) / 2
    ep_err = np.nanmean(errs, 1)
    ood = (gg < 4) | (gg > 8)
    out.update(readout_err_px_in=float(np.nanmean(ep_err[~ood])), readout_err_px_ood=float(np.nanmean(ep_err[ood])),
               seconds=time.time() - t0, device=a.device)
    (ROOT / "results" / "v2").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "results" / "v2" / f"vjepa2_probe{a.tag}.npz", gs_g=gg, gs_gprobe=gp, readout_err=ep_err)
    (ROOT / "results" / "v2" / f"vjepa2_probe{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
