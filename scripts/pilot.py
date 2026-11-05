"""Pilot / competence check on train+val only (never touches test, gsweep or OOD splits).

Trains one model on a world variant and reports whether it passes the competence gate:
    in-range g-probe R^2 on h_C (2-fold within val), and the slope of g_eff vs g from open-loop
    rollouts of val episodes.

    python scripts/pilot.py --world w12 --family state --steps 6000 --batch 64 --cur_frac 0.6
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes.analysis import C, Episodes, encode_context, g_eff, pos_error, rollout_from_h  # noqa: E402
from vibes.data import SPECS, generate, load, save  # noqa: E402
from vibes.train import DEFAULTS, train  # noqa: E402

WORLDS = {
    "v1": {},
    "w12": dict(n_choices=(1, 2), vx_max=0.75),
    "w123": dict(n_choices=(1, 2, 3), vx_max=0.75),
    "w12fast": dict(n_choices=(1, 2)),
    "w1": dict(n_choices=(1,), vx_max=0.75),
}


def world_data(w, split, N, T=40):
    path = ROOT / "data" / "pilot" / f"{w}_{split}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        SPECS[f"{w}_{split}"] = replace(SPECS["train"], **WORLDS[w])
        from vibes import data as D
        D.SEEDS[f"{w}_{split}"] = {"train": 0, "val": 1}[split]
        save(generate(f"{w}_{split}", N, T=T), path)
    return path


def competence(run_dir, val_path):
    from vibes.analysis import load_run
    m, cfg = load_run(run_dir)
    p = load(val_path)
    ep = Episodes.from_split(p)
    h = encode_context(m, ep).cpu().numpy()
    g = p["g"]
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), h, g, cv=2)
    r2 = 1 - ((pred - g) ** 2).sum() / ((g - g.mean()) ** 2).sum()
    pos = rollout_from_h(m, ep, torch.as_tensor(h, device="cuda"), 32)
    ge = g_eff(pos, ep)
    ok = np.isfinite(ge)
    slope = np.polyfit(g[ok], ge[ok], 1)[0] if ok.sum() > 10 else np.nan
    err = pos_error(pos, p["states"][:, C:C + 32], p["active"])
    err8 = pos_error(pos[:, :8], p["states"][:, C:C + 8], p["active"])
    return dict(r2_g=float(r2), geff_slope=float(slope), geff_valid=float(ok.mean()),
                geff_mean=float(np.nanmean(ge)), poserr_px=float(err.mean() * 64),
                poserr8_px=float(err8.mean() * 64))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="w12")
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval_only", action="store_true")
    for k, v in DEFAULTS.items():
        if k in ("train_path", "val_path"):
            continue
        ap.add_argument(f"--{k}", type=(lambda s: s.lower() == "true") if isinstance(v, bool) else type(v),
                        default=v)
    a = vars(ap.parse_args())
    w, tag, eval_only = a.pop("world"), a.pop("tag"), a.pop("eval_only")
    a["train_path"] = str(world_data(w, "train", 16000))
    a["val_path"] = str(world_data(w, "val", 1000))
    name = f"{w}_{a['family']}_{a['size']}_b{a['batch']}_s{a['steps']}_c{a['cur_frac']}{tag}"
    out = ROOT / "runs" / "pilot" / name
    if not eval_only:
        train(a, out)
    res = competence(out, a["val_path"])
    res["name"] = name
    print("COMPETENCE", json.dumps(res), flush=True)
    with open(ROOT / "runs" / "pilot" / "competence.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
