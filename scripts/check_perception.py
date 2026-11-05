"""Pilot check of a stage-1 encoder (val/test frames only): how precisely can object positions be
read out of single-frame latents? Gravity inference needs ~1 px precision or better.

    python scripts/check_perception.py runs/v2/perc_rec_M_train_s0
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes.analysis2 import pos_err_px  # noqa: E402
from vibes.stage2 import Source, fit_pos_probe, probe_positions  # noqa: E402

if __name__ == "__main__":
    perc = Path(sys.argv[1])
    perc = perc if perc.is_absolute() else ROOT / perc
    fam = "rec" if "rec" in perc.name else "jepa"
    val = Source(fam, "val", perc)
    test = Source(fam, "test", perc)
    probe = fit_pos_probe(val, steps=6000)
    idx = torch.arange(test.N, device="cuda")
    errs = []
    with torch.no_grad():
        for t0 in range(0, test.T, 8):
            z = test.window(idx, t0, 1)
            pos = probe_positions(probe, z, test.pool.color[idx], test.pool.active[idx]).cpu().numpy()
            ref = test.p["states"][:, t0:t0 + 1]
            errs.append(pos_err_px(pos, ref, test.p["active"]))
    e = np.concatenate(errs)
    res = {"perception": perc.name, "probe_err_px_mean": float(e.mean()), "probe_err_px_median": float(np.median(e)),
           "probe_err_px_p90": float(np.quantile(e, 0.9))}
    print(json.dumps(res), flush=True)
    (perc / "probe_check.json").write_text(json.dumps(res, indent=1))
