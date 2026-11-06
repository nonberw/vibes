"""Pilot check (pilot split only): position precision of the post-hoc decoder readout of an encoder.

    python scripts/check_readout.py runs/v2/perc_rec_M_train_s0 [more ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes import world  # noqa: E402
from vibes.analysis2 import pos_err_px  # noqa: E402
from vibes.perception import load_perception  # noqa: E402
from vibes.stage2 import decoder_positions, fit_readout_decoder  # noqa: E402
from vibes.world import Pool  # noqa: E402

if __name__ == "__main__":
    p = world.ensure("pilot")
    pool = Pool(p, "cuda")
    idx = torch.arange(300, device="cuda")
    for d in sys.argv[1:]:
        d = Path(d) if Path(d).is_absolute() else ROOT / d
        perc, _ = load_perception(d)
        dec = fit_readout_decoder(d)
        with torch.no_grad():
            xy = pool.window(idx, 0, 40)[:, ::4]
            x = pool.render(idx, xy)
            z = perc.encode(x).float()
            pos = decoder_positions(dec, z, pool.color[idx], pool.active[idx], pool.bg[idx])
        e = pos_err_px(pos.cpu().numpy(), p["states"][:300, ::4], p["active"][:300])
        res = {"perception": d.name, "readout_err_px_mean": float(e.mean()), "median": float(np.median(e)),
               "p90": float(np.quantile(e, 0.9))}
        print(json.dumps(res), flush=True)
        (d / "readout_check_pilot.json").write_text(json.dumps(res, indent=1))
