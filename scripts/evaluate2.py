"""Evaluate one v2 chunk world model on every pre-registered experiment (E1-E6).

    python scripts/evaluate2.py runs/v2/dyn_state_M_gru_train_100k_s0 [--skip e4,e5]
Writes results/v2/runs/<name>.npz and .json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes import experiments2 as X  # noqa: E402
from vibes.analysis2 import WM, competence  # noqa: E402

OUT = ROOT / "results" / "v2" / "runs"


def main(run_dir, skip=()):
    run_dir = Path(run_dir) if Path(run_dir).is_absolute() else ROOT / run_dir
    name = run_dir.name
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wm = WM(run_dir)
    cfg = wm.cfg
    summ = {"name": name, **{k: cfg[k] for k in ("family", "size", "seed", "n_train", "core", "split")}}
    res = {}
    summ.update(competence(wm, "val"))
    print(f"[{name}] competence {json.dumps({k: round(v, 3) for k, v in summ.items() if isinstance(v, float)})}",
          flush=True)
    r1, probes, _ = X.e1_probes(wm, X.untrained_copy(wm))
    res.update({k: v for k, v in r1.items() if isinstance(v, np.ndarray)})
    summ.update({k: v for k, v in r1.items() if not isinstance(v, np.ndarray)})
    print(f"[{name}] E1 {time.time() - t0:.0f}s r2_g={summ['r2_g_trained']:.3f}", flush=True)
    res.update(X.e2_natural(wm, probes["g"]))
    print(f"[{name}] E2 {time.time() - t0:.0f}s", flush=True)
    u, _ = X.matched_direction(wm)
    res["u_g"] = u
    st = X.e3_steering(wm, probes["g"], u)
    summ["steer_w_dot_u"] = st.pop("steer_w_dot_u")
    res.update(st)
    print(f"[{name}] E3 steering {time.time() - t0:.0f}s", flush=True)
    res.update(X.e3_inversion(wm))
    print(f"[{name}] E3 inversion {time.time() - t0:.0f}s", flush=True)
    if "e4" not in skip:
        res.update(X.e4_velocity(wm))
        print(f"[{name}] E4 {time.time() - t0:.0f}s", flush=True)
    if "e5" not in skip:
        res.update(X.e5_drift(wm))
        print(f"[{name}] E5 {time.time() - t0:.0f}s", flush=True)
    res.update(X.e6_ood(wm))
    for k in list(res):
        if k.endswith("_err_px") and k.split("_err_px")[0] in ("test", "count45", "speed", "massratio", "mass_ctrl"):
            summ[k] = float(np.mean(res[k]))
    summ["eval_seconds"] = time.time() - t0
    np.savez_compressed(OUT / f"{name}.npz", **res)
    (OUT / f"{name}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    for r in a.runs:
        main(r, tuple(s for s in a.skip.split(",") if s))
