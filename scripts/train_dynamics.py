"""Stage 2: train a chunk world model on frozen latents (REC/JEPA) or features (STATE).

    python scripts/train_dynamics.py --family state --size M --seed 0 --steps 60000
    python scripts/train_dynamics.py --family rec --perc runs/v2/perc_rec_M_train_s0 --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes.analysis2 import WM, competence  # noqa: E402
from vibes.stage2 import train_dynamics  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="state")
    ap.add_argument("--size", default="M")
    ap.add_argument("--core", default="gru")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--perc", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ntag = f"{a.n_train // 1000}k" if a.n_train else "100k"
    out = a.out or f"runs/v2/dyn_{a.family}_{a.size}_{a.core}_{a.split}_{ntag}_s{a.seed}"
    train_dynamics(a.family, a.size, a.seed, a.steps, ROOT / out, perc_dir=a.perc and str(ROOT / a.perc),
                   split=a.split, n_train=a.n_train, core=a.core, batch=a.batch, lr=a.lr)
    wm = WM(ROOT / out)
    comp = competence(wm, "wide_val" if a.split == "wide" else "val")
    print("COMPETENCE", json.dumps(comp), flush=True)
    (ROOT / out / "competence.json").write_text(json.dumps(comp, indent=1))
