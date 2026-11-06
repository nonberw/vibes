"""Stage 1: train a frame encoder.   python scripts/train_perception.py --kind rec --size M --seed 0"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes.perception import train_perception  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="rec")
    ap.add_argument("--size", default="M")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pool", default="train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--w_var", type=float, default=1.0)
    ap.add_argument("--w_cov", type=float, default=0.04)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--enc", default="conv")
    ap.add_argument("--beta", type=float, default=0.0)
    a = ap.parse_args()
    out = a.out or f"runs/v2/perc_{a.kind}_{a.size}_{a.pool}_s{a.seed}"
    train_perception(a.kind, a.size, a.seed, a.steps, ROOT / out, pool_name=a.pool, batch=a.batch,
                     w_var=a.w_var, w_cov=a.w_cov, horizon=a.horizon, enc=a.enc, beta=a.beta)
