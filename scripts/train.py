"""Train one world model.  Example:
    python scripts/train.py --family rec --size M --seed 0 --n_train 16000 --out runs/rec_M_16k_s0
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes.train import DEFAULTS, train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=(lambda s: s.lower() == "true") if isinstance(v, bool) else type(v),
                        default=v)
    ap.add_argument("--out", required=True)
    a = vars(ap.parse_args())
    out = a.pop("out")
    a["train_path"] = str(ROOT / a["train_path"])
    a["val_path"] = str(ROOT / a["val_path"])
    print(json.dumps(a))
    train(a, ROOT / out)
