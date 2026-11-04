"""Generate every dataset split into data/*.npz (deterministic, ~1 minute)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vibes.data import generate, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": (16000, 40), "val": (1000, 40), "test": (2000, 40), "gsweep": (2300, 40),
          "long": (1000, 72), "count45": (1000, 40), "speed": (1000, 40),
          "massratio": (1000, 40), "mass_ctrl": (1000, 40)}

if __name__ == "__main__":
    (ROOT / "data").mkdir(exist_ok=True)
    for name, (N, T) in SPLITS.items():
        t = time.time()
        p = generate(name, N, T=T)
        save(p, ROOT / "data" / f"{name}.npz")
        print(f"{name:10s} N={N:6d} T={T} {time.time() - t:.1f}s")
