"""Run the pre-registered training grid sequentially (one GPU job at a time; the GPU is shared).

Skips runs that already have a `done` marker. Order: main configs first, then scaling cells.
    python scripts/run_grid.py            # everything
    python scripts/run_grid.py --only main
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAIN = [(f, "M", 16000, s) for s in (0, 1, 2) for f in ("rec", "jepa", "state")]
SCALE = [(f, sz, 16000, s) for s in (0, 1) for f in ("rec", "jepa") for sz in ("S", "L")] + \
        [(f, "M", n, s) for s in (0, 1) for f in ("rec", "jepa") for n in (1000, 4000)]


def name(f, sz, n, s):
    return f"{f}_{sz}_{n // 1000}k_s{s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["main", "scale", "all"], default="all")
    ap.add_argument("--family", default=None)
    ap.add_argument("--exclude_family", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    grid = {"main": MAIN, "scale": SCALE, "all": MAIN + SCALE}[a.only]
    grid = [c for c in grid if (a.family is None or c[0] == a.family) and c[0] != a.exclude_family]
    for f, sz, n, s in grid:
        out = ROOT / "runs" / name(f, sz, n, s)
        if (out / "done").exists():
            continue
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / "train.py"), "--family", f, "--size", sz,
               "--n_train", str(n), "--seed", str(s), "--device", a.device, "--out", f"runs/{name(f, sz, n, s)}"]
        print(">>", " ".join(cmd), flush=True)
        with open(ROOT / "runs" / f"{name(f, sz, n, s)}.out", "w") as fh:
            r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        print("<<", name(f, sz, n, s), "exit", r.returncode, flush=True)


if __name__ == "__main__":
    main()
