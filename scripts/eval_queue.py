"""Evaluate every finished run (runs/*/done) that has no results yet; keep polling until all
`--expect` runs are evaluated.  python scripts/eval_queue.py --expect 25"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", type=int, default=25)
    a = ap.parse_args()
    while True:
        done = sorted(p.parent for p in (ROOT / "runs").glob("*/done") if not p.parent.name.startswith("_"))
        todo = [d for d in done if not (ROOT / "results" / "runs" / f"{d.name}.json").exists()]
        for d in todo:
            print(">> eval", d.name, flush=True)
            with open(ROOT / "runs" / f"{d.name}.eval.out", "w") as fh:
                r = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "evaluate.py"), f"runs/{d.name}"],
                                   cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            print("<< eval", d.name, "exit", r.returncode, flush=True)
        n = len(list((ROOT / "results" / "runs").glob("*.json")))
        if n >= a.expect:
            break
        time.sleep(60)
