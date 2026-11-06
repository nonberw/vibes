"""Run the pre-registered v2 grid (PREREGISTRATION_v2.md): perception -> dynamics -> evaluation.

A small dependency-aware job queue with P parallel GPU jobs. Finished jobs (marker files) are
skipped, so the script can be restarted at any time.

    python scripts/run_v2.py --parallel 2 [--only main]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PERC_STEPS = 12000
MIN_FREE_GB = 5.0


def free_ram_gb():
    """Free physical memory (Windows GlobalMemoryStatusEx; falls back to 'plenty')."""
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 30
    except Exception:
        return 99.0
DYN_STEPS = 30000


def perc_dir(kind, seed):
    return f"runs/v2/perc_kp_{kind}_s{seed}"


def dyn_dir(fam, size, core, split, n, seed):
    ntag = f"{n // 1000}k"
    return f"runs/v2/dyn_{fam}_{size}_{core}_{split}_{ntag}_s{seed}"


def grid(only):
    """Jobs in priority order (deviation 2026-11-06): main seeds, then WIDE/LRU controls,
    data scaling (2 seeds), size scaling (1 seed)."""
    jobs = []   # (name, cmd, done_marker, deps)

    def perc(kind, s):
        d = perc_dir(kind, s)
        jobs.append((d, [PY, "-u", "scripts/train_perception.py", "--kind", kind, "--seed", str(s),
                         "--steps", str(PERC_STEPS), "--enc", "kp", "--out", d], f"{d}/done", []))

    cells = []
    for s in (0, 1, 2):
        perc("jepa", s)
        if s > 0:
            perc("rec", s)
        for fam in ("state", "rec", "jepa"):
            cells.append(("main", fam, "M", "gru", "train", 100000, s))
    if only in ("all", "scale"):
        for fam in ("state", "rec", "jepa"):
            cells.append(("wide", fam, "M", "gru", "wide", 100000, 0))
            cells.append(("lru", fam, "M", "lru", "train", 100000, 0))
        for s in (0, 1):
            for fam in ("state", "rec", "jepa"):
                for n in (1000, 8000):
                    cells.append(("data", fam, "M", "gru", "train", n, s))
        for fam in ("state", "rec", "jepa"):
            for size in ("S", "L"):
                cells.append(("size", fam, size, "gru", "train", 100000, 0))
    for _, fam, size, core, split, n, s in cells:
        d = dyn_dir(fam, size, core, split, n, s)
        cmd = [PY, "-u", "scripts/train_dynamics.py", "--family", fam, "--size", size, "--core", core,
               "--seed", str(s), "--steps", str(DYN_STEPS), "--split", split, "--out", d]
        if n != 100000:
            cmd += ["--n_train", str(n)]
        deps = []
        if fam != "state":
            cmd += ["--perc", perc_dir(fam, s)]
            deps = [f"{perc_dir(fam, s)}/done"]
        jobs.append((d, cmd, f"{d}/done", deps))
        name = Path(d).name
        jobs.append((f"eval {name}", [PY, "-u", "scripts/evaluate2.py", d],
                     f"results/v2/runs/{name}.json", [f"{d}/competence.json"]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--only", default="all", choices=["all", "main", "scale"])
    ap.add_argument("--families", default="state,rec,jepa")
    a = ap.parse_args()
    fams = a.families.split(",")
    jobs = [j for j in grid(a.only)
            if any(f"_{f}_" in Path(j[0]).name or Path(j[0]).name.startswith(f"perc_{f}") or
                   Path(j[0]).name.startswith(f"dyn_{f}_") for f in fams)]
    running = {}
    logdir = ROOT / "runs" / "v2" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    while True:
        for name, (proc, fh) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                print(f"<< {name} exit {proc.returncode}", flush=True)
                del running[name]
        pending = [j for j in jobs if not (ROOT / j[2]).exists() and j[0] not in running]
        failed_marker = lambda j: (logdir / (Path(j[0].replace(" ", "_")).name + ".failed")).exists()
        pending = [j for j in pending if not failed_marker(j)]
        if not pending and not running:
            print("all done", flush=True)
            return
        for j in pending:
            if len(running) >= a.parallel:
                break
            if running and free_ram_gb() < MIN_FREE_GB:
                break
            if all((ROOT / d).exists() for d in j[3]):
                log = open(logdir / (Path(j[0].replace(" ", "_")).name + ".log"), "w")
                print(f">> {j[0]}", flush=True)
                running[j[0]] = (subprocess.Popen(j[1], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log)
        # mark jobs that finished without producing their marker as failed (no infinite retries)
        for j in jobs:
            lg = logdir / (Path(j[0].replace(" ", "_")).name + ".log")
            if j[0] not in running and lg.exists() and not (ROOT / j[2]).exists() and lg.stat().st_size > 0:
                txt = lg.read_text(errors="ignore")
                if "Traceback" in txt:
                    (logdir / (Path(j[0].replace(" ", "_")).name + ".failed")).write_text("failed")
                    print(f"!! {j[0]} failed", flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
