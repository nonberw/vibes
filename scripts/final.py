"""Everything after the grid, in one command:
aggregate -> figures -> paper macros/tables -> README table -> Russian overview numbers -> PDFs.

    python scripts/final.py            (VIBES_NB=200 for a quick pass)
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TECTONIC = os.environ.get("TECTONIC", r"C:\Users\Timur\tools\tectonic\tectonic.exe")


def run(cmd, cwd=ROOT):
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"failed: {cmd}")


def main():
    run([PY, "scripts/aggregate2.py"])
    run([PY, "scripts/paper_tables.py"])
    run([PY, "scripts/readme_table.py"])
    run([PY, "scripts/overview_numbers.py"])
    if (ROOT / "runs" / "v2" / "dyn_rec_M_gru_train_100k_s0" / "done").exists():
        run([PY, "scripts/fig_world2.py"])
    tex = Path(TECTONIC)
    if tex.exists():
        run([str(tex), "main.tex"], cwd=ROOT / "paper")
        run([str(tex), "vibes_overview_ru.tex"], cwd=ROOT / "docs")
    else:
        print("tectonic not found; skipping PDF builds")


if __name__ == "__main__":
    main()
