"""Generate paper/tables_hyp.tex and paper/tables_cfg.tex from results/v2/summary.json, and
paper/numbers.tex via paper_numbers.py (with '--' stubs for any macro the .tex files use that has
no value yet, so the paper always compiles).   python scripts/paper_tables.py"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def fmt(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if x != x else f"{x:.{nd}f}"


def main():
    sp = ROOT / "results" / "v2" / "summary.json"
    S = json.loads(sp.read_text()) if sp.exists() else {"configs": {}, "pvals": {}, "pvals_holm": {}}
    # hypotheses table
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Test & $p$ & $p$ (Holm) & supported\\\\", "\\midrule"]
    for k in sorted(S.get("pvals", {})):
        p, ph = S["pvals"][k], S["pvals_holm"][k]
        L.append(f"{k.replace('_', ' ')} & {fmt(p, 3)} & {fmt(ph, 3)} & {'yes' if ph < 0.05 else 'no'}\\\\")
    if len(L) == 4:
        L.append("(no results yet) & & & \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tables_hyp.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    # per-config table
    L = ["\\begin{tabular}{lccccccc}", "\\toprule",
         "Config & seeds & slope & $\\rho_{\\rm hi}$ nat & $\\rho_{\\rm hi}$ inv-$z$ & $\\rho_{\\rm hi}$ inv-$h$ & "
         "$\\rho_{\\rm lo}$ nat & locus hi / lo\\\\", "\\midrule"]
    for tag, o in S.get("configs", {}).items():
        import numpy as np
        sl = np.mean([v for v in o["geff_slope"] if v is not None])
        L.append(f"{tag.replace('_', ' ')} & {o['n_seeds']} & {fmt(sl)} & {fmt(o['nat_hi'][0])} & "
                 f"{fmt(o['inv_z_hi'][0])} & {fmt(o['inv_h_hi'][0])} & {fmt(o['nat_lo'][0])} & "
                 f"{o['locus_hi']} / {o['locus_lo']}\\\\")
    if len(L) == 4:
        L.append("(no results yet) & & & & & & & \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tables_cfg.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    # numbers.tex + stubs
    if sp.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "paper_numbers.py")], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    nums = (PAPER / "numbers.tex").read_text(encoding="utf-8") if (PAPER / "numbers.tex").exists() else ""
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", nums))
    used = set()
    for f in PAPER.glob("*.tex"):
        if f.name in ("numbers.tex", "stubs.tex"):
            continue
        used |= set(re.findall(r"\\((?:Rec|Jepa|State|P)[A-Za-z]+)", f.read_text(encoding="utf-8")))
    stubs = ["% stubs for macros without a value yet (generated)"]
    for m in sorted(used - defined):
        stubs.append(f"\\providecommand{{\\{m}}}{{--}}")
    (PAPER / "stubs.tex").write_text("\n".join(stubs) + "\n", encoding="utf-8")
    if "\\input{stubs}" not in nums:
        (PAPER / "numbers.tex").write_text(nums + "\n\\input{stubs}\n", encoding="utf-8")
    print(f"{len(defined)} macros defined, {len(used - defined)} stubbed")


if __name__ == "__main__":
    main()
