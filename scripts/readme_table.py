"""Fill the results table in README.md from results/v2/summary.json.   python scripts/readme_table.py"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def f(x, nd=2):
    try:
        x = float(x[0] if isinstance(x, list) else x)
        return "--" if x != x else f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def ci(v):
    return f"{f(v)} [{f(v[1])}, {f(v[2])}]" if isinstance(v, list) and len(v) == 3 else f(v)


def main():
    S = json.loads((ROOT / "results" / "v2" / "summary.json").read_text())
    rows = ["| | STATE (true positions) | REC (pixels) | JEPA (latents) |", "|---|---|---|---|"]
    fams = [("state", "state_M_gru_train_100000"), ("rec", "rec_M_gru_train_100000"), ("jepa", "jepa_M_gru_train_100000")]
    cfg = {k: S["configs"].get(t) for k, t in fams}

    def row(label, fn):
        cells = []
        for k, _ in fams:
            o = cfg[k]
            cells.append(fn(o) if o else "--")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    row("seeds", lambda o: str(o["n_seeds"]))
    row("competence gate (in-range slope of g_eff vs g; need >= 0.5)", lambda o: f(np.mean(o["geff_slope"])))
    row("gravity probe R^2 (trained / untrained dynamics)", lambda o: f"{f(np.mean(o['r2_g_trained']))} / {f(np.mean(o['r2_g_untrained']))}")
    row("natural rho_hi (1 = law, 0 = clamp) [95% CI]", lambda o: ci(o["nat_hi"]))
    row("natural rho_lo [95% CI]", lambda o: ci(o["nat_lo"]))
    row("probe reading rho_hi", lambda o: ci(o["inf_hi"]))
    row("linear write rho_hi", lambda o: ci(o["exe_hi"]))
    row("invert belief rho_hi (execution test)", lambda o: ci(o["inv_h_hi"]))
    row("invert context rho_hi (reachability test)", lambda o: ci(o["inv_z_hi"]))
    row("anti-physics control: held-out g_eff under invert-belief (target -4)", lambda o: ci(o["inv_h_anti_geff"]))
    row("locus, high side (pre-registered rule)", lambda o: f"**{o['locus_hi']}**")
    row("locus, low side", lambda o: f"**{o['locus_lo']}**")
    row("in-range 20-frame error (px)", lambda o: f(np.mean(o["test_err_px"]), 1))
    table = "\n".join(rows)
    extra = []
    for k, _ in fams:
        if f"H4_{k}" in S:
            extra.append(f"- {k.upper()}: natural rho_hi changes by {ci(S[f'H4_{k}']['slope_rho_hi_per_decade'])} per "
                         f"decade of training data; in-range error by {ci(S[f'H4_{k}']['slope_test_err_px_per_decade'])} px per decade.")
    text = table + ("\n\nScaling (1k -> 8k -> 100k episodes):\n" + "\n".join(extra) if extra else "")
    hyp = (ROOT / "results" / "v2" / "hypotheses.md").read_text() if (ROOT / "results" / "v2" / "hypotheses.md").exists() else ""
    if hyp:
        text += "\n\nPre-registered tests (Holm-corrected): see [`results/v2/hypotheses.md`](results/v2/hypotheses.md)."
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    new = re.sub(r"<!-- RESULTS_TABLE_START -->.*?<!-- RESULTS_TABLE_END -->",
                 "<!-- RESULTS_TABLE_START -->\n" + text + "\n<!-- RESULTS_TABLE_END -->", readme, flags=re.S)
    (ROOT / "README.md").write_text(new, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
