"""Fill the results table in README.md from results/v2/summary.json.   python scripts/readme_table.py"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMS = ("state", "rec", "jepa")


def f(x, nd=2):
    try:
        x = float(x[0] if isinstance(x, list) else x)
        return "--" if x != x else f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def ci(v):
    return f"{f(v)} [{f(v[1])}, {f(v[2])}]" if isinstance(v, list) and len(v) == 3 else f(v)


def pct(v):
    g = lambda x: f"{100 * float(x):.0f}%"
    return f"{g(v[0])} [{g(v[1])}, {g(v[2])}]"


def main():
    S = json.loads((ROOT / "results" / "v2" / "summary.json").read_text())
    C = S["configs"]
    cfg = {k: C.get(f"{k}_M_gru_train_100000") for k in FAMS}
    ceil = {k: S.get(f"ceiling_{k}") for k in FAMS}
    rows = ["| | STATE (true positions) | REC (pixels) | JEPA (latents) |", "|---|---|---|---|"]

    def row(label, fn):
        cells = []
        for k in FAMS:
            try:
                cells.append(fn(k))
            except (KeyError, TypeError, IndexError):
                cells.append("--")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    def data_cells(k, key, nd=2):
        out = []
        for n in ("1000", "8000", "100000"):
            o = C.get(f"{k}_M_gru_train_{n}")
            v = o[key] if o else None
            out.append(f(np.mean(v) if isinstance(v, list) and len(v) != 3 else v, nd) if v is not None else "--")
        return " / ".join(out)

    row("seeds", lambda k: str(cfg[k]["n_seeds"]))
    row("competence gate: in-range slope of g_eff vs g (need >= 0.5)", lambda k: f(np.mean(cfg[k]["geff_slope"])))
    row("in-range 20-frame error, px", lambda k: f(np.mean(cfg[k]["test_err_px"]), 1))
    row("gravity probe R^2, trained / untrained dynamics",
        lambda k: f"{f(np.mean(cfg[k]['r2_g_trained']))} / {f(np.mean(cfg[k]['r2_g_untrained']))}")
    row("**belief reading of out-of-range g**, slope above / below range (1 = tracks it, 0 = clamps)",
        lambda k: f"{ci(cfg[k]['inf_hi'])} / {ci(cfg[k]['inf_lo'])}")
    row("perception: extra position error on out-of-range frames, px", lambda k: ci(cfg[k]["perception_ood_excess_px"]))
    row("steering validity: in-range transfer slope (need >= 0.5), CER",
        lambda k: f"{ci(cfg[k]['exe_s'])}, {f(cfg[k]['cer_in_range'])}")
    row("pre-registered natural rho_hi (1 = law, 0 = clamp)", lambda k: ci(cfg[k]["nat_hi"]))
    row("pre-registered natural rho_lo (biased upward, see paper)", lambda k: ci(cfg[k]["nat_lo"]))
    row("pre-registered locus label, above / below range", lambda k: f"{cfg[k]['locus_hi']} / {cfg[k]['locus_lo']}")
    row("*exploratory* law fraction, natural, above / below range",
        lambda k: f"{ci(cfg[k]['lambda_nat_hi'])} / {ci(cfg[k]['lambda_nat_lo'])}")
    row("*exploratory* law fraction after writing true g into the belief, above / below",
        lambda k: f"{f(cfg[k]['lambda_steer_hi'])} / {f(cfg[k]['lambda_steer_lo'])}"
                  + ("" if k == "state" else " (write not valid)"))
    row("*exploratory* held-out law fraction above range: natural -> optimized belief -> WIDE ceiling",
        lambda k: " -> ".join(f(ceil[k]["lambda_hold_hi"][m]) for m in ("nat", "inv_h", "wide_ceiling")))
    row("*exploratory* held-out law fraction below range: natural -> optimized belief -> WIDE ceiling",
        lambda k: " -> ".join(f(ceil[k]["lambda_hold_lo"][m]) for m in ("nat", "inv_h", "wide_ceiling")))
    row("*exploratory* share of the gap to WIDE closed by the optimized belief, above / below",
        lambda k: f"{pct(ceil[k]['gap_closed_inv_h_hi'])} / {pct(ceil[k]['gap_closed_inv_h_lo'])}")
    row("anti-physics control: law fraction toward upward gravity under optimized belief",
        lambda k: f(cfg[k]["lambda_inv_h_anti"]))
    row("data scaling 1k / 8k / 100k episodes: in-range error, px", lambda k: data_cells(k, "test_err_px", 1))
    row("data scaling 1k / 8k / 100k: *exploratory* law fraction above range", lambda k: data_cells(k, "lambda_nat_hi"))
    row("energy drift (median relative error), frames 0-20 -> 60-80",
        lambda k: f"{f(cfg[k]['drift_E_median_by_chunk'][0])} -> {f(cfg[k]['drift_E_median_by_chunk'][-1])}")
    text = "\n".join(rows)
    hyp = S.get("pvals_holm", {})
    if hyp:
        sup = sorted(k for k, v in hyp.items() if v < 0.05)
        text += (f"\n\nPre-registered confirmatory tests surviving Holm correction: **{len(sup)} of {len(hyp)}** "
                 f"({', '.join(sup)}). Full table: [`results/v2/hypotheses.md`](results/v2/hypotheses.md).")
    vp = ROOT / "results" / "v2" / "vjepa2_probe.json"
    if vp.exists():
        V = json.loads(vp.read_text())
        text += (f"\n\nOutside reference: a ridge probe on frozen V-JEPA 2 (ViT-L) features of the same context frames "
                 f"reads gravity at cross-validated R^2 {f(V['r2_g_val_cv'])}, and its out-of-range slope above the "
                 f"range is {f(V['probe_rho_hi'])}.")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    new = re.sub(r"<!-- RESULTS_TABLE_START -->.*?<!-- RESULTS_TABLE_END -->",
                 lambda m: "<!-- RESULTS_TABLE_START -->\n" + text + "\n<!-- RESULTS_TABLE_END -->", readme, flags=re.S)
    (ROOT / "README.md").write_text(new, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
