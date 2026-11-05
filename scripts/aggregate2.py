"""Aggregate v2 per-run results into pre-registered statistics, tables and figures.

    python scripts/aggregate2.py            (VIBES_NB=200 for a quick pass)
Writes results/v2/summary.json, results/v2/hypotheses.md, results/v2/*.png, paper/fig_*.pdf.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes.world import G_GRID  # noqa: E402

RES = ROOT / "results" / "v2"
NB = int(os.environ.get("VIBES_NB", "2000"))
RNG = np.random.default_rng(12345)
FAMS = ("rec", "jepa", "state")
LABEL = {"rec": "REC (pixel reconstruction)", "jepa": "JEPA (latent prediction)", "state": "STATE (true positions)"}
COLOR = {"rec": "#c8553d", "jepa": "#2a7f9e", "state": "#555555"}
G_INV = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0])


# ------------------------------------------------------------------------------------ loading
def load_runs():
    runs = defaultdict(list)
    for j in sorted((RES / "runs").glob("*.json")):
        s = json.loads(j.read_text())
        d = dict(np.load(j.with_suffix(".npz")))
        d["summary"] = s
        key = (s["family"], s["size"], s["core"], s["split"], int(s["n_train"]))
        runs[key].append(d)
    return runs


def main_key(f):
    return (f, "M", "gru", "train", 100000)


# ------------------------------------------------------------------------------------ statistics
def gmean(g, v, x):
    m = (np.abs(g - x) < 1e-6) & np.isfinite(v)
    return v[m].mean() if m.any() else np.nan


def slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[m], y[m], 1)[0] if m.sum() >= 3 else np.nan


def curve_gs(g, v):
    """Pre-registered rho for the gsweep grid."""
    inr = (g >= 4 - 1e-6) & (g <= 8 + 1e-6)
    s = slope(g[inr], v[inr])
    gm = lambda a, b: np.nanmean([gmean(g, v, a), gmean(g, v, b)])
    return s, (gm(11.5, 12.0) - gm(7.5, 8.0)) / (4 * s), (gm(4.0, 4.5) - gm(1.0, 1.5)) / (3 * s)


def curve_inv(g, v):
    """Pre-registered rho for the inversion grid."""
    inr = np.isin(g, [4.0, 6.0, 8.0])
    s = slope(g[inr], v[inr])
    hi = (np.nanmean([gmean(g, v, 11.0), gmean(g, v, 12.0)]) - gmean(g, v, 8.0)) / (3.5 * s)
    lo = (gmean(g, v, 4.0) - np.nanmean([gmean(g, v, 1.0), gmean(g, v, 2.0)])) / (2.5 * s)
    return s, hi, lo


def resample(r, keys_by_axis):
    """Resample episodes of one run. keys_by_axis: list of (keys, axis) groups sharing indices."""
    out = dict(r)
    for keys, axis in keys_by_axis:
        n = r[keys[0]].shape[axis]
        idx = RNG.integers(0, n, n)
        for k in keys:
            out[k] = np.take(r[k], idx, axis=axis)
    return out


def hboot(runs, stat, groups, nb=NB):
    pt = stat(runs)
    smp = np.empty(nb)
    for i in range(nb):
        pick = RNG.integers(0, len(runs), len(runs))
        smp[i] = stat([resample(runs[p], groups) for p in pick])
    return pt, smp


def ci(s):
    s = s[np.isfinite(s)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))] if len(s) else [np.nan, np.nan]


def cat(rs, k, axis=0):
    return np.concatenate([r[k] for r in rs], axis=axis)


G_GS = [(["gs_g", "gs_geff", "gs_gprobe", "gs_err_px", "gs_ctx_readout_err_px"], 0)]
G_ST = [(["steer_geff", "steer_cer", "steer_spec_px"], 1)]
INV_KEYS = ["inv_g"] + [f"inv_{m}_{q}" for m in ("nat", "z", "h") for q in ("err_fit", "err_hold", "geff_hold",
                                                                           "geff_fit")] + ["inv_true_geff_hold"]
G_INVK = [(INV_KEYS, 0)]
G_T = [(["test_g", "gpred_trained", "gpred_untrained"], 0)]


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def steer_curve(rs):
    grid = rs[0]["steer_grid"]
    ge = cat(rs, "steer_geff", 1)
    return np.repeat(grid[:, None], ge.shape[1], 1).ravel(), ge.ravel()


def holm(p):
    names = sorted(p, key=lambda k: p[k])
    m, run, adj = len(names), 0.0, {}
    for i, n in enumerate(names):
        run = max(run, min(1.0, (m - i) * p[n]))
        adj[n] = run
    return adj


# ------------------------------------------------------------------------------------ analysis
def analyse_config(rs, confirm, fam, pvals):
    o = {"n_seeds": len(rs)}
    for k in ("geff_slope", "geff_corr", "r2_g", "err8_px", "errR_px", "r2_g_trained", "r2_g_untrained",
              "r2_e_trained", "r2_e_untrained", "r2_mu_trained", "r2_mu_untrained", "r2_E_trained",
              "r2_E_untrained", "r2_px_trained", "r2_px_untrained", "test_err_px", "count45_err_px",
              "speed_err_px", "massratio_err_px", "mass_ctrl_err_px", "steer_w_dot_u"):
        o[k] = [r["summary"].get(k) for r in rs]
    o["gate_pass"] = bool(np.mean(o["geff_slope"]) >= 0.5)
    # E2 natural, E3a internal estimate
    for nm, key in (("nat", "gs_geff"), ("inf", "gs_gprobe")):
        for j, w in enumerate(("s", "hi", "lo")):
            pt, sm = hboot(rs, lambda rr, key=key, j=j: curve_gs(cat(rr, "gs_g"), cat(rr, key))[j], G_GS)
            o[f"{nm}_{w}"] = [pt, *ci(sm)]
            if confirm and nm == "nat" and w in ("hi", "lo") and o["gate_pass"]:
                pvals[f"H2_{fam}_{w}"] = float(np.mean(sm >= 0.5))
    # E3b steering
    for j, w in enumerate(("s", "hi", "lo")):
        pt, sm = hboot(rs, lambda rr, j=j: curve_gs(*steer_curve(rr))[j], G_ST)
        o[f"exe_{w}"] = [pt, *ci(sm)]
        if w == "s":
            s_exe = sm
    grid = rs[0]["steer_grid"]
    inr = (grid >= 4) & (grid <= 8) & (np.abs(grid - 6) >= 1)
    pt, cs = hboot(rs, lambda rr: float(np.nanmean(cat(rr, "steer_cer", 1)[inr])), G_ST)
    o["cer_in_range"] = [pt, *ci(cs)]
    o["spec_px_in_range"] = float(np.nanmean(cat(rs, "steer_spec_px", 1)[(grid >= 4) & (grid <= 8)]))
    if confirm:
        pvals[f"H3_{fam}"] = float(max(np.mean(s_exe < 0.5), np.mean(cs <= 0)))
    # E3c nested inversion
    for mode in ("nat", "z", "h"):
        for j, w in enumerate(("s", "hi", "lo")):
            pt, sm = hboot(rs, lambda rr, mode=mode, j=j: curve_inv(cat(rr, "inv_g"), cat(rr, f"inv_{mode}_geff_hold"))[j],
                           G_INVK)
            o[f"inv_{mode}_{w}"] = [pt, *ci(sm)]
        anti = lambda rr, mode=mode: float(np.nanmean(cat(rr, f"inv_{mode}_geff_hold")[cat(rr, "inv_g") < 0]))
        pt, sm = hboot(rs, anti, G_INVK)
        o[f"inv_{mode}_anti_geff"] = [pt, *ci(sm)]
        for q in ("err_hold", "err_fit"):
            o[f"inv_{mode}_{q}_by_g"] = {str(gv): float(np.nanmean(cat(rs, f"inv_{mode}_{q}")[cat(rs, "inv_g") == gv]))
                                         for gv in list(G_INV) + [-4.0]}
    for side in ("hi", "lo"):
        j = 1 if side == "hi" else 2
        pt, sm = hboot(rs, lambda rr, j=j: curve_inv(cat(rr, "inv_g"), cat(rr, "inv_z_geff_hold"))[j] -
                       curve_inv(cat(rr, "inv_g"), cat(rr, "inv_nat_geff_hold"))[j], G_INVK)
        o[f"inv_z_minus_nat_{side}"] = [pt, *ci(sm)]
        rh, rz = o[f"inv_h_{side}"], o[f"inv_z_{side}"]
        if rh[2] < 0.5:
            v = "execution-limited"
        elif rh[0] >= 0.5 and rz[0] >= 0.5 and ci(sm)[0] > 0:
            v = "inference-limited"
        elif rh[0] >= 0.5 and rz[0] < 0.5:
            v = "unreachable belief"
        else:
            v = "mixed"
        o[f"locus_{side}"] = v
    o["inv_h_reproduces_antiphysics"] = bool(o["inv_h_anti_geff"][0] < 0)
    # H5 perception check (latent families)
    if fam in ("rec", "jepa"):
        def excess(rr):
            g, e = cat(rr, "gs_g"), cat(rr, "gs_ctx_readout_err_px")
            ood = (g < 4) | (g > 8)
            return float(e[ood].mean() - e[~ood].mean())
        pt, sm = hboot(rs, excess, G_GS)
        o["perception_ood_excess_px"] = [pt, *ci(sm)]
        if confirm:
            pvals[f"H5_{fam}"] = float(np.mean(sm >= 0.5))
    o["ctx_readout_err_px"] = float(np.mean(cat(rs, "gs_ctx_readout_err_px")))
    # H1
    if confirm:
        pt, sm = hboot(rs, lambda rr: r2(cat(rr, "test_g"), cat(rr, "gpred_trained")), G_T)
        pt2, sm2 = hboot(rs, lambda rr: r2(cat(rr, "test_g"), cat(rr, "gpred_trained")) -
                         r2(cat(rr, "test_g"), cat(rr, "gpred_untrained")), G_T)
        o["H1_r2_g"], o["H1_r2_diff"] = [pt, *ci(sm)], [pt2, *ci(sm2)]
        pvals[f"H1_{fam}"] = float(max(np.mean(sm < 0.5), np.mean(sm2 <= 0)))
    # drift and velocity edits
    if "long_Erel" in rs[0]:
        o["drift_E_median_by_chunk"] = [float(np.nanmedian(cat(rs, "long_Erel")[:, i * 20:(i + 1) * 20]))
                                        for i in range(4)]
    if "vx_cer" in rs[0]:
        o["vx_cer"] = float(np.nanmean(cat(rs, "vx_cer")))
        dm, dt_ = cat(rs, "vx_dpx_model"), cat(rs, "vx_dpx_true")
        o["vx_dpx_ratio_by_t"] = [float(np.nanmean(dm[:, i]) / np.nanmean(dt_[:, i])) for i in range(0, dm.shape[1], 3)]
        de, det = cat(rs, "vx_dE_model"), cat(rs, "vx_dE_true")
        o["vx_dE_slope"] = float(slope(det[:, 2], de[:, 2]))
    return o


def h4(runs, pvals, out):
    for fam in FAMS:
        cells = [(n, runs.get((fam, "M", "gru", "train", n), [])) for n in (1000, 8000, 100000)]
        if any(len(rs) == 0 for _, rs in cells):
            continue
        sr, se = [], []
        for i in range(NB + 1):
            xs, rho, err = [], [], []
            for n, rs in cells:
                sel = rs if i == 0 else [resample(rs[j], G_GS) for j in RNG.integers(0, len(rs), len(rs))]
                xs.append(np.log10(n))
                rho.append(curve_gs(cat(sel, "gs_g"), cat(sel, "gs_geff"))[1])
                err.append(np.mean([r["summary"]["test_err_px"] for r in sel]))
            if i == 0:
                pr, pe = slope(np.array(xs), np.array(rho)), slope(np.array(xs), np.array(err))
            else:
                sr.append(slope(np.array(xs), np.array(rho)))
                se.append(slope(np.array(xs), np.array(err)))
        sr, se = np.array(sr), np.array(se)
        out[f"H4_{fam}"] = {"slope_rho_hi_per_decade": [pr, *ci(sr)], "slope_test_err_px_per_decade": [pe, *ci(se)]}
        pvals[f"H4_{fam}"] = float(max(np.mean(sr >= 0.15), np.mean(se >= 0)))


def main():
    runs = load_runs()
    out, pvals = {"configs": {}}, {}
    for key, rs in sorted(runs.items()):
        fam = key[0]
        tag = "_".join(str(x) for x in key)
        out["configs"][tag] = analyse_config(rs, key == main_key(fam), fam, pvals)
        o = out["configs"][tag]
        print(tag, f"seeds {o['n_seeds']} gate {o['gate_pass']} nat_hi {o['nat_hi'][0]:.2f} exe_hi {o['exe_hi'][0]:.2f} "
                   f"invh_hi {o['inv_h_hi'][0]:.2f} invz_hi {o['inv_z_hi'][0]:.2f} locus {o['locus_hi']}/{o['locus_lo']}",
              flush=True)
    h4(runs, pvals, out)
    out["pvals"], out["pvals_holm"] = pvals, holm(pvals) if pvals else {}
    (RES / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    write_md(out)
    figures(runs, out)


def write_md(out):
    L = ["# v2 pre-registered hypotheses: outcomes", "",
         "One-sided hierarchical-bootstrap p-values; Holm over all confirmatory tests.", "",
         "| Test | p (raw) | p (Holm) | Supported |", "|---|---|---|---|"]
    for k in sorted(out["pvals"]):
        L.append(f"| {k} | {out['pvals'][k]:.4f} | {out['pvals_holm'][k]:.4f} | "
                 f"{'yes' if out['pvals_holm'][k] < 0.05 else 'no'} |")
    L += ["", "## Locus of OOD failure (decision rule, Section 6)", "",
          "| Config | gate | high side | low side | inv_h reproduces anti-physics |", "|---|---|---|---|---|"]
    for tag, o in out["configs"].items():
        L.append(f"| {tag} | {o['gate_pass']} | {o['locus_hi']} | {o['locus_lo']} | {o['inv_h_reproduces_antiphysics']} |")
    (RES / "hypotheses.md").write_text("\n".join(L) + "\n")


def figures(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vibes.experiments2 import bayes_reference
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    grid = np.array(G_GRID)
    fams = [f for f in FAMS if main_key(f) in runs]
    if not fams:
        return
    gb, b1 = bayes_reference(1.0)
    _, b25 = bayes_reference(0.25)
    fig, ax = plt.subplots(len(fams), 4, figsize=(11, 2.5 * len(fams)), squeeze=False, sharex=True, sharey=True)
    for i, f in enumerate(fams):
        rs = runs[main_key(f)]
        for j in range(4):
            a = ax[i, j]
            a.axvspan(4, 8, color="#eeeeee", zorder=0)
            a.plot([0, 13], [0, 13], "k--", lw=0.7)
            a.plot(grid, np.clip(grid, 4, 8), "k:", lw=0.7)
            a.set_ylim(-1, 13)
            a.set_xlim(0.5, 12.5)
        c = np.stack([[gmean(r["gs_g"], r["gs_geff"], x) for x in grid] for r in rs])
        ax[i, 0].plot(grid, [gmean(gb, b1, x) for x in grid], color="#999999", lw=1, label="Bayes, 1 px")
        ax[i, 0].plot(grid, [gmean(gb, b25, x) for x in grid], color="#999999", lw=1, ls="-.", label="Bayes, 0.25 px")
        ax[i, 0].plot(grid, c.mean(0), "o-", ms=2.5, color=COLOR[f], label="model")
        c = np.stack([[gmean(r["gs_g"], r["gs_gprobe"], x) for x in grid] for r in rs])
        ax[i, 1].plot(grid, c.mean(0), "o-", ms=2.5, color=COLOR[f])
        c = np.stack([np.nanmean(r["steer_geff"], 1) for r in rs])
        ax[i, 2].plot(grid, c.mean(0), "o-", ms=2.5, color=COLOR[f])
        for mode, ls, lab in (("nat", "-", "natural"), ("z", "--", "invert context"), ("h", ":", "invert belief")):
            c = np.stack([[gmean(r["inv_g"], r[f"inv_{mode}_geff_hold"], x) for x in G_INV] for r in rs])
            ax[i, 3].plot(G_INV, c.mean(0), ls, marker="o", ms=2.5, color=COLOR[f], label=lab)
            anti = np.mean([np.nanmean(r[f"inv_{mode}_geff_hold"][r["inv_g"] < 0]) for r in rs])
            ax[i, 3].plot([0.7], [anti], marker="x", color=COLOR[f], ms=5)
        ax[i, 0].set_ylabel(f"{LABEL[f].split(' (')[0]}\ngravity")
        if i == 0:
            ax[i, 0].legend(fontsize=6, frameon=False, loc="upper left")
            ax[i, 3].legend(fontsize=6, frameon=False, loc="upper left")
    for j, t in enumerate(["(a) natural rollouts $g_{\\rm eff}$", "(b) internal estimate (probe on $h_C$)",
                           "(c) linear write: steer to $g^*$", "(d) nested inversion, held-out $g_{\\rm eff}$"]):
        ax[0, j].set_title(t, fontsize=8)
        ax[-1, j].set_xlabel("true $g$" if j < 2 else "target $g^*$")
    fig.tight_layout()
    fig.savefig(RES / "fig_locus.png", dpi=200)
    fig.savefig(ROOT / "paper" / "fig_locus.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
