"""Aggregate per-run results into the pre-registered statistics, tables and figures.

    python scripts/aggregate.py
Writes results/summary.json, results/hypotheses.md and results/*.png.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes.analysis import bayes_reference  # noqa: E402
from vibes.data import load  # noqa: E402

RES = ROOT / "results"
NB = int(__import__("os").environ.get("VIBES_NB", "2000"))
RNG = np.random.default_rng(12345)
FAM_LABEL = {"rec": "REC (pixel)", "jepa": "JEPA (latent)", "state": "STATE (true positions)"}
FAM_COLOR = {"rec": "#d1495b", "jepa": "#2e86ab", "state": "#5c5c5c"}


# ---------------------------------------------------------------- loading
def load_runs():
    runs = defaultdict(list)
    for j in sorted((RES / "runs").glob("*.json")):
        s = json.loads(j.read_text())
        d = dict(np.load(j.with_suffix(".npz")))
        d["summary"] = s
        runs[(s["family"], s["size"], s["n_train"])].append(d)
    return runs


# ---------------------------------------------------------------- statistics
def hboot(runs, stat, nb=NB, keys=None):
    """Hierarchical bootstrap: resample runs (seeds), then episodes within each run.

    `stat(list_of_resampled_runs)` -> scalar. Episode resampling is applied to every array
    whose first axis is the episode axis listed in `keys` (dict key -> axis length name).
    Returns (point estimate, samples)."""
    point = stat(runs)
    out = np.empty(nb)
    for i in range(nb):
        pick = RNG.integers(0, len(runs), len(runs))
        rs = []
        for p in pick:
            r = runs[p]
            rr = {}
            for grp, ks in keys.items():
                n = len(r[ks[0]]) if ks[0] in r else 0
                idx = RNG.integers(0, n, n)
                for k in ks:
                    a = r[k]
                    rr[k] = a[:, idx] if (k.startswith("steer_") and a.ndim == 2) else a[idx]
            for k, v in r.items():
                rr.setdefault(k, v)
            rs.append(rr)
        out[i] = stat(rs)
    return point, out


def ci(samples):
    s = samples[np.isfinite(samples)]
    return (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))) if len(s) else (np.nan, np.nan)


def group_mean(g, v, gv):
    m = (np.abs(g - gv) < 1e-6) & np.isfinite(v)
    return v[m].mean() if m.any() else np.nan


def ols_slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return np.polyfit(x[m], y[m], 1)[0]


def curve_stats(g, v):
    """s_ID, rho_hi, rho_lo for a response curve v(g) given per-episode g (pre-registered)."""
    inr = (g >= 4 - 1e-6) & (g <= 8 + 1e-6)
    s = ols_slope(g[inr], v[inr])
    gm = lambda a, b: np.nanmean([group_mean(g, v, a), group_mean(g, v, b)])
    hi = (gm(11.5, 12.0) - gm(7.5, 8.0)) / (4 * s)
    lo = (gm(4.0, 4.5) - gm(1.0, 1.5)) / (3 * s)
    return s, hi, lo


def pooled(rs, k):
    return np.concatenate([r[k] for r in rs])


def nat_stat(which):
    def f(rs):
        s, hi, lo = curve_stats(pooled(rs, "gs_g"), pooled(rs, "gs_geff"))
        return {"s": s, "hi": hi, "lo": lo}[which]
    return f


def inf_stat(which):
    def f(rs):
        s, hi, lo = curve_stats(pooled(rs, "gs_g"), pooled(rs, "gs_gprobe"))
        return {"s": s, "hi": hi, "lo": lo}[which]
    return f


def steer_curve(rs):
    grid = rs[0]["steer_grid"]
    ge = np.concatenate([r["steer_geff"] for r in rs], 1)        # (G, N)
    g = np.repeat(grid[:, None], ge.shape[1], 1)
    return g.ravel(), ge.ravel()


def exe_stat(which):
    def f(rs):
        g, v = steer_curve(rs)
        s, hi, lo = curve_stats(g, v)
        return {"s": s, "hi": hi, "lo": lo}[which]
    return f


def cer_stat(rs):
    grid = rs[0]["steer_grid"]
    c = np.concatenate([r["steer_cer"] for r in rs], 1)
    m = (grid >= 4) & (grid <= 8) & (np.abs(grid - 6) >= 1)   # targets that differ from baseline
    return float(np.nanmean(c[m]))


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


KEYS_GS = {"gs": ["gs_g", "gs_geff", "gs_gprobe", "gs_poserr"]}
KEYS_ST = {"st": ["steer_geff", "steer_cer", "steer_spec_px"]}
KEYS_T = {"t": ["test_g", "test_gpred", "test_gpred_untrained", "test_poserr"]}


def fix_keys(runs, keys):
    # steer arrays are (G, N): episode axis is 1; make hboot index the right axis via a proxy
    for r in runs:
        r.setdefault("_st_n", np.zeros(r["steer_geff"].shape[1]))
    return runs


def steer_boot(runs, stat):
    """Bootstrap for steering arrays (episode axis = 1)."""
    point = stat(runs)
    out = np.empty(NB)
    for i in range(NB):
        pick = RNG.integers(0, len(runs), len(runs))
        rs = []
        for p in pick:
            r = runs[p]
            n = r["steer_geff"].shape[1]
            idx = RNG.integers(0, n, n)
            rs.append({**r, "steer_geff": r["steer_geff"][:, idx], "steer_cer": r["steer_cer"][:, idx],
                       "steer_spec_px": r["steer_spec_px"][:, idx]})
        out[i] = stat(rs)
    return point, out


def paired_boot(runs, stat_a, keys_a, stat_b):
    """Joint bootstrap for (steering statistic) - (gsweep statistic) on the same resampled seeds."""
    pa = stat_a(runs) - stat_b(runs)
    out = np.empty(NB)
    for i in range(NB):
        pick = RNG.integers(0, len(runs), len(runs))
        rs = []
        for p in pick:
            r = runs[p]
            n = r["steer_geff"].shape[1]
            idx = RNG.integers(0, n, n)
            m = len(r["gs_g"])
            jdx = RNG.integers(0, m, m)
            rs.append({**r, "steer_geff": r["steer_geff"][:, idx], "gs_g": r["gs_g"][jdx],
                       "gs_geff": r["gs_geff"][jdx], "gs_gprobe": r["gs_gprobe"][jdx]})
        out[i] = stat_a(rs) - stat_b(rs)
    return pa, out


def holm(pvals):
    names = list(pvals)
    order = np.argsort([pvals[n] for n in names])
    m = len(names)
    adj, running = {}, 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m - rank) * pvals[names[i]])
        running = max(running, a)
        adj[names[i]] = running
    return adj


# ---------------------------------------------------------------- main
def main():
    runs = load_runs()
    out = {"configs": {}}
    pvals = {}
    for key, rs in sorted(runs.items()):
        fam, size, n = key
        tag = f"{fam}_{size}_{n // 1000}k"
        o = {"n_seeds": len(rs), "n_params": rs[0]["summary"].get("n_params")}
        for k in ("r2_g", "r2_g_untrained", "r2_e", "r2_e_untrained", "r2_mu", "r2_mu_untrained",
                  "r2_E", "r2_E_untrained", "r2_px", "r2_px_untrained", "test_poserr_px",
                  "count45_poserr_px", "speed_poserr_px", "massratio_poserr_px",
                  "mass_ctrl_poserr_px", "pair_probe_slope"):
            o[k] = [r["summary"].get(k) for r in rs]
        # natural response, internal estimate, execution
        for nm, fn, keys in (("nat", nat_stat, KEYS_GS), ("inf", inf_stat, KEYS_GS)):
            for w in ("s", "hi", "lo"):
                pt, sm = hboot(rs, fn(w), keys=keys)
                o[f"{nm}_{w}"] = [pt, *ci(sm)]
                if nm == "nat" and w in ("hi", "lo") and size == "M" and n == 16000 and fam != "state":
                    pvals[f"H2_{fam}_{w}"] = float(np.mean(sm >= 0.5))
        for w in ("s", "hi", "lo"):
            pt, sm = steer_boot(rs, exe_stat(w))
            o[f"exe_{w}"] = [pt, *ci(sm)]
            if w == "s":
                s_samp = sm
        pt, cs = steer_boot(rs, cer_stat)
        o["cer_in_range"] = [pt, *ci(cs)]
        o["spec_px_in_range"] = float(np.nanmean(np.concatenate(
            [r["steer_spec_px"][(r["steer_grid"] >= 4) & (r["steer_grid"] <= 8)].ravel() for r in rs])))
        for side in ("hi", "lo"):
            pt, sm = paired_boot(rs, exe_stat(side), None, nat_stat(side))
            o[f"exe_minus_nat_{side}"] = [pt, *ci(sm)]
            ex = o[f"exe_{side}"]
            if o["exe_s"][0] < 0.5 or o["cer_in_range"][0] <= 0:
                verdict = "not identifiable (steering invalid)"
            elif ci(sm)[0] > 0 and ex[0] >= 0.5:
                verdict = "inference-limited"
            elif ex[2] < 0.5:
                verdict = "execution-limited"
            else:
                verdict = "mixed/undetermined"
            o[f"locus_{side}"] = verdict
        # H1, H3
        if size == "M" and n == 16000 and fam != "state":
            def r2g(rs_):
                return r2(pooled(rs_, "test_g"), pooled(rs_, "test_gpred"))

            def r2diff(rs_):
                return r2g(rs_) - r2(pooled(rs_, "test_g"), pooled(rs_, "test_gpred_untrained"))
            pt, sm = hboot(rs, r2g, keys=KEYS_T)
            pt2, sm2 = hboot(rs, r2diff, keys=KEYS_T)
            o["H1_r2_g"], o["H1_r2_diff"] = [pt, *ci(sm)], [pt2, *ci(sm2)]
            pvals[f"H1_{fam}"] = float(max(np.mean(sm < 0.8), np.mean(sm2 <= 0)))
            pvals[f"H3_{fam}"] = float(max(np.mean(s_samp < 0.5), np.mean(cs <= 0)))
        out["configs"][tag] = o
        print(tag, json.dumps({k: v for k, v in o.items() if k.startswith(("nat_", "exe_", "inf_", "locus", "cer"))}),
              flush=True)

    # H4: scaling with data (M size; 1k, 4k, 16k)
    for fam in ("rec", "jepa"):
        cells = [(n, runs.get((fam, "M", n), [])) for n in (1000, 4000, 16000)]
        if any(len(rs) == 0 for _, rs in cells):
            continue
        slopes_err, slopes_rho = [], []
        for i in range(NB + 1):
            xs, ye, yr = [], [], []
            for n, rs in cells:
                if i == 0:
                    sel = rs
                else:
                    sel = [rs[j] for j in RNG.integers(0, len(rs), len(rs))]
                    sel = [{**r, **{k: r[k][idx] for k in ("gs_g", "gs_geff", "test_poserr")}}
                           for r in sel for idx in [RNG.integers(0, len(r["gs_g"]), len(r["gs_g"]))]]
                xs.append(np.log10(n))
                ye.append(np.mean([r["test_poserr"].mean() for r in sel]) * 64)
                yr.append(nat_stat("hi")(sel))
            if i == 0:
                pe, pr = ols_slope(np.array(xs), np.array(ye)), ols_slope(np.array(xs), np.array(yr))
            else:
                slopes_err.append(ols_slope(np.array(xs), np.array(ye)))
                slopes_rho.append(ols_slope(np.array(xs), np.array(yr)))
        se, sr = np.array(slopes_err), np.array(slopes_rho)
        out[f"H4_{fam}"] = {"slope_poserr_px_per_decade": [pe, *ci(se)],
                            "slope_rho_hi_per_decade": [pr, *ci(sr)]}
        pvals[f"H4_{fam}"] = float(max(np.mean(se >= 0), np.mean(sr >= 0.15)))

    adj = holm(pvals) if pvals else {}
    out["pvals"] = pvals
    out["pvals_holm"] = adj
    (RES / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    write_hypotheses(out)
    make_figures(runs, out)


def write_hypotheses(out):
    L = ["# Pre-registered hypotheses: outcomes", "",
         "One-sided hierarchical-bootstrap p-values, Holm-corrected over the 10 confirmatory tests.", "",
         "| Test | p (raw) | p (Holm) | Supported at 0.05 |", "|---|---|---|---|"]
    for k in sorted(out["pvals"]):
        L.append(f"| {k} | {out['pvals'][k]:.4f} | {out['pvals_holm'][k]:.4f} | "
                 f"{'yes' if out['pvals_holm'][k] < 0.05 else 'no'} |")
    L += ["", "## Locus of OOD failure (E3 decision rule)", "",
          "| Config | high side | low side |", "|---|---|---|"]
    for tag, o in out["configs"].items():
        L.append(f"| {tag} | {o['locus_hi']} | {o['locus_lo']} |")
    (RES / "hypotheses.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- figures
def make_figures(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    grid = np.round(np.arange(1.0, 12.01, 0.5), 2)
    gs = load(ROOT / "data" / "gsweep.npz")
    bay = {s: bayes_reference(gs, np.arange(len(gs["g"])), sigma=s / 64) for s in (1.0, 0.25)}

    def curve(g, v):
        return np.array([group_mean(g, v, x) for x in grid])

    fams = [f for f in ("rec", "jepa", "state") if (f, "M", 16000) in runs]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    titles = ["(a) natural rollouts: $g_{\\rm eff}$", "(b) internal estimate: probe($h_C$)",
              "(c) steered belief: $g_{\\rm eff}$ after setting probe to $g^*$"]
    for a, t in zip(ax, titles):
        a.axvspan(4, 8, color="#eeeeee", zorder=0)
        a.plot(grid, grid, "k--", lw=0.8, label="law ($y=x$)")
        a.plot(grid, np.clip(grid, 4, 8), "k:", lw=0.8, label="clamp to train range")
        a.set_title(t, fontsize=9)
        a.set_xlabel("$g$ true (a, b) or target $g^*$ (c)")
    for s, ls in ((1.0, "-"), (0.25, "-.")):
        ax[0].plot(grid, curve(gs["g"], bay[s]), color="#999999", ls=ls, lw=1,
                   label=f"Bayes-optimal, $\\sigma$={s:g}px")
    for f in fams:
        rs = runs[(f, "M", 16000)]
        c = np.stack([curve(r["gs_g"], r["gs_geff"]) for r in rs])
        ax[0].plot(grid, c.mean(0), "o-", ms=2.5, color=FAM_COLOR[f], label=FAM_LABEL[f])
        c = np.stack([curve(r["gs_g"], r["gs_gprobe"]) for r in rs])
        ax[1].plot(grid, c.mean(0), "o-", ms=2.5, color=FAM_COLOR[f])
        c = np.stack([np.nanmean(r["steer_geff"], 1) for r in rs])
        ax[2].plot(grid, c.mean(0), "o-", ms=2.5, color=FAM_COLOR[f])
    ax[0].set_ylabel("gravity")
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    for a in ax:
        a.set_ylim(0, 13)
    fig.tight_layout()
    fig.savefig(RES / "fig_locus.png", dpi=200)
    fig.savefig(ROOT / "paper" / "fig_locus.pdf")
    plt.close(fig)

    # decodability
    fig, a = plt.subplots(figsize=(5.2, 2.6))
    qs = ["g", "e", "mu", "E", "px"]
    wbar = 0.13
    for i, f in enumerate(fams):
        o = out["configs"][f"{f}_M_16k"]
        a.bar(np.arange(5) + (2 * i - 2.5) * wbar, [np.mean(o[f"r2_{q}"]) for q in qs], wbar,
              color=FAM_COLOR[f], label=FAM_LABEL[f] + " trained")
        a.bar(np.arange(5) + (2 * i - 1.5) * wbar, [np.mean(o[f"r2_{q}_untrained"]) for q in qs], wbar,
              color=FAM_COLOR[f], alpha=0.35, label="untrained")
    a.set_xticks(range(5), ["$g$", "$e$", "$\\mu$", "$E$", "$p_x$"])
    a.set_ylabel("linear probe $R^2$ on $h_C$")
    a.set_ylim(-0.1, 1.05)
    a.legend(fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.3))
    fig.tight_layout()
    fig.savefig(RES / "fig_probes.png", dpi=200)
    fig.savefig(ROOT / "paper" / "fig_probes.pdf")
    plt.close(fig)

    # drift
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    for f in fams:
        rs = runs[(f, "M", 16000)]
        d = np.stack([np.nanmedian(r["long_Edrift"], 0) for r in rs]).mean(0)
        ax[0].plot(np.arange(len(d)) + 1, d, color=FAM_COLOR[f], label=FAM_LABEL[f])
        p = np.stack([r["long_poserr_t"] for r in rs]).mean(0) * 64
        ax[1].plot(np.arange(len(p)) + 1, p, color=FAM_COLOR[f])
    ax[0].set_xlabel("rollout step")
    ax[0].set_ylabel("median $|\\hat E-E|/E$")
    ax[1].set_xlabel("rollout step")
    ax[1].set_ylabel("position error (px)")
    ax[0].axvline(24, color="k", lw=0.5, ls=":")
    ax[1].axvline(24, color="k", lw=0.5, ls=":")
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "fig_drift.png", dpi=200)
    fig.savefig(ROOT / "paper" / "fig_drift.pdf")
    plt.close(fig)

    # scaling
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.7))
    for f in ("rec", "jepa"):
        for xk, xs_key, axes_i in (("data", [("M", 1000), ("M", 4000), ("M", 16000)], 0),
                                   ("size", [("S", 16000), ("M", 16000), ("L", 16000)], 1)):
            xs, err, rho, rexe = [], [], [], []
            for sz, n in xs_key:
                tag = f"{f}_{sz}_{n // 1000}k"
                if tag not in out["configs"]:
                    continue
                o = out["configs"][tag]
                xs.append(n if xk == "data" else o["n_params"])
                err.append(np.mean(o["test_poserr_px"]))
                rho.append(o["nat_hi"])
                rexe.append(o["exe_hi"])
            if not xs:
                continue
            a = ax[axes_i]
            rho = np.array(rho)
            a.errorbar(xs, rho[:, 0], yerr=[rho[:, 0] - rho[:, 1], rho[:, 2] - rho[:, 0]], fmt="o-",
                       color=FAM_COLOR[f], capsize=2, label=f"{FAM_LABEL[f]} natural $\\rho_{{hi}}$")
            rexe = np.array(rexe)
            a.errorbar(xs, rexe[:, 0], yerr=[rexe[:, 0] - rexe[:, 1], rexe[:, 2] - rexe[:, 0]], fmt="s--",
                       color=FAM_COLOR[f], capsize=2, alpha=0.6, label=f"{FAM_LABEL[f]} steered $\\rho_{{hi}}$")
            a.set_xscale("log")
            a.set_xlabel("training episodes" if xk == "data" else "parameters")
            if xk == "data":
                ax[2].plot(xs, err, "o-", color=FAM_COLOR[f], label=FAM_LABEL[f])
    ax[0].set_ylabel("extrapolation $\\rho_{hi}$ (1 = law, 0 = clamp)")
    ax[0].axhline(0.5, color="k", lw=0.5, ls=":")
    ax[1].axhline(0.5, color="k", lw=0.5, ls=":")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("training episodes")
    ax[2].set_ylabel("ID rollout position error (px)")
    ax[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "fig_scaling.png", dpi=200)
    fig.savefig(ROOT / "paper" / "fig_scaling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
