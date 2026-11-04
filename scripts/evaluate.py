"""Evaluate one trained run on every pre-registered experiment; writes results/runs/<name>.npz.

    python scripts/evaluate.py runs/rec_M_16k_s0
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes import sim  # noqa: E402
from vibes.analysis import (C, DT, Episodes, encode_context, forward_from, g_eff, load_run,  # noqa: E402
                            matched_pairs, pos_error, probe_r2, rollout_from_h, untrained_like)
from vibes.data import load  # noqa: E402
from vibes.readout import unwrap_x  # noqa: E402

import os
QUICK = int(os.environ.get("VIBES_QUICK", "0"))
G_GRID = np.round(np.arange(1.0, 12.01, 0.5), 2)


def energy_from_pos(pos, ep):
    """Energy and p_x per frame from readout positions via central differences.
    pos (N, T, K, 2) -> E (N, T-2), px (N, T-2) for frames 1..T-2."""
    x = unwrap_x(np.moveaxis(pos[..., 0], 1, -1))          # (N, K, T)
    y = np.moveaxis(pos[..., 1], 1, -1)
    vx = (x[..., 2:] - x[..., :-2]) / (2 * DT)
    vy = (y[..., 2:] - y[..., :-2]) / (2 * DT)
    m = (ep.mass * ep.active)[..., None]
    E = (0.5 * m * (vx ** 2 + vy ** 2) + m * ep.g[:, None, None] * y[..., 1:-1]).sum(1)
    px = (m * vx).sum(1)
    return E, px


def _vx_pairs(p, idx, dv):
    """Matched-state contexts: same frame-(C-1) state except object 0's vx shifted by dv."""
    sC = p["states"][idx, C - 1].astype(np.float64)
    s2 = sC.copy()
    s2[:, 0, 2] += dv
    args = (p["kind"][idx], p["size"][idx], p["mass"][idx], p["n"][idx])
    from vibes.analysis import reversed_context
    c1 = reversed_context(sC, *args, p["g"][idx]).astype(np.float32)
    c2 = reversed_context(s2, *args, p["g"][idx]).astype(np.float32)
    mk = lambda st: Episodes(st, p["kind"][idx], p["size"][idx], p["color"][idx], p["active"][idx],
                             p["mass"][idx], p["g"][idx], np.ones(len(idx)), np.zeros(len(idx)))
    return mk(c1), mk(c2), s2


def velocity_edit(model, S, name, rng, R=32, dv_test=0.5):
    out = {}
    v, t = S["val"], S["test"]
    cons_v = np.where((v["e"] == 1) & (v["mu"] == 0))[0]
    idx = rng.choice(cons_v, 600, replace=True)
    dv = rng.choice([-1, 1], 600) * rng.uniform(0.3, 0.8, 600)
    e1, e2, _ = _vx_pairs(v, idx, dv)
    h1 = encode_context(model, e1).cpu().numpy()
    h2 = encode_context(model, e2).cpu().numpy()
    u = ((h2 - h1) / dv[:, None]).mean(0)
    cons_t = np.where((t["e"] == 1) & (t["mu"] == 0))[0][:300]
    ep = Episodes.from_split(t, cons_t)
    h = encode_context(model, ep)
    base = rollout_from_h(model, ep, h, R)
    U = torch.as_tensor(u, device="cuda", dtype=torch.float32)
    edit = rollout_from_h(model, ep, h + dv_test * U, R)
    s2 = t["states"][cons_t, C - 1].astype(np.float64)
    s2[:, 0, 2] += dv_test
    n1 = len(cons_t)
    cf = forward_from(s2, t["kind"][cons_t], t["size"][cons_t], t["mass"][cons_t], t["n"][cons_t],
                      t["g"][cons_t], np.ones(n1), np.zeros(n1), R + 1)[:, 1:]
    out["vx_cer"] = 1 - pos_error(edit, cf, ep.active) / np.maximum(pos_error(base, cf, ep.active), 1e-6)
    Eb, pb = energy_from_pos(base, ep)
    Ee, pe = energy_from_pos(edit, ep)
    m0 = t["mass"][cons_t, 0]
    vx0 = t["states"][cons_t, C - 1, 0, 2]
    out["vx_dpx_model"] = pe - pb                       # (N, R-2) frames C+1..C+R-2
    out["vx_dpx_true"] = m0 * dv_test
    dE_true = 0.5 * m0 * ((vx0 + dv_test) ** 2 - vx0 ** 2)
    out["vx_dE_model"] = Ee - Eb
    out["vx_dE_true"] = dE_true
    return out


def main(run_dir):
    run_dir = ROOT / run_dir if not Path(run_dir).is_absolute() else Path(run_dir)
    name = run_dir.name
    out = ROOT / "results" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, cfg = load_run(run_dir)
    rnd = untrained_like(cfg)
    S = {k: load(ROOT / "data" / f"{k}.npz") for k in
         ("val", "test", "gsweep", "long", "count45", "speed", "massratio", "mass_ctrl")}
    if QUICK:  # smoke test only
        S = {k: {kk: vv[::QUICK] for kk, vv in v.items()} for k, v in S.items()}
    E = {k: Episodes.from_split(v) for k, v in S.items()}
    res, summ = {}, {"name": name, **{k: cfg[k] for k in ("family", "size", "seed", "n_train")},
                     "n_params": cfg.get("n_params")}

    # ---------------- E1: probes on h_C ----------------
    hv = encode_context(model, E["val"]).cpu().numpy()
    ht = encode_context(model, E["test"]).cpu().numpy()
    hv0 = encode_context(rnd, E["val"]).cpu().numpy()
    ht0 = encode_context(rnd, E["test"]).cpu().numpy()
    targets = {"g": "g", "e": "e", "mu": "mu"}
    for tname, key in targets.items():
        r2, _ = probe_r2(hv, S["val"][key], ht, S["test"][key])
        r20, _ = probe_r2(hv0, S["val"][key], ht0, S["test"][key])
        summ[f"r2_{tname}"], summ[f"r2_{tname}_untrained"] = r2, r20
    for tname, key in (("E", "energy"), ("px", "px")):
        r2, _ = probe_r2(hv, S["val"][key][:, C - 1], ht, S["test"][key][:, C - 1])
        r20, _ = probe_r2(hv0, S["val"][key][:, C - 1], ht0, S["test"][key][:, C - 1])
        summ[f"r2_{tname}"], summ[f"r2_{tname}_untrained"] = r2, r20
    _, gprobe = probe_r2(hv, S["val"]["g"], ht, S["test"]["g"])
    _, gprobe0 = probe_r2(hv0, S["val"]["g"], ht0, S["test"]["g"])
    res["test_gpred"], res["test_gpred_untrained"] = gprobe.predict(ht), gprobe0.predict(ht0)
    w, b = gprobe.coef_, gprobe.intercept_
    print(f"[{name}] probes done {time.time() - t0:.0f}s  r2_g={summ['r2_g']:.3f}", flush=True)

    # ID rollout quality on test
    R = 32
    pos_t = rollout_from_h(model, E["test"], torch.as_tensor(ht, device="cuda"), R)
    res["test_poserr"] = pos_error(pos_t, S["test"]["states"][:, C:C + R], S["test"]["active"])
    res["test_poserr_t"] = np.array([pos_error(pos_t[:, i:i + 1], S["test"]["states"][:, C + i:C + i + 1],
                                               S["test"]["active"]).mean() for i in range(R)])
    res["test_geff"] = g_eff(pos_t, E["test"])
    res["test_g"] = S["test"]["g"]
    summ["test_poserr_px"] = float(res["test_poserr"].mean() * 64)

    # dissipation check (test episodes with e < 1): model vs true E(end)/E(start)
    Eh, _ = energy_from_pos(pos_t, E["test"])
    Etrue = S["test"]["energy"][:, C + 1:C + R - 1]
    res["test_Eratio_model"] = Eh[:, -4:].mean(1) / Eh[:, :4].mean(1)
    res["test_Eratio_true"] = Etrue[:, -4:].mean(1) / Etrue[:, :4].mean(1)
    res["test_e"], res["test_mu"] = S["test"]["e"], S["test"]["mu"]

    # ---------------- E2 / E3a: gsweep natural rollouts and internal estimate ----------------
    hg = encode_context(model, E["gsweep"])
    res["gs_g"] = S["gsweep"]["g"]
    res["gs_gprobe"] = hg.cpu().numpy() @ w + b
    pos_g = rollout_from_h(model, E["gsweep"], hg, R)
    res["gs_geff"] = g_eff(pos_g, E["gsweep"])
    res["gs_poserr"] = pos_error(pos_g, S["gsweep"]["states"][:, C:C + R], S["gsweep"]["active"])
    print(f"[{name}] gsweep done {time.time() - t0:.0f}s", flush=True)

    # ---------------- E3b: steering along the matched-pair encoding direction ----------------
    rng = np.random.default_rng(0)
    e1, e2, _ = matched_pairs(S["val"], 600, rng)
    h1 = encode_context(model, e1).cpu().numpy()
    h2 = encode_context(model, e2).cpu().numpy()
    u = ((h2 - h1) / (e2.g - e1.g)[:, None]).mean(0)
    res["u_g"] = u
    res["pair_probe_slope"] = np.polyfit(e2.g - e1.g, (h2 - h1) @ w, 1)[0]
    summ["pair_probe_slope"] = float(res["pair_probe_slope"])
    summ["w_dot_u"] = float(w @ u)
    sel = np.where((S["test"]["g"] >= 5) & (S["test"]["g"] <= 7))[0][:200]
    ep = E["test"].sub(sel)
    hs = torch.as_tensor(ht[sel], device="cuda")
    base_pos = pos_t[sel]
    gp0 = ht[sel] @ w + b
    U = torch.as_tensor(u, device="cuda", dtype=torch.float32)
    st_geff = np.full((len(G_GRID), len(sel)), np.nan)
    cer = np.full((len(G_GRID), len(sel)), np.nan)
    spec = np.full((len(G_GRID), len(sel)), np.nan)
    sC = S["test"]["states"][sel, C - 1].astype(np.float64)
    st_args = (S["test"]["kind"][sel], S["test"]["size"][sel], S["test"]["mass"][sel], S["test"]["n"][sel])
    for gi, gs in enumerate(G_GRID):
        delta = (gs - gp0) / (w @ u)
        h_new = hs + torch.as_tensor(delta, device="cuda", dtype=torch.float32)[:, None] * U
        pos_s = rollout_from_h(model, ep, h_new, R)
        st_geff[gi] = g_eff(pos_s, ep)
        dx = pos_s[:, 0] - base_pos[:, 0]
        dx[..., 0] -= np.round(dx[..., 0])
        dd = np.hypot(dx[..., 0], dx[..., 1])
        spec[gi] = (dd * ep.active).sum(1) / ep.active.sum(1) * 64
        cf = forward_from(sC, *st_args, np.full(len(sel), gs), ep.e, ep.mu, R + 1)[:, 1:]
        err_s = pos_error(pos_s, cf, ep.active)
        err_0 = pos_error(base_pos, cf, ep.active)
        cer[gi] = 1 - err_s / np.maximum(err_0, 1e-6)
    res.update(steer_grid=G_GRID, steer_geff=st_geff, steer_cer=cer, steer_spec_px=spec,
               steer_g0=S["test"]["g"][sel])
    print(f"[{name}] steering done {time.time() - t0:.0f}s", flush=True)

    # ---------------- E4 (exploratory): velocity edits and conservation bookkeeping ----------
    res.update(velocity_edit(model, S, name, rng))
    print(f"[{name}] velocity edits done {time.time() - t0:.0f}s", flush=True)

    # ---------------- E5: long-horizon conservation drift ----------------
    hl = encode_context(model, E["long"])
    RL = 64
    pos_l = rollout_from_h(model, E["long"], hl, RL)
    El, pxl = energy_from_pos(pos_l, E["long"])
    Et = S["long"]["energy"][:, C + 1:C + RL - 1]
    pxt = S["long"]["px"][:, C + 1:C + RL - 1]
    res["long_Edrift"] = np.abs(El - Et) / Et                  # (N, RL-2), frames C+1..C+RL-2
    res["long_pxerr"] = np.abs(pxl - pxt) / (S["long"]["mass"] * S["long"]["active"]).sum(1)[:, None]
    res["long_poserr_t"] = np.array([pos_error(pos_l[:, i:i + 1], S["long"]["states"][:, C + i:C + i + 1],
                                               S["long"]["active"]).mean() for i in range(RL)])

    # ---------------- E6: other OOD ----------------
    for k in ("count45", "speed", "massratio", "mass_ctrl"):
        hk = encode_context(model, E[k])
        pk = rollout_from_h(model, E[k], hk, R)
        res[f"{k}_poserr"] = pos_error(pk, S[k]["states"][:, C:C + R], S[k]["active"])
        summ[f"{k}_poserr_px"] = float(res[f"{k}_poserr"].mean() * 64)

    np.savez_compressed(out / f"{name}.npz", **res)
    summ["eval_seconds"] = time.time() - t0
    (out / f"{name}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r)
