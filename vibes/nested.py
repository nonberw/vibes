"""E8: nested-inversion evaluation on gsweep (natural / inv_z / inv_h) with held-out metrics."""
import numpy as np
import torch

from .analysis import C, Episodes, forward_from, g_eff, model_inputs, pos_error, readout_positions
from .inversion import invert

G_INV = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0)
ANTI_G = -4.0   # anti-physics control: upward gravity, never seen in training


def select_episodes(p, per_g=24, rng=None):
    """Pick per_g gsweep episodes for each value in G_INV, plus anti-physics controls built from
    in-range episodes (g in [5, 7]) by re-simulating their future with g = ANTI_G."""
    rng = rng or np.random.default_rng(0)
    idx, futs, gs = [], [], []
    for gv in G_INV:
        cand = np.where(np.abs(p["g"] - gv) < 1e-6)[0]
        pick = rng.choice(cand, size=min(per_g, len(cand)), replace=False)
        idx.append(pick)
        futs.append(p["states"][pick, C:C + 32])
        gs.append(np.full(len(pick), gv))
    cand = np.where((p["g"] >= 5) & (p["g"] <= 7))[0]
    pick = rng.choice(cand, size=per_g, replace=False)
    sC = p["states"][pick, C - 1].astype(np.float64)
    fut = forward_from(sC, p["kind"][pick], p["size"][pick], p["mass"][pick], p["n"][pick],
                       np.full(per_g, ANTI_G), p["e"][pick], p["mu"][pick], 33)[:, 1:].astype(np.float32)
    idx.append(pick)
    futs.append(fut)
    gs.append(np.full(per_g, ANTI_G))
    return np.concatenate(idx), np.concatenate(futs), np.concatenate(gs)


def nested_inversion(model, p, R=32, F=16, iters=300, chunk=48, per_g=24, device="cuda"):
    idx, fut, gtar = select_episodes(p, per_g)
    ep = Episodes.from_split(p, idx)
    ep_fut = Episodes(fut, ep.kind, ep.size, ep.color, ep.active, ep.mass, gtar, ep.e, ep.mu)
    out = {k: [] for k in ("err_fit", "err_hold", "geff_hold", "geff_full")}
    res = {}
    for mode in ("nat", "z", "h"):
        ef, eh, gh, gf = [], [], [], []
        for i in range(0, len(idx), chunk):
            sl = slice(i, i + chunk)
            e = ep.sub(sl)
            ctx = model_inputs(model, e, 0, C, device)
            kind = torch.as_tensor(e.kind, device=device)
            size = torch.as_tensor(e.size, dtype=torch.float32, device=device)
            color = torch.as_tensor(e.color, device=device)
            act = torch.as_tensor(e.active, device=device)
            fs = torch.as_tensor(fut[sl], device=device)
            if mode == "nat":
                with torch.no_grad():
                    _, h = model.observe(model.encode(ctx))
            else:
                with torch.enable_grad():
                    h = invert(model, mode, ctx, fs, kind, size, color, act, R, F=F, iters=iters)
            with torch.no_grad():
                zr, _ = model.imagine(h, R)
                pos = readout_positions(model, e, zr.float(), device).cpu().numpy()
            ef.append(pos_error(pos[:, :F], fut[sl][:, :F], e.active))
            eh.append(pos_error(pos[:, F:], fut[sl][:, F:], e.active))
            gh.append(g_eff(pos[:, F:], e))
            gf.append(g_eff(pos, e))
        res[f"inv_{mode}_err_fit"] = np.concatenate(ef)
        res[f"inv_{mode}_err_hold"] = np.concatenate(eh)
        res[f"inv_{mode}_geff_hold"] = np.concatenate(gh)
        res[f"inv_{mode}_geff_full"] = np.concatenate(gf)
    # ground-truth g_eff of the held-out continuation (estimator ceiling)
    res["inv_true_geff_hold"] = g_eff(fut[:, F:, :, :2], ep_fut)
    res["inv_gtarget"] = gtar
    return res
