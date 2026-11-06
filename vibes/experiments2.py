"""v2 experiments (PREREGISTRATION_v2.md): E1-E6 for one trained chunk world model."""
import time

import numpy as np
import torch
from scipy.stats import truncnorm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from . import sim, world
from .analysis2 import Meta, g_eff_pred, pos_err_px
from .dynamics import ChunkWM
from .world import C, DT, G_GRID, R

G_INV = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0)
ANTI_G = -4.0
F_FIT = 4             # inversion fit window: first F predicted frames
HOLD = (4, 14)        # held-out window for inversion metrics (runs must start in its first 5 frames)


# ---------------------------------------------------------------------------------------------
# simulation helpers

def sim_from(state, meta, g, e=None, mu=None, T=R + 1):
    """Simulate T frames starting at `state` (B, K, 4) (frame 0 = state)."""
    e = meta.e if e is None else e
    mu = meta.mu if mu is None else mu
    return sim.simulate_batch(np.ascontiguousarray(state, dtype=np.float64), meta.kind, meta.size, meta.mass,
                              meta.n, np.asarray(g, np.float64), np.asarray(e, np.float64),
                              np.asarray(mu, np.float64), T, DT)


def reversed_context(state, meta, g):
    """C context frames ending exactly in `state` under gravity g (conservative dynamics only)."""
    x0 = np.array(state, np.float64)
    x0[..., 2:] *= -1
    one = np.ones(meta.B)
    back = sim.simulate_batch(x0, meta.kind, meta.size, meta.mass, meta.n, np.asarray(g, np.float64),
                              one, 0 * one, C, DT)
    ctx = back[:, ::-1].copy()
    ctx[..., 2:] *= -1
    return ctx


def conservative(p):
    return np.where((p["e"] == 1.0) & (p["mu"] == 0.0))[0]


def law_fraction(pred, law, clamp, active):
    """Exploratory 'law fraction' (added after pre-registration, see Deviations).

    Projection of the prediction onto the difference between two simulator counterfactuals from the
    same state: the law (target g) and the clamp (g clipped to the training range [4, 8]).
    lambda = <pred - clamp, law - clamp> / ||law - clamp||^2 over frames and active objects
    (periodic x by minimum image). 1 = follows the law, 0 = follows the clamp. NaN when law == clamp.
    """
    def d(a, b):
        v = a[..., :2] - b[..., :2]
        v[..., 0] -= np.round(v[..., 0])
        return v
    a = active[:, None, :, None].astype(float)
    num = (d(pred, clamp) * d(law, clamp) * a).sum((1, 2, 3))
    den = (d(law, clamp) ** 2 * a).sum((1, 2, 3))
    out = np.full(len(num), np.nan)
    ok = den > (0.5 / 64) ** 2
    out[ok] = num[ok] / den[ok]
    return out


# ---------------------------------------------------------------------------------------------
# E1 probes

def e1_probes(wm, untrained):
    v, t = world.load("val"), world.load("test")
    mv, mt = Meta(v), Meta(t)
    res = {}
    H = {}
    for name, model in (("trained", wm), ("untrained", untrained)):
        H[name] = (model.run(v["states"][:, :C, :, :2], mv)["h"].cpu().numpy(),
                   model.run(t["states"][:, :C, :, :2], mt)["h"].cpu().numpy())
    targets = {"g": (v["g"], t["g"]), "e": (v["e"], t["e"]), "mu": (v["mu"], t["mu"]),
               "E": (v["energy"][:, C - 1], t["energy"][:, C - 1]), "px": (v["px"][:, C - 1], t["px"][:, C - 1])}
    probes = {}
    for q, (yv, yt) in targets.items():
        for name in H:
            r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(H[name][0], yv)
            res[f"r2_{q}_{name}"] = float(r2_score(yt, r.predict(H[name][1])))
            if q == "g":
                res[f"gpred_{name}"] = r.predict(H[name][1])
                if name == "trained":
                    probes["g"] = r
    res["test_g"] = t["g"]
    return res, probes, H["trained"]


# ---------------------------------------------------------------------------------------------
# E2 natural response + E3a internal estimate

def e2_natural(wm, gprobe):
    p = world.load("gsweep")
    meta = Meta(p)
    out = wm.run(p["states"][:, :C, :, :2], meta)
    pos = out["pos"].cpu().numpy()
    fut = p["states"][:, C:C + R]
    # E3(d) perception check: positions read out of the *context* latents of real frames
    with torch.no_grad():
        ctx_pos = torch.cat([wm.positions(out["zn"][i:i + 256], meta.sub(slice(i, i + 256)))
                             for i in range(0, meta.B, 256)]).cpu().numpy()
    clamp = sim_from(p["states"][:, C - 1], meta, np.clip(p["g"], 4.0, 8.0))[:, 1:]
    lam = law_fraction(pos[:, :10], fut[:, :10], clamp[:, :10], meta.active)
    return dict(gs_g=p["g"], gs_geff=g_eff_pred(pos, fut, p["g"], meta.active), gs_lambda=lam,
                gs_gprobe=gprobe.predict(out["h"].cpu().numpy()),
                gs_err_px=pos_err_px(pos, fut, meta.active),
                gs_ctx_readout_err_px=pos_err_px(ctx_pos, p["states"][:, :C], meta.active))


def bayes_reference(sigma_px=1.0, a=4.0, b=8.0, n_mc=200, seed=0):
    """Expected posterior mean of g under U[a, b] given the gsweep contexts with Gaussian position
    noise sigma (Prop. 2). Information comes from each object's longest contact-free run of
    context frames (quadratic fit; Fisher information summed over objects)."""
    p = world.load("gsweep")
    rng = np.random.default_rng(seed)
    sigma = sigma_px / 64

    def sd_g(n):
        t = np.arange(n) * DT
        A = np.stack([np.ones(n), t, t ** 2], 1)
        return 2 * sigma * np.sqrt((np.linalg.pinv(A)[2] ** 2).sum())

    out = np.zeros(len(p["g"]))
    for j in range(len(p["g"])):
        info = 0.0
        for k in range(p["n"][j]):
            y = p["states"][j, :C, k, 1].astype(np.float64)
            ok = np.abs(np.diff(y, 2) + p["g"][j] * DT ** 2) < 2e-5
            best = run = 0
            for o in ok:
                run = run + 1 if o else 0
                best = max(best, run)
            if best >= 2:
                info += 1 / sd_g(best + 2) ** 2
        if info == 0:
            out[j] = (a + b) / 2
            continue
        sd = 1 / np.sqrt(info)
        gh = p["g"][j] + sd * rng.standard_normal(n_mc)
        out[j] = truncnorm.mean((a - gh) / sd, (b - gh) / sd, loc=gh, scale=sd).mean()
    return p["g"], out


# ---------------------------------------------------------------------------------------------
# E3b steering along the matched-pair encoding direction

def matched_direction(wm, n_pairs=800, seed=0):
    v = world.load("val")
    rng = np.random.default_rng(seed)
    cons = conservative(v)
    idx = rng.choice(cons, n_pairs, replace=True)
    meta = Meta(v, idx)
    g1 = rng.uniform(4, 8, n_pairs)
    g2 = rng.uniform(4, 8, n_pairs)
    while (bad := np.abs(g2 - g1) < 1).any():
        g2[bad] = rng.uniform(4, 8, bad.sum())
    sC = v["states"][idx, C - 1]
    c1 = reversed_context(sC, meta, g1)
    c2 = reversed_context(sC, meta, g2)
    h1 = wm.run(c1[..., :2], meta)["h"].cpu().numpy()
    h2 = wm.run(c2[..., :2], meta)["h"].cpu().numpy()
    return ((h2 - h1) / (g2 - g1)[:, None]).mean(0), (g1, g2, h1, h2)


def e3_steering(wm, gprobe, u, n_ep=200):
    t = world.load("test")
    sel = np.where((t["g"] >= 5) & (t["g"] <= 7))[0][:n_ep]
    meta = Meta(t, sel)
    out = wm.run(t["states"][sel, :C, :, :2], meta)
    h0, zlast = out["h"], out["zn"][:, -1]
    base = out["pos"].cpu().numpy()
    w, b = gprobe.coef_, gprobe.intercept_
    gp0 = out["h"].cpu().numpy() @ w + b
    U = torch.as_tensor(u, dtype=torch.float32, device="cuda")
    sC = t["states"][sel, C - 1]
    geff = np.full((len(G_GRID), len(sel)), np.nan)
    cer = np.full_like(geff, np.nan)
    spec = np.full_like(geff, np.nan)
    lam = np.full_like(geff, np.nan)
    for gi, gs in enumerate(G_GRID):
        delta = torch.as_tensor((gs - gp0) / (w @ u), dtype=torch.float32, device="cuda")
        pos = wm.from_belief(h0 + delta[:, None] * U, zlast, meta).cpu().numpy()
        cf = sim_from(sC, meta, np.full(len(sel), gs))[:, 1:]
        geff[gi] = g_eff_pred(pos, cf, np.full(len(sel), gs), meta.active)
        cer[gi] = 1 - pos_err_px(pos, cf, meta.active) / np.maximum(pos_err_px(base, cf, meta.active), 1e-6)
        spec[gi] = pos_err_px(pos[:, :1], base[:, :1], meta.active)
        if gs < 4 or gs > 8:
            cl = sim_from(sC, meta, np.full(len(sel), float(np.clip(gs, 4, 8))))[:, 1:]
            lam[gi] = law_fraction(pos[:, :10], cf[:, :10], cl[:, :10], meta.active)
    return dict(steer_grid=np.array(G_GRID), steer_geff=geff, steer_cer=cer, steer_spec_px=spec, steer_lambda=lam,
                steer_w_dot_u=float(w @ u))


# ---------------------------------------------------------------------------------------------
# E3c nested inversion

def _select_inversion(p, per_g, rng):
    idx, gt = [], []
    for gv in G_INV:
        cand = np.where(np.abs(p["g"] - gv) < 1e-6)[0]
        idx.append(rng.choice(cand, per_g, replace=False))
        gt.append(np.full(per_g, gv))
    return np.concatenate(idx), np.concatenate(gt)


def e3_inversion(wm, per_g=80, iters=400, lr=0.05, seed=0):
    p = world.load("gsweep")
    rng = np.random.default_rng(seed)
    idx, gt = _select_inversion(p, per_g, rng)
    # anti-physics control: in-range contexts whose future is re-simulated with upward gravity
    cand = np.where((p["g"] >= 5) & (p["g"] <= 7))[0]
    anti = rng.choice(cand, per_g, replace=False)
    all_idx = np.concatenate([idx, anti])
    meta = Meta(p, all_idx)
    fut = p["states"][all_idx, C:C + R].copy()
    ma = Meta(p, anti)
    fut[len(idx):] = sim_from(p["states"][anti, C - 1], ma, np.full(per_g, ANTI_G))[:, 1:]
    g_target = np.concatenate([gt, np.full(per_g, ANTI_G)])
    ctx = p["states"][all_idx, :C, :, :2]
    res = {"inv_g": g_target}
    clamp = sim_from(p["states"][all_idx, C - 1], meta, np.clip(g_target, 4.0, 8.0))[:, 1:]
    with torch.no_grad():
        nat = wm.run(ctx, meta)
        zf = torch.cat([wm.dyn.norm(wm.latents(torch.as_tensor(fut[i:i + 128, :, :, :2], device="cuda"),
                                                meta.sub(slice(i, i + 128))))
                        for i in range(0, meta.B, 128)])
    ks_fit = torch.arange(F_FIT, device="cuda")
    is_gru = wm.dyn.core == "gru"
    for mode in ("nat", "z", "h"):
        if mode == "nat":
            pos = nat["pos"].cpu().numpy()
        else:
            if mode == "h":
                h0 = nat["h"].clamp(-0.999, 0.999)
                var = (torch.atanh(h0) if is_gru else h0).clone().requires_grad_(True)
            else:
                var = nat["zn"].clone().requires_grad_(True)
            opt = torch.optim.Adam([var], lr=lr)
            for it in range(iters):
                if mode == "h":
                    h = torch.tanh(var) if is_gru else var
                    zl = nat["zn"][:, -1]
                else:
                    h = wm.dyn.belief(var)
                    zl = var[:, -1]
                pr = wm.dyn.decode(h, zl, ks_fit)
                loss = ((pr - zf[:, :F_FIT]) ** 2).mean((1, 2)).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                if mode == "h":
                    h = torch.tanh(var) if is_gru else var
                    zl = nat["zn"][:, -1]
                else:
                    h = wm.dyn.belief(var)
                    zl = var[:, -1]
                pos = wm.from_belief(h, zl, meta).cpu().numpy()
        a, b = HOLD
        res[f"inv_{mode}_err_fit"] = pos_err_px(pos[:, :F_FIT], fut[:, :F_FIT], meta.active)
        res[f"inv_{mode}_err_hold"] = pos_err_px(pos[:, a:b], fut[:, a:b], meta.active)
        res[f"inv_{mode}_geff_hold"] = g_eff_pred(pos[:, a:b], fut[:, a:b], g_target, meta.active, max_start=5)
        res[f"inv_{mode}_geff_fit"] = g_eff_pred(pos[:, :b], fut[:, :b], g_target, meta.active, max_start=F_FIT)
        res[f"inv_{mode}_lambda_hold"] = law_fraction(pos[:, a:b], fut[:, a:b], clamp[:, a:b], meta.active)
    a, b = HOLD
    res["inv_true_geff_hold"] = g_eff_pred(fut[:, a:b, :, :2], fut[:, a:b], g_target, meta.active, max_start=5)
    return res


# ---------------------------------------------------------------------------------------------
# E4 velocity edits (exploratory)

def e4_velocity(wm, n_pairs=800, n_test=300, dv_test=0.5, seed=1):
    v, t = world.load("val"), world.load("test")
    rng = np.random.default_rng(seed)
    idx = rng.choice(conservative(v), n_pairs, replace=True)
    meta = Meta(v, idx)
    dv = rng.choice([-1, 1], n_pairs) * rng.uniform(0.3, 0.8, n_pairs)
    s1 = v["states"][idx, C - 1].astype(np.float64)
    s2 = s1.copy()
    s2[:, 0, 2] += dv
    h1 = wm.run(reversed_context(s1, meta, meta.g)[..., :2], meta)["h"].cpu().numpy()
    h2 = wm.run(reversed_context(s2, meta, meta.g)[..., :2], meta)["h"].cpu().numpy()
    u = ((h2 - h1) / dv[:, None]).mean(0)
    sel = conservative(t)[:n_test]
    mt = Meta(t, sel)
    out = wm.run(t["states"][sel, :C, :, :2], mt)
    base = out["pos"].cpu().numpy()
    U = torch.as_tensor(u, dtype=torch.float32, device="cuda")
    edit = wm.from_belief(out["h"] + dv_test * U, out["zn"][:, -1], mt).cpu().numpy()
    sC = t["states"][sel, C - 1].astype(np.float64)
    sC[:, 0, 2] += dv_test
    cf = sim_from(sC, mt, mt.g)[:, 1:]
    res = {"vx_cer": 1 - pos_err_px(edit, cf, mt.active) / np.maximum(pos_err_px(base, cf, mt.active), 1e-6)}

    def px_E(pos):
        x = np.unwrap(2 * np.pi * pos[..., 0], axis=1) / (2 * np.pi)
        y = pos[..., 1]
        vx = (x[:, 2:] - x[:, :-2]) / (2 * DT)
        vy = (y[:, 2:] - y[:, :-2]) / (2 * DT)
        m = (mt.mass * mt.active)[:, None, :]
        return (m * vx).sum(-1), (0.5 * m * (vx ** 2 + vy ** 2) + m * mt.g[:, None, None] * y[:, 1:-1]).sum(-1)

    pb, Eb = px_E(base)
    pe, Ee = px_E(edit)
    pc, Ec = px_E(cf[..., :2])
    pt, Et = px_E(t["states"][sel, C:C + R, :, :2])
    res.update(vx_dpx_model=pe - pb, vx_dpx_true=pc - pt, vx_dE_model=Ee - Eb, vx_dE_true=Ec - Et)
    return res


# ---------------------------------------------------------------------------------------------
# E5 conservation drift with chunked autoregression

def e5_drift(wm, n_chunks=4):
    p = world.load("long")
    meta = Meta(p)
    with torch.no_grad():
        zn = torch.cat([wm.dyn.norm(wm.latents(torch.as_tensor(p["states"][i:i + 256, :C, :, :2], device="cuda"),
                                                meta.sub(slice(i, i + 256))))
                        for i in range(0, meta.B, 256)])
        preds = []
        for c in range(n_chunks):
            h = wm.dyn.belief(zn)
            pr = wm.dyn.decode(h, zn[:, -1])
            preds.append(pr)
            zn = pr[:, -C:]
        zall = torch.cat(preds, 1)
        pos = torch.cat([wm.positions(zall[i:i + 256], meta.sub(slice(i, i + 256))) for i in range(0, meta.B, 256)])
    pos = pos.cpu().numpy()
    Tn = pos.shape[1]
    ref = p["states"][:, C:C + Tn]
    x = np.unwrap(2 * np.pi * pos[..., 0], axis=1) / (2 * np.pi)
    y = pos[..., 1]
    vx = (x[:, 2:] - x[:, :-2]) / (2 * DT)
    vy = (y[:, 2:] - y[:, :-2]) / (2 * DT)
    m = (meta.mass * meta.active)[:, None, :]
    E = (0.5 * m * (vx ** 2 + vy ** 2) + m * meta.g[:, None, None] * y[:, 1:-1]).sum(-1)
    px = (m * vx).sum(-1)
    Et = p["energy"][:, C + 1:C + Tn - 1]
    pxt = p["px"][:, C + 1:C + Tn - 1]
    err_t = np.array([pos_err_px(pos[:, i:i + 1], ref[:, i:i + 1], meta.active).mean() for i in range(Tn)])
    return dict(long_Erel=np.abs(E - Et) / Et, long_pxerr=np.abs(px - pxt) / m.sum(-1),
                long_err_t=err_t)


# ---------------------------------------------------------------------------------------------
# E6 other OOD

def e6_ood(wm):
    res = {}
    for split in ("test", "count45", "speed", "massratio", "mass_ctrl"):
        p = world.load(split)
        meta = Meta(p)
        pos = wm.run(p["states"][:, :C, :, :2], meta)["pos"].cpu().numpy()
        res[f"{split}_err_px"] = pos_err_px(pos, p["states"][:, C:C + R], meta.active)
        res[f"{split}_err8_px"] = pos_err_px(pos[:, :8], p["states"][:, C:C + 8], meta.active)
    return res


def untrained_copy(wm, seed=123):
    import copy
    torch.manual_seed(seed)
    u = copy.copy(wm)
    d = ChunkWM(wm.dyn.din, wm.dyn.size, wm.dyn.core).cuda().eval()
    d.mu.copy_(wm.dyn.mu)
    d.sd.copy_(wm.dyn.sd)
    for q in d.parameters():
        q.requires_grad_(False)
    u.dyn = d
    return u
