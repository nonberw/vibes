"""Core analysis machinery: rollouts with physical readouts, probes, matched-state counterfactual
contexts (time reversal), steering, and the Bayes reference curve."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import truncnorm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from . import sim
from .data import GPUData, load
from .models import WorldModel, state_obs
from .readout import centroids, g_eff_batch
from .render import render

C = 8          # context frames; h_C is the GRU state after frame index C-1
DT = sim.DT


def load_run(run_dir, device="cuda"):
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    m = WorldModel(cfg["family"], cfg["size"]).to(device)
    m.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location=device))
    m.eval()
    return m, cfg


def untrained_like(cfg, seed=123, device="cuda"):
    torch.manual_seed(seed)
    return WorldModel(cfg["family"], cfg["size"]).to(device).eval()


# ---------------------------------------------------------------------------------------------
# episode tensors and model inputs

class Episodes:
    """A batch of episodes given explicitly by states + static properties (numpy)."""

    def __init__(self, states, kind, size, color, active, mass, g, e, mu):
        self.states, self.kind, self.size, self.color = states, kind, size, color
        self.active, self.mass, self.g, self.e, self.mu = active, mass, g, e, mu

    @classmethod
    def from_split(cls, p, idx=None):
        idx = np.arange(len(p["g"])) if idx is None else idx
        return cls(p["states"][idx], p["kind"][idx], p["size"][idx], p["color"][idx],
                   p["active"][idx], p["mass"][idx], p["g"][idx], p["e"][idx], p["mu"][idx])

    def __len__(self):
        return len(self.g)

    def sub(self, idx):
        return Episodes(*(a[idx] for a in (self.states, self.kind, self.size, self.color,
                                           self.active, self.mass, self.g, self.e, self.mu)))


def model_inputs(model, ep, t0, L, device="cuda"):
    st = torch.as_tensor(ep.states[:, t0:t0 + L], dtype=torch.float32, device=device)
    kind = torch.as_tensor(ep.kind, device=device)
    size = torch.as_tensor(ep.size, dtype=torch.float32, device=device)
    act = torch.as_tensor(ep.active, device=device)
    if model.family == "state":
        return state_obs(st, kind, size, act)
    return render(st[..., :2], kind, size, torch.as_tensor(ep.color, device=device), act)


@torch.no_grad()
def encode_context(model, ep, chunk=128, device="cuda"):
    """Returns h_C (N, H) for the first C frames of each episode."""
    hs = []
    for i in range(0, len(ep), chunk):
        e = ep.sub(slice(i, i + chunk))
        z = model.encode(model_inputs(model, e, 0, C, device))
        _, h = model.observe(z)
        hs.append(h.float())
    return torch.cat(hs)


@torch.no_grad()
def readout_positions(model, ep, zr, device="cuda"):
    """Decode predicted latents zr (B, R, DZ) of episodes ep to object positions (B, R, K, 2)."""
    if model.family == "state":
        out = model.decode(zr).float()                      # (B, R, K, 3): cos, sin, y
        x = (torch.atan2(out[..., 1], out[..., 0]) / (2 * math.pi)) % 1.0
        return torch.stack([x, out[..., 2]], -1)
    frames = model.decode(zr).float().clamp(0, 1)
    pos, _ = centroids(frames, torch.as_tensor(ep.color, device=device),
                       torch.as_tensor(ep.active, device=device))
    return pos


@torch.no_grad()
def rollout_from_h(model, ep, h, R, chunk=64, device="cuda"):
    """Open-loop rollout of R frames from belief states h; returns readout positions (N, R, K, 2)."""
    out = []
    for i in range(0, len(ep), chunk):
        hh = h[i:i + chunk]
        zr, _ = model.imagine(hh, R)
        out.append(readout_positions(model, ep.sub(slice(i, i + chunk)), zr.float(), device).cpu())
    return torch.cat(out).numpy()


def g_eff(pos, ep):
    return g_eff_batch(pos, ep.size, ep.active, DT)


def pos_error(pos, states, active):
    """Mean Euclidean (periodic-x) position error per episode over frames and active objects."""
    dx = pos[..., 0] - states[..., 0]
    dx = dx - np.round(dx)
    d = np.hypot(dx, pos[..., 1] - states[..., 1])
    a = active[:, None, :].astype(float)
    return (d * a).sum((1, 2)) / (a.sum((1, 2)) * 1.0)


# ---------------------------------------------------------------------------------------------
# probes

def fit_ridge(X, y):
    r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X, y)
    return r


def probe_r2(Xtr, ytr, Xte, yte):
    r = fit_ridge(Xtr, ytr)
    return r2_score(yte, r.predict(Xte)), r


# ---------------------------------------------------------------------------------------------
# matched-state counterfactual contexts by time reversal (conservative episodes only)

def reversed_context(state_C, kind, size, mass, n, g):
    """Context frames 0..C-1 whose last frame is exactly state_C under gravity g (e=1, mu=0).

    Simulates the velocity-reversed state forward under g for C frames and reverses time.
    Valid because conservative dynamics are time-reversible.
    """
    N = len(g)
    x0 = state_C.copy()
    x0[..., 2:] *= -1
    one = np.ones(N)
    back = sim.simulate_batch(x0, kind, size, mass, n, g, one, 0.0 * one, C)
    ctx = back[:, ::-1].copy()
    ctx[..., 2:] *= -1
    return ctx


def forward_from(state, kind, size, mass, n, g, e, mu, T):
    return sim.simulate_batch(state.copy(), kind, size, mass, n, g, e, mu, T)


def matched_pairs(p, n_pairs, rng, gmin=4.0, gmax=8.0, min_gap=1.0):
    """Pairs of contexts ending in the same state (frame C-1 of a conservative episode) under
    two gravities g, g' in [gmin, gmax]. Returns two Episodes objects and (g, g')."""
    cons = np.where((p["e"] == 1.0) & (p["mu"] == 0.0))[0]
    idx = rng.choice(cons, size=n_pairs, replace=len(cons) < n_pairs)
    g1 = rng.uniform(gmin, gmax, n_pairs)
    g2 = rng.uniform(gmin, gmax, n_pairs)
    bad = np.abs(g2 - g1) < min_gap
    while bad.any():
        g2[bad] = rng.uniform(gmin, gmax, bad.sum())
        bad = np.abs(g2 - g1) < min_gap
    sC = p["states"][idx, C - 1].astype(np.float64)
    args = (p["kind"][idx], p["size"][idx], p["mass"][idx], p["n"][idx])
    c1 = reversed_context(sC, *args, g1).astype(np.float32)
    c2 = reversed_context(sC, *args, g2).astype(np.float32)
    mk = lambda st, g: Episodes(st, p["kind"][idx], p["size"][idx], p["color"][idx],
                                p["active"][idx], p["mass"][idx], g, np.ones(n_pairs),
                                np.zeros(n_pairs))
    return mk(c1, g1), mk(c2, g2), idx


# ---------------------------------------------------------------------------------------------
# Bayes reference (Prop. 2): posterior mean of g under U[a, b] given noisy context positions

def bayes_reference(p, idx, a=4.0, b=8.0, sigma=1 / 64, W=7, n_mc=200, rng=None):
    """For each episode, the expected posterior mean E[g | context] under prior U[a, b] when the
    context positions carry iid Gaussian noise sigma. Uses free-flight 7-frame windows of the
    context (ground-truth geometry), combined across objects by inverse variance."""
    rng = rng or np.random.default_rng(0)

    def sd_g_of(n):
        t = np.arange(n) * DT
        A = np.stack([np.ones(n), t, t ** 2], 1)
        return 2 * sigma * np.sqrt((np.linalg.pinv(A)[2] ** 2).sum())

    out = np.full(len(idx), np.nan)
    for j, b_ in enumerate(idx):
        st = p["states"][b_]
        info = 0.0   # Fisher information about g, summed over objects (longest free segment each)
        for k in range(p["n"][b_]):
            y = st[:C, k, 1]
            # exact contact-free frames from the true trajectory: ballistic iff the second
            # difference equals -g dt^2 on both sides
            dd = np.diff(y, 2) + p["g"][b_] * DT ** 2
            ok_mid = np.abs(dd) < 1e-6
            # longest run of consecutive frames whose interior second differences are ballistic
            best, run = 0, 0
            for i in range(C - 2):
                run = run + 1 if ok_mid[i] else 0
                best = max(best, run)
            npts = best + 2 if best > 0 else 0
            if npts >= 4:
                info += 1.0 / sd_g_of(npts) ** 2
        g_true = p["g"][b_]
        if info == 0:
            out[j] = (a + b) / 2
            continue
        sd = 1 / np.sqrt(info)
        ghat = g_true + sd * rng.standard_normal(n_mc)
        lo, hi = (a - ghat) / sd, (b - ghat) / sd
        out[j] = truncnorm.mean(lo, hi, loc=ghat, scale=sd).mean()
    return out
