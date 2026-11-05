"""Proxy competence experiments on ground-truth features (pilot only; train/val pools generated
in memory, never the evaluation splits). Used to choose the world timing and the dynamics
architecture before pre-registration v2.

    python scripts/proxy.py --arch gru_chunk --dt 0.05 --C 10 --R 20 --steps 20000
"""
import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes import sim  # noqa: E402
from vibes.data import SPECS, sample_params  # noqa: E402
from vibes.models import state_obs  # noqa: E402
from vibes.readout import g_eff_ref_batch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="gru_chunk")
ap.add_argument("--dt", type=float, default=0.05)
ap.add_argument("--C", type=int, default=10)
ap.add_argument("--R", type=int, default=20)
ap.add_argument("--T", type=int, default=40)
ap.add_argument("--N", type=int, default=200000)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--B", type=int, default=128)
ap.add_argument("--H", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--n", default="1,2,3")
ap.add_argument("--vis", type=int, default=1)
ap.add_argument("--eval_every", type=int, default=10000)
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)
C, R = a.C, a.R


def pool(N, seed):
    spec = replace(SPECS["train"], n_choices=tuple(int(x) for x in a.n.split(",")))
    p = sample_params(spec, N, np.random.default_rng(seed))
    st = sim.simulate_batch(p["x0"], p["kind"], p["size"], p["mass"], p["n"], p["g"], p["e"], p["mu"], a.T, a.dt)
    t = lambda x, d=torch.float32: torch.as_tensor(x, dtype=d, device=dev)
    return dict(xy=torch.as_tensor(st[..., :2], dtype=torch.float16 if N > 50000 else torch.float32, device=dev),
                kind=t(p["kind"], torch.int64), size=t(p["size"]), active=t(p["active"], torch.bool),
                g=t(p["g"]), e=t(p["e"]), mu=t(p["mu"]), N=N, states=st)


t0 = time.time()
TR = pool(a.N, 100)
VA = pool(1000, 101)
print(f"pools ready {time.time() - t0:.0f}s", flush=True)


def feats(xy, P, idx):
    st = torch.zeros(*xy.shape[:-1], 4, device=dev)
    st[..., :2] = xy
    o = state_obs(st, P["kind"][idx], P["size"][idx], P["active"][idx])
    d = torch.cat([torch.zeros_like(o[:, :1]), o[:, 1:] - o[:, :-1]], 1)
    f = [o, 10 * d]
    if a.vis:
        f.append(torch.stack([(P["e"][idx] - 0.8) * 5, P["mu"][idx] * 5], -1)[:, None].expand(-1, o.shape[1], 2))
    return torch.cat(f, -1)


DIN = 60 + 2 * a.vis


class Model(nn.Module):
    def __init__(s):
        super().__init__()
        H = a.H
        if a.arch in ("gru_chunk", "ar"):
            s.bel = nn.GRU(DIN, H, batch_first=True)
        else:
            s.bel = nn.Sequential(nn.Linear(DIN * C, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, H))
        if a.arch == "ar":
            s.cell = nn.GRUCell(DIN, H)
            s.out = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 10))
        else:
            s.k = nn.Embedding(R, 64)
            s.dec = nn.Sequential(nn.Linear(H + 64, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 10))

    def belief(s, f):
        if a.arch in ("gru_chunk", "ar"):
            return s.bel(f)[0][:, -1]
        return s.bel(f.flatten(1))

    def forward(s, xy_ctx, P, idx, Rr=R):
        f = feats(xy_ctx, P, idx)
        h = s.belief(f)
        last = xy_ctx[:, -1]
        if a.arch != "ar":
            Bn = h.shape[0]
            e = s.k.weight[None, :Rr].expand(Bn, Rr, 64)
            return last[:, None] + s.dec(torch.cat([h[:, None].expand(Bn, Rr, h.shape[1]), e], -1)).view(Bn, Rr, 5, 2), h
        preds, prev, cur, hh = [], xy_ctx[:, -2], last, h
        for k in range(Rr):
            nxt = cur + 0.05 * s.out(hh).view(-1, 5, 2)
            preds.append(nxt)
            ff = feats(torch.stack([cur, nxt], 1), P, idx)[:, 1]
            hh = s.cell(ff, hh)
            prev, cur = cur, nxt
        return torch.stack(preds, 1), h


m = Model().to(dev)
opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=1e-4)
gen = torch.Generator(device=dev).manual_seed(0)


def lossf(pred, fut, act):
    dx = pred[..., 0] - fut[..., 0]
    dx = dx - torch.round(dx)
    w = act[:, None, :].float()
    return ((dx ** 2 + (pred[..., 1] - fut[..., 1]) ** 2) * w).sum() / (w.sum() * pred.shape[1])


def evaluate(step, loss):
    with torch.no_grad():
        xy = VA["xy"][:, :C + R].float()
        idx = torch.arange(1000, device=dev)
        pred, h = m(xy[:, :C], VA, idx)
    P = pred.cpu().numpy()
    P[..., 0] %= 1.0
    S = VA["states"][:, C:C + R]
    g = VA["g"].cpu().numpy()
    A = VA["active"].cpu().numpy()
    ge = g_eff_ref_batch(P, S, g, A, a.dt, max_start=8)
    ok = np.isfinite(ge)
    slope = np.polyfit(g[ok], ge[ok], 1)[0]
    corr = np.corrcoef(g[ok], ge[ok])[0, 1]
    Hn = h.cpu().numpy()
    pr = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), Hn, g, cv=2)
    r2 = 1 - ((pr - g) ** 2).sum() / ((g - g.mean()) ** 2).sum()

    def err(k):
        dx = P[:, :k, :, 0] - S[:, :k, :, 0]
        dx -= np.round(dx)
        d = np.hypot(dx, P[:, :k, :, 1] - S[:, :k, :, 1])
        return d[np.broadcast_to(A[:, None], d.shape)].mean() * 64

    print(f"{a.arch} dt{a.dt} C{C} R{R} n{a.n} vis{a.vis} step {step} loss {loss:.5f} | geff slope {slope:.2f} "
          f"corr {corr:.2f} | r2_g {r2:.2f} | err8 {err(8):.2f}px errR {err(R):.2f}px | {time.time() - t0:.0f}s",
          flush=True)


for step in range(1, a.steps + 1):
    for gp in opt.param_groups:
        gp["lr"] = a.lr * min(1, step / 500) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
    idx = torch.randint(0, TR["N"], (a.B,), device=dev, generator=gen)
    t0i = torch.randint(0, a.T - (C + R) + 1, (a.B,), device=dev, generator=gen)
    xy = TR["xy"][idx[:, None], t0i[:, None] + torch.arange(C + R, device=dev)[None]].float()
    pred, _ = m(xy[:, :C], TR, idx)
    loss = lossf(pred, xy[:, C:], TR["active"][idx])
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if step % a.eval_every == 0 or step == a.steps:
        evaluate(step, loss.item())
