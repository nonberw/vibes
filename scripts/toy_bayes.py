"""Illustration of Proposition 2 in the cleanest setting (CPU, minutes): 1D free fall, no walls.

Input: 10 noisy heights y_0..y_9 (dt = 0.05) of a particle with y0 ~ U[0.3, 0.9], v0 ~ U[-1.5, 1.5]
and g ~ U[4, 8]; target: the next 20 heights (noise-free). Predictors trained by squared loss:
  - 'linear'  : y_future = W y_past + b     (the exact law is linear in the data for free fall)
  - 'mlp'     : 2-hidden-layer ReLU MLP
  - 'mlp_tanh': same with tanh
Observation noise sigma in {0, 0.25, 1} px (1 px = 1/64). For each model we measure the implied
gravity of its prediction on test gravities 1..12 and compare with the exact Bayes posterior-mean
curve (truncated normal) and the law. Writes paper/fig_toy.pdf and results/v2/toy_bayes.json.

    python scripts/toy_bayes.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vibes.stats import truncnorm_mean  # noqa: E402

DT, C, R = 0.05, 10, 20
torch.set_num_threads(4)


def episodes(N, g_lo, g_hi, rng, g_values=None):
    y0 = rng.uniform(0.3, 0.9, N)
    v0 = rng.uniform(-1.5, 1.5, N)
    g = rng.uniform(g_lo, g_hi, N) if g_values is None else rng.choice(g_values, N)
    t = np.arange(C + R) * DT
    y = y0[:, None] + v0[:, None] * t[None] - 0.5 * g[:, None] * t[None] ** 2
    return y.astype(np.float32), g


def implied_g(pred):
    t = np.arange(R) * DT
    A = np.stack([np.ones(R), t, t ** 2], 1)
    c = np.linalg.lstsq(A, pred.T, rcond=None)[0]
    return -2 * c[2]


def bayes_curve(y_ctx_noisy, sigma, a=4.0, b=8.0):
    """Posterior mean of g given noisy context heights, flat prior on (y0, v0), U[a,b] on g."""
    t = np.arange(C) * DT
    A = np.stack([np.ones(C), t, -0.5 * t ** 2], 1)
    pinv = np.linalg.pinv(A)
    ghat = (pinv @ y_ctx_noisy.T)[2]
    s = sigma * np.sqrt((pinv[2] ** 2).sum())
    return truncnorm_mean(ghat, s, a, b)


def make(kind):
    if kind == "linear":
        return nn.Linear(C, R)
    act = nn.ReLU if kind == "mlp" else nn.Tanh
    return nn.Sequential(nn.Linear(C, 256), act(), nn.Linear(256, 256), act(), nn.Linear(256, R))


def train(kind, y, sigma, seed, steps=6000, B=256):
    torch.manual_seed(seed)
    g_ = torch.Generator().manual_seed(seed)
    m = make(kind)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    Y = torch.as_tensor(y)
    for step in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * step / steps))
        idx = torch.randint(0, len(Y), (B,), generator=g_)
        ctx = Y[idx, :C] + sigma / 64 * torch.randn(B, C, generator=g_)
        loss = ((m(ctx) - Y[idx, C:]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


def main():
    rng = np.random.default_rng(0)
    ytr, _ = episodes(200_000, 4, 8, rng)
    grid = np.round(np.arange(1, 12.01, 0.5), 2)
    yte, gte = episodes(4000, 1, 12, rng, g_values=grid)
    out = {"grid": grid.tolist(), "curves": {}}
    for sigma in (0.0, 0.25, 1.0):
        noise = sigma / 64 * rng.standard_normal((len(yte), C)).astype(np.float32)
        ctx = yte[:, :C] + noise
        bay = bayes_curve(ctx, sigma / 64)
        out["curves"][f"bayes_{sigma}"] = [float(bay[gte == x].mean()) for x in grid]
        for kind in ("linear", "mlp", "mlp_tanh"):
            curves = []
            for seed in range(3):
                m = train(kind, ytr, sigma, seed)
                with torch.no_grad():
                    pred = m(torch.as_tensor(ctx)).numpy()
                gi = implied_g(pred)
                curves.append([float(gi[gte == x].mean()) for x in grid])
            out["curves"][f"{kind}_{sigma}"] = np.mean(curves, 0).tolist()
            print(sigma, kind, "g=1:", round(out["curves"][f"{kind}_{sigma}"][0], 2),
                  "g=12:", round(out["curves"][f"{kind}_{sigma}"][-1], 2), flush=True)
    (ROOT / "results" / "v2").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "v2" / "toy_bayes.json").write_text(json.dumps(out, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 2.6), sharey=True)
    for a, sigma in zip(ax, (0.0, 0.25, 1.0)):
        a.axvspan(4, 8, color="#eeeeee", zorder=0)
        a.plot(grid, grid, "k--", lw=0.8, label="law")
        a.plot(grid, np.clip(grid, 4, 8), "k:", lw=0.8, label="clamp")
        a.plot(grid, out["curves"][f"bayes_{sigma}"], color="#999999", lw=2, label="Bayes posterior mean")
        for kind, col, lab in (("linear", "#2a7f9e", "linear"), ("mlp", "#c8553d", "ReLU MLP"),
                               ("mlp_tanh", "#7b5ea7", "tanh MLP")):
            a.plot(grid, out["curves"][f"{kind}_{sigma}"], "o-", ms=2.5, color=col, label=lab)
        a.set_title(f"observation noise $\\sigma$ = {sigma:g} px", fontsize=8)
        a.set_xlabel("true $g$")
        a.set_ylim(-1, 13)
    ax[0].set_ylabel("implied gravity of prediction")
    ax[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "paper" / "fig_toy.pdf")
    fig.savefig(ROOT / "results" / "v2" / "fig_toy.png", dpi=200)
    print("saved")


if __name__ == "__main__":
    main()
