"""Illustration of Proposition 2 in the cleanest setting (CPU, minutes): 1D free fall, no walls.

Input: 10 noisy heights y_0..y_9 (dt = 0.05) of a particle with y0 ~ U[0.3, 0.9], v0 ~ U[-1.5, 1.5]
and g ~ U[4, 8]; target: the next 20 heights (noise-free). The law does not depend on absolute
height, so the models see the differences y_t - y_9 (standardized) and predict offsets
y_future - y_9. Predictors trained by squared loss:
  - 'linear_cf' : the exact empirical-risk minimizer of a linear predictor (closed form)
  - 'linear'    : the same model trained by Adam
  - 'mlp'       : 2-hidden-layer ReLU MLP (Adam)
  - 'mlp_tanh'  : same with tanh
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
    return y.astype(np.float64), g


def implied_g(pred):
    """pred: (N, R) predicted future heights (any constant offset) -> -2 x quadratic coefficient."""
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


class Feat:
    """Differences to the last context height, standardized with training statistics."""

    def __init__(self, y_ctx):
        d = y_ctx[:, :-1] - y_ctx[:, -1:]
        self.mu, self.sd = d.mean(0), d.std(0) + 1e-9

    def __call__(self, y_ctx):
        d = y_ctx[:, :-1] - y_ctx[:, -1:]
        return (d - self.mu) / self.sd


def make(kind):
    if kind == "linear":
        return nn.Linear(C - 1, R)
    act = nn.ReLU if kind == "mlp" else nn.Tanh
    return nn.Sequential(nn.Linear(C - 1, 256), act(), nn.Linear(256, 256), act(), nn.Linear(256, R))


def train(kind, X, T, seed, steps=15000, B=512, lr=2e-3):
    torch.manual_seed(seed)
    g_ = torch.Generator().manual_seed(seed)
    m = make(kind)
    opt = torch.optim.Adam(m.parameters(), lr)
    X, T = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(T, dtype=torch.float32)
    for step in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = lr * min(1, step / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        idx = torch.randint(0, len(X), (B,), generator=g_)
        loss = ((m(X[idx]) - T[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


def main():
    js = ROOT / "results" / "v2" / "toy_bayes.json"
    if "--plot" in sys.argv and js.exists():
        out = json.loads(js.read_text())
        return plot(out, np.array(out["grid"]))
    rng = np.random.default_rng(0)
    ytr, gtr = episodes(200_000, 4, 8, rng)
    grid = np.round(np.arange(1, 12.01, 0.5), 2)
    yte, gte = episodes(4000, 1, 12, rng, g_values=grid)
    out = {"grid": grid.tolist(), "curves": {}, "in_range_mse": {}}
    inr = (gte >= 4) & (gte <= 8)
    for sigma in (0.0, 0.25, 1.0):
        s = sigma / 64
        ctx_tr = ytr[:, :C] + s * rng.standard_normal((len(ytr), C))
        ctx_te = yte[:, :C] + s * rng.standard_normal((len(yte), C))
        feat = Feat(ctx_tr)
        Xtr, Ttr = feat(ctx_tr), ytr[:, C:] - ctx_tr[:, -1:]
        Xte, Tte = feat(ctx_te), yte[:, C:] - ctx_te[:, -1:]
        bay = bayes_curve(ctx_te, s)
        out["curves"][f"bayes_{sigma}"] = [float(bay[gte == x].mean()) for x in grid]
        # closed-form linear ERM (with intercept)
        A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        W = np.linalg.lstsq(A, Ttr, rcond=None)[0]
        pred = np.hstack([Xte, np.ones((len(Xte), 1))]) @ W
        gi = implied_g(pred)
        out["curves"][f"linear_cf_{sigma}"] = [float(gi[gte == x].mean()) for x in grid]
        out["in_range_mse"][f"linear_cf_{sigma}"] = float(((pred - Tte) ** 2)[inr].mean())
        for kind in ("linear", "mlp", "mlp_tanh"):
            curves, mses = [], []
            for seed in range(3):
                m = train(kind, Xtr, Ttr, seed)
                with torch.no_grad():
                    pred = m(torch.as_tensor(Xte, dtype=torch.float32)).numpy().astype(np.float64)
                curves.append([float(implied_g(pred)[gte == x].mean()) for x in grid])
                mses.append(float(((pred - Tte) ** 2)[inr].mean()))
            out["curves"][f"{kind}_{sigma}"] = np.mean(curves, 0).tolist()
            out["in_range_mse"][f"{kind}_{sigma}"] = float(np.mean(mses))
            print(f"sigma {sigma} {kind:9s} in-range mse {np.mean(mses):.2e} (closed form "
                  f"{out['in_range_mse'][f'linear_cf_{sigma}']:.2e}) | implied g at 1: "
                  f"{out['curves'][f'{kind}_{sigma}'][0]:.2f}, at 12: {out['curves'][f'{kind}_{sigma}'][-1]:.2f}",
                  flush=True)
    (ROOT / "results" / "v2").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "v2" / "toy_bayes.json").write_text(json.dumps(out, indent=1))
    plot(out, grid)


def plot(out, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 2.6), sharey=True)
    for a, sigma in zip(ax, (0.0, 0.25, 1.0)):
        a.axvspan(4, 8, color="#eeeeee", zorder=0)
        a.plot(grid, grid, "k--", lw=0.8, label="law")
        a.plot(grid, np.clip(grid, 4, 8), "k:", lw=0.8, label="clamp")
        bay = np.array(out["curves"][f"bayes_{sigma}"])
        if sigma == 0:   # without noise the posterior is undefined for out-of-range contexts (Prop. 2 iii)
            bay = np.where((grid >= 4) & (grid <= 8), bay, np.nan)
        a.plot(grid, bay, color="#999999", lw=2.5, label="Bayes posterior mean")
        for kind, col, lab, mk in (("linear_cf", "#2a7f9e", "linear, closed form", "s"),
                                   ("linear", "#2a7f9e", "linear, Adam", "o"),
                                   ("mlp", "#c8553d", "ReLU MLP", "o"), ("mlp_tanh", "#7b5ea7", "tanh MLP", "o")):
            a.plot(grid, out["curves"][f"{kind}_{sigma}"], marker=mk, ms=2.5, lw=1, color=col,
                   mfc="none" if kind == "linear_cf" else col, label=lab)
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
