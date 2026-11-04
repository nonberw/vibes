"""Episode samplers, dataset generation and GPU batch loading.

Every split is defined by a `Spec` (distribution over per-episode physics) and a fixed seed, so
all datasets are bit-reproducible from `scripts/gen_data.py`.
"""
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from . import sim
from .render import render

KMAX = 5
T_TRAIN = 40


@dataclass
class Spec:
    n_choices: tuple = (1, 2, 3)
    p_block: float = 0.3
    ball_r: tuple = (0.06, 0.10)
    block_a: tuple = (0.05, 0.085)
    g: tuple = (4.0, 8.0)            # uniform range; or use g_values
    g_values: tuple = ()             # if set, g is drawn uniformly from this list
    p_elastic: float = 0.5           # P(e = 1)
    e_range: tuple = (0.6, 1.0)
    p_frictionless: float = 0.5      # P(mu = 0)
    mu_range: tuple = (0.05, 0.4)
    y_range: tuple = (0.25, 0.85)
    v_max: float = 1.5               # vx, vy ~ U[-v_max, v_max]
    v_min_speed: float = 0.0         # reject if |v| below this (for speed-OOD sets)
    force_conservative: bool = False
    sizes_fixed: tuple = ()          # explicit sizes for the first objects (mass-ratio OOD)
    kinds_fixed: tuple = ()


TRAIN = Spec()
SPECS = {
    "train": TRAIN,
    "val": TRAIN,
    "test": TRAIN,
    # gravity sweep: everything in-distribution except g
    "gsweep": replace(TRAIN, g_values=tuple(np.round(np.arange(1.0, 12.01, 0.5), 2))),
    # long conservative episodes for drift
    "long": replace(TRAIN, force_conservative=True),
    "count45": replace(TRAIN, n_choices=(4, 5)),
    "speed": replace(TRAIN, v_max=3.0, v_min_speed=2.2),
    # two balls with a 12x mass ratio (sizes are visual OOD too) vs an in-distribution control
    "massratio": replace(TRAIN, n_choices=(2,), sizes_fixed=(0.04, 0.14), kinds_fixed=(0, 0)),
    "mass_ctrl": replace(TRAIN, n_choices=(2,), kinds_fixed=(0, 0)),
}
SEEDS = {"train": 0, "val": 1, "test": 2, "gsweep": 3, "long": 4, "count45": 5, "speed": 6,
         "massratio": 7, "mass_ctrl": 8}


def sample_params(spec: Spec, N: int, rng: np.random.Generator):
    n = rng.choice(np.array(spec.n_choices), size=N)
    kind = (rng.random((N, KMAX)) < spec.p_block).astype(np.int64)
    r_ball = rng.uniform(*spec.ball_r, size=(N, KMAX))
    a_blk = rng.uniform(*spec.block_a, size=(N, KMAX))
    for i, kf in enumerate(spec.kinds_fixed):
        kind[:, i] = kf
    size = np.where(kind == sim.BALL, r_ball, a_blk)
    for i, sf in enumerate(spec.sizes_fixed):
        size[:, i] = sf
    active = (np.arange(KMAX)[None, :] < n[:, None])
    mass = sim.area(kind, size) * active + (~active) * 1.0  # inactive: dummy mass 1
    if spec.g_values:
        g = rng.choice(np.array(spec.g_values), size=N)
    else:
        g = rng.uniform(*spec.g, size=N)
    e = np.where(rng.random(N) < spec.p_elastic, 1.0, rng.uniform(*spec.e_range, size=N))
    mu = np.where(rng.random(N) < spec.p_frictionless, 0.0, rng.uniform(*spec.mu_range, size=N))
    if spec.force_conservative:
        e[:] = 1.0
        mu[:] = 0.0
    color = np.stack([rng.permutation(8)[:KMAX] for _ in range(N)])
    x0 = np.zeros((N, KMAX, 4))
    for b in range(N):
        placed = []
        for k in range(n[b]):
            for _ in range(1000):
                x = rng.random()
                y = rng.uniform(max(spec.y_range[0], size[b, k] + 0.01),
                                min(spec.y_range[1], 1 - size[b, k] - 0.01))
                ok = True
                for (xx, yy, ss) in placed:
                    dx = (x - xx) - np.floor(x - xx + 0.5)
                    if np.hypot(dx, y - yy) < (ss + size[b, k]) * 1.45 + 0.02:
                        ok = False
                        break
                if ok:
                    break
            placed.append((x, y, size[b, k]))
            while True:
                v = rng.uniform(-spec.v_max, spec.v_max, size=2)
                if np.hypot(*v) >= spec.v_min_speed:
                    break
            x0[b, k] = [x, y, v[0], v[1]]
        for k in range(n[b], KMAX):  # park inactive objects off-screen (never simulated)
            x0[b, k] = [0.0, -10.0, 0.0, 0.0]
    return dict(x0=x0, kind=kind, size=size, mass=mass, n=n.astype(np.int64), g=g, e=e, mu=mu,
                color=color.astype(np.int64), active=active)


def generate(split: str, N: int, T: int = T_TRAIN, seed=None):
    spec = SPECS[split]
    rng = np.random.default_rng(SEEDS[split] if seed is None else seed)
    p = sample_params(spec, N, rng)
    states = sim.simulate_batch(p["x0"], p["kind"], p["size"], p["mass"], p["n"], p["g"], p["e"],
                                p["mu"], T)
    p["states"] = states.astype(np.float32)
    p["energy"] = sim.energy(states, p["mass"], p["g"], p["active"]).astype(np.float32)
    pxm, pym = sim.momentum(states, p["mass"], p["active"])
    p["px"], p["py"] = pxm.astype(np.float32), pym.astype(np.float32)
    return p


def save(p, path):
    np.savez_compressed(path, **p)


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


class GPUData:
    """Holds a split on the GPU and renders random windows on the fly."""

    def __init__(self, p, device="cuda", n_limit=None):
        sl = slice(0, n_limit)
        self.N = len(p["g"][sl])
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a)[sl], dtype=dt, device=device)
        self.states = t(p["states"])
        self.kind = t(p["kind"], torch.int64)
        self.size = t(p["size"])
        self.color = t(p["color"], torch.int64)
        self.active = t(p["active"], torch.bool)
        self.mass = t(p["mass"])
        self.g, self.e, self.mu = t(p["g"]), t(p["e"]), t(p["mu"])
        self.T = self.states.shape[1]
        self.device = device

    def frames(self, idx, t0, L):
        """Render episodes idx (LongTensor) from frame t0 (LongTensor or int) for L frames."""
        if isinstance(t0, int):
            st = self.states[idx, t0:t0 + L]
        else:
            ar = t0[:, None] + torch.arange(L, device=self.device)[None]
            st = self.states[idx[:, None], ar]
        x = render(st[..., :2], self.kind[idx], self.size[idx], self.color[idx], self.active[idx])
        return x, st

    def sample(self, B, L, gen):
        idx = torch.randint(0, self.N, (B,), device=self.device, generator=gen)
        t0 = torch.randint(0, self.T - L + 1, (B,), device=self.device, generator=gen)
        x, st = self.frames(idx, t0, L)
        return x, st, idx
