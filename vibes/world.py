"""VIBES v2 world: splits, generation and GPU pools.

Same physics as v1 (vibes/sim.py) with three changes made after the v1 pilot (see
PREREGISTRATION_v2.md): 20 fps (dt = 0.05) with 10 context + 20 predicted frames, a 100k-episode
training pool (v1 models memorized 16k episodes), and visible materials: restitution and friction
tint the background, so gravity is the one physical parameter that must be inferred from motion.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import sim
from .data import SPECS as V1_SPECS
from .data import KMAX, sample_params
from .render import material_bg, render

DT = 0.05
C = 10            # context frames
R = 20            # predicted frames
T = 40            # frames per episode
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v2"

BASE = V1_SPECS["train"]
G_GRID = tuple(np.round(np.arange(1.0, 12.01, 0.5), 2))
SPLITS = {
    #  name        spec                                                   N        T    seed
    "train":     (BASE,                                                   100_000, T,   20),
    "val":       (BASE,                                                   2_000,   T,   21),
    "test":      (BASE,                                                   2_000,   T,   22),
    "gsweep":    (replace(BASE, g_values=G_GRID),                         2_300,   T,   23),
    "wide":      (replace(BASE, g=(1.0, 12.0)),                           100_000, T,   24),
    "wide_val":  (replace(BASE, g=(1.0, 12.0)),                           2_000,   T,   30),
    "long":      (replace(BASE, force_conservative=True),                 1_000,   90,  25),
    "count45":   (replace(BASE, n_choices=(4, 5)),                        1_000,   T,   26),
    "speed":     (replace(BASE, v_max=3.0, v_min_speed=2.2),              1_000,   T,   27),
    "massratio": (replace(BASE, n_choices=(2,), sizes_fixed=(0.04, 0.14), kinds_fixed=(0, 0)), 1_000, T, 28),
    "mass_ctrl": (replace(BASE, n_choices=(2,), kinds_fixed=(0, 0)),      1_000,   T,   29),
    "pilot":     (BASE,                                                   1_000,   T,   31),   # pilot checks only
}


def generate(name, N=None, seed=None):
    spec, n0, Tn, s0 = SPLITS[name]
    N = N or n0
    rng = np.random.default_rng(s0 if seed is None else seed)
    p = sample_params(spec, N, rng)
    st = sim.simulate_batch(p["x0"], p["kind"], p["size"], p["mass"], p["n"], p["g"], p["e"], p["mu"], Tn, DT)
    p["states"] = st.astype(np.float32)
    p["energy"] = sim.energy(st, p["mass"], p["g"], p["active"]).astype(np.float32)
    pxm, pym = sim.momentum(st, p["mass"], p["active"])
    p["px"], p["py"] = pxm.astype(np.float32), pym.astype(np.float32)
    p.pop("x0")
    return p


def path(name):
    return DATA / f"{name}.npz"


def load(name):
    with np.load(path(name)) as f:
        return {k: f[k] for k in f.files}


def ensure(name):
    if not path(name).exists():
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path(name), **generate(name))
    return load(name)


class Pool:
    """A split on the GPU (positions in fp16 for large pools) with on-the-fly rendering."""

    def __init__(self, p, device="cuda", n_limit=None, half=None):
        sl = slice(0, n_limit)
        self.N = len(p["g"][sl])
        half = self.N > 20_000 if half is None else half
        t = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a)[sl], dtype=d, device=device)
        self.xy = t(p["states"][..., :2], torch.float16 if half else torch.float32)
        self.kind, self.size = t(p["kind"], torch.int64), t(p["size"])
        self.color, self.active = t(p["color"], torch.int64), t(p["active"], torch.bool)
        self.mass, self.n = t(p["mass"]), t(p["n"], torch.int64)
        self.g, self.e, self.mu = t(p["g"]), t(p["e"]), t(p["mu"])
        self.bg = material_bg(self.e, self.mu)
        self.T = self.xy.shape[1]
        self.device = device

    def window(self, idx, t0, L):
        if isinstance(t0, int):
            return self.xy[idx, t0:t0 + L].float()
        ar = t0[:, None] + torch.arange(L, device=self.device)[None]
        return self.xy[idx[:, None], ar].float()

    def render(self, idx, xy):
        return render(xy, self.kind[idx], self.size[idx], self.color[idx], self.active[idx], bg=self.bg[idx])

    def sample(self, B, L, gen):
        idx = torch.randint(0, self.N, (B,), device=self.device, generator=gen)
        t0 = torch.randint(0, self.T - L + 1, (B,), device=self.device, generator=gen)
        return idx, t0, self.window(idx, t0, L)
