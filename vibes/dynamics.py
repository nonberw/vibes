"""Stage 2: chunk world models on frozen per-frame latents (or ground-truth features for STATE).

    belief   h_C = Core([z_t, 10 (z_t - z_{t-1})] for t in context)      (GRU or diagonal LRU)
    predict  z_hat_{C+k} = z_{C-1} + Dec(h_C, k),   k = 0..R-1             (all horizons at once)

Chunk prediction is how video world models such as diffusion video generators and V-JEPA
predictors emit the future. It avoids compounding open-loop errors during training, which made
autoregressive GRU models in the v1 pilot hedge toward the mean.

The belief h_C is the object we probe, steer and invert; Dec is "execution".
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .models import KMAX, LRUCell, state_obs
from .world import C, R

DYN_SIZES = {"S": (128, 256), "M": (256, 512), "L": (512, 1024)}   # (belief H, decoder width)


class ChunkWM(nn.Module):
    def __init__(self, din, size="M", core="gru"):
        super().__init__()
        H, W = DYN_SIZES[size]
        self.din, self.H, self.size, self.core = din, H, size, core
        self.register_buffer("mu", torch.zeros(din))
        self.register_buffer("sd", torch.ones(din))
        if core == "gru":
            self.cell = nn.GRUCell(2 * din, H)
        else:
            self.inp = nn.Sequential(nn.Linear(2 * din, H), nn.SiLU())
            self.cell = LRUCell(H, H)
        self.k = nn.Embedding(R, 64)
        self.dec = nn.Sequential(nn.Linear(H + 64, W), nn.SiLU(), nn.Linear(W, W), nn.SiLU(), nn.Linear(W, din))

    def norm(self, z):
        return (z - self.mu) / self.sd

    def belief(self, zn):
        """zn: (B, C, din) standardized context latents -> h_C (B, H)."""
        d = torch.cat([torch.zeros_like(zn[:, :1]), zn[:, 1:] - zn[:, :-1]], 1)
        f = torch.cat([zn, 10 * d], -1)
        h = zn.new_zeros(zn.shape[0], self.H)
        for t in range(zn.shape[1]):
            x = f[:, t] if self.core == "gru" else self.inp(f[:, t])
            h = self.cell(x, h)
        return h

    def decode(self, h, zlast, ks=None):
        """Standardized future latents for horizons ks (default all R): (B, len(ks), din)."""
        ks = torch.arange(R, device=h.device) if ks is None else ks
        Bn = h.shape[0]
        e = self.k(ks)[None].expand(Bn, len(ks), 64)
        return zlast[:, None] + self.dec(torch.cat([h[:, None].expand(Bn, len(ks), self.H), e], -1))

    def forward(self, zn_ctx):
        h = self.belief(zn_ctx)
        return self.decode(h, zn_ctx[:, -1]), h


# ---------------------------------------------------------------------------------------------
# STATE features: ground-truth positions (no velocities, no g) + visible material (e, mu)

def state_features(xy, kind, size, active, e, mu):
    """xy: (B, L, K, 2) -> (B, L, 32): per slot cos/sin 2pi x, y, kind, size, active; plus e, mu."""
    st = torch.zeros(*xy.shape[:-1], 4, device=xy.device)
    st[..., :2] = xy
    o = state_obs(st, kind, size, active)
    em = torch.stack([e, mu], -1)[:, None].expand(-1, o.shape[1], 2)
    return torch.cat([o, em], -1)


def state_positions(f):
    """Inverse of state_features for positions: (..., 32) -> (..., KMAX, 2)."""
    s = f[..., :KMAX * 6].reshape(*f.shape[:-1], KMAX, 6)
    x = (torch.atan2(s[..., 1], s[..., 0]) / (2 * math.pi)) % 1.0
    return torch.stack([x, s[..., 2]], -1)
