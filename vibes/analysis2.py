"""v2 analysis: a uniform interface over the three families of chunk world models.

A `WM` takes *trajectories* (positions of the episode's objects over context frames), renders and
encodes them (REC/JEPA) or featurizes them (STATE), forms the belief h_C, and decodes predicted
object positions for the next R frames. Everything downstream (probes, steering, inversion,
OOD sweeps) works on this interface, so the families are measured identically.
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

from . import world
from .dynamics import state_features, state_positions
from .perception import load_perception
from .readout import g_eff_ref_batch
from .render import material_bg, render
from .stage2 import Source, fit_pos_probe, load_dynamics, probe_positions
from .world import C, DT, R

ROOT = Path(__file__).resolve().parents[1]


class Meta:
    """Static per-episode properties (numpy arrays, length B) + torch views on the GPU."""

    def __init__(self, p, idx=None):
        idx = np.arange(len(p["g"])) if idx is None else np.asarray(idx)
        self.idx = idx
        for k in ("kind", "size", "color", "active", "mass", "n", "g", "e", "mu"):
            setattr(self, k, np.asarray(p[k])[idx])
        self.B = len(idx)

    def t(self, k, dtype=torch.float32):
        return torch.as_tensor(getattr(self, k), dtype=dtype, device="cuda")

    def sub(self, sl):
        m = Meta.__new__(Meta)
        for k in ("kind", "size", "color", "active", "mass", "n", "g", "e", "mu", "idx"):
            setattr(m, k, getattr(self, k)[sl])
        m.B = len(m.g)
        return m


class WM:
    def __init__(self, run_dir, probe_steps=6000):
        self.dyn, self.cfg = load_dynamics(run_dir)
        self.family = self.cfg["family"]
        self.run_dir = Path(run_dir)
        self.perc, self.readout = None, None
        if self.family != "state":
            from .stage2 import fit_readout_decoder
            self.perc, _ = load_perception(self.cfg["perc"])
            self.readout = fit_readout_decoder(self.cfg["perc"])
            for p in self.readout.parameters():
                p.requires_grad_(False)
        for p in self.dyn.parameters():
            p.requires_grad_(False)

    # ----- inputs -----
    @torch.no_grad()
    def latents(self, xy, meta):
        """xy: (B, L, K, 2) tensor of object positions -> raw latents (B, L, din)."""
        if self.family == "state":
            return state_features(xy, meta.t("kind", torch.int64), meta.t("size"), meta.t("active", torch.bool),
                                  meta.t("e"), meta.t("mu"))
        x = render(xy, meta.t("kind", torch.int64), meta.t("size"), meta.t("color", torch.int64),
                   meta.t("active", torch.bool), bg=material_bg(meta.t("e"), meta.t("mu")))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.perc.encode(x).float()

    # ----- model -----
    def belief(self, zn):
        return self.dyn.belief(zn)

    def predict_latents(self, h, zlast, ks=None):
        return self.dyn.decode(h, zlast, ks)

    def positions(self, zn_pred, meta):
        """standardized predicted latents (B, L, din) -> object positions (B, L, K, 2)."""
        z = zn_pred * self.dyn.sd + self.dyn.mu
        if self.family == "state":
            return state_positions(z)
        from .stage2 import decoder_positions
        return decoder_positions(self.readout, z, meta.t("color", torch.int64), meta.t("active", torch.bool),
                                 material_bg(meta.t("e"), meta.t("mu")))

    # ----- convenience -----
    @torch.no_grad()
    def run(self, ctx_xy, meta, chunk=256):
        """ctx_xy: (B, C, K, 2) numpy/tensor. Returns dict with h (B,H), zn_ctx, positions (B,R,K,2)."""
        hs, zs, ps = [], [], []
        for i in range(0, meta.B, chunk):
            m = meta.sub(slice(i, i + chunk))
            xy = torch.as_tensor(np.asarray(ctx_xy[i:i + chunk]), dtype=torch.float32, device="cuda")
            zn = self.dyn.norm(self.latents(xy, m))
            h = self.belief(zn)
            pr = self.predict_latents(h, zn[:, -1])
            hs.append(h)
            zs.append(zn)
            ps.append(self.positions(pr, m))
        return dict(h=torch.cat(hs), zn=torch.cat(zs), pos=torch.cat(ps))

    @torch.no_grad()
    def from_belief(self, h, zlast, meta, chunk=256):
        ps = []
        for i in range(0, meta.B, chunk):
            m = meta.sub(slice(i, i + chunk))
            pr = self.predict_latents(h[i:i + chunk], zlast[i:i + chunk])
            ps.append(self.positions(pr, m))
        return torch.cat(ps)


def pos_err_px(pos, ref, active):
    """Mean periodic-x Euclidean error in pixels per episode. pos/ref (B, L, K, 2) numpy."""
    dx = pos[..., 0] - ref[..., 0]
    dx = dx - np.round(dx)
    d = np.hypot(dx, pos[..., 1] - ref[..., 1])
    a = np.broadcast_to(active[:, None, :].astype(float), d.shape)
    return (d * a).sum((1, 2)) / a.sum((1, 2)) * 64


def g_eff_pred(pos, ref_states, g_ref, active, max_start=8):
    """Effective gravity of predicted positions where the reference trajectory is in free flight."""
    return g_eff_ref_batch(pos, ref_states, g_ref, active, DT, max_start=max_start)


def competence(wm, split="val", n=None):
    p = world.load(split)
    meta = Meta(p, None if n is None else np.arange(n))
    st = p["states"][meta.idx]
    out = wm.run(st[:, :C, :, :2], meta)
    pos = out["pos"].cpu().numpy()
    fut = st[:, C:C + R]
    g = meta.g
    ge = g_eff_pred(pos, fut, g, meta.active)
    ok = np.isfinite(ge)
    H = out["h"].cpu().numpy()
    pr = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), H, g, cv=2)
    r2 = 1 - ((pr - g) ** 2).sum() / ((g - g.mean()) ** 2).sum()
    return dict(r2_g=float(r2), geff_slope=float(np.polyfit(g[ok], ge[ok], 1)[0]),
                geff_corr=float(np.corrcoef(g[ok], ge[ok])[0, 1]), geff_mean=float(np.nanmean(ge)),
                geff_valid=float(ok.mean()), err8_px=float(pos_err_px(pos[:, :8], fut[:, :8], meta.active).mean()),
                errR_px=float(pos_err_px(pos, fut, meta.active).mean()))
