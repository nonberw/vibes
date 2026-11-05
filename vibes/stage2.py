"""Latent caches, position readouts and stage-2 training for the v2 chunk world models."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import world
from .dynamics import ChunkWM, state_features, state_positions
from .perception import load_perception
from .render import PALETTE
from .world import C, R, Pool

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "v2" / "latents"


# ---------------------------------------------------------------------------------------------
# latents

@torch.no_grad()
def encode_frames(perc, pool, idx, t0, L, chunk=32):
    """Latents of frames t0..t0+L-1 of episodes idx: (B, L, DZ) float32."""
    out = []
    for i in range(0, len(idx), chunk):
        ii = idx[i:i + chunk]
        tt = t0[i:i + chunk] if torch.is_tensor(t0) else t0
        xy = pool.window(ii, tt, L)
        x = pool.render(ii, xy)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = perc.encode(x)
        out.append(z.float())
    return torch.cat(out)


@torch.no_grad()
def cache_latents(perc_dir, split, n_limit=None):
    """Encode every frame of a split with a frozen perception model; cached as fp16 .npy."""
    perc_dir = Path(perc_dir)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{perc_dir.name}__{split}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    perc, _ = load_perception(perc_dir)
    pool = Pool(world.ensure(split), "cuda", n_limit=n_limit)
    zs = np.zeros((pool.N, pool.T, 64), np.float16)
    idx_all = torch.arange(pool.N, device="cuda")
    for i in range(0, pool.N, 512):
        idx = idx_all[i:i + 512]
        zs[i:i + len(idx)] = encode_frames(perc, pool, idx, 0, pool.T).cpu().numpy().astype(np.float16)
    np.save(path, zs)
    return np.load(path, mmap_mode="r")


class Source:
    """What the dynamics model sees for a split: cached latents (REC/JEPA) or features (STATE)."""

    def __init__(self, family, split, perc_dir=None, n_limit=None, device="cuda"):
        self.family, self.split = family, split
        self.p = world.ensure(split)
        self.pool = Pool(self.p, device, n_limit=n_limit)
        self.N, self.T = self.pool.N, self.pool.T
        if family == "state":
            self.z = None
            self.din = 32
        else:
            z = cache_latents(perc_dir, split)
            self.z = torch.from_numpy(np.ascontiguousarray(z[:self.N]))      # CPU fp16
            if device == "cuda":
                self.z = self.z.pin_memory()
            self.din = 64
        self.device = device

    def window(self, idx, t0, L):
        if self.family == "state":
            xy = self.pool.window(idx, t0, L)
            P = self.pool
            return state_features(xy, P.kind[idx], P.size[idx], P.active[idx], P.e[idx], P.mu[idx])
        i = idx.cpu()
        if torch.is_tensor(t0):
            ar = t0.cpu()[:, None] + torch.arange(L)[None]
            w = self.z[i[:, None], ar]
        else:
            w = self.z[i, t0:t0 + L]
        return w.to(self.device, non_blocking=True).float()


# ---------------------------------------------------------------------------------------------
# position readout from latents (REC / JEPA): colour-indexed probe fit on real val latents

class PosProbe(nn.Module):
    """latent -> for each of the 8 palette colours: (cos 2pi x, sin 2pi x, y)."""

    def __init__(self, din=64, width=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, len(PALETTE) * 3))

    def forward(self, z):
        return self.net(z).view(*z.shape[:-1], len(PALETTE), 3)


def colour_targets(xy, color, active):
    """xy (B, L, K, 2) -> per-colour targets (B, L, 8, 3) and mask (B, 8)."""
    B, L, K, _ = xy.shape
    tgt = torch.zeros(B, L, len(PALETTE), 3, device=xy.device)
    mask = torch.zeros(B, len(PALETTE), dtype=torch.bool, device=xy.device)
    ang = 2 * math.pi * xy[..., 0]
    vals = torch.stack([torch.cos(ang), torch.sin(ang), xy[..., 1]], -1)       # (B, L, K, 3)
    for k in range(K):
        a = active[:, k]
        c = color[:, k]
        bi = torch.where(a)[0]
        tgt[bi, :, c[bi]] = vals[bi, :, k]
        mask[bi, c[bi]] = True
    return tgt, mask


def fit_pos_probe(src, steps=6000, B=256, seed=0):
    """Fit on all frames of `src` (a val Source of a latent family)."""
    torch.manual_seed(seed)
    probe = PosProbe(src.din).cuda()
    opt = torch.optim.AdamW(probe.parameters(), 1e-3, weight_decay=1e-4)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    P = src.pool
    for step in range(1, steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * step / steps))
        idx = torch.randint(0, src.N, (B,), device="cuda", generator=gen)
        t0 = torch.randint(0, src.T, (B,), device="cuda", generator=gen)
        z = src.window(idx, t0, 1)
        xy = P.window(idx, t0, 1)
        tgt, mask = colour_targets(xy, P.color[idx], P.active[idx])
        out = probe(src_norm(z))
        loss = (((out - tgt) ** 2).sum(-1) * mask[:, None]).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return probe.eval()


def src_norm(z):
    return z      # latents are already layer-normalised scale; kept as a hook


def probe_positions(probe, z, color, active):
    """z (B, L, 64) -> positions (B, L, K, 2) for the episode's objects (by colour)."""
    out = probe(z)                                                    # (B, L, 8, 3)
    B, L = z.shape[:2]
    K = color.shape[1]
    sel = out[torch.arange(B, device=z.device)[:, None], :, color]    # (B, K, L, 3)
    sel = sel.permute(0, 2, 1, 3)
    x = (torch.atan2(sel[..., 1], sel[..., 0]) / (2 * math.pi)) % 1.0
    return torch.stack([x, sel[..., 2]], -1)


# ---------------------------------------------------------------------------------------------
# stage-2 training

def train_dynamics(family, size, seed, steps, out, perc_dir=None, split="train", n_train=None, core="gru",
                   batch=128, lr=1e-3, log_every=1000):
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = Source(family, split, perc_dir, n_limit=n_train)
    m = ChunkWM(src.din, size, core).cuda()
    # standardization statistics from a sample of training windows
    gen = torch.Generator(device="cuda").manual_seed(seed)
    idx = torch.randint(0, src.N, (2048,), device="cuda", generator=gen)
    zz = src.window(idx, 0, src.T).reshape(-1, src.din)
    m.mu.copy_(zz.mean(0))
    m.sd.copy_(zz.std(0).clamp_min(1e-3))
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-4)
    cfg = dict(family=family, size=size, seed=seed, steps=steps, perc=str(perc_dir) if perc_dir else None,
               split=split, n_train=src.N, core=core, batch=batch, lr=lr, din=src.din)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(out / "log.txt", "w")
    t0 = time.time()
    for step in range(1, steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = lr * min(1, step / 500) * 0.5 * (1 + math.cos(math.pi * step / steps))
        idx = torch.randint(0, src.N, (batch,), device="cuda", generator=gen)
        t0i = torch.randint(0, src.T - (C + R) + 1, (batch,), device="cuda", generator=gen)
        zn = m.norm(src.window(idx, t0i, C + R))
        pred, _ = m(zn[:, :C])
        loss = ((pred - zn[:, C:]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            msg = f"step {step} loss {loss.item():.5f} t {time.time() - t0:.0f}s"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
    torch.save(m.state_dict(), out / "dynamics.pt")
    (out / "done").write_text(f"{time.time() - t0:.0f}")
    return m


def load_dynamics(run_dir, device="cuda"):
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    m = ChunkWM(cfg["din"], cfg["size"], cfg["core"]).to(device)
    m.load_state_dict(torch.load(Path(run_dir) / "dynamics.pt", map_location=device))
    return m.eval(), cfg


# ---------------------------------------------------------------------------------------------
# position readout v2: a post-hoc pixel decoder fit identically for both latent families
# (on frozen latents of real val frames), followed by colour centroids.

def fit_readout_decoder(perc_dir, steps=8000, B=64, seed=0):
    """Train (or load) the readout decoder for a frozen perception model."""
    from .models import ConvDec, SIZES, fg_weight
    perc_dir = Path(perc_dir)
    path = perc_dir / "readout_dec.pt"
    perc, cfg = load_perception(perc_dir)
    dec = ConvDec(SIZES[cfg["size"]][0]).cuda()
    if path.exists():
        dec.load_state_dict(torch.load(path))
        return dec.eval()
    torch.manual_seed(seed)
    pool = Pool(world.ensure("val"), "cuda")
    opt = torch.optim.AdamW(dec.parameters(), 1e-3, weight_decay=0.0)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for step in range(1, steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = 1e-3 * min(1, step / 300) * 0.5 * (1 + math.cos(math.pi * step / steps))
        idx, t0, xy = pool.sample(B, 1, gen)
        x = pool.render(idx, xy)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = perc.encode(x).float()
        xh = dec(z.flatten(0, 1)).view_as(x)
        loss = (fg_weight(x, 4.0) * (xh - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(dec.state_dict(), path)
    return dec.eval()


@torch.no_grad()
def decoder_positions(dec, z, color, active, bg, chunk=16):
    """z (B, L, 64) raw latents -> object positions (B, L, K, 2) via decoded frames."""
    from .readout import centroids
    out = []
    for i in range(0, z.shape[0], chunk):
        zz = z[i:i + chunk]
        x = dec(zz.flatten(0, 1)).view(zz.shape[0], zz.shape[1], 3, 64, 64).float().clamp(0, 1)
        pos, _ = centroids(x, color[i:i + chunk], active[i:i + chunk], bg=bg[i:i + chunk])
        out.append(pos)
    return torch.cat(out)
