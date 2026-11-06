"""Stage 1: frame encoders that define the two pixel families.

REC  - convolutional autoencoder (frame -> 64-d latent -> frame), foreground-weighted pixel MSE.
JEPA - the same encoder trained to *predict* the next frame's latent (EMA target encoder) from the
       latents of the two previous frames, with VICReg variance/covariance regularization.
       It has no decoder; nothing in its objective asks it to reproduce pixels.

Both encoders are frozen afterwards and the dynamics (stage 2) is trained on their latents, as in
World Models (Ha & Schmidhuber 2018) and V-JEPA 2-AC. The families therefore differ only in how
their representation was learned.
"""
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import ConvDec, ConvEnc, SIZES, SSEnc, fg_weight, vicreg_terms
from .world import Pool

DZ = 64


class Perception(nn.Module):
    def __init__(self, kind="rec", size="M", horizon=1, enc="conv", beta=0.0):
        super().__init__()
        c, _ = SIZES[size]
        self.kind, self.size, self.horizon, self.enc_kind, self.beta = kind, size, horizon, enc, beta
        self.dz = 80 if enc == "kp" else DZ
        self.enc = SSEnc(c) if enc == "ss" else (SSEnc(c, raw=True) if enc == "kp" else ConvEnc(c))
        if beta > 0:   # VAE: a second head for log-variance
            self.logvar = nn.Linear(self.dz, self.dz)
        self.dec = ConvDec(c, self.dz)  # REC: trained jointly. JEPA: readout only (never used in loss)
        if kind == "jepa":
            self.target = copy.deepcopy(self.enc)
            for p in self.target.parameters():
                p.requires_grad_(False)
            self.pred = nn.Sequential(nn.Linear(2 * self.dz, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                      nn.Linear(256, self.dz * horizon))

    def encode(self, x):
        """x: (..., 3, 64, 64) -> (..., DZ)."""
        sh = x.shape[:-3]
        return self.enc(x.reshape(-1, *x.shape[-3:])).view(*sh, self.dz)

    def decode(self, z):
        sh = z.shape[:-1]
        out = self.dec(z.reshape(-1, self.dz))
        return out.view(*sh, *out.shape[1:])

    @torch.no_grad()
    def ema(self, tau):
        for pt, po in zip(self.target.parameters(), self.enc.parameters()):
            pt.mul_(tau).add_(po.detach(), alpha=1 - tau)


def train_perception(kind, size, seed, steps, out, pool_name="train", batch=64, lr=1e-3, w_fg=4.0,
                     w_var=1.0, w_cov=0.04, ema=0.996, log_every=500, horizon=1, enc="conv", beta=0.0):
    from . import world
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pool = Pool(world.ensure(pool_name), "cuda")
    m = Perception(kind, size, horizon, enc, beta).cuda()
    params = [p for n, p in m.named_parameters() if p.requires_grad and (kind == "rec" or not n.startswith("dec."))]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cfg = dict(kind=kind, size=size, seed=seed, steps=steps, batch=batch, lr=lr, pool=pool_name,
               w_var=w_var, w_cov=w_cov, horizon=horizon, enc=enc, beta=beta)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(out / "log.txt", "w")
    t0 = time.time()
    for step in range(1, steps + 1):
        L = 3 if kind == "rec" else 2 + horizon
        idx, t0i, xy = pool.sample(batch, L, gen)
        x = pool.render(idx, xy)                         # (B, L, 3, 64, 64): frames t-1, t, t+1, ...
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "rec":
                z = m.encode(x)
                if beta > 0:
                    lv = m.logvar(z.float()).clamp(-10, 5)
                    zs = z.float() + torch.exp(0.5 * lv) * torch.randn_like(lv)
                    kl = 0.5 * (z.float() ** 2 + lv.exp() - 1 - lv).sum(-1).mean()
                else:
                    zs, kl = z, torch.zeros((), device=x.device)
                xh = m.decode(zs)
                rec = (fg_weight(x, w_fg) * (xh.float() - x) ** 2).mean()
                loss = rec + beta * kl
                logs = {"rec": rec.item(), "kl": kl.item()}
            else:
                z = m.encode(x[:, :2]).float()
                with torch.no_grad():
                    zt = m.target(x[:, 2:].flatten(0, 1)).float().view(x.shape[0], horizon * m.dz)
                zp = m.pred(torch.cat([z[:, 0], z[:, 1]], -1)).float()
                pred = F.mse_loss(zp, zt)
                var, cov = vicreg_terms(z.flatten(0, 1))
                loss = pred + w_var * var + w_cov * cov
                logs = {"pred": pred.item(), "var": var.item(), "cov": cov.item()}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if kind == "jepa":
            m.ema(1 - (1 - ema) * 0.5 * (1 + math.cos(math.pi * step / steps)))
        if step % log_every == 0 or step == 1:
            msg = f"step {step} loss {loss.item():.5f} " + " ".join(f"{k} {v:.5f}" for k, v in logs.items()) + \
                  f" t {time.time() - t0:.0f}s"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
    torch.save(m.state_dict(), out / "perception.pt")
    (out / "done").write_text(f"{time.time() - t0:.0f}")
    return m


def load_perception(run_dir, device="cuda"):
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    m = Perception(cfg["kind"], cfg["size"], cfg.get("horizon", 1), cfg.get("enc", "conv"), cfg.get("beta", 0.0)).to(device)
    m.load_state_dict(torch.load(Path(run_dir) / "perception.pt", map_location=device))
    return m.eval(), cfg
