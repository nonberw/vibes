"""World models: a shared recurrent core with three kinds of perception/training signal.

REC   - CNN encoder + deconv decoder, trained by pixel reconstruction of open-loop predictions.
JEPA  - CNN encoder, EMA target encoder, trained by latent prediction + VICReg regularization.
STATE - MLP encoder on ground-truth object positions (no velocities, no physics parameters).

All three run the same loop: a GRU consumes encoded context frames, then feeds back its own
predicted latents (open-loop "imagination"). `h_C`, the GRU state after the last context frame,
is the belief state that the analysis probes and steers.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .render import BG

DZ = 64
SIZES = {"S": (16, 128), "M": (32, 256), "L": (64, 512)}
KMAX = 5
OBS_DIM = KMAX * 6   # per slot: cos 2pi x, sin 2pi x, y, kind, size, active


class ConvEnc(nn.Module):
    def __init__(self, c):
        super().__init__()
        ch = [3, c, 2 * c, 4 * c, 8 * c]
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b, 4, 2, 1), nn.GroupNorm(8, b), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(8 * c * 16, DZ)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class ConvDec(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.fc = nn.Linear(DZ, 8 * c * 16)
        ch = [8 * c, 4 * c, 2 * c, c]
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.ConvTranspose2d(a, b, 4, 2, 1), nn.GroupNorm(8, b), nn.SiLU()]
        layers += [nn.ConvTranspose2d(c, 3, 4, 2, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return torch.sigmoid(self.net(self.fc(z).view(-1, 8 * self.c, 4, 4)))


class StateEnc(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(),
                                 nn.Linear(H, DZ))

    def forward(self, o):
        return self.net(o)


class StateDec(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DZ, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(),
                                 nn.Linear(H, KMAX * 3))

    def forward(self, z):
        return self.net(z).view(*z.shape[:-1], KMAX, 3)


def state_obs(st, kind, size, active):
    """Ground-truth observation for STATE: positions only. st (B, L, K, 4)."""
    B, L, K, _ = st.shape
    ang = 2 * math.pi * st[..., 0]
    f = torch.stack([torch.cos(ang), torch.sin(ang), st[..., 1],
                     kind[:, None].expand(B, L, K).float(),
                     size[:, None].expand(B, L, K).float(),
                     active[:, None].expand(B, L, K).float()], -1)
    f = f * active[:, None, :, None]
    return f.flatten(2)


def state_target(st):
    ang = 2 * math.pi * st[..., 0]
    return torch.stack([torch.cos(ang), torch.sin(ang), st[..., 1]], -1)


class WorldModel(nn.Module):
    def __init__(self, family="rec", size="M"):
        super().__init__()
        c, H = SIZES[size]
        self.family, self.size, self.H = family, size, H
        if family == "state":
            self.enc = StateEnc(H)
            self.dec = StateDec(H)
        else:
            self.enc = ConvEnc(c)
            self.dec = ConvDec(c)  # REC: trained jointly; JEPA: readout decoder fit post hoc
        self.norm = nn.LayerNorm(DZ, elementwise_affine=False)
        self.gru = nn.GRUCell(DZ, H)
        self.head = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, DZ))
        if family == "jepa":
            self.target_enc = copy.deepcopy(self.enc)
            for p in self.target_enc.parameters():
                p.requires_grad_(False)

    # ----- perception -----
    def encode(self, x, target=False):
        """x: (B, L, ...) frames or observations -> (B, L, DZ) normalized latents."""
        B, L = x.shape[:2]
        enc = self.target_enc if target else self.enc
        z = enc(x.flatten(0, 1))
        return self.norm(z).view(B, L, DZ)

    def decode(self, z):
        B, L = z.shape[:2]
        out = self.dec(z.flatten(0, 1))
        return out.view(B, L, *out.shape[1:])

    # ----- dynamics -----
    def observe(self, z_ctx, h=None):
        """Teacher-forced pass over context latents. Returns one-step predictions and h_C."""
        B, C, _ = z_ctx.shape
        if h is None:
            h = z_ctx.new_zeros(B, self.H)
        preds = []
        for t in range(C):
            h = self.gru(z_ctx[:, t], h)
            preds.append(self.norm(self.head(h)))
        return torch.stack(preds, 1), h

    def imagine(self, h, R, first=None):
        """Open-loop rollout of R latents starting from belief state h (= h_C).

        The first predicted latent is head(h); each next one feeds the previous prediction back.
        Returns (B, R, DZ) predictions and the list of hidden states.
        """
        z = self.norm(self.head(h)) if first is None else first
        out, hs = [z], [h]
        for _ in range(R - 1):
            h = self.gru(z, h)
            z = self.norm(self.head(h))
            out.append(z)
            hs.append(h)
        return torch.stack(out, 1), torch.stack(hs, 1)

    def rollout(self, z_ctx, R):
        preds_ctx, h = self.observe(z_ctx)
        zr, hs = self.imagine(h, R)
        return preds_ctx, zr, h, hs

    @torch.no_grad()
    def ema_update(self, tau):
        for pt, po in zip(self.target_enc.parameters(), self.enc.parameters()):
            pt.mul_(tau).add_(po.detach(), alpha=1 - tau)


def vicreg_terms(z):
    """z: (N, D). Variance hinge (target std 1) and off-diagonal covariance penalty."""
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    var = F.relu(1.0 - std).mean()
    cov = (z.T @ z) / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var, (off ** 2).sum() / z.shape[1]


def fg_weight(x, w_fg):
    bg = torch.as_tensor(BG, device=x.device).view(1, 1, 3, 1, 1)
    fg = ((x - bg).abs().sum(2, keepdim=True) > 0.05).float()
    return 1.0 + w_fg * fg


def n_params(m):
    return sum(p.numel() for n, p in m.named_parameters()
               if p.requires_grad and not n.startswith("target_enc"))
