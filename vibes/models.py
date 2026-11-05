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


class SSEnc(nn.Module):
    """CNN with a spatial-softmax keypoint head (Levine et al. 2016; Finn et al. 2016).

    Features at 32x32 (two stride-1 blocks after one stride-2 block keep resolution); each of
    `n_kp` channels becomes a soft keypoint: E[cos 2pi x], E[sin 2pi x] (periodic x), E[y], plus a
    presence logit (max activation). Keypoints are concatenated with a pooled feature vector and
    mapped to the latent. Sub-pixel position precision comes for free.
    """

    def __init__(self, c, n_kp=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, c, 4, 2, 1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, 2 * c, 3, 1, 1), nn.GroupNorm(8, 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, 2 * c, 3, 1, 1), nn.GroupNorm(8, 2 * c), nn.SiLU())
        self.kp = nn.Conv2d(2 * c, n_kp, 1)
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.pool = nn.Sequential(nn.Conv2d(2 * c, 4 * c, 4, 4), nn.SiLU())  # 32 -> 8
        self.fc = nn.Linear(n_kp * 4 + 4 * c * 64, DZ)
        self.n_kp = n_kp

    def forward(self, x):
        B, _, H, W = x.shape
        ys = 1.0 - (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) / H
        xs = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) / W
        ang = 2 * math.pi * xs
        coords = torch.stack([torch.cos(ang).view(1, W).expand(H, W),
                              torch.sin(ang).view(1, W).expand(H, W)], 0)
        f = self.net(torch.cat([x, coords.unsqueeze(0).expand(B, 2, H, W)], 1))
        k = self.kp(f)                                          # (B, n_kp, h, w)
        h, w = k.shape[-2:]
        p = torch.softmax((k * self.temp).flatten(2), -1).view(B, self.n_kp, h, w)
        yy = 1.0 - (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) / h
        xa = 2 * math.pi * (torch.arange(w, device=x.device, dtype=x.dtype) + 0.5) / w
        ey = (p.sum(3) * yy).sum(2)
        ec = (p.sum(2) * torch.cos(xa)).sum(2)
        es = (p.sum(2) * torch.sin(xa)).sum(2)
        pres = k.flatten(2).amax(-1)
        kp = torch.cat([ec, es, ey, pres], 1)
        return self.fc(torch.cat([kp, self.pool(f).flatten(1)], 1))


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


class LRUCell(nn.Module):
    """Diagonal linear recurrence h_t = a * h_{t-1} + B z_t (a in (0, 1), learned per unit).

    The belief update is linear in the (encoded) inputs; all nonlinearity lives in the encoder and
    in the prediction head that closes the open-loop feedback.
    """

    def __init__(self, dz, H):
        super().__init__()
        self.B = nn.Linear(dz, H)
        r = torch.empty(H).uniform_(0.5, 0.995)
        self.logit_a = nn.Parameter(torch.log(r / (1 - r)))

    def forward(self, z, h):
        return torch.sigmoid(self.logit_a) * h + self.B(z)


class WorldModel(nn.Module):
    def __init__(self, family="rec", size="M", core="gru", enc="conv"):
        super().__init__()
        c, H = SIZES[size]
        self.family, self.size, self.H, self.core = family, size, H, core
        if family == "state":
            self.enc = StateEnc(H)
            self.dec = StateDec(H)
        else:
            self.enc = SSEnc(c) if enc == "ss" else ConvEnc(c)
            self.dec = ConvDec(c)  # REC: trained jointly; JEPA: readout decoder fit post hoc
        self.norm = nn.LayerNorm(DZ, elementwise_affine=False)
        self.gru = nn.GRUCell(DZ, H) if core == "gru" else LRUCell(DZ, H)
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
