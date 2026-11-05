"""Anti-aliased 64x64 RGB rendering of VIBES states, in torch (GPU) so frames never hit disk.

Pixel (row 0) is the top of the box (y = 1). x is periodic: distances use the minimum image, so
an object crossing x = 1 appears split across both edges.
"""
import numpy as np
import torch

RES = 64
PALETTE = np.array([
    [0.95, 0.30, 0.25],  # red
    [0.25, 0.75, 0.30],  # green
    [0.25, 0.45, 0.95],  # blue
    [0.95, 0.80, 0.20],  # yellow
    [0.85, 0.35, 0.85],  # magenta
    [0.20, 0.85, 0.85],  # cyan
    [0.97, 0.55, 0.15],  # orange
    [0.92, 0.92, 0.92],  # white
], dtype=np.float32)
BG = np.array([0.08, 0.08, 0.10], dtype=np.float32)


def material_bg(e, mu):
    """v2 worlds: the scene's material is visible as a dark background tint.

    Red channel encodes restitution e in [0.6, 1], green encodes surface friction mu in [0, 0.4].
    Gravity is not visible anywhere and must be inferred from motion.
    e, mu: tensors (B,). Returns (B, 3).
    """
    r = 0.04 + 0.14 * ((e - 0.6) / 0.4).clamp(0, 1)
    g = 0.04 + 0.14 * (mu / 0.4).clamp(0, 1)
    b = torch.full_like(r, 0.10)
    return torch.stack([r, g, b], -1)


def render(xy, kind, size, color_id, active, res=RES, bg=None):
    """xy: (B, T, K, 2) float tensor; kind, size, color_id, active: (B, K); bg: optional (B, 3)
    per-episode background color (v2 material tint).

    Returns (B, T, 3, res, res) float32 in [0, 1]. Later objects are painted over earlier ones.
    """
    dev = xy.device
    B, T, K, _ = xy.shape
    px = (torch.arange(res, device=dev, dtype=torch.float32) + 0.5) / res
    gx = px.view(1, 1, 1, res)                 # x of column
    gy = (1.0 - px).view(1, 1, res, 1)         # y of row (row 0 at top)
    pal = torch.as_tensor(PALETTE, device=dev)
    if bg is None:
        img = torch.as_tensor(BG, device=dev).view(1, 1, 3, 1, 1).expand(B, T, 3, res, res).clone()
    else:
        img = bg.float().view(B, 1, 3, 1, 1).expand(B, T, 3, res, res).clone()
    pix = 1.0 / res
    for k in range(K):
        cx = xy[:, :, k, 0].view(B, T, 1, 1)
        cy = xy[:, :, k, 1].view(B, T, 1, 1)
        dx = gx - cx
        dx = dx - torch.floor(dx + 0.5)
        dy = gy - cy
        s = size[:, k].view(B, 1, 1, 1).float()
        is_ball = (kind[:, k] == 0).view(B, 1, 1, 1)
        sd_ball = torch.sqrt(dx * dx + dy * dy) - s
        sd_box = torch.maximum(dx.abs(), dy.abs()) - s
        sd = torch.where(is_ball, sd_ball, sd_box)
        cov = (0.5 - sd / pix).clamp(0.0, 1.0) * active[:, k].view(B, 1, 1, 1).float()
        col = pal[color_id[:, k].long()].view(B, 1, 3, 1, 1)
        cov = cov.unsqueeze(2)
        img = img * (1 - cov) + col * cov
    return img


def render_np(states, kind, size, color_id, active, res=RES):
    """Numpy convenience wrapper. states (B, T, K, >=2). Returns uint8 (B, T, res, res, 3)."""
    with torch.no_grad():
        xy = torch.as_tensor(np.asarray(states)[..., :2], dtype=torch.float32)
        out = render(xy, torch.as_tensor(kind), torch.as_tensor(size), torch.as_tensor(color_id),
                     torch.as_tensor(active), res)
    return (out.permute(0, 1, 3, 4, 2).numpy() * 255).round().astype(np.uint8)
