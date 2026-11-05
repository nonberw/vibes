"""Physics readouts from (decoded or real) frames, and the effective-gravity estimator."""
import math

import numpy as np
import torch

from .render import BG, PALETTE, RES


def centroids(frames, color_id, active, temp=0.02, chunk=8, bg=None):
    """Chunked over the batch to bound GPU memory (see _centroids). bg: optional (B, 3)."""
    outs, ms = [], []
    for i in range(0, frames.shape[0], chunk):
        p, m = _centroids(frames[i:i + chunk], color_id[i:i + chunk], active[i:i + chunk], temp,
                          None if bg is None else bg[i:i + chunk])
        outs.append(p)
        ms.append(m)
    return torch.cat(outs), torch.cat(ms)


def _centroids(frames, color_id, active, temp=0.02, bg_col=None):
    """Soft color-assignment centroids.

    frames: (B, T, 3, H, W) in [0, 1]; color_id, active: (B, K).
    Each pixel is softly assigned to the background or one of the active objects' colors
    (softmax over negative squared RGB distance / temp). Returns positions (B, T, K, 2) in box
    units (x via circular mean, periodic) and the total mass of each assignment (B, T, K).
    """
    dev = frames.device
    B, T, _, H, W = frames.shape
    K = color_id.shape[1]
    pal = torch.as_tensor(PALETTE, device=dev)[color_id]                  # (B, K, 3)
    bg = (torch.as_tensor(BG, device=dev).view(1, 1, 3).expand(B, 1, 3) if bg_col is None
          else bg_col.float().view(B, 1, 3))
    cols = torch.cat([bg, pal], 1)                                          # (B, K+1, 3)
    f = frames.permute(0, 1, 3, 4, 2)                                       # (B, T, H, W, 3)
    d2 = ((f[..., None, :] - cols[:, None, None, None]) ** 2).sum(-1)       # (B, T, H, W, K+1)
    mask = torch.cat([torch.ones(B, 1, device=dev, dtype=torch.bool), active.bool()], 1)
    logits = -d2 / temp
    logits = logits.masked_fill(~mask[:, None, None, None], -1e9)
    resp = torch.softmax(logits, -1)[..., 1:]                               # (B, T, H, W, K)
    px = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) / W
    py = 1.0 - (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) / H
    m = resp.sum((2, 3)) + 1e-8
    ang = 2 * math.pi * px
    cs = (resp * torch.cos(ang).view(1, 1, 1, W, 1)).sum((2, 3))
    sn = (resp * torch.sin(ang).view(1, 1, 1, W, 1)).sum((2, 3))
    x = (torch.atan2(sn, cs) / (2 * math.pi)) % 1.0
    y = (resp * py.view(1, 1, H, 1, 1)).sum((2, 3)) / m
    return torch.stack([x, y], -1), m


def unwrap_x(x):
    """Unwrap periodic x tracks along time (numpy, last axis = time)."""
    d = np.diff(x, axis=-1)
    d = d - np.round(d)
    return np.concatenate([x[..., :1], x[..., :1] + np.cumsum(d, -1)], -1)


def g_eff_episode(pos, size, active, dt, W=7, margin=0.03, max_rms=1.5 / 64):
    """Effective gravity of one episode's predicted tracks.

    pos: (T, K, 2) readout positions; size, active: (K,). Uses every W-frame window in which an
    object stays in free flight (>= size+margin from floor/ceiling, not near other objects) and
    returns the median of -2 * fitted curvature, or nan if no window qualifies.
    """
    T, K, _ = pos.shape
    t = np.arange(W) * dt
    A = np.stack([np.ones(W), t, t ** 2], 1)
    pinv = np.linalg.pinv(A)
    ests = []
    xs = unwrap_x(pos[:, :, 0].T).T                       # (T, K) unwrapped x

    def step_len(a):
        """Largest displacement to a neighbouring frame (contacts can happen between frames)."""
        d = np.abs(np.diff(a, axis=0))
        out = np.zeros_like(a)
        out[1:] = d
        out[:-1] = np.maximum(out[:-1], d)
        return out

    for k in range(K):
        if not active[k]:
            continue
        y = pos[:, k, 1]
        sy = step_len(y)
        free = (y - size[k] - sy > margin) & (y + size[k] + sy < 1 - margin)
        for j in range(K):
            if j == k or not active[j]:
                continue
            dx = pos[:, k, 0] - pos[:, j, 0]
            dx = dx - np.round(dx)
            d = np.hypot(dx, y - pos[:, j, 1])
            rel = np.hypot(step_len(xs[:, k] - xs[:, j]), step_len(y - pos[:, j, 1]))
            free &= d - rel > 1.5 * (size[k] + size[j]) + margin
        for s in range(T - W + 1):
            if free[s:s + W].all():
                ys = y[s:s + W]
                c = pinv @ ys
                if np.sqrt(np.mean((A @ c - ys) ** 2)) > max_rms:
                    continue  # not a parabola: an undetected contact inside the window
                ests.append(-2 * c[2])
    return float(np.median(ests)) if ests else float("nan")


def g_eff_batch(pos, size, active, dt, **kw):
    return np.array([g_eff_episode(pos[b], size[b], active[b], dt, **kw) for b in range(len(pos))])


def ballistic_mask(true_y, g, dt, pad=1, tol=2e-5):
    """True where an object is in exact free flight on the reference trajectory.

    true_y: (T, K); frame t counts as ballistic if the second differences centred on t-pad..t+pad
    all equal -g dt^2 (no contact anywhere near t). Ends of the sequence are False.
    """
    T, K = true_y.shape
    dd = np.diff(true_y, 2, axis=0) + g * dt * dt           # (T-2, K), centred on 1..T-2
    ok = np.zeros((T, K), bool)
    ok[1:-1] = np.abs(dd) < tol
    out = ok.copy()
    for s in range(1, pad + 1):
        out[s:] &= ok[:-s]
        out[:-s] &= ok[s:]
        out[:s] = False
        out[-s:] = False
    return out


def _weighted_median(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def g_eff_ref_episode(pos, ref_y, g_ref, active, dt, W=5, max_start=None):
    """Effective gravity of predicted tracks, measured only where the *reference* (true or
    counterfactual) trajectory is in free flight.

    pos: (T, K, 2) predictions aligned with ref_y (T, K). For every object, each maximal run of
    reference free-flight frames of length >= W that starts before `max_start` (default: any) gets
    a quadratic fit over the whole run. Runs are combined by a weighted median whose weights are
    the curvature precisions (sum of squared t^2 residuals after removing 1 and t). Returns nan if
    there is no qualifying run."""
    T, K = ref_y.shape
    free = ballistic_mask(ref_y, g_ref, dt)
    last = T if max_start is None else max_start
    ests, wts = [], []
    for k in range(K):
        if not active[k]:
            continue
        f = free[:, k]
        s = 0
        while s < T:
            if not f[s]:
                s += 1
                continue
            e = s
            while e < T and f[e]:
                e += 1
            if e - s >= W and s < last:
                t = np.arange(e - s) * dt
                A = np.stack([np.ones_like(t), t, t ** 2], 1)
                c = np.linalg.lstsq(A, pos[s:e, k, 1].astype(np.float64), rcond=None)[0]
                q = t ** 2 - A[:, :2] @ np.linalg.lstsq(A[:, :2], t ** 2, rcond=None)[0]
                ests.append(-2 * c[2])
                wts.append((q ** 2).sum())
            s = e
    if not ests:
        return float("nan")
    return _weighted_median(np.array(ests), np.array(wts))


def g_eff_ref_batch(pos, ref_states, g_ref, active, dt, **kw):
    """pos (N, T, K, 2), ref_states (N, T, K, >=2) aligned frame by frame, g_ref (N,)."""
    return np.array([g_eff_ref_episode(pos[b], ref_states[b, :, :, 1], g_ref[b], active[b], dt, **kw)
                     for b in range(len(pos))])
