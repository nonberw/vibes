"""Nested inversion: locate an OOD failure in perception, belief formation, or execution.

For a frozen world model and an episode with true future tau (frames C..C+R-1):

  natural : encoder(context frames) -> GRU over context -> h_C -> open-loop rollout
  inv_z   : optimize the C context *latents* (free vectors, then the model's own layer norm),
            run the frozen GRU over them -> h_C -> rollout.     Bypasses perception.
  inv_h   : optimize h_C directly -> rollout.                   Bypasses perception and belief
                                                                formation; tests execution only.

The inversion objective is the family's own training loss on the first F future frames only
(the *fit window*). Every metric is computed on the held-out continuation (frames F..R-1), so a
belief that merely memorizes the fit window does not count: the frozen dynamics must keep
executing the right physics on its own.

Nesting gives err_nat >= err_inv_z >= err_inv_h (up to optimization error), and the gaps
  perception gap = err_nat - err_inv_z,   belief gap = err_inv_z - err_inv_h,   execution floor = err_inv_h
attribute the failure to a stage.
"""
import torch

from .models import fg_weight, state_obs, state_target
from .render import render


def _targets(model, st, kind, size, color, active):
    """Family-specific targets for future frames. st: (B, F, K, 4) tensor."""
    if model.family == "state":
        return state_target(st)
    x = render(st[..., :2], kind, size, color, active)
    if model.family == "rec":
        return x
    with torch.no_grad():
        return model.encode(x, target=True).float()


def _fit_loss(model, zr, tgt, active, w_fg=4.0):
    if model.family == "state":
        act = active[:, None, :, None].float()
        per = (((model.decode(zr) - tgt) ** 2) * act).sum((1, 2, 3)) / (act.sum((1, 2, 3)) * 3 * zr.shape[1])
        return per
    if model.family == "rec":
        xh = model.decode(zr)
        return (fg_weight(tgt, w_fg) * (xh - tgt) ** 2).mean((1, 2, 3, 4))
    return ((zr - tgt) ** 2).mean((1, 2))


def invert(model, mode, ctx_inputs, fut_states, kind, size, color, active, R, F=16, iters=300,
           lr=0.03, h_init=None):
    """Return the optimized belief h_C (B, H) for mode in {'h', 'z'}.

    ctx_inputs: model inputs for the C context frames (frames or observations), used for init.
    fut_states: (B, >=F, K, 4) true future states (frames C..C+F-1 are the fit window).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        z0 = model.enc(ctx_inputs.flatten(0, 1)).view(*ctx_inputs.shape[:2], -1).float()
        _, h0 = model.observe(model.norm(z0))
    tgt = _targets(model, fut_states[:, :F], kind, size, color, active)
    if mode == "h":
        var = (h0 if h_init is None else h_init).clone().requires_grad_(True)
    else:
        var = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([var], lr=lr)
    best = None
    for it in range(iters):
        if mode == "h":
            h = var
        else:
            _, h = model.observe(model.norm(var))
        zr, _ = model.imagine(h, F)
        per = _fit_loss(model, zr, tgt, active)
        loss = per.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        if mode == "h":
            h = var.detach()
        else:
            _, h = model.observe(model.norm(var.detach()))
    return h
