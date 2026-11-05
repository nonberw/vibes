"""Training loop shared by all world-model families."""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import GPUData, load
from .models import (WorldModel, fg_weight, n_params, state_obs, state_target, vicreg_terms)

C, L = 8, 32  # context frames, window length (24 open-loop frames)


def batch_inputs(model, x, st, data, idx):
    if model.family == "state":
        return state_obs(st, data.kind[idx], data.size[idx], data.active[idx])
    return x


def horizon(step, cfg):
    """Open-loop horizon K at a training step (curriculum: 1 -> 24 over cfg['cur_frac'])."""
    if cfg.get("cur_frac", 0) <= 0:
        return L - C
    return int(max(1, min(L - C, round((L - C) * step / (cfg["cur_frac"] * cfg["steps"])))))


def loss_fn(model, x, st, data, idx, cfg):
    """x, st: windows of length C + K; predictions for frames 1..C+K-1."""
    Lw = st.shape[1]
    inp = batch_inputs(model, x, st, data, idx)
    z = model.encode(inp)
    preds_ctx, zr, h, _ = model.rollout(z[:, :C], Lw - C)
    zhat = torch.cat([preds_ctx[:, :-1], zr], 1)   # predictions for frames 1..L-1
    logs = {}
    if model.family == "rec":
        # decode a random subset of predicted frames (cost); gradients still reach every step
        ti = torch.randperm(Lw - 1, device=x.device)[:cfg["n_dec"]].sort().values
        xt = x[:, 1:][:, ti]
        xh = model.decode(zhat[:, ti])
        pred = (fg_weight(xt, cfg["w_fg"]) * (xh - xt) ** 2).mean()
        xa = model.decode(z[:, :C])  # autoencoding term on context frames only (cost)
        ae = (fg_weight(x[:, :C], cfg["w_fg"]) * (xa - x[:, :C]) ** 2).mean()
        loss = pred + ae
        logs.update(pred=pred.item(), ae=ae.item())
    elif model.family == "state":
        tgt = state_target(st)
        act = data.active[idx][:, None, :, None].float()
        pred = (((model.decode(zhat) - tgt[:, 1:]) ** 2) * act).sum() / (act.sum() * 3 * (Lw - 1))
        ae = (((model.decode(z) - tgt) ** 2) * act).sum() / (act.sum() * 3 * Lw)
        loss = pred + ae
        logs.update(pred=pred.item(), ae=ae.item())
    else:  # jepa
        with torch.no_grad():
            zt = model.encode(inp, target=True)
        pred = F.mse_loss(zhat, zt[:, 1:])
        var, cov = vicreg_terms(z.flatten(0, 1).float())
        loss = pred + cfg["w_var"] * var + cfg["w_cov"] * cov
        logs.update(pred=pred.item(), var=var.item(), cov=cov.item(),
                    tstd=zt.flatten(0, 1).std(0).mean().item())
    return loss, logs


def train(cfg, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    dev = cfg.get("device", "cuda")
    if dev == "cpu":
        torch.set_num_threads(cfg.get("threads", 8))
    data = GPUData(load(cfg["train_path"]), dev, n_limit=cfg["n_train"])
    model = WorldModel(cfg["family"], cfg["size"], cfg.get("core", "gru"), cfg.get("enc", "conv")).to(dev)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    if model.family == "jepa":  # decoder is a post-hoc readout; never trained here
        params = [p for n, p in model.named_parameters()
                  if p.requires_grad and not n.startswith("dec.")]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = cfg["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
    cfg = dict(cfg, n_params=n_params(model))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(out / "log.txt", "w")
    t0 = time.time()
    for step in range(1, steps + 1):
        K = horizon(step, cfg)
        x, st, idx = data.sample(cfg["batch"], C + K, gen, do_render=model.family != "state")
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=cfg["amp"] and dev == "cuda"):
            loss, logs = loss_fn(model, x, st, data, idx, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if model.family == "jepa":
            tau = 1 - (1 - cfg["ema"]) * 0.5 * (1 + math.cos(math.pi * step / steps))
            model.ema_update(tau)
        if step % 250 == 0 or step == 1:
            msg = f"step {step} K {K} loss {loss.item():.5f} " + " ".join(
                f"{k} {v:.5f}" for k, v in logs.items()) + f" t {time.time() - t0:.0f}s"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
    torch.save(model.state_dict(), out / "model.pt")
    if model.family == "jepa":
        fit_readout_decoder(model, cfg, out)
    (out / "done").write_text(f"{time.time() - t0:.0f}")
    return model


def fit_readout_decoder(model, cfg, out, steps=None):
    """JEPA readout: fit the pixel decoder on frozen target-encoder latents of real *val* frames."""
    dev = "cuda"
    steps = steps or cfg.get("dec_steps", 4000)
    data = GPUData(load(cfg["val_path"]), dev)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.dec.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.dec.parameters(), lr=1e-3, weight_decay=0.0)
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"] + 1000)
    for step in range(1, steps + 1):
        x, st, idx = data.sample(16, 8, gen)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"]):
            z = model.encode(x, target=True)
        xh = model.decode(z.float())
        loss = (fg_weight(x, cfg["w_fg"]) * (xh - x) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"readout-decoder step {step} loss {loss.item():.5f}", flush=True)
    torch.save(model.state_dict(), out / "model.pt")


DEFAULTS = dict(family="rec", size="M", seed=0, n_train=16000, steps=8000, batch=16, lr=1e-3,
                wd=1e-4, w_fg=4.0, w_var=1.0, w_cov=0.05, ema=0.996, amp=True, dec_steps=4000, n_dec=12, cur_frac=0.0, core="gru", enc="conv",
                device="cuda", threads=8, train_path="data/train.npz", val_path="data/val.npz")
