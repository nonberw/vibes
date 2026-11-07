import numpy as np
import torch

from vibes import world
from vibes.analysis2 import Meta, pos_err_px
from vibes.dynamics import ChunkWM, state_features, state_positions
from vibes.experiments2 import reversed_context, sim_from
from vibes.readout import ballistic_mask, g_eff_ref_batch
from vibes.render import material_bg, render
from vibes.stage2 import colour_targets


def small(name, N=200):
    return world.generate(name, N=N)


def test_world_generation_is_deterministic_and_uses_dt():
    a, b = small("val", 30), small("val", 30)
    np.testing.assert_array_equal(a["states"], b["states"])
    # ballistic frames obey y'' = -g at dt = 0.05
    y = a["states"][:, :, 0, 1]
    dd = np.diff(y, 2, axis=1) + a["g"][:, None] * world.DT ** 2
    assert (np.abs(dd) < 1e-5).mean() > 0.5


def test_material_tint_encodes_e_and_mu():
    e = torch.tensor([0.6, 1.0, 0.8])
    mu = torch.tensor([0.0, 0.4, 0.2])
    bg = material_bg(e, mu)
    assert bg[1, 0] > bg[2, 0] > bg[0, 0]          # red tracks restitution
    assert bg[1, 1] > bg[2, 1] > bg[0, 1]          # green tracks friction
    x = render(torch.zeros(3, 1, 1, 2), torch.zeros(3, 1, dtype=torch.long), torch.full((3, 1), 0.05),
               torch.zeros(3, 1, dtype=torch.long), torch.zeros(3, 1, dtype=torch.bool), bg=bg)
    np.testing.assert_allclose(x[:, 0, :, 10, 10].numpy(), bg.numpy(), atol=1e-6)


def test_reversed_context_ends_in_state_under_any_gravity():
    p = small("val", 300)
    cons = np.where((p["e"] == 1) & (p["mu"] == 0))[0][:40]
    meta = Meta(p, cons)
    s = p["states"][cons, world.C - 1]
    for g in (4.0, 7.5):
        ctx = reversed_context(s, meta, np.full(len(cons), g))
        assert np.abs(ctx[:, -1] - s).max() < 1e-5
        fw = sim_from(ctx[:, 0], meta, np.full(len(cons), g), np.ones(len(cons)), np.zeros(len(cons)), T=world.C)
        dx = fw[..., 0] - ctx[..., 0]
        dx -= np.round(dx)
        assert np.median(np.abs(dx)) < 1e-6


def test_geff_ref_exact_on_truth_and_ignores_contacts():
    p = small("gsweep", 300)
    S = p["states"][:, world.C:world.C + world.R]
    ge = g_eff_ref_batch(S[..., :2], S, p["g"], p["active"], world.DT, max_start=8)
    ok = np.isfinite(ge)
    assert ok.mean() > 0.5
    assert np.abs(ge - p["g"])[ok].max() < 1e-3
    # a prediction that "hedges" with zero curvature everywhere gives g_eff ~ 0, not g
    flat = S[..., :2].copy()
    flat[..., 1] = flat[:, :1, :, 1]
    ge0 = g_eff_ref_batch(flat, S, p["g"], p["active"], world.DT, max_start=8)
    assert np.nanmax(np.abs(ge0)) < 1e-6


def test_ballistic_mask_pads_contacts():
    y = np.array([0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5])[:, None]    # a kink at index 3
    m = ballistic_mask(y, g=0.0, dt=0.05, pad=1)
    assert not m[2:5].any()


def test_state_features_round_trip():
    p = small("val", 20)
    meta = Meta(p)
    xy = torch.as_tensor(p["states"][:, :5, :, :2])
    f = state_features(xy, meta.t("kind", torch.int64).cpu(), meta.t("size").cpu(),
                       meta.t("active", torch.bool).cpu(), meta.t("e").cpu(), meta.t("mu").cpu())
    back = state_positions(f).numpy()
    err = pos_err_px(back, p["states"][:, :5], p["active"])
    assert err.max() < 1e-3


def test_colour_targets_place_objects_by_colour():
    xy = torch.tensor([[[[0.25, 0.5], [0.75, 0.2], [0.0, -10.0]]]])        # (1, 1, 3, 2)
    color = torch.tensor([[3, 6, 0]])
    active = torch.tensor([[True, True, False]])
    tgt, mask = colour_targets(xy, color, active)
    assert mask[0].nonzero().flatten().tolist() == [3, 6]
    np.testing.assert_allclose(tgt[0, 0, 3].numpy(), [0.0, 1.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(tgt[0, 0, 6, 2].item(), 0.2, atol=1e-6)


def test_chunk_model_shapes_and_belief_dependence():
    m = ChunkWM(32, "S", "gru")
    z = torch.randn(4, world.C, 32)
    pred, h = m(z)
    assert pred.shape == (4, world.R, 32) and h.shape == (4, 128)
    m2 = ChunkWM(32, "S", "lru")
    pred2, h2 = m2(z)
    assert pred2.shape == (4, world.R, 32)


def test_truncnorm_mean_is_stable_and_correct():
    from scipy.stats import truncnorm
    from vibes.stats import truncnorm_mean
    mu, s = np.array([5.0, 6.0, 7.9, 3.0, 12.0, 20.0, 1e3]), np.array([1.0, 0.1, 0.5, 0.3, 0.2, 1.0, 1e-3])
    ref = truncnorm.mean((4 - mu[:3]) / s[:3], (8 - mu[:3]) / s[:3], loc=mu[:3], scale=s[:3])
    np.testing.assert_allclose(truncnorm_mean(mu, s, 4, 8)[:3], ref, rtol=1e-6)
    got = truncnorm_mean(mu, s, 4, 8)
    assert 4 <= got[3] < 4.5 and 7.5 < got[4] <= 8 and abs(got[5] - 8) < 1e-9 and got[6] == 8
    np.testing.assert_allclose(truncnorm_mean(np.array([1.0, 6.0, 20.0]), 0.0, 4, 8), [4, 6, 8])
