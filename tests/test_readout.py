import numpy as np
import torch

from vibes import sim
from vibes.analysis import C, bayes_reference, matched_pairs, reversed_context
from vibes.data import generate
from vibes.readout import centroids, g_eff_batch
from vibes.render import render


def _frames(p, n):
    st = torch.as_tensor(p["states"][:n, :, :, :2])
    x = render(st, torch.as_tensor(p["kind"][:n]), torch.as_tensor(p["size"][:n]),
               torch.as_tensor(p["color"][:n]), torch.as_tensor(p["active"][:n]))
    return x, st


def test_centroid_readout_is_subpixel_on_true_frames():
    p = generate("test", 24)
    x, st = _frames(p, 24)
    pos, _ = centroids(x, torch.as_tensor(p["color"][:24]), torch.as_tensor(p["active"][:24]))
    dx = pos[..., 0] - st[..., 0]
    dx = dx - torch.round(dx)
    err = torch.sqrt(dx ** 2 + (pos[..., 1] - st[..., 1]) ** 2)
    a = torch.as_tensor(p["active"][:24])[:, None, :].expand_as(err)
    e = err[a] * 64
    assert e.median() < 0.3, e.median()
    assert torch.quantile(e, 0.9) < 1.5   # occlusions make a few frames worse


def test_geff_recovers_gravity_from_true_tracks_and_rendered_frames():
    p = generate("gsweep", 120)
    g = p["g"]
    ge = g_eff_batch(p["states"][:, C:, :, :2], p["size"], p["active"], sim.DT)
    ok = ~np.isnan(ge)
    assert ok.mean() > 0.7
    assert np.abs(ge - g)[ok].max() < 0.05    # exact up to bounces clipped by the margin
    x, _ = _frames(p, 120)
    pos, _ = centroids(x, torch.as_tensor(p["color"]), torch.as_tensor(p["active"]))
    ger = g_eff_batch(pos.numpy()[:, C:], p["size"], p["active"], sim.DT)
    ok = ~np.isnan(ger)
    assert np.median(np.abs(ger - g)[ok]) < 0.3, np.median(np.abs(ger - g)[ok])
    slope = np.polyfit(g[ok], ger[ok], 1)[0]
    assert 0.9 < slope < 1.1, slope


def test_time_reversed_contexts_end_in_the_same_state_and_replay_forward():
    p = generate("val", 300)
    e1, e2, idx = matched_pairs(p, 50, np.random.default_rng(0))
    target = p["states"][idx, C - 1]
    assert np.abs(e1.states[:, C - 1] - target).max() < 1e-5
    assert np.abs(e2.states[:, C - 1] - target).max() < 1e-5
    fw = sim.simulate_batch(e1.states[:, 0].astype(np.float64), e1.kind, e1.size, e1.mass,
                            p["n"][idx], e1.g, e1.e, e1.mu, C)
    dx = fw[..., 0] - e1.states[..., 0]
    dx -= np.round(dx)
    # reversibility is exact for walls/ballistics; object contacts add tiny correction errors
    assert np.median(np.abs(dx)) < 1e-6 and np.quantile(np.abs(dx), 0.99) < 5e-3
    assert not np.allclose(e1.states[:, 0], e2.states[:, 0])  # the pasts differ


def test_bayes_reference_clamps_out_of_range():
    p = generate("gsweep", 400)
    ref = bayes_reference(p, np.arange(400), sigma=0.25 / 64)
    g = p["g"]
    inside = (g >= 4.5) & (g <= 7.5)
    assert np.median(np.abs(ref[inside] - g[inside])) < 0.3
    assert (ref <= 8.0 + 1e-9).all() and (ref >= 4.0 - 1e-9).all()
    hi = g >= 10
    assert np.median(ref[hi]) > 7.5


def test_pos_error_is_mean_over_frames_and_active_objects():
    from vibes.analysis import pos_error
    states = np.zeros((2, 5, 3, 4))
    states[..., 1] = 0.5
    states[:, :, 2, 1] = -10.0          # inactive, parked off-screen
    pos = states[..., :2].copy()
    pos[..., 1] += 0.1                  # constant 0.1 error on every object
    active = np.array([[True, True, False], [True, False, False]])
    np.testing.assert_allclose(pos_error(pos, states, active), [0.1, 0.1])
