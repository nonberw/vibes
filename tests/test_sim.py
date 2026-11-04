import numpy as np
import pytest

from vibes import sim
from vibes.data import SPECS, generate, sample_params
from dataclasses import replace


def one_ball(x, y, vx, vy, r=0.08, g=6.0, e=1.0, mu=0.0, T=40, kind=0):
    x0 = np.zeros((1, 4)); x0[0] = [x, y, vx, vy]
    k = np.array([kind]); s = np.array([r]); m = sim.area(k, s)
    return sim.simulate(x0, k, s, m, 1, g, e, mu, T), m


def test_ballistic_exact():
    st, _ = one_ball(0.3, 0.6, 0.4, 0.5, g=6.0, T=8)  # apex 0.62, lands after > 0.32 s
    t = np.arange(8) * sim.DT
    np.testing.assert_allclose(st[:, 0, 1], 0.6 + 0.5 * t - 3.0 * t ** 2, atol=1e-9)
    np.testing.assert_allclose(st[:, 0, 0], (0.3 + 0.4 * t) % 1.0, atol=1e-9)


def test_elastic_floor_bounce_conserves_energy_exactly():
    st, m = one_ball(0.5, 0.7, 0.3, 0.0, g=7.0, T=120)
    E = 0.5 * m[0] * (st[:, 0, 2] ** 2 + st[:, 0, 3] ** 2) + m[0] * 7.0 * st[:, 0, 1]
    assert np.abs(E / E[0] - 1).max() < 1e-9


def test_restitution_apex_ratio():
    # drop from rest: apex after one bounce is e^2 of the drop height above the floor contact
    e, r, h0 = 0.8, 0.07, 0.8
    st, _ = one_ball(0.5, h0, 0.0, 0.0, r=r, g=5.0, e=e, T=60)
    y = st[:, 0, 1]
    land = np.argmax(np.diff(y) > 0)  # first frame moving up
    apex = y[land:].max()
    assert abs((apex - r) / (h0 - r) - e ** 2) < 0.02


def test_friction_reduces_horizontal_speed():
    st, _ = one_ball(0.5, 0.5, 1.0, 0.0, g=6.0, e=0.9, mu=0.3, T=40)
    assert st[-1, 0, 2] < 0.9


@pytest.mark.parametrize("seed", [0, 1])
def test_conservation_multi_object(seed):
    spec = replace(SPECS["train"], force_conservative=True, n_choices=(3,))
    p = generate("long", 300, T=72, seed=seed) if False else None
    rng = np.random.default_rng(seed)
    p = sample_params(spec, 300, rng)
    st = sim.simulate_batch(p["x0"], p["kind"], p["size"], p["mass"], p["n"], p["g"], p["e"], p["mu"], 72)
    E = sim.energy(st, p["mass"], p["g"], p["active"])
    pxm, _ = sim.momentum(st, p["mass"], p["active"])
    relE = np.abs(E / E[:, :1] - 1).max(1)
    # positional correction at object contacts is the only source of error
    assert np.median(relE) < 2e-3, np.median(relE)
    assert np.quantile(relE, 0.95) < 2e-2, np.quantile(relE, 0.95)
    # momentum: exact up to float error
    assert np.abs(pxm - pxm[:, :1]).max() < 1e-9


def test_momentum_conserved_with_inelastic_contacts_when_frictionless():
    spec = replace(SPECS["train"], n_choices=(3,), p_elastic=0.0, p_frictionless=1.0)
    p = sample_params(spec, 200, np.random.default_rng(3))
    st = sim.simulate_batch(p["x0"], p["kind"], p["size"], p["mass"], p["n"], p["g"], p["e"], p["mu"], 40)
    pxm, _ = sim.momentum(st, p["mass"], p["active"])
    assert np.abs(pxm - pxm[:, :1]).max() < 1e-9
    E = sim.energy(st, p["mass"], p["g"], p["active"])
    assert (E[:, -1] <= E[:, 0] + 1e-6).all()  # dissipative: energy never increases


def test_objects_stay_in_box_and_do_not_interpenetrate():
    p = generate("train", 400, seed=11)
    st, a, s = p["states"], p["active"], p["size"]
    y = st[..., 1]
    lo = (y - s[:, None, :] >= -1e-4) | ~a[:, None, :]
    hi = (y + s[:, None, :] <= 1 + 1e-4) | ~a[:, None, :]
    assert lo.all() and hi.all()
    # pairwise overlaps (bounding-circle proxy for balls only) are tiny
    worst = 0.0
    for b in range(400):
        n = p["n"][b]
        for i in range(n):
            for j in range(i + 1, n):
                if p["kind"][b, i] == 0 and p["kind"][b, j] == 0:
                    dx = st[b, :, i, 0] - st[b, :, j, 0]; dx -= np.floor(dx + 0.5)
                    d = np.hypot(dx, st[b, :, i, 1] - st[b, :, j, 1])
                    worst = max(worst, (s[b, i] + s[b, j] - d).max())
    assert worst < 0.01, worst


def test_generation_is_deterministic():
    a = generate("test", 20)
    b = generate("test", 20)
    np.testing.assert_array_equal(a["states"], b["states"])


def test_saved_quantities_match_states():
    p = generate("val", 50)
    E = sim.energy(p["states"].astype(np.float64), p["mass"], p["g"], p["active"])
    np.testing.assert_allclose(p["energy"], E, rtol=1e-4, atol=1e-5)
