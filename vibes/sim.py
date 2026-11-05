"""2D rigid-body simulator for the VIBES worlds.

World: the unit box, periodic in x (x in [0, 1)), with a floor at y=0 and a ceiling at y=1.
Objects are balls (radius s) or axis-aligned non-rotating square blocks (half-size s). Mass is
density * area with density 1, so mass is visible from size. Per-episode parameters: gravity g
(acceleration -g in y), restitution e (all contacts), and surface friction mu (Coulomb friction
at the floor and ceiling only). Objects collide with each other frictionlessly.

Conservation by construction:
- total horizontal momentum p_x = sum m vx is conserved exactly when mu = 0 (no external
  horizontal force: x is periodic, floor/ceiling normals are vertical, object contacts are internal);
- total mechanical energy E = sum (1/2 m |v|^2 + m g y) is conserved when e = 1 and mu = 0,
  up to the integrator error measured in tests/test_sim.py.

Integration: exact ballistic motion within substeps; floor and ceiling contacts use the exact time
of impact (solving the quadratic); object-object contacts use overlap detection + impulse + a
mass-weighted positional correction.
"""
import math

import numpy as np
from numba import njit, prange

BALL, BLOCK = 0, 1
DT = 0.04          # seconds per rendered frame
NSUB = 32          # physics substeps per frame
V_REST = 0.03      # below this bounce speed an object comes to rest on the floor (e < 1 only)


@njit(cache=True)
def _wrap(dx):
    return dx - math.floor(dx + 0.5)


@njit(cache=True)
def _wall_step(y, vy, vx, s, g, e, mu, h, resting):
    """Advance one object's vertical motion by h with exact floor/ceiling impacts.

    Returns (y, vy, vx, resting, x_advance_time) where x moves with constant vx between impacts;
    vx can change at impacts (friction), so we return the horizontal displacement instead.
    """
    rem = h
    dxs = 0.0
    for _ in range(8):
        if rem <= 0.0:
            break
        if resting:
            # sliding on the floor: Coulomb deceleration mu*g
            dec = mu * g * rem
            if abs(vx) <= dec:
                if mu * g > 0:
                    tstop = abs(vx) / (mu * g)
                else:
                    tstop = rem
                dxs += vx * tstop * 0.5
                vx = 0.0
            else:
                dxs += vx * rem - 0.5 * math.copysign(mu * g, vx) * rem * rem
                vx -= math.copysign(dec, vx)
            y = s
            vy = 0.0
            rem = 0.0
            break
        # time to floor (y = s)
        c = y - s
        t_floor = 1e9
        if g > 1e-12:
            disc = vy * vy + 2.0 * g * max(c, 0.0)
            t_floor = (vy + math.sqrt(disc)) / g
        elif vy < 0.0:
            t_floor = max(c, 0.0) / (-vy)
        # time to ceiling (y = 1 - s)
        cc = y - (1.0 - s)  # <= 0
        t_ceil = 1e9
        if vy > 0.0:
            if g > 1e-12:
                disc = vy * vy + 2.0 * g * min(cc, 0.0)
                if disc >= 0.0:
                    t_ceil = (vy - math.sqrt(disc)) / g
            else:
                t_ceil = -min(cc, 0.0) / vy
        t_hit = min(t_floor, t_ceil)
        if t_hit >= rem:
            y = y + vy * rem - 0.5 * g * rem * rem
            vy = vy - g * rem
            dxs += vx * rem
            rem = 0.0
            break
        t_hit = max(t_hit, 0.0)
        dxs += vx * t_hit
        vy_hit = vy - g * t_hit
        if t_floor <= t_ceil:
            y = s
            vin = -vy_hit if vy_hit < 0.0 else 0.0
            vy = e * vin
        else:
            y = 1.0 - s
            vin = vy_hit if vy_hit > 0.0 else 0.0
            vy = -e * vin
        # Coulomb friction impulse at the surface
        jt = mu * (1.0 + e) * vin
        if abs(vx) <= jt:
            vx = 0.0
        else:
            vx -= math.copysign(jt, vx)
        rem -= t_hit
        if t_floor <= t_ceil and e < 1.0 and vy < V_REST and g > 1e-12:
            resting = True
    return y, vy, vx, resting, dxs


@njit(cache=True)
def _contact(kind_i, si, kind_j, sj, dx, dy):
    """Contact normal (from j to i) and penetration depth for i at offset (dx, dy) from j."""
    if kind_i == BALL and kind_j == BALL:
        d = math.sqrt(dx * dx + dy * dy)
        pen = si + sj - d
        if pen <= 0.0:
            return 0.0, 0.0, -1.0
        if d < 1e-12:
            return 0.0, 1.0, pen
        return dx / d, dy / d, pen
    if kind_i == BLOCK and kind_j == BLOCK:
        px = si + sj - abs(dx)
        py = si + sj - abs(dy)
        if px <= 0.0 or py <= 0.0:
            return 0.0, 0.0, -1.0
        if px < py:
            return (1.0 if dx >= 0 else -1.0), 0.0, px
        return 0.0, (1.0 if dy >= 0 else -1.0), py
    # ball vs block: work in the frame of the block, ball center at (bx, by)
    if kind_i == BALL:
        bx, by, r, a, sign = dx, dy, si, sj, 1.0
    else:
        bx, by, r, a, sign = -dx, -dy, sj, si, -1.0
    cx = min(max(bx, -a), a)
    cy = min(max(by, -a), a)
    ox, oy = bx - cx, by - cy
    d = math.sqrt(ox * ox + oy * oy)
    if d > 1e-12:
        pen = r - d
        if pen <= 0.0:
            return 0.0, 0.0, -1.0
        nx, ny = ox / d, oy / d
    else:  # ball center inside the block: push out along the axis of least penetration
        px = a + r - abs(bx)
        py = a + r - abs(by)
        if px < py:
            nx, ny, pen = (1.0 if bx >= 0 else -1.0), 0.0, px
        else:
            nx, ny, pen = 0.0, (1.0 if by >= 0 else -1.0), py
    # normal points from block to ball; we need from j to i
    return sign * nx, sign * ny, pen


@njit(cache=True)
def simulate(x0, kind, size, mass, n, g, e, mu, T, dt=DT):
    """Simulate one episode. x0: (K, 4) [x, y, vx, vy]; only the first n objects are active.

    Returns states (T, K, 4), with states[0] = x0.
    """
    K = x0.shape[0]
    st = np.zeros((T, K, 4))
    cur = x0.copy()
    resting = np.zeros(K, dtype=np.bool_)
    st[0] = cur
    h = dt / NSUB
    for t in range(1, T):
        for _ in range(NSUB):
            for i in range(n):
                y, vy, vx, rs, dxs = _wall_step(cur[i, 1], cur[i, 3], cur[i, 2], size[i],
                                               g, e, mu, h, resting[i])
                cur[i, 0] = (cur[i, 0] + dxs) % 1.0
                cur[i, 1], cur[i, 2], cur[i, 3] = y, vx, vy
                resting[i] = rs
            for i in range(n):
                for j in range(i + 1, n):
                    dx = _wrap(cur[i, 0] - cur[j, 0])
                    dy = cur[i, 1] - cur[j, 1]
                    nx, ny, pen = _contact(kind[i], size[i], kind[j], size[j], dx, dy)
                    if pen <= 0.0:
                        continue
                    rvx = cur[i, 2] - cur[j, 2]
                    rvy = cur[i, 3] - cur[j, 3]
                    vn = rvx * nx + rvy * ny
                    wi, wj = 1.0 / mass[i], 1.0 / mass[j]
                    if vn < 0.0:
                        jimp = -(1.0 + e) * vn / (wi + wj)
                        cur[i, 2] += jimp * wi * nx
                        cur[i, 3] += jimp * wi * ny
                        cur[j, 2] -= jimp * wj * nx
                        cur[j, 3] -= jimp * wj * ny
                        # a contact can lift a resting object
                        resting[i] = False
                        resting[j] = False
                    # positional correction, mass weighted
                    ci = pen * wi / (wi + wj)
                    cj = pen * wj / (wi + wj)
                    cur[i, 0] = (cur[i, 0] + ci * nx) % 1.0
                    cur[i, 1] += ci * ny
                    cur[j, 0] = (cur[j, 0] - cj * nx) % 1.0
                    cur[j, 1] -= cj * ny
                    for k in (i, j):
                        if cur[k, 1] < size[k]:
                            cur[k, 1] = size[k]
                        if cur[k, 1] > 1.0 - size[k]:
                            cur[k, 1] = 1.0 - size[k]
        st[t] = cur
    return st


@njit(parallel=True, cache=True)
def simulate_batch(x0, kind, size, mass, n, g, e, mu, T, dt=DT):
    N, K = x0.shape[0], x0.shape[1]
    out = np.zeros((N, T, K, 4))
    for b in prange(N):
        out[b] = simulate(x0[b], kind[b], size[b], mass[b], n[b], g[b], e[b], mu[b], T, dt)
    return out


def area(kind, size):
    return np.where(kind == BALL, np.pi * size ** 2, 4.0 * size ** 2)


def energy(states, mass, g, active):
    """Total mechanical energy per frame. states (..., T, K, 4), mass (..., K), g (...), active (..., K)."""
    v2 = states[..., 2] ** 2 + states[..., 3] ** 2
    e = 0.5 * mass[..., None, :] * v2 + mass[..., None, :] * g[..., None, None] * states[..., 1]
    return (e * active[..., None, :]).sum(-1)


def momentum(states, mass, active):
    """Total (p_x, p_y) per frame."""
    w = (mass * active)[..., None, :]
    return (w * states[..., 2]).sum(-1), (w * states[..., 3]).sum(-1)
