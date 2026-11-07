"""Small numerically stable statistics helpers."""
import numpy as np
from scipy.special import ndtr


def truncnorm_mean(mu, s, a, b, cut=8.0):
    """Mean of N(mu, s^2) truncated to [a, b], elementwise; stable for tiny s and far-off mu.

    If the whole interval lies more than `cut` standard deviations above (below) mu, the mean is the
    nearer endpoint to machine precision, so we return it directly. With s = 0 the mean is
    clip(mu, a, b).
    """
    mu = np.asarray(mu, dtype=np.float64)
    s = np.broadcast_to(np.asarray(s, dtype=np.float64), mu.shape)
    out = np.clip(mu, a, b)
    ok = s > 0
    if not ok.any():
        return out
    al = (a - mu[ok]) / s[ok]
    be = (b - mu[ok]) / s[ok]
    res = np.empty_like(al)
    hi = al > cut            # interval far above mu: mean -> a
    lo = be < -cut           # interval far below mu: mean -> b
    mid = ~(hi | lo)
    res[hi] = a
    res[lo] = b
    if mid.any():
        phi = lambda x: np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
        Z = ndtr(be[mid]) - ndtr(al[mid])
        res[mid] = mu[ok][mid] + s[ok][mid] * (phi(al[mid]) - phi(be[mid])) / np.maximum(Z, 1e-300)
    out[ok] = res
    return out
