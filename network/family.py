"""The bounded exponential family of the exposed coordinate, on a finite grid and in the continuum."""
import numpy as np

def grid(G):
    """Cell-centred grid of G levels on [-1, 1]: equal numbers of microstates per level give the
    discrete counterpart of the uniform resting law."""
    return -1.0 + (np.arange(G) + 0.5) * 2.0 / G

def _lse(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)

def A(theta, mu, w=None, q=0.0):
    """Log-partition of w(mu) exp(theta mu + q mu^2) on the grid mu, normalized to 0 at theta = q = 0."""
    w = np.ones_like(mu) if w is None else np.asarray(w, float)
    th = np.asarray(theta, float)[..., None]
    return _lse(th * mu + q * mu ** 2 + np.log(w)) - np.log(w.sum())

def probs(theta, mu, w=None, q=0.0):
    w = np.ones_like(mu) if w is None else np.asarray(w, float)
    th = np.asarray(theta, float)[..., None]
    a = th * mu + q * mu ** 2 + np.log(w); a = a - a.max(-1, keepdims=True); p = np.exp(a)
    return p / p.sum(-1, keepdims=True)

def L(theta, mu, w=None, q=0.0):
    """Mean map (the unit's activation) on the grid."""
    return (probs(theta, mu, w, q) * mu).sum(-1)

def V(theta, mu, w=None, q=0.0):
    p = probs(theta, mu, w, q); m = (p * mu).sum(-1, keepdims=True)
    return (p * (mu - m) ** 2).sum(-1)

def langevin(t):
    """Continuum mean map L(theta) = coth(theta) - 1/theta of the continuous Bernoulli on [-1, 1]."""
    t = np.asarray(t, float); out = np.empty_like(t); sm = np.abs(t) < 1e-4
    out[sm] = t[sm] / 3; ts = t[~sm]; out[~sm] = 1 / np.tanh(ts) - 1 / ts
    return out

def cb_density(x, theta):
    """Continuous Bernoulli density on [-1, 1] with natural parameter theta."""
    if abs(theta) < 1e-8:
        return np.full_like(np.asarray(x, float), 0.5)
    return theta * np.exp(theta * x) / (2 * np.sinh(theta))
