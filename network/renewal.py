"""Networks of conditional units with causal renewal.

Fast state. Each unit's exposed coordinate takes G equally spaced levels in [-1, 1]. At every renewal all units
are redrawn together, each from its local conditional given the preceding network state x:
    p(mu_i' | x) ∝ exp((h_i + PI x_i^obs) mu_i' - PI/2 mu_i'^2),   h = W x,
i.e. the unit's prediction h_i tilted by an observation of value x_i^obs with precision PI. Between presentations
the state carries over; it is never reset.

Structure. Every synapse registers the renewal with the statistic
    s_j^+ (mu_i^+ - L(h_i))            (lagged pairing: prediction from the preceding state, renewed boundary),
and moves by lr times this statistic (the low-noise limit of the structural chain). Two controls change only the
timing: 'preceding' pairs the error with the preceding boundary s_j (the pseudo-likelihood gradient), and
'reset' sets the fast state to zero at every switch.
"""
import numpy as np, jax, jax.numpy as jnp
from functools import partial

G = 21
MU = jnp.linspace(-1, 1, G)


def Lg(th):
    """Mean map of the G-level family (the discrete Langevin function)."""
    return (jax.nn.softmax(jnp.asarray(th)[..., None] * MU, -1) * MU).sum(-1)


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def learn(X, PI, lr, key, W0, n_pres, renewals, pairing, cyclic):
    """Present the rows of X (observed values) for `renewals` renewals each, n_pres times, in random or cyclic
    order. pairing: 0 lagged (the network rule), 1 preceding boundary, 2 lagged with reset at every switch."""
    P, N = X.shape
    mask = 1 - jnp.eye(N)

    def presentation(c, k):
        W, x, kprev = c
        k1, k2 = jax.random.split(k)
        kk = (kprev + 1) % P if cyclic else jax.random.randint(k1, (), 0, P)
        xo = X[kk]
        x = jnp.where(pairing == 2, jnp.zeros_like(x), x)

        def renewal(c2, kr):
            W, x = c2
            h = W @ x
            xn = MU[jax.random.categorical(kr, (h + PI * xo)[:, None] * MU - 0.5 * PI * MU ** 2)]
            pre = jnp.where(pairing == 1, x, xn)
            return (W + lr * jnp.outer(xn - Lg(h), pre) * mask, xn), None

        (W, x), _ = jax.lax.scan(renewal, (W, x), jax.random.split(k2, renewals))
        return (W, x, kk), None

    (W, _, _), _ = jax.lax.scan(presentation, (W0, jnp.zeros(X.shape[1]), 0), jax.random.split(key, n_pres))
    return W


@partial(jax.jit, static_argnums=(3,))
def free_run(W, x0, key, T):
    """Undriven renewals at frozen couplings; returns the T successive network states."""
    def renewal(x, k):
        xn = MU[jax.random.categorical(k, (W @ x)[:, None] * MU)]
        return xn, xn
    return jax.lax.scan(renewal, x0, jax.random.split(key, T))[1]


def attractors(W, seeds, iters=400):
    """Deterministic readout: iterate x <- L(W x) from each seed."""
    Wj = jnp.asarray(W)
    out = []
    for s in seeds:
        x = jnp.asarray(s)
        for _ in range(iters):
            x = Lg(Wj @ x)
        out.append(np.asarray(x))
    return np.array(out)


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def distinct_attractors(A, thr=0.99, min_norm=0.05):
    """Distinct attractors, with a and -a counted as one (evidence for or against the same pattern).
    Returns the list of attractors and, for each seed, (index, sign)."""
    U, lab = [], []
    for a in A:
        if np.linalg.norm(a) < min_norm * np.sqrt(len(a)):
            lab.append((-1, 0)); continue
        for i, u in enumerate(U):
            r = np.corrcoef(a, u)[0, 1]
            if abs(r) > thr:
                lab.append((i, int(np.sign(r)))); break
        else:
            U.append(a); lab.append((len(U) - 1, 1))
    return np.array(U), lab


def orthogonality(Y):
    """Mean deviation from 90 degrees over all pairs of rows."""
    a = [angle(Y[i], Y[j]) for i in range(len(Y)) for j in range(i + 1, len(Y))]
    return float(np.mean(np.abs(90 - np.array(a)))) if a else np.nan


def gain_ratio(W, X):
    """Gain of the symmetric couplings on the mean pattern relative to the centred patterns."""
    S = (W + W.T) / 2
    m = X.mean(0)
    R = lambda v: v @ S @ v / (v @ v)
    return float(R(m) / np.mean([R(x) for x in X - m]))
