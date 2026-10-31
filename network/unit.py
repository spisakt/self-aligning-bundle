"""Exact generators for a single unit: a finite-state continuous-time jump process.

Coordinates
    mu  exposed coordinate, G levels on [-1, 1], each level holding d microstates (uniform resting law)
    s   boundary coordinate, Gs levels, driven by the presentation (directed: it does not feel mu or W)
    W   structure (coupling), K levels
    k   presentation (environmental condition), P values
    g   encounter gate: closed (0) or open (1)  [gated chain only]

Rates (every rate obeys local detailed balance with respect to the energy it involves)
    micro hops within a level                     k_x
    gate opens (an encounter begins)              kappa
    while open: level changes of mu               k_o * exp(dE/2) / (G d),  dE = (W s + v_k) dmu + q_k d(mu^2)
    gate closes (the encounter ends)              k_c; the same event attempts, with probability p_W, a
                                                  structural step W -> W +- eps, accepted with the Glauber
                                                  probability 1/(1 + exp(-D)), D = beta d[W s mu - A(W s)] + d log p0(W)
    boundary moves                                k_s * exp(h_k ds / 2)
    presentation switches                         1 / tau_u
No inference or learning rule is coded: the posterior, the prediction error and the learning rule are
properties of the stationary and averaged behaviour of these rates.

`EncounterUnit` is the limit of fast, complete equilibration within a window (k_x, k_o >> k_c >> rest):
mu is redrawn from its local conditional at rate kappa, and the structural step uses the fresh pair.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .family import grid, A, probs

class _Base:
    def _finish(self, rows, cols, vals, n):
        rows, cols, vals = (np.concatenate(x) for x in (rows, cols, vals))
        keep = (rows != cols) & (vals > 0)
        R = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n))
        return (R - sp.diags(np.asarray(R.sum(1)).ravel())).tocsc()

    def logp0(self):
        return -0.5 * (self.W / self.prior_sd) ** 2

    def boundary_law(self):
        """Time-stationary law rho(k, s) of the autonomous presentation-boundary chain (W-independent)."""
        P, Gs = len(self.pres), len(self.s)
        n = P * Gs; Q = np.zeros((n, n))
        for k in range(P):
            for b in range(Gs):
                i = k * Gs + b
                if P > 1:
                    for k2 in range(P):
                        if k2 != k: Q[i, k2 * Gs + b] += 1 / self.tau_u / (P - 1)
                for db in (-1, 1):
                    b2 = b + db
                    if 0 <= b2 < Gs: Q[i, k * Gs + b2] += self.k_s * np.exp(0.5 * self.pres[k][0] * (self.s[b2] - self.s[b]))
        np.fill_diagonal(Q, -Q.sum(1))
        wv, vv = np.linalg.eig(Q.T); p = np.real(vv[:, np.argmin(np.abs(wv))])
        return (p / p.sum()).reshape(P, Gs)

    def ell(self, W=None):
        """Average log-evidence l(W) = E_rho[A_q(W s + v) - A(W s)] over boundary-presentation pairs."""
        W = self.W if W is None else np.asarray(W)
        rho = self.boundary_law(); out = np.zeros_like(W, dtype=float)
        for k, (h, v, q) in enumerate(self.pres):
            for b, sb in enumerate(self.s):
                out += rho[k, b] * (A(W * sb + v, self.mu, q=q) - A(W * sb, self.mu))
        return out

    def theory_W_law(self):
        lp = self.logp0() + self.beta * self.ell(); p = np.exp(lp - lp.max()); return p / p.sum()

    def _W_step_probs(self, c, sb, mua):
        """Acceptance-weighted probabilities of W -> W +- 1 grid step at pair (s, mu); arrays over states."""
        out = {}
        lp0 = self.logp0()
        for dc in (-1, 1):
            c2 = c + dc; ok = (c2 >= 0) & (c2 < len(self.W)); c2c = np.clip(c2, 0, len(self.W) - 1)
            W1, W2 = self.W[c], self.W[c2c]
            d = self.beta * ((W2 - W1) * sb * mua - (A(W2 * sb, self.mu) - A(W1 * sb, self.mu))) + lp0[c2c] - lp0[c]
            out[dc] = (c2c, np.where(ok, 0.5 * self.p_W / (1.0 + np.exp(-np.clip(d, -50, 50))), 0.0))   # Glauber (Barker) acceptance
        return out

    def averaged_W_generator(self):
        """Learning law of the structure: the W-chain whose rates are the encounter-averaged step rates
        (averaging principle; the fast pair (s, mu') is sampled at encounters)."""
        K = len(self.W); rho = self.boundary_law(); Qw = np.zeros((K, K)); lp0 = self.logp0()
        for c in range(K):
            for dc in (-1, 1):
                c2 = c + dc
                if not 0 <= c2 < K: continue
                r = 0.0
                for k, (h, v, q) in enumerate(self.pres):
                    for b, sb in enumerate(self.s):
                        pm = probs(self.W[c] * sb + v, self.mu, q=q)
                        d = self.beta * ((self.W[c2] - self.W[c]) * sb * self.mu - (A(self.W[c2] * sb, self.mu) - A(self.W[c] * sb, self.mu))) + lp0[c2] - lp0[c]
                        r += rho[k, b] * (pm / (1.0 + np.exp(-np.clip(d, -50, 50)))).sum()
                Qw[c, c2] = self.kappa * self.p_W * 0.5 * r
            Qw[c, c] = -Qw[c].sum()
        return Qw

    def stationary(self, Q=None):
        Q = self.generator() if Q is None else Q
        n = Q.shape[0]; M = Q.T.tolil(); M[0, :] = 1.0
        rhs = np.zeros(n); rhs[0] = 1.0
        p = spla.spsolve(M.tocsc(), rhs); p = np.maximum(p, 0); return p / p.sum()


class GatedUnit(_Base):
    def __init__(self, G=11, d=2, Gs=5, K=41, Wmax=3.0, pres=((2.0, 2.0, -2.0), (-2.0, -2.0, -2.0)),
                 tau_u=2.0, k_x=300.0, k_o=300.0, k_c=30.0, kappa=1.0, k_s=10.0, p_W=0.05, beta=4.0,
                 prior_sd=1.5):
        self.mu, self.s, self.W = grid(G), grid(Gs), np.linspace(-Wmax, Wmax, K)
        self.G, self.d, self.pres, self.tau_u = G, d, [tuple(p) for p in pres], tau_u
        self.k_x, self.k_o, self.k_c, self.kappa, self.k_s = k_x, k_o, k_c, kappa, k_s
        self.p_W, self.beta, self.prior_sd = p_W, beta, prior_sd
        self.shape = (len(self.pres), K, Gs, G, d, 2)

    def generator(self):
        P, K, Gs, G, d, _ = self.shape; n = int(np.prod(self.shape))
        k, c, b, a, x, g = (arr.ravel() for arr in np.indices(self.shape))
        idx = lambda *t: np.ravel_multi_index(t, self.shape)
        i0 = np.arange(n); rows, cols, vals = [], [], []
        def add(msk, j, r):
            rows.append(i0[msk]); cols.append(j[msk] if np.ndim(j) else np.full(msk.sum(), j)); vals.append(np.broadcast_to(r, i0.shape)[msk])
        if P > 1:
            for k2 in range(P):
                m = k != k2; add(m, idx(np.full(n, k2), c, b, a, x, g), np.full(n, 1 / self.tau_u / (P - 1)))
        h = np.array([p[0] for p in self.pres])[k]
        for db in (-1, 1):
            b2 = b + db; m = (b2 >= 0) & (b2 < Gs); b2c = np.clip(b2, 0, Gs - 1)
            add(m, idx(k, c, b2c, a, x, g), self.k_s * np.exp(0.5 * h * (self.s[b2c] - self.s[b])))
        for x2 in range(d):                                                   # micro hops within a level
            m = x != x2; add(m, idx(k, c, b, a, np.full(n, x2), g), np.full(n, self.k_x))
        m = g == 0; add(m, idx(k, c, b, a, x, np.ones(n, int)), np.full(n, self.kappa))       # encounter begins
        th = self.W[c] * self.s[b]; v = np.array([p[1] for p in self.pres])[k]; q = np.array([p[2] for p in self.pres])[k]
        for a2 in range(G):                                                   # open window: mu equilibrates
            dE = (th + v) * (self.mu[a2] - self.mu[a]) + q * (self.mu[a2] ** 2 - self.mu[a] ** 2)
            for x2 in range(d):
                m = (g == 1) & (a2 != a); add(m, idx(k, c, b, np.full(n, a2), np.full(n, x2), g), self.k_o * np.exp(0.5 * dE) / (G * d))
        steps = self._W_step_probs(c, self.s[b], self.mu[a])                  # encounter ends; structure registers it
        stay = 1.0 - steps[-1][1] - steps[1][1]
        m = g == 1
        add(m, idx(k, c, b, a, x, np.zeros(n, int)), self.k_c * stay)
        for dc in (-1, 1):
            c2c, pr = steps[dc]; add(m & (pr > 0), idx(k, c2c, b, a, x, np.zeros(n, int)), self.k_c * pr)
        return self._finish(rows, cols, vals, n)

    def marginal(self, p, axis):
        names = ['k', 'c', 'b', 'a', 'x', 'g']; P = p.reshape(self.shape)
        keep = names.index(axis); return P.sum(tuple(i for i in range(6) if i != keep))


class EncounterUnit(_Base):
    def __init__(self, G=11, Gs=5, K=41, Wmax=3.0, pres=((2.0, 2.0, -2.0), (-2.0, -2.0, -2.0)), tau_u=2.0,
                 kappa=1.0, k_s=10.0, p_W=0.05, beta=4.0, prior_sd=1.5):
        self.mu, self.s, self.W = grid(G), grid(Gs), np.linspace(-Wmax, Wmax, K)
        self.G, self.pres, self.tau_u, self.kappa, self.k_s = G, [tuple(p) for p in pres], tau_u, kappa, k_s
        self.p_W, self.beta, self.prior_sd = p_W, beta, prior_sd
        self.shape = (len(self.pres), K, Gs, G)

    def generator(self):
        P, K, Gs, G = self.shape; n = int(np.prod(self.shape))
        k, c, b, a = (arr.ravel() for arr in np.indices(self.shape))
        idx = lambda *t: np.ravel_multi_index(t, self.shape)
        i0 = np.arange(n); rows, cols, vals = [], [], []
        def add(msk, j, r):
            rows.append(i0[msk]); cols.append(j[msk]); vals.append(np.broadcast_to(r, i0.shape)[msk])
        if P > 1:
            for k2 in range(P):
                add(k != k2, idx(np.full(n, k2), c, b, a), np.full(n, 1 / self.tau_u / (P - 1)))
        h = np.array([p[0] for p in self.pres])[k]
        for db in (-1, 1):
            b2 = b + db; m = (b2 >= 0) & (b2 < Gs); b2c = np.clip(b2, 0, Gs - 1)
            add(m, idx(k, c, b2c, a), self.k_s * np.exp(0.5 * h * (self.s[b2c] - self.s[b])))
        th = self.W[c] * self.s[b]; v = np.array([p[1] for p in self.pres])[k]; q = np.array([p[2] for p in self.pres])[k]
        lp = (th + v)[:, None] * self.mu[None, :] + q[:, None] * self.mu[None, :] ** 2
        lp -= lp.max(1, keepdims=True); pm = np.exp(lp); pm /= pm.sum(1, keepdims=True)
        for a2 in range(G):
            steps = self._W_step_probs(c, self.s[b], np.full(n, self.mu[a2]))
            stay = 1.0 - steps[-1][1] - steps[1][1]
            add(np.ones(n, bool), idx(k, c, b, np.full(n, a2)), self.kappa * pm[:, a2] * stay)
            for dc in (-1, 1):
                c2c, pr = steps[dc]; add(pr > 0, idx(k, c2c, b, np.full(n, a2)), self.kappa * pm[:, a2] * pr)
        return self._finish(rows, cols, vals, n)

    def W_marginal(self, p):
        return p.reshape(self.shape).sum((0, 2, 3))

    def transient_W(self, times, W0_index=None):
        """Exact law of W(t) from the blind state (W = 0, boundary and mu at stationarity of their own)."""
        P, K, Gs, G = self.shape; Q = self.generator()
        p0 = np.zeros(self.shape); c0 = K // 2 if W0_index is None else W0_index
        rho = self.boundary_law()
        for k in range(P):
            for b in range(Gs):
                p0[k, c0, b, :] = rho[k, b] / G
        out = []; p = p0.ravel(); t_prev = 0.0
        for t in times:
            p = spla.expm_multiply(Q.T * (t - t_prev), p); t_prev = t
            out.append(p.reshape(self.shape).sum((0, 2, 3)))
        return np.array(out)


def spectrum(unit, k=None):
    """Relaxation rates (-Re of the generator's eigenvalues) of a small unit, dense eigen-decomposition."""
    Q = unit.generator().toarray()
    ev = np.linalg.eigvals(Q)
    return np.sort(-ev.real)[1:], ev
