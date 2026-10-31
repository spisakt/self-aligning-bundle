"""Monte Carlo sample paths of a single unit: a NumPy Gillespie path and a JAX ensemble of paths."""
import numpy as np
import jax, jax.numpy as jnp


def unit_sample_path(unit, T, seed=0, W0=None, record_every=1.0):
    """Gillespie sample path of an EncounterUnit (NumPy). Returns times, W values, and the presentation,
    boundary and exposed states at the recording times."""
    rng = np.random.default_rng(seed)
    mu, s, Wg = unit.mu, unit.s, unit.W; P, K, Gs, G = unit.shape
    lp0 = unit.logp0()
    from .family import A as A_f
    k, c, b, a = 0, (K // 2 if W0 is None else W0), Gs // 2, G // 2
    t, next_rec = 0.0, 0.0; rec = []
    hmax = max(abs(p[0]) for p in unit.pres)
    rs_max = unit.k_s * np.exp(0.5 * hmax * (s[1] - s[0]))
    R_tot = (1 / unit.tau_u if P > 1 else 0.0) + 2 * rs_max + unit.kappa
    while t < T:
        t += rng.exponential(1 / R_tot)
        while next_rec <= t and next_rec <= T:
            rec.append((next_rec, Wg[c], k, s[b], mu[a])); next_rec += record_every
        u = rng.uniform() * R_tot
        if P > 1 and u < 1 / unit.tau_u:
            k = (k + 1 + rng.integers(P - 1)) % P; continue
        u -= (1 / unit.tau_u if P > 1 else 0.0)
        if u < 2 * rs_max:                                    # boundary move (thinned)
            db = -1 if u < rs_max else 1; b2 = b + db
            if 0 <= b2 < Gs and rng.uniform() * rs_max < unit.k_s * np.exp(0.5 * unit.pres[k][0] * (s[b2] - s[b])):
                b = b2
            continue
        h, v, q = unit.pres[k]; th = Wg[c] * s[b]           # encounter: redraw mu, register the fresh pair
        lg = (th + v) * mu + q * mu ** 2; pr = np.exp(lg - lg.max()); pr /= pr.sum()
        a = rng.choice(G, p=pr)
        if rng.uniform() < unit.p_W:
            dc = -1 if rng.uniform() < 0.5 else 1; c2 = c + dc
            if 0 <= c2 < K:
                d = unit.beta * ((Wg[c2] - Wg[c]) * s[b] * mu[a] - (A_f(Wg[c2] * s[b], mu) - A_f(Wg[c] * s[b], mu))) + lp0[c2] - lp0[c]
                if rng.uniform() < 1 / (1 + np.exp(-d)): c = c2
    rec = np.array(rec)
    return rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3], rec[:, 4]


def unit_mc(unit, T, n_paths=64, n_rec=200, seed=0):
    """JAX Monte Carlo of an EncounterUnit: n_paths independent sample paths from the blind state.
    Returns recording times and the structure W at those times, shape (n_paths, n_rec)."""
    mu, s, Wg = (jnp.array(x) for x in (unit.mu, unit.s, unit.W)); P, K, Gs, G = unit.shape
    lp0 = jnp.array(unit.logp0()); pres = jnp.array(unit.pres)
    A = lambda th: jax.scipy.special.logsumexp(th * mu) - jnp.log(G)
    hmax = max(abs(p[0]) for p in unit.pres)
    rs_max = unit.k_s * np.exp(0.5 * hmax * (unit.s[1] - unit.s[0]))
    r_sw = 1 / unit.tau_u if P > 1 else 0.0
    R_tot = r_sw + 2 * rs_max + unit.kappa
    def step(st, key):
        k, c, b, a = st
        ks = jax.random.split(key, 8)
        u = jax.random.uniform(ks[0]) * R_tot
        is_sw = u < r_sw
        is_s = (~is_sw) & (u < r_sw + 2 * rs_max)
        is_e = (~is_sw) & (~is_s)
        k2 = jnp.where(is_sw, (k + 1 + jax.random.randint(ks[1], (), 0, max(P - 1, 1))) % P, k)
        db = jnp.where(u - r_sw < rs_max, -1, 1); b2 = b + db; okb = (b2 >= 0) & (b2 < Gs); b2c = jnp.clip(b2, 0, Gs - 1)
        accb = jax.random.uniform(ks[2]) * rs_max < unit.k_s * jnp.exp(0.5 * pres[k, 0] * (s[b2c] - s[b]))
        b3 = jnp.where(is_s & okb & accb, b2c, b)
        th = Wg[c] * s[b]
        logits = (th + pres[k, 1]) * mu + pres[k, 2] * mu ** 2
        anew = jax.random.categorical(ks[3], logits)
        a2 = jnp.where(is_e, anew, a)
        dc = jnp.where(jax.random.uniform(ks[4]) < 0.5, -1, 1); c2 = c + dc; okc = (c2 >= 0) & (c2 < K); c2c = jnp.clip(c2, 0, K - 1)
        d = unit.beta * ((Wg[c2c] - Wg[c]) * s[b] * mu[a2] - (A(Wg[c2c] * s[b]) - A(th))) + lp0[c2c] - lp0[c]
        accw = is_e & okc & (jax.random.uniform(ks[5]) < unit.p_W) & (jax.random.uniform(ks[6]) < jax.nn.sigmoid(d))
        c3 = jnp.where(accw, c2c, c)
        return (k2, c3, b3, a2), None
    steps_per = int(T * R_tot) // n_rec
    @jax.jit
    def run(key):
        st = (jnp.int32(0), jnp.int32(K // 2), jnp.int32(Gs // 2), jnp.int32(G // 2))
        def outer(st, kk):
            st, _ = jax.lax.scan(step, st, jax.random.split(kk, steps_per))
            return st, st[1]
        _, rec = jax.lax.scan(outer, st, jax.random.split(key, n_rec))
        return rec
    rec = jax.vmap(run)(jax.random.split(jax.random.PRNGKey(seed), n_paths))
    return np.arange(1, n_rec + 1) * steps_per / R_tot, np.asarray(unit.W)[np.array(rec)]
