OUT = globals().get('OUT', '../fig')
import numpy as np, matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, FancyBboxPatch, Circle, PathPatch
from matplotlib.path import Path
from scipy.spatial import cKDTree
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'mathtext.fontset': 'cm',
                     'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7})
INK = '#2b2a28'; INK2 = '#5c5b57'; MUTED = '#9a9994'; FAINT = '#dcdbd7'
BLUE = '#2a78d6'; ORANGE = '#eb6834'; PURPLE = '#8a5fd1'
W_MM, H_MM = 180, 84
fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W_MM); ax.set_ylim(0, H_MM); ax.axis('off'); ax.set_aspect('equal')
rng = np.random.default_rng(3)
rgba = mpl.colors.to_rgba

def arrow(p, q, color=INK, lw=0.8, ms=6, rad=0.0, z=5, ls='-', alpha=1, style='-|>'):
    ax.add_patch(FancyArrowPatch(tuple(p), tuple(q), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, zorder=z,
                                 connectionstyle=f'arc3,rad={rad}', linestyle=ls, alpha=alpha, shrinkA=0, shrinkB=0))
def label(x, y, s, **kw):
    for k, v in dict(ha='center', va='center', color=INK, fontsize=7).items(): kw.setdefault(k, v)
    ax.text(x, y, s, **kw)
def curve(c, **kw): c = np.asarray(c); ax.plot(c[:, 0], c[:, 1], solid_capstyle='round', **kw)

# ================================================================== A: microstates -> observables
cx, cy, rx, ry = 27, 42, 22, 26
pts = []
for i in range(-7, 8):
    for j in range(-9, 10):
        x = cx + 3.3 * i + (1.65 if j % 2 else 0); y = cy + 2.85 * j
        if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 1:
            pts.append((x + rng.normal(0, 0.4), y + rng.normal(0, 0.4)))
pts = np.array(pts)
for (i, j) in cKDTree(pts).query_pairs(4.2):
    if rng.uniform() < 0.5:
        a, b = (pts[i], pts[j]) if rng.uniform() < 0.5 else (pts[j], pts[i]); u = (b - a) / np.linalg.norm(b - a)
        arrow(a + 0.8 * u, b - 0.8 * u, color=FAINT, lw=0.4, ms=2.8, rad=0.3 * rng.choice([-1, 1]), z=1)
ax.scatter(pts[:, 0], pts[:, 1], s=3.2, color='#b9b8b3', zorder=2, lw=0)
# the slow observable W: a coarse, large cell; the fast observables mu and s: overlapping smaller cells
Wcell = Ellipse((27, 42), 38, 44, angle=10, fc=rgba(PURPLE, 0.035), ec=rgba(PURPLE, 0.75), lw=0.8, ls=(0, (3, 2)), zorder=3)
ax.add_patch(Wcell)
for (xy, w, h, ang, col) in [((22, 50), 17, 12, -25, BLUE), ((31, 43), 17, 11.5, 28, ORANGE), ((24, 34), 13, 9, 5, ORANGE)]:
    ax.add_patch(Ellipse(xy, w, h, angle=ang, fc=rgba(col, 0.10), ec=rgba(col, 0.9), lw=0.8, zorder=3))
label(15.2, 55.5, r'$\mu$', color=BLUE, fontsize=10); label(39.5, 48.2, r'$s$', color=ORANGE, fontsize=10)
label(9.5, 64, r'$W$', color=PURPLE, fontsize=10)

# ================================================================== B: bundle
OX, OY = 71, 7
def P3(x, y, z): return np.array([OX + x + 0.62 * y, OY + 0.42 * y + z])
def bz(x, y): return 2.6 * np.sin(x / 11 + 0.3) + 0.018 * (y - 11) ** 2 - 1.5 * np.cos(y / 7)
X0, X1, Y1 = 0, 42, 20
xs = np.linspace(X0, X1, 80); ys = np.linspace(0, Y1, 40)
edge = [P3(x, 0, bz(x, 0)) for x in xs] + [P3(X1, y, bz(X1, y)) for y in ys] + [P3(x, Y1, bz(x, Y1)) for x in xs[::-1]] + [P3(X0, y, bz(X0, y)) for y in ys[::-1]]
ax.add_patch(Polygon(edge, closed=True, fc=rgba(PURPLE, 0.07), ec=INK2, lw=0.6, zorder=1))
for y0 in np.linspace(3, Y1 - 3, 5): curve([P3(x, y0, bz(x, y0)) for x in xs], color=rgba(PURPLE, 0.18), lw=0.45, zorder=1)
for x0 in np.linspace(5, X1 - 3, 7): curve([P3(x0, y, bz(x0, y)) for y in ys], color=rgba(PURPLE, 0.18), lw=0.45, zorder=1)
yb = 9                                                   # fibers stand over a path in the base
fx = np.array([4, 12.5, 21, 29.5, 38]); TOP = 64
def fib(xf, z): return P3(xf + 1.1 * np.sin((z - bz(xf, yb)) / 14), yb, z)
curve([P3(x, yb, bz(x, yb)) for x in xs[4:-4]], color=PURPLE, lw=1.0, zorder=2)
def sec(x): return 30 + 10 * np.sin((x + 3) / 12)       # stationary section sigma_u
def sec0(x): return 19 + 4 * np.sin((x + 12) / 14)      # resting section sigma_0
for xf in fx:
    z0 = bz(xf, yb) + 0.3
    curve([fib(xf, z) for z in np.linspace(z0, TOP - 2.5 * np.cos(xf / 9), 40)], color=INK2, lw=0.7, zorder=3)
    ax.scatter(*P3(xf, yb, z0), s=7, color=PURPLE, zorder=4, lw=0)
xx = np.linspace(1.0, 40.5, 90)
curve([fib(x, sec0(x)) for x in xx], color=rgba(BLUE, 0.6), lw=0.8, ls=(0, (2.4, 1.8)), zorder=4)
curve([fib(x, sec(x)) for x in xx], color=BLUE, lw=1.4, zorder=4)
# the current state: a point on the fiber over W*, relaxing toward the section (perception)
XS = 21; zs = 55
ax.scatter(*fib(XS, zs), s=14, color=BLUE, zorder=7, lw=0)
arrow(fib(XS, zs - 1.2), fib(XS, sec(XS) + 1.2), color=BLUE, lw=1.3, ms=7, z=6)
# learning: motion of the structure along the section
seg = [fib(x, sec(x)) + np.array([0.6, 2.2]) for x in np.linspace(25, 36, 30)]
curve(seg[:-2], color=PURPLE, lw=1.3, zorder=6); arrow(seg[-3], seg[-1], color=PURPLE, lw=1.3, ms=7, z=6)
bs = [P3(x, yb, bz(x, yb)) + np.array([0, -2.6]) for x in np.linspace(25, 36, 30)]
curve(bs[:-2], color=PURPLE, lw=0.9, zorder=6, alpha=0.8); arrow(bs[-3], bs[-1], color=PURPLE, lw=0.9, ms=5, z=6, alpha=0.8)
# labels
label(*(P3(X1, Y1, bz(X1, Y1)) + np.array([3.2, 0.5])), r'$\mathcal{W}$', color=PURPLE, fontsize=11)
label(*(fib(fx[0], TOP - 2.5 * np.cos(fx[0] / 9)) + np.array([-3.6, 1.2])), r'$\mathcal{E}_W$', color=INK2, fontsize=10)
label(*(fib(40.5, sec(40.5)) + np.array([3.6, 0.2])), r'$\sigma_u$', color=BLUE, fontsize=10)
label(*(fib(40.5, sec0(40.5)) + np.array([3.6, -0.4])), r'$\sigma_0$', color=rgba(BLUE, 0.75), fontsize=10)
p = fib(XS, 47); label(p[0] - 1.6, p[1], 'perception', color=BLUE, fontsize=6.2, style='italic', ha='right')
q = seg[12]; label(q[0] - 0.5, q[1] + 3.0, 'learning', color=PURPLE, fontsize=6.2, style='italic')

# ================================================================== A -> B: funnels narrowing into the bundle
def funnel(top, bot, tip, color, alpha, bend=0.0):
    top, bot, tip = map(np.asarray, (top, bot, tip)); m = (top + bot) / 2
    c1 = m + (tip - m) * 0.55 + np.array([0, bend])
    verts = [top, c1 + (top - m) * 0.25, tip, c1 + (bot - m) * 0.25, bot, top]
    ax.add_patch(PathPatch(Path(verts, [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
                           fc=rgba(color, alpha), ec='none', zorder=2))
tipS = fib(XS, zs); tipW = P3(XS, yb, bz(XS, yb))
funnel((27, 56), (29, 45), tipS - np.array([1.2, 0]), BLUE, 0.10, bend=3)
funnel((38, 47), (35, 37.5), tipS - np.array([1.2, 0.4]), ORANGE, 0.07, bend=1)
funnel((44, 50), (41, 26), tipW - np.array([1.0, 0]), PURPLE, 0.06, bend=-3)
label(53, 60.5, 'fast', color=BLUE, fontsize=6.2, style='italic')
label(55, 18, 'slow', color=PURPLE, fontsize=6.2, style='italic')

# timescale separation: three bands of relaxation times, colours as above
ix, iy, iw, ih = 4, 3.5, 44, 6.5
ax.plot([ix, ix + iw], [iy, iy], color=INK2, lw=0.5); arrow((ix + iw - 0.5, iy), (ix + iw + 1.8, iy), color=INK2, lw=0.5, ms=3.5)
for (c0, wd, col, lab) in [(ix + 6, 3.0, '#b9b8b3', r'$\tau_x$'), (ix + 19.2, 3.2, ORANGE, None), (ix + 20.8, 3.2, BLUE, r'$\tau_z$'), (ix + 36, 3.0, PURPLE, r'$\tau_W$')]:
    t = np.linspace(c0 - 2.4 * wd, c0 + 2.4 * wd, 60); g = iy + ih * np.exp(-0.5 * ((t - c0) / (wd / 2)) ** 2)
    ax.fill_between(t, iy, g, color=col, alpha=0.3, lw=0); ax.plot(t, g, color=col, lw=0.8)
    if lab: label(c0 + 3.4, iy + ih - 0.8, lab, color=col if col != '#b9b8b3' else MUTED, fontsize=8, ha='left')
label(ix + iw + 3, iy, 'log relaxation time', fontsize=5.5, color=MUTED, style='italic', ha='left')

# ================================================================== C: centred tangent on the simplex (orthant view)
n_ = np.ones(3) / np.sqrt(3); u1 = np.array([1, -1, 0]) / np.sqrt(2); u2 = -np.array([1, 1, -2]) / np.sqrt(6)
SC, CX, CY = 24.0, 155.5, 56.0
def Pj(q):
    q = np.asarray(q, float); return np.array([CX + SC * (q @ u1), CY + SC * (0.42 * (q @ u2) + 0.95 * (q @ n_ - 1 / np.sqrt(3)))])
Or = Pj(np.zeros(3))
for e, lab_, off in zip(np.eye(3), [r'$x_1$', r'$x_2$', r'$x_3$'], [(-1.8, 0.2), (1.9, 0.2), (0.0, 1.9)]):
    curve([Or, Pj(1.22 * e)], color=FAINT, lw=0.5, zorder=0)
    label(*(Pj(1.22 * e) + np.array(off)), lab_, color=MUTED, fontsize=8)
tri = [Pj(v) for v in np.eye(3)]
ax.add_patch(Polygon(tri, closed=True, fc=rgba('#e7e6e1', 0.8), ec=INK2, lw=0.7, zorder=1))
for k in range(1, 5):
    t = k / 5
    for a_, b_ in [((t, 1 - t, 0), (t, 0, 1 - t)), ((1 - t, t, 0), (0, t, 1 - t)), ((1 - t, 0, t), (0, 1 - t, t))]:
        curve([Pj(a_), Pj(b_)], color='white', lw=0.55, zorder=1)
p0 = np.array([0.38, 0.34, 0.28]); f = np.array([0.1, 1.5, 1.2])
fam = lambda t: (lambda q: q / q.sum())(p0 * np.exp(t * f))
curve([Pj(fam(t)) for t in np.linspace(-1.5, 1.2, 80)], color=INK2, lw=0.8, zorder=3)
label(*(Pj(fam(-1.5)) + np.array([-1.2, -1.6])), r'$p_t$', color=INK2, fontsize=8)
sc_ = 1.1
Pp = Pj(p0); Praw = Pj(p0 + sc_ * p0 * f); Pcen = Pj(p0 + sc_ * p0 * (f - p0 @ f))
curve([Praw, Pcen], color=MUTED, lw=0.6, ls=':', zorder=5)
arrow(Pp, Praw, color=MUTED, lw=0.8, ms=5, ls=(0, (2.2, 1.6)), z=5)
arrow(Pp, Pcen, color=BLUE, lw=1.4, ms=7, z=6)
ax.scatter(*Pp, s=12, color=INK, zorder=7, lw=0)
label(*(Pp + np.array([1.2, 2.2])), r'$p$', fontsize=9)
label(*(Praw + np.array([-0.2, 2.0])), r'$f$', color=MUTED, fontsize=9)
label(*(Pcen + np.array([-0.5, -3.4])), r'$f-\mathbb{E}_p[f]$', color=BLUE, fontsize=8.5, ha='center')
m_ = (Praw + Pcen) / 2; label(m_[0] - 0.9, m_[1] + 1.0, r'$\mathbb{E}_p[f]$', color=MUTED, fontsize=7, ha='right')
label(135, 79.5, 'prediction error: a centred tangent', fontsize=6.6, color=INK2, ha='left')
label(135, 75.4, r'$\varepsilon_\mu=\mu-L(\theta)$', fontsize=8, color=BLUE, ha='left')

# ================================================================== D: Legendre pair, curvature = metric
ox, oy, sx, sy = 145, 5.0, 4.0, 2.0
psi = lambda w: 2.5 * np.log(np.cosh(1.1 * w)) + 0.2 * w
dpsi = lambda w: 2.75 * np.tanh(1.1 * w) + 0.2
d2 = lambda w: 3.025 / np.cosh(1.1 * w) ** 2
ww = np.linspace(-1.9, 4.0, 140); y_off = 1.3
toS = lambda w, y: np.array([ox + sx * w, oy + sy * y])
ax.plot([ox - 10, ox + 22], [oy, oy], color=INK2, lw=0.5); arrow((ox + 21, oy), (ox + 23.5, oy), color=INK2, lw=0.5, ms=3.5)
curve([toS(w, psi(w) + y_off) for w in ww], color=INK, lw=1.1, zorder=3)
w0 = 0.35; y0 = psi(w0) + y_off; sl = dpsi(w0)
curve([toS(w, y0 + sl * (w - w0)) for w in [w0 - 1.3, w0 + 3.3]], color=ORANGE, lw=0.9, zorder=4)
ax.scatter(*toS(w0, y0), s=12, color=INK, zorder=6, lw=0)
curve([toS(w0, 0), toS(w0, y0)], color=MUTED, lw=0.5, ls=':', zorder=2)
wa, wb = w0 + 1.7, w0 + 2.9
curve([toS(wa, y0 + sl * (wa - w0)), toS(wb, y0 + sl * (wa - w0)), toS(wb, y0 + sl * (wb - w0))], color=ORANGE, lw=0.6, zorder=4)
label(*(toS(wb, y0 + sl * (wa - w0 + 0.6)) + np.array([2.2, 0])), r'$C$', color=ORANGE, fontsize=9)
k_ = d2(w0) * sy / sx ** 2 / (1 + (sl * sy / sx) ** 2) ** 1.5; R = 1 / k_
nrm = np.array([-sl * sy / sx, 1.0]); nrm /= np.linalg.norm(nrm)
cc = toS(w0, y0) + R * nrm
ax.add_patch(Circle(cc, R, fc=rgba(PURPLE, 0.06), ec=rgba(PURPLE, 0.75), lw=0.6, ls=(0, (2, 1.5)), zorder=2))
label(*(cc + np.array([-R - 1.6, 0.6])), r'$g$', color=PURPLE, fontsize=9)
label(*(toS(w0, 0) + np.array([0, -2.2])), r'$W$', color=PURPLE, fontsize=9)
label(*(toS(4.0, psi(4.0) + y_off) + np.array([3.2, -1.0])), r'$\psi(W)$', fontsize=8.5)
label(135, 37.5, 'a coupling is dual to a correlation', fontsize=6.6, color=INK2, ha='left')
label(135, 33.4, r'$C=\partial_W\psi=\mathbb{E}[s\mu],\;\; g=\partial_W C$', fontsize=7.5, color=INK2, ha='left')

label(6, 79.5, 'a jump process and its coarse observables', fontsize=6.6, color=INK2, ha='left')
label(66, 79.5, 'a bundle of conditional ensembles', fontsize=6.6, color=INK2, ha='left')
for s_, x, y in [('A', 2, 79.5), ('B', 62, 79.5), ('C', 130.5, 79.5), ('D', 130.5, 37.5)]:
    ax.text(x, y, s_, fontsize=9, fontweight='bold', va='center')
import os
os.makedirs(OUT, exist_ok=True)
for ext in ('svg', 'pdf'): fig.savefig(os.path.join(OUT, 'fig1_theory.' + ext))
fig.savefig(os.path.join(OUT, 'fig1_theory.png'), dpi=300)
