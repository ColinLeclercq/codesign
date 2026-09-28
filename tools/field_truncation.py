#!/usr/bin/env python3
"""field_truncation.py -- how much of an optimised FB/FC field matters (2026-09-25).
Project the fields of a record onto the Laplacian basis the run used, keep the
lowest N modes (or damp them with a Sobolev filter 1/(1 + l^2 lambda_k)), and
recompute the certified H-infinity norm with everything else fixed (dense
level-set certificate on the full model: GL sizes only)."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import numpy as np, scipy.sparse as sp
from fe_codesign import FECoDesign
import fieldbasis as FB, controller as ctl, certify_sparse as csp
rec = _s.argv[1]; ne = int(_s.argv[2]) if len(_s.argv) > 2 else 100
d = np.load(rec, allow_pickle=True); D = FECoDesign(ne=ne, r=len(d['d']))
aa = (d['d'], d['e'], d['f']); nn = d['nvec']; KK = complex(d['K']); xa, xs = float(d['xa']), float(d['xs'])
Phi, lam = FB.laplace_basis(D, -38.9, 39.2, 80); M = D.M.tocsc()
def gamma(va, vs):
    b, c, *_ = D.bc(xa, xs, va, vs); cl = D.cl(b, c)
    sig = lambda w: cl.gain(float(w), aa, nn, KK, want_vectors=False)['sigma']
    ws = np.linspace(-30, 30, 1201); g0 = max(sig(w) for w in ws)
    g, w, _, ok = csp.hinf_dense_levelset(sp.csc_matrix(D.A), sp.csc_matrix(D.M), b, c, ctl.A_of(*aa),
                                          np.full(D.r, D.mc, complex), nn, KK, sig, gam_lb=g0, wlo=-30, whi=30, rtol=1e-8)
    return g
va0, vs0 = np.asarray(d['va'], complex), np.asarray(d['vs'], complex)
ba, ea = FB.project(Phi, M, va0); bs_, es = FB.project(Phi, M, vs0)
print(f"projection error of the stored fields onto 80 modes: {ea:.1e} / {es:.1e}")
e_a = np.abs(ba) ** 2 / np.sum(np.abs(ba) ** 2); e_s = np.abs(bs_) ** 2 / np.sum(np.abs(bs_) ** 2)
# (2026-09-28: the basis is in eigenvalue order since fieldbasis._m_orth was fixed; before, these bands were meaningless)
for lo_, hi_ in ((0, 20), (20, 40), (40, 60), (60, 80)):
    print(f"  modes {lo_:2d}-{hi_:2d} (lambda {lam[lo_]:.2f}-{lam[hi_ - 1]:.2f}): energy actuator {e_a[lo_:hi_].sum():.2e}, sensor {e_s[lo_:hi_].sum():.2e}")
print(f"RMS wavenumber sqrt(sum lambda |beta|^2 / sum |beta|^2): actuator {np.sqrt(e_a @ lam):.3f}, sensor {np.sqrt(e_s @ lam):.3f}")
g_ref = gamma(va0, vs0); print(f"gamma, stored fields: {g_ref:.6f}")
for N in (60, 40, 30, 20):
    g = gamma(Phi[:, :N] @ ba[:N], Phi[:, :N] @ bs_[:N]); print(f"gamma, lowest {N} modes: {g:.6f} ({100 * (g / g_ref - 1):+.3f} %)")
for ell in (0.5, 1.0, 2.0):
    f = 1.0 / (1.0 + ell ** 2 * lam)
    g = gamma(Phi @ (f * ba), Phi @ (f * bs_)); print(f"gamma, Sobolev filter l = {ell}: {g:.6f} ({100 * (g / g_ref - 1):+.3f} %)")
# the TAILS in physical space: a smooth window (1 inside |x - centre| < a, cosine taper of width 3)
x = np.asarray(D.xn, float)
def win(a, c=0.0):
    r = np.abs(x - c); w = np.ones_like(r); t = (r > a) & (r < a + 3); w[t] = 0.5 * (1 + np.cos(np.pi * (r[t] - a) / 3)); w[r >= a + 3] = 0.0
    return w
for a in (15.0, 10.0, 7.0):
    g = gamma(win(a) * va0, win(a) * vs0); print(f"gamma, fields windowed to |x| < {a:g} (+3 taper): {g:.6f} ({100 * (g / g_ref - 1):+.3f} %)")
tail = lambda v: np.sqrt(np.sum(np.abs(v[np.abs(x) > 12]) ** 2) / np.sum(np.abs(v) ** 2))
print(f"tail fraction (|x| > 12, nodal l2): actuator {tail(va0):.3f}, sensor {tail(vs0):.3f}")
# is the GRID-SCALE ripple itself doing work?  Smooth the fields with a Gaussian
# kernel (width s) everywhere / only in the tails, keeping the rest
def smooth(v, s):
    K_ = np.exp(-0.5 * ((x[:, None] - x[None, :]) / s) ** 2); K_ /= K_.sum(1, keepdims=True); return K_ @ v
for s_ in (0.5, 1.0):
    for where in ('tails', 'all'):
        m_ = (np.abs(x) > 12) if where == 'tails' else np.ones_like(x, bool)
        va_ = np.where(m_, smooth(va0, s_), va0); vs_ = np.where(m_, smooth(vs0, s_), vs0)
        g = gamma(va_, vs_); print(f"gamma, Gaussian smoothing s = {s_} ({where}): {g:.6f} ({100 * (g / g_ref - 1):+.3f} %)")
