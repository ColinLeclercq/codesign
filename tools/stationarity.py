#!/usr/bin/env python3
"""stationarity.py -- first-order stationarity of a design, done right.

What was wrong before.  The "balance residual" column of the stage-0 harness
read tools/check_optimality's `rel_grad`, which is (a) the gradient of ONE
branch, sigma_1 at one frequency -- the file's own comment says that is only
meaningful with a single active frequency -- and (b) evaluated at the peak
D.hinf finds when seeded from the rightmost eigenvalue, which on K+B+C is a
LOCAL peak at w = -1.01 (7.81), not the active one at -0.53 (7.964).  The
1.45e4 that the handoff's stage-1 gate targets is that number: one branch, at a
frequency that is not active.  It has no reason to vanish and does not move
when the design improves (7.964 -> 7.950 under sigma_2 enrollment: still 1.45e4).

What this computes.  The optimality condition of  min_theta max_w sigma_1(w;
theta)  is  0 in conv{ grad sigma_1(w_i; theta) : w_i active }.  Where sigma_1
and sigma_2 (nearly) coincide at an active w_i, sigma_1 is not differentiable
and its Clarke subdifferential contains grad sigma_2 too, so the convex hull
must include it.  The residual is

    rho = min_{lam >= 0, sum lam = 1} || sum_i lam_i g_i ||  *  ||z|| / gamma

in the optimiser's own coordinates z (Packer), dimensionless, with the g_i the
sigma_1 gradients at the active peaks and, where gap < --gap-tol, the sigma_2
gradients as well.  Reported both ways so the effect of the sigma_2 columns is
visible.

    python tools/stationarity.py --ne 200 --cases K+B+C,KSP+B+C,KGC+B
    python tools/stationarity.py --ne 200 --results results_polish_on12_ne200 --cases K+B+C
"""
from __future__ import annotations
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
import argparse, json, os
import numpy as np
from scipy.optimize import nnls

from fe_codesign import FECoDesign, parse_case
from packer import Packer
from ctrl_order import infer_r


def hull_residual(G):
    """min_{lam>=0, sum lam=1} ||G lam||  for columns g_i of G (real, stacked
    re/im).  Solved as nnls on [G; s*1^T] with a large s enforcing sum=1."""
    if G.shape[1] == 1:
        return float(np.linalg.norm(G[:, 0])), np.array([1.0])
    s = 1e3 * max(float(np.linalg.norm(G, axis=0).max()), 1e-30)
    A = np.vstack([G, s * np.ones((1, G.shape[1]))])
    bvec = np.concatenate([np.zeros(G.shape[0]), [s]])
    lam, _ = nnls(A, bvec, maxiter=10000)
    lam = lam / max(lam.sum(), 1e-30)
    return float(np.linalg.norm(G @ lam)), lam


def refine_peak(cl, w0, dw, arow, nvec, K, iters=30):
    lo, hi = w0 - dw, w0 + dw
    f = lambda w: cl.gain(float(w), arow, nvec, K, want_vectors=False)['sigma']
    for _ in range(iters):
        m1 = hi - 0.618 * (hi - lo); m2 = lo + 0.618 * (hi - lo)
        if f(m1) > f(m2):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results')
    ap.add_argument('--cases', default='K+B+C,KSP+B+C,KGC+B')
    ap.add_argument('--active-rtol', type=float, default=2e-3)
    ap.add_argument('--gap-tol', type=float, default=0.05,
                    help='add the sigma_2 gradient where (s1-s2)/s1 < this')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    r = infer_r(ne=a.ne, results=a.results, quiet=True)
    out = {}
    print(f"{'case':9s} {'gamma':>8s} {'#act':>4s} {'kinked':>6s} "
          f"{'rho(s1 only)':>13s} {'rho(+s2)':>10s}  lambda")
    for case in [c.strip() for c in a.cases.split(',') if c.strip()]:
        tag = case.replace('+', '_') + f"_s0_ne{a.ne}"
        rec = json.load(open(os.path.join(a.results, tag + '.json')))
        d = np.load(os.path.join(a.results, tag + '.npz'), allow_pickle=True)
        gamma = float(rec['gamma'])
        fl = parse_case(case)
        D = FECoDesign(ne=a.ne, r=r)
        xa, xs = float(d['xa']), float(d['xs'])
        va = d['va'] if ('va' in d.files and np.asarray(d['va']).size) else None
        vs = d['vs'] if ('vs' in d.files and np.asarray(d['vs']).size) else None
        b, c, pa, ps, na, ns = D.bc(xa, xs, va, vs)
        arow = (d['d'], d['e'], d['f']); nvec = d['nvec']; K = complex(d['K'])
        cl = D.cl(b, c)
        xia = d['xi_a'] if ('xi_a' in d.files and np.asarray(d['xi_a']).size) else None
        xis = d['xi_s'] if ('xi_s' in d.files and np.asarray(d['xi_s']).size) else None
        pk = Packer(D, fl, arow, nvec, K, xa, xs,
                    xia if xia is not None else D.gauss(xa),
                    xis if xis is not None else D.gauss(xs))
        z = pk.z0(); znorm = max(float(np.linalg.norm(z)), 1e-30)

        # active peaks from the record's working set, refined
        Om = np.asarray(d['active_freqs'], float)
        s1 = np.array([cl.gain(float(w), arow, nvec, K, want_vectors=False)['sigma'] for w in Om])
        cand = [i for i in range(len(Om)) if s1[i] >= gamma * (1 - a.active_rtol)]
        # merge candidates that refine to the same peak
        peaks = []
        dw = float(np.median(np.diff(np.sort(Om)))) if len(Om) > 1 else 0.05
        for i in cand:
            w = refine_peak(cl, float(Om[i]), dw, arow, nvec, K)
            if all(abs(w - p) > 0.5 * dw for p in peaks):
                peaks.append(w)
        cols1, cols2, blocks, info = [], [], [], []
        for w in peaks:
            pairs = cl.gain_k(w, arow, nvec, K, k=2)
            sv1, sv2 = pairs[0]['sigma'], pairs[1]['sigma']
            gap = (sv1 - sv2) / sv1
            G = lambda x_, z_: np.asarray(pk.grad(D.grads(
                cl, x_, z_, arow, nvec, K, xa, xs, pa, ps, na, ns, fl)), float)
            g11 = G(pairs[0]['x'], pairs[0]['z'])
            cols1.append(g11)
            if gap < a.gap_tol:
                g22 = G(pairs[1]['x'], pairs[1]['z'])
                # CROSS TERMS: the Clarke subdifferential of a singular value of
                # multiplicity 2 is {Re tr(Y U2^H dM V2) : Y >= 0, tr Y = 1};
                # e1e1^T and e2e2^T give grad sigma_1, grad sigma_2, and the
                # off-diagonal Y gives the mixed pairs (u1, v2), (u2, v1).
                g12 = G(pairs[1]['x'], pairs[0]['z'])   # right vector 2, left vector 1
                g21 = G(pairs[0]['x'], pairs[1]['z'])
                cols2.append(g22)
                blocks.append((g11, g22, g12, g21))
            else:
                blocks.append((g11, None, None, None))
            info.append(dict(omega=float(w), sigma1=float(sv1), sigma2=float(sv2), gap=float(gap)))
        G1 = np.column_stack(cols1)
        rho1, lam1 = hull_residual(G1)
        G2 = np.column_stack(cols1 + cols2) if cols2 else G1
        rho2, lam2 = hull_residual(G2)
        # ---- the actual test: min over lam (convex, across peaks) and, per
        # kinked peak, over Y = [[a, b], [conj b, 1-a]] >= 0 (|b|^2 <= a(1-a)) of
        # || sum_i lam_i g_i(Y_i) ||, with g(Y) = a g11 + (1-a) g22 + 2 Re(b) g12s
        # + 2 Im(b) g12a, where g12s/g12a are the symmetric / antisymmetric parts
        # of the mixed pair.  Small nonconvex problem in (lam, a_i, b_i); solved
        # by scipy from several starts.  rho_clarke <= rho2 always.
        from scipy.optimize import minimize
        def g_of(blk, a_, br, bi):
            g11, g22, g12, g21 = blk
            if g22 is None:
                return g11
            gs = 0.5 * (g12 + g21); ga = 0.5 * (g12 - g21)
            return a_ * g11 + (1 - a_) * g22 + 2 * br * gs + 2 * bi * ga
        nk = len(peaks)
        def unpack(x):
            lam = np.abs(x[:nk]) / max(np.abs(x[:nk]).sum(), 1e-30)
            Ys = []
            for j in range(nk):
                a_ = 0.5 * (1 + np.tanh(x[nk + 3 * j]))
                rad = np.sqrt(max(a_ * (1 - a_), 0.0))
                br = rad * np.tanh(x[nk + 3 * j + 1]) * np.cos(x[nk + 3 * j + 2])
                bi = rad * np.tanh(x[nk + 3 * j + 1]) * np.sin(x[nk + 3 * j + 2])
                Ys.append((a_, br, bi))
            return lam, Ys
        def obj(x):
            lam, Ys = unpack(x)
            v = sum(lam[j] * g_of(blocks[j], *Ys[j]) for j in range(nk))
            return float(np.linalg.norm(v))
        best = None
        rng = np.random.default_rng(0)
        for trial in range(12):
            x0 = np.concatenate([np.ones(nk) if trial == 0 else rng.uniform(0.1, 1, nk),
                                 rng.normal(0, 1, 3 * nk)])
            res = minimize(obj, x0, method='Nelder-Mead',
                           options=dict(maxiter=4000, xatol=1e-10, fatol=1e-14))
            if best is None or res.fun < best.fun:
                best = res
        lamc, Ysc = unpack(best.x)
        rho_c = best.fun * znorm / gamma
        rho1 *= znorm / gamma; rho2 *= znorm / gamma
        gnorm_ref = float(np.mean([np.linalg.norm(c) for c in cols1])) * znorm / gamma
        print(f"{case:9s} {gamma:8.4f} {len(peaks):4d} {len(cols2):6d} "
              f"{rho1:13.3e} {rho2:10.3e}  {np.round(lam2, 3).tolist()}")
        print(f"{'':9s} {'':8s}      rho_clarke (PSD Y over the 2x2 block) = {rho_c:.3e}"
              f"   = {rho_c / gnorm_ref:.2e} x mean branch-gradient norm ({gnorm_ref:.2e})")
        for j, (a_, br, bi) in enumerate(Ysc):
            if blocks[j][1] is not None:
                print(f"{'':9s} {'':8s}      peak {j}: lam={lamc[j]:.3f}  Y=[[{a_:.3f}, "
                      f"{br:+.3f}{bi:+.3f}i], [., {1 - a_:.3f}]]")
        for p in info:
            print(f"{'':9s} {'':8s}      w={p['omega']:+.4f} s1={p['sigma1']:.4f} "
                  f"s2={p['sigma2']:.4f} gap={p['gap']:.4f}")
        out[case] = dict(gamma=gamma, peaks=info, rho_sigma1=rho1, rho_with_sigma2=rho2,
                         rho_clarke=rho_c, gnorm_ref=gnorm_ref,
                         lam=lam2.tolist(), n_active=len(peaks), n_kinked=len(cols2))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1); print(f"-> {a.json}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
