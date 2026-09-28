#!/usr/bin/env python3
"""newton_polish.py -- note item 9 on the manifold (STAGE2 3.3).

    python tools/newton_polish.py --ne 100 --record R.npz --kkt kkt.json --case K+B+C

Unknowns: z (NZ), t, lambda_i (active peaks), mu_1..3 (the compensator pair
conditions).  The pair's conditions are written through SMOOTH quantities:
with M_s = (A_c + eps I)'s two rightmost eigenvalues l1, l2 (symmetric
functions of a separated cluster: smooth at the double pole),
    tau = l1 + l2,  p = l1 l2,    c1 + i c2 = tau^2 - 4 p = 0 (coalesced),
    c3 = Re tau = 0 (on the margin, both at Re = -eps).
Equations (square): grad_z L = sum lambda_i grad sigma_i + sum mu_k grad c_k
= 0; sigma_i(z) - t = 0; c_k = 0; sum lambda_i = 1.  Residual evaluated on
the reduced oracle (sigma rows, lifted pairs -> D.grads) and the r x r
compensator (finite differences for grad c_k); Jacobian by scipy's finite
differences (fsolve).  The result is checked by the FULL certificate and the
KKT residual outside.  Cost independent of n except the reduced evaluations.
"""
from __future__ import annotations
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse, json
import numpy as np
from scipy.optimize import fsolve, least_squares
from fe_codesign import FECoDesign, parse_case
from packer import Packer
from ctrl_order import infer_r
from reduced import ReducedOracle
import controller as ctl


def pair_conditions(Ac, eps):
    lam = np.linalg.eigvals(np.asarray(Ac, complex) + eps * np.eye(Ac.shape[0]))
    l = lam[np.argsort(-lam.real)[:2]]
    tau = l[0] + l[1]; p = l[0] * l[1]; disc = tau * tau - 4.0 * p
    return np.array([disc.real, disc.imag, tau.real])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=100); ap.add_argument('--record', required=True)
    ap.add_argument('--kkt', required=True, help='json from kkt_residual.py (active set and weights)')
    ap.add_argument('--case', default='K+B+C'); ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--results', default=None); ap.add_argument('--wmin', type=float, default=1e-3)
    ap.add_argument('--maxfev', type=int, default=4000); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    r = infer_r(ne=a.ne, results=a.results, quiet=True); D = FECoDesign(ne=a.ne, r=r); fl = parse_case(a.case)
    d = np.load(a.record, allow_pickle=True); xa, xs = float(d['xa']), float(d['xs'])
    aa = (d['d'], d['e'], d['f']); nn = d['nvec']; KK = complex(d['K'])
    pk = Packer(D, fl, aa, nn, KK, xa, xs, D.gauss(xa), D.gauss(xs)); z0 = pk.z0(); NZ = len(z0)
    kk = json.load(open(a.kkt)); wts = np.array(kk['weights']); fr = np.array(kk['active_freqs'])
    keep = wts > a.wmin; Wact = fr[keep]; lam0 = wts[keep] / wts[keep].sum(); na_ = len(Wact)
    print(f"active set from the KKT residual: {na_} peaks at {np.round(Wact, 3)} with weights {np.round(lam0, 3)}")
    b, c, *_ = D.bc(xa, xs, None, None); cl = D.cl(b, c)
    ro = ReducedOracle(D)
    for w in Wact:
        ro.enrich(cl, float(w), aa, nn, KK, nvec_dirs=3)
    al, lamcl, *_ = D.alpha_certified(aa, nn, KK, b, c, eps=a.eps); ro.enrich_eig(cl, [lamcl], aa, nn, KK)

    def sig_and_grads(z):
        arow, nvec, K, xa_, xs_, va_, vs_ = pk.unpack(z)
        b_, c_, pa, ps, na, ns = D.bc(xa_, xs_, va_, vs_)
        rc = ro.make_cl(b_, c_, D.r, D.mc)
        s = np.zeros(na_); G = np.zeros((na_, NZ))
        for i, w in enumerate(Wact):
            res = rc.gain(float(w), arow, nvec, K)
            s[i] = res['sigma']
            G[i] = np.asarray(pk.grad(D.grads(rc.proxy, res['x'], res['z'], arow, nvec, K, xa_, xs_, pa, ps, na, ns, fl)), float)
        return s, G

    def cond_and_grads(z):
        arow = pk.unpack(z)[0]; c0 = pair_conditions(ctl.A_of(*arow), a.eps)
        Gc = np.zeros((3, NZ)); h = 1e-6
        for j in range(NZ):
            ez = np.zeros(NZ); ez[j] = h
            cp = pair_conditions(ctl.A_of(*pk.unpack(z + ez)[0]), a.eps); cm = pair_conditions(ctl.A_of(*pk.unpack(z - ez)[0]), a.eps)
            Gc[:, j] = (cp - cm) / (2 * h)
        return c0, Gc

    s0, G0 = sig_and_grads(z0); c0, Gc0 = cond_and_grads(z0)
    # initial multipliers for the pair conditions: least squares on the stationarity equation
    rhs = -(G0.T @ lam0); mu0, *_ = np.linalg.lstsq(Gc0.T, rhs, rcond=None)
    x0 = np.concatenate([z0, [s0.max()], lam0, mu0])
    print(f"start: max sigma {s0.max():.6f} spread {np.ptp(s0):.2e}  pair conditions {np.round(c0, 6)}  |grad L| {np.linalg.norm(G0.T @ lam0 + Gc0.T @ mu0):.3e}")
    sz = np.maximum(np.abs(z0), 1.0)

    def F(x):
        z = x[:NZ]; t = x[NZ]; lam = x[NZ + 1:NZ + 1 + na_]; mu = x[NZ + 1 + na_:]
        s, G = sig_and_grads(z); cc, Gc = cond_and_grads(z)
        return np.concatenate([(G.T @ lam + Gc.T @ mu) / 10.0, s - t, cc * 10.0, [lam.sum() - 1.0]])

    res = least_squares(F, x0, method='lm', max_nfev=a.maxfev, xtol=1e-12, ftol=1e-12)
    x = res.x; z = x[:NZ]; t = x[NZ]; lam = x[NZ + 1:NZ + 1 + na_]
    s, G = sig_and_grads(z); cc, _ = cond_and_grads(z)
    print(f"after Newton/LM ({res.nfev} evaluations, status {res.status}): |F| {np.linalg.norm(res.fun):.3e}  max sigma (reduced) {s.max():.6f}  "
          f"spread {np.ptp(s):.2e}  pair conditions {np.round(cc, 8)}  lambda {np.round(lam, 3)}  |z-z0| {np.linalg.norm(z - z0):.3e}")
    # truth: full certificate at the polished design
    import run_case_epi as R
    pk2 = pk; zc = z
    arow, nvec, K, xa_, xs_, va_, vs_ = pk.unpack(zc); b_, c_, *_ = D.bc(xa_, xs_, va_, vs_)
    g, w, ok, pks = R.certified_norm(D, pk, zc, -30.0, 30.0, gam_lb=1e-8)
    al2, *_ = D.alpha_certified(arow, nvec, K, b_, c_, eps=a.eps)
    pol = np.sort(np.linalg.eigvals(ctl.A_of(*arow)).real)[::-1][:2]
    print(f"FULL certificate at the polished design: gamma {g:.6f} at w={w:+.3f} (start record gamma {float(d['gamma']):.6f}); alpha {al2:+.5f}; compensator poles {np.round(pol, 6)}")
    if a.out:
        np.savez(a.out, d=arow[0], e=arow[1], f=arow[2], nvec=nvec, K=K, xa=xa_, xs=xs_, gamma=g, alpha=al2,
                 active_freqs=np.asarray(d['active_freqs'], float), wlo=-30.0, whi=30.0)
        json.dump(dict(case=a.case, gamma=float(g), alpha=float(al2), note='newton polish'), open(a.out.replace('.npz', '.json'), 'w'))
        print("->", a.out)


if __name__ == '__main__':
    raise SystemExit(main())
