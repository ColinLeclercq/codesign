"""
test_eigcount.py — does the argument-principle count match a dense eigensolve?

The count must be right on designs that are stable, on designs that are NOT,
and in particular on the failure mode that motivated it: an eigenvalue sitting
to the RIGHT of the line, which the line certificate cannot see.

    python test_eigcount.py --ne 80
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, sys, time
import numpy as np
from scipy.linalg import eig
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_codesign, controller as ctl, eigcount


def dense_count(D, arow, nvec, Kfb, b, c, eps):
    n, r = D.n, len(np.atleast_1d(arow[0]))
    Ad = np.asarray(D.A.todense()); Md = np.asarray(D.M.todense())
    Ac = ctl.A_of(*arow); Bc = np.full(r, 1.0, complex)
    Acl = np.zeros((n + r, n + r), complex); Mcl = np.zeros((n + r, n + r), complex)
    Acl[:n, :n] = Ad + Kfb * np.outer(b, np.conj(c))
    Acl[:n, n:] = b.reshape(-1, 1) @ np.asarray(nvec).reshape(1, r)
    Acl[n:, :n] = Bc.reshape(-1, 1) @ np.conj(c).reshape(1, n)
    Acl[n:, n:] = Ac
    Mcl[:n, :n] = Md; Mcl[n:, n:] = np.eye(r)
    lam = eig(Acl, Mcl, right=False)
    lam = lam[np.isfinite(lam)]
    return int((lam.real > -eps).sum()), float(lam.real.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=80)
    ap.add_argument('--r', type=int, default=7)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--R', type=float, default=60.0)
    ap.add_argument('--nodes', type=int, default=48)
    ap.add_argument('--probes', type=int, default=24)
    a = ap.parse_args()
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=0.8)
    r = a.r
    b, c, *_ = D.bc(-1.03, 0.98)
    base_d = np.linspace(-1., -4., r).astype(complex)
    e = 0.15 * np.ones(r - 1, complex); f = 0.10 * np.ones(r - 1, complex)
    nv = 1e-4 * np.linspace(1., 0.2, r).astype(complex)

    cases = [
        ('stable, weak coupling', base_d, nv),
        ('one pole at Re=+0.4 (RIGHT of the line -- the failure mode)',
         np.concatenate(([0.4 + 0j], base_d[1:])), nv),
        ('two poles right of the line',
         np.concatenate(([0.4 + 0j, 0.9 + 0j], base_d[2:])), nv),
        ('pole just inside the margin (Re=-0.02 > -eps)',
         np.concatenate(([-0.02 + 0j], base_d[1:])), nv),
        ('strong coupling', base_d, np.linspace(1., 0.2, r).astype(complex)),
    ]
    print(f"ne={a.ne} n={D.n} r={r} eps={a.eps}  R={a.R}  "
          f"nodes={a.nodes} probes={a.probes}\n")
    print(f"{'case':52s} {'dense':>6s} {'count':>6s} {'resid':>7s} "
          f"{'sec':>6s}  verdict")
    nbad = 0
    for name, dd, nn in cases:
        arow = (dd, e, f)
        nd, amax = dense_count(D, arow, nn, 0j, b, c, a.eps)
        cl = D.cl(b, c)
        t0 = time.time()
        nc, res = eigcount.count_right_of(cl, arow, nn, 0j, a.eps, R=a.R,
                                          nodes=a.nodes, probes=a.probes)
        dt = time.time() - t0
        ok = (nc == nd)
        if not ok:
            nbad += 1
        print(f"{name:52s} {nd:6d} {nc:6d} {res:7.3f} {dt:6.2f}  "
              f"{'MATCH' if ok else '*** MISMATCH ***'}  (alpha={amax:+.4f})")
    print()
    print("ALL COUNTS MATCH" if nbad == 0 else f"{nbad} MISMATCH(ES)")
    print("\nThe second and third rows are the point: an eigenvalue to the RIGHT")
    print("of Re = -eps is invisible to the line certificate, which is how three")
    print("designs were reported stable when they were not.")
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
