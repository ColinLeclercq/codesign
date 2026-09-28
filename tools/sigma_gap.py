#!/usr/bin/env python3
"""sigma_gap.py -- the sigma_1 - sigma_2 gap at the active frequencies of a design.

Handoff, stage 1 (item 3), first step: "measure the sigma_1 - sigma_2 gap at
every active frequency of the stalled rows from the dump -- one number each,
before writing any code."  The frozen records were not dumped, so the active
set is recovered here from the design itself: sigma_1(omega) is swept on a
fine grid over the band, its local maxima within --active-rtol of the record's
gamma are the active frequencies, and at each one sigma_2 is the second
eigenvalue of the same operator whose first eigenvalue is sigma_1^2
(closedloop_sparse.gain uses eigs(k=1); this uses k=2).

Why it matters: the epigraph reformulation constrains sigma_1(omega) <= t.
Where sigma_1 and sigma_2 coincide (or nearly), sigma_1 is not differentiable
-- the "kink" of the optimisation note -- and a smooth-constraint solver sees a
gradient that flips between the two branches.  A small gap at an active
frequency is the signature; enrolling sigma_2 <= t there (item 3) removes it.

    python tools/sigma_gap.py --ne 200 --cases K+B+C,KSP+B+C,KGC+B
    python tools/sigma_gap.py --ne 200 --cases K+B+C --results results_x --grid 1200
"""
from __future__ import annotations
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse.linalg as spla

from fe_codesign import FECoDesign, parse_case
from ctrl_order import infer_r


def sigmas(cl, omega, arow, nvec, K, k=2, tol=1e-9):
    """Top k singular values of the closed-loop map at omega."""
    n, r = cl.n, cl.r
    cl.setup(omega, arow, nvec, K)

    def Tv(v):
        return cl.solve(np.concatenate([v, np.zeros(r, complex)]))[:n]

    def THv(w):
        return cl.solveH(np.concatenate([w, np.zeros(r, complex)]))[:n]

    Wop = spla.LinearOperator((n, n), matvec=lambda v: cl.M @ THv(cl.M @ Tv(v)),
                              dtype=complex)
    lam = spla.eigs(Wop, k=k, which='LM', tol=tol, maxiter=5000,
                    return_eigenvectors=False)
    lam = np.sort(np.real(lam))[::-1]
    return np.sqrt(np.maximum(lam, 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results')
    ap.add_argument('--cases', default='K+B+C,KSP+B+C,KGC+B')
    ap.add_argument('--grid', type=int, default=800, help='sweep points over the band')
    ap.add_argument('--active-rtol', type=float, default=1e-3,
                    help='a local max of sigma_1 within this of gamma is active')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    r = infer_r(ne=a.ne, results=a.results, quiet=True)
    out = {}
    print(f"{'case':10s} {'gamma':>9s} {'omega':>9s} {'sigma1':>9s} {'sigma2':>9s} "
          f"{'gap':>9s} {'gap/s1':>8s}")
    for case in [c.strip() for c in a.cases.split(',') if c.strip()]:
        tag = case.replace('+', '_') + f"_s0_ne{a.ne}"
        rec = json.load(open(os.path.join(a.results, tag + '.json')))
        d = np.load(os.path.join(a.results, tag + '.npz'), allow_pickle=True)
        gamma = float(rec['gamma'])
        flags = parse_case(case)
        xa, xs = float(d['xa']), float(d['xs'])
        # fixed-field records store va/vs as EMPTY arrays, which bc() must see as None
        va = d['va'] if ('va' in d.files and np.asarray(d['va']).size) else None
        vs = d['vs'] if ('vs' in d.files and np.asarray(d['vs']).size) else None
        D = FECoDesign(ne=a.ne, r=r)
        b, c, *_ = D.bc(xa, xs, va, vs)
        arow = (d['d'], d['e'], d['f']); nvec = d['nvec']; K = complex(d['K'])
        cl = D.cl(b, c)
        Om = np.asarray(d['active_freqs'], float) if 'active_freqs' in d.files else None
        t0 = time.time()
        if Om is not None and Om.size:
            # the record's working set Omega: sigma_1 on it, the points within
            # rtol of gamma are the active frequencies
            s1 = np.array([sigmas(cl, w, arow, nvec, K, k=1)[0] for w in Om])
            act = [i for i in range(len(Om)) if s1[i] >= gamma * (1 - a.active_rtol)]
            grid = Om
        else:
            grid = np.linspace(wlo, whi, a.grid)
            s1 = np.array([sigmas(cl, w, arow, nvec, K, k=1)[0] for w in grid])
            act = [i for i in range(1, len(grid) - 1)
                   if s1[i] >= s1[i - 1] and s1[i] >= s1[i + 1]
                   and s1[i] >= gamma * (1 - a.active_rtol)]
        rows = []
        for i in act:
            w = float(grid[i])
            sv = sigmas(cl, w, arow, nvec, K, k=2)
            rows.append(dict(omega=w, sigma1=float(sv[0]), sigma2=float(sv[1]),
                             gap=float(sv[0] - sv[1]),
                             gap_rel=float((sv[0] - sv[1]) / sv[0])))
            print(f"{case:10s} {gamma:9.4f} {w:9.4f} {sv[0]:9.4f} {sv[1]:9.4f} "
                  f"{sv[0]-sv[1]:9.4f} {(sv[0]-sv[1])/sv[0]:8.4f}")
        # and the three largest non-active peaks, for context
        order = np.argsort(s1)[::-1]
        ctx = [int(i) for i in order if i not in act][:3]
        for i in ctx:
            print(f"{'':10s} {'':9s} {grid[i]:9.4f} {s1[i]:9.4f}   (next-highest, not active)")
        if not rows:
            print(f"{case:10s} {gamma:9.4f}   (no local maximum within rtol of gamma; "
                  f"grid max {s1.max():.4f})")
        out[case] = dict(gamma=gamma, n_active_record=rec.get('n_active'),
                         active=rows, sweep_s=round(time.time() - t0, 1),
                         grid_max=float(s1.max()))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)
        print(f"-> {a.json}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
