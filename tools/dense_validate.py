#!/usr/bin/env python3
"""dense_validate.py -- independent DENSE check of a GL design (2026-09-25).

At GL sizes (n ~ 200-400) the closed loop fits in dense linear algebra, so a
design can be checked without any of the loop's machinery:
  * the closed-loop spectral abscissa from a dense generalized eigenvalue solve
    (D.alpha_dense), against the line -eps;
  * the compensator poles (r x r);
  * the H-infinity norm from the dense level-set (Boyd-Balakrishnan /
    Bruinsma-Steinbuch) certificate on the FULL matrices, plus a 20001-point
    sweep of sigma_1 on the band as a sanity check.
    python tools/dense_validate.py --ne 100 --case K+B+C --record results_n100_kbc_cont/K_B_C_s0_ne100.npz
"""
from __future__ import annotations
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse, time
import numpy as np
from fe_codesign import FECoDesign, parse_case
from ctrl_order import infer_r
import controller as ctl, certify_sparse as csp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=100); ap.add_argument('--record', required=True)
    ap.add_argument('--case', default='K+B+C'); ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--wlo', type=float, default=-30.0); ap.add_argument('--whi', type=float, default=30.0)
    ap.add_argument('--field-basis-n', type=int, default=0)
    a = ap.parse_args()
    d = np.load(a.record, allow_pickle=True)
    D = FECoDesign(ne=a.ne, r=len(d['d']))                 # the record's controller order
    aa = (d['d'], d['e'], d['f']); nn = d['nvec']; KK = complex(d['K']); xa, xs = float(d['xa']), float(d['xs'])
    va = d['va'] if 'va' in d.files and np.size(d['va']) else None
    vs = d['vs'] if 'vs' in d.files and np.size(d['vs']) else None
    b, c, *_ = D.bc(xa, xs, va, vs); cl = D.cl(b, c)
    t0 = time.time(); al, lam = D.alpha_dense(aa, nn, KK, b, c)[:2]
    pc = np.linalg.eigvals(ctl.A_of(*aa))
    print(f"closed-loop abscissa (dense): {al:+.6f} at {complex(lam):.4f}  (line {-a.eps:+.4f}: "
          f"{'OK' if al <= -a.eps + 1e-9 else 'VIOLATED'}, margin {-a.eps - al:+.2e})  [{time.time() - t0:.1f} s]")
    # the rightmost PAIR from its sum and product (exact at a double pole, whose
    # computed eigenvalues split by ~sqrt(eps ||A_c||)), the rest from eig
    import hurwitz as hw
    U2, V2, lam2 = hw.cluster_block(ctl.A_of(*aa), 2, pick=hw.pick_pair(ctl.A_of(*aa)))
    B2 = U2.conj().T @ ctl.A_of(*aa) @ V2; s2 = np.trace(B2); p2 = np.linalg.det(B2)
    rt = np.sqrt(s2 * s2 / 4 - p2 + 0j); pair_max = float(s2.real / 2 + abs(rt.real))
    rest = [x for x in pc if min(abs(x - l) for l in lam2) > 1e-6]
    rest_max = max((float(x.real) for x in rest), default=-np.inf)
    cmax = max(pair_max, rest_max)
    print(f"compensator: rightmost pair (from its sum/product) max Re {pair_max:+.10f}, |l1-l2| {2 * abs(rt):.1e}; "
          f"other poles max Re {rest_max:+.10f}  ({'OK' if cmax <= -a.eps + 1e-8 else 'VIOLATED'}; eig alone: {pc.real.max():+.10f})")
    sig = lambda w: cl.gain(float(w), aa, nn, KK, want_vectors=False)['sigma']
    t0 = time.time()
    ws = np.linspace(a.wlo, a.whi, 20001); gs = np.array([sig(w) for w in ws]); j = int(np.argmax(gs))
    print(f"sweep (20001 pts on [{a.wlo}, {a.whi}]): max {gs[j]:.6f} at {ws[j]:+.4f}  [{time.time() - t0:.0f} s]")
    import scipy.sparse as sp
    t0 = time.time()
    Ac = ctl.A_of(*aa); Bc = np.full(D.r, D.mc, complex)
    g, w, nit, ok = csp.hinf_dense_levelset(sp.csc_matrix(D.A), sp.csc_matrix(D.M), b, c, Ac, Bc, nn, KK, sig,
                                            gam_lb=float(gs[j]), wlo=a.wlo, whi=a.whi, rtol=1e-8)
    print(f"dense level-set certificate on the full model: {g:.8f} at {w:+.5f}, certified={ok} ({nit} levels, "
          f"{time.time() - t0:.1f} s)")
    if 'gamma' in d.files:
        print(f"recorded gamma: {float(d['gamma']):.8f}  (rel diff {abs(g - float(d['gamma'])) / g:.1e})")


if __name__ == '__main__':
    main()
