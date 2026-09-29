#!/usr/bin/env python3
"""dual_design.py -- the actuator/sensor DUAL of a MIMO design and its gamma
(STAGE6 6.15).

The Ginzburg-Landau operator satisfies A^T = P A P and M = P M P, P the
reflection x -> -x of the nodes (checked to 5e-15).  Then R(s)^T = P R(s) P for
R = (sM - A)^-1, and the design
    B' = P conj(C),   C' = P conj(B),   K'(s) = K(s)^T
(actuator j <- the reflected, conjugated field of sensor j and vice versa; the
compensator transposed: A_c^T, B_c' = C_c^T, C_c' = B_c^T, D_c' = D_c^T) has the
transposed closed-loop map up to P, hence the SAME singular values at every w:
gamma is invariant.  A design is SELF-DUAL when it equals its dual (SISO:
x_a = -x_s, psi_a = conj(P psi_s)).  This tool builds the dual of a record and
certifies both gammas over all w (dense level set, validation size).

    python tools/dual_design.py results_mimo_prod/kbc_cen/K_B_C_s0_ne100_mimo2x2_central.npz
"""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import numpy as np
import controller as ctl
import certify_sparse as csp
from check_mimo import load


def gamma_all_w(D, b, c, arow, nvec, K):
    cl = D.cl(b, c)
    sof = lambda w: cl.gain(float(w), arow, nvec, K, want_vectors=False)['sigma']
    g0 = max(sof(w) for w in np.linspace(-3, 3, 61))
    return csp.hinf_dense_levelset(D.A, D.M, b, c, ctl.A_of(*arow[:3]), np.asarray(arow[3]), nvec, K, sof,
                                   0.95 * g0, wlo=-np.inf, whi=np.inf, rtol=1e-8)[:2]


def dual(D, b, c, arow, nvec, K):
    Pn = np.arange(D.n)[::-1]
    b2 = np.conj(c)[Pn]; c2 = np.conj(b)[Pn]
    d, e, f, Bc = arow
    arow2 = (d, f, e, np.asarray(nvec).T.copy())          # A_c^T: e and f swap; B_c' = C_c^T
    return b2, c2, arow2, np.asarray(Bc).T.copy(), np.asarray(K).T.copy()


def self_duality(D, b, c):
    """min over port permutations pi of || C - P conj(B) Pi || / || C ||, with a
    unit phase per port allowed (fields are defined up to one: the gauge);
    0 for a self-dual set of ports."""
    from itertools import permutations
    Pn = np.arange(D.n)[::-1]
    B2 = np.conj(np.atleast_2d(b.T).T)[Pn]; C = np.atleast_2d(c.T).T
    best = np.inf
    for pi in permutations(range(B2.shape[1])):
        X = B2[:, list(pi)]
        ph = np.array([np.vdot(X[:, j], C[:, j]) for j in range(C.shape[1])])
        X = X * (ph / np.maximum(np.abs(ph), 1e-300))
        best = min(best, float(np.linalg.norm(C - X) / np.linalg.norm(C)))
    return best


if __name__ == '__main__':
    for rec in _s.argv[1:]:
        z, D, b, c, arow, Cc, Dc = load(rec)
        g, w = gamma_all_w(D, b, c, arow, Cc, Dc)
        b2, c2, arow2, Cc2, Dc2 = dual(D, b, c, arow, Cc, Dc)
        g2, w2 = gamma_all_w(D, b2, c2, arow2, Cc2, Dc2)
        print(f"{rec}: gamma {g:.10f} (w {w:+.4f}), dual {g2:.10f} (w {w2:+.4f}), rel {abs(g2 - g) / g:.1e}; "
              f"distance to self-dual ||C - P conj(B)||/||C|| = {self_duality(D, b, c):.3e}; "
              f"||D_c - D_c^T|| / ||D_c|| = {np.linalg.norm(Dc - Dc.T) / np.linalg.norm(Dc):.3e}", flush=True)
