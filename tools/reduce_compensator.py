#!/usr/bin/env python3
"""reduce_compensator.py -- a MIMO start record with a compensator of lower
order (STAGE6 6.27, user 2026-09-30: "work with an order 5 compensator, then
orders 10 and 15").

K(s) = D_c + C_c (s I - A_c)^-1 B_c of the source record (A_c tridiagonal,
order R) is reduced to order r by BALANCED TRUNCATION (A_c Hurwitz: the
compensator poles are constrained to Re <= 0 in the loop; a pole on or right of
the axis stops the tool), then written in MODAL form -- A_r = diag(eigenvalues),
e = f = 0, which is the tridiagonal family with zero off-diagonals.  D_c, the
fields and the positions are kept.  The driver certifies the start (anchor), so
the reduced design is checked before any round; this tool reports the
compensator error sup_w ||K - K_r|| and the Hankel singular values.

    python tools/reduce_compensator.py results_mimo_prod/kbc_dec/K_B_C_s0_ne100_mimo2x2_decentral.npz \\
        --order 5 --out results_mimo_prod/start_kbc_cen5.npz
"""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, cholesky, svd, eig
import controller as ctl


def balanced_truncation(A, B, C, r):
    """Square-root balanced truncation of a stable (complex) realisation."""
    P = solve_continuous_lyapunov(A, -B @ B.conj().T)
    Q = solve_continuous_lyapunov(A.conj().T, -C.conj().T @ C)
    P = 0.5 * (P + P.conj().T); Q = 0.5 * (Q + Q.conj().T)
    Lp = cholesky(P + 1e-300 * np.eye(len(A)), lower=True)
    Lq = cholesky(Q + 1e-300 * np.eye(len(A)), lower=True)
    U, hsv, Vh = svd(Lq.conj().T @ Lp)
    S = np.diag(hsv[:r] ** -0.5)
    T = Lp @ Vh[:r].conj().T @ S               # right projection (n x r)
    Ti = S @ U[:, :r].conj().T @ Lq.conj().T    # left projection (r x n)
    return Ti @ A @ T, Ti @ B, C @ T, hsv


def kfreq(A, B, C, D, w):
    return D + C @ np.linalg.solve(1j * w * np.eye(len(A)) - A, B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rec'); ap.add_argument('--order', type=int, required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    z = dict(np.load(a.rec, allow_pickle=True))
    A = ctl.A_of(np.asarray(z['d'], complex), np.asarray(z['e'], complex), np.asarray(z['f'], complex))
    R = len(A); m, p = np.asarray(z['K']).shape
    B = np.asarray(z['Bc'], complex).reshape(R, p); C = np.asarray(z['nvec'], complex).reshape(m, R)
    D = np.asarray(z['K'], complex).reshape(m, p)
    lam = np.linalg.eigvals(A)
    if np.max(lam.real) >= 0:
        raise SystemExit(f"compensator not Hurwitz (max Re {np.max(lam.real):+.3e}): balanced truncation undefined")
    Ar, Br, Cr, hsv = balanced_truncation(A, B, C, a.order)
    lr, T = eig(Ar)
    Ti = np.linalg.inv(T)
    Bm, Cm = Ti @ Br, Cr @ T
    ws = np.concatenate([-np.logspace(-3, 3, 600)[::-1], np.logspace(-3, 3, 600)])
    err = max(np.linalg.norm(kfreq(A, B, C, D, w) - kfreq(np.diag(lr), Bm, Cm, D, w), 2) for w in ws)
    kn = max(np.linalg.norm(kfreq(A, B, C, D, w), 2) for w in ws)
    print(f"order {R} -> {a.order}: Hankel singular values {np.array2string(hsv, precision=3)}")
    print(f"  kept poles {np.array2string(lr, precision=3)}")
    print(f"  sup_w ||K - K_r|| = {err:.3e} (sup_w ||K|| = {kn:.3e}, 2 x tail bound {2 * hsv[a.order:].sum():.3e})")
    out = {k: v for k, v in z.items() if k not in ('d', 'e', 'f', 'Bc', 'nvec', 'gamma', 'loop_orders')}
    out.update(d=lr, e=np.zeros(a.order - 1, complex), f=np.zeros(a.order - 1, complex), Bc=Bm, nvec=Cm,
               loop_orders=np.array([a.order // 2 + a.order % 2, a.order // 2]),
               gamma=np.float64(z['gamma']))            # a placeholder, as in mimo_start.py: the driver certifies its start
    np.savez(a.out, **out)
    print(f"-> {a.out}")


if __name__ == '__main__':
    main()
