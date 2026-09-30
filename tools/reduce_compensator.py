#!/usr/bin/env python3
"""reduce_compensator.py -- a MIMO start record with a compensator of lower
order (STAGE6 6.27, user 2026-09-30: "work with an order 5 compensator, then
orders 10 and 15").

K(s) = D_c + C_c (s I - A_c)^-1 B_c of the source record (order R) is reduced
to order r by CLOSED-LOOP WEIGHTED balanced truncation (Enns): the error
K - K_r enters the loop through M = (I - G K)^-1 G (from the plant input to the
measurement), so the reachability gramian of the compensator states is taken
in the cascade M -> K (the closed loop driven at the plant input, feeding the
compensator), the observability gramian from K alone.  Plain balanced
truncation of K was tried first: its error sits in the band and every order
<= 8 destabilised the loop (kbc_dec 5.339: max Re +1.52 at order 5).  The
reduced A_r is written in MODAL form (diagonal: the tridiagonal family with
e = f = 0); D_c, the fields and the positions are kept.  The tool reports the
closed-loop abscissa (dense, validation size) and a grid gamma; the driver
certifies the start anyway (anchor).

    python tools/reduce_compensator.py results_mimo_prod/prev_0e7f1c1/kbc_dec/K_B_C_s0_ne100_mimo2x2_decentral.npz \\
        --order 5 --out results_mimo_prod/start_kbc_cen5.npz
"""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import argparse, json
import numpy as np
import scipy.linalg as sl
from scipy.linalg import solve_continuous_lyapunov as lyap
import controller as ctl
from check_mimo import load, dense_parts


def weighted_bt(D, b, c, arow, nvec, K, r):
    R = len(arow[0]); p = np.asarray(K).shape[1]
    A = ctl.A_of(*arow[:3]); B = np.asarray(arow[3]).reshape(R, p); C = np.asarray(nvec).reshape(-1, R)
    Md, Acl, Mcl = dense_parts(D, b, c, arow, nvec, K)
    Mi = np.linalg.inv(Mcl)
    Aw = Mi @ Acl; Bw = Mi @ np.vstack([b, np.zeros((R, b.shape[1]))])     # w added to the plant input
    Cy = np.hstack([c.conj().T, np.zeros((c.shape[1], R))])               # y = c^H x_p
    N = Aw.shape[0]
    Ab = np.block([[Aw, np.zeros((N, R))], [B @ Cy, A]]); Bb = np.vstack([Bw, np.zeros((R, Bw.shape[1]))])
    P = lyap(Ab, -Bb @ Bb.conj().T)[N:, N:]; P = 0.5 * (P + P.conj().T)
    Q = lyap(A.conj().T, -C.conj().T @ C); Q = 0.5 * (Q + Q.conj().T)
    reg = lambda X: X + 1e-14 * np.trace(X).real / R * np.eye(R)
    Lp = sl.cholesky(reg(P), lower=True); Lq = sl.cholesky(reg(Q), lower=True)
    U, hsv, Vh = sl.svd(Lq.conj().T @ Lp)
    S = np.diag(hsv[:r] ** -0.5); T = Lp @ Vh[:r].conj().T @ S; Ti = S @ U[:, :r].conj().T @ Lq.conj().T
    lr, V = sl.eig(Ti @ A @ T); Vi = np.linalg.inv(V)
    return lr, Vi @ (Ti @ B), (C @ T) @ V, hsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rec'); ap.add_argument('--order', type=int, required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    z, D, b, c, arow, nvec, K = load(a.rec)
    r = a.order
    lr, Bm, Cm, hsv = weighted_bt(D, b, c, arow, nvec, K, r)
    ar = (lr, np.zeros(r - 1, complex), np.zeros(r - 1, complex), Bm)
    D.r = r
    Md, Acl, Mcl = dense_parts(D, b, c, ar, Cm, K)
    lam = sl.eigvals(Acl, Mcl); lam = lam[np.isfinite(lam)]
    cl = D.cl(b, c); ws = np.linspace(-30, 30, 1201)
    g = max(cl.gain(float(w), ar, Cm, K, want_vectors=False)['sigma'] for w in ws) if lam.real.max() < 0 else np.inf
    print(f"order {len(arow[0])} -> {r}: weighted Hankel singular values {np.array2string(hsv, precision=3)}")
    print(f"  compensator poles {np.array2string(lr, precision=3)}")
    print(f"  closed loop: max Re {lam.real.max():+.4f} (dense); grid gamma |w| <= 30: {g:.4f}")
    zz = dict(np.load(a.rec, allow_pickle=True))
    out = {k: v for k, v in zz.items() if k not in ('d', 'e', 'f', 'Bc', 'nvec', 'gamma', 'loop_orders')}
    out.update(d=lr, e=np.zeros(r - 1, complex), f=np.zeros(r - 1, complex), Bc=Bm, nvec=Cm,
               loop_orders=np.array([r // 2 + r % 2, r // 2]), gamma=np.float64(g))   # grid value; the driver certifies
    np.savez(a.out, **out)
    json.dump(dict(ne=int(json.load(open(a.rec[:-4] + '.json'))['ne']), gamma_grid=float(g), source=a.rec, order=r,
                   alpha_dense=float(lam.real.max())), open(a.out[:-4] + '.json', 'w'))
    print(f"-> {a.out}")


if __name__ == '__main__':
    main()
