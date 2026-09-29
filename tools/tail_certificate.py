#!/usr/bin/env python3
"""tail_certificate.py -- make a record's certified H-infinity value GLOBAL in
frequency (2026-09-26, STAGE5 30; closes "analytic tail bound", STAGE1 8).

    python tools/tail_certificate.py --ne 100 --record results_n100_kbc/K_B_C_s0_ne100.npz

The loop's certificate proves sup sigma_1 over the band [wlo, whi] only.  For
|omega| >= W this tool bounds the closed-loop gain analytically, and certifies
the band [-W, W] with the same sparse level-set procedure, so that together
they prove sup over ALL omega.

The gain is sigma_1 = || M^{1/2} T_cl(i w) M^{1/2} ||_2, T_cl the closed-loop
resolvent restricted to the plant (ClosedLoopResolvent.gain).  With
R = (sM - A)^{-1}, G = c^H R b and the compensator k(s) = K + n^T (sI - A_c)^{-1} B_c,
    T_cl = R + R b k (1 - k G)^{-1} c^H R          (rank-one loop, Sherman-Morrison)
and, with nu = || M^{-1/2} A M^{-1/2} ||_2 (Lanczos on the pencil (A^H M^-1 A, M): sparse products
and a sparse LU of M only -- no square root or Cholesky factor of M),
beta = || M^{-1/2} b ||, gamma_c = || M^{-1/2} c ||, a_c = ||A_c||_2, for |s| > nu:
    || M^{1/2} R M^{1/2} || <= 1 / (|s| - nu),
    |G| <= beta gamma_c / (|s| - nu),
    |k| <= |K| + ||n|| ||B_c|| / (|s| - a_c)          (|s| > a_c; Neumann, no normality assumed),
    sigma_1 <= 1/(|s| - nu) + |k| beta gamma_c / ((|s| - nu)^2 (1 - |k||G|))   when |k||G| < 1.
Every factor decreases in |s| = |w| on the imaginary axis, so the bound at W
holds for all |w| >= W.  W is doubled from 2 nu until the bound is below
gamma / 10.  The band is certified by the DENSE level-set algorithm by default
(all eigenvalues of the even pencil; at GL sizes a second of work): the sparse
procedure's shift-invert disks can fail far out -- at ne=200, w = 278, ARPACK
converged no eigenvalue for any k / subspace / budget (STAGE5 30.5).  At GL sizes only (the dense norm); at the cavity nu grows like
1/h^2 and the tail must come from the operator's own structure (numerical
range) -- a cavity-readiness item."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse, json, time
import numpy as np
import scipy.sparse.linalg as spla
from scipy.linalg import svdvals
from fe_codesign import FECoDesign
from ctrl_order import infer_r
import controller as ctl, certify_sparse as csp


def tail_bound(W, nu, beta, gc, K, nrm_n, nrm_B, a_c):
    if W <= max(nu, a_c):
        return np.inf
    kmax = abs(K) + nrm_n * nrm_B / (W - a_c)
    Gmax = beta * gc / (W - nu)
    if kmax * Gmax >= 1.0:
        return np.inf
    return 1.0 / (W - nu) + kmax * beta * gc / ((W - nu) ** 2 * (1.0 - kmax * Gmax))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=100)
    ap.add_argument('--record', required=True)
    ap.add_argument('--rtol', type=float, default=1e-6)
    ap.add_argument('--json', default=None)
    ap.add_argument('--disk', choices=['strict', 'frozen'], default='strict')
    ap.add_argument('--method', choices=['dense', 'sparse'], default='dense',
                    help='dense: all eigenvalues of the even pencil at each level (GL sizes; nothing can be missed); '
                         'sparse: the loop\'s shift-invert procedure')
    a = ap.parse_args()
    z = np.load(a.record, allow_pickle=True)
    r = len(z['d'])
    D = FECoDesign(ne=a.ne, r=r)
    arow = (z['d'], z['e'], z['f']); nvec = np.asarray(z['nvec'], complex); K = complex(z['K'])
    va = z['va'] if 'va' in z.files and np.asarray(z['va']).size else None
    vs = z['vs'] if 'vs' in z.files and np.asarray(z['vs']).size else None
    b, c, *_ = D.bc(float(z['xa']), float(z['xs']), va, vs)
    # 2026-09-28 (user: never a square root or a Cholesky factor of M): the
    # constants from products with the sparse A, M and solves with a sparse LU
    # of M (factored once per mesh, same sparsity class as the resolvent's LU):
    #   nu^2 = lambda_max of the Hermitian pencil (A^H M^-1 A, M)  (= ||M^-1/2 A M^-1/2||^2),
    #   beta^2 = b^H M^-1 b,  gamma_c^2 = c^H M^-1 c.
    # Lanczos (ARPACK, tol 1e-12) instead of the dense SVD of L^-1 A L^-H it
    # replaces (agreement 1e-15 at ne=100 and 200).
    Ms = D.M.tocsc().astype(complex); As = D.A.tocsc().astype(complex)
    Mlu = spla.splu(Ms)
    OP = spla.LinearOperator(Ms.shape, dtype=complex, matvec=lambda x: As.conj().T @ Mlu.solve(As @ x))
    Minv = spla.LinearOperator(Ms.shape, dtype=complex, matvec=lambda x: Mlu.solve(x))
    nu = float(np.sqrt(spla.eigsh(OP, k=1, M=Ms, Minv=Minv, which='LA', tol=1e-12, maxiter=20000,
                                  v0=np.random.default_rng(12345).standard_normal(Ms.shape[0]).astype(complex),
                                  return_eigenvectors=False)[0].real))
    bb = np.asarray(b, complex); cc = np.asarray(c, complex)
    beta = float(np.sqrt(abs(np.vdot(bb, Mlu.solve(bb)))))
    gc = float(np.sqrt(abs(np.vdot(cc, Mlu.solve(cc)))))
    Ac = ctl.A_of(*arow); a_c = float(svdvals(Ac)[0])
    Bc = np.full(r, D.mc, complex); nrm_n = float(np.linalg.norm(nvec)); nrm_B = float(np.linalg.norm(Bc))
    g_rec = float(z['gamma']) if 'gamma' in z.files else float(json.load(open(a.record[:-4] + '.json'))['gamma'])
    W = 2.0 * max(nu, a_c)
    while tail_bound(W, nu, beta, gc, K, nrm_n, nrm_B, a_c) > 0.1 * g_rec:
        W *= 2.0
    tb = tail_bound(W, nu, beta, gc, K, nrm_n, nrm_B, a_c)
    print(f"nu = ||M^-1/2 A M^-1/2|| = {nu:.4g}; ||A_c|| = {a_c:.4g}; beta {beta:.3g}, gamma_c {gc:.3g}; "
          f"W = {W:.4g}: sup_(|w| >= W) sigma_1 <= {tb:.4g} (record gamma {g_rec:.6f})", flush=True)
    csp.DISK_STRICT = (a.disk == 'strict')       # partial ARPACK sets: retry / bisect (budgeted)
    cl = D.cl(b, c)
    sig = lambda w: cl.gain(float(w), arow, nvec, K, want_vectors=False)['sigma']
    t0 = time.time()
    _cert = csp.hinf_dense_levelset if a.method == 'dense' else csp.hinf_certified_sparse
    g, w, ncall, ok = _cert(D.A, D.M, b, c, Ac, Bc, nvec, K, sig, gam_lb=g_rec * (1 - 1e-9),
                            wlo=-W, whi=W, rtol=a.rtol)
    glob_ok = bool(ok) and tb < g
    print(f"{a.method} level-set certificate on [-W, W]: {g:.8f} at w={w:+.5f} ({'certified' if ok else 'INCONCLUSIVE'}, "
          f"{ncall} calls, {time.time() - t0:.1f} s); record {g_rec:.8f} (rel {abs(g - g_rec) / g_rec:.1e}); "
          f"GLOBAL: {'yes' if glob_ok else 'NO'} (tail bound {tb:.3g} < {g:.4f}); partial ARPACK sets {csp.N_DISK_UNTRUSTED} "
          f"(retried {csp.N_DISK_RETRIED}, bisected {csp.N_DISK_BISECTED}, uncovered {csp.N_DISK_UNCOVERED})", flush=True)
    if a.json:
        json.dump(dict(record=a.record, nu=nu, a_c=a_c, beta=beta, gamma_c=gc, W=W, tail_bound=tb,
                       gamma_band=float(g), w_star=float(w), certified=bool(ok), gamma_record=g_rec, global_certified=glob_ok,
                       method=a.method),
                  open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
