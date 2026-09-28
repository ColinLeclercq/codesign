"""
bench_invit.py — can INVERSE ITERATION replace ARPACK for eigenvector recovery?

The profile of the recovery step is damning: of 140 ms, `znaupd` (ARPACK's
Arnoldi bookkeeping) is 71% and `zneupd` (extraction) 23%, while the actual
sparse solves are 2.5%.  ARPACK runs ~26 Arnoldi iterations to converge ONE
eigenpair whose eigenvalue the certificate has ALREADY located to 1e-11.

Inverse iteration should not need a Krylov space at all: with lambda known,
    v <- (A_cl - lambda M_cl)^{-1} M_cl v,  normalise
converges geometrically at a rate set by the eigenvalue separation, and the
adjoint iteration gives u.  The factorisation is already built, so each step is
one triangular solve.

This is a HYPOTHESIS, not a measurement -- an earlier estimate of 140x was
arithmetic from the solve cost alone and ignores per-iteration Python overhead
and the possibility of slow convergence near a cluster.  What matters is
whether the recovered vectors give the SAME GRADIENT, and at what cost.

    python bench_invit.py --ne 200
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_codesign


def inv_iter(cl, lam, arow, nvec, K, nit=30, tol=1e-10, seed=0):
    """Right and left eigenvectors of (A_cl, M_cl) at a KNOWN lambda.

    Shift-invert power iteration: v <- S(lambda)^{-1} M v, normalised, where
    S = lambda M_cl - A_cl is the operator setup() already factorises.  The
    left vector comes from the adjoint iteration on the same factors.
    Returns (v, u, iters, residual).
    """
    N = cl.n + cl.r
    cl.setup(-1j * complex(lam), arow, nvec, K)     # i*omega = lam
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    v /= np.linalg.norm(v)
    it = 0
    for it in range(1, nit + 1):
        w = cl.solve(cl.Mcl_apply(v))
        nw = np.linalg.norm(w)
        if nw == 0 or not np.isfinite(nw):
            break
        w /= nw
        if np.linalg.norm(w - v * (np.vdot(v, w))) < tol:
            v = w
            break
        v = w
    u = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    u /= np.linalg.norm(u)
    for _ in range(it):
        z = cl.solveH(cl.Mcl_apply(u))
        nz = np.linalg.norm(z)
        if nz == 0 or not np.isfinite(nz):
            break
        u = z / nz
    res = np.linalg.norm(cl.Acl_apply(v, arow, nvec, K)
                         - lam * cl.Mcl_apply(v)) / max(np.linalg.norm(v), 1e-30)
    return v, u, it, float(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--r', type=int, default=7)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--rep', type=int, default=5)
    a = ap.parse_args()
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=0.8)
    ini = np.load(f"init_ne{a.ne}.npz")
    arow = (ini['d'], ini['e'], ini['f']); nvec = ini['nvec']
    b, c, *_ = D.bc(float(ini['xa']), float(ini['xs']))
    cl = D.cl(b, c)

    al, lam, v0, u0, _, _ = D.alpha_certified(arow, nvec, 0j, b, c, eps=a.eps)
    print(f"lambda from the certificate: {lam:.10g}   alpha={al:+.10f}")
    sig = complex(lam.real + 0.02, lam.imag)

    cl.alpha_krylov(arow, nvec, 0j, shifts=[sig], k=1)
    t0 = time.time()
    for _ in range(a.rep):
        alA, lamA, vA, uA = cl.alpha_krylov(arow, nvec, 0j, shifts=[sig], k=1)
    tA = (time.time() - t0) / a.rep

    inv_iter(cl, lam, arow, nvec, 0j)
    t0 = time.time()
    for _ in range(a.rep):
        vI, uI, it, resI = inv_iter(cl, lam, arow, nvec, 0j)
    tI = (time.time() - t0) / a.rep

    def resid(vv):
        return float(np.linalg.norm(cl.Acl_apply(vv, arow, nvec, 0j)
                                    - lam * cl.Mcl_apply(vv))
                     / max(np.linalg.norm(vv), 1e-30))

    def align(x, y):
        x = x / np.linalg.norm(x); y = y / np.linalg.norm(y)
        return float(abs(np.vdot(x, y)))

    print(f"\nARPACK  : {tA*1e3:8.2f} ms   residual(v) {resid(vA):.2e}")
    print(f"INV ITER: {tI*1e3:8.2f} ms   residual(v) {resI:.2e}   "
          f"{it} iterations")
    print(f"speedup : {tA/tI:6.1f}x")
    print(f"\nalignment |<v_arpack, v_invit>| = {align(vA, vI):.8f}  (1 = same)")
    print(f"alignment |<u_arpack, u_invit>| = {align(uA, uI):.8f}")

    # the number that actually matters: does the GRADIENT agree?
    eta_A = np.vdot(uA, cl.Mcl_apply(vA))
    eta_I = np.vdot(uI, cl.Mcl_apply(vI))
    from fe_codesign import parse_case
    fl = parse_case('K')
    bb, cc, pa, ps, na, ns = D.bc(float(ini['xa']), float(ini['xs']))
    gA = D.grads(cl, vA, uA, arow, nvec, 0j, float(ini['xa']),
                 float(ini['xs']), pa, ps, na, ns, fl, eta=eta_A)
    gI = D.grads(cl, vI, uI, arow, nvec, 0j, float(ini['xa']),
                 float(ini['xs']), pa, ps, na, ns, fl, eta=eta_I)
    for key in ('d', 'e', 'f', 'n'):
        if key in gA:
            x, y = np.atleast_1d(gA[key]).ravel(), np.atleast_1d(gI[key]).ravel()
            den = max(np.linalg.norm(x), 1e-30)
            print(f"  grad[{key}] relative difference: "
                  f"{np.linalg.norm(x-y)/den:.2e}")
    print("\nThe GRADIENT agreement is what decides it: the eigenvectors are")
    print("only ever used to build it.  A speedup with a wrong gradient is")
    print("worthless; agreement to ~1e-8 makes the substitution safe.")


if __name__ == "__main__":
    main()
