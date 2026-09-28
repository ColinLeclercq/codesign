"""
bounds.py — fundamental performance limits for ONE actuator and ONE sensor.

With a single actuator b and a single sensor c the closed loop is, by
Sherman-Morrison, a RANK-ONE correction of the open-loop energy resolvent:

    G_cl(iw) = T(iw) + q(w) * u(w) r(w)^H ,      q free in C,

where the LEFT factor u is fixed by the ACTUATOR and the RIGHT factor r by the
SENSOR.  A rank-one correction with a fixed right factor can at best annihilate
the part of T acting along r; with a fixed left factor, the part of the response
lying along u.  Hence three lower bounds on the achievable H-infinity norm,
valid for ANY controller of ANY order:

  sensor-limited     gamma >= sup_w  max{ ||q||_M/||f||_M : c^H q = 0 }
                     "worst amplification of a disturbance the sensor cannot see"

  actuator-limited   gamma >= sup_w  max ||P_u^perp q||_M/||f||_M
                     "worst amplification the actuator cannot reach"

  absolute floor     gamma >= sup_w  sigma_2( T(iw) )
                     any rank-one correction, i.e. any single-input/single-output
                     placement whatsoever

The binding bound is the larger of the first two.  All three are computed here
MATRIX-FREE, with the same sparse LU of S = i w M - A reused for the direct and
adjoint solves; M appears only as a mat-vec.

    python bounds.py --ne 200 --xa -1.03 --xs 0.98
"""
from __future__ import annotations
# --- repository root on sys.path (moved into tools/ by cleanup_archive.py)
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
# --- end root shim
import os as _os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import argparse, sys
import numpy as np
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from fe_gl import assemble_gl, SIGMA


# ---------------------------------------------------------------------------
# TWO IMPLEMENTATIONS, and why both are here.
#
#   --method krylov  (default)  fully matrix-free: one sparse LU of S per
#                               frequency, reused for the direct and adjoint
#                               solves; M only as a mat-vec; no L, no M^{-1}.
#                               THIS is the one that ports to the cavity, where
#                               the dense route is impossible (a dense T at
#                               n ~ 1e5 needs 1e10 entries and n^3 flops).
#   --method dense              n x n reference: Cholesky of M, explicit inverse
#                               of S, dense SVD.  Only for validating the above
#                               at GL scale.
#
# They agree to ~2e-4 on every bound (see --method both).
#
# The deflations, matrix-free.  With v = M f the open-loop map is q = S^{-1} v
# and sigma^2 = lambda_max(W), W = M S^{-H} M S^{-1}.  Each bound replaces
# S^{-1} by a rank-one-corrected solve operator R and takes lambda_max(M R^H M R):
#   sensor-limited : annihilate the disturbances the sensor CAN see,
#         R v = S^{-1}v - q_c (psi^H v)/(c^H q_c),
#         psi = S^{-H}c,  q_c = S^{-1} M psi        (constraint c^H q = 0)
#   actuator-limited: remove the response component the actuator CAN reach,
#         R v = q - phi (phi^H M q)/(phi^H M phi),  phi = S^{-1}b,  q = S^{-1}v
#   absolute floor  : second eigenvalue of W itself.
# ---------------------------------------------------------------------------


def _sigma_krylov(M, apply_R, n, k=1, tol=1e-10):
    """sqrt of the largest eigenvalue(s) of W = M R^H M R (standard EVP)."""
    import scipy.sparse.linalg as spla
    def mv(v):
        t = apply_R(v, False)
        t = M @ t
        t = apply_R(t, True)
        return M @ t
    W = spla.LinearOperator((n, n), matvec=mv, dtype=complex)
    lam = spla.eigs(W, k=k, which='LM', tol=tol, maxiter=8000,
                    ncv=min(n - 1, max(4 * k + 4, 24)),
                    return_eigenvectors=False)
    lam = np.sort(np.abs(lam.real))[::-1]
    return np.sqrt(np.maximum(lam, 0.0))


def bounds_at_krylov(Asp, Msp, b, c, w):
    """MATRIX-FREE (sensor, actuator, sigma_1, sigma_2) at frequency w."""
    import scipy.sparse.linalg as spla
    n = Asp.shape[0]
    S = ((1j * w) * Msp - Asp).tocsc()
    lu = spla.splu(S)                       # the ONLY factorisation
    M = Msp

    R0 = lambda v, adj: lu.solve(v, trans='H') if adj else lu.solve(v)
    s12 = _sigma_krylov(M, R0, n, k=2)
    s1 = float(s12[0]); s2 = float(s12[1] if s12.size > 1 else 0.0)

    psi = lu.solve(c, trans='H')
    qc = lu.solve(M @ psi)
    den = np.vdot(c, qc)                    # c^H q_c  (real positive)
    def Rs(v, adj):
        if not adj:
            return lu.solve(v) - qc * (np.vdot(psi, v) / den)
        return lu.solve(v, trans='H') - psi * (np.vdot(qc, v) / np.conj(den))
    sen = float(_sigma_krylov(M, Rs, n, k=1)[0])

    phi = lu.solve(b)
    Mphi = M @ phi
    nphi = np.vdot(phi, Mphi)
    def Ra(v, adj):
        if not adj:
            q = lu.solve(v)
            return q - phi * (np.vdot(phi, M @ q) / nphi)
        t = v - Mphi * (np.vdot(phi, v) / np.conj(nphi))
        return lu.solve(t, trans='H')
    act = float(_sigma_krylov(M, Ra, n, k=1)[0])
    return sen, act, s1, s2


def bounds_at(A, M, L, b, c, w):
    """(sensor-limited, actuator-limited, sigma_1, sigma_2) at frequency w.

    T = L^H S^{-1} L is the energy resolvent (M = L L^H), so that
    sigma_i(T) are the energy amplification gains;
    r = L^H S^{-H} c is the sensor direction and u = L^H S^{-1} b the actuator
    direction of the rank-one closed-loop correction.
    """
    from scipy.linalg import svdvals
    n = A.shape[0]
    S = 1j * w * M - A
    Si = np.linalg.solve(S, np.eye(n))
    T = L.conj().T @ Si @ L
    I = np.eye(n)
    r = L.conj().T @ np.linalg.solve(S.conj().T, c)
    u = L.conj().T @ np.linalg.solve(S, b)
    Pr = np.outer(r, r.conj()) / np.vdot(r, r)
    Pu = np.outer(u, u.conj()) / np.vdot(u, u)
    sen = float(svdvals(T @ (I - Pr))[0])     # disturbance the sensor cannot see
    act = float(svdvals((I - Pu) @ T)[0])     # response the actuator cannot reach
    sv = svdvals(T)
    return sen, act, float(sv[0]), float(sv[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--mu0', type=float, default=0.38,
                    help='MUST match the study (0.38). The module default MU0_STABLE=0.38178 is NOT the same operator: near criticality a 0.5%% change in mu0 moves sigma_1 by 10%%.')
    ap.add_argument('--xa', type=float, default=-1.03)
    ap.add_argument('--xs', type=float, default=0.98)
    ap.add_argument('--wlo', type=float, default=-1.2)
    ap.add_argument('--whi', type=float, default=0.2)
    ap.add_argument('--nw', type=int, default=81)
    ap.add_argument('--method', choices=('krylov','dense','both'),
                    default='krylov')
    ap.add_argument('--json', default=None,
                    help='write the bounds with the positions and grid they assume (read by plot_results.py)')
    a = ap.parse_args()
    from scipy.linalg import cholesky
    Asp, Msp, xn, free = assemble_gl(ne=a.ne, mu0=a.mu0)
    A = np.asarray(Asp.todense())
    M = np.asarray(Msp.todense()).real
    L = cholesky(M, lower=True)

    def prof(x0):
        g = np.exp(-(xn - x0) ** 2 / (2 * SIGMA ** 2)).astype(complex)
        return g / np.sqrt(np.real(np.vdot(g, M @ g)))      # unit L2
    b = M @ prof(a.xa)
    c = M @ prof(a.xs)
    def prof_sp(x0):
        g = np.exp(-(xn - x0) ** 2 / (2 * SIGMA ** 2)).astype(complex)
        return g / np.sqrt(np.real(np.vdot(g, Msp @ g)))

    best = dict(sen=(0, 0), act=(0, 0), s1=(0, 0), s2=(0, 0))
    Aspc = Asp.tocsc(); Mspc = Msp.tocsc()
    bs = Mspc @ prof_sp(a.xa); cs = Mspc @ prof_sp(a.xs)
    for w in np.linspace(a.wlo, a.whi, a.nw):
        if a.method == 'dense':
            sen, act, s1, s2 = bounds_at(A, M, L, b, c, w)
        else:
            sen, act, s1, s2 = bounds_at_krylov(Aspc, Mspc, bs, cs, w)
            if a.method == 'both':
                d = bounds_at(A, M, L, b, c, w)
                for nm, kv, dv in zip(('sen','act','s1','s2'),
                                      (sen,act,s1,s2), d):
                    rel = abs(kv-dv)/max(abs(dv),1e-30)
                    if rel > 1e-3:
                        print(f"   w={w:+.3f} {nm}: krylov {kv:.6f} vs dense "
                              f"{dv:.6f}  rel={rel:.1e}")
        for key, val in (('sen', sen), ('act', act), ('s1', s1), ('s2', s2)):
            if val > best[key][0]:
                best[key] = (val, w)
    print(f"ne={a.ne}  mu0={a.mu0}  actuator xa={a.xa}  sensor xs={a.xs}")
    print(f"  open-loop norm            sup_w sigma_1 = {best['s1'][0]:10.6f} "
          f"at w={best['s1'][1]:+.4f}")
    print(f"  SENSOR-limited bound                    = {best['sen'][0]:10.6f} "
          f"at w={best['sen'][1]:+.4f}")
    print(f"  ACTUATOR-limited bound                  = {best['act'][0]:10.6f} "
          f"at w={best['act'][1]:+.4f}")
    print(f"  absolute rank-one floor   sup_w sigma_2 = {best['s2'][0]:10.6f} "
          f"at w={best['s2'][1]:+.4f}")
    print(f"\n  binding bound for this (b,c): "
          f"{max(best['sen'][0], best['act'][0]):.6f}")
    if a.json:
        import json
        json.dump(dict(ne=a.ne, mu0=a.mu0, xa=a.xa, xs=a.xs, wlo=a.wlo, whi=a.whi, nw=a.nw, method=a.method,
                       open_loop=best['s1'][0], sensor_limited=best['sen'][0], actuator_limited=best['act'][0],
                       floor=best['s2'][0],
                       note='sensor-limited applies to designs with the sensor at xs, actuator-limited to designs '
                            'with the actuator at xa, the floor to any single actuator/sensor pair; sup over the '
                            'grid [wlo, whi] (a lower bound on the sup over all w)'),
                  open(a.json, 'w'), indent=1)
        print(f"  written: {a.json}")


if __name__ == "__main__":
    main()
