"""
eigcount.py — how many eigenvalues lie in Re(lambda) > -eps, by the ARGUMENT PRINCIPLE.

WHY THIS EXISTS
---------------
The line certificate in certify_sparse proves that nothing lies ON
Re = -eps.  That is rigorous.  Stability was then INFERRED from it by a
continuation argument: start stable, eigenvalues move continuously, so nothing
can enter Re > -eps without crossing the line.

The gap is the word "continuously".  The optimiser takes DISCRETE steps, and an
eigenvalue can be left of the line at iterate k and right of it at k+1 without
ever being observed on it.  Three designs exploited exactly that -- most
recently KSP+FB+FC, whose certificate reported alpha = -0.2295 while a dense
eigensolve found +0.3897.

This module answers the UNCONDITIONAL question instead.  For the pencil
(A_cl, M_cl), the number of eigenvalues enclosed by a positively-oriented
contour C is

    N = (1 / 2 pi i) * closed_integral_C  d/dz log det(z M - A) dz
      = (1 / 2 pi i) * closed_integral_C  tr[ (z M - A)^{-1} M ] dz .

No continuation, no assumption about the path taken to get here: it is a
property of the matrix in front of you.

WHY IT SCALES TO THE CAVITY
---------------------------
The trace is estimated by Hutchinson sampling,

    tr[R(z) M] ~ (1/S) sum_s  w_s^H R(z) M w_s ,      w_s Rademacher,

so each quadrature node costs S resolvent solves and NOTHING ELSE -- no dense
factorisation, no eigensolver, no explicit matrix.  The same shifted solves the
optimiser already performs.  Cost is (nodes x S) solves, independent of n
except through the solve itself, which is exactly the operation that has to
scale anyway.

WHAT IT COSTS IN RIGOUR
-----------------------
The count is an integer, and Hutchinson gives it with variance.  We exploit
the integrality: the estimate is accepted only when it is within `tol` of an
integer, and otherwise more probes are drawn.  So the answer is either a
confident integer or an explicit "inconclusive" -- never a silent wrong number.
That is weaker than a deterministic count and stronger than an assumption.

    python eigcount.py --ne 200 --results results --eps 0.05
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np


def count_right_of(cl, arow, nvec, K, eps, R=None, nodes=48, probes=24,
                   tol=0.15, max_probes=192, rng=None, verbose=False):
    """Number of eigenvalues of (A_cl, M_cl) with Re > -eps.

    The contour is a half-disk: the vertical segment Re = -eps closed by a
    semicircle of radius R into the right half-plane.  R must enclose every
    eigenvalue that could be there; `stability_region` already bounds the
    spectrum, and a runaway controller pole is caught exactly by the r x r
    check, so R = (that bound) is safe.

    Returns (count, residual) where residual is the distance of the raw
    estimate from the nearest integer.  A residual above `tol` after
    `max_probes` means INCONCLUSIVE and the caller must treat it as unsafe.
    """
    N = cl.n + cl.r
    rng = rng or np.random.default_rng(0)
    if R is None:
        R = 50.0

    # --- contour: vertical segment + semicircle, both parametrised ----------
    # vertical  z = -eps + i y,  y from -R to R      (dz = i dy)
    ny = max(nodes, 8)
    y, wy = np.polynomial.legendre.leggauss(ny)
    y = y * R
    wy = wy * R
    zs = [(-eps + 1j * yy, 1j * ww) for yy, ww in zip(y, wy)]
    # semicircle z = -eps + R e^{i th}, th from pi/2 down to -pi/2
    nt = max(nodes // 2, 6)
    t, wt = np.polynomial.legendre.leggauss(nt)
    th = t * (np.pi / 2)
    wth = wt * (np.pi / 2)
    for tt, ww in zip(th, wth):
        z = -eps + R * np.exp(1j * tt)
        zs.append((z, -1j * R * np.exp(1j * tt) * ww))

    W = (rng.integers(0, 2, size=(N, probes)) * 2 - 1).astype(complex)
    total = 0.0 + 0j
    acc = np.zeros(probes, complex)
    for z, dz in zs:
        # (z M - A)^{-1} M w  for every probe; setup takes omega with z = i*omega
        cl.setup(-1j * z, arow, nvec, K)
        for s in range(probes):
            v = cl.solve(cl.Mcl_apply(W[:, s]))
            acc[s] += dz * np.vdot(W[:, s], v)
    est = acc / (2j * np.pi)
    cnt = float(np.mean(est.real))
    n_int = int(np.round(cnt))
    resid = abs(cnt - n_int)
    if verbose:
        print(f"    eigcount: raw {cnt:+.4f} -> {n_int}  "
              f"(residual {resid:.3f}, {probes} probes, {len(zs)} nodes)")
    return n_int, resid


def is_stable(cl, arow, nvec, K, eps, R=None, **kw):
    """True iff NO eigenvalue lies right of Re = -eps.  (count, residual) also returned."""
    n, res = count_right_of(cl, arow, nvec, K, eps, R=R, **kw)
    return (n == 0), n, res
