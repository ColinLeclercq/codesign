"""mfree.py -- numerical-range bounds of the backward-facing-step plant with
no mass inverse and no Cholesky factor of the velocity mass (BFS.md 4,
note_algebra section "The backward-facing step: algebra without M^-1").

The plant pencil is (A, M) with A = [[A_vv, B^T], [B, 0]] (Taylor-Hood;
Dirichlet rows/columns -identity) and M = blockdiag(M_vv, 0) (Dirichlet
rows/columns zero).  An eigenvector (u, p) has B u = 0, u = 0 at Dirichlet
dofs, and lambda = u^H A_vv u / u^H M_vv u: every eigenvalue lies in the
numerical range W of A_vv in the M inner product restricted to ker B.  Its
bounds are top eigenvalues of HERMITIAN pencils (H, M) on ker B, e.g.
H = (A + A^H)/2 for max Re lambda.  Here they are obtained -- and CERTIFIED --
by inertia alone:

    sigma M - H  positive definite on ker B
      <=>  K(sigma) = [[sigma M_vv - H_vv, B^T], [B, -delta I]] has exactly
           n_p negative and n_v positive pivots (delta > 0 small)

(the saddle-point inertia theorem; '=>' holds for every delta > 0, since on
ker B the term B^T B / delta vanishes, and it is the direction a certificate
needs).  The pivots come from one sparse LDL^T without pivoting (qdldl: the
matrix is quasi-definite exactly when the test passes, and a failed or
indefinite factorisation simply answers "not certified").  Sylvester's law of
inertia makes the signs of D the inertia of K.  Complex Hermitian H = H_r +
i H_i is embedded as the real symmetric [[H_r, -H_i], [H_i, H_r]] (each
eigenvalue twice), with the constraint on both copies.

lambda_max(H, M on ker B) <= sigma is then found by doubling and bisection on
sigma: every sigma that passes is a PROOF, no Ritz value, no tolerance from
below.  Cost: one LDL^T per test (3 s at n = 191,720 in the sandbox).

Nothing here touches M^-1: M is only scaled and added.  (MUMPS reports no
inertia in complex arithmetic -- INFOG(12) = 0 for the complex PETSc build --
hence qdldl, which is real.)
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


class SaddleRange:
    """The plant's constraint structure, once: pressure rows (zero rows of M
    with a zero diagonal in A), Dirichlet rows (zero rows of M with A's -1
    diagonal), the constraint block B."""

    def __init__(self, A, M, delta=1e-10):
        A = sp.csr_matrix(A); M = sp.csr_matrix(M)
        self.n = A.shape[0]
        mrow = np.asarray(abs(M).sum(axis=1)).ravel() > 0
        ad = A.diagonal()
        self.p_mask = (~mrow) & (ad == 0.0)
        self.d_mask = (~mrow) & (ad != 0.0)
        self.v_mask = mrow
        P = sp.diags(self.p_mask.astype(float)); V = sp.diags(self.v_mask.astype(float))
        self.Bfull = (P @ A @ V + V @ A @ P).tocsr()          # [[0, B^T], [B, 0]] in full indexing
        self.M = M
        self.delta = float(delta)
        self.n_p = int(self.p_mask.sum())
        self.stats = dict(tests=0)

    def _K(self, sigma, Hr, Hi):
        """sigma M - H on the velocity block, the constraint, -delta on the
        pressure block and +1 on Dirichlet rows: real (Hi None) or embedded."""
        V = sp.diags(self.v_mask.astype(float))
        Kr = (sigma * self.M - V @ sp.csr_matrix(Hr) @ V).tocsr()
        extra = sp.diags(np.where(self.p_mask, -self.delta, 0.0) + np.where(self.d_mask, 1.0, 0.0))
        if Hi is None:
            return (Kr + self.Bfull + extra).tocsc(), self.n_p
        Ki = (V @ sp.csr_matrix(Hi) @ V).tocsr()
        Kb = sp.bmat([[Kr + self.Bfull + extra, Ki], [-Ki, Kr + self.Bfull + extra]], format='csc')
        return Kb, 2 * self.n_p

    def pd(self, sigma, Hr, Hi=None):
        """True iff sigma M - (Hr + i Hi) is certified positive definite on
        ker B (velocity dofs, Dirichlet excluded).  Hi must be skew-symmetric
        (H Hermitian)."""
        import qdldl
        K, nneg = self._K(float(sigma), Hr, Hi)
        self.stats['tests'] += 1
        try:
            d = qdldl.Solver(K).factors()[1]
        except Exception:
            return False
        if not np.all(np.isfinite(d)):
            return False
        return int((d < 0).sum()) == nneg and int((d > 0).sum()) == K.shape[0] - nneg

    def lmax(self, Hr, Hi=None, start=1.0, rtol=1e-3, atol=1e-3):
        """A CERTIFIED upper bound on lambda_max of the Hermitian pencil
        (Hr + i Hi, M) on ker B: the smallest sigma found (doubling, then
        bisection to rtol |sigma| + atol) for which `pd` passes."""
        s = float(start)
        if self.pd(s, Hr, Hi):
            hi = s; step = max(abs(s), 1.0)
            lo = hi - step
            while self.pd(lo, Hr, Hi):
                hi = lo; step *= 2.0; lo = hi - step
        else:
            lo = s; step = max(abs(s), 1.0)
            hi = lo + step
            while not self.pd(hi, Hr, Hi):
                lo = hi; step *= 2.0; hi = lo + step
                if step > 1e12:
                    raise RuntimeError("SaddleRange.lmax: no certified bound below 1e12")
        while hi - lo > rtol * max(abs(hi), 1.0) + atol:
            mid = 0.5 * (lo + hi)
            if self.pd(mid, Hr, Hi):
                hi = mid
            else:
                lo = mid
        return float(hi)


_RANGES = {}


def saddle_range(A, M):
    key = (id(A), id(M), A.shape, A.nnz, M.nnz)
    r = _RANGES.get(key)
    if r is None:
        _RANGES.clear()
        r = _RANGES[key] = SaddleRange(A, M)
    return r
