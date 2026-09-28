"""
controller.py — TRIDIAGONAL controller parametrisation (replaces canonical).

Why tridiagonal (measured, not asserted):
  * companion/canonical form has cond(eigenvector matrix) ~ 5e4 and coefficients
    ~1e8 for these designs (the roots->coefficients map is Wilkinson-ill-
    conditioned when poles span decades: here |p| spans 7.9e3).  That destroyed
    the even-pencil certification and badly scales the optimiser's steps.
  * tridiagonal with the SAME spectrum: cond(V) ~ 3.6.  Nearly as good as modal
    (cond 1) but, unlike a real modal form, it represents real poles and
    complex-conjugate pairs with ONE smooth parametrisation -- eigenvalues move
    on and off the real axis continuously.  That matters for the (real) cavity.
  * MIMO generalises directly: A tridiagonal (r x r), B (r x m), C (p x r).

Parameters (SISO, order r):
    d  : r     main diagonal
    e  : r-1   super-diagonal
    f  : r-1   sub-diagonal
    nv : r     output row C_c
    K  : 1     feedthrough
  B_c is FIXED to e_r (input gauge), removing the diagonal-scaling redundancy.
Total 4r-1 complex parameters, vs the minimal 2r+1; the ~2r excess buys the
conditioning and the smooth real/complex transition.  (Negligible next to the
~n field unknowns.)
"""
from __future__ import annotations
import numpy as np




def pack(d, e, f, nv, K):
    return np.concatenate([d, e, f, nv, [K]])


def unpack(theta, r):
    i = 0
    d = theta[i:i + r]; i += r
    e = theta[i:i + r - 1]; i += r - 1
    f = theta[i:i + r - 1]; i += r - 1
    nv = theta[i:i + r]; i += r
    K = theta[i]
    return d, e, f, nv, K


def A_of(d, e, f):
    """Tridiagonal controller state matrix."""
    r = len(d)
    A = np.zeros((r, r), complex)
    A[np.arange(r), np.arange(r)] = d
    if r > 1:
        A[np.arange(r - 1), np.arange(1, r)] = e
        A[np.arange(1, r), np.arange(r - 1)] = f
    return A


def B_of(r, mc=1.0):
    """Input gauge B_c = mc * ones(r).

    Why ones and not e_r: the tridiagonal family CONTAINS the diagonal (modal)
    case e = f = 0, which is the numerically exact starting point.  With A
    diagonal, B = e_r would leave every mode but the last uncontrollable; B =
    ones keeps the realisation minimal for both diagonal and tridiagonal A, and
    still removes the diagonal-scaling gauge freedom.
    """
    return np.full((r, 1), mc, dtype=complex)


def from_state_space(Ac, Bc, Cc, mc=1.0, cond_warn=1e6):
    """Convert a (small, well-conditioned) realisation to the tridiagonal
    parametrisation, via its MODAL form:

        A_c = diag(p_1..p_r),  B_c = mc*ones,  C_c = residues
        K(s) = sum_j  c_j / (s - p_j)

    diag(p) is a tridiagonal matrix with e = f = 0, so this is an EXACT point of
    the tridiagonal parameter space, reached by a plain eigendecomposition of a
    small r x r matrix -- no Lanczos, no breakdown.  During optimisation e and f
    become non-zero, which is what allows poles to move on and off the real axis
    continuously (the property a real modal form lacks).

    Raises if the modal basis is ill-conditioned (near-repeated poles), which is
    the one genuine degeneracy of this parametrisation.
    """
    Ac = np.asarray(Ac, complex)
    Bc = np.asarray(Bc, complex).reshape(-1, 1)
    Cc = np.asarray(Cc, complex).reshape(1, -1)
    r = Ac.shape[0]
    lam, V = np.linalg.eig(Ac)
    kappa = np.linalg.cond(V)
    if not np.isfinite(kappa) or kappa > cond_warn:
        raise ValueError(f"modal basis ill-conditioned (cond={kappa:.2e}): "
                         f"near-repeated controller poles")
    W = np.linalg.inv(V)                    # rows are left eigenvectors
    # residues of C (sI-A)^-1 B :  c_j = (C v_j)(w_j^T B)
    cv = (Cc @ V).ravel()
    wb = (W @ Bc).ravel()
    res = cv * wb
    # our gauge is B_c = mc*ones, so fold the 1/mc into the residues
    nv = res / mc
    d = lam.astype(complex)
    e = np.zeros(max(r - 1, 0), complex)
    f = np.zeros(max(r - 1, 0), complex)
    return d, e, f, nv, 0j


def transfer(theta, r, s, mc=1.0):
    """K(s) for the tridiagonal parametrisation (used for validation)."""
    d, e, f, nv, K = unpack(theta, r)
    A = A_of(d, e, f); B = B_of(r, mc)
    return K + (np.asarray(nv).reshape(1, r)
                @ np.linalg.solve(s * np.eye(r) - A, B))[0, 0]
