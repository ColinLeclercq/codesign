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


def A_of(d, e, f, *rest):
    """Tridiagonal controller state matrix.  (MIMO: arow = (d, e, f, B_c); the
    input matrix B_c, if present, is ignored here -- see Bc_of.)"""
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


def Bc_of(arow, r, mc=1.0):
    """The compensator input matrix from arow: SISO arow = (d, e, f) with the
    fixed gauge B_c = mc * ones(r) (a VECTOR, as the SISO code uses it); MIMO
    arow = (d, e, f, B_c) with B_c an r x p matrix (2026-09-29, STAGE6)."""
    if len(arow) > 3 and arow[3] is not None:
        return np.asarray(arow[3], complex).reshape(r, -1)
    return np.full(r, mc, complex)


# ---------------------------------------------------------------- modal form
# (20261010e, user: the compensator in real 2 x 2 diagonal blocks.)  A_c =
# blockdiag(M_1, ..., M_k) per loop, M_j = [[a, b], [c, g]] a general real 2 x 2
# block (the tridiagonal with e = f = 0 between blocks; one 1 x 1 block closes a
# loop of odd order).  The block's poles are left of the line L iff
#     tr(M - L I) <= 0  and  det(M - L I) >= 0        (Routh-Hurwitz, degree 2)
# -- polynomial in the parameters, smooth everywhere, also where poles merge
# within a block or across blocks: in the tridiagonal form a double or triple
# pole on the line moved like a square or cube root of a step and froze the SQP
# (BFS.md 5).  A 2 x 2 block carries a complex pair or two real poles and moves
# continuously between the two.

def modal_blocks(orders):
    """The blocks' state indices, loop by loop: [(i, i + 1), ..., (j,)]."""
    out, s0 = [], 0
    for ok in orders:
        for j in range(ok // 2):
            out.append((s0 + 2 * j, s0 + 2 * j + 1))
        if ok % 2:
            out.append((s0 + ok - 1,))
        s0 += ok
    return out


def modal_cut(orders):
    """The tridiagonal off-diagonal positions i (between states i and i + 1)
    that join two blocks: e[i] = f[i] = 0 there."""
    inside = set(b[0] for b in modal_blocks(orders) if len(b) == 2)
    return [i for i in range(sum(orders) - 1) if i not in inside]


def modal_form(d, e, f, orders):
    """A tridiagonal (d, e, f) in modal 2 x 2 blocks with the same poles, loop
    by loop (a loop's tridiagonal block must not be coupled to the next: e = f
    = 0 at the loop boundaries).  The input and output matrices are NOT
    transformed: exact for the zero controller (C_c = D_c = 0) only -- the
    caller checks.  Complex pairs a +- i b fill [[a, b], [-b, a]]; the real poles,
    in decreasing order, the remaining 2 x 2 blocks two by two (diagonal) and
    the closing 1 x 1 block of an odd loop."""
    A = A_of(np.real(d), np.real(e), np.real(f)).real
    dd, ee, ff = np.zeros(len(d)), np.zeros(len(d) - 1), np.zeros(len(d) - 1)
    s0 = 0
    for ok in orders:
        w = np.linalg.eigvals(A[s0:s0 + ok, s0:s0 + ok])
        pairs = sorted([x for x in w if x.imag > 1e-12], key=lambda x: -x.real)
        reals = sorted([x.real for x in w if abs(x.imag) <= 1e-12], reverse=True)
        slots = modal_blocks([ok])
        k = 0
        for b in slots:
            i = s0 + b[0]
            if len(b) == 2 and k < len(pairs):
                dd[i] = dd[i + 1] = pairs[k].real; ee[i] = pairs[k].imag; ff[i] = -pairs[k].imag; k += 1
            elif len(b) == 2:
                dd[i], dd[i + 1] = reals.pop(0), reals.pop(0)
            else:
                dd[i] = reals.pop(0)
        s0 += ok
    return dd.astype(complex), ee.astype(complex), ff.astype(complex)


def modal_rows(blocks, d, e, f, L):
    """The modal compensator constraints in the form of the SQP's compensator
    rows, value v <= L (v a pole's real part in the per-pole form): per 2 x 2
    block v1 = tr(M) / 2 and v2 = L - det(M - L I) / s (s = |a - L| + |g - L|,
    at least 1e-3: v2 in the units of a real part near the line), per 1 x 1
    block its entry.  Returns [(v, {'d': {i: dv/dd_i}, 'e': {...}, 'f': {...}})]."""
    d, e, f = (np.real(np.asarray(x)) for x in (d, e, f))
    out = []
    for b in blocks:
        if len(b) == 1:
            i = b[0]
            out.append((float(d[i]), dict(d={i: 1.0}, e={}, f={})))
            continue
        i, j = b
        a, g, bb, cc = d[i], d[j], e[i], f[i]
        out.append((float(0.5 * (a + g)), dict(d={i: 0.5, j: 0.5}, e={}, f={})))
        s = max(abs(a - L) + abs(g - L), 1e-3)
        det = (a - L) * (g - L) - bb * cc
        out.append((float(L - det / s), dict(d={i: -(g - L) / s, j: -(a - L) / s}, e={i: cc / s}, f={i: bb / s})))
    return out
