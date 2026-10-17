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


def modal_rows(blocks, d, e, f, L, scales=None):
    """The modal compensator constraints in the form of the SQP's compensator
    rows, value v <= L (v a pole's real part in the per-pole form): per 2 x 2
    block v1 = tr(M) / 2 and v2 = L - det(M - L I) / s (s = |a - L| + |g - L|,
    at least 1e-3: v2 in the units of a real part near the line), per 1 x 1
    block its entry.  scales: per block a constant s (the normal form's om,
    normal_scales; None entries and scales=None: the s above).  The gradient
    holds s fixed, exact where s is a constant.  Returns
    [(v, {'d': {i: dv/dd_i}, 'e': {...}, 'f': {...}})]."""
    d, e, f = (np.real(np.asarray(x)) for x in (d, e, f))
    out = []
    for k, b in enumerate(blocks):
        if len(b) == 1:
            i = b[0]
            out.append((float(d[i]), dict(d={i: 1.0}, e={}, f={})))
            continue
        i, j = b
        a, g, bb, cc = d[i], d[j], e[i], f[i]
        out.append((float(0.5 * (a + g)), dict(d={i: 0.5, j: 0.5}, e={}, f={})))
        s = float(scales[k]) if (scales is not None and scales[k] is not None) else max(abs(a - L) + abs(g - L), 1e-3)
        det = (a - L) * (g - L) - bb * cc
        out.append((float(L - det / s), dict(d={i: -(g - L) / s, j: -(a - L) / s}, e={i: cc / s}, f={i: bb / s})))
    return out


# ------------------------------------------------- the normal form (20261010s)
# The modal blocks above keep a similarity gauge: T in GL(2) per 2 x 2 block
# (M -> T M T^-1, B -> T B, C -> C T^-1) leaves K(s) unchanged, and fixing one
# B_c entry per row (or B_c = mc * ones) removes only its diagonal part.  The
# two directions left per block are flat: the iterates drift along them, and
# the representative sets the metric of every step.  The NORMAL FORM is a
# slice through every orbit: with the block's input column b = beta [1, 1]
# fixed,
#     M = [[a, g + om], [a - om, g]]          (f_i = d_i - om, e_i = d_j + om)
# a, g free, om > 0 a constant of the job (sqrt(det M) when the job starts,
# where a = g = tr / 2).  It is R1 C R1^-1 for the companion C = [[0, om],
# [-det/om, tr]] of b = [0, beta], R1 = [[1, 1], [-1, 1]] (sqrt 2 x a
# rotation: the block's eigenvector conditioning is the companion's): unique
# (no gauge: the SISO block's Jacobian has rank 4 of its 4 parameters, the
# general block 4 of 6), smooth through a double pole and the real / complex
# transition (rank 4 there too), and
#     tr M = a + g,   det(M - L I) = L^2 - L (a + g) + om (g - a) + om^2
# are AFFINE in (a, g): with s = om the two Routh-Hurwitz rows are linear in
# z, their gradients (1/2, 1/2) and ((L + om) / om, (L - om) / om) never
# vanish nor align (the general block's det gradient vanished at M = L I).
# Of the companion slices measured on the eight BFS records (T0 = [[1, 1],
# [0, 1]] in place of R1, and the balanced sigma I + [[0, kap], [c, 0]] whose
# det row is quadratic) this one has the best-conditioned packed Jacobian
# z -> K(jw) on all eight (P_report, 20261010s).  1 x 1 blocks are unchanged.
# On the BFS the start is converted by an exact similarity per block
# (normal_convert); every job start re-centres om (a resume rebalances).

NORMAL_COND_MAX = 1e12     # cond([b, M b]) beyond which a block is refused (not controllable from its pivot input)
NORMAL_OM_REL = 1e-12      # the om fallback's floor relative to |tr| (det = 0: a pole at 0)
NORMAL_TIE_TOL = 1e-12     # normal_check: a tie held to this x the block's scale (rounding of a - om, g + om)


def normal_ties(blocks):
    """The normal form's tied entries: [((dst kind, i), (src kind, j))] meaning
    th[dst] = th[src] + (th0[dst] - th0[src]): f_i follows d_i and e_i follows
    d_j (j = i + 1), per 2 x 2 block (the offsets -om and +om)."""
    out = []
    for b in blocks:
        if len(b) == 2:
            i, j = b
            out += [(('f', i), ('d', i)), (('e', i), ('d', j))]
    return out


def normal_scales(blocks, d, e):
    """om per block of a compensator in the normal form (e_i - d_j), None for a
    1 x 1 block: the constant s of its v2 row (modal_rows)."""
    d, e = np.real(np.asarray(d)), np.real(np.asarray(e))
    return [None if len(b) == 1 else float(e[b[0]] - d[b[1]]) for b in blocks]


def normal_check(blocks, d, e, f):
    """The largest violation of the normal form over the 2 x 2 blocks,
    relative to the block's scale: |(d_i - f_i) - (e_i - d_j)| (the two ties'
    om), inf where om = e_i - d_j <= 0.  At most NORMAL_TIE_TOL (rounding) for a
    compensator in the normal form."""
    d, e, f = (np.real(np.asarray(x)) for x in (d, e, f))
    v = 0.0
    for b in blocks:
        if len(b) == 2:
            i, j = b
            om = e[i] - d[j]
            sc = max(abs(d[i]), abs(d[j]), abs(e[i]), abs(f[i]), 1e-300)
            v = max(v, abs((d[i] - f[i]) - om) / sc if om > 0 else np.inf)
    return v


def block_cond(blocks, d, e, f):
    """The eigenvector condition number of each 2 x 2 block (inf at a
    defective block; 1 for a 1 x 1): the representative's conditioning."""
    A = A_of(np.real(d), np.real(e), np.real(f)).real
    out = []
    for b in blocks:
        if len(b) == 1:
            out.append(1.0)
            continue
        Mb = A[np.ix_(b, b)]
        try:
            out.append(float(np.linalg.cond(np.linalg.eig(Mb)[1])))
        except np.linalg.LinAlgError:
            out.append(np.inf)
    return out


def _pivot_score(M, b):
    """sigma_min of [b / |b|, M b / |M b|]: how far b is from an eigenvector of M
    (0: the block is not controllable from b), independent of the scales of b
    and M -- the central pivot is the input column with the largest score."""
    nb = np.linalg.norm(b); Mb = M @ b; nMb = np.linalg.norm(Mb)
    if nb == 0 or nMb == 0:
        return 0.0
    return float(np.linalg.svd(np.column_stack([b / nb, Mb / nMb]), compute_uv=False)[-1])


def normal_convert(d, e, f, Bc, Cc, blocks, fixed_b, pivots=None):
    """The compensator (A_c in the modal blocks `blocks`, B_c r x p, C_c m x r;
    real) in the normal form by an exact similarity per 2 x 2 block, T = R1 Qc
    Q^-1 (Q = [b, M b], Qc = [bc, C bc] its companion's, bc = [0, beta]): M ->
    T M T^-1 (set from tr, det exactly), B_c rows of the block -> T B_c, C_c
    columns -> C_c T^-1, om = sqrt|det M| (a block left of the line L < 0 has
    det >= L^2 > 0).

    fixed_b: True (SISO, decentral) -- B_c is fixed: the block's input column
    is its one nonzero column, already beta [1, 1] (beta = mc), and stays so;
    False (central) -- the pivot column per block (pivots[k] when given and
    its _pivot_score is within 1e-3 of the best, else the best) becomes beta
    [1, 1], beta = |b| / sqrt(2), the others follow T.  A block with no output
    (its C_c columns zero: the zero start, K = 0 on the block) is transformed
    by T where Q is regular, else its M is set from (tr, det) and its B_c kept
    (pivot column beta [1, 1]).

    Returns d, e, f, Bc, Cc (new arrays) and info dict(om, pivots, beta, condQ)
    per block (None for 1 x 1 blocks).  Raises ValueError naming the block when
    cond(Q) > NORMAL_COND_MAX and the block has an output."""
    d, e, f = (np.array(np.real(np.asarray(x)), float) for x in (d, e, f))
    B = np.array(np.real(np.asarray(Bc)), float).reshape(len(d), -1)
    C = np.array(np.real(np.asarray(Cc)), float).reshape(-1, len(d))
    A = A_of(d, e, f).real
    R1 = np.array([[1.0, 1.0], [-1.0, 1.0]])
    info = dict(om=[], pivots=[], beta=[], condQ=[])
    for k, blk in enumerate(blocks):
        if len(blk) == 1:
            for key_ in info:
                info[key_].append(None)
            continue
        i, j = blk
        M = A[np.ix_(blk, blk)]
        tr, det = float(M[0, 0] + M[1, 1]), float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
        om = float(np.sqrt(abs(det)))
        if not om > NORMAL_OM_REL * max(abs(tr), 1.0):
            om = max(abs(tr), 1e-3)
        Bb = B[[i, j], :]
        if fixed_b:
            nz = [q for q in range(Bb.shape[1]) if np.any(Bb[:, q] != 0)]
            if len(nz) != 1 or Bb[0, nz[0]] != Bb[1, nz[0]]:
                raise ValueError(f"normal form: block {blk} -- the fixed B_c is not beta * [1, 1] in one column "
                                 f"({Bb.tolist()})")
            piv = nz[0]; beta = float(Bb[0, piv])
        else:
            sc_ = [_pivot_score(M, Bb[:, q]) for q in range(Bb.shape[1])]
            piv = int(np.argmax(sc_))
            if pivots is not None and k < len(pivots) and pivots[k] is not None and 0 <= int(pivots[k]) < Bb.shape[1] \
                    and sc_[int(pivots[k])] >= 1e-3 * max(sc_):
                piv = int(pivots[k])                     # the record's pivot, unless far worse than the best
            beta = float(np.linalg.norm(Bb[:, piv]) / np.sqrt(2.0))
        b = Bb[:, piv]
        Q = np.column_stack([b, M @ b])
        cq = float(np.linalg.cond(Q)) if np.all(np.isfinite(Q)) else np.inf
        g_a = det / om - om                              # g - a
        a_n = 0.5 * (tr - g_a); g_n = 0.5 * (tr + g_a)
        Cb = C[:, [i, j]]
        if cq <= NORMAL_COND_MAX and beta != 0:
            Mc = np.array([[0.0, om], [-det / om, tr]]); bc = np.array([0.0, beta])
            T = R1 @ np.column_stack([bc, Mc @ bc]) @ np.linalg.inv(Q)
            Bn = T @ Bb; Cn = Cb @ np.linalg.inv(T)
        elif not np.any(Cb):
            Bn = Bb.copy(); Cn = Cb.copy()               # no output: K = 0 on the block, any realisation of its poles
            if beta == 0:
                beta = 1.0
        else:
            raise ValueError(f"normal form: block {blk} is not controllable from its input column {piv} "
                             f"(cond [b, M b] = {cq:.3g} > {NORMAL_COND_MAX:g}); resume it with --ctrl-modal on")
        Bn[:, piv] = beta                                # exactly beta [1, 1]
        B[[i, j], :] = Bn; C[:, [i, j]] = Cn
        d[i], d[j], e[i], f[i] = a_n, g_n, g_n + om, a_n - om
        info['om'].append(om); info['pivots'].append(int(piv)); info['beta'].append(beta); info['condQ'].append(cq)
    return d, e, f, B, C, info


def tf_samples(d, e, f, Bc, Cc, Dc, w):
    """K(i w) = D_c + C_c (i w - A_c)^-1 B_c on the frequencies w (len(w) x m x p)."""
    A = A_of(np.asarray(d), np.asarray(e), np.asarray(f))
    r = A.shape[0]
    B = np.asarray(Bc, complex).reshape(r, -1); C = np.asarray(Cc, complex).reshape(-1, r)
    Dm = np.asarray(Dc, complex).reshape(C.shape[0], B.shape[1])
    return np.array([Dm + C @ np.linalg.solve(1j * x * np.eye(r) - A, B) for x in np.asarray(w, float)])
