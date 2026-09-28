"""hurwitz.py -- the compensator stability constraint in its smooth coordinates.

Decision 2026-09-20 (TRACKER H): the compensator abscissa is not Lipschitz at
a double pole (Burke-Overton), and 3 of 3 designs so far end with two poles
coalescing on the margin.  Hermite's criterion states "A_c + eps I Hurwitz" as
POSITIVE DEFINITENESS of a matrix that is a smooth (polynomial) function of
the entries of A_c: the Bezoutian of P(s) and P(-s), P the real polynomial
of degree 2r obtained as p(s) p#(s) with p the characteristic polynomial of
the shifted compensator and p#(s) = conj-coefficient polynomial (roots =
conjugates).  P's roots are p's and their conjugates, so P is Hurwitz iff p
is.  Hermite-Fujiwara: the inertia of Bez(P(s), P(-s)) counts the roots in
the two half-planes; all in the left half-plane <=> the matrix is definite.
On the boundary (a root on the axis, a double root there) the matrix is
singular but perfectly well-defined -- a corner of a smooth constraint, not a
blow-up -- which is what the SQP needs.

Scaling: roots span 0.05-100 in magnitude, so the coefficients of a degree-16
polynomial would span ~1e16; the polynomial is written in s/rho with rho a
scale of the spectrum, which keeps the coefficients O(1)-ish, and the matrix
is normalised by its trace at the start of a solve.

Gradients: finite differences on the 8 x 8 -> 16 x 16 object are negligible
(a few hundred evaluations of a small polynomial), and the constraint is
smooth, so central differences at 1e-6 are accurate.
"""
from __future__ import annotations
import numpy as np


def charpoly(A):
    """Monic characteristic polynomial coefficients, highest first, complex."""
    return np.poly(np.asarray(A, complex))


def bezoutian(f, g):
    """Bezoutian of two polynomials of the same degree n (coefficients highest
    first): B[i, j] from (f(x) g(y) - f(y) g(x)) / (x - y) = sum b_ij x^i y^j."""
    f = np.asarray(f, complex)[::-1]; g = np.asarray(g, complex)[::-1]   # lowest first
    n = len(f) - 1
    B = np.zeros((n, n), complex)
    for i in range(n):
        for j in range(n):
            s = 0j
            for k in range(min(i, j) + 1):
                a = i + j + 1 - k
                if a <= n:
                    s += f[a] * g[k] - f[k] * g[a]
            B[i, j] = s
    return B


def hermite_matrix(Ac, eps, rho=None):
    """Bez(P(s), P(-s)) for P = p p#, p = charpoly(A_c + eps I) in s/rho.
    Real symmetric (up to rounding) of size 2r.  Definite (sign fixed by the
    convention below) iff A_c + eps I is Hurwitz.  Returns (H, rho)."""
    Ac = np.asarray(Ac, complex); r = Ac.shape[0]
    B = Ac + eps * np.eye(r)
    if rho is None:
        rho = max(float(np.max(np.abs(np.linalg.eigvals(B)))), 1e-6)
    p = charpoly(B / rho)                                   # roots scaled by 1/rho
    psharp = np.conj(p)                                      # roots conjugated
    P = np.real(np.polymul(p, psharp))                        # degree n = 2r, real
    # Hermite: P(i w) = h(w) + i g(w), h and g real polynomials in w; the
    # Bezoutian of h and g is definite iff P is Hurwitz (Hermite-Biehler
    # interlacing).  Coefficients highest first: P(iw) = sum c_k (i w)^k.
    n = len(P) - 1
    hc = np.zeros(n + 1); gc = np.zeros(n + 1)               # in w, highest first
    for k in range(n + 1):                                   # power k of s
        c = P[n - k] * (1j ** k)
        hc[n - k] = c.real; gc[n - k] = c.imag
    H = bezoutian(hc, gc)
    H = np.real(0.5 * (H + H.T))
    ref = _reference_sign(n)
    return ref * H, rho


def _reference_sign(n):
    """Sign making the Hermite matrix positive for a Hurwitz polynomial of
    degree n (all roots at -1)."""
    P = np.poly(-np.ones(n)); hc = np.zeros(n + 1); gc = np.zeros(n + 1)
    for k in range(n + 1):
        c = P[n - k] * (1j ** k); hc[n - k] = c.real; gc[n - k] = c.imag
    B = np.real(bezoutian(hc, gc)); ev = np.linalg.eigvalsh(0.5 * (B + B.T))
    return 1.0 if ev.sum() > 0 else -1.0


def hurwitz_margin(Ac, eps, rho=None):
    """lambda_min of the (trace-normalised) Hermite matrix: > 0 iff A_c + eps I
    is Hurwitz; = 0 on the boundary."""
    H, rho = hermite_matrix(Ac, eps, rho)
    tr = float(np.trace(H)); H = H / (abs(tr) if abs(tr) > 1e-300 else 1.0)
    return float(np.linalg.eigvalsh(H).min()), rho


def selftest(seed=0, ntest=200):
    rng = np.random.default_rng(seed); bad = 0
    for _ in range(ntest):
        r = int(rng.integers(2, 9))
        lam = rng.normal(size=r) + 1j * rng.normal(size=r)
        lam = lam - rng.uniform(-1.0, 3.0)               # some left, some right
        A = np.diag(lam)
        Q, _ = np.linalg.qr(rng.normal(size=(r, r)) + 1j * rng.normal(size=(r, r)))
        A = Q @ A @ Q.conj().T
        H, _ = hermite_matrix(A, 0.0)
        ev = np.linalg.eigvalsh(H)
        n_left = int(np.sum(lam.real < 0))
        # count by SIGN (2026-09-24): a relative zero threshold (1e-9 of the
        # largest) misclassified ~1/4 of the draws -- the Bezoutian of a degree-16
        # polynomial has eigenvalue spreads >= 1e9; by sign the count is exact in
        # all 200, and definiteness (what the SQP uses) never disagreed
        n_pos = int(np.sum(ev > 0)); n_neg = int(np.sum(ev < 0))
        # Hermite-Fujiwara: positive count = 2 x left-half-plane roots of p (P has 2r roots)
        if n_pos != 2 * n_left or n_neg != 2 * (r - n_left):
            bad += 1
    return bad


if __name__ == '__main__':
    b = selftest()
    print("hurwitz selftest: inertia mismatches", b, "of 200")


# ------------------------------------------------------------ the 2x2 block
def cluster_block_eig(Ac, k=2):
    """(Before 2026-09-25: cluster_block.)  Biorthogonal EIGENVECTOR bases,
    M = U^H A V, U^H V = I.  Ill-conditioned at a (near-)defective cluster: the
    two right eigenvectors become parallel, G = U^H V singular."""
    Ac = np.asarray(Ac, complex)
    from scipy.linalg import eig as deig
    w, VL, VR = deig(Ac, left=True, right=True)
    idx = np.argsort(-w.real)[:k]
    V = VR[:, idx]; U = VL[:, idx]
    G = U.conj().T @ V                        # k x k, invertible for a separated cluster
    U = U @ np.linalg.inv(G).conj().T          # now U^H V = I
    return U, V, w[idx]


def pick_pair(Ac, near=1e-3, sep_max=5e-2):
    """Which two compensator poles form THE pair (2026-09-25): normally the two
    rightmost; but when three or more poles sit within `near` of the rightmost
    real part (ne=200 K+B+C: a double pole at -0.05 - 0.506i AND a simple pole
    at -0.05 + 6.598i, all on the line), the closest two of them if they are
    within sep_max of each other (the nearly defective pair must be one block,
    never split between the block and the per-pole rows)."""
    w = np.linalg.eigvals(np.asarray(Ac, complex))
    o = np.argsort(-w.real); top = w[o[0]].real
    grp = [int(i) for i in o if w[i].real >= top - near]
    if len(grp) >= 3:
        best = None
        for a_ in range(len(grp)):
            for b_ in range(a_ + 1, len(grp)):
                dd = abs(w[grp[a_]] - w[grp[b_]])
                if best is None or dd < best[0]:
                    best = (dd, grp[a_], grp[b_])
        if best[0] <= sep_max:
            return w[[best[1], best[2]]]
    return w[o[:2]]


def cluster_block(Ac, k=2, pick=None):
    """Compress A_c onto the invariant subspace of its k rightmost eigenvalues:
    M = U^H A V, U^H V = I, U^H A = M U^H, A V = V M.  M's eigenvalues are the
    cluster's; symmetric functions of the cluster (trace, determinant, hence the
    characteristic polynomial) are smooth in A when the cluster is separated
    from the rest, with first derivatives tr(U^H dA V) etc.  Burke-Overton's
    coordinates for the double-pole optimum.

    2026-09-25: bases from the ORDERED SCHUR FORM (V = the first k Schur vectors,
    orthonormal; U from the Sylvester equation T11 Z - Z T22 = T12, the left
    invariant subspace), not from eigenvectors: at a double pole the right
    eigenvectors are parallel and the eigenvector construction breaks down,
    while the invariant SUBSPACE stays well conditioned (its conditioning is the
    separation from the other poles).  Same M up to similarity, same trace,
    determinant, Hermite matrix and derivatives."""
    from scipy.linalg import schur, solve_sylvester
    Ac = np.asarray(Ac, complex); n = Ac.shape[0]
    w = np.linalg.eigvals(Ac)
    if pick is not None:
        pick = np.asarray(pick, complex); k = len(pick)
        others = np.array([x for x in w if np.min(np.abs(x - pick)) > 1e-12 * max(1.0, abs(x))] or [np.inf])
        # a pole belongs to the selection if it is closer to a picked value than
        # a quarter of the gap to the nearest non-picked pole
        gap = max(float(np.min(np.abs(others[:, None] - pick[None, :]))) if np.all(np.isfinite(others)) else 1.0, 1e-300)
        sel = lambda x: bool(np.min(np.abs(x - pick)) < 0.25 * gap)
    else:
        order = np.sort(w.real)[::-1]
        thr = 0.5 * (order[k - 1] + order[k]) if n > k else -np.inf
        sel = lambda x: x.real > thr
    T, Qf, sdim = schur(Ac, output='complex', sort=sel)
    if sdim != k:                              # a tie across the threshold: fall back
        return cluster_block_eig(Ac, k)
    T11, T12, T22 = T[:k, :k], T[:k, k:], T[k:, k:]
    Z = solve_sylvester(T11, -T22, T12) if n > k else np.zeros((k, 0), complex)
    V = Qf[:, :k]
    U = Qf @ np.vstack([np.eye(k), Z.conj().T])  # U^H = [I, Z] Qf^H
    return U, V, np.diag(T11).copy()


def block_hermite(Ac, eps, U, V, rho=None):
    """Hermite matrix (4 x 4 real) of the shifted 2x2 block's polynomial:
    positive definite iff both cluster eigenvalues have Re < -eps."""
    M = U.conj().T @ np.asarray(Ac, complex) @ V
    return hermite_matrix(M, eps, rho)


# ------------------------------------------------ analytic block gradient
def _H_of_tq(tr, ti, qr, qi, rho_unused=None):
    """Hermite matrix of p(s) = s^2 - t s + q, t = tr + i ti, q = qr + i qi
    (the shifted, scaled block's characteristic polynomial)."""
    p = np.array([1.0, -(tr + 1j * ti), qr + 1j * qi], complex)
    P = np.real(np.polymul(p, np.conj(p))); n = len(P) - 1
    hc = np.zeros(n + 1); gc = np.zeros(n + 1)
    for k in range(n + 1):
        cc = P[n - k] * (1j ** k); hc[n - k] = cc.real; gc[n - k] = cc.imag
    H = np.real(bezoutian(hc, gc)); H = 0.5 * (H + H.T)
    return _reference_sign(n) * H


def block_hermite_and_grad(Ac, eps, U, V, dAc_list, rho=None):
    """H (4x4, unnormalised) of the shifted block B = U^H (A_c + eps I) V / rho
    and its EXACT derivatives along the directions dAc_list (dA_c / d theta_j;
    None = zero).  H depends on the block only through t = tr B and q = det B:
      dt = tr(U^H dA V) / rho,   dq = tr(adj(B) U^H dA V) / rho
    and dH/d(Re t, Im t, Re q, Im q) by a 5-point stencil, exact for H's
    per-variable degree (<= 4) up to rounding.  Cost: 16 tiny polynomial
    evaluations + one 2x2 trace per nonzero direction (finite differences
    cost two compressions per design parameter, zero or not)."""
    Ac = np.asarray(Ac, complex); r = Ac.shape[0]
    Mb = U.conj().T @ (Ac + eps * np.eye(r)) @ V
    dBs = [None if dA is None else (U.conj().T @ dA @ V) for dA in dAc_list]
    return hermite_from_block(Mb, dBs, rho)


def hermite_from_block(Bs, dBs, rho=None):
    """Hermite matrix (4 x 4 real) of a SHIFTED 2x2 block Bs (positive definite
    iff both eigenvalues of Bs lie in the open left half-plane) and its exact
    derivatives along the block derivatives dBs (list of 2x2 complex, None =
    zero).  H depends on Bs only through t = tr and q = det:
      dt = tr(dB) / rho,  dq = tr(adj(B) dB) / rho  (B = Bs / rho),
    dH/d(Re t, Im t, Re q, Im q) by a 5-point stencil (exact for H's per-variable
    degree <= 4).  Shared by the compensator block and the closed-loop block."""
    Bs = np.asarray(Bs, complex)
    if rho is None:
        rho = max(float(np.max(np.abs(np.linalg.eigvals(Bs)))), 1e-6)
    B = Bs / rho
    t = np.trace(B); q = np.linalg.det(B)
    x0 = np.array([t.real, t.imag, q.real, q.imag])
    H0 = _H_of_tq(*x0)
    dHdx = []
    for k in range(4):
        h = 1e-3 * max(1.0, abs(x0[k]))
        e = np.zeros(4); e[k] = h
        f = [_H_of_tq(*(x0 + m * e)) for m in (-2, -1, 1, 2)]
        dHdx.append((f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * h))
    adjB = np.array([[B[1, 1], -B[0, 1]], [-B[1, 0], B[0, 0]]])
    grads = []
    for dB in dBs:
        if dB is None:
            grads.append(np.zeros_like(H0)); continue
        dB = np.asarray(dB, complex) / rho
        dt = np.trace(dB); dq = np.trace(adjB @ dB)
        dx = np.array([dt.real, dt.imag, dq.real, dq.imag])
        grads.append(sum(dx[k] * dHdx[k] for k in range(4)))
    return H0, grads, rho

