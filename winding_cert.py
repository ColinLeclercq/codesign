"""
winding_cert.py — UNCONDITIONAL stability certificate by the argument principle.

Every stability guarantee in the pipeline so far is CONDITIONAL: the line sweep
proves "nothing ON Re = -eps" and stability follows only along a continuation
chain (stable anchor + gated steps).  The dense eigensolve is the only
unconditional test, and it does not exist at cavity scale.  This tool closes
that gap with the Nyquist idea of Apkarian & Noll (IJRNC 2018) transplanted
into the sparse + low-rank structure of this problem.

Because the closed loop differs from the DECOUPLED pair (plant, compensator)
by a rank-2 coupling, the characteristic determinant factorises exactly
(verified to 1e-15 against a dense determinant):

    det(s M_cl - A_cl) = det(s M - A) * det(s I - A_c) * f(s),
    f(s) = 1 - k(s) G(s),
    k(s) = K + n^T (s I - A_c)^{-1} B_c        (r x r, trivial),
    G(s) = c^H (s M - A)^{-1} b                (ONE sparse solve).

The argument principle on the region { Re s > line } then gives the exact
eigenvalue count

    #{ closed-loop eigenvalues with Re > line }
        = winding(f along the boundary) + P_plant + P_ctrl,

where P_plant = #{ plant eigenvalues right of the line } is DESIGN-INDEPENDENT
(computed once; dense at GL scale, supplied or computed by other means at
cavity scale) and P_ctrl = #{ eig(A_c) right of the line } is exact and free.
A design is certified stable-with-margin iff the count is ZERO.  No
continuation, no anchor, no small-steps premise: this is a standalone
certificate for an ARBITRARY design, at one sparse LU + a few solves per
sample point along the line.

Phase tracking is adaptive.  Each sample computes f and f' exactly (f' needs
one extra solve on the SAME LU), the step is proposed from the local
logarithmic derivative |f|/|f'|, and a move is accepted only if the phase
increment is below pi/2 AND the chord is small against |f| at both ends;
otherwise the step is bisected (down to a floor, which aborts loudly rather
than guess).  The two tails are closed analytically: outward marching stops
once the certified tail bound

    |G(s)| <= ||c|| ||b|| / (lam_min(M) (|s| - nu)),    nu = ||M^{-1}A||,

(with nu and lam_min(M) estimated once per plant) keeps |k G| < 1/2 for all
larger |omega|, after which f stays in the disk |w - 1| < 1/2, which cannot
wind, and the tail contribution is the principal argument difference.

The count is exact under one honest caveat, stated rather than hidden: between
samples the tracker assumes f does not oscillate through more than the
accepted phase increment, which the a-posteriori chord/argument checks
validate but do not prove.  A fully rigorous segment bound (Neumann radius per
sample, ~8 extra matvecs per point) is NOT IMPLEMENTED HERE.  Earlier editions
of this docstring advertised it behind a --rigor flag; no such option ever
existed in the argument parser and no such bound was ever computed.  It is
scheduled as the per-interval Neumann bound of pass 2 of the parallel winding
scheme -- algebra note to-do item 5, merged into item 4, implementation stage 6
-- where it doubles as the test that closes quiet intervals without further
samples.  Until then this module offers the fast mode only, which in the
cross-validation below reproduces the dense eigenvalue count exactly on every
tested design, stable and unstable, coupled and static-gain.

    python winding_cert.py --ne 200 --results results            # check all
    python winding_cert.py --ne 200 --results results --case K+B+C
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, glob, json, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sparse_lu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controller as ctl
import massops


# ---------------------------------------------------------------- plant data
def plant_tail_constants(A, M):
    """nu = ||M^{-1}A||_2 and lam_min(M), computed once per plant, used for
    the certified tail bound (Lanczos with a sparse LU of M; see below)."""
    key = (id(A), id(M), A.shape, A.nnz, M.nnz)
    if key in _TAIL_CACHE:
        return _TAIL_CACHE[key]
    out = _plant_tail_constants(A, M)
    _TAIL_CACHE.clear(); _TAIL_CACHE[key] = out
    return out


_TAIL_CACHE = {}


def _plant_tail_constants(A, M):
    """nu = ||M^-1 A||_2 by Lanczos on the Hermitian operator
    (M^-1 A)^H (M^-1 A) = A^H M^-H M^-1 A: sparse products with A and solves
    with a sparse LU of M only (2026-09-28, user: never densify M; replaces a
    dense SVD of M^-1 A at n <= 3000 and a power iteration x 1.3 above it --
    same value to 1e-15 at ne=100 and 200), and lam_min(M)."""
    n = A.shape[0]
    Mlu = massops.lu(M)                                   # the run's one factor of M
    As = A.tocsc().astype(complex)
    OP = spla.LinearOperator((n, n), dtype=complex,
                             matvec=lambda x: As.conj().T @ Mlu.solve(Mlu.solve(As @ x), trans='H'))
    v0 = np.random.default_rng(12345).standard_normal(n)          # fixed start: deterministic runs
    nu = float(np.sqrt(spla.eigsh(OP, k=1, which='LA', tol=1e-12, maxiter=20000, v0=v0.astype(complex),
                                  return_eigenvectors=False)[0].real))
    lam_min = float(spla.eigsh(M.tocsc().real, k=1, which='SM', v0=v0,
                               return_eigenvectors=False)[0])
    return nu, lam_min


def plant_pole_data(A, M, line, keep=0.6, method='sparse', nmax=3000):
    """(count right of line, hot list) where hot = [(Im p, |Re p - line|)] for
    plant eigenvalues within `keep` of the line.  Design-independent: computed
    once.  The hot list matters because the poles of f are exactly the plant
    (and A_c) eigenvalues, and near a pole close to the line the phase of f
    swings through ~2 pi over a width ~ |Re p - line| in omega -- a feature an
    adaptive tracker can step clean over, since |f| is LARGE there and both
    endpoints look benign (measured: a stabilised design read winding 0
    instead of -1).  Known poles get mandatory resolution instead.

    method 'sparse' (default since 2026-09-29, STAGE6 6.13): the eigenvalues
    right of line - keep from a certified sparse enclosure
    (sparse_spectrum.plant_eigs_right: numerical-range rectangle + tiles proved
    empty or exhausted by shift-invert); 'dense': the full generalised
    eigenproblem, a validation reference for n <= nmax only."""
    if method == 'sparse':
        from sparse_spectrum import plant_pole_data_sparse
        return plant_pole_data_sparse(A, M, line, keep)
    n = A.shape[0]
    if n > nmax:
        raise RuntimeError(f"plant_pole_data(method='dense'): n={n} > {nmax}")
    from scipy.linalg import eig as deig
    lam = deig(np.asarray(A.todense()), np.asarray(M.todense()), right=False)
    lam = lam[np.isfinite(lam)]
    cnt = int(np.count_nonzero(lam.real > line))
    hot = [(float(p.imag), float(abs(p.real - line)))
           for p in lam if abs(p.real - line) < keep]
    return cnt, hot


def plant_poles_right(A, M, line):
    """Back-compatible wrapper: count only."""
    return plant_pole_data(A, M, line)[0]


# ---------------------------------------------------------------- the tracker
def _poly_and_deriv(s, roots):
    """prod(s - r_i) and its derivative, by the product rule (r small)."""
    d = s - np.asarray(roots, complex)
    p = complex(np.prod(d))
    pp = 0j
    for j in range(d.size):
        pp += complex(np.prod(np.delete(d, j)))
    return p, pp


def _arg_change_vertical(line, w0, w1, roots):
    """Continuous change of sum_i arg(s - r_i) along Re s = line, Im s from w0
    to w1 (exact).  None if a root lies ON the line."""
    tot = 0.0
    for r_ in np.asarray(roots, complex):
        a_ = line - r_.real
        if a_ == 0.0:
            return None
        tot += np.sign(a_) * (np.arctan((w1 - r_.imag) / abs(a_)) - np.arctan((w0 - r_.imag) / abs(a_)))
    return float(tot)


class _F:
    """f(s) = 1 - k(s) G(s) and f'(s), one sparse LU per sample point."""
    def __init__(self, A, M, b, c, arow, nvec, K, mc=1.0):
        self.A, self.M = A.tocsc(), M.tocsc()
        self.b = np.asarray(b, complex).ravel()
        self.ch = np.conj(np.asarray(c, complex).ravel())
        self.Ac = np.asarray(ctl.A_of(*arow), complex)
        self.r = self.Ac.shape[0]
        self.nv = np.asarray(nvec, complex).ravel()
        self.K = complex(K)
        self.Bc = np.full(self.r, mc, complex)
        self.nsolve = 0

    def __call__(self, s):
        lu = sparse_lu.factor(s * self.M - self.A)
        x = lu.solve(self.b)                       # S^{-1} b
        y = lu.solve(self.M @ x)                   # S^{-1} M S^{-1} b
        self.nsolve += 2
        G = complex(self.ch @ x)
        Gp = complex(-self.ch @ y)
        if self.r and np.any(self.nv):
            Sc = s * np.eye(self.r) - self.Ac
            w = np.linalg.solve(Sc, self.Bc)
            k = self.K + complex(self.nv @ w)
            kp = complex(-self.nv @ np.linalg.solve(Sc, w))
        else:
            k, kp = self.K, 0.0
        f = 1.0 - k * G
        fp = -(kp * G + k * Gp)
        if getattr(self, 'ctrl_exact', False) and self.r and np.any(self.nv):
            # g = p f with p(s) = det(sI - A_c): no poles at the compensator's
            # eigenvalues.  q = p k = (K - 1) p + p2, p2 = det(sI - (A_c - B_c n^T))
            # (matrix determinant lemma), both evaluated from their roots.
            p_, pp_ = _poly_and_deriv(s, self.lam_c)
            p2_, pp2_ = _poly_and_deriv(s, self.lam_c2)
            q_ = (self.K - 1.0) * p_ + p2_
            qp_ = (self.K - 1.0) * pp_ + pp2_
            g = p_ - q_ * G
            gp = pp_ - (qp_ * G + q_ * Gp)
            return g, gp, k, G
        return f, fp, k, G


class _FM:
    """MIMO (2026-09-29, STAGE6): f(s) = det(I_m - K(s) G(s)) and f'(s),
        G(s) = C^H (s M - A)^-1 B  (p x m),   K(s) = D_c + C_c (s I - A_c)^-1 B_c  (m x p),
    det(s M_cl - A_cl) = det(s M - A) det(s I - A_c) f(s) (Schur complements).
    ctrl_exact: g(s) = det(s I - A_c) f(s) = det([[s I - A_c, -B_c G], [-C_c, I - D_c G]])
    -- an (r + m) x (r + m) determinant with no poles at the compensator's
    eigenvalues; g'/g = tr(Blk^-1 Blk').  One sparse LU and 2m solves per point."""
    def __init__(self, A, M, B, C, arow, nvec, K, mc=1.0):
        self.A, self.M = A.tocsc(), M.tocsc()
        B = np.asarray(B, complex); C = np.asarray(C, complex)
        self.b = B[:, None] if B.ndim == 1 else B
        self.ch = (C[:, None] if C.ndim == 1 else C).conj().T          # p x n
        self.m, self.p = self.b.shape[1], self.ch.shape[0]
        self.Ac = np.asarray(ctl.A_of(*arow), complex)
        self.r = self.Ac.shape[0]
        self.Bc = np.asarray(ctl.Bc_of(arow, self.r, mc), complex).reshape(self.r, self.p)
        self.nv = np.asarray(nvec, complex).reshape(self.m, self.r)     # C_c
        self.K = np.asarray(K, complex).reshape(self.m, self.p)         # D_c
        self.nsolve = 0

    def __call__(self, s):
        lu = sparse_lu.factor(s * self.M - self.A)
        X = lu.solve(self.b)                        # S^-1 B
        Y = lu.solve(np.asarray(self.M @ X))        # S^-1 M S^-1 B
        self.nsolve += 2 * self.m
        G = self.ch @ X                             # p x m
        Gp = -self.ch @ Y
        r, m = self.r, self.m
        if getattr(self, 'ctrl_exact', False) and r:
            Blk = np.zeros((r + m, r + m), complex); dBlk = np.zeros_like(Blk)
            Blk[:r, :r] = s * np.eye(r) - self.Ac; Blk[:r, r:] = -self.Bc @ G
            Blk[r:, :r] = -self.nv;                 Blk[r:, r:] = np.eye(m) - self.K @ G
            dBlk[:r, :r] = np.eye(r);               dBlk[:r, r:] = -self.Bc @ Gp
            dBlk[r:, r:] = -self.K @ Gp
            g = np.linalg.det(Blk)
            gp = g * np.trace(np.linalg.solve(Blk, dBlk))
            return g, gp, None, G
        if r:
            Sc = s * np.eye(r) - self.Ac
            W = np.linalg.solve(Sc, self.Bc)
            Ks = self.K + self.nv @ W
            Kp = -self.nv @ np.linalg.solve(Sc, W)
        else:
            Ks, Kp = self.K, np.zeros_like(self.K)
        X_ = np.eye(m) - Ks @ G
        f = np.linalg.det(X_)
        fp = f * np.trace(np.linalg.solve(X_, -(Kp @ G + Ks @ Gp)))
        return f, fp, Ks, G


def _h_hot(w, hot):
    """Mandatory resolution near known poles of f and hinted zeros: within
    their influence window the step may not exceed max(dist/2, |w - Im p|/4),
    so a ~2 pi swing of width ~dist is sampled by many points and the approach
    is geometric."""
    h = np.inf
    for wi, di in hot:
        if abs(w - wi) < 6.0 * di + 1.0:
            h = min(h, max(0.5 * di, 0.25 * abs(w - wi)))
    return h


def _march(F, line, wa, wb, knots, hot, hmin, WT, maxpts):
    """Adaptive march of arg f along Re s = line from Im s = wa to wb (the whole
    contour -W_T..W_T in one call is the serial count).  The step never passes
    an unvisited knot.  Returns dict(total, w, f, f0, npts, ok)."""
    s_of = lambda w: complex(line, w)
    w = wa
    ki = 0
    while ki < len(knots) and knots[ki] <= w + 1e-12:
        ki += 1
    f, fp, *_ = F(s_of(w))
    f0 = f
    total = 0.0
    npts = 1
    ok = True
    h = max(hmin, 0.35 * abs(f) / max(abs(fp), 1e-300))
    h = min(h, WT / 8.0, _h_hot(w, hot))
    while w < wb and npts < maxpts:
        h = min(h, wb - w)
        if ki < len(knots):
            h = min(h, max(hmin, knots[ki] - w))
        while True:
            fm, fpm, *_ = F(s_of(w + 0.5 * h))    # midpoint
            f2, fp2, *_ = F(s_of(w + h))
            npts += 2
            d1 = np.angle(fm / f)
            d2 = np.angle(f2 / fm)
            ok_half = (abs(d1) <= 0.5 * np.pi and abs(d2) <= 0.5 * np.pi
                       and abs(fm - f) <= max(0.75 * min(abs(f), abs(fm)),
                                              1e-14)
                       and abs(f2 - fm) <= max(0.75 * min(abs(fm), abs(f2)),
                                               1e-14))
            # far-side activity check: if either forward point reports a much
            # smaller phase-safe scale than the step, the step landed on or
            # past a feature edge -- bisect even if the increments look calm.
            scale = min(abs(fm) / max(abs(fpm), 1e-300),
                        abs(f2) / max(abs(fp2), 1e-300))
            if ok_half and h <= 1.4 * scale:
                break
            h *= 0.5
            if h < hmin:
                ok = False
                break
        if not ok:
            break
        total += d1 + d2
        w += h
        while ki < len(knots) and knots[ki] <= w + 1e-12:
            ki += 1
        f, fp = f2, fp2
        h = max(hmin, min(0.35 * abs(f) / max(abs(fp), 1e-300), 2.5 * h,
                          WT / 8.0, _h_hot(w, hot)))
    return dict(total=total, w=w, f=f, f0=f0, npts=npts, ok=ok, nsolve=F.nsolve)


def _make_F(A, M, b, c, arow, nvec, K, mc, ctrl_exact, lam_c, lam_c2):
    mimo = np.ndim(b) == 2 or np.ndim(c) == 2
    F = (_FM if mimo else _F)(A, M, b, c, arow, nvec, K, mc=mc)
    if ctrl_exact:
        F.ctrl_exact = True
        F.lam_c = lam_c
        if lam_c2 is not None:
            F.lam_c2 = lam_c2
    return F


def _seg_task(t):
    """One segment of the march in a worker (fullpar): the plant from the
    registered context, the design and the march parameters from the task."""
    import fullpar
    (b, c, arow, nvec, K, mc, ctrl_exact, lam_c, lam_c2, line, wa, wb, knots, hot, hmin, WT, maxpts) = t
    D = fullpar.CTX['D']
    F = _make_F(D.A, D.M, b, c, arow, nvec, K, mc, ctrl_exact, lam_c, lam_c2)
    return _march(F, line, wa, wb, knots, hot, hmin, WT, maxpts)


SEG_PER_WORKER = 4      # segments per worker when the pool is up (dynamic load balance)


# THE TAIL CUTOFF (2026-10-16).  winding_count marches up to W_T, the first rung
# w0 2^k of a doubling ladder where the certified bound on ||k(s) G(s)|| over
# |w'| >= W_T is below q.  The ladder is kept (a design that passed keeps its W_T,
# its knots and its march points); its cap, 1e7 since STAGE5, is now a sanity cap
# of 1e9: with the BFS's H^1 tail alone (decaying like d^-1/2, W_T growing with
# the SQUARE of the feedthrough) the decentral 369.87 record needed 7.9e6, the
# last rung below 1e7, and every later decentral design of 20261010n raised --
# an artefact of the bound (the discrete ||G|| decays like 1/w), not of the
# design.  The plant's L2 tail (BFSPlant.gbound) brings that W_T to 7680.
WT_CAP = 1e9


class TailBoundError(RuntimeError):
    """No tail cutoff: the certified |k| gb(W) stays >= q on every rung up to
    WT_CAP.  Raised BEFORE the march (no LU of the pencil, no time spent), so a
    caller can tell it from a count: the save-time veto does not charge it
    against --veto-counts (run_case_epi.winding_veto).  W the last rung tried,
    kgb = |k| gb there, q the tail's threshold."""
    def __init__(self, W, kgb, q):
        self.W, self.kgb, self.q = float(W), float(kgb), float(q)
        super().__init__(f"no tail cutoff up to W = {self.W:.6g} (cap {WT_CAP:.0e}): |k| gb(W) = {self.kgb:.4g} "
                         f">= q = {self.q:g} (the plant's tail bound times the compensator's high-frequency gain)")

    def __reduce__(self):                  # picklable across the pool
        return (TailBoundError, (self.W, self.kgb, self.q))


def tail_cutoff(kgb, q, w0=30.0):
    """(W_T, W*) for kgb(w), a bound on ||k(s) G(s)|| valid for every |w'| >= w
    (non-increasing in w): W_T the first rung max(w0, 1) 2^k with kgb < q (the
    count's truncation), W* the threshold itself, bisected (geometric, to 1e-6)
    inside the last doubling -- for the logs, which show how far a design is
    from needing a longer march or from the cap.  TailBoundError when no rung up
    to WT_CAP certifies."""
    WT = max(w0, 1.0)
    while not kgb(WT) < q:
        if 2.0 * WT > WT_CAP:
            raise TailBoundError(WT, kgb(WT), q)
        WT *= 2.0
    lo, hi = (0.5 * WT if WT > max(w0, 1.0) else 1.0), WT
    if kgb(lo) < q:
        return WT, lo
    while hi > lo * (1.0 + 1e-6):
        mid = float(np.sqrt(lo * hi))
        if kgb(mid) < q:
            hi = mid
        else:
            lo = mid
    return WT, hi


def _tail_bound(F, mimo, A, M, line, tail, gbound):
    """(kgb, q): w -> the certified bound on ||k(s) G(s)|| for |w'| >= w (inf
    where none), and the threshold q below which f cannot wind on the tails."""
    if gbound is not None:
        # the plant supplies its own bound on |G(line + i w)| (the backward-
        # facing step: no M^-1, a wall-shear sensor not of the form M psi;
        # bfs/plant.BFSPlant.gbound, BFS.md 4)
        nu = lam_min = s_far_nr = Hd_nr = beta_nr = nrm_bc = None
    else:
        nu, lam_min = tail if tail is not None else plant_tail_constants(A, M)
        nrm_bc = float(np.linalg.norm(F.b, 2 if mimo else None)) * float(np.linalg.norm(F.ch, 2 if mimo else None))
    # MESH-INDEPENDENT tail bound (STAGE6 6.29): ||C^H (sM - A)^-1 B|| <=
    # ||M^-1/2 B|| ||M^-1/2 C|| / dist(s, W(A, M)) with the numerical-range
    # distance of the plant heights (sparse_spectrum, once per mesh).  The
    # ||M^-1 A|| bound below grows like h^-2 (W_T = 960 at ne = 100, 194 of 673
    # points beyond |w| = 30); the smaller of the two valid bounds is used.
    if gbound is None:
        from sparse_spectrum import plant_heights, numrange_distance
        s_far_nr, Hd_nr = plant_heights(A, M)
        Mlu_ = massops.lu(M)
        _Bm = np.asarray(F.b, complex).reshape(A.shape[0], -1); _Cm = np.asarray(F.ch, complex).reshape(-1, A.shape[0]).conj().T
        beta_nr = float(np.sqrt(max(np.linalg.eigvalsh(_Bm.conj().T @ Mlu_.solve(_Bm)).max(), 0.0)
                                * max(np.linalg.eigvalsh(_Cm.conj().T @ Mlu_.solve(_Cm)).max(), 0.0)))
    # tail: |det(I - K G) - ...|: with ||K G||_2 <= q every eigenvalue mu of K G has
    # |mu| <= q, arg det = sum arg(1 - mu_i) stays within m arcsin(q) -- no winding
    # for m arcsin(q) < pi/2; SISO keeps q = 1/2 (m = 1), MIMO uses q = 1/(2m)
    q_tail = 0.5 / (F.m if mimo else 1)

    a_c = float(np.linalg.norm(F.Ac, 2)) if F.r else 0.0

    def kmax_at(w):
        # |k(s)| bound for |s| >= w: |K| + ||n|| ||Bc|| ||(sI - A_c)^-1||, with
        # the resolvent bounded by 1/(|s| - ||A_c||) (Neumann; 2026-09-26: was
        # 1/(distance to the poles), which holds only for NORMAL A_c -- the
        # tridiagonal compensator is not, ||A_c|| = 66 against |Im poles| <= 17)
        if mimo:
            kD = float(np.linalg.norm(F.K, 2))
            if F.r and np.any(F.nv):
                if w <= a_c:
                    return np.inf
                return kD + float(np.linalg.norm(F.nv, 2)) * float(np.linalg.norm(F.Bc, 2)) / (w - a_c)
            return kD
        if F.r and np.any(F.nv):
            if w <= a_c:
                return np.inf
            return abs(F.K) + np.linalg.norm(F.nv) * np.linalg.norm(F.Bc) / (w - a_c)
        return abs(F.K)

    def kgb(w):
        # a bound on ||k G|| valid for EVERY |w'| >= w (the bounds on |G| and |k| are
        # non-increasing in w); inf where either has none
        if gbound is not None:
            gb = gbound(w)
            km = kmax_at(w)
            return km * gb if (np.isfinite(gb) and np.isfinite(km)) else np.inf
        s_abs = np.hypot(line, w)
        gb = nrm_bc / (lam_min * (s_abs - nu)) if s_abs > nu else np.inf
        dnr = numrange_distance(w, s_far_nr, Hd_nr, line=line)
        if dnr > 0:
            gb = min(gb, beta_nr / dnr)
        km = kmax_at(w)
        return km * gb if (np.isfinite(gb) and np.isfinite(km)) else np.inf
    return kgb, q_tail


def tail_check(A, M, b, c, arow, nvec, K, line=-0.025, w0=30.0, mc=1.0, tail=None, gbound=None):
    """winding_count's tail cutoff for a design, without the march (no LU of the
    pencil; milliseconds with a plant gbound): dict(wtail, wstar, q, kD, kgb)
    with kD = ||D_c||_2 (|K| SISO) and kgb the bound at W_T.  Raises
    TailBoundError exactly where winding_count would (the driver's pre-screen
    and its tail line at each acceptance, 2026-10-16)."""
    mimo = np.ndim(b) == 2 or np.ndim(c) == 2
    F = (_FM if mimo else _F)(A, M, b, c, arow, nvec, K, mc=mc)
    kgb, q = _tail_bound(F, mimo, A, M, line, tail, gbound)
    WT, Ws = tail_cutoff(kgb, q, w0)
    return dict(wtail=float(WT), wstar=float(Ws), q=float(q), kgb=float(kgb(WT)),
                kD=float(np.linalg.norm(F.K, 2)) if mimo else float(abs(F.K)))


def winding_count(A, M, b, c, arow, nvec, K, line=-0.025, w0=30.0,
                  mc=1.0, p_plant=None, tail=None, hot=None, zeros_hint=None,
                  hmin_frac=1e-6, maxpts=20000, verbose=False, ctrl_exact=False, gbound=None, real_sym=False,
                  hmin_abs=None):
    """#{closed-loop eigenvalues with Re > line}, by the argument principle.

    `hot` is the list of (Im p, dist-to-line) for near-line PLANT poles (from
    plant_pole_data); near-line A_c poles are added here per design.
    `zeros_hint` is a list of complex NEAR-LINE CLOSED-LOOP eigenvalues (e.g.
    the watch list the SMV disk sweep harvests anyway): the fast-phase
    features of f sit at its zeros, which are exactly those eigenvalues, so
    hinted zeros get the same mandatory resolution as known poles.  Unhinted
    zeros are caught by midpoint validation (every accepted step is checked
    at its midpoint, so an aliased feature must fool three collinear samples,
    not two).

    Returns dict(count, winding, p_plant, p_ctrl, npts, nsolve, wtail, wstar, ok)
    (wtail the march's cutoff W_T, wstar the tail bound's threshold W* <= W_T;
    TailBoundError before any march when no W_T up to WT_CAP certifies).
    ok=False means the tracker could not validate a segment down to the step
    floor (f passes too close to zero -- an eigenvalue essentially ON the
    line) and the count must not be trusted; the caller should move the line.
    """
    mimo = np.ndim(b) == 2 or np.ndim(c) == 2
    F = (_FM if mimo else _F)(A, M, b, c, arow, nvec, K, mc=mc)
    pol = np.linalg.eigvals(F.Ac) if F.r else np.zeros(0)
    p_ctrl = int(np.count_nonzero(pol.real > line))
    # CTRL_EXACT (2026-09-26, STAGE5 30): track g = det(sI - A_c) f instead of
    # f.  The compensator's poles are poles of f; on the pair's face S two of
    # them sit 1e-6 from the margin line (a near-double pole), where f swings
    # through ~2 pi (twice) over a width ~1e-6 -- the tracker hit its step floor
    # and the count came back INCONCLUSIVE in half the continuation refusals.
    # g has no poles there; the phase of det(sI - A_c) along the line is added
    # back in closed form from its (exact, r x r) roots.
    ctrl_exact = bool(ctrl_exact and F.r and np.any(F.nv))
    if ctrl_exact:
        F.ctrl_exact = True
        F.lam_c = pol
        if not mimo:
            F.lam_c2 = np.linalg.eigvals(F.Ac - np.outer(F.Bc, F.nv))
    if p_plant is None or hot is None:
        p_plant2, hot2 = plant_pole_data(A, M, line)
        p_plant = p_plant2 if p_plant is None else p_plant
        hot = hot2 if hot is None else hot
    hot = list(hot) + ([] if ctrl_exact else
                       [(float(p.imag), max(1e-6, float(abs(p.real - line))))
                        for p in pol if abs(p.real - line) < 0.6])
    for z in (zeros_hint or []):
        z = complex(z)
        if abs(z.real - line) < 0.6:
            hot.append((float(z.imag), max(1e-6, float(abs(z.real - line)))))
    kgb, q_tail = _tail_bound(F, mimo, A, M, line, tail, gbound)
    WT, Wstar = tail_cutoff(kgb, q_tail, w0)

    # MANDATORY KNOTS.  The adaptive step is bounded only by WT/8, and WT is
    # set by the tail bound (WT ~ 500-4000 at ne = 200 since nu = ||M^-1 A||
    # grows like 1/h^2), so a single step can be hundreds wide and leap the
    # ENTIRE central region where the loop actually acts.  Whether the march
    # slowed down there used to depend on whether some sample happened to
    # LAND near the structure -- eight deeply stable designs measured
    # winding 0 instead of -1 in 49 points while their neighbours measured
    # correctly in 250+ (the false alarms the dense cross-check caught).
    # Coverage must be deterministic, not incidental: the march may never
    # step past an unvisited knot.  Knots = a uniform grid over the core
    # (the band where the loop transfer lives) plus the edges and centres of
    # every hot-pole window.
    Wcore = min(WT, max(w0, (max((abs(wi) for wi, _ in hot), default=0.0)
                             + 2.0)))
    ks = set(float(x) for x in np.linspace(-Wcore, Wcore, 121))
    for wi, di in hot:
        for x in (wi - 6.0 * di - 1.0, wi - 2.0 * di, wi - 0.5 * di, wi,
                  wi + 0.5 * di, wi + 2.0 * di, wi + 6.0 * di + 1.0):
            if -WT < x < WT:
                ks.add(float(x))
    knots = np.array(sorted(ks))
    # adaptive march upward along Re = line from -WT to +WT: in one piece
    # (serial), or in segments between knots farmed to the full-model pool
    # (fullpar, STAGE6 6.30).  The phase changes of the segments add up to the
    # whole contour's; each segment restarts its step rule at its first knot,
    # so the points (not the count) differ from the one-piece march.
    s_of = lambda w: complex(line, w)
    # step floor: relative to the truncation W_T (SISO, unchanged); MIMO caps the
    # reference at 4000 -- a high static gain (abs(D_c) ~ 5e3 with the atom fields)
    # pushes W_T to ~1e5 and a floor of 0.1 in w cannot resolve the plant's
    # near-line poles (INCONCLUSIVE with the right count, 2026-09-29)
    hmin = hmin_frac * (min(WT, 4000.0) if mimo else WT)
    if hmin_abs is not None:
        # an ABSOLUTE floor (20261010j stage B, gate-refusal.md G1): a count right
        # of -eps at a design pinned 1e-4 from that line needs steps well below
        # the eigenvalue's distance, whatever W_T is.  The relative floor is
        # 1e-6 x 4000 = 4e-3 for MIMO and 1e-6 x W_T (3e-2 at the SISO no-strip
        # record's W_T = 30720) for SISO, and GL's 1e-8 x W_T worked only because
        # its W_T was ~60 (6e-7; STAGE5 22: 6e-5 was INCONCLUSIVE at 1.2e-4).
        # A march whose phase scale never fell below the coarse floor takes the
        # same points; where it did, the march goes on with finer steps instead of
        # ending INCONCLUSIVE (or proposing the floor itself).
        hmin = min(hmin, float(hmin_abs))
    import fullpar
    _D = fullpar.CTX.get('D')
    nseg = (min(SEG_PER_WORKER * fullpar.NPROC, max(1, len(knots) // 4))
            if (fullpar.active() and _D is not None and A is _D.A and M is _D.M) else 1)
    # REAL SYSTEMS (real_sym, the backward-facing step): f(conj s) = conj f(s),
    # so the continuous phase obeys psi(-w) = 2 psi(0) - psi(w) and the change
    # over [-W_T, W_T] is twice the change over [0, W_T]; f(line - i W_T) =
    # conj f(line + i W_T) closes the contour.  Half the points.
    w_lo = 0.0 if real_sym else -WT
    if nseg <= 1:
        mr = _march(F, line, w_lo, WT, knots, hot, hmin, WT, maxpts)
        total, w, f, npts, ok = mr['total'], mr['w'], mr['f'], mr['npts'], mr['ok']
        ok = ok and w >= WT          # stopped short (the point budget): the contour is open, not a count
        args = [np.angle(mr['f0'])]
    else:
        kn_ = knots[knots > w_lo]
        idx = np.unique(np.rint(np.linspace(0, len(kn_) - 1, nseg + 1)).astype(int))[1:-1]
        edges = [w_lo] + [float(kn_[j]) for j in idx if w_lo < kn_[j] < WT] + [WT]
        lc2 = getattr(F, 'lam_c2', None)
        segs = fullpar.pmap(_seg_task, [(b, c, arow, nvec, K, mc, ctrl_exact, (F.lam_c if ctrl_exact else None), lc2,
                                         line, edges[j], edges[j + 1], knots, hot, hmin, WT, maxpts)
                                        for j in range(len(edges) - 1)])
        total = float(sum(sg['total'] for sg in segs))
        npts = int(sum(sg['npts'] for sg in segs))
        F.nsolve += int(sum(sg['nsolve'] for sg in segs))
        # a segment that stopped short (step floor, point budget) leaves the
        # contour open: not a count
        ok = all(sg['ok'] and sg['w'] >= edges[j + 1] for j, sg in enumerate(segs))
        w, f = segs[-1]['w'], segs[-1]['f']
        args = [np.angle(segs[0]['f0'])]
    if real_sym:
        total = 2.0 * total
        args = [-np.angle(f)]              # the start of the full contour, line - i W_T
    # close the contour: on the tails |f - 1| < 1/2, so f cannot wind there
    # and the tail contribution is the principal-argument return to f = 1.
    # Contour orientation: the line traversed UPWARD with the region to its
    # RIGHT closed at infinity is negatively oriented for that region, so
    # Z - P = -(total + closing)/2pi.
    if ctrl_exact:
        # total is the phase change of g = p f; f's is total - (change of arg p),
        # and the closing uses f itself at the two ends
        dp = _arg_change_vertical(line, -WT, w, F.lam_c)
        if dp is None:
            ok = False
            dp = 0.0
        p_start, _ = _poly_and_deriv(s_of(-WT), F.lam_c)
        p_end, _ = _poly_and_deriv(s_of(w), F.lam_c)
        total = total - dp
        closing = -np.angle(f / p_end) + np.angle(np.exp(1j * args[0]) / p_start)
    else:
        closing = -np.angle(f) + args[0]          # back to arg(1) = 0 both ends
    Wnum = -(total + closing) / (2.0 * np.pi)
    Wint = int(np.rint(Wnum))
    if abs(Wnum - Wint) > 0.25:
        ok = False
    return dict(count=Wint + p_plant + p_ctrl, winding=Wint,
                winding_raw=float(Wnum), p_plant=int(p_plant),
                p_ctrl=p_ctrl, npts=npts, nsolve=F.nsolve, wtail=float(WT),
                wstar=float(Wstar), ok=bool(ok))


# ---------------------------------------------------------------- CLI
def main():
    import fe_codesign
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results')
    ap.add_argument('--r', type=int, default=-1)
    ap.add_argument('--sob_len', type=float, default=0.8)
    ap.add_argument('--case', default=None)
    ap.add_argument('--eps', type=float, default=0.05,
                    help='stability margin of the runs; the counting line is '
                         'placed at -eps/2 so margin-tight designs (alpha = '
                         '-eps exactly, e.g. K+B+C) keep f bounded away from '
                         'zero on the line')
    ap.add_argument('--line', type=float, default=None,
                    help='override the counting line Re s = line')
    ap.add_argument('--p_plant', type=int, default=-1,
                    help='#plant eigenvalues right of the line; -1 computes '
                         'it (dense at GL scale)')
    ap.add_argument('--plain', action='store_true',
                    help='track f itself (the pre-2026-09-26 count); default: det(sI - A_c) f, compensator phase exact')
    ap.add_argument('--csv', default=None,
                    help='write the completed reference table (every record: gamma, alpha, winding count, verdict) '
                         '-- CLEANUP 5.1, 2026-09-26')
    a = ap.parse_args()
    if a.r < 0:
        from ctrl_order import infer_r
        a.r = infer_r(a.ne, getattr(a, 'results', None))
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=a.sob_len)
    line = a.line if a.line is not None else -0.5 * a.eps
    tail = plant_tail_constants(D.A, D.M)
    pp, hot = plant_pole_data(D.A, D.M, line)
    if a.p_plant >= 0:
        pp = a.p_plant
    print(f"counting line Re = {line:+.4f}; plant poles right of it: {pp}; "
          f"tail constants nu={tail[0]:.3g}, lam_min(M)={tail[1]:.3g}")
    print(f"{'case':12s} {'gamma':>9s} {'#right':>7s} {'wind':>5s} "
          f"{'pts':>5s} {'dense a':>10s}  verdict")
    rows = []
    for jf in sorted(glob.glob(os.path.join(a.results, f"*_ne{a.ne}.json"))):
        m = json.load(open(jf))
        if a.case and m.get('case') != a.case:
            continue
        d = np.load(jf[:-5] + '.npz', allow_pickle=True)
        if 'd' not in d.files:
            continue
        arow = (d['d'], d['e'], d['f'])
        va = d['va'] if d['va'].size else None
        vs = d['vs'] if d['vs'].size else None
        b, c, *_ = D.bc(float(d['xa']), float(d['xs']), va, vs)
        out = winding_count(D.A, D.M, b, c, arow, d['nvec'],
                            complex(d['K']), line=line, mc=D.mc,
                            p_plant=pp, tail=tail, hot=hot, ctrl_exact=not a.plain)
        ad = (f"{float(d['alpha_dense']):+.4f}"
              if 'alpha_dense' in d.files and
              np.isfinite(float(d['alpha_dense'])) else "   --  ")
        if not out['ok']:
            verdict = 'INCONCLUSIVE (f ~ 0 on the line: eigenvalue at the '\
                      'margin; move --line)'
        elif out['count'] == 0:
            verdict = 'CERTIFIED: no eigenvalue right of the line'
        else:
            verdict = f"*** {out['count']} EIGENVALUE(S) RIGHT OF THE LINE ***"
        print(f"{m['case']:12s} {m['gamma']:9.4f} {out['count']:7d} "
              f"{out['winding']:5d} {out['npts']:5d} {ad:>10s}  {verdict}")
        rows.append((m['case'], f"{m['gamma']:.10g}", f"{float(m.get('alpha', np.nan)):.6g}", ad.strip(),
                     str(out['count']) if out['ok'] else '', str(out['winding']), str(out['npts']),
                     f"{line:g}", str(pp), verdict))
    if a.csv:
        with open(a.csv, 'w') as fh:
            fh.write('case,gamma,alpha,alpha_dense,count_right_of_line,winding,points,line,plant_poles_right,verdict\n')
            for r_ in rows:
                fh.write(','.join(r_) + '\n')
        print(f"written: {a.csv} ({len(rows)} records)")


if __name__ == "__main__":
    main()
