"""
sparse_spectrum.py -- the eigenvalues of a large sparse pencil right of a
vertical line, found and CERTIFIED COMPLETE with sparse operations only
(2026-09-29, STAGE6 6.13; user: "everything needs to be sparse, we are about to
increase dimension a lot").  Replaces the dense spectra that set the frequency
band (`band_from_closedloop`), the plant pole data of the winding count
(`winding_cert.plant_pole_data`) and the dumped spectrum.

------------------------------------------------------------------- the plant
Eigenvalues of (A, M), M Hermitian positive definite, with Re lambda >= s0.
With H = (A + A^H)/2, S = (A - A^H)/(2i) (both Hermitian), an eigenpair gives
    Re lambda = v^H H v / v^H M v,     Im lambda = v^H S v / v^H M v.
1. Right edge: Re lambda <= s_far = lambda_max(H, M)            (Lanczos).
2. Height (S-lemma): if Re lambda >= s0 then v^H (H - s0 M) v >= 0, so for any
   tau >= 0
       +-Im lambda <= lambda_max(+-S + tau (H - s0 M), M)       (Lanczos),
   minimised over tau (a few values).  The diffusion that makes the high modes
   stiff makes H very negative on them, so the bound stays mesh-independent
   where the bare numerical range of S grows like 1/h^2.
3. Every eigenvalue with Re >= s0 lies in the rectangle [s0, s_far] x [-W-, W+].
   Cover it with disks: shift-invert Arnoldi at a centre sigma returns the k
   eigenvalues nearest sigma; the disk of radius |lambda_k - sigma| then holds
   exactly the k-1 nearer ones.  k grows until the disk covers its tile, tiles
   split otherwise.  All residuals are checked.

--------------------------------------------------------------- the closed loop
Eigenvalues of (A_cl, M_cl) with Re >= s0 for a given design.  The low-rank
coupling with a high gain defeats the numerical-range bounds (the Hermitian
part carries |D_c| ~ 1e4), so instead:
  - LOCATE candidates: the plant eigenvalues right of s0 - pad and the
    compensator poles (a finite-rank perturbation moves the rest little), the
    high-gain estimates eig(D_c C^H M^-1 B) (the branches that go far), and the
    near-line eigenvalues the certificate harvested; each refined by
    shift-invert on the closed-loop pencil (sparse LU + Woodbury);
  - COUNT the eigenvalues right of s0 with the winding number (argument
    principle, global in w; `winding_cert.winding_count`, whose plant data
    comes from the plant enclosure above);
  - COMPLETE iff located == counted.  Otherwise a line sweep on Re = s0 over a
    widened range harvests more, and the count is compared again.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import massops


def _v0(n, seed=12345):
    return np.random.default_rng(seed).standard_normal(n)


def _lmax(Hop, M, Mlu=None, tol=1e-10):
    """lambda_max of the Hermitian pencil (Hop, M), Lanczos (M positive definite)."""
    n = M.shape[0]
    Minv = None
    if Mlu is not None:
        Minv = spla.LinearOperator((n, n), dtype=complex, matvec=lambda x: Mlu.solve(np.asarray(x, complex)))
    lam = spla.eigsh(Hop, k=1, M=M, Minv=Minv, which='LA', tol=tol, maxiter=20000,
                     v0=_v0(n).astype(complex), return_eigenvectors=False)
    return float(np.max(lam.real))


_SFAR_CACHE = {}


def plant_rectangle(A, M, s0, taus=(0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)):
    """(s_far, W_lo, W_hi): every eigenvalue of (A, M) with Re >= s0 lies in
    [s0, s_far] x [-W_lo, W_hi].  Sparse Lanczos only."""
    Mlu = massops.lu(M); key = (id(A), id(M))
    A0, M0 = A, M
    A = sp.csc_matrix(A, dtype=complex); M = sp.csc_matrix(M, dtype=complex)
    H = ((A + A.conj().T) * 0.5).tocsc()
    S = ((A - A.conj().T) * (-0.5j)).tocsc()
    e = _SFAR_CACHE.get(key)
    if e is None or e[0] is not A0 or e[1] is not M0:     # s_far = lambda_max(H, M): once per plant (was per call)
        e = _SFAR_CACHE[key] = (A0, M0, _lmax(H, M, Mlu))
    s_far = e[2]
    if s_far < s0:
        return s_far, 0.0, 0.0
    G = (H - s0 * M).tocsc()

    def best(sgn):
        # every tau >= 0 gives a valid bound; the minimum over tau is unimodal in
        # practice: stop after two consecutive increases (fewer Lanczos runs)
        b_, up = np.inf, 0
        for t in sorted(taus):
            v = _lmax((sgn * S + t * G).tocsc(), M, Mlu)
            if v < b_:
                b_, up = v, 0
            else:
                up += 1
                if up >= 2:
                    break
        return b_
    Whi = best(1.0); Wlo = best(-1.0)
    return s_far, max(Wlo, 0.0), max(Whi, 0.0)


class PlantPencil:
    """(A, M) sparse: shift solves by a sparse LU."""
    def __init__(self, A, M):
        self._Mlu = massops.lu(M)
        self.A = sp.csc_matrix(A, dtype=complex); self.M = sp.csc_matrix(M, dtype=complex)
        self.n = self.A.shape[0]; self._sig = None; self._lu = None
        self.Mmat = self.M

    def Minv(self, x):
        return self._Mlu.solve(np.asarray(x, complex))

    def shift(self, sigma):
        if self._sig != sigma:
            self._lu = spla.splu((self.A - sigma * self.M).tocsc()); self._sig = sigma

    def solve(self, x):                      # (A - sigma M)^-1 x
        return self._lu.solve(np.asarray(x, complex))

    def solveH(self, x):                     # (A - sigma M)^-H x
        return self._lu.solve(np.asarray(x, complex), trans='H')

    def Aop(self, x):
        return self.A @ x

    def Mop(self, x):
        return self.M @ x


class ClosedLoopPencil:
    """(A_cl, M_cl) of a design: shift solves through the closed-loop
    resolvent's sparse LU + Woodbury (ClosedLoopResolvent / ...MIMO)."""
    def __init__(self, cl, arow, nvec, K):
        self.cl, self.arow, self.nvec, self.K = cl, arow, nvec, K
        self.n = cl.n + cl.r; self._sig = None
        r = cl.r
        self.Mmat = sp.block_diag([sp.csc_matrix(cl.M, dtype=complex), sp.identity(r, dtype=complex, format='csc')]).tocsc()
        self._M = cl.M; self._r = r

    def Minv(self, x):                       # blockdiag(M, I)^-1: the run's one factor of M (massops)
        return massops.block_solve(self._M, self._r, x)

    def shift(self, sigma):
        if self._sig != sigma:
            self.cl.setup(-1j * sigma, self.arow, self.nvec, self.K); self._sig = sigma   # i omega = sigma

    def solve(self, x):                      # (A_cl - sigma M_cl)^-1 = -(sigma M_cl - A_cl)^-1
        return -self.cl.solve(np.asarray(x, complex))

    def solveH(self, x):
        return -self.cl.solveH(np.asarray(x, complex))

    def Aop(self, x):
        return self.cl.Acl_apply(x, self.arow, self.nvec, self.K)

    def Mop(self, x):
        return self.cl.Mcl_apply(x)


def free_radius(P, sigma, safety=1.01):
    """Radius of a PROVABLY eigenvalue-free disk around sigma, for any
    (non-normal) pencil: an eigenpair gives v = (lambda - sigma) (A - sigma M)^-1 M v,
    so |lambda - sigma| >= 1 / ||X||_M with X = (A - sigma M)^-1 M and the M-norm
    ||X||_M^2 = lambda_max of the Hermitian pencil (M X^H... = X^H M X, M) -- Lanczos
    (two shift solves per step; M products and the LU of the shift only).  The
    Ritz value approaches lambda_max from below: `safety` covers the tolerance."""
    P.shift(sigma); n = P.n

    def XHMX(v):
        x = P.solve(P.Mop(v))                       # X v
        return P.Mop(P.solveH(P.Mop(x)))            # M^H (A - sigma M)^-H M x  (M Hermitian)
    Hop = spla.LinearOperator((n, n), dtype=complex, matvec=XHMX)
    Minv = spla.LinearOperator((n, n), dtype=complex, matvec=P.Minv) if hasattr(P, 'Minv') else None
    lam = spla.eigsh(Hop, k=1, M=spla.aslinearoperator(P.Mmat) if hasattr(P, 'Mmat') else None, Minv=Minv,
                     which='LA', tol=1e-6, maxiter=2000, v0=_v0(n).astype(complex), return_eigenvectors=False)
    return 1.0 / (safety * np.sqrt(max(float(lam[0].real), 1e-300)))


def nearest_k(P, sigma, k, maxiter=600):
    """The k eigenvalues of the pencil nearest sigma (shift-invert Arnoldi),
    residual-verified.  Returns (lam sorted by distance, relative residuals)."""
    P.shift(sigma); n = P.n
    op = spla.LinearOperator((n, n), dtype=complex, matvec=lambda x: P.solve(P.Mop(x)))
    k = int(min(k, n - 2))
    mu, V = spla.eigs(op, k=k, which='LM', tol=1e-11, maxiter=maxiter, v0=_v0(n).astype(complex),
                      ncv=min(n - 1, max(2 * k + 1, 20)))
    lam = sigma + 1.0 / mu
    res = np.array([np.linalg.norm(P.Aop(V[:, j]) - lam[j] * P.Mop(V[:, j])) /
                    max(np.linalg.norm(P.Aop(V[:, j])) + abs(lam[j]) * np.linalg.norm(P.Mop(V[:, j])), 1e-300)
                    for j in range(len(lam))])
    o = np.argsort(np.abs(lam - sigma))
    return lam[o], res[o]


def _tile(P, x0, x1, y0, y1, kmax, res_tol, depth, maxdepth, stats):
    c = complex(0.5 * (x0 + x1), 0.5 * (y0 + y1))
    R = 0.5 * np.hypot(x1 - x0, y1 - y0)                  # circumradius
    stats['shifts'] += 1
    if free_radius(P, c) >= R:                           # provably empty
        stats['free'] += 1
        return [], True
    # nearest-k at the centre, k doubled until the k-th nearest lies outside the
    # tile.  An Arnoldi failure (no convergence, or a residual above res_tol) also
    # doubles k rather than splitting: it comes from several eigenvalues at about
    # the same distance (|1/(lambda - c)| clustered), which a larger subspace
    # separates and a smaller tile does not (GL, 2 away from its spectrum: every
    # split failed again).  Split only when kmax is reached.
    k = 4
    while True:
        try:
            lam, res = nearest_k(P, c, k)
            if np.all(res[:-1] <= res_tol) and abs(lam[-1] - c) > R * (1 + 1e-12):
                inside = [complex(l_) for l_ in lam[:-1]
                          if x0 <= l_.real <= x1 and y0 <= l_.imag <= y1]
                return inside, True                      # the k-th nearest lies outside: all inside found
        except spla.ArpackNoConvergence:
            stats['arpack_fail'] = stats.get('arpack_fail', 0) + 1
        if k >= kmax:
            break
        k = min(2 * k, kmax)
    if depth >= maxdepth:
        return [], False
    xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    out, ok = [], True
    for (a0, a1, b0, b1) in ((x0, xm, y0, ym), (xm, x1, y0, ym), (x0, xm, ym, y1), (xm, x1, ym, y1)):
        e_, o_ = _tile(P, a0, a1, b0, b1, kmax, res_tol, depth + 1, maxdepth, stats)
        ok = ok and o_; out += e_
    return out, ok


def eigs_in_rectangle(P, x0, x1, y0, y1, kmax=40, res_tol=1e-8, tile=2.0, maxdepth=6):
    """All eigenvalues of the pencil P in [x0, x1] x [y0, y1]: tiles of side
    <= `tile`, each either PROVED empty (free_radius) or exhausted by the
    nearest-k eigenvalues of its centre (the k-th lying outside the tile),
    split otherwise.  Returns (eigenvalues, ok, stats)."""
    nx = max(1, int(np.ceil((x1 - x0) / tile))); ny = max(1, int(np.ceil((y1 - y0) / tile)))
    xs = np.linspace(x0, x1, nx + 1); ys = np.linspace(y0, y1, ny + 1)
    stats = dict(shifts=0, free=0); out = []; ok = True
    for i in range(nx):
        for j in range(ny):
            e_, o_ = _tile(P, xs[i], xs[i + 1], ys[j], ys[j + 1], kmax, res_tol, 0, maxdepth, stats)
            ok = ok and o_
            for l_ in e_:
                if all(abs(l_ - q) > 1e-8 * max(1.0, abs(l_)) for q in out):
                    out.append(l_)
    return out, ok, stats


_PLANT_CACHE = {}


def plant_eigs_right(A, M, s0, cache=True):
    """All eigenvalues of the plant pencil (A, M) with Re >= s0, certified
    complete by the rectangle enclosure.  Returns (eigenvalues, info)."""
    key = (id(A), id(M), A.shape, float(s0))
    if cache and key in _PLANT_CACHE:
        return _PLANT_CACHE[key]
    s_far, Wlo, Whi = plant_rectangle(A, M, s0)
    # Lanczos Ritz values approach lambda_max from BELOW: pad the edges so the
    # rectangle stays an enclosure (tolerance 1e-10 relative; the pad is 1e3 x)
    s_far += 1e-7 * max(1.0, abs(s_far)) + 1e-6
    Wlo += 1e-7 * max(1.0, Wlo) + 1e-6; Whi += 1e-7 * max(1.0, Whi) + 1e-6
    info = dict(s_far=s_far, W_lo=Wlo, W_hi=Whi)
    if s_far < s0:
        out = (np.zeros(0, complex), dict(info, ok=True, n_shifts=0))
    else:
        lam, ok, st = eigs_in_rectangle(PlantPencil(A, M), s0, s_far, -Wlo, Whi)
        out = (np.array(sorted(lam, key=lambda z: -z.real), complex), dict(info, ok=ok, **st))
    if cache:
        _PLANT_CACHE[key] = out
    return out


def plant_pole_data_sparse(A, M, line, keep=0.6):
    """Sparse replacement of winding_cert.plant_pole_data: (count right of line,
    hot list [(Im p, |Re p - line|)] of the plant eigenvalues within `keep`)."""
    lam, info = plant_eigs_right(A, M, line - keep)
    if not info['ok']:
        raise RuntimeError(f"plant_pole_data_sparse: enclosure not certified ({info})")
    cnt = int(np.count_nonzero(lam.real > line))
    hot = [(float(p.imag), float(abs(p.real - line))) for p in lam if abs(p.real - line) < keep]
    return cnt, hot


def highgain_estimates(M, b, c, K, Mlu=None):
    """eig(D_c C^H M^-1 B): where the high-gain branches of a static feedthrough
    D_c go (STAGE6 6.4).  One sparse solve with the LU of M."""
    B = np.asarray(b, complex); C = np.asarray(c, complex)
    B = B[:, None] if B.ndim == 1 else B; C = C[:, None] if C.ndim == 1 else C
    if Mlu is None:
        Mlu = massops.lu(M)
    G0 = C.conj().T @ Mlu.solve(B)
    Kc = np.asarray(K, complex).reshape(B.shape[1], C.shape[1])
    return np.linalg.eigvals(Kc @ G0)


def closed_loop_right_of(cl, arow, nvec, K, s0, plant_lam, Ac=None, extra=(), Mlu=None, k=4,
                         res_tol=1e-8, band=None, x_right=0.5, tile=2.0, inner=0.0):
    """Closed-loop eigenvalues with Re >= s0, LOCATED from candidates (no
    completeness claim by itself: compare with the winding count right of s0):
    the plant eigenvalues right of s0 - 0.6 (`plant_lam`), the compensator poles,
    the high-gain estimates eig(D_c C^H M^-1 B) and `extra` (e.g. the near-line
    eigenvalues the certificate harvested); each refined by the k eigenvalues of
    the closed-loop pencil nearest it.  Returns (eigenvalues, n_shifts)."""
    P = ClosedLoopPencil(cl, arow, nvec, K)
    cands = [complex(z) for z in plant_lam]
    if Ac is not None and np.size(Ac):
        cands += [complex(z) for z in np.linalg.eigvals(np.asarray(Ac, complex))]
    Bm = cl.b if np.ndim(cl.b) == 2 else np.asarray(cl.b)[:, None]
    Cm = cl.c if np.ndim(cl.c) == 2 else np.asarray(cl.c)[:, None]
    cands += [complex(z) for z in highgain_estimates(cl.M, Bm, Cm, K, Mlu)]
    cands += [complex(z) for z in extra]
    if band is not None and inner > 0.0:
        # the loop's band decision: only eigenvalues beyond |Im| = inner can
        # widen the band; candidates well inside it are skipped
        cands = [z for z in cands if abs(z.imag) > inner - 1.0]
    # merge candidates closer than 1e-3 (one shift serves both)
    shifts = []
    for z in cands:
        if all(abs(z - q) > 1e-3 * max(1.0, abs(z)) for q in shifts):
            shifts.append(z)
    out = []; ns = 0
    for z in shifts:
        try:
            lam, res = nearest_k(P, z, k); ns += 1
        except spla.ArpackNoConvergence:
            try:
                lam, res = nearest_k(P, z, 2 * k); ns += 1
            except spla.ArpackNoConvergence:
                continue
        for l_, r_ in zip(lam, res):
            if r_ <= res_tol and l_.real >= s0 and all(abs(l_ - q) > 1e-8 * max(1.0, abs(l_)) for q in out):
                out.append(complex(l_))
    info = dict(candidate_shifts=ns)
    if band is not None:
        # the band itself, exhaustively: tiles of [s0, X] x [-band, band] proved
        # empty or exhausted by nearest-k (eigs_in_rectangle).  With inner > 0 (the
        # loop's band decision) the tiling is skipped: only eigenvalues beyond
        # |Im| = inner can widen the band, and those come from the candidates
        # (tiling the band's edge strips cost 25 % of a SISO round at ne = 100:
        # Arnoldi far from a non-normal spectrum converges slowly); the whole band
        # is tiled at save, with the count.
        X = max(x_right, max([l_.real for l_ in out] + [s0]) + 0.1)
        if inner > 0.0:
            inb, ok, st = [], True, {}         # the loop's band decision: candidates only (tiling at save)
        else:
            inb, ok, st = eigs_in_rectangle(P, s0, X, -band, band, tile=tile)
        info.update(band_ok=ok, **{'band_' + k_: v_ for k_, v_ in st.items()})
        for l_ in inb:
            if l_.real >= s0 and all(abs(l_ - q) > 1e-8 * max(1.0, abs(l_)) for q in out):
                out.append(complex(l_))
    return sorted(out, key=lambda z: -z.real), info


def closed_loop_set(A, M, cl, arow, nvec, K, s0, band, mc=1.0, count=True, extra=(), ctrl_exact=True, inner=0.0):
    """Every closed-loop eigenvalue with Re >= s0: located (closed_loop_right_of,
    candidates + the band tiled) and, with count=True, COMPARED with the winding
    number right of s0 (argument principle, global in w).  info['complete'] is
    True only when the two agree -- then no eigenvalue right of s0 is missing,
    at any |Im|."""
    import controller as _ctl
    plant_lam, _ = plant_eigs_right(A, M, s0 - 0.6)
    loc, info = closed_loop_right_of(cl, arow, nvec, K, s0, plant_lam, Ac=_ctl.A_of(*arow[:3]),
                                     extra=extra, band=band, inner=(0.0 if count else inner))
    info['located'] = len(loc)
    if count:
        import winding_cert as _wc
        cnt, hot = plant_pole_data_sparse(A, M, s0)
        zh = [z for z in loc if abs(z.real - s0) < 0.3]
        o = _wc.winding_count(A, M, cl.b, cl.c, arow, nvec, K, line=s0, mc=mc, p_plant=cnt, hot=hot,
                              zeros_hint=zh, hmin_frac=1e-8, ctrl_exact=ctrl_exact)
        info.update(count=(int(o['count']) if o['ok'] else None), count_pts=int(o['npts']),
                    complete=bool(o['ok'] and int(o['count']) == len(loc)))
    return loc, info


# ------------------------------------------------------------------ tail bound
_HEIGHT_CACHE = {}


def plant_heights(A, M, depths=(0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5),
                  taus=(0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 1e4, 1e5)):
    """H(d) = the S-lemma bound on |Im z| over the numerical range W(A, M) right of
    Re z = -d (plant_rectangle), at a few depths; with s_far.  Once per mesh."""
    key = (id(A), id(M), A.shape, tuple(depths))
    if key in _HEIGHT_CACHE:
        return _HEIGHT_CACHE[key]
    Hd = []
    for d in depths:
        sf, wl, wh = plant_rectangle(A, M, -float(d), taus=taus)
        Hd.append((float(d), max(wl, wh) + 1e-7 * max(1.0, wl, wh) + 1e-6))
    out = (sf + 1e-7 * max(1.0, abs(sf)) + 1e-6, Hd)
    _HEIGHT_CACHE[key] = out
    return out


def numrange_distance(w, s_far, Hd):
    """A lower bound on dist(i w, W(A, M)): the part of W right of -d lies within
    |Im| <= H(d) (distance >= |w| - H(d)), the part left of -d is >= d away."""
    best = 0.0
    for d, h in Hd:
        best = max(best, min(d, abs(w) - h))
    return best


def tail_frequency(gamma, s_far, Hd, beta_gamma, kbound, wmin=1.0, wmax=1e9):
    """Smallest W such that sigma_1(w) < gamma for every |w| >= W, from
        sigma_1 <= 1 / (dist(i w, W(A, M)) - rho_K(w)),   rho_K(w) = ||M^-1/2 B|| ||M^-1/2 C|| kbound(|w|)
    (the numerical range of the closed-loop plant block A + B K(iw) C^H lies in
    W(A) + the disk of radius rho_K); kbound(|w|) bounds ||K(i w')|| for all
    |w'| >= |w| (non-increasing).  inf if no W up to wmax works."""
    def ok(w):
        rho = beta_gamma * kbound(w)
        dist = numrange_distance(w, s_far, Hd) - rho
        return np.isfinite(rho) and dist > 0 and 1.0 / dist < gamma
    hi = wmin
    while not ok(hi):
        hi *= 2.0
        if hi > wmax:
            return np.inf
    lo = hi / 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def compensator_bound(Ac, Bc, Cc, Dc):
    """kbound(w): a bound on ||K(i w')|| = ||D_c + C_c (i w' - A_c)^-1 B_c|| valid for
    every |w'| >= w, non-increasing: ||D_c|| + min(Neumann ||C_c|| ||B_c|| / (w - ||A_c||),
    modal ||C_c V|| ||V^-1 B_c|| / min_k dist(i w', lambda_k)) -- the modal distance
    bounded below over |w'| >= w by the poles' |Re| and (w - |Im lambda_k|)."""
    Ac = np.asarray(Ac, complex); r = Ac.shape[0]
    Bc = np.asarray(Bc, complex).reshape(r, -1); Cc = np.atleast_2d(np.asarray(Cc, complex))
    Cc = Cc.reshape(-1, r)
    Dn = float(np.linalg.norm(np.atleast_2d(Dc), 2))
    An = float(np.linalg.norm(Ac, 2)); CBn = float(np.linalg.norm(Cc, 2) * np.linalg.norm(Bc, 2))
    lam, V = np.linalg.eig(Ac)
    try:
        CBm = float(np.linalg.norm(Cc @ V, 2) * np.linalg.norm(np.linalg.solve(V, Bc), 2))
    except np.linalg.LinAlgError:
        CBm = np.inf

    def kb(w):
        neu = CBn / (w - An) if w > An else np.inf
        # min over |w'| >= w and over k of |i w' - lambda_k|: at least |Re lambda_k|,
        # and at least w - |Im lambda_k| when w > |Im lambda_k|
        dk = [max(abs(l.real), w - abs(l.imag)) for l in lam]
        mod = CBm / min(dk) if min(dk) > 0 else np.inf
        return Dn + min(neu, mod)
    return kb, dict(D=Dn, Ac=An, CcBc=CBn, CcBc_modal=CBm)


def closed_loop_tail(A, M, b, c, Ac, Bc, Cc, Dc, gamma, Mlu=None):
    """The tail frequency of a design (tail_frequency) with its constants, all
    sparse: ||M^-1/2 B||^2 = lambda_max(B^H M^-1 B) (one LU of M), the
    compensator r x r (compensator_bound)."""
    B = np.asarray(b, complex); C = np.asarray(c, complex)
    B = B[:, None] if B.ndim == 1 else B; C = C[:, None] if C.ndim == 1 else C
    if Mlu is None:
        Mlu = massops.lu(M)
    bn = float(np.sqrt(np.linalg.eigvalsh(B.conj().T @ Mlu.solve(B)).max()))
    cn = float(np.sqrt(np.linalg.eigvalsh(C.conj().T @ Mlu.solve(C)).max()))
    s_far, Hd = plant_heights(A, M)
    kb, kinfo = compensator_bound(Ac, Bc, Cc, Dc)
    W = tail_frequency(gamma, s_far, Hd, bn * cn, kb)
    return W, dict(beta=bn, gamma_c=cn, s_far=s_far, heights=Hd, **kinfo)
