"""lattice.py -- the inner loop on the FULL model, its factorisations kept on a
lattice and owned by the MPI ranks (BFS.md 3 and 5; user 2026-10-06: drop the
reduced model, parallelise the full one).

Why a lattice.  The plant factor of i w M - A (or s M - A on the stability
line) does not depend on the design: ports and controller enter by Woodbury.
A factor computed once can therefore serve every later design -- but only if
the SAME shift comes back.  The working frequencies (|Omega|, the followed
peaks' tops) and the stability line's disk centres are chosen adaptively, so
they are put on a fixed dyadic lattice: w = k 2^-l.  Points at a coarse level
are points of every finer level, so refining a bracket or a gap never discards
what is cached.  Each rank keeps the factors it computed (least recently used
dropped beyond LU_CAP), and the driver sends every task to the rank that holds
its point (`LatticeOracle`): a task is then a few tens of solves instead of a
factorisation plus those solves.

What runs per SQP evaluation -- rows, line and strip TOGETHER, in waves
(LatticeOracle.evaluate): each wave sends every outstanding request at once,
one batch per rank, the driver computing its own share:
  rows      sigma_1 and its gradient at every member of Omega.  A fixed member at
            its lattice point on 2^-ROW_LEVEL (cached).  A member that follows
            its peak by safeguarded NEWTON at exact frequencies (_newton: the
            reduced path's peakfollow.follow on the full model, one member per
            rank, a step and its halvings side by side): the row is the top's
            exact value and gradient.  Its points are new at every evaluation, so
            their factors stay out of the lattice cache (side cache, XCAP).
            (2026-10-08/09: a bracket on the lattice left the row up to 1e-6 x
            sigma below the top, the point jumping between lattice points from one
            evaluation to the next -- noise in the SQP's merit.)  sigma_2 blocks
            at the row point.
  line      the stability line Re = -eps, Im in [wlo, whi], covered by
            eigenvalue-free disks (the certificate of certify_sparse.
            stability_and_abscissa) in WAVES: the base grid (ranks/2 equispaced
            centres) and every centre that covered the line at the previous call
            (cached, in parallel), then new centres in the remaining gaps, on
            lattice points, kept for the next calls -- until the line is covered,
            an eigenvalue is found on or right of it (not clear), or a gap is
            shorter than the finest level (inconclusive, never clear).  The proof
            is the cover; where the centres sit changes the cost, not the rigour.
  strip     the strip [-eps - ROW_REACH, -eps], covered by a quadtree of cells and
            the disks of every shift of the evaluation (LatticeOracle._strip):
            every eigenvalue in it is exhibited and gets its stability row (left
            eigenvector on the same factor), exact.  The line alone saw only the
            eigenvalues nearest to some point of it (2026-10-09).
  sweep     sigma_1 and its slope on a coarse level of the band (missed peaks),
            values for the certificate's reference sweep.

Determinism: every eigensolve starts from the fixed ARPACK vector of the serial
code, so a row or a disk does not depend on the rank that computed it.

Worker state lives in this module (one per process, the driver's included):
the cache _CACHE (a dict in least-recently-used order), the last designs, and
the eigenvectors of the last line call.  The tasks are module-level functions
of picklable arguments, run by bfs/mpipool.py's ranks or in this process; one
task kind (eval_task) serves every request.  A new point goes to the least
loaded rank holding the fewest factors (LatticeOracle._assign).
"""
from __future__ import annotations
import time
import numpy as np

# ---- settings (UPPERCASE: synced to the ranks by bfs/mpipool.module_snapshot)
LU_CAP = 24          # plant factorisations kept per rank (least recently used dropped; 133 MB each on the
                     # test plant: 24 x 96 ranks ~ 350 GB of a 450 GB allocation -- 8 on 192 GB nodes)
ROW_LEVEL = 10       # a fixed member of Omega is evaluated at its lattice point on 2^-10
FOLLOW_GTOL = 1e-7   # a followed top: |sigma'| <= FOLLOW_GTOL x sigma (value error ~ sigma'^2 / 2|sigma''|, ~1e-15)
FOLLOW_FD = 1e-5     # finite-difference step for sigma'' when no secant curvature is carried
FOLLOW_ITMAX = 8     # Newton steps per evaluation
FOLLOW_TRUST = 0.05  # largest Newton step (and the step when the curvature is not negative)
FOLLOW_HALVE = 3     # a Newton step and its halvings evaluated side by side (one wave); twice before giving up
XCAP = 2             # factors of the exact (off-lattice) points kept per rank, apart from the lattice cache
LINE_BASE_MIN = 4    # the line's base grid: ranks/2 equispaced centres, at least this many
LINE_LMAX = 17       # finest centre spacing 2^-17 (7.6e-6, the serial certificate's eps_int 1e-5)
LINE_K = 1           # nearest eigenvalues per centre (k = 3 costs 2-4 x the applications; 2026-10-06, test plant)
LINE_SHRINK = 0.9    # disk used = LINE_SHRINK x the distance to the nearest eigenvalue (serial code: 0.9)
LINE_WAVES = 60      # waves before the sweep is declared inconclusive
LINE_TOL = 1e-9      # ARPACK tolerance at a centre (the serial certificate's)
ROW_REACH = 0.25     # the STRIP [-eps - ROW_REACH, -eps]: covered at every evaluation, every eigenvalue in it a row
STRIP_RHO = 2.0 ** -6   # the cover may leave out a ball of this radius around each eigenvalue it exhibits
STRIP_HMIN = 2.0 ** -10 # a cell this small and still uncovered: the strip is reported incomplete
STRIP_SPLIT = 2      # geometric refinements of an uncovered cell per wave before new shifts are placed
STRIP_WAVES = 30     # waves before the strip is reported incomplete
WARM_NCV = 0         # > 0: sigma_1 restarted from the last vector at the same point with this subspace (4: 4-5
                     # applications instead of 9-17).  Off by default: where sigma_1 and sigma_2 swap between designs
                     # (kinks; the MIMO plateau) the restart vector lies along the new sigma_2 and a small subspace can
                     # converge there; cold, a row does not depend on its rank.  The line's disks are always cold.
SWEEP_LEVEL = 5      # missed-peak sweep and the certificate's reference sweep on 2^-5
TASK_COST = dict(row=1.0, value=1.0, slope=1.0, value2=2.0, block=2.0, centre=3.0,
                 rowx=1.0, slopex=1.0, blockx=2.0)   # relative costs of the requests
LU_COST = 2.0        # ... and of a factorisation (balancing, LatticeOracle._assign)

_CACHE = {}          # complex shift key (closedloop's: complex(omega)) -> factor, LRU order
_DES = {}            # design key -> dict(b, c, pa, ps, na, ns, cl)
_VEC = {}            # (design key, theta, rounded lambda) -> (v, u) of the last line tasks
_V0 = {}             # point key -> the last right singular vector of sigma_1 there (warm starts)
_XCACHE = {}         # the exact points' factors (XCAP, FIFO): kept out of the lattice cache
STATS = dict(hit=0, miss=0)       # this process's cache accounting


def snap(w, level):
    """The lattice point of level `level` nearest w (an exact dyadic float)."""
    h = 2.0 ** -int(level)
    return float(np.round(float(w) / h) * h)


# --------------------------------------------------------------------- worker
def _design(dz):
    """The design's ports and closed loop, built once per design and process
    (the last four kept); its closed loop uses this module's cache."""
    import fullpar
    d = _DES.get(dz['key'])
    if d is not None:
        return d
    D = fullpar.CTX['D']
    b, c, pa, ps, na, ns = D.bc(dz['xa'], dz['xs'])
    cl = D.cl(b, c)
    cl.lu_cache, cl.lu_cache_max, cl.lu_lru, cl.lu_pin = _CACHE, int(LU_CAP), True, None
    d = dict(b=b, c=c, pa=pa, ps=ps, na=na, ns=ns, cl=cl, D=D)
    while len(_DES) >= 4:
        _DES.pop(next(iter(_DES)))
    _DES[dz['key']] = d
    for k in [k for k in _VEC if k[0] != dz['key']]:
        _VEC.pop(k)
    return d


def _note(key):
    if complex(key) in _CACHE:
        STATS['hit'] += 1
    else:
        STATS['miss'] += 1


def _grads(d, dz, x, z, eta=1.0):
    """The raw gradient dict of D.grads (packed by the driver's packer)."""
    D = d['D']
    return D.grads(d['cl'], x, z, dz['arow'], dz['nvec'], dz['K'], dz['xa'], dz['xs'],
                   d['pa'], d['ps'], d['na'], d['ns'], dz['flags'], eta=eta)


def _v0w(n):
    from closedloop_sparse import _v0
    return _v0(n)


def _gain(d, dz, w, vectors=True):
    """sigma_1 at w, restarted from the last vector there when WARM_NCV."""
    key = complex(w)
    _note(key)
    v0 = _V0.get(key) if WARM_NCV else None
    r = d['cl'].gain(float(w), dz['arow'], dz['nvec'], dz['K'], want_vectors=vectors,
                     v0=(_v0w(d['cl'].n) if (v0 is None and WARM_NCV) else v0),
                     ncv=(WARM_NCV if v0 is not None else None))
    if WARM_NCV:
        _V0[key] = r['v'] / np.linalg.norm(r['v'])
        if len(_V0) > 4 * max(int(LU_CAP), 1):
            for k_ in [k_ for k_ in _V0 if k_ not in _CACHE]:
                _V0.pop(k_)
    return r


def _exact(d, fn):
    """fn() with the closed loop's factor cache switched to the exact points'
    own (a Newton point is new at every evaluation: in the lattice cache it would
    only push the lattice's factors out)."""
    cl = d['cl']
    saved = (cl.lu_cache, cl.lu_cache_max, cl.lu_lru)
    cl.lu_cache, cl.lu_cache_max, cl.lu_lru = _XCACHE, int(XCAP), False
    try:
        return fn()
    finally:
        cl.lu_cache, cl.lu_cache_max, cl.lu_lru = saved


def _hermite_top(x0, x1, f0, f1, d0, d1):
    """Maximum of the cubic Hermite interpolant on [x0, x1]: (w, value,
    second derivative there)."""
    h = x1 - x0
    a1, a2, a3 = h * d0, 3.0 * (f1 - f0) - h * (2.0 * d0 + d1), 2.0 * (f0 - f1) + h * (d0 + d1)
    ts = [0.0, 1.0]
    # roots of p'(t) = a1 + 2 a2 t + 3 a3 t^2, in the cancellation-free form
    # (a3 -> 0 near a quadratic top: the textbook formula lost the root)
    disc = a2 * a2 - 3.0 * a3 * a1
    if disc >= 0.0:
        q = -(a2 + np.copysign(np.sqrt(disc), a2))
        if abs(a3) > 1e-300:
            ts.append(q / (3.0 * a3))
        if abs(q) > 1e-300:
            ts.append(a1 / q)
    best = max((f0 + a1 * t + a2 * t * t + a3 * t ** 3, t) for t in ts if 0.0 <= t <= 1.0)
    return x0 + best[1] * h, best[0], (2.0 * a2 + 6.0 * a3 * best[1]) / (h * h)


def _resident():
    return list(_CACHE.keys())


def _block(d, dz, w):
    """The sigma_2 (Clarke 2x2) block at w: the two leading pairs and the raw
    gradients of the entries the SQP reads; sigma_1's row is its (0, 0) entry."""
    _note(complex(w))
    P = d['cl'].gain_k(float(w), dz['arow'], dz['nvec'], dz['K'], k=2)
    B = dict(sig=[P[0]['sigma'], P[1]['sigma']],
             g00=_grads(d, dz, P[0]['x'], P[0]['z']), g11=_grads(d, dz, P[1]['x'], P[1]['z']),
             g01=_grads(d, dz, P[1]['x'], P[0]['z']), g01i=_grads(d, dz, P[1]['x'], 1j * P[0]['z']),
             g10=_grads(d, dz, P[0]['x'], P[1]['z']), g10i=_grads(d, dz, P[0]['x'], 1j * P[1]['z']))
    return dict(sigma=P[0]['sigma'], gprime=P[0]['gprime'], g=B['g00'], block=B)


def _nearest(cl, s, dz, k):
    """The k eigenvalues of the closed loop nearest the shift s, verified by
    residual (closedloop_sparse.nearest_eig, all of them and their vectors)."""
    import scipy.sparse.linalg as spla
    from closedloop_sparse import _v0
    N = cl.n + cl.r
    arow, nvec, K = dz['arow'], dz['nvec'], dz['K']
    cl.setup(-1j * s, arow, nvec, K)
    OP = spla.LinearOperator((N, N), matvec=lambda x: -cl.solve(cl.Mcl_apply(x)), dtype=complex)
    kk = min(k, N - 2)
    th = V = None
    complete = True
    try:
        th, V = spla.eigs(OP, k=kk, which='LM', tol=LINE_TOL, maxiter=300, v0=_v0(N), ncv=min(N - 1, max(2 * kk + 2, 24)))
    except spla.ArpackNoConvergence as e:
        complete = False
        if e.eigenvalues is not None and len(e.eigenvalues):
            th, V = e.eigenvalues, e.eigenvectors
    if th is None or len(th) == 0:
        return [], [], 0.0
    keep = np.abs(th) > 1e-300
    th, V = th[keep], V[:, keep]
    lam = s + 1.0 / th
    ok = []
    for i in range(len(lam)):
        v = V[:, i]; nv = np.linalg.norm(v)
        if nv == 0:
            continue
        res = np.linalg.norm(cl.Acl_apply(v, arow, nvec, K) - lam[i] * cl.Mcl_apply(v)) / nv
        if res < 1e-6 * max(1.0, abs(lam[i])):
            ok.append(i)
    # the free radius: the distance to the nearest of ALL returned Ritz values
    # (an unverified one only shrinks the disk), and no disk at all when the k
    # nearest did not all converge or one failed its residual -- the nearest
    # eigenvalue may then be missing from the set (the serial strict code's
    # untrusted disks); verified eigenvalues still count as witnesses and rows
    rad = float(np.min(np.abs(lam - s))) if (complete and len(ok) == len(lam) and len(lam)) else 0.0
    return [complex(lam[i]) for i in ok], [V[:, i] for i in ok], rad


def _left(cl, s, lams):
    """Left eigenvectors (u^H A_cl = lam u^H M_cl) of the eigenvalues `lams`
    nearest s, on the same factor: OP2 u = (A_cl - s M_cl)^-H M_cl u has the
    eigenvalues conj(1 / (lam - s))."""
    import scipy.sparse.linalg as spla
    from closedloop_sparse import _v0
    N = cl.n + cl.r
    OP = spla.LinearOperator((N, N), matvec=lambda x: -cl.solveH(cl.Mcl_apply(x)), dtype=complex)
    kk = min(max(len(lams), 1), N - 2)
    try:
        th, U = spla.eigs(OP, k=kk, which='LM', tol=LINE_TOL, maxiter=300, v0=_v0(N, seed=54321),
                          ncv=min(N - 1, max(2 * kk + 2, 24)))
    except spla.ArpackNoConvergence as e:
        th, U = (e.eigenvalues, e.eigenvectors) if e.eigenvalues is not None else (np.zeros(0), None)
    out = []
    for lam in lams:
        tgt = np.conj(1.0 / (lam - s))
        if th is None or len(th) == 0:
            out.append(None); continue
        j = int(np.argmin(np.abs(th - tgt)))
        out.append(U[:, j] if abs(th[j] - tgt) <= 1e-6 * abs(tgt) else None)
    return out


def _centre(d, dz, sre, th, rlo):
    """One disk centre s = sre + i th (on the line Re = -eps, or inside the
    strip): the verified nearest eigenvalues, the free radius, and the stability
    rows of those with Re >= rlo (left eigenvectors on the same factor)."""
    cl = d['cl']
    s = complex(float(sre), float(th))
    _note(complex(-1j * s))
    lams, vs, rad = _nearest(cl, s, dz, LINE_K)
    rows = []
    near = [j for j, l_ in enumerate(lams) if l_.real >= rlo]
    if near:
        us = _left(cl, s, [lams[j] for j in near])
        for j, u in zip(near, us):
            if u is None:
                continue
            v = vs[j]
            eta = np.vdot(u, cl.Mcl_apply(v))
            if abs(eta) < 1e-14 * np.linalg.norm(u) * np.linalg.norm(v):
                continue
            g = _grads(d, dz, v, u, eta=eta)
            rows.append(dict(lam=lams[j], g=g, eta=complex(eta)))
            _VEC[(dz['key'], float(sre), float(th), round(lams[j].real, 10), round(lams[j].imag, 10))] = (v, u, complex(eta))
    return dict(sre=float(sre), theta=float(th), lams=lams, rad=rad, rows=rows)


def eval_task(t):
    """Every evaluation of one rank in one wave: [(request)] -> results.
    Requests (LatticeOracle._drive): ('row', w) sigma_1, its slope and raw
    gradient; ('block', w) the sigma_2 block (and sigma_1's row); ('value', w)
    sigma_1 alone; ('slope', w) sigma_1 and its slope; ('value2', w) sigma_1
    and sigma_2; ('rowx', w), ('slopex', w), ('blockx', w) the same at an exact
    (off-lattice) point, factor in the side cache; ('centre', sre, theta, rlo)
    a disk centred at sre + i theta, rows for its eigenvalues with Re >= rlo."""
    dz, reqs = t
    t0 = time.perf_counter()
    d = _design(dz)
    out = []
    for q in reqs:
        kind = q[0]
        if kind == 'row':
            r = _gain(d, dz, q[1])
            res = dict(sigma=r['sigma'], gprime=r['gprime'], g=_grads(d, dz, r['x'], r['z']))
        elif kind == 'block':
            res = _block(d, dz, q[1])
        elif kind == 'value':
            res = dict(sigma=_gain(d, dz, q[1], vectors=False)['sigma'])
        elif kind == 'slope':
            r = _gain(d, dz, q[1])
            res = dict(sigma=r['sigma'], gprime=r['gprime'])
        elif kind in ('rowx', 'slopex'):
            w = float(q[1])
            r = _exact(d, lambda: d['cl'].gain(w, dz['arow'], dz['nvec'], dz['K']))
            res = dict(sigma=r['sigma'], gprime=r['gprime'])
            if kind == 'rowx':
                res['g'] = _grads(d, dz, r['x'], r['z'])
        elif kind == 'blockx':
            res = _exact(d, lambda: _block(d, dz, q[1]))
        elif kind == 'value2':
            _note(complex(q[1]))
            P = d['cl'].gain_k(float(q[1]), dz['arow'], dz['nvec'], dz['K'], k=2)
            res = dict(sigma=P[0]['sigma'], sigma2=P[1]['sigma'])
        elif kind == 'centre':
            res = _centre(d, dz, q[1], q[2], q[3])
        else:
            raise ValueError(kind)
        out.append(res)
    return dict(out=out, resident=_resident(), dt=time.perf_counter() - t0, hit=STATS['hit'], miss=STATS['miss'])


def vec_task(t):
    """The (v, u, eta) a centre task of this design kept, by (sre, theta, lambda)."""
    key, items = t
    return dict(out=[_VEC.get((key, float(sr), float(th), round(l_.real, 10), round(l_.imag, 10))) for sr, th, l_ in items],
                resident=_resident(), dt=0.0, hit=STATS['hit'], miss=STATS['miss'])


def stats_task(t):
    return dict(out=[], resident=_resident(), dt=0.0, hit=STATS['hit'], miss=STATS['miss'])


def drop_task(t):
    """Forget everything (a new plant order, a new run phase)."""
    _CACHE.clear(); _DES.clear(); _VEC.clear(); _V0.clear(); _XCACHE.clear()
    return dict(out=[], resident=[], dt=0.0, hit=STATS['hit'], miss=STATS['miss'])




# --------------------------------------------------------------------- driver
def _newton(w0, rad, hint, st, real, R):
    """One followed peak by safeguarded Newton at EXACT frequencies (the reduced
    path's peakfollow.follow, on the full model, wave by wave): a generator that
    yields its requests and reads their results from R once the driver has
    evaluated them (LatticeOracle._drive); returns (top, evaluations).

    The slope sigma' comes exactly from the gain's own solves; the curvature is
    carried from the last evaluation (st['g2'], secant-updated at every step),
    else taken by finite differences (two points, one wave).  A step and its
    halvings are evaluated side by side (FOLLOW_HALVE points, one wave), and the
    longest that does not lower sigma is taken; twice before giving up.  Stops
    at |sigma'| <= FOLLOW_GTOL x sigma.  The row is the top's exact value and
    gradient -- no lattice, so no grid noise in the SQP's merit (2026-10-09: the
    bracket on the lattice left the row up to 1e-6 x sigma below the top, the
    point jumping between lattice points from one evaluation to the next).  The
    points' factors go to the side cache (eval_task, _exact)."""
    lo, hi = w0 - rad, w0 + rad
    if real:
        lo = max(lo, 0.0)
    x = float(hint) if (hint is not None and lo <= float(hint) <= hi) else float(w0)
    yield [('rowx', x)]
    e = R[('rowx', x)]; n = 1
    g2 = st.get('g2')
    carried = g2 is not None
    trust = min(FOLLOW_TRUST, rad)
    for _ in range(FOLLOW_ITMAX):
        g1 = e['gprime']
        if abs(g1) <= FOLLOW_GTOL * max(1.0, e['sigma']):
            break
        if g2 is None or not g2 < 0.0:
            pts = [x + FOLLOW_FD, x - FOLLOW_FD]
            yield [('slopex', p_) for p_ in pts]; n += 2
            g2 = (R[('slopex', pts[0])]['gprime'] - R[('slopex', pts[1])]['gprime']) / (2.0 * FOLLOW_FD)
            carried = False
        stp = -g1 / g2 if (g2 < 0.0 and abs(g2) > 1e-16) else float(np.sign(g1)) * trust
        stp = float(np.clip(stp, -trust, trust))
        xt = None
        for batch in range(2):
            cands = [x + stp * 0.5 ** (k + batch * FOLLOW_HALVE) for k in range(FOLLOW_HALVE)]
            cands = [c for c in cands if lo <= c <= hi and c != x]
            if not cands:
                continue
            yield [('rowx', c) for c in cands]; n += len(cands)
            ok = [c for c in cands if R[('rowx', c)]['sigma'] >= e['sigma']]
            if ok:
                xt = ok[0]
                break
        if xt is None:
            if carried:
                g2 = None; carried = False             # the carried curvature failed: one fresh FD attempt
                continue
            break
        et = R[('rowx', xt)]
        dx = xt - x
        if abs(dx) > 1e-12:
            g2s = (et['gprime'] - g1) / dx             # secant update of the curvature
            g2 = g2s if g2s < 0.0 else None
        x, e = xt, et
    if g2 is not None and g2 < 0.0:
        st['g2'] = float(g2)
    return x, n


def _key(q):
    """The cache key of a request's shift (closedloop's complex(omega), omega =
    -i s: a centre s = sre + i theta has the key theta - i sre)."""
    return complex(q[2], -q[1]) if q[0] == 'centre' else complex(q[1])


class LatticeOracle:
    """The driver's side: where each lattice point lives, the batches, the
    waves.  pool: bfs/mpipool.MPIPool (rank 0 = this process computes its own
    share) or None (everything here)."""

    def __init__(self, D, pk, flags, pool=None, use_driver=True, real=True):
        self.D, self.pk, self.flags, self.pool = D, pk, flags, pool
        self.size = pool.size if pool is not None else 1
        self.use_driver = bool(use_driver) or self.size == 1
        self.real = bool(real)
        self.held = {}                   # rank -> the point keys whose factors it holds (its last report)
        self.tops = {}                   # the followed tops of the last evaluation
        self.line_prev = {}              # eps -> centres that covered the line at the last call
        self.strip_prev = {}             # eps -> the strip's shifts (sre, theta) that served its last cover
        self.calls = {}                  # kind -> [calls, tasks, wall, critical path (longest rank), misses]
        self.verbose = False
        self._dkey = None; self._dz = None
        self._line_memo = {}             # (design key, eps, wlo, whi, eps_int) -> (reach, result): the last few designs
        self._rot = 0                    # the ranks' tie-break order rotates from one wave to the next

    # ---------------------------------------------------------------- plumbing
    def design(self, z):
        z = np.asarray(z, float)
        zk = hash(z.tobytes())
        if self._dkey == zk:
            return self._dz
        arow, nvec, K, xa, xs, va, vs = self.pk.unpack(z)
        if va is not None or vs is not None:
            raise NotImplementedError("lattice oracle: free fields are not supported")
        self._dkey, self._dz = zk, self.design_raw(arow, nvec, K, xa, xs)
        return self._dz

    def design_raw(self, arow, nvec, K, xa, xs):
        """The design dict from its parts (the gates' calls carry no packed z)."""
        parts = [np.asarray(v, complex).ravel() for v in tuple(arow) + (nvec, K)] + \
                [np.asarray(xa, float).ravel(), np.asarray(xs, float).ravel()]
        key = hash(b''.join(np.ascontiguousarray(v).tobytes() for v in parts))
        return dict(key=key, arow=arow, nvec=nvec, K=K, xa=xa, xs=xs, flags=dict(self.flags))

    def _ranks(self):
        return list(range(self.size)) if self.use_driver else list(range(1, self.size))

    def _assign(self, reqs):
        """reqs -> {rank: [request]}.  A point goes to a rank holding its factor
        unless that rank's batch would then cost more than a factorisation on the
        least loaded one (TASK_COST, LU_COST).  A point no rank holds goes to the
        least loaded rank and, among equally loaded ones, to the one holding the
        fewest factors -- it evicts nothing while that rank has room -- the
        remaining ties in an order rotating from one wave to the next (the first
        ranks used to take every new point of the small waves, and their caches
        thrashed: 5816 factorisations on the driver in the central MIMO run)."""
        ranks = self._ranks()
        n = len(ranks)
        self._rot = (self._rot + 1) % n
        order = {r: (j - self._rot) % n for j, r in enumerate(ranks)}
        load = {r: 0.0 for r in ranks}
        nheld = {r: len(self.held.get(r, ())) for r in ranks}
        out = {r: [] for r in ranks}
        where = {}
        for r, ks in self.held.items():
            if r in load:
                for k in ks:
                    where.setdefault(k, []).append(r)

        def least():
            return min(ranks, key=lambda r_: (load[r_], nheld[r_], order[r_]))
        items = [(_key(q), q, TASK_COST[q[0]]) for q in reqs]
        for k, q, c in [it for it in items if it[0] in where]:
            r = min(where[k], key=load.get)
            rmin = least()
            if load[r] > load[rmin] + LU_COST:
                r = rmin; c += LU_COST; nheld[r] += 1
            out[r].append(q); load[r] += c
        for k, q, c in [it for it in items if it[0] not in where]:
            r = least()
            out[r].append(q); load[r] += c + LU_COST; nheld[r] += 1
        return {r: v for r, v in out.items() if v}

    def _run(self, fn, per_rank, kind, ntask=None):
        """fn(args) on each rank of per_rank {rank: args}; this process does
        rank 0's share meanwhile.  Returns {rank: result}."""
        t0 = time.perf_counter()
        res = {}
        others = {r: a for r, a in per_rank.items() if r != 0}
        if others and self.pool is not None:
            ranks = self.pool.each(fn, others)
            try:
                if 0 in per_rank:
                    res[0] = fn(per_rank[0])
            finally:
                res.update(self.pool.collect(ranks))
        else:
            for r, a in per_rank.items():
                res[r] = fn(a)
        for r, o in res.items():
            self.held[r] = set(complex(k) for k in o['resident'])
        c = self.calls.setdefault(kind, [0, 0, 0.0, 0.0, 0])
        c[0] += 1; c[1] += int(ntask if ntask is not None else len(per_rank))
        c[2] += time.perf_counter() - t0
        c[3] += max((o['dt'] for o in res.values()), default=0.0)
        c[4] = sum(o['miss'] for o in res.values())
        return res

    def _drive(self, dz, agents, kind, R):
        """Run the agents -- generators of requests (eval_task) that read their
        results from R -- side by side: every wave sends all their outstanding
        requests at once, one batch per rank, and resumes each agent.
        Returns ({name: the agent's return value}, waves)."""
        done, pend = {}, {}

        def step(nm, g):
            try:
                while True:
                    q = [x for x in next(g) if x not in R]
                    if q:
                        pend[nm] = (g, q)
                        return
            except StopIteration as e:
                done[nm] = e.value
                pend.pop(nm, None)
        for nm, g in agents.items():
            step(nm, g)
        waves = 0
        while pend:
            waves += 1
            reqs = list(dict.fromkeys(q for _, qs in pend.values() for q in qs))
            per = self._assign(reqs)
            res = self._run(eval_task, {r: (dz, qs) for r, qs in per.items()}, kind, ntask=len(reqs))
            for r, qs in per.items():
                for q, o in zip(qs, res[r]['out']):
                    o['rank'] = r
                    R[q] = o
            for nm in list(pend):
                step(nm, pend[nm][0])
        return done, waves

    def report(self):
        """One line for the ledger: calls (waves), tasks, wall and critical path
        by kind, and the factors held."""
        held = sum(len(v) for v in self.held.values())
        parts = [f"{k} {c[0]} waves / {c[1]} tasks, {c[2]:.0f} s wall (longest rank {c[3]:.0f} s)"
                 for k, c in sorted(self.calls.items())]
        return f"lattice ({self.size} ranks): " + "; ".join(parts) + f"; {held} factors held"

    def counts(self):
        """(factorisations, cached hits) over all ranks so far."""
        res = self._run(stats_task, {r: None for r in self._ranks()}, 'stats')
        return sum(o['miss'] for o in res.values()), sum(o['hit'] for o in res.values())

    # -------------------------------------------------------------- evaluation
    def evaluate(self, z, W, follow=None, w_star=None, state=None, enroll=(), line=None, strip=True):
        """sigma_1 rows at every member of W and, with line = (eps, wlo, whi),
        the stability line and the strip -- in ONE sequence of waves: the first
        carries the fixed rows, every followed member's first Newton point, the
        line's and the strip's first centres; the later ones the Newton steps,
        the line's gap centres and the strip's new shifts side by side.
        follow: {member key: radius} (the followed peaks); w_star / state: per
        member, the last top and the carried curvature (updated here); enroll:
        the members whose sigma_2 block is built, at the row point.  Returns
        (s, Gs, blocks, tops, info, line result or None); the line result's rows
        are every eigenvalue the line and the strip exhibited in the strip.
        strip = False: the line only (a line-search trial: its admissibility is
        the line's certificate; the rows the QP reads come from the iterate)."""
        dz = self.design(z)
        follow = follow or {}
        w_star = {} if w_star is None else w_star
        state = {} if state is None else state
        enroll = set(int(i) for i in (enroll or ()))
        R = {}

        def member(w, rad, hint, st, blk):
            if rad:
                x, nev = yield from _newton(float(w), float(rad), hint, st, self.real, R)
                if blk:
                    yield [('blockx', x)]
                return x, ('blockx' if blk else 'rowx'), nev
            x = snap(float(w), ROW_LEVEL)
            yield [('block' if blk else 'row', x)]
            return x, ('block' if blk else 'row'), 1

        agents = {}
        for i, w in enumerate(W):
            k6 = round(float(w), 6)
            rad = follow.get(k6)
            hint = st = None
            if rad:
                st = state.setdefault(k6, {})
                hint = w_star.get(k6)
                if hint is None and self.tops:
                    # a member placed at a top of an earlier evaluation (the
                    # round's end moves members there): start on that top
                    q = min(self.tops, key=lambda x: abs(x - float(w)))
                    if abs(q - float(w)) <= 0.5 * float(rad):
                        hint = q
            agents[i] = member(w, rad, hint, st, i in enroll)
        mk = lr = None
        if line is not None:
            eps, wlo, whi = (float(v) for v in line)
            mk = (dz['key'], round(eps, 12), wlo, whi, None)
            hit = self._line_memo.get(mk)
            if hit is not None and hit[0] >= ROW_REACH:
                lr = hit[1]                                # this design's line, swept already
            else:
                agents['line'] = self._line(eps, wlo, whi, ROW_REACH, None, R)
            if strip and (lr is None or 'strip' not in lr):
                agents['strip'] = self._strip(eps, wlo, whi, ROW_REACH, R)
        done, waves = self._drive(dz, agents, 'eval', R)
        if 'line' in done:
            lr = done['line']
        if 'strip' in done:
            lr = self._merge(lr, done['strip'], R)
        if 'line' in done or 'strip' in done:
            self._memo(mk, ROW_REACH, lr)
        m = len(W)
        if follow:
            self.tops = {}
        NZ = len(self.pk.z0())
        s = np.zeros(m); Gs = np.zeros((m, NZ)); blocks = {}; tops = {}
        info = dict(nev=0, newton_max=0, waves=waves)
        grad = lambda g_: np.asarray(self.pk.grad(g_), float)
        for i in range(m):
            x, kind, nev = done[i]
            r = R[(kind, x)]
            s[i] = r['sigma']; Gs[i] = grad(r['g']); info['nev'] += nev
            if kind in ('rowx', 'blockx'):
                k6 = round(float(W[i]), 6)
                w_star[k6] = float(x); tops[k6] = float(x)
                self.tops[float(x)] = None
                info['newton_max'] = max(info['newton_max'], nev)
            if i in enroll:
                B = r['block']
                Gc = np.zeros((2, 2, NZ), complex)
                Gc[0, 0] = grad(B['g00']); Gc[1, 1] = grad(B['g11'])
                Gc[0, 1] = grad(B['g01']) + 1j * grad(B['g01i'])
                Gc[1, 0] = grad(B['g10']) + 1j * grad(B['g10i'])
                blocks[i] = dict(sig=np.array(B['sig']), Gc=Gc)
        return s, Gs, blocks, tops, info, lr

    def sigmas(self, z, ws, k=1, slope=False):
        """sigma_1 values (no vectors) at the lattice points ws; k = 2: the
        pairs (sigma_1, sigma_2) as an (len(ws), 2) array; slope: the pairs
        (sigma_1, its slope in w)."""
        dz = self.design(z)
        reqs = [('slope' if slope else ('value' if k == 1 else 'value2'), float(w)) for w in ws]
        R = {}

        def once():
            yield reqs
        self._drive(dz, {0: once()}, 'sweep', R)
        if slope:
            return np.array([(R[q]['sigma'], R[q]['gprime']) for q in reqs]).reshape(len(ws), 2)
        if k == 1:
            return np.array([R[q]['sigma'] for q in reqs])
        return np.array([(R[q]['sigma'], R[q]['sigma2']) for q in reqs]).reshape(len(ws), 2)

    def grid(self, wlo, whi, level=None):
        """The lattice points of `level` in [wlo, whi]."""
        h = 2.0 ** -int(SWEEP_LEVEL if level is None else level)
        k0, k1 = int(np.ceil(wlo / h - 1e-12)), int(np.floor(whi / h + 1e-12))
        return np.arange(k0, k1 + 1) * h

    # -------------------------------------------------------------------- line
    def line(self, z, eps, wlo, whi, reach=None, eps_int=None):
        """The stability line Re = -eps, Im in [wlo, whi], alone (the gates;
        evaluate() sweeps it beside the rows).  z: the packed design, or a design
        dict (design_raw).  reach = -inf: no stability rows.  Returns dict(clear,
        alpha, witness, rows, seen, nshift, waves, uncovered)."""
        dz = z if isinstance(z, dict) else self.design(z)
        reach = ROW_REACH if reach is None else float(reach)
        mk = (dz['key'], round(float(eps), 12), float(wlo), float(whi), eps_int)
        hit = self._line_memo.get(mk)
        if hit is not None and hit[0] >= reach:
            return hit[1]                                  # this design's line, swept already (same or more rows)
        R = {}
        done, _ = self._drive(dz, {0: self._line(float(eps), float(wlo), float(whi), reach, eps_int, R)}, 'line', R)
        self._memo(mk, reach, done[0])
        return done[0]

    def _memo(self, mk, reach, out):
        self._line_memo[mk] = (reach, out)
        while len(self._line_memo) > 8:
            self._line_memo.pop(next(iter(self._line_memo)))

    def base_grid(self, wlo, whi):
        """The line's base grid: ranks/2 equispaced centres on [wlo, whi] (at
        least LINE_BASE_MIN), each on its nearest point of 2^-LINE_LMAX -- one wave,
        half the ranks, the other half free for the gaps (user, 2026-10-09)."""
        n = max(self.size // 2, LINE_BASE_MIN)
        return sorted(set(snap(wlo + j * (whi - wlo) / (n - 1), LINE_LMAX) for j in range(n)))

    def _line(self, eps, wlo, whi, reach, eps_int, R):
        """The line's waves (module doc), a generator of ('centre', -eps,
        theta, -eps - reach) requests for _drive; returns the line's result."""
        hmin = 2.0 ** -LINE_LMAX if eps_int is None else float(eps_int)
        key = round(float(eps), 12)
        rlo = -eps - reach
        cent = sorted(set(self.base_grid(wlo, whi)) |
                      set(c for c in self.line_prev.get(key, []) if wlo - 1e-12 <= c <= whi + 1e-12))
        done = {}
        waves = 0; witness = None; nshift = 0
        while cent and waves < LINE_WAVES:
            waves += 1
            reqs = [('centre', -eps, float(th), rlo) for th in cent]
            yield reqs
            for q in reqs:
                done[q[2]] = R[q]
            nshift += len(cent)
            for c in done.values():
                for l_ in c['lams']:
                    if l_.real > -eps or abs(l_.real + eps) < 1e-9 * max(1.0, abs(l_)):
                        if witness is None or l_.real > witness.real:
                            witness = l_
            if witness is not None:
                break
            gaps = self._gaps(done, wlo, whi)
            cent = []
            for a_, b_ in gaps:
                L = b_ - a_
                if L < hmin:
                    continue
                # as many centres as the neighbouring disks' radii suggest (one
                # when they are large: bisection), evenly over the gap, each on
                # the lattice point nearest its place at the first level finer
                # than a quarter of the spacing
                ra = [c['rad'] for th, c in done.items() if c['rad'] > 0 and
                      (abs(th + LINE_SHRINK * c['rad'] - a_) < 1e-12 or abs(th - LINE_SHRINK * c['rad'] - b_) < 1e-12)]
                r_est = min(ra) if ra else L
                nc = int(min(32, max(1, np.ceil(L / (2.0 * LINE_SHRINK * r_est)))))
                sp = L / nc
                lev = min(LINE_LMAX, int(np.ceil(np.log2(4.0 / sp))))
                for j in range(nc):
                    c = snap(a_ + (j + 0.5) * sp, lev)
                    if c in done:
                        # a centre whose eigensolve failed (no disk) sits there:
                        # the quarter points, one level finer
                        for c2 in (snap(c - 0.25 * sp, lev + 1), snap(c + 0.25 * sp, lev + 1)):
                            if a_ < c2 < b_ and c2 not in done and lev + 1 <= LINE_LMAX:
                                cent.append(c2)
                    elif a_ < c < b_:
                        cent.append(c)
            cent = sorted(set(cent))
            if self.verbose:
                print(f"      line wave {waves}: {nshift} shifts so far, {len(gaps)} gap(s) "
                      f"({sum(b_ - a_ for a_, b_ in gaps):.3g} uncovered), {len(cent)} new centre(s)", flush=True)
        gaps = self._gaps(done, wlo, whi)
        uncovered = float(sum(b_ - a_ for a_, b_ in gaps))
        clear = witness is None and uncovered <= 0.0
        if witness is None:
            # next call's first wave: the base grid and a cover of the line (the
            # rows of the eigenvalues near the line come from the strip)
            self.line_prev[key] = sorted(self._cover(done, wlo, whi))
        seen, rows = [], []
        for th in sorted(done):
            c = done[th]
            for l_ in c['lams']:
                if all(abs(l_ - q) > 1e-8 * max(1.0, abs(l_)) for q in seen):
                    seen.append(l_)
            for rw in c['rows']:
                if all(abs(rw['lam'] - q['lam']) > 1e-8 * max(1.0, abs(rw['lam'])) for q in rows):
                    rows.append(dict(rw, sre=c['sre'], theta=th, rank=c['rank']))
        rows.sort(key=lambda r_: -r_['lam'].real)
        alpha = max((l_.real for l_ in seen), default=-np.inf)
        if witness is not None:
            alpha = max(alpha, witness.real)
        return dict(clear=bool(clear), alpha=float(alpha), witness=witness, rows=rows, seen=seen,
                    nshift=nshift, waves=waves, uncovered=uncovered)

    # ------------------------------------------------------------------- strip
    def _strip(self, eps, wlo, whi, d, R):
        """The strip [-eps - d, -eps] x [wlo, whi], COVERED AT EVERY EVALUATION
        (user, 2026-10-09), a generator of ('centre', sre, theta, -eps - d)
        requests for _drive beside the line's.  Every eigenvalue in the strip is
        then exhibited, and each gets its stability row: the line's centres see
        only the eigenvalues nearest to some point of the line, and one behind
        another (-0.1444 behind -0.0723 on the central record 191.11) had no row,
        so the QP could not see it coming.

        The cover: a quadtree of cells over the strip (first cells d x d); a cell
        is covered when it lies inside one eigenvalue-free disk of ANY shift of
        this evaluation (line or strip; radius LINE_SHRINK x the distance to the
        nearest eigenvalue, as on the line) or inside the ball of radius
        STRIP_RHO around an exhibited eigenvalue.  An uncovered cell is refined
        STRIP_SPLIT times on the disks at hand, and each sub-cell still uncovered
        gets a shift at its centre (on a 2-D dyadic point: the factor serves the
        next evaluations) in the next wave.  Complete when no cell is left; a
        cell refined down to STRIP_HMIN, or STRIP_WAVES waves, leaves it
        incomplete (reported).  What the cover proves: no eigenvalue in the strip
        other than the exhibited ones, up to the STRIP_RHO balls around them.
        Returns dict(complete, shifts, waves, holes)."""
        key = round(float(eps), 12)
        rlo = -eps - d
        x0, x1 = -eps - d, -eps
        h0 = 2.0 ** np.floor(np.log2(d))
        nre = max(1, int(np.ceil((x1 - x0) / h0 - 1e-12)))
        nim = max(1, int(np.ceil((whi - wlo) / h0 - 1e-12)))
        cells = [(x0 + i * h0, min(x0 + (i + 1) * h0, x1), wlo + j * h0, min(wlo + (j + 1) * h0, whi))
                 for i in range(nre) for j in range(nim)]
        prev = self.strip_prev.get(key)
        shifts = set(prev) if prev else {self._cshift(c) for c in cells}
        asked, used, holes = set(), [], []
        waves = 0; complete = False
        while True:
            new = sorted(sh for sh in shifts if sh not in asked)
            if new:
                waves += 1
                asked |= set(new)
                yield [('centre', sr, th, rlo) for sr, th in new]
            S, r, own, E = self._disks(R)
            todo = []
            for c in cells:
                todo += self._resolve(c, S, r, own, E, STRIP_SPLIT, used)
            small = [c for c in todo if max(c[1] - c[0], c[3] - c[2]) <= STRIP_HMIN]
            holes += small
            todo = [c for c in todo if max(c[1] - c[0], c[3] - c[2]) > STRIP_HMIN]
            if not todo:
                complete = not holes
                break
            if waves >= STRIP_WAVES:
                holes += todo
                break
            # the uncovered cells' shifts; a cell whose shift was evaluated already
            # is refined further on the disks at hand (no new wave for it)
            shifts = {self._cshift(c) for c in todo}
            cells = todo
        # the next evaluation's first wave: a small set of the strip's shifts
        # whose disks cover every covered cell (greedy set cover: each cell lists
        # the disks containing it), and those that exhibited an eigenvalue in the
        # strip (the line's centres come back with the line); without the
        # reduction the set grew with every move of the design (sandbox, central
        # zero start: 800 shifts at the fourth iterate)
        keep = set()
        left = [set(o_) for o_ in used]
        while left:
            cnt = {}
            for o_ in left:
                for sh in o_:
                    cnt[sh] = cnt.get(sh, 0) + 1
            best = max(sorted(cnt), key=lambda sh: cnt[sh])
            keep.add(best)
            left = [o_ for o_ in left if best not in o_]
        keep = set(sh for sh in keep if sh[0] != -eps)
        for q, o in R.items():
            if q[0] == 'centre' and q[1] != -eps and any(l_.real >= rlo for l_ in o['lams']):
                keep.add((q[1], q[2]))
        self.strip_prev[key] = sorted(keep)
        if self.verbose:
            print(f"      strip: {len(asked)} shift(s) in {waves} wave(s), {'complete' if complete else 'INCOMPLETE'} "
                  f"({len(holes)} hole(s)); {len(keep)} kept", flush=True)
        return dict(complete=bool(complete), shifts=len(asked), waves=waves, holes=len(holes))

    @staticmethod
    def _cshift(c):
        """A cell's shift: its centre on the dyadic point of the first level finer
        than an eighth of the cell (inside the cell)."""
        w_ = max(c[1] - c[0], c[3] - c[2])
        lev = int(np.ceil(-np.log2(w_))) + 3
        sr = min(max(snap(0.5 * (c[0] + c[1]), lev), c[0]), c[1])
        th = min(max(snap(0.5 * (c[2] + c[3]), lev), c[2]), c[3])
        return (float(sr), float(th))

    @staticmethod
    def _disks(R):
        """The eigenvalue-free disks (centre, radius, (sre, theta)) and the
        exhibited eigenvalues of every centre evaluated so far."""
        S, r, own, E = [], [], [], []
        for q in sorted(k for k in R if k[0] == 'centre'):
            o = R[q]
            if o['rad'] > 0:
                S.append(complex(q[1], q[2])); r.append(LINE_SHRINK * o['rad']); own.append((q[1], q[2]))
            for l_ in o['lams']:
                if all(abs(l_ - e_) > 1e-8 * max(1.0, abs(l_)) for e_ in E):
                    E.append(l_)
        return np.array(S, complex), np.array(r, float), own, np.array(E, complex)

    @classmethod
    def _resolve(cls, c, S, r, own, E, depth, used):
        """The sub-cells of the cell c left uncovered after `depth` refinements
        (a rectangle lies in a disk iff its four corners do); records the disks
        containing each covered cell."""
        corners = np.array([complex(c[0], c[2]), complex(c[0], c[3]), complex(c[1], c[2]), complex(c[1], c[3])])
        if len(S):
            j = np.flatnonzero(np.all(np.abs(corners[:, None] - S[None, :]) <= r[None, :], axis=0))
            if len(j):
                used.append(tuple(own[int(i_)] for i_ in j))
                return []
        if len(E) and np.any(np.all(np.abs(corners[:, None] - E[None, :]) <= STRIP_RHO, axis=0)):
            return []
        if depth == 0 or max(c[1] - c[0], c[3] - c[2]) <= STRIP_HMIN:
            return [c]
        xm, ym = 0.5 * (c[0] + c[1]), 0.5 * (c[2] + c[3])
        out = []
        for ch in ((c[0], xm, c[2], ym), (xm, c[1], c[2], ym), (c[0], xm, ym, c[3]), (xm, c[1], ym, c[3])):
            out += cls._resolve(ch, S, r, own, E, depth - 1, used)
        return out

    @staticmethod
    def _merge(lr, st, R):
        """The line's result with the rows and eigenvalues of every centre of the
        evaluation (line and strip), and the strip's report."""
        seen, rows = list(lr['seen']), list(lr['rows'])
        for q in sorted(k for k in R if k[0] == 'centre'):
            o = R[q]
            for l_ in o['lams']:
                if all(abs(l_ - e_) > 1e-8 * max(1.0, abs(l_)) for e_ in seen):
                    seen.append(l_)
            for rw in o['rows']:
                if all(abs(rw['lam'] - x_['lam']) > 1e-8 * max(1.0, abs(rw['lam'])) for x_ in rows):
                    rows.append(dict(rw, sre=o['sre'], theta=o['theta'], rank=o['rank']))
        rows.sort(key=lambda r_: -r_['lam'].real)
        # alpha stays the line's: a trial (line only) and its iterate (line and
        # strip) must read the same value for the same design
        return dict(lr, rows=rows, seen=seen, strip=st)

    @staticmethod
    def _gaps(done, wlo, whi):
        iv = sorted((th - LINE_SHRINK * c['rad'], th + LINE_SHRINK * c['rad']) for th, c in done.items() if c['rad'] > 0)
        gaps, x = [], wlo
        for a_, b_ in iv:
            if a_ > x:
                gaps.append((x, min(a_, whi)))
            x = max(x, b_)
            if x >= whi:
                break
        if x < whi:
            gaps.append((x, whi))
        return [(a_, b_) for a_, b_ in gaps if b_ > a_]

    @staticmethod
    def _cover(done, wlo, whi):
        """A small subset of the centres whose disks still cover [wlo, whi]
        (greedy interval cover): the next call's first wave."""
        iv = sorted(((th - LINE_SHRINK * c['rad'], th + LINE_SHRINK * c['rad'], th) for th, c in done.items()
                     if c['rad'] > 0), key=lambda q: q[0])
        out, x, j = [], wlo, 0
        while x < whi and j < len(iv):
            best = None
            while j < len(iv) and iv[j][0] <= x:
                if best is None or iv[j][1] > best[1]:
                    best = iv[j]
                j += 1
            if best is None:
                x = iv[j][0]                                  # a gap: continue past it
                continue
            out.append(best[2]); x = best[1]
        return out

    def vectors(self, z, rows):
        """(v, u, eta) of line rows, from the ranks that computed them."""
        dz = z if isinstance(z, dict) else self.design(z)
        per = {}
        for j, rw in enumerate(rows):
            per.setdefault(rw['rank'], []).append((j, rw['sre'], rw['theta'], rw['lam']))
        res = self._run(vec_task, {r: (dz['key'], [(sr, th, l_) for _, sr, th, l_ in v]) for r, v in per.items()},
                        'vectors', ntask=len(rows))
        out = [None] * len(rows)
        for r, v in per.items():
            for (j, _, _, _), vec in zip(v, res[r]['out']):
                out[j] = vec
        return out

    def drop(self):
        self._run(drop_task, {r: None for r in self._ranks()}, 'drop')
        self.held.clear(); self.line_prev.clear(); self.strip_prev.clear(); self.tops.clear(); self._line_memo.clear()
