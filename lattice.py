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

What runs on the lattice, per SQP evaluation (all tasks of a step at once, one
batch per rank, the driver computing its own share):
  rows      sigma_1 and its gradient at every member of Omega; a member that
            follows its peak is placed by BRACKET AND REFINE on the lattice: walk
            along the slope (sigma' from the same solves) to a sign change, the
            cubic Hermite fit on the bracket estimates how far the top lies above
            the better end point, and the bracket is halved (one finer point)
            until that deficit is below FOLLOW_TOL x sigma.  The row is the end
            point: an EXACT value and gradient of sigma_1 at a lattice frequency
            (t >= sigma_1(w) holds at any w), at most the deficit below the top;
            the fit never enters the QP.  sigma_2 blocks likewise, at the point.
  line      the stability line Re = -eps, Im in [wlo, whi], covered by
            eigenvalue-free disks (the certificate of certify_sparse.
            stability_and_abscissa) in WAVES: first every centre that covered
            the line at the previous call (cached, in parallel), then one new
            centre per remaining gap, on the lattice point nearest its middle at
            the first level finer than a quarter of the gap -- until the line is
            covered, an eigenvalue is found on or right of it (not clear), or a
            gap is shorter than the finest level (inconclusive, never clear).
            The proof is the cover; where the centres sit changes the cost, not
            the rigour.  The LINE_K nearest eigenvalues at each centre are
            verified by residual; those within ROW_REACH of the line come back
            with their left eigenvectors' gradient rows (the SQP's stability
            rows, exact), and the two rightmost with their vectors on request
            (the closed-loop Hermite block).
  sweep     sigma_1 values on a coarse level of the band (missed peaks, the
            certificate's reference sweep).

Determinism: every eigensolve starts from the fixed ARPACK vector of the serial
code, so a row or a disk does not depend on the rank that computed it.

Worker state lives in this module (one per process, the driver's included):
the cache _CACHE (a dict in least-recently-used order), the last designs, and
the eigenvectors of the last line call.  The tasks are module-level functions
of picklable arguments, run by bfs/mpipool.py's ranks or in this process.
"""
from __future__ import annotations
import time
import numpy as np

# ---- settings (UPPERCASE: synced to the ranks by bfs/mpipool.module_snapshot)
LU_CAP = 16          # plant factorisations kept per rank (least recently used dropped)
ROW_LEVEL = 10       # a fixed member of Omega is evaluated at its lattice point on 2^-10
FOLLOW_L0 = 6        # a followed peak's first bracket spacing 2^-6
FOLLOW_LMAX = 16     # finest bracket 2^-16 (1.5e-5)
FOLLOW_TOL = 1e-6    # bracket refined until the Hermite top is within this (relative) of the row
FOLLOW_STEPS = 12    # walk steps along the slope before the bracket search gives up (window edge)
LINE_L0 = 1          # first call of a run: line centres on 2^-1, then the gaps' waves
LINE_LMAX = 17       # finest centre spacing 2^-17 (7.6e-6, the serial certificate's eps_int 1e-5)
LINE_K = 1           # nearest eigenvalues per centre (k = 3 costs 2-4 x the applications; 2026-10-06, test plant)
LINE_SHRINK = 0.9    # disk used = LINE_SHRINK x the distance to the nearest eigenvalue (serial code: 0.9)
LINE_WAVES = 60      # waves before the sweep is declared inconclusive
LINE_TOL = 1e-9      # ARPACK tolerance at a centre (the serial certificate's)
ROW_REACH = 0.25     # stability rows for the eigenvalues within this of the line (to its left)
WARM_NCV = 4         # sigma_1 restarted from the last vector at the same point with this subspace (0: cold, the
                     # serial code's start); the line's disks always cold (a warm start did not help there)
SWEEP_LEVEL = 5      # missed-peak sweep and the certificate's reference sweep on 2^-5
GAIN_COST, LU_COST = 1.0, 3.0   # relative costs for the balancing (a cached task, a factorisation)

_CACHE = {}          # complex shift key (closedloop's: complex(omega)) -> factor, LRU order
_DES = {}            # design key -> dict(b, c, pa, ps, na, ns, cl)
_VEC = {}            # (design key, theta, rounded lambda) -> (v, u) of the last line tasks
_V0 = {}             # point key -> the last right singular vector of sigma_1 there (warm starts)
STATS = dict(hit=0, miss=0)       # this process's cache accounting


def level_of(h):
    return int(round(-np.log2(h)))


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


def _hermite_top(x0, x1, f0, f1, d0, d1):
    """Maximum of the cubic Hermite interpolant on [x0, x1]: (w, value)."""
    h = x1 - x0
    a1, a2, a3 = h * d0, 3.0 * (f1 - f0) - h * (2.0 * d0 + d1), 2.0 * (f0 - f1) + h * (d0 + d1)
    ts = [0.0, 1.0]
    if abs(a3) > 1e-300:
        disc = a2 * a2 - 3.0 * a3 * a1
        if disc >= 0.0:
            r = np.sqrt(disc)
            ts += [(-a2 + r) / (3.0 * a3), (-a2 - r) / (3.0 * a3)]
    elif abs(a2) > 1e-300:
        ts.append(-a1 / (2.0 * a2))
    best = max((f0 + a1 * t + a2 * t * t + a3 * t ** 3, t) for t in ts if 0.0 <= t <= 1.0)
    return x0 + best[1] * h, best[0]


def _follow(d, dz, job):
    """Bracket and refine on the lattice (module doc).  Returns
    (w_row, gain at w_row, w_top estimate, deficit, level, evaluations)."""
    w0, rad, hint, lvl = float(job['w']), float(job['radius']), job.get('hint'), int(job.get('lvl') or FOLLOW_L0)
    lvl = min(max(lvl, 1), FOLLOW_LMAX)
    lo, hi = w0 - rad, w0 + rad
    if job.get('real', True):
        lo = max(lo, 0.0)
    ev = {}

    def E(w):
        w = float(w)
        if w not in ev:
            ev[w] = _gain(d, dz, w)
        return ev[w]

    h = 2.0 ** -lvl
    a = snap(float(hint) if (hint is not None and lo <= float(hint) <= hi) else w0, lvl)
    if a < lo:
        a += h * np.ceil((lo - a) / h - 1e-12)
    if a > hi:
        a -= h * np.ceil((a - hi) / h - 1e-12)
    ea = E(a)
    gtol = 1e-12 * max(ea['sigma'], 1.0)

    def sgn(e):
        g = e['gprime']
        return 0 if abs(g) <= gtol else (1 if g > 0 else -1)
    s_a = sgn(ea)
    if s_a == 0:
        # a flat point (w = 0 by symmetry, or an exact top): a top if both
        # neighbours are lower
        nb = [x for x in (a - h, a + h) if lo <= x <= hi]
        up = [x for x in nb if E(x)['sigma'] > ea['sigma']]
        if not up:
            return a, ea, a, 0.0, lvl, len(ev)
        a2 = max(up, key=lambda x: E(x)['sigma'])
        a, ea = a2, E(a2); s_a = sgn(ea)
        if s_a == 0:
            return a, ea, a, 0.0, lvl, len(ev)
    # walk along the slope to a sign change
    steps, bracket = 0, None
    while steps < FOLLOW_STEPS:
        b = a + s_a * h
        if b < lo - 1e-15 or b > hi + 1e-15:
            break                                          # the top lies beyond the member's window
        eb = E(b); s_b = sgn(eb)
        if s_b != s_a:
            bracket = (a, b) if a < b else (b, a)
            break
        a, ea = b, eb; steps += 1
    if bracket is None:
        return a, ea, a, float('nan'), lvl, len(ev)       # edge of the window: the row at the last point
    x0, x1 = bracket
    while True:
        e0, e1 = E(x0), E(x1)
        wt, ft = _hermite_top(x0, x1, e0['sigma'], e1['sigma'], e0['gprime'], e1['gprime'])
        fb = max(e0['sigma'], e1['sigma'])
        deficit = max(ft - fb, 0.0)
        if deficit <= FOLLOW_TOL * max(ft, 1e-300) or (x1 - x0) <= 2.0 ** -FOLLOW_LMAX * 1.0000001:
            break
        m = 0.5 * (x0 + x1); em = E(m); sm = sgn(em)
        if sm == 0:
            x0 = x1 = m
            wt, ft, deficit = m, em['sigma'], 0.0
            break
        if sm > 0:
            x0 = m
        else:
            x1 = m
    wrow = max((x0, x1), key=lambda x: E(x)['sigma'])
    lvl_out = level_of(x1 - x0) if x1 > x0 else lvl
    if deficit < FOLLOW_TOL * max(ft, 1e-300) / 16.0 and lvl_out > FOLLOW_L0:
        lvl_out -= 1                                       # well inside the tolerance: try a coarser bracket next time
    return wrow, E(wrow), float(wt), float(deficit), int(lvl_out), len(ev)


def _resident():
    return list(_CACHE.keys())


def rows_task(t):
    """The rows of one rank: [(i, kind, ...)] -> results (module doc)."""
    dz, jobs = t
    t0 = time.perf_counter()
    d = _design(dz)
    out = []
    for job in jobs:
        kind = job['kind']
        if kind == 'fix':
            w = float(job['w'])
            r = _gain(d, dz, w)
            res = dict(i=job['i'], w=w, sigma=r['sigma'], g=_grads(d, dz, r['x'], r['z']), nev=1)
        elif kind == 'follow':
            wrow, r, wt, df, lv, nev = _follow(d, dz, job)
            res = dict(i=job['i'], w=wrow, sigma=r['sigma'], g=_grads(d, dz, r['x'], r['z']),
                       wtop=wt, deficit=df, lvl=lv, nev=nev)
        elif kind == 'value':
            w = float(job['w'])
            res = dict(i=job['i'], w=w, sigma=_gain(d, dz, w, vectors=False)['sigma'], nev=1)
        elif kind == 'value2':
            w = float(job['w']); _note(complex(w))
            P = d['cl'].gain_k(w, dz['arow'], dz['nvec'], dz['K'], k=2)
            res = dict(i=job['i'], w=w, sigma=P[0]['sigma'], sigma2=P[1]['sigma'], nev=1)
        else:
            raise ValueError(kind)
        if job.get('block'):
            # the sigma_2 (Clarke 2x2) block at the row's point: the two leading
            # pairs and the raw gradients of the entries the SQP reads
            _note(complex(res['w']))
            P = d['cl'].gain_k(float(res['w']), dz['arow'], dz['nvec'], dz['K'], k=2)
            res['block'] = dict(sig=[P[0]['sigma'], P[1]['sigma']],
                                g00=_grads(d, dz, P[0]['x'], P[0]['z']), g11=_grads(d, dz, P[1]['x'], P[1]['z']),
                                g01=_grads(d, dz, P[1]['x'], P[0]['z']), g01i=_grads(d, dz, P[1]['x'], 1j * P[0]['z']),
                                g10=_grads(d, dz, P[0]['x'], P[1]['z']), g10i=_grads(d, dz, P[0]['x'], 1j * P[1]['z']))
            res['sigma'] = P[0]['sigma']; res['g'] = res['block']['g00']
        out.append(res)
    return dict(out=out, resident=_resident(), dt=time.perf_counter() - t0, hit=STATS['hit'], miss=STATS['miss'])


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
    try:
        th, V = spla.eigs(OP, k=kk, which='LM', tol=LINE_TOL, maxiter=300, v0=_v0(N), ncv=min(N - 1, max(2 * kk + 2, 24)))
    except spla.ArpackNoConvergence as e:
        if e.eigenvalues is not None and len(e.eigenvalues):
            th, V = e.eigenvalues, e.eigenvectors
    if th is None or len(th) == 0:
        return [], []
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
    return [complex(lam[i]) for i in ok], [V[:, i] for i in ok]


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


def line_task(t):
    """Disk centres on Re = -eps: for each theta, the verified nearest
    eigenvalues, the free radius, and the stability rows of those within
    ROW_REACH of the line (left eigenvectors on the same factor)."""
    dz, eps, thetas, reach = t
    t0 = time.perf_counter()
    d = _design(dz)
    cl = d['cl']
    out = []
    for th in thetas:
        s = complex(-eps, float(th))
        _note(complex(-1j * s))
        lams, vs = _nearest(cl, s, dz, LINE_K)
        rad = float(min(abs(l_ - s) for l_ in lams)) if lams else 0.0
        rows = []
        near = [j for j, l_ in enumerate(lams) if l_.real >= -eps - reach]
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
                _VEC[(dz['key'], float(th), round(lams[j].real, 10), round(lams[j].imag, 10))] = (v, u, complex(eta))
        out.append(dict(theta=float(th), lams=lams, rad=rad, rows=rows))
    return dict(out=out, resident=_resident(), dt=time.perf_counter() - t0, hit=STATS['hit'], miss=STATS['miss'])


def vec_task(t):
    """The (v, u, eta) a line task of this design kept, by (theta, lambda)."""
    key, items = t
    return dict(out=[_VEC.get((key, float(th), round(l_.real, 10), round(l_.imag, 10))) for th, l_ in items],
                resident=_resident(), dt=0.0, hit=STATS['hit'], miss=STATS['miss'])


def stats_task(t):
    return dict(out=[], resident=_resident(), dt=0.0, hit=STATS['hit'], miss=STATS['miss'])


def drop_task(t):
    """Forget everything (a new plant order, a new run phase)."""
    _CACHE.clear(); _DES.clear(); _VEC.clear(); _V0.clear()
    return dict(out=[], resident=[], dt=0.0, hit=STATS['hit'], miss=STATS['miss'])


# --------------------------------------------------------------------- driver
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
        self.tops = {}                   # followed rows of the last evaluation: lattice point -> bracket level
        self.line_prev = {}              # eps -> centres that covered the line at the last call
        self.calls = {}                  # kind -> [calls, tasks, wall, critical path (longest rank), misses]
        self.verbose = False
        self._dkey = None; self._dz = None
        self._line_memo = {}             # (design key, eps, wlo, whi) -> (reach, result): the last few designs

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

    def _assign(self, items):
        """items: [(point key, payload)] -> {rank: [payload]}.  A point goes to a
        rank holding its factor unless that rank's batch would then cost more than
        a factorisation on the least loaded one (GAIN_COST, LU_COST)."""
        ranks = self._ranks()
        load = {r: 0.0 for r in ranks}
        out = {r: [] for r in ranks}
        where = {}
        for r, ks in self.held.items():
            if r in load:
                for k in ks:
                    where.setdefault(k, []).append(r)
        known = [(k, p) for k, p in items if k in where]
        unknown = [(k, p) for k, p in items if k not in where]
        for k, p in known:
            hold = where[k]
            rmin = min(load, key=load.get)
            r = min(hold, key=load.get) if hold else rmin
            if not hold or load[r] + GAIN_COST > load[rmin] + LU_COST:
                r, c = rmin, LU_COST
            else:
                c = GAIN_COST
            out[r].append(p); load[r] += c
        for k, p in unknown:
            r = min(load, key=load.get)
            out[r].append(p); load[r] += LU_COST
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

    def report(self):
        """One line for the ledger: calls, tasks, wall and critical path by kind,
        and the factors held."""
        held = sum(len(v) for v in self.held.values())
        parts = [f"{k} {c[0]} calls / {c[1]} tasks, {c[2]:.0f} s wall (longest rank {c[3]:.0f} s)"
                 for k, c in sorted(self.calls.items())]
        return f"lattice ({self.size} ranks): " + "; ".join(parts) + f"; {held} factors held"

    def counts(self):
        """(factorisations, cached hits) over all ranks so far."""
        res = self._run(stats_task, {r: None for r in self._ranks()}, 'stats')
        return sum(o['miss'] for o in res.values()), sum(o['hit'] for o in res.values())

    # -------------------------------------------------------------------- rows
    def rows(self, z, W, follow=None, w_star=None, state=None, enroll=(), radius=0.1):
        """sigma_1 rows at every member of W.  follow: {member key: radius} (the
        followed peaks), w_star / state: per member, the last row point and its
        bracket level (updated here).  Returns (s, Gs, blocks, tops, info)."""
        dz = self.design(z)
        follow = follow or {}
        w_star = {} if w_star is None else w_star
        state = {} if state is None else state
        enroll = set(int(i) for i in (enroll or ()))
        items = []
        for i, w in enumerate(W):
            k6 = round(float(w), 6)
            rad = follow.get(k6)
            if rad:
                st = state.setdefault(k6, {})
                hint = w_star.get(k6)
                if hint is None and self.tops:
                    # a member placed at a top of an earlier evaluation (the
                    # round's end moves members there): start on that point
                    q = min(self.tops, key=lambda x: abs(x - float(w)))
                    if abs(q - float(w)) <= 0.5 * float(rad):
                        hint = q; st.setdefault('lvl', self.tops[q])
                start = snap(hint if hint is not None else float(w), int(st.get('lvl', FOLLOW_L0)))
                job = dict(kind='follow', i=i, w=float(w), radius=float(rad), hint=hint, lvl=st.get('lvl', FOLLOW_L0),
                           real=self.real, block=(i in enroll))
                items.append((complex(start), job))
            else:
                wl = snap(float(w), ROW_LEVEL)
                items.append((complex(wl), dict(kind='fix', i=i, w=wl, block=(i in enroll))))
        per = self._assign(items)
        res = self._run(rows_task, {r: (dz, jobs) for r, jobs in per.items()}, 'rows', ntask=len(items))
        m = len(W)
        if follow:
            self.tops = {}
        NZ = len(self.pk.z0())
        s = np.zeros(m); Gs = np.zeros((m, NZ)); blocks = {}; tops = {}; info = dict(nev=0, deficit=0.0)
        grad = lambda g_: np.asarray(self.pk.grad(g_), float)
        for o in res.values():
            for r in o['out']:
                i = r['i']; s[i] = r['sigma']; Gs[i] = grad(r['g']); info['nev'] += r['nev']
                k6 = round(float(W[i]), 6)
                if 'wtop' in r:
                    w_star[k6] = float(r['w']); tops[k6] = float(r['w'])
                    state.setdefault(k6, {})['lvl'] = r['lvl']
                    self.tops[float(r['w'])] = int(r['lvl'])
                    if np.isfinite(r['deficit']):
                        info['deficit'] = max(info['deficit'], r['deficit'] / max(r['sigma'], 1e-300))
                if 'block' in r:
                    B = r['block']
                    Gc = np.zeros((2, 2, NZ), complex)
                    Gc[0, 0] = grad(B['g00']); Gc[1, 1] = grad(B['g11'])
                    Gc[0, 1] = grad(B['g01']) + 1j * grad(B['g01i'])
                    Gc[1, 0] = grad(B['g10']) + 1j * grad(B['g10i'])
                    blocks[i] = dict(sig=np.array(B['sig']), Gc=Gc)
        return s, Gs, blocks, tops, info

    def sigmas(self, z, ws, k=1):
        """sigma_1 values (no vectors) at the lattice points ws; k = 2: the
        pairs (sigma_1, sigma_2) as an (len(ws), 2) array."""
        dz = self.design(z)
        items = [(complex(float(w)), dict(kind=('value' if k == 1 else 'value2'), i=i, w=float(w))) for i, w in enumerate(ws)]
        res = self._run(rows_task, {r: (dz, jobs) for r, jobs in self._assign(items).items()}, 'sweep', ntask=len(items))
        out = np.zeros(len(ws)) if k == 1 else np.zeros((len(ws), 2))
        for o in res.values():
            for r in o['out']:
                if k == 1:
                    out[r['i']] = r['sigma']
                else:
                    out[r['i']] = (r['sigma'], r['sigma2'])
        return out

    def grid(self, wlo, whi, level=None):
        """The lattice points of `level` in [wlo, whi]."""
        h = 2.0 ** -int(SWEEP_LEVEL if level is None else level)
        k0, k1 = int(np.ceil(wlo / h - 1e-12)), int(np.floor(whi / h + 1e-12))
        return np.arange(k0, k1 + 1) * h

    # -------------------------------------------------------------------- line
    def line(self, z, eps, wlo, whi, reach=None, eps_int=None):
        """The stability line Re = -eps, Im in [wlo, whi], in waves (module
        doc).  z: the packed design, or a design dict (design_raw).  reach =
        -inf: no stability rows (the gates).  Returns dict(clear, alpha,
        witness, rows, seen, nshift, waves, uncovered)."""
        dz = z if isinstance(z, dict) else self.design(z)
        reach = ROW_REACH if reach is None else float(reach)
        mk = (dz['key'], round(float(eps), 12), float(wlo), float(whi))
        hit = self._line_memo.get(mk)
        if hit is not None and hit[0] >= reach:
            return hit[1]                                  # this design's line, swept already (same or more rows)
        out = self._line(dz, eps, wlo, whi, reach, eps_int)
        self._line_memo[mk] = (reach, out)
        while len(self._line_memo) > 8:
            self._line_memo.pop(next(iter(self._line_memo)))
        return out

    def _line(self, dz, eps, wlo, whi, reach, eps_int):
        hmin = 2.0 ** -LINE_LMAX if eps_int is None else float(eps_int)
        key = round(float(eps), 12)
        cent = sorted(c for c in self.line_prev.get(key, []) if wlo - 1e-12 <= c <= whi + 1e-12)
        if not cent:
            cent = list(self.grid(wlo, whi, LINE_L0))
            if not cent or cent[0] > wlo:
                cent = [snap(wlo, LINE_LMAX)] + cent
        done = {}
        waves = 0; witness = None; nshift = 0
        while cent and waves < LINE_WAVES:
            waves += 1
            items = [(complex(float(th), float(eps)), float(th)) for th in cent]
            per = self._assign(items)
            res = self._run(line_task, {r: (dz, float(eps), ths, reach) for r, ths in per.items()}, 'line',
                            ntask=len(items))
            for r_, o in res.items():
                for c in o['out']:
                    c['rank'] = r_
                    done[c['theta']] = c
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
            # next call's first wave: a cover of the line, plus every centre whose
            # nearest eigenvalue lies within ROW_REACH of it -- an eigenvalue near
            # the line need not be the nearest to any centre of a cover (the
            # plant's -0.1148 was not, at the zero start), and its row must not
            # disappear from one evaluation to the next
            keep = {th for th, c in done.items() if any(l_.real >= -eps - ROW_REACH for l_ in c['lams'])}
            self.line_prev[key] = sorted(set(self._cover(done, wlo, whi)) | keep)
        seen, rows = [], []
        for th in sorted(done):
            c = done[th]
            for l_ in c['lams']:
                if all(abs(l_ - q) > 1e-8 * max(1.0, abs(l_)) for q in seen):
                    seen.append(l_)
            for rw in c['rows']:
                if all(abs(rw['lam'] - q['lam']) > 1e-8 * max(1.0, abs(rw['lam'])) for q in rows):
                    rows.append(dict(rw, theta=th, rank=c['rank']))
        rows.sort(key=lambda r_: -r_['lam'].real)
        alpha = max((l_.real for l_ in seen), default=-np.inf)
        if witness is not None:
            alpha = max(alpha, witness.real)
        return dict(clear=bool(clear), alpha=float(alpha), witness=witness, rows=rows, seen=seen,
                    nshift=nshift, waves=waves, uncovered=uncovered)

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
            per.setdefault(rw['rank'], []).append((j, rw['theta'], rw['lam']))
        res = self._run(vec_task, {r: (dz['key'], [(th, l_) for _, th, l_ in v]) for r, v in per.items()}, 'vectors',
                        ntask=len(rows))
        out = [None] * len(rows)
        for r, v in per.items():
            for (j, _, _), vec in zip(v, res[r]['out']):
                out[j] = vec
        return out

    def drop(self):
        self._run(drop_task, {r: None for r in self._ranks()}, 'drop')
        self.held.clear(); self.line_prev.clear(); self.tops.clear(); self._line_memo.clear()
