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

What runs per SQP evaluation -- rows and line TOGETHER, in waves
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
            stability_and_abscissa) in WAVES: the base grid (LINE_BASE = 48
            equispaced centres, ranks/2 before 20261010j stage C) and every
            centre that covered the line at the previous call (cached, in
            parallel), then new centres in the remaining gaps, on
            lattice points, kept for the next calls -- until the line is covered,
            an eigenvalue is found on or right of it (not clear), or a gap is
            shorter than the finest level (inconclusive, never clear).  The proof
            is the cover; where the centres sit changes the cost, not the rigour.
  strip     (a DIAGNOSTIC since 2026-10-16: evaluate(strip=True), tools/
            check_lattice.py and the Evaluator's sqp_inner.LAT_STRIP; no
            optimiser path covers it -- the strip arm of 2026-10-09 to 20261010n
            was removed, BFS.md 8) the strip [-eps - ROW_REACH, -eps], covered by
            a quadtree of cells and the disks of every shift of the evaluation
            (LatticeOracle._strip): every eigenvalue in it is exhibited and gets
            its stability row (left eigenvector on the same factor), exact.  The
            line alone sees only the eigenvalues nearest to some point of it.
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
FOLLOW_EXTEND = 2    # a Newton stopped at its window's edge, the slope still pointing out: the window re-centred on the
                     # point, at most this many times per evaluation (20261010j; 770632 round 11: the member based at the
                     # 2^-5 grid point 4.03125 stopped exactly on its edge 4.13125, sigma 143.48, slope +284, while its
                     # top 4.180605 stood at 151.4775 -- the round's SQP never saw it; missed-peaks.md P3).  Every
                     # window stays within the base +- (1 + FOLLOW_EXTEND) x radius, from a hint too (the review of
                     # stage A: unclipped, the row alternated between edge and top at one design; _newton)
XCAP = 2             # factors of the exact (off-lattice) points kept per rank, apart from the lattice cache
LINE_BASE = 48       # the line's and the level set's base grid: this many equispaced centres WHATEVER THE NUMBER OF
                     # RANKS (--lattice-base; 20261010j stage C, eval-waves.md P1 / packing.md P2).  It was ranks/2 (user,
                     # 2026-10-09: 'the other half of the ranks fills the gaps in the same time'): 48 at 96 ranks, so the
                     # 96-rank runs are unchanged, but 24 at 48 ranks -- other line covers, other no-strip rows, other
                     # paths than the 96-rank parents, and the next batch runs at 48 ranks, two per node (user,
                     # 2026-10-14).  Cross-rank checks were only ever made at 1-3 ranks, where both rules give 4.
                     # 0: ranks/2 again
LINE_BASE_MIN = 4    # ... at least this many
LINE_LMAX = 17       # finest centre spacing 2^-17 (7.6e-6, the serial certificate's eps_int 1e-5)
LINE_K = 1           # nearest eigenvalues per centre (k = 3 costs 2-4 x the applications; 2026-10-06, test plant)
LINE_SHRINK = 0.9    # disk used = LINE_SHRINK x the distance to the nearest eigenvalue (serial code: 0.9)
LINE_WAVES = 60      # waves before the sweep is declared inconclusive
LINE_TOL = 1e-9      # ARPACK tolerance at a centre (the serial certificate's)
ROW_REACH = 0.25     # the reach of the line's rows (its memo) and the diagnostic STRIP [-eps - ROW_REACH, -eps]
STRIP_RHO = 2.0 ** -6   # the cover may leave out a ball of this radius around each eigenvalue it exhibits
STRIP_HMIN = 2.0 ** -10 # a cell this small and still uncovered: the strip is reported incomplete
STRIP_SPLIT = 2      # geometric refinements of an uncovered cell per wave before new shifts are placed
STRIP_WAVES = 30     # waves before the strip is reported incomplete
WARM_NCV = 0         # > 0: sigma_1 restarted from the last vector at the same point with this subspace (4: 4-5
                     # applications instead of 9-17).  Off by default: where sigma_1 and sigma_2 swap between designs
                     # (kinks; the MIMO plateau) the restart vector lies along the new sigma_2 and a small subspace can
                     # converge there; cold, a row does not depend on its rank.  The line's disks are always cold.
SWEEP_LEVEL = 5      # the certificate's reference sweep on 2^-5
LEVEL_K = 2          # level-set disks: nearest eigenvalues of the even pencil per axis centre (a mirrored pair;
                     # k = 4 cost 2-10 x the applications: 35-357 against 21-39, central record 191.11)
LEVEL_ARNOLDI = ((24, 2), (100, 3))   # ... and their Arnoldi: (Krylov subspace, restarts) tried in turn.  Near a peak the
                     # small subspace converges at once (21-39 applications, 5-8 s; central record 191.11); midway
                     # between peaks and the far field the nearest eigenvalues are many at similar distances: ncv 24
                     # failed in 8 restarts where ncv 100 converged in ~100 applications (zero start, w = 1.5: 23 s,
                     # radius 0.49 against the norm disk's 0.0022); ncv 12-40 needed 350-850 at w = 6-10
LEVEL_PAIR = True    # (20261010s) at a level-set centre whose last disk needed the second Arnoldi tier (or neither
                     # converged), the two tiers at once on two ranks ('levelp' requests): the first tier's disk where
                     # it converged, else the second's -- the disk the tiers in turn give, bit for bit (each tier starts
                     # from the same vector, _v0, so the second does not depend on the first; the applications and CG
                     # counts are the two tiers' sums, as in turn).  A hard disk then takes the second tier's time instead
                     # of both (the first tier's failed restarts are 21-42 % of it); a centre needs two factors
LEVEL_NORM_MIN = 2.0 ** -5   # a level-set centre whose norm disk is at least this needs no pencil eigenvalues
LEVEL_CG_TOL = 1e-12 # the inner CG on the plant factor (_level_disk): relative residual in the M semi-norm
LEVEL_CG_MAX = 200   # ... iterations at most
LEVEL_IMAG = 1e-8    # a pencil eigenvalue with |Re| <= this x max(1, |lambda|) is ON the axis: a level crossing
LEVEL_CONFIRM = 1e-6 # a crossing is confirmed where sigma_1 >= gam (1 - this) (certify_greater's sampling)
LEVEL_WATCH_PAD = 1e-3  # a region a followed row watches is left out of the cover with this margin on each side (the
                     # members' own spacing: a region within it of a watched one is that region; 20261010j, P2)
TASK_COST = dict(row=1.0, value=1.0, slope=1.0, value2=2.0, block=2.0, centre=3.0,
                 rowx=1.0, slopex=1.0, blockx=2.0, level=10.0, levelp=10.0)   # relative costs of the requests
LU_COST = 2.0        # ... and of a factorisation (balancing, LatticeOracle._assign)
TASK_COSTS = 'static'    # 'static' (the default; --lattice-costs): the units above in request order, as since
                     # 2026-10-08.  'measured' (20261010j stage C, eval-waves.md P7; opt-in): _assign balances on the
                     # MUMPS SOLVES each kind of request took (running means, the factorisation apart; a level-set disk
                     # at a centre of an earlier call: that disk's own count), a factorisation LU_SOLVES solves, longest
                     # first.  The static units were never measured: a cluster LU (0.46-0.49 s) costs about one row, not
                     # two, and a level disk 1-2 s (norm) or 5-210 s (pencil), not 10 rows -- two slow disks could share
                     # a rank while another had none (at 48 ranks every no-strip level wave holds more disks, 62-111,
                     # than ranks; packing.md P4).  COUNTS, NOT SECONDS (the review of stage C): placement by the
                     # perf_counter seconds of earlier requests differed from one submission to the next, and a task's
                     # result does depend on its rank at rounding level (driver vs worker rows 7e-13, BFS.md 8; any
                     # rounding change moves a MIMO path from round 0, LESSONS 18 rules 5-6, STAGE6 6.8): two identical
                     # jobs would have followed different paths.  The solves of a request are a function of its inputs
                     # and of the rank it ran on, which earlier placements fixed: the same placement every run.  The
                     # seconds stay in the ledger and its projection.  Not the default: the gain is <= ~5 % of the eval
                     # wall, unmeasured (P7 verifier), it moves work between ranks (a held point leaves a loaded holder
                     # sooner: more duplicate factors), and rule 3 keeps defaults frozen
LU_SOLVES = 30.0     # 'measured': a factorisation in solves (test plant: 0.5 s = 32 solves of 0.016 s in the sandbox,
                     # BFS.md 8 2026-10-09; cluster 0.46-0.49 s against 17-21 ms, 22-29).  Fixed, not measured: a
                     # running mean of seconds would bring the clock back into the placement
COST_UNIT = 26.0     # 'measured': solves per static unit before a kind has been measured (a row: 18-34 solves, BFS.md 8)
COST_EMA = 0.02      # the running means' weight once a kind has more than 1 / COST_EMA samples (drift: the level
                     # set's disks got 6x dearer over 770632's 12 rounds)
REQ_MEMO = 2         # designs whose row requests (row, block, rowx, slopex, blockx) evaluate() keeps for the next
                     # evaluation of the SAME design (--lattice-memo; 20261010j stage C, eval-waves.md P8): the refine
                     # re-evaluates every iterate it adds members at (and the strip arm, until 2026-10-16, every
                     # accepted trial with its strip), recomputing every fixed row and every first Newton point.  The
                     # line's centres are memoised apart (_line_memo), the strip's never (it reads every centre of the
                     # evaluation, R).
                     # 0: off
ROW_MERGE = 1e-8     # stability rows whose eigenvalues agree within this x max(1, |lambda|) are ONE row (--lattice-row-
                     # merge).  <= 1e-8 (the default): the rule before, 1e-8 and the first copy in centre order.  > 1e-8
                     # (20261010j stage C, strip-ab.md AB3; opt-in): the copy from the nearest shift kept -- one
                     # eigenvalue seen from several shifts comes back 2e-8 .. 1.5e-7 apart (the probe at 151.48: 30
                     # strip rows, 19 eigenvalues, up to 6 copies), each copy with its own elastic slack and a share of
                     # the multiplier the penalty update reads; 1e-6 is safe (the strip exhibits one eigenvalue per
                     # ball of STRIP_RHO = 2^-6 anyway), and 'seen' (alpha, admissibility) keeps 1e-8 whatever this is.
                     # Off by default (the review of stage C): the replay at the 151.48 iterate (impl/C D1/D2) changed
                     # nothing beyond rounding (smax 145.4755638 / 637, nu and dt equal) and no stability row was active
                     # in any of its QPs, so it could show no effect; yet it changes the iterates, and a rounding-level
                     # change moves a MIMO path from round 0 (LESSONS 17.4 item 23, 20.6 item 3: measure on replayed QPs
                     # first, both arms only if it changes something; the AB3 verifier).  The copies are counted and
                     # tagged instead (ROW_DUP), so the batch shows whether a copy is ever active or given up in a QP
ROW_DUP = 1e-6       # ... with the merge off: rows within this of a row from a nearer shift are COUNTED as copies and
                     # tagged ('dup'; the QP stability rows' line reports them active / given up), not merged
MEMBER_BINS = (0.5, 0.8, 0.95, 0.99)    # the per-member counts by s_i / smax (omega-growth.md P1)
PROJ_RANKS = (48, 32)    # the ledger's projection: every wave's measured requests replayed longest first on this many
                     # ranks (packing.md P3; no factor moves with them, so the misses of a smaller cache are not in it)

_CACHE = {}          # complex shift key (closedloop's: complex(omega)) -> factor, LRU order
_DES = {}            # design key -> dict(b, c, pa, ps, na, ns, cl)
_VEC = {}            # (design key, theta, rounded lambda) -> (v, u) of the last line tasks
_V0 = {}             # point key -> the last right singular vector of sigma_1 there (warm starts)
_XCACHE = {}         # the exact points' factors (XCAP, FIFO): kept out of the lattice cache
_ASIDE = {}          # the asynchronous level set's design where _DES lacks it (20261010j stage D): built aside
_VSNAP = []          # [(_DES, _VEC)] as at a look-ahead's start, put back when it is discarded (vecstate_task)
STATS = dict(hit=0, miss=0, xhit=0, xmiss=0)       # this process's cache accounting: lattice points (_note) and,
                     # 20261010j stage C, the exact points' side cache (_exact; never counted before: the Newton points
                     # are most of the factorisations, driver-serial.md)
_ML = []             # bfs.mumps_lu, once imported (the factorisations and solves of each request, eval_task)
_MEMO_KINDS = ('row', 'block', 'rowx', 'slopex', 'blockx')    # the requests evaluate() keeps per design (REQ_MEMO)


def _mumps_mod():
    """bfs.mumps_lu, or None where it cannot be imported."""
    if not _ML:
        try:
            from bfs import mumps_lu
            _ML.append(mumps_lu)
        except Exception:
            _ML.append(None)
    return _ML[0]


def _mumps():
    """(factorisations, their seconds, solves, their seconds) of this process so far (bfs/mumps_lu.STATS,
    every context; zeros where MUMPS is not the factoriser)."""
    m = _mumps_mod()
    if m is None:
        return 0, 0.0, 0, 0.0
    S = m.STATS
    nf = sf = ns = ss = 0
    for k, v in S.items():
        if k.startswith('factor'):
            nf += v[0]; sf += v[1]
        elif k.startswith('solve'):
            ns += v[0]; ss += v[1]
    return nf, sf, ns, ss


def snap(w, level):
    """The lattice point of level `level` nearest w (an exact dyadic float)."""
    h = 2.0 ** -int(level)
    return float(np.round(float(w) / h) * h)


def _lpt(ts, n):
    """The makespan of the task times ts on n ranks, longest first (each to the
    least loaded rank): the ledger's projection of a wave (packing.md P3)."""
    if not ts:
        return 0.0
    if len(ts) <= n:
        return float(max(ts))
    import heapq
    h = [0.0] * int(n)
    for t_ in sorted(ts, reverse=True):
        heapq.heapreplace(h, h[0] + t_)
    return float(max(h))


# --------------------------------------------------------------------- worker
def _design(dz):
    """The design's ports and closed loop, built once per design and process
    (the last four kept); its closed loop uses this module's cache.
    dz['aside'] (20261010j stage D, the asynchronous level set): a design _DES
    lacks is built in its own slot (_ASIDE), so that the level set's disks,
    which run between the next iteration's requests, never evict a design nor
    purge a vector the synchronous order would have kept (level-set-waves.md
    LS3, driver-serial.md P3 verifier (a): a build purges _VEC of every other
    design, and the trial's Hermite block would silently go)."""
    import fullpar
    d = _DES.get(dz['key'])
    if d is not None:
        return d
    if dz.get('aside'):
        d = _ASIDE.get(dz['key'])
        if d is not None:
            return d
    D = fullpar.CTX['D']
    b, c, pa, ps, na, ns = D.bc(dz['xa'], dz['xs'])
    cl = D.cl(b, c)
    cl.lu_cache, cl.lu_cache_max, cl.lu_lru, cl.lu_pin = _CACHE, int(LU_CAP), True, None
    d = dict(b=b, c=c, pa=pa, ps=ps, na=na, ns=ns, cl=cl, D=D)
    if dz.get('aside'):
        _ASIDE.clear(); _ASIDE[dz['key']] = d
        return d
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


def _exact(d, fn, w=None):
    """fn() with the closed loop's factor cache switched to the exact points'
    own (a Newton point is new at every evaluation: in the lattice cache it would
    only push the lattice's factors out).  w: the point (its side-cache hit or
    miss counted)."""
    cl = d['cl']
    if w is not None:
        STATS['xhit' if complex(w) in _XCACHE else 'xmiss'] += 1
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


def _ports(cl, dz):
    """(B, C, A_c, B_c, C_c) of the closed loop as 2-D arrays (SISO and MIMO):
    A_cl = blockdiag(A, A_c) + U V, U = [[B, 0], [0, B_c]], V = [[D_c C^H, C_c],
    [C^H, 0]] (certify_sparse.SparsePencil)."""
    B = cl.b if np.ndim(cl.b) == 2 else cl.b[:, None]
    C = cl.c if np.ndim(cl.c) == 2 else cl.c[:, None]
    m, p, r = B.shape[1], C.shape[1], cl.r
    Bc = cl.Bc_of(dz['arow']) if hasattr(cl, 'Bc_of') else cl.B_of()[:, None]
    return B, C, cl.L_of_a(dz['arow']), np.asarray(Bc, complex).reshape(r, p), np.asarray(dz['nvec'], complex).reshape(m, r)


def _norm_disk(d, dz, th, gam):
    """(sigma_1(th), a radius rho such that ||G(lam)|| < gam on the disk |lam - i th|
    < rho) from norms at i th alone, the controller handled exactly.  Then the
    even pencil has no eigenvalue in the disk (it is singular at lam only where
    ||G(lam)|| ||G(-conj lam)|| >= gam^2), so sigma_1 < gam on its diameter.

    G(lam) = R S(lam)^-1 R^T with S(lam) = lam M - A - B K(lam) C^H, K(lam) = D_c
    + C_c (lam - A_c)^-1 B_c (the controller states eliminated; R^T R = M, never
    formed).  At lam = i th + delta, S = S_0 + Q Delta P with P = [R; a C^H], Q =
    [R^T, b B], Delta = diag(delta I, -(K(lam) - K(i th)) / (a b)), so by the push-
    through identity G(lam) is a block of Y (I + Delta Y)^-1, Y = P S_0^-1 Q, and
    ||G(lam)|| <= ||Y|| / (1 - ||Delta|| ||Y||).  ||Y|| <= the 2-norm of the 2 x 2
    matrix of its blocks' norms: sigma_1, ||R S_0^-1 B||, ||C^H S_0^-1 R^T||,
    ||C^H S_0^-1 B|| (m + p solves, the rest dense m x m and p x p);  ||K(lam) -
    K(i th)|| <= |delta| ||C_c R_0|| ||R_0 B_c|| / (1 - |delta| ||R_0||), R_0 = (i th -
    A_c)^-1 (r x r).  The weights a, b (powers of 2) maximise the radius.  A
    controller resonance near th (||R_0|| large) or sigma_1 near gam make it
    small; elsewhere it is the cheap disk -- far from the peaks the pencil's
    nearest eigenvalues are many at similar distances and its Arnoldi needs
    350-850 applications (90-220 s on the test plant, w = 6-10) where this costs
    one sigma_1 and m + p solves (1-2 s) for radii 0.07-0.22 (central record
    191.11; checked against ||G|| on the disks' boundaries, BFS.md 5)."""
    cl = d['cl']
    n, r = cl.n, cl.r
    s1 = _gain(d, dz, th, vectors=False)['sigma'] * (1.0 + 1e-6)       # Arnoldi approaches sigma_1 from below
    B, C, Ac, Bc, Cc = _ports(cl, dz)
    zr = np.zeros(r, complex)
    TB = np.column_stack([cl.solve(np.concatenate([B[:, j], zr]))[:n] for j in range(B.shape[1])])
    THC = np.column_stack([cl.solveH(np.concatenate([C[:, j], zr]))[:n] for j in range(C.shape[1])])
    M = cl.M
    n12 = float(np.sqrt(max(np.linalg.eigvalsh(TB.conj().T @ (M @ TB)).max(), 0.0)))
    n21 = float(np.sqrt(max(np.linalg.eigvalsh(THC.conj().T @ (M @ THC)).max(), 0.0)))
    n22 = float(np.linalg.norm(C.conj().T @ TB, 2))
    if r:
        R0 = np.linalg.inv(1j * float(th) * np.eye(r) - Ac)
        nR0 = float(np.linalg.norm(R0, 2)); c1 = float(np.linalg.norm(Cc @ R0, 2) * np.linalg.norm(R0 @ Bc, 2))
    else:
        nR0 = c1 = 0.0
    best = 0.0
    for la in range(-30, 31):
        for lb in range(-30, 31):
            wa, wb = 2.0 ** la, 2.0 ** lb
            ny = float(np.linalg.norm(np.array([[s1, wb * n12], [wa * n21, wa * wb * n22]]), 2))
            lim = 1.0 / ny - 1.0 / float(gam)            # needs ||Delta|| < lim
            if lim <= 0.0:
                continue
            # ||Delta|| <= max(rho, c1 rho / ((1 - rho ||R_0||) a b)) < lim  <=>  rho < both:
            rho = min(lim, lim / (c1 / (wa * wb) + lim * nR0)) * (1.0 - 1e-9)
            best = max(best, rho)
    return s1, best


def _level_disk(d, dz, th, gam, tiers=None):
    """One disk of the LEVEL-SET certificate at the axis point i th, at level
    gam: a radius around i th free of eigenvalues of the even pencil (so free of
    level crossings), the pencil eigenvalues found, and sigma_1(th).  First the
    norm disk (_norm_disk: cheap, large far from the peaks); where it is smaller
    than LEVEL_NORM_MIN, the pencil's nearest eigenvalues as well (Schwerdtner-
    Mengi-Voigt, certify_sparse.certify_greater), applied through the closed
    loop's plant factor of i th M - A -- the lattice's own, shared with the rows
    -- with no factor of the 2n pencil (it carries M / gam^2 in a full block: no
    factor serves two levels; user, 2026-10-09).

    The pencil (certify_sparse.SparsePencil) at the shift s = i th:
        A_H - s E_H = [[-S, g MP], [MP, -S^H]],   E_H = diag(M_cl, -M_cl),
    S = i th M_cl - A_cl (the closed loop's matrix, cl.solve = S^-1), MP =
    diag(M, 0), g = 1 / gam^2.  OP v = (A_H - s E_H)^-1 E_H v by block
    elimination: with f = M_cl v1, a = S^-1 f, c = S^-H [0; v2_c],
        (I - B) y = v2_p - a_p + g (S^-1 [M c_p; 0])_p,   B = g T M T^H M,
        x2 = S^-H [M y; 0] + c,   x1 = S^-1 (g MP x2 - f),
    T u = (S^-1 [u; 0])_p.  B is self-adjoint and positive semidefinite in the
    M semi-inner product, with eigenvalues sigma_i^2 / gam^2 < 1 (the closed
    loop's singular values at th): CG in that inner product converges in 5-11
    iterations even where sigma_1 is within 1e-4 of gam (restarted GMRES in
    the Euclidean product needed 350-730 there: central record 191.11, w = 0).
    Each iteration is one solve and one adjoint solve on the factor held;
    LEVEL_K = 2 nearest eigenvalues (they come in pairs mirrored across the
    axis, at equal distance from it); Arnoldi with a small subspace first, a
    large one if that does not converge in 2 restarts (LEVEL_ARNOLDI): 21-39
    applications, 5-8 s on the test plant near the peaks, ~100 (20-25 s)
    midway between them (BFS.md 5).  The disk is the larger of the two.
    tiers: the indices of LEVEL_ARNOLDI to try, in turn (None: all; a
    'levelp' request one, LEVEL_PAIR)."""
    import scipy.sparse.linalg as spla
    from closedloop_sparse import _v0
    s1, rn = _norm_disk(d, dz, th, gam)
    # the disk's cost log (20261010j stage C, level-set-waves.md LS1): Arnoldi
    # applications, the tier that converged (0 = the first of LEVEL_ARNOLDI, -1 =
    # none), the CG iterations -- the slow disks were inferred from aggregates only
    if s1 >= gam or rn >= LEVEL_NORM_MIN:
        return dict(theta=float(th), lams=[], rad=(0.0 if s1 >= gam else rn), sigma=s1, cg_max=0, pencil=False,
                    napp=0, tier=None, cg_sum=0, cg_n=0)
    cl = d['cl']
    n, r = cl.n, cl.r
    N = n + r
    M = cl.M
    g = 1.0 / float(gam) ** 2
    cl.setup(float(th), dz['arow'], dz['nvec'], dz['K'])
    zr, zn = np.zeros(r, complex), np.zeros(n, complex)
    S, SH = cl.solve, cl.solveH
    nit = []; napp = [0]

    def B(y):
        return g * S(np.concatenate([M @ SH(np.concatenate([M @ y, zr]))[:n], zr]))[:n]

    def cg(y0):
        y = np.zeros(n, complex); res = y0.copy(); rr = float(np.vdot(res, M @ res).real); rr0 = rr
        p = res.copy(); it = 0
        while rr > LEVEL_CG_TOL ** 2 * rr0 and it < LEVEL_CG_MAX:
            Ap = p - B(p)
            a = rr / float(np.vdot(p, M @ Ap).real)
            y += a * p; res -= a * Ap
            rn_ = float(np.vdot(res, M @ res).real)
            p = res + (rn_ / rr) * p; rr = rn_; it += 1
        nit.append(it)
        return y0 + B(y)                     # the components M does not see, from the converged M y

    def op(v):
        napp[0] += 1
        v1, v2 = v[:N], v[N:]
        f = np.concatenate([M @ v1[:n], v1[n:]])
        a = S(f)
        c = SH(np.concatenate([zn, v2[n:]]))
        y = cg(v2[:n] - a[:n] + g * S(np.concatenate([M @ c[:n], zr]))[:n])
        x2 = SH(np.concatenate([M @ y, zr])) + c
        x1 = S(g * np.concatenate([M @ x2[:n], zr]) - f)
        return np.concatenate([x1, x2])
    O = spla.LinearOperator((2 * N, 2 * N), matvec=op, dtype=complex)
    s = 1j * float(th)
    tier = -1
    for j_, (ncv, mxit) in enumerate(LEVEL_ARNOLDI):
        if tiers is not None and j_ not in tiers:
            continue
        conv = True
        try:
            mu = spla.eigs(O, k=LEVEL_K, which='LM', tol=LINE_TOL, maxiter=mxit, v0=_v0(2 * N), ncv=ncv,
                           return_eigenvectors=False)
        except spla.ArpackNoConvergence as e:
            conv = False
            mu = e.eigenvalues if e.eigenvalues is not None else np.zeros(0, complex)
        if conv:
            tier = j_
            break
    mu = np.asarray(mu)[np.abs(mu) > 1e-300]
    lam = s + 1.0 / mu
    # the pencil's disk only when the LEVEL_K nearest all converged (a partial
    # set need not hold the nearest); converged eigenvalues still count as
    # crossings
    rp = float(np.min(np.abs(lam - s))) if (conv and len(lam) == LEVEL_K) else 0.0
    return dict(theta=float(th), lams=[complex(l_) for l_ in lam], rad=max(rp, rn), sigma=s1,
                cg_max=max(nit, default=0), pencil=True, napp=int(napp[0]), tier=int(tier),
                cg_sum=int(sum(nit)), cg_n=len(nit))


def eval_task(t):
    """Every evaluation of one rank in one wave: [(request)] -> results.
    Requests (LatticeOracle._drive): ('row', w) sigma_1, its slope and raw
    gradient; ('block', w) the sigma_2 block (and sigma_1's row); ('value', w)
    sigma_1 alone; ('slope', w) sigma_1 and its slope; ('value2', w) sigma_1
    and sigma_2; ('rowx', w), ('slopex', w), ('blockx', w) the same at an exact
    (off-lattice) point, factor in the side cache; ('centre', sre, theta, rlo)
    a disk centred at sre + i theta, rows for its eigenvalues with Re >= rlo;
    ('level', theta, gam) a level-set disk at i theta (_level_disk);
    ('levelp', theta, gam, k) the same with the Arnoldi tier k alone (LEVEL_PAIR).
    Each request's own cost comes back in 'req' (20261010j stage C, eval-waves.md
    P0 / P7, driver-serial.md P1): (seconds, factorisations, their seconds,
    solves, their seconds) -- the ledger's per-kind work and critical path, the
    measured costs of _assign, each rank's time per solve; dt also counts the
    design's set-up."""
    dz, reqs = t
    t0 = time.perf_counter()
    d = _design(dz)
    out = []; rec = []
    for q in reqs:
        t1 = time.perf_counter(); m1 = _mumps()
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
            r = _exact(d, lambda: d['cl'].gain(w, dz['arow'], dz['nvec'], dz['K']), w)
            res = dict(sigma=r['sigma'], gprime=r['gprime'])
            if kind == 'rowx':
                res['g'] = _grads(d, dz, r['x'], r['z'])
        elif kind == 'blockx':
            res = _exact(d, lambda: _block(d, dz, q[1]), q[1])
        elif kind == 'value2':
            _note(complex(q[1]))
            P = d['cl'].gain_k(float(q[1]), dz['arow'], dz['nvec'], dz['K'], k=2)
            res = dict(sigma=P[0]['sigma'], sigma2=P[1]['sigma'])
        elif kind == 'centre':
            res = _centre(d, dz, q[1], q[2], q[3])
        elif kind == 'level':
            res = _level_disk(d, dz, q[1], q[2])
        elif kind == 'levelp':
            res = _level_disk(d, dz, q[1], q[2], tiers=(int(q[3]),))
        else:
            raise ValueError(kind)
        out.append(res)
        m2 = _mumps()
        rec.append((time.perf_counter() - t1, m2[0] - m1[0], m2[1] - m1[1], m2[2] - m1[2], m2[3] - m1[3]))
    return dict(out=out, resident=_resident(), dt=time.perf_counter() - t0, req=rec, **_counts())


def _counts():
    return dict(hit=STATS['hit'], miss=STATS['miss'], xhit=STATS['xhit'], xmiss=STATS['xmiss'])


def vec_task(t):
    """The (v, u, eta) a centre task of this design kept, by (sre, theta, lambda)."""
    key, items = t
    return dict(out=[_VEC.get((key, float(sr), float(th), round(l_.real, 10), round(l_.imag, 10))) for sr, th, l_ in items],
                resident=_resident(), dt=0.0, **_counts())


def stats_task(t):
    return dict(out=[], resident=_resident(), dt=0.0, **_counts())


def drop_task(t):
    """Forget everything (a new plant order, a new run phase)."""
    _CACHE.clear(); _DES.clear(); _VEC.clear(); _V0.clear(); _XCACHE.clear(); _ASIDE.clear(); _VSNAP.clear()
    return dict(out=[], resident=[], dt=0.0, **_counts())


def vecstate_task(op):
    """This process's designs and line vectors around a look-ahead (20261010j
    stage D): 'mark' keeps (_DES, _VEC) as they are at its start (references:
    nothing is copied but the two dicts), 'restore' puts them back when it is
    discarded -- the trials it evaluated built their designs and purged the
    iterate's vectors, which the synchronous continuation (the re-evaluation
    with the new members) reads through vectors(); the factor caches stay
    (a factor is the same whoever computed it, only the cost differs)."""
    if op == 'mark':
        _VSNAP[:] = [(dict(_DES), dict(_VEC))]
    elif op == 'restore' and _VSNAP:
        des, vec = _VSNAP[0]
        _DES.clear(); _DES.update(des); _VEC.clear(); _VEC.update(vec)
        _VSNAP.clear()
    elif op == 'drop':
        _VSNAP.clear()
    return dict(out=[], resident=_resident(), dt=0.0, **_counts())




# --------------------------------------------------------------------- driver
def _newton(w0, rad, hint, st, real, R, kind='rowx'):
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
    points' factors go to the side cache (eval_task, _exact).

    The WINDOW (20261010j, missed-peaks.md P3) is the base +- rad, as before,
    but it follows the top: where Newton stops at (within the trust step of) its
    edge with the slope still pointing out (|sigma'| > FOLLOW_GTOL sigma), the
    window is re-centred on the point and Newton continues, at most
    FOLLOW_EXTEND times per evaluation (770632 round 11: from the 2^-5 grid
    point 4.03125 the member stopped exactly on its edge 4.13125, sigma 143.48,
    slope +284; from 4.13125 Newton climbs to the top 4.180605, 151.4775, in 15
    evaluations).  A hint (the last top) outside the base window but within the
    reach of the extensions starts the walk, the window centred on it; one
    beyond restarts from the base.  EVERY window is clipped to that REACH, the
    base +- (1 + FOLLOW_EXTEND) rad, so what a walk can reach does not depend
    on where it starts (20261010j, the review of stage A: unclipped, a window
    centred on a hint reached the hint +- 3 rad, up to 6 rad from the base; on
    the final 770632 design a member based at 3.78125 then read 127.166 on the
    reach edge 4.08125 from its base, 151.4775 on the top 4.180605 from that
    edge, 127.161 from the base again (the top now beyond the hint rule), at
    successive evaluations of ONE design: a trial and its iterate disagreeing
    on the row by 16 %, the iterate accepted on a false value of STAGE5 29.2).
    A top beyond the reach is the same flagged edge stop (edge True, top
    False) at every evaluation, and the level set at max(t, smax)(1 + tol)
    enrols a member at it (missed-peaks.md P1).  An edge the re-centring
    cannot push further (the reach's own) stops the walk.  A walk that never
    meets an edge is the one of 20261010i, point for point; one that stays
    within the reach, the one of the unclipped windows.  The member's state
    records st['edge'] (the last window's edge still stopped it), st['ext']
    (re-centrings used) and st['top'] (the point is a top: converged, or off
    the edges with Newton's own distance to the top |sigma'/sigma''| below
    1e-4 -- the round's end moves a member only onto a top).  kind: the
    request of the walk's points ('rowx': value, slope and gradient, a row;
    'slopex': value and slope, the certificate's peak locator,
    LatticeOracle.peak_tops)."""
    reach = (1 + FOLLOW_EXTEND) * rad
    rlo, rhi = w0 - reach, w0 + reach              # the windows' bounds, whatever the start
    if real:
        rlo = max(rlo, 0.0)                        # w = 0: the band's own end on a real plant

    def window(c):
        return max(c - rad, rlo), min(c + rad, rhi)
    lo, hi = window(w0)
    x = float(w0)
    if hint is not None:
        if lo <= float(hint) <= hi:
            x = float(hint)
        elif rlo <= float(hint) <= rhi:
            # the top walked out of the base window at an earlier evaluation; the same
            # float bounds as the windows', so a walk's own stop on the reach edge
            # (x <= rhi exactly) is always taken back as its next hint
            x = float(hint); lo, hi = window(x)
    yield [(kind, x)]
    e = R[(kind, x)]; n = 1
    g2 = st.get('g2')
    carried = g2 is not None
    trust = min(FOLLOW_TRUST, rad)
    ext = 0
    while True:
        blocked = False
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
            blocked = not (lo <= x + stp <= hi)            # the full step leaves the window
            xt = None
            for batch in range(2):
                cands = [x + stp * 0.5 ** (k + batch * FOLLOW_HALVE) for k in range(FOLLOW_HALVE)]
                cands = [c for c in cands if lo <= c <= hi and c != x]
                if not cands:
                    continue
                yield [(kind, c) for c in cands]; n += len(cands)
                ok = [c for c in cands if R[(kind, c)]['sigma'] >= e['sigma']]
                if ok:
                    xt = ok[0]
                    break
            if xt is None:
                if carried:
                    g2 = None; carried = False             # the carried curvature failed: one fresh FD attempt
                    continue
                break
            et = R[(kind, xt)]
            dx = xt - x
            if abs(dx) > 1e-12:
                g2s = (et['gprime'] - g1) / dx             # secant update of the curvature
                g2 = g2s if g2s < 0.0 else None
            x, e = xt, et
        g1 = e['gprime']
        conv = abs(g1) <= FOLLOW_GTOL * max(1.0, e['sigma'])
        # stopped by the window: its last full step (at most a trust step) left the
        # window -- not a top inside it that a step could not improve (a kink on the
        # MIMO plateau); w = 0 is the band's own end on a real plant, not an edge
        edge = (not conv) and blocked and not (real and lo <= 0.0 and x + stp < lo)
        if edge and ext < FOLLOW_EXTEND:
            lo2, hi2 = window(x)
            # re-centre only where the window then reaches past the edge that stopped
            # the step; the reach's own edge does not move, and a re-centring on it would
            # only repeat the stop (a fresh FD pair each time on a convex flank).  Within
            # the reach this always holds (x within a trust step <= rad of the edge)
            if (hi2 > hi) if stp > 0.0 else (lo2 < lo):
                ext += 1
                lo, hi = lo2, hi2
                continue
        break
    if g2 is not None and g2 < 0.0:
        st['g2'] = float(g2)
    # a TOP: converged, or (not on an edge) Newton's own estimate of the distance to
    # the top |sigma' / sigma''| within 1e-4, the tops' agreement tolerance (a kink of
    # the MIMO plateau, where sigma_1 and sigma_2 cross, never reaches the slope test)
    near = g2 is not None and g2 < 0.0 and abs(g1) <= 1e-4 * abs(g2)
    st['edge'] = bool(edge); st['ext'] = int(ext); st['top'] = bool(conv or (near and not edge))
    return x, n


def _key(q):
    """The cache key of a request's shift (closedloop's complex(omega), omega =
    -i s: a centre s = sre + i theta has the key theta - i sre)."""
    return complex(q[2], -q[1]) if q[0] == 'centre' else complex(q[1])


class _BgLevel:
    """One level-set call running beside the next SQP iteration (20261010j stage
    D; LatticeOracle.level_start): its generator (_level) and results R, the
    requests not yet sent (pend), those out (seq -> (rank, request)), the
    current wave's batches and results for the ledger, the lines it printed."""

    def __init__(self, dz, gen, R):
        import io
        self.dz, self.gen, self.R = dz, gen, R
        self.pend, self.out, self.seq = [], {}, 0
        self.done, self.value, self.err = False, None, None
        self.waves, self.tw = 0, time.perf_counter()
        self.per, self.res = {}, {}
        self.buf = io.StringIO()


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
        self.level_prev = []             # the level-set centres that covered the band at the last call
        # kind -> [waves, tasks, wall, critical path (longest rank), factorisations, busy (all ranks)]; [4] was
        # assigned the sum of the cumulative miss counters of the wave's ranks -- not a count of anything
        # (eval-waves.md P0 verifier; 20261010j stage C: the factorisations of the kind's requests)
        self.calls = {}
        self.verbose = False
        self._dkey = None; self._dz = None
        self._line_memo = {}             # (design key, eps, wlo, whi, eps_int) -> (reach, result): the last few designs
        self._req_memo = {}              # design key -> {row request: result}, the last REQ_MEMO designs (evaluate)
        self._rot = 0                    # the ranks' tie-break order rotates from one wave to the next
        # ---- 20261010j stage C: the measured costs (TASK_COSTS) and the ledger's measurements
        self._cost = {}                  # cost kind (_rkind without the result) -> [samples, running mean SOLVES]
                                         #   ('measured' placement: counts, never seconds -- the review of stage C)
        self._lu = [0, 0.0]              # factorisations, running mean s (the ledger only)
        self._lvl_last = {}              # level-set centre theta -> its disk's last count of solves
        self._lvl_tier = {}              # level-set centre theta -> its disk's last Arnoldi tier (0: the first or
                                         #   no pencil; 1, -1: the second or none -- paired next time, LEVEL_PAIR)
        self._pstat = [0, 0, 0.0, 0, 0, 0.0, 0]   # paired disks: pairs, settled by the second tier, the first tier's s
                                         #   there; settled by the first, of which the second not sent, the second's s;
                                         #   settled by the second with the halves in turn (asynchronous: no saving)
        self._sre_line = set()           # the lines' Re = -eps (a centre there is the line's, else the strip's)
        self._req = {}                   # ledger request kind -> [tasks, work s, critical s, factorisations, their s,
                                         #   longest s, solves]: critical = on each wave's slowest rank (sums to [3])
        self._busy = {}                  # rank -> busy s in the waves
        self._solve = {}                 # rank -> [solves, their s] (a slow node, two runs on one: level-set-waves LS1)
        self._slow = [0, 0]              # waves rank 0 was the slowest rank in, waves
        self._rstat = {}                 # rank -> its latest cumulative (hit, miss, xhit, xmiss)
        self._proj = {}                  # ledger kind -> [critical path, replay on this many ranks, on PROJ_RANKS]
        # ---- 20261010j stage D: the asynchronous level set (level_start / level_join; sqp_inner.LAT_ASYNC)
        self._bg = None                  # the level-set call running beside the look-ahead (_BgLevel), or None
        self._fly = {}                   # rank -> 'b' (a level disk) or 'm' (a wave's batch): the task it has out
        self._astat = dict(launch=0, kept=0, found=0, redo=0, wait=0.0)   # calls; look-aheads kept / discarded
        self._vec_hold = set()           # ranks holding the current evaluation's row vectors: no level disk until fetched

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
        thrashed: 5816 factorisations on the driver in the central MIMO run).
        TASK_COSTS 'measured' (20261010j stage C, opt-in): the costs in MUMPS
        solves as counted (_cost_of), a factorisation LU_SOLVES, and each of the
        two passes (points held, then new points) LONGEST FIRST -- the rule above
        and its tie-break unchanged.  Counts, not seconds (the review of stage
        C): the same placement in every run of the same job."""
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

        # the two tiers of a paired level disk (LEVEL_PAIR) on two ranks: the rank of the one placed first is
        # barred to the other
        sib = {}

        def bar(q):
            return sib.get((q[1], q[2])) if q[0] == 'levelp' else None

        def placed(q, r):
            if q[0] == 'levelp':
                sib[(q[1], q[2])] = r

        def least(x=None):
            return min([r_ for r_ in ranks if r_ != x] or ranks, key=lambda r_: (load[r_], nheld[r_], order[r_]))
        if TASK_COSTS == 'measured':
            lu = LU_SOLVES
            items = sorted(((_key(q), q, self._cost_of(q)) for q in reqs), key=lambda it: -it[2])   # stable: ties in order
        else:
            lu = LU_COST
            items = [(_key(q), q, TASK_COST[q[0]]) for q in reqs]
        new = []
        for k, q, c in items:
            if k not in where:
                new.append((k, q, c))
                continue
            x = bar(q)
            hs = [r_ for r_ in where[k] if r_ != x]
            if not hs:
                new.append((k, q, c))                      # its factor held only on its pair's rank: a new point
                continue
            r = min(hs, key=load.get)
            rmin = least(x)
            if load[r] > load[rmin] + lu:
                r = rmin; c += lu; nheld[r] += 1
            out[r].append(q); load[r] += c; placed(q, r)
        for k, q, c in new:
            r = least(bar(q))
            out[r].append(q); load[r] += c + lu; nheld[r] += 1; placed(q, r)
        return {r: v for r, v in out.items() if v}

    # ------------------------------------------------- costs and the ledger
    def _rkind(self, q, o=None):
        """The ledger's kind of a request: its own, a centre split into the line's
        and the strip's, a level-set disk into norm and pencil (o: its result)."""
        k = q[0]
        if k == 'centre':
            return 'line' if float(q[1]) in self._sre_line else 'strip'
        if k in ('level', 'levelp') and o is not None:
            return 'level-pencil' if o.get('pencil') else 'level-norm'
        return 'level' if k == 'levelp' else k

    def _cost_of(self, q):
        """A request's expected MUMPS solves, its factorisation apart (TASK_COSTS
        'measured'): a level-set disk at a centre of an earlier call its own last
        count (1-2 s or 5-210 s, by the centre: norm or pencil disk; a pencil
        application is 7 + 2 x CG solves, BFS.md 5, so the count carries the
        disk's applications and CG), else its kind's running mean, else the static
        units x COST_UNIT.  Never seconds: they differ from run to run, and so
        would the placement (the review of stage C)."""
        if q[0] == 'level':
            v = self._lvl_last.get(float(q[1]))
            if v is not None:
                return v
        e = self._cost.get(self._rkind(q))
        return e[1] if e is not None else TASK_COST[q[0]] * COST_UNIT

    def _account(self, kind, per, res):
        """One wave's requests into the ledger and the measured costs: per request
        kind the tasks, the work, the share on the wave's slowest rank (the
        critical path), the factorisations and the solves; the replay of the
        wave's requests longest first on this many ranks and on PROJ_RANKS (in
        seconds: the ledger's).  The placement's costs (_cost, _lvl_last) are
        counts of solves only, so that they are the same in every run."""
        if not res:
            return
        rslow = max(res, key=lambda r_: res[r_]['dt'])
        ts = []
        for r, qs in per.items():
            o = res[r]
            tot = 0.0
            sv = self._solve.setdefault(r, [0, 0.0])
            for q, out, rc in zip(qs, o['out'], o.get('req') or ()):
                dt, nf, sf = float(rc[0]), int(rc[1]), float(rc[2])
                ns = int(rc[3]) if len(rc) > 3 else 0
                if len(rc) > 4:
                    sv[0] += ns; sv[1] += float(rc[4])
                k = self._rkind(q, out)
                a = self._req.setdefault(k, [0, 0.0, 0.0, 0, 0.0, 0.0, 0])
                a[0] += 1; a[1] += dt; a[3] += nf; a[4] += sf; a[5] = max(a[5], dt); a[6] += ns
                if r == rslow:
                    a[2] += dt
                tot += dt; ts.append(dt)
                if ns > 0:
                    # the placement's cost: the request's solves (none recorded: SuperLU, or
                    # no solve -- the kind keeps its static units)
                    e = self._cost.setdefault(self._rkind(q), [0, 0.0])
                    e[0] += 1; e[1] += (ns - e[1]) * max(1.0 / e[0], COST_EMA)
                    if q[0] == 'level':
                        self._lvl_last.pop(float(q[1]), None); self._lvl_last[float(q[1])] = float(ns)   # the latest last
                if nf > 0:
                    self._lu[0] += nf; self._lu[1] += (sf / nf - self._lu[1]) * max(float(nf) / self._lu[0], COST_EMA)
            # the rest of the rank's task: the design's ports and closed loop (once per design and rank)
            a = self._req.setdefault('setup', [0, 0.0, 0.0, 0, 0.0, 0.0, 0])
            oh = max(float(o['dt']) - tot, 0.0)
            a[1] += oh; a[5] = max(a[5], oh)
            if r == rslow:
                a[2] += oh
        while len(self._lvl_last) > 4096:
            self._lvl_last.pop(next(iter(self._lvl_last)))
        p = self._proj.setdefault(kind, [0.0] * (2 + len(PROJ_RANKS)))
        p[0] += max(float(o['dt']) for o in res.values())
        for j, nr in enumerate((len(self._ranks()),) + tuple(PROJ_RANKS)):
            p[1 + j] += _lpt(ts, int(nr))

    def _run(self, fn, per_rank, kind, ntask=None):
        """fn(args) on each rank of per_rank {rank: args}; this process does
        rank 0's share meanwhile.  Returns {rank: result}.  While a level set
        runs beside (20261010j stage D): _run_async."""
        if self._bg is not None:
            return self._run_async(fn, per_rank, kind, ntask)
        t0 = time.perf_counter()
        res = {}
        others = {r: a for r, a in per_rank.items() if r != 0}
        ml = _mumps_mod()
        if others and self.pool is not None:
            ranks = self.pool.each(fn, others)
            try:
                if 0 in per_rank:
                    # rank 0's own share of the wave, booked apart from the driver's
                    # serial MUMPS work (bfs/mumps_lu 'wave'; driver-serial.md P1: the
                    # ledger's 'driver' was rank 0 in any context)
                    if ml is not None:
                        ml.IN_WAVE = True
                    try:
                        res[0] = fn(per_rank[0])
                    finally:
                        if ml is not None:
                            ml.IN_WAVE = False
            finally:
                res.update(self.pool.collect(ranks))
        else:
            if ml is not None:
                ml.IN_WAVE = True
            try:
                for r, a in per_rank.items():
                    res[r] = fn(a)
            finally:
                if ml is not None:
                    ml.IN_WAVE = False
        return self._book(res, kind, ntask if ntask is not None else len(per_rank), t0)

    def _book(self, res, kind, ntask, t0):
        """A wave's results into the driver's state and the ledger: the factors
        each rank holds, its counters and busy time, the kind's waves, tasks,
        wall, critical path, factorisations and busy time, the waves rank 0 was
        the slowest in."""
        for r, o in res.items():
            self.held[r] = set(complex(k) for k in o['resident'])
            self._rstat[r] = tuple(int(o.get(k_, 0)) for k_ in ('hit', 'miss', 'xhit', 'xmiss'))
            self._busy[r] = self._busy.get(r, 0.0) + float(o['dt'])
        c = self.calls.setdefault(kind, [0, 0, 0.0, 0.0, 0, 0.0])
        c[0] += 1; c[1] += int(ntask)
        c[2] += time.perf_counter() - t0
        c[3] += max((o['dt'] for o in res.values()), default=0.0)
        c[4] += sum(int(rc[1]) for o in res.values() for rc in (o.get('req') or ()))
        c[5] += sum(o['dt'] for o in res.values())       # busy time summed over the ranks (utilisation)
        if len(res) > 1 and max(o['dt'] for o in res.values()) > 0.0:     # a wave of tasks (not stats, vectors)
            self._slow[1] += 1
            self._slow[0] += int(max(res, key=lambda r_: res[r_]['dt']) == 0)
        return res

    def _drive(self, dz, agents, kind, R, track=None):
        """Run the agents -- generators of requests (eval_task) that read their
        results from R -- side by side: every wave sends all their outstanding
        requests at once, one batch per rank, and resumes each agent.
        Returns ({name: the agent's return value}, waves).  track (a dict;
        20261010j stage C, omega-growth.md P1): per agent the requests it was the
        first to ask for in a wave -- the tasks it cost ('own'; an identical chain
        of another member costs none) -- and the wave it finished in ('fin').
        Each result carries its rank and its own seconds ('dt_')."""
        done, pend, fin = {}, {}, {}
        waves = 0

        def step(nm, g):
            try:
                while True:
                    q = [x for x in next(g) if x not in R]
                    if q:
                        pend[nm] = (g, q)
                        return
            except StopIteration as e:
                done[nm] = e.value; fin[nm] = waves
                pend.pop(nm, None)
        for nm, g in agents.items():
            step(nm, g)
        own = track.setdefault('own', {}) if track is not None else None
        while pend:
            waves += 1
            owner = {}
            for nm, (_, qs) in pend.items():
                for q in qs:
                    owner.setdefault(q, nm)
            reqs = list(owner)                     # the requests in the order first asked, each once
            if own is not None:
                for nm in owner.values():
                    own[nm] = own.get(nm, 0) + 1
            if self._bg is None:
                per = self._assign(reqs)
                res = self._run(eval_task, {r: (dz, qs) for r, qs in per.items()}, kind, ntask=len(reqs))
            else:
                per, res = self._run_flow(dz, reqs, kind)      # a level set runs beside (20261010j stage D)
            self._account(kind, per, res)
            for r, qs in per.items():
                rq = res[r].get('req') or [(0.0,)] * len(qs)
                for q, o, rc in zip(qs, res[r]['out'], rq):
                    o['rank'] = r; o['dt_'] = float(rc[0])
                    R[q] = o
            for nm in list(pend):
                step(nm, pend[nm][0])
        if track is not None:
            track.setdefault('fin', {}).update(fin)
        return done, waves

    # ------------------------------------- the asynchronous level set (stage D)
    # 20261010j stage D (level-set-waves.md LS3 and its verifier, eval-waves.md
    # P5, driver-serial.md P3; --lattice-async, sqp_inner.LAT_ASYNC).  In the
    # no-strip runs the missed-peak level set ran alone after every accepted
    # step, every wave as long as its slowest pencil disk (central 770632: 65 % of
    # the wall, ranks busy 8 %), while the next iteration's QP and trials waited.
    # level_start sends the call's first wave and returns; the SQP goes on with the
    # next iteration (its look-ahead) and level_join waits for the result.  The
    # level set's disks go one request a message to the worker ranks with no task
    # out -- NEVER rank 0, which must stay free to serve the look-ahead and to
    # release each rank as its disk returns (recv by source, MPIPool.post /
    # recv_any: not one outstanding wave per pool) --, and its later waves (gaps,
    # slope confirmations) as soon as the earlier one is answered, onto the ranks
    # the look-ahead's waves leave free.  The look-ahead's own waves go one request
    # a message to the workers as each becomes free, before the level set's
    # pending disks (_run_flow: the driver dispatches only); a task for a named
    # rank still busy (the vectors of its rows) is posted the moment that rank
    # returns (_run_async), and a rank holding the evaluation's row vectors gets
    # no level disk until vectors() has fetched them (_vec_hold).  The level set's
    # own result is the synchronous one: the same generator (_level) from the same
    # state (level_prev, the base grid, the watch rows), each disk a function of
    # its request alone (Determinism, above; placement changes only the cost), and
    # its tasks run on their design built aside (_design: no design evicted, no
    # vector purged between the look-ahead's requests).  What the SQP does with
    # the look-ahead (kept when the call adds no member, else discarded and the
    # synchronous continuation taken) is sqp_inner's.

    def level_start(self, z, gam, wlo, whi, watch=None):
        """level() BESIDE the next iteration: the call's first wave sent, its job
        returned for level_join.  None where it cannot run so (one rank, the
        serial path: the caller then calls level()).  Every rank is idle here
        (no task out): their designs and line vectors are marked first
        (vecstate_task), for a discarded look-ahead to put back."""
        if self._bg is not None:
            raise RuntimeError("lattice: a level set is already running beside the look-ahead")
        dz = z if isinstance(z, dict) else self.design(z)
        lk = (dz['key'], float(gam), float(wlo), float(whi))
        if self.pool is None or self.size < 2:
            return None
        self._vecstate('mark')
        R = {}
        job = _BgLevel(dict(dz, aside=True), self._level(*lk[1:], R, watch=watch), R)
        self._bg = job
        self._astat['launch'] += 1
        self._bg_step(job)
        self._bg_pump()
        return job

    def level_join(self, job):
        """level_start's result (the dict level() returns), waiting for it --
        every level disk still out received, its later waves sent as before."""
        t0 = time.perf_counter()
        self._vec_hold.clear()                             # nothing else to run: every worker the call's
        try:
            while not job.done or job.out:
                if job.err is None:
                    self._bg_pump()
                if not self._fly:
                    if job.err is not None:
                        break
                    raise RuntimeError("lattice: the asynchronous level set is not done and has nothing out")
                self._recv_one()
        finally:
            if not self._fly:
                self._bg = None
        self._astat['wait'] += time.perf_counter() - t0
        if job.err is not None:
            raise RuntimeError(f"lattice: the asynchronous level set failed: {job.err}")
        return job.value

    def note_async(self, what):
        """The SQP's verdict on a look-ahead: 'kept', 'found' (discarded: members
        added) or 'redo' (discarded: --lattice-async redo) -- the ledger's line."""
        self._astat[what] = self._astat.get(what, 0) + 1

    def async_restore(self):
        """A discarded look-ahead: every rank's designs and line vectors back to
        its start (vecstate_task 'restore'; the factor caches stay)."""
        self._vecstate('restore')

    def async_drop(self):
        """A kept look-ahead: the ranks' marks dropped (vecstate_task 'drop'; they
        hold the designs and vectors the look-ahead let go)."""
        self._vecstate('drop')

    def async_abort(self):
        """After an error: every task still out received and dropped (no stale
        result may reach a later call), the level set beside forgotten."""
        try:
            while self._fly and self.pool is not None:
                src, _, _, _ = self.pool.recv_any()
                self._fly.pop(src, None)
        finally:
            self._fly.clear(); self._bg = None

    def _vecstate(self, op):
        """vecstate_task(op) on every rank, this process's included (no task is
        out when it is called: a look-ahead's start, or after its join)."""
        vecstate_task(op)
        if self.pool is not None and self.size > 1:
            others = {r: op for r in range(1, self.size)}
            self.pool.each(vecstate_task, others)
            self.pool.collect(list(others))

    def _run_flow(self, dz, reqs, kind):
        """A wave of the look-ahead while a level set runs beside: its requests
        one a message to the worker ranks as each becomes free, BEFORE the level
        set's pending disks (a disk already running is never waited for by the
        wave as a whole: a batch assigned to a rank still busy with a 50-s pencil
        disk would hold it; and a wave assigned over the ranks free at its start
        packed itself onto them -- the first sandbox run, every worker on a disk,
        put a whole wave on rank 0 while the workers then sat idle).  A request
        goes first to a free rank holding its factor, then to the free ranks
        holding the fewest factors -- unless a rank busy with another request of
        the wave holds it (it waits for that rank); one held by a rank on a level
        disk is factored again elsewhere.  The driver only dispatches here (a
        request it computed would leave every worker that returns meanwhile
        idle).  Returns (per, res) as _assign / _run: {rank: [requests]} and
        {rank: its results merged in that order}."""
        t0 = time.perf_counter()
        pend = list(reqs)
        per, res, err = {}, {}, None
        nleft = len(pend)
        workers = list(range(1, self.size))

        def post(r, q):
            pend.remove(q)
            self.pool.post(r, eval_task, (dz, [q]), ('m', q)); self._fly[r] = 'm'
            per.setdefault(r, []).append(q)
        while nleft:
            free = [r for r in workers if r not in self._fly]
            if pend and free:
                n = len(free); rot = self._rot % n
                order = {r: (j - rot) % n for j, r in enumerate(free)}
                self._rot = (self._rot + 1) % max(len(workers), 1)
                left = []
                for r in sorted(free, key=order.get):          # a free rank holding a request's factor takes it
                    hs = self.held.get(r, ())
                    q = next((q_ for q_ in pend if _key(q_) in hs), None)
                    if q is None:
                        left.append(r)
                    else:
                        post(r, q)
                holders = {}
                for r, ks in self.held.items():
                    for k in ks:
                        holders.setdefault(k, []).append(r)
                for r in sorted(left, key=lambda r_: (len(self.held.get(r_, ())), order[r_])):
                    # then a request no rank busy with a wave's request holds (held by none, by a rank
                    # on a level disk or by rank 0: factored here); one such a rank holds waits for it
                    q = next((q_ for q_ in pend if not any(self._fly.get(h_) == 'm' for h_ in holders.get(_key(q_), ()))),
                             None)
                    if q is None:
                        break
                    post(r, q)
            self._bg_pump()                                    # the ranks the wave leaves free
            src, tid, ok, o = self._recv_one()
            if tid[0] != 'm':
                continue
            nleft -= 1
            if not ok:
                err = err or f"MPI worker {src} failed: {o}"
                continue
            # a rank runs one request at a time and is sent the next only once this
            # result is in: its results arrive in the order of per[src]
            m = res.setdefault(src, dict(out=[], req=[], dt=0.0))
            m['out'] += o['out']; m['req'] += list(o.get('req') or [(0.0,)] * len(o['out'])); m['dt'] += float(o['dt'])
            if tid[1][0] == 'centre' and o['out'] and o['out'][0].get('rows'):
                # its row vectors live there: no level disk on it until vectors() has fetched them (the
                # first sandbox run waited 144 s for row ranks busy with 50-90 s pencil disks)
                self._vec_hold.add(src)
            for k_ in ('resident', 'hit', 'miss', 'xhit', 'xmiss'):
                m[k_] = o.get(k_, 0)
        if err is not None:
            raise RuntimeError(err)
        self._book(res, kind, len(reqs), t0)
        return per, res

    def _recv_one(self):
        """The next result of a task out (MPIPool.recv_any), its rank released; a
        level disk's goes to the level set (_bg_result).  (rank, tid, ok, result)."""
        src, tid, ok, o = self.pool.recv_any()
        self._fly.pop(src, None)
        if tid[0] == 'b':
            self._bg_result(src, tid, ok, o)
        return src, tid, ok, o

    def _run_async(self, fn, per_rank, kind, ntask=None):
        """_run while a level set runs beside: each batch posted as soon as its
        rank has no task out (a rank still busy with a level disk gets its batch
        the moment that disk returns), rank 0's share computed here meanwhile,
        the results gathered one at a time (the level set's as they come, its
        next disks sent to the ranks left free).  Every result of the wave is
        received before an error is raised (MPIPool.collect's rule)."""
        t0 = time.perf_counter()
        res, err, wait = {}, None, []
        ml = _mumps_mod()
        for r, a in per_rank.items():
            if r == 0:
                continue
            if r in self._fly:
                wait.append(r)
            else:
                self.pool.post(r, fn, a, ('m', r)); self._fly[r] = 'm'
        if 0 in per_rank:
            if ml is not None:
                ml.IN_WAVE = True
            try:
                res[0] = fn(per_rank[0])
            finally:
                if ml is not None:
                    ml.IN_WAVE = False
        need = sum(1 for r in per_rank if r != 0)
        while need:
            src, tid, ok, o = self._recv_one()
            if tid[0] == 'm':
                need -= 1
                if ok:
                    res[src] = o
                elif err is None:
                    err = f"MPI worker {src} failed: {o}"
            if src in wait:
                wait.remove(src)
                self.pool.post(src, fn, per_rank[src], ('m', src)); self._fly[src] = 'm'
            self._bg_pump()
        if err is not None:
            raise RuntimeError(err)
        return self._book(res, kind, ntask if ntask is not None else len(per_rank), t0)

    def _bg_step(self, job):
        """Resume the level set's generator once its wave is answered (as
        _drive's step): its next requests into pend, or its result.  Its
        printed lines go to the job's own buffer."""
        import contextlib
        try:
            with contextlib.redirect_stdout(job.buf):
                while True:
                    q = [x for x in next(job.gen) if x not in job.R]
                    if q:
                        job.pend = list(dict.fromkeys(q))
                        job.waves += 1; job.tw = time.perf_counter(); job.per, job.res = {}, {}
                        return
        except StopIteration as e:
            job.done, job.value = True, e.value
        except Exception as e:
            import traceback
            job.err = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"

    def _bg_pump(self):
        """The level set's pending requests to the free WORKER ranks, one request a
        rank: a point first to a free rank holding its factor, then to the free
        ranks holding the fewest factors (_assign's rule for a new point), the
        ties in the rotating order; not to a rank whose row vectors the current
        evaluation has yet to fetch (_vec_hold)."""
        job = self._bg
        if job is None or job.err is not None or not job.pend:
            return
        free = [r for r in range(1, self.size) if r not in self._fly and r not in self._vec_hold]
        if not free:
            return
        n = len(free); rot = self._rot % n
        order = {r: (j - rot) % n for j, r in enumerate(free)}
        left = []
        for q in job.pend:
            hold = [r for r in free if _key(q) in self.held.get(r, ())] if free else []
            if hold:
                self._bg_post(job, hold[0], q); free.remove(hold[0])
            else:
                left.append(q)
        rest = []
        for q in left:
            if not free:
                rest.append(q); continue
            r = min(free, key=lambda r_: (len(self.held.get(r_, ())), order[r_]))
            self._bg_post(job, r, q); free.remove(r)
        job.pend = rest

    def _bg_post(self, job, r, q):
        job.seq += 1
        self.pool.post(r, eval_task, (job.dz, [q]), ('b', job.seq))
        job.out[job.seq] = (r, q); self._fly[r] = 'b'

    def _bg_result(self, src, tid, ok, o):
        """A level disk's result: into the call's R (its rank and seconds, as
        _drive's), the ledger's per-rank state; its wave complete, the wave
        booked ('level-bg': its wall overlaps the look-ahead's waves) and the
        generator resumed."""
        job = self._bg
        _, q = job.out.pop(tid[1])
        if not ok:
            if job.err is None:
                job.err = f"MPI worker {src} failed: {o}"
            return
        self.held[src] = set(complex(k) for k in o['resident'])
        self._rstat[src] = tuple(int(o.get(k_, 0)) for k_ in ('hit', 'miss', 'xhit', 'xmiss'))
        self._busy[src] = self._busy.get(src, 0.0) + float(o['dt'])
        rc = (o.get('req') or [(0.0,)])[0]
        res = o['out'][0]
        res['rank'] = src; res['dt_'] = float(rc[0])
        job.R[q] = res
        if q[0] == 'levelp' and int(q[3]) == 0:
            # did the pair's halves overlap (the ledger credits the pair's saving only then;
            # the review of 20261010s: a second half sent after the first had returned ran in
            # turn, and was counted as saving the first's seconds)
            sib_ = ('levelp', q[1], q[2], 1)
            res['sib_overlap'] = bool(sib_ in job.R or any(v_[1] == sib_ for v_ in job.out.values()))
        if q[0] == 'levelp' and int(q[3]) == 0 and (not res.get('pencil') or res.get('tier') == 0):
            # a paired disk's first tier final (LEVEL_PAIR): its second, if not yet sent, is not needed
            sib = ('levelp', q[1], q[2], 1)
            if sib in job.pend:
                job.pend.remove(sib); job.R[sib] = dict(skipped=True)
                self._astat['pair_skip'] = self._astat.get('pair_skip', 0) + 1
        job.per.setdefault(src, []).append(q)
        m = job.res.setdefault(src, dict(out=[], req=[], dt=0.0))
        m['out'].append(res); m['req'].append(rc); m['dt'] += float(o['dt'])
        if job.out or job.pend or job.done or job.err is not None:
            return
        self._account('level-bg', job.per, job.res)
        c = self.calls.setdefault('level-bg', [0, 0, 0.0, 0.0, 0, 0.0])
        c[0] += 1; c[1] += sum(len(v) for v in job.per.values())
        c[2] += time.perf_counter() - job.tw
        c[3] += max((m_['dt'] for m_ in job.res.values()), default=0.0)
        c[4] += sum(int(rc_[1]) for m_ in job.res.values() for rc_ in m_['req'] if len(rc_) > 1)
        c[5] += sum(m_['dt'] for m_ in job.res.values())
        self._bg_step(job)

    def report(self):
        """One line for the ledger: calls (waves), tasks, wall and critical path
        by kind, and the factors held."""
        held = sum(len(v) for v in self.held.values())
        nr = len(self._ranks())
        parts = [f"{k} {c[0]} waves / {c[1]} tasks, {c[2]:.0f} s wall (longest rank {c[3]:.0f} s, ranks busy "
                 f"{100.0 * c[5] / max(nr * c[2], 1e-300):.0f} %)"
                 for k, c in sorted(self.calls.items())]
        return f"lattice ({self.size} ranks): " + "; ".join(parts) + f"; {held} factors held"

    def report_lines(self):
        """The ledger's measurements beside report() (20261010j stage C; cumulative,
        each line 'lattice ...' for the '[ledger]' prefix):
          rank 0    its busy time in the waves against the mean and the busiest
                    rank, how often it was the slowest rank, the time per solve
                    (rank 0, and over the ranks: a slow or shared node shows
                    there), and its own serial MUMPS work outside the waves
                    (driver-serial.md P1);
          requests  per request kind: tasks, work, the critical path (the share on
                    each wave's slowest rank: the kinds sum to the 'longest rank'
                    seconds), the longest task, the mean seconds and solves
                    (eval-waves.md P0; the solves are what 'measured' balances);
          factors   the factorisations (MUMPS, per request: lattice points and the
                    exact points apart), the cache hits and misses of both caches,
                    the running mean of a factorisation (eval-waves.md P0 verifier:
                    c[4] was not a miss count, the exact points were not counted)
                    and that mean in solves (the check of LU_SOLVES), the
                    placement's units;
          projection  every wave's requests replayed longest first on this many
                    ranks and on PROJ_RANKS, against the measured critical path
                    (packing.md P3: no factor moves with a task there, so the misses
                    of a smaller cache, the driver's dispatch and a slower node are
                    not in it)."""
        out = []
        ranks = self._ranks()
        if self._busy:
            b = [self._busy.get(r, 0.0) for r in ranks]
            rmax = max(ranks, key=lambda r_: self._busy.get(r_, 0.0))
            ml = _mumps_mod()
            ser = ''
            if ml is not None:
                f_, s_ = ml.STATS.get('factor_driver', [0, 0.0]), ml.STATS.get('solve_driver', [0, 0.0])
                ser = f"; serial MUMPS outside the waves {f_[0]} factorisations ({f_[1]:.0f} s), {s_[0]} solves ({s_[1]:.0f} s)"
            ms = {r: 1e3 * v[1] / v[0] for r, v in self._solve.items() if v[0] > 0}
            sol = ''
            if ms:
                q_ = np.percentile(list(ms.values()), [0, 50, 100])
                sol = (f"; ms per solve: rank 0 {ms.get(0, float('nan')):.1f}, ranks {q_[0]:.1f} / {q_[1]:.1f} / {q_[2]:.1f} "
                       f"(min / median / max)")
            out.append(f"lattice rank 0: busy {self._busy.get(0, 0.0):.0f} s in the waves (mean rank "
                       f"{np.mean(b):.0f} s, busiest rank {rmax} {self._busy.get(rmax, 0.0):.0f} s), the slowest rank in "
                       f"{self._slow[0]} of {self._slow[1]} waves" + ('' if self.use_driver else ' (--lattice-driver off)')
                       + sol + ser)
        if self._req:
            order = sorted(self._req, key=lambda k_: -self._req[k_][2])
            out.append("lattice requests (tasks / work / critical / longest, s): " + "; ".join(
                f"{k} {a[0]} / {a[1]:.0f} / {a[2]:.0f} / {a[5]:.1f}" + (f" ({a[1] / a[0]:.2f} s" + (
                    f", {a[6] / a[0]:.0f} solves" if len(a) > 6 and a[6] else '') + " each)" if a[0] else '')
                for k, a in ((k_, self._req[k_]) for k_ in order)))
            fx = sum(self._req[k][3] for k in self._req if k in ('rowx', 'slopex', 'blockx'))
            fl = sum(self._req[k][3] for k in self._req) - fx
            sf = sum(self._req[k][4] for k in self._req)
            h, m_, xh, xm = (sum(v[j] for v in self._rstat.values()) for j in range(4))
            # a factorisation in mean solves over the ranks: the check of LU_SOLVES ('measured' placement)
            ns_ = sum(v[0] for v in self._solve.values()); ss_ = sum(v[1] for v in self._solve.values())
            lus = f" = {self._lu[1] / (ss_ / ns_):.0f} solves" if ns_ and ss_ > 0.0 and self._lu[0] else ''
            out.append(f"lattice factors: {fl + fx} factorisations in the requests ({sf:.0f} s; "
                       f"{self._lu[1]:.2f} s each, running mean{lus}): lattice points {fl} (cache {h} hits / {m_} misses), "
                       f"exact points {fx} (side cache {xh} hits / {xm} misses); "
                       f"{sum(len(v) for v in self.held.values())} held; placement "
                       + (f"by solves counted, a factorisation {LU_SOLVES:g}" if TASK_COSTS == 'measured'
                          else "by the static units"))
        if self._astat['launch']:
            # the asynchronous level set (20261010j stage D): its calls beside the next iteration, the
            # look-aheads kept (the call added no member) and discarded, the joins' wait (the level set's
            # time not hidden behind the look-ahead's waves)
            a_, c_ = self._astat, self.calls.get('level-bg', [0, 0, 0.0, 0.0, 0, 0.0])
            out.append(f"lattice async: {a_['launch']} level-set call(s) beside the next iteration, look-aheads "
                       f"{a_['kept']} kept / {a_['found']} discarded for members found"
                       + (f" / {a_['redo']} redone (--lattice-async redo)" if a_.get('redo') else '')
                       + f"; their {c_[0]} waves {c_[2]:.0f} s open, {c_[5]:.0f} s of disks; the joins waited "
                         f"{a_['wait']:.0f} s")
        if self._pstat[0]:
            p_ = self._pstat
            out.append(f"lattice paired level disks (LEVEL_PAIR): {p_[0]} pairs; {p_[1]} settled by the second tier "
                       f"(the first tier's {p_[2]:.0f} s off their walls), {p_[3]} by the first ({p_[4]} second tiers "
                       f"not sent, {p_[5]:.0f} s of second tiers wasted)"
                       + (f"; {p_[6]} settled by the second tier with the halves in turn (no saving)" if p_[6] else ''))
        if self._proj:
            nr = len(ranks)
            out.append("lattice projection (requests replayed longest first; critical path / on " +
                       " / ".join(str(r_) for r_ in (nr,) + tuple(PROJ_RANKS)) + " ranks, s): " + "; ".join(
                f"{k} {p[0]:.0f} / " + " / ".join(f"{x:.0f}" for x in p[1:]) for k, p in sorted(self._proj.items())))
        return out

    def look_state(self):
        """The driver's state an evaluation changes and a later one reads, as at a
        look-ahead's start (20261010j stage D; put back by look_restore when the
        look-ahead is discarded): the followed tops, the line's and the strip's
        kept centres (the next call's first wave: the rows depend on the line's
        cover), the line and request memos, the last design.  Not
        level_prev (the level call running beside sets it, as the synchronous
        call did before the next iteration), nor the placement (held, the
        rotation, the measured costs: the cost only) and the ledger."""
        return dict(tops=dict(self.tops), line_prev=dict(self.line_prev), strip_prev=dict(self.strip_prev),
                    line_memo=dict(self._line_memo),
                    req_memo={k: dict(v) for k, v in self._req_memo.items()}, dkey=self._dkey, dz=self._dz)

    def look_restore(self, S):
        self.tops = dict(S['tops']); self.line_prev = dict(S['line_prev']); self.strip_prev = dict(S['strip_prev'])
        self._line_memo = dict(S['line_memo'])
        self._req_memo = {k: dict(v) for k, v in S['req_memo'].items()}
        self._dkey, self._dz = S['dkey'], S['dz']

    def counts(self):
        """(factorisations, cached hits) over all ranks so far."""
        res = self._run(stats_task, {r: None for r in self._ranks()}, 'stats')
        return sum(o['miss'] for o in res.values()), sum(o['hit'] for o in res.values())

    # -------------------------------------------------------------- evaluation
    def evaluate(self, z, W, follow=None, w_star=None, state=None, enroll=(), line=None, strip=False, dedup=False):
        """sigma_1 rows at every member of W and, with line = (eps, wlo, whi),
        the stability line -- in ONE sequence of waves: the first carries the
        fixed rows, every followed member's first Newton point, the line's first
        centres; the later ones the Newton steps and the line's gap centres side
        by side.  follow: {member key: radius} (the followed peaks); w_star /
        state: per member, the last top and the carried curvature (updated
        here); enroll: the members whose sigma_2 block is built, at the row
        point.  Returns (s, Gs, blocks, tops, info, line result or None); the
        line result's rows are every eigenvalue the line (and the strip)
        exhibited in the strip [-eps - ROW_REACH, -eps].
        strip = True: a DIAGNOSTIC since 2026-10-16 (tools/check_lattice.py, the
        Evaluator's sqp_inner.LAT_STRIP): the strip covered in the same waves,
        its shifts beside the line's, every eigenvalue in it a row.  The
        optimiser's evaluations are the line alone (the strip arm, which covered
        it at every iterate the QP read and re-evaluated each accepted trial with
        it and the level set in the same waves -- evaluate(level=...), removed
        with it -- is gone: BFS.md 8).
        dedup (20261010j, missed-peaks.md P4; the reduced path's FOLLOW_DEDUP,
        STAGE5 29.2, on the lattice): (i) a member without a hint starts on the
        nearest top of an earlier evaluation within half its radius only if no
        member nearer to that top claims it -- the others start from their own
        base (two members on one previous top ran identical chains and stayed
        locked together: the 'sigma evaluations' outnumbered the tasks in every
        20261010g no-strip run, eval-waves.md); (ii) members whose tops agree
        within 1e-4 keep one, the one whose base is nearest, and the others
        restart once from their own base, cold, in more waves; a restart back on
        a shared top marks the member merged (state 'merged'), not restarted
        again near that top (the round's end merges it).
        The row requests of the last REQ_MEMO designs are kept (20261010j stage
        C, eval-waves.md P8): an evaluation of the same design (the refine's
        re-evaluation) finds its fixed rows and the points its Newton chains
        revisit answered -- the same values, the same decisions, fewer tasks;
        the centres are not (the line has its own memo, the strip reads every
        centre of its evaluation).  info['members']: per
        member kind the points asked for and the tasks caused, by s / smax, and
        which agents set the last wave (omega-growth.md P1, strip-ab.md AB4)."""
        dz = self.design(z)
        self._vec_hold.clear()                             # (stage D) the last evaluation's vectors are not fetched now
        follow = follow or {}
        w_star = {} if w_star is None else w_star
        state = {} if state is None else state
        enroll = set(int(i) for i in (enroll or ()))
        R = {}
        if REQ_MEMO > 0 and dz['key'] in self._req_memo:
            R.update(self._req_memo[dz['key']])
        track = {}

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
        claimed = None
        if dedup and self.tops:
            # (i): each earlier top to one member, the nearest by base
            claim = {}
            for i, w in enumerate(W):
                k6 = round(float(w), 6)
                rad = follow.get(k6)
                if rad and w_star.get(k6) is None:
                    q = min(self.tops, key=lambda x: abs(x - float(w)))
                    if abs(q - float(w)) <= 0.5 * float(rad) and \
                            (q not in claim or abs(float(w) - q) < abs(float(W[claim[q]]) - q)):
                        claim[q] = i
            claimed = {i: q for q, i in claim.items()}
        cold = set()                                       # members whose walk starts from their own base
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
                    if claimed is not None:
                        hint = claimed.get(i)
                    else:
                        q = min(self.tops, key=lambda x: abs(x - float(w)))
                        if abs(q - float(w)) <= 0.5 * float(rad):
                            hint = q
                if hint is None:
                    cold.add(i)
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
        done, waves = self._drive(dz, agents, 'eval', R, track=track)
        ndedup = nev_redo = 0
        if dedup and follow:
            # (ii): members on one top -- groups of tops within 1e-4 (Newton's tops
            # agree to ~1e-6 on a flat peak, not 1e-9; the reduced path's rule)
            ft = sorted((float(done[i][0]), i) for i in range(len(W)) if follow.get(round(float(W[i]), 6)))
            groups, g_ = [], []
            for tp, i in ft:
                if g_ and tp - g_[-1][0] > 1e-4:
                    groups.append(g_); g_ = []
                g_.append((tp, i))
            if g_:
                groups.append(g_)
            redo = []
            for g_ in groups:
                if len(g_) < 2:
                    continue
                tp = g_[0][0]
                keep = min(g_, key=lambda q_: abs(float(W[q_[1]]) - tp))[1]
                for _, i in g_:
                    k6 = round(float(W[i]), 6)
                    if i == keep or abs(float(W[i]) - tp) <= 1e-3:
                        continue                           # the member at the top itself
                    if i in cold:
                        state[k6]['merged'] = tp           # its own base's walk came here: merged already
                    elif abs(state[k6].get('merged', np.inf) - tp) > 5e-3:
                        redo.append(i)
            if redo:
                ag2 = {}
                for i in redo:
                    k6 = round(float(W[i]), 6)
                    st_new = {k_: v_ for k_, v_ in state[k6].items() if k_ == 'merged'}
                    state[k6] = st_new                     # cold: no carried curvature either
                    nev_redo += done[i][2]
                    track.setdefault('nev0', {})[i] = int(done[i][2])     # the first chain's points (the member log)
                    ag2[i] = member(W[i], follow[k6], None, st_new, i in enroll)
                tr2 = {}
                done2, waves2 = self._drive(dz, ag2, 'eval', R, track=tr2)
                for nm_, c_ in tr2.get('own', {}).items():
                    track['own'][nm_] = track['own'].get(nm_, 0) + c_
                for nm_, f_ in tr2.get('fin', {}).items():
                    track['fin'][nm_] = waves + f_         # the restarts' waves follow the first drive's
                waves += waves2
                for i in redo:
                    done[i] = done2[i]
                    x_ = float(done2[i][0])
                    if any(abs(x_ - tp) <= 1e-4 for tp, _ in ft):
                        state[round(float(W[i]), 6)]['merged'] = x_    # back on a shared top
                ndedup = len(redo)
        if REQ_MEMO > 0:
            mem_ = self._req_memo.pop(dz['key'], {})       # the most recent design last (dropped last)
            mem_.update((q, o) for q, o in R.items() if q[0] in _MEMO_KINDS)
            self._req_memo[dz['key']] = mem_
            while len(self._req_memo) > int(REQ_MEMO):
                self._req_memo.pop(next(iter(self._req_memo)))
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
        # edge: followed members whose last window's edge still stopped them (FOLLOW_EXTEND used up);
        # ext: window re-centrings; dedup: members restarted from their own base (P4 (ii))
        info = dict(nev=nev_redo, newton_max=0, waves=waves, edge=0, ext=0, dedup=ndedup)
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
                info['edge'] += int(bool(state[k6].get('edge'))); info['ext'] += int(state[k6].get('ext', 0))
            if i in enroll:
                B = r['block']
                Gc = np.zeros((2, 2, NZ), complex)
                Gc[0, 0] = grad(B['g00']); Gc[1, 1] = grad(B['g11'])
                Gc[0, 1] = grad(B['g01']) + 1j * grad(B['g01i'])
                Gc[1, 0] = grad(B['g10']) + 1j * grad(B['g10i'])
                blocks[i] = dict(sig=np.array(B['sig']), Gc=Gc)
        info['members'] = self._member_log(W, follow, done, track, s, waves)
        return s, Gs, blocks, tops, info, lr

    @staticmethod
    def _member_log(W, follow, done, track, s, waves):
        """The evaluation's per-member counts (20261010j stage C, omega-growth.md
        P1 and its verifier): per kind (fixed, followed) the members, the points
        their agents asked for (nev, the round's 'sigma evaluations') and the
        tasks they caused (the requests each was the first to ask for: what was
        computed), the followed members that caused none (a chain identical to
        another's, or answered by the memo); the same by s / smax (MEMBER_BINS);
        and which agents finished in the last wave (members by kind, the line,
        the strip, the level set: the evaluation's critical path in waves)."""
        own, fin, nev0 = track.get('own', {}), track.get('fin', {}), track.get('nev0', {})
        smax = float(np.max(s)) if len(s) else 0.0
        nb = len(MEMBER_BINS) + 1
        out = dict(n=[0, 0], pts=[0, 0], own=[0, 0], none=0, bn=[0] * nb, bpts=[0] * nb, bown=[0] * nb)
        for i in range(len(W)):
            j = int(bool(follow.get(round(float(W[i]), 6))))
            nev, ow = int(done[i][2]) + int(nev0.get(i, 0)), int(own.get(i, 0))
            out['n'][j] += 1; out['pts'][j] += nev; out['own'][j] += ow
            out['none'] += int(j == 1 and ow == 0)
            b = int(np.searchsorted(MEMBER_BINS, float(s[i]) / smax, side='right')) if smax > 0 else 0
            out['bn'][b] += 1; out['bpts'][b] += nev; out['bown'][b] += ow
        last = set()
        for nm, f_ in fin.items():
            if f_ == waves:
                last.add(nm if isinstance(nm, str) else ('followed' if follow.get(round(float(W[nm]), 6)) else 'fixed'))
        out['last'] = sorted(last)
        return out

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

    def peak_tops(self, z, ws, rad, states=None):
        """The tops of the peaks whose grid maxima are ws (20261010j, missed-
        peaks.md P6): each located by the followed members' Newton (_newton, value
        and slope only: 'slopex'), all side by side in one sequence of waves,
        window w +- rad and its re-centrings.  The certificate's peaks used to be
        enrolled at their 2^-5 grid points (k/32: 4.03125 in 770632), so a new
        member's window sat off its top by up to a grid step -- one of the two
        causes of the round-11 edge stop.  Returns [(top, sigma_1 there)], each
        at least its grid point's value (Newton accepts only ascent).  states: a
        dict per point, filled with the walk's state (_newton: 'top', 'edge',
        'ext', the curvature 'g2') -- the in-solve level set's regions, located
        before they are enrolled (20261010j stage E), are enrolled at the point
        only where it is a top."""
        dz = self.design(z)
        R = {}
        sts = list(states) if states is not None else [{} for _ in ws]
        agents = {i: _newton(float(w), float(rad), None, sts[i], self.real, R, kind='slopex') for i, w in enumerate(ws)}
        done, _ = self._drive(dz, agents, 'locate', R)
        return [(float(done[i][0]), float(R[('slopex', done[i][0])]['sigma'])) for i in range(len(ws))]

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
        """The line's (and the level set's first) base grid: LINE_BASE equispaced
        centres on [wlo, whi] whatever the number of ranks (20261010j stage C;
        LINE_BASE = 0: ranks/2, at 96 ranks the same 48 -- one wave, half the
        ranks, the other half free for the gaps, user 2026-10-09), at least
        LINE_BASE_MIN, each on its nearest point of 2^-LINE_LMAX."""
        n = max(int(LINE_BASE) if int(LINE_BASE) > 0 else self.size // 2, LINE_BASE_MIN)
        return sorted(set(snap(wlo + j * (whi - wlo) / (n - 1), LINE_LMAX) for j in range(n)))

    def _line(self, eps, wlo, whi, reach, eps_int, R):
        """The line's waves (module doc), a generator of ('centre', -eps,
        theta, -eps - reach) requests for _drive; returns the line's result."""
        hmin = 2.0 ** -LINE_LMAX if eps_int is None else float(eps_int)
        key = round(float(eps), 12)
        rlo = -eps - reach
        self._sre_line.add(float(-eps))                    # the ledger: a centre at Re = -eps is the line's
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
            cent = self._fill(done, gaps, hmin)
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
        seen, cand = [], []
        for th in sorted(done):
            c = done[th]
            for l_ in c['lams']:
                if all(abs(l_ - q) > 1e-8 * max(1.0, abs(l_)) for q in seen):
                    seen.append(l_)
            cand += [dict(rw, sre=c['sre'], theta=th, rank=c['rank']) for rw in c['rows']]
        rows, ndup, old = self._rows_merged(cand)
        rows.sort(key=lambda r_: -r_['lam'].real)
        alpha = max((l_.real for l_ in seen), default=-np.inf)
        if witness is not None:
            alpha = max(alpha, witness.real)
        out = dict(clear=bool(clear), alpha=float(alpha), witness=witness, rows=rows, seen=seen,
                   nshift=nshift, waves=waves, uncovered=uncovered, rows_dup=int(ndup))
        if ROW_MERGE > 1e-8:
            # the merge on: the strip's _merge reads the line's rows of the 1e-8 rule,
            # so that it counts (and chooses among) every copy once, even when the line
            # came from its memo and R has none of its centres
            out['rows_old'] = sorted(old, key=lambda r_: -r_['lam'].real)
        return out

    @staticmethod
    def _rows_merged(cand):
        """One stability row per eigenvalue among the candidate rows `cand` (each
        with the centre sre + i theta that found it).  ROW_MERGE <= 1e-8 (the
        default): the rule before -- eigenvalues within 1e-8 x max(1, |lambda|)
        are one, the first copy in `cand`'s order -- and the copies that rule
        keeps (within ROW_DUP of a row from a NEARER shift: 2e-8 .. 1.5e-7 apart
        at 151.48, strip-ab.md AB3) tagged 'dup' and counted, not merged (the
        review of stage C: the merge stays off until a replay shows an effect).
        ROW_MERGE > 1e-8 (opt-in): eigenvalues within ROW_MERGE are one, the copy
        from the nearest shift kept -- its eigenvalue and gradient are the most
        accurate.  The same row computed twice (one centre's, in the line's
        result and again in R) is not a copy.  Returns (rows, copies, the rows of
        the 1e-8 rule): the copies the 1e-8 rule keeps beyond one per eigenvalue,
        merged (ROW_MERGE > 1e-8) or tagged; the count is over `cand` as a whole,
        so a caller passes every candidate once (_merge: the line's 1e-8 rows,
        not its merged ones -- the review of stage C: the line's copies were
        counted twice in the strip arm)."""
        uniq, ks = [], set()
        for rw in cand:
            k_ = (float(rw['sre']), float(rw['theta']), complex(rw['lam']))
            if k_ not in ks:
                ks.add(k_); uniq.append(rw)
        def keep(rws, order, tol):
            rows = []
            for i in order:
                rw = rws[i]
                if all(abs(rw['lam'] - q['lam']) > tol * max(1.0, abs(rw['lam'])) for q in rows):
                    rows.append(rw)
            return rows

        def near(rws):
            return sorted(range(len(rws)), key=lambda i: (abs(complex(rws[i]['lam']) -
                                                               complex(rws[i]['sre'], rws[i]['theta'])), i))
        old = keep(uniq, range(len(uniq)), 1e-8)
        if ROW_MERGE <= 1e-8:
            one = set(id(rw) for rw in keep(old, near(old), float(ROW_DUP)))     # one per eigenvalue, the nearest's
            rows = [dict(rw, dup=id(rw) not in one) for rw in old]              # the same rows, the copies tagged
            return rows, len(old) - len(one), old
        rows = keep(uniq, near(uniq), float(ROW_MERGE))
        return rows, len(old) - len(rows), old

    # --------------------------------------------------------------- level set
    def level(self, z, gam, wlo, whi, watch=None):
        """Is sigma_1 < gam on all of [wlo, whi]?  The level-set certificate of
        certify_sparse.certify_greater on the lattice (in-solve missed peaks;
        user, 2026-10-09: the 321-point sweep saw a peak only where it rose above
        a grid point).  Returns dict(clear, cross, shifts, waves, uncovered,
        cg_max, pencil, watched, outcome): cross lists (w, sigma_1) for each
        region above gam found that no followed row watches (a top estimate;
        empty when clear); watched the regions a followed row does (_level).
        watch = (tol, [(w, sigma_1)]): the followed rows' tops and values at
        this design."""
        dz = z if isinstance(z, dict) else self.design(z)
        lk = (dz['key'], float(gam), float(wlo), float(whi))
        if self._bg is not None:
            # level_prev is the running call's to set (stage D: the look-ahead never asks for a level set)
            raise RuntimeError("lattice: level() while a level set runs beside the look-ahead")
        R = {}
        done, _ = self._drive(dz, {0: self._level(*lk[1:], R, watch=watch)}, 'level', R)
        return done[0]

    def _level(self, gam, wlo, whi, R, watch=None):
        """The level set's waves, as the line's (_line): the centres that
        covered the band at the last call (their factors held) or, at the first
        call, the base grid (LINE_BASE equispaced axis points), then centres in the
        gaps, until the band is covered -- sigma_1 < gam everywhere, since every
        disk is centred at a point below gam (a centre at or above gam has none)
        and holds no crossing -- or a region above gam is found that no followed
        row watches.  A region is seen directly (centres with sigma_1 >= gam) or
        through its crossings: a pencil eigenvalue on the axis is a crossing
        sigma_j = gam at Im lambda, confirmed by sigma_1 there (one wave, exact
        points); an unconfirmed one (an eigenvalue just off the axis) is passed
        over, its neighbourhood then left to the finest level.  In w order, an
        upward crossing or a centre above gam opens a region, a downward
        crossing or a centre below gam closes it (_regions); its top estimate is
        its highest centre or the cubic Hermite fit on its bracketing pair.

        20261010j (missed-peaks.md P2).  (i) Every wave's crossings are looked at
        before the search stops: it used to stop at the first wave in which a
        centre stood above gam, so a region visible only through its crossings
        (the 4.18 peak of 770632 round 11: crossings at 4.13598 and 4.22189 from
        the pencil disks of centres 3.617-4.894, no centre above) was never
        reported beside it.  (ii) 'Watched' is judged by value, not distance: a
        region is watched only where a followed row within 1e-3 of it (of the
        extent its crossings and centres show; a top estimate from the Hermite
        fit or a centre can sit 0.002-0.02 off the top: 0.13443 for the top at
        0.132094 at gam = 143.70 on the 770632 design) reads at least its top
        estimate x (1 - tol) (watch = (tol, [(w, sigma_1)]), the rows' tops and
        values at this design).  A followed
        member's base, or a row on a lower neighbouring top or on its window's
        edge, watches nothing (BFS.md 8, 20261009b: the distance filter already
        bit once with a fixed member; 770632 r11: the region was dropped while
        the row near it read 143.48 against 151.48).  (iii) When every region of
        a wave is watched, the waves go on over the rest of the band, the
        watched regions left out of the cover with LEVEL_WATCH_PAD on each side
        (the search used to return nothing there, the rest of the band
        unexamined); the result is then 'watched' (clear outside them) or
        'inconclusive'."""
        hmin = 2.0 ** -LINE_LMAX
        wtol, wrows = (float(watch[0]), list(watch[1] or [])) if watch is not None else (0.0, [])
        # the first wave: the last call's cover where there is one (86 centres
        # on the central record: with the base grid as well, 130 tasks for 96
        # ranks, two rounds of 5-8 s pencil disks), else the base grid
        cent = sorted(c for c in self.level_prev if wlo - 1e-12 <= c <= whi + 1e-12) or self.base_grid(wlo, whi)
        done = {}
        waves = 0; nshift = 0; spurious = set(); conf = {}
        cross, watched, excl = [], [], []
        wmax = []                                          # the slowest disk of each wave (LS1: the call's critical path)
        pair = LEVEL_PAIR and len(self._ranks()) >= 2
        while cent and waves < LINE_WAVES:
            waves += 1
            reqs, disk = [], {}
            for th in cent:
                th = float(th)
                if pair and self._lvl_tier.get(th, 0) != 0:
                    # its last disk needed the second tier (or neither converged): both at once (LEVEL_PAIR)
                    disk[th] = (('levelp', th, gam, 0), ('levelp', th, gam, 1))
                else:
                    disk[th] = (('level', th, gam),)
                reqs += list(disk[th])
            yield reqs
            for th, qs in disk.items():
                done[th] = R[qs[0]] if len(qs) == 1 else self._level_pair(R[qs[0]], R[qs[1]])
                c_ = done[th]
                if len(qs) == 2:
                    # the ledger (report_lines): pairs, those the second tier settled and the first tier's
                    # seconds there (the wall the pair saved, less a factorisation and a norm disk), those the
                    # first settled (the second's work wasted, or not sent: asynchronous path)
                    a_, b_ = R[qs[0]], R[qs[1]]
                    ps_ = self._pstat
                    ps_[0] += 1
                    if not a_.get('pencil') or a_.get('tier') == 0 or b_.get('skipped'):
                        ps_[3] += 1; ps_[4] += int(bool(b_.get('skipped'))); ps_[5] += float(b_.get('dt_', 0.0))
                    elif a_.get('sib_overlap', True):          # (synchronous waves: the halves on two ranks at once)
                        ps_[1] += 1; ps_[2] += float(a_.get('dt_', 0.0))
                    else:
                        ps_[6] += 1                             # settled by the second tier, the halves in turn: no saving
                self._lvl_tier.pop(th, None)
                self._lvl_tier[th] = (int(c_['tier']) if c_.get('pencil') and c_.get('tier') is not None else 0)
            while len(self._lvl_tier) > 4096:
                self._lvl_tier.pop(next(iter(self._lvl_tier)))
            wmax.append(max(float(R[q].get('dt_', 0.0)) for q in reqs if q in R))
            nshift += len(cent)
            on = []
            known = list(conf) + list(spurious)
            for w_ in sorted(float(l_.imag) for c in done.values() for l_ in c['lams']
                             if abs(l_.real) <= LEVEL_IMAG * max(1.0, abs(l_)) and wlo - 1e-12 <= l_.imag <= whi + 1e-12):
                # one crossing seen from several centres: its Im agrees to ~1e-10, not to
                # the digit (the SISO record 641.71 gave the region at 1.581 three times)
                if (not on or w_ - on[-1] > 1e-8 * max(1.0, abs(w_))) and \
                        all(abs(w_ - q_) > 1e-8 * max(1.0, abs(w_)) for q_ in known):
                    on.append(w_)
            if on:
                yield [('slopex', w_) for w_ in on]
                for w_ in on:
                    if R[('slopex', w_)]['sigma'] >= gam * (1.0 - LEVEL_CONFIRM):
                        conf[w_] = R[('slopex', w_)]
                    else:
                        spurious.add(w_)
            cross, watched = [], []
            for rg in self._regions(done, conf, gam):
                by = [(w_, v_) for w_, v_ in wrows if rg['lo'] - 1e-3 <= w_ <= rg['hi'] + 1e-3 and v_ >= rg['v'] * (1.0 - wtol)]
                if by:
                    watched.append(dict(rg, by=max(by, key=lambda q_: q_[1])))
                else:
                    cross.append((rg['w'], rg['v']))
            if cross:
                break
            excl = [(rg['lo'] - LEVEL_WATCH_PAD, rg['hi'] + LEVEL_WATCH_PAD) for rg in watched]
            cent = self._fill(done, self._minus(self._gaps(done, wlo, whi), excl), hmin)
            if self.verbose:
                print(f"      level wave {waves}: {nshift} shifts so far, {len(cent)} new centre(s)"
                      + (f", {len(watched)} watched region(s) left out" if watched else ''), flush=True)
        # one top per region: estimates within 1e-3 of a higher one are the same
        # region (the members' own spacing in missed_peaks)
        kept = []
        for w_, v_ in sorted(cross, key=lambda q_: -q_[1]):
            if all(abs(w_ - k_[0]) > 1e-3 for k_ in kept):
                kept.append((w_, v_))
        cross = kept
        gaps = self._minus(self._gaps(done, wlo, whi), excl if not cross else [])
        uncovered = float(sum(b_ - a_ for a_, b_ in gaps))
        if not cross:
            self.level_prev = sorted(self._cover(done, wlo, whi))
        clear = bool(not cross and uncovered <= 0.0)
        outcome = 'region' if cross else ('inconclusive' if not clear else ('watched' if watched else 'clear'))
        return dict(clear=clear, cross=cross, shifts=nshift, waves=waves,
                    uncovered=uncovered, cg_max=max((c['cg_max'] for c in done.values()), default=0),
                    pencil=int(sum(c['pencil'] for c in done.values())), outcome=outcome,
                    watched=[dict(w=float(r_['w']), v=float(r_['v']), lo=float(r_['lo']), hi=float(r_['hi']),
                                  by_w=float(r_['by'][0]), by_v=float(r_['by'][1])) for r_ in watched],
                    disks=self._disk_log(done, wmax))

    @staticmethod
    def _level_pair(a, b):
        """A paired level disk (LEVEL_PAIR): a the first tier's result, b the
        second's (None: not computed -- the asynchronous path drops the second
        tier while still unsent once the first has converged).  The first's where
        it is final (no pencil: the norm disk; or its tier converged), else the
        second's with the counts of both, as _level_disk gives in turn: the
        applications, the CG iterations (sum, count, largest).  Its 'dt_' is the
        longer of the two (the disk's wall), 'paired' marks it."""
        if not a.get('pencil') or a.get('tier') == 0 or b is None or b.get('skipped'):
            return dict(a, paired=True)
        return dict(b, napp=int(a['napp']) + int(b['napp']), cg_max=max(int(a['cg_max']), int(b['cg_max'])),
                    cg_sum=int(a['cg_sum']) + int(b['cg_sum']), cg_n=int(a['cg_n']) + int(b['cg_n']),
                    dt_=max(float(a.get('dt_', 0.0)), float(b.get('dt_', 0.0))), paired=True)

    @staticmethod
    def _disk_log(done, wmax):
        """The level-set call's disk costs (20261010j stage C, level-set-waves.md
        LS1): per path -- 'norm' (the norm disk alone), 'tier k' (the pencil's
        Arnoldi converged at LEVEL_ARNOLDI[k]), 'failed' (no tier converged: the
        norm disk's radius only) -- [disks, seconds, longest]; the applications
        and CG iterations; the critical path (each wave's slowest disk) against the
        work; the three slowest disks [(theta, s, path, applications)]."""
        paths = {}
        app_max = cg_sum = cg_n = cg_max = 0
        slow = []
        for th, c in done.items():
            dt = float(c.get('dt_', 0.0))
            tr = c.get('tier')
            p = 'norm' if not c.get('pencil') else ('failed' if tr is None or tr < 0 else f"tier {tr}")
            a = paths.setdefault(p, [0, 0.0, 0.0])
            a[0] += 1; a[1] += dt; a[2] = max(a[2], dt)
            app_max = max(app_max, int(c.get('napp', 0)))
            cg_sum += int(c.get('cg_sum', 0)); cg_n += int(c.get('cg_n', 0)); cg_max = max(cg_max, int(c.get('cg_max', 0)))
            slow = sorted(slow + [(float(th), dt, p, int(c.get('napp', 0)))], key=lambda q_: -q_[1])[:3]
        return dict(paths=paths, app_max=app_max, cg_sum=cg_sum, cg_n=cg_n, cg_max=cg_max,
                    crit=float(sum(wmax)), work=float(sum(a[1] for a in paths.values())), slow=slow)

    @staticmethod
    def _regions(done, conf, gam):
        """The regions above gam the level set's waves show so far (_level):
        done its centres (sigma_1 each), conf its confirmed crossings (w ->
        sigma_1 and slope there).  In w order an upward crossing (slope > 0) or a
        centre above gam opens a region, a downward crossing or a centre below
        gam closes it.  Each: dict(w, v) the top estimate (the highest centre,
        or the cubic Hermite fit on the bracketing pair, whichever is higher; a
        lone crossing its own point), (lo, hi) the extent seen."""
        pts = sorted([(float(th), 0, c) for th, c in done.items()] + [(float(w_), 1, e_) for w_, e_ in conf.items()],
                     key=lambda p_: (p_[0], p_[1]))
        out, cur = [], None
        for w_, kind, o in pts:
            if kind == 0:
                if o['sigma'] >= gam:
                    if cur is None:
                        cur = dict(lo=w_, hi=w_, up=None, dn=None, cen=[])
                    cur['cen'].append((w_, float(o['sigma']))); cur['hi'] = w_
                elif cur is not None:
                    out.append(cur); cur = None
            elif o['gprime'] > 0.0:
                if cur is not None:
                    out.append(cur)
                cur = dict(lo=w_, hi=w_, up=(w_, o), dn=None, cen=[])
            else:
                if cur is None:
                    cur = dict(lo=w_, hi=w_, up=None, dn=None, cen=[])
                cur['dn'] = (w_, o); cur['hi'] = w_
                out.append(cur); cur = None
        if cur is not None:
            out.append(cur)
        res = []
        for rg in out:
            cands = list(rg['cen'])
            if rg['up'] is not None and rg['dn'] is not None:
                (wa, ea), (wb, eb) = rg['up'], rg['dn']
                wt, ft, _ = _hermite_top(wa, wb, ea['sigma'], eb['sigma'], ea['gprime'], eb['gprime'])
                cands.append((float(wt), float(max(ft, ea['sigma'], eb['sigma']))))
            elif not cands:
                q_ = rg['up'] if rg['up'] is not None else rg['dn']
                cands.append((float(q_[0]), float(q_[1]['sigma'])))
            w_, v_ = max(cands, key=lambda c_: c_[1])
            res.append(dict(w=w_, v=v_, lo=float(rg['lo']), hi=float(rg['hi'])))
        return res

    @staticmethod
    def _minus(gaps, excl):
        """The gaps of a cover without the intervals excl (the watched regions)."""
        out = list(gaps)
        for a_, b_ in excl:
            nxt = []
            for c_, d_ in out:
                if d_ <= a_ or c_ >= b_:
                    nxt.append((c_, d_))
                    continue
                if c_ < a_:
                    nxt.append((c_, a_))
                if d_ > b_:
                    nxt.append((b_, d_))
            out = nxt
        return out

    # ------------------------------------------------------------------- strip
    def _strip(self, eps, wlo, whi, d, R):
        """The strip [-eps - d, -eps] x [wlo, whi], covered (evaluate(strip=True):
        a DIAGNOSTIC since 2026-10-16 -- tools/check_lattice.py, the closed-loop
        eigenvalues left of the line for a stationarity measure; from 2026-10-09
        to 20261010n the strip arm covered it at every iterate the QP read, and
        the 20261010n A/B removed that arm: BFS.md 8), a generator of ('centre',
        sre, theta, -eps - d) requests for _drive beside the line's.  Every
        eigenvalue in the strip is then exhibited, and each gets its stability
        row: the line's centres see only the eigenvalues nearest to some point
        of the line, and one behind another (-0.1444 behind -0.0723 on the
        central record 191.11) has no row there.

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
        # the line's rows of the 1e-8 rule (its own rows with the merge off: the rule
        # before exactly), then every centre's of the evaluation
        seen, cand = list(lr['seen']), list(lr.get('rows_old') or lr['rows'])
        for q in sorted(k for k in R if k[0] == 'centre'):
            o = R[q]
            for l_ in o['lams']:
                if all(abs(l_ - e_) > 1e-8 * max(1.0, abs(l_)) for e_ in seen):
                    seen.append(l_)
            cand += [dict(rw, sre=o['sre'], theta=o['theta'], rank=o['rank']) for rw in o['rows']]
        rows, ndup, old = LatticeOracle._rows_merged(cand)     # one row per eigenvalue ('seen' keeps 1e-8)
        rows.sort(key=lambda r_: -r_['lam'].real)
        # alpha stays the line's: a trial (line only) and its iterate (line and
        # strip) must read the same value for the same design.  The copies counted
        # once, over the line's and the strip's candidates together (the review of
        # stage C: the line's count added on top counted its copies twice)
        out = dict(lr, rows=rows, seen=seen, strip=st, rows_dup=int(ndup))
        if 'rows_old' in lr:
            out['rows_old'] = sorted(old, key=lambda r_: -r_['lam'].real)
        return out

    @staticmethod
    def _fill(done, gaps, hmin):
        """The next centres in the gaps of a cover of an interval (the line's or
        the level set's; done: centre -> result with its free radius 'rad')."""
        cent = []
        for a_, b_ in gaps:
            L = b_ - a_
            if L < hmin:
                # a sliver between two disks that nearly touch (1.2e-6 wide on the
                # central record's level set): the finest lattice points next to
                # it, inside a neighbouring disk -- their own disks cover it
                c, h = snap(0.5 * (a_ + b_), LINE_LMAX), 2.0 ** -LINE_LMAX
                cent += [c2 for c2 in (c, c - h, c + h) if c2 not in done][:1]
                continue
            # as many centres as the neighbouring disks' radii suggest (one when
            # they are large: bisection), evenly over the gap, each on the lattice
            # point nearest its place at the first level finer than a quarter of
            # the spacing
            ra = [c['rad'] for th, c in done.items() if c['rad'] > 0 and
                  (abs(th + LINE_SHRINK * c['rad'] - a_) < 1e-12 or abs(th - LINE_SHRINK * c['rad'] - b_) < 1e-12)]
            r_est = min(ra) if ra else L
            nc = int(min(32, max(1, np.ceil(L / (2.0 * LINE_SHRINK * r_est)))))
            sp = L / nc
            lev = min(LINE_LMAX, int(np.ceil(np.log2(4.0 / sp))))
            for j in range(nc):
                c = snap(a_ + (j + 0.5) * sp, lev)
                if c in done:
                    # a centre whose eigensolve failed (no disk) sits there: the
                    # quarter points, one level finer
                    for c2 in (snap(c - 0.25 * sp, lev + 1), snap(c + 0.25 * sp, lev + 1)):
                        if a_ < c2 < b_ and c2 not in done and lev + 1 <= LINE_LMAX:
                            cent.append(c2)
                elif a_ < c < b_:
                    cent.append(c)
        return sorted(set(cent))

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
        self._vec_hold.clear()
        return out

    def drop(self):
        if self._bg is not None:
            raise RuntimeError("lattice: drop() while a level set runs beside the look-ahead")
        self._run(drop_task, {r: None for r in self._ranks()}, 'drop')
        self.held.clear(); self.line_prev.clear(); self.strip_prev.clear(); self.level_prev = []; self.tops.clear(); self._line_memo.clear()
        self._req_memo.clear()           # the row requests too (the review of stage C: a plant change would serve stale rows)
