"""fullpar.py -- one process pool for the FULL-MODEL tasks (STAGE6 6.30).

The full-model work of a round is independent factorisations at different
points: the winding count's segments, the certificate's decision shifts (one
generation of critical intervals at a time) and its location shifts, the full
sigma samples (peak reference, start sweep, candidates), the enrichment anchors
and eigen-anchors, the band's refinement shifts, the plant heights.  At the
backward-facing step each is a MUMPS-sized factorisation (seconds, ~1.5 GB),
so they go to forked workers:

    fullpar.start(nproc, D=D, ...)   fork (or re-fork) the workers; everything
                                     passed as context (the plant, the mesh
                                     object, the input bases) is inherited by
                                     fork -- nothing large goes through a pipe
    fullpar.pmap(fn, args)           [fn(a) for a in args], in order; in the
                                     workers when the pool is up, else in this
                                     process (the 1-task path IS the serial code)
    fullpar.CTX                      the context, in the workers and here

Every task is a pure function of its arguments (no warm starts, deterministic
ARPACK start vectors), so a result does not depend on which process computed it:
N tasks give the serial results bit for bit (the winding count, whose march is
sequential, is the exception: split into segments its point count differs,
never its count -- winding_cert).

LU accounting: a task run in a worker returns its lu_counter delta, merged here
under the submitting component's site (`<driver line> <- <factorising line>`,
the key format of the serial run).
"""
from __future__ import annotations
import multiprocessing as _mp

import time as _time

POOL = None
NPROC = 1
CTX = {}
STATS = dict(calls=0, tasks=0, work=0.0, wall=0.0)    # pmap calls through the pool (report())
# every pmap call with >= 2 tasks, pooled or not: (number of tasks, task seconds,
# longest task, task function) -- what a pool of W workers would take
# (estimate(W), BFS.md 7), and by task kind (by_task)
CALLS = []
IN_TASK = False          # True while a task runs in THIS process (serial pmap; bfs/mumps_lu.py accounting)


_MPI = None          # bfs/mpipool.MPIPool when the run is an MPI job (backward-facing step, BFS.md 3)


def start(nproc, **ctx):
    """(Re)start the pool with nproc workers holding ctx (fork inheritance).
    nproc <= 1: no pool, every pmap runs in this process.  Under MPI (more
    than one rank) the pool is the other ranks instead (bfs/mpipool.py): the
    context is SENT to them, nproc is the number of ranks."""
    global POOL, NPROC, _STAMP, _MPI
    from bfs import mpipool as _mpp
    if _mpp.active():
        CTX.clear(); CTX.update(ctx)
        if _MPI is None:
            _MPI = _mpp.MPIPool()
        _MPI.set_context(dict(ctx))
        POOL = _MPI
        NPROC = _MPI.size
        _STAMP = _stamp()
        D = CTX.get('D')
        if D is not None and getattr(D, 'cert_pool', None) is not None:
            D.cert_pool = POOL
        return POOL
    stop()
    CTX.clear(); CTX.update(ctx)
    NPROC = max(1, int(nproc or 1))
    _STAMP = _stamp()
    if NPROC > 1:
        POOL = _mp.get_context('fork').Pool(processes=NPROC)
    D = CTX.get('D')
    if D is not None and getattr(D, 'cert_pool', None) is not None:
        D.cert_pool = POOL                   # the abscissa certificate's segments (--cert-nseg) share the pool
    return POOL


def _stamp():
    """What the workers' copy of the mesh object must still match: a change
    here (the driver sets the compensator order after the warm-up fork) makes
    pmap re-fork before running anything."""
    D = CTX.get('D')
    return None if D is None else (id(D.A), id(D.M), getattr(D, 'r', None), getattr(D, 'mc', None))


_STAMP = None


def stop():
    global POOL
    if POOL is not None and POOL is _MPI:
        POOL = None                          # the MPI ranks stay up (they stop at exit)
        return
    if POOL is not None:
        try:
            POOL.close(); POOL.join()
        except Exception:
            pass
    POOL = None


def active():
    return POOL is not None


def _site():
    try:
        import lu_counter as _lc
        if not _lc._installed:
            return None
        s = _lc._caller()
        return s.split(' <- ')[0]
    except Exception:
        return None


def _call(t):
    fn, arg, site = t
    t0 = _time.perf_counter()
    if site is None:
        out = fn(arg)
        return out, None, _time.perf_counter() - t0
    import lu_counter as _lc
    b0 = _lc.snapshot()
    out = fn(arg)
    b1 = _lc.snapshot()
    d = dict(n_lu=b1['n_lu'] - b0['n_lu'], n_lu_full=b1['n_lu_full'] - b0['n_lu_full'],
             n_solve=b1['n_solve'] - b0['n_solve'],
             full={k: v - b0['by_site_full'].get(k, 0) for k, v in b1['by_site_full'].items()
                   if v != b0['by_site_full'].get(k, 0)})
    return out, d, _time.perf_counter() - t0


def _merge(site, d):
    import lu_counter as _lc
    with _lc._lock:
        _lc._n_lu += d['n_lu']; _lc._n_lu_big += d['n_lu_full']; _lc._n_solve += d['n_solve']
        for k, v in d['full'].items():
            _lc._by_site_full[f"{site} <- {k.split(' <- ')[-1]}"] += v
            _lc._by_site_lu[f"{site} <- {k.split(' <- ')[-1]}"] += v


def pmap(fn, args):
    """[fn(a) for a in args] in order; in the workers when the pool is up and
    there are at least two tasks (dynamic scheduling, one task at a time)."""
    global IN_TASK
    args = list(args)
    if POOL is None or len(args) < 2:
        if len(args) < 2:
            return [fn(a) for a in args]
        out, dts = [], []
        IN_TASK = True
        try:
            for a_ in args:
                t0 = _time.perf_counter(); out.append(fn(a_)); dts.append(_time.perf_counter() - t0)
        finally:
            IN_TASK = False
        CALLS.append((len(args), float(sum(dts)), float(max(dts)), getattr(fn, '__name__', '?')))
        return out
    if _stamp() != _STAMP:
        start(NPROC, **dict(CTX))            # the workers' mesh object is stale: re-fork
    site = _site()
    t0 = _time.perf_counter()
    res = POOL.map(_call, [(fn, a, site) for a in args], chunksize=1)
    STATS['wall'] += _time.perf_counter() - t0
    STATS['calls'] += 1; STATS['tasks'] += len(args)
    out = []
    CALLS.append((len(args), float(sum(r_[2] for r_ in res)), float(max(r_[2] for r_ in res)), getattr(fn, '__name__', '?')))
    for r, d, dt in res:
        STATS['work'] += dt
        if d is not None:
            _merge(site, d)
        out.append(r)
    return out


def pqueue(fn, args, more):
    """A dynamic task queue (20261010s, the winding count's split segments):
    fn(a) for every a of args and, for each task's result r, for every a' of
    more(a, r) (its follow-up tasks).  Returns [(a, r)] for every task run, in
    no particular order (the caller sorts).  On the MPI pool each task goes to
    a worker the moment one is free (MPIPool.post / recv_any), a follow-up as
    soon as its parent is in; on a fork pool the tasks run in generations (one
    pmap each: a task's follow-ups in the next); without a pool, here, in turn.
    The tasks run and their results do not depend on the pool or the order:
    more() sees only its own task and result."""
    pend = list(args)
    out = []
    if POOL is None or POOL is not _MPI or getattr(POOL, 'size', 1) < 2 or getattr(POOL, '_posted', 0):
        while pend:
            res = pmap(fn, pend)
            nxt = []
            for a_, r_ in zip(pend, res):
                out.append((a_, r_))
                nxt += list(more(a_, r_))
            pend = nxt
        return out
    if _stamp() != _STAMP:
        start(NPROC, **dict(CTX))            # the workers' context is stale: sent again
    site = _site()
    t0 = _time.perf_counter()
    free = list(range(1, POOL.size))
    fly, seq, dts, err = {}, 0, [], None
    try:
        while pend or fly:
            while pend and free and err is None:
                w = free.pop(0); a_ = pend.pop(0)
                POOL.post(w, _call, (fn, a_, site), seq); fly[seq] = a_; seq += 1
            if not fly:
                break
            src, tid, ok, res = POOL.recv_any()
            a_ = fly.pop(tid); free.append(src)
            if not ok:
                err = err or f"MPI worker {src} failed: {res}"
                continue
            r_, d_, dt_ = res
            dts.append(dt_); STATS['work'] += dt_
            if d_ is not None:
                _merge(site, d_)
            out.append((a_, r_))
            if err is None:
                pend[:0] = list(more(a_, r_))   # follow-ups first: the long chains start early
    finally:
        while fly:                           # an error: every task out received, none left for a later call
            src, tid, _, _ = POOL.recv_any()
            fly.pop(tid, None)
    if err is not None:
        raise RuntimeError(err)
    STATS['wall'] += _time.perf_counter() - t0
    STATS['calls'] += 1; STATS['tasks'] += len(dts)
    if dts:
        CALLS.append((len(dts), float(sum(dts)), float(max(dts)), getattr(fn, '__name__', '?')))
    return out


def estimate(W):
    """Seconds the recorded pmap calls would take on W workers (dynamic
    scheduling, no dispatch cost): per call max(work / W, longest task)."""
    return float(sum(max(w / max(W, 1), m) for _, w, m, _ in CALLS))


def by_task(W=95):
    """One line for the log: per task function, calls, tasks, task time and
    the time on W workers (max(work / W, longest task) per call: at large W
    this is the critical path, which more workers do not shorten)."""
    agg = {}
    for n, w, m, f in CALLS:
        a = agg.setdefault(f, [0, 0, 0.0, 0.0])
        a[0] += 1; a[1] += n; a[2] += w; a[3] += max(w / max(W, 1), m)
    rows = sorted(agg.items(), key=lambda kv: -kv[1][3])
    return "; ".join(f"{f} {c} calls / {n} tasks, {w:.0f} s, ~{e:.0f} s on {W}" for f, (c, n, w, e) in rows)


def report():
    """One line for the log: the pool's work (the tasks' own times, i.e. what
    they would cost serially, less the dispatch) against its wall time."""
    if not STATS['calls']:
        if CALLS:
            return (f"full-model tasks run serially: {sum(c[0] for c in CALLS)} tasks in {len(CALLS)} calls, "
                    f"{sum(c[1] for c in CALLS):.1f} s (on 11 / 23 / 95 workers ~{estimate(11):.1f} / {estimate(23):.1f} / {estimate(95):.1f} s)")
        return "full-model pool: not used"
    return (f"full-model pool ({NPROC} workers): {STATS['tasks']} tasks in {STATS['calls']} calls, "
            f"{STATS['work']:.1f} s of task time in {STATS['wall']:.1f} s wall "
            f"(x{STATS['work'] / max(STATS['wall'], 1e-9):.2f}; on 11 / 23 / 95 workers ~{estimate(11):.1f} / {estimate(23):.1f} / "
            f"{estimate(95):.1f} s)")


# ------------------------------------------------------------ common tasks
def _sigma_task(t):
    """sigma_1 of the full closed loop at the frequencies of one task."""
    b, c, arow, nvec, K, ws = t
    cl = CTX['D'].cl(b, c)
    return [cl.gain(float(w), arow, nvec, K, want_vectors=False)['sigma'] for w in ws]


def sigmas(b, c, arow, nvec, K, ws):
    """Full-model sigma_1 at every w of ws, one frequency per task."""
    out = pmap(_sigma_task, [(b, c, arow, nvec, K, [float(w)]) for w in ws])
    return [o[0] for o in out]
