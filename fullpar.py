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


def start(nproc, **ctx):
    """(Re)start the pool with nproc workers holding ctx (fork inheritance).
    nproc <= 1: no pool, every pmap runs in this process."""
    global POOL, NPROC, _STAMP
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
    args = list(args)
    if POOL is None or len(args) < 2:
        return [fn(a) for a in args]
    if _stamp() != _STAMP:
        start(NPROC, **dict(CTX))            # the workers' mesh object is stale: re-fork
    site = _site()
    t0 = _time.perf_counter()
    res = POOL.map(_call, [(fn, a, site) for a in args], chunksize=1)
    STATS['wall'] += _time.perf_counter() - t0
    STATS['calls'] += 1; STATS['tasks'] += len(args)
    out = []
    for r, d, dt in res:
        STATS['work'] += dt
        if d is not None:
            _merge(site, d)
        out.append(r)
    return out


def report():
    """One line for the log: the pool's work (the tasks' own times, i.e. what
    they would cost serially, less the dispatch) against its wall time."""
    if not STATS['calls']:
        return "full-model pool: not used"
    return (f"full-model pool ({NPROC} workers): {STATS['tasks']} tasks in {STATS['calls']} calls, "
            f"{STATS['work']:.1f} s of task time in {STATS['wall']:.1f} s wall "
            f"(x{STATS['work'] / max(STATS['wall'], 1e-9):.2f})")


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
