"""mpipool.py -- the full-model pool as an MPI task farm (BFS.md 3).

On GL the full-model tasks run in a FORKED multiprocessing pool (fullpar.py):
the workers inherit the plant by fork.  With PETSc/MUMPS (MPI-initialised
libraries) forking is unsafe, so on the cluster the pool is MPI ranks instead:

    srun -n 96 python run_case_epi.py --plant bfs ...      (or mpirun)

    rank 0       the driver (optimiser, reduced model, bookkeeping)
    ranks 1..    workers: each loads the operators from the --ops file ONCE,
                 then serves tasks; every task is a module-level function of
                 picklable arguments (fullpar's tasks already are), run with
                 the worker's own sequential MUMPS factorisations

fullpar.start() calls MPIPool.set_context() instead of forking: the context
(the plant, as a recipe -- the ops file, the order and its scalar settings --
and any arrays such as the reduced oracle's input bases) and a snapshot of the
module-level settings (UPPERCASE globals of the modules below) go to every
worker, so a task computes exactly what it would in the driver's process.

    MPIPool.map(fn, args)    in order, dynamic scheduling (one task at a time)
    MPIPool.each(fn, {rank: arg}) / collect(ranks)
                             one task to each named rank, gathered later (the
                             driver computes its own share in between)
    MPIPool.post(rank, fn, arg, tid) / recv_any()
                             one task to an IDLE rank, its result tagged tid,
                             gathered one at a time from whichever rank is done
                             (the lattice's asynchronous level set, 20261010j
                             stage D: each rank released as its task returns);
                             map, set_context, each and collect refuse to run
                             while such a task is out (they would block on, or
                             steal the result of, a busy rank)

Workers never return from serve(); rank 0 sends a stop message at exit.
"""
from __future__ import annotations
import atexit
import importlib
import numpy as np

SYNC_MODULES = ('sparse_lu', 'closedloop_sparse', 'certify_sparse', 'sparse_spectrum', 'winding_cert',
                'reduced', 'massops', 'fullpar', 'peakfollow', 'lattice')
_TAG_TASK, _TAG_RES, _TAG_CTX, _TAG_STOP = 1, 2, 3, 4
_SIMPLE = (str, int, float, bool, type(None), complex)


def comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


_LAUNCHER_SIZE = ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'SLURM_STEP_NUM_TASKS')


def active():
    """True when this process is part of an MPI job with workers.  Decided
    from the launcher's environment FIRST: importing mpi4py initialises MPI,
    and a singleton MPI_Init outside a job allocation aborts on the cluster
    (check_env.py's notes) -- a GL run must never pay for that."""
    import os
    if not any(int(os.environ.get(k, '1') or 1) > 1 for k in _LAUNCHER_SIZE):
        return False
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_size() > 1
    except Exception:
        return False


def _simple(v):
    if isinstance(v, _SIMPLE):
        return True
    if isinstance(v, (tuple, list)) and len(v) < 64:
        return all(_simple(x) for x in v)
    return False


def module_snapshot():
    out = {}
    for m in SYNC_MODULES:
        try:
            mod = importlib.import_module(m)
        except Exception:
            continue
        out[m] = {k: v for k, v in vars(mod).items() if k.isupper() and not k.startswith('_') and _simple(v)}
    return out


def plant_recipe(D):
    """What a worker needs to rebuild the driver's plant: the ops file, the
    order, and the scalar attributes the driver set."""
    attrs = {k: v for k, v in vars(D).items() if not k.startswith('_') and _simple(v)}
    return dict(kind=type(D).__name__, ops=getattr(D, 'ops_path', None), r=D.r, mc=D.mc, attrs=attrs)


class MPIPool:
    def __init__(self):
        self.comm = comm()
        self.size = self.comm.Get_size()
        self.nworkers = self.size - 1
        atexit.register(self.stop)
        self._stopped = False
        self._posted = 0                 # post()ed tasks whose results recv_any() has not returned yet

    def _idle(self, what):
        """The blocking calls below assume no post()ed task is out: map and
        set_context would wait on (send to) a rank still busy with it, collect
        would take its result for theirs (20261010j stage D)."""
        if self._posted:
            raise RuntimeError(f"MPIPool.{what} with {self._posted} asynchronous task(s) still out "
                               f"(post without recv_any: the lattice's level set must be joined first)")

    def set_context(self, ctx):
        """Send the context (plant recipe + arrays) and the module settings to
        every worker (blocking: each acknowledges)."""
        self._idle('set_context')
        msg = {}
        for k, v in ctx.items():
            msg[k] = ('__plant__', plant_recipe(v)) if k == 'D' and v is not None else v
        payload = dict(ctx=msg, modules=module_snapshot())
        for w in range(1, self.size):
            self.comm.send(payload, dest=w, tag=_TAG_CTX)
        for w in range(1, self.size):
            self.comm.recv(source=w, tag=_TAG_CTX)

    def map(self, fn, args, chunksize=1):
        from mpi4py import MPI
        self._idle('map')
        args = list(args)
        out = [None] * len(args)
        nxt, busy = 0, 0
        st = MPI.Status()
        for w in range(1, self.size):
            if nxt >= len(args):
                break
            self.comm.send((nxt, fn, args[nxt]), dest=w, tag=_TAG_TASK); nxt += 1; busy += 1
        while busy:
            i, ok, res = self.comm.recv(source=MPI.ANY_SOURCE, tag=_TAG_RES, status=st)
            busy -= 1
            if not ok:
                raise RuntimeError(f"MPI worker {st.Get_source()} failed on task {i}: {res}")
            out[i] = res
            if nxt < len(args):
                self.comm.send((nxt, fn, args[nxt]), dest=st.Get_source(), tag=_TAG_TASK); nxt += 1; busy += 1
        return out

    def each(self, fn, args):
        """fn(args[w]) on worker rank w for every w of the dict args, WITHOUT
        waiting: the driver works meanwhile, then collect() gathers (the
        reduced model's rows, reduced._MPIRowPool: a frequency always goes to
        the same rank, which keeps its warm starts)."""
        self._idle('each')
        for w, a in args.items():
            self.comm.send((w, fn, a), dest=w, tag=_TAG_TASK)
        return list(args)

    def collect(self, ranks):
        """The results of each(ranks), {rank: result}; every one is received
        before an error is raised, so no stale result reaches a later map."""
        from mpi4py import MPI
        self._idle('collect')
        out, err, st = {}, None, MPI.Status()
        for _ in ranks:
            i, ok, res = self.comm.recv(source=MPI.ANY_SOURCE, tag=_TAG_RES, status=st)
            if ok:
                out[i] = res
            elif err is None:
                err = f"MPI worker {st.Get_source()} failed: {res}"
        if err is not None:
            raise RuntimeError(err)
        return out

    def post(self, w, fn, a, tid):
        """fn(a) on worker rank w WITHOUT waiting, its result tagged tid
        (recv_any).  The caller posts only to a rank with no task out: a send
        to a busy rank could block until that rank's task ends (a large
        message waits for its receive), and the driver would stop serving the
        others (20261010j stage D, level-set-waves.md LS3 verifier)."""
        self.comm.send((tid, fn, a), dest=w, tag=_TAG_TASK)
        self._posted += 1

    def recv_any(self):
        """The next result of a post()ed task, from whichever rank finishes
        first: (rank, tid, ok, result); ok False: result is the worker's error."""
        from mpi4py import MPI
        st = MPI.Status()
        i, ok, res = self.comm.recv(source=MPI.ANY_SOURCE, tag=_TAG_RES, status=st)
        self._posted -= 1
        return st.Get_source(), i, ok, res

    def close(self):
        pass

    def join(self):
        pass

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        for w in range(1, self.size):
            try:
                self.comm.send(None, dest=w, tag=_TAG_STOP)
            except Exception:
                pass


def _build_plant(rec, cache):
    key = (rec['kind'], rec['ops'], rec['r'])
    D = cache.get(key)
    if D is None:
        if rec['kind'] != 'BFSPlant':
            raise RuntimeError(f"MPI workers build BFS plants only (got {rec['kind']})")
        from bfs.plant import BFSPlant
        import certify_sparse
        D = cache[key] = BFSPlant(rec['ops'], r=rec['r'], mc=rec['mc'])
        certify_sparse.register_plant(D.A, D.M)      # tasks carry a tag for the plant, not A and M
    for k, v in rec['attrs'].items():
        try:
            setattr(D, k, v)
        except Exception:
            pass
    return D


def serve():
    """Worker loop (ranks > 0): never returns normally."""
    from mpi4py import MPI
    import fullpar
    c = comm(); st = MPI.Status(); plants = {}
    while True:
        msg = c.recv(source=0, tag=MPI.ANY_TAG, status=st)
        tag = st.Get_tag()
        if tag == _TAG_STOP:
            break
        if tag == _TAG_CTX:
            for m, attrs in msg['modules'].items():
                try:
                    mod = importlib.import_module(m)
                    for k, v in attrs.items():
                        setattr(mod, k, v)
                except Exception:
                    pass
            fullpar.CTX.clear()
            for k, v in msg['ctx'].items():
                fullpar.CTX[k] = _build_plant(v[1], plants) if (isinstance(v, tuple) and len(v) == 2
                                                                and v[0] == '__plant__') else v
            c.send(True, dest=0, tag=_TAG_CTX)
            continue
        i, fn, arg = msg
        try:
            res, ok = fn(arg), True
        except Exception as e:
            import traceback
            res, ok = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}", False
        c.send((i, ok, res), dest=0, tag=_TAG_RES)
    import sys
    sys.exit(0)
