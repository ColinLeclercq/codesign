"""
parallel.py — process-parallel oracle blocks for the co-design loop.

Two blocks dominate an oracle call and both are embarrassingly parallel
(measured at ne = 200, per call):

    alpha shift loop            3.463 s  over 16 INDEPENDENT shifts   <- 90%
    |Omega| frequency constraints 0.374 s  over |Omega| INDEPENDENT solves

Design notes, all of them learned the hard way in this project:

  * PROCESSES, not threads.  Each worker pins OMP/MKL to one thread: the
    parallelism is across frequencies, and nesting a threaded BLAS inside it is
    what produced SIGSEGV in every case (DIAGNOSE_segfault.md).
  * SuperLU objects are NOT picklable, so factorisations never cross a process
    boundary.  A persistent pool builds one FECoDesign per worker in its
    initialiser and keeps its own LU cache; tasks carry only (omega, z).
  * MEMORY, not core count, is the limit at cavity scale: each worker holds its
    own factorisation, of order 1 GB on a 2-D mesh.  Size the pool accordingly
    (`nproc`), and leave lu_cache_max at 0 there.

Usage:

    from parallel import ParallelOracle
    po = ParallelOracle(ne=200, r=7, sob_len=0.8, case='K+FB+FC',
                        packer_args=pk_args, nproc=16)
    sig, gz = po.constraints(z, omegas)      # arrays (m,), (m, NZ)
    al, lam = po.alpha(z)                    # spectral abscissa, eigenvalue
    po.close()
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np

_W = {}          # per-worker state


def _init(ne, r, sob_len, case, pk_args, lu_cache_max, flags_extra=None):
    """Runs ONCE per worker: build the plant and a Packer identical to the
    parent's, so that gradients come back already in packed coordinates.
    `flags_extra` carries run-level flags that parse_case cannot know (the
    symmetric-subspace mode): the worker's Packer MUST match the parent's
    layout, or unpack/grad silently misread every design vector."""
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[v] = '1'
    import fe_codesign
    from fe_codesign import parse_case
    from packer import Packer
    D = fe_codesign.FECoDesign(ne=ne, r=r, sob_len=sob_len)
    D.lu_cache_max = lu_cache_max
    flags = parse_case(case)
    if flags_extra:
        flags.update(flags_extra)
    _W['D'] = D
    _W['flags'] = flags
    _W['pk'] = Packer(D, flags, *pk_args)


def _unpack(z):
    D, pk, flags = _W['D'], _W['pk'], _W['flags']
    arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
    b, c, pa, ps, na, ns = D.bc(xa, xs, va, vs)
    return D, pk, flags, arow, nvec, K, xa, xs, b, c, pa, ps, na, ns


def _task_gain(args):
    """sigma_1(theta, omega) and its gradient in PACKED coordinates."""
    w, z = args
    D, pk, flags, arow, nvec, K, xa, xs, b, c, pa, ps, na, ns = _unpack(z)
    cl = D.cl(b, c)
    res = cl.gain(float(w), arow, nvec, K)
    gr = D.grads(cl, res['x'], res['z'], arow, nvec, K, xa, xs,
                 pa, ps, na, ns, flags)
    return float(res['sigma']), pk.grad(gr)


def _task_shift(args):
    """Candidate eigenvalues near one shift -- NO eigenvectors, so the payload
    stays tiny.  The parent picks the rightmost and recovers its vectors once."""
    sig, z = args
    D, pk, flags, arow, nvec, K, xa, xs, b, c, pa, ps, na, ns = _unpack(z)
    cl = D.cl(b, c)
    lam, dist = cl.nearest_eig(complex(sig), arow, nvec, K, k=6)
    return (complex(lam) if lam is not None else None)


class ParallelOracle:
    @staticmethod
    def available_cores():
        """Cores THIS PROCESS may use.

        `multiprocessing.cpu_count()` reports the machine (96 on an ONERA node)
        regardless of the SLURM allocation, so using it would oversubscribe a
        job that asked for fewer.  SLURM_CPUS_PER_TASK, then the CPU affinity
        mask, are the authoritative values.
        """
        v = os.environ.get('SLURM_CPUS_PER_TASK')
        if v and v.isdigit() and int(v) > 0:
            return int(v)
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            import multiprocessing as mp
            return mp.cpu_count()

    def __init__(self, ne, r, sob_len, case, packer_args, nproc=None,
                 lu_cache_max=2, flags_extra=None):
        import multiprocessing as mp
        self.nproc = int(nproc or self.available_cores())
        ctx = mp.get_context('fork')
        self.pool = ctx.Pool(
            processes=self.nproc, initializer=_init,
            initargs=(ne, r, sob_len, case, packer_args, lu_cache_max,
                      flags_extra))
        # the parent keeps a serial copy for the cheap, sequential steps
        _init(ne, r, sob_len, case, packer_args, lu_cache_max, flags_extra)
        self.D, self.pk, self.flags = _W['D'], _W['pk'], _W['flags']

    # ---------------- the |Omega| epigraph constraints ----------------
    def constraints(self, z, omegas):
        out = self.pool.map(_task_gain, [(w, z) for w in omegas],
                            chunksize=1)
        sig = np.array([o[0] for o in out])
        G = np.array([o[1] for o in out])
        return sig, G

    # ---------------- the spectral abscissa ----------------
    def alpha(self, z, shifts=None):
        """Parallel over shifts; the rightmost eigenvalue and its eigenvectors
        are then recovered SERIALLY at the winning shift, which is one cheap
        extra solve."""
        D, pk, flags = self.D, self.pk, self.flags
        arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
        b, c, *_ = D.bc(xa, xs, va, vs)
        cl = D.cl(b, c)
        if shifts is None:
            shifts = cl.default_shifts(arow)
        lams = self.pool.map(_task_shift, [(s, z) for s in shifts],
                             chunksize=1)
        lams = [l for l in lams if l is not None]
        if not lams:
            return cl.alpha_krylov(arow, nvec, K)      # fall back, serial
        lam = max(lams, key=lambda l: l.real)
        # one serial pass at a shift next to the winner recovers v and u
        al, lamb, v, u = cl.alpha_krylov(arow, nvec, K,
                                        shifts=[complex(lam.real + 0.05,
                                                        lam.imag)])
        if lamb is None or lamb.real < lam.real - 1e-8:
            al, lamb, v, u = cl.alpha_krylov(arow, nvec, K)   # disagree: serial
        return al, lamb, v, u, cl

    def close(self):
        try:
            self.pool.close(); self.pool.join()
        except Exception:
            pass
