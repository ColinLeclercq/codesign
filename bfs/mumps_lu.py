"""mumps_lu.py -- complex sparse LU through PETSc/MUMPS (petsc4py), the
cluster's factorisation (BFS.md 3).

Same interface as SciPy's SuperLU object:
    lu = MumpsLU(S);  x = lu.solve(b, trans='N'|'T'|'H')   (b: (n,) or (n, k))
so sparse_lu.factor() returns it for ORDER = 'mumps' and no caller changes.

Every factorisation is SEQUENTIAL MUMPS on PETSC_COMM_SELF: the parallelism of
the run is many independent factorisations at once (one per worker of the MPI
task farm, bfs/mpipool.py), not one distributed factorisation.  MUMPS's own
threads (OpenMP / BLAS) follow OMP_NUM_THREADS of the worker.

The user's PETSc is built with complex scalars (env.sh: PETSc 3.25.3 complex):
the resolvent pencils i w M - A are complex, so that is the right build; the
real operators are converted on the way in.

Options (environment, read once):
    BFS_MUMPS_ORDERING   ICNTL(7): 5 = METIS (default), 7 = automatic
    BFS_MUMPS_MEM        ICNTL(14): workspace increase in percent (default 50)
"""
import os
import time
import itertools
import numpy as np
import scipy.sparse as sp

_count = itertools.count()
_PETSc = None
# this process's MUMPS work, for the run-time estimate (BFS.md 7): factorisations
# and single-RHS solves with their seconds, split by whether they ran inside a
# full-model pool task (fullpar.IN_TASK), inside a lattice wave as rank 0's own
# share (IN_WAVE, set by lattice.LatticeOracle._run; 20261010j stage C, driver-
# serial.md P1: 'driver' was rank 0 in any context, its share of the waves --
# 0.51-1.85x an average rank's work, run beside the workers -- read as serial
# work), or in the driver itself: 'driver' is now the truly serial work
STATS = {k: [0, 0.0] for k in ('factor_driver', 'factor_task', 'solve_driver', 'solve_task',
                               'factor_wave', 'solve_wave')}
IN_WAVE = False


def _account(kind, dt):
    import fullpar
    e = STATS[kind + ('_wave' if IN_WAVE else ('_task' if fullpar.IN_TASK else '_driver'))]
    e[0] += 1; e[1] += dt


def report():
    return "MUMPS (this process): " + ", ".join(f"{k} {v[0]} ({v[1]:.1f} s)" for k, v in STATS.items())


def _petsc():
    global _PETSc
    if _PETSc is None:
        import petsc4py
        # PETSc on COMM_SELF: PetscInitialize is collective over its world, and
        # in the MPI task farm every rank initialises on its own, at its first
        # factorisation (on COMM_WORLD a worker waited forever for the driver)
        from mpi4py import MPI
        petsc4py.init([], comm=MPI.COMM_SELF)
        from petsc4py import PETSc
        if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
            raise RuntimeError("bfs/mumps_lu.py needs PETSc built with complex scalars")
        _PETSc = PETSc
    return _PETSc


class MumpsLU:
    def __init__(self, S):
        t0 = time.perf_counter()
        PETSc = _petsc()
        S = sp.csr_matrix(S, dtype=complex)
        S.sum_duplicates(); S.sort_indices()
        self.n = S.shape[0]; self.shape = S.shape
        comm = PETSc.COMM_SELF
        self._A = PETSc.Mat().createAIJ(size=S.shape, csr=(S.indptr.astype(PETSc.IntType),
                                                            S.indices.astype(PETSc.IntType), S.data), comm=comm)
        self._A.assemble()
        pre = f"bfslu{next(_count)}_"
        opts = PETSc.Options()
        opts[pre + 'mat_mumps_icntl_7'] = int(os.environ.get('BFS_MUMPS_ORDERING', 5))
        opts[pre + 'mat_mumps_icntl_14'] = int(os.environ.get('BFS_MUMPS_MEM', 50))
        ksp = PETSc.KSP().create(comm)
        ksp.setOptionsPrefix(pre)
        ksp.setOperators(self._A)
        ksp.setType('preonly')
        pc = ksp.getPC(); pc.setType('lu'); pc.setFactorSolverType('mumps')
        ksp.setFromOptions()
        ksp.setUp()                                     # the factorisation
        for k in ('mat_mumps_icntl_7', 'mat_mumps_icntl_14'):
            opts.delValue(pre + k)
        self._ksp = ksp
        F = pc.getFactorMatrix()
        try:
            self.memory_mb = float(F.getMumpsInfog(17))      # total MB, all processes
        except Exception:
            self.memory_mb = float('nan')
        self._x, self._b = self._A.createVecs()
        _account('factor', time.perf_counter() - t0)

    def _one(self, b, trans):
        t0 = time.perf_counter()
        self._b.setArray(b if trans != 'H' else np.conj(b))
        if trans == 'N':
            self._ksp.solve(self._b, self._x)
            x = self._x.getArray().copy()
        else:
            self._ksp.solveTranspose(self._b, self._x)      # A^T x = b ;  A^H x = b  <=>  A^T conj(x) = conj(b)
            x = self._x.getArray().copy()
            x = x if trans == 'T' else np.conj(x)
        _account('solve', time.perf_counter() - t0)
        return x

    def solve(self, b, trans='N'):
        b = np.asarray(b, complex)
        if b.ndim == 1:
            return self._one(b, trans)
        return np.column_stack([self._one(np.ascontiguousarray(b[:, k]), trans) for k in range(b.shape[1])])

    def __del__(self):
        try:
            self._ksp.destroy(); self._A.destroy(); self._x.destroy(); self._b.destroy()
        except Exception:
            pass
