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
import itertools
import numpy as np
import scipy.sparse as sp

_count = itertools.count()
_PETSc = None


def _petsc():
    global _PETSc
    if _PETSc is None:
        import petsc4py
        petsc4py.init([])
        from petsc4py import PETSc
        if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
            raise RuntimeError("bfs/mumps_lu.py needs PETSc built with complex scalars")
        _PETSc = PETSc
    return _PETSc


class MumpsLU:
    def __init__(self, S):
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

    def _one(self, b, trans):
        self._b.setArray(b if trans != 'H' else np.conj(b))
        if trans == 'N':
            self._ksp.solve(self._b, self._x)
            return self._x.getArray().copy()
        self._ksp.solveTranspose(self._b, self._x)      # A^T x = b ;  A^H x = b  <=>  A^T conj(x) = conj(b)
        x = self._x.getArray().copy()
        return x if trans == 'T' else np.conj(x)

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
