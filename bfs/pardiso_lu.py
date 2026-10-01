"""pardiso_lu.py -- complex sparse LU through MKL PARDISO (ctypes), the SANDBOX
stand-in for MUMPS (BFS.md 3).

The sandbox has no PETSc/MUMPS, and SciPy's SuperLU runs out of memory on the
backward-facing-step pencil (n = 191,720: > 6 GB).  MKL ships with numpy's
wheels here, and PARDISO (nested-dissection ordering, like MUMPS) factors it.
The object has SuperLU's interface, .solve(b, trans='N'|'T'|'H') for one or
several right-hand sides, so sparse_lu.factor() returns it for
ORDER = 'pardiso' without touching any caller.  On the cluster the same place
returns a MUMPS factor (bfs/mumps_lu.py).
"""
import ctypes
import ctypes.util
import os
import numpy as np
import scipy.sparse as sp

_lib = None


def _mkl():
    global _lib
    if _lib is None:
        path = os.environ.get('MKL_RT') or ctypes.util.find_library('mkl_rt') or '/usr/local/lib/libmkl_rt.so.3'
        _lib = ctypes.CDLL(path)
        _lib.pardiso.restype = None
        _lib.pardisoinit.restype = None
    return _lib


_i = ctypes.POINTER(ctypes.c_int)


def _ip(x):
    return ctypes.byref(ctypes.c_int(int(x)))


class PardisoLU:
    """Complex unsymmetric LU (mtype 13) of a square sparse matrix."""

    def __init__(self, S):
        lib = _mkl()
        S = sp.csr_matrix(S, dtype=complex)
        S.sum_duplicates()        # PARDISO takes no duplicate entries (sorted, summed CSR)
        S.sort_indices()
        self.n = S.shape[0]
        self.shape = S.shape
        self._a = np.ascontiguousarray(S.data)
        self._ia = np.ascontiguousarray(S.indptr, dtype=np.int32)
        self._ja = np.ascontiguousarray(S.indices, dtype=np.int32)
        self._pt = np.zeros(64, dtype=np.int64)
        self._iparm = np.zeros(64, dtype=np.int32)
        self._mtype = 13
        lib.pardisoinit(self._pt.ctypes.data_as(ctypes.c_void_p), _ip(self._mtype),
                        self._iparm.ctypes.data_as(_i))
        self._iparm[0] = 1        # the settings below are used (0 would reset them all to defaults)
        self._iparm[1] = 3        # parallel nested dissection (METIS)
        # No scaling / weighted matching (iparm 11, 13): with them this MKL's complex
        # factorisation of the BFS saddle-point pencil solves to a relative error
        # of 5e3 (no perturbed pivot reported); without, 9e-10 with 236
        # perturbed pivots, and 2.4 s instead of 20 s (n = 191,720).  Two steps
        # of iterative refinement remove the pivot perturbation's error.
        self._iparm[10] = 0
        self._iparm[12] = 0
        self._iparm[7] = 2
        self._iparm[34] = 1       # zero-based indices
        self._call(12)            # analysis + numerical factorisation

    def _call(self, phase, b=None, x=None, nrhs=1):
        err = ctypes.c_int(0)
        dummy = np.zeros(1, complex)
        _mkl().pardiso(self._pt.ctypes.data_as(ctypes.c_void_p), _ip(1), _ip(1), _ip(self._mtype), _ip(phase),
                       _ip(self.n), self._a.ctypes.data_as(ctypes.c_void_p), self._ia.ctypes.data_as(_i),
                       self._ja.ctypes.data_as(_i), None, _ip(nrhs), self._iparm.ctypes.data_as(_i), _ip(0),
                       (b if b is not None else dummy).ctypes.data_as(ctypes.c_void_p),
                       (x if x is not None else dummy).ctypes.data_as(ctypes.c_void_p), ctypes.byref(err))
        if err.value != 0:
            raise RuntimeError(f"PARDISO phase {phase}: error {err.value}")

    def solve(self, b, trans='N'):
        b = np.asarray(b)
        one = b.ndim == 1
        B = np.asfortranarray(b.reshape(self.n, -1), dtype=complex)
        X = np.zeros_like(B, order='F')
        self._iparm[11] = {'N': 0, 'H': 1, 'T': 2}[trans]
        self._call(33, B, X, nrhs=B.shape[1])
        return X[:, 0] if one else X

    @property
    def memory_mb(self):
        return float(max(self._iparm[14], self._iparm[15] + self._iparm[16])) / 1024.0

    def __del__(self):
        try:
            self._call(-1)
        except Exception:
            pass
