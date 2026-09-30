"""sparse_lu.py -- every full-model sparse factorisation of the pipeline goes
through factor() (STAGE6 6.29).

Until 2026-09-30 each module called scipy.sparse.linalg.splu with its defaults
(COLAMD column ordering, partial pivoting): fine for the banded 1-D GL
operators, poor for 2-D/3-D finite-element matrices, which are structurally
symmetric.  On a 2-D model problem (n = 14,400) the Hamiltonian-pencil LU took
35.1 s with the defaults and 3.4 s with a symmetric ordering of A + A^T; the
resolvent LU 0.51 s -> 0.21 s (review, 6.28).  One place now decides:

    ORDER = 'colamd'      scipy's default (the frozen path, bit-identical)
            'mmd_at_plus_a'   minimum degree on A + A^T, SymmetricMode with a
                          relaxed pivot threshold (diag_pivot_thresh = 0.01):
                          for structurally symmetric FE matrices (driver
                          --lu-order)

The returned object has SuperLU's interface (.solve(b, trans='N'|'T'|'H')), so
a distributed direct solver (MUMPS through PETSc, the stage-7 plan for the
backward-facing step) can be added here as another ORDER/backend without
touching the callers.  The factorisation is called as spla.splu (attribute
access), so lu_counter still counts every one.
"""
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ORDER = 'colamd'


def factor(S):
    S = sp.csc_matrix(S)
    if ORDER == 'mmd_at_plus_a':
        return spla.splu(S, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.01,
                         options=dict(SymmetricMode=True))
    return spla.splu(S)
