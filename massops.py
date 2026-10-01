"""massops.py -- the mass matrix's factorisations, ONCE per matrix (STAGE6 6.29).

M never changes during a run (user, 2026-09-30: "it should be done once in a
warm-up phase of the code and cached").  Before this module, seven call sites
factorised it on their own (certificate region, high-gain estimates, the all-w
tail, the plant enclosure and pencils, the winding tail, the field rows,
smooth_unmap), several of them on every call.  Now:

    massops.warmup(M)        at start-up (run_case_epi, right after the plant
                             is assembled): factorises and reports the time
    massops.lu(M)            the complex LU of M (the one every complex solve
                             uses; bitwise the factor the old sites built)
    massops.lu_real(M)       the real LU (smooth_unmap's real/imaginary split),
                             built on first use
    massops.block_solve(M, r, x)   (blockdiag(M, I_r))^-1 x

Keyed on the matrix OBJECT, which the registry keeps alive (an id cannot be
reused).  GL: M is real symmetric positive definite.  At the backward-facing
step M is the VELOCITY mass (singular on the pressure block): these helpers
then do not apply as such -- the M-norm bounds must be taken on the velocity
subspace (STAGE6 6.29, open item).
"""
import time
import numpy as np
import scipy.sparse as sp
import sparse_lu

_REG = {}          # id(M) -> dict(M=M, c=complex LU, r=real LU)
# Backward-facing step (BFS.md 4): M is the velocity mass, singular, and must
# never be factorised or inverted (user, 2026-10-01).  The BFS plant sets
# FORBIDDEN; every helper then raises instead of factorising (a singular LU can
# 'succeed' with perturbed pivots and return garbage silently).
FORBIDDEN = False


def _entry(M):
    if FORBIDDEN:
        raise RuntimeError("massops: the mass matrix is singular here (backward-facing step) and is "
                           "never factorised -- this code path needs an M-free replacement (BFS.md 4)")
    e = _REG.get(id(M))
    if e is None or e['M'] is not M:
        e = _REG[id(M)] = dict(M=M, c=None, r=None)
    return e


def lu(M):
    e = _entry(M)
    if e['c'] is None:
        e['c'] = sparse_lu.factor(sp.csc_matrix(M).astype(complex))
    return e['c']


def lu_real(M):
    e = _entry(M)
    if e['r'] is None:
        e['r'] = sparse_lu.factor(sp.csc_matrix(M))
    return e['r']


def block_solve(M, r, x):
    x = np.asarray(x, complex)
    n = M.shape[0]
    out = x.copy()
    out[:n] = lu(M).solve(x[:n])
    return out


def warmup(M, verbose=True):
    t0 = time.time()
    lu(M)
    if verbose:
        print(f"warm-up: mass matrix factorised once (n = {M.shape[0]}, {time.time() - t0:.2f} s)", flush=True)
