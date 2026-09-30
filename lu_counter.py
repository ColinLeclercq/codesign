"""lu_counter.py -- count sparse factorisations and triangular solves.

The handoff's stage-0 harness must report "factorisations per inner iteration /
per outer round", and its cost principle is counts before times: at GL an LU
costs about five triangular solves, so wall-clock speedups are meaningless and
the quantity to project through the 2-D cost ratio C = 50-100 is the
factorisation and solve count.

This module obtains those counts without editing a single line of the
production or certification code.  Every factorisation in the pipeline goes
through `scipy.sparse.linalg.splu` by ATTRIBUTE access:

    closedloop_sparse.py:145   lu = spla.splu(Sp.tocsc())   # the ONLY factorisation
    certify_sparse.py:112      self._lu = spla.splu(S)
    certify_sparse.py:314      lu = spla.splu(S)
    fe_codesign.py (_Hlu)      Sobolev LU, once per mesh, on first use (free-field cases)
    massops.py                 lu(M) / lu_real(M)            # mass, ONCE per run (warm-up)

so replacing the module attribute intercepts all of them, and the returned
SuperLU object is wrapped in a proxy that counts `.solve()` calls and forwards
everything else untouched.  Nothing downstream can tell the difference, which
is the point: the certification stack never changes.

Counts are attributed to the CALLER's file:line, so the once-per-mesh
factorisations (fe_codesign) can be separated from the per-frequency ones
(closedloop_sparse) rather than lumped into a single number that would flatter
or slander any variant depending on how often the mesh was rebuilt.

Usage:

    import lu_counter
    lu_counter.install()
    ...
    lu_counter.reset()
    <work>
    c = lu_counter.snapshot()      # dict: n_lu, n_solve, by_site, ...

or, scoped:

    with lu_counter.counting() as c:
        <work>
    print(c.n_lu, c.n_solve)
"""
from __future__ import annotations

import contextlib
import threading
import sys
from collections import Counter

_state = threading.local()
_installed = False
_orig_splu = None

# module-level totals (single process; the worker pool is separate processes,
# and their counts are reported by the worker itself when asked)
_n_lu = 0
_n_solve = 0
_by_site_lu: Counter = Counter()
_by_site_solve: Counter = Counter()
_lock = threading.Lock()


def _caller(skip: int = 3) -> str:
    """file:line of the first frame outside this module.  Walks the frames
    with sys._getframe (2026-09-29: traceback.extract_stack read source lines
    through linecache on every counted solve -- 6 % of a MIMO round)."""
    try:
        f = sys._getframe(1)
        imm = None
        while f is not None:
            fn = f.f_code.co_filename
            if not fn.endswith("lu_counter.py"):
                fn = fn.rsplit("/", 1)[-1]
                if imm is None:
                    imm = f"{fn}:{f.f_lineno}"
                # the COMPONENT: innermost frame in the driver or the inner solver
                if fn in ("run_case_epi.py", "sqp_inner.py", "stabilise.py"):
                    return f"{fn}:{f.f_lineno} <- {imm}"
            f = f.f_back
        if imm is not None:
            return imm
    except Exception:
        pass
    return "?"


class _LUProxy:
    """Forwards everything to the real SuperLU object, counting .solve()."""

    __slots__ = ("_lu", "_site")

    def __init__(self, lu, site):
        object.__setattr__(self, "_lu", lu)
        object.__setattr__(self, "_site", site)

    def solve(self, *args, **kwargs):
        global _n_solve
        with _lock:
            _n_solve += 1
            _by_site_solve[_caller()] += 1
        return self._lu.solve(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_lu"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_lu"), name, value)

    def __repr__(self):
        return f"<counted {self._lu!r}>"


# 2026-09-22: every splu was counted as a factorisation, including the
# ~100-dimensional reduced pencils of the reduced-model certificate.  A size
# floor separates FULL-model factorisations (the cost that scales with n).
BIG_N = 0            # set by the driver to the plant dimension; 0 = count all as big
REDUCED_CONTEXT = False   # set True around reduced-model computations: their factorisations are never 'full',
                          # whatever their size (a reduced Hamiltonian pencil 2(k+r) can exceed n on GL)
_n_lu_big = 0
import collections as _col
_by_site_full = _col.Counter()


def _splu_counted(*args, **kwargs):
    global _n_lu, _n_lu_big
    site = _caller()
    try:
        nrow = int(args[0].shape[0]) if args else int(kwargs.get('A').shape[0])
    except Exception:
        nrow = -1
    with _lock:
        _n_lu += 1
        if (nrow < 0 or nrow >= BIG_N) and not REDUCED_CONTEXT:
            _n_lu_big += 1
            _by_site_full[site] += 1
        _by_site_lu[site] += 1
    return _LUProxy(_orig_splu(*args, **kwargs), site)


def install() -> bool:
    """Patch scipy.sparse.linalg.splu.  Idempotent; returns True if patched."""
    global _installed, _orig_splu
    if _installed:
        return False
    import scipy.sparse.linalg as spla
    _orig_splu = spla.splu
    spla.splu = _splu_counted
    _installed = True
    return True


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    import scipy.sparse.linalg as spla
    spla.splu = _orig_splu
    _installed = False


def reset() -> None:
    global _n_lu, _n_solve, _n_lu_big
    with _lock:
        _n_lu = 0
        _n_lu_big = 0
        _n_solve = 0
        _by_site_lu.clear()
        _by_site_full.clear()
        _by_site_solve.clear()


def snapshot() -> dict:
    with _lock:
        return dict(
            n_lu=_n_lu,
            n_lu_full=_n_lu_big,
            by_site_full=dict(_by_site_full),
            n_solve=_n_solve,
            solves_per_lu=(float(_n_solve) / _n_lu) if _n_lu else 0.0,
            by_site_lu=dict(_by_site_lu),
            by_site_solve=dict(_by_site_solve),
            installed=_installed,
        )


class _Counts:
    def __init__(self):
        self.n_lu = 0
        self.n_solve = 0
        self.by_site_lu: dict = {}
        self.by_site_solve: dict = {}

    def _fill(self, d):
        self.n_lu = d["n_lu"]
        self.n_solve = d["n_solve"]
        self.by_site_lu = d["by_site_lu"]
        self.by_site_solve = d["by_site_solve"]

    def as_dict(self):
        return dict(n_lu=self.n_lu, n_solve=self.n_solve,
                    by_site_lu=self.by_site_lu,
                    by_site_solve=self.by_site_solve)


@contextlib.contextmanager
def counting():
    """Scoped counting.  Installs if needed; restores the previous totals."""
    install()
    before = snapshot()
    reset()
    c = _Counts()
    try:
        yield c
    finally:
        c._fill(snapshot())
        # restore the outer totals plus what happened inside, so nesting works
        global _n_lu, _n_solve
        with _lock:
            _n_lu = before["n_lu"] + c.n_lu
            _n_solve = before["n_solve"] + c.n_solve
            for k, v in before["by_site_lu"].items():
                _by_site_lu[k] += v
            for k, v in before["by_site_solve"].items():
                _by_site_solve[k] += v


def selftest() -> int:
    """Verify the proxy is transparent and the counts are right."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    install()
    A = sp.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    rhs = np.array([1.0, 2.0])
    with counting() as c:
        lu = spla.splu(A)
        x1 = lu.solve(rhs)
        x2 = lu.solve(2 * rhs)
    ok_val = (np.allclose(A @ x1, rhs) and np.allclose(A @ x2, 2 * rhs))
    ok_cnt = (c.n_lu == 1 and c.n_solve == 2)
    # transparency: attributes of the real object are reachable
    ok_attr = hasattr(lu, "L") and hasattr(lu, "perm_r")
    print(f"  values correct : {ok_val}")
    print(f"  counts 1 LU / 2 solves : {ok_cnt} (got {c.n_lu}/{c.n_solve})")
    print(f"  proxy transparent (.L, .perm_r) : {ok_attr}")
    print(f"  sites: {c.by_site_lu}")
    return 0 if (ok_val and ok_cnt and ok_attr) else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
