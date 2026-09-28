#!/usr/bin/env python3
"""lu_breakdown.py -- attribute every FULL-model factorisation of a run to the
loop component that asked for it (2026-09-26; STAGE5 30).

    python tools/lu_breakdown.py --eps 0.05 <other run_case_epi.py arguments ...>

Wraps scipy.sparse.linalg.splu BEFORE the driver installs lu_counter, records
for each call of full size (n >= the plant's order) the innermost frame in
run_case_epi.py (function:line) and the certificate routine if any, runs the
driver's main() unchanged, and prints the counts per component and per round
(rounds delimited by the driver's 'outer N:' lines).  Also prints the sparse
certificate's untrusted-disk and failed-climb counters."""
import os, sys, traceback, collections, builtins
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scipy.sparse.linalg as spla

_orig = spla.splu
COUNTS = collections.Counter()
PER_ROUND = collections.defaultdict(collections.Counter)
STATE = {'round': 'start', 'nmin': 150}
DEPTH = int(os.environ.get('LU_DEPTH', '1'))   # driver frames shown per call site
WATCH = ('certify_sparse.py', 'winding_cert.py', 'reduced.py', 'closedloop_sparse.py', 'fe_codesign.py', 'sqp_inner.py')


def _tag():
    st = traceback.extract_stack()[:-2]
    drv = [f for f in st if f.filename.endswith('run_case_epi.py')]
    inner = [f for f in st if os.path.basename(f.filename) in WATCH]
    d = ' < '.join(f"{f.name}:{f.lineno}" for f in drv[::-1][:DEPTH]) if drv else '?'
    i = f"{os.path.basename(inner[-1].filename)}:{inner[-1].name}" if inner else ''
    return f"{d} <- {i}"


def splu(A, *a, **k):
    import lu_counter as _lc
    if A.shape[0] >= STATE['nmin'] and not _lc.REDUCED_CONTEXT:
        t = _tag(); COUNTS[t] += 1; PER_ROUND[STATE['round']][t] += 1
    return _orig(A, *a, **k)


spla.splu = splu
_print = builtins.print


def _p(*a, **k):
    s = ' '.join(str(x) for x in a)
    if s.lstrip().startswith('outer ') and ':' in s[:12]:
        STATE['round'] = int(s.split()[1].rstrip(':')) + 1
    _print(*a, **k)


builtins.print = _p
import run_case_epi as R
sys.argv = ['run_case_epi.py'] + sys.argv[1:]
ne = int(sys.argv[sys.argv.index('--ne') + 1]) if '--ne' in sys.argv else 200
STATE['nmin'] = 2 * ne - 1  # the plant's order (199 at ne=100); reduced work is excluded by lu_counter.REDUCED_CONTEXT
try:
    R.main()
finally:
    import certify_sparse as csp
    builtins.print = _print
    rounds = [r for r in PER_ROUND if r != 'start']
    print("\n==== full-size factorisations by component (all rounds) ====")
    for t, n in COUNTS.most_common():
        print(f"{n:7d}  {t}")
    if rounds:
        tot = collections.Counter()
        for r in rounds:
            tot.update(PER_ROUND[r])
        print(f"\n==== per round, mean over rounds {min(rounds)}..{max(rounds)} (start excluded) ====")
        for t, n in tot.most_common():
            print(f"{n / len(rounds):8.1f}  {t}")
        print(f"{sum(tot.values()) / len(rounds):8.1f}  TOTAL per round")
    print(f"\nsparse certificate: untrusted disks {csp.N_DISK_UNTRUSTED}, failed climbs {csp.N_CLIMB_FAILED}")
