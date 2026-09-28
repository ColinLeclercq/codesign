"""
bench_segcert.py — does SEGMENTING the stability certificate parallelise?

The certificate is an adaptive interval bisection over the line Re = -eps: each
shift's radius decides where the next one goes, so it is sequential by
construction and currently caps the oracle (it is ~43% of a call, and Amdahl
therefore limits the whole thing to ~2.3x however many cores are thrown at the
frequency constraints).

The idea under test: split the band into P segments and bisect each
INDEPENDENTLY.  Correctness is unaffected -- a disk from one segment spilling
into its neighbour is simply re-derived there, so leakage costs REDUNDANT WORK,
never a wrong answer, and no communication is needed.  The cost is that the
total shift count grows with P.

This was measured once before at 2.2x and plateaued, but that was with k = 6,
where three high-frequency shifts carried 79% of the work and splitting a
segment could not split a shift.  At k = 1 those shifts became 30-100x cheaper,
so the per-shift cost should now be roughly uniform and the ceiling much higher.

Reports, versus P:
  * total shifts (the redundancy cost),
  * the SLOWEST segment (what a parallel run would actually wait for),
  * the implied speedup, and the measured one with a real process pool.

    python bench_segcert.py --ne 200
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_codesign, controller as ctl, certify_sparse as csp

_W = {}


def _init(ne, r, sob_len, xa, xs):
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[v] = '1'
    import fe_codesign as _f
    D = _f.FECoDesign(ne=ne, r=r, sob_len=sob_len)
    D.lu_cache_max = 0
    b, c = D.bc(xa, xs)[:2]
    _W.update(D=D, b=b, c=c)


def _seg(args):
    """Certify ONE segment of the line; independent of every other segment."""
    lo, hi, arow, nvec, Kfb, eps, shrink = args
    D, b, c = _W['D'], _W['b'], _W['c']
    Ac = ctl.A_of(*arow)
    Bc = np.full(len(np.atleast_1d(arow[0])), 1.0, complex)
    t0 = time.time()
    clear, al, ns, unc, wit = csp.stability_and_abscissa(
        D.A, D.M, b, c, Ac, Bc, nvec, Kfb, eps=eps, shrink=shrink,
        wlo=lo, whi=hi)
    return dict(clear=bool(clear), alpha=float(al), shifts=int(ns),
                uncovered=float(unc), sec=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--r', type=int, default=7)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--band', type=float, default=43.65,
                    help='half-width; the sweep uses 3/2 x wmax')
    ap.add_argument('--segs', default='1,2,4,8,16,24,32')
    ap.add_argument('--nproc', type=int, default=0,
                    help='0 = use the SLURM allocation')
    a = ap.parse_args()

    from parallel import ParallelOracle
    ncore = a.nproc or ParallelOracle.available_cores()
    import multiprocessing as mp
    print(f"host={os.uname().nodename}  allocated={ncore}  "
          f"band=[{-a.band:g},{a.band:g}]  eps={a.eps}")

    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=0.8)
    ini = np.load(f"init_ne{a.ne}.npz")
    arow = (ini['d'], ini['e'], ini['f']); nvec = ini['nvec']
    xa, xs = float(ini['xa']), float(ini['xs'])
    b, c = D.bc(xa, xs)[:2]
    Ac = ctl.A_of(*arow); Bc = np.full(a.r, 1.0, complex)

    # --- reference: one sequential sweep over the whole band ---------------
    t0 = time.time()
    clear0, al0, ns0, unc0, _ = csp.stability_and_abscissa(
        D.A, D.M, b, c, Ac, Bc, nvec, 0j, eps=a.eps,
        wlo=-a.band, whi=a.band)
    t_seq = time.time() - t0
    print(f"\nSEQUENTIAL: {ns0} shifts, {t_seq:.3f} s, alpha={al0:+.6f}, "
          f"clear={clear0}")
    print(f"  per-shift: {t_seq/max(ns0,1)*1e3:.1f} ms mean")

    ctx = mp.get_context('fork')
    print(f"\n{'P':>4} {'shifts':>7} {'work':>6} {'slowest':>9} {'ideal':>7} "
          f"{'nproc':>6} {'measured':>9} {'alpha err':>10} {'unc':>7}")
    for P in [int(x) for x in a.segs.split(',')]:
        edges = np.linspace(-a.band, a.band, P + 1)
        tasks = [(float(edges[i]), float(edges[i + 1]), arow, nvec, 0j,
                  a.eps, 0.9) for i in range(P)]
        # serial pass over the segments: exact shift counts and per-segment time
        _init(a.ne, a.r, 0.8, xa, xs)
        res = [_seg(t) for t in tasks]
        tot = sum(r['shifts'] for r in res)
        slow = max(r['sec'] for r in res)
        al = max(r['alpha'] for r in res)
        unc = sum(r['uncovered'] for r in res)
        ideal = t_seq / slow if slow > 0 else float('nan')
        # measured, with a real pool
        p = min(P, ncore)
        meas = float('nan')
        if P > 1 and p > 1:
            pool = ctx.Pool(processes=p, initializer=_init,
                            initargs=(a.ne, a.r, 0.8, xa, xs))
            pool.map(_seg, tasks[:1])                 # warm
            t0 = time.time(); pool.map(_seg, tasks, chunksize=1)
            tm = time.time() - t0
            pool.close(); pool.join()
            meas = t_seq / tm
        print(f"{P:4d} {tot:7d} {tot/max(ns0,1):5.2f}x {slow:8.3f}s "
              f"{ideal:6.2f}x {p:6d} {meas:8.2f}x {abs(al-al0):10.1e} "
              f"{unc:7.1e}")

    print("\nRead 'measured'.  'ideal' is what a perfect pool would give (the")
    print("slowest segment); 'work' is the redundancy cost of splitting -- a")
    print("disk spilling across a segment boundary is simply re-derived, which")
    print("is why no communication is needed and why the shift count grows.")
    print("'alpha err' and 'unc' must stay at 0: segmenting must not change the")
    print("ANSWER, only the cost.")


if __name__ == "__main__":
    main()
