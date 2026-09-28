"""
bench_par2.py — what would parallelising actually buy, on the CURRENT code?

The earlier benchmark is stale.  It was taken when alpha cost 2.0 s of a 2.23 s
oracle via a 16-shift Krylov search, so alpha looked like the thing to
parallelise (it scaled only 2.3x, load imbalance).  Alpha now comes from the 1-D
certificate and costs ~0.17 s, so the |Omega| constraints dominate instead --
and unlike alpha they are genuinely independent.

Two things matter and both are measured here:

  * the CONSTRAINT loop scaling, at the |Omega| values the production runs
    actually reached (27 to 79, not the 40 the old benchmark assumed);
  * the SERIAL fraction.  alpha_certified is an adaptive interval BISECTION --
    each shift's result decides the next -- so it does not parallelise, and it
    sets a hard Amdahl ceiling.

    python bench_par2.py --ne 200
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_codesign
from fe_codesign import parse_case
from packer import Packer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--r', type=int, default=7)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--case', default='K+FB+FC')
    ap.add_argument('--procs', default='1,4,8,16,24,32')
    ap.add_argument('--omegas', default='27,50,79')
    a = ap.parse_args()

    from parallel import ParallelOracle
    ncore = ParallelOracle.available_cores()
    import multiprocessing as mp
    print(f"host={os.uname().nodename}  machine={mp.cpu_count()}  "
          f"allocated={ncore}")

    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=0.8)
    ini = np.load(f"init_ne{a.ne}.npz")
    arow = (ini['d'], ini['e'], ini['f'])
    pk_args = (arow, ini['nvec'], 0j, float(ini['xa']), float(ini['xs']),
               D.gauss(float(ini['xa'])), D.gauss(float(ini['xs'])))
    pk = Packer(D, parse_case(a.case), *pk_args)
    z = pk.z0()
    a2, n2, K2, xa2, xs2, va2, vs2 = pk.unpack(z)
    b, c, *_ = D.bc(xa2, xs2, va2, vs2)
    cl = D.cl(b, c)

    # --- the SERIAL block: the certificate is a bisection, not a loop --------
    D.alpha_certified(a2, n2, K2, b, c, eps=a.eps)
    t0 = time.time()
    for _ in range(3):
        D.alpha_certified(a2, n2, K2, b, c, eps=a.eps)
    t_a = (time.time() - t0) / 3
    print(f"\nSERIAL alpha_certified: {t_a:.4f} s  "
          f"(adaptive bisection -- does NOT parallelise)")

    for nw in [int(x) for x in a.omegas.split(',')]:
        W = np.linspace(-1.2, 0.2, nw)
        cl.gain(float(W[0]), a2, n2, K2)
        t0 = time.time()
        for w in W:
            cl.gain(float(w), a2, n2, K2)
        t_c = time.time() - t0
        f = t_c / (t_c + t_a)
        print(f"\n|Omega|={nw:3d}  constraints {t_c:.4f} s  "
              f"oracle {t_c+t_a:.4f} s   parallel fraction f={f:.3f}"
              f"   Amdahl ceiling {1/(1-f):.2f}x")
        for p in [int(x) for x in a.procs.split(',')]:
            if p > ncore:
                continue
            po = ParallelOracle(a.ne, a.r, 0.8, a.case, pk_args, nproc=p)
            po.constraints(z, W[:2])
            t0 = time.time(); po.constraints(z, W); tp = time.time() - t0
            po.close()
            print(f"    nproc={p:3d}  constraints {tp:.4f}s "
                  f"({t_c/tp:5.2f}x)   ORACLE {(tp+t_a):.4f}s "
                  f"({(t_c+t_a)/(tp+t_a):5.2f}x)")

    print("\nRead the ORACLE column, not the constraints column: alpha is")
    print("serial and caps the whole thing.  Wiring parallel.py in is worth")
    print("doing if the oracle speedup is 2.5x or better at |Omega| ~ 79;")
    print("below that, cutting the ITERATION count is the better target.")


if __name__ == "__main__":
    main()
