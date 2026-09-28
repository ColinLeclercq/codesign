"""
bench_oracle_now.py — the oracle as it now stands, serial vs pooled.

bench_par2.py predates two changes and cannot see them: it holds
alpha_certified serial (and so reports an Amdahl ceiling of ~2.4x), and it was
written before inverse iteration replaced ARPACK in the eigenvector recovery.

This measures the REAL path: one pool shared by the |Omega| constraint loop and
the segmented certificate, exactly as `run_case_epi.py --nproc N` uses it.

    python bench_oracle_now.py --ne 200
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
    ap.add_argument('--band', type=float, default=43.65)
    a = ap.parse_args()

    import multiprocessing as mp
    from parallel import ParallelOracle
    ncore = ParallelOracle.available_cores()
    print(f"host={os.uname().nodename}  allocated={ncore}  "
          f"band=+-{a.band:g}  eps={a.eps}")

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

    # --- serial reference, CURRENT code -----------------------------------
    D.cert_nseg, D.cert_pool = 1, None
    D.alpha_certified(a2, n2, K2, b, c, eps=a.eps,
                      wlo=-a.band, whi=a.band)
    t0 = time.time()
    for _ in range(3):
        D.alpha_certified(a2, n2, K2, b, c, eps=a.eps,
                          wlo=-a.band, whi=a.band)
    t_a = (time.time() - t0) / 3
    print(f"\nalpha_certified, serial, CURRENT code : {t_a:.4f} s")
    print("  (bench_par2 measured 0.1952 s before inverse iteration)")

    ctx = mp.get_context('fork')
    for nw in [int(x) for x in a.omegas.split(',')]:
        W = np.linspace(-1.2, 0.2, nw)
        cl.gain(float(W[0]), a2, n2, K2)
        t0 = time.time()
        for w in W:
            cl.gain(float(w), a2, n2, K2)
        t_c = time.time() - t0
        tot = t_c + t_a
        print(f"\n|Omega|={nw:3d}  constraints {t_c:.4f}  alpha {t_a:.4f}  "
              f"oracle {tot:.4f} s")
        for p in [int(x) for x in a.procs.split(',')]:
            if p > ncore:
                continue
            po = ParallelOracle(a.ne, a.r, 0.8, a.case, pk_args, nproc=p)
            po.constraints(z, W[:2])
            t0 = time.time(); po.constraints(z, W); tp = time.time() - t0
            po.close()
            # certificate with the band split p ways
            pool = ctx.Pool(processes=p) if p > 1 else None
            D.cert_nseg, D.cert_pool = (p, pool) if p > 1 else (1, None)
            D.alpha_certified(a2, n2, K2, b, c, eps=a.eps,
                              wlo=-a.band, whi=a.band)
            t0 = time.time()
            for _ in range(3):
                al_p = D.alpha_certified(a2, n2, K2, b, c, eps=a.eps,
                                         wlo=-a.band, whi=a.band)[0]
            ta_p = (time.time() - t0) / 3
            if pool is not None:
                pool.close(); pool.join()
            D.cert_nseg, D.cert_pool = 1, None
            print(f"    nproc={p:3d}  constraints {tp:.4f} ({t_c/tp:5.2f}x)  "
                  f"alpha {ta_p:.4f} ({t_a/ta_p:5.2f}x)  "
                  f"ORACLE {tp+ta_p:.4f} ({tot/(tp+ta_p):5.2f}x)")

    print("\nBoth blocks are now pooled.  bench_par2 held alpha serial and so")
    print("reported a ceiling of ~2.4x; that ceiling is what the segmented")
    print("certificate removes.")


if __name__ == "__main__":
    main()
