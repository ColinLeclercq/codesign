#!/usr/bin/env python3
"""bench_mumps.py -- what one full-model task costs on a FULL node (BFS.md 7).

Every MPI rank does what a worker of the task farm does: loads the operator
file, factors i w M - A with sequential MUMPS (bfs/mumps_lu.py) at its own
frequency, and solves N right-hand sides, plain and adjoint.  All ranks run at
once, so the times include the memory-bandwidth contention of a full node --
the number the run-time estimate needs (a single factorisation on an idle node
is faster).  Rank 0 prints min / median / max over ranks.

    mpiexec -n 96 python bfs/bench_mumps.py --ops bfs/data/ops_fx.npz
"""
import os, sys, time, argparse, resource
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ops', default='bfs/data/ops_fx.npz')
    ap.add_argument('--nsolve', type=int, default=20)
    ap.add_argument('--repeat', type=int, default=2)
    a = ap.parse_args()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD; rank, size = comm.Get_rank(), comm.Get_size()
    from bfs.plant_io import load_ops
    from bfs.mumps_lu import MumpsLU
    t0 = time.time()
    o = load_ops(a.ops); A, M = o['A'].tocsc(), o['M'].tocsc(); n = A.shape[0]
    t_load = time.time() - t0
    rng = np.random.default_rng(rank)
    w = 0.05 + 8.0 * (rank + 0.5) / size
    tf, ts, err, mem = [], [], [], 0.0
    for _ in range(a.repeat):
        comm.Barrier()
        t0 = time.time(); S = (1j * w * M - A).tocsc(); lu = MumpsLU(S); tf.append(time.time() - t0)
        mem = max(mem, lu.memory_mb)
        B = rng.standard_normal((n, a.nsolve)) + 1j * rng.standard_normal((n, a.nsolve))
        t0 = time.time()
        X = lu.solve(B); Y = lu.solve(B, trans='H')
        ts.append((time.time() - t0) / (2 * a.nsolve))
        err.append(max(np.linalg.norm(S @ X - B) / np.linalg.norm(B),
                       np.linalg.norm(S.conj().T @ Y - B) / np.linalg.norm(B)))
        del lu
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    rows = comm.gather(dict(load=t_load, fac=float(np.median(tf)), sol=float(np.median(ts)),
                            err=float(max(err)), mem=mem, rss=rss), root=0)
    if rank == 0:
        def st(k):
            v = np.array([r[k] for r in rows])
            return f"{v.min():.3g} / {np.median(v):.3g} / {v.max():.3g}"
        print(f"bench_mumps: n = {n}, {size} ranks at once (min / median / max over ranks)")
        print(f"  load ops [s]           {st('load')}")
        print(f"  factor i w M - A [s]   {st('fac')}")
        print(f"  one solve (N or H) [s] {st('sol')}")
        print(f"  relative residual      {st('err')}")
        print(f"  MUMPS memory [MB]      {st('mem')}")
        print(f"  process max RSS [MB]   {st('rss')}  (node total ~ {sum(r['rss'] for r in rows) / 1024:.0f} GB)",
              flush=True)


if __name__ == '__main__':
    main()
