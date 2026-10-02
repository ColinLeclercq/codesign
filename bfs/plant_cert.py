#!/usr/bin/env python3
"""plant_cert.py -- the plant's design-independent stability data, ONCE per
mesh, cached next to the operator file (BFS.md 4):

  OPS_range.npz              the numerical range of A_vv on the divergence-free
                             velocities: right edge s_far, S-lemma heights, and
                             the H^1 constant beta of the wall-shear tail bound
                             -- certified by inertia (bfs/mfree.py)
  OPS_range_poles_<line>.npz the plant eigenvalues right of line - 0.6, certified
                             complete (sparse_spectrum.plant_eigs_right: the
                             rectangle above + tiles exhausted by nearest-k
                             shift-invert): the count right of the line and the
                             near-line (hot) poles of the winding count

Every production run then reads these files (BFSPlant.range_data /
plant_poles).  The depths and the tiles are independent: run under MPI they
spread over the full-model pool.

    mpiexec -n 24 python bfs/plant_cert.py --ops bfs/data/ops_fx.npz --eps 0.05
"""
import os, sys, time, argparse, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ops', default='bfs/data/ops_fx.npz')
    ap.add_argument('--eps', type=float, default=0.05)
    a = ap.parse_args()
    from bfs import mpipool
    if mpipool.active() and mpipool.comm().Get_rank() > 0:
        mpipool.serve()
    import sparse_lu
    sparse_lu.ORDER = 'mumps'
    import fullpar, certify_sparse
    from bfs.plant import BFSPlant
    D = BFSPlant(a.ops, r=5)
    certify_sparse.register_plant(D.A, D.M)
    fullpar.start(1, D=D)
    t0 = time.time()
    s_far, Hd, beta, nu1 = D.range_data()
    out = dict(ops=a.ops, n=D.n, s_far=s_far, heights=Hd, beta=beta, nu1=nu1, range_seconds=round(time.time() - t0, 1))
    print(json.dumps(out), flush=True)
    # the widest enclosure first (line -eps): the narrower one (-eps/2) is filtered from it
    for line in (-a.eps, -0.5 * a.eps):
        t0 = time.time()
        cnt, hot = D.plant_poles(line)
        print(json.dumps(dict(line=line, count=cnt, hot=len(hot), seconds=round(time.time() - t0, 1))), flush=True)
    print(fullpar.report(), flush=True)
    fullpar.stop()


if __name__ == '__main__':
    main()
