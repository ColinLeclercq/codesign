#!/usr/bin/env python3
"""check_gains.py -- the assembled operators against the user's resolvent gains
(archive results/gains_fx.txt, dolfinx + SLEPc), BFS.md 1.

Two forcing spaces, both without any mass inverse or Cholesky factor:
  'archive'   the archive's: forcing f in the whole P2 velocity space (its
              Dirichlet nodal values included), load Z Mfull f, norms Mfull.
              sigma^2 = lambda_max of  S = P_u Z T^H Mfull T Z Mfull  (self-
              adjoint in the Mfull inner product: Mfull^-1 cancels against
              the load map, the same identity as Qf^-1 P^T = E^T)
  'codesign'  the GL code's: load v = M f with M the Dirichlet-zeroed mass
              (forcing vanishing at Dirichlet dofs), W = M T^H M T

    python bfs/check_gains.py --ops bfs/data/ops_fx.npz --omegas 0.75 2
"""
import os, sys, time, argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ops', default='bfs/data/ops_fx.npz')
    ap.add_argument('--omegas', type=float, nargs='+', default=[0.75, 2.0])
    ap.add_argument('--ref', default='bfs/archive_dsr/results/gains_fx.txt')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--lu', default='pardiso', help="sparse_lu backend: pardiso (sandbox) or mumps (cluster)")
    a = ap.parse_args()
    import sparse_lu
    sparse_lu.ORDER = a.lu
    from bfs.plant_io import load_ops
    import bfs.ops_skfem as osk
    o = load_ops(a.ops)
    A, M = o['A'], o['M']
    # the unconstrained velocity mass for the archive's forcing space
    ops = osk.assemble('bfs/archive_dsr/fx_data', 'baseflow_fenicsx.npz')
    Mfull = ops['Mfull']
    n = A.shape[0]; vel = o['comp'] < 2
    keep = np.ones(n); keep[o['bc']] = 0.0
    ref = np.loadtxt(a.ref)
    rng = np.random.default_rng(0)
    for w in a.omegas:
        t0 = time.time()
        lu = sparse_lu.factor(1j * w * M - A)
        tf = time.time() - t0
        T = lambda x: lu.solve(np.asarray(x, complex))
        TH = lambda x: lu.solve(np.asarray(x, complex), trans='H')
        S = spla.LinearOperator((n, n), dtype=complex,
                                matvec=lambda f: vel * (keep * TH(Mfull @ T(keep * (Mfull @ f)))))
        W = spla.LinearOperator((n, n), dtype=complex, matvec=lambda v: M @ TH(M @ T(v)))
        v0 = (rng.standard_normal(n) * vel).astype(complex)
        t0 = time.time()
        ls = np.sort(np.real(spla.eigs(S, k=a.k, which='LM', v0=v0, tol=1e-10, return_eigenvectors=False)))[::-1]
        lw = np.sort(np.real(spla.eigs(W, k=a.k, which='LM', v0=M @ v0, tol=1e-10, return_eigenvectors=False)))[::-1]
        r = ref[np.argmin(abs(ref[:, 0] - w))]
        print(f"w = {w}: LU {tf:.1f} s, eigs {time.time() - t0:.1f} s")
        print(f"   archive (dolfinx + SLEPc): " + " ".join(f"{x:.6e}" for x in r[1:1 + a.k]))
        print(f"   archive forcing space    : " + " ".join(f"{x:.6e}" for x in ls)
              + f"   rel {abs(ls[0] - r[1]) / r[1]:.1e}")
        print(f"   codesign forcing space   : " + " ".join(f"{x:.6e}" for x in lw)
              + f"   rel {abs(lw[0] - r[1]) / r[1]:.1e}", flush=True)


if __name__ == '__main__':
    main()
