#!/usr/bin/env python3
"""check_gains.py -- the plant operators (bfs/ops_dolfinx.py) against the user's
resolvent gains (archive results/gains_fx.txt, resolvent_gains.py: dolfinx +
SLEPc), with the code's own LU (sparse_lu, MUMPS), BFS.md 1.

Two forcing spaces, both without any mass inverse or Cholesky factor:
  'archive'   the archive's: forcing f in the whole P2 velocity space (its
              Dirichlet nodal values included), load Z Mfull f, norms Mfull.
              sigma^2 = lambda_max of  S = P_u Z T^H Mfull T Z Mfull  (self-
              adjoint in the Mfull inner product: Mfull^-1 cancels against
              the load map, the same identity as Qf^-1 P^T = E^T)
  'codesign'  the GL code's: load v = M f with M the Dirichlet-zeroed mass
              (forcing vanishing at Dirichlet dofs), W = M T^H M T

    python bfs/check_gains.py --data bfs/archive_dsr/fx_data --omegas 0.75 2
"""
import os, sys, time, argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='bfs/archive_dsr/fx_data')
    ap.add_argument('--baseflow', default='baseflow_fenicsx.npz')
    ap.add_argument('--omegas', type=float, nargs='+', default=[0.75, 2.0])
    ap.add_argument('--ref', default='bfs/archive_dsr/results/gains_fx.txt')
    ap.add_argument('--k', type=int, default=3)
    a = ap.parse_args()
    import sparse_lu
    sparse_lu.ORDER = 'mumps'
    from bfs.ops_dolfinx import assemble
    o = assemble(a.data, a.baseflow)
    A, M, Mfull = o['A'], o['M'], o['Mfull']       # Mfull: the archive's forcing space
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
