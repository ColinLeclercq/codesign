#!/usr/bin/env python3
"""ops_dolfinx.py -- the backward-facing-step operators from the user's
nsforms.py (dolfinx 0.11), BFS.md 1.

The discretisation of the archive's resolvent_gains.py (Dergham, Sipp &
Robinet 2013, Re = 600, linear-analysis mesh [-1.5, 30] x [0, 3.5]):
Taylor-Hood P2-P2-P1, the Jacobian dF/dW of the steady NS residual at the base
flow, the velocity mass, and the perturbation Dirichlet conditions of
nsforms.small_domain_bcs (u = v = 0 on labels 1, 2; v = 0 on label 4).  With
dolfinx's assemble_matrix(bcs, diag):

    -J   Dirichlet rows/columns -> identity  (diag 1)       A = J
    M    Dirichlet rows/columns -> 0         (diag 0), pressure block 0

so the plant is  M q' = A q + ...  with resolvent (i w M - A)^-1, real.  The
matrices are assembled on ONE process (seconds at n = 191,720) and written in
the plant_io format; the optimiser's processes only read that file.

    python bfs/ops_dolfinx.py --data bfs/archive_dsr/fx_data --baseflow baseflow_fenicsx.npz --out bfs/data/ops_fx.npz
"""
import os, sys, argparse, time
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(HERE, 'archive_dsr')]


def _csr(Mat):
    ia, ja, a = Mat.getValuesCSR()
    S = sp.csr_matrix((a, ja, ia), shape=Mat.getSize())
    if S.nnz and np.abs(S.data.imag).max() > 0.0:
        raise RuntimeError("assembled operator is not real")
    S = sp.csr_matrix((S.data.real, S.indices, S.indptr), shape=S.shape)
    S.eliminate_zeros()
    return S


def assemble(folder, baseflow, Re=600.0):
    """The operators (and the unconstrained velocity mass Mfull, for
    check_gains.py) on folder's mesh, base flow folder/baseflow."""
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import fem, mesh as dmesh
    from dolfinx.fem.petsc import assemble_matrix
    import nsforms as ns
    t0 = time.time()
    comm = MPI.COMM_SELF
    msh, ft = ns.read_xdmf_mesh(comm, folder)
    W, _ = ns.taylor_hood(msh)
    subs = ns.scalar_subspaces(W)
    wb = ns.load_field(W, np.load(os.path.join(folder, baseflow)), 'baseflow', subs)
    nu = fem.Constant(msh, PETSc.ScalarType(1.0 / Re))
    bcs = ns.small_domain_bcs(W, ft)
    mJ = assemble_matrix(fem.form(ns.jacobian(wb, nu)), bcs=bcs, diag=1.0); mJ.assemble()
    Mb = assemble_matrix(fem.form(ns.mass(W)), bcs=bcs, diag=0.0); Mb.assemble()
    Mf = assemble_matrix(fem.form(ns.mass(W))); Mf.assemble()
    A, M, Mfull = -_csr(mJ), _csr(Mb), _csr(Mf)
    n = A.shape[0]
    comp = np.full(n, -1, dtype=np.int8); xy = np.zeros((n, 2))
    for c, (X, dofs) in enumerate(subs):            # 0 = u, 1 = v, 2 = p
        comp[dofs] = c; xy[dofs] = X
    if (comp < 0).any():
        raise RuntimeError("dofs outside the u, v, p subspaces")
    bc = ns.owned_bc_rows(W, bcs)
    tri = np.asarray(msh.geometry.dofmaps[0])
    edges = dmesh.entities_to_geometry(msh, 1, ft.indices)
    print(f"assembled n = {n} ({int((comp < 2).sum())} velocity, {int((comp == 2).sum())} pressure dofs), "
          f"{len(bc)} Dirichlet dofs, nnz(A) = {A.nnz}, nnz(M) = {M.nnz} ({time.time() - t0:.1f} s)", flush=True)
    return dict(A=A, M=M, Mfull=Mfull, comp=comp, xy=xy, bc=bc, Re=Re,
                pts=msh.geometry.x[:, :2].copy(), tri=tri, edges=np.asarray(edges), edge_labels=ft.values.copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='bfs/archive_dsr/fx_data')
    ap.add_argument('--baseflow', default='baseflow_fenicsx.npz')
    ap.add_argument('--Re', type=float, default=600.0)
    ap.add_argument('--out', default='bfs/data/ops_fx.npz')
    a = ap.parse_args()
    ops = assemble(a.data, a.baseflow, a.Re)
    from bfs.plant_io import save_ops
    save_ops(a.out, ops, source=f'dolfinx (nsforms.py), {a.data}/{a.baseflow}, Re {a.Re}')


if __name__ == '__main__':
    main()
