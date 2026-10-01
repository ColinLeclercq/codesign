#!/usr/bin/env python3
"""ops_skfem.py -- the backward-facing-step operators WITHOUT dolfinx (BFS.md 1).

Same discretisation as the user's nsforms.py / resolvent_gains.py (Dergham,
Sipp & Robinet 2013, Re = 600, linear-analysis mesh [-1.5, 30] x [0, 3.5]):
Taylor-Hood P2-P2-P1 on the given triangles, the steady NS residual

    F(U, P; v, q) = (U.grad U, v) + nu (grad U, grad v) - (P, div v) - (div U, q),

its Jacobian dF/dW at the base flow, J = -dF/dW, the velocity mass M, and the
perturbation Dirichlet conditions u = v = 0 on labels 1 (inlet) and 2 (walls),
v = 0 on label 4 (top), natural at the outlet (label 3).  Dirichlet rows and
columns: identity in -J (diag 1), zero in M (diag 0), as dolfinx's
assemble_matrix(bcs, diag) does.  So

    plant   M q' = A q + ...,   A = J  (real),   resolvent (i w M - A)^-1.

Used in the sandbox (no dolfinx / PETSc here) to develop and test the fork on
the very matrices of the cluster's dolfinx assembly (bfs/ops_dolfinx.py): the
two write the same file format (bfs/plant_io.py).  Validated against the
user's resolvent gains (results/gains_fx.txt): tools in bfs/check_gains.py.

    python bfs/ops_skfem.py --data bfs/archive_dsr/fx_data --baseflow baseflow_fenicsx.npz --out bfs/data/ops_fx.npz
"""
import os, sys, argparse, time
import numpy as np
import scipy.sparse as sp

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]


def read_mesh(folder):
    import h5py
    m = h5py.File(os.path.join(folder, 'mesh.h5'), 'r')
    p, t = m['data0'][:], m['data1'][:]
    f = h5py.File(os.path.join(folder, 'facets.h5'), 'r')
    e, lab = f['data1'][:], f['data2'][:]
    return p, t, e, lab


def assemble(folder, baseflow, Re=600.0):
    import skfem as fe
    from scipy.spatial import cKDTree
    p, t, e, lab = read_mesh(folder)
    mesh = fe.MeshTri(p.T.copy(), t.T.copy())
    eu, ep = fe.ElementVector(fe.ElementTriP2()), fe.ElementTriP1()
    bu = fe.Basis(mesh, eu, intorder=5)
    bp = fe.Basis(mesh, ep, intorder=5)
    bs = fe.Basis(mesh, fe.ElementTriP2(), intorder=5)        # scalar P2: the base flow's components
    nu = 1.0 / Re
    # base flow at the P2 dof locations (the archive stores it at dof coordinates)
    z = np.load(os.path.join(folder, baseflow))
    X = bs.doflocs.T
    d, idx = cKDTree(z['baseflow_xy2']).query(X)
    if d.max() > 1e-6:
        raise RuntimeError(f"base flow does not match the P2 dofs (max distance {d.max():.1e})")
    Ub = np.vstack([z['baseflow_u'][idx], z['baseflow_v'][idx]])    # (2, nP2)
    U0 = bs.interpolate(Ub[0]); V0 = bs.interpolate(Ub[1])
    gU = np.stack([U0.grad, V0.grad])                               # gU[i, j] = d U_i / d x_j
    Uv = np.stack([U0.value, V0.value])

    @fe.BilinearForm
    def a_uu(u, v, w):
        # (U0.grad u + u.grad U0, v) + nu (grad u, grad v)
        conv = np.einsum('ij...,j...->i...', u.grad, w['Uv']) + np.einsum('ij...,j...->i...', w['gU'], u.value)
        return np.einsum('i...,i...', conv, v.value) + nu * np.einsum('ij...,ij...', u.grad, v.grad)

    @fe.BilinearForm
    def b_pu(p_, v, w):                     # -(p, div v)    rows: velocity test, cols: pressure trial
        return -p_.value * (v.grad[0, 0] + v.grad[1, 1])

    @fe.BilinearForm
    def m_uu(u, v, w):
        return np.einsum('i...,i...', u.value, v.value)

    t0 = time.time()
    Kuu = a_uu.assemble(bu, Uv=Uv, gU=gU)
    Bvp = b_pu.assemble(bp, bu)             # trial pressure (cols), test velocity (rows)
    Muu = m_uu.assemble(bu)
    nu_, np_ = bu.N, bp.N
    # dF/dW = [[Kuu, Bvp], [Bvp^T, 0]]  (-(div u, q) = (column of -(p, div v))^T)
    dF = sp.bmat([[Kuu, Bvp], [Bvp.T, None]], format='csr')
    Mfull = sp.bmat([[Muu, None], [None, sp.csr_matrix((np_, np_))]], format='csr')
    # dof bookkeeping: component (0 = u, 1 = v, 2 = p) and coordinates of every dof
    comp = np.concatenate([np.tile([0, 1], nu_ // 2), np.full(np_, 2)])
    # ElementVector interleaves the components: dof 2k is u, 2k + 1 is v at location k
    xy_u = np.repeat(bs.doflocs.T, 2, axis=0)
    xy = np.vstack([xy_u, bp.doflocs.T])
    # Dirichlet dofs from the labelled boundary edges
    bfac = mesh.boundary_facets()
    fv = np.sort(mesh.facets[:, bfac].T, axis=1)
    key = {tuple(r): f for r, f in zip(fv, bfac)}
    def facets_of(L):
        out = []
        for a_, b_ in np.sort(e[lab == L], axis=1):
            f = key.get((a_, b_))
            if f is None:
                raise RuntimeError(f"label {L} edge ({a_}, {b_}) is not a boundary facet")
            out.append(f)
        return np.array(out)
    def vel_dofs(facets, comps):
        D = bu.get_dofs(facets=facets)
        names = {0: 'u^1', 1: 'u^2'}
        return np.concatenate([D.nodal[names[c]] for c in comps] + [D.facet[names[c]] for c in comps])
    bc = np.unique(np.concatenate([vel_dofs(facets_of(1), (0, 1)), vel_dofs(facets_of(2), (0, 1)),
                                   vel_dofs(facets_of(4), (1,))]))
    n = dF.shape[0]
    keep = np.ones(n); keep[bc] = 0.0
    Z = sp.diags(keep)
    E = sp.diags(1.0 - keep)
    minusJ = (Z @ dF @ Z + E).tocsr()          # Dirichlet rows/cols -> identity
    Mbc = (Z @ Mfull @ Z).tocsr()               # Dirichlet rows/cols -> 0
    A = (-minusJ).tocsr()
    A.eliminate_zeros(); Mbc.eliminate_zeros()
    print(f"assembled n = {n} ({nu_} velocity, {np_} pressure dofs), {len(bc)} Dirichlet dofs, "
          f"nnz(A) = {A.nnz}, nnz(M) = {Mbc.nnz} ({time.time() - t0:.1f} s)", flush=True)
    return dict(A=A, M=Mbc, Mfull=Mfull.tocsr(), comp=comp, xy=xy, bc=bc, Re=Re,
                tri=t, pts=p, edges=e, edge_labels=lab, Ub=Ub, xy_p2=bs.doflocs.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='bfs/archive_dsr/fx_data')
    ap.add_argument('--baseflow', default='baseflow_fenicsx.npz')
    ap.add_argument('--Re', type=float, default=600.0)
    ap.add_argument('--out', default='bfs/data/ops_fx.npz')
    a = ap.parse_args()
    ops = assemble(a.data, a.baseflow, a.Re)
    from bfs.plant_io import save_ops
    save_ops(a.out, ops, source=f'scikit-fem, {a.data}/{a.baseflow}, Re {a.Re}')


if __name__ == '__main__':
    main()
