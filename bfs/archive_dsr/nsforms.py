"""
Shared pieces: Taylor-Hood space, steady Navier-Stokes residual, its Jacobian (used both by
Newton and, evaluated at the base flow, as the linear operator of the resolvent) and mass matrix.

Convention:  M dx/dt = J x + f,   J = - dF/dW (F = steady NS residual below),
             resolvent  R(omega) = (i omega M - J)^-1
"""
import numpy as np
import ufl, basix.ufl
from petsc4py import PETSc
from scipy.spatial import cKDTree
from dolfinx import fem, io

inner, grad, div, dot, dx = ufl.inner, ufl.grad, ufl.div, ufl.dot, ufl.dx


def taylor_hood(msh):
    P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(2,))
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    return fem.functionspace(msh, basix.ufl.mixed_element([P2, P1])), fem.functionspace(msh, P2)


def residual(w, nu):
    """Steady NS residual F(W; v, q) = (U.grad U, v) + nu (grad U, grad v) - (P, div v) - (div U, q)."""
    W = w.function_space
    U, P = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    return (inner(dot(grad(U), U), v) + nu * inner(grad(U), grad(v))
            - inner(P, div(v)) - inner(div(U), q)) * dx


def jacobian(w, nu):
    """dF/dW at w  (the Newton matrix).  The linearised NS operator is J = -jacobian."""
    return ufl.derivative(residual(w, nu), w, ufl.TrialFunction(w.function_space))


def mass(W):
    (u, p), (v, q) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
    return inner(u, v) * dx


# ---------------------------------------------------------------- I/O helpers
def read_xdmf_mesh(comm, folder):
    with io.XDMFFile(comm, f"{folder}/mesh.xdmf", "r") as f:
        msh = f.read_mesh(name="Grid")
    msh.topology.create_connectivity(1, 2)
    with io.XDMFFile(comm, f"{folder}/facets.xdmf", "r") as f:
        ft = f.read_meshtags(msh, name="Grid")
    return msh, ft


def scalar_subspaces(W):
    out = []
    for sub in (W.sub(0).sub(0), W.sub(0).sub(1), W.sub(1)):
        Vc, dofs = sub.collapse()
        out.append((Vc.tabulate_dof_coordinates()[:, :2], np.asarray(dofs).ravel()))
    return out


def load_field(W, data, name, subs=None):
    """Function in W from a FreeFEM-style export (values at dof coordinates, see freefem_to_fenicsx.py)."""
    subs = subs or scalar_subspaces(W)
    w = fem.Function(W)
    for (X, dofs), comp, xy in zip(subs, ("u", "v", "p"), ("xy2", "xy2", "xy1")):
        d, idx = cKDTree(data[f"{name}_{xy}"]).query(X)
        if d.max() > 1e-6:
            raise RuntimeError(f"{name}.{comp}: dof coordinates do not match (max dist {d.max():.2e})")
        w.x.array[dofs] = data[f"{name}_{comp}"][idx]
    w.x.scatter_forward()
    return w


def export_field(w, name, subs=None):
    """Inverse of load_field (serial): dict of arrays at dof coordinates."""
    subs = subs or scalar_subspaces(w.function_space)
    (X2, du), (_, dv), (X1, dp) = subs
    a = w.x.array
    return {f"{name}_xy2": X2, f"{name}_u": a[du], f"{name}_v": a[dv], f"{name}_xy1": X1, f"{name}_p": a[dp]}


def small_domain_bcs(W, ft):
    """Perturbation BCs of the paper / svd2.edp: u=v=0 on labels 1 (inlet) and 2 (walls), v=0 on 4 (top)."""
    V0, _ = W.sub(0).collapse()
    V01, _ = W.sub(0).sub(1).collapse()
    d12 = fem.locate_dofs_topological((W.sub(0), V0), 1, np.concatenate([ft.find(1), ft.find(2)]))
    d4 = fem.locate_dofs_topological((W.sub(0).sub(1), V01), 1, ft.find(4))
    return [fem.dirichletbc(fem.Function(V0), d12, W.sub(0)),
            fem.dirichletbc(fem.Function(V01), d4, W.sub(0).sub(1))]


def owned_bc_rows(W, bcs):
    n = W.dofmap.index_map.size_local
    r = np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs]))
    return r[r < n].astype(np.int32)


def lu_solver(A, comm):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC(); pc.setType("lu"); pc.setFactorSolverType("mumps")
    ksp.setFromOptions()
    return ksp
