"""
Optimal harmonic gains of the backward-facing step flow (Dergham, Sipp & Robinet 2013, fig. 2-3),
built from the Jacobian J and mass matrix M assembled on the linear-analysis mesh:

    R(omega) = (i omega M - J)^-1 ,   J = -dF/dW (F: steady NS residual, nsforms.py) at the base flow
    P^T R^H Q R P f = lambda^2 Qf f   (energy norms, SLEPc Krylov-Schur, generalized Hermitian)

Usage:  python resolvent_gains.py --baseflow fx_data/baseflow_fenicsx.npz --omegas 0.25 0.75 2 3 --nev 12
Writes gains_<tag>.txt and modes_<tag>.npz (u and v of forcings/responses at P2 dof coordinates).
"""
import argparse, time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
import nsforms as ns

parser = argparse.ArgumentParser()
parser.add_argument("--baseflow", default="fx_data/baseflow_fenicsx.npz")
parser.add_argument("--tag", default="fx")
parser.add_argument("--Re", type=float, default=600.0)
parser.add_argument("--omegas", type=float, nargs="+", default=[0.25, 0.75, 1.5, 2.0, 3.0, 4.0])
parser.add_argument("--nev", type=int, default=12)
parser.add_argument("--data", default="fx_data")
args = parser.parse_args()
comm = MPI.COMM_WORLD
def log(*a):
    if comm.rank == 0: print(*a, flush=True)

msh, ft = ns.read_xdmf_mesh(comm, args.data)
W, F = ns.taylor_hood(msh)
subs = ns.scalar_subspaces(W)
wb = ns.load_field(W, np.load(args.baseflow), "baseflow", subs)
nu = fem.Constant(msh, PETSc.ScalarType(1.0 / args.Re))
bcs = ns.small_domain_bcs(W, ft)
rows = ns.owned_bc_rows(W, bcs)

# ---------------------------------------------------------------- Jacobian and mass matrix
# minusJ = dF/dW at the base flow (Dirichlet rows/cols -> identity); Mbc: mass with Dirichlet rows/cols zeroed
t0 = time.time()
minusJ = assemble_matrix(fem.form(ns.jacobian(wb, nu)), bcs=bcs, diag=1.0); minusJ.assemble()
Mbc = assemble_matrix(fem.form(ns.mass(W)), bcs=bcs, diag=0.0); Mbc.assemble()
Q = assemble_matrix(fem.form(ns.mass(W))); Q.assemble()                    # energy norm (velocity mass)
f, g = ufl.TrialFunction(F), ufl.TestFunction(F)
(_, _), (v, q) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
Qf = assemble_matrix(fem.form(ufl.inner(f, g) * ufl.dx)); Qf.assemble()    # forcing norm
P = assemble_matrix(fem.form(ufl.inner(f, v) * ufl.dx)); P.assemble()      # forcing -> momentum rhs
log(f"J, M assembled: {W.dofmap.index_map.size_global} dofs, forcing space {F.dofmap.index_map.size_global*2} "
    f"({time.time()-t0:.1f} s)")

A = minusJ.copy()
ksp = ns.lu_solver(A, comm)

def zero_bc(x):
    a = x.getArray(); a[rows] = 0.0; x.setArray(a)

class Op:  # y = P^T R^H Q R P x
    def __init__(self):
        self.t1, self.t2 = Q.createVecRight(), Q.createVecRight()
    def response(self, x, out):
        P.mult(x, self.t1); zero_bc(self.t1); ksp.solve(self.t1, out)
    def mult(self, mat, x, y):
        self.response(x, self.t2)
        Q.mult(self.t2, self.t1); zero_bc(self.t1)
        self.t1.conjugate(); ksp.solveTranspose(self.t1, self.t2); self.t2.conjugate()   # R^H
        P.multTranspose(self.t2, y)
op = Op()
L = PETSc.Mat().createPython(Qf.getSizes(), op, comm=comm); L.setUp()

(X2, du), (_, dv), _ = subs
Fu, _ = F.sub(0).collapse(); Fv, _ = F.sub(1).collapse()
fu_map = np.asarray(F.sub(0).collapse()[1]).ravel(); fv_map = np.asarray(F.sub(1).collapse()[1]).ravel()
fX = Fu.tabulate_dof_coordinates()[:, :2]
out, modes = [], {"resp_xy": X2, "forc_xy": fX}
fvec, rvec = fem.Function(F), fem.Function(W)
for om in args.omegas:
    t0 = time.time()
    A.zeroEntries(); A.axpy(1.0, minusJ, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
    A.axpy(1j * om, Mbc, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)      # i omega M - J
    ksp.setOperators(A)
    eps = SLEPc.EPS().create(comm)
    eps.setOperators(L, Qf); eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    eps.setDimensions(args.nev); eps.setTolerances(1e-10, 1000)
    st = eps.getST(); st.setType("shift"); sk = st.getKSP(); sk.setType("preonly")
    sk.getPC().setType("cholesky"); sk.getPC().setFactorSolverType("mumps")
    eps.solve()
    lam = np.array([eps.getEigenvalue(i).real for i in range(eps.getConverged())])
    order = np.argsort(-lam)[:args.nev]
    out.append([om] + list(lam[order]))
    log(f"omega = {om:5.2f}: lambda^2 = " + " ".join(f"{l:.4e}" for l in lam[order][:8]) + f"  ({time.time()-t0:.0f} s)")
    for k, i in enumerate(order):
        eps.getEigenvector(int(i), fvec.x.petsc_vec)
        op.response(fvec.x.petsc_vec, rvec.x.petsc_vec)
        r_u, r_v = rvec.x.array[du].copy(), rvec.x.array[dv].copy()
        ph = np.exp(-1j * np.angle(r_u[np.argmax(np.abs(r_u))]))          # phase: max |u| response real > 0
        modes[f"w{om}_k{k}_resp_u"] = r_u * ph
        modes[f"w{om}_k{k}_resp_v"] = r_v * ph
        modes[f"w{om}_k{k}_forc_u"] = fvec.x.array[fu_map] * ph
        modes[f"w{om}_k{k}_forc_v"] = fvec.x.array[fv_map] * ph
    eps.destroy()
np.savetxt(f"gains_{args.tag}.txt", np.array(out), header="omega  lambda_1^2 ... lambda_n^2")
np.savez(f"modes_{args.tag}.npz", **modes)
log(f"written gains_{args.tag}.txt, modes_{args.tag}.npz")
