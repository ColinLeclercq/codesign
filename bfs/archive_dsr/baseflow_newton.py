"""
Base flow of the rounded backward-facing step (Dergham, Sipp & Robinet, JFM 2013, section 2.1):
steady Navier-Stokes solved by Newton on the large domain (mesh_large.py), with continuation
in Re, then interpolated onto the linear-analysis mesh [-1.5,30]x[0,3.5] (fx_data/).

Output: fx_data/baseflow_fenicsx.npz  (same layout as fields.npz: values at dof coordinates)
        large_baseflow.npz             (large-domain solution, same layout)
"""
import argparse, time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import nsforms as ns

parser = argparse.ArgumentParser()
parser.add_argument("--Re", type=float, nargs="+", default=[50, 150, 300, 450, 600])
parser.add_argument("--mesh", default="large.msh")
parser.add_argument("--small", default="fx_data")
parser.add_argument("--tol", type=float, default=1e-11)
args = parser.parse_args()
comm = MPI.COMM_WORLD
def log(*a):
    if comm.rank == 0: print(*a, flush=True)

md = gmshio.read_from_msh(args.mesh, comm, 0, gdim=2)
msh, ft = md.mesh, md.facet_tags
W, _ = ns.taylor_hood(msh)
log(f"large mesh: {msh.topology.index_map(2).size_global} triangles, {W.dofmap.index_map.size_global} dofs")

# ------------------------------------------------ boundary conditions (tags from mesh_large.py)
V0, _ = W.sub(0).collapse()
V1, _ = W.sub(0).sub(1).collapse()
uin = fem.Function(V0); uin.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
zero0, zero1 = fem.Function(V0), fem.Function(V1)
def dofs(space, tags):
    return fem.locate_dofs_topological(space, 1, np.concatenate([ft.find(t) for t in tags]))
d_in = dofs((W.sub(0), V0), [1])
bcs = [fem.dirichletbc(uin,   d_in, W.sub(0)),                                   # inlet u = 1, v = 0
       fem.dirichletbc(zero0, dofs((W.sub(0), V0), [3]), W.sub(0)),              # no-slip walls
       fem.dirichletbc(zero1, dofs((W.sub(0).sub(1), V1), [2, 5]), W.sub(0).sub(1))]  # free slip / symmetry: v = 0
# homogeneous version for the Newton increment
bcs0 = [fem.dirichletbc(fem.Function(V0), d_in, W.sub(0)), bcs[1], bcs[2]]
rows = ns.owned_bc_rows(W, bcs)

# ------------------------------------------------ Newton
w = fem.Function(W)
w.sub(0).interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))   # uniform flow
for bc in bcs:
    bc.set(w.x.array)
nu = fem.Constant(msh, PETSc.ScalarType(1.0 / args.Re[0]))
F = fem.form(ns.residual(w, nu))
Jf = fem.form(ns.jacobian(w, nu))
dw = fem.Function(W)
A = assemble_matrix(Jf, bcs=bcs0); A.assemble()
ksp = ns.lu_solver(A, comm)

t0 = time.time()
for Re in args.Re:
    nu.value = 1.0 / Re
    for it in range(30):
        b = assemble_vector(F)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b.array[rows] = 0.0                          # w already satisfies the Dirichlet values
        res = b.norm()
        log(f"  Re = {Re:5.0f}  it {it:2d}  |F| = {res:.3e}")
        if res < args.tol:
            break
        A.zeroEntries(); assemble_matrix(A, Jf, bcs=bcs0); A.assemble()
        ksp.setOperators(A)
        b.scale(-1.0)
        ksp.solve(b, dw.x.petsc_vec)
        dw.x.scatter_forward()
        # damped Newton: halve the step until the residual decreases
        w0, step = w.x.array.copy(), 1.0
        while True:
            w.x.array[:] = w0 + step * dw.x.array
            bn = assemble_vector(F)
            bn.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            bn.array[rows] = 0.0
            if bn.norm() < res or step < 1e-3:
                break
            step *= 0.5
        if step < 1:
            log(f"      damping {step}")
    else:
        raise RuntimeError("Newton did not converge")
log(f"Newton done in {time.time() - t0:.0f} s")
np.savez("large_baseflow.npz", **ns.export_field(w, "baseflow"))

# ------------------------------------------------ interpolate onto the linear-analysis mesh
msh_s, _ = ns.read_xdmf_mesh(comm, args.small)
Ws, _ = ns.taylor_hood(msh_s)
ws = fem.Function(Ws)
cells = np.arange(msh_s.topology.index_map(2).size_local, dtype=np.int32)
for sub_l, sub_s in [(w.sub(0).sub(0), Ws.sub(0).sub(0)), (w.sub(0).sub(1), Ws.sub(0).sub(1)), (w.sub(1), Ws.sub(1))]:
    src = sub_l.collapse()                         # scalar component on the large mesh
    Vs, dmap = sub_s.collapse()
    dst = fem.Function(Vs)
    dst.interpolate_nonmatching(src, cells, fem.create_interpolation_data(Vs, src.function_space, cells, 1e-10))
    ws.x.array[dmap] = dst.x.array
np.savez(f"{args.small}/baseflow_fenicsx.npz", **ns.export_field(ws, "baseflow"))
log(f"written {args.small}/baseflow_fenicsx.npz")
