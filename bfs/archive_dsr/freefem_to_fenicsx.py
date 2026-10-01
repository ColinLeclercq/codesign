"""
Convert FreeFEM data for the FEniCSx resolvent script (run once, in serial).

Inputs
  mesh.msh                  FreeFEM mesh (readmesh format)
  ff_export/*_P2.txt, *_P1.txt   written by export_for_fenicsx.edp

Outputs (in fx_data/)
  mesh.xdmf/.h5             triangles
  facets.xdmf/.h5           boundary edges with their FreeFEM labels (1..5)
  fields.npz                base flow, actuator, sensor at FreeFEM dof coordinates

Requires: numpy, meshio, h5py
"""
import os, sys
import numpy as np
import meshio

msh_file = sys.argv[1] if len(sys.argv) > 1 else "mesh.msh"
exp_dir = sys.argv[2] if len(sys.argv) > 2 else "ff_export"
out = "fx_data"
os.makedirs(out, exist_ok=True)

# ---- FreeFEM mesh:  nv nt nbe / x y label (nv) / i j k region (nt) / i j label (nbe), 1-based
tok = open(msh_file).read().split()
nv, nt, nbe = map(int, tok[:3])
a = np.array(tok[3:], dtype=float)
verts = a[:3 * nv].reshape(nv, 3)
tris = a[3 * nv:3 * nv + 4 * nt].reshape(nt, 4).astype(np.int64)
edges = a[3 * nv + 4 * nt:3 * nv + 4 * nt + 3 * nbe].reshape(nbe, 3).astype(np.int64)
pts = verts[:, :2]

# orient triangles counter-clockwise (dolfinx does not need it, but it is harmless)
t = tris[:, :3] - 1
e1, e2 = pts[t[:, 1]] - pts[t[:, 0]], pts[t[:, 2]] - pts[t[:, 0]]
area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
t[area < 0] = t[area < 0][:, [0, 2, 1]]

meshio.write(os.path.join(out, "mesh.xdmf"),
             meshio.Mesh(points=pts, cells=[("triangle", t)],
                         cell_data={"region": [tris[:, 3].astype(np.int32)]}))
meshio.write(os.path.join(out, "facets.xdmf"),
             meshio.Mesh(points=pts, cells=[("line", edges[:, :2] - 1)],
                         cell_data={"label": [edges[:, 2].astype(np.int32)]}))
labels, counts = np.unique(edges[:, 2], return_counts=True)
print(f"mesh: {nv} vertices, {nt} triangles; edge labels {dict(zip(labels.tolist(), counts.tolist()))}")

# ---- fields exported at FreeFEM dof coordinates
data = {}
for name in ["baseflow", "actuator", "sensor"]:
    f2, f1 = os.path.join(exp_dir, name + "_P2.txt"), os.path.join(exp_dir, name + "_P1.txt")
    if not os.path.exists(f2):
        print(f"  ({name} not found in {exp_dir}, skipped)")
        continue
    A2, A1 = np.loadtxt(f2), np.loadtxt(f1)
    data[name + "_xy2"] = A2[:, :2]
    data[name + "_u"] = A2[:, 2] + 1j * A2[:, 3]
    data[name + "_v"] = A2[:, 4] + 1j * A2[:, 5]
    data[name + "_xy1"] = A1[:, :2]
    data[name + "_p"] = A1[:, 2] + 1j * A1[:, 3]
    print(f"  {name}: {len(A2)} P2 dofs, {len(A1)} P1 dofs")
np.savez(os.path.join(out, "fields.npz"), **data)
print("written to", out)
