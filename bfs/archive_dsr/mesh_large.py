"""
Large domain for the base flow (Dergham, Sipp & Robinet, JFM 2013, section 2.1):
x in [-20, 100], y in [0, 20]; flat plate y = 1 up to x = 0, circular arc (centre (0,-1.5),
R = 2.5, tangent to the plate) down to (2, 0), then y = 0.

Boundary tags
  1 inlet x=-20         u = (1, 0)
  2 plate, -20<x<-2     free slip (v = 0)
  3 walls (plate -2<x<0, arc, y=0)   no slip
  4 outlet x=100        natural (traction free)
  5 top y=20            symmetry (v = 0)
Refinement mimics the mesh used for the linear analysis ([-1.5,30]x[0,3.5]).
"""
import sys
import gmsh

scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0   # >1 coarsens, <1 refines
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("bfs_large")
g = gmsh.model.geo
p = [g.addPoint(*xy, 0) for xy in [(-20, 1), (-2, 1), (0, 1), (2, 0), (100, 0), (100, 20), (-20, 20)]]
c = g.addPoint(0, -1.5, 0)
l_slip = g.addLine(p[0], p[1])
l_plate = g.addLine(p[1], p[2])
arc = g.addCircleArc(p[2], c, p[3])
l_bot = g.addLine(p[3], p[4])
l_out = g.addLine(p[4], p[5])
l_top = g.addLine(p[5], p[6])
l_in = g.addLine(p[6], p[0])
loop = g.addCurveLoop([l_slip, l_plate, arc, l_bot, l_out, l_top, l_in])
surf = g.addPlaneSurface([loop])
g.synchronize()
for tag, curves in {1: [l_in], 2: [l_slip], 3: [l_plate, arc, l_bot], 4: [l_out], 5: [l_top]}.items():
    gmsh.model.addPhysicalGroup(1, curves, tag)
gmsh.model.addPhysicalGroup(2, [surf], 1)

f = gmsh.model.mesh.field
def box(x0, x1, y0, y1, hin, hout, th):
    b = f.add("Box")
    for k, v in dict(VIn=hin * scale, VOut=hout * scale, XMin=x0, XMax=x1, YMin=y0, YMax=y1, Thickness=th).items():
        f.setNumber(b, k, v)
    return b
fields = [
    box(-2.3, 3.0, 0.8, 1.2, 0.02, 3.0, 0.5),     # step corner / separated shear layer origin
    box(-1.5, 12.0, 0.0, 1.5, 0.035, 3.0, 0.8),   # recirculation bubble
    box(-3.0, 30.0, 0.0, 1.8, 0.10, 3.0, 1.5),    # shear layer / boundary layer
    box(-5.0, 30.0, 0.0, 3.5, 0.20, 3.0, 3.0),    # rest of the linear-analysis domain
    box(30.0, 100.0, 0.0, 3.0, 0.30, 3.0, 3.0),   # downstream boundary layer
]
d = f.add("Distance"); f.setNumbers(d, "PointsList", [p[1]])   # slip / no-slip junction
t = f.add("Threshold")
for k, v in dict(InField=d, SizeMin=0.01 * scale, SizeMax=3.0 * scale, DistMin=0.05, DistMax=1.0).items():
    f.setNumber(t, k, v)
fields.append(t)
mn = f.add("Min"); f.setNumbers(mn, "FieldsList", fields)
f.setAsBackgroundMesh(mn)
for opt in ["Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"]:
    gmsh.option.setNumber(opt, 0)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.model.mesh.generate(2)
gmsh.write("large.msh")
_, nodes, _ = gmsh.model.mesh.getNodes()
_, els, _ = gmsh.model.mesh.getElements(2)
print(f"large mesh: {len(nodes)//3} vertices, {len(els[0])} triangles")
gmsh.finalize()
