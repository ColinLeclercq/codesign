"""Compare the FEniCSx base flow (fx_data/baseflow_fenicsx.npz) with FreeFEM cb_newton (fx_data/fields.npz)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.spatial import cKDTree

tok = open("mesh.msh").read().split(); nv, nt, nb = map(int, tok[:3]); a = np.array(tok[3:], float)
V = a[:3*nv].reshape(nv, 3)[:, :2]; T = a[3*nv:3*nv+4*nt].reshape(nt, 4)[:, :3].astype(int) - 1
B = a[3*nv+4*nt:].reshape(nb, 3).astype(int)
tri = mtri.Triangulation(V[:, 0], V[:, 1], T)

def at_vertices(f):
    i2 = cKDTree(f["baseflow_xy2"]).query(V)[1]; i1 = cKDTree(f["baseflow_xy1"]).query(V)[1]
    return f["baseflow_u"].ravel()[i2].real, f["baseflow_v"].ravel()[i2].real, f["baseflow_p"].ravel()[i1].real
ff = at_vertices(np.load("fx_data/fields.npz"))
fx = at_vertices(np.load("fx_data/baseflow_fenicsx.npz"))

# P2-exact norms on the P2 dofs (same dof sets, matched by coordinates)
A, Bf = np.load("fx_data/fields.npz"), np.load("fx_data/baseflow_fenicsx.npz")
j = cKDTree(Bf["baseflow_xy2"]).query(A["baseflow_xy2"])[1]
du = np.abs(A["baseflow_u"].ravel() - Bf["baseflow_u"].ravel()[j]); dv = np.abs(A["baseflow_v"].ravel() - Bf["baseflow_v"].ravel()[j])
print(f"max |dU| = {du.max():.2e}   max |dV| = {dv.max():.2e}   rms |dU| = {np.sqrt((du**2).mean()):.2e}")
k = np.argmax(du); print("  largest dU at", A["baseflow_xy2"][k])

def diagnostics(u, v, name):
    fu, fv = mtri.LinearTriInterpolator(tri, u), mtri.LinearTriInterpolator(tri, v)
    wall = np.unique(B[B[:, 2] == 2][:, :2] - 1)
    wall = wall[np.argsort(V[wall, 0] + 1e-3 * (1 - V[wall, 1]))]
    P = V[wall]; t = np.gradient(P, axis=0); t /= np.linalg.norm(t, axis=1)[:, None]
    n = np.c_[-t[:, 1], t[:, 0]]; n[n[:, 1] < 0] *= -1
    Q = P + 2e-3 * n
    ut = np.ma.filled(fu(Q[:, 0], Q[:, 1]), np.nan) * t[:, 0] + np.ma.filled(fv(Q[:, 0], Q[:, 1]), np.nan) * t[:, 1]
    ok = np.isfinite(ut) & (P[:, 0] > -1.4); P, ut = P[ok], ut[ok]
    s = np.where(np.diff(np.sign(ut)) != 0)[0]
    pts = [P[i] + (P[i+1] - P[i]) * ut[i] / (ut[i] - ut[i+1]) for i in s]
    y = np.linspace(1, 3.5, 4001); uu = np.ma.filled(fu(np.zeros_like(y), y), np.nan)
    dstar = np.trapezoid(1 - uu / uu[-1], y)
    ymin = min(np.ma.filled(fu(np.full(2000, x), np.linspace(0, 1.5, 2000)), 9).min() for x in np.linspace(2, 12, 101))
    print(f"{name:10s} separation (x,y) = {np.round(pts[0], 3)}, reattachment x = {pts[-1][0]:.3f}, "
          f"delta* at x=0: {dstar:.4f} (Re_delta* = {600*dstar:.1f}), min U = {ymin:.4f}")
    return pts
print("paper:     separation x ~ 0.6, reattachment x ~ 11, delta* = 0.082, Re_delta* = 49.2")
pff = diagnostics(ff[0], ff[1], "FreeFEM")
pfx = diagnostics(fx[0], fx[1], "FEniCSx")

div = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4a8a0", "#d9463b", "#8a1c14"])
fig, axs = plt.subplots(3, 1, figsize=(15, 7.6), constrained_layout=True)
ink, muted = "#1f1f1e", "#6b6b67"
for ax, f, title in [(axs[0], ff[0], "U  — FreeFEM (cb_newton.txt)"), (axs[1], fx[0], "U  — FEniCSx Newton (large domain, interpolated)")]:
    cs = ax.tricontourf(tri, f, levels=np.linspace(-0.14, 1.08, 41), cmap=div, norm=TwoSlopeNorm(0, -0.14, 1.08))
    ax.tricontour(tri, ff[0], levels=[0], colors="#ffffff", linewidths=2.2)
    ax.tricontour(tri, fx[0], levels=[0], colors=ink, linewidths=0.9, linestyles="--")
    ax.set_title(title + "   (U = 0: white FreeFEM, dashed FEniCSx)", loc="left", fontsize=11, color=ink)
    fig.colorbar(cs, ax=ax, shrink=0.9, pad=0.01, ticks=[-0.1, 0, 0.5, 1])
d = fx[0] - ff[0]; m = np.abs(d).max()
cs = axs[2].tricontourf(tri, d, levels=np.linspace(-m, m, 41), cmap=div)
fig.colorbar(cs, ax=axs[2], shrink=0.9, pad=0.01, format="%.0e")
axs[2].set_title(f"Difference U(FEniCSx) − U(FreeFEM)   (max {m:.1e})", loc="left", fontsize=11, color=ink)
for ax in axs:
    for e in B[B[:, 2] == 2][:, :2] - 1: ax.plot(V[e, 0], V[e, 1], color=ink, lw=1)
    ax.set_aspect("equal"); ax.set_xlim(-1.5, 30); ax.set_ylim(0, 3.5); ax.tick_params(colors=muted, labelsize=9)
fig.savefig("baseflow_comparison.png", dpi=120, facecolor="#fcfcfb")
