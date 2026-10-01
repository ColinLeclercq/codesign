"""Figure 2 (gains at the circled points vs digitised paper values) and figure 3 (A1 forcing/response)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree

tag = "fx"
m = np.load(f"modes_{tag}.npz"); G = np.loadtxt(f"gains_{tag}.txt"); Gff = np.loadtxt("gains_ff.txt")
ink, muted, grid = "#1f1f1e", "#6b6b67", "#e4e4e0"

# ---------------- branch identification (see classify.py)
Xf, Xr = m["forc_xy"], m["resp_xy"]
shear_f = (Xf[:, 0] < 4) & (np.abs(Xf[:, 1] - 1.0) < 0.4); low_r = Xr[:, 1] < 1.2
def fracs(om, k):
    ef = np.abs(m[f"w{om}_k{k}_forc_u"])**2 + np.abs(m[f"w{om}_k{k}_forc_v"])**2
    er = np.abs(m[f"w{om}_k{k}_resp_u"])**2 + np.abs(m[f"w{om}_k{k}_resp_v"])**2
    return ef[shear_f].sum() / ef.sum(), er[low_r].sum() / er.sum()
branch = {}
for row in G:
    om = row[0]; fr = [fracs(om, k) for k in range(len(row) - 1)]
    A = [k for k, (a, b) in enumerate(fr) if b > 0.8 or a > 0.5]      # shear-layer family
    Bk = [k for k, (a, b) in enumerate(fr) if b < 0.35 and a < 0.1]   # free-stream family (ordered)
    a1 = max(range(len(fr)), key=lambda k: fr[k][0])
    branch[(om, "A1")] = a1
    rest = [k for k in A if k != a1]
    if rest: branch[(om, "A2")] = rest[0]
    for i, k in enumerate(Bk[:3]): branch[(om, f"B{i+1}")] = k

paper = {("A1", 0.25): 1.60e5, ("A1", 0.75): 2.22e6, ("A1", 2.0): 7.74e3, ("A1", 3.0): 170,
         ("A2", 0.75): 620, ("A2", 1.5): 215, ("B1", 2.0): 606, ("B1", 4.0): 231,
         ("B2", 2.0): 298, ("B3", 2.0): 226}
rows = []
print(f"{'branch':6s} {'omega':>5s} {'paper (digitised)':>18s} {'FEniCSx BF':>11s} {'FreeFEM BF':>11s} {'diff %':>7s} {'index':>5s}")
for (br, om), pv in paper.items():
    k = branch[(om, br)]
    i = list(G[:, 0]).index(om)
    v, vff = G[i, k + 1], Gff[i, k + 1]
    rows.append((br, om, pv, v, vff, k))
    print(f"{br:6s} {om:5.2f} {pv:18.4g} {v:11.4g} {vff:11.4g} {100*(v/pv-1):+7.1f} {k+1:5d}")

# ---------------- figure 2 comparison
fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
for row in G:
    ax.plot(np.full(len(row) - 1, row[0]), row[1:], "o", ms=4, color="#b8b8b3", zorder=1)
col = {"A1": "#d9463b", "A2": "#ec835a", "B1": "#256abf", "B2": "#3987e5", "B3": "#86b6ef"}
for br in col:
    pts = [(om, pv, v) for (b, om, pv, v, vff, k) in rows if b == br]
    if not pts: continue
    om, pv, v = map(np.array, zip(*pts))
    ax.plot(om, pv, "o", ms=12, mfc="none", mec=col[br], mew=2, zorder=2)
    ax.plot(om, v, "o", ms=5, color=col[br], zorder=3, label=br)
for (b, om, pv, v, vff, k) in rows:
    ax.annotate(f"{b}", (om, v), textcoords="offset points", xytext=(9, -3), fontsize=8, color=muted)
ax.set_yscale("log"); ax.set_xlim(0, 4.5); ax.set_ylim(30, 5e6)
ax.set_xlabel(r"$\omega$", color=ink); ax.set_ylabel(r"$\lambda_i^2$", color=ink)
ax.grid(True, which="major", color=grid, lw=0.8); ax.tick_params(colors=muted)
for s in ax.spines.values(): s.set_color("#c9c9c4")
ax.plot([], [], "o", ms=12, mfc="none", mec=ink, mew=1.5, label="paper (fig. 2, digitised)")
ax.plot([], [], "o", ms=4, color="#b8b8b3", label="other computed $\\lambda_i^2$ (12 per $\\omega$)")
ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
ax.set_title("Optimal gains at the circled points of fig. 2 (FEniCSx base flow + J, M)", loc="left", fontsize=10.5, color=ink)
fig.savefig("fig2_check.png", dpi=150, facecolor="#fcfcfb")

# ---------------- figure 3: P2 field on a 4-split triangulation
tok = open("mesh.msh").read().split(); nv, nt, nb = map(int, tok[:3]); a = np.array(tok[3:], float)
V = a[:3*nv].reshape(nv, 3)[:, :2]; T = a[3*nv:3*nv+4*nt].reshape(nt, 4)[:, :3].astype(int) - 1
tree_r, tree_f = cKDTree(Xr), cKDTree(Xf)
mid = lambda i, j: (V[T[:, i]] + V[T[:, j]]) / 2
pts = np.vstack([V, mid(0, 1), mid(1, 2), mid(2, 0)])
n0 = nv; e01, e12, e20 = n0 + np.arange(nt), n0 + nt + np.arange(nt), n0 + 2*nt + np.arange(nt)
T4 = np.vstack([np.c_[T[:, 0], e01, e20], np.c_[e01, T[:, 1], e12], np.c_[e20, e12, T[:, 2]], np.c_[e01, e12, e20]])
tri4 = mtri.Triangulation(pts[:, 0], pts[:, 1], T4)
ir, i_f = tree_r.query(pts)[1], tree_f.query(pts)[1]
cmap = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4a8a0", "#d9463b", "#8a1c14"])
th = np.linspace(0, np.arctan2(1.5, 2) + np.pi/2, 60)
step = np.vstack([[-1.5, 0], [-1.5, 1], [0, 1], np.c_[2.5*np.sin(np.linspace(0, np.arcsin(0.8), 40)), -1.5 + 2.5*np.cos(np.linspace(0, np.arcsin(0.8), 40))], [2, 0]])
fig, axs = plt.subplots(4, 2, figsize=(15, 5.6), constrained_layout=True)
for r, om in enumerate([0.25, 0.75, 2.0, 3.0]):
    k = branch[(om, "A1")]; lam2 = G[list(G[:, 0]).index(om), k + 1]
    for c, (what, idx) in enumerate([("forc", i_f), ("resp", ir)]):
        f = m[f"w{om}_k{k}_{what}_u"].real[idx]; s = np.abs(f).max()
        ax = axs[r, c]
        ax.tripcolor(tri4, f, cmap=cmap, vmin=-s, vmax=s, shading="gouraud", rasterized=True)
        ax.fill(step[:, 0], step[:, 1], color=ink)
        ax.set_xlim(-1.5, 30); ax.set_ylim(0, 3.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_color("#c9c9c4")
        lbl = "abcdefgh"[2*r + c]
        ax.set_title(f"({lbl}) {'forcing' if c == 0 else 'response'}  $\\omega$ = {om},  A1 = $\\lambda_{k+1}$,  $\\lambda^2$ = {lam2:.3g}",
                     loc="left", fontsize=9.5, color=ink)
fig.suptitle("Re of streamwise velocity of the A1 optimal forcing / response (compare with fig. 3 of the paper)",
             x=0.01, ha="left", fontsize=11.5, color=ink)
fig.savefig("fig3_check.png", dpi=130, facecolor="#fcfcfb")
