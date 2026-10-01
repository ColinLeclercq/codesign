#!/usr/bin/env python3
"""mesh_coarse.py -- a COARSE backward-facing-step test case for the sandbox
(BFS.md 6): the same domain as the linear-analysis mesh ([-1.5, 30] x [0, 3.5],
plate y = 1 for x < 0, arc of centre (0, -1.5), R = 2.5 down to (2, 0), y = 0
beyond), meshed by mapping a graded rectangle (x, t) to (x, wall(x) + t (3.5 -
wall(x))), and the archive's base flow interpolated onto its P2 nodes.

The interpolated base flow is not a steady solution on the coarse mesh: the
operators are a realistic TEST plant (same structure, size n ~ 3e4 instead of
1.9e5), not a model of the flow.  Everything that matters for a physical answer
runs on the user's mesh and base flow.

    python bfs/mesh_coarse.py --nx 140 --nt 26 --out bfs/data/coarse
    python bfs/ops_dolfinx.py --data bfs/data/coarse --baseflow baseflow.npz --out bfs/data/ops_coarse.npz
"""
import os, sys, argparse
import numpy as np
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]


def wall(x):
    x = np.asarray(x, float)
    return np.where(x <= 0.0, 1.0, np.where(x >= 2.0, 0.0, -1.5 + np.sqrt(np.clip(6.25 - x ** 2, 0, None))))


def graded(a, b, n, fine_at, ratio):
    """n + 1 points on [a, b], spacing growing geometrically away from fine_at."""
    s = np.linspace(0.0, 1.0, n + 1)
    c = (fine_at - a) / (b - a)
    k = np.log(ratio)
    f = np.where(s < c, c - c * np.expm1(k * (c - s) / max(c, 1e-12)) / np.expm1(k) if c > 0 else s,
                 c + (1 - c) * np.expm1(k * (s - c) / max(1 - c, 1e-12)) / np.expm1(k))
    return a + (b - a) * f


def build(nx, nt):
    xs = graded(-1.5, 30.0, nx, 0.5, 12.0)
    ts = graded(0.0, 1.0, nt, 0.0, 6.0)
    X, T = np.meshgrid(xs, ts, indexing='ij')
    Y = wall(X) + T * (3.5 - wall(X))
    pts = np.column_stack([X.ravel(), Y.ravel()])
    idx = np.arange(pts.shape[0]).reshape(nx + 1, nt + 1)
    tri = []
    for i in range(nx):
        for j in range(nt):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            tri += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    tri = np.array(tri)
    edges, labels = [], []
    for j in range(nt):
        edges.append([idx[0, j], idx[0, j + 1]]); labels.append(1)          # inlet
        edges.append([idx[nx, j], idx[nx, j + 1]]); labels.append(3)        # outlet
    for i in range(nx):
        edges.append([idx[i, 0], idx[i + 1, 0]]); labels.append(2)          # wall
        edges.append([idx[i, nt], idx[i + 1, nt]]); labels.append(4)        # top
    return pts, tri, np.array(edges), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=140)
    ap.add_argument('--nt', type=int, default=26)
    ap.add_argument('--fine', default='bfs/archive_dsr/fx_data/baseflow_fenicsx.npz')
    ap.add_argument('--out', default='bfs/data/coarse')
    a = ap.parse_args()
    import meshio
    from scipy.interpolate import LinearNDInterpolator
    pts, tri, edges, labels = build(a.nx, a.nt)
    os.makedirs(a.out, exist_ok=True)
    # the archive's format (freefem_to_fenicsx.py), read by nsforms.read_xdmf_mesh
    meshio.write(os.path.join(a.out, 'mesh.xdmf'),
                 meshio.Mesh(points=pts, cells=[('triangle', tri)], cell_data={'region': [np.zeros(len(tri), np.int32)]}))
    meshio.write(os.path.join(a.out, 'facets.xdmf'),
                 meshio.Mesh(points=pts, cells=[('line', edges)], cell_data={'label': [labels.astype(np.int32)]}))
    z = np.load(a.fine)
    # P2 nodes of the straight-sided triangles: vertices and edge midpoints; P1: vertices
    ed = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1), axis=0)
    X1 = pts
    X2 = np.vstack([pts, 0.5 * (pts[ed[:, 0]] + pts[ed[:, 1]])])
    out = dict(baseflow_xy2=X2, baseflow_xy1=X1)
    for k, xy, src in (('u', X2, 'xy2'), ('v', X2, 'xy2'), ('p', X1, 'xy1')):
        f = LinearNDInterpolator(z[f'baseflow_{src}'], z[f'baseflow_{k}'])
        val = f(xy)
        bad = ~np.isfinite(val)
        if bad.any():                    # outside the fine mesh's hull by rounding: nearest value
            from scipy.spatial import cKDTree
            _, j = cKDTree(z[f'baseflow_{src}']).query(xy[bad]); val[bad] = z[f'baseflow_{k}'][j]
        out[f'baseflow_{k}'] = val
    np.savez(os.path.join(a.out, 'baseflow.npz'), **out)
    print(f"coarse BFS mesh: {len(pts)} vertices, {len(tri)} triangles -> {a.out}")


if __name__ == '__main__':
    main()
