"""plant_io.py -- the operator file shared by the two assemblies (BFS.md 1).

Written by bfs/ops_dolfinx.py (the user's nsforms.py), read by bfs/plant.py.
One .npz:

  A, M        real CSR (n x n): A = J = -dF/dW at the base flow with the
              Dirichlet rows/columns set to -identity; M = velocity mass with
              the Dirichlet rows/columns (and the pressure block) zero
  comp        (n,) 0 = u, 1 = v, 2 = p           xy  (n, 2) dof coordinates
  bc          Dirichlet dof indices
  pts, tri    mesh vertices / triangles; edges, edge_labels (boundary geometry
              for the actuator / sensor position constraints)
  L           the velocity stiffness (grad u, grad v) with the Dirichlet rows/
              columns identity: the H^1 seminorm of the sensor tail bound
  shear_S, shear_x, shear_w
              the floor's wall-shear trace (ops_dolfinx.floor_shear_trace):
              (S q)_k = du/dy at (x_k, 0), Gauss weights w_k
  Re, source  provenance
"""
import numpy as np
import scipy.sparse as sp


def _put(d, name, S):
    S = sp.csr_matrix(S)
    d[name + '_data'], d[name + '_indices'], d[name + '_indptr'] = S.data, S.indices, S.indptr
    d[name + '_shape'] = np.array(S.shape)


def _get(z, name):
    return sp.csr_matrix((z[name + '_data'], z[name + '_indices'], z[name + '_indptr']),
                         shape=tuple(z[name + '_shape']))


def save_ops(path, ops, source=''):
    d = {}
    for k in ('A', 'M', 'L', 'shear_S'):
        _put(d, k, ops[k])
    for k in ('comp', 'xy', 'bc', 'pts', 'tri', 'edges', 'edge_labels', 'shear_x', 'shear_w'):
        d[k] = np.asarray(ops[k])
    d['Re'] = float(ops['Re']); d['source'] = np.array(source)
    np.savez(path, **d)
    print(f"-> {path}", flush=True)


def load_ops(path):
    z = np.load(path, allow_pickle=False)
    out = {k: _get(z, k) for k in ('A', 'M', 'L', 'shear_S')}
    for k in ('comp', 'xy', 'bc', 'pts', 'tri', 'edges', 'edge_labels', 'shear_x', 'shear_w'):
        out[k] = z[k]
    out['Re'] = float(z['Re']); out['source'] = str(z['source'])
    return out
