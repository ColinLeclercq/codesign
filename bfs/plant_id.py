#!/usr/bin/env python3
"""plant_id.py -- where the plant files of THIS code live (BFS.md 7).

    python bfs/plant_id.py coarse|fx            the id (16 hex digits)
    python bfs/plant_id.py --dir coarse|fx      the directory: <data root>/<kind>-<id>
    python bfs/plant_id.py --root OLD coarse    the id of another code directory's generators

The plant files (mesh, operators, certified range and poles) are built once
per id into their own directory and never rebuilt in place, so code versions
that share a data root (bfs/home.sh) never overwrite each other's files: a
version whose generators differ gets another directory (user, 2026-10-14;
the layout review of 20261010i).

The id hashes what the files are made from:
  * the generators and what they import (bfs/mesh_coarse.py for the test
    plant, bfs/ops_dolfinx.py, bfs/archive_dsr/nsforms.py, bfs/plant_io.py,
    bfs/plant_cert.py) -- their syntax trees without comments or docstrings,
    so an edit of a comment costs no rebuild;
  * the input data (the archive's mesh, facets and base flow);
  * RANGE_VERSION in bfs/plant.py: the certified range and poles are computed
    by bfs/plant.py, bfs/mfree.py and sparse_spectrum.py, which change for
    other reasons too, so their computation is versioned by hand -- bump
    RANGE_VERSION whenever it changes.

The data root: $BFS_HOME/data, where BFS_HOME is the environment variable or
the nearest ancestor of the code directory holding the marker file .bfs_home
(bfs/home.sh INIT=1); without one, <code directory>/bfs/data.
"""
import ast, hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)

PY = {'coarse': ['bfs/mesh_coarse.py', 'bfs/ops_dolfinx.py', 'bfs/archive_dsr/nsforms.py', 'bfs/plant_io.py',
                 'bfs/plant_cert.py'],
      'fx': ['bfs/ops_dolfinx.py', 'bfs/archive_dsr/nsforms.py', 'bfs/plant_io.py', 'bfs/plant_cert.py']}
DATA = {'coarse': ['bfs/archive_dsr/fx_data/baseflow_fenicsx.npz'],
        'fx': ['bfs/archive_dsr/fx_data/mesh.h5', 'bfs/archive_dsr/fx_data/facets.h5',
               'bfs/archive_dsr/fx_data/baseflow_fenicsx.npz']}


def _code_hash(path):
    """The syntax tree of a Python file without docstrings (comments are not in it)."""
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], 'value', None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def range_version(root):
    """RANGE_VERSION = <int> in bfs/plant.py, read without importing it."""
    for node in ast.parse(open(os.path.join(root, 'bfs/plant.py')).read()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'RANGE_VERSION' for t in node.targets):
            return int(node.value.value)
    return 1        # code from before the constant (20261010i and earlier) computes version 1


def plant_id(kind, root=CODE):
    parts = [f"kind={kind}", f"range_version={range_version(root)}"]
    parts += [f"{p}:{_code_hash(os.path.join(root, p))}" for p in PY[kind]]
    parts += [f"{p}:{_file_hash(os.path.join(root, p))}" for p in DATA[kind]]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def bfs_home(start=CODE):
    if os.environ.get('BFS_HOME'):
        return os.path.abspath(os.environ['BFS_HOME'])
    d = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(d, '.bfs_home')):
            return d
        up = os.path.dirname(d)
        if up == d:
            return None
        d = up


def data_root():
    home = bfs_home()
    return os.path.join(home, 'data') if home else os.path.join(CODE, 'bfs', 'data')


if __name__ == '__main__':
    args = sys.argv[1:]
    root, want_dir = CODE, False
    if '--root' in args:
        i = args.index('--root'); root = os.path.abspath(args[i + 1]); del args[i:i + 2]
    if '--dir' in args:
        args.remove('--dir'); want_dir = True
    if len(args) != 1 or args[0] not in PY:
        raise SystemExit("usage: plant_id.py [--dir] [--root CODE_DIR] coarse|fx")
    pid = plant_id(args[0], root)
    print(os.path.join(data_root(), f"{args[0]}-{pid}") if want_dir else pid)
