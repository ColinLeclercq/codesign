#!/usr/bin/env python3
"""plant_id.py -- where the plant files of THIS code live (BFS.md 7).

    python bfs/plant_id.py coarse|fx            the id (16 hex digits)
    python bfs/plant_id.py --dir coarse|fx      the directory: <data root>/<kind>-<id>
    python bfs/plant_id.py --root OLD coarse    the id of another code directory's generators
    python bfs/plant_id.py --adopt coarse [DIR] give an existing plant directory this id (lists the candidates)

The plant files (mesh, operators, certified range and poles) are built once
per id into their own directory and never rebuilt in place, so code versions
that share a data root (bfs/home.sh) never overwrite each other's files: a
version whose generators differ gets another directory (user, 2026-10-14;
the layout review of 20261010i).

The id hashes what the files are made from:
  * the generators and what they import (bfs/mesh_coarse.py for the test
    plant, bfs/ops_dolfinx.py, bfs/archive_dsr/nsforms.py, bfs/plant_io.py,
    bfs/plant_cert.py) -- their source without comments and blank lines, the
    same for every Python version, so an edit of a comment costs no rebuild;
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
    """The source without its comments, trailing blanks and blank lines.  The
    same for every Python version (20261010k): the first version hashed
    ast.dump, whose format changes between 3.12 and 3.13, so the login node
    (home.sh) and the jobs (the conda env) disagreed on the id of the same
    files and the jobs rebuilt the test plant while the others waited.
    Docstrings count (an edit of one costs one rebuild); comments do not.
    tokenize only LOCATES the comments: its token stream itself differs
    between versions (f-strings since 3.12), their positions do not."""
    import io, tokenize
    src = open(path).read()
    lines = src.splitlines()
    cut = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            cut[tok.start[0] - 1] = tok.start[1]
    out = []
    for k, line in enumerate(lines):
        line = (line[:cut[k]] if k in cut else line).rstrip()
        if line:
            out.append(line)
    return hashlib.sha256("\n".join(out).encode()).hexdigest()


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
            return int(ast.literal_eval(node.value))      # ast.Num before Python 3.8, ast.Constant after
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


def adopt(kind, src=None):
    """Give an existing plant directory this code's id (20261010k: the ids of
    20261010j depended on the Python version, so complete files sit under
    ids no job computes any more).  Without SRC: list the candidates.  The
    caller vouches that SRC was made by the same generators (they are
    unchanged since 20261010g); SRC is renamed, never copied or deleted."""
    want = os.path.join(data_root(), f"{kind}-{plant_id(kind)}")
    if src is None:
        import glob
        print(f"this code's {kind} directory: {want} ({'exists' if os.path.isdir(want) else 'missing'})")
        for d in sorted(glob.glob(os.path.join(data_root(), f"{kind}-*"))):
            if os.path.abspath(d) == want:
                continue
            files = sorted(os.listdir(d))
            poles = [f"ops_{kind}_range_poles_{x}.npz" for x in ('-0.050000', '-0.025000')]
            ok = all(f in files for f in [f"ops_{kind}.npz", f"ops_{kind}_range.npz"] + poles)
            lock = ', build lock present (a job may still write it)' if '.build.lockd' in files else ''
            print(f"  candidate {d}: {'complete (operators, range, poles)' if ok else 'incomplete'}{lock}; {len(files)} entries")
        print(f"adopt one: python bfs/plant_id.py --adopt {kind} <directory>")
        return 0
    src = os.path.abspath(src)
    if os.path.exists(want):
        print(f"*** {want} exists already: nothing to adopt"); return 1
    if not os.path.isfile(os.path.join(src, f"ops_{kind}.npz")):
        print(f"*** {src} holds no ops_{kind}.npz: not a {kind} plant directory"); return 1
    if os.path.isdir(os.path.join(src, '.build.lockd')):
        print(f"*** {src} has a build lock (.build.lockd): a job may be writing it; remove the lock once no job runs"); return 1
    os.rename(src, want)
    with open(os.path.join(want, 'ADOPTED.txt'), 'a') as f:
        import datetime
        f.write(f"{datetime.datetime.now().isoformat()} {src} -> {want} (bfs/plant_id.py --adopt)\n")
    print(f"{src} -> {want}")
    return 0


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--adopt':
        if len(args) not in (2, 3) or args[1] not in PY:
            raise SystemExit("usage: plant_id.py --adopt coarse|fx [DIRECTORY]")
        sys.exit(adopt(args[1], args[2] if len(args) == 3 else None))
    root, want_dir = CODE, False
    if '--root' in args:
        i = args.index('--root'); root = os.path.abspath(args[i + 1]); del args[i:i + 2]
    if '--dir' in args:
        args.remove('--dir'); want_dir = True
    if len(args) != 1 or args[0] not in PY:
        raise SystemExit("usage: plant_id.py [--dir] [--root CODE_DIR] coarse|fx")
    pid = plant_id(args[0], root)
    print(os.path.join(data_root(), f"{args[0]}-{pid}") if want_dir else pid)
