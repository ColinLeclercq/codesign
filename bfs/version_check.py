#!/usr/bin/env python3
"""version_check.py -- is this checkout ONE consistent version of the code?
(BFS.md 7.)  bfs/MANIFEST lists the sha256 of every code file the runs execute
(the top-level and bfs/ *.py, *.sh, *.slurm), written when the package was
made.  The slurm scripts run this first: a file from another version (a
partial copy, an old file left behind) stops the job before it computes
anything, with the files named.

Site settings are not code and are left out (2026-10-04, a change of cluster
stopped every job): env.sh (modules, conda) is not listed, and the #SBATCH
lines of the slurm scripts (partition, account, memory, time) are not hashed.

    python bfs/version_check.py            check (exit 1 on a mismatch;
                                           ALLOW_MODIFIED=1: warn only)
    python bfs/version_check.py --write L  (packaging) rewrite the manifest,
                                           version label L
"""
import os, sys, hashlib, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAN = os.path.join(HERE, 'MANIFEST')


SITE = ('env.sh',)              # site settings, not code: never listed


def _sha(path):
    """sha256 of the file; for a slurm script, without its #SBATCH lines."""
    with open(path, 'rb') as f:
        data = f.read()
    if path.endswith('.slurm'):
        data = b''.join(l for l in data.splitlines(keepends=True) if not l.lstrip().startswith(b'#SBATCH'))
    return hashlib.sha256(data).hexdigest()


def _files():
    out = subprocess.run(['git', 'ls-files', '*.py', '*.sh', '*.slurm'], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    return sorted(p for p in out if not p.startswith(('tools/', 'attic/')) and p not in SITE)


def write(label):
    rows = [f"{_sha(os.path.join(ROOT, p))}  {p}" for p in _files()]
    with open(MAN, 'w') as f:
        f.write(f"# version {label}\n" + "\n".join(rows) + "\n")
    print(f"{MAN}: {len(rows)} files, version {label}")


def _git(*a):
    """git output, or None when git fails (no repository, no git)."""
    try:
        r = subprocess.run(['git', '--no-optional-locks', *a], cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, universal_newlines=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _git_line():
    """The commit, when the code directory is a git clone or worktree (bfs/home.sh, 20261010i)."""
    head = _git('rev-parse', '--short', 'HEAD')
    if not head:
        return
    st = _git('status', '--porcelain', '--untracked-files=no')
    where = _git('rev-parse', '--abbrev-ref', 'HEAD')
    print(f"version_check: git {where} {head}"
          + (", git status failed" if st is None else
             (f", {len([l for l in st.splitlines() if l.strip()])} tracked file(s) modified" if st else ", clean")))


def _main_clone_under_home():
    """True when this is the MAIN clone of a BFS_HOME layout (bfs/stage.sh): jobs
    are submitted from the per-version worktrees BFS/v/<version>, never from
    the clone that git pull updates (20261010j)."""
    gd, cd = _git('rev-parse', '--absolute-git-dir'), _git('rev-parse', '--git-common-dir')
    if not gd or not cd:
        return False
    cd = os.path.abspath(os.path.join(ROOT, cd)) if not os.path.isabs(cd) else cd
    if os.path.realpath(gd) != os.path.realpath(cd):
        return False                     # a linked worktree
    d = ROOT
    while True:
        if os.path.isfile(os.path.join(d, '.bfs_home')):
            return True
        up = os.path.dirname(d)
        if up == d:
            return False
        d = up


def check():
    if not os.path.exists(MAN):
        print("version_check: bfs/MANIFEST missing -- this checkout is older than the check (2026-10-02)")
        return 1
    label, bad = '?', []
    n = 0
    for line in open(MAN):
        if line.startswith('# version'):
            label = line.split(None, 2)[2].strip()
            continue
        sha, path = line.split(None, 1)
        path = path.strip(); n += 1
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            bad.append(f"missing   {path}")
        elif _sha(full) != sha:
            bad.append(f"different {path}")
    if not bad:
        print(f"version_check: code version {label}, {n} files consistent")
        _git_line()
        if _main_clone_under_home() and os.environ.get('ALLOW_CLONE') != '1':
            print("version_check: this is the git clone that git pull updates: submit from its per-version worktree")
            print("   (bash bfs/stage.sh prints it; ALLOW_CLONE=1 runs here anyway)")
            return 1
        return 0
    print(f"version_check: this checkout is NOT the code version {label}: {len(bad)} of {n} files differ")
    for b in bad:
        print("   " + b)
    print("   (a file you edited on purpose: ALLOW_MODIFIED=1 sbatch ... runs anyway; in a git clone (bfs/home.sh)")
    print("    git status / git diff show what differs and git pull brings the clone to one version; from a package:")
    print("    tar xzf <package>.tar.gz -O codesign_bfs/<file> | diff - <file>, or extract the package again)")
    return 0 if os.environ.get('ALLOW_MODIFIED') == '1' else 1


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--write':
        write(sys.argv[2])
    else:
        sys.exit(check())
