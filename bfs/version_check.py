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


def _git_line():
    """The commit, when the code directory is a git clone (bfs/home.sh, 20261010h)."""
    try:
        run = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        head = run('rev-parse', '--short', 'HEAD')
        if head:
            dirty = [l for l in run('status', '--porcelain', '--untracked-files=no').splitlines() if l.strip()]
            print(f"version_check: git {run('rev-parse', '--abbrev-ref', 'HEAD')} {head}"
                  + (f", {len(dirty)} tracked file(s) modified" if dirty else ", clean"))
    except Exception:
        pass


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
