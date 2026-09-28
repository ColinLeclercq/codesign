#!/usr/bin/env python3
"""check_invocations.py -- every caller of run_case_epi must pass --eps.

The bug this exists to prevent
------------------------------
`harness.py` invoked run_case_epi without `--eps`, so run_case_epi fell back to
`packer.EPS_STAB_DEFAULT = 1e-3` while `case.slurm` -- which produced the frozen
suite -- passes `--eps 0.05`.  The whole stage-1 baseline (job 16016215, 6 h on
96 cores) therefore solved a problem with a stability margin 50x weaker than the
specification, and so did the determinism job before it.

Nothing failed.  Every run exited 0, every plot was written, every certificate
was issued.  Two of the three designs were infeasible at the real margin and the
certification chain passed them, because `verify_dense` defaults to the same
1e-3 and `winding_cert` counts across a line at -eps/2.

The cost of catching it was zero seconds of compute.  The cost of not catching
it was two jobs and most of a day.  Hence this file.

What it checks
--------------
Every command line in the tree that invokes run_case_epi carries `--eps`, and
every one that invokes verify_dense or winding_cert does too, since their
defaults disagree with each other (1e-3 and 0.05 respectively).

Deliberately a dumb textual scan.  It cannot be fooled by a flag that is present
but wrong -- that is what the harness's own EPS row check is for -- but it also
cannot be fooled by anything clever, and it runs in well under a second on a
login node with no environment at all.

    python tools/check_invocations.py         # exit 1 if anything is missing

Exclusions: attic/ (not run), and this file.
"""
from __future__ import annotations
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# script -> flags every invocation of it must carry, and why
REQUIRED = {
    'run_case_epi.py': [
        ('--eps', 'falls back to packer.EPS_STAB_DEFAULT = 1e-3, a 50x weaker '
                  'margin than the suite 0.05 -- a different problem'),
    ],
    'verify_dense.py': [
        ('--eps', 'defaults to 1e-3, so it asks "is it stable" rather than '
                  '"does it meet the spec"'),
    ],
    'winding_cert.py': [
        ('--eps', 'defaults to 0.05 while verify_dense defaults to 1e-3; the '
                  'two disagreeing is how one design got opposite verdicts'),
    ],
}

SKIP_DIRS = {'attic', '__pycache__', '.git', 'results', 'figs', 'dumps',
             'cleanup_evidence'}   # frozen snapshots kept for reference, never run
SCAN_EXT = ('.py', '.sh', '.slurm')


def invocations(text: str, script: str):
    """Yield (line_no, command) for each invocation of `script` in `text`.

    A command may be continued over several lines with a trailing backslash, and
    in Python may be a list of arguments spanning several lines.  Both are
    joined before the flag is looked for, otherwise a perfectly correct
    multi-line call reads as a violation.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if script in lines[i] and not lines[i].lstrip().startswith('#'):
            buf, j = lines[i], i
            # shell line continuations
            while buf.rstrip().endswith('\\') and j + 1 < len(lines):
                j += 1
                buf += ' ' + lines[j]
            # python argv lists: keep going while brackets are unbalanced
            depth = buf.count('[') - buf.count(']')
            while depth > 0 and j + 1 < len(lines):
                j += 1
                buf += ' ' + lines[j]
                depth += lines[j].count('[') - lines[j].count(']')
            yield i + 1, buf
            i = j + 1
        else:
            i += 1


def main() -> int:
    bad = []
    n_files = n_calls = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(SCAN_EXT) or fn == os.path.basename(__file__):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT)
            try:
                text = open(path, encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            n_files += 1
            for script, flags in REQUIRED.items():
                for lineno, cmd in invocations(text, script):
                    # the argparse declaration inside the script itself, and
                    # prose mentioning it, are not invocations
                    if rel == script or rel.endswith('/' + script):
                        continue
                    if not re.search(r'(python[\w.]*\s|sys\.executable|'
                                     r'ROOT,\s*["\'])', cmd):
                        continue
                    n_calls += 1
                    for flag, why in flags:
                        if flag not in cmd:
                            bad.append((rel, lineno, script, flag, why,
                                        ' '.join(cmd.split())[:100]))

    print(f"scanned {n_files} files, {n_calls} invocation(s)")
    if not bad:
        print("OK: every invocation carries its required flags")
        return 0
    print(f"\n*** {len(bad)} invocation(s) missing a required flag\n")
    for rel, lineno, script, flag, why, cmd in bad:
        print(f"  {rel}:{lineno}")
        print(f"    calls {script} without {flag}")
        print(f"    {flag} {why}")
        print(f"    > {cmd}")
        print()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
