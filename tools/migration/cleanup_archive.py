#!/usr/bin/env python3
"""cleanup_archive.py -- HANDOFF_implementation.md section 4, executed.

Layout produced:

  <root>/   production closure of run_case_epi.py, the certification stack,
            and the gates (selftest, check_env, patch_pygranso)
  tools/    campaign tools: seeding, sweeps, plots, symmetry, diagnostics
  attic/    reached by nothing: benchmarks, one-off tests, abandoned tools

Two departures from the handoff's lists, both recorded in the run log:

  * check_optimality.py: handoff says attic.  It is reached by no import, but
    it is the last step of sweep.sh's documented post-processing chain and of
    report_v7 section 9's reproducibility chain.  An import graph does not see
    subprocess invocation.  -> tools/, beside collect.py and check_band.py.

  * fe_pipeline.py: handoff says tools/.  selftest.py -- the first gate, which
    must stay at the top level -- imports it, and a top-level module cannot
    import from a subdirectory.  -> top level, with the rest of selftest's
    import closure.

  * bounds.py: handoff says attic.  It is the source of the three certified
    floors (9.831865 / 9.660005 / 3.509066) quoted in report_v7 section 3 and
    hand-typed into plot_results.py.  -> tools/.

Everything moved into tools/ needs the repository root on sys.path: 13 of the
15 campaign tools import root modules (fe_codesign, controller, ...).  Several
already carry `sys.path.insert(0, dirname(abspath(__file__)))`, which inserted
the root while they sat in it and inserts tools/ afterwards.  Rather than
rewrite that line, a canonical root shim is inserted above it; the old line
then merely puts tools/ on the path as well, which is harmless.

Idempotent: re-running is a no-op.
"""
from __future__ import annotations
import ast
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (this script lives in tools/migration/)

KEEP_TOP = [
    # production closure of run_case_epi.py
    "run_case_epi.py", "certify_sparse.py", "closedloop_sparse.py",
    "controller.py", "fe_codesign.py", "fe_gl.py", "packer.py",
    "parallel.py", "warmstart.py", "winding_cert.py",
    # certification stack (independent of the optimiser)
    "verify_dense.py", "certify_results.py", "check_kkt.py",
    # gates
    "selftest.py", "check_env.py", "patch_pygranso.py",
    # fe_pipeline.py: handoff assigns it to tools/, but selftest.py imports
    # it directly and a top-level module cannot import from a subdirectory.
    "fe_pipeline.py",
]

TO_TOOLS = [
    "cases.py", "prepare.py", "seeds.py", "nesting.py", "sweep.sh",
    "case.slurm", "collect.py", "check_band.py", "check_symmetry.py",
    "check_sym_pair.py", "plot_symmetry.py", "plot_results.py",
    "plot_bode.py", "gauge_fields.py",
    "check_optimality.py",   # departure from handoff (was: attic)
    "bounds.py",             # departure from handoff (was: attic)
]

TO_ATTIC = [
    "ablate.py", "bench_invit.py", "bench_oracle_now.py", "bench_par2.py",
    "bench_segcert.py", "check_incomplete.py", "eigcount.py",
    "selftest_cert.py", "test_continuation.py", "test_eigcount.py",
    "test_substitution.py",
]

SHIM_MARK = "# --- repository root on sys.path (moved into tools/ by cleanup_archive.py)"
SHIM = (
    f"{SHIM_MARK}\n"
    "import os as _os_root, sys as _sys_root\n"
    "_sys_root.path.insert(0, _os_root.path.dirname(\n"
    "    _os_root.path.dirname(_os_root.path.abspath(__file__))))\n"
    "# --- end root shim\n"
)


def insertion_line(src: str) -> int:
    """0-based line index at which the shim may be inserted.

    After any shebang, encoding cookie, module docstring and __future__
    imports -- all of which must stay first.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    lines = src.splitlines()
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    if len(lines) > idx and re.match(r"^#.*coding[:=]", lines[idx]):
        idx += 1
    for node in tree.body:
        is_doc = (isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Constant)
                  and isinstance(node.value.value, str))
        is_future = (isinstance(node, ast.ImportFrom)
                     and node.module == "__future__")
        if is_doc or is_future:
            idx = max(idx, node.end_lineno)
        else:
            break
    return idx


def add_shim(path: str) -> bool:
    src = open(path).read()
    if SHIM_MARK in src:
        return False
    lines = src.splitlines(keepends=True)
    i = insertion_line(src)
    lines.insert(i, SHIM)
    open(path, "w").write("".join(lines))
    return True


def needs_shim(path: str, rootmods: set) -> bool:
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in rootmods:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and \
                    node.module.split(".")[0] in rootmods:
                return True
    return False


def move(name: str, dest: str, log: list) -> None:
    src = os.path.join(ROOT, name)
    dst = os.path.join(ROOT, dest, name)
    if os.path.exists(dst) and not os.path.exists(src):
        log.append(f"  already in {dest}/  {name}")
        return
    if not os.path.exists(src):
        log.append(f"  MISSING          {name}")
        return
    os.makedirs(os.path.join(ROOT, dest), exist_ok=True)
    shutil.move(src, dst)
    log.append(f"  -> {dest}/{' ' * (6 - len(dest))} {name}")


def main() -> int:
    rootmods = {f[:-3] for f in KEEP_TOP if f.endswith(".py")}
    log: list = []

    log.append("attic/  (reached by nothing; kept, not deleted)")
    for f in TO_ATTIC:
        move(f, "attic", log)

    log.append("")
    log.append("tools/  (campaign tools)")
    for f in TO_TOOLS:
        move(f, "tools", log)

    # root shim for the moved python tools that import root modules
    log.append("")
    log.append("root shim inserted into tools/ scripts importing root modules:")
    for f in TO_TOOLS:
        if not f.endswith(".py"):
            continue
        p = os.path.join(ROOT, "tools", f)
        if not os.path.exists(p):
            continue
        if needs_shim(p, rootmods):
            log.append(f"  {'added ' if add_shim(p) else 'present'}  {f}")

    log.append("")
    log.append("kept at top level:")
    for f in KEEP_TOP:
        mark = "ok " if os.path.exists(os.path.join(ROOT, f)) else "MISSING"
        log.append(f"  {mark} {f}")

    print("\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
