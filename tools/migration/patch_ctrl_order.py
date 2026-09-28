#!/usr/bin/env python3
"""Replace the `except Exception: a.r = 7` fallback in every tool that has it.

Each block becomes a call to ctrl_order.infer_r, which prefers the design
records, then the seed catalogue, and raises rather than guessing.

The replacement is line-wise and bounded, not a multi-line regex.  A regex with
a non-greedy body happily matched an unrelated `try:` 110 lines earlier in
plot_results.py and deleted everything in between, so: locate `a.r = 7`, walk
back at most MAXBACK lines to the `try:` that owns it, require a seed-catalogue
mention strictly inside that span, and replace exactly those lines.  Anything
that does not fit the shape is reported and left untouched.

Idempotent.
"""
from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (this script lives in tools/migration/)
MAXBACK = 8

TARGETS = [
    "certify_results.py", "check_kkt.py", "verify_dense.py", "winding_cert.py",
    "tools/check_band.py", "tools/check_optimality.py",
    "tools/check_sym_pair.py", "tools/check_symmetry.py",
    "tools/gauge_fields.py", "tools/plot_results.py", "tools/plot_symmetry.py",
]


def patch_one(path: str) -> str:
    lines = open(path).read().splitlines(keepends=True)

    hits = [i for i, ln in enumerate(lines) if ln.strip() == "a.r = 7"]
    if not hits:
        return "no `a.r = 7` found"
    if len(hits) > 1:
        return f"{len(hits)} occurrences of `a.r = 7` -- CHECK BY HAND"
    end = hits[0]

    if lines[end - 1].strip() not in ("except Exception:", "except:"):
        return "line above `a.r = 7` is not an except -- CHECK BY HAND"

    start = None
    for i in range(end - 2, max(-1, end - 2 - MAXBACK), -1):
        if lines[i].strip() == "try:":
            start = i
            break
    if start is None:
        return f"no `try:` within {MAXBACK} lines above -- CHECK BY HAND"

    span = "".join(lines[start:end + 1])
    if "seeds_ne" not in span:
        return "block does not mention the seed catalogue -- CHECK BY HAND"

    indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
    repl = (f"{indent}from ctrl_order import infer_r\n"
            f"{indent}a.r = infer_r(a.ne, getattr(a, 'results', None))\n")
    new = lines[:start] + [repl] + lines[end + 1:]
    open(path, "w").write("".join(new))
    return f"patched (replaced lines {start + 1}-{end + 1})"


def main() -> int:
    bad = 0
    for rel in TARGETS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(f"  MISSING   {rel}")
            bad += 1
            continue
        if "from ctrl_order import infer_r" in open(p).read():
            print(f"  {rel:32s} already patched")
            continue
        msg = patch_one(p)
        print(f"  {rel:32s} {msg}")
        if "CHECK BY HAND" in msg or "no `a.r = 7`" in msg:
            bad += 1
    print(f"\n{bad} file(s) need attention")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
