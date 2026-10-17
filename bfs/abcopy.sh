#!/bin/bash
# abcopy.sh -- an A/B arm from a shared parent run (BFS.md 8; strip-ab.md AB1).
#
#   bash bfs/abcopy.sh testplant_bfs/<parent run> testplant_bfs/<arm>
#   CONTINUE=1 OUT=testplant_bfs/<arm> <the arm's one switch, e.g. ASYNC=0> ... sbatch ... bfs/slurm/testplant.slurm
#   (the strip A/B of 20261010n used STRIP=0|1; the strip was removed on 2026-10-16, BFS.md 8)
#
# A clean A/B starts both arms from the SAME state: the copy carries the
# parent's record or checkpoint and, since 20261010j, its loop state (inner
# budget, trust radius, peak set: the 20261010g arms continued different
# parents, and five of the six restarted at budget 10 -- a confound).  The
# parent is never changed.  Refused: an existing target, a parent that a live
# job is running (its run lock), a parent without a record or checkpoint.  The
# copy's code_versions.txt says where it came from.
set -eo pipefail
[ -n "${1:-}" ] && [ -n "${2:-}" ] || { echo "usage: abcopy.sh PARENT_RUN_DIR NEW_ARM_DIR"; exit 2; }
SRC=$(realpath -m "${1%/}"); DST=$(realpath -m "${2%/}")      # relative to where it is called from
source "$(dirname "$0")/lib.sh"
[ -d "$SRC/run" ] || { echo "*** $SRC/run missing: not a run directory"; exit 1; }
compgen -G "$SRC/run/K_B_C_*.npz" > /dev/null || { echo "*** $SRC holds no record or checkpoint"; exit 1; }
[ ! -e "$DST" ] || { echo "*** $DST exists: not overwritten"; exit 1; }
if [ -d "$SRC/.job.lockd" ]; then
    owner=$(cat "$SRC/.job.lockd/owner" 2>/dev/null || true)
    if job_alive "$owner"; then echo "*** $SRC is being run by job $owner: copy it once the job has ended"; exit 1; fi
fi
mkdir -p "$(dirname "$DST")"
cp -a "$SRC" "$DST.tmp.$$"
rm -rf "$DST.tmp.$$/.job.lockd"
mv "$DST.tmp.$$" "$DST"
echo "$(date -Is) copied from $SRC (bfs/abcopy.sh)" >> "$DST/code_versions.txt"
echo "$DST: a copy of $SRC ($(du -sh "$DST" | cut -f1)); continue it with CONTINUE=1 OUT=${2%/} ... sbatch ..."
echo "   (keep OUT relative, testplant_bfs/...: bfs/report.sh finds a run by the '-> testplant_bfs/...' of its log)"
