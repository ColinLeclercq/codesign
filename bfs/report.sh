#!/bin/bash
# report.sh -- everything Claude needs from this code directory, in one place.
#
#   bash bfs/report.sh                 (from the code directory, at any time: running jobs too)
#   PUSH=1 bash bfs/report.sh          (and commit it to the branch `results` of the GitHub
#                                       repository, folder <version>/, and push: user, 2026-10-14)
#   PUSH=1 NOTE="what looks strange: ..." bash bfs/report.sh   (adds the text to <version>/NOTE.txt)
#   ALL=1 bash bfs/report.sh           (every log and run, not only this code version's)
#   VERSION=20261010g bash bfs/report.sh   (the logs and runs of another code version)
#
# Makes for_claude/ (emptied and refilled at each call) and for_claude.tar.gz
# next to it.  It holds:
#   SUMMARY.txt       per job log: what it ran, every round's line, the lattice
#                     lines and the lattice [ledger] line (ranks busy %), refusals
#                     (also the gate's and the winding check's), errors, the final
#                     state (the first thing to read)
#   version.txt       bfs/version_check.py (code version, git commit)
#   logs/             the job logs of this code version (logs/*.log whose
#                     version_check line names it, and the jobs still running)
#   runs/<run>/       per run directory those logs wrote (testplant_bfs/*, results_bfs/*):
#                     the record's JSON and npz (not the checkpoint), starts.json,
#                     the figures' post.json
#   figs/<run>/       the run's figures (*_post/*.png)
# Since 20261010h logs/ and the run directories are shared by every code
# version (bfs/home.sh), hence the selection by version.  Nothing in the code
# directory or in the runs is changed.
#
# PUSH=1: a git worktree of the branch `results` in .results_wt/ (made once,
# then pulled); for_claude/ is copied into <version>/ there (files of an
# earlier report of the same version are overwritten, others kept: a NOTE.txt
# written on GitHub stays), committed and pushed.  Needs git's user.name /
# user.email and push access (HTTPS token or SSH key) on this node; without them
# the tarball is still made.
set -eo pipefail
cd "$(dirname "$0")/.."
OUTD=for_claude
VER=${VERSION:-$(sed -n 's/^# version //p' bfs/MANIFEST 2>/dev/null)}
rm -rf "$OUTD" "$OUTD.tar.gz"
mkdir -p "$OUTD/logs" "$OUTD/runs" "$OUTD/figs"
python3 bfs/version_check.py > "$OUTD/version.txt" 2>&1 || true
shopt -s nullglob
running=" $( { squeue -u "$USER" -h -o "%i" 2>/dev/null || true; } | tr '\n' ' ') "
logs=()
for f in logs/*.log; do
    if [ "${ALL:-0}" = 1 ] || grep -q -m1 "^version_check: code version $VER," "$f" 2>/dev/null; then
        logs+=("$f"); continue
    fi
    # a job of this version still in the queue (its log may not have the line yet)
    id=$(basename "$f" .log | grep -oE '[0-9]+(_[0-9]+)?$' || true)
    [ -n "$id" ] && { [[ "$running" == *" $id "* ]] || [[ "$running" == *" ${id%%_*} "* ]]; } && logs+=("$f")
done
for f in "${logs[@]}"; do cp -p "$f" "$OUTD/logs/"; done
# the run directories: those the selected logs name ("=== host ... -> DIR ==="
# of testplant.slurm, "-> DIR" of codesign.slurm), or all with ALL=1
runs=()
if [ "${ALL:-0}" = 1 ]; then
    for run in testplant_bfs/*/ results_bfs/*/ results_bfs/; do runs+=("${run%/}"); done
else
    while read -r d; do [ -n "$d" ] && runs+=("$d"); done < <(
        for f in "${logs[@]}"; do grep -m3 -oE -- '-> (testplant_bfs|results_bfs|smoke_bfs)/[^ ]+' "$f" || true; done \
            | sed 's/^-> //' | sort -u)
fi
for run in "${runs[@]}"; do
    [ -d "$run/run" ] || [ -d "$run" ] || continue
    rd="$run/run"; [ -d "$rd" ] || rd="$run"
    name=$(echo "$run" | tr '/' '_')
    mkdir -p "$OUTD/runs/$name"
    [ -f "$run/starts.json" ] && cp -p "$run/starts.json" "$OUTD/runs/$name/"
    for f in "$rd"/*.json "$rd"/*.npz "$rd"/*_post/post.json; do
        case "$f" in *_ckpt*) continue ;; esac
        cp -p "$f" "$OUTD/runs/$name/"
    done
    pngs=("$rd"/*_post/*.png)
    if [ ${#pngs[@]} -gt 0 ]; then
        mkdir -p "$OUTD/figs/$(basename "$run")"
        cp -p "${pngs[@]}" "$OUTD/figs/$(basename "$run")/"
    fi
done
{
    echo "report $(date -Is) from $(pwd) (version ${VER:-?}$([ "${ALL:-0}" = 1 ] && echo ', ALL=1'))"
    cat "$OUTD/version.txt"
    echo "jobs still queued or running:"
    squeue -u "$USER" -h -o "  %i %j %T %M" 2>/dev/null || echo "  (squeue not available)"
    for f in "${logs[@]}"; do
        echo
        echo "================ $f ($(stat -c %y "$f" | cut -c1-19))"
        # the job's own output: slurm appends the script itself after "Command=",
        # whose echo lines would match too; its State / wall lines are kept
        awk '/^Command=/{s=1} !s || /^State:|^Job Wall-clock/' "$f" | grep -E "^=== host|^record |^  outer [0-9]+:|^    lattice:|refusal:|QP failed|crossing \(it|REJECTS|REFUSED|refused|INCONCLUSIVE|^  \[ledger\] lattice|^(rows|sigma_2 blocks|sweep|follow|level set|  2n certificate|line|strip|moved design)|Traceback|Error|\*\*\* |^State:|^Job Wall-clock|best gamma after each round|final gamma|=== done" \
            | grep -v "^#" | cut -c1-1000 || true
    done
} > "$OUTD/SUMMARY.txt"
tar czf "$OUTD.tar.gz" "$OUTD"
echo "$OUTD/ and $OUTD.tar.gz ($(du -h "$OUTD.tar.gz" | cut -f1)): ${#logs[@]} log(s), ${#runs[@]} run(s)"

[ "${PUSH:-0}" = 1 ] || { echo "(PUSH=1 commits it to the branch results and pushes; else upload $OUTD.tar.gz)"; exit 0; }
DEST=${LABEL:-${VER:-unversioned}}
WT=.results_wt
git config user.email > /dev/null || { echo "*** PUSH: git user.email is not set (git config --global user.email ...); upload $OUTD.tar.gz"; exit 1; }
git fetch -q origin results || { echo "*** PUSH: cannot fetch the branch results from origin; upload $OUTD.tar.gz"; exit 1; }
if [ ! -d "$WT" ]; then
    git worktree add -q -B results "$WT" origin/results
else
    git -C "$WT" checkout -q results
    git -C "$WT" pull -q --ff-only origin results
fi
mkdir -p "$WT/$DEST"
cp -rp "$OUTD"/. "$WT/$DEST/"
if [ -n "${NOTE:-}" ]; then
    printf '%s (%s)\n%s\n\n' "--- $(date -Is)" "$(hostname)" "$NOTE" >> "$WT/$DEST/NOTE.txt"
fi
git -C "$WT" add -A "$DEST"
if git -C "$WT" diff --cached --quiet; then
    echo "results/$DEST: nothing new to commit"
else
    git -C "$WT" commit -q -m "results $DEST: ${#logs[@]} log(s), ${#runs[@]} run(s) ($(date -Is), $(hostname))"
    git -C "$WT" push -q origin results || { git -C "$WT" pull -q --rebase origin results && git -C "$WT" push -q origin results; }
    echo "pushed: branch results, folder $DEST/ ($(git -C "$WT" rev-parse --short HEAD))"
fi
