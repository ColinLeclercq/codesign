#!/bin/bash
# report.sh -- everything Claude needs from this code directory, in one place.
#
#   bash bfs/report.sh            (from the code directory, at any time: running jobs too)
#
# Makes for_claude/ (emptied and refilled at each call) and for_claude.tar.gz
# next to it -- upload the .tar.gz.  It holds:
#   SUMMARY.txt       per job log: what it ran, every round's line, the lattice
#                     lines, refusals, errors, the final state (the first thing
#                     to read)
#   logs/             every job log of this directory (logs/*.log)
#   runs/<run>/       per run directory (testplant_bfs/*, results_bfs/*): the
#                     record's JSON and npz (not the checkpoint), starts.json,
#                     the figures' post.json
# Nothing in the code directory is changed.
set -eo pipefail
cd "$(dirname "$0")/.."
OUTD=for_claude
rm -rf "$OUTD" "$OUTD.tar.gz"
mkdir -p "$OUTD/logs" "$OUTD/runs"
python3 bfs/version_check.py > "$OUTD/version.txt" 2>&1 || true
shopt -s nullglob
logs=(logs/*.log)
for f in "${logs[@]}"; do cp -p "$f" "$OUTD/logs/"; done
for run in testplant_bfs/*/ results_bfs/*/ results_bfs/; do
    [ -d "$run/run" ] || continue
    name=$(echo "${run%/}" | tr '/' '_')
    mkdir -p "$OUTD/runs/$name"
    [ -f "$run/starts.json" ] && cp -p "$run/starts.json" "$OUTD/runs/$name/"
    for f in "$run"/run/*.json "$run"/run/*.npz "$run"/run/*_post/post.json; do
        case "$f" in *_ckpt*) continue ;; esac
        cp -p "$f" "$OUTD/runs/$name/"
    done
done
{
    echo "report $(date -Is) from $(pwd)"
    cat "$OUTD/version.txt"
    echo "jobs still queued or running:"
    squeue -u "$USER" -h -o "  %i %j %T %M" 2>/dev/null || echo "  (squeue not available)"
    for f in "${logs[@]}"; do
        echo
        echo "================ $f ($(stat -c %y "$f" | cut -c1-19))"
        # the job's own output: slurm appends the script itself after "Command=",
        # whose echo lines would match too; its State / wall lines are kept
        awk '/^Command=/{s=1} !s || /^State:|^Job Wall-clock/' "$f" | grep -E "^=== host|^record |^  outer [0-9]+:|^    lattice:|refusal:|QP failed|crossing \(it|^(rows|sigma_2 blocks|sweep|follow|level set|  2n certificate|line|strip|moved design)|Traceback|Error|\*\*\* |^State:|^Job Wall-clock|best gamma after each round|final gamma|=== done" \
            | grep -v "^#" | cut -c1-400 || true
    done
} > "$OUTD/SUMMARY.txt"
tar czf "$OUTD.tar.gz" "$OUTD"
echo "$OUTD/ and $OUTD.tar.gz ($(du -h "$OUTD.tar.gz" | cut -f1)): upload $OUTD.tar.gz"
