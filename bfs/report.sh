#!/bin/bash
# report.sh -- everything Claude needs from this code version, in one place.
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
#                     (also the gate's and the winding check's), the excess
#                     diagnoses (missed peak / t lag / within tol, 20261010j),
#                     errors, the final state (the first thing to read)
#   version.txt       bfs/version_check.py (code version, git commit)
#   logs/             this version's job logs: those whose version_check line names
#                     it or says the checkout is NOT it, and those without a
#                     version_check line (a job that died before it) whose job is
#                     still queued or that are not older than this version's
#                     oldest log
#   runs/<run>/       per run directory those logs name (testplant_bfs/*,
#                     results_bfs/*, smoke_bfs/*): the record's JSON and npz (not
#                     the checkpoint), starts.json, code_versions.txt, the
#                     figures' post.json
#   figs/<run>/       the run's figures (*_post/*.png)
#   NOTE.txt          NOTE="..." when given
# Since 20261010i logs/ and the run directories are shared by every code
# version (bfs/home.sh), hence the selection by version.  Nothing in the code
# directory or in the runs is changed.
#
# PUSH=1: a git worktree of the branch `results` (BFS_HOME/.results_wt, else
# .results_wt here), made once; every report resets it to origin/results,
# MIRRORS for_claude/ into <version>/ (SUMMARY.txt, version.txt, logs/, runs/,
# figs/ replaced, so a figure the new report no longer has is not kept on
# GitHub; a NOTE.txt there, written on GitHub, is kept and NOTE= appended),
# adds with -f (the branch inherited bfs's .gitignore, whose logs/ line kept
# every log of 20261010g off GitHub), commits and pushes, three tries if
# someone pushed in between.  Needs git's user.name / user.email and push
# access (HTTPS token or SSH key) on this node; without them the tarball is
# still made.  Files above 50 MB are left out (BIG=1 keeps them) and listed.
set -eo pipefail
cd "$(dirname "$0")/.."
CODE=$(pwd)
OUTD=for_claude
VER=${VERSION:-$(sed -n 's/^# version //p' bfs/MANIFEST 2>/dev/null)}
rm -rf "$OUTD" "$OUTD.tar.gz"
mkdir -p "$OUTD/logs" "$OUTD/runs" "$OUTD/figs"
python3 bfs/version_check.py > "$OUTD/version.txt" 2>&1 || true
shopt -s nullglob
running=" $( { squeue -u "$USER" -h -o "%i" 2>/dev/null || true; } | tr '\n' ' ') "
own=(); nover=()
for f in logs/*.log; do
    if [ "${ALL:-0}" = 1 ] || grep -q -m1 -e "^version_check: code version $VER," -e "NOT the code version $VER:" "$f" 2>/dev/null; then
        own+=("$f")
    elif ! grep -q -m1 "^version_check: code version" "$f" 2>/dev/null && ! grep -q -m1 "NOT the code version" "$f" 2>/dev/null; then
        nover+=("$f")
    fi
done
logs=("${own[@]}")
oldest=""
for f in "${own[@]}"; do [ -z "$oldest" ] || [ "$f" -ot "$oldest" ] && oldest=$f; done
for f in "${nover[@]}"; do
    id=$(basename "$f" .log | grep -oE '[0-9]+(_[0-9]+)?$' || true)
    if [ -n "$id" ] && { [[ "$running" == *" $id "* ]] || [[ "$running" == *" ${id%%_*} "* ]]; }; then
        logs+=("$f")
    elif [ -n "$oldest" ] && ! [ "$f" -ot "$oldest" ]; then
        logs+=("$f")
    fi
done
for f in "${logs[@]}"; do cp -p "$f" "$OUTD/logs/"; done
# the run directories: those the selected logs name ("=== host ... -> DIR ==="
# of testplant.slurm, "-> DIR" of codesign.slurm and smoke.slurm), or all with ALL=1
runs=()
if [ "${ALL:-0}" = 1 ]; then
    for run in testplant_bfs/*/ results_bfs/*/ smoke_bfs/*/; do runs+=("${run%/}"); done
else
    while read -r d; do [ -n "$d" ] && runs+=("$d"); done < <(
        for f in "${logs[@]}"; do grep -oE -- '-> (testplant_bfs|results_bfs|smoke_bfs)/[^ ]+' "$f" | head -3 || true; done \
            | sed 's/^-> //' | sort -u)
fi
BIGF=()
cpcap() {   # cp, leaving out files above 50 MB unless BIG=1
    local f=$1 dst=$2
    if [ "${BIG:-0}" != 1 ] && [ "$(stat -c %s "$f")" -gt 52428800 ]; then BIGF+=("$f ($(du -h "$f" | cut -f1))"); return 0; fi
    cp -p "$f" "$dst"
}
for run in "${runs[@]}"; do
    [ -d "$run" ] || continue
    rd="$run/run"; [ -d "$rd" ] || rd="$run"
    name=$(echo "$run" | tr '/' '_')
    mkdir -p "$OUTD/runs/$name"
    for f in "$run/starts.json" "$run/code_versions.txt"; do [ -f "$f" ] && cp -p "$f" "$OUTD/runs/$name/"; done
    for f in "$rd"/*.json "$rd"/*.npz "$rd"/*_post/post.json; do
        case "$f" in *_ckpt*) continue ;; esac
        cpcap "$f" "$OUTD/runs/$name/"
    done
    pngs=("$rd"/*_post/*.png)
    if [ ${#pngs[@]} -gt 0 ]; then
        mkdir -p "$OUTD/figs/$(basename "$run")"
        cp -p "${pngs[@]}" "$OUTD/figs/$(basename "$run")/"
    fi
done
[ -n "${NOTE:-}" ] && printf '%s (%s)\n%s\n\n' "--- $(date -Is)" "$(hostname)" "$NOTE" > "$OUTD/NOTE.txt"
{
    echo "report $(date -Is) from $(pwd) (version ${VER:-?}$([ "${ALL:-0}" = 1 ] && echo ', ALL=1'))"
    cat "$OUTD/version.txt"
    echo "jobs still queued or running:"
    squeue -u "$USER" -h -o "  %i %j %T %M %Z" 2>/dev/null || echo "  (squeue not available)"
    [ ${#BIGF[@]} -eq 0 ] || { echo "left out (above 50 MB; BIG=1 keeps them):"; printf '  %s\n' "${BIGF[@]}"; }
    for f in "${logs[@]}"; do
        echo
        echo "================ $f ($(stat -c %y "$f" | cut -c1-19))"
        # the job's own output: slurm appends the script itself after "Command=",
        # whose echo lines would match too; its State / wall lines are kept
        awk '/^Command=/{s=1} !s || /^State:|^Job Wall-clock/' "$f" | grep -E "^=== host|^record |^version_check|^  outer [0-9]+:|^    lattice:|refusal:|missed peak:|t lag:|within tol:|^activity:|QP failed|crossing \(it|REJECTS|REFUSED|refused|INCONCLUSIVE|^  \[ledger\] lattice|^(rows|sigma_2 blocks|sweep|follow|level set|  2n certificate|line|strip|moved design)|Traceback|Error|\*\*\* |^State:|^Job Wall-clock|best gamma after each round|final gamma|=== done" \
            | grep -v "^#" | cut -c1-1000 || true
    done
} > "$OUTD/SUMMARY.txt"
tar czf "$OUTD.tar.gz" "$OUTD"
echo "$OUTD/ and $OUTD.tar.gz ($(du -h "$OUTD.tar.gz" | cut -f1)): ${#logs[@]} log(s), ${#runs[@]} run(s)"

[ "${PUSH:-0}" = 1 ] || { echo "(PUSH=1 commits it to the branch results and pushes; else upload $OUTD.tar.gz)"; exit 0; }
DEST=${LABEL:-${VER:-unversioned}}
HOME_DIR=$(python3 -c "import sys; sys.path.insert(0, 'bfs'); import plant_id; print(plant_id.bfs_home() or '')" 2>/dev/null || true)
WT=${HOME_DIR:-$CODE}/.results_wt
git config user.email > /dev/null || { echo "*** PUSH: git user.email is not set (git config --global user.email ...); upload $OUTD.tar.gz"; exit 1; }
exec 9> "$WT.lock"
command -v flock > /dev/null && { flock -n 9 || { echo "*** PUSH: another report is pushing ($WT.lock); try again"; exit 1; }; }
sync_wt() {     # the worktree = origin/results, whatever state an earlier report left
    git fetch -q origin +refs/heads/results:refs/remotes/origin/results || return 1
    git worktree prune
    if [ ! -d "$WT/.git" ] && [ ! -f "$WT/.git" ]; then
        git worktree add -q --detach "$WT" refs/remotes/origin/results || return 1
    fi
    git -C "$WT" rebase --abort 2>/dev/null || true
    git -C "$WT" merge --abort 2>/dev/null || true
    # (detached: the branch results may be checked out in another worktree)
    git -C "$WT" checkout -q --detach refs/remotes/origin/results || return 1
    git -C "$WT" reset -q --hard refs/remotes/origin/results || return 1
    git -C "$WT" clean -fdq
}
fill_wt() {
    mkdir -p "$WT/$DEST"
    rm -rf "$WT/$DEST"/{SUMMARY.txt,version.txt,logs,runs,figs}
    for e in SUMMARY.txt version.txt logs runs figs; do [ -e "$OUTD/$e" ] && cp -rp "$OUTD/$e" "$WT/$DEST/"; done
    [ -n "${NOTE:-}" ] && cat "$OUTD/NOTE.txt" >> "$WT/$DEST/NOTE.txt"
    git -C "$WT" add -A -f -- "$DEST"
}
for try in 1 2 3; do
    sync_wt || { echo "*** PUSH: cannot fetch the branch results from origin; upload $OUTD.tar.gz"; exit 1; }
    fill_wt
    if git -C "$WT" diff --cached --quiet; then echo "results/$DEST: nothing new to commit"; exit 0; fi
    git -C "$WT" commit -q -m "results $DEST: ${#logs[@]} log(s), ${#runs[@]} run(s) ($(date -Is), $(hostname))"
    if git -C "$WT" push -q origin HEAD:refs/heads/results; then
        echo "pushed: branch results, folder $DEST/ ($(git -C "$WT" rev-parse --short HEAD))"; exit 0
    fi
    echo "   push rejected (someone pushed in between): try $try of 3 again from origin/results"
done
echo "*** PUSH: three pushes rejected; upload $OUTD.tar.gz"; exit 1
