#!/bin/bash
# home.sh -- one fixed root for the runs, the plant data and the logs (user,
# 2026-10-14: "change the origin to /tmp_user/sator/cleclerc/BFS"), BFS.md 7.
#
#   bash bfs/home.sh [BFS_HOME]                 (once, in the code directory)
#   FROM=<old code directory> bash bfs/home.sh  (also MOVE an old version's runs, logs and data in)
#
# The code directory is a git clone (e.g. BFS_HOME/codesign, updated by git
# pull); what the jobs write lives in BFS_HOME, shared by every code version:
#
#   BFS_HOME/testplant_bfs/   test-plant runs      <- codesign/testplant_bfs (symlink)
#   BFS_HOME/results_bfs/     full-mesh runs       <- codesign/results_bfs
#   BFS_HOME/smoke_bfs/       the smoke test's run <- codesign/smoke_bfs
#   BFS_HOME/logs/            slurm logs           <- codesign/logs
#   BFS_HOME/data/            plant files (operators, meshes, certified ranges) <- codesign/bfs/data
#
# The scripts keep their relative paths (logs/, testplant_bfs/, bfs/data/...):
# only the five names above become symlinks.  So CONTINUE=1 after a git pull
# needs no copying, and the plant files are built once -- their generators'
# hashes are kept next to them (data/STAMP_*, written by the jobs that build
# them), and testplant.slurm rebuilds the test plant when they change.
#
# Default BFS_HOME: the parent of the code directory.  Nothing is ever deleted
# or overwritten: a real directory in the code directory is emptied into
# BFS_HOME by mv -n (an entry already there is left in place and reported),
# then removed only if empty; FROM= moves the old version's entries the same way.
set -eo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
HOME_DIR=${1:-${BFS_HOME:-$(dirname "$CODE")}}
mkdir -p "$HOME_DIR"
HOME_DIR=$(cd "$HOME_DIR" && pwd)
case "$HOME_DIR/" in "$CODE"/*) echo "*** BFS_HOME ($HOME_DIR) must be outside the code directory ($CODE)"; exit 1 ;; esac
echo "code directory: $CODE"
echo "BFS_HOME:       $HOME_DIR"

# name in the code directory -> directory in BFS_HOME
LINKS=("testplant_bfs:testplant_bfs" "results_bfs:results_bfs" "smoke_bfs:smoke_bfs" "logs:logs" "bfs/data:data")
KEPT=0

# move_into SRC DST: every entry of SRC into DST, never overwriting
move_into() {
    local src=$1 dst=$2 e
    shopt -s nullglob dotglob
    for e in "$src"/*; do
        if [ -e "$dst/$(basename "$e")" ] || [ -L "$dst/$(basename "$e")" ]; then
            echo "   kept in place (already in $dst): $e"; KEPT=1
        else
            mv -n "$e" "$dst/"
        fi
    done
    shopt -u nullglob dotglob
}

for pair in "${LINKS[@]}"; do
    name=${pair%%:*}; tgt="$HOME_DIR/${pair##*:}"
    mkdir -p "$tgt"
    here="$CODE/$name"
    if [ -L "$here" ]; then
        if [ "$(readlink -f "$here")" = "$(readlink -f "$tgt")" ]; then
            echo "ok    $name -> $tgt"; continue
        fi
        echo "*** $here is a symlink to $(readlink -f "$here"), not to $tgt: left as it is"; KEPT=1; continue
    fi
    if [ -d "$here" ]; then
        echo "move  $name/* -> $tgt/"
        move_into "$here" "$tgt"
        if rmdir "$here" 2>/dev/null; then :; else
            echo "*** $here still holds entries (see above): left as a directory, no symlink"; KEPT=1; continue
        fi
    elif [ -e "$here" ]; then
        echo "*** $here is a file: left as it is"; KEPT=1; continue
    fi
    ln -s "$tgt" "$here"
    echo "link  $name -> $tgt"
done

if [ -n "${FROM:-}" ]; then
    OLD=$(cd "$FROM" && pwd)
    [ "$OLD" != "$CODE" ] || { echo "*** FROM is this code directory"; exit 1; }
    echo "from  $OLD"
    for pair in "${LINKS[@]}"; do
        name=${pair%%:*}; tgt="$HOME_DIR/${pair##*:}"
        if [ -d "$OLD/$name" ] && [ ! -L "$OLD/$name" ]; then
            echo "move  $OLD/$name/* -> $tgt/"
            move_into "$OLD/$name" "$tgt"
        fi
    done
    # plant files from an older version carry no stamp: they are kept, and
    # testplant.slurm rebuilds the test plant once (a few minutes); ADOPT=1
    # stamps them as built by THIS code instead (only when the generators did
    # not change between the two versions: diff the two directories' bfs/mesh_coarse.py,
    # bfs/ops_dolfinx.py, bfs/plant_cert.py first)
    if [ "${ADOPT:-0}" = 1 ]; then
        ( cd "$CODE" && bash bfs/data_stamp.sh write coarse && bash bfs/data_stamp.sh write fx ) || true
        echo "stamped the plant files in $HOME_DIR/data as built by this code (ADOPT=1)"
    fi
fi
[ "$KEPT" = 0 ] || echo "*** some entries were left in place (above): nothing was deleted; resolve by hand"
echo "done.  The runs: $HOME_DIR/testplant_bfs, the logs: $HOME_DIR/logs (sbatch from $CODE as before)"
