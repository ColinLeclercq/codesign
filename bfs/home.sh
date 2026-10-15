#!/bin/bash
# home.sh -- one fixed root for the runs, the plant files and the logs (user,
# 2026-10-14: "change the origin to /tmp_user/sator/cleclerc/BFS"; BFS.md 7).
#
#   INIT=1 bash bfs/home.sh /tmp_user/sator/cleclerc/BFS   once: mark the root (BFS/.bfs_home)
#   bash bfs/home.sh                    in a code directory under the root: link it to the root
#   FROM=<old code directory> bash bfs/home.sh    also MOVE an old version's runs, logs and plant files in
#
# The layout (the root is found from the marker file .bfs_home, in an ancestor
# of the code directory, or from $BFS_HOME):
#
#   BFS/codesign/          the git clone, updated by git pull (never submitted from)
#   BFS/v/<version>/       one git worktree per code version (bfs/stage.sh): jobs are
#                          submitted from there, so a pull never changes a running job's code
#   BFS/testplant_bfs/     test-plant runs        <- <code>/testplant_bfs (symlink)
#   BFS/results_bfs/       full-mesh runs         <- <code>/results_bfs
#   BFS/smoke_bfs/         smoke tests            <- <code>/smoke_bfs
#   BFS/logs/              slurm logs             <- <code>/logs
#   BFS/data/<kind>-<id>/  plant files by content id (bfs/plant_id.py); the scripts find them themselves
#   BFS/.results_wt/       the worktree of the branch results (bfs/report.sh PUSH=1)
#
# Nothing is ever deleted or overwritten: an entry already in the root is left
# in place and reported, a real directory in the code directory is emptied
# into the root by mv -n and replaced by the symlink only once empty.  Refused
# (FORCE=1 to override): a job still queued or running from the code directory
# or from FROM (its next checkpoint would go to a moved directory -- the final
# record write is not guarded), a FROM on another file system (mv would copy).
# Lock directories and half-written temporaries (.build.lockd, .job.lockd,
# .coarse.*, .ops.*) are never moved.  Old plant files (bfs/data/ops_coarse*.npz,
# coarse/, ops_fx*.npz of a version before 20261010j) are moved into
# BFS/data/<kind>-<id>/ only when FROM's generators give the same id as this
# code's (bfs/plant_id.py --root); otherwise they stay where they are.
set -eo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python3}

if [ "${INIT:-0}" = 1 ]; then
    [ -n "${1:-}" ] || { echo "usage: INIT=1 bash bfs/home.sh ROOT"; exit 2; }
    mkdir -p "$1"; touch "$1/.bfs_home"
    echo "marked $(cd "$1" && pwd) as BFS_HOME (.bfs_home)"
fi
HOME_DIR=${1:-${BFS_HOME:-$(cd "$CODE" && $PY -c "import sys; sys.path.insert(0, 'bfs'); import plant_id; print(plant_id.bfs_home() or '')")}}
[ -n "$HOME_DIR" ] || { echo "*** no BFS_HOME: no .bfs_home above $CODE (INIT=1 bash bfs/home.sh ROOT marks one)"; exit 1; }
HOME_DIR=$(cd "$HOME_DIR" && pwd)
[ -f "$HOME_DIR/.bfs_home" ] || { echo "*** $HOME_DIR is not marked as BFS_HOME (INIT=1 bash bfs/home.sh $HOME_DIR)"; exit 1; }
case "$HOME_DIR/" in "$CODE"/*) echo "*** BFS_HOME ($HOME_DIR) must be outside the code directory ($CODE)"; exit 1 ;; esac
echo "code directory: $CODE"
echo "BFS_HOME:       $HOME_DIR"

# jobs in flight from a directory we would move out of
busy_in() {
    local dir=$1
    command -v squeue > /dev/null 2>&1 || return 1
    local id wd
    while read -r id wd; do
        wd=$(readlink -f "$wd" 2>/dev/null || echo "$wd")
        [ "$wd" = "$dir" ] || [ "${wd#"$dir"/}" != "$wd" ] && return 0
    done < <(squeue -h -u "$USER" -o '%i %Z' 2>/dev/null || true)
    return 1
}
# only where something would MOVE: re-linking a directory jobs run from is harmless
_moves=0
for name in testplant_bfs results_bfs smoke_bfs logs; do [ -d "$CODE/$name" ] && [ ! -L "$CODE/$name" ] && _moves=1; done
for d in $([ "$_moves" = 1 ] && readlink -f "$CODE") ${FROM:+"$(readlink -f "$FROM")"}; do
    if busy_in "$d" && [ "${FORCE:-0}" != 1 ]; then
        echo "*** jobs are queued or running from $d (squeue -o '%i %Z'): wait for them (or FORCE=1)"; exit 1
    fi
done

LINKS=("testplant_bfs" "results_bfs" "smoke_bfs" "logs")
KEPT=0
skip_entry() { case "$(basename "$1")" in .build.lockd|.job.lockd|.coarse.*|.ops.*|*.stale.*) return 0 ;; esac; return 1; }

# move_into SRC DST: every entry of SRC into DST, never overwriting
move_into() {
    local src=$1 dst=$2 e
    if [ "$(stat -c %d "$src")" != "$(stat -c %d "$dst")" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "*** $src and $dst are on different file systems (mv would copy): nothing moved (FORCE=1 to copy)"; KEPT=1; return 0
    fi
    shopt -s nullglob dotglob
    for e in "$src"/*; do
        if skip_entry "$e"; then echo "   left in place (lock / temporary): $e"; KEPT=1; continue; fi
        if [ -e "$dst/$(basename "$e")" ] || [ -L "$dst/$(basename "$e")" ]; then
            echo "   kept in place (already in $dst): $e"; KEPT=1
        else
            mv -n "$e" "$dst/"
        fi
    done
    shopt -u nullglob dotglob
}

for name in "${LINKS[@]}"; do
    tgt="$HOME_DIR/$name"; here="$CODE/$name"
    mkdir -p "$tgt"
    if [ -L "$here" ]; then
        if [ "$(readlink -f "$here")" = "$(readlink -f "$tgt")" ]; then echo "ok    $name -> $tgt"; continue; fi
        echo "*** $here is a symlink to $(readlink -f "$here"), not to $tgt: left as it is"; KEPT=1; continue
    fi
    if [ -d "$here" ]; then
        echo "move  $name/* -> $tgt/"
        move_into "$here" "$tgt"
        rmdir "$here" 2>/dev/null || { echo "*** $here still holds entries (see above): left as a directory, no symlink"; KEPT=1; continue; }
    elif [ -e "$here" ]; then
        echo "*** $here is a file: left as it is"; KEPT=1; continue
    fi
    ln -s "$tgt" "$here"
    echo "link  $name -> $tgt"
done
mkdir -p "$HOME_DIR/data"

# plant files of the old layout (bfs/data/ops_*.npz, bfs/data/coarse) into their id directories
adopt_plant() {
    local root=$1 kind=$2 idold idnew dst f
    idold=$(cd "$CODE" && $PY bfs/plant_id.py --root "$root" "$kind" 2>/dev/null) || { echo "   $kind: no id for $root"; KEPT=1; return 0; }
    idnew=$(cd "$CODE" && $PY bfs/plant_id.py "$kind")
    if [ "$idold" != "$idnew" ]; then
        echo "   $kind: the generators of $root differ from this code's (id $idold vs $idnew): its files stay in place"; KEPT=1; return 0
    fi
    dst="$HOME_DIR/data/$kind-$idnew"
    shopt -s nullglob
    local files=("$root"/bfs/data/ops_${kind}.npz "$root"/bfs/data/ops_${kind}_range*.npz)
    [ "$kind" = coarse ] && files+=("$root/bfs/data/coarse")
    shopt -u nullglob
    for f in "${files[@]}"; do
        [ -e "$f" ] || continue
        if [ -e "$dst/$(basename "$f")" ]; then echo "   kept in place (already in $dst): $f"; KEPT=1
        else mkdir -p "$dst"; mv -n "$f" "$dst/"; echo "   $f -> $dst/"; fi
    done
}

if [ -n "${FROM:-}" ]; then
    OLD=$(cd "$FROM" && pwd)
    [ "$OLD" != "$CODE" ] || { echo "*** FROM is this code directory"; exit 1; }
    echo "from  $OLD"
    for name in "${LINKS[@]}"; do
        if [ -d "$OLD/$name" ] && [ ! -L "$OLD/$name" ]; then
            echo "move  $OLD/$name/* -> $HOME_DIR/$name/"
            move_into "$OLD/$name" "$HOME_DIR/$name"
        fi
    done
    if [ -d "$OLD/bfs/data" ] && [ ! -L "$OLD/bfs/data" ]; then
        echo "plant files of $OLD/bfs/data:"
        adopt_plant "$OLD" coarse
        adopt_plant "$OLD" fx
    fi
fi
[ "$KEPT" = 0 ] || echo "*** some entries were left in place (above): nothing was deleted; resolve by hand"
echo "done.  Runs: $HOME_DIR/testplant_bfs, logs: $HOME_DIR/logs, plant files: $HOME_DIR/data/<kind>-<id>"
