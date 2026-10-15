#!/bin/bash
# stage.sh -- the directory to submit this code version from (BFS.md 7).
#
#   cd /tmp_user/sator/cleclerc/BFS/codesign && git pull && bash bfs/stage.sh
#       -> BFS/v/<version>: a git worktree at this commit, linked to the shared
#          runs, logs and plant files (bfs/home.sh); prints the cd to submit from
#
# Why (the layout review of 20261010i): a git pull in the directory jobs run
# from changes the code under them -- the driver imports some modules only at
# the end of a 13 h run, and queued jobs start on whatever the directory then
# holds.  A worktree per version is never pulled: the code a job starts with is
# the code it ends with, as the per-version directories of 20261010a-g were.
# <version> is the MANIFEST's label; a label already staged at another commit
# is refused (the MANIFEST must change with the code: bfs/version_check.py).
# Old worktrees are removed by hand once no job runs from them:
#   git worktree remove ../v/<version>     (refused by git if it holds changes)
set -eo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
cd "$CODE"
ALLOW_CLONE=1 python3 bfs/version_check.py > /dev/null || { ALLOW_CLONE=1 python3 bfs/version_check.py; echo "*** this clone is not one consistent version: commit/pull first"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "*** the clone has local changes (git status): a worktree holds only commits"; exit 1; }
HOME_DIR=$(python3 -c "import sys; sys.path.insert(0, 'bfs'); import plant_id; print(plant_id.bfs_home() or '')")
[ -n "$HOME_DIR" ] || { echo "*** no BFS_HOME above $CODE: INIT=1 bash bfs/home.sh <root> first"; exit 1; }
VER=$(sed -n 's/^# version //p' bfs/MANIFEST)
HEAD=$(git rev-parse HEAD)
DEST=$HOME_DIR/v/$VER
if [ -d "$DEST" ]; then
    have=$(git -C "$DEST" rev-parse HEAD 2>/dev/null || echo none)
    [ "$have" = "$HEAD" ] || { echo "*** $DEST exists at commit ${have:0:12}, this clone is at ${HEAD:0:12} with the same label $VER: the MANIFEST label must change with the code"; exit 1; }
    echo "$DEST already staged at ${HEAD:0:12}"
else
    mkdir -p "$HOME_DIR/v"
    git worktree add -q --detach "$DEST" "$HEAD"
    echo "staged $VER at ${HEAD:0:12}: $DEST"
fi
( cd "$DEST" && bash bfs/home.sh "$HOME_DIR" | grep -v '^ok ' )
# the plant files of this version's id (bfs/plant_id.py): complete, or what to do
for k in coarse fx; do
    d=$(cd "$DEST" && python3 bfs/plant_id.py --dir "$k")
    if [ -f "$d/ops_$k.npz" ] && [ -f "$d/ops_${k}_range.npz" ]; then
        echo "plant $k: $d (complete)"
    else
        echo "plant $k: $d MISSING -- 'python3 bfs/plant_id.py --adopt $k' lists directories built by"
        echo "        unchanged generators under another id; else $([ "$k" = coarse ] && echo "PLANT_ONLY=1 sbatch --ntasks=48 bfs/slurm/testplant.slurm first, then the batch with --dependency=afterok:<that job>" || echo "smoke.slurm step 1 and plant_cert.slurm (full mesh)")"
    fi
done
echo
echo "submit from there:  cd $DEST && sbatch ... bfs/slurm/testplant.slurm"
