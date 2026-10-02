#!/bin/bash
#SBATCH --job-name=bfs-codesign
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=240G
#SBATCH --time=15:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/bfs_codesign_%A_%a.log
#
#   sbatch bfs/slurm/codesign.slurm                  (after bfs/slurm/smoke.slurm passed)
#   sbatch --array=0-15 bfs/slurm/codesign.slurm     (more starts)
#
# SISO co-design on the backward-facing step (BFS.md 5): one HIFOO start per
# array task (bfs/hifoo_init.py, SEED, start = the array index: HIFOO's
# controller draw at the devices of Barbagallo et al. 2012, B2 and C1), each on
# 24 cores -- rank 0 the driver, 23 MUMPS workers (bfs/mpipool.py) -- and 240
# GB: TWO starts per 96-core, 512-GB node.  The gamma certificate factors the
# 2n even pencil (58 s, 3.9 GB in the sandbox at n = 191,720) in every worker
# at once.  Order-5 real controller; actuator: Gaussian body force on the
# vertical momentum, free centre; sensor: floor shear-stress window, free start
# (BFS.md 2).  gamma CERTIFIED on [0, 10] every round; at the end, the
# all-frequency winding count.
#
# Needs bfs/slurm/plant_cert.slurm first (the plant's numerical range and
# certified poles, once per mesh).
#
# Time: the driver stops starting rounds when one more would not fit before
# DEADLINE (14 h 15 min after its start; with the driver's 30 min reserve the
# last round ends by 13 h 45, leaving 75 min for the final certificate, the
# winding count, the save and the figures), so the job finishes inside its 15 h with a record
# (OUT/start_XXX/K_B_C_s0_bfs.json/.npz).  Re-submitting the same command
# resumes every unfinished start from its checkpoint (accepted designs, the
# working set and the band are stored); a finished start is not redone.
set -eo pipefail
trap 'rc=$?; echo ""; echo "*** FAILED: line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ./env.sh
LAUNCHER=${LAUNCHER:-mpiexec}
SEED=${SEED:-0}; NSTART=${NSTART:-64}
OUT=${OUT:-results_bfs}
IDX=${SLURM_ARRAY_TASK_ID:-0}
NAME=$(printf "hifoo_%03d" "$IDX")
RUN=$OUT/start_$(printf "%03d" "$IDX")
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none} (array ${SLURM_ARRAY_JOB_ID:-}_$IDX)  $(date -Is) ==="
echo "=== python: $(command -v python) ($(python -V 2>&1)); start $NAME (seed $SEED) -> $RUN ==="
[ -f bfs/data/ops_fx.npz ] || { echo "*** bfs/data/ops_fx.npz missing: run bfs/slurm/smoke.slurm first"; exit 1; }
[ -f bfs/data/ops_fx_range.npz ] || { echo "*** bfs/data/ops_fx_range.npz missing: run bfs/slurm/plant_cert.slurm first"; exit 1; }
post() {      # the figures and data of the saved record (bfs/postproc.py, BFS.md 7)
    $LAUNCHER -n "${SLURM_NTASKS:-24}" python -u bfs/postproc.py --ops bfs/data/ops_fx.npz --rec "$RUN/K_B_C_s0_bfs.npz" \
        || echo "*** postproc failed (the record is saved; rerun: sbatch the same array)"
}
if [ -f "$RUN/K_B_C_s0_bfs.json" ] && [ ! -f "$RUN/K_B_C_s0_bfs_ckpt.npz" ]; then
    echo "start $NAME already finished ($RUN/K_B_C_s0_bfs.json)"
    [ -f "$RUN/K_B_C_s0_bfs_post/post.json" ] || post
    exit 0
fi
# the catalogue is deterministic in (SEED, NSTART): every array task draws the
# same set into its own directory and takes its own entry
mkdir -p "$RUN"
python bfs/hifoo_init.py --ops bfs/data/ops_fx.npz --n "$NSTART" --seed "$SEED" --out "$RUN/catalogue" > "$RUN.catalogue.json"
export OPS=bfs/data/ops_fx.npz OUTER=${OUTER:-40} DEADLINE=${DEADLINE:-51300} PY=python
export LAUNCH="$LAUNCHER -n ${SLURM_NTASKS:-24}"
bash bfs/run_bfs.sh "$RUN" "$RUN/catalogue/$NAME.npz"
# the figures of the saved record, finished or stopped at the deadline (then they show the progress)
[ -f "$RUN/K_B_C_s0_bfs.json" ] && post
echo "=== done $(date -Is) ==="
