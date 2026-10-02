#!/bin/bash
#SBATCH --job-name=bfs-codesign
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=15:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/bfs_codesign_%A_%a.log
#
#   sbatch bfs/slurm/codesign.slurm                  (after bfs/slurm/smoke.slurm passed)
#   sbatch --array=0-15 bfs/slurm/codesign.slurm     (more starts)
#
# SISO co-design on the backward-facing step (BFS.md 5): one HIFOO start per
# array task (bfs/hifoo_init.py, SEED, start = the array index), each on 24
# cores -- rank 0 the driver, 23 MUMPS workers (bfs/mpipool.py) -- so slurm
# packs FOUR starts per 96-core node.  Measured (BFS.md 7): the pool's calls
# hold ~14 tasks on average, so 23 workers do a round as fast as 95; memory
# ~3 GB per rank (one 1-GB factor, two cached).  Order-5 real controller,
# actuator (y-momentum) and sensor (u) Gaussians of std 0.5 with free centres
# at least --bfs-margin inside the domain.
#
# Time: the driver stops starting rounds when one more would not fit before
# DEADLINE (14 h 30 min after its start, with 30 min reserved for the final
# certificate and save), so the job finishes inside its 15 h with a record
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
if [ -f "$RUN/K_B_C_s0_bfs.json" ] && [ ! -f "$RUN/K_B_C_s0_bfs_ckpt.npz" ]; then
    echo "start $NAME already finished ($RUN/K_B_C_s0_bfs.json)"; exit 0
fi
# the catalogue is deterministic in (SEED, NSTART): every array task draws the
# same set into its own directory and takes its own entry
mkdir -p "$RUN"
python bfs/hifoo_init.py --ops bfs/data/ops_fx.npz --n "$NSTART" --seed "$SEED" --out "$RUN/catalogue" > "$RUN.catalogue.json"
export OPS=bfs/data/ops_fx.npz OUTER=${OUTER:-40} DEADLINE=${DEADLINE:-52200} PY=python
export LAUNCH="$LAUNCHER -n ${SLURM_NTASKS:-24}"
bash bfs/run_bfs.sh "$RUN" "$RUN/catalogue/$NAME.npz"
echo "=== done $(date -Is) ==="
