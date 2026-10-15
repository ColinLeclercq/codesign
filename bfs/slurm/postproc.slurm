#!/bin/bash
#SBATCH --comment=36973004F
#SBATCH --job-name=bfs-post
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bfs_post_%j.log
#
#   REC=testplant_bfs/zero_o10_s0_r300/run/K_B_C_s0_bfs.npz sbatch bfs/slurm/postproc.slurm
#   REC=results_bfs/zero_000/K_B_C_s0_bfs.npz OPS=$(python bfs/plant_id.py --dir fx)/ops_fx.npz sbatch bfs/slurm/postproc.slurm
#
# The figures and post.npz of a saved record (bfs/postproc.py, BFS.md 5), on
# their own: when the run's own (--post 1) did not make them.  OPS is the
# operator file the record was computed on (default: the test plant).
# Test plant: about 1 min on 24 ranks; full mesh 5 to 10 min.
set -eo pipefail
trap 'rc=$?; echo ""; echo "*** FAILED: line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ./env.sh
python bfs/version_check.py || { echo "*** mixed code versions (bfs/version_check.py): stop"; exit 1; }
[ -n "$REC" ] && [ -f "$REC" ] || { echo "*** REC=<record .npz> required (got '$REC')"; exit 1; }
OPS=${OPS:-$(python bfs/plant_id.py --dir coarse)/ops_coarse.npz}     # the test plant of this code's plant id
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)  $REC on $OPS ==="
case "$REC" in */run/*) echo "=== -> ${REC%%/run/*} ===" ;; esac      # its run directory, for bfs/report.sh
time ${LAUNCHER:-mpiexec} -n "${SLURM_NTASKS:-24}" python -u bfs/postproc.py --ops "$OPS" --rec "$REC"
echo "=== done $(date -Is) ==="
