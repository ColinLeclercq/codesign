#!/bin/bash
#SBATCH --job-name=bfs-plantcert
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bfs_plantcert_%j.log
#
#   sbatch bfs/slurm/plant_cert.slurm          (once per mesh, after smoke.slurm)
#
# The plant's design-independent stability data, cached next to the operator
# file (bfs/plant_cert.py, BFS.md 4): the numerical range of A_vv on the
# divergence-free velocities (right edge, S-lemma heights, the H^1 constant of
# the wall-shear tail) certified by inertia, and the plant eigenvalues right of
# -eps - 0.6 certified complete.  Every production run reads them.  Test plant:
# 41 + 36 min serial; here the depths and tiles spread over 47 workers.
set -eo pipefail
trap 'rc=$?; echo ""; echo "*** FAILED: line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ./env.sh
LAUNCHER=${LAUNCHER:-mpiexec}
OPS=${OPS:-bfs/data/ops_fx.npz}
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)  ops $OPS ==="
[ -f "$OPS" ] || { echo "*** $OPS missing: run bfs/slurm/smoke.slurm first"; exit 1; }
time $LAUNCHER -n "${SLURM_NTASKS:-48}" python -u bfs/plant_cert.py --ops "$OPS" --eps 0.05
ls -la "${OPS%.npz}"_range*.npz
echo "=== done $(date -Is) ==="
