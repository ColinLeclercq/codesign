#!/bin/bash
#SBATCH --job-name=bfs-checklat
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=450G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bfs_checklat_%j.log
#
# tools/check_lattice.py on the MPI ranks (BFS.md 5, 8): the lattice's rows,
# followed peaks, LEVEL SET (in-solve missed peaks; with LEVEL_REF=1 against the
# 2n certificate, certify_sparse.certify_greater), line and strip at a record.
#
#   REC=path/to/K_B_C_s0_bfs_mimo2x2_central.npz MIMO="2 2 central" sbatch bfs/slurm/check_lattice.slurm
#   REC=path/to/K_B_C_s0_bfs.npz sbatch bfs/slurm/check_lattice.slurm            (SISO)
#   LEVEL_REF=0 ...    (skip the 2n reference: minutes per decision)
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ./env.sh
LAUNCHER=${LAUNCHER:-mpiexec}
[ -n "${REC:-}" ] || { echo "check_lattice.slurm: set REC=record.npz"; exit 2; }
ARGS=("$REC" --ops "${OPS:-bfs/data/ops_coarse.npz}" --cap "${LATTICE_CAP:-24}")
[ -n "${MIMO:-}" ] && ARGS+=(--mimo $MIMO)
[ "${LEVEL_REF:-1}" = 1 ] && ARGS+=(--level-ref)
time $LAUNCHER -n "${SLURM_NTASKS:-96}" python -u tools/check_lattice.py "${ARGS[@]}"
