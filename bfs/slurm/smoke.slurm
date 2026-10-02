#!/bin/bash
#SBATCH --job-name=bfs-smoke
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/bfs_smoke_%j.log
#
#   bash bfs/setup_cluster.sh          (once, login node)
#   sbatch bfs/slurm/smoke.slurm
#
# The backward-facing step on the cluster, before any long job (BFS.md 7):
#   1. the operators from your nsforms.py (bfs/ops_dolfinx.py; serial, seconds)
#      and the small test plant (bfs/mesh_coarse.py)
#   2. the gains against your archive's gains_fx.txt, with the code's MUMPS
#      (bfs/check_gains.py; expect rel ~1e-15)
#   3. MUMPS on a FULL node: 96 ranks factor and solve at once
#      (bfs/bench_mumps.py) -- the per-task cost behind the run-time estimate
#   4. the MPI task farm end to end: one HIFOO start on the test plant,
#      stabilisation phase + one round, 8 ranks (bfs/run_bfs.sh)
# Send me the log: steps 3 and 4 fix the production job's size.
#
# LAUNCH: the conda env's mpiexec (the MPI petsc4py/mpi4py were built with).
# Set LAUNCHER="srun --mpi=pmix" to try srun instead.
set -eo pipefail
trap 'rc=$?; echo ""; echo "*** FAILED: line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is) ==="
source ./env.sh
echo "=== python: $(command -v python) ($(python -V 2>&1)) ==="
LAUNCHER=${LAUNCHER:-mpiexec}
python -c "import clarabel" || { echo "*** clarabel missing: run bash bfs/setup_cluster.sh on the login node"; exit 1; }

echo "=== 1. operators (dolfinx, nsforms.py) ==="
[ -f bfs/data/ops_fx.npz ] || python bfs/ops_dolfinx.py --data bfs/archive_dsr/fx_data \
    --baseflow baseflow_fenicsx.npz --out bfs/data/ops_fx.npz
[ -f bfs/data/coarse/mesh.xdmf ] || python bfs/mesh_coarse.py
[ -f bfs/data/ops_coarse.npz ] || python bfs/ops_dolfinx.py --data bfs/data/coarse --baseflow baseflow.npz \
    --out bfs/data/ops_coarse.npz

echo "=== 2. gains vs the archive (MUMPS) ==="
time python bfs/check_gains.py --omegas 0.75 2.0

echo "=== 3. MUMPS, 96 ranks at once, n = 191,720 ==="
time $LAUNCHER -n 96 python bfs/bench_mumps.py --ops bfs/data/ops_fx.npz

echo "=== 4. the MPI task farm: HIFOO start, stabilisation + 1 round, test plant, 8 ranks ==="
python bfs/hifoo_init.py --ops bfs/data/ops_coarse.npz --n 1 --seed 0 --out smoke_bfs/starts
export OPS=bfs/data/ops_coarse.npz OUTER=1 LAUNCH="$LAUNCHER -n 8" PY=python
time bash bfs/run_bfs.sh smoke_bfs/run smoke_bfs/starts/hifoo_000.npz
echo "=== smoke test done $(date -Is) ==="
