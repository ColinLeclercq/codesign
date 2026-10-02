#!/bin/bash
#SBATCH --job-name=bfs-smoke
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
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
#   4. the test plant's certified numerical range and plant poles
#      (bfs/plant_cert.py; once, cached)
#   5. one start end to end on the test plant, 24 ranks (bfs/run_bfs.sh):
#      feasible start, one round with gamma certified on [0, 10], the
#      all-frequency winding count
#   6. its figures (bfs/postproc.py)
# Send me the log: steps 3 to 5 fix the production job's size.
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
python -c "import clarabel, qdldl, matplotlib" || { echo "*** clarabel / qdldl / matplotlib missing: run bash bfs/setup_cluster.sh on the login node"; exit 1; }

echo "=== 1. operators (dolfinx, nsforms.py) ==="
# always rebuilt (seconds): the file format grows with the code (the shear trace
# and the stiffness were added on 2026-10-02); a stale range cache goes with it
python bfs/ops_dolfinx.py --data bfs/archive_dsr/fx_data --baseflow baseflow_fenicsx.npz --out bfs/data/ops_fx.npz
[ -f bfs/data/coarse/mesh.xdmf ] || python bfs/mesh_coarse.py
python bfs/ops_dolfinx.py --data bfs/data/coarse --baseflow baseflow.npz --out bfs/data/ops_coarse.npz
rm -f bfs/data/ops_coarse_range*.npz

echo "=== 2. gains vs the archive (MUMPS) ==="
time python bfs/check_gains.py --omegas 0.75 2.0

echo "=== 3. MUMPS, 96 ranks at once, n = 191,720 ==="
time $LAUNCHER -n 96 python bfs/bench_mumps.py --ops bfs/data/ops_fx.npz

echo "=== 4. the test plant's certified range and poles (bfs/plant_cert.py), 24 ranks ==="
time $LAUNCHER -n 24 python -u bfs/plant_cert.py --ops bfs/data/ops_coarse.npz

echo "=== 5. one start end to end on the test plant, 24 ranks: feasible start, one round with gamma"
echo "===    certified on [0, 10], the all-frequency winding count at the end ==="
python bfs/hifoo_init.py --ops bfs/data/ops_coarse.npz --n 1 --seed 0 --out smoke_bfs/starts
export OPS=bfs/data/ops_coarse.npz OUTER=1 LAUNCH="$LAUNCHER -n 24" PY=python
time bash bfs/run_bfs.sh smoke_bfs/run smoke_bfs/starts/hifoo_000.npz

echo "=== 6. its figures (bfs/postproc.py), 24 ranks: singular values with/without control, modes, Bode ==="
time $LAUNCHER -n 24 python -u bfs/postproc.py --ops bfs/data/ops_coarse.npz --rec smoke_bfs/run/K_B_C_s0_bfs.npz
ls smoke_bfs/run/K_B_C_s0_bfs_post
echo "=== smoke test done $(date -Is) ==="
