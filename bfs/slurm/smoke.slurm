#!/bin/bash
#SBATCH --comment=36973004F
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
python bfs/version_check.py || { echo "*** mixed code versions (bfs/version_check.py): stop"; exit 1; }
echo "=== python: $(command -v python) ($(python -V 2>&1)) ==="
LAUNCHER=${LAUNCHER:-mpiexec}
python -c "import clarabel, qdldl, matplotlib" || { echo "*** clarabel / qdldl / matplotlib missing: run bash bfs/setup_cluster.sh on the login node"; exit 1; }

echo "=== 1. operators (dolfinx, nsforms.py) ==="
# into their content-id directories (bfs/plant_id.py, 20261010j): built once
# per id (seconds), never rebuilt in place -- the id changes with the
# generators, so a new file format makes a new directory -- under the build
# lock of bfs/lib.sh, written under private names and renamed whole
source bfs/lib.sh
export HDF5_USE_FILE_LOCKING=FALSE
PDF=$(python bfs/plant_id.py --dir fx); PDC=$(python bfs/plant_id.py --dir coarse)
OPSF=$PDF/ops_fx.npz; OPSC=$PDC/ops_coarse.npz
mkdir -p "$PDF" "$PDC"
echo "=== plant files: $PDF, $PDC ==="
take_lock "$PDF/.build.lockd" "full mesh operators"
[ -f "$OPSF" ] || { python bfs/ops_dolfinx.py --data bfs/archive_dsr/fx_data --baseflow baseflow_fenicsx.npz --out "$PDF/.ops.$$.npz"; mv "$PDF/.ops.$$.npz" "$OPSF"; }
drop_lock "$PDF/.build.lockd"
take_lock "$PDC/.build.lockd" "test plant's files"
if [ ! -f "$PDC/coarse/mesh.xdmf" ]; then
    rm -rf "$PDC/.coarse.$$"; python bfs/mesh_coarse.py --out "$PDC/.coarse.$$"
    [ -e "$PDC/coarse" ] && mv "$PDC/coarse" "$PDC/.coarse.broken.$$"
    mv "$PDC/.coarse.$$" "$PDC/coarse"
fi
[ -f "$OPSC" ] || { python bfs/ops_dolfinx.py --data "$PDC/coarse" --baseflow baseflow.npz --out "$PDC/.ops.$$.npz"; mv "$PDC/.ops.$$.npz" "$OPSC"; }

echo "=== 2. gains vs the archive (MUMPS) ==="
time python bfs/check_gains.py --omegas 0.75 2.0

echo "=== 3. MUMPS, 96 ranks at once, n = 191,720 ==="
time $LAUNCHER -n 96 python bfs/bench_mumps.py --ops "$OPSF"

echo "=== 4. the test plant's certified range and poles (bfs/plant_cert.py), 24 ranks ==="
[ -f "$PDC/ops_coarse_range.npz" ] || time $LAUNCHER -n 24 python -u bfs/plant_cert.py --ops "$OPSC"
drop_lock "$PDC/.build.lockd"

echo "=== 5. one start end to end on the test plant, 24 ranks: feasible start, one round with gamma"
echo "===    certified on [0, 10], the all-frequency winding count at the end ==="
# its own directory per job (smoke_bfs is shared by every code version since
# 20261010i: a previous version's record must not be re-validated)
SM=smoke_bfs/$(sed -n 's/^# version //p' bfs/MANIFEST)_${SLURM_JOB_ID:-$$}
echo "=== -> $SM ==="
python bfs/hifoo_init.py --ops "$OPSC" --n 1 --seed 0 --out "$SM/starts"
export OPS=$OPSC OUTER=1 LAUNCH="$LAUNCHER -n 24" PY=python
time bash bfs/run_bfs.sh "$SM/run" "$SM/starts/hifoo_000.npz"

echo "=== 6. its figures (bfs/postproc.py), 24 ranks: singular values with/without control, modes, Bode ==="
time $LAUNCHER -n 24 python -u bfs/postproc.py --ops "$OPSC" --rec "$SM/run/K_B_C_s0_bfs.npz"
ls "$SM/run/K_B_C_s0_bfs_post"
echo "=== smoke test done $(date -Is) ==="
