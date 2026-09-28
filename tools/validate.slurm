#!/bin/bash
#SBATCH --job-name=gl-validate
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=01:29:00
#SBATCH --output=logs/validate_%j.log
#
#   sbatch tools/validate.slurm
#   NE=200 OUT=results sbatch tools/validate.slurm
#
# The full validation chain: environment gate, selftest, the three certificates
# over all 27 archived designs, the section 3 table, the stage-0 harness replay,
# and a checksum proving results/ was not written to.  The 2026-09-11 run took
# 24 minutes; the hour and a half is margin.
#
# ONE LOG, NOT TWO.  An earlier version of this file sent stdout to _%j.out and
# stderr to _%j.err.  A failure then put the diagnosis in one file and the
# context in the other, interleaved with nothing, and the run looked like it had
# died silently.  Both streams now go to a single _%j.log in order.
#
# NO BLANKET srun.  That earlier version also wrapped the WHOLE script in srun,
# to give the PETSc probe a PMIX transport key.  That is a large hammer for one
# small nail: it puts every step -- selftest, the certificates, the harness and
# its worker pool -- inside a job step, and any srun-side problem then kills the
# run before a single line is printed.  srun is now used for exactly the command
# that needs it, inside validate_cleanup.sh, and only when a job allocation is
# present.  check_env.py also isolates each probe in its own process, so an MPI
# abort costs one row even without srun.
#
# --exclusive: the harness replay reports per-case wall times, and a shared node
# makes those meaningless.  --cpus-per-task is what parallel.py reads
# (SLURM_CPUS_PER_TASK) to size the worker pool, so it must be set explicitly.
set -eo pipefail          # NOT -u : the site conda.sh trips on an unbound PS1

# Say where it broke.  Without this, `set -e` exits silently and the only
# evidence is slurm's "Exited with exit code 1".
trap 'rc=$?; echo ""; echo "*** FAILED: ${BASH_SOURCE[0]} line $LINENO"; \
      echo "*** command: $BASH_COMMAND"; echo "*** exit status: $rc"; \
      exit $rc' ERR

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs cleanup_evidence
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is) ==="
echo "=== submit dir: $SLURM_SUBMIT_DIR ==="

# Check the preconditions BEFORE sourcing anything, so a missing file is
# reported as a missing file rather than as a shell error 200 lines later.
for f in env.sh tools/validate_cleanup.sh harness.py check_env.py selftest.py; do
    [ -f "$f" ] || { echo "*** missing: $f"; echo "*** the cluster tree is not "\
        "the one these scripts came from; copy the package over first"; exit 1; }
done

echo "=== sourcing env.sh ==="
source ./env.sh
echo "=== python: $(command -v python)  $(python -V 2>&1) ==="

# ---------------------------------------------------------------------------
# Guard: is `python` actually the conda env's?
#
# env.sh does `module load python/3.8.14-intel2220` and then
# `conda activate codesign`, relying on the activation to put the conda python
# ahead of the module's on PATH.  That works from a CLEAN shell.  It does NOT
# work if codesign is already active: conda treats the activate as a no-op,
# makes no PATH change, and the module python (3.8, numpy 1.23.4, no torch)
# wins.  Measured on f02 -- `. env.sh` in an already-activated shell turns a
# working environment into one where check_env.py reports
# "REQUIRED CHECK FAILED: pytorch, pygranso".
#
# sbatch defaults to --export=ALL, so a job submitted from a shell with
# codesign active inherits CONDA_DEFAULT_ENV and hits exactly this.  The job
# would then fail deep inside selftest with an import error, far from the
# cause.  Catch it here instead.
PYBIN="$(python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
echo "=== python: $PYBIN ($(python -V 2>&1)) ==="
if [ -n "${CONDA_PREFIX:-}" ] && case "$PYBIN" in "$CONDA_PREFIX"/*) false;; *) true;; esac; then
    echo "*** python is NOT from the active conda env"
    echo "***   python      : $PYBIN"
    echo "***   CONDA_PREFIX: $CONDA_PREFIX"
    echo "***   LOADEDMODULES: ${LOADEDMODULES:-none}"
    echo "*** A site module has shadowed the conda interpreter.  This happens"
    echo "*** when env.sh is sourced in a shell where codesign was ALREADY"
    echo "*** activated: the activate becomes a no-op and the module python"
    echo "*** wins.  Submit from a clean shell (conda deactivate first), or"
    echo "*** use sbatch --export=NONE."
    exit 1
fi
python -c "import numpy, torch, pygranso" 2>/dev/null || {
    echo "*** the active python cannot import numpy/torch/pygranso"
    echo "*** python: $PYBIN"
    echo "*** run check_env.py directly to see which one and why"
    exit 1; }

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export NE=${NE:-200}
export OUT=${OUT:-results}

echo "=== SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}  ne=$NE  results=$OUT ==="

bash tools/validate_cleanup.sh

echo "=== validation job complete: $(date -Is) ==="
