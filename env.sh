# env.sh — sourced by every job.  Uses the ALREADY-VALIDATED conda env
# `codesign` (dolfinx 0.11 + complex PETSc 3.25.3 + SLEPc 3.25.1 + torch 2.6
# + pyGRANSO patched for numpy>=2).  Verified 6/6 by check_env.py.
#
# NOTE: no `set -u` anywhere near the conda activation — the site's older
# anaconda conda.sh references an unbound PS1 and aborts under `set -u`.

module purge
module load python/3.8.14-intel2220     # site module (compilers / MPI libs)

# conda AFTER the module: its python (3.12) then takes precedence on PATH.
source "$(conda info --base)/etc/profile.d/conda.sh"

# If codesign is ALREADY active, `conda activate codesign` is a no-op: conda
# sees the env it is being asked for and makes no PATH change.  But the module
# load above has just put the site python 3.8 in front, and it stays there.
# You then get python 3.8.14 with numpy 1.23.4 and no torch, from a script
# whose whole purpose is to select 3.12 -- and `conda env list` still shows
# codesign starred, so it looks correct.
#
# sbatch defaults to --export=ALL, so a job submitted from such a shell
# inherits it.  Deactivating first makes the activate real in every case.
conda deactivate 2>/dev/null || true
conda activate codesign

# stop ~/.local (python3.8 site-packages) from shadowing the conda env
export PYTHONNOUSERSITE=1

# Let the conda env's own libstdc++ win.
#
# numpy's C extension needs GLIBCXX_3.4.29.  On the login node something in the
# default environment supplies it, so this was invisible there.  On a COMPUTE
# node it is not supplied: job 15966182 on n046 reported
#
#   /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by
#   .../site-packages/numpy/_core/_multiarray_umath...so)
#
# and torch and pyGRANSO both failed to import -- while numpy itself passed,
# because it had already been imported by a probe that did not trip the same
# resolution order.  The conda env ships a new enough libstdc++ in its own lib
# directory; this simply puts it first.
#
# PREPEND, and do it AFTER conda activate so CONDA_PREFIX is set.  Note this
# also puts the conda BLAS ahead of the module's MKL, which is a change to what
# runs underneath every number: after adding this, re-run tools/validate.slurm
# and confirm all 27 certified gammas still reproduce before trusting anything
# measured afterwards.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# single-threaded tasks: MKL would otherwise grab all 96 cores PER TASK
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

# --- did the two fixes above actually take effect? ------------------------
# Both failure modes are silent: the wrong python still runs, and the missing
# libstdc++ only surfaces as an "Importing the numpy C-extensions failed"
# traceback from whichever module imports first.  Say so here instead.
if [ -n "${CONDA_PREFIX:-}" ]; then
    case "$(command -v python)" in
        "$CONDA_PREFIX"/*) : ;;
        *) echo "env.sh WARNING: python is $(command -v python), not from" \
                "$CONDA_PREFIX -- a site module has shadowed it." >&2 ;;
    esac
    case ":$LD_LIBRARY_PATH:" in
        *":$CONDA_PREFIX/lib:"*) : ;;
        *) echo "env.sh WARNING: $CONDA_PREFIX/lib is not on LD_LIBRARY_PATH;" \
                "torch/pyGRANSO will fail on compute nodes (GLIBCXX)." >&2 ;;
    esac
fi
