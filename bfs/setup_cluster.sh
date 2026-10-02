#!/bin/bash
# setup_cluster.sh -- ONE-TIME additions to the conda env `codesign` for the
# backward-facing step (BFS.md 7).  Run on the login node, from the repository
# root:
#
#     bash bfs/setup_cluster.sh
#
# env.sh's env already has dolfinx 0.11 / complex PETSc 3.25 / SLEPc 3.25
# (MUMPS inside), mpi4py, torch and pyGRANSO (check_env.py, 6/6).  The BFS fork
# also needs:
#   clarabel   the SQP subproblem's conic solver (sqp_inner.py; GL used it too,
#              but check_env.py never checked it)
#   qdldl      the sparse LDL^T whose inertia certifies the numerical-range
#              bounds without M^-1 (bfs/mfree.py; MUMPS gives no inertia in
#              complex arithmetic)
#   meshio     only for bfs/mesh_coarse.py (the small test plant)
#   h5py       idem
#   matplotlib the figures of bfs/postproc.py (run at the end of every start)
# Each missing one is pip-installed INTO the active env (codesign's
# $CONDA_PREFIX), so the env itself is completed, once; present ones are left
# alone.  Only matplotlib brings its own small dependencies (pillow,
# kiwisolver, ...); numpy/scipy/PETSc are never upgraded (pip installs only
# what is missing).
set -eo pipefail
cd "$(dirname "$0")/.."
source ./env.sh
echo "python: $(command -v python) ($(python -V 2>&1))"
for mod in clarabel qdldl meshio h5py matplotlib; do
    if python -c "import $mod" 2>/dev/null; then
        echo "  $mod: present ($(python -c "import $mod; print(getattr($mod, '__version__', '?'))"))"
    else
        echo "  $mod: MISSING -> pip install $mod (into $CONDA_PREFIX)"
        python -m pip install $([ "$mod" = matplotlib ] || echo --no-deps) "$mod"
        python -c "import $mod" && echo "  $mod: installed"
    fi
done
# the pieces the fork imports, in one go (no MPI_Init here: petsc4py is
# imported without initialising on a login node)
python - <<'EOF'
import numpy, scipy, clarabel, qdldl, torch
from pygranso.pygranso import pygranso
import mpi4py, petsc4py, slepc4py, dolfinx
print(f"numpy {numpy.__version__}, scipy {scipy.__version__}, clarabel {clarabel.__version__}, "
      f"torch {torch.__version__}, dolfinx {dolfinx.__version__}, petsc4py {petsc4py.__version__}, "
      f"slepc4py {slepc4py.__version__}, mpi4py {mpi4py.__version__}")
EOF
echo "setup_cluster.sh: done.  Next: sbatch bfs/slurm/smoke.slurm"
