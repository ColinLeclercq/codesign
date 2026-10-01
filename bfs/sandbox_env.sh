# sandbox_env.sh -- the sandbox's FEniCSx/PETSc environment (BFS.md 7).
# Venv /opt/fenicsx: Ubuntu 24.04's complex PETSc 3.19.6 / SLEPc 3.19.2 (MUMPS
# inside, Open MPI), petsc4py/slepc4py rebuilt against them, basix 0.11 built
# from source with the venv's nanobind 3.1, DOLFINx 0.11 (C++ in
# /opt/fenicsx/lib) built from source.  (The cluster uses env.sh: the user's
# conda env `codesign`.)
export PETSC_DIR=/usr/lib/petscdir/petsc3.19/x86_64-linux-gnu-complex
export SLEPC_DIR=/usr/lib/slepcdir/slepc3.19/x86_64-linux-gnu-complex
export LD_LIBRARY_PATH=/opt/fenicsx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=/opt/fenicsx/bin:$PATH
export PY=/opt/fenicsx/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
