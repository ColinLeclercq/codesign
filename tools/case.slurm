#!/bin/bash
#SBATCH --comment=36973004F
#SBATCH --job-name=case
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=05:59:00
#SBATCH --output=logs/case_%j.out
#SBATCH --error=logs/case_%j.err
#
# ONE case, started from EVERY parent in turn.
#
# This is what makes a nesting violation structurally impossible.  A child's
# feasible set contains each parent's design, and `gbest` inside the exchange
# loop is monotone, so a run started from a parent CANNOT finish above that
# parent's gamma.  Starting from every parent therefore bounds the child by the
# BEST of them -- and the no-regress guard keeps whichever attempt won.
#
# A case with no parents is a root and is cold started; there is nothing to
# inherit and nothing it must dominate.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ./env.sh
# One thread per worker.  The parallelism is ACROSS processes -- |Omega|
# independent gain evaluations -- and a threaded BLAS nested inside them is
# what produced SIGSEGV.  run_case_epi reads SLURM_CPUS_PER_TASK and sizes
# its pool to the whole allocation, so --cpus-per-task above sets it.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
echo "allocation: ${SLURM_CPUS_PER_TASK:-?} cores"

CASE=${CASE:?set CASE}
NE=${NE:-200}; EPS=${EPS:-0.05}; OUT=${OUT:-results}
OUTER=${OUTER:-40}; WALL=${WALL:-18000}
# WALL is the budget for the WHOLE job and is DIVIDED among the attempts: a root
# case splits it across the seed catalogue, a child across its parents plus its
# donors plus a polish.  With 8 seeds and WALL=1600 each attempt got 200 s --
# far too little (FB+FC needed 750 s to converge from cold), which is why the
# KSP job timed out having barely moved.  Budget per ATTEMPT, not per job:
# WALL should be about (number of attempts) x 600 s.  --time above must exceed
# WALL with margin for the seed synthesis and verification either side.
TAG="${CASE//+/_}_s0_ne${NE}"

PARENTS=$(python tools/nesting.py --parents "$CASE")
DONORS=$(python tools/nesting.py --donors "$CASE")
NSTART=${NSTART:-3}
echo "=== $(hostname): $CASE  ne=$NE eps=$EPS ==="

if [ -z "$PARENTS" ]; then
  # ROOT: nothing to inherit, so MULTISTART over the SEED CATALOGUE.
  #
  # The seeds are physically motivated, not random jitter.  prepare.py builds
  # SIXTEEN controllers (see the catalogue in seeds.py):
  #
  #   lqgbt              1  classical LQG, FULL-FIELD noise (B1 = I).  Differs
  #                         in the NOISE MODEL, not the authority.
  #   hinf_(dx)          3  DGKF central controller on rank-one channels,
  #                         disturbance at -dx and performance at +dx,
  #                         dx = 1,2,3, gamma bisected to near-optimal.
  #   lqg_(dx)_r(rho)   12  LQG on the SAME channels, dx = 1,2,3 crossed with
  #                         rho = 1e-4, 1e-2, 1, 1e2.  rho = 1e-4 is the
  #                         HIGH-GAIN limit, where the H-infinity optimum
  #                         usually lives; rho = 1e2 the cheap-control end.
  #
  # So the catalogue varies WHERE the exogenous channels sit (dx), HOW MUCH
  # control authority is bought (rho), and WHICH synthesis is used (LQG vs
  # H-infinity) -- three independent axes.  All are balanced-truncated to one
  # common order, taken from the shoulder in the Hankel singular values, so any
  # seed can warm start any case.  Random jitter around a single seed explores
  # far less.
  #
  # Fields are NOT multistarted: they always begin as the nominal Gaussians at
  # their default positions.  Static-gain cases begin at D_c = 0.
  # Only seeds that already fit the band are usable: one whose poles sit at
  # |Im| = 56 cannot start feasible against a cap of 20, and letting it widen
  # the band instead would make gammas from different runs incomparable.
  SEEDS=$(WMAX=${WMAX:-20} python - <<'PY'
import numpy as np, os, sys
sys.path.insert(0, '.')
import controller as ctl
f = f"seeds_ne{os.environ.get('NE','200')}.npz"
wmax = float(os.environ.get('WMAX', 20))
try:
    sd = np.load(f, allow_pickle=True)
    keep, drop = [], []
    for nm in sd['names']:
        nm = str(nm)
        A = ctl.A_of(sd[f'{nm}__d'], sd[f'{nm}__e'], sd[f'{nm}__f'])
        im = float(np.max(np.abs(np.linalg.eigvals(A).imag)))
        (keep if im <= wmax else drop).append((nm, im))
    if drop:
        print('# excluded (|Im| > %g): ' % wmax
              + ', '.join(f'{n}:{v:.0f}' for n, v in drop), file=sys.stderr)
    print(' '.join(n for n, _ in keep))
except Exception as e:
    print('', file=sys.stdout)
PY
)
  MODE=$(python -c "
import sys; sys.path.insert(0,'.')
from fe_codesign import parse_case
f = parse_case('$CASE')
print(f.get('ctrl_mode') if f.get('K') else 'base')")
  if [ "$MODE" = "gain" ] || [ "$MODE" = "gainr" ] || [ -z "$SEEDS" ]; then
    # a static gain has one free number and starts at D_c = 0: nothing to vary
    echo "--- root case, $MODE: single cold start ---"
    python run_case_epi.py --case "$CASE" --start 0 --ne "$NE" \
           --wall "$WALL" --outer "$OUTER" --eps "$EPS" --cold --outdir "$OUT"
  else
    ns=$(echo "$SEEDS" | wc -w)
    per=$(( WALL / ns ))
    echo "--- root case: $ns catalogue seeds, ${per}s each ---"
    for S in $SEEDS; do
      echo "--- seed $S ---"
      python run_case_epi.py --case "$CASE" --start 0 --ne "$NE" \
             --wall "$per" --outer "$OUTER" --eps "$EPS" --cold \
             --seed "$S" --outdir "$OUT" || true
    done
  fi
else
  # split the budget across the parents; each attempt is bounded by its parent
  n=$(( $(echo "$PARENTS" | wc -w) + $(echo "$DONORS" | wc -w) + 1 ))
  per=$(( WALL / n ))
  echo "--- parents: $PARENTS  ($n attempts, ${per}s each) ---"
  for P in $PARENTS; do
    SRC="$OUT/${P//+/_}_s0_ne${NE}.npz"
    if [ ! -f "$SRC" ]; then
      echo "  parent $P has no result yet -- skipped"
      continue
    fi
    echo "--- from $P ---"
    python run_case_epi.py --case "$CASE" --start 0 --ne "$NE" \
           --wall "$per" --outer "$OUTER" --eps "$EPS" \
           --resume "$SRC" --outdir "$OUT" || true
  done
  # FIELD DONORS: not nesting parents, so they carry NO guarantee, but their
  # optimised FIELDS are a good start.  KG+FBR+FCR cannot reproduce FBR+FCR's
  # design (KG forces n = 0) yet FBR+FCR's fields are exactly where to begin --
  # optimise the fields first, then swap the compensator.  The no-regress guard
  # discards the attempt if it does not help.
  for Dn in $DONORS; do
    SRC="$OUT/${Dn//+/_}_s0_ne${NE}.npz"
    [ -f "$SRC" ] || continue
    echo "--- from field donor $Dn (no guarantee) ---"
    python run_case_epi.py --case "$CASE" --start 0 --ne "$NE" \
           --wall "$per" --outer "$OUTER" --eps "$EPS" \
           --resume "$SRC" --outdir "$OUT" || true
  done
  # and once more from its own best so far, to polish
  if [ -f "$OUT/$TAG.npz" ]; then
    echo "--- polish from its own best ---"
    python run_case_epi.py --case "$CASE" --start 0 --ne "$NE" \
           --wall "$per" --outer "$OUTER" --eps "$EPS" \
           --resume "$OUT/$TAG.npz" --outdir "$OUT" || true
  fi
fi

echo "--- verification ---"
python verify_dense.py --eps "$EPS" --ne "$NE" --results "$OUT" 2>&1 \
  | grep -E "^${CASE//+/\\+} |UNSTABLE" || true
