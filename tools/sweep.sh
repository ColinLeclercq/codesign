#!/bin/bash
# sweep.sh — the whole study in ONE pass, with no possible nesting violation.
#
#   bash sweep.sh                          # the FULL 79-case lattice
#   GL_CASES=standard bash sweep.sh        # the original 19-case subset
#   EPS=0.001 OUT=results_eps3 bash sweep.sh
#
# Waves are DERIVED from the nesting relations (nesting.py), so every parent
# finishes before its children start.  Each child is then run from EVERY parent
# in turn, and the no-regress guard keeps the best attempt.  Since a child's
# feasible set contains each parent's design and gbest is monotone, the child
# cannot finish above the best parent -- a violation is structurally impossible
# rather than something to check for afterwards.
#
# Roots are cold started.  There is no second pass.
set -eo pipefail
cd "$(dirname "$0")/.."   # repository root: run_case_epi.py and the certification stack live there
NE=${NE:-200}; EPS=${EPS:-0.05}; OUT=${OUT:-results}
OUTER=${OUTER:-40}; WALL=${WALL:-18000}
mkdir -p logs "$OUT"
NCASE=$(python tools/nesting.py --waves | wc -w)
NWAVE=$(python tools/nesting.py --waves | wc -l)
echo "ne=$NE  eps=$EPS  outdir=$OUT  outer=$OUTER"
echo "$NCASE cases in $NWAVE waves  (GL_CASES=${GL_CASES:-full})"
dep=""; n=0
while read -r wave; do
  [ -z "$wave" ] && continue
  n=$((n+1)); ids=()
  echo "--- wave $n ---"
  for C in $wave; do
    jid=$(sbatch --parsable $dep --job-name="w$n-$C" \
          --export=ALL,CASE="$C",NE="$NE",EPS="$EPS",OUT="$OUT",\
OUTER="$OUTER",WALL="$WALL" tools/case.slurm)
    ids+=("$jid")
    P=$(python tools/nesting.py --parents "$C")
    echo "  $C -> $jid  ${P:+(from: $P)}${P:-(root, cold)}"
  done
  dep="--dependency=afterany:$(IFS=:; echo "${ids[*]}")"
done < <(python tools/nesting.py --waves)
cat <<MSG

When the last wave finishes (ORDER MATTERS: verify_dense --update writes the
verified abscissa into every npz, which is what certify/collect then rank on;
check_band and certify_results derive their bands from each run's own record
and controller poles, so no hand-typed band can drift out of sync):
    python verify_dense.py --eps ${EPS:-0.05} --ne $NE --results $OUT --update  # 0 INVALID
    python winding_cert.py --eps ${EPS:-0.05} --ne $NE --results $OUT   # every case CERTIFIED (unconditional)
    python tools/check_band.py --ne $NE --results $OUT
    python certify_results.py  --results $OUT --ne $NE --band=poles --update
    python tools/collect.py    --results $OUT --ne $NE   # ALL NESTED OK
    python tools/check_optimality.py --results $OUT --ne $NE

If collect.py still reports a violation, it is NOT a scheduling problem: the
child was started from that parent and the optimiser is monotone.  Check
whether the parent's result file appeared after the child's job began.
MSG
