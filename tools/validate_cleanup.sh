#!/bin/bash
# validate_cleanup.sh -- prove the section 4 cleanup changed no number.
#
#   bash tools/validate_cleanup.sh              # ne=200, results/
#   NE=200 OUT=results bash tools/validate_cleanup.sh
#
# Run from anywhere; it cds to the repository root itself.  Nothing here uses
# --update, so the frozen records are never written to: the whole point is to
# re-derive their contents and compare.
#
# What it proves, in order:
#   0. the environment gate                       (check_env.py)
#   1. kernels, gradients, certification, and a
#      tiny end-to-end run of each case group     (selftest.py)
#   2. the controller order is inferred as 8 with
#      no --r flag anywhere                       (ctrl_order.py)
#   3. every archived design is stable, dense     (verify_dense.py)
#   4. every archived design is UNCONDITIONALLY
#      stable, count 0                            (winding_cert.py)
#   5. all 27 gammas re-certify at rtol 1e-6      (certify_results.py)
#   6. the records still match the handoff's
#      section 3 table                            (tools/check_frozen_table.py)
#   7. the stage-0 harness replays all 27 with no
#      disagreement, and the nesting lattice and
#      collected summary still build              (harness.py, nesting.py,
#                                                  collect.py)
#   8. the frozen records are byte-identical      (git status / checksums)
#
# selftest.py dominates the wall time on a single core (~10 min here; the
# handoff's ~6 min assumes several).  Steps 3-7 together take under 3 minutes
# at GL scale.
set -Eeo pipefail   # -E: the ERR trap must be inherited by the brace group,
                   # which runs in a SUBSHELL because it is piped into tee

# Name the failing command.  Everything below runs inside a brace group piped
# into tee, so `set -e` on its own exits 1 with a truncated log and no clue.
CURRENT_STEP="startup"
# The brace group at the end is piped into tee, so it runs in a SUBSHELL: a
# variable set inside it cannot be read outside, and the ERR trap fires twice --
# once inside, with the right step name, and once outside on the pipeline, with
# none.  A marker file carries the one fact the outer shell needs: that the
# failure has already been reported.
TRAPMARK="$(mktemp -t glvalidate.XXXXXX)"
trap 'rm -f "$TRAPMARK"' EXIT
trap 'rc=$?; if [ -s "$TRAPMARK" ]; then exit $rc; fi; \
      echo "reported" > "$TRAPMARK"; \
      echo ""; echo "*** validate_cleanup.sh FAILED"; \
      echo "***   step   : $CURRENT_STEP"; \
      echo "***   line   : $LINENO"; \
      echo "***   command: $BASH_COMMAND"; \
      echo "***   status : $rc"; \
      echo "*** (steps before this one succeeded; the log above is complete "; \
      echo "***  up to the failure)"; exit $rc' ERR

cd "$(dirname "$0")/.."
NE=${NE:-200}
OUT=${OUT:-results}
# The suite's stability margin.  verify_dense defaults to 1e-3 -- "is it
# stable" rather than "does it meet the spec" -- so steps 3 and 4 were checking
# something weaker than the design constraint until this was made explicit.
EPS=${EPS:-0.05}
LOG=${LOG:-cleanup_evidence/validate_$(date +%Y%m%d_%H%M%S).log}
# The harness writes its per-case diagnostics into a workdir.  Two jobs sharing
# one workdir overwrite each other's check_optimality.json and kkt_*.json, so
# the default is job-specific: this script and tools/stage1_baseline.slurm can
# then run on different nodes at the same time without interfering.
WORKDIR=${WORKDIR:-harness_out/validate_${SLURM_JOB_ID:-$$}}
mkdir -p "$(dirname "$LOG")"

# checksum the frozen records BEFORE anything runs, so step 8 is meaningful
SUM_BEFORE=$(find "$OUT" -type f \( -name '*.json' -o -name '*.npz' \) \
             | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)

run () {  # run <label> <cmd...>
  CURRENT_STEP="$1"
  echo ""
  echo "=================================================================="
  echo "== $1"
  echo "=================================================================="
  shift
  "$@"
}

{
echo "validate_cleanup.sh  ne=$NE  results=$OUT  $(date -Is)"
echo "host $(hostname)  python $(python -V 2>&1)"

# Step 0 must FAIL LOUDLY if the gate could not run.  It used to be
#   python check_env.py || echo "NOTE: ... not needed for GL"
# which masked the difference between "PETSc is absent, fine for GL" and "the
# gate aborted and reported nothing".  On 2026-09-11 the second happened -- the
# PETSc probe called MPI_Init on a login node, Open MPI called MPI_Abort, and
# the run continued as though step 0 had passed.  check_env.py now isolates
# each probe in a subprocess and exits 0 iff numpy, PyTorch and pyGRANSO pass,
# so its exit status is meaningful and is no longer discarded.
echo ""
echo "=================================================================="
echo "== 0. environment gate"
echo "=================================================================="
# PMIX supplies Open MPI's transport key only inside a job STEP, so the PETSc
# probe needs srun -- but ONLY that probe, and only when an allocation exists.
# Wrapping the whole validation in srun (an earlier version did) risks losing
# the entire run to an srun-side problem before anything is printed.
CURRENT_STEP="0. environment gate"
# Do NOT run this under srun by default.
#
# The PETSc probe imports petsc4py, which calls MPI_Init.  On this site two MPI
# runtimes are in play at once: env.sh loads python/3.8.14-intel2220, which
# pulls Intel MPI (impi/.22.2.0), while the conda env's petsc4py is built
# against Open MPI.  MPI_Init therefore fails, and Open MPI's
# MPI_ERRORS_ARE_FATAL calls MPI_Abort.
#
# Outside a job step that costs one Python process and nothing else -- which is
# what happened on 2026-09-11 16:00, where the gate aborted and the remaining
# eight steps all passed.  INSIDE an srun step, MPI_Abort asks the resource
# manager to kill the step, taking this script with it: job 15937568 printed the
# step-0 header and died, with no output from check_env.py at all.
#
# check_env.py now isolates each probe in its own process, so the abort costs
# one row either way.  srun is therefore not needed, and being in a step is
# actively harmful.  Set GATE_SRUN=1 to try it anyway (e.g. once the MPI stacks
# are reconciled and you want the PETSc/SLEPc/dolfinx rows to pass); the result
# is checked and falls back rather than being trusted.
if [ "${GATE_SRUN:-0}" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ] \
   && command -v srun >/dev/null 2>&1; then
  echo "(GATE_SRUN=1: running the gate inside a job step)"
  ENVOUT="$(srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
            python check_env.py 2>&1)" && ENVRC=0 || ENVRC=$?
  if ! echo "$ENVOUT" | grep -q "Co-design toolchain check"; then
    echo "(the gate produced no report inside the step -- almost certainly an"
    echo " MPI_Abort killing the step; retrying outside it)"
    echo "$ENVOUT" | tail -3
    ENVOUT="$(python check_env.py 2>&1)" && ENVRC=0 || ENVRC=$?
  fi
else
  ENVOUT="$(python check_env.py 2>&1)" && ENVRC=0 || ENVRC=$?
fi
echo "$ENVOUT"
if [ $ENVRC -ne 0 ]; then
  echo "*** environment gate FAILED: GL needs numpy, PyTorch and pyGRANSO ***"
  exit 1
fi
for want in "numpy" "PyTorch" "pyGRANSO"; do
  echo "$ENVOUT" | grep -q "^\[PASS\].*$want" || {
    echo "*** environment gate did not report a PASS for $want ***"
    echo "*** check_env.py isolates each probe, so this means a REQUIRED ***"
    echo "*** library is genuinely missing, not that a probe aborted.    ***"
    exit 1; }
done
echo "PETSc/SLEPc/dolfinx are not needed for GL; they are stage-7 (cavity)."

run "1. selftest (the first gate)"   python selftest.py

run "2. controller order, inferred"  python -c \
  "from ctrl_order import infer_r; r=infer_r($NE,'$OUT'); assert r==8, r; print('r =',r,'OK')"

run "3. dense cross-check"           python verify_dense.py --eps "$EPS" --ne "$NE" --results "$OUT"
run "4. winding certificate"         python winding_cert.py --eps "$EPS" --ne "$NE" --results "$OUT"
run "5. peak certification 1e-6"     python certify_results.py --results "$OUT" \
                                       --ne "$NE" --band=poles --rtol 1e-6
run "6. handoff section 3 table"     python tools/check_frozen_table.py "$OUT"
run "7. stage-0 harness, 27-case replay"  python harness.py --mode replay \
                                       --cases all --ne "$NE" --results "$OUT" \
                                       --workdir "$WORKDIR"
run "7a. nesting lattice"            python tools/nesting.py --waves
run "7b. collected summary"          python tools/collect.py --results "$OUT" --ne "$NE"

echo ""
echo "=================================================================="
echo "== 8. frozen records untouched"
echo "=================================================================="
SUM_AFTER=$(find "$OUT" -type f \( -name '*.json' -o -name '*.npz' \) \
            | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
echo "before $SUM_BEFORE"
echo "after  $SUM_AFTER"
if [ "$SUM_BEFORE" = "$SUM_AFTER" ]; then
  echo "RECORDS UNCHANGED"
else
  echo "*** RECORDS MODIFIED -- something ran with --update; investigate ***"
  exit 1
fi

echo ""
echo "validation complete: $(date -Is)"
} 2>&1 | tee "$LOG"

echo ""
echo "log written to $LOG"
