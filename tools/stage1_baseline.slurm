#!/bin/bash
#SBATCH --job-name=gl-stage1base
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=05:59:00
#SBATCH --output=logs/stage1base_%j.out
#SBATCH --error=logs/stage1base_%j.err
#
#   sbatch tools/stage1_baseline.slurm
#   CASES=K+B+C,KSP+B+C sbatch tools/stage1_baseline.slurm
#
# The stage-1 baseline: run the three stage-1 cases through the harness under a
# REPRODUCIBLE inner budget, with the per-outer-round dump on, and draw them.
# Everything stage 1 measures gets compared against what this produces, so it
# must be run once, on an exclusive node, before any solver setting is touched.
#
# The three cases (harness.CASE_SETS["stage1"]):
#   K+B+C     balance 1.47e+04, TERMINATED  -- the known stall
#   KSP+B+C   balance 1.73e+03, CONVERGED   -- same pathology, arrives labelled
#                                              healthy, so a fix is falsifiable
#   KGC+B     balance 2.74e-01, TERMINATED  -- the CONTROL: a fix must NOT move it
#
# Time: frozen wall times are 143 / 237 / 1439 s at 96 workers.  Under
# --inner-budget iters the inner runs to its own stopping criterion rather than
# being cut off, which cost about 2x the oracle calls at ne = 100, so budget
# ~2-3x the frozen figures.  Six hours is generous; KGC+B with 28 outer rounds
# is the one that could surprise.
#
# NOTE ON COMPARABILITY.  These rows are NOT directly comparable to the section
# 3 table's `oracle calls` column: that column was produced under a wall-clock
# budget that truncated inner solves.  gamma is comparable (and the harness
# checks it at rtol); call counts are not.  Do not re-baseline all 27 this way
# -- the frozen suite stays the reference.
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs figs
source ./env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1

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

# SMOKE=1: exercise the whole pipeline end to end on ONE case and ONE outer
# round -- about five minutes instead of six hours.  It proves the plumbing
# (eps reaching the record, dumps, plots, certification, harness verdict, exit
# codes) and NOTHING about the optimisation.
#
#     SMOKE=1 sbatch --time=00:20:00 tools/stage1_baseline.slurm
#
# It does not replace tools/check_invocations.py, which catches the missing
# flag class of bug in under a second with no allocation at all.
# Accept it as a positional argument too: `sbatch tools/stage1_baseline.slurm smoke`.
# Job 16032466 was submitted as `SMOKE=1 sbatch ...` and arrived with smoke=0 --
# the variable did not propagate, for reasons not established.  A positional
# argument goes through sbatch's own argument list and cannot be dropped by an
# export policy.  --export=ALL,SMOKE=1 is the other reliable form.
[ "${1:-}" = "smoke" ] && SMOKE=1
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
    : "${CASES:=K+B+C}"; : "${OUTER:=1}"; : "${WALL:=155}"; : "${MAXIT:=25}"
    : "${OUTDIR:=results_smoke_${SLURM_JOB_ID:-local}}"
    : "${DUMPDIR:=dumps_smoke_${SLURM_JOB_ID:-local}}"
    echo "=== SMOKE MODE: one case, one outer round, plumbing only ==="
    echo "=== gammas from this run mean NOTHING; do not record them ==="
fi

CASES=${CASES:-stage1}
NE=${NE:-200}
# The stability margin.  MUST be explicit: run_case_epi falls back to
# packer.EPS_STAB_DEFAULT = 1e-3 if --eps is absent, which is a 50x weaker
# constraint than the frozen suite's 0.05 and silently a different problem.
# Job 16016215 ran the whole stage-1 baseline that way.
EPS=${EPS:-0.05}
WALL=${WALL:-2250}
OUTER=${OUTER:-40}
MAXIT=${MAXIT:-200}
VARIANT=${VARIANT:-frozen_iters}
# Job-specific by default.  run_case_epi writes <case>_ckpt.npz on every
# accepted iterate and at startup RESUMES from that checkpoint whenever its
# gamma beats the chosen start (run_case_epi.py ~1111).  So a results directory
# that already holds records is not a blank slate: job 16034591 reused
# results_stage1_baseline, found job 16016215's eps=0.001 checkpoints there,
# resumed KSP+B+C from the void solution and kept it, and exited KGC+B in 11 s.
# Two of three rows were the previous job's numbers.
OUTDIR=${OUTDIR:-results_stage1_baseline_${SLURM_JOB_ID:-local}}
DUMPDIR=${DUMPDIR:-dumps_stage1_baseline_${SLURM_JOB_ID:-local}}

# And refuse to run on top of someone else's checkpoints even when OUTDIR is
# given explicitly, unless the caller says RESUME=1 and means it.
if [ "${RESUME:-0}" != "1" ] && ls "$OUTDIR"/*_ckpt.npz >/dev/null 2>&1; then
    echo "*** $OUTDIR already contains checkpoints:"
    ls -la "$OUTDIR"/*_ckpt.npz
    echo "*** run_case_epi would resume from them, and a checkpoint written"
    echo "*** under different settings (eps, budget) silently becomes this"
    echo "*** run's answer.  Use a fresh OUTDIR, delete them, or set RESUME=1"
    echo "*** if resuming is what you want."
    exit 1
fi
# job-specific, so this can run beside tools/validate.slurm on another node
WORKDIR=${WORKDIR:-harness_out/stage1_${SLURM_JOB_ID:-local}}

# ------------------------------------------------------------ cold start
# init_ne$NE.npz holds the cold-start controller and the field positions
# xa/xs that run_case_epi starts every case from.  Neither it nor
# seeds_ne$NE.npz was in the frozen archive; the catalogue was supplied by
# hand afterwards.  Build the init if it is absent.
#
# Deliberately NOT `prepare.py`: that rewrites seeds_ne$NE.npz too, and every
# certified gamma was warm started from the catalogue that is sitting there
# now.  tools/make_init.py builds init alone and takes r from the existing
# catalogue.  (Verified bit-identical to prepare.py's init at ne=100.)
if [ ! -f "init_ne${NE}.npz" ]; then
    if [ ! -f "seeds_ne${NE}.npz" ]; then
        echo "*** neither init_ne${NE}.npz nor seeds_ne${NE}.npz is present."
        echo "*** The seed catalogue cannot be reconstructed safely here --"
        echo "*** rebuilding it would change what every case warm starts from."
        echo "*** Put seeds_ne${NE}.npz in $(pwd) and resubmit, or run"
        echo "***   python tools/prepare.py --ne ${NE}"
        echo "*** by hand if you accept a fresh catalogue."
        exit 1
    fi
    echo "=== init_ne${NE}.npz missing; building it (seeds left alone) ==="
    python tools/make_init.py --ne "$NE" || exit 1
fi

echo "=== $(hostname): cpus=${SLURM_CPUS_PER_TASK:-unset} ==="
echo "=== cases=$CASES ne=$NE eps=$EPS wall=$WALL outer=$OUTER maxit=$MAXIT smoke=$SMOKE ==="

# harness.py exits 1 when any row disagrees with the frozen suite.  Under
# `set -e` that would kill the job before the plots and, worse, before the
# CERTIFICATION below -- leaving uncertified designs on disk and no figure to
# read them by.  A disagreement is a result to look at, not a reason to discard
# the run, so capture the status and decide at the end.
HARNESS_RC=0
python harness.py --mode run \
       --cases "$CASES" --ne "$NE" \
       --wall "$WALL" --outer "$OUTER" --cold \
       --inner-budget iters --eps "$EPS" --inner-maxit "$MAXIT" \
       --variant "$VARIANT" \
       --outdir "$OUTDIR" --dumpdir "$DUMPDIR" \
       --workdir "$WORKDIR" \
       --json "harness_out/stage1_baseline_${SLURM_JOB_ID:-local}.json" \
  || HARNESS_RC=$?
echo "=== harness exit status: $HARNESS_RC "\
     "(nonzero = at least one row differs from the frozen suite; "\
     "read the table, do not assume the run is void) ==="

# ---- draw every dump the run produced -------------------------------------
# Cheap (matplotlib only) and the panels are how a stall is read: sigma_1 over
# Omega with t, the exchange set per round, the balance residual on a log axis,
# and the spectrum against -eps.
for D in "$DUMPDIR"/*/; do
  [ -d "$D" ] || continue
  TAG=$(basename "$D")
  echo "=== plotting $TAG ==="
  python plot_omega.py --dump "$D" --tag "$TAG" --outdir figs || \
    echo "    (plot failed for $TAG; the dump is still on disk)"
done

# ---- the certificates must still hold on the new designs ------------------
# A reproducible budget is not a licence to skip certification: these are new
# designs and nothing is certified until the stack says so.
echo ""
echo "=== certifying the baseline designs ==="
python verify_dense.py --eps "$EPS"    --ne "$NE" --results "$OUTDIR"
python winding_cert.py --eps "$EPS"    --ne "$NE" --results "$OUTDIR"
python certify_results.py --ne "$NE" --results "$OUTDIR" --band=poles --rtol 1e-6

# ------------------------------------------------------- smoke checklist
if [ "${SMOKE:-0}" = "1" ]; then
    echo
    echo "================ SMOKE CHECKLIST ================"
    J="harness_out/stage1_baseline_${SLURM_JOB_ID:-local}.json"
    SMOKE_RC=0
    python - "$J" "$EPS" <<'PYEOF' || SMOKE_RC=$?
import json, sys
rows = json.load(open(sys.argv[1])); eps = float(sys.argv[2])
ok = True
for r in rows:
    got = r.get("eps")
    good = got is not None and abs(float(got) - eps) < 1e-12
    ok &= good
    print(f"  eps in record      : {got}  "
          f"{'OK' if good else 'WRONG -- the flag is not reaching run_case_epi'}")
    m = r.get("alpha_margin")
    print(f"  alpha              : {r.get('alpha_dense')}   margin {m}")
    print(f"  feasibility check  : "
          f"{'ran' if m is not None else 'DID NOT RUN -- alpha or eps missing'}")
    ok &= m is not None
    inp = r.get("inputs") or {}
    print(f"  inputs fingerprint : init={inp.get('init')} seeds={inp.get('seeds')}")
    ok &= bool(inp.get("init"))
    print(f"  problems           : {r.get('problems')}")
print("  ---")
print("  PLUMBING OK" if ok else "  *** PLUMBING BROKEN -- fix before the real run")
sys.exit(0 if ok else 1)
PYEOF
    ls -la figs/ 2>/dev/null | tail -3
    echo "  (a nonempty problems list is EXPECTED: one outer round will not"
    echo "   reach the frozen gamma.  What matters is that eps is right, the"
    echo "   feasibility check ran, and the inputs are fingerprinted.)"
    echo "================================================"
    [ "$SMOKE_RC" -eq 0 ] || exit 1
fi

echo "=== stage-1 baseline complete: $(date -Is) ==="
if [ "$HARNESS_RC" -ne 0 ]; then
  echo "=== NOTE: the harness flagged at least one row.  gamma should match"
  echo "===       the frozen values at rtol; oracle-call counts should NOT,"
  echo "===       because the frozen column was produced under a truncating"
  echo "===       wall-clock budget.  Check which column actually moved. ==="
fi
echo "=== designs in $OUTDIR, dumps in $DUMPDIR, figures in figs/ ==="
