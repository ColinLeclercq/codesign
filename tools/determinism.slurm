#!/bin/bash
#SBATCH --job-name=gl-determ
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=02:59:00
#SBATCH --output=logs/determ_%j.out
#SBATCH --error=logs/determ_%j.err
#
#   sbatch tools/determinism.slurm
#   CASE=K+B+C REPEAT=3 sbatch tools/determinism.slurm
#
# Answers the two questions stage 1 cannot start without, both of which are
# about the WALL-CLOCK inner budget:
#
#   A. How often does the clock actually truncate an inner solve at PRODUCTION
#      settings?  In the sandbox, at a stressed 10 s inner limit, pyGRANSO
#      termination code 5 (clock limit) fired on 2 of 3 outer rounds and cost a
#      relative spread of 1.1e-08 in gamma.  Production is not that setting:
#      case.slurm uses WALL=18000 split over 8 seeds, OUTER=40, so
#      per_outer = (2250 - elapsed - 45)/40 ~ 55 s.  Whether code 5 fires at 55 s
#      has never been measured -- the frozen records predate termination_codes --
#      and it sizes the variance every stage-1 measurement would carry.
#
#   B. Is --inner-budget iters actually deterministic at ne = 200 with the pool
#      on?  It is at ne = 100 on one core (three identical runs, bit-identical
#      gamma).  The pool introduces a second candidate source of nondeterminism
#      -- worker scheduling -- which only a real run can rule out.
#
# --exclusive is not optional here: A is a measurement OF machine load, so a
# shared node would answer a different question.
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs cleanup_evidence/determinism_ne200
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

CASE=${CASE:-K+B+C}
NE=${NE:-200}
# The stability margin.  MUST be explicit: run_case_epi falls back to
# packer.EPS_STAB_DEFAULT = 1e-3 if --eps is absent, which is a 50x weaker
# constraint than the frozen suite's 0.05 and silently a different problem.
# Job 16016215 ran the whole stage-1 baseline that way.
EPS=${EPS:-0.05}
REPEAT=${REPEAT:-3}
# production-equivalent single-attempt budget: case.slurm's WALL=18000 over 8
# seeds.  Do NOT raise this to "be safe" -- the point is to reproduce the clock
# pressure the frozen campaign ran under.
WALL=${WALL:-2250}
OUTER=${OUTER:-40}
MAXIT=${MAXIT:-200}
TAG=${CASE//+/_}

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

echo "=== $(hostname): cpus=${SLURM_CPUS_PER_TASK:-unset} case=$CASE ne=$NE ==="
echo "=== wall=$WALL outer=$OUTER  => per_outer ~ $(( (WALL - 45) / OUTER )) s ==="

# ---------------------------------------------------------------- A + B
for MODE in clock iters; do
  for K in $(seq 1 "$REPEAT"); do
    OD="cleanup_evidence/determinism_ne200/${MODE}_${K}"
    rm -rf "$OD"; mkdir -p "$OD"
    if [ "$MODE" = "clock" ]; then
      EXTRA=""
    else
      EXTRA="--inner-budget iters --inner-maxit $MAXIT"
    fi
    echo ""
    echo "=== $MODE repeat $K/$REPEAT ==="
    # shellcheck disable=SC2086
    python run_case_epi.py --case "$CASE" --start 0 --ne "$NE" \
           --wall "$WALL" --outer "$OUTER" --eps "$EPS" --cold \
           --outdir "$OD" $EXTRA
  done
done

# ---------------------------------------------------------------- report
python - <<'PY'
import glob, json, os, statistics
base = "cleanup_evidence/determinism_ne200"
out = {}
for mode in ("clock", "iters"):
    gs, rows = [], []
    for d in sorted(glob.glob(os.path.join(base, f"{mode}_*"))):
        f = glob.glob(os.path.join(d, "*.json"))
        if not f:
            continue
        r = json.load(open(f[0]))
        h = r.get("history") or []
        codes = [x.get("termination_code") for x in h]
        gs.append(r["gamma"])
        rows.append((os.path.basename(d), r["gamma"], r["outer"], r["calls"],
                     codes, r.get("clock_guard_bound")))
    print(f"\n=== --inner-budget {mode} ===")
    for name, g, o, c, codes, guard in rows:
        n5 = sum(1 for x in codes if x == 5)
        print(f"  {name:10s} gamma={g!r} outer={o} calls={c}")
        print(f"             codes={codes}  code5(clock)={n5}/{len(codes)}"
              + (f"  GUARD BOUND" if guard else ""))
    if len(gs) > 1:
        spread = (max(gs) - min(gs)) / min(gs)
        allcodes = [x for _, _, _, _, cs, _ in rows for x in cs]
        n5 = sum(1 for x in allcodes if x == 5)
        print(f"  identical across repeats : {len(set(gs)) == 1}")
        print(f"  relative spread in gamma : {spread:.3e}")
        print(f"  clock truncations        : {n5}/{len(allcodes)} outer rounds"
              f" ({100.0 * n5 / max(len(allcodes), 1):.0f}%)")
        out[mode] = dict(gammas=gs, identical=len(set(gs)) == 1,
                         rel_spread=spread, code5=n5, rounds=len(allcodes))
json.dump(out, open(os.path.join(base, "summary.json"), "w"), indent=1)
print(f"\nsummary -> {base}/summary.json")
print("\nREAD THIS AS: if clock shows code5 = 0/N, the wall budget never bound "
      "at production settings and the frozen numbers carry no clock variance. "
      "If it shows a nonzero count with a nonzero spread, every stage-1 "
      "measurement needs --inner-budget iters to be trustworthy.")
PY

echo "=== determinism job complete: $(date -Is) ==="
