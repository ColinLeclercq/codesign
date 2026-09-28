#!/bin/bash
#SBATCH --job-name=gl-stage1gate
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=05:59:00
#SBATCH --output=logs/stage1gate_%j.out
#SBATCH --error=logs/stage1gate_%j.err
#
# STAGE 1 GATE.  The three stage-1 cases at ne=200 under the stage-1
# configuration (STAGE1.md section 5), each started from its FROZEN parent in
# results/, then the full certification chain on the new designs and the
# harness row against the frozen table.
#
#     sbatch tools/stage1_gate.slurm
#
# Nothing in the code has changed default: every stage-1 flag is passed here
# explicitly, so the frozen suite is reproducible from the same tree.  Flip the
# defaults only after this job has been read.
#
# What "pass" means here: each case certifies (dense, winding, peak at rtol
# 1e-6), is feasible at eps=0.05, has ZERO gate refusals in its log, and its
# gamma is at or below the frozen value.  The last is the design claim; the
# first three are the correctness claims and matter more.
#
# Budget: iters/100, 8 outer rounds, first parent only (the gate is a
# confirmation, not the suite).  At ne=100 one round at maxit 100 is ~250
# oracle calls; with the 96-worker pool a case should take 30-60 min here.
set -eo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="
source ./env.sh
PYBIN=$(python -c 'import sys; print(sys.executable)')
case "$PYBIN" in *conda*codesign*) : ;; *) echo "*** wrong python: $PYBIN"; exit 1 ;; esac
echo "=== python: $PYBIN ($(python -V 2>&1)) ==="

# ---- inputs the frozen suite needs
for f in seeds_ne200.npz results/K_B_s0_ne200.npz results/B_C_s0_ne200.npz results/KGC_s0_ne200.npz; do
    [ -f "$f" ] || { echo "*** missing $f"; exit 1; }
done
if [ ! -f init_ne200.npz ]; then
    echo "=== init_ne200.npz missing; building it (seeds left alone) ==="
    python tools/make_init.py --ne 200 || exit 1
fi
python tools/check_invocations.py || exit 1

NE=200; EPS=0.05
CASES=${CASES:-K+B+C,KSP+B+C,KGC+B}
MAXIT=${MAXIT:-25}; OUTER=${OUTER:-8}; WALL=${WALL:-3600}
OUT=results_stage1_gate_${SLURM_JOB_ID}
# --cont_depth 8: the continuation gate refused maxit-100 steps at depth 4
# ("split depth exhausted", job 16139009) -- 256 sub-steps instead of 16.
STAGE1="--select-cons on --ctrl-imcap off --cont-anchor defer --cont_depth 8 --inner-maxit $MAXIT"
echo "=== ne=$NE eps=$EPS cases=$CASES maxit=$MAXIT outer=$OUTER out=$OUT ==="
echo "=== stage-1 flags: $STAGE1 ==="

# ---- the three cases, from the frozen parents
# EVERY parent, as the suite does -- job 16139009 ran only the first
# (alphabetically B+C, controller-free) and that is the one the borrowed
# controller start went wrong on.  Each attempt is 5-11 min on 96 cores.
python tools/lattice.py --ne "$NE" --eps "$EPS" --out "$OUT" --parent-dir results \
    --targets-only --closure "$CASES" \
    --inner-budget iters --inner-maxit "$MAXIT" --outer "$OUTER" --wall "$WALL" \
    --extra $STAGE1

# ---- refusals, by kind.  The stage-1 claim is zero COMPENSATOR refusals;
# on-the-line and continuation refusals are reported separately (both were
# seen in job 16139009 and are addressed: --sel-margin, --cont_depth).
echo; echo "=== refusals per attempt: compensator / on-the-line / continuation ==="
for f in "$OUT"/logs/*.log; do
  printf "  %-44s %2s %2s %2s   start %s\n" "$(basename "$f")" \
    "$(grep -c 'REJECTS: controller pole' "$f" || true)" \
    "$(grep -c 'on Re=' "$f" || true)" \
    "$(grep -c 'continuation REJECTS' "$f" || true)" \
    "$(grep -m1 -o 'start gamma=[0-9.]*' "$f")"
done

# ---- certification of the new designs, at the suite's eps
echo; echo "=== certification ==="
python verify_dense.py --eps "$EPS" --ne "$NE" --results "$OUT"
python winding_cert.py --eps "$EPS" --ne "$NE" --results "$OUT"
python certify_results.py --ne "$NE" --results "$OUT" --band=poles --rtol 1e-6

# ---- harness row against the frozen table (feasibility + eps checks inside)
echo; echo "=== harness rows vs frozen ==="
python harness.py --mode replay --cases "$CASES" --ne "$NE" --eps "$EPS" \
    --results "$OUT" --workdir "harness_out/stage1gate_${SLURM_JOB_ID}" \
    --json "harness_out/stage1_gate_${SLURM_JOB_ID}.json" || true

echo; echo "=== frozen vs stage-1 ==="
python - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
for c in ("K+B+C", "KSP+B+C", "KGC+B"):
    t = c.replace("+", "_") + "_s0_ne200.json"
    f0, f1 = os.path.join("results", t), os.path.join(out, t)
    g0 = json.load(open(f0))["gamma"] if os.path.exists(f0) else None
    if os.path.exists(f1):
        r = json.load(open(f1)); g1, a1 = r["gamma"], r.get("alpha_dense", r.get("alpha"))
        print(f"  {c:8s} frozen {g0:9.4f}  stage-1 {g1:9.4f}  ({100*(g1-g0)/g0:+.1f} %)  "
              f"alpha {a1:+.4f}  calls {r['calls']}  codes {r.get('termination_codes')}")
    else:
        print(f"  {c:8s} frozen {g0:9.4f}  stage-1 NO RECORD")
PY
echo; echo "=== nesting among the stage-1 designs (child <= parent) ==="
python - "$OUT" <<'PYNEST'
import json, os, subprocess, sys
out = sys.argv[1]
def g(c):
    for d in (out, "results"):
        f = os.path.join(d, c.replace("+", "_") + "_s0_ne200.json")
        if os.path.exists(f):
            return json.load(open(f))["gamma"], ("stage-1" if d == out else "frozen")
    return None, None
for c in ("K+B+C", "KSP+B+C", "KGC+B"):
    gc, _ = g(c)
    for p in subprocess.run([sys.executable, "tools/nesting.py", "--parents", c],
                            capture_output=True, text=True).stdout.split():
        gp, src = g(p)
        if gc is None or gp is None: continue
        # the peak certificate refines stored gammas by up to ~1e-5 relative
        # (job 16205236: KSP+B+C stored 7.9515, re-certified 7.9520 when
        # transplanted into K+B+C), so the check must not be stricter than
        # the certificate that produced the numbers
        ok = gc <= gp * (1 + 2e-5)
        print(f"  {c:8s} {gc:9.4f}  <=  {p:8s} {gp:9.4f} ({src:7s})  {'OK' if ok else '*** VIOLATED'}")
PYNEST
echo "=== stage-1 gate complete: $(date -Is) ==="
