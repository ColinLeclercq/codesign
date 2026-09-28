#!/usr/bin/env bash
# item 2, fair version: SI, KSP+B+C from B+C, maxit 100, FOUR rounds, per tolerance.
set -uo pipefail; cd "$(dirname "$0")/.."
BASE=results_ne100_base; SI="--select-cons on --ctrl-imcap off"
for tol in 1e-10 1e-6 1e-4; do
  out=results_ne100_tol100x4_$tol; tag=KSP_B_C_s0_ne100; mkdir -p "$out/logs"
  [ -f "$out/$tag.json" ] && continue
  echo "== maxit100 x4 tol=$tol KSP+B+C  $(date +%H:%M)"
  python run_case_epi.py --case KSP+B+C --start 0 --ne 100 --eps 0.05 --wall 2400 --outer 4 \
    --inner-budget iters --inner-maxit 100 --outdir "$out" --resume "$BASE/B_C_s0_ne100.npz" \
    $SI --opt-tol $tol > "$out/logs/$tag.log" 2>&1
  grep -o '"gamma": [0-9.]*' "$out/logs/$tag.log" | tail -1
done
echo "=== tol4 done"
