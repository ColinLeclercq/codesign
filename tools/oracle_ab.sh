#!/usr/bin/env bash
# Stage 5 phase 1: full vs reduced oracle, same flags, same budget, with dumps.
set -uo pipefail; cd "$(dirname "$0")/.."
S1="--select-cons on --ctrl-imcap off --cont-anchor defer --sigma2 on"
for orc in full reduced; do
  out=results_ne100_oracle_$orc; mkdir -p "$out/logs"
  [ -f "$out/K_B_C_s0_ne100.json" ] && continue
  echo "== $orc $(date +%H:%M)"
  python run_case_epi.py --case K+B+C --start 0 --ne 100 --eps 0.05 --wall 600 --outer 6 \
    --inner-budget iters --inner-maxit 20 --outdir "$out" --resume results_ne100_base/K_B_s0_ne100.npz \
    $S1 --oracle $orc --oracle-band-frac 0.02 --oracle-sweep-tol 1e-6 --oracle-sweep-anchors 4 --dump "$out/dump" > "$out/logs/K_B_C.log" 2>&1
  grep -o '"gamma": [0-9.]*' "$out/logs/K_B_C.log" | tail -1
done
echo "=== oracle A/B done"
