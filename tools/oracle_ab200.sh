#!/usr/bin/env bash
# Stage 5 at ne=200: the terminal K+B+C polish, full vs reduced oracle, 4 rounds, stage-1b flags, dumps.
set -uo pipefail; cd "$(dirname "$0")/.."
S1="--select-cons on --ctrl-imcap off --cont-anchor defer --sigma2 on --inner-maxit 25"
for orc in reduced full; do
  out=results_ne200_oracle_$orc; mkdir -p "$out/logs"
  [ -f "$out/K_B_C_s0_ne200.json" ] && continue
  echo "== $orc $(date +%H:%M)"
  python run_case_epi.py --case K+B+C --start 0 --ne 200 --eps 0.05 --wall 7200 --outer 4 \
    --inner-budget iters --outdir "$out" --resume results/K_B_C_s0_ne200.npz $S1 --oracle $orc \
    --oracle-band-frac 0.05 --oracle-sweep-tol 1e-5 --dump "$out/dump" > "$out/logs/K_B_C.log" 2>&1
  grep -o '"gamma": [0-9.]*' "$out/logs/K_B_C.log" | tail -1
done
echo "=== ne200 oracle A/B done"
