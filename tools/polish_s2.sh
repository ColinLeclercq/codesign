#!/usr/bin/env bash
# Item 3 measurement: the terminal polish of a frozen design, sigma_2 off vs on.
set -uo pipefail; cd "$(dirname "$0")/.."
CASE=${1:-K+B+C}; NE=${2:-200}; TAG=${CASE//+/_}_s0_ne$NE
S1="--select-cons on --ctrl-imcap off --cont-anchor defer --inner-maxit 25 --cont_depth 8"
for arm in off on; do
  out=results_polish_${arm}_ne$NE; mkdir -p "$out/logs"
  [ -f "$out/$TAG.json" ] && { echo "== $arm done"; continue; }
  echo "== polish $CASE sigma2=$arm  $(date +%H:%M)"
  python run_case_epi.py --case "$CASE" --start 0 --ne "$NE" --eps 0.05 --wall 3600 --outer 4 \
    --inner-budget iters --outdir "$out" --resume "results/$TAG.npz" $S1 --sigma2 $arm \
    > "$out/logs/$TAG.log" 2>&1
  grep -o '"gamma": [0-9.]*' "$out/logs/$TAG.log" | tail -1
done
echo "=== polish done"
