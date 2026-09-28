#!/usr/bin/env bash
# Stage-1 A/B at ne=100, sequential, from FIXED baseline parents.
# Baseline rows (results_ne100_base) are the frozen configuration.
#   S  : --select-cons on   (KSP+B+C only: ~50 min/case, cap unsatisfiable, not a candidate)
#   SI : --select-cons on --ctrl-imcap off
# Same eps/budget/outer as the baseline so the columns compare.
set -uo pipefail
cd "$(dirname "$0")/.."
BASE=results_ne100_base
COMMON="--ne 100 --wall 600 --outer 4 --inner-budget iters --inner-maxit 20"
run() {  # cfg case parent extra...
  local cfg=$1 case=$2 parent=$3; shift 3
  local out=results_ne100_$cfg tag=${case//+/_}_s0_ne100
  mkdir -p "$out/logs"
  [ -f "$out/$tag.json" ] && { echo "== $cfg $case done"; return; }
  echo "== $cfg $case from $parent  $(date +%H:%M)"
  if [ "$parent" = "-" ]; then src=(--cold --seed hinf_3); else src=(--resume "$BASE/${parent//+/_}_s0_ne100.npz"); fi
  python run_case_epi.py --case "$case" --start 0 --eps 0.05 $COMMON --outdir "$out" "${src[@]}" "$@" \
     > "$out/logs/$tag.log" 2>&1
  grep -o '"gamma": [0-9.]*' "$out/logs/$tag.log" | tail -1
}
S="--select-cons on"; SI="--select-cons on --ctrl-imcap off"
# N: everything off -- the cost of the compensator-stability constraint itself
N="--select-cons on --ctrl-imcap off --ctrl-stab off"
run SI KSP+B+C B+C $SI;  run S KSP+B+C B+C $S;  run N KSP+B+C B+C $N
run SI KSP+C   C   $SI;  run N KSP+C   C   $N
run SI K+B+C   K+B $SI;  run N K+B+C   K+B $N
run SI KSP     -   $SI;  run N KSP     -   $N
echo "=== A/B done $(date +%H:%M)"
