#!/usr/bin/env bash
# certification frequency: every round (1) vs every 4th round (4); K+B+C ne=100 from K+B
set -uo pipefail; cd "$(dirname "$0")/.."
for k in 4 1; do
  out=results_ce$k; mkdir -p $out/dump
  [ -f $out/K_B_C_s0_ne100.json ] && continue
  python run_case_epi.py --case K+B+C --start 0 --ne 100 --eps 0.05 --wall 1800 --outer 12 \
    --inner-budget iters --inner-maxit 10 --inner-maxit-adapt --outdir $out \
    --resume results_ne100_base/K_B_s0_ne100.npz --stabilise --select-cons on --ctrl-imcap off \
    --cont-anchor defer --sigma2 on --oracle reduced --ctrl-line eps --inner sqp --sqp-carry \
    --sqp-trust 0.01 --sqp-ctrl block --cert-sweep 61 --cert-every $k --oracle-inputs on --dump $out/dump > $out/run.log 2>&1
done
echo "=== cert-every A/B done"
