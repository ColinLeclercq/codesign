#!/usr/bin/env bash
# coarse-then-fine certification: full sweep 61 vs 241 points; K+B+C ne=100
# from K+B, the stabilisation phase bringing the compensator inside -eps.
set -uo pipefail; cd "$(dirname "$0")/.."
for n in 61 241; do
  out=results_cs$n; mkdir -p $out/dump
  [ -f $out/K_B_C_s0_ne100.json ] && continue
  python run_case_epi.py --case K+B+C --start 0 --ne 100 --eps 0.05 --wall 1200 --outer 6 \
    --inner-budget iters --inner-maxit 10 --inner-maxit-adapt --outdir $out \
    --resume results_ne100_base/K_B_s0_ne100.npz --stabilise --select-cons on --ctrl-imcap off \
    --cont-anchor defer --sigma2 on --oracle reduced --ctrl-line eps --inner sqp --sqp-carry \
    --sqp-trust 0.01 --sqp-ctrl block --cert-sweep $n --dump $out/dump > $out/run.log 2>&1
done
echo "=== cert-sweep A/B done"
