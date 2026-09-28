#!/usr/bin/env bash
# Controlled A/B at ne=100 (STAGE5 18): row screening off vs on, K+B+C from the
# transferred start, 8 rounds, current code; measure the per-round gap between
# the SQP's t and the full certificate.
set -uo pipefail; cd "$(dirname "$0")/.."
LOOP="--ne 100 --wall 14400 --outer 8 --inner-budget iters --inner-maxit 10 --inner-maxit-adapt \
 --select-cons on --ctrl-imcap off --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on \
 --ctrl-line eps --cert-sweep 61 --cert-rtol 1e-6 --alpha-argmax on --enroll-rel 0.05 --peak-follow on \
 --inner sqp --sqp-carry --sqp-trust 0.01 --sqp-ctrl block --sqp-row-res-tol 1e-3 --sqp-refine 1e-4"
for scr in 0 0.05; do
  out=results_ab_scr$scr; mkdir -p $out
  [ -f $out/K_B_C_s0_ne100.json ] && continue
  echo "=== launch $(date +%s) row-screen $scr" >> $out/run.log
  python run_case_epi.py --eps 0.05 --case K+B+C --start 0 $LOOP --row-screen $scr \
    --resume results_ne100/K_B_C_s0_ne100.npz --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
done
echo "=== A/B done"
