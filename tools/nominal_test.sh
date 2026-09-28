#!/usr/bin/env bash
# Working-set fixes on the nominal loop (TRACKER K): the fair SQP arm's settings
# (no refinement) + all near-maximal peaks enrolled at their tops + following rows.
set -uo pipefail; cd "$(dirname "$0")/.."
out=results_nominal_v1; mkdir -p $out
[ -f $out/K_B_C_s0_ne200.json ] && exit 0
echo "=== launch $(date +%s)" >> $out/run.log
python run_case_epi.py --case K+B+C --start 0 --ne 200 --eps 0.05 --wall 14400 --outer 6 --inner-budget iters \
  --inner-maxit 10 --inner-maxit-adapt --resume results/K_B_C_s0_ne200.npz --select-cons on \
  --ctrl-imcap off --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on --ctrl-line eps \
  --cert-sweep 61 --cert-rtol 1e-6 --alpha-argmax on --inner sqp --sqp-carry --sqp-trust 0.01 \
  --sqp-ctrl block --sqp-row-res-tol 1e-3 --enroll-rel 0.05 --peak-follow on \
  --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
