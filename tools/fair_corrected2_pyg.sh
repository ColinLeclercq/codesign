#!/usr/bin/env bash
# SQP vs pyGRANSO on the CORRECTED loop (TRACKER K.3): identical start, reduced
# model, flags, tolerances; all near-maximal peaks enrolled at their tops;
# following rows including the sigma_2 rows; no in-solve refinement; run to
# convergence (compare gamma, KKT residual, full-model factorisations).
set -uo pipefail; cd "$(dirname "$0")/.."
COMMON="--case K+B+C --start 0 --ne 200 --eps 0.05 --wall 14400 --outer 8 --inner-budget iters \
 --inner-maxit 10 --inner-maxit-adapt --resume results/K_B_C_s0_ne200.npz --select-cons on \
 --ctrl-imcap off --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on --ctrl-line eps \
 --cert-sweep 61 --cert-rtol 1e-6 --alpha-argmax on --enroll-rel 0.05 --peak-follow on --converge"
for s in pygranso; do
  out=results_corr_$s; mkdir -p $out
  [ -f $out/K_B_C_s0_ne200.json ] && continue
  extra=""; [ $s = sqp ] && extra="--sqp-carry --sqp-trust 0.01 --sqp-ctrl block --sqp-row-res-tol 1e-3"
  echo "=== launch $(date +%s) $s" >> $out/run.log
  python run_case_epi.py --eps 0.05 $COMMON --inner $s $extra --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
done
echo "=== corrected comparison done"
