#!/usr/bin/env bash
# CONTROLLED solver comparison at ne=200 (TRACKER J): SQP vs pyGRANSO on the
# identical position-robust reduced oracle, flags, tolerances and rounds;
# WITH refinement for both: SQP in-solve refinement vs pyGRANSO band-max row (TRACKER J).
set -uo pipefail; cd "$(dirname "$0")/.."
COMMON="--case K+B+C --start 0 --ne 200 --eps 0.05 --wall 14400 --outer 6 --inner-budget iters \
 --inner-maxit 10 --inner-maxit-adapt --resume results/K_B_C_s0_ne200.npz --select-cons on \
 --ctrl-imcap off --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on --ctrl-line eps \
 --cert-sweep 61 --cert-rtol 1e-6 --alpha-argmax on"
for s in pygranso_band sqp_refine; do
  out=results_fair_$s; mkdir -p $out; inner=${s%%_*}
  [ -f $out/K_B_C_s0_ne200.json ] && continue
  extra="--pyg-bandmax on"; [ $inner = sqp ] && extra="--sqp-carry --sqp-trust 0.01 --sqp-ctrl block --sqp-row-res-tol 1e-3 --sqp-refine 1e-4"
  echo "=== launch $(date +%s) $s" >> $out/run.log
  python run_case_epi.py --eps 0.05 $COMMON --inner $inner $extra --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
done
echo "=== fair comparison (with refinement) done"
