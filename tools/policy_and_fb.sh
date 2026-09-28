#!/usr/bin/env bash
# (1) note items 4-5 (A-N satellites + retirement, |Omega| <= 50) on the
#     corrected SQP loop, vs results_corr_sqp (same loop, no policy);
# (3) K+FB+FC, fields on 80 Dirichlet-Laplacian modes, corrected SQP loop,
#     from the frozen design projected on the basis (3.927660; frozen 3.925456).
set -uo pipefail; cd "$(dirname "$0")/.."
LOOP="--ne 200 --eps 0.05 --wall 14400 --outer 8 --inner-budget iters --inner-maxit 10 --inner-maxit-adapt \
 --select-cons on --ctrl-imcap off --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on \
 --ctrl-line eps --cert-sweep 61 --cert-rtol 1e-6 --alpha-argmax on --enroll-rel 0.05 --peak-follow on --converge \
 --inner sqp --sqp-carry --sqp-trust 0.01 --sqp-ctrl block --sqp-row-res-tol 1e-3"
out=results_policy_an; mkdir -p $out
if [ ! -f $out/K_B_C_s0_ne200.json ]; then
  echo "=== launch $(date +%s)" >> $out/run.log
  python run_case_epi.py --eps 0.05 --case K+B+C --start 0 $LOOP --resume results/K_B_C_s0_ne200.npz \
    --omega-policy an --omega-cap 50 --omega-skeleton 32 --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
fi
out=results_fb80; mkdir -p $out
if [ ! -f $out/K_FB_FC_s0_ne200.json ]; then
  echo "=== launch $(date +%s)" >> $out/run.log
  python run_case_epi.py --eps 0.05 --case K+FB+FC --start 0 $LOOP --resume results/K_FB_FC_s0_ne200.npz \
    --field-basis laplace --field-n 80 --field-lo -38.9 --field-hi 39.2 --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
fi
echo "=== policy + FB done"
