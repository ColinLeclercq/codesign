#!/usr/bin/env bash
# Debugging at ne=100 (user, 2026-09-24): the corrected nominal loop, run to
# convergence, on K+B+C and K+FB+FC (80 Laplacian modes per field), from the
# frozen ne=200 designs transferred to ne=100 (tools/transfer_ne.py).
set -uo pipefail; cd "$(dirname "$0")/.."
LOOP="--ne 100 --wall 14400 --outer 20 --inner-budget iters --inner-maxit 10 --inner-maxit-adapt \
 --select-cons on --ctrl-imcap off --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on \
 --ctrl-line eps --cert-sweep 61 --cert-rtol 1e-6 --alpha-argmax on --enroll-rel 0.05 --peak-follow on \
 --row-screen 0.05 --converge --inner sqp --sqp-carry --sqp-trust 0.01 --sqp-ctrl block --sqp-row-res-tol 1e-3 --sqp-refine 1e-4 --sqp-clblock on"
out=results_n100_kbc; mkdir -p $out
if [ ! -f $out/K_B_C_s0_ne100.json ]; then
  echo "=== launch $(date +%s)" >> $out/run.log
  python run_case_epi.py --eps 0.05 --case K+B+C --start 0 $LOOP --resume results_ne100/K_B_C_s0_ne100.npz \
    --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
fi
out=results_n100_fb80; mkdir -p $out
if [ ! -f $out/K_FB_FC_s0_ne100.json ]; then
  echo "=== launch $(date +%s)" >> $out/run.log
  python run_case_epi.py --eps 0.05 --case K+FB+FC --start 0 $LOOP --resume results_ne100/K_FB_FC_s0_ne100.npz \
    --field-basis laplace --field-n 80 --field-lo -38.9 --field-hi 39.2 \
    --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
fi
echo "=== ne100 nominal done"
