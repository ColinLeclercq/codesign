#!/usr/bin/env bash
# Clean comparison at ne=200 (TRACKER I): certify every round vs every 4th,
# both from the frozen K+B+C design, identical flags and tolerances
# (cert-rtol 1e-6, alpha-argmax, row filter 1e-3, in-solve refinement 1e-4,
# position-robust model).  One dump directory per launch (resumes restart the
# dump numbering); an arm's full-model cost = sum over its launches of the
# last n_lu_full_cum.
set -uo pipefail; cd "$(dirname "$0")/.."
for k in 1 4; do
  out=results_p200f_ce$k; mkdir -p $out
  [ -f $out/K_B_C_s0_ne200.json ] && continue
  echo "=== launch $(date +%s) cert-every $k" >> $out/run.log
  python run_case_epi.py --case K+B+C --start 0 --ne 200 --eps 0.05 --wall 14400 --outer 8 \
    --inner-budget iters --inner-maxit 10 --inner-maxit-adapt --outdir $out \
    --resume results/K_B_C_s0_ne200.npz --select-cons on --ctrl-imcap off --cont-anchor defer \
    --sigma2 on --oracle reduced --oracle-inputs on --ctrl-line eps --inner sqp --sqp-carry \
    --sqp-trust 0.01 --sqp-ctrl block --cert-sweep 61 --cert-rtol 1e-6 --alpha-argmax on \
    --sqp-row-res-tol 1e-3 --sqp-refine 1e-4 --cert-every $k \
    --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
done
echo "=== p200 pair done"
