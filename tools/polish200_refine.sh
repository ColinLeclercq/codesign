#!/usr/bin/env bash
# ne=200 terminal polish of the frozen K+B+C design with the position-robust
# reduced model (STAGE5 5): certification every round (1) vs every 4th (4).
set -uo pipefail; cd "$(dirname "$0")/.."
for k in refine; do
  out=results_p200_$k; mkdir -p $out/dump
  [ -f $out/K_B_C_s0_ne200.json ] && continue
  python run_case_epi.py --case K+B+C --start 0 --ne 200 --eps 0.05 --wall 7200 --outer 8 \
    --inner-budget iters --inner-maxit 10 --inner-maxit-adapt --outdir $out \
    --resume results/K_B_C_s0_ne200.npz --select-cons on --ctrl-imcap off --cont-anchor defer \
    --sigma2 on --oracle reduced --oracle-inputs on --ctrl-line eps --inner sqp --sqp-carry \
    --sqp-trust 0.01 --sqp-ctrl block --cert-sweep 61 --cert-every 1 --sqp-refine 1e-4 \
    --dump $out/dump > $out/run.log 2>&1
done
echo "=== p200 cert-every A/B done"
