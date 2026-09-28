#!/usr/bin/env bash
# (a) reduced-model certification on K+B+C (user, 2026-09-23): the corrected SQP
#     loop of results_corr_sqp + --cert-every 4 --prov-check 0 --cert-fail continue;
#     compare gamma, KKT residual, full-model factorisations;
# (b) then K+FB+FC, 80 Laplacian modes, on the faster code (tools/fb80.sh).
set -uo pipefail; cd "$(dirname "$0")/.."
out=results_romcert; mkdir -p $out
if [ ! -f $out/K_B_C_s0_ne200.json ]; then
  echo "=== launch $(date +%s)" >> $out/run.log
  python run_case_epi.py --eps 0.05 --case K+B+C --start 0 --ne 200 --wall 14400 --outer 12 --inner-budget iters \
    --inner-maxit 10 --inner-maxit-adapt --resume results/K_B_C_s0_ne200.npz --select-cons on \
    --ctrl-imcap off --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on --ctrl-line eps \
    --cert-sweep 61 --cert-rtol 1e-6 --alpha-argmax on --enroll-rel 0.05 --peak-follow on --converge \
    --inner sqp --sqp-carry --sqp-trust 0.01 --sqp-ctrl block --sqp-row-res-tol 1e-3 \
    --cert-every 4 --prov-check 0 --cert-fail continue --prov-gate reduced \
    --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
fi
bash tools/fb80.sh
