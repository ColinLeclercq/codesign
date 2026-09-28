#!/usr/bin/env bash
# K+FB+FC with the free fields on 80 Dirichlet-Laplacian modes, new method
# (SQP + block + position/field-robust reduced model + in-solve refinement),
# from the frozen design projected onto the basis (3.927660); 6 rounds.
# Reference: frozen nodal+Sobolev optimum 3.925456.
set -uo pipefail; cd "$(dirname "$0")/.."
out=results_fb_basis80; mkdir -p $out
[ -f $out/K_FB_FC_s0_ne200.json ] && exit 0
echo "=== launch $(date +%s)" >> $out/run.log
python run_case_epi.py --case K+FB+FC --start 0 --ne 200 --eps 0.05 --wall 14400 --outer 6 \
  --inner-budget iters --inner-maxit 10 --inner-maxit-adapt --resume results/K_FB_FC_s0_ne200.npz \
  --select-cons on --ctrl-imcap off --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on \
  --ctrl-line eps --cert-sweep 61 --cert-rtol 1e-6 --alpha-argmax on --inner sqp --sqp-carry \
  --sqp-trust 0.01 --sqp-ctrl block --sqp-row-res-tol 1e-3 --sqp-refine 1e-4 \
  --field-basis laplace --field-n 80 --field-lo -38.9 --field-hi 39.2 \
  --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
