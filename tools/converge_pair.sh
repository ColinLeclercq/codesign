#!/usr/bin/env bash
# One start to CONVERGENCE on both inner solvers, Re <= -eps everywhere: the
# stage-2 correction's test on one case (K+B+C, ne=100, hifoo_000).
set -uo pipefail; cd "$(dirname "$0")/.."
C="--case K+B+C --start 0 --ne 100 --wall 3000 --outer 40 --inner-budget iters --inner-maxit 10 --inner-maxit-adapt --cold --seed hifoo_000 --seed_file rand_hifoo_ne100.npz --stabilise --select-cons on --ctrl-imcap off --cont-anchor strict --cont-accept-winding --sigma2 on --oracle reduced --ctrl-line eps --converge --band-first --band-check"
for s in sqp pygranso; do
  out=results_conv_$s; mkdir -p $out
  [ -f $out/K_B_C_s0_ne100.json ] && continue
  extra=""; [ $s = sqp ] && extra="--sqp-carry --sqp-trust 0.01"
  python run_case_epi.py --eps 0.05 $C --inner $s $extra --outdir $out > $out/run.log 2>&1
  grep -o '"gamma": [0-9.]*' $out/run.log | tail -1
done
echo "=== pair done"
