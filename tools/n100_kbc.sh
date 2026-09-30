#!/usr/bin/env bash
# Debugging at ne=100 (user, 2026-09-24): the corrected nominal loop, run to
# convergence, on K+B+C and K+FB+FC (80 Laplacian modes per field), from the
# frozen ne=200 designs transferred to ne=100 (tools/transfer_ne.py).
set -uo pipefail; cd "$(dirname "$0")/.."
# single-threaded BLAS per process, as env.sh does on the cluster (2026-09-25: unset, two runs on 2 cores
# oversubscribed the threads and a 178x178 eigenvalue solve took 1.9 s instead of 0.03 s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOOP="--ne 100 --wall 14400 --outer 80 --inner-budget iters --inner-maxit 10 --inner-maxit-adapt \
 --select-cons on --ctrl-imcap off --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on \
 --ctrl-line eps --cert-sweep 0 --cert-disk strict --cert-near-k 2 --cert-climb slope --wind-ctrl exact --cert-rtol 1e-6 --alpha-argmax on --enroll-rel 0.05 --peak-follow on \
 --row-screen 0.05 --converge --inner sqp --sqp-carry --sqp-trust 0.01 --sqp-ctrl block --sqp-row-res-tol 1e-3 --sqp-refine 1e-4 --sqp-clblock on --cont-accept-winding --sigma2-active-gap 0.15 --follow-scan 0 --stop-progress 3:0.0015"
# env OUT / EXTRA: another output directory and extra driver flags (2026-09-28: --cont-track reduced, algebra item 11)
out=${OUT:-results_n100_kbc}; EXTRA=${EXTRA:-}; mkdir -p $out
if [ ! -f $out/K_B_C_s0_ne100.json ]; then
  echo "=== launch $(date +%s)" >> $out/run.log
  python run_case_epi.py --eps 0.05 --case K+B+C --start 0 $LOOP $EXTRA --resume results_ne100/K_B_C_s0_ne100.npz \
    --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
fi
# constraint-aware KKT residual at active-rtol 1e-4, reported with the progress stop (STAGE5 28, 30)
if [ -f $out/K_B_C_s0_ne100.npz ] && [ ! -f $out/kkt_1e-4.json ]; then
  python tools/kkt_residual.py --ne 100 --case K+B+C --record $out/K_B_C_s0_ne100.npz --active-rtol 1e-4 \
    --json $out/kkt_1e-4.json >> $out/run.log 2>&1
fi
echo "=== kbc done"
