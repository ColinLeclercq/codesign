#!/usr/bin/env bash
# Continuation of the ne=100 K+B+C run (2026-09-25) from its 20-round end point
# (first: 7.929388 on the old SQP; now: the new-code run, 7.846865 after 20 rounds and still descending)
# ended in Clarabel failures with both compensator poles ON the line (H0 = 0).
set -uo pipefail; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOOP=$(sed -n '/^LOOP=/,/stop-progress [0-9:.]*"/p' tools/n100_kbc.sh | tr -d '\\\n' | sed 's/^LOOP="//; s/"$//' | sed 's/--outer 40/--outer 30/')
out=results_n100_kbc_cont; mkdir -p $out
echo "=== launch $(date +%s)" >> $out/run.log
python run_case_epi.py --eps 0.05 --case K+B+C --start 0 $LOOP --resume results_n100_kbc/K_B_C_s0_ne100.npz \
  --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
echo "=== kbc cont done"
