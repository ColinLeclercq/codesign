#!/usr/bin/env bash
# Continuation of tools/n200_kbc.sh from its end point (2026-09-25), same flags.
set -uo pipefail; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOOP=$(sed -n '/^LOOP=/,/stop-progress [0-9:.]*"/p' tools/n100_kbc.sh | tr -d '\\\n' | sed 's/^LOOP="//; s/"$//' | sed 's/--ne 100/--ne 200/; s/--wall 14400/--wall 36000/')
src=${SRC:-results_n200_kbc/K_B_C_s0_ne200.npz}; out=${OUT:-results_n200_kbc_cont}; mkdir -p $out
echo "=== launch $(date +%s) from $src" >> $out/run.log
python run_case_epi.py --eps 0.05 --case K+B+C --start 0 $LOOP --resume $src \
  --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
echo "=== n200 kbc cont done"
