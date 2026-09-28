#!/usr/bin/env bash
# ne=200 K+B+C on the code validated at ne=100 (2026-09-25; user: ne=200 only
# once ne=100 works): same flags as tools/n100_kbc.sh, from the frozen design.
set -uo pipefail; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOOP=$(sed -n '/^LOOP=/,/stop-progress [0-9:.]*"/p' tools/n100_kbc.sh | tr -d '\\\n' | sed 's/^LOOP="//; s/"$//' | sed 's/--ne 100/--ne 200/; s/--outer 20/--outer 40/; s/--wall 14400/--wall 36000/')
out=results_n200_kbc; mkdir -p $out
echo "=== launch $(date +%s): $LOOP" >> $out/run.log
python run_case_epi.py --eps 0.05 --case K+B+C --start 0 $LOOP --resume results/K_B_C_s0_ne200.npz \
  --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
# constraint-aware KKT residual at active-rtol 1e-4, reported with the progress stop (STAGE5 28, 30)
if [ -f $out/K_B_C_s0_ne200.npz ] && [ ! -f $out/kkt_1e-4.json ]; then
  python tools/kkt_residual.py --ne 200 --case K+B+C --record $out/K_B_C_s0_ne200.npz --active-rtol 1e-4 \
    --json $out/kkt_1e-4.json >> $out/run.log 2>&1
fi
echo "=== n200 kbc done"
