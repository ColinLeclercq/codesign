#!/usr/bin/env bash
# SISO path of the MIMO branch bit-identical to the SISO tree (STAGE6 6.2):
# the same 5-round K+B+C ne=100 run (nominal loop flags, resumed from
# results_ne100) in THIS tree and in a reference tree (default: the siso-v1
# checkout at ../work); the certified gammas must agree to the last digit.
# This tree runs with --sqp-stat rel (the siso-v1 stationarity test; the new
# default 'pred' is an algorithm change, STAGE6 6.14): everything else must be
# identical.
#   tools/check_siso_identity.sh [REF_TREE]
set -uo pipefail; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ref=${1:-../work}
LOOP=$(sed -n '/^LOOP=/,/stop-progress [0-9:.]*"/p' tools/n100_kbc.sh | tr -d '\\\n' | sed 's/^LOOP="//; s/"$//' | sed -E 's/--outer [0-9]+/--outer 5/')
tmp=$(mktemp -d)
run() {   # $1 tree, $2 out, $3 extra driver flags
  (cd "$1" && python run_case_epi.py --eps 0.05 --case K+B+C --start 0 $LOOP ${3:-} \
      --resume results_ne100/K_B_C_s0_ne100.npz --outdir "$2" > "$2.log" 2>&1)
  grep -oE '"gamma": [0-9.]+' "$2.log" | tail -1 | awk '{print $2}'
}
g1=$(run "$PWD" $tmp/this "--sqp-stat rel")
g0=$(run "$(cd "$ref" && pwd)" $tmp/ref)
[ -n "$g1" ] && [ "$g1" = "$g0" ] && echo "PASS SISO identity ($g1)" || echo "FAIL SISO identity (this $g1, reference $g0)"
rm -rf $tmp
