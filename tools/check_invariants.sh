#!/usr/bin/env bash
# Invariant checks for the nominal loop (2026-09-25, after two defects that the
# logs showed for days: the fake certificate witness and --converge never firing).
#  1. frozen path bit-identical (case B, ne=100): 10.23128516441147
#  2. a short K+B+C ne=100 run on the nominal flags: every round's witness attains
#     the certified value (witness_invariant_violations == 0 in the result json)
#  3. hurwitz selftest 0/200
set -uo pipefail; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
tmp=$(mktemp -d)
g=$(python run_case_epi.py --case B --ne 100 --wall 60 --outer 2 --eps 0.05 --cold --outdir $tmp/rg \
    --inner-budget iters --inner-maxit 25 2>&1 | grep -oE '"gamma": [0-9.]+' | awk '{print $2}')
[ "$g" = "10.23128516441147" ] && echo "PASS frozen path ($g)" || echo "FAIL frozen path ($g)"
LOOP=$(sed -n '/^LOOP=/,/stop-progress [0-9:.]*"/p' tools/n100_kbc.sh | tr -d '\\\n' | sed 's/^LOOP="//; s/"$//' | sed -E 's/--outer [0-9]+/--outer 4/')
python run_case_epi.py --eps 0.05 --case K+B+C --start 0 $LOOP --resume results_ne100/K_B_C_s0_ne100.npz --outdir $tmp/kbc > $tmp/kbc.log 2>&1
python - "$tmp/kbc/K_B_C_s0_ne100.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1])); v = j.get('witness_invariant_violations')
print(("PASS" if v == 0 else "FAIL") + f" witness invariant ({v} violations in {j.get('outer')} rounds)")
PY
python hurwitz.py | tail -1
rm -rf $tmp
