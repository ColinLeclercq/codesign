#!/usr/bin/env bash
# SISO path of the MIMO branch against the SISO tree (STAGE6 6.2, 6.28):
# the same 5-round K+B+C ne=100 run (nominal loop flags, resumed from
# results_ne100) in THIS tree and in a reference tree (default: the siso-v1
# checkout at ../work); the certified gammas must agree to 1e-7 (see below).
# This tree runs with --sqp-stat rel, --cert-near-k 8, --cert-climb brent and
# --oracle-orth column (the siso-v1 stationarity test, certificate and basis
# orthogonalisation; the block orthogonalisation changes the near-dependent
# basis columns at 2e-8 and the 5-round value at 1.4e-4, STAGE6 6.29; the new
# default 'pred' is an algorithm change, STAGE6 6.14): everything else must be
# identical.
#   tools/check_siso_identity.sh [REF_TREE]
set -uo pipefail; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ref=${1:-../work}
LOOP=$(sed -n '/^LOOP=/,/stop-progress [0-9:.]*"/p' tools/n100_kbc.sh | tr -d '\\\n' | sed 's/^LOOP="//; s/"$//' | sed -E 's/--outer [0-9]+/--outer 5/')
tmp=$(mktemp -d)
# the reference tree predates --cert-near-k / --cert-climb: its run gets the loop without them
LOOP_REF=$(echo "$LOOP" | sed -E 's/--cert-near-k [0-9]+//; s/--cert-climb [a-z]+//')
run() {   # $1 tree, $2 out, $3 extra driver flags, $4 loop flags
  (cd "$1" && python run_case_epi.py --eps 0.05 --case K+B+C --start 0 $4 ${3:-} \
      --resume results_ne100/K_B_C_s0_ne100.npz --outdir "$2" > "$2.log" 2>&1)
  grep -oE '"gamma": [0-9.]+' "$2.log" | tail -1 | awk '{print $2}'
}
g1=$(run "$PWD" $tmp/this "--sqp-stat rel --cert-near-k 8 --cert-climb brent --oracle-orth column" "$LOOP")
g0=$(run "$(cd "$ref" && pwd)" $tmp/ref "" "$LOOP_REF")
# Bit-identical until 2026-09-30; since then the sigma_2 blocks keep their warm
# subspace across block sizes (reduced._fit_subspace, STAGE6 6.28), which changes
# the 5-round value at 3.6e-9 (8.754365078411837 against 8.754365110002194):
# agreement to 1e-7 relative is required, identity reported.  STAGE6 6.29 (with
# --oracle-orth column): 8.754365072650645, 4.3e-9.
python3 - "$g1" "$g0" <<'PY'
import sys
try:
    g1, g0 = float(sys.argv[1]), float(sys.argv[2])
except ValueError:
    print(f"FAIL SISO agreement (this {sys.argv[1]!r}, reference {sys.argv[2]!r})"); sys.exit()
rel = abs(g1 - g0) / g0
tag = 'identical' if g1 == g0 else f'rel {rel:.1e}'
print(("PASS" if rel <= 1e-7 else "FAIL") + f" SISO agreement ({g1!r} vs reference {g0!r}: {tag})")
PY
rm -rf $tmp
