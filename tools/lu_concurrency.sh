#!/usr/bin/env bash
# Aggregate factorisation throughput vs concurrency on ONE node (cavity).
# Usage: bash tools/lu_concurrency.sh A.dat M.dat [plant|pencil2n] [threads_per_job]
# Runs k = 6, 12, 24, 48 concurrent tools/lu_footprint.py at distinct
# frequencies; prints factorisations per minute for each k.  The number of
# frequencies to run in parallel is where this stops rising (bandwidth) or
# memory (INFOG(17) x k) approaches ~85 % of 512 GB, whichever comes first.
set -uo pipefail
A=$1; M=$2; WHICH=${3:-plant}; T=${4:-2}
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
for k in 6 12 24 48; do
  t0=$(date +%s.%N)
  for i in $(seq 1 $k); do
    python "$(dirname "$0")/lu_footprint.py" --A "$A" --M "$M" --which "$WHICH" \
      --omega "$(python -c "print(0.1 + 0.05 * $i)")" > "lu_conc_${WHICH}_k${k}_$i.log" 2>&1 &
  done
  wait
  t1=$(date +%s.%N)
  python - "$k" "$t0" "$t1" <<'PY'
import sys; k = int(sys.argv[1]); dt = float(sys.argv[3]) - float(sys.argv[2])
print(f"k = {k:3d} concurrent: {dt:7.1f} s wall -> {60 * k / dt:6.1f} factorisations / minute")
PY
done
