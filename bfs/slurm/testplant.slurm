#!/bin/bash
#SBATCH --job-name=bfs-testplant
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=logs/bfs_testplant_%j.log
#
#   sbatch bfs/slurm/testplant.slurm                       (zero controller, 3 rounds)
#   KIND=hifoo OUTER=5 sbatch bfs/slurm/testplant.slurm    (HIFOO start, 5 rounds)
#   OUTER=300 sbatch --time=15:00:00 bfs/slurm/testplant.slurm
#   R=10 OUTER=300 sbatch --time=15:00:00 bfs/slurm/testplant.slurm   (order 10)
#        (as many rounds as fit: the driver stops before the job's end, or when
#         progress stops)
#   RESUME=1 OUT=testplant_bfs/zero_s0_r5 OUTER=10 sbatch bfs/slurm/testplant.slurm
#        (10 more rounds for that run: from its checkpoint if it was stopped, from
#         its record if it finished; without RESUME the directory is started over)
#
# One start on the small test plant (n = 33,593, bfs/mesh_coarse.py), the
# production path on 24 ranks (BFS.md 6): the start (KIND=zero: HIFOO's
# compensator draw with the loop open, gamma = the open-loop peak), OUTER
# rounds with gamma certified on [0, 10], the all-frequency count, the figures
# (bfs/postproc.py).  The test plant's operators and certified range/poles are
# made first if missing or stale (as smoke.slurm steps 1 and 4).
# Expected (from the sandbox's serial timings): start-up 5 to 10 min, 5 to 8
# min per round, 5 min for the end; about 35 to 45 min for 3 rounds, 10 min
# more when the plant data are made.
set -eo pipefail
trap 'rc=$?; echo ""; echo "*** FAILED: line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ./env.sh
python bfs/version_check.py || { echo "*** mixed code versions (bfs/version_check.py): stop"; exit 1; }
LAUNCHER=${LAUNCHER:-mpiexec}
KIND=${KIND:-zero}; OUTER=${OUTER:-3}; SEED=${SEED:-0}
export R=${R:-5}                          # controller order (R=10: an order-10 controller)
OUT=${OUT:-testplant_bfs/${KIND}$([ "$R" = 5 ] || echo "_o$R")_s${SEED}_r${OUTER}}
OPS=bfs/data/ops_coarse.npz
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)  $KIND start, order $R, $OUTER rounds -> $OUT ==="
python -c "import clarabel, qdldl, matplotlib" || { echo "*** clarabel / qdldl / matplotlib missing: run bash bfs/setup_cluster.sh on the login node"; exit 1; }
if ! python -c "import numpy as np, sys; sys.exit(0 if 'shear_S_data' in np.load('$OPS').files else 1)" 2>/dev/null; then
    echo "=== test plant operators (missing or without the shear trace) ==="
    [ -f bfs/data/coarse/mesh.xdmf ] || python bfs/mesh_coarse.py
    python bfs/ops_dolfinx.py --data bfs/data/coarse --baseflow baseflow.npz --out "$OPS"
    rm -f bfs/data/ops_coarse_range*.npz
fi
if [ ! -f bfs/data/ops_coarse_range.npz ]; then
    echo "=== the test plant's certified range and poles (bfs/plant_cert.py) ==="
    time $LAUNCHER -n "${SLURM_NTASKS:-24}" python -u bfs/plant_cert.py --ops "$OPS"
fi
CONT=()
if [ "${RESUME:-0}" = "1" ] && [ -f "$OUT/run/K_B_C_s0_bfs.npz" -o -f "$OUT/run/K_B_C_s0_bfs_ckpt.npz" ]; then
    echo "=== RESUME: $OUT/run continues (checkpoint if present, else the finished record) ==="
    CONT=(--continue-record 1)
else
    rm -rf "$OUT"; mkdir -p "$OUT"
    python bfs/hifoo_init.py --ops "$OPS" --n 1 --seed "$SEED" --kind "$KIND" --r "$R" --out "$OUT/starts" > "$OUT/starts.json"
fi
# the deadline from the job's own end time (any --time): no round starts that
# would not end, 5 min kept for the figures (bfs/slurm_remaining.py)
export DEADLINE=${DEADLINE:-$(python bfs/slurm_remaining.py --reserve 300)}
export OPS OUTER PY=python LAUNCH="$LAUNCHER -n ${SLURM_NTASKS:-24}"
time bash bfs/run_bfs.sh "$OUT/run" "$OUT/starts/${KIND}_000.npz" --deadline-final 600 "${CONT[@]}"   # the test plant's end: ~3 min
time $LAUNCHER -n "${SLURM_NTASKS:-24}" python -u bfs/postproc.py --ops "$OPS" --rec "$OUT/run/K_B_C_s0_bfs.npz"
python - "$OUT/run/K_B_C_s0_bfs.json" "$OUT/run/K_B_C_s0_bfs_post/post.json" <<'EOF'
import json, sys
r, p = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
print("best gamma after each round:", [round(h['gamma_best'], 4) for h in r['history']],
      "(candidates certified:", [round(h['gamma_certified'], 4) for h in r['history']], ")")
print(f"final gamma {r['gamma']:.6f}, open-loop peak {p['peak_open']['sigma'][0]:.6f} "
      f"(ratio {r['gamma'] / p['peak_open']['sigma'][0]:.4f}); wind_count {r['wind_count']}; xa {r['xa']} xs {r['xs']}")
EOF
echo "=== done $(date -Is) ==="
