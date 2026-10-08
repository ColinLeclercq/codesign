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
#   M=2 P=2 STRUCT=central   OUTER=300 sbatch --time=15:00:00 bfs/slurm/testplant.slurm
#   M=2 P=2 STRUCT=decentral OUTER=300 sbatch --time=15:00:00 bfs/slurm/testplant.slurm
#        (MIMO, BFS.md 5: two actuators at B2 and (1.6, 0.9), two sensors at C1 and C2;
#         total order R = 10 by default, decentral = two loops of 5; each case has its
#         own directory, e.g. testplant_bfs/zero_m2p2_central_o10_s0_r300)
#        (as many rounds as fit: the driver stops before the job's end, or when
#         progress stops)
#   RESUME=1 OUT=testplant_bfs/zero_s0_r5 OUTER=10 sbatch bfs/slurm/testplant.slurm
#        (10 more rounds for that run: from its checkpoint if it was stopped, from
#         its record if it finished; CONTINUE=1 is the same.  A directory that
#         holds a run is never deleted: without RESUME/CONTINUE the job refuses
#         to start; FRESH=1 starts over and keeps the old one as OUT.old-<date>)
#   KIND=hifoo R=10 OUTER=300 sbatch --array=0-7 --time=15:00:00 bfs/slurm/testplant.slurm
#   KIND=hifoo M=2 P=2 STRUCT=decentral OUTER=300 sbatch --array=0-7 --time=15:00:00 bfs/slurm/testplant.slurm
#        (MULTIPLE STARTS: array task i takes entry i of the catalogue of NSTART
#         draws (default 64, deterministic in SEED) into its own directory,
#         e.g. testplant_bfs/hifoo_o10_s0_r300_i003; without --array the
#         directory names are those above, entry 0)
#   ORACLE=lattice LATTICE_CAP=24 ... sbatch ...
#        (the inner loop on the FULL model, lattice.py, BFS.md 5: every rank keeps
#         up to LATTICE_CAP plant factors (default 24) -- 133 MB each on the test
#         plant, ~1.05 GB on the full mesh -- so give the job ranks x cap x that in
#         memory; the directories get _lat, e.g. zero_m2p2_central_o10_lat_s0_r300)
#   ORACLE=lattice RETIRE=0.95:3 ... sbatch ...
#        (members of Omega below 0.95 gamma for 3 rounds retire, --omega-retire;
#         the directories get _ret, e.g. zero_m2p2_central_o10_lat_ret_s0_r300)
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
export M=${M:-1} P=${P:-1} STRUCT=${STRUCT:-central}   # MIMO: M actuators, P sensors, central | decentral
MIMO=$([ "$M" = 1 ] && [ "$P" = 1 ] && echo 0 || echo 1)
export R=${R:-$([ "$MIMO" = 1 ] && echo 10 || echo 5)}   # controller order (MIMO: the total, default 10)
SUF=""
if [ "$MIMO" = 1 ]; then SUF="_m${M}p${P}_${STRUCT}"; fi
if [ "$R" != 5 ]; then SUF="${SUF}_o$R"; fi
if [ "${ORACLE:-reduced}" = lattice ]; then SUF="${SUF}_lat"; fi
if [ "${ORACLE:-reduced}" = lattice ] && [ -n "${RETIRE:-}" ]; then SUF="${SUF}_ret"; fi
export ORACLE=${ORACLE:-reduced} LATTICE_CAP=${LATTICE_CAP:-24} RETIRE=${RETIRE:-}
IDX=${SLURM_ARRAY_TASK_ID:-}                           # array job: the start index (catalogue entry)
NSTART=${NSTART:-64}
if [ -n "$IDX" ]; then
    OUT=${OUT:-testplant_bfs/${KIND}${SUF}_s${SEED}_r${OUTER}_i$(printf "%03d" "$IDX")}
else
    IDX=0; NSTART=1
    OUT=${OUT:-testplant_bfs/${KIND}${SUF}_s${SEED}_r${OUTER}}
fi
NAME=$(printf "%s_%03d" "$KIND" "$IDX")
TAG=K_B_C_s0_bfs                                       # the driver's record name
if [ "$MIMO" = 1 ]; then TAG=K_B_C_s0_bfs_mimo${M}x${P}_${STRUCT}; fi
OPS=bfs/data/ops_coarse.npz
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)  $KIND start $IDX, ${M}x${P} $([ "$MIMO" = 1 ] && echo "$STRUCT "), order $R, $OUTER rounds -> $OUT ==="
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
# NEVER delete a run (2026-10-06: CONTINUE=1, codesign.slurm's word, was not
# read here, and the directory of a 12.5 h run was started over): RESUME=1 or
# CONTINUE=1 continues it; without them a directory holding a run is refused,
# and FRESH=1 starts over after renaming the old one (OUT.old-<date>)
CONT=()
HAVE=0
if [ -f "$OUT/run/$TAG.npz" ] || [ -f "$OUT/run/${TAG}_ckpt.npz" ]; then HAVE=1; fi
if [ "${RESUME:-0}" = "1" ] || [ "${CONTINUE:-0}" = "1" ]; then
    if [ "$HAVE" != 1 ]; then
        echo "*** RESUME/CONTINUE: no record or checkpoint in $OUT/run ($TAG): nothing to continue"; exit 1
    fi
    echo "=== RESUME: $OUT/run continues (checkpoint if present, else the finished record) ==="
    CONT=(--continue-record 1)
else
    if [ -e "$OUT" ]; then
        if [ "$HAVE" = 1 ] && [ "${FRESH:-0}" != "1" ]; then
            echo "*** $OUT holds a run: RESUME=1 (or CONTINUE=1) continues it; FRESH=1 starts over (the old directory is kept as $OUT.old-<date>)"; exit 1
        fi
        mv "$OUT" "$OUT.old-$(date +%Y%m%d-%H%M%S)"
        echo "=== the previous $OUT kept as $OUT.old-<date> ==="
    fi
    mkdir -p "$OUT"
    python bfs/hifoo_init.py --ops "$OPS" --n "$NSTART" --seed "$SEED" --kind "$KIND" --r "$R" --m "$M" --p "$P" --struct "$STRUCT" --out "$OUT/starts" > "$OUT/starts.json"
fi
# the deadline from the job's own end time (any --time): no round starts that
# would not end, 5 min kept for the figures (bfs/slurm_remaining.py)
export DEADLINE=${DEADLINE:-$(python bfs/slurm_remaining.py --reserve 300)}
export OPS OUTER PY=python LAUNCH="$LAUNCHER -n ${SLURM_NTASKS:-24}"
time bash bfs/run_bfs.sh "$OUT/run" "$OUT/starts/$NAME.npz" --deadline-final 600 "${CONT[@]}"   # the test plant's end: ~3 min
# the figures are made by the driver itself (--post 1, same job and pool); a
# second launch only if they are missing, bounded in time (a launch that hangs
# must not hold the allocation to its limit)
if [ -f "$OUT/run/$TAG.npz" ] && [ ! -f "$OUT/run/${TAG}_post/post.json" ]; then
    timeout 3600 $LAUNCHER -n "${SLURM_NTASKS:-24}" python -u bfs/postproc.py --ops "$OPS" --rec "$OUT/run/$TAG.npz" \
        || echo "*** postproc failed (the record is saved; bfs/slurm/postproc.slurm redoes it)"
fi
[ -f "$OUT/run/${TAG}_post/post.json" ] && python - "$OUT/run/$TAG.json" "$OUT/run/${TAG}_post/post.json" <<'EOF'
import json, sys
r, p = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
print("best gamma after each round:", [round(h['gamma_best'], 4) for h in r['history']],
      "(candidates certified:", [round(h['gamma_certified'], 4) for h in r['history']], ")")
print(f"final gamma {r['gamma']:.6f}, open-loop peak {p['peak_open']['sigma'][0]:.6f} "
      f"(ratio {r['gamma'] / p['peak_open']['sigma'][0]:.4f}); wind_count {r['wind_count']}; xa {r['xa']} xs {r['xs']}")
EOF
echo "=== done $(date -Is) ==="
