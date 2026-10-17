#!/bin/bash
#SBATCH --comment=36973004F
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
#         progress stops -- under 0.15 % over 8 rounds, SISO and MIMO, since
#         2026-10-16; bfs/run_bfs.sh)
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
#   ORACLE=lattice LATTICE_K=2 ... sbatch ...
#        (two eigenvalues per disk centre of the stability line, each with its
#         row, --lattice-k 2; the directories get _k2)
#   ORACLE=lattice CROSS=cut ... sbatch ...
#        (a trial across the closed loop's line from an admissible iterate is
#         halved, as up to 20261010c, instead of weighed by the merit; the
#         directories get _cut)
#   ORACLE=lattice SHRINK=1 ... sbatch ...
#        (trust radius 2 alpha |d| after a partial step, --sqp-partial-shrink on;
#         off by default since 20261008c; the directories get _shr)
#   PLANT_ONLY=1 sbatch --ntasks=48 --mem=100G --time=03:00:00 bfs/slurm/testplant.slurm
#        (build this code's test plant files, if missing, and stop; then submit the
#         batch with --dependency=afterok:<that job>: a job finding the plant being
#         built by another job stops with that hint instead of waiting on its
#         node; WAIT_PLANT=1 waits)
#   ORACLE=lattice ASYNC=0 ... sbatch ...
#        (the missed-peak level set synchronous, --lattice-async off: the
#         measurement arm.  The default since 2026-10-16 runs it beside the next
#         SQP iteration -- the same iterates, 0.71-0.80 x the round wall in
#         20261010n; the directories get _sync)
#   EXTRA_ARGS="--inner-maxit 20 --sqp-trust 0.02" ... sbatch ...
#        (driver flags appended last, so they win over bfs/run_bfs.sh's; words
#         split on spaces, no quoting inside.  The directory name does not change:
#         a flag that changes the method needs its own OUT=)
#   START_PERTURB=1e-12 [START_PERTURB_SEED=1] ... sbatch ...
#        (the start moved by 1e-12 x max(|z0|, 1) along a seeded direction,
#         --start-perturb: the second arm of an A/A pair, the path noise an A/B
#         is read against -- 20261010s; the directory name does not change)
#   STRIP=1 is refused: the strip arm was removed on 2026-10-16 (BFS.md 8).
#   Directories: until then a lattice directory WITHOUT _nostrip held a strip
#   run, and the default names are those again -- a RESUME/CONTINUE of one
#   continues it without the strip.  An old *_nostrip* (or *_nostrip_async)
#   run continues with its explicit OUT=, e.g.
#   RESUME=1 OUT=testplant_bfs/zero_m2p2_central_o10_lat_nostrip_s0_r300 ORACLE=lattice M=2 P=2 STRUCT=central ... sbatch ...
#   bash bfs/abcopy.sh testplant_bfs/<run> testplant_bfs/<copy>; CONTINUE=1 OUT=testplant_bfs/<copy> ... sbatch ...
#        (an A/B arm from a shared parent, BFS.md 8: the copy carries the
#         parent's checkpoint or record and its loop state)
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
source bfs/lib.sh
strip_refused || exit 1                                # STRIP=1: removed on 2026-10-16 (BFS.md 8)
unset STRIP
AF=$(lattice_async_flag) || exit 1                     # ASYNC: 1 (default) or 0
read -ra XA <<< "${EXTRA_ARGS:-}"                      # EXTRA_ARGS: driver flags, appended last
SPA=()                                                 # START_PERTURB: the A/A pair's moved start (--start-perturb)
if [ -n "${START_PERTURB:-}" ] && [ "${START_PERTURB}" != 0 ]; then
    SPA=(--start-perturb "$START_PERTURB" --start-perturb-seed "${START_PERTURB_SEED:-1}")
fi
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
if [ "${ORACLE:-reduced}" = lattice ] && [ "${SHRINK:-0}" = 1 ]; then SUF="${SUF}_shr"; fi
if [ "${ORACLE:-reduced}" = lattice ] && [ "${LATTICE_K:-1}" != 1 ]; then SUF="${SUF}_k${LATTICE_K}"; fi
if [ "${ORACLE:-reduced}" = lattice ] && [ "${CROSS:-merit}" = cut ]; then SUF="${SUF}_cut"; fi
# ASYNC=0: the synchronous level set, a measurement arm (bfs/run_bfs.sh passes
# --lattice-async; on by default since 2026-10-16, no suffix then -- the _async
# directories of 20261010n were named by hand)
if [ "${ORACLE:-reduced}" = lattice ] && [ "$AF" = off ]; then SUF="${SUF}_sync"; fi
export ORACLE=${ORACLE:-reduced} LATTICE_CAP=${LATTICE_CAP:-24} RETIRE=${RETIRE:-} SHRINK=${SHRINK:-0} LATTICE_K=${LATTICE_K:-1} CROSS=${CROSS:-merit} ASYNC=${ASYNC:-1}
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
# The test plant's files (mesh, operators, certified range and poles) live in
# a directory named by their content id (bfs/plant_id.py, 20261010j): built
# once per id, never rebuilt in place, shared by every code version with the
# same generators (BFS_HOME/data/coarse-<id>, bfs/home.sh), and a version with
# other generators gets its own directory.  The builder holds a lock directory
# (mkdir is atomic on the shared file system, across nodes; 2026-10-08: jobs
# 765560-765562 built the same files at once and corrupted mesh.h5; 766730:
# flock was node-local) with its job id in it (bfs/lib.sh: a dead owner's
# lock is taken over, a live one waited for).  Each file is written under a
# private name and renamed into place whole.
export HDF5_USE_FILE_LOCKING=FALSE
PD=$(python bfs/plant_id.py --dir coarse)
OPS=$PD/ops_coarse.npz
mkdir -p "$PD"
echo "=== test plant files: $PD ==="
LOCKD=$PD/.build.lockd
# complete = operators with the shear trace, the certified range and the poles
# at both lines plant_cert.py makes; then the files are only read, no lock
plant_complete() {
    python -c "import numpy as np, sys; sys.exit(0 if 'shear_S_data' in np.load('$OPS').files else 1)" 2>/dev/null \
        && [ -f "$PD/ops_coarse_range.npz" ] && [ -f "$PD/ops_coarse_range_poles_-0.050000.npz" ] \
        && [ -f "$PD/ops_coarse_range_poles_-0.025000.npz" ]
}
if ! plant_complete; then
    # 20261010k: a job no longer WAITS for another job's build while holding its
    # node (20261010j: eight 48-rank jobs waited on 774317's plant_cert): it
    # stops and says what to resubmit after (WAIT_PLANT=1 waits as before);
    # PLANT_ONLY=1 builds the files and stops -- submit it first, then the
    # batch with --dependency=afterok:<its job id>
    owner=$(cat "$LOCKD/owner" 2>/dev/null || true)
    rc=0; job_alive "$owner" || rc=$?
    if [ -d "$LOCKD" ] && [ "$rc" = 0 ] && [ "$owner" != "${SLURM_JOB_ID:-}" ] && [ "${WAIT_PLANT:-0}" != 1 ]; then
        echo "*** the test plant $PD is being built by job $owner: resubmit this job with sbatch --dependency=afterok:$owner ... (or WAIT_PLANT=1 to wait on the node)"
        exit 3
    fi
    take_lock "$LOCKD" "test plant's files"
    trap 'drop_lock "$LOCKD"' EXIT
fi
if ! python -c "import numpy as np, sys; sys.exit(0 if 'shear_S_data' in np.load('$OPS').files else 1)" 2>/dev/null; then
    echo "=== test plant operators (missing or without the shear trace) ==="
    if ! python -c "import h5py, numpy as np; [h5py.File(f'$PD/coarse/{f}.h5', 'r') for f in ('mesh', 'facets')]; np.load('$PD/coarse/baseflow.npz')" 2>/dev/null; then
        TMPD=$PD/.coarse.$$
        rm -rf "$TMPD"
        python bfs/mesh_coarse.py --out "$TMPD"
        [ -e "$PD/coarse" ] && mv "$PD/coarse" "$PD/.coarse.broken.$$"
        mv "$TMPD" "$PD/coarse"
    fi
    python bfs/ops_dolfinx.py --data "$PD/coarse" --baseflow baseflow.npz --out "$PD/.ops.$$.npz"
    mv "$PD/.ops.$$.npz" "$OPS"
fi
if [ ! -f "$PD/ops_coarse_range.npz" ] || [ ! -f "$PD/ops_coarse_range_poles_-0.050000.npz" ] || [ ! -f "$PD/ops_coarse_range_poles_-0.025000.npz" ]; then
    echo "=== the test plant's certified range and poles (bfs/plant_cert.py) ==="
    time $LAUNCHER -n "${SLURM_NTASKS:-24}" python -u bfs/plant_cert.py --ops "$OPS"
fi
drop_lock "$LOCKD"
trap - EXIT
if [ "${PLANT_ONLY:-0}" = 1 ]; then
    plant_complete && echo "=== PLANT_ONLY: the test plant is complete in $PD ===" || { echo "*** PLANT_ONLY: the test plant in $PD is still incomplete"; exit 1; }
    exit 0
fi
# the run directory, for bfs/report.sh (20261010n: this script printed none, so
# report.sh found no run of a testplant job; it now also reads "RESUME: DIR/run")
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}${PAIR_ARM:+ (pair arm $PAIR_ARM)}  ${SLURM_NTASKS:-24} ranks  $(date -Is) -> $OUT ==="
[ ${#XA[@]} -eq 0 ] || echo "=== EXTRA_ARGS: ${XA[*]} (appended to bfs/run_bfs.sh's flags) ==="
[ ${#SPA[@]} -eq 0 ] || echo "=== START_PERTURB: ${SPA[*]} (the start moved: an A/A pair's second arm) ==="
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
# one driver per run directory (bfs/lib.sh run_lock; the directories are
# shared by every code version since 20261010i)
run_lock "$OUT" || exit 1
trap 'drop_lock "$OUT/.job.lockd"' EXIT
# a run continues on the plant it was made on: the plant id is noted with the
# code version, and a RESUME/CONTINUE on another plant id is refused
# (PLANT_CHANGED=1 moves it on purpose); lines before 20261010j carry no id
PID=$(basename "$PD")
LASTP=$( [ -f "$OUT/code_versions.txt" ] && awk '$5 ~ /^(coarse|fx)-[0-9a-f]+$/ {p = $5} END {print p}' "$OUT/code_versions.txt" || true)
if [ ${#CONT[@]} -gt 0 ] && [ -n "$LASTP" ] && [ "$LASTP" != "$PID" ] && [ "${PLANT_CHANGED:-0}" != 1 ]; then
    echo "*** $OUT was run on the plant $LASTP, this code's is $PID (bfs/plant_id.py): PLANT_CHANGED=1 continues it on the new plant"; exit 1
fi
echo "$(date -Is) $(sed -n 's/^# version //p' bfs/MANIFEST) $(git rev-parse --short HEAD 2>/dev/null || echo -) ${SLURM_JOB_ID:-none} $PID" >> "$OUT/code_versions.txt"
# the deadline from the job's own end time (any --time): no round starts that
# would not end, 5 min kept for the figures (bfs/slurm_remaining.py)
export DEADLINE=${DEADLINE:-$(python bfs/slurm_remaining.py --reserve 300)}
export OPS OUTER PY=python LAUNCH="$LAUNCHER -n ${SLURM_NTASKS:-24}"
# --deadline-final 1200 (was 600, the end's ~3 min): since 20261010j stage B the
# save counts right of -eps with the absolute step floor (the 369.87 record: 5434
# points against 3208 right of -eps/2) and a veto counts up to --veto-counts more
# designs while the deadline holds one
time bash bfs/run_bfs.sh "$OUT/run" "$OUT/starts/$NAME.npz" --deadline-final 1200 "${CONT[@]}" "${SPA[@]}" "${XA[@]}"
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
