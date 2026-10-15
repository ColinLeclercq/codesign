#!/bin/bash
#SBATCH --comment=36973004F
#SBATCH --job-name=bfs-codesign
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=240G
#SBATCH --time=15:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/bfs_codesign_%A_%a.log
#
#   sbatch bfs/slurm/codesign.slurm                  (after bfs/slurm/smoke.slurm passed)
#   sbatch --array=0-15 bfs/slurm/codesign.slurm     (more starts)
#   CONTINUE=1 sbatch --array=0-7 bfs/slurm/codesign.slurm
#                     (finished starts continue from their records for OUTER
#                      more rounds; unfinished ones resume from their checkpoints
#                      in any case)
#   KIND=hifoo sbatch --array=0-1 bfs/slurm/codesign.slurm
#                     (HIFOO's draws scaled to stability instead of the zero
#                      controller, the default since 2026-10-03: the zero start
#                      begins at the open-loop peak, the HIFOO ones above it)
#   M=2 P=2 STRUCT=central   sbatch --array=0-1 bfs/slurm/codesign.slurm
#   M=2 P=2 STRUCT=decentral sbatch --array=0-1 bfs/slurm/codesign.slurm
#                     (MIMO, BFS.md 5: two actuators, two sensors, order 10 in
#                      all; directories <KIND>_m2p2_<STRUCT>_o10_XXX)
#   ORACLE=lattice LATTICE_CAP=16 ... sbatch ...
#        (the inner loop on the FULL model, lattice.py, BFS.md 5: every rank keeps
#         up to LATTICE_CAP plant factors (default 16) -- 133 MB each on the test
#         plant, ~1.05 GB on the full mesh -- so give the job ranks x cap x that in
#         memory; the directories get _lat, e.g. zero_m2p2_central_o10_lat_s0_r300)
#   ORACLE=lattice LATTICE_K=2 ... sbatch ...
#        (two eigenvalues per disk centre of the stability line, each with its
#         row, --lattice-k 2; the directories get _k2)
#   ORACLE=lattice STRIP=0 ... sbatch ...
#        (no strip, --lattice-strip off: the stability rows are those of the
#         eigenvalues the line's disks see -- the A/B of whether the strip pays;
#         the directories get _nostrip)
#   ORACLE=lattice CROSS=cut ... sbatch ...
#        (a trial across the closed loop's line from an admissible iterate is
#         halved, as up to 20261010c, instead of weighed by the merit; the
#         directories get _cut)
#   ORACLE=lattice SHRINK=1 ... sbatch ...
#        (trust radius 2 alpha |d| after a partial step, --sqp-partial-shrink on;
#         off by default since 20261008c; the directories get _shr)
#   ORACLE=lattice RETIRE=0.95:3 ... sbatch ...
#        (members of Omega below 0.95 gamma for 3 rounds retire, --omega-retire;
#         the directories get _ret, e.g. zero_m2p2_central_o10_lat_ret_s0_r300)
#
# SISO or MIMO co-design on the backward-facing step (BFS.md 5): one start per
# array task (bfs/hifoo_init.py, SEED, start = the array index: HIFOO's
# compensator draw at the devices of Barbagallo et al. 2012, B2 and C1; MIMO
# adds the actuator at (1.6, 0.9) and C2), each on
# 24 cores -- rank 0 the driver, 23 MUMPS workers (bfs/mpipool.py) -- and 240
# GB: TWO starts per 96-core, 512-GB node.  The gamma certificate factors the
# 2n even pencil (58 s, 3.9 GB in the sandbox at n = 191,720) in every worker
# at once.  Real controller of order 5 (MIMO: 10 in all); actuator: Gaussian body force on the
# vertical momentum, free centre; sensor: floor shear-stress window, free start
# (BFS.md 2).  gamma CERTIFIED on [0, 10] every round; at the end, the
# all-frequency winding count.
#
# Needs bfs/slurm/plant_cert.slurm first (the plant's numerical range and
# certified poles, once per mesh).
#
# Time: the driver stops starting rounds when one more would not fit before
# DEADLINE (from the job's end time, whatever --time is: 45 min before it; with
# the driver's own 30 min reserve the last round ends at least 75 min before the
# end, for the final certificate, the winding count, the save and the figures),
# so the job finishes inside its limit with a record
# (OUT/KIND_XXX/$TAG.json/.npz).  Re-submitting the same command
# resumes every unfinished start from its checkpoint (accepted designs, the
# working set and the band are stored); a finished start is not redone.
set -eo pipefail
trap 'rc=$?; echo ""; echo "*** FAILED: line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ./env.sh
python bfs/version_check.py || { echo "*** mixed code versions (bfs/version_check.py): stop"; exit 1; }
LAUNCHER=${LAUNCHER:-mpiexec}
SEED=${SEED:-0}; NSTART=${NSTART:-64}
OUT=${OUT:-results_bfs}
IDX=${SLURM_ARRAY_TASK_ID:-0}
KIND=${KIND:-zero}                       # zero (HIFOO's compensator draw, loop open: gamma starts at the open-loop peak) | hifoo
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
if [ "${ORACLE:-reduced}" = lattice ] && [ "${STRIP:-1}" = 0 ]; then SUF="${SUF}_nostrip"; fi
if [ "${ORACLE:-reduced}" = lattice ] && [ "${CROSS:-merit}" = cut ]; then SUF="${SUF}_cut"; fi
export ORACLE=${ORACLE:-reduced} LATTICE_CAP=${LATTICE_CAP:-16} RETIRE=${RETIRE:-} SHRINK=${SHRINK:-0} LATTICE_K=${LATTICE_K:-1} STRIP=${STRIP:-1} CROSS=${CROSS:-merit}
TAG=K_B_C_s0_bfs                                       # the driver's record name
if [ "$MIMO" = 1 ]; then TAG=K_B_C_s0_bfs_mimo${M}x${P}_${STRUCT}; fi
NAME=$(printf "%s_%03d" "$KIND" "$IDX")
RUN=$OUT/${KIND}${SUF}_$(printf "%03d" "$IDX")
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none} (array ${SLURM_ARRAY_JOB_ID:-}_$IDX)  $(date -Is) ==="
echo "=== python: $(command -v python) ($(python -V 2>&1)); start $NAME (seed $SEED, order $R) -> $RUN ==="
# the full mesh's plant files: their content-id directory (bfs/plant_id.py,
# 20261010j), made by smoke.slurm (operators) and plant_cert.slurm (range, poles)
PD=$(python bfs/plant_id.py --dir fx)
OPS=$PD/ops_fx.npz
[ -f "$OPS" ] || { echo "*** $OPS missing: run bfs/slurm/smoke.slurm first (this code's plant id)"; exit 1; }
[ -f "$PD/ops_fx_range.npz" ] || { echo "*** $PD/ops_fx_range.npz missing: run bfs/slurm/plant_cert.slurm first"; exit 1; }
source bfs/lib.sh
post() {      # the figures and data of the saved record (bfs/postproc.py, BFS.md 7)
    timeout 3600 $LAUNCHER -n "${SLURM_NTASKS:-24}" python -u bfs/postproc.py --ops "$OPS" --rec "$RUN/$TAG.npz" \
        || echo "*** postproc failed (the record is saved; rerun: sbatch the same array)"
}
CONT=()
# the run directories are shared by every code version (bfs/home.sh): a
# checkpoint left by ANOTHER version is not resumed silently (the STAGE0 17.1
# rule): RESUME=1 (or CONTINUE=1) says so; every job notes its version in
# RUN/code_versions.txt
VER=$(sed -n 's/^# version //p' bfs/MANIFEST)
if [ -f "$RUN/${TAG}_ckpt.npz" ] && [ -f "$RUN/code_versions.txt" ] && [ "${RESUME:-0}" != 1 ] && [ "${CONTINUE:-0}" != 1 ] \
        && [ "$(tail -1 "$RUN/code_versions.txt" | awk '{print $2}')" != "$VER" ]; then
    echo "*** $RUN holds a checkpoint of code version $(tail -1 "$RUN/code_versions.txt" | awk '{print $2}'), this is $VER: RESUME=1 resumes it"; exit 1
fi
if [ -f "$RUN/$TAG.json" ] && [ ! -f "$RUN/${TAG}_ckpt.npz" ]; then
    if [ "${CONTINUE:-0}" = "1" ]; then
        echo "start $NAME finished: CONTINUE=1, OUTER more rounds from its record"
        CONT=(--continue-record 1)
    else
        echo "start $NAME already finished ($RUN/$TAG.json; CONTINUE=1 to give it more rounds)"
        [ -f "$RUN/${TAG}_post/post.json" ] || post
        exit 0
    fi
fi
# the catalogue is deterministic in (SEED, NSTART): every array task draws the
# same set into its own directory and takes its own entry
mkdir -p "$RUN"
run_lock "$RUN" || exit 1
trap 'drop_lock "$RUN/.job.lockd"' EXIT
PID=$(basename "$PD")
LASTP=$( [ -f "$RUN/code_versions.txt" ] && awk 'NF >= 5 {p = $5} END {print p}' "$RUN/code_versions.txt" || true)
if [ -f "$RUN/${TAG}_ckpt.npz" -o -f "$RUN/$TAG.json" ] && [ -n "$LASTP" ] && [ "$LASTP" != "$PID" ] && [ "${PLANT_CHANGED:-0}" != 1 ]; then
    echo "*** $RUN was run on the plant $LASTP, this code's is $PID (bfs/plant_id.py): PLANT_CHANGED=1 continues it on the new plant"; exit 1
fi
echo "$(date -Is) $VER $(git rev-parse --short HEAD 2>/dev/null || echo -) ${SLURM_JOB_ID:-none} $PID" >> "$RUN/code_versions.txt"
python bfs/hifoo_init.py --ops "$OPS" --n "$NSTART" --seed "$SEED" --kind "$KIND" --r "$R" --m "$M" --p "$P" --struct "$STRUCT" --out "$RUN/catalogue" > "$RUN.catalogue.json"
# the deadline from the job's own end time (any --time): 45 min kept after the
# driver's last round for the figures, on top of its own 30 min for the final
# certificate and count (bfs/slurm_remaining.py)
export OPS OUTER=${OUTER:-40} DEADLINE=${DEADLINE:-$(python bfs/slurm_remaining.py --reserve 2700)} PY=python
export LAUNCH="$LAUNCHER -n ${SLURM_NTASKS:-24}"
bash bfs/run_bfs.sh "$RUN" "$RUN/catalogue/$NAME.npz" "${CONT[@]}"
# the figures: made by the driver itself (--post 1); a bounded second launch only if missing
[ -f "$RUN/$TAG.json" ] && [ ! -f "$RUN/${TAG}_post/post.json" ] && post
echo "=== done $(date -Is) ==="
