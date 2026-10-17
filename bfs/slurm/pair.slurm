#!/bin/bash
#SBATCH --comment=36973004F
#SBATCH --job-name=bfs-pair
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=15:00:00
#SBATCH --output=logs/bfs_pair_%j.log
#
#   ARM_A="OUT=testplant_bfs/ab_c151_sync ASYNC=0" ARM_B="OUT=testplant_bfs/ab_c151_async" \
#   CONTINUE=1 ORACLE=lattice M=2 P=2 STRUCT=central OUTER=300 sbatch --job-name=ab_c151 bfs/slurm/pair.slurm
#
#   ARM_A="OUT=testplant_bfs/pr_c151 STRUCT=central" ARM_B="OUT=testplant_bfs/pr_d370 STRUCT=decentral" \
#   ARM_C="OUT=testplant_bfs/pr_c177 STRUCT=central" ARM_D="OUT=testplant_bfs/pr_d412 STRUCT=decentral" \
#   ARM_C_ARGS="--inner-maxit 20" CONTINUE=1 ORACLE=lattice M=2 P=2 OUTER=300 sbatch bfs/slurm/pair.slurm
#        (a QUEUE: A and B start on the two halves, C takes the first half to
#         become free, D the next)
#
# Runs of bfs/slurm/testplant.slurm in ONE job on one 96-core node, two at a
# time, each on its own half of the cores (its own NUMA side) with half of the
# ranks.  The variables common to every arm are set as for testplant.slurm
# (EXTRA_ARGS included).  ARM_A, ARM_B, ARM_C, ... (letters in order: the
# queue) hold each arm's own VAR=value words, OUT required and distinct, no
# spaces inside a value; ARM_<L>_ARGS that arm's driver flags, appended to the
# common EXTRA_ARGS (an arm's words cannot carry EXTRA_ARGS: its value has
# spaces).  The first two arms start at once; each later one starts on the half
# whose arm ended first, as soon as it ends (20261010n: the 774364 c151 async
# half sat idle 1.8 h after its progress stop), unless less than QUEUE_MIN
# seconds (default 3600) are left in the job -- it is then reported, not
# started (a remaining time it cannot read: started).  Each arm writes
# logs/bfs_testplant_<job>_<OUT's name>.log, a log of the same form as a
# testplant job's (the names must differ); this job's own log says where and
# when each arm ran and how it ended, at a signal (the time limit) too.  STRIP=1
# (common or an arm's) is refused before anything starts: the strip arm was
# removed on 2026-10-16 (BFS.md 8); so is a DEADLINE that would reach a queued
# arm (it is counted from the driver's start).
#
# Why (2026-10-15): sator cancels a second job of the same user started on a
# node, as root, 6 s in, before its prolog (774342, 774343, 774346: "CANCELLED
# by 0", exit 0:54; the first job on each node ran on), and runs at most 8
# jobs per user (QOSMaxJobsPerUserLimit).  So 48-rank runs were one per node,
# half of each node idle.  Two arms per job give the "48 ranks, 2 per node" of
# the 20261010j plan (user, 2026-10-14), and both arms of an A/B pair on the
# same node.
#
# How: MPICH's Hydra launches under Slurm through an srun step, and a second
# launch in a job hung once ("srun: Unable to confirm allocation", BFS.md 8,
# LESSONS 9).  Here every launch uses Hydra's fork launcher (HYDRA_LAUNCHER=fork:
# the ranks are children of this script, no step), under taskset with one half
# of the job's CPUs (bfs/cpu_halves.py); the ranks inherit the mask.  A check
# launch per half proves it before the arms start.
set -eo pipefail
trap 'rc=$?; echo ""; echo "*** FAILED: line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ./env.sh
python bfs/version_check.py || { echo "*** mixed code versions (bfs/version_check.py): stop"; exit 1; }
source bfs/lib.sh
LAUNCHER=${LAUNCHER:-mpiexec}
export HYDRA_LAUNCHER=fork I_MPI_HYDRA_BOOTSTRAP=fork
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is): arms two at a time, MPI launcher $($LAUNCHER --version 2>&1 | grep -m1 -E '^ *Version:' | tr -s ' ' || echo '?') ==="

# the queue: ARM_A, ARM_B, ... in letter order, VAR=value words only (no spaces inside a value)
strip_refused || exit 1
unset STRIP
lattice_async_flag > /dev/null || exit 1
TAGS=(); WORDS=(); OUTS=(); XARGS=()
for L in {A..Z}; do
    v=ARM_$L; words=${!v:-}
    [ -n "$words" ] || continue
    for w in $words; do
        [[ "$w" =~ ^[A-Z_][A-Z0-9_]*=[^[:space:]]*$ ]] || { echo "*** $v: '$w' is not VAR=value"; exit 1; }
        case "$w" in EXTRA_ARGS=*) echo "*** $v: an arm's driver flags go in ARM_${L}_ARGS=\"...\" (appended to EXTRA_ARGS)"; exit 1 ;; esac
    done
    read -ra wa <<< "$words"
    strip_refused "${wa[@]}" || { echo "*** (in $v)"; exit 1; }
    for w in "${wa[@]}"; do case "$w" in ASYNC=*) ASYNC=${w#ASYNC=} lattice_async_flag > /dev/null || { echo "*** (in $v)"; exit 1; } ;; esac; done
    out=""
    for w in "${wa[@]}"; do case "$w" in OUT=*) out=${w#OUT=} ;; esac; done
    [ -n "$out" ] || { echo "*** $v has no OUT=: each arm needs its own run directory"; exit 1; }
    x=ARM_${L}_ARGS; xa=${EXTRA_ARGS:-}
    if [ -n "${!x:-}" ]; then xa="${xa:+$xa }${!x}"; fi
    TAGS+=("${L,,}"); WORDS+=("$words"); OUTS+=("$out"); XARGS+=("$xa")
done
NA=${#TAGS[@]}
[ "$NA" -ge 1 ] || { echo "*** no arm: ARM_A=\"OUT=... ...\" ARM_B=\"OUT=... ...\" (ARM_C, ... queue behind them)"; exit 1; }
# DEADLINE, the driver's seconds from ITS start (testplant.slurm reads the job's end when it is
# unset), cannot be given for an arm the queue starts later: counted from that arm's start, it
# would run the driver past the job's end, cut by the time limit before its save (the review of
# 2026-10-17).  The first two arms start at once and may carry it, as before the queue.
for ((i = 2; i < NA; i++)); do
    dl=${DEADLINE:-}
    read -ra wa <<< "${WORDS[$i]}"
    for w in "${wa[@]}"; do case "$w" in DEADLINE=*) dl=${w#DEADLINE=} ;; esac; done
    [ -z "$dl" ] || { echo "*** DEADLINE=$dl reaches the queued arm ${TAGS[$i]}: leave it unset (each arm then takes the job's end) or give it in ARM_A/ARM_B's words only"; exit 1; }
done
for ((i = 0; i < NA; i++)); do
    for ((j = i + 1; j < NA; j++)); do
        [ "$(realpath -m "${OUTS[$i]}")" != "$(realpath -m "${OUTS[$j]}")" ] \
            || { echo "*** arms ${TAGS[$i]} and ${TAGS[$j]} both have OUT=${OUTS[$i]}"; exit 1; }
        [ "$(basename "${OUTS[$i]}")" != "$(basename "${OUTS[$j]}")" ] \
            || { echo "*** arms ${TAGS[$i]} and ${TAGS[$j]}: OUT names $(basename "${OUTS[$i]}") alike (their logs would collide)"; exit 1; }
    done
done
for ((i = 0; i < NA; i++)); do
    echo "=== queue $((i + 1))/$NA: arm ${TAGS[$i]}: ${WORDS[$i]}${XARGS[$i]:+  [driver flags: ${XARGS[$i]}]} ==="
done

# the halves: CPUs and ranks
mapfile -t H < <(python bfs/cpu_halves.py)
python bfs/cpu_halves.py --describe
NT=${SLURM_NTASKS:-96}
HALF=$(( NT / 2 ))
for k in 0 1; do
    n=$(python -c "import sys; sys.path.insert(0, 'bfs'); import cpu_halves as c; print(len(c._expand('${H[$k]}')))")
    [ "$n" -ge "$HALF" ] || { echo "*** half $k has $n CPUs for $HALF ranks"; exit 1; }
done
for k in 0 1; do
    timeout 300 taskset -c "${H[$k]}" $LAUNCHER -n "$HALF" python bfs/cpu_halves.py --check "${H[$k]}" \
        || { echo "*** the check launch on half $k failed: the arms would share cores; stop"; exit 1; }
done

# the plant files first, once, on the whole node (testplant.slurm's PLANT_ONLY:
# if complete, a few seconds; the arms then only read them)
SLURM_NTASKS=$NT PLANT_ONLY=1 bash bfs/slurm/testplant.slurm

# the arms: each in the background on its half; its exit code in a file of RCD
# (bash 4.4 on EL8 has wait -n but not wait -p: which arm ended is read from
# the files, not from wait; node-local, every arm runs on this node)
RCD=$(mktemp -d "${TMPDIR:-/tmp}/pair_${SLURM_JOB_ID:-none}.XXXXXX")
trap 'rm -rf "$RCD"' EXIT
QUEUE_MIN=${QUEUE_MIN:-3600}
LOGS=(); RCS=(); HOW=()
for ((i = 0; i < NA; i++)); do LOGS+=("logs/bfs_testplant_${SLURM_JOB_ID:-none}_$(basename "${OUTS[$i]}").log"); RCS+=(""); HOW+=(""); done
start_arm() {          # start_arm HALF I: arm I on half HALF, in the background
    local k=$1 i=$2 wa
    read -ra wa <<< "${WORDS[$i]}"
    {
        echo "=== pair job ${SLURM_JOB_ID:-none} arm ${TAGS[$i]} ($((i + 1))/$NA) on $(hostname): half $k, CPUs ${H[$k]}, $HALF ranks, ${WORDS[$i]}${XARGS[$i]:+  [driver flags: ${XARGS[$i]}]} ==="
        r=0
        env "${wa[@]}" EXTRA_ARGS="${XARGS[$i]}" SLURM_NTASKS=$HALF PAIR_ARM=${TAGS[$i]} taskset -c "${H[$k]}" \
            bash bfs/slurm/testplant.slurm || r=$?
        echo "$r" > "$RCD/$i"
    } > "${LOGS[$i]}" 2>&1 &
    PIDS[$k]=$!; ON[$k]=$i
    HOW[$i]="half $k, started $(date -Is)"
    echo "=== arm ${TAGS[$i]} ($((i + 1))/$NA) on half $k (CPUs ${H[$k]}): ${WORDS[$i]} -> ${OUTS[$i]} (log ${LOGS[$i]}, pid ${PIDS[$k]}, $(date -Is)) ==="
}
# the time limit (or scancel): Slurm signals every process of the job, the arms' with it
# (their subshells reset the trap), and the summary at the end of this script never comes; this
# log then says which arms were cut and which never started (the review of 2026-10-17)
on_signal() {
    trap - TERM INT
    echo ""
    echo "*** pair job ${SLURM_JOB_ID:-none}: signal received $(date -Is); the queue then:"
    local i
    for ((i = 0; i < NA; i++)); do
        if [ -n "${RCS[$i]}" ]; then
            echo "    arm ${TAGS[$i]}: ended with exit ${RCS[$i]} (${HOW[$i]})"
        elif [ -n "${HOW[$i]}" ] && [[ "${HOW[$i]}" != "not started"* ]]; then
            echo "    arm ${TAGS[$i]}: CUT by the signal (${HOW[$i]}; its log ${LOGS[$i]})"
        else
            echo "    arm ${TAGS[$i]}: ${HOW[$i]:-not started}: ${WORDS[$i]}"
        fi
    done
    exit 143
}
trap on_signal TERM INT
PIDS=("" ""); ON=("" ""); NEXT=0
for k in 0 1; do
    if [ "$NEXT" -lt "$NA" ]; then start_arm "$k" "$NEXT"; NEXT=$((NEXT + 1)); fi
done
while [ -n "${PIDS[0]}${PIDS[1]}" ]; do
    wait -n 2> /dev/null || true                      # an arm ended (or a spurious return: the files decide)
    seen=0
    for k in 0 1; do
        i=${ON[$k]}
        [ -n "${PIDS[$k]}" ] && [ -f "$RCD/$i" ] || continue
        wait "${PIDS[$k]}" 2> /dev/null || true        # reap it
        RCS[$i]=$(cat "$RCD/$i"); HOW[$i]="${HOW[$i]}, ended $(date -Is)"; PIDS[$k]=""; ON[$k]=""; seen=1
        echo "=== arm ${TAGS[$i]} ended with exit ${RCS[$i]} on half $k ($(date -Is)) ==="
        while [ "$NEXT" -lt "$NA" ]; do
            # a queued arm only with QUEUE_MIN left (its start-up, the anchor's count and one round)
            left=$(python bfs/slurm_remaining.py --reserve 0 2> /dev/null || echo 0)
            if [ "$left" = 0 ] || [ "$left" -ge "$QUEUE_MIN" ]; then
                start_arm "$k" "$NEXT"; NEXT=$((NEXT + 1)); break
            fi
            echo "*** arm ${TAGS[$NEXT]} NOT started: ${left} s left in the job, under QUEUE_MIN=$QUEUE_MIN"
            HOW[$NEXT]="not started (${left} s left)"; NEXT=$((NEXT + 1))
        done
    done
    [ "$seen" = 1 ] || sleep 5
done
rc=0
for ((i = 0; i < NA; i++)); do
    if [ -z "${RCS[$i]}" ]; then
        echo "=== arm ${TAGS[$i]}: ${HOW[$i]}: ${WORDS[$i]}"
        continue
    fi
    [ "${RCS[$i]}" = 0 ] || rc=${RCS[$i]}
    echo "=== arm ${TAGS[$i]} ended with exit ${RCS[$i]} (${HOW[$i]}); its last lines (${LOGS[$i]}):"
    tail -4 "${LOGS[$i]}" | sed 's/^/    /'
done
exit $rc
