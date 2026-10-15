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
#   ARM_A="OUT=testplant_bfs/ab_c151_strip STRIP=1" ARM_B="OUT=testplant_bfs/ab_c151_nostrip STRIP=0" \
#   CONTINUE=1 ORACLE=lattice M=2 P=2 STRUCT=central OUTER=300 sbatch --job-name=ab_c151 bfs/slurm/pair.slurm
#
# Two runs of bfs/slurm/testplant.slurm in ONE job on one 96-core node, each on
# its own half of the cores (its own NUMA side) with half of the ranks.  The
# variables common to both arms are set as for testplant.slurm; ARM_A and
# ARM_B hold each arm's own (VAR=value words, OUT required and distinct).
# Each arm writes logs/bfs_testplant_<job>_<OUT's name>.log, a log of the same
# form as a testplant job's; this job's own log says where each arm ran and
# how it ended.
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
LAUNCHER=${LAUNCHER:-mpiexec}
export HYDRA_LAUNCHER=fork I_MPI_HYDRA_BOOTSTRAP=fork
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is): two arms, MPI launcher $($LAUNCHER --version 2>&1 | grep -m1 -E '^ *Version:' | tr -s ' ' || echo '?') ==="

# the arms' own variables: VAR=value words only (no spaces inside a value)
parse_arm() {
    local name=$1 words=$2 w
    [ -n "$words" ] || { echo "*** $name is empty: ARM_A=\"OUT=... STRIP=...\" ARM_B=\"OUT=... ...\""; exit 1; }
    for w in $words; do
        [[ "$w" =~ ^[A-Z_][A-Z0-9_]*=[^[:space:]]*$ ]] || { echo "*** $name: '$w' is not VAR=value"; exit 1; }
    done
    [[ " $words " == *" OUT="* ]] || { echo "*** $name has no OUT=: each arm needs its own run directory"; exit 1; }
}
parse_arm ARM_A "${ARM_A:-}"
parse_arm ARM_B "${ARM_B:-}"
read -ra A <<< "$ARM_A"
read -ra B <<< "$ARM_B"
out_of() { local w; for w in "$@"; do case "$w" in OUT=*) echo "${w#OUT=}" ;; esac; done | tail -1; }
OA=$(out_of "${A[@]}"); OB=$(out_of "${B[@]}")
[ "$(realpath -m "$OA")" != "$(realpath -m "$OB")" ] || { echo "*** both arms have OUT=$OA"; exit 1; }

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

# the arms
pids=(); logs=()
for k in 0 1; do
    if [ $k = 0 ]; then words=("${A[@]}"); out=$OA; tag=a; else words=("${B[@]}"); out=$OB; tag=b; fi
    log=logs/bfs_testplant_${SLURM_JOB_ID:-none}_$(basename "$out").log
    {
        echo "=== pair job ${SLURM_JOB_ID:-none} arm $tag on $(hostname): CPUs ${H[$k]}, $HALF ranks, ${words[*]} ==="
        exec env "${words[@]}" SLURM_NTASKS=$HALF PAIR_ARM=$tag taskset -c "${H[$k]}" bash bfs/slurm/testplant.slurm
    } > "$log" 2>&1 &
    pids+=($!); logs+=("$log")
    echo "=== arm $tag: ${words[*]} on CPUs ${H[$k]} -> $out (log $log, pid ${pids[$k]}) ==="
done
rc=0; rcs=()
for k in 0 1; do
    r=0; wait "${pids[$k]}" || r=$?
    rcs+=("$r"); [ "$r" = 0 ] || rc=$r
done
for k in 0 1; do
    echo "=== arm $([ $k = 0 ] && echo a || echo b) ended with exit ${rcs[$k]} ($(date -Is)); its last lines (${logs[$k]}):"
    tail -4 "${logs[$k]}" | sed 's/^/    /'
done
exit $rc
