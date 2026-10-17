# lib.sh -- shell functions shared by the slurm scripts (source bfs/lib.sh; BFS.md 7)
#
#   take_lock DIR [WHAT]   wait for, then take, the lock directory DIR (mkdir is
#                          atomic on the shared file system, across nodes); the
#                          owner's job id is written into it.  A lock whose owner
#                          is no longer in the queue is taken over (renamed away:
#                          one winner); without squeue, one older than 3 h is
#   drop_lock DIR          release it (only our own)
#   job_alive ID           true while the job is queued or running
#   strip_refused WORDS... false, with the reason, when STRIP (the environment's,
#                          or a STRIP=... among WORDS) asks for the strip
#   lattice_async_flag     the --lattice-async word of ASYNC (refused: other values)
#
# 2026-10-08: jobs 765560-765562 built the same files at once and corrupted
# them; 766730: flock was node-local on sator.  20261010j: the owner check
# replaces the age-only takeover, which could break the lock of a live job.

job_alive() {
    [ -n "$1" ] || return 1
    command -v squeue > /dev/null 2>&1 || return 2
    [ -n "$(squeue -h -j "$1" -o %i 2>/dev/null)" ]
}

take_lock() {
    local d=$1 what=${2:-files} me=${SLURM_JOB_ID:-pid$$} owner rc
    until mkdir "$d" 2>/dev/null; do
        owner=$(cat "$d/owner" 2>/dev/null || true)
        rc=0; job_alive "$owner" || rc=$?
        if [ "$rc" = 1 ] && [ -n "$owner" ] && [ "${owner#pid}" = "$owner" ]; then
            if mv "$d" "$d.stale.$$" 2>/dev/null; then
                echo "=== lock $d of job $owner (no longer in the queue) taken over ==="; rm -rf "$d.stale.$$"
            fi
            continue
        fi
        if [ -n "$(find "$d" -maxdepth 0 -mmin +180 2>/dev/null)" ] && { [ "$rc" = 2 ] || [ -z "$owner" ]; }; then
            if mv "$d" "$d.stale.$$" 2>/dev/null; then
                echo "=== stale lock $d (older than 3 h, owner unknown) taken over ==="; rm -rf "$d.stale.$$"
            fi
            continue
        fi
        echo "=== waiting for job ${owner:-?} to finish the $what ($(date -Is)) ==="; sleep 30
    done
    echo "$me" > "$d/owner"
}

drop_lock() {
    local d=$1 me=${SLURM_JOB_ID:-pid$$}
    [ "$(cat "$d/owner" 2>/dev/null)" = "$me" ] && rm -rf "$d"
    return 0
}

# run_lock OUT: one driver per run directory (20261010j: duplicate submissions
# happen -- 770564-770623 -- and two drivers would both write the checkpoint).
# Refuses while another live job holds it; a dead owner's lock is taken over.
run_lock() {
    local d="$1/.job.lockd" owner rc
    mkdir -p "$1"
    if ! mkdir "$d" 2>/dev/null; then
        owner=$(cat "$d/owner" 2>/dev/null || true)
        rc=0; job_alive "$owner" || rc=$?
        if [ "$rc" = 0 ]; then
            echo "*** $1 is being run by job $owner (still in the queue): not starting a second driver"; return 1
        fi
        mv "$d" "$d.stale.$$" 2>/dev/null && rm -rf "$d.stale.$$"
        mkdir "$d" || { echo "*** could not take the run lock $d"; return 1; }
        echo "=== run lock of job ${owner:-?} (no longer in the queue) taken over ==="
    fi
    echo "${SLURM_JOB_ID:-pid$$}" > "$d/owner"
}

# strip_refused [WORDS...]: the strip arm (--lattice-strip on, the default until
# 20261010n) was REMOVED on 2026-10-16 (user, after the 20261010n A/B: it bought
# no stability, cost ~45 % more evaluations a round, and only the no-strip path
# runs the level set beside the next iteration; BFS.md 8).  STRIP=1 (or any value
# but 0) is refused before anything runs; STRIP=0, what the code now always does,
# is accepted and noted.  WORDS: an arm's VAR=value words (pair.slurm).
strip_refused() {
    local v=${STRIP:-} w
    for w in "$@"; do case "$w" in STRIP=*) v=${w#STRIP=} ;; esac; done
    case "$v" in
        '') return 0 ;;
        0)  echo "=== STRIP=0: the only path since 2026-10-16 (no strip; the variable is no longer needed) ==="; return 0 ;;
        *)  echo "*** STRIP=$v: the strip arm was removed on 2026-10-16 (BFS.md 8: the 20261010n A/B); the lattice runs without"
            echo "    the strip, as STRIP=0 did.  Unset STRIP.  A run made with the strip continues without it (RESUME=1 or"
            echo "    CONTINUE=1 OUT=<its directory>); an old *_nostrip* directory continues with its explicit OUT="
            return 1 ;;
    esac
}

# lattice_async_flag: --lattice-async from ASYNC (2026-10-16: on by default, the
# synchronous call with ASYNC=0 for measurements; directories get _sync)
lattice_async_flag() {
    case "${ASYNC:-1}" in
        1) echo on ;;
        0) echo off ;;
        *) echo "*** ASYNC=$ASYNC: 1 (default, the level set beside the next SQP iteration) or 0 (synchronous)" >&2; return 1 ;;
    esac
}
