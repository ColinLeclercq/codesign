#!/bin/bash
#SBATCH --comment=36973004F
#SBATCH --job-name=bfs-stat
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=450G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bfs_stat_%j.log
#
# tools/bfs_stationarity.py on the MPI ranks: how far each record is from
# Clarke stationarity (SISO, MIMO decentral, MIMO central), the first-order
# decrease still available, and a certified lower design when there is one
# (LESSONS.md 21; the tool's docstring says what each number means).
#
#   RECS="testplant_bfs/r_c124/run/K_B_C_s0_bfs_mimo2x2_central.npz testplant_bfs/r_d370/run/K_B_C_s0_bfs_mimo2x2_decentral.npz" \
#       sbatch tools/stationarity.slurm
#   RECS="..." DESCENT=on COMPLETE=on sbatch tools/stationarity.slurm   (no certificate; the level-set completeness of the pieces)
#
# One job for several records, one after the other, on the whole node: sator
# runs one job of a user per node (BFS.md 8), so records are queued here, not
# as separate jobs.  Each record's report goes to <record>_stationarity.log and
# .json next to it (bfs/report.sh copies the run directory's *.json).
# A record is a finished record (.npz) or a checkpoint (_ckpt.npz).  The tool
# reads the structure (SISO / MIMO, central / decentral) from the record.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ./env.sh
python bfs/version_check.py || { echo "*** mixed code versions (bfs/version_check.py): stop"; exit 1; }
LAUNCHER=${LAUNCHER:-mpiexec}
[ -n "${RECS:-}" ] || { echo "stationarity.slurm: set RECS=\"record.npz [record.npz ...]\""; exit 2; }
OPS=${OPS:-$(python bfs/plant_id.py --dir coarse)/ops_coarse.npz}
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)  $(git rev-parse --short HEAD 2>/dev/null || echo -)  ops $OPS ==="
rc=0
for REC in $RECS; do
    [ -f "$REC" ] || { echo "*** $REC: no such record"; rc=1; continue; }
    base=${REC%.npz}
    echo "=== $(date -Is) $REC -> ${base}_stationarity.log ==="
    if timeout "${PER_REC:-5400}" $LAUNCHER -n "${SLURM_NTASKS:-96}" python -u tools/bfs_stationarity.py "$REC" --ops "$OPS" \
            --eps "${EPS:-1e-6,1e-5,1e-4,1e-3,1e-2}" --fd "${FD:-6}" --descent "${DESCENT:-certify}" \
            --complete "${COMPLETE:-off}" --json "${base}_stationarity.json" > "${base}_stationarity.log" 2>&1; then
        grep -E "^(verdict|VERDICT)|stationary to|NOT STATIONARY|certified" "${base}_stationarity.log" | tail -4 | sed 's/^/    /'
    else
        r=$?; rc=$r
        echo "*** $REC: the tool ended with exit $r (timeout ${PER_REC:-5400} s = 124); its last lines:"; tail -5 "${base}_stationarity.log" | sed 's/^/    /'
    fi
done
echo "=== done $(date -Is) ==="
exit $rc
