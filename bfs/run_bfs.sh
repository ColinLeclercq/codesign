#!/bin/bash
# run_bfs.sh OUTDIR INIT [extra run_case_epi.py flags]
#   -- one backward-facing-step co-design run (BFS.md 5), the flags in ONE place
#
# Environment:
#   PY        python (default: python)
#   LAUNCH    MPI launcher prefix, e.g. "mpiexec -n 96" (default: none = serial)
#   OPS       operator file (default bfs/data/ops_fx.npz)
#   OUTER     rounds (default 40)
#   DEADLINE  seconds after which no new round starts (default 0 = off)
#   WALL      the driver's --wall (inner-solve clock budgets; default 54000)
#   CERT_NSEG line-certificate segments over the MPI pool (default 16; serial runs: 1)
#   STABILISE 1 = HIFOO's abscissa-descent phase when the start is infeasible (default 1)
#
# The flags are the nominal SISO set of the GL production (STAGE6, the K+B+C
# runs: SQP inner solve, reduced oracle, slope-climb certificate, progress
# stop), plus the BFS ones: --plant bfs, MUMPS, the sampled gamma
# (--bfs-sampled), the band [0, --bfs-band], the interior margin.
# --nproc 1: the full-model pool is the MPI ranks (bfs/mpipool.py), never a
# fork pool, and the GL |Omega|-constraint pool (parallel.py) is not used.
set -eo pipefail
out=$1; init=$2; shift 2
[ -n "$out" ] && [ -n "$init" ] || { echo "usage: run_bfs.sh OUTDIR INIT [flags]"; exit 2; }
out=$(realpath -m "$out"); init=$(realpath -m "$init")
mkdir -p "$out"
cd "$(dirname "$0")/.."
STAB=()
[ "${STABILISE:-1}" = "1" ] && STAB=(--stabilise)
exec ${LAUNCH} ${PY:-python} -u run_case_epi.py --plant bfs --ops "${OPS:-bfs/data/ops_fx.npz}" --lu-order mumps \
    --init "$init" --cold --case K+B+C --r 5 --eps 0.05 --start 0 --nproc 1 --cert_nseg "${CERT_NSEG:-16}" --wlo 0 --whi 8 \
    --outer "${OUTER:-40}" --deadline "${DEADLINE:-0}" --wall "${WALL:-54000}" "${STAB[@]}" \
    --inner-budget iters --inner-maxit 10 --inner-maxit-adapt --select-cons on --ctrl-imcap off \
    --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on --ctrl-line eps --cert-sweep 0 \
    --cert-disk strict --cert-near-k 2 --cert-climb slope --wind-ctrl exact --cert-rtol 1e-6 --alpha-argmax on \
    --enroll-rel 0.05 --peak-follow on --row-screen 0.05 --converge --inner sqp --sqp-carry --sqp-trust 0.01 \
    --sqp-ctrl block --sqp-row-res-tol 1e-3 --sqp-refine 1e-4 --sqp-clblock on --sigma2-active-gap 0.15 \
    --follow-scan 0 --stop-progress 3:0.0015 --wind-accept off --dense_check 0 --outdir "$out" "$@"
