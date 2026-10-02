#!/bin/bash
# run_bfs.sh OUTDIR INIT [extra run_case_epi.py flags]
#   -- one backward-facing-step co-design run (BFS.md 5), the flags in ONE place
#
# Environment:
#   PY        python (default: python)
#   LAUNCH    MPI launcher prefix, e.g. "mpiexec -n 96" (default: none = serial)
#   OPS       operator file (default bfs/data/ops_fx.npz)
#   OUTER     rounds (default 40)
#   R         controller order (default 5; the start file must have the same order; MIMO: the total)
#   M, P      actuators, sensors (default 1, 1; MIMO when either is > 1: INIT is then the MIMO start,
#             bfs/hifoo_init.py --m --p, and the record is K_B_C_s0_bfs_mimo<M>x<P>_<STRUCT>)
#   STRUCT    MIMO compensator: central (one m x p) or decentral (m = p SISO loops); default central
#   DEADLINE  seconds after which no new round starts (default 0 = off)
#   WALL      the driver's --wall (inner-solve clock budgets; default 54000)
#   CERT_NSEG line-certificate segments over the MPI pool (default 16; serial runs: 1)
#   CERT_SPLIT first generation of the gamma certificate's disk sweep (default 23 = the workers)
#   STABILISE 1 = HIFOO's abscissa-descent phase when the start is infeasible (default 1)
#
# The flags are the nominal SISO set of the GL production (STAGE6, the K+B+C
# runs: SQP inner solve, reduced oracle, slope-climb certificate, progress
# stop), plus the BFS ones: --plant bfs, MUMPS, gamma certified on [0, 10]
# (option A; --bfs-sampled N for the sampled value), the interior margin, the
# nearest-eigenvalue disks with k = 4 (the even pencil's pairs), and the certified
# prefix of a step the continuation gate refuses (--gate-prefix 1).
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
MIMOF=()
if [ "${M:-1}" != 1 ] || [ "${P:-1}" != 1 ]; then
    MIMOF=(--n-act "$M" --n-sens "$P" --ctrl-struct "${STRUCT:-central}" --mimo-start "$init")
fi
exec ${LAUNCH} ${PY:-python} -u run_case_epi.py --plant bfs --ops "${OPS:-bfs/data/ops_fx.npz}" --lu-order mumps \
    --init "$init" --cold --case K+B+C --r "${R:-5}" --eps 0.05 --start 0 --nproc 1 --cert_nseg "${CERT_NSEG:-16}" --cert-split "${CERT_SPLIT:-23}" --wlo 0 --whi 10 \
    --outer "${OUTER:-40}" --deadline "${DEADLINE:-0}" --wall "${WALL:-54000}" "${STAB[@]}" \
    --inner-budget iters --inner-maxit 10 --inner-maxit-adapt --select-cons on --ctrl-imcap off \
    --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on --ctrl-line eps --cert-sweep 0 \
    --gate-prefix 1 --post 1 --cert-disk strict --cert-near-k 4 --cert-climb slope --wind-ctrl exact --cert-rtol 1e-6 --alpha-argmax on \
    --enroll-rel 0.05 --peak-follow on --row-screen 0.05 --converge --inner sqp --sqp-carry --sqp-trust 0.01 \
    --sqp-ctrl block --sqp-row-res-tol 1e-3 --sqp-refine 1e-4 --sqp-clblock on --sigma2-active-gap 0.15 \
    --follow-scan 0 --stop-progress 3:0.0015 --wind-accept off --dense_check 0 --outdir "$out" "${MIMOF[@]}" "$@"
