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
#   ORACLE    the inner loop's model: reduced (default; the projected model) or lattice (2026-10-06:
#             the FULL model on the MPI ranks, its factors cached on a lattice of frequencies and
#             line shifts, lattice.py, BFS.md 5)
#   LATTICE_CAP  with ORACLE=lattice: plant factorisations kept per rank (default 24; 133 MB each on
#             the test plant, ~1.05 GB on the full mesh -- size the ranks per node accordingly)
#   SHRINK    with ORACLE=lattice: 1 = trust radius 2 alpha |d| after a partial step
#             (--sqp-partial-shrink on; default 0 since 20261008c)
#   LATTICE_K with ORACLE=lattice: eigenvalues per disk centre of the line, each with its
#             stability row (--lattice-k; default 1)
#   STRIP     with ORACLE=lattice: 0 = no strip (--lattice-strip off; default 1)
#   RETIRE    with ORACLE=lattice: FRAC:ROUNDS, members of Omega below FRAC x gamma for ROUNDS
#             rounds retire (--omega-retire, e.g. 0.95:3); unset = Omega only grows
#
# The flags are the nominal SISO set of the GL production (STAGE6, the K+B+C
# runs: SQP inner solve, reduced oracle, slope-climb certificate, progress
# stop), plus the BFS ones: --plant bfs, MUMPS, gamma certified on [0, 10]
# (option A; --bfs-sampled N for the sampled value), the interior margin, the
# nearest-eigenvalue disks with k = 4 (the even pencil's pairs), and the certified
# prefix of a step the continuation gate refuses (--gate-prefix 1).
# --nproc 1: the full-model pool is the MPI ranks (bfs/mpipool.py), never a
# fork pool, and the GL |Omega|-constraint pool (parallel.py) is not used.
# --row-workers mpi: the reduced model's rows (the SQP's |Omega| rows, the peak
# follower, the fine sweeps) are shared by the same ranks while they would
# otherwise wait, and so are the near-line eigenpairs' tracking iterations
# (reduced._MPIRowPool; serial when there is no MPI pool).
# --sqp-pen adaptive: the SQP's merit penalty and elastic slack costs are an
# exact penalty -- steered up while a step gives up a linearised stability
# constraint, and kept above the stability multipliers; the fixed 100 of GL let
# the SQP trade the compensator's margin for sigma ~ 750 (BFS.md 5, 8).
# --sqp-row-res-tol 1e-5 (GL nominal: 1e-3): the reduced closed-loop pencil of
# this strongly non-normal flow has SPURIOUS Ritz values right of the line
# (+0.03 to +0.11 at the central 2x2 result, certified alpha -0.0723) with full
# residuals 4e-4 to 2e-3, which the 1e-3 filter let through as stability rows;
# the true near-line eigenvalues, whose eigenvectors the basis holds, have
# 1e-10 to 1e-15 (BFS.md 8).
# --oracle-union new: the union Galerkin step projects W's new columns only
# (same span, the basis equal to rounding; STAGE6 6.30) -- the serial part of
# the enrichment.
set -eo pipefail
out=$1; init=$2; shift 2
[ -n "$out" ] && [ -n "$init" ] || { echo "usage: run_bfs.sh OUTDIR INIT [flags]"; exit 2; }
out=$(realpath -m "$out"); init=$(realpath -m "$init")
mkdir -p "$out"
cd "$(dirname "$0")/.."
STAB=()
[ "${STABILISE:-1}" = "1" ] && STAB=(--stabilise)
ORC=(--oracle reduced --oracle-inputs on --row-workers mpi --oracle-union new)
if [ "${ORACLE:-reduced}" = "lattice" ]; then
    # SHRINK=1: the trust radius shrinks to 2 alpha |d| after a partial step
    # (sqp_inner.PARTIAL_SHRINK).  Off by default since 20261008c: on in 20261008a,
    # the four cluster runs stalled (code 6, "stalled solve") from round 3-6 on at
    # values the October 6 code went well below (BFS.md 8)
    ORC=(--oracle lattice --lattice-cap "${LATTICE_CAP:-24}" --sqp-partial-shrink "$([ "${SHRINK:-0}" = 1 ] && echo on || echo off)")
    if [ -n "${RETIRE:-}" ]; then ORC+=(--omega-retire "$RETIRE"); fi
    # LATTICE_K=2: the two nearest eigenvalues per disk centre, each with its
    # stability row (default 1; an eigenvalue behind another, seen from the
    # line, then gets a row too)
    ORC+=(--lattice-k "${LATTICE_K:-1}")
    # STRIP=0: no strip -- the stability rows are those of the eigenvalues the
    # line's disks see (as before 20261009a); the A/B of whether the strip pays
    ORC+=(--lattice-strip "$([ "${STRIP:-1}" = 0 ] && echo off || echo on)")
elif [ "${ORACLE:-reduced}" != "reduced" ]; then
    echo "run_bfs.sh: ORACLE must be reduced or lattice (got ${ORACLE})"; exit 2
fi
MIMOF=()
if [ "${M:-1}" != 1 ] || [ "${P:-1}" != 1 ]; then
    MIMOF=(--n-act "$M" --n-sens "$P" --ctrl-struct "${STRUCT:-central}" --mimo-start "$init")
fi
exec ${LAUNCH} ${PY:-python} -u run_case_epi.py --plant bfs --ops "${OPS:-bfs/data/ops_fx.npz}" --lu-order mumps \
    --init "$init" --cold --case K+B+C --r "${R:-5}" --eps 0.05 --start 0 --nproc 1 --cert_nseg "${CERT_NSEG:-16}" --cert-split "${CERT_SPLIT:-23}" --wlo 0 --whi 10 \
    --outer "${OUTER:-40}" --deadline "${DEADLINE:-0}" --wall "${WALL:-54000}" "${STAB[@]}" \
    --inner-budget iters --inner-maxit 10 --inner-maxit-adapt --select-cons on --ctrl-imcap off \
    --cont-anchor defer --sigma2 on "${ORC[@]}" --ctrl-line eps --cert-sweep 0 \
    --gate-prefix 1 --post 1 --sqp-pen adaptive --cert-disk strict --cert-near-k 4 --cert-climb slope --wind-ctrl exact --cert-rtol 1e-6 --alpha-argmax on \
    --enroll-rel 0.05 --peak-follow on --row-screen 0.05 --converge --inner sqp --sqp-carry --sqp-trust 0.01 \
    --sqp-ctrl block --sqp-row-res-tol 1e-5 --sqp-refine 1e-4 --sqp-clblock on --sigma2-active-gap 0.15 \
    --follow-scan 0 --stop-progress 3:0.0015 --wind-accept off --dense_check 0 --outdir "$out" "${MIMOF[@]}" "$@"
