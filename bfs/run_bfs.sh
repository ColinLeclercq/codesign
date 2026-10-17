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
#   DEADLINE  the job's time limit in seconds (default 0 = off): the SQP stops on time inside
#             the round, no round starts without its reserve and a minimum inner solve
#             (--deadline-reserve, --deadline-min-inner; 20261010j stage B, was: 1.25 x the
#             longest round), --deadline-final kept for the final count, veto and save
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
#             (--sqp-partial-shrink on; default 0 since 20261008c; only with TRUST=legacy)
#   TRUST     the SQP's trust-radius rule (--sqp-trust-rule): ratio (default since 2026-10-17) or
#             legacy (the rule of every run before; directories get _leg)
#   LATTICE_K with ORACLE=lattice: eigenvalues per disk centre of the line, each with its
#             stability row (--lattice-k; default 1)
#   STRIP     refused (bfs/lib.sh strip_refused): the strip arm (--lattice-strip on) was removed on
#             2026-10-16 (BFS.md 8); STRIP=0, the only path, is accepted
#   ASYNC     with ORACLE=lattice: 0 = the missed-peak level set synchronous (--lattice-async off, for
#             measurements); default 1, beside the next SQP iteration (--lattice-async on, since 2026-10-16)
#   CROSS     with ORACLE=lattice: cut = trials across the closed loop's line halved (--lattice-cross cut;
#             default merit)
#   MODAL     0 = the tridiagonal compensator (--ctrl-modal off: a run made before 20261010e); default 1,
#             the modal 2 x 2 blocks
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
# --cont-accept-winding (20261010j stage B, gate-refusal.md G1): a step the
# continuation gate refuses, line-certified and improving, is accepted when its
# all-frequency count right of -eps is a conclusive 0 (the GL default since
# STAGE5 22: round 6 at ne=100, 7.929704), the chain restarted there; an
# INCONCLUSIVE count or one > 0 refuses, as before.  Never passed here before
# (770633 lost its round 1, 430.89, to an uncounted refusal).  The BFS counts
# right of -eps take an absolute step floor (run_case_epi.WIND_HMIN_ABS).
# --wind-accept stays off: a count at EVERY acceptance (2-7 min on the test
# plant, 20-40 min on the full mesh, BFS.md 9) is the user's call; the save-time
# count right of -eps vetoes (the record falls back to the last accepted design
# whose count is 0; --veto-counts).
# --stop-progress 8:0.0015 (2026-10-16; 3:0.0015 before, for every case): LESSONS
# 1 ("use 8 rounds for MIMO and field cases": GL central stopped at 0.126 % over
# 3 rounds, then gained 2 % in 24 more), and the 20261010n central async arm
# stopped on 3:0.0015 at a design a sandbox probe shows is not stationary.  The
# same rule for SISO.
# --sqp-stat pred for SISO too (2026-10-16; MIMO had it through auto): auto =
# rel for SISO fired code 0 on a loose threshold -- the SISO records 554.97 and
# 598.11 stopped at stat 12.6 and 6.0 against thresholds 15.5 and 27.0 (gradient
# scales 1.55e5 and 2.7e5) while a certified linear-model step still gains 0.20 %
# (the sandbox probe of 2026-10-16, STAGE6 6.14's mechanism).  GL untouched: its
# scripts pass their own flags, and the driver's default stays auto.
# Driver flags after INIT come last and win (argparse): testplant.slurm's
# EXTRA_ARGS, an e2e test's --inner-maxit.
set -eo pipefail
out=$1; init=$2; shift 2
[ -n "$out" ] && [ -n "$init" ] || { echo "usage: run_bfs.sh OUTDIR INIT [flags]"; exit 2; }
out=$(realpath -m "$out"); init=$(realpath -m "$init")
mkdir -p "$out"
cd "$(dirname "$0")/.."
source bfs/lib.sh
strip_refused || exit 2
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
    # CROSS=cut: a trial across the closed loop's line from an admissible iterate
    # is halved (2026-10-06 to 20261010c); default merit (sqp_inner.LAT_CROSS)
    ORC+=(--lattice-cross "${CROSS:-merit}")
    # 20261010j stage C and its review: the base grid fixed at 48 centres whatever
    # the ranks, static task costs (placement by measured solves is opt-in: the
    # seconds made it differ from run to run), the row requests memoised for 2
    # designs, the duplicate stability rows counted but NOT merged (no effect shown
    # yet, strip-ab.md AB3) -- passed explicitly, so the configuration is the
    # script's, not the code defaults' (LESSONS 18 rule 3).  No knobs: an A/B of one
    # of them needs its own directory suffix first (testplant.slurm, codesign.slurm)
    ORC+=(--lattice-base 48 --lattice-costs static --lattice-memo 2 --lattice-row-merge 0)
    # the missed-peak level set beside the next SQP iteration (20261010j stage D), on
    # since 2026-10-16: measured in 20261010n, traces identical to the synchronous
    # arms' over 8 central and 11 decentral rounds, rounds 0.71-0.80 x their wall
    # (BFS.md 8).  ASYNC=0: the synchronous call, for measurements
    AF=$(lattice_async_flag) || exit 2
    ORC+=(--lattice-async "$AF")
elif [ "${ORACLE:-reduced}" != "reduced" ]; then
    echo "run_bfs.sh: ORACLE must be reduced or lattice (got ${ORACLE})"; exit 2
fi
MIMOF=()
if [ "${M:-1}" != 1 ] || [ "${P:-1}" != 1 ]; then
    MIMOF=(--n-act "$M" --n-sens "$P" --ctrl-struct "${STRUCT:-central}" --mimo-start "$init")
fi
# MODAL=0: the tridiagonal compensator (older records, a resume of a run made
# before 20261010e); default the modal 2 x 2 blocks (--ctrl-modal, BFS.md 5)
MODF=(--ctrl-modal "$([ "${MODAL:-1}" = 0 ] && echo off || echo on)")
# TRUST (2026-10-17): the trust radius by the ratio rule (sqp_inner.TRUST_RULE): the
# line search from alpha = 1, the radius 2 alpha |d| after a partial step, x 2 / x 1 /
# |d| / 2 by the ratio of the merit's decrease to the model's after a full one, a
# failed QP retried with bounded equilibration.  Under the legacy rule (TRUST=legacy,
# every run before) the radius stood 100-1000 x the accepted step on the BFS, 37-40
# of 40 QP steps on its boundary at alpha 2^-9 .. 2^-10 (20261010n, LESSONS 21)
case "${TRUST:-ratio}" in
    ratio|legacy) ;;
    *) echo "run_bfs.sh: TRUST must be ratio (default) or legacy (got ${TRUST})"; exit 2 ;;
esac
if [ "${TRUST:-ratio}" = ratio ] && [ "${SHRINK:-0}" = 1 ]; then
    echo "run_bfs.sh: SHRINK=1 (--sqp-partial-shrink on) is a legacy-rule arm: give TRUST=legacy with it"; exit 2
fi
TRF=(--sqp-trust-rule "${TRUST:-ratio}")
exec ${LAUNCH} ${PY:-python} -u run_case_epi.py --plant bfs --ops "${OPS:-bfs/data/ops_fx.npz}" --lu-order mumps \
    --init "$init" --cold --case K+B+C --r "${R:-5}" --eps 0.05 --start 0 --nproc 1 --cert_nseg "${CERT_NSEG:-16}" --cert-split "${CERT_SPLIT:-23}" --wlo 0 --whi 10 \
    --outer "${OUTER:-40}" --deadline "${DEADLINE:-0}" --wall "${WALL:-54000}" "${STAB[@]}" \
    --inner-budget iters --inner-maxit 10 --inner-maxit-adapt --select-cons on --ctrl-imcap off \
    --cont-anchor defer --sigma2 on "${ORC[@]}" --ctrl-line eps --cert-sweep 0 \
    --gate-prefix 1 --post 1 --sqp-pen adaptive --cert-disk strict --cert-near-k 4 --cert-climb slope --wind-ctrl exact --cert-rtol 1e-6 --alpha-argmax on \
    --enroll-rel 0.05 --peak-follow on --row-screen 0.05 --converge --inner sqp --sqp-carry --sqp-trust 0.01 \
    --sqp-ctrl block --sqp-row-res-tol 1e-5 --sqp-refine 1e-4 --sqp-clblock on --sigma2-active-gap 0.15 \
    --follow-scan 0 --stop-progress 8:0.0015 --sqp-stat pred --wind-accept off --cont-accept-winding --dense_check 0 --outdir "$out" "${MIMOF[@]}" "${MODF[@]}" "${TRF[@]}" "$@"
