#!/usr/bin/env bash
# modes_vs_atoms.sh -- why did 4 atoms (3.876296) beat 160 Laplacian modes with the
# H^2 row (3.911740)?  (STAGE6 6.36, user 2026-10-01 16:07: representation, convergence
# or local minimum?)  The 4-atom design lies inside the H^2 problem's admissible set
# (160-mode projection error 5e-6, H^2 measure 0.60 against the bound 1), so S2 did
# not reach its own problem's optimum.  Two runs of S2's problem (ne=200, 160 modes,
# H^2 row kmax = 1, the STAGE5 loop of tools/fields_mesh.sh), at most 15 rounds:
#   from_atoms   from the 4-atom design A2'' (its fields projected onto the modes)
#   s2_cont      S2 continued from its own record
set -uo pipefail; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COMMON="--wall 3600 --outer 15 --inner-budget iters --inner-maxit 10 --inner-maxit-adapt \
 --select-cons on --ctrl-imcap off --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on \
 --ctrl-line eps --cert-sweep 0 --cert-disk strict --wind-ctrl exact --cert-rtol 1e-6 --alpha-argmax on --enroll-rel 0.05 --peak-follow on \
 --row-screen 0.05 --converge --inner sqp --sqp-carry --sqp-trust 0.01 --sqp-ctrl block --sqp-row-res-tol 1e-3 --sqp-refine 1e-4 --sqp-clblock on --cont-accept-winding --sigma2-active-gap 0.15 --follow-scan 0 --stop-progress 3:0.0005"
ARGS="--field-basis laplace --field-lo -38.9 --field-hi 39.2 --field-n 160 --field-kmax 1.0 --field-order 2"
run() {
  out=results_modes_vs_atoms/$1; mkdir -p $out
  python run_case_epi.py --eps 0.05 --case K+FB+FC --start 0 --ne 200 $COMMON --resume $2 $ARGS --outdir $out > $out/run.log 2>&1
  echo "$1: $(tail -1 $out/run.log | cut -c1-150)"
}
run from_atoms results_fb_atoms_n200_long/K_FB_FC_s0_ne200.npz &
run s2_cont results_fb_sob_n200_h2/K_FB_FC_s0_ne200.npz &
wait
