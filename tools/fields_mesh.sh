#!/usr/bin/env bash
# Mesh independence of K+FB+FC (user, 2026-09-28; STAGE5 32).
#   tools/fields_mesh.sh <case>      one run per call (sequential by design)
# cases:
#   R1   unconstrained, ne=200, 160 Laplacian modes, from the 80-mode ne=100 optimum transferred to ne=200
#   S1   Sobolev shape row (RMS wavenumber <= KMAX, order ORD), ne=100, 80 modes, from the smooth start
#   S2   same, ne=200, 160 modes, from the frozen ne=200 design
#   S3   same, ne=400, 160 modes, from the frozen ne=200 design transferred to ne=400 (results_ne400)
#   A1   4 Gabor atoms (width >= 1, |kappa| <= 0.7071) + the H^1 row (KMAX 1), ne=100 (input basis 80 modes)
#   A2   same, ne=200 (input basis 160 modes);  A3  same, ne=400 (input basis 160 modes)
# env: KMAX (default 1.0), ORD (default 1), OUTER (default 20), TAG (suffix of the output directory),
#      NF (number of Laplacian modes; default 80 at ne=100, 160 otherwise)
set -uo pipefail; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
KMAX=${KMAX:-1.0}; ORD=${ORD:-1}; OUTER=${OUTER:-20}; TAG=${TAG:-}
COMMON="--wall 36000 --outer $OUTER --inner-budget iters --inner-maxit 10 --inner-maxit-adapt \
 --select-cons on --ctrl-imcap off --cont-anchor defer --sigma2 on --oracle reduced --oracle-inputs on \
 --ctrl-line eps --cert-sweep 0 --cert-disk strict --wind-ctrl exact --cert-rtol 1e-6 --alpha-argmax on --enroll-rel 0.05 --peak-follow on \
 --row-screen 0.05 --converge --inner sqp --sqp-carry --sqp-trust 0.01 --sqp-ctrl block --sqp-row-res-tol 1e-3 --sqp-refine 1e-4 --sqp-clblock on --cont-accept-winding --sigma2-active-gap 0.15 --follow-scan 0 --stop-progress 3:0.0015"
BASIS="--field-basis laplace --field-lo -38.9 --field-hi 39.2"
SOB="--field-kmax $KMAX --field-order $ORD"
# ne=400: no init/seed catalogue for that mesh (prepare.py is a dense step, >10 min); the controller is
# mesh-independent and comes from the resumed record, so the ne=200 files supply only the order (8)
INIT400="--init init_ne200.npz --seed_file seeds_ne200.npz"
ATOMS="--field-atoms 4 --atom-smin 1.0 --atom-kmax 0.7071 --field-kmax 1.0 --field-order 1"
case "$1" in
  R1) ne=200; args="$BASIS --field-n 160"; start=results_ne200_fb/K_FB_FC_s0_ne200.npz; out=results_fb_refine_n200_N160 ;;
  S1) ne=100; args="$BASIS --field-n ${NF:-80} $SOB"; start=results_ne100/K_FB_FC_s0_ne100.npz; out=results_fb_sob_n100$TAG ;;
  S2) ne=200; args="$BASIS --field-n ${NF:-160} $SOB"; start=results/K_FB_FC_s0_ne200.npz; out=results_fb_sob_n200$TAG ;;
  S3) ne=400; args="$BASIS --field-n ${NF:-160} $SOB"; start=results_ne400/K_FB_FC_s0_ne400.npz; out=results_fb_sob_n400$TAG; args="$args $INIT400" ;;
  A1) ne=100; args="$BASIS --field-n ${NF:-80} $ATOMS"; start=results_ne100/K_FB_FC_s0_ne100.npz; out=results_fb_atoms_n100$TAG ;;
  A2) ne=200; args="$BASIS --field-n ${NF:-160} $ATOMS"; start=results/K_FB_FC_s0_ne200.npz; out=results_fb_atoms_n200$TAG ;;
  A3) ne=400; args="$BASIS --field-n ${NF:-160} $ATOMS"; start=results_ne400/K_FB_FC_s0_ne400.npz; out=results_fb_atoms_n400$TAG; args="$args $INIT400" ;;
  *) echo "unknown case $1"; exit 2 ;;
esac
mkdir -p $out
if [ ! -f $out/K_FB_FC_s0_ne$ne.json ]; then
  echo "=== launch $1 $(date +%s) KMAX=$KMAX ORD=$ORD" >> $out/run.log
  python run_case_epi.py --eps 0.05 --case K+FB+FC --start 0 --ne $ne $COMMON --resume $start $args \
    --outdir $out --dump $out/dump_$(date +%s) >> $out/run.log 2>&1
fi
echo "=== $1 done"
