#!/bin/bash
# atom_count_runs.sh -- optimised K+FB+FC gamma against the number of atoms per
# field (STAGE6 6.34; user 2026-10-01: no extra constraints on the atoms first).
#
# Every run: ne=200, the nominal loop (tools/n100_kbc.sh via lattice_check.
# loop_flags), N Gabor atoms of width >= 0.4 and |wavenumber| <= 0.7071 per
# field, the atoms' tangent space as the reduced model's input basis
# (--field-input tangent, 6.33).  Same start for every N: the H^2 ne=200 record
# (results_fb_sob_n200_h2, 3.911740) -- its controller, and its fields fitted
# by N atoms (the fit errors of 6.32; the start can be far above the record
# with the controller held -- 81.9 at N = 4 -- so --resume-guard 0).  Progress stop 0.05 % over 3 rounds
# (the nominal 0.15 % stopped atom runs early, STAGE5 32.6), at most 60 rounds.
# --dense-audit: every candidate's dense spectrum in the log (check only; the
# first set of runs, results_atom_count/runs_v1_escape, had an escape at N = 2).
#
#   bash tools/atom_count_runs.sh 8 6 4 2        (two at a time, largest first)
#   CONT=1 bash tools/atom_count_runs.sh 4       (continue runs/N4 from its record
#                                                 into runs/N4_c: same flags)
cd "$(dirname "$0")/.."
OUT=results_atom_count/runs
START=results_fb_sob_n200_h2/K_FB_FC_s0_ne200.npz
LOOP=$(python3 -c "import sys; sys.path[:0]=['tools','.']; from lattice_check import loop_flags; print(' '.join(loop_flags(200, 60)))" \
       | sed 's|--stop-progress 3:0.0015|--stop-progress 3:0.0005|; s|--wall 36000|--wall 5400|')
FIELDS="--field-basis laplace --field-lo -38.9 --field-hi 39.2 --field-n 160 --atom-smin 0.4 --atom-kmax 0.7071 \
--field-kmax 1.8 --field-order 1 --field-input tangent --resume-guard 0 --dense-audit"
run() {
    N=$1; d=$OUT/N$N; src=$START
    if [ -n "$CONT" ]; then src=$OUT/N$N/K_FB_FC_s0_ne200.npz; d=$OUT/N${N}_c; fi
    mkdir -p $d
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u run_case_epi.py --case K+FB+FC --start 0 \
        --eps 0.05 $LOOP $FIELDS --field-atoms $N --resume $src --outdir $d > $d/run.log 2>&1
    echo "N=$N: $(tail -1 $d/run.log | cut -c1-160)"
}
export -f run; export OUT START LOOP FIELDS CONT
printf '%s\n' "$@" | xargs -P 2 -I{} bash -c 'run {}'
