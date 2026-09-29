#!/usr/bin/env bash
# MIMO (2 actuators, 2 sensors) at ne=100 in the production loop (user, 2026-09-29; STAGE6).
#   tools/mimo_n100.sh <case>
#   kbc_dec   K+B+C decentral from the SISO optimum, pair 2 at -+4 (tools/mimo_start.py)
#   kbc_cen   K+B+C central from the kbc_dec result
#   at_dec    KGC+FB+FC (4 Gabor atoms per field, static gain) decentral from the SISO atoms optimum
#   at_cen    same, central from the at_dec result
#   at_dec_p  KGC+FB+FC decentral from the PROTOTYPE's decentral atoms optimum (results_mimo/atoms_dec.npz, 2.387150)
# All MIMO cases stop on 8 rounds without 0.15 % progress, not 3.  Atoms: with
# the band taken from each candidate (STAGE6 6.6) candidates whose far
# resonance is new are refused while their witnesses join the working set, and
# the 3-round rule stopped at_dec at 3.2999 after two such rounds (STAGE6 6.7;
# the SISO field cases showed the same, STAGE5 32.6).  K+B+C central: sigma_1
# within 1 % of gamma over |w| <~ 16 with 28 local maxima -- the slow tail of an
# equalised optimum; the 3-round rule stopped it at 3.792962 while 24 more
# rounds reached 3.717885 (STAGE6 6.15).  env EXTRA: extra driver flags.
set -uo pipefail; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOOP=$(sed -n '/^LOOP=/,/stop-progress [0-9:.]*"/p' tools/n100_kbc.sh | tr -d '\\\n' | sed 's/^LOOP="//; s/"$//' | sed -E 's/--outer [0-9]+/--outer 60/')
EXTRA="--stop-progress 8:0.0015 ${EXTRA:-}"
ATOMS="--field-basis laplace --field-n 80 --field-lo -38.9 --field-hi 39.2 --field-atoms 4 --atom-smin 1.0 --atom-kmax 0.7071 --field-kmax 1.0 --field-order 1"
out=results_mimo_prod; mkdir -p $out
case "$1" in
  kbc_dec) [ -f $out/start_kbc_p4.npz ] || python tools/mimo_start.py --siso results_n100_kbc/K_B_C_s0_ne100.npz --pair2=-4,4 --out $out/start_kbc_p4.npz
           args="--case K+B+C --ctrl-struct decentral --mimo-start $out/start_kbc_p4.npz"; od=$out/kbc_dec ;;
  kbc_cen) args="--case K+B+C --ctrl-struct central --mimo-start $out/kbc_dec/K_B_C_s0_ne100_mimo2x2_decentral.npz"; od=$out/kbc_cen ;;
  at_dec)  [ -f $out/start_atoms.npz ] || python tools/mimo_start.py --siso results_fb_atoms_n200_long/K_FB_FC_s0_ne200.npz --atoms --out $out/start_atoms.npz
           args="--case KGC+FB+FC --ctrl-struct decentral --mimo-start $out/start_atoms.npz $ATOMS"; od=$out/at_dec ;;
  at_dec_p) [ -f $out/start_proto_atoms_dec.npz ] || python tools/mimo_start.py --proto results_mimo/atoms_dec.npz --out $out/start_proto_atoms_dec.npz
           args="--case KGC+FB+FC --ctrl-struct decentral --mimo-start $out/start_proto_atoms_dec.npz $ATOMS"; od=$out/at_dec_p ;;
  at_cen)  args="--case KGC+FB+FC --ctrl-struct central --mimo-start $out/at_dec/KGC_FB_FC_s0_ne100_mimo2x2_decentral.npz $ATOMS"; od=$out/at_cen ;;
  *) echo "unknown case $1"; exit 2 ;;
esac
mkdir -p $od
echo "=== launch $1 $(date +%s)" >> $od/run.log
python run_case_epi.py --eps 0.05 --start 0 $LOOP --n-act 2 --n-sens 2 $args $EXTRA --outdir $od --dump $od/dump_$(date +%s) >> $od/run.log 2>&1
echo "=== $1 done"
