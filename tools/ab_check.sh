#!/usr/bin/env bash
# A/B harness for exact code changes (STAGE6 6.29): short MIMO runs whose round
# values, certified values and full-model LU counts by call site are compared
# between two trees (e.g. HEAD before and after a change).  Serial, deterministic.
#   tools/ab_check.sh OUTDIR            run THIS tree's cases into OUTDIR
#   tools/ab_check.sh --compare A B     compare two OUTDIRs
# Cases: kbc5  K+B+C central order 5 from results_mimo_prod/start_kbc_cen5.npz, 4 rounds
#        at    atoms central from the at_dec record, 2 rounds (static gains, wide band)
set -uo pipefail; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ "${1:-}" = "--compare" ]; then
  python3 - "$2" "$3" <<'PY'
import sys, json, glob, os
def load(d, case):
    runs = sorted(glob.glob(f"{d}/{case}/dump/outer_*.json"))
    rounds = [json.load(open(f)) for f in runs]
    rec = [f for f in glob.glob(f"{d}/{case}/*.json") if '_check' not in f and 'ckpt' not in f]
    r = json.load(open(rec[0])) if rec else {}
    lu = r.get('lu_full_by_site') or (rounds[-1]['lu']['by_site_full_cum'] if rounds and 'lu' in rounds[-1] else {})
    t = open(f"{d}/{case}.time").read().split() if os.path.exists(f"{d}/{case}.time") else ['?']
    return rounds, r, lu, t[0]
for case in ('kbc5', 'at'):
    A, B = load(sys.argv[1], case), load(sys.argv[2], case)
    print(f"== {case}: wall {A[3]} s -> {B[3]} s")
    for a, b in zip(A[0], B[0]):
        print(f"  round {a['outer']}: t {a['t']:.10f} / {b['t']:.10f}  cert {a['gamma_certified']:.10f} / {b['gamma_certified']:.10f}"
              f"  rel {abs(a['gamma_certified'] - b['gamma_certified']) / a['gamma_certified']:.1e}")
    ga, gb = A[1].get('gamma'), B[1].get('gamma')
    if ga and gb:
        print(f"  record gamma {ga:.12f} / {gb:.12f} rel {abs(ga - gb) / ga:.1e}; dense all-w {A[1].get('gamma_dense_all_w')} / {B[1].get('gamma_dense_all_w')}")
    la, lb = sum(A[2].values()), sum(B[2].values())
    print(f"  full-model LUs (whole run incl. save when recorded) {la} -> {lb}")
    # site keys carry line numbers, which move with edits: aggregate by the factorising callee FILE
    ca, cb = {}, {}
    for d, c in ((A[2], ca), (B[2], cb)):
        for k, v in d.items():
            f = k.split('<-')[-1].strip().split(':')[0]; c[f] = c.get(f, 0) + v
    for f in sorted(set(ca) | set(cb), key=lambda f: -max(ca.get(f, 0), cb.get(f, 0))):
        print(f"    {ca.get(f, 0):7d} -> {cb.get(f, 0):7d}  {f}")
    print("    top sites B:", ", ".join(f"{k} {v}" for k, v in sorted(B[2].items(), key=lambda q: -q[1])[:5]))
PY
  exit 0
fi
out=${1:?OUTDIR}; mkdir -p $out
LOOP=$(sed -n '/^LOOP=/,/stop-progress [0-9:.]*"/p' tools/n100_kbc.sh | tr -d '\\\n' | sed 's/^LOOP="//; s/"$//')
ATOMS="--field-basis laplace --field-n 80 --field-lo -38.9 --field-hi 39.2 --field-atoms 4 --atom-smin 1.0 --atom-kmax 0.7071 --field-kmax 1.0 --field-order 1"
run() {  # name rounds args
  local od=$out/$1; rm -rf $od; mkdir -p $od
  local t0=$(date +%s)
  python run_case_epi.py --eps 0.05 --start 0 $LOOP --outer $2 --stop-progress 8:0.0015 --n-act 2 --n-sens 2 ${@:3} \
      --lucount --outdir $od --dump $od/dump > $od/run.log 2>&1
  echo $(( $(date +%s) - t0 )) > $out/$1.time
}
run kbc5 4 --case K+B+C --ctrl-struct central --mimo-start results_mimo_prod/start_kbc_cen5.npz
run at 2 --case KGC+FB+FC --ctrl-struct central --mimo-start results_mimo_prod/at_dec/KGC_FB_FC_s0_ne100_mimo2x2_decentral.npz $ATOMS
echo "ab_check done: $out"
