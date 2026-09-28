"""
check_incomplete.py — which results might be corrupted by the alpha-gradient bug?

The bug: `alpha_certified` fetched the eigenvectors for the alpha GRADIENT with
a single targeted shift, and when that shift failed to converge `alpha_krylov`
RAISED, killing the run (`FBR+FCR` died exactly there).  Two consequences:

  * a case that CRASHED wrote nothing, or kept an older, worse file;
  * a case that survived is NOT suspect: the value always came from the
    certificate, and the gradient is only a search direction, so a run that
    completed produced a valid design -- `verify_dense.py` confirms it
    independently.

So the question is not "is this result wrong" but "is this result MISSING or
STALE".  This reports what is present, what is absent, and which entries look
under-converged (the exchange hit its outer cap rather than converging).

    python check_incomplete.py --ne 200 --results results_two
"""
from __future__ import annotations
import argparse, glob, json, os, sys

EXPECTED = ['K', 'K+B+C', 'FBR+FCR', 'KG+FBR+FCR', 'K+FBR+FCR',
            'FB+FC', 'K+FB+FC']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results_two')
    ap.add_argument('--cases', default=','.join(EXPECTED))
    a = ap.parse_args()
    want = [c.strip() for c in a.cases.split(',') if c.strip()]
    found = {}
    for jf in glob.glob(os.path.join(a.results, f"*_ne{a.ne}.json")):
        try:
            m = json.load(open(jf))
        except Exception:
            continue
        found[m['case']] = m
    print(f"{'case':12s} {'gamma':>10s} {'alpha':>9s} {'outer':>6s} "
          f"{'|Omega|':>8s}  status")
    missing, weak = [], []
    for c in want:
        m = found.get(c)
        if m is None:
            print(f"{c:12s} {'---':>10s} {'---':>9s} {'-':>6s} {'-':>8s}  "
                  f"MISSING -- re-run")
            missing.append(c)
            continue
        conv = m.get('exchange_converged', False)
        stall = m.get('exchange_stalled', False)
        note = ('converged' if conv else
                ('stalled' if stall else 'OUTER CAP -- under-converged'))
        if not conv:
            weak.append(c)
        print(f"{c:12s} {m['gamma']:10.4f} {m['alpha']:9.4f} "
              f"{m.get('outer','?'):>6} {m.get('nOmega','?'):>8}  {note}")
    print()
    if missing:
        print(f"MISSING ({len(missing)}): {' '.join(missing)}")
        print(f"  CASES=\"{' '.join(missing)}\" CASE=<case> sbatch case.slurm")
    if weak:
        print(f"NOT CONVERGED ({len(weak)}): {' '.join(weak)}")
        print(f"  CASES=\"{' '.join(weak)}\" OUTER=60 CASE=<case> sbatch case.slurm")
    if not missing and not weak:
        print("complete and converged.")
    print("\nNote: results that COMPLETED are not suspect.  The alpha VALUE "
          "always\ncame from the certificate; the bug only lost a search "
          "direction, and\nverify_dense.py checks each design independently.")


if __name__ == "__main__":
    main()
