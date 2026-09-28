"""
test_substitution.py — is the coincidence of gamma between designs a genuine
SUBSTITUTION, or merely a shared active constraint?

Several designs were observed to agree to six significant figures (e.g. K, K+B
and FB all give 9.831876; K+C and FC give 9.660144).  Two readings are possible:

  (a) SUBSTITUTION: the designs realise the *same closed loop*, so
      sigma_1(omega) coincides at EVERY frequency;
  (b) SHARED ACTIVE CONSTRAINT: different closed loops that happen to be limited
      by the same bound, so only the PEAK coincides.

The test discriminates them by comparing the stored gain curves over the whole
band.  Reported metrics, for each group of coincident designs:

    max |sigma_a - sigma_b| / max(sigma)       relative sup-norm difference
    max |sigma_a - sigma_b| / max(sigma)  away from the peak (|w-w*|>0.2)
    correlation of the two curves

    python test_substitution.py --ne 200
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np


def best(results, case, ne):
    cand = []
    for jf in glob.glob(os.path.join(results,
                                     f"{case.replace('+','_')}_s*_ne{ne}.json")):
        try:
            cand.append((json.load(open(jf))['gamma'], jf[:-5] + '.npz'))
        except Exception:
            pass
    return min(cand) if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results')
    ap.add_argument('--tol', type=float, default=1e-4,
                    help='relative tolerance for calling two curves identical')
    a = ap.parse_args()

    # groups of designs observed to share a gamma
    groups = [('sensor fixed',  ['K', 'K+B', 'FB', 'KSP', 'KGC', 'KG']),
              ('sensor free',   ['K+C', 'FC'])]

    for title, cases in groups:
        got = [(c, best(a.results, c, a.ne)) for c in cases]
        got = [(c, g) for c, g in got if g is not None]
        if len(got) < 2:
            continue
        print(f"\n=== group: {title} ===")
        for c, (g, _) in got:
            print(f"    {c:8s} gamma = {g:.6f}")
        ref_case, (ref_g, ref_npz) = got[0]
        dref = np.load(ref_npz)
        w = dref['omega_grid']; sref = dref['gain']
        wstar = float(w[np.argmax(sref)])
        far = np.abs(w - wstar) > 0.2
        print(f"    reference {ref_case}, peak at w* = {wstar:+.4f}")
        print(f"    {'vs':10s} {'dgamma':>10s} {'sup-diff':>10s} "
              f"{'off-peak':>10s} {'corr':>8s}  verdict")
        for c, (g, npz) in got[1:]:
            s = np.load(npz)['gain']
            if s.shape != sref.shape:
                print(f"    {c:10s} (different omega grid, skipped)"); continue
            scale = max(sref.max(), s.max())
            sup = float(np.max(np.abs(s - sref)) / scale)
            off = float(np.max(np.abs(s - sref)[far]) / scale) if far.any() else np.nan
            corr = float(np.corrcoef(s, sref)[0, 1])
            if sup < a.tol:
                verdict = 'IDENTICAL closed loop  -> substitution'
            elif abs(g - ref_g) / max(ref_g, 1e-30) < a.tol:
                verdict = 'peaks agree, curves DIFFER -> shared active constraint'
            else:
                verdict = 'different'
            print(f"    {c:10s} {abs(g-ref_g):10.2e} {sup:10.2e} {off:10.2e} "
                  f"{corr:8.5f}  {verdict}")


if __name__ == "__main__":
    main()
