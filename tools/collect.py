"""collect.py — aggregate results, nesting check, convergence + spike flags.
    python collect.py
Writes summary.csv and prints diagnostics.  A case is flagged SPIKE-SUSPECT if
it converged onto a barely-stable pole (alpha near the -1e-3 bound and/or a very
peaky gain curve): its gamma reflects 1/|alpha|, not genuine performance.
"""
from __future__ import annotations
# --- repository root on sys.path (moved into tools/ by cleanup_archive.py)
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
# --- end root shim
import glob, json, os
import numpy as np

# The case list is GENERATED (cases.py), not listed.  A hand-written list of 19
# left gaps that showed up as missing nesting parents -- KG+FBR+FCR ought to be
# bounded by KG+FBR and KG+FCR, and neither existed.  A partial lattice gives
# partial guarantees.  Set GL_CASES=standard for the original 19.
import os as _os
import cases as _cases
_W = _os.environ.get('GL_CASES', 'full')
ORDER = (_cases.validation() if _W == 'validation' else
         _cases.standard() if _W == 'standard' else _cases.full())
# The nesting relations are DERIVED from the case flags by nesting.py, not
# listed here.  The hand-written table asserted relations that do not hold --
# KSP was said to contain KGC (it forces D_c = 0, so it cannot), and the static
# gain cases were said to contain their field parents (they zero n, so D_c = 0
# removes the compensator rather than recovering the parent).  Each wrong entry
# cost a round of re-runs chasing a violation that was not one.
try:
    from fe_codesign import parse_case as _pc
    import nesting as _nst
    NEST = _nst.build(_pc, ORDER)
except Exception:                       # keep collect usable if that fails
    NEST = {}

def peakiness(tag):
    try:
        d = np.load(tag + '.npz')
        g = d['gain']
        return float(g.max() / max(np.median(g), 1e-30))
    except Exception:
        return float('nan')

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results',
                    help="results directory (e.g. results_epi for the "
                         "epigraph runs)")
    ap.add_argument('--ne', type=int, default=None,
                    help='restrict to one mesh')
    a = ap.parse_args()
    RES = a.results
    pat = f"*_ne{a.ne}.json" if a.ne else "*.json"
    rows = [json.load(open(f))
            for f in sorted(glob.glob(os.path.join(RES, pat)))]
    if not rows:
        print(f"no {RES}/*.json found"); return
    # Safety net: the optimiser's gamma can UNDER-report (a seeded local search
    # may miss a higher peak).  The stored gain curve is sampled on [-4,4], so
    # take the max of the two before ranking.  certify_results.py gives the
    # rigorous value; this just stops a silent under-report reaching the table.
    for r in rows:
        tag = os.path.join(RES, f"{r['case'].replace('+','_')}"
                                f"_s{r['start']}_ne{r['ne']}")
        try:
            gcur = float(np.max(np.load(tag + '.npz')['gain']))
            if gcur > r['gamma'] * (1 + 1e-9):
                r['gamma_optimiser'] = r['gamma']
                r['gamma'] = gcur
                r['under_reported'] = True
        except Exception:
            pass
    # VERIFICATION BEFORE RANKING.  A "lowest gamma wins" ranking is actively
    # attracted to invalid designs (that is how a low gamma is obtained), and
    # a nesting check over invalid designs manufactures violations: the two
    # VIOLATED lines of one sweep compared three designs that were ALL
    # unstable.  Rank on evidence: alpha_dense written by the run's save veto
    # or by verify_dense.py --update where it exists, the stored alpha only as
    # a fallback -- and say which was used.
    def _verified_alpha(r):
        tag_ = os.path.join(RES, f"{r['case'].replace('+','_')}"
                                 f"_s{r['start']}_ne{r['ne']}")
        try:
            d_ = np.load(tag_ + '.npz', allow_pickle=True)
            if 'alpha_dense' in d_.files and np.isfinite(float(d_['alpha_dense'])):
                return float(d_['alpha_dense']), 'dense'
        except Exception:
            pass
        ad = r.get('alpha_dense')
        if ad is not None:
            try:
                if np.isfinite(float(ad)):
                    return float(ad), 'dense'
            except (TypeError, ValueError):
                pass
        return float(r['alpha']), 'stored'
    n_unverified = 0
    for r in rows:
        r['_alpha_v'], r['_alpha_src'] = _verified_alpha(r)
        r['_valid'] = r['_alpha_v'] <= -1e-3
        if r['_alpha_src'] == 'stored':
            n_unverified += 1
    if n_unverified:
        print(f"NOTE: {n_unverified} result(s) carry no alpha_dense -- run "
              f"'python verify_dense.py --eps 0.05 --update' so the ranking and the "
              f"nesting check rest on verified stability.")
    n_invalid = sum(1 for r in rows if not r['_valid'])
    if n_invalid:
        print(f"EXCLUDED from ranking and nesting: {n_invalid} INVALID "
              f"result(s):")
        for r in rows:
            if not r['_valid']:
                print(f"    {r['case']:12s} gamma={r['gamma']:.4f} "
                      f"alpha[{r['_alpha_src']}]={r['_alpha_v']:+.4f}")
    best = {}
    for r in rows:
        if not r['_valid']:
            continue
        c = r['case']
        if c not in best or r['gamma'] < best[c]['gamma']:
            best[c] = r
    hdr = ("case,method,gamma,alpha,n_active,spike,peak/median,xa,xs,NZ,"
           "calls,seconds,starts,gamma_spread")
    lines = [hdr]
    for c in ORDER:
        if c not in best: continue
        gs = [r['gamma'] for r in rows if r['case'] == c]
        b = best[c]
        tag = os.path.join(RES, f"{c.replace('+','_')}"
                                f"_s{b['start']}_ne{b['ne']}")
        pk = b.get('gain_peak_over_median', peakiness(tag))
        spike = b.get('spike_suspect', b['alpha'] > -0.01)
        if b.get('under_reported'):
            print(f"   NOTE {c}: optimiser reported {b['gamma_optimiser']:.4f}, "
                  f"gain curve peaks at {b['gamma']:.4f} -> using the curve; "
                  f"run certify_results.py")
        lines.append(f"{c},{b.get('method','ladder')},{b['gamma']:.6f},{b['alpha']:.6f},{b['n_active']},"
                     f"{'YES' if spike else 'no'},{pk:.1f},{b['xa']:.4f},"
                     f"{b['xs']:.4f},{b['NZ']},{b['calls']},{b['seconds']},"
                     f"{len(gs)},{max(gs)-min(gs):.4f}")
    open('summary.csv', 'w').write("\n".join(lines) + "\n")
    print("\n".join(lines))

    print("\n--- nesting sanity check (child <= every parent) ---")
    stale_children = set()
    ok = True
    for child, parents in NEST.items():
        if child not in best: continue
        for p in parents:
            if p not in best: continue
            # Tolerance = the CERTIFICATE'S own reproducibility.  The sweep's
            # monotonicity guarantee is exact for the optimiser's certified
            # value, but certify_results re-finds each peak to rtol = 1e-4
            # RELATIVE, so two measurements of the SAME design can differ by
            # ~1e-3 at gamma ~ 10 (measured: KSP+FB stored 9.8313, certified
            # 9.8320).  Comparing with an ABSOLUTE 1e-4 flags that measurement
            # noise as a violation.  Anything beyond the relative tolerance is
            # a real regression and still fires.
            tol = 1e-4 * max(1.0, best[p]['gamma'])
            good = best[child]['gamma'] <= best[p]['gamma'] + tol
            ok &= good
            note = ''
            if not good:
                bc = best[child]
                pgam = bc.get('parent_gamma')
                if pgam is not None and pgam > best[p]['gamma'] * (1 + 1e-6):
                    note = (f"  <-- STALE: it started from {child}'s parent at "
                            f"gamma={pgam:.4f}, but {p} has since improved to "
                            f"{best[p]['gamma']:.4f}. Re-run {child}.")
                    stale_children.add(child)
                else:
                    note = "  <-- child worse than parent: re-run it from the parent"
                    stale_children.add(child)
            print(f"  {child:9s} {best[child]['gamma']:9.4f}  <=  {p:7s} "
                  f"{best[p]['gamma']:9.4f}   {'OK' if good else 'VIOLATED'}{note}")
    if ok:
        print("ALL NESTED RELATIONS OK")
    else:
        print("SOME VIOLATIONS")
        # map case -> tasks.txt line numbers so the fix is copy-pasteable
        try:
            lines = [l.split() for l in open('tasks.txt')
                     if l.strip() and not l.startswith('#')]
            idx = []
            for n, fields in enumerate(lines, start=1):
                if fields[0] in stale_children:
                    idx.append(str(n))
            if idx:
                ne = best[list(stale_children)[0]]['ne']
                print(f"\n  FIX: re-run the stale children (their parents moved):\n"
                      f"    sbatch --array={','.join(idx)} "
                      f"--export=ALL,NE={ne},SOB=0.8 submit.slurm")
        except Exception:
            pass

    print("\n--- convergence + spike diagnostics (best start) ---")
    for c in ORDER:
        if c not in best: continue
        b = best[c]
        tag = os.path.join(RES, f"{c.replace('+','_')}"
                                f"_s{b['start']}_ne{b['ne']}")
        if b.get('method') == 'epigraph':
            # the epigraph driver keeps no per-call trace; its `history` records
            # one entry per EXCHANGE step, which is the meaningful unit here
            h = b.get('history', [])
            if not h:
                print(f"  {c:9s} epigraph, no history recorded")
                continue
            g_first = h[0].get('gamma_certified', h[0].get('gamma', np.nan))
            g_last = h[-1].get('gamma_best', h[-1].get('gamma', np.nan))
            conv = bool(b.get('exchange_converged', False))
            stalled = bool(b.get('exchange_stalled', False))
            spike = b.get('spike_suspect', b['alpha'] > -0.01)
            gi = b.get('gate_inconclusive', 0); gl = b.get('gate_located', 0)
            gate = (f" gate[{gl} found/{gi} inconcl]" if (gi or gl) else "")
            print(f"  {c:9s} EPIGRAPH  outer={len(h):2d} |Omega|={b.get('nOmega','?'):3} "
                  f"gamma {g_first:.3f} -> {g_last:.3f}  #active={b['n_active']}{gate}  "
                  + ('exchange CONVERGED' if conv else
                     ('exchange TERMINATED: frequency set stationary, inner solver at its limit -> best known' if stalled
                      else 'outer limit reached -> raise --outer'))
                  + ('; SPIKE-SUSPECT' if spike else ''))
            continue
        try:
            d = np.genfromtxt(tag + '.oracle.csv', delimiter=',', names=True)
            bg = np.atleast_1d(d['best_gamma'])
        except Exception:
            continue
        # Measure convergence on the FINAL LADDER STAGE only, and on the
        # running MINIMUM.  The raw best_gamma trace now contains upward jumps
        # from in-loop certification reseeding (the oracle was tracking a local
        # peak; the certificate replaced it with the true, larger value).  Those
        # jumps are corrections, not divergence, and they must not be read as
        # "still descending".
        try:
            st = np.genfromtxt(tag + '.stage.csv', delimiter=',', names=True)
            last_stage = int(np.atleast_1d(st['stage'])[-1])
            sel = np.atleast_1d(d['stage']) == last_stage
            if np.count_nonzero(sel) > 10:
                bg = bg[sel]
        except Exception:
            pass
        bg = np.minimum.accumulate(bg[np.isfinite(bg)]) if np.any(np.isfinite(bg)) else bg
        finite = bg[np.isfinite(bg)]
        if finite.size == 0:
            print(f"  {c:9s} calls={b['calls']:5d}  NO FEASIBLE POINT in this run "
                  f"(saved gamma={b['gamma']:.3f} kept from a previous run) -> "
                  f"resume with a FINE ladder, e.g. LADDER=0.02,0.005")
            continue
        tail = finite[-max(1, len(finite)//10):]
        chg = float(tail.max() - tail.min())
        stalled = chg < 1e-3 * max(1.0, tail.mean())   # 0.1% of gamma
        spike = b.get('spike_suspect', b['alpha'] > -0.01)
        flags = []
        if not stalled: flags.append('STILL DESCENDING -> RESUME=1')
        if spike: flags.append(f"SPIKE-SUSPECT (alpha={b['alpha']:+.4f})")
        print(f"  {c:9s} calls={b['calls']:5d} gamma {finite[0]:.3f} -> {finite[-1]:.3f}  "
              f"last-decile={chg:.1e}  #active={b['n_active']}  "
              f"{'; '.join(flags) if flags else 'converged, clean'}")

if __name__ == "__main__":
    main()
