"""
check_band.py — is the frequency band wide enough for the CLOSED loop?

The band was derived from an OPEN-loop sweep.  But the compensator contributes
its own poles, so the closed loop can resonate outside it: one design's true
peak sits at omega = -9.69, outside both the run window [-5.5, 7.5] and the
certification band [-5.40, 7.60], and its reported gamma was consequently the
height of a LESSER peak.

This rescans every stored design over a wide band and reports the discrepancy.
It also prints the controller poles, which are what set the outer resonances and
should therefore set the band.

    python check_band.py --ne 200 --results ../results_m05 --wide 20
"""
from __future__ import annotations
# --- repository root on sys.path (moved into tools/ by cleanup_archive.py)
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
# --- end root shim
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, glob, json, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_codesign, controller as ctl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results_m05')
    ap.add_argument('--r', type=int, default=-1,
                    help='controller order; -1 reads it from '
                         'the seed catalogue, which is where '
                         'it is decided')
    ap.add_argument('--wide', type=float, default=-1.0,
                    help='half-width of the wide rescan.  <0 (default) sizes '
                         'it PER CASE as 1.5 x the outermost controller-pole '
                         '|Im| (with a floor of 20): a resonance sits near '
                         'each pole, so a fixed number is either wasteful or '
                         '-- as with a pole at |Im| = 138 against a scan of '
                         '20 -- blind.')
    ap.add_argument('--nw', type=int, default=8001)
    ap.add_argument('--band', default='auto',
                    help="the band the runs actually used.  'auto' (default) "
                         "reads wlo/whi from each result's own json -- three "
                         "different hand-typed bands across the post-hoc "
                         "tools is how a peak at +7.99 was flagged against a "
                         "band the run never used.  Or 'wlo,whi' to force "
                         "one.")
    a = ap.parse_args()
    if getattr(a, 'r', -1) is not None and a.r < 0:
        from ctrl_order import infer_r
        a.r = infer_r(a.ne, getattr(a, 'results', None))
    lo_f, hi_f = (None, None) if a.band == 'auto' \
        else tuple(float(x) for x in a.band.split(','))
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=0.8)
    print(f"{'case':12s} {'stored':>9s} {'in band':>9s} {'WIDE':>9s} "
          f"{'peak w':>8s} {'ctrl |Im|max':>12s} {'band':>15s}  verdict")
    nbad = 0
    for jf in sorted(glob.glob(os.path.join(a.results, f"*_ne{a.ne}.json"))):
        m = json.load(open(jf)); d = np.load(jf[:-5] + '.npz', allow_pickle=True)
        if 'd' not in d.files:
            continue
        arow = (d['d'], d['e'], d['f']); nv = d['nvec']; K = complex(d['K'])
        va = d['va'] if d['va'].size else None
        vs = d['vs'] if d['vs'].size else None
        b, c, *_ = D.bc(float(d['xa']), float(d['xs']), va, vs)
        cl = D.cl(b, c)
        pol = np.linalg.eigvals(ctl.A_of(*arow))
        cim = float(np.max(np.abs(pol.imag))) if pol.size else 0.0
        # per-case band from the run's own record; per-case wide scan from the
        # poles it actually has
        if lo_f is None:
            lo = float(m.get('wlo', -5.5)); hi = float(m.get('whi', 7.5))
        else:
            lo, hi = lo_f, hi_f
        wide = a.wide if a.wide > 0 else max(20.0, 1.5 * cim)
        W = np.linspace(-wide, wide, a.nw)
        g = np.array([cl.gain(float(w), arow, nv, K,
                              want_vectors=False)['sigma'] for w in W])
        inb = (W >= lo) & (W <= hi)
        g_in, g_wide = float(g[inb].max()), float(g.max())
        wpk = float(W[int(g.argmax())])
        bad = g_wide > g_in * (1 + 1e-6)
        if bad:
            nbad += 1
        print(f"{m['case']:12s} {m['gamma']:9.4f} {g_in:9.4f} {g_wide:9.4f} "
              f"{wpk:+8.2f} {cim:12.2f} [{lo:+6.1f},{hi:+6.1f}]  "
              f"{'*** PEAK OUTSIDE THE BAND ***' if bad else 'band OK'}")
    print(f"\n{nbad} design(s) have their true peak OUTSIDE the band their "
          f"run used.")
    print("The band must cover the CONTROLLER poles: their |Im| is the last")
    print("column, and a resonance sits near each of them.  Re-run with")
    print("--wlo/--whi wide enough, and widen certify_results' band to match.")


if __name__ == "__main__":
    main()
