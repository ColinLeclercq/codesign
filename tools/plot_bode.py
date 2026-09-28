"""
plot_bode.py — Bode comparison of the designed CONTROLLERS (and the loop).

    python plot_bode.py --ne 200 [--outdir figs]

Produces figs/fig_bode.{png,pdf} with three rows:
  1. controller  K(i w)          magnitude (dB) and phase (deg)
  2. plant       G(i w) = c^H (i w M - A)^{-1} b   for the case's own b, c
  3. loop gain   L = G K         magnitude and phase

NOTE (complex system): Ginzburg-Landau is COMPLEX, so K(-i w) is NOT the
conjugate of K(i w).  The full band must be plotted, not just w >= 0 -- unlike
a real system where the negative half is redundant.
"""
from __future__ import annotations
# --- repository root on sys.path (moved into tools/ by cleanup_archive.py)
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
# --- end root shim
import argparse, glob, json, os
import sys
import numpy as np
import matplotlib
# The backend must be chosen BEFORE pyplot is imported, so we peek at argv
# rather than waiting for argparse: with 'Agg' (file-only) a later plt.show()
# only warns "FigureCanvasAgg is non-interactive".
# matplotlib.use() does not validate, so each candidate is actually tested by
# importing the toolkit it needs.
_SHOW = ('--show' in sys.argv)
_BACKEND_NOTE = ''
if _SHOW:
    import os as _os
    if not _os.environ.get('DISPLAY') and _os.name != 'nt':
        _BACKEND_NOTE = ("no $DISPLAY: reconnect with  ssh -X  (or -Y). "
                         "Figures were still written to disk.")
        matplotlib.use('Agg')
    else:
        for _bk, _mod in (('TkAgg', 'tkinter'), ('QtAgg', 'PyQt5'),
                          ('Qt5Agg', 'PyQt5'), ('GTK3Agg', 'gi')):
            try:
                __import__(_mod)
                matplotlib.use(_bk)
                break
            except Exception:
                continue
        else:
            _BACKEND_NOTE = ("no interactive toolkit found (tried tkinter, "
                             "PyQt5, gi).  conda install -n codesign tk")
            matplotlib.use('Agg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import fe_codesign, controller as ctl

# cases that actually contain a controller, plus their style
CASES = [('LQG-BT (init)',   None,         'k',  '-',  2.2),
         ('KG  static real',  'KG',         'C7', '--', 1.6),
         ('KGC static cplx',  'KGC',        'C6', '--', 1.6),
         ('KSP strictly prop','KSP',        'C8', ':',  1.5),
         ('K   proper r=7',   'K',          'C4', '-.', 1.8),
         ('K+C',              'K+C',        'C9', ':',  1.5),
         ('K+B+C',            'K+B+C',      'C0', '-',  2.0),
         ('K+FBR+FCR (real)', 'K+FBR+FCR',  'C2', '-',  1.8),
         ('K+FB+FC (complex)','K+FB+FC',    'C3', '-',  2.4)]


def best_npz(results, case, ne):
    pref = case.replace('+', '_')
    cand = []
    for jf in glob.glob(os.path.join(results, f"{pref}_s*_ne{ne}.json")):
        try:
            cand.append((json.load(open(jf))['gamma'], jf[:-5] + '.npz'))
        except Exception:
            pass
    return min(cand)[1] if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--band', default='auto',
                    help="'auto' reads wlo/whi from the result "
                         "files, or give 'lo,hi'")
    ap.add_argument('--show', action='store_true',
                    help='display the figures interactively as well as saving them (needs an X connection: ssh -X / -Y)')
    ap.add_argument('--results', default='results')
    ap.add_argument('--outdir', default='figs')
    ap.add_argument('--r', type=int, default=7)
    ap.add_argument('--wlo', type=float, default=-4.0)
    ap.add_argument('--whi', type=float, default=4.0)
    ap.add_argument('--nw', type=int, default=601)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r)
    # Plot over the band the RUNS actually used, not a fixed default.  The
    # band is derived from the seed poles (wmax = 3/2 max|Im|, band = 3/2 wmax)
    # and is the SAME band that certifies the sigma_1 peak and the closed-loop
    # stability line, so the Bode plot shows exactly the region in which the
    # design was certified -- a compensator resonance outside it would mean the
    # certification missed something.
    if a.band != 'auto':
        _lo, _hi = [float(x) for x in a.band.split(',')]
    else:
        _lo, _hi = a.wlo, a.whi
        try:
            import glob as _g, json as _j
            _b = [(json.load(open(f)).get('wlo'), json.load(open(f)).get('whi'))
                  for f in _g.glob(os.path.join(a.results,
                                                f"*_ne{a.ne}.json"))]
            _b = [(l, h) for l, h in _b if l is not None and h is not None]
            if _b:
                _lo, _hi = min(l for l, _ in _b), max(h for _, h in _b)
        except Exception:
            pass
    print(f"Bode band: [{_lo:.2f}, {_hi:.2f}]"
          + ("  (from the result files)" if a.band == 'auto' else ""))
    ws = np.linspace(_lo, _hi, a.nw)

    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    rows = []
    for label, case, col, ls, lw in CASES:
        if case is None:
            ini = np.load(f"init_ne{a.ne}.npz")
            arow = (ini['d'], ini['e'], ini['f']); nvec = ini['nvec']
            Kfb = 0j; xa, xs = float(ini['xa']), float(ini['xs'])
            va = vs = None
        else:
            npz = best_npz(a.results, case, a.ne)
            if npz is None:
                continue
            z = np.load(npz)
            arow = (z['d'], z['e'], z['f']); nvec = z['nvec']
            Kfb = complex(z['K']); xa, xs = float(z['xa']), float(z['xs'])
            va = z['va'] if z['va'].size else None
            vs = z['vs'] if z['vs'].size else None

        rr = len(np.atleast_1d(arow[0]))          # order actually stored
        Ac = ctl.A_of(*arow); Bc = np.full(rr, 1.0, complex)
        nv = np.asarray(nvec, complex).reshape(1, rr)
        # controller K(s)
        Kf = np.array([Kfb + (nv @ np.linalg.solve(1j * w * np.eye(rr) - Ac,
                                                   Bc.reshape(-1, 1)))[0, 0]
                       for w in ws])
        # plant G(s) = c^H (s M - A)^{-1} b  for THIS case's actuator/sensor
        b, c, *_ = D.bc(xa, xs, va, vs)
        Gf = np.empty(a.nw, complex)
        for i, w in enumerate(ws):
            lu = spla.splu(((1j * w) * D.M - D.A).tocsc())
            Gf[i] = np.vdot(c, lu.solve(b))
        Lf = Gf * Kf
        rows.append((label, np.abs(Kf).max(), np.abs(Kf).min()))

        for k, F in enumerate((Kf, Gf, Lf)):
            ax[k, 0].plot(ws, 20 * np.log10(np.abs(F) + 1e-300), ls,
                          color=col, lw=lw, label=label)
            ax[k, 1].plot(ws, np.degrees(np.unwrap(np.angle(F))), ls,
                          color=col, lw=lw, label=label)

    titles = ['controller  $K(i\\omega)$', 'plant  $G(i\\omega)=c^H(i\\omega M-A)^{-1}b$',
              'loop gain  $L=GK$']
    for k in range(3):
        ax[k, 0].set_ylabel('magnitude [dB]'); ax[k, 1].set_ylabel('phase [deg]')
        ax[k, 0].set_title(titles[k]); ax[k, 1].set_title(titles[k] + '  (phase)')
        for j in (0, 1):
            ax[k, j].grid(alpha=0.3); ax[k, j].set_xlabel(r'$\omega$')
            ax[k, j].set_xlim(a.wlo, a.whi)
    ax[0, 0].legend(fontsize=8, ncol=2)
    # 0 dB / -180 deg references on the loop row
    ax[2, 0].axhline(0, color='gray', lw=0.8, ls='--')
    ax[2, 1].axhline(-180, color='gray', lw=0.8, ls='--')
    fig.suptitle('Bode comparison of the optimised designs '
                 '(complex system: negative $\\omega$ is NOT redundant)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(a.outdir, f'fig_bode.{ext}'), dpi=140)
    print(f"{'design':20s} {'max|K| dB':>10s} {'min|K| dB':>10s} "
          f"{'variation':>10s}")
    for lab, mx, mn in rows:
        var = 20*np.log10(mx/max(mn, 1e-300))
        print(f"  {lab:20s} {20*np.log10(mx):10.2f} {20*np.log10(mn):10.2f} "
              f"{var:9.2f} dB   {'(static gain: flat)' if var < 1e-6 else ''}")
    if a.show:
        if _BACKEND_NOTE:
            print(f"\n!! --show: {_BACKEND_NOTE}")
        else:
            try:
                plt.show()
            except Exception as _e:
                print(f"\n!! --show failed ({type(_e).__name__}: {_e}); "
                      f"figures were still written to disk.")
    print(f"wrote {a.outdir}/fig_bode.png|pdf")


if __name__ == "__main__":
    main()
