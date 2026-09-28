"""
plot_results.py — figures from the cluster results.

  python plot_results.py [--ne 200] [--results results] [--outdir figs]

Reads results/<CASE>_s<START>_ne<NE>.{json,npz}; for each case picks the BEST
(lowest-gamma) start.  Produces:
  fig_sigma1.png / .pdf   sigma_1(omega): uncontrolled + all cases
  fig_fields.png  / .pdf   actuator/sensor Gaussian centres and optimised fields
Everything is read straight from the .npz (gain curve, psi_a, psi_s) that
run_case.py already saved -- no re-optimisation, no plant solves needed here,
except the uncontrolled curve and the nominal Gaussians (cheap, via fe_codesign).
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

# ---- fixed display order and styles ----
# every case, grouped so the figure stays readable (--group)
CASE_STYLE = [
    ('uncontrolled',  'k',   '-',  2.4, 'all'),
    # --- compensator structure (actuator & sensor at the nominal Gaussians) ---
    ('KG',            'C7',  '--', 1.5, 'structure'),
    ('KGC',           'C6',  '--', 1.5, 'structure'),
    ('KSP',           'C8',  ':',  1.5, 'structure'),
    ('K',             'C4',  '-.', 1.7, 'structure'),
    # --- placement ---
    ('B',             'C1',  '--', 1.5, 'placement'),
    ('C',             'C2',  '--', 1.5, 'placement'),
    ('B+C',           'C5',  '-',  1.7, 'placement'),
    ('K+B',           'C9',  ':',  1.4, 'placement'),
    ('K+C',           'C3',  ':',  1.4, 'placement'),
    ('K+B+C',         'C0',  '-',  2.0, 'placement'),
    # --- REAL-constrained fields ---
    ('FBR',           'C7',  '-.', 1.5, 'fields'),
    ('FCR',           'C8',  '-.', 1.5, 'fields'),
    ('FBR+FCR',       'C2',  '--', 1.9, 'fields'),
    ('KG+FBR+FCR',    'C5',  '--', 1.7, 'fields'),
    ('K+FBR+FCR',     'C0',  '--', 2.1, 'fields'),
    # --- COMPLEX fields ---
    ('FB',            'C6',  '-',  1.5, 'fields'),
    ('FC',            'C4',  '-',  1.5, 'fields'),
    ('FB+FC',         'C1',  '-',  2.1, 'fields'),
    ('K+FB+FC',       'C3',  '-',  2.5, 'fields'),
]

# Fundamental LOWER bounds, READ from the file tools/bounds.py writes
# (python tools/bounds.py --ne 200 --json results/bounds_ne200.json), never typed
# in (CLEANUP 5.2-5.3, closed 2026-09-26).  They are FLOORS: the sensor-limited
# one holds for designs whose sensor sits at the file's xs, the actuator-limited
# one for designs whose actuator sits at its xa, the rank-one floor for any
# single actuator/sensor pair.
def load_bounds(path):
    import json as _json
    try:
        B = _json.load(open(path))
    except Exception as e:
        print(f"bounds file {path} unavailable ({type(e).__name__}): no bound lines drawn")
        return []
    return [(B['sensor_limited'], f"lower bound: sensor-limited ($x_s$ = {B['xs']:g})", 'C0'),
            (B['actuator_limited'], f"lower bound: actuator-limited ($x_a$ = {B['xa']:g})", 'C2'),
            (B['floor'], 'lower bound: absolute floor  $\\sup_\\omega\\sigma_2$', 'C3')]


def best_npz(results, case, ne):
    pref = case.replace('+', '_')
    cands = []
    for jf in glob.glob(os.path.join(results, f"{pref}_s*_ne{ne}.json")):
        try:
            d = json.load(open(jf))
            cands.append((d['gamma'], jf[:-5] + '.npz', d))
        except Exception:
            pass
    if not cands:
        return None
    g, npz, meta = min(cands, key=lambda t: t[0])
    return npz, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--show', action='store_true',
                    help='display the figures interactively as well as saving them (needs an X connection: ssh -X / -Y)')
    ap.add_argument('--results', default='results')
    ap.add_argument('--outdir', default='figs')
    ap.add_argument('--r', type=int, default=-1,
                    help='controller order; -1 (default) reads it from the '
                         'seed catalogue, which is where it is decided '
                         '(hardcoding 7 broke the curve recompute against '
                         'r = 8 designs)')
    ap.add_argument('--group', default='all',
                    choices=('all','structure','placement','fields'),
                    help='which family to draw in fig_sigma1')
    ap.add_argument('--nobounds', action='store_true')
    ap.add_argument('--band', default='auto',
                    help="frequency band over which sigma_1(omega) is "
                         "COMPUTED.  'auto' (default) uses each run's own "
                         "certification band from its json (wlo/whi, e.g. "
                         "[-30, 30]); or 'lo,hi'.  Independent of the "
                         "plotting range --xlim.")
    ap.add_argument('--xlim', default='-10,10',
                    help="plotting range of the sigma_1 panel (default "
                         "-10,10); the curve is computed over --band "
                         "regardless, so this is a pure viewing choice")
    ap.add_argument('--nw', type=int, default=1501,
                    help='points for any curve that must be recomputed '
                         'because the stored grid is narrower than --band')
    ap.add_argument('--update', action='store_true',
                    help='write recomputed curves back into the npz '
                         '(omega_grid/gain), so the recomputation happens '
                         'once; the old narrow grid is kept as '
                         'omega_grid_44/gain_44')
    a = ap.parse_args()
    if a.r < 0:
        from ctrl_order import infer_r
        a.r = infer_r(a.ne, getattr(a, 'results', None))
    os.makedirs(a.outdir, exist_ok=True)

    # plant handle for the uncontrolled curve + nominal Gaussians
    import fe_codesign
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r)
    x = D.xn

    # ================= FIGURE 1: sigma_1(omega) =================
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    rows = []
    warn = []
    _xlo, _xhi = (float(v) for v in a.xlim.split(','))
    _blo = _bhi = None
    if a.band != 'auto':
        _blo, _bhi = (float(v) for v in a.band.split(','))
    sel = [t for t in CASE_STYLE
           if a.group == 'all' or t[4] in ('all', a.group)]
    for case, col, ls, lw, _grp in sel:
        if case == 'uncontrolled':
            lo = _blo if _blo is not None else -30.0
            hi = _bhi if _bhi is not None else 30.0
            # open loop: plant-only energy resolvent, no controller at all.
            # ~a minute at ne = 200 over +-30, so cached on disk per band.
            _cache = os.path.join(a.results,
                                  f"olcurve_ne{a.ne}_{lo:g}_{hi:g}.npz")
            import scipy.sparse.linalg as spla
            if os.path.exists(_cache):
                _z = np.load(_cache)
                ws, g = _z['ws'], _z['g']
            else:
                ws = np.linspace(lo, hi, a.nw)
                def openloop_sigma(w):
                    S = (1j * w) * D.M - D.A
                    lu = spla.splu(S.tocsc())
                    def Wmv(v):
                        t = lu.solve(v); t = D.M @ t
                        t = lu.solve(t, trans='H'); return D.M @ t
                    op = spla.LinearOperator((D.n, D.n), matvec=Wmv,
                                             dtype=complex)
                    lam = spla.eigs(op, k=1, which='LM', tol=1e-9,
                                    return_eigenvectors=False, maxiter=5000)
                    return float(np.sqrt(abs(lam[0].real)))
                g = np.array([openloop_sigma(w) for w in ws])
                np.savez(_cache, ws=ws, g=g)
            gpk = g.max(); wpk = ws[g.argmax()]
        else:
            got = best_npz(a.results, case, a.ne)
            if got is None:
                continue
            npz, meta = got
            d = np.load(npz)
            ws = d['omega_grid']; g = d['gain']
            lo = _blo if _blo is not None else float(meta.get('wlo', -30.0))
            hi = _bhi if _bhi is not None else float(meta.get('whi', 30.0))
            _need = ws.min() > lo + 1e-9 or ws.max() < hi - 1e-9
            if not _need:
                # coverage alone is not enough: the first wide-band pass
                # stored a UNIFORM 0.04 grid, which covers the band but
                # under-samples the peaks (K+B+C's zoom).  Recompute whenever
                # the CORE resolution is coarser than the 0.01 target.
                _core = np.sort(ws[(ws > -8.0) & (ws < 8.0)])
                if _core.size < 3 or float(np.median(np.diff(_core))) > 0.015:
                    _need = True
            if _need:
                # stored grid narrower than the requested band (old runs
                # stored [-4, 4]): recompute over the band, optionally cache
                # back so this happens once.
                import controller as _ctl
                arow_ = (d['d'], d['e'], d['f'])
                va_ = d['va'] if d['va'].size else None
                vs_ = d['vs'] if d['vs'].size else None
                b_, c_, *_ = D.bc(float(d['xa']), float(d['xs']), va_, vs_)
                # the resolvent must match THIS file's controller order, not
                # the global default (designs store their own r)
                from closedloop_sparse import ClosedLoopResolvent as _CLR
                cl_ = _CLR(D.M, D.A, b_, c_, r=len(np.asarray(d['d'])),
                           mc=D.mc)
                # NONUNIFORM grid: the peaks live in the core, the wings
                # are smooth resonant tails -- a uniform grid over +-30 at
                # plot resolution wastes 4x the solves or (worse) degrades
                # the core to 0.04 spacing, which is what made K+B+C's zoom
                # look under-sampled.
                _core = min(8.0, 0.5 * (hi - lo))
                ws = np.unique(np.concatenate([
                    np.arange(lo, -_core, 0.05),
                    np.arange(-_core, _core, 0.01),
                    np.arange(_core, hi, 0.05), [hi]]))
                g = np.array([cl_.gain(float(w), arow_, d['nvec'],
                                       complex(d['K']),
                                       want_vectors=False)['sigma']
                              for w in ws])
                print(f"  {case:12s}: stored grid was "
                      f"[{float(d['omega_grid'].min()):g}, "
                      f"{float(d['omega_grid'].max()):g}]; recomputed over "
                      f"[{lo:g}, {hi:g}]"
                      + ("  (written back)" if a.update else ""))
                if a.update:
                    payload = {k: d[k] for k in d.files}
                    if 'omega_grid_44' not in d.files:   # keep the FIRST original
                        payload['omega_grid_44'] = d['omega_grid']
                        payload['gain_44'] = d['gain']
                    payload['omega_grid'] = ws
                    payload['gain'] = g
                    np.savez(npz, **payload)
            # prefer a CERTIFIED gamma if certify.py has written one; otherwise
            # take the max over the sampled curve, never the optimiser's value
            # alone (a seeded search can under-report -- see README).
            gpk = float(meta.get('gamma_certified', max(float(meta['gamma']),
                                                        float(g.max()))))
            wpk = ws[g.argmax()]
        rows.append((case, gpk, wpk))
        if case != 'uncontrolled':
            cmax = float(g.max())
            if gpk > cmax * (1 + 1e-3):
                warn.append((case, gpk, cmax, float(ws.min()), float(ws.max())))
        lab = f"{case}  ($\\gamma$={gpk:.2f})"
        ax[0].semilogy(ws, g, ls, color=col, lw=lw, label=lab)
        ax[1].plot(ws, g, ls, color=col, lw=lw, label=lab)

    if not a.nobounds:
        for val, lab, col in load_bounds(os.path.join(a.results, f'bounds_ne{a.ne}.json')):
            for k in (0, 1):
                ax[k].axhline(val, color=col, lw=1.1, ls=(0, (7, 4)),
                              zorder=0, alpha=0.75,
                              label=(f"{lab} = {val:.3f}" if k == 1 else None))
    ax[0].set_xlabel(r'$\omega$'); ax[0].set_ylabel(r'$\sigma_1(\omega)$')
    ax[0].set_title(r'Energy-resolvent gain $\sigma_1(\omega)$ (all cases)')
    ax[0].grid(alpha=0.3, which='both'); ax[0].set_xlim(_xlo, _xhi)
    ax[0].legend(fontsize=7, ncol=2, loc='upper right')
    ax[1].set_xlim(-1.1, 0.1); ax[1].set_ylim(2, 12)
    ax[1].set_xlabel(r'$\omega$'); ax[1].set_ylabel(r'$\sigma_1(\omega)$')
    ax[1].set_title('Zoom on the controlled peaks')
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7, ncol=2, loc='lower left')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(a.outdir, f'fig_sigma1.{ext}'), dpi=140)
    print("sigma_1 peaks:")
    for c, gp, wp in rows:
        print(f"  {c:12s} gamma={gp:9.4f}  omega*={wp:+.4f}")

    # ============ FIGURE 2: fields, REAL and IMAGINARY parts (2x2) ============
    # The nominal Gaussians are REAL, so a complex optimised field has an extra
    # degree of freedom the reference never had.  Plotting Re and Im separately
    # makes that explicit; the FBR/FCR cases are the fair, real-constrained
    # comparison.
    def norm(v):
        return v / np.sqrt(np.real(np.vdot(v, D.M @ v)))

    gA = norm(D.gauss(-1.03).astype(complex))
    gC = norm(D.gauss(0.98).astype(complex))

    fig2, ax2 = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    ax2[0, 0].plot(x, gA.real, 'k-', lw=2.2, label='original B (Gaussian, real)')
    ax2[0, 1].plot(x, gC.real, 'k-', lw=2.2, label='original C (Gaussian, real)')

    place_a = [('B', 'C1', 'optim B'), ('B+C', 'C5', 'optim B+C'),
               ('K+B+C', 'C0', 'optim K+B+C')]
    for case, col, lab in place_a:
        got = best_npz(a.results, case, a.ne)
        if got is None:
            continue
        _, meta = got
        ax2[0, 0].plot(x, norm(D.gauss(meta['xa']).astype(complex)).real,
                       color=col, lw=1.5, label=f"{lab} ($x_a$={meta['xa']:+.2f})")
    place_s = [('C', 'C2', 'optim C'), ('B+C', 'C5', 'optim B+C'),
               ('K+B+C', 'C0', 'optim K+B+C')]
    for case, col, lab in place_s:
        got = best_npz(a.results, case, a.ne)
        if got is None:
            continue
        _, meta = got
        ax2[0, 1].plot(x, norm(D.gauss(meta['xs']).astype(complex)).real,
                       color=col, lw=1.5, label=f"{lab} ($x_s$={meta['xs']:+.2f})")

    fields = [('FBR',     'C7', '--', 'field FBR (real-constrained)'),
              ('FBR+FCR', 'C2', '--', 'field FBR+FCR (real)'),
              ('FB',      'C6', '-.', 'field FB (complex)'),
              ('FB+FC',   'C1', '-',  'field FB+FC (complex)'),
              ('K+FB+FC', 'C3', '-',  'field K+FB+FC (complex)')]
    for case, col, ls, lab in fields:
        got = best_npz(a.results, case, a.ne)
        if got is None:
            continue
        npz, _ = got
        d = np.load(npz)
        if d['psi_a'].size:
            ax2[0, 0].plot(x, d['psi_a'].real, ls, color=col, lw=1.9, label=lab)
            ax2[1, 0].plot(x, d['psi_a'].imag, ls, color=col, lw=1.9, label=lab)
    fields_s = [('FCR',     'C7', '--', 'field FCR (real-constrained)'),
                ('FBR+FCR', 'C2', '--', 'field FBR+FCR (real)'),
                ('FC',      'C4', '-.', 'field FC (complex)'),
                ('FB+FC',   'C1', '-',  'field FB+FC (complex)'),
                ('K+FB+FC', 'C3', '-',  'field K+FB+FC (complex)')]
    for case, col, ls, lab in fields_s:
        got = best_npz(a.results, case, a.ne)
        if got is None:
            continue
        npz, _ = got
        d = np.load(npz)
        if d['psi_s'].size:
            ax2[0, 1].plot(x, d['psi_s'].real, ls, color=col, lw=1.9, label=lab)
            ax2[1, 1].plot(x, d['psi_s'].imag, ls, color=col, lw=1.9, label=lab)

    ax2[0, 0].set_title(r'actuator  $\mathrm{Re}\,\psi_a$')
    ax2[0, 1].set_title(r'sensor  $\mathrm{Re}\,\psi_s$')
    ax2[1, 0].set_title(r'actuator  $\mathrm{Im}\,\psi_a$  '
                        r'(zero for the real-constrained cases)')
    ax2[1, 1].set_title(r'sensor  $\mathrm{Im}\,\psi_s$  '
                        r'(zero for the real-constrained cases)')
    for k in range(2):
        ax2[1, k].set_xlabel('x')
        ax2[k, 0].set_ylabel(r'$\psi$  (unit $L^2$)')
        for m in range(2):
            ax2[k, m].set_xlim(-15, 15); ax2[k, m].grid(alpha=0.3)
            ax2[k, m].axhline(0, color='gray', lw=0.6)
    ax2[0, 0].legend(fontsize=7, loc='upper left')
    ax2[0, 1].legend(fontsize=7, loc='upper right')
    fig2.tight_layout()
    for ext in ('png', 'pdf'):
        fig2.savefig(os.path.join(a.outdir, f'fig_fields.{ext}'), dpi=140)

    if warn:
        print("\n!! the certified peak lies OUTSIDE the stored omega window "
              "for these cases:")
        for c, gp, cm, w0, w1 in warn:
            print(f"   {c:12s} certified gamma={gp:.4f} but the curve on "
                  f"[{w0:+.1f},{w1:+.1f}] only reaches {cm:.4f}")
        print("   -> the drawn curve does not show the peak.  Re-run those "
              "cases (run_case stores the gain on [-4,4]) or widen it.")
    if a.show:
        if _BACKEND_NOTE:
            print(f"\n!! --show: {_BACKEND_NOTE}")
        else:
            try:
                plt.show()
            except Exception as _e:
                print(f"\n!! --show failed ({type(_e).__name__}: {_e}); "
                      f"figures were still written to disk.")
    print(f"\nwrote {a.outdir}/fig_sigma1.png|pdf and {a.outdir}/fig_fields.png|pdf")


if __name__ == "__main__":
    main()
