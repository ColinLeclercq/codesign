"""
plot_symmetry.py — the figure that actually SHOWS the mirror symmetry.

fig_fields plots Re psi_a and Re psi_s in separate side-by-side panels, and
the symmetry psi_s(x) = conj(psi_a(-x)) relates them through a left-right
FLIP plus a SIGN FLIP on Im -- a relation no eye can verify across two
panels, and one that is visually drowned when the genuinely asymmetric
fixed-compensator case (FB+FC) shares the axes.  Each field individually is
NOT symmetric about x = 0 and should not look it.

This figure overlays, per case, psi_s(x) as lines against conj(psi_a(-x)) as
markers: for the gauge-fixed free-gain optima the curves coincide to
roundoff (printed per case), which IS the symmetry, visibly.  A residual
panel shows |psi_s(x) - conj(psi_a(-x))| on a log scale, where the exact
cases sit at ~1e-12 and FB+FC / KSP+FB+FC display their genuine residual.

    python plot_symmetry.py --ne 200 --results results
    -> fig_symmetry.png / .pdf
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_codesign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results')
    ap.add_argument('--r', type=int, default=-1)
    ap.add_argument('--sob_len', type=float, default=0.8)
    ap.add_argument('--cases', default='K+FB+FC,KGC+FB+FC,KSP+FB+FC,FB+FC')
    ap.add_argument('--out', default='fig_symmetry')
    a = ap.parse_args()
    if a.r < 0:
        from ctrl_order import infer_r
        a.r = infer_r(a.ne, getattr(a, 'results', None))
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=a.sob_len)
    x = D.xn
    want = [c.strip() for c in a.cases.split(',')]
    data = {}
    for jf in sorted(glob.glob(os.path.join(a.results, f"*_ne{a.ne}.json"))):
        m = json.load(open(jf))
        if m.get('case') not in want:
            continue
        d = np.load(jf[:-5] + '.npz', allow_pickle=True)
        if 'psi_a' in d.files and d['psi_a'].size and d['psi_s'].size:
            data[m['case']] = (np.asarray(d['psi_a'], complex),
                               np.asarray(d['psi_s'], complex),
                               float(m['gamma']))
    ncase = len(data)
    if not ncase:
        print("no matching cases with fields"); return
    fig, ax = plt.subplots(3, ncase, figsize=(4.2 * ncase, 8.4),
                           sharex=True, squeeze=False)
    for j, c in enumerate([c for c in want if c in data]):
        pa, ps, gam = data[c]
        mir = np.conj(pa[::-1])                 # conj(psi_a(-x))
        res = float(np.linalg.norm(ps - mir) / np.linalg.norm(ps))
        print(f"{c:12s} gamma={gam:.4f}   ||psi_s - conj(R psi_a)|| / "
              f"||psi_s|| = {res:.2e}")
        th = max(1, len(x) // 60)
        ax[0, j].plot(x, ps.real, '-', color='C0', lw=1.8,
                      label=r'$\mathrm{Re}\,\psi_s(x)$')
        ax[0, j].plot(x[::th], mir.real[::th], 'o', color='C3', ms=3.2,
                      mfc='none', label=r'$\mathrm{Re}\,\overline{\psi_a}(-x)$')
        ax[1, j].plot(x, ps.imag, '-', color='C0', lw=1.8,
                      label=r'$\mathrm{Im}\,\psi_s(x)$')
        ax[1, j].plot(x[::th], mir.imag[::th], 'o', color='C3', ms=3.2,
                      mfc='none', label=r'$\mathrm{Im}\,\overline{\psi_a}(-x)$')
        ax[2, j].semilogy(x, np.maximum(np.abs(ps - mir), 1e-16),
                          '-', color='C2', lw=1.6)
        ax[2, j].set_ylim(1e-16, 3.0)
        ax[0, j].set_title(f"{c}   $\\gamma$={gam:.4f}   res={res:.1e}")
        ax[2, j].set_xlabel('$x$')
        if j == 0:
            ax[0, j].legend(fontsize=8, loc='best')
            ax[1, j].legend(fontsize=8, loc='best')
            ax[2, j].set_ylabel(r'$|\psi_s(x)-\overline{\psi_a}(-x)|$')
    fig.suptitle('mirror symmetry of the optimised fields:  '
                 r'$\psi_s(x)$ (lines)  vs  $\overline{\psi_a}(-x)$ '
                 '(markers) -- coincidence IS the symmetry', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ('png', 'pdf'):
        fig.savefig(f"{a.out}.{ext}", dpi=180)
    print(f"wrote {a.out}.png / .pdf")


if __name__ == "__main__":
    main()
