#!/usr/bin/env python3
"""band_study.py -- closed-loop sigma_1 over a wide band for several designs
(2026-09-28, STAGE5 32.3): how far the equalised (flat) band extends, against
the wavenumber content of the fields.

    python tools/band_study.py out_prefix label:ne:record ...

Also prints, for each design, the band where sigma_1 >= 0.95 gamma and the
GL dispersion wavenumber of its edges (mode e^{ikx}: w = k^2 - 2k)."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fe_codesign import FECoDesign
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']


def main():
    pre = _s.argv[1]; specs = [a.split(':', 2) for a in _s.argv[2:]]
    ws = np.unique(np.concatenate([np.linspace(-80, 80, 1601), np.linspace(-3, 3, 301)]))
    fig, ax = plt.subplots(figsize=(12, 4.2)); rows = []
    for j, (lab, ne, rec) in enumerate(specs):
        d = np.load(rec, allow_pickle=True); ne = int(ne)
        D = FECoDesign(ne=ne, r=len(d['d']))
        aa = (d['d'], d['e'], d['f']); nn = np.asarray(d['nvec'], complex); KK = complex(d['K'])
        va = np.asarray(d['va'], complex); vs = np.asarray(d['vs'], complex)
        b, c, *_ = D.bc(float(d['xa']), float(d['xs']), va, vs); cl = D.cl(b, c)
        s = np.array([cl.gain(float(w), aa, nn, KK, want_vectors=False)['sigma'] for w in ws])
        g = s.max(); hi = ws[s >= 0.95 * g]
        kk = lambda w: 1 + np.sqrt(1 + abs(w))
        print(f"{lab}: max sigma_1 {g:.5f}; sigma_1 >= 0.95 max on w in [{hi.min():+.2f}, {hi.max():+.2f}] "
              f"(|k| at the edges ~ {kk(hi.min()):.2f}, {kk(hi.max()):.2f}); sigma_1 at w = +-40: {s[np.argmin(abs(ws + 40))]:.3f}, "
              f"{s[np.argmin(abs(ws - 40))]:.3f}; at +-80: {s[0]:.3f}, {s[-1]:.3f}; |K| = {abs(KK):.3g}", flush=True)
        ax.semilogy(ws, s, color=COL[j % 6], lw=1.5, label=f"{lab} (max {g:.4f})")
        rows.append(dict(label=lab, gamma_grid=float(g), band=[float(hi.min()), float(hi.max())], K=abs(KK)))
    ax.set_xlabel('w'); ax.set_ylabel('closed-loop sigma_1'); ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color='#ecebe7'); fig.tight_layout(); fig.savefig(pre + '.png', dpi=140)
    json.dump(rows, open(pre + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
