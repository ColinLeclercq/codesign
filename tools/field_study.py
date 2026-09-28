#!/usr/bin/env python3
"""field_study.py -- compare K+FB+FC designs across meshes and parametrisations
(2026-09-28, STAGE5 32): certified gamma, RMS wavenumber of each field, the
energy spectrum of each field on the Laplacian eigenbasis of its own mesh (in
eigenvalue order), and the fields themselves.

    python tools/field_study.py out_prefix  label:ne:record ...

Spectrum: |beta_k|^2 / sum |beta|^2 against the wavenumber k = sqrt(lambda_k),
binned in k (0.25 wide), so meshes and bases of different sizes are comparable."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fe_codesign import FECoDesign
import fieldbasis as FB

COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2 = '#0b0b0b', '#52514e'


def load(ne, rec):
    d = np.load(rec, allow_pickle=True)
    D = FECoDesign(ne=ne, r=len(d['d']))
    va = np.asarray(d['va'], complex); vs = np.asarray(d['vs'], complex)
    g = json.load(open(rec[:-4] + '.json'))['gamma'] if _o.path.exists(rec[:-4] + '.json') else float(d['gamma'])
    return D, va, vs, g


def main():
    pre = _s.argv[1]
    specs = [a.split(':', 2) for a in _s.argv[2:]]
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.grid': True, 'grid.color': '#ecebe7', 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
                         'savefig.facecolor': '#fcfcfb'})
    fs, ax_s = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    ff, ax_f = plt.subplots(2, 2, figsize=(13, 6.5), sharex=True)
    edges = np.arange(0, 8.01, 0.25)
    rows = []
    for j, (lab, ne, rec) in enumerate(specs):
        D, va, vs, g = load(int(ne), rec)
        Nb = min(D.n - 2, int(0.9 * D.n))
        Phi, lam = FB.laplace_basis(D, -38.9, 39.2, Nb)
        x = np.asarray(D.xn)
        out = [lab, int(ne), g]
        for i, (v, nm) in enumerate(((va, 'actuator'), (vs, 'sensor'))):
            v = v / D.norm_M(v)
            R = float(np.real(np.vdot(v, D.Kstiff @ v)) / np.real(np.vdot(v, D.M @ v)))
            beta = Phi.T @ (D.M @ v); e = np.abs(beta) ** 2 / np.sum(np.abs(beta) ** 2)
            k = np.sqrt(np.maximum(lam, 0))
            hist = np.array([e[(k >= edges[m]) & (k < edges[m + 1])].sum() for m in range(len(edges) - 1)])
            ax_s[i].semilogy(0.5 * (edges[1:] + edges[:-1]), np.maximum(hist, 1e-12), '-o', ms=3, color=COL[j % 8], lw=1.6,
                             label=f"{lab}: gamma {g:.4f}, RMS k {np.sqrt(R):.2f}")
            ax_s[i].set_title(f"{nm} field: energy per wavenumber bin (Laplacian eigenbasis of its mesh)", loc='left',
                              fontsize=10, color=INK)
            vv = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
            ax_f[0, i].plot(x, np.abs(vv), color=COL[j % 8], lw=1.5, label=lab)
            ax_f[1, i].plot(x, vv.imag, color=COL[j % 8], lw=1.3)
            ax_f[0, i].set_title(f"{nm} field |v| (unit M-norm)", loc='left', fontsize=10, color=INK)
            ax_f[1, i].set_title(f"{nm} field Im v (phase: largest entry real)", loc='left', fontsize=10, color=INK)
            out += [np.sqrt(R), float(e[k > 3].sum())]
        rows.append(out)
        print(f"{lab}: ne={ne} gamma {g:.6f}; RMS wavenumber actuator {out[3]:.3f}, sensor {out[5]:.3f}; "
              f"energy at k > 3: {out[4]:.2e} / {out[6]:.2e}", flush=True)
    for i in range(2):
        ax_s[i].set_xlabel('wavenumber k = sqrt(lambda)'); ax_f[1, i].set_xlabel('x'); ax_f[1, i].set_xlim(-25, 25)
    ax_s[0].set_ylabel('energy fraction'); ax_s[0].legend(frameon=False, fontsize=8); ax_f[0, 0].legend(frameon=False, fontsize=8)
    fs.tight_layout(); fs.savefig(pre + '_spectra.png', dpi=140)
    ff.tight_layout(); ff.savefig(pre + '_fields.png', dpi=140)
    json.dump(rows, open(pre + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
