#!/usr/bin/env python3
"""mimo_plots.py -- gain curves, fields and controller responses of the
2-actuator / 2-sensor prototypes (tools/mimo_proto.py; STAGE5 34).

    python tools/mimo_plots.py            # computes figs/mimo_curves.npz, then plots
    python tools/mimo_plots.py --plot     # plots from the saved npz

Two families, each with three designs: the SISO optimum (loop 2 switched off),
decentralized (2 SISO loops) and centralized (2x2):
  gauss -- Gaussian actuators/sensors at free positions + dynamic compensator (K+B+C);
  atoms -- Gabor-atom fields + static gain (K+FB+FC).
All values on the full model (sparse LU + rank-4 Woodbury, M products only)."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fe_codesign import FECoDesign
import mimo_proto as MP
import fieldbasis as FB

FAM = {'gauss': ('K+B+C: Gaussian actuators/sensors, dynamic compensator',
                 [('siso', 'SISO optimum', 'results_mimo/siso_gauss.npz'),
                  ('dec', 'decentralized 2 x SISO', 'results_mimo/dec_p4b.npz'),
                  ('cen', 'centralized 2x2', 'results_mimo/cen3.npz')]),
       'atoms': ('K+FB+FC: Gabor-atom fields, static gain',
                 [('siso', 'SISO optimum', 'results_mimo/siso_atoms.npz'),
                  ('dec', 'decentralized 2 x SISO', 'results_mimo/atoms_dec.npz'),
                  ('cen', 'centralized 2x2', 'results_mimo/atoms_cen.npz')])}
COL = dict(siso='#1baf7a', dec='#eb6834', cen='#2a78d6')
GRAY, INK, INK2 = '#8a8983', '#0b0b0b', '#52514e'
OUT = 'figs/mimo_curves.npz'
ws = np.unique(np.concatenate([np.arange(-40, -8, 0.1), np.arange(-8, 8.0001, 0.01), np.arange(8, 40.0001, 0.1)]))
FLOOR2, FLOOR3 = 3.509145, 2.383533     # sup_w sigma_2, sigma_3 of the open loop at ne=100 (rank-1 / rank-2 floors)


def load(rec, D):
    z = np.load(rec, allow_pickle=True)
    fields = str(z['fields']) if 'fields' in z.files else 'gauss'
    na = int(z['na']) if 'na' in z.files else 4
    mm = MP.Mimo(D, int(z['R']), str(z['mode']), fields=fields, na=na)
    th = np.asarray(z['theta'], complex); pos = np.asarray(z['pos'], float)
    return mm, mm.build(th, pos), float(z['gamma']), pos


def compute():
    D = FECoDesign(ne=100, r=8); out = dict(omega=ws)
    # open loop: loop 2 off and loop 1 off = the SISO gauss record with Dc = 0, Cc = 0
    mm, S, _, _ = load(FAM['gauss'][1][0][2], D)
    S0 = dict(S); S0['Dc'] = 0 * S['Dc']; S0['Cc'] = 0 * S['Cc']
    out['ol'] = np.array([mm.sig(float(w), S0, k=3, vectors=False)[0]['sigma'] for w in ws])
    for fam, (_, recs) in FAM.items():
        for key, lab, rec in recs:
            if not _o.path.exists(rec):
                continue
            mm, S, g, pos = load(rec, D)
            s = np.array([[r['sigma'] for r in mm.sig(float(w), S, k=3, vectors=False)] for w in ws])
            k = f"{fam}_{key}"
            out[k + '_s'] = s; out[k + '_g'] = g
            x = np.asarray(D.xn, float); out[k + '_x'] = x
            # unit-M-norm fields: B = M psi -> psi from the stored raw fields / Gaussians
            if mm.fields == 'gauss':
                psis = [D.gauss(p) / D.norm_M(D.gauss(p)) for p in pos]
            else:
                psis = [v / nu for v, nu in zip(S['raw'], S['nu'])]
            out[k + '_psi'] = np.array(psis)
            # controller response K(i w) = D_c + C_c (i w - A_c)^-1 B_c (2x2), on a grid
            wk = np.linspace(-40, 40, 801); Kw = []
            for w in wk:
                Kc = S['Dc'].copy()
                if mm.R:
                    Kc = Kc + S['Cc'] @ np.linalg.solve(1j * w * np.eye(mm.R) - S['Ac'], S['Bc'])
                Kw.append(Kc)
            out[k + '_K'] = np.array(Kw); out[k + '_wk'] = wk; out[k + '_pos'] = pos
            print(f"{fam}/{key}: record {g:.6f}, grid max sigma_1 {s[:, 0].max():.6f}; D_c = {np.round(S['Dc'], 3).tolist()}", flush=True)
    np.savez(OUT, **out)


def style():
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb', 'lines.linewidth': 2})


def plot():
    style(); z = np.load(OUT); w = z['omega']
    # 1. gains: sigma_1 of the three designs (log, full band) and sigma_1..3 of the centralized one (linear zoom)
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for i, (fam, (title, recs)) in enumerate(FAM.items()):
        a0, a1 = axs[i]
        a0.semilogy(w, z['ol'], color=GRAY, ls='--', lw=1.5, label='no control')
        for key, lab, _ in recs:
            k = f"{fam}_{key}"
            if k + '_s' in z.files:
                a0.semilogy(w, z[k + '_s'][:, 0], color=COL[key], lw=1.8, label=f"{lab}: gamma {float(z[k + '_g']):.4f}")
        floors = ((FLOOR2, 'floor, 1 pair (open-loop sup sigma_2)'), (FLOOR3, 'floor, 2 pairs (open-loop sup sigma_3)'))
        for fl, nm in floors:
            a0.axhline(fl, color=INK2, lw=0.9, ls=':'); a1.axhline(fl, color=INK2, lw=0.9, ls=':')
        a0.set_xlim(-40, 40); a0.set_ylim(0.05, 2e3); a0.set_title(title, loc='left', fontsize=10, color=INK)
        a0.set_ylabel('energy-resolvent gain sigma_1'); a0.legend(frameon=False, fontsize=8, loc='upper right')
        k = f"{fam}_cen" if f"{fam}_cen_s" in z.files else f"{fam}_dec"
        if k + '_s' in z.files:
            for j, ls in enumerate(('-', '--', '-.')):
                a1.plot(w, z[k + '_s'][:, j], color=COL[k.split('_')[1]], ls=ls, lw=1.8 - 0.3 * j, label=f'sigma_{j + 1}')
            a1.plot(w, z['ol'], color=GRAY, ls='--', lw=1.2)
            a1.set_xlim(-3, 3); a1.set_ylim(0, 1.3 * float(z[k + '_g']))
            a1.set_title(f"{'centralized' if k.endswith('cen') else 'decentralized'}: sigma_1, sigma_2, sigma_3 (zoom)",
                         loc='left', fontsize=10, color=INK)
            a1.legend(frameon=False, fontsize=8, loc='lower left')
            for fl, nm in floors:
                if fl < a1.get_ylim()[1]:
                    a1.text(2.95, fl, nm, va='bottom', ha='right', fontsize=8, color=INK2, clip_on=True)
        if i == 1:
            a0.set_xlabel('w'); a1.set_xlabel('w')
    fig.tight_layout(); fig.savefig('figs/mimo_sigma.png', dpi=140)

    # 2. fields: actuators and sensors, |psi| (unit M-norm); pair 1 solid, pair 2 dashed
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 7.5), sharex=True)
    for i, (fam, (title, recs)) in enumerate(FAM.items()):
        for j, nm in enumerate(('actuators', 'sensors')):
            ax = axs[i, j]
            for key, lab, _ in recs:
                k = f"{fam}_{key}"
                if k + '_psi' not in z.files:
                    continue
                P = z[k + '_psi']; x = z[k + '_x']
                ax.plot(x, np.abs(P[2 * j]), color=COL[key], ls='-', lw=1.7, label=f"{lab}, pair 1")
                if key != 'siso':
                    ax.plot(x, np.abs(P[2 * j + 1]), color=COL[key], ls='--', lw=1.5, label=f"{lab}, pair 2")
            ax.set_title(f"{title.split(':')[0]} -- {nm}, |field| (unit M-norm)", loc='left', fontsize=10, color=INK)
            ax.set_xlim(-25, 25)
            if i == 1:
                ax.set_xlabel('x')
        axs[i, 0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout(); fig.savefig('figs/mimo_fields.png', dpi=140)

    # 3. controller responses |K_ij(i w)| of the K+B+C designs (the atoms ones are static gains)
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4), sharey=True)
    for ax, key in zip(axs, ('dec', 'cen')):
        k = f"gauss_{key}"
        if k + '_K' not in z.files:
            continue
        Kw, wk = z[k + '_K'], z[k + '_wk']
        pairs = ((0, 0), (1, 1)) if key == 'dec' else ((0, 0), (1, 1), (0, 1), (1, 0))   # decentral: K_12 = K_21 = 0
        for (a, b), ls in zip(pairs, ('-', '--', ':', '-.')):
            ax.semilogy(wk, np.maximum(np.abs(Kw[:, a, b]), 1e-6), color=COL[key], ls=ls, lw=1.6,
                        label=f"|K_{a + 1}{b + 1}|  (sensor {b + 1} -> actuator {a + 1})")
        ax.set_title(f"K+B+C {'decentralized (K_12 = K_21 = 0)' if key == 'dec' else 'centralized'}: compensator frequency response",
                     loc='left', fontsize=10, color=INK)
        ax.set_xlabel('w'); ax.legend(frameon=False, fontsize=8)
    axs[0].set_ylabel('|K_ij(i w)|')
    fig.tight_layout(); fig.savefig('figs/mimo_controller.png', dpi=140)
    print('figs/mimo_sigma.png, figs/mimo_fields.png, figs/mimo_controller.png')


if __name__ == '__main__':
    if '--plot' not in _s.argv:
        compute()
    plot()
