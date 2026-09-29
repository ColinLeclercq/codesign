#!/usr/bin/env python3
"""mimo_prod_plots.py -- gain curves, fields and compensator responses of the
production MIMO designs (2 actuators, 2 sensors, ne=100; STAGE6 6.6).

    python tools/mimo_prod_plots.py            # computes figs/mimo_prod_curves.npz, then plots
    python tools/mimo_prod_plots.py --plot     # plots from the saved npz

Per family (K+B+C: Gaussians at free positions + dynamic compensator; atoms:
Gabor-atom fields + static gains) three designs: the SISO optimum (loop 2
switched off: the start record), decentralized, centralized.  sigma_1..3 from
ClosedLoopResolventMIMO (sparse LU + Woodbury, M products only) on a grid that is
linear near the plant's band and logarithmic out to |w| = 3000 (the high-gain
branches of the static-gain designs resonate at |w| ~ 700-800), plus the records'
active frequencies.  The prototype's sigma_1 (tools/mimo_plots.py,
figs/mimo_curves.npz) is overlaid in the zoom for comparison."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fe_codesign import FECoDesign
import controller as ctl

P = 'results_mimo_prod/'
FAM = {'gauss': ('K+B+C: Gaussian actuators/sensors, dynamic compensator',
                 [('siso', 'SISO optimum', P + 'start_kbc_p4.npz'),
                  ('dec', 'decentralized 2 x SISO', P + 'kbc_dec/K_B_C_s0_ne100_mimo2x2_decentral.npz'),
                  ('cen', 'centralized 2x2', P + 'kbc_cen/K_B_C_s0_ne100_mimo2x2_central.npz')]),
       'atoms': ('K+FB+FC: Gabor-atom fields, static gains',
                 [('siso', 'SISO optimum', P + 'start_atoms.npz'),
                  ('dec', 'decentralized 2 x SISO', P + 'at_dec/KGC_FB_FC_s0_ne100_mimo2x2_decentral.npz'),
                  ('cen', 'centralized 2x2', P + 'at_cen/KGC_FB_FC_s0_ne100_mimo2x2_central.npz')])}
COL = dict(siso='#1baf7a', dec='#eb6834', cen='#2a78d6')
GRAY, INK, INK2 = '#8a8983', '#0b0b0b', '#52514e'
OUT = 'figs/mimo_prod_curves.npz'
FLOOR2, FLOOR3 = 3.509145, 2.383533     # sup_w sigma_2, sigma_3 of the open loop at ne=100 (STAGE5 34.3)
_far = np.geomspace(40, 3000, 400)
WS = np.unique(np.concatenate([-_far[::-1], np.arange(-40, -8, 0.1), np.arange(-8, 8.0001, 0.01),
                               np.arange(8, 40.0001, 0.1), _far]))


def design(rec, D):
    z = np.load(rec, allow_pickle=True)
    va = [np.asarray(v) for v in z['va']] if z['va'].size else None
    vs = [np.asarray(v) for v in z['vs']] if z['vs'].size else None
    b, c, pa, ps, *_ = D.bc(np.asarray(z['xa'], float), np.asarray(z['xs'], float), va, vs)
    arow = tuple(np.asarray(z[k], complex) for k in ('d', 'e', 'f')) + (np.asarray(z['Bc'], complex),)
    # a start record's gamma is its SISO source's (another mesh for the atoms): not used
    g = None if _o.path.basename(rec).startswith('start_') else float(z['gamma'])
    af = np.asarray(z['active_freqs'], float) if 'active_freqs' in z.files else np.zeros(0)
    return dict(b=b, c=c, pa=pa, ps=ps, arow=arow, nvec=np.asarray(z['nvec'], complex),
                K=np.asarray(z['K'], complex), gamma=g, af=af,
                xa=np.asarray(z['xa'], float), xs=np.asarray(z['xs'], float))


def compute():
    out = {}
    for fam, (_, recs) in FAM.items():
        for key, lab, rec in recs:
            if not _o.path.exists(rec):
                continue
            r = len(np.load(rec)['d']); D = FECoDesign(ne=100, r=r)
            S = design(rec, D); cl = D.cl(S['b'], S['c'])
            ws = np.unique(np.concatenate([WS, S['af']]))
            s = np.array([[q['sigma'] for q in cl.gain_k(float(w), S['arow'], S['nvec'], S['K'], k=3)] for w in ws])
            k = f"{fam}_{key}"
            out[k + '_w'] = ws; out[k + '_s'] = s; out[k + '_x'] = np.asarray(D.xn, float)
            out[k + '_psi_a'] = S['pa'].T; out[k + '_psi_s'] = S['ps'].T          # unit M-norm columns (FECoDesign.bc)
            out[k + '_g'] = S['gamma'] if S['gamma'] is not None else float(s[:, 0].max())   # certified, else the grid max
            out[k + '_Dc'] = S['K']
            wk = np.concatenate([-np.geomspace(1e-2, 1e3, 300)[::-1], np.geomspace(1e-2, 1e3, 300)])
            Ac = ctl.A_of(*S['arow'][:3]); Bc = np.asarray(S['arow'][3]).reshape(r, -1)
            out[k + '_wk'] = wk
            out[k + '_Kw'] = np.array([S['K'] + S['nvec'] @ np.linalg.solve(1j * w * np.eye(r) - Ac, Bc) for w in wk])
            if key == 'siso' and fam == 'gauss':
                zer = np.zeros_like(S['K'])
                out['ol_w'] = WS
                out['ol'] = np.array([[q['sigma'] for q in cl.gain_k(float(w), S['arow'], 0 * S['nvec'], zer, k=3)]
                                      for w in WS])
            print(f"{fam}/{key}: record {S['gamma']}, grid max sigma_1 {s[:, 0].max():.6f} at w = "
                  f"{ws[np.argmax(s[:, 0])]:+.3f}; D_c = {np.round(S['K'], 3).tolist()}", flush=True)
    np.savez(OUT, **out)


def style():
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb', 'lines.linewidth': 2})


def plot():
    style(); z = np.load(OUT)
    proto = np.load('figs/mimo_curves.npz') if _o.path.exists('figs/mimo_curves.npz') else None
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.2), gridspec_kw=dict(width_ratios=[1.45, 1]))
    for i, (fam, (title, recs)) in enumerate(FAM.items()):
        a0, a1 = axs[i]
        a0.plot(z['ol_w'], z['ol'][:, 0], color=GRAY, ls='--', lw=1.4, label='no control')
        best = None
        for key, lab, _ in recs:
            k = f"{fam}_{key}"
            if k + '_s' not in z.files:
                continue
            a0.plot(z[k + '_w'], z[k + '_s'][:, 0], color=COL[key], lw=1.7, label=f"{lab}: gamma {float(z[k + '_g']):.4f}")
            if key != 'siso' and (best is None or float(z[k + '_g']) < float(z[best + '_g'])):
                best = k
        a0.set_xscale('symlog', linthresh=10); a0.set_yscale('log')
        a0.set_xlim(-3000, 3000); a0.set_ylim(0.02, 2e3)
        for fl in (FLOOR2, FLOOR3):
            a0.axhline(fl, color=INK2, lw=0.9, ls=':'); a1.axhline(fl, color=INK2, lw=0.9, ls=':')
        a0.set_title(title, loc='left', fontsize=10, color=INK)
        a0.set_ylabel('energy-resolvent gain sigma_1'); a0.legend(frameon=False, fontsize=8, loc='upper right')
        if best is not None:
            for j, ls in enumerate(('-', '--', '-.')):
                a1.plot(z[best + '_w'], z[best + '_s'][:, j], color=COL[best.split('_')[1]], ls=ls, lw=1.8 - 0.3 * j,
                        label=f'sigma_{j + 1} ({best.split("_")[1]})')
            if proto is not None and f"{fam}_{best.split('_')[1]}_s" in proto.files:
                pk_ = f"{fam}_{best.split('_')[1]}"
                a1.plot(proto['omega'], proto[pk_ + '_s'][:, 0], color=INK, ls=':', lw=1.2,
                        label=f"prototype sigma_1 (gamma {float(proto[pk_ + '_g']):.4f})")
            a1.plot(z['ol_w'], z['ol'][:, 0], color=GRAY, ls='--', lw=1.2)
            a1.set_xlim(-3, 3); a1.set_ylim(0, 1.35 * float(z[best + '_g']))
            a1.set_title(f"best MIMO design: sigma_1..3 near the plant's band (zoom)", loc='left', fontsize=10, color=INK)
            a1.legend(frameon=False, fontsize=8, loc='lower left')
            for fl, nm in ((FLOOR2, 'floor, 1 pair'), (FLOOR3, 'floor, 2 pairs')):
                a1.text(2.95, fl, nm, va='bottom', ha='right', fontsize=8, color=INK2)
        if i == 1:
            a0.set_xlabel('w (symlog)'); a1.set_xlabel('w')
    fig.tight_layout(); fig.savefig('figs/mimo_prod_sigma.png', dpi=140)

    fig, axs = plt.subplots(2, 2, figsize=(14, 7.5), sharex=True)
    for i, (fam, (title, recs)) in enumerate(FAM.items()):
        for j, (side, nm) in enumerate((('psi_a', 'actuators'), ('psi_s', 'sensors'))):
            ax = axs[i, j]
            for key, lab, _ in recs:
                k = f"{fam}_{key}"
                if k + '_' + side not in z.files:
                    continue
                Pm = z[k + '_' + side]; x = z[k + '_x']
                ax.plot(x, np.abs(Pm[0]), color=COL[key], ls='-', lw=1.7, label=f"{lab}, port 1")
                if key != 'siso':
                    ax.plot(x, np.abs(Pm[1]), color=COL[key], ls='--', lw=1.5, label=f"{lab}, port 2")
            ax.set_title(f"{title.split(':')[0]} -- {nm}, |field| (unit M-norm)", loc='left', fontsize=10, color=INK)
            ax.set_xlim(-25, 25)
            if i == 1:
                ax.set_xlabel('x')
        axs[i, 0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout(); fig.savefig('figs/mimo_prod_fields.png', dpi=140)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, key in zip(axs, ('dec', 'cen')):
        k = f"gauss_{key}"
        if k + '_Kw' not in z.files:
            continue
        Kw, wk = z[k + '_Kw'], z[k + '_wk']
        pairs = ((0, 0), (1, 1)) if key == 'dec' else ((0, 0), (1, 1), (0, 1), (1, 0))
        for (a, b), ls in zip(pairs, ('-', '--', ':', '-.')):
            ax.plot(wk, np.maximum(np.abs(Kw[:, a, b]), 1e-8), color=COL[key], ls=ls, lw=1.6,
                    label=f"|K_{a + 1}{b + 1}|  (sensor {b + 1} -> actuator {a + 1})")
        ax.set_xscale('symlog', linthresh=1); ax.set_yscale('log')
        ax.set_title(f"K+B+C {'decentralized (K_12 = K_21 = 0)' if key == 'dec' else 'centralized'}: compensator K(i w)",
                     loc='left', fontsize=10, color=INK)
        ax.set_xlabel('w (symlog)'); ax.legend(frameon=False, fontsize=8)
    axs[0].set_ylabel('|K_ij(i w)|')
    fig.tight_layout(); fig.savefig('figs/mimo_prod_controller.png', dpi=140)
    print('figs/mimo_prod_sigma.png, figs/mimo_prod_fields.png, figs/mimo_prod_controller.png')


if __name__ == '__main__':
    if '--plot' not in _s.argv:
        compute()
    plot()
