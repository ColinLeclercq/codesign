#!/usr/bin/env python3
"""final_curves.py -- uncontrolled sigma_1 against the closed loop's sigma_1 and
sigma_2, and the final actuator/sensor fields, for the current best designs
(2026-09-27; STATUS.md).

    python tools/final_curves.py            # computes figs/final_curves.npz, then plots
    python tools/final_curves.py --plot     # plots from the saved npz

All values on the FULL model (no reduced model): sigma_k of the energy
resolvent M^{1/2} T M^{1/2}, T the closed-loop resolvent restricted to the
plant (ClosedLoopResolvent.gain_k).  'No control' = the plant alone (K = 0,
n = 0): the same curve for every design on a mesh, independent of b and c."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fe_codesign import FECoDesign

RECS = [('kbc100', 'K+B+C, ne=100', 100, 'results_n100_kbc/K_B_C_s0_ne100.npz'),
        ('kbc200', 'K+B+C, ne=200', 200, 'results_n200_kbc/K_B_C_s0_ne200.npz'),
        ('fbfcat', 'K+FB+FC, ne=200 (4 Gabor atoms per field)', 200, 'results_fb_atoms_n200_long/K_FB_FC_s0_ne200.npz')]
# 2026-09-28 (STAGE5 32): the K+FB+FC curve is the best atoms design (ne=200; 3.879283 on ne=800); the
# 80-mode ne=100 design it replaces was withdrawn (5.52 on ne=200, a discretisation artefact)
FLOOR = 3.509193          # sup_w sigma_2 of the open loop (tools/bounds.py, 1401-point grid)
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8983'
INK, INK2 = '#0b0b0b', '#52514e'
OUT = 'figs/final_curves.npz'
ws = np.unique(np.concatenate([np.arange(-30, -8, 0.1), np.arange(-8, 8.0001, 0.01), np.arange(8, 30.0001, 0.1)]))


def compute():
    out = dict(omega=ws)
    if os.path.exists(OUT):                      # keep what is already computed for the same grid and record
        old = dict(np.load(OUT))
        if np.array_equal(old.get('omega'), ws):
            out.update(old)
    for key, lab, ne, rec in RECS:
        if key + '_s1' in out and out.get(key + '_rec') == rec:
            continue
        d = np.load(rec, allow_pickle=True)
        D = FECoDesign(ne=ne, r=len(d['d']))
        aa = (d['d'], d['e'], d['f']); nn = np.asarray(d['nvec'], complex); KK = complex(d['K'])
        va = d['va'] if np.size(d['va']) else None
        vs = d['vs'] if np.size(d['vs']) else None
        b, c, *_ = D.bc(float(d['xa']), float(d['xs']), va, vs); cl = D.cl(b, c)
        s = np.array([[p['sigma'] for p in cl.gain_k(float(w), aa, nn, KK, k=2)] for w in ws])
        so = np.array([cl.gain(float(w), aa, np.zeros_like(nn), 0j, want_vectors=False)['sigma'] for w in ws])
        g = json.load(open(rec[:-4] + '.json'))['gamma']
        x = np.asarray(D.xn, float)
        fa = np.asarray(va if va is not None else D.gauss(float(d['xa'])), complex)
        fs = np.asarray(vs if vs is not None else D.gauss(float(d['xs'])), complex)
        out.update({key + '_s1': s[:, 0], key + '_s2': s[:, 1], key + '_ol': so, key + '_gamma': g,
                    key + '_x': x, key + '_fa': fa / D.norm_M(fa), key + '_fs': fs / D.norm_M(fs),
                    key + '_xa': float(d['xa']), key + '_xs': float(d['xs']), key + '_rec': rec})
        print(f"{lab}: max sigma_1 {s[:, 0].max():.6f} (record {g:.6f}); open loop max {so.max():.3f}; "
              f"closed-loop max sigma_2 {s[:, 1].max():.4f}", flush=True)
    np.savez(OUT, **out)


def style():
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb', 'lines.linewidth': 2})


def plot():
    style()
    z = np.load(OUT); w = z['omega']
    # 1. sigma curves: full band (log scale) and a linear zoom on the controlled peaks
    fig, axs = plt.subplots(len(RECS), 2, figsize=(13, 3.4 * len(RECS)), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for i, (key, lab, ne, rec) in enumerate(RECS):
        g = float(z[key + '_gamma'])
        for j, ax in enumerate(axs[i]):
            ax.plot(w, z[key + '_ol'], color=GRAY, lw=1.6, ls='--', label=r'$\sigma_1$, no control')
            ax.plot(w, z[key + '_s2'], color=ORANGE, lw=1.6, label=r'$\sigma_2$, controlled')
            ax.plot(w, z[key + '_s1'], color=BLUE, lw=2.0, label=r'$\sigma_1$, controlled')
            ax.axhline(FLOOR, color=INK2, lw=0.9, ls=':')
        a0, a1 = axs[i]
        a0.set_yscale('log'); a0.set_xlim(-30, 30); a0.set_ylim(0.05, 2e3)
        a0.set_title(f"{lab}: $\\gamma$ = {g:.4f}  (no control: {float(z[key + '_ol'].max()):.0f})",
                     loc='left', fontsize=10, color=INK)
        a1.set_xlim(-3, 3); a1.set_ylim(0, 1.25 * g)
        a1.set_title('zoom, linear scale; dotted: absolute floor %.3f (sup of the uncontrolled sigma_2)' % FLOOR,
                     loc='left', fontsize=9, color=INK2)
        a0.set_ylabel('energy-resolvent gain')
        if i == len(RECS) - 1:
            a0.set_xlabel(r'$\omega$'); a1.set_xlabel(r'$\omega$')
    axs[0, 0].legend(loc='upper right', frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig('figs/final_sigma.png', dpi=140)

    # 2. fields: K+B+C (Gaussians, both meshes) and K+FB+FC (free fields); unit M-norm,
    #    global phase fixed so the largest entry is real positive (the controller gauge absorbs it)
    fig, axs = plt.subplots(3, 2, figsize=(13, 8.5), sharex=True)
    for j, (sfx, nm) in enumerate((('_fa', 'actuator field b(x)'), ('_fs', 'sensor field c(x)'))):
        for key, lab, col, ls in (('kbc100', 'K+B+C ne=100', AQUA, '--'), ('kbc200', 'K+B+C ne=200', ORANGE, ':'),
                                  ('fbfcat', 'K+FB+FC ne=200 (atoms)', BLUE, '-')):
            x = z[key + '_x']; v = z[key + sfx]
            v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
            axs[0, j].plot(x, np.abs(v), color=col, ls=ls, lw=1.8, label=lab)
            axs[1, j].plot(x, v.real, color=col, ls=ls, lw=1.5)
            axs[2, j].plot(x, v.imag, color=col, ls=ls, lw=1.5)
        axs[0, j].set_title(nm + ' -- unit M-norm, phase: largest entry real', loc='left', fontsize=10, color=INK)
        axs[0, j].set_ylabel('|v|'); axs[1, j].set_ylabel('Re v'); axs[2, j].set_ylabel('Im v')
        axs[2, j].set_xlabel('x'); axs[2, j].set_xlim(-25, 25)
    axs[0, 0].legend(frameon=False, fontsize=9)
    xa, xs = float(z['kbc200_xa']), float(z['kbc200_xs'])
    fig.suptitle(f"Final fields.  K+B+C: Gaussians (width 0.4) at x_a = {xa:+.3f}, x_s = {xs:+.3f} (ne=200); "
                 f"K+FB+FC: 4 Gabor atoms per field (width >= 1, wavenumber <= 0.707), ne=200", x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig('figs/final_fields.png', dpi=140)
    print('figs/final_sigma.png, figs/final_fields.png')


if __name__ == '__main__':
    if '--plot' not in _s.argv:
        compute()
    plot()
