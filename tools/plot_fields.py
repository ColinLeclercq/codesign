import sys, os; sys.path.insert(0, '.'); import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from fe_codesign import FECoDesign
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#8a8983', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
recs = [('K+FB+FC ne=100, new (3.9012)', 100, 'results_n100_fb80/K_FB_FC_s0_ne100_ckpt.npz', BLUE),
        ('K+FB+FC ne=200, frozen (3.9254)', 200, 'results/K_FB_FC_s0_ne200.npz', ORANGE)]
kbc = [('K+B+C ne=200 new, Gaussians at -2.335 / +2.335', 200, 'results_n200_kbc_cont/K_B_C_s0_ne200.npz', AQUA)]
fig, axs = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
for lab, ne, f, col in recs + kbc:
    D = FECoDesign(ne=ne, r=8); d = np.load(f, allow_pickle=True); x = np.asarray(D.xn)
    va = d['va'] if np.size(d['va']) else D.gauss(float(d['xa']))
    vs = d['vs'] if np.size(d['vs']) else D.gauss(float(d['xs']))
    for j, (v, nm) in enumerate(((va, 'actuator field (FB)'), (vs, 'sensor field (FC)'))):
        v = np.asarray(v, complex) / D.norm_M(v)
        # fix the global phase (the controller gauge absorbs it): largest entry real positive
        v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
        ls = '--' if 'K+B+C' in lab else '-'
        axs[0, j].plot(x, np.abs(v), color=col, lw=1.5, ls=ls, label=lab)
        axs[1, j].plot(x, v.real, color=col, lw=1.3, ls=ls)
        axs[2, j].plot(x, v.imag, color=col, lw=1.3, ls=ls)
        axs[0, j].set_title(nm + ' (unit M-norm, phase gauge: largest entry real)', loc='left', fontsize=10, color=INK)
for j in range(2):
    axs[0, j].set_ylabel('|v|'); axs[1, j].set_ylabel('Re v'); axs[2, j].set_ylabel('Im v'); axs[2, j].set_xlabel('x')
    axs[2, j].set_xlim(-25, 25)
axs[0, 0].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig('figs/fields.png', dpi=140); print('ok')
