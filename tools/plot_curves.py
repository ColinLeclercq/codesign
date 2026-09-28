import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, YELLOW, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a8983'
INK, INK2 = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
z = np.load('figs/curves.npz'); w = z['omega']; FLOOR = 3.509066
cases = [('kbc200_frozen', 'K+B+C, ne=200, frozen design'), ('kbc200', 'K+B+C, ne=200, new'),
         ('kbc100', 'K+B+C, ne=100, new'), ('fbfc100', 'K+FB+FC, ne=100, new (80 Laplacian modes per field)')]
fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
for ax, (lab, title) in zip(axs.ravel(), cases):
    g = float(z[lab + '_gamma'])
    ax.plot(w, z[lab + '_ol2'], color=GRAY, lw=1.2, ls='--', label=r'open-loop $\sigma_2$ (pointwise floor)')
    ax.plot(w, z[lab + '_s2'], color=ORANGE, lw=1.3, label=r'closed-loop $\sigma_2$')
    ax.plot(w, z[lab + '_s1'], color=BLUE, lw=1.6, label=r'closed-loop $\sigma_1$')
    ax.axhline(FLOOR, color=GRAY, lw=0.8, ls=':')
    ax.set_title(f"{title}:  $\\gamma$ = {g:.4f}  (floor {FLOOR:.3f}, +{100*(g/FLOOR-1):.0f} %)", fontsize=10, color=INK, loc='left')
    ax.set_xlim(-30, 30); ax.set_ylim(0, 8.6)
for ax in axs[1]: ax.set_xlabel(r'$\omega$')
for ax in axs[:, 0]: ax.set_ylabel('energy-resolvent gain')
axs[0, 0].legend(loc='upper right', frameon=False, fontsize=9)
fig.suptitle(r'Closed-loop $\sigma_1$, $\sigma_2$ over the certified band $[-30, 30]$ (dotted: absolute floor $\sup_\omega \sigma_2^{ol}$ = 3.509)', color=INK, x=0.01, ha='left')
fig.tight_layout(); fig.savefig('figs/sigma12.png', dpi=140)

# zoom on the core
fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for ax, (lab, title) in zip(axs.ravel(), cases):
    g = float(z[lab + '_gamma'])
    ax.plot(w, z[lab + '_ol2'], color=GRAY, lw=1.2, ls='--', label=r'open-loop $\sigma_2$')
    ax.plot(w, z[lab + '_s2'], color=ORANGE, lw=1.3, label=r'closed-loop $\sigma_2$')
    ax.plot(w, z[lab + '_s1'], color=BLUE, lw=1.6, label=r'closed-loop $\sigma_1$')
    ax.set_xlim(-3, 3); ax.set_ylim(max(0, 0.8 * FLOOR) if 'fbfc' in lab else 2.5, g * 1.03)
    ax.set_title(f"{title}:  $\\gamma$ = {g:.4f}", fontsize=10, color=INK, loc='left')
for ax in axs[1]: ax.set_xlabel(r'$\omega$')
axs[0, 0].legend(loc='lower left', frameon=False, fontsize=9)
fig.suptitle(r'Zoom on $|\omega| \leq 3$ (y-axis per panel)', color=INK, x=0.01, ha='left')
fig.tight_layout(); fig.savefig('figs/sigma12_zoom.png', dpi=140)

# controller frequency response
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
cols = {'kbc200_frozen': ORANGE, 'kbc200': BLUE, 'kbc100': AQUA, 'fbfc100': YELLOW}
names = {'kbc200_frozen': 'K+B+C ne=200 frozen', 'kbc200': 'K+B+C ne=200 new', 'kbc100': 'K+B+C ne=100 new', 'fbfc100': 'K+FB+FC ne=100 new'}
for lab in ['kbc200_frozen', 'kbc200', 'kbc100', 'fbfc100']:
    K = z[lab + '_K']
    a1.semilogy(w, np.abs(K), color=cols[lab], lw=1.4, label=names[lab])
    a2.plot(w, np.degrees(np.unwrap(np.angle(K))), color=cols[lab], lw=1.4)
a1.set_ylabel(r'$|k(i\omega)|$'); a2.set_ylabel('phase (deg, unwrapped)'); a2.set_xlabel(r'$\omega$')
a1.legend(frameon=False, fontsize=9, ncol=4, loc='upper left')
a1.set_title(r'Compensator $k(s) = K + n^\top (sI - A_c)^{-1} B_c$ on $s = i\omega$ (complex coefficients: not symmetric in $\omega$)', color=INK, loc='left', fontsize=10)
fig.tight_layout(); fig.savefig('figs/controller_bode.png', dpi=140)

# gamma vs wall time
fig, ax = plt.subplots(figsize=(11, 4.8))
for lab, col, name in [('ne200', BLUE, 'K+B+C ne=200 (from the frozen 7.964)'), ('ne100', AQUA, 'K+B+C ne=100 (from 10.99)')]:
    T, G = np.load(f'figs/hist_{lab}.npy')
    ax.plot(T / 60, G, color=col, lw=1.6, marker='o', ms=3, label=name)
    ax.axhspan(G.min(), G.min() * 1.01, color=col, alpha=0.10, lw=0)
ax.set_xlabel('wall time (min, two or three runs sharing two cores)'); ax.set_ylabel(r'certified $\gamma$')
ax.set_ylim(6.2, 8.1); ax.legend(frameon=False)
ax.set_title(r'Certified $\gamma$ per outer round (shaded: within 1 % of the end value)', color=INK, loc='left', fontsize=10)
fig.tight_layout(); fig.savefig('figs/gamma_vs_time.png', dpi=140)
print('ok')
