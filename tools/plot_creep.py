#!/usr/bin/env python3
"""plot_creep.py -- best certified gamma vs wall time, ne=100 K+B+C runs v4/v5/v6 (STAGE5 29; figs/gamma_vs_time_n100_creep.png); argument 200: ne=200 v3/v4 (figs/gamma_vs_time_n200.png)."""
import re, glob, os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
runs = [('v4 (28)', 'runs_old/results_n100_kbc_17_v4_6.410499', '#c0392b'),
        ('v5 (29.1, governor fix)', 'runs_old/results_n100_kbc_18_v5_tausplit_6.410212', '#e69f00'),
        ('v6 (29.2, face entry + shared tops)', 'runs_old/results_n100_kbc_19_v6_6.404234', '#0072b2')]
import sys
if len(sys.argv) > 1 and sys.argv[1] == 'loose':
    # STAGE5 30: the loose-ends session's code (run with --stop-progress 3:0.001) against the previous results
    runs = [('ne=100 v6 (29)', 'runs_old/results_n100_kbc_19_v6_6.404234', '#e69f00'),
            ('ne=100 v8 (30)', 'runs_old/results_n100_kbc_22_v8_stop001_6.389368', '#0072b2'),
            ('ne=200 v4 (29)', 'runs_old/results_n200_kbc_9_v4c_6.486694', '#c0392b'),
            ('ne=200 v5 (30)', 'runs_old/results_n200_kbc_11_v5_stop001_6.484857', '#009e73')]
elif len(sys.argv) > 1 and sys.argv[1] == '200':
    runs = [('ne=200 v3 (28)', 'runs_old/results_n200_kbc_6_v3_6.485693', '#c0392b'),
            ('ne=200 v4 (29)', 'runs_old/results_n200_kbc_9_v4c_6.486694', '#0072b2')]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for lab, d, col in runs:
    t0 = int(re.search(r'=== launch (\d+)', open(f'{d}/run.log').read()).group(1))
    dd = glob.glob(f'{d}/dump_*')[0]
    cert = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r'^  outer (\d+):.*?best=([0-9.]+)', open(f'{d}/run.log').read(), re.M)}
    ts = {int(os.path.basename(f)[6:9]): os.path.getmtime(f) - t0 for f in glob.glob(f'{dd}/outer_*.npz')}
    ks = sorted(k for k in cert if k in ts)
    x = np.array([ts[k] for k in ks]) / 60; y = np.array([cert[k] for k in ks])
    for a in ax: a.plot(x, y, '-o', ms=2.5, color=col, label=f'{lab}: {y[-1]:.4f}')
for a in ax:
    if not (len(sys.argv) > 1 and sys.argv[1] == 'loose'):
        a.axhline(6.438335 if len(sys.argv) > 1 else 6.400015, color='k', lw=0.8, ls='--', label='best known ' + ('6.438335' if len(sys.argv) > 1 else '6.400015'))
    a.set_xlabel('wall time [min]'); a.grid(alpha=0.3)
ax[0].set_ylabel('best certified gamma'); ax[0].set_title({'200': 'ne=200 K+B+C', 'loose': 'K+B+C, previous vs current code'}.get(sys.argv[1] if len(sys.argv) > 1 else '', 'ne=100 K+B+C: the creep at 7.93 and its fixes')); ax[0].legend(fontsize=8)
ax[1].set_ylim((6.38, 6.6) if (len(sys.argv) > 1 and sys.argv[1] == 'loose') else ((6.42, 6.8) if len(sys.argv) > 1 else (6.38, 6.8))); ax[1].set_title('tail')
fig.tight_layout(); fig.savefig({'200': 'figs/gamma_vs_time_n200.png', 'loose': 'figs/gamma_vs_time_loose_ends.png'}.get(sys.argv[1] if len(sys.argv) > 1 else '', 'figs/gamma_vs_time_n100_creep.png'), dpi=140); print('ok')
