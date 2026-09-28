#!/usr/bin/env python3
"""mode_response.py -- how strongly the plant responds to a forcing field of
wavenumber k (2026-09-28, STAGE5 32.2: why viscosity does not regularise the
field design).

    python tools/mode_response.py --ne 200 [--omega -0.646 0 0.75]

For each Laplacian eigenmode phi_k (unit M-norm, wavenumber k = sqrt(lambda_k))
used as the actuator field, the state response u_k = (i w M - A)^{-1} M phi_k:
its energy ||u_k||_M, and its projection on the smooth start's sensor field
|c^H u_k|.  The plant is diffusive (gamma d_xx), so both decay like 1/k^2 for
k beyond the plant's natural wavenumbers: the forward map field -> response is
COMPACT.  Output: figs/mode_response.png and a table every ~0.5 in k."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fe_codesign import FECoDesign
import fieldbasis as FB

COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--omega', type=float, nargs='+', default=[-0.646, 0.0, 0.75])
    ap.add_argument('--sensor', default='results/K_FB_FC_s0_ne200.npz')
    ap.add_argument('--out', default='figs/mode_response.png')
    a = ap.parse_args()
    D = FECoDesign(ne=a.ne, r=4)
    Phi, lam = FB.laplace_basis(D, -38.9, 39.2, int(0.9 * D.n))
    k = np.sqrt(np.maximum(lam, 0))
    z = np.load(a.sensor, allow_pickle=True)
    vs = np.asarray(z['vs'], complex)
    if vs.size != D.n:
        xs_ = np.linspace(D.xn[0], D.xn[-1], vs.size)
        vs = np.interp(D.xn, xs_, vs.real) + 1j * np.interp(D.xn, xs_, vs.imag)
    c = D.M @ (vs / D.norm_M(vs))
    MPhi = (D.M @ Phi).astype(complex)
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.color': '#ecebe7', 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for j, w in enumerate(a.omega):
        lu = spla.splu((1j * w * D.M - D.A).tocsc().astype(complex))
        U = lu.solve(MPhi)
        en = np.sqrt(np.real(np.einsum('ij,ij->j', U.conj(), (D.M @ U))))
        out = np.abs(c.conj() @ U)
        ax[0].loglog(k[1:], en[1:], '.', ms=3, color=COL[j % 4], label=f'w = {w:+.3f}')
        ax[1].loglog(k[1:], out[1:], '.', ms=3, color=COL[j % 4], label=f'w = {w:+.3f}')
        print(f"w = {w:+.3f}:  k    ||u_k||_M    |c^H u_k|", flush=True)
        for kk in (0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12):
            i = int(np.argmin(np.abs(k - kk)))
            if abs(k[i] - kk) < 0.2:
                print(f"          {k[i]:5.2f}  {en[i]:10.3e}  {out[i]:10.3e}")
    kr = np.linspace(2, k.max(), 50)
    for i in range(2):
        ref = (ax[i].get_lines()[0].get_ydata()[np.argmin(np.abs(k[1:] - 4))]) * (4 / kr) ** 2
        ax[i].loglog(kr, ref, 'k--', lw=1, label='~ 1/k^2')
        ax[i].set_xlabel('wavenumber k of the actuator mode'); ax[i].legend(frameon=False, fontsize=8)
    ax[0].set_title('energy of the response to a unit forcing mode', loc='left', fontsize=10)
    ax[1].set_title('what the (smooth) sensor sees of it', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(a.out, dpi=140)
    print(a.out)


if __name__ == '__main__':
    main()
