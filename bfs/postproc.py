#!/usr/bin/env python3
"""postproc.py -- what a backward-facing-step result looks like (BFS.md 7):

  sigma.png        sigma_1, sigma_2, sigma_3 of the noise-to-state map on
                   [0, W] (W = the run's final band edge, at least 10), without
                   control (open loop) and with it (closed loop), and the
                   record's certified gamma as a horizontal line
  modes_w<w>.png   at each of a common set of frequencies, the optimal forcing
                   (right singular vector of sigma_1) and the optimal response
                   (left singular vector), without and with control, with the
                   actuator (half-amplitude circle of its Gaussian) and the
                   sensor (its window on the floor) drawn on every panel
  controller.png   Bode plot of the controller k(s) = K + n^T (sI - A_c)^-1 B_c
  post.npz         every number and field behind the figures (below)
  post.json        a summary (peaks, frequencies, timings)

The singular values come with no inverse of the mass matrix (note_algebra,
BFS section): the forcing is a velocity field f with ||f||^2 = f^H M f and
enters in weak form, the response is q = T M f with ||q||^2 = q^H M q,
T = (i w M - A)^-1 (open loop) or the plant block of the closed-loop
resolvent (Woodbury, closedloop_sparse.ClosedLoopResolvent).  Then

    sigma_k^2 = eigenvalues of  T^H M T M      (acting on f),

the operator of closedloop_sparse.gain with its factors cycled so that the
eigenvector IS the forcing field (velocity part; the pressure and Dirichlet
parts lie in the null space of M and are set to zero) instead of its load
M f, which only M^-1 would turn back into a field.  One sparse LU per
frequency serves both loops; ARPACK on the operator, k = 3.

The frequencies are independent: under MPI they spread over the full-model
pool (bfs/mpipool.py), as the driver's tasks do.

    mpiexec -n 24 python bfs/postproc.py --ops bfs/data/ops_fx.npz \\
        --rec results_bfs/start_000/K_B_C_s0_bfs.npz

post.npz: omega, sigma_open, sigma_closed (N x 3), gamma, band; ctrl_omega,
ctrl_k (complex); vec_omega, f_open, q_open, f_closed, q_closed (one row per
frequency, complex, n entries in the operators' dof order; f: velocity part,
||f||_M = 1; q = T M f, ||q||_M = sigma_1), sigma1_vec_open/closed; xy, comp
(the dof coordinates and components 0 = u, 1 = v, 2 = p); xa, xs.
"""
import os, sys, time, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE)]

NCV = 12          # Arnoldi subspace for k = 3 (each application: 2 solves with the plant factor)
TOL = 1e-8


# ------------------------------------------------------------------ numerics
def _design(rec, start=None):
    """The record's design; the START positions from the record (xa_start,
    xs_start, written since 2026-10-02) or else from the --init file `start`
    (None: not drawn)."""
    z = np.load(rec, allow_pickle=True)
    re = lambda key: np.real(np.asarray(z[key])).astype(float)       # a real controller, stored complex
    mimo = 'Bc' in z.files                                            # m x p (BFS.md 5)
    if mimo:
        m, p = int(z['m']), int(z['p']); r = len(np.atleast_1d(z['d']))
        arow = (re('d'), re('e'), re('f'), re('Bc').reshape(r, p))
        nvec, K = re('nvec').reshape(m, r), re('K').reshape(m, p)
    else:
        m = p = 1
        arow = (re('d'), re('e'), re('f'))
        nvec, K = re('nvec'), complex(z['K'])
    shp = lambda v, k: np.asarray(v, float).reshape(-1, k)          # positions as (ports, dims)
    if 'xa_start' in z.files:
        xa0, xs0 = shp(z['xa_start'], 2), shp(z['xs_start'], 1)
    elif start:
        s0 = np.load(start, allow_pickle=True)
        xa0, xs0 = shp(s0['xa'], 2), shp(s0['xs'], 1)
    else:
        xa0 = xs0 = None
    xa, xs = shp(z['xa'], 2), shp(z['xs'], 1)
    if not mimo:
        xa, xs = xa[0], xs[0]
    return dict(arow=arow, nvec=nvec, K=K, m=m, p=p, mimo=mimo, struct=str(z['struct']) if 'struct' in z.files else '',
                xa=xa, xs=xs, xa0=xa0, xs0=xs0, gamma=float(z['gamma']),
                active=np.asarray(z['active_freqs'], float) if 'active_freqs' in z.files else np.zeros(0))


def svd_at(D, w, des, k=3, vectors=False):
    """Top-k singular values at frequency w, open and closed loop, from one
    LU; with vectors, sigma_1's forcing f and response q of each."""
    import scipy.sparse.linalg as spla
    from closedloop_sparse import _v0
    b, c, *_ = D.bc(des['xa'], des['xs'])
    R = D.cl(b, c)                       # SISO or MIMO resolvent (one LU)
    R.lu_cache = None; R.lu_pin = None
    lu = R.setup(float(w), des['arow'], des['nvec'], des['K'])['lu']
    n, r, M = D.n, D.r, D.M
    free = (D.comp < 2) & np.isin(np.arange(n), D.bc_dofs, invert=True)
    pad = lambda v: np.concatenate([v, np.zeros(r, complex)])
    loops = {'open': (lambda v: lu.solve(v), lambda y: lu.solve(y, trans='H')),
             'closed': (lambda v: R.solve(pad(v))[:n], lambda y: R.solveH(pad(y))[:n])}
    out = dict(omega=float(w))
    for key, (T, TH) in loops.items():
        op = spla.LinearOperator((n, n), dtype=complex,
                                 matvec=lambda f, T=T, TH=TH: TH(M @ T(M @ np.asarray(f, complex).ravel())))
        lam, V = spla.eigs(op, k=k, which='LM', tol=TOL, ncv=max(NCV, 2 * k + 2), maxiter=5000, v0=_v0(n))
        order = np.argsort(-np.real(lam))
        out['sigma_' + key] = np.sqrt(np.maximum(np.real(lam[order]), 0.0))
        if vectors:
            f = np.where(free, V[:, order[0]], 0.0)
            f = f / np.sqrt(np.real(np.vdot(f, M @ f)))
            q = T(M @ f)
            # phase: the response's largest streamwise entry real positive (a
            # singular pair is defined up to one common phase)
            iu = np.where(D.comp == 0)[0]
            ph = np.exp(-1j * np.angle(q[iu[np.argmax(np.abs(q[iu]))]]))
            out['f_' + key], out['q_' + key] = f * ph, q * ph
            out['sigma1_vec_' + key] = float(np.sqrt(np.real(np.vdot(q, M @ q))))
    return out


def _task(t):
    import fullpar
    w, des, k, vectors = t
    return svd_at(fullpar.CTX['D'], w, des, k, vectors)


def controller_tf(D, des, ws):
    """k(i w) = K + n^T (i w I - A_c)^-1 B_c (closedloop_sparse's controller:
    u = K y + n^T x, x' = A_c x + B_c y); MIMO: the m x p matrix
    D_c + C_c (i w I - A_c)^-1 B_c, one per frequency (N x m x p)."""
    from closedloop_sparse import ClosedLoopResolvent
    R = ClosedLoopResolvent(D.M, D.A, np.zeros(1), np.zeros(1), r=D.r, mc=D.mc)
    Ac = R.L_of_a(tuple(des['arow'][:3]))
    Bc = np.asarray(des['arow'][3], complex) if des.get('mimo') else R.B_of()
    I = np.eye(D.r)
    return np.array([des['K'] + des['nvec'] @ np.linalg.solve(1j * w * I - Ac, Bc) for w in ws]), np.linalg.eigvals(Ac)


def freq_grid(W, extra=()):
    g = np.concatenate([np.arange(0.0, min(3.0, W), 0.02), np.arange(3.0, W, 0.1), [W],
                        [x for x in extra if 0.0 <= x <= W]])
    return np.unique(np.round(g, 9))


def _pick(ws, s_open, s_closed, user):
    """The common frequencies of the singular-vector figures: the open-loop
    and closed-loop peaks, then the user's list (duplicates within 0.02
    dropped)."""
    out = []
    for w in [float(ws[np.argmax(s_open)]), float(ws[np.argmax(s_closed)])] + list(user):
        if all(abs(w - x) > 0.02 for x in out):
            out.append(float(w))
    return out


# ------------------------------------------------------------------ figures
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
START = '#8a8984'                              # the start positions (grey; the final ones black)
BAR = 0.1                                      # height of the sensor-window bars, standing on the wall y = 0
C_OPEN, C_CLOSED = '#eb6834', '#2a78d6'       # categorical slots 2 and 1 (dataviz reference palette)


def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_sigma(path, ws, so, sc, gamma, W, wvec):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    _style(ax)
    for x in wvec:
        ax.axvline(x, color=GRID, lw=1.0, zorder=0)
    dashes = ['-', (0, (5, 2)), (0, (1, 1.5))]
    for j in range(so.shape[1]):
        ax.semilogy(ws, so[:, j], color=C_OPEN, lw=2.0 if j == 0 else 1.4, ls=dashes[j],
                    label=f'$\\sigma_{j + 1}$ without control')
        ax.semilogy(ws, sc[:, j], color=C_CLOSED, lw=2.0 if j == 0 else 1.4, ls=dashes[j],
                    label=f'$\\sigma_{j + 1}$ with control')
    ax.axhline(gamma, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.annotate(f'$\\gamma$ = {gamma:.6g} (certified on [0, {W:g}])', xy=(W, gamma), xytext=(-4, 4),
                textcoords='offset points', ha='right', va='bottom', color=INK, fontsize=9)
    io = int(np.argmax(so[:, 0]))
    ax.annotate(f'without control: peak {so[io, 0]:.5g} at $\\omega$ = {ws[io]:.3g}', xy=(ws[io], so[io, 0]),
                xytext=(40, -14), textcoords='offset points', color=INK2, fontsize=8,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    ax.set_xlim(0, W)
    ax.set_xlabel('$\\omega$', color=INK)
    ax.set_ylabel('singular value (noise $\\to$ state, energy norms)', color=INK)
    ax.set_title('Singular values without and with control (grey verticals: frequencies of the mode figures;\n'
                 'where only blue shows, the two curves coincide)',
                 color=INK, fontsize=10, loc='left')
    ax.legend(ncol=3, fontsize=8, frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_modes(path, D, w, rec, des, comp='u'):
    """Four stacked panels: forcing / response, without / with control; the
    real part of one velocity component, each panel scaled to its own
    maximum; actuator and sensor overlaid."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from scipy.spatial import cKDTree
    from matplotlib.patches import Rectangle
    ic = {'u': 0, 'v': 1}[comp]
    idx = np.where(D.comp == ic)[0]
    dist, j = cKDTree(D.xyd[idx]).query(D.pts)
    if dist.max() > 1e-8:
        raise RuntimeError("postproc: mesh vertices without a velocity dof")
    vid = idx[j]                                  # the P2 dof at each vertex
    tri = Triangulation(D.pts[:, 0], D.pts[:, 1], D.tri)
    x_lo, x_hi, y_lo, y_hi = D.box()
    XA = np.asarray(des['xa'], float).reshape(-1, 2); XS = np.asarray(des['xs'], float).reshape(-1)
    xzoom = min(x_hi, max(8.0, float(XS.max()) + D.sensor_len + 2.0, float(XA[:, 0].max()) + 3.0))
    # forcing: zoomed on the step (it lives there), side by side; response: the whole domain
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.0, 1.0], hspace=0.45, wspace=0.08)
    panels = [(fig.add_subplot(gs[0, 0]), 'f_open', 'forcing, without control', xzoom),
              (fig.add_subplot(gs[0, 1]), 'f_closed', 'forcing, with control', xzoom),
              (fig.add_subplot(gs[1, :]), 'q_open', 'response, without control', x_hi),
              (fig.add_subplot(gs[2, :]), 'q_closed', 'response, with control', x_hi)]
    th = np.linspace(0, 2 * np.pi, 120)
    xx = np.linspace(x_lo, x_hi, 600)
    for ax, key, title, xr in panels:
        val = np.real(rec[key][vid])
        sel = D.pts[:, 0] <= xr + 1e-9
        m = float(np.abs(val[sel]).max()) or 1.0
        pc = ax.tripcolor(tri, val / m, cmap='RdBu_r', vmin=-1, vmax=1, shading='gouraud', rasterized=True)
        ax.plot(xx, D.wall(xx), color=INK2, lw=0.8)
        lab = len(XA) > 1 or len(XS) > 1          # MIMO: number the ports
        if des.get('xa0') is not None:
            # the start in grey, each actuator's move as an arrow; the sensor
            # windows as equal bars standing on the wall (Barbagallo et al. 2012,
            # figure 1), the final ones drawn over the start ones
            XA0 = np.asarray(des['xa0'], float).reshape(-1, 2); XS0 = np.asarray(des['xs0'], float).reshape(-1)
            for i, xa0 in enumerate(XA0):
                ax.plot(xa0[0] + 0.5 * np.cos(th), xa0[1] + 0.5 * np.sin(th), color=START, lw=0.9, ls=(0, (1.5, 1.5)))
                ax.plot(*xa0, marker='x', color=START, ms=5, mew=1.0)
                if i < len(XA) and np.linalg.norm(XA[i] - xa0) > 1e-3:
                    ax.annotate('', xy=tuple(XA[i]), xytext=tuple(xa0),
                                arrowprops=dict(arrowstyle='->', color=INK, lw=0.9, shrinkA=2, shrinkB=2))
            for xs0 in XS0:
                ax.add_patch(Rectangle((xs0, 0.0), D.sensor_len, BAR, color=START, lw=0, zorder=5))
        for i, xa_ in enumerate(XA):
            ax.plot(xa_[0] + 0.5 * np.cos(th), xa_[1] + 0.5 * np.sin(th), color=INK, lw=1.0, ls=(0, (3, 2)))
            ax.plot(*xa_, marker='+', color=INK, ms=7, mew=1.2)
            if lab:
                ax.annotate(f'a{i + 1}', xy=(xa_[0], xa_[1] + 0.55), ha='center', va='bottom', color=INK, fontsize=7)
        for j, xs_ in enumerate(XS):
            ax.add_patch(Rectangle((xs_, 0.0), D.sensor_len, BAR, color=INK, lw=0, zorder=6))
            if lab:
                ax.annotate(f's{j + 1}', xy=(xs_ + 0.5 * D.sensor_len, BAR), ha='center', va='bottom', color=INK,
                            fontsize=7)
        ax.set_aspect('equal')
        ax.set_xlim(x_lo, xr); ax.set_ylim(y_lo, y_hi)          # the bottom of the panel is the wall
        sig = rec['sigma1_vec_' + key.split('_')[1]]
        ax.set_title(f'optimal {title} ($\\sigma_1$ = {sig:.5g})', color=INK, fontsize=9, loc='left')
        ax.tick_params(colors=INK2, labelsize=8)
    fig.colorbar(pc, ax=[p[0] for p in panels], shrink=0.5, pad=0.01,
                 label=f'Re {comp}, each panel scaled to max |.| = 1')
    moved = ('; grey: their START positions (dotted circle, x, bar under the black one), arrow: the actuator\'s move'
             if des.get('xa0') is not None else '')
    fig.suptitle(f'$\\omega$ = {w:.4g}: singular vectors of $\\sigma_1$; actuator (dashed: half-amplitude circle, '
                 f'+: centre), sensor window (bar on the wall){moved}', color=INK, fontsize=9.5, x=0.01, y=0.95,
                 ha='left', wrap=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_controller(path, wk, kk, poles, W):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax in (a1, a2):
        _style(ax)
        ax.axvspan(wk[0], W, color='#f0efec', zorder=0)
        for p in poles:
            if p.imag > 1e-9 * max(abs(p), 1.0) and wk[0] <= p.imag <= wk[-1]:
                ax.axvline(p.imag, color=GRID, lw=1.0, ls=(0, (2, 2)))
    kk = np.asarray(kk)
    if kk.ndim == 3:
        # MIMO: every entry that is not identically zero (decentral: the diagonal),
        # k_ij = sensor j -> actuator i
        cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
        ent = [(i, j) for i in range(kk.shape[1]) for j in range(kk.shape[2]) if np.abs(kk[:, i, j]).max() > 0]
        for q, (i, j) in enumerate(ent):
            a1.loglog(wk, np.abs(kk[:, i, j]), color=cols[q % 4], lw=2.0, label=f'$k_{{{i + 1}{j + 1}}}$ (sensor {j + 1} -> actuator {i + 1})')
            a2.semilogx(wk, np.degrees(np.unwrap(np.angle(kk[:, i, j]))), color=cols[q % 4], lw=2.0)
        a1.legend(fontsize=8, frameon=False)
    else:
        a1.loglog(wk, np.abs(kk), color=C_CLOSED, lw=2.0)
        a2.semilogx(wk, np.degrees(np.unwrap(np.angle(kk))), color=C_CLOSED, lw=2.0)
    a1.set_ylabel('|k(i$\\omega$)|', color=INK)
    a2.set_ylabel('phase (deg)', color=INK)
    a2.set_xlabel('$\\omega$', color=INK)
    ps = ', '.join((f'{p.real:.3g}{p.imag:+.3g}i' if abs(p.imag) > 1e-9 * max(abs(p), 1.0) else f'{p.real:.3g}')
                   for p in sorted(poles, key=lambda z: (z.real, z.imag)))
    a1.set_title(f'Controller Bode plot (shaded: certified band [0, {W:g}]; dashed: |Im| of its poles)\n'
                 f'poles {ps}', color=INK, fontsize=9, loc='left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ops', default='bfs/data/ops_fx.npz')
    ap.add_argument('--rec', required=True, help='the run record (.npz; its .json gives the final band)')
    ap.add_argument('--out', default=None, help='output directory (default: <record>_post)')
    ap.add_argument('--omegas', default='0.3,1.5', help='frequencies of the mode figures, beside the two peaks')
    ap.add_argument('--whi', type=float, default=10.0, help='lower bound of the plotted range (the band edge if larger)')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--comp', default='u', choices=('u', 'v'), help='velocity component shown in the mode figures')
    ap.add_argument('--start', default=None,
                    help='the --init file of the run, for its start positions when the record lacks them (pre-2026-10-02)')
    ap.add_argument('--replot', action='store_true', help='only redraw the figures from OUT/post.npz (serial)')
    a = ap.parse_args()
    out = a.out or a.rec[:-4] + '_post'
    if a.replot:
        from bfs.plant import BFSPlant
        des = _design(a.rec, a.start)
        plot(out, BFSPlant(a.ops, r=len(des['arow'][0])), des, a.comp)
        return
    from bfs import mpipool
    if mpipool.active() and mpipool.comm().Get_rank() > 0:
        mpipool.serve()
    import sparse_lu
    sparse_lu.ORDER = 'mumps'
    import fullpar
    from bfs.plant import BFSPlant
    des = _design(a.rec, a.start)
    js = a.rec[:-4] + '.json'
    band = json.load(open(js)).get('band_final', [0.0, a.whi]) if os.path.exists(js) else [0.0, a.whi]
    W = max(float(a.whi), float(band[1]))
    os.makedirs(out, exist_ok=True)
    D = BFSPlant(a.ops, r=len(des['arow'][0]))
    fullpar.start(1, D=D)
    t0 = time.time()
    ws = freq_grid(W, des['active'])
    res = fullpar.pmap(_task, [(w, des, a.k, False) for w in ws])
    so = np.array([r_['sigma_open'] for r_ in res]); sc = np.array([r_['sigma_closed'] for r_ in res])
    t1 = time.time()
    wvec = _pick(ws, so[:, 0], sc[:, 0], [float(x) for x in a.omegas.split(',') if x.strip()])
    vec = fullpar.pmap(_task, [(w, des, a.k, True) for w in wvec])
    t2 = time.time()
    wk = np.logspace(-2, np.log10(max(100.0, 10.0 * W)), 600)
    kk, poles = controller_tf(D, des, wk)
    np.savez(os.path.join(out, 'post.npz'), omega=ws, sigma_open=so, sigma_closed=sc, gamma=des['gamma'],
             band=np.array([0.0, W]), ctrl_omega=wk, ctrl_k=kk, ctrl_poles=poles, vec_omega=np.array(wvec),
             **{key: np.array([v[key] for v in vec]) for key in ('f_open', 'q_open', 'f_closed', 'q_closed',
                                                                 'sigma1_vec_open', 'sigma1_vec_closed')},
             xy=D.xyd, comp=D.comp, xa=des['xa'], xs=des['xs'])
    io, ic = int(np.argmax(so[:, 0])), int(np.argmax(sc[:, 0]))
    summ = dict(rec=a.rec, n=D.n, band=[0.0, W], nfreq=len(ws), gamma_record=des['gamma'],
                peak_open=dict(omega=float(ws[io]), sigma=[float(x) for x in so[io]]),
                peak_closed=dict(omega=float(ws[ic]), sigma=[float(x) for x in sc[ic]]),
                closed_over_gamma=float(sc[:, 0].max() / des['gamma']),
                vec_omega=wvec, sigma1_vec_open=[float(v['sigma1_vec_open']) for v in vec],
                sigma1_vec_closed=[float(v['sigma1_vec_closed']) for v in vec],
                seconds=dict(curves=round(t1 - t0, 1), vectors=round(t2 - t1, 1)), pool=fullpar.report())
    json.dump(summ, open(os.path.join(out, 'post.json'), 'w'), indent=1)
    print(json.dumps(summ), flush=True)
    fullpar.stop()
    plot(out, D, des, a.comp)


def plot(out, D, des, comp='u'):
    """The figures, from OUT/post.npz alone (and the mesh): no solve."""
    z = np.load(os.path.join(out, 'post.npz'))
    ws, W, wvec = z['omega'], float(z['band'][1]), [float(w) for w in z['vec_omega']]
    fig_sigma(os.path.join(out, 'sigma.png'), ws, z['sigma_open'], z['sigma_closed'], float(z['gamma']), W, wvec)
    for i, w in enumerate(wvec):
        rec = {key: z[key][i] for key in ('f_open', 'q_open', 'f_closed', 'q_closed', 'sigma1_vec_open',
                                           'sigma1_vec_closed')}
        fig_modes(os.path.join(out, f'modes_w{w:.3f}.png'), D, w, rec, des, comp=comp)
    fig_controller(os.path.join(out, 'controller.png'), z['ctrl_omega'], z['ctrl_k'], z['ctrl_poles'], W)
    print(f"figures in {out}", flush=True)

if __name__ == '__main__':
    main()
