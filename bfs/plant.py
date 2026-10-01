"""plant.py -- the backward-facing-step plant as a co-design object (BFS.md 2).

BFSPlant has the interface the GL pipeline uses on FECoDesign (A, M, n, r,
mc, cl, bc, grads, norm_M, alpha_certified, hinf, ...), on the operators of
bfs/plant_io.py:

    M q' = A q + b u + M f,     y = c^H q,      performance ||q||_M

  A = J = -dF/dW (real, Dirichlet rows -identity), M = velocity mass with the
  Dirichlet rows/columns and the pressure block zero (singular).  The
  disturbance is a momentum forcing f (load M f: forcing vanishing at
  Dirichlet dofs; the archive's gains are reproduced to 1e-10 with it), the
  performance the kinetic energy.  M only ever multiplies: no M^-1, no
  Cholesky factor (the gain eigenproblem W = M T^H M T, closedloop_sparse).

Ports (user, 2026-10-01): isotropic Gaussians of standard deviation h/2 = 0.5
(step height h = 1) with FREE CENTRES (x, y):
  actuator  forces the y-momentum only:   psi_a = (0, g(x - xa)),  b = M psi_a
  sensor    measures u only:               psi_s = (g(x - xs), 0),  c = M psi_s,
            y = c^H q = int g_s u dx
both normalised to ||psi||_M = 1 (the controller absorbs the scale, as in GL).
A position is a 2-vector; the placement gradient has two components.

The plant, the ports and the controller are REAL: sigma(-w) = sigma(w) and
eigenvalues come in conjugate pairs (the driver's --real-plant uses w >= 0).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from fe_codesign import FECoDesign
from closedloop_sparse import ClosedLoopResolvent

SIGMA_PORT = 0.5          # h / 2


class BFSPlant(FECoDesign):
    pos_dim = 2
    real_plant = True

    def __init__(self, ops, r=5, mc=1.0, sigma=SIGMA_PORT):
        from bfs.plant_io import load_ops
        o = load_ops(ops) if isinstance(ops, str) else ops
        self.ops_path = ops if isinstance(ops, str) else None
        self.A, self.M = o['A'].tocsc(), o['M'].tocsc()
        self.n = self.A.shape[0]; self.r = r; self.mc = mc; self.ne = 0
        self.comp, self.xyd = o['comp'], o['xy']
        self.iu, self.iv = np.where(self.comp == 0)[0], np.where(self.comp == 1)[0]
        self.bc_dofs = o['bc']
        self.pts, self.tri = o['pts'], o['tri']
        self.edges, self.edge_labels = o['edges'], o['edge_labels']
        self.Re = o['Re']
        self.sigma_port = float(sigma)
        self.xn = None                       # no 1-D node array (field cases are GL only)
        self.Kstiff = None
        # as FECoDesign.__init__ (the attributes its generic methods use)
        self._lu_cache = {}; self._lu_pin = {}
        self._warm_lam = None; self._warm_fp = None
        self.cert_eps_int = 1e-5; self.cert_itmax = 800; self.cert_shrink = 0.9
        self.cert_nseg = 1; self.cert_pool = None
        self.alpha_warm_maxstep = 0.02; self.alpha_warm = True; self.alpha_full_every = 20
        self.alpha_stats = dict(warm=0, full=0, since_full=10 ** 9, warm_empty=0, disagree=0,
                                worst_disagree=0.0, last_warm_alpha=None)
        self.lu_cache_max = 2                # one factor is ~1 GB here
        self.sob_len = 0.0; self._Hlu_f = None
        self._Mdpsi_cache = {}
        import massops
        massops.FORBIDDEN = True             # M is singular: never factorised (BFS.md 4)
        import closedloop_sparse
        closedloop_sparse.GAIN_NCV = 8       # BFS.md 5

    # ------------------------------------------------------------------ ports
    def _g(self, x0, idx):
        X = self.xyd[idx]
        d2 = (X[:, 0] - x0[0]) ** 2 + (X[:, 1] - x0[1]) ** 2
        return np.exp(-d2 / (2.0 * self.sigma_port ** 2))

    def raw(self, x0, which):
        """The unnormalised port field on the full dof vector: 'a' -> v
        component (y-momentum forcing), 's' -> u component (measured)."""
        x0 = np.asarray(x0, float).ravel()
        idx = self.iv if which == 'a' else self.iu
        v = np.zeros(self.n, complex)
        v[idx] = self._g(x0, idx)
        return v

    def draw(self, x0, which):
        """d raw / d(x0, y0): (2, n)."""
        x0 = np.asarray(x0, float).ravel()
        idx = self.iv if which == 'a' else self.iu
        X = self.xyd[idx]
        g = self._g(x0, idx)
        out = np.zeros((2, self.n), complex)
        out[0, idx] = (X[:, 0] - x0[0]) / self.sigma_port ** 2 * g
        out[1, idx] = (X[:, 1] - x0[1]) / self.sigma_port ** 2 * g
        return out

    def jet(self, x0, which):
        """The raw port field and its first and second derivatives in the
        centre, (6, n): g, g_x, g_y, g_xx, g_xy, g_yy.  Their M-images span the
        port's moves to second order: the reduced model's input basis for a free
        centre (BFS.md 5; the GL range basis covered a 1-D interval)."""
        x0 = np.asarray(x0, float).ravel()
        idx = self.iv if which == 'a' else self.iu
        X = self.xyd[idx]; s2 = self.sigma_port ** 2
        g = self._g(x0, idx); dx = X[:, 0] - x0[0]; dy = X[:, 1] - x0[1]
        rows = [g, dx / s2 * g, dy / s2 * g, (dx * dx / s2 - 1.0) / s2 * g, dx * dy / s2 ** 2 * g,
                (dy * dy / s2 - 1.0) / s2 * g]
        out = np.zeros((6, self.n), complex)
        for k, r_ in enumerate(rows):
            out[k, idx] = r_
        return out

    def gauss(self, x0):
        raise NotImplementedError("BFS ports differ by component: use raw(x0, 'a' | 's')")

    def bc(self, xa, xs, va=None, vs=None):
        if va is not None or vs is not None:
            raise NotImplementedError("free fields (FB/FC) are not part of the BFS fork")
        if np.ndim(xa) > 1 or np.ndim(xs) > 1:
            raise NotImplementedError("MIMO is not part of the BFS fork yet")
        ra, rs = self.raw(xa, 'a'), self.raw(xs, 's')
        na, ns = self.norm_M(ra), self.norm_M(rs)
        if not (np.isfinite(na) and na > 1e-300 and np.isfinite(ns) and ns > 1e-300):
            raise FloatingPointError(f"bc(): degenerate port field (|psi_a| {na}, |psi_s| {ns}): "
                                     f"centre outside the domain?")
        pa, ps = ra / na, rs / ns
        return self.M @ pa, self.M @ ps, pa, ps, na, ns

    def Mdpsi(self, x0, which):
        """M d(psi)/d(x0, y0) for the UNIT-M-norm port field (the tangent
        projection of the raw derivative, as FECoDesign.dpsi_dx): (2, n),
        memoised on (x0, which)."""
        key = (which, float(x0[0]), float(x0[1]))
        v = self._Mdpsi_cache.get(key)
        if v is None:
            if len(self._Mdpsi_cache) >= 16:
                self._Mdpsi_cache.pop(next(iter(self._Mdpsi_cache)))
            g = self.raw(x0, which); dg = self.draw(x0, which)
            nv = self.norm_M(g); p = g / nv
            Mp = self.M @ p
            rows = []
            for k in range(2):
                t = float(np.real(np.vdot(dg[k], Mp)))
                rows.append(self.M @ ((dg[k] - t * p) / nv))
            v = self._Mdpsi_cache[key] = np.array(rows)
        return v

    # ------------------------------------------------------------- gradients
    def grads(self, cl, x, z, arow, nvec, Kfb, xa, xs, pa, ps, na, ns, flags, eta=1.0, _full=False):
        """FECoDesign.grads with 2-D placements: 'xa' / 'xs' are 2-vectors."""
        if getattr(cl, 'is_reduced', False) and not _full and len(x) == cl.kd + cl._r.r:
            return cl.grads_reduced(self, x, z, arow, nvec, Kfb, xa, xs, pa, ps, na, ns, flags, eta)
        n = self.n
        xp, xc = x[:n], x[n:]
        zp, zc = z[:n], z[n:]
        b, c = cl.b, cl.c
        nvec = np.asarray(nvec, complex).ravel()
        out = {}
        if flags['K']:
            g = cl.design_grad_tri(dict(x=x, z=z))
            for k_ in ('d', 'e', 'f', 'n'):
                out[k_] = np.asarray(g[k_]) / eta
            out['K'] = g['K'] / eta
        Sa = Kfb * np.vdot(c, xp) + np.dot(nvec, xc)
        Sc = np.vdot(zp, b) * Kfb + self.mc * np.sum(np.conj(zc))
        if flags['B']:
            ba = self.Mdpsi(xa, 'a')
            out['xa'] = np.array([float(np.real(np.vdot(zp, ba[k]) * Sa / eta)) for k in range(2)])
        if flags['C']:
            cs = self.Mdpsi(xs, 's')
            out['xs'] = np.array([float(np.real(Sc * np.vdot(cs[k], xp) / eta)) for k in range(2)])
        return out

    # --------------------------------------------------------------- geometry
    def wall(self, x):
        """Lower wall of the linear-analysis domain (mesh_large.py): y = 1 for
        x <= 0 (the plate), the arc of centre (0, -1.5), R = 2.5 for 0 <= x <= 2,
        y = 0 beyond."""
        x = np.asarray(x, float)
        arc = -1.5 + np.sqrt(np.clip(6.25 - x ** 2, 0.0, None))
        return np.where(x <= 0.0, 1.0, np.where(x >= 2.0, 0.0, arc))

    def dwall(self, x):
        x = np.asarray(x, float)
        d = -x / np.sqrt(np.clip(6.25 - x ** 2, 1e-300, None))
        return np.where((x > 0.0) & (x < 2.0), d, 0.0)

    def box(self):
        return (float(self.pts[:, 0].min()), float(self.pts[:, 0].max()),
                float(self.pts[:, 1].min()), float(self.pts[:, 1].max()))

    def inside(self, x0, margin=0.0):
        x_lo, x_hi, y_lo, y_hi = self.box()
        x, y = float(x0[0]), float(x0[1])
        return (x_lo + margin <= x <= x_hi - margin and y <= y_hi - margin
                and y >= float(self.wall(x)) + margin)

    # ---------------------------------------------------------------- loop
    def hinf(self, arow, nvec, Kfb, b, c, lam=None, warm=None, nsw=9, wmin=0.0, wmax=4.0, tol_active=2e-3):
        """FECoDesign.hinf (the seeded local search for a lower bound) on
        w >= 0 only: the plant, the ports and the controller are real, so
        sigma(-w) = sigma(w)."""
        return FECoDesign.hinf(self, arow, nvec, Kfb, b, c, lam=lam, warm=warm, nsw=nsw,
                               wmin=max(0.0, wmin), wmax=wmax, tol_active=tol_active)

    band_of = None           # the driver's band, a callable -> the current upper edge

    @property
    def stab_whi(self):
        """Upper edge of the stability line certificate: the driver's band
        (band_of) or 10."""
        return float(self.band_of()) if self.band_of is not None else 10.0

    def alpha_certified(self, arow, nvec, Kfb, b, c, eps=1e-3, wlo=None, whi=None, eigs_seen=None):
        """FECoDesign.alpha_certified on Im in [0, stab_whi] (conjugate
        pairs: the lower half is the mirror image).  The GL default region came
        from the plant's numerical-range enclosure (M^-1); here the plant's
        spectrum near the axis is computed once (plant_spectrum) and the band
        is the driver's."""
        return FECoDesign.alpha_certified(self, arow, nvec, Kfb, b, c, eps=eps,
                                          wlo=(0.0 if wlo is None else max(0.0, wlo)),
                                          whi=(self.stab_whi if whi is None else whi), eigs_seen=eigs_seen)

    def plant_spectrum(self, wmax, nshift=None, k=8, tol=1e-8, maxiter=200):
        """Eigenvalues of (A, M) near the imaginary axis on [0, wmax], by
        shift-invert at shifts i w_j (in the full-model pool), conjugates
        added.  Fills the certificate's plant cache (certify_sparse.
        stability_region), so nothing there touches M^-1.  NOT a proof that
        none is missed (the GL enclosure needed M^-1; BFS.md 4)."""
        import fullpar, certify_sparse as csp
        nshift = nshift or max(4, int(np.ceil(wmax / 1.0)) + 1)
        shifts = [1j * w for w in np.linspace(0.0, wmax, nshift)]
        res = fullpar.pmap(_plant_shift_task, [(s, k, tol, maxiter) for s in shifts])
        ev = np.concatenate([r[0] for r in res]) if res else np.zeros(0, complex)
        ev = ev[np.isfinite(ev)]
        ev = np.unique(np.round(np.concatenate([ev, ev.conj()]), 10))
        self.plant_ev = ev
        self.plant_ev_ok = all(r[1] for r in res)
        key = (id(self.A), id(self.M), self.A.shape, self.A.nnz, self.M.nnz)
        csp._PLANT_EV_CACHE.clear(); csp._PLANT_EV_CACHE[key] = ev
        return ev

    def cl(self, b, c):
        R = ClosedLoopResolvent(self.M, self.A, b, c, r=self.r, mc=self.mc)
        R.lu_cache = self._lu_cache
        R.lu_cache_max = self.lu_cache_max
        R.lu_pin = self._lu_pin
        return R


# ---------------------------------------------------------------- plant spectrum
def _plant_shift_task(t):
    """Eigenvalues of (A, M) nearest the shift s (shift-invert Arnoldi on
    OP = (A - s M)^-1 M: one LU of A - s M, M only multiplies; the infinite
    eigenvalues of the singular pencil map to theta = 0 and stay out of the
    largest-magnitude set).  Runs in the full-model pool."""
    import fullpar, sparse_lu
    import scipy.sparse.linalg as spla
    s, k, tol, maxiter = t
    D = fullpar.CTX['D']
    lu = sparse_lu.factor((D.A - s * D.M).tocsc())
    n = D.n
    OP = spla.LinearOperator((n, n), dtype=complex, matvec=lambda x: lu.solve(D.M @ np.asarray(x, complex)))
    v0 = np.random.default_rng(12345).standard_normal(n) + 0j
    v0 = OP.matvec(OP.matvec(v0))            # purified start: in the range of OP^2 (index-2 pencil)
    try:
        th = spla.eigs(OP, k=k, which='LM', tol=tol, maxiter=maxiter, v0=v0, return_eigenvectors=False)
        ok = True
    except spla.ArpackNoConvergence as e:
        th = e.eigenvalues if e.eigenvalues is not None else np.zeros(0); ok = False
    th = th[np.abs(th) > 1e-14]
    return s + 1.0 / th, ok
