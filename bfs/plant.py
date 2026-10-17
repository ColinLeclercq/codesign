"""plant.py -- the backward-facing-step plant as a co-design object (BFS.md 2).

BFSPlant has the interface the GL pipeline uses on FECoDesign (A, M, n, r,
mc, cl, bc, grads, norm_M, alpha_certified, hinf, ...), on the operators of
bfs/plant_io.py:

    M q' = A q + b u + M P w,     y = c^H q,      performance ||q||_M

  A = J = -dF/dW (real, Dirichlet rows -identity), M = Q, the velocity mass
  with the Dirichlet rows/columns and the pressure block zero (singular).
  The noise w is a [u, v] field (P pads a zero pressure), measured in the
  velocity mass Q'; the performance is the whole state in the Q seminorm;
  so B B^T = C^T C = Q (BFS.md 4).  M only ever multiplies: no M^-1, no
  Cholesky factor.

Ports (user, 2026-10-02): the devices of Barbagallo, Dergham, Sipp, Schmid &
Robinet (JFM 2012), with free positions:
  actuator  a body force on the VERTICAL momentum (paper section 2.3; the
            user's actuator_sensor.edp forces u -- `act_comp`), shaped like
            the script's Gaussians: exp(-|x - x_a|^2 / (2 sig^2)),
            sig = 0.5 / sqrt(ln 2) (half amplitude at distance 0.5), peak 1,
            in weak form b = M psi_a; free centre x_a = (x, y).  The script
            truncates it at radius 2 (an amplitude jump of 0.4 %, 1e-5 of
            its energy): not done here, so b is smooth in x_a.
  sensor    wall shear stress on the floor y = 0,
            y = int chi(x - x_s) du/dy dx,  c = S^T (w * chi)
            (the script's int1d(th, 2)(dy(u) * select)); chi a window of
            length L = 0.6 (Cp = [11, 11.6] in both; figure 1 for C1..C4)
            with tanh edges of width delta = 0.05 (the script's P2
            interpolant of the indicator ramps over one element, 0.04 on the
            fine floor); free start x_s (one number).
Unnormalised (physical) ports: the controller gain absorbs the scale.

The plant, the ports and the controller are REAL: sigma(-w) = sigma(w) and
eigenvalues come in conjugate pairs (the driver's band is w >= 0).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from fe_codesign import FECoDesign
from closedloop_sparse import ClosedLoopResolvent

SIG_ACT = 0.5 / np.sqrt(np.log(2.0))   # Barbagallo et al. / actuator_sensor.edp
SENSOR_LEN = 0.6
SENSOR_EDGE = 0.05
FLOOR_X0 = 2.0                         # the floor y = 0 starts at the end of the arc
POS_MARGIN = 0.05                      # centres / windows this far inside (BFS.md 2; --bfs-margin)



# The version of the certified range and pole data (range_data, plant_poles,
# bfs/mfree.py, sparse_spectrum.py): part of the plant files' id
# (bfs/plant_id.py), so a change of how they are computed makes new files
# instead of reusing the cached ones.  Bump it with every such change.
RANGE_VERSION = 1

class BFSPlant(FECoDesign):
    pos_dims = {'a': 2, 's': 1}      # actuator centre (x, y); sensor window start x_s
    real_plant = True

    def __init__(self, ops, r=5, mc=1.0, act_comp='v'):
        from bfs.plant_io import load_ops
        o = load_ops(ops) if isinstance(ops, str) else ops
        self.ops_path = ops if isinstance(ops, str) else None
        self.A, self.M = o['A'].tocsc(), o['M'].tocsc()
        self.L = o['L'].tocsr()
        self.n = self.A.shape[0]; self.r = r; self.mc = mc; self.ne = 0
        self.comp, self.xyd = o['comp'], o['xy']
        self.iu, self.iv = np.where(self.comp == 0)[0], np.where(self.comp == 1)[0]
        self.bc_dofs = o['bc']
        self.pts, self.tri = o['pts'], o['tri']
        self.edges, self.edge_labels = o['edges'], o['edge_labels']
        self.Re = o['Re']
        self.act_comp = act_comp
        self.sig_act = SIG_ACT
        self.shear_S = o['shear_S'].tocsr(); self.shear_x = o['shear_x']; self.shear_w = o['shear_w']
        self.shear_S = sp.csr_matrix(self.shear_S.multiply(
            np.isin(np.arange(self.n), self.bc_dofs, invert=True)[None, :]))   # Dirichlet columns: zero, as b's rows
        self.sensor_len, self.sensor_edge = SENSOR_LEN, SENSOR_EDGE
        self.pos_margin = POS_MARGIN
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
        import certify_sparse                # the gamma certificate's disks (BFS.md 4)
        certify_sparse.FREE_DISK = False
        certify_sparse.STRICT_MAXITER = 60
        certify_sparse.NEAR_NCV = 100
        import sparse_spectrum
        sparse_spectrum.FREE_TILE = False

    # ------------------------------------------------------------------ ports
    def _gauss(self, x0, idx):
        X = self.xyd[idx]
        d2 = (X[:, 0] - x0[0]) ** 2 + (X[:, 1] - x0[1]) ** 2
        return np.exp(-d2 / (2.0 * self.sig_act ** 2))

    def _act_idx(self):
        return self.iv if self.act_comp == 'v' else self.iu

    def _window(self, xs, order=0):
        """chi(x_q - x_s) and its derivatives in x_s (order 0, 1, 2) at the
        floor quadrature points."""
        t1 = np.tanh((self.shear_x - xs) / self.sensor_edge)
        t2 = np.tanh((self.shear_x - xs - self.sensor_len) / self.sensor_edge)
        if order == 0:
            return 0.5 * (t1 - t2)
        e = self.sensor_edge
        if order == 1:                     # d/dxs: -(1/e) sech^2 terms
            return -0.5 * ((1.0 - t1 ** 2) - (1.0 - t2 ** 2)) / e
        return -((1.0 - t1 ** 2) * t1 - (1.0 - t2 ** 2) * t2) / e ** 2   # d2/dxs2

    def act_field(self, xa):
        v = np.zeros(self.n, complex)
        idx = self._act_idx(); v[idx] = self._gauss(np.asarray(xa, float).ravel(), idx)
        return v

    def sensor_vec(self, xs, order=0):
        xs = float(np.ravel(xs)[0])
        return np.asarray(self.shear_S.T @ (self.shear_w * self._window(xs, order)), complex)

    def bc(self, xa, xs, va=None, vs=None):
        """b = M psi_a (the actuator's weak-form load), c = S^T (w chi) (the
        shear sensor); returns (b, c, psi_a, c, 1, 1) -- unnormalised ports."""
        if va is not None or vs is not None:
            raise NotImplementedError("free fields (FB/FC) are not part of the BFS fork")
        if np.ndim(xa) == 2 or np.ndim(xs) == 2:
            # MIMO (BFS.md 5): one column per port -- B = M Psi_a (n x m), C (n x p)
            XA = np.asarray(xa, float).reshape(-1, 2); XS = np.asarray(xs, float).reshape(-1, 1)
            PA = np.column_stack([self.act_field(x_) for x_ in XA])
            B = np.asarray(self.M @ PA)
            C = np.column_stack([self.sensor_vec(x_) for x_ in XS])
            if not (np.all(np.linalg.norm(B, axis=0) > 0.0) and np.all(np.linalg.norm(C, axis=0) > 0.0)):
                raise FloatingPointError("bc(): a port vanishes (centre / window outside the domain?)")
            return B, C, PA, C, np.ones(B.shape[1]), np.ones(C.shape[1])
        pa = self.act_field(xa)
        b = self.M @ pa
        c = self.sensor_vec(xs)
        if not (np.linalg.norm(b) > 0.0 and np.linalg.norm(c) > 0.0):
            raise FloatingPointError("bc(): a port vanishes (centre / window outside the domain?)")
        return b, c, pa, c, 1.0, 1.0

    def Mdpsi(self, x0, which):
        """d(port vector)/d(position): actuator (2, n) = M d psi_a/d(x, y),
        sensor (1, n) = d c/d x_s.  Memoised (reduced.placement, grads)."""
        x0 = np.atleast_1d(np.asarray(x0, float))
        key = (which,) + tuple(float(t) for t in x0)
        v = self._Mdpsi_cache.get(key)
        if v is None:
            if len(self._Mdpsi_cache) >= 16:
                self._Mdpsi_cache.pop(next(iter(self._Mdpsi_cache)))
            if which == 'a':
                idx = self._act_idx(); X = self.xyd[idx]; g = self._gauss(x0, idx)
                rows = []
                for k in range(2):
                    d = np.zeros(self.n, complex)
                    d[idx] = (X[:, k] - x0[k]) / self.sig_act ** 2 * g
                    rows.append(self.M @ d)
                v = np.array(rows)
            else:
                v = self.sensor_vec(x0, 1)[None, :]
            self._Mdpsi_cache[key] = v
        return v

    def port_jet(self, x0, which):
        """The port vector and its first and second derivatives in the position:
        actuator (6, n) M-images of g, g_x, g_y, g_xx, g_xy, g_yy; sensor (3, n)
        c, c', c''.  Their span is the reduced model's input basis for a free
        port (BFS.md 5)."""
        x0 = np.atleast_1d(np.asarray(x0, float))
        if which == 's':
            return np.array([self.sensor_vec(x0, k) for k in range(3)])
        idx = self._act_idx(); X = self.xyd[idx]; s2 = self.sig_act ** 2
        g = self._gauss(x0, idx); dx = X[:, 0] - x0[0]; dy = X[:, 1] - x0[1]
        rows = [g, dx / s2 * g, dy / s2 * g, (dx * dx / s2 - 1.0) / s2 * g, dx * dy / s2 ** 2 * g,
                (dy * dy / s2 - 1.0) / s2 * g]
        out = np.zeros((6, self.n), complex)
        for k, r_ in enumerate(rows):
            v = np.zeros(self.n, complex); v[idx] = r_
            out[k] = self.M @ v
        return out

    # ------------------------------------------------------------- gradients
    def grads(self, cl, x, z, arow, nvec, Kfb, xa, xs, pa, ps, na, ns, flags, eta=1.0, _full=False):
        """FECoDesign.grads with vector placements ('xa' (2,), 'xs' (1,)):
        d sigma / d pos = Re(z^H db S_a) and Re(S_c dc^H x) with db, dc the
        derivatives of the port vectors (Mdpsi)."""
        if getattr(cl, 'is_reduced', False) and not _full and len(x) == cl.kd + cl._r.r:
            return cl.grads_reduced(self, x, z, arow, nvec, Kfb, xa, xs, pa, ps, na, ns, flags, eta)
        if getattr(cl, 'm', None) is not None:
            return self.grads_mimo(cl, x, z, arow, nvec, Kfb, xa, xs, flags, eta)
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
            out['xa'] = np.array([float(np.real(np.vdot(zp, ba[k]) * Sa / eta)) for k in range(len(ba))])
        if flags['C']:
            cs = self.Mdpsi(xs, 's')
            out['xs'] = np.array([float(np.real(Sc * np.vdot(cs[k], xp) / eta)) for k in range(len(cs))])
        return out

    def grads_mimo(self, cl, x, z, arow, nvec, Kfb, xa, xs, flags, eta=1.0):
        """MIMO (BFS.md 5): the controller entries (d, e, f, B_c, C_c, D_c) from
        the resolvent, and per port the SISO formulas with the actuator-side
        act_i and sensor-side sen_j (ClosedLoopResolventMIMO.port_sides):
        'xa' (m, 2), 'xs' (p, 1)."""
        n = self.n
        xp, zp = x[:n], z[:n]
        res = dict(x=x, z=z)
        out = {}
        if flags['K']:
            g = cl.design_grad_tri(res)
            for k_ in ('d', 'e', 'f', 'Bc', 'n', 'K'):
                out[k_] = np.asarray(g[k_]) / eta
        act, sen = cl.port_sides(res, arow, nvec, Kfb)
        if flags['B']:
            XA = np.asarray(xa, float).reshape(-1, 2)
            out['xa'] = np.array([[float(np.real(np.vdot(zp, ba) * act[i] / eta)) for ba in self.Mdpsi(XA[i], 'a')]
                                  for i in range(len(XA))])
        if flags['C']:
            XS = np.asarray(xs, float).reshape(-1, 1)
            out['xs'] = np.array([[float(np.real(sen[j] * np.vdot(cs, xp) / eta)) for cs in self.Mdpsi(XS[j], 's')]
                                  for j in range(len(XS))])
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

    def d2wall(self, x):
        x = np.asarray(x, float)
        on = (x > 0.0) & (x < 2.0)
        return np.where(on, -6.25 / np.where(on, 6.25 - x ** 2, 1.0) ** 1.5, 0.0)

    def box(self):
        return (float(self.pts[:, 0].min()), float(self.pts[:, 0].max()),
                float(self.pts[:, 1].min()), float(self.pts[:, 1].max()))

    def clearance(self, x0):
        """Distance of the centre to the lower wall's tangent line at x
        (= the distance to the wall where it is straight; first order on the
        arc), with its gradient in (x, y)."""
        x, y = float(x0[0]), float(x0[1])
        w, dw, ddw = float(self.wall(x)), float(self.dwall(x)), float(self.d2wall(x))
        s = np.sqrt(1.0 + dw * dw)
        g = (y - w) / s
        return g, np.array([-dw / s - (y - w) * dw * ddw / s ** 3, 1.0 / s])

    def position_constraints(self, x0, which='a', margin=None):
        """The interior constraint of one port, as rows c(x0) <= 0 with their
        gradients (m = pos_margin; BFS.md 2):
          actuator centre  x_lo + m <= x <= x_hi - m, y <= y_hi - m,
                           clearance from the lower wall >= m;
          sensor window    on the floor: FLOOR_X0 <= x_s, x_s + L <= x_hi - m."""
        m = self.pos_margin if margin is None else margin
        x_lo, x_hi, _, y_hi = self.box()
        if which == 's':
            xs = float(np.ravel(x0)[0])
            return [(FLOOR_X0 - xs, np.array([-1.0])), (xs + self.sensor_len - x_hi + m, np.array([1.0]))]
        x, y = float(x0[0]), float(x0[1])
        g, dg = self.clearance(x0)
        return [(x_lo + m - x, np.array([-1.0, 0.0])), (x - x_hi + m, np.array([1.0, 0.0])),
                (y - y_hi + m, np.array([0.0, 1.0])), (m - g, -dg)]

    def inside(self, x0, which='a', margin=None):
        return all(c <= 0.0 for c, _ in self.position_constraints(x0, which, margin))

    def position_rows(self, xa, xs, flags):
        """The interior rows of the FREE ports for the SQP subproblem (hard
        linear rows c + g.d <= 0, sqp_inner.Evaluator): [(c, {'xa'|'xs': g})]."""
        out = []
        for key, side, x0, on in (('xa', 'a', xa, flags.get('B')), ('xs', 's', xs, flags.get('C'))):
            if not on:
                continue
            if np.ndim(x0) == 2:
                # MIMO: every port's rows, each gradient in the packer's (ports, dims) shape
                X = np.asarray(x0, float)
                for i in range(len(X)):
                    for c, g in self.position_constraints(X[i], side):
                        G = np.zeros_like(X); G[i] = g
                        out.append((c, {key: G}))
            else:
                out += [(c, {key: g}) for c, g in self.position_constraints(x0, side)]
        return out

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

    # ------------------------------------------------- all-frequency stability
    # The winding count of winding_cert, M-free (BFS.md 4): the plant's
    # numerical-range data and pole data are design-independent and computed
    # once per mesh (sparse_spectrum with massops.FORBIDDEN -> bfs/mfree.py),
    # cached on disk next to the operator file.
    def _range_path(self):
        return (self.ops_path[:-4] + '_range.npz') if self.ops_path else None

    def range_data(self):
        """(s_far, Hd, beta, nu1): the right edge s_far and the S-lemma
        heights Hd of the numerical range W of A_vv on the divergence-free
        velocities (sparse_spectrum.plant_heights), and the H^1 constant
        beta = lambda_max(sym(A_vv) + nu1 L, M) on ker B with nu1 = nu / 2 --
        all certified upper bounds (inertia).  Once per mesh."""
        if getattr(self, '_range', None) is not None:
            return self._range
        path = self._range_path()
        import os
        if path and os.path.exists(path):
            z = np.load(path)
            self._range = (float(z['s_far']), [tuple(r) for r in z['Hd']], float(z['beta']), float(z['nu1']))
            return self._range
        from sparse_spectrum import plant_heights
        from bfs.mfree import saddle_range
        # depths to 50: on the test plant the S-lemma heights reach the bare
        # bound max |Im W| by depth 50 (2447 s serial for GL's 17 depths, the
        # last 10 all equal); the bare bound, valid at every depth, closes the
        # list (numrange_distance then grows without a cap)
        s_far, Hd = plant_heights(self.A, self.M, depths=(0.5, 1, 2, 5, 10, 20, 50))
        R = saddle_range(self.A, self.M)
        T = ((self.A - self.A.T) * 0.5).tocsr(); Z = sp.csr_matrix(self.A.shape)
        h_bare = max(R.lmax(Z, -T, rtol=1e-2), R.lmax(Z, T, rtol=1e-2))
        Hd = list(Hd) + [(1e12, h_bare)]
        nu1 = 0.5 / self.Re
        H = ((self.A + self.A.T) * 0.5).tocsr()
        beta = R.lmax(H + nu1 * self.L)
        self._range = (float(s_far), [tuple(map(float, r)) for r in Hd], float(beta), float(nu1))
        if path:
            np.savez(path, s_far=self._range[0], Hd=np.array(self._range[1]), beta=beta, nu1=nu1)
        return self._range

    def _Llu(self):
        if getattr(self, '_Llu_', None) is None:
            import sparse_lu
            v = np.isin(np.arange(self.n), self.bc_dofs, invert=True) & (self.comp < 2)
            self._Lv = np.where(v)[0]
            self._Llu_ = sparse_lu.factor(self.L[self._Lv][:, self._Lv].tocsc())
        return self._Llu_

    def _mvv_floor(self):
        """A certified lower bound on lambda_min(M_vv), the velocity mass on the
        free velocity dofs (_Lv), with no factor and no inverse of it (the user's
        rule, massops.FORBIDDEN): the element floor of note_algebra (the GL
        tail's thrift level (b)).  M = sum_e P_e^T M_e P_e gives x^H M x >=
        min_e lambda_min(M_e) ||x||^2 (every dof is in an element), and M_vv is a
        principal submatrix of the assembled mass (Cauchy interlacing).  A P2
        component on a straight triangle has M_e = |T| M_ref, so the floor is
        lambda_min(M_ref) min |T|.  The premise is CHECKED: M is reassembled from
        pts / tri / xy (dofs matched by coordinates) and compared with the stored
        one on the free velocity block; the floor is lowered by a bound on the
        difference's norm, sqrt(||E||_1 ||E||_inf).  None when the reassembly does
        not match (another element, a curved geometry): the caller then has no
        L2 tail (gbound falls back to the H^1 bound alone).  Once per plant
        (test plant, 2026-10-16: 1.94e-5, a match to 6e-17; the true lambda_min
        is 1.08e-4)."""
        if getattr(self, '_mvv_floor_', 0) != 0:
            return self._mvv_floor_
        from math import factorial
        from scipy.spatial import cKDTree
        self._mvv_floor_ = None
        T = np.asarray(self.tri)
        if T.ndim != 2 or T.shape[1] != 3:
            print(f"    gbound: no velocity-mass floor (geometry cells of {T.shape[-1]} nodes, not straight "
                  f"triangles): the H^1 tail alone", flush=True)
            return None
        # M_ref / |T|: P2 Lagrange basis in barycentric coordinates, lam_i (2 lam_i - 1)
        # at the vertices, 4 lam_j lam_k at the midpoint of edge (j, k); int over T of
        # lam^a = 2 |T| a1! a2! a3! / (a1 + a2 + a3 + 2)!
        basis = []
        for i in range(3):
            e2 = [0, 0, 0]; e2[i] = 2; e1 = [0, 0, 0]; e1[i] = 1
            basis.append({tuple(e2): 2.0, tuple(e1): -1.0})
        for (j, k) in ((1, 2), (0, 2), (0, 1)):
            e = [0, 0, 0]; e[j] = 1; e[k] = 1
            basis.append({tuple(e): 4.0})
        Mref = np.zeros((6, 6))
        for p_ in range(6):
            for q_ in range(6):
                Mref[p_, q_] = sum(ca * cb * 2.0 * np.prod([factorial(x + y) for x, y in zip(ea, eb)])
                                   / factorial(sum(ea) + sum(eb) + 2)
                                   for ea, ca in basis[p_].items() for eb, cb in basis[q_].items())
        X = np.asarray(self.pts, float)[T]
        area = 0.5 * np.abs((X[:, 1, 0] - X[:, 0, 0]) * (X[:, 2, 1] - X[:, 0, 1])
                            - (X[:, 2, 0] - X[:, 0, 0]) * (X[:, 1, 1] - X[:, 0, 1]))
        nodes = np.concatenate([X, 0.5 * (X[:, [1, 0, 0]] + X[:, [2, 2, 1]])], axis=1).reshape(-1, 2)
        scale = float(np.ptp(self.pts, axis=0).max())
        rows, cols, vals, covered = [], [], [], []
        for comp in (0, 1):
            idx = np.where(self.comp == comp)[0]
            dist, j = cKDTree(self.xyd[idx]).query(nodes)
            if dist.max() > 1e-9 * scale:
                print(f"    gbound: no velocity-mass floor (a P2 node of the mesh is {dist.max():.2g} from every "
                      f"dof of component {comp}): the H^1 tail alone", flush=True)
                return None
            dofs = idx[j].reshape(-1, 6)
            rows.append(np.repeat(dofs, 6, axis=1).ravel()); cols.append(np.tile(dofs, (1, 6)).ravel())
            vals.append((area[:, None, None] * Mref[None]).ravel()); covered.append(dofs.ravel())
        Mre = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=self.M.shape)
        Lv = self._Lv
        if not np.isin(Lv, np.concatenate(covered)).all():
            print("    gbound: no velocity-mass floor (a free velocity dof in no element): the H^1 tail alone", flush=True)
            return None
        E = abs(self.M.tocsr()[Lv][:, Lv] - Mre[Lv][:, Lv])
        e_nrm = float(np.sqrt(E.sum(axis=0).max() * E.sum(axis=1).max())) if E.nnz else 0.0
        floor = float(np.linalg.eigvalsh(Mref).min()) * float(area.min())
        # (1 - 1e-9): the rounding of the 6 x 6 eigenvalue and of the areas
        floor = floor * (1.0 - 1e-9) - e_nrm
        if not (e_nrm <= 1e-8 * floor) or floor <= 0.0:
            print(f"    gbound: no velocity-mass floor (the reassembled P2 mass differs from M by {e_nrm:.3g} "
                  f"in norm, floor {floor:.3g}): the H^1 tail alone", flush=True)
            return None
        self._mvv_floor_ = floor
        return floor

    def tail_norms(self, c):
        """(cL, cM) of the sensor matrix c: cL = ||C||_{L^-1} = sqrt(lambda_max(C^H
        L^-1 C)) (H^1 tail) and cM = ||C||_{M_vv^-1} = sqrt(lambda_max(C_v^H M_vv^-1
        C_v)) (L2 tail; None where it does not apply), cached per sensor
        configuration (the bytes of c: the veto screens and counts several).

        cM never factorises or inverts the velocity mass (the user's rule): CG on
        M_vv (Jacobi-preconditioned, M_vv is SPD and well conditioned), then the
        residual-corrected bound, exact up to rounding: with R = C_v - M_vv Y,
        C_v^H M_vv^-1 C_v = C_v^H Y + Y^H R + R^H M_vv^-1 R, hence lambda_max <=
        lambda_max(sym(C_v^H Y + Y^H R)) + ||R||_2^2 / lambda_floor (_mvv_floor).
        CG's tolerance sets only how tight it is (test plant, the 369.87 record:
        64 iterations, 0.3 s, cM = 738.0455, the LU value), and a CG that does not
        converge only loosens it.  It needs C zero off the free velocity dofs
        (Cauchy-Schwarz in M_vv): checked, else None; None too, logged, when the
        L2 computation fails at all, the element floor included (the H^1 tail then
        stands alone: a failure here must not become a failed count).  A C with
        entries on pressure dofs has no tail bound at all (cL = inf: neither bound
        sees c^H p)."""
        import hashlib
        c = np.asarray(c, complex)
        self._Llu()
        cv = c[self._Lv]
        key = hashlib.blake2b(np.ascontiguousarray(c).tobytes() + str(c.shape).encode(), digest_size=16).hexdigest()
        cache = getattr(self, '_tail_norm_cache', None)
        if cache is None:
            cache = self._tail_norm_cache = {}
        if key in cache:
            return cache[key]
        Cv = cv.reshape(cv.shape[0], -1)
        LC = np.column_stack([self._Llu_.solve(Cv[:, j]) for j in range(Cv.shape[1])])
        cL = float(np.sqrt(max(np.linalg.eigvalsh(Cv.conj().T @ LC).max(), 0.0)))
        cM = None
        cr = c.reshape(self.n, -1)
        off = np.ones(self.n, bool); off[self._Lv] = False
        if np.any(cr[self.comp >= 2] != 0):
            # c^H x then reads the pressure, which neither bound controls (both bound
            # c^H u through the velocity alone): no tail bound at all -- gbound inf, the
            # count's TailBoundError before any march -- not the H^1 bound on c_v this
            # branch claimed (the review of 2026-10-17; the shear sensor has no pressure
            # entries, so no run took it)
            print("    gbound: the sensor has entries on pressure dofs: no tail bound (neither the H^1 nor the L2 "
                  "one bounds c^H p)", flush=True)
            cL = np.inf
        elif np.any(cr[off] != 0):
            # on the Dirichlet dofs only: the response is zero there (M's rows are zero,
            # A's -I), so c^H x = c_v^H u_v and the H^1 bound holds; the L2 one keeps its
            # stricter premise
            print("    gbound: the sensor has entries on Dirichlet dofs: the H^1 tail alone", flush=True)
        else:
            # the floor inside the try too (the review of 2026-10-17): a failure of its
            # reassembly must leave the H^1 tail, not fail the count
            try:
                lf = self._mvv_floor()
                if lf is not None:
                    cM = self._cM_bound(Cv, lf)
            except Exception as _e:
                print(f"    gbound: the L2 tail norm failed ({type(_e).__name__}: {_e}): the H^1 tail alone", flush=True)
        if len(cache) >= 8:
            cache.pop(next(iter(cache)))
        cache[key] = (cL, cM)
        return cL, cM

    def _cM_bound(self, Cv, lf):
        """sqrt of the certified upper bound on lambda_max(C_v^H M_vv^-1 C_v) of
        tail_norms: Jacobi-preconditioned CG on M_vv (scipy's, or the same few
        lines where its keywords differ), never a factor of it."""
        import scipy.sparse.linalg as spla
        Mvv = getattr(self, '_Mvv_', None)
        if Mvv is None:
            Mvv = self._Mvv_ = self.M.tocsr()[self._Lv][:, self._Lv].tocsr()
        dg = Mvv.diagonal()
        P = spla.LinearOperator(Mvv.shape, matvec=lambda x: x / dg, dtype=float)

        def _cg(rhs):
            try:
                return spla.cg(Mvv, rhs, rtol=1e-13, atol=0.0, maxiter=20000, M=P)[0]
            except TypeError:                   # scipy < 1.12 calls rtol tol
                return spla.cg(Mvv, rhs, tol=1e-13, atol=0.0, maxiter=20000, M=P)[0]
        cols = []
        for j in range(Cv.shape[1]):
            # M_vv is real: the real and imaginary parts are two real solves
            y = np.zeros(Cv.shape[0], complex)
            for part, unit in ((Cv[:, j].real, 1.0), (Cv[:, j].imag, 1j)):
                if np.any(part):
                    y = y + unit * _cg(part)
            cols.append(y)
        Y = np.column_stack(cols)
        R = Cv - Mvv @ Y
        S = Cv.conj().T @ Y + Y.conj().T @ R
        ub = float(np.linalg.eigvalsh(0.5 * (S + S.conj().T)).max()) + float(np.linalg.norm(R, 2)) ** 2 / lf
        if not np.isfinite(ub):
            raise FloatingPointError(f"the bound is {ub}")
        # (1 + 1e-9): the rounding of the length-n_v products (n_v eps ~ 6e-12 relative)
        return float(np.sqrt(max(ub, 0.0) * (1.0 + 1e-9)))

    def gbound(self, pa, c, line):
        """w -> a bound on |G(line + i w')| for every |w'| >= w, G = c^H (s M -
        A)^-1 M psi_a, M-free (note_algebra BFS section), the smaller of two:
            H1  |G| <= ||c||_{L^-1} ||psi_a||_M sqrt((1/d + (beta - line)_+ / d^2) / nu1)
            L2  |G| <= ||c||_{M_vv^-1} ||psi_a||_M / d
        with d <= dist(line + i w, W) from the heights (non-decreasing in w, so
        both bounds hold beyond w).  The response x = (u, p) has u = 0 on the
        Dirichlet dofs and B u = 0, and step (i) of the note's derivation gives
        ||u||_M <= ||psi_a||_M / d.  The shear sensor is not of the form M psi;
        H1 bounds c^H u by ||u||_L (the viscous term), L2 by Cauchy-Schwarz in
        M_vv: |c^H u| = |(M_vv^-1 c_v)^H M_vv u| <= ||c||_{M_vv^-1} ||u||_M, c being
        zero off the free velocity dofs (tail_norms checks it).  H1 decays like
        d^-1/2, L2 like 1/d: L2 is the smaller from w ~ 100 up (369.87 record:
        51.9 vs 102.5 at w = 100, 1.5e-4 vs 0.16 at 5e6), and the cutoff W_T it
        needs grows linearly in the feedthrough instead of with its square
        (2026-10-16; with H1 alone the 369.87 record needed W_T 7.9e6 against the
        count's cap of 1e7 and every later decentral design of 20261010n raised).
        The returned function carries the norms (gb.norms: cL, cM, pM) for the
        driver's tail line."""
        from sparse_spectrum import numrange_distance
        s_far, Hd, beta, nu1 = self.range_data()
        cL, cM = self.tail_norms(c)
        PA = np.asarray(pa, complex)
        if PA.ndim == 2:
            # MIMO: operator norms, ||G||_2 <= ||C|| ||Psi_a||_M (...) with
            # ||Psi_a||_M^2 = lambda_max(Psi_a^H M Psi_a)
            pM = float(np.sqrt(max(np.linalg.eigvalsh(PA.conj().T @ (self.M @ PA)).max(), 0.0)))
        else:
            pM = float(np.sqrt(max(np.real(np.vdot(PA, self.M @ PA)), 0.0)))
        ex = max(beta - line, 0.0)

        def gb(w):
            d = numrange_distance(w, s_far, Hd, line=line)
            if d <= 0:
                return np.inf
            g1 = cL * pM * np.sqrt((1.0 / d + ex / d ** 2) / nu1)
            return g1 if cM is None else min(g1, cM * pM / d)
        gb.norms = dict(cL=cL, cM=cM, pM=pM)
        return gb

    def plant_poles(self, line, keep=0.6):
        """(count right of line, hot list) of the plant (winding_cert), from the
        certified sparse enclosure (sparse_spectrum.plant_eigs_right, M-free),
        cached on disk per line."""
        cache = getattr(self, '_poles', None)
        if cache is None:
            cache = self._poles = {}
        key = round(float(line), 9)
        if key not in cache:
            import os
            path = self._range_path()
            pp = path[:-4] + f'_poles_{key:+.6f}.npz' if path else None
            if pp and os.path.exists(pp):
                z = np.load(pp)
                cache[key] = (int(z['count']), [tuple(h) for h in z['hot']])
            else:
                from sparse_spectrum import plant_pole_data_sparse
                cnt, hot = plant_pole_data_sparse(self.A, self.M, line, keep)
                cache[key] = (cnt, hot)
                if pp:
                    np.savez(pp, count=cnt, hot=np.array(hot).reshape(-1, 2))
        return cache[key]

    def winding(self, pa, b, c, arow, nvec, K, line, zeros_hint=None, hmin_abs=None):
        """#{closed-loop eigenvalues with Re > line}, all frequencies
        (winding_cert.winding_count with the M-free plant data and tail).
        hmin_abs: an absolute step floor for the march (the driver's counts right
        of -eps pass one, 20261010j stage B: the relative floor 1e-6 x min(W_T,
        4000) is as coarse as the 1e-4 feature a design pinned at the line makes)."""
        import winding_cert as wc
        cnt, hot = self.plant_poles(line)
        if not np.any(nvec) and not np.any(K):
            # the open loop (zero controller): f = 1 - k G = 1 identically, so the
            # count is the plant's (certified) plus the compensator's, exactly
            import controller as ctl
            lam_c = np.linalg.eigvals(ctl.A_of(*(np.asarray(x, complex) for x in arow[:3])))
            pc = int(np.sum(lam_c.real > line))
            return dict(count=cnt + pc, winding=0, winding_raw=0.0, p_plant=int(cnt), p_ctrl=pc, npts=0,
                        nsolve=0, wtail=0.0, ok=True, open_loop=True)
        return wc.winding_count(self.A, self.M, b, c, arow, nvec, K, line=line, mc=self.mc, p_plant=cnt, hot=hot,
                                zeros_hint=zeros_hint, ctrl_exact=True, gbound=self.gbound(pa, c, line), real_sym=True,
                                hmin_abs=hmin_abs)

    def tail_check(self, pa, b, c, arow, nvec, K, line):
        """The tail cutoff `winding` would march to, without the march
        (winding_cert.tail_check with this plant's gbound; milliseconds once the
        sensor's norms are cached): dict(wtail, wstar, q, kgb, kD, cL, cM, pM).
        Raises winding_cert.TailBoundError where the count would (the driver's
        veto pre-screen and its tail line at each acceptance, 2026-10-16)."""
        import winding_cert as wc
        gb = self.gbound(pa, c, line)
        out = wc.tail_check(self.A, self.M, b, c, arow, nvec, K, line=line, mc=self.mc, gbound=gb)
        out.update(gb.norms)
        return out

    def cl(self, b, c):
        if np.ndim(b) == 2 or np.ndim(c) == 2:
            from closedloop_mimo import ClosedLoopResolventMIMO
            R = ClosedLoopResolventMIMO(self.M, self.A, b, c, r=self.r, mc=self.mc)
        else:
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
