"""reduced.py -- the projected oracle (Stage 5; algebra note sections on projection).

Two-sided Petrov-Galerkin reduction of the PLANT.  With V, W in C^{n x k}
orthonormal,

    A~ = W^H A V,   M~ = W^H M V          (once per basis)
    b~ = W^H b,     c~ = V^H c            (per design: two matvecs)

and the closed loop keeps its Woodbury structure with n -> k: the frozen
ClosedLoop class is instantiated on the k x k dense plant, so setup / solve /
solveH / design_grad_tri are the same code.  The design enters only through
the scalar k(s) and the two small vectors -- the formal reason the basis is
reusable across designs (note: 5.6e-6 at a fixed design, 1e-4 across designs
at dim 40 on GL).

Singular values.  The full oracle's gain() solves the Euclidean eigenproblem
of M T^H M T, whose eigenvalues are ||M^{1/2} T M^{1/2}||^2: loads are measured
in the M^{-1} norm, states in the M norm.  Reduced: with T~ = V R~ W^H,

    sigma_j = svd( G_V^{1/2}  R~  G_W^{1/2} ),    G_V = V^H M V,  G_W = W^H M W,

a k x k dense SVD.  The pairs (x, z) that D.grads expects are lifted:
x_p = V x~_p, z_p = W z~_p, controller parts as computed, normalised as gain()
does (||x_p||_M = sigma).

Enrichment (note, "our variant"): at an anchor omega_a, add the plant-state
parts of the top-n_vec singular pairs of the FULL closed loop at omega_a (they
span R M^{1/2} v_j and R b) to V, the adjoint parts to W, plus the two Woodbury
columns R b, R^H c from the plant LU the full oracle already holds; then
re-orthonormalise.  Interpolation theorem: sigma_1, sigma_2 and d sigma/d
omega of the reduced function equal the full ones at the anchors, per vector.

The certified lower bound of the note (one-sided ||T V||) is not built here.
The reduced model is never trusted: the outer certification stays on the full
model and decides where to enrich (Stage-5 gate: identical certified gamma,
zero factorisations per inner iteration, reduced-vs-certified 1e-5 at peaks).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sparse_lu
from scipy.linalg import sqrtm, svd, solve

from closedloop_sparse import ClosedLoopResolvent as ClosedLoop

ORTH = 'block'          # basis orthogonalisation: 'block' or 'column' (driver --oracle-orth)
BATCH_BYTES = 512 * 2 ** 20   # gain_batch working memory per chunk
SVCACHE_MAX = 20000     # warm-start subspaces kept by frequency (gain_batch), ~kd x 6 complex each
UNION = 'full'          # union Galerkin step: 'full' (W against V entire) or 'new' (W's new columns only; --oracle-union)


class _ProxyCL:
    """Stands in for the full ClosedLoop in D.grads for a REDUCED row.  The
    closed-loop class computes the controller gradient from the singular pair
    at ITS n (= kd), so it must see the reduced vectors.  (Handing it the lifted
    n-vectors sliced them at kd: gradient cosine 0.27 against the finite
    difference, inner solve stuck at its start.)
    Since STAGE6 6.29 the rows carry the REDUCED pair x~ = [x~_p; x_c],
    z~ = [z~_p; z_c] (x_p = V x~_p, z_p = W z~_p) and grads_reduced forms every
    gradient without a length-n vector: port sides with W^H B, V^H C, placement
    terms with W^H M dpsi(x_a), V^H M dpsi(x_s) (once per design).  A full-length
    pair (legacy callers) is still accepted and projected (_reduce)."""
    is_reduced = True

    def __init__(self, b, c, rcl, V, W, n):
        self.b, self.c, self._r, self._V, self._W, self._n = b, c, rcl, V, W, n
        self.kd = V.shape[1]
        self._pl = {}
        self._fimg = None                  # field-gradient images, once per design (None: not built)

    def _reduce(self, res):
        x, z = res['x'], res['z']
        if len(x) == self.kd + self._r.r:
            return res
        n = self._n
        xr = np.concatenate([self._V.conj().T @ x[:n], x[n:]])
        zr = np.concatenate([self._W.conj().T @ z[:n], z[n:]])
        return dict(res, x=xr, z=zr)

    def design_grad_tri(self, res):
        return self._r.design_grad_tri(self._reduce(res))

    def design_grad(self, res):
        return self._r.design_grad(self._reduce(res))

    def lift(self, x, z):
        """The full-length pair (field gradients until they are reduced too)."""
        kd = self.kd
        return (np.concatenate([self._V @ x[:kd], x[kd:]]), np.concatenate([self._W @ z[:kd], z[kd:]]))

    def placement(self, D, side, x0):
        """W^H M dpsi(x0) (actuator side) / V^H M dpsi(x0) (sensor side), once per
        design and position."""
        key = (side, float(x0))
        v = self._pl.get(key)
        if v is None:
            B = self._W if side == 'a' else self._V
            v = self._pl[key] = B.conj().T @ D.Mdpsi_dx(float(x0))
        return v

    def grads_reduced(self, D, x, z, arow, nvec, Kfb, xa, xs, pa, ps, na, ns, flags, eta=1.0):
        """fe_codesign.grads on the reduced pair (SISO), term by term."""
        kd = self.kd
        if (flags.get('FB') or flags.get('FC')) and not self._fields_reducible(D, pa, ps, na, ns):
            xf, zf = self.lift(x, z)
            return D.grads(self, xf, zf, arow, nvec, Kfb, xa, xs, pa, ps, na, ns, flags, eta=eta, _full=True)
        xp, xc, zp, zc = x[:kd], x[kd:], z[:kd], z[kd:]
        nv = np.asarray(nvec, complex).ravel()
        out = {}
        if flags['K']:
            if isinstance(arow, (tuple, list)) and len(arow) == 3:
                g = self._r.design_grad_tri(dict(x=x, z=z))
                for k_ in ('d', 'e', 'f', 'n'):
                    out[k_] = np.asarray(g[k_]) / eta
                out['K'] = g['K'] / eta
            else:
                g = self._r.design_grad(dict(x=x, z=z))
                out['a'] = np.asarray(g['a']) / eta; out['n'] = np.asarray(g['n']) / eta; out['K'] = g['K'] / eta
        bt = np.asarray(self._r.b, complex).ravel(); ct = np.asarray(self._r.c, complex).ravel()
        Sa = Kfb * np.vdot(ct, xp) + np.dot(nv, xc)
        Sc = np.vdot(zp, bt) * Kfb + D.mc * np.sum(np.conj(zc))
        if flags['B']:
            out['xa'] = float(np.real(np.vdot(zp, self.placement(D, 'a', xa)) * Sa / eta))
        if flags['C']:
            out['xs'] = float(np.real(Sc * np.vdot(self.placement(D, 's', xs), xp) / eta))
        if flags.get('FB'):
            out['va_packed'] = self._field_row('va', 0, zp, np.conj(Sa) / eta)
        if flags.get('FC'):
            out['vs_packed'] = self._field_row('vs', 0, xp, Sc / eta)
        return out

    # ---------------- field gradients in the reduced space (STAGE6 6.29) ----------------
    # The nodal field gradient of a row is retract(g) with g = s M W z~ (actuator)
    # or s M V x~ (sensor), retract(g) = (g - Re(g^H psi) M psi) / nv, and the
    # packer maps it to the field parameters by Re(G^H J) (J = pk.field_jacobians).
    # So the packed gradient is [Re(conj(s) u~^H X) - Re(conj(s) u~^H h) y] / nv with
    # X = B^H M J, h = B^H M psi, y = Re(psi^H M J), B = W or V: projected ONCE per
    # design, O(kd P) per row instead of the length-n lift.
    def _fields_reducible(self, D, pa, ps, na, ns):
        pk = getattr(self, 'pk', None)
        if pk is None or np.iscomplexobj(D.M.data):
            return False
        if self._fimg is None:
            jac = pk.field_jacobians()
            if any(J is None for L in jac.values() for J in L):
                self._fimg = False
            else:
                img = {}
                for key, L in jac.items():
                    B = self._W if key == 'va' else self._V
                    P_, N_ = (pa, na) if key == 'va' else (ps, ns)
                    for j, J in enumerate(L):
                        psi = np.asarray(P_, complex).reshape(D.n, -1)[:, j]
                        nv = float(np.atleast_1d(N_)[j])
                        MJ = D.M @ J; Mpsi = D.M @ psi
                        img[(key, j)] = (B.conj().T @ MJ, B.conj().T @ Mpsi, np.real(psi.conj() @ MJ), nv)
                self._fimg = img
        return self._fimg is not False

    def _field_row(self, key, j, u, s):
        X, h, y, nv = self._fimg[(key, j)]
        cs = np.conj(s)
        t = float(np.real(cs * np.vdot(u, h)))
        return (np.real(cs * (np.conj(u) @ X)) - t * y) / nv


class _ProxyCLMIMO(_ProxyCL):
    """MIMO proxy (STAGE6): the controller gradient of the REDUCED loop and the
    port-side scalars from the reduced ports W^H B, V^H C."""
    def __init__(self, b, c, rcl, V, W, n):
        super().__init__(b, c, rcl, V, W, n)
        self.m, self.p = rcl.m, rcl.p

    def port_sides(self, res, arow, nvec, K):
        res = self._reduce(res)
        kd, r, m, p = self.kd, self._r.r, self.m, self.p
        x, z = res['x'], res['z']
        xp, xc, zp, zc = x[:kd], x[kd:], z[:kd], z[kd:]
        Cc = np.asarray(nvec, complex).reshape(m, r); Dc = np.asarray(K, complex).reshape(m, p)
        zB = np.conj(np.asarray(self._r.b, complex).conj().T @ zp)
        act = Dc @ (np.asarray(self._r.c, complex).conj().T @ xp) + Cc @ xc
        sen = zB @ Dc + np.conj(zc) @ self._r.Bc_of(arow)
        return act, sen

    def grads_reduced(self, D, x, z, arow, nvec, Kfb, xa, xs, pa, ps, na, ns, flags, eta=1.0):
        """fe_codesign.grads_mimo on the reduced pair, term by term."""
        if (flags.get('FB') or flags.get('FC')) and not self._fields_reducible(D, pa, ps, na, ns):
            xf, zf = self.lift(x, z)
            return D.grads(self, xf, zf, arow, nvec, Kfb, xa, xs, pa, ps, na, ns, flags, eta=eta, _full=True)
        kd = self.kd
        xp, zp = x[:kd], z[:kd]
        out = {}
        res = dict(x=x, z=z)
        if flags['K']:
            g = self._r.design_grad_tri(res)
            for k_ in ('d', 'e', 'f', 'Bc', 'n', 'K'):
                out[k_] = np.asarray(g[k_]) / eta
        act, sen = self.port_sides(res, arow, nvec, Kfb)
        if flags['B']:
            out['xa'] = np.array([float(np.real(np.vdot(zp, self.placement(D, 'a', xa[i])) * act[i] / eta))
                                  for i in range(self.m)])
        if flags['C']:
            out['xs'] = np.array([float(np.real(sen[j] * np.vdot(self.placement(D, 's', xs[j]), xp) / eta))
                                  for j in range(self.p)])
        if flags.get('FB'):
            out['va_packed'] = [self._field_row('va', i, zp, np.conj(act[i]) / eta) for i in range(self.m)]
        if flags.get('FC'):
            out['vs_packed'] = [self._field_row('vs', j, xp, sen[j] / eta) for j in range(self.p)]
        return out


class ReducedOracle:
    def __init__(self, D):
        self.D = D
        self.n = D.n
        self.M = D.M.tocsc(); self.A = D.A.tocsc()
        self.V = np.zeros((self.n, 0), complex)
        self.W = np.zeros((self.n, 0), complex)
        self.anchors = []
        self.nfac_inner = 0                      # must stay 0: the gate
        self._proj = None

    # ------------------------------------------------------------ basis
    def _orth(self, Q, X):
        """Append columns X to orthonormal Q (ORTH selects the implementation).
        Deflation is RELATIVE in both: a column is kept while its remaining norm
        exceeds 1e-8 of its original norm (resolvent images have norms ~1e6, and
        an absolute 1e-10 let numerically dependent columns through: a basis of
        738 > n = 399, a singular pencil)."""
        return (self._orth_block if ORTH == 'block' else self._orth_column)(Q, X)

    @staticmethod
    def _orth_block(Q, X):
        """Two BLOCK Gram-Schmidt passes against Q (BLAS3), then the new
        columns one by one against the columns already kept from X only (CGS2),
        then one more block pass against Q (STAGE6 6.29).  Same deflation
        decisions as _orth_column (same test on the same remainder).  The
        columns differ from _orth_column's by rounding -- up to 2e-8 on the
        near-dependent ones, whose remainder is 1e-7 of the original norm: the
        first projection leaves eps x (original norm) in the complement, in
        either implementation."""
        X = np.asarray(X, complex)
        if X.shape[1] == 0:
            return Q
        n0 = np.linalg.norm(X, axis=0)
        for _ in range(2):
            if Q.shape[1]:
                X = X - Q @ (Q.conj().T @ X)
        U = np.zeros((X.shape[0], 0), complex)
        for j in range(X.shape[1]):
            x = X[:, j]
            for _ in range(2):
                if U.shape[1]:
                    x = x - U @ (U.conj().T @ x)
            nrm = np.linalg.norm(x)
            if nrm > 1e-8 * max(n0[j], 1e-300):
                U = np.column_stack([U, x / nrm])
        if U.shape[1] == 0:
            return Q
        if Q.shape[1]:
            U = U - Q @ (Q.conj().T @ U)
            return np.hstack([Q, U])
        return U

    @staticmethod
    def _orth_column(Q, X):
        """The reference implementation (until STAGE6 6.29): every column
        re-projected against the WHOLE basis, twice (BLAS2 with the full basis
        per column: ~47 s per union step at n = 1e5, kd = 130; 4x the block
        version at +12 columns).  Kept for the SISO agreement check against the
        siso-v1 tree (tools/check_siso_identity.sh, --oracle-orth column)."""
        x0 = np.asarray(X, complex).copy()
        for _ in range(2):
            if Q.shape[1]:
                X = X - Q @ (Q.conj().T @ X)
        keep = []
        for j in range(X.shape[1]):
            x = X[:, j]
            n0 = np.linalg.norm(x0[:, j])
            for _ in range(2):
                for q in keep:
                    x = x - q * np.vdot(q, x)
                if Q.shape[1]:
                    x = x - Q @ (Q.conj().T @ x)
            nrm = np.linalg.norm(x)
            if nrm > 1e-8 * max(n0, 1e-300):
                keep.append(x / nrm)
        if keep:
            Q = np.column_stack([Q] + keep) if Q.shape[1] else np.column_stack(keep)
        return Q

    def set_input_basis(self, Phi_b=None, Psi_c=None):
        """INPUT-FIELD BASIS (TRACKER I.1): orthonormal columns spanning the
        actuator vectors b(x_a) over the actuator range (Phi_b) and the sensor
        vectors c(x_s) over the sensor range (Psi_c).  Every later anchor adds
        R(iw) Phi_b to V and R(iw)^H Psi_c to W -- triangular solves with the
        LU the anchor already holds, no factorisation -- so the model is exact
        in the positions across the range (n-width: ~20 modes at 1e-8 for
        sigma = 0.4 over a range of 3)."""
        self.Phi_b, self.Psi_c = Phi_b, Psi_c
        self.galerkin = True          # the input basis requires the union space (see enrich)

    @staticmethod
    def enrich_columns(cl_full, omega, arow, nvec, K, nvec_dirs=2, Phi_b=None, Psi_c=None):
        """The anchor's new columns (one full-model factorisation at omega):
        n_vec singular pairs, the two Woodbury columns, the input-basis images.
        Independent of the basis, so the anchors of a round are computed in the
        full-model pool (enrich_many) and applied in order."""
        pairs = cl_full.gain_k(float(omega), arow, nvec, K, k=max(1, nvec_dirs))
        lu = cl_full._st['lu']
        n = cl_full.n
        cols_V = [p['x'][:n] for p in pairs] + [lu.solve(cl_full.b)]
        cols_W = [p['z'][:n] for p in pairs] + [lu.solve(cl_full.c, trans='H')]
        # tangential interpolation: INPUT directions to the right (V), OUTPUT
        # directions to the left (W).  (A first version added both to both
        # sides to force equal dimensions; the forced truncation cut
        # interpolating columns and the model was wrong even at its anchors.)
        if Phi_b is not None:                           # all columns in one solve (STAGE6 6.29)
            Yb = lu.solve(np.ascontiguousarray(Phi_b, complex)); cols_V += [Yb[:, k] for k in range(Yb.shape[1])]
        if Psi_c is not None:
            Yc = lu.solve(np.ascontiguousarray(Psi_c, complex), trans='H'); cols_W += [Yc[:, k] for k in range(Yc.shape[1])]
        return np.column_stack(cols_V), np.column_stack(cols_W)

    def enrich(self, cl_full, omega, arow, nvec, K, nvec_dirs=2, cols=None, project=True):
        """One full-model factorisation at omega (the one the certifier makes
        anyway), n_vec singular pairs, the two Woodbury columns (`cols`: the
        columns already computed, enrich_many; project=False: the projection is
        left to the caller, which makes it once after its last anchor)."""
        XV, XW = cols if cols is not None else self.enrich_columns(
            cl_full, omega, arow, nvec, K, nvec_dirs, getattr(self, 'Phi_b', None), getattr(self, 'Psi_c', None))
        k0 = self._union_k0()
        self.V = self._orth(self.V, XV)
        self.W = self._orth(self.W, XW)
        # equal dimension is required for a square reduced pencil
        if getattr(self, 'galerkin', False):
            # UNION GALERKIN: V = W = orth(V u W).  Contains R b and R^H c, so
            # Hermite interpolation at the anchors holds; the pencil
            # U^H (iwM - A) U does not lose rank the way the oblique coupling
            # of unrelated input/output spaces does (cond 1e18 measured).
            U = self._orth(self.V, self.W[:, k0:])
            self.V, self.W = U, U.copy(); self.cut = (0, 0); k = U.shape[1]
        else:
            k = min(self.V.shape[1], self.W.shape[1])
            self.cut = (self.V.shape[1] - k, self.W.shape[1] - k)
            self.V, self.W = self.V[:, :k], self.W[:, :k]
        self.anchors.append(float(omega))
        if project:
            self._project()
        return k

    def _union_k0(self):
        """Columns of W the union step may skip: with --oracle-union new and V = W
        (every union step), W's leading columns ARE V's and deflate to nothing
        against it -- only the new ones are projected (O(n kd k_new) instead of
        O(n kd^2); same span, same kept columns, rounding-level different
        vectors, STAGE6 6.30).  0 = the full step (the default)."""
        if UNION == 'new' and getattr(self, 'galerkin', False) and self.V.shape == self.W.shape \
                and np.array_equal(self.V, self.W):
            return self.V.shape[1]
        return 0

    def enrich_many(self, cl_full, omegas, arow, nvec, K, nvec_dirs=2, bc=None):
        """enrich() at every omega, in order.  With the full-model pool up
        (fullpar) and the design's (b, c) given, the anchors' factorisations and
        columns are computed in the workers; the basis updates stay in order
        here, so the basis is the serial one bit for bit (STAGE6 6.30)."""
        import fullpar
        omegas = [float(w) for w in omegas]
        if not omegas:
            return self.dim
        if bc is None or not fullpar.active() or len(omegas) < 2:
            cols = [None] * len(omegas)
        else:
            cols = fullpar.pmap(_enrich_task, [(bc[0], bc[1], arow, nvec, K, w, nvec_dirs) for w in omegas])
        # the basis updates in order; the projection (A V, M V, the Gram
        # matrices, the QZ form: O(nnz kd + n kd^2 + kd^3)) once, after the
        # last one -- it depends on the final basis only (STAGE6 6.30)
        for w, c_ in zip(omegas, cols):
            self.enrich(cl_full, w, arow, nvec, K, nvec_dirs=nvec_dirs, cols=c_, project=False)
        self._project()
        return self.dim

    def enrich_eig(self, cl_full, lams, arow, nvec, K, bc=None):
        """Right/left eigenvectors of the FULL closed loop at known eigenvalues
        (one factorisation each, shift-invert at the located value) into V, W:
        the frequency-response anchors do not put the near-line eigenvectors
        in the basis, and without them the reduced pencil need not contain the
        rightmost eigenvalue at all.  These are the watch-list eigenvalues the
        continuation gate already tracks.  With the full-model pool up and the
        design's (b, c) given, the eigenvector solves run in the workers."""
        import fullpar
        n = self.n; cv, cw = [], []
        if bc is not None and fullpar.active() and len(lams) > 1:
            vecs = fullpar.pmap(_eigvec_task, [(bc[0], bc[1], arow, nvec, K, complex(lam)) for lam in lams])
        else:
            vecs = [cl_full.eigvecs_at(complex(lam), arow, nvec, K) for lam in lams]
        for v, u, res in vecs:
            if v is None or not np.isfinite(res) or res > 1e-6:
                continue
            cv.append(v[:n]); cw.append(u[:n])
        if cv:
            k0 = self._union_k0()
            self.V = self._orth(self.V, np.column_stack(cv))
            self.W = self._orth(self.W, np.column_stack(cw))
            if getattr(self, 'galerkin', False):
                U = self._orth(self.V, self.W[:, k0:]); self.V, self.W = U, U.copy()
            else:
                k = min(self.V.shape[1], self.W.shape[1])
                self.V, self.W = self.V[:, :k], self.W[:, :k]
            self._project()
        return len(cv)

    def _project(self):
        V, W = self.V, self.W
        self._proj_ver = getattr(self, '_proj_ver', 0) + 1     # the parallel row pool follows the basis
        At = W.conj().T @ (self.A @ V)
        Mt = W.conj().T @ (self.M @ V)
        GV = V.conj().T @ (self.M @ V); GV = 0.5 * (GV + GV.conj().T)
        GW = W.conj().T @ (self.M @ W); GW = 0.5 * (GW + GW.conj().T)
        GW_h = sqrtm(GW)
        # Gram matrices for FULL-MODEL residuals of reduced vectors (STAGE6 6.29):
        # ||(A - l M) V y + b w|| from kd x kd products, no lift per pair
        AV = self.A @ V; MV = self.M @ V
        self._proj = dict(At=At, Mt=Mt, GV=GV, GW=GW,
                          GV_h=sqrtm(GV), GW_h=GW_h, GW_mh=np.linalg.inv(GW_h),
                          GAA=AV.conj().T @ AV, GAM=AV.conj().T @ MV, GMM=MV.conj().T @ MV)
        # Hessenberg-triangular (QZ) form of the reduced pencil, ONCE per basis
        # update: At = Q AA Z^H, Mt = Q BB Z^H with AA, BB upper triangular, so
        # (i w Mt - At)^{-1} y = Z (i w BB - AA)^{-1} Q^H y -- one triangular
        # solve, O(k^2), at any frequency, with no factorisation (Laub's method;
        # backward-stable, unlike diagonalising a non-normal pencil)
        try:
            from scipy.linalg import qz
            AA, BB, Q, Z = qz(At, Mt, output='complex')
            self._proj.update(qzAA=AA, qzBB=BB, qzQ=Q, qzZ=Z,
                              qzQH=np.ascontiguousarray(Q.conj().T), qzZH=np.ascontiguousarray(Z.conj().T))
        except Exception:
            pass

    @property
    def dim(self):
        return self.V.shape[1]

    def compact(self, cl_full, anchors, arow, nvec, K, lams=(), nvec_dirs=3, bc=None):
        """Restart the basis from a set of anchors (frequencies) and eigenvalues
        at the CURRENT design: the basis grows ~10 per round with old
        directions from designs since left behind; past a cap it is rebuilt
        from what the current design needs.  One factorisation per anchor and
        per eigenvalue, the same as the round's anchoring."""
        self.V = np.zeros((self.n, 0), complex); self.W = np.zeros((self.n, 0), complex)
        self.anchors = []
        self.enrich_many(cl_full, anchors, arow, nvec, K, nvec_dirs=nvec_dirs, bc=bc)
        if lams:
            self.enrich_eig(cl_full, lams, arow, nvec, K, bc=bc)
        return self.dim

    # ------------------------------------------------------------ oracle
    def make_cl(self, b, c, r, mc):
        """The reduced closed loop for a design's fields: the frozen class on
        the k x k plant.  Its 'factorisations' are k x k dense; not counted."""
        P = self._proj
        if np.ndim(b) == 2 or np.ndim(c) == 2:                     # MIMO (STAGE6)
            from closedloop_mimo import ClosedLoopResolventMIMO, as_ports
            B, C = as_ports(b, c)
            rcl = ClosedLoopResolventMIMO(sp.csc_matrix(P['Mt']), sp.csc_matrix(P['At']),
                                          self.W.conj().T @ B, self.V.conj().T @ C, r, mc=mc)
            out = _ReducedCL(self, rcl, B, C)
        else:
            rcl = ClosedLoop(sp.csc_matrix(P['Mt']), sp.csc_matrix(P['At']),
                             self.W.conj().T @ b, self.V.conj().T @ c, r, mc=mc)
            out = _ReducedCL(self, rcl, b, c)
        out._mk = (r, mc, getattr(self, '_proj_ver', 0))
        return out


def _fit_subspace(Vc, kd, p, V0):
    """A cached warm-start subspace (kd x p') resized to p columns: the leading
    p when p' >= p, else completed by the random start's columns and
    re-orthonormalised; V0 when there is none.  (2026-09-30: the k = 1 rows
    cache 5 columns and the k = 2 sigma_2 blocks 6, at the SAME keys -- a
    followed top is evaluated by both in every evaluation -- and each discarded
    the other's entry as the wrong shape: every block row restarted from the
    random subspace, 4x the iterations, measured.)"""
    if Vc is None or Vc.shape[0] != kd:
        return V0
    if Vc.shape[1] >= p:
        return Vc[:, :p]
    return np.linalg.qr(np.hstack([Vc, V0[:, :p - Vc.shape[1]]]))[0]


ROW_WORKERS = 0           # forked processes evaluating rows besides this one (--row-workers; 0 = serial)
PAR_MIN_ROWS = 32         # smaller batches stay in this process (a round trip costs ~1 ms)


class _RowPool:
    """Frequency-parallel gain_batch (2026-09-29, user: parallelise the rows).
    ROW_WORKERS processes are FORKED from this one, so each holds the reduced
    oracle as it is at that moment (basis, projection, QZ form) without pickling
    it; they are re-forked whenever the basis changes (_project increments
    _proj_ver).  A call sends each worker the design (ports B, C, compensator,
    frequencies) -- O(n m) numbers -- and the worker rebuilds the per-design
    reduced closed loop with make_cl and runs the serial gain_batch on its
    share; this process does its own share meanwhile.  A frequency always goes
    to the same process (blocks of 0.064 in w), so each keeps warm subspace
    caches for its frequencies, as the serial path does for all of them."""
    BLOCK = 64                           # w-block of 0.064 per slot (key = round(1000 w))

    def __init__(self, ro, ver, nw):
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        self.ro_id, self.ver, self.procs, self.conns = id(ro), ver, [], []
        self._sent = [None] * nw; self._keep = [None] * nw     # the design each worker holds (kept alive: id is a token)
        for _ in range(nw):
            a, b = ctx.Pipe()
            pr = ctx.Process(target=_row_worker, args=(ro, b), daemon=True)
            pr.start(); b.close()
            self.procs.append(pr); self.conns.append(a)

    def alive(self):
        return all(p.is_alive() for p in self.procs)

    def close(self):
        for c in self.conns:
            try:
                c.send(None); c.close()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def gain_batch(self, cl, om, arow, nvec, K, wv, maxit, k):
        ns = len(self.procs) + 1
        slot = (np.round(om * 1000.0).astype(np.int64) // self.BLOCK) % ns
        r, mc, _ = cl._mk
        for j, c in enumerate(self.conns):
            idx = np.where(slot == j + 1)[0]
            # the design's ports (length n) go to a worker ONCE per design: the
            # worker keeps the per-design reduced loop (make_cl) under this token
            first = self._sent[j] != id(cl)
            self._sent[j] = id(cl); self._keep[j] = cl
            c.send((id(cl), cl.b if first else None, cl.c if first else None, r, mc, om[idx], arow, nvec, K, wv[idx], maxit, k))
        out = [None] * len(om)
        own = np.where(slot == 0)[0]
        for i, o in zip(own, cl.gain_batch(om[own], arow, nvec, K, wv[own], maxit=maxit, k=k, par=False)):
            out[i] = o
        for j, c in enumerate(self.conns):
            ok, res, nc = c.recv()
            if not ok:
                raise RuntimeError(f"row worker {j + 1}: {res}")
            cl.ncalls += nc
            for i, o in zip(np.where(slot == j + 1)[0], res):
                out[i] = o
        return out


_POOL = [None]


def _row_pool(cl):
    """The pool for cl's oracle and basis version, forked if needed; None when
    cl was built on an older basis than the oracle's current one (a worker would
    rebuild it on the wrong basis) -- the caller then stays serial."""
    if cl._mk is None or cl._mk[2] != getattr(cl.ro, '_proj_ver', 0):
        return None
    p = _POOL[0]
    if p is not None and p.ro_id == id(cl.ro) and p.ver == cl._mk[2] and p.alive():
        return p
    if p is not None:
        p.close()
    _POOL[0] = _RowPool(cl.ro, cl._mk[2], ROW_WORKERS)
    return _POOL[0]


def _row_worker(ro, conn):
    global ROW_WORKERS
    ROW_WORKERS = 0                      # no pools inside a worker
    held = (None, None)
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        tok, B, C, r, mc, om, arow, nvec, K, wv, maxit, k = msg
        try:
            if B is not None or held[0] != tok:
                held = (tok, ro.make_cl(B, C, r, mc))
            cl = held[1]
            nc0 = cl.ncalls
            out = cl.gain_batch(om, arow, nvec, K, wv, maxit=maxit, k=k, par=False) if len(om) else []
            conn.send((True, out, cl.ncalls - nc0))
        except Exception as e:           # reported to the caller, which raises
            conn.send((False, repr(e), 0))


LANCZOS_MIN_DIM = 40      # below this a dense SVD is cheaper than Lanczos
SUBSPACE = True           # warm-started subspace iteration (else ARPACK) for the leading singular values
TRACK_EIGS = True         # near-line eigenpairs by Rayleigh-quotient tracking, O(k^2) per step
TRACK_FULL_EVERY = 10     # full dense eig every this many calls (new modes)
TRACK_MIN_DIM = 150       # below this the O(k^3) full eig is cheaper than tracking's per-step overhead


class _DenseLU:
    """A k x k dense factorisation with the interface setup() expects of an
    splu object (.solve(x, trans='N'|'H')).  Injected through the frozen
    class's lu_cache so the reduced oracle never calls splu: lu_counter counts
    every splu regardless of size, and the k x k ones are not factorisations
    in the gate's sense (68 of them per evaluation were being counted)."""
    def __init__(self, S):
        from scipy.linalg import lu_factor
        self._lu = lu_factor(np.asarray(S, complex))
    def solve(self, x, trans='N'):
        from scipy.linalg import lu_solve
        return lu_solve(self._lu, x, trans=(2 if trans == 'H' else 0))


class _TriStack:
    """Lazy stack of the QZ-form triangular matrices (i w BB - AA)."""
    def __init__(self, om, AA, BB):
        self.om, self.AA, self.BB = np.asarray(om, float), AA, BB

    def __getitem__(self, i):
        return np.asfortranarray((1j * self.om[i]) * self.BB - self.AA)


class _TriLU:
    """(i w Mt - At) in QZ form: solve = two unitary products + one triangular
    solve, O(k^2).  Same interface as _DenseLU."""
    def __init__(self, P, omega):
        self._T = (1j * float(omega)) * P['qzBB'] - P['qzAA']
        self._Q, self._Z = P['qzQ'], P['qzZ']
        # conjugate transposes computed ONCE per basis (2026-09-23: recomputing
        # them in every solve was ~15 % of an SQP evaluation)
        self._QH, self._ZH = P['qzQH'], P['qzZH']
    def solve(self, x, trans='N'):
        from scipy.linalg import solve_triangular
        x = np.asarray(x, complex)
        if trans == 'H':
            # (Q T Z^H)^H = Z T^H Q^H  ->  y = Q T^{-H} Z^H x
            return self._Q @ solve_triangular(self._T, self._ZH @ x, lower=False, trans='C', check_finite=False)
        return self._Z @ solve_triangular(self._T, self._QH @ x, lower=False, check_finite=False)


class _ReducedCL:
    """Per-design reduced oracle with gain / gain_k in the full oracle's
    conventions, plus a proxy for D.grads."""
    def __init__(self, ro, rcl, b, c):
        self.ro, self.rcl = ro, rcl
        # SNAPSHOT the basis and projection: the oracle may be enriched while
        # this object is alive (a 16-dim projection was read through a 12-dim
        # closed loop once), and a per-design object must stay consistent
        self.V, self.W, self.P, self.n, self.M, self.A = ro.V, ro.W, dict(ro._proj), ro.n, ro.M, ro.A
        self.mimo = getattr(rcl, 'm', None) is not None
        if self.mimo:
            self.b, self.c = np.asarray(b, complex), np.asarray(c, complex)
            self.proxy = _ProxyCLMIMO(self.b, self.c, rcl, self.V, self.W, self.n)
        else:
            self.b, self.c = np.asarray(b, complex).ravel(), np.asarray(c, complex).ravel()
            self.proxy = _ProxyCL(self.b, self.c, rcl, self.V, self.W, self.n)
        self.ncalls = 0
        self._mk = None                     # (r, mc, basis version): set by make_cl
        self._res_ports = None              # b^H A V, b^H M V, b^H b (residuals, once per design)

    def gain_k(self, omega, arow, nvec, K, k=2, want_vectors=True):
        rcl = self.rcl
        P, V, W, M, n = self.P, self.V, self.W, self.M, self.n
        kd = V.shape[1]; r = rcl.r
        # dense LU of the k x k plant block, injected so setup() finds it cached
        key = complex(float(omega))
        if not hasattr(rcl, 'lu_cache') or rcl.lu_cache is None:
            rcl.lu_cache = {}; rcl.lu_cache_max = 4
        if key not in rcl.lu_cache:
            if len(rcl.lu_cache) >= getattr(rcl, 'lu_cache_max', 4):
                rcl.lu_cache.pop(next(iter(rcl.lu_cache)))
            if 'qzAA' in P:
                rcl.lu_cache[key] = _TriLU(P, omega)          # O(k^2) per frequency
            else:
                rcl.lu_cache[key] = _DenseLU((1j * float(omega)) * P['Mt'] - P['At'])
        rcl.setup(float(omega), arow, nvec, K)
        if kd > LANCZOS_MIN_DIM and 'qzAA' in P and SUBSPACE:
            # value-only calls need sigma only: its error goes as the residual
            # squared, so 1e-6 gives ~1e-12; vectors (gradients) need 1e-10
            s, U, Vh = self._leading_svd(omega, k, tol=(1e-10 if want_vectors else 1e-6))
        elif kd > LANCZOS_MIN_DIM and 'qzAA' in P:
            # the LEADING k singular values only (2026-09-23): Lanczos on
            # S = GV^1/2 R~ GW^1/2, each product one reduced closed-loop solve
            # (triangular plant solve inside the Woodbury correction); forming
            # R~ by kd solves and a full SVD cost O(kd^3) per frequency
            from scipy.sparse.linalg import LinearOperator, svds
            GVh, GWh = P['GV_h'], P['GW_h']
            zr = np.zeros(r, complex)
            mv = lambda x: GVh @ rcl.solve(np.concatenate([GWh @ np.asarray(x, complex).ravel(), zr]))[:kd]
            rmv = lambda y: GWh.conj().T @ rcl.solveH(np.concatenate([GVh.conj().T @ np.asarray(y, complex).ravel(), zr]))[:kd]
            Sop = LinearOperator((kd, kd), matvec=mv, rmatvec=rmv, dtype=complex)
            kk = max(1, min(k, kd - 2))
            U, s, Vh = svds(Sop, k=kk, tol=1e-13, which='LM',
                            v0=np.ones(kd, complex) / np.sqrt(kd))
            order = np.argsort(-s); U, s, Vh = U[:, order], s[order], Vh[order]
        else:
            # R~_cl: the plant block of the reduced coupled inverse, by kd solves
            Rt = np.zeros((kd, kd), complex)
            for j in range(kd):
                e = np.zeros(kd + r, complex); e[j] = 1.0
                Rt[:, j] = rcl.solve(e)[:kd]
            S = P['GV_h'] @ Rt @ P['GW_h']
            U, s, Vh = svd(S)
        self.ncalls += 1
        out = []
        for j in range(min(k, len(s))):
            sig = float(s[j])
            if not want_vectors:
                out.append(dict(omega=omega, sigma=sig)); continue
            shat = Vh[j].conj()                        # right singular vector of S
            # optimal load u = M W G_W^{-1/2} shat (unit M^{-1} norm); its
            # reduced right-hand side is W^H u = G_W^{1/2} shat
            rhs = np.concatenate([P['GW_h'] @ shat, np.zeros(r, complex)])
            xt = rcl.solve(rhs)                        # reduced state, ctrl part
            # REDUCED pair (STAGE6 6.29): x = [V x~_p; x_c], z = [W z~_p; z_c] are
            # never formed -- x_p^H M x_p = x~^H G_V x~, V^H M x_p = G_V x~,
            # z_p^H M x_p = z~^H M~ x~ (M~ = W^H M V); the gradients take the
            # reduced vectors (_ProxyCL.grads_reduced)
            qMq = float(np.real(np.vdot(xt[:kd], P['GV'] @ xt[:kd])))
            sc = sig / np.sqrt(qMq) if qMq > 0 else 1.0
            xt = xt * sc
            rhsH = np.concatenate([P['GV'] @ xt[:kd], np.zeros(r, complex)])
            zt = rcl.solveH(rhsH) / sig
            out.append(dict(omega=omega, sigma=sig, x=xt, z=zt, reduced=True,
                            gprime=float(np.imag(np.vdot(zt[:kd], P['Mt'] @ xt[:kd]) +
                                                 np.vdot(zt[kd:], xt[kd:])))))
        return out

    def gain(self, omega, arow, nvec, K, tol=1e-10, want_vectors=True):
        return self.gain_k(omega, arow, nvec, K, k=1, want_vectors=want_vectors)[0]

    # -------- BATCHED rows (2026-09-25, user: evaluate all frequencies at once) --------
    def _Tstack(self, om):
        """(i w BB - AA) for each frequency of the batch, built on ACCESS in
        Fortran order (STAGE6 6.29: the stacked array was nf x kd x kd -- 331 MB
        at nf = 1225, kd = 130, 1 GB for the save curve -- rebuilt at every
        call, and each C-ordered slice was copied again by ztrtrs)."""
        return _TriStack(om, self.P['qzAA'], self.P['qzBB'])

    def _tri(self, om, Y, Ts=None, idx=None):
        """(i w BB - AA) X = Y for every frequency: one LAPACK ztrtrs per
        frequency on the QZ triangular matrix (the system _TriLU solves); a
        vectorised back substitution over the k rows was 2x slower."""
        from scipy.linalg import lapack
        Ts = self._Tstack(om) if Ts is None else Ts
        idx = np.arange(len(om)) if idx is None else idx
        X = np.empty_like(Y); tr = lapack.ztrtrs
        for j, i in enumerate(idx):
            X[j] = tr(Ts[i], Y[j])[0]
        return X

    def _triH(self, om, Y, Ts=None, idx=None):
        from scipy.linalg import lapack
        Ts = self._Tstack(om) if Ts is None else Ts
        idx = np.arange(len(om)) if idx is None else idx
        X = np.empty_like(Y); tr = lapack.ztrtrs
        for j, i in enumerate(idx):
            X[j] = tr(Ts[i], Y[j], trans=2)[0]
        return X

    def _R(self, om, X, Ts=None, idx=None):
        P = self.P
        return np.matmul(P['qzZ'], self._tri(om, np.matmul(P['qzQH'], X), Ts, idx))

    def _RH(self, om, X, Ts=None, idx=None):
        P = self.P
        return np.matmul(P['qzQ'], self._triH(om, np.matmul(P['qzZH'], X), Ts, idx))

    def _batch_setup(self, om, arow, nvec, K):
        """Per-frequency scalars of the rank-2 Woodbury coupling, batched: the
        plant block of the closed loop is T~ = R - g R b c^H R (g from the 2x2
        G of closedloop_sparse.setup), its adjoint R^H - conj(g) R^H c b^H R^H."""
        rcl = self.rcl; m = len(om); kd = self.P['qzAA'].shape[0]
        bt = np.asarray(rcl.b, complex).ravel(); ct = np.asarray(rcl.c, complex).ravel()
        Ts = self._Tstack(om)
        Rb = self._R(om, np.broadcast_to(bt[None, :, None], (m, kd, 1)).copy(), Ts)[:, :, 0]
        RHc = self._RH(om, np.broadcast_to(ct[None, :, None], (m, kd, 1)).copy(), Ts)[:, :, 0]
        cHb = np.einsum('k,mk->m', np.conj(ct), Rb)
        L = rcl.L_of_a(arow); r = L.shape[0]; er = rcl.B_of(); nv = np.asarray(nvec, complex).ravel()
        SK = 1j * om[:, None, None] * np.eye(r)[None] - L[None]
        SKe = np.linalg.solve(SK, np.broadcast_to(er[None, :, None], (m, r, 1)))[:, :, 0]
        SKHn = np.linalg.solve(np.conj(np.transpose(SK, (0, 2, 1))),
                               np.broadcast_to(np.conj(nv)[None, :, None], (m, r, 1)))[:, :, 0]
        nHe = SKe @ nv
        # G = [[1 - K cHb, -nHe], [-cHb, 1]];  G^{-1} [-K, -1]
        g00 = 1 - K * cHb; g01 = -nHe; g10 = -cHb; g11 = np.ones(m, complex)
        det = g00 * g11 - g01 * g10
        v0, v1 = -K, -1.0
        g = (g11 * v0 - g01 * v1) / det            # (G^{-1} [-K, -1])[0]
        g2 = (-g10 * v0 + g00 * v1) / det          # (G^{-1} [-K, -1])[1]
        Gi00 = np.conj(g11 / det)                  # (G^{-H})[0, 0]
        return dict(om=om, bt=bt, ct=ct, Rb=Rb, RHc=RHc, g=g, g2=g2, Gi00=Gi00, SKe=SKe, SKHn=SKHn, Ts=Ts)

    def _Tt(self, bs, X, idx):
        RX = self._R(bs['om'], X, bs['Ts'], idx)
        cRX = np.einsum('k,mkp->mp', np.conj(bs['ct']), RX)
        return RX - bs['Rb'][idx][:, :, None] * (bs['g'][idx][:, None] * cRX)[:, None, :]

    def _TtH(self, bs, X, idx):
        RX = self._RH(bs['om'], X, bs['Ts'], idx)
        bRX = np.einsum('k,mkp->mp', np.conj(bs['bt']), RX)
        return RX - bs['RHc'][idx][:, :, None] * (np.conj(bs['g'][idx])[:, None] * bRX)[:, None, :]

    def _batch_setup_mimo(self, om, arow, nvec, K):
        """MIMO (STAGE6 6.16): the plant block of the closed loop is
            T~ = R + R B~ Gam C~^H R,   Gam = K(iw) (I - G K(iw))^-1,  G = C~^H R B~,
        K(iw) = D_c + C_c (iw - A_c)^-1 B_c (m x p) -- Woodbury on
        (iw M - A - B K(iw) C^H)^-1; the compensator parts of the singular pairs
        x_c = (iw - A_c)^-1 B_c C^H x_p,  z_c = (iw - A_c)^-H C_c^H B^H z_p."""
        rcl = self.rcl; nf = len(om); kd = self.P['qzAA'].shape[0]
        bt = np.asarray(rcl.b, complex); ct = np.asarray(rcl.c, complex)      # kd x m, kd x p
        m, p = bt.shape[1], ct.shape[1]
        Ts = self._Tstack(om)
        Rb = self._R(om, np.broadcast_to(bt[None], (nf, kd, m)).copy(), Ts)          # nf x kd x m
        RHc = self._RH(om, np.broadcast_to(ct[None], (nf, kd, p)).copy(), Ts)        # nf x kd x p
        G = np.einsum('kp,fkm->fpm', np.conj(ct), Rb)                                  # nf x p x m
        L = rcl.L_of_a(arow); r = L.shape[0]
        Bc = rcl.Bc_of(arow); Cc = np.asarray(nvec, complex).reshape(m, r); Dc = np.asarray(K, complex).reshape(m, p)
        SK = 1j * om[:, None, None] * np.eye(r)[None] - L[None]
        SKiBc = np.linalg.solve(SK, np.broadcast_to(Bc[None], (nf, r, p)))
        SKHiCcH = np.linalg.solve(np.conj(np.transpose(SK, (0, 2, 1))), np.broadcast_to(Cc.conj().T[None], (nf, r, m)))
        Kw = Dc[None] + np.matmul(Cc[None], SKiBc)                                     # nf x m x p
        Gam = np.matmul(Kw, np.linalg.inv(np.eye(p)[None] - np.matmul(G, Kw)))        # nf x m x p
        return dict(om=om, bt=bt, ct=ct, Rb=Rb, RHc=RHc, Gam=Gam, SKiBc=SKiBc, SKHiCcH=SKHiCcH, Ts=Ts)

    def _Tt_mimo(self, bs, X, idx):
        RX = self._R(bs['om'], X, bs['Ts'], idx)
        cRX = np.einsum('kp,fkq->fpq', np.conj(bs['ct']), RX)
        return RX + np.matmul(bs['Rb'][idx], np.matmul(bs['Gam'][idx], cRX))

    def _TtH_mimo(self, bs, X, idx):
        RX = self._RH(bs['om'], X, bs['Ts'], idx)
        bRX = np.einsum('km,fkq->fmq', np.conj(bs['bt']), RX)
        return RX + np.matmul(bs['RHc'][idx], np.matmul(np.conj(np.transpose(bs['Gam'][idx], (0, 2, 1))), bRX))

    def gain_batch(self, omegas, arow, nvec, K, want_vectors, maxit=40, k=1, par=True):
        """sigma_1 at every frequency of `omegas` in one batched subspace
        iteration (warm-started from the same per-frequency cache as
        _leading_svd), and the singular pairs (x, z, gprime) in gain()'s
        conventions where want_vectors[i].  Rows that do not converge fall back
        to gain().  Same mathematics as gain_k/_leading_svd; tolerance 1e-10
        (vectors) / 1e-6 (values) on the residual.
        k > 1 (2026-09-29, the sigma_2 blocks): the leading k pairs, block size
        k + 4 as in _leading_svd, every one of the k residuals converged; the
        result per frequency is gain_k()'s list of k dicts."""
        om = np.asarray(omegas, float); m = len(om); wv = np.asarray(want_vectors, bool)
        if m == 0:
            return []
        if par and ROW_WORKERS > 0 and m >= PAR_MIN_ROWS:
            pool = _row_pool(self)
            if pool is not None:
                return pool.gain_batch(self, om, arow, nvec, K, wv, maxit, k)
        P = self.P; GVh, GWh = P['GV_h'], P['GW_h']; kd = GVh.shape[0]
        # MEMORY: the batch holds ~60 complex vectors of length kd per frequency
        # (measured 276 MB for 1001 frequencies at kd = 307); a long batch at a
        # large basis (the save curve, 4001 points at kd ~ 300 with two field
        # input bases) is split into chunks of BATCH_BYTES.  Below it (every GL
        # ne=100 run) nothing changes; a chunk warm-starts from the cache the
        # previous chunks filled (STAGE6 6.31)
        nchunk = int(np.ceil(m * kd * 60 * 16 / BATCH_BYTES))
        if nchunk > 1:
            out = []
            for idx in np.array_split(np.arange(m), nchunk):
                out += self.gain_batch(om[idx], arow, nvec, K, wv[idx], maxit=maxit, k=k, par=False)
            return out
        GVhH, GWhH = np.conj(GVh.T), np.conj(GWh.T)
        if self.mimo:
            bs = self._batch_setup_mimo(om, arow, nvec, K); Tt, TtH = self._Tt_mimo, self._TtH_mimo
        else:
            bs = self._batch_setup(om, arow, nvec, K); Tt, TtH = self._Tt, self._TtH
        p = min(kd, 5) if k == 1 else min(kd, k + 4)
        cache = getattr(self.ro, '_svcache', None)
        if cache is None or getattr(self.ro, '_svcache_dim', None) != kd:
            cache = {}; self.ro._svcache = cache; self.ro._svcache_dim = kd
        if len(cache) > SVCACHE_MAX:              # bounded: the oldest half goes (dicts keep insertion order)
            for key_ in list(cache)[:len(cache) // 2]:
                cache.pop(key_)
        V = np.empty((m, kd, p), complex); rng = np.random.default_rng(12345)
        V0 = np.linalg.qr(rng.normal(size=(kd, p)) + 1j * rng.normal(size=(kd, p)))[0]
        for i, w in enumerate(om):
            key = int(round(float(w) * 1000.0)); Vc = cache.get(key)
            if Vc is None:
                for dk in (1, -1, 2, -2, 5, -5, 10, -10):
                    Vc = cache.get(key + dk)
                    if Vc is not None:
                        break
            V[i] = _fit_subspace(Vc, kd, p, V0)
        tol = np.where(wv, 1e-10, 1e-6)
        sig = np.full((m, k), np.nan); v1 = np.zeros((m, kd, k), complex)
        act = np.arange(m)
        for it in range(maxit):
            Va = V[act]
            B = np.matmul(GVh, Tt(bs, np.matmul(GWh, Va), act))
            Ub, sb, Whb = np.linalg.svd(B, full_matrices=False)
            C = np.matmul(GWhH, TtH(bs, np.matmul(GVhH, B), act))
            WhH = np.conj(np.transpose(Whb, (0, 2, 1)))
            Vr = np.matmul(Va, WhH)
            SHU = np.matmul(C, WhH) / np.maximum(sb, 1e-300)[:, None, :]
            if k == 1:
                res = np.linalg.norm(SHU[:, :, 0] - Vr[:, :, 0] * sb[:, :1], axis=1)
            else:
                res = np.linalg.norm(SHU[:, :, :k] - Vr[:, :, :k] * sb[:, None, :k], axis=1).max(axis=1)
            done = res <= tol[act] * sb[:, 0]
            for jj in np.where(done)[0]:
                i = act[jj]; sig[i] = sb[jj, :k]; v1[i] = Vr[jj, :, :k]
                # a COPY: Vr[jj] is a view that kept the whole batch array
                # (frequencies x kd x p) alive for as long as the entry -- the
                # cache pinned GBs (K+FB+FC at ne = 200, kd ~ 300: killed for
                # memory at save; STAGE6 6.31)
                cache[int(round(float(om[i]) * 1000.0))] = Vr[jj].copy()
            keep = ~done
            if not np.any(keep):
                act = act[:0]; break
            V[act[keep]] = np.linalg.qr(C[keep])[0]
            act = act[keep]
        out = [None] * m
        for i in act:                                   # not converged: the per-row path
            out[i] = (self.gain(float(om[i]), arow, nvec, K, want_vectors=bool(wv[i])) if k == 1 else
                      self.gain_k(float(om[i]), arow, nvec, K, k=k, want_vectors=bool(wv[i])))
        for i in range(m):
            if out[i] is None and not wv[i]:
                out[i] = (dict(omega=float(om[i]), sigma=float(sig[i, 0])) if k == 1 else
                          [dict(omega=float(om[i]), sigma=float(sig[i, j])) for j in range(k)])
        iv = np.array([i for i in range(m) if out[i] is None], int)
        if iv.size:
            self.ncalls += iv.size
            pairs = [self._batch_pairs(bs, Tt, TtH, iv, m, sig[iv, j], v1[iv, :, j]) for j in range(k)]
            for jj, i in enumerate(iv):
                out[i] = pairs[0][jj] if k == 1 else [pairs[j][jj] for j in range(k)]
        return out

    def _batch_pairs(self, bs, Tt, TtH, iv, m, s_, shat):
        """gain()'s singular pairs (x, z, gprime) at the frequencies iv of a
        batch, from the converged singular values s_ and right singular vectors
        shat of S = GV^1/2 T~ GW^1/2."""
        P = self.P; GWh = P['GW_h']; om = bs['om']
        bsv = {k_: (v_[iv] if (isinstance(v_, np.ndarray) and v_.shape[:1] == (m,) and k_ != 'Ts') else v_) for k_, v_ in bs.items()}
        rhs = (GWh @ shat.T).T[:, :, None]
        if self.mimo:
            xtp = Tt(bs, rhs, iv)[:, :, 0]
            xtc = np.matmul(bs['SKiBc'][iv], np.einsum('kp,fk->fp', np.conj(bs['ct']), xtp)[:, :, None])[:, :, 0]
        else:
            RX = self._R(bs['om'], rhs, bs['Ts'], iv); cHq = np.einsum('k,mkp->mp', np.conj(bsv['ct']), RX)[:, 0]
            xtp = RX[:, :, 0] - bsv['Rb'] * (bsv['g'] * cHq)[:, None]
            xtc = -bsv['SKe'] * (bsv['g2'] * cHq)[:, None]
        GV = P['GV']
        qMq = np.real(np.einsum('mk,kl,ml->m', np.conj(xtp), GV, xtp))
        sc = np.where(qMq > 0, s_ / np.sqrt(np.maximum(qMq, 1e-300)), 1.0)
        xtp = xtp * sc[:, None]; xtc = xtc * sc[:, None]
        w = (GV @ xtp.T).T[:, :, None]
        if self.mimo:
            ztp = TtH(bs, w, iv)[:, :, 0] / s_[:, None]
            ztc = np.matmul(bs['SKHiCcH'][iv], np.einsum('km,fk->fm', np.conj(bs['bt']), ztp)[:, :, None])[:, :, 0]
        else:
            RHw = self._RH(bs['om'], w, bs['Ts'], iv); bRH = np.einsum('k,mkp->mp', np.conj(bsv['bt']), RHw)[:, 0]
            ztp = (RHw[:, :, 0] - bsv['RHc'] * (np.conj(bsv['g']) * bRH)[:, None]) / s_[:, None]
            ztc = (bsv['SKHn'] * (bsv['Gi00'] * bRH)[:, None]) / s_[:, None]
        # REDUCED pairs (STAGE6 6.29): no lift to length n; gprime from M~ = W^H M V
        gp = np.imag(np.einsum('fk,kl,fl->f', np.conj(ztp), P['Mt'], xtp) + np.einsum('fk,fk->f', np.conj(ztc), xtc))
        res = []
        for jj, i in enumerate(iv):
            res.append(dict(omega=float(om[i]), sigma=float(s_[jj]), x=np.concatenate([xtp[jj], xtc[jj]]),
                            z=np.concatenate([ztp[jj], ztc[jj]]), reduced=True, gprime=float(gp[jj])))
        return res

    # -------- block closed-loop solves (plant-only right-hand sides) --------
    def _blkS(self, X):
        """Plant block of the reduced coupled inverse applied to the columns
        of X: rank-2 Woodbury of the frozen class, with the plant solve done
        once for all columns (triangular, QZ form)."""
        st = self.rcl._st; kd = X.shape[0]
        Tp = st['lu'].solve(X)
        if self.mimo:
            rc = self.rcl
            Vt = rc._Vmat(np.vstack([Tp, np.zeros((rc.r, Tp.shape[1]), complex)]))
            return Tp - st['Y'][:kd] @ np.linalg.solve(st['G'], Vt)
        if not hasattr(self, '_cconj'):
            self._cconj = np.conj(self.rcl.c); self._bconj = np.conj(self.rcl.b)
        cHq = self._cconj @ Tp                             # c~^H Tp  (row)
        Vb = np.vstack([-st['K'] * cHq, -cHq])             # V t, controller part of t zero
        return Tp - st['Y'][:kd] @ np.linalg.solve(st['G'], Vb)

    def _blkSH(self, X):
        st = self.rcl._st; kd = X.shape[0]
        Tp = st['lu'].solve(X, trans='H')
        if self.mimo:
            rc = self.rcl
            Ub = np.vstack([rc.b.conj().T @ Tp, np.zeros((rc.p, Tp.shape[1]), complex)])
            return Tp - st['Z'][:kd] @ np.linalg.solve(st['G'].conj().T, Ub)
        if not hasattr(self, '_bconj'):
            self._cconj = np.conj(self.rcl.c); self._bconj = np.conj(self.rcl.b)
        Ub = np.vstack([self._bconj @ Tp, np.zeros(X.shape[1], complex)])
        return Tp - st['Z'][:kd] @ np.linalg.solve(st['G'].conj().T, Ub)

    def _leading_svd(self, omega, k, tol=1e-10, maxit=40):
        """Leading k singular triplets of S = GV^1/2 T~ GW^1/2 by subspace
        iteration on S^H S with a Rayleigh-Ritz step, block size k + 4,
        WARM-STARTED from this frequency's last subspace (cached on the oracle:
        the optimiser evaluates the same frequencies at nearby designs).
        Converged when every residual ||S^H u_j - s_j v_j|| <= tol s_1; falls
        back to ARPACK if not."""
        P = self.P; GVh, GWh = P['GV_h'], P['GW_h']; kd = GVh.shape[0]
        if 'GV_hH' not in P:
            P['GV_hH'] = np.ascontiguousarray(GVh.conj().T); P['GW_hH'] = np.ascontiguousarray(GWh.conj().T)
        GVhH, GWhH = P['GV_hH'], P['GW_hH']
        p = min(kd, k + 4)          # k+2 converged slowly where sigma_2 ~ sigma_3
        cache = getattr(self.ro, '_svcache', None)
        if cache is None or getattr(self.ro, '_svcache_dim', None) != kd:
            cache = {}; self.ro._svcache = cache; self.ro._svcache_dim = kd
        # keyed on w rounded to 1e-3, neighbours looked up DIRECTLY (2026-09-24:
        # keying at 1e-8 with a nearest-key scan grew the cache by every peak
        # search's new frequencies -- 2709 keys after 10 evaluations -- and the
        # scan made each evaluation slower than the last)
        key = int(round(float(omega) * 1000.0))
        V = cache.get(key)
        if V is None and cache:
            for dk in (1, -1, 2, -2, 5, -5, 10, -10, 20, -20, 50, -50):
                V = cache.get(key + dk)
                if V is not None:
                    break
        rng = np.random.default_rng(12345)
        V = _fit_subspace(V, kd, p, np.linalg.qr(rng.normal(size=(kd, p)) + 1j * rng.normal(size=(kd, p)))[0])
        for it in range(maxit):
            B = GVh @ self._blkS(GWh @ V)                    # S V
            Ub, sb, Whb = svd(B, full_matrices=False)
            C = GWhH @ self._blkSH(GVhH @ B)                 # S^H S V
            Vr = V @ Whb.conj().T
            SHU = (C @ Whb.conj().T) / np.maximum(sb, 1e-300)
            res = np.linalg.norm(SHU[:, :k] - Vr[:, :k] * sb[:k], axis=0)
            if np.all(res <= tol * sb[0]):
                cache[key] = Vr
                self.ro._sv_iters = getattr(self.ro, '_sv_iters', 0) + it + 1
                return sb[:k], Ub[:, :k], Vr[:, :k].conj().T
            V = np.linalg.qr(C)[0]
        from scipy.sparse.linalg import LinearOperator, svds       # fallback
        zr = np.zeros(self.rcl.r, complex)
        mv = lambda x: GVh @ self._blkS((GWh @ np.asarray(x, complex).reshape(-1, 1)))[:, 0]
        rmv = lambda y: GWh.conj().T @ self._blkSH((GVh.conj().T @ np.asarray(y, complex).reshape(-1, 1)))[:, 0]
        try:
            U, s_, Vh = svds(LinearOperator((kd, kd), matvec=mv, rmatvec=rmv, dtype=complex), k=max(1, min(k, kd - 2)), tol=1e-13,
                             v0=np.ones(kd, complex) / np.sqrt(kd))
        except Exception:
            # last resort: the dense path (form the reduced operator, full SVD)
            Sfull = GVh @ self._blkS(GWh @ np.eye(kd, dtype=complex))
            U, s_, Vh = svd(Sfull); U, s_, Vh = U[:, :k], s_[:k], Vh[:k]
        o = np.argsort(-s_)
        self.ro._sv_fallback = getattr(self.ro, '_sv_fallback', 0) + 1
        return s_[o], U[:, o], Vh[o]

    def near_line_pairs(self, arow, nvec, K, eps, reach=1.0, res_tol=1e-4, mc=1.0, kmax=8):
        """Every residual-filtered Ritz pair with Re >= -eps - reach, rightmost
        first: the rows note item 2 asks for.  Same eigensolve as abscissa()."""
        from scipy.linalg import eig as deig
        import controller as ctl
        P, V, W, n = self.P, self.V, self.W, self.n
        kd = V.shape[1]; r = len(np.asarray(arow[0]))
        if self.mimo:
            Acl, Mcl, Uc, Vc = self._reduced_pencil_mimo(arow, nvec, K, mc)
            self._cpl = (Uc, Vc)
        else:
            Ac = ctl.A_of(*arow); Bc = np.full(r, mc, complex)
            nv = np.asarray(nvec, complex).ravel()
            bt = np.asarray(self.rcl.b, complex).ravel(); ct = np.asarray(self.rcl.c, complex).ravel()   # W^H b, V^H c
            Acl = np.zeros((kd + r, kd + r), complex); Mcl = np.zeros_like(Acl)
            Acl[:kd, :kd] = P['At'] + complex(K) * np.outer(bt, np.conj(ct))
            Acl[:kd, kd:] = np.outer(bt, nv); Acl[kd:, :kd] = np.outer(Bc, np.conj(ct)); Acl[kd:, kd:] = Ac
            Mcl[:kd, :kd] = P['Mt']; Mcl[kd:, kd:] = np.eye(r)
            # the rank-2 controller coupling, EXACTLY: A_cl - blkdiag(At, Ac) = U V^H
            Uc = np.zeros((kd + r, 2), complex); Uc[:kd, 0] = bt; Uc[kd:, 1] = Bc
            Vc = np.zeros((kd + r, 2), complex); Vc[:kd, 0] = np.conj(complex(K)) * ct; Vc[kd:, 0] = np.conj(nv); Vc[:kd, 1] = ct
            self._cpl = (Uc, Vc)
        self._red_pencil = (Acl, Mcl)          # W^H A_cl V, W^H M_cl V (the closed-loop block, STAGE6 6.29)
        lam, VL, VR = self._near_line_eigs(Acl, Mcl, eps, reach, kmax)
        out = []
        for j in np.argsort(-lam.real):
            if lam[j].real < -eps - reach or len(out) >= kmax:
                break
            # the FULL-model residual of the lifted Ritz pair, from Gram matrices
            # (STAGE6 6.29; was a lift + A_cl, M_cl products at length n per pair)
            res = self._full_residual(VR[:, j], complex(lam[j]), arow, nvec, K, mc)
            if res > res_tol:
                continue
            vr, ul = VR[:, j], VL[:, j]
            eta = np.vdot(ul[:kd], P['Mt'] @ vr[:kd]) + np.vdot(ul[kd:], vr[kd:])      # u^H M_cl v
            out.append(dict(lam=complex(lam[j]), v=vr, u=ul, eta=eta, res=float(res), reduced=True))
        return out

    def _full_residual(self, y, lam, arow, nvec, K, mc):
        """||A_cl v - lam M_cl v|| / (|lam| ||M_cl v|| + ||A_cl v||) for the lifted
        v = [V y_p; y_c], EXACTLY as cl_full.Acl_apply / Mcl_apply would give it,
        from the basis Gram matrices (V^H A^H A V, V^H A^H M V, V^H M^H M V: once
        per basis) and the design's port products (b^H A V, b^H M V, b^H b: once
        per design) -- O(kd^2) per pair instead of O(n kd)."""
        import controller as ctl
        P, kd = self.P, self.V.shape[1]
        yp, yc = y[:kd], y[kd:]
        if self._res_ports is None:
            Bf = np.asarray(self.b, complex).reshape(self.n, -1)
            self._res_ports = ((self.A.conj().T @ Bf).conj().T @ self.V, (self.M @ Bf).conj().T @ self.V,
                               Bf.conj().T @ Bf)
        bA, bM, bb = self._res_ports
        ct = np.asarray(self.rcl.c, complex).reshape(kd, -1)
        cy = ct.conj().T @ yp                                   # c^H x_p
        if self.mimo:
            m, p, r = self.rcl.m, self.rcl.p, self.rcl.r
            wv = np.asarray(K, complex).reshape(m, p) @ cy + np.asarray(nvec, complex).reshape(m, r) @ yc
            bot = self.rcl.L_of_a(arow) @ yc + self.rcl.Bc_of(arow) @ cy
        else:
            wv = complex(K) * cy + np.array([np.dot(np.asarray(nvec, complex).ravel(), yc)])
            bot = ctl.A_of(*arow) @ yc + np.full(len(yc), mc, complex) * cy[0]
        def plant2(l):
            G = P['GAA'] - l * P['GAM'] - np.conj(l) * P['GAM'].conj().T + abs(l) ** 2 * P['GMM']
            return float(np.real(np.vdot(yp, G @ yp) + 2 * np.vdot(wv, (bA - l * bM) @ yp) + np.vdot(wv, bb @ wv)))
        r2 = max(plant2(lam), 0.0) + float(np.linalg.norm(bot - lam * yc) ** 2)
        a2 = max(plant2(0.0), 0.0) + float(np.linalg.norm(bot) ** 2)
        m2 = max(float(np.real(np.vdot(yp, P['GMM'] @ yp))), 0.0) + float(np.linalg.norm(yc) ** 2)
        return float(np.sqrt(r2) / max(abs(lam) * np.sqrt(m2) + np.sqrt(a2), 1e-300))

    def _reduced_pencil_mimo(self, arow, nvec, K, mc):
        """The reduced closed-loop pencil (A_cl, M_cl) and its exact controller
        coupling A_cl - blkdiag(At, A_c) = Uc Vc^H, MIMO (STAGE6):
            Uc = [[B~, 0], [0, B_c]],   Vc = [[C~ D_c^H, C~], [C_c^H, 0]]."""
        import controller as ctl
        P, V, W = self.P, self.V, self.W
        kd = V.shape[1]; r = len(np.asarray(arow[0]))
        rc = self.rcl; m, p = rc.m, rc.p
        Ac = ctl.A_of(*arow); Bc = rc.Bc_of(arow)
        Cc = np.asarray(nvec, complex).reshape(m, r); Dc = np.asarray(K, complex).reshape(m, p)
        Bt = np.asarray(rc.b, complex); Ct = np.asarray(rc.c, complex)     # W^H B, V^H C: formed once, in make_cl
        Acl = np.zeros((kd + r, kd + r), complex); Mcl = np.zeros_like(Acl)
        Acl[:kd, :kd] = P['At'] + Bt @ Dc @ Ct.conj().T
        Acl[:kd, kd:] = Bt @ Cc; Acl[kd:, :kd] = Bc @ Ct.conj().T; Acl[kd:, kd:] = Ac
        Mcl[:kd, :kd] = P['Mt']; Mcl[kd:, kd:] = np.eye(r)
        Uc = np.zeros((kd + r, m + p), complex); Uc[:kd, :m] = Bt; Uc[kd:, m:] = Bc
        Vc = np.zeros((kd + r, m + p), complex); Vc[:kd, :m] = Ct @ Dc.conj().T; Vc[kd:, :m] = Cc.conj().T
        Vc[:kd, m:] = Ct
        return Acl, Mcl, Uc, Vc

    def _near_line_eigs(self, Acl, Mcl, eps, reach, kmax, full_every=1):
        # full_every = 1: the shift-invert path is OFF (2026-09-23).  Its left
        # eigenvectors, matched to the right ones by nearest eigenvalue, can pair
        # the wrong vectors near close eigenvalues -> eta ~ 0 -> exploding
        # abscissa gradients -> NumericalError in the SQP subproblem at K+B+C.
        # Fix to do: left vectors by inverse iteration at each eigenvalue.
        """Eigenvalues of the reduced closed-loop pencil near the stability line,
        with left/right eigenvectors.  2026-09-23: a full dense eig is O(dim^3)
        (0.35 s at dim 240, the largest single cost of an SQP evaluation) and
        only the near-line ones are used: shift-invert Arnoldi with ONE dense LU
        per shift, at the previous evaluation's near-line eigenvalues (cached on
        the oracle, discarded when the basis changes); a full eig every
        `full_every` calls or when nothing is cached catches new modes.  The
        residual filter and the full certificate at the gate stay the safety nets."""
        from scipy.linalg import eig as deig, lu_factor, lu_solve
        from scipy.sparse.linalg import eigs, LinearOperator
        ro = self.ro; nd = Acl.shape[0]
        if getattr(ro, '_nl_dim', None) != nd:
            ro._nl_shifts, ro._nl_dim, ro._nl_count = None, nd, 0
        ro._nl_count = getattr(ro, '_nl_count', 0) + 1
        shifts = ro._nl_shifts
        tracked = getattr(ro, '_nl_track', None)
        # MIMO: no tracking (STAGE6 6.4).  Tracking follows the eigenvalues
        # already near the line and finds newcomers only at the full eig every
        # TRACK_FULL_EVERY calls; a high-gain branch of a two-loop design went
        # from Re = -23 to +38 (Im ~ -164) in ONE accepted SQP step, unseen, and
        # the solve then stalled on an unstable iterate.  The dense eig of the
        # reduced pencil (dim ~ 170) sees every mode of the basis at every call.
        _track = TRACK_EIGS and not getattr(self, 'mimo', False)
        if (_track and tracked and ro._nl_count % TRACK_FULL_EVERY != 0 and nd > TRACK_MIN_DIM
                and getattr(ro, '_nl_track_dim', None) == nd):
            res_ = self._track_eigs(Acl, Mcl, tracked)
            if res_ is not None:
                lam, VL, VR = res_
                keep = lam.real >= -eps - reach
                ro._nl_track = [(complex(lam[j]), VR[:, j], VL[:, j]) for j in range(len(lam)) if keep[j]]
                return lam, VL, VR
        if _track and nd > TRACK_MIN_DIM and ro._nl_count % TRACK_FULL_EVERY != 0 and not tracked:
            ro._nl_count = 0                                   # force a full eig now
        if _track and nd > TRACK_MIN_DIM:
            lam, VL, VR = deig(Acl, Mcl, left=True, right=True)
            fin = np.isfinite(lam); lam, VL, VR = lam[fin], VL[:, fin], VR[:, fin]
            keep = lam.real >= -eps - reach
            ro._nl_track = [(complex(lam[j]), VR[:, j], VL[:, j]) for j in range(len(lam)) if keep[j]]
            ro._nl_track_dim = nd
            return lam, VL, VR
        if shifts is None or len(shifts) == 0 or ro._nl_count % full_every == 0 or nd <= 60:
            lam, VL, VR = deig(Acl, Mcl, left=True, right=True)
            fin = np.isfinite(lam); lam, VL, VR = lam[fin], VL[:, fin], VR[:, fin]
        else:
          try:
              L_, R_, V_ = [], [], []
              for sh in shifts:
                  luf = lu_factor(Acl - sh * Mcl)
                  op = LinearOperator((nd, nd), matvec=lambda x, luf=luf: lu_solve(luf, Mcl @ x), dtype=complex)
                  mu, X = eigs(op, k=min(3, nd - 2), which='LM', tol=1e-12, ncv=min(nd - 1, 20), maxiter=400)
                  lam_s = sh + 1.0 / mu
                  for j, lj in enumerate(lam_s):
                      if any(abs(lj - q) < 1e-6 * max(1.0, abs(lj)) for q in L_):
                          continue
                      # LEFT eigenvector AT THIS EIGENVALUE by inverse iteration on
                      # the adjoint (2026-09-23): matching vectors from a separate
                      # adjoint Arnoldi run by nearest eigenvalue mis-paired them near
                      # close eigenvalues (eta ~ 0, exploding abscissa gradients)
                      xj = X[:, j]
                      luj = lu_factor((Acl - lj * Mcl).conj().T)
                      y = lu_solve(luj, Mcl.conj().T @ xj)
                      y /= np.linalg.norm(y)
                      y = lu_solve(luj, y); y /= np.linalg.norm(y)          # second inverse-iteration step
                      L_.append(lj); R_.append(xj); V_.append(y)
              lam = np.array(L_); VR = np.array(R_).T; VL = np.array(V_).T
          except Exception:
            # ARPACK can fail to converge near clustered eigenvalues (K+B+C
            # ne=200: 3/4 in 1261 iterations, which ended a run): fall back to
            # the full dense eig for this call and drop the cached shifts
            ro._sv_nl_fallback = getattr(ro, '_sv_nl_fallback', 0) + 1
            lam, VL, VR = deig(Acl, Mcl, left=True, right=True)
            fin = np.isfinite(lam); lam, VL, VR = lam[fin], VL[:, fin], VR[:, fin]
        keep = lam.real >= -eps - reach
        near = lam[keep]
        # cache distinct near-line shifts (merged within 0.05) for the next call
        # EVERY distinct near-line eigenvalue is a shift for the next call (a
        # merged, truncated shift list missed 3 of 7 near-line eigenvalues --
        # dropped constraints)
        sh = []
        for q in near[np.argsort(-near.real)]:
            if all(abs(q - p_) > 1e-3 for p_ in sh):
                sh.append(complex(q))
        ro._nl_shifts = sh if sh else None
        return lam, VL, VR

    def _pencil_solve(self, Acl, Mcl, sh, X, adjoint=False):
        """(A_cl - s M_cl)^{-1} X (or its adjoint) in O(k^2): A_cl - s M_cl =
        S0(s) + U V^H with S0 = blkdiag(At - s Mt, Ac - s I) -- the plant block
        TRIANGULAR in the QZ form computed once per basis, for any complex s --
        and a rank-2 controller coupling U V^H read off A_cl (Woodbury).  A
        shift that IS a compensator pole (a static gain leaves A_c's poles in
        the closed loop, decoupled, and the Rayleigh-quotient iteration
        converges onto them exactly) is moved by 1e-10 relative: the solve is
        then a large multiple of the eigenvector, what the iteration needs
        (STAGE6 6.31, KGC+FB at ne = 200)."""
        try:
            return self._pencil_solve0(Acl, Mcl, sh, X, adjoint)
        except np.linalg.LinAlgError:
            return self._pencil_solve0(Acl, Mcl, sh + 1e-10 * max(1.0, abs(sh)) * (1 + 1j), X, adjoint)

    def _pencil_solve0(self, Acl, Mcl, sh, X, adjoint=False):
        P = self.P; kd = P['qzAA'].shape[0]; nd = Acl.shape[0]; r = nd - kd
        from scipy.linalg import solve_triangular
        T = P['qzAA'] - sh * P['qzBB']                               # At - s Mt = Q T Z^H
        Ac_ = Acl[kd:, kd:]; SK = Ac_ - sh * np.eye(r)
        Uc, Vc = self._cpl; m = Uc.shape[1]                          # exact factors (near_line_pairs)
        def S0solve(Y, adj):
            top, bot = Y[:kd], Y[kd:]
            if adj:
                t = P['qzQ'] @ solve_triangular(T, P['qzZH'] @ top, lower=False, trans='C', check_finite=False)
                b = np.linalg.solve(SK.conj().T, bot)
            else:
                t = P['qzZ'] @ solve_triangular(T, P['qzQH'] @ top, lower=False, check_finite=False)
                b = np.linalg.solve(SK, bot)
            return np.vstack([t, b])
        X = np.asarray(X, complex).reshape(nd, -1)
        if not adjoint:
            Y0 = S0solve(X, False); S0U = S0solve(Uc, False)
            Gm = np.eye(m) + Vc.conj().T @ S0U
            return Y0 - S0U @ np.linalg.solve(Gm, Vc.conj().T @ Y0)
        Y0 = S0solve(X, True); S0V = S0solve(Vc, True)
        Gm = np.eye(m) + Uc.conj().T @ S0V
        return Y0 - S0V @ np.linalg.solve(Gm, Uc.conj().T @ Y0)

    def _track_eigs(self, Acl, Mcl, tracked, steps=6, tol=1e-9):
        """Two-sided Rayleigh-quotient iteration for each tracked near-line
        eigenpair, O(k^2) per step (no factorisation): right and left vectors
        belong to the SAME eigenvalue by construction (no matching)."""
        L_, R_, V_ = [], [], []
        for lam0, x, y in tracked:
            lam = complex(lam0); x = np.asarray(x, complex); y = np.asarray(y, complex)
            ok = False; res_prev = None
            for _ in range(steps):
                x = self._pencil_solve(Acl, Mcl, lam, Mcl @ x)[:, 0]; x /= np.linalg.norm(x)
                y = self._pencil_solve(Acl, Mcl, lam, Mcl.conj().T @ y, adjoint=True)[:, 0]; y /= np.linalg.norm(y)
                den = np.vdot(y, Mcl @ x)
                if abs(den) < 1e-14:
                    break
                lam_new = np.vdot(y, Acl @ x) / den
                res = np.linalg.norm(Acl @ x - lam_new * (Mcl @ x)) / max(abs(lam_new) * np.linalg.norm(Mcl @ x) + np.linalg.norm(Acl @ x), 1e-300)
                lam = lam_new
                # converged when below tol AND no longer decreasing: the floor is
                # 1e-14 .. 4e-11 depending on the mode's non-normality (a fixed
                # 1e-12 rejected every tracked set)
                if res < tol and (res_prev is not None and res > 0.3 * res_prev or res < 1e-13):
                    ok = True; break
                res_prev = res
            if not ok:
                return None                                  # fall back to the full eig
            if any(abs(lam - q) < 1e-8 * max(1.0, abs(lam)) for q in L_):
                return None                                  # two tracks merged: re-derive
            L_.append(lam); R_.append(x); V_.append(y)
        return np.array(L_), np.array(V_).T, np.array(R_).T

    def abscissa(self, arow, nvec, K, cl_full, res_tol=1e-6, mc=1.0):
        """Rightmost eigenvalue of the REDUCED closed-loop pencil, with the
        full-model residual filter (Mengi 2022/2023): dense generalised
        eigensolve of size k + r, Ritz pairs lifted, residual
        ||(A_cl - lam M_cl) v|| / (|lam| ||M_cl v|| + ||A_cl v||) by sparse
        matvecs -- no factorisation.  Returns (alpha, lam, v, u, eta, res, ok):
        ok = a Ritz value passed the filter.  The certificate stays outside."""
        from scipy.linalg import eig as deig
        import controller as ctl
        P, V, W, n = self.P, self.V, self.W, self.n
        kd = V.shape[1]; r = len(np.asarray(arow[0]))
        if self.mimo:
            Acl, Mcl, _, _ = self._reduced_pencil_mimo(arow, nvec, K, mc)
        else:
            Ac = ctl.A_of(*arow); Bc = np.full(r, mc, complex)
            nv = np.asarray(nvec, complex).ravel()
            bt = W.conj().T @ self.b; ct = V.conj().T @ self.c
            Acl = np.zeros((kd + r, kd + r), complex); Mcl = np.zeros_like(Acl)
            Acl[:kd, :kd] = P['At'] + complex(K) * np.outer(bt, np.conj(ct))
            Acl[:kd, kd:] = np.outer(bt, nv)
            Acl[kd:, :kd] = np.outer(Bc, np.conj(ct))
            Acl[kd:, kd:] = Ac
            Mcl[:kd, :kd] = P['Mt']; Mcl[kd:, kd:] = np.eye(r)
        lam, VL, VR = deig(Acl, Mcl, left=True, right=True)
        fin = np.isfinite(lam); lam, VL, VR = lam[fin], VL[:, fin], VR[:, fin]
        order = np.argsort(-lam.real)
        best = None
        for j in order[:min(12, len(order))]:
            vr = np.concatenate([V @ VR[:kd, j], VR[kd:, j]])
            Av = cl_full.Acl_apply(vr, arow, nvec, K)
            Mv = cl_full.Mcl_apply(vr)
            res = np.linalg.norm(Av - lam[j] * Mv) / max(abs(lam[j]) * np.linalg.norm(Mv) + np.linalg.norm(Av), 1e-300)
            if res <= res_tol:
                ul = np.concatenate([W @ VL[:kd, j], VL[kd:, j]])
                eta = np.vdot(ul, Mv)
                best = (float(lam[j].real), complex(lam[j]), vr, ul, eta, float(res))
                break
        if best is None:
            return None, None, None, None, None, np.inf, False
        return (*best, True)



def input_basis(D, which, lo, hi, other=0.0, nsnap=121, tol=1e-8):
    """Orthonormal basis of the actuator (which='b') or sensor ('c') vectors
    over positions [lo, hi], by SVD of snapshots from D.bc: the same vectors
    the closed loop uses, so the span is exact for them."""
    S = []
    for x in np.linspace(lo, hi, nsnap):
        b, c, *_ = D.bc(x, other, None, None) if which == 'b' else D.bc(other, x, None, None)
        v = np.asarray(b if which == 'b' else c, complex).ravel()
        S.append(v / np.linalg.norm(v))
    U, sv, _ = np.linalg.svd(np.array(S).T, full_matrices=False)
    k = int(np.searchsorted(-sv / sv[0], -tol))
    return U[:, :max(k, 1)], sv / sv[0]


# ------------------------------------------------ full-model pool tasks (fullpar)
def _enrich_task(t):
    """An anchor's columns in a worker: the design's closed loop from the
    registered mesh object, the input bases from the pool's context."""
    import fullpar
    b, c, arow, nvec, K, w, nvec_dirs = t
    cl = fullpar.CTX['D'].cl(b, c)
    return ReducedOracle.enrich_columns(cl, w, arow, nvec, K, nvec_dirs,
                                        fullpar.CTX.get('Phi_b'), fullpar.CTX.get('Psi_c'))


def _eigvec_task(t):
    import fullpar
    b, c, arow, nvec, K, lam = t
    return fullpar.CTX['D'].cl(b, c).eigvecs_at(lam, arow, nvec, K)
