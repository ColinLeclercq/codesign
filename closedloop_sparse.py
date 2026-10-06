"""
closedloop_sparse.py
====================
Closed-loop energy (L2->L2) resolvent of the descriptor plant + fixed-order
dynamic controller, in the spirit of the note:

  * the PLANT block keeps full sparsity: the only factorisation is the sparse
    LU of  S_p(w) = i w M - A  (reused for both S_p and S_p^H via trans='H');
  * M appears ONLY as a sparse mat-vec -- never M^{-1}, never M^{+-1/2},
    never a Cholesky factor;
  * the controller enters as a RANK-2 correction, handled by the
    Sherman-Morrison-Woodbury identity with a 2x2 capacitance matrix.

------------------------------------------------------------------- the system
plant      M q_dot = A q + b u ,      y = c^H q            (c = M s)
controller x_dot   = L(a) x + m_c e_r y ,   u = K y + n^T x
disturbance enters the plant equation as a load vector:  ... + M f
performance output is the plant energy  ||q||_M .

closed loop:   M_cl xi_dot = A_cl xi + [M f; 0],   M_cl = diag(M, I_r)

    A_cl = [[ A + b K c^H ,  b n^H ],
            [ m_c e_r c^H ,  L(a)  ]]

    S_cl(w) = i w M_cl - A_cl = S_0 + U V ,      S_0 = diag(S_p, S_K)
    S_p = i w M - A (sparse) ,  S_K = i w I_r - L(a)  (r x r dense, tiny)
    U = [[b, 0],[0, m_c e_r]]          (n+r) x 2
    V = [[-K c^H, -n^H],[-c^H, 0]]     2 x (n+r)

Woodbury:  S_cl^{-1} = S_0^{-1} - Y G^{-1} V S_0^{-1},  Y = S_0^{-1}U,
           G = I_2 + V Y                                  (2x2!)
adjoint :  S_cl^{-H} = S_0^{-H} - Z (G^H)^{-1} U^H S_0^{-H},  Z = S_0^{-H}V^H

--------------------------------------------------------------- the gain
With T = P1^T S_cl^{-1} P1  (plant-load-vector -> plant-state) and v = M f,

    sigma^2 = max_v  (v^H T^H M T v)/(v^H M^{-1} v)
            = lambda_max( W_cl ),     W_cl = M T^H M T        (standard EVP)

so again NO M^{-1} is ever applied.  Auxiliary vectors (Benner-Mitchell):

    x = S_cl^{-1} [v; 0]          q = x[:n]      (normalise q^H M q = sigma^2)
    z = S_cl^{-H} [M q; 0]/sigma
    g'(w)        = Im( z^H M_cl x ) = Im( z_p^H M x_p + z_c^H x_c )
    dsigma/dth   = Re( z^H (dA_cl/dth) x )
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sparse_lu

# Krylov subspace of the gain eigensolver (ARPACK ncv).  None = ARPACK's
# default (2k + 1, at least 20): the GL path, bit-identical.  At the
# backward-facing step every operator application is two solves with a
# 1-GB factor: ncv = 8 needs 7-16 applications where the default needs 21
# (sigma equal to 1e-11; BFS.md 5), so the BFS plant sets it.
GAIN_NCV = None

def _safe_solve(Gm, rhs):
    """Solve with the Woodbury capacitance, tolerating a singular one.

    G = I + V S0^{-1} U is singular precisely when the shift coincides with a
    closed-loop eigenvalue.  The warm alpha shift sits at lam + 0.05, so it can
    land on a NEIGHBOURING eigenvalue; np.linalg.solve then raises, the
    exception propagates through the torch autograd wrapper, and the whole
    optimisation aborts with an opaque "check opts.globalAD" -- after eight
    good outer iterations, as observed.  A least-squares solve returns the
    minimum-norm answer instead; the shift is simply a poor one and the caller
    discards the result on its residual check.
    """
    try:
        return np.linalg.solve(Gm, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(Gm, rhs, rcond=None)[0]


def _v0(n, seed=12345):
    """Deterministic start vector for ARPACK.

    `spla.eigs` draws a RANDOM v0 by default, so two calls with identical input
    can converge to different Ritz values.  Measured on one design,
    `alpha_krylov` returned alpha = +1.3529 and alpha = -0.4862 for the SAME
    input -- a stable design reported as violently unstable.  That
    non-determinism is behind the spurious `violates the margin` rejections, the
    alpha = +15.5 design, and the run-to-run scatter in the field cases.
    Pinning v0 does not make the eigensolver correct, but it makes it
    reproducible, which is a precondition for trusting anything downstream.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return v / np.linalg.norm(v)




class ClosedLoopResolvent:
    def __init__(self, M, A, b, c, r, mc=1.0):
        self.M = M.tocsc()
        self.A = A.tocsc()
        self.n = M.shape[0]
        self.b = np.asarray(b, complex).ravel()      # actuator load vector
        self.c = np.asarray(c, complex).ravel()      # sensor vector (c = M s)
        self.r = r
        self.mc = mc
        self.tridiag = True   # set False to use the legacy companion form
        self.nfac = 0

    # ---------------- controller matrices from parameters ----------------
    def L_of_a(self, arow):
        """Controller state matrix.

        Two parametrisations are accepted:
          * TRIDIAGONAL (preferred): arow = (d, e, f) -> diag(d) + super(e) + sub(f)
          * companion  (legacy)   : arow = 1-D coefficient vector
        The tridiagonal family contains the diagonal/modal case (e = f = 0) and
        is what the optimiser now uses; the companion path is kept only so old
        result files remain readable.
        """
        r = self.r
        if isinstance(arow, (tuple, list)) and len(arow) == 3 \
                and np.ndim(arow[0]) == 1:
            d, e, f = (np.asarray(v, complex) for v in arow)
            L = np.zeros((r, r), complex)
            L[np.arange(r), np.arange(r)] = d
            if r > 1:
                L[np.arange(r - 1), np.arange(1, r)] = e
                L[np.arange(1, r), np.arange(r - 1)] = f
            return L
        L = np.zeros((r, r), complex)
        if r > 1:
            L[:-1, 1:] = np.eye(r - 1)
        L[-1, :] = -np.asarray(arow, complex)
        return L

    def B_of(self):
        """Controller input gauge.  ones() for tridiagonal, e_r for companion."""
        r = self.r
        if self.tridiag:
            return np.full(r, self.mc, dtype=complex)
        er = np.zeros(r, complex); er[-1] = self.mc
        return er

    # ---------------- per-frequency setup (ONE sparse LU) ----------------
    def setup(self, omega, arow, nvec, K):
        n, r = self.n, self.r
        cache = getattr(self, 'lu_cache', None)
        pin = getattr(self, 'lu_pin', None)
        key = complex(omega)
        if cache is not None and key in cache:
            lu = cache[key]                         # reused across iterates
            if getattr(self, 'lu_lru', False):
                cache[key] = cache.pop(key)         # least-recently-used order (lattice.py)
        elif pin is not None and key in pin:
            lu = pin[key]                           # pinned anchor frequency (STAGE6 6.29)
        else:
            Sp = (1j * omega) * self.M - self.A
            lu = sparse_lu.factor(Sp)               # the ONLY factorisation (sparse_lu: ordering)
            self.nfac += 1
            # MEMORY.  Each sparse LU is a full factorisation: negligible for
            # a banded 1-D operator (a few MB) but of order 1 GB for a 2-D
            # cavity discretisation with n ~ 1e5, where nested-dissection fill
            # is 10-100x the matrix nnz.  Caching a few tens of frequencies
            # would then need tens of GB, so the cache is bounded (FIFO) and is
            # disabled entirely by FECoDesign.lu_cache_max = 0.
            cmax = getattr(self, 'lu_cache_max', 8)
            if cache is not None and cmax > 0:
                while len(cache) >= cmax:
                    cache.pop(next(iter(cache)))
                cache[key] = lu
        SK = (1j * omega) * np.eye(r) - self.L_of_a(arow)
        nvec = np.asarray(nvec, complex).ravel()

        er = self.B_of()

        # Y = S_0^{-1} U   (2 columns)
        y1 = np.concatenate([lu.solve(self.b), np.zeros(r, complex)])
        y2 = np.concatenate([np.zeros(n, complex), np.linalg.solve(SK, er)])
        Y = np.column_stack([y1, y2])

        # Z = S_0^{-H} V^H  (2 columns)
        # V^H[:,0] = [-conj(K) c ; -nvec] ,  V^H[:,1] = [-c ; 0]
        z1 = np.concatenate([lu.solve(-np.conj(K) * self.c, trans='H'),
                             np.linalg.solve(SK.conj().T, -np.conj(nvec))])
        z2 = np.concatenate([lu.solve(-self.c, trans='H'),
                             np.zeros(r, complex)])
        Z = np.column_stack([z1, z2])

        # G = I2 + V Y   (2x2)
        cHb = np.vdot(self.c, y1[:n])               # c^H (S_p^{-1} b)
        nHe = float(0) + np.dot(nvec, y2[n:])       # n^T (S_K^{-1} m_c e_r)
        G = np.eye(2, dtype=complex)
        G[0, 0] += -K * cHb
        G[0, 1] += -nHe
        G[1, 0] += -cHb
        self._st = dict(lu=lu, SK=SK, Y=Y, Z=Z, G=G, nvec=nvec, K=K, er=er,
                        omega=omega)
        return self._st

    # ---------------- S_0 solves (block diagonal) ----------------
    def _S0(self, w, adjoint=False):
        n, st = self.n, self._st
        if adjoint:
            top = st['lu'].solve(w[:n], trans='H')
            bot = np.linalg.solve(st['SK'].conj().T, w[n:])
        else:
            top = st['lu'].solve(w[:n])
            bot = np.linalg.solve(st['SK'], w[n:])
        return np.concatenate([top, bot])

    def _Vapply(self, w):
        """V w  (2-vector)."""
        n, st = self.n, self._st
        cHq = np.vdot(self.c, w[:n])                # c^H w_p
        return np.array([-st['K'] * cHq - np.dot(st['nvec'], w[n:]),
                         -cHq], complex)

    def _UHapply(self, w):
        """U^H w  (2-vector)."""
        n, st = self.n, self._st
        return np.array([np.vdot(self.b, w[:n]),
                         np.vdot(st['er'], w[n:])], complex)

    # ---------------- Woodbury solves ----------------
    def solve(self, w):
        st = self._st
        t = self._S0(w)
        return t - st['Y'] @ _safe_solve(st['G'], self._Vapply(t))

    def solveH(self, w):
        st = self._st
        t = self._S0(w, adjoint=True)
        return t - st['Z'] @ _safe_solve(st['G'].conj().T,
                                             self._UHapply(t))

    # ---------------- gain at one frequency ----------------
    def gain(self, omega, arow, nvec, K, tol=1e-10, want_vectors=True, v0=None, ncv=None):
        n, r = self.n, self.r
        self.setup(omega, arow, nvec, K)

        def Tv(v):
            return self.solve(np.concatenate([v, np.zeros(r, complex)]))[:n]

        def THv(w):
            return self.solveH(np.concatenate([w, np.zeros(r, complex)]))[:n]

        Wop = spla.LinearOperator(
            (n, n), matvec=lambda v: self.M @ THv(self.M @ Tv(v)),
            dtype=complex)
        # v0 / ncv: a warm start (lattice.py: the last vector at the same
        # frequency, ncv 4 -- 4-5 applications instead of 9-17, sigma within 1e-13)
        lam, V = spla.eigs(Wop, k=1, which='LM', tol=tol, maxiter=5000, ncv=(GAIN_NCV if ncv is None else ncv),
                           v0=(_v0(Wop.shape[0]) if v0 is None else v0))
        lam = float(np.real(lam[0]))
        sigma = float(np.sqrt(max(lam, 0.0)))
        out = dict(omega=omega, sigma=sigma, lam=lam)
        if not want_vectors:
            if v0 is not None or ncv is not None:
                out['v'] = V[:, 0]                  # the direction, for the next warm start
            return out
        v = V[:, 0]
        x = self.solve(np.concatenate([v, np.zeros(r, complex)]))
        q = x[:n]
        qMq = float(np.real(np.vdot(q, self.M @ q)))
        s = sigma / np.sqrt(qMq) if qMq > 0 else 1.0
        v, x, q = v * s, x * s, q * s
        Mq = self.M @ q
        z = self.solveH(np.concatenate([Mq, np.zeros(r, complex)])) / sigma
        # g'(w) = Im(z^H M_cl x)
        gp = float(np.imag(np.vdot(z[:n], self.M @ x[:n]) +
                           np.vdot(z[n:], x[n:])))
        out.update(v=v, x=x, z=z, gprime=gp)
        return out

    def gain_k(self, omega, arow, nvec, K, k=2, tol=1e-10):
        """Top-k singular values of the closed-loop map at omega, each with the
        (x, z) pair `design_grad`/`D.grads` need -- the same construction as
        gain() applied to the k largest eigenpairs of W = M T^H M T.

        Stage 1b, note item 3: sigma_2 <= t is enrolled as its own constraint
        at the active frequencies, so the second pair is needed there.  gain()
        is left untouched (frozen path); this is only called for enrolled rows.
        Returns a list of k dicts sorted by sigma descending.
        """
        n, r = self.n, self.r
        self.setup(omega, arow, nvec, K)

        def Tv(v):
            return self.solve(np.concatenate([v, np.zeros(r, complex)]))[:n]

        def THv(w):
            return self.solveH(np.concatenate([w, np.zeros(r, complex)]))[:n]

        Wop = spla.LinearOperator(
            (n, n), matvec=lambda v: self.M @ THv(self.M @ Tv(v)),
            dtype=complex)
        lam, V = spla.eigs(Wop, k=k, which='LM', tol=tol, maxiter=5000,
                           ncv=(max(GAIN_NCV, 2 * k + 2) if GAIN_NCV else None), v0=_v0(Wop.shape[0]))
        order = np.argsort(np.real(lam))[::-1]
        out = []
        for j in order:
            lj = float(np.real(lam[j]))
            sigma = float(np.sqrt(max(lj, 0.0)))
            v = V[:, j]
            x = self.solve(np.concatenate([v, np.zeros(r, complex)]))
            q = x[:n]
            qMq = float(np.real(np.vdot(q, self.M @ q)))
            sc = sigma / np.sqrt(qMq) if qMq > 0 else 1.0
            v, x, q = v * sc, x * sc, q * sc
            Mq = self.M @ q
            z = self.solveH(np.concatenate([Mq, np.zeros(r, complex)])) / sigma
            gp = float(np.imag(np.vdot(z[:n], self.M @ x[:n]) +
                               np.vdot(z[n:], x[n:])))
            out.append(dict(omega=omega, sigma=sigma, lam=lj, v=v, x=x, z=z,
                            gprime=gp))
        return out

    # ---------------- analytic design gradients ----------------
    def design_grad_tri(self, res):
        """Gradients for the TRIDIAGONAL parametrisation (d, e, f, nv, K).

        Every structural derivative is a single matrix entry, so each gradient
        component is one scalar product -- simpler than the companion case:
            dA_cl/dd_j  = E_{n+j, n+j}
            dA_cl/de_j  = E_{n+j, n+j+1}
            dA_cl/df_j  = E_{n+j+1, n+j}
            dA_cl/dnv_j = b  in column n+j
            dA_cl/dK    = b c^H
        """
        n, r = self.n, self.r
        x, z = res['x'], res['z']
        xc, zc = x[n:], z[n:]
        zp, xp = z[:n], x[:n]
        zb = np.vdot(zp, self.b)
        cx = np.vdot(self.c, xp)
        g_d = np.array([np.conj(zc[j]) * xc[j] for j in range(r)])
        g_e = np.array([np.conj(zc[j]) * xc[j + 1] for j in range(r - 1)])
        g_f = np.array([np.conj(zc[j + 1]) * xc[j] for j in range(r - 1)])
        g_n = np.array([zb * xc[j] for j in range(r)])
        g_K = zb * cx
        return dict(d=g_d, e=g_e, f=g_f, n=g_n, K=g_K)

    def design_grad(self, res):
        """d sigma / d(theta) for theta = (arow, nvec, K). Pure scalar work."""
        n, r = self.n, self.r
        x, z = res['x'], res['z']
        xp, xc = x[:n], x[n:]
        zp, zc = z[:n], z[n:]
        zpb = np.vdot(zp, self.b)                 # z_p^H b
        cHxp = np.vdot(self.c, xp)                # c^H x_p
        g_a = np.array([-np.conj(zc[-1]) * xc[j] for j in range(r)])
        g_n = np.array([zpb * xc[j] for j in range(r)])
        g_K = zpb * cHxp
        return dict(a=g_a, n=g_n, K=g_K)

    # ---------------- safeguarded Newton peak ----------------
    def peak(self, omega0, arow, nvec, K, trust=0.4, fd=1e-5, itmax=40,
             gtol=1e-11):
        e = self.gain(omega0, arow, nvec, K); best = e; w = omega0
        for _ in range(itmax):
            g1 = e['gprime']
            if abs(g1) <= gtol * max(1.0, e['sigma']):
                break
            gp = self.gain(w + fd, arow, nvec, K)['gprime']
            gm = self.gain(w - fd, arow, nvec, K)['gprime']
            g2 = (gp - gm) / (2 * fd)
            st = -g1 / g2 if (g2 < 0 and abs(g2) > 1e-16) else np.sign(g1) * trust
            st = float(np.clip(st, -trust, trust))
            ok = False
            for _ in range(40):
                et = self.gain(w + st, arow, nvec, K)
                if et['sigma'] >= best['sigma']:
                    w += st; e = et; best = et; ok = True; break
                st *= 0.5
            if not ok or abs(st) < 1e-13:
                break
        return best

    def hinf(self, arow, nvec, K, seeds=(-0.646, -0.54, -0.2, 0.0, -1.1)):
        best = None
        for w0 in seeds:
            p = self.peak(w0, arow, nvec, K)
            if best is None or p['sigma'] > best['sigma']:
                best = p
        return best

    # ================================================================
    # MATRIX-FREE spectral abscissa: shift-invert Krylov (Arnoldi /
    # Krylov-Schur).  NO dense algebra.
    #   A_cl - sigma*M_cl = -S_cl(sigma), so
    #   OP = (A_cl - sigma M_cl)^{-1} M_cl = -S_cl(sigma)^{-1} M_cl
    # is applied with exactly the Woodbury machinery above (one sparse LU of
    # S_p(sigma) + a 2x2 capacitance).  lambda = sigma + 1/theta.
    # Left eigenvectors from the adjoint operator (same factors, trans='H').
    # ================================================================
    def Mcl_apply(self, x):
        n = self.n
        return np.concatenate([self.M @ x[:n], x[n:]])

    # Shift-invert converges only to eigenvalues NEAR a shift, so the shift set
    # must cover wherever the rightmost closed-loop mode can sit.  We use
    #   (a) a wide band along the imaginary axis,
    #   (b) the CONTROLLER poles (roots of the companion polynomial) pushed to
    #       the right half-plane - closed-loop modes cluster near them,
    #   (c) any `extra` shifts (e.g. the previous iterate's eigenvalue: during
    #       optimisation the poles move continuously, so this is very effective).
    # Each shift is one more sparse LU (~10 ms).  See `verify=` below.
    BAND = (0.0, -0.65, 0.65, -1.5, 1.5, -3.5, 3.5, -7.0, 7.0)

    def default_shifts(self, arow, extra=None):
        sh = [0.05 + 1j * w for w in self.BAND]
        # Controller poles: closed-loop modes cluster near them, so they must be
        # in the shift set.  For the TRIDIAGONAL parametrisation arow is the
        # triple (d, e, f) of unequal lengths -- np.asarray on it raises, and a
        # bare except used to swallow that, silently disabling these shifts.
        try:
            pol = np.linalg.eigvals(self.L_of_a(arow))
            sh += [0.05 + 1j * float(np.imag(p)) for p in pol
                   if np.isfinite(p) and abs(np.imag(p)) < 1e4]
        except Exception as exc:
            import warnings
            warnings.warn(f"controller-pole shifts unavailable: {exc}")
        if extra is not None:
            sh += [0.05 + 1j * float(np.imag(e)) for e in np.atleast_1d(extra)]
        ded = []
        for z in sh:
            if not any(abs(z - q) < 1e-6 for q in ded):
                ded.append(z)
        return tuple(ded)

    def alpha_krylov(self, arow, nvec, K, shifts=None, k=6, tol=1e-10,
                     want_vectors=True, extra_shifts=None):
        if shifts is None:
            shifts = self.default_shifts(arow, extra_shifts)
        n, r = self.n, self.r
        N = n + r
        best = None
        for sig in shifts:
            sig = complex(sig)
            self.setup(-1j * sig, arow, nvec, K)   # i*omega = sigma
            OP = spla.LinearOperator(
                (N, N), matvec=lambda x: -self.solve(self.Mcl_apply(x)),
                dtype=complex)
            th = V = None
            try:
                th, V = spla.eigs(OP, k=min(k, N - 2), which='LM',
                                  v0=_v0(N),
                                  tol=max(tol, 1e-7), maxiter=150,
                                  ncv=min(N - 1, max(2 * k + 2, 24)))
            except spla.ArpackNoConvergence as e:
                # keep whatever Ritz pairs DID converge -- usually enough
                if e.eigenvalues is not None and len(e.eigenvalues):
                    th, V = e.eigenvalues, e.eigenvectors
            except Exception:
                pass
            if th is None or len(th) == 0:
                continue
            # RESIDUAL VERIFICATION.  The comment that used to sit here claimed
            # only well-converged Ritz values were accepted; nothing actually
            # checked.  A near-zero Ritz value theta maps to
            # lambda = sigma + 1/theta of enormous modulus, and taking the max
            # real part over unverified candidates then returns garbage: on the
            # stored designs this produced alpha = 1062, 4102, 545, 291 for
            # closed loops whose true alpha is about -0.22.  It failed ONLY on
            # cases with a dynamic controller, where the pencil has the extra
            # r states and ARPACK has more room to return junk.
            keep = np.abs(th) > 1e-12
            if not np.any(keep):
                continue
            th_k, V_k = th[keep], V[:, keep]
            lam = sig + 1.0 / th_k
            # Two tiers.  Prefer residual-VERIFIED pairs; but never discard
            # everything, because a hard reject raises out through the torch
            # autograd wrapper and pyGRANSO reports it as an opaque
            # "check opts.globalAD".  If nothing passes the tolerance, fall
            # back to the candidate with the SMALLEST relative residual and
            # record that the shift was doubtful.
            fallback = None
            for jj in range(len(lam)):
                vv = V_k[:, jj]
                nv_ = np.linalg.norm(vv)
                if nv_ == 0:
                    continue
                res = np.linalg.norm(self.Acl_apply(vv, arow, nvec, K)
                                     - lam[jj] * self.Mcl_apply(vv)) / nv_
                rel = res / max(1.0, abs(lam[jj]))
                if rel <= 1e-6:
                    if best is None or lam[jj].real > best[0].real:
                        best = (lam[jj], vv, sig, th_k[jj])
                elif fallback is None or rel < fallback[0]:
                    fallback = (rel, lam[jj], vv, th_k[jj])
            if best is None and fallback is not None:
                self.nunverified = getattr(self, 'nunverified', 0) + 1
                _, lm, vv, tj = fallback
                best = (lm, vv, sig, tj)
        if best is None:
            raise RuntimeError("alpha_krylov: no eigenvalue converged")
        lamb, v, sig, thj = best
        if not want_vectors:
            return float(lamb.real), lamb, None, None
        self.setup(-1j * sig, arow, nvec, K)
        OPH = spla.LinearOperator(
            (N, N), matvec=lambda x: -self.Mcl_apply(self.solveH(x)),
            dtype=complex)
        u = None
        try:
            thH, W = spla.eigs(OPH, k=min(k, N - 2), which='LM',
                               v0=_v0(N),
                               tol=max(tol, 1e-7), maxiter=150,
                               ncv=min(N - 1, max(2 * k + 2, 24)))
            jj = int(np.argmin(np.abs(thH - np.conj(thj))))
            u = -self.solveH(W[:, jj])
        except spla.ArpackNoConvergence as e:
            if e.eigenvalues is not None and len(e.eigenvalues):
                thH, W = e.eigenvalues, e.eigenvectors
                jj = int(np.argmin(np.abs(thH - np.conj(thj))))
                u = -self.solveH(W[:, jj])
        except Exception:
            pass
        # Refine (or build) the LEFT eigenvector by inverse iteration at the
        # KNOWN eigenvalue: solve (A_cl - lam M_cl)^H u = M_cl^H u_prev, i.e.
        # u <- -S_cl(lam)^{-H} (M_cl^H u_prev).  Factor S_cl at s=lam once.
        self.setup(-1j * lamb, arow, nvec, K)     # i*omega = lam
        if u is None:
            rng = np.random.default_rng(0)
            u = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        u = u / np.linalg.norm(u)
        for _ in range(6):
            u2 = -self.solveH(self.Mcl_apply(u))   # Mcl real-symmetric here
            nrm = np.linalg.norm(u2)
            if nrm == 0:
                break
            u2 = u2 / nrm
            if np.linalg.norm(u2 - u * np.vdot(u, u2)) < 1e-12:
                u = u2
                break
            u = u2
        return float(lamb.real), lamb, v, u

    def eigvecs_at(self, lam, arow, nvec, K, nit=30, tol=1e-11, seed=0):
        """Right and left eigenvectors of (A_cl, M_cl) at a KNOWN lambda.

        Shift-invert power iteration.  When lambda is already located -- as it
        is, by the certificate, to ~1e-11 -- a Krylov space is unnecessary:
        v <- S(lambda)^{-1} M_cl v converges geometrically, and the adjoint
        iteration on the same factors gives u.

        Measured against `alpha_krylov` with a single targeted shift: 0.72 ms
        against 6.59 ms (9.1x), same residual 2.5e-11, eigenvector alignment
        1.00000000, and gradients agreeing to 1e-12.  ARPACK was spending its
        time on Arnoldi bookkeeping for an eigenvalue nobody needed it to find.

        Returns (v, u, residual); the caller should fall back if the residual
        is not small -- inverse iteration degrades near a clustered eigenvalue.
        """
        N = self.n + self.r
        try:
            self.setup(-1j * complex(lam), arow, nvec, K)   # i*omega = lam
        except np.linalg.LinAlgError:
            # lam IS a compensator pole (a static gain leaves A_c's poles in the
            # closed loop, decoupled): invert at a shift 1e-7 away -- the
            # iteration still converges to lam's eigenvector, the residual is
            # measured at lam (STAGE6 6.31, KGC+B at ne = 200)
            self.setup(-1j * (complex(lam) + 1e-7 * max(1.0, abs(lam)) * (1 + 1j)), arow, nvec, K)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        v /= np.linalg.norm(v)
        it = 0
        for it in range(1, nit + 1):
            w = self.solve(self.Mcl_apply(v))
            nw = np.linalg.norm(w)
            if nw == 0 or not np.isfinite(nw):
                return None, None, np.inf
            w /= nw
            conv = np.linalg.norm(w - v * np.vdot(v, w)) < tol
            v = w
            if conv:
                break
        u = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        u /= np.linalg.norm(u)
        for _ in range(max(it, 3)):
            z = self.solveH(self.Mcl_apply(u))
            nz = np.linalg.norm(z)
            if nz == 0 or not np.isfinite(nz):
                return None, None, np.inf
            u = z / nz
        res = float(np.linalg.norm(self.Acl_apply(v, arow, nvec, K)
                                   - lam * self.Mcl_apply(v))
                    / max(np.linalg.norm(v), 1e-30))
        return v, u, res

    def nearest_eig(self, sigma, arow, nvec, K, k=6, tol=1e-9, maxiter=300):
        """Eigenvalue of (A_cl, M_cl) NEAREST the complex shift sigma, and its
        distance.  This is the primitive for the disk-exhaustion stability
        certificate: shift-invert returns the eigenvalues closest to sigma, so
        the distance to the nearest one is the radius of a provably
        EIGENVALUE-FREE disk centred there.
        """
        N = self.n + self.r
        self.setup(-1j * sigma, arow, nvec, K)      # i*omega = sigma
        OP = spla.LinearOperator(
            (N, N), matvec=lambda x: -self.solve(self.Mcl_apply(x)),
            dtype=complex)
        kk = min(k, N - 2)
        th = V = None
        try:
            th, V = spla.eigs(OP, k=kk, which='LM', tol=tol, maxiter=maxiter,
                              v0=_v0(N),
                              ncv=min(N - 1, max(2 * kk + 2, 24)))
        except spla.ArpackNoConvergence as e:
            if e.eigenvalues is not None and len(e.eigenvalues):
                th, V = e.eigenvalues, e.eigenvectors
        if th is None or len(th) == 0:
            return None, 0.0
        keep = np.abs(th) > 1e-300
        th, V = th[keep], V[:, keep]
        lam = sigma + 1.0 / th
        # VERIFY each candidate by its residual.  Unconverged Ritz values appear
        # as huge |theta|, i.e. eigenvalues spuriously CLOSE to the shift -- at
        # one shift this produced a distance of 3e-3 where the truth was 0.62,
        # which would wrongly shrink the eigenvalue-free disk (or, worse, be
        # read as a witness).  Only residual-verified pairs are accepted.
        ok = []
        for i in range(len(lam)):
            v = V[:, i]
            nv = np.linalg.norm(v)
            if nv == 0:
                continue
            res = np.linalg.norm(self.Acl_apply(v, arow, nvec, K)
                                 - lam[i] * self.Mcl_apply(v)) / nv
            if res < 1e-6 * max(1.0, abs(lam[i])):
                ok.append(i)
        if not ok:
            return None, 0.0
        lam = lam[ok]
        j = int(np.argmin(np.abs(lam - sigma)))
        return complex(lam[j]), float(abs(lam[j] - sigma))

    def Acl_apply(self, x, arow, nvec, K):
        """A_cl x, matrix-free (sparse blocks + the rank-2 coupling)."""
        n = self.n
        xp, xc = x[:n], x[n:]
        nv = np.asarray(nvec, complex).ravel()
        cHx = np.vdot(self.c, xp)
        top = self.A @ xp + self.b * (K * cHx + np.dot(nv, xc))
        bot = self.L_of_a(arow) @ xc + self.B_of() * cHx
        return np.concatenate([top, bot])

    # ---------------- DENSE reference (uses Cholesky) for validation ----
    def dense_reference(self, omega, arow, nvec, K):
        """Naive dense implementation with an explicit Cholesky of M -- ONLY
        for validating the sparse/Woodbury path."""
        from scipy.linalg import cholesky, svdvals
        n, r = self.n, self.r
        Md = np.asarray(self.M.todense())
        Lc = cholesky(Md, lower=True)
        Ad = np.asarray(self.A.todense())
        nvec = np.asarray(nvec, complex).ravel()
        er = self.B_of().reshape(r, 1)      # gauge-consistent (ones/e_r)
        Acl = np.zeros((n + r, n + r), complex)
        Acl[:n, :n] = Ad + K * np.outer(self.b, self.c.conj())
        Acl[:n, n:] = np.outer(self.b, nvec)
        Acl[n:, :n] = er @ self.c.conj().reshape(1, n)
        Acl[n:, n:] = self.L_of_a(arow)
        Mcl = np.zeros((n + r, n + r), complex)
        Mcl[:n, :n] = Md; Mcl[n:, n:] = np.eye(r)
        S = 1j * omega * Mcl - Acl
        Sinv = np.linalg.inv(S)
        # R = M^{1/2} [S^{-1}]_{11} M^{1/2}
        R = Lc.conj().T @ Sinv[:n, :n] @ Lc
        return float(svdvals(R)[0])
