"""
fe_codesign.py — FE co-design object (fully sparse run-time path).

Design-variable groups, selected by a '+'-separated case string:
    K   controller canonical params (arow, nvec, Kfb)   2r+1 complex
    B   actuator centre  xa                             1 real
    C   sensor   centre  xs                             1 real
    FB  actuator FIELD  psi_a, ||psi_a||_L2 = 1         n complex
    FC  sensor   FIELD  psi_s, ||psi_s||_L2 = 1         n complex
e.g.  "K", "B+C", "K+B+C", "FB+FC", "K+FB+FC"

Fields use the retraction psi = v/||v||_M so the unit-L2 constraint holds
EXACTLY at every iterate.  b = M psi_a, c = M psi_s (consistent load vectors
of the FE interpolants), so y = c^H q = <q, psi_s>_M.

ALL eigen/singular-value problems are Krylov on matrix-free LinearOperators:
  * closed-loop gain  sigma_max  -> eigs on W_cl = M T^H M T   (T via Woodbury)
  * spectral abscissa alpha      -> shift-invert eigs on (A_cl - sM_cl)^{-1}M_cl
The only factorisation is the sparse LU of S_p(s) = sM - A.
"""
from __future__ import annotations
import numpy as np
from fe_gl import assemble_gl, SIGMA
from closedloop_sparse import ClosedLoopResolvent


def parse_case(case):
    """Parse a '+'-separated case string into design-group flags.

    Controller structure (choose at most one):
        K    dynamic, proper          : (d,e,f,n,Dc)      4r-1 complex
        KSP  dynamic, STRICTLY proper : (d,e,f,n), Dc=0   4r-2 complex
        KGC  static COMPLEX gain      : Dc only           1 complex
        KG   static REAL gain         : Re(Dc) only       1 real
    Field realness:
        FB / FC    complex field
        FBR / FCR  field constrained REAL (fair comparison with a real Gaussian)
    """
    toks = [t.strip().upper() for t in case.split('+') if t.strip()]
    known = {'K', 'KSP', 'KGC', 'KG', 'B', 'C', 'FB', 'FC', 'FBR', 'FCR'}
    bad = [t for t in toks if t not in known]
    if bad:
        raise ValueError(f"unknown case token(s) {bad}; allowed: {sorted(known)}")
    ctrl = [t for t in toks if t in ('K', 'KSP', 'KGC', 'KG')]
    if len(ctrl) > 1:
        raise ValueError(f"at most one controller structure, got {ctrl}")
    mode = {'K': 'full', 'KSP': 'sp', 'KGC': 'gain', 'KG': 'gainr'}
    f = dict(K=bool(ctrl), ctrl_mode=mode.get(ctrl[0], 'full') if ctrl else 'full',
             B='B' in toks, C='C' in toks,
             FB=('FB' in toks or 'FBR' in toks),
             FC=('FC' in toks or 'FCR' in toks),
             real_fb='FBR' in toks, real_fc='FCR' in toks)
    if f['B'] and f['FB']:
        raise ValueError("case has both B and FB: the actuator field "
                         "supersedes its centre - use one or the other")
    if f['C'] and f['FC']:
        raise ValueError("case has both C and FC: the sensor field "
                         "supersedes its centre - use one or the other")
    if 'FB' in toks and 'FBR' in toks:
        raise ValueError("use FB or FBR, not both")
    if 'FC' in toks and 'FCR' in toks:
        raise ValueError("use FC or FCR, not both")
    return f


class FECoDesign:
    def __init__(self, ne=200, L=56.06, mu0=0.38, r=4, mc=1.0, sob_len=0.0):
        from fe_gl import assemble_gl_K
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        A, M, K, xn, free = assemble_gl_K(ne=ne, L=L, mu0=mu0)
        self.A, self.M, self.xn = A.tocsc(), M.tocsc(), xn
        self.Kstiff = K.tocsc()
        self.n = A.shape[0]; self.r = r; self.mc = mc; self.ne = ne
        # Sobolev (H1) gradient preconditioner  P = (M + sob_len^2 K)^{-1} M .
        # sob_len = 0 recovers the plain L2 gradient.  Factor once.
        self._lu_cache = {}          # omega -> sparse LU of S_p
        # warm-started alpha: tracked rightmost eigenvalue + bookkeeping
        self._warm_lam = None
        self._warm_fp = None
        # certificate resolution.  Near the stability boundary the
        # eigenvalue-free disks are small and the sweep can end
        # INCONCLUSIVE, which rejects perfectly good iterates:
        # KG+FBR+FCR logged 6 inconclusive rejections and 0 located,
        # and stalled above its own parent as a result.  A finer
        # eps_int and a larger budget let it prove what is true.
        self.cert_eps_int = 1e-5
        self.cert_itmax = 800
        self.cert_shrink = 0.9
        # segmented certificate: set cert_nseg > 1 and attach a
        # multiprocessing pool to split the band (3.9x measured)
        self.cert_nseg = 1
        self.cert_pool = None
        # relative design change above which the warm shift is NOT trusted
        self.alpha_warm_maxstep = 0.02
        self.alpha_warm = True       # set False to force the full shift set
        self.alpha_full_every = 20   # full re-check cadence
        self.alpha_stats = dict(warm=0, full=0, since_full=10**9,
                                warm_empty=0, disagree=0,
                                worst_disagree=0.0,
                                last_warm_alpha=None)
        # how many factorisations to keep.  8 is safe on a 1-D mesh;
        # set to 0 at cavity scale, where one factor can be ~1 GB.
        self.lu_cache_max = 8
        self.sob_len = float(sob_len)
        self._Hlu_f = None                   # the Sobolev factor, on first use (_Hlu)

    @property
    def _Hlu(self):
        """LU of M + l^2 K for the smoothing change of variable; None when
        sob_len == 0.  Factorised on FIRST USE (2026-09-30 review: the default
        --sob_len 0.8 factorised it at construction for every case, while only
        the free-field cases use it -- one LU plus its fill held all run and
        inherited by every forked worker)."""
        if self.sob_len <= 0:
            return None
        if self._Hlu_f is None:
            import scipy.sparse.linalg as _spla
            self._Hlu_f = _spla.splu((self.M + (self.sob_len ** 2) * self.Kstiff).tocsc())
        return self._Hlu_f


    def smooth_map(self, xi):
        """psi = P xi with P = (M + l^2 K)^{-1} M : maps a raw variable xi to a
        SMOOTH field.  Used as a change of variable so BFGS optimises in smooth
        space (grid-scale xi -> smooth psi).  Identity if sob_len == 0."""
        if self._Hlu is None:
            return xi
        Mx = self.M @ xi
        return self._Hlu.solve(Mx.real) + 1j * self._Hlu.solve(Mx.imag)

    def smooth_unmap(self, psi):
        """Inverse of smooth_map:  xi = P^{-1} psi = M^{-1}(M + l^2 K) psi.

        Needed because the result files store the SMOOTHED field psi = P xi,
        while the optimiser variable is xi.  Feeding psi back in as xi applies P
        twice and silently degrades every field warm start.
        """
        if self._Hlu is None:
            return psi
        import massops
        Mlu = massops.lu_real(self.M)
        t = (self.M + (self.sob_len ** 2) * self.Kstiff) @ psi
        return Mlu.solve(t.real) + 1j * Mlu.solve(t.imag)

    def smooth_map_adj(self, g):
        """Adjoint P^T = M (M + l^2 K)^{-1} (M,H symmetric) for the chain rule."""
        if self._Hlu is None:
            return g
        t = self._Hlu.solve(g.real) + 1j * self._Hlu.solve(g.imag)
        return self.M @ t

    # ---------- profiles ----------
    def gauss(self, x0):
        return np.exp(-(self.xn - x0) ** 2 / (2 * SIGMA ** 2)).astype(complex)

    def dgauss(self, x0):
        g = self.gauss(x0)
        return ((self.xn - x0) / SIGMA ** 2) * g

    def Mdpsi_dx(self, x0):
        """M @ dpsi_dx(x0), memoised on x0 (2026-09-30): the gradient of every
        row at a design needs the same placement derivative, and recomputing it
        per row was 8 % of a MIMO round.  Exact: a pure function of x0."""
        c = self.__dict__.setdefault('_Mdpsi_cache', {})
        key = float(x0)
        v = c.get(key)
        if v is None:
            if len(c) >= 16:
                c.pop(next(iter(c)))
            v = c[key] = self.M @ self.dpsi_dx(key)
        return v

    def dpsi_dx(self, x0):
        """d/dx0 of the UNIT-L2 profile psi = g(x0)/||g(x0)||_M.

        bc() normalises the placement Gaussian, so the placement gradient needs
        the tangent projection of the raw derivative -- the same retraction as
        for the field cases.  Omitting it makes dgamma/dxa and dgamma/dxs wrong
        (the selftest caught exactly this).
        """
        g = self.gauss(x0)
        dg = self.dgauss(x0)
        nv = self.norm_M(g)
        p = g / nv
        t = float(np.real(np.vdot(p, self.M @ dg)))
        return (dg - t * p) / nv

    def norm_M(self, v):
        return float(np.sqrt(np.real(np.vdot(v, self.M @ v))))

    # ---------- build b, c from the active parameterisation ----------
    def bc(self, xa, xs, va=None, vs=None):
        if np.ndim(xa) >= 1 or np.ndim(xs) >= 1 or isinstance(va, (list, tuple)) or isinstance(vs, (list, tuple)):
            return self.bc_mimo(xa, xs, va, vs)
        return self._bc_siso(xa, xs, va, vs)

    def bc_mimo(self, xa, xs, va=None, vs=None):
        """MIMO (2026-09-29, STAGE6): m actuators, p sensors.  xa (m,), xs (p,)
        centres; va / vs lists of m / p raw fields (entries None -> the Gaussian
        at the centre).  Returns B (n x m), C (n x p), PA (n x m), PS (n x p),
        NA (m,), NS (p,) -- each column the SISO bc() of that port."""
        xa = np.atleast_1d(np.asarray(xa, float)); xs = np.atleast_1d(np.asarray(xs, float))
        va = list(va) if va is not None else [None] * len(xa)
        vs = list(vs) if vs is not None else [None] * len(xs)
        m, p = max(len(xa), len(va)), max(len(xs), len(vs))
        cols = []
        for i in range(max(m, p)):
            cols.append(self._bc_siso(float(xa[i]) if i < len(xa) else 0.0,
                                      float(xs[i]) if i < len(xs) else 0.0,
                                      va[i] if i < len(va) else None,
                                      vs[i] if i < len(vs) else None))
        B = np.column_stack([cols[i][0] for i in range(m)]); PA = np.column_stack([cols[i][2] for i in range(m)])
        C = np.column_stack([cols[j][1] for j in range(p)]); PS = np.column_stack([cols[j][3] for j in range(p)])
        NA = np.array([cols[i][4] for i in range(m)]); NS = np.array([cols[j][5] for j in range(p)])
        return B, C, PA, PS, NA, NS

    def _bc_siso(self, xa, xs, va=None, vs=None):
        # profiles are ALWAYS unit-L2 (||psi||_M = 1), whether they come from a
        # Gaussian centre (placement) or an optimised field.  The controller
        # gauge absorbs the scaling, so gamma is invariant; this guarantees B/C
        # and FB/FC are compared at the same area under the curve.
        raw_a = va if va is not None else self.gauss(xa)
        raw_s = vs if vs is not None else self.gauss(xs)
        # HARD FAILURE on a degenerate field.  A zero or NaN raw vector used to
        # emit a RuntimeWarning and then silently poison the run: psi = NaN
        # makes gamma, every gradient and the certificate NaN, which merely
        # looks like a rejected step, so an entire wall-clock allocation can be
        # burnt on garbage without a single error (observed: two sweep jobs
        # warning "invalid value encountered in divide" from this line).  The
        # only ways a raw field gets here as zero/NaN are inheritance defects
        # (a case handed a field it does not carry) or a NaN design vector --
        # both bugs to surface, never states to iterate from.
        na = self.norm_M(raw_a)
        ns = self.norm_M(raw_s)
        if not (np.isfinite(na) and na > 1e-300):
            raise FloatingPointError(
                f"bc(): actuator field has invalid M-norm {na!r} "
                f"(va {'given' if va is not None else 'from gauss'}, "
                f"finite entries {int(np.isfinite(raw_a).sum())}/{raw_a.size})")
        if not (np.isfinite(ns) and ns > 1e-300):
            raise FloatingPointError(
                f"bc(): sensor field has invalid M-norm {ns!r} "
                f"(vs {'given' if vs is not None else 'from gauss'}, "
                f"finite entries {int(np.isfinite(raw_s).sum())}/{raw_s.size})")
        pa = raw_a / na
        ps = raw_s / ns
        return self.M @ pa, self.M @ ps, pa, ps, na, ns

    def cl(self, b, c):
        if np.ndim(b) == 2 or np.ndim(c) == 2:          # MIMO (STAGE6)
            from closedloop_mimo import ClosedLoopResolventMIMO
            R = ClosedLoopResolventMIMO(self.M, self.A, b, c, r=self.r, mc=self.mc)
            R.lu_cache = self._lu_cache
            R.lu_cache_max = self.lu_cache_max
            return R
        R = ClosedLoopResolvent(self.M, self.A, b, c, r=self.r, mc=self.mc)
        # S_p(omega) = i*omega*M - A is INDEPENDENT of the design: the
        # controller and the actuator/sensor enter only through the low-rank
        # Woodbury correction.  With a fixed frequency set (the epigraph
        # formulation) the factorisation can therefore be computed once per
        # omega and reused for every iterate.  Cheap here (the LU is ~3% of a
        # gain evaluation on a banded 1-D operator) but potentially decisive at
        # cavity scale, where the LU dominates.
        R.lu_cache = self._lu_cache
        R.lu_cache_max = self.lu_cache_max
        return R


    # ---------- H-infinity (global over omega, Krylov gain) ----------
    def hinf(self, arow, nvec, Kfb, b, c, lam=None, warm=None, nsw=9,
             wmin=-4.0, wmax=4.0, tol_active=2e-3):
        cl = self.cl(b, c)
        seeds = list(warm) if warm is not None else []
        if lam is not None:
            seeds += [float(lam[i].imag) for i in np.argsort(-lam.real)[:6]]
        ws = np.linspace(wmin, wmax, nsw)
        gs = [cl.gain(w, arow, nvec, Kfb, want_vectors=False)['sigma'] for w in ws]
        seeds.append(float(ws[int(np.argmax(gs))]))
        ded = []
        for w0 in seeds:
            if not any(abs(w0 - q) < 1e-3 for q in ded):
                ded.append(float(w0))
        peaks = []
        for w0 in ded:
            p = cl.peak(w0, arow, nvec, Kfb)
            if not any(abs(p['omega'] - q['omega']) < 1e-4 for q in peaks):
                peaks.append(p)
        peaks.sort(key=lambda d: -d['sigma'])
        gam = peaks[0]['sigma']
        active = [p['omega'] for p in peaks if p['sigma'] >= gam * (1 - tol_active)]
        res = cl.gain(peaks[0]['omega'], arow, nvec, Kfb)
        mem = list(dict.fromkeys(active + [p['omega'] for p in peaks[:4]]))
        return gam, peaks[0]['omega'], res, cl, mem, len(active)

    def alpha(self, arow, nvec, Kfb, b, c, shifts=None, warm_lam=None,
              warm=None):
        """Spectral abscissa, with an optional WARM-STARTED single shift.

        The full shift set costs 16 factorisations per call and is ~90% of an
        oracle call, but its cost is wildly uneven: three high-frequency shifts
        (Im = 7, 9.6, 19.4) carry 79% of the work, which is also why
        parallelising the loop plateaus at 2.3x.  Those are exactly the shifts a
        warm start makes unnecessary: during optimisation the design moves
        continuously, so the rightmost eigenvalue moves continuously and is
        nowhere near Im = 19.

        A residual check proves the returned lambda IS an eigenvalue but NOT
        that it is the rightmost, so the warm shift cannot be trusted
        indefinitely.  The scheme is therefore self-monitoring: every
        `alpha_full_every` calls the FULL set is used instead, and if it
        disagrees with what the warm shift reported the counter is reset and the
        disagreement recorded in `alpha_stats`.  Use `certify_stability` for the
        rigorous rare check.
        """
        cl = self.cl(b, c)
        use = self.alpha_warm if warm is None else warm
        st = self.alpha_stats
        # STEP GATE.  A warm single shift tracks the rightmost eigenvalue only
        # while the design moves slowly.  Measured on a trajectory of 23 designs
        # with steps from 0.002 to 1.0, the warm value agreed to 7e-11 on the
        # small steps but was 100% WRONG on the two largest -- and a wrong alpha
        # means a wrong stability constraint.  So the warm path is used only when
        # the design has barely moved since the last FULL evaluation.
        fp = np.concatenate([np.asarray(arow[0]).ravel(),
                             np.asarray(arow[1]).ravel(),
                             np.asarray(arow[2]).ravel(),
                             np.asarray(nvec).ravel(),
                             np.atleast_1d(Kfb).ravel(),
                             np.asarray(b).ravel(), np.asarray(c).ravel()])
        moved = np.inf
        if self._warm_fp is not None and self._warm_fp.shape == fp.shape:
            den = max(float(np.linalg.norm(self._warm_fp)), 1e-30)
            moved = float(np.linalg.norm(fp - self._warm_fp)) / den
        if use and self._warm_lam is not None \
                and moved <= self.alpha_warm_maxstep \
                and st['since_full'] < self.alpha_full_every:
            sig = complex(self._warm_lam.real + 0.05, self._warm_lam.imag)
            al, lamb, v, u = cl.alpha_krylov(arow, nvec, Kfb, shifts=[sig], k=8)
            if lamb is not None and np.isfinite(al):
                st['warm'] += 1; st['since_full'] += 1
                st['last_warm_alpha'] = float(al)
                self._warm_lam = lamb
                return al, lamb, v, u, cl
            st['warm_empty'] += 1
        al, lamb, v, u = cl.alpha_krylov(arow, nvec, Kfb, shifts=shifts,
                                         extra_shifts=warm_lam)
        # self-monitoring: did the warm shift agree with the full set?
        prev = st.get('last_warm_alpha')
        if prev is not None and np.isfinite(al):
            if abs(prev - al) > 1e-4 * max(1.0, abs(al)):
                st['disagree'] += 1
                st['worst_disagree'] = max(st['worst_disagree'],
                                           abs(prev - al))
        st['full'] += 1; st['since_full'] = 0; st['last_warm_alpha'] = None
        st['max_moved'] = max(st.get('max_moved', 0.0),
                              0.0 if not np.isfinite(moved) else moved)
        self._warm_lam = lamb
        self._warm_fp = fp                 # reference for the step gate
        return al, lamb, v, u, cl

    def cert_region_mimo(self, Ac, Kfb, b, c, wlo=None, whi=None):
        """The part of the line Re = -eps the MIMO abscissa certificate sweeps
        (STAGE6 6.4).  certify_sparse.stability_region brackets the plant's
        rightmost modes and the compensator poles; with m loops at high gain
        the closed loop also has HIGH-GAIN branches far from both, near the
        eigenvalues of D_c C^H M^-1 B (the static part of the loop at infinite
        frequency; atoms, ne=100: estimate -1.7-144.7i against the true
        -0.05-156.9i, which the band [-30, 30] missed).  The region is widened
        to cover each estimate with a margin of |lambda| / 4 + 2.  One sparse
        solve with the cached LU of M per call; M only as a factorised product."""
        import certify_sparse as _csp
        w0, w1, _ = _csp.stability_region(self.A, self.M, Ac)
        import massops
        G0 = np.asarray(c, complex).conj().T @ massops.lu(self.M).solve(np.asarray(b, complex))
        m_, p_ = G0.shape[1], G0.shape[0]
        hg = np.linalg.eigvals(np.asarray(Kfb, complex).reshape(m_, p_) @ G0)
        for l_ in hg:
            pad = 0.25 * abs(l_) + 2.0
            w0 = min(w0, l_.imag - pad); w1 = max(w1, l_.imag + pad)
        return (w0 if wlo is None else wlo), (w1 if whi is None else whi)

    def alpha_certified(self, arow, nvec, Kfb, b, c, eps=1e-3,
                        wlo=None, whi=None, eigs_seen=None):
        """Spectral abscissa from the 1-D CERTIFICATE, with eigenvectors.

        Replaces `alpha` as the constraint source.  Two steps:

          1. `stability_and_abscissa` sweeps the line Re = -eps with adaptive
             eigenvalue-free disks and returns the rightmost eigenvalue seen.
             Validated against a dense eigensolve on 20 designs: agreement
             1e-12 to 1e-15, in 9-15 shifts.  `alpha_krylov`'s fixed shift set
             gave -0.1783 where the truth was +0.0016 and +1062 where it was
             -0.486, and is no longer consulted.
          2. one shift placed AT the eigenvalue just found recovers the left and
             right eigenvectors for the gradient.  Searching is the unreliable
             part; once the location is known a single shift is reliable.

        Returns (alpha, lam, v, u, cl, clear) -- `clear` False means the line
        was not proven free (inconclusive counts as not clear).
        """
        import certify_sparse as _csp
        import controller as _ctl
        cl = self.cl(b, c)
        Ac = _ctl.A_of(*arow)
        Bc = _ctl.Bc_of(arow, self.r, self.mc)
        if (wlo is None or whi is None) and np.ndim(b) == 2:
            wlo, whi = self.cert_region_mimo(Ac, Kfb, b, c, wlo, whi)
        clear, a_seen, ns, unc, wit = _csp.stability_and_abscissa(
            self.A, self.M, b, c, Ac, Bc, nvec, Kfb, eps=eps,
            wlo=wlo, whi=whi, eps_int=self.cert_eps_int,
            itmax=self.cert_itmax, shrink=self.cert_shrink,
            nseg=self.cert_nseg, pool=self.cert_pool,
            eigs_seen=eigs_seen)
        lam0 = wit if wit is not None else complex(a_seen, 0.0)
        if wit is None and getattr(self, 'alpha_argmax', False):
            # the eigenvalue ATTAINING a_seen (certify_sparse.LAST_ARGMAX), not
            # the real point (a_seen, 0): the latter recovers the eigenvector of
            # whichever mode is nearest the real axis (flag: --alpha-argmax)
            _la = getattr(_csp, 'LAST_ARGMAX', None)
            if _la is not None and abs(_la.real - a_seen) <= 1e-12 * max(1.0, abs(a_seen)):
                lam0 = complex(_la)
        if not np.isfinite(a_seen):
            try:
                al, lamb, v, u = cl.alpha_krylov(arow, nvec, Kfb)
            except Exception:
                return -np.inf, None, None, None, cl, clear
            return al, lamb, v, u, cl, clear
        # Targeted shift, only to recover the EIGENVECTORS for the gradient --
        # the VALUE already came from the certificate.  Try a few offsets: a
        # single shift can fail to converge, and alpha_krylov then RAISES,
        # which propagates through the torch autograd wrapper and kills the run
        # ("check opts.globalAD").  Losing the gradient costs convergence
        # speed; losing the run costs an hour.
        # INVERSE ITERATION first: lambda is already known to ~1e-11, so the
        # eigenvectors cost 3 solves rather than 26 Arnoldi iterations (0.72 ms
        # against 6.59 ms, gradients agreeing to 1e-12).  Fall back to the
        # targeted Krylov shift if the residual is not small -- inverse
        # iteration degrades near a clustered eigenvalue.
        al, lamb, v, u = float(a_seen), None, None, None
        try:
            v_, u_, res_ = cl.eigvecs_at(lam0, arow, nvec, Kfb)
            if v_ is not None and res_ < 1e-6 * max(1.0, abs(lam0)):
                return float(a_seen), lam0, v_, u_, cl, clear
        except Exception:
            pass
        for off in (0.02, 0.05, 0.005, 0.2):
            sig = complex(lam0.real + off, getattr(lam0, 'imag', 0.0))
            try:
                al_, lamb_, v_, u_ = cl.alpha_krylov(arow, nvec, Kfb,
                                                     shifts=[sig], k=4)
            except Exception:
                continue
            if lamb_ is not None and np.isfinite(al_):
                lamb, v, u = lamb_, v_, u_
                if al_ >= a_seen - 1e-8:
                    al = float(al_)
                break
        return al, lamb, v, u, cl, clear


    def alpha_dense(self, arow, nvec, Kfb, b, c, return_spectrum=False):
        """Spectral abscissa by DENSE generalised eigensolve.  CHECK ONLY.

        This is the standalone acceptance test of verify_dense.py made callable
        in-process, so the run can veto its own output.  It must never feed the
        optimiser (value or gradient) and never gate an ITERATE: at cavity
        scale it does not exist, and the algorithm must not depend on it.  Its
        two legitimate call sites are (i) the save gate, where a design is
        about to be written to disk, and (ii) the no-regress guard, where a
        stored file is about to block a new result.  Cost at ne = 200
        (n + r = 406): about a second.
        """
        import controller as _ctl
        from scipy.linalg import eig as _deig
        n, r = self.n, self.r
        Ac = _ctl.A_of(*arow)
        Bc = _ctl.Bc_of(arow, r, self.mc)
        Ad = np.asarray(self.A.todense())
        Md = np.asarray(self.M.todense())
        Acl = np.zeros((n + r, n + r), complex)
        Mcl = np.zeros((n + r, n + r), complex)
        if np.ndim(b) == 2 or np.ndim(c) == 2:          # MIMO (STAGE6)
            B = np.asarray(b, complex).reshape(n, -1); C = np.asarray(c, complex).reshape(n, -1)
            m, p = B.shape[1], C.shape[1]
            Cc = np.asarray(nvec, complex).reshape(m, r); Dc = np.asarray(Kfb, complex).reshape(m, p)
            Acl[:n, :n] = Ad + B @ Dc @ C.conj().T
            Acl[:n, n:] = B @ Cc
            Acl[n:, :n] = np.asarray(Bc).reshape(r, p) @ C.conj().T
        else:
            nv = np.asarray(nvec, complex).ravel()
            Acl[:n, :n] = Ad + complex(Kfb) * np.outer(b, np.conj(c))
            Acl[:n, n:] = np.asarray(b).reshape(-1, 1) @ nv.reshape(1, r)
            Acl[n:, :n] = Bc.reshape(-1, 1) @ np.conj(c).reshape(1, n)
        Acl[n:, n:] = Ac
        Mcl[:n, :n] = Md
        Mcl[n:, n:] = np.eye(r)
        lam = _deig(Acl, Mcl, right=False)
        lam = lam[np.isfinite(lam)]
        j = int(np.argmax(lam.real))
        if return_spectrum:
            return float(lam.real[j]), complex(lam[j]), lam
        return float(lam.real[j]), complex(lam[j])

    # ---------- gradients for every group, from (x, z) or (v, u) ----------
    def grads(self, cl, x, z, arow, nvec, Kfb, xa, xs, pa, ps, na, ns,
              flags, eta=1.0, _full=False):
        if getattr(cl, 'is_reduced', False) and not _full and len(x) == cl.kd + cl._r.r:
            # a REDUCED pair (STAGE6 6.29): no length-n vector (reduced._ProxyCL)
            return cl.grads_reduced(self, x, z, arow, nvec, Kfb, xa, xs, pa, ps, na, ns, flags, eta)
        if getattr(cl, 'm', None) is not None:          # MIMO (STAGE6)
            return self.grads_mimo(cl, x, z, arow, nvec, Kfb, xa, xs, pa, ps, na, ns, flags, eta)
        n = self.n
        xp, xc = x[:n], x[n:]
        zp, zc = z[:n], z[n:]
        b, c = cl.b, cl.c
        nvec = np.asarray(nvec, complex).ravel()
        out = {}
        if flags['K']:
            tri = isinstance(arow, (tuple, list)) and len(arow) == 3
            if tri:
                g = cl.design_grad_tri(dict(x=x, z=z))
                out['d'] = np.asarray(g['d']) / eta
                out['e'] = np.asarray(g['e']) / eta
                out['f'] = np.asarray(g['f']) / eta
                out['n'] = np.asarray(g['n']) / eta
                out['K'] = g['K'] / eta
            else:
                g = cl.design_grad(dict(x=x, z=z))
                out['a'] = np.asarray(g['a']) / eta
                out['n'] = np.asarray(g['n']) / eta
                out['K'] = g['K'] / eta
        Sa = Kfb * np.vdot(c, xp) + np.dot(nvec, xc)      # actuator-side scalar
        # sensor-side scalar.  B_c = mc*ones (tridiagonal gauge) injects the
        # measurement into EVERY controller state, hence the SUM; with the old
        # companion gauge B_c = mc*e_r it was just the last one.
        Sc = np.vdot(zp, b) * Kfb + self.mc * np.sum(np.conj(zc))
        if flags['B']:
            ba = self.Mdpsi_dx(xa)
            out['xa'] = float(np.real(np.vdot(zp, ba) * Sa / eta))
        if flags['C']:
            cs = self.Mdpsi_dx(xs)
            out['xs'] = float(np.real(Sc * np.vdot(cs, xp) / eta))
        if flags['FB']:
            Ga = np.conj(self.M @ zp) * Sa / eta
            out['va'] = self._retract(Ga.real, -Ga.imag, pa, na)
        if flags['FC']:
            Gs = Sc * (self.M @ xp) / eta
            out['vs'] = self._retract(Gs.real, +Gs.imag, ps, ns)
        return out

    def grads_mimo(self, cl, x, z, arow, nvec, Kfb, xa, xs, pa, ps, na, ns, flags, eta=1.0):
        """MIMO gradients (STAGE6): the controller entries (d, e, f, B_c, C_c,
        D_c) from the resolvent, and per port the SISO formulas with the
        actuator-side scalar act_i and the sensor-side scalar sen_j
        (ClosedLoopResolventMIMO.port_sides).  xa / xs / va / vs come back as
        arrays / lists over the ports."""
        n = self.n
        xp = x[:n]; zp = z[:n]
        out = {}
        res = dict(x=x, z=z)
        if flags['K']:
            g = cl.design_grad_tri(res)
            for k_ in ('d', 'e', 'f', 'Bc', 'n', 'K'):
                out[k_] = np.asarray(g[k_]) / eta
        act, sen = cl.port_sides(res, arow, nvec, Kfb)
        if flags['B']:
            out['xa'] = np.array([float(np.real(np.vdot(zp, self.Mdpsi_dx(float(xa[i]))) * act[i] / eta))
                                  for i in range(cl.m)])
        if flags['C']:
            out['xs'] = np.array([float(np.real(sen[j] * np.vdot(self.Mdpsi_dx(float(xs[j])), xp) / eta))
                                  for j in range(cl.p)])
        if flags['FB']:
            Mz = self.M @ zp
            out['va'] = []
            for i in range(cl.m):
                Ga = np.conj(Mz) * act[i] / eta
                out['va'].append(self._retract(Ga.real, -Ga.imag, pa[:, i], na[i]))
        if flags['FC']:
            Mx = self.M @ xp
            out['vs'] = []
            for j in range(cl.p):
                Gs = sen[j] * Mx / eta
                out['vs'].append(self._retract(Gs.real, +Gs.imag, ps[:, j], ns[j]))
        return out

    def _retract(self, gR, gI, psi, nv):
        pR, pI = psi.real, psi.imag
        t = float(np.dot(gR, pR) + np.dot(gI, pI))
        # plain L2 retraction gradient; smoothing is done by the xi change of
        # variable in run_case (BFGS-consistent), not here.
        return ((gR - t * (self.M @ pR)) + 1j * (gI - t * (self.M @ pI))) / nv
