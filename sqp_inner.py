"""sqp_inner.py -- an SQP for the epigraph inner problem (note item 8, prototype).

    min_{z, t}  t   s.t.  sigma_1(w_i; z) <= t  (i in Omega),  alpha(z) + eps <= 0,
                          [alpha_ctrl(z) - ctrl_max + eps <= 0],  |d| <= Delta.

What it does that pyGRANSO's BFGS-SQP cannot:

1. The subproblem is solved EXACTLY (Clarabel, interior point, QP + cones):
   every step certified optimal, multipliers usable for the Lagrangian.
2. At a frequency where sigma_1 and sigma_2 (nearly) coincide the two scalar
   rows sigma_1 <= t, sigma_2 <= t are replaced by the 2x2 BLOCK constraint

       (t + dt) I - diag(sigma_1, sigma_2) - sym( sum_j d_j G_j )  >= 0,

   G_j = U_2^H (dM/dtheta_j) V_2, which is the first-order model of sigma_1 at
   a repeated singular value (STAGE1B.md section 3.1) and carries the CROSS
   TERMS a diagonal Y cannot.  A 2x2 Hermitian PSD constraint is one
   second-order cone, (p + r, p - r, 2 Re q, 2 Im q) in SOC, so the subproblem
   stays a conic QP.  Validated: along a random direction at the K+B+C kink,
   FD d sigma_1 = 4.58484, block lambda_max = 4.58530, grad sigma_1 alone
   4.57156.

Otherwise a plain SQP: l1 merit with backtracking, damped BFGS on the
Lagrangian, box trust region, and the returned iterate chosen exactly as
run_case_epi's --select-cons on does (certificate cleared, closed loop and
compensator inside the margin, lowest sigma_max).  Same return contract as
solve_epigraph: (best, ncalls, H, stats).
"""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp

REDUCED_CERT = 'dense'   # refinement's certificate on the reduced pencil (2026-09-25; driver sets it)
PAIR_MANIFOLD = True    # compensator pair as a double pole on the line: equality rows + retraction (2026-09-25; driver --sqp-pair-manifold)
PAIR_KAPPA = 1e-6       # the double pole is held kappa inside the line (its computed poles split by ~sqrt(eps))
PAIR_ENTER_SEP = 0.1    # enter when the pair is close (|l1 - l2| <= this; 5e-2 until 2026-09-26, see STAGE5 29) ...
PAIR_ENTER_U = 1e-3     # ... and its rightmost pole within this of the line
PAIR_ON_LINE = 1e-6     # or: both poles within this of the line, any separation
ALPHA_WARM = True       # line search starts at 2 x the last accepted alpha (2026-09-25)
PAIR_BALL = True        # on S: 2-norm trust region shrunk to the accepted step (measured: 7.8241 -> 7.7943 in 25 iterations / 71
                        # evaluations at the stalled ne=200 point; box, no shrink: 7.8236 in 113)
PAIR_EXIT_TOL = 1e-2    # release S at an S-stationary point when the pair is inside and |mu_b| > this (duals of the rows sum to 1;
                        # 6.7e-4 released at ne=200 and 200 evaluations in the pole regime gained nothing)
PEN_STEER = 6           # adaptive penalty: at most this many x10 steering increases per QP
PEN_SLACK_TOL = 1e-8    # an elastic slack above this gives up a linearised stability constraint (the compensator line's own tolerance)
DEBUG_LS = False        # print every line-search trial's merit components (diagnostics)
BATCH_ROWS = True       # batched reduced-model rows (reduced.gain_batch, 2026-09-25)
FOLLOW_GTOL = 1e-7      # |dsigma/dw| <= gtol sigma at a followed top: the value error is g1^2/(2|g2|) ~ 1e-15, the
                        # top's position error g1/|g2| ~ 1e-10 (1e-10 cost 2-3 extra Newton steps per member, 2026-09-25)
FOLLOW_SCAN = 21        # value-only window scan points per followed member per evaluation (driver: --follow-scan)
FIELD_KMAX2 = None      # FB/FC shape constraint (v^H K v)/(v^H M v) <= this (squared RMS wavenumber; driver --field-kmax; STAGE5 32)
FIELD_ORDER = 1         # 1: H^1 form v^H K v; 2: H^2 form v^H K M^-1 K v <= kmax^4 v^H M v (driver --field-order; STAGE5 32)
FOLLOW_DEDUP = True     # members sharing a top: all but the nearest restart from their own base (2026-09-26)

try:
    import clarabel
except Exception:              # pragma: no cover
    clarabel = None


# ------------------------------------------------------------ LMI helpers
def _emb(Hc):
    """Real symmetric embedding of a complex Hermitian matrix: PSD iff Hc is."""
    R, I = np.real(Hc), np.imag(Hc)
    return np.block([[R, -I], [I, R]])


def _svec(S):
    """Clarabel PSDTriangle ordering: upper triangle, column-major, off-diagonals
    scaled by sqrt 2 (so svec(X).svec(Y) = <X, Y>_F).  Verified against lambda_max."""
    n = S.shape[0]; out = np.empty(n * (n + 1) // 2)
    k = 0
    for j in range(n):
        for i in range(j + 1):
            out[k] = S[i, j] * (1.0 if i == j else np.sqrt(2.0)); k += 1
    return out


def _herm_basis(r):
    """Real basis of the r x r complex Hermitian matrices (r^2 elements)."""
    B = []
    for i in range(r):
        M = np.zeros((r, r), complex); M[i, i] = 1.0; B.append(M)
    for i in range(r):
        for j in range(i + 1, r):
            M = np.zeros((r, r), complex); M[i, j] = 1.0; M[j, i] = 1.0; B.append(M)
            M = np.zeros((r, r), complex); M[i, j] = 1j; M[j, i] = -1j; B.append(M)
    return B


# ----------------------------------------------------------------- evaluation
def _seig(A, left=False, right=False):
    from scipy.linalg import eig
    return eig(A, left=left, right=right)


class Evaluator:
    """Everything the subproblem needs at one z: sigma_1 on W with gradients,
    the abscissa rows, and the 2x2 blocks at the enrolled frequencies."""

    def __init__(self, D, pk, flags, W, eps, ctrl_max, enroll, kink_block=True, ctrl_lmi=False, mg=0.0,
                 oracle=None, reach=1.0, res_tol=1e-4, ctrl_block=False, follow=None, row_screen=0.0,
                 cl_block=False):
        self.D, self.pk, self.flags, self.W = D, pk, flags, np.asarray(W, float)
        self.eps, self.ctrl_max = eps, ctrl_max
        self.enroll = sorted(set(int(i) for i in (enroll or [])))
        self.kink_block = kink_block
        self.NZ = len(pk.z0())
        self.ncalls = 0
        import controller as ctl
        self.ctl = ctl
        self.ctrl_lmi = ctrl_lmi
        self.mg = mg
        self._E = None
        self.oracle, self.reach, self.res_tol = oracle, reach, res_tol
        self.n_alpha_rows, self.n_fallback = [], 0
        self.ctrl_block = ctrl_block
        self._hscale = None
        self.cl_block = bool(cl_block)       # closed-loop Hermite block (2 rightmost near-line eigenvalues)
        self._hscale_cl = None
        self.n_clblock_fail = 0
        self.refine_log = []                 # every in-solve refinement decision (2026-09-24)
        self.follow = dict(follow or {})     # {member frequency (rounded 6): window radius}
        self.row_screen = float(row_screen)
        self.screen_stats = []
        self.n_dedup = []                    # followed members restarted from their base, per evaluation (FOLLOW_DEDUP)
        self.follow_merged = {}              # member -> the shared top its cold restart returned to
        self.w_star = {}
        self.follow_state = {}               # per followed member: carried curvature (peakfollow, 2026-09-25)
        self.pair_mode = False               # compensator pair on the double-pole manifold (2026-09-25)
        self._E_ctrl = None       # dA_c/dz_j (A_c affine in z), for the exact block gradient
        self.force_full_alpha = False    # set when the reduced abscissa disagrees with the certified one at the start

    def _ctrl_dirs(self):
        """dA_c/dz_j (A_c is affine in z), cached; None for non-compensator coordinates."""
        if self._E_ctrl is None:
            NZ = self.NZ
            A00 = self.ctl.A_of(*self.pk.unpack(np.zeros(NZ))[0])
            Ed = []
            for j in range(NZ):
                ez = np.zeros(NZ); ez[j] = 1.0
                dA = self.ctl.A_of(*self.pk.unpack(ez)[0]) - A00
                Ed.append(dA if np.any(dA) else None)
            self._E_ctrl = Ed
        return self._E_ctrl

    def _pos_rows(self, z):
        """Position constraints of the plant (2-D domains, BFS.md 2): hard
        linear rows c + g.d <= 0 with g in z (through the packer); none on GL."""
        fn = getattr(self.D, 'position_rows', None)
        if fn is None:
            return []
        _, _, _, xa, xs, _, _ = self.pk.unpack(z)
        rows = []
        for c, gd in fn(xa, xs, self.flags):
            g = self.pk.zero_grad(); g.update(gd)
            rows.append((float(c), np.asarray(self.pk.grad(g), float)))
        return rows

    def _field_rows(self, z):
        """SOBOLEV SHAPE CONSTRAINT on the free fields (2026-09-28, STAGE5 32):
        R(v) = (v^H K v) / (v^H M v) <= FIELD_KMAX2, K the FE stiffness matrix --
        the squared RMS wavenumber of the field.  Scale-invariant, as the problem
        is (the compensator gain absorbs the field's scale), so it restricts the
        SHAPE; on the unit L2 sphere it is an H^1 ball, compact in L2, so the
        design problem has a minimiser and its discretisations converge to it
        (the unconstrained problem piles its energy at whatever cutoff the
        discretisation has).  Rows c = R / kmax^2 - 1 <= 0 with their gradient
        in z (chain rule through the packer, any field parametrisation).
        FIELD_ORDER = 2: the H^2 form (v^H K M^-1 K v)/(v^H M v) <= kmax^4 --
        the energy a field can hold above wavenumber k is <= (kmax/k)^4 instead
        of (kmax/k)^2, so the cutoff content vanishes faster under refinement."""
        D, pk, flags = self.D, self.pk, self.flags
        _, _, _, _, _, va, vs = pk.unpack(z)
        rows = []
        mimo = getattr(pk, 'mimo', False)
        items = []
        for key, v in (('va', va), ('vs', vs)):
            if v is None or not flags.get('FB' if key == 'va' else 'FC'):
                continue
            if mimo:                                      # one row per port (STAGE6)
                items += [(key, j, vj) for j, vj in enumerate(v)]
            else:
                items.append((key, None, v))
        for key, j_port, v in items:
            v = np.asarray(v, complex)
            Kv = D.Kstiff @ v; Mv = D.M @ v
            if FIELD_ORDER == 2:                              # S v = K M^-1 K v (M factored once per mesh)
                import massops
                Kv = D.Kstiff @ massops.lu(D.M).solve(np.asarray(Kv, complex))
            kk = FIELD_KMAX2 ** FIELD_ORDER
            vMv = float(np.real(np.vdot(v, Mv))); R = float(np.real(np.vdot(v, Kv))) / vMv
            gv = 2.0 * (Kv - R * Mv) / vMv / kk               # d(R/kmax^2p)/dRe v + i d/dIm v
            r = D.r
            if mimo:
                g = pk.zero_grad(); g[key] = list(g[key]); g[key][j_port] = gv
            else:
                g = dict(d=np.zeros(r, complex), e=np.zeros(max(r - 1, 0), complex), f=np.zeros(max(r - 1, 0), complex),
                         n=np.zeros(r, complex), K=0j, xa=0.0, xs=0.0,
                         va=np.zeros(D.n, complex), vs=np.zeros(D.n, complex))
                g[key] = gv
            rows.append((R / kk - 1.0, np.asarray(pk.grad(g), float)))
        return rows

    def pair_data(self, z, sc_d=None):
        """THE COMPENSATOR PAIR AS A DOUBLE POLE ON THE LINE (2026-09-25, user:
        apply the fix).  The two rightmost compensator poles through their sum s
        and product p (accurate at a defective pair; Schur-based invariant
        subspace, hurwitz.cluster_block), and the three real conditions of the
        manifold M_kappa = {double pole at Re = line - kappa}:
            c_u = Re(s)/2 - line + kappa,   c_a = Re(Delta)/sc_d,   c_b = Im(Delta)/sc_d,
        Delta = s^2/4 - p (the pair is s/2 +- sqrt(Delta)).  All three are smooth
        with non-vanishing gradients at a double pole -- unlike the Hermite
        matrix's small eigenvalue (quadratic in the poles' excursion there) and
        unlike the poles themselves (sqrt of the perturbation).  The feasible set
        near M is {c_u <= 0, c_a <~ 0, ...}: the multipliers of c_u and c_a must
        be >= 0 at a KKT point ON M (both constraints are one-sided), c_b's is
        free.  Returns dict(c, J, exc, sep, lam, sc_d); exc = (max Re of the
        pair) - line, computed from s and p."""
        import hurwitz as _hw
        arow = self.pk.unpack(np.asarray(z, float))[0]
        Acm = self.ctl.A_of(*arow)
        U2, V2, lam2 = _hw.cluster_block(Acm, 2, pick=_hw.pick_pair(Acm))
        B = U2.conj().T @ Acm @ V2
        sm = np.trace(B); pr = np.linalg.det(B)
        line = max(-self.eps, self.ctrl_max)
        Dl = sm * sm / 4.0 - pr
        if sc_d is None:
            sc_d = max(abs(sm) / 2.0, 1e-3)
        c = np.array([sm.real / 2.0 - line + PAIR_KAPPA, Dl.real / sc_d, Dl.imag / sc_d])
        NZ = self.NZ; J = np.zeros((3, NZ))
        adjB = np.array([[B[1, 1], -B[0, 1]], [-B[1, 0], B[0, 0]]])
        for j, dA in enumerate(self._ctrl_dirs()):
            if dA is None:
                continue
            dB = U2.conj().T @ dA @ V2
            ds = np.trace(dB); dp = np.trace(adjB @ dB); dD = 0.5 * sm * ds - dp
            J[:, j] = [ds.real / 2.0, dD.real / sc_d, dD.imag / sc_d]
        rt = np.sqrt(Dl + 0j)
        exc = float(sm.real / 2.0 + abs(rt.real) - line)
        low = float(sm.real / 2.0 - abs(rt.real) - line)
        return dict(c=c, J=J, exc=exc, low=low, sep=float(2 * abs(rt)), lam=lam2, sc_d=sc_d)

    def retract(self, z, maxit=10, tol=1e-15):
        """Gauss-Newton back onto S = {equal real parts (c_b = 0), c_u <= 0,
        c_a <= 0} over the compensator coordinates (minimum-norm corrections;
        only A_c is evaluated, no plant work): c_b = 0 always, and c_u = 0 /
        c_a = 0 as well when the step left them positive (poles past the line /
        a horizontal split).  Returns (z, residual)."""
        z = np.asarray(z, float).copy()
        pd = self.pair_data(z); sc_d = pd['sc_d']
        rows = [2]
        for _pass in range(3):
            for _ in range(maxit):
                r = float(np.max(np.abs(pd['c'][rows])))
                if r <= tol:
                    break
                J = pd['J'][rows]
                try:
                    z = z - J.T @ np.linalg.solve(J @ J.T, pd['c'][rows])
                except np.linalg.LinAlgError:
                    break
                pd = self.pair_data(z, sc_d)
            new = [k for k in (0, 1) if k not in rows and pd['c'][k] > tol]
            if not new:
                break
            rows = sorted(rows + new)
        return z, float(max(np.max(np.abs(pd['c'][rows])), pd['c'][0], pd['c'][1], 0.0))

    def set_W(self, W, enroll_freqs):
        """Replace the working set (frequencies) and remap the enrolled rows."""
        self.W = np.asarray(sorted(set(float(w) for w in W)), float)
        self.enroll = sorted(set(int(np.argmin(np.abs(self.W - f))) for f in enroll_freqs))

    def missed_peaks(self, z, t, ws, tol):
        """A-N on the reduced model: sweep sigma_1 on `ws` at z (cheap on the
        oracle), return the local maxima above t (1 + tol) not yet in W."""
        if self.oracle is None:
            return []
        D, pk = self.D, self.pk
        arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
        b, c, *_ = D.bc(xa, xs, va, vs)
        rc = self.oracle.make_cl(b, c, D.r, D.mc)
        g = lambda w: rc.gain(float(w), arow, nvec, K, want_vectors=False)['sigma']
        # the LEVEL-SET certificate on the reduced matrices first (the same
        # procedure the outer loop certifies with, ~0.6 s at dim 20): exact for
        # every peak the model represents, whatever its width -- the sweep at
        # 0.05 spacing missed a peak at +2.34 that the model held at 1e-15
        try:
            import scipy.sparse as _sp, controller as _ctl, certify_sparse as _csp
            P = rc.P; Ac = _ctl.A_of(*arow); Bc = _ctl.Bc_of(arow, D.r, D.mc)
            _cert = _csp.hinf_dense_levelset if REDUCED_CERT == 'dense' else _csp.hinf_certified_sparse
            gr, wr, _, okr = _cert(
                _sp.csc_matrix(P['At']), _sp.csc_matrix(P['Mt']), rc.rcl.b, rc.rcl.c,   # W^H b, V^H c (make_cl)
                Ac, Bc, nvec, K, g, gam_lb=(max(float(t), 1e-8) if REDUCED_CERT == 'dense' else max(0.5 * t, 1e-8)),
                wlo=float(ws[0]), whi=float(ws[-1]))   # dense: ask the question itself -- anything above t (1 + rtol)?
            _near = [float(q) for q in self.W if abs(wr - q) <= 1e-3] if np.isfinite(wr) else []
            self.refine_log.append(dict(t=float(t), gmax=float(gr), wmax=float(wr), ok=bool(okr),
                                        above=bool(gr > t * (1.0 + tol)), near_member=_near))
            if okr:
                # the certificate PROVES the maximum: either it is a missed peak
                # above the threshold, or there is none -- the sweep below is
                # only for a failed certificate (it cost ~700 s a round at
                # ne=200 when run after every successful certificate)
                if np.isfinite(wr) and gr > t * (1.0 + tol) and not _near:
                    return [(float(wr), float(gr))]
                return []
        except Exception as _e:
            self.refine_log.append(dict(t=float(t), error=f"{type(_e).__name__}: {str(_e)[:60]}"))
        sv = np.array([g(w) for w in ws])
        out = []
        for i in range(1, len(ws) - 1):
            if sv[i] >= sv[i - 1] and sv[i] >= sv[i + 1] and sv[i] > 0.5 * t:
                # GL peaks are 0.02-0.1 wide against a grid of 0.05-0.25: refine
                # each local maximum by golden section on the reduced model
                # (cheap), then judge it against t (1 + tol)
                lo, hi = float(ws[i - 1]), float(ws[i + 1])
                x1, x2 = hi - 0.618 * (hi - lo), lo + 0.618 * (hi - lo); f1, f2 = g(x1), g(x2)
                for _ in range(20):
                    if f1 > f2:
                        hi, x2, f2 = x2, x1, f1; x1 = hi - 0.618 * (hi - lo); f1 = g(x1)
                    else:
                        lo, x1, f1 = x1, x2, f2; x2 = lo + 0.618 * (hi - lo); f2 = g(x2)
                wp, vp = (x1, f1) if f1 > f2 else (x2, f2)
                if vp > t * (1.0 + tol) and all(abs(wp - q) > 1e-3 for q in self.W):
                    out.append((float(wp), float(vp)))
        out.sort(key=lambda p: -p[1])
        return out

    def __call__(self, z):
        D, pk, flags = self.D, self.pk, self.flags
        NZ, W = self.NZ, self.W
        arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
        b, c, pa, ps, na, ns = D.bc(xa, xs, va, vs)
        if self.oracle is not None:
            cl = self.oracle.make_cl(b, c, D.r, D.mc); clg = cl.proxy   # reduced sigma rows
            clg.pk = pk                         # field gradients in the reduced space (reduced._ProxyCL)
            clf = D.cl(b, c)                                            # full matvecs for residuals
        else:
            cl = D.cl(b, c); clg = cl; clf = cl
        G = lambda cl_, x_, z_, **kw: np.asarray(
            pk.grad(D.grads(clg, x_, z_, arow, nvec, K, xa, xs, pa, ps, na, ns, flags, **kw)),
            float)
        m = len(W)
        s = np.zeros(m); Gs = np.zeros((m, NZ))
        blocks = {}

        def _block_set(i, P):
            # the sigma_2 (Clarke 2x2) block of row i from its two leading singular
            # pairs P: sigma_1's row, and the block's entry gradients
            # Gc[a, b] = G(x_b, z_a) + i G(x_b, i z_a).  Both consumers (the
            # subproblem's h11, h22, h12 and grad_L's Hermitian part) read only
            # Re Gc[a, a] and Gc[0, 1] + conj Gc[1, 0], so the diagonal's
            # imaginary part is never computed, and Gc[0, 0] is sigma_1's row:
            # 6 gradient calls per block instead of 9 (STAGE6 6.29, exact)
            s[i] = P[0]['sigma']
            Gc = np.zeros((2, 2, NZ), complex)
            Gc[0, 0] = Gs[i] = G(cl, P[0]['x'], P[0]['z'])
            Gc[1, 1] = G(cl, P[1]['x'], P[1]['z'])
            for a_, b_ in ((0, 1), (1, 0)):
                Gc[a_, b_] = (G(cl, P[b_]['x'], P[a_]['z'])
                              + 1j * G(cl, P[b_]['x'], 1j * P[a_]['z']))
            blocks[i] = dict(sig=np.array([P[0]['sigma'], P[1]['sigma']]), Gc=Gc)

        def _block_at(i, wb):
            _block_set(i, cl.gain_k(wb, arow, nvec, K, k=2))
        _batched = BATCH_ROWS and self.oracle is not None and hasattr(cl, 'gain_batch')
        _pending = []
        _blk_pending = []
        _fb = {}
        if BATCH_ROWS and self.follow and self.oracle is not None and FOLLOW_SCAN <= 2 and hasattr(cl, 'gain_batch'):
            # BATCHED PEAK FOLLOWING (2026-09-25): every followed member's Newton
            # iteration in one batched evaluation (peakfollow.follow_batch)
            # (the sigma_2-enrolled rows too, 2026-09-29: their tops were found one
            # row at a time -- with 1/3 of the rows enrolled on an equalised MIMO
            # optimum, 15 % of a round)
            _fi = [i for i, w in enumerate(W) if self.follow.get(round(float(w), 6))]
            if _fi:
                from peakfollow import follow_batch as _fbatch
                _keys = [round(float(W[i]), 6) for i in _fi]
                _res = _fbatch(cl, [float(W[i]) for i in _fi], arow, nvec, K, [self.follow[k_] for k_ in _keys],
                               [self.w_star.get(k_) for k_ in _keys], [self.follow_state.setdefault(k_, {}) for k_ in _keys],
                               gtol=FOLLOW_GTOL)
                _fb = dict(zip(_fi, _res))
        for i, w in enumerate(W):
            if self.kink_block and i in self.enroll:
                # the block is built at the FOLLOWED top too (2026-09-22): the
                # sigma_2-enrolled rows are the most active peaks, and a fixed
                # block row let the -0.95 peak drift 0.0105 and rise to 8.027
                # unseen (refused round, K+B+C ne=200)
                wb = float(w)
                radb = self.follow.get(round(float(w), 6)) if (self.follow and self.oracle is not None) else None
                if radb and i in _fb:
                    wb = _fb[i][0]
                    self.w_star[round(float(w), 6)] = wb
                elif radb:
                    from peakfollow import follow as _follow
                    wb, _ = _follow(cl, w, arow, nvec, K, radb, hint=self.w_star.get(round(float(w), 6)), nscan=FOLLOW_SCAN, gtol=FOLLOW_GTOL,
                                         state=(self.follow_state.setdefault(round(float(w), 6), {}) if FOLLOW_SCAN <= 2 else None))
                    self.w_star[round(float(w), 6)] = wb
                if _batched:
                    _blk_pending.append((i, wb))              # batched below
                else:
                    _block_at(i, wb)
            else:
                wf = float(w)
                rad = self.follow.get(round(float(w), 6)) if (self.follow and self.oracle is not None) else None
                if rad and i in _fb:
                    wf, r = _fb[i]
                    self.w_star[round(float(w), 6)] = wf
                    s[i] = r['sigma']; Gs[i] = G(cl, r['x'], r['z'])
                elif rad:
                    # PEAK-FOLLOWING row: value and gradient at the relocated top
                    # (safeguarded Newton; its evaluation at the top carries the
                    # vectors, so the row needs no further gain call)
                    from peakfollow import follow as _follow
                    wf, r = _follow(cl, w, arow, nvec, K, rad, hint=self.w_star.get(round(float(w), 6)), nscan=FOLLOW_SCAN, gtol=FOLLOW_GTOL,
                                         state=(self.follow_state.setdefault(round(float(w), 6), {}) if FOLLOW_SCAN <= 2 else None))
                    self.w_star[round(float(w), 6)] = wf
                    s[i] = r['sigma']; Gs[i] = G(cl, r['x'], r['z'])
                elif self.row_screen > 0 and self.oracle is not None:
                    # ROW SCREENING (2026-09-23), pass 1: value only (cheap,
                    # loose tolerance, warm); vectors + gradient in pass 2 only
                    # for rows near the maximum
                    if BATCH_ROWS:
                        _pending.append((i, wf))              # batched below
                    else:
                        s[i] = cl.gain(wf, arow, nvec, K, want_vectors=False)['sigma']
                        _pending.append((i, wf))
                else:
                    r = cl.gain(wf, arow, nvec, K)
                    s[i] = r['sigma']; Gs[i] = G(cl, r['x'], r['z'])
        if _blk_pending:
            # BATCHED sigma_2 blocks (2026-09-29): the two leading pairs at every
            # enrolled top in one subspace iteration (reduced.gain_batch, k = 2)
            _Pb = cl.gain_batch(np.array([wb for _, wb in _blk_pending]), arow, nvec, K,
                                np.ones(len(_blk_pending), bool), k=2)
            for (i, _), P_ in zip(_blk_pending, _Pb):
                _block_set(i, P_)
        if _pending and BATCH_ROWS and hasattr(cl, 'gain_batch'):
            # BATCHED rows (2026-09-25): every pending frequency in one subspace
            # iteration (reduced.gain_batch; agreement with the per-row path 4e-13
            # in sigma, 2e-11 in the singular vectors, 8e-15 in the gradients;
            # 401 rows 0.31 s against 1.44 s).  Pass 1 values, pass 2 vectors for
            # the rows within row_screen of the maximum.
            _wp = np.array([wf for _, wf in _pending])
            r1 = cl.gain_batch(_wp, arow, nvec, K, np.zeros(len(_wp), bool))
            for (i, wf), r in zip(_pending, r1):
                s[i] = r['sigma']
            thr = (1.0 - self.row_screen) * float(np.max(s))
            sel = [(i, wf) for i, wf in _pending if s[i] >= thr]
            if sel:
                r2 = cl.gain_batch(np.array([wf for _, wf in sel]), arow, nvec, K, np.ones(len(sel), bool))
                for (i, wf), r in zip(sel, r2):
                    s[i] = r['sigma']; Gs[i] = G(cl, r['x'], r['z'])
            self.screen_stats.append((len(_pending), len(sel)))
        elif _pending:
            # pass 2: rows within row_screen of the maximum get their gradient;
            # the rest keep their TRUE value with a zero gradient -- an inactive
            # constant row a factor (1 - row_screen) below the level, re-checked
            # at every evaluation (a rising row gets its gradient from then on)
            thr = (1.0 - self.row_screen) * float(np.max(s))
            n_full = 0
            for i, wf in _pending:
                if s[i] >= thr:
                    r = cl.gain(wf, arow, nvec, K)
                    s[i] = r['sigma']; Gs[i] = G(cl, r['x'], r['z']); n_full += 1
            self.screen_stats.append((len(_pending), n_full))
        if FOLLOW_DEDUP and self.follow and self.oracle is not None:
            # ONE TOP, SEVERAL MEMBERS (2026-09-26).  A followed member starts its
            # Newton at its previous top, so members drift and collect on one
            # peak; a neighbouring peak then goes unwatched, and when it overtakes
            # the evaluation UNDER-states smax.  ne=100 K+B+C round 14: members
            # based at -0.5280, -0.5105, -0.5094 all on the top at -0.524950
            # (7.9057032) while the one at -0.509496 stood at 7.9063061 -- an
            # iterate accepted on that value made every later trial "worse" than
            # an impossible reference, and the solve stalled (radius 1e-10) or crept
            # (alpha 1/128).  Where several members share a top, all but the one
            # whose base is nearest restart from their OWN base (cold Newton).
            _ft = []
            for i, w in enumerate(W):
                k_ = round(float(w), 6)
                if self.follow.get(k_) and k_ in self.w_star:
                    _ft.append((float(self.w_star[k_]), i))
            _ft.sort()
            _groups, _g = [], []
            for tp, i in _ft:
                if _g and tp - _g[-1][0] > 1e-4:           # Newton's tops agree to ~1e-6 on a flat peak, not 1e-9
                    _groups.append(_g); _g = []
                _g.append((tp, i))
            if _g:
                _groups.append(_g)
            _redo = []
            for _g in _groups:
                if len(_g) > 1:
                    tp = _g[0][0]
                    keep = min(_g, key=lambda q: abs(float(W[q[1]]) - tp))[1]
                    # not a member whose base is at the top, nor one whose cold restart
                    # already returned to (about) this top: truly merged
                    _redo += [i for _, i in _g if i != keep and abs(float(W[i]) - tp) > 1e-3
                              and abs(self.follow_merged.get(round(float(W[i]), 6), np.inf) - tp) > 5e-3]
            if _redo:
                from peakfollow import follow as _follow, follow_batch as _fbatch
                _keys = [round(float(W[i]), 6) for i in _redo]
                for k_ in _keys:
                    self.follow_state[k_] = {}
                if BATCH_ROWS and FOLLOW_SCAN <= 2 and hasattr(cl, 'gain_batch'):
                    _rr = _fbatch(cl, [float(W[i]) for i in _redo], arow, nvec, K, [self.follow[k_] for k_ in _keys],
                                  [None] * len(_redo), [self.follow_state[k_] for k_ in _keys], gtol=FOLLOW_GTOL)
                else:
                    _rr = [_follow(cl, float(W[i]), arow, nvec, K, self.follow[k_], hint=None, nscan=FOLLOW_SCAN, gtol=FOLLOW_GTOL,
                                   state=(self.follow_state[k_] if FOLLOW_SCAN <= 2 else None)) for i, k_ in zip(_redo, _keys)]
                for i, k_, (wf, r) in zip(_redo, _keys, _rr):
                    self.w_star[k_] = wf
                    if DEBUG_LS:
                        print(f"        dedup: member {k_:+.4f} restarted -> top {wf:+.6f} sigma {r['sigma']:.7f}", flush=True)
                    if any(abs(wf - q[0]) <= 1e-4 for q in _ft):
                        self.follow_merged[k_] = wf        # back on the shared top: merged, not restarted again here
                    if i in blocks:
                        _block_at(i, wf)
                    else:
                        s[i] = r['sigma']; Gs[i] = G(cl, r['x'], r['z'])
            self.n_dedup.append(len(_redo))
        self.ncalls += 1
        # closed-loop abscissa: ONE ROW PER NEAR-LINE EIGENVALUE (note item 2),
        # from the reduced pencil's residual-filtered Ritz pairs when the
        # oracle is on -- no factorisation; the full certificate (rightmost
        # only) when it is off or when no pair passes the filter.
        al_rows = []                     # list of (Re lam_j, g_j)
        clr = True
        pairs = []
        if self.oracle is not None and not self.force_full_alpha:
            pairs = cl.near_line_pairs(arow, nvec, K, self.eps, reach=self.reach,
                                       res_tol=self.res_tol, mc=D.mc)
            for pr in pairs:
                gj = np.asarray(pk.grad(D.grads(clg, pr['v'], pr['u'], arow, nvec, K, xa, xs,
                                                pa, ps, na, ns, flags, eta=pr['eta'])), float)
                al_rows.append((float(pr['lam'].real), gj))
            self.n_alpha_rows.append(len(al_rows))
        # ---- CLOSED-LOOP HERMITE BLOCK (2026-09-24, user: by default).  The two
        # rightmost near-line closed-loop eigenvalues, compressed to a 2x2 block
        # on orthonormal bases of their right/left eigenvector spans,
        # biorthonormalised with M_cl; Hermite's criterion on the block shifted to
        # the admissibility line is smooth where the per-eigenvalue rows are not
        # (coalescence: eta -> 0).  Same construction as the compensator block;
        # the block's entry derivatives through D.grads exactly as the sigma
        # block's.  The two eigenvalues leave the per-eigenvalue rows.
        hermite_cl = None
        al_rows_qp = al_rows
        if self.cl_block and len(pairs) >= 2 and len(al_rows) >= 2:
            try:
                hermite_cl = self._closed_loop_block(pairs[:2], cl, clf, clg, arow, nvec, K,
                                                     xa, xs, pa, ps, na, ns, flags, NZ)
                al_rows_qp = al_rows[2:]
            except Exception as _e:
                hermite_cl = None; self.n_clblock_fail += 1
        if not al_rows:
            al, lamb, vv, uu, cla, clr = D.alpha_certified(arow, nvec, K, b, c, eps=self.eps)
            self.n_fallback += 1
            if vv is None or uu is None:
                zero = np.zeros(D.n + D.r, complex)
                ga = np.asarray(pk.grad(D.grads(cla, zero, zero, arow, nvec, K, xa, xs,
                                                pa, ps, na, ns, flags)), float)
            else:
                eta = np.vdot(uu, cla.Mcl_apply(vv))
                ga = np.asarray(pk.grad(D.grads(cla, vv, uu, arow, nvec, K, xa, xs,
                                                pa, ps, na, ns, flags, eta=eta)), float)
            al_rows.append((float(al), ga))
        al, ga = al_rows[0]
        # Controller abscissa: ONE ROW PER POLE (item 2 in miniature).  At the
        # frozen K+B+C two poles sit on the margin (-0.05000, -0.05004); a
        # single max-Re row linearises one of them and the other violates on
        # the first step (measured: +0.043 after a 2e-3 step, predicted 0).
        # The controller is r x r dense, so all r rows cost nothing.
        ctrl_rows = []              # list of (al_k, g_k)
        if flags['K'] and flags.get('ctrl_mode', 'full') in ('full', 'sp'):
            Acm = self.ctl.A_of(*arow); rr = Acm.shape[0]
            try:
                w_, VL, VR = _seig(Acm, left=True, right=True)
                for k in np.argsort(-w_.real):
                    if w_[k].real < -self.eps - 1.0:      # far inside: skip
                        continue
                    vr, wl = VR[:, k], VL[:, k]; den = np.vdot(wl, vr)
                    if abs(den) < 1e-12:
                        continue
                    Gm = np.outer(wl.conj(), vr) / den
                    gc = pk.zero_grad() if (getattr(pk, 'mimo', False) or getattr(pk, 'vec_pos', False)) \
                        else dict(n=np.zeros(rr, complex), K=0j, xa=0.0, xs=0.0)
                    gc.update(d=np.array([Gm[i, i] for i in range(rr)]),
                              e=np.array([Gm[i, i + 1] for i in range(rr - 1)]),
                              f=np.array([Gm[i + 1, i] for i in range(rr - 1)]))
                    ctrl_rows.append((float(w_[k].real), np.real(np.asarray(pk.grad(gc), float))))
            except Exception:
                ctrl_rows.append((float(np.max(np.linalg.eigvals(Acm).real)), np.zeros(NZ)))
        al_ctrl = max((r_[0] for r_ in ctrl_rows), default=-np.inf)
        # ---- compensator constraint in Burke-Overton's coordinates (TRACKER H):
        # the two rightmost poles compressed to a 2x2 block with biorthogonal
        # bases; Hermite's criterion on the block's shifted polynomial is a
        # 4 x 4 real matrix, positive definite iff both poles have Re < -eps,
        # smooth in A_c (the cluster is separated), well-defined ON the margin.
        # Gradients by central differences with the bases held fixed (first-
        # order exact for the cluster's symmetric functions).
        hermite = None
        al_ctrl_rest = al_ctrl
        pair = None
        if self.pair_mode and flags['K']:
            # ON the double-pole manifold: the pair enters as three equality rows
            # (pair_data), the Hermite block is not built
            pair = self.pair_data(z)
            ctrl_rows = [(al_k, g_k) for (al_k, g_k) in ctrl_rows
                         if not any(abs(al_k - l.real) < 1e-6 for l in pair['lam'])]
            al_ctrl_rest = max((r_[0] for r_ in ctrl_rows), default=-np.inf)
        elif self.ctrl_block and flags['K'] and flags.get('ctrl_mode', 'full') in ('full', 'sp'):
            try:
                import hurwitz as _hw
                Acm = self.ctl.A_of(*arow)
                U2, V2, lam2 = _hw.cluster_block(Acm, 2, pick=_hw.pick_pair(Acm))
                lim_c = max(-self.eps, self.ctrl_max)
                # EXACT gradient (2026-09-22): H depends on the block only
                # through its trace and determinant; A_c is affine in z, so the
                # directions dA_c/dz_j are fixed (cached) and zero for every
                # non-compensator parameter.  Replaces central differences
                # (two compressions per design parameter, noise on the ~zero
                # directions); agreement 7e-10 of the largest gradient.
                if self._E_ctrl is None:
                    A00 = self.ctl.A_of(*pk.unpack(np.zeros(NZ))[0])
                    Ed = []
                    for j in range(NZ):
                        ez = np.zeros(NZ); ez[j] = 1.0
                        dA = self.ctl.A_of(*pk.unpack(ez)[0]) - A00
                        Ed.append(dA if np.any(dA) else None)
                    self._E_ctrl = Ed
                _Mb = U2.conj().T @ (Acm - lim_c * np.eye(Acm.shape[0])) @ V2
                _dBs = [None if dA is None else (U2.conj().T @ dA @ V2) for dA in self._E_ctrl]
                H0, dHl, rho = _hw.hermite_from_block(_Mb, _dBs)
                # scale: the trace, but never below the change a step of 1e-3 makes
                # (2026-09-25: with BOTH poles on the line H0 = 0 to 1e-13, the
                # trace normalisation multiplied the gradients by 1e13 and Clarabel
                # failed -- NumericalError / InsufficientProgress -- two rounds
                # running at the ne=100 K+B+C end point)
                scale = max(abs(float(np.trace(H0))),
                            1e-3 * max((float(np.linalg.norm(g_)) for g_ in dHl), default=0.0), 1e-12)
                if self._hscale is None:
                    self._hscale = scale
                sc = self._hscale
                dH = [g / sc for g in dHl]
                H0 = H0
                hermite = dict(H0=H0 / sc, dH=dH, cluster=lam2, rho=rho, shift=-lim_c, sc=sc)
                # the cluster's poles leave the per-pole rows (the block covers them)
                ctrl_rows = [(al_k, g_k) for (al_k, g_k) in ctrl_rows
                             if not any(abs(al_k - l.real) < 1e-9 for l in lam2)]
                al_ctrl_rest = max((r_[0] for r_ in ctrl_rows), default=-np.inf)
            except Exception as _e:
                hermite = None
        # ---- compensator-stability LMI data (note: "implement the LMI approach")
        # A_c is affine in z (verified), so A_c(z + d) = A0 + sum_j d_j E_j exactly.
        # P_k solves the Lyapunov equation for A0 at a slightly smaller margin than
        # eps, so it exists even when poles sit ON the margin (as at the frozen
        # K+B+C, where two do).  Both are 8 x 8 for any plant size.
        lmi = None
        if self.ctrl_lmi and hermite is None and pair is None and flags['K'] and flags.get('ctrl_mode', 'full') in ('full', 'sp'):
            from scipy.linalg import solve_continuous_lyapunov
            A0 = self.ctl.A_of(*arow); rr = A0.shape[0]
            if self._E is None:
                E = []
                for j in range(NZ):
                    ez = np.zeros(NZ); ez[j] = 1.0
                    E.append(self.ctl.A_of(*pk.unpack(z + ez)[0]) - A0)
                self._E = E
            _lim_c = max(-self.eps, self.ctrl_max)
            margin = min(0.8 * (-_lim_c + self.mg), max(-al_ctrl - 0.1 * self.eps, 0.0))
            As = A0 + margin * np.eye(rr)
            try:
                P = solve_continuous_lyapunov(As.conj().T, -np.eye(rr))
                P = 0.5 * (P + P.conj().T)
                if np.linalg.eigvalsh(P).min() <= 0:
                    raise ValueError('P not positive: A_c is not Hurwitz at this margin')
                P = P * (rr / np.trace(P).real)
                lmi = dict(A0=A0, P=P, E=self._E, r=rr, mg=self.mg)
            except Exception:
                lmi = None
        field_rows = self._field_rows(z) if FIELD_KMAX2 else []
        pos_rows = self._pos_rows(z)
        return dict(s=s, Gs=Gs, smax=float(s.max()), blocks=blocks, tops=(dict(self.w_star) if DEBUG_LS else {}), field_rows=field_rows,
                    pos_rows=pos_rows,
                    al=float(al), ga=ga, clr=bool(clr), al_rows=al_rows_qp, al_rows_all=al_rows,
                    al_ctrl=al_ctrl, al_ctrl_rest=al_ctrl_rest, ctrl_rows=ctrl_rows, lmi=lmi, hermite=hermite, pair=pair,
                    hermite_cl=hermite_cl, z=z.copy())

    def _closed_loop_block(self, pr2, cl, clf, clg, arow, nvec, K, xa, xs, pa, ps, na, ns, flags, NZ):
        """Hermite data for the two rightmost near-line closed-loop eigenvalues.
        Reduced pairs (STAGE6 6.29): the lift blockdiag(V, I) / blockdiag(W, I)
        has orthonormal columns, so the QR, M_cl-biorthonormalisation and the
        compression Y^H A_cl X are those of the reduced pencil W^H (A_cl, M_cl) V
        -- no length-n vector."""
        import hurwitz as _hw
        D, pk = self.D, self.pk
        X = np.column_stack([p_['v'] for p_ in pr2]); Y = np.column_stack([p_['u'] for p_ in pr2])
        Xq, _ = np.linalg.qr(X); Yq, _ = np.linalg.qr(Y)
        if pr2[0].get('reduced'):
            Acl_r, Mcl_r = cl._red_pencil
            MX = Mcl_r @ Xq; AX = Acl_r @ Xq
        else:
            MX = np.column_stack([clf.Mcl_apply(Xq[:, j]) for j in range(2)])
            AX = np.column_stack([clf.Acl_apply(Xq[:, j], arow, nvec, K) for j in range(2)])
        Gm = Yq.conj().T @ MX
        Yt = Yq @ np.linalg.inv(Gm).conj().T                 # Yt^H M_cl Xq = I
        B = Yt.conj().T @ AX
        shift = self.eps + self.mg                           # the admissibility line of the rows
        Cg = lambda x_, z_: np.asarray(pk.grad(D.grads(clg, x_, z_, arow, nvec, K, xa, xs,
                                                        pa, ps, na, ns, flags)), float)
        dB = np.zeros((NZ, 2, 2), complex)
        for a_ in range(2):
            for b_ in range(2):
                dB[:, a_, b_] = Cg(Xq[:, b_], Yt[:, a_]) + 1j * Cg(Xq[:, b_], 1j * Yt[:, a_])
        H0, dHl, rho = _hw.hermite_from_block(B + shift * np.eye(2),
                                              [dB[j] if np.any(dB[j]) else None for j in range(NZ)])
        scale = max(abs(float(np.trace(H0))),
                    1e-3 * max((float(np.linalg.norm(g_)) for g_ in dHl), default=0.0), 1e-12)   # see the compensator block
        if self._hscale_cl is None:
            self._hscale_cl = scale
        sc = self._hscale_cl
        return dict(H0=H0 / sc, dH=[g_ / sc for g_ in dHl], cluster=np.linalg.eigvals(B),
                    B=B, dB=dB, lam_pairs=np.array([p_['lam'] for p_ in pr2]))


# ----------------------------------------------------------------- subproblem
def subproblem(ev, t, H, Delta, eps, ctrl_max, enroll_block=True, mg_row=0.0, disp=None, pen=1.0):
    """min dt + 0.5 d'Hd (+ nu_s s)  s.t. linearised rows, block SOCs, abscissa
    rows, box, and -- when ev['lmi'] is set -- the linearised compensator
    Lyapunov LMI with an elastic slack s.  Variables x = [d (NZ), dt, q (r^2
    Hermitian coordinates of dP), s].  pen scales the elastic slacks' costs
    with the merit's penalty: pen = nu / 100 (solve_epigraph_sqp).  Returns (dict, status)."""
    NZ = len(ev['z'])
    lmi = ev.get('lmi'); use_lmi = lmi is not None
    nq = lmi['r'] ** 2 if use_lmi else 0
    # elastic slacks (S-l1-QP): one per abscissa row and per compensator row,
    # penalised at nu_s in the objective, so a start that violates a margin
    # (the K+B design's compensator sits at -0.034 against the inner
    # problem's -eps) is walked into feasibility instead of declared
    # PrimalInfeasible by a trust box too small to repair it in one step
    n_el = len(ev['al_rows']) + len(ev.get('field_rows', [])) + (0 if use_lmi else len(ev['ctrl_rows']))
    # Hermite blocks: the compensator's and (2026-09-24) the closed loop's, each
    # a PSD cone with its own elastic slack
    herm_list = [h_ for h_ in (ev.get('hermite'), ev.get('hermite_cl')) if h_ is not None]
    use_h = bool(herm_list)
    n = NZ + 1 + nq + (1 if use_lmi else 0) + n_el + len(herm_list)
    iq, is_ = NZ + 1, NZ + 1 + nq
    iel = NZ + 1 + nq + (1 if use_lmi else 0)
    ih = iel + n_el                                       # block-Hermite slack (if any)
    rows_A, rows_b, cones = [], [], []
    # --- nonnegative cone rows: A x + s = b, s >= 0  <=>  A x <= b
    lin_A, lin_b = [], []
    for i in range(len(ev['s'])):
        if enroll_block and i in ev['blocks']:
            continue
        a = np.zeros(n); a[:NZ] = ev['Gs'][i]; a[NZ] = -1.0
        lin_A.append(a); lin_b.append(t - ev['s'][i])
    kel = 0
    stab_rows = []                                        # rows whose duals are the stability multipliers (adaptive penalty)
    for al_j, g_j in ev['al_rows']:                       # one row per near-line eigenvalue
        # the ADMISSIBILITY line (alpha + eps <= -mg), not the constraint line:
        # rows at +0 left every iterate exactly on the line and inadmissible
        stab_rows.append(len(lin_A))
        a = np.zeros(n); a[:NZ] = g_j; a[iel + kel] = -1.0; lin_A.append(a); lin_b.append(-(al_j + eps + mg_row))
        a = np.zeros(n); a[iel + kel] = -1.0; lin_A.append(a); lin_b.append(0.0)      # slack >= 0
        kel += 1
    for c_f, g_f in ev.get('field_rows', []):             # field shape rows c + g.d <= 0 (elastic)
        stab_rows.append(len(lin_A))
        a = np.zeros(n); a[:NZ] = g_f; a[iel + kel] = -1.0; lin_A.append(a); lin_b.append(-c_f)
        a = np.zeros(n); a[iel + kel] = -1.0; lin_A.append(a); lin_b.append(0.0)
        kel += 1
    for c_p, g_p in ev.get('pos_rows', []):               # position rows c + g.d <= 0 (hard: linear, start feasible)
        a = np.zeros(n); a[:NZ] = g_p; lin_A.append(a); lin_b.append(-c_p)
    if not use_lmi:
        # the compensator rows use the GATE's line (allowance ctrl_max, i.e.
        # Re <= 0 by decision), not the frozen inner's -eps: the K+B design
        # sits between the two and a stricter row is infeasible at the start
        lim_c = max(-eps, ctrl_max)
        for al_k, g_k in ev['ctrl_rows']:
            stab_rows.append(len(lin_A))
            a = np.zeros(n); a[:NZ] = g_k; a[iel + kel] = -1.0; lin_A.append(a)
            lin_b.append(-(al_k - lim_c))
            a = np.zeros(n); a[iel + kel] = -1.0; lin_A.append(a); lin_b.append(0.0)
            kel += 1
    else:
        a = np.zeros(n); a[is_] = -1.0; lin_A.append(a); lin_b.append(0.0)   # s >= 0
    for kh in range(len(herm_list)):
        a = np.zeros(n); a[ih + kh] = -1.0; lin_A.append(a); lin_b.append(0.0)    # s_h >= 0
    use_ball = PAIR_BALL and ev.get('pair') is not None
    if not use_ball:
        for j in range(NZ):                  # box trust region
            a = np.zeros(n); a[j] = 1.0; lin_A.append(a); lin_b.append(Delta)
            a = np.zeros(n); a[j] = -1.0; lin_A.append(a); lin_b.append(Delta)
    i_pair = None
    if ev.get('pair') is not None:
        # the pair on S: c_u <= 0 (both poles, equal real parts, left of the line
        # by kappa) and c_a <= 0 (vertical split or double pole; > 0 would be a
        # horizontal split): the last two linear rows
        i_pair = len(lin_A)
        for k in (0, 1):
            a = np.zeros(n); a[:NZ] = ev['pair']['J'][k]; lin_A.append(a); lin_b.append(-ev['pair']['c'][k])
    n_lin = len(lin_A)
    rows_A += lin_A; rows_b += lin_b; cones.append(clarabel.NonnegativeConeT(n_lin))
    # --- block SOCs: p = t+dt-s1-h11, r = t+dt-s2-h22, q = -h12  (h = sym(G d))
    #     (p+r, p-r, 2Re q, 2Im q) in SOC.  Each entry = b_k - A_k x.
    blk_idx = []
    for i, B in ev['blocks'].items():
        if not enroll_block:
            continue
        Gc = B['Gc']; s1, s2 = B['sig']
        h11 = np.real(Gc[0, 0]); h22 = np.real(Gc[1, 1])
        h12 = 0.5 * (Gc[0, 1] + np.conj(Gc[1, 0]))          # complex vector
        # p = (t - s1) + dt - h11.d ;  r = (t - s2) + dt - h22.d ;  q = -h12.d
        def row(coef_d, coef_dt, const):
            # cone entry = b_k - A_k x  must equal  const + coef_d.d + coef_dt.dt
            # => A_k = -coef, b_k = const.  (A sign slip here once flipped dt and
            # the block gradients: the subproblem then predicted dt = -0.30 where
            # the true sigma_1 allowed -0.07, and every step was rejected.)
            a = np.zeros(n); a[:NZ] = coef_d; a[NZ] = coef_dt
            return -a, const
        A1, b1 = row(-(h11 + h22), 2.0, (t - s1) + (t - s2))   # p + r
        A2, b2 = row(-(h11 - h22), 0.0, (t - s1) - (t - s2))   # p - r
        A3, b3 = row(-2 * np.real(h12), 0.0, 0.0)              # 2 Re q
        A4, b4 = row(-2 * np.imag(h12), 0.0, 0.0)              # 2 Im q
        for A_, b_ in ((A1, b1), (A2, b2), (A3, b3), (A4, b4)):
            rows_A.append(A_); rows_b.append(b_)
        cones.append(clarabel.SecondOrderConeT(4)); blk_idx.append(i)
    # --- total-displacement bound (2026-09-25): ||(z - z0) + d||_2 <= rho, the
    #     driver's step governor imposed INSIDE the solve, so every iterate (and
    #     the design returned) satisfies it and has been seen by refinement; the
    #     post-hoc clip certified points the SQP had never evaluated
    if disp is not None:
        dz0, rho_d = disp
        A_ = np.zeros((1 + NZ, n)); b_ = np.zeros(1 + NZ)
        b_[0] = rho_d                                         # cone entry 0: rho
        for j in range(NZ):
            A_[1 + j, j] = -1.0; b_[1 + j] = dz0[j]           # entries: dz0 + d
        rows_A += list(A_); rows_b += list(b_); cones.append(clarabel.SecondOrderConeT(1 + NZ))
        blk_idx_disp = True
    if use_ball:
        # 2-norm trust region (2026-09-25): with a box and a Hessian far below the
        # rows' curvature the QP step sat on a corner of the box (a sign pattern,
        # |d|_2 = 8 x the radius in 64 dimensions); the ball gives the
        # Levenberg-Marquardt direction
        A_ = np.zeros((1 + NZ, n)); b_ = np.zeros(1 + NZ); b_[0] = Delta
        for j in range(NZ):
            A_[1 + j, j] = -1.0
        rows_A += list(A_); rows_b += list(b_); cones.append(clarabel.SecondOrderConeT(1 + NZ))
    # --- compensator LMI:  -M_lin(d, Q) + s I >= 0  and  P + Q - delta I >= 0
    #     M_lin = M0 + sum_j d_j (E_j^H P + P E_j) + (A0^H Q + Q A0 + 2 eps Q),
    #     M0 = A0^H P + P A0 + 2 eps P.   Real embedding, cone entry = b - A x.
    n_lmi_rows = 0
    if use_lmi:
        A0, Pk, E, rr = lmi['A0'], lmi['P'], lmi['E'], lmi['r']
        Bq = _herm_basis(rr)
        # the LMI certifies the GATE's line: Re <= ctrl_max (0 by decision), plus
        # the admissibility cushion -- not eps, which the frozen inner over-asked
        eps_l = -max(-eps, ctrl_max)                  # the LMI certifies the exact line
        M0 = A0.conj().T @ Pk + Pk @ A0 + 2 * eps_l * Pk
        nsv = (2 * rr) * (2 * rr + 1) // 2
        Ablk = np.zeros((nsv, n)); bblk = _svec(_emb(-M0))
        for j in range(NZ):
            Cj = E[j].conj().T @ Pk + Pk @ E[j]
            if np.any(Cj):
                Ablk[:, j] = _svec(_emb(Cj))
        for k, Bk in enumerate(Bq):
            Dk = A0.conj().T @ Bk + Bk @ A0 + 2 * eps_l * Bk
            Ablk[:, iq + k] = _svec(_emb(Dk))
        Ablk[:, is_] = -_svec(_emb(np.eye(rr)))
        rows_A += list(Ablk); rows_b += list(bblk); cones.append(clarabel.PSDTriangleConeT(2 * rr))
        # trust region on dP:  -rho P <= dP <= rho P  (implies P + dP > 0).  Without
        # it the subproblem satisfies the LINEARISED LMI with ||dP|| = 2 ||P||
        # (measured) and the neglected bilinear term E(d)^H dP + dP E(d) is
        # anything but small: the true pole then moves outward linearly along
        # the ray the subproblem called stability-preserving.
        rho = lmi.get('rho', 0.5)
        for sgn in (+1.0, -1.0):                      # rho P - dP >= 0 ;  rho P + dP >= 0
            Ar = np.zeros((nsv, n)); br = _svec(_emb(rho * Pk))
            for k, Bk in enumerate(Bq):
                Ar[:, iq + k] = sgn * _svec(_emb(Bk))
            rows_A += list(Ar); rows_b += list(br); cones.append(clarabel.PSDTriangleConeT(2 * rr))
        n_lmi_rows = nsv
    # --- block-Hermite cones:  H0 + sum_j d_j dH_j + s_h I >= 0  (compensator, closed loop)
    n_h_list = []
    for kh, herm in enumerate(herm_list):
        Hs = herm['H0'].shape[0]; nsv_h = Hs * (Hs + 1) // 2
        Ah = np.zeros((nsv_h, n)); bh = _svec(herm['H0'])
        for j in range(NZ):
            if np.any(herm['dH'][j]):
                Ah[:, j] = -_svec(herm['dH'][j])
        Ah[:, ih + kh] = -_svec(np.eye(Hs))              # slack column
        rows_A += list(Ah); rows_b += list(bh); cones.append(clarabel.PSDTriangleConeT(Hs))
        n_h_list.append(nsv_h)
    n_h = int(sum(n_h_list))
    # --- the compensator pair on its double-pole manifold: J d = -c (zero cone, last)
    n_pair = 0
    if ev.get('pair') is not None:
        Jp, cp = ev['pair']['J'], ev['pair']['c']
        a = np.zeros(n); a[:NZ] = Jp[2]; rows_A.append(a); rows_b.append(-cp[2])   # c_b = 0: equal real parts
        cones.append(clarabel.ZeroConeT(1)); n_pair = 1
    A = sp.csc_matrix(np.array(rows_A)); bvec = np.array(rows_b, float)
    Pd = np.zeros((n, n)); Pd[:NZ, :NZ] = 0.5 * (H + H.T); Pd[NZ, NZ] = 1e-8
    if use_lmi:
        Pd[iq:iq + nq, iq:iq + nq] = 1e-6 * np.eye(nq)     # keep dP bounded
    P = sp.csc_matrix(Pd)
    q = np.zeros(n); q[NZ] = 1.0
    if use_lmi:
        q[is_] = 1e3 * pen                                    # elastic slack, expensive
    q[iel:iel + n_el] = 1e2 * pen                             # row slacks, at the merit's nu
    for kh in range(len(herm_list)):
        q[ih + kh] = 1e3 * pen                                # block-Hermite slacks, expensive
    st = clarabel.DefaultSettings(); st.verbose = False
    st.max_iter = 200; st.tol_gap_abs = 1e-10; st.tol_gap_rel = 1e-10; st.tol_feas = 1e-10
    # ONE thread (2026-09-30): Clarabel's default (max_threads = 0: all cores)
    # made the K+B+C central QPs 28 % slower on 2 cores than one thread, with
    # bitwise the same solution, and competed with the row workers
    st.max_threads = 1
    sol = clarabel.DefaultSolver(P, q, A, bvec, cones, st).solve()
    status = str(sol.status)
    if 'Solved' not in status:
        return None, status
    x = np.asarray(sol.x); zdual = np.asarray(sol.z)
    n_soc = 4 * len(blk_idx)
    out = dict(d=x[:NZ], dt=float(x[NZ]), lin=zdual[:n_lin], soc=zdual[n_lin:n_lin + n_soc],
               blk=blk_idx, n_lin=n_lin)
    # for the adaptive penalty (solve_epigraph_sqp, pen_adapt): the largest
    # elastic slack the step uses (the linearised stability constraints it
    # gives up), and the largest stability multiplier
    _sl = list(x[iel:iel + n_el]) + [x[ih + kh] for kh in range(len(herm_list))] + ([x[is_]] if use_lmi else [])
    out['slack_max'] = float(max(_sl)) if _sl else 0.0
    out['stab_dual'] = float(max((abs(zdual[i_]) for i_ in stab_rows), default=0.0))
    if use_h:
        hd = zdual[len(zdual) - n_pair - n_h: len(zdual) - n_pair]; off = 0; duals = []
        for nh_ in n_h_list:
            duals.append(hd[off:off + nh_]); off += nh_
        out['hermite_duals'] = duals                          # in herm_list order
        out['hermite_list'] = herm_list
    if n_pair:
        out['pair_dual'] = np.array([zdual[i_pair], zdual[i_pair + 1], zdual[len(zdual) - 1]])   # (mu_u >= 0, mu_a >= 0, mu_b free)
    if use_lmi:
        n_disp = ((1 + NZ) if disp is not None else 0) + ((1 + NZ) if use_ball else 0)   # the displacement / ball cones precede the LMI cones
        out['lmi_dual'] = zdual[n_lin + n_soc + n_disp: n_lin + n_soc + n_disp + n_lmi_rows]
        out['slack'] = float(x[is_])
        out['dP'] = sum(x[iq + k] * Bk for k, Bk in enumerate(Bq))
    return out, status


# ---------------------------------------------------------------------- SQP
def solve_epigraph_sqp(D, pk, flags, z0, Omega, t0, eps, maxit=25, ctrl_max=0.0,
                       enroll=None, kink_block=True, trust0=None, tol=1e-6,
                       sel_margin=1e-3, verbose=False, H0=None, ctrl_lmi=False,
                       oracle=None, reach=1.0, res_tol=1e-4,
                       refine_tol=None, refine_grid=None, refine_max=3,
                       alpha_ref=None, alpha_ref_tol=1e-3, stat_rel=1e-4, pred_rel=1e-7, ctrl_block=False,
                       follow=None, follow_radius=0.1, row_screen=0.0, validate_full=True,
                       cl_block=False, max_disp=None, pen_adapt=False):
    if clarabel is None:
        raise RuntimeError("clarabel is not installed (pip install clarabel)")
    W = np.asarray(sorted(Omega), float)
    mg = sel_margin * eps
    ev_fn = Evaluator(D, pk, flags, W, eps, ctrl_max, enroll, kink_block, ctrl_lmi, mg,
                      oracle=oracle, reach=reach, res_tol=res_tol, ctrl_block=ctrl_block, follow=follow,
                      row_screen=row_screen, cl_block=cl_block)
    if oracle is not None and alpha_ref is not None and np.isfinite(alpha_ref):
        # the reduced pencil's abscissa at the START of a round can be stale
        # after a large move (round 1 of scaled_002 read a certified design as
        # inadmissible and returned one 5x worse); compare with the certified
        # value the gate just produced, and use the full certificate for this
        # solve when they disagree
        e_chk = ev_fn(np.asarray(z0, float))
        if abs(e_chk['al'] - alpha_ref) > alpha_ref_tol:
            ev_fn.force_full_alpha = True
            if verbose:
                print(f"    sqp: reduced abscissa {e_chk['al']:+.5f} vs certified {alpha_ref:+.5f} "
                      f"at the start: full certificate for this solve", flush=True)
    NZ = len(z0)
    z = np.asarray(z0, float).copy()
    z_start = z.copy()                                   # for the total-displacement bound (max_disp)
    pair_log = []
    def _try_enter(z_, where):
        # ENTER the double-pole manifold when the pair is (nearly) a double pole
        # at the line: project onto it (compensator coordinates only).  Returns
        # the projected point, or None.
        # enter when the pair is nearly defective near the line, or when BOTH its
        # poles sit on the line (any separation: the face S is exact there, and
        # the retraction removes the second-order excursions the pole-based
        # merit refused -- ne=200: both poles at -0.05 +- 1e-9, 0.057 apart,
        # every step cut to alpha ~ 1e-4)
        try:
            _pd0 = ev_fn.pair_data(z_)
            _l2 = np.asarray(_pd0['lam'], complex)
            if (flags.get('real_ctrl') and abs(_l2[0] - _l2[1]) > PAIR_ENTER_SEP
                    and abs(_l2[0] - np.conj(_l2[1])) <= 1e-8 * max(1.0, abs(_l2[0]))):
                # a REAL compensator's complex-conjugate pair on the line, far
                # apart, is one smooth constraint (its two real parts are equal by
                # symmetry), not a double pole: never the face S (2026-10-05, BFS:
                # entering it with the pair 3.6 apart projected z by 0.49 and the
                # iterate went unstable).  A CLOSE conjugate pair is a double real
                # pole split by rounding (a defective pole's eigenvalues come out
                # as -0.05 +- i 1e-8 and worse): that one is the face S's case.
                return None
            if (-PAIR_ENTER_U <= _pd0['exc'] <= 1e-8 and
                    (_pd0['sep'] <= PAIR_ENTER_SEP or _pd0['low'] >= -PAIR_ON_LINE)):
                _zp, _res = ev_fn.retract(z_)
                if _res <= 1e-12:
                    pair_log.append(dict(event='enter', where=where, sep=_pd0['sep'], exc=_pd0['exc'],
                                         move=float(np.linalg.norm(_zp - np.asarray(z_, float))), res=_res))
                    if verbose:
                        print(f"    sqp: pair ON the double-pole manifold ({where}; sep {_pd0['sep']:.2e}, "
                              f"exc {_pd0['exc']:+.2e}; projection {pair_log[-1]['move']:.2e})", flush=True)
                    return _zp
                pair_log.append(dict(event='retraction failed', where=where, sep=_pd0['sep'], res=_res))
        except Exception as _e:
            pair_log.append(dict(event='error', msg=f"{type(_e).__name__}: {str(_e)[:80]}"))
        return None
    _pair_ok = PAIR_MANIFOLD and ctrl_block and flags.get('K')
    if _pair_ok:
        _zp = _try_enter(z, 'start')
        if _zp is not None:
            ev_fn.pair_mode = True; z = _zp
    e = ev_fn(z); t = max(float(t0), e['smax'])
    t0_start = float(e['smax'])                       # the start's value on the SQP's own model
    H = np.eye(NZ) if H0 is None else np.asarray(H0, float).copy()
    Delta = trust0 if trust0 else 1e-3 * max(float(np.linalg.norm(z)), 1.0)   # row gradients ~1e3-1e4: the linear model holds for steps ~1e-4 |z|
    # merit penalty nu, and the QP's elastic slack costs (nu for the rows, 10 nu
    # for the Hermite and LMI slacks).  Fixed at 100 (GL), or ADAPTIVE
    # (pen_adapt, BFS.md 5, 2026-10-05): an exact penalty needs nu above the
    # stability rows' multipliers, which scale with sigma -- at the fixed 100 a
    # step bought 12 of sigma ~ 750 with a compensator pole 0.025 across its
    # line, and the solve was trapped on that inadmissible iterate.  The rules
    # (Byrd, Nocedal & Waltz 2008, steering; Powell's update):
    #   steering: while the QP step gives up a linearised stability constraint
    #     (an elastic slack > 0) that a larger nu keeps, nu x 10 and solve again;
    #   update:   when it keeps them all, nu >= 1.5 x the largest stability
    #     multiplier; nu never decreases within a solve.
    nu = 100.0
    best = {'g': np.inf, 'z': None, 'al': None, 't': np.inf, 'clr': None}
    lim = max(-eps, ctrl_max)
    hist = []; code = 4; t_start = time.time(); added = []; exit_info = None
    sub_status_log = []                                  # subproblem failures (status strings)
    if refine_grid is None:
        refine_grid = np.linspace(float(W[0]), float(W[-1]), 1201)

    def _sub(e_, t_, H_, D_, disp_):
        """The QP at (e_, t_) with the current nu; with pen_adapt, nu steered and
        updated (see above)."""
        nonlocal nu
        sub_, st_ = subproblem(e_, t_, H_, D_, eps, ctrl_max, kink_block, mg_row=mg, disp=disp_, pen=nu / 100.0)
        if not pen_adapt or sub_ is None:
            return sub_, st_
        for _ in range(PEN_STEER):
            if sub_['slack_max'] <= PEN_SLACK_TOL:
                break
            s2, st2 = subproblem(e_, t_, H_, D_, eps, ctrl_max, kink_block, mg_row=mg, disp=disp_, pen=10.0 * nu / 100.0)
            if s2 is None:
                break
            if s2['slack_max'] > 0.9 * sub_['slack_max']:
                break                                   # the linearisation is infeasible here: more weight does not help
            if verbose:
                print(f"      penalty steered: nu {nu:.3g} -> {10 * nu:.3g} (slack {sub_['slack_max']:.2e} -> "
                      f"{s2['slack_max']:.2e})", flush=True)
            nu *= 10.0; sub_, st_ = s2, st2
        if sub_['slack_max'] <= PEN_SLACK_TOL:
            nu = max(nu, 1.5 * sub_['stab_dual'])
        return sub_, st_

    _Ec = [None]; n_soc = [0]
    def _ctrl_restore(zn_, en_):
        """SECOND-ORDER CORRECTION on the compensator (pen_adapt; 2026-10-05): a
        step along the linearised boundary Re lambda(A_c) = lim leaves it at
        second order (the line is curved in z), and with an exact penalty the
        merit then refuses ever smaller steps (Maratos).  The poles at or over
        the line are put back on it by the least-norm Newton correction in the
        compensator's coordinates -- A_c is affine in z, the 10 x 10 eigenproblem
        is exact and cheap.  Skipped (None) where two of them nearly coincide
        (their gradients blow up: the double-pole face S handles that case)."""
        Am = ev_fn.ctl.A_of(*pk.unpack(zn_)[0])
        if _Ec[0] is None:
            A00 = ev_fn.ctl.A_of(*pk.unpack(np.zeros(NZ))[0])
            _Ec[0] = [ev_fn.ctl.A_of(*pk.unpack(np.eye(NZ)[j])[0]) - A00 for j in range(NZ)]
        from scipy.linalg import eig as _deig
        lam_, VL_, VR_ = _deig(Am, left=True, right=True)
        act = [i for i in range(len(lam_)) if lam_[i].real > lim - 1e-6 and lam_[i].imag >= -1e-12]
        # a nearly double pole (two within 1e-3; a double REAL pole is computed as a
        # close conjugate pair, so its partner may have Im < 0) is the face S's:
        # on the face it is left to the retraction, off it nothing is corrected
        close = [i for i in act if any(abs(lam_[i] - lam_[j]) < 1e-3 for j in range(len(lam_)) if j != i)]
        if close and not ev_fn.pair_mode:
            return None
        act = [i for i in act if i not in close]
        if not act or len(act) > 4:
            return None
        G = np.zeros((len(act), NZ)); r_ = np.zeros(len(act))
        for k, i in enumerate(act):
            u, v = VL_[:, i], VR_[:, i]; den = np.vdot(u, v)
            if abs(den) < 1e-8:
                return None
            G[k] = [np.real(np.vdot(u, Ej @ v) / den) if np.any(Ej) else 0.0 for Ej in _Ec[0]]
            r_[k] = (lim - 1e-7) - lam_[i].real
        try:
            dz = G.T @ np.linalg.solve(G @ G.T, r_)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(dz)) or np.linalg.norm(dz) > 0.5 * max(np.linalg.norm(zn_ - z), 1e-12):
            return None                                  # not a small correction: leave the trial as it is
        zc_ = zn_ + dz
        if ev_fn.pair_mode:
            zc_, _ = ev_fn.retract(zc_)                  # the pair back on its face
        ec_ = ev_fn(zc_)
        key_ = 'al_ctrl_rest' if ev_fn.pair_mode else 'al_ctrl'
        return (zc_, ec_) if ec_[key_] <= lim + 1e-8 else None

    def viol(e_, t_):
        v = max(e_['smax'] - t_, 0.0) + max(e_['al'] + eps + mg, 0.0)
        v += sum(max(c_f, 0.0) for c_f, _ in e_.get('field_rows', []))
        v += sum(max(c_p, 0.0) for c_p, _ in e_.get('pos_rows', []))
        if e_.get('pair') is not None:
            # on the manifold (retracted iterates): the pair's excursion is -kappa;
            # it counts only if the retraction failed.  The other poles by Re.
            v += max(e_['pair']['exc'] - 1e-8, 0.0)
            if np.isfinite(e_.get('al_ctrl_rest', -np.inf)):
                v += max(e_['al_ctrl_rest'] - lim - 1e-8, 0.0)
            return v
        if np.isfinite(e_['al_ctrl']):
            # against the ADMISSIBILITY line lim - mg, the same line the LMI
            # asks for: a start with poles ON the margin is then a violation
            # the subproblem reduces at first order, and the merit rewards it
            # deadband of mg on the margin: an excursion smaller than the
            # admissibility margin is not penalised (pyGRANSO's viol tolerance
            # plays the same role); the LMI keeps the poles from crossing, and
            # admissibility of the RETURNED iterate stays strict
            v += max(e_['al_ctrl'] - lim - 1e-8, 0.0)   # exact line for the compensator (no cushion; rounding tolerance)
        return v

    def merit(e_, t_):
        # max(t, smax) + nu * (stability violations)   (2026-09-25).  The epigraph
        # rows' duals sum to 1 (the dual of dt), so weight 1 on smax - t is the
        # exact threshold, and the merit IS the objective when t is tight.  (A
        # weight nu = 100 on it, the earlier merit, gave the same path in the
        # replays -- the compensator decided the line search there -- and is
        # removed.)
        return max(t_, e_['smax']) + nu * (viol(e_, t_) - max(e_['smax'] - t_, 0.0))

    def admissible(e_):
        ok_c = (not np.isfinite(e_['al_ctrl']) or e_['al_ctrl'] <= lim + 1e-8)   # exact constraint, boundary by design: rounding tolerance only
        if e_.get('pair') is not None:
            # the pair judged from its sum and product (exact at a double pole,
            # where the computed poles split by sqrt(eps)); the others by Re
            ok_c = (e_['pair']['exc'] <= 1e-8 and
                    (not np.isfinite(e_.get('al_ctrl_rest', -np.inf)) or e_['al_ctrl_rest'] <= lim + 1e-8))
        ok_f = all(c_f <= 1e-9 for c_f, _ in e_.get('field_rows', []) + e_.get('pos_rows', []))
        return e_['clr'] and e_['al'] + eps <= -mg and ok_c and ok_f

    def grad_L(e_, sub):
        g = np.zeros(NZ)
        if sub is None:
            return g
        k = 0
        for i in range(len(e_['s'])):
            if kink_block and i in e_['blocks']:
                continue
            g += sub['lin'][k] * e_['Gs'][i]; k += 1
        for al_j, g_j in e_['al_rows']:
            g += sub['lin'][k] * g_j; k += 2          # row, then its slack>=0 row
        for c_f, g_f in e_.get('field_rows', []):
            g += sub['lin'][k] * g_f; k += 2
        for c_p, g_p in e_.get('pos_rows', []):
            g += sub['lin'][k] * g_p; k += 1
        if e_.get('lmi') is None:
            for al_k, g_k in e_['ctrl_rows']:
                g += sub['lin'][k] * g_k; k += 2
        if sub.get('pair_dual') is not None and e_.get('pair') is not None:
            g += sub['pair_dual'] @ e_['pair']['J']          # the pair's equality rows (same convention as the rows)
        if sub.get('hermite_duals') is not None:
            for herm_, lamh in zip(sub['hermite_list'], sub['hermite_duals']):
                for j in range(NZ):
                    if np.any(herm_['dH'][j]):
                        g[j] -= float(lamh @ _svec(herm_['dH'][j]))
        if e_.get('hermite') is None:
            k += 1                                   # the s >= 0 row
            lam = sub.get('lmi_dual')
            if lam is not None:
                Pk, E = e_['lmi']['P'], e_['lmi']['E']
                for j in range(NZ):
                    Cj = E[j].conj().T @ Pk + Pk @ E[j]
                    if np.any(Cj):
                        g[j] += float(lam @ _svec(_emb(Cj)))
        for j, i in enumerate(sub['blk']):
            y = sub['soc'][4 * j: 4 * j + 4]
            Y = np.array([[0.5 * (y[0] + y[1]), 0.5 * (y[2] + 1j * y[3])],
                          [0.5 * (y[2] - 1j * y[3]), 0.5 * (y[0] - y[1])]])
            Gc = e_['blocks'][i]['Gc']
            Hs = 0.5 * (Gc + np.conj(np.transpose(Gc, (1, 0, 2))))
            g += np.real(np.einsum('ab,bak->k', Y, Hs))
        return g

    def _argw(e_):
        # the frequency where the rows' maximum is attained (the followed top when
        # the member follows): the certificate's witness when it proves this value
        i_ = int(np.argmax(e_['s'])); w0_ = float(ev_fn.W[i_])
        return float(ev_fn.w_star.get(round(w0_, 6), w0_))

    _alpha_prev = 1.0; _last_pair_mu = None
    def _tnorm(v):
        # the trust region's norm: the ball on the pair's face (PAIR_BALL), else the box
        if PAIR_BALL and ev_fn.pair_mode:
            return float(np.linalg.norm(v))
        return float(np.max(np.abs(v)))
    cands = []                                            # (smax, z, al, t) admissible iterates, for the certified fallback
    def _note(e_, t_):
        if admissible(e_):
            cands.append((e_['smax'], e_['z'].copy(), e_['al'], t_))
            cands.sort(key=lambda c: c[0]); del cands[6:]
    _note(e, t)
    if admissible(e) and e['smax'] < best['g']:
        best.update(g=e['smax'], z=z.copy(), al=e['al'], t=t, clr=e['clr'], w=_argw(e))
    for it in range(int(maxit)):
        _disp = (z - z_start, float(max_disp)) if max_disp is not None else None
        sub, status = _sub(e, t, H, Delta, _disp)
        if sub is None:
            # 2026-09-25: at the ne=100 K+B+C fixed point the subproblem failed
            # (code 12) three rounds running, while the same subproblem with a
            # fresh Hessian solves: reset the quasi-Newton matrix and retry once
            sub_status_log.append(str(status))
            H = np.eye(NZ) * max(float(np.mean(np.abs(np.diag(H)))), 1e-8)
            sub, status = _sub(e, t, H, Delta, _disp)
            if sub is None:
                sub_status_log.append(str(status) + ' (after reset)')
                code = 12; hist.append(dict(it=it, status=status)); break
        if ev_fn.pair_mode and sub.get('pair_dual') is not None:
            _last_pair_mu = sub['pair_dual']
        d, dt = sub['d'], sub['dt']
        gL = grad_L(e, sub)
        stat = float(np.linalg.norm(gL))
        pred = merit(e, t) - (t + dt)                   # model merit at the QP step: t + dt (rows and stability rows linearly satisfied)
        m0 = merit(e, t)
        # WARM line search (2026-09-25): start at twice the last accepted alpha --
        # in the tail every search walked down from 1 to 1/64 .. 1/128, six or
        # seven evaluations per iteration (44 % of a round's evaluations)
        # ... but not where the radius itself is shrunk to the accepted step (the
        # pair's face): there the radius is the memory, and a warm alpha < 1 on
        # top of it ratcheted both down (ne=100 round 11: radius 2.5e-7, alpha
        # never back to 1; the same design from alpha = 1 moved 7.9285 -> 7.8808)
        _shrinking = PAIR_BALL and ev_fn.pair_mode
        alpha, accepted = (min(1.0, 2.0 * _alpha_prev) if (ALPHA_WARM and not _shrinking) else 1.0), False
        _a0 = alpha
        for _ls in range(8):
            zn = z + alpha * d; tn = t + alpha * dt
            if ev_fn.pair_mode:
                zn, _rres = ev_fn.retract(zn)             # back onto the manifold (second order in the step)
            en = ev_fn(zn)
            _kc = 'al_ctrl_rest' if ev_fn.pair_mode else 'al_ctrl'
            if (pen_adapt and np.isfinite(en.get(_kc, -np.inf)) and en[_kc] > lim + 1e-8
                    and e.get(_kc, -np.inf) <= lim + 1e-8):
                _rc = _ctrl_restore(zn, en)
                if _rc is not None:
                    zn, en = _rc; n_soc[0] += 1
            if DEBUG_LS:
                # the row attaining the trial's max: actual value vs its linear model
                _i = int(np.argmax(en['s']))
                if len(en['s']) == len(e['s']):
                    print(f"      ls it {it} a={alpha:.3g}: max row {_i} w={float(ev_fn.W[_i]):+.4f}"
                          f"{' (kink block)' if _i in e['blocks'] else ''}: {en['s'][_i]:.7f} vs model "
                          f"{e['s'][_i] + alpha * float(e['Gs'][_i] @ d):.7f} (was {e['s'][_i]:.7f}); "
                          f"retraction moved {np.linalg.norm(zn - z - alpha * d):.2e} of {alpha * np.linalg.norm(d):.2e}", flush=True)
                _pp = en.get('pair')
                print(f"      ls it {it} a={alpha:.3g}: merit {merit(en, tn):.7f} vs {m0 - 1e-4 * alpha * max(pred, 0.0):.7f} "
                      f"(t {tn:.7f} smax {en['smax']:.7f} al {en['al']:+.5f} alc {en['al_ctrl']:+.8f} "
                      f"rest {en.get('al_ctrl_rest', np.nan):+.8f} pair {'-' if _pp is None else '%+.2e/%.1e' % (_pp['exc'], _pp['sep'])} "
                      f"res {(_rres if ev_fn.pair_mode else 0.0):.1e} viol {viol(en, tn):.2e})", flush=True)
            if merit(en, tn) <= m0 - 1e-4 * alpha * max(pred, 0.0):
                accepted = True; break
            alpha *= 0.5
        if DEBUG_LS and not accepted:
            _er = ev_fn(z)
            print(f"      ls it {it}: REJECTED; the current iterate re-evaluated: smax {_er['smax']:.7f} (recorded {e['smax']:.7f})", flush=True)
            for _i in np.argsort(-np.abs(_er['s'] - e['s']))[:6]:
                _k = round(float(ev_fn.W[_i]), 6)
                print(f"        row {_i} w={_k:+.4f}: recorded {e['s'][_i]:.7f} at {e['tops'].get(_k, _k):+.6f}, "
                      f"re-evaluated {_er['s'][_i]:.7f} at {_er['tops'].get(_k, _k):+.6f}", flush=True)
        _alpha_prev = alpha if accepted else 1.0
        hist.append(dict(it=it, t=t, smax=e['smax'], al=e['al'], stat=stat,
                         step=float(np.linalg.norm(alpha * d)), alpha=alpha,
                         Delta=Delta, accepted=accepted, nblk=len(e['blocks'])))
        if verbose:
            print(f"    sqp {it:3d}: t={t:.6f} smax={e['smax']:.6f} al={e['al']:+.4f} alc={e['al_ctrl']:+.5f} "
                  + (f"rest={e['al_ctrl_rest']:+.7f} exc={e['pair']['exc']:+.1e} " if e.get('pair') is not None else '') +
                  f"{'adm' if admissible(e) else 'NOT adm'} stat={stat:.2e} step={np.linalg.norm(alpha*d):.2e} a={alpha:g} "
                  f"D={Delta:.2e} nu={nu:.3g} {'ok' if accepted else 'REJ'}", flush=True)
        if not accepted:
            Delta *= 0.5
            if Delta < 1e-10:
                code = 6; break
            continue
        # damped BFGS on the Lagrangian
        subn, _ = subproblem(en, tn, H, Delta, eps, ctrl_max, kink_block,
                             disp=((zn - z_start, float(max_disp)) if max_disp is not None else None), pen=nu / 100.0)
        gLn = grad_L(en, subn) if subn is not None else gL
        sv = zn - z; yv = gLn - gL
        sHs = float(sv @ H @ sv); sy = float(sv @ yv)
        if sHs > 0:
            theta = 1.0 if sy >= 0.2 * sHs else 0.8 * sHs / (sHs - sy)
            r_ = theta * yv + (1 - theta) * (H @ sv)
            H = H - np.outer(H @ sv, H @ sv) / sHs + np.outer(r_, r_) / max(float(sv @ r_), 1e-16)
        z, t, e = zn, tn, en
        if _pair_ok and not ev_fn.pair_mode and not any(p_.get('event') == 'exit' for p_ in pair_log):
            # entry DURING the solve too (2026-09-26): the ne=100 pair closed from
            # 0.078 to 0.047 over rounds 12-15 and was let onto its face only at
            # the next round's start (no re-entry after a release in this solve)
            _zp = _try_enter(z, f'it {it}')
            if _zp is not None:
                ev_fn.pair_mode = True; z = _zp
                e = ev_fn(z); t = max(t, e['smax']); _alpha_prev = 1.0
        if PAIR_BALL and ev_fn.pair_mode and alpha < 1.0:
            # the line search found where the model holds: a radius of twice the
            # accepted step (2 alpha failed), so the next QP picks its direction
            # for the region it can use instead of a box corner cut by alpha
            Delta = min(Delta, max(2.0 * alpha * _tnorm(d), 1e-10))
        if oracle is not None and refine_tol is not None:
            # exchange refinement INSIDE the inner solve: the reduced model
            # knows where the new design's peaks are; add the ones the working
            # set misses, then re-evaluate so the rows include them.  The full
            # certification outside is unchanged.  (Gap it closes: inner t
            # 8.7-9.0 against certified 10.0-10.7 with one frequency per round.)
            mp = ev_fn.missed_peaks(z, t, refine_grid, refine_tol)[:refine_max]
            if mp:
                enrolled_f = [float(ev_fn.W[i]) for i in ev_fn.enroll]
                ev_fn.set_W(list(ev_fn.W) + [w for w, _ in mp], enrolled_f)
                if ev_fn.follow:
                    for w_, _ in mp:
                        ev_fn.follow[round(float(w_), 6)] = follow_radius
                W = ev_fn.W; m = len(W)
                e = ev_fn(z); t = max(t, e['smax'])
                added.extend(w for w, _ in mp)
                if verbose:
                    print(f"    sqp refine: +{len(mp)} freq {[round(w, 3) for w, _ in mp]} "
                          f"(reduced peaks {[round(v, 3) for _, v in mp]} > t {t:.4f})", flush=True)
        _note(e, t)
        if admissible(e) and e['smax'] < best['g']:
            best.update(g=e['smax'], z=z.copy(), al=e['al'], t=t, clr=e['clr'], w=_argw(e))
        if alpha == 1.0 and _tnorm(d) > 0.9 * Delta:
            Delta *= 2.0
        _v = max(e['smax'] - t, 0.0) + max(e['al'] + eps, 0.0)
        # stationarity RELATIVE to the gradient scale of the active rows: the
        # absolute 1e-6 never fired at a fixed point pinned by a four-way peak
        # tie and the compensator margin (stat 4e-2 against row gradients
        # ~1e3), so --converge spun on it.  Same spirit as stationarity.py
        # (rho_Clarke / branch-gradient scale, 6e-4 at a stationary kink).
        # the same scale as stationarity.py: the MEAN norm of the ACTIVE rows'
        # gradients (the largest row's norm, 7e3 at ne=200, made a threshold
        # of 7 and the SQP declared stationarity after one iteration)
        _act = [i for i in range(len(e['s'])) if e['s'][i] >= e['smax'] - 1e-3 * abs(e['smax'])]
        _gscale = max(float(np.mean([np.linalg.norm(e['Gs'][i]) for i in _act])) if _act else 1.0, 1e-12)
        # ... AND the QP at this iterate predicts no decrease worth a step
        # (STAGE6 6.14): the gradient scale is the MEAN norm of the active rows'
        # gradients, and one very sensitive row (the atom amplitudes, norms
        # 1e2-1e3; a far resonance, 1e6) made 1e-4 x that scale a loose test --
        # MIMO atoms runs left the SQP after 1-3 iterations with the model still
        # predicting a decrease.  The predicted decrease is in the objective's
        # own units, independent of the parametrisation.
        _pred_small = pred_rel is None or pred <= pred_rel * max(abs(t), 1.0)
        if (stat < tol or (stat < stat_rel * _gscale and _pred_small)) and _v < tol:
            if (ev_fn.pair_mode and _last_pair_mu is not None and e.get('pair') is not None
                    and e['pair']['c'][0] < -10 * PAIR_KAPPA and abs(_last_pair_mu[2]) > PAIR_EXIT_TOL):
                # ACTIVE-SET RULE (2026-09-25): S is stationary, the pair is strictly
                # inside the line and the equal-real-parts row carries a multiplier:
                # release it (inside, F allows unequal real parts) and continue.
                # Releasing at once instead (on the first QP) lost the face at a
                # near-double pole: ne=200 7.901824 -> 7.901729 in 62 evaluations,
                # against 7.878932 in 27 staying on S until stationary.
                pair_log.append(dict(event='exit', it=it, mu=[float(x) for x in _last_pair_mu]))
                if verbose:
                    print(f"    sqp: S stationary, pair released (multipliers {_last_pair_mu})", flush=True)
                ev_fn.pair_mode = False
                e = ev_fn(z); t = max(t, e['smax'])
                continue
            code = 0; exit_info = dict(stat=stat, gscale=_gscale, viol=_v, n_active=len(_act), pred=float(pred), t=float(t))
            if hasattr(pk, 'blocks') and _act:
                # where the active rows' gradient lives (STAGE6 6.14): the row of
                # largest gradient norm, by block of the design vector
                _i = max(_act, key=lambda i_: np.linalg.norm(e['Gs'][i_]))
                _g = np.asarray(e['Gs'][_i]); _gl = np.asarray(gL)
                exit_info.update(row_w=float(ev_fn.W[_i]) if _i < len(ev_fn.W) else None,
                                 row_blocks={nm_: round(float(np.linalg.norm(_g[ix])), 6) for nm_, ix in pk.blocks()},
                                 stat_blocks={nm_: round(float(np.linalg.norm(_gl[ix])), 6) for nm_, ix in pk.blocks()})
            break
    if oracle is not None and added and cands:
        # the working set grew during the solve: a candidate's recorded smax is
        # on the set of its time; re-evaluate all on the FINAL set (cheap on
        # the oracle) before choosing, so t handed back is honest (9.167
        # recorded against 9.667 on the final set, once)
        re = []
        for smax_c, z_c, al_c, t_c in cands:
            e_c = ev_fn(z_c)
            if admissible(e_c):
                re.append((e_c['smax'], z_c, e_c['al'], max(t_c, e_c['smax'])))
        cands = sorted(re, key=lambda c: c[0])
        if cands:
            best.update(g=cands[0][0], z=cands[0][1], al=cands[0][2], t=cands[0][3], clr=True)
        else:
            best.update(g=np.inf, z=None, al=None, t=np.inf, clr=None)
    if oracle is not None and best['z'] is not None and validate_full:
        # THE CERTIFICATE OUTSIDE: the reduced pencil's abscissa is a predictor
        # that goes stale after a large step (a design read as admissible was
        # certified at -0.0485, 1.5e-3 outside the margin, and refused twice at
        # the gate).  One full certificate per returned candidate, best first.
        n_cert = 0; chosen = None
        for smax_c, z_c, al_c, t_c in cands:
            arow_c, nvec_c, K_c, xa_c, xs_c, va_c, vs_c = pk.unpack(z_c)
            b_c, c_c, *_ = D.bc(xa_c, xs_c, va_c, vs_c)
            al_true, _, _, _, _, clr_true = D.alpha_certified(arow_c, nvec_c, K_c, b_c, c_c, eps=eps)
            n_cert += 1
            if verbose:
                print(f"    sqp candidate smax {smax_c:.6f} (reduced alpha {al_c:+.5f}): full alpha {al_true:+.5f}, "
                      f"line {'clear' if clr_true else 'NOT clear'} -> "
                      f"{'returned' if (clr_true and al_true + eps <= -mg) else 'refused'}", flush=True)
            if clr_true and al_true + eps <= -mg:
                chosen = (smax_c, z_c, al_true, t_c); break
        if chosen is None:
            best.update(g=np.inf, z=None, al=None, t=np.inf, clr=None)   # nothing certified: return the start
        else:
            best.update(g=chosen[0], z=chosen[1], al=chosen[2], t=chosen[3], clr=True)
        code_cert = n_cert
    else:
        code_cert = 0
    if best['z'] is not None and np.isfinite(best['g']) and t0_start is not None and best['g'] > t0_start * (1 + 1e-9):
        # never hand back a design worse than the start: the exchange would
        # adopt it (it did: 211 -> 1019, then froze on it)
        best.update(g=np.inf, z=None, al=None, t=np.inf, clr=None)
    def _tops_at(zb):
        ev_fn.w_star = {}; ev_fn(zb); return dict(ev_fn.w_star)
    if verbose and pen_adapt:
        print(f"    sqp: penalty nu = {nu:.3g} at the end; {n_soc[0]} compensator correction(s)", flush=True)
    stats = dict(iters=len(hist), fn_evals=ev_fn.ncalls, termination_code=code, exit_info=exit_info, certified_checks=code_cert,
                 nu_final=float(nu), ctrl_corrections=int(n_soc[0]),
                 stat_value=(hist[-1].get('stat') if hist else None),
                 wall=time.time() - t_start, inner='sqp', kink_block=kink_block,
                 ctrl_lmi=bool(ctrl_lmi), alpha_rows=ev_fn.n_alpha_rows, alpha_fallback=ev_fn.n_fallback,
                 trust_final=float(Delta), added_freqs=added, W_final=[float(w) for w in ev_fn.W], pair_log=pair_log, pair_mode_end=bool(ev_fn.pair_mode), z_last=[float(x) for x in z], e_last_al_ctrl=float(e["al_ctrl"]), e_last_al=float(e["al"]),
                 screen=(ev_fn.screen_stats[-1] if ev_fn.screen_stats else None),
                 refine_log=list(ev_fn.refine_log[-40:]), clblock_fail=int(ev_fn.n_clblock_fail),
                 sub_failures=list(sub_status_log),
                 w_star=(_tops_at(best['z']) if (ev_fn.follow and best['z'] is not None) else {}),
                 n_blocks=len(e['blocks']), dedup=int(sum(ev_fn.n_dedup)), history=hist)
    return best, ev_fn.ncalls, H, stats
