#!/usr/bin/env python3
"""mimo_proto.py -- 2 actuators / 2 sensors (algebra note item 7), a standalone
PROTOTYPE at GL size (2026-09-28, user: decentralized 2 x SISO and centralized
2x2 controllers at ne=100, "just to see if it works"; STAGE5 34).

    python tools/mimo_proto.py --mode decentral --out results_mimo/dec
    python tools/mimo_proto.py --mode central --start results_mimo/dec.npz --out results_mimo/cen

The design: two Gaussian actuators and two Gaussian sensors (width 0.4, as
K+B+C) at free positions, and a dynamic compensator of total order R,
    x_c' = A_c x_c + B_c y,   u = D_c y + C_c x_c,   y = C^H q   (C = M [psi_s1, psi_s2]),
    M q' = A q + B u + M f                                     (B = M [psi_a1, psi_a2]),
A_c complex tridiagonal (as the SISO code), B_c (R x 2), C_c (2 x R), D_c (2 x 2).
  * decentral: A_c = blockdiag(two tridiagonals of order R/2), B_c = blockdiag(ones, ones),
    C_c = blockdiag(n_1^T, n_2^T), D_c = diag(K_1, K_2): loop k reads sensor k and drives
    actuator k -- two SISO compensators of the SISO code's form;
  * central: every entry of the tridiagonal A_c, of B_c, C_c and D_c free (the decentral
    structure is the special case, so a central run started from a decentral optimum can
    only improve on it).
Objective: gamma = sup_w sigma_1 of the energy resolvent, as the SISO code:
sigma^2 = lambda_max(M T^H M T), T the closed-loop resolvent restricted to the
plant, applied through ONE sparse LU of i w M - A and a RANK-4 Woodbury
correction for the controller -- products with the sparse M only, never M^-1,
M^(+-1/2) or a factor of M.  Constraints: closed-loop and compensator spectral
abscissae <= -eps (eps = 0.05, as the SISO runs: stable loop, stable compensator).

Method (a prototype, not the production loop): epigraph over a working set
Omega of frequencies (sigma_1 and sigma_2 rows) plus the rightmost closed-loop
and compensator eigenvalues as rows, solved by SLSQP in a trust box of scaled
variables; each round the candidate is CERTIFIED by the dense level-set
algorithm on the even pencil [[A_cl, M/g^2], [M, A_cl^H]] - s diag(M_cl, -M_cl)
(all its eigenvalues, so every crossing of the imaginary axis at any w is found:
the value is global in w) and by the dense closed-loop spectrum; a candidate
that is not certified better is refused and the box shrinks.  The dense steps
are GL-size reference methods (as tools/tail_certificate.py); the production
loop would use its sparse certificate, reduced model and winding count."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eig as deig, eigvals as deigvals
from scipy.optimize import minimize
from fe_codesign import FECoDesign
import controller as ctl
import fieldbasis as FB

EPS = 0.05
STAB_TOL = 1e-6      # margin tolerance at acceptance (SLSQP's feasibility on the near-double compensator pair)


def _v0(n):
    rng = np.random.default_rng(12345)
    return rng.standard_normal(n) + 1j * rng.standard_normal(n)


class Mimo:
    """Parameters: complex (d, e, f, Bc, Cc, Dc) of the order-R compensator and
    real positions (xa1, xa2, xs1, xs2); a boolean FREE mask selects the
    structure (decentral / central)."""

    def __init__(self, D, R, mode, fields='gauss', na=4, smin=1.0, kmax=0.7071):
        self.D, self.R, self.mode = D, R, mode
        self.fields, self.na, self.smin, self.kmax = fields, na, smin, kmax
        self.npos = 4 if fields == 'gauss' else 4 * FB.NATOM_PAR * na
        self.n = D.n; self.x = np.asarray(D.xn, float)
        self.M = D.M.tocsc(); self.A = D.A.tocsc()
        r = R
        self.sizes = [('d', r), ('e', max(r - 1, 0)), ('f', max(r - 1, 0)), ('Bc', 2 * r), ('Cc', 2 * r), ('Dc', 4)]
        self.nc = sum(s for _, s in self.sizes)
        free = np.ones(self.nc, bool)
        if mode == 'decentral':
            h = r // 2; o = self._off
            if r > 1:
                free[o('e') + h - 1] = False; free[o('f') + h - 1] = False   # block boundary
            free[o('Bc'):o('Bc') + 2 * r] = False                             # B_c fixed gauge
            Cc_free = np.zeros((2, r), bool); Cc_free[0, :h] = True; Cc_free[1, h:] = True
            free[o('Cc'):o('Cc') + 2 * r] = Cc_free.ravel()
            free[o('Dc'):o('Dc') + 4] = np.array([True, False, False, True])
        self.free_c = free
        self.lu_cache = {}

    def _off(self, name):
        o = 0
        for nm, s in self.sizes:
            if nm == name:
                return o
            o += s
        raise KeyError(name)

    # ------------------------------------------------------------ packing
    def split(self, th):
        r = self.R; o = self._off
        d = th[o('d'):o('d') + r]; e = th[o('e'):o('e') + max(r - 1, 0)]; f = th[o('f'):o('f') + max(r - 1, 0)]
        Bc = th[o('Bc'):o('Bc') + 2 * r].reshape(r, 2)
        Cc = th[o('Cc'):o('Cc') + 2 * r].reshape(2, r)
        Dc = th[o('Dc'):o('Dc') + 4].reshape(2, 2)
        return d, e, f, Bc, Cc, Dc

    def to_real(self, th, pos):
        c = th[self.free_c]
        return np.concatenate([c.real, c.imag, pos])

    def from_real(self, u, th0):
        k = int(self.free_c.sum())
        th = th0.copy(); th[self.free_c] = u[:k] + 1j * u[k:2 * k]
        return th, u[2 * k:2 * k + self.npos].copy()

    # ------------------------------------------------------------ system
    def build(self, th, pos):
        D = self.D
        d, e, f, Bc, Cc, Dc = self.split(th)
        Ac = ctl.A_of(d, e, f)
        B = np.zeros((self.n, 2), complex); C = np.zeros((self.n, 2), complex)
        dB = np.zeros((self.n, 2), complex); dC = np.zeros((self.n, 2), complex)
        if self.fields == 'gauss':
            for i in range(2):
                b, c, *_ = D.bc(float(pos[i]), float(pos[2 + i]))
                B[:, i] = b; C[:, i] = c
                dB[:, i] = D.M @ D.dpsi_dx(float(pos[i])); dC[:, i] = D.M @ D.dpsi_dx(float(pos[2 + i]))
            return dict(Ac=Ac, Bc=Bc, Cc=Cc, Dc=Dc, B=B, C=C, dB=dB, dC=dC, d=d, e=e, f=f)
        # atoms: fields (actuator 1, actuator 2, sensor 1, sensor 2), each na Gabor atoms,
        # used at unit M-norm (the gauge), psi = v / ||v||_M
        m = FB.NATOM_PAR * self.na; raw, nu = [], []
        for k in range(4):
            a_, c_, rho, tau = FB.split_atoms(np.asarray(pos[k * m:(k + 1) * m], float), self.na)
            v = FB.atom_field(self.x, a_, c_, rho, tau, self.smin, self.kmax)
            nv = float(np.sqrt(np.real(np.vdot(v, D.M @ v)))); raw.append(v); nu.append(nv)
            (B if k < 2 else C)[:, k % 2] = D.M @ (v / nv)
        return dict(Ac=Ac, Bc=Bc, Cc=Cc, Dc=Dc, B=B, C=C, d=d, e=e, f=f, raw=raw, nu=nu, pos=np.asarray(pos, float))

    def _lu(self, w):
        key = float(w)
        if key not in self.lu_cache:
            if len(self.lu_cache) > 2000:
                self.lu_cache.clear()
            self.lu_cache[key] = spla.splu(((1j * w) * self.M - self.A).tocsc())
        return self.lu_cache[key]

    def _setup(self, w, S):
        """Woodbury data at w: S_cl = S0 - U V', U = [[B,0],[0,Bc]], V' = [[Dc C^H, Cc],[C^H, 0]]."""
        n, R = self.n, self.R
        lu = self._lu(w); SK = (1j * w) * np.eye(R) - S['Ac']
        SKlu = np.linalg.inv(SK)
        Y = np.zeros((n + R, 4), complex)
        Y[:n, :2] = lu.solve(S['B']); Y[n:, 2:] = SKlu @ S['Bc']
        G = np.eye(4, dtype=complex) - self._Vp(Y, S)
        Gi = np.linalg.inv(G)
        return dict(lu=lu, SKi=SKlu, Y=Y, Gi=Gi)

    def _Vp(self, X, S):
        """V' X for X (n+R) x m."""
        n = self.n
        CHx = S['C'].conj().T @ X[:n]
        return np.vstack([S['Dc'] @ CHx + S['Cc'] @ X[n:], CHx])

    def _UH(self, X, S):
        n = self.n
        return np.vstack([S['B'].conj().T @ X[:n], S['Bc'].conj().T @ X[n:]])

    def _S0(self, W, st, H=False):
        n = self.n
        top = st['lu'].solve(W[:n], trans='H' if H else 'N')
        bot = (st['SKi'].conj().T if H else st['SKi']) @ W[n:]
        return np.concatenate([top, bot])

    def solve(self, w_, st, S):
        t = self._S0(w_, st)
        return t + st['Y'] @ (st['Gi'] @ self._Vp(t[:, None], S)[:, 0])

    def solveH(self, w_, st, S):
        # S_cl^-H = S0^-H + S0^-H V'^H (I - U^H S0^-H V'^H)^-1 U^H S0^-H
        t = self._S0(w_, st, H=True)
        if 'Z' not in st:
            n, R = self.n, self.R
            VpH = np.zeros((n + R, 4), complex)
            VpH[:n, :2] = S['C'] @ S['Dc'].conj().T; VpH[n:, :2] = S['Cc'].conj().T; VpH[:n, 2:] = S['C']
            Z = np.column_stack([self._S0(VpH[:, j], st, H=True) for j in range(4)])
            st['Z'] = Z; st['GHi'] = np.linalg.inv(np.eye(4, dtype=complex) - self._UH(Z, S))
        return t + st['Z'] @ (st['GHi'] @ self._UH(t[:, None], S)[:, 0])

    # ------------------------------------------------------------ gains
    def sig(self, w, S, k=2, vectors=True):
        n, R = self.n, self.R
        st = self._setup(w, S)
        pad = np.zeros(R, complex)
        Tv = lambda v: self.solve(np.concatenate([v, pad]), st, S)[:n]
        THv = lambda v: self.solveH(np.concatenate([v, pad]), st, S)[:n]
        Wop = spla.LinearOperator((n, n), matvec=lambda v: self.M @ THv(self.M @ Tv(v)), dtype=complex)
        lam, V = spla.eigs(Wop, k=k, which='LM', tol=1e-10, maxiter=5000, v0=_v0(n))
        o = np.argsort(-lam.real); out = []
        for j in o:
            s = float(np.sqrt(max(lam[j].real, 0.0)))
            if not vectors:
                out.append(dict(sigma=s)); continue
            x = self.solve(np.concatenate([V[:, j], pad]), st, S)
            q = x[:n]; qMq = float(np.real(np.vdot(q, self.M @ q)))
            x = x * (s / np.sqrt(qMq))
            z = self.solveH(np.concatenate([self.M @ x[:n], pad]), st, S) / s
            out.append(dict(sigma=s, x=x, z=z))
        return out

    def grad(self, l, x, S):
        """Gradient of Re(l^H dA_cl x) w.r.t. the REAL free variables."""
        n, R = self.n, self.R
        xp, xc, lp, lc = x[:n], x[n:], l[:n], l[n:]
        lB = S['B'].conj().T @ lp                     # l_p^H B_i  -> conj(.) ; use vdot form below
        lB = np.array([np.vdot(lp, S['B'][:, i]) for i in range(2)])
        Cx = np.array([np.vdot(S['C'][:, j], xp) for j in range(2)])
        g = np.zeros(self.nc, complex); o = self._off
        g[o('d'):o('d') + R] = np.conj(lc) * xc
        if R > 1:
            g[o('e'):o('e') + R - 1] = np.conj(lc[:-1]) * xc[1:]
            g[o('f'):o('f') + R - 1] = np.conj(lc[1:]) * xc[:-1]
        g[o('Bc'):o('Bc') + 2 * R] = np.outer(np.conj(lc), Cx).ravel()
        g[o('Cc'):o('Cc') + 2 * R] = np.outer(lB, xc).ravel()
        g[o('Dc'):o('Dc') + 4] = np.outer(lB, Cx).ravel()
        gc = g[self.free_c]
        act = S['Dc'] @ Cx + S['Cc'] @ xc             # (D_c C^H x_p + C_c x_c)_i
        sen = lB @ S['Dc'] + np.conj(lc) @ S['Bc']      # (l_p^H B D_c + l_c^H B_c)_j
        if self.fields == 'gauss':
            gpos = np.array([np.real(np.vdot(lp, S['dB'][:, i]) * act[i]) for i in range(2)]
                            + [np.real(sen[j] * np.vdot(S['dC'][:, j], xp)) for j in range(2)])
            return np.concatenate([gc.real, -gc.imag, gpos])
        # atoms: G_psi (dF = Re(G^H dpsi)) for each unit field, through the
        # normalisation psi = v/nu to G_v, then the packer's atom chain rule
        Ml, Mx = self.M @ lp, self.M @ xp
        Gpsi = [np.conj(act[0]) * Ml, np.conj(act[1]) * Ml, sen[0] * Mx, sen[1] * Mx]
        m = FB.NATOM_PAR * self.na; out = []
        for k in range(4):
            v, nu = S['raw'][k], S['nu'][k]
            Gv = Gpsi[k] / nu - np.real(np.vdot(v, Gpsi[k])) * (self.M @ v) / nu ** 3
            p = S['pos'][k * m:(k + 1) * m]
            a_, c_, rho, tau = FB.split_atoms(p, self.na)
            out.append(FB.atom_grad(self.x, a_, c_, rho, tau, self.smin, self.kmax, Gv))
        return np.concatenate([gc.real, -gc.imag] + out)

    # ------------------------------------------------------------ spectra
    def Acl_dense(self, S):
        n, R = self.n, self.R
        Ad = self.A.toarray(); Md = self.M.toarray()
        Acl = np.zeros((n + R, n + R), complex); Mcl = np.zeros_like(Acl)
        Acl[:n, :n] = Ad + S['B'] @ S['Dc'] @ S['C'].conj().T
        Acl[:n, n:] = S['B'] @ S['Cc']; Acl[n:, :n] = S['Bc'] @ S['C'].conj().T; Acl[n:, n:] = S['Ac']
        Mcl[:n, :n] = Md; Mcl[n:, n:] = np.eye(R)
        return Acl, Mcl

    def eig_rows(self, S, m_cl=6, m_c=4, grads=True):
        """Rightmost m_cl closed-loop and m_c compensator eigenvalues: values (Re) and gradients."""
        n, R = self.n, self.R
        Acl, Mcl = self.Acl_dense(S)
        lam, VL, VR = deig(Acl, Mcl, left=True, right=True)
        fin = np.isfinite(lam); lam, VL, VR = lam[fin], VL[:, fin], VR[:, fin]
        o = np.argsort(-lam.real)[:m_cl]
        rows = []
        for j in o:
            u, v = VL[:, j], VR[:, j]; eta = np.vdot(u, Mcl @ v)
            rows.append((float(lam[j].real), self.grad(u / np.conj(eta), v, S) if grads else None, complex(lam[j])))
        crow = []
        if self.R == 0:
            return rows, crow
        mu, UL, UR = deig(S['Ac'], left=True, right=True)
        o = np.argsort(-mu.real)[:m_c]
        for j in o:
            u, v = UL[:, j], UR[:, j]; eta = np.vdot(u, v)
            l = np.concatenate([np.zeros(n, complex), u / np.conj(eta)]); x = np.concatenate([np.zeros(n, complex), v])
            crow.append((float(mu[j].real), self.grad(l, x, S) if grads else None, complex(mu[j])))
        return rows, crow

    # ------------------------------------------------------------ certificate
    def hinf(self, S, gam_lb, w_lb=np.nan, rtol=1e-7, itmax=40):
        """Dense level set on the even pencil: global in w (all eigenvalues)."""
        n, R = self.n, self.R
        Acl, Mcl = self.Acl_dense(S)
        MP = np.zeros_like(Mcl); MP[:n, :n] = Mcl[:n, :n]
        N = n + R
        E = np.zeros((2 * N, 2 * N), complex); E[:N, :N] = Mcl; E[N:, N:] = -Mcl
        gam, wst = float(gam_lb), float(w_lb)     # the lower bound's own frequency
        sfun = lambda w: self.sig(float(w), S, k=1, vectors=False)[0]['sigma']
        for it in range(itmax):
            gt = gam * (1 + rtol)
            H = np.zeros((2 * N, 2 * N), complex)
            H[:N, :N] = Acl; H[:N, N:] = MP / gt ** 2; H[N:, :N] = MP; H[N:, N:] = Acl.conj().T
            ev = deigvals(H, E); ev = ev[np.isfinite(ev)]
            sc = np.maximum(1.0, np.abs(ev))
            wc = np.sort(np.unique(np.round(ev[np.abs(ev.real) < 1e-7 * sc].imag, 11)))
            if wc.size == 0:
                break
            cand = np.concatenate([wc, 0.5 * (wc[:-1] + wc[1:])]) if wc.size > 1 else wc
            vals = np.array([sfun(w) for w in cand]); m = int(np.argmax(vals))
            if vals[m] <= gam * (1 + 1e-12):
                break
            gam, wst = float(vals[m]), float(cand[m])
        else:
            return gam, wst, False
        if np.isfinite(wst):                    # polish to the local top
            from scipy.optimize import minimize_scalar
            rr = minimize_scalar(lambda w: -sfun(w), bounds=(wst - 0.05, wst + 0.05), method='bounded',
                                 options=dict(xatol=1e-9))
            if -rr.fun > gam:
                gam, wst = float(-rr.fun), float(rr.x)
        return gam, wst, True


def scan(mm, S, ws):
    return np.array([mm.sig(float(w), S, k=1, vectors=False)[0]['sigma'] for w in ws])


def peaks_of(ws, s, rel=0.9):
    i = [j for j in range(1, len(s) - 1) if s[j] >= s[j - 1] and s[j] >= s[j + 1] and s[j] >= rel * s.max()]
    return [float(ws[j]) for j in i]


def run(a):
    D = FECoDesign(ne=a.ne, r=8)
    t0 = time.time()
    fields, na = a.fields, a.na
    if a.start:
        z = np.load(a.start, allow_pickle=True)
        th0 = np.asarray(z['theta'], complex); pos = np.asarray(z['pos'], float); R = int(z['R'])
        fields = str(z['fields']) if 'fields' in z.files else 'gauss'; na = int(z['na']) if 'na' in z.files else na
    elif fields == 'gauss':
        # from the SISO K+B+C optimum: loop 1 = that design, loop 2 switched off
        # (n_2 = 0, K_2 = 0) with a well-damped compensator and its pair at --pair2 (default -6, +6)
        z = np.load(a.siso, allow_pickle=True)
        r1 = len(z['d']); R = 2 * r1
        mm0 = Mimo(D, R, 'central')
        th0 = np.zeros(mm0.nc, complex); o = mm0._off
        d2 = -1.0 + 1j * np.linspace(-3, 3, r1)
        th0[o('d'):o('d') + R] = np.concatenate([z['d'], d2])
        th0[o('e'):o('e') + r1 - 1] = z['e']; th0[o('f'):o('f') + r1 - 1] = z['f']
        Bc = np.zeros((R, 2), complex); Bc[:r1, 0] = D.mc; Bc[r1:, 1] = D.mc
        th0[o('Bc'):o('Bc') + 2 * R] = Bc.ravel()
        Cc = np.zeros((2, R), complex); Cc[0, :r1] = z['nvec']
        th0[o('Cc'):o('Cc') + 2 * R] = Cc.ravel()
        th0[o('Dc'):o('Dc') + 4] = np.array([complex(z['K']), 0, 0, 0])
        pos = np.array([float(z['xa']), a.pair2[0], float(z['xs']), a.pair2[1]])
    else:
        # from the SISO K+FB+FC atoms optimum (a static gain, as there): loop 1 = that
        # design (atoms are functions of x: exact on any mesh), loop 2 switched off
        # (K_2 = 0) with copies of the loop-1 fields shifted by -4 (actuator) / +4 (sensor)
        z = np.load(a.siso_atoms, allow_pickle=True)
        R = 0; na = len(np.asarray(z['atoms_a_p'])) // FB.NATOM_PAR
        th0 = np.array([complex(z['K']), 0, 0, 0], complex)
        pa = np.asarray(z['atoms_a_p'], float); ps = np.asarray(z['atoms_s_p'], float)
        pa2 = pa.copy(); pa2[2 * na:3 * na] -= 4.0
        ps2 = ps.copy(); ps2[2 * na:3 * na] += 4.0
        pos = np.concatenate([pa, pa2, ps, ps2])
    mm = Mimo(D, R, a.mode, fields=fields, na=na)
    S = mm.build(th0, pos)
    ws_scan = np.unique(np.concatenate([np.arange(-40, -12, 0.25), np.arange(-12, 16, 0.05), np.arange(16, 40.01, 0.25)]))
    s_scan = scan(mm, S, ws_scan)
    g, wst, ok = mm.hinf(S, s_scan.max() * (1 - 1e-9), ws_scan[int(np.argmax(s_scan))])
    rows, crow = mm.eig_rows(S, grads=False)
    print(f"[{a.mode}] start: certified gamma {g:.6f} at w={wst:+.4f} ({'ok' if ok else 'NOT certified'}); "
          f"closed-loop abscissa {rows[0][0]:+.4f}, compensator {(crow[0][0] if crow else float('nan')):+.4f}; order {R}, "
          f"{int(mm.free_c.sum())} free complex controller parameters + {mm.npos} field parameters ({fields})", flush=True)
    if a.fdcheck:
        u0 = mm.to_real(th0, pos); rng = np.random.default_rng(1); dv = rng.standard_normal(u0.size) * np.maximum(np.abs(u0), 1) * 1e-7
        for w in (wst, 0.3):
            r0 = mm.sig(w, S, k=1)[0]; gg = mm.grad(r0['z'], r0['x'], S)
            thp, pp = mm.from_real(u0 + dv, th0); thm, pm = mm.from_real(u0 - dv, th0)
            sp_ = mm.sig(w, mm.build(thp, pp), k=1, vectors=False)[0]['sigma']; sm_ = mm.sig(w, mm.build(thm, pm), k=1, vectors=False)[0]['sigma']
            print(f"  FD check sigma_1 at w={w:+.3f}: FD {(sp_ - sm_) / 2:.6e} analytic {gg @ dv:.6e}", flush=True)
        (lr, lg, _), = mm.eig_rows(S, m_cl=1, m_c=0)[0][:1]
        thp, pp = mm.from_real(u0 + dv, th0); thm, pm = mm.from_real(u0 - dv, th0)
        lp_ = mm.eig_rows(mm.build(thp, pp), 1, 1, False)[0][0][0]; lm_ = mm.eig_rows(mm.build(thm, pm), 1, 1, False)[0][0][0]
        print(f"  FD check abscissa: FD {(lp_ - lm_) / 2:.6e} analytic {lg @ dv:.6e}", flush=True)
    Omega = sorted(set(np.round(peaks_of(ws_scan, s_scan, 0.8) + ([wst] if np.isfinite(wst) else []) +
                                list(ws_scan[(s_scan >= 0.85 * g) & (np.abs(ws_scan) <= 16)][::4]), 6)))
    best = dict(g=g, th=th0.copy(), pos=pos.copy(), w=wst)
    delta = a.delta; hist = [dict(round=-1, gamma=g, t=0.0)]
    for it in range(a.rounds):
        th_c, pos_c = best['th'], best['pos']; S = mm.build(th_c, pos_c)
        u0 = mm.to_real(th_c, pos_c)
        # scaling: a unit step in each scaled variable changes the rows by ~ gamma
        # (from the gradients at the current design), capped by the variable's size
        grads = []
        for w in Omega:
            for rr_ in mm.sig(w, S, k=1):
                grads.append(mm.grad(rr_['z'], rr_['x'], S))
        gmax = np.max(np.abs(np.array(grads)), axis=0)
        sc = np.minimum(best['g'] / np.maximum(gmax, 1e-300), 10 * np.maximum(np.abs(u0), 1.0))
        sc[-mm.npos:] = np.minimum(sc[-mm.npos:], 1.0)
        cache = {}

        def evalc(y):
            key = y.tobytes()
            if key in cache:
                return cache[key]
            u = u0 + sc * y[:-1]; t = y[-1] * best['g']
            th, ps = mm.from_real(u, th_c); Sx = mm.build(th, ps)
            vals, jac = [], []
            for w in Omega:
                for rr_ in mm.sig(w, Sx, k=2):
                    gg = mm.grad(rr_['z'], rr_['x'], Sx) * sc
                    vals.append((t - rr_['sigma']) / best['g']); jac.append(np.concatenate([-gg / best['g'], [1.0]]))
            rows, crow = mm.eig_rows(Sx)
            for val, gg, _ in rows + crow:
                vals.append((-EPS - val) / EPS); jac.append(np.concatenate([-(gg * sc) / EPS, [0.0]]))
            cache[key] = (np.array(vals), np.array(jac))
            return cache[key]

        nv = u0.size
        y0 = np.concatenate([np.zeros(nv), [1.0]])
        bnds = [(-delta, delta)] * nv + [(0.0, 2.0)]
        res = minimize(lambda y: (y[-1], np.concatenate([np.zeros(nv), [1.0]])), y0, jac=True, method='SLSQP',
                       bounds=bnds, constraints=[dict(type='ineq', fun=lambda y: evalc(y)[0], jac=lambda y: evalc(y)[1])],
                       options=dict(maxiter=a.inner, ftol=1e-10))
        u = u0 + sc * res.x[:-1]; th, ps = mm.from_real(u, th_c); Sx = mm.build(th, ps)
        rows, crow = mm.eig_rows(Sx, grads=False)
        stab = rows[0][0] <= -EPS + STAB_TOL and (not crow or crow[0][0] <= -EPS + STAB_TOL)
        s2 = scan(mm, Sx, ws_scan)
        gc, wc, okc = mm.hinf(Sx, s2.max() * (1 - 1e-9), ws_scan[int(np.argmax(s2))])
        acc = bool(stab and okc and gc < best['g'])
        print(f"  round {it:2d}: |Omega|={len(Omega):3d} t={res.x[-1] * best['g']:.6f} certified={gc:.6f} at w={wc:+.3f} "
              f"alpha_cl={rows[0][0]:+.4f} alpha_c={(crow[0][0] if crow else float('nan')):+.4f} box={delta:.3g} -> "
              f"{'ACCEPT' if acc else 'refuse'}  ({time.time() - t0:.0f} s)", flush=True)
        newp = peaks_of(ws_scan, s2, 0.9) + ([wc] if np.isfinite(wc) else [])
        Omega = sorted(set(np.round(list(Omega) + newp, 6)))
        if acc:
            gain = (best['g'] - gc) / best['g']
            best = dict(g=gc, th=th, pos=ps, w=wc); delta = min(2 * delta, a.delta_max) if gain > 1e-3 else delta
        else:
            delta /= 2
        hist.append(dict(round=it, gamma=best['g'], t=time.time() - t0, accepted=acc, candidate=gc, box=delta))
        if delta < 1e-6:
            print("  box collapsed: stop", flush=True); break
        g4 = [h['gamma'] for h in hist[-9:]]
        if len(hist) > 10 and (g4[0] - g4[-1]) / g4[0] < a.stop:
            print(f"  progress stop: {(g4[0] - g4[-1]) / g4[0] * 100:.3f} % over the last 8 rounds", flush=True); break
    S = mm.build(best['th'], best['pos'])
    rows, crow = mm.eig_rows(S, grads=False)
    d, e, f, Bc, Cc, Dc = mm.split(best['th'])
    ac = crow[0][0] if crow else float('nan')
    out = dict(theta=best['th'], pos=best['pos'], R=R, mode=a.mode, gamma=best['g'], w_star=best['w'],
               alpha_cl=rows[0][0], alpha_c=ac, ne=a.ne, fields=fields, na=na)
    _o.makedirs(_o.path.dirname(a.out) or '.', exist_ok=True)
    np.savez(a.out + '.npz', **out)
    js = dict(mode=a.mode, fields=fields, ne=a.ne, R=R, gamma=best['g'], w_star=best['w'], alpha_cl=rows[0][0], alpha_c=ac,
              positions=(dict(xa1=best['pos'][0], xa2=best['pos'][1], xs1=best['pos'][2], xs2=best['pos'][3]) if fields == 'gauss'
                         else None),
              Dc=[[str(v) for v in row] for row in Dc], seconds=time.time() - t0, rounds=len(hist) - 1, history=hist,
              start=a.start or a.siso)
    json.dump(js, open(a.out + '.json', 'w'), indent=1, default=float)
    print(f"[{a.mode}/{fields}] final: certified gamma {best['g']:.6f} (global in w); alpha_cl {rows[0][0]:+.4f}, "
          f"alpha_c {ac:+.4f}; Dc = {np.round(Dc, 3).tolist()}; {time.time() - t0:.0f} s -> {a.out}.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['decentral', 'central'], required=True)
    ap.add_argument('--ne', type=int, default=100)
    ap.add_argument('--siso', default='results_n100_kbc/K_B_C_s0_ne100.npz')
    ap.add_argument('--fields', choices=['gauss', 'atoms'], default='gauss',
                    help='gauss: Gaussians (width 0.4) at free positions + dynamic compensator (K+B+C); '
                         'atoms: Gabor-atom fields + static gain (K+FB+FC)')
    ap.add_argument('--na', type=int, default=4)
    ap.add_argument('--pair2', type=lambda t: [float(v) for v in t.split(',')], default=[-6.0, 6.0],
                    help='gauss start: initial (actuator, sensor) positions of pair 2')
    ap.add_argument('--siso-atoms', dest='siso_atoms', default='results_fb_atoms_n200_long/K_FB_FC_s0_ne200.npz')
    ap.add_argument('--start', default=None, help='a mimo_proto record to start from (e.g. the decentral optimum)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=40)
    ap.add_argument('--inner', type=int, default=40)
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--delta-max', dest='delta_max', type=float, default=0.5)
    ap.add_argument('--stop', type=float, default=0.0005)
    ap.add_argument('--fdcheck', action='store_true')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
