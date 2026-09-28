"""
fe_pipeline.py
==============
Full closed-loop chain on the FINITE-ELEMENT discretisation, for
mesh-convergence validation (prerequisite for the cavity).

    weak form :  M q_dot = A q + b(xa) u ,   y = c(xs)^H q ,  c = M s(xs)
    energy    :  ||q||_M ;  disturbance enters as the load vector M f

Design-time (dense, unavoidable for LQG): whiten with M = Lc Lc^H,
    At = Lc^{-1} A Lc^{-H},  B2 = Lc^{-1} b,  C2 = c^H Lc^{-H}
so the disturbance covariance and the state weight are both identity in the
whitened coordinates (mass-consistent).  LQG -> balanced truncation -> complex
controllable canonical form, exactly as on the Hermite grid.

Run-time (sparse): the closed-loop H-infinity norm is evaluated with the
augmented-Woodbury kernel `ClosedLoopResolvent`, which never factorises M.

usage:  python fe_pipeline.py [ne1 ne2 ...]
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.linalg import (cholesky, solve_triangular, solve_continuous_are,
                          solve_lyapunov, svd, eigh, eig)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fe_gl import assemble_gl, load_vector, SIGMA
from closedloop_sparse import ClosedLoopResolvent

# LQG design weights (were imported from the Hermite module; inlined so the
# cluster package is self-contained)
BETA = 7.0        # state-cost weight  Q = BETA^2 * I
VHALF = 2e-4      # sensor-noise std   V = VHALF^2

R_ORDER = 4


def gauss_nodal(xn, x0):
    return np.exp(-(xn - x0) ** 2 / (2 * SIGMA ** 2))


class FEPlant:
    """FE plant + design-time whitened realisation."""

    def __init__(self, ne=200, L=56.06, mu0=0.38, xa=-1.03, xs=0.98):
        A, M, xn, free = assemble_gl(ne=ne, L=L, mu0=mu0)
        self.A, self.M, self.xn, self.ne, self.L = A, M, xn, ne, L
        self.n = A.shape[0]
        # actuator LOAD vector (consistent) and sensor vector c = M s
        self.b = load_vector(xn, xa, ne, L).astype(complex)
        s = gauss_nodal(xn, xs).astype(complex)
        self.c = (M @ s)
        self.xa, self.xs = xa, xs

    # ---------- design-time whitened realisation (dense) ----------
    def whiten(self):
        Md = np.asarray(self.M.todense()).real
        Lc = cholesky(Md, lower=True)
        Ad = np.asarray(self.A.todense())
        # At = Lc^{-1} A Lc^{-H}
        T1 = solve_triangular(Lc, Ad, lower=True)
        At = solve_triangular(Lc, T1.conj().T, lower=True).conj().T
        B2 = solve_triangular(Lc, self.b, lower=True).reshape(-1, 1)
        C2 = solve_triangular(Lc, self.c, lower=True).conj().T.reshape(1, -1)
        return At, B2, C2, Lc

    def lqg_bt_canonical(self, r=R_ORDER, beta=BETA, vhalf=VHALF):
        At, B2, C2, Lc = self.whiten()
        n = self.n
        Q = beta ** 2 * np.eye(n); Rc = np.array([[1.0]])
        V = np.array([[vhalf ** 2]])
        X = solve_continuous_are(At, B2, Q, Rc)
        Y = solve_continuous_are(At.conj().T, C2.conj().T, np.eye(n), V)
        F = (B2.conj().T @ X); L_ = (Y @ C2.conj().T) / V[0, 0]
        AK = At - B2 @ F - L_ @ C2
        BK = L_; CK = -F
        # ---- balanced truncation (square-root, robust) ----
        Pc = solve_lyapunov(AK, -(BK @ BK.conj().T))
        Qc = solve_lyapunov(AK.conj().T, -(CK.conj().T @ CK))
        def sqrt_psd(G):
            G = (G + G.conj().T) / 2
            d, Vv = eigh(G); d = np.clip(d, 0, None)
            return Vv @ np.diag(np.sqrt(d))
        Lp, Lq = sqrt_psd(Pc), sqrt_psd(Qc)
        U, S, Vh = svd(Lp.conj().T @ Lq); Vv = Vh.conj().T
        si = np.diag(1.0 / np.sqrt(S[:r]))
        W1 = Lp @ U[:, :r] @ si; W2 = Lq @ Vv[:, :r] @ si
        AKr = W2.conj().T @ AK @ W1
        BKr = W2.conj().T @ BK
        CKr = CK @ W1
        # ---- complex controllable canonical form ----
        den = np.poly(AKr); a = den[1:]
        sk = 0.7 + 1j * np.linspace(0.5, 2.5, r)
        Kv = np.array([(CKr @ np.linalg.solve(s_ * np.eye(r) - AKr, BKr))[0, 0]
                       for s_ in sk])
        ncoef = np.linalg.solve(np.vander(sk, r), Kv * np.polyval(den, sk))
        arow = a[::-1].copy()            # L(a) last row = -arow
        nvec = ncoef[::-1].copy()
        return dict(arow=arow, nvec=nvec, S=S, AK=AK, BK=BK, CK=CK,
                    AKr=AKr, BKr=BKr, CKr=CKr, At=At, B2=B2, C2=C2)


def closed_loop_gamma(P: FEPlant, arow, nvec, K=0j, seeds=None):
    """Sparse augmented-Woodbury closed-loop H-infinity norm."""
    cl = ClosedLoopResolvent(P.M, P.A, P.b, P.c, r=len(arow), mc=1.0)
    if seeds is None:
        seeds = (-0.646, -0.54, -0.30, -0.9, -1.9, 2.5)
    best = cl.hinf(arow, nvec, K, seeds=seeds)
    return best['sigma'], best['omega'], cl


def open_loop_gamma(P: FEPlant):
    cl = ClosedLoopResolvent(P.M, P.A, P.b, P.c, r=1, mc=1.0)
    z = np.zeros(1, complex)
    best = cl.hinf(np.array([1.0 + 0j]), z, 0j, seeds=(-0.646,))
    return best['sigma'], best['omega']


if __name__ == "__main__":
    nes = [int(v) for v in sys.argv[1:]] or [150, 200, 300]
    print(f"{'ne':>5s} {'dofs':>6s} {'h_node':>7s} {'n/2sig':>7s} "
          f"{'gam_open':>9s} {'H2_full':>8s} {'H2_BT':>8s} {'gam_cl':>9s} "
          f"{'omega*':>10s} {'t(s)':>6s}")
    for ne in nes:
        t0 = time.time()
        P = FEPlant(ne=ne)
        hnode = float(np.diff(P.xn).min())
        nres = int((np.abs(P.xn - P.xa) <= 2 * SIGMA).sum())
        go, wo = open_loop_gamma(P)
        d = P.lqg_bt_canonical()
        At, B2, C2 = d['At'], d['B2'], d['C2']

        def h2(AK, BK, CK):
            n = At.shape[0]; r = AK.shape[0]
            Acl = np.block([[At, B2 @ CK], [BK @ C2, AK]])
            Bcl = np.block([[np.eye(n), np.zeros((n, 1))],
                            [np.zeros((r, n)), BK * VHALF]])
            Ccl = np.block([[BETA * np.eye(n), np.zeros((n, r))],
                            [np.zeros((1, n)), CK]])
            if np.max(eig(Acl, right=False).real) >= 0:
                return np.inf
            Pl = solve_lyapunov(Acl, -(Bcl @ Bcl.conj().T))
            return float(np.sqrt(np.trace(Ccl @ Pl @ Ccl.conj().T).real))

        h2_full = h2(d['AK'], d['BK'], d['CK'])
        h2_red = h2(d['AKr'], d['BKr'], d['CKr'])
        gc, wc, cl = closed_loop_gamma(P, d['arow'], d['nvec'])
        print(f"{ne:5d} {P.n:6d} {hnode:7.3f} {nres:7d} {go:9.3f} "
              f"{h2_full:8.3f} {h2_red:8.3f} {gc:9.4f} {wc:+10.5f} "
              f"{time.time()-t0:6.1f}")
