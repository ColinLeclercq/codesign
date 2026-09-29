"""
closedloop_mimo.py -- the closed-loop energy resolvent with m actuators and
p sensors (2026-09-29; algebra note item 7, STAGE6).

Same algebra and API as closedloop_sparse.ClosedLoopResolvent, which is kept
untouched for the SISO case (m = p = 1) so the SISO path stays bit-identical;
this class is used whenever the load/sensor vectors are matrices.

------------------------------------------------------------------- the system
plant      M q' = A q + B u ,        y = C^H q          (B = M Psi_a  n x m, C = M Psi_s  n x p)
controller x'   = A_c x + B_c y ,    u = D_c y + C_c x  (A_c r x r tridiagonal, B_c r x p,
                                                          C_c m x r, D_c m x p)
closed loop    A_cl = [[A + B D_c C^H, B C_c], [B_c C^H, A_c]],   M_cl = diag(M, I_r)

Argument conventions (the SISO names, so the driver's plumbing carries both):
    arow = (d, e, f, B_c)      -- the compensator's state part (A_c and its input matrix)
    nvec = C_c  (m x r)        -- its output matrix
    K    = D_c  (m x p)        -- its feedthrough
SISO is the special case B_c = mc * ones(r, 1), C_c = nvec[None, :], D_c = [[K]].

    S_cl(w) = i w M_cl - A_cl = S_0 + U V,     S_0 = diag(i w M - A, i w I - A_c)
    U = [[B, 0], [0, B_c]]                      (n + r) x (m + p)
    V = [[-D_c C^H, -C_c], [-C^H, 0]]           (m + p) x (n + r)
Woodbury with an (m + p) x (m + p) capacitance G = I + V S_0^-1 U; ONE sparse
LU of i w M - A per frequency; M only as a sparse product (sigma^2 =
lambda_max(M T^H M T), as the SISO class).

Gradient of sigma (or of an eigenvalue, with (x, z) -> (v, u / conj(u^H M_cl v)))
w.r.t. every controller entry: d sigma = Re(z^H dA_cl x):
    D_c[i,j] : (z_p^H B_i)(C_j^H x_p)      C_c[i,k] : (z_p^H B_i) x_c[k]
    B_c[k,j] : conj(z_c[k]) (C_j^H x_p)    A_c      : conj(z_c[k]) x_c[l] at the tridiagonal entries
"""
from __future__ import annotations
import numpy as np
import scipy.sparse.linalg as spla
from closedloop_sparse import ClosedLoopResolvent, _safe_solve


def as_ports(b, c):
    """(B, C) as 2-D arrays (n x m, n x p)."""
    B = np.asarray(b, complex); C = np.asarray(c, complex)
    return (B[:, None] if B.ndim == 1 else B), (C[:, None] if C.ndim == 1 else C)


def is_mimo(b, c=None):
    return np.ndim(b) == 2 or (c is not None and np.ndim(c) == 2)


class ClosedLoopResolventMIMO(ClosedLoopResolvent):
    def __init__(self, M, A, B, C, r, mc=1.0):
        super().__init__(M, A, np.zeros(M.shape[0]), np.zeros(M.shape[0]), r, mc)
        self.b, self.c = as_ports(B, C)           # n x m, n x p
        self.m, self.p = self.b.shape[1], self.c.shape[1]

    # ---------------- controller matrices ----------------
    def L_of_a(self, arow):
        return super().L_of_a(tuple(arow[:3]))

    def Bc_of(self, arow):
        if len(arow) > 3 and arow[3] is not None:
            return np.asarray(arow[3], complex).reshape(self.r, self.p)
        return np.full((self.r, self.p), self.mc, complex)

    def B_of(self):
        raise NotImplementedError("MIMO: B_c is part of arow (use Bc_of(arow))")

    # ---------------- per-frequency setup ----------------
    def setup(self, omega, arow, nvec, K):
        n, r, m, p = self.n, self.r, self.m, self.p
        cache = getattr(self, 'lu_cache', None)
        key = complex(omega)
        if cache is not None and key in cache:
            lu = cache[key]
        else:
            lu = spla.splu(((1j * omega) * self.M - self.A).tocsc())
            self.nfac += 1
            cmax = getattr(self, 'lu_cache_max', 8)
            if cache is not None and cmax > 0:
                while len(cache) >= cmax:
                    cache.pop(next(iter(cache)))
                cache[key] = lu
        Bc = self.Bc_of(arow)
        Cc = np.asarray(nvec, complex).reshape(m, r)
        Dc = np.asarray(K, complex).reshape(m, p)
        SK = (1j * omega) * np.eye(r) - self.L_of_a(arow)
        SKi = np.linalg.inv(SK) if r else np.zeros((0, 0), complex)
        Y = np.zeros((n + r, m + p), complex)
        Y[:n, :m] = lu.solve(self.b) if m > 1 else lu.solve(self.b[:, 0])[:, None]
        Y[n:, m:] = SKi @ Bc
        # Z = S_0^-H V^H,  V^H = [[-C D_c^H, -C], [-C_c^H, 0]]
        VH = np.zeros((n + r, m + p), complex)
        VH[:n, :m] = -self.c @ Dc.conj().T; VH[n:, :m] = -Cc.conj().T; VH[:n, m:] = -self.c
        Z = np.zeros_like(VH)
        Z[:n] = lu.solve(VH[:n], trans='H')
        Z[n:] = SKi.conj().T @ VH[n:]
        st = dict(lu=lu, SK=SK, SKi=SKi, Y=Y, Z=Z, Bc=Bc, Cc=Cc, Dc=Dc, omega=omega)
        self._st = st
        st['G'] = np.eye(m + p, dtype=complex) + self._Vmat(Y)
        return st

    def _S0(self, w, adjoint=False):
        n, st = self.n, self._st
        if adjoint:
            return np.concatenate([st['lu'].solve(w[:n], trans='H'), st['SKi'].conj().T @ w[n:]])
        return np.concatenate([st['lu'].solve(w[:n]), st['SKi'] @ w[n:]])

    def _Vmat(self, X):
        """V X for X (n+r) x k  ->  (m+p) x k."""
        n, st = self.n, self._st
        CHx = self.c.conj().T @ X[:n]
        return np.vstack([-(st['Dc'] @ CHx) - st['Cc'] @ X[n:], -CHx])

    def _Vapply(self, w):
        return self._Vmat(w[:, None])[:, 0]

    def _UHapply(self, w):
        n, st = self.n, self._st
        return np.concatenate([self.b.conj().T @ w[:n], st['Bc'].conj().T @ w[n:]])

    def solve(self, w):
        st = self._st
        t = self._S0(w)
        return t - st['Y'] @ _safe_solve(st['G'], self._Vapply(t))

    def solveH(self, w):
        st = self._st
        t = self._S0(w, adjoint=True)
        return t - st['Z'] @ _safe_solve(st['G'].conj().T, self._UHapply(t))

    def Acl_apply(self, x, arow, nvec, K):
        n, r, m, p = self.n, self.r, self.m, self.p
        xp, xc = x[:n], x[n:]
        Cc = np.asarray(nvec, complex).reshape(m, r); Dc = np.asarray(K, complex).reshape(m, p)
        CHx = self.c.conj().T @ xp
        top = self.A @ xp + self.b @ (Dc @ CHx + Cc @ xc)
        bot = self.L_of_a(arow) @ xc + self.Bc_of(arow) @ CHx
        return np.concatenate([top, bot])

    def default_shifts(self, arow, extra=None):
        return super().default_shifts(tuple(arow[:3]), extra)

    # ---------------- analytic design gradients ----------------
    def design_grad_tri(self, res, arow=None, nvec=None, K=None):
        """Complex gradients g (d sigma = Re(g . d theta)) for the tridiagonal
        (d, e, f), B_c (r x p), C_c (m x r) and D_c (m x p)."""
        n, r = self.n, self.r
        x, z = res['x'], res['z']
        xp, xc, zp, zc = x[:n], x[n:], z[:n], z[n:]
        zB = self.b.conj().T @ zp                    # conj(z_p^H B_i)
        zB = np.conj(zB)                             # z_p^H B_i
        Cx = self.c.conj().T @ xp                    # C_j^H x_p
        out = dict(d=np.conj(zc) * xc,
                   e=np.conj(zc[:-1]) * xc[1:] if r > 1 else np.zeros(0, complex),
                   f=np.conj(zc[1:]) * xc[:-1] if r > 1 else np.zeros(0, complex),
                   Bc=np.outer(np.conj(zc), Cx),
                   n=np.outer(zB, xc),
                   K=np.outer(zB, Cx))
        return out

    def port_sides(self, res, arow, nvec, K):
        """The actuator-side and sensor-side vectors of the gradient w.r.t. the
        load vectors: d sigma = Re(sum_i (z_p^H dB_i) act_i + sum_j sen_j (dC_j^H x_p)),
            act = D_c C^H x_p + C_c x_c   (m),    sen = z_p^H B D_c + z_c^H B_c   (p)."""
        n, m, p, r = self.n, self.m, self.p, self.r
        x, z = res['x'], res['z']
        xp, xc, zp, zc = x[:n], x[n:], z[:n], z[n:]
        Cc = np.asarray(nvec, complex).reshape(m, r); Dc = np.asarray(K, complex).reshape(m, p)
        zB = np.conj(self.b.conj().T @ zp)
        act = Dc @ (self.c.conj().T @ xp) + Cc @ xc
        sen = zB @ Dc + np.conj(zc) @ self.Bc_of(arow)
        return act, sen

    def dense_reference(self, omega, arow, nvec, K):
        raise NotImplementedError("dense reference: SISO class only")
