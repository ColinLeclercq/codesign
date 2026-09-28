"""
fe_gl.py
========
Full-order *finite-element* discretisation of the linearised complex
Ginzburg-Landau operator, producing SPARSE (A, M) in descriptor form

        M q_dot = A q,

with continuous P2 elements and homogeneous Dirichlet BCs, following the
benchmark used in Clement's report (400 cells, P2, [-56.06, 56.06], 799
free complex DOFs) and the physics of Chen & Rowley (2011).

This is a stand-in for a dolfinx assembly: the ONLY thing dolfinx would
change is `assemble_gl` (it would return the same sparse A, M, and the same
load vectors b(xa), c(xs)).  Everything downstream is plain sparse/dense
linear algebra and is identical whether A,M come from here or from dolfinx.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from numpy.polynomial.legendre import leggauss

# ---- physical parameters (subcritical / globally stable) ----
U, cu, cd, mu2 = 2.0, 0.2, -1.0, -0.01
SIGMA = 0.4
NU = U + 2j * cu          # 2 + 0.4i
GAMMA = 1.0 + 1j * cd     # 1 - 1i
_h = np.sqrt(-2 * mu2 * GAMMA)
_Umax = U + 2 * cd * cu
_mu_t = _Umax ** 2 / (4 * abs(GAMMA) ** 2)
MU_C = _mu_t + abs(_h) * np.cos(np.angle(GAMMA) / 2) / 2      # ~0.3977
MU0_STABLE = 0.96 * MU_C                                      # ~0.3818 (report: 0.38)


# ---- P2 reference shape functions on [0,1], nodes at 0, 0.5, 1 ----
def _shape(xi):
    N = np.array([2 * (xi - 0.5) * (xi - 1.0),
                  4 * xi * (1 - xi),
                  2 * xi * (xi - 0.5)])
    dN = np.array([4 * xi - 3.0,
                   4 - 8 * xi,
                   4 * xi - 1.0])          # d/dxi
    return N, dN


def assemble_gl(ne=400, L=56.06, mu0=MU0_STABLE, nq=5):
    """Assemble sparse A, M and nodal coordinates for the GL operator.
    Returns (A, M, x_nodes, free) where `free` indexes non-Dirichlet dofs."""
    nn = 2 * ne + 1
    xn = np.linspace(-L, L, nn)            # P2 node coordinates (uniform)
    h = (2 * L) / ne
    xg, wg = leggauss(nq)                  # on [-1,1]
    xq = 0.5 * (xg + 1.0)                  # -> [0,1]
    wq = 0.5 * wg
    Nq = np.array([_shape(x)[0] for x in xq])    # (nq,3)
    dNq = np.array([_shape(x)[1] for x in xq])   # (nq,3)

    rows, cols, valsA, valsM, valsK = [], [], [], [], []
    def mu_fun(x):
        return mu0 - cu ** 2 + 0.5 * mu2 * x ** 2

    for e in range(ne):
        g = [2 * e, 2 * e + 1, 2 * e + 2]           # global node ids
        xl = xn[g[0]]
        xphys = xl + xq * h
        Me = np.zeros((3, 3))
        Ke = np.zeros((3, 3))
        Ce = np.zeros((3, 3))
        Mu_e = np.zeros((3, 3), complex)
        for q in range(nq):
            N = Nq[q]; dN = dNq[q]
            Me += wq[q] * np.outer(N, N) * h
            Ke += wq[q] * np.outer(dN, dN) / h        # (dN/dx)(dN/dx)*h = dN*dN/h
            Ce += wq[q] * np.outer(N, dN)             # N*(dN/dx)*h = N*dN
            Mu_e += wq[q] * mu_fun(xphys[q]) * np.outer(N, N) * h
        # A_e = -nu*C + Mu - gamma*K   (weak form, IBP on diffusion)
        Ae = -NU * Ce + Mu_e - GAMMA * Ke
        for a in range(3):
            for b in range(3):
                rows.append(g[a]); cols.append(g[b])
                valsA.append(Ae[a, b]); valsM.append(Me[a, b])
                valsK.append(Ke[a, b])

    A = sp.csr_matrix((valsA, (rows, cols)), shape=(nn, nn), dtype=complex)
    M = sp.csr_matrix((valsM, (rows, cols)), shape=(nn, nn), dtype=float)
    K = sp.csr_matrix((valsK, (rows, cols)), shape=(nn, nn), dtype=float)
    free = np.arange(1, nn - 1)            # drop Dirichlet nodes 0 and nn-1
    A = A[free][:, free]; M = M[free][:, free]; K = K[free][:, free]
    global _LAST_STIFFNESS
    _LAST_STIFFNESS = K.tocsc()            # picked up by FECoDesign for H1 grad
    return A.tocsr(), M.tocsr(), xn[free], free


_LAST_STIFFNESS = None


def assemble_gl_K(ne=400, L=56.06, mu0=MU0_STABLE, nq=5):
    """Same as assemble_gl but also returns the stiffness matrix K."""
    A, M, xn, free = assemble_gl(ne=ne, L=L, mu0=mu0, nq=nq)
    return A, M, _LAST_STIFFNESS, xn, free


def load_vector(xn, x0, ne=400, L=56.06, nq=6):
    """Consistent load vector b_i = int psi(x;x0) phi_i dx for a Gaussian
    actuator/sensor profile centred at x0 (restricted to free dofs).
    Same routine serves actuator b(xa) and sensor c(xs)."""
    nn = 2 * ne + 1
    xfull = np.linspace(-L, L, nn)
    h = (2 * L) / ne
    xg, wg = leggauss(nq)
    xq = 0.5 * (xg + 1.0); wq = 0.5 * wg
    Nq = np.array([_shape(x)[0] for x in xq])
    b = np.zeros(nn)
    for e in range(ne):
        g = [2 * e, 2 * e + 1, 2 * e + 2]
        xl = xfull[g[0]]
        xphys = xl + xq * h
        psi = np.exp(-(xphys - x0) ** 2 / (2 * SIGMA ** 2))
        for a in range(3):
            b[g[a]] += np.sum(wq * psi * Nq[:, a]) * h
    return b[1:nn - 1]



def analytic_eigs(mu0, nmax=6):
    """Closed-form GL eigenvalues (Chen & Rowley eq. 2.3a) - validation only."""
    n = np.arange(nmax)
    return mu0 - cu ** 2 - NU ** 2 / (4 * GAMMA) - (n + 0.5) * _h


if __name__ == "__main__":
    import scipy.sparse.linalg as spla
    A, M, xn, free = assemble_gl(ne=200)
    print(f"FE assembly: sparse {A.shape}, nnz/row={A.nnz/A.shape[0]:.1f}, "
          f"mu0={MU0_STABLE:.4f}")
    v = spla.eigs(A, k=6, M=M, sigma=-0.02 - 0.65j, which='LM',
                  maxiter=5000, return_eigenvectors=False)
    v = v[np.argsort(-v.real)][:6]
    a = analytic_eigs(MU0_STABLE, 6); a = a[np.argsort(-a.real)]
    print("  leading eigenvalues (sparse shift-invert | analytic):")
    for k in range(6):
        print(f"    {v[k].real:+.5f}{v[k].imag:+.5f}j | {a[k].real:+.5f}{a[k].imag:+.5f}j")
    print(f"  max error = {np.max(np.abs(v - a)):.2e}")
