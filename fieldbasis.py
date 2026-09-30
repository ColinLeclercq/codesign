"""fieldbasis.py -- truncated bases for the free actuator/sensor fields
(TRACKER I.1-I.2, user 2026-09-22).

The free fields va (actuator) and vs (sensor) are full nodal vectors, smoothed
by a Sobolev map.  Here they are parametrised instead as  v = Phi beta  with Phi
real, M-orthonormal, supported on the admissible region, and truncated; the
reduced model becomes exact in beta once R(iw) Phi is in its basis.
2026-09-28: truncation does NOT regularise the design problem -- the optimum
puts its energy at whatever cutoff the basis has (STAGE5 32); a scale-invariant
Sobolev constraint (sqp_inner field rows, driver --field-kmax) does.

  hermite  : Hermite functions H_k((x-c)/s) exp(-((x-c)/s)^2/2), centre c = the
             region's midpoint, scale s (default 1.5 sigma) -- localised; a
             translated Gaussian has a rapidly converging expansion in them.
  laplace  : lowest Dirichlet eigenfunctions of the Laplacian on the region
             (K phi = lambda M phi on the region's nodes, zero outside) -- the
             geometry-general Fourier/Chebyshev (Laplace-Beltrami on a wall
             patch at the cavity); cosines/sines on GL's segment.  Returned in
             eigenvalue order (column k is the k-th eigenfunction).

Both returned M-orthonormal (Phi^T M Phi = I), so beta0 = Phi^T M v0 and the
projection error of a field is ||v - Phi Phi^T M v||_M.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.linalg import eigh


def _m_orth(Phi, M, tol=1e-10):
    """M-orthonormal basis of span(Phi), ORDER-PRESERVING and rank-revealing:
    modified Gram-Schmidt in the M inner product (twice), dropping columns
    whose residual falls below tol of their norm (high-order functions the
    mesh cannot resolve become numerically dependent).
    2026-09-28: this was an SVD of Mh^T Phi, whose U is an ARBITRARY rotation
    when the singular values coincide -- for an already M-orthonormal input
    (the Laplacian eigenvectors) every singular value is 1 and the returned
    columns were scrambled combinations of the eigenfunctions: same span (the
    optimisation is unaffected), but "the lowest N modes" meant nothing, and the
    truncation and per-mode energy figures of STAGE5 29.4 were wrong (32)."""
    # 2026-09-28 (user: never densify M): sparse M products only
    Md = M.tocsr().real if hasattr(M, 'tocsr') else np.asarray(M).real
    Q, MQ = [], []                                  # accepted vectors and M times them
    for j in range(Phi.shape[1]):
        v = np.asarray(Phi[:, j], float).copy()
        n0 = np.sqrt(max(float(v @ (Md @ v)), 0.0))
        for _ in range(2):                          # classical GS, twice (= modified GS accuracy)
            if Q:
                v -= np.array(Q).T @ (np.array(MQ) @ v)
        Mv = Md @ v
        nv = np.sqrt(max(float(v @ Mv), 0.0))
        if n0 > 0 and nv > tol * n0:
            Q.append(v / nv); MQ.append(Mv / nv)
    return np.array(Q).T


def hermite_basis(D, lo, hi, N, scale=None):
    from fe_codesign import SIGMA
    x = np.asarray(D.xn, float); c = 0.5 * (lo + hi)
    s = float(scale) if scale else 1.5 * SIGMA
    xi = (x - c) / s
    B = []
    for k in range(N):
        co = np.zeros(k + 1); co[k] = 1.0
        B.append(hermval(xi, co) * np.exp(-xi ** 2 / 2))
    B = np.array(B).T
    B /= np.maximum(np.abs(B).max(axis=0), 1e-300)
    return _m_orth(B, D.M.tocsc())


def laplace_basis(D, lo, hi, N, pad=0.0):
    """Lowest N Dirichlet eigenfunctions on [lo - pad, hi + pad]."""
    x = np.asarray(D.xn, float)
    idx = np.where((x > lo - pad) & (x < hi + pad))[0]      # interior nodes (Dirichlet: ends excluded)
    M = D.M.tocsr(); K = D.Kstiff.tocsr()
    Ms = M[idx][:, idx].real.tocsc(); Ks = K[idx][:, idx].real.tocsc()
    Ms = 0.5 * (Ms + Ms.T); Ks = 0.5 * (Ks + Ks.T)
    N = min(int(N), len(idx))                                # at most one per interior node
    # 2026-09-28 (user: never densify M): sparse shift-invert Lanczos for the N
    # smallest eigenpairs of (K, M) (K is positive definite with the Dirichlet
    # ends removed, so sigma = 0 is below the spectrum).  ARPACK needs N < size - 1;
    # asking for (nearly) every mode of a small mesh is the only dense case.
    if N < len(idx) - 1:
        import scipy.sparse.linalg as _spla
        # fixed start vector and sign (2026-09-29, STAGE6 6.8): ARPACK's random
        # start made the basis -- and through the reduced model's input basis
        # every field run -- differ from run to run (t at 1e-4 in round 0)
        v0 = np.random.default_rng(12345).standard_normal(Ks.shape[0])
        lam, V = _spla.eigsh(Ks, k=N, M=Ms, sigma=0.0, which='LM', tol=1e-12, v0=v0)
        o = np.argsort(lam); lam, V = lam[o], V[:, o]
        V = V * np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])
    else:
        lam, V = eigh(Ks.toarray(), Ms.toarray())
    Phi = np.zeros((len(x), N))
    Phi[idx, :] = V[:, :N]
    return _m_orth(Phi, D.M.tocsc()), lam[:N]


def project(Phi, M, v):
    """beta = Phi^T M v and the relative M-norm projection error."""
    beta = Phi.T @ (M @ v)
    r = v - Phi @ beta
    nv = np.sqrt(abs(np.vdot(v, M @ v)))
    return beta, float(np.sqrt(abs(np.vdot(r, M @ r))) / max(nv, 1e-300))


# ---------------------------------------------------------------- atoms (2026-09-28, STAGE5 32)
# A field as a sum of Gabor ATOMS (Gaussian envelope x plane wave) of free
# complex amplitude a_j, centre c_j, width s_j >= s_min and wavenumber
# |kappa_j| <= kappa_max:
#     v(x) = sum_j a_j exp(-(x - c_j)^2 / (2 s_j^2) + i kappa_j (x - c_j)).
# Parametrised by (Re a, Im a, c, rho, tau) with s = s_min (1 + rho^2) and
# kappa = kappa_max tanh(tau), so both bounds hold without constraint rows.
# Each atom's squared RMS wavenumber is kappa^2 + 1/(2 s^2) <= kappa_max^2 +
# 1/(2 s_min^2): the bounds are the regularisation, and the design set is a
# finite-dimensional smooth family -- mesh-independent once the mesh resolves
# the smallest atom.  (Plain Gaussians cannot carry the optimal fields' phase
# gradient: 6 of them fit the frozen fields to 15 % only.)
NATOM_PAR = 5


def atom_widths(rho, smin):
    return smin * (1.0 + np.asarray(rho, float) ** 2)


def atom_wavenumbers(tau, kmax):
    return kmax * np.tanh(np.asarray(tau, float))


def _atoms(x, c, rho, tau, smin, kmax):
    x = np.asarray(x, float)[:, None]; c = np.asarray(c, float)[None, :]
    s = atom_widths(rho, smin)[None, :]; k = atom_wavenumbers(tau, kmax)[None, :]
    return np.exp(-(x - c) ** 2 / (2.0 * s ** 2) + 1j * k * (x - c)), x - c, s, k


def atom_field(x, a, c, rho, tau, smin, kmax):
    phi, *_ = _atoms(x, c, rho, tau, smin, kmax)
    return phi @ np.asarray(a, complex)


def atom_jacobian(x, a, c, rho, tau, smin, kmax):
    """The complex field derivatives dv/dtheta (n x 5 n_atoms), theta = (Re a,
    Im a, c, rho, tau): atom_grad(G) = Re(G^H J).  (STAGE6 6.29: the reduced
    gradients project J once per design instead of lifting every row.)"""
    phi, dx, s, k = _atoms(x, c, rho, tau, smin, kmax)
    a = np.asarray(a, complex)[None, :]
    return np.concatenate([phi, 1j * phi,
                           a * phi * (dx / s ** 2 - 1j * k),
                           a * phi * dx ** 2 / s ** 3 * (2.0 * smin * np.asarray(rho, float))[None, :],
                           a * phi * 1j * dx * (kmax / np.cosh(np.asarray(tau, float)) ** 2)[None, :]], axis=1)


def atom_grad(x, a, c, rho, tau, smin, kmax, G):
    """Gradient of a real F(v) w.r.t. (Re a, Im a, c, rho, tau), given the nodal
    G = dF/dRe v + i dF/dIm v (the packer's convention: dF = Re(conj(G) . dv))."""
    return np.real(np.conj(np.asarray(G, complex)) @ atom_jacobian(x, a, c, rho, tau, smin, kmax))


def split_atoms(p, n):
    return p[:n] + 1j * p[n:2 * n], p[2 * n:3 * n], p[3 * n:4 * n], p[4 * n:5 * n]


def fit_atoms(x, v, M, n_atoms, smin, kmax):
    """Least-squares fit (M-norm) of n_atoms Gabor atoms to a nodal field v:
    greedy (one atom at a time, placed at the residual's maximum), each stage a
    joint quasi-Newton refinement of all atoms so far.  Returns the parameter
    vector p = (Re a, Im a, c, rho, tau) (field at unit M-norm) and the relative
    M-norm error.  2026-09-28 (user: never a factor of M): the objective
    r^H M r and its gradient 2 Re(J^H M r) (atom_grad with G = 2 M r) use only
    products with the sparse M; the earlier version least-squared a dense
    Cholesky factor of M."""
    from scipy.optimize import minimize
    x = np.asarray(x, float); v = np.asarray(v, complex)
    Ms = M.tocsr() if hasattr(M, 'tocsr') else np.asarray(M)
    v = v / np.sqrt(abs(np.vdot(v, Ms @ v)))

    def fg(q, k):
        a, c, rho, tau = split_atoms(q, k)
        r = atom_field(x, a, c, rho, tau, smin, kmax) - v
        Mr = Ms @ r
        return float(np.real(np.vdot(r, Mr))), atom_grad(x, a, c, rho, tau, smin, kmax, 2.0 * Mr)
    p = np.zeros(0); res = v.copy()
    for j in range(n_atoms):
        k = j + 1
        a0 = np.zeros(k, complex); c0 = np.zeros(k); r0 = np.ones(k); t0 = np.zeros(k)
        if j:
            a0[:j], c0[:j], r0[:j], t0[:j] = split_atoms(p, j)
        i_ = int(np.argmax(np.abs(res))); a0[j] = res[i_]; c0[j] = x[i_]
        q0 = np.concatenate([a0.real, a0.imag, c0, r0, t0])
        p = minimize(lambda q: fg(q, k), q0, jac=True, method='BFGS',
                     options=dict(gtol=1e-12, maxiter=20000)).x
        a, c, rho, tau = split_atoms(p, k)
        res = v - atom_field(x, a, c, rho, tau, smin, kmax)
    a, c, rho, tau = split_atoms(p, n_atoms)
    f = atom_field(x, a, c, rho, tau, smin, kmax); nf = np.sqrt(abs(np.vdot(f, Ms @ f)))
    err = float(np.sqrt(abs(np.vdot(f / nf - v, Ms @ (f / nf - v)))))
    return np.concatenate([a.real / nf, a.imag / nf, c, rho, tau]), err


def p2_eval(Ds, v, x):
    """The P2 finite-element function with interior nodal values v on Ds's mesh
    (zero Dirichlet ends), evaluated at the points x (2026-09-28, STAGE5 32.7).
    Carrying a field to another mesh this way keeps it the SAME function (the
    coarse P2 space is nested in the refined one); linear interpolation of the
    nodal values does not, and smooths away content near the mesh scale."""
    ne = Ds.ne; dx = Ds.xn[1] - Ds.xn[0]
    X = np.linspace(Ds.xn[0] - dx, Ds.xn[-1] + dx, 2 * ne + 1)
    V = np.concatenate([[0], np.asarray(v, complex), [0]])
    h = (X[-1] - X[0]) / ne
    x = np.asarray(x, float)
    e = np.clip(np.floor((x - X[0]) / h).astype(int), 0, ne - 1)
    xi = (x - X[2 * e]) / h
    N0 = 2 * (xi - 0.5) * (xi - 1.0); N1 = 4 * xi * (1 - xi); N2 = 2 * xi * (xi - 0.5)
    return N0 * V[2 * e] + N1 * V[2 * e + 1] + N2 * V[2 * e + 2]
