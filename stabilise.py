"""stabilise.py -- the abscissa-descent phase for random starts (Stage 2).

HIFOO's convention (Burke, Henrion, Lewis, Overton 2006, following Burke et al.
2003): "stabilization is implemented by applying optimization to the spectral
abscissa objective, stopping as soon as a stabilizing controller is found";
the engine is HANSO's BFGS phase on the nonsmooth abscissa with an eigenvector
gradient, the bundle and sampling phases only if BFGS stalls first.  Here the
engine is pyGRANSO unconstrained -- the same BFGS with an inexact line search.

The objective is the MAX of the closed-loop abscissa and the compensator's
(minus its allowance), both dense at GL scale (n + r ~ 200-400; the reduced
pencil is not used here because its basis is built from a design and the
phase moves the design a lot).  Gradients from the left/right eigenvectors of
the rightmost eigenvalue through the same D.grads the inner solve uses, and a
dense (r x r) pair for the compensator.

The stop is alpha <= -eps - delta with a cushion delta (default 3e-3, i.e. a
few x 1e-3 as the reviewer set): the stage-1 finding that a design exactly ON
the line is refused by the sweep makes the cushion mandatory.  A fixed
iteration cap is the budget; a start that does not reach the margin within it
counts as NOT STABILISED -- the fraction-stabilised number of the stage-2
gate.  Positions are design variables in the phase too (a start that can only
be stabilised by moving the sensor is a legitimate start).
"""
from __future__ import annotations
import time
import numpy as np
from scipy.linalg import eig as deig

import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct


def dense_abscissa_pair(D, arow, nvec, K, b, c):
    """Rightmost eigenvalue of the dense closed-loop pencil with its left and
    right eigenvectors (n + r), and eta = u^H M_cl v."""
    import controller as ctl
    n, r = D.n, D.r
    Ac = ctl.A_of(*arow); Bc = np.full(r, D.mc, complex)
    Ad = np.asarray(D.A.todense()); Md = np.asarray(D.M.todense())
    nv = np.asarray(nvec, complex).ravel()
    Acl = np.zeros((n + r, n + r), complex); Mcl = np.zeros_like(Acl)
    Acl[:n, :n] = Ad + complex(K) * np.outer(b, np.conj(c))
    Acl[:n, n:] = np.outer(b, nv)
    Acl[n:, :n] = np.outer(Bc, np.conj(c))
    Acl[n:, n:] = Ac
    Mcl[:n, :n] = Md; Mcl[n:, n:] = np.eye(r)
    lam, VL, VR = deig(Acl, Mcl, left=True, right=True)
    fin = np.isfinite(lam); lam, VL, VR = lam[fin], VL[:, fin], VR[:, fin]
    j = int(np.argmax(lam.real))
    v, u = VR[:, j], VL[:, j]
    eta = np.vdot(u, Mcl @ v)
    return float(lam[j].real), complex(lam[j]), v, u, eta


def stabilise(D, pk, flags, z0, eps, delta=3e-3, maxit=150, ctrl_max=0.0,
              verbose=False):
    """Returns (z, alpha_cl, alpha_ctrl, iters, evals, ok, wall)."""
    import controller as ctl
    NZ = len(z0); t0 = time.time()
    cache = {}
    lim = max(-eps, ctrl_max)
    target = -eps - delta

    def phi(z):
        key = tuple(np.round(z, 12))
        if key in cache:
            return cache[key]
        arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
        b, c, pa, ps, na, ns = D.bc(xa, xs, va, vs)
        al, lam, v, u, eta = dense_abscissa_pair(D, arow, nvec, K, b, c)
        cl = D.cl(b, c)
        g_cl = np.asarray(pk.grad(D.grads(cl, v, u, arow, nvec, K, xa, xs, pa, ps, na, ns,
                                          flags, eta=eta)), float)
        # compensator abscissa and its gradient (dense r x r, the per-pole form)
        Ac = ctl.A_of(*arow); w_, VLc, VRc = deig(Ac, left=True, right=True)
        k = int(np.argmax(w_.real)); al_c = float(w_[k].real)
        vr, wl = VRc[:, k], VLc[:, k]; den = np.vdot(wl, vr)
        r = Ac.shape[0]; g_c = np.zeros(NZ)
        if abs(den) > 1e-12:
            Gm = np.outer(wl.conj(), vr) / den
            gc = dict(d=np.array([Gm[i, i] for i in range(r)]),
                      e=np.array([Gm[i, i + 1] for i in range(r - 1)]),
                      f=np.array([Gm[i + 1, i] for i in range(r - 1)]),
                      n=np.zeros(r, complex), K=0j, xa=0.0, xs=0.0)
            g_c = np.real(np.asarray(pk.grad(gc), float))
        # objective: the worse of the two margins
        f1, f2 = al, al_c - lim
        if f1 >= f2:
            f, g = f1, g_cl
        else:
            f, g = f2, g_c
        cache[key] = (f, g, al, al_c)
        return cache[key]

    class _H:
        pass
    state = dict(best=None, evals=0)

    class _Phi(torch.autograd.Function):
        """pyGRANSO differentiates by autograd: the analytic gradient goes in
        through a custom Function's backward (as solve_epigraph does), not
        through .grad -- set by hand it is ignored and pyGRANSO reports
        'converged' at iteration 0."""
        @staticmethod
        def forward(ctx, zt):
            z = zt.detach().cpu().numpy().ravel()
            f, g, al, al_c = phi(z)
            state['evals'] += 1
            if state['best'] is None or max(al, al_c - lim) < state['best'][1]:
                state['best'] = (z.copy(), max(al, al_c - lim), al, al_c)
            ctx.save_for_backward(torch.tensor(g.reshape(-1, 1), dtype=zt.dtype))
            return torch.tensor(float(f), dtype=zt.dtype)
        @staticmethod
        def backward(ctx, gout):
            (g,) = ctx.saved_tensors
            return gout * g

    def fn(X):
        return _Phi.apply(X.z), None, None

    var = {'z': (NZ, 1)}
    opts = pygransoStruct()
    opts.x0 = torch.tensor(np.asarray(z0, float).reshape(-1, 1))
    opts.maxit = int(maxit); opts.print_level = 1 if verbose else 0
    opts.opt_tol = 1e-8; opts.torch_device = torch.device('cpu')
    opts.double_precision = True
    # HIFOO: stop at the first stabilising controller (with the cushion)
    def halt(it, x, pf, d, bfgs, Hreg, lse, alpha, ng, sv, stat, fb):
        z = x.detach().cpu().numpy().ravel()
        f, g, al, al_c = phi(z)
        return bool(max(al, al_c - lim) <= target)
    opts.halt_log_fn = halt
    try:
        pygranso(var_spec=var, combined_fn=fn, user_opts=opts)
    except Exception as e:      # the phase is best-effort; the record says so
        if verbose:
            print(f"    stabilise: pygranso raised {type(e).__name__}: {e}")
    zb, fb, al, al_c = state['best']
    ok = fb <= target
    return zb, al, al_c, state['evals'], ok, time.time() - t0
