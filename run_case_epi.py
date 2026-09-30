"""
run_case_epi.py — EPIGRAPH reformulation with an exchange loop.

Instead of handing the solver the non-smooth objective

    min_theta  max_omega sigma_1(theta, omega)

we solve the smooth semi-infinite programme

    min_{theta, t}  t
      s.t.  sigma_1(theta, omega_i) <= t     for omega_i in the active set Omega
            alpha(theta) + eps      <= 0

and enlarge Omega in an outer loop whenever the CERTIFIED norm exceeds t.

Why this should behave better (and the evidence for it).  The objective is now
LINEAR, so no curvature has to be accumulated across a kink; every constraint
sigma_1(theta, omega_i) is a *simple* singular value at a FIXED frequency and is
therefore smooth in theta -- measured singular-value gaps at the optima of the
previous study lie between 1.8e-2 and 9e-1, nowhere near degenerate.  The active
set is handled explicitly by SQP with multipliers rather than being inferred
from function values, which is what forced the per-case step-size tuning in the
max formulation.

Interface, output files and CLI mirror run_case.py exactly, so collect.py,
certify_results.py, check_optimality.py and the plotting scripts all work
unchanged on the results.

    python run_case_epi.py --case K+B+C --start 0 --ne 200 --wall 1500
"""
from __future__ import annotations
import os as _os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import argparse, json, os, sys, time
import numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fe_codesign import FECoDesign, parse_case
import controller as ctl
import certify_sparse as csp
from packer import Packer, EPS_STAB_DEFAULT


# --------------------------------------------------------------------------
# CONTINUATION GATE (matrix-free).
#
# The line certificate proves "nothing ON Re = -eps"; stability follows only by
# continuation, i.e. no eigenvalue crosses BETWEEN two checked iterates.  That
# premise was underwritten by the trust region bounding ||dtheta|| -- and the
# trust region defaults to OFF because an ablation measured it harmful on gamma
# and wall clock.  The ablation did not measure the invariant it protected:
# with unbounded inner steps, eigenvalues jumped the line between accepted
# iterates, and eight designs were saved whose in-band abscissa read -0.216
# while their true rightmost eigenvalue sat at Re = +0.33 to +10.1 (three of
# them exactly on the algebraic sigma_2 floor 3.5091, which is where an
# optimiser freed of the stability constraint lands).
#
# The gate below restores the premise without dense algebra and without
# throttling the optimiser:
#
#   * a WATCH LIST of near-line eigenvalues is harvested for free from the
#     line sweep of every accepted iterate (eigs_seen);
#   * each candidate step is checked by TRACKING those eigenvalues onto the
#     candidate design -- one targeted shift-invert query per eigenvalue;
#   * a tracked eigenvalue found at Re > -eps is a caught crossing: reject;
#   * motion too large for tracking to be conclusive triggers an ADAPTIVE
#     HOMOTOPY: the design segment is bisected, the midpoint gets its own
#     line sweep and fresh watch list, and both halves are re-checked.  The
#     segment is accepted only when every sub-step is small enough that the
#     tracked spectrum demonstrably did not cross.
#
# Everything here is sparse-LU + short Arnoldi: it ports to the cavity
# unchanged.  Dense algebra appears ONLY in the save veto and the no-regress
# guard below, explicitly as a check, and is skipped above --dense_nmax.
# --------------------------------------------------------------------------
def _topk_watch(eigs, k, eps, reach=3.0):
    """Rightmost near-line eigenvalues, deduplicated: the crossing candidates.

    `reach` keeps only eigenvalues within that distance of the line -- an
    eigenvalue at Re = -20 cannot reach the line in one continuation-safe
    sub-step, and tracking it wastes a solve per candidate.
    """
    out = []
    for lam in sorted((complex(e) for e in eigs), key=lambda z: -z.real):
        if lam.real < -eps - reach:
            continue
        if any(abs(lam - q) < 1e-8 for q in out):
            continue
        out.append(lam)
        if len(out) >= k:
            break
    if not out:
        # everything sits deeper than `reach`: watch the rightmost anyway --
        # an empty watch list would make the gate reject every candidate.
        for lam in sorted((complex(e) for e in eigs), key=lambda z: -z.real):
            if any(abs(lam - q) < 1e-8 for q in out):
                continue
            out.append(lam)
            if len(out) >= k:
                break
    return out


def continuation_gate(D, pk, z0, z1, watch0, eps, depth=4,
                      motion_frac=0.5, motion_floor=0.05, kwatch=8,
                      verbose=True, pool=None):
    """Certify the design segment [z0, z1] free of stability-line crossings.

    Returns (ok, watch1, nquery, nsplit).  `watch1` is the watch list at z1
    (valid only when ok).  Conservative by construction: an unconverged track,
    an infeasible midpoint, or exhausted depth all reject.  The packed design
    vector is real and every packed quantity is linear in it, so the straight
    segment is a continuous path through design space, which is all
    continuation needs.
    """
    z0 = np.asarray(z0, float)
    z1 = np.asarray(z1, float)
    stats = dict(nq=0, nsplit=0)

    def _design(z):
        arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
        b, c, *_ = D.bc(xa, xs, va, vs)
        return arow, nvec, K, b, c, ctl.A_of(*arow)

    def _seg(za, zb, wa, d):
        wa_use = [w for w in wa if w is not None]
        if not wa_use:
            if verbose:
                print("    continuation: empty watch list -> cannot prove; "
                      "reject", flush=True)
            return False, wa
        arow_b, nvec_b, K_b, b_b, c_b, Ac_b = _design(zb)
        Bc = ctl.Bc_of(arow_b, D.r, D.mc)
        ok, split, wb, worst = csp.continuation_step(
            D.A, D.M, b_b, c_b, Ac_b, Bc, nvec_b, K_b, wa_use, eps=eps,
            motion_frac=motion_frac, motion_floor=motion_floor, pool=pool)
        stats['nq'] += len(wa_use)
        if not ok:
            if verbose:
                print(f"    continuation REJECTS: tracked eigenvalue crossed "
                      f"to {worst:.6g}", flush=True)
            return False, wb
        if not split:
            return True, wb
        if d <= 0:
            if verbose:
                print("    continuation REJECTS: split depth exhausted with "
                      "motion still above the allowance (step too large to "
                      "certify)", flush=True)
            return False, wb
        stats['nsplit'] += 1
        zm = 0.5 * (za + zb)
        # The midpoint is CHECKED, not merely interpolated: its own line sweep
        # (crossings happen ON the line, which is exactly what it detects) and
        # a fresh watch list for the second half.
        seen_m = []
        arow_m, nvec_m, K_m, b_m, c_m, _ = _design(zm)
        al_m = D.alpha_certified(arow_m, nvec_m, K_m, b_m, c_m, eps=eps,
                                 eigs_seen=seen_m)[0]
        if al_m + eps > 0.0:
            if verbose:
                print(f"    continuation REJECTS: homotopy midpoint is "
                      f"infeasible (alpha={al_m:+.4f})", flush=True)
            return False, wb
        ok1, wm = _seg(za, zm, wa, d - 1)
        if not ok1:
            return False, wm
        wm2 = _topk_watch(list(wm) + seen_m, kwatch, eps)
        return _seg(zm, zb, wm2, d - 1)

    ok, w1 = _seg(z0, z1, list(watch0), depth)
    return ok, w1, stats['nq'], stats['nsplit']


def continuation_gate_reduced(D, pk, z0, z1, watch0, eps, oracle, depth=4,
                              motion_frac=0.5, motion_floor=0.05, kwatch=8,
                              verbose=True, res_tol=1e-3, reach=1.0):
    """Algebra note item 11 (2026-09-28, STAGE5 33): the continuation gate with
    the eigenvalues TRACKED ON THE REDUCED PENCIL (Mengi's interpolatory
    framework).  Same decisions and structure as continuation_gate; what changes
    is where the eigenvalues come from:
      * at every point of the segment the reduced closed-loop pencil's Ritz
        pairs near the line, each accepted only if its FULL-model residual
        (sparse matvecs, no factorisation) is below res_tol;
      * a watched eigenvalue is tracked by the nearest accepted Ritz value; when
        none matches within the motion allowance it is tracked on the full model
        (one shift-invert, as the full gate does) -- so every watched eigenvalue
        is followed at least as reliably as before;
      * a midpoint's line certificate (10-15 factorisations) is replaced by the
        rightmost accepted Ritz value there; with no accepted Ritz pair the full
        certificate is used.
    Returns (ok, watch1, nquery, nsplit, n_full) with n_full the full-model
    fallbacks (tracks + midpoint certificates)."""
    z0 = np.asarray(z0, float); z1 = np.asarray(z1, float)
    stats = dict(nq=0, nsplit=0, nfull=0)

    def _design(z):
        arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
        b, c, *_ = D.bc(xa, xs, va, vs)
        return arow, nvec, K, b, c

    def _ritz(z):
        arow, nvec, K, b, c = _design(z)
        rc = oracle.make_cl(b, c, D.r, D.mc)
        pairs = rc.near_line_pairs(arow, nvec, K, eps, reach=reach, res_tol=res_tol, mc=D.mc, kmax=16)
        return [complex(p_['lam']) for p_ in pairs], (arow, nvec, K, b, c)

    def _seg(za, zb, wa, d):
        wa_use = [complex(w) for w in wa if w is not None]
        if not wa_use:
            if verbose:
                print("    continuation (reduced): empty watch list -> reject", flush=True)
            return False, wa
        Lb, (arow_b, nvec_b, K_b, b_b, c_b) = _ritz(zb)
        stats['nq'] += len(wa_use)
        if any(l.real > -eps for l in Lb):
            if verbose:
                print(f"    continuation (reduced) REJECTS: Ritz value {max(Lb, key=lambda q: q.real):.6g} "
                      f"right of the line", flush=True)
            return False, Lb
        need_split, wb, cl_b = False, [], None
        for lam0 in wa_use:
            gap = -eps - lam0.real
            allow = max(motion_frac * max(gap, 0.0), motion_floor)
            lam1 = min(Lb, key=lambda q: abs(q - lam0)) if Lb else None
            if lam1 is None or abs(lam1 - lam0) > allow:
                # not matched on the reduced pencil: the full model's shift-invert
                if cl_b is None:
                    from closedloop_sparse import ClosedLoopResolvent as _CLR
                    cl_b = _CLR(D.M, D.A, b_b, c_b, r=D.r, mc=D.mc)
                lam1 = csp._track_one(cl_b, arow_b, nvec_b, K_b, lam0, max(0.3 * motion_floor, 1e-4))
                stats['nfull'] += 1
                if lam1 is None:
                    need_split = True; continue
            if lam1.real > -eps:
                if verbose:
                    print(f"    continuation (reduced) REJECTS: tracked eigenvalue crossed to {lam1:.6g}", flush=True)
                return False, wb
            if abs(lam1 - lam0) > allow:
                need_split = True
            wb.append(lam1)
        wb = _topk_watch(wb + Lb, kwatch, eps)
        if not need_split:
            return True, wb
        if d <= 0:
            if verbose:
                print("    continuation (reduced) REJECTS: split depth exhausted", flush=True)
            return False, wb
        stats['nsplit'] += 1
        zm = 0.5 * (za + zb)
        Lm, (arow_m, nvec_m, K_m, b_m, c_m) = _ritz(zm)
        if Lm:
            al_m = max(l.real for l in Lm); seen_m = list(Lm)
        else:
            seen_m = []
            al_m = D.alpha_certified(arow_m, nvec_m, K_m, b_m, c_m, eps=eps, eigs_seen=seen_m)[0]
            stats['nfull'] += 1
        if al_m + eps > 0.0:
            if verbose:
                print(f"    continuation (reduced) REJECTS: homotopy midpoint is infeasible (alpha={al_m:+.4f})", flush=True)
            return False, wb
        ok1, wm = _seg(za, zm, wa, d - 1)
        if not ok1:
            return False, wm
        return _seg(zm, zb, _topk_watch(list(wm) + seen_m, kwatch, eps), d - 1)

    ok, w1 = _seg(z0, z1, list(watch0), depth)
    return ok, w1, stats['nq'], stats['nsplit'], stats['nfull']


# --------------------------------------------------------------------------
def solve_epigraph(D, pk, flags, z0, Omega, t0, eps, wall, H0mode='auto',
                   verbose=False, H0=None, trust=None, ctrl_max=0.0,
                   wmax=1e12, par=None, tau0=1.0,
                   budget='clock', maxit=5000, clock_guard=8.0, ls_fallback=False, imcap=True, select_cons=False, opt_tol=1e-10, sel_margin=1e-3, enroll=None, qp_log=False, qp_eps=None, qp_maxiter=None, level_norm=True, qp_solver='osqp', oracle=None, oracle_res_tol=1e-4, ctrl_line_gate=False, bandmax=None, follow=None, follow_tops=None):
    """One inner solve of the epigraph NLP over (z, tau), t = tau * t0.

    `budget` decides what STOPS the inner solve, and therefore whether a run is
    reproducible at all.

      'clock'  (the frozen behaviour, and the default)
               maxit = 5000, maxclocktime = wall.  In practice 5000 is never
               reached and the clock is the binding limit, so the number of
               BFGS-SQP iterations depends on machine speed and load.  Three
               identical invocations of case B at ne = 100 returned gamma =
               10.259071, 10.355313 and 10.263187.

      'iters'  maxit = `maxit`, and the clock is demoted to a GUARD at
               clock_guard x wall -- present only so a pathological run cannot
               overrun a SLURM allocation, and loud when it binds, because a
               bound guard means the clock decided after all and the result is
               not reproducible.

    The default is deliberately unchanged: switching it would silently move
    every number the frozen suite is compared against.
    """
    follow_tops = {} if follow_tops is None else follow_tops
    import torch
    from pygranso.pygranso import pygranso
    from pygranso.pygransoStruct import pygransoStruct
    torch.set_default_dtype(torch.double)

    NZ = len(z0)
    W = np.asarray(sorted(Omega), float)
    m = len(W)
    # ---- STAGE 1b, note item 11 (candidate): the level scaling.  With
    # level_norm the rows are sigma/t0 - tau and the objective is tau, so every
    # quantity the penalty function mixes is O(1).  With it off, t0 -> 1: rows
    # sigma - t, objective t, in the norm's own units.  Opt-in; the note names
    # "the level scaling constant" without saying which; --tau0 (the other
    # candidate) is already 1.0.
    t_level = t0
    if not level_norm:
        t0 = 1.0
    cache = {}
    best = {'g': np.inf, 'z': None, 'al': None, 't': np.inf, 'clr': None}

    def ev(v):
        key = v.tobytes()
        if key in cache:
            return cache[key]
        z, tau = v[:NZ], float(v[NZ])
        arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
        b, c, pa, ps, na, ns = D.bc(xa, xs, va, vs)
        if oracle is not None:
            # STAGE 5: the sigma rows on the PROJECTED oracle.  Same gain /
            # gain_k interface, k x k dense instead of an LU per frequency;
            # D.grads takes the proxy (full-length b, c; reduced ctrl state).
            cl = oracle.make_cl(b, c, D.r, D.mc)
            clg = cl.proxy
        else:
            cl = D.cl(b, c)
            clg = cl
        # constraints sigma_1(theta, w_i)/t0 - tau <= 0
        # The pole-frequency row only makes sense when the compensator is
        # FREE.  With K fixed (FB+FC and friends) it is a CONSTANT with an
        # identically zero gradient: pyGRANSO cannot satisfy or improve it, the
        # QP is degenerate, and the line search fails at every step size --
        # 0 iterations, 41 evaluations, termination code 6.
        _pole_row = imcap and bool(flags['K']) and flags.get('ctrl_mode', 'full') in (
            'full', 'sp', 'gain', 'gainr')
        # STAGE 1b, note item 3: sigma_2(omega_i) <= t enrolled as its own
        # smooth constraint at the frequencies in `enroll` (indices into W).
        # Appended AFTER every existing row so no index used elsewhere moves.
        _E = list(enroll) if enroll else []
        ncon = m + 1 + (1 if _pole_row else 0) + (1 if trust else 0) + len(_E) + (1 if (bandmax and oracle is not None) else 0)
        cv = np.empty(ncon)
        J = np.zeros((ncon, NZ + 1))
        smax = -np.inf
        # The |Omega| frequency constraints are independent and equal-cost, so
        # they are the block that actually rewards a pool: 13.2x at 32 workers.
        if par is not None and oracle is None:
            sig_v, G = par.constraints(z, W)
            smax = float(np.max(sig_v))
            cv[:m] = sig_v / t0 - tau
            J[:m, :NZ] = G / t0
            J[:m, NZ] = -1.0
        else:
            for i, w in enumerate(W):
                wf = float(w)
                rad = follow.get(round(float(w), 6)) if (follow and oracle is not None) else None
                if rad:
                    # PEAK-FOLLOWING row (peakfollow.py): value and gradient at the relocated top
                    from peakfollow import follow as _follow
                    wf, _ = _follow(cl, w, arow, nvec, K, rad)
                    follow_tops[round(float(w), 6)] = wf
                res = cl.gain(wf, arow, nvec, K)
                s = res['sigma']; smax = max(smax, s)
                gr = D.grads(clg, res['x'], res['z'], arow, nvec, K, xa, xs,
                             pa, ps, na, ns, flags)
                cv[i] = s / t0 - tau
                J[i, :NZ] = pk.grad(gr) / t0
                # d cv_i / d tau = -1 for EVERY row.  This line was outside the
                # loop, so only the LAST constraint carried it and all the
                # others reported that reducing tau was free.  The line search
                # then bracketed a minimiser it could never accept -- pyGRANSO
                # code 6, zero iterations, at any step size and independent of
                # the pool, H0 or the starting tau.
                J[i, NZ] = -1.0
        # CONTROLLER-POLE GUARD, inside the oracle.  A_c is r x r, so its
        # spectrum is exact and costs microseconds.  Shift-invert with shifts
        # near the imaginary axis can NEVER see a pole at Re = 11681, which is
        # exactly where the optimiser drove them; adding the controller's own
        # abscissa to the stability constraint closes that blind spot at the
        # only place it matters -- inside the line search.
        # exact rightmost controller pole AND its gradient.  A_c is r x r, so
        # both come from one small dense eigendecomposition:  for a simple
        # eigenvalue lam with right/left vectors v, w,
        #     d lam / d A_c = conj(w) v^T / (w^H v)
        # and dA_c/dd_j, /de_j, /df_j are single entries, so the design
        # gradient is read straight off that matrix.
        Acm = ctl.A_of(*arow)
        _rr = len(arow[0])
        g_ctrl = np.zeros(NZ)
        # LEFT eigenvectors directly, never by inverting V: the eigenvector
        # matrix is singular whenever the controller has repeated poles, and
        # np.linalg.inv then raises straight through the autograd wrapper
        # (pyGRANSO reports it as "check opts.globalAD").  scipy's eig returns
        # both sides without an inversion.
        from scipy.linalg import eig as _seig
        # POLE-FREQUENCY CONSTRAINT.  A compensator pole at |Im| = 138 is a
        # resonance where the plant has nothing (the GL spectrum decays from
        # Im ~ 0.65), is not implementable at cavity scale, and sits outside
        # the band the model is valid over.  The optimiser places one there
        # because a pole OUTSIDE the certified band is invisible to sup
        # sigma_1 while still contributing feedback inside it -- the same
        # blind-spot exploitation as the runaway real parts.  Constraining
        # |Im lambda| <= wmax removes the incentive rather than widening the
        # band to chase it, and keeps the certification cheap.
        try:
            _w_, _VL_, _VR_ = _seig(Acm, left=True, right=True)
            _jr = int(np.argmax(_w_.real))
            _ji = int(np.argmax(np.abs(_w_.imag)))
            im_ctrl = float(np.abs(_w_[_ji].imag))
            al_ctrl = float(_w_[_jr].real)
            _vr = _VR_[:, _jr]; _wl = _VL_[:, _jr]
            _den = np.vdot(_wl, _vr)
            if abs(_den) < 1e-12:          # defective: gradient undefined
                raise ValueError('defective eigenvalue')
            _G = np.outer(_wl.conj(), _vr) / _den
            # and the same for the WIDEST pole, for d|Im lambda|/dtheta
            _vi = _VR_[:, _ji]; _wi = _VL_[:, _ji]
            _dni = np.vdot(_wi, _vi)
            _Gi = (np.outer(_wi.conj(), _vi) / _dni
                   if abs(_dni) > 1e-12 else None)
            _sgn = 1.0 if _w_[_ji].imag >= 0 else -1.0
        except Exception:
            # keep the VALUE (still exact) and drop the gradient: a zero
            # gradient merely slows the solver, whereas an exception aborts it
            _ev = np.linalg.eigvals(Acm)
            al_ctrl = float(np.max(_ev.real))
            im_ctrl = float(np.max(np.abs(_ev.imag)))
            _G = None
            _Gi = None
        g_im = np.zeros(NZ)
        if _Gi is not None and flags['K'] \
                and flags.get('ctrl_mode', 'full') in ('full', 'sp'):
            # d|Im lambda|/dA_c = sign(Im lambda) * i * conj(w v^H / w^H v).
            # Verified against finite differences in BOTH the real and the
            # imaginary parameter directions to 2e-9; the other three sign and
            # conjugation conventions each get one direction wrong.
            _Gm = _sgn * 1j * _Gi.conj()
            gdi = np.array([_Gm[i, i] for i in range(_rr)])
            gei = np.array([_Gm[i, i + 1] for i in range(_rr - 1)])
            gfi = np.array([_Gm[i + 1, i] for i in range(_rr - 1)])
            try:
                _gz = pk.zero_grad() if getattr(pk, 'mimo', False) else dict(n=np.zeros(_rr, complex), K=0j, xa=0.0, xs=0.0)
                _gz.update(d=gdi, e=gei, f=gfi)
                g_im = np.real(pk.grad(_gz))
            except Exception:
                g_im = np.zeros(NZ)
        if _G is not None and flags['K'] \
                and flags.get('ctrl_mode', 'full') in ('full', 'sp'):
            gd = np.array([_G[i, i] for i in range(_rr)])
            ge = np.array([_G[i, i + 1] for i in range(_rr - 1)])
            gf = np.array([_G[i + 1, i] for i in range(_rr - 1)])
            gc = pk.zero_grad() if getattr(pk, 'mimo', False) else dict(n=np.zeros(_rr, complex), K=0j, xa=0.0, xs=0.0)
            gc.update(d=gd.real * 0 + gd, e=ge, f=gf)
            try:
                g_ctrl = np.real(pk.grad(gc))
            except Exception:
                g_ctrl = np.zeros(NZ)
        # stability: the 1-D CERTIFICATE supplies the value, a targeted shift
        # the eigenvectors.  alpha_krylov is no longer the authority.
        _red_ab = None
        if oracle is not None:
            # STAGE 5 phase 2: the abscissa from the reduced pencil with the
            # full-model residual filter (no factorisation).  A Ritz value that
            # passes the filter IS an eigenvalue; whether it is the rightmost
            # depends on the watch-list eigenvectors being in the basis, which
            # anchoring maintains and the gate re-certifies.  When nothing
            # passes, fall back to the full certificate and count it.
            _clf = D.cl(b, c)
            _ab = cl.abscissa(arow, nvec, K, _clf, mc=D.mc, res_tol=oracle_res_tol)
            if _ab[-1]:
                al, lamb, vv, uu, _eta_red, _res_red = _ab[0], _ab[1], _ab[2], _ab[3], _ab[4], _ab[5]
                cla = clg; _clr = (al + eps <= 0.0)
                _red_ab = True
                _RED_AB_STATS['reduced'] += 1
            else:
                _RED_AB_STATS['fallback'] += 1
        if _red_ab is None:
            al, lamb, vv, uu, cla, _clr = D.alpha_certified(arow, nvec, K, b, c,
                                                            eps=eps)
        # No eigenvectors this iterate means the targeted shift did not
        # converge.  The certificate still supplied the VALUE, so the
        # constraint is enforced; only its gradient is missing, and a zero
        # slows convergence rather than killing the run.
        if vv is None or uu is None:
            zero = np.zeros(D.n + D.r, complex)
            ga = D.grads(cla, zero, zero, arow, nvec, K, xa, xs,
                         pa, ps, na, ns, flags)
        else:
            eta = _eta_red if _red_ab else np.vdot(uu, cla.Mcl_apply(vv))
            ga = D.grads(cla, vv, uu, arow, nvec, K, xa, xs, pa, ps, na, ns,
                         flags, eta=eta)
        # the constraint is the WORSE of the closed-loop abscissa and the
        # controller's own: a controller that is itself unstable can never give
        # a stable loop here, and unlike alpha it is computed exactly
        # the constraint is the WORSE of the closed-loop abscissa and the
        # controller's own, WITH the matching gradient: changing the value
        # without the gradient hands the solver an inconsistent pair and it
        # simply stalls (observed: every iterate rejected, no progress).
        # An unstable CONTROLLER can legitimately stabilise a plant, so
        # demanding al_ctrl <= -eps is over-restrictive.  What is not
        # negotiable is that its poles stay somewhere the closed-loop check can
        # SEE: shift-invert near the imaginary axis cannot find a pole at
        # Re = 11681.  So the controller is allowed into the right half plane
        # only as far as `ctrl_max`, and the rectangle certificate is sized to
        # cover that far.  ctrl_max = 0 recovers "the controller must itself be
        # stable", which is the safe default.
        # the compensator row.  Frozen: (al_ctrl - ctrl_max) + eps <= 0, i.e.
        # Re <= ctrl_max - eps, stricter than the gate's Re <= ctrl_max (the
        # decision) -- the SQP uses the gate's line, so the two solvers were
        # not solving the same problem (STAGE2 correction).  --ctrl-line gate
        # gives this row the gate's line too; the frozen default is unchanged.
        _cline = ctrl_max if ctrl_line_gate else (ctrl_max - eps)
        if al_ctrl - _cline > al + eps:
            cv[m] = (al_ctrl - _cline)
            J[m, :NZ] = g_ctrl
        else:
            cv[m] = al + eps
            J[m, :NZ] = pk.grad(ga)
        # |Im lambda_ctrl| <= wmax, scaled by wmax so it is O(1) alongside the
        # others.  Keeping the compensator inside the certified band is what
        # makes the band trustworthy: a pole outside it is invisible to
        # sup sigma_1 yet still feeds back inside.
        _k = m + 1
        if _pole_row:
            cv[_k] = (im_ctrl - wmax) / max(wmax, 1e-12)
            J[_k, :NZ] = g_im / max(wmax, 1e-12)
            _k += 1
        if trust:
            # TRUST REGION on the design step.  The inner problem only sees
            # Omega, so left unconstrained it walks to the edge of the
            # surrogate's validity: roughly half of all outer iterations
            # returned points whose CERTIFIED norm was 10-40x the surrogate's t
            # (79.5, 31.2, 36.1, 103.8 in one run), and were wasted.  Bounding
            # the step keeps the iterate where the constraint set is a faithful
            # model.  Radius relative to ||z0||.
            dz = z - z0
            rad = trust * max(float(np.linalg.norm(z0)), 1.0)
            cv[_k] = float(dz @ dz) - rad * rad
            J[_k, :NZ] = 2.0 * dz
        if _E:
            # sigma_2 rows.  Where sigma_1 and sigma_2 (nearly) coincide the
            # constraint function sigma_1 <= t kinks (optimisation note,
            # "Residual kinks"): the two branches have different gradients,
            # BFGS pairs straddling the crossing encode fictitious curvature,
            # and the QP step is wrong until the line search shrinks it to
            # nothing -- measured on K+B+C (gap 0.5 %, balance residual 1.5e4)
            # and KSP+B+C (0.1-0.2 %, 1.7e3).  Naming the second branch as its
            # own constraint makes every branch smooth.  Gradient: the same
            # D.grads on the second singular pair (closedloop.gain_k).
            _k0 = ncon - len(_E)
            for j, i in enumerate(_E):
                # at the FOLLOWED top when the member follows (same frequency as its sigma_1 row)
                pairs = cl.gain_k(float(follow_tops.get(round(float(W[i]), 6), W[i]) if follow else W[i]), arow, nvec, K, k=2)
                s2 = pairs[1]['sigma']
                gr2 = D.grads(clg, pairs[1]['x'], pairs[1]['z'], arow, nvec, K,
                              xa, xs, pa, ps, na, ns, flags)
                cv[_k0 + j] = s2 / t0 - tau
                J[_k0 + j, :NZ] = pk.grad(gr2) / t0
                J[_k0 + j, NZ] = -1.0
        if bandmax and oracle is not None:
            # REFINEMENT IN PYGRANSO'S TERMS (2026-09-22): one extra row, the
            # reduced model's certified maximum of sigma_1 over the whole band,
            # with the gradient at the maximising frequency -- a nonsmooth
            # max-type constraint, which pyGRANSO is built for.  pyGRANSO's
            # constraint set is fixed for a call, so frequencies cannot be
            # ADDED mid-solve as the SQP does; this row makes the band visible
            # instead.  Last row, so no index used elsewhere moves.
            try:
                import scipy.sparse as _sp
                _P = cl.P; _Ac = ctl.A_of(*arow); _Bc = ctl.Bc_of(arow, D.r, D.mc)
                _sr = lambda w_: cl.gain(float(w_), arow, nvec, K, want_vectors=False)['sigma']
                _gb, _wb, _, _okb = _reduced_cert(
                    _sp.csc_matrix(_P['At']), _sp.csc_matrix(_P['Mt']), cl.W.conj().T @ b, cl.V.conj().T @ c,
                    _Ac, _Bc, nvec, K, _sr, gam_lb=max(0.5 * t0, 1e-8), wlo=bandmax[0], whi=bandmax[1])
                if not (_okb and np.isfinite(_wb)):
                    raise ValueError('reduced band certificate inconclusive')
                _res = cl.gain(float(_wb), arow, nvec, K)
                _grb = D.grads(clg, _res['x'], _res['z'], arow, nvec, K, xa, xs, pa, ps, na, ns, flags)
                cv[ncon - 1] = _res['sigma'] / t0 - tau
                J[ncon - 1, :NZ] = pk.grad(_grb) / t0
                J[ncon - 1, NZ] = -1.0
                smax = max(smax, float(_res['sigma']))      # the iterate's cost is judged on the band
            except Exception:
                cv[ncon - 1] = smax / t0 - tau               # fall back to the working-set max (inactive duplicate)
                J[ncon - 1, :NZ] = 0.0; J[ncon - 1, NZ] = -1.0
        out = dict(cv=cv, J=J, tau=tau, smax=smax, al=al)
        # a point is admissible if stable; its true cost is the worst sigma_1
        # `best` must track points that satisfy the EPIGRAPH constraints too,
        # not only stability.  Recording any stable iterate let pyGRANSO's
        # excursions to tau ~ 0 be stored, and `best['t']` then came back as
        # 0.000000 -- meaningless, since every constraint reads
        # sigma_1(theta,w_i) <= t with sigma_1 ~ 9.7.  The exchange's
        # termination test compares the certified norm against that t, so a
        # degenerate value corrupts it.
        # The floor rejects a DEGENERATE t (tau collapsing toward 0 while the
        # design does not improve).  It is a sanity bound, not a constraint on
        # the optimisation: if t is ever reported AT the floor exactly, the
        # inner solve is running away in tau and the floor is masking it rather
        # than fixing it -- that is what happened before tau0 > 1 gave the line
        # search room to take a real step.
        # STAGE 1, item 1.  The frozen test admits an iterate on CLOSED-LOOP
        # feasibility alone (al + eps <= 0), while the inner problem also
        # constrains the compensator abscissa (cv[m], the WORSE of the two)
        # and, when the cap is on, |Im| of its poles (cv[m+1]).  So the
        # returned "best" could be -- and repeatedly was -- an iterate the
        # inner problem itself considers infeasible; the gate then refused it,
        # the exchange restarted from the same point, and the run stalled with
        # a better feasible design left behind.  Measured at ne=100 (KSP+B
        # 3.8 %, KSP+C 4.6 %, KSP+B+C 7.8 %) and ne=200 (KSP+B+C 1.8 %).
        #
        # select_cons=True admits only iterates feasible for EVERY constraint
        # the inner problem enforces (trust row excluded: it bounds the step,
        # not the design).  Off by default until the A/B has been read.
        if select_cons:
            # Admit exactly what the GATE accepts, no more: closed loop at
            # margin, compensator abscissa within the gate's allowance
            # max(-eps, ctrl_max) (certify_sparse.py:367), |Im| under the cap
            # when the cap is on.  A first version tested cv[m] <= 0, i.e. the
            # INNER constraint al_ctrl <= -eps -- eps stricter than the gate --
            # and on K+B+C (no refusals) it discarded acceptable iterates for
            # nothing: 8 rounds gave 9.367 against the frozen 9.046.
            _lim = max(-eps, ctrl_max)
            # Strictly inside the margin, not on it.  The certificate's line
            # sweep proves "no eigenvalue ON Re = -eps"; an iterate with the
            # stability constraint exactly active puts one there and is
            # refused ("eigenvalue at -0.05+5.89j on Re=-0.05", KGC+B at the
            # stage-1 gate).  1e-3 of eps keeps the design inside the line
            # while leaving the constraint effectively active.
            _mg = sel_margin * eps
            # ... AND the certificate's VERDICT, not just its value.
            # alpha_certified returns a_seen (abscissa of the eigenvalues the
            # line sweep saw) and `clear` (the line proven free).  The inner
            # constraint uses the value alone; an eigenvalue that has crossed
            # far to the right is not near the line, so it is not "seen", the
            # value reads feasible, and the iterate is admitted -- then the
            # gate refuses it with the witness (stage-1 gate job 16152993,
            # KSP+B+C from B+C: t=7.90 admitted, certificate alpha +0.754;
            # eight rounds, no progress).  Small steps hid this; maxit 100 at
            # ne=200 does not.
            # the compensator's poles are exact (dense r x r): no cushion.  With
            # one, the optimum -- poles ON the margin -- is inadmissible and the
            # solver returns its start (the 9.749 'fixed point', 2026-09-20)
            _adm = (bool(_clr) and al + eps <= -_mg and al_ctrl <= _lim + 1e-8
                    and (not _pole_row or cv[m + 1] <= 0.0))
        else:
            _adm = al + eps <= 0
        if _adm and smax < best['g'] and tau * t0 >= 0.5 * smax:
            best.update(g=smax, z=z.copy(), al=al, t=tau * t0, clr=bool(_clr))
        cache[key] = out
        if len(cache) > 1500:
            cache.clear()
        return out

    class _Obj(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v):
            ev(v.detach().numpy().ravel())
            return torch.tensor(float(v[NZ, 0]))
        @staticmethod
        def backward(ctx, go):
            g = torch.zeros(NZ + 1, dtype=torch.double); g[NZ] = 1.0
            return go * g.view(-1, 1)

    class _Cons(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v):
            r = ev(v.detach().numpy().ravel())
            ctx.save_for_backward(torch.tensor(r['J']))
            return torch.tensor(r['cv'].reshape(-1, 1))
        @staticmethod
        def backward(ctx, go):
            (J,) = ctx.saved_tensors
            return (J.transpose(0, 1) @ go.view(-1, 1)).view(-1, 1)

    def comb(X):
        f = _Obj.apply(X.v)
        ci = pygransoStruct(); ci.c1 = _Cons.apply(X.v)
        return f, ci, None

    # START WITH SLACK IN t.
    #
    # t0 = max_i sigma_1(theta,w_i), so tau = 1 puts the iterate EXACTLY on the
    # boundary of the binding constraint: max cv = +0.0000.  The objective is
    # f = tau with gradient (0,...,0,1), so the first trial step is a unit step
    # in tau, which is immediately infeasible -- pyGRANSO reported
    #
    #     Iterations: 0,  Function evaluations: 41
    #     termination code 6: line search bracketed a minimizer but failed to
    #     satisfy Wolfe conditions at a (strictly) feasible point
    #
    # having moved tau by 1e-6 before violating.  Every outer iteration cost 41
    # evaluations and achieved nothing, and the exchange then "converged" on the
    # starting design.
    #
    # Starting slightly above the peak gives the line search room: tau can fall
    # to 1 before any constraint activates, which is enough for BFGS to take a
    # real step and start accumulating curvature.
    v0 = np.concatenate([z0, [tau0 if level_norm else tau0 * t_level]])
    o = pygransoStruct()
    o.QPsolver = 'osqp'; o.print_level = 0
    o.x0 = torch.tensor(v0.reshape(-1, 1))
    # STAGE 1, item 2.  1e-10 is the frozen value.  Eleven of twelve inner
    # solves on KSP+B+C (ne=200) ended on code 6 -- Wolfe bracket failure --
    # at stat_value 1e-5..2e-5, never near 1e-10: the tolerance is below what
    # the eigenvalue oracle resolves, so the solve grinds to a line-search
    # failure instead of stopping at stationarity.  Opt-in; measure first.
    o.opt_tol = float(opt_tol)
    # ---- STAGE 1b, note item 0: read mu and the QP status off the run ------
    # pyGRANSO exposes mu only as `mu_lowest` on failure, and never reads
    # OSQP's `res.info.status` (solveQP.py:130 uses res.x regardless of whether
    # OSQP reported 'solved', 'solved_inaccurate' or 'maximum iterations
    # reached').  Both are captured here without touching pyGRANSO's code:
    # a halt_log_fn for the per-iterate mu / violation / stationarity, and a
    # wrapper on osqp.OSQP.solve for the status histogram.
    _qplog = dict(it=[], mu=[], tv=[], f=[], feas=[], stat=[], alpha=[],
                  ls=[], fb=[], osqp=[])
    if qp_log:
        def _hlf(it, x, pf, d, bfgs_state, Hreg, ls_evals, alpha, ngrad,
                 stat_vec, stat_val, fallback):
            try:
                _qplog['it'].append(int(it))
                _qplog['mu'].append(float(getattr(pf, 'mu', np.nan)))
                _qplog['tv'].append(float(getattr(pf, 'tv', np.nan)))
                _qplog['f'].append(float(getattr(pf, 'f', np.nan)))
                _qplog['feas'].append(bool(getattr(pf, 'feasible_to_tol', False)))
                _qplog['stat'].append(float(stat_val) if stat_val is not None else np.nan)
                _qplog['alpha'].append(float(alpha) if alpha is not None else np.nan)
                _qplog['ls'].append(int(ls_evals) if ls_evals is not None else 0)
                _qplog['fb'].append(int(fallback) if fallback is not None else 0)
            except Exception:
                pass
            return False
        o.halt_log_fn = _hlf
        try:
            import osqp as _osqp
            if not getattr(_osqp.OSQP, '_codesign_wrapped', False):
                _orig_solve = _osqp.OSQP.solve
                def _solve_logged(self, *a_, **k_):
                    res = _orig_solve(self, *a_, **k_)
                    try:
                        _OSQP_STATUS.append(str(res.info.status))
                    except Exception:
                        pass
                    return res
                _osqp.OSQP.solve = _solve_logged
                _osqp.OSQP._codesign_wrapped = True
        except Exception:
            pass
        _OSQP_STATUS.clear()
    if qp_eps is not None or qp_maxiter is not None:
        # ---- item 0, the intervention: pyGRANSO hardcodes eps_abs = eps_rel
        # = 1e-12 with OSQP's default max_iter (4000); measured, 33 of 39
        # steering QPs on the K+B+C polish and 35 of 45 on case B end on
        # 'maximum iterations reached' and the unconverged answer is used.
        # Override the setup kwargs without touching pyGRANSO.
        try:
            import osqp as _osqp
            if not getattr(_osqp.OSQP, '_codesign_setup_wrapped', False):
                _orig_setup = _osqp.OSQP.setup
                def _setup_override(self, *a_, **k_):
                    if _QP_OVERRIDE.get('eps') is not None:
                        k_['eps_abs'] = _QP_OVERRIDE['eps']; k_['eps_rel'] = _QP_OVERRIDE['eps']
                    if _QP_OVERRIDE.get('maxiter') is not None:
                        k_['max_iter'] = int(_QP_OVERRIDE['maxiter'])
                    return _orig_setup(self, *a_, **k_)
                _osqp.OSQP.setup = _setup_override
                _osqp.OSQP._codesign_setup_wrapped = True
        except Exception:
            pass
        _QP_OVERRIDE['eps'] = qp_eps; _QP_OVERRIDE['maxiter'] = qp_maxiter
    if qp_solver == 'daqp':
        # ---- item 0, the active-set alternative.  qpOASES has no pip
        # package; DAQP (Arnstrom-Bemporad-Axehill 2022) is a dual active-set QP
        # solver with the same role.  pyGRANSO's solveQP builds
        #   min 0.5 y'Hy + f'y  s.t.  LB <= A y <= UB
        # with equality rows as LB = UB and an identity block for the bounds;
        # that is DAQP's native form.  The replacement is bound into the two
        # pyGRANSO modules that imported solveQP by name.
        import pygranso.private.qpSteeringStrategy as _qss
        import pygranso.private.qpTerminationCondition as _qtc
        import daqp as _daqp
        import scipy.sparse as _sp

        def _solveQP_daqp(H, f, A, b, LB, UB, QPsolver, device, double_precision, *rest):
            _t = lambda v: (v.detach().cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v))
            Hn = np.ascontiguousarray(_t(H), dtype=np.float64)
            Hn = 0.5 * (Hn + Hn.T)
            fn = np.ascontiguousarray(_t(f).reshape(-1), dtype=np.float64)
            nvar = Hn.shape[0]
            lb = np.ascontiguousarray(_t(LB).reshape(-1), dtype=np.float64)
            ub = np.ascontiguousarray(_t(UB).reshape(-1), dtype=np.float64)
            if A is not None and b is not None:
                An = _t(A); An = An.toarray() if _sp.issparse(An) else np.asarray(An)
                An = np.ascontiguousarray(An, dtype=np.float64)
                bn = np.ascontiguousarray(_t(b).reshape(-1), dtype=np.float64)
                Afull = np.vstack([An, np.eye(nvar)])
                bup = np.concatenate([bn, ub]); blo = np.concatenate([bn, lb])
                sense = np.concatenate([5 * np.ones(len(bn), dtype=np.int32),
                                        np.zeros(nvar, dtype=np.int32)])
            else:
                Afull = np.eye(nvar); bup = ub; blo = lb
                sense = np.zeros(nvar, dtype=np.int32)
            x, fval, flag, info = _daqp.solve(Hn, fn, Afull, bup, blo, sense)
            _OSQP_STATUS.append(f"daqp:{int(flag)}")
            y = torch.tensor(np.asarray(x, dtype=np.float64).reshape(-1, 1))
            return y
        _qss.solveQP = _solveQP_daqp
        _qtc.solveQP = _solveQP_daqp
    # pyGRANSO's DEFAULT is halt_on_linesearch_bracket = True: the first time a
    # line search brackets a minimizer but cannot satisfy the Wolfe conditions
    # at a feasible point, the inner solve stops with termination code 6 --
    # without consulting the fallback ladder (degraded steering, plain BFGS,
    # steepest descent) that pyGRANSO carries for exactly that situation.
    #
    # Measured (job 16047382, KSP+B+C, eps 0.05, maxit 200): eleven of twelve
    # inner solves ended on code 6, at 39 to 128 iterations, stat_value ~1e-5
    # against opt_tol 1e-10.  With an eigenvalue oracle accurate to ~1e-8, a
    # Wolfe bracket failure near the noise floor is close to guaranteed, so the
    # inner problem is never solved to stationarity -- which is why balance
    # residuals are large across the suite.
    #
    # Opt-in so the frozen behaviour is untouched by default.  Stage-1 item 0.
    if ls_fallback:
        o.halt_on_linesearch_bracket = False
    if budget == 'iters':
        o.maxit = int(maxit)
        # guard only: large enough that it should never bind, finite so a
        # pathological inner cannot eat the whole allocation
        o.maxclocktime = max(float(wall) * float(clock_guard), 600.)
    else:
        o.maxit = 5000
        o.maxclocktime = max(wall, 10.)
    def _is_pd(Hm):
        # pyGRANSO validates H0 and ABORTS if it is not positive definite.  The
        # BFGS matrix carried over from the previous exchange step can lose
        # definiteness (a rejected step, a near-singular update), and the run
        # then dies mid-way -- KG+FBR+FCR crashed exactly there after one good
        # outer iteration.  Check before handing it over; a fresh H0 merely
        # costs curvature, whereas an invalid one costs the whole run.
        try:
            Hs = 0.5 * (Hm + Hm.T)
            return (np.all(np.isfinite(Hs))
                    and np.min(np.linalg.eigvalsh(Hs)) > 1e-10)
        except Exception:
            return False
    if H0 is not None and H0.shape[0] == NZ + 1 and _is_pd(H0):
        # WARM-START THE HESSIAN.  Each exchange step is a fresh pyGRANSO solve,
        # so by default the BFGS approximation is rebuilt from scratch 8-20
        # times per run -- which is where the epigraph's extra oracle calls go
        # (2-4x the ladder's).  Passing the previous step's final H recovers
        # that curvature; the design has barely moved and only a few constraints
        # were added.
        import torch as _t
        o.H0 = _t.tensor(0.5 * (H0 + H0.T)); o.scaleH0 = False
    elif H0mode == 'calibrated':
        r0 = ev(v0.copy())
        gn = float(np.linalg.norm(r0['J']))
        cH = max(gn / max(0.10 * max(np.linalg.norm(v0), 1.0), 1e-9), 1.0)
        o.H0 = torch.eye(NZ + 1, dtype=torch.double) * cH
        o.scaleH0 = False
    # 'auto' leaves pyGRANSO's own defaults (H0 = I, scaleH0 = True): with a
    # LINEAR objective there is no curvature to mis-scale, which is the point.
    Hout = None
    stats = {}
    try:
        soln = pygranso(var_spec={"v": [NZ + 1, 1]}, combined_fn=comb,
                        user_opts=o)
        Hf = getattr(getattr(soln, 'H_final', None), 'detach', None)
        if Hf is not None:
            Hout = soln.H_final.detach().cpu().numpy()
        # STAGE 0 INSTRUMENTATION.  pyGRANSO already reports the quantities the
        # harness needs and they were being thrown away with `soln`: the inner
        # ITERATION count (distinct from len(cache), which counts distinct
        # design evaluations), the termination code that tells "solved" from
        # "gave up" -- report v7 section 7, P6, the failure the exchange's own
        # test cannot see -- the stationarity measure, and the QP failure rate,
        # which is the direct read-out for the OSQP-accuracy item (optimisation
        # note item 0) that stage 1 opens with.
        def _num(x):
            try:
                return float(x.detach().cpu().item())
            except Exception:
                try:
                    return float(x)
                except Exception:
                    return None
        stats = dict(
            iters=getattr(soln, 'iters', None),
            fn_evals=getattr(soln, 'fn_evals', None),
            termination_code=getattr(soln, 'termination_code', None),
            stat_value=_num(getattr(soln, 'stat_value', None)),
            qp_log=(None if not qp_log else dict(
                n_iter=len(_qplog['it']),
                mu_first=(_qplog['mu'][0] if _qplog['mu'] else None),
                mu_min=(float(np.nanmin(_qplog['mu'])) if _qplog['mu'] else None),
                mu_final=(_qplog['mu'][-1] if _qplog['mu'] else None),
                tv_final=(_qplog['tv'][-1] if _qplog['tv'] else None),
                feas_final=(_qplog['feas'][-1] if _qplog['feas'] else None),
                stat_final=(_qplog['stat'][-1] if _qplog['stat'] else None),
                fallback_max=(max(_qplog['fb']) if _qplog['fb'] else None),
                osqp_status={st: _OSQP_STATUS.count(st) for st in sorted(set(_OSQP_STATUS))},
                # mu-regrow detector (note, "Driving the parameter"): mu driven
                # far down by steering and never raised again starves the
                # objective; feasible-and-stalled with a tiny mu is the signature
                mu_starved=bool(_qplog['mu'] and _qplog['mu'][-1] < 1e-4
                                and _qplog['feas'] and _qplog['feas'][-1]),
                trace=dict(it=_qplog['it'], mu=_qplog['mu'], tv=_qplog['tv'],
                           stat=_qplog['stat'], alpha=_qplog['alpha'], fb=_qplog['fb']))),
            BFGS_updates=_num(getattr(soln, 'BFGS_updates', None)),
            quadprog_failure_rate=_num(
                getattr(soln, 'quadprog_failure_rate', None)),
            budget=budget,
            maxit=int(o.maxit),
        )
        # pyGRANSO code 5 is "clock/wall time limit exceeded".  Under
        # budget='iters' that means the guard bound, i.e. the clock decided
        # after all and THIS RUN IS NOT REPRODUCIBLE.  Say so where it will be
        # seen, and record it so the harness can refuse to compare the row.
        if stats['termination_code'] == 12:
            # pyGRANSO code 12 = "qp failure".  Worth its own message because
            # it is NOT a budget outcome and the exchange cannot tell the
            # difference: collect.py reports any stall as "inner solver at its
            # limit", which is a hardcoded guess, and on code 12 it is wrong.
            #
            # What happens: OSQP fails to factor the steering KKT matrix
            # (error 4, "non-convex" -- H not numerically PSD),
            # solveSteeringDualQP's except block prints and then falls through
            # to use an unassigned `y`, and the resulting UnboundLocalError
            # makes bfgssqp call prepareTermination(12) and return at once.
            # That skips pyGRANSO's own fallback ladder (degraded steering ->
            # BFGS -> steepest descent) sitting immediately below it, so the
            # inner solve does not degrade, it stops.
            #
            # Consequence for this round: the returned point is the best
            # iterate seen BEFORE the failure, at an arbitrary place in the
            # inner solve.  Not wrong, but not converged either, and the outer
            # round's call count understates the work the round needed.
            stats['qp_failure'] = True
            print(f"    WARNING: inner ABORTED on a quadprog failure "
                  f"(pyGRANSO code 12), not on its budget.  Returning the "
                  f"best iterate found before the failure.", flush=True)
        else:
            stats['qp_failure'] = False
        if budget == 'iters' and stats['termination_code'] == 5:
            stats['clock_guard_bound'] = True
            # Measured (job 15975493, n011, K+B+C ne=200): four rounds hit
            # this guard and all three repeats still came back bit-identical.
            # A clock cut is reproducible when it lands on the same ITERATION
            # every time, which it does once per-iteration cost is stable --
            # exclusive node, warm pool.  It is unreproducible when the cut
            # lands mid-iteration, which is what ne=100 under a stressed 10s
            # limit was doing.  So this is a caution, not a verdict, and it
            # says nothing about reproducing on DIFFERENT hardware.
            print(f"    NOTE: inner hit the CLOCK GUARD "
                  f"({o.maxclocktime:.0f}s) under --inner-budget iters, so "
                  f"this round stopped on time, not on --inner-maxit.  It may "
                  f"still reproduce on this node; it is less likely to on "
                  f"another.  Raise --wall or lower --inner-maxit to remove "
                  f"the doubt.", flush=True)
        else:
            stats['clock_guard_bound'] = False
    except Exception as e:
        stats = dict(error=f"{type(e).__name__}: {str(e)[:80]}")
        if verbose:
            print(f"    pygranso: {type(e).__name__}: {str(e)[:60]}", flush=True)
    return best, len(cache), Hout, stats


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
_OSQP_STATUS: list = []   # filled by the osqp wrapper when --qp-log is on
_RED_AB_STATS: dict = {'reduced': 0, 'fallback': 0}   # stage 5 phase 2: abscissa source per evaluation
_QP_OVERRIDE: dict = {}   # eps / max_iter overrides for OSQP (--qp-eps, --qp-maxiter)


def _input_fingerprint(ne: int, init_path: str | None) -> dict:
    """sha256 of the two files that fix the cold start.

    Why this is in the record.  Job 15969703 diverged from every later run of
    the same case on the same node, from outer 0, with an identical printed
    start (`start gamma=10.758915`, |Omega|=92, anchor alpha=-0.2261).  It was
    the only run that did not build init_ne200.npz fresh.  Nothing in the
    record said which init had been used, so the difference could only be
    guessed at after the fact -- and the first guess was the hardware, which
    cost two jobs to rule out.

    init_ne{ne}.npz comes out of a dense balanced truncation, the most
    BLAS-sensitive step in the pipeline.  A last-bit difference in d, e, f is
    invisible in a start gamma printed to six digits and still sends the inner
    solve down a different path.  seeds_ne{ne}.npz sets the controller order
    and every warm start.  Neither is under version control; both have been
    rebuilt by hand at least once.

    Hashing costs microseconds and makes "same inputs?" answerable from the
    record instead of from memory.

    Why the FILE hash and not a hash of the array contents.  .npz is a zip, and
    zip entries carry a modification time, so the obvious worry is that two
    byte-identical rebuilds would hash differently and raise a false alarm.
    Checked: numpy pins every entry's date_time to (1980, 1, 1, 0, 0, 0), so
    savez output is byte-reproducible -- two make_init runs five seconds apart
    give the same sha256.  The file hash is therefore a content fingerprint and
    the extra machinery is not needed.

    That is an implementation detail of numpy, not a promise.  If a future
    version stops pinning it, or if anything switches to savez_compressed with
    a different compression level, this field starts reporting differences that
    are not differences.  The symptom would be every run disagreeing with every
    other; the fix would be to hash sorted (key, dtype, shape, tobytes())
    instead.
    """
    import hashlib
    out = {}
    for label, fn in (('init', init_path or f"init_ne{ne}.npz"),
                      ('seeds', f"seeds_ne{ne}.npz")):
        try:
            with open(fn, 'rb') as fh:
                out[label] = hashlib.sha256(fh.read()).hexdigest()[:16]
            out[label + '_file'] = os.path.basename(fn)
        except OSError:
            out[label] = None
    return out


def dump_outer(path, it, D, pk, flags, z, Omega, t, gamma_cert, gamma_best,
               eps, par=None, inner_stats=None, lu=None, gate=None,
               extra=None):
    """Write one outer round's diagnostic state.  Optimisation note item 10.

    Per outer round: Omega, sigma_1 on Omega, t, the multipliers, and the
    closed-loop eigenvalues -- the four things one wants when asking why an
    exchange stalled, and none of which the result record keeps.  plot_omega.py
    reads what this writes.

    Two design points.

    The MULTIPLIERS are not pyGRANSO's QP multipliers.  Those belong to the
    restricted problem in the optimiser's own scaled coordinates and are gone
    by the time the outer loop sees anything.  What is wanted here is the
    exchange-level statement: at a balanced optimum the active peaks' gradients
    must have zero in their convex hull, so the multipliers are the nonnegative
    weights lambda_i, sum 1, minimising ||sum_i lambda_i grad sigma_1(w_i)||
    over the active set.  That residual, relative to the mean gradient norm, IS
    the balance residual the stage-1 gate asks to drive from ~1e4 to O(10), and
    it is computed from the same gradients check_optimality reports.

    The cost is one evaluation of sigma_1 and its design gradient at every
    frequency in Omega -- about one inner iteration per outer round, on the
    same worker pool.  That is why this is behind --dump rather than always on.
    """
    import numpy as _np
    rec = dict(outer=int(it), t=float(t), gamma_certified=float(gamma_cert),
               gamma_best=float(gamma_best), eps=float(eps),
               nOmega=int(len(Omega)))
    W = _np.asarray(sorted(Omega), float)

    arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
    b, c, pa, ps, na, ns = D.bc(xa, xs, va, vs)

    # ---- sigma_1 on Omega, with design gradients (one pass) ----
    sig = _np.full(len(W), _np.nan)
    G = None
    try:
        if par is not None:
            sig, G = par.constraints(z, W)
            sig = _np.asarray(sig, float)
            G = _np.asarray(G, float)
        else:
            # the dump's sigma rows on the REDUCED model when one is attached:
            # on the full oracle this diagnostic cost |Omega| factorisations a
            # round (167 of 374 at ne=200) -- as much as the method itself
            _ro_d = getattr(certified_norm, 'oracle', None)
            cl = _ro_d.make_cl(b, c, D.r, D.mc) if (_ro_d is not None and _ro_d.dim > 0) else D.cl(b, c)
            clg_d = getattr(cl, 'proxy', cl)          # D.grads needs the reduced model's proxy
            rows = []
            for w in W:
                res = cl.gain(float(w), arow, nvec, K)
                gr = D.grads(clg_d, res['x'], res['z'], arow, nvec, K, xa, xs,
                             pa, ps, na, ns, flags)
                rows.append(pk.grad(gr))
                sig[len(rows) - 1] = res['sigma']
            G = _np.asarray(rows, float)
    except Exception as e:
        rec['sigma_error'] = f"{type(e).__name__}: {str(e)[:80]}"

    # ---- active set and balance multipliers ----
    lam = _np.zeros(0)
    bal = None
    act = _np.zeros(0, int)
    if G is not None and _np.all(_np.isfinite(sig)):
        smax = float(_np.max(sig))
        # active = within 1% of the peak; the frozen optima show ties at
        # 0.18%-1.7%, so a tighter window would call every stall single-active
        act = _np.flatnonzero(sig >= smax * (1.0 - 1e-2))
        if act.size:
            Ga = G[act]
            try:
                from scipy.optimize import nnls
                # min ||Ga^T lam|| s.t. lam >= 0, sum lam = 1, by appending the
                # normalisation as a heavily weighted row
                wgt = float(_np.linalg.norm(Ga)) + 1.0
                A = _np.vstack([Ga.T, wgt * _np.ones((1, act.size))])
                rhs = _np.zeros(A.shape[0])
                rhs[-1] = wgt
                lam, _ = nnls(A, rhs)
                s = float(lam.sum())
                if s > 0:
                    lam = lam / s
                resid = float(_np.linalg.norm(Ga.T @ lam))
                # Normalised EXACTLY as check_optimality's rel |grad|:
                # ||g|| ||theta|| / gamma, dimensionless.  With one active
                # frequency lam = [1] and this reduces to that tool's number,
                # so the dump and the diagnostic report the same quantity and
                # the stage-1 gate ("~1e4 toward O(10)") reads off either.
                # Normalising by the mean gradient norm instead would return
                # 1.0 for every single-active round, which says nothing.
                znorm = max(float(_np.linalg.norm(_np.asarray(z, float))),
                            1e-30)
                bal = resid * znorm / max(smax, 1e-30)
                rec['balance_resid_raw'] = resid
                rec['balance_resid_hull'] = resid / (
                    float(_np.mean(_np.linalg.norm(Ga, axis=1))) or 1.0)
            except Exception as e:
                rec['multiplier_error'] = f"{type(e).__name__}: {str(e)[:60]}"

    # ---- closed-loop eigenvalues ----
    # To watch eigenvalues MOVE from round to round (which pair approaches the
    # line, whether a stall is pinned by one mode or several).  SPARSE since
    # 2026-09-29 (STAGE6 6.13; was the dense spectrum up to n + r = 4000): the
    # eigenvalues the last band computation located (the far ones and the band's
    # edge strips) and the ones the abscissa certificate harvested near the line
    # -- a diagnostic, never a gate.
    eig = _np.zeros(0, complex)
    try:
        _lam = list(getattr(D, '_band_last_lam', []) or [])
        _seen = []
        D.alpha_certified(arow, nvec, K, b, c, eps=eps, eigs_seen=_seen)
        for _z in _seen:
            if all(abs(complex(_z) - _q) > 1e-8 * max(1.0, abs(_z)) for _q in _lam):
                _lam.append(complex(_z))
        eig = _np.asarray(_lam, complex)
    except Exception as e:
        rec['eig_error'] = f"{type(e).__name__}: {str(e)[:60]}"

    # Whether the pool was active decides how the LU counts must be read: the
    # workers factorise in their OWN processes, which this counter cannot see,
    # so a pooled run reports far fewer factorisations than it performs.  The
    # counts are a per-process quantity and the flag says which process.
    rec.update(pool_active=bool(par is not None),
               sigma_max=(float(_np.nanmax(sig)) if sig.size else None),
               n_active=int(act.size),
               balance_residual=(float(bal) if bal is not None else None),
               inner=dict(inner_stats or {}),
               lu=dict(lu or {}),
               gate=dict(gate or {}))
    if extra:
        rec.update(extra)

    os.makedirs(path, exist_ok=True)
    stem = os.path.join(path, f"outer_{it:03d}")
    _np.savez(stem + ".npz", omega=W, sigma=sig, active=act,
              multipliers=lam, eigs=eig, z=_np.asarray(z, float),
              design_d=arow[0], design_e=arow[1], design_f=arow[2], design_nvec=nvec, design_K=K,
              design_xa=xa, design_xs=xs, design_va=(va if va is not None else _np.zeros(0)), design_vs=(vs if vs is not None else _np.zeros(0)))   # self-contained: z is relative to the process's packer
    json.dump(rec, open(stem + ".json", "w"), indent=1, default=str)
    print(f"    dump: {stem}.npz  (|Omega|={len(W)}, {act.size} active, "
          f"balance residual "
          f"{('%.3e' % bal) if bal is not None else 'n/a'})", flush=True)
    return rec


# --------------------------------------------------------------------------
def _enrol_peaks(wgrid, gvals, sig_fn=None, rel=0.0, cap=16):
    """The certifier's peak list.  Frozen (rel = 0): the top 4 local maxima on
    the grid.  rel > 0 (2026-09-22): EVERY local maximum within rel of the
    maximum (up to `cap`), each located by golden-section search on sig_fn
    (the reduced model: free) so the Omega member sits on the peak's TOP --
    both refused rounds of the fair SQP arm were near-maximal narrow peaks
    (97 % and 96 % of the max before the step) outside Omega, 0.11 and 0.04
    from the nearest member."""
    wgrid = np.asarray(wgrid, float); gvals = np.asarray(gvals, float)
    i = np.arange(1, len(gvals) - 1)
    loc = i[(gvals[i] >= gvals[i - 1]) & (gvals[i] >= gvals[i + 1])]
    loc = loc[np.argsort(-gvals[loc])]
    if rel <= 0.0:
        return [float(wgrid[k]) for k in loc[:4]]
    top = float(gvals.max())
    loc = [k for k in loc if gvals[k] >= (1.0 - rel) * top][:cap]
    out = []
    for k in loc:
        if sig_fn is None:
            out.append(float(wgrid[k])); continue
        lo, hi = float(wgrid[k - 1]), float(wgrid[k + 1]); g = 0.618
        x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo); f1, f2 = sig_fn(x1), sig_fn(x2)
        for _ in range(24):
            if f1 > f2:
                hi, x2, f2 = x2, x1, f1; x1 = hi - g * (hi - lo); f1 = sig_fn(x1)
            else:
                lo, x1, f1 = x1, x2, f2; x2 = lo + g * (hi - lo); f2 = sig_fn(x2)
        out.append(float(x1 if f1 > f2 else x2))
    return out


def _reduced_cert(*args, **kw):
    """Certificate on a REDUCED pencil: the dense level-set algorithm by default
    (2026-09-25: the sparse shift-invert decision procedure could certify a
    maximum the reduced model itself exceeded; for a pencil of dimension ~200
    all eigenvalues cost milliseconds, so no crossing can be missed)."""
    if REDUCED_CERT == 'sparse':
        return csp.hinf_certified_sparse(*args, **kw)
    kw.pop('k', None); kw.pop('eps_eigs', None); kw.pop('verbose', None)
    return csp.hinf_dense_levelset(*args, **kw)


REDUCED_CERT = 'dense'


def certified_norm(D, pk, z, wlo, whi, gam_lb, w_hint=None):
    """Certified max_omega sigma_1 and a witness frequency, for the outer loop."""
    arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
    b, c, *_ = D.bc(xa, xs, va, vs)
    cl = D.cl(b, c)
    sig = lambda w: cl.gain(w, arow, nvec, K, want_vectors=False)['sigma']
    Ac = ctl.A_of(*arow); Bc = ctl.Bc_of(arow, D.r, D.mc)
    # PROVISIONAL rounds return here, BEFORE the full certificate (2026-09-24:
    # placed after it, provisional rounds paid for the full level-set
    # certificate -- its heaviest, most sequential part -- and discarded it)
    ro_ = getattr(certified_norm, 'oracle', None)
    if getattr(certified_norm, 'provisional', False) and ro_ is not None and ro_.dim > 0:
        # PROVISIONAL round (--cert-every): the reduced model's own level-set
        # certificate and fine sweep, no full-model work.  Exact for every
        # peak the basis represents; the full certificate every k-th round
        # catches what it cannot, and the chain rolls back if it disagrees.
        rc_ = ro_.make_cl(b, c, D.r, D.mc); P_ = rc_.P
        sigr = lambda w_: rc_.gain(float(w_), arow, nvec, K, want_vectors=False)['sigma']
        import scipy.sparse as _sp
        try:
            import lu_counter as _lc; _lc.REDUCED_CONTEXT = True      # reduced-size work, not 'full'
        except Exception:
            _lc = None
        if not np.isfinite(gam_lb) or gam_lb <= 0:
            # a strictly positive lower bound from the reduced model itself
            gam_lb = 0.99 * max(sigr(x) for x in np.linspace(wlo, whi, 241))
        gr, wr, _, okr = _reduced_cert(_sp.csc_matrix(P_['At']), _sp.csc_matrix(P_['Mt']),
                                                   rc_.W.conj().T @ b, rc_.V.conj().T @ c, Ac, Bc, nvec, K, sigr,
                                                   gam_lb=max(gam_lb, 1e-8), wlo=wlo, whi=whi)
        if _lc is not None:
            _lc.REDUCED_CONTEXT = False
        wf = np.linspace(wlo, whi, 1201); gf = np.array([sigr(x) for x in wf])
        i = np.arange(1, len(gf) - 1); loc = i[(gf[i] >= gf[i - 1]) & (gf[i] >= gf[i + 1])]
        peaks = _enrol_peaks(wf, gf, sigr, rel=float(getattr(certified_norm, 'enroll_rel', 0.0)))
        certified_norm.last_provisional = True
        return float(gr), (float(wr) if np.isfinite(wr) else float(wf[gf.argmax()])), bool(okr), peaks
    # the certificate needs a strictly positive lower bound; use the cheap
    # seeded search when the caller has none (first outer iteration)
    if not np.isfinite(gam_lb) or gam_lb <= 0:
        ro0 = getattr(certified_norm, 'oracle', None)
        if ro0 is not None and ro0.dim > 0 and getattr(ro0, 'galerkin', False):
            # the position-robust reduced model's maximum (exact to ~1e-8):
            # 0.99 x it is a safe strictly positive lower bound, for free
            # (D.hinf cost ~12 full factorisations)
            rc0 = ro0.make_cl(b, c, D.r, D.mc)
            gam_lb = 0.99 * max(rc0.gain(float(x), arow, nvec, K, want_vectors=False)['sigma']
                                for x in np.linspace(wlo, whi, 241))
        else:
            gam_lb = float(D.hinf(arow, nvec, K, b, c)[0])
    g, w, ncall, ok = csp.hinf_certified_sparse(
        D.A, D.M, b, c, Ac, Bc, nvec, K, sig, gam_lb=max(gam_lb, 1e-8),
        wlo=wlo, whi=whi, rtol=float(getattr(certified_norm, 'rtol', 1e-4)))
    # ALL FREQUENCIES (STAGE6 6.21, --cert-all-w): the band is a heuristic; the
    # tail bound W (numerical range, sparse_spectrum.closed_loop_tail) proves
    # sigma_1 < g beyond W, and one more level-set question on [-W, W] at g
    # proves nothing exceeds g inside -- g is then the norm over ALL w.  A peak
    # found outside the band is returned (and the caller widens the band).
    certified_norm.last_all_w = None
    _below = getattr(certified_norm, 'all_w_below', None)     # only a value that would be accepted needs it
    if getattr(certified_norm, 'all_w', False) and ok and np.isfinite(g) and g > 0 \
            and (_below is None or g < _below):
        try:
            import sparse_spectrum as _ss
            Wt, _ = _ss.closed_loop_tail(D.A, D.M, b, c, Ac, Bc, nvec, K, float(g))
            aw = dict(W=float(Wt), band=float(max(whi, -wlo)))
            if np.isfinite(Wt) and Wt > max(whi, -wlo):
                g2, w2, nc2, ok2 = csp.hinf_certified_sparse(
                    D.A, D.M, b, c, Ac, Bc, nvec, K, sig, gam_lb=float(g),
                    wlo=-Wt, whi=Wt, rtol=float(getattr(certified_norm, 'rtol', 1e-4)))
                aw.update(g=float(g2), w=float(w2), calls=int(nc2), ok=bool(ok2))
                if ok2 and g2 > g * (1.0 + 1e-9):
                    g, w = g2, w2
                ok = ok and ok2
            elif not np.isfinite(Wt):
                aw.update(ok=False)
            certified_norm.last_all_w = aw
        except Exception as _e:
            certified_norm.last_all_w = dict(error=f"{type(_e).__name__}: {str(_e)[:60]}")
    # collect the local peaks too: they are the frequencies most likely to
    # become active next, so seeding them saves outer iterations
    # coarse-then-fine (STAGE 5 4.2): the full sweep is the truth reference
    # for the reduced model's anchoring and can be coarse (`cert_sweep_n`
    # points; frozen 241); the extra-peaks list comes from a FINE sweep on
    # the reduced model when one is attached (free), else from the full one.
    nsw = int(getattr(certified_norm, 'sweep_n', 241))
    certified_norm.last_provisional = False
    _peak_ref = nsw <= 0 and ro_ is not None and ro_.dim > 0
    if nsw <= 0 and not _peak_ref:
        nsw = 241                      # no reduced model yet: the frozen reference
    if not _peak_ref:
        ws = np.linspace(wlo, whi, nsw)
        gs = np.array([sig(x) for x in ws])
    if ro_ is not None and ro_.dim > 0:
        rc_ = ro_.make_cl(b, c, D.r, D.mc)
        wf = np.linspace(wlo, whi, 1201)
        gf = np.array([rc_.gain(float(x), arow, nvec, K, want_vectors=False)['sigma'] for x in wf])
        i = np.arange(1, len(gf) - 1)
        loc = i[(gf[i] >= gf[i - 1]) & (gf[i] >= gf[i + 1])]
        _sr2 = lambda w_: rc_.gain(float(w_), arow, nvec, K, want_vectors=False)['sigma']
        peaks = _enrol_peaks(wf, gf, _sr2, rel=float(getattr(certified_norm, 'enroll_rel', 0.0)))
        if _peak_ref:
            # PEAK REFERENCE (--cert-sweep 0; 2026-09-26, STAGE5 30): the full
            # model is sampled only where the reduced model puts its peaks (the
            # enrolled ones, the fine-sweep argmax, the 4 highest local maxima),
            # not on a 61-point grid: the certificate is the proof, the sample
            # only checks the model where the H-infinity norm lives and drives
            # the disagreement anchoring.  ~50 fewer full factorisations a round.
            _top = [float(wf[j]) for j in loc[np.argsort(-gf[loc])][:4]] if loc.size else []
            ws = np.array(sorted(set([round(float(x), 9) for x in list(peaks)[:12] + _top + [float(wf[int(np.argmax(gf))])]])))
            gs = np.array([sig(x) for x in ws])
    else:
        i = np.arange(1, len(gs) - 1)
        loc = i[(gs[i] >= gs[i - 1]) & (gs[i] >= gs[i + 1])]
        peaks = _enrol_peaks(ws, gs, None, rel=float(getattr(certified_norm, 'enroll_rel', 0.0)))
    # STAGE 5: the sweep is 241 full-model evaluations already paid for; the
    # projected oracle compares itself against it to decide where to anchor
    certified_norm.last_sweep = (ws.copy(), gs.copy())
    certified_norm.last_sweep_z = np.asarray(z, float).copy()     # whose sweep it is (2026-09-26)
    if getattr(certified_norm, 'band_check', False):
        # BAND CHECK (2026-09-20): D.hinf climbs from eigenvalue seeds without a
        # band and a resumed start reported 11.54 where the gate had certified
        # 10.39 for the same design.  Compare; on disagreement locate the peak
        # by a dense sweep over 3x the band and report whether it lies outside.
        try:
            gh, wh = D.hinf(arow, nvec, K, b, c)[:2]
            if gh > float(g) * (1.0 + 1e-6):
                W3 = np.linspace(3 * wlo, 3 * whi, 3601)
                g3 = np.array([sig(x) for x in W3]); j3 = int(np.argmax(g3))
                where = 'OUTSIDE' if not (wlo <= W3[j3] <= whi) else 'INSIDE'
                print(f"    band check: D.hinf {gh:.4f} at w={wh:+.3f} > certificate {float(g):.4f}; "
                      f"dense 3x-band sweep max {g3[j3]:.4f} at w={W3[j3]:+.3f} ({where} the band)", flush=True)
                certified_norm.band_check_hit = (float(W3[j3]), float(g3[j3]), where)
        except Exception as _e:
            print(f"    band check failed: {type(_e).__name__}", flush=True)
    if not np.isfinite(w):
        # WITNESS WHEN THE LOWER BOUND IS ALREADY THE NORM (2026-09-25).  The
        # certificate proved nothing exceeds gam_lb*(1+rtol) at its first
        # question and returned gam_lb with no witness.  The fallback used to be
        # the argmax of the COARSE full sweep (61 points over [-30, 30] with
        # --cert-sweep 61: a grid point such as w = 2.0, where sigma was 8.08
        # against a certified 10.91).  That fake witness was reported as a 10%
        # reduced-model error every round, anchored the model at a non-peak and
        # was enrolled in Omega.  With a reduced model attached: its fine-sweep
        # argmax, polished on the model (free), as the witness.
        # 2026-09-25 (invariant check): the fine-sweep argmax can be a DIFFERENT
        # near-equal peak than the one attaining gam_lb (flat multi-peak tops):
        # sigma_1 = 7.929682 at the chosen witness against 7.929851 certified.
        # Candidates: where the caller's lower bound was attained (w_hint, the
        # SQP's maximal row), the model's polished fine-sweep argmax, and the
        # model's top peaks; the witness is the one with the largest FULL value
        # (a handful of full solves per round).
        cands_w = []
        if w_hint is not None and np.isfinite(w_hint):
            cands_w.append(float(w_hint))
        if ro_ is not None and ro_.dim > 0:
            j = int(np.argmax(gf)); wj = float(wf[j]); dw = float(wf[1] - wf[0])
            try:
                from scipy.optimize import minimize_scalar as _msc; rr = _msc(lambda x: -_sr2(x), bounds=(wj - dw, wj + dw),
                                     method='bounded', options=dict(xatol=1e-9))
                cands_w.append(float(rr.x) if -float(rr.fun) >= gf[j] else wj)
            except Exception:
                cands_w.append(wj)
            cands_w += [float(x) for x in list(peaks)[:3]]
        cands_w.append(float(ws[gs.argmax()]))
        w = max(cands_w, key=lambda x: sig(x))
    return float(g), float(w), bool(ok), peaks


def _mimo_setup(a, D, flags, tag):
    """MIMO start (STAGE6): the design from --mimo-start (or this run's own
    checkpoint when it is better), the controller order from it, the flags the
    MIMO packer needs, and the atoms of every port.  Record format (as written
    by tools/mimo_start.py and by this driver): d, e, f, Bc (r x p), nvec = C_c
    (m x r), K = D_c (m x p), xa (m), xs (p), va / vs (m / p x n raw fields,
    or empty), atoms_a_p / atoms_s_p (m / p x 5 n_atoms, or empty), gamma."""
    import fieldbasis as _fb
    src = a.mimo_start; g0 = None
    try:
        g0 = float(json.load(open(src[:-4] + '.json'))['gamma'])
    except Exception:
        try:
            g0 = float(np.load(src, allow_pickle=True)['gamma'])
        except Exception:
            g0 = None
    ck = tag + '_ckpt.npz'
    if os.path.exists(ck):
        try:
            gck = float(json.load(open(tag + '_ckpt.json'))['gamma'])
            if g0 is None or gck < g0:
                print(f"CHECKPOINT found (gamma={gck:.6f}) -- resuming from {ck}", flush=True)
                src, g0 = ck, gck
        except Exception:
            pass
    z = np.load(src, allow_pickle=True)
    m, p = int(a.n_act), int(a.n_sens)
    R = len(np.atleast_1d(z['d']))
    D.r = R
    arow0 = (np.asarray(z['d'], complex), np.asarray(z['e'], complex), np.asarray(z['f'], complex),
             np.asarray(z['Bc'], complex).reshape(R, p))
    nvec0 = np.asarray(z['nvec'], complex).reshape(m, R)
    K0 = np.asarray(z['K'], complex).reshape(m, p)
    xa0 = np.atleast_1d(np.asarray(z['xa'], float)); xs0 = np.atleast_1d(np.asarray(z['xs'], float))
    def _fields(key, xs_, cnt):
        v = np.asarray(z[key]) if key in z.files else np.zeros(0)
        if v.size == cnt * D.n:
            return [np.asarray(row, complex) for row in v.reshape(cnt, D.n)]
        return [D.gauss(float(x)) for x in xs_]
    va0 = _fields('va', xa0, m) if flags.get('FB') else None
    vs0 = _fields('vs', xs0, p) if flags.get('FC') else None
    orders = a.loop_orders
    if orders is None and 'loop_orders' in z.files and np.asarray(z['loop_orders']).size == m:
        orders = [int(v) for v in np.asarray(z['loop_orders'])]
    if orders is None:
        orders = [R // m] * m
    flags.update(m=m, p=p, struct=a.ctrl_struct, loop_orders=orders)
    if a.field_atoms:
        for key, fk, v0_, cnt in (('a', 'FB', va0, m), ('s', 'FC', vs0, p)):
            if not flags.get(fk):
                continue
            P = np.asarray(z[f'atoms_{key}_p']) if f'atoms_{key}_p' in z.files else np.zeros(0)
            npar = _fb.NATOM_PAR * a.field_atoms
            if P.size == cnt * npar:
                p0 = [np.asarray(row, float) for row in P.reshape(cnt, npar)]; how = 'from the record'
            else:
                p0 = [_fb.fit_atoms(D.xn, v_, D.M, a.field_atoms, a.atom_smin, a.atom_kmax)[0] for v_ in v0_]
                how = 'fitted to the start fields'
            setattr(D, 'atoms_' + key, dict(n=int(a.field_atoms), smin=float(a.atom_smin), kmax=float(a.atom_kmax), p0=p0))
            print(f"field atoms ({'actuators' if key == 'a' else 'sensors'}): {cnt} x {a.field_atoms} Gabor atoms; {how}",
                  flush=True)
    return (arow0, nvec0, K0, xa0, xs0, va0, vs0), g0



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--wall', type=float, default=1500.)
    ap.add_argument('--r', type=int, default=7)
    ap.add_argument('--alpha_warm', type=int, default=0,
                    help='warm-started single alpha shift; 0 = OFF (default). '
                         'alpha now comes from the certificate, so the warm '
                         'path is redundant, and the ablation confirms off is '
                         'better: 9.8319 vs 9.8492.')
    ap.add_argument('--alpha_maxstep', type=float, default=0.02,
                    help='relative design change above which the '
                         'warm alpha shift is not trusted')
    ap.add_argument('--sob_len', type=float, default=0.8)
    ap.add_argument('--eps', type=float, default=EPS_STAB_DEFAULT)
    ap.add_argument('--outdir', default='results_epi')
    ap.add_argument('--dump', default='',
                    help='directory for the per-outer-round diagnostic dump '
                         '(Omega, sigma_1 on Omega, t, multipliers, '
                         'closed-loop eigenvalues; optimisation note item 10).'
                         '  Costs one sigma_1+gradient pass over Omega per '
                         'outer round, on the same pool.  Empty = off.')
    ap.add_argument('--inner-budget', dest='inner_budget', default='clock',
                    choices=('clock', 'iters'),
                    help="what STOPS the inner solve.  'clock' (default) is "
                         "the frozen behaviour: maxit=5000 and the wall clock "
                         "binds, so the iteration count -- and the answer -- "
                         "depend on machine speed and load.  'iters' fixes "
                         "maxit and demotes the clock to a guard, which is "
                         "what makes a run reproducible and a stage-1 "
                         "measurement trustworthy.")
    ap.add_argument('--level-norm', dest='level_norm', choices=['on', 'off'],
                    default='on',
                    help='item 11 candidate: on (frozen) = inner rows sigma/t0 - tau, '
                         'objective tau; off = rows sigma - t, objective t.')
    ap.add_argument('--inner-maxit-adapt', dest='inner_maxit_adapt', action='store_true',
                    help='STAGE 3: adapt the inner iteration budget to the surrogate\'s '
                         'validity measured by the certificate: x2 when certified/t <= '
                         '1 + tol, /2 when > 1 + 3 tol, within [min, max]')
    ap.add_argument('--inner-adapt-tol', dest='inner_adapt_tol', type=float, default=0.05)
    ap.add_argument('--inner-maxit-min', dest='inner_maxit_min', type=int, default=5)
    ap.add_argument('--inner-maxit-max', dest='inner_maxit_max', type=int, default=40)
    ap.add_argument('--row-workers', dest='row_workers', type=int, default=0,
                    help='reduced-model rows (gain_batch) evaluated in this many FORKED processes besides this one, '
                         'frequencies split by w-blocks (reduced._RowPool; STAGE6 6.26). 0 = serial')
    ap.add_argument('--oracle', choices=['full', 'reduced'], default='full',
                    help='STAGE 5: sigma rows of the inner solve on the full oracle '
                         '(frozen) or on the projected one (reduced.py).')
    ap.add_argument('--oracle-nvec', dest='oracle_nvec', type=int, default=3,
                    help='singular pairs per anchor')
    ap.add_argument('--oracle-tol', dest='oracle_tol', type=float, default=1e-3,
                    help='greedy: anchor where the reduced model exceeds the certified level by this')
    ap.add_argument('--oracle-sweep-tol', dest='oracle_sweep_tol', type=float, default=1e-4,
                    help='anchor where the reduced sweep disagrees with the certifier\'s sweep by more than this (relative)')
    ap.add_argument('--oracle-band-frac', dest='oracle_band_frac', type=float, default=0.2,
                    help='only where the full response is above this fraction of its maximum')
    ap.add_argument('--oracle-sweep-anchors', dest='oracle_sweep_anchors', type=int, default=3,
                    help='sweep-disagreement anchors per round at most')
    ap.add_argument('--oracle-dim-cap', dest='oracle_dim_cap', type=int, default=120,
                    help='restart the basis from the current design when it exceeds this')
    ap.add_argument('--sqp-row-res-tol', dest='sqp_row_res_tol', type=float, default=1e-3,
                    help='residual filter for the SQP abscissa rows (Ritz pairs of the reduced pencil).  1e-4 dropped an accurate binding mode (value within 2e-6, residual 1.0e-4: the normalisation by ||A|| inflates it) and the SQP lost the row; spurious Ritz values sit near 1e-2.  The full-certificate validation remains the safety net')
    ap.add_argument('--oracle-res-tol', dest='oracle_res_tol', type=float, default=1e-4,
                    help='phase 2: relative residual a reduced Ritz pair must pass to serve as the abscissa (else the full certificate)')
    ap.add_argument('--oracle-gate', dest='oracle_gate', type=float, default=1e-5,
                    help='enrich at the witness when the reduced value there differs from the certified one by more than this')
    ap.add_argument('--oracle-greedy', dest='oracle_greedy', type=int, default=4,
                    help='greedy anchors per round at most')
    ap.add_argument('--stabilise', action='store_true',
                    help='STAGE 2: abscissa-descent phase (stabilise.py) when the start '
                         'is outside the margin plus the cushion; HIFOO convention.')
    ap.add_argument('--stab-delta', dest='stab_delta', type=float, default=3e-3,
                    help='cushion inside the margin the phase must reach')
    ap.add_argument('--stab-maxit', dest='stab_maxit', type=int, default=150,
                    help='budget; a start not inside the margin by then is NOT STABILISED')
    ap.add_argument('--cont-accept-winding', dest='cont_accept_winding', action='store_true',
                    help='STAGE 2: accept a continuation-refused design if it passes the unconditional winding certificate; chain restarts there')
    ap.add_argument('--converge', action='store_true',
                    help='STAGE 2: outer loop runs until converged (inner stationary + certificate agrees + no new frequency) or --outer cap; the stall rule is off')
    ap.add_argument('--prov-check', dest='prov_check', type=int, default=16,
                    help='provisional rounds: coarse full spot check on this many grid points (+ the reduced peaks)')
    ap.add_argument('--prov-tol', dest='prov_tol', type=float, default=1e-3)
    ap.add_argument('--oracle-inputs', dest='oracle_inputs', choices=['on', 'off'], default='off',
                    help='position-robust reduced model: input bases over the position ranges + union Galerkin (STAGE5 5)')
    ap.add_argument('--xa-lo', dest='xa_lo', type=float, default=-3.0)
    ap.add_argument('--xa-hi', dest='xa_hi', type=float, default=0.0)
    ap.add_argument('--xs-lo', dest='xs_lo', type=float, default=0.0)
    ap.add_argument('--xs-hi', dest='xs_hi', type=float, default=3.0)
    ap.add_argument('--prov-gate', dest='prov_gate', choices=['full', 'reduced'], default='full',
                    help='provisional rounds (--cert-every > 1): stability gate on the full model (full) or from the reduced pencil with full-operator residual checks, the chain re-certified on the full model at the next full round (reduced)')
    ap.add_argument('--cert-fail', dest='cert_fail', choices=['rollback', 'continue'], default='rollback',
                    help='when a full check disagrees with the provisional chain: roll back to the last fully certified design, or continue from the checked design with its true value (enrolling what was missed)')
    ap.add_argument('--cert-every', dest='cert_every', type=int, default=1,
                    help='full certification every k-th round (and the last); other rounds use the reduced model\'s level-set certificate + one anchoring factorisation, accepted provisionally, rolled back if the next full certificate disagrees')
    ap.add_argument('--band-check', dest='band_check', action='store_true',
                    help='at every certification compare with D.hinf; on disagreement locate the peak by a dense 3x-band sweep and say whether it is outside the band')
    ap.add_argument('--band-first', dest='band_first', action='store_true',
                    help='widen the band from the CANDIDATE before its certificate (frozen: after acceptance, one candidate late)')
    ap.add_argument('--sqp-pair-manifold', dest='sqp_pair_manifold', choices=['on', 'off'], default='on',
                    help='compensator pair within 1e-2 of a double pole ON the line: hold it on the double-pole '
                         'manifold (three equality rows + retraction), leave on the multipliers (STAGE5 26)')
    ap.add_argument('--follow-scan', dest='follow_scan', type=int, default=21,
                    help='value-only scan points in each followed member\'s window, per SQP evaluation, before '
                         'Newton (2026-09-24; 0 = Newton from the member and the previous top only).  With '
                         'in-solve refinement on, a twin rising in the window is caught at the next accepted '
                         'iterate; the scan was 2/3 of a K+FB+FC evaluation (1,008 of 1,564 reduced solves)')
    ap.add_argument('--stop-progress', dest='stop_progress', default=None,
                    type=lambda v: tuple([int(v.split(':')[0])] + [float(x) for x in v.split(':')[1:]]),
                    help="'k:tol[:drop[:rounds]]' -- stop when the certified value gained less than tol (relative) over "
                         "the last k rounds, e.g. 3:0.003 (STAGE5 28); ARMED only once the run has gained drop "
                         "(relative, default 0.01) over its start value or after `rounds` rounds (default 20) -- "
                         "the frozen ne=200 start crawls for a few rounds on the pair's face before descending "
                         "(STAGE5 30).  Off by default.")
    ap.add_argument('--sqp-carry-h', dest='sqp_carry_h', choices=['on', 'off'], default='off',
                    help='carry the SQP quasi-Newton matrix across outer rounds (with --sqp-carry; the trust radius is '
                         'carried either way).  Default OFF (2026-09-25): replaying a plateau round of the ne=200 K+B+C '
                         'run with H = I went 6.8649 -> 6.8063 in 25 iterations; the run, carrying H, needed ~280 '
                         'iterations to reach 6.8257 (STAGE5 28)')
    ap.add_argument('--reduced-cert', dest='reduced_cert', choices=['dense', 'sparse'], default='dense',
                    help='certificate on REDUCED pencils (provisional rounds, pyGRANSO band row; the SQP refinement follows via sqp_inner.REDUCED_CERT): dense level-set (default) or the sparse decision procedure')
    ap.add_argument('--sqp-clblock', dest='sqp_clblock', choices=['on', 'off'], default='on',
                    help='SQP: closed-loop Hermite block on the two rightmost near-line eigenvalues (reduced model), smooth at coalescence; ON by default (user, 2026-09-24).  SQP path only; the frozen pyGRANSO path is unaffected')
    ap.add_argument('--row-screen', dest='row_screen', type=float, default=0.0,
                    help='SQP rows: values for all, vectors + gradients only within this fraction of the maximum (0 = all rows full)')
    ap.add_argument('--peak-follow', dest='peak_follow', choices=['on', 'off'], default='off',
                    help='Omega rows enrolled as peaks follow their peak (re-located on the reduced model at each evaluation; STAGE5 9)')
    ap.add_argument('--follow-radius', dest='follow_radius', type=float, default=0.1)
    ap.add_argument('--enroll-rel', dest='enroll_rel', type=float, default=0.0,
                    help='certifier enrols EVERY local maximum within this fraction of the max (refined on the reduced model), not just the top 4 (0 = frozen)')
    ap.add_argument('--pyg-bandmax', dest='pyg_bandmax', choices=['on', 'off'], default='off',
                    help="pyGRANSO's form of the SQP's in-solve refinement: one extra row = the reduced model's certified band maximum (needs --oracle reduced)")
    ap.add_argument('--field-basis', dest='field_basis', choices=['none', 'laplace', 'hermite'], default='none',
                    help='free fields (FB/FC) on a truncated basis instead of nodal values + Sobolev (STAGE5 6)')
    ap.add_argument('--field-n', dest='field_n', type=int, default=40)
    ap.add_argument('--field-kmax', dest='field_kmax', type=float, default=None,
                    help='FB/FC shape constraint: RMS wavenumber sqrt(v^H K v / v^H M v) <= this for each free field '
                         '(scale-invariant Sobolev constraint in the SQP; makes the field problem well-posed, STAGE5 32)')
    ap.add_argument('--field-order', dest='field_order', type=int, choices=[1, 2], default=1,
                    help='order of the --field-kmax form: 1 = H^1 (v^H K v), 2 = H^2 (v^H K M^-1 K v, bound kmax^4)')
    ap.add_argument('--field-atoms', dest='field_atoms', type=int, default=0,
                    help='FB/FC fields as sums of this many Gabor atoms (fieldbasis; STAGE5 32): width >= --atom-smin, '
                         '|wavenumber| <= --atom-kmax; the reduced model uses the --field-basis/--field-n basis as its input basis')
    ap.add_argument('--atom-smin', dest='atom_smin', type=float, default=1.0)
    ap.add_argument('--atom-kmax', dest='atom_kmax', type=float, default=0.7071)
    ap.add_argument('--field-lo', dest='field_lo', type=float, default=None)
    ap.add_argument('--field-hi', dest='field_hi', type=float, default=None)
    ap.add_argument('--alpha-argmax', dest='alpha_argmax', choices=['on', 'off'], default='off',
                    help='alpha_certified returns the eigenvalue/eigenvectors ATTAINING alpha (on) instead of those nearest the real point (alpha, 0) (off = frozen); affects the abscissa gradient on every path')
    ap.add_argument('--cert-rtol', dest='cert_rtol', type=float, default=1e-4,
                    help='relative tolerance of the level-set certificate (frozen 1e-4); a terminal polish gaining ~1e-3 per round needs ~1e-6')
    ap.add_argument('--wind-ctrl', dest='wind_ctrl', choices=['plain', 'exact'], default='plain',
                    help='winding counts (continuation acceptance, save-time certificate): plain = track f (frozen); '
                         'exact = track det(sI - A_c) f and add the compensator\'s phase in closed form -- conclusive '
                         'when compensator poles sit on the counting line (face S), STAGE5 30')
    ap.add_argument('--cert-disk', dest='cert_disk', choices=['frozen', 'strict'], default='frozen',
                    help='sparse certificate, ARPACK returned a PARTIAL eigenvalue set at a shift: frozen = size the '
                         'eigenvalue-free disk from it anyway (counted); strict = retry for the nearest eigenvalue '
                         'alone, bisect the interval if that fails too (STAGE5 30)')
    ap.add_argument('--cert-sweep', dest='cert_sweep', type=int, default=241,
                    help='points of the FULL sweep in the outer certification (frozen 241); the reduced oracle supplies the fine sweep for the peaks list.  '
                         '0 = the full model sampled only at the reduced model\'s peaks (at most ~17 points; 241 before the model exists) -- STAGE5 30')
    ap.add_argument('--ctrl-line', dest='ctrl_line', choices=['frozen', 'eps'], default='frozen',
                    help="compensator stability line.  frozen = the frozen combination (inner row Re <= ctrl_max - eps, "
                         "gate Re <= ctrl_max), kept for bit-identity.  eps = Re <= -eps EVERYWHERE (decision 2026-09-20): "
                         "gate, admissibility, both inner solvers, the LMI, the stabilisation phase.")
    ap.add_argument('--omega-policy', dest='omega_policy', choices=['frozen', 'an'],
                    default='frozen',
                    help='STAGE 3, note items 4-5.  frozen: Omega only grows.  an: '
                         'log-spaced satellites around each new peak (item 4), '
                         'retirement of the starter grid after the first certified '
                         'round and of members below (1-eps_w) gamma for two rounds, '
                         'cap at --omega-cap (item 5).')
    ap.add_argument('--omega-epsw', dest='omega_epsw', type=float, default=0.5,
                    help='level fraction: members below (1 - eps_w) gamma are dormant '
                         '(Apkarian-Noll eps_omega in [0.05, 0.5])')
    ap.add_argument('--omega-delta0', dest='omega_delta0', type=float, default=0.005,
                    help='satellite base offset')
    ap.add_argument('--omega-rho', dest='omega_rho', type=float, default=2.0,
                    help='satellite ratio; offsets delta0 * rho^k')
    ap.add_argument('--omega-nsat', dest='omega_nsat', type=int, default=6,
                    help='satellites per side per new peak')
    ap.add_argument('--omega-cap', dest='omega_cap', type=int, default=50)
    ap.add_argument('--omega-skeleton', dest='omega_skeleton', type=int, default=32,
                    help='uniform subsample of the starter grid that never retires (coverage of the band); 0 = none')
    ap.add_argument('--omega-refuter-rounds', dest='omega_refuter_rounds', type=int, default=4,
                    help='a frequency that refuted a trial design cannot retire for this many rounds')
    ap.add_argument('--omega-cap-after', dest='omega_cap_after', type=int, default=0,
                    help='retire and cap only from this outer round on; seeding runs from the start')
    ap.add_argument('--omega-floor', dest='omega_floor', type=int, default=30,
                    help='never retire below this many members')
    ap.add_argument('--inner', choices=['pygranso', 'sqp'], default='pygranso',
                    help='note item 8: inner solver.  pygranso = frozen.  sqp = '
                         'sqp_inner.py (Clarabel subproblem, 2x2 block cones at '
                         'enrolled frequencies when --sqp-block on).')
    ap.add_argument('--sqp-block', dest='sqp_block', choices=['on', 'off'], default='on')
    ap.add_argument('--sqp-ctrl', dest='sqp_ctrl', choices=['rows', 'block'], default='rows',
                    help="compensator constraint in the SQP: rows = per-pole rows (or the LMI with --sqp-lmi on); block = Burke-Overton coordinates, the two rightmost poles' 2x2 block with Hermite's criterion as a PSD cone (TRACKER H)")
    ap.add_argument('--sqp-refine', dest='sqp_refine', type=float, default=0.0,
                    help='exchange refinement inside the SQP on the reduced model: add local maxima of the reduced sigma_1 above t (1+tol) to the working set at each accepted step; 0 = off')
    ap.add_argument('--sqp-refine-max', dest='sqp_refine_max', type=int, default=3)
    ap.add_argument('--sqp-carry', dest='sqp_carry', action='store_true',
                    help='item 8 restart overhead: carry the SQP trust radius and BFGS matrix across outer rounds')
    ap.add_argument('--sqp-reach', dest='sqp_reach', type=float, default=1.0,
                    help='item 2: closed-loop eigenvalues with Re >= -eps - reach get their own row in the SQP (reduced pencil)')
    ap.add_argument('--sqp-lmi', dest='sqp_lmi', choices=['on', 'off'], default='on',
                    help='compensator stability as a Lyapunov LMI in the SQP subproblem (r x r, any plant size); off = one row per pole')
    ap.add_argument('--sqp-trust', dest='sqp_trust', type=float, default=0.0,
                    help='initial box trust radius (0 = 0.05 |z0|)')
    ap.add_argument('--qp-solver', dest='qp_solver', choices=['osqp', 'daqp'],
                    default='osqp',
                    help='item 0: steering/termination QP solver.  osqp = frozen '
                         '(ADMM, 1e-12 target it mostly does not reach); daqp = '
                         'dual active-set, the pip-installable stand-in for qpOASES.')
    ap.add_argument('--qp-eps', dest='qp_eps', type=float, default=None,
                    help='item 0: OSQP eps_abs = eps_rel for the steering QPs '
                         '(pyGRANSO hardcodes 1e-12, which OSQP mostly fails to '
                         'reach within its iteration limit).  None = frozen.')
    ap.add_argument('--qp-maxiter', dest='qp_maxiter', type=int, default=None,
                    help='item 0: OSQP max_iter (default 4000).  None = frozen.')
    ap.add_argument('--qp-log', dest='qp_log', action='store_true',
                    help='STAGE 1b, note item 0: record mu, violation, '
                         'stationarity and step per inner iterate, and the '
                         'histogram of OSQP statuses, into each round\'s record.')
    ap.add_argument('--sigma2', dest='sigma2', choices=['on', 'off'], default='off',
                    help='STAGE 1b, note item 3: enroll sigma_2(omega_i) <= t as its '
                         'own constraint -- always at the active frequencies, '
                         'thresholded with hysteresis elsewhere.  off = frozen.')
    ap.add_argument('--sigma2-active', type=float, default=1e-3,
                    help='sigma_1 within this of the max is ACTIVE: always enrolled')
    ap.add_argument('--sigma2-near', type=float, default=0.02,
                    help='sigma_1 within this of the max is a candidate for the threshold')
    ap.add_argument('--sigma2-active-gap', dest='sigma2_active_gap', type=float, default=None,
                    help='active frequencies are enrolled only if (sigma1-sigma2)/sigma1 <= this (None = always, frozen Stage-1b policy)')
    ap.add_argument('--sigma2-on', type=float, default=0.05,
                    help='candidate enrolled when sigma_2/sigma_1 >= 1 - this')
    ap.add_argument('--sigma2-off', type=float, default=0.15,
                    help='an enrolled row is kept while sigma_2/sigma_1 >= 1 - this')
    ap.add_argument('--sel-margin', dest='sel_margin', type=float, default=1e-3,
                    help='with --select-cons on: the returned iterate must sit '
                         'inside the stability margin by this fraction of eps, '
                         'so the certificate never finds an eigenvalue ON the '
                         'line')
    ap.add_argument('--opt-tol', dest='opt_tol', type=float, default=1e-10,
                    help='pyGRANSO opt_tol for the inner solve.  1e-10 is '
                         'frozen.  Stage-1 item 2: below the oracle noise '
                         'floor the solve ends on code 6 instead of code 0.')
    ap.add_argument('--cont-anchor', dest='cont_anchor',
                    choices=['strict', 'defer'], default='strict',
                    help='strict (current): an infeasible start is refused.  '
                         'defer: run without the continuation gate until the '
                         'first feasible design is accepted, anchor there.  '
                         'This is what the frozen campaign effectively did '
                         '(the gate did not exist); KGC needs it.')
    # ---- DEFAULTS ARE FROZEN (reverted 2026-09-17).  They were flipped on the
    # 16th without the 27-case regression the handoff requires; reverted until
    # that regression passes.  The stage-1 configuration is
    #   --select-cons on --ctrl-imcap off --cont-anchor defer --inner-maxit 25
    # and every stage-1 script passes it explicitly.  Previous text kept:
    # ---- STAGE-1 DEFAULTS (2026-09-17).  The frozen configuration is
    #   --select-cons off --ctrl-imcap on --cont-anchor strict --inner-maxit 200
    # (with --inner-budget clock, the frozen default, maxit is not binding).
    # Evidence: STAGE1.md sections 4-12; stage-1 gate jobs 16152993 and
    # 16205236 at ne=200.  --inner-maxit 25 because the 1-D certificate proves
    # only "nothing ON the line" and relies on small steps for the rest
    # (certify_sparse.py:433); maxit 100 from a poor start carried an
    # eigenvalue across unseen, five refusals, no progress; maxit 25 on the
    # same start: none, 9.674 -> 8.958 in three rounds.
    ap.add_argument('--ctrl-stab', dest='ctrl_stab', choices=['on', 'off'],
                    default='on',
                    help='on (frozen): the compensator must itself be stable '
                         '-- inner constraint, best-iterate selection (with '
                         '--select-cons on) and gate, all via ctrl_max=0.  '
                         'off: ctrl_max=1e9, i.e. no constraint on the '
                         'compensator abscissa anywhere.  Kept on by decision '
                         '(practitioners want it); its cost is measurable.')
    ap.add_argument('--ctrl-imcap', dest='ctrl_imcap', choices=['on', 'off'],
                    default='on',
                    help='on (frozen): compensator poles |Im| <= wmax as an '
                         'inner constraint and at the gate; the band is '
                         '3/2 x wmax.  off: no such constraint, and the band '
                         'follows the CLOSED-LOOP spectrum at each accepted '
                         'design (see band_from_closedloop).  The two halves '
                         'are one switch on purpose.')
    ap.add_argument('--band-recut', dest='band_recut', type=float, default=1.0,
                    help='with --ctrl-imcap off, closed-loop eigenvalues with '
                         'Re >= -band_recut set the band')
    ap.add_argument('--select-cons', dest='select_cons', choices=['on', 'off'],
                    default='off',
                    help='on: the inner solve returns its best iterate that is '
                         'feasible for EVERY inner constraint.  off (frozen): '
                         'closed-loop feasibility alone, which hands the gate '
                         'iterates the inner problem itself rejects.  Stage-1 '
                         'item 1; default flips once the A/B is read.')
    ap.add_argument('--ls-fallback', dest='ls_fallback', action='store_true',
                    help='set pyGRANSO halt_on_linesearch_bracket=False so a '
                         'Wolfe bracket failure (termination code 6) falls '
                         'through to its fallback ladder instead of ending the '
                         'inner solve.  DEFAULT OFF: the frozen suite was '
                         'produced with the pyGRANSO default (True).  Stage-1 '
                         'item 0; measure before adopting.')
    ap.add_argument('--inner-maxit', dest='inner_maxit', type=int, default=200,
                    help='inner BFGS-SQP iteration cap under '
                         '--inner-budget iters')
    ap.add_argument('--clock-guard', dest='clock_guard', type=float, default=8.0,
                    help='under --inner-budget iters the clock limit is set to '
                         'this multiple of the per-outer wall budget, as a '
                         'guard only; if it binds, the run says so')
    ap.add_argument('--lucount', action='store_true',
                    help='count sparse factorisations and triangular solves '
                         '(lu_counter.py; no change to the pipeline, scipy is '
                         'wrapped).  Implied by --dump.')
    ap.add_argument('--init', default=None)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--parent', default=None)
    ap.add_argument('--cold', action='store_true',
                    help='no warm start at all')
    ap.add_argument('--warm_from',
                    default='results_two,results_epi,results',
                    help='comma list of directories to search')
    ap.add_argument('--warm_dense', type=int, default=1,
                    help='1 = dense-verify each candidate '
                         '(GL scale); 0 = trust the stored alpha')
    ap.add_argument('--seed', default=None)
    ap.add_argument('--seed_file', default=None)
    # The band must cover the CLOSED-loop resonances, not the open-loop ones.
    # The compensator brings its own poles and a resonance sits near each: one
    # design peaked at omega = -9.69, outside the old [-5.5, 7.5], so the run
    # optimised a lesser peak and reported it as the norm.  These are now only
    # a FLOOR -- the band is widened at run time to cover the controller poles
    # (see band_from_controller below).
    ap.add_argument('--wlo', type=float, default=-8.0)
    ap.add_argument('--whi', type=float, default=8.0)
    ap.add_argument('--wpad', type=float, default=3.0,
                    help='margin beyond the outermost controller pole')
    ap.add_argument('--wmax_plant', type=float, default=20.0,
                    help='FIXED cap on |Im| of the compensator poles, and hence '
                         'the band (3/2 x this).  The plant itself has '
                         '|Im| <= 1 and peaks at omega = -0.65, so 20 is a '
                         'large margin, and it admits every seed once the '
                         'exogenous channels are widened (seeds.SIGMA_CHANNEL).  Common to every case, because '
                         'gamma is a supremum over the band and per-case bands '
                         'would not be comparable.')
    ap.add_argument('--ctrl_wmax', type=float, default=-1.0,
                    help='max |Im| of a controller pole.  <0 (default) sets it '
                         'to 2/3 of the certification band half-width, so the '
                         'compensator stays well inside the region where the '
                         'gain peak and the stability line are certified.')
    ap.add_argument('--outer', type=int, default=20,
                    help='max exchange steps.  FB+FC was still descending at 12 from cold, so 20 is a better floor; the sweep overrides it with OUTER (default 40 in sweep.sh).')
    ap.add_argument('--rtol', type=float, default=1e-4)
    ap.add_argument('--h0', choices=('auto', 'calibrated'), default='auto')
    ap.add_argument('--warm_h', type=int, default=0,
                    help='carry the BFGS Hessian across exchange steps; '
                         '0 = OFF (default).  MEASURED HARMFUL: off is faster '
                         '(199 s vs 243 s), uses fewer oracle calls (481 vs '
                         '555) and gives a better margin (-0.222 vs -0.092).')
    ap.add_argument('--ctrl_max', type=float, default=0.0,
                    help='how far right the CONTROLLER poles may go. '
                         '0 = the controller must itself be stable '
                         '(safe default); a positive value must be '
                         'seen by the closed-loop check.')
    ap.add_argument('--wind-accept', dest='wind_accept', choices=['auto', 'on', 'off'], default='auto',
                    help='winding count (global in w) on every candidate the continuation gate accepts; '
                         'auto = on for MIMO, off for SISO (STAGE6)')
    ap.add_argument('--n-act', dest='n_act', type=int, default=1,
                    help='MIMO (STAGE6): number of actuators (with --n-sens > 1 or > 1 itself: the MIMO path)')
    ap.add_argument('--n-sens', dest='n_sens', type=int, default=1, help='MIMO: number of sensors')
    ap.add_argument('--sqp-stat', dest='sqp_stat', choices=['auto', 'pred', 'rel'], default='auto',
                    help="SQP stationarity: 'rel' stat < 1e-4 x the active rows' gradient scale (to 2026-09-29); "
                         "'pred' also requires the QP's predicted decrease <= 1e-7 t (STAGE6 6.14); "
                         "'auto': pred for MIMO, rel for SISO (the ne=100 K+B+C A/B: 6.404012 in 22.2 min vs "
                         "6.399134 in 22.6 -- faster early, stopped 0.08 % higher; SISO adoption waits for the cluster)")
    ap.add_argument('--cert-all-w', dest='cert_all_w', choices=['on', 'off'], default='on',
                    help='every full certificate also proves the value over ALL w: tail bound + one level-set '
                         'question on [-W, W] (STAGE6 6.21)')
    ap.add_argument('--force-mimo', dest='force_mimo', action='store_true',
                    help='run m = p = 1 through the MIMO path (validation of the MIMO code on SISO, STAGE6 6.16)')
    ap.add_argument('--ctrl-struct', dest='ctrl_struct', choices=['central', 'decentral'], default='central',
                    help='MIMO: one m x p compensator (central) or m = p SISO loops, sensor k -> actuator k (decentral)')
    ap.add_argument('--loop-orders', dest='loop_orders', type=lambda t: [int(v) for v in t.split(',')], default=None,
                    help='MIMO decentral: the compensator order of each loop (default: equal split of the start record)')
    ap.add_argument('--mimo-start', dest='mimo_start', default=None,
                    help='MIMO: the start record (tools/mimo_start.py or a previous MIMO record)')
    ap.add_argument('--cont-track', dest='cont_track', choices=['full', 'reduced'], default='full',
                    help='continuation gate: eigenvalues tracked on the full model (shift-invert per watched '
                         'eigenvalue, line certificate at every homotopy midpoint) or on the reduced pencil with '
                         'full-model residual checks and full fallbacks (algebra note item 11, STAGE5 33)')
    ap.add_argument('--cont', type=int, default=1,
                    help='CONTINUATION GATE (default ON).  The line '
                         'certificate proves stability only along a path of '
                         'small steps; this gate enforces that premise by '
                         'tracking the near-line eigenvalues of the last '
                         'accepted iterate onto every candidate, bisecting '
                         'the design segment (with a line sweep at each '
                         'midpoint) whenever the motion is too large to be '
                         'conclusive.  Matrix-free; this is the mechanism '
                         'that scales to the cavity.  Turning it off '
                         'reopens the escape that produced eight unstable '
                         'saved designs.')
    ap.add_argument('--cont_k', type=int, default=8,
                    help='watch-list size: rightmost near-line eigenvalues '
                         'tracked per step')
    ap.add_argument('--cont_depth', type=int, default=4,
                    help='max homotopy bisections per candidate step '
                         '(up to 2^depth sub-segments); exhausted depth '
                         'REJECTS the candidate')
    ap.add_argument('--cont_motion', type=float, default=0.5,
                    help='allowed tracked-eigenvalue motion as a fraction of '
                         'its distance to the line; more than this triggers '
                         'a bisection')
    ap.add_argument('--cont_floor', type=float, default=0.05,
                    help='absolute motion allowance floor, so eigenvalues '
                         'far from the line do not trigger needless splits')
    ap.add_argument('--dense_check', type=int, default=1,
                    help='CHECK-ONLY dense eigensolves at the three anchor '
                         'points of the continuation argument: the START '
                         'design, the SAVE gate, and the no-regress guard '
                         '(re-verifying a stored file before it may block a '
                         'new result).  Never called on iterates and never '
                         'feeds the optimiser; automatically skipped above '
                         '--dense_nmax, where the continuation gate is the '
                         'protection.')
    ap.add_argument('--dense_nmax', type=int, default=3000,
                    help='largest n + r for which the check-only dense '
                         'eigensolve is used; beyond it a loud message is '
                         'printed and the matrix-free chain stands alone')
    ap.add_argument('--sym', type=int, default=0,
                    help='optimise on the SYMMETRIC SUBSPACE '
                         'psi_s = s*conj(R psi_a) (and xs = -xa).  This is '
                         'the definitive test of whether an asymmetric '
                         'optimum is spontaneous symmetry breaking or '
                         'optimiser slack: it finds the BEST symmetric '
                         'design, which the projection test in '
                         'check_symmetry.py cannot.  Halves the field dofs. '
                         'Use a SEPARATE --outdir: the no-regress guard '
                         'would otherwise discard a (legitimately) worse '
                         'symmetric result in favour of the unconstrained '
                         'one in the same directory.')
    ap.add_argument('--sym_branch', type=float, default=0.0,
                    help='+1 or -1 selects the fixed-point branch '
                         '(psi_s = +/- conj(R psi_a)); 0 (default) picks '
                         'the branch nearest the starting design.  Both are '
                         'legitimate optima; averaging ACROSS branches '
                         'annihilates the field.')
    ap.add_argument('--tau-split', dest='tau_split', choices=['ignore', 'cap'], default='ignore',
                    help="SQP arm: a homotopy split of the continuation gate (the gate PASSED) caps the step ('cap', "
                         "earlier) or not ('ignore', default, 2026-09-25: the cap made ne=100 K+B+C creep, STAGE5 29)")
    ap.add_argument('--tau', type=int, default=1,
                    help='bundle-style step governor (proximity control): a '
                         'self-calibrating cap on the accepted-step length, '
                         'OFF until the continuation gate produces evidence '
                         '(a split or a rejection), then set from the '
                         'offending step and relaxed after consecutive clean '
                         'acceptances.  Unlike the ablated inner trust '
                         'region it never throttles the inner solver and '
                         'never acts before evidence.')
    ap.add_argument('--wind', type=int, default=1,
                    help='winding-number stability certificate at SAVE time '
                         '(argument principle on the return difference '
                         'f = 1 - k(s)G(s); one sparse solve per sample). '
                         'UNCONDITIONAL: unlike the line sweep it does not '
                         'rest on continuation.  Cross-validated 80/80 '
                         'against dense eigenvalue counts.  Advisory by '
                         'default; --wind_veto 1 makes a nonzero count '
                         'block the save.')
    ap.add_argument('--wind_veto', type=int, default=0)
    ap.add_argument('--wind_pplant', type=int, default=-1,
                    help='#plant eigenvalues right of the counting line '
                         '(Re = -eps/2); -1 computes it, dense, when '
                         'n <= --dense_nmax.  At cavity scale supply it '
                         '(the near-axis plant modes are physically known '
                         'and design-independent).')
    ap.add_argument('--nproc', type=int, default=-1,
                    help='pool size for the |Omega| FREQUENCY constraints.  '
                         '-1 (default) takes the WHOLE SLURM allocation; '
                         '0 forces serial.  These are independent, equal-cost '
                         'gain evaluations and scale well: 13.2x at 32 '
                         'workers, and |Omega| grows past 120 as the exchange '
                         'proceeds, so a large pool keeps paying off.')
    ap.add_argument('--cert_nseg', type=int, default=1,
                    help='segments for the STABILITY constraint; 1 = SERIAL '
                         '(default).  It parallelises poorly -- the band is '
                         'mostly empty spectrum, so one segment does nearly '
                         'all the work while the rest finish at once: 1.43x at '
                         '8 segments, 0.75x at 4, 0.90x at 32.  The 24%% it can '
                         'add is not worth the fragility; set >1 to try it.')
    ap.add_argument('--cert_eps_int', type=float, default=1e-5,
                    help='certificate interval resolution; smaller '
                         'proves more but costs shifts')
    ap.add_argument('--cert_itmax', type=int, default=800)
    ap.add_argument('--cert_shrink', type=float, default=0.9,
                    help='disk shrink factor; closer to 1 proves '
                         'more but leans harder on the eigensolver')
    # note item 11: the level-scaling constant tau0 > 1 is gone; 1.0 everywhere.
    ap.add_argument('--tau0', type=float, default=1.0,
                    help='initial t as a multiple of max sigma_1.  1.0 starts '
                         'exactly ON the binding constraint, which is fine: '
                         'the >1 default this briefly carried was introduced '
                         'to work around a Jacobian bug (the d/dtau column was '
                         'set outside the constraint loop, so only the last '
                         'row carried it), not because the boundary start is '
                         'a problem.')
    ap.add_argument('--trust', type=float, default=0.0,
                    help='trust radius on the inner step; 0 = OFF (default). '
                         'MEASURED HARMFUL: the ablation gives 9.8492 with it '
                         'on and 9.8319 (the sensor bound) with it off, at the '
                         'same escape count -- so it never did the job it was '
                         'added for.')
    ap.add_argument('--trust_warm', type=float, default=0.0,
                    help='trust radius when --resume/--parent is given; '
                         '<0 uses --trust.  DEFAULT 0 (off) since the A/B '
                         'measurement: the same KGC+C polish took 33,022 '
                         'oracle calls and 12,639 s at trust 0.1 versus '
                         '1,666 calls and 1,069 s at trust 0 -- and reached '
                         'a BETTER gamma (11.7180 vs 11.7256).  The safety '
                         'role the warm trust radius was quietly playing is '
                         'now carried explicitly by the continuation gate '
                         'and the tau governor.')
    ap.add_argument('--rect_cert', type=int, default=1,
                    help='1 = the stability certificate gates every accepted '
                         'iterate (0 falls back to alpha alone -- not advised)')
    # KEEP 48: the ablation shows 24/0.25 is worse on every axis -- gamma
    # 9.8381 vs 9.8492, THREE surrogate escapes instead of one, and 772 oracle
    # calls instead of 555.  This is the one tuned parameter that earned its
    # place.
    ap.add_argument('--nseed', type=int, default=160,
                    help='max frequencies in the initial active set')
    ap.add_argument('--seed_dw', type=float, default=0.25,
                    help='Omega seeding RESOLUTION in omega.  A fixed COUNT '
                         'over a wide band gives a grid coarser than the peak; '
                         'a fixed resolution does not.  With dw = 0.25 over '
                         '+-37.5 the grid is 301 points; sigma_1 is sharply '
                         'peaked (10.4 at w = -0.5), so seed_frac decides how '
                         'many survive: 0.10 keeps 36, 0.02 keeps 106.')
    ap.add_argument('--seed_frac', type=float, default=0.02,
                    help='seed Omega where sigma_1 >= frac * peak')
    a = ap.parse_args()
    certified_norm.rtol = float(a.cert_rtol)      # BEFORE the start certification: one tolerance for every certified value of a run
    global REDUCED_CERT
    REDUCED_CERT = a.reduced_cert
    import sqp_inner as _sqi; _sqi.REDUCED_CERT = a.reduced_cert; _sqi.FOLLOW_SCAN = a.follow_scan; _sqi.PAIR_MANIFOLD = (a.sqp_pair_manifold == 'on'); _sqi.FIELD_KMAX2 = (a.field_kmax ** 2 if a.field_kmax else None); _sqi.FIELD_ORDER = a.field_order
    certified_norm.enroll_rel = float(a.enroll_rel)
    if a.ctrl_line == 'eps':
        # Re <= -eps everywhere: the gate's allowance and every consumer of ctrl_max
        a.ctrl_max = min(float(a.ctrl_max), -float(a.eps))
    os.makedirs(a.outdir, exist_ok=True)
    MIMO = (a.n_act > 1 or a.n_sens > 1 or a.force_mimo)
    if MIMO:
        if a.resume:
            raise SystemExit("MIMO: start from --mimo-start (a MIMO record), not --resume")
        if not a.mimo_start:
            raise SystemExit("MIMO: --mimo-start REC is required (tools/mimo_start.py builds one from a SISO record)")
        if not a.band_first:
            # STAGE6 6.7: high-gain branches move far between candidates (|Im| 30 ->
            # 700 in one step); certified on the band of the PREVIOUS design, a
            # candidate with a peak of 12.25 at w = -700.5 was accepted as 2.5877
            a.band_first = True
            print("MIMO: --band-first on (the candidate's band before its certificate)", flush=True)
    _wind_accept = (a.wind_accept == 'on') or (a.wind_accept == 'auto' and MIMO)
    n_wind_accept, n_wind_refused, n_gate_wind = [0], [0], [0]

    def _wind_right_of_margin(bb_, cc_, aa_, nn_, KK_, seen_):
        """Exact count of closed-loop eigenvalues right of Re = -eps (argument
        principle, global in w).  None when the plant pole data is unavailable
        (n above the dense limit).  The line certificate just proved nothing lies
        ON Re = -eps, which is what the argument principle needs there, so count
        0 <=> alpha <= -eps, unconditionally."""
        import winding_cert as _wc
        _line = -a.eps
        if getattr(D, '_wind_plant_eps', None) is None:
            D._wind_plant_eps = _wc.plant_pole_data(D.A, D.M, _line)      # sparse enclosure (STAGE6 6.13)
        _pp, _hot = (D._wind_plant_eps if getattr(D, '_wind_plant_eps', None) else (None, None))
        if _pp is None:
            return None
        # the near-line eigenvalues as zeros_hint and a fine step floor: a design
        # pinned at the margin (alpha -0.050123, 1.2e-4 from the line) was
        # INCONCLUSIVE without them, conclusive with (957 pts)
        _zh = [complex(x) for x in (seen_ or []) if abs(complex(x).real - _line) < 0.3]
        return _wc.winding_count(D.A, D.M, bb_, cc_, aa_, nn_, KK_, line=_line, mc=D.mc,
                                 p_plant=_pp, hot=_hot, zeros_hint=_zh, hmin_frac=1e-8,
                                 ctrl_exact=(a.wind_ctrl == 'exact'))
    tag = os.path.join(a.outdir, f"{a.case.replace('+','_')}_s{a.start}_ne{a.ne}"
                       + (f"_mimo{a.n_act}x{a.n_sens}_{a.ctrl_struct}" if MIMO else ''))
    t_start = time.time()

    # The controller order is set by the SEED CATALOGUE (the common order from
    # the Hankel shoulder), not by --r.  Enlarging the catalogue moved it from
    # 7 to 8, and a stale --r 7 built the worker Packers with the wrong layout:
    # "operands could not be broadcast together with shapes (27,) (30,)", i.e.
    # 4r-1 for r = 7 against r = 8.
    try:
        _sd = np.load(a.seed_file or f"seeds_ne{a.ne}.npz", allow_pickle=True)
        _r = int(_sd['r'])
        if _r != a.r:
            print(f"controller order {a.r} -> {_r} (from the seed catalogue)",
                  flush=True)
            a.r = _r
    except Exception:
        pass
    D = FECoDesign(ne=a.ne, r=a.r, sob_len=a.sob_len)
    import massops; massops.warmup(D.M)          # the mass matrix, factorised ONCE (STAGE6 6.29)
    D.alpha_argmax = (a.alpha_argmax == 'on')     # eigenvector of the eigenvalue ATTAINING alpha (2026-09-22)
    D.alpha_warm = bool(a.alpha_warm)
    D.alpha_warm_maxstep = a.alpha_maxstep
    # PROCESS POOL, shared by the constraint loop and the certificate.  Both
    # blocks are independent per task; the eigenvector recovery is not, but
    # inverse iteration made it 1.5% of the call so it no longer matters.
    # The two constraint blocks parallelise very differently and are sized
    # separately.  The |Omega| frequency constraints are independent, uniform
    # gain evaluations: 13.2x at 32 workers.  The stability constraint is one
    # adaptive bisection over a band that is mostly EMPTY spectrum, so the
    # segment holding the eigenvalue does nearly all the work and the rest
    # finish immediately -- 1.43x at 8 segments, and WORSE than serial beyond
    # that once the redundant shifts and pool overhead are counted.
    D.cert_eps_int = a.cert_eps_int
    D.cert_itmax = a.cert_itmax
    D.cert_shrink = a.cert_shrink
    flags = parse_case(a.case)
    ini = np.load(a.init or f"init_ne{a.ne}.npz")
    arow0 = (ini['d'], ini['e'], ini['f']); nvec0 = ini['nvec']; K0 = 0j
    _ctrl_borrowed, _ctrl_default = False, None
    _resume_gamma = None
    xa0, xs0 = float(ini['xa']), float(ini['xs'])
    va0 = D.gauss(xa0); vs0 = D.gauss(xs0)
    if a.seed:
        sd = np.load(a.seed_file or f"seeds_ne{a.ne}.npz", allow_pickle=True)
        arow0 = (sd[a.seed + '__d'], sd[a.seed + '__e'], sd[a.seed + '__f'])
        nvec0 = sd[a.seed + '__nvec']
        # STAGE 2: a random-start catalogue (tools/randstart.py) also carries
        # the gain K and the start positions; a seed catalogue does not
        if a.seed + '__K' in sd.files:
            K0 = complex(sd[a.seed + '__K'])
        if a.seed + '__xa' in sd.files and flags.get('B'):
            xa0 = float(sd[a.seed + '__xa']); va0 = D.gauss(xa0)
        if a.seed + '__xs' in sd.files and flags.get('C'):
            xs0 = float(sd[a.seed + '__xs']); vs0 = D.gauss(xs0)
        print(f"controller seed = {a.seed}", flush=True)
    _al = D.norm_M(D.gauss(xa0).astype(complex))
    _be = D.norm_M(D.gauss(xs0).astype(complex))
    nvec0 = nvec0 * (_al * _be); K0 = K0 * (_al * _be)

    src = a.resume
    # CHECKPOINT AUTO-RESUME.  An interrupted run (SLURM time limit, node
    # failure) used to lose ALL progress, because the result is written only
    # at the end: one FB+FC job spent four hours improving 4.4016 -> 4.3419
    # and saved nothing.  Every ACCEPTED iterate now writes tag_ckpt.npz
    # (below), and at startup the checkpoint is preferred over the chosen
    # start whenever its recorded gamma is better.  Safety is unchanged: the
    # checkpoint goes through the same resume path and the same anchor
    # verification (line sweep + dense at GL) as any other start.
    _ck = tag + '_ckpt.npz'
    if os.path.exists(_ck) and not MIMO:
        try:
            _ckg = float(json.load(open(tag + '_ckpt.json'))['gamma'])
        except Exception:
            _ckg = None
        _srcg = None
        if a.resume:
            try:
                _srcg = float(json.load(
                    open(a.resume[:-4] + '.json'))['gamma'])
            except Exception:
                _srcg = None
        if _ckg is not None and (_srcg is None or _ckg < _srcg):
            print(f"CHECKPOINT found (gamma={_ckg:.6f}"
                  + (f" vs resume {_srcg:.6f}" if _srcg is not None else "")
                  + f") -- resuming from {_ck}", flush=True)
            src = _ck
            a.resume = _ck
    # When --resume is given explicitly the file need NOT be this case's own
    # previous result -- it is routinely a PARENT's.  Assuming otherwise made
    # pf['K'] true for KG+FBR+FCR resuming from FBR+FCR, so the controller was
    # read from a file that has none (Dc = 0), the seed branch never fired, and
    # the run began from the OPEN LOOP at gamma = 882.88 instead of the
    # parent's 5.50.  Read the case from the file's own summary.
    src_case = None
    if a.resume:
        _j = a.resume[:-4] + '.json' if a.resume.endswith('.npz') else None
        try:
            _resume_gamma = float(np.load(a.resume, allow_pickle=True)['gamma'])
        except Exception:
            _resume_gamma = None
        if _j and os.path.exists(_j):
            try:
                src_case = json.load(open(_j)).get('case')
            except Exception:
                src_case = None
        if src_case is None:
            src_case = a.case          # fall back to the old assumption
        elif src_case != a.case:
            print(f"resume file is case {src_case}; inheriting only the groups "
                  f"it optimised", flush=True)
    # (MIMO starts from --mimo-start: the SISO warm-start search is not needed,
    # and its stability check is a DENSE eig of the plant -- 160 GB at n = 1e5;
    # 2026-09-30 review)
    if src is None and not a.cold and not MIMO:
        # warm-start selection lives in warmstart.py: it filters out UNSTABLE
        # candidates (the old directories hold closed loops with alpha up to
        # +4.3e6, which look attractive to a lowest-gamma search precisely
        # because they are unstable) and prefers a start that actually carries
        # a controller when this case needs one -- resuming KG+FBR+FCR from the
        # controller-free FBR+FCR set Dc = 0 and restarted it from the OPEN
        # LOOP at gamma = 882.88.
        import warmstart
        dirs = [d for d in a.warm_from.split(',') if d.strip()]
        src, src_case = warmstart.select(D, a.case, a.ne, dirs, parse_case,
                                         eps=a.eps, dense=bool(a.warm_dense))
    if src and os.path.exists(src):
        ck = np.load(src, allow_pickle=True)
        # Inherit ONLY the groups the SOURCE optimised, intersected with the
        # ones this case frees.  Using this case's flags made a controller case
        # read a controller-free parent's ZERO controller -- the open-loop
        # restart that sent KG+FBR+FCR back to gamma = 882.88.
        if a.parent and not a.resume:
            pf0 = parse_case(a.parent)
        elif src_case:
            pf0 = parse_case(src_case)
        else:
            pf0 = flags
        pf = {k: bool(pf0.get(k)) and bool(flags.get(k))
              for k in ('K', 'B', 'C', 'FB', 'FC')}
        if pf['K']:
            arow0 = (ck['d'], ck['e'], ck['f'])
            nvec0, K0 = ck['nvec'], complex(ck['K'])
        elif flags['K']:
            # This case frees a controller but the SOURCE had none, so pf['K']
            # is False and the controller keeps its seed default -- which for a
            # static-gain case means Dc = 0, i.e. an OPEN LOOP.  Resuming
            # KG+FBR+FCR from FBR+FCR therefore started at gamma = 882.88 (the
            # uncontrolled value) instead of the parent's 5.50, and never
            # recovered.
            #
            # Fix: keep the parent's FIELDS and borrow a controller from the
            # best available result that HAS one and is compatible with this
            # case's controller structure.  Falls back to the seed if nothing
            # suitable exists.
            try:
                import warmstart as _ws
                _dirs = [d for d in a.warm_from.split(',') if d.strip()]
                _src, _sc = _ws.select_controller(
                    D, a.case, a.ne, _dirs, parse_case,
                    eps=a.eps, dense=bool(a.warm_dense))
                if _src is not None:
                    _ck = np.load(_src, allow_pickle=True)
                    # keep the default so the two starts can be COMPARED: the
                    # borrowed controller is the lowest-gamma one on file, not
                    # one tuned for THIS parent's fields (stage-1 gate: K+FB+FC's
                    # controller on B+C's fields started KSP+B+C at 935.9
                    # against the parent's 9.67, and the run never recovered)
                    _ctrl_default = (arow0, nvec0, K0)
                    _ctrl_borrowed = True
                    arow0 = (_ck['d'], _ck['e'], _ck['f'])
                    nvec0, K0 = _ck['nvec'], complex(_ck['K'])
                    print(f"  controller seeded from {os.path.basename(_src)} "
                          f"(case {_sc})", flush=True)
            except Exception as _e:
                print(f"  controller seed unavailable ({type(_e).__name__})",
                      flush=True)
        if pf['B']: xa0 = float(ck['xa'])
        if pf['C']: xs0 = float(ck['xs'])
        if pf['FB']:
            va0 = ck['xi_a'] if ('xi_a' in ck.files and ck['xi_a'].size) \
                else D.smooth_unmap(ck['va'])
        if pf['FC']:
            vs0 = ck['xi_s'] if ('xi_s' in ck.files and ck['xi_s'].size) \
                else D.smooth_unmap(ck['vs'])
        print(f"warm start from {src}", flush=True)
    if a.start > 0:
        rng = np.random.default_rng(1000 + a.start)
        def jit(v):
            v = np.asarray(v, complex)
            return v if v.size == 0 else v * (
                1 + 0.12 * (rng.standard_normal(v.size)
                            + 1j * rng.standard_normal(v.size)))
        if flags['K']:
            arow0 = tuple(jit(x) for x in arow0); nvec0 = jit(nvec0)
        if flags['B']: xa0 += 0.4 * rng.standard_normal()
        if flags['C']: xs0 += 0.4 * rng.standard_normal()

    # The controller order comes from the SEED CATALOGUE, but arow0 may come
    # from init_ne*.npz -- and if that file predates the current catalogue the
    # two disagree, giving "value array of shape (7,) could not be broadcast to
    # indexing result of shape (8,)".  Detect it and take the controller from
    # the catalogue instead of failing deep inside the resolvent.
    if len(np.atleast_1d(arow0[0])) != D.r:
        _got = len(np.atleast_1d(arow0[0]))
        try:
            _sd = np.load(a.seed_file or f"seeds_ne{a.ne}.npz", allow_pickle=True)
            _nm = str(_sd['names'][0])
            arow0 = (_sd[f'{_nm}__d'], _sd[f'{_nm}__e'], _sd[f'{_nm}__f'])
            nvec0 = _sd[f'{_nm}__nvec']
            K0 = complex(_sd[f'{_nm}__K']) if f'{_nm}__K' in _sd else 0j
            _al = D.norm_M(D.gauss(xa0).astype(complex))
            _be = D.norm_M(D.gauss(xs0).astype(complex))
            nvec0 = nvec0 * (_al * _be); K0 = K0 * (_al * _be)
            print(f"init file has order {_got} but the catalogue has {D.r}; "
                  f"taking the controller from seed '{_nm}'.  Re-run prepare.py "
                  f"so init_ne{a.ne}.npz matches.", flush=True)
        except Exception as _e:
            raise SystemExit(
                f"controller order mismatch: init has {_got}, catalogue has "
                f"{D.r}, and the catalogue could not be read ({_e}).  "
                f"Re-run prepare.py --ne {a.ne}.")
    if a.sym:
        _cf = bool(flags.get('FB')) and bool(flags.get('FC'))
        _cp = bool(flags.get('B')) and bool(flags.get('C'))
        if not (_cf or _cp):
            print("--sym ignored: this case frees neither both fields nor "
                  "both positions, so the mirror map takes it to a DIFFERENT "
                  "case", flush=True)
            a.sym = 0
    if a.sym:
        # Project the START onto the subspace (the parameterization keeps
        # every iterate on it; the start must join it).  Averaging is done on
        # UNIT profiles so the two sides carry equal weight, on the branch
        # nearest the start unless forced -- averaging across branches
        # annihilates the field.
        _s = float(a.sym_branch)
        if _cf:
            _var = D.smooth_map(va0) if va0 is not None else None
            _vsr = D.smooth_map(vs0) if vs0 is not None else None
            _pa0, _ps0 = D.bc(xa0, xs0, _var, _vsr)[2:4]
            _mir = np.conj(_ps0[::-1])
            if _s == 0.0:
                _s = 1.0 if (np.linalg.norm(_pa0 - _mir)
                             <= np.linalg.norm(_pa0 + _mir)) else -1.0
            _psym = 0.5 * (_pa0 + _s * _mir)
            if np.linalg.norm(_psym) < 1e-10:
                print("FATAL: symmetric projection annihilated the start "
                      "field (start sits on the OPPOSITE branch); rerun "
                      "with --sym_branch=-1 or a different start.",
                      flush=True)
                return
            va0 = D.smooth_unmap(_psym)
            vs0 = _s * np.conj(va0[::-1])
        if _cp:
            if _s == 0.0:
                _s = 1.0
            _xm = 0.5 * (xa0 - xs0)
            xa0, xs0 = float(_xm), float(-_xm)
        flags['sym'] = True
        flags['sym_branch'] = float(_s)
        print(f"SYMMETRIC-SUBSPACE mode: psi_s = "
              f"{'+' if _s > 0 else '-'}conj(R psi_a)"
              f"{', xs = -xa' if _cp else ''}; start projected onto the "
              f"subspace (branch chosen as nearest)", flush=True)
    pk_args = (arow0, nvec0, K0, xa0, xs0, va0, vs0)
    if a.field_basis != 'none' and (flags.get('FB') or flags.get('FC')):
        # FREE FIELDS ON A TRUNCATED BASIS (STAGE5 6): va = Phi_a beta_a,
        # vs = Phi_s beta_s; the region defaults to the start field's support
        # (|v| > 1 % of max) padded by 1
        import fieldbasis as _fb
        def _region(v):
            if a.field_lo is not None and a.field_hi is not None:
                return a.field_lo, a.field_hi
            av = np.abs(np.asarray(v)); sup = np.asarray(D.xn)[av > 1e-2 * av.max()]
            return float(sup.min()) - 1.0, float(sup.max()) + 1.0
        def _basis(v):
            lo_, hi_ = _region(v)
            if a.field_basis == 'laplace':
                return _fb.laplace_basis(D, lo_, hi_, a.field_n)[0], (lo_, hi_)
            return _fb.hermite_basis(D, lo_, hi_, a.field_n), (lo_, hi_)
        # the loader provides the NODAL path's coordinates xi (va = smooth_map(xi),
        # Sobolev length a.sob_len), not the field; basis mode has no smoothing
        # step, so project the FIELD (projecting xi started K+FB+FC from a
        # different design: certified 4.805 instead of 3.928)
        _va0 = D.smooth_map(pk_args[5]) if (flags.get('FB') and pk_args[5] is not None) else pk_args[5]
        _vs0 = D.smooth_map(pk_args[6]) if (flags.get('FC') and pk_args[6] is not None) else pk_args[6]
        pk_args = pk_args[:5] + (_va0, _vs0) + pk_args[7:]
        if flags.get('FB') and _va0 is not None:
            D.field_basis_a, _ra = _basis(_va0)
            print(f"field basis ({a.field_basis}): actuator {D.field_basis_a.shape[1]} modes on "
                  f"[{_ra[0]:.2f}, {_ra[1]:.2f}]", flush=True)
        if flags.get('FC') and _vs0 is not None:
            D.field_basis_s, _rs = _basis(_vs0)
            print(f"field basis ({a.field_basis}): sensor {D.field_basis_s.shape[1]} modes on "
                  f"[{_rs[0]:.2f}, {_rs[1]:.2f}]", flush=True)
    if a.field_atoms and (flags.get('FB') or flags.get('FC')):
        # GABOR ATOMS (STAGE5 32): the packing is the atom parameters; the field
        # basis built above becomes the reduced model's INPUT basis only
        import fieldbasis as _fb
        if getattr(D, 'field_basis_a', None) is not None:
            D.field_input_a = D.field_basis_a; D.field_basis_a = None
        if getattr(D, 'field_basis_s', None) is not None:
            D.field_input_s = D.field_basis_s; D.field_basis_s = None
        _srcz = np.load(a.resume, allow_pickle=True) if (a.resume and os.path.exists(a.resume)) else None
        for key, fk, v0_ in (('a', 'FB', pk_args[5]), ('s', 'FC', pk_args[6])):
            if not flags.get(fk) or v0_ is None:
                continue
            if _srcz is not None and f'atoms_{key}_p' in _srcz.files and \
                    len(_srcz[f'atoms_{key}_p']) == _fb.NATOM_PAR * a.field_atoms:
                p0 = np.asarray(_srcz[f'atoms_{key}_p'], float); how = 'from the record'
            else:
                v0n = D.smooth_map(v0_) if getattr(D, 'field_basis_' + key, None) is None and \
                    getattr(D, 'field_input_' + key, None) is None else v0_
                p0, err = _fb.fit_atoms(D.xn, v0n, D.M, a.field_atoms, a.atom_smin, a.atom_kmax)
                how = f'fitted to the start field (M-norm error {err:.3f})'
            setattr(D, 'atoms_' + key, dict(n=int(a.field_atoms), smin=float(a.atom_smin), kmax=float(a.atom_kmax), p0=p0))
            print(f"field atoms ({'actuator' if key == 'a' else 'sensor'}): {a.field_atoms} Gabor atoms, width >= {a.atom_smin}, "
                  f"|wavenumber| <= {a.atom_kmax}; {how}", flush=True)
    if MIMO:
        pk_args, _mimo_g0 = _mimo_setup(a, D, flags, tag)
        from packer import PackerMIMO
        pk = PackerMIMO(D, flags, *pk_args)
        print(f"MIMO: {flags['m']} actuators, {flags['p']} sensors, {flags['struct']} compensator of order {D.r}"
              + (f" (loops {flags['loop_orders']})" if flags['struct'] == 'decentral' else '')
              + f"; {pk.nth} free complex controller entries; start {a.mimo_start}"
              + (f" (record gamma {_mimo_g0:.6f})" if _mimo_g0 is not None else ''), flush=True)
    else:
        pk = Packer(D, flags, *pk_args)
    z = pk.z0()
    _stab_rec = None
    if a.stabilise:
        # ---- STAGE 2: abscissa-descent phase (stabilise.py) before anything
        # else, when the start is not inside the margin plus the cushion.
        from stabilise import stabilise as _stab_fn, dense_abscissa_pair as _dap
        _aa_, _nn_, _KK_, _xa_, _xs_, _va_, _vs_ = pk.unpack(z)
        _bb_, _cc_, *_ = D.bc(_xa_, _xs_, _va_, _vs_)
        _al_start = _dap(D, _aa_, _nn_, _KK_, _bb_, _cc_)[0]
        # the phase's objective is max(alpha_cl, alpha_ctrl - allowance): skip
        # only when BOTH are inside (two HIFOO draws had a stable closed loop
        # by chance with an unstable compensator and skipped the phase)
        import controller as _ctl
        _alc_start = float(np.max(np.linalg.eigvals(_ctl.A_of(*_aa_)).real)) - max(-a.eps, a.ctrl_max)
        _target = -a.eps - a.stab_delta
        # trigger on INFEASIBILITY (the constraint), stop at the target (the
        # cushion): triggering on the target re-stabilised every feasible
        # checkpoint on resume (a design certified 9.93 came back as 11.09)
        if max(_al_start + a.eps, _alc_start) > 0.0 and not os.path.exists(tag + '_ckpt.npz'):
            _zs, _al_s, _alc_s, _ev_s, _ok_s, _wall_s = _stab_fn(
                D, pk, flags, z, a.eps, delta=a.stab_delta, maxit=a.stab_maxit,
                ctrl_max=a.ctrl_max)
            _stab_rec = dict(alpha_start=float(_al_start), alpha_end=float(_al_s),
                             alpha_ctrl_end=float(_alc_s), evals=int(_ev_s),
                             stabilised=bool(_ok_s), wall=float(_wall_s))
            print(f"stabilise: alpha {_al_start:+.4f} -> {_al_s:+.4f} (ctrl {_alc_s:+.4f}) "
                  f"in {_ev_s} evaluations, {'OK' if _ok_s else 'NOT STABILISED'}", flush=True)
            if not _ok_s:
                print(f"FATAL: start not stabilised within {a.stab_maxit} iterations "
                      f"(alpha {_al_s:+.4f} > target {_target:+.4f}); counted as such.", flush=True)
                if a.dump:
                    os.makedirs(a.dump, exist_ok=True)
                    json.dump(dict(case=a.case, seed=a.seed, stab=_stab_rec),
                              open(os.path.join(a.dump, 'not_stabilised.json'), 'w'), indent=1)
                return
            z = _zs
        else:
            _stab_rec = dict(alpha_start=float(_al_start), alpha_end=float(_al_start),
                             evals=0, stabilised=True, wall=0.0, skipped=True)

    # Take the whole allocation unless told otherwise.  |Omega| starts around
    # 48 and grows past 120, so early calls leave workers idle while later ones
    # use them all; sizing the pool to the node costs nothing and the constraint
    # block is the one that scales.
    if a.nproc < 0:
        try:
            from parallel import ParallelOracle as _PO
            a.nproc = _PO.available_cores()
        except Exception:
            a.nproc = 1
    _pool = None
    _par = None
    if a.nproc and a.nproc > 1:
        import multiprocessing as _mp
        _pool = _mp.get_context('fork').Pool(processes=a.nproc)
        nseg = max(1, a.cert_nseg)
        D.cert_pool = _pool if nseg > 1 else None
        D.cert_nseg = nseg
        try:
            if MIMO:
                raise NotImplementedError('MIMO: the worker pool builds SISO packers (the SQP path does not use it)')
            from parallel import ParallelOracle
            _par = ParallelOracle(a.ne, a.r, a.sob_len, a.case, pk_args,
                                  nproc=a.nproc,
                                  flags_extra=(dict(sym=True,
                                                    sym_branch=flags[
                                                        'sym_branch'])
                                               if a.sym else None))
        except Exception as _e:
            print(f"  parallel constraints unavailable ({type(_e).__name__}); "
                  f"running them serially", flush=True)
            _par = None
        print(f"pool: {a.nproc} workers for the |Omega| constraints; "
              + (f"stability certificate SERIAL"
                 if D.cert_nseg <= 1 else
                 f"certificate in {D.cert_nseg} segments"), flush=True)

    _band_last = dict(lam=[], info={})          # the last located set (dumps, records)
    n_allw_outside = [0]                         # all-w certificates that found the norm outside the band
    n_band_incomplete = [0]

    def band_from_closedloop(arow_, nvec_, K_, b_, c_, count=False, extra=()):
        """Half-width covering every closed-loop eigenvalue that can raise a
        peak: those with Re >= -a.band_recut.  1.5x their max |Im|, plus the
        pad, floored at the current band.

        SPARSE since 2026-09-29 (STAGE6 6.13; was the dense spectrum of the
        closed loop at every acceptance).  Only eigenvalues that can WIDEN the
        band matter: those right of -band_recut with 1.5 |Im| + pad > the band.
        They are LOCATED from candidates -- plant eigenvalues, compensator poles,
        the high-gain estimates eig(D_c C^H M^-1 B) (the branches that go far),
        the near-line eigenvalues the abscissa certificate harvested (`extra`) --
        refined by shift-invert (a few shifts per acceptance).
        count=True (at save): the whole band tiled and the located set compared
        with the winding number right of -band_recut -- equal means none is
        missing at any |Im|.  Same sets as the dense spectrum on every record
        tested (STAGE6 6.13)."""
        import sparse_spectrum as _ss
        half = max(a.whi, -a.wlo)
        lam, info = _ss.closed_loop_set(D.A, D.M, D.cl(b_, c_), arow_, nvec_, K_, -a.band_recut,
                                        band=half, mc=D.mc, count=count, inner=max(0.0, (half - a.wpad) / 1.5 - 0.5),
                                        extra=[complex(z_) for z_ in extra],
                                        ctrl_exact=(a.wind_ctrl == 'exact'))
        _band_last.update(lam=list(lam), info=info)
        D._band_last_lam = list(lam)
        if count and not info.get('complete'):
            n_band_incomplete[0] += 1
            print(f"    WARNING band: {info.get('located')} closed-loop eigenvalues located right of "
                  f"Re={-a.band_recut:g} but the winding count says {info.get('count')}: the band may miss "
                  f"a resonance", flush=True)
        need = 1.5 * max(abs(z.imag) for z in lam) + a.wpad if lam else 0.0
        return max(need, a.whi)

    def band_from_controller(arow_):
        """Widen the band so it covers every controller pole.

        A resonance sits near each controller pole, so a band that stops short
        of them hides peaks: the reported gamma is then the height of whatever
        peak happened to be inside.  The controller block is r x r, so its
        spectrum is free to compute.  The band only ever GROWS -- --wlo/--whi
        are a floor, not the value.
        """
        try:
            im = np.abs(np.linalg.eigvals(ctl.A_of(*arow_)).imag)
            need = float(im.max()) + a.wpad if im.size else 0.0
        except Exception:
            need = 0.0
        return min(a.wlo, -need), max(a.whi, need)

    # With |Im lambda_ctrl| <= wmax enforced, the band no longer has to chase
    # the poles: it is fixed, and the constraint keeps the compensator inside
    # it.  The band is only widened if the STARTING design already violates
    # the constraint, so the first iterates are still certified honestly.
    # OMEGA_MAX IS FIXED AND COMMON, AND COMES FROM THE PLANT.
    #
    # It cannot come from the seeds.  gamma is a supremum over the band, so a
    # run with a narrow band can report a LOWER gamma than one with a wide band
    # for the same design -- per-seed bands are not comparable.  And the seed
    # spread is enormous: |Im lambda| runs 7.8 to 56.3 across the catalogue, so
    # the worst member (rho = 1e-4) would set +-127 for every case, tripling the
    # frequencies to certify.
    #
    # The plant's own scale is small: the twenty rightmost modes have
    # |Im lambda| <= 1.0 and sigma_1 peaks near omega = -0.65.  A compensator
    # resonating at |Im| = 56 is fifty times beyond anything the plant does; it
    # is an artefact of cheap control, not a design.  omega_max is therefore a
    # FIXED multiple of the plant scale, the band is 3/2 of it, and seeds that
    # do not fit are EXCLUDED rather than allowed to widen the band.
    if a.ctrl_stab == 'off':
        a.ctrl_max = 1e9   # the abscissa branch never fires; the gate allows all
    wmax = a.ctrl_wmax if a.ctrl_wmax > 0 else a.wmax_plant
    _half = 1.5 * wmax
    a.wlo, a.whi = -_half, _half
    print(f"band [{a.wlo:.2f}, {a.whi:.2f}]; controller poles |Im| <= "
          f"{wmax:.2f} (3/2 x max seed |Im|, band = 3/2 x wmax)", flush=True)

    wlo_run, whi_run = band_from_controller(arow0)
    if (wlo_run, whi_run) != (a.wlo, a.whi):
        print(f"start design has poles beyond the band; widened to "
              f"[{wlo_run:.2f}, {whi_run:.2f}] for the first iterates",
              flush=True)
    a.wlo, a.whi = wlo_run, whi_run
    if a.ctrl_imcap == 'on':
        print(f"controller poles constrained to |Im| <= {wmax:.2f} "
              f"(2/3 of the certified band)", flush=True)
    else:
        print(f"controller poles NOT constrained in |Im| (--ctrl-imcap off); "
              f"the band follows the closed-loop spectrum at each accepted "
              f"design, from [{a.wlo:.2f}, {a.whi:.2f}]", flush=True)
    print(f"stage-1 flags: select-cons={a.select_cons} ctrl-imcap={a.ctrl_imcap} "
          f"ctrl-stab={a.ctrl_stab} cont-anchor={a.cont_anchor} "
          f"opt-tol={a.opt_tol:g}", flush=True)

    # ---- initial active set ----
    # A surrogate built on one or two frequencies is far too coarse: the inner
    # solve then walks to a point where an UNMODELLED frequency is enormous
    # (certified norms of 60-400 were observed with |Omega| <= 3), and several
    # outer iterations are wasted before the exchange catches up.  Seed Omega
    # instead with every local maximum plus a grid covering the band wherever
    # sigma_1 is within `seed_frac` of the peak.  Each extra frequency costs one
    # resolvent solve per oracle call, so a few tens are cheap.
    g0, w0, ok0, peaks0 = certified_norm(D, pk, z, a.wlo, a.whi, gam_lb=0.0)
    if _ctrl_borrowed:
        # The parent guarantee -- "a child started from a parent cannot finish
        # above it" -- holds only if the parent's own design is a candidate
        # start.  Evaluate the default controller on the same fields and keep
        # whichever start is lower.
        _pk_alt = Packer(D, flags, *_ctrl_default, xa0, xs0, va0, vs0)
        _z_alt = _pk_alt.z0()
        _g_alt, _w_alt, _ok_alt, _pk_alt_peaks = certified_norm(
            D, _pk_alt, _z_alt, a.wlo, a.whi, gam_lb=0.0)
        print(f"start candidates: borrowed controller gamma={g0:.6f}, "
              f"default controller gamma={_g_alt:.6f}", flush=True)
        if _g_alt < g0:
            print("  -> the borrowed controller is WORSE on these fields; "
                  "starting from the default instead", flush=True)
            pk, z = _pk_alt, _z_alt
            g0, w0, ok0, peaks0 = _g_alt, _w_alt, _ok_alt, _pk_alt_peaks
    arow_, nvec_, K_, xa_, xs_, va_, vs_ = pk.unpack(z)
    b_, c_, *_ = D.bc(xa_, xs_, va_, vs_)
    cl_ = D.cl(b_, c_)
    # SEED OMEGA ON A RESOLUTION, NOT A COUNT.
    #
    # A fixed count spread uniformly over the band fails as soon as the band is
    # wide: at +-43.65 with nseed = 48 the spacing is 1.8, coarser than the peak
    # itself (near omega = -0.5), so the surrogate never sees the region that
    # matters -- |Omega| collapsed to 4-15 and every result degraded (FB+FC
    # went 4.34 -> 9.71).  Sample at a fixed RESOLUTION instead, so a wider band
    # costs more points rather than a coarser grid, and keep only where
    # sigma_1 is significant.
    dw = a.seed_dw
    wg = np.linspace(a.wlo, a.whi, max(int((a.whi - a.wlo) / dw) + 1, 64))
    sg = np.array([cl_.gain(float(w), arow_, nvec_, K_,
                            want_vectors=False)['sigma'] for w in wg])
    keep = list(wg[sg >= a.seed_frac * sg.max()])
    # plus every local maximum, however small -- these are the frequencies the
    # exchange would otherwise have to discover one at a time
    i_ = np.arange(1, len(sg) - 1)
    loc = i_[(sg[i_] >= sg[i_ - 1]) & (sg[i_] >= sg[i_ + 1])]
    keep += [float(wg[j]) for j in loc]
    if len(keep) > a.nseed:
        idx = np.linspace(0, len(keep) - 1, a.nseed).astype(int)
        keep = [keep[j] for j in idx]
    _resume_state = None
    Omega = sorted(set(round(float(w), 6)
                       for w in list(keep) + [w0] + peaks0))
    # a resumed run continues with the working set it had (the checkpoint now
    # carries it): without this, Omega restarted from the design's own peaks
    # and the rounds spent discovering the band were lost (87 -> 76 members)
    try:
        # checkpoint resumes only: a frozen --resume from a source record keeps
        # the frozen working-set construction (the 27-case replay depends on it)
        if os.path.exists(tag + '_ckpt.npz'):
            _srcw = np.load(tag + '_ckpt.npz', allow_pickle=True)
            # the loop's own state (2026-09-22): a resume that reset the adaptive
            # inner budget to its start value and dropped the SQP trust radius
            # and the peak members biased every comparison crossing a restart
            if 'maxit_cur' in _srcw.files:
                _resume_state = dict(maxit_cur=int(_srcw['maxit_cur']),
                                     sqp_trust=float(_srcw['sqp_trust']),
                                     peakset=[float(x) for x in np.asarray(_srcw['peakset']).ravel()])
            if 'whi' in _srcw.files and float(_srcw['whi']) > a.whi:
                a.wlo, a.whi = float(_srcw['wlo']), float(_srcw['whi'])   # the band the run had reached
            if 'active_freqs' in _srcw.files and len(_srcw['active_freqs']):
                Omega = sorted(set(list(Omega) + [round(float(x), 6) for x in _srcw['active_freqs']
                                                  if a.wlo <= float(x) <= a.whi]))
    except Exception:
        pass
    # ---- STAGE 3, note items 4-5 (--omega-policy an): working-set state.
    _starter = set(round(float(w), 6) for w in keep)   # the uniform grid, retired after the first certified round
    # a coarse uniform SKELETON of the grid that never retires: without it the
    # surrogate has no coverage of the band and the inner solve creates peaks
    # unobserved (hard cap at 50 with no skeleton: six rounds, certified
    # 29-39 against t ~ 9.7, no progress at all)
    _skel = sorted(_starter)
    if a.omega_skeleton > 0 and len(_skel) > a.omega_skeleton:
        _idx = np.linspace(0, len(_skel) - 1, a.omega_skeleton).round().astype(int)
        _skel = [_skel[j] for j in _idx]
    _skeleton = set(_skel)
    _dormant = {}                                       # freq -> consecutive rounds below (1 - eps_w) gamma
    _born = {w: 0 for w in Omega}                       # round a member was added (never retire the young)
    _refuter = {}                                       # freq -> round it refuted a trial design
    _omega_log = []
    t0 = g0
    print(f"start gamma={g0:.6f}, |Omega|={len(Omega)}", flush=True)
    if a.resume and _resume_gamma is not None and g0 > 10.0 * _resume_gamma:
        # A resumed start an order of magnitude above the record it came from
        # is a failed transplant (a controller-carrying donor into a
        # static-gain case lands on the open loop, 882.88), not a start.  Job
        # 16152993 ran three such donor attempts to their full budget, ~25 min
        # of a 96-core node for nothing.  Refuse in seconds instead.
        print(f"FATAL: resumed start gamma={g0:.4f} is {g0 / _resume_gamma:.0f}x "
              f"the source record's {_resume_gamma:.4f}; the transplant lost "
              f"the design.  Not a usable start.", flush=True)
        return

    hist = []
    Hprev = None
    nostall = 0
    # STAGE 0 INSTRUMENTATION.  lu_counter wraps scipy's splu, so nothing in
    # the production or certification code is touched; the counts are of THIS
    # process only, which is what "per inner iteration" means -- the pool
    # workers factorise in their own processes and are accounted for by the
    # oracle-call count instead.
    _luc, _lu_prev = None, None
    if a.dump or a.lucount:
        try:
            import lu_counter as _luc
            _luc.BIG_N = int(D.n)            # full-model factorisations only in n_lu_full
            _luc.install()
            _luc.reset()
            _lu_prev = _luc.snapshot()
            print("lu_counter: factorisations and solves are being counted",
                  flush=True)
        except Exception as _e:
            print(f"lu_counter unavailable ({type(_e).__name__}); "
                  f"factorisation counts will be absent", flush=True)
            _luc = None
    n_inconcl = [0]      # rejections because nothing could be PROVEN
    n_located = [0]      # rejections because an eigenvalue was FOUND
    n_cont_fallback = [0]  # reduced continuation gate: full-model fallbacks (tracks + midpoint certificates)
    n_clpair = [0]       # rounds with two closed-loop eigenvalues coalescing on the line (detector, STAGE5 30)
    n_witness_bad = [0]  # rounds whose certificate witness does not attain the certified value (must stay 0)
    # Trust radius depends on the START.  A COLD start must travel far, so it
    # needs a loose radius (1.5).  A WARM start begins at a good design and the
    # first inner solve can wreck it -- KG+FBR+FCR resumed from 4.1971 and its
    # first outer returned 42.9 -- so it needs a tight one.
    warm_started = bool(src and os.path.exists(src))
    trust = a.trust if a.trust_warm < 0 or not warm_started else a.trust_warm
    if warm_started:
        print(f"warm start -> trust radius {trust:g}", flush=True)
    exch_converged = False
    exch_fixed_point = False
    exch_progress_stop = False
    _gh_hist = []
    exch_stalled = False
    per_outer = (a.wall - (time.time() - t_start) - 45.) / max(a.outer, 1)
    gbest, zbest, albest = g0, z.copy(), None
    # ---- continuation anchor ----
    # The whole stability argument is a chain: stable start, then no crossing
    # on any accepted step.  The chain is only as good as its anchor, so the
    # START design is verified here -- matrix-free line sweep always (which
    # also harvests the initial watch list for free), plus a CHECK-ONLY dense
    # eigensolve at GL scale.  An infeasible anchor aborts: every certificate
    # downstream would be answering a conditional question whose premise is
    # already false (that is precisely how one unstable parent propagated
    # byte-identically into its child).
    watch_cur, zwatch = [], zbest.copy()
    _tau_rho, _tau_streak = None, 0     # step governor: off until evidence
    _anchor_ok = True                   # False = chain not yet anchored
    _lam0_start = None
    _sigma2_prev = set()                # sigma_2 rows enrolled last round (frequencies)
    _maxit_cur = int(a.inner_maxit)     # adaptive inner budget (--inner-maxit-adapt)
    _sqp_trust_prev = None              # SQP: trust radius carried across rounds (--sqp-carry)
    _peakset = set(round(float(x), 6) for x in ([w0] + list(peaks0)))   # members enrolled as PEAKS (followed with --peak-follow)
    _peaks_of_best = [float(w0)] + [float(x) for x in peaks0]
    if _resume_state:
        _maxit_cur = max(int(_resume_state['maxit_cur']), int(a.inner_maxit_min)) if a.inner_maxit_adapt else _maxit_cur
        _sqp_trust_prev = (_resume_state['sqp_trust'] or None)
        _peakset.update(round(float(x), 6) for x in _resume_state['peakset'])
        print(f"resume: inner budget {_maxit_cur}, SQP trust {_sqp_trust_prev}, {len(_resume_state['peakset'])} peak members restored", flush=True)
    gbest_cert, zbest_cert = float(gbest), (zbest.copy() if zbest is not None else None)   # --cert-every: last FULLY certified best
    _al_accepted = None                 # certified abscissa of the accepted design (SQP start check)
    # ---- STAGE 5: the projected oracle for the inner loop ------------------
    _ro = None
    if a.oracle == 'reduced':
        from reduced import ReducedOracle
        import reduced as _red_mod
        _red_mod.ROW_WORKERS = max(0, int(a.row_workers))
        if _red_mod.ROW_WORKERS:
            print(f"rows: gain_batch in {_red_mod.ROW_WORKERS} forked worker(s) + this process", flush=True)
        _ro = ReducedOracle(D)
        certified_norm.oracle = _ro          # fine reduced sweep for the peaks list
        if a.oracle_inputs == 'on':
            # position-robust model (STAGE5 5): input bases over the position
            # ranges + union Galerkin; for K-only cases the Galerkin alone
            from reduced import input_basis
            _Pb = input_basis(D, 'b', a.xa_lo, a.xa_hi, other=0.98)[0] if flags.get('B') else None
            _Pc = input_basis(D, 'c', a.xs_lo, a.xs_hi, other=-1.03)[0] if flags.get('C') else None
            # free fields on a basis: b = M va/||va||_M lies in span(M Phi_a)
            for _fbk, _fik, _w in (('field_basis_a', 'field_input_a', 'b'), ('field_basis_s', 'field_input_s', 'c')):
                _Phi = getattr(D, _fbk, None)
                if _Phi is None:
                    _Phi = getattr(D, _fik, None)          # atoms: the basis is the model's input basis only
                if _Phi is not None:
                    _Q = np.linalg.qr(np.asarray(D.M @ _Phi, complex))[0]
                    if _w == 'b':
                        _Pb = _Q
                    else:
                        _Pc = _Q
            if _Pb is not None or _Pc is not None:
                _ro.set_input_basis(_Pb, _Pc)
            _ro.galerkin = True
            print(f"oracle: position-robust (union Galerkin"
                  f"{', input basis b %d' % _Pb.shape[1] if _Pb is not None else ''}"
                  f"{', c %d' % _Pc.shape[1] if _Pc is not None else ''})", flush=True)
    certified_norm.sweep_n = int(a.cert_sweep)
    csp.DISK_STRICT = (a.cert_disk == 'strict')
    certified_norm.rtol = float(a.cert_rtol)
    if a.oracle == 'reduced':
        # MEMOISE the closed-loop abscissa certificate per design: the SQP's
        # certified validation and the gate certified the identical design one
        # after the other (24 full factorisations each, per round).  The
        # eigs_seen list the gate reads is replayed from the memo.
        _ac_orig = D.alpha_certified
        _ac_memo = {}

        def _ac_cached(arow_, nvec_, K_, b_, c_, eps=1e-3, eigs_seen=None, **kw):   # same default as fe_codesign
            key = (tuple(np.round(np.concatenate([np.asarray(v, complex).ravel() for v in arow_]), 14)),
                   tuple(np.round(np.asarray(nvec_, complex).ravel(), 14)), tuple(np.round(np.asarray(K_, complex).ravel(), 14)),
                   hash(np.round(np.asarray(b_, complex), 14).tobytes()),
                   hash(np.round(np.asarray(c_, complex), 14).tobytes()), eps, tuple(sorted(kw.items())))
            if key in _ac_memo:
                out, seen = _ac_memo[key]
                if eigs_seen is not None:
                    eigs_seen.extend(seen)
                return out
            seen_local = []
            out = _ac_orig(arow_, nvec_, K_, b_, c_, eps=eps, eigs_seen=seen_local, **kw)
            if eigs_seen is not None:
                eigs_seen.extend(seen_local)
            if len(_ac_memo) > 8:
                _ac_memo.pop(next(iter(_ac_memo)))
            _ac_memo[key] = (out, list(seen_local))
            return out
        D.alpha_certified = _ac_cached
    certified_norm.band_check = bool(a.band_check)
    certified_norm.all_w = (a.cert_all_w == 'on')
    _anchored = {}                       # freq -> (xa, xs) it was last anchored with
    _last_peaks = [float(w0)] + [float(x) for x in peaks0]
    _oracle_log = []
    _al0_lam = None                      # rightmost eigenvalue of the accepted design (from the gate)
    try:
        _al0_lam = complex(_lam0_start)  # the start certification's rightmost eigenvalue (round-0 eigen-anchor)
    except Exception:
        pass

    def _sweep_anchor(z, rounds=4):
        """Anchor where the reduced model disagrees with the certifier's last
        full sweep (241 points already paid for), largest disagreements first,
        until the band agrees to oracle_sweep_tol or `rounds` passes."""
        _sw = getattr(certified_norm, 'last_sweep', None)
        if _sw is None:
            return 0.0
        # the sweep must be OF this design (2026-09-26): after a compaction the
        # loop compared the model at the best design with the sweep of the last
        # certified CANDIDATE -- a refused one differs (ne=200 rounds 4 and 6:
        # 'disagreements' of 70 %, three spurious anchors each time)
        _zs = getattr(certified_norm, 'last_sweep_z', None)
        if _zs is None or _zs.shape != np.shape(z) or not np.allclose(_zs, np.asarray(z, float), rtol=0, atol=1e-14):
            return float('nan')
        _ws_, _gs_ = _sw
        aa_, nn_, KK_, xa_, xs_, va_, vs_ = pk.unpack(z)
        bb_, cc_, *_ = D.bc(xa_, xs_, va_, vs_)
        _mask = _gs_ >= a.oracle_band_frac * float(np.max(_gs_))
        worst = np.inf
        for _ in range(rounds):
            rc_ = _ro.make_cl(bb_, cc_, D.r, D.mc)
            rs_ = np.array([rc_.gain(float(x), aa_, nn_, KK_, want_vectors=False)['sigma'] for x in _ws_])
            rel = np.where(_mask, np.abs(rs_ - _gs_) / np.maximum(_gs_, 1e-12), 0.0)
            worst = float(rel.max())
            cand = [int(j) for j in np.argsort(-rel)[:a.oracle_sweep_anchors] if rel[j] > a.oracle_sweep_tol]
            cand = [j for j in cand if all(abs(_ws_[j] - q) > 1e-6 for q in _anchored)]
            if not cand:
                break
            _enrich([_ws_[j] for j in cand], z, f"sweep disagreement {[f'{rel[j]:.0e}' for j in cand]}")
        return worst

    def _enrich(freqs, z, why):
        """One full-model factorisation per anchor (the certifier's kind)."""
        aa_, nn_, KK_, xa_, xs_, va_, vs_ = pk.unpack(z)
        bb_, cc_, *_ = D.bc(xa_, xs_, va_, vs_)
        clf = D.cl(bb_, cc_)
        nadd = 0
        for w_ in freqs:
            w_ = round(float(w_), 6)
            prev = _anchored.get(w_)
            # re-anchor when the fields moved (Woodbury columns refresh)
            if prev is not None and np.all(np.abs(np.asarray(prev[0]) - xa_) < 1e-9) \
                    and np.all(np.abs(np.asarray(prev[1]) - xs_) < 1e-9):
                continue
            _ro.enrich(clf, w_, aa_, nn_, KK_, nvec_dirs=a.oracle_nvec)
            _anchored[w_] = (np.array(xa_, float), np.array(xs_, float)); nadd += 1
        if nadd:
            print(f"    oracle: +{nadd} anchor(s) ({why}) -> dim {_ro.dim}", flush=True)
        return nadd
    if a.cont:
        _seen0 = []
        _aa0, _nn0, _KK0, _xa0, _xs0, _va0, _vs0 = pk.unpack(zbest)
        _bb0, _cc0, *_ = D.bc(_xa0, _xs0, _va0, _vs0)
        _al0_tuple = D.alpha_certified(_aa0, _nn0, _KK0, _bb0, _cc0, eps=a.eps,
                                       eigs_seen=_seen0)
        _al0 = _al0_tuple[0]
        _lam0_start = _al0_tuple[1]      # stage 5: the round-0 eigen-anchor
        if _al0 + a.eps > 0.0 and a.cont_anchor == 'defer':
            # STAGE 1, item 3.  The continuation gate postdates the frozen
            # campaign (no frozen record carries it; case.slurm never passed
            # it), and the frozen KGC record shows round 0 starting from the
            # open loop at gamma 882.88 -- exactly the start this check refuses.
            # The chain argument needs a feasible ANCHOR, not a feasible start:
            # run without the gate until the first feasible design is accepted,
            # anchor there, and enforce the chain from then on.  The final
            # winding certificate is unconditional either way.  Measured at
            # ne=100: KGC from the open loop reaches 11.684, certified.
            _anchor_ok = False
            print(f"start design is infeasible (certificate alpha="
                  f"{_al0:+.4f}); continuation gate DEFERRED until the first "
                  f"feasible accepted design", flush=True)
        elif _al0 + a.eps > 0.0:
            print(f"FATAL: start design is infeasible (certificate alpha="
                  f"{_al0:+.4f}); the continuation chain has no anchor. "
                  f"Fix the seed/warm start instead of running -- or pass "
                  f"--cont-anchor defer (see the frozen KGC record).",
                  flush=True)
            return
        if _anchor_ok and a.dense_check and (D.n + D.r) <= a.dense_nmax:
            _ad0, _ld0 = D.alpha_dense(_aa0, _nn0, _KK0, _bb0, _cc0)
            if _ad0 + a.eps > 0.0:
                print(f"FATAL: start design FAILS the dense check "
                      f"(alpha_dense={_ad0:+.4f} at {_ld0:.4g}) although the "
                      f"line sweep read {_al0:+.4f} -- an inherited escape. "
                      f"Not running.", flush=True)
                return
        elif a.dense_check:
            print(f"dense anchor check SKIPPED (n+r={D.n + D.r} > "
                  f"{a.dense_nmax}): the continuation gate is the only "
                  f"protection at this scale", flush=True)
        if _anchor_ok:
            watch_cur = _topk_watch(_seen0, a.cont_k, a.eps)
            print(f"continuation anchor verified: alpha={_al0:+.4f}, watching "
                  f"{len(watch_cur)} near-line eigenvalue(s)", flush=True)
    for it in range(a.outer):
        # ALWAYS restart the inner solve from the best CERTIFIED point.  The
        # inner problem only sees Omega, so its optimum can sit where an
        # unmodelled frequency is huge; accepting that point and continuing from
        # it makes the exchange diverge.  The certified value is the only
        # trustworthy measure of progress.
        # ---- STAGE 5: anchor the reduced oracle for this round --------------
        if _ro is not None:
            _nlu = 0
            if _ro.dim > a.oracle_dim_cap:
                # restart from the current design's certified peaks and the
                # watch-list eigenvalues: the basis carries directions from
                # designs long left behind
                _aa_, _nn_, _KK_, _xa_, _xs_, _va_, _vs_ = pk.unpack(zbest)
                _bb_, _cc_, *_ = D.bc(_xa_, _xs_, _va_, _vs_)
                _keep = sorted(set(round(float(x), 6) for x in _last_peaks)
                               | set(round(float(w), 6) for w in sorted(Omega)[:0]))
                _lams_c = [complex(x) for x in (watch_cur or [])] + ([_al0_lam] if _al0_lam is not None else [])
                _old = _ro.dim
                _ro.compact(D.cl(_bb_, _cc_), _keep, _aa_, _nn_, _KK_, lams=_lams_c, nvec_dirs=a.oracle_nvec)
                _anchored.clear()
                for w_ in _keep:
                    _anchored[w_] = (np.array(_xa_, float), np.array(_xs_, float))
                _nlu += len(_keep) + len(_lams_c)
                print(f"    oracle: basis compacted {_old} -> {_ro.dim}", flush=True)
                _sweep_anchor(zbest)
            _nlu += _enrich(_last_peaks, zbest, "certified peaks")
            # phase 2: the near-line eigenvectors (the continuation watch list
            # plus the accepted design's rightmost) into the basis, one
            # factorisation each, so the reduced pencil's rightmost Ritz value
            # is the true abscissa (residual-filtered)
            _aa_, _nn_, _KK_, _xa_, _xs_, _va_, _vs_ = pk.unpack(zbest)
            _bb_, _cc_, *_ = D.bc(_xa_, _xs_, _va_, _vs_)
            _clf_ = D.cl(_bb_, _cc_)
            _lams = [complex(x) for x in (watch_cur or [])]
            if _al0_lam is None and _lam0_start is not None:
                _al0_lam = complex(_lam0_start)   # round 0: the start certificate's rightmost eigenvalue
            if _al0_lam is not None:
                _lams.append(_al0_lam)
            _ne = _ro.enrich_eig(_clf_, _lams, _aa_, _nn_, _KK_)
            _nlu += _ne
            _RED_AB_STATS['reduced'] = _RED_AB_STATS['fallback'] = 0
            if it == 0:
                # warm-up before the first inner solve, against the START
                # certification's sweep: round 0 otherwise runs on a basis of
                # dimension ~12 that is 1 % off and sets a different path
                _sweep_anchor(zbest)
                if a.enroll_rel > 0:
                    # the START certification ran before the model had anchors,
                    # so its peak list came from the coarse full reference sweep
                    # (spacing 1.0 with --cert-sweep 61; the frozen 241 points
                    # gave 0.25) and missed near-maximal narrow peaks (1.91 at
                    # 99.1 % at K+B+C ne=200).  Re-derive it on the now-anchored
                    # model: fine sweep, every peak within enroll_rel, at its top.
                    _aa_, _nn_, _KK_, _xa_, _xs_, _va_, _vs_ = pk.unpack(zbest)
                    _bb_, _cc_, *_ = D.bc(_xa_, _xs_, _va_, _vs_)
                    _rc0 = _ro.make_cl(_bb_, _cc_, D.r, D.mc)
                    _s0f = lambda w_: _rc0.gain(float(w_), _aa_, _nn_, _KK_, want_vectors=False)['sigma']
                    _wf0 = np.linspace(a.wlo, a.whi, 1201)
                    _pk0 = _enrol_peaks(_wf0, np.array([_s0f(x) for x in _wf0]), _s0f, rel=float(a.enroll_rel))
                    _new0 = [round(float(x), 6) for x in _pk0 if all(abs(x - q) > 1e-3 for q in Omega)]
                    Omega = sorted(set(list(Omega) + _new0)); _peakset.update(round(float(x), 6) for x in _pk0)
                    print(f"    start peaks re-derived on the reduced model: {len(_pk0)} within "
                          f"{100 * a.enroll_rel:.0f} % ({len(_new0)} new): {[round(x, 3) for x in sorted(_pk0)]}", flush=True)
            # greedy (ABMSV): while the reduced model claims a peak above the
            # certified level at an unanchored frequency, anchor there
            _aa_, _nn_, _KK_, _xa_, _xs_, _va_, _vs_ = pk.unpack(zbest)
            _bb_, _cc_, *_ = D.bc(_xa_, _xs_, _va_, _vs_)
            _Wg = np.union1d(np.asarray(sorted(Omega), float), np.linspace(a.wlo, a.whi, 241))
            for _g in range(a.oracle_greedy):
                _rc = _ro.make_cl(_bb_, _cc_, D.r, D.mc)
                _red = np.array([_rc.gain(float(w), _aa_, _nn_, _KK_, want_vectors=False)['sigma'] for w in _Wg])
                _j = int(np.argmax(_red))
                if _red[_j] <= gbest * (1.0 + a.oracle_tol) or any(abs(_Wg[_j] - q) < 1e-6 for q in _anchored):
                    break
                _nlu += _enrich([_Wg[_j]], zbest, f"greedy, reduced {_red[_j]:.4f} > certified {gbest:.4f}")
            _oracle_log.append(dict(round=it, dim=_ro.dim, anchors=len(_anchored), lu_anchoring=_nlu, eig_anchors=_ne))
        # ---- STAGE 1b, item 3: choose the sigma_2 rows for this round -------
        _enroll_idx = None
        if a.sigma2 == 'on':
            _W = np.asarray(sorted(Omega), float)
            _aa2, _nn2, _KK2, _xa2, _xs2, _va2, _vs2 = pk.unpack(zbest)
            _bb2, _cc2, *_ = D.bc(_xa2, _xs2, _va2, _vs2)
            # on the reduced oracle when it exists: |Omega| full LUs per round otherwise
            _cl2 = _ro.make_cl(_bb2, _cc2, D.r, D.mc) if _ro is not None else D.cl(_bb2, _cc2)
            _s1 = np.array([_cl2.gain(float(w), _aa2, _nn2, _KK2,
                                      want_vectors=False)['sigma'] for w in _W])
            _gref = float(np.max(_s1))
            _new = set()
            for _i, _w in enumerate(_W):
                _active = _s1[_i] >= _gref * (1.0 - a.sigma2_active)
                _near = _s1[_i] >= _gref * (1.0 - a.sigma2_near)
                _was = any(abs(_w - _wp) < 1e-9 for _wp in _sigma2_prev)
                if not (_active or _near or _was):
                    continue
                _p2 = _cl2.gain_k(float(_w), _aa2, _nn2, _KK2, k=2)
                _ratio = _p2[1]['sigma'] / max(_p2[0]['sigma'], 1e-300)
                # always at the active frequencies; elsewhere thresholded with
                # hysteresis: enter above 1 - sigma2_on, leave below 1 - sigma2_off
                # 2026-09-25: an ACTIVE frequency far from a crossing (gap > sigma2_active_gap) is
                # not a kink -- K+FB+FC's flat optimum enrolled 43 blocks with gaps ~0.87
                if _active and a.sigma2_active_gap is not None and (1.0 - _ratio) > a.sigma2_active_gap:
                    _active = False
                if _active or _ratio >= 1.0 - a.sigma2_on \
                        or (_was and _near and _ratio >= 1.0 - a.sigma2_off):
                    _new.add(float(_w))
                    print(f"    sigma2 row at w={_w:+.4f}: sigma1={_p2[0]['sigma']:.4f} "
                          f"sigma2={_p2[1]['sigma']:.4f} gap={1 - _ratio:.4f}"
                          f"{' (active)' if _active else ''}", flush=True)
            _sigma2_prev = _new
            _enroll_idx = [int(np.argmin(np.abs(_W - w))) for w in sorted(_new)]
            print(f"    sigma2 enrolled at {len(_enroll_idx)} of {len(_W)} frequencies",
                  flush=True)
        # PEAK-FOLLOWING members (STAGE5 9): every member enrolled as a peak
        _follow_dict = ({round(float(w_), 6): float(a.follow_radius) for w_ in Omega
                         if round(float(w_), 6) in _peakset}
                        if (a.peak_follow == 'on' and _ro is not None) else None)
        _prov_pre = bool(a.cert_every > 1 and _ro is not None and (it % a.cert_every != 0)
                         and it < a.outer - 1 and _ro.dim > 0 and a.prov_gate == 'reduced')
        if a.inner == 'sqp':
            from sqp_inner import solve_epigraph_sqp
            best, ncalls, Hcur, inner_stats = solve_epigraph_sqp(
                # the CURRENT certified best as the start level (t0 is the run's
                # start value, never updated: a stationary SQP reported it back
                # as t, e.g. 7.964064 at a design certified 7.951836)
                D, pk, flags, zbest, Omega, float(gbest), a.eps,
                maxit=_maxit_cur, ctrl_max=a.ctrl_max,
                enroll=_enroll_idx, kink_block=(a.sqp_block == 'on'),
                trust0=(max(_sqp_trust_prev, 0.1 * a.sqp_trust * max(float(np.linalg.norm(zbest)), 1.0))
                        if (a.sqp_carry and _sqp_trust_prev) else (a.sqp_trust if a.sqp_trust > 0 else None)),   # a collapsed radius is not carried (7.8e-5 froze the polish)
                sel_margin=a.sel_margin, verbose=bool(getattr(a, 'verbose', False)),
                H0=(Hprev if (a.warm_h or (a.sqp_carry and a.sqp_carry_h == 'on')) else None), ctrl_lmi=(a.sqp_lmi == 'on'),
                ctrl_block=(a.sqp_ctrl == 'block'), follow=_follow_dict, follow_radius=a.follow_radius,
                row_screen=a.row_screen, validate_full=not _prov_pre,
                cl_block=(a.sqp_clblock == 'on'),
                max_disp=(float(_tau_rho) if (a.tau and _tau_rho is not None) else None),   # the step governor INSIDE the solve
                oracle=_ro, reach=a.sqp_reach, res_tol=a.sqp_row_res_tol,
                refine_tol=(a.sqp_refine if a.sqp_refine > 0 else None),
                refine_grid=np.linspace(a.wlo, a.whi, 1201), refine_max=a.sqp_refine_max,
                alpha_ref=(float(_al_accepted) if _al_accepted is not None else None),
                pred_rel=(1e-7 if (a.sqp_stat == 'pred' or (a.sqp_stat == 'auto' and MIMO)) else None))
            _sqp_trust_prev = inner_stats.get('trust_final')
            # STALL RECOVERY (2026-09-25): a solve that ended on a collapsed radius
            # (code 6) or with failed subproblems hands its quasi-Newton matrix and
            # radius to the next round no more.  The ne=100 K+B+C run sat at
            # 7.770563 for four rounds (codes 6, subproblem failures) while the same
            # design with H = I and the default radius went to 7.665533 in 25
            # iterations offline.
            if inner_stats.get('termination_code') in (6, 12) or inner_stats.get('sub_failures'):
                _reset_carry = True
            else:
                _reset_carry = False
            # frequencies the inner solve found and added on the reduced model
            # join the exchange's working set (the certification is outside)
            _addf = inner_stats.get('added_freqs') or []
            if _addf:
                Omega = sorted(set(list(Omega) + [round(float(x), 6) for x in _addf]))
                print(f"    sqp refine: working set +{len(_addf)} -> |Omega|={len(Omega)}", flush=True)
            inner_stats.setdefault('quadprog_failure_rate', None)
            inner_stats.setdefault('qp_failure', False)
        else:
          best, ncalls, Hcur, inner_stats = solve_epigraph(
            D, pk, flags, zbest, Omega, t0,
              a.eps, per_outer, H0mode=a.h0,
              H0=(Hprev if a.warm_h else None),
              trust=(trust if a.trust > 0 else None),
              ctrl_max=a.ctrl_max, wmax=wmax,
              par=_par, tau0=a.tau0,
              budget=a.inner_budget, maxit=_maxit_cur,
              clock_guard=a.clock_guard, ls_fallback=a.ls_fallback, oracle=_ro,
              ctrl_line_gate=(a.ctrl_line == 'eps'), follow=_follow_dict,
              bandmax=((a.wlo, a.whi) if a.pyg_bandmax == 'on' else None),
              oracle_res_tol=a.oracle_res_tol,
              imcap=(a.ctrl_imcap == 'on'), select_cons=(a.select_cons == 'on'),
              opt_tol=a.opt_tol, sel_margin=a.sel_margin, enroll=_enroll_idx,
              qp_log=bool(a.qp_log), qp_eps=a.qp_eps, qp_maxiter=a.qp_maxiter,
              level_norm=(a.level_norm == 'on'), qp_solver=a.qp_solver)
        Hprev = Hcur
        if a.inner == 'sqp' and locals().get('_reset_carry'):
            Hprev = None; _sqp_trust_prev = None
            print("    sqp: stalled solve -- quasi-Newton matrix and radius reset for the next round", flush=True)
        if _ro is not None:
            _oracle_log[-1].update(abscissa_reduced=_RED_AB_STATS['reduced'],
                                   abscissa_fallback=_RED_AB_STATS['fallback'])
            print(f"    oracle: abscissa from the reduced pencil in {_RED_AB_STATS['reduced']} "
                  f"evaluations, full certificate in {_RED_AB_STATS['fallback']}", flush=True)
        if a.qp_log and inner_stats.get('qp_log'):
            _q = inner_stats['qp_log']
            print(f"    qp-log: {_q['n_iter']} it, mu {_q['mu_first']:.3g} -> "
                  f"{_q['mu_final']:.3g} (min {_q['mu_min']:.3g}), tv {_q['tv_final']:.2e}, "
                  f"stat {_q['stat_final']:.2e}, fallback<= {_q['fallback_max']}, "
                  f"osqp {_q['osqp_status']}"
                  + ("  *** mu STARVED" if _q['mu_starved'] else ""), flush=True)
        zc = best['z'] if best['z'] is not None else zbest
        if os.environ.get('CODESIGN_DEBUG_ROWS') and best['z'] is not None:
            # DEBUG (2026-09-24): the rows the SQP saw at its returned design
            _ws = (inner_stats or {}).get('w_star') or {}
            _aa_d, _nn_d, _KK_d, _xa_d, _xs_d, _va_d, _vs_d = pk.unpack(zc)
            _bb_d, _cc_d, *_ = D.bc(_xa_d, _xs_d, _va_d, _vs_d)
            _clf_d = D.cl(_bb_d, _cc_d)
            _rc_d = _ro.make_cl(_bb_d, _cc_d, D.r, D.mc) if _ro is not None else None
            print(f"    DEBUG rows: SQP t {best['t']:.6f}, smax {best['g']:.6f}; followed dict "
                  f"{'None' if _follow_dict is None else len(_follow_dict)} members; tops at the returned design:", flush=True)
            for _m in sorted(_follow_dict or {}):
                _top = _ws.get(_m)
                if _top is None:
                    print(f"      member {_m:+.4f}: NO top recorded", flush=True); continue
                _vr = _rc_d.gain(float(_top), _aa_d, _nn_d, _KK_d, want_vectors=False)['sigma'] if _rc_d else float('nan')
                _vf = _clf_d.gain(float(_top), _aa_d, _nn_d, _KK_d, want_vectors=False)['sigma']
                print(f"      member {_m:+.4f} -> top {_top:+.4f}: reduced {_vr:.6f}, full {_vf:.6f}", flush=True)
        if _follow_dict and best['z'] is not None:
            # move each followed member to its peak's top AT THE CANDIDATE, so
            # the next round starts on the tops (reduced model: no factorisation)
            from peakfollow import follow as _follow
            _aa_, _nn_, _KK_, _xa_, _xs_, _va_, _vs_ = pk.unpack(zc)
            _bb_, _cc_, *_ = D.bc(_xa_, _xs_, _va_, _vs_)
            _rcf = _ro.make_cl(_bb_, _cc_, D.r, D.mc)
            _moved = 0; _newO = set(Omega)
            for _w in list(_follow_dict):
                _ws, _ = _follow(_rcf, _w, _aa_, _nn_, _KK_, float(a.follow_radius))
                _ws = round(float(_ws), 6)
                if abs(_ws - _w) > 1e-4 and all(abs(_ws - q) > 1e-3 for q in _newO if q != _w):
                    _newO.discard(_w); _newO.add(_ws); _peakset.discard(_w); _peakset.add(_ws); _moved += 1
            # MERGE members closer than 1e-3 (following can bring several onto one
            # top; duplicate rows made the QP nearly degenerate), keeping a peak
            # member where there is one
            _merged = []
            for _q in sorted(_newO):
                if _merged and abs(_q - _merged[-1]) < 1e-3:
                    if _q in _peakset and _merged[-1] not in _peakset:
                        _merged[-1] = _q
                    continue
                _merged.append(_q)
            _nmerge = len(_newO) - len(_merged)
            Omega = _merged
            if _moved or _nmerge:
                print(f"    peak-following: {_moved} member(s) moved to their tops, {_nmerge} merged", flush=True)
        # BUNDLE-STYLE STEP GOVERNOR (proximity control, Apkarian-Noll).  In
        # the proximity-control bundle method the parameter tau bounds the
        # trial step and is the load-bearing piece of the convergence theory;
        # here it does double duty as the mechanism that keeps steps inside
        # what the CONTINUATION GATE can certify cheaply.  Unlike the ablated
        # BFGS trust region (which throttled the inner solver and was measured
        # harmful on gamma), this cap acts on the ACCEPTED-STEP level and is
        # SELF-CALIBRATING: it stays infinite until the gate produces evidence
        # (a homotopy split or a rejection), is then set from the offending
        # step length, and relaxes again after consecutive clean acceptances.
        # It never throttles before evidence and never touches the inner
        # solver.
        if a.tau and _tau_rho is not None and a.inner != 'sqp':   # the SQP honours the cap inside its solve (max_disp)
            _dz = zc - zbest
            _ndz = float(np.linalg.norm(_dz))
            if _ndz > _tau_rho:
                zc = zbest + (_tau_rho / _ndz) * _dz
                print(f"    tau governor: step clipped {_ndz:.3g} -> "
                      f"{_tau_rho:.3g}", flush=True)
        # the band the CANDIDATE needs, before its certificate (the frozen
        # order widened it afterwards: a design whose loop puts a near-axis
        # eigenvalue outside the current band was certified over too narrow a
        # band and accepted with a too-low gamma; the next candidate then saw
        # the wider band).  The band only grows.
        if a.ctrl_imcap == 'off' and a.band_first:
            try:
                _aa0, _nn0, _KK0, _xa0, _xs0, _va0, _vs0 = pk.unpack(zc)
                _bb0, _cc0, *_ = D.bc(_xa0, _xs0, _va0, _vs0)
                _hw0 = band_from_closedloop(_aa0, _nn0, _KK0, _bb0, _cc0)
                if _hw0 > a.whi * (1.0 + 1e-9):
                    print(f"    band widened to [{-_hw0:.2f}, {_hw0:.2f}] from the candidate's "
                          f"closed-loop spectrum (before its certificate)", flush=True)
                    a.wlo, a.whi = -_hw0, _hw0
            except Exception:
                pass
        _prov = bool(a.cert_every > 1 and _ro is not None and (it % a.cert_every != 0)
                     and it < a.outer - 1 and _ro.dim > 0)
        certified_norm.provisional = _prov
        certified_norm.all_w_below = float(gbest)         # a candidate above the best is refused anyway
        g, w, ok, peaks = certified_norm(D, pk, zc, a.wlo, a.whi,
                                         gam_lb=best['g'], w_hint=best.get('w'))
        certified_norm.all_w_below = None
        _aw = getattr(certified_norm, 'last_all_w', None)
        if _aw and _aw.get('ok') and np.isfinite(_aw.get('w', np.nan)) and abs(_aw['w']) > a.whi \
                and _aw.get('g', 0.0) >= g * (1.0 - 1e-12):
            # the all-w question found the norm OUTSIDE the band (STAGE6 6.21): the
            # band grows to cover it, the witness joins Omega as usual
            _hw_new = 1.2 * abs(_aw['w']) + a.wpad
            print(f"    all-w certificate: peak {_aw['g']:.6f} at w = {_aw['w']:+.3f} outside the band; band widened "
                  f"to [{-_hw_new:.2f}, {_hw_new:.2f}]", flush=True)
            a.wlo, a.whi = -_hw_new, _hw_new
            n_allw_outside[0] += 1
        if not _prov and np.isfinite(g) and ok and np.isfinite(w):
            # WITNESS INVARIANT (2026-09-25): the certificate's witness must attain
            # the certified value, sigma_1(w) >= g (1 - rtol).  One full solve per
            # round (the certifier's kind).  The fake grid witness of STAGE5 24
            # ran for days in the logs as "10 % model error" without anyone
            # asserting this; now it is checked and reported every round.
            _aw, _nw, _Kw, _xaw, _xsw, _vaw, _vsw = pk.unpack(zc)
            _bw, _cw, *_ = D.bc(_xaw, _xsw, _vaw, _vsw)
            _sw = D.cl(_bw, _cw).gain(float(w), _aw, _nw, _Kw, want_vectors=False)['sigma']
            if _sw < g * (1.0 - max(10 * float(a.cert_rtol), 1e-6)):
                print(f"    WITNESS INVARIANT VIOLATED: sigma_1({w:+.5f}) = {_sw:.6f} < certified {g:.6f}", flush=True)
                n_witness_bad[0] += 1
        if _prov:
            # one factorisation: anchor the reduced witness so the model stays
            # exact where it just made a claim (ABMSV's greedy)
            _enrich([w], zc, f"provisional witness {g:.4f}")
            # coarse FULL spot check (a few factorisations): the reduced model
            # cannot see a peak its basis does not span (a claim of 8.55 was
            # 17.33 in truth); where the full value exceeds the claim, escalate
            # to a full certificate for this round
            _aa_, _nn_, _KK_, _xa_, _xs_, _va_, _vs_ = pk.unpack(zc)
            _bb_, _cc_, *_ = D.bc(_xa_, _xs_, _va_, _vs_); _clf = D.cl(_bb_, _cc_)
            _wchk = list(np.linspace(a.wlo, a.whi, a.prov_check)) + [float(x) for x in peaks[:4]] + [float(w)]
            _gchk = max(_clf.gain(float(x), _aa_, _nn_, _KK_, want_vectors=False)['sigma'] for x in _wchk)
            if _gchk > g * (1.0 + a.prov_tol):
                print(f"    provisional claim {g:.4f} vs full spot check {_gchk:.4f}: escalating to a full certificate",
                      flush=True)
                certified_norm.provisional = False
                g, w, ok, peaks = certified_norm(D, pk, zc, a.wlo, a.whi, gam_lb=best['g'], w_hint=best.get('w'))
                _prov = False
            else:
                print(f"    provisional certification (reduced model): {g:.6f} at w={w:+.3f} "
                      f"(spot check {_gchk:.4f} on {len(_wchk)} full points)", flush=True)
        # STABILITY GATE.  The certified norm is computed regardless of
        # stability, and for an unstable closed loop it is meaningless -- one
        # run reported gamma = 4.17 for a design with alpha = +15.5.  Accept an
        # iterate only if it is genuinely feasible.
        aa, nn, KK, xa_, xs_, va_, vs_ = pk.unpack(zc)
        bb, cc, *_ = D.bc(xa_, xs_, va_, vs_)
        _seen_c = []
        _wind_done = None                 # the candidate's winding count, once computed
        _gate_reduced = False
        if _prov and a.prov_gate == 'reduced':
            # PROVISIONAL GATE (2026-09-24): the closed-loop abscissa from the
            # reduced pencil's near-line Ritz pairs, each residual-checked
            # against the FULL operator (products, no factorisation); no
            # rectangle certificate, no continuation.  Completeness near the
            # line is established at the next full round, which re-certifies
            # the provisional best's stability (rollback if it fails).
            try:
                _rcg = _ro.make_cl(bb, cc, D.r, D.mc)
                _pairs_g = _rcg.near_line_pairs(aa, nn, KK, a.eps, reach=1.0,
                                                res_tol=a.sqp_row_res_tol, mc=D.mc)
                if _pairs_g:
                    _jg = int(np.argmax([p_['lam'].real for p_ in _pairs_g]))
                    al_c = float(_pairs_g[_jg]['lam'].real); _lam_c = complex(_pairs_g[_jg]['lam'])
                    _clr_c = True; _gate_reduced = True
            except Exception as _eg:
                print(f"    provisional gate: reduced abscissa failed ({type(_eg).__name__}); full gate", flush=True)
        if not _gate_reduced:
            al_c, _lam_c, _, _, _, _clr_c = D.alpha_certified(aa, nn, KK, bb, cc,
                                                              eps=a.eps,
                                                              eigs_seen=_seen_c)
        if _ro is not None and _lam_c is not None and np.isfinite(complex(_lam_c)):
            _al0_lam = complex(_lam_c)     # the gate's rightmost eigenvalue: next round's eigen-anchor
        # CLOSED-LOOP PAIR AT THE LINE (2026-09-26, detector only): two distinct
        # closed-loop eigenvalues within 1e-3 of the line and 1e-2 of each other
        # -- the closed-loop analogue of the compensator's double pole, which has
        # no face-S treatment (per-eigenvalue rows + the 2x2 Hermite block).
        # Never hit on GL; counted so that the cavity runs show it if it happens.
        _nl = []
        for _x in (_seen_c or []):
            _x = complex(_x)
            if abs(_x.real + a.eps) <= 1e-3 and all(abs(_x - _y) > 1e-9 for _y in _nl):
                _nl.append(_x)
        if any(abs(_x - _y) < 1e-2 for _i, _x in enumerate(_nl) for _y in _nl[_i + 1:]):
            n_clpair[0] += 1
            print(f"    NOTE: closed-loop pair at the line: {[f'{_x:.5f}' for _x in _nl]}", flush=True)
        if best.get('z') is not None and (best.get('clr') != bool(_clr_c)
                                          or abs(float(best.get('al') or 0.0) - al_c) > 1e-6):
            # the inner solve and the gate certified the SAME design and got
            # different answers -- this must not happen silently
            print(f"    CERTIFICATE DISAGREEMENT at the returned iterate: inner "
                  f"alpha={best.get('al')} clear={best.get('clr')} vs gate "
                  f"alpha={al_c:+.4f} clear={bool(_clr_c)}", flush=True)
        # THE CERTIFICATE IS THE AUTHORITY, alpha_krylov is only an aid.
        #
        # This was wired the other way round and it was the wrong way: the gate
        # consulted the certificate ONLY when alpha_krylov had already declared
        # the point feasible, so a certificate could confirm a "yes" but never
        # overturn a false "no".  alpha_krylov is a heuristic shift set with no
        # residual verification, and it was measured returning +1.3529 for a
        # design whose true alpha is -0.4862 -- rejecting a perfectly good warm
        # start and, through the phase chain, discarding it.
        #
        # So: if the rectangle certificate proves the region
        # Re in [-eps, s_far] free of eigenvalues, the design IS feasible,
        # whatever alpha_krylov says.  alpha_krylov is consulted only when the
        # certificate is unavailable.
        feasible = None
        if a.rect_cert and not _gate_reduced:
            try:
                # the part of the line the (memoised) abscissa certificate of
                # this design already proved clear is not swept again
                # (reduced-oracle loop only; 2026-09-26, STAGE5 30)
                _skip = None
                if _ro is not None and bool(_clr_c) and int(getattr(D, 'cert_nseg', 1)) <= 1:
                    _skip = csp.stability_region(D.A, D.M, ctl.A_of(*aa))[:2]
                okr, why, nsr = csp.stability_gate(
                    D.A, D.M, bb, cc, ctl.A_of(*aa),
                    ctl.Bc_of(aa, D.r, D.mc), nn, KK, eps=a.eps,
                    shrink=a.cert_shrink, wlo=a.wlo, whi=a.whi,
                    ctrl_max=a.ctrl_max, skip=_skip)
                if not okr and why.startswith('INCONCLUSIVE') and _wind_accept:
                    # MIMO (STAGE6 6.5): with the loops at high gain the band follows a
                    # closed-loop branch out to |Im| ~ 850 (the band grew to +-1269 at
                    # ne=100 atoms) and the line sweep runs out of shifts on it (85
                    # uncovered); the exact winding count, global in w, decides.
                    _o = _wind_right_of_margin(bb, cc, aa, nn, KK, _seen_c)
                    if _o is not None:
                        _wind_done = _o
                        if _o['ok'] and _o['count'] == 0:
                            print(f"    gate: line sweep {why}; winding count right of Re=-eps is 0 "
                                  f"({_o['npts']} pts): clear", flush=True)
                            okr = True; n_gate_wind[0] += 1
                # BOTH must hold.  The certificate can only CONFIRM
                # feasibility, never overrule a positive abscissa: its region is
                # bounded and alpha is now trustworthy.
                feasible = bool(okr) and (al_c + a.eps <= 0.0)
                if okr and al_c + a.eps > 0.0:
                    print(f"    REJECTED: alpha={al_c:+.4f} > -eps, although "
                          f"the bounded region looks clear ({why})", flush=True)
                elif not okr:
                    if why.startswith('INCONCLUSIVE'):
                        n_inconcl[0] += 1
                    else:
                        n_located[0] += 1
                    print(f"    gate REJECTS: {why}"
                          f"  (certificate alpha {al_c:+.4f})", flush=True)
            except Exception as e:
                print(f"    gate unavailable ({type(e).__name__}); "
                      f"falling back to alpha_krylov", flush=True)
        if feasible is None:
            feasible = (al_c + a.eps <= 0.0)
        # POLE CAP, enforced at acceptance and not only inside the inner solve.
        # The inner |Im lambda(A_c)| <= wmax row is a soft constraint the
        # solver can end in violation of: one sweep saved KSP+B+C with a
        # controller pole at |Im| = 138 against a cap of 20, i.e. a resonance
        # far outside every band any downstream tool certifies.  Exact and
        # essentially free (r x r).
        if feasible and a.ctrl_imcap == 'on' and flags['K'] and \
                flags.get('ctrl_mode', 'full') in ('full', 'sp'):
            _polc = np.linalg.eigvals(ctl.A_of(*aa))
            _imc = float(np.max(np.abs(_polc.imag))) if _polc.size else 0.0
            if _imc > wmax * (1.0 + 1e-6):
                feasible = False
                n_located[0] += 1
                print(f"    gate REJECTS: controller pole |Im|={_imc:.2f} "
                      f"beyond the cap {wmax:.2f} (soft inner constraint "
                      f"ended violated)", flush=True)
        # BAND FROM THE CLOSED LOOP (--ctrl-imcap off).  The |Im| cap on the
        # compensator was never a design requirement; it kept the compensator's
        # poles inside the band the exchange samples and the peak certificate
        # scans, because a peak outside that band is invisible to both.  But
        # controller poles are only a proxy for closed-loop poles -- the loop
        # moves them -- so with the cap off the band is set from the thing
        # that actually makes peaks: closed-loop eigenvalues near the axis.
        # Dense at GL (n+r ~ 400, milliseconds); the analytic tail bound that
        # makes this scale-free belongs to stage 5.  The band only grows.
        if feasible and a.ctrl_imcap == 'off':
            try:
                _hw = band_from_closedloop(aa, nn, KK, bb, cc, extra=_seen_c or [])
                if _hw > a.whi * (1.0 + 1e-9):
                    print(f"    band widened to [{-_hw:.2f}, {_hw:.2f}] from "
                          f"the closed-loop spectrum", flush=True)
                    a.wlo, a.whi = -_hw, _hw
            except Exception as _e:
                print(f"    band_from_closedloop unavailable "
                      f"({type(_e).__name__}); band kept", flush=True)
        # CONTINUATION GATE: the candidate is reachable from the last accepted
        # iterate without any eigenvalue crossing the stability line.  This is
        # the matrix-free replacement for the small-steps assumption; see the
        # block comment at continuation_gate.
        #
        # Run it ONLY when the candidate would otherwise be ACCEPTED
        # (line-feasible AND improving).  A candidate rejected on gamma never
        # becomes the new anchor, so certifying its segment buys nothing --
        # and on reject-heavy cases the gate's homotopy sweeps were a large
        # share of the wall clock (KGC+C paid for 28 rejections).  The
        # tracked queries themselves run on the |Omega| worker pool.
        watch_cand = None
        _nsp_seen = 0
        if feasible and a.cont and _anchor_ok and g < gbest * (1 - 1e-12) and not _gate_reduced:
            if a.cont_track == 'reduced' and _ro is not None and _ro.dim > 0:
                # algebra note item 11 (STAGE5 33): eigenvalues tracked on the
                # reduced pencil, full-model fallbacks counted
                _okc, watch_cand, _nq, _nsp, _nfb = continuation_gate_reduced(
                    D, pk, zwatch, zc, watch_cur, a.eps, _ro,
                    depth=a.cont_depth, motion_frac=a.cont_motion,
                    motion_floor=a.cont_floor, kwatch=a.cont_k, res_tol=a.sqp_row_res_tol)
                n_cont_fallback[0] += int(_nfb)
            else:
                _okc, watch_cand, _nq, _nsp = continuation_gate(
                    D, pk, zwatch, zc, watch_cur, a.eps,
                    depth=a.cont_depth, motion_frac=a.cont_motion,
                    motion_floor=a.cont_floor, kwatch=a.cont_k, pool=_pool)
            _nsp_seen = _nsp
            _ndz_used = float(np.linalg.norm(zc - zwatch))
            if not _okc and a.cont_accept_winding:
                # STAGE 2: a random start's chain is not a lattice chain.  When
                # the homotopy from the accepted design crosses the margin but
                # the NEW design passes the unconditional winding certificate
                # (the standalone guarantee, stage 1), accept it and restart the
                # chain there.  (scaled_002: a design certified at 7.237 was
                # refused on the path from its 95.0 start.)
                try:
                    # count right of the MARGIN itself, Re = -eps (2026-09-25; was
                    # -eps/2, which left the gap (-eps, -eps/2) the Stage-0 report
                    # flagged)
                    _o = _wind_right_of_margin(bb, cc, aa, nn, KK, _seen_c)
                    if _o is not None:
                        _wind_done = _o
                        if not (_o['ok'] and _o['count'] == 0):
                            print(f"    continuation refused; winding count right of Re=-eps: "
                                  f"{'INCONCLUSIVE' if not _o['ok'] else _o['count']} ({_o['npts']} pts): not accepted",
                                  flush=True)
                        if _o['ok'] and _o['count'] == 0:
                            print("    continuation refused, but the winding count right of Re=-eps is 0 "
                                  f"({_o['npts']} pts): UNCONDITIONALLY stable at the margin, accepted, chain restarted",
                                  flush=True)
                            _okc = True; watch_cand = _topk_watch(list(_seen_c or []), a.cont_k, a.eps)
                except Exception as _e:
                    print(f"    winding check failed: {type(_e).__name__}", flush=True)
            if _okc and _wind_accept and _wind_done is None:
                # STAGE6 (2026-09-29): the continuation gate watches the near-line
                # eigenvalues of the certified BAND; a high loop gain can bring an
                # eigenvalue from far left to Re > 0 far OUTSIDE the band in one
                # step (MIMO atoms, ne=100: +17.1 - 111.8i, accepted, caught only by
                # the dense save veto).  The winding count is global in w: every
                # accepted candidate must have none right of Re = -eps.
                try:
                    _o = _wind_right_of_margin(bb, cc, aa, nn, KK, _seen_c)
                    n_wind_accept[0] += 1
                    if not (_o is not None and _o['ok'] and _o['count'] == 0):
                        _okc = False
                        print(f"    winding check at acceptance: "
                              + ('unavailable (no plant pole data)' if _o is None else
                                 f"{'INCONCLUSIVE' if not _o['ok'] else str(_o['count']) + ' eigenvalue(s) right of Re=-eps'} "
                                 f"({_o['npts']} pts)") + ": REFUSED", flush=True)
                        n_wind_refused[0] += 1
                except Exception as _e:
                    _okc = False
                    print(f"    winding check at acceptance failed ({type(_e).__name__}): refused", flush=True)
            if not _okc:
                feasible = False
                n_located[0] += 1
                print(f"    ({_nq} tracked queries, {_nsp} splits)",
                      flush=True)
                if a.tau:
                    _tau_rho = min(_tau_rho or np.inf, 0.25 * _ndz_used)
                    _tau_streak = 0
                    print(f"    tau governor: cap -> {_tau_rho:.3g} "
                          f"(gate rejection)", flush=True)
            elif a.tau and _nsp > 0 and not (a.inner == 'sqp' and a.tau_split == 'ignore'):
                _tau_rho = min(_tau_rho or np.inf, 0.5 * _ndz_used)
                _tau_streak = 0
                print(f"    tau governor: cap -> {_tau_rho:.3g} "
                      f"({_nsp} homotopy splits)", flush=True)
            elif a.tau and _nsp > 0:
                # a SPLIT is the gate succeeding (it subdivided the path and
                # certified it): no cap (2026-09-25).  ne=100 K+B+C: one split at
                # round 8 capped the displacement at 0.0021, relaxed x2 every two
                # clean rounds; rounds 10-16 all ended on that cap (code 6) and
                # the loop crept 7.9369 -> 7.9145 in eight rounds
                print(f"    tau governor: {_nsp} homotopy split(s), gate passed -- no cap", flush=True)
        if not ok and np.isfinite(g):
            # INCONCLUSIVE certificate (2026-09-24): g is only a lower bound (the
            # decision procedure found a higher level set it could not locate);
            # never accept it as certified.  The witness is enrolled below
            # (newf) like any certificate witness.
            print(f"    certificate INCONCLUSIVE at {g:.6f} (a higher level set exists, not located; "
                  f"witness {w:+.4f}): round not accepted", flush=True)
        improved = feasible and bool(ok) and g < gbest * (1 - 1e-12)
        # PROGRESS for the fixed-point test (2026-09-25): an improvement below
        # the certificate's own tolerance is not progress.  The ne=100 K+B+C
        # continuation repeated 7.657552 for five rounds (gains ~1e-9) and
        # --converge never fired because each counted as an improvement.
        # Acceptance of the design is unchanged (strict).
        _progress = improved and g < gbest * (1 - max(1e-12, float(a.cert_rtol)))
        if improved:
            _al_accepted = float(al_c)      # the design the next round starts from
            _peaks_of_best = [float(w)] + [float(x) for x in peaks]
        if not _prov and _ro is not None and a.cert_every > 1:
            # FULL round.  First settle the provisional chain: fully certify
            # the current provisional best (if it was never certified), keep
            # it if it beats the last certified best, else roll back.
            if zbest_cert is None or not np.array_equal(zbest, zbest_cert):
                certified_norm.provisional = False
                g_pb, w_pb, _, pk_pb = certified_norm(D, pk, zbest, a.wlo, a.whi, gam_lb=1e-8)
                if a.prov_gate == 'reduced':
                    # the provisional chain's STABILITY, on the full model: the
                    # abscissa certificate + the rectangle certificate; sigma is
                    # meaningless for an unstable loop, so failure = rollback
                    _aa_p, _nn_p, _KK_p, _xa_p, _xs_p, _va_p, _vs_p = pk.unpack(zbest)
                    _bb_p, _cc_p, *_ = D.bc(_xa_p, _xs_p, _va_p, _vs_p)
                    _seen_p = []
                    _al_p, _, _, _, _, _clr_p = D.alpha_certified(_aa_p, _nn_p, _KK_p, _bb_p, _cc_p,
                                                                  eps=a.eps, eigs_seen=_seen_p)
                    _okr_p = True
                    if a.rect_cert:
                        try:
                            _okr_p, _, _ = csp.stability_gate(D.A, D.M, _bb_p, _cc_p, ctl.A_of(*_aa_p),
                                                              ctl.Bc_of(_aa_p, D.r, D.mc), _nn_p, _KK_p, eps=a.eps,
                                                              shrink=a.cert_shrink, wlo=a.wlo, whi=a.whi,
                                                              ctrl_max=a.ctrl_max)
                        except Exception:
                            _okr_p = bool(_clr_p)
                    if not (_clr_p and _okr_p and _al_p + a.eps <= 0.0) and zbest_cert is not None:
                        print(f"    provisional best FAILS the full stability check (alpha {_al_p:+.5f}, "
                              f"line clear {bool(_clr_p)}, rectangle {bool(_okr_p)}): rolling back", flush=True)
                        gbest, zbest = gbest_cert, zbest_cert.copy()
                        g_pb = np.inf
                    else:
                        print(f"    provisional best: stability confirmed on the full model (alpha {_al_p:+.5f})", flush=True)
                        watch_cur = _topk_watch(_seen_p, a.cont_k, a.eps); zwatch = zbest.copy(); _anchor_ok = True
                if g_pb <= gbest_cert * (1.0 + a.rtol) or zbest_cert is None:
                    gbest_cert, zbest_cert, gbest = float(g_pb), zbest.copy(), float(g_pb)
                    print(f"    provisional best fully certified: {g_pb:.6f}", flush=True)
                elif a.cert_fail == 'continue':
                    # user's scheme (2026-09-23): keep the design with its TRUE
                    # value; what the reduced chain missed becomes Omega members,
                    # peak members and anchors, and the run continues (only the
                    # final design needs the guarantee)
                    print(f"    provisional best is {g_pb:.6f} in truth (claimed {gbest:.6f}): continuing "
                          f"from it, witness {w_pb:+.4f} enrolled", flush=True)
                    gbest, gbest_cert, zbest_cert = float(g_pb), float(g_pb), zbest.copy()
                    _miss = [round(float(x), 6) for x in ([w_pb] + list(pk_pb))]
                    Omega = sorted(set(list(Omega) + _miss)); _peakset.update(_miss)
                    if _ro is not None:
                        _enrich(_miss, zbest, "full check disagreed")
                else:
                    print(f"    provisional best is {g_pb:.6f} in truth > last certified {gbest_cert:.6f}: "
                          f"rolling back the provisional chain", flush=True)
                    gbest, zbest = gbest_cert, zbest_cert.copy()
            if improved and feasible:
                gbest_cert, zbest_cert = float(g), zc.copy()
        if improved:
            gbest, zbest, albest = g, zc.copy(), float(al_c)
            # CHECKPOINT: the best accepted design survives any interruption
            # (SLURM kill, node loss).  Verified state only -- this iterate
            # just passed the line certificate, the pole cap and the
            # continuation gate -- and the startup anchor re-verifies it
            # (dense at GL) before any resumed run trusts it.
            try:
                _apa, _aps = pk.atom_params(zbest)
                np.savez(tag + '_ckpt.npz', d=aa[0], e=aa[1], f=aa[2],
                         nvec=nn, K=KK, xa=xa_, xs=xs_,
                         **(dict(Bc=np.asarray(aa[3]), loop_orders=np.array(flags['loop_orders'])) if MIMO else {}),
                         atoms_a_p=(_apa if _apa is not None else np.zeros(0)),
                         atoms_s_p=(_aps if _aps is not None else np.zeros(0)),
                         va=(va_ if va_ is not None else np.zeros(0)),
                         vs=(vs_ if vs_ is not None else np.zeros(0)),
                         gamma=g, alpha=float(al_c), wlo=float(a.wlo), whi=float(a.whi),
                         maxit_cur=int(_maxit_cur), sqp_trust=float(_sqp_trust_prev or 0.0),
                         peakset=np.array(sorted(_peakset), float),
                         active_freqs=np.array(sorted(set(list(Omega) + [round(float(x), 6) for x in ([w] + peaks)]))))   # the working set: a resume that lacks it re-discovers the band
                json.dump(dict(case=a.case, gamma=float(g),
                               alpha=float(al_c),
                               note=('checkpoint (PROVISIONAL: reduced-model certificate; a resume re-certifies it in full)'
                                     if getattr(certified_norm, 'last_provisional', False) else 'checkpoint')),
                          open(tag + '_ckpt.json', 'w'))
            except Exception as _e:
                print(f"    checkpoint write failed: {_e!r}", flush=True)
            if a.tau and _tau_rho is not None:
                if _nsp_seen == 0 or (a.inner == 'sqp' and a.tau_split == 'ignore'):
                    _tau_streak += 1
                    # the SQP arm relaxes after EVERY clean acceptance (the frozen
                    # arm after two): a cap is evidence about one step, not a regime
                    if _tau_streak >= (1 if a.inner == 'sqp' else 2):
                        _tau_rho *= 2.0
                        _tau_streak = 0
                        if _tau_rho > 1e6:
                            _tau_rho = None     # effectively off again
                        print(f"    tau governor: cap relaxed -> "
                              f"{_tau_rho if _tau_rho else 'off'}",
                              flush=True)
                else:
                    _tau_streak = 0
            if a.cont and not _anchor_ok:
                # deferred anchor: this is the first FEASIBLE accepted design
                if not _gate_reduced:   # a reduced gate has no certified eigenvalue list: keep the last certified anchor
                    watch_cur = _topk_watch(_seen_c, a.cont_k, a.eps)
                    zwatch = zc.copy()
                    _anchor_ok = True
                print(f"    continuation anchor SET at outer {it}: alpha="
                      f"{al_c:+.4f}, watching {len(watch_cur)} eigenvalue(s); "
                      f"the chain is enforced from here", flush=True)
            elif a.cont:
                # New anchor: the watch list merges the eigenvalues TRACKED
                # onto zc with those the gate's own line sweep just saw there.
                _merged = [w for w in (watch_cand or []) if w is not None]
                if not _gate_reduced:   # a reduced gate has no certified eigenvalue list: keep the last certified anchor
                    watch_cur = _topk_watch(_merged + _seen_c, a.cont_k, a.eps)
                    zwatch = zc.copy()

        # adapt the trust radius: a faithful step earns a bigger one, a step
        # whose certified norm blew past t earns a smaller one
        if a.trust > 0:
            if np.isfinite(g) and g <= best['t'] * 1.5:
                trust = min(trust * 1.6, 4.0)
            else:
                trust = max(trust * 0.4, 0.02)
        newf = [x for x in ([w] + peaks)
                if all(abs(x - q) > 1e-3 for q in Omega)]
        if np.isfinite(g) and np.isfinite(best['t']) and g > best['t'] * (1.0 + 10 * a.rtol):
            # REFUSAL DIAGNOSIS (2026-09-22): where the certificate's witness sits
            # relative to Omega, and whether its nearest member was a followed peak
            _Oa = np.asarray(Omega, float); _jn = int(np.argmin(np.abs(_Oa - w))); _wn = float(_Oa[_jn])
            print(f"    refusal: witness w={w:+.4f} (certified {g:.6f} vs t {best['t']:.6f}); nearest member "
                  f"{_wn:+.4f} (distance {abs(_wn - w):.4f}), "
                  f"{'FOLLOWED peak' if round(_wn, 6) in _peakset else 'NOT a peak member'}", flush=True)
        _last_peaks = [float(w)] + [float(x) for x in peaks]
        if _ro is not None:
            # ABMV enrichment rule: the reduced optimum's TRUE maximiser, at
            # the design the inner solve returned (refuted or not) -- that is
            # where the reduced model was wrong.  Also the 1e-5 gate: reduced
            # value at the witness against the certified one.
            _aa_, _nn_, _KK_, _xa_, _xs_, _va_, _vs_ = pk.unpack(zc)
            _bb_, _cc_, *_ = D.bc(_xa_, _xs_, _va_, _vs_)
            _rc = _ro.make_cl(_bb_, _cc_, D.r, D.mc)
            _rw = _rc.gain(float(w), _aa_, _nn_, _KK_, want_vectors=False)['sigma']
            _err = abs(_rw - g) / max(g, 1e-12)
            _oracle_log[-1].update(witness=float(w), reduced_at_witness=float(_rw),
                                   certified=float(g), rel_err=float(_err))
            print(f"    oracle: reduced at the witness {_rw:.6f} vs certified {g:.6f} "
                  f"(rel {_err:.1e})", flush=True)
            _dbg_err = os.environ.get('CODESIGN_DEBUG_ROM_ERR')
            if ((os.environ.get('CODESIGN_DEBUG_ROM') and _err < 1e-6 and np.isfinite(best['t'])
                    and g > best['t'] * (1.0 + 10 * a.rtol))
                    or (_dbg_err is not None and _err > float(_dbg_err))):
                # DEBUG (2026-09-25): a refusal whose witness the model knows
                # exactly -- save the model and the design for an offline
                # test of the reduced level-set certificate
                # the model's numerical state only (V, W): it holds cached
                # sparse LU factors that cannot be pickled -- rebuild with
                # ReducedOracle(D); ro.V, ro.W = V, W; ro._project()
                _fn = os.path.join(a.outdir, f"debug_rom_outer{it:03d}.npz")
                np.savez(_fn, V=_ro.V, W=_ro.W, galerkin=bool(getattr(_ro, 'galerkin', False)),
                         d=_aa_[0], e=_aa_[1], f=_aa_[2], nvec=_nn_, K=_KK_, xa=_xa_, xs=_xs_,
                         va=(_va_ if _va_ is not None else np.zeros(0)), vs=(_vs_ if _vs_ is not None else np.zeros(0)),
                         t=float(best['t']), witness=float(w), g=float(g), wlo=a.wlo, whi=a.whi,
                         anchors=np.asarray(_ro.anchors, float))
                print(f"    DEBUG: reduced model + design saved -> {_fn}", flush=True)
            if _err > a.oracle_gate:
                _enrich([w], zc, f"ABMV: witness of the returned design, err {_err:.1e}")
            _oracle_log[-1].update(sweep_max_rel=_sweep_anchor(zc))
        hist.append(dict(outer=it, inner_calls=ncalls, trust=float(trust), gamma_certified=float(g),
                         gamma_best=float(gbest), t=float(best['t']),
                         nOmega=len(Omega), added=len(newf),
                         # stage-0 instrumentation, cheap enough to be always
                         # on: these come from the pyGRANSO solution object,
                         # which was previously discarded.  inner_iters is NOT
                         # inner_calls -- the first counts optimiser steps, the
                         # second distinct design evaluations, and the ratio is
                         # what stage 1 moves.
                         inner_iters=inner_stats.get('iters'),
                         inner_fn_evals=inner_stats.get('fn_evals'),
                         termination_code=inner_stats.get('termination_code'),
                         stat_value=inner_stats.get('stat_value'),
                         qp_failure_rate=inner_stats.get(
                             'quadprog_failure_rate'),
                         # True = this round's inner solve ABORTED on a QP
                         # factorization failure (code 12), so its call count
                         # is a floor, not the work it needed.
                         qp_failure=inner_stats.get('qp_failure'),
                         qp_log=inner_stats.get('qp_log'),
                         sigma2_rows=sorted(_sigma2_prev) if a.sigma2 == 'on' else None,
                         clock_guard_bound=inner_stats.get(
                             'clock_guard_bound'),
                         lu=((lambda s: dict(
                             n_lu=s['n_lu'] - _lu_prev['n_lu'],
                             n_lu_full=s.get('n_lu_full', 0) - _lu_prev.get('n_lu_full', 0),
                             n_solve=s['n_solve'] - _lu_prev['n_solve']))(
                                 _luc.snapshot())
                             if (_luc is not None and _lu_prev is not None)
                             else None)))
        if a.dump:
            try:
                _lu = (_luc.snapshot() if _luc is not None else None)
                if _lu is not None and _lu_prev is not None:
                    _lu = dict(n_lu=_lu['n_lu'] - _lu_prev['n_lu'],
                               n_solve=_lu['n_solve'] - _lu_prev['n_solve'],
                               n_lu_cum=_lu['n_lu'],
                               n_lu_full_cum=_lu.get('n_lu_full'),
                               by_site_full_cum=_lu.get('by_site_full'),
                               n_solve_cum=_lu['n_solve'],
                               per_inner_iter=(
                                   (_lu['n_lu'] - _lu_prev['n_lu'])
                                   / max(int(inner_stats.get('iters') or 0), 1)),
                               per_oracle_call=(
                                   (_lu['n_lu'] - _lu_prev['n_lu'])
                                   / max(ncalls, 1)))
                dump_outer(a.dump, it, D, pk, flags, zc, Omega,
                           best['t'], g, gbest, a.eps, par=_par,
                           inner_stats=inner_stats, lu=_lu,
                           gate=dict(feasible=bool(feasible),
                                     improved=bool(improved),
                                     located=int(n_located[0]),
                                     inconclusive=int(n_inconcl[0]),
                                     alpha=float(al_c)),
                           extra=dict(inner_calls=int(ncalls),
                                      added=int(len(newf)),
                                      wall_s=round(time.time() - t_start, 1)))
                if _luc is not None:
                    _lu_prev = _luc.snapshot()
            except Exception as _e:
                print(f"    dump failed: {type(_e).__name__}: {_e}", flush=True)
        print(f"  outer {it}: |Omega|={len(Omega)}  t={best['t']:.6f}  "
              f"certified={g:.6f}  best={gbest:.6f}  "
              f"(+{len(newf)} freq, {ncalls} calls)", flush=True)
        if a.field_basis != 'none' or a.field_atoms:       # STAGE5 32: the fields' shape, every round
            try:
                _fv = pk.unpack(zbest)[5:7]
                _fv = [v_ for grp in _fv if grp is not None for v_ in (grp if isinstance(grp, list) else [grp])]
                _rk = [float(np.sqrt(np.real(np.vdot(v, D.Kstiff @ v)) / np.real(np.vdot(v, D.M @ v))))
                       for v in _fv]
                print("    fields (best): RMS wavenumber " + ", ".join(f"{x:.4f}" for x in _rk)
                      + (f" (bound {a.field_kmax:g}, order {a.field_order})" if a.field_kmax else ''), flush=True)
            except Exception as _e:
                print(f"    fields: shape report failed ({type(_e).__name__}: {_e})", flush=True)
        if a.inner_maxit_adapt:
            # STAGE 3: the inner budget follows the surrogate's validity.  The
            # certification just measured how far the restricted optimum was
            # from the truth: agreement -> a longer inner solve next round;
            # disagreement -> a shorter one (it overfits the gaps).  Trust-
            # region logic on the iteration count instead of the step radius,
            # where the measurement says the loss occurs (6x20: gamma 9.41 but
            # certified/t up to 2.1; 24x5: certified/t <= 1.18 but 9.99).
            _ratio = (g / max(float(best['t']), 1e-12)) if np.isfinite(best['t']) else np.inf   # nothing admissible: shorten
            if _ratio <= 1.0 + a.inner_adapt_tol:
                _maxit_cur = min(int(_maxit_cur * 2), int(a.inner_maxit_max))
            elif _ratio > 1.0 + 3 * a.inner_adapt_tol:
                _maxit_cur = max(int(_maxit_cur // 2), int(a.inner_maxit_min))
            print(f"    inner budget -> {_maxit_cur} (certified/t = {_ratio:.3f})", flush=True)
        # converged when the inner optimum is genuinely the global max AND the
        # inner solve itself was stationary (or under a clock budget, where the
        # frozen rule applied).  Under an iteration budget with in-solve
        # refinement (--sqp-refine) the certificate agrees with t every round
        # while the inner solve is budget-limited, and the frozen rule stopped
        # at round 0 having just made progress.
        _inner_stationary = (int(inner_stats.get('termination_code') or -1) == 0) \
            if a.inner_budget == 'iters' else True
        # convergence only on a FULL round: a provisional value is the reduced
        # model's (--cert-every); the final design must carry the full guarantee
        if abs(g - best['t']) <= a.rtol * max(g, 1e-12) and not newf and _inner_stationary and not _prov:
            print("  exchange converged: t is the certified norm", flush=True)
            exch_converged = True
            break
        Omega = sorted(set(list(Omega) + [round(x, 6) for x in newf]))
        _peakset.update(round(float(x), 6) for x in newf)
        for x in newf:
            _born.setdefault(round(x, 6), it + 1)
            _refuter[round(x, 6)] = it + 1          # added by REFUTATION: protected until the design moves off it
        if a.omega_policy == 'an':
            # ---- item 4, seeding: log-spaced satellites around each NEW peak,
            # w_p +- delta0 rho^k, kept where sigma_1 >= (1 - eps_w) gamma;
            # the peak is then bracketed at every scale, so drift within an
            # inner step has somewhere to land.  Re-seeded only for peaks that
            # are currently active (within the certification tolerance of gamma).
            _aa_, _nn_, _KK_, _xa_, _xs_, _va_, _vs_ = pk.unpack(zbest)
            _bb_, _cc_, *_ = D.bc(_xa_, _xs_, _va_, _vs_)
            # reduced model when attached (2026-09-22): the full closed loop cost
            # |Omega| + satellites factorisations a round
            _clw = _ro.make_cl(_bb_, _cc_, D.r, D.mc) if (_ro is not None and _ro.dim > 0) else D.cl(_bb_, _cc_)
            _s1w = lambda w: _clw.gain(float(w), _aa_, _nn_, _KK_, want_vectors=False)['sigma']
            _level = (1.0 - a.omega_epsw) * gbest
            _seeded = []
            # seed around the certified peaks of the accepted design that are
            # currently within the level (all of them, new or not: the note
            # says "re-seeds only the currently active ones", and a first
            # version that seeded new peaks only left |Omega| = 19 after the
            # grid retired and the inner solve collapsed to t = 1.5)
            # the accepted design's peaks were computed when it was certified; a
            # second full certification per round (~150 factorisations) is not needed
            _pk_now = list(_peaks_of_best)
            _seed_from = sorted(set([round(float(x), 6) for x in newf]
                                    + [round(float(x), 6) for x in _pk_now]))
            for x in _seed_from:
                if _s1w(x) < _level:
                    continue
                for k in range(a.omega_nsat):
                    dlt = a.omega_delta0 * (a.omega_rho ** k)
                    for w in (x - dlt, x + dlt):
                        if not (a.wlo <= w <= a.whi):
                            continue
                        w = round(float(w), 6)
                        if any(abs(w - q) < 1e-4 for q in Omega):
                            continue
                        if _s1w(w) >= _level:
                            _seeded.append(w)
            if _seeded:
                Omega = sorted(set(list(Omega) + _seeded))
                for w in _seeded:
                    _born.setdefault(w, it + 1)
            # ---- item 5, retirement: sigma_1 on Omega at the accepted design;
            # members below the level for two consecutive rounds retire; the
            # starter grid retires after the first certified round; the young
            # (added this round or last) and the active never retire; cap the
            # rest at |Omega| <= omega_cap.  Certification makes this safe: a
            # wrongly retired frequency is re-located and re-added.
            _sv = {w: _s1w(w) for w in Omega}
            _active = {w for w in Omega if _sv[w] >= gbest * (1.0 - a.rtol * 10)}
            for w in Omega:
                _dormant[w] = _dormant.get(w, 0) + 1 if _sv[w] < _level else 0
            # a refuting frequency is dormant at the ACCEPTED design by
            # construction (it refuted a different one); retiring it re-derives
            # the refuted design next round and the loop cycles (measured:
            # rounds 3-5 identical, +1 freq each).  Protect it for
            # omega_refuter_rounds rounds after it refuted anything.
            _prot = {w for w, r0 in _refuter.items() if (it + 1) - r0 < a.omega_refuter_rounds}
            _retire = set()
            # delayed cap (user, 2026-09-18): keep the uniform grid while the
            # design is moving a lot -- the early rounds are where the
            # surrogate overfits most -- and only retire/cap from round
            # omega_cap_after on, when the peaks are where the satellites are
            _cap_on = it >= a.omega_cap_after
            # the starter grid retires by DORMANCY, one round instead of two,
            # and only once a design has been accepted: retiring it on a round
            # count (first version) removed 64 members while 9 satellites
            # existed, and the restricted problem lost its structure
            _accepted_once = gbest < g0 * (1 - 1e-12)
            _retire |= {w for w in Omega if _cap_on and w in _starter and w not in _skeleton and _accepted_once
                        and _dormant.get(w, 0) >= 1 and w not in _active and w not in _prot
                        and round(float(w), 6) not in _peakset}
            _retire |= {w for w in Omega if _cap_on and _dormant.get(w, 0) >= 2 and w not in _active
                        and w not in _skeleton and w not in _prot and (it + 1) - _born.get(w, 0) >= 2
                        and round(float(w), 6) not in _peakset}
            # never shrink below a floor: the surrogate needs a skeleton
            if len(Omega) - len(_retire) < a.omega_floor:
                _keep_back = sorted((w for w in _retire), key=lambda w: -_sv[w])
                for w in _keep_back[:a.omega_floor - (len(Omega) - len(_retire))]:
                    _retire.discard(w)
            _kept = [w for w in Omega if w not in _retire]
            if _cap_on and len(_kept) > a.omega_cap:
                # HARD cap: drop the lowest-sigma_1 members that are neither
                # active nor this round's new peaks -- age is no protection.
                # (Exempting the young left one round on 111 members, 5,244
                # gains against ~2,500, because the seeding fired before the
                # grid was eligible to retire.)
                _newset = set(round(float(x), 6) for x in newf)
                _order = sorted((w for w in _kept if w not in _active and w not in _newset and w not in _skeleton and w not in _prot
                                 and round(float(w), 6) not in _peakset),   # the cap must not drop followed peaks either
                                key=lambda w: _sv[w])
                _drop = set(_order[:max(0, len(_kept) - a.omega_cap)])
                _retire |= _drop; _kept = [w for w in _kept if w not in _drop]
            _omega_log.append(dict(round=it, before=len(Omega) - len(_seeded), seeded=len(_seeded),
                                   retired=len(_retire), after=len(_kept), active=len(_active)))
            print(f"    omega policy: +{len(_seeded)} satellites, -{len(_retire)} retired "
                  f"-> |Omega|={len(_kept)} ({len(_active)} active)", flush=True)
            Omega = sorted(_kept)
        if not newf and not (_progress if a.converge else improved):
            # nothing new to add and no progress: enlarge Omega with the peaks
            # of the CURRENT best design instead, then stop if still nothing
            _, _, _, pk2 = certified_norm(D, pk, zbest, a.wlo, a.whi,
                                          gam_lb=gbest)
            extra = [x for x in pk2 if all(abs(x - q) > 1e-3 for q in Omega)]
            if not extra:
                nostall += 1
                if nostall >= 3 and a.converge and not _prov:
                    print("  FIXED POINT under --converge: certificate agrees with t, no new "
                          "frequency, no admissible improvement for 3 rounds, inner solve NOT "
                          "stationary by its own measure -- stalled, not converged "
                          "(measure Clarke stationarity of the endpoint)", flush=True)
                    exch_fixed_point = True
                    break
                if nostall >= 3 and not a.converge:
                    print("  exchange stalled: no new frequency and no "
                          "progress for 3 outers", flush=True)
                    exch_stalled = True
                    break
                print(f"  no new frequency ({nostall}/3) -- continuing with "
                      f"the current Omega", flush=True)
            else:
                nostall = 0
                Omega = sorted(set(list(Omega)
                                   + [round(x, 6) for x in extra]))
        else:
            nostall = 0
        # PROGRESS STOP (2026-09-25, user: an acceptable result in less time):
        # stop when the certified value gained less than tol (relative) over the
        # last k rounds.  Measured on the ne=200 K+B+C run with the quasi-Newton
        # fix: k = 3, tol = 0.3 % stops at round 34 (56 min), 0.15 % above where
        # the run was going 21 rounds later.  (Before that fix the same rule
        # stopped on plateaus 20-24 % high: the plateaus were stale curvature.)
        _gh_hist.append(float(gbest))
        if a.stop_progress:
            _k, _tol = int(a.stop_progress[0]), float(a.stop_progress[1])
            _arm_drop = float(a.stop_progress[2]) if len(a.stop_progress) > 2 else 0.01
            _arm_rounds = int(a.stop_progress[3]) if len(a.stop_progress) > 3 else 20
            _armed = (len(_gh_hist) > _arm_rounds
                      or (_gh_hist[0] - _gh_hist[-1]) >= _arm_drop * _gh_hist[0])
            # not on a STALLED solve (code 6 / 12): its slowness is the solver's, and
            # the next round starts over (stall recovery).  ne=100 K+B+C stopped at
            # 7.9319 on two such rounds; the runs that went on reached 6.40.
            _stalled = (a.inner == 'sqp' and isinstance(inner_stats, dict)
                        and inner_stats.get('termination_code') in (6, 12))
            # nor when the last solve was ACCELERATING as its iteration budget ran
            # out (2026-09-26): half or more of its decrease in its last quarter.
            # ne=200 K+B+C from the frozen design: rounds 0-3 crept on the pair's
            # face with a radius ~1e-5 until round 3 took 7.9553 -> 7.9493 in its
            # last iterations (share 0.78) -- and the rule stopped the run there.
            # Tails of the runs that went on: shares 0-0.48.
            _accel = False
            if a.inner == 'sqp' and isinstance(inner_stats, dict) and inner_stats.get('history'):
                _th = [h_['t'] for h_ in inner_stats['history'] if 't' in h_]
                if len(_th) >= 4 and _th[0] > min(_th):
                    _accel = (min(_th[:len(_th) * 3 // 4]) - min(_th)) >= 0.5 * (_th[0] - min(_th))
            # nor when this round's candidate was BETTER by more than tol but refused
            # (gate / admissibility): the landscape offers progress, the governor
            # retries with a smaller step (ne=200 round 4: 7.4964 refused against
            # the best 7.9490, and the rule stopped the run on the three slow rounds
            # before it)
            _refused = (not improved) and bool(np.isfinite(g)) and g < gbest * (1.0 - _tol)
            if (len(_gh_hist) > _k and (_gh_hist[-_k - 1] - _gh_hist[-1]) < _tol * _gh_hist[-1]
                    and not _prov and not _stalled and not _accel and not _refused and _armed):
                print(f"  PROGRESS STOP: {100 * (_gh_hist[-_k - 1] - _gh_hist[-1]) / _gh_hist[-1]:.3f} % over the last "
                      f"{_k} rounds (< {100 * _tol:.2f} %)", flush=True)
                exch_progress_stop = True
                break

    # ---- final evaluation and save (same format as run_case.py) ----
    arow, nvec, K, xa, xs, va, vs = pk.unpack(zbest)
    b, c, pa, ps, na, ns = D.bc(xa, xs, va, vs)
    # the alpha recorded on disk MUST come from the certificate: it is what
    # collect.py reports and what the no-regress guard compares.  Taking it
    # from alpha_krylov is how designs with alpha = +8.55 were stored
    # looking feasible.
    al = D.alpha_certified(arow, nvec, K, b, c, eps=a.eps)[0]
    if al + a.eps > 0.0:
        # same inversion as the gate: check the certificate before refusing
        why_final = ''
        try:
            _okf, why_final, _ = csp.stability_gate(
                D.A, D.M, b, c, ctl.A_of(*arow),
                ctl.Bc_of(arow, D.r, D.mc), nvec, K, eps=a.eps,
                shrink=a.cert_shrink, wlo=a.wlo, whi=a.whi,
                ctrl_max=a.ctrl_max)
        except Exception:
            pass
        print(f"NOT SAVING: final design is unstable (alpha={al:+.4f}; "
              f"{why_final})", flush=True)
        return
    # DENSE SAVE VETO -- check only, never part of the algorithm.  The
    # certificate above is conditional (continuation); a design about to be
    # written to disk deserves the standalone question, which at GL scale a
    # dense eigensolve answers in about a second.  A discrepancy between the
    # two is itself a finding: it means the continuation chain broke somewhere
    # upstream and must be reported loudly, not silently absorbed.  Above
    # --dense_nmax this veto does not exist and the continuation gate carries
    # the guarantee alone -- which is why the gate defaults ON.
    al_d = None
    if a.dense_check and (D.n + D.r) <= a.dense_nmax:
        al_d, lam_d = D.alpha_dense(arow, nvec, K, b, c)
        if abs(al_d - al) > 1e-3 * max(1.0, abs(al_d)):
            print(f"WARNING: certificate alpha {al:+.6f} vs dense "
                  f"{al_d:+.6f} at {lam_d:.6g} -- the line sweep missed the "
                  f"rightmost eigenvalue; the continuation chain is suspect",
                  flush=True)
        if al_d + a.eps > 0.0:
            print(f"NOT SAVING: final design FAILS the dense check "
                  f"(alpha_dense={al_d:+.4f} at {lam_d:.6g}) although the "
                  f"certificate read {al:+.4f}.  This design escaped the "
                  f"matrix-free chain; investigate before re-running.",
                  flush=True)
            return
    elif a.dense_check:
        print(f"dense save veto SKIPPED (n+r={D.n + D.r} > {a.dense_nmax}); "
              f"the saved alpha rests on the continuation chain", flush=True)
    # WINDING CERTIFICATE -- the UNCONDITIONAL test.  Everything above is
    # either conditional (line sweep + continuation) or dense (does not exist
    # at cavity scale).  The argument principle on the scalar return
    # difference f = 1 - k(s)G(s) counts closed-loop eigenvalues right of
    # Re = -eps/2 exactly, matrix-free, with the near-line eigenvalues the
    # sweep already harvested serving as resolution hints for the tracker.
    wind_n = None
    if a.wind:
        try:
            import winding_cert as _wc
            _line = -0.5 * a.eps
            if a.wind_pplant >= 0:
                _pp, _hot = a.wind_pplant, []
            else:
                if not hasattr(D, '_wind_plant'):
                    D._wind_plant = _wc.plant_pole_data(D.A, D.M, _line)   # sparse enclosure (STAGE6 6.13)
                _pp, _hot = D._wind_plant
            _out = _wc.winding_count(
                D.A, D.M, b, c, arow, nvec, K, line=_line, mc=D.mc,
                p_plant=_pp, hot=_hot,
                zeros_hint=[z for z in (watch_cur or [])
                            if z is not None],
                ctrl_exact=(a.wind_ctrl == 'exact'))
            wind_n = int(_out['count']) if _out['ok'] else None
            if _out['ok'] and _out['count'] == 0:
                print(f"winding certificate: 0 eigenvalues right of "
                      f"Re={_line:+.4f} ({_out['npts']} pts) -- "
                      f"UNCONDITIONALLY stable with eps/2 margin",
                      flush=True)
            elif _out['ok']:
                print(f"winding certificate: *** {_out['count']} "
                      f"eigenvalue(s) right of Re={_line:+.4f} ***",
                      flush=True)
                if a.wind_veto:
                    print("NOT SAVING (winding veto).", flush=True)
                    return
            else:
                print("winding certificate INCONCLUSIVE (f ~ 0 on the "
                      "line: an eigenvalue sits at the eps/2 margin)",
                      flush=True)
        except Exception as _e:
            print(f"winding certificate failed non-fatally: {_e!r}",
                  flush=True)
    # FINAL BAND CHECK (STAGE6 6.13, sparse): the whole band tiled, the far
    # candidates refined, the located set compared with the winding count right
    # of -band_recut.  A resonance the loop's (edge-strip) band updates missed
    # widens the band, and the design is certified again over it.
    band_check = {}
    if a.ctrl_imcap == 'off':
        try:
            _hwf = band_from_closedloop(arow, nvec, K, b, c, count=True)
            band_check = {k_: v_ for k_, v_ in _band_last['info'].items() if k_ in ('located', 'count', 'complete')}
            if _hwf > a.whi * (1.0 + 1e-9):
                a.wlo, a.whi = -_hwf, _hwf
                _gw, _ww, _okw, _ = certified_norm(D, pk, zbest, a.wlo, a.whi, gam_lb=gbest)
                print(f"final band check: band widened to [{-_hwf:.2f}, {_hwf:.2f}]; certified {_gw:.6f} "
                      f"(loop {gbest:.6f})", flush=True)
                if np.isfinite(_gw) and _gw > gbest:
                    gbest = float(_gw)
                band_check['widened'] = float(_hwf)
            print(f"final band check: {band_check.get('located')} eigenvalues right of Re={-a.band_recut:g} "
                  f"located, winding count {band_check.get('count')} -- "
                  f"{'COMPLETE' if band_check.get('complete') else 'NOT PROVEN COMPLETE'}", flush=True)
        except Exception as _e:
            print(f"final band check failed non-fatally: {_e!r}", flush=True)
    gam, wst, res, clh, mem, nact = D.hinf(arow, nvec, K, b, c)
    # store the gain curve over the RUN'S OWN certification band, at the same
    # ~0.01 resolution the old [-4, 4] x 801 grid had (capped at 4001 points).
    # A hand-typed narrow grid is how plots ended up unable to show the band
    # the certificates actually cover.
    _core = min(8.0, 0.5 * (a.whi - a.wlo))
    ws = np.unique(np.concatenate([
        np.arange(a.wlo, -_core, 0.05),
        np.arange(-_core, _core, 0.01),
        np.arange(_core, a.whi, 0.05), [a.whi]]))
    # 2026-09-26: on the reduced model when one is attached -- the curve is for
    # plots (the certified value and witness come from the certificate), and on
    # the full model it cost ~2,500 factorisations at the end of every run
    # (hours at the cavity).  Exact to the model's accuracy (1e-13 at its
    # anchors, sweep checks 1e-8..1e-13); tools/dense_validate.py re-checks.
    _gain_model = 'full'
    _clg = clh
    if _ro is not None and _ro.dim > 0:
        try:
            _clg = _ro.make_cl(b, c, D.r, D.mc); _gain_model = 'reduced'
        except Exception:
            _clg = clh
    gain = np.array([_clg.gain(w, arow, nvec, K, want_vectors=False)['sigma']
                     for w in ws])
    gam = max(gam, gbest)
    # DENSE ALL-FREQUENCY CHECK (STAGE6 6.7) -- check only, like the dense save
    # veto on alpha.  The loop certifies gamma over its band; the band follows the
    # closed-loop spectrum, which is a heuristic for where peaks can be.  The
    # dense level set on the full even pencil computes ALL its eigenvalues at each
    # level, so no crossing of the imaginary axis at any frequency is missed.  A
    # higher value means the band missed a peak: reported loudly, and the record
    # stores the true value.
    # SPARSE ALL-FREQUENCY CERTIFICATE (STAGE6 6.19): a tail frequency W beyond
    # which sigma_1 < gamma is PROVED (numerical range of the plant from sparse
    # Lanczos -- the S-lemma heights of sparse_spectrum -- plus the loop's gain
    # bound), then the sparse level set proves nothing exceeds gamma on [-W, W].
    # Together: gamma over ALL w, at any n.
    all_w = {}
    try:
        import sparse_spectrum as _ss
        _Wt, _ti = _ss.closed_loop_tail(D.A, D.M, b, c, ctl.A_of(*arow[:3]), ctl.Bc_of(arow, D.r, D.mc),
                                        nvec, K, gam)
        all_w = dict(tail_W=float(_Wt))
        if np.isfinite(_Wt):
            _sof = lambda w_: clh.gain(float(w_), arow, nvec, K, want_vectors=False)['sigma']
            _ga, _wa, _nc, _oka = csp.hinf_certified_sparse(
                D.A, D.M, b, c, ctl.A_of(*arow[:3]), ctl.Bc_of(arow, D.r, D.mc), nvec, K, _sof, gam,
                wlo=-_Wt, whi=_Wt, rtol=1e-6)
            all_w.update(gamma=float(_ga), certified=bool(_oka), calls=int(_nc))
            if _oka and _ga > gam * (1.0 + 1e-6):
                print(f"WARNING: sparse all-frequency certificate: gamma {_ga:.6f} at w = {_wa:+.4f} above the loop's "
                      f"{gam:.6f}; the record stores it", flush=True)
                gam = float(_ga)
            print(f"sparse all-frequency certificate: tail bound W = {_Wt:.2f}, level set on [-W, W] "
                  f"{'CERTIFIED' if _oka else 'NOT CERTIFIED'} at {_ga:.6f} ({_nc} calls) -- "
                  f"{'gamma holds over ALL w' if _oka else 'all-w claim not proven'}", flush=True)
        else:
            print("sparse all-frequency certificate: no finite tail bound (gain too high for the "
                  "numerical-range bound up to w = 1e9)", flush=True)
    except Exception as _e:
        all_w['error'] = f"{type(_e).__name__}: {str(_e)[:80]}"
        print(f"sparse all-frequency certificate failed non-fatally: {_e!r}", flush=True)
    gam_dense_all = None
    if a.dense_check and (D.n + D.r) <= min(a.dense_nmax, 1000):
        try:
            _sof = lambda w_: clh.gain(float(w_), arow, nvec, K, want_vectors=False)['sigma']
            _gd = csp.hinf_dense_levelset(D.A, D.M, b, c, ctl.A_of(*arow[:3]), ctl.Bc_of(arow, D.r, D.mc),
                                          nvec, K, _sof, 0.95 * gam, wlo=-np.inf, whi=np.inf, rtol=1e-6)
            gam_dense_all = float(_gd[0])
            if gam_dense_all > gam * (1.0 + 1e-5):
                print(f"WARNING: dense level set over ALL w gives gamma {gam_dense_all:.6f} at w = {_gd[1]:+.4f}, "
                      f"above the loop's certified {gam:.6f} over [{a.wlo:.2f}, {a.whi:.2f}]: the band missed a "
                      f"peak; the record stores the dense value", flush=True)
                gam = gam_dense_all
            else:
                print(f"dense all-frequency check: gamma {gam_dense_all:.6f} (certified {gam:.6f})", flush=True)
        except Exception as _e:
            print(f"dense all-frequency check failed non-fatally: {_e!r}", flush=True)
    if MIMO:          # per-port fields; the pre-Sobolev coordinates are a SISO-resume device only
        xia = xis = np.zeros(0)
    else:
        xia = D.smooth_unmap(va) if va is not None else np.zeros(0)
        xis = D.smooth_unmap(vs) if vs is not None else np.zeros(0)
    prev = None
    if os.path.exists(tag + '.npz'):
        try:
            pz = np.load(tag + '.npz', allow_pickle=True)
            prev = float(pz['gamma'])
            # only a STABLE previous result may block a new one -- and its
            # stability must be RE-VERIFIED, not read from the file.  The
            # stored alpha is exactly the quantity that is wrong when a design
            # escaped: eight files carried alpha = -0.216 over eigenvalues at
            # Re = +0.33 to +10.1, and each blocked every stable replacement
            # (all of which have a higher gamma) through clean re-runs.  At GL
            # scale the check is dense (about a second, definitive); at cavity
            # scale it trusts a stored alpha_dense if the file carries one,
            # and otherwise falls back to the stored alpha with a warning --
            # acceptable there only because the run that wrote it enforced
            # the continuation gate.
            prev_ok = False
            if MIMO:
                prev_ok = ('alpha_dense' in pz.files and np.isfinite(float(pz['alpha_dense']))
                           and float(pz['alpha_dense']) + a.eps <= 0.0)
            elif a.dense_check and (D.n + D.r) <= a.dense_nmax:
                import warmstart as _ws
                prev_ok = _ws.is_stable(D, tag + '.npz', eps=a.eps,
                                        dense=True)
            elif 'alpha_dense' in pz.files:
                prev_ok = float(pz['alpha_dense']) + a.eps <= 0.0
            else:
                prev_ok = float(pz['alpha']) + a.eps <= 0.0
                print("guard: previous file has no alpha_dense and n is too "
                      "large for the dense check -- trusting its stored "
                      "alpha", flush=True)
            if not prev_ok:
                print(f"previous result gamma={prev:.6f} FAILS re-"
                      f"verification (stored alpha "
                      f"{float(pz['alpha']):+.4f}) -- overwriting",
                      flush=True)
                prev = None
        except Exception:
            prev = None
    if prev is not None and prev < gam * (1 - 1e-12):
        print(f"keeping previous better result gamma={prev:.6f}", flush=True)
        return
    _apa, _aps = pk.atom_params(zbest)
    np.savez(tag + '.npz', d=arow[0], e=arow[1], f=arow[2], nvec=nvec, K=K,
             xa=xa, xs=xs, xi_a=xia, xi_s=xis,
             atoms_a_p=(_apa if _apa is not None else np.zeros(0)),
             atoms_s_p=(_aps if _aps is not None else np.zeros(0)),
             va=(va if va is not None else np.zeros(0)),
             vs=(vs if vs is not None else np.zeros(0)),
             psi_a=pa, psi_s=ps, gamma=gam, alpha=al,
             **(dict(Bc=np.asarray(arow[3]), loop_orders=np.array(flags['loop_orders']), m=int(flags['m']),
                     p=int(flags['p']), struct=flags['struct']) if MIMO else {}),
             alpha_dense=(al_d if al_d is not None else np.nan),
             n_active=nact,
             omega_grid=ws, gain=gain, gain_model=_gain_model, active_freqs=np.array(sorted(Omega)))
    summ = dict(case=a.case, start=a.start, ne=a.ne, n=D.n, NZ=len(z) + 1,
                gamma=float(gam), alpha=float(al),
                alpha_dense=(float(al_d) if al_d is not None else None),
                gamma_dense_all_w=gam_dense_all,
                band_incomplete=int(n_band_incomplete[0]), band_check=band_check, all_w=all_w,
                all_w_outside_band=int(n_allw_outside[0]),
                eps=float(a.eps),
                wind_count=wind_n,
                sym=bool(a.sym),
                sym_branch=(float(flags.get('sym_branch', 1.0))
                            if a.sym else None),
                n_active=int(nact),
                calls=int(sum(h['inner_calls'] for h in hist)),
                # What stopped the inner solve, recorded so a reader can tell a
                # reproducible row from one the clock decided.  A row with
                # inner_budget='clock', or with clock_guard_bound true, must
                # not be compared digit-for-digit against another machine.
                inner_budget=a.inner_budget,
                ls_fallback=bool(a.ls_fallback),
                ctrl_stab=a.ctrl_stab, ctrl_imcap=a.ctrl_imcap,
                cont_anchor=a.cont_anchor,
                sigma2=a.sigma2,
                qp_eps=a.qp_eps, qp_maxiter=a.qp_maxiter,
                level_norm=a.level_norm,
                qp_solver=a.qp_solver,
                inner=a.inner, sqp_block=a.sqp_block,
                omega_policy=a.omega_policy,
                oracle=a.oracle,
                stabilise=_stab_rec,
                oracle_log=(_oracle_log if a.oracle == 'reduced' else None),
                omega_log=(_omega_log if a.omega_policy == 'an' else None),
                opt_tol=float(a.opt_tol),
                select_cons=a.select_cons,
                band_final=[float(a.wlo), float(a.whi)],
                inner_maxit=(int(a.inner_maxit)
                             if a.inner_budget == 'iters' else None),
                clock_guard_bound=bool(any(
                    h.get('clock_guard_bound') for h in hist)),
                termination_codes=[h.get('termination_code') for h in hist],
                # which init/seeds this run actually started from -- see
                # _input_fingerprint for why guessing this afterwards is costly
                inputs=_input_fingerprint(a.ne, a.init),
                seconds=round(time.time() - t_start, 1),
                lu_full_by_site=(_luc.snapshot().get('by_site_full') if _luc is not None else None),   # whole run incl. save (--lucount / --dump)
                method='epigraph', outer=len(hist), nOmega=len(Omega),
                wlo=float(a.wlo), whi=float(a.whi),
                ctrl_wmax=float(wmax),
                gate_inconclusive=int(n_inconcl[0]),
                gate_located=int(n_located[0]), witness_invariant_violations=int(n_witness_bad[0]), closed_loop_pair_rounds=int(n_clpair[0]), cont_track=a.cont_track, cont_fallbacks=int(n_cont_fallback[0]),
                cert_disks=dict(untrusted=int(csp.N_DISK_UNTRUSTED), retried=int(getattr(csp, 'N_DISK_RETRIED', 0)), oversized=int(getattr(csp, 'N_DISK_OVERSIZED', 0)),
                                bisected=int(getattr(csp, 'N_DISK_BISECTED', 0)), uncovered=int(getattr(csp, 'N_DISK_UNCOVERED', 0)), climb_failed=int(csp.N_CLIMB_FAILED)),
                exchange_converged=bool(exch_converged),
                exchange_fixed_point=bool(exch_fixed_point), exchange_progress_stop=bool(exch_progress_stop),
                exchange_stalled=bool(exch_stalled),
                ladder=[], xa=(np.atleast_1d(xa).astype(float).tolist() if MIMO else float(xa)),
                xs=(np.atleast_1d(xs).astype(float).tolist() if MIMO else float(xs)),
                norm_psi_a=([D.norm_M(pa[:, j]) for j in range(pa.shape[1])] if MIMO else D.norm_M(pa)),
                norm_psi_s=([D.norm_M(ps[:, j]) for j in range(ps.shape[1])] if MIMO else D.norm_M(ps)),
                wind_accept=dict(on=bool(_wind_accept), checks=int(n_wind_accept[0]), refused=int(n_wind_refused[0]),
                                 gate_inconclusive_cleared=int(n_gate_wind[0])),
                mimo=(dict(m=int(flags['m']), p=int(flags['p']), struct=flags['struct'], loop_orders=list(flags['loop_orders']),
                           start=a.mimo_start, D_c=[[str(v) for v in row] for row in np.asarray(K).reshape(flags['m'], flags['p'])])
                      if MIMO else None),
                eps_stab=a.eps, sob_len=a.sob_len, parent=a.parent,
                parent_gamma=None, spike_suspect=bool(al > -0.01),
                gain_peak_over_median=float(gain.max() /
                                            max(np.median(gain), 1e-30)),
                history=hist)
    json.dump(summ, open(tag + '.json', 'w'), indent=1)
    for _c in (tag + '_ckpt.npz', tag + '_ckpt.json'):
        try:
            if os.path.exists(_c):
                os.remove(_c)
        except OSError:
            pass
    if _pool is not None:
        try:
            _pool.close(); _pool.join()
        except Exception:
            pass
    print(json.dumps({k: summ[k] for k in
                      ('case', 'gamma', 'alpha', 'n_active', 'outer',
                       'nOmega', 'calls', 'seconds')}), flush=True)


if __name__ == "__main__":
    main()
