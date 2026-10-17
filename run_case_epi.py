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

# The backward-facing step's winding counts right of Re = -eps (the feasible
# start, a refused continuation, a resumed anchor, the save-time veto): an
# ABSOLUTE step floor for the march (20261010j stage B, gate-refusal.md G1 with
# its verifier's correction).  The records sit on the rows' target, ~1e-4 left
# of the line (alpha -0.0501042 / -0.0501307); the relative floor 1e-6 x
# min(W_T, 4000) is 4e-3 for MIMO, and GL's 1e-8 x W_T would be 4e-5 here, about
# the 6e-5 that failed in STAGE5 22.
WIND_HMIN_ABS = 1e-6


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


def _track_report(tr, eps, worst_only=True):
    """The offending tracks of one continuation_step (20261010j stage B,
    gate-refusal.md G2): lam0 -> lam1, the motion and its allowance -- crossed
    (Re > -eps), moved beyond the allowance, or not converged."""
    out = []
    for l0, l1, al_ in zip(tr.get('lam0', []), tr.get('lam1', []), tr.get('allow', [])):
        if l1 is None:
            out.append(f"{l0:.6g} -> not converged")
        elif l1.real > -eps or abs(l1 - l0) > al_ or not worst_only:
            out.append(f"{l0:.6g} -> {l1:.6g} (moved {abs(l1 - l0):.3g}, allowance {al_:.3g}"
                       + (", CROSSED" if l1.real > -eps else '') + ")")
    return out


def gate_refusal_report(pts, pinf, glogs, iseg, eps, log=print):
    """A refusal along the SQP's path, readable (20261010j stage B,
    gate-refusal.md G2): the failing segment's ends -- the iteration that
    reached each, its step alpha, whether the step left an admissible iterate
    (False: the segment closes a merit excursion), returns to the last
    admissible iterate in between, each end's own line (alpha, rightmost seen,
    rows) -- and every sub-segment that did not pass, with its tracks lam0 ->
    lam1.  770633 round 1 printed one line, 'crossed to -0.0498897', which could
    not tell a hidden eigenvalue from a disagreement of two computations.
    pts: the gate's points (the last accepted design first), pinf: the SQP's
    path_info for each (None for the first), glogs: _gate_along's per-segment
    logs, iseg: the failing segment."""
    def _end(j):
        p = pinf[j] if j < len(pinf) else None
        if p is None:
            return "the last accepted design"
        rs = sorted(p.get('seen') or [], key=lambda q: -q.real)[:3]
        al_s = p.get('alpha')
        return (f"SQP it {p.get('it')} (step alpha {('%.4g' % al_s) if al_s is not None else '-'}"
                + ('' if p.get('prev_adm') is None else
                   (', from an admissible iterate' if p['prev_adm'] else ', from an INADMISSIBLE iterate (merit excursion)'))
                + (f", {p['backs']} return(s) to the last admissible iterate before" if p.get('backs') else '')
                + (f", {p['how']}" if p.get('how') else '')
                + f"; line alpha {p.get('al', np.nan):+.7f}, rightmost seen {[f'{q:.6g}' for q in rs]}, "
                  f"{len(p.get('rows') or [])} row(s))")
    log(f"    gate refusal: path segment {iseg + 1} of {len(pts) - 1}, |dz| "
        f"{float(np.linalg.norm(np.asarray(pts[iseg + 1]) - np.asarray(pts[iseg]))):.4g}: from {_end(iseg)} to "
        f"{_end(iseg + 1)}", flush=True)
    seg = glogs[iseg] if iseg < len(glogs) else None
    for ent in (seg['log'] if seg else []):
        if ent.get('verdict') in ('certified', 'split'):
            continue
        log(f"      [{ent['ta']:.4g}, {ent['tb']:.4g}] (depth left {ent['depth']}): {ent.get('verdict')}"
            + (f", midpoint alpha {ent['alpha_mid']:+.5f}" if 'alpha_mid' in ent else '') + ": "
            + "; ".join(_track_report(ent, eps, worst_only=False)), flush=True)


def _js(o):
    """JSON-safe (complex as [re, im]): gate_fail_k.npz's 'info', the veto's record."""
    if isinstance(o, dict):
        return {str(k): _js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_js(v) for v in o]
    if isinstance(o, np.ndarray):
        return _js(o.tolist())
    if isinstance(o, (complex, np.complexfloating)):
        return [float(np.real(o)), float(np.imag(o))]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def save_gate_fail(outdir, rec, pk, flags, mimo, eps, gamma_best, cont, log=print):
    """A gate refusal as gate_fail_k.npz in outdir, k the next free index (a
    resumed job keeps the earlier jobs' files) -- 20261010j stage B,
    gate-refusal.md G2, as qp_fail_k.npz (20261010f).  rec: the round, the
    anchor and candidate designs (z0, z1) and the candidate's gamma, alpha and
    seen list, the anchor's watch list, the path (pts), its SQP info (pinf) and
    segment logs (glogs), the failing segment (iseg; None: the path was not
    gated or passed), the straight chord's log and verdict (clog, chord_ok: a
    path refusal the chord then certified is saved too), and the fallbacks'
    outcome (accepted, count, prefix_accepted).
    Saved: the designs in record form (anchor_*, cand_*, and the failing path
    segment's ends seg0_*, seg1_*: d, e, f, Bc, nvec, K, xa, xs, va, vs and z),
    the path, and 'info' (JSON) -- enough to replay the refusal offline (a count
    right of -eps, a LATTICE_K=2 line at the failing end).  Never fatal.
    Returns the file name, or None."""
    try:
        k = 1
        while os.path.exists(os.path.join(outdir, f"gate_fail_{k}.npz")):
            k += 1
        fn = os.path.join(outdir, f"gate_fail_{k}.npz")
        arr = {}

        def _des(pre, z_):
            ar, nv, K, xa, xs, va, vs = pk.unpack(z_)
            arr.update({pre + 'z': np.asarray(z_, float), pre + 'd': ar[0], pre + 'e': ar[1], pre + 'f': ar[2],
                        pre + 'nvec': nv, pre + 'K': K, pre + 'xa': xa, pre + 'xs': xs})
            if mimo:
                arr[pre + 'Bc'] = np.asarray(ar[3])
            for key, v in (('va', va), ('vs', vs)):     # the free fields (FB / FC), when the case has them
                if v is not None:
                    arr[pre + key] = np.asarray(v)
        _des('anchor_', rec['z0']); _des('cand_', rec['z1'])
        if rec.get('pts') is not None and rec.get('iseg') is not None:
            _des('seg0_', rec['pts'][rec['iseg']]); _des('seg1_', rec['pts'][rec['iseg'] + 1])
            arr['path'] = np.array([np.asarray(p, float) for p in rec['pts']])
        if mimo:
            arr.update(loop_orders=np.array(flags['loop_orders']), m=int(flags['m']), p=int(flags['p']),
                       struct=str(flags['struct']))
        info = dict(round=rec['round'], eps=float(eps), gamma_best=float(gamma_best), gamma_cand=float(rec['g']),
                    alpha_cand=float(rec['al']), seen_cand=list(rec.get('seen') or []), watch_anchor=rec['watch'],
                    path_info=rec.get('pinf'), path_logs=rec.get('glogs'), path_failed_segment=rec.get('iseg'),
                    chord_log=rec.get('clog'), chord_ok=rec.get('chord_ok'), accepted=rec.get('accepted'), count=rec.get('count'),
                    prefix_accepted=rec.get('prefix_accepted'), cont=dict(cont), code_version=_code_version())
        np.savez(fn, info=json.dumps(_js(info)), **arr)
        log(f"    gate refusal saved -> {fn} (replay: the designs anchor_*, cand_*, seg0_*/seg1_*; "
            f"np.load(...)['info'] is JSON)", flush=True)
        return fn
    except Exception as e:
        log(f"    gate refusal: save failed ({type(e).__name__}: {e})", flush=True)
        return None


def continuation_gate(D, pk, z0, z1, watch0, eps, depth=4,
                      motion_frac=0.5, motion_floor=0.05, kwatch=8,
                      verbose=True, pool=None, reach=None, log=None):
    """Certify the design segment [z0, z1] free of stability-line crossings.

    Returns (ok, watch1, nquery, nsplit).  `watch1` is the watch list at z1
    (valid only when ok).  Conservative by construction: an unconverged track,
    an infeasible midpoint, or exhausted depth all reject.  The packed design
    vector is real and every packed quantity is linear in it, so the straight
    segment is a continuous path through design space, which is all
    continuation needs.

    reach (a dict, optional): filled with the end of the CERTIFIED PREFIX of
    the segment -- the subsegments are certified left to right, so the last
    one passed ends at the furthest point reached from z0 without a crossing:
    reach['z'], its watch list reach['watch'] (BFS.md 5, --gate-prefix).

    log (a list, optional; 20261010j stage B, gate-refusal.md G2): one entry per
    sub-segment checked -- its place [ta, tb] on [z0, z1], the split depth left,
    every track (lam0 -> lam1, allowance), the verdict, and a midpoint's alpha
    and seen list -- for the refusal report and gate_fail_k.npz.  Decisions
    are unchanged.
    """
    z0 = np.asarray(z0, float)
    z1 = np.asarray(z1, float)
    stats = dict(nq=0, nsplit=0)

    def _design(z):
        arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
        b, c, *_ = D.bc(xa, xs, va, vs)
        return arow, nvec, K, b, c, ctl.A_of(*arow)

    def _seg(za, zb, wa, d, ta=0.0, tb=1.0):
        wa_use = [w for w in wa if w is not None]
        ent = dict(ta=float(ta), tb=float(tb), depth=int(d))
        if log is not None:
            log.append(ent)
        if not wa_use:
            if verbose:
                print("    continuation: empty watch list -> cannot prove; "
                      "reject", flush=True)
            ent.update(verdict='empty watch list')
            return False, wa
        arow_b, nvec_b, K_b, b_b, c_b, Ac_b = _design(zb)
        Bc = ctl.Bc_of(arow_b, D.r, D.mc)
        tr = {}
        ok, split, wb, worst = csp.continuation_step(
            D.A, D.M, b_b, c_b, Ac_b, Bc, nvec_b, K_b, wa_use, eps=eps,
            motion_frac=motion_frac, motion_floor=motion_floor, pool=pool, info=tr)
        ent.update(tr, ok=bool(ok), split=bool(split), worst=worst)
        stats['nq'] += len(wa_use)
        if not ok:
            if verbose:
                print(f"    continuation REJECTS: tracked eigenvalue crossed "
                      f"to {worst:.6g} on [{ta:.4g}, {tb:.4g}] of the segment: "
                      + "; ".join(_track_report(tr, eps)), flush=True)
            ent.update(verdict='crossed')
            return False, wb
        if not split:
            if reach is not None:
                reach.update(z=np.array(zb, float), watch=list(wb))
            ent.update(verdict='certified')
            return True, wb
        if d <= 0:
            if verbose:
                print("    continuation REJECTS: split depth exhausted with "
                      "motion still above the allowance (step too large to "
                      f"certify) on [{ta:.4g}, {tb:.4g}]: " + "; ".join(_track_report(tr, eps)), flush=True)
            ent.update(verdict='depth exhausted')
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
        ent.update(alpha_mid=float(al_m), seen_mid=list(seen_m))
        if al_m + eps > 0.0:
            if verbose:
                print(f"    continuation REJECTS: homotopy midpoint is "
                      f"infeasible (alpha={al_m:+.4f}) at {0.5 * (ta + tb):.4g} of the segment; "
                      f"its rightmost seen {[f'{q:.5g}' for q in sorted(seen_m, key=lambda q: -q.real)[:3]]}", flush=True)
            ent.update(verdict='midpoint infeasible')
            return False, wb
        ent.update(verdict='split')
        tm = 0.5 * (ta + tb)
        ok1, wm = _seg(za, zm, wa, d - 1, ta, tm)
        if not ok1:
            return False, wm
        wm2 = _topk_watch(list(wm) + seen_m, kwatch, eps)
        return _seg(zm, zb, wm2, d - 1, tm, tb)

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
                _gz = pk.zero_grad() if (getattr(pk, 'mimo', False) or getattr(pk, 'vec_pos', False)) else dict(n=np.zeros(_rr, complex), K=0j, xa=0.0, xs=0.0)
                _gz.update(d=gdi, e=gei, f=gfi)
                g_im = np.real(pk.grad(_gz))
            except Exception:
                g_im = np.zeros(NZ)
        if _G is not None and flags['K'] \
                and flags.get('ctrl_mode', 'full') in ('full', 'sp'):
            gd = np.array([_G[i, i] for i in range(_rr)])
            ge = np.array([_G[i, i + 1] for i in range(_rr - 1)])
            gf = np.array([_G[i + 1, i] for i in range(_rr - 1)])
            gc = pk.zero_grad() if (getattr(pk, 'mimo', False) or getattr(pk, 'vec_pos', False)) else dict(n=np.zeros(_rr, complex), K=0j, xa=0.0, xs=0.0)
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


def _round_lattice_lines(li):
    """The round's lattice measurements (20261010j stage C), three lines after the
    round's 'lattice:' lines: the members' points and tasks per kind and by s /
    smax, and the agents that set the last wave (omega-growth.md P1, strip-ab.md
    AB4); the level set's disks by path, their applications, CG and critical path
    (level-set-waves.md LS1); the stability rows at the QP solutions -- which
    bind, how deep, at what radius against the box (strip-ab.md AB4, the
    'boxing' hypothesis of BFS.md 8)."""
    import lattice as _lm
    m = li.get('members')
    if m and m['evals']:
        E = float(m['evals'])
        pc = lambda v_, t_: '/'.join(f"{100.0 * x_ / max(t_, 1):.0f}" for x_ in v_)
        bb = list(_lm.MEMBER_BINS)
        bl = '/'.join([f"<{bb[0]:g}"] + [f"{a_:g}-{b_:g}" for a_, b_ in zip(bb[:-1], bb[1:])] + [f">={bb[-1]:g}"])
        print(f"    lattice: members per evaluation (mean of {m['evals']}): fixed {m['n'][0] / E:.0f} "
              f"({m['pts'][0] / E:.0f} points, {m['own'][0] / E:.0f} tasks), followed {m['n'][1] / E:.0f} "
              f"({m['pts'][1] / E:.0f} points, {m['own'][1] / E:.0f} tasks; {m['none'] / E:.1f} caused none); by s/smax "
              f"{bl}: members {'/'.join(f'{x_ / E:.0f}' for x_ in m['bn'])}, "
              f"points {pc(m['bpts'], sum(m['bpts']))} %, tasks {pc(m['bown'], sum(m['bown']))} %; the last wave set by "
              + (", ".join(f"{k_} {v_}" for k_, v_ in sorted(m['last'].items(), key=lambda kv: -kv[1])) or 'nothing')
              + f" (of {m['evals']} evaluations)", flush=True)
    d = li.get('level_disks')
    if d:
        def lab(p_):
            if p_.startswith('tier '):
                j_ = int(p_.split()[1])
                return f"ncv {_lm.LEVEL_ARNOLDI[j_][0]}" if j_ < len(_lm.LEVEL_ARNOLDI) else p_
            return p_
        ps = sorted(d['paths'].items(), key=lambda kv: (kv[0] != 'norm', kv[0]))
        sl = d.get('slow')
        print(f"    lattice: level disks of {d['calls']} call(s): "
              + ", ".join(f"{lab(p_)} {a_[0]} ({a_[1] / max(a_[0], 1):.1f} s each, up to {a_[2]:.1f})" for p_, a_ in ps)
              + f"; applications up to {d['app_max']}, CG {d['cg_sum'] / max(d['cg_n'], 1):.1f} per solve (up to "
              f"{d['cg_max']}); critical path {d['crit']:.0f} s of {d['work']:.0f} s of disks"
              + ("; the slowest at " + ", ".join(f"{q_[0]:.4f} ({lab(q_[2])}, {q_[1]:.1f} s, {q_[3]} applications)"
                                                  for q_ in sl) if sl else ''),
              flush=True)
    b = li.get('box')
    if b:
        f_ = lambda v_, fm: ('-' if v_ is None else format(v_, fm))
        print(f"    lattice: QP stability rows: active in {b['with_act']} of {b['qps']} QPs (median {f_(b['act_med'], 'g')}, "
              f"up to {b['act_max']} of {b['rows_max']}; {b['given_up']} given up), the deepest active Re "
              f"{f_(b['deep'], '+.4f')} (|g| up to {b['gact']:.1e}); the active rows bind at radius {f_(b['rb_act_med'], '.1e')} "
              f"(median) against the trust radius {f_(b['Delta_med'], '.1e')} ({f_(b['in_box_med'], 'g')} rows could bind inside it), "
              f"{b['on_box']} steps on its boundary, a Hermite block active in {b['herm']}; accepted alpha "
              f"{f_(b['alpha_med'], 'g')}, step {f_(b['step_med'], '.1e')} (medians); "
              + (f"copies of one eigenvalue (within {_lm.ROW_DUP:g}, not merged; the review of stage C): up to "
                 f"{b.get('dup_max', 0)} in a QP, active in {b.get('with_act_dup', 0)} QPs (up to {b.get('act_dup_max', 0)}), "
                 f"{b.get('el_dup', 0)} given up" if _lm.ROW_MERGE <= 1e-8 else f"copies merged within {_lm.ROW_MERGE:g}"),
              flush=True)


_CODE_VERSION = []


def _code_version() -> str:
    """'<MANIFEST label> <git commit>[+modified]' of the code running (20261010j
    stage B): written into the record, the checkpoint and the _partN.json, and
    printed on a resume ('checkpoint written by X, continued by Y').  The run
    directories are shared by every code version (bfs/home.sh), and a record
    or checkpoint carried no trace of the code that wrote it -- only the slurm
    scripts' code_versions.txt beside it, one line per job, not per file.
    '+modified': tracked files differ from the commit (a sandbox edit; on the
    cluster bfs/version_check.py refuses a checkout whose files differ from its
    MANIFEST).  Once per process."""
    if _CODE_VERSION:
        return _CODE_VERSION[0]
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    label = '?'
    try:
        with open(os.path.join(here, 'bfs', 'MANIFEST')) as fh:
            for line in fh:
                if line.startswith('# version'):
                    label = line.split(None, 2)[2].strip()
                    break
    except (OSError, IndexError):
        pass

    def _git(*args):
        try:
            r = subprocess.run(['git', '--no-optional-locks', *args], cwd=here, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, universal_newlines=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    head = _git('rev-parse', '--short', 'HEAD')
    st = _git('status', '--porcelain', '--untracked-files=no') if head else None
    _CODE_VERSION.append(f"{label} {head or 'no-git'}" + ('+modified' if st else ''))
    return _CODE_VERSION[0]


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


def _insolve_summary(add, moved, nottop, dropped, Omega, peakset, enrol):
    """The round's in-solve additions after the round's end (lattice, 20261010j
    stage E): add the additions' keys (round 6), moved {key: the in-solve top it
    was moved to}, nottop the keys left in place (no top at the candidate),
    dropped the keys left out on another follower's top, Omega and peakset after
    the round's end (the 1e-3 merge included), enrol the SQP's location counts
    (sqp_inner._enrol_at_tops).  An addition is FOLLOWED from the next round on
    where its member (moved or not) is in Omega and the peak set, else MERGED:
    on another follower's top, or within 1e-3 of a member; a merged one leaves
    the peak set too.  Returns the counts, the followed members and the round's
    one-line summary ('line')."""
    Of = set(round(float(w_), 6) for w_ in Omega)
    fin = {k_: moved.get(k_, k_) for k_ in add}
    fol = [k_ for k_ in add if fin[k_] in Of and fin[k_] in peakset]
    mrg = [k_ for k_ in add if k_ not in fol]
    for k_ in mrg:
        if fin[k_] not in Of:
            peakset.discard(fin[k_])
    sh = sum(k_ in dropped for k_ in mrg)
    out = dict(added=len(add), followed=len(fol), merged=len(mrg), shared=int(sh), near=len(mrg) - int(sh),
               moved=sum(k_ in moved for k_ in fol), not_top=sum(k_ in nottop for k_ in fol),
               members=sorted(fin[k_] for k_ in fol),
               regions=int(enrol.get('regions', 0)), at_top=int(enrol.get('at_top', 0)),
               at_estimate=int(enrol.get('not_top', 0)), on_followed=int(enrol.get('watched', 0)),
               same=int(enrol.get('same', 0)), dist_max=float(enrol.get('dist_max', 0.0)))
    out['line'] = (f"    in-solve additions: {out['added']} this round ({out['regions']} region(s) located: {out['at_top']} at a "
                   f"top, up to {out['dist_max']:.4f} from the estimate, {out['at_estimate']} at the estimate (no top), "
                   f"{out['on_followed']} on a followed member already); {out['followed']} followed from the next round "
                   f"({out['moved']} moved to their in-solve tops, {out['not_top']} not at a top left in place), "
                   f"{out['merged']} merged ({out['shared']} on another follower's top, {out['near']} within 1e-3 of a member)")
    return out


def _round_end_follow(Omega, peakset, members, ins_add, wst, wtop, follow_fn=None):
    """The round's end for the followed members (--peak-follow): each moved to
    its peak's top AT THE CANDIDATE, so that the next round starts on the tops;
    then members closer than 1e-3 merged.  Omega the working set (keys round 6),
    peakset the peak members (changed in place), members the followed keys in
    the order they are moved (this round's followers, then the in-solve
    additions ins_add, stage E); wst / wtop the lattice's tops at the candidate
    and whether each is one (the SQP's w_star / w_top), follow_fn the reduced
    model's locator (w -> top; None: the lattice).  Returns (Omega, counts
    {moved, merged, shared, nottop}, the in-solve additions moved {key: top},
    those not at a top, the keys dropped on another member's top).  (Moved out
    of main() unchanged in 20261010j stage E, for its unit test.)"""
    moved = 0; newO = set(Omega); nottop = 0; shared = 0; dropped = set()
    ins_add = set(ins_add); ins_mv = {}; ins_nt = set()
    if follow_fn is None:
        # MEMBERS TRULY ON ONE TOP (lattice, 20261010j, missed-peaks.md P4): the
        # tops at the candidate (after the solve's dedup restarts) agreeing within
        # 1e-4 are one peak -- one member keeps it (a peak member, followed next
        # round; then the one whose base is nearest), the others leave Omega.  The
        # 1e-3 merge below never caught them: a member whose top lies within 1e-3
        # of another member is not moved (eval-waves.md).  Every member followed
        # in the solve counts (the in-solve additions too: they have tops in
        # w_star, and are peak members now -- stage E -- so between an addition
        # and an older follower on one top the nearest base keeps it)
        grp = sorted((round(float(wst[round(float(w), 6)]), 6), round(float(w), 6), w) for w in Omega
                     if round(float(w), 6) in wst and wtop.get(round(float(w), 6), True))
        g = []
        for tp, k, w in grp + [(np.inf, None, None)]:
            if g and tp - g[-1][0] > 1e-4:
                if len(g) > 1:
                    kw = min(g, key=lambda q_: (q_[1] not in peakset, abs(q_[1] - q_[0])))[1]
                    for _, k2, w2 in g:
                        if k2 != kw:
                            newO.discard(w2); peakset.discard(k2); dropped.add(k2); shared += 1
                g = []
            g.append((tp, k, w))
    for w in members:
        if round(float(w), 6) in dropped:
            continue                                   # on another member's top: dropped just above
        if follow_fn is None:
            ws = wst.get(round(float(w), 6), w)
            if not wtop.get(round(float(w), 6), True):
                # NOT A TOP (P3): stopped on a window's edge or still climbing --
                # never a member's new place (770632 r11 moved the member onto the
                # edge point 4.13125, slope +284); it stays where it is
                nottop += 1
                if w in ins_add:
                    ins_nt.add(w)
                continue
        else:
            ws = follow_fn(w)
        ws = round(float(ws), 6)
        if abs(ws - w) > 1e-4 and all(abs(ws - q) > 1e-3 for q in newO if q != w):
            newO.discard(w); newO.add(ws); peakset.discard(w); peakset.add(ws); moved += 1
            if w in ins_add:
                ins_mv[w] = ws
    # MERGE members closer than 1e-3 (following can bring several onto one
    # top; duplicate rows made the QP nearly degenerate), keeping a peak
    # member where there is one
    merged = []
    for q in sorted(newO):
        if merged and abs(q - merged[-1]) < 1e-3:
            if q in peakset and merged[-1] not in peakset:
                merged[-1] = q
            continue
        merged.append(q)
    return merged, dict(moved=moved, merged=len(newO) - len(merged), shared=shared, nottop=nottop), ins_mv, ins_nt, dropped


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


def _sigmas(cl_, ws, arow, nvec, K):
    """sigma_1 at every frequency: one batched call on a reduced model
    (reduced.gain_batch, values only), the per-frequency path otherwise
    (STAGE6 6.29: the loop's reduced sweeps -- up to 4001 points -- were Python
    loops of single-frequency calls)."""
    ws = np.asarray(ws, float)
    if hasattr(cl_, 'gain_batch'):
        # maxit 200 (gain_batch's default 40): a row the batched subspace
        # iteration has not converged falls back to a per-row ARPACK svds, ~1 s
        # at kd ~ 300; near-equal sigma_1, sigma_2 (two optimised fields, high
        # gain) converge slowly, and the 4001-point save curve of K+FB+FC at
        # ne = 200 spent > 20 min in fallbacks.  Rows that converge within 40
        # iterations are unaffected (STAGE6 6.31)
        return np.array([r_['sigma'] for r_ in cl_.gain_batch(ws, arow, nvec, K, np.zeros(len(ws), bool), maxit=200)],
                        float)
    return np.array([cl_.gain(float(w), arow, nvec, K, want_vectors=False)['sigma'] for w in ws], float)


PIN_MAX = 24          # pinned anchor LUs held from a certificate to the next enrichment (STAGE6 6.29)


def _pin_lus(D, freqs):
    """Hold the factors of iwM - A at these frequencies (those still in the
    FIFO cache) until the next enrichment consumes them; at most PIN_MAX."""
    if not getattr(certified_norm, 'pin', False):
        return
    for x in freqs:
        if not np.isfinite(x):
            continue
        key = complex(float(x))
        lu = D._lu_cache.get(key)
        if lu is not None and key not in D._lu_pin and len(D._lu_pin) < PIN_MAX:
            D._lu_pin[key] = lu


def _true_lower_bound(sig, sigd, sig_many, ro_, b, c, D, arow, nvec, K, wlo, whi, w_hint):
    """(w, sigma_1(w)) with the FULL model, as high as cheaply found: the
    candidates are the caller's hint and the reduced model's three highest
    local maxima (fine sweep, free), sampled at once (the pool), the best
    climbed by its slope.  (None, None) without candidates."""
    cw = []
    if w_hint is not None and np.isfinite(w_hint) and wlo <= w_hint <= whi:
        cw.append(float(w_hint))
    if ro_ is not None and ro_.dim > 0:
        rc0 = ro_.make_cl(b, c, D.r, D.mc)
        wf0 = np.linspace(wlo, whi, 1201)
        gf0 = _sigmas(rc0, wf0, arow, nvec, K)
        i0 = np.arange(1, len(gf0) - 1)
        loc0 = i0[(gf0[i0] >= gf0[i0 - 1]) & (gf0[i0] >= gf0[i0 + 1])]
        cw += [float(wf0[j]) for j in loc0[np.argsort(-gf0[loc0])][:3]] + [float(wf0[int(np.argmax(gf0))])]
    if not cw:
        return None, None
    cw = np.array(sorted(set(round(x, 9) for x in cw)))
    vals = np.array(sig_many(cw) if sig_many else [sig(x) for x in cw])
    j = int(np.argmax(vals))
    span = 0.05 * (whi - wlo)
    s0, d0 = sigd(float(cw[j]))
    wc, gc = csp._climb_slope(sigd, float(cw[j]), s0, d0, max(wlo, cw[j] - span), min(whi, cw[j] + span),
                              xtol=1e-7 * (whi - wlo))
    return (float(wc), float(gc)) if gc >= vals[j] else (float(cw[j]), float(vals[j]))


def certified_norm(D, pk, z, wlo, whi, gam_lb, w_hint=None):
    """Certified max_omega sigma_1 and a witness frequency, for the outer loop."""
    arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
    b, c, *_ = D.bc(xa, xs, va, vs)
    cl = D.cl(b, c)
    sig = lambda w: cl.gain(w, arow, nvec, K, want_vectors=False)['sigma']
    sigd = lambda w: (lambda r_: (r_['sigma'], r_['gprime']))(cl.gain(float(w), arow, nvec, K))   # the slope climb (--cert-climb)
    import fullpar
    # several full samples at once: in the full-model pool when it is up (STAGE6 6.30)
    sig_many = (lambda ws_: fullpar.sigmas(b, c, arow, nvec, K, ws_)) if fullpar.active() else None
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
    if getattr(certified_norm, 'sampled', 0):
        # BACKWARD-FACING STEP, scale test (BFS.md 5, user 2026-10-01): no
        # level-set certificate (shift-invert Arnoldi on the 2n even pencil does
        # not converge away from its spectrum there, and the free-radius bound
        # needs an M-norm).  gamma = the largest FULL-model sigma_1 over a grid
        # of the band, the reduced model's peaks and the caller's hint, each
        # local maximum climbed (sigma and its slope from the full model).  Not
        # a certificate: the record says so (certified_norm.last_sampled).
        ng = int(certified_norm.sampled)
        cands = list(np.linspace(wlo, whi, ng))
        if w_hint is not None and np.isfinite(w_hint):
            cands.append(float(w_hint))
        if ro_ is not None and ro_.dim > 0:
            _rc = ro_.make_cl(b, c, D.r, D.mc)
            _wf = np.linspace(wlo, whi, 1201); _gf = _sigmas(_rc, _wf, arow, nvec, K)
            _i = np.arange(1, len(_gf) - 1)
            _loc = _i[(_gf[_i] >= _gf[_i - 1]) & (_gf[_i] >= _gf[_i + 1])]
            cands += [float(_wf[j]) for j in _loc[np.argsort(-_gf[_loc])][:6]]
        cands = np.array(sorted(set(round(float(x), 9) for x in cands)))
        vals = np.array(sig_many(cands) if sig_many else [sig(x) for x in cands])
        j_ = np.arange(len(cands))
        is_max = (np.r_[True, vals[1:] >= vals[:-1]]) & (np.r_[vals[:-1] >= vals[1:], True])
        tops = j_[is_max][np.argsort(-vals[is_max])][:3]
        span = max(1e-6, 0.5 * float(np.min(np.diff(np.linspace(wlo, whi, ng)))))
        g, w = float(vals.max()), float(cands[int(vals.argmax())])
        for j in tops:
            try:
                s0_, d0_ = sigd(float(cands[j]))
                # xtol 1e-7 x band: sigma is flat at a peak (its error is
                # quadratic in the frequency error), ~5 fewer evaluations a climb
                wc_, gc_ = csp._climb_slope(sigd, float(cands[j]), s0_, d0_,
                                            max(wlo, cands[j] - span), min(whi, cands[j] + span),
                                            xtol=1e-7 * (whi - wlo))
                if gc_ > g:
                    g, w = float(gc_), float(wc_)
            except Exception:
                pass
        certified_norm.last_sampled = dict(points=len(cands), climbs=len(tops))
        certified_norm.last_all_w = None
        ok = True
    else:
        g = w = ok = None
    # the certificate needs a strictly positive lower bound; use the cheap
    # seeded search when the caller has none (first outer iteration)
    if g is not None:
        pass
    elif not np.isfinite(gam_lb) or gam_lb <= 0:
        ro0 = getattr(certified_norm, 'oracle', None)
        if ro0 is not None and ro0.dim > 0 and getattr(ro0, 'galerkin', False):
            # the position-robust reduced model's maximum (exact to ~1e-8):
            # 0.99 x it is a safe strictly positive lower bound, for free
            # (D.hinf cost ~12 full factorisations)
            rc0 = ro0.make_cl(b, c, D.r, D.mc)
            gam_lb = 0.99 * float(np.max(_sigmas(rc0, np.linspace(wlo, whi, 241), arow, nvec, K)))
        else:
            gam_lb = float(D.hinf(arow, nvec, K, b, c)[0])
    w_lb = None
    if g is None and getattr(certified_norm, 'true_lb', False):
        # BFS (2026-10-02): the caller's gam_lb is the inner problem's t, a
        # REDUCED-model value.  Where the model overestimates the full sigma_1,
        # the certificate proves nothing above t (1 + rtol) and returns t itself:
        # an upper bound that no frequency attains (test plant, cluster: 1209.568
        # certified, 1208.919 at the witness; the invariant failed in 2 of 3
        # rounds).  A TRUE lower bound instead, attained by the full model; the
        # certificate then answers within rtol of an attained value.
        w_lb, g_lb = _true_lower_bound(sig, sigd, sig_many, ro_, b, c, D, arow, nvec, K, wlo, whi, w_hint)
        if w_lb is not None:
            gam_lb = g_lb
    if g is None:
        g, w, ncall, ok = csp.hinf_certified_sparse(
            D.A, D.M, b, c, Ac, Bc, nvec, K, sig, gam_lb=max(gam_lb, 1e-8),
            wlo=wlo, whi=whi, rtol=float(getattr(certified_norm, 'rtol', 1e-4)), slope_of_w=sigd,
            sigma_many=sig_many)
        if not np.isfinite(w) and w_lb is not None and g == gam_lb:
            w = w_lb                       # certified at the first question: the lower bound's own frequency
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
                # ONLY OUTSIDE THE BAND (STAGE6 6.29): [wlo, whi] was just proved
                # free of sigma_1 = g (1 + rtol), the level this question asks at
                # first; a peak found outside climbs within its own interval, and
                # the rest stays proved at the lower level.
                g2, w2, nc2, ok2 = float(g), w, 0, True
                for (lo_, hi_) in ((-Wt, wlo), (whi, Wt)):
                    if hi_ - lo_ <= 1e-9:
                        continue
                    g_i, w_i, nc_i, ok_i = csp.hinf_certified_sparse(
                        D.A, D.M, b, c, Ac, Bc, nvec, K, sig, gam_lb=float(g),
                        wlo=lo_, whi=hi_, rtol=float(getattr(certified_norm, 'rtol', 1e-4)), slope_of_w=sigd,
        sigma_many=sig_many)
                    nc2 += nc_i; ok2 = ok2 and ok_i
                    if g_i > g2:
                        g2, w2 = g_i, w_i
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
    _lat_lev = None                    # the lattice reference sweep's level (its peaks then located at their tops)
    if getattr(certified_norm, 'sampled', 0) and not _peak_ref:
        ws, gs = cands, vals           # BFS sampled mode: its grid is the reference (no 241-point sweep)
    elif not _peak_ref and getattr(certified_norm, 'lattice', None) is not None:
        # --oracle lattice: the reference sweep on lattice points of 2^-SWEEP_LEVEL
        import lattice as _lt
        _lc = certified_norm.lattice
        _lat_lev = _lt.SWEEP_LEVEL if _lt.SWEEP_LEVEL > 0 else 5
        ws = _lc.grid(wlo, whi, _lat_lev)
        gs = _lc.sigmas(z, ws)
    elif not _peak_ref:
        ws = np.linspace(wlo, whi, nsw)
        gs = np.array(sig_many(ws) if sig_many else [sig(x) for x in ws])
    if ro_ is not None and ro_.dim > 0:
        rc_ = ro_.make_cl(b, c, D.r, D.mc)
        wf = np.linspace(wlo, whi, 1201)
        gf = _sigmas(rc_, wf, arow, nvec, K)
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
            gs = np.array(sig_many(ws) if sig_many else [sig(x) for x in ws])
            _pin_lus(D, list(ws) + list(peaks) + [w])        # the next enrichment's anchors (STAGE6 6.29)
    else:
        i = np.arange(1, len(gs) - 1)
        loc = i[(gs[i] >= gs[i - 1]) & (gs[i] >= gs[i + 1])]
        peaks = _enrol_peaks(ws, gs, None, rel=float(getattr(certified_norm, 'enroll_rel', 0.0)))
        if _lat_lev is not None and peaks:
            # AT THEIR TOPS on the lattice too (20261010j, missed-peaks.md P6): each
            # grid maximum located by the followed members' Newton (value and slope,
            # window one grid step either side, all side by side) -- the reduced
            # path's golden section of 2026-09-22 had been lost here, and members
            # were enrolled at grid points k/32 (4.03125 in 770632, whose top stood
            # 0.15 away when its window ran out a round later).  Tops within 1e-3 of
            # a higher one, or of the witness, are that peak.
            _pt0 = _lc.peak_tops(z, peaks, 2.0 ** -_lat_lev)
            _pt = sorted(_pt0, key=lambda q_: -q_[1])
            _kp = []
            for _x, _v in _pt:
                if all(abs(_x - q_) > 1e-3 for q_ in _kp) and not (w is not None and np.isfinite(w) and abs(_x - float(w)) <= 1e-3):
                    _kp.append(float(_x))
            print(f"    certificate peaks at their tops: {len(peaks)} grid maxima, moved by up to "
                  f"{max(abs(x_ - p_) for (x_, _), p_ in zip(_pt0, peaks)):.4f}; {len(_kp)} distinct "
                  f"(none within 1e-3 of the witness)", flush=True)
            peaks = _kp
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
                g3 = np.array(sig_many(W3) if sig_many else [sig(x) for x in W3]); j3 = int(np.argmax(g3))
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


def winding_veto(final, earlier, count_fn, line, cap=3, time_ok=None, log=print, tail_fn=None):
    """THE SAVE-TIME WINDING VETO's choice (20261010j stage B; the user's decision
    of 2026-10-14: the save-time count vetoes, never an unstable design as the
    record).  final: the ledger entry of the design about to be saved; earlier:
    this job's other accepted designs, oldest first, the anchor (its start) first
    of all; an entry is dict(round (-1: the anchor), z, g, watch, count) with
    count a dict(count, ok, ..., how) when a count right of `line` is known for
    it (the feasible start's, the anchor's at the start, one at acceptance),
    else None.  count_fn(entry) counts one (a dict with count, ok, npts, ...);
    tail_fn(entry), when given, is the count's instant tail test (no march:
    BFSPlant.tail_check), run first on every design to be counted.

    Every outcome is classified (2026-10-16): 'zero' (a conclusive 0),
    'positive' (a conclusive count > 0), 'inconclusive' (the march ran and could
    not close), 'pre-march' (no march at all: the tail test or the count raised
    winding_cert.TailBoundError, or the tail test failed otherwise) and 'error'
    (the count raised during or before its march for another reason).  A
    PRE-MARCH outcome costs nothing and proves nothing: it is never charged
    against `cap` (in 20261010n three instant tail failures, 0 pts each,
    exhausted --veto-counts 3 and every decentral arm fell back to its anchor,
    although about three real counts still fitted before the deadline).

    The final design stands when its count is a conclusive 0 (known, or counted
    here).  Otherwise the record falls back to the LATEST earlier design whose
    count is a conclusive 0: free where known, else counted -- at most `cap`
    charged counts, none once time_ok() says the deadline cannot hold one; a
    design whose tail test fails is passed over uncharged; the designs passed
    over are listed with the reason.  Returns (chosen entry or None, its count
    dict with 'source', the veto dict or None when the final design passed;
    veto['final']['outcome'] is the final design's outcome, which the save
    path's checkpoint rule reads); prints the '*** WINDING VETO' lines."""
    from winding_cert import TailBoundError

    def _known0(e_):
        c_ = e_.get('count')
        return bool(c_) and bool(c_.get('ok')) and c_.get('count') == 0

    def _err(_e):
        return f"{type(_e).__name__}: {str(_e)[:200]}"

    def _screen(e_):
        # the instant tail test: None when it passes (or there is none), else the
        # outcome of a design that cannot be counted (nothing marched).  Any failure
        # here is pre-march: the test is the count's own code before its march (the
        # ports, the plant's tail bound, the cutoff), so the count would fail alike
        if tail_fn is None:
            return None
        try:
            tail_fn(e_)
            return None
        except Exception as _e:
            return dict(count=None, ok=False, outcome='pre-march', error=_err(_e), npts=0, seconds=0.0)

    def _count(e_):
        t0_ = time.time()
        try:
            o_ = count_fn(e_)
        except Exception as _e:
            return dict(count=None, ok=False, outcome=('pre-march' if isinstance(_e, TailBoundError) else 'error'),
                        error=_err(_e), npts=0, seconds=round(time.time() - t0_, 1))
        cnt_ = None if o_.get('count') is None else int(o_['count'])
        ok_ = bool(o_.get('ok')) and cnt_ is not None
        return dict(count=cnt_, ok=ok_, outcome=(('zero' if cnt_ == 0 else 'positive') if ok_ else 'inconclusive'),
                    npts=int(o_.get('npts', 0) or 0), seconds=o_.get('seconds'),
                    wtail=float(o_.get('wtail', np.nan)), wstar=float(o_.get('wstar', np.nan)),
                    p_plant=int(o_.get('p_plant', -1)))

    def _desc(e_):
        return ("the anchor (this job's start)" if e_.get('round') == -1 else
                (f"round {e_['round']}'s design" if e_.get('round') is not None else "the final design")) + \
            f" (gamma {e_['g']:.6f})"

    def _what(c_):
        o_ = c_['outcome']
        return {'zero': f"0 right of Re={line:g}", 'positive': f"{c_['count']} right of Re={line:g}",
                'inconclusive': 'INCONCLUSIVE (the march could not close)',
                'pre-march': f"not counted, pre-march ({c_.get('error')}): not charged",
                'error': f"the count failed ({c_.get('error')})"}[o_]
    counted = []
    if _known0(final):
        return final, dict(final['count'], source=f"known: {final['count'].get('how')}"), None
    cf = _screen(final) or _count(final)
    counted.append(dict(cf, round=final.get('round'), gamma=float(final['g'])))
    if cf['outcome'] == 'zero':
        final['count'] = dict(cf, how='counted at the save')
        return final, dict(cf, source='counted at the save'), None
    log(f"*** WINDING VETO: {_desc(final)} has "
        + {'positive': f"{cf.get('count')} closed-loop eigenvalue(s)",
           'inconclusive': 'an INCONCLUSIVE count (the march could not close)',
           'pre-march': f"no count (pre-march, nothing marched: {cf.get('error')})",
           'error': f"a failed count ({cf.get('error')})"}[cf['outcome']]
        + f" right of Re={line:g} over all frequencies ({cf.get('npts', 0)} pts): it is not the record; "
        f"falling back to the last accepted design with a conclusive 0", flush=True)
    chosen, nc, skipped = None, 0, []
    for e_ in reversed(list(earlier)):
        if _known0(e_):
            chosen = e_
            break
        pre = _screen(e_)
        if pre is not None:
            skipped.append(dict(round=e_.get('round'), why=f"pre-march: {pre['error']}"))
            log(f"    veto: {_desc(e_)}: {_what(pre)}", flush=True)
            continue
        if nc >= int(cap):
            skipped.append(dict(round=e_.get('round'), why=f'cap of {cap} counts'))
            continue
        if time_ok is not None and not time_ok():
            skipped.append(dict(round=e_.get('round'), why='the deadline'))
            continue
        ce = _count(e_)
        nc += 0 if ce['outcome'] == 'pre-march' else 1
        counted.append(dict(ce, round=e_.get('round'), gamma=float(e_['g'])))
        log(f"    veto: {_desc(e_)}: {_what(ce)} ({ce.get('npts', 0)} pts, {ce.get('seconds')} s)", flush=True)
        if ce['outcome'] == 'zero':
            e_['count'] = dict(ce, how='counted at the save (veto)')
            chosen = e_
            break
    if skipped:
        log(f"    veto: {len(skipped)} earlier design(s) not counted (pre-march, at most {cap} charged counts, or the "
            f"deadline): {[(s_['round'], s_['why'][:40]) for s_ in skipped]}", flush=True)
    veto = dict(line=float(line),
                final=dict(round=final.get('round'), gamma=float(final['g']), outcome=cf['outcome'],
                           **({'error': cf['error']} if cf.get('error') else {})),
                counted=counted, skipped=skipped, cap=int(cap), charged=int(nc),
                fallback=(None if chosen is None else
                          dict(round=chosen.get('round'), gamma=float(chosen['g']), count_source=chosen['count'].get('how'))))
    if chosen is None:
        return None, None, veto
    log(f"*** WINDING VETO: the record is {_desc(chosen)}, not the final design (gamma {final['g']:.6f}); "
        f"its count right of Re={line:g}: 0 ({chosen['count'].get('how')})", flush=True)
    return chosen, dict(chosen['count'], source=f"veto fallback: {chosen['count'].get('how')}"), veto


# NO ACCEPTED DESIGN IS EVER LOST (BFS, 2026-10-16).  In 20261010n the five final
# decentral designs (367.38, 367.40, 366.49, 388.44, 390.82) existed in no file:
# the veto fell back to the anchor, the deadline stop rewrote the checkpoint with
# it, the record held the anchor and the JSON gammas only.  Now the save writes
# the job's final design to <tag>_final_<jobid>.npz/.json (never overwritten), and
# every acceptance appends the design to <tag>_accepted.npz.

def job_id():
    """The SLURM job id, else 'local<pid>' (the final file's name)."""
    return os.environ.get('SLURM_JOB_ID') or f"local{os.getpid()}"


def design_arrays(pk, z, mimo, loop_orders=None):
    """The small arrays of a design (d, e, f, nvec, K, xa, xs; MIMO also Bc and
    loop_orders; the atoms' va / vs where the ports have them), under the
    record's keys: what the record readers (count_rec-style tools) load."""
    aa, nn, KK, xa, xs, va, vs = pk.unpack(np.asarray(z, float))
    out = dict(d=np.asarray(aa[0]), e=np.asarray(aa[1]), f=np.asarray(aa[2]), nvec=np.asarray(nn),
               K=np.asarray(KK), xa=np.asarray(xa), xs=np.asarray(xs))
    if mimo:
        out.update(Bc=np.asarray(aa[3]), loop_orders=np.asarray(loop_orders))
    out.update({k_: np.asarray(v_) for k_, v_ in (('va', va), ('vs', vs)) if v_ is not None})
    return out


def write_final_design(base, final, record, veto, arrays_of, extra=None, jobid=None, log=print):
    """<base>_final_<jobid>.npz/.json: the job's final design (the last accepted,
    the one the veto judged) under the plain record keys, so it reads like a
    record; when the veto saved another design as the record, that one under
    'record_' keys.  final / record: ledger entries (z, g, round, al); veto: the
    veto dict or None; arrays_of(z): design_arrays; extra: further npz arrays
    (the loop state, the working set).  A name that exists already gets _2, _3,
    ...: nothing overwrites it.  Returns the npz path (None if the write failed,
    which is logged and never fatal)."""
    jobid = jobid or job_id()
    try:
        stem = f"{base}_final_{jobid}"
        k = 1
        while os.path.exists(stem + '.npz') or os.path.exists(stem + '.json'):
            k += 1
            stem = f"{base}_final_{jobid}_{k}"
        arr = dict(arrays_of(final['z']))
        arr.update(gamma=float(final['g']), alpha=float(final.get('al', np.nan)),
                   round=int(-2 if final.get('round') is None else final['round']))
        same = record is not None and np.array_equal(np.asarray(record['z']), np.asarray(final['z']))
        if record is not None and not same:
            arr.update({'record_' + k_: v_ for k_, v_ in arrays_of(record['z']).items()})
            arr.update(record_gamma=float(record['g']), record_round=int(-2 if record.get('round') is None else record['round']))
        arr.update(dict(extra or {}))
        arr['veto'] = np.array(json.dumps(_js(veto)) if veto is not None else '')
        np.savez(stem + '.npz', **arr)
        json.dump(_js(dict(job=str(jobid), gamma=float(final['g']), round=final.get('round'),
                           record_design=('the final design' if (record is not None and same) else
                                            (f"round {record.get('round')} (gamma {record['g']:.6f}), the veto's fallback"
                                             if record is not None else 'none (NOT SAVING)')),
                           veto=veto, code_version=_code_version())),
                  open(stem + '.json', 'w'), indent=1)
        log(f"final design: {os.path.basename(stem)}.npz (gamma {final['g']:.6f}"
            + ('' if record is None or same else f"; the record: round {record.get('round')}, gamma {record['g']:.6f}")
            + ')', flush=True)
        return stem + '.npz'
    except Exception as _e:
        log(f"    final-design write failed: {_e!r}", flush=True)
        return None


def append_accepted(path, arrays, meta, log=print):
    """Append one accepted design to the ledger `path` (npz): its arrays as
    a<k>_<key>, the index arrays n, gamma, round, job beside them; meta: gamma,
    round, job (and alpha, t).  Rewritten whole (a few kB per design) through a
    temporary file and os.replace, so an interruption leaves the previous ledger.
    Returns the new length (None if the write failed: logged, never fatal)."""
    try:
        old = {}
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as zz:
                old = {k_: zz[k_] for k_ in zz.files}
        n0 = int(old.get('n', 0))
        out = {k_: v_ for k_, v_ in old.items() if k_ not in ('n', 'gamma', 'round', 'job')}
        for k_, v_ in dict(arrays, **meta).items():
            out[f"a{n0:04d}_{k_}"] = np.asarray(v_)
        out['n'] = np.array(n0 + 1)
        out['gamma'] = np.append(np.asarray(old.get('gamma', np.zeros(0)), float), float(meta['gamma']))
        out['round'] = np.append(np.asarray(old.get('round', np.zeros(0)), int), int(meta['round']))
        out['job'] = np.append(np.asarray(old.get('job', np.zeros(0, dtype='<U1')), str), str(meta['job']))
        tmp = path[:-4] + '.tmp.npz'
        np.savez(tmp, **out)
        os.replace(tmp, path)
        return n0 + 1
    except Exception as _e:
        log(f"    accepted-design ledger write failed: {_e!r}", flush=True)
        return None


def ckpt_after_veto(veto, final, chosen, anchor):
    """The design a checkpoint may hold once the save-time veto has run
    (2026-10-16), and why: the final design when it passed (veto None) or when
    its veto was PRE-MARCH (nothing was counted, so nothing is known against it:
    the next job's start count decides, FATAL on a conclusive count > 0); else
    the veto's fallback; else (NOT SAVING) the anchor, the job's start; else
    None (no design may stay).  Never a design whose march came back
    INCONCLUSIVE or with a count > 0 (20261010j stage B: the next job must not
    resume from a vetoed design; the NOT SAVING path used to leave it there).
    The save applies it right after the veto (2026-10-17), before anything that
    can end the save early."""
    if veto is None:
        return final, 'the final design'
    if veto['final'].get('outcome') == 'pre-march':
        return final, 'the final design (the veto could not count it: pre-march)'
    if chosen is not None:
        return chosen, "the winding veto's fallback"
    if anchor is not None:
        return anchor, "the anchor, this job's start (no design of the job has a conclusive 0)"
    return None, 'none (the veto found no conclusive 0)'


def _atoms_record_keys(D, apa, aps):
    """The atoms in natural form (Re a, Im a, c, width, wavenumber) and the
    bounds they were made with, for a record (STAGE6 6.35): its (rho, tau)
    alone mean different fields under different bounds."""
    import fieldbasis as _fb
    out = {}
    for side, P in (('a', apa), ('s', aps)):
        A = getattr(D, 'atoms_' + side, None)
        if P is None or A is None:
            continue
        rows = [np.asarray(r, float) for r in (P if isinstance(P, (list, tuple)) else [P])]
        Q = np.array([_fb.atoms_to_natural(r, A['n'], A['smin'], A['kmax']) for r in rows])
        out[f'atoms_{side}_nat'] = Q if isinstance(P, (list, tuple)) else Q[0]
        out[f'atoms_{side}_bounds'] = np.array([A['smin'], A['kmax']])
    return out


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
            # a start without a certified gamma (bfs/hifoo_init.py writes inf; older
            # starts nan) never beats the run's own checkpoint: with nan the comparison
            # was False and a MIMO resume restarted from the start (2026-10-03)
            if g0 is None or not np.isfinite(g0) or gck < g0:
                print(f"CHECKPOINT found (gamma={gck:.6f}) -- resuming from {ck}", flush=True)
                src, g0 = ck, gck
                a.start_from_ckpt = True          # the anchor is inherited: counted at the start (G4, 20261010j)
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
            P, how = _fb.atoms_from_record(z, key, a.field_atoms, a.atom_smin, a.atom_kmax, D.xn)
            if P is not None and len(P) == cnt:
                p0 = [np.asarray(row, float) for row in P]; how = 'from the record, ' + how
            else:
                p0 = [_fb.fit_atoms(D.xn, v_, D.M, a.field_atoms, a.atom_smin, a.atom_kmax)[0] for v_ in v0_]
                how = 'fitted to the start fields' + (f' ({how})' if P is None and how != 'no atoms in the record' else '')
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
    ap.add_argument('--deadline', type=float, default=0.0,
                    help='seconds: the job time limit (0 = off; a resume restarts the clock).  --inner sqp: the SQP '
                         'stops (code 4) when another iteration would leave less than the round reserve '
                         '(--deadline-reserve) + --deadline-final, and that round ends the job as a deadline stop; no '
                         'new round once elapsed + its setup + that reserve + a minimum inner budget '
                         '(--deadline-min-inner) + --deadline-final would pass it (20261010j stage B; was, and still is '
                         'for pyGRANSO: 1.25 x the longest round, which ended the four MIMO jobs of 20261010g 1.2-2.3 h '
                         'early)')
    ap.add_argument('--deadline-final', dest='deadline_final', type=float, default=1800.0,
                    help='seconds reserved after the last round for the final certificate, the save-time winding count '
                         '(and its veto\'s counts, BFS) and the save (--deadline)')
    ap.add_argument('--deadline-reserve', dest='deadline_reserve', type=float, default=900.0,
                    help='--deadline: seconds kept after a time-truncated inner solve for its own tail and the '
                         'round\'s certificate, gate and checkpoint (at least; 1.25 x the longest such phase seen, plus '
                         'the longest winding count with --cont-accept-winding, when that is more)')
    ap.add_argument('--deadline-min-inner', dest='deadline_min_inner', type=float, default=600.0,
                    help='--deadline: no round starts with less inner-solve time than this (at least 3 SQP iterations)')
    ap.add_argument('--veto-counts', dest='veto_counts', type=int, default=3,
                    help='--plant bfs: the save-time winding veto counts at most this many earlier accepted designs '
                         'when the final one is not a conclusive 0 right of -eps (designs with a known count are free)')
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
    ap.add_argument('--plant', choices=['gl', 'bfs'], default='gl',
                    help='gl: the Ginzburg-Landau mesh (--ne); bfs: the backward-facing step from --ops (BFS.md)')
    ap.add_argument('--ops', default=None, help='--plant bfs: the operator file (bfs/plant_io.py)')
    ap.add_argument('--bfs-sampled', dest='bfs_sampled', type=int, default=0,
                    help='--plant bfs: 0 = gamma CERTIFIED on the band by the level-set disk sweep (option A, BFS.md 4); '
                         'N > 0 = the sampled maximum of a full-model grid of N points + the reduced model\'s peaks, '
                         'climbed (not a certificate)')
    ap.add_argument('--bfs-margin', dest='bfs_margin', type=float, default=0.05,
                    help='--plant bfs: the actuator centre stays this far inside the domain (box sides, top, the lower '
                         'wall\'s tangent line) and the sensor window this far from the outlet; hard SQP rows (BFS.md 2)')
    ap.add_argument('--bfs-act-comp', dest='bfs_act_comp', choices=['v', 'u'], default='v',
                    help='--plant bfs: the momentum component the actuator forces (v: Barbagallo et al. 2012; u: the '
                         'user\'s actuator_sensor.edp)')
    ap.add_argument('--bfs-band', dest='bfs_band', type=float, default=10.0,
                    help='--plant bfs: gamma is certified on [0, this] (real plant: sigma(-w) = sigma(w); b = 10, '
                         'user 2026-10-02), widened to cover the controller poles (BFS.md 4)')
    ap.add_argument('--lu-order', dest='lu_order', choices=['colamd', 'mmd_at_plus_a', 'mumps'], default='colamd',
                    help='column ordering of every full-model sparse LU (sparse_lu.py): colamd = scipy default '
                         '(bit-identical frozen path); mmd_at_plus_a = symmetric minimum degree for 2-D/3-D FE '
                         'matrices (STAGE6 6.29); mumps = PETSc/MUMPS (--plant bfs, BFS.md 3)')
    ap.add_argument('--resume-guard', dest='resume_guard', type=float, default=10.0,
                    help='refuse a resumed start whose certified gamma exceeds this multiple of the source '
                         'record\'s (a failed transplant); 0 = off -- for starts that are poor by construction, '
                         'e.g. fields fitted by few atoms with the controller held (STAGE6 6.34)')
    ap.add_argument('--field-input', dest='field_input', choices=['modes', 'tangent'], default='modes',
                    help='atom fields (--field-atoms) with --oracle-inputs on: the reduced model\'s input basis for '
                         'the fields.  modes = the --field-basis modes (exact for any field in their span; 160 per '
                         'field at ne=200, kd ~300); tangent = the current atom fields and their derivatives in '
                         'centre, width and wavenumber, whose complex span is phi, (x-c) phi, (x-c)^2 phi per atom '
                         '(3 x n_atoms columns per field), rebuilt at every round (STAGE6 6.33)')
    ap.add_argument('--oracle-orth', dest='oracle_orth', choices=['block', 'column'], default='block',
                    help='orthogonalisation of the reduced basis (reduced.py): block = BLAS3 passes against the '
                         'basis (default, STAGE6 6.29); column = the reference column-by-column version, for the '
                         'SISO agreement check against the siso-v1 tree')
    ap.add_argument('--oracle-union', dest='oracle_union', choices=['full', 'new'], default='full',
                    help="union Galerkin step of the reduced basis: full = W's columns all projected against V "
                         "(default, bit-identical); new = only W's new columns (V = W before the step): O(n kd k_new) "
                         "instead of O(n kd^2) per anchor, the serial part of the enrichment at the cavity; same span, "
                         "rounding-level different basis (STAGE6 6.30)")
    ap.add_argument('--sqp-pen', dest='sqp_pen', choices=['fixed', 'adaptive'], default='fixed',
                    help='SQP merit penalty and elastic slack costs: fixed = 100 / 1e3 (GL); adaptive = steered '
                         'and raised above the stability multipliers (exact penalty; run_bfs.sh; BFS.md 5)')
    ap.add_argument('--sqp-verbose', dest='verbose', action='store_true',
                    help='print every SQP iteration: t, smax, stationarity, step, radius, accepted (diagnostics)')
    ap.add_argument('--row-workers', dest='row_workers', default='0',
                    help='reduced-model rows (gain_batch) evaluated in this many FORKED processes besides this one, '
                         'frequencies split by w-blocks (reduced._RowPool; STAGE6 6.26). 0 = serial; '
                         'mpi = on the ranks of the MPI pool, the driver included (reduced._MPIRowPool; BFS.md 5)')
    ap.add_argument('--oracle', choices=['full', 'reduced', 'lattice'], default='full',
                    help='STAGE 5: sigma rows of the inner solve on the full oracle '
                         '(frozen) or on the projected one (reduced.py); lattice (BFS only, 2026-10-06): the full '
                         'model on the MPI ranks, its factors cached on a dyadic lattice of frequencies and line '
                         'shifts (lattice.py, BFS.md 5)')
    ap.add_argument('--lattice-cap', dest='lattice_cap', type=int, default=24,
                    help='--oracle lattice: plant factorisations kept per rank (least recently used dropped); '
                         'test plant 133 MB each, full mesh ~1.05 GB')
    ap.add_argument('--lattice-tol', dest='lattice_tol', type=float, default=1e-7,
                    help='--oracle lattice: a followed peak\'s Newton stops at |sigma\'| <= this x sigma '
                         '(lattice.FOLLOW_GTOL)')
    ap.add_argument('--lattice-sweep', dest='lattice_sweep', type=int, default=5,
                    help='--oracle lattice: the certificate\'s reference sweep on the lattice level 2^-N of the band '
                         '(the in-solve missed peaks are the level-set certificate, lattice.LatticeOracle.level)')
    ap.add_argument('--ctrl-modal', dest='ctrl_modal', choices=['on', 'off'], default='off',
                    help='the compensator in real 2 x 2 diagonal blocks (controller.modal_*), its stability as two '
                         'smooth exact rows per block (trace, determinant) -- merging poles stay smooth (BFS.md 5); '
                         'off: the tridiagonal form (GL, older records)')
    ap.add_argument('--sqp-debug-ls', dest='sqp_debug_ls', choices=['on', 'off'], default='off',
                    help='print every line-search trial (merit components, the max row against its model) '
                         '(sqp_inner.DEBUG_LS; diagnostics)')
    ap.add_argument('--lattice-cross', dest='lattice_cross', choices=['merit', 'cut'], default='merit',
                    help='--oracle lattice: a trial across the closed loop\'s line from an admissible iterate is weighed by '
                         'the merit (merit; only admissible iterates become the result) or halved (cut, as 2026-10-06 to '
                         '20261010c) (sqp_inner.LAT_CROSS)')
    ap.add_argument('--lattice-reach', dest='lattice_reach', type=float, default=0.25,
                    help='--oracle lattice: stability rows for the closed-loop eigenvalues within this of the line')
    ap.add_argument('--lattice-k', dest='lattice_k', type=int, default=1,
                    help='--oracle lattice: nearest eigenvalues computed at each disk centre of the line')
    ap.add_argument('--lattice-warm', dest='lattice_warm', type=int, default=0,
                    help='--oracle lattice: > 0 restarts sigma_1 from the last vector at the same point with this '
                         'Arnoldi subspace (4: about half the solves); 0 (default): cold, as the serial code -- a warm '
                         'start can land on sigma_2 where the two swap')
    ap.add_argument('--lattice-driver', dest='lattice_driver', choices=['on', 'off'], default='on',
                    help='--oracle lattice: the driver (rank 0) computes a share of every batch')
    # 20261010j stage C (eval-waves.md P1, P7, P8; packing.md P2; strip-ab.md AB3)
    ap.add_argument('--lattice-base', dest='lattice_base', type=int, default=48,
                    help='--oracle lattice: the line\'s and the level set\'s base grid, this many equispaced centres '
                         'whatever the number of ranks (lattice.LINE_BASE; the same as ranks/2 at 96 ranks); 0 = ranks/2 '
                         '(the rule of 2026-10-09: 24 centres at 48 ranks)')
    ap.add_argument('--lattice-costs', dest='lattice_costs', choices=['measured', 'static'], default='static',
                    help='--oracle lattice: the batches balanced on the static units in request order (static, the default, '
                         'as since 2026-10-08), or on the MUMPS solves each kind of request took, a factorisation '
                         'lattice.LU_SOLVES solves, longest first (measured; counts, not seconds, so that every run of a '
                         'job places alike; lattice.TASK_COSTS).  Placement only, but a task\'s result depends on its rank '
                         'at rounding level (driver vs worker 7e-13): another placement is another MIMO path')
    ap.add_argument('--lattice-memo', dest='lattice_memo', type=int, default=2,
                    help='--oracle lattice: designs whose row requests are kept for the next evaluation of the same design '
                         '(the refine\'s re-evaluation; lattice.REQ_MEMO); 0 = off')
    ap.add_argument('--lattice-row-merge', dest='lattice_row_merge', type=float, default=0.0,
                    help='--oracle lattice: > 1e-8: stability rows whose eigenvalues agree within this x max(1, |lambda|) '
                         'are one row, the copy from the nearest shift kept (lattice.ROW_MERGE; strip-ab.md AB3, e.g. '
                         '1e-6; changes the QP\'s rows, hence the path); <= 1e-8 (default 0): 1e-8 and the first copy, as '
                         'before, the copies within lattice.ROW_DUP counted and tagged in the QP rows\' line')
    # 20261010j stage D (level-set-waves.md LS3, eval-waves.md P5, driver-serial.md P3); on by default since
    # 2026-10-16, measured in 20261010n (BFS.md 8)
    ap.add_argument('--lattice-async', dest='lattice_async', choices=['off', 'on', 'redo'], default='on',
                    help='--oracle lattice: the missed-peak level set after an accepted step runs BESIDE the next '
                         'iteration (on, the default since 2026-10-16: its disks on the worker ranks, never the driver, '
                         'while the next QP and line-search trials run on the others; the look-ahead kept when the call '
                         'adds no member, else put back and the synchronous continuation taken -- the same iterates: '
                         'identical traces over 8 central and 11 decentral rounds on the cluster, 0.71-0.80 x the round '
                         'wall, 20261010n); redo: every look-ahead put back and redone (the restore\'s test); off: the '
                         'synchronous call, for measurements (sqp_inner.LAT_ASYNC)')
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
    ap.add_argument('--cert-near-k', dest='cert_near_k', type=int, default=8,
                    help='eigenvalues asked at each certificate decision shift (only the nearest sizes the disk; '
                         '8 = the frozen value, the nominal scripts use 2: 5-9x fewer solves, STAGE6 6.29)')
    ap.add_argument('--cert-split', dest='cert_split', type=int, default=1,
                    help='certificate decision: start from this many equal critical intervals (1 = the serial '
                         'algorithm, bit for bit at any --nproc); ~--nproc widens the first generations for the '
                         'full-model pool: other shifts, the same rigorous decision (STAGE6 6.30)')
    ap.add_argument('--cert-climb', dest='cert_climb', choices=['brent', 'slope'], default='brent',
                    help="certificate witness climb: brent (frozen, derivative-free to 1e-11) or slope (the "
                         "analytic g', ~10 evaluations instead of ~30-40; STAGE6 6.29)")
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
    ap.add_argument('--omega-retire', dest='omega_retire', default=None, metavar='FRAC:ROUNDS',
                    help='--oracle lattice (2026-10-08): a member of Omega whose sigma_1 stays below FRAC x gamma at '
                         'the accepted design for ROUNDS consecutive accepted rounds retires (never the certified '
                         'peaks, the sigma_2-enrolled members, a recent refuter, the highest member), e.g. 0.95:3.  '
                         'Off by default')
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
                    help='compensator stability as a Lyapunov LMI in the SQP subproblem (r x r, any plant size); off = one row per pole; '
                         'never with --ctrl-modal (its two exact rows per block)')
    ap.add_argument('--sqp-partial-shrink', dest='sqp_partial_shrink', choices=['on', 'off'], default='off',
                    help='after a step accepted at alpha < 1 the trust radius becomes 2 alpha |d| (sqp_inner.'
                         'PARTIAL_SHRINK; off = the frozen path, where only the double-pole face does this)')
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
    ap.add_argument('--post', type=int, default=0,
                    help='BFS: 1 = the figures and post.npz (bfs/postproc.py) at the end, on the same pool')
    ap.add_argument('--continue-record', dest='continue_record', type=int, default=0,
                    help='1: a finished run in OUTDIR (record, no checkpoint) continues from its record for '
                         '--outer more rounds (BFS.md 7)')
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
                         'auto = on for MIMO and for cases with a free field (FB/FC), off otherwise (STAGE6 6.34)')
    ap.add_argument('--n-act', dest='n_act', type=int, default=1,
                    help='MIMO (STAGE6): number of actuators (with --n-sens > 1 or > 1 itself: the MIMO path)')
    ap.add_argument('--n-sens', dest='n_sens', type=int, default=1, help='MIMO: number of sensors')
    # (%% in the help: argparse formats it, and the unescaped '0.08 % higher' made --help raise TypeError until 2026-10-16)
    ap.add_argument('--sqp-stat', dest='sqp_stat', choices=['auto', 'pred', 'rel'], default='auto',
                    help="SQP stationarity: 'rel' stat < 1e-4 x the active rows' gradient scale (to 2026-09-29); "
                         "'pred' also requires the QP's predicted decrease <= 1e-7 t (STAGE6 6.14); "
                         "'auto': pred for MIMO, rel for SISO (the ne=100 K+B+C A/B: 6.404012 in 22.2 min vs "
                         "6.399134 in 22.6 -- faster early, stopped 0.08 %% higher; SISO adoption waits for the cluster).  "
                         "The BFS passes pred for SISO too (bfs/run_bfs.sh, 2026-10-16: rel fired code 0 at the SISO records "
                         "with a certified step still gaining 0.20 %%)")
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
    ap.add_argument('--dense-audit', dest='dense_audit', action='store_true',
                    help='DIAGNOSTIC (STAGE6 6.34): a dense eigensolve of every round\'s candidate, printed with '
                         'the gate\'s decision -- to locate where a design escaped the matrix-free chain.  '
                         'Check only: never gates and never feeds the optimiser')
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
    ap.add_argument('--gate-path', dest='gate_path', choices=['auto', 'on', 'off'], default='auto',
                    help='continuation gate along the SQP\'s accepted iterates (each certified on the full model with '
                         '--oracle lattice) instead of the straight segment from the last accepted design; the straight '
                         'segment if the path fails.  auto = on with --oracle lattice (BFS.md 5)')
    ap.add_argument('--gate-prefix', dest='gate_prefix', type=int, default=0,
                    help='when the continuation gate refuses a step, try the end of its certified prefix as the '
                         'candidate (line + gamma certificates) instead of losing the round (BFS.md 5)')
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
    if any(v_ == '--lattice-strip' or v_.startswith('--lattice-strip=') for v_ in sys.argv[1:]):
        # removed 2026-10-16 (user, after the 20261010n A/B): a clear refusal, not argparse's 'unrecognized'
        raise SystemExit("--lattice-strip: removed on 2026-10-16 -- the lattice oracle runs without the strip, as "
                         "--lattice-strip off did (BFS.md 8: the A/B of 20261010n); drop the flag")
    a = ap.parse_args()
    if a.plant == 'bfs':
        # MPI job (BFS.md 3): ranks > 0 are the full-model pool's workers
        from bfs import mpipool as _mpp
        if _mpp.active() and _mpp.comm().Get_rank() > 0:
            if a.lu_order:
                import sparse_lu; sparse_lu.ORDER = a.lu_order
            _mpp.serve()
    if a.oracle == 'lattice':
        if a.plant != 'bfs':
            raise SystemExit("--oracle lattice: backward-facing step only (real plant, lattice.py)")
        import lattice as _lat_mod                # its settings go to the ranks with every context (mpipool)
        _lat_mod.LU_CAP = int(a.lattice_cap); _lat_mod.FOLLOW_GTOL = float(a.lattice_tol)
        _lat_mod.SWEEP_LEVEL = int(a.lattice_sweep); _lat_mod.ROW_REACH = float(a.lattice_reach)
        _lat_mod.LINE_K = int(a.lattice_k); _lat_mod.WARM_NCV = int(a.lattice_warm)
        _lat_mod.LINE_BASE = int(a.lattice_base); _lat_mod.TASK_COSTS = a.lattice_costs
        _lat_mod.REQ_MEMO = max(int(a.lattice_memo), 0); _lat_mod.ROW_MERGE = float(a.lattice_row_merge)
    _retire_frac = _retire_rounds = None
    if a.omega_retire:
        if a.oracle != 'lattice':
            raise SystemExit("--omega-retire: --oracle lattice only (sigma_1 on the full model)")
        _rf, _rr = a.omega_retire.split(':')
        _retire_frac, _retire_rounds = float(_rf), int(_rr)
        if not (0.0 < _retire_frac < 1.0 and _retire_rounds >= 1):
            raise SystemExit("--omega-retire FRAC:ROUNDS with 0 < FRAC < 1 and ROUNDS >= 1")
    certified_norm.rtol = float(a.cert_rtol)      # BEFORE the start certification: one tolerance for every certified value of a run
    global REDUCED_CERT
    REDUCED_CERT = a.reduced_cert
    import sqp_inner as _sqi; _sqi.REDUCED_CERT = a.reduced_cert; _sqi.FOLLOW_SCAN = a.follow_scan; _sqi.PAIR_MANIFOLD = (a.sqp_pair_manifold == 'on'); _sqi.FIELD_KMAX2 = (a.field_kmax ** 2 if a.field_kmax else None); _sqi.FIELD_ORDER = a.field_order
    _sqi.PARTIAL_SHRINK = (a.sqp_partial_shrink == 'on')
    _sqi.LAT_CROSS = a.lattice_cross; _sqi.DEBUG_LS = (a.sqp_debug_ls == 'on')
    _sqi.LAT_ASYNC = a.lattice_async
    if a.oracle == 'lattice':
        _sqi.QP_DUMP = a.outdir
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
    # 'auto': MIMO (STAGE6), and every case with a free field (6.34: SISO
    # K+FB+FC with 2 atoms, ne=200 -- round 0 accepted a design with an
    # eigenvalue at +51.9 - 20.1i, the gate's watch list never saw it, and the
    # whole chain after it was unstable; caught only by the dense save veto)
    _wind_accept = (a.wind_accept == 'on') or (a.wind_accept == 'auto' and
                                                (MIMO or 'FB' in a.case.split('+') or 'FC' in a.case.split('+')))
    n_wind_accept, n_wind_refused, n_gate_wind = [0], [0], [0]
    n_wind_premarch = [0]               # counts that could not run (winding_cert.TailBoundError), 2026-10-16
    import winding_cert as _wc_mod      # (its TailBoundError at the count sites)
    _wind_sec = [0.0]                   # the longest BFS count of this job (s): the deadline's round reserve, the veto's guard

    def _wind_right_of_margin(bb_, cc_, aa_, nn_, KK_, seen_, pa_=None):
        """Exact count of closed-loop eigenvalues right of Re = -eps (argument
        principle, global in w).  None when the plant pole data is unavailable
        (n above the dense limit).  The line certificate just proved nothing lies
        ON Re = -eps, which is what the argument principle needs there, so count
        0 <=> alpha <= -eps, unconditionally."""
        import winding_cert as _wc
        _line = -a.eps
        if BFS:
            # the M-free count (BFSPlant.winding: certified plant data, the smaller of
            # the H^1 and L2 tails, 2026-10-16),
            # with the ABSOLUTE step floor (20261010j stage B, gate-refusal.md G1):
            # the designs sit on the rows' target, ~1e-4 left of this line
            _zh = []
            for x in (seen_ or []):
                x = complex(x)
                if abs(x.real - _line) < 0.3 and all(abs(x - q) > 1e-10 for q in _zh):
                    _zh.append(x)
            _t0w = time.time()
            _o = D.winding(pa_, bb_, cc_, aa_, nn_, KK_, _line, zeros_hint=_zh, hmin_abs=WIND_HMIN_ABS)
            _o['seconds'] = round(time.time() - _t0w, 1); _o['line'] = float(_line)
            _wind_sec[0] = max(_wind_sec[0], _o['seconds'])
            return _o
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
    tag = os.path.join(a.outdir, f"{a.case.replace('+','_')}_s{a.start}"
                       + ('_bfs' if a.plant == 'bfs' else f"_ne{a.ne}")
                       + (f"_mimo{a.n_act}x{a.n_sens}_{a.ctrl_struct}" if MIMO else ''))
    t_start = time.time()
    a.start_from_ckpt = False                     # set where the start design is this run's checkpoint (G4 below)

    def _keep_part(why):
        """tag.json -> the next free tag_partN.json (its round history would
        otherwise be overwritten at this run's end), with a 'part' block: why,
        the code that wrote it and the code continuing it (20261010j stage B).
        Not again when the newest part already holds that summary (a
        resubmission after a job that saved nothing: strip-ab.md AB2, the
        cancelled restarts 770564-770623).  Returns N, or None (no summary)."""
        if not os.path.exists(tag + '.json'):
            return None
        _k = 1
        while os.path.exists(f"{tag}_part{_k}.json"):
            _k += 1
        try:
            _cur = json.load(open(tag + '.json'))
        except Exception:
            _cur = None
        if _k > 1 and _cur is not None:
            try:
                _last = json.load(open(f"{tag}_part{_k - 1}.json")); _last.pop('part', None)
                if json.dumps(_last, sort_keys=True) == json.dumps(_cur, sort_keys=True):
                    print(f"  (the summary is already kept as {os.path.basename(tag)}_part{_k - 1}.json)", flush=True)
                    return _k - 1
            except Exception:
                pass
        if _cur is None:
            import shutil                             # unreadable as JSON: kept byte for byte
            shutil.copyfile(tag + '.json', f"{tag}_part{_k}.json")
        else:
            _cur['part'] = dict(n=_k, why=why, written_by=_cur.get('code_version') or 'a version that did not record it (before 20261010j)',
                                continued_by=_code_version(), kept=time.strftime('%Y-%m-%dT%H:%M:%S'))
            json.dump(_cur, open(f"{tag}_part{_k}.json", 'w'), indent=1)
        return _k
    if a.continue_record and not os.path.exists(tag + '_ckpt.npz') and os.path.exists(tag + '.npz'):
        # CONTINUE A FINISHED RUN (--continue-record; BFS.md 7, 2026-10-02): its
        # record becomes the checkpoint, so the run resumes from the saved design
        # with its working set (the record's active frequencies) and, since
        # 20261010j stage B, its loop state (inner budget, SQP trust radius, peak
        # members, tau cap: the record now carries them, omega-growth.md P6),
        # through the same anchor verification as any resume.  The finished run's
        # summary is kept as tag_partN.json.
        import shutil
        _k = _keep_part('continued from the finished record')
        shutil.copyfile(tag + '.npz', tag + '_ckpt.npz')
        _gr = float(np.load(tag + '.npz', allow_pickle=True)['gamma'])
        try:
            _wb = json.load(open(tag + '.json')).get('code_version')
        except Exception:
            _wb = None
        json.dump(dict(case=a.case, gamma=_gr, note=f'continued from the finished record (part {_k})',
                       code_version=_wb),                # None: the record predates the field
                  open(tag + '_ckpt.json', 'w'))
        print(f"CONTINUE: the finished record (gamma={_gr:.6f}) is the checkpoint; "
              + (f"its summary kept as {os.path.basename(tag)}_part{_k}.json" if _k else "it has no summary to keep"),
              flush=True)
    elif os.path.exists(tag + '_ckpt.npz') and os.path.exists(tag + '.json'):
        # A DEADLINE-STOPPED RUN resumed (its checkpoint kept, its summary written;
        # strip-ab.md AB2): the summary of the job that stopped is kept as well --
        # only the finished-record branch above kept it, and the next job's end
        # overwrote it (the 20261010e central strip summary was lost that way)
        _k = _keep_part('a deadline-stopped run resumed from its checkpoint')
        if _k is not None:
            print(f"RESUME: the stopped job's summary is {os.path.basename(tag)}_part{_k}.json", flush=True)

    # The controller order is set by the SEED CATALOGUE (the common order from
    # the Hankel shoulder), not by --r.  Enlarging the catalogue moved it from
    # 7 to 8, and a stale --r 7 built the worker Packers with the wrong layout:
    # "operands could not be broadcast together with shapes (27,) (30,)", i.e.
    # 4r-1 for r = 7 against r = 8.
    try:
        if a.plant == 'bfs':
            raise LookupError('the BFS start sets the order (--r)')
        _sd = np.load(a.seed_file or f"seeds_ne{a.ne}.npz", allow_pickle=True)
        _r = int(_sd['r'])
        if _r != a.r:
            print(f"controller order {a.r} -> {_r} (from the seed catalogue)",
                  flush=True)
            a.r = _r
    except Exception:
        pass
    import sparse_lu; sparse_lu.ORDER = a.lu_order
    import reduced as _reduced_mod; _reduced_mod.ORTH = a.oracle_orth; _reduced_mod.UNION = a.oracle_union
    BFS = a.plant == 'bfs'
    if BFS:
        # the backward-facing step (BFS.md): its M is the velocity mass,
        # singular, never factorised (no massops warm-up)
        from bfs.plant import BFSPlant
        D = BFSPlant(a.ops, r=a.r, act_comp=a.bfs_act_comp); D.pos_margin = float(a.bfs_margin)
        print(f"BFS plant: {a.ops}, n = {D.n}, Re = {D.Re:g}, ports: Gaussian {D.act_comp}-forcing (sig {D.sig_act:.4f}), "
              f"floor shear window (L {D.sensor_len}, edge {D.sensor_edge}), "
              f"LU backend {a.lu_order}", flush=True)
    else:
        D = FECoDesign(ne=a.ne, r=a.r, sob_len=a.sob_len)
        import massops; massops.warmup(D.M)          # the mass matrix, factorised ONCE (STAGE6 6.29)
    # THE FULL-MODEL POOL (fullpar, STAGE6 6.30).  Take the whole allocation
    # unless told otherwise.  Forked here, after the plant and its mass LU, so
    # the workers hold them; re-forked before the rounds with the reduced
    # oracle's input bases.  --nproc 1: no pool, every task in this process
    # (the serial code, bit for bit).
    if a.nproc < 0:
        try:
            from parallel import ParallelOracle as _PO
            a.nproc = _PO.available_cores()
        except Exception:
            a.nproc = 1
    import fullpar
    csp.register_plant(D.A, D.M)          # the continuation gate's tracked queries carry a tag (STAGE6 6.29)
    fullpar.start(a.nproc, D=D)
    if fullpar._MPI is not None:
        a.nproc = fullpar.NPROC              # MPI job: the ranks are the pool (BFS.md 3)
        print(f"full-model pool: {fullpar.NPROC - 1} MPI worker ranks (bfs/mpipool.py)", flush=True)
    if BFS and a.nproc > 1 and a.cert_nseg > 1:
        # the line certificate in segments over the pool (each shift is a full
        # LU here); the GL |Omega| pool below (parallel.py, forked) is not used
        D.cert_nseg, D.cert_pool = int(a.cert_nseg), fullpar.POOL
        print(f"BFS: line certificate in {D.cert_nseg} segments over the pool", flush=True)
    def _plant_warmup():
        # once per mesh, before the loop (STAGE6 6.29): the widest plant
        # enclosure (every narrower one is filtered from it) and the
        # numerical-range heights of the tail bounds
        import sparse_spectrum as _ss
        _t0 = time.time()
        _ss.plant_eigs_right(D.A, D.M, -float(a.band_recut) - 0.6)
        _t1 = time.time()
        _ss.plant_heights(D.A, D.M)
        print(f"warm-up: plant enclosure right of {-float(a.band_recut) - 0.6:+.2f} ({_t1 - _t0:.1f} s), "
              f"numerical-range heights ({time.time() - _t1:.1f} s)", flush=True)
    if MIMO:
        if not BFS:
            _plant_warmup()
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
    _lat = None                                   # lattice.LatticeOracle (--oracle lattice), made with the pool below

    def _gate(arow_, b_, c_, nvec_, K_, **kw):
        """csp.stability_gate (the compensator's poles exactly, then the line
        Re = -eps clear); with --oracle lattice the line is the lattice's sweep
        through D.alpha_certified (memoised: the design's own sweep costs nothing
        again), on the band of the abscissa certificate, which contains the
        gate's."""
        if _lat is None:
            return csp.stability_gate(D.A, D.M, b_, c_, ctl.A_of(*arow_), ctl.Bc_of(arow_, D.r, D.mc), nvec_, K_, **kw)
        eps_ = float(kw.get('eps', 1e-3)); lim_ = max(-eps_, float(kw.get('ctrl_max', 0.0)))
        pol_ = np.linalg.eigvals(np.asarray(ctl.A_of(*arow_), complex))
        if pol_.size and float(np.max(pol_.real)) > lim_ + 1e-8:
            j_ = int(np.argmax(pol_.real))
            return False, f"controller pole Re={pol_[j_].real:+.6g} > allowance {lim_:+.6g}", 0
        al_, lam_, _, _, _, clr_ = D.alpha_certified(arow_, nvec_, K_, b_, c_, eps=eps_)
        if clr_:
            return True, "controller poles stable and the line is clear (lattice)", 0
        if lam_ is not None and (lam_.real > -eps_ or abs(lam_.real + eps_) < 1e-9 * max(1.0, abs(lam_))):
            return False, f"eigenvalue at {lam_:.6g} on Re=-{eps_:g}", 0
        return False, f"INCONCLUSIVE: could not prove the line Re=-{eps_:g} clear (lattice)", 0

    def _path_to(path_, z0_, z1_, info_=None, info_out=None):
        """The SQP's accepted iterates from z0_ (the last accepted design) to the
        candidate z1_, consecutive duplicates removed; None if z1_ is not on it.
        info_ (the SQP's path_info, parallel to path_) and info_out (a list):
        info_out gets the entries of the points kept, None for z0_ (20261010j
        stage B, G2)."""
        if not path_:
            return None
        P_ = [np.asarray(p_, float) for p_ in path_]
        j_ = next((i_ for i_ in range(len(P_) - 1, -1, -1) if np.array_equal(P_[i_], np.asarray(z1_, float))), None)
        if j_ is None:
            return None
        pts_ = [np.asarray(z0_, float)]
        inf_ = [None]
        for k_, p_ in enumerate(P_[:j_ + 1]):
            if not np.array_equal(p_, pts_[-1]):
                pts_.append(p_)
                inf_.append(info_[k_] if (info_ is not None and k_ < len(info_)) else None)
        if info_out is not None:
            info_out[:] = inf_
        return pts_

    def _gate_along(pts_, watch_, logs_=None):
        """continuation_gate on each segment of pts_ in turn.  Returns (ok, watch
        at the end, queries, splits, certified prefix {z, watch}, failing segment).
        logs_ (a list): one continuation_gate log per segment (G2)."""
        nq_ = ns_ = 0; w_ = list(watch_)
        for i_ in range(len(pts_) - 1):
            r_ = {}
            lg_ = [] if logs_ is not None else None
            if logs_ is not None:
                logs_.append(dict(seg=i_, watch=list(w_), log=lg_))
            ok_, w2_, q_, sp_ = continuation_gate(D, pk, pts_[i_], pts_[i_ + 1], w_, a.eps, depth=a.cont_depth,
                                                  motion_frac=a.cont_motion, motion_floor=a.cont_floor,
                                                  kwatch=a.cont_k, pool=fullpar.POOL, reach=r_, log=lg_)
            nq_ += q_; ns_ += sp_
            if not ok_:
                if r_.get('z') is not None:
                    pre_ = r_
                elif i_ > 0:
                    pre_ = dict(z=np.array(pts_[i_], float), watch=list(w_))
                else:
                    pre_ = {}
                return False, w2_, nq_, ns_, pre_, i_
            w_ = list(w2_)
        return True, w_, nq_, ns_, {}, len(pts_) - 1

    _n_gate_fail = [0]

    def _save_gate_fail(rec_):
        """The first three gate refusals of the job as gate_fail_k.npz in the run's
        directory (save_gate_fail; lattice runs only, where QP_DUMP is set too)."""
        if _lat is None or _n_gate_fail[0] >= 3:
            return
        _n_gate_fail[0] += 1
        save_gate_fail(a.outdir, rec_, pk, flags, MIMO, a.eps, gbest,
                       dict(k=a.cont_k, depth=a.cont_depth, motion=a.cont_motion, floor=a.cont_floor))
    _ctrl_borrowed, _ctrl_default = False, None
    _resume_gamma = None
    if BFS:
        # the start (--init: bfs/hifoo_init.py, or the paper's devices): a REAL
        # order-r controller, the actuator centre (x, y) and the sensor window
        # start x_s; no seed catalogue, no rescaling
        flags['real_ctrl'] = True
        flags['ctrl_modal'] = (a.ctrl_modal == 'on')
        if MIMO:
            # MIMO (BFS.md 5): the start record (--mimo-start, bfs/hifoo_init.py --m --p),
            # read once here so the BFS start checks below see its m x p design
            (arow0, nvec0, K0, xa0, xs0, va0, vs0), _mimo_g0 = _mimo_setup(a, D, flags, tag)
            xa0 = np.asarray(xa0, float).reshape(int(flags['m']), -1)
            xs0 = np.asarray(xs0, float).reshape(int(flags['p']), -1)
            _mz = np.load(a.mimo_start, allow_pickle=True)
            _pos_start = dict(xa_start=np.asarray(_mz['xa'], float), xs_start=np.asarray(_mz['xs'], float))
        else:
            ini = np.load(a.init, allow_pickle=True)
            arow0 = tuple(np.asarray(ini[k], complex) for k in ('d', 'e', 'f'))
            nvec0 = np.asarray(ini['nvec'], complex); K0 = complex(ini['K'])
            xa0, xs0 = np.asarray(ini['xa'], float).ravel(), np.atleast_1d(np.asarray(ini['xs'], float))
            _pos_start = dict(xa_start=xa0.copy(), xs_start=xs0.copy())   # the --init positions, for the record (a resume keeps them)
            va0 = vs0 = None
    else:
        ini = np.load(a.init or f"init_ne{a.ne}.npz")
        arow0 = (ini['d'], ini['e'], ini['f']); nvec0 = ini['nvec']; K0 = 0j
        xa0, xs0 = float(ini['xa']), float(ini['xs'])
        va0 = D.gauss(xa0); vs0 = D.gauss(xs0)
    if a.seed and not BFS:
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
    if not BFS:
        _al = D.norm_M(D.gauss(xa0).astype(complex))
        _be = D.norm_M(D.gauss(xs0).astype(complex))
        nvec0 = nvec0 * (_al * _be); K0 = K0 * (_al * _be)
    _pos = (lambda v: np.asarray(v, float)) if BFS else (lambda v: float(v))

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
            a.start_from_ckpt = True              # the anchor is inherited: counted at the start (G4, 20261010j)
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
    if src is None and not a.cold and not MIMO and not BFS:
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
            # The parent's controller, fixed there, is part of the parent's
            # design: it is this child's default start (STAGE6 6.31: the
            # nesting guarantee needs the parent's design among the starts;
            # the seed default only coincided with it when the parent had
            # started from the same seed)
            if all(k_ in ck.files for k_ in ('d', 'e', 'f', 'nvec', 'K')) and len(np.atleast_1d(ck['d'])) == D.r:
                arow0 = (ck['d'], ck['e'], ck['f'])
                nvec0, K0 = ck['nvec'], complex(ck['K'])
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
        if pf['B']: xa0 = _pos(ck['xa'])
        if pf['C']: xs0 = _pos(ck['xs'])
        if pf['FB']:
            va0 = ck['xi_a'] if ('xi_a' in ck.files and ck['xi_a'].size) \
                else D.smooth_unmap(ck['va'])
        if pf['FC']:
            vs0 = ck['xi_s'] if ('xi_s' in ck.files and ck['xi_s'].size) \
                else D.smooth_unmap(ck['vs'])
        # A child that frees a FIELD its parent did not free (B -> FB, C -> FC,
        # B+C -> FB+FC, K+B -> K+FB, FB -> FB+FC for the sensor, ...): the
        # parent's actuator / sensor IS a field -- its Gaussian at the parent's
        # optimised or fixed position -- and the child must start from it.  The
        # groups rule above inherited nothing for it and the child started from
        # the default field: FB from B at 9.9949 against B's 10.1374, FB+FC from
        # FB at 8.6106 against 9.8694 -- not the parent's design, so no nesting
        # guarantee (STAGE6 6.31, lattice check at ne = 200).
        for _fk, _pk_, _key in (('FB', 'B', 'a'), ('FC', 'C', 's')):
            if flags.get(_fk) and not pf0.get(_fk):
                _psi = (np.asarray(ck[f'psi_{_key}'], complex) if f'psi_{_key}' in ck.files and ck[f'psi_{_key}'].size
                        else np.asarray(D.bc(float(ck['xa']), float(ck['xs']), None, None)[2 if _key == 'a' else 3], complex))
                if _key == 'a':
                    va0 = D.smooth_unmap(_psi)
                else:
                    vs0 = D.smooth_unmap(_psi)
                print(f"  {_fk} starts from the parent's {'actuator' if _key == 'a' else 'sensor'} field "
                      f"(its Gaussian at x = {float(ck['xa' if _key == 'a' else 'xs']):+.4f}, "
                      f"{'optimised' if pf0.get(_pk_) else 'fixed'} there)", flush=True)
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
    _stab_rec = None
    if BFS:
        # the plant's spectrum near the axis, once, before any certificate (the
        # stabilisation phase's included): the band [0, --bfs-band] (+ pad)
        a.wlo, a.whi = 0.0, float(a.bfs_band)
        D.band_of = lambda: a.whi + a.wpad        # the stability certificate's band follows the run's
        _t0 = time.time()
        _ev = D.plant_spectrum(a.whi + a.wpad)
        _ev = _ev[_ev.imag >= 0]
        print(f"BFS plant spectrum near the axis (shift-invert, Im in [0, {a.whi + a.wpad:g}]): "
              f"{len(_ev)} eigenvalues, rightmost {_ev[np.argmax(_ev.real)]:.4f} "
              f"({'converged' if D.plant_ev_ok else 'NOT all shifts converged'}, {time.time() - _t0:.0f} s)", flush=True)
        _bad = [k_ for k_, x_, sd_ in (('actuator', xa0, 'a'), ('sensor', xs0, 's'))
                if not all(D.inside(x1_, sd_) for x1_ in (np.asarray(x_, float).reshape(-1, D.pos_dims[sd_])))]
        if _bad:
            raise SystemExit(f"BFS: the start's {' and '.join(_bad)} centre is not inside the domain "
                             f"(margin {D.pos_margin:g}, --bfs-margin)")
        if a.stabilise and src is None and not os.path.exists(tag + '_ckpt.npz'):
            # HIFOO's phase at this scale: continuation in the loop gain from
            # the open loop (bfs/hifoo_init.feasible_start, BFS.md 6)
            from bfs.hifoo_init import feasible_start
            _t0 = time.time()
            _bb_, _cc_, *_ = D.bc(xa0, xs0)
            arow0, nvec0, K0, _fi = feasible_start(D, arow0, nvec0, K0, _bb_, _cc_, a.eps, delta=a.stab_delta,
                                                   pool=fullpar.POOL)
            _stab_rec = dict(_fi, wall=round(time.time() - _t0, 1), stabilised=bool(_fi['tau'] > 0.0),
                             method='loop-gain continuation')
            from bfs import mumps_lu as _ml
            print(f"  [ledger] feasible start {_stab_rec['wall']:.0f} s; {_ml.report()}; {fullpar.report()}",
                  flush=True)
            if _fi['tau'] <= 0.0:
                print("FATAL: the open loop is not certified inside the margin (BFS.md 6)", flush=True)
                return
            if a.wind:
                # the continuation chain's result, checked independently of it:
                # the all-frequency count right of -eps (BFS.md 4)
                _bb_, _cc_, _pa_, *_ = D.bc(xa0, xs0)
                _t0 = time.time()
                _wo = D.winding(_pa_, _bb_, _cc_, arow0, nvec0, K0, -a.eps, hmin_abs=WIND_HMIN_ABS)
                _wind_sec[0] = max(_wind_sec[0], time.time() - _t0)
                _stab_rec['count_right_of_eps'] = int(_wo['count']) if _wo['ok'] else None
                # the count's own data, for the anchor's entry in the veto's ledger (a
                # record that falls back to it reports them; 2026-10-16)
                _stab_rec['count_info'] = dict(npts=int(_wo.get('npts', 0) or 0), seconds=round(time.time() - _t0, 1),
                                               wtail=float(_wo.get('wtail', np.nan)), wstar=float(_wo.get('wstar', np.nan)),
                                               p_plant=int(_wo.get('p_plant', -1)))
                print(f"  feasible start: {_stab_rec['count_right_of_eps']} closed-loop eigenvalue(s) right of "
                      f"Re = {-a.eps:g} over all frequencies (winding count, {_wo['npts']} pts, "
                      f"{time.time() - _t0:.0f} s)", flush=True)
                if _wo['ok'] and _wo['count'] > 0:
                    print("FATAL: the count contradicts the continuation chain (BFS.md 6)", flush=True)
                    return
    if flags.get('ctrl_modal') and flags.get('K'):
        # MODAL COMPENSATOR (--ctrl-modal; 20261010e, controller.modal_*): A_c in
        # real 2 x 2 diagonal blocks.  A start already in that form (a modal run's
        # record or checkpoint) is kept; the zero controller (C_c = D_c = 0: no
        # transfer function to keep) is brought to it with the same poles; any
        # other start is refused (its B_c / C_c would have to follow the similarity)
        import controller as _ctl_m
        _ord = ([int(v) for v in flags['loop_orders']] if (MIMO and flags.get('struct') == 'decentral')
                else [len(np.atleast_1d(arow0[0]))])
        _cut = _ctl_m.modal_cut(_ord)
        if any(np.asarray(arow0[1])[i] != 0 or np.asarray(arow0[2])[i] != 0 for i in _cut):
            if np.any(np.asarray(nvec0)) or np.any(np.asarray(K0)):
                raise SystemExit("--ctrl-modal on: the start's compensator is neither in modal 2 x 2 blocks nor the "
                                 "zero controller (resume a tridiagonal run with --ctrl-modal off, MODAL=0)")
            _mf = _ctl_m.modal_form(arow0[0], arow0[1], arow0[2], _ord)
            print(f"modal compensator: the zero controller's poles in 2 x 2 blocks (loops {_ord}); "
                  f"abscissa {float(np.max(np.linalg.eigvals(_ctl_m.A_of(*_mf)).real)):+.6f}", flush=True)
            arow0 = tuple(_mf) + tuple(arow0[3:])
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
            _P0, _how0 = (_fb.atoms_from_record(_srcz, key, a.field_atoms, a.atom_smin, a.atom_kmax, D.xn)
                          if _srcz is not None else (None, ''))
            if _P0 is not None and len(_P0) == 1:
                p0 = _P0[0]; how = 'from the record, ' + _how0
            else:
                v0n = D.smooth_map(v0_) if getattr(D, 'field_basis_' + key, None) is None and \
                    getattr(D, 'field_input_' + key, None) is None else v0_
                p0, err = _fb.fit_atoms(D.xn, v0n, D.M, a.field_atoms, a.atom_smin, a.atom_kmax)
                how = f'fitted to the start field (M-norm error {err:.3f})' + \
                    (f'; {_how0}' if _P0 is None and _how0 not in ('', 'no atoms in the record') else '')
            setattr(D, 'atoms_' + key, dict(n=int(a.field_atoms), smin=float(a.atom_smin), kmax=float(a.atom_kmax), p0=p0))
            print(f"field atoms ({'actuator' if key == 'a' else 'sensor'}): {a.field_atoms} Gabor atoms, width >= {a.atom_smin}, "
                  f"|wavenumber| <= {a.atom_kmax}; {how}", flush=True)
    if MIMO:
        if not BFS:
            pk_args, _mimo_g0 = _mimo_setup(a, D, flags, tag)
        from packer import PackerMIMO
        pk = PackerMIMO(D, flags, *pk_args)
        print(f"MIMO: {flags['m']} actuators, {flags['p']} sensors, {flags['struct']} compensator of order {D.r}"
              + (f" (loops {flags['loop_orders']})" if flags['struct'] == 'decentral' else '')
              + f"; {pk.nth} free {'real' if getattr(pk, 'real_ctrl', False) else 'complex'} controller entries; start {a.mimo_start}"
              + (f" (record gamma {_mimo_g0:.6f})" if _mimo_g0 is not None else ''), flush=True)
    else:
        pk = Packer(D, flags, *pk_args)
    z = pk.z0()
    if a.stabilise and not BFS:
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

    _par = None
    if a.nproc and a.nproc > 1 and not BFS:
        nseg = max(1, a.cert_nseg)
        D.cert_pool = fullpar.POOL if nseg > 1 else None
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
        print(f"pool: {a.nproc} workers for the full-model tasks (fullpar) and the |Omega| constraints; "
              + (f"abscissa certificate SERIAL"
                 if D.cert_nseg <= 1 else
                 f"abscissa certificate in {D.cert_nseg} segments"), flush=True)

    _band_last = dict(lam=[], info={})          # the last located set (dumps, records)
    n_allw_outside = [0]                         # all-w certificates that found the norm outside the band
    n_band_incomplete = [0]

    _band_memo = {}

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
        if BFS:
            # no plant enclosure without M^-1 (BFS.md 4): the band covers the
            # compensator poles only
            return max(a.whi, band_from_controller(arow_)[1])
        # the same design is asked twice a round (band-first, then at acceptance
        # with the harvested eigenvalues): the shift results are memoised per
        # design (STAGE6 6.29)
        _key = tuple(np.asarray(x_).tobytes() for x_ in (*arow_[:3], np.asarray(arow_[3]) if len(arow_) > 3 else 0, nvec_, K_, b_, c_))
        if _band_memo.get('key') != _key:
            _band_memo.clear(); _band_memo['key'] = _key; _band_memo['shifts'] = {}
        lam, info = _ss.closed_loop_set(D.A, D.M, D.cl(b_, c_), arow_, nvec_, K_, -a.band_recut,
                                        band=half, mc=D.mc, count=count, inner=max(0.0, (half - a.wpad) / 1.5 - 0.5),
                                        extra=[complex(z_) for z_ in extra],
                                        ctrl_exact=(a.wind_ctrl == 'exact'), memo=_band_memo['shifts'])
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

    if BFS:
        # real plant: w >= 0 (sigma(-w) = sigma(w), eigenvalues in conjugate
        # pairs); a fixed band [0, --bfs-band] widened to the controller poles
        a.wlo, a.whi = 0.0, float(a.bfs_band)
        print(f"BFS: band [0, {a.whi:g}] (real plant: w >= 0); no all-frequency tail bound (BFS.md 4)", flush=True)
    wlo_run, whi_run = band_from_controller(arow0)
    if BFS:
        wlo_run = 0.0
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
    # THE CERTIFICATE'S SETTINGS BEFORE THE START IS CERTIFIED (STAGE6 6.31):
    # they were applied after it, so the start of every run was certified with
    # the frozen disks, tolerance 1e-4 and no all-w question while the rounds
    # used the strict ones -- a child resumed from its parent's record started
    # 4e-4 BELOW the parent's certified value (K+FB+FC from KSP+FB+FC, ne=200:
    # 4.154741 against 4.156543), the same design measured less rigorously
    csp.DISK_STRICT = (a.cert_disk == 'strict')
    csp.NEAR_K = int(a.cert_near_k); csp.CLIMB = a.cert_climb; csp.CERT_SPLIT = int(a.cert_split)
    certified_norm.rtol = float(a.cert_rtol)
    certified_norm.all_w = (a.cert_all_w == 'on') and not BFS
    certified_norm.sampled = int(a.bfs_sampled) if BFS else 0
    certified_norm.true_lb = bool(BFS)            # the certificate starts from an attained full-model value
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
    _ls, _lz = getattr(certified_norm, 'last_sweep', None), getattr(certified_norm, 'last_sweep_z', None)
    if _ls is not None and _lz is not None and np.array_equal(_lz, np.asarray(z, float)) \
            and len(_ls[0]) == len(wg) and np.allclose(_ls[0], wg, rtol=0, atol=1e-12):
        sg = np.asarray(_ls[1], float)       # the start certificate's full sweep, same grid (STAGE6 6.29)
    else:
        sg = np.array(fullpar.sigmas(b_, c_, arow_, nvec_, K_, wg) if fullpar.active() else
                      [cl_.gain(float(w), arow_, nvec_, K_, want_vectors=False)['sigma'] for w in wg])
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
                # the tau cap and its clean-acceptance streak (20261010j stage B): a
                # cap set by the gate's evidence (770633 round 1: 3.5) held across a
                # job boundary, as it would within one job; nan = off (older files: off)
                if 'tau_rho' in _srcw.files and np.isfinite(float(_srcw['tau_rho'])):
                    _resume_state.update(tau_rho=float(_srcw['tau_rho']),
                                         tau_streak=int(_srcw['tau_streak']) if 'tau_streak' in _srcw.files else 0)
            # PROVENANCE (20261010j stage B): which code wrote the file this run resumes
            if getattr(a, 'start_from_ckpt', False):
                _wb = None
                try:
                    _wb = json.load(open(tag + '_ckpt.json')).get('code_version')
                except Exception:
                    pass
                if not _wb and 'code_version' in _srcw.files:
                    _wb = str(_srcw['code_version'])
                print(f"resume: checkpoint written by {_wb or 'a version that did not record it (before 20261010j)'}, "
                      f"continued by {_code_version()}", flush=True)
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
    _retire_streak = {}                                 # --omega-retire: member -> consecutive rounds below FRAC x gamma
    t0 = g0
    print(f"start gamma={g0:.6f}, |Omega|={len(Omega)}", flush=True)
    if a.resume and _resume_gamma is not None and a.resume_guard > 0 and g0 > a.resume_guard * _resume_gamma:
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
    n_prefix = [0]       # refused steps replaced by their certified prefix (--gate-prefix)
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
    _n_noimp = 0           # consecutive rounds without an accepted design (--stop-progress mismatch exemption)
    exch_stalled = False
    per_outer = (a.wall - (time.time() - t_start) - 45.) / max(a.outer, 1)
    gbest, zbest, albest = g0, z.copy(), None
    _wind_best = None       # (design, acceptance-time winding count right of -eps) of the best design
    _allw_best = None       # (design, level, acceptance-time all-w certificate) of the best design
    _t_best = None          # the SQP's t, smax and rows at the best design's round (the record's activity vs t, 20261010j)
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
        # within this job's [--inner-maxit-min, --inner-maxit-max] (the cap since
        # 20261010j stage B: a record now carries the budget too, and a short test
        # round from one -- --inner-maxit-max 3 -- would otherwise run its 40)
        _maxit_cur = (min(max(int(_resume_state['maxit_cur']), int(a.inner_maxit_min)), int(a.inner_maxit_max))
                      if a.inner_maxit_adapt else _maxit_cur)
        _sqp_trust_prev = (_resume_state['sqp_trust'] or None)
        _peakset.update(round(float(x), 6) for x in _resume_state['peakset'])
        if a.tau and _resume_state.get('tau_rho') is not None:
            _tau_rho, _tau_streak = float(_resume_state['tau_rho']), int(_resume_state.get('tau_streak', 0))
        print(f"resume: inner budget {_maxit_cur}, SQP trust {_sqp_trust_prev}, {len(_resume_state['peakset'])} peak members restored"
              + (f", tau cap {_tau_rho:.4g} (streak {_tau_streak})" if _tau_rho is not None else ", no tau cap"), flush=True)
    gbest_cert, zbest_cert = float(gbest), (zbest.copy() if zbest is not None else None)   # --cert-every: last FULLY certified best
    _al_accepted = None                 # certified abscissa of the accepted design (SQP start check)
    # ---- STAGE 5: the projected oracle for the inner loop ------------------
    _ro = None
    _tangent_sides = set()                   # fields whose input basis is their atoms' tangent space (6.33)
    if a.oracle == 'reduced':
        from reduced import ReducedOracle
        import reduced as _red_mod
        _red_mod.ROW_MPI = a.row_workers == 'mpi'
        _red_mod.ROW_WORKERS = 0 if _red_mod.ROW_MPI else max(0, int(a.row_workers))
        if _red_mod.ROW_WORKERS:
            print(f"rows: gain_batch in {_red_mod.ROW_WORKERS} forked worker(s) + this process", flush=True)
        if _red_mod.ROW_MPI:
            print(f"rows: gain_batch on the MPI ranks ({fullpar.NPROC if fullpar.active() else 1}, this one included)",
                  flush=True)
        _ro = ReducedOracle(D)
        certified_norm.oracle = _ro          # fine reduced sweep for the peaks list
        certified_norm.pin = True            # hold the anchor LUs to the next enrichment (STAGE6 6.29)
        if a.oracle_inputs == 'on':
            # position-robust model (STAGE5 5): input bases over the position
            # ranges + union Galerkin; for K-only cases the Galerkin alone
            from reduced import input_basis
            # --field-input tangent (STAGE6 6.33): an atom field moves on a
            # 5 n_atoms-parameter family; to first order its changes lie in
            # span(M J), J = the atoms and their centre / width / wavenumber
            # derivatives (the amplitudes' real and imaginary parts span the
            # same complex columns; centre and wavenumber give the same complex
            # direction (x-c) phi, so the span is phi, (x-c) phi, (x-c)^2 phi:
            # 3 per atom, the SVD drops the rest).  That is the field side's input basis, at
            # the current design, rebuilt every round -- in place of the
            # --field-basis modes, which made the model exact for every field
            # in their span (kd ~300 at ne=200, STAGE6 6.31).
            _tangent_on = a.field_input == 'tangent' and bool(a.field_atoms)

            def _tangent_basis(z_, key):
                J = pk.field_jacobians(z_).get(key)
                if not J or any(j is None for j in J):
                    return None
                cols = []
                for j in J:
                    na = j.shape[1] // 5
                    cols.append(np.hstack([j[:, :na], j[:, 2 * na:]]))
                U_, s_, _ = np.linalg.svd(np.asarray(D.M @ np.hstack(cols), complex), full_matrices=False)
                _tangent_sides.add(key)
                return U_[:, s_ > 1e-12 * s_[0]]
            if BFS:
                # free ports (BFS.md 5): the port vectors' second-order jets in
                # their positions at the round's design, rebuilt every round (as
                # the atoms' tangent)
                def _jet_basis(z_, key):
                    _xa_j, _xs_j = pk.unpack(z_)[3:5]
                    _sd = 'a' if key == 'xa' else 's'
                    _X = np.asarray(_xa_j if key == 'xa' else _xs_j, float).reshape(-1, D.pos_dims[_sd])
                    J_ = np.vstack([D.port_jet(x1_, _sd) for x1_ in _X])      # MIMO: every port's jet
                    U_, s_, _ = np.linalg.svd(np.asarray(J_.T, complex), full_matrices=False)
                    _tangent_sides.add(key)
                    return U_[:, s_ > 1e-12 * s_[0]]
                _tangent_basis_fields = _tangent_basis

                def _tangent_basis(z_, key):
                    return _jet_basis(z_, key) if key in ('xa', 'xs') else _tangent_basis_fields(z_, key)
                _Pb = _tangent_basis(zbest, 'xa') if flags.get('B') else None
                _Pc = _tangent_basis(zbest, 'xs') if flags.get('C') else None
            else:
                _Pb = input_basis(D, 'b', a.xa_lo, a.xa_hi, other=0.98)[0] if flags.get('B') else None
                _Pc = input_basis(D, 'c', a.xs_lo, a.xs_hi, other=-1.03)[0] if flags.get('C') else None
            # free fields on a basis: b = M va/||va||_M lies in span(M Phi_a)
            for _fbk, _fik, _w in (('field_basis_a', 'field_input_a', 'b'), ('field_basis_s', 'field_input_s', 'c')):
                if _tangent_on:
                    _Q = _tangent_basis(zbest, 'va' if _w == 'b' else 'vs')
                    if _Q is not None:                     # atoms: their tangent space at this design (6.33)
                        if _w == 'b':
                            _Pb = _Q
                        else:
                            _Pc = _Q
                        continue
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
                _ro.set_input_basis(_Pb, _Pc, per_design=bool(_tangent_sides))
            _ro.galerkin = True
            print(f"oracle: position-robust (union Galerkin"
                  f"{', input basis b %d' % _Pb.shape[1] if _Pb is not None else ''}"
                  f"{', c %d' % _Pc.shape[1] if _Pc is not None else ''})", flush=True)
    certified_norm.sweep_n = int(a.cert_sweep)
    if fullpar.active():
        # re-fork the full-model pool now that the counters, the certificate
        # settings and the reduced oracle's input bases exist (the enrichment
        # anchors' tasks use them)
        fullpar.start(a.nproc, D=D, Phi_b=getattr(_ro, 'Phi_b', None) if _ro is not None else None,
                      Psi_c=getattr(_ro, 'Psi_c', None) if _ro is not None else None)
    if a.oracle == 'lattice':
        # THE FULL MODEL ON THE LATTICE (lattice.py; BFS.md 5, 2026-10-06): the
        # SQP's rows, its stability rows and the gates' line certificates on the
        # MPI ranks, each rank keeping the plant factors of its lattice points
        import lattice as _lat_mod
        _lat = _lat_mod.LatticeOracle(D, pk, flags, pool=(fullpar.POOL if fullpar._MPI is not None else None),
                                      use_driver=(a.lattice_driver == 'on'), real=True)
        certified_norm.lattice = _lat
        print(f"oracle: the full model on the lattice ({_lat.size} rank(s){', the driver included' if _lat.use_driver else ''}; "
              f"{_lat_mod.LU_CAP} factors kept per rank, fixed rows on 2^-{_lat_mod.ROW_LEVEL}, followed peaks by Newton "
              f"to |sigma'| <= {_lat_mod.FOLLOW_GTOL:g} sigma, missed peaks by the level-set certificate ("
              f"{_lat_mod.LEVEL_K} pencil eigenvalues per axis disk, through the plant factors); the line from "
              f"{_lat.base_grid(0.0, 1.0).__len__()} equispaced centres, "
              + "the stability rows those of the eigenvalues the line's disks see (no strip since 2026-10-16)"
              + f"; rows aimed {1 + _sqi.LAT_CUSHION:g} mg inside the line, trials across it "
              + ("cut back" if _sqi.LAT_CROSS == 'cut' else "weighed by the merit")
              + f"; {_lat_mod.LINE_K} eigenvalue(s) per disk; batches by "
              + (f"the solves counted, a factorisation {_lat_mod.LU_SOLVES:g}, longest first" if _lat_mod.TASK_COSTS == 'measured'
                 else "static costs")
              + (f", row requests kept for {_lat_mod.REQ_MEMO} design(s)" if _lat_mod.REQ_MEMO > 0 else ", no request memo")
              + (f"; rows merged within {_lat_mod.ROW_MERGE:g}, the nearest shift's copy kept" if _lat_mod.ROW_MERGE > 1e-8
                 else f"; rows merged within 1e-8 (copies within {_lat_mod.ROW_DUP:g} counted, not merged)") + ")",
              flush=True)
        # 20261010j stage D: a line of its own (the oracle line above is the same in every arm); the
        # synchronous call (off, a measurement arm since 2026-10-16) says so too
        if _sqi.LAT_ASYNC != 'off':
            print(f"oracle: the missed-peak level set BESIDE the next SQP iteration (--lattice-async {_sqi.LAT_ASYNC}"
                  + ("; every look-ahead put back and redone -- the restore's test" if _sqi.LAT_ASYNC == 'redo' else '')
                  + (f"): its disks on the {_lat.size - 1} worker rank(s), never the driver" if _lat.size > 1 else
                     "): NOT applied -- one rank"), flush=True)
        else:
            print("oracle: the missed-peak level set synchronous, after each accepted step (--lattice-async off)", flush=True)
        # the ports' positions behind every (b, c) this process builds: the gates
        # call D.alpha_certified with b and c, and the ranks build them from the
        # positions
        _bc_orig = D.bc; _bc_reg = {}

        def _bc_lat(xa_, xs_, va_=None, vs_=None):
            out_ = _bc_orig(xa_, xs_, va_, vs_)
            _bc_reg[(hash(np.asarray(out_[0]).tobytes()), hash(np.asarray(out_[1]).tobytes()))] = \
                (np.array(xa_, float), np.array(xs_, float))
            while len(_bc_reg) > 256:
                _bc_reg.pop(next(iter(_bc_reg)))
            return out_
        D.bc = _bc_lat
        _ac_orig_l = D.alpha_certified

        def _ac_lattice(arow_, nvec_, K_, b_, c_, eps=1e-3, wlo=None, whi=None, eigs_seen=None):
            pos_ = _bc_reg.get((hash(np.asarray(b_).tobytes()), hash(np.asarray(c_).tobytes())))
            if pos_ is None:                          # ports not built here: the serial certificate
                return _ac_orig_l(arow_, nvec_, K_, b_, c_, eps=eps, wlo=wlo, whi=whi, eigs_seen=eigs_seen)
            r_ = _lat.line(_lat.design_raw(arow_, nvec_, K_, pos_[0], pos_[1]), eps,
                           0.0 if wlo is None else max(0.0, wlo), D.stab_whi if whi is None else whi, reach=-np.inf)
            if not np.isfinite(r_['alpha']):                # no eigenvalue verified anywhere: the serial certificate
                return _ac_orig_l(arow_, nvec_, K_, b_, c_, eps=eps, wlo=wlo, whi=whi, eigs_seen=eigs_seen)
            if eigs_seen is not None:
                eigs_seen.extend(r_['seen'])
            lam_ = r_['witness'] if r_['witness'] is not None else max(r_['seen'], key=lambda l_: l_.real, default=None)
            # an INCONCLUSIVE line (no witness, a gap left) reports alpha = +inf: every
            # caller then refuses -- the continuation gate's midpoint test reads alpha
            # only, and the rightmost eigenvalue seen says nothing about the gap
            al_ = float(r_['alpha']) if (r_['clear'] or r_['witness'] is not None) else np.inf
            return al_, lam_, None, None, D.cl(b_, c_), bool(r_['clear'])
        D.alpha_certified = _ac_lattice
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
            rs_ = _sigmas(rc_, _ws_, aa_, nn_, KK_)
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
        todo = []
        for wf_ in freqs:
            w_ = round(float(wf_), 6)
            prev = _anchored.get(w_)
            # re-anchor when the fields moved (Woodbury columns refresh)
            if prev is not None and np.all(np.abs(np.asarray(prev[0]) - xa_) < 1e-9) \
                    and np.all(np.abs(np.asarray(prev[1]) - xs_) < 1e-9):
                continue
            _anchored[w_] = (np.array(xa_, float), np.array(xs_, float))
            todo.append(float(wf_))                 # the exact frequency: its LU may be pinned
        # the anchors' factorisations in the full-model pool, the basis updates
        # in order (reduced.enrich_many; serial without the pool)
        _ro.enrich_many(clf, todo, aa_, nn_, KK_, nvec_dirs=a.oracle_nvec, bc=(bb_, cc_))
        nadd = len(todo)
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
    # THE ACCEPTED DESIGNS OF THIS JOB (20261010j stage B; the user's decision of
    # 2026-10-14: the save-time winding count VETOES): the anchor, then every
    # accepted design, each with its watch list and, where one was computed, its
    # conclusive count right of -eps -- the fallbacks of the veto at the save
    _accepted = []
    _anchor_count = None
    if BFS and a.wind and a.cont and _anchor_ok:
        if _stab_rec is not None and 'count_right_of_eps' in _stab_rec:
            # the anchor is this job's feasible start, counted above: 0 is known; an
            # INCONCLUSIVE count there is not repeated (the same design, the same march)
            if _stab_rec['count_right_of_eps'] == 0:
                _anchor_count = dict(count=0, ok=True, **dict(_stab_rec.get('count_info') or {}), how='feasible start')
        else:
            # THE INHERITED ANCHOR COUNTED (gate-refusal.md G4 with its verifier's
            # correction): a checkpoint, a continued record or a start record whose
            # feasible-start count did not run (STABILISE=0) was anchored by its line
            # alone -- the feasible start's count right of -eps (BFS.md 6) runs only
            # on a start without a checkpoint -- and the parents' records hold
            # counts right of -eps/2 only, which do not exclude (-eps, -eps/2)
            # (STAGE6 6.29: 0 right of -eps implies 0 right of -eps/2, not the
            # converse), while the k = 1 line can hide an eigenvalue behind a pinned
            # one (alpha -0.0501042 at the decentral no-strip record).  FATAL only on
            # a conclusive count > 0; INCONCLUSIVE: logged, and the line-verified
            # anchor is kept (deferring it would drop the gate until the first
            # acceptance, a weaker chain than the inherited one)
            try:
                _o = _wind_right_of_margin(_bb0, _cc0, _aa0, _nn0, _KK0, list(_seen0) + list(watch_cur),
                                           pa_=D.bc(_xa0, _xs0, _va0, _vs0)[2])
                if _o['ok'] and _o['count'] > 0:
                    print(f"FATAL: the inherited anchor has {_o['count']} closed-loop eigenvalue(s) right of "
                          f"Re={-a.eps:g} (winding count, {_o['npts']} pts, {_o['seconds']:.0f} s) although its line "
                          f"is clear: the chain broke upstream (an escape); not running", flush=True)
                    return
                if _o['ok']:
                    # wtail / p_plant too: a record that falls back to the anchor reported
                    # wtail nan and p_plant -1 (20261010n)
                    _anchor_count = dict(count=0, ok=True, npts=int(_o['npts']), seconds=_o['seconds'],
                                         wtail=float(_o.get('wtail', np.nan)), wstar=float(_o.get('wstar', np.nan)),
                                         p_plant=int(_o.get('p_plant', -1)), how='counted at the start')
                    print(f"continuation anchor: 0 closed-loop eigenvalues right of Re={-a.eps:g} over all frequencies "
                          f"(winding count, {_o['npts']} pts, {_o['seconds']:.0f} s): the inherited anchor is certified",
                          flush=True)
                else:
                    print(f"*** continuation anchor: the winding count right of Re={-a.eps:g} is INCONCLUSIVE "
                          f"({_o['npts']} pts, {_o['seconds']:.0f} s): the line-verified anchor is kept, its premise "
                          f"inherited", flush=True)
            except Exception as _e:
                print(f"*** continuation anchor: winding count failed ({type(_e).__name__}: {_e}); the line-verified "
                      f"anchor is kept", flush=True)
    if a.cont and _anchor_ok:
        _accepted.append(dict(round=-1, z=zbest.copy(), g=float(gbest), al=float(_al0),
                              watch=list(watch_cur) + list(_seen0), count=_anchor_count, how='anchor', prov=False))
    _t_round = []                                # round start times (--deadline)
    _deadline_stop = False                       # stopped by --deadline: the checkpoint stays (resume)
    # --deadline with the SQP (20261010j stage B, P5): each round's phase before the
    # inner solve (its rows' setup), after its last iteration (the SQP's own
    # re-evaluations, the certificate, gate and checkpoint) and the longest SQP
    # iteration, measured
    _t_sqp_end, _pre_dur, _post_dur, _it_wall_max = [None], [], [], [0.0]

    def _round_reserve():
        """Seconds kept after a time-truncated inner solve: the round's certificate,
        gate and checkpoint (1.25 x the longest such phase seen) plus, with
        --cont-accept-winding, the longest winding count of this job (a refused
        step is counted: 2-7 min on the test plant, 20-40 min on the full mesh,
        BFS.md 9); never below --deadline-reserve."""
        return max(float(a.deadline_reserve),
                   1.25 * max(_post_dur, default=0.0) + (_wind_sec[0] if a.cont_accept_winding else 0.0))

    def _loop_state():
        """The loop's own state, in the checkpoint and (20261010j stage B,
        omega-growth.md P6) in the record, so that CONTINUE=1 from a finished
        record restores what a checkpoint restores: the adaptive inner budget,
        the SQP trust radius, the peak members (followed with --peak-follow), the
        tau cap and its streak (nan: off), and the code that wrote it.  Five of
        the six 20261010g jobs restarted at budget 10 with the start's peaks only
        (no 'resume:' line), 770629 alone resumed at 40 with 39 peak members."""
        return dict(maxit_cur=int(_maxit_cur), sqp_trust=float(_sqp_trust_prev or 0.0),
                    peakset=np.array(sorted(_peakset), float),
                    tau_rho=float(_tau_rho if _tau_rho is not None else np.nan), tau_streak=int(_tau_streak),
                    code_version=_code_version())

    def _write_ckpt(z_, g_, al_, freqs_, note_=None):
        """tag_ckpt.npz/.json for the design z_ (certified g_, abscissa al_) with
        the working set freqs_ and the loop state: written at every acceptance and,
        refreshed with the end-of-job state, at a deadline stop (the save below)."""
        try:
            aa_w, nn_w, KK_w, xa_w, xs_w, va_w, vs_w = pk.unpack(z_)
            _apa, _aps = pk.atom_params(z_)
            np.savez(tag + '_ckpt.npz', d=aa_w[0], e=aa_w[1], f=aa_w[2],
                     nvec=nn_w, K=KK_w, xa=xa_w, xs=xs_w,
                     **(dict(Bc=np.asarray(aa_w[3]), loop_orders=np.array(flags['loop_orders'])) if MIMO else {}),
                     atoms_a_p=(_apa if _apa is not None else np.zeros(0)),
                     atoms_s_p=(_aps if _aps is not None else np.zeros(0)),
                     **_atoms_record_keys(D, _apa, _aps),
                     va=(va_w if va_w is not None else np.zeros(0)),
                     vs=(vs_w if vs_w is not None else np.zeros(0)),
                     gamma=g_, alpha=float(al_), wlo=float(a.wlo), whi=float(a.whi),
                     **_loop_state(),
                     active_freqs=np.array(sorted(set(round(float(x), 6) for x in freqs_))))   # the working set: a resume that lacks it re-discovers the band
            json.dump(dict(case=a.case, gamma=float(g_),
                           alpha=float(al_),
                           note=note_ or ('checkpoint (PROVISIONAL: reduced-model certificate; a resume re-certifies it in full)'
                                          if getattr(certified_norm, 'last_provisional', False) else 'checkpoint'),
                           code_version=_code_version()),
                      open(tag + '_ckpt.json', 'w'))
        except Exception as _e:
            print(f"    checkpoint write failed: {_e!r}", flush=True)
    for it in range(a.outer):
        if BFS:
            # the run-time ledger (BFS.md 7): wall so far, the driver's own MUMPS
            # work and the full-model tasks (with their 95-worker estimate)
            from bfs import mumps_lu as _ml
            print(f"  [ledger] round {it} starts at {time.time() - t_start:.0f} s; {_ml.report()}; "
                  f"{fullpar.report()}" + (f"; {_red_mod.rows_report()}" if _ro is not None and _red_mod.ROW_MPI else ''),
                  flush=True)
            print(f"  [ledger] pool by task (cumulative): {fullpar.by_task(95)}", flush=True)
            if _lat is not None:
                print(f"  [ledger] {_lat.report()}", flush=True)
                for _ll in _lat.report_lines():        # rank 0, request kinds, factors, projection (20261010j stage C)
                    print(f"  [ledger] {_ll}", flush=True)
        if a.deadline > 0:
            # stop BEFORE a round that would not finish: the job's time limit
            # then never cuts a round (or the final save) in half (BFS.md 7)
            _t_round.append(time.time())
            _dur = max(np.diff(_t_round)) if len(_t_round) > 1 else 0.0
            if _t_sqp_end[0] is not None:
                _post_dur.append(_t_round[-1] - _t_sqp_end[0]); _t_sqp_end[0] = None
            if a.inner == 'sqp':
                # 20261010j stage B (driver-serial.md P5): the SQP stops on time itself
                # (its deadline below), so a round needs only the reserve for its
                # certificate, gate and checkpoint and a minimum inner budget -- not
                # 1.25 x the longest round, which ended the four MIMO jobs of 20261010g
                # 1.2-2.3 h before their limit (rounds of 1.1-1.9 h); plus the phase
                # before the solve (1.25 x the longest seen)
                _need = (1.25 * max(_pre_dur, default=0.0) + _round_reserve()
                         + max(float(a.deadline_min_inner), 3.0 * _it_wall_max[0]))
                _why = (f"a round needs {_need / 3600:.2f} h (reserve {_round_reserve() / 3600:.2f} h + the minimum "
                        f"inner solve + the setup before it)")
            else:
                _need = 1.25 * _dur
                _why = f"the longest round took {_dur / 3600:.2f} h"
            if it > 0 and (_t_round[-1] - t_start) + _need + a.deadline_final > a.deadline:
                print(f"  DEADLINE STOP: {(_t_round[-1] - t_start) / 3600:.2f} h elapsed, {_why}, "
                      f"--deadline {a.deadline / 3600:.2f} h", flush=True)
                _deadline_stop = True
                break
        # ALWAYS restart the inner solve from the best CERTIFIED point.  The
        # inner problem only sees Omega, so its optimum can sit where an
        # unmodelled frequency is huge; accepting that point and continuing from
        # it makes the exchange diverge.  The certified value is the only
        # trustworthy measure of progress.
        # ---- STAGE 5: anchor the reduced oracle for this round --------------
        if _ro is not None:
            _nlu = 0
            _eig_done = set()                    # eigenvalues whose eigenvectors are already in (compaction)
            if _tangent_sides and it > 0:
                # the atoms' tangent space at the round's design (6.33): every
                # anchor of this round adds the responses to it
                _kb = 'va' if 'va' in _tangent_sides else ('xa' if 'xa' in _tangent_sides else None)
                _kc = 'vs' if 'vs' in _tangent_sides else ('xs' if 'xs' in _tangent_sides else None)
                _ro.set_input_basis(_tangent_basis(zbest, _kb) if _kb else _ro.Phi_b,
                                    _tangent_basis(zbest, _kc) if _kc else _ro.Psi_c,
                                    per_design=True)
            if _ro.dim > a.oracle_dim_cap:
                # restart from the current design's certified peaks and the
                # watch-list eigenvalues: the basis carries directions from
                # designs long left behind
                _aa_, _nn_, _KK_, _xa_, _xs_, _va_, _vs_ = pk.unpack(zbest)
                _bb_, _cc_, *_ = D.bc(_xa_, _xs_, _va_, _vs_)
                _keep = sorted({round(float(x), 6): float(x) for x in _last_peaks}.values())   # exact frequencies (pinned LUs)
                _lams_c = [complex(x) for x in (watch_cur or [])] + ([_al0_lam] if _al0_lam is not None else [])
                _old = _ro.dim
                _ro.compact(D.cl(_bb_, _cc_), _keep, _aa_, _nn_, _KK_, lams=_lams_c, nvec_dirs=a.oracle_nvec,
                            bc=(_bb_, _cc_))
                _anchored.clear()
                for w_ in _keep:
                    _anchored[round(w_, 6)] = (np.array(_xa_, float), np.array(_xs_, float))
                _nlu += len(_keep) + len(_lams_c)
                _eig_done = set(_lams_c)
                print(f"    oracle: basis compacted {_old} -> {_ro.dim}", flush=True)
                _sweep_anchor(zbest)
            _nlu += _enrich(_last_peaks, zbest, "certified peaks")
            D._lu_pin.clear()                    # the anchors are consumed (STAGE6 6.29)
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
            # a compaction this round already factorised at these eigenvalues for
            # the same design: their eigenvectors are in the basis (STAGE6 6.29;
            # repeating them only re-deflated the same columns)
            _lams = [l_ for l_ in _lams if l_ not in _eig_done]
            _ne = _ro.enrich_eig(_clf_, _lams, _aa_, _nn_, _KK_, bc=(_bb_, _cc_)) if _lams else 0
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
                _red = _sigmas(_rc, _Wg, _aa_, _nn_, _KK_)
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
            if _lat is not None:
                # the rows' own lattice points (cached factors), sigma_1 and sigma_2 at once
                _s12 = _lat.sigmas(zbest, [_lat_mod.snap(w_, _lat_mod.ROW_LEVEL) for w_ in _W], k=2)
                _s1 = _s12[:, 0]
            else:
                _s1 = _sigmas(_cl2, _W, _aa2, _nn2, _KK2)
            _gref = float(np.max(_s1))
            _new = set()
            for _i, _w in enumerate(_W):
                _active = _s1[_i] >= _gref * (1.0 - a.sigma2_active)
                _near = _s1[_i] >= _gref * (1.0 - a.sigma2_near)
                _was = any(abs(_w - _wp) < 1e-9 for _wp in _sigma2_prev)
                if not (_active or _near or _was):
                    continue
                _p2 = (_cl2.gain_k(float(_w), _aa2, _nn2, _KK2, k=2) if _lat is None else
                       [dict(sigma=_s12[_i, 0]), dict(sigma=_s12[_i, 1])])
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
                        if (a.peak_follow == 'on' and (_ro is not None or _lat is not None)) else None)
        _prov_pre = bool(a.cert_every > 1 and _ro is not None and (it % a.cert_every != 0)
                         and it < a.outer - 1 and _ro.dim > 0 and a.prov_gate == 'reduced')
        _truncated = False                       # the inner solve was cut by --deadline (P5)
        _ins_add, _ins_sum = [], None            # the in-solve additions followed from the next round (lattice, stage E)
        if a.inner == 'sqp':
            from sqp_inner import solve_epigraph_sqp
            if a.deadline > 0:
                _pre_dur.append(time.time() - _t_round[-1])
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
                pred_rel=(1e-7 if (a.sqp_stat == 'pred' or (a.sqp_stat == 'auto' and MIMO)) else None),
                pen_adapt=(a.sqp_pen == 'adaptive'), lattice=_lat,
                # the job's deadline less the round's reserve and the final save (P5)
                deadline=((t_start + a.deadline - a.deadline_final - _round_reserve()) if a.deadline > 0 else None))
            _t_sqp_end[0] = float(inner_stats.get('t_loop_end') or time.time())   # the reserve covers the SQP's own tail
            _it_wall_max[0] = max(_it_wall_max[0], float(inner_stats.get('it_wall_max') or 0.0))
            _truncated = bool(inner_stats.get('deadline_hit'))
            if _truncated:
                print(f"    inner solve cut by the deadline after {inner_stats.get('iters')} iteration(s) (code 4): "
                      f"this round is the last of the job, exempt from the progress and stall rules", flush=True)
            _sqp_trust_prev = inner_stats.get('trust_final')
            if _lat is not None and inner_stats.get('lattice'):
                _li = inner_stats['lattice']
                print(f"    lattice: {_li['evaluations']} evaluations, {_li['gain_evals']} sigma evaluations (Newton: at "
                      f"most {_li['newton_max']} per member); line {_li['line_shifts']} shifts (at most "
                      f"{_li['line_waves_max']} waves, {_li['line_not_clear']} not clear); up to {_li['stab_rows_max']} "
                      f"stability rows (up to {_li.get('rows_dup_max', 0)} copies "
                      + (f"merged beyond 1e-8" if _lat_mod.ROW_MERGE > 1e-8 else f"within {_lat_mod.ROW_DUP:g}, kept")
                      + "); "
                      f"{_li['stab_cuts']} trial(s) across the line from an admissible iterate ("
                      + ("cut back" if _li.get('cross_policy') == 'cut' else "weighed by the merit")
                      + f"), {_li['stab_cuts_unseen']} of them by an eigenvalue without a row at the iterate"
                      + (f"; {_li['back']} return(s) to the last admissible iterate" if _li.get('back') else ''), flush=True)
                _cr = _li.get('cross')
                if _cr:
                    print(f"    lattice: crossings: past the line by up to {_cr['depth_max']:.2e} (median "
                          f"{_cr['depth_med']:.2e}); {_cr['with_row']} had a row, whose linearisation predicted "
                          + (f"a median {_cr['pred_med']:+.2e} past it ({_cr['pred_pos']} > 0: the step gave the row up)"
                             if _cr['pred_med'] is not None else "nothing")
                          + f"; {_cr['full_step']} at a full step", flush=True)
                print(f"    lattice: level set {_li['level_calls']} call(s), up to {_li['level_shifts_max']} shifts "
                      f"({_li['level_pencil_max']} with pencil eigenvalues, the rest norm disks) / {_li['level_waves_max']} "
                      f"waves, {_li['level_not_clear']} inconclusive, {_li.get('level_watched', 0)} on watched regions "
                      f"only, inner CG up to {_li['level_cg_max']} iterations", flush=True)
                # the followed members' windows (P3) and shared tops (P4), 20261010j
                print(f"    lattice: followed members: {_li.get('follow_ext', 0)} window re-centring(s), "
                      f"{_li.get('follow_edge', 0)} stop(s) on a window's edge after them "
                      f"({_li.get('follow_edge_last', 0)} at the last evaluation); "
                      f"{inner_stats.get('dedup', 0)} restart(s) from their own base (shared tops)", flush=True)
                _round_lattice_lines(_li)                  # members, level-set disks, QP rows (20261010j stage C)
            # STALL RECOVERY (2026-09-25): a solve that ended on a collapsed radius
            # (code 6) or with failed subproblems hands its quasi-Newton matrix and
            # radius to the next round no more.  The ne=100 K+B+C run sat at
            # 7.770563 for four rounds (codes 6, subproblem failures) while the same
            # design with H = I and the default radius went to 7.665533 in 25
            # iterations offline.
            if inner_stats.get('sub_failures'):
                for _sf in inner_stats['sub_failures'][:4]:
                    print(f"    QP failed: {_sf}", flush=True)
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
                if _follow_dict is not None and _lat is not None:
                    # IN-SOLVE ADDITIONS FOLLOWED AFTER THEIR ROUND (20261010j stage E,
                    # missed-peaks.md VERIFIER missed (1)): the level set's members were
                    # followed inside their solve only -- they joined Omega, not the peak
                    # set, so _follow_dict left them out and the move below skipped them:
                    # fixed rows on 2^-10 from the next round on, at the region's centre
                    # theta or Hermite estimate (the final 770632 Omega: 10 members on the
                    # exact 48 base-grid centres, ~25 on 2^-6..2^-12 points, 0.134766 beside
                    # the 0.132094 top; rows under-stating drifting tops, regions re-added,
                    # |Omega| 116 -> 214).  Now peak members: moved below to their in-solve
                    # tops (the w_star of the solve's end) under stage A's rules -- never
                    # onto an edge or slope point, one member per shared top (the nearest
                    # base) -- and followed from the next round on.  The lattice only: the
                    # reduced path (GL nominal) has the same gap and is left as it is.
                    _ins_add = sorted(set(round(float(x), 6) for x in _addf))
                    _peakset.update(_ins_add)
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
        _ins_mv, _ins_nt, _dropped = {}, set(), set()   # in-solve additions moved (key -> top) / not at a top; shared tops dropped
        if (_follow_dict or _ins_add) and best['z'] is not None:
            # move each followed member to its peak's top AT THE CANDIDATE, so
            # the next round starts on the tops (reduced model: no factorisation;
            # lattice: the rows' points at the candidate, from the SQP's stats)
            from peakfollow import follow as _follow
            _aa_, _nn_, _KK_, _xa_, _xs_, _va_, _vs_ = pk.unpack(zc)
            _bb_, _cc_, *_ = D.bc(_xa_, _xs_, _va_, _vs_)
            _rcf = _ro.make_cl(_bb_, _cc_, D.r, D.mc) if _ro is not None else None
            # the reduced model: the member's top by peakfollow.follow at the candidate (no
            # factorisation); the lattice: its top in the SQP's stats (w_star, w_top)
            _ffn = (None if _rcf is None else
                    (lambda w_: _follow(_rcf, w_, _aa_, _nn_, _KK_, float(a.follow_radius))[0]))
            Omega, _pf, _ins_mv, _ins_nt, _dropped = _round_end_follow(
                Omega, _peakset, list(_follow_dict or {}) + [k_ for k_ in _ins_add if k_ not in (_follow_dict or {})],
                _ins_add, inner_stats.get('w_star') or {}, inner_stats.get('w_top') or {}, follow_fn=_ffn)
            if _pf['moved'] or _pf['merged'] or _pf['shared'] or _pf['nottop']:
                print(f"    peak-following: {_pf['moved']} member(s) moved to their tops, {_pf['merged']} merged"
                      + (f", {_pf['shared']} on another member's top dropped" if _pf['shared'] else '')
                      + (f", {_pf['nottop']} not at a top (edge or slope) left in place" if _pf['nottop'] else ''), flush=True)
        if _ins_add:
            _ins_sum = _insolve_summary(_ins_add, _ins_mv, _ins_nt, _dropped, Omega, _peakset, inner_stats.get('enrol') or {})
            print(_ins_sum.pop('line'), flush=True)
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
                    a.wlo, a.whi = (0.0 if BFS else -_hw0), _hw0
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
        # CANDIDATES THAT CANNOT BE ACCEPTED (STAGE6 6.29): a certified value not
        # below the best (or an inconclusive certificate) is refused whatever the
        # gate says, and the next round restarts from the best.  The gate -- the
        # abscissa certificate, the line sweep, the band from the closed loop, all
        # full-model factorisations -- is skipped for it (it ran on every
        # candidate).
        _acceptable = bool(ok) and bool(np.isfinite(g)) and g < gbest * (1 - 1e-12)
        al_c, _lam_c, _clr_c, feasible = float('nan'), None, None, False
        if not _acceptable:
            print(f"    gate skipped: certified {g:.6f} cannot be accepted (best {gbest:.6f})", flush=True)
        if _acceptable:
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
                    okr, why, nsr = _gate(
                        aa, bb, cc, nn, KK, eps=a.eps,
                        shrink=a.cert_shrink, wlo=a.wlo, whi=a.whi,
                        ctrl_max=a.ctrl_max, skip=_skip)
                    if not okr and why.startswith('INCONCLUSIVE') and _wind_accept:
                        # MIMO (STAGE6 6.5): with the loops at high gain the band follows a
                        # closed-loop branch out to |Im| ~ 850 (the band grew to +-1269 at
                        # ne=100 atoms) and the line sweep runs out of shifts on it (85
                        # uncovered); the exact winding count, global in w, decides.
                        # a count that cannot run (a pre-march TailBoundError, or any failure)
                        # leaves the gate's INCONCLUSIVE standing: it used to reach the
                        # handler below, 'gate unavailable', and the alpha fallback could
                        # accept the very design the gate could not clear (2026-10-16)
                        try:
                            _o = _wind_right_of_margin(bb, cc, aa, nn, KK, _seen_c,
                                                       pa_=(D.bc(xa_, xs_, va_, vs_)[2] if BFS else None))
                        except Exception as _e:
                            _o = None
                            _pre = isinstance(_e, _wc_mod.TailBoundError)
                            n_wind_premarch[0] += int(_pre)
                            print(f"    gate: line sweep {why}; the winding count could not run "
                                  f"({'pre-march, ' if _pre else ''}{type(_e).__name__}: {_e}): the INCONCLUSIVE stands",
                                  flush=True)
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
                        a.wlo, a.whi = (0.0 if BFS else -_hw), _hw
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
        _reach = {}
        _gref = None                      # a refusal of the gate this round (G2: printed, saved as gate_fail_k.npz)
        _prefix_acc = False               # the step replaced by its certified prefix (G2's record of the outcome)
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
                _reach = {}
                _okc = None
                _gpath = (a.gate_path == 'on' or (a.gate_path == 'auto' and _lat is not None)) and a.inner == 'sqp'
                _pinf = []                              # the kept path points' SQP info (G2)
                _pts = _path_to(inner_stats.get('path'), zwatch, zc, inner_stats.get('path_info'), _pinf) if _gpath else None
                _glogs, _clog = [], []                  # the path's segment logs, the straight chord's (G2)
                _path_refused = False
                if _pts is not None and len(_pts) > 2:
                    # THE SQP'S OWN PATH (--gate-path; BFS.md 5, 2026-10-06): its accepted
                    # iterates in order, segment by segment -- short steps, each end
                    # certified on the full model by the SQP itself (lattice oracle) --
                    # where the straight segment from the last accepted design to the
                    # candidate can leave the feasible set (test plant, order 10: a
                    # midpoint at alpha -0.0458 refused a round whose every iterate was
                    # admissible)
                    _okc, watch_cand, _nq, _nsp, _reach_p, _iseg = _gate_along(_pts, watch_cur, _glogs)
                    print(f"    continuation along the SQP's path: {len(_pts) - 1} segments, "
                          f"{'certified' if _okc else 'refused at segment %d' % (_iseg + 1)} "
                          f"({_nq} tracked queries, {_nsp} splits)", flush=True)
                    if not _okc:
                        _path_refused = True
                        gate_refusal_report(_pts, _pinf, _glogs, _iseg, a.eps)
                if not _okc:
                    _okc, watch_cand, _nq2, _nsp2 = continuation_gate(
                        D, pk, zwatch, zc, watch_cur, a.eps,
                        depth=a.cont_depth, motion_frac=a.cont_motion,
                        motion_floor=a.cont_floor, kwatch=a.cont_k, pool=fullpar.POOL, reach=_reach, log=_clog)
                    if _pts is not None and len(_pts) > 2:
                        _nq += _nq2; _nsp += _nsp2
                        if not _okc and _reach_p.get('z') is not None:
                            _reach = _reach_p           # the path's certified prefix: along improving iterates
                    else:
                        _nq, _nsp = _nq2, _nsp2
                if not _okc:
                    # the straight chord's ends (G2): their own lines -- the anchor's
                    # abscissa and watch list, the candidate's abscissa and seen list
                    print(f"    gate refusal: the straight chord, |dz| {float(np.linalg.norm(zc - zwatch)):.4g}, from the "
                          f"last accepted design (alpha {(albest if albest is not None else _al0):+.7f}, watching "
                          f"{[f'{q:.6g}' for q in (watch_cur or [])[:4]]}) to the candidate (alpha {al_c:+.7f}, rightmost seen "
                          f"{[f'{q:.6g}' for q in sorted(_seen_c or [], key=lambda q: -q.real)[:3]]}); "
                          f"{sum(e_.get('verdict') not in ('certified', 'split') for e_ in _clog)} sub-segment(s) failed",
                          flush=True)
                if _path_refused or not _okc:
                    # the refusal -- the path's (the chord then certified or not) or the
                    # chord's -- for gate_fail_k.npz below, once the fallbacks decided
                    _gref = dict(round=int(it), pts=_pts, pinf=_pinf, glogs=_glogs, iseg=(_iseg if _path_refused else None),
                                 clog=_clog, chord_ok=bool(_okc), watch=list(watch_cur), z0=np.array(zwatch, float),
                                 z1=np.array(zc, float), g=float(g), al=float(al_c), seen=list(_seen_c or []))
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
                    # flagged).  Hints: the candidate's line and, 20261010j stage B (G1,
                    # verifier's correction 2), the BFS gate's tracked eigenvalues -- the
                    # tracker's offset shift finds an eigenvalue the k = 1 line can
                    # hide behind a pinned one (770633 round 1: -0.0498897; GL as before)
                    _o = _wind_right_of_margin(bb, cc, aa, nn, KK,
                                               list(_seen_c or []) + ([q for q in (watch_cand or []) if q is not None]
                                                                      if BFS else []),
                                               pa_=(D.bc(xa_, xs_, va_, vs_)[2] if BFS else None))
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
                    # the refusal stands; a pre-march TailBoundError (nothing marched) says
                    # so and why (the cap of the tail cutoff, |k| gb(W) against q)
                    _pre = isinstance(_e, _wc_mod.TailBoundError)
                    n_wind_premarch[0] += int(_pre)
                    print(f"    continuation refused; the winding count could not run "
                          f"({'pre-march, ' if _pre else ''}{type(_e).__name__}: {_e}): not accepted", flush=True)
            if not _okc and a.gate_prefix and not _prov and (_reach.get('z') is None
                                                              or float(np.linalg.norm(_reach['z'] - zwatch)) == 0.0):
                print(f"    certified prefix: none (no piece of the step certified: its first 1/{2 ** a.cont_depth}, or the "
                      f"first midpoint, already fails)", flush=True)
            if not _okc and a.gate_prefix and not _prov and _reach.get('z') is not None \
                    and float(np.linalg.norm(_reach['z'] - zwatch)) > 0.0:
                # THE CERTIFIED PREFIX (--gate-prefix; BFS.md 5, 2026-10-02).  The
                # gate certifies the segment's pieces left to right; when it gives
                # up (a crossing ahead, or the motion still too large at full
                # depth) the path up to its last certified piece is already proved
                # free of crossings.  Instead of losing the round (cluster test
                # plant, zero start: the first closing of the loop, 1464 -> 1140
                # refused), that point is the candidate: its own line certificate
                # (and the rectangle gate) and gamma certificate, accepted if it
                # improves.  The step cap is then the certified length.
                _zp = np.asarray(_reach['z'], float)
                _th = float(np.linalg.norm(_zp - zwatch) / max(np.linalg.norm(zc - zwatch), 1e-300))
                _ap, _np_, _Kp, _xap, _xsp, _vap, _vsp = pk.unpack(_zp)
                _bp, _cp, *_ = D.bc(_xap, _xsp, _vap, _vsp)
                _seen_p = []
                _alp, _lamp, _, _, _, _clrp = D.alpha_certified(_ap, _np_, _Kp, _bp, _cp, eps=a.eps,
                                                                 eigs_seen=_seen_p)
                _okp = bool(_clrp) and _alp + a.eps <= 0.0
                _why_p = f"alpha {_alp:+.4f}" + ('' if _clrp else ', line not clear')
                if _okp and a.rect_cert:
                    _okr_p, _why_r, _ = _gate(
                        _ap, _bp, _cp, _np_, _Kp, eps=a.eps,
                        shrink=a.cert_shrink, wlo=a.wlo, whi=a.whi, ctrl_max=a.ctrl_max)
                    _okp = bool(_okr_p)
                    _why_p += f"; {_why_r}"
                _gp = np.nan
                if _okp:
                    _gp, _wp, _okgp, _pkp = certified_norm(D, pk, _zp, a.wlo, a.whi, gam_lb=best['g'],
                                                           w_hint=best.get('w'))
                    _okp = bool(_okgp) and bool(np.isfinite(_gp)) and _gp < gbest * (1 - 1e-12)
                print(f"    certified prefix: {_th:.3f} of the step, {_why_p}, gamma {_gp:.6f} "
                      f"(best {gbest:.6f}): {'ACCEPTED' if _okp else 'not accepted'}", flush=True)
                if _okp:
                    zc, g, w, ok, peaks = _zp, float(_gp), float(_wp), bool(_okgp), _pkp
                    aa, nn, KK, xa_, xs_, va_, vs_ = _ap, _np_, _Kp, _xap, _xsp, _vap, _vsp
                    bb, cc = _bp, _cp
                    al_c, _lam_c, _clr_c, _seen_c = float(_alp), _lamp, _clrp, _seen_p
                    _wind_done = None
                    watch_cand = _topk_watch(list(_reach.get('watch') or []) + _seen_p, a.cont_k, a.eps)
                    _okc = True
                    _ndz_used = float(np.linalg.norm(_zp - zwatch))
                    n_prefix[0] += 1; _prefix_acc = True
                    if a.tau:
                        _tau_rho = min(_tau_rho or np.inf, _ndz_used)
                        _tau_streak = 0
                        print(f"    tau governor: cap -> {_tau_rho:.3g} (the certified prefix)", flush=True)
            if _okc and _wind_accept and _wind_done is None:
                # STAGE6 (2026-09-29): the continuation gate watches the near-line
                # eigenvalues of the certified BAND; a high loop gain can bring an
                # eigenvalue from far left to Re > 0 far OUTSIDE the band in one
                # step (MIMO atoms, ne=100: +17.1 - 111.8i, accepted, caught only by
                # the dense save veto).  The winding count is global in w: every
                # accepted candidate must have none right of Re = -eps.
                try:
                    _o = _wind_right_of_margin(bb, cc, aa, nn, KK, _seen_c, pa_=(D.bc(xa_, xs_, va_, vs_)[2] if BFS else None))
                    n_wind_accept[0] += 1
                    if not (_o is not None and _o['ok'] and _o['count'] == 0):
                        _okc = False
                        print(f"    winding check at acceptance: "
                              + ('unavailable (no plant pole data)' if _o is None else
                                 f"{'INCONCLUSIVE' if not _o['ok'] else str(_o['count']) + ' eigenvalue(s) right of Re=-eps'} "
                                 f"({_o['npts']} pts)") + ": REFUSED", flush=True)
                        n_wind_refused[0] += 1
                except _wc_mod.TailBoundError as _e:
                    # PRE-MARCH (nothing counted, nothing known against the design): as at
                    # the save, the design stays -- accepted on the gate, its count unknown
                    # (None in the veto's ledger, which counts or screens it at the save) --
                    # instead of a refusal that would repeat on every later candidate as
                    # long as the tail cannot be cut (a stall; 2026-10-16)
                    # -- on the BFS only, where that veto exists (`if BFS and a.wind` at the
                    # save): GL has no save-time veto (its dense save check refuses the whole
                    # record) and keeps its refusal, as before 9bfdee0 (the review, 2026-10-17)
                    n_wind_premarch[0] += 1
                    if BFS and a.wind:
                        print(f"    *** winding check at acceptance: no count (pre-march: {_e}): accepted on the "
                              f"gate, the save-time veto decides", flush=True)
                    else:
                        _okc = False
                        print(f"    winding check at acceptance failed (pre-march: {_e}): refused", flush=True)
                except Exception as _e:
                    _okc = False
                    print(f"    winding check at acceptance failed ({type(_e).__name__}): refused", flush=True)
            if _gref is not None:
                _gref.update(accepted=bool(_okc), count=_wind_done, prefix_accepted=bool(_okc and _prefix_acc))
                _save_gate_fail(_gref)
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
        if a.dense_audit and (D.n + D.r) <= a.dense_nmax:
            try:
                _ada, _adl = D.alpha_dense(aa, nn, KK, bb, cc)[:2]
                print(f"    dense audit: candidate alpha_dense {_ada:+.6f} at {complex(_adl):.5g} "
                      f"(chain's alpha {float(al_c):+.6f}); {'ACCEPTED' if improved else 'not accepted'}"
                      + ('  <-- ESCAPE' if improved and _ada + a.eps > 0.0 else ''), flush=True)
            except Exception as _e:
                print(f"    dense audit failed: {type(_e).__name__}", flush=True)
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
                            _okr_p, _, _ = _gate(_aa_p, _bb_p, _cc_p, _nn_p, _KK_p, eps=a.eps,
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
            if best.get('z') is not None and np.array_equal(np.asarray(best['z'], float), zc):
                _t_best = dict(z=zc.copy(), t=float(best['t']), smax=float(best['g']), rows=(inner_stats or {}).get('rows_best'))
            else:
                _t_best = None                              # the step was clipped: not the SQP's design
            _wind_best = (zbest.copy(), _wind_done) if (_wind_done is not None and _wind_done.get('ok') and _wind_done.get('count') == 0) else None
            _aw = getattr(certified_norm, 'last_all_w', None)
            _allw_best = (zbest.copy(), float(g), dict(_aw)) if (_aw and _aw.get('ok')) else None
            # CHECKPOINT: the best accepted design survives any interruption
            # (SLURM kill, node loss).  Verified state only -- this iterate
            # just passed the line certificate, the pole cap and the
            # continuation gate -- and the startup anchor re-verifies it
            # (dense at GL) before any resumed run trusts it.
            _write_ckpt(zbest, g, al_c, list(Omega) + [w] + list(peaks))
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
            # the veto's ledger (above): this design, its hints for a count (the watch
            # list and the line's seen list) and its count right of -eps when this
            # round computed one (a refused step accepted on its count, --wind-accept)
            # (prov: a reduced-model value, --cert-every; never a fallback)
            _accepted.append(dict(round=int(it), z=zbest.copy(), g=float(gbest), al=float(al_c),
                                  watch=[q for q in (watch_cur or []) if q is not None] + list(_seen_c or []),
                                  count=(dict(count=0, ok=True, npts=int(_wind_done.get('npts', 0)), how='at acceptance')
                                         if (_wind_done is not None and _wind_done.get('ok') and _wind_done.get('count') == 0)
                                         else None),
                                  how='accepted', prov=bool(_prov)))
            if BFS:
                # the ledger on disk (append_accepted): no accepted design is lost,
                # whatever the veto, the deadline or a later checkpoint does
                append_accepted(tag + '_accepted.npz', design_arrays(pk, zbest, MIMO, flags.get('loop_orders')),
                                dict(gamma=float(gbest), alpha=float(al_c), round=int(it), job=job_id(),
                                     t=round(time.time() - t_start, 1), prov=bool(_prov)))
                # the count's tail at this design, no march (BFSPlant.tail_check): W*
                # and the norms that set it, rounds before a design nears the cap
                try:
                    _tc = D.tail_check(D.bc(xa_, xs_, va_, vs_)[2], bb, cc, aa, nn, KK, -a.eps)
                    _cMs = 'none' if _tc['cM'] is None else f"{_tc['cM']:.4g}"
                    print(f"    tail: W* {_tc['wstar']:.4g} (W_T {_tc['wtail']:.4g}): ||D_c|| {_tc['kD']:.4g}, "
                          f"cL {_tc['cL']:.4g}, cM {_cMs}, pM {_tc['pM']:.4g}", flush=True)
                except Exception as _e:
                    print(f"    *** tail: {type(_e).__name__}: {_e}", flush=True)

        # adapt the trust radius: a faithful step earns a bigger one, a step
        # whose certified norm blew past t earns a smaller one
        if a.trust > 0:
            if np.isfinite(g) and g <= best['t'] * 1.5:
                trust = min(trust * 1.6, 4.0)
            else:
                trust = max(trust * 0.4, 0.02)
        newf = [x for x in ([w] + peaks)
                if all(abs(x - q) > 1e-3 for q in Omega)]
        _excess = None
        if np.isfinite(g) and np.isfinite(best['t']) and g > best['t'] * (1.0 + 10 * a.rtol):
            # EXCESS DIAGNOSIS (2026-09-22, as 'refusal:'): where the certificate's
            # witness sits relative to Omega, and whether its nearest member was a
            # followed peak.  20261010j (missed-peaks.md P7): not a refusal -- all
            # eight such rounds of 20261010g were accepted, they only gained less
            # than claimed -- and two different things: 't lag' when the rows knew
            # the value (certified within 1e-6 of the SQP's smax, best['g']: the
            # epigraph t lagged its rows), 'missed peak' when the certified value
            # exceeds smax (1 + tol), tol the in-solve level set's (--sqp-refine),
            # 'within tol' in between.  Greppable words, the line otherwise as before.
            _Oa = np.asarray(Omega, float); _jn = int(np.argmin(np.abs(_Oa - w))); _wn = float(_Oa[_jn])
            _sm = float(best.get('g', np.nan))
            _tolx = float(a.sqp_refine) if a.sqp_refine > 0 else 10 * a.rtol
            _excess = ('t lag' if np.isfinite(_sm) and g <= _sm * (1.0 + 1e-6) else
                       ('missed peak' if (not np.isfinite(_sm) or g > _sm * (1.0 + _tolx)) else 'within tol'))
            print(f"    {_excess}: witness w={w:+.4f} (certified {g:.6f} vs t {best['t']:.6f}, smax {_sm:.6f}: "
                  f"certified/smax - 1 = {g / _sm - 1.0:+.2e}); nearest member "
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
                         # the SQP's own maximum of its rows at the returned design, and what the
                         # certificate's excess over t was (20261010j, P7); the in-solve level-set
                         # decisions (last 40: level, outcome, regions, watching rows, additions)
                         smax=(float(best['g']) if best.get('g') is not None and np.isfinite(best['g']) else None),
                         excess=_excess,
                         refine_log=(inner_stats or {}).get('refine_log'),
                         insolve=_ins_sum,       # the in-solve additions: located, followed, merged (stage E)
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
        print(f"  outer {it}: |Omega|={len(Omega)}  t={best['t']:.6f}  smax={best['g']:.6f}  "
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
        if a.inner_maxit_adapt and _truncated:
            # an inner solve cut by the deadline (P5) ran fewer iterations than the
            # budget: its agreement says nothing about the budget, which the next
            # job resumes with (the checkpoint, the record) unchanged
            print(f"    inner budget kept at {_maxit_cur} (the solve was cut by the deadline)", flush=True)
        elif a.inner_maxit_adapt:
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
        if _retire_frac is not None and improved:
            # ---- RETIREMENT on the full model (--omega-retire FRAC:ROUNDS,
            # lattice; 2026-10-08), after an ACCEPTED round only (the followed
            # members sit on their tops at the candidate, which is then the
            # accepted design; a refused round leaves the streaks as they are):
            # sigma_1 at every member's lattice point on 2^-10 (a followed member
            # sits on its exact top, within 2^-11 of that point); a member below
            # FRAC x gamma for
            # ROUNDS consecutive accepted rounds leaves Omega.  The streak
            # follows a member that moved with its peak (the nearest member of
            # last round within half the follow radius).  Never retired: the
            # certified peaks of the accepted design, the sigma_2-enrolled
            # members, a member that refuted a trial design within
            # --omega-refuter-rounds, the highest member.  A peak that rises
            # again is caught by the in-solve sweep and by the certificate, which
            # add it back.
            _Wr = sorted(Omega)
            _sr = _lat.sigmas(zbest, [_lat_mod.snap(w_, _lat_mod.ROW_LEVEL) for w_ in _Wr])
            _prev = _retire_streak
            _retire_streak = {}
            for w_, s_ in zip(_Wr, _sr):
                q_ = min(_prev, key=lambda x_: abs(x_ - w_)) if _prev else None
                st_ = _prev[q_] if (q_ is not None and abs(q_ - w_) <= 0.5 * a.follow_radius) else 0
                _retire_streak[w_] = st_ + 1 if s_ < _retire_frac * gbest else 0
            _keep_r = set(round(float(x_), 6) for x_ in _peaks_of_best)
            _keep_r |= set(round(float(x_), 6) for x_ in _sigma2_prev)
            _keep_r |= {w_ for w_, r0 in _refuter.items() if (it + 1) - r0 < a.omega_refuter_rounds}
            _keep_r.add(_Wr[int(np.argmax(_sr))])
            _retired = {w_ for w_ in _Wr if _retire_streak[w_] >= _retire_rounds
                        and all(abs(w_ - q_) > 1e-3 for q_ in _keep_r)}
            if _retired:
                Omega = [w_ for w_ in Omega if w_ not in _retired]
                _peakset.difference_update(_retired)
                for w_ in _retired:
                    _retire_streak.pop(w_)
            _omega_log.append(dict(round=it, kind='retire', retired=len(_retired), after=len(Omega),
                                   below=int(sum(s_ < _retire_frac * gbest for s_ in _sr))))
            print(f"    omega retire: {_omega_log[-1]['below']} of {len(_Wr)} member(s) below {_retire_frac:g} x gamma, "
                  f"-{len(_retired)} retired after {_retire_rounds} accepted round(s) -> |Omega|={len(Omega)}", flush=True)
        if not newf and not (_progress if a.converge else improved) and not _truncated:
            # nothing new to add and no progress: enlarge Omega with the peaks
            # of the CURRENT best design instead, then stop if still nothing
            # (not after an inner solve cut by the deadline, P5: its iterations
            # say nothing about a fixed point; the job stops below anyway)
            if _ro is not None or _lat is not None:
                # the best design was certified when it was accepted; its peak list
                # (reduced fine sweep + full peak reference) is _peaks_of_best --
                # a second full certificate only to list them (level set, all-w,
                # ~17 full gains) was redundant (STAGE6 6.29)
                pk2 = list(_peaks_of_best)
            else:
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
            # nor (BFS) when the round was refused because the reduced model had
            # MISJUDGED the candidate (certified > 1 % above the model's t): the
            # flatness is the model's, not the landscape's; the witness was just
            # anchored and enrolled.  Order-10 test plant (2026-10-02): rounds 10-11
            # certified 1015.96 / 1004.41 against t = 776.07 / 820.49 (a peak at
            # w ~ 0.27 the model missed) and the rule stopped the run at 868.27.
            # At most 2 k consecutive rounds without an accepted design.
            _n_noimp = 0 if improved else _n_noimp + 1
            _mismatch = (BFS and (not improved) and bool(np.isfinite(g)) and np.isfinite(best.get('t', np.nan))
                         and g > best['t'] * 1.01 and _n_noimp < 2 * _k)
            # nor after an inner solve cut by the deadline (20261010j stage B, P5 with
            # its verifier's correction): a short last round with a small gain would
            # mark the run FINISHED (checkpoint removed) instead of deadline-stopped
            if (len(_gh_hist) > _k and (_gh_hist[-_k - 1] - _gh_hist[-1]) < _tol * _gh_hist[-1]
                    and not _prov and not _stalled and not _accel and not _refused and not _mismatch and _armed
                    and not _truncated):
                print(f"  PROGRESS STOP: {100 * (_gh_hist[-_k - 1] - _gh_hist[-1]) / _gh_hist[-1]:.3f} % over the last "
                      f"{_k} rounds (< {100 * _tol:.2f} %)"
                      + (f"; {_n_noimp} rounds without an accepted design" if _n_noimp else ""), flush=True)
                exch_progress_stop = True
                break
        if _truncated:
            # the inner solve stopped on the deadline: this round was the job's last, and
            # the run ends as at a DEADLINE STOP -- its checkpoint stays for the next job
            print(f"  DEADLINE STOP: {(time.time() - t_start) / 3600:.2f} h elapsed, the round's inner solve was cut "
                  f"by the deadline, --deadline {a.deadline / 3600:.2f} h", flush=True)
            _deadline_stop = True
            break

    # ---- final evaluation and save (same format as run_case.py) ----
    # THE SAVE-TIME WINDING VETO (BFS; 20261010j stage B, the user's decision of
    # 2026-10-14: never report an unstable design as the record).  The final
    # design's all-frequency count right of Re = -EPS -- the margin the record's
    # alpha claims; the count was right of -eps/2 and advisory, so a record could
    # carry an eigenvalue in (-eps, -eps/2), or one the k = 1 line hid behind a
    # pinned one, and the chain's checkpoints and next round rested on the gate
    # alone (gate-refusal.md).  The line is -eps as for every acceptance-time count
    # since STAGE5 22 (which closed the (-eps, -eps/2) gap there); 0 right of -eps
    # implies 0 right of -eps/2 (STAGE6 6.29), so the old field's meaning holds.
    # Not a conclusive 0 (a count > 0, INCONCLUSIVE, a failure): the record falls
    # back to the last accepted design of this job whose count is a conclusive 0
    # -- free where known (the anchor, counted at the start; a design accepted on
    # its count), else counted, at most --veto-counts of them and none the
    # deadline cannot hold -- and says so (wind_veto in the JSON, a '*** ' line).
    # No such design: nothing is saved.  The final design is written to
    # <tag>_final_<jobid> whatever the veto decides (above write_final_design).
    _wind_final, _veto, _chosen, _ck_keep, _final_path = None, None, None, None, None
    if BFS:
        _fin = next((e_ for e_ in reversed(_accepted) if np.array_equal(e_['z'], zbest)), None)
        if _fin is None:
            _fin = dict(round=None, z=zbest.copy(), g=float(gbest), al=(np.nan if albest is None else float(albest)),
                        watch=[q for q in (watch_cur or []) if q is not None], count=None, how='final')
    if BFS and a.wind:
        def _count_entry(e_):
            aa_v, nn_v, KK_v, xa_v, xs_v, va_v, vs_v = pk.unpack(e_['z'])
            bb_v, cc_v, pa_v, *_ = D.bc(xa_v, xs_v, va_v, vs_v)
            return _wind_right_of_margin(bb_v, cc_v, aa_v, nn_v, KK_v, list(e_.get('watch') or []), pa_=pa_v)

        def _tail_entry(e_):
            # the count's tail cutoff, no march (milliseconds): the veto's pre-screen
            aa_v, nn_v, KK_v, xa_v, xs_v, va_v, vs_v = pk.unpack(e_['z'])
            bb_v, cc_v, pa_v, *_ = D.bc(xa_v, xs_v, va_v, vs_v)
            return D.tail_check(pa_v, bb_v, cc_v, aa_v, nn_v, KK_v, -a.eps)
        # a further count only while it fits before the job's deadline with a quarter
        # of --deadline-final left for the rest of the save (the record's gain curve,
        # the figures: ~3 min on the test plant); never a provisional design
        _chosen, _wind_final, _veto = winding_veto(
            _fin, [e_ for e_ in _accepted if not e_.get('prov') and not np.array_equal(e_['z'], zbest)],
            _count_entry, -a.eps, cap=int(a.veto_counts),
            time_ok=((lambda: (time.time() - t_start) + 1.25 * _wind_sec[0] + 0.25 * a.deadline_final <= a.deadline)
                     if a.deadline > 0 else None), tail_fn=_tail_entry)
        # which design a checkpoint may hold now (ckpt_after_veto): the final one
        # after a pass or a PRE-MARCH veto, the fallback otherwise, the anchor
        # when nothing is saved
        _ck_keep = ckpt_after_veto(_veto, _fin, _chosen, next((e_ for e_ in _accepted if e_.get('round') == -1), None))
        if _ck_keep[0] is not _fin:
            # APPLIED AT ONCE (the review of 2026-10-17): until here the checkpoint is
            # the last acceptance's file, the final design the veto just refused, and
            # whatever ends the save from now on -- the time limit inside the record's
            # certificate or figures, an exception, the NOT SAVING of an unstable record
            # below, which returns before any checkpoint line -- left it there for the
            # next job.  The deadline stop rewrites it below with the end-of-job state,
            # a finished run removes it.
            _ck_e, _ck_why = _ck_keep
            if _ck_e is not None:
                _write_ckpt(_ck_e['z'], _ck_e['g'], _ck_e.get('al', np.nan), list(Omega),
                            note_=f"checkpoint (the save-time veto{', NOT SAVING' if _chosen is None else ''}; {_ck_why})")
            else:
                for _c in (tag + '_ckpt.npz', tag + '_ckpt.json'):
                    try:
                        if os.path.exists(_c):
                            os.remove(_c)
                    except OSError:
                        pass
            print(f"    checkpoint: {_ck_why}" + (f" (gamma {_ck_e['g']:.6f})" if _ck_e is not None else '')
                  + ", set at the veto", flush=True)
    if BFS:
        _final_path = write_final_design(
            tag, _fin, (_chosen if a.wind else _fin), _veto, lambda z_: design_arrays(pk, z_, MIMO, flags.get('loop_orders')),
            extra=dict(_loop_state(), wlo=float(a.wlo), whi=float(a.whi),
                       active_freqs=np.array(sorted(set(round(float(x), 6) for x in Omega)))))
    if BFS and a.wind and _chosen is None:
        print(f"*** WINDING VETO: no accepted design of this job back to the anchor has a conclusive count of 0 "
              f"right of Re={-a.eps:g}: NOT SAVING (the veto is in {os.path.basename(tag)}_veto.json)", flush=True)
        try:
            json.dump(_js(dict(case=a.case, wind_veto=_veto, history=hist, code_version=_code_version(),
                               seconds=round(time.time() - t_start, 1))), open(tag + '_veto.json', 'w'), indent=1)
        except Exception as _e:
            print(f"    veto record failed: {_e!r}", flush=True)
        # the checkpoint follows the deadline stop's rule below: kept after a
        # PRE-MARCH veto (refreshed at a deadline stop with the end-of-job state),
        # else moved back to the anchor or removed at the veto above -- never left
        # at a vetoed design
        _ck_e, _ck_why = _ck_keep
        if _ck_e is _fin:
            if _deadline_stop:
                _write_ckpt(_fin['z'], _fin['g'], _fin.get('al', np.nan), list(Omega),
                            note_=f'checkpoint (the deadline stop, NOT SAVING; {_ck_why})')
            print(f"    checkpoint: {_ck_why}", flush=True)
        return
    if _veto is not None:
        zbest, gbest = np.array(_chosen['z'], float), float(_chosen['g'])
        watch_cur = [q for q in (_chosen.get('watch') or []) if q is not None]
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
            _okf, why_final, _ = _gate(
                arow, b, c, nvec, K, eps=a.eps,
                shrink=a.cert_shrink, wlo=a.wlo, whi=a.whi,
                ctrl_max=a.ctrl_max)
        except Exception:
            pass
        print(f"NOT SAVING: final design is unstable (alpha={al:+.4f}; "
              f"{why_final})", flush=True)
        return
    if _deadline_stop:
        # A DEADLINE STOP keeps the checkpoint for the next job: refreshed here with
        # the end-of-job loop state (inner budget, trust radius, peak members, tau
        # cap: the acceptance-time file held the state before that round's
        # adaptation and any later round's) and, after a veto, the fallback design
        # -- the next job must not resume from a vetoed one (20261010j stage B) --
        # unless the veto was PRE-MARCH (2026-10-16, ckpt_after_veto): nothing was
        # counted, so the final design stays and the next job's start count decides
        # (20261010n: the fallback here erased five decentral jobs' progress)
        if _ck_keep is not None and _ck_keep[0] is _fin and _veto is not None:
            _write_ckpt(_fin['z'], _fin['g'], _fin.get('al', np.nan), list(Omega),
                        note_=f'checkpoint (the deadline stop; {_ck_keep[1]}; the record is the fallback)')
            print(f"    checkpoint: {_ck_keep[1]} (gamma {_fin['g']:.6f}); the record is the veto's fallback", flush=True)
        else:
            _write_ckpt(zbest, gbest, al, list(Omega),
                        note_=('checkpoint (the deadline stop' + (', the winding veto\'s fallback' if _veto else '') + ')'))
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
    # FINAL BAND CHECK (STAGE6 6.13, sparse): the whole band tiled, the far
    # candidates refined, the located set compared with the winding count right
    # of -band_recut.  A resonance the loop's (edge-strip) band updates missed
    # widens the band, and the design is certified again over it.  Run BEFORE
    # the winding certificate (STAGE6 6.29): a complete located set right of
    # -band_recut with none right of -eps/2 is that certificate.
    band_check = {}
    _band_lam_final = []                  # the located set of the final band check
    if BFS:
        # gamma on the band [0, b] (the level-set certificate, or the sampled
        # maximum with --bfs-sampled > 0); stability over ALL frequencies by
        # the M-free winding count below; the final band check is not run
        print(f"BFS: gamma is the {'sampled maximum' if certified_norm.sampled else 'certified level-set value'} "
              f"on [0, {a.whi:g}]; stability: the winding count over all frequencies (BFS.md 4)", flush=True)
    if a.ctrl_imcap == 'off' and not BFS:
        try:
            _hwf = band_from_closedloop(arow, nvec, K, b, c, count=True)
            band_check = {k_: v_ for k_, v_ in _band_last['info'].items() if k_ in ('located', 'count', 'complete')}
            _band_lam_final = list(_band_last['lam'])
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
    # WINDING CERTIFICATE -- the UNCONDITIONAL test.  Everything above is
    # either conditional (line sweep + continuation) or dense (does not exist
    # at cavity scale).  The argument principle on the scalar return
    # difference f = 1 - k(s)G(s) counts closed-loop eigenvalues right of
    # Re = -eps/2 exactly, matrix-free, with the near-line eigenvalues the
    # sweep already harvested serving as resolution hints for the tracker.
    wind_n = None
    wind_info = None
    if a.wind and BFS:
        # the count right of -eps of the design saved: the veto's above (the M-free
        # count, bfs/plant.BFSPlant.winding: plant data from the inertia-certified
        # enclosure, H^1 tail for the wall-shear sensor) -- counted there, or known
        # for this design (the anchor's count at the start, a count at acceptance)
        wind_n = 0
        wind_info = dict(count=0, line=float(-a.eps), npts=int(_wind_final.get('npts', 0) or 0),
                         wtail=float(_wind_final.get('wtail', np.nan)), wstar=float(_wind_final.get('wstar', np.nan)),
                         p_plant=int(_wind_final.get('p_plant', -1)),
                         seconds=_wind_final.get('seconds'), source=_wind_final.get('source'))
        print(f"winding certificate: 0 eigenvalue(s) right of Re={-a.eps:+.4f} over all frequencies "
              f"({wind_info['source']}; {wind_info['npts']} pts"
              + (f", tail from |w| = {wind_info['wtail']:.1f}" if np.isfinite(wind_info['wtail']) else '')
              + (f", {wind_info['seconds']:.0f} s" if wind_info['seconds'] is not None else '')
              + ") -- UNCONDITIONALLY stable with the margin eps (so with eps/2 too, STAGE6 6.29)", flush=True)
    if a.wind and not BFS:
        # 0 right of -eps at acceptance (the same design) implies 0 right of
        # -eps/2: the save-time count is then not repeated (STAGE6 6.29)
        _implied = _wind_best is not None and np.array_equal(_wind_best[0], zbest)
        # likewise the final band check: EVERY eigenvalue right of -band_recut
        # located (their number equals the winding count there) and none of
        # them right of -eps/2
        _implied_band = (band_check.get('complete') and -a.band_recut <= -0.5 * a.eps
                         and all(complex(z_).real < -0.5 * a.eps for z_ in _band_lam_final))
        if _implied:
            wind_n = 0
            print(f"winding certificate: 0 eigenvalues right of Re={-0.5 * a.eps:+.4f} -- implied by the "
                  f"acceptance-time count right of Re={-a.eps:+.4f} ({_wind_best[1]['npts']} pts) -- "
                  f"UNCONDITIONALLY stable with eps/2 margin", flush=True)
        elif _implied_band:
            wind_n = 0
            print(f"winding certificate: 0 eigenvalues right of Re={-0.5 * a.eps:+.4f} -- implied by the "
                  f"final band check (all {band_check.get('count')} right of Re={-a.band_recut:g} located, "
                  f"rightmost {max([complex(z_).real for z_ in _band_lam_final], default=-np.inf):+.4f}) -- "
                  f"UNCONDITIONALLY stable with eps/2 margin", flush=True)
        else:
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
    if (_ro is not None and _ro.dim > 0) or _lat is not None:
        # the record's value is the CERTIFIED gbest (and the all-w certificate
        # below); D.hinf's seeded Newton with finite differences (tens of full
        # factorisations) only fed max(gam, gbest) and the active count, which
        # the reduced (or lattice) gain curve below gives (STAGE6 6.29)
        gam, clh, nact = float(gbest), D.cl(b, c), None
    else:
        gam, _, _, clh, _, nact = D.hinf(arow, nvec, K, b, c)
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
    if _lat is not None:
        # --oracle lattice: the full model on the ranks, at the lattice points of
        # 2^-6 on the band (w >= 0: real plant) -- serially here it was ~1700
        # factorisations (37 min on the test plant, hours on the full mesh)
        ws = _lat.grid(max(float(a.wlo), 0.0), float(a.whi), 6)
        gain = _lat.sigmas(zbest, ws)
        _gain_model = 'full (lattice)'
    else:
        gain = _sigmas(_clg, ws, arow, nvec, K)
    if nact is None:                          # local maxima within 2e-3 of the top (D.hinf's tol_active)
        _i = np.arange(1, len(gain) - 1)
        _loc = _i[(gain[_i] >= gain[_i - 1]) & (gain[_i] >= gain[_i + 1])]
        nact = int(np.count_nonzero(gain[_loc] >= gain.max() * (1 - 2e-3))) if _loc.size else 1
    # ACTIVITY RELATIVE TO t, END POINTS INCLUDED (20261010j; omega-growth.md P5,
    # missed-peaks.md P7).  n_active (kept as it was) counts INTERIOR grid maxima
    # within 2e-3 of the curve's maximum: 770632's record read 1 (one excursion at
    # 1.045 t over a plateau of 8 ripple tops at 0.99-1.0 t), 770629's 0 (its
    # maximum sits at the end point w = 0).  Here: the grid's local maxima with
    # the end points, within 2e-3 of gamma and of the SQP's t at the best design's
    # round, and the rows within 1e-3 / 1e-2 of that t.
    active = None
    try:
        _gn = np.asarray(gain, float); _wsn = np.asarray(ws, float)
        _j = np.arange(len(_gn))
        _lm = _j[((_j == 0) | (_gn >= np.roll(_gn, 1))) & ((_j == len(_gn) - 1) | (_gn >= np.roll(_gn, -1)))]
        active = dict(grid_max_w=[round(float(x), 6) for x in _wsn[_lm[np.argsort(-_gn[_lm])][:12]]],
                      grid_max_v=[round(float(x), 6) for x in np.sort(_gn[_lm])[::-1][:12]],
                      n_gamma=int(np.count_nonzero(_gn[_lm] >= gam * (1 - 2e-3))),
                      ends=[round(float(_gn[0]), 6), round(float(_gn[-1]), 6)])
        if _t_best is not None and np.array_equal(_t_best['z'], zbest):
            _tt = float(_t_best['t'])
            active.update(t=_tt, smax=float(_t_best['smax']),
                          n_t=int(np.count_nonzero(_gn[_lm] >= _tt * (1 - 2e-3))))
            _rb = _t_best.get('rows')
            if _rb and _rb.get('s'):
                _sv = np.asarray(_rb['s'], float)
                active.update(rows_1e3=int(np.count_nonzero(_sv >= _tt * (1 - 1e-3))),
                              rows_1e2=int(np.count_nonzero(_sv >= _tt * (1 - 1e-2))), rows=int(_sv.size))
        print(f"activity: {active['n_gamma']} grid maxima (end points included) within 0.2 % of gamma"
              + (f", {active['n_t']} of t = {active['t']:.6f}" if 't' in active else '')
              + (f"; rows within 0.1 % / 1 % of t: {active['rows_1e3']} / {active['rows_1e2']} of {active['rows']}"
                 if 'rows' in active else '')
              + f"; sigma at the band's ends {active['ends'][0]:.6f}, {active['ends'][1]:.6f}", flush=True)
    except Exception as _e:
        active = dict(error=f"{type(_e).__name__}: {str(_e)[:60]}")
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
    # BFS: gamma is the level-set value on the band (certified, or the sampled maximum with
    # --bfs-sampled N); beyond the band nothing is checked (BFS.md 4)
    all_w = (dict(certified=not a.bfs_sampled, sampled=bool(a.bfs_sampled), band=[0.0, float(a.whi)],
                  beyond_band='not checked') if BFS else {})
    if _allw_best is not None and np.array_equal(_allw_best[0], zbest) and gam <= _allw_best[1] * (1.0 + 1e-12):
        # the best design's own all-w certificate, at acceptance, at this level:
        # not repeated (STAGE6 6.29)
        _aw = _allw_best[2]
        all_w = dict(tail_W=float(_aw.get('W', np.nan)), gamma=float(_allw_best[1]), certified=True,
                     calls=0, reused='acceptance')
        print(f"sparse all-frequency certificate: from acceptance (tail bound W = {all_w['tail_W']:.2f}, "
              f"CERTIFIED at {_allw_best[1]:.6f}) -- gamma holds over ALL w", flush=True)
    if not all_w:
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
            if MIMO and not BFS:
                prev_ok = ('alpha_dense' in pz.files and np.isfinite(float(pz['alpha_dense']))
                           and float(pz['alpha_dense']) + a.eps <= 0.0)
            elif a.dense_check and (D.n + D.r) <= a.dense_nmax:
                import warmstart as _ws
                prev_ok = _ws.is_stable(D, tag + '.npz', eps=a.eps,
                                        dense=True)
            elif 'alpha_dense' in pz.files and not (BFS and not np.isfinite(float(pz['alpha_dense']))):
                prev_ok = float(pz['alpha_dense']) + a.eps <= 0.0
            elif BFS:
                # the BFS record has no dense abscissa (nan); its alpha is the line
                # certificate's, its stability the all-frequency count (2026-10-02:
                # a resumed run reported its stable start as failing re-verification)
                _wc = None
                try:
                    _wc = json.load(open(tag + '.json')).get('wind_count')
                except Exception:
                    pass
                prev_ok = float(pz['alpha']) + a.eps <= 0.0 and _wc in (0, None)
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
             **_atoms_record_keys(D, _apa, _aps),
             va=(va if va is not None else np.zeros(0)),
             vs=(vs if vs is not None else np.zeros(0)),
             psi_a=pa, psi_s=ps, gamma=gam, alpha=al, **(_pos_start if BFS else {}),
             **(dict(Bc=np.asarray(arow[3]), loop_orders=np.array(flags['loop_orders']), m=int(flags['m']),
                     p=int(flags['p']), struct=flags['struct']) if MIMO else {}),
             alpha_dense=(al_d if al_d is not None else np.nan),
             n_active=nact,
             omega_grid=ws, gain=gain, gain_model=_gain_model, active_freqs=np.array(sorted(Omega)),
             # what a checkpoint restores and the record lacked (20261010j stage B,
             # omega-growth.md P6): CONTINUE=1 copies this file to the checkpoint, and the
             # resume reads the loop state and the band from it; the code that wrote it
             wlo=float(a.wlo), whi=float(a.whi), **_loop_state())
    summ = dict(case=a.case, start=a.start, ne=a.ne, n=D.n, NZ=len(z) + 1,
                gamma=float(gam), alpha=float(al),
                alpha_dense=(float(al_d) if al_d is not None else None),
                gamma_dense_all_w=gam_dense_all,
                band_incomplete=int(n_band_incomplete[0]), band_check=band_check, all_w=all_w,
                all_w_outside_band=int(n_allw_outside[0]),
                eps=float(a.eps),
                # BFS since 20261010j stage B: the count right of -eps (wind_line), the
                # save-time veto (None: the final design passed); GL: right of -eps/2
                wind_count=wind_n, wind_info=wind_info, wind_line=(float(-a.eps) if BFS else float(-0.5 * a.eps)),
                wind_veto=(_js(_veto) if _veto is not None else None),
                final_file=(os.path.basename(_final_path) if _final_path else None),
                code_version=_code_version(),
                loop_state=dict(maxit_cur=int(_maxit_cur), sqp_trust=float(_sqp_trust_prev or 0.0), peak_members=len(_peakset),
                                tau_rho=(float(_tau_rho) if _tau_rho is not None else None), tau_streak=int(_tau_streak)),
                sym=bool(a.sym),
                sym_branch=(float(flags.get('sym_branch', 1.0))
                            if a.sym else None),
                n_active=int(nact), active=active,
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
                omega_log=(_omega_log if (a.omega_policy == 'an' or _retire_frac is not None) else None),
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
                gate_located=int(n_located[0]), gate_prefix_accepts=int(n_prefix[0]), witness_invariant_violations=int(n_witness_bad[0]), closed_loop_pair_rounds=int(n_clpair[0]), cont_track=a.cont_track, cont_fallbacks=int(n_cont_fallback[0]),
                cert_disks=dict(untrusted=int(csp.N_DISK_UNTRUSTED), retried=int(getattr(csp, 'N_DISK_RETRIED', 0)), oversized=int(getattr(csp, 'N_DISK_OVERSIZED', 0)),
                                bisected=int(getattr(csp, 'N_DISK_BISECTED', 0)), uncovered=int(getattr(csp, 'N_DISK_UNCOVERED', 0)), climb_failed=int(csp.N_CLIMB_FAILED)),
                exchange_converged=bool(exch_converged),
                exchange_fixed_point=bool(exch_fixed_point), exchange_progress_stop=bool(exch_progress_stop),
                exchange_stalled=bool(exch_stalled),
                ladder=[], xa=(np.atleast_1d(xa).astype(float).tolist() if MIMO or BFS else float(xa)),
                xs=(np.atleast_1d(xs).astype(float).tolist() if MIMO or BFS else float(xs)),
                norm_psi_a=([D.norm_M(pa[:, j]) for j in range(pa.shape[1])] if MIMO else D.norm_M(pa)),
                norm_psi_s=([D.norm_M(ps[:, j]) for j in range(ps.shape[1])] if MIMO else D.norm_M(ps)),
                wind_accept=dict(on=bool(_wind_accept), checks=int(n_wind_accept[0]), refused=int(n_wind_refused[0]),
                                 gate_inconclusive_cleared=int(n_gate_wind[0]), pre_march=int(n_wind_premarch[0])),
                mimo=(dict(m=int(flags['m']), p=int(flags['p']), struct=flags['struct'], loop_orders=list(flags['loop_orders']),
                           start=a.mimo_start, D_c=[[str(v) for v in row] for row in np.asarray(K).reshape(flags['m'], flags['p'])])
                      if MIMO else None),
                eps_stab=a.eps, sob_len=a.sob_len, parent=a.parent,
                parent_gamma=None, spike_suspect=bool(al > -0.01),
                gain_peak_over_median=float(gain.max() /
                                            max(np.median(gain), 1e-30)),
                history=hist, deadline_stop=bool(_deadline_stop))
    json.dump(summ, open(tag + '.json', 'w'), indent=1)
    # a deadline stop is not the end of the run: the checkpoint stays, so the
    # re-submitted job resumes from it (BFS.md 7); a finished run removes it
    for _c in (() if _deadline_stop else (tag + '_ckpt.npz', tag + '_ckpt.json')):
        try:
            if os.path.exists(_c):
                os.remove(_c)
        except OSError:
            pass
    print(fullpar.report(), flush=True)
    if BFS:
        print(f"full-model pool by task: {fullpar.by_task(95)}", flush=True)
        from bfs import mumps_lu as _ml
        print(_ml.report(), flush=True)
        if _lat is not None:
            # the lattice's ledger at the end too: the last round was unledgered
            # (770628; 20261010j stage C)
            print(f"  [ledger] {_lat.report()} (at the end, {time.time() - t_start:.0f} s)", flush=True)
            for _ll in _lat.report_lines():
                print(f"  [ledger] {_ll}", flush=True)
        if _ro is not None and _red_mod.ROW_MPI:
            print(_red_mod.rows_report(), flush=True)
    if BFS and a.post:
        # the figures in this job and on this pool (bfs/postproc.run_post): no
        # second MPI launch -- the cluster's controller refused one (srun socket
        # timeout) and the job then hung until its time limit (2026-10-02)
        try:
            from bfs import postproc as _pp
            _pp.run_post(D, tag + '.npz')
        except Exception as _e:
            print(f"postproc failed non-fatally: {_e!r} (the record is saved; bfs/slurm/postproc.slurm redoes it)",
                  flush=True)
    fullpar.stop()
    print(json.dumps({k: summ[k] for k in
                      ('case', 'gamma', 'alpha', 'n_active', 'outer',
                       'nOmega', 'calls', 'seconds')}), flush=True)


if __name__ == "__main__":
    main()
