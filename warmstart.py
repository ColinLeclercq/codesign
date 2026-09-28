"""
warmstart.py — choose a warm start that is STABLE, RELEVANT and BETTER.

Three separate mistakes cost us several days of cluster time, all of them
because this logic lived in shell scripts and was re-implemented ad hoc:

  * resuming from an UNSTABLE design.  The old result directories contain
    closed loops with alpha up to +4.3e6.  They look attractive to a
    "lowest gamma wins" search precisely BECAUSE they are unstable -- pushing
    poles into the right half plane buys a low sigma_1 on the imaginary axis.
  * resuming a CONTROLLER case from a controller-free parent.  `KG+FBR+FCR`
    resumed from `FBR+FCR`, inherited its fields but a zero gain, and therefore
    restarted from the OPEN LOOP at gamma = 882.88 -- then spent twenty outer
    iterations crawling back and still finished worse than its own previous
    result.
  * a stale worse result blocking a better one, because the no-regress guard
    compared gamma alone.

`select` applies one ranking, in the code, with the stability filter built in.
"""
from __future__ import annotations
import glob
import json
import os

import numpy as np


def _tag(case):
    return case.replace('+', '_')


def is_stable(D, npz, eps=1e-3, dense=True):
    """Is this stored design a STABLE closed loop?

    At GL scale a dense eigensolve (n ~ 400) costs about a second and is
    definitive, which is what we want when deciding where to spend an hour of
    optimisation.  At cavity scale set dense=False and the stored alpha is
    trusted instead -- acceptable only because the run that produced it now
    refuses to save an unstable design.
    """
    try:
        d = np.load(npz, allow_pickle=True)
        if 'd' not in d.files:
            return False
        if not dense:
            return float(d['alpha']) + eps <= 0.0
        from scipy.linalg import eig
        import controller as ctl
        n = D.n
        r = len(np.atleast_1d(d['d']))
        arow = (d['d'], d['e'], d['f'])
        nv = np.asarray(d['nvec']); Kf = complex(d['K'])
        va = d['va'] if d['va'].size else None
        vs = d['vs'] if d['vs'].size else None
        b, c = D.bc(float(d['xa']), float(d['xs']), va, vs)[:2]
        Ad = np.asarray(D.A.todense()); Md = np.asarray(D.M.todense())
        Ac = ctl.A_of(*arow); Bc = np.full(r, 1.0, complex)
        Acl = np.zeros((n + r, n + r), complex)
        Mcl = np.zeros((n + r, n + r), complex)
        Acl[:n, :n] = Ad + Kf * np.outer(b, np.conj(c))
        Acl[:n, n:] = b.reshape(-1, 1) @ nv.reshape(1, r)
        Acl[n:, :n] = Bc.reshape(-1, 1) @ np.conj(c).reshape(1, n)
        Acl[n:, n:] = Ac
        Mcl[:n, :n] = Md; Mcl[n:, n:] = np.eye(r)
        lam = eig(Acl, Mcl, right=False); lam = lam[np.isfinite(lam)]
        return float(lam.real.max()) + eps <= 0.0
    except Exception:
        return False


def has_controller(npz):
    """Does the stored design actually USE its controller?

    A controller-free parent stores nvec = 0 and Dc = 0, so a controller case
    resuming from it starts OPEN LOOP.  That is the trap that sent
    KG+FBR+FCR back to gamma = 882.88.
    """
    try:
        d = np.load(npz, allow_pickle=True)
        return (float(np.max(np.abs(d['nvec']))) > 1e-12
                or abs(complex(d['K'])) > 1e-12)
    except Exception:
        return False


def is_ancestor(parse_case, p, c):
    """Is case `p` a structural parent of `c` (frees a subset of the groups)?

    REALNESS MATTERS and the group flags do not carry it: parse_case sets FB/FC
    identically for FB and FBR, so `FB+FC` looked like a valid ancestor of
    `FBR+FCR`.  Taking a COMPLEX field design and discarding its imaginary part
    gave gamma = 8.52 instead of 4.35, and the run never recovered -- it ended
    at alpha = +0.75 and was (correctly) refused by the save gate.

    A real-field case may inherit from a real-field one only.  A complex-field
    case may inherit from either, since a real field IS a complex field with
    zero imaginary part.
    """
    try:
        fp, fc = parse_case(p), parse_case(c)
    except Exception:
        return False
    for key in ('K', 'B', 'C', 'FB', 'FC'):
        if fp.get(key) and not fc.get(key):
            return False
    for key, real in (('FB', 'real_fb'), ('FC', 'real_fc')):
        if fc.get(key) and fc.get(real) and not fp.get(real):
            return False          # child is REAL, parent is COMPLEX
    return True


def select(D, case, ne, dirs, parse_case, eps=1e-3, dense=True, verbose=True):
    """Best warm start for `case`, or None.

    Ranking, best first:
      lowest GAMMA wins, with the tier as a tie-break only:
      0. the case's own previous result,
      1. an ancestor with an active controller,
      2. any other ancestor,
      3. an ancestor without a controller when this case needs one.  Unstable candidates are filtered out
    first, whatever their gamma.

    Returns (path, source_case).  The SOURCE CASE matters: only the groups the
    source actually optimised may be inherited.  Returning a bare path let the
    caller assume the source shared this case's flags, so a controller case
    resuming from a controller-free parent read that parent's ZERO controller
    and restarted from the open loop.
    """
    want_ctrl = bool(parse_case(case).get('K'))
    cands = []
    for d_ in dirs:
        if not os.path.isdir(d_):
            continue
        for jf in glob.glob(os.path.join(d_, f"*_s*_ne{ne}.json")):
            try:
                m = json.load(open(jf))
            except Exception:
                continue
            other = m.get('case')
            if other is None:
                continue
            npz = jf[:-5] + '.npz'
            if other == case:
                tier = 0
            elif is_ancestor(parse_case, other, case):
                tier = 1 if (want_ctrl and has_controller(npz)) else 2
            else:
                continue
            if want_ctrl and tier == 2 and not has_controller(npz):
                tier = 3          # usable, but starts from an open loop
            cands.append((tier, float(m['gamma']), npz, other))
    # Rank by GAMMA first, tier only as a tie-break.
    #
    # Ranking by tier first meant a case's OWN previous result always won, even
    # when an ancestor was far better: K+FB+FC preferred its own 5.5967 over
    # FB+FC's 4.3499 and duly finished at 5.5967 again.  Since the caller now
    # inherits only the groups the SOURCE optimised, taking a controller-free
    # ancestor is safe -- the child keeps its own controller seed -- so there is
    # no reason to prefer a worse start.
    cands.sort(key=lambda x: (x[1], x[0]))
    for tier, g, npz, other in cands:
        if not is_stable(D, npz, eps=eps, dense=dense):
            if verbose:
                print(f"  warm-start candidate {os.path.basename(npz)} "
                      f"(gamma={g:.4f}) REJECTED: unstable", flush=True)
            continue
        if verbose:
            note = {0: 'own previous result',
                    1: 'ancestor with an active controller',
                    2: 'ancestor',
                    3: 'ancestor WITHOUT a controller (starts open loop)'}[tier]
            print(f"  warm start: {os.path.basename(npz)} "
                  f"(case {other}, gamma={g:.4f}) -- {note}", flush=True)
        return npz, other
    if verbose:
        print("  warm start: none usable -- cold start", flush=True)
    return None, None


def select_controller(D, case, ne, dirs, parse_case, eps=1e-3, dense=True,
                      verbose=True):
    """Best stored CONTROLLER to seed `case` with, when its field parent has none.

    Inheriting a controller-free parent leaves Dc = 0, i.e. an open loop:
    KG+FBR+FCR resumed from FBR+FCR started at gamma = 882.88 instead of 5.50.
    The fields should come from the parent and the CONTROLLER from somewhere
    that actually has one.

    Compatibility is by controller STRUCTURE, not by case identity: a static
    gain (KG/KGC) may only borrow from another static-gain design, because a
    dynamic controller's (d, e, f, nvec) would be discarded by the packer and
    only its Dc kept -- which is usually a poor gain on its own.  A dynamic
    case may borrow from any controller-bearing design.
    """
    mode = parse_case(case).get('ctrl_mode', 'full')
    want_static = mode in ('gain', 'gainr')
    cands = []
    for d_ in dirs:
        if not os.path.isdir(d_):
            continue
        for jf in glob.glob(os.path.join(d_, f"*_s*_ne{ne}.json")):
            try:
                m = json.load(open(jf))
            except Exception:
                continue
            other = m.get('case')
            if other is None:
                continue
            try:
                fo = parse_case(other)
            except Exception:
                continue
            if not fo.get('K'):
                continue
            npz = jf[:-5] + '.npz'
            if not has_controller(npz):
                continue
            other_static = fo.get('ctrl_mode', 'full') in ('gain', 'gainr')
            if want_static != other_static:
                continue          # structures are not interchangeable
            cands.append((float(m['gamma']), npz, other))
    cands.sort()
    for g, npz, other in cands:
        if not is_stable(D, npz, eps=eps, dense=dense):
            continue
        if verbose:
            print(f"  controller seed: {os.path.basename(npz)} "
                  f"(case {other}, gamma={g:.4f})", flush=True)
        return npz, other
    return None, None
