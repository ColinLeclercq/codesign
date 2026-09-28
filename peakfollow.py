"""peakfollow.py -- Omega rows that follow their peak (STAGE5 9; Newton 2026-09-24).

A working-set member placed at a peak's top one round sits on its flank the
next: narrow resonances drift with the design.  A following row re-locates
the local maximum of sigma_1 near its member before evaluating; by the
envelope theorem the row's gradient is the ordinary one taken at the top w*.

Located by the SAFEGUARDED NEWTON method the frozen code has always used for
peaks (closedloop_sparse.ClosedLoopResolvent.peak): the analytic
d sigma/d w ('gprime', from the singular vectors) and a finite-difference
second derivative, a trust radius, step halving on non-increase.  From the
previous top it converges in 1-2 iterations to machine precision; the golden
section used for a while here ignored the derivative (11-16 evaluations per
peak, window-limited accuracy).  Returns the top AND the gain evaluation there
(vectors included), so the caller's row needs no further evaluation.
"""
from __future__ import annotations
import numpy as np


def follow(cl, w, arow, nvec, K, radius, hint=None, fd=1e-5, itmax=8, gtol=1e-10, nscan=21, state=None):
    """... (see module doc).  2026-09-24: Newton climbs to the NEAREST local
    maximum; where two narrow peaks share a window (the K+B+C cluster near
    w ~ -2.45) a member followed the lower one while its neighbour rose beside
    it (ne=100 rounds 4-7: model exact at the witness to 1e-14, rows 2-4 % low).
    A value-only scan of the window first (nscan points, reduced model, warm),
    then Newton from the best of the scan and the previous top.

    2026-09-25 (cost): with nscan <= 2 and a previous top (`hint`), Newton starts
    AT the previous top (no value comparison with the member's base), and when
    `state` (a per-member dict) is given the second derivative is carried --
    from the previous call, then by secant updates on gprime -- so a warm call
    costs one evaluation per Newton iteration instead of three; the finite
    difference is taken only when no curvature is known or a carried one fails
    to give ascent.  (48 followed members made following 2/3 of a K+FB+FC
    evaluation.)"""
    w0 = float(w)
    hint_ok = hint is not None and abs(float(hint) - w0) < radius
    starts = [w0] + ([float(hint)] if hint_ok else [])
    if nscan and nscan > 2:
        grid = np.linspace(w0 - radius, w0 + radius, int(nscan))
        vals = [cl.gain(float(g), arow, nvec, K, want_vectors=False)['sigma'] for g in grid]
        starts.append(float(grid[int(np.argmax(vals))]))
        x = max(starts, key=lambda q: cl.gain(q, arow, nvec, K, want_vectors=False)['sigma'])
    elif hint_ok:
        x = float(hint)                                  # warm: the previous top
    elif len(starts) > 1:
        x = max(starts, key=lambda q: cl.gain(q, arow, nvec, K, want_vectors=False)['sigma'])
    else:
        x = w0
    e = cl.gain(x, arow, nvec, K)
    best, xb = e, x
    trust = min(0.05, radius)
    g2 = state.get('g2') if state is not None else None
    carried = g2 is not None
    for _ in range(itmax):
        g1 = e['gprime']
        if abs(g1) <= gtol * max(1.0, e['sigma']):
            break
        if g2 is None or not (g2 < 0):
            gp = cl.gain(x + fd, arow, nvec, K)['gprime']
            gm = cl.gain(x - fd, arow, nvec, K)['gprime']
            g2 = (gp - gm) / (2 * fd); carried = False
        st = -g1 / g2 if (g2 < 0 and abs(g2) > 1e-16) else np.sign(g1) * trust
        st = float(np.clip(st, -trust, trust))
        ok = False
        for _ in range(6):                               # bounded halving (a 20-halving safeguard cost 80 calls on a flat top)
            xt = x + st
            if abs(xt - w0) > radius:                       # stay in the member's window
                st *= 0.5; continue
            et = cl.gain(xt, arow, nvec, K)
            if et['sigma'] >= best['sigma']:
                ok = True; break
            st *= 0.5
        if not ok:
            if carried:
                g2 = None; carried = False               # the carried curvature failed: one fresh FD attempt
                continue
            break
        # secant update of the curvature from the two gprime values
        dx = xt - x
        if abs(dx) > 1e-12:
            g2s = (et['gprime'] - g1) / dx
            g2 = g2s if g2s < 0 else None
        x, e, best, xb = xt, et, et, xt
        if abs(st) < 1e-13:
            break
    if state is not None and g2 is not None and g2 < 0:
        state['g2'] = float(g2)
    return float(xb), best


def follow_batch(cl, ws0, arow, nvec, K, radii, hints, states, fd=1e-5, itmax=8, gtol=1e-7):
    """follow() for many members at once on a reduced closed loop with
    gain_batch (2026-09-25): the same safeguarded Newton -- start at the
    previous top, carried curvature with secant updates, finite difference when
    none is known or a carried one fails, bounded halving inside each member's
    window -- with every evaluation of an iteration done in one batched call.
    Returns [(top, gain-at-top dict with vectors)]."""
    m = len(ws0)
    w0 = np.asarray(ws0, float); rad = np.asarray(radii, float)
    x = np.array([float(h) if (h is not None and abs(h - w0[i]) < rad[i]) else w0[i] for i, h in enumerate(hints)])
    e = cl.gain_batch(x, arow, nvec, K, np.ones(m, bool))
    best = list(e); xb = x.copy()
    g2 = np.array([st.get('g2', np.nan) if st is not None else np.nan for st in states], float)
    carried = np.isfinite(g2)
    trust = np.minimum(0.05, rad)
    act = np.arange(m)
    for _ in range(itmax):
        g1 = np.array([best[i]['gprime'] for i in act]); sg = np.array([best[i]['sigma'] for i in act])
        keep = np.abs(g1) > gtol * np.maximum(1.0, sg)
        act, g1 = act[keep], g1[keep]
        if act.size == 0:
            break
        need = [i for i in act if not (np.isfinite(g2[i]) and g2[i] < 0)]
        if need:
            need = np.array(need)
            ep = cl.gain_batch(xb[need] + fd, arow, nvec, K, np.ones(len(need), bool))
            em = cl.gain_batch(xb[need] - fd, arow, nvec, K, np.ones(len(need), bool))
            for j, i in enumerate(need):
                g2[i] = (ep[j]['gprime'] - em[j]['gprime']) / (2 * fd); carried[i] = False
        st = np.where(g2[act] < 0, -g1 / np.where(g2[act] < 0, g2[act], -1.0), np.sign(g1) * trust[act])
        st = np.clip(st, -trust[act], trust[act])
        pend = np.arange(len(act)); ok = np.zeros(len(act), bool)
        for _h in range(6):
            if pend.size == 0:
                break
            xt = xb[act[pend]] + st[pend]
            inside = np.abs(xt - w0[act[pend]]) <= rad[act[pend]]
            st[pend[~inside]] *= 0.5
            ev_idx = pend[inside]
            if ev_idx.size:
                et = cl.gain_batch(xb[act[ev_idx]] + st[ev_idx], arow, nvec, K, np.ones(ev_idx.size, bool))
                for j, jj in enumerate(ev_idx):
                    i = act[jj]
                    if et[j]['sigma'] >= best[i]['sigma']:
                        dx = st[jj]
                        g2s = (et[j]['gprime'] - best[i]['gprime']) / dx if abs(dx) > 1e-12 else g2[i]
                        g2[i] = g2s if g2s < 0 else np.nan
                        carried[i] = True               # a secant value, not a fresh difference
                        xb[i] = xb[i] + dx; best[i] = et[j]; ok[jj] = True
                    else:
                        st[jj] *= 0.5
            pend = np.array([jj for jj in pend if not ok[jj]], int)
        # a member without ascent: one fresh finite difference if its curvature was
        # carried, otherwise it stops (as follow())
        nxt = []
        for jj, i in enumerate(act):
            if ok[jj]:
                nxt.append(i)
            elif carried[i]:
                g2[i] = np.nan; carried[i] = False; nxt.append(i)
        act = np.array(nxt, int)
    for i, stt in enumerate(states):
        if stt is not None and np.isfinite(g2[i]) and g2[i] < 0:
            stt['g2'] = float(g2[i])
    return [(float(xb[i]), best[i]) for i in range(m)]
