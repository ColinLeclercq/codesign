#!/usr/bin/env python3
"""hifoo_init.py -- starts for the backward-facing step: HIFOO's controller
draw (STAGE2: the `hifoo` arm of tools/randstart.py, the better of the two
arms at GL), REAL, at the devices of Barbagallo et al. (2012) (BFS.md 6).

Per start: every controller entry N(0, 1) -- the tridiagonal A_c = (d, e, f),
the output row nvec, the feedthrough K, all real.  Positions (--positions):
  paper   (default, user 2026-10-02) the actuator B2 at (0.6, 1.0) -- near
          separation, figure 1 of the paper (the user's actuator_sensor.edp
          has (0.87, 1.0)) -- and the sensor C1 window [2.0, 2.6] (figure 1;
          the script's upstream spy is [2.0, 2.4])
  random  the actuator centre uniform in the domain, the sensor window start
          uniform on the floor, both at least --margin inside (rejection)
The draw is NOT stabilised here: run_case_epi.py --stabilise makes it
feasible (feasible_start below) when it is not.

--kind zero (user 2026-10-02): the same draws with the loop OPEN -- nvec = 0,
K = 0, only the compensator A_c kept (feasible_start shifts it stable).  The
start's gamma is then the open-loop peak exactly, the floor any useful
controller must go below; the HIFOO draws themselves start above it (test
plant: 1655 against 1464).  At the zero controller the gradient in the gains
is generically nonzero, those in A_c and the positions vanish (they act only
through the loop) and grow as soon as the gains do.

    python bfs/hifoo_init.py --ops bfs/data/ops_fx.npz --n 8 --seed 0 --out bfs/data/starts
writes bfs/data/starts/hifoo_000.npz ... (the --init format of run_case_epi.py);
with --kind zero, zero_000.npz ...
"""
import os, sys, argparse, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE)]


PAPER_XA = (0.6, 1.0)          # B2, Barbagallo et al. (2012) figure 1
PAPER_XS = 2.0                 # C1 = [2.0, 2.6], ibid.


def draw(rng, r, D, positions='paper', max_tries=100000):
    ctrl = dict(d=rng.normal(size=r), e=rng.normal(size=r - 1), f=rng.normal(size=r - 1),
                nvec=rng.normal(size=r), K=float(rng.normal()))
    if positions == 'paper':
        return dict(ctrl, xa=np.array(PAPER_XA), xs=np.array([PAPER_XS]))
    x_lo, x_hi, y_lo, y_hi = D.box()
    for _ in range(max_tries):
        xa = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])
        if D.inside(xa, 'a'):
            break
    else:
        raise RuntimeError("no interior actuator centre drawn")
    from bfs.plant import FLOOR_X0
    xs = np.array([rng.uniform(FLOOR_X0, x_hi - D.pos_margin - D.sensor_len)])
    return dict(ctrl, xa=xa, xs=xs)


def feasible_start(D, arow, nvec, K, b, c, eps, delta=3e-3, tau0=1e-6, pool=None, log=print):
    """HIFOO's phase at BFS scale: the draw made feasible (BFS.md 6).

    GL's phase (stabilise.py) descends the closed-loop ABSCISSA, from a dense
    eigensolve.  Here no method gives the abscissa of an UNSTABLE design: the
    line certificate proves only that nothing lies ON Re = -eps and sees only
    the eigenvalues near its shifts (a line at -0.08 missed -0.044+0.66j on the
    test plant).  So feasibility is reached the way every later step of the
    run is certified -- by continuation from a design whose spectrum is known:

      1. the compensator, exactly (r x r): A_c - s I with s = max(0,
         alpha(A_c) + eps + 2 delta) (the tridiagonal diagonal shifted);
      2. the loop gain: (nvec, K) -> tau (nvec, K).  At tau ~ 0 the closed
         loop is the plant (stable; its near-axis spectrum computed once) and
         the compensator; tau grows to 1 by steps each gated like an accepted
         design of the run: the watch list tracked onto the new tau
         (certify_sparse.continuation_step: a crossing rejects, a large
         motion halves the step) and the line certificate at -eps (clear,
         every eigenvalue it sees left of -eps - delta: the phase's cushion).
         Geometric growth until the first refusal, then bisection in log tau
         to 5 % between the last certified and the smallest refused tau.

    Returns (arow, nvec, K, info); info['tau'] < 1 when the full gain could
    not be reached (the start is then the largest certified fraction)."""
    import controller as ctl
    import certify_sparse as csp
    d, e, f = (np.asarray(x, complex) for x in arow[:3])
    al_c = float(np.max(np.linalg.eigvals(ctl.A_of(d, e, f)).real))
    s = max(0.0, al_c + eps + 2.0 * delta)
    d = d - s
    Ac = ctl.A_of(d, e, f); Bc = ctl.Bc_of((d, e, f), D.r, D.mc)
    nvec = np.asarray(nvec, complex); K = complex(K)
    line = eps                     # the driver's line; accepted designs keep the cushion delta from it

    def cert(tau):
        seen = []
        al, lam, v, u, cl, clear = D.alpha_certified((d, e, f), tau * nvec, tau * K, b, c, eps=line,
                                                     eigs_seen=seen)
        near = sorted(set(complex(x) for x in seen), key=lambda x: -x.real)
        return bool(clear) and al <= -line - delta, al, near[:12]

    ok0, al0, watch = cert(tau0)
    pe = [complex(x) for x in np.atleast_1d(getattr(D, 'plant_ev', [])) if x.imag >= 0]
    watch = sorted(set(watch + [x for x in pe if x.real > -line - 0.5]
                       + [complex(x) for x in np.linalg.eigvals(Ac) if x.imag >= 0]), key=lambda x: -x.real)[:16]
    info = dict(ctrl_shift=s, tau=0.0, steps=0, rejected=0, split=0, alpha=al0)
    if not ok0:
        log(f"  feasible_start: tau = {tau0:g} is not certified (alpha {al0:+.4f}): the plant itself is "
            f"not inside the margin?")
        return (d, e, f), 0.0 * nvec, 0j, info
    if not np.any(nvec) and K == 0:
        # the zero controller (--kind zero): the loop is open at every tau, so
        # the certified anchor IS the start -- nothing to continue
        info['tau'] = 1.0
        log(f"  feasible_start: zero controller (open loop), compensator shifted by {s:.4f}; alpha {al0:+.4f}")
        return (d, e, f), nvec, K, info
    tau, ratio, t_bad = tau0, 4.0, np.inf
    while tau < 1.0 and ratio > 1.05 and t_bad > 1.05 * tau:
        # geometric growth until the first refusal; then bisection (in log tau)
        # between the last certified tau and the smallest refused one
        t1 = min(1.0, tau * ratio, np.sqrt(tau * t_bad))
        ok, split, new_watch, worst = csp.continuation_step(D.A, D.M, b, c, Ac, Bc, t1 * nvec, t1 * K, watch,
                                                            eps=line, pool=pool)
        info['steps'] += 1
        okc = None
        if ok and not split:
            okc, al1, near = cert(t1)
        log(f"    tau {t1:.3g}: tracked {'crossing' if not ok else ('large motion' if split else 'ok')}"
            + ('' if okc is None else f", certificate alpha {al1:+.4f} {'ok' if okc else 'refused'}"))
        if okc is not None:
            if okc:
                tau = t1; info['tau'] = tau; info['alpha'] = al1
                watch = sorted(set([w for w in new_watch if w is not None] + near),
                               key=lambda x: -x.real)[:16]
                ratio = min(ratio * 2.0, 16.0)
                continue
            info['rejected'] += 1
            t_bad = min(t_bad, t1)     # the cushion binds at t1
        elif split:
            info['split'] += 1         # too large a step to track: shorter, not refused
            ratio = np.sqrt(t1 / tau)
        else:
            info['rejected'] += 1
            t_bad = min(t_bad, t1)     # a crossing before t1
    log(f"  feasible_start: compensator shifted by {s:.4f}; loop gain x {tau:.4g} "
        f"({info['steps']} steps, {info['split']} split, {info['rejected']} refused), alpha {info['alpha']:+.4f}")
    return (d, e, f), tau * nvec, tau * K, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ops', default='bfs/data/ops_fx.npz')
    ap.add_argument('--r', type=int, default=5)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--margin', type=float, default=0.05, help='as run_case_epi.py --bfs-margin')
    ap.add_argument('--positions', choices=['paper', 'random'], default='paper')
    ap.add_argument('--kind', choices=['hifoo', 'zero'], default='hifoo',
                    help='zero: the same draws with nvec = 0, K = 0 (the open loop; gamma = its peak)')
    ap.add_argument('--out', default='bfs/data/starts')
    a = ap.parse_args()
    from bfs.plant import BFSPlant
    D = BFSPlant(a.ops, r=a.r); D.pos_margin = a.margin
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for i in range(a.n):
        s = draw(rng, a.r, D, a.positions)
        if a.kind == 'zero':
            s.update(nvec=np.zeros(a.r), K=0.0)
        nm = f"{a.kind}_{i:03d}"
        np.savez(os.path.join(a.out, nm + '.npz'), **s, seed=a.seed, index=i, margin=a.margin, positions=a.positions, kind=a.kind)
        import controller as ctl
        al_c = float(np.max(np.linalg.eigvals(ctl.A_of(s['d'] + 0j, s['e'] + 0j, s['f'] + 0j)).real))
        rows.append(dict(name=nm, xa=np.round(s['xa'], 3).tolist(), xs=np.round(s['xs'], 3).tolist(),
                         K=round(s['K'], 3), alpha_ctrl=round(al_c, 3)))
    print(json.dumps(dict(out=a.out, n=a.n, seed=a.seed, r=a.r, margin=a.margin, positions=a.positions, kind=a.kind,
                     starts=rows), indent=1))


if __name__ == '__main__':
    main()
