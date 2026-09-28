#!/usr/bin/env python3
"""multistart.py -- Stage 2: isolated random starts, and the report.

    python tools/multistart.py run  --ne 100 --case K+B+C --arm scaled --catalogue rand_scaled_ne100.npz --out ms_scaled --outer 8
    python tools/multistart.py report --ne 100 --case K+B+C --dirs ms_scaled,ms_hifoo --ref results/K_B_C_s0_ne200.json

Isolation (reviewer): each start gets its own OUTDIR, starts --cold from its
own catalogue entry (controller, K, positions), no --resume, no donors, no
shared checkpoints; the record carries the seed's name and the input
fingerprint.  Resumable: a start with a record or a not_stabilised marker is
skipped, so the pilot runs in chunks.

Report (the stage-2 gate):
  - fraction stabilised within the phase's budget, per arm;
  - best certified gamma per arm against the reference (section 3 / frozen);
  - number of DISTINCT local optima, clustered by gamma (rtol 1e-3) and by the
    controller's transfer function k(i w) sampled on the band plus (xa, xs)
    for the position cases -- not by (d, e, f), whose tridiagonal form has a
    gauge freedom that would count every optimum once per coordinate copy;
  - the rarefaction curve: distinct optima against starts, to decide whether
    32 or 96 starts are needed.
"""
from __future__ import annotations
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse, json, os, subprocess, sys, time
import numpy as np


def run(a):
    sd = np.load(a.catalogue, allow_pickle=True)
    names = [str(x) for x in sd['names']]
    if a.max_starts:
        names = names[:a.max_starts]
    os.makedirs(a.out, exist_ok=True)
    common = ['--ne', str(a.ne), '--wall', str(a.wall), '--outer', str(a.outer),
              '--inner-budget', 'iters', '--inner-maxit', str(a.inner_maxit), '--cold', '--stabilise',
              '--stab-maxit', str(a.stab_maxit), '--select-cons', 'on', '--ctrl-imcap', 'off',
              '--cont-anchor', 'strict', '--cont-accept-winding', '--sigma2', 'on', '--oracle', a.oracle,
              '--inner', a.inner, '--ctrl-line', 'eps'] + (['--sqp-carry', '--sqp-trust', '0.01'] if a.inner == 'sqp' else []) + \
             (['--converge'] if a.converge else []) + \
             (['--inner-maxit-adapt'] if a.adapt else []) + a.extra
    for nm in names:
        od = os.path.join(a.out, nm)
        tag = f"{a.case.replace('+', '_')}_s0_ne{a.ne}"
        if os.path.exists(os.path.join(od, tag + '.json')) or \
           os.path.exists(os.path.join(od, 'dump', 'not_stabilised.json')):
            continue
        os.makedirs(os.path.join(od, 'dump'), exist_ok=True)
        argv = [sys.executable, os.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))), 'run_case_epi.py'),
                '--case', a.case, '--start', '0', '--eps', str(a.eps), '--seed', nm, '--seed_file', a.catalogue,
                '--outdir', od, '--dump', os.path.join(od, 'dump')] + common
        t0 = time.time()
        with open(os.path.join(od, 'run.log'), 'w') as fh:
            r = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, text=True)
        txt = open(os.path.join(od, 'run.log')).read()
        st = [l for l in txt.splitlines() if l.startswith('stabilise:')]
        g = None
        for l in reversed(txt.splitlines()):
            if l.startswith('{"case"'):
                g = json.loads(l).get('gamma'); break
        print(f"  {nm:12s} {time.time() - t0:5.0f}s  {st[0][11:60] if st else '(no phase)':50s}  "
              f"{'gamma=%.4f' % g if g is not None else 'NO RECORD'}", flush=True)
    return 0


def _kiw(D, arow, nvec, K, ws):
    """Controller transfer function k(i w) = K + n^T (i w I - A_c)^{-1} b_c on the band."""
    import controller as ctl
    Ac = ctl.A_of(*arow); r = Ac.shape[0]; Bc = np.full(r, D.mc, complex)
    nv = np.asarray(nvec, complex).ravel()
    return np.array([complex(K) + nv @ np.linalg.solve(1j * w * np.eye(r) - Ac, Bc) for w in ws])


def report(a):
    from fe_codesign import FECoDesign
    from ctrl_order import infer_r
    r = infer_r(ne=a.ne, results=a.results, quiet=True)
    D = FECoDesign(ne=a.ne, r=r)
    ws = np.linspace(a.wlo, a.whi, 61)
    tag = f"{a.case.replace('+', '_')}_s0_ne{a.ne}"
    ref = None
    if a.ref and os.path.exists(a.ref):
        ref = json.load(open(a.ref)).get('gamma')
    out = {}
    for d in [x for x in a.dirs.split(',') if x]:
        rows, nstab, ntot = [], 0, 0
        for nm in sorted(os.listdir(d)):
            od = os.path.join(d, nm)
            if not os.path.isdir(od):
                continue
            ntot += 1
            f = os.path.join(od, tag + '.json')
            if not os.path.exists(f):
                continue
            rec = json.load(open(f)); dz = np.load(os.path.join(od, tag + '.npz'), allow_pickle=True)
            st = rec.get('stabilise') or {}
            if st.get('stabilised', True):
                nstab += 1
            k = _kiw(D, (dz['d'], dz['e'], dz['f']), dz['nvec'], complex(dz['K']), ws)
            rows.append(dict(name=nm, gamma=float(rec['gamma']), alpha=float(rec.get('alpha_dense', rec.get('alpha'))),
                             xa=float(dz['xa']), xs=float(dz['xs']), k=k, calls=rec.get('calls'),
                             phase_evals=st.get('evals'), phase_alpha_start=st.get('alpha_start')))
        # clustering: gamma within rtol AND transfer function within rtol (relative L2 on the band) AND positions
        clusters = []
        for row in sorted(rows, key=lambda x: x['gamma']):
            for cl in clusters:
                c0 = cl[0]
                same_g = abs(row['gamma'] - c0['gamma']) <= a.rtol * c0['gamma']
                same_k = np.linalg.norm(row['k'] - c0['k']) <= a.ktol * max(np.linalg.norm(c0['k']), 1e-12)
                same_x = abs(row['xa'] - c0['xa']) <= a.xtol and abs(row['xs'] - c0['xs']) <= a.xtol
                if same_g and same_k and same_x:
                    cl.append(row); break
            else:
                clusters.append([row])
        # rarefaction: distinct optima as a function of the number of starts (catalogue order)
        seen_c = []; curve = []
        order = {rw['name']: i for i, rw in enumerate(sorted(rows, key=lambda x: x['name']))}
        for i, row in enumerate(sorted(rows, key=lambda x: x['name'])):
            cid = next(j for j, cl in enumerate(clusters) if row in cl)
            if cid not in seen_c:
                seen_c.append(cid)
            curve.append(len(seen_c))
        best = min((rw['gamma'] for rw in rows), default=None)
        out[d] = dict(starts=ntot, records=len(rows), stabilised=nstab, best_gamma=best,
                      n_distinct=len(clusters), rarefaction=curve,
                      clusters=[dict(n=len(cl), gamma=cl[0]['gamma'], xa=cl[0]['xa'], xs=cl[0]['xs'],
                                     members=[x['name'] for x in cl]) for cl in clusters])
        print(f"=== {d}: {ntot} starts, {len(rows)} records, {nstab}/{ntot} stabilised, "
              f"best gamma {best if best is None else round(best, 4)}"
              + (f" (reference {ref:.4f}, {100 * (best - ref) / ref:+.1f} %)" if ref and best else "")
              + f", {len(clusters)} distinct optima; rarefaction {curve}")
        for cl in out[d]['clusters']:
            print(f"    gamma {cl['gamma']:9.4f}  x={cl['n']:2d}  xa={cl['xa']:+.3f} xs={cl['xs']:+.3f}  {cl['members'][:6]}")
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1); print(f"-> {a.json}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--ne', type=int, default=100); r.add_argument('--case', default='K+B+C')
    r.add_argument('--arm', default='scaled'); r.add_argument('--catalogue', required=True)
    r.add_argument('--out', required=True); r.add_argument('--eps', type=float, default=0.05)
    r.add_argument('--wall', type=int, default=1200); r.add_argument('--outer', type=int, default=8)
    r.add_argument('--inner-maxit', dest='inner_maxit', type=int, default=10)
    r.add_argument('--adapt', action='store_true', default=True)
    r.add_argument('--stab-maxit', dest='stab_maxit', type=int, default=150)
    r.add_argument('--oracle', default='reduced'); r.add_argument('--max-starts', dest='max_starts', type=int, default=0)
    r.add_argument('--converge', action='store_true',
                   help='run each start to CONVERGENCE (stationary inner solve, certificate agrees, no new frequency) under the --outer cap, instead of a fixed number of rounds')
    r.add_argument('--inner', default='sqp', choices=['sqp', 'pygranso'],
                   help='the SQP is correct on the block and the per-eigenvalue rows and ties pyGRANSO at equal cost (STAGE1B section 8)')
    r.add_argument('--extra', nargs=argparse.REMAINDER, default=[])
    p = sub.add_parser('report')
    p.add_argument('--ne', type=int, default=100); p.add_argument('--case', default='K+B+C')
    p.add_argument('--dirs', required=True); p.add_argument('--ref', default=None)
    p.add_argument('--results', default=None); p.add_argument('--wlo', type=float, default=-30.0)
    p.add_argument('--whi', type=float, default=30.0); p.add_argument('--rtol', type=float, default=1e-3)
    p.add_argument('--ktol', type=float, default=1e-2); p.add_argument('--xtol', type=float, default=1e-2)
    p.add_argument('--json', default=None)
    a = ap.parse_args()
    return run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    raise SystemExit(main())
