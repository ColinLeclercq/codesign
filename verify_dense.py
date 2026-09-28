"""
verify_dense.py — STANDALONE verification of a design, by dense eigensolve.

The 1-D certificate answers a CONDITIONAL question ("given a stable start and
continuous steps, has anything crossed the line?").  It cannot verify a design
handed to it out of context: an eigenvalue already at Re = +94 is not ON the
line, so "nothing on the line" is true and useless.  That is why the previous
harness reported six UNSAFE results -- it was asking the certificate a question
it does not answer.

This asks the standalone question directly.  At GL scale (n = 399) a dense
eigensolve takes seconds and is definitive.  Run it on the FINAL design of every
production run; it is the acceptance test, not a certificate to be optimised
against.

    python verify_dense.py --ne 200 --results results_chk
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, glob, json, sys
import numpy as np
from scipy.linalg import eig
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_codesign, controller as ctl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results')
    ap.add_argument('--r', type=int, default=-1,
                    help='controller order; -1 reads it from '
                         'the seed catalogue, which is where '
                         'it is decided')
    ap.add_argument('--eps', type=float, default=1e-3)
    ap.add_argument('--wmax', type=float, default=-1.0,
                    help='cap on the controller-pole |Im|.  <0 (default) '
                         'reads ctrl_wmax from each result json, falling '
                         'back to 20 (the run default).  The inner |Im| '
                         'constraint is SOFT and one sweep ended with a pole '
                         'at |Im| = 138 against a cap of 20 -- a resonance '
                         'outside every certified band -- so the acceptance '
                         'test checks it exactly.')
    ap.add_argument('--update', action='store_true',
                    help='write alpha_dense (and ctrl_immax) back into each '
                         'npz, so the no-regress guard and collect.py can '
                         'use the VERIFIED abscissa instead of the stored '
                         'one')
    a = ap.parse_args()
    if getattr(a, 'r', -1) is not None and a.r < 0:
        from ctrl_order import infer_r
        a.r = infer_r(a.ne, getattr(a, 'results', None))
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=0.8)
    n, r = D.n, a.r
    Ad = np.asarray(D.A.todense()); Md = np.asarray(D.M.todense())
    print(f"{'case':12s} {'gamma':>10s} {'stored a':>10s} {'DENSE a':>12s} "
          f"{'Re>0':>5s} {'ctrl|Im|':>9s}  verdict")
    nbad = 0
    for jf in sorted(glob.glob(os.path.join(a.results, f"*_ne{a.ne}.json"))):
        m = json.load(open(jf)); d = np.load(jf[:-5] + '.npz')
        if 'd' not in d.files:
            continue
        arow = (d['d'], d['e'], d['f']); nv = d['nvec']; Kf = complex(d['K'])
        va = d['va'] if d['va'].size else None
        vs = d['vs'] if d['vs'].size else None
        b, c, *_ = D.bc(float(d['xa']), float(d['xs']), va, vs)
        Ac = ctl.A_of(*arow); Bc = np.full(r, 1.0, complex)
        Acl = np.zeros((n + r, n + r), complex)
        Mcl = np.zeros((n + r, n + r), complex)
        Acl[:n, :n] = Ad + Kf * np.outer(b, np.conj(c))
        Acl[:n, n:] = b.reshape(-1, 1) @ np.asarray(nv).reshape(1, r)
        Acl[n:, :n] = Bc.reshape(-1, 1) @ np.conj(c).reshape(1, n)
        Acl[n:, n:] = Ac
        Mcl[:n, :n] = Md; Mcl[n:, n:] = np.eye(r)
        lam = eig(Acl, Mcl, right=False); lam = lam[np.isfinite(lam)]
        ad = float(lam.real.max()); npos = int((lam.real > 0).sum())
        pol = np.linalg.eigvals(np.asarray(Ac, complex))
        cim = float(np.max(np.abs(pol.imag))) if pol.size else 0.0
        # per-case cap: the run recorded what it enforced; a global override
        # is available for old files that did not.
        wmax_c = a.wmax if a.wmax > 0 else float(m.get('ctrl_wmax', 20.0))
        stab = (ad + a.eps <= 0.0)
        cap_ok = (cim <= wmax_c * (1.0 + 1e-6))
        if stab and cap_ok:
            verdict = 'STABLE - valid'
        elif not stab:
            verdict = '*** UNSTABLE - INVALID ***'
            nbad += 1
        else:
            verdict = (f'*** POLE CAP VIOLATED (|Im|={cim:.1f} > '
                       f'{wmax_c:.1f}) ***')
            nbad += 1
        print(f"{m['case']:12s} {m['gamma']:10.4f} {float(d['alpha']):10.4f} "
              f"{ad:12.6f} {npos:5d} {cim:9.2f}  {verdict}")
        if a.update:
            # write the VERIFIED abscissa back, so the no-regress guard and
            # collect.py can rank on evidence instead of on the stored alpha
            # (which is exactly the quantity that is wrong when a design has
            # escaped the continuation chain).
            payload = {k: d[k] for k in d.files}
            payload['alpha_dense'] = np.float64(ad)
            payload['ctrl_immax'] = np.float64(cim)
            np.savez(jf[:-5] + '.npz', **payload)
    print(f"\n{nbad} design(s) are INVALID (unstable or pole cap violated) "
          f"and must not be reported.")
    if a.update:
        print("alpha_dense written back into every npz (--update).")


if __name__ == "__main__":
    main()
