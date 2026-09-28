#!/usr/bin/env python3
"""gate_replay.py -- the continuation gate replayed on a run's steps against the
DENSE truth (2026-09-28; algebra note item 11, STAGE5 33).

    python tools/gate_replay.py --ne 100 --run runs_old/results_n100_kbc_22_v8_stop001_6.389368 [--case K+B+C]

For every step the run took (accepted design -> next round's candidate, from
the dumps and the log): the closed-loop abscissa along the straight segment on
a dense grid (dense eigensolves, GL only -- the truth: does any eigenvalue
cross Re = -eps on the way, is the end point stable), the FULL gate
(continuation_gate, the loop's) and the REDUCED gate (continuation_gate_reduced,
item 11) with a reduced model anchored at the step's start (certified peaks +
the watch list's eigenvectors, as the loop anchors every round), with their
full-model factorisation counts.  The designs are interpolated in the physical
parameters (the packing is linear, so this is the gate's segment)."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse, glob, json, os, re
import numpy as np
from fe_codesign import FECoDesign
import controller as ctl
import lu_counter
import run_case_epi as R
from reduced import ReducedOracle, input_basis


class LinPack:
    """Linear packing of (d, e, f, nvec, K, xa, xs) for K+B+C designs."""
    def __init__(self, r): self.r = r
    def pack(self, des):
        c = np.concatenate([des['d'], des['e'], des['f'], des['nvec'], [des['K']]])
        return np.concatenate([c.real, c.imag, [des['xa'], des['xs']]])
    def unpack(self, z):
        r = self.r; m = 4 * r - 1
        c = z[:m] + 1j * z[m:2 * m]
        d, e, f = c[:r], c[r:2 * r - 1], c[2 * r - 1:3 * r - 2]
        nvec, K = c[3 * r - 2:4 * r - 2], c[4 * r - 2]
        return (d, e, f), nvec, K, float(z[2 * m]), float(z[2 * m + 1]), None, None


def design(fz):
    z = np.load(fz)
    return dict(d=z['design_d'], e=z['design_e'], f=z['design_f'], nvec=z['design_nvec'],
                K=complex(z['design_K']), xa=float(z['design_xa']), xs=float(z['design_xs']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=100); ap.add_argument('--run', required=True)
    ap.add_argument('--eps', type=float, default=0.05); ap.add_argument('--ntau', type=int, default=101)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    dd = glob.glob(os.path.join(a.run, 'dump_*'))[0]
    log = open(os.path.join(a.run, 'run.log')).read()
    acc = {}
    for m in re.finditer(r'^  outer (\d+): .*?certified=([0-9.]+)\s+best=([0-9.]+)', log, re.M):
        acc[int(m.group(1))] = abs(float(m.group(2)) - float(m.group(3))) < 1e-12
    files = sorted(glob.glob(os.path.join(dd, 'outer_*.npz')))
    des = {int(os.path.basename(f)[6:9]): design(f) for f in files}
    r = len(des[min(des)]['d'])
    D = FECoDesign(ne=a.ne, r=r); pk = LinPack(r); eps = a.eps
    lu_counter.BIG_N = D.n; lu_counter.install()
    base = None; rows = []
    for k in sorted(des):
        if base is not None:
            z0, z1 = pk.pack(base), pk.pack(des[k])
            # truth: dense abscissa along the segment
            al = []
            for tau in np.linspace(0, 1, a.ntau):
                arow, nvec, K, xa, xs, *_ = pk.unpack((1 - tau) * z0 + tau * z1)
                b, c, *_ = D.bc(xa, xs, None, None)
                al.append(float(D.alpha_dense(arow, nvec, K, b, c)[0]))
            al = np.array(al); cross = bool(np.any(al > -eps)); end_ok = bool(al[-1] <= -eps)
            # watch list at the start (the loop's: the certificate's near-line eigenvalues)
            arow0, nvec0, K0, xa0, xs0, *_ = pk.unpack(z0); b0, c0, *_ = D.bc(xa0, xs0, None, None)
            seen = []; D.alpha_certified(arow0, nvec0, K0, b0, c0, eps=eps, eigs_seen=seen)
            watch = R._topk_watch(seen, 8, eps)
            # full gate
            lu_counter.reset()
            okf, _, nqf, nsf = R.continuation_gate(D, pk, z0, z1, watch, eps, verbose=False)
            luf = lu_counter.snapshot()['n_lu_full']
            # reduced model anchored at the start (outside the count: the loop pays it anyway)
            ro = ReducedOracle(D); ro.set_input_basis(input_basis(D, 'b', -3, 0, other=0.98)[0],
                                                       input_basis(D, 'c', 0, 3, other=-1.03)[0]); ro.galerkin = True
            cl0 = D.cl(b0, c0)
            ws = np.linspace(-30, 30, 241); gs = np.array([cl0.gain(float(w), arow0, nvec0, K0, want_vectors=False)['sigma'] for w in ws])
            for w in R._enrol_peaks(ws, gs, None, rel=0.3):
                ro.enrich(cl0, float(w), arow0, nvec0, K0, nvec_dirs=3)
            ro.enrich_eig(cl0, watch, arow0, nvec0, K0)
            lu_counter.reset()
            okr, _, nqr, nsr, nfr = R.continuation_gate_reduced(D, pk, z0, z1, watch, eps, ro, verbose=False)
            lur = lu_counter.snapshot()['n_lu_full']
            row = dict(round=k, accepted=acc.get(k), truth_cross=cross, truth_end_stable=end_ok,
                       max_alpha_path=float(al.max()), full_ok=bool(okf), full_lu=int(luf), full_splits=int(nsf),
                       red_ok=bool(okr), red_lu=int(lur), red_splits=int(nsr), red_fallbacks=int(nfr))
            rows.append(row)
            verdict = lambda ok: 'accept' if ok else 'reject'
            print(f"round {k:2d}: truth {'CROSSES' if cross else 'clear  '} (max alpha on the path {al.max():+.4f}, "
                  f"end {'stable' if end_ok else 'UNSTABLE'}) | full gate {verdict(okf)} ({luf} LU, {nsf} splits) | "
                  f"reduced gate {verdict(okr)} ({lur} LU, {nsr} splits, {nfr} fallbacks)", flush=True)
        if base is None or acc.get(k):
            base = des[k]
    fa = sum(1 for x in rows if x['truth_cross'] and x['red_ok']); ff = sum(1 for x in rows if x['truth_cross'] and x['full_ok'])
    print(f"\n{len(rows)} steps; false accepts (path crosses, gate accepts): full {ff}, reduced {fa}; "
          f"agreement full/reduced {sum(x['full_ok'] == x['red_ok'] for x in rows)}/{len(rows)}; "
          f"full-model LU: full {sum(x['full_lu'] for x in rows)}, reduced {sum(x['red_lu'] for x in rows)}")
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    import os
    main()
