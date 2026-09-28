#!/usr/bin/env python3
"""transfer_ne.py -- carry a design record to another mesh (debugging at ne=100).

Controller (d, e, f, nvec, K) and positions (xa, xs) do not depend on the
mesh; free fields va, vs are interpolated between node sets (real and
imaginary parts), the pre-Sobolev coordinates xi = smooth_unmap(field)
regenerated at the target mesh (the loader resumes from xi), and gamma
re-certified at the target mesh (rtol 1e-6).  The working set is kept.

    python tools/transfer_ne.py --src results/K_B_C_s0_ne200.npz --case K+B+C --ne 100 --out results_ne100
"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--case', required=True)
    ap.add_argument('--ne', type=int, default=100); ap.add_argument('--out', default='results_ne100')
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--sob_len', type=float, default=0.8, help='MUST match the driver (run_case_epi --sob_len, default 0.8): xi = smooth_unmap(field) depends on it')
    a = ap.parse_args()
    from fe_codesign import FECoDesign, parse_case
    from packer import Packer
    import run_case_epi as R
    src = np.load(a.src, allow_pickle=True)
    r = len(np.asarray(src['d']))
    Ds = FECoDesign(ne=(len(np.asarray(src['va'])) + 1) // 2 if 'va' in src.files and np.asarray(src['va']).size else 200, r=r)   # n = 2 ne - 1
    D = FECoDesign(ne=a.ne, r=r, sob_len=a.sob_len)   # 2026-09-24: the default sob_len 0 wrote xi = field, and the run's start smoothed it again (18 % field error, start 6.18 instead of 3.94)
    fl = parse_case(a.case)
    out = {k: src[k] for k in src.files}
    for key, xikey in (('va', 'xi_a'), ('vs', 'xi_s')):
        if key in src.files and np.asarray(src[key]).size:
            v = np.asarray(src[key], complex)
            xt = np.asarray(D.xn, float)
            # 2026-09-28 (STAGE5 32.7): the P2 function itself, not a linear interpolation of its
            # nodal values (which smoothed the 80-mode design: 4.02 at ne=200 instead of 5.52)
            from fieldbasis import p2_eval
            vt = p2_eval(Ds, v, xt)
            out[key] = vt
            out[xikey] = D.smooth_unmap(vt)
            _rt = np.linalg.norm(D.smooth_map(out[xikey]) - vt) / max(np.linalg.norm(vt), 1e-300)
            if _rt > 1e-8:
                raise RuntimeError(f'{key}: smooth_map(smooth_unmap(field)) round trip {_rt:.1e}')
    aa = (src['d'], src['e'], src['f']); nn = src['nvec']; KK = complex(src['K'])
    xa, xs = float(src['xa']), float(src['xs'])
    # the Packer takes the PRE-SMOOTHING coordinates xi (va = smooth_map(xi)), as the driver's loader does
    va = out['xi_a'] if ('xi_a' in out and fl.get('FB')) else D.gauss(xa)
    vs = out['xi_s'] if ('xi_s' in out and fl.get('FC')) else D.gauss(xs)
    pk = Packer(D, fl, aa, nn, KK, xa, xs, va, vs)
    R.certified_norm.rtol = 1e-6; R.certified_norm.sweep_n = 241
    g, w, ok, _ = R.certified_norm(D, pk, pk.z0(), -30.0, 30.0, gam_lb=1e-8)
    b, c, *_ = D.bc(xa, xs, out['va'] if fl.get('FB') else None, out['vs'] if fl.get('FC') else None)
    al = D.alpha_certified(aa, nn, KK, b, c, eps=a.eps)[0]
    out['gamma'] = float(g); out['alpha'] = float(al)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, re.sub(r'_ne\d+', f'_ne{a.ne}', os.path.basename(a.src)).replace('.npz', ''))   # 2026-09-28: was replace('ne200', ...): a source at another ne kept its own name
    np.savez(stem + '.npz', **out)
    json.dump(dict(case=a.case, gamma=float(g), alpha=float(al), note=f'transferred from {a.src} to ne={a.ne}'),
              open(stem + '.json', 'w'))
    print(f"{a.case}: ne={Ds.ne} gamma {float(src['gamma']):.6f} -> ne={a.ne} certified {g:.6f} (witness {w:+.3f}); alpha {al:+.4f}; -> {stem}.npz")


if __name__ == '__main__':
    sys.exit(main())
