"""
check_sym_pair.py — is an asymmetric optimum SPONTANEOUSLY BROKEN, or wrong?

The averaging test in check_symmetry.py asks "is the design AT the symmetric
fixed point".  That is the wrong question for a symmetry-broken optimum: when
the fixed point ceases to be a minimum, the optimum splits into a PAIR of
mirror-image solutions, neither of which is symmetric, and averaging them lands
on the saddle between (which is why gamma goes UP).

The right question is whether the MAP takes an optimum to an optimum.  Apply

    psi_a -> R conj(psi_s) ,   psi_s -> R conj(psi_a)

(the actuator/sensor exchange composed with reflection and conjugation, under
which the problem is invariant because A^H = R conj(A) R exactly).  If gamma is
unchanged, the design has a distinct mirror twin with identical performance:
spontaneous symmetry breaking, which is a RESULT.  If gamma changes, the
invariance is not being realised and something is wrong.

    python check_sym_pair.py --ne 200 --results results_m05
"""
from __future__ import annotations
# --- repository root on sys.path (moved into tools/ by cleanup_archive.py)
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
# --- end root shim
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, glob, json, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_codesign


def peak(D, arow, nvec, K, b, c, W):
    cl = D.cl(b, c)
    return max(cl.gain(float(w), arow, nvec, K,
                       want_vectors=False)['sigma'] for w in W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results_m05')
    ap.add_argument('--r', type=int, default=-1,
                    help='controller order; -1 reads it from '
                         'the seed catalogue, which is where '
                         'it is decided')
    ap.add_argument('--nw', type=int, default=1601)
    a = ap.parse_args()
    if getattr(a, 'r', -1) is not None and a.r < 0:
        from ctrl_order import infer_r
        a.r = infer_r(a.ne, getattr(a, 'results', None))
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=0.8)
    W = np.linspace(-8, 8, a.nw)
    print(f"{'case':12s} {'gamma':>9s} {'gamma mirrored':>15s} {'change':>9s} "
          f"{'residual':>9s}  verdict")
    for jf in sorted(glob.glob(os.path.join(a.results, f"*_ne{a.ne}.json"))):
        m = json.load(open(jf)); d = np.load(jf[:-5] + '.npz', allow_pickle=True)
        pa = d['psi_a'] if 'psi_a' in d.files else np.zeros(0)
        ps = d['psi_s'] if 'psi_s' in d.files else np.zeros(0)
        if pa.size == 0 or ps.size == 0 or pa.size != ps.size:
            continue
        arow = (d['d'], d['e'], d['f']); nv = d['nvec']; K = complex(d['K'])
        b0, c0, *_ = D.bc(float(d['xa']), float(d['xs']), pa, ps)
        g0 = peak(D, arow, nv, K, b0, c0, W)
        # the symmetry MAP: exchange, reflect, conjugate
        pa_m = np.conj(ps[::-1])
        ps_m = np.conj(pa[::-1])
        b1, c1, *_ = D.bc(float(d['xa']), float(d['xs']), pa_m, ps_m)
        g1 = peak(D, arow, nv, K, b1, c1, W)
        mir = np.conj(ps[::-1]); na_ = max(np.linalg.norm(pa), 1e-30)
        res = min(np.linalg.norm(pa - mir), np.linalg.norm(pa + mir)) / na_
        rel = (g1 - g0) / g0
        if abs(rel) < 2e-3 and res < 0.02:
            v = 'symmetric optimum (its own mirror)'
        elif abs(rel) < 2e-3:
            v = 'BROKEN symmetry: a distinct mirror twin, same gamma'
        else:
            v = '*** map changes gamma -- invariance NOT realised ***'
        print(f"{m['case']:12s} {g0:9.4f} {g1:15.4f} {rel:+8.2%} "
              f"{res:9.2%}  {v}")
    print("\nA controller is NOT re-mapped here: the same (d,e,f,n,Dc) is kept.")
    print("That is deliberate -- for a symmetric or static compensator it is")
    print("its own image, so any change in gamma is attributable to the fields.")


if __name__ == "__main__":
    main()
