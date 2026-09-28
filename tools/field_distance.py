#!/usr/bin/env python3
"""field_distance.py -- gauge-invariant distance between the fields of two
designs, possibly on different meshes (2026-09-28, STAGE5 32.5).

    python tools/field_distance.py --ne-ref 400 label:ne:record ...

Each field is interpolated (piecewise linear, real and imaginary parts) onto
the nodes of the reference mesh and normalised in its M-norm; the distance is
min over the complex unit gauge of ||v1 - e^{i theta} v2||_M
= sqrt(2 - 2 |<v1, v2>_M|) (0: same shape; sqrt 2: orthogonal).  Prints the
matrix for actuator and sensor fields."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse
import numpy as np
from fe_codesign import FECoDesign


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ne-ref', type=int, default=400); ap.add_argument('specs', nargs='+')
    a = ap.parse_args()
    Dr = FECoDesign(ne=a.ne_ref, r=4); xr = np.asarray(Dr.xn)
    labs, F = [], {'va': [], 'vs': []}
    for sp in a.specs:
        lab, ne, rec = sp.split(':', 2)
        D = FECoDesign(ne=int(ne), r=4); x = np.asarray(D.xn); d = np.load(rec, allow_pickle=True)
        labs.append(lab)
        for key in ('va', 'vs'):
            v = np.asarray(d[key], complex)
            w = np.interp(xr, x, v.real, left=0, right=0) + 1j * np.interp(xr, x, v.imag, left=0, right=0)
            F[key].append(w / Dr.norm_M(w))
    for key, nm in (('va', 'actuator'), ('vs', 'sensor')):
        print(f"{nm} fields: gauge-invariant M-distance")
        print(' ' * 34 + ''.join(f"{i:>8d}" for i in range(len(labs))))
        for i, li in enumerate(labs):
            row = [np.sqrt(max(0.0, 2 - 2 * abs(np.vdot(F[key][i], Dr.M @ F[key][j])))) for j in range(len(labs))]
            print(f"{i:2d} {li[:30]:30s} " + ''.join(f"{r:8.4f}" for r in row))


if __name__ == '__main__':
    main()
