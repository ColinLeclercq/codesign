#!/usr/bin/env python3
"""atom_resolution.py -- which atoms does a mesh capture?  (STAGE6 6.35, user
2026-10-01: the highest wavenumber of an atom should be set by the
discretisation, so that every admissible atom is properly captured.)

What the optimiser sees of a field is the plant's response to it.  So the test
is that response, not the interpolation of the field: for an actuator atom at
c = -2 and a sensor atom at c = +2 with the same width s and wavenumber kappa
(the atoms are functions of x, sampled exactly on every mesh), the plant
transfer T(w) = c^H (i w M - A)^{-1} b on ne = 100, 200, 400 against ne = 1600,

    err(ne; s, kappa) = max_w |T_ne(w) - T_1600(w)| / max_w |T_1600(w)|,  w in [-4, 4].

    python tools/atom_resolution.py --out results_atom_count/atom_resolution.json
"""
import os, sys, json, argparse
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import scipy.sparse.linalg as spla

S = [0.2, 0.3, 0.4, 0.6, 1.0, 2.0]
KAP = [0.0, 0.7071, 1.0, 1.5, 2.0, 3.0, 4.0]


def transfers(D, ws, atoms):
    import fieldbasis as fb
    out = []
    lus = [spla.splu((1j * w * D.M - D.A).tocsc()) for w in ws]
    for s, k in atoms:
        rho = 0.0                                        # with smin = s: width s
        tau = np.arctanh(k / 10.0)                       # kmax = 10: the wavenumber itself
        va = fb.atom_field(D.xn, [1.0], [-2.0], [rho], [tau], s, 10.0)
        vs = fb.atom_field(D.xn, [1.0], [+2.0], [rho], [tau], s, 10.0)
        b, c, *_ = D.bc(0.0, 0.0, va, vs)
        out.append([complex(np.vdot(c, lu.solve(b))) for lu in lus])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, nargs='+', default=[100, 200, 400])
    ap.add_argument('--ref', type=int, default=1600)
    ap.add_argument('--out', default='results_atom_count/atom_resolution.json')
    a = ap.parse_args()
    from fe_codesign import FECoDesign
    ws = np.linspace(-4, 4, 81)
    atoms = [(s, k) for s in S for k in KAP]
    Dref = FECoDesign(ne=a.ref, r=1)
    Tref = transfers(Dref, ws, atoms)
    res = dict(ws=[-4, 4, 81], ref=a.ref, s=S, kappa=KAP, err={})
    for ne in a.ne:
        D = FECoDesign(ne=ne, r=1)
        T = transfers(D, ws, atoms)
        err = np.max(np.abs(T - Tref), axis=1) / np.max(np.abs(Tref), axis=1)
        res['err'][str(ne)] = err.reshape(len(S), len(KAP)).tolist()
        h = (D.xn[-1] - D.xn[0]) / (len(D.xn) - 1)
        print(f"\nne={ne} (node spacing {h:.3f}): relative transfer error against ne={a.ref}")
        print("  s \\ kappa " + " ".join(f"{k:9.4g}" for k in KAP))
        for i, s in enumerate(S):
            print(f"  {s:8.2f}  " + " ".join(f"{e:9.1e}" for e in err[i * len(KAP):(i + 1) * len(KAP)]))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
