#!/usr/bin/env python3
"""atom_count.py -- how many Gabor atoms represent the fields of earlier
high-dimensional designs?  (STAGE6 6.32, user 2026-10-01)

For each reference record (fields optimised in a large basis, or the atom
design itself as a check) and each of its two fields (actuator psi_a, sensor
psi_s, unit M-norm), fieldbasis.fit_atoms with N = 1..NMAX atoms at the atom
bounds (width >= smin, |wavenumber| <= kmax) gives the relative M-norm error
of the best N-atom field (complex amplitudes free, so the error is invariant
to the field's phase and scale -- the gauge the controller absorbs).

    python tools/atom_count.py --nmax 10 --smin 0.4 1.0 --out results_atom_count
"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

REFS = [  # (label, record, ne)
    ('H2 kmax=1, 160 modes, ne=400 (S3)', 'results_fb_sob_n400_h2/K_FB_FC_s0_ne400.npz', 400),
    ('H2 kmax=1, 160 modes, ne=200', 'results_fb_sob_n200_h2/K_FB_FC_s0_ne200.npz', 200),
    ('unconstrained, 160 modes, ne=200 (R1; mesh artefact)', 'results_fb_refine_n200_N160/K_FB_FC_s0_ne200.npz', 200),
    ('frozen report design, nodal + Sobolev map, ne=200', 'results_cross/frozen_n200_on400.npz', 200),
    ('4 atoms width>=1 (A2\'\', check)', 'results_fb_atoms_n200_long/K_FB_FC_s0_ne200.npz', 200),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nmax', type=int, default=10)
    ap.add_argument('--smin', type=float, nargs='+', default=[0.4, 1.0])
    ap.add_argument('--kmax', type=float, default=0.7071)
    ap.add_argument('--out', default='results_atom_count')
    a = ap.parse_args()
    from fe_codesign import FECoDesign
    import fieldbasis as fb
    os.makedirs(a.out, exist_ok=True)
    res, meshes = [], {}
    for label, rec, ne in REFS:
        z = np.load(rec, allow_pickle=True)
        if ne not in meshes:
            meshes[ne] = FECoDesign(ne=ne, r=1)
        D = meshes[ne]
        for key, name in (('psi_a', 'actuator'), ('psi_s', 'sensor')):
            v = np.asarray(z[key], complex).ravel()
            for smin in a.smin:
                errs, t0 = [], time.time()
                for N in range(1, a.nmax + 1):
                    errs.append(fb.fit_atoms(D.xn, v, D.M, N, smin, a.kmax)[1])
                r = dict(ref=label, record=rec, gamma=float(z['gamma']), field=name, smin=smin,
                         kmax=a.kmax, err=errs, seconds=round(time.time() - t0, 1))
                res.append(r)
                print(f"{label:52s} {name:8s} smin {smin}: " + " ".join(f"{e:.3f}" for e in errs), flush=True)
    json.dump(res, open(os.path.join(a.out, 'atom_count.json'), 'w'), indent=1)
    hdr = "| reference (gamma) | field | width >= | " + " | ".join(f"N={n}" for n in range(1, a.nmax + 1)) + " |"
    lines = [hdr, "|" + "---|" * (3 + a.nmax)]
    for r in res:
        lines.append(f"| {r['ref']} ({r['gamma']:.6f}) | {r['field']} | {r['smin']} | "
                     + " | ".join(f"{e:.3f}" for e in r['err']) + " |")
    open(os.path.join(a.out, 'atom_count.md'), 'w').write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
