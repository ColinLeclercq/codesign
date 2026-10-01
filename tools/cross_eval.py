#!/usr/bin/env python3
"""cross_eval.py -- evaluate a K+FB+FC design on ANOTHER mesh with nothing
re-optimised (2026-09-28, STAGE5 32.5-32.6): the clean test of mesh
independence, free of optimisation noise (two runs on two meshes also differ
by path and local minimum).

    python tools/cross_eval.py --src rec.npz --ne-src 200 --ne-dst 400 --out results_cross/x.npz
    python tools/tail_certificate.py --ne 400 --record results_cross/x.npz --json ...

Fields: an atom record (atoms_a_p / atoms_s_p) is re-sampled EXACTLY at the
target nodes (the atoms are functions of x); any other field is evaluated as
the P2 finite-element function it is on the source mesh (element shape
functions, zero Dirichlet ends), not interpolated linearly.  The controller
and positions are copied.  The gamma of the written record is the target mesh's grid maximum (a valid
lower bound to start the level set from); certify it with tail_certificate.py (dense level set + tail bound: all w)."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse
import numpy as np
from fe_codesign import FECoDesign
import fieldbasis as FB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--ne-src', type=int, required=True)
    ap.add_argument('--ne-dst', type=int, required=True); ap.add_argument('--out', required=True)
    # the bounds the written atoms are expressed in (STAGE6 6.35: the record's own
    # bounds are read from it, or identified from its field for a legacy record;
    # the defaults contain every legacy bound pair, so the conversion is exact)
    ap.add_argument('--atom-smin', type=float, default=0.4); ap.add_argument('--atom-kmax', type=float, default=0.7071)
    a = ap.parse_args()
    z = np.load(a.src, allow_pickle=True); out = {k: z[k] for k in z.files}
    r = len(z['d'])
    Ds = FECoDesign(ne=a.ne_src, r=r); Dd = FECoDesign(ne=a.ne_dst, r=r); x = np.asarray(Dd.xn, float)
    for key, side in (('va', 'a'), ('vs', 's')):
        pkey = f'atoms_{side}_p'
        n = (np.asarray(z[f'atoms_{side}_nat']).size if f'atoms_{side}_nat' in z.files
             else np.asarray(z[pkey]).size if pkey in z.files else 0) // FB.NATOM_PAR
        P, how = FB.atoms_from_record(z, side, n, a.atom_smin, a.atom_kmax, Ds.xn) if n else (None, '')
        if P is not None:
            ac, c, rho, tau = FB.split_atoms(P[0], n)
            v = FB.atom_field(x, ac, c, rho, tau, a.atom_smin, a.atom_kmax); how = f'atoms (exact; {how})'
            # consistency: the source record's field is the same function
            vs_ = FB.atom_field(np.asarray(Ds.xn), ac, c, rho, tau, a.atom_smin, a.atom_kmax)
            vsrc = np.asarray(z[key], complex)
            err = abs(abs(np.vdot(vs_, Ds.M @ vsrc)) / (Ds.norm_M(vs_) * Ds.norm_M(vsrc)) - 1)
            how += f', source field reproduced to {err:.1e}'
            out[pkey] = P[0]
            out[f'atoms_{side}_nat'] = FB.atoms_to_natural(P[0], n, a.atom_smin, a.atom_kmax)
            out[f'atoms_{side}_bounds'] = np.array([a.atom_smin, a.atom_kmax])
        else:
            v = FB.p2_eval(Ds, z[key], x); how = 'P2 evaluation'
        out[key] = v / Dd.norm_M(v)
        print(f"{key}: {how}", flush=True)
    # mesh-sized companions, so the record also RESUMES on the target mesh (as
    # tools/transfer_ne.py writes them): the driver's pre-Sobolev coordinates
    # xi = smooth_unmap(field) at its default sob_len 0.8; psi_* dropped
    Dx = FECoDesign(ne=a.ne_dst, r=r, sob_len=0.8)
    for key, xikey in (('va', 'xi_a'), ('vs', 'xi_s')):
        out[xikey] = Dx.smooth_unmap(out[key])
    for k in ('psi_a', 'psi_s'):
        out.pop(k, None)
    # a VALID LOWER BOUND for the certificate's starting level (the dense level
    # set returns its start level if nothing crosses it): the grid maximum of
    # sigma_1 at the target mesh
    b, c, *_ = Dd.bc(float(z['xa']), float(z['xs']), out['va'], out['vs']); cl = Dd.cl(b, c)
    aa = (z['d'], z['e'], z['f']); nn = np.asarray(z['nvec'], complex); KK = complex(z['K'])
    ws = np.linspace(-30, 30, 601)
    sg = [cl.gain(float(w), aa, nn, KK, want_vectors=False)['sigma'] for w in ws]
    out['gamma'] = float(np.max(sg)); out['gamma_src'] = float(z['gamma'])
    print(f"grid max of sigma_1 at ne={a.ne_dst}: {out['gamma']:.6f} (source record {float(z['gamma']):.6f})", flush=True)
    _o.makedirs(_o.path.dirname(a.out) or '.', exist_ok=True)
    np.savez(a.out, **out)
    print(f"-> {a.out} (certify with tools/tail_certificate.py --ne {a.ne_dst})")


if __name__ == '__main__':
    main()
