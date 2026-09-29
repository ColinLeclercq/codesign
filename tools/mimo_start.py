#!/usr/bin/env python3
"""mimo_start.py -- build a MIMO start record for the production driver
(run_case_epi.py --n-act 2 --n-sens 2 --mimo-start REC; STAGE6).

    # from a SISO K+B+C optimum: loop 1 = that design, loop 2 switched off
    python tools/mimo_start.py --siso results_n100_kbc/K_B_C_s0_ne100.npz --pair2=-4,4 --out results_mimo_prod/start_kbc.npz
    # from a SISO atoms optimum (static gain + passive compensator): loop 2 = shifted copies of the fields
    python tools/mimo_start.py --siso results_fb_atoms_n200_long/K_FB_FC_s0_ne200.npz --atoms --ne 100 --out ...
    # from a prototype record (tools/mimo_proto.py)
    python tools/mimo_start.py --proto results_mimo/cen3.npz --out ...

Record: d, e, f (A_c tridiagonal, order r), Bc (r x p), nvec = C_c (m x r),
K = D_c (m x p), xa (m), xs (p), va / vs (m / p x n raw fields or empty),
atoms_a_p / atoms_s_p (m / p x 5 n_atoms or empty), loop_orders, gamma (a
placeholder unless certified: the driver certifies its start)."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse, json
import numpy as np
from fe_codesign import FECoDesign
import fieldbasis as FB


def from_siso(a):
    z = np.load(a.siso, allow_pickle=True)
    r1 = len(z['d']); m = p = 2; D = FECoDesign(ne=a.ne, r=r1)
    if a.atoms:
        # static gain K on pair 1 (the SISO atoms design), passive compensator kept
        # (C_c = 0), pair 2: the same fields shifted by -dx (actuator) / +dx (sensor), K_2 = 0
        R = r1
        d, e, f = (np.asarray(z[k], complex) for k in ('d', 'e', 'f'))
        Bc = np.zeros((R, p), complex); Bc[:, 0] = 1.0
        nvec = np.zeros((m, R), complex)
        K = np.array([[complex(z['K']), 0], [0, 0]], complex)
        pa = np.asarray(z['atoms_a_p'], float); ps = np.asarray(z['atoms_s_p'], float); na = len(pa) // FB.NATOM_PAR
        pa2, ps2 = pa.copy(), ps.copy(); pa2[2 * na:3 * na] -= a.shift; ps2[2 * na:3 * na] += a.shift
        A = np.array([pa, pa2]); S = np.array([ps, ps2])
        va = np.array([FB.atom_field(D.xn, *FB.split_atoms(q, na), 1.0, 0.7071) for q in A])
        vs = np.array([FB.atom_field(D.xn, *FB.split_atoms(q, na), 1.0, 0.7071) for q in S])
        xa = np.zeros(m); xs = np.zeros(p)
        orders = [R // 2, R - R // 2]
        return dict(d=d, e=e, f=f, Bc=Bc, nvec=nvec, K=K, xa=xa, xs=xs, va=va, vs=vs,
                    atoms_a_p=A, atoms_s_p=S, loop_orders=np.array(orders), gamma=float(z['gamma']))
    R = 2 * r1
    d = np.concatenate([np.asarray(z['d'], complex), -1.0 + 1j * np.linspace(-3, 3, r1)])
    e = np.concatenate([np.asarray(z['e'], complex), [0], np.zeros(r1 - 1, complex)])
    f = np.concatenate([np.asarray(z['f'], complex), [0], np.zeros(r1 - 1, complex)])
    Bc = np.zeros((R, p), complex); Bc[:r1, 0] = D.mc; Bc[r1:, 1] = D.mc
    nvec = np.zeros((m, R), complex); nvec[0, :r1] = np.asarray(z['nvec'], complex)
    K = np.array([[complex(z['K']), 0], [0, 0]], complex)
    xa = np.array([float(z['xa']), a.pair2[0]]); xs = np.array([float(z['xs']), a.pair2[1]])
    return dict(d=d, e=e, f=f, Bc=Bc, nvec=nvec, K=K, xa=xa, xs=xs, va=np.zeros(0), vs=np.zeros(0),
                atoms_a_p=np.zeros(0), atoms_s_p=np.zeros(0), loop_orders=np.array([r1, r1]), gamma=float(z['gamma']))


def from_proto(a):
    sys_ = _s.modules
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    import mimo_proto as MP
    z = np.load(a.proto, allow_pickle=True)
    R = int(z['R']); fields = str(z['fields']) if 'fields' in z.files else 'gauss'
    D = FECoDesign(ne=a.ne, r=max(R, 1))
    mm = MP.Mimo(D, R, str(z['mode']), fields=fields, na=int(z['na']) if 'na' in z.files else 4)
    th = np.asarray(z['theta'], complex); pos = np.asarray(z['pos'], float)
    d, e, f, Bc, Cc, Dc = mm.split(th)
    out = dict(K=np.asarray(Dc, complex), gamma=float(z['gamma']))
    if fields == 'gauss':
        out.update(d=d, e=e, f=f, Bc=Bc, nvec=Cc, xa=pos[:2], xs=pos[2:], va=np.zeros(0), vs=np.zeros(0),
                   atoms_a_p=np.zeros(0), atoms_s_p=np.zeros(0), loop_orders=np.array([R // 2, R - R // 2]))
        return out
    # atoms prototype (static gain, no compensator): add the passive compensator
    # of the SISO atoms record the prototype started from
    zs = np.load(a.passive, allow_pickle=True)
    Rp = len(zs['d']); Bc2 = np.zeros((Rp, 2), complex); Bc2[:, 0] = 1.0
    npar = FB.NATOM_PAR * mm.na
    A = pos[:2 * npar].reshape(2, npar); S = pos[2 * npar:].reshape(2, npar)
    va = np.array([FB.atom_field(D.xn, *FB.split_atoms(q, mm.na), 1.0, 0.7071) for q in A])
    vs = np.array([FB.atom_field(D.xn, *FB.split_atoms(q, mm.na), 1.0, 0.7071) for q in S])
    out.update(d=np.asarray(zs['d'], complex), e=np.asarray(zs['e'], complex), f=np.asarray(zs['f'], complex),
               Bc=Bc2, nvec=np.zeros((2, Rp), complex), xa=np.zeros(2), xs=np.zeros(2), va=va, vs=vs,
               atoms_a_p=A, atoms_s_p=S, loop_orders=np.array([Rp // 2, Rp - Rp // 2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--siso'); ap.add_argument('--proto'); ap.add_argument('--out', required=True)
    ap.add_argument('--ne', type=int, default=100)
    ap.add_argument('--pair2', type=lambda t: [float(v) for v in t.split(',')], default=[-4.0, 4.0])
    ap.add_argument('--atoms', action='store_true'); ap.add_argument('--shift', type=float, default=4.0)
    ap.add_argument('--passive', default='results_fb_atoms_n200_long/K_FB_FC_s0_ne200.npz',
                    help='atoms prototype records: the SISO record whose (decoupled) compensator states are kept')
    a = ap.parse_args()
    out = from_siso(a) if a.siso else from_proto(a)
    _o.makedirs(_o.path.dirname(a.out) or '.', exist_ok=True)
    np.savez(a.out, **out)
    json.dump(dict(gamma=None, note=f"MIMO start from {a.siso or a.proto} (gamma not certified; the source's "
                                    f"{out['gamma']:.6f} holds only for the unchanged loop)"),
              open(a.out[:-4] + '.json', 'w'))
    print(f"-> {a.out}: r = {len(out['d'])}, K = {np.round(out['K'], 4).tolist()}, xa = {out['xa']}, xs = {out['xs']}")


if __name__ == '__main__':
    main()
