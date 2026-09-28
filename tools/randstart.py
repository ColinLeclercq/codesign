#!/usr/bin/env python3
"""randstart.py -- random controller starts, two arms (Stage 2, note item 1).

Writes a catalogue in the seed format run_case_epi reads with --seed NAME
--seed-file FILE (names, and per name d, e, f, nvec), plus K, xa, xs and the
scales used, so a start is fully specified and reproducible.

Arm `scaled` (note section 5.1, with the reviewer's refinement):
  omega_* per start, log-uniform between the plant's least-damped mode
  frequency and the band half-width (about 1.7 decades at GL) -- pinned to
  either end the arm is not distinct from the other or thirty times too fast;
  K_* = 1 / max_w |c^H R(iw) b| over the band, the SISO loop gain the
  controller actually sees (not the performance channel's norm);
  A_c tridiagonal: diagonal -omega_* |xi|, off-diagonals omega_* xi, xi ~ N(0,1),
  redrawn while unstable; n ~ K_* N(0,1); K ~ 0.1 K_* N(0,1); positions
  uniform in their ranges (position cases).
Arm `hifoo`: every entry N(0,1), positions uniform (HIFOO's unscaled draw).

    python tools/randstart.py --ne 100 --case K+B+C --arm scaled --n 16 --out rand_scaled_ne100.npz
"""
from __future__ import annotations
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse, json
import numpy as np
import scipy.sparse.linalg as spla
from fe_codesign import FECoDesign, parse_case
from ctrl_order import infer_r
import controller as ctl


def plant_scales(D, xa, xs, wlo, whi, npts=241):
    lam = spla.eigs(D.A.tocsc(), k=6, M=D.M.tocsc(), sigma=0.0, which='LM',
                    return_eigenvectors=False)
    lam = lam[np.argsort(-lam.real)]
    w_plant = float(abs(lam[0].imag)) or 0.5
    b, c, *_ = D.bc(xa, xs, None, None)
    ws = np.linspace(wlo, whi, npts); G = np.zeros(npts)
    for i, w in enumerate(ws):
        lu = spla.splu((1j * w * D.M - D.A).tocsc().astype(complex))
        G[i] = abs(np.vdot(c, lu.solve(b.astype(complex))))
    return w_plant, float(G.max()), float(ws[np.argmax(G)])


def draw_tridiag(rng, r, omega_star, eps, max_tries=200):
    """Stable tridiagonal A_c at scale omega_*: redraw while any Re >= -eps."""
    for _ in range(max_tries):
        xi = rng.normal(size=r); xe = rng.normal(size=r - 1); xf = rng.normal(size=r - 1)
        d = -omega_star * np.abs(xi) + 0j
        e = omega_star * xe + 0j; f = omega_star * xf + 0j
        if np.max(np.linalg.eigvals(ctl.A_of(d, e, f)).real) < -eps:
            return d, e, f
    return d, e, f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ne', type=int, default=100)
    ap.add_argument('--case', default='K+B+C')
    ap.add_argument('--arm', choices=['scaled', 'hifoo'], default='scaled')
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--wlo', type=float, default=-30.0)
    ap.add_argument('--whi', type=float, default=30.0)
    ap.add_argument('--xa-range', nargs=2, type=float, default=[-3.0, 0.0])
    ap.add_argument('--xs-range', nargs=2, type=float, default=[0.0, 3.0])
    ap.add_argument('--results', default=None, help='where to read r from')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    r = infer_r(ne=a.ne, results=a.results, quiet=True)
    D = FECoDesign(ne=a.ne, r=r)
    fl = parse_case(a.case)
    rng = np.random.default_rng(a.seed)
    w_plant, gpeak, w_at = plant_scales(D, -1.03, 0.98, a.wlo, a.whi)
    K_star = 1.0 / gpeak
    out = dict(r=r, names=[], arm=a.arm, case=a.case, ne=a.ne, w_plant=w_plant,
               K_star=K_star, loop_gain_peak=gpeak, loop_gain_at=w_at, seed=a.seed)
    for i in range(a.n):
        nm = f"{a.arm}_{i:03d}"
        xa = rng.uniform(*a.xa_range) if fl.get('B') else -1.03
        xs = rng.uniform(*a.xs_range) if fl.get('C') else 0.98
        if a.arm == 'scaled':
            ws_ = float(np.exp(rng.uniform(np.log(w_plant), np.log(0.5 * (a.whi - a.wlo)))))
            d, e, f = draw_tridiag(rng, r, ws_, a.eps)
            nv = K_star * (rng.normal(size=r) + 1j * rng.normal(size=r)) / np.sqrt(2)
            K = 0.1 * K_star * complex(rng.normal(), rng.normal()) / np.sqrt(2)
        else:
            ws_ = 1.0
            d = (rng.normal(size=r) + 1j * rng.normal(size=r)) / np.sqrt(2)
            e = (rng.normal(size=r - 1) + 1j * rng.normal(size=r - 1)) / np.sqrt(2)
            f = (rng.normal(size=r - 1) + 1j * rng.normal(size=r - 1)) / np.sqrt(2)
            nv = (rng.normal(size=r) + 1j * rng.normal(size=r)) / np.sqrt(2)
            K = complex(rng.normal(), rng.normal()) / np.sqrt(2)
        al_c = float(np.max(np.linalg.eigvals(ctl.A_of(d, e, f)).real))
        out['names'].append(nm)
        out.update({f"{nm}__d": d, f"{nm}__e": e, f"{nm}__f": f, f"{nm}__nvec": nv,
                    f"{nm}__K": K, f"{nm}__xa": xa, f"{nm}__xs": xs,
                    f"{nm}__omega_star": ws_, f"{nm}__alpha_ctrl": al_c})
    np.savez(a.out, **out)
    print(json.dumps(dict(file=a.out, arm=a.arm, n=a.n, r=r, w_plant=round(w_plant, 4),
                          K_star=round(K_star, 5), loop_gain_peak=round(gpeak, 4), at=round(w_at, 3),
                          ctrl_stable=int(sum(out[f"{n}__alpha_ctrl"] < -a.eps for n in out['names'])))))


if __name__ == '__main__':
    raise SystemExit(main())
