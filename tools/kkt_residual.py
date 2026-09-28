#!/usr/bin/env python3
"""kkt_residual.py -- constraint-aware stationarity at a design (TRACKER H).

    python tools/kkt_residual.py --ne 100 --record results_conv_sqp3/K_B_C_s0_ne100_ckpt.npz --case K+B+C

min over lambda >= 0 (sum 1), Lambda >= 0 (4 x 4), mu >= 0 of
    || sum_i lambda_i grad sigma_1(w_i)  -  [tr(Lambda dH_j)]_j  +  mu grad alpha ||
active peaks: sigma within --active-rtol of the max; the compensator enters
through the 2x2-block Hermite matrix H(theta) >= 0 (hurwitz.py) with a PSD
multiplier; the closed-loop abscissa row if within --alpha-tol of -eps.
Cost: the gradients of the active peaks (one closed-loop solve each on the
full model at GL; the reduced oracle at the cavity) and the block's finite
differences on the r x r compensator -- independent of n.  Solved with
Clarabel as an SOCP + PSD cone.  Reported relative to the branch-gradient
scale, as stationarity.py does.
"""
from __future__ import annotations
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse, json
import numpy as np, scipy.sparse as sp, clarabel
from fe_codesign import FECoDesign, parse_case
from packer import Packer
from ctrl_order import infer_r
import hurwitz as hw, controller as ctl
from sqp_inner import _svec, _herm_basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=100); ap.add_argument('--record', required=True)
    ap.add_argument('--case', default='K+B+C'); ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--results', default=None); ap.add_argument('--active-rtol', type=float, default=2e-3)
    ap.add_argument('--alpha-tol', type=float, default=2e-3); ap.add_argument('--json', default=None)
    a = ap.parse_args()
    r = infer_r(ne=a.ne, results=a.results, quiet=True); D = FECoDesign(ne=a.ne, r=r); fl = parse_case(a.case)
    d = np.load(a.record, allow_pickle=True); xa, xs = float(d['xa']), float(d['xs'])
    aa = (d['d'], d['e'], d['f']); nn = d['nvec']; KK = complex(d['K'])
    pk = Packer(D, fl, aa, nn, KK, xa, xs, D.gauss(xa), D.gauss(xs)); z = pk.z0(); NZ = len(z)
    b, c, pa, ps, na, ns = D.bc(xa, xs, None, None); cl = D.cl(b, c)
    W = np.sort(np.asarray(d['active_freqs'], float))
    sv = np.array([cl.gain(float(w), aa, nn, KK, want_vectors=False)['sigma'] for w in W]); g = sv.max()
    act = [int(i) for i in np.where(sv >= g * (1 - a.active_rtol))[0]]
    G = []
    for i in act:
        res = cl.gain(float(W[i]), aa, nn, KK)
        G.append(np.asarray(pk.grad(D.grads(cl, res['x'], res['z'], aa, nn, KK, xa, xs, pa, ps, na, ns, fl)), float))
    G = np.array(G); gscale = float(np.mean(np.linalg.norm(G, axis=1)))
    # compensator block
    Ac = ctl.A_of(*aa); U2, V2, lam2 = hw.cluster_block(Ac, 2, pick=hw.pick_pair(Ac)); lim = -a.eps
    H0, rho = hw.block_hermite(Ac, -lim, U2, V2); sc = max(abs(float(np.trace(H0))), 1e-12)
    dH = []
    for j in range(NZ):
        ez = np.zeros(NZ); ez[j] = 1e-6
        Hp, _ = hw.block_hermite(ctl.A_of(*pk.unpack(z + ez)[0]), -lim, U2, V2, rho)
        Hm, _ = hw.block_hermite(ctl.A_of(*pk.unpack(z - ez)[0]), -lim, U2, V2, rho)
        dH.append((Hp - Hm) / 2e-6)
    # scale-free block (2026-09-25): normalise so the block's gradients are on the
    # peaks' scale (the trace normalisation blew up to 1e13 with both poles on
    # the line, H0 = 0, and Clarabel returned NumericalError)
    sc = max(max(float(np.linalg.norm(dh)) for dh in dH) / max(gscale, 1e-300), 1e-300)
    dH = [dh / sc for dh in dH]
    Hs = H0.shape[0]; nsv = Hs * (Hs + 1) // 2
    DH = np.array([_svec(dh) for dh in dH])              # NZ x nsv:  tr(Lambda dH_j) = svec(Lambda) . svec(dH_j)
    # abscissa row if active
    al, lamb, vv, uu, cla, clr = D.alpha_certified(aa, nn, KK, b, c, eps=a.eps)
    ga = None
    if al + a.eps > -a.alpha_tol and vv is not None:
        eta = np.vdot(uu, cla.Mcl_apply(vv))
        ga = np.asarray(pk.grad(D.grads(cla, vv, uu, aa, nn, KK, xa, xs, pa, ps, na, ns, fl, eta=eta)), float)
    # the OTHER compensator poles on the line (2026-09-25: ne=200 had a simple
    # pole at -0.05 + 6.598i besides the double pole): one row each, mu >= 0
    Xc = [ga] if ga is not None else []
    _pair = hw.pick_pair(Ac)
    from scipy.linalg import eig as _deig
    _w, _VL, _VR = _deig(Ac, left=True, right=True); _r = Ac.shape[0]
    n_other = 0
    for k_ in range(_r):
        if np.min(np.abs(_w[k_] - _pair)) < 1e-9 * max(1.0, abs(_w[k_])) or _w[k_].real < lim - a.alpha_tol:
            continue
        vr, wl = _VR[:, k_], _VL[:, k_]; den = np.vdot(wl, vr)
        Gm = np.outer(wl.conj(), vr) / den
        gc = dict(d=np.array([Gm[i, i] for i in range(_r)]), e=np.array([Gm[i, i + 1] for i in range(_r - 1)]),
                  f=np.array([Gm[i + 1, i] for i in range(_r - 1)]), n=np.zeros(_r, complex), K=0j, xa=0.0, xs=0.0)
        Xc.append(np.real(np.asarray(pk.grad(gc), float))); n_other += 1
    # variables: lambda (na), s = svec(Lambda) (nsv), mu (len(Xc)), t ;  minimise t s.t. ||R|| <= t
    na_ = len(act); nmu = len(Xc); n = na_ + nsv + nmu + 1
    it = n - 1
    # residual R = G^T lambda - DH^T s + ga mu   (NZ x 1)
    Rm = np.zeros((NZ, n)); Rm[:, :na_] = G.T; Rm[:, na_:na_ + nsv] = -DH
    for k_, col in enumerate(Xc):
        Rm[:, na_ + nsv + k_] = col
    rows_A, rows_b, cones = [], [], []
    # equality sum lambda = 1
    e = np.zeros(n); e[:na_] = 1.0; rows_A.append(e); rows_b.append(1.0); cones.append(clarabel.ZeroConeT(1))
    # lambda >= 0, mu >= 0
    for i in range(na_):
        e = np.zeros(n); e[i] = -1.0; rows_A.append(e); rows_b.append(0.0)
    for k_ in range(nmu):
        e = np.zeros(n); e[na_ + nsv + k_] = -1.0; rows_A.append(e); rows_b.append(0.0)
    cones.append(clarabel.NonnegativeConeT(na_ + nmu))
    # SOC: (t, R) in K_soc
    e = np.zeros(n); e[it] = -1.0; rows_A.append(e); rows_b.append(0.0)
    for k in range(NZ):
        rows_A.append(-Rm[k]); rows_b.append(0.0)
    cones.append(clarabel.SecondOrderConeT(NZ + 1))
    # PSD: Lambda >= 0
    for k in range(nsv):
        e = np.zeros(n); e[na_ + k] = -1.0; rows_A.append(e); rows_b.append(0.0)
    cones.append(clarabel.PSDTriangleConeT(Hs))
    A = sp.csc_matrix(np.array(rows_A)); bvec = np.array(rows_b); P = sp.csc_matrix((n, n)); q = np.zeros(n); q[it] = 1.0
    st = clarabel.DefaultSettings(); st.verbose = False
    sol = clarabel.DefaultSolver(P, q, A, bvec, cones, st).solve()
    x = np.asarray(sol.x); rho_kkt = float(x[it]); lam = x[:na_]; Lam_n = float(np.linalg.norm(x[na_:na_ + nsv]))
    # the sigma-hull-only number for comparison
    from scipy.optimize import nnls
    Aeq = np.vstack([G.T, 1e3 * np.ones((1, na_))]); beq = np.concatenate([np.zeros(NZ), [1e3]])
    w, _ = nnls(Aeq, beq); rho_hull = float(np.linalg.norm(G.T @ w))
    out = dict(record=a.record, gamma=float(g), n_active=na_, active_freqs=[float(W[i]) for i in act],
               cluster=[complex(v).real for v in lam2], hurwitz_lambda_min=float(np.linalg.eigvalsh(H0 / sc).min()),
               alpha=float(al), alpha_row_active=bool(ga is not None), gscale=gscale,
               rho_hull=rho_hull, rho_hull_rel=rho_hull / gscale,
               rho_kkt=rho_kkt, rho_kkt_rel=rho_kkt / gscale, weights=lam.tolist(), Lambda_norm=Lam_n, status=str(sol.status))
    # ---- the pair ON its face of the line (2026-09-25): when the compensator pair
    # is nearly a double pole at the line, first-order stationarity is with the
    # rows of S = {c_u <= 0, c_a <= 0, c_b = 0} (sqp_inner.pair_data; at a double
    # pole ON the line the tangent cone of the feasible set is exactly
    # {dc_u <= 0, dc_a <= 0, dc_b = 0}); the Hermite form above is degenerate there
    try:
        import sqp_inner as _sq
        _ev = _sq.Evaluator(D, pk, fl, np.array([0.0]), a.eps, -a.eps, None, False)
        _pd = _ev.pair_data(z)
        if _pd['sep'] <= _sq.PAIR_ENTER_SEP and -_sq.PAIR_ENTER_U <= _pd['exc']:
            Jp = _pd['J']
            n2 = na_ + 3 + nmu + 1; it2 = n2 - 1
            R2 = np.zeros((NZ, n2)); R2[:, :na_] = G.T; R2[:, na_:na_ + 3] = Jp.T
            for k_, col in enumerate(Xc):
                R2[:, na_ + 3 + k_] = col
            rA, rb, cn = [], [], []
            e_ = np.zeros(n2); e_[:na_] = 1.0; rA.append(e_); rb.append(1.0); cn.append(clarabel.ZeroConeT(1))
            nn_ = 0
            for i in list(range(na_)) + [na_, na_ + 1] + [na_ + 3 + k_ for k_ in range(nmu)]:
                e_ = np.zeros(n2); e_[i] = -1.0; rA.append(e_); rb.append(0.0); nn_ += 1
            cn.append(clarabel.NonnegativeConeT(nn_))
            e_ = np.zeros(n2); e_[it2] = -1.0; rA.append(e_); rb.append(0.0)
            for k in range(NZ):
                rA.append(-R2[k]); rb.append(0.0)
            cn.append(clarabel.SecondOrderConeT(NZ + 1))
            q2 = np.zeros(n2); q2[it2] = 1.0
            sol2 = clarabel.DefaultSolver(sp.csc_matrix((n2, n2)), q2, sp.csc_matrix(np.array(rA)), np.array(rb), cn, st).solve()
            x2 = np.asarray(sol2.x)
            out.update(pair_face=True, pair_sep=_pd['sep'], pair_exc=_pd['exc'], pair_c=_pd['c'].tolist(),
                       rho_kkt_pair=float(x2[it2]), rho_kkt_pair_rel=float(x2[it2]) / gscale,
                       pair_mu=[float(v) for v in x2[na_:na_ + 3]], pair_status=str(sol2.status))
    except Exception as _e:
        out.update(pair_face_error=f"{type(_e).__name__}: {_e}")
    print(f"gamma {g:.6f}  active peaks {na_} at {np.round([W[i] for i in act], 3)}  cluster Re {np.round(out['cluster'], 5)}  "
          f"block H lambda_min {out['hurwitz_lambda_min']:+.2e}  alpha {al:+.4f} (row {'active' if nmu else 'inactive'})")
    print(f"rho on the sigma hull only: {rho_hull:.3e} = {rho_hull / gscale:.2e} x gradient scale")
    print(f"rho KKT (hull + block PSD multiplier{' + abscissa' if ga is not None else ''}{f' + {n_other} other pole(s)' if n_other else ''}): {rho_kkt:.3e} = {rho_kkt / gscale:.2e} x gradient scale   "
          f"[{sol.status}; |Lambda| {Lam_n:.2e}; lambda {np.round(lam, 3)}]")
    if out.get('pair_face'):
        print(f"rho KKT with the pair ON its face (S rows: mu_u, mu_a >= 0, mu_b free{' + abscissa' if ga is not None else ''}{f' + {n_other} other pole(s)' if n_other else ''}): "
              f"{out['rho_kkt_pair']:.3e} = {out['rho_kkt_pair_rel']:.2e} x gradient scale   [{out['pair_status']}; "
              f"sep {out['pair_sep']:.1e}, exc {out['pair_exc']:+.1e}; mu {np.round(out['pair_mu'], 4)}]")
    elif out.get('pair_face_error'):
        print("pair face:", out['pair_face_error'])
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    raise SystemExit(main())
