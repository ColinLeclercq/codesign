#!/usr/bin/env python3
"""gradsample.py -- gradient-sampling certificate of Clarke stationarity
(Burke-Lewis-Overton) at a design, on the reduced oracle.

    python tools/gradsample.py --ne 100 --record results_conv_sqp3/K_B_C_s0_ne100_ckpt.npz --case K+B+C

phi_nu(z) = max_i sigma_1(w_i; z) over the working set  +  nu * max(0, -lambda_min(H_block(z)))
Sample m points z* + rad |z*| u (u uniform in the unit ball); at each, the
gradient of phi_nu is the argmax peak's gradient (reduced oracle, lifted
pairs -> D.grads) plus nu * (-v^T dH v) with v the eigenvector of
lambda_min (finite differences on the r x r compensator).  rho_GS = min-norm
element of the convex hull of the sampled gradients (nnls).  Small relative
to the gradient scale <=> Clarke-stationary for phi_nu at the sampling
radius.  Cost per sample: one reduced evaluation (no factorisation) + the
compensator's ~3r finite differences: independent of n.
"""
from __future__ import annotations
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import argparse, json
import numpy as np
from scipy.optimize import nnls
from fe_codesign import FECoDesign, parse_case
from packer import Packer
from ctrl_order import infer_r
from reduced import ReducedOracle
import hurwitz as hw, controller as ctl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=100); ap.add_argument('--record', required=True)
    ap.add_argument('--case', default='K+B+C'); ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--results', default=None); ap.add_argument('--m', type=int, default=0)
    ap.add_argument('--rad', type=float, default=1e-3); ap.add_argument('--nu', type=float, default=100.0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--json', default=None)
    a = ap.parse_args()
    r = infer_r(ne=a.ne, results=a.results, quiet=True); D = FECoDesign(ne=a.ne, r=r); fl = parse_case(a.case)
    d = np.load(a.record, allow_pickle=True); xa, xs = float(d['xa']), float(d['xs'])
    aa = (d['d'], d['e'], d['f']); nn = d['nvec']; KK = complex(d['K'])
    pk = Packer(D, fl, aa, nn, KK, xa, xs, D.gauss(xa), D.gauss(xs)); z0 = pk.z0(); NZ = len(z0)
    W = np.sort(np.asarray(d['active_freqs'], float))
    b, c, *_ = D.bc(xa, xs, None, None); cl = D.cl(b, c)
    sv = np.array([cl.gain(float(w), aa, nn, KK, want_vectors=False)['sigma'] for w in W])
    # reduced oracle anchored at the endpoint's top peaks (a few LUs) so the
    # samples' sigma rows are accurate in the ball
    ro = ReducedOracle(D)
    for i in np.argsort(-sv)[:10]:
        ro.enrich(cl, float(W[i]), aa, nn, KK, nvec_dirs=3)
    al, lam, *_ = D.alpha_certified(aa, nn, KK, b, c, eps=a.eps); ro.enrich_eig(cl, [lam], aa, nn, KK)
    lim = -a.eps
    Ac0 = ctl.A_of(*aa); U2, V2, lam2 = hw.cluster_block(Ac0, 2)
    H0, rho0 = hw.block_hermite(Ac0, -lim, U2, V2); sc = max(abs(float(np.trace(H0))), 1e-12)

    def grad_phi(z):
        arow, nvec, K, xa_, xs_, va_, vs_ = pk.unpack(z)
        b_, c_, pa, ps, na, ns = D.bc(xa_, xs_, va_, vs_)
        rc = ro.make_cl(b_, c_, D.r, D.mc)
        s = np.array([rc.gain(float(w), arow, nvec, K, want_vectors=False)['sigma'] for w in W])
        i = int(np.argmax(s)); res = rc.gain(float(W[i]), arow, nvec, K)
        g = np.asarray(pk.grad(D.grads(rc.proxy, res['x'], res['z'], arow, nvec, K, xa_, xs_, pa, ps, na, ns, fl)), float)
        # block constraint at the SAMPLE (bases from the sample's own cluster)
        Ac = ctl.A_of(*arow); U, V, _ = hw.cluster_block(Ac, 2)
        H, rho = hw.block_hermite(Ac, -lim, U, V, rho0); H = H / sc
        ev, vecs = np.linalg.eigh(H); k = int(np.argmin(ev)); v = vecs[:, k]
        viol = max(0.0, -ev[k])
        if viol > 0:
            gc = np.zeros(NZ)
            for j in range(NZ):
                ez = np.zeros(NZ); ez[j] = 1e-6
                Hp, _ = hw.block_hermite(ctl.A_of(*pk.unpack(z + ez)[0]), -lim, U, V, rho0)
                Hm, _ = hw.block_hermite(ctl.A_of(*pk.unpack(z - ez)[0]), -lim, U, V, rho0)
                gc[j] = -float(v @ ((Hp - Hm) / 2e-6 / sc) @ v)
            g = g + a.nu * gc
        return g, float(s.max()), viol

    rng = np.random.default_rng(a.seed); m = a.m or 2 * NZ
    G = []; vals = []; nviol = 0
    g0, f0, v0 = grad_phi(z0); G.append(g0); vals.append(f0)
    for _ in range(m):
        u = rng.normal(size=NZ); u /= np.linalg.norm(u); u *= rng.uniform() ** (1.0 / NZ)
        g, f, viol = grad_phi(z0 + a.rad * max(np.linalg.norm(z0), 1.0) * u)
        G.append(g); vals.append(f); nviol += int(viol > 0)
    G = np.array(G); gscale = float(np.mean(np.linalg.norm(G, axis=1)))
    A = np.vstack([G.T, 1e3 * np.ones((1, len(G)))]); bvec = np.concatenate([np.zeros(NZ), [1e3]])
    w, _ = nnls(A, bvec); rho_gs = float(np.linalg.norm(G.T @ w))
    out = dict(record=a.record, m=m, rad=a.rad, nu=a.nu, phi0=f0, block_violations_in_ball=nviol,
               gscale=gscale, rho_gs=rho_gs, rho_gs_rel=rho_gs / gscale, spread=float(np.ptp(vals)))
    print(f"gradient sampling: m={m} rad={a.rad} nu={a.nu}  phi(z*)={f0:.6f}  samples with block violation {nviol}/{m}  "
          f"phi spread in the ball {np.ptp(vals):.2e}")
    print(f"rho_GS (min-norm element of the sampled-gradient hull) = {rho_gs:.3e} = {rho_gs / gscale:.2e} x gradient scale")
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    raise SystemExit(main())
