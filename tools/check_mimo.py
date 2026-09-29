#!/usr/bin/env python3
"""check_mimo.py -- validation of the MIMO production path on a record (STAGE6 6.2).

    python tools/check_mimo.py results_mimo_prod/at_dec/KGC_FB_FC_s0_ne100_mimo2x2_decentral.npz

Checks, each against an independent DENSE reference (validation only -- the
dense references use an explicit Cholesky factor of M, which the production path
never forms):
  1. sigma_1 of ClosedLoopResolventMIMO (sparse LU + rank-(m+p) Woodbury, M
     products only) vs sigma_max(L^H [S_cl^-1]_11 L), L L^H = M, at the record's
     active frequencies and a grid;
  2. the same class at m = p = 1 vs the SISO class (ClosedLoopResolvent) on a
     SISO record (the first loop of this design);
  3. finite differences of sigma_1 w.r.t. every free controller entry kind
     (D_c, C_c, B_c, A_c diagonal) and the port-side vectors (d sigma = Re(z_p^H dB_i
     act_i + sen_j dC_j^H x_p)) along random load/sensor perturbations;
  4. the record's certified gamma vs the dense level set on the full even pencil
     (certify_sparse.hinf_dense_levelset: ALL eigenvalues at each level) over the
     imaginary axis (every crossing, all frequencies);
  5. the winding count right of Re = -eps and of Re = 0 vs the dense spectrum;
  6. the abscissa certificate (FECoDesign.alpha_certified, MIMO region) vs the
     dense spectrum.
"""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import json
import numpy as np
import scipy.linalg as sl
from fe_codesign import FECoDesign
import controller as ctl
import certify_sparse as csp
import winding_cert as wc
from closedloop_sparse import ClosedLoopResolvent
from closedloop_mimo import ClosedLoopResolventMIMO

SISO_REC = 'results_n100_kbc/K_B_C_s0_ne100.npz'


def load(rec):
    z = np.load(rec, allow_pickle=True)
    r = len(z['d']); D = FECoDesign(ne=int(json.load(open(rec[:-4] + '.json'))['ne']), r=r)
    va = list(z['va']) if z['va'].size else None; vs = list(z['vs']) if z['vs'].size else None
    b, c, *_ = D.bc(np.asarray(z['xa'], float), np.asarray(z['xs'], float), va, vs)
    arow = (np.asarray(z['d'], complex), np.asarray(z['e'], complex), np.asarray(z['f'], complex),
            np.asarray(z['Bc'], complex))
    return z, D, b, c, arow, np.asarray(z['nvec'], complex), np.asarray(z['K'], complex)


def dense_parts(D, b, c, arow, nvec, K):
    n, r = D.n, D.r
    Md = D.M.toarray(); Ad = D.A.toarray()
    Ac = ctl.A_of(*arow[:3]); Bc = np.asarray(arow[3]).reshape(r, -1)
    Acl = np.block([[Ad + b @ K @ c.conj().T, b @ nvec], [Bc @ c.conj().T, Ac]])
    Mcl = sl.block_diag(Md, np.eye(r))
    return Md, Acl, Mcl


def sigma_dense(Md, Lc, Acl, Mcl, n, w):
    Si = np.linalg.inv(1j * w * Mcl - Acl)
    return float(sl.svdvals(Lc.conj().T @ Si[:n, :n] @ Lc)[0])


def main():
    rec = _s.argv[1]
    z, D, b, c, arow, Cc, Dc = load(rec)
    n, r = D.n, D.r
    eps = 0.05
    out = {}
    cl = D.cl(b, c)
    assert isinstance(cl, ClosedLoopResolventMIMO)
    Md, Acl, Mcl = dense_parts(D, b, c, arow, Cc, Dc)
    Lc = sl.cholesky(Md, lower=True)                 # validation only
    # 1. sigma vs dense
    ws = list(np.asarray(z['active_freqs'], float)) + list(np.linspace(-30, 30, 13))
    err = 0.0
    for w in ws:
        s1 = cl.gain(float(w), arow, Cc, Dc, want_vectors=False)['sigma']
        s0 = sigma_dense(Md, Lc, Acl, Mcl, n, float(w))
        err = max(err, abs(s1 - s0) / s0)
    out['sigma_vs_dense_maxrel'] = err
    print(f"1. sigma_1 sparse/Woodbury vs dense Cholesky reference at {len(ws)} frequencies: max rel {err:.1e}")
    # 2. m = p = 1 against the SISO class, on the SISO K+B+C record
    zs = np.load(SISO_REC, allow_pickle=True); r1 = len(zs['d'])
    D1 = FECoDesign(ne=D.ne, r=r1)
    b1, c1, *_ = D1.bc(float(zs['xa']), float(zs['xs']))
    a1 = tuple(np.asarray(zs[k], complex) for k in ('d', 'e', 'f'))
    n1 = np.asarray(zs['nvec'], complex); k1 = complex(zs['K'])
    cs = ClosedLoopResolvent(D1.M, D1.A, b1, c1, r=r1, mc=D1.mc)
    cm = ClosedLoopResolventMIMO(D1.M, D1.A, b1[:, None], c1[:, None], r=r1, mc=D1.mc)
    e2 = eg = 0.0
    for w in (-3.0, -0.7, 0.0, 0.4, 2.5):
        gs = cs.gain(w, a1, n1, k1)
        gm = cm.gain(w, a1 + (np.full((r1, 1), D1.mc, complex),), n1[None, :], np.array([[k1]]))
        e2 = max(e2, abs(gs['sigma'] - gm['sigma']) / gs['sigma'])
        ds = cs.design_grad_tri(gs); dm = cm.design_grad_tri(gm)
        eg = max(eg, max(np.max(np.abs(np.ravel(ds[k]) - np.ravel(dm[k]))) / max(np.max(np.abs(ds[k])), 1e-300)
                         for k in ('d', 'e', 'f', 'n')))
    out['siso_vs_mimo_m1'] = dict(sigma=e2, grad=eg)
    print(f"2. MIMO class at m = p = 1 vs the SISO class ({SISO_REC}): sigma max rel {e2:.1e}, gradients {eg:.1e}")
    # 3. finite differences at the active frequency of largest sigma
    _af = np.asarray(z['active_freqs'], float) if np.size(z['active_freqs']) else np.array([-0.7])
    w0 = float(_af[np.argmax([cl.gain(float(w), arow, Cc, Dc, want_vectors=False)['sigma'] for w in _af])])
    res = cl.gain(w0, arow, Cc, Dc)
    g = cl.design_grad_tri(res); act, sen = cl.port_sides(res, arow, Cc, Dc)
    rng = np.random.default_rng(1); fd = []
    s0 = res['sigma']

    def sig(ar, nv, K_, bb=b, cc=c):
        return D.cl(bb, cc).gain(w0, ar, nv, K_, want_vectors=False)['sigma']

    def rnd(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    base = dict(K=Dc, n=Cc, Bc=arow[3], d=arow[0])
    for kind in ('K', 'n', 'Bc', 'd'):
        h = 1e-7 * max(1.0, float(np.max(np.abs(base[kind]))))
        dth = rnd(np.shape(base[kind]))
        pred = float(np.real(np.sum(np.reshape(g[kind], np.shape(dth)) * dth)))
        ar, nv, K_ = list(arow), Cc, Dc
        if kind == 'K':
            K_ = Dc + h * dth
        elif kind == 'n':
            nv = Cc + h * dth
        elif kind == 'Bc':
            ar[3] = arow[3] + h * dth
        else:
            ar[0] = arow[0] + h * dth
        fd.append((kind, pred, (sig(tuple(ar), nv, K_) - s0) / h))
    h = 1e-7
    for j in range(b.shape[1]):
        db = np.zeros_like(b); db[:, j] = D.M @ rnd(n) * np.linalg.norm(b[:, j]) / np.sqrt(n)
        pred = float(np.real(np.vdot(res['z'][:n], db[:, j]) * act[j]))
        fd.append((f'B{j + 1}', pred, (sig(arow, Cc, Dc, b + h * db, c) - s0) / h))
    for j in range(c.shape[1]):
        dc = np.zeros_like(c); dc[:, j] = D.M @ rnd(n) * np.linalg.norm(c[:, j]) / np.sqrt(n)
        pred = float(np.real(sen[j] * np.vdot(dc[:, j], res['x'][:n])))
        fd.append((f'C{j + 1}', pred, (sig(arow, Cc, Dc, b, c + h * dc) - s0) / h))
    efd = max(abs(p - f) / max(abs(f), abs(p)) for _, p, f in fd if max(abs(f), abs(p)) > 1e-9)
    out['fd'] = [(k, p, f) for k, p, f in fd]; out['fd_maxrel'] = efd
    print(f"3. finite differences at w = {w0:+.4f} (sigma {s0:.6f}), analytic/FD: " +
          ", ".join(f"{k} {p:+.4e}/{f:+.4e}" for k, p, f in fd) + f"  -> max rel {efd:.1e}")
    # 4. dense level set on the full even pencil
    wb = np.inf                                        # every crossing of the imaginary axis
    Ac = ctl.A_of(*arow[:3]); Bc = np.asarray(arow[3])
    sof = lambda w: cl.gain(float(w), arow, Cc, Dc, want_vectors=False)['sigma']
    gd, wd, nl, ok = csp.hinf_dense_levelset(D.A, D.M, b, c, Ac, Bc, Cc, Dc, sof,
                                             0.95 * float(z['gamma']), wlo=-wb, whi=wb, rtol=1e-6)
    out['gamma_record'] = float(z['gamma']); out['gamma_dense_levelset'] = gd
    print(f"4. gamma: record {float(z['gamma']):.6f}, dense level set {gd:.6f} at w = {wd:+.4f} over all w "
          f"({nl} levels, certified {ok}); rel {abs(gd - float(z['gamma'])) / gd:.1e}")
    # 5. winding count vs dense spectrum
    lam = sl.eigvals(Acl, Mcl); lam = lam[np.isfinite(lam)]
    pp = wc.plant_pole_data(D.A, D.M, -eps)
    for line in (-eps, 0.0):
        pp_ = pp if line == -eps else wc.plant_pole_data(D.A, D.M, line)
        o = wc.winding_count(D.A, D.M, b, c, arow, Cc, Dc, line=line, mc=D.mc, p_plant=pp_[0], hot=pp_[1],
                             hmin_frac=1e-8, ctrl_exact=True)
        nd = int(np.count_nonzero(lam.real > line))
        out[f'winding_{line}'] = (o['count'], nd, bool(o['ok']))
        print(f"5. eigenvalues right of Re = {line:+.3f}: winding count {o['count']} (ok {o['ok']}, {o['npts']} pts), dense {nd}")
    # 6. abscissa certificate vs dense
    D.alpha_argmax = True
    al = D.alpha_certified(arow, Cc, Dc, b, c, eps=eps)
    i = int(np.argmax(lam.real))
    out['alpha_cert'] = float(al[0]); out['alpha_dense'] = float(lam[i].real)
    print(f"6. abscissa: certificate {al[0]:+.6f} (clear {al[5]}), dense {lam[i].real:+.6f} at {lam[i]:.4f}")
    json.dump(out, open(rec[:-4] + '_check.json', 'w'), indent=1, default=str)
    print(f"-> {rec[:-4]}_check.json")


if __name__ == '__main__':
    main()
