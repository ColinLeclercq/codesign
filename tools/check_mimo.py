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
  3. central finite differences of sigma_1 w.r.t. every controller entry kind
     (D_c, C_c, B_c, A_c diagonal) and the port-side vectors (d sigma = Re(z_p^H dB_i
     act_i + sen_j dC_j^H x_p)) along random load/sensor perturbations;
  4. the record's certified gamma vs the dense level set on the full even pencil
     (certify_sparse.hinf_dense_levelset: ALL eigenvalues at each level) over the
     imaginary axis (every crossing, all frequencies);
  5. the winding count right of Re = -eps and of Re = 0 vs the dense spectrum;
  6. the abscissa certificate (FECoDesign.alpha_certified, MIMO region) vs the
     dense spectrum;
  7. the SPARSE all-frequency certificate (tail bound from the numerical range +
     sparse level set, sparse_spectrum.closed_loop_tail) vs 4.
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
    # 3. central finite differences, at the active frequency of largest sigma and
    #    at a generic frequency (w = +2); error relative to the largest derivative
    #    tested (at a design on the rank floor the derivatives at the witness are ~0)
    _af = np.asarray(z['active_freqs'], float) if np.size(z['active_freqs']) else np.array([-0.7])
    w_top = float(_af[np.argmax([cl.gain(float(w), arow, Cc, Dc, want_vectors=False)['sigma'] for w in _af])])
    rng = np.random.default_rng(1)

    def rnd(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    def fd_at(w0):
        res = cl.gain(w0, arow, Cc, Dc)
        g = cl.design_grad_tri(res); act, sen = cl.port_sides(res, arow, Cc, Dc)
        s12 = [q['sigma'] for q in cl.gain_k(w0, arow, Cc, Dc, k=2)]

        def sig(ar, nv, K_, bb=b, cc=c):
            return D.cl(bb, cc).gain(w0, ar, nv, K_, want_vectors=False)['sigma']

        def cd(fp, fm, h):
            return (fp - fm) / (2 * h)
        fd = []
        base = dict(K=Dc, n=Cc, Bc=arow[3], d=arow[0])
        for kind in ('K', 'n', 'Bc', 'd'):
            h = 1e-7 * max(1.0, float(np.max(np.abs(base[kind]))))
            dth = rnd(np.shape(base[kind]))
            pred = float(np.real(np.sum(np.reshape(g[kind], np.shape(dth)) * dth)))
            vals = []
            for sgn in (1, -1):
                ar, nv, K_ = list(arow), Cc, Dc
                if kind == 'K':
                    K_ = Dc + sgn * h * dth
                elif kind == 'n':
                    nv = Cc + sgn * h * dth
                elif kind == 'Bc':
                    ar[3] = arow[3] + sgn * h * dth
                else:
                    ar[0] = arow[0] + sgn * h * dth
                vals.append(sig(tuple(ar), nv, K_))
            fd.append((kind, pred, cd(*vals, h)))
        h = 1e-6
        for j in range(b.shape[1]):
            db = np.zeros_like(b); db[:, j] = D.M @ rnd(n) * np.linalg.norm(b[:, j]) / np.sqrt(n)
            pred = float(np.real(np.vdot(res['z'][:n], db[:, j]) * act[j]))
            fd.append((f'B{j + 1}', pred, cd(sig(arow, Cc, Dc, b + h * db, c), sig(arow, Cc, Dc, b - h * db, c), h)))
        for j in range(c.shape[1]):
            dc = np.zeros_like(c); dc[:, j] = D.M @ rnd(n) * np.linalg.norm(c[:, j]) / np.sqrt(n)
            pred = float(np.real(sen[j] * np.vdot(dc[:, j], res['x'][:n])))
            fd.append((f'C{j + 1}', pred, cd(sig(arow, Cc, Dc, b, c + h * dc), sig(arow, Cc, Dc, b, c - h * dc), h)))
        scale = max(max(abs(f), abs(p)) for _, p, f in fd)
        e = max(abs(p - f) for _, p, f in fd) / max(scale, 1e-300)
        print(f"3. finite differences at w = {w0:+.4f} (sigma_1 {s12[0]:.6f}, sigma_2 {s12[1]:.6f}), analytic/FD: " +
              ", ".join(f"{k} {p:+.4e}/{f:+.4e}" for k, p, f in fd) +
              f"  -> max error {e:.1e} of the largest derivative {scale:.2e}")
        return dict(w=w0, sigma12=s12, fd=[(k, p, f) for k, p, f in fd], err=e, scale=scale)
    out['fd'] = [fd_at(w_top), fd_at(2.0)]
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
    # 7. the sparse all-frequency certificate (the driver's, at save since STAGE6 6.19):
    #    tail bound W from the numerical range, then the sparse level set on [-W, W]
    import sparse_spectrum as ss
    t0 = __import__('time').time()
    Wt, ti = ss.closed_loop_tail(D.A, D.M, b, c, ctl.A_of(*arow[:3]), np.asarray(arow[3]), Cc, Dc, float(z['gamma']))
    if np.isfinite(Wt):
        gs, ws_, nc, oks = csp.hinf_certified_sparse(D.A, D.M, b, c, ctl.A_of(*arow[:3]), np.asarray(arow[3]), Cc, Dc,
                                                     sof, float(z['gamma']), wlo=-Wt, whi=Wt, rtol=1e-6)
    else:
        gs, ws_, nc, oks = np.nan, np.nan, 0, False
    out['all_w_sparse'] = dict(tail_W=Wt, gamma=gs, certified=oks, calls=nc)
    print(f"7. sparse all-w certificate: tail bound W = {Wt:.2f}; level set on [-W, W] certified {oks} at {gs:.6f} "
          f"({nc} calls, {__import__('time').time() - t0:.1f} s); dense (4.) {gd:.6f}")
    json.dump(out, open(rec[:-4] + '_check.json', 'w'), indent=1, default=str)
    print(f"-> {rec[:-4]}_check.json")


if __name__ == '__main__':
    main()
