#!/usr/bin/env python3
"""check_parallel.py -- the full-model pool against the serial code (STAGE6 6.30).

    python tools/check_parallel.py --nproc N --out res_N.npz     (in any tree)
    python tools/check_parallel.py --compare A.npz B.npz

Computes, on the order-5 K+B+C record and the atoms record (MIMO) and the SISO
K+B+C record, every task the pool runs: full sigma sweeps, the sparse
H-infinity certificate (decision generations, location shifts, candidate
samples), the enrichment of a reduced basis (anchors + eigen-anchors), the
closed-loop set right of -1 (refinement shifts, band tiles, winding count), the
plant enclosure and heights, winding counts.  --compare prints, per quantity,
whether two runs agree BIT FOR BIT (the winding point counts may differ when
the march is split into segments; the counts may not).  A tree without
fullpar.py (before 6.30) runs everything serially: the reference."""
import os, sys, time, argparse, json
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

RECS = ['results_mimo_prod/kbc_cen5/K_B_C_s0_ne100_mimo2x2_central.npz',
        'results_mimo_prod/at_dec/KGC_FB_FC_s0_ne100_mimo2x2_decentral.npz',
        'results_ne100/K_B_C_s0_ne100.npz']


def load(rec):
    from fe_codesign import FECoDesign
    z = np.load(rec, allow_pickle=True)
    r = len(z['d'])
    D = FECoDesign(ne=100, r=r)
    if 'Bc' in z.files:                                    # MIMO
        va = list(z['va']) if z['va'].size else None; vs = list(z['vs']) if z['vs'].size else None
        b, c, *_ = D.bc(np.asarray(z['xa'], float), np.asarray(z['xs'], float), va, vs)
        arow = tuple(np.asarray(z[k], complex) for k in ('d', 'e', 'f', 'Bc'))
        return z, D, b, c, arow, np.asarray(z['nvec'], complex), np.asarray(z['K'], complex)
    b, c, *_ = D.bc(float(z['xa']), float(z['xs']), None, None)
    arow = tuple(np.asarray(z[k], complex) for k in ('d', 'e', 'f'))
    return z, D, b, c, arow, np.asarray(z['nvec'], complex), complex(z['K'])


def run(nproc, out):
    import controller as ctl, certify_sparse as csp, sparse_spectrum as ss, winding_cert as wc
    from reduced import ReducedOracle, input_basis
    try:
        import fullpar
    except ImportError:
        fullpar = None
    res, tim = {}, {}
    for ir, rec in enumerate(RECS):
        z, D, b, c, arow, Cc, Dc = load(rec)
        import massops; massops.warmup(D.M, verbose=False)
        mimo = len(arow) == 4
        Pb = input_basis(D, 'b', -1.5, 1.5, other=0.98)[0]
        Pc = input_basis(D, 'c', -1.5, 1.5, other=-1.03)[0]
        if fullpar is not None:
            csp.register_plant(D.A, D.M)
            fullpar.start(nproc, D=D, Phi_b=Pb, Psi_c=Pc)
        cl = D.cl(b, c)
        sig = lambda w: cl.gain(float(w), arow, Cc, Dc, want_vectors=False)['sigma']
        pre = f"r{ir}_"
        t0 = time.time()
        ws = np.linspace(-6, 6, 25)
        res[pre + 'sweep'] = np.array(fullpar.sigmas(b, c, arow, Cc, Dc, ws) if (fullpar and fullpar.active()) else [sig(w) for w in ws])
        tim[pre + 'sweep'] = time.time() - t0
        # certificate (strict disks, near-k 2, as the nominal loop)
        csp.DISK_STRICT = True; csp.NEAR_K = 2
        Ac = ctl.A_of(*arow[:3]); Bc = ctl.Bc_of(arow, D.r, D.mc)
        kw = {}
        if fullpar is not None and fullpar.active():
            kw['sigma_many'] = lambda ws_: fullpar.sigmas(b, c, arow, Cc, Dc, ws_)
        t0 = time.time()
        g, w, nc, ok = csp.hinf_certified_sparse(D.A, D.M, b, c, Ac, Bc, Cc, Dc, sig, gam_lb=0.9 * float(z['gamma']),
                                                 wlo=-30.0, whi=30.0, rtol=1e-6, **kw)
        tim[pre + 'cert'] = time.time() - t0
        res[pre + 'cert'] = np.array([g, w, nc, float(ok)])
        # enrichment: 6 anchors + 2 eigen-anchors
        ro = ReducedOracle(D); ro.galerkin = True; ro.set_input_basis(Pb, Pc)
        t0 = time.time()
        anch = [-2.0, -0.7, 0.3, 1.9, 5.0, -0.52]
        if hasattr(ro, 'enrich_many'):
            ro.enrich_many(cl, anch, arow, Cc, Dc, nvec_dirs=3, bc=(b, c))
        else:
            for a_ in anch:
                ro.enrich(cl, a_, arow, Cc, Dc, nvec_dirs=3)
        lam_ = np.linalg.eigvals(np.linalg.solve(ro._proj['Mt'], ro._proj['At']))
        lams = [complex(x) for x in lam_[np.argsort(-lam_.real)][:2]]
        try:
            ro.enrich_eig(cl, lams, arow, Cc, Dc, bc=(b, c))
        except TypeError:
            ro.enrich_eig(cl, lams, arow, Cc, Dc)
        tim[pre + 'enrich'] = time.time() - t0
        res[pre + 'basis'] = ro.V
        # closed-loop set right of -1 with the band tiled and the count
        t0 = time.time()
        loc, info = ss.closed_loop_set(D.A, D.M, D.cl(b, c), arow, Cc, Dc, -1.0, band=12.0, mc=D.mc, count=True)
        tim[pre + 'clset'] = time.time() - t0
        res[pre + 'clset'] = np.array(sorted(loc, key=lambda q: (q.real, q.imag)), complex)
        res[pre + 'clset_count'] = np.array([info.get('count') if info.get('count') is not None else -1,
                                             int(bool(info.get('complete')))])
        # winding counts at the two lines of the loop
        t0 = time.time()
        for L in (-0.025, -0.05):
            o = wc.winding_count(D.A, D.M, b, c, arow, Cc, Dc, line=L, mc=D.mc, ctrl_exact=True)
            res[pre + f'wind{L}'] = np.array([o['count'], int(o['ok'])])
            res[pre + f'windpts{L}'] = np.array([o['npts']])
        tim[pre + 'wind'] = time.time() - t0
        if ir == 0:
            # plant enclosure and heights, fresh caches
            ss._PLANT_CACHE.clear(); ss._HEIGHT_CACHE.clear()
            t0 = time.time()
            lamp, infop = ss.plant_eigs_right(D.A, D.M, -1.6)
            s_far, Hd = ss.plant_heights(D.A, D.M)
            tim['plant'] = time.time() - t0
            res['plant_eigs'] = np.array(lamp, complex)
            res['plant_heights'] = np.array([s_far] + [h for _, h in Hd])
        if fullpar is not None:
            fullpar.stop()
    np.savez(out, **res)
    json.dump(tim, open(out + '.time.json', 'w'), indent=1)
    print(json.dumps({k: round(v, 3) for k, v in tim.items()}))


def compare(fa, fb):
    A, B = np.load(fa), np.load(fb)
    ok_all = True
    for k in sorted(A.files):
        a, b = A[k], B[k]
        if 'windpts' in k:
            print(f"  {k:24s} {int(a[0])} / {int(b[0])} points (segments may differ)")
            continue
        same = a.shape == b.shape and np.array_equal(a, b)
        ok_all &= same
        extra = '' if same or a.shape != b.shape else f"  max diff {np.max(np.abs(a - b)):.1e}"
        print(f"  {k:24s} {'IDENTICAL' if same else 'DIFFERENT'}{extra}")
    print("ALL IDENTICAL" if ok_all else "DIFFERENCES FOUND")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=1)
    ap.add_argument('--out')
    ap.add_argument('--compare', nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run(a.nproc, a.out)
