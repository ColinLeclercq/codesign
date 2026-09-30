#!/usr/bin/env python3
"""bench_parallel_2d.py -- speed-up of the full-model pool against the size of a
2-D operator (STAGE6 6.30).

The GL plant is 1-D: its LU is banded and costs ~0.1 ms at n = 199, so on GL the
pool's overhead (fork-pool dispatch, pickling the ports) is a large fraction of
every task.  At the backward-facing step each task is a 2-D finite-element LU
(seconds).  This benchmark runs the SAME task code (fullpar, winding_cert,
certify_sparse, reduced) on a 2-D convection-diffusion-reaction operator
(5-point finite differences on an N x N grid, n = N^2, complex reaction so the
spectrum has oscillatory modes), with 2 actuators, 2 sensors and the order-5
central compensator of the ne=100 record, serial and with the pool:

    python tools/bench_parallel_2d.py --N 40 80 160 --nproc 2 --out bench.json

Per size and task: serial time, pool time, speed-up, and whether the results
agree (bitwise; the winding count by its count).  Nothing else must run on the
machine meanwhile.
"""
import os, sys, time, json, argparse
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import scipy.sparse as sp


class Plant2D:
    """FECoDesign's interface (A, M, n, r, mc, cl) on a 2-D model operator."""
    def __init__(self, N, r):
        L = 20.0; h = L / (N + 1)
        x = np.arange(1, N + 1) * h
        I = sp.identity(N, format='csc')
        D2 = sp.diags([np.ones(N - 1), -2 * np.ones(N), np.ones(N - 1)], [-1, 0, 1], format='csc') / h ** 2
        D1 = sp.diags([-np.ones(N - 1), np.ones(N - 1)], [-1, 1], format='csc') / (2 * h)
        X, Y = np.meshgrid(x, x, indexing='ij')
        mu = (0.2 - 0.02 * (X - 0.4 * L) ** 2 - 0.02 * (Y - 0.5 * L) ** 2) + 1j * (0.5 + 0.02 * X)
        lap = sp.kron(D2, I) + sp.kron(I, D2)
        adv = sp.kron(D1, I)
        self.A = (0.5 * lap - 1.0 * adv + sp.diags(mu.ravel())).tocsc().astype(complex)
        self.M = sp.identity(N * N, format='csc', dtype=float)
        self.n, self.r, self.mc = N * N, r, 1.0
        self.X, self.Y, self.L = X, Y, L
        self._lu_cache = {}

    def bump(self, x0, y0, s=1.0):
        v = np.exp(-((self.X - x0) ** 2 + (self.Y - y0) ** 2) / (2 * s ** 2)).ravel().astype(complex)
        return v / np.linalg.norm(v)

    def cl(self, b, c):
        from closedloop_mimo import ClosedLoopResolventMIMO
        R = ClosedLoopResolventMIMO(self.M, self.A, b, c, r=self.r, mc=self.mc)
        R.lu_cache = self._lu_cache; R.lu_cache_max = 8
        return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, nargs='+', default=[40, 80, 160])
    ap.add_argument('--nproc', type=int, default=2)
    ap.add_argument('--out', default='bench_parallel_2d.json')
    ap.add_argument('--tasks', nargs='+', default=['sweep', 'cert', 'enrich', 'wind'])
    a = ap.parse_args()
    import fullpar, sparse_lu, massops, controller as ctl, certify_sparse as csp, winding_cert as wc
    from reduced import ReducedOracle
    sparse_lu.ORDER = 'mmd_at_plus_a'           # the 2-D ordering (--lu-order)
    z = np.load('results_mimo_prod/kbc_cen5/K_B_C_s0_ne100_mimo2x2_central.npz', allow_pickle=True)
    arow = tuple(np.asarray(z[k], complex) for k in ('d', 'e', 'f', 'Bc'))
    Cc = 0.2 * np.asarray(z['nvec'], complex); Dc = 0.2 * np.asarray(z['K'], complex)
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for N in a.N:
        D = Plant2D(N, len(arow[0]))
        massops.warmup(D.M, verbose=False)
        L = D.L
        B = np.column_stack([D.bump(0.3 * L, 0.45 * L), D.bump(0.3 * L, 0.55 * L)])
        C = np.column_stack([D.bump(0.6 * L, 0.45 * L), D.bump(0.6 * L, 0.55 * L)])
        rng = np.random.default_rng(0)
        Pb = np.linalg.qr(rng.standard_normal((D.n, 12)) + 0j)[0]; Pc = np.linalg.qr(rng.standard_normal((D.n, 12)) + 0j)[0]
        import sparse_spectrum as ss
        ss.plant_heights(D.A, D.M); wc.plant_tail_constants(D.A, D.M)      # once per mesh, not timed
        t0 = time.time(); D.cl(B, C).gain(0.5, arow, Cc, Dc, want_vectors=False); t_lu = time.time() - t0
        rec = dict(n=D.n, t_gain=t_lu)
        print(f"N={N} n={D.n}: one gain {t_lu:.3f} s", flush=True)
        for mode in ('serial', 'pool'):
            fullpar.start(1 if mode == 'serial' else a.nproc, D=D, Phi_b=Pb, Psi_c=Pc)
            csp.register_plant(D.A, D.M)
            D._lu_cache.clear()
            cl = D.cl(B, C)
            sig = lambda w: cl.gain(float(w), arow, Cc, Dc, want_vectors=False)['sigma']
            res = {}
            if 'sweep' in a.tasks:
                ws = np.linspace(-3, 3, 16)
                t0 = time.time()
                res['sweep'] = np.array(fullpar.sigmas(B, C, arow, Cc, Dc, ws) if fullpar.active() else [sig(w) for w in ws])
                rec[f'sweep_{mode}'] = time.time() - t0
            if 'cert' in a.tasks:
                csp.DISK_STRICT = True; csp.NEAR_K = 2
                Ac = ctl.A_of(*arow[:3]); Bc = ctl.Bc_of(arow, D.r, D.mc)
                kw = dict(sigma_many=(lambda ws_: fullpar.sigmas(B, C, arow, Cc, Dc, ws_))) if fullpar.active() else {}
                gl = 0.9 * max(res['sweep']) if 'sweep' in res else 1e-3
                t0 = time.time()
                g, w, nc, ok = csp.hinf_certified_sparse(D.A, D.M, B, C, Ac, Bc, Cc, Dc, sig, gam_lb=gl,
                                                         wlo=-6.0, whi=6.0, rtol=1e-4, **kw)
                rec[f'cert_{mode}'] = time.time() - t0
                res['cert'] = np.array([g, w, nc, float(ok)])
            if 'enrich' in a.tasks:
                ro = ReducedOracle(D); ro.galerkin = True; ro.set_input_basis(Pb, Pc)
                t0 = time.time()
                ro.enrich_many(cl, [-2.0, -1.0, -0.3, 0.3, 1.0, 2.0], arow, Cc, Dc, nvec_dirs=3, bc=(B, C))
                rec[f'enrich_{mode}'] = time.time() - t0
                res['enrich'] = ro.V
            if 'wind' in a.tasks:
                t0 = time.time()
                o = wc.winding_count(D.A, D.M, B, C, arow, Cc, Dc, line=-0.025, mc=D.mc, p_plant=0, hot=[],
                                     ctrl_exact=True)
                rec[f'wind_{mode}'] = time.time() - t0
                res['wind'] = np.array([o['count'], int(o['ok'])])
                rec[f'wind_pts_{mode}'] = int(o['npts'])
            fullpar.stop()
            if mode == 'serial':
                ref = res
            else:
                rec['identical'] = {k: bool(np.array_equal(ref[k], res[k])) for k in res}
        for k in a.tasks:
            if f'{k}_serial' in rec and f'{k}_pool' in rec:
                rec[f'{k}_speedup'] = rec[f'{k}_serial'] / max(rec[f'{k}_pool'], 1e-12)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        out[str(D.n)] = rec
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
