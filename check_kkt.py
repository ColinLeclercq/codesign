"""
check_kkt.py — is a stability-constrained design actually KKT-stationary?

Three cases sit exactly on the stability boundary (alpha = -eps): there the
peaks need NOT equalise and the free gradient of gamma need NOT vanish -- the
optimality condition is

    grad gamma + mu * grad alpha = 0,   mu > 0,

i.e. the two gradients are ANTI-COLLINEAR with a positive multiplier.  The
existing diagnostics assert this picture ("|grad| is the multiplier, not a
stall") without testing it; this tool tests it, by central finite differences
in the run's own packed coordinates:

    gamma(z):  sigma_1 at the dominant peak, re-polished per evaluation
               (locally smooth while the 2nd peak stays separated);
    alpha(z):  the DENSE spectral abscissa (check-only oracle, GL scale).

It reports the multiplier mu* = -<g_gam, g_al>/||g_al||^2 and the KKT
residual  ||g_gam + mu* g_al|| / ||g_gam||.  A small residual with mu* > 0
CERTIFIES the constrained stationarity that the non-equalised peak picture
merely suggests; a large one means the design can still slide ALONG the
stability boundary and is genuinely unconverged.

Cost: 2 * NZ * (one peak polish + one dense eigensolve) -- a few minutes for
the position cases (NZ ~ 65).  Refuses field cases (NZ ~ 1600 FD solves of a
dense eigenproblem is not a check, it is a job) -- none of them is
alpha-active anyway.

    python check_kkt.py --ne 200 --results results --case K+B+C
    python check_kkt.py --ne 200 --results results          # all alpha-active
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, glob, json, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_codesign
from fe_codesign import parse_case
from packer import Packer
import controller as ctl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results')
    ap.add_argument('--r', type=int, default=-1)
    ap.add_argument('--sob_len', type=float, default=0.8)
    ap.add_argument('--case', default=None)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--h', type=float, default=1e-5,
                    help='central FD step in packed coordinates')
    ap.add_argument('--json', default='',
                    help='also write the results as JSON to this path '
                         '(consumed by harness.py)')
    ap.add_argument('--gbounds', default='',
                    help='comma-separated certified lower bounds on gamma; '
                         'a design at a bound gets that verdict instead of '
                         'a spurious KKT complaint')
    ap.add_argument('--probe', type=int, default=1,
                    help='after a non-stationary verdict, step along the '
                         'PROJECTED descent direction (tangent to the '
                         'stability boundary) and report gamma/alpha -- '
                         'turns the verdict into a demonstrated improvement')
    ap.add_argument('--save', action='store_true',
                    help='write the best probe point as {tag}_kkt.npz for '
                         'run_case_epi.py --resume')
    ap.add_argument('--nzmax', type=int, default=200,
                    help='refuse cases with more packed dofs than this '
                         '(field cases would need ~3000 dense eigensolves)')
    a = ap.parse_args()
    if a.r < 0:
        from ctrl_order import infer_r
        a.r = infer_r(a.ne, getattr(a, 'results', None))
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=a.sob_len)
    print(f"{'case':12s} {'gamma':>9s} {'alpha':>9s} {'NZ':>4s} "
          f"{'mu*':>10s} {'KKT resid':>10s}  verdict")
    _rows = {}
    _nrun = 0
    for jf in sorted(glob.glob(os.path.join(a.results, f"*_ne{a.ne}.json"))):
        m = json.load(open(jf))
        if a.case and m.get('case') != a.case:
            continue
        d = np.load(jf[:-5] + '.npz', allow_pickle=True)
        if 'd' not in d.files:
            continue
        al = float(d['alpha_dense']) if 'alpha_dense' in d.files \
            else float(d['alpha'])
        if not (a.case or abs(al + a.eps) < 0.005 * a.eps):
            continue                     # default: only alpha-active designs
        _nrun += 1
        fl = parse_case(m['case'])
        arow0 = (np.asarray(d['d'], complex), np.asarray(d['e'], complex),
                 np.asarray(d['f'], complex))
        nvec0 = np.asarray(d['nvec'], complex)
        K0 = complex(d['K'])
        xa0, xs0 = float(d['xa']), float(d['xs'])
        va0 = (D.smooth_unmap(np.asarray(d['va'], complex))
               if d['va'].size else None)
        vs0 = (D.smooth_unmap(np.asarray(d['vs'], complex))
               if d['vs'].size else None)
        pk = Packer(D, fl, arow0, nvec0, K0, xa0, xs0, va0, vs0)
        z0 = pk.z0().astype(float)
        if len(z0) > a.nzmax:
            print(f"{m['case']:12s} {m['gamma']:9.4f} {al:9.4f} "
                  f"{len(z0):4d} {'':>10s} {'':>10s}  SKIPPED "
                  f"(NZ > {a.nzmax}: FD not a check at this size)")
            continue
        # dominant peak frequency: seed of every gamma evaluation
        wpk = float(m.get('omega_peak',
                          d['omega_grid'][int(np.argmax(d['gain']))]
                          if 'omega_grid' in d.files else 0.0))

        def design(z):
            arow, nvec, K, xa, xs, va, vs = pk.unpack(z)
            b, c, *_ = D.bc(xa, xs, va, vs)
            return arow, nvec, K, b, c

        def f_gamma(z):
            arow, nvec, K, b, c = design(z)
            cl = D.cl(b, c)
            p = cl.peak(wpk, arow, nvec, K)
            return float(p['sigma'])

        def f_alpha(z):
            arow, nvec, K, b, c = design(z)
            return float(D.alpha_dense(arow, nvec, K, b, c)[0])

        def spectrum(z):
            arow, nvec, K, b, c = design(z)
            return D.alpha_dense(arow, nvec, K, b, c,
                                 return_spectrum=True)[2]

        # ACTIVE SET: every eigenvalue pinned near Re = -eps, not only the
        # rightmost.  An optimiser pressed against a rejection gate
        # manufactures exactly this -- it pushes until SEVERAL modes touch
        # the line -- and then alpha = max is NONSMOOTH: a single FD
        # "gradient" of it is a subgradient mixture, the resulting "tangent"
        # has a first-order component along the other pinned eigenvalues,
        # and both the stationarity verdict and the probe are wrong
        # (measured: the probe left the boundary at slope 0.76 along a
        # direction built to be tangent).  The correct object is the VERTEX:
        # per-eigenvalue gradients, KKT by NNLS over all of them, probe in
        # the intersection tangent.
        lam0 = spectrum(z0)
        act = [int(j) for j in np.argsort(-lam0.real)
               if lam0.real[j] > -a.eps - 0.05 * a.eps][:8]
        mact = len(act)
        nz = len(z0)
        gg = np.zeros(nz)
        gA = np.zeros((mact, nz))
        for j in range(nz):
            zp = z0.copy(); zp[j] += a.h
            zm = z0.copy(); zm[j] -= a.h
            gg[j] = (f_gamma(zp) - f_gamma(zm)) / (2 * a.h)
            lp, lm = spectrum(zp), spectrum(zm)
            for k, idx in enumerate(act):
                # matched FD: at h = 1e-5 each eigenvalue moves O(h), far
                # below the spectral spacing, so nearest-match is exact --
                # and every active gradient comes from the SAME two dense
                # sweeps, so the multi-constraint version costs no more
                # than the single-constraint one did.
                lam_k = lam0[idx]
                rp = lp[np.argmin(np.abs(lp - lam_k))].real
                rm = lm[np.argmin(np.abs(lm - lam_k))].real
                gA[k, j] = (rp - rm) / (2 * a.h)
        ngg = float(np.linalg.norm(gg))
        active = abs(al + a.eps) < 0.005 * a.eps
        gb_hit = next((gb for gb in
                       [float(v) for v in a.gbounds.split(',') if v.strip()]
                       if abs(m['gamma'] - gb) < 1e-4 * gb), None)
        _rows.setdefault(m['case'], {})
        if not active:
            note = (f"AT FUNDAMENTAL BOUND {gb_hit:.6f}: KKT trivial, "
                    f"mu = 0 expected" if gb_hit is not None else
                    (f"constraint INACTIVE (alpha "
                     f"{abs(al+a.eps)/a.eps:.1%} of eps inside): plain "
                     f"stationarity applies, ||grad gamma|| = {ngg:.3e}"
                     if al < -a.eps else
                     f"design NOT margin-feasible (alpha = {al:+.4f} > "
                     f"-eps)?! verify it before anything else"))
            print(f"{m['case']:12s} {m['gamma']:9.4f} {al:9.4f} {nz:4d} "
                  f"{'--':>10s} {'--':>10s}  {note}")
            _rows[m['case']] = dict(case=m['case'], gamma=float(m['gamma']),
                                    alpha=float(al), NZ=int(nz),
                                    alpha_active=False, mu_sum=None,
                                    kkt_residual=None, grad_gamma_norm=ngg,
                                    at_bound=(float(gb_hit) if gb_hit else None),
                                    note=note)
            continue
        from scipy.optimize import nnls
        mu, _ = nnls(gA.T, -gg)
        resid = float(np.linalg.norm(gg + gA.T @ mu) / max(ngg, 1e-30))
        print(f"{m['case']:12s} {m['gamma']:9.4f} {al:9.4f} {nz:4d} "
              f"{'|mu|=' + format(float(np.sum(mu)), '.2e'):>10s} "
              f"{resid:10.2e}  {mact} eigenvalue(s) pinned at the line: "
              + (f"KKT-STATIONARY at the VERTEX (residual {resid:.1%}) -- "
                 f"stability-limited optimum CERTIFIED"
                 if resid < 0.05 else
                 f"near-stationary (residual {resid:.1%})" if resid < 0.2
                 else f"NOT stationary (residual {resid:.1%}): descent "
                      f"exists along the vertex tangent"))
        _rows[m['case']] = dict(case=m['case'], gamma=float(m['gamma']),
                                alpha=float(al), NZ=int(nz),
                                alpha_active=True,
                                mu_sum=float(np.sum(mu)),
                                kkt_residual=float(resid),
                                grad_gamma_norm=ngg,
                                n_pinned=int(mact), at_bound=None)
        if a.probe and resid >= 0.05:
            # probe in the INTERSECTION tangent: remove every active
            # eigenvalue's first-order motion, not just the rightmost's.
            G = gA
            P = np.eye(nz) - G.T @ np.linalg.pinv(G @ G.T) @ G
            dtan = -(P @ gg)
            nd = float(np.linalg.norm(dtan))
            if nd < 1e-12 * ngg:
                print("    projected gradient ~ 0: vertex-stationary to "
                      "numerical precision")
                continue
            dtan /= nd
            g0v = f_gamma(z0)
            best_pt = None
            print(f"    probe along the vertex tangent "
                  f"(gamma0 = {g0v:.6f}, alpha0 = {al:+.6f}, "
                  f"{mact} pinned):")
            ga0 = (gA.T @ np.linalg.pinv(gA @ gA.T)) if mact else None
            for t in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
                zt = z0 + t * dtan
                at = f_alpha(zt)
                # RESTORATION: the boundary is strongly CURVED (measured
                # c ~ 8e3: a straight tangent step drifts off it
                # quadratically and is infeasible past t ~ 1e-5), so pull
                # the point back onto alpha = -eps by Newton steps along the
                # active-gradient directions -- projected gradient descent
                # with restoration, the thing a rejection gate cannot do.
                nrest = 0
                while at > -a.eps - 1e-5 and nrest < 6:
                    r_vec = np.array([f_alpha(zt) + a.eps + 2e-5])
                    zt = zt - (ga0 @ r_vec).ravel()
                    at = f_alpha(zt)
                    nrest += 1
                gt = f_gamma(zt)
                tag_ok = at <= -a.eps + 1e-6
                print(f"      t={t:7.1e}: gamma = {gt:.6f} "
                      f"({(gt-g0v)/g0v:+.2e}), alpha = {at:+.6f}"
                      f"  [{nrest} restorations]"
                      f"{'' if tag_ok else '  (infeasible)'}")
                if tag_ok and gt < g0v and \
                        (best_pt is None or gt < best_pt[1]):
                    best_pt = (t, gt, at, zt)
            if best_pt and a.save:
                t, gt, at, zt = best_pt
                arow2, nvec2, K2, xa2, xs2, va2, vs2 = pk.unpack(zt)
                tagp = jf[:-5] + '_kkt'
                np.savez(tagp + '.npz', d=arow2[0], e=arow2[1], f=arow2[2],
                         nvec=nvec2, K=K2, xa=xa2, xs=xs2,
                         va=(va2 if va2 is not None else np.zeros(0)),
                         vs=(vs2 if vs2 is not None else np.zeros(0)),
                         gamma=gt, alpha=at, alpha_dense=at)
                json.dump(dict(case=m['case'], gamma=float(gt),
                               alpha=float(at),
                               note='vertex-tangent probe point'),
                          open(tagp + '.json', 'w'))
                print(f"    saved {os.path.basename(tagp)}.npz "
                      f"(gamma {gt:.6f}) -- polish with:\n"
                      f"      python run_case_epi.py --case '{m['case']}' --eps {a.eps} "
                      f"--start 0 --ne {a.ne} --eps {a.eps} "
                      f"--resume {tagp}.npz --outdir {a.results}")


    if _nrun == 0:
        print("no margin-active design in this directory: the stability "
              "constraint is inactive everywhere (use --case to force one).")
    _write_json(a.json, _rows)


def _write_json(path, rows):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(rows, open(path, 'w'), indent=1, default=str)
    print(f"json -> {path}")


if __name__ == "__main__":
    main()
