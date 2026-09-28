"""
check_symmetry.py — is the actuator/sensor mirror symmetry exact?

The Ginzburg-Landau operator A = -nu d_x + gamma d_xx + mu(x) has mu REAL and
EVEN (mu = mu0 - cu^2 + mu2 x^2 / 2 with mu0, cu, mu2 real), so with R the
spatial reflection,

    A^H = R conj(A) R          exactly (verified: 3e-16 on the assembled A),

and since the d->z channels (full-field load, energy output) are
R-invariant and the compensator is SISO (its scalar transfer function is
invariant under transposition), the whole co-design problem -- gamma AND the
spectral abscissa -- is invariant under

    S:  psi_a  ->  R conj(psi_s),   psi_s  ->  R conj(psi_a),
        xa -> -xs,  xs -> -xa,      compensator unchanged.

(Verified on a fixed-compensator design: gamma and alpha match to 1e-14.
B+C found xa = -xs = 1.2826 to four decimals without being told.)

At any optimum, therefore, EITHER the design is S-symmetric, OR optima come
in S-pairs with equal gamma (spontaneous symmetry breaking).  A visibly
asymmetric result is decided by ONE evaluation -- the SYMMETRIZED design:

    strictly better  ->  the run was NOT converged; the symmetrized design is
                         a free warm start (--save writes a resume file);
    UNSTABLE         ->  the pair is separated by an unstable ridge: the
                         asymmetry is PROTECTED by the stability constraint;
                         genuine pair, converged;
    strictly worse   ->  genuine pair with a performance barrier; converged.

Two lessons from this study are baked in: the symmetrized design is
STABILITY-CHECKED before its gamma is believed (an unstable design's gain
curve means nothing), and the evaluation band is each run's OWN band from its
json, not a hand-typed one.

BOTH fixed points of the symmetry are valid.  psi_a = -R conj(psi_s) is as
good as psi_a = +R conj(psi_s) when the loop can absorb the sign (b -> -b,
Dc -> -Dc), and for the free-gain cases a CONTINUOUS phase is gauge
(b -> e^{i phi} b is absorbed by K and n).  Symmetrising on the wrong branch
averages two near-cancelling fields and destroys the design -- one case
reported gamma = 81 that way, an artefact of the test, not of the design.
The script tries every applicable branch and keeps the best STABLE one.

    python check_symmetry.py --ne 200 --results results
    python check_symmetry.py --ne 200 --results results --case FB+FC --save
"""
from __future__ import annotations
# --- repository root on sys.path (moved into tools/ by cleanup_archive.py)
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
# --- end root shim
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, glob, json, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_codesign
from fe_codesign import parse_case


def peak(D, arow, nvec, K, b, c, W):
    """Grid max over the run's own band, polished by the same local peak
    climber the runs use.  Both designs of a comparison see the same grid, so
    the CHANGE is meaningful even where the absolute value carries grid
    error."""
    cl = D.cl(b, c)
    g = np.array([cl.gain(float(w), arow, nvec, K,
                          want_vectors=False)['sigma'] for w in W])
    w0 = float(W[int(g.argmax())])
    try:
        p = cl.peak(w0, arow, nvec, K)
        return max(float(g.max()), float(p['sigma']))
    except Exception:
        return float(g.max())


def alpha_of(D, arow, nvec, K, b, c, eps):
    if (D.n + D.r) <= 3000:
        return D.alpha_dense(arow, nvec, K, b, c)[0], 'dense'
    return D.alpha_certified(arow, nvec, K, b, c, eps=eps)[0], 'line'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results')
    ap.add_argument('--r', type=int, default=-1,
                    help='controller order; -1 reads it from '
                         'the seed catalogue, which is where it is decided')
    ap.add_argument('--nw', type=int, default=3001)
    ap.add_argument('--eps', type=float, default=0.05,
                    help='stability margin the runs used')
    ap.add_argument('--case', default=None, help='restrict to one case')
    ap.add_argument('--save', action='store_true',
                    help="write {tag}_sym.npz/.json whenever symmetrising "
                         "strictly improves, for run_case_epi.py --resume")
    a = ap.parse_args()
    if getattr(a, 'r', -1) is not None and a.r < 0:
        from ctrl_order import infer_r
        a.r = infer_r(a.ne, getattr(a, 'results', None))
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=0.8)
    # the licence for everything below, checked rather than assumed:
    # R A^T R = A  (equivalently A^H = R conj(A) R)
    A = D.A.tocsr()
    e_op = float(abs(A.T.tocsr()[::-1].tocsc()[:, ::-1].tocsr()
                     - A).max() / abs(A).max())
    print(f"operator: ||R A^T R - A|| / ||A|| = {e_op:.1e}"
          + ("  (symmetry holds)" if e_op < 1e-12 else
             "  *** SYMMETRY BROKEN -- verdicts below do not apply ***"))
    print(f"{'case':12s} {'gamma':>9s} {'gamma sym':>10s} {'change':>9s} "
          f"{'residual':>9s} {'branch':>6s} {'alpha sym':>10s}  verdict")
    for jf in sorted(glob.glob(os.path.join(a.results, f"*_ne{a.ne}.json"))):
        m = json.load(open(jf))
        if a.case and m.get('case') != a.case:
            continue
        d = np.load(jf[:-5] + '.npz', allow_pickle=True)
        if 'd' not in d.files:
            continue
        fl = parse_case(m['case'])
        both_fld = bool(fl.get('FB')) and bool(fl.get('FC'))
        both_pos = bool(fl.get('B')) and bool(fl.get('C'))
        if not (both_fld or both_pos):
            continue            # S maps one-sided cases to DIFFERENT cases
        arow = (d['d'], d['e'], d['f']); nv = d['nvec']; K = complex(d['K'])
        xa, xs = float(d['xa']), float(d['xs'])
        wlo = float(m.get('wlo', -30.0)); whi = float(m.get('whi', 30.0))
        W = np.linspace(wlo, whi, a.nw)
        pa = d['psi_a'] if 'psi_a' in d.files else np.zeros(0)
        ps = d['psi_s'] if 'psi_s' in d.files else np.zeros(0)

        b0, c0, *_ = D.bc(xa, xs,
                          pa if (both_fld and pa.size) else None,
                          ps if (both_fld and ps.size) else None)
        g0 = peak(D, arow, nv, K, b0, c0, W)

        # candidate symmetrizations: every applicable branch
        cands = []
        if both_fld and pa.size and ps.size and pa.size == ps.size:
            mir = np.conj(ps[::-1])
            na_ = max(np.linalg.norm(pa), 1e-30)
            r_p = float(np.linalg.norm(pa - mir) / na_)
            r_m = float(np.linalg.norm(pa + mir) / na_)
            cands.append(('sym', r_p, 0.5 * (pa + mir), xa, xs))
            cands.append(('ANTI', r_m, 0.5 * (pa - mir), xa, xs))
            if fl.get('K'):
                # free gain: a CONTINUOUS phase is gauge, so align it first
                z = complex(np.vdot(pa, mir))
                if abs(z) > 0:
                    mir_al = mir * np.exp(-1j * np.angle(z))
                    r_f = float(np.linalg.norm(pa - mir_al) / na_)
                    cands.append(('phase', r_f, 0.5 * (pa + mir_al), xa, xs))
        else:
            # position case: the representable symmetrization moves the
            # Gaussians to mirrored positions
            xm = 0.5 * (xa - xs)
            cands.append(('pos', abs(xa + xs), None, xm, -xm))

        best = None
        for name, res, pa_s, xa2, xs2 in cands:
            if pa_s is not None:
                sgn = -1.0 if name == 'ANTI' else 1.0
                ps_s = sgn * np.conj(pa_s[::-1])
                try:
                    b1, c1, *_ = D.bc(xa2, xs2, pa_s, ps_s)
                except FloatingPointError:
                    continue        # branch annihilated the field
                sv = dict(va=pa_s, vs=ps_s)
            else:
                b1, c1, *_ = D.bc(xa2, xs2, None, None)
                sv = dict(va=None, vs=None)
            # STABILITY FIRST: an unstable design's gain curve means nothing
            al1, how1 = alpha_of(D, arow, nv, K, b1, c1, a.eps)
            if al1 + a.eps > 0.0:
                cand = (name, res, None, al1, sv, xa2, xs2)
            else:
                g1 = peak(D, arow, nv, K, b1, c1, W)
                cand = (name, res, g1, al1, sv, xa2, xs2)
            if best is None or (cand[2] is not None
                                and (best[2] is None or cand[2] < best[2])):
                # prefer the branch with the smallest residual among equals
                if best is None or best[2] is None or cand[2] is None \
                        or cand[2] < best[2] - 1e-12 or cand[1] < best[1]:
                    best = cand
        if best is None:
            continue
        name, res, g1, al1, sv, xa2, xs2 = best
        if g1 is None:
            verdict = ("symmetric point UNSTABLE -> asymmetry PROTECTED by "
                       "the stability constraint; genuine pair, converged")
            g1s = float('nan'); rel = float('nan')
        else:
            rel = (g1 - g0) / g0
            g1s = g1
            if abs(rel) < 5e-3:
                verdict = 'symmetry EXACT to optimiser tolerance'
            elif rel < 0:
                verdict = 'symmetrising HELPS -- the run was under-converged'
            else:
                verdict = 'genuine asymmetry -- report it as approximate'
        print(f"{m['case']:12s} {g0:9.4f} {g1s:10.4f} {rel:+8.2%} "
              f"{res:9.2%} {name:>6s} {al1:+10.4f}  {verdict}")
        if a.save and g1 is not None and rel < -5e-3:
            tag = jf[:-5] + '_sym'
            va2 = sv['va']; vs2 = sv['vs']
            np.savez(tag + '.npz', d=arow[0], e=arow[1], f=arow[2], nvec=nv,
                     K=K, xa=xa2, xs=xs2,
                     va=(va2 if va2 is not None else np.zeros(0)),
                     vs=(vs2 if vs2 is not None else np.zeros(0)),
                     xi_a=(D.smooth_unmap(va2) if va2 is not None
                           else np.zeros(0)),
                     xi_s=(D.smooth_unmap(vs2) if vs2 is not None
                           else np.zeros(0)),
                     psi_a=(va2 if va2 is not None else np.zeros(0)),
                     psi_s=(vs2 if vs2 is not None else np.zeros(0)),
                     gamma=g1, alpha=al1, alpha_dense=al1)
            json.dump(dict(case=m['case'], gamma=float(g1), alpha=float(al1),
                           wlo=wlo, whi=whi,
                           note='symmetrized by check_symmetry.py'),
                      open(tag + '.json', 'w'))
            print(f"    saved {os.path.basename(tag)}.npz -- polish with:\n"
                  f"      python run_case_epi.py --case '{m['case']}' --eps {a.eps} "
                  f"--start 0 --ne {a.ne} --eps {a.eps} "
                  f"--resume {tag}.npz --outdir {a.results}")
    print("\nNote: gamma here is a fine sweep on each run's OWN band plus a")
    print("local polish; only the CHANGE matters.  'residual' is the distance")
    print("to the nearest symmetric branch (before symmetrising).")


if __name__ == "__main__":
    main()
