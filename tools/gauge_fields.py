"""
gauge_fields.py — rotate stored designs onto the LITERAL symmetric branch.

For cases with a free complex gain, the global phase of each field is GAUGE:
the closed loop is invariant under

    psi_a -> e^{i ta} psi_a,   psi_s -> e^{i ts} psi_s,
    K     -> K e^{i(ts-ta)},   n     -> n e^{i(ts-ta)},

so an optimum that is exactly mirror-symmetric up to phase
(psi_s = e^{i phi} conj(R psi_a), the 'phase' branch of check_symmetry.py)
PLOTS as asymmetric Re/Im panels, because Re psi_s(x) is a phase-rotated
mixture of Re and Im of psi_a(-x).  That is presentation, not physics.

This tool rotates such designs onto the literal branch
psi_s = conj(R psi_a) exactly -- choosing ta so that psi_a is real-positive
at its own modulus peak (a reproducible canonical gauge) and ts to close the
mirror -- and VERIFIES before writing that gamma-relevant invariants are
untouched: the dense spectral abscissa must match to 1e-10 and the mirror
mismatch must drop to roundoff.  Only then does --apply rewrite the npz
(va, vs, xi_a, xi_s, psi_a, psi_s, K, nvec).  Without --apply it is a dry
run that prints what it would do.

Cases WITHOUT a free gain (no K in the case name) have no field-phase gauge:
their loop is pinned by the fixed compensator, any asymmetry is genuine, and
this tool leaves them alone with a note.

    python gauge_fields.py --ne 200 --results results            # dry run
    python gauge_fields.py --ne 200 --results results --apply
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results')
    ap.add_argument('--r', type=int, default=-1)
    ap.add_argument('--sob_len', type=float, default=0.8)
    ap.add_argument('--case', default=None)
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    if a.r < 0:
        from ctrl_order import infer_r
        a.r = infer_r(a.ne, getattr(a, 'results', None))
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=a.sob_len)
    print(f"{'case':12s} {'branch res':>10s} {'-> literal':>10s} "
          f"{'phi':>8s} {'|dalpha|':>9s}  action")
    for jf in sorted(glob.glob(os.path.join(a.results,
                                            f"*_ne{a.ne}.json"))):
        m = json.load(open(jf))
        if a.case and m.get('case') != a.case:
            continue
        d = np.load(jf[:-5] + '.npz', allow_pickle=True)
        if 'd' not in d.files or not d['va'].size or not d['vs'].size:
            continue
        fl = parse_case(m['case'])
        if not fl.get('K'):
            print(f"{m['case']:12s} {'':>10s} {'':>10s} {'':>8s} {'':>9s}  "
                  f"no free gain: field phase is PHYSICAL, not gauge -- "
                  f"left untouched")
            continue
        arow = (d['d'], d['e'], d['f'])
        nv = np.asarray(d['nvec'], complex)
        K = complex(d['K'])
        xa, xs = float(d['xa']), float(d['xs'])
        va = np.asarray(d['va'], complex)
        vs = np.asarray(d['vs'], complex)
        _, _, pa, ps, *_ = D.bc(xa, xs, va, vs)
        mir = np.conj(ps[::-1])
        r0 = float(np.linalg.norm(pa - mir) / np.linalg.norm(pa))
        # ta: two-stage canonical gauge.  Stage 1 (peak-real) is replaced by
        # the visually canonical choice: the literal branch psi_s=conj(R psi_a)
        # is preserved by the ONE-PARAMETER family psi_a -> e^{i th} psi_a,
        # psi_s -> e^{-i th} psi_s, and th = -arg(sum psi_a^2)/2 maximises the
        # real-part energy of psi_a ("as real as possible"), which is what an
        # eye reads as symmetric in the Re/Im panels -- the sym-run's natural
        # gauge lands close to it.  Any th is physically identical.
        j = int(np.argmax(np.abs(pa)))
        ta = -np.angle(pa[j])
        pa2 = pa * np.exp(1j * ta)
        # ts: close the mirror onto the literal branch
        z = complex(np.vdot(np.conj(pa2[::-1]), ps))
        ts = -np.angle(z) if abs(z) > 0 else 0.0
        ps2 = ps * np.exp(1j * ts)
        # common-phase stage: maximise ||Re psi_a||^2 within the branch
        th = -0.5 * np.angle(np.sum(pa2 * pa2))
        if np.linalg.norm((pa2 * np.exp(1j * th)).real) < \
           np.linalg.norm((pa2 * np.exp(1j * (th + 0.5 * np.pi))).real):
            th += 0.5 * np.pi          # the other stationary point is the max
        pa2 = pa2 * np.exp(1j * th)
        ps2 = ps2 * np.exp(-1j * th)
        ta, ts = ta + th, ts - th
        if pa2.real[int(np.argmax(np.abs(pa2)))] < 0:   # pi is also in the family
            pa2, ps2 = -pa2, -ps2
            ta, ts = ta + np.pi, ts + np.pi
        r1 = float(np.linalg.norm(pa2 - np.conj(ps2[::-1]))
                   / np.linalg.norm(pa2))
        rot = np.exp(1j * (ts - ta))
        K2, nv2 = K * rot, nv * rot
        al0, _ = D.alpha_dense(arow, nv, K, *D.bc(xa, xs, va, vs)[:2])
        b2, c2, *_ = D.bc(xa, xs, pa2, ps2)
        al1, _ = D.alpha_dense(arow, nv2, K2, b2, c2)
        dal = abs(al1 - al0)
        ok = dal < 1e-10 and r1 < 1e-6
        act = ('APPLIED' if (a.apply and ok) else
               'would apply' if ok else
               'SKIPPED (invariance check failed -- genuine asymmetry?)')
        print(f"{m['case']:12s} {r0:9.2%} {r1:10.2e} {ts - ta:+8.4f} "
              f"{dal:9.2e}  {act}")
        if a.apply and ok:
            payload = {k: d[k] for k in d.files}
            payload.update(va=pa2, vs=ps2, psi_a=pa2, psi_s=ps2,
                           xi_a=D.smooth_unmap(pa2),
                           xi_s=D.smooth_unmap(ps2),
                           K=np.complex128(K2), nvec=nv2)
            np.savez(jf[:-5] + '.npz', **payload)
    if not a.apply:
        print("\ndry run: re-run with --apply to write the rotated designs "
              "in place (each is verified gamma/alpha-identical first).")


if __name__ == "__main__":
    main()
