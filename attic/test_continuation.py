"""
test_continuation.py — regression test for the continuation gate.

Reproduces, in miniature, the failure that put eight unstable designs on disk:
a design step large enough to carry a closed-loop eigenvalue across the
stability line between two checks.  Verifies that

  1. bc() now FAILS LOUDLY on a degenerate field (the silent NaN of the sweep
     logs, fe_codesign.py:178-179);
  2. the line certificate is, as documented, BLIND to an eigenvalue already to
     the right of the line -- this is the conditional question of Section 9.3,
     demonstrated rather than assumed;
  3. continuation_step (matrix-free tracking) catches exactly that escape;
  4. the adaptive bisection localises the crossing when the raw step is too
     large for tracking to be conclusive;
  5. alpha_dense (the check-only oracle) agrees with a direct dense eigensolve.

Runs in ~a minute at ne = 50.  No pyGRANSO, no cluster.

    python test_continuation.py
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fe_codesign import FECoDesign
import certify_sparse as csp

NE, R, EPS = 50, 7, 1e-3
npass, nfail = 0, 0


def check(name, ok, detail=""):
    global npass, nfail
    npass += ok; nfail += (not ok)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail
                                                    else ""))


def main():
    D = FECoDesign(ne=NE, r=R, sob_len=0.8)
    # a stable, decoupled compensator block: only the static gain K acts
    d = np.array([-1.0 - 0.3j * j for j in range(R)], complex)
    arow = (d, np.zeros(R - 1, complex), np.zeros(R - 1, complex))
    nvec = np.zeros(R, complex)
    b, c, *_ = D.bc(-1.03, 0.98)

    print("1. bc() hard failure on degenerate fields")
    for bad in (np.zeros(D.n, complex),
                np.full(D.n, np.nan + 0j)):
        try:
            D.bc(-1.03, 0.98, va=bad)
            check("raises on zero/NaN actuator field", False)
        except FloatingPointError:
            check("raises on zero/NaN actuator field", True)

    print("2. find a DEEPLY destabilising static gain (dense scan, test-only)")
    # The real escapes sat at Re = +0.33 to +10.1, SHIELDED from every line
    # shift by nearer stable eigenvalues -- that is what made them invisible.
    # Reproduce that geometry: keep increasing the gain until the escaped
    # eigenvalue is further from the line than the shield of stable modes, so
    # a shift-invert query from the line can never converge to it.
    Kc = None
    for mag in np.linspace(0.5, 400.0, 120):
        K = mag * np.exp(1j * 0.7)
        ad, lam = D.alpha_dense(arow, nvec, K, b, c)
        if ad > 2.5:
            Kc = K
            break
    check("a deep escape exists on the ray (alpha_dense > 2.5)",
          Kc is not None,
          f"K={Kc:.3g}, alpha_dense={ad:+.4f} at {lam:.4g}" if Kc is not None
          else "")
    if Kc is None:
        report(); return

    # sanity: alpha_dense against an independent dense eigensolve
    from scipy.linalg import eig as deig
    import controller as ctl
    n, r = D.n, D.r
    Acl = np.zeros((n + r, n + r), complex)
    Mcl = np.zeros((n + r, n + r), complex)
    Acl[:n, :n] = np.asarray(D.A.todense()) + Kc * np.outer(b, np.conj(c))
    Acl[:n, n:] = np.asarray(b).reshape(-1, 1) @ nvec.reshape(1, r)
    Acl[n:, :n] = np.ones((r, 1)) @ np.conj(c).reshape(1, n)
    Acl[n:, n:] = ctl.A_of(*arow)
    Mcl[:n, :n] = np.asarray(D.M.todense()); Mcl[n:, n:] = np.eye(r)
    lam_all = deig(Acl, Mcl, right=False)
    lam_all = lam_all[np.isfinite(lam_all)]
    ad_ref = float(lam_all.real.max())
    check("alpha_dense matches an independent dense eigensolve",
          abs(ad - ad_ref) < 1e-8 * max(1.0, abs(ad_ref)),
          f"{ad:+.6f} vs {ad_ref:+.6f}")

    print("3. the line certificate is blind to the escaped design "
          "(the documented conditional)")
    seen_bad = []
    al_line = D.alpha_certified(arow, nvec, Kc, b, c, eps=EPS,
                                eigs_seen=seen_bad)[0]
    check("standalone line sweep reads the UNSTABLE design as stable",
          al_line + EPS <= 0.0,
          f"line alpha={al_line:+.4f} vs dense {ad:+.4f} -- this is why the "
          f"gate exists")

    print("4. continuation tracking catches the same escape")
    seen0 = []
    al0 = D.alpha_certified(arow, nvec, 0j, b, c, eps=EPS, eigs_seen=seen0)[0]
    check("anchor (K=0, open loop) is stable and harvests a watch list",
          al0 + EPS <= 0.0 and len(seen0) > 0,
          f"alpha={al0:+.4f}, {len(seen0)} eigenvalue(s) seen")
    from run_case_epi import _topk_watch
    watch = _topk_watch(seen0, 8, EPS)
    Bc = np.full(R, D.mc, complex)
    ok, split, wnew, worst = csp.continuation_step(
        D.A, D.M, b, c, ctl.A_of(*arow), Bc, nvec, Kc, watch, eps=EPS)
    check("one-shot step K: 0 -> K* is rejected or flagged for splitting",
          (not ok) or split,
          f"ok={ok}, need_split={split}, worst={worst:.4g}"
          if worst is not None else f"ok={ok}, need_split={split}")

    print("5. adaptive bisection localises the crossing")
    # emulate the gate's homotopy on the 1-parameter path K(t) = t*K*:
    # track from the last point PROVEN stable; bisect on large motion; a
    # midpoint gets its own line sweep (crossings happen ON the line).
    def line_alpha(K, seen=None):
        return D.alpha_certified(arow, nvec, K, b, c, eps=EPS,
                                 eigs_seen=seen)[0]

    def seg(t0, t1, w0, depth):
        oks, split, w1, worst = csp.continuation_step(
            D.A, D.M, b, c, ctl.A_of(*arow), Bc, nvec, t1 * Kc,
            [w for w in w0 if w is not None], eps=EPS)
        if not oks:
            return 'crossed', t1
        if not split:
            return 'safe', w1
        if depth <= 0:
            return 'reject', t1
        tm = 0.5 * (t0 + t1)
        seen_m = []
        al_m = line_alpha(tm * Kc, seen_m)
        if al_m + EPS > 0.0:
            return 'crossed', tm
        r1 = seg(t0, tm, w0, depth - 1)
        if r1[0] != 'safe':
            return r1
        wm = sorted(set(list(r1[1]) + seen_m), key=lambda z: -z.real)[:8]
        return seg(tm, t1, wm, depth - 1)

    verdict = seg(0.0, 1.0, watch, 6)
    check("homotopy verdict is 'crossed' or 'reject', never 'safe'",
          verdict[0] in ('crossed', 'reject'), f"verdict={verdict[0]}")

    print("6. a genuinely safe interval terminates 'safe' (no deadlock)")
    # NOTE the anchor's margin here is only ~0.01, so near-boundary tracking
    # legitimately asks for splits even on small steps -- that is the gate
    # being conservative where it should be.  What must hold is that the
    # bisection then TERMINATES 'safe' on an interval that never crosses,
    # rather than rejecting everything.
    verdict_s = seg(0.0, 0.02, watch, 6)
    check("homotopy on the safe interval [0, 0.02*K] returns 'safe'",
          verdict_s[0] == 'safe', f"verdict={verdict_s[0]}")

    report()


def report():
    print(f"\n{npass} passed, {nfail} failed")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
