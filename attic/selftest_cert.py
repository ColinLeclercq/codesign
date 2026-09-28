"""
selftest_cert.py — validate the CLEANED certification stack, standalone.

Runs in about a minute on a login node and needs no result files.  Every check
targets a failure that actually occurred in this project, so a pass here means
the specific bugs that cost us weeks cannot recur silently.

    python selftest_cert.py
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import numpy as np
from scipy.linalg import eig
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_codesign, controller as ctl, certify_sparse as csp

NFAIL = 0


def check(name, ok, detail=''):
    global NFAIL
    print(f"[{'PASS' if ok else 'FAIL'}] {name:48s} {detail}")
    if not ok:
        NFAIL += 1


def dense_alpha(D, arow, nvec, Kfb, b, c):
    n = D.n; r = len(np.atleast_1d(arow[0]))
    Ad = np.asarray(D.A.todense()); Md = np.asarray(D.M.todense())
    Ac = ctl.A_of(*arow); Bc = np.full(r, 1.0, complex)
    Acl = np.zeros((n + r, n + r), complex); Mcl = np.zeros((n + r, n + r), complex)
    Acl[:n, :n] = Ad + Kfb * np.outer(b, np.conj(c))
    Acl[:n, n:] = b.reshape(-1, 1) @ np.asarray(nvec).reshape(1, r)
    Acl[n:, :n] = Bc.reshape(-1, 1) @ np.conj(c).reshape(1, n)
    Acl[n:, n:] = Ac
    Mcl[:n, :n] = Md; Mcl[n:, n:] = np.eye(r)
    lam = eig(Acl, Mcl, right=False); lam = lam[np.isfinite(lam)]
    return float(lam.real.max())


def main():
    D = fe_codesign.FECoDesign(ne=80, r=7, sob_len=0.8)
    r = 7
    b, c, *_ = D.bc(-1.03, 0.98)
    Bc = np.full(r, 1.0, complex)
    stable = (np.linspace(-1., -4., r).astype(complex),
              0.15 * np.ones(r - 1, complex), 0.10 * np.ones(r - 1, complex))
    # WEAK coupling, so the closed loop really is stable: the plant is stable
    # and the controller is stable, and a small nvec keeps the rank-2 coupling
    # from moving anything across.  With a strong coupling the design is
    # UNSTABLE and check 5 passes vacuously -- which it did on the first
    # attempt (dense alpha = +0.674 and the gate said False, "correctly").
    nvec = 1e-4 * np.linspace(1., 0.2, r).astype(complex)

    # --- 1. the abscissa must match a dense eigensolve -------------------
    al = D.alpha_certified(stable, nvec, 0j, b, c, eps=1e-3)[0]
    ad = dense_alpha(D, stable, nvec, 0j, b, c)
    check("abscissa vs dense eigensolve", abs(al - ad) < 1e-8,
          f"cert={al:+.8f} dense={ad:+.8f} err={abs(al-ad):.1e}")

    # --- 2. determinism: pinned ARPACK start vectors ---------------------
    vals = [D.alpha_certified(stable, nvec, 0j, b, c, eps=1e-3)[0]
            for _ in range(4)]
    check("abscissa is deterministic", max(vals) - min(vals) == 0.0,
          f"spread={max(vals)-min(vals):.1e}")

    # --- 3. a RUNAWAY controller pole must be caught ---------------------
    #     this is the +11681 failure: shifts near the axis cannot see it, so
    #     the exact r x r pole check must
    d2 = np.array(stable[0], complex); d2[0] = 11681.4
    ok, why, ns = csp.stability_gate(D.A, D.M, b, c,
                                     ctl.A_of(d2, stable[1], stable[2]),
                                     Bc, nvec, 0j, eps=1e-3)
    check("runaway controller pole rejected", (not ok) and 'controller' in why,
          f"{why[:44]}")

    # --- 4. an eigenvalue just right of the line must be caught ----------
    d3 = np.array(stable[0], complex); d3[0] = 0.5
    ok3, why3, _ = csp.stability_gate(D.A, D.M, b, c,
                                      ctl.A_of(d3, stable[1], stable[2]),
                                      Bc, nvec, 0j, eps=1e-3)
    check("marginal controller pole rejected", not ok3, f"{why3[:44]}")

    # --- 5. a genuinely stable design must be ACCEPTED -------------------
    #     the rectangle failed this near the boundary and pinned a case at
    #     5.88 while refusing points at 5.54.  Assert the design IS stable
    #     first, so the check cannot pass vacuously.
    ad5 = dense_alpha(D, stable, nvec, 0j, b, c)
    ok5, why5, ns5 = csp.stability_gate(D.A, D.M, b, c, ctl.A_of(*stable),
                                        Bc, nvec, 0j, eps=1e-3)
    check("test design is genuinely stable (guard)", ad5 + 1e-3 <= 0,
          f"dense alpha={ad5:+.6f}")
    check("stable design accepted", ok5 and (ad5 + 1e-3 <= 0),
          f"gate={ok5} shifts={ns5} dense={ad5:+.4f}")

    # --- 6. the sweep must TERMINATE near the boundary -------------------
    #     eps chosen so the margin is nearly active: the 2-D rectangle went
    #     INCONCLUSIVE here, the 1-D line must not
    eps_tight = max(-ad - 0.02, 1e-4)
    clear, a6, ns6, unc, wit = csp.stability_and_abscissa(
        D.A, D.M, b, c, ctl.A_of(*stable), Bc, nvec, 0j, eps=eps_tight)
    check("terminates near the stability boundary", unc == 0.0 and ns6 < 400,
          f"eps={eps_tight:.4f} shifts={ns6} uncovered={unc:.1e}")

    # --- 7. the dead rectangle path must be GONE -------------------------
    check("rectangle certificate removed",
          not hasattr(csp, 'certify_stability_rect'),
          "certify_stability_rect absent")

    print()
    print("ALL CERTIFICATION CHECKS PASSED" if NFAIL == 0
          else f"{NFAIL} CHECK(S) FAILED - do not run production")
    return 1 if NFAIL else 0


if __name__ == "__main__":
    sys.exit(main())
