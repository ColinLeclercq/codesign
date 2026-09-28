"""
seeds.py — build a DIVERSE set of order-r initial controllers for the H-infinity
co-design, all converted to complex controllable canonical form.

The optimisation objective is unchanged (full-field energy resolvent).  These
controllers only provide different STARTING BASINS.  Each is synthesised with
LOW-RANK exogenous channels

    disturbance   w = phi * a(t) ,   B1 = phi = Lc^{-1} (M g_upstream)  (whitened)
    performance   z = psi^T x    ,   C1 = psi^T = (Lc^{-1} (M g_down))^H

so both DGKF H-infinity synthesis and LQG cost only a rank-1-driven Riccati,
cheap enough to survive to the cavity.  (The control channels b, c are the
actuator/sensor and are unchanged.)

SEED CATALOGUE -- 16 controllers
================================

All are synthesised on the SAME plant and then balanced-truncated to a common
order, so they differ only in the SYNTHESIS PROBLEM each solves.  They are
starting basins, not candidate answers; the objective they are all optimised
against afterwards is the full-field energy resolvent.

1.  `lqgbt`                                                        1 controller
    ------------------------------------------------------------------------
    Classical LQG with FULL-FIELD process and sensor noise (B1 = I, every state
    excited equally), control weight beta^2 and measurement weight vhalf^2.

    It is the odd one out: it differs from the rest in the NOISE MODEL, not in
    the control authority.  It is NOT a high-gain limit -- for that see
    rho = 1e-4 below.

2.  `hinf_(dx)`  for dx = 1, 2, 3                                 3 controllers
    ------------------------------------------------------------------------
    DGKF central H-infinity controller on the rank-one channels

        disturbance   phi = Gaussian centred at -dx   (UPSTREAM)
        performance   psi = Gaussian centred at +dx   (DOWNSTREAM)

    from the coupled Riccati pair (X_inf, Y_inf) with the spectral-radius test
    rho(XY) < gamma^2, gamma bisected per dx to just above the achievable
    optimum.

    ONLY dx varies.  Nothing in the catalogue sweeps gamma at fixed dx, so the
    diversity here is entirely in WHERE the disturbance enters and where
    performance is measured -- and, because the mean flow advects downstream,
    dx also sets how far apart the observable and controllable regions sit.

3.  `lqg_(dx)_r(rho)`  for dx = 1, 2, 3 and rho = 1e-4, 1e-2, 1, 1e2
                                                                 12 controllers
    ------------------------------------------------------------------------
    LQG on the SAME rank-one channels as the H-infinity seeds, so that the two
    synthesis methods can be compared at matched channel placement, with the
    control penalty rho swept over four decades:

        rho = 1e-4   HIGH-GAIN limit.  Cheap control, aggressive feedback; this
                     is the region where the H-infinity optimum usually lives
                     and therefore the most useful starting basin.  Caveat: as
                     rho falls the closed-loop poles run outward, and since
                     omega_max = 3/2 max|Im lambda| over the WHOLE catalogue,
                     an aggressive seed widens the certified band for EVERY
                     case.  Balanced truncation limits this; watch the band
                     printed at startup.
        rho = 1e-2   moderately aggressive
        rho = 1      balanced
        rho = 1e2    cheap-control / low-authority end

    The grid is a full PRODUCT of dx and rho.  It was previously a zip(), which
    PAIRS the lists -- giving only (1, 1e-2), (2, 1), (3, 1e2), three seeds in
    which neither channel placement nor control authority was ever varied on
    its own.

COMMON ORDER
============
Each controller is balanced-truncated and its EFFECTIVE order taken as the
number of Hankel singular values above HANKEL_REL * sigma_1 -- the shoulder in
their decay.  The catalogue order is the MAXIMUM of these, so the parametrisation
is rich enough to represent every seed; all are then re-truncated to that common
r and canonicalised to the complex tridiagonal form.  This guarantees a single
parameter-vector shape for the optimiser, which is what lets any seed warm start
any case.

At ne = 60 the effective orders were: lqgbt 7; lqg_1_* 4-5; lqg_2_* 6;
lqg_3_* 5-8; hinf_* 4 -- common order 8, set by lqg_3_r1e-4, the most
aggressive seed at the widest channel separation.
"""
from __future__ import annotations
# --- repository root on sys.path (moved into tools/ by cleanup_archive.py)
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
# --- end root shim
import numpy as np
from scipy.linalg import (cholesky, solve_triangular, solve_continuous_are,
                          solve_lyapunov, svd, eigh)
from fe_gl import assemble_gl, load_vector, SIGMA  # SIGMA = hardware width
from fe_pipeline import FEPlant, BETA, VHALF
import controller as ctl

# (phi,psi) upstream/downstream translations and LQG control penalties
DX_LIST = (1.0, 2.0, 3.0)
# Control-cost weights.  rho -> 0 is the HIGH-GAIN limit, which is where the
# H-infinity optimum usually lives and therefore a natural place to start;
# rho -> infinity is the cheap-control, low-authority end.  1e-4 is included
# for that reason.  Note the trade: as rho falls the LQG poles run to infinity,
# and since omega_max = 3/2 max|Im| over the catalogue, an aggressive seed
# widens the certified band for every case.  Balanced truncation limits this,
# but watch the band printed at startup.
# rho = 1e-4 was dropped: at n = 399 each LQG synthesis costs ~36 s, and every
# 1e-4 seed is rejected after truncation anyway (|Im| of 24.5, 25.0 and 104.1
# against the cap of 20).  Paying 1.8 minutes to build three controllers that
# are then discarded is not worth it.  The high-gain end is therefore
# represented by 1e-2, which does survive.
RHO_LIST = (1e-2, 1.0, 1e2)
HANKEL_REL = 1e-3
# Reject any seed whose FULL-ORDER poles exceed this in |Im|: the plant
# itself has |Im| <= 1, so a controller far above that is an artefact of
# the synthesis and would force the certified band open for every case.
SEED_WMAX = 20.0                     # max |Im lambda| of a usable seed
# gamma bisection cost: each step is one hinf_central call = TWO dense n x n
# Riccati solves, ~36 s at n = 399.  0.5% needed ~11 steps per dx and put the
# H-infinity block alone at 20 minutes.  10% needs ~6, and a seed synthesised
# at 1.1 x gamma_opt is an equally good STARTING BASIN -- nothing downstream
# depends on gamma being optimal.
GAM_TOL = 1.10
GAM_ITERS = 8


# Width of the EXOGENOUS channels (phi, psi) used for synthesis only.  These
# are fictitious -- the real hardware is b, c, which keep fe_gl.SIGMA -- so the
# two need not agree, and they should not.
#
# A spatially NARROW channel is BROADBAND in frequency, so LQG buys authority
# across a wide band and places its poles accordingly.  Measured (dx = 2,
# rho = 1), max |Im lambda| of the resulting controller:
#
#     channel sigma   0.4 -> 43.0     (= fe_gl.SIGMA, the old value)
#                     1.0 -> 18.6
#                     2.0 -> 17.6
#                     4.0 -> 17.7
#
# so it falls sharply to sigma ~ 1 and then plateaus.  Over the WHOLE catalogue
# (the worst member being the widest dx at the smallest rho) the maxima are
#
#     sigma = 0.4 -> 56.3      sigma = 1.0 -> 34.5
#     sigma = 1.5 -> 24.7      sigma = 2.0 -> 24.3
#
# so 1.5 brings every seed under 25 and the band to +-37.5, against +-121 with
# the original width -- with no truncation tricks and the whole rho sweep still
# usable.
#
# This is also why `lqgbt` (full-field noise, B1 = I) was always inside the band
# while the rank-one seeds were not: it is the CHANNEL WIDTH that matters, not
# rho.  Six decades of rho move |Im| by only about 20%; the channel width moves
# it by a factor of two.
SIGMA_CHANNEL = 1.5


def _gauss(xn, x0):
    return np.exp(-(xn - x0) ** 2 / (2 * SIGMA_CHANNEL ** 2))


def _sqrt_psd(G):
    G = (G + G.conj().T) / 2
    d, V = eigh(G); d = np.clip(d, 0.0, None)
    return V @ np.diag(np.sqrt(d))


def _bt(AK, BK, CK, r):
    """Square-root balanced truncation to order r; returns (Ar,Br,Cr, hankel)."""
    Pc = solve_lyapunov(AK, -(BK @ BK.conj().T))
    Qc = solve_lyapunov(AK.conj().T, -(CK.conj().T @ CK))
    Lp, Lq = _sqrt_psd(Pc), _sqrt_psd(Qc)
    U, S, Vh = svd(Lp.conj().T @ Lq); V = Vh.conj().T
    si = np.diag(1.0 / np.sqrt(S[:r]))
    W1 = Lp @ U[:, :r] @ si
    W2 = Lq @ V[:, :r] @ si
    return W2.conj().T @ AK @ W1, W2.conj().T @ BK, CK @ W1, S


def _canonical(AK, BK, CK):
    r = AK.shape[0]
    den = np.poly(AK)
    sk = 0.7 + 1j * np.linspace(0.5, 2.5, r)
    Kv = np.array([(CK @ np.linalg.solve(s * np.eye(r) - AK, BK))[0, 0]
                   for s in sk])
    ncoef = np.linalg.solve(np.vander(sk, r), Kv * np.polyval(den, sk))
    return den[1:][::-1].copy(), ncoef[::-1].copy()      # (arow, nvec)


class SeedBuilder:
    def __init__(self, ne=200, L=56.06, mu0=0.38, xa=-1.03, xs=0.98):
        self.P = FEPlant(ne=ne, L=L, mu0=mu0, xa=xa, xs=xs)
        self.At, self.B2, self.C2, self.Lc = self.P.whiten()
        self.n = self.P.n
        self.xn = self.P.xn

    # whitened low-rank exogenous channels for a given (phi,psi) translation
    def _channels(self, dx):
        g_up = self.P.M @ _gauss(self.xn, -abs(dx))      # disturbance load
        g_dn = self.P.M @ _gauss(self.xn, +abs(dx))      # performance functional
        B1 = solve_triangular(self.Lc, g_up, lower=True).reshape(-1, 1)
        C1 = solve_triangular(self.Lc, g_dn, lower=True).conj().reshape(1, -1)
        B1 = B1 / np.linalg.norm(B1)
        C1 = C1 / np.linalg.norm(C1)
        return B1, C1

    # ---- LQG at (phi,psi) with control penalty rho ----
    def lqg(self, dx, rho):
        At, B2, C2 = self.At, self.B2, self.C2
        B1, C1 = self._channels(dx)
        n = self.n
        Q = (C1.conj().T @ C1) + 1e-6 * np.eye(n)        # performance weight
        X = solve_continuous_are(At, B2, Q, np.array([[rho]]))
        W = (B1 @ B1.conj().T) + 1e-6 * np.eye(n)        # process-noise cov
        Y = solve_continuous_are(At.conj().T, C2.conj().T, W,
                                 np.array([[VHALF ** 2]]))
        F = (B2.conj().T @ X) / rho
        Lg = (Y @ C2.conj().T) / VHALF ** 2
        return At - B2 @ F - Lg @ C2, Lg, -F

    # ---- H-infinity central controller (DGKF) at (phi,psi), level gamma ----
    def hinf_central(self, dx, gam):
        At, B2, C2 = self.At, self.B2, self.C2
        B1, C1 = self._channels(dx)
        n = self.n
        g2 = gam ** 2
        # X_inf : A^T X + X A + C1^T C1 + X(B1 B1^T/g2 - B2 B2^T)X = 0
        X = solve_continuous_are(
            At, np.hstack([B1, B2]), C1.conj().T @ C1,
            np.diag([-g2, 1.0]).astype(complex))
        # Y_inf : dual
        Y = solve_continuous_are(
            At.conj().T, np.hstack([C1.conj().T, C2.conj().T]),
            B1 @ B1.conj().T, np.diag([-g2, 1.0]).astype(complex))
        rho = np.max(np.abs(np.linalg.eigvals(X @ Y)))
        if rho >= g2:
            raise ValueError(f"spectral radius {rho:.3g} >= gamma^2 {g2:.3g}")
        Z = np.linalg.solve(np.eye(n) - Y @ X / g2, np.eye(n))
        F = -(B2.conj().T @ X)
        Lc_ = -(Y @ C2.conj().T)
        AK = (At + B1 @ B1.conj().T @ X / g2 + B2 @ F
              + Z @ Lc_ @ C2)
        return AK, -Z @ Lc_, F

    # ---- build the whole catalogue, common order, canonical forms ----
    def _lqg_fullfield(self):
        """Full-field LQG (the classic LQG-BT source), full order n."""
        At, B2, C2 = self.At, self.B2, self.C2
        n = self.n
        X = solve_continuous_are(At, B2, BETA ** 2 * np.eye(n),
                                 np.array([[1.0]]))
        Y = solve_continuous_are(At.conj().T, C2.conj().T, np.eye(n),
                                 np.array([[VHALF ** 2]]))
        F = (B2.conj().T @ X)
        Lg = (Y @ C2.conj().T) / VHALF ** 2
        return At - B2 @ F - Lg @ C2, Lg, -F, None

    def build(self, r_target=None):
        raw = {}          # name -> (AK,BK,CK, hankel)
        # 1) existing LQG-BT: build the FULL-ORDER LQG controller (full-field
        #    process/sensor noise) so it is truncated to the common r like the
        #    others, not pre-reduced to 4.
        raw['lqgbt'] = self._lqg_fullfield()
        # 2) LQG over the FULL grid of (dx, rho).
        #
        # This used zip(), which PAIRS the lists instead of crossing them: it
        # produced (1,1e-2), (2,1.0), (3,1e2) only -- three seeds in which dx
        # and rho were never varied independently, so neither the channel
        # placement nor the control authority was explored on its own.  It also
        # silently discarded any extra rho.
        import itertools as _it
        for dx, rho in _it.product(DX_LIST, RHO_LIST):
            try:
                AK, BK, CK = self.lqg(dx, rho)
                raw[f'lqg_{int(dx)}_r{rho:g}'] = (AK, BK, CK, None)
            except Exception as e:
                print(f"  [seed lqg dx={dx} rho={rho}] skipped: {e}")
        # 3) H-infinity central controllers, GAMMA AS SMALL AS FEASIBLE.
        #
        # This used to walk a coarse list (200, 120, 80, 50, 30) and take the
        # first that solved, which is not the optimum -- it is whichever of five
        # arbitrary levels happened to be feasible.  Bisect instead: the DGKF
        # conditions fail below gamma_opt and hold above it, so the boundary is
        # found to 0.5% in about 40 solves.
        # Each hinf_central call solves TWO dense n x n Riccati equations, so
        # the bisection tolerance is what sets the cost: 0.5% needs ~12 solves
        # per dx at n = 399 and blew the 30-minute wall.  5% needs ~8, and the
        # seed is a STARTING BASIN -- a controller synthesised at 1.05 gamma_opt
        # rather than 1.005 is an equally good place to start, and nothing
        # downstream depends on gamma being optimal.  Bracket from a modest
        # upper bound too, rather than 1e6, which wastes the first few solves
        # confirming that a huge gamma is feasible.
        for dx in DX_LIST:
            lo, hi, best = 1.0, 1e3, None
            for _ in range(GAM_ITERS):
                gam = np.sqrt(lo * hi)
                try:
                    best = self.hinf_central(dx, gam)
                    hi = gam
                except Exception:
                    lo = gam
                if hi / lo < GAM_TOL:
                    break
            if best is not None:
                raw[f'hinf_{int(dx)}'] = (best[0], best[1], best[2], None)
            else:
                print(f"  [seed hinf dx={dx}] no feasible gamma")

        # fallback: LQG at that (phi,psi) so we still get a distinct seed
                try:
                    AK, BK, CK = self.lqg(dx, 1.0)
                    raw[f'hinf_{int(dx)}_fallbackLQG'] = (AK, BK, CK, None)
                except Exception as e:
                    print(f"  [seed hinf dx={dx}] skipped: {e}")

        # effective orders via a first BT pass at generous order
        eff = {}
        rmax_cap = min(12, self.n - 1)
        for name, (AK, BK, CK, _) in raw.items():
            _, _, _, S = _bt(AK, BK, CK, min(rmax_cap, AK.shape[0]))
            thr = HANKEL_REL * S[0]
            eff[name] = int(max(1, np.sum(S > thr)))
        r = r_target or min(rmax_cap, max(eff.values()))
        print(f"  effective orders: {eff}  -> common r = {r}")

        seeds, rejected = {}, []
        for name, (AK, BK, CK, _) in raw.items():
            Ar, Br, Cr, _ = _bt(AK, BK, CK, r)
            # TRIDIAGONAL parametrisation, obtained from the BALANCED reduced
            # realisation via its modal form (diag(poles) is tridiagonal with
            # e = f = 0).  The companion/canonical form is never built.
            try:
                d, e, f, nv, K0 = ctl.from_state_space(Ar, Br, Cr)
            except Exception as exc:
                print(f"  [seed {name}] tridiagonal conversion failed: {exc}")
                continue
            # REJECT AFTER TRUNCATION, not before.  Balanced truncation
            # preserves input-output behaviour but is free to place poles
            # anywhere: lqg_3_r0.0001 goes from |Im| ~ 35 at full order to 104
            # at r = 7.  So the pole frequency of the SEED AS USED is only known
            # once it has been reduced, and that is where the cut must be made.
            im = float(np.max(np.abs(np.linalg.eigvals(ctl.A_of(d, e, f)).imag)))
            if im > SEED_WMAX:
                rejected.append((name, im))
                continue
            seeds[name] = dict(d=d, e=e, f=f, nvec=nv, K=K0,
                               eff_order=eff[name])
        if rejected:
            print("  rejected after truncation (|Im| > %g): " % SEED_WMAX
                  + ", ".join(f"{n}:{v:.1f}" for n, v in rejected))
        print(f"  seeds kept: {len(seeds)} of {len(raw)}")
        if not seeds:
            raise RuntimeError("every seed was rejected; raise SEED_WMAX")
        return seeds, r

    def _from_canon(self, arow, nvec):
        """Reconstruct a state-space controller from canonical (arow,nvec)."""
        r = len(arow)
        Ac = np.zeros((r, r), complex)
        if r > 1:
            Ac[:-1, 1:] = np.eye(r - 1)
        Ac[-1, :] = -arow
        Bc = np.zeros((r, 1), complex); Bc[-1, 0] = 1.0
        Cc = np.asarray(nvec, complex).reshape(1, r)
        return Ac, Bc, Cc, None


if __name__ == "__main__":
    import sys, json
    ne = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    SB = SeedBuilder(ne=ne)
    seeds, r = SB.build()
    out = f"seeds_ne200.npz"
    np.savez(out, r=r, names=list(seeds.keys()),
             **{f"{k}__d": v['d'] for k, v in seeds.items()},
             **{f"{k}__e": v['e'] for k, v in seeds.items()},
             **{f"{k}__f": v['f'] for k, v in seeds.items()},
             **{f"{k}__nvec": v['nvec'] for k, v in seeds.items()})
    print(json.dumps(dict(file=out, r=int(r),
                          seeds={k: v['eff_order'] for k, v in seeds.items()})))
