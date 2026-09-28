#!/usr/bin/env python3
"""plot_omega.py -- read what the --dump patch writes and draw it.

Optimisation note item 10: Omega, sigma_1 on Omega, t, the multipliers and the
spectrum, per outer round.  Four panels per figure:

  (a) sigma_1 over Omega for every outer round, with t and the certified gamma
      as horizontal lines and the active frequencies marked.  This is the panel
      that shows an exchange equalising -- or failing to.
  (b) the exchange set itself: which frequencies were present at which round,
      so retirement (stage 3, |Omega| <= 50) has something to be measured
      against.
  (c) the balance residual and gamma per round, log scale.  Stage 1's gate is
      the left axis of this panel going from ~1e4 to O(10).
  (d) the closed-loop spectrum near the stability line, coloured by round,
      with -eps drawn.  Report v7 section 5.4's "pinned on the boundary" is a
      statement about this panel.

    python plot_omega.py --dump harness_out/dump/K_B_C
    python plot_omega.py --dump ... --outdir figs --tag K+B+C

Backend is Agg unless --show, so it runs headless on a compute node.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(dumpdir):
    recs = []
    for jf in sorted(glob.glob(os.path.join(dumpdir, "outer_*.json"))):
        npz = jf[:-5] + ".npz"
        if not os.path.exists(npz):
            continue
        try:
            recs.append((json.load(open(jf)), np.load(npz)))
        except Exception as e:
            print(f"  skipping {os.path.basename(jf)}: {e}")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="dump directory")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--tag", default="", help="case label for the title")
    ap.add_argument("--xlim", default="",
                    help="'lo,hi' for the frequency panels")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    recs = load(a.dump)
    if not recs:
        print(f"no outer_*.npz/json pairs in {a.dump}")
        return 1
    tag = a.tag or os.path.basename(os.path.normpath(a.dump))
    print(f"{tag}: {len(recs)} outer round(s)")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    cmap = plt.get_cmap("viridis")
    n = len(recs)

    def col(i):
        return cmap(i / max(n - 1, 1))

    # ---- (a) sigma_1 on Omega ----
    A = ax[0, 0]
    for i, (r, d) in enumerate(recs):
        w, s = d["omega"], d["sigma"]
        o = np.argsort(w)
        A.plot(w[o], s[o], "-", lw=1.2, color=col(i),
               label=f"outer {r['outer']}")
        act = d["active"]
        if act.size:
            A.plot(w[act], s[act], "o", ms=5, color=col(i), mfc="none")
        if r.get("t") is not None and np.isfinite(r["t"]):
            A.axhline(r["t"], color=col(i), ls=":", lw=0.8)
    g = recs[-1][0].get("gamma_best")
    if g:
        A.axhline(g, color="k", ls="--", lw=1.2,
                  label=f"certified $\\gamma$ = {g:.4f}")
    A.set_xlabel(r"$\omega$")
    A.set_ylabel(r"$\sigma_1(\omega)$")
    A.set_title(f"{tag}: $\\sigma_1$ on $\\Omega$ (dotted = $t$, "
                f"circles = active)")
    A.legend(fontsize=7, ncol=2)
    A.grid(alpha=.3)
    if a.xlim:
        A.set_xlim(*[float(v) for v in a.xlim.split(",")])

    # ---- (b) the exchange set ----
    B = ax[0, 1]
    for i, (r, d) in enumerate(recs):
        w = d["omega"]
        B.plot(w, np.full(w.size, r["outer"]), "|", ms=6, color=col(i))
        act = d["active"]
        if act.size:
            B.plot(w[act], np.full(act.size, r["outer"]), "o", ms=5,
                   color="crimson")
    B.set_xlabel(r"$\omega$")
    B.set_ylabel("outer round")
    B.set_title(r"exchange set $\Omega$ (red = active)")
    B.grid(alpha=.3)
    if a.xlim:
        B.set_xlim(*[float(v) for v in a.xlim.split(",")])

    # ---- (c) balance residual and gamma ----
    C = ax[1, 0]
    it = [r["outer"] for r, _ in recs]
    bal = [r.get("balance_residual") for r, _ in recs]
    gb = [r.get("gamma_best") for r, _ in recs]
    ok = [i for i, v in enumerate(bal) if v is not None and np.isfinite(v)
          and v > 0]
    if ok:
        C.semilogy([it[i] for i in ok], [bal[i] for i in ok], "o-",
                   color="crimson", label="balance residual")
    C.set_xlabel("outer round")
    C.set_ylabel("balance residual (rel. gradient)", color="crimson")
    C.grid(alpha=.3)
    C2 = C.twinx()
    C2.plot(it, gb, "s--", color="navy", label=r"$\gamma$ best")
    C2.set_ylabel(r"certified $\gamma$", color="navy")
    C.set_title("stage 1's gate lives on the left axis "
                r"($\sim 10^4 \rightarrow O(10)$)")

    # ---- (d) spectrum near the line ----
    D = ax[1, 1]
    eps = recs[-1][0].get("eps", 0.05)
    for i, (r, d) in enumerate(recs):
        e = d["eigs"]
        if e.size:
            D.plot(e.real, e.imag, ".", ms=4, color=col(i))
    D.axvline(-eps, color="k", ls="--", lw=1.2, label=rf"$-\epsilon$ = {-eps}")
    D.axvline(0.0, color="r", ls="-", lw=0.8)
    D.set_xlabel(r"Re $\lambda$")
    D.set_ylabel(r"Im $\lambda$")
    D.set_title("closed-loop spectrum (colour = round)")
    D.legend(fontsize=8)
    D.grid(alpha=.3)
    # focus on the part that matters: a decade around the line
    try:
        allre = np.concatenate([d["eigs"].real for _, d in recs
                                if d["eigs"].size])
        if allre.size:
            D.set_xlim(min(-10 * eps, allre.max() - 1.0), max(0.05,
                                                              allre.max() + 0.05))
    except Exception:
        pass

    fig.tight_layout()
    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.join(a.outdir, f"omega_{tag.replace('+', '_')}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140)
    print(f"wrote {stem}.png / .pdf")
    if a.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
