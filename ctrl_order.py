"""ctrl_order.py -- one place that answers "what is the controller order r?".

Eleven tools used to answer it like this:

    try:
        a.r = int(np.load(f'seeds_ne{a.ne}.npz', allow_pickle=True)['r'])
    except Exception:
        a.r = 7

The intent was report_v7 section 7's P4 fix, "derive from where it is decided".
The defect is the fallback.  seeds_ne200.npz is not part of the frozen archive,
so on any machine that did not also hold the seed catalogue the except branch
fired and every one of those tools silently adopted r = 7 while all 27 frozen
designs carry r = 8 -- the exact hardcoded constant P4 was written about.
verify_dense.py happens to raise a shape error; the tools that index rather
than reshape would instead have produced quietly wrong numbers.

The order is recoverable from three independent places, tried in this order:

  1. the design records themselves -- len(nvec) per npz.  This is the most
     direct source: it is the order of the controller that was actually
     optimised and certified, it is per design, and it cannot drift from the
     record it describes.  Disagreement between records is an error, not
     something to average over.
  2. the seed catalogue seeds_ne{ne}.npz, resolved against the repository
     root rather than the current directory.
  3. nothing -- raise.  A missing order is a stop, never a default.
"""
from __future__ import annotations

import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def r_from_records(results: str, ne: int) -> int | None:
    """Controller order from the archived designs; None if none are readable."""
    if not results:
        return None
    pat = os.path.join(results, f"*_ne{ne}.npz")
    orders: dict[int, list] = {}
    for f in sorted(glob.glob(pat)):
        try:
            d = np.load(f)
            if "nvec" not in d.files:
                continue
            orders.setdefault(int(len(d["nvec"])), []).append(
                os.path.basename(f))
        except Exception:
            continue
    if not orders:
        return None
    if len(orders) > 1:
        detail = "; ".join(
            f"r={k}: {len(v)} record(s) e.g. {v[0]}" for k, v in
            sorted(orders.items()))
        raise SystemExit(
            f"ctrl_order: the designs in {results} disagree on the controller "
            f"order ({detail}).  Certifying a mixed set under one r would be "
            f"meaningless; pass --r explicitly for a homogeneous subset.")
    return next(iter(orders))


def r_from_seeds(ne: int) -> int | None:
    """Controller order from the seed catalogue, looked up at the repo root."""
    for cand in (os.path.join(ROOT, f"seeds_ne{ne}.npz"), f"seeds_ne{ne}.npz"):
        try:
            return int(np.load(cand, allow_pickle=True)["r"])
        except Exception:
            continue
    return None


def infer_r(ne: int, results: str | None = None, quiet: bool = False) -> int:
    """r for mesh `ne`, preferring the designs in `results` over the seeds.

    Raises SystemExit -- never guesses -- when neither source is available.
    """
    r_rec = r_from_records(results, ne) if results else None
    r_sed = r_from_seeds(ne)
    if r_rec is not None and r_sed is not None and r_rec != r_sed:
        raise SystemExit(
            f"ctrl_order: the designs in {results} have r={r_rec} but "
            f"seeds_ne{ne}.npz declares r={r_sed}.  One of them is not the "
            f"catalogue these designs came from; pass --r explicitly.")
    r = r_rec if r_rec is not None else r_sed
    if r is None:
        raise SystemExit(
            f"ctrl_order: cannot determine the controller order for ne={ne}.  "
            f"No readable design records"
            + (f" in {results}" if results else "")
            + f" and no seeds_ne{ne}.npz at {ROOT}.  Pass --r explicitly.  "
            f"(This used to default to r=7, which is wrong for every design "
            f"in the frozen suite -- see report_v7 section 7, P4.)")
    if not quiet:
        src = "design records" if r_rec is not None else "seed catalogue"
        print(f"controller order r={r} (from the {src})", flush=True)
    return int(r)
