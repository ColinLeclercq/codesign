"""make_init.py — build init_ne{ne}.npz WITHOUT touching seeds_ne{ne}.npz.

Why this exists
---------------
`prepare.py --ne 200` writes two files: the seed catalogue seeds_ne200.npz and
the cold-start controller init_ne200.npz.  The frozen archive shipped NEITHER,
and the seed catalogue was supplied separately by hand.  Running prepare.py to
recover the missing init would REBUILD the catalogue on top of the supplied one.

That is not a safe thing to do.  Every one of the 27 certified gammas was warm
started from a seed in that catalogue; `case.slurm` selects seeds by name from
it; and `ctrl_order.py` takes the controller order r from it.  A rebuild that
differed in any way -- a different BLAS reducing the Hankel singular values
slightly differently, a shoulder landing on r=7 instead of r=8 -- would silently
make the frozen suite incomparable, and nothing downstream would notice.

So this script builds ONLY init, and takes r from the EXISTING catalogue rather
than re-deriving it.  It refuses to run if that would overwrite a catalogue.

The construction is copied from prepare.py (lqg_bt_canonical -> from_state_space
-> savez, same field names, same dtypes) and is checked against it below.
prepare.py remains the source of truth for a mesh built from scratch; use that
one when there is no catalogue to protect.

    python tools/make_init.py --ne 200

Defaults xa = -1.03, xs = 0.98 are prepare.py's, and are the positions the
certified floors in bounds.py were computed at.  run_case_epi reads xa0/xs0 out
of this file, so writing it at other positions moves the cold start of every
case that does not override them.
"""
from __future__ import annotations
# --- repository root on sys.path
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
# --- end root shim
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fe_pipeline import FEPlant
from ctrl_order import infer_r

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--xa', type=float, default=-1.03)
    ap.add_argument('--xs', type=float, default=0.98)
    ap.add_argument('--r', type=int, default=-1,
                    help='force the controller order; -1 (default) takes it '
                         'from the existing seed catalogue')
    ap.add_argument('--out', default='init')
    ap.add_argument('--force', action='store_true',
                    help='overwrite an existing init file')
    a = ap.parse_args()
    t0 = time.time()

    # cwd, not ROOT: this is where prepare.py writes it and where
    # run_case_epi looks for it.  Every job script cd's to the root first.
    fn = f"{a.out}_ne{a.ne}.npz"
    if os.path.exists(fn) and not a.force:
        print(f"{fn} exists already; not overwriting (use --force).")
        return 0

    # r from the catalogue, never re-derived.  ctrl_order raises with a usable
    # message if there is no catalogue and no design records to read it from.
    if a.r > 0:
        r = a.r
        src = "--r on the command line"
    else:
        r = infer_r(ne=a.ne)
        src = f"seeds_ne{a.ne}.npz / design records"

    P = FEPlant(ne=a.ne, xa=a.xa, xs=a.xs)
    d = P.lqg_bt_canonical(r=r)
    import controller as ctl
    td, te, tf, tnv, tK = ctl.from_state_space(d['AKr'], d['BKr'], d['CKr'])

    got = len(np.atleast_1d(td))
    if got != r:
        raise SystemExit(
            f"canonical form came back at order {got} but the catalogue says "
            f"{r}.  These must agree or run_case_epi will fall back to a seed "
            f"controller mid-run.  Do not paper over this.")

    np.savez(fn, d=td, e=te, f=tf, nvec=tnv, hankel=d['S'][:12],
             ne=a.ne, xa=a.xa, xs=a.xs, r=r, n=P.n)
    print(json.dumps(dict(file=fn, ne=a.ne, n=int(P.n),
                          r=int(r), r_from=src, xa=a.xa, xs=a.xs,
                          hankel=[float(v) for v in d['S'][:8]],
                          seconds=round(time.time() - t0, 1))))
    print(f"seeds_ne{a.ne}.npz was NOT touched.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
