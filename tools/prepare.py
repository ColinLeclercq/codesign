"""prepare.py — one-time per mesh: LQG -> balanced truncation -> canonical.
This is the ONLY dense step (design-time scaffolding, GL-specific).
Cache it so the job array does not repeat it.
    python prepare.py --ne 200
"""
from __future__ import annotations
# --- repository root on sys.path (moved into tools/ by cleanup_archive.py)
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
# --- end root shim
import argparse, json, time, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fe_pipeline import FEPlant

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--xa', type=float, default=-1.03)
    ap.add_argument('--xs', type=float, default=0.98)
    ap.add_argument('--r', type=int, default=-1,
                    help='force the controller order; -1 (default) takes it\n                         from the seed catalogue')
    ap.add_argument('--out', default='init')
    a = ap.parse_args()
    t0 = time.time()
    P = FEPlant(ne=a.ne, xa=a.xa, xs=a.xs)

    # The SEED CATALOGUE decides the controller order: it is the maximum
    # effective order over the catalogue, taken from the shoulder in the Hankel
    # singular values.  Build it FIRST, so that init_ne*.npz can be written at
    # the SAME order.
    #
    # These were previously independent: init was written at the hardcoded
    # --r 7 and the catalogue then derived r = 8 on its own.  Nothing linked
    # them, so a run that took its order from the catalogue and its controller
    # from init failed with "value array of shape (7,) could not be broadcast
    # to indexing result of shape (8,)" -- deep inside the resolvent, with
    # nothing to suggest the two files disagreed.
    from seeds import SeedBuilder
    SB = SeedBuilder(ne=a.ne, xa=a.xa, xs=a.xs)
    seeds, r = SB.build(r_target=(a.r if a.r > 0 else None))
    sfn = f"seeds_ne{a.ne}.npz"
    np.savez(sfn, r=r, names=list(seeds.keys()),
             **{f"{k}__d": v['d'] for k, v in seeds.items()},
             **{f"{k}__e": v['e'] for k, v in seeds.items()},
             **{f"{k}__f": v['f'] for k, v in seeds.items()},
             **{f"{k}__nvec": v['nvec'] for k, v in seeds.items()})

    d = P.lqg_bt_canonical(r=r)
    fn = f"{a.out}_ne{a.ne}.npz"
    import controller as ctl
    td, te, tf, tnv, tK = ctl.from_state_space(d['AKr'], d['BKr'], d['CKr'])
    assert len(np.atleast_1d(td)) == r, (
        f"init order {len(np.atleast_1d(td))} != catalogue order {r}")
    np.savez(fn, d=td, e=te, f=tf, nvec=tnv, hankel=d['S'][:12],
             ne=a.ne, xa=a.xa, xs=a.xs, r=r, n=P.n)
    print(json.dumps(dict(seed_file=sfn, common_order=int(r),
                          seeds={k: v['eff_order'] for k, v in seeds.items()})))
    print(json.dumps(dict(file=fn, ne=a.ne, n=P.n, r=int(r),
                          hankel=[float(v) for v in d['S'][:8]],
                          seconds=round(time.time() - t0, 1))))
