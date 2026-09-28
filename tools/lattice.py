#!/usr/bin/env python3
"""lattice.py -- run cases in nesting order with the suite's start protocol.

This is tools/case.slurm's start logic, portable.  case.slurm runs ONE case per
SLURM task and relies on the array to sequence the lattice; this runs a closed
set of cases in one process, in topological order, so the same protocol gives
an ne=100 development baseline on a one-core sandbox and an ne=200 confirmation
on a cluster node.

The protocol (case.slurm lines 51-160, unchanged):

  ROOT (no nesting parents)
      multistart over the seed catalogue, one --cold --seed S run per usable
      seed; usable = every controller pole within |Im| <= wmax.  A static-gain
      case (ctrl_mode gain/gainr) or an empty catalogue: one --cold run.
  CHILD
      one --resume run from EACH parent's record (the feasible set contains the
      parent's design and gbest is monotone, so the child cannot finish above
      the parent -- that is what makes the nesting check meaningful), then one
      from each field donor (no guarantee, good fields), then one polish from
      its own best.  run_case_epi's no-regress guard keeps the best across
      attempts.

Why a baseline built with --cold instead (jobs 16034591, 16047382) was wrong:
KGC+B's cold start is the open loop, infeasible at eps=0.05 and refused; the
frozen `_s0_` record was reached from its parent KGC.  And K+B+C cold reaches a
different point (9.82, stationary) from K+B+C via its parents (7.96, frozen).
The suite's numbers are only reproducible under the suite's protocol.

Departures, all opt-in and recorded in each run's record by run_case_epi:
  --inner-budget iters   reproducible (section 15 of STAGE0.md); the suite
                         used clock.  Recommended for A/B work.
  --seeds N              cap root multistart at the first N usable seeds.
  --first-parent-only    one attempt per child.  Cheap; breaks the
                         "bounded by EVERY parent" guarantee -- development only.

Resumable: a case whose record already exists in --out is skipped, so a killed
run continues where it stopped.  Each attempt's stdout goes to
<out>/logs/<tag>__<attempt>.log, and any FATAL line is echoed.

    python tools/lattice.py --ne 100 --out results_ne100 --eps 0.05 \\
        --closure K+B+C,KSP+B+C,KGC+B --inner-budget iters --inner-maxit 20 \\
        --outer 4 --wall 600 --seeds 1 --first-parent-only
"""
from __future__ import annotations
# --- repository root on sys.path
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
# --- end root shim
import argparse, json, os, subprocess, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def tag_of(case, ne):
    return f"{case.replace('+', '_')}_s0_ne{ne}"


def nesting(flag, case):
    r = subprocess.run([PY, os.path.join(ROOT, 'tools', 'nesting.py'),
                        flag, case], capture_output=True, text=True)
    return r.stdout.split()


def closure(targets):
    """Every case needed to run `targets`, in an order where each case's
    PARENTS come first.

    Parents only.  Field donors are not nesting relations and they form cycles
    (KSP's donors include K, whose parent is KSP); walking them with the parents
    produced an order with roots AFTER their children, so B was skipped as
    "no record" by KSP+B and everything downstream fell over.  Donors are used
    at run time if their record happens to exist, never for ordering.
    """
    seen, order = set(), []

    def visit(c):
        if c in seen:
            return
        seen.add(c)
        for p in nesting('--parents', c):
            visit(p)
        order.append(c)
    for t in targets:
        visit(t)
    return order


def usable_seeds(ne, wmax):
    """case.slurm's seed filter: keep seeds whose poles fit the band."""
    import controller as ctl
    f = os.path.join(ROOT, f"seeds_ne{ne}.npz")
    if not os.path.exists(f):
        return []
    sd = np.load(f, allow_pickle=True)
    keep = []
    for nm in sd['names']:
        nm = str(nm)
        A = ctl.A_of(sd[f'{nm}__d'], sd[f'{nm}__e'], sd[f'{nm}__f'])
        if float(np.max(np.abs(np.linalg.eigvals(A).imag))) <= wmax:
            keep.append(nm)
    return keep


def ctrl_mode(case):
    from fe_codesign import parse_case
    f = parse_case(case)
    return f.get('ctrl_mode') if f.get('K') else 'base'


def _state_path(a):
    return os.path.join(a.out, 'lattice_state.json')


def load_state(a):
    try:
        return json.load(open(_state_path(a)))
    except Exception:
        return {}


def mark_done(a, case, attempt):
    """Record a COMPLETED attempt.  Resumability keys on this, not on whether a
    record exists: a record appears after the first successful attempt, and a
    chunked run that was killed mid-multistart would otherwise skip the rest of
    the seeds next time (KSP lost four of five that way)."""
    st = load_state(a)
    st.setdefault(case, [])
    if attempt not in st[case]:
        st[case].append(attempt)
    json.dump(st, open(_state_path(a), 'w'), indent=1)


def run_one(a, case, attempt, extra_args, logdir):
    argv = [PY, os.path.join(ROOT, 'run_case_epi.py'), '--case', case,
            '--start', '0', '--ne', str(a.ne), '--wall', str(a.wall),
            '--outer', str(a.outer), '--eps', str(a.eps),
            '--outdir', a.out, '--inner-budget', a.inner_budget,
            '--inner-maxit', str(a.inner_maxit)] + extra_args + a.extra
    logf = os.path.join(logdir, f"{tag_of(case, a.ne)}__{attempt}.log")
    t0 = time.time()
    with open(logf, 'w') as fh:
        r = subprocess.run(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                           text=True)
    dt = time.time() - t0
    txt = open(logf).read()
    fatal = [ln for ln in txt.splitlines() if ln.startswith('FATAL')]
    g = None
    for ln in reversed(txt.splitlines()):
        if ln.startswith('{"case"'):
            try:
                g = json.loads(ln).get('gamma')
            except Exception:
                pass
            break
    kept = [ln for ln in txt.splitlines()
            if ln.startswith('keeping previous better result')]
    status = (f"REFUSED: {fatal[0][7:70]}" if fatal
              else f"gamma={g:.6f}" if g is not None
              else f"no improvement; {kept[0]}" if kept
              else f"rc={r.returncode}, no summary line")
    print(f"    {attempt:28s} {dt:7.0f}s  {status}", flush=True)
    if not fatal:
        # a REFUSED attempt is not a completed one: it must run again once
        # the reason is fixed (KGC's cold start was refused, then fixed with
        # --cont-anchor defer, then skipped as "done")
        mark_done(a, case, attempt)
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ne', type=int, default=100)
    ap.add_argument('--out', default='results_lattice')
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--closure', default='K+B+C,KSP+B+C,KGC+B',
                    help='target cases; everything they need is run first')
    ap.add_argument('--wall', type=int, default=600,
                    help='per attempt; under iters it is only the clock guard')
    ap.add_argument('--outer', type=int, default=4)
    ap.add_argument('--inner-budget', default='iters', choices=['clock', 'iters'])
    ap.add_argument('--inner-maxit', type=int, default=20)
    ap.add_argument('--seeds', type=int, default=0,
                    help='cap root multistart at the first N usable seeds; '
                         '0 = all, as the suite')
    ap.add_argument('--first-parent-only', action='store_true')
    ap.add_argument('--parent-dir', default=None,
                    help='fallback directory for parent/donor records when '
                         '--out has none yet (e.g. the frozen results/ for a '
                         'stage gate).  --out is always searched first.')
    ap.add_argument('--targets-only', action='store_true',
                    help='run only the --closure cases themselves; their '
                         'ancestors must already exist in --parent-dir')
    ap.add_argument('--wmax', type=float, default=20.0)
    ap.add_argument('--force', action='store_true', help='rerun done cases')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                    help='everything after --extra goes to run_case_epi '
                         '(e.g. --extra --ls-fallback)')
    a = ap.parse_args()

    targets = [c.strip() for c in a.closure.split(',') if c.strip()]
    order = closure(targets)
    if a.targets_only:
        order = [c for c in order if c in targets]
    pdir = a.parent_dir or a.out
    os.makedirs(a.out, exist_ok=True)
    logdir = os.path.join(a.out, 'logs')
    os.makedirs(logdir, exist_ok=True)

    print(f"lattice: ne={a.ne} eps={a.eps} out={a.out} budget={a.inner_budget}"
          f"/{a.inner_maxit} outer={a.outer} wall={a.wall}"
          + (" first-parent-only" if a.first_parent_only else "")
          + (f" seeds<={a.seeds}" if a.seeds else "")
          + (f" parents<-{a.parent_dir}" if a.parent_dir else "")
          + (" targets-only" if a.targets_only else "")
          + (" " + " ".join(a.extra) if a.extra else ""))
    print(f"closure of {targets}: {len(order)} case(s)")
    print("  " + "  ".join(order), flush=True)
    if a.dry_run:
        return 0

    seeds_all = usable_seeds(a.ne, a.wmax)
    summary = {}
    t_start = time.time()
    for case in order:
        tag = tag_of(case, a.ne)
        rec = os.path.join(a.out, tag + '.json')
        done_att = set() if a.force else set(load_state(a).get(case, []))
        parents = nesting('--parents', case)
        donors = nesting('--donors', case)
        print(f"== {case:10s} parents={parents or '-'} donors={donors or '-'}"
              + (f"  ({len(done_att)} attempt(s) already done)" if done_att else ""),
              flush=True)
        if not parents:
            mode = ctrl_mode(case)
            seeds = seeds_all
            if a.seeds:
                seeds = seeds[:a.seeds]
            if mode in ('gain', 'gainr') or not seeds:
                if 'cold' not in done_att:
                    run_one(a, case, 'cold', ['--cold'], logdir)
            else:
                for s in seeds:
                    lab = f'cold seed {s}'
                    if lab in done_att:
                        continue
                    run_one(a, case, lab, ['--cold', '--seed', s], logdir)
        else:
            srcs = [(f'from {p}', p) for p in parents]
            if not a.first_parent_only:
                srcs += [(f'from donor {d}', d) for d in donors]
            done = sum(1 for l, _ in srcs if l in done_att)
            for label, p in srcs:
                if label in done_att:
                    continue
                if a.first_parent_only and done:
                    break   # first parent WITH a record, not first in the list
                # this run's own record of the parent first (improvements
                # propagate down the lattice, as in the suite), then the
                # reference directory (stage-1 gate: K+B+C started from the
                # FROZEN KSP+B+C at 8.2465 while the same job had just made
                # one at 7.9515)
                src = os.path.join(a.out, tag_of(p, a.ne) + '.npz')
                if not os.path.exists(src):
                    src = os.path.join(pdir, tag_of(p, a.ne) + '.npz')
                if not os.path.exists(src):
                    print(f"    {label:28s}   -- no record, skipped", flush=True)
                    continue
                run_one(a, case, label, ['--resume', src], logdir)
                done += 1
            own = os.path.join(a.out, tag + '.npz')
            if (os.path.exists(own) and not a.first_parent_only
                    and 'polish' not in done_att):
                run_one(a, case, 'polish', ['--resume', own], logdir)
        g = json.load(open(rec)).get('gamma') if os.path.exists(rec) else None
        summary[case] = g
        print(f"   -> {case} {('gamma=%.6f' % g) if g is not None else 'NO RECORD'}"
              f"   [{(time.time() - t_start) / 60:.0f} min elapsed]", flush=True)

    print("\n=== lattice summary ===")
    for c in order:
        g = summary.get(c)
        print(f"  {c:10s} {('%.6f' % g) if g is not None else 'NO RECORD'}")
    json.dump(dict(args=vars(a), order=order, gamma=summary),
              open(os.path.join(a.out, 'lattice_summary.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
