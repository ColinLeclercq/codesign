#!/usr/bin/env python3
"""lattice_check.py -- do children end below their parents?  (STAGE6 6.31)

The nesting guarantee of the suite protocol: a child resumed from a parent's
record STARTS at that parent's design (the child's feasible set contains it)
and its best value never rises, so it ends at or below the parent.  A short
run checks the guarantee itself; convergence is not needed:

  roots      one cold run (KSP: the first usable seed)
  children   one run from EVERY parent's record, each in its own directory;
             the case's record is the best attempt
  no donors, no polish; --outer 3 by default.

Per attempt the log gives the START value (the parent's design certified by
the child) and the END value.  The report checks, for every child and every
parent p:

  inclusion   start(from p) == gamma(p)  (to the certificate's tolerance)
  descent     end(from p)   <= start(from p)
  nesting     gamma(child)  <= gamma(p)

Attempts are independent once their parent is final: --jobs of them run at a
time, each a separate process with --nproc 1.  Resumable (state file); a killed
attempt is rerun.

    python tools/lattice_check.py --ne 200 --out results_lattice_check_ne200 --jobs 2
    python tools/lattice_check.py --out results_lattice_check_ne200 --report
"""
from __future__ import annotations
import os, sys, json, time, shutil, argparse, subprocess, threading
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from lattice import nesting, closure, usable_seeds, tag_of, ROOT, PY

ALL27 = ['B', 'C', 'KGC', 'KSP', 'B+C', 'FB', 'FC', 'KGC+B', 'KGC+C', 'KSP+B', 'KSP+C', 'K',
         'FB+FC', 'KGC+B+C', 'KGC+FB', 'KGC+FC', 'KSP+B+C', 'KSP+FB', 'KSP+FC', 'K+B', 'K+C',
         'KGC+FB+FC', 'KSP+FB+FC', 'K+B+C', 'K+FB', 'K+FC', 'K+FB+FC']
# field cases: 4 Gabor atoms of width >= 0.4 (the actuator's own width, so a
# parent's Gaussian is an admissible field: fit error 3e-12 at ne = 200) and the
# H^1 row at 1.8 (the Gaussian's RMS wavenumber is 1.759); the Laplacian modes
# are only the reduced model's input basis (user, 2026-09-30: minimum width 0.4);
# since 6.33 the model's field input basis is the atoms' tangent space instead
# (--field-input tangent; the 6.31 check ran with the modes)
ATOMS = ("--field-basis laplace --field-lo -38.9 --field-hi 39.2 --field-n {nf} --field-atoms 4 "
         "--atom-smin 0.4 --atom-kmax 0.7071 --field-kmax 1.8 --field-order 1 --field-input tangent")


def loop_flags(ne, outer):
    """The nominal loop (tools/n100_kbc.sh) at this mesh and round count."""
    txt = open(os.path.join(ROOT, 'tools', 'n100_kbc.sh')).read()
    i = txt.index('LOOP="'); j = txt.index('"', i + 6)
    fl = txt[i + 6:j].replace('\\\n', ' ').split()
    out, skip = [], False
    for k, t in enumerate(fl):
        if skip:
            skip = False; continue
        if t in ('--ne', '--outer', '--wall'):
            skip = True; continue
        out.append(t)
    return out + ['--ne', str(ne), '--outer', str(outer), '--wall', '36000', '--nproc', '1']


class State:
    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        try:
            self.d = json.load(open(path))
        except Exception:
            self.d = {}

    def get(self, key):
        with self.lock:
            return self.d.get(key)

    def put(self, key, val):
        with self.lock:
            self.d[key] = val
            json.dump(self.d, open(self.path + '.tmp', 'w'), indent=1)
            os.replace(self.path + '.tmp', self.path)


def attempts_of(case, ne, seeds):
    parents = nesting('--parents', case)
    if not parents:
        from lattice import ctrl_mode
        if ctrl_mode(case) in ('gain', 'gainr') or not seeds:
            return [('cold', None, ['--cold'])]
        return [(f'cold seed {seeds[0]}', None, ['--cold', '--seed', seeds[0]])]
    return [(f'from {p}', p, None) for p in parents]


def parse_log(txt):
    start = end = None
    for ln in txt.splitlines():
        if ln.startswith('start gamma='):
            try:
                start = float(ln.split('=')[1].split(',')[0])
            except Exception:
                pass
        if ln.startswith('{"case"'):
            try:
                end = json.loads(ln).get('gamma')
            except Exception:
                pass
    fatal = [ln for ln in txt.splitlines() if ln.startswith('FATAL')]
    return start, end, (fatal[0] if fatal else None)


def run_attempt(a, st, case, label, parent, extra):
    key = f"{case} | {label}"
    tag = tag_of(case, a.ne)
    adir = os.path.join(a.out, 'attempts', f"{tag}__{label.replace(' ', '_')}")
    if os.path.isdir(adir):
        shutil.rmtree(adir)                       # a killed attempt starts over
    os.makedirs(adir)
    args = list(extra or [])
    if parent is not None:
        src = os.path.join(a.out, tag_of(parent, a.ne) + '.npz')
        if not os.path.exists(src):
            rec = dict(case=case, label=label, parent=parent, start=None, end=None,
                       fatal='no parent record', rc=None, seconds=0.0, dir=adir)
            st.put(key, rec)
            print(f"  {case:10s} {label:22s}   skipped: no parent record", flush=True)
            return rec
        args = ['--resume', src]
    fl = loop_flags(a.ne, a.outer)
    if 'FB' in case or 'FC' in case:
        fl += ATOMS.format(nf=80 if a.ne <= 100 else 160).split()
    argv = [PY, os.path.join(ROOT, 'run_case_epi.py'), '--case', case, '--start', '0', '--eps', str(a.eps)] \
        + fl + args + ['--outdir', adir]
    logf = os.path.join(adir, 'run.log')
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with open(logf, 'w') as fh:
        fh.write(' '.join(argv) + '\n'); fh.flush()
        r = subprocess.run(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env)
    start, end, fatal = parse_log(open(logf).read())
    rec = dict(case=case, label=label, parent=parent, start=start, end=end, fatal=fatal, rc=r.returncode,
               seconds=round(time.time() - t0, 1), dir=adir)
    st.put(key, rec)
    print(f"  {case:10s} {label:22s} {rec['seconds']:7.0f}s  start {start}  end {end}"
          + (f"  {fatal}" if fatal else '') + (f"  rc={r.returncode}" if r.returncode else ''), flush=True)
    return rec


def finalise(a, st, case, atts):
    """The case's record = its best attempt (lowest certified value)."""
    tag = tag_of(case, a.ne)
    best = None
    for label, _, _ in atts:
        r = st.get(f"{case} | {label}")
        if r and r.get('end') is not None and os.path.exists(os.path.join(r['dir'], tag + '.npz')):
            if best is None or r['end'] < best['end']:
                best = r
    if best is not None:
        for ext in ('.json', '.npz'):
            shutil.copy(os.path.join(best['dir'], tag + ext), os.path.join(a.out, tag + ext))
    st.put(f"{case} | FINAL", dict(gamma=(best['end'] if best else None), from_=(best['label'] if best else None)))
    print(f"== {case:10s} final {best['end'] if best else 'NO RECORD'}"
          + (f"  ({best['label']})" if best else ''), flush=True)


def run(a):
    os.makedirs(os.path.join(a.out, 'attempts'), exist_ok=True)
    st = State(os.path.join(a.out, 'check_state.json'))
    order = closure(ALL27)
    seeds = usable_seeds(a.ne, 20.0)
    atts = {c: attempts_of(c, a.ne, seeds) for c in order}
    parents = {c: nesting('--parents', c) for c in order}
    print(f"lattice check: ne={a.ne} outer={a.outer} jobs={a.jobs}; {len(order)} cases, "
          f"{sum(len(v) for v in atts.values())} attempts", flush=True)
    done = {c for c in order if st.get(f"{c} | FINAL") is not None}
    running, lock = {}, threading.Lock()
    t0 = time.time()

    def worker(c, lab, par, ex):
        run_attempt(a, st, c, lab, par, ex)

    while len(done) < len(order):
        with lock:
            # finalise cases whose attempts are all complete
            for c in order:
                if c in done:
                    continue
                if all(st.get(f"{c} | {lab}") is not None for lab, _, _ in atts[c]) and \
                        not any(k[0] == c for k in running):
                    finalise(a, st, c, atts[c]); done.add(c)
            # launch ready attempts
            for c in order:
                if c in done or not all(p in done for p in parents[c]):
                    continue
                for lab, par, ex in atts[c]:
                    if len(running) >= a.jobs:
                        break
                    if st.get(f"{c} | {lab}") is not None or (c, lab) in running:
                        continue
                    th = threading.Thread(target=worker, args=(c, lab, par, ex), daemon=True)
                    running[(c, lab)] = th; th.start()
            for k in [k for k, th in running.items() if not th.is_alive()]:
                running.pop(k)
        time.sleep(2.0)
    print(f"lattice check done in {(time.time() - t0) / 3600:.2f} h", flush=True)
    report(a)


def report(a):
    st = State(os.path.join(a.out, 'check_state.json'))
    order = closure(ALL27)
    g = {c: (st.get(f"{c} | FINAL") or {}).get('gamma') for c in order}
    lines = ["| child | parent | parent gamma | start from parent | end from parent | child gamma | inclusion | descent | nesting |",
             "|---|---|---|---|---|---|---|---|---|"]
    bad = 0
    for c in order:
        for p in nesting('--parents', c):
            r = st.get(f"{c} | from {p}") or {}
            gp, s, e, gc = g.get(p), r.get('start'), r.get('end'), g.get(c)
            inc = (s is not None and gp is not None and abs(s - gp) <= 1e-5 * gp)
            # the log prints the start with 6 decimals: half a unit of the 6th
            # decimal is the resolution of this test (an attempt that cannot
            # move ends at the parent's full-precision value)
            des = (s is not None and e is not None and e <= s + 5e-7)
            nes = (gc is not None and gp is not None and gc <= gp * (1 + 1e-9))
            bad += (not inc) + (not des) + (not nes)
            f = lambda v: 'n/a' if v is None else f"{v:.6f}"
            lines.append(f"| {c} | {p} | {f(gp)} | {f(s)} | {f(e)} | {f(gc)} | {'ok' if inc else 'NO'} | "
                         f"{'ok' if des else 'NO'} | {'ok' if nes else 'NO'} |")
    roots = [c for c in order if not nesting('--parents', c)]
    lines.append("")
    lines.append("roots: " + ", ".join(f"{c} {g.get(c)}" for c in roots))
    lines.append(f"failed checks: {bad}")
    txt = "\n".join(lines)
    open(os.path.join(a.out, 'check_report.md'), 'w').write(txt + "\n")
    print(txt)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--out', default='results_lattice_check_ne200')
    ap.add_argument('--outer', type=int, default=3)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    report(a) if a.report else run(a)
