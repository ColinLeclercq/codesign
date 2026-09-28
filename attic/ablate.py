"""
ablate.py — does each tuned option actually earn its place?

Every structural change in this project helped; every TUNED parameter has so far
proved neutral or harmful.  This runs the same case with one option changed at a
time and reports gamma, wall time, stability, and (for --trust) how often the
inner solve escaped the surrogate.

    python ablate.py --case K --ne 80 --wall 120
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, json, re, subprocess, sys, time


VARIANTS = [
    ('baseline',        []),
    ('trust 0 (off)',   ['--trust', '0']),
    ('trust 0.3',       ['--trust', '0.3']),
    ('warm_h 0 (off)',  ['--warm_h', '0']),
    ('nseed 24/0.25',   ['--nseed', '24', '--seed_frac', '0.25']),
    ('alpha_warm 0',    ['--alpha_warm', '0']),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='K')
    ap.add_argument('--ne', type=int, default=80)
    ap.add_argument('--wall', type=float, default=120.)
    ap.add_argument('--outer', type=int, default=6)
    ap.add_argument('--keep', action='store_true',
                    help='keep the per-variant scratch directories')
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"case={a.case}  ne={a.ne}  wall={a.wall}s  outer={a.outer}")
    print(f"{'variant':18s} {'gamma':>10s} {'alpha':>10s} {'sec':>7s} "
          f"{'calls':>6s} {'escapes':>8s}  note")
    for name, extra in VARIANTS:
        out = os.path.join(here, f"_abl_{re.sub('[^a-z0-9]+','_',name.lower())}")
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(out):
            os.remove(os.path.join(out, f))
        cmd = [sys.executable, 'run_case_epi.py', '--case', a.case,
               '--start', '0', '--ne', str(a.ne), '--wall', str(a.wall),
               '--outer', str(a.outer), '--cold', '--outdir', out] + extra
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=here)
        dt = time.time() - t0
        # an "escape" is an outer whose CERTIFIED norm far exceeded the
        # surrogate's t -- the behaviour --trust was introduced to prevent
        esc = 0
        for m in re.finditer(r't=([\d.eE+-]+)\s+certified=([\d.eE+-]+)',
                             p.stdout):
            try:
                t_, g_ = float(m.group(1)), float(m.group(2))
                if g_ > 1.5 * t_:
                    esc += 1
            except ValueError:
                pass
        jf = os.path.join(out, f"{a.case.replace('+','_')}_s0_ne{a.ne}.json")
        keep = a.keep
        if os.path.exists(jf):
            d = json.load(open(jf))
            print(f"{name:18s} {d['gamma']:10.4f} {d['alpha']:10.4f} "
                  f"{dt:7.1f} {d.get('calls',0):6d} {esc:8d}  "
                  f"{'stable' if d['alpha'] < 0 else 'UNSTABLE'}")
        else:
            tail = (p.stdout or p.stderr).strip().split('\n')[-1][:44]
            print(f"{name:18s} {'---':>10s} {'---':>10s} {dt:7.1f} "
                  f"{'-':>6s} {esc:8d}  no result: {tail}")
        if not keep:
            import shutil
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
