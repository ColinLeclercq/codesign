"""
certify_results.py — CERTIFY every optimisation result.

For each results/<CASE>_s<START>_ne<NE>.{json,npz} this rebuilds the closed loop
(tridiagonal controller + actuator/sensor centres or fields) and runs the
Schwerdtner-Mengi-Voigt decision procedure, sparse and matrix-free.  A case whose
stored gamma was a LOCAL maximum is corrected and flagged.

    python certify_results.py --ne 200                # report
    python certify_results.py --ne 200 --update       # write gamma back
    python certify_results.py --ne 200 --band auto    # band from open-loop sweep
"""
from __future__ import annotations
# ---------------------------------------------------------------------------
# Thread pinning MUST happen before numpy/scipy are imported.
#
# ARPACK is called here with a Python callback that itself calls SuperLU
# (shift-and-invert on a matrix-free operator).  With a threaded BLAS this
# re-entrancy is not safe: on a 96-core login node an unpinned MKL spawns a
# thread team inside the callback and the process dies with SIGSEGV -- every
# case, regardless of its data.  env.sh sets these for batch jobs; this makes
# the script safe when it is run interactively without sourcing env.sh.
# Override deliberately with e.g. OMP_NUM_THREADS=4 if you know what you want.
import os as _os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
_os.environ.setdefault('MKL_DYNAMIC', 'FALSE')
# ---------------------------------------------------------------------------
import argparse, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fe_codesign, controller as ctl
import certify_sparse as cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ne', type=int, default=200)
    ap.add_argument('--results', default='results')
    ap.add_argument('--r', type=int, default=-1,
                    help='controller order; -1 reads it from '
                         'the seed catalogue, which is where '
                         'it is decided')
    ap.add_argument('--sob_len', type=float, default=0.8)
    ap.add_argument('--band', default='poles',
                    help="'wlo,whi', 'auto' (open-loop sweep + margin -- "
                         "blind to compensator resonances), or 'poles' "
                         "(default: 1.5 x the outermost stored controller-"
                         "pole |Im| across all results, floored by the runs' "
                         "own recorded bands)")
    ap.add_argument('--rtol', type=float, default=1e-4)
    ap.add_argument('--update', action='store_true')
    ap.add_argument('--case', default=None,
                    help='only this case, e.g. K+FB+FC')
    ap.add_argument('--start', type=int, default=None,
                    help='only this start index')
    ap.add_argument('--isolate', action='store_true',
                    help='run each case in a subprocess: a hard crash '
                         '(segfault) then kills only that case')
    a = ap.parse_args()
    if getattr(a, 'r', -1) is not None and a.r < 0:
        from ctrl_order import infer_r
        a.r = infer_r(a.ne, getattr(a, 'results', None))

    import numpy as _np, scipy as _sp
    print(f"python {sys.version.split()[0]}  numpy {_np.__version__}  "
          f"scipy {_sp.__version__}  OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS')}")
    D = fe_codesign.FECoDesign(ne=a.ne, r=a.r, sob_len=a.sob_len)
    files = sorted(glob.glob(os.path.join(a.results, f"*_ne{a.ne}.json")))
    if not files:
        print("no results found"); return

    if a.band == 'auto':
        t0 = time.time()
        wlo, whi = cs.band_from_open_loop(D.A, D.M, wmax_scan=8.0, nscan=81,
                                          margin=2.0)
        print(f"band from open-loop sweep: [{wlo:.2f}, {whi:.2f}] "
              f"({time.time()-t0:.1f}s)")
    elif a.band == 'poles':
        # Band from the CLOSED loop's own resonances.  The open-loop sweep is
        # blind to compensator poles (a run certified over [-8, 8] while a
        # design carried a pole at |Im| = 138), and a hand-typed band drifts
        # out of sync with the runs.  One global band, so gamma stays a
        # supremum over the same set for every case: 1.5 x the outermost
        # controller-pole |Im| across all stored designs, floored by the runs'
        # own recorded bands.
        import controller as _ctl
        half = 0.0
        for jf in files:
            try:
                m0 = json.load(open(jf))
                d0 = np.load(jf[:-5] + '.npz', allow_pickle=True)
                if 'd' in d0.files:
                    p0 = np.linalg.eigvals(_ctl.A_of(d0['d'], d0['e'],
                                                     d0['f']))
                    if p0.size:
                        half = max(half, 1.5 * float(np.max(np.abs(p0.imag))))
                half = max(half, abs(float(m0.get('wlo', 0.0))),
                           abs(float(m0.get('whi', 0.0))))
            except Exception:
                pass
        half = max(half, 8.0)
        wlo, whi = -half, half
        print(f"band from stored controller poles + run records: "
              f"[{wlo:.2f}, {whi:.2f}]")
    else:
        wlo, whi = (float(v) for v in a.band.split(','))
        print(f"band (fixed): [{wlo:.2f}, {whi:.2f}]")

    if a.isolate:
        import subprocess
        # NB '=' form throughout: a negative band such as '-5.4,7.6' would
        # otherwise be parsed as an option.
        base = [sys.executable, os.path.abspath(__file__),
                f'--ne={a.ne}', f'--results={a.results}', f'--r={a.r}',
                f'--sob_len={a.sob_len}', f'--band={a.band}',
                f'--rtol={a.rtol}'] + (['--update'] if a.update else [])
        ncrash = 0
        for jf in files:
            m = json.load(open(jf))
            env = dict(os.environ)
            env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       MKL_DYNAMIC='FALSE')
            p = subprocess.run(base + [f"--case={m['case']}",
                                       f"--start={m['start']}"],
                               capture_output=True, text=True, env=env)
            out = [l for l in p.stdout.splitlines()
                   if l.strip() and not l.startswith(('band', 'case'))
                   and 'case(s) had' not in l]
            for l in out:
                print(l)
            if p.returncode != 0:
                ncrash += 1
                sig = -p.returncode if p.returncode < 0 else p.returncode
                print(f"{m['case']:10s} {m['start']:5d} "
                      f"{'CRASHED':>23s}  (exit/signal {sig}) "
                      f"-- see stderr below")
                if p.stderr.strip():
                    print('   ' + p.stderr.strip().splitlines()[-1][:100])
        print(f"\n{ncrash} case(s) crashed hard.")
        return
    print(f"{'case':10s} {'start':>5s} {'stored':>11s} {'certified':>11s} "
          f"{'ratio':>7s} {'calls':>5s} {'sec':>6s}  verdict")
    nbad = 0; nstale = 0
    for jf in files:
        meta = json.load(open(jf))
        if a.case and meta['case'] != a.case:
            continue
        if a.start is not None and meta['start'] != a.start:
            continue
        d = np.load(jf[:-5] + '.npz')
        if 'd' not in d.files:
            # written by a pre-tridiagonal run: different parametrisation, the
            # closed loop cannot be rebuilt from it.  Skip loudly.
            nstale += 1
            print(f"\r{meta['case']:10s} {meta['start']:5d} "
                  f"{meta['gamma']:11.4f} {'STALE':>11s}"
                  f"          -- old canonical format, delete and re-run",
                  flush=True)
            continue
        arow = (d['d'], d['e'], d['f'])
        nvec = d['nvec']; Kfb = complex(d['K'])
        va = d['va'] if d['va'].size else None
        vs = d['vs'] if d['vs'].size else None
        b, c, pa, ps, na, ns = D.bc(float(d['xa']), float(d['xs']), va, vs)
        cl = D.cl(b, c)
        sig = lambda w: cl.gain(w, arow, nvec, Kfb,
                                want_vectors=False)['sigma']
        Ac = ctl.A_of(*arow)
        Bc = np.full(a.r, 1.0, complex)
        lb = max(float(meta['gamma']), float(np.max(d['gain'])))
        t0 = time.time()
        try:
            g, w, nc, ok = cs.hinf_certified_sparse(
                D.A, D.M, b, c, Ac, Bc, nvec, Kfb, sig, gam_lb=lb,
                wlo=wlo, whi=whi, rtol=a.rtol)
        except Exception as e:
            print(f"\r{meta['case']:10s} {meta['start']:5d} "
                  f"{meta['gamma']:11.4f} {'FAILED':>11s}   {str(e)[:34]}",
                  flush=True)
            continue
        dt = time.time() - t0
        ratio = g / float(meta['gamma'])
        rel = ratio - 1.0
        if not ok:
            verdict = 'NOT CERTIFIED'
        elif rel > 2e-4:
            # correction larger than the tolerance any previous pass could
            # justify: a peak the runs genuinely never saw.
            verdict = f'** MISSED PEAK (+{rel:.1e}) -> corrected **'
            nbad += 1
        elif rel > 10 * a.rtol:
            # the stored value was CORRECT at the tolerance it was computed
            # at; this pass simply measures finer.  Not an alarm.
            verdict = f'refined +{rel:.1e} (within prior rtol)'
        else:
            verdict = 'certified'
        print(f"\r{meta['case']:10s} {meta['start']:5d} {meta['gamma']:11.4f} "
              f"{g:11.4f} {ratio:7.3f} {nc:5d} {dt:6.1f}  {verdict}", flush=True)
        if a.update:
            meta['gamma_uncertified'] = float(meta['gamma'])
            meta['gamma_certified'] = float(g)
            meta['certified'] = bool(ok)
            meta['gamma'] = float(g)
            json.dump(meta, open(jf, 'w'), indent=1)
    if nstale:
        print(f"\n!! {nstale} stale result(s) from a previous parametrisation "
              f"were skipped.\n   rm -f results/* and re-run: mixing formats "
              f"invalidates the comparison.")
    print(f"\n{nbad} case(s) had a missed peak."
          + ("  Re-run collect.py / plot_results.py." if nbad and a.update
             else ""))


if __name__ == "__main__":
    main()
