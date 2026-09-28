"""selftest.py — verify the package is complete and working BEFORE submitting.
    python selftest.py
Runs in ~60 s at ne=100 and checks: imports, assembly, sparse kernels against a
dense reference, gradients against finite differences, prepare, and one short
optimisation of every parameterisation group.
"""
from __future__ import annotations
import os, sys, subprocess, tempfile, shutil
import sys, os
if sys.version_info[:2] < (3, 10):
    sys.exit("selftest: wrong Python (%d.%d). Run  `source env.sh`  first so the "
             "conda 'codesign' env (python 3.12) is active, not ~/.local/python3.8."
             % sys.version_info[:2])

import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ok = True

def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name:44s} {detail}")

# 1 ---- imports
import fe_gl, closedloop_sparse, fe_codesign, fe_pipeline
check("imports", True)

# 2 ---- assembly + spectrum vs closed form
A, M, xn, free = fe_gl.assemble_gl(ne=100)
import scipy.sparse.linalg as spla
# shift-invert near the leading eigenvalue; ncv generous for a small matrix,
# and fall back to a couple of alternative shifts if ARPACK cannot build a
# factorisation (can happen on tiny systems / old ARPACK builds).
def _leading(k=3):
    for sig in (-0.02 - 0.65j, -0.05 - 0.65j, -0.02 - 0.60j, 0.0 - 0.648j):
        try:
            w = spla.eigs(A.tocsc(), k=k, M=M.tocsc(), sigma=sig, which='LM',
                          ncv=min(A.shape[0] - 1, max(3 * k + 2, 30)),
                          maxiter=10000, tol=1e-9,
                          return_eigenvectors=False)
            return w[np.argsort(-w.real)]
        except Exception:
            continue
    return None
v = _leading(3)
a = fe_gl.analytic_eigs(fe_gl.MU0_STABLE, 3); a = a[np.argsort(-a.real)]
if v is None:
    check("FE spectrum vs analytic", False, "ARPACK could not converge "
          "(check you are on the conda python, not ~/.local/python3.8)")
else:
    err = float(np.max(np.abs(v[:3] - a[:3])))
    check("FE spectrum vs analytic", err < 1e-2,
          f"err={err:.1e}, nnz/row={A.nnz/A.shape[0]:.1f}")

# 3 ---- sparse closed-loop gain vs dense reference
D = fe_codesign.FECoDesign(ne=100, r=7, sob_len=0.0)
# tridiagonal controller: (d, e, f) with stable poles, plus output row
_r = 7
arow = (np.linspace(-1.0, -4.0, _r).astype(complex),
        0.15 * np.ones(_r - 1, complex),
        0.10 * np.ones(_r - 1, complex))
nvec = np.linspace(1.0, 0.2, _r).astype(complex)
b, c, pa, ps, na, ns = D.bc(-1.03, 0.98)
cl = D.cl(b, c)
gs = cl.gain(-0.54, arow, nvec, 0j, want_vectors=False)['sigma']
gd = cl.dense_reference(-0.54, arow, nvec, 0j)
check("sparse gain vs dense reference", abs(gs-gd)/gd < 1e-10,
      f"rel={abs(gs-gd)/gd:.1e}")

# 4 ---- Krylov alpha vs dense eig
from scipy.linalg import eig as _eig
al_k, lam, vv, uu = cl.alpha_krylov(arow, nvec, 0j)   # production defaults
n = D.n; r = D.r
Ad = np.asarray(D.A.todense()); Md = np.asarray(D.M.todense())
er = np.full(r, 1.0, complex)          # tridiagonal input gauge B_c = ones
import controller as _ctl
L = _ctl.A_of(*arow)
Acl = np.zeros((n+r, n+r), complex); Mcl = np.zeros((n+r, n+r), complex)
Acl[:n,:n]=Ad; Acl[:n,n:]=np.outer(b,nvec); Acl[n:,:n]=np.outer(er,c.conj()); Acl[n:,n:]=L
Mcl[:n,:n]=Md; Mcl[n:,n:]=np.eye(r)
al_d = float(np.max(_eig(Acl, Mcl, right=False).real))
check("Krylov alpha vs dense eig", abs(al_d-al_k) < 1e-7,
      f"err={abs(al_d-al_k):.1e}")

# 5 ---- gradients vs finite differences (each group)
flags = dict(K=True, B=True, C=True, FB=True, FC=True)
va = D.gauss(-1.03); vs = D.gauss(0.98)
def gam_of(arow_, nvec_, K_, xa_, xs_, va_, vs_):
    bb, cc, pa_, ps_, na_, ns_ = D.bc(xa_, xs_, va_, vs_)
    g, w, res, clh, mem, nact = D.hinf(arow_, nvec_, K_, bb, cc)
    return g, res, clh, pa_, ps_, na_, ns_
g0, res, clh, pa_, ps_, na_, ns_ = gam_of(arow, nvec, 0j, -1.03, 0.98, va, vs)
gg = D.grads(clh, res['x'], res['z'], arow, nvec, 0j, -1.03, 0.98,
             pa_, ps_, na_, ns_, flags)
# placement gradients: probe with fields OFF (bc() uses the field when given)
g0p, resp, clhp, pap, psp, nap, nsp = gam_of(arow, nvec, 0j, -1.03, 0.98, None, None)
ggp = D.grads(clhp, resp['x'], resp['z'], arow, nvec, 0j, -1.03, 0.98,
              pap, psp, nap, nsp, dict(K=False, B=True, C=True, FB=False, FC=False))
h = 1e-6
rng = np.random.default_rng(0)
da = rng.standard_normal(D.n)+1j*rng.standard_normal(D.n)
ds = rng.standard_normal(D.n)+1j*rng.standard_normal(D.n)
tests = {
 'd/dxa': (float(np.real(ggp['xa'])),
           (gam_of(arow,nvec,0j,-1.03+h,0.98,None,None)[0]-gam_of(arow,nvec,0j,-1.03-h,0.98,None,None)[0])/(2*h)),
 'd/dxs': (float(np.real(ggp['xs'])),
           (gam_of(arow,nvec,0j,-1.03,0.98+h,None,None)[0]-gam_of(arow,nvec,0j,-1.03,0.98-h,None,None)[0])/(2*h)),
 'd/dfield_a': (float(np.real(np.vdot(gg['va'].real,da.real)+np.vdot(gg['va'].imag,da.imag))),
           (gam_of(arow,nvec,0j,-1.03,0.98,va+h*da,vs)[0]-gam_of(arow,nvec,0j,-1.03,0.98,va-h*da,vs)[0])/(2*h)),
 'd/dfield_s': (float(np.real(np.vdot(gg['vs'].real,ds.real)+np.vdot(gg['vs'].imag,ds.imag))),
           (gam_of(arow,nvec,0j,-1.03,0.98,va,vs+h*ds)[0]-gam_of(arow,nvec,0j,-1.03,0.98,va,vs-h*ds)[0])/(2*h)),
}
for k,(an,fd) in tests.items():
    check(f"gradient {k}", abs(an-fd)/max(abs(fd),1e-12) < 5e-5,
          f"analytic={an:+.6e} FD={fd:+.6e}")

# 5b2 ---- tridiagonal parametrisation: exact modal round-trip
try:
    import controller as _ct
    _rng = np.random.default_rng(3)
    _r2 = 5
    _A = np.diag(-1 - _rng.random(_r2)) + 0.2 * np.diag(_rng.standard_normal(_r2 - 1), 1)
    _B = np.full((_r2, 1), 1.0, complex)
    _C = (_rng.standard_normal((1, _r2)) + 0j)
    _d, _e, _f, _nv, _K = _ct.from_state_space(_A, _B, _C)
    _th = _ct.pack(_d, _e, _f, _nv, _K)
    _s = 0.5 + 1.3j
    _ref = (_C @ np.linalg.solve(_s * np.eye(_r2) - _A, _B))[0, 0]
    _got = _ct.transfer(_th, _r2, _s)
    check("tridiagonal round-trip (transfer fn)", abs(_ref - _got) / abs(_ref) < 1e-10,
          f"rel err={abs(_ref-_got)/abs(_ref):.1e}")
except Exception as e:
    check("tridiagonal round-trip (transfer fn)", False, str(e)[:50])

# 5c ---- Sobolev change-of-variable produces smoother fields than plain L2
try:
    import numpy as _np
    Dp = fe_codesign.FECoDesign(ne=100, r=7, sob_len=0.8)
    xi = _np.cos(3*Dp.xn) + 0.4*((-1.0)**_np.arange(Dp.n))   # low-k + grid noise
    psi = Dp.smooth_map(xi.astype(complex))
    r_in = _np.linalg.norm(_np.diff(_np.abs(xi), 2))
    r_out = _np.linalg.norm(_np.diff(_np.abs(psi), 2))
    check("Sobolev map smooths fields", r_out < 0.5 * r_in,
          f"roughness {r_in:.2f} -> {r_out:.2f}")
except Exception as e:
    check("Sobolev map smooths fields", False, str(e)[:50])

# 5d ---- certification decision procedure (Schwerdtner-Mengi-Voigt)
try:
    import certify_sparse as _cs, controller as _c2
    _Dc = fe_codesign.FECoDesign(ne=100, r=7, sob_len=0.0)
    _ini = _np2 = None
    _d = np.linspace(-1.0, -4.0, 7).astype(complex)
    _e = 0.15 * np.ones(6, complex); _f = 0.10 * np.ones(6, complex)
    _nv = np.linspace(1.0, 0.2, 7).astype(complex)
    _b, _c, _pa, _ps, _na, _ns = _Dc.bc(-1.03, 0.98)
    _cl = _Dc.cl(_b, _c)
    _sig = lambda w: _cl.gain(w, (_d, _e, _f), _nv, 0j,
                              want_vectors=False)['sigma']
    _ws = np.linspace(-4, 4, 601)
    _gt = max(_sig(w) for w in _ws)
    _lo, _, _ = _cs.certify_greater(_Dc.A, _Dc.M, _b, _c, _c2.A_of(_d, _e, _f),
                                 np.full(7, 1.0, complex), _nv, 0j,
                                 0.5 * _gt, _sig, -4, 4)
    _hi, _, _ = _cs.certify_greater(_Dc.A, _Dc.M, _b, _c, _c2.A_of(_d, _e, _f),
                                 np.full(7, 1.0, complex), _nv, 0j,
                                 2.0 * _gt, _sig, -4, 4)
    check("certification decision (below/above norm)", _lo and (not _hi),
          f"gamma/2 -> {_lo} (want True), 2*gamma -> {_hi} (want False)")
except Exception as e:
    check("certification decision (below/above norm)", False, str(e)[:55])

# 6 ---- end-to-end: prepare + one tiny run of each group
tmp = tempfile.mkdtemp()
try:
    for f in os.listdir(HERE):
        if f.endswith(('.py', '.txt')): shutil.copy(os.path.join(HERE, f), tmp)
    # the campaign tools now live in tools/ (see cleanup_archive.py); the
    # sandbox needs them too, and their root shim resolves to tmp because the
    # top-level modules above were copied there.
    if os.path.isdir(os.path.join(HERE, 'tools')):
        shutil.copytree(os.path.join(HERE, 'tools'),
                        os.path.join(tmp, 'tools'), dirs_exist_ok=True)
    os.makedirs(os.path.join(tmp, 'results'), exist_ok=True)
    r = subprocess.run([sys.executable, os.path.join('tools', 'prepare.py'),
                        '--ne', '100'],
                       cwd=tmp, capture_output=True, text=True, timeout=600)
    check("prepare.py", r.returncode == 0 and os.path.exists(os.path.join(tmp,'init_ne100.npz')),
          r.stdout.strip()[:60])
    for case in ['B', 'K', 'FC']:
        r = subprocess.run([sys.executable, 'run_case_epi.py', '--case', case,
                            '--start', '0', '--ne', '100', '--wall', '25',
                            # explicit: run_case_epi's own default is 1e-3,
                            # 50x weaker than the suite's margin, so without
                            # this the gate exercises a different problem
                            '--eps', '0.05',
                            '--outer', '3', '--cold',
                            '--outdir', 'results'], cwd=tmp,
                           capture_output=True,
                           text=True, timeout=900)
        tag = os.path.join(tmp, 'results', f"{case.replace('+','_')}_s0_ne100")
        # .oracle.csv / .stage.csv were the LADDER driver's per-call traces;
        # the epigraph driver records its history in the .json instead.
        good = r.returncode == 0 and all(os.path.exists(tag + e)
                                         for e in ('.json', '.npz'))
        check(f"run_case_epi.py --case {case}", good,
              (r.stdout.strip().splitlines() or [''])[-1][:70])
    r = subprocess.run([sys.executable, os.path.join('tools', 'collect.py')],
                       cwd=tmp, capture_output=True, text=True, timeout=300)
    check("collect.py", r.returncode == 0 and 'nesting' in r.stdout)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("ALL CHECKS PASSED - ready to sbatch" if ok else
              "SOME CHECKS FAILED - do not submit"))
sys.exit(0 if ok else 1)
