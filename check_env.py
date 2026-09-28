"""
check_env.py — verify the co-design toolchain is present and consistent.

Runs a small *functional* test of each component (not just an import) and
prints a PASS/FAIL summary.  Most important check: PETSc must use COMPLEX
scalars, otherwise the Ginzburg-Landau / resolvent work cannot run.

Run:  python check_env.py
"""
import os
import sys
import traceback

results = []          # (name, ok, detail)
REGISTRY = []         # (key, name, fn), in declaration order


def _key(name):
    return name.split("(")[0].strip().replace(" ", "_").lower()


def check(name):
    """Register a probe.

    Probes are REGISTERED here and run later, in a subprocess by default.
    They used to run at decoration time, in this process, with the report
    printed at the end -- which meant any probe that died without raising a
    Python exception took the whole gate down and lost every result including
    the ones that had already passed.

    That is not hypothetical.  On the cluster (2026-09-11, host f02) the PETSc
    probe called MPI_Init outside a job allocation, PMIX had no transport key,
    and Open MPI's MPI_ERRORS_ARE_FATAL called MPI_Abort.  The process died;
    numpy, torch and pyGRANSO were never reported at all, and the validation
    script's `|| echo NOTE` masked it.  The gate looked like it had run.

    MPI_Abort, a segfault and a C-level exit are not catchable in Python, so
    the only containment is a process boundary.
    """
    def deco(fn):
        REGISTRY.append((_key(name), name, fn))
        return fn
    return deco


def run_one(key):
    """Run a single probe in THIS process and report it as one JSON line."""
    import json
    for k, name, fn in REGISTRY:
        if k == key:
            try:
                print("__RESULT__" + json.dumps(
                    dict(key=k, ok=True, detail=str(fn()))))
            except Exception as e:
                if "-v" in sys.argv:
                    traceback.print_exc()
                print("__RESULT__" + json.dumps(
                    dict(key=k, ok=False,
                         detail=f"{type(e).__name__}: {e}")))
            return 0
    print(f"__RESULT__{{\"key\": \"{key}\", \"ok\": false, "
          f"\"detail\": \"no such check\"}}")
    return 2


def run_all_isolated():
    """Each probe in its own process, so a hard abort costs one row."""
    import json
    import subprocess
    out = []
    for k, name, _fn in REGISTRY:
        argv = [sys.executable, os.path.abspath(__file__), "--only", k]
        if "-v" in sys.argv:
            argv.append("-v")
        try:
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=300)
        except subprocess.TimeoutExpired:
            out.append((name, False, "timed out after 300 s"))
            continue
        line = next((ln for ln in r.stdout.splitlines()
                     if ln.startswith("__RESULT__")), None)
        if line is None:
            # died without reporting: MPI_Abort, segfault, os._exit
            tail = " | ".join(
                [ln for ln in (r.stderr or "").strip().splitlines()
                 if ln.strip()][-2:])[:160]
            why = (f"probe died without reporting (exit {r.returncode})"
                   + (f": {tail}" if tail else ""))
            err = r.stderr or ""
            if "MPI" in err or "PMIX" in err:
                # Measured, in this order:
                #   f02 (login), Intel module loaded : MPI_Init FAILS
                #   f02 (login), after module purge  : PETSc/SLEPc PASS
                #   n046 (compute, in a job, Intel
                #        module loaded via env.sh)   : PETSc/SLEPc/dolfinx PASS
                # The third observation rules out the module conflict that the
                # second one suggested: Intel MPI and the conda Open MPI
                # coexist perfectly well inside an allocation.  The cause is
                # the allocation, not the modules.
                why += ("  -- MPI_Init failed.  Usual cause: no job "
                        "allocation.  Open MPI needs one; on a compute node "
                        "under sbatch this passes with the site Intel module "
                        "loaded (verified, job 15966182 on n046).  On a login "
                        "node `module purge` also happens to make it pass, "
                        "but that is not the fix -- submitting a job is.")
            out.append((name, False, why))
            continue
        d = json.loads(line[len("__RESULT__"):])
        detail = d["detail"]
        if "GLIBCXX" in detail or "libstdc++" in detail:
            # Seen on f02 after `module purge`: the conda numpy C-extension
            # needs a newer libstdc++ than /lib64 ships, and the site Intel
            # module was the thing supplying it.  Purging fixes MPI and breaks
            # this; loading the module fixes this and breaks MPI.  Pointing
            # LD_LIBRARY_PATH at the conda env's own lib satisfies both.
            detail += ("\n        -> try: export LD_LIBRARY_PATH="
                       "$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   (the conda env "
                       "ships its own libstdc++; /lib64's is older)")
        if "libimf.so" in detail:
            # libimf is Intel's math runtime.  The Intel module supplies it,
            # but loading that module also drags in Intel MPI, which is what
            # broke PETSc in the first place.  Two ways out, both narrower than
            # reloading the module:
            #   - add ONLY the Intel compiler runtime directory, APPENDED so it
            #     cannot displace the conda libstdc++ that torch needs;
            #   - or install the runtime into the env, which needs no module.
            detail += (
                "\n        -> libimf is Intel's math runtime.  Do NOT reload "
                "the full Intel module: it re-adds Intel MPI and breaks PETSc."
                "\n           Either  find / -name 'libimf.so' 2>/dev/null | "
                "head   and then"
                "\n                   export LD_LIBRARY_PATH="
                "\"$LD_LIBRARY_PATH:<that_dir>\"   (APPEND -- prepending puts "
                "Intel's libstdc++ ahead of the conda one and breaks torch),"
                "\n           or      pip install intel-cmplr-lib-rt   "
                "(ships libimf.so inside the env; no module needed).")
        out.append((name, bool(d["ok"]), detail))
    return out


def run_all_in_process():
    out = []
    for _k, name, fn in REGISTRY:
        try:
            out.append((name, True, str(fn())))
        except Exception as e:
            if "-v" in sys.argv:
                traceback.print_exc()
            out.append((name, False, f"{type(e).__name__}: {e}"))
    return out


@check("numpy")
def _():
    import numpy as np
    return f"numpy {np.__version__}"


@check("PETSc (complex scalars)")
def _():
    import numpy as np
    from petsc4py import PETSc
    kind = np.dtype(PETSc.ScalarType).kind
    ver = ".".join(map(str, PETSc.Sys.getVersion()))
    if kind != "c":
        raise RuntimeError(f"PETSc ScalarType is {PETSc.ScalarType} (REAL). "
                           "Rebuild the env with a *complex* PETSc.")
    return f"PETSc {ver}, ScalarType={PETSc.ScalarType.__name__}"


@check("SLEPc (tiny eigensolve)")
def _():
    from petsc4py import PETSc
    from slepc4py import SLEPc
    # 3x3 diagonal-ish complex matrix; check we recover an eigenvalue
    n = 3
    A = PETSc.Mat().createAIJ([n, n]); A.setUp()
    for i in range(n):
        A.setValue(i, i, complex(i + 1, i))
    A.assemble()
    eps = SLEPc.EPS().create()
    eps.setOperators(A)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setDimensions(nev=1)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.solve()
    lam = eps.getEigenvalue(0)
    if abs(lam - complex(3, 2)) > 1e-8:
        raise RuntimeError(f"unexpected eigenvalue {lam}")
    return f"SLEPc OK, largest-real eig = {lam:.3g}"


@check("PyTorch (CPU autograd)")
def _():
    import torch
    x = torch.randn(4, requires_grad=True, dtype=torch.double)
    (x.pow(2).sum()).backward()
    err = float((x.grad - 2 * x).abs().max())
    if err > 1e-10:
        raise RuntimeError("autograd gradient wrong")
    return f"torch {torch.__version__}, cuda_available={torch.cuda.is_available()}"


@check("pyGRANSO (constrained solve, OSQP)")
def _():
    import torch
    from pygranso.pygranso import pygranso
    from pygranso.pygransoStruct import pygransoStruct

    def comb_fn(X):
        f = X.x ** 2 + X.y ** 2
        ci = pygransoStruct()
        ci.c1 = 1.0 - (X.x + X.y)          # x + y >= 1
        return f, None, ci

    opts = pygransoStruct()
    opts.QPsolver = "osqp"                  # <-- avoids Gurobi license
    opts.print_level = 0
    opts.opt_tol = 1e-8
    opts.maxit = 200
    opts.x0 = torch.tensor([[2.0], [-1.0]], dtype=torch.double)
    soln = pygranso(var_spec={"x": [1, 1], "y": [1, 1]},
                    combined_fn=comb_fn, user_opts=opts)
    if abs(float(soln.final.f) - 0.5) > 1e-4:
        raise RuntimeError(f"wrong optimum f={soln.final.f}")
    return f"pyGRANSO OK, f*={float(soln.final.f):.4f} (analytic 0.5)"


@check("dolfinx (assemble a mass matrix into PETSc)")
def _():
    from mpi4py import MPI
    import dolfinx
    from dolfinx import mesh as dmesh, fem
    from dolfinx.fem.petsc import assemble_matrix
    import ufl
    msh = dmesh.create_interval(MPI.COMM_WORLD, 20, [-1.0, 1.0])
    # functionspace() (>=0.8) vs FunctionSpace() (<=0.7)
    try:
        V = fem.functionspace(msh, ("Lagrange", 2))
    except AttributeError:
        V = fem.FunctionSpace(msh, ("Lagrange", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(ufl.inner(u, v) * ufl.dx)
    A = assemble_matrix(a)
    A.assemble()
    m, n = A.getSize()
    return f"dolfinx {dolfinx.__version__}, mass matrix {m}x{n} ({A.getType()})"


REQUIRED = ("numpy", "pytorch", "pygranso")   # the three GL needs


if __name__ == "__main__":
    if "--only" in sys.argv:
        sys.exit(run_one(sys.argv[sys.argv.index("--only") + 1]))

    results = (run_all_in_process() if "--in-process" in sys.argv
               else run_all_isolated())

    print("=" * 64)
    print("Co-design toolchain check")
    print("=" * 64)
    # Which interpreter is actually being probed.  Without this line, an
    # environment where a site module has shadowed the conda python looks like
    # "torch is missing" rather than "you are running a different python".
    print(f"python  : {sys.executable}")
    print(f"conda   : {os.environ.get('CONDA_PREFIX', '(none)')}")
    _inenv = (os.environ.get('CONDA_PREFIX') and
              sys.executable.startswith(os.environ['CONDA_PREFIX']))
    if os.environ.get('CONDA_PREFIX') and not _inenv:
        print("WARNING : CONDA_PREFIX is set but this python is NOT from it --")
        print("          a module has shadowed the conda interpreter on PATH.")
    print(f"modules : {os.environ.get('LOADEDMODULES', '(none)') or '(none)'}")
    print("=" * 64)
    width = max(len(n) for n, _, _ in results) if results else 0
    all_ok = True
    for name, ok, detail in results:
        tag = "PASS" if ok else "FAIL"
        all_ok &= ok
        print(f"[{tag}] {name:<{width}}  {detail}")
    print("=" * 64)

    # GL needs numpy, torch and pyGRANSO.  PETSc, SLEPc and dolfinx are for the
    # cavity stage and their absence is not a GL failure -- but it must be
    # REPORTED as a distinct outcome, not folded into one boolean, or a caller
    # cannot tell "the three that matter passed" from "nothing ran".
    got = {_key(n): ok for n, ok, _ in results}
    missing = [k for k in REQUIRED if not got.get(k)]
    if missing:
        print(f"REQUIRED CHECK FAILED: {', '.join(missing)}")
        print("GL cannot run without these three.")
        sys.exit(1)
    if all_ok:
        print("ALL GOOD — `conda activate codesign` and you're set.")
        sys.exit(0)
    print("numpy / PyTorch / pyGRANSO PASS: GL is runnable.")
    print("Other checks failed; they are needed only at cavity scale "
          "(stage 7).")
    sys.exit(0)
