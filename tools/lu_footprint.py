#!/usr/bin/env python3
"""lu_footprint.py -- how many frequencies fit in parallel on one node (cavity).

Run on ONE cluster node, in the dolfinx/PETSc environment (petsc4py with MUMPS):

    python tools/lu_footprint.py --A A.dat --M M.dat --omega 1.0 --which plant
    python tools/lu_footprint.py --A A.dat --M M.dat --omega 1.0 --which pencil2n

A.dat, M.dat: the cavity's assembled Jacobian and mass matrix in PETSc binary
format (MatView to a binary viewer), complex scalars.

  plant    : factor S = i w M - A (n x n) -- what every sweep frequency and
             every reduced-model anchor factorises.
  pencil2n : factor [[S, M], [-M, -S^H]] (2n x 2n) -- the sparsity structure of
             the level-set certificate's Hamiltonian-type pencil (its fill is the
             tighter memory constraint).

Prints MUMPS's own numbers: INFOG(9) (entries in the factors), INFOG(16)
(max MB per process for the numeric factorisation), INFOG(17) (total MB),
and wall times of the analysis and of the numeric factorisation.  With
OMP_NUM_THREADS / -mat_mumps_use_omp_threads set outside, also the thread
scaling.  The number of concurrent frequencies per node is then about
    0.85 x 512 GB / INFOG(17)
capped by where aggregate throughput stops rising (tools/lu_concurrency.sh).
"""
import argparse, time, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--A', required=True); ap.add_argument('--M', required=True)
    ap.add_argument('--omega', type=float, default=1.0)
    ap.add_argument('--which', choices=['plant', 'pencil2n'], default='plant')
    a = ap.parse_args()
    from petsc4py import PETSc
    def load(f):
        v = PETSc.Viewer().createBinary(f, 'r'); m = PETSc.Mat().load(v); v.destroy(); return m
    A = load(a.A); M = load(a.M)
    S = M.copy(); S.scale(1j * a.omega); S.axpy(-1.0, A)          # S = i w M - A
    if a.which == 'pencil2n':
        SH = S.copy(); SH.hermitianTranspose(); SH.scale(-1.0)
        Mn = M.copy(); Mn.scale(-1.0)
        K = PETSc.Mat().createNest([[S, M], [Mn, SH]]); K = K.convert('aij')
    else:
        K = S
    n = K.getSize()[0]
    ksp = PETSc.KSP().create(); ksp.setOperators(K); ksp.setType('preonly')
    pc = ksp.getPC(); pc.setType('lu'); pc.setFactorSolverType('mumps')
    t0 = time.time(); pc.setUp(); t_fact = time.time() - t0
    F = pc.getFactorMatrix()
    infog = {k: F.getMumpsInfog(k) for k in (9, 16, 17)}
    print(f"{a.which}: n = {n}, nnz(A-structure) = {K.getInfo()['nz_used']:.3e}")
    print(f"  MUMPS INFOG(9)  entries in factors : {infog[9]}")
    print(f"  MUMPS INFOG(16) max MB per process : {infog[16]}")
    print(f"  MUMPS INFOG(17) total MB           : {infog[17]}")
    print(f"  analysis + numeric factorisation   : {t_fact:.1f} s")
    b = K.createVecLeft(); b.set(1.0); x = K.createVecRight()
    t0 = time.time(); ksp.solve(b, x); print(f"  one solve                          : {time.time() - t0:.2f} s")
    print(f"  => concurrent factorisations per 512 GB node (85 %): {int(0.85 * 512e3 / max(infog[17], 1))}")


if __name__ == '__main__':
    sys.exit(main())
