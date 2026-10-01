# Dergham, Sipp & Robinet (JFM 2013): FEniCSx reproduction of figures 2 and 3

Tested with DOLFINx 0.11 using complex PETSc/SLEPc and MUMPS, on a 2-core, 8 GB machine.

## Pipeline
```bash
python mesh_large.py                       # gmsh: large domain [-20,100]x[0,20], 60k triangles
python baseflow_newton.py                  # Newton, Re = 50 -> 150 -> 300 -> 450 -> 600 (~5 min, ~2 GB)
                                           #   -> fx_data/baseflow_fenicsx.npz (interpolated on the small mesh)
python compare_baseflow.py                 # vs cb_newton.txt (fx_data/fields.npz), needs mesh.msh
python resolvent_gains.py --baseflow fx_data/baseflow_fenicsx.npz --tag fx   # ~40 s per omega, ~1.5 GB
python resolvent_gains.py --baseflow fx_data/fields.npz --tag ff             # same with your FreeFEM base flow
python figures.py                          # branch identification, fig. 2 check, fig. 3 panels
```
`fx_data/` already contains your mesh (XDMF), your base flow (`fields.npz`, from cb_newton.txt)
and mine (`baseflow_fenicsx.npz`). To redo it from FreeFEM files, use `export_for_fenicsx.edp`
and then `freefem_to_fenicsx.py`.

## Formulation (`nsforms.py`)
* Taylor–Hood P2-P2-P1 elements. Steady NS residual `F(U,P)`. Jacobian `dF/dW = ufl.derivative(F)`.
* Newton uses dF/dW. The linear operator is `J = -dF/dW` taken at the base flow, and the mass
  matrix `M` is the velocity mass. The resolvent is `R = (i omega M - J)^-1`, and the gains solve
  `P^T R^H Q R P f = lambda^2 Qf f` with SLEPc (GHEP). One MUMPS LU per omega; R^H reuses the
  same factors.
* Base-flow BCs (paper section 2.1): u = (1,0) at x = -20; free slip on y = 1 for -20 < x < -2;
  no slip from x = -2 onwards (plate, arc, y = 0); symmetry at y = 20; traction-free outlet at x = 100.
  The arc is centred at (0,-1.5) with R = 2.5, tangent to the plate (fitted to your mesh).
* Perturbation BCs: u = 0 at inlet and walls, v = 0 at the top, natural at the outlet (as in svd2.edp).

## Results
Base flow: max |U_FEniCSx - U_FreeFEM| = 6.5e-4 on the linear-analysis domain. Separation at
(0.802, 0.868) for both; reattachment at x = 10.643 (FEniCSx) and 10.642 (FreeFEM).

Figure 2 circles (digitised from the 300-dpi page, about ±3 % reading accuracy):

| branch | omega | paper | FEniCSx BF | FreeFEM BF | diff | index |
|---|---|---|---|---|---|---|
| A1 | 0.25 | 1.60e5 | 1.617e5 | 1.614e5 | +1.1 % | 1 |
| A1 | 0.75 | 2.22e6 | 2.214e6 | 2.215e6 | -0.3 % | 1 |
| A1 | 2    | 7.74e3 | 7.80e3  | 7.85e3  | +0.8 % | 1 |
| A1 | 3    | 170    | 168.2   | 168.8   | -1.1 % | 4 |
| A2 | 0.75 | 620    | 626.7   | 626.3   | +1.1 % | 3 |
| A2 | 1.5  | 215    | 218.8   | 218.9   | +1.8 % | 5 |
| B1 | 2    | 606    | 615.0   | 615.1   | +1.5 % | 2 |
| B1 | 4    | 231    | 227.8   | 227.7   | -1.4 % | 1 |
| B2 | 2    | 298    | 298.4   | 298.3   | +0.1 % | 3 |
| B3 | 2    | 226    | 227.4   | 227.4   | +0.6 % | 4 |

Branches are identified by where the energy sits (`classify.py`). A1's forcing has at least 94 %
of its energy near separation. B-branch responses have less than 35 % of their energy below y = 1.2.

Figure 3: see `results/fig3_side_by_side.png`. The phase is arbitrary; I fixed it so the response
maximum of u is real.

## Open points
* The paper quotes separation at x ≈ 0.6 and δ* = 0.082 at x = 0. Both your base flow and mine give
  separation at x = 0.80. δ* at x = 0 depends on the definition, because the flow is strongly
  accelerated at the corner: I get 0.037 integrating up to y = 3.5 with U_e = u(3.5). The two base
  flows agree with each other, so this is about the paper's definitions, not the computations.
* The full figure 2 (40 gains, omega in [0, 6]) was not recomputed. Estimate: about 2–3 min per
  omega with nev = 40 on this machine, so about 2–3 h for 60 frequencies. That is where a cluster
  helps: one omega per job, embarrassingly parallel.
