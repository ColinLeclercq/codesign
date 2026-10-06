# BFS.md -- the backward-facing-step fork

Every modification made for the backward-facing step (BFS), in one place.
Branch `bfs`, forked from `mimo` (2026-10-01).  The GL path is unchanged:
frozen path bit-identical (10.23128516441147) after every change below.

The case (user, 2026-10-01, revised 2026-10-02): SISO co-design on the
Dergham-Sipp-Robinet (2013) step at Re = 600, on the user's linear-analysis
mesh and base flow (`bfs/archive_dsr`, the user's zip).
- **Plant:** real.
- **Controller:** real, structured, order 5, proper but not strictly proper.
- **Actuator** (since 2026-10-02): the B2 of Barbagallo, Dergham, Sipp, Schmid & Robinet (JFM 2012), a Gaussian body force on the vertical momentum, free centre (x, y).
- **Sensor** (since 2026-10-02): their wall-shear sensor C1, `int chi du/dy` over a window of the floor, free window start.
- **Start:** HIFOO's controller draw at the paper's B2 (0.6, 1.0) and C1 [2.0, 2.6].
- **Disturbance and performance:** a [u, v] forcing field in the velocity mass in, the whole state in the mass seminorm out (the archive's weighted SVD): B B^T = C^T C = Q.
- **gamma:** certified on [0, 10] (option A, user 2026-10-02).
- **Stability:** certified over all frequencies (the winding count, M-free).
- **Rule:** no inverse and no Cholesky factor of the (velocity) mass, anywhere.

(Until 2026-10-02 the ports were two isotropic Gaussians of std 0.5, the
actuator forcing v and the sensor measuring u, with free 2-D centres; that
version is described in the git history, commits 36010c7 and 2f744e4.)

Contents:
1. Operators
2. The plant: ports, symmetry, positions
3. Linear algebra and parallelism
4. Stability and the singular mass matrix
5. The driver
6. Starts, and the small test plant
7. Environment, cluster scripts, run time
8. Validation log
9. Open items

---------------------------------------------------------------------------

## 1. Operators

`bfs/ops_dolfinx.py` assembles the operators with the user's own
`nsforms.py` (DOLFINx 0.11). It runs on one process, in 2 s at n = 191,720.

**Discretisation and boundary conditions**
- Taylor-Hood P2-P2-P1 on the linear-analysis mesh [-1.5, 30] x [0, 3.5].
- `-J = dF/dW` at the base flow, assembled with `bcs`, `diag=1`.
- `M` is the velocity mass, assembled with `bcs`, `diag=0`.
- The perturbation BCs are those of `small_domain_bcs`: u = v = 0 on labels 1 and 2, v = 0 on label 4.

**Output file** (`bfs/plant_io.py`, one .npz):
- `A = J` and `M`, both real CSR;
- per-dof component (u / v / p) and coordinates;
- Dirichlet rows;
- mesh vertices and triangles;
- labelled boundary edges.

The optimiser's processes only read this file. The plant is
`M q' = A q + b u + M f`, `y = c^H q`, with resolvent `(i w M - A)^-1`.

**Validation** (`bfs/check_gains.py`): the optimal gains against the archive's
`results/gains_fx.txt` (resolvent_gains.py: DOLFINx + SLEPc Krylov-Schur
GHEP), using the code's own LU (MUMPS), in two forcing spaces:

| w | archive lambda_1^2 | archive forcing space | co-design forcing space (load M f) |
|---|---|---|---|
| 0.75 | 2.214398e+06 | rel 8.4e-16 | rel 2.1e-16 |
| 2.0 | 7.803320e+03 | rel 3.7e-15 | rel 4.1e-15 |

Neither forcing space uses a mass inverse or a Cholesky factor (section 3).

**History**
- The first version of the fork assembled the operators with scikit-fem, because the sandbox had no DOLFINx.
- That version agreed with DOLFINx to 5e-16 (A) and 3e-15 (M), with identical Dirichlet sets.
- It was removed once DOLFINx was installed here (section 7), so there is now one assembly path, the user's forms.

## 2. The plant: ports, symmetry, positions

`bfs/plant.py`, `BFSPlant(FECoDesign)`, has the interface the GL pipeline uses.

**Ports** (`bc`, `act_field`, `sensor_vec`, `Mdpsi`, `port_jet`): the devices of Barbagallo et al. (2012) and of the user's `actuator_sensor.edp`, with free positions.
- Actuator: `psi_a = (0, g)` with `g = exp(-|x - x_a|^2 / (2 sig^2))`, `sig = 0.5 / sqrt(ln 2) = 0.6006` (half amplitude at distance 0.5, as in the script); peak 1; weak form `b = M psi_a`; free centre `x_a = (x, y)`.
  - The paper forces the vertical momentum; the script forces u ("careful I am forcing on u, not v"). `--bfs-act-comp {v,u}`, default v.
  - The script truncates the Gaussian at radius 2 (an amplitude jump of 0.4 %, 1e-5 of its energy). Not done here, so b is smooth in x_a.
- Sensor: the wall shear stress on the floor y = 0, `y = int chi(x - x_s) du/dy dx`, i.e. `c = S^T (w * chi)`.
  - `S` is the floor's shear trace, exported with the operators (`ops_dolfinx.floor_shear_trace`): six Gauss points per floor facet, `(S q)_k = du/dy (x_k, 0)` from the P2 basis of the facet's cell (local Vandermonde fit, exact for P2 on straight triangles; checked to 4e-13 on a P2 field).
  - `chi` is a window of length 0.6 (the paper's Cp = [11, 11.6] in the text, and C1..C4 in its figure 1) with tanh edges of width 0.05. The script's selector is the P2 interpolant of an indicator, which ramps over one element (0.04 on the fine floor); the tanh edge is the same measurement made differentiable in x_s.
  - Free window start `x_s` (one number). Dirichlet columns of c are zero (as b's rows).
- Both ports are unnormalised (physical); the controller gain absorbs the scale.
- Placement gradients: `Mdpsi` returns d b / d x_a (2 rows) and d c / d x_s (1 row); `grads` uses them in the GL formulas. Finite differences agree to 1e-9 (test plant).
- Reduced-model input basis: `port_jet`, the port vectors and their first and second derivatives in the position (6 for the actuator, 3 for the sensor).
- Positions are per port (`pos_dims = {'a': 2, 's': 1}`; packer `pd_a`, `pd_s`).

**Positions from the paper** (figure 1, calibrated on Cp's stated [11, 11.6]):
- B2 sits on the arc at x = 0.6; its centre is placed at (0.6, 1.0), 0.07 above the wall. The user's script has (0.87, 1.0).
- C1 = [2.0, 2.6], C2 = [3.6, 4.2], C3 = [5.3, 5.9], C4 = [7.3, 7.9], all length 0.6. The script's upstream spy is [2.0, 2.4].
- B1 (the paper's noise) is not used: the noise here is a distributed field.
- The paper's domain is [-20, 100] x [0, 20] (n = 400,775); the user's linear-analysis domain is the truncated [-1.5, 30] x [0, 3.5] of Dergham et al. (2013), same boundary-condition types (no slip on the walls, v = 0 on top, natural outlet; the inlet moves from x = -20 to -1.5).

**Real system** (`real_plant = True`)
- `sigma(-w) = sigma(w)`, and eigenvalues come in conjugate pairs.
- Every frequency search, band and certificate runs on w >= 0, Im >= 0 (`hinf`, `alpha_certified`, the band).
- The packer's `real_ctrl` keeps only the real parts of (d, e, f, n, K) as design variables (`packer.py`).
- Complex arithmetic remains only where it is intrinsic: the resolvent at w != 0.

**Interior constraint** (`position_constraints(x0, side)`, `position_rows`):
- actuator centre: `x_lo + m <= x <= x_hi - m`, `y <= y_hi - m`, and `clearance >= m`, the clearance being the distance to the lower wall's tangent line at x (the true distance where the wall is straight, first order on the arc). The lower wall: the plate y = 1 for x <= 0, the arc of centre (0, -1.5), R = 2.5, for 0 <= x <= 2, y = 0 beyond.
- sensor window: on the floor, `2.0 <= x_s` and `x_s + 0.6 <= x_hi - m`.
- `m = --bfs-margin`, default 0.05 (the paper's devices sit 0.05 to 0.16 from the wall; it was 0.25 for the Gaussian ports).

**How the constraint enters the SQP** (`sqp_inner.py`, `Evaluator._pos_rows`, `subproblem`)
- The rows are hard linear rows `c + g.d <= 0`, with g in z through the packer.
- They are not elastic, because they are linear and the start is feasible.
- They also count in the merit function's violation, in admissibility, and in the Lagrangian gradient.
- GL has no `position_rows`, so nothing changes there.

**Start check:** the driver refuses a start whose centres are not inside.

## 3. Linear algebra and parallelism

**No mass inverse, no Cholesky factor.** `massops.FORBIDDEN = True` makes
any M-factorisation path raise (`massops._entry`).

**The gain**
- `sigma^2 = lambda_max(W)`, `W = M T^H M T`, with `T = (i w M_cl - A_cl)^-1` and the load `M f`.
- W is self-adjoint in the M semi-inner product and touches M only through mat-vecs (`closedloop_sparse`).
- ARPACK uses ncv = 8 here (`closedloop_sparse.GAIN_NCV`; None keeps GL's default). It needs 7-16 operator applications where the default needs 21, and sigma agrees to 1e-11.

**LU: MUMPS through petsc4py** (`bfs/mumps_lu.py`, `sparse_lu.ORDER = 'mumps'`, `--lu-order mumps`)
- Sequential MUMPS on PETSC_COMM_SELF, with KSP preonly.
- PETSc is initialised on COMM_SELF (`petsc4py.init(comm=MPI.COMM_SELF)`). PetscInitialize is collective over its world, and in the task farm each rank initialises at its own first factorisation. On COMM_WORLD the first worker waited forever for the driver; this was found in the sandbox with Open MPI, 2 ranks.
- ICNTL(7) = METIS by default (`BFS_MUMPS_ORDERING`); ICNTL(14) = 50 % (`BFS_MUMPS_MEM`).
- `solve(b, trans)`: 'T' is solveTranspose; 'H' is conj, then T, then conj.
- Sandbox measurements:
  - n = 191,720: factor 6.7-7.9 s, 1.05 GB, solve 0.17 s, residual 1e-13;
  - n = 33,593 (test plant): 1.2 s.

The MKL PARDISO stand-in used before DOLFINx was installed is removed. It
was inaccurate on this saddle-point pencil (1e-8 to 1e-9, with perturbed
pivots that moved the computed eigenvalues).

**The full-model pool as an MPI task farm** (`bfs/mpipool.py`)
- GL forks its pool (`fullpar.py`). Forking after MPI/PETSc initialisation is unsafe, so under an MPI launch (`mpiexec -n N python run_case_epi.py ...`) rank 0 runs the driver and ranks 1..N-1 serve tasks.
- Each worker:
  - rebuilds the plant from the ops file once;
  - registers it with `certify_sparse` (tasks then carry a tag, not A and M);
  - receives a snapshot of the module-level settings (UPPERCASE globals of the listed modules) and the driver's simple plant attributes with every context;
  - runs the same module-level task functions as the fork pool, with its own MUMPS factors.
- Scheduling is dynamic: one task at a time per worker.
- Tested in the sandbox with 3 ranks: bitwise identical to serial.
- `mpipool.active()` reads the launcher's environment (OMPI_COMM_WORLD_SIZE, PMI_SIZE, SLURM_STEP_NUM_TASKS) before importing mpi4py. A GL run therefore never initialises MPI: a singleton MPI_Init outside an allocation aborts on the cluster (`check_env.py` notes).
- Under MPI the driver uses `--nproc 1`. The GL |Omega| pool (`parallel.py`, forked) is never created for the BFS (`a.nproc > 1 and not BFS`).

**The reduced model on the same ranks** (`--row-workers mpi` in `run_bfs.sh`, 2026-10-03; user: use the idle workers)
- Between full-model calls the workers wait while the driver works on the reduced model. Two parts of that work are split over the ranks (`reduced._MPIRowPool`), the driver keeping its own share:
  - **the rows** (`gain_batch`: the SQP's |Omega| rows and sigma_2 blocks, the peak follower, the 1201-point fine sweeps): frequencies in w-blocks of 0.064, a block always to the same rank, so each rank keeps warm subspaces for its frequencies (the fork pool `--row-workers N` of GL, on MPI ranks);
  - **the tracking of the near-line closed-loop eigenpairs** (SISO; one Rayleigh-quotient iteration per tracked eigenvalue, 20 to 40 of them here): pair j to rank j mod size; the merge check is made by the driver on all of them.
- The ranks hold no basis (n x kd): the projection (kd x kd: A~, M~, the Gram roots, the QZ form) is installed on every rank once per basis version (`each`/`collect` in `bfs/mpipool.py`), and a call carries the design's reduced ports and the compensator.
- Results: tracking bitwise equal to serial; rows equal from cold caches, 1e-12 apart with warm ones (the subspace iteration's tolerance; the fork pool behaves the same). SISO and MIMO central, 3 ranks (section 8).
- Without an MPI pool (serial runs, GL) nothing changes.

**The line certificate in segments over the pool** (`--cert_nseg`, default 16 in `run_bfs.sh`)
- Each shift is a full LU here, so the sweep's segments run on the workers (`certify_sparse._seg_worker`).
- The workers now return the eigenvalues their sweeps saw. The continuation gate's watch list, and the argmax used for the eigenvector, are therefore harvested in pooled mode too. Before, they were serial-only; GL with `--cert_nseg > 1` benefits as well.

**The full model on a lattice** (`--oracle lattice`, `ORACLE=lattice` in `run_bfs.sh` and the slurm scripts; `lattice.py`; 2026-10-06, user: drop the reduced model, parallelise the full one)
- *Why a lattice.* The plant factor of `i w M - A` (or `s M - A` on the stability line) does not depend on the design: ports and controller enter by Woodbury. A factor computed once serves every later design, if the same shift comes back. So the working frequencies and the line's disk centres are put on a dyadic lattice, `w = k 2^-l`. A coarse point is a point of every finer level, so refining a bracket or a gap never discards what is cached.
- *Where factors live.* Each rank keeps the factors it computed (`--lattice-cap`, default 16 per rank, least recently used dropped; the closed loop's cache refreshes on a hit, `lu_lru`). The driver knows what each rank holds (every result reports it) and sends a point to a rank holding its factor, unless that rank's batch would then cost more than a factorisation on the least loaded one (relative costs 1 and 3). A step's tasks all go out at once, one batch per rank; the driver computes its own share meanwhile (`--lattice-driver on`).
- *Rows* (per SQP evaluation). A fixed member of Omega is evaluated at its point on 2^-10. A followed peak is placed by **bracket and refine**: from its last row point, walk along the slope (sigma' from the same solves) to a sign change; the cubic Hermite fit on the bracket estimates how far the top lies above the better end; halve the bracket (one finer point) until that is below `--lattice-tol` (1e-6) x sigma. The row is the end point: an exact value and gradient of sigma_1 at a lattice frequency (`t >= sigma_1(w)` holds at any w), at most the tolerance below the top; the fit never enters the QP. The level reached is kept for the next evaluation (one coarser when well inside the tolerance). sigma_2 blocks at the row point. sigma_1 restarts from the last singular vector at the same point with ncv 4 (4-5 applications instead of 9-17; sigma within 1e-13).
- *Stability rows and admissibility* (per SQP evaluation). The line `Re = -eps`, `Im in [0, band]`, covered by eigenvalue-free disks in **waves**: the first wave is the previous call's centres (a cover of the line plus every centre whose nearest eigenvalue lies within the row reach -- an eigenvalue near the line need not be nearest to a centre of a cover); then each gap gets as many centres as the neighbouring radii suggest, on the lattice point nearest each place at the first level finer than a quarter of the spacing; until covered, a witness on or right of the line (not clear), or a gap below 2^-17 (inconclusive, never clear). The proof is the cover, as in `stability_and_abscissa`; where the centres sit changes the cost, not the rigour. The SQP's `clear` is this certificate, every evaluation. Each verified eigenvalue within `--lattice-reach` (0.25) of the line gets its row from its left eigenvector on the same factor (adjoint Arnoldi), the two rightmost their vectors on request (the closed-loop Hermite block). `--lattice-k` nearest eigenvalues per centre (default 1: k = 3 costs 2-4 x the Arnoldi applications).
- *Missed peaks* (after each accepted SQP step): sigma_1 on the lattice level 2^-`--lattice-sweep` (default 5: 321 points on [0, 10]); a local maximum above t (1 + tol) away from every member and top joins Omega and is followed. The certificate's reference sweep uses the same points.
- *The gates* (`D.alpha_certified`, the stability gate, the continuation gate's line checks): the same lattice line, memoised per design, so a candidate the SQP evaluated is not swept again. The ports' positions behind each (b, c) the driver builds are registered (the ranks build b and c from them).
- *Not on the lattice:* the gamma certificate (its 2n pencil carries gamma in a full-rank block, M / gamma^2: no factor serves two levels; an elimination onto the plant factors converges -- sigma_2 of the open-loop resolvent is at most 36 against gamma ~ 174 on the test plant -- but costs 3.5 x more per Arnoldi step than the 2n factor, which is only a quarter to a third of a shift's 100-190 steps); the continuation gate's tracking; the winding count (3400 samples per count, more than any cache, run twice per run: 133 s of critical path in job 762583).
- *Costs measured* (test plant, one core; `tools/check_lattice.py`): a line centre is 25-85 Arnoldi applications (0.6-1.9 s) on top of the factor (1 s), so the lattice saves the factor there, not the Arnoldi (a warm start does not help on the line: 37-154 applications); a sigma_1 row is 9-17 applications cold, 4-5 warm.
- *Memory:* about |Omega| + the line's centres (~25) + the sweep (321) + the followed brackets' points, spread over the ranks. Test plant: 133 MB per factor, so ~900 factors = 120 GB on one node. Full mesh: ~1.05 GB per factor: ~900 GB, i.e. several nodes (`LATTICE_CAP` x ranks x 1.05 GB per job).

## 4. Certification without M^-1

The algebra is in `docs/note_algebra_clean.tex`, section "The backward-facing
step: algebra without M^-1".  What each GL ingredient needed and what the BFS
uses:

| GL ingredient | needed | BFS |
|---|---|---|
| gamma pencil (even, sparse + low rank) | nothing | the same pencil: with the weak-form ports B B^T = C^T C = Q, so no inverse ever appears |
| eigenvalue-free disks of the gamma certificate | nearest eigenvalues; norm-based fallback with M^-1 | nearest eigenvalues only (Arnoldi, converged); the norm fallback is M-free now but useless here, off |
| numerical range (enclosures, heights, tail) | Lanczos with M^-1 | tops of Hermitian pencils on ker B, certified by inertia (`bfs/mfree.py`) |
| winding count tail | M-norms of b and c | an H^1 bound (the wall-shear sensor is not of the form M psi) |
| plant poles of the winding count | enclosure with M^-1 | the same enclosure, M-free; cached per mesh (`bfs/plant_cert.py`) |
| massops | M factor | forbidden (raise) |

**gamma: option A, certified on [0, 10]** (user 2026-10-02: b = 10)
- The level-set certificate of GL (`certify_sparse.hinf_certified_sparse`, Schwerdtner-Mengi-Voigt disks) on [0, b], every round.
- Disks from the converged nearest eigenvalues of the even pencil at each shift i theta. Measured on the test plant: the norm-based free radius (now M-free, below) is 1e-7 to 1e-10 where the nearest eigenvalue is 0.015 to 0.5 away; the pencil is that non-normal. So `certify_sparse.FREE_DISK = False` for the BFS: a set that does not converge is retried with a larger subspace, else the interval is bisected (strict disks).
- Arnoldi: k = 4 nearest (the pencil's eigenvalues come in pairs mirrored across the axis), `NEAR_NCV = 100`, first attempt 60 restarts (`STRICT_MAXITER`). ncv 100 needs 100-190 applications where 30 needed 450-1480, same nearest eigenvalue; a loose tolerance (1e-6) returned a wrong one, so the tolerance stays 1e-9.
- Cost, test plant, serial: 334 s for one certificate (it was 1040 s with ncv 30). The full-size 2n pencil: LU 58 s and 3.9 GB in the sandbox, solve 0.6 s; the first generation of the sweep is split over the workers (`--cert-split 23`).
- `--bfs-sampled N > 0` keeps the sampled maximum (option B) for comparison.
- The certificate's starting value (2026-10-02): the inner problem's t is a reduced-model value; where the model overestimates the full sigma_1, the certificate proved nothing above t (1 + rtol) and returned t, an upper bound no frequency attains (cluster test plant: 1209.568 certified, 1208.919 at the witness, "WITNESS INVARIANT VIOLATED" in 2 of 3 rounds; the record's gamma was safe but 5e-4 loose). For the BFS the certificate now starts from an attained full value (`_true_lower_bound`: the hint and the reduced model's top peaks sampled over the pool, the best climbed by its slope; about 10 full evaluations), so the certified value is attained to rtol (checked: 1645.2109553 attained vs the earlier record's 1645.2109574).

**The free-radius bound without M^-1** (`SparsePencil.free_radius`, `sparse_spectrum.free_radius`)
- E = R^T J R with R never formed: rho((A - sE)^-1 E) <= ||R P^-1 R^T||, whose square is lambda_max(P^-H W P^-1 W), W = R^T R = blockdiag(M_cl, M_cl) semi-definite: Arnoldi, no inverse. For an SPD M it equals the earlier lambda_max(X^H W X, W): identical to 1e-15 on GL, so one code path.

**The numerical range on divergence-free velocities, certified by inertia** (`bfs/mfree.py`)
- Every eigenvalue is a Rayleigh quotient u^H A_vv u / u^H M u with B u = 0 (B the divergence block), so it lies in the numerical range W on ker B. Its bounds are tops of Hermitian pencils (H, M) on ker B.
- sigma M - H is positive definite on ker B iff [[sigma M - H, B^T], [B, -delta I]] has exactly n_p negative pivots (saddle-point inertia; the direction a certificate needs holds for every delta > 0). One sparse LDL^T (qdldl, real arithmetic: MUMPS reports no inertia in complex), doubling and bisection on sigma: every accepted sigma is a proof.
- Complex Hermitian H is embedded as a real symmetric matrix of twice the size.
- Validated on a 1,279-dof mesh against dense generalised eigenproblems on ker B: equal to 1e-7, from above. 3 s per test at n = 191,720 (real).
- Test plant: max Re W <= 1.758, |Im W| <= 87.9, S-lemma heights 26.8 (right of -0.5) to 87.9 (right of -50); the H^1 constant beta = 2.48.
- `sparse_spectrum.plant_rectangle` uses it whenever `massops.FORBIDDEN`; the BFS's depths stop at 50 (the heights reach the bare bound there), and the bare bound closes the list.

**All-frequency stability: the winding count** (`BFSPlant.winding`)
- `winding_cert.winding_count` with the plant's data and a plant-supplied tail bound (`gbound`), `real_sym=True` (the march covers [0, W_T] and doubles: f(conj s) = conj f(s)).
- Tail: |G(s)| <= ||c||_{L^-1} ||psi_a||_M sqrt((1/d + (beta - Re s)_+ / d^2) / nu1), d the numerical-range distance, L the velocity stiffness (exported with the operators), nu1 = nu/2 -- the H^1 estimate (note). It decays like d^-1/2 (the Stokes layer makes that rate genuine), so W_T grows like the square of the feedthrough K; test plant, a feasible start with K = 0.005: W_T = 120.
- Plant data: the plant eigenvalues right of line - 0.6, certified complete by the M-free enclosure (rectangle + tiles exhausted by nearest-k shift-invert; the norm-based empty tile is off for the BFS, `sparse_spectrum.FREE_TILE = False`). Test plant: none right of -0.025, 282 within 0.6 of it (36 min serial, once per mesh).
- Used at the end of every run (record: `wind_count`, `wind_info`) and to check the feasible start (count right of -eps, which must be 0).
- Where the points go (test plant, 2026-10-02): 13 of 3617 beyond w = 30 with W_T = 120; 17 with W_T = 1e3 or 1e4 (the step grows to W_T / 8 once f is smooth). The march's cost is the hot region below w = 30: the plant poles near the line and the loop's zeros, which do not depend on the sensor type. A velocity sensor (c = M psi_s, tail ~ 1/d instead of d^-1/2, W_T ~ |K| instead of K^2) would save a few points per count, nothing more.
- `bfs/plant_cert.py` (`bfs/slurm/plant_cert.slurm`) computes the plant data once per mesh, over the pool, cached next to the operator file.

**Plant spectrum near the axis** (`plant_spectrum`): shift-invert at shifts on [0, band + pad], once, cached for the line certificate's region (test plant: rightmost -0.1148; fine mesh: -0.1149).

**Stability between rounds**: as at GL scale, the line certificate (nothing on Re = -eps over the band) plus continuation (the watch list tracked at every accepted step); the winding count checks the start and the result over all frequencies.

## 5. The driver

`run_case_epi.py --plant bfs --ops FILE` takes everything below. `bfs/run_bfs.sh`
holds the full flag set in one place: the GL production SISO flags plus the BFS
ones.

**Plant and start**
- `--plant bfs`: builds `BFSPlant` with no massops warm-up and no seed catalogue.
- The start comes from `--init` (d, e, f, nvec, K, xa (2), xs (1)), with no rescaling and no warm-start selection.

**Band**
- `[0, --bfs-band]` (default 10), plus the pad.
- `D.band_of` makes the stability certificate follow the run's band.
- The band is widened to the compensator poles only (no plant enclosure).

**Gamma**
- Certified on the band every round (`--bfs-sampled 0`, the default; section 4). `--bfs-sampled N` gives the sampled maximum instead.

**Reduced oracle input basis**
- Built from the ports' second-order jets in their positions (`BFSPlant.port_jet`: 6 actuator vectors, 3 sensor vectors).
- Rebuilt at every round at the round's design, as the atoms' tangent basis is (`_jet_basis`).
- Without it the reduced pencil missed the true near-line pairs, and the full certificate ran every evaluation.
- Union Galerkin step on W's new columns only (`--oracle-union new`, 2026-10-03; GL default `full`): same span, basis equal to rounding (STAGE6 6.30); the basis updates are serial O(n kd^2) work in the driver, 10 s of the round measured below.

**MIMO: two actuators, two sensors** (user 2026-10-02; `M=2 P=2 STRUCT=central|decentral` in `testplant.slurm` / `codesign.slurm`)
- Ports of the same kinds: m vertical Gaussian forces, p wall-shear windows; `BFSPlant.bc` returns B = M Psi_a (n x m), C (n x p) when the positions come as (m, 2) / (p, 1).
- Controller (`closedloop_mimo`, the GL MIMO algebra, M-free): central = one compensator of order R (default 10), B_c (R x p), C_c (m x R), D_c (m x p); decentral = m = p SISO loops of order R/m (5 + 5), sensor k -> actuator k. Real entries (`PackerMIMO` real_ctrl), positions per port (pos_dims), the zero-gain scale floor as the SISO packer's.
- Starts (`hifoo_init.py --m 2 --p 2 --struct ...`, kind zero by default in testplant): actuators at B2 (0.6, 1.0) and (1.6, 0.9), sensors at the paper's C1 (2.0) and C2 (3.6); the compensator HIFOO's draw (decentral: block tridiagonal, B_c the SISO gauge per loop), shifted to -eps - 0.5.
- Gradients (`BFSPlant.grads_mimo`, `_ProxyCLMIMO` with vector positions) checked against central differences on the test plant at w = 0.7: full model 6e-11 to 1e-8 relative for every group (d, e, f, B_c, C_c, D_c, both actuators' (x, y), both windows), central and decentral; reduced model 5e-10 to 8e-7.
- The rest is the SISO path: gamma certified on [0, 10] (the MIMO even pencil), line certificate, continuation gate, certified prefix, winding count (the _FM count with the H^1 tail, operator norms), figures (each port numbered, k_ij Bode entries). Records: `K_B_C_s0_bfs_mimo2x2_<struct>`; directories `_m2p2_<struct>_o10`, so MIMO runs never share a directory with SISO ones.
- No separation constraint between the two actuators (or windows): they may approach or overlap.
- Sandbox check (test plant, serial, 1 round each, `--wind 0`): zero start 1464.37 -> central **1017.97** (alpha -0.0895), decentral **1109.67** (alpha -0.1163) after round 0 (SISO round 0: 1296.62); the records, the figures and the MIMO winding march (plumbing, 30 points) ran. Found on the way: a winding march stopped by the point budget counted as complete in the serial path (`winding_cert`, now not ok).

**Interior margin**
- `--bfs-margin` (section 2).

**SQP penalty: adaptive exact penalty** (`--sqp-pen adaptive` in `run_bfs.sh`, 2026-10-05; GL: `fixed`, unchanged)
- The SQP's merit is max(t, smax) + nu x (stability violations), and its QP pays nu (10 nu for the Hermite and LMI slacks) per unit of elastic slack (Fletcher's S-l1-QP: the slacks keep the QP solvable when the linearised constraints cannot all be met inside the trust region). It is an exact penalty -- its minimisers are the constrained ones -- only if nu exceeds the stability rows' multipliers, and these scale with sigma: GL's fixed nu = 100 is ~10 x sigma there, 0.13 x sigma here.
- So the SQP bought sigma with stability: on the order-10 SISO run (section 8) a step took sigma 752.96 -> 740.74 by moving a compensator pole from -0.0508 to -0.0258, 0.024 across its line, for a penalty of 2.4. From that inadmissible point every step back cost more sigma than the penalty returned: all rejected, the start returned, the same in every round until the progress rule stopped the run. Accepted designs were never unstable (each is certified on the full model; the compensator check is exact), but the solve was lost.
- Rules (Byrd, Nocedal & Waltz 2008; Powell): steering -- while the QP step gives up a linearised stability constraint (an elastic slack > 1e-10) that a larger nu keeps, nu x 10 and solve again (at most 6 times; stopped when the slack no longer drops, i.e. the linearisation is infeasible there); update -- when the step keeps them all, nu >= 1.5 x the largest stability multiplier. nu starts at 100 and never decreases within a solve. Scale-free: no constant tied to sigma.
- (2026-10-04 to 05 the interim fix was `--sqp-pen-rel 0.1`, nu = 0.1 gamma; replaced.)
- **Back to the line (second-order correction)**, with the adaptive penalty: a step along the linearised boundary Re lambda(A_c) = lim leaves it at second order (the boundary is curved in z), and an exact penalty then cuts every step to alpha ~ 1e-3 (the Maratos effect: measured, 752.96 -> 743.07 in 10 iterations). So each trial whose compensator poles crossed the line is corrected before the merit test: the poles at or over the line are put back on it by the least-norm Newton step in the compensator's coordinates (A_c is affine in z; a 10 x 10 eigenproblem, exact). Not for two nearly coincident poles (their gradients blow up): that is the face S.
- **An admissible iterate never steps outside the compensator's line** (2026-10-06, with the adaptive penalty): a trial whose compensator crossed (checked first, r x r, before the plant is evaluated) is put back on the line by the correction above, or, when that fails (a large step: the correction must be under half of it), the step is halved. The constraint is exact and free, so nothing is gained by letting the merit trade it: on the test plant (order 10, zero start, round 1, lattice oracle) a step of 2.2 in z took a pole to +0.17 for 87 of sigma against a penalty of 22, and the solve sat on inadmissible iterates (t 1091 -> 858, smax 973, every iterate NOT admissible) until stopped.
- **Conjugate pairs are not double poles:** the face S was entered whenever the two rightmost compensator poles sat on the line, at any separation (a GL rule: GL's controllers are complex). For the BFS's REAL compensator the two rightmost are often a conjugate pair -- one smooth constraint, its real parts equal by symmetry -- and entering the face for it projected z by 0.49 (pair 3.6 apart) into an unstable iterate. A real compensator's conjugate pair more than 0.1 apart no longer enters the face. A CLOSE conjugate pair still does: it is a double real pole split by rounding -- the central 2x2 result's double pole at -0.05 is computed as -0.050001 +- 5e-8 i (a first version of the rule, refusing every conjugate pair, kept that one off the face too, and the SQP then crawled at alpha ~ 1e-7). This affected every earlier BFS run whose rightmost compensator poles were a distant conjugate pair on the line (the central 2x2 result has one, -0.05 +- 3.13i).

- The compensator constraint keeps GL's form: per-pole rows, the 2x2 Hermite block of the two rightmost poles (Burke-Overton coordinates), and the double-pole face S with its retraction (a pair of poles meeting on the line is held there as a double pole, every step projected back onto the face). Four poles on the line (the central 2x2 result: a double real pole at -0.05 and -0.05 +- 3.13i) are the double pole on its face plus a conjugate pair, which is smooth (per-pole rows). The Lyapunov LMI (`--sqp-lmi`) was tried on GL and fails exactly at near-double poles (cond P ~ 1e7, the QP solver's NumericalError; STAGE1B 7.1), so it stays off.

**A refused step keeps its certified prefix** (`--gate-prefix 1`, 2026-10-02; GL default 0, unchanged)
- The continuation gate certifies the segment from the last accepted design to the candidate piece by piece, left to right (halving up to `--cont-depth`). When it gives up -- a crossing ahead, or the eigenvalue motion still too large at full depth -- the path up to its last certified piece is already proved free of crossings (`continuation_gate(..., reach=)`).
- That point becomes the candidate: its own line certificate (and the rectangle gate) and its gamma certificate; accepted if it improves on the best, with the step cap set to the certified length. The round is no longer lost (cluster test plant, zero start: round 0's step 1464 -> 1140 was refused whole, at depth 4).
- Cost when it fires: one line certificate and one gamma certificate (5 to 10 min on the full mesh), against a whole round (35 to 50 min).
- An empty prefix (the first 1/2^depth piece already refused) is reported ("certified prefix: none"). That was the case in the second cluster run: the zero start's compensator pole 0.006 from the line (fixed in section 6).

**The continuation gate along the SQP's path** (`--gate-path`, auto = on with `--oracle lattice`; 2026-10-06)
- The gate certified the STRAIGHT segment from the last accepted design to the candidate. A round's step can be long (2.4 in z on the test plant, order 10, round 1) and the straight segment can leave the feasible set although every SQP iterate was admissible: "homotopy midpoint is infeasible (alpha -0.0458)", no certified prefix, the round lost.
- With the lattice oracle every SQP iterate's line is certified on the full model, so the natural path is the SQP's own: its accepted iterates in order (`stats['path']`), segment by segment (each a line-search step), the watch list carried along. If it fails, the straight segment is tried; the certified prefix (`--gate-prefix`) is then the path's (along improving iterates) when the straight segment has none.

**Stopping before the time limit**
- `--deadline S` (general; default 0 = off): no new round once elapsed time + 1.25 x the longest round + `--deadline-final` (default 1800 s) would pass S.
- A job time limit then never cuts a round or the final save in half.
- A resubmission resumes from the checkpoint (auto-resume). A deadline stop saves the record (`deadline_stop: true`) and now keeps the checkpoint; before 2026-10-02 the save removed it, so the resubmitted job took the start for finished.

**End of run and record**
- The all-frequency winding count right of -eps/2 (section 4); the final band check is not run.
- The record stores 2-D centres as lists.
- The tag is `K_B_C_s0_bfs` (it was `_ne200`).

**What a run writes** (in its OUTDIR)
- `K_B_C_s0_bfs.npz`: the design (d, e, f, nvec, K, xa, xs), psi_a and the sensor vector c, gamma, alpha, the certificate's active frequencies, and sigma_1 of the closed loop on the band (`omega_grid`, `gain`, from the reduced model, for a quick look).
- `K_B_C_s0_bfs.json`: the summary: gamma, alpha, the band, `all_w` (`certified` on [0, whi] or `sampled`; beyond the band not checked), the feasible start (`stabilise`, its count right of -eps), `wind_count`/`wind_info`, the certificate's disk statistics, the per-round history, timings.
- `K_B_C_s0_bfs_ckpt.npz`: the checkpoint (resume); removed when the run finishes, kept after a deadline stop.
- The log (slurm `logs/`), with a `[ledger]` line per round.
- `K_B_C_s0_bfs_post/` (`bfs/postproc.py`, run by `codesign.slurm` after every saved record, or by hand):
  - `sigma_loglog.png` (2026-10-04): the same curves on log-log axes, the grid with 16 log-spaced points on [1e-3, 0.02], the value at omega = 0 in the title;
  - `sigma.png`: sigma_1, sigma_2, sigma_3 on [0, W], W = the final band edge (at least 10), without control (open loop) and with it, and gamma as a horizontal line;
  - `modes_w<w>.png`: at the open-loop peak, the closed-loop peak and `--omegas` (default 0.3, 1.5), the optimal forcing (right singular vector of sigma_1) and response (left), without and with control, Re u (`--comp v` for v), with the actuator (half-amplitude circle and centre) and the sensor window (a black bar standing on the wall, as in the paper's figure 1; the panel's bottom is the wall) drawn on every panel, and in grey their START positions (dotted circle, and an equal bar under the black one) with an arrow for the actuator's move (the record carries `xa_start`, `xs_start` since 2026-10-02; for older records `--start INIT.npz`); the forcing is zoomed on the step;
  - `controller.png`: the controller's Bode plot (|k(i w)|, phase; the band shaded; its poles listed);
  - `post.npz`: every number and field behind them (singular values, the forcing f and response q of sigma_1 as dof vectors with their coordinates, k(i w)); `post.json`: peaks and timings. `--replot` redraws the figures from `post.npz` alone.
- How the vectors come without M^-1: sigma_k^2 are the eigenvalues of T^H M T M acting on the forcing f (the driver's M T^H M T with its factors cycled, so the eigenvector is the field itself, not its load M f); one LU per frequency serves both loops (Woodbury for the closed loop); k = 3, ARPACK.

## 6. Starts, and the small test plant

**HIFOO draw** (`bfs/hifoo_init.py`)
- The `hifoo` arm of `tools/randstart.py` (STAGE2: the better arm at GL), made real: every controller entry N(0, 1) -- d, e, f, nvec and K.
- Positions (`--positions`): `paper` (default, user 2026-10-02) puts every start at the paper's B2 (0.6, 1.0) and C1 window start 2.0; `random` draws the actuator centre uniformly inside the domain and the sensor window start uniformly on the floor.
- The catalogue is deterministic in (seed, n); the slurm array takes entry `SLURM_ARRAY_TASK_ID`.

**The zero controller** (`--kind zero`, user 2026-10-02; the default `KIND` of both slurm scripts)
- The same draws with the loop open: nvec = 0, K = 0, only A_c kept (shifted stable by `feasible_start`). gamma starts at the open-loop peak exactly (test plant 1463.9), the floor a useful controller must go below; the HIFOO draws start above it (1655.3; after one round 1645.2).
- At k = 0 the gradient in the gains is generically nonzero (dsigma = Re(dk (u^H T b)(c^H T v)), dk = dK + dnvec (sI - A_c)^-1 B_c, any complex value at the peak); those in A_c and the positions vanish (they act only through the loop) and grow with the gains.
- `feasible_start` returns the certified anchor directly (nothing to continue: 38 s on the test plant); the all-frequency count of an open loop is the plant's plus the compensator's, exactly (`BFSPlant.winding`, no march).
- The compensator is shifted to put its rightmost pole at -eps - 0.5 (`ZERO_CTRL_DEPTH`), not at -eps - 2 delta as for the HIFOO draws (2026-10-02). With the loop open its poles are closed-loop eigenvalues: at -0.056 (0.006 from the line) the continuation gate allowed 0.003 of motion per piece, and the first step closing the loop was refused whole -- not even 1/16 of it certified -- in both cluster runs. Now the start's alpha is the plant's, -0.1148 (allowance 0.032).

**Why the GL phase does not carry over**
- GL's phase (`stabilise.py`) descends the closed-loop abscissa from a dense eigensolve.
- No method here gives the abscissa of an unstable design. The line certificate proves only that nothing lies on Re = -eps, and it reports the first witness or the eigenvalues near its shifts.
- Measured on the test plant:
  - a phase driven by the certificate at Re = -0.08 stopped at "alpha = -0.0655";
  - the certificate at -0.05 then found -0.0438 + 0.66i;
  - the round from that start could not reach feasibility (t = inf).

**Feasibility by continuation** (`feasible_start`, run by `--stabilise` with no resume and no checkpoint)
1. The compensator, exactly (r x r):
   - `A_c - s I`, with `s = max(0, alpha(A_c) + eps + 2 delta)`, i.e. the tridiagonal diagonal shifted.
2. The loop gain, `(nvec, K) -> tau (nvec, K)`:
   - At tau = 1e-6 the closed loop is the stable plant plus the compensator. That point is certified (the anchor).
   - tau grows geometrically. Each step is gated like an accepted design of the run: the watch list is tracked onto the new tau (`certify_sparse.continuation_step`); a crossing refuses, and a large motion halves the step.
   - Each step then needs the line certificate at -eps to be clear, with every eigenvalue it sees left of -eps - delta (the cushion, delta = `--stab-delta`).
   - Growth is geometric (x4, then doubling, capped at x16) until the first refusal. After that it is bisection in log tau, down to 5 %, between the last certified tau and the smallest refused one. A large motion shortens the step without counting as a refusal.
3. Result: the largest certified tau <= 1. This is HIFOO's draw, with the compensator shifted and the gain scaled until feasible.
   - Then checked independently of the chain: the all-frequency winding count right of -eps must be 0 (section 4); a contradiction aborts the start.
   - Test plant, hifoo_000 at the Gaussian ports: compensator shifted by 1.18, gain x 0.018 to 0.020, alpha -0.054 to -0.053.
   - Test plant, hifoo_000 at the paper's devices: gain x 0.0111, alpha -0.0531 (11 steps).
   - 16 steps (the first version, which re-grew the step after refusals, took 38).
   - The driver's anchor verified it (alpha = -0.0530).
- The pyGRANSO phase is not used for the BFS. `stabilise.py` is untouched.

**Test plant** (`bfs/mesh_coarse.py`)
- The same domain meshed by a graded mapped rectangle (140 x 26 cells, 3807 vertices), written in the archive's XDMF format (meshio).
- The archive's base flow is interpolated onto its P2/P1 nodes, then assembled by `ops_dolfinx.py`: n = 33,593.
- It is a realistic test plant (same structure, gains close to the fine ones), not a model of the flow.

## 7. Environment, cluster scripts, run time

**Sandbox** (`bfs/sandbox_env.sh`): venv `/opt/fenicsx`, built from source in this session.
- Ubuntu 24.04's complex PETSc 3.19.6 and SLEPc 3.19.2 (MUMPS inside), with Open MPI.
- petsc4py and slepc4py rebuilt against them: Cython 0.29.36, setuptools < 70, and a compiler wrapper that strips Ubuntu's `-Wl,-z,defs` and LTO flags.
- Basix 0.11 rebuilt from source with the venv's nanobind 3.1. The wheel's nanobind ABI (v19) does not match DOLFINx's (v22).
- DOLFINx 0.11 (C++ and Python) built from source.
- Local fixes:
  - DOLFINx's Python passes cffi pointers to nanobind: 4 `int(...)` casts in `fem/forms.py` and `fem/function.py`;
  - a `libpetsc.so` symlink (Ubuntu names it `libpetsc_complex.so`).
- torch 2.6, pyGRANSO 1.2.0 (patched, with osqp and gurobipy), clarabel, meshio, h5py.
- Check: the archive's `resolvent_gains.py` on 4 ranks reproduces `gains_fx.txt` to 1e-14.

**Cluster** (env.sh: conda `codesign`, DOLFINx 0.11, complex PETSc 3.25.3, SLEPc 3.25.1)
- `bfs/setup_cluster.sh`, once, on the login node:
  - installs clarabel (the SQP's solver; `check_env.py` never checked it) and qdldl (the inertia certificates) if they are missing, meshio and h5py for the test plant, and matplotlib for the figures;
  - then imports everything the fork needs.
- `bfs/slurm/smoke.slurm` (1 node, 3 h limit; expected 1 to 1.5 h):
  1. the operators (always rebuilt: the file now carries the shear trace and the stiffness);
  2. the gains against the archive;
  3. `bench_mumps.py` (96 ranks factor and solve at once, which is the per-task cost under full-node contention);
  4. the test plant's certified range and poles (`plant_cert.py`, 24 ranks);
  5. one start end to end on the test plant, 24 ranks: feasible start and its count, one round with gamma certified on [0, 10], the all-frequency count at the end.
  6. its figures (`postproc.py`, 24 ranks).
- `bfs/slurm/plant_cert.slurm` (once per mesh, 48 ranks, under 8 h): the full mesh's certified range and poles, cached next to `ops_fx.npz`.
- `bfs/slurm/testplant.slurm` (24 ranks, 1 h 30 limit): one start on the test plant on the production path (several with `--array`: task i takes entry i of a catalogue of `NSTART` draws, default 64, deterministic in `SEED`, into `testplant_bfs/<KIND><suffix>_s<SEED>_r<OUTER>_i<iii>`; without `--array` the names are unchanged), `KIND` (default zero) and `OUTER` (default 3) rounds, then the figures and a one-line verdict (gamma per round against the open-loop peak). Makes the test plant's operators and plant data first if missing. Expected 35 to 45 min (serial sandbox timings: start-up 1422 s with the HIFOO continuation, round 0 587 s of which 334 s is the gamma certificate, which splits over the workers).
- `bfs/slurm/codesign.slurm`:
  - an array job, one start per task: by default (since 2026-10-03) the zero controller, HIFOO's compensator draw with the loop open (`KIND=hifoo` for the scaled draws), at the paper's B2 and C1; MIMO with `M=2 P=2 STRUCT=central|decentral` (directories `<KIND>_m2p2_<STRUCT>_o10_XXX`, so SISO and MIMO runs never share one);
  - 24 ranks per start (the driver and 23 workers) and 240 GB: two starts per 96-core node (the gamma certificate's 2n pencil factor is 3.9 GB in every worker);
  - 15 h limit; `--deadline` from the job's own end time (`bfs/slurm_remaining.py`, scontrol's EndTime, so any `--time` works): 45 min before it, and with the driver's 30 min reserve the last round ends at least 75 min before the end, for the final certificate, the count, the save and the figures;
  - then the figures (`bfs/postproc.py`), made by the driver on its own pool (`--post 1`), also after a deadline stop (they then show the progress); a separate, time-bounded launch only if they are missing; a finished start re-submitted only draws missing figures;
  - resubmitting resumes unfinished starts (a deadline stop or a killed job: from the checkpoint written at every accepted round); `CONTINUE=1` gives finished starts OUTER more rounds from their records (`--continue-record`, the finished summary kept as `K_B_C_s0_bfs_partN.json`); it refuses to start without the plant certificate.
- The controller order is `R` (default 5) in `run_bfs.sh`, `codesign.slurm` and `testplant.slurm` (the starts are drawn at that order; directories carry `_o<R>` when it is not 5). Per round the order costs little: the controller enters the full model as a rank-2 correction whatever r, and only the SQP's variables grow (22 -> 42 at r = 10); the number of rounds to converge is what may grow. Order 10 on the test plant: section 8 (job 16894514).
- `testplant.slurm` with `RESUME=1` (or `CONTINUE=1`) `OUT=<its directory>` does the same for a test-plant run. A directory holding a run is never deleted (2026-10-06: `CONTINUE=1`, the word of `codesign.slurm`, was not read by `testplant.slurm`, and the 44-round central 2x2 run was started over): without RESUME/CONTINUE the job refuses; `FRESH=1` starts over and keeps the old directory as `OUT.old-<date>`. Its deadline also comes from the job's end (5 min kept for the figures, `--deadline-final 600`): `OUTER=300 sbatch --time=15:00:00 bfs/slurm/testplant.slurm` runs as many rounds as fit, or until progress stops. Before 2026-10-02 it had no deadline: a run that did not stop on progress would have been killed at the 1 h 30 limit (with its checkpoint saved).
- The launcher is the conda env's `mpiexec` (`LAUNCHER=` to change).
- `ORACLE=lattice` (both scripts, through `run_bfs.sh`; section 3): the inner loop on the full model. Each rank then holds up to `LATTICE_CAP` (default 16) plant factors besides its usual work (the gamma certificate keeps two 2n factors), so the job's memory is about ranks x (1 + LATTICE_CAP x factor) GB: test plant (133 MB a factor) `--ntasks=96 --mem=450G` on a 512 GB node; full mesh (~1.05 GB a factor, 3.9 GB per 2n factor) ~25 GB a rank at cap 16, i.e. ~20 ranks per 512 GB node and several nodes. Directories get `_lat`, so lattice and reduced runs never share one.
- Every slurm script first runs `bfs/version_check.py`: `bfs/MANIFEST` holds the sha256 of the 63 code files the runs execute (top level and `bfs/`, *.py, *.sh, *.slurm), written with each package; a checkout mixing versions stops before computing, with the files named (`ALLOW_MODIFIED=1` runs anyway). Added 2026-10-02 after a cluster copy ran the version-j `testplant.slurm` with the first package's `hifoo_init.py` (no `--kind`). Site settings are not checked (2026-10-04, after a change of cluster stopped the jobs): `env.sh` is not listed, and the slurm scripts' `#SBATCH` lines are not hashed, so a new partition, account or module set needs no `ALLOW_MODIFIED`.

**Run time** (measured 2026-10-02; projected to the cluster)

*Ledger.* Each BFS round prints a `[ledger]` line (and, since 2026-10-05, a second one with the pool's work by task function: calls, tasks, task time, and the time on 95 workers -- at that size the sum of each call's longest task, i.e. the critical path that more workers do not shorten):
- wall time;
- the driver's own MUMPS factorisations and solves (`bfs/mumps_lu.STATS`);
- the pool's task time, with what it would take on 23 and on 95 workers (`fullpar.CALLS` / `estimate(W)`: per call, `max(work / W, longest task)`).

*Measured on the test plant, Gaussian ports, sampled gamma* (MPI, 2 ranks, one core; start hifoo_000):

| phase | pool task time | on 23 / 95 workers | driver MUMPS | driver other |
|---|---|---|---|---|
| plant spectrum + feasible start (15 steps) | 916 s | 167 / 167 s | 8 s | ~10 s |
| anchor + reduced-model set-up | 175 s | 22 / 20 s | 13 s | ~80 s |
| round 0 (gamma 1517.7 -> 1448.8) | 702 s | 103 / 103 s | 62 s (56 factors, 2153 solves) | ~62 s |

The pool's calls hold about 14 tasks on average (407 tasks in 30 calls), so
23 workers are as fast as 95.

*Certified gamma* (test plant, paper devices, serial): one certificate on
[0, 10] costs 334 s; a round goes from 241 s (sampled) to 587 s (certified),
the start-up from 150 s to 710 s (the start's certificate).

*Per-operation costs* (`bench_mumps.py`; the cluster numbers are from the
smoke test, 96 ranks at once on one node):

| | sandbox, idle | cluster, 96 at once |
|---|---|---|
| n LU, n = 191,720 | 7.85 s | 4.5 s |
| n solve | 0.17 s | 0.20 s |
| 2n LU of the gamma pencil | 58 s, 3.9 GB | (~35 s expected) |
| 2n solve | 0.6 s | |
| n LU, test plant (n = 33,593) | 0.57 s | |

*Projection to n = 191,720, per start, on the cluster:*
- **Plant certificate** (once per mesh): about 2 h on 48 ranks.
- **Start-up:** plant spectrum, feasible start, its winding count, the anchor and the start's certificate: about 1 to 1.5 h.
- **One round:** 30 to 40 min as before (the cluster's faster factorisation offsets the slower solves under load) plus the certificate, 5 to 10 min over 23 workers: **35 to 50 min**.
- **End:** the final certificate and the all-frequency count, 30 to 50 min.
- **Figures** (`postproc.py`): test plant 245 frequencies x 1.7 s (both loops, k = 3, one LU each); full mesh about 24 s each, so about 100 min of task time, **5 to 10 min** over 23 workers.
- **In the 15 h job:** about **15 to 20 rounds**. GL K+B+C needed 20 to 40 rounds to converge; a second job (resume from the checkpoint) may be needed.
- Re-projected from the cluster test plant (2026-10-02, zero start, 5 rounds of 154 to 255 s: pool 58 to 95 s, driver MUMPS 45 to 57 s, the rest 50 to 100 s; full-mesh factor x 11, solve x 14): 25 to 50 min a round, start-up ~25 min, end ~30 min, so **17 to 30 rounds** in 15 h. OUTER = 40 with the deadline: no need to choose.
- **Throughput:** 2 starts per node; 8 starts on 4 nodes.
- **Several starts on one node** (second cluster, 2026-10-04): the scheduler shares nodes -- the three test-plant jobs of 2026-10-04 (24 tasks, 60 GB each) ran at once on one node (n1200). Packing is the scheduler's: each job asks for its tasks and memory, and jobs that fit share a node. Test plant: 37-38 GB used by a 24-rank start (1.5 GB a rank), so on a 96-core / 512 GB node 4 starts at 24 ranks, on a 48-core / 192 GB node 2, on a 44-core node 1 (or 2 at `--ntasks=22`). Fewer ranks per start (`sbatch --ntasks=12 --mem=30G ...`, nothing else to change) doubles the starts per node; the `[ledger]` now projects the pool's work on 11 workers too, to see what that costs a round. Full mesh: 240 GB per start, Sapphire Rapids nodes only (2 per node).

## 8. Validation log

- Frozen path (GL case B, ne = 100) is bit-identical, 10.23128516441147, after:
  - the DOLFINx/MUMPS switch;
  - the SQP position rows;
  - the deadline;
  - the pooled-sweep harvest;
  - the mpipool environment test;
  - the record's `all_w` label and the checkpoint kept after a deadline stop (2026-10-02);
  - the certificate's attained lower bound (BFS only), the certified prefix (`--gate-prefix`, default 0) and the start positions in the BFS record (2026-10-02).
- Operators and gains: section 1.
- MPI task farm, 3 ranks: identical to serial.
- Test plant, MUMPS, serial (1 core), from `bfs/data/init_test.npz` (a stable hand-made start; file removed with the Gaussian ports):
  - plant spectrum 82 s;
  - round 0: gamma 2240.09 -> 1413.73 (about 10 min);
  - round 1: gamma -> 1398.64 (about 6 min).
- Test plant, HIFOO start hifoo_000:
  - serial: feasible start (gain x 0.0185), round 0 gamma 1520.36 -> 1447.08;
  - with the first homotopy (gain x 0.0201): 1449.65, then 1439.61 after round 1 (53 min on one core);
  - MPI, 2 ranks: feasible start (gain x 0.0194), round 0 1517.68 -> 1448.82, identical in two runs.
- MPI task-farm deadlock found and fixed: PETSc is now initialised on COMM_SELF (section 3).

*Paper devices and M-free certification (2026-10-02; test plant, n = 33,593)*
- Ports: placement derivatives db/dx_a (2 rows) and dc/dx_s against finite differences, 1e-9.
- Inertia bounds (`bfs/mfree.py`) on a 1,279-dof mesh against dense generalised eigenproblems on ker B: equal to 1e-7, from above.
- Free radius M-free vs the M-based one on GL: equal to 1e-15. On the step: 1e-7 to 1e-10 where the nearest eigenvalue is 0.015 to 0.5 away (switched off).
- Plant data: max Re W <= 1.758, beta = 2.4785, heights 26.76 to 87.88 (2447 s serial); plant poles right of -0.025: 0, hot 282 (2166 s serial).
- HIFOO start hifoo_000 at the paper's B2 (0.6, 1.0) and C1 (2.0): compensator shifted by 1.1847, gain x 0.01109 (11 steps, 4 refused), alpha -0.0531.
- One round with gamma certified on [0, 10]: 1655.32 -> 1645.2110 (2009 s serial); the certificate at that design: 1645.211708, equal to the sampled maximum, every disk converged, nothing retried or bisected (334 s; 1040 s with ncv 30).
- All-frequency winding count right of -0.025 at that design (alpha = -0.0501): **0** (winding -4.6e-15), W_T = 120, 3617 points on [0, 120] (real symmetry), 2161 s serial, no hints.
- The same count pooled (MPI, 2 ranks: 8 segments on the workers, the path of every production run): **0** (winding 8.2e-16), 3374 points, 2013 s of task time; the ledger's projection on 23 workers is 261 s (the longest segment).

*Post-processing (2026-10-02; test plant, the design above)*
- sigma_1 with control equals the driver's `gain` to 1e-10 at w = 0, 0.69, 5; at the certificate's peak 1645.2081 vs the record's certified 1645.2110 (the grid's 245 points, active frequencies included).
- **Without control the peak is lower: 1463.87 at w = 0.74.** The HIFOO start (a random controller scaled to feasibility, gamma 1655.3) and its first round (1645.2) amplify more than no controller at all, everywhere below w = 1 (at w = 0: 121.8 vs 51.8). See section 9.
- Forcing at the peak: concentrated on the step's corner and shear layer, under the actuator; response: the shear-layer wavepacket from x = 5 to the outlet at x = 30 (not decayed there: the truncated domain).
- 425 s for the curves, 7 s for the vectors (MPI, 2 ranks).

*Cluster, test plant, zero start, 3 rounds (job 16879784, 24 ranks, 2026-10-02)*
- Plant data (range + poles at -0.05 and -0.025) 14 min; the run 17 min; the figures 70 s; 33 min in all.
- Start: gamma 1464.37 (= the open-loop peak), alpha -0.0560, count 0 exactly (no march).
- Round 0: the SQP step (to 1140.0) was refused by the continuation gate (split depth exhausted: the first closing of the loop moves the spectrum too far to track); the tau governor capped the next steps. Since fixed: the certified prefix of the step is tried (section 5). Rounds 1 and 2 accepted: 1367.68, 1209.57. Final **1209.57 = 0.826 x the open-loop peak**, alpha -0.0836, two active peaks (a flattened top near w = 0.73 to 0.91), winding count 0 right of -0.025 (3580 points, 94 s over 23 workers).
- Positions: actuator (0.6, 1.0) -> (1.19, 1.12), onto the shear layer past the arc; sensor window 2.0 -> 2.12.
- Controller: one compensator pole at -0.05, on the constraint alpha(A_c) <= -eps; |k| rises towards w = 0 (163 vs 52 open loop at w = 0, below gamma).
- Not converged: -11.6 % in the last round.
- Found: the certified value was the reduced t, not attained (fixed above); the Bode plot's axis ran to 1e-15 for a real pole with Im ~ 1e-16 (fixed); the job's verdict printed candidates, not the accepted values (fixed).

*Cluster, the same run after those fixes (job 16887057, 12 min for the run)*
- Round 0 refused again: the certified prefix was EMPTY (4 splits, all on the leftmost piece; the gate gave up on the first 1/16). Cause: the zero start's compensator pole at -0.056, 0.006 from the line (fixed in section 6: -0.55).
- Round 1 accepted the same design as before (1367.6621; the first run's 1367.6776 was the unattained reduced t, the witness then gave 1367.6617). Round 2 differs: 1244.845 against 1209.568, ratio 0.850 against 0.826.
- Why: the certificate now returns an attained value and its own witness; the witness and the certified level feed the next round's working set and the reduced model's anchors (round 2: |Omega| 42 vs 41, different disagreement anchors), so the SQP solved a different round-2 problem. Three rounds of a nonsmooth local method are path dependent; neither run had converged (-11.6 % and -9.0 % in their last round). The comparison that matters is the converged value (OUTER = 20).

*Cluster, zero start with the compensator at -0.55, 5 rounds (job 16887198, 19 min for the run)*
- Start alpha -0.1148 (the plant's); round 0 ACCEPTED: 1464.37 -> 1296.62.
- Best after each round: 1296.62, 1147.23, 1057.35, 989.95, **960.36 = 0.656 x the open-loop peak**; still -3.0 % in the last round. alpha -0.1016, winding count 0 (3296 points, 108 s). Actuator (0.6, 1.0) -> (1.34, 1.33), sensor window 2.0 -> 2.48.
- Rounds 172, 154, 195, 255, ~200 s; they grow with |Omega| (21 -> 72) and the inner calls (24 -> 107).
- Restart, tested in the sandbox on this start (2026-10-02, MPI 2 ranks): (A) `--deadline` stopped it after round 0 (1296.6185, `deadline_stop`, the checkpoint kept); (B) the same command again resumed from the checkpoint -- gamma 1296.618509 restored, |Omega| 25, the inner budget, the SQP trust radius and 2 peak members restored, the anchor verified -- and its round reached 1145.98; the finished run removed the checkpoint; (C) `--continue-record 1` on a copy with the checkpoint deleted (a finished run) made the record the checkpoint, kept the summary as `_part1.json` and resumed the same way. Found on the way: the BFS record's alpha_dense is nan, so the guard that keeps a better previous record called a stable record "FAILS re-verification"; it now uses the record's certified alpha and winding count.
- Reduced vs full at the certificate's witness: 3.3e-7, 8.5e-4, 3.5e-2, 2.2e-2, 1.9e-5. The large ones are where the design moved far (positions) and the witness sat between the working-set frequencies ("NOT a peak member"); certified / predicted t was then 1.004 and 1.017, the step still accepted, and the witness becomes an anchor. The certified gamma never depends on the reduced model.

*Cluster, zero start, OUTER = 50 (job 16887217, version j; 67 min in all)*
- Stopped by the progress rule after 21 rounds (0.042 % over the last 3 < 0.15 %), at **769.573 = 0.526 x the open-loop peak**; alpha -0.0877, winding count 0 (2780 points, 95 s).
- Best after each round: 1296.6, 1147.2, 1057.4, 990.0, 960.4, 940.9, 940.9 (round 6 refused: a homotopy midpoint at alpha -0.0298, a real approach to the margin; no certified prefix), 913.0, 870.9, 847.0, 842.6, 831.1, 831.1, 831.1 (rounds 12-13: the SQP returned the same design; the working set then grew by 21 and the descent resumed), 824.9, 783.0, 775.1, 769.9, 769.7, 769.6, 769.6.
- Devices: actuator (0.6, 1.0) -> (2.18, 0.91), past the step over the floor; sensor window 2.0 -> 3.41 (the paper's C2 is [3.6, 4.2]).
- Rounds 129 to 294 s, mean 184 s; |Omega| 17 -> 213. Start-up 107 s, end 197 s (the last round, the final count), figures 48 s.
- Reduced vs full at the witness: below 1e-3 from round 8 on except two rounds (1.6e-2, 8.7e-4 ...); the rounds that returned the same design had 1e-13.
- The progress rule (3 rounds below 0.15 %) stopped it after a stall like that of rounds 12-13; continuing it (`RESUME=1`) shows whether more descent is there.

*Cluster, zero start, ORDER 10, OUTER = 300 (job 16894514, version o)*
- Best after each round: 1223.07, 1166.52, 1107.21, 1030.07, 972.64, 954.73, 931.27, 892.13, 868.27; round 9 (813.18) refused by the gate at depth 4, its certified prefix (0.938 of the step) at 928.26, not better; rounds 10-11 refused by their certificates, 1015.96 / 1004.41 against the model's t = 776.07 / 820.49 (a peak near w = 0.27 the reduced model missed by 24 % / 19 %). The progress rule then stopped the run at **868.27** (order 5: 769.57). Not Clarke stationarity, not convergence: a stop on three rounds the model had misjudged. Since fixed: such rounds no longer count towards the progress stop (at most 2 k rounds without an accepted design).
- The record was saved; the figures' separate `mpiexec` launch then failed in SLURM (`srun: Unable to confirm allocation ... Socket timed out`) and hung, holding the job. Since fixed: the driver makes the figures itself on its pool (`--post 1`, `postproc.run_post`), a second launch happens only if they are missing and is bounded (`timeout 3600`); `bfs/slurm/postproc.slurm` makes them for any record.
- Time per round, from the ledger: pool 49 %, driver MUMPS 21 %, the driver's own Python (reduced model, SQP over |Omega|) 30 %. The pool's calls hold ~9 tasks on average (7 to 11), so 23 workers are busy 25 to 37 % of the time; the ledger's 95-worker estimate is no faster. More ranks would not shorten a round; about 12 would barely lengthen it (to be measured), which would let 4 starts share a node instead of 2.

*The driver's own work, profiled; the reduced model on the ranks (2026-10-03, sandbox, MPI 2 ranks, zero start, cProfile on rank 0)*
- Round 0: 707 s wall = pool 603 s + driver MUMPS 34 s + the driver's Python ~70 s. The Python is many serial pieces, none dominant:
  - near-line eigenpairs of the reduced pencil (`near_line_pairs`, 24 SQP evaluations): 18.4 s -- the tracking iterations 8.2 s (about 190 tracked eigenvalues per call: reach 1 to the left of the line holds many weakly damped modes here), dense eigs 8.4 s, Ritz residuals 4.0 s;
  - rows (`gain_batch` and single-frequency gains): ~10 s at |Omega| = 21, growing with |Omega| (8.8 s of it the start's 1201-point peak re-derivation, once per run);
  - the basis updates 9.8 s and the projection 7.0 s (enrichment), the continuation gate 7 s;
  - start-up only: the inertia certificates (`mfree`) ~30 s.
- So the 30 % cannot all go to the workers. The rows and the tracking now do (section 3: `--row-workers mpi`), and the union step projects only the new columns (`--oracle-union new`, section 5). Expected on 24 ranks: ~20 s of the round's Python down to ~2 s, plus ~40 % of the basis updates; the cluster's `[ledger]` lines report the pooled rows and pairs.
- Checks (`rowpool_test`, 3 ranks, SISO and MIMO central): rows bitwise equal to serial from cold caches, 1e-12 with warm ones (gradients' sigma' 2e-13); tracking bitwise equal. The tracking split (`_track_each` + `_track_merge`) is bitwise the committed `_track_eigs` (6 pencils with the SISO coupling structure, a merged pair among them). GL frozen path 10.23128516441147.
- SISO on the production script (`run_bfs.sh`, all of the above, 2 ranks): start 1464.37, round 0 -> **1296.618347** (1296.618396 before: the union step's rounding), winding count 0 (3534 points), figures by the driver; ledger: 5126 rows (2571 by the driver) in 33 calls, 4067 tracked pairs in 21 calls, projection installed 5 times.
- **MIMO resume was broken, fixed:** `_mimo_setup` compared the checkpoint's gamma with the start's, and a start with gamma nan (older MIMO starts) made the comparison False, so `RESUME` / `CONTINUE` restarted MIMO runs from their start -- and skipped the feasible start, since a checkpoint existed (alpha -0.0025, infeasible). A non-finite start gamma now never beats the checkpoint (SISO was not affected).
- MIMO central on the production script after the fix (`--continue-record` of the sandbox's 1-round record 1017.97, 2 ranks): resumed from it (alpha -0.0895, anchor verified), round -> **904.48** (|Omega| 41, model t 916.92), the end-of-run winding count with the controller (a march, not the open loop's exact count): 0 right of -0.025 (3568 points, tail from |w| = 240), the MIMO figures by the driver; 6439 rows (3277 by the driver) in 111 calls. A first attempt alongside the SISO run was killed by the sandbox's 8 GB memory limit (240 GB per start on the cluster).

*Second cluster, test plant, order 10, zero starts (2026-10-04, jobs 760989 / 761011 / 761012, version 20261004a)*
- SISO: 1464.18 -> **752.96** in 9 rounds, then 3 rounds returning the same design and the progress stop (order 5 on the first cluster: 769.57). MIMO decentral (2 loops of 5): 1463.87 -> **808.39** in 4 rounds, then the same stall. MIMO central (order 10): still running, 221.38 after 28 rounds (0.15 of the open-loop peak).
- The stalls are not stationarity: every stalled round ended on the SQP's iteration budget (code 4), its stationarity 6 to 117, the reduced model still finding descent; the design handed back was the start.
- Cause, reproduced in the sandbox from the SISO record with `--sqp-verbose` and the line-search trace: the SQP's first step took sigma 752.96 -> 740.74 with a compensator pole moving from -0.0508 to -0.0258 (its line is -0.05); the merit's penalty at nu = 100 was 2.4 against a gain of 12, so the step was accepted; from that inadmissible iterate every step back cost more sigma than the penalty returned (all rejected), so no admissible candidate but the start -- in every round. The order-10 SISO design has three compensator poles at the line (-0.0528, -0.0508 +- 1.78i), the decentral two (-0.055 +- 0.001i, -0.0515). The penalty was GL's absolute 100, ~10 x sigma there, 0.13 x sigma here: the stall is a scaling error, not the start.
- Fix: the penalty relative to gamma (`--sqp-pen-rel 0.1`, since replaced by the adaptive penalty, section 5). Same records, same sandbox (2 ranks, `--continue-record`), the SQP now riding the compensator line (alc -0.0500 +- 1e-4):
  - SISO order 10: 752.96 -> 735.49 -> 716.84 accepted; round 2 certified 685.08, refused by the continuation gate (a homotopy midpoint had a compensator pole at -0.026: the straight path between two admissible designs left the set), so the tau governor capped the next step -- the gate doing its job;
  - MIMO decentral: 808.39 -> 773.43 in the first round (10 SQP iterations, all but one admissible).
- Every earlier BFS run had the absolute penalty (order 5's 769.57 included).

*Second cluster, MIMO central, order 10, zero start (job 761011, version 20261003a, finished 2026-10-05)*
- 1464.18 -> **191.11** (0.131 of the open-loop peak) in 44 rounds, 12.5 h; stopped by the progress rule. Winding count 0, alpha -0.0723. Actuators (2.92, 0.10) and (4.87, 0.44), sensor windows at 4.50 and 6.94.
- sigma_1 with control is FLAT at gamma from omega = 0 to 2.4 (log-log figure): a plateau, not isolated peaks -- the max over a continuum of active frequencies; the 8 local maxima on the grid (0, 0.10, 0.42, 0.79, 1.13, 1.31, 1.79, 2.35) are ripples on it. At omega = 0, sigma_1 is 191.1 against 51.8 open loop. |Omega| = 474, all on [0, 2.87].
- 13 of 44 rounds accepted nothing: the reduced model under-predicted a peak at the SQP's design (by 2 to 17 %: e.g. round 29, 206.8 predicted at w = 2.59, 249.5 certified) or the continuation gate refused the step. The last four rounds moved gamma by 1e-6: the compensator has four poles ON its line (a double real pole at -0.05 and -0.05 +- 3.13i), the run had the absolute SQP penalty (section 5), and its SQP ended on codes 12 / 6 / 4.
- Time: rounds grew from ~5 min to 20-30 min (pool 61 % of the run, driver MUMPS 8 %, driver Python 31 % of which the pooled rows 11 %); the pool's work per round grew tenfold (to be split by phase: the ledger has only totals).

*Adaptive penalty, correction, conjugate pairs (2026-10-05, sandbox, the SISO order-10 record 752.96, 2 ranks, `--continue-record`, `--sqp-verbose`)*
- Adaptive penalty alone: steering took nu from 100 to 1e5-1e7 on the first QP (the slack fell 2.5e-2 -> 7e-6 -> 2e-7 -> 7e-10); every iterate admissible, but the line search cut the steps to alpha ~ 1e-3 on the curved line: 752.96 -> 743.07 in 10 iterations.
- With the correction: the face S was entered for the conjugate pair -0.05 +- 1.8i (projection 0.49 in z, iterate unstable, sigma 4e3) -- the bug of section 5; with that fixed: all iterates admissible, 3 corrections per round, steps at alpha 1 to 1/8: rounds **752.96 -> 732.09 -> 702.07** (the interim nu = 0.1 gamma: 735.49, 716.84). Round 0 refused nothing; its witness was 0.4 % above the model (w = 0.251, not a member), still accepted.

*Spurious unstable Ritz values (2026-10-05, the central 2x2 result, test plant)*
- Continued with the fixes above, the central run's SQP saw the closed loop UNSTABLE at its own start (reduced abscissa +0.0431; certified -0.0723) and rejected every step: the reduced pencil (dimension 192-264, built as the driver builds it) has Ritz values at +0.03 to +0.11, Im -0.6 to -2.35, whose full residuals (4e-4 to 2e-3) pass the SQP's filter of 1e-3. With the full model's 8 near-line eigenvectors in the basis (`enrich_eig`), those 8 have residuals 1e-10 to 1e-15 and the spurious ones keep 4e-4 to 1.6e-3: a gap of six orders.
- Why: sigma_1 = 191 flat on [0, 2.4] means the resolvent is large all along that segment of the axis, so the pseudospectrum bulges across it; a Galerkin projection puts Ritz values anywhere in the field of values, and a small relative residual does not certify an eigenvalue there. The last rounds of the central run (SQP codes 12 and 6) fit this.
- A conjugation-closed real basis (the note's 10.4, not implemented) does not remove them (real spurious values at +1.4 to +9.2, a pair at -0.0225 +- 1.49i with 4.9e-4).
- Fix: `--sqp-row-res-tol 1e-5` in `run_bfs.sh` (GL keeps 1e-3).
- The central record continued with everything above (2 ranks): the SQP saw alpha -0.0723 (the true value), entered the face S for the double real pole (computed -0.050001 +- 5e-8 i), kept the conjugate pair -0.05 +- 3.13i on its line by 16 corrections, every iterate admissible: **191.11 -> 187.93** certified in the first round (model 187.926907, certified 187.926902), where the original run had stopped on four rounds of 1e-6.

*Reduced model away from its design (2026-10-05, sandbox, test plant, the central 2x2 result as design A)*
- The model built as the driver builds it at A (port jets at A, anchors at A's 8 peaks, greedy to the full model at A): dimension 480. Evaluated against the full model on 118 frequencies at designs moved from A (error relative to the peak):

  | design | full peak | error |
  |---|---|---|
  | A | 191.1 | 2.9e-2 (at w = 4.6, far below the peak; the greedy stopped at 12 anchors) |
  | ports moved 0.1 (actuators also 0.03 in y) | **2795.8** (at w = 0) | 3.6e-3 |
  | controller x (1 + 0.03 N(0,1)) | 283.6 | 1.9e-2 |
  | ports moved 0.3 | 1288.7 | 3.9e-2 |
  | controller x (1 + 0.2 N(0,1)) | 436.2 | 1.3e-2 |
  | ports 0.8, controller 30 % | 764.2 | **2.2e-1** |

- Two findings. (1) The optimum is extremely sharp: moving the ports by 0.1 multiplies sigma at w = 0 by 15 (the compensator's poles sit at -0.05, near integrators, and the low-frequency gain relies on a near-cancellation). (2) The model is good to 0.4-4 % for moves up to 0.3, 22 % for large moves. Near the optimum each round gains < 1-2 % while sigma is flat at gamma on [0, 2.4]: an error of 1-2 % anywhere on the plateau is enough for the certificate to refuse -- the 13 refused rounds of the central run.
- Port-range input bases (the actuator's Gaussian over x in [-1, 8], y in [0.1, 2.3]; the sensor window over [2, 10]): numerical rank 56 / 87 / 117 / 169 at 1e-2 / 1e-3 / 1e-4 / 1e-6 for the actuator (214 positions), 67 / 76 / 77 / 77 for the sensor (81 windows) -- they barely compress (sharp window edges, a 2-D range). Added at every anchor they give 1000-3000 basis vectors. Adding only the jets at a second set of positions (dimension 1186) produced a spurious near-axis reduced pole (sigma 3e7 at w = 3): a Galerkin projection of this non-normal operator does not preserve stability, and grows that risk with the dimension. Not pursued as is.
- Full-model costs measured for the alternative (test plant, MUMPS, one core): LU 0.97 s and 133 MB, one solve 14 ms (10 right-hand sides 114 ms). The plant factor at iw does not depend on the design (ports and controller enter by Woodbury), so factors on Omega can be cached for a whole run.

*Central 2x2 continued on 96 ranks (job 762583, version 20261006a, 2026-10-06)*
- 191.11 -> 187.93 -> 184.29 -> (2 refused) -> **173.77** (0.119 of the open-loop peak) in 5 rounds, 2 h 49; winding count 0; actuator 1 now on the wall margin (y = 0.050).
- Per round: wall 1130 / 1330 / 2470 / 2610 / ~2200 s; pool 70 / 560 / 780 / 910 / ~1100 s (critical path: line-certificate segments `_seg_worker` 2759 s of the run, gamma certificate `_shift_task` 521 s, winding 133 s; sigma samples 18 s); driver MUMPS 50-120 s; **the driver's own Python 1000-1600 s a round, 50-65 %**.
- Refusals: model at the witness 4 % / 13 % / 21 % low (w = 2.99, 3.26, 2.70: beyond the plateau's edge), and one continuation rejection.
- Profile of that driver time (sandbox, the same record, cProfile on rank 0, round 0's SQP, reduced dimension 481): the reduced level-set search for missed peaks after each accepted SQP step (`missed_peaks` -> `hinf_dense_levelset`, dense eigenvalues of the 982 x 982 Hamiltonian) 44 s a call, 393 s for 9 calls; the SQP evaluations 16 s each (MIMO: a dense eig of the 491 x 491 reduced pencil every evaluation, 6 s; the compensator correction re-evaluates the whole design, 14 s); the reduced rows 0.73 s per batch. All of it is reduced-model work growing as dim^3 -- and the dimension grows with the plateau.

*The full model on the lattice (2026-10-06, sandbox, test plant; `tools/check_lattice.py`, `--oracle lattice`)*
- Unit checks against the serial full model, SISO order-10 record (752.96) and central 2x2 record (191.11), serial and 3 MPI ranks:
  - fixed rows (sigma_1 and the packed gradient) identical to `cl.gain` + `D.grads` (0.0 relative), sigma_2 blocks likewise; 3 ranks identical to serial (rows, followed rows, line);
  - followed peaks: the lattice row 1.2e-7 to 5.6e-7 (SISO) and 5.8e-9 to 9.6e-7 (central's flat tops) below the exact top (safeguarded Newton on the full model), levels 2^-6 to 2^-11; cold 3-6 evaluations per member, 2 on the same design;
  - the line: same clear and same alpha as `stability_and_abscissa` (-0.0793426752 SISO, -0.0722913502 central); the rightmost eigenvalue's gradient row against the serial eigenvectors 2.4e-8 / 1.7e-10; 29 / 32 shifts in 3 / 4 waves on the first call, the next design (moved 1e-4) 22 / 24 shifts in ONE wave (the kept cover);
  - costs (above, section 3): a line centre is 25-85 Arnoldi applications; k = 3 nearest eigenvalues cost 2-4 x that, so k = 1; a warm start helps sigma_1 (9 -> 4-5 applications), not the line.
- Found on the way: a cover of the line can miss the near-line eigenvalue nearest to no centre of it (the plant's -0.1148 at the zero start: the SQP saw alpha -0.1504); the next call's first wave now keeps every centre whose eigenvalue lies within the row reach (alpha -0.1148 since).
- End to end, SISO order 10, zero start, `ORACLE=lattice`, 2 ranks, cap 8 (the sandbox's 7 GB: the caches thrash, so its times say nothing about the cluster's):
  - round 0: 1464.37 -> **1091.07** certified (t 1091.073094, certificate 1091.073197: the rows' lattice deficit, 1e-7); 26 SQP evaluations, 1047 sigma evaluations, the line 609 shifts (at most 2 waves an evaluation), at most 7 stability rows, largest bracket deficit 8.4e-7; 4 frequencies added by the full-model sweep;
  - round 1: the first try sat on inadmissible iterates (a compensator pole at +0.17, section 5) -- fixed; then every iterate admissible (11 corrections back onto the line, 4 trials cut), candidate certified **804.77**; the straight-segment gate refused it (midpoint alpha -0.0458, no certified prefix); along the SQP's path (section 5) its 10 segments were certified (166 tracked queries, 6 splits): accepted;
  - for comparison, the reduced model from the same start on the cluster (job 760989, the fixed penalty of that version): 1223.05 and 1211.88 after rounds 0 and 1 (752.96 after 9).
  - Found: without a reduced model the end of the run computed the record's gain curve serially on the full model (~1700 factorisations, 37 min on the test plant, hours on the full mesh) and `D.hinf`'s seeded Newton; with the lattice the curve is on the ranks at the points of 2^-6, and `D.hinf` is skipped as with the reduced model.

## 9. Open items

- **Starts below the open loop:** the zero-controller start works (test plant, cluster: 0.826 x the open-loop peak in 3 rounds, section 8). The certified prefix (section 5) is to be seen on the cluster rerun.

- **The disks of the gamma certificate** are certificates by exhibition: each comes from eigenvalues Arnoldi returned converged (residual 1e-9) nearest the shift. A set that does not converge is retried, then the interval bisected; nothing is assumed. The norm-based (eigenvalue-free) radius is rigorous but 1e-7 to 1e-10 here, so it cannot replace them (section 4).
- **Plant certificate on the full mesh:** about 2 h on 48 ranks projected from the test plant (41 + 36 min serial); not yet run.
- **Winding count cost:** about 0.6 s per frequency point on the test plant (3400 to 3600 points), dense near the 282 hot plant poles; on the full mesh the n LU is 14 times dearer. Over 23 workers (92 segments) the test plant's 2013 s of task time is 90 s of wall, so 20 to 40 min on the full mesh, twice per start (feasible start, end of run); to be measured on the cluster.
- **The paper's own domain** ([-20, 100] x [0, 20], n = 400,775) is larger than the archive's ([-1.5, 30] x [0, 3.5]); gains differ by orders of magnitude (fig. 2a ~2e3 vs 2.2e6 here). The devices and the floor positions are the paper's; the domain stays the user's.
- **The driver's own work** (30 % of a round on the test plant): the rows and the near-line tracking are on the ranks (section 8). What remains serial: the dense eigs of the reduced pencil (MIMO every evaluation, SISO every 10th), the basis updates and projection, the gate, the driver's MUMPS solves. To be measured on the cluster (`[ledger]`).
- **Reduced model vs full model in the inner loop** (2026-10-05; the full model is now built, `--oracle lattice`, section 3; to compare on the cluster). Near the optimum the needed accuracy is the per-round gain (< 1 %) over a plateau of active frequencies; the reduced model gives 0.4-4 % for moves up to 0.3, 22 % for large ones (section 8), and port-range bases do not compress (1000-3000 vectors). The full-model alternative: plant factors on Omega cached for the run (design-independent; 133 MB each on the test plant, ~1.5 GB on the full mesh), an SQP evaluation = solves only (sigma_1 with warm vectors: ~40 solves, 14 ms each on the test plant, ~0.2 s on the full mesh) spread over the ranks. |Omega| = 474: test plant ~3 s per evaluation on 95 workers (~5 min a round), 63 GB of factors; full mesh ~40 s per evaluation on 95 workers, ~10 s on 380 (~15-20 min a round), ~700 GB of factors (2 Sapphire Rapids nodes at least, 4 to be comfortable). Hybrid: the reduced model while rounds gain several %, full-model rows once the gains drop below the model's error.
- **The KS weight:** kept for later (user).
