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

**The line certificate in segments over the pool** (`--cert_nseg`, default 16 in `run_bfs.sh`)
- Each shift is a full LU here, so the sweep's segments run on the workers (`certify_sparse._seg_worker`).
- The workers now return the eigenvalues their sweeps saw. The continuation gate's watch list, and the argmax used for the eigenvector, are therefore harvested in pooled mode too. Before, they were serial-only; GL with `--cert_nseg > 1` benefits as well.

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

**Interior margin**
- `--bfs-margin` (section 2).

**A refused step keeps its certified prefix** (`--gate-prefix 1`, 2026-10-02; GL default 0, unchanged)
- The continuation gate certifies the segment from the last accepted design to the candidate piece by piece, left to right (halving up to `--cont-depth`). When it gives up -- a crossing ahead, or the eigenvalue motion still too large at full depth -- the path up to its last certified piece is already proved free of crossings (`continuation_gate(..., reach=)`).
- That point becomes the candidate: its own line certificate (and the rectangle gate) and its gamma certificate; accepted if it improves on the best, with the step cap set to the certified length. The round is no longer lost (cluster test plant, zero start: round 0's step 1464 -> 1140 was refused whole, at depth 4).
- Cost when it fires: one line certificate and one gamma certificate (5 to 10 min on the full mesh), against a whole round (35 to 50 min).

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
  - `sigma.png`: sigma_1, sigma_2, sigma_3 on [0, W], W = the final band edge (at least 10), without control (open loop) and with it, and gamma as a horizontal line;
  - `modes_w<w>.png`: at the open-loop peak, the closed-loop peak and `--omegas` (default 0.3, 1.5), the optimal forcing (right singular vector of sigma_1) and response (left), without and with control, Re u (`--comp v` for v), with the actuator (half-amplitude circle and centre) and the sensor window (a black outline on the wall, as in the paper's figure 1) drawn on every panel, and in green their START positions (dotted circle, dashed taller outline, so overlapping windows stay apart) with an arrow for the actuator's move (the record carries `xa_start`, `xs_start` since 2026-10-02; for older records `--start INIT.npz`); the forcing is zoomed on the step;
  - `controller.png`: the controller's Bode plot (|k(i w)|, phase; the band shaded; its poles listed);
  - `post.npz`: every number and field behind them (singular values, the forcing f and response q of sigma_1 as dof vectors with their coordinates, k(i w)); `post.json`: peaks and timings. `--replot` redraws the figures from `post.npz` alone.
- How the vectors come without M^-1: sigma_k^2 are the eigenvalues of T^H M T M acting on the forcing f (the driver's M T^H M T with its factors cycled, so the eigenvector is the field itself, not its load M f); one LU per frequency serves both loops (Woodbury for the closed loop); k = 3, ARPACK.

## 6. Starts, and the small test plant

**HIFOO draw** (`bfs/hifoo_init.py`)
- The `hifoo` arm of `tools/randstart.py` (STAGE2: the better arm at GL), made real: every controller entry N(0, 1) -- d, e, f, nvec and K.
- Positions (`--positions`): `paper` (default, user 2026-10-02) puts every start at the paper's B2 (0.6, 1.0) and C1 window start 2.0; `random` draws the actuator centre uniformly inside the domain and the sensor window start uniformly on the floor.
- The catalogue is deterministic in (seed, n); the slurm array takes entry `SLURM_ARRAY_TASK_ID`.

**The zero controller** (`--kind zero`, user 2026-10-02; `KIND=zero` in the slurm scripts)
- The same draws with the loop open: nvec = 0, K = 0, only A_c kept (shifted stable by `feasible_start`). gamma starts at the open-loop peak exactly (test plant 1463.9), the floor a useful controller must go below; the HIFOO draws start above it (1655.3; after one round 1645.2).
- At k = 0 the gradient in the gains is generically nonzero (dsigma = Re(dk (u^H T b)(c^H T v)), dk = dK + dnvec (sI - A_c)^-1 B_c, any complex value at the peak); those in A_c and the positions vanish (they act only through the loop) and grow with the gains.
- `feasible_start` returns the certified anchor directly (nothing to continue: 38 s on the test plant, alpha -0.0560); the all-frequency count of an open loop is the plant's plus the compensator's, exactly (`BFSPlant.winding`, no march).

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
- `bfs/slurm/testplant.slurm` (24 ranks, 1 h 30 limit): one start on the test plant on the production path, `KIND` (default zero) and `OUTER` (default 3) rounds, then the figures and a one-line verdict (gamma per round against the open-loop peak). Makes the test plant's operators and plant data first if missing. Expected 35 to 45 min (serial sandbox timings: start-up 1422 s with the HIFOO continuation, round 0 587 s of which 334 s is the gamma certificate, which splits over the workers).
- `bfs/slurm/codesign.slurm`:
  - an array job, one start per task (HIFOO's controller draw at the paper's B2 and C1);
  - 24 ranks per start (the driver and 23 workers) and 240 GB: two starts per 96-core node (the gamma certificate's 2n pencil factor is 3.9 GB in every worker);
  - 15 h limit, `--deadline` at 14 h 15: with the driver's 30 min reserve the last round ends by 13 h 45, leaving 75 min for the final certificate, the count, the save and the figures;
  - then `bfs/postproc.py` on the saved record (also after a deadline stop: the figures then show the progress); a finished start re-submitted only draws missing figures;
  - resubmitting resumes unfinished starts; it refuses to start without the plant certificate.
- The launcher is the conda env's `mpiexec` (`LAUNCHER=` to change).

**Run time** (measured 2026-10-02; projected to the cluster)

*Ledger.* Each BFS round prints a `[ledger]` line:
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
- **Throughput:** 2 starts per node; 8 starts on 4 nodes.

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

## 9. Open items

- **Starts below the open loop:** the zero-controller start works (test plant, cluster: 0.826 x the open-loop peak in 3 rounds, section 8). The certified prefix (section 5) is to be seen on the cluster rerun.

- **The disks of the gamma certificate** are certificates by exhibition: each comes from eigenvalues Arnoldi returned converged (residual 1e-9) nearest the shift. A set that does not converge is retried, then the interval bisected; nothing is assumed. The norm-based (eigenvalue-free) radius is rigorous but 1e-7 to 1e-10 here, so it cannot replace them (section 4).
- **Plant certificate on the full mesh:** about 2 h on 48 ranks projected from the test plant (41 + 36 min serial); not yet run.
- **Winding count cost:** about 0.6 s per frequency point on the test plant (3400 to 3600 points), dense near the 282 hot plant poles; on the full mesh the n LU is 14 times dearer. Over 23 workers (92 segments) the test plant's 2013 s of task time is 90 s of wall, so 20 to 40 min on the full mesh, twice per start (feasible start, end of run); to be measured on the cluster.
- **The paper's own domain** ([-20, 100] x [0, 20], n = 400,775) is larger than the archive's ([-1.5, 30] x [0, 3.5]); gains differ by orders of magnitude (fig. 2a ~2e3 vs 2.2e6 here). The devices and the floor positions are the paper's; the domain stays the user's.
- **MIMO** (2 x 2, decentralised): not part of the fork yet (`bc` raises).
- **The KS weight:** kept for later (user).
