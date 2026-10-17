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
- *Where factors live.* Each rank keeps the factors it computed (`--lattice-cap`, default 24 per rank since 2026-10-08, 16 before; least recently used dropped; the closed loop's cache refreshes on a hit, `lu_lru`). The driver knows what each rank holds (every result reports it) and sends a point to a rank holding its factor, unless that rank's batch would then cost more than a factorisation on the least loaded one (relative costs: a row 1, a sigma_2 block 2, a line centre 3, a factorisation 2 more). A point no rank holds goes to the least loaded rank and, among equally loaded ones, to the one holding the fewest factors, the remaining ties in an order that rotates from one wave to the next. Before 2026-10-08 the ties went to the lowest rank: every new point of a small wave landed on the driver and the first ranks, whose caches then thrashed (central MIMO run: 5816 factorisations on the driver against 358 in the SISO run, all 96 x 16 slots full).
- *Waves* (2026-10-08). An SQP evaluation (`LatticeOracle.evaluate`) runs the rows and the stability line together. Each wave sends every outstanding request at once, one batch per rank, and the driver computes its own share meanwhile (`--lattice-driver on`). The first wave carries the fixed rows, every followed member's first points and the line's first centres. The later waves carry the members' walks and refinements beside the line's gap centres. One task kind serves them all (`eval_task`). Before, the rows ran first, then the line, and each followed member's bisection ran sequentially on one rank: the critical path was the longest bisection plus the line's waves.
- *Rows* (per SQP evaluation). A fixed member of Omega is evaluated at its point on 2^-10 (cached). A followed peak is followed by **safeguarded Newton at exact frequencies** (`lattice._newton`, 2026-10-09; the reduced path's `peakfollow.follow`, on the full model). The members run side by side, one per rank, and each Newton step is one wave:
  - The slope sigma' comes exactly from the gain's own solves. The curvature is carried from the last evaluation and secant-updated at every step; without it, two finite-difference points in one wave.
  - A step and its two halvings are evaluated side by side, and the longest that does not lower sigma is taken; a second batch of three before giving up.
  - Stop at |sigma'| <= `--lattice-tol` (1e-7) x sigma. The row is the top's exact value and gradient, with no grid noise in the merit.
  - The points are new at every evaluation, so their factors go to a small side cache (`XCAP` = 2 per rank) and never push the lattice's factors out.
  - Checked against the serial Newton on the central record 191.11: the same tops, sigma within 2e-14. Cold, 4 members took 6 waves (at most 15 evaluations per member, counting the parallel halvings); on the same design again, 1 wave.
  - Replaced (2026-10-08): bracket and refine on the lattice, whose row was up to 1e-6 x sigma below the top and could jump between lattice points from one evaluation to the next.
  sigma_2 blocks at the row point.
- *Stability rows and admissibility* (per SQP evaluation). The line `Re = -eps`, `Im in [0, band]`, covered by eigenvalue-free disks in **waves**: the first wave is the base grid (ranks/2 equispaced centres, at least 4; user 2026-10-09: the other half of the ranks fills the gaps in the same time) and the previous call's cover; then each gap gets as many centres as the neighbouring radii suggest, on the lattice point nearest each place at the first level finer than a quarter of the spacing; until covered, a witness on or right of the line (not clear), or a gap below 2^-17 (inconclusive, never clear). The proof is the cover, as in `stability_and_abscissa`; where the centres sit changes the cost, not the rigour. The SQP's `clear` is this certificate, every evaluation. A disk's radius is the distance to the nearest of all the Ritz values Arnoldi returned, and there is no disk when they did not all converge and pass their residual check (the nearest eigenvalue may then be missing; the serial strict code's untrusted disks); an inconclusive line reports alpha = +inf to the gates, so the continuation gate's midpoint test (alpha only) refuses it. Each verified eigenvalue within `--lattice-reach` (0.25) of the line gets its row from its left eigenvector on the same factor (adjoint Arnoldi), the two rightmost their vectors on request (the closed-loop Hermite block). `--lattice-k` nearest eigenvalues per centre (default 1: k = 3 costs 2-4 x the Arnoldi applications).
- *The strip* (2026-10-09, user: a full cover at all times). Every evaluation the SQP builds its QP from also covers the strip [-eps - `--lattice-reach` (0.25), -eps] x [0, band] (`LatticeOracle._strip`), and every eigenvalue in it gets its stability row. The line alone gives rows only for eigenvalues nearest to some point of the line; one behind another (-0.1444 behind -0.0723 on the central record) had none, and the QP kept proposing steps across it.
  - The cover is a quadtree of cells over the strip, first cells 0.25 x 0.25. A cell is covered when it lies inside one eigenvalue-free disk of any shift of the evaluation (line or strip; radius 0.9 x the distance to the nearest eigenvalue, 1 eigenvalue per shift), or inside a ball of radius `STRIP_RHO` = 2^-6 around an exhibited eigenvalue.
  - An uncovered cell is refined twice on the disks at hand, and each sub-cell still uncovered gets a shift at its centre in the next wave, on a 2-D dyadic point. The shifts that served are kept for the next evaluation.
  - What it proves: no eigenvalue in the strip other than the exhibited ones, up to the 2^-6 balls around them (a cluster tighter than that shows one member). A cell refined to 2^-10 still uncovered, or 30 waves, leaves it incomplete; the round's `lattice:` line counts those.
  - Built only for the iterates whose rows the QP reads (the start, every accepted step): a line-search trial's admissibility is the line's certificate, so the trials skip the strip. alpha stays the line's, so a trial and its iterate read the same value.
  - Measured on the central record 191.11 (serial, cold): complete, 253 shifts in 3 waves, 25 eigenvalues in it; an independent search (30 nearest at shifts every 0.5 along the strip's middle) found 17 there, all exhibited. On a design moved by 1e-4: 224 shifts (216 kept), 4 waves.
  - A simulator of the cover with the exact nearest eigenvalues (892 eigenvalues from a dense search, 31 in the strip): ~250 shifts in 3 waves the first time, ~205 in 2 waves on the same design again with ranks/2 = 48 line centres. Refining further before placing shifts (3-5 levels) placed 1.1-8x more shifts; a coarser ball (2^-5) saved a third but leaves eigenvalues within 0.011 of another unexhibited.
- *Missed peaks* (after each accepted SQP step; 20261009c, user: the sweep replaced by something robust): the **level-set certificate** at gam = t (1 + tol) on [0, band] (`LatticeOracle.level`, `lattice._level_disk`), the Schwerdtner-Mengi-Voigt disks of `certify_sparse.certify_greater` on the lattice. Either sigma_1 < gam is proven on the whole band, or the regions above gam are found and each gives a new member (followed). The sweep it replaces (sigma_1 and its slope on 2^-5) saw a peak only where it rose above a grid point.
  - *The disks.* Every centre is an axis point i theta on 2^-17, its plant factor the lattice's own (shared with the rows). A disk around i theta free of eigenvalues of the even pencil at level gam holds no level crossing, so sigma_1 - gam keeps its sign on the disk's diameter; the band holds a point below gam (every row is at most t), so a covered band is below gam everywhere. Two disks per centre, the larger used:
    - the **norm disk** (`_norm_disk`, cheap: one sigma_1 and m + p solves, 1-2 s): ||G(lam)|| < gam on |lam - i theta| < rho, from the push-through identity with the controller states eliminated exactly, G(lam) = R S(lam)^-1 R^T, S(lam) = lam M - A - B K(lam) C^H. At lam = i theta + delta the change is Q Delta P with P = [R; a C^H], Q = [R^T, b B], Delta = diag(delta I, -(K(lam) - K(i theta))/(ab)), so ||G(lam)|| <= ||Y|| / (1 - ||Delta|| ||Y||), Y = P S_0^-1 Q. ||Y|| is bounded by the 2 x 2 matrix of its blocks' norms (sigma_1, ||R S_0^-1 B||, ||C^H S_0^-1 R^T||, ||C^H S_0^-1 B||: dense m x m and p x p after the solves), ||K(lam) - K(i theta)|| by the controller's resolvent at i theta (r x r); the weights a, b (powers of 2) maximise rho. No M^-1, no Cholesky: R appears only through M = R^T R. Measured on the central record 191.11: rho = 0.069, 0.098, 0.15, 0.22 at theta = 4.5, 6, 8, 10; 0.0014 at theta = 3 (a controller resonance: ||(i3 - A_c)^-1|| = 18), 0 near the peaks. Checked: ||G|| on each disk's boundary (8 points, direct solves at complex lam) stayed below gam.
    - the **pencil disk**, where the norm disk is below 2^-5: the LEVEL_K = 2 eigenvalues of the even pencil nearest i theta (a pair mirrored across the axis, at equal distance), Arnoldi on OP = (A_H - s E_H)^-1 E_H applied through the plant factor of i theta M - A, **no factor of the 2n pencil** (it carries M / gam^2 in a full block, and gam changes at every call). Block elimination leaves (I - B) y = rhs with B = g T M T^H M, T u = (S^-1 [u; 0])_p, g = 1/gam^2: B is self-adjoint and positive semidefinite in the M semi-inner product with eigenvalues sigma_i^2 / gam^2 < 1, so **CG in that inner product** converges in 5-12 iterations, even 1e-4 below a peak (restarted GMRES in the Euclidean product needed 350-730 there, at theta = 0 on the central record: the operator is far from normal in that product). A pencil application is then 7 + 2 x (CG iterations) solves on the factor held. Measured (central record, gam = 191.13): 21-39 applications, 5-8 s, radius 0.0025 (theta = 0, the peak at w = 0), 0.078 (0.5), 0.198 (3), the same eigenvalues as the 2n factor (`SparsePencil`, 10-15 s with its factorisation). Far from the peaks the pencil's nearest eigenvalues are many at similar distances: 350-850 applications (90-220 s) at theta = 6-10 with ncv 12-40, 101 (28 s) with ncv 100 -- the norm disk covers that part. k = 4 cost 2-10 x k = 2 (35-357 applications); ncv 24 was the cheapest of 12/16/20/24 at theta = 3. Midway between peaks at a high level the norm disk is tiny and ncv 24 slow: at the SISO zero start (gam 1464.5) theta = 1.5 has sigma_1 347, norm disk 0.0022, ncv 24 did not converge in 8 restarts (32 s), ncv 100 converged in ~100 applications (23 s) with radius 0.49; theta = 0.25: ncv 24 25 s, 60 12 s, 100 21 s. Hence `LEVEL_ARNOLDI`: ncv 24 for 2 restarts, then ncv 100 for 3; if neither converges, the norm disk alone.
  - *Waves*, as the line's: the first call starts from the base grid (ranks/2 equispaced points); every later call from the previous call's cover (with the base grid too, 130 tasks for 96 ranks on the central record: two rounds of 5-8 s disks); then centres in the gaps (`_fill`, shared with the line), and a sliver narrower than 2^-17 between two nearly touching disks gets the finest lattice point next to it (it was left uncovered, 1.2e-6 wide, on the first try).
  - *Crossings.* A centre with sigma_1 >= gam is a region above, found directly (one per run of consecutive centres above gam, the highest). A pencil eigenvalue with |Re| <= 1e-8 max(1, |lambda|) is a crossing sigma_j = gam at Im lambda; it is confirmed by sigma_1 there (>= gam (1 - 1e-6), one wave at exact points), an upward and the next downward crossing bracket one region whose top the cubic Hermite fit on their values and slopes estimates, and an unconfirmed one (an eigenvalue just off the axis) is passed over. The search stops at the first wave that finds a region: the SQP adds the members, re-evaluates, and the next accepted step checks again.
  - *When.* Beside the accepted iterate's strip evaluation, in the same waves (`evaluate(level=...)`, memoised for `missed_peaks`), so its 5-8 s disks overlap the strip's waves; alone (`level()`) where there is no strip evaluation (`--lattice-strip off`, a pair entry). Up to 20261010c a trial put back on the compensator's line (`_ctrl_restore`) was evaluated with its strip, so when accepted it had no second evaluation and its level set ran alone: in central round 1, 903 s of a 2300 s round in separate level-set waves. Since 20261010d such a trial is evaluated as any other (line only), and the strip and the level set follow together once it is accepted. A member near a followed member or its top (1e-3) is not new; a top near a FIXED member is.
  - The round's `lattice:` lines report calls, shifts (pencil / norm), waves, inconclusive calls and the largest CG count. `--lattice-sweep` now sets only the certificate's reference sweep.
- *The gates* (`D.alpha_certified`, the stability gate, the continuation gate's line checks): the same lattice line, memoised per design, so a candidate the SQP evaluated is not swept again. The ports' positions behind each (b, c) the driver builds are registered (the ranks build b and c from them).
- *Not on the lattice:* the round's gamma certificate (its 2n pencil carries gamma in a full-rank block, M / gamma^2: no factor serves two levels; the in-solve level set above applies the same pencil through the plant factors instead, by CG in the M semi-inner product -- an earlier estimate of that elimination, 3.5 x the cost per Arnoldi step of the 2n factor, used restarted GMRES, which stalls near a peak); the continuation gate's tracking; the winding count (3400 samples per count, more than any cache, run twice per run: 133 s of critical path in job 762583).
- *Costs measured* (test plant, one core; `tools/check_lattice.py`): a line centre is 25-85 Arnoldi applications (0.6-1.9 s) on top of the factor (1 s), so the lattice saves the factor there, not the Arnoldi (a warm start does not help on the line: 37-154 applications); a sigma_1 row is 9-17 applications cold, 4-5 warm.
- *Memory:* about |Omega| + the line's centres (~25) + the sweep (321) + the followed brackets' points, spread over the ranks. Test plant: 133 MB per factor, so ~900 factors = 120 GB on one node. Full mesh: ~1.05 GB per factor: ~900 GB, i.e. several nodes (`LATTICE_CAP` x ranks x 1.05 GB per job).
- *Retirement* (`--omega-retire FRAC:ROUNDS`, `RETIRE=0.95:3` in the scripts, directories `_ret`; lattice only; 2026-10-08). At the end of each ACCEPTED round, sigma_1 is computed at every member's lattice point at the accepted design: a followed member's top (its lattice point, recovered on 2^-16 from the 1e-6 rounding), the others on 2^-10. All are points the last SQP evaluation factored. The followed members sit on their tops at the candidate, which is then the accepted design; after a refused round they do not, so the streaks are left as they are. A member below FRAC x gamma for ROUNDS consecutive accepted rounds leaves Omega. The streak follows a member that moved with its peak (the nearest member of the last round within half the follow radius). Never retired: the certified peaks of the accepted design, the sigma_2-enrolled members, a member that refuted a trial design within `--omega-refuter-rounds`, and the highest member. A peak that rises again is caught by the in-solve sweep and by the certificate, which add it back. Retirement on the reduced model (`--omega-policy an`, 2026-09) did not help; the question is whether it was the model's inaccuracy, since on the lattice the values are exact. The streak is not saved on resume (it restarts at 0).

## 4. Certification without M^-1

The algebra is in `docs/note_algebra_clean.tex`, section "The backward-facing
step: algebra without M^-1".  What each GL ingredient needed and what the BFS
uses:

| GL ingredient | needed | BFS |
|---|---|---|
| gamma pencil (even, sparse + low rank) | nothing | the same pencil: with the weak-form ports B B^T = C^T C = Q, so no inverse ever appears |
| eigenvalue-free disks of the gamma certificate | nearest eigenvalues; norm-based fallback with M^-1 | nearest eigenvalues only (Arnoldi, converged); the norm fallback is M-free now but useless here, off |
| numerical range (enclosures, heights, tail) | Lanczos with M^-1 | tops of Hermitian pencils on ker B, certified by inertia (`bfs/mfree.py`) |
| winding count tail | M-norms of b and c | the smaller of an H^1 bound and an L2 bound in the velocity mass (the wall-shear sensor is not of the form M psi; the L2 norm by CG, no factor of M) |
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
- Tail (`BFSPlant.gbound`, 2026-10-16): the smaller of two bounds, d the numerical-range distance.
  - H^1: |G(s)| <= ||c||_{L^-1} ||psi_a||_M sqrt((1/d + (beta - Re s)_+ / d^2) / nu1), L the velocity stiffness (exported with the operators), nu1 = nu/2 (note). It decays like d^-1/2, so the cutoff W_T it needs grows like the square of the feedthrough K; test plant, a feasible start with K = 0.005: W_T = 120.
  - L2: |G(s)| <= ||c||_{M_vv^-1} ||psi_a||_M / d. Step (i) of the same derivation gives ||u||_M <= ||psi_a||_M / d; c is zero off the free velocity dofs, so Cauchy-Schwarz in M_vv (the velocity mass on them) bounds c^H u. It decays like 1/d and W_T grows linearly in K. ||c||_{M_vv^-1}^2 = lambda_max(C^H M_vv^-1 C) comes from CG on M_vv with a residual-corrected bound, c^H M^-1 c <= c^H y + y^H r + ||r||^2 / lambda_floor. The floor lambda_floor is the P2 element floor min_e lambda_min(M_e), checked by reassembling M from the mesh. The velocity mass is never factorised or inverted (user, 2026-10-01). Test plant: floor 1.94e-5 (true lambda_min 1.08e-4), cM = 738.05 at the 369.87 record (the LU value is 738), 64 CG iterations, 0.3 s, cached per sensor configuration.
  - ERRATA (2026-10-16): this entry said the d^-1/2 rate was genuine (the Stokes layer, a continuum argument). The discrete test plant decays like 1/w: w ||G(-0.05 + i w)|| = 11.74 from w = 1e2 to 1e7 at the 369.87 record. With the H^1 bound alone, that record needed W_T = 7.86e6, the last rung below the count's old cap of 1e7. Every later decentral design of 20261010n raised "tail bound never certifies" there, and the veto (three instant failures charged against `--veto-counts 3`, then the checkpoint rewritten with the anchor) lost those jobs' designs; the veto and the save were fixed with the bound (`run_case_epi.winding_veto`, `ckpt_after_veto`, the `_final_<jobid>` and `_accepted` files). min(H1, L2) gives W_T 7680 (369.87), 3840 (412.28) and 1920 (central 123.92), against 7.86e6, 1.97e6 and 491520. The cap is now 1e9 (`winding_cert.WT_CAP`), and the error is `winding_cert.TailBoundError`, which names W, |k| gb(W) and q. The ladder w0 2^k is unchanged, so a bound that certified before gives the same W_T.
- Plant data: the plant eigenvalues right of line - 0.6, certified complete by the M-free enclosure (rectangle + tiles exhausted by nearest-k shift-invert; the norm-based empty tile is off for the BFS, `sparse_spectrum.FREE_TILE = False`). Test plant: none right of -0.025, 282 within 0.6 of it (36 min serial, once per mesh).
- Used at the end of every run (record: `wind_count`, `wind_info`) and to check the feasible start (count right of -eps, which must be 0).
- Where the points go (test plant, 2026-10-02): 13 of 3617 beyond w = 30 with W_T = 120; 17 with W_T = 1e3 or 1e4 (the step grows to W_T / 8 once f is smooth). The march's cost is the hot region below w = 30: the plant poles near the line and the loop's zeros, which do not depend on the sensor type. A velocity sensor (c = M psi_s, tail ~ 1/d instead of d^-1/2, W_T ~ |K| instead of K^2) would save a few points per count, nothing more. (2026-10-16: the L2 tail above gives the shear sensor that 1/d rate too.)
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
- **Trials across the closed loop's line, since 20261010d** (`--lattice-cross merit`, default; `CROSS=cut` in the scripts restores the rule below, directories `_cut`): weighed by the merit, as on the reduced path. An iterate may then sit outside the line until its rows pull it back; only admissible iterates become the solve's result, its certified fallback candidates, or points of the continuation gate's path (the gate certifies the straight segments between consecutive admissible iterates). And the stability rows (and the closed-loop Hermite block) aim at -eps - 2 mg, one mg (`LAT_CUSHION`) inside the admissibility line -eps - mg: aimed at the line itself, a step that kept an active eigenvalue on its linearised boundary landed on it to rounding, half the time just outside, and with the rule below every such trial and its halvings were cut (section 8, 20261010c). Why: section 8, the 20261010c runs.
- **With the lattice oracle, an admissible iterate is never left for an inadmissible one** (2026-10-06 to 20261010c; `--lattice-cross cut` since): every trial's line is certified on the full model anyway, so a trial that is not (line not clear, or alpha above -eps - margin) halves the step instead of being weighed by the merit. Found on the MIMO central zero start (sandbox): a step took a closed-loop eigenvalue to -0.0496, across the line, for a sigma decrease worth more than the penalty at nu = 1e6. The reduced model keeps the merit (its abscissa is a prediction). With it, every iterate of the SQP's path is admissible, which is what the gate along the path (below) relies on. (Since 20261010d, under the merit rule, the path can leave the admissible set between gated points: the gate then refuses, and since 20261010j the refused candidate is counted right of -eps, `--cont-accept-winding`.)
- **How far the cut trials cross** (20261009c, user: measure it). Each trial cut back inside the closed loop's line records how far past the line its offending eigenvalue went (Re lambda + eps of the line's witness, else of the rightmost seen; a lower bound, the line stops at its first witness) and, when that eigenvalue had a stability row at the iterate, what the row's linearisation predicted for the same step, Re lambda_j + alpha grad Re lambda_j . d + eps. The round's `lattice:` line prints the largest and median depth, how many had a row, the median prediction and how many predictions were already past the line, and how many cuts were at a full step (`stats['lattice']['cross']`). A prediction past the line means the QP's step gave the row up through its elastic slack (the penalty nu was cheaper than the row); a prediction inside it means the eigenvalue moved further than its linearisation (curvature, or two eigenvalues meeting).
  - What the rows do: each eigenvalue in the strip is a linearised constraint Re lambda_j + grad Re lambda_j . d <= -eps (the two rightmost as the closed-loop Hermite block), elastic, at penalty nu. When one is active the QP's step lies on the linearised frontier -- the projection of the descent onto the frontier, to first order. The true frontier is curved, so a step along the tangent leaves it at second order, and with the lattice rule that trial is halved, not projected back: there is a second-order correction for the compensator's line (exact, r x r), none for the closed loop's. The reduced path never cut such trials: the merit (sigma + nu x violation) weighed them, and an iterate could sit slightly outside until the penalty pulled it back.
  - Two remedies, to choose from the measurement: (a) a closed-loop second-order correction, the crossing eigenvalue put back on the line by the least-norm step along its gradient at the trial (its left and right vectors there: one more evaluation); (b) the merit decides again, as on the reduced path, while only admissible iterates may become the solve's result.
- **The strip on or off** (`--lattice-strip on|off`, `STRIP=0` in the scripts, directories `_nostrip`; 20261009c, user: measure whether it pays). Off: the stability rows are those of the eigenvalues the line's disks see (with `--lattice-k` per centre), as before 20261009a, and the accepted iterate is not evaluated a second time.
- **The modal compensator** (`--ctrl-modal on`, the default of `run_bfs.sh` since 20261010e; `MODAL=0` for the tridiagonal form, needed to resume a run made before; user, 2026-10-11, after the coalescing poles of 20261010d, section 8). A_c is block diagonal in real 2 x 2 blocks M = [[a, b], [c, g]] per loop (a 1 x 1 block closes a loop of odd order): the tridiagonal with e = f = 0 between blocks, those entries out of the design vector (`controller.modal_blocks`, `modal_cut`; both packers). A block's poles are left of the line L iff tr(M - L I) <= 0 and det(M - L I) >= 0 (Routh-Hurwitz for degree 2): two rows per block in the SQP, v1 = tr M / 2 <= L and v2 = L - det(M - L I)/s <= L (s = |a - L| + |g - L|, at least 1e-3, frozen at the iterate: v2 in the units of a real part near the line), one per 1 x 1 block (`controller.modal_rows`). Polynomial in the parameters, exact, smooth everywhere -- a double pole inside a block, or poles of different blocks meeting, are ordinary parameter values, where the tridiagonal per-pole rows had unbounded gradients (cube root of a step at a triple pole). A block moves continuously between a complex pair and two real poles. No Hermite block and no double-pole face in this mode; the second-order correction is Newton's least-norm step on the violated block rows (a trace is linear in z, a determinant quadratic). The admissibility test is unchanged (the abscissa of A_c). Starts: a compensator already in blocks (a modal run's record or checkpoint) is kept; the zero controller (C_c = D_c = 0) is brought to blocks with the same poles (`modal_form`: complex pairs [[a, b], [-b, a]], real poles two by two on the diagonal); any other start is refused -- its B_c and C_c would have to follow the similarity. Checked: the block rows' Jacobians through both packers against finite differences (7.6e-11 relative, SISO, central, decentral); zero starts converted with their poles unchanged; GL frozen path bit-identical; SISO zero start end to end in the sandbox (3 iterations: 10 compensator rows, no compensator block, the steps accepted at alpha 1). Not converted: a record with a controller (C_c != 0) -- at the 20261010d records' triple poles the similarity to blocks is nearly singular (its B_c and C_c would blow up), so those runs restart cold.
- **Three compensator poles on the line** (fixed 20261010d): the poles the Hermite block (or the face S) covers leave the per-pole rows, and they were picked out by REAL PART. A third pole with the same real part -- the conjugate pair -0.05 +- 2.454i in the block and the real pole -0.05 beside it, on the SISO record 641.71 -- left too, and had no row at all: every QP step pushed it across at first order (4.6e-3 at a full step of 0.01; slack 6e-10, the QP saw nothing), the correction back onto the line moved the design by 40 % of the step whatever alpha, sigma rose by 0.35-0.7 instead of falling, and the line search halved to 1/256 and failed (debugged on the record, `--sqp-debug-ls on`). Now exactly the covered poles leave, each cluster pole taking its nearest row (`_cluster_idx`): a third pole on the line, or a third pole of a triple one (the central no-strip record 679.76 has a triple real pole at -0.05), keeps its row. The same on the face S (whose real-part rule dropped the conjugate pair -0.05 +- 3.13i beside the double pole of the central record 191.11) and in `_alc_cheap`.
- **Conjugate pairs are not double poles:** the face S was entered whenever the two rightmost compensator poles sat on the line, at any separation (a GL rule: GL's controllers are complex). For the BFS's REAL compensator the two rightmost are often a conjugate pair -- one smooth constraint, its real parts equal by symmetry -- and entering the face for it projected z by 0.49 (pair 3.6 apart) into an unstable iterate. A real compensator's conjugate pair more than 0.1 apart no longer enters the face. A CLOSE conjugate pair still does: it is a double real pole split by rounding -- the central 2x2 result's double pole at -0.05 is computed as -0.050001 +- 5e-8 i (a first version of the rule, refusing every conjugate pair, kept that one off the face too, and the SQP then crawled at alpha ~ 1e-7). This affected every earlier BFS run whose rightmost compensator poles were a distant conjugate pair on the line (the central 2x2 result has one, -0.05 +- 3.13i).

- The compensator constraint keeps GL's form: per-pole rows, the 2x2 Hermite block of the two rightmost poles (Burke-Overton coordinates), and the double-pole face S with its retraction (a pair of poles meeting on the line is held there as a double pole, every step projected back onto the face). Four poles on the line (the central 2x2 result: a double real pole at -0.05 and -0.05 +- 3.13i) are the double pole on its face plus a conjugate pair, which is smooth (per-pole rows). The Lyapunov LMI (`--sqp-lmi`) was tried on GL and fails exactly at near-double poles (cond P ~ 1e7, the QP solver's NumericalError; STAGE1B 7.1), so it stays off.

**Trust radius after a partial step** (`--sqp-partial-shrink on`; `SHRINK=1` in the scripts, off by default since 20261008c after the cluster runs stalled with it, section 8; GL default off, unchanged)
- A step accepted at alpha < 1 sets the radius to 2 alpha |d|: the line search found where the model holds (2 alpha failed), so the next QP picks its direction for that region instead of a box corner cut back again.
- The line search then starts at alpha = 1, not at twice the last alpha (`ALPHA_WARM`). With both the radius and alpha remembering the cut, they ratchet each other down, as on the face S (ne=100, round 11: radius 2.5e-7).
- Before, only the double-pole face S did this (`PAIR_BALL`). Elsewhere the radius only halved on a rejection and doubled after a full step at the boundary.
- Why: the SISO lattice run (job 763575) spent 5-8 evaluations per SQP iteration in its last rounds (rounds 13-18: 245/38, 203/40, 246/31, 187/40) against 2-3 in its first (rounds 0-3: 23/10, 56/20, 97/40, 109/40) -- line-search trials of the steps the box kept proposing.

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
  - After a save-time veto (2026-10-16, `run_case_epi.ckpt_after_veto`), a deadline stop keeps the final design in the checkpoint only when the veto could not count it before the march (a `TailBoundError` or another failure of the tail test: nothing is known against it, and the next job's start count decides, FATAL on a conclusive count > 0). After a marched INCONCLUSIVE, a conclusive count > 0 or a failed march, the checkpoint holds the veto's fallback. With nothing saved (NOT SAVING), the same rule applies, and the anchor (the job's start) takes the fallback's place; it is no longer left at the vetoed design.
- `<tag>_final_<jobid>.npz` / `.json` (2026-10-16, `write_final_design`): every BFS job's final design (the last accepted, the one the veto judges), written at the save whatever the veto decides. The npz has the record's design keys, gamma, alpha, round, the loop state and the working set; when the record is another design (the veto's fallback), that design under `record_` keys; the veto as JSON. An existing name gets `_2`, `_3`: nothing overwrites it. In 20261010n the five final decentral designs were in no file.
- `<tag>_accepted.npz` (2026-10-16, `append_accepted`): every accepted design of every job of the run, appended at acceptance (`a<k>_d`, `a<k>_K`, ... with the index arrays `gamma`, `round`, `job`), a few kB each, so no accepted design is lost.
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
- **One root for everything, the code by git** (20261010i-j; user, 2026-10-14: "change the origin to /tmp_user/sator/cleclerc/BFS", exchange through GitHub; the layout review of 20261010i, LESSONS.md 15 and 19). Until 20261010g every code version was unpacked into its own directory (`BFS/20261010g/codesign_bfs`): the test plant's files were rebuilt for each version, and a run had to be copied into the new version's `testplant_bfs/` before `CONTINUE=1`.
  - **The root** `/tmp_user/sator/cleclerc/BFS` carries the marker `.bfs_home` (`INIT=1 bash bfs/home.sh <root>`). In it:
    - `codesign/`: the git clone of the branch `bfs`, updated by `git pull` and **never submitted from**. `version_check.py` refuses to run a job in it (`ALLOW_CLONE=1` overrides).
    - `v/<version>/`: one git worktree per code version, made by `bash bfs/stage.sh` at the clone's commit. Jobs are submitted from there. A pull in a directory that jobs run from changes their code: Slurm spools only the batch script, and the driver loads `winding_cert` and `postproc` only at the end of a resumed run. The worktree is never pulled. `stage.sh` refuses a version label already staged at another commit (the MANIFEST label must change with the code).
    - `testplant_bfs/`, `results_bfs/`, `smoke_bfs/`, `logs/`: shared by every version. Each worktree links to them (`bfs/home.sh`, run by `stage.sh`). The smoke test writes `smoke_bfs/<version>_<job>/`.
    - `data/<kind>-<id>/`: the plant files (mesh, operators, certified range and poles) by content id (`bfs/plant_id.py`). The id hashes the generators' source without comments and blank lines (`bfs/mesh_coarse.py`, `ops_dolfinx.py`, `archive_dsr/nsforms.py`, `plant_io.py`, `plant_cert.py`), the same under every Python version (since 20261010k; 20261010j hashed syntax trees, whose dump changes between Python 3.12 and 3.13), the archive's input data, and `RANGE_VERSION` in `bfs/plant.py`. Bump `RANGE_VERSION` by hand whenever the computation of the range or poles changes. An id's files are built once, under a build lock that names its owner job (`bfs/lib.sh`: a dead owner's lock is taken over, a live one waited for). They are never rebuilt in place, so a version with other generators gets its own directory and the final winding certificate of a running job always reads its own plant's data. The scripts find their operators with `python bfs/plant_id.py --dir coarse|fx`. The 20261010g files have the current id (the generators are unchanged) and are adopted as they are.
    - `.results_wt/`: the worktree of the branch `results` (`report.sh PUSH=1`).
  - **Runs.** One driver per run directory: `.job.lockd` holds the job id, a dead owner's lock is taken over, a live one refused. Each job appends a line to `code_versions.txt` (date, version, commit, job, plant id). A `RESUME`/`CONTINUE` on another plant id is refused (`PLANT_CHANGED=1` moves the run on purpose). `codesign.slurm` refuses to auto-resume a checkpoint left by another version (`RESUME=1`). The records carry `code_version`, and a resume prints "checkpoint written by X, continued by Y".
  - **`bfs/home.sh`** never deletes or overwrites. A real directory is emptied into the root with `mv -n`, and an entry already there is reported. Lock directories and temporaries are left alone. It refuses while jobs are queued or running from a directory it would move out of (squeue `%Z`), or across file systems. `FROM=<old code directory>` also moves an old version's runs, logs and plant files; plant files move only when their id equals this code's.
  - **One-time setup on sator:**
    ```
    cd /tmp_user/sator/cleclerc/BFS
    squeue -u $USER -o '%i %T %Z'          # nothing may still run from 20261010g/codesign_bfs
    git clone -b bfs https://github.com/ColinLeclercq/codesign.git codesign     # or git@github.com:ColinLeclercq/codesign.git
    cd codesign && git config user.name "Colin Leclercq" && git config user.email colin.leclercq@onera.fr   # the identity of report.sh's results commits
    INIT=1 bash bfs/home.sh /tmp_user/sator/cleclerc/BFS
    FROM=../20261010g/codesign_bfs bash bfs/home.sh                              # moves the six 20261010g runs, logs, plant files in
    bash bfs/stage.sh                                                            # -> BFS/v/20261010j
    ```
  - Pushing (report.sh PUSH=1, 2026-10-16): GitHub refuses the account password for git. With an SSH key registered on GitHub (`ssh -T git@github.com` answers "Hi ColinLeclercq"), the clone's remote is `git remote set-url origin git@github.com:ColinLeclercq/codesign.git`, shared by every worktree. If port 22 is closed, `~/.ssh/config` gets `Host github.com` / `Hostname ssh.github.com` / `Port 443`. Otherwise a fine-grained personal access token limited to this repository (Contents: read and write) is typed as the password.
  - **Each new version:** `cd /tmp_user/sator/cleclerc/BFS/codesign && git pull && bash bfs/stage.sh`, then `cd ../v/<version>` and `sbatch` there. Remove an old worktree with `git worktree remove ../v/<version>` once no job, pending ones included, runs from it.
- **What to send back** (since 20261010j): `PUSH=1 bash bfs/report.sh` from the worktree. It makes `for_claude/` (below), mirrors it into the folder `<version>/` of the branch `results`, and pushes.
  - Mirroring replaces the folder's `SUMMARY.txt`, `version.txt`, `logs/`, `runs/` and `figs/`, so a figure the new report no longer has is not kept. A `NOTE.txt` written on GitHub stays, and `NOTE="..."` appends to it.
  - Files are added with `-f`: the branch inherited `bfs`'s `.gitignore`, whose `logs/` line kept every 20261010g log off GitHub.
  - Each report starts from `origin/results` and retries the push 3 times.
  - The logs taken are this version's, now that logs and runs are shared:
    - those whose `version_check` line names the version, or says the checkout is NOT it;
    - logs without a version line (a job that died before the check) whose job is queued, or that are not older than the version's oldest log.
  - The runs taken are those the logs name (`-> testplant_bfs/...`; keep `OUT` relative). The figures are taken too (`figs/<run>/`), and files above 50 MB are left out and listed (`BIG=1` keeps them).
  - `ALL=1` takes everything, and `VERSION=20261010g` another version. Without `PUSH=1` the tarball is made as before.
- **What to send back** (2026-10-10, user): `bash bfs/report.sh` in the code directory, at any time, fills `for_claude/` and `for_claude.tar.gz`: `SUMMARY.txt` (per job log: the run, every round's line, the lattice lines and the lattice `[ledger]` line (ranks busy %), refusals (also the gate's and the winding check's since 20261010h), errors, the final state; slurm's copy of the script at the end of each log is left out), every log, and per run directory the record's JSON and npz (not the checkpoint), `starts.json` and the figures' `post.json`. Every slurm script carries `#SBATCH --comment=36973004F` (the cluster's project code).
- **The dolfinx JIT cache** (2026-10-10): `env.sh` points `XDG_CACHE_HOME` at `<code directory>/.cache` and sets `CC=gcc CXX=g++`, so dolfinx/ffcx compile the forms into a cache of the code directory with gcc (an Intel `CC` exported by the submitting shell would otherwise reach the job: sbatch passes the environment on). The six test-plant jobs of 20261010a (767360-767365) died building the operators in a fresh directory: `~/.cache/fenics` (on sator `/stck/cleclerc/.cache/fenics`) held a module for the same form compiled elsewhere with the Intel compiler, and loading it needed `libimf.so`, absent here (`ImportError: libimf.so`, after `FileExistsError` on its `.c` file). Until this reaches the cluster: `rm -rf /stck/cleclerc/.cache/fenics` before launching.
- `ORACLE=lattice` (both scripts, through `run_bfs.sh`; section 3): the inner loop on the full model. Each rank then holds up to `LATTICE_CAP` plant factors besides its usual work (the gamma certificate keeps two 2n factors), so the job's memory is about ranks x (1 + LATTICE_CAP x factor) GB. Test plant (`testplant.slurm`, default cap 24 since 2026-10-08): 133 MB a factor, `--ntasks=96 --mem=450G` on a 512 GB node. The October 6 runs used 211-253 GB at cap 16; cap 24 adds ~100 GB. On 192 GB nodes use `LATTICE_CAP=8`. Full mesh (`codesign.slurm`, default cap 16): ~1.05 GB a factor, 3.9 GB per 2n factor, ~25 GB a rank at cap 16, i.e. ~20 ranks per 512 GB node and several nodes. With `ORACLE=lattice`, `run_bfs.sh` passes `--sqp-partial-shrink` on with `SHRINK=1` (off by default since 20261008c; section 5) and `--omega-retire` with `RETIRE=FRAC:ROUNDS` (section 3), `--lattice-strip off` with `STRIP=0` (section 5). Directories get `_lat` (and `_shr`, `_ret`, `_k2`, `_nostrip`), so lattice and reduced runs never share one.
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
  - fixed rows (sigma_1 and the packed gradient) identical to `cl.gain` + `D.grads` (0.0 relative), sigma_2 blocks likewise (0.0 against `sqp_inner`'s own construction); 3 ranks identical to serial (rows, followed rows, line); a second evaluation of the same design identical (cold starts, the default);
  - followed peaks: the lattice row 1.2e-7 to 5.6e-7 (SISO) and 5.8e-9 to 9.6e-7 (central's flat tops) below the exact top (safeguarded Newton on the full model), levels 2^-6 to 2^-11; cold 3-6 evaluations per member, 2 on the same design;
  - the line: same clear and same alpha as `stability_and_abscissa` (-0.0793426752 SISO, -0.0722913502 central); the rightmost eigenvalue's gradient row against the serial eigenvectors 2.4e-8 / 1.7e-10; 29 / 32 shifts in 3 / 4 waves on the first call, the next design (moved 1e-4) 22 / 24 shifts in ONE wave (the kept cover);
  - costs (above, section 3): a line centre is 25-85 Arnoldi applications; k = 3 nearest eigenvalues cost 2-4 x that, so k = 1; a warm start helps sigma_1 (9 -> 4-5 applications), not the line.
- Found on the way: a cover of the line can miss the near-line eigenvalue nearest to no centre of it (the plant's -0.1148 at the zero start: the SQP saw alpha -0.1504); the next call's first wave now keeps every centre whose eigenvalue lies within the row reach (alpha -0.1148 since).
- End to end, SISO order 10, zero start, `ORACLE=lattice`, 2 ranks, cap 8 (the sandbox's 7 GB: the caches thrash, so its times say nothing about the cluster's):
  - round 0: 1464.37 -> **1091.07** certified (t 1091.073094, certificate 1091.073197: the rows' lattice deficit, 1e-7); 26 SQP evaluations, 1047 sigma evaluations, the line 609 shifts (at most 2 waves an evaluation), at most 7 stability rows, largest bracket deficit 8.4e-7; 4 frequencies added by the full-model sweep;
  - round 1: the first try sat on inadmissible iterates (a compensator pole at +0.17, section 5) -- fixed; then every iterate admissible (11 corrections back onto the line, 4 trials cut), candidate certified **804.77**; the straight-segment gate refused it (midpoint alpha -0.0458, no certified prefix); along the SQP's path (section 5) its 10 segments were certified (166 tracked queries, 6 splits): accepted;
  - for comparison, the reduced model from the same start on the cluster (job 760989, the fixed penalty of that version): 1223.05 and 1211.88 after rounds 0 and 1 (752.96 after 9).
- End to end, MIMO central 2x2, order 10, zero start, 2 ranks, cap 6, one round: 1464.37 -> **1065.63** certified (alpha -0.0734); 26 evaluations, the line 658 shifts (at most 3 waves, at most 10 stability rows), largest bracket deficit 7.3e-7; the gate along the SQP's path certified its 7 segments (67 tracked queries, 1 split); the end of the run (curve on the lattice, the gate, the record) ran. The sandbox's sweep level (2^-3) missed a narrow peak near w = 0.25 until a later sweep (1116 on the grid, 1347 at its top): the cluster default 2^-5 samples 4 x finer. This run also found the closed-loop crossing that led to the rule of section 5 (an admissible iterate is never left, lattice oracle). (The sandbox run with the reduced model of 2026-10-02: 1017.97 after one round, with the fixed penalty of that version.)
- The SISO run continued on the final code (from 804.77): every iterate admissible (alpha -0.064, the compensator on its line at -0.0500), smax 804.77 -> 794.02 in the first 3 iterations, steps cut to alpha = 1/16 - 1/32; the sandbox's container restarted there, so the round did not finish.
  - Found: without a reduced model the end of the run computed the record's gain curve serially on the full model (~1700 factorisations, 37 min on the test plant, hours on the full mesh) and `D.hinf`'s seeded Newton; with the lattice the curve is on the ranks at the points of 2^-6, and `D.hinf` is skipped as with the reduced model.

*The three cluster runs on the lattice (2026-10-06 code, test plant, 96 ranks, 450 GB; read 2026-10-08)*
- SISO order 10, zero start: **632.24** after 19 rounds, 7.8 h, 211 GB. Decentral 2x2, zero start: **547.30** after 12 rounds, 7.6 h, 243 GB. Central 2x2, resumed from the reduced run's 173.77 with its 632 members: **145.80** after 15 rounds (|Omega| 632 -> 667), 13.3 h, 253 GB. Winding count 0 in all three.
- Not stationary: no round stopped on "exchange converged"; all three stopped on the progress rule. The inner solves ended on codes 4 and 6 (budget, radius collapse): SISO had code 0 in 8 of 19 rounds; decentral never; central code 12 once.
- Where the time went (ledger): rows 47 / 47 / 71 % (SISO / decentral / central), line 37 / 39 / 20 %, sweep 6 %, gamma certificate 3-7 %. Rows per evaluation 8.3 / 9.2 / 18.8 s.
- Central thrashed its caches: all 96 x 16 slots full, 5816 factorisations on the driver against 358 in the SISO run. New points of small waves went to the lowest ranks first.
- sigma_2 enrolment on central: none in rounds 0-5, then 23-25 members, all at w 0.778-0.883, gaps 4.7-14.5 %.
- Late SISO rounds: 5-8 evaluations per SQP iteration against 2-3 early (above, section 5).

*First cluster rounds on 20261008a (jobs 764747, 764752-764754, test plant, 96 ranks; read after 3-4 rounds)*
- Rows and line together: 7.7 s per SQP evaluation (central rounds 0-2: 1478 s in 191 evaluations, 3.8 waves each). The October 6 code took 8.3 s for the rows alone on SISO, and 6-7 s more for the line.
- SISO: 1078.54, 726.28, 691.79 after rounds 0-2 (round 3 starts at 2463 s). The October 6 code: 1084.2, 824.3, 751.5, 687.9 after rounds 0-3 (687.9 reached at ~4500-5600 s).
- Decentral: 939.47, 702.40, 679.27 after rounds 0-2. The October 6 code: 935.0, 602.0, 595.6, so another path, behind for now (multimodal).
- Central, cold: 1040.66, 720.63, then round 2 refused (the missed peak above). |Omega| stays at 22-45 in all four runs. Retirement (0.95:3) had not yet acted: 14 of 25 and 8 of 37 members below 0.95 gamma, none for three rounds yet.
- Caches: all 96 x 24 slots full by round 3, yet only 118-129 factorisations on the driver, against 5816 in the October 6 central run.

*The same four runs to the end (read 2026-10-08 16:00; decentral still running)*

| run | 20261008a | rounds, wall | October 6 code |
|---|---|---|---|
| SISO | **690.52**, stopped by the progress rule | 7, 2.1 h | 632.24 (19 rounds, 7.8 h) |
| decentral | 679.17 after 17 rounds, still running, ~0.002 %/round since round 4 | 6 h so far | 547.30 (12 rounds, 7.6 h) |
| central, cold | **654.57**, fixed point ("stalled, not converged") | 13, 5.7 h | (resumed run: 145.80) |
| central, cold, `RETIRE=0.95:3` | **720.44**, stopped by the progress rule | 6, 1.7 h | -- |

- Every run reached its value fast, then stalled. From round 3-6 on, every round ends on "stalled solve" (code 6, a collapsed radius; or 12, or failed subproblems), with t moving by 0.001-0.03 a round. The certificate agrees with t, so it is the inner solve that stops, not the exchange.
- The October 6 code went on from similar values: SISO 687.9 -> 674.0 -> 670.6 -> ... -> 632.2, half of those rounds ending stationary (code 0). The decentral optimum is a central design too, so central's optimum is at most 547.30, and the cold central run stalled at 654.57.
- Suspect: the trust radius after a partial step (`--sqp-partial-shrink`, section 5), the only one of the October 8 changes that alters the SQP's path. It helped the large early steps (sandbox: alpha 1/2 instead of 1/16-1/32 from the 804.77 checkpoint). In the tail, each partial step cuts the radius to 2 alpha |d|, and the line search no longer starts warm. The radius then reaches 1e-10 at the first kink the model does not see, and the round ends. Not yet proven: the JSONs' termination codes and final radii will tell. Since 20261008c it is off by default (`SHRINK=1` turns it on, directories `_shr`) for an A/B on the cluster.
- The JSONs (read 17:00) support the suspicion, though they do not prove it:
  - Termination codes: SISO 4, 4, 4, **6, 12, 12**, 4 and central `_ret` 4, 4, 12, 4, 12, 4. Code 12 means the subproblem failed even after a Hessian reset.
  - stat_value jumps to 1.1e7 (SISO round 3) and 1.6e7 / 1.1e17 (central rounds 3 and 5), from 9-38 before.
  - The October 6 SISO run had no code 12 in 19 rounds. Its stat stayed at 1.3-98 up to round 12, and its first code 6 came at round 13 (at 639.5).
  - One reading: with the radius cut to 2 alpha |d| at every partial step, the box becomes too small for the linearised stability rows. The elastic slacks and the steered penalty then grow until the QP fails.
- The A/B did not start. The two `SHRINK=0` jobs (765398 SISO, 765400 central) found the 20261008a runs' directories and refused, as designed. Only `SHRINK=1` SISO (765399, `_shr`) runs: 1078.44, 726.39, 704.29 after rounds 0-2. 20261008a had 691.79 after round 2, but 20261008c also has the slope sweep, so the paths differ.
- 765399 (20261008c, `SHRINK=1`, SISO) after 2.2 h: 1078.44, 726.39, 704.29, 689.10, 685.32, then 685.302, 685.283, 685.282 with "stalled solve" in rounds 5-7. No refusals in 8 rounds (the slope sweep added 5-13 members a round; |Omega| 59). The October 6 code was at 670.6 after round 5 and went on to 632.2. So the shrink stalls again, now with the missed-peak fix in. The `SHRINK=0` arm (three jobs, launched in a new directory on 2026-10-08 17:xx) is the comparison.

*20261008c, the default (`SHRINK=0`) against `SHRINK=1` (read 2026-10-08 22:00)*

| run | result | rounds, wall | inner codes | October 6 code |
|---|---|---|---|---|
| SISO, `SHRINK=0` (765568) | **641.52** | 13, 1.75 h | 0 in 8 of 13 rounds, 12 twice | 632.24 (19 rounds, 7.8 h) |
| SISO, `SHRINK=1` (765399) | 685.28, fixed point | 12, 3.2 h | 12 in every round from 5 on | -- |
| decentral, `SHRINK=0` (765570) | 605.97, progress rule | 6, 1.9 h | 4 in every round (budget), stat 11-36 | 547.30 (12 rounds, 7.6 h) |
| central, `SHRINK=0` (765569) | 474.85 after 9 rounds, running | 2.9 h so far | -- | (145.80: resumed from the reduced run's 173.77) |

- The A/B settles the trust radius. The same code and start reach 641.52 without the shrink and stall at 685.28 with it, where every round from 5 on ends on a failed subproblem (code 12). The shrink stays off.
- SISO without it: 641.52 against the October 6 code's 632.24, at 1/4.5 of the time, with 8 of its 13 rounds ending stationary on Omega (code 0). That is another local minimum, not a stall.
- Decentral: 605.97 against 547.30. The paths split at round 1 (623.94 against 602.0). Rounds 2-5 each gained 0.01-0.03 % with the inner budget spent (code 4, stat 11-36: not stationary), so the progress rule (3 rounds below 0.15 %) stopped it. The October 6 decentral run had a 3.9 % jump at round 4 (590.5 -> 567.6) that this one did not get.
- Central: 145.80 is not comparable to a cold start. It is the end of 49 reduced-model rounds (173.77, 632 members) followed by 15 lattice rounds. The cold central run went 1040.66 -> 576.98 -> 533.09 -> ... -> 474.85 in 9 rounds.

*Resume test and the cold central run to its end (read 2026-10-09 08:00)*
- 20261008c from the 145.80 record (job 765718, the October 6 run's directory copied, `RESUME=1`): 145.78, 145.75, 145.66, 145.63, 145.60, 145.59, 145.586 (rounds 0-6), then a fixed point after 10 rounds (5.4 h, |Omega| 674 -> 677, 11-13 sigma_2 blocks), winding count 0. The new code holds and improves the best central design: no regression in the local refinement.
- Cold central (765569) ended at **472.22**, a fixed point after 15 rounds (5.7 h, |Omega| 74). The reduced-model run from the SAME zero start (the October 6 run's first 44 rounds) went 1036.8, 773.1, 733.2, 733.2, 680.3, 585.8, 530.0, 460.5, 415.2, 381.9, ... 191.1, with 5-10 % a round for 12 rounds. The lattice run was ahead until round 5 (477.5 against 585.8), then stalled at 472. The starts are identical and the runs deterministic, so this is not luck: the lattice inner solve stops descending where the reduced one went on.
- The designs differ in kind. At 145.59 the actuators are at (2.64, 0.22) and (5.00, 0.05), near the wall and far downstream, with the sensors at 4.73 and 7.15. At 472.22 they are at (2.12, 0.87) and (2.97, 0.93), at mid-height, with the sensors at 3.70 and 4.39.
- The symptom: in the cold central run's late rounds most trials were rejected because they crossed the closed loop's line. The line was "not clear" in 62-77 of 141-144 evaluations (rounds 3-5) and 120-131 of 156-173 (rounds 13-14). A QP that saw every eigenvalue near the line would not keep proposing such steps.
- A suspect, measured on the central record 191.11 (the lattice line alone, serial). With one eigenvalue per disk centre (`LINE_K` 1) the line gives 8 stability rows. With two it gives 10, and the new pair -0.1444 +- 0.0446i lies 0.094 left of the line, BEHIND -0.0723 +- 0.0140i as seen from the line, with no row at k = 1. Same alpha and certificate; the line costs 3.2x (439 s against 139 s, serial, cold cache).
- Added (20261008e): the round's `lattice:` line counts the trials cut back inside the closed loop's line and, of those, the ones whose crossing eigenvalue had no stability row at the iterate the trial left (`_had_row`, since 20261009c `_offender`: within 0.05 of a row eigenvalue). `LATTICE_K=2` in the scripts (directories `_k2`) gives the two nearest eigenvalues per centre, each with its row.
- What is and is not affected (2026-10-09, after the user's question):
  - The certificate is not. The disk cover on Re = -eps proves that no eigenvalue lies ON the line; the winding count and the continuation gate do the rest. One eigenvalue per centre is enough for that, as in Schwerdtner et al. and in `stability_and_abscissa`, and the lattice line gave the same clear and the same alpha as the serial certificate in every check. Every gamma reported is rigorous, and so is the winding count; the line's alpha taken alone is not (correction, 2026-10-14: with one eigenvalue per centre, an eigenvalue right of the line behind another at the same Im is never the nearest to any centre on the line, so the line reads clear and alpha reads the other one; the gate, the fallback count and the save-time count are the guarantee; LESSONS.md 5).
  - The SQP's stability ROWS are, and that was a shortcut taken on October 6. The rows are linearised constraints for the eigenvalues that may cross within a step. The reduced path took one for every reduced Ritz pair within `--sqp-reach` 1.0 of the line (`near_line_pairs`). The lattice took one for each centre's nearest eigenvalue within 0.25. An eigenvalue behind another, seen from the line, then has no row: the certificate stays correct, but the QP cannot see it coming, its trials are cut, and the solve stalls.
  - The followed peaks are a discretised continuation. Each evaluation re-brackets the top from the last one, on finer and finer dyadic points down to 1.5e-5. The row is an exact sigma_1 at a point at most 1e-6 x sigma below the top (the reduced path's Newton: ~1e-15). The row point can move between lattice points from one evaluation to the next, so the SQP's merit carries noise of that size (~5e-4 at sigma ~ 470). That is harmless for the guarantee but can stop a line search whose decrease is smaller; not measured yet.
*20261009b on the cluster (jobs 766765-766767, read after 6-7 rounds, 2026-10-09 20:00)*

| run | rounds 0, 1, 2, 3, 4, 5, 6 | October 6 code | 20261008c |
|---|---|---|---|
| SISO | 1073.1, 757.7, 672.4, 659.2, 640.4, 636.1, 633.8 (3.1 h) | 1084, 824, 752, 688, 674, 671, 664 (632.2 after 19) | 1078, 766, 688, 676, 666, 666, 649 (641.5 after 13) |
| decentral | 820.5, 585.6, 581.2, 566.2, 565.4, 565.0 (2.8 h) | 935, 602, 596, 591, 568, 553 (547.3 after 12) | 935, 624, 606, 606, 606, 606 |
| central | 1047.7, 671.2, refused (663.6), 652.1, 647.0, 646.0 (3 h) | (reduced: 1037, 773, 733, 733, 680, 586, 530) | 1041, 577, 533, 524, 492, 478, 477 (472.2 after 15) |

- The strip did its job: no incomplete cover, 37-64 stability rows, and from round 3 on 0 of the trials cut back at the line came from an eigenvalue without a row (154 and 194 cuts in central rounds 2-3, 49-137 in decentral). So the stalls are not hidden eigenvalues any more: the trials cross the line with every eigenvalue's linearised row in the QP -- the eigenvalues move non-linearly over the step (or coalesce), and the strict admissibility rule (section 5) halves the step.
- Cost: 11-12 s per SQP evaluation (rows, line and strip; central round 2: 2464 s for 223) against 5-7 s with 20261008c; the strip takes 560-800 shifts per accepted iterate.
- Central round 2 was refused on a peak at w = 0.0656, 5 % above t, 0.0008 from the fixed member 0.0664. The in-solve sweep skipped tops within 1e-3 of ANY member, fixed or followed; now only followed members and their tops count. The interval [0, 1/32] is examined too: the slope is 0 at w = 0 by symmetry, and the test required it positive. A mock of that peak (half-width ~1.6e-3, between two sweep points 1/32 apart) is still missed, slopes or not: a peak narrower than the sweep's spacing needs the level-set check on the plant factors (to build).

- `testplant.slurm`'s build of the test plant's files in a fresh directory: the `flock` of 20261008d did not exclude jobs on other nodes on sator. Job 766730 hit "file signature not found" on mesh.h5 again; its rebuild-if-incomplete step could also remove a directory another job was writing. Since 20261009b the lock is a directory (`mkdir`, atomic on the shared file system; taken over after 3 h). The mesh and the operators are written under private names and renamed into place whole, with `HDF5_USE_FILE_LOCKING=FALSE`. Tested with three concurrent stub jobs: one builds, two wait and find the files.
- Built (20261009a, user 2026-10-09): the base grid of ranks/2 line centres, the strip [-eps - 0.25, -eps] covered at every iterate the QP reads, followed peaks by exact Newton (section 3). Checks, sandbox, test plant:
  - `tools/check_lattice.py` on the central record 191.11 (serial): fixed rows and sigma_2 blocks identical to the serial construction (0.0). The lattice Newton tops agree with the serial Newton's to 1e-14 in sigma. The line gives the same clear and alpha as the serial certificate (-0.0722913502), gradient of the rightmost 1.3e-10. The strip is complete with 25 eigenvalues; the independent search's 17 are all exhibited.
  - GL frozen path bit-identical (10.23128516441147).
  - k nearest eigenvalues per strip shift (central record, factor cached, serial; ncv 24): k = 1 2.35 s per shift, k = 2 11.0 s, 4 9.9 s, 6 10.2 s. The cover simulator with k nearest and a disk out to the k-th: 248, 176, 134, 104 shifts for k = 1, 2, 4, 6. Work: k = 1 583 shift-seconds, k = 4 1340, k = 6 1060, so k = 1 (user's guess).
  - A greedy smallest set of kept shifts (each covered cell lists the disks containing it) changed little: the cover needs ~200-250 disks of k = 1 for d = 0.25, whatever is kept. A large SQP step moves the eigenvalues far enough that the kept disks no longer cover: 513 and 800 shifts at the first iterates of the zero start (sandbox), 220-230 for a move of 0.004.
  - Smoke, central zero start, 2 ranks, cap 6, one SQP iteration then the round's end: certified 1446.392650 = the SQP's smax exactly (the exact Newton tops; the bracket left 1446.39 vs 1445.98 before); strip 513 shifts, complete, 26 stability rows; 5.5 GB on 7. With cap 8 and the side cache at 4 the sandbox ran out of memory at the round's end: the side cache is 2 (cluster: cap 24 + 2 = 26 factors a rank, ~3.5 GB on the test plant).
- Can a cached factor serve a nearby frequency (2026-10-09, sandbox, test plant, MUMPS, one core)? The factor of i w M - A was used as GMRES preconditioner for i (w + dw) M - A, solved to 1e-12 at w = 0.25, 0.75, 1.6. Iterations, one factor solve each:
  - dw 2e-5: 3-4; 1e-4: 4; 1e-3: 5-6; 5e-3: 7-9; 2e-2: 9-12.
  - A factorisation costs 0.5 s here, about 32 solves (0.016 s each).
  - A sigma_1 evaluation needs 18-34 solves, so at 4-12 iterations a solve, reusing a nearby factor costs as much as a new one or more. On the test plant, reuse pays only at the same frequency (the lattice), and an exact top is cheaper with its own factorisation. On the full mesh the factor-to-solve ratio is larger; not measured.
- The original Newton follow (`peakfollow.follow`) takes the slope exactly from the gain's own solves and carries the curvature (secant) between evaluations. In steady state that is 1-3 gain evaluations per member, each with its own factorisation. On 96 ranks, one member per rank, that is 1-3 waves of ~1 s, about what the lattice's waves cost (3.8 waves of ~2 s per evaluation, cold central).
- Retirement: the two central runs are identical through round 3 (720.44). After it, the `_ret` run retired 8 of 47 members, its round 4 stalled after 13 evaluations, and round 5 made no progress, so the progress rule stopped it at 720.44. The run without retirement went on to 703.83 and then 660.40 in rounds 4-5. One pair of runs, but a clean one: retirement made things worse here, as on the reduced model.

*Speed-ups (2026-10-08, sandbox, test plant; `tools/check_lattice.py` on the SISO record 752.96)*
- Serial, against the previous code (cap 10): fixed rows, sigma_2 blocks, followed rows, the line's alpha and clear, and the moved design's alpha are all **identical** (0.0). The followed rows land on the same lattice points: 26 evaluations in 3 waves, against 23 one after the other. On the moved design, the followed rows and the line ran in 2 waves together.
- 3 MPI ranks against serial: rows 7e-13 (sigma ~750), gradients 4e-12, followed rows 5e-13, alpha 5e-15. The previous code's 3-rank run gives the same numbers against its own serial run: rounding on the worker ranks, not the change.
- Synthetic peaks (`_follow` driven as `_drive` does; Lorentzian, Gaussian and tilted peaks, 3000 cold starts and 1500 x 8 drifting designs; the previous sequential walk for comparison). Waves per member: 3.85 cold, 2.68 drifting, against 10.6 and 7.5. Points: 11.5 and 8.7, against 10.6 and 7.5. Bracketed within the walk: 94 % and 95 %, against 62 % and 66 %. Largest deficit 3.5e-6 in both versions: the Hermite estimate, not the walk. A four-point jump (with a constant-step walk) cost 12.9 points for 3.62 waves, against 11.5 for 3.70 with two points, so two points.
- GL frozen path bit-identical (10.23128516441147).
- End to end (2 ranks, cap 8; the sandbox thrashes, so the times say little about the cluster):
  - SISO order 10, zero start, round 0: 1464.37 -> **1077.62** certified, in 23 evaluations and ~2000 s. The previous code gave 1091.07 in 26 evaluations and 2388 s. Iterations 0-4 are identical; from iteration 5 the followed rows sit on other lattice points of the same tops. The line took at most 1 wave per evaluation.
  - From the 804.77 checkpoint with `--sqp-partial-shrink on` and `RETIRE=0.95:1` (one round, to exercise the code): after the first step (alpha 1/16) the radius went 2.56 -> 0.32. The next steps were taken at alpha 1/2: smax 797.61 -> 755.06 -> 734.20 -> ... -> 665.78 in 10 iterations. The previous code from the same checkpoint (box radius 2.56 kept) went 797.61 -> 794.02 at alpha 1/32-1/16. Certified **666.04** (round 0: 35 evaluations, 2930 s). Retirement: 16 of 38 members below 0.95 gamma, retired -> |Omega| = 22.

*The level-set missed-peak check on the plant factors (20261009c, sandbox, test plant)*
- The pencil disk through the plant factor against the 2n factor (`SparsePencil.eigs_near`), central record 191.11, gam = 191.1286: the same nearest eigenvalues (0.0075 + 0.42193i pair at theta = 0.5, distance 0.07843055 both ways). The first block row of the eliminated solve has relative residual 4e-6 at theta = 0 (where 1 - sigma_1^2/gam^2 = 2e-4 amplifies), 2e-10 to 3e-15 elsewhere.
- Restarted GMRES on the Schur complement in the Euclidean product: 353 iterations mean, 734 max at theta = 0 (227 s for one disk); CG in the M semi-inner product: 10-11 (7.8 s). 5-8 elsewhere.
- Norm disks checked against ||G|| on their boundaries (8 points each, direct closed-loop solves at complex lam): below gam at theta = 0.5-10.
- `LatticeOracle.level` serially with the base grid of 96 ranks (48 points), central record: gam = t (1 + 1e-4): **clear**, 121 shifts (70 with the pencil), 3 waves (48, 63, 10), 742 s of serial work, cover kept 86; the same design again: 86 shifts (35 pencil), **1 wave**, clear. A first version left a 1.2e-6 sliver uncovered (the sliver rule, section 5). gam = t (1 - 1e-3) (first version, before the runs of centres were grouped): regions above found in the first wave at 0, 0.098, 0.4255, 0.789, 1.131, 1.316, 1.780, 2.354 -- the record's equal-ripple tops, all within 1e-4 of t.
- Zero start (SISO order 10, gam 1464.5): theta = 2.5-10 by norm disks alone (0.04-0.23, 1 s each); theta = 0.25, 0.75, 1.5 by the pencil (radius 0.20, 0.021, 0.49).
- End to end (SISO order 10, zero start, 2 ranks, cap 6, one round of 3 SQP iterations): the level set beside each accepted iterate's strip, 3 calls, up to 88 shifts (25 with the pencil) in up to 3 waves, none inconclusive, CG at most 7; no missed peak; the round certified 1452.32 against t 1452.45 (the certificate agrees with the rows). No trial crossed the line, so the crossing statistics stayed empty (`_cross_stats` checked on synthetic cuts). The same with `STRIP=0` (2 iterations): the same iterates, 4 evaluations instead of 8 (no second evaluation of the accepted iterate), 3 stability rows instead of 24, the level set alone (2 calls, 88 shifts, 3 waves).
- GL frozen path bit-identical (10.23128516441147).

*20261010c on the cluster (cold zero starts, 96 ranks, test plant; read 2026-10-10 22:50)*

| run | rounds (certified) | rounds / wall | stop |
|---|---|---|---|
| SISO, strip | 1061.18, 759.08, 675.24, 642.00, 641.74, 641.74, 641.71, (641.77) | 8 / 7.9 h | progress rule |
| central, strip | 1038.00, 708.09, 697.64, 697.64, 697.64, 697.64 | 6 / 5.8 h | progress rule |
| decentral, strip | 820.49, 589.60, 575.92, 567.71, 563.33, 561.24, 560.47, 560.44, 560.33, 560.19 | 10 / 7.5 h | progress rule |
| SISO, no strip | 1084.34, 873.23, 873.11, 873.10, 873.10, 873.10 | 6 / 4.2 h | progress rule |
| central, no strip | 1067.05, 698.85, 680.56, 679.76, then 3 rounds of 2 evaluations (code 12) | 7 / 2.8 h | progress rule |
| decentral, no strip | 793.40, 621.17, 572.71, 553.61, 552.47, 552.27, 552.21, 552.10 | 8 / 5.1 h | progress rule |

For comparison: the reduced path's cold central went to 191 in 44 rounds (October 6), SISO 632.2 and decentral 547.3 with the October 6 code.
- The level set did its job: no round refused, 0 inconclusive calls; `check_lattice` (job 767386, the 20261009b central record): rows and sigma_2 blocks identical to the serial construction, both level-set decisions agree with the 2n certificate (above the largest sampled top it found the region at 0.0745, sigma 618.53 against the sweep's 618.32; the 2n witness 0.0764), the line identical to the serial certificate (alpha -0.0500505066), the strip complete (44 eigenvalues, 36 rows). Its cost is high: 45-58 s a call (one wave: the slowest disk, ncv 100), 903 s of a 2300 s round when it ran apart from the strip evaluation (central, round 1).
- **The stall is the cut rule.** Every run stopped on the progress rule after 5-10 rounds at a plateau, with 26-128 trials a round cut back inside the closed loop's line. Two kinds:
  - SISO and decentral (from round 2-3 on; the same in the no-strip runs): the cut trials' eigenvalue sits at Re + eps = -5.00e-05 = -mg, exactly where its row put it (prediction -5.00e-05): the rows aimed at the admissibility line itself, the step landed on it to rounding, and the strict test (alpha + eps <= -mg) refused it and every halving of it. A rounding-level defect, now the cushion (section 5).
  - Central: the eigenvalue 0.13-0.15 past the line (up to 0.23), its row predicting the boundary, and in 60-90 % of the cuts no row eigenvalue within 0.05 of it at the iterate (it moved more than that in one step, or came from elsewhere). Halving does not bring such a crossing back (a coalescing pair moves like the square root of the step). The reduced path, which reached 191 from the same start, let the merit weigh such trials. Hence `--lattice-cross merit` (section 5); the first three crossings of each solve are now printed in full (eigenvalue at the trial, nearest row eigenvalue at the iterate, prediction) to see which they are.
- **And a third cause, the deepest, found by replaying the SISO record 641.71 in the sandbox** (`--continue-record`, `--sqp-debug-ls on`, the cushion and the merit rule already in): every trial was still refused, now by the merit, with no closed-loop crossing at all. The QP's compensator rows had dropped the real pole -0.05 (section 5, "Three compensator poles on the line"); each step pushed it across, its correction cost sigma 0.35-0.7. Fixed in 20261010d. The same replay with the fix: the real pole's row back (5 compensator rows), the penalty steered 100 -> 1000 by it, the correction back onto the line 1.3e-5 of a 3e-2 step (second order, as designed), and the first step accepted at alpha 1/2: 641.707 -> 641.359 (rounds 4-7 of the cluster run had gained 0.03 together); the next accepted at 1/8.
- The level set returned the region at 1.581 three times (one crossing seen from three centres, Im equal to 1e-10, not to the digit); crossings are now merged within 1e-8 and the tops within 1e-3.
- Strip A/B: undecided, both arms stalled by the same rule (SISO 641.71 with, 873.10 without; central 697.64 / 679.76; decentral 560.19 / 552.10). Without the strip a round costs about half (no second evaluation of the accepted iterate, no strip waves). To redo with 20261010d.

*20261010d on the cluster (cold zero starts, 96 ranks; read 2026-10-11 20:45)*

| run | rounds (certified) | rounds / wall | end |
|---|---|---|---|
| SISO, strip | 1062.85, 729.95, 675.97, 656.10, 649.40, 644.80, 641.18, 641.03 x 4 | 11 / 8.5 h | codes 12, 6, 6, 6 from round 7 |
| central, strip | 1042.66, 769.32, 672.77, 634.58, 634.27, 634.27, 634.26, 634.26, 634.26 | 9 / 7.8 h | code 12 in rounds 3, 5-7 |
| decentral, strip | 820.72, 617.25, 597.34, 597.34, 590.45, 573.60 x 5 | 10 / 6.1 h | code 12 at iteration 3 of every round from 6 |
| SISO, no strip | 1084.43, 853.62, 850.09, 849.47 x 3 | 6 / 3.5 h | |
| central, no strip | 1043.19, 939.85, 655.57, 654.33, 563.48, 414.67, 404.70, 389.15, 373.66, 370.44, 365.28, 364.92, **364.64** x 4 | 16 / 10.2 h | code 12 at iteration 1 of rounds 13-15 |
| decentral, no strip | 793.33, 609.84, 573.62, 554.31 x 4 | 7 / 3.9 h | |

- The compensator fix and the cushion took the stalls of 20261010c away (SISO kept descending to round 7, 641.03 against 641.71; central no strip went to 364.64, the best cold lattice result so far), but every run stops again, now on three things:
  - **QP failures** (termination code 12: the subproblem failed twice, the second time after a quasi-Newton reset), from round 3-7 on in four runs; deterministic (the same iteration every round). The statuses were not printed; since 20261010e each failure is, with the problem's size and coefficient range.
  - **Hypersensitive closed-loop eigenvalues near the line.** Crossings of 0.03-0.35 for box steps of 1e-3-1e-2 (central: 0.0693 + 1.302i at a trial 2.5e-3 from an iterate whose nearest row eigenvalue was -0.266 + 1.210i; SISO: the row predicted -0.0813, the trial had -0.0490). Random design perturbations of 2-norm 5e-3 at the decentral record move its rightmost eigenvalue by 0.035 (-0.0553 + 1.893i -> -0.0692 + 1.926i). The linearised rows then mislead the QP by more than any cushion of the order of mg.
  - **Leaving the line under the merit rule:** SISO round 7 accepted a crossing (6e-4 past), and in rounds 8-10 247 of 251 evaluations were not clear -- the solve never came back to an admissible iterate. Since 20261010e it returns to the last admissible iterate after `LAT_BACK` = 3 iterations outside, with a quarter of the radius.
- **The compensator's own poles coalesce on its line.** The records' compensators: central no strip (364.64) and SISO no strip (849.47) have a TRIPLE real pole at -0.05 (computed -0.05 +- 6e-5 i, -0.050002); central with strip (634.26) a double one; SISO with strip (641.03) a conjugate pair and a real pole on the line. At a triple pole the poles move like the cube root of a step: a perturbation of 1e-6 of the tridiagonal entries moved them 0.015-0.035, 1e-3 up to 0.13 across (measured on the central record's A_c). No per-pole linearisation and no 2 x 2 block (Hermite or the face S) describes that: replayed from the central no-strip record (`--continue-record`, `--sqp-debug-ls on`), the QP's compensator rows predicted -0.05000012 and the step put the abscissa at +0.0902; all eight halvings were cut, iteration after iteration. A 3 x 3 Hermite block on the cluster (tried in the sandbox, not kept) is hopeless there too: the block's eigenvalues are ~1e-4 in size, its scaling blows up, and the first-order model is wrong at a perturbation of 1e-6. A design question for the user: how the compensator's stability is to be constrained (or parametrised) so that coalescing poles stay smooth.
- Strip A/B, now with the stall rules gone: central 364.64 without against 634.26 with; decentral 554.31 against 573.60; SISO 849.47 against 641.03. Without the strip a round costs 40-60 % (no second evaluation, no strip waves). With the strip the QP carries 40-55 stability rows, among them the hypersensitive eigenvalues deeper in the strip, whose huge gradients box the step in.

*20261010e on the cluster (modal compensator, cold zero starts, 96 ranks; read 2026-10-13)*

| run | certified after rounds | rounds / wall | end |
|---|---|---|---|
| SISO, strip | 1104.25, 756.96, 607.13, then 606.8 ... 590.04, 577.58, 576.85, **572.09** | 19 / 6.2 h | QP NumericalError from round 13, fixed point |
| central, strip | 821.30, 491.87, 474.71, 450.49, 391.38, 391.38, 379.72, 346.76, 299.55, 296.44, 285.59, 285.41, **284.96** | 13 / 13.2 h | deadline (corrected 2026-10-14: not the progress rule) |
| decentral, strip | 1203.01, 730.15, 702.60, 544.31, 540.10, 538.47, 535.63, ..., **532.29** | 16 / 5.2 h | QP NumericalError / InsufficientProgress in every round from 1 |
| SISO, no strip | 1140.85, 761.94, 686.48, ..., 662.85, **649.82** | 23 / 4.1 h | QP InsufficientProgress from round 18 |
| central, no strip | 820.73, 578.19, 432.81, 402.18, 394.51, ..., **387.06** | 13 / 8.0 h | progress rule |
| decentral, no strip | 1203.30, 606.61 (refused), 680.27, 595.54, ..., 539.996, 534.35, **533.66** | 19 / 11.2 h | QP NumericalError from round 13 |

- The modal compensator took the merging-pole stalls away: every run went further than with 20261010d (SISO 572.09 against 641.03, decentral 532.29 against 554.31 and below the October 6 code's 547.3, central 284.96 against 364.64 -- the best cold lattice result so far, the reduced path's cold central went to 191 in 44 rounds). Crossings are rare now (9-44 printed in a whole run).
- **What stops them now: the QP solver.** Clarabel's NumericalError / InsufficientProgress at the tolerances 1e-10, on 770-900 constraints whose rows range from 1 (box) to 1e4-1e5 (sigma rows), with or without the closed-loop Hermite block; deterministic (the same QP at every round once it fails). Since 20261010f a failing QP is solved again at 1e-8, then 1e-6 (lattice only), and the first three failing QPs of a run are saved in the run's directory (`qp_fail_k.npz`: P, q, A, b, the cones) for an offline replay.
- Strip A/B: with the strip SISO 572.09 / 649.82 without, central 284.96 / 387.06, decentral 532.29 / 533.66: the strip pays in all three now, at roughly 1.5 x the wall per round.

*20261010e post mortem (2026-10-13: the logs, and the first failing QP of the decentral strip record replayed in the sandbox)*

- **The QP failures were the compensator LMI, not the row scaling.** With `--ctrl-modal` the Evaluator still built the Lyapunov LMI (`--sqp-lmi on`, the default), and with the LMI the per-pole rows leave the QP (`subproblem`: rows only without it) -- the modal rows of 20261010e were used by the second-order correction only. Every failing QP of the six runs had 770-900 constraints: 223 linear rows + three 20 x 20 PSD cones (the LMI and its trust region rho P +- dP, 630 rows), most with the closed-loop 4 x 4 Hermite block. The replayed QP (`qp_fail_1.npz`, decentral strip, iteration 1 of the continued record 532.29): NumericalError at 1e-10, 1e-8 and 1e-6; InsufficientProgress with static regularisation or more refinement; NumericalError at once with stronger equilibration. Dropping pieces: without the LMI cones it solves in 28 iterations (with or without the Hermite block); the LMI without its trust region: NumericalError; the LMI kept and the Hermite block dropped: InsufficientProgress. The LMI's data: Lyapunov P with eigenvalues 1.3e-5 .. 5 (poles ON the line next to poles far left), so the cones rho P +- dP are nearly degenerate, and -M0 has an eigenvalue -0.198 (the linearised LMI violated at d = 0 by 0.2, only the slack satisfies it).
- **Row scaling is not the problem.** The sigma rows' gradients range 1.2e-5 .. 3.7e3 (t = 532), the right-hand sides 5e-12 .. 521; on the LMI-free QP, dividing the sigma rows by t (systune-like normalisation) gives the same solution in 26 iterations instead of 28, and every row scaled to unit max coefficient changes nothing (28): Clarabel's own Ruiz equilibration already does it.
- **Fix (20261010g):** no LMI with the modal compensator; its two exact rows per block enter the QP as elastic rows (as `--sqp-lmi off` does for the tridiagonal compensator). Sandbox replay of the same record with the fix: iteration 0 the same (step 5.2e-2 against 5.26e-2), then iteration 1 solved and accepted where the cluster run failed (t 532.29 -> 531.15 after the in-solve sweep added two peaks; the record had stalled at 532.29 for 12 rounds); GL frozen path bit-identical.
- **Crossings (140 printed in the six logs):** 44 end within 2e-4 of the line, 38 of them had a row whose linearisation predicted -1e-4 (the cushion, 2 mg = 1e-4): curvature over a step, normal, and cheap (the trial is refused or the merit decides). 55 are more than 5e-3 past (up to 0.19); 57 had no row within 0.05 of the crossed eigenvalue (it came from further left, |step| 0.3-7.5 in the box norm): the step moved it by more than the linearisation of any row covers. 6 sit at -0.05000 exactly with no row: a closed-loop eigenvalue following a compensator pole onto the compensator's line (-0.05), whereas the closed loop must stay mg = 5e-5 inside it. Cost: 18 / 1182, 72 / 1580, 21 / 1019, 15 / 484, 96 / 1465, 231 / 1329 evaluations were crossing trials (SISO, central, decentral with strip; then without): 1.5-17 %.
- **Parallel efficiency (96 ranks).** The lattice dominates (central strip: 44 527 s of a 47 428 s run). The evaluation waves hold 26-64 tasks for 96 ranks (SISO 56, central 64, decentral 44 with strip; 26, 35, 37 without); each wave waits for its slowest task (wall = the critical path to 0.1 %); the level-set waves of the no-strip runs hold 49-74 tasks of 22-32 s each. So at least a third, and in the no-strip evaluation waves two thirds, of the ranks wait in every wave -- an upper bound on utilisation of 27-67 % before the spread of task costs. The ledger prints `ranks busy X %` per kind since 20261010g (task time summed over the ranks / ranks x wall; the sandbox's 2-rank line certificate: 93 %). Ways to use the idle ranks: the line search's trials alpha = 1, 1/2, 1/4 evaluated speculatively in the same waves (a rejected step costs no extra round trip), or fewer ranks per run and two runs per node.

*20261010g on the cluster (no LMI with the modal compensator; `CONTINUE=1` from the six 20261010e records, 96 ranks, 15 h; read 2026-10-14)*

| run | from (20261010e) | certified after rounds | rounds / wall | end |
|---|---|---|---|---|
| SISO, strip | 572.09 | 571.81, 570.77, 555.25, 555.16, 555.06, **554.97** | 6 / 2.0 h | progress rule; code 0 (stationary on Omega) from round 3 |
| central, strip | 284.96 | 231.16, 224.30, 206.07, 201.97, 195.68, 194.97, 192.99, **176.71** | 8 / 12.8 h | deadline; code 4 in every round, -8.4 % in the last |
| decentral, strip | 532.29 | 518.72, 499.87, 498.96, 483.34, 474.34, 464.04, 424.28, 418.69, 417.50, 414.53, **412.28** | 11 / 12.7 h | deadline; -0.5 % in the last round |
| SISO, no strip | 649.82 | 641.72, 624.95, 607.07, 603.85, ..., 598.39, **598.11** | 15 / 3.9 h | progress rule; code 0 from round 2 |
| central, no strip | 387.06 | 381.44, 355.44, 248.09, 245.09, 218.67, 194.07, 190.15, 187.08, 165.86, 160.67, 151.73, **151.48** | 12 / 12.8 h | deadline; code 4 in every round |
| decentral, no strip | 533.66 | 504.81, (430.89 not accepted), 461.62, 417.85, 393.07, 380.88, ..., 370.17, **369.87** | 16 / 13.8 h | deadline; -0.08 % in the last round |

- **The QP failures are gone.** Not one code 12 in the 68 rounds, against the 20261010e runs, which failed in every round once they had started to. Each of the 20261010e records that had stalled on the failing QP moved at once (decentral strip: 12 rounds at 532.29, then 518.72 in the first round). The LMI was the whole cause (post mortem above).
- **Central, no strip: 151.48**, the best cold lattice result. It is below the reduced path's cold 191.11 (44 rounds) and 4 % above the 145.59 that was resumed from the reduced run. The design is in the same basin as 145.59: actuators at (2.65, 0.31) and (4.99, 0.05) against (2.64, 0.22) and (5.00, 0.05); sensors at 4.76 and 7.21 against 4.73 and 7.15. The second actuator sits on the wall-clearance bound (`POS_MARGIN` 0.05) in both, and in the strip run too (176.71: (3.05, 0.05), (5.08, 0.05)). The lattice path and the reduced path now reach the same design from the zero start.
- **The four MIMO runs were stopped by the deadline, still descending** (code 4: inner budget, not stationary). Central with the strip gained 8.4 % in its last round. **Both SISO runs are stationary** (code 0 from round 2-3) at two different designs: (5.13, 0.64), xs 6.30 at 554.97 against (2.43, 0.71), xs 4.15 at 598.11. These are local minima, not stalls; the lever for SISO is multistart.
- **Strip A/B: no longer a clean comparison.** The arms started from different 20261010e records (central 284.96 against 387.06). Per round from those parents: central with the strip is ahead (round 7: 176.71 against 187.08); decentral without the strip is ahead (round 10: 371.63 against 412.28). A round costs ~1.6 h with the strip against ~1.07 h without (central). SISO: 554.97 with, 598.11 without.
- **Crossings: 201 trials across the line in 5463 evaluations (3.7 %).** Two kinds:
  - Decentral with the strip, from round 1 on: an eigenvalue pinned at the admissibility line. The row predicted -0.0501 (the cushion), the trial landed within 1e-4 of the line; 12 returns to the last admissible iterate in round 2. This is curvature at the scale of the cushion, and cheap.
  - Central with the strip, round 0: 28 crossings by real eigenvalues at -0.043 to -0.049, whose rows predicted -0.06 to -0.08. The linearisation errs by 0.02-0.04 over a step of 0.085: curvature again, larger.
  - A crossing eigenvalue with no row within 0.05 at the iterate: 24 of the 201, in four runs (decentral no strip 10, mostly at |step| 1.28; central no strip 6, SISO no strip 5, central strip 3; none in SISO or decentral with the strip). The six crossings at exactly -0.05000 with no row (a closed-loop eigenvalue on a compensator pole, post mortem) do not recur.
- Decentral no strip, round 1: the certified candidate 430.89 was not accepted (`gate_located` 1 in the record: the continuation gate). The log lines saying why were not in `SUMMARY.txt`. Round 2 then reached 461.62. `report.sh` now also picks up gate and winding refusals, INCONCLUSIVE certificates, and the lattice `[ledger]` line with `ranks busy %`. That line was missing too, so utilisation is still to read.
- The decentral strip record's compensator has poles at -0.05 (double) and -0.05 +- 4.26i, all on its own line: a resonance of the controller at w = 4.26 (sigma_1(K) ~ 37).
- Jobs 770564-770623 (the first submissions, cancelled within 20 min) and 770610 / 770614 (exit 255 after 5 min, no traceback in the summary) are not part of the results; the six runs are jobs 770628-770633.
- **Figures (user, 2026-10-14):**
  - *"The actuator positions in the MIMO cases depend on the mode frequency"*: they do not. Each run directory held one `modes_w*.png` of the parent record (the 20261010e design). Its closed-loop peak was at another frequency, so it had another file name, and nothing overwrote it. Example: decentral strip `modes_w0.204.png` shows sigma_1 = 532.29 with control, while the run's other figures show 412.28. The same happened in the SISO directories. `postproc.plot` now removes the mode figures that are not among the current record's frequencies.
  - The MIMO `controller.png` replaces the Bode plot of the entries with three panels: (a) sigma_1, sigma_2 of K(iw); (b) |v_1^H e_1|, the alignment with (1, 0)^T of K's optimal input direction (sensor space); (c) |u_1^H e_1|, that of its optimal output direction (actuator space). SISO keeps the Bode plot. Drawn from the four 20261010g records in the sandbox: the poles agree with the cluster's Bode plots. Central 151.48: sigma_1/sigma_2 ~ 70 at w = 0, and v_1 stays near sensor 1 (0.79-0.99) while u_1 turns from actuator 1 (0.67 at w = 0) to actuator 2 (0.06 at w ~ 1.4). Decentral: K is diagonal, so the alignments are 0 or 1 and switch where |k_11| = |k_22|.

*20261010g post mortem (2026-10-14; the full logs, nine analyses each checked by an adversarial verifier: LESSONS.md 19 and 20 hold the details and sources)*

- **"Only one active frequency" (central, no strip, 151.48) was a missed peak, not an optimum.** The SQP's last round pushed an equal-ripple plateau down to t = 144.774 (tops at 0.132, 0.835, 1.913, 3.078). Meanwhile the peak at 4.18 rose, unseen, to 151.4775. Its follower's window was centred on its base, the 2^-5 grid point 4.03125, so Newton stopped on the window edge 4.13125 (sigma 143.48, slope +284). The in-solve level set ran at the lagging epigraph t. It stopped at its first wave with a centre above that level, before it looked at that wave's crossings. Stage A of 20261010j fixes both, together with the defects the lattice path had and the reduced path did not:
  - no deduplication of followed members on one top (duplicate Newton chains were 25 % of the requested points in 770633);
  - certificate peaks enrolled at grid points instead of at their tops;
  - in-solve additions not followed after their round.

  Eight rounds of the six runs ended certified above t by more than 1e-3. All eight were accepted: they gained less than they claimed. Three or four were epigraph lag only.
- **The gate refused once in 68 rounds** (770633 round 1: a tracked real eigenvalue crossed to -0.04989; certified prefix: none). `--cont-accept-winding`, the GL fallback since STAGE5 22, had never been passed by `run_bfs.sh`. A count right of -eps at a design pinned 1e-4 from the line also needed an absolute step floor: the relative one was 4e-3 for MIMO. MIMO ran with `--wind-accept off` against the STAGE6 6.3 rule, and the save-time count at -eps/2 did not veto.
- **`CONTINUE=1` from a finished record lost the loop state.** Five of the six jobs restarted at inner budget 10 with no peak set; only central strip resumed a checkpoint. The record carried neither, and this confounds the strip comparison.
- **The strip A/B is still undecided.** The 20261010c/d/e pairs were each dominated by a defect fixed since (the cut rule, code 12, the LMI). In 20261010g the arms continued different parents. The cleanest datum, the decentral pair whose parents are 0.26 % apart, has no strip 10.7 % ahead at 12.5 h. Per trial, crossings are 6.0 % with the strip against 3.3 % without. The "no row within 0.05" count mostly reflects slides along the line and the step size, not hidden eigenvalues.
- **Where the time goes (96 ranks):**
  - In the no-strip runs the level set takes 54-65 % of the wall, with the ranks 8-12 % busy. A call is one wave of 37-78 disks for 96 ranks, waiting for its slowest pencil disk (ncv 100, 66-210 s).
  - The evaluation waves keep the ranks 27-42 % busy.
  - The driver's own serial work is about 0.4 %: the ledger's "driver" MUMPS counts were rank 0's share of the waves.
  - The deadline rule ended the MIMO jobs 1.2-2.3 h before their limit.
  - Measured memory is 3-3.9 GB a rank at cap 24. A 48-rank run would cost about 1.13-1.27 times as much per round, so two per node give about 1.6-1.8x more rounds per node-hour, once the line's base grid no longer depends on the rank count (it was ranks/2).
- **The figures:**
  - The actuator positions seemed to "depend on the frequency". That came from mode figures of the parent record left in the run directory (fixed in 20261010h).
  - The MIMO controller figure is now the singular values plus the alignment of its optimal input and output directions with (1, 0)^T.

*20261010j: what changed and how it was checked (sandbox: conda-forge DOLFINx 0.11, complex PETSc/MUMPS, 4 cores; test plant n = 33,593 rebuilt; GL frozen path 10.231285164411261 in this environment, from unchanged code, against the 10.23128516441147 of the older environments: compare against the unchanged code in the same environment)*

- **A, missed peaks** (`e70baab`, review fix `3b5544a`):
  - The in-solve level set runs at max(t, smax)(1 + tol). Every wave's crossings are confirmed before the search stops.
  - A region is "watched" only by a followed row inside it that reads at least its value; when all regions of a wave are watched, the waves go on.
  - Follower windows re-centre at their edge, at most twice. The reach is bounded from the base whatever the hint (review: a hint-dependent reach let one row alternate between 127.17 and 151.48 at the same design).
  - Followers on one top are deduplicated. Certificate peaks are enrolled at their tops.
  - The diagnosis lines read 't lag:' / 'missed peak:' / 'within tol:' instead of 'refusal:'. smax is printed beside t, the round's level decisions go into the history, and the record gains `active`.
  - Checks:
    - from base 4.03125 the follower now reaches 4.180605 (151.4775);
    - the level set at 144.7887 finds the 4.18 region in one wave;
    - `check_lattice` gives rows, sigma_2 blocks, line and strip identical (0.0);
    - a 3-iteration round from the 151.48 record: 151.48 -> 145.04 certified;
    - the GL frozen path is bit-identical.
- **B, stability guarantees, resume, provenance, deadline** (`b0a88ad`):
  - `--cont-accept-winding`, with an absolute step floor of 1e-6 for every BFS count right of -eps. The count at the decentral record 369.87 (pinned at -0.0501042) is 0 and conclusive: 5434 points, 1552 s on 4 ranks. With the old floor it was INCONCLUSIVE.
  - The save-time count vetoes (user, 2026-10-14). If the final design's count right of -eps is not a conclusive 0, the record falls back to the latest earlier accepted design whose count is (`--veto-counts`, default 3). Nothing unstable is ever saved.
    - 2026-10-16: a count that cannot start (the tail test fails before the march, `winding_cert.TailBoundError`) is no longer charged against `--veto-counts`. Every design is pre-screened by the instant tail test, and each outcome is classified and logged (zero, positive, inconclusive, pre-march, error). In 20261010n three such instant failures had used the cap, and the deadline stop then rewrote the checkpoint with the anchor (section 4, the tail ERRATA; section 5, the files).
  - Gate refusals are printed with their tracks and saved as `gate_fail_k.npz`.
  - An inherited anchor (checkpoint or record) is counted right of -eps: FATAL on a conclusive count > 0, logged when INCONCLUSIVE.
  - `CONTINUE=1` restores the inner budget, trust radius and peak set from the record, and the summary of a deadline-stopped run is kept.
  - `code_version` in the records.
  - The SQP stops on the job's deadline inside the round (`--deadline-reserve`, `--deadline-min-inner`). A cut round is exempt from the progress rule. `testplant.slurm` passes `--deadline-final 1200`.
  - Checks: unit tests of the veto (7/7) and of the gate dumps (15/15); three end-to-end resumes and continuations; the GL frozen and GL SQP paths identical.
- **C, efficiency and measurement** (`cd88648`, review fix `cdfee7d`):
  - `--lattice-base 48`: the line's and the level set's base grid no longer depend on the rank count. This is identical at 96 ranks and keeps the 96-rank geometry at 48.
  - A per-design request memo between a trial and its accepted re-evaluation.
  - New ledger lines: rank 0's in-wave share separate from serial work, the critical path per request kind, the factors and cache misses, a per-disk log of the level set (kind, ncv tier, Arnoldi applications, CG, seconds), per-member counts, and a projection to 48 and 32 ranks.
  - Duplicate stability rows are counted and tagged. Merging them (`--lattice-row-merge`) and measured task costs (`--lattice-costs measured`) stay opt-in (review, and LESSONS.md: measure first; placement by wall-clock seconds is not deterministic).
  - Checks: end to end against the previous commit at the old geometry, the 38 record arrays bitwise equal; `check_lattice` 0.0 on every saved number; GL bit-identical.
- **D, asynchronous level set** (`1cbeecf`, `--lattice-async off|on|redo`, default off):
  - In the no-strip arm the missed-peak level set runs on the ranks it needs, while the next SQP iteration runs on the others. Every state the look-ahead touches is snapshotted, and it is restored when the level set adds members.
  - Checks:
    - on against off: the SQP trace (every iterate, trial, t, smax, level decision) and the 38 record arrays are identical;
    - with injected regions the look-ahead is discarded and the results are again identical;
    - with redo, every look-ahead is put back and redone, again identical.
  - On 4 sandbox ranks it is not faster (no idle ranks). The gain is to be measured on the cluster at 48 ranks (`ASYNC=1`) before any default change.
- **E, in-solve additions followed** (`3e2b4a1`): a peak the in-solve level set adds is located at its top before enrolment (`LatticeOracle.peak_tops`). From the round's end on it is a followed member, moved to its in-solve top under A's rules. A round line counts the additions followed and merged. Checks: 20/20 unit checks, including equivalence with the old round-end block over 3000 random rounds; at the 770632 design the estimates 0.134766, 0.851067, 1.914894 and 4.13125 locate at the tops 0.132094, 0.834653, 1.912789 and 4.180605; a two-round run with provoked additions, the additions followed in round 1; async on gives a trace identical to off; GL bit-identical. Not changed: the reduced-oracle path (GL nominal) has the same defect. Records continued from 20261010g keep their earlier additions as fixed rows (about 35 in the 151.48 record), the same in both arms of an A/B.
- **Scripts:**
  - the layout of section 7 (`bfs/stage.sh`, `bfs/plant_id.py`, `bfs/lib.sh`, `bfs/home.sh`);
  - `report.sh` with `-f`, mirroring and log selection;
  - runs note their plant id;
  - `ASYNC=1`;
  - `bfs/abcopy.sh` for A/B arms from a shared parent.

*The next batch (20261010j; user, 2026-10-14: 48 ranks, two runs per node)*

- **Clean strip A/B and continuation**, all from one commit, submitted from `BFS/v/20261010j`.
  - Each 20261010g MIMO record is copied twice with `bash bfs/abcopy.sh`, one copy per arm:
    - central 151.48: `testplant_bfs/ab_c151_strip`, `ab_c151_nostrip`;
    - decentral 369.87: `ab_d370_*`;
    - central strip 176.71: `ab_c177_*`;
    - decentral strip 412.28: `ab_d412_*`.
  - The originals are not touched.
  - Every job: `CONTINUE=1 OUT=testplant_bfs/ab_<pair>_<arm> STRIP=1|0 ORACLE=lattice M=2 P=2 STRUCT=central|decentral OUTER=300 sbatch --ntasks=48 --mem=225G --time=15:00:00 bfs/slurm/testplant.slurm`, both arms of a pair on the same node kind.
  - The verdict, per pair: D(T) = ln(gamma_nostrip(T)/gamma_strip(T)), with T counted from each log's round-0 ledger stamp.
    - Read at T = 4 h, 8 h and at the end of the shorter arm.
    - The strip pays if D >= +1 % in at least 3 pairs and <= -1 % in none; no strip wins in the mirror case; anything else is undecided.
- **The asynchronous level set, measured:** one more job, `ab_c151_nostrip_async` (`ASYNC=1`), against `ab_c151_nostrip`. The iterates should be identical, so the comparison is wall time only, and also a cluster check of the identity.

*20261010j on the cluster: the plant id depended on the Python version (2026-10-15)*

- What happened (jobs 774312-774320, the batch of section 8 at 48 ranks):
  - every job computed the test plant's id as `coarse-7afd94cc112e1bbc` and found no files there;
  - `home.sh` had moved the 20261010g files under another id: it computes the id with the login node's `python3`, while the jobs use the conda env's Python 3.12, and `ast.dump`, which the 20261010j id hashed, differs between Python versions;
  - job 774317 rebuilt the mesh and the operators and started `plant_cert.py`; the eight others waited on its build lock, each holding 48 ranks, until they were cancelled.
- No run was touched: the jobs stop in the plant section, before any run directory is opened.
- Fixed in 20261010k:
  - the id hashes the source without comments and blank lines: the same under Python 3.11, 3.12 and 3.13, and equal to the id of the 20261010g tree;
  - `python3 bfs/plant_id.py --adopt coarse [DIR]` gives an existing plant directory this id, and lists the candidates without DIR;
  - a job that finds the plant incomplete while a live job builds it stops (exit 3) with the `--dependency=afterok:<job>` to resubmit with, instead of waiting on its node (`WAIT_PLANT=1` waits);
  - `PLANT_ONLY=1` builds the plant and stops;
  - complete files (operators, range and both pole files) are read without the lock;
  - `stage.sh` reports whether each plant is complete.
- Lesson (LESSONS.md 15): a content id must not depend on the interpreter. Compute it the same way on the login node and in the jobs, and check that before relying on it.

*20261010m on the cluster: one job per node, eight per user (2026-10-15)*

- The nine jobs of the batch (774339-774347, 48 ranks, `--mem=225G`): Slurm put two on each of three nodes. The second job on each of those nodes was cancelled 6 s in, before the site prolog wrote its header ("CANCELLED by 0", exit 0:54): 774342 (d370 no strip), 774343 (c177 strip), 774346 (d412 no strip). The six others run, one per node, half of each node idle. The nodes have 96 cores and 515 GB (`scontrol show node`), so it was not the resources: sator refuses two jobs of one user on a node. The QOS also runs at most 8 jobs per user (`QOSMaxJobsPerUserLimit`).
- So the plan's "48 ranks, two per node" needs both runs in one batch job: `bfs/slurm/pair.slurm` (20261010n).
  - It runs two `testplant.slurm` arms (`ARM_A`, `ARM_B`: each arm's own VAR=value words, OUT required) on one 96-core node, with 48 ranks each, each on one half of the CPUs, one NUMA side each (`bfs/cpu_halves.py`).
  - Every launch uses Hydra's fork launcher (`HYDRA_LAUNCHER=fork`: no srun step, so no repeat of the second-launch hang of section 8). The ranks inherit the half's `taskset` mask, and a check launch per half proves that before the arms start.
  - The plant files are checked once first (`PLANT_ONLY=1` on the whole node).
  - Each arm logs to `logs/bfs_testplant_<job>_<OUT's name>.log`.
  - Sandbox check (MPICH 5.0.1 Hydra, fake Slurm environment): without the fork launcher, Hydra tried srun and hung; with it, two concurrent 2-rank launches ran on CPUs 0-1 and 2-3; a failing arm's exit code reaches the job's.
- Found on the way: `testplant.slurm` printed no `-> DIR` line, so `report.sh` found no run directory of a testplant job. It now prints one, after the `PLANT_ONLY` exit, and `report.sh` also reads `=== RESUME: DIR/run continues` for the older logs.
- `stage.sh`: a directory that git does not list as a worktree, without a complete checkout (what an interrupted `git worktree add` left at `v/20261010m`), is moved aside as `<dir>.partial-<date>` and the checkout redone. A complete checkout that is not a worktree is refused.
- 20261010o-q: `report.sh PUSH=1` stopped at the commit with "fatal: name consists only of disallowed characters: ...". The clone's `.git/config` held `user.name ...`: the setup line of section 7 had `git config user.name ...` with `...` as a placeholder, and it was typed as written (the second placeholder typed literally, after `coarse-XXXX`: commands given to the user carry real values or none). A clone's setting overrides `--global`. EL8's git refuses the name; git 2.43 accepts it. Then the push asked for a GitHub password, which git over https refuses since 2021, and report.sh read the failure as a race and asked twice more; it now retries only when origin/results moved, and section 7 says how to push (SSH key or a token). The report itself was built (`for_claude.tar.gz`). `report.sh` now checks the name before it touches the results worktree and says what to set.

*The paired batch 774360-774364 at about 10 h (20261010n, report of 2026-10-16 09:24; jobs end about 13:30)*

- Five pair jobs, ten arms of 48 ranks, two per node (`pair.slurm`): each half's ranks checked on their own CPUs (48 + 48 python processes per node, by `ps`). Every arm resumed its parent's checkpoint with its loop state (inner budget 40, trust 0.32, peak set) and certified the inherited anchor (winding count 0 right of -0.05).
- No incident in any arm: no gate refusal, no veto, no FATAL, and no missed peak. The level-set notes read only "t lag" (certified = smax within 1e-14). Every arm also has 0-7 trials across the line, weighed by the merit. One QP gave up (async c151, round 6: NumericalError, then InsufficientProgress at 1e-8 and 1e-6) and was handled.
- **The strip A/B**, D(T) = ln(gamma_nostrip / gamma_strip), with T from each log's round-0 stamp:

  | pair | T = 4 h | T = 8 h | end of the shorter arm |
  |---|---|---|---|
  | central 151.48 | -0.18 % | -4.75 % | -3.55 % (9.3 h: 125.81 against 130.35) |
  | decentral 369.87 | -0.03 % | -0.03 % | -0.02 % (9.8 h: 368.43 against 368.51) |
  | central 176.71 | -0.21 % | -0.12 % | -0.11 % (9.4 h: 155.14 against 155.31) |
  | decentral 412.28 | +0.46 % | -0.44 % | -0.38 % (9.4 h: 391.67 against 393.15) |

  By the rule set before the batch (1 % in at least 3 pairs, and none against) this is **undecided**. No strip leads in all four pairs at the end, by more than 1 % only in central 151.48; the strip never leads by more than 0.5 %. The strip costs about 45 % more evaluations per round (123-135 calls against 83-95) and 10-25 % more wall per round, and it bought no stability here (no refusal, no veto in either arm). The asynchronous level set exists only without the strip: with it, no strip leads by 4.9 % (central 151.48, 124.06 against 130.35 at 9.3 h) and 0.14 % (decentral 369.87).
- **The asynchronous level set** (`ASYNC=1`) on the cluster: the trace of each async arm is identical to its synchronous partner's, every round line, t lag note and crossing line, over 6 rounds (central) and 8 rounds (decentral), as in the sandbox. Its rounds take 0.74-0.82 of the synchronous wall in central and about 0.70 in decentral (2760-3180 s against 3990-4450 s). The level set's own wall went from 2200-4400 s a round to about 60 s: it now runs beside the SQP. By 10 h the async arms are 2 rounds (central) and 4 rounds (decentral) ahead.
- **48 ranks against 96** (2 arms a node against 1 run a node, the partner's memory traffic included): round wall per SQP evaluation, no-strip arms in the 20261010g logs at 96 ranks against these logs at 48 ranks, decentral 31-40 s against 45-51 s (ratio about 1.26), central 57-67 s against 59-72 s (about 1.1). Evaluation waves take 2.1-2.6 s against 1.7-1.8 s. That is the 1.13-1.27 per round of the projection, so about 1.6x more rounds per node-hour. (The reading of 2026-10-16 that "1.8-2 h a round" meant a much worse ratio was wrong: these designs' rounds took 1.5-1.9 h at 96 ranks too.)
- Progress in 10 h: central 151.48 -> 124.53 (no strip; 123.94 async), central 176.71 -> 155.14, decentral 369.87 -> 368.43 (367.24 async, 367.77 strip), decentral 412.28 -> 391.67.

## 9. Open items

- **Starts below the open loop:** the zero-controller start works (test plant, cluster: 0.826 x the open-loop peak in 3 rounds, section 8). The certified prefix (section 5) is to be seen on the cluster rerun.

- **The disks of the gamma certificate** are certificates by exhibition: each comes from eigenvalues Arnoldi returned converged (residual 1e-9) nearest the shift. A set that does not converge is retried, then the interval bisected; nothing is assumed. The norm-based (eigenvalue-free) radius is rigorous but 1e-7 to 1e-10 here, so it cannot replace them (section 4).
- **Plant certificate on the full mesh:** about 2 h on 48 ranks projected from the test plant (41 + 36 min serial); not yet run.
- **Winding count cost:** about 0.6 s per frequency point on the test plant (3400 to 3600 points), dense near the 282 hot plant poles; on the full mesh the n LU is 14 times dearer. Over 23 workers (92 segments) the test plant's 2013 s of task time is 90 s of wall, so 20 to 40 min on the full mesh, twice per start (feasible start, end of run); to be measured on the cluster.
- **The paper's own domain** ([-20, 100] x [0, 20], n = 400,775) is larger than the archive's ([-1.5, 30] x [0, 3.5]); gains differ by orders of magnitude (fig. 2a ~2e3 vs 2.2e6 here). The devices and the floor positions are the paper's; the domain stays the user's.
- **The driver's own work** (30 % of a round on the test plant): the rows and the near-line tracking are on the ranks (section 8). What remains serial: the dense eigs of the reduced pencil (MIMO every evaluation, SISO every 10th), the basis updates and projection, the gate, the driver's MUMPS solves. To be measured on the cluster (`[ledger]`).
- **Reduced model vs full model in the inner loop** (2026-10-05; the full model is now built, `--oracle lattice`, section 3; to compare on the cluster). Near the optimum the needed accuracy is the per-round gain (< 1 %) over a plateau of active frequencies; the reduced model gives 0.4-4 % for moves up to 0.3, 22 % for large ones (section 8), and port-range bases do not compress (1000-3000 vectors). The full-model alternative: plant factors on Omega cached for the run (design-independent; 133 MB each on the test plant, ~1.5 GB on the full mesh), an SQP evaluation = solves only (sigma_1 with warm vectors: ~40 solves, 14 ms each on the test plant, ~0.2 s on the full mesh) spread over the ranks. |Omega| = 474: test plant ~3 s per evaluation on 95 workers (~5 min a round), 63 GB of factors; full mesh ~40 s per evaluation on 95 workers, ~10 s on 380 (~15-20 min a round), ~700 GB of factors (2 Sapphire Rapids nodes at least, 4 to be comfortable). Hybrid: the reduced model while rounds gain several %, full-model rows once the gains drop below the model's error.
- **The KS weight:** kept for later (user).
