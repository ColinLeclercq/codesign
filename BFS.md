# BFS.md -- the backward-facing-step fork

Every modification made for the backward-facing step (BFS), in one place.
Branch `bfs`, forked from `mimo` (2026-10-01).  The GL path is unchanged:
frozen path bit-identical (10.23128516441147) after every change below.

The case (user, 2026-10-01): SISO co-design on the Dergham-Sipp-Robinet
(2013) step at Re = 600, on the user's linear-analysis mesh and base flow
(`bfs/archive_dsr`, the user's zip).
- **Plant:** real.
- **Controller:** real, structured, order 5, proper but not strictly proper.
- **Actuator:** an isotropic Gaussian of std h/2 = 0.5 forcing the y-momentum only.
- **Sensor:** the same Gaussian, measuring u only.
- **Design variables:** the controller and the two centres (x, y), each free inside the domain.
- **Disturbance and performance:** momentum forcing in, kinetic energy out (the archive's weighted SVD).
- **Starts:** HIFOO style.

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

**Ports** (`raw`, `bc`)
- Each port is `g(x) = exp(-|x - x0|^2 / (2 sigma^2))` with sigma = 0.5 = h/2.
- Actuator: `psi_a = (0, g)`, forcing the v-momentum; `b = M psi_a`.
- Sensor: `psi_s = (g, 0)`; `c = M psi_s`, so `y = int g u`.
- Both are normalised to `||psi||_M = 1`; the controller absorbs the scale.
- Dirichlet rows are zero through M, so a port overlapping a wall loses that part of itself.
- `pos_dim = 2`: a position is (x, y), and the placement gradient has two components (`grads`).
- `Mdpsi` is memoised.

**Real system** (`real_plant = True`)
- `sigma(-w) = sigma(w)`, and eigenvalues come in conjugate pairs.
- Every frequency search, band and certificate runs on w >= 0, Im >= 0 (`hinf`, `alpha_certified`, the band).
- The packer's `real_ctrl` keeps only the real parts of (d, e, f, n, K) as design variables (`packer.py`).
- Complex arithmetic remains only where it is intrinsic: the resolvent at w != 0.

**Interior constraint** (`position_constraints`, `position_rows`) applies to each free centre:
- `x_lo + m <= x <= x_hi - m`;
- `y <= y_hi - m`;
- `clearance >= m`, where the clearance is the distance to the lower wall's tangent line at x. This is the true distance where the wall is straight; on the arc it is first order.
- The lower wall is: the plate y = 1 for x <= 0, the arc of centre (0, -1.5) with R = 2.5 for 0 <= x <= 2, and y = 0 beyond. The geometry is from `mesh_large.py`.
- `m = --bfs-margin`, default 0.25 = sigma/2.

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

## 4. Stability and the singular mass matrix

What GL used that needs M^-1, and what replaces it:

| GL ingredient | needs | BFS |
|---|---|---|
| plant enclosure / numerical range (`sparse_spectrum`, `stability_region`) | M^-1 | plant spectrum near the axis by shift-invert, once (`plant_spectrum`), cached for the certificate's region; NOT a proof that nothing is missed |
| free-radius bound of the level set, all-w tail bound | M-norm | none: gamma is SAMPLED (section 5) |
| winding count, final band check | the enclosure's tail | not run; the run prints so and the record says `all_w = {sampled: true}` |
| massops, `plant_rectangle` | M factor | forbidden (raise) |

**Plant spectrum** (`plant_spectrum`)
- Shift-invert Arnoldi on `OP = (A - s M)^-1 M`, at shifts i w_j on [0, band + pad], in the pool.
- The start vector is purified (`OP^2 v0`), so the infinite eigenvalues of the singular pencil stay out.
- Test plant: 95 eigenvalues; the rightmost is -0.1148 (the BFS at Re = 600 is globally stable).

**Stability of a design**
- Stability is established as at GL scale: the line certificate (nothing on Re = -eps over the band) plus continuation from a stable anchor (the watch list tracked at every accepted step).
- The anchor is the start of section 6, which is reached by continuation from the open loop.

**Certification, option B** (proposed 2026-10-01, no reply yet; proceeding with B)
- The level-set certificate does not work at this scale: shift-invert Arnoldi on the 2n even pencil needs 300-560 operator applications per shift away from the spectrum.
- So gamma is the sampled maximum: a full-model grid of the band, plus the reduced model's peaks, each local maximum climbed with the full model's sigma and slope.
- It is a lower bound flagged as uncertified.
- Option A, a certified level set with an M-free radius bound, remains open (section 9).

## 5. The driver

`run_case_epi.py --plant bfs --ops FILE` takes everything below. `bfs/run_bfs.sh`
holds the full flag set in one place: the GL production SISO flags plus the BFS
ones.

**Plant and start**
- `--plant bfs`: builds `BFSPlant` with no massops warm-up and no seed catalogue.
- The start comes from `--init` (d, e, f, nvec, K, xa(2), xs(2)), with no rescaling and no warm-start selection.

**Band**
- `[0, --bfs-band]` (default 8), plus the pad.
- `D.band_of` makes the stability certificate follow the run's band.
- The band is widened to the compensator poles only (no plant enclosure).

**Gamma**
- `--bfs-sampled N` (default 33): the sampled gamma of section 4. Its grid replaces the 241-point reference sweep.

**Reduced oracle input basis**
- Built from the ports' second-order jets (g, g_x, g_y, g_xx, g_xy, g_yy), via `BFSPlant.jet` and their M-images.
- Rebuilt at every round at the round's design, as the atoms' tangent basis is (`_jet_basis`).
- Without it the reduced pencil missed the true near-line pairs, and the full certificate ran every evaluation.

**Interior margin**
- `--bfs-margin` (section 2).

**Stopping before the time limit**
- `--deadline S` (general; default 0 = off): no new round once elapsed time + 1.25 x the longest round + `--deadline-final` (default 1800 s) would pass S.
- A job time limit then never cuts a round or the final save in half.
- A resubmission resumes from the checkpoint (auto-resume, unchanged).

**End of run and record**
- The M^-1 certificates are skipped with a message (section 4).
- The record stores 2-D centres as lists.
- The tag is `K_B_C_s0_bfs` (it was `_ne200`).

## 6. Starts, and the small test plant

**HIFOO draw** (`bfs/hifoo_init.py`)
- The `hifoo` arm of `tools/randstart.py` (STAGE2: the better arm at GL), made real.
- Every controller entry is N(0, 1): d, e, f, nvec and K.
- Each centre is uniform in the domain, at least the margin inside (rejection sampling).
- The catalogue is deterministic in (seed, n); the slurm array takes entry `SLURM_ARRAY_TASK_ID`.

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
   - Test plant, hifoo_000: compensator shifted by 1.18, gain x 0.018 to 0.020, alpha -0.054 to -0.053.
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
  - installs clarabel (the SQP's solver; `check_env.py` never checked it) if it is missing, and meshio and h5py for the test plant;
  - then imports everything the fork needs.
- `bfs/slurm/smoke.slurm` (1 node, under 1 h):
  1. the operators;
  2. the gains against the archive;
  3. `bench_mumps.py` (96 ranks factor and solve at once, which is the per-task cost under full-node contention);
  4. one HIFOO start on the test plant (feasible start plus 1 round) over 8 MPI ranks.
- `bfs/slurm/codesign.slurm`:
  - an array job, one HIFOO start per task;
  - 24 ranks per start (the driver and 23 workers), so slurm packs 4 starts per 96-core node; 120 GB per start;
  - 15 h limit, `--deadline` at 14 h 30, with 30 min reserved for the final save;
  - resubmitting resumes unfinished starts.
- The launcher is the conda env's `mpiexec` (`LAUNCHER=` to change).

**Run time** (measured 2026-10-02 in the sandbox; projected to the cluster)

*Ledger.* Each BFS round prints a `[ledger]` line:
- wall time;
- the driver's own MUMPS factorisations and solves (`bfs/mumps_lu.STATS`);
- the pool's task time, with what it would take on 23 and on 95 workers (`fullpar.CALLS` / `estimate(W)`: per call, `max(work / W, longest task)`).

*Measured on the test plant* (MPI, 2 ranks, one core; start hifoo_000):

| phase | pool task time | on 23 / 95 workers | driver MUMPS | driver other |
|---|---|---|---|---|
| plant spectrum + feasible start (15 steps) | 916 s | 167 / 167 s | 8 s | ~10 s |
| anchor + reduced-model set-up | 175 s | 22 / 20 s | 13 s | ~80 s |
| round 0 (gamma 1517.7 -> 1448.8) | 702 s | 103 / 103 s | 62 s (56 factors, 2153 solves) | ~62 s |

The pool's calls hold about 14 tasks on average (407 tasks in 30 calls), so
23 workers are as fast as 95. The longest task (the line-certificate segment
holding the eigenvalue, a slope climb) sets each call's time.

*Scale factor.* One rank, idle machine (`bench_mumps.py`):

| | factor | solve |
|---|---|---|
| n = 191,720 | 7.85 s | 0.17 s |
| n = 33,593 | 0.57 s | 0.012 s |

So every MUMPS-bound time scales by 14. The rest is the reduced model, the QPs
and length-n vector operations, and scales by between 1 and n/n_test = 5.7.

*Projection to n = 191,720, per start:*
- **Start-up:** 167 x 14 + 21 x 14 + ~90 to 500 s, about 50 to 55 min.
- **One round:** 103 x 14 (pool) + 62 x 14 (driver MUMPS) + 62 to 350 s, about 2400 to 2700 s, i.e. **40 to 45 min**.
- **Full-node contention:** four starts per node means 96 MUMPS ranks at once. Memory-bandwidth contention may add 10 to 30 % to the MUMPS parts. The smoke test's `bench_mumps` (96 ranks) measures it.
- **In the 15 h job:** about 50 min start-up, 30 min reserved, about **17 to 20 rounds** (15 with heavy contention). GL K+B+C needed 20 to 40 rounds to converge.
- **Expected outcome:** one job gives a substantially optimised design. Convergence (the progress stop: 3 rounds under 0.15 %) may need a second 15 h job, which resumes from the checkpoint.
- **Throughput:** 8 starts on 2 nodes; 16 starts on 4 nodes.

*The main serial cost left* is the driver's own MUMPS work, about 35 % of a
round: the slope climbs of the sampled gamma (sequential sigma and slope
evaluations), the eigenvector solves, and the SQP refinement's full-model
checks. Moving the climbs to the pool is the next speed-up if the cluster
numbers confirm this.

## 8. Validation log

- Frozen path (GL case B, ne = 100) is bit-identical, 10.23128516441147, after:
  - the DOLFINx/MUMPS switch;
  - the SQP position rows;
  - the deadline;
  - the pooled-sweep harvest;
  - the mpipool environment test.
- Operators and gains: section 1.
- MPI task farm, 3 ranks: identical to serial.
- Test plant, MUMPS, serial (1 core), from `bfs/data/init_test.npz` (a stable hand-made start):
  - plant spectrum 82 s;
  - round 0: gamma 2240.09 -> 1413.73 (about 10 min);
  - round 1: gamma -> 1398.64 (about 6 min).
- Test plant, HIFOO start hifoo_000:
  - serial: feasible start (gain x 0.0185), round 0 gamma 1520.36 -> 1447.08;
  - with the first homotopy (gain x 0.0201): 1449.65, then 1439.61 after round 1 (53 min on one core);
  - MPI, 2 ranks: feasible start (gain x 0.0194), round 0 1517.68 -> 1448.82, identical in two runs.
- MPI task-farm deadlock found and fixed: PETSc is now initialised on COMM_SELF (section 3).

## 9. Open items

- **Certification, option A or B** (section 4): awaiting the user's choice. Running with B.
- **Plant spectrum:** it is not a proof that no eigenvalue is missed (no M-free enclosure).
- **MIMO** (2 x 2, decentralised): not part of the fork yet (`bc` raises).
- **The KS weight:** kept for later (user).
