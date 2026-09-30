# TRACKER -- the plan versus what is done (updated 2026-09-28; rows marked [2026-09-25: ...] or [2026-09-26: ...] are superseded as stated)

**Short current-status view: STATUS.md** (stages, every note item, added items; one line each). This file keeps the full history.

One table per source document. Status words: **done** (built, measured,
recorded), **partial** (built or measured, not both, or with a caveat named),
**not done**, **superseded** (the plan's item was answered by a different
finding, stated). Every row names the evidence. Nothing is deleted from this
file; rows are updated.

## STATUS 2026-09-26 (evening) -- read first

**Where things stand.** Inner solver: SQP (Clarabel) with in-solve refinement,
closed-loop and compensator blocks, the compensator pair on its face S (entry
at separation <= 0.1, at a solve's start or after any accepted step), no
quasi-Newton memory across rounds, batched reduced rows and peak following
with shared-top restarts. Certification: sparse level-set certificate with
strict disks, full model sampled only at the reduced model's peaks, gate
sweeping only what the abscissa certificate did not, winding counts exact in
the compensator (STAGE5 29-30). Frozen path bit-identical (B ne=100
10.23128516441147). Results (K+B+C): **ne=100 6.399134 in 22.6 min** with the
scripts' stop (3:0.0015), 6.389368 in 32 min with 3:0.001 (best known);
**ne=200 6.486505 in 31.4 min** (6.484857 in 36 min), other known minimum
6.438335; certified over ALL frequencies at GL (tools/tail_certificate.py).
~100 full factorisations per round without a winding event (was ~190), ~70
with reduced-model certification every 4th round (option). Absolute floor
3.509193 (bounds.py on a 1401-point grid; was 3.509066 on a coarser one).

**Open before the cluster** (details in STAGE5 28-30):
1. the 27-case regression on the current defaults (cluster);
2. cavity port: operators A, M, b(x), c(x) from the user's solver, MUMPS via
   PETSc; LU/solve/memory on one node; the reduced dimension the cavity
   needs; a tail bound from the cavity operator's structure;
3. parallelisation: reduced rows over workers, the winding count (400-960
   factorisations per refused continuation, independent samples), the gate
   and certificate shifts (stage 6, with the `--rigor` segment bound);
4. reduced-model certification every k-th round: validated on GL (halves the
   factorisations, slower wall time at GL) -- decide at the cavity;
5. multistart over controllers (two local minima at ne=200; cluster);
6. field parametrisation for FB/FC (left as is at the user's request,
   2026-09-26; STAGE5 29.4). **[2026-09-28: done -- atoms / H^2 shape row,
   the unconstrained value withdrawn; STAGE5 32. Algebra item 11 done as an
   option, STAGE5 33. Next, before the cluster: 2 actuators and 2 sensors.]**
Everything else in this file is done or closed by a recorded decision
(STAGE5 30.10).

Order actually followed so far: Stage 0 -> "Stage 1" (constraint work, note
item 7 + start defects) -> Stage 1b (the handoff's Stage 1: items 0, 3, 11) ->
item 8 (inner solver, pulled forward by the kink evidence) -> items 4-5
(the handoff's Stage 3) -> Stage 5 (reduced oracle, K+B+C/K+FB+FC at ne=100
and ne=200) -> **now: preparing the cluster (list above)**.

## A. Handoff stages

| stage | plan | status | evidence |
|---|---|---|---|
| 0 harness + dump | harness, dump, plot, LU counter; gate: 27 gamma and gate chains bit for bit | **done** | STAGE0.md; 27/27 replay on the cluster; bit-identity across repeats, budgets, nodes, input builds (sections 15.1-15.3) |
| 1 solver settings (items 0, 3, 11) | see B; gate: balance residual on stalled rows -> O(10), three mechanisms attributed, 27-case regression | **partial** | STAGE1B.md. Items 0, 3, 11 done (B). Gate: met on the frozen designs by measuring correctly (rho_Clarke 0.58 / 0.55 / 0.27 -- the old 1e4 was one branch at a non-active frequency, STAGE1B section 3); mechanisms attributed (section 4). **27-case regression not run** (cluster job; one ne=100 case run, KSP+C 9.698 vs 10.151). Defaults frozen until it passes |
| 1 (as executed, off-plan) | constraint handling: compensator cap off, band from closed loop, selection defects, KGC start, degenerate-donor guard | **done** | STAGE1.md sections 1-12; gate job 16205236 passed on its criteria |
| 2 initialisation (item 1) | random draws, two arms, abscissa-descent phase, vs LQG seeds; K+B+C, K+FB+FC, 96 starts; gate: fraction stabilised, best gamma vs section 3, # distinct optima | **pilot done, one arm to rerun** (STAGE2.md section 2): 16+16 starts, 100 % stabilised on both arms, best 9.519 (HIFOO draw, beats the warm-started baseline 9.984), 16 distinct optima from 16 on both arms (rarefaction = identity), scaled draw worse than unscaled (median 33 vs 15). HIFOO arm was on pyGRANSO: rerun on the SQP before the cross-arm conclusion. 96 starts / K+FB+FC / cluster: not done | plan agreed with the reviewer's refinements (omega_* log-uniform per start, K_* from SISO loop gain, isolated starts, dense-abscissa phase with certified handover and cushion, cluster by transfer function, positions free in the phase); HIFOO + Lewis-Overton papers received; `tools/randstart.py`, `--stabilise`, multistart driver, `multistart_report.py` all to write. Pilot at ne=100 first |
| 3 working-set policy (items 4, 5) | A-N log-spaced seeding (omega-grid and level-ladder readings), retirement to |Omega| <~ 50, coarse-then-fine certification tolerance; gate: gamma unchanged to rtol, outer rounds and factorisations fall, 27-case regression | **partial** | STAGE3.md: `--omega-policy an` built (omega-grid reading), three defects fixed by running (soft cap, no skeleton, refuter retirement); exact accounting: 25 % cheaper per round, but |Omega| <~ 50 costs quality under a once-per-round exchange (10.16 vs 9.41 at 6 rounds) and a loose cap gives 9.52 at comparable cost. Delayed cap (user's suggestion) also negative (STAGE3 section 6): |Omega| <~ 50 is incompatible with once-per-round certification on GL; the machinery stays opt-in for Stage 5's projected oracle. Gate not met. Trust region (frozen ball) confirmed harmful under the iteration budget; short inner solves safe but slow; **adaptive inner budget (`--inner-maxit-adapt`, STAGE3 section 7) beats the frozen policy at equal cost on both kink cases: K+B+C 9.279 vs 9.410 at ~250 evals (8.953 at 443); KSP+B+C 9.449 vs 9.607 at ~270** -- ne=100 only. Not done: ne=200 for the adaptive rule, level-ladder reading, LU-weighted accounting, regression | **[2026-09-26: OFF, final; adaptive budget runs at ne=200 in the nominal scripts (STAGE5 30.10)]**
| 4 explicit constraints + inner solver (items 2, 8, 9), full-model oracle | per-eigenvalue margin rows; IPOPT vs active-set SQP vs GRANSO, restart overhead; Newton polish; gate: K+B+C rides the boundary unaided, KKT points, restart number decides | **partial** | item 8 built and measured (STAGE1B section 7): SQP with exact subproblem, block cones, Lyapunov LMI for the compensator. IPOPT not done (no pip wheel), restart overhead not measured, Newton polish not done. Item 2 not done (see B). The "K+B+C rides the boundary" gate has a new obstacle: the optimiser parks two compensator poles on the margin (section 7.1) | **[2026-09-25: the parked pair is handled by its face S, STAGE5 26]** **[2026-09-26: IPOPT, restart overhead, Newton polish dropped (STAGE5 30.10)]**
| 5 projection oracle (algebra items 8, 11, 12, 13) | V once per mesh; reduced inner with certified outer as enrichment; margin on reduced pencil; watch-list by interpolation; global opt for position cases; gate: identical gamma, 0 factorisations per inner iteration, 1e-5 reduced-vs-certified | **in progress** (user: Stage 5 first, 2026-09-18) | `reduced.py` built; correctness anchor reproduces the note's table (STAGE5.md section 2). **Phase 1 in the exchange (`--oracle reduced`): LU per inner evaluation 90-102 -> 24-29 (sigma rows 0; the rest is the abscissa certificate), witness agreement 1e-8..1e-14 (gate met), gamma within path variability (the exchange is chaotic in its surrogate: STAGE5 3.1).** **Phase 2 done: abscissa from the reduced pencil with the full-model residual filter; inner evaluation ~1 LU (from ~95); per round 450-611 (from ~4,000), the remainder being the outer certification (331 measured) -- STAGE5 section 4.** Not done: dim cap, coarse-then-fine certification, tail bound, one-sided bound, A-N inner on the reduced model, ne=200, second case. Cost model must use the measured LU ratio 12-25x, not the handoff's 5x (STAGE0.md); the analytic tail bound for the peak certificate belongs here (STAGE1.md section 12.3). **Design target from STAGE3 section 8: A-N-style inner loop on the reduced model inside an exchange certified on the full model; this answers the cavity memory question (|Omega| no longer costs factorisations in the inner loop)** | **[2026-09-26: all listed items done or closed: dim cap exercised, coarse-then-fine -> peak reference, tail bound (tools/tail_certificate.py, GL), one-sided bound dropped, SQP on the reduced model = the A-N inner, ne=200 and K+FB+FC run (STAGE5 30)]**
| 6 parallel certificates (algebra items 4, 6, 10) | parallel gate and winding, replay-validated bit for bit; critical path at 1/8/32/96 workers | **not done** | cluster only by nature; on the separate list |
| 7 cavity readiness (algebra items 1, 2, 3, 9) | coarse cavity / Stokes benchmark; subspace dimension, anchor LU, descriptor tail bound, P_plant recipe | **not done** | the compensator LMI (STAGE1B section 7) is r x r and carries over as is |

## B. Optimisation-note to-do items (0-12)

| item | text (short) | status | evidence |
|---|---|---|---|
| 0 | QP accuracy: OSQP tolerance x1e3 / qpOASES; log mu, regrow detector | **done, with a substitution** | STAGE1B section 4: pyGRANSO hardcodes 1e-12 (already tight); OSQP ends 60-80 % of steering QPs on its iteration limit; converged OSQP and DAQP (active set, pip) change gamma in the 9th-12th digit: QP accuracy is not the stall mechanism. `--qp-log`, `--qp-eps`, `--qp-maxiter`, `--qp-solver daqp`; mu logged, regrow detector in place (did not fire). qpOASES unavailable (no pip package); QPALM installed, not measured |
| 1 | random-start multistart with stabilisation phase | **in progress** | STAGE2.md |
| 2 | permanent per-eigenvalue rows for tracked near-boundary eigenvalues | **done in the SQP on the reduced oracle** (STAGE1B section 8); not in pyGRANSO | evidence attached: the certificate proves only "nothing ON the line" (STAGE1.md 12.1); compensator side done in miniature (per-pole rows, then the LMI, STAGE1B 7) |
| 3 | sigma_2 enrolled at active frequencies, hysteresis elsewhere | **done** | `--sigma2 on`; `gain_k`; STAGE1B section 2; terminal polish 7.9641 -> 7.9499 where sigma_2-off stalls at 7.957 |
| 4 | neighbour seeding: level-filtered log-spaced satellites per new peak (both readings) | **partial** | omega-grid reading built (STAGE3.md); level-ladder reading not | **[2026-09-26: OFF, final (STAGE5 30.10)]**
| 5 | retirement, |Omega| <~ 50; starter grid after the first certified round, dormant members after two | **done, measured once** | STAGE3.md: dormancy retirement, cap 50, floor 30; the literal round-count rule collapsed the surrogate and was replaced by dormancy |
| 6 | parallel gate validated by replaying the 27 chains | **not done** | Stage 6; replay tooling exists (harness replay) |
| 7 | controller-pole cap -> optional regulariser; band-safety certificate is the guarantee | **done** | STAGE1.md: `--ctrl-imcap off` + band from the closed loop; decision 2026-09-16 |
| 8 | inner solver, two levels: IPOPT and active-set SQP, restart overhead; qpOASES/QPALM; bundle Plan B | **partial** | STAGE1B section 7: SQP built (Clarabel, block cones, LMI), measured: better inner, worse partner for the one-frequency exchange. IPOPT, restart overhead, bundle: not done | **[2026-09-25: the SQP is the inner solver now; 'worse partner' was measured on stalled points]**
| 9 | Newton polish of the final balance system | **not done** | Stage 4 | **[2026-09-26: dropped (STAGE5 30.10)]**
| 10 | visualisation: Omega, sigma_1 on Omega, t, multipliers, spectrum | **done** | `plot_omega.py`, dump (Stage 0) |
| 11 | remove the level-scaling constant (= tau0, confirmed by the author) | **done** | STAGE1B section 5; both defaults 1.0, dead 1.05 removed |
| 12 | reduced model: random-start study free on it; global optimisation of the reduced problem for position cases | **not done** | Stage 5 | **[2026-09-26: positions are not the lever (the two ne=200 minima share them); random starts = cluster multistart (STAGE5 30.10)]**

## C. Findings outside the plan, and what they oblige

| finding | where | obligation |
|---|---|---|
| harness and frozen `case.slurm` post-hoc check ran at eps = 1e-3 | STAGE0/1 | fixed; `check_invocations.py` audit runs before every job |
| the "balance residual" of the Stage-0 table was one branch at a non-active frequency | STAGE1B section 3 | `stationarity.py` (rho, rho_Clarke) is the measure; the old column is retired |
| kink optima: 0 in the Clarke subdifferential needs the cross terms (PSD Y) | STAGE1B 3.1 | done for the TEST; the inner solve has the diagonal only (pyGRANSO) or the full block (SQP) |
| the optimiser parks two compensator poles on the margin; every constraint formulation fails at that point | STAGE1B 7.1 | the LMI prevents arrival; a design already there needs a start elsewhere -- relevant to Stage 2's phase and to item 2 |
| tight opt_tol is early stopping that regularises against overfitting Omega; the exact SQP overfits harder | STAGE1B 7.2, STAGE1.md 7 | items 4-5 (now); re-measure the SQP after them | **[2026-09-25: superseded: the SQP is the inner solver (STAGE1B errata)]**
| code 6 at ne=200 is real (Wolfe failure at the oracle's noise floor) | STAGE1.md 7 caveat | Stage 5 (oracle precision) |
| defaults were flipped without the 27-case regression; reverted | STAGE1B intro | the flip waits for the regression, one cluster job |

## D. Cluster-only list (deliberately not attempted in the sandbox)

- 27-case regression at ne=200 under the stage-1/1b configuration (one job; gates the default flip).
- Stage 2's 96-start arms (SLURM array; 32 per arm first, extend if the rarefaction curve has not flattened).
- IPOPT arm of item 8 (conda-forge cyipopt).
- Stage 6 in full.

## E. Two questions from the user (2026-09-19), tracked until closed

| question | test (`stationarity.py`) | SQP inner (`--inner sqp`) | default inner (pyGRANSO) | status |
|---|---|---|---|---|
| Q1: sigma_2 enrolled with the full 2x2 Clarke block (cross terms), not the diagonal | **yes** (PSD Y, STAGE1B 3.1) | **yes** (block as an SOC, FD-validated) | **no** -- two scalar rows, diagonal Y only | **closed in the SQP**, and the SQP is the inner solver for all development from 2026-09-19 (ties pyGRANSO at equal evaluations, STAGE1B 8.1); frozen default unchanged until the cluster regression | **[2026-09-25: 8.1 superseded; see K.5 and STAGE5 24-29 -- those comparisons were on stalled values]**
| Q2: the stability constraint in the gradient when active, ALL near-line eigenvalues (note item 2) | n/a | **closed loop: one row per residual-filtered near-line Ritz pair (reduced pencil); compensator: per-pole rows + LMI on the gate's line; certified validation of the returned iterate** | closed loop: rightmost only; compensator: single max-Re row (frozen) | **closed in the SQP** (= note item 2, on the reduced oracle); pyGRANSO unchanged |

Decision (user, 2026-09-19): the Stage-2 pilot is paused (HIFOO arm reported,
scaled arm 2/16, resumable) until both are closed. Plan: (1) the SQP on the
reduced oracle, re-measured under the adaptive budget (its earlier
over-exploitation verdict predates both); (2) per-eigenvalue closed-loop rows
in the SQP from the reduced pencil's residual-filtered Ritz values near the
line; (3) resume the pilot with the SQP inner.

## F. Exchange refinement inside the inner solve (opened 2026-09-19)

The bottleneck (STAGE1B 8.1): inner t 8.7-9.0 against certified 10.0-10.7,
every late round, any inner solver -- Omega gains one frequency per outer
round. Independent of the reduced oracle, which only makes the fix affordable:
on the reduced model a 241-point sweep costs ~0.1 s, so the SQP can refine
Omega at each accepted step (the A-N rebuild on the reduced model), with the
full-model certification unchanged outside. Decision: do this before resuming
the Stage-2 pilot (paused at scaled 1/16 on the SQP; HIFOO arm on pyGRANSO
reported, best 9.519, 16/16 distinct).

| item | status |
|---|---|
| reduced-model peak search inside the SQP; add missed peaks to the working set mid-solve | **built** (`--sqp-refine`; level-set certificate on the reduced matrices as the finder) |
| measure on K+B+C from K+B (SQP, reduced, adaptive, 8 and 14 rounds): gap between inner t and certified, gamma, evaluations | **measured** (STAGE1B 8.2): honest rounds (certified/t 1.00-1.01) but slower (10.15 at 306 evals vs 9.848 at 221 without); structural limit -- the reduced model cannot see peaks its basis does not span; opt-in |
| then resume the pilot, both arms on the SQP | **resuming** (SQP, no refinement) |
| three loop defects fixed on the way: abscissa rows/merit/admissibility on one line; exchange convergence requires inner stationarity under an iteration budget; frozen path bit-identical | done |
| remaining exchange lever: many cheap rounds (reduced oracle + adaptive budget); a cheaper outer certification (coarse-then-fine, tail bound) | coarse-then-fine done (374 -> 195 LU/round, STAGE2 3.5); certifying less often measured NEGATIVE (STAGE2 3.6: the reduced model's validity horizon is shorter than one SQP step); open: the tail bound, parallel shifts (cluster) |

## G. Decisions of 2026-09-20 and the K+B+C validation programme

- **Re <= -eps everywhere** (user): `--ctrl-line eps` sets the compensator
  line for the gate, admissibility, both inner solvers, the LMI and the
  stabilisation phase (it sets ctrl_max = -eps and the inner row's form).
  The frozen combination (inner -eps, gate 0) stays as `frozen` for
  bit-identity only. The 'gate' (Re <= 0) variant of 2026-09-19 is withdrawn;
  STAGE1B 8-8.2 numbers taken under it are marked superseded where they
  compare solvers.
- **No parametric studies in the sandbox** (user): the 32 x 2 convergence
  study, 96 starts, K+FB+FC and the 27-case regression are cluster jobs.
  Every remaining idea is validated on K+B+C (ne=100 first, ne=200 where the
  reduced oracle makes it affordable), one case, then handed over.
- The eight-round pilot (STAGE2 section 2) is an eight-round record, not a
  set of optima (correction in that section).

Programme on K+B+C, sandbox, in order:

| item | why | status |
|---|---|---|
| one start to convergence on both solvers, Re <= -eps, cross-polish each endpoint with the other solver | the stage-2 correction's test: correct local solvers may reach different optima by path but must not disagree at a stationary point | **done on one case** (STAGE2 3.1): four resumability fixes; SQP endpoint 9.748986 = a fixed point (six-way peak tie + compensator double pole on the margin); pyGRANSO from it does not move; rho_Clarke on the hull 3.4e-2 (no normal cone in the test). Open: KKT at a nonsmooth point -> bundle Plan B or the Newton polish (item 9) with all active peaks and the constraint | **[2026-09-25: 9.748986 was a cushion artefact (row H), not a fixed point]**
| cheaper outer certification (`--cert-sweep N`): coarse full sweep as the anchoring reference, fine reduced sweep for the peaks list, level-set certificate on the full model | the certification is ~330 factorisations a round, 241 of them the sweep | **built; measuring 61 vs 241 on K+B+C** (an earlier measurement was invalid: the warm start violated the new line and no phase ran; the driver now passes --stabilise for any start) | **[2026-09-25: done (row F); the 61-point fallback caused the fake witness, STAGE5 24]**
| ne=200 terminal polish with the SQP + adaptive budget + reduced oracle | the campaign's case; single core affordable now | **done** (STAGE2 3.7): 7.9620 at 170-274 LU/round; the SQP's steps refused by the certificate each round; the short-step sigma_2 polish's 7.9499 stands | **[2026-09-25: best ne=200 K+B+C is now 6.438335 (STAGE5 26), 7.96 was a stall]**
| Newton polish of the final balance system (note item 9) | **tried on the manifold** (STAGE2 3.4): equalises the guessed active set while an unlisted peak rises to 10.12 and a pole leaves the margin -- an unsafeguarded solve at a corner; the safeguarded form is the SQP with the block cone. Not a shortcut past the cusp on this case |
| basis-dimension cap: exercise it (it has not triggered) | Stage 5 | to test | **[2026-09-26: closed: exercised in every run since STAGE5 23 (compaction at 120)]**
| the note's one-sided lower bound ||T V|| | Stage 5 | to build, low priority | **[2026-09-26: dropped (STAGE5 30.10)]**
| IPOPT arm | needs conda (cluster); its role in item 8 ('two levels, let the number decide') is now filled by the Clarabel-based SQP -- an interior-point NLP would be a third inner solver on a loop whose bottleneck is the exchange, not the inner solve (STAGE1B 8.1-8.2) | **deferred to the cluster; not before the ne=200 polish** | **[2026-09-26: dropped (STAGE5 30.10)]**
| level-ladder reading of item 4 | the working-set experiments (STAGE3) showed |Omega| is not where the cost is once the reduced oracle exists, and satellites are not exploited between certifications | **dropped, with reason; revisit only if the cavity's certification cost makes rounds scarce** |
| bundle / gradient-sampling phase (item 8 Plan B) | both first-order solvers stall at the same nonsmooth point on K+B+C | raised in priority; after the Newton polish | **[2026-09-25: the common stall was the compensator pair + stale quasi-Newton memory (STAGE5 25-28), not a solver-class limit]**

## H. The compensator double pole (2026-09-20): decision

Finding (STAGE2 3.1): the 9.749 "fixed point" was the admissibility cushion
applied to the compensator (removed: the compensator's poles are exact, the
cushion belongs to the certified closed-loop abscissa only); without it the
SQP moves to 9.746695 and stops with both compensator poles EXACTLY on the
margin -- a nonderogatory double eigenvalue on the constraint boundary, where
the abscissa is not Lipschitz (square-root splitting) and neither per-pole
rows nor the linearised Lyapunov LMI is its subdifferential. Generic, not
luck: 3 of 3 compensator designs so far (frozen K+B+C, frozen KSP+B+C, this
random start) end with two poles coalescing on the margin (Burke-Lewis-
Overton: abscissa optima sit at Jordan blocks). Will recur at the cavity;
conjugate symmetry changes the pattern (pairs), not the phenomenon.

Decision: (1) reformulate the compensator constraint in the SMOOTH
coordinates -- Routh-Hurwitz / Bilharz polynomial inequalities on the
characteristic polynomial of A_c + eps I (exact on the boundary; a double
pole becomes a corner of smooth constraints) as SQP rows, gradients by finite
differences on the tiny polynomial; (2) gradient sampling at the endpoint on
the reduced oracle as the independent stationarity certificate; (3) the
Burke-Overton regular subdifferential as a test only. Then the Newton polish.

| item | status |
|---|---|
| Hurwitz rows for the compensator in the SQP; test at the 9.7467 endpoint | **done** (`hurwitz.py`, `--sqp-ctrl block`; whole-compensator Hermite numerically useless at degree 16, the 2x2-block version works; the SQP walks the corner: 9.7473 -> 9.7380). Cavity cost independent of n |
| constraint-aware KKT residual (`tools/kkt_residual.py`, hull + block PSD multiplier + abscissa) | **done**: 1.65e-2 x gradient scale at the endpoint -- not KKT, still descending |
| gradient-sampling certificate at the endpoint | **run** (`tools/gradsample.py`): inconclusive by construction -- the feasible set is a CUSP (every sample infeasible at every radius); objective-only says descent exists; the KKT residual stays the certificate (STAGE2 3.3) | **[2026-09-25: the KKT residual is not a certificate on flat optima (STAGE5 28)]**
| the 9.749 cushion artefact: fixed (admissibility, merit, LMI); frozen path bit-identical | done |

### I.1 Input-field basis (user, 2026-09-22) -- measured
Loewner: not pursued -- our projection is already tangential Hermite
interpolation from the operator; the failure is the parametric dependence on
the inputs, which parametric Loewner would face the same way.
n-width of the unit Gaussians (sigma = 0.4) over x_a in [-3,0] (M-norm POD):
9 / 14 / 18 / 20 modes for 1e-2 / 1e-4 / 1e-6 / 1e-8 (29 over [-3,3] at 1e-6).
Hermite functions at scale 1.5 sigma centred on the range: 30 modes -> 1e-7,
40 -> 3e-13. Taylor columns (b, b', b''): 2e-6 for a move of 0.01, 2.6e-4 at
0.05, 5e-2 at 0.3 (local only). Decision: (a) position cases -- enrich each
anchor with R(iw) Phi_b, R(iw)^H Psi_c, Phi/Psi = POD (or Hermite) modes of
the Gaussian family over the position range: triangular solves with the LU
held, no factorisations; (b) free-field cases -- the Hermite-function basis
as the design parametrization, truncated (replaces Sobolev regularization;
the reduced model becomes exact in beta).

| item | status |
|---|---|
| input-basis enrichment in reduced.py; test reduced-vs-full after position moves | **done** (STAGE5 section 5): union Galerkin + input basis, dim 98, worst error 2.6e-9..4.9e-8 for moves up to 1.0 (two-sided was 7.6e-3..9.2e-2) |
| wire into the loop (`--oracle-inputs`), rerun cert-every A/B and the ne=200 polish | next | **[2026-09-25: done (--oracle-inputs on)]**
| free-field (FB/FC) parametrization on a truncated Hermite basis | to do, after the test | **[2026-09-25: done with the Laplacian basis instead (row I.2)]**
| SQP ne=200 polish with a small uncarried trust radius | pending (environment reset 2026-09-22: torch, pyGRANSO, Clarabel reinstalled; gurobipy for pyGRANSO's import) | **[2026-09-25: obsolete]**


Environment reset 2026-09-22: torch 2.14, pyGRANSO (+tkinter patch in patch_pygranso.py), Clarabel/OSQP/DAQP reinstalled; init/seeds regenerated by tools/prepare.py; frozen path bit-identical (10.23128516441147). Inputs are no longer excluded from packages.

### I.2 Basis choices (user, 2026-09-22)
- Gramian / BPOD bases: not adopted. The performance channel is full-field
  (distributed disturbance to full state): its Gramians are full-rank, nothing
  to truncate; GL is unstable (coprime/frequency-limited variants needed);
  Gramians depend on b, c (same parametric problem, costlier); BPOD modes are
  localised (adjoint upstream, direct downstream) and bias exploration. BPOD's
  PAIRING (SVD of the adjoint-direct product) is a valid alternative to union
  Galerkin against rank loss -- half the dimension, but may drop interpolating
  directions; union Galerkin kept.
- Exploration: the ROM input basis does not restrict the design -- snapshot
  SVD of whatever family the design uses (any geometry), plus adaptive
  enrichment by R(iw) x (current field, recent step directions) for free
  fields; the per-round certificate adds what is missed.
- Design parametrisation on general domains: Laplacian (Laplace-Beltrami on a
  wall patch) eigenfunctions of the admissible region, computed once with the
  FE mesh -- the geometry-general Fourier/Chebyshev; truncation = hard
  version of the Sobolev penalty (same modes, weights (1+l^2 lambda)^s).
  Hermite functions on GL's line; Laplacian eigenfunctions for the cavity.

| item | status |
|---|---|
| ne=200 polish with --oracle-inputs on | **refinement closes the gap** (STAGE5 5.2): with `--sqp-refine 1e-4` the round the plain SQP lost (t 7.956 vs certified 8.489) is accepted at 7.9530, certified/t = 1.000; 1 % tolerance added nothing. Result: 7.9637 -> 7.9621 -> 7.9530 -> 7.9435 -> 7.9379, below the sigma_2 pyGRANSO polish's 7.9499 (best at ne=200 so far); full-model factorisations per round ~200, same as the plain arm -- the refinement is free (its apparent 500-1,200 were reduced-size splus; counter fixed, STAGE5 5.2.1). Earlier text: certify every round (7.9637 -> 7.9622 -> 7.9594 after 3 rounds; round 1's refusal was an Omega-gap the model saw exactly, not model error) and + in-solve refinement `--sqp-refine 0.01` (STAGE5 5.1). Certify-every-4 postponed: the refinement arm tests the sharper hypothesis | **[2026-09-25: 7.9x values superseded: stalls]**
| FB/FC parametrisation: truncated Laplacian eigenbasis vs Hermite vs Sobolev | **measured** (STAGE5 6): frozen fields are wide ([-24, 24]); Laplacian-Dirichlet 40 modes: field error 3.4 %, gamma +0.14 % without re-optimisation (20 modes: +2.2 %); Hermite wrong for wide fields. Next: packer basis mode + optimisation in the basis |
| adaptive input enrichment for free fields (R(iw) x recent step directions) | to do, with the above | **[2026-09-26: left: forcing parametrisation untouched at the user's request]**

| certify every 4th round WITH refinement and the position-robust model | **provisional rounds hold** (STAGE5 5.3): 2 of 2 re-certifications valid, claim above truth by ~7e-5; per-round cost itemised; two redundant full-model costs removed (memoised abscissa certificate, free lower bound). Running on; totals vs the every-round arm to come | **[2026-09-25: superseded by row 451 / STAGE5 14.2]**
| missed_peaks: return after a successful reduced certificate (the 1201-pt reduced sweep ran after every one: ~700 s/round wall at ne=200, no full LUs) | wall-time fix, math unchanged except the fallback | **done** 2026-09-22; running arms keep the old code for comparability |
| certificate tolerance at the polish: rtol 1e-4 under-reports up to ~1e-4; the SQP guard mixed models; a collapsed trust radius was carried | **fixed** (STAGE5 5.3.1): --cert-rtol set before the start certification (frozen 1e-4, polish 1e-6); guard on the SQP's own start value; trust floor. Numbers reported at 1e-4 are comparable only above ~1e-3 |
| SQP validation rejecting its best iterates | **explained and fixed** (STAGE5 5.4): alpha_certified returned the eigenvalue nearest the real point, not the attaining one (wrong eigenvector, wrong abscissa GRADIENT on every path, frozen included) -> `--alpha-argmax on`; the binding Ritz row (value within 2e-6) was dropped by a 1e-4 residual filter -> `--sqp-row-res-tol 1e-3`. **Proposal: include --alpha-argmax on in the cluster regression** (correctness of the constraint gradient) |
| packer/driver basis mode for FB/FC (`--field-basis`, `--field-n`) | **built and FD-checked** (STAGE5 6.1; 160 parameters instead of 1,596). Next: optimise FB+FC in the basis from the projected frozen design; compare with 4.3412 |
| clean ne=200 pair (every round vs every 4th, cert-rtol 1e-6, alpha-argmax, row filter 1e-3, refinement 1e-4; `tools/p200_pair.sh`) | running: every-round arm 7.9641 (start at 1e-6) -> 7.9622 -> 7.9617 | **[2026-09-25: finished/abandoned; superseded by STAGE5 14]**

## J. K+B+C status (2026-09-22) -- what is and is not concluded

Established:
- Best certified value at ne=200: 7.9379 (frozen 7.9641; previous best, sigma_2
  pyGRANSO polish, 7.9499), with SQP + Clarke block constraints +
  position-robust reduced model + in-solve refinement; certified/t = 1.000
  every round (at rtol 1e-4; difference > tolerance).
- Defects found and fixed: wrong-eigenvalue abscissa gradient (all paths,
  frozen included; --alpha-argmax), compensator double pole (Burke-Overton
  block), certificate tolerance at the polish (--cert-rtol), resumability,
  reduced model not robust to positions (input basis + union Galerkin).
- Cost structure: ~170-200 full-model factorisations per round (outer
  certification ~80, stability gate ~50, anchors ~14; SQP validation now
  memoised with the gate), independent of inner iterations; the inner loop
  costs none.

Not established:
- a CONVERGED endpoint at ne=200 (the polish was still improving);
- a CONTROLLED SQP vs pyGRANSO comparison: 7.9379 vs 7.9499 confounds solver,
  oracle (reduced vs full), refinement (SQP-only), flags and rounds; the two
  controlled tests so far (ne=100 equal evaluations; ne=100 cross-polish at
  the SQP's endpoint) show no difference. The SQP is more CORRECT (represents
  the Clarke block, per-eigenvalue rows, the compensator double pole);
  better OUTCOMES are not yet attributable to it;
- wall-clock: after the missed_peaks fix, 150 s per round inner wall ~= pyGRANSO's ~160 s (UPDATE: the following figures predate the fix) (SQP+refinement round 500-900 s inner wall vs
  ~160 s for a full-oracle pyGRANSO round; Python-level reduced-model work;
  one ~700 s/round waste removed, not re-timed); a GL factorisation costs
  ~1 ms, so GL wall time is the wrong measure for the cavity;
- whether certifying less often pays (clean pair running). **[2026-09-25: finished; see STAGE5 14.2]**

Cavity-scale factorisations: ~170-200 per round, 20-40x below the full- **[2026-09-25: measured: 172 per round over 35-69 rounds (STAGE5 28)]**
oracle loop and flat in inner iterations. ~20 rounds -> ~4,000; ASSUMING
~30 s and GBs per factorisation at 1e5-1e6 unknowns (not measured), >1 day
serial, ~hours with the parallel certificate (cluster item). Borderline.
Cuts in reach: memo (done, -24), shrink the 61-point reference sweep (the
position-robust model may make most of it unnecessary), certify every k-th
round (measuring), analytic tail bound; target ~50/round, the irreducible
part being the certificate (~15-30) and the stability line (~25).

| item | status |
|---|---|
| CONTROLLED solver comparison at ne=200: pyGRANSO and SQP on the identical reduced oracle, flags, rounds (refinement off for both, then SQP with refinement) | to run after the pair | **[2026-09-25: never run on the current code; the K.5 comparison was on stalled values]** **[2026-09-26: dropped (STAGE5 30.10)]**
| re-time an SQP round after the missed_peaks fix | **done**: 150 s inner wall at ne=200 (was 500-900 s; full-oracle pyGRANSO ~160 s) -- on GL now as fast as pyGRANSO, not faster; 166 full-model LU for start certification + one round (per-round increment to come) |
| reference sweep 61 -> ~10 points with the position-robust model: gamma and factorisations | to do | **[2026-09-26: done: --cert-sweep 0 samples the reduced model's peaks (~10-17 points); 192 -> 104 full LU per round with the gate fix (STAGE5 30.3)]**
| block-Hermite gradient by finite differences (inefficient, noisy on ~zero directions) | **replaced by the exact gradient** (STAGE5 7): 7e-10 agreement, 8x faster at K+B+C, zero cost for non-compensator parameters |
| cavity parallel footprint | **estimate** (STAGE5 7): ~3-10 GB per plant LU, ~24-48 frequencies/node throughput-limited, ~10-20 certificate pencils; **measure on one node**: tools/lu_footprint.py + tools/lu_concurrency.sh (cluster list) |
| per-round full LU at ne=200 | **~207 for the method** (certification 144, gate 25, anchors 22, other 16); the diagnostic dump had added |Omega| = 167 per round on the full oracle -- now on the reduced model (STAGE5 7) |
| CONTROLLED solver comparison (tools/fair_solver.sh) | **running**: SQP arm round 0 7.9622, round 1 refused (t 7.9599, certified 8.927: the Omega-gap, refinement off by design); pyGRANSO arm next | **[2026-09-25: not running; see K.5 (stalled values)]**
| K+FB+FC in the basis | truncation measured (STAGE5 6.2): 80 modes needed (+0.06 %), 40 modes +17 %; run queued: tools/fb_basis.sh (new method, 80 modes, vs frozen 3.925456) | **[2026-09-25: K+FB+FC ne=100 in the basis: 3.9012, not converged (STAGE5 28-29)]**
| refinement for pyGRANSO | **built**: `--pyg-bandmax on` = one row with the reduced model's certified band max (pyGRANSO cannot add rows mid-call); with-refinement pair scripted (tools/fair_solver_refine.sh) after the no-refinement pair (STAGE5 8) |
| long steps refused | **diagnosed** (STAGE5 9): near-maximal narrow peaks outside Omega (97 % / 96 % of max before the step); (1) enrolment cap of 4 peaks -> `--enroll-rel` (built, catches the +1.89 case); (2) drift of fixed-frequency members (-1.39 case) -> peak-following rows (to build, next) |
| certify on the reduced model, full model as a failing check on an adaptive schedule + at the end | proposal recorded (STAGE5 10); to build after the solver comparison |

## K. The nominal algorithm on solid ground (2026-09-22; user: reduced-model certification paused)

Working-set audit against the frozen snapshot (26 Aug): initial Omega = grid at
spacing 0.25 where sigma >= 2 % of max + the grid's local maxima (<= 160);
each round + the certificate's witness + its TOP 4 local maxima (241-pt sweep,
spacing 0.25); fixed-frequency members; no retirement. So the enrolment
weakness was ALWAYS there, not a degradation -- with one exception WE
introduced: `--cert-sweep 61` coarsened the full sweep to spacing 1.0, and at
the START (reduced model not yet anchored) the peak list came from it, missing
the near-maximal narrow peak at 1.91 (99.1 %) that refused round 0.
Note items 4-5 (A-N satellites, retirement) were built in Stage 3
(`--omega-policy an`), measured with the OLD model, never adopted.

Fixes (all behind flags, frozen path bit-identical):
| fix | what | status |
|---|---|---|
| `--enroll-rel 0.05` | every local max within 5 % of the max enrolled, located at its top (golden-section on the reduced model) | built, verified (catches 1.91) | **[2026-09-25: located by the safeguarded Newton now (row 453), not golden section]**
| start peaks re-derived on the anchored model | the start's peak list recomputed on the reduced model's fine sweep after round 0's anchoring | built (6 peaks, 5 new, incl. 1.911) |
| `--peak-follow on` | rows of peak members re-locate their peak's top at every evaluation (envelope theorem for the gradient); members moved to their tops after each solve | built |
| A-N satellites / retirement (Stage 3) under the current model | re-measure | to do | **[2026-09-25: measured in L; decision still open]** **[2026-09-26: OFF, final (STAGE5 30.10)]**

First run with the fixes (tools/nominal_test.sh; SQP, no refinement, ne=200):
round 0 7.961376 accepted, round 1 7.960147 accepted (certified 4e-4 above t),
against the unfixed arm's 7.9622, refused, refused, 7.9542, refused, 7.9485.
Running to 6 rounds; then the A-N policy, then the solver comparison on the **[2026-09-25: finished (K.1-K.4)]**
corrected loop.

### K.1 Progress of the corrected loop, and three record-keeping fixes (2026-09-22)
Rounds (SQP, no refinement, enroll-rel + peak-follow): 7.9641 -> 7.9614 ->
7.9601 -> 7.9593 accepted; round 3 REFUSED (t 7.9538 on Omega, certified 8.027)
although the reduced model agreed with the certificate AT the witness to
1.2e-14 -- the model knew the peak; no row sat on it. To be diagnosed by the
refusal log (below). Inner solves end on budget with small trust radii in
BOTH the corrected and the unfixed loop (not caused by the following rows).
Fixes:
- **resume state**: a resume reset the adaptive inner budget to its start
  value and dropped the SQP trust radius and the peak members, biasing every
  comparison that crossed a turn-boundary restart; the checkpoint now stores
  and a resume restores all three (sandbox-only effect; no resumes on the
  cluster);
- **self-contained dumps**: the dumped z is RELATIVE to the process's packer
  (scaled by the design the process started from); decoding a resumed
  process's dump with the frozen design's packer gave designs that never
  existed (my 8.30 / 8.40 "certificate discrepancy" was that, not a
  certificate failure). Dumps now store the unpacked design;
- **refusal diagnosis** in the log: witness, nearest member, distance,
  whether that member is a followed peak.
Frozen path bit-identical throughout.

### K.2 The corrected loop converges to a KKT point (2026-09-22) **[2026-09-25: all K.2-K.5 'KKT points' 7.95x were stalls of the compensator pair; see STAGE5 25-28]**
After 6 rounds (7.9641 -> 7.9614 -> 7.9601 -> 7.9593 -> refused -> 7.9573 ->
7.95721, gains falling to 1e-5): constraint-aware KKT residual (tools/
kkt_residual.py) = 2.4e-4 x the gradient scale (frozen KSP+B+C stationary
kink: 6e-4); 29 near-active peaks, multipliers on -0.529 / -0.500; one
compensator pole ON the margin (-0.0500000, Hermite lambda_min 4e-9). The
nominal corrected loop takes certified steps and stops at a stationary point
of the constrained problem. The unfixed loop (7.9485) and the refinement arm
(7.9379), from the same start and still descending, reached LOWER certified
values by long steps carrying them past this nearby KKT point: basin, not
correctness. Globalisation (multistart; long steps + refinement) is a
separate, deliberate choice.
Solver comparison on the corrected loop (tools/fair_corrected.sh): SQP arm =
this run (KKT at 7.95721); pyGRANSO arm with the same enrolment + following.

### K.3 Why round 3 was refused: the sigma_2 block rows did not follow (2026-09-22)
Reconstructed (reference chain decoded from launch 1's dump; every decoded
design re-certifies within 5e-6 of the log): witness w = -0.9612, 0.012 from
the followed peak member -0.949/-0.9507 -- but that member was sigma_2-
ENROLLED, and the block branch evaluated at the member's FIXED frequency (the
following code came after it). The sigma_2 policy enrols at every active
frequency, so the most active peaks were exactly the ones not followed.
Reproduction of that round (start design, start-of-round Omega, same
enrolment and followed members): block rows fixed -> t 7.956658, certified
7.965218 at -0.9469, REFUSED; block rows following -> t 7.955734, certified
7.955734, ACCEPTED (exact agreement).
Fixes: block rows (SQP) and sigma_2 rows (pyGRANSO) at the followed top;
members closer than 1e-3 merged after following (duplicates at -0.5292/-0.5293
made the QP rows nearly degenerate). Also noted, not changed: the sigma_2
policy enrolled -0.9507 with a sigma_1-sigma_2 gap of 0.61 (no kink) -- a gap
threshold would restrict blocks to real kinks (Stage-1b configuration kept).
The KKT point of K.2 (7.95721, residual 2.4e-4) stands: its certification and **[2026-09-25: a stall, not a minimum (6.438335 found later)]**
KKT test do not depend on the defect; the path to it did.
Solver comparison restarted on the fixed code (tools/fair_corrected2.sh).

### K.4 Corrected loop (block rows following) -- SQP arm (2026-09-22)
Rounds: 7.9641 -> 7.955184 -> 7.951836 -> (stationary, code 0) -> resume ->
7.951610. KKT residual at 7.951836: 1.87e-4 x gradient scale (7 near-active **[2026-09-25: a stall]**
peaks at -0.946, -0.529, -0.511, 1.898; multipliers on -0.529/-0.511;
compensator pole on the margin). Better than the pre-fix corrected loop's
KKT point (7.95721) with every step certified.
Defect found and fixed: the loop passed the SQP `t0`, the run's START
certified value, never updated; a stationary SQP returned it as t (7.964064
at a design certified 7.951836), which also blocked `--converge` (t must
agree with the certificate). The SQP now starts from the current certified
best; pyGRANSO path (t0 = a fixed scale) unchanged, frozen path bit-identical.
pyGRANSO arm follows the SQP arm in tools/fair_corrected2.sh.

### K.5 Solver comparison on the corrected loop (running) + pyGRANSO and the Clarke set
SQP endpoint 7.950855 (KKT residual 2.0e-4 x scale, 11 near-active peaks).
pyGRANSO, identical settings: round 0 7.962768, round 1 7.961807 (SQP:
7.955184, 7.951836); both take certified steps (t vs certificate <= 5e-6).
pyGRANSO and Clarke (user question): its inequality constraints are SCALAR
functions, possibly nonsmooth, one gradient each per evaluation. It uses the
Clarke subdifferential implicitly (BFGS across kinks; stationarity test =
min-norm element of the hull of recent gradients), not in the STEP: its QP sees
one gradient per constraint (+ our sigma_2 rows: the diagonal of the 2x2
Clarke set, no cross terms). Ways to plug the block in: (1) smooth scalar
reformulation of the 2x2 block (tr >= 0, det >= 0 of t^2 I - B^H B, bases
fixed per evaluation) -- exact and smooth off coalescence, but det's gradient
vanishes AT an exact coalescence optimum (constraint qualification fails where
it matters); (2) several gradients per constraint -- not in pyGRANSO's API;
(3) cones in pyGRANSO's QP -- makes it the SQP. Optional test of (1) listed.
| item | status |
|---|---|
| pyGRANSO with the (tr, det) block reformulation | optional, after the comparison | **[2026-09-26: dropped (STAGE5 30.10)]**
**Result (K+B+C ne=200, corrected loop, identical settings, no refinement):** **[2026-09-25: both solvers were stalled at these values]**
| | after 1 round | after 2 | current | KKT residual (x gradient scale) | full LU / round |
|---|---|---|---|---|---|
| SQP | 7.955184 | 7.951836 | 7.950855 (5 rounds) | **2.0e-4** (stationary) | ~150 |
| pyGRANSO | 7.962768 | 7.961807 | 7.961104 (4 rounds) | **3.4e-2** (not stationary) | ~150 |
pyGRANSO is stuck against the compensator DOUBLE POLE on the margin (poles
-0.0500 / -0.05001, Hermite lambda_min -5e-9): its frozen single row
max Re(pole) <= -eps is not Lipschitz there (Burke-Overton), every step that
improves sigma_1 splits the pair, the line search makes crumbs of progress.
The SQP's 2x2-block Hermite constraint is smooth there and reaches a KKT point.
**The SQP is better on this case, for a structural reason, measured at equal
cost per round.** (pyGRANSO could be given the same smooth block constraint
as two scalar rows -- trace/determinant of the block's Hermite form -- a
listed option; it would then still lack the sigma block's cross terms.)

## L. Note items 4-5 re-measured on the corrected loop; K+FB+FC in the basis (2026-09-22)
Policy implementation fixed before measuring: its sigma evaluations went
through the FULL closed loop (|Omega| + satellites factorisations a round) --
now the reduced model; it re-certified the accepted design every round
(~150 factorisations) only to get its peak list -- now kept from that design's
own certification; retirement AND the cap could remove followed peak members
-- now protected.
First attempt (cap unprotected): round 1 t 7.9278 on |Omega| = 49, certified
**40.99** at w = 3.4885, nearest member 0.49 away (the 3.356 peak member had
been capped away): a 50-member set leaves gaps the SQP step exploits.
With peaks protected (tools/policy_and_fb.sh, cap 50, skeleton 32):
round 0 7.955184 (= no policy), round 1 7.953120 accepted on 49 members
(no policy: 7.951836 on 163). Running.
**Item 1 result** (converged; tools/policy_and_fb.sh arm 1): **[2026-09-25: 'converge' = stall; the OFF decision stands only on its cost/risk argument]**
| | converged gamma | KKT residual (x scale) | full-model LU / round |
|---|---|---|---|
| corrected SQP loop, no policy | 7.951836 -> 7.950855 | 1.9e-4 / 2.0e-4 | 128-171 |
| + A-N satellites + retirement (cap 50, skeleton 32, peaks protected) | 7.953104 | 3.6e-4 | 149-169 |
Both converge to KKT points; the policy lands on a neighbouring one, 2e-4
relative higher. Same full-model cost: with the reduced model the per-round
cost is the certification, independent of |Omega|, so retirement's purpose
(cheaper inner evaluations) is already met, and satellites' purpose (drifting
narrow peaks) is met more directly by following; what remains is the risk
shown by the unprotected first attempt (certified 40.99 vs t 7.93). Decision
proposed: items 4-5 OFF in the nominal algorithm; opt-in for a full-model
inner loop, where |Omega| costs factorisations. Caveat: the policy's own
bookkeeping (dormancy, ages, refuters, starter grid) is not in the checkpoint;
a resume restarts it.

**Item 3 (K+FB+FC, 80 Laplacian modes) -- a start defect found and fixed**: the
warm start hands the packer the NODAL path's pre-Sobolev coordinates xi
(va = smooth_map(xi), sob_len 0.8), not the field; basis mode projected xi as
if it were the field and started from a different design (certified 4.805183
instead of 3.927660). Basis mode now projects smooth_map(xi); the start
certifies at 3.927660, exactly the truncation test. Frozen path unchanged.
Start peaks on the reduced model: 9 within 5 % (-6.07 ... 8.41). Running
(tools/fb80.sh), reference: frozen nodal optimum 3.925456.
| stage files | **updated 2026-09-22**: STAGE5 sections 11 (working set corrected), 12 (reduced-model certification status), 6.3 (K+FB+FC model size); STAGE3 section 9 (items 4-5 re-measured, recommendation OFF); STAGE1B 8.3 (SQP vs pyGRANSO on the corrected loop) |
| reduced-model certification | reconsidered (STAGE5 12): proposed as a measured experiment on K+B+C after K+FB+FC, pending the user's agreement | **[2026-09-25: done: row 451]**
| reduced sigma_1 evaluation O(dim^3) | **fixed** (STAGE5 13): QZ once per basis + block Woodbury + warm-started subspace iteration for the leading singular values; exact to 1e-14 (values) / 1e-11 (gradients); 3.9x per row, 10.6x for peak following, ~5x per K+FB+FC inner evaluation |
| stage-file audit (user, 2026-09-23) | **done**: ERRATA blocks prepended to STAGE1, STAGE1B, STAGE2, STAGE3, STAGE5 listing every superseded conclusion, the correct reading and where it is established; STAGE0 unaffected |
| SQP evaluation cost, K+FB+FC | **reduced further** (STAGE5 13.1): 2.7 s -> 1.9 s per evaluation along a path at dim 232 (shift-invert near-line eigenvalues, warm narrow peak following, overheads, row screening); remaining cost parallel across rows |
| reduced-model certification on K+B+C (`--cert-every 4 --prov-check 0 --cert-fail continue`; convergence only on full rounds) | **running** (tools/romcert_then_fb.sh), vs results_corr_sqp | **[2026-09-25: finished: row 451]**
| shift-invert near-line eigenvalues | **reverted to OFF**: wrong left-vector pairing -> exploding abscissa gradients -> SQP NumericalError (STAGE5 13.1 correction); fix listed (inverse iteration for left vectors); ARPACK failures now fall back to dense |
| shift-invert near-line eigenvalues, second attempt | left vectors by inverse iteration; still 8 vs 7 rows and a gradient mismatch against the full eig: **not adopted**, OFF (STAGE5 13.1) |
| reduced-model certification, K+B+C | **provisional chain confirmed by the full certificate to 1e-7** across a 0.35 % jump (7.922910 vs 7.922909); first small model discrepancy after a large step (0.19 %, caught); running (STAGE5 14) |
| cavity scaling of the reduced certification | cost-weighted ~1.9x (as implemented), ~2.8x with the stability gate reduced in provisional rounds; more on the critical path (STAGE5 14) |
| near-line eigenpairs at the cavity | **done** (STAGE5 15): Rayleigh-quotient tracking with O(k^2) solves via the plant QZ + exact rank-2 Woodbury; same rows as the full eig, values 1e-10, gradients 1.2e-8; 15x at k = 232, ~60x (k=1000) to ~130x (k=2000) extrapolated; on above k = 150 |
| lu_counter | reduced-model factorisations tagged explicitly (REDUCED_CONTEXT), no longer by size |
| provisional rounds paid for the full certificate (early return placed after it) | **fixed** 2026-09-24 (STAGE5 14); provisional cost to be re-measured; evidence so far unaffected |
| stability gate on the reduced model in provisional rounds (`--prov-gate reduced`) | **built** (STAGE5 14.1); full stability re-certification of the provisional chain at full rounds, rollback on failure; running in the K+B+C experiment |
| provisional round cost with the reduced gate | **22 full-model LU** (vs ~165 full round): cavity estimate ~3x total work over a 4-round cycle, more on the critical path; chain's full re-certification pending | **[2026-09-25: still the cavity estimate]**
| reduced gate, first full check | **passed**: sigma 7.849747 vs claim 7.849748; alpha -0.0542 certified; 3 chains confirmed so far (STAGE5 14.1) |
| reduced-model certification, K+B+C | **concluded** (STAGE5 14.2): converged to a KKT point (7.848591, residual 3.1e-4); 3 chains confirmed, no failure; ~2.6x fewer factorisations / ~3x cost-weighted per 4-round cycle at the cavity; recommended as an option to validate in the cluster regression | **[2026-09-25: 7.848591 was a stall; the reliability and cost findings stand]**
| K+FB+FC, 80 Laplacian modes, current code | **running** (tools/fb80.sh), reference frozen 3.925456 | **[2026-09-25: 3.9012 at ne=100, not converged]**
| peak following by golden section | **replaced by the frozen safeguarded Newton** (STAGE5 16): exact tops, 7-15 calls cold, 1-2 iterations warm; evaluation at the top reused for the row |
| debugging at ne=100 | **adopted** (user): tools/transfer_ne.py; K+B+C 10.990169, K+FB+FC 3.935632 at ne=100; corrected loop running on both (tools/ne100_nominal.sh) |
| closed-loop eigenvalue coalescence at the margin | known gap in both certification regimes: per-eigenvalue rows only (eta -> 0); a 2x2 block like the compensator's would be the fix; not hit so far | **[2026-09-26: detector added (closed_loop_pair_rounds), 0 in every run; the closed-loop Hermite block covers coalescence off the line (STAGE5 30.7)]**
| ne=100 K+B+C: SQP rows 0.4-4 % below the certificate (rounds 4-7) | model exact at the witnesses (1e-14) and across the band (rebuild) -- NOT the anchors; window scan added to following; cause OPEN: replay round 5's SQP with per-row logging (STAGE5 17) | **[2026-09-25: diagnosed (row 461)]**
| peak Newton: secant step on dsigma/dw (1 evaluation per iteration instead of 3) | to do | **[2026-09-25: done (STAGE5 24)]**
| winding-number (eigenvalue count) certificate | in place at save time (--wind 1); run now on both K+B+C stationary points: 0 eigenvalues right of Re=-0.025 -- unconditionally stable (STAGE5 18) | **[2026-09-25: the winding results stand; the points were stalls]**
| closed-loop Hermite block | **built, ON by default in the SQP** (`--sqp-clblock`, STAGE5 20): exact derivatives (2e-8 .. 4e-8 vs FD) |
| ne=100 rows below the certificate | replay did NOT reproduce; prime suspect ROW SCREENING (zero gradients below 95 % of the max); controlled A/B at ne=100 next | **[2026-09-25: diagnosed (row 461)]**
| ne=100 rows below the certificate | **diagnosed** (STAGE5 19): not screening, not the model -- a resonance below 95 % at the start rising > 20 % in one solve, and a narrow twin peak 0.0055 from a followed one; **in-solve refinement ON in the nominal loop** (same round: t = certified = 9.003048, vs 8.146/9.861 without) |
| closed-loop Hermite block: where | reduced model (where the SQP linearises); full model only for a full-model inner loop; certification needs none (line + winding count) |
| hurwitz.py selftest 51/200 mismatches | **test defect** (zero threshold on badly scaled Bezoutians); by sign 0/200; definiteness never disagreed (STAGE5 20) |
| ne=100 K+B+C and K+FB+FC, refinement + closed-loop block | first runs: K+B+C 10.990 -> 7.944454 (3 refused rounds), K+FB+FC started wrong (transfer defect); three defects fixed (STAGE5 21); **rerunning** (tools/n100_kbc.sh, tools/n100_fb80.sh) | **[2026-09-25: superseded by rows 473-500]**
| certificate returned certified=True after finding a higher level set | **fixed**: inconclusive (certify_sparse, STAGE5 21); not reproduced as round 8's cause |
| outer loop ignored the certificate's ok flag | **fixed**: inconclusive never accepts a round (STAGE5 21) |
| transfer_ne.py Sobolev length | **fixed** (STAGE5 21) |
| post-hoc tau clip of the SQP step (refinement never saw the certified design) | **fixed**: cap inside the SQP as a displacement cone (STAGE5 22) |
| certificates on reduced pencils | **dense level-set by default** (`--reduced-cert dense`); sparse procedure missed a narrow peak (137 s, inconclusive) that dense found in 0.26 s (STAGE5 22) |
| sparse certify_greater sizes disks from partial ARPACK sets | **open, counted** (N_DISK_UNTRUSTED); bisecting stalled the frozen suite; for the cluster regression (STAGE5 22) | **[2026-09-26: closed: --cert-disk strict (retry for the nearest eigenvalue on held factors, bisect otherwise), counters in every record (STAGE5 30.5)]**
| winding acceptance of continuation-refused designs | at the margin Re=-eps (was -eps/2); on in the ne=100 scripts (STAGE5 22) |
| sandbox BLAS threads | pinned to 1 in the run scripts; earlier concurrent wall times inflated up to 60x (STAGE5 22) |
| ne=100 K+B+C nominal loop | **converged: 7.925385** from 10.990, every round t = certified, winding-certified, KKT 1.9e-4 (STAGE5 23); SQP subproblem failures at the fixed point -> Hessian reset on failure; clean rerun running | **[2026-09-25: a stall, not a minimum]**
| sigma_2 blocks at non-kink active frequencies | `--sigma2-active-gap 0.15` in the ne=100 scripts (K+FB+FC had 43 blocks, gaps ~0.87) |
| fake witness when the certificate proves the lower bound at once | **fixed** (STAGE5 24): witness = the reduced model's fine-sweep argmax (was the coarse 61-point full sweep's grid argmax: "10 % model error" every round, anchors and Omega members at non-peaks); frozen path unchanged (B: 10.23128516441147) |
| Hermite block normalised by its trace | **fixed** (STAGE5 24): with both compensator poles ON the line H0 = 0 and the gradients were scaled by 1e13 (Clarabel failures, KKT tool NumericalError); now max(|tr H0|, 1e-3 max ||dH_j||); KKT tool scale-free |
| ne=100 K+B+C clean rerun (old SQP) | 10.908 -> **7.929388** in 20 rounds (3,005 s), certified <= t (1+2e-6) every round, winding-certified, KKT 8.3e-4; pair both exactly on the line; the earlier run's 7.925385 (KKT 1.9e-4) is a nearby point (pair nearly coalescent) | **[2026-09-25: a stall; the rerun has finished]**
| slow K+B+C tail | diagnosed (STAGE5 24): the compensator pair on the line (Hermite block's second-order residual lands in its two small eigenvalues); merit weight, exact block curvature, cushion, trust shrink/ball, SOC, H scaling tried offline -- none decisive; kept as off-by-default options; merit now 'max' (`--sqp-merit`) | **[2026-09-25: mainly stale quasi-Newton memory (row 494) + the pair (face S, row 487)]**
| SQP evaluation cost | **4.6 -> 2.2 s** per K+FB+FC evaluation: `--follow-scan 0` (refinement covers twins), warm Newton with carried curvature + secant, top tolerance 1e-7 (STAGE5 24) |
| ne=100 reruns on the new code | **running**: K+B+C from 10.99 (tools/n100_kbc.sh), K+FB+FC from its round-7 checkpoint 3.913182 (tools/n100_fb80.sh) | **[2026-09-25: finished; see rows 491-500]**
| ne=100 K+B+C, new code | **converged 7.657552** (35 rounds, 2,617 s), dense-validated on the full model (level-set 7.657557, abscissa -0.050283), compensator DOUBLE POLE on the line (-0.05 - 0.3734i), KKT 2.1e-4, winding-certified (STAGE5 24); a different, lower basin than 7.925/7.929 | **[2026-09-25: a stall (later runs 6.40-6.41)]**
| KKT residual as a stopping signal | 1.5e-4 at 7.846865, then 2.4 % further descent: necessary, not sufficient (loose active set on flat tops) |
| --converge never fired on 1e-9 gains | **fixed**: gains below cert-rtol are not progress for the fixed-point test |
| tools/dense_validate.py | new: dense closed-loop abscissa, compensator poles, dense level-set H-infinity on the full model (GL sizes only) |
| ne=200 K+B+C, new code | **7.824127** (18 rounds, 1,914 s), dense-validated (8.8e-12), winding-certified -- but STALLED, KKT 3.6e-3: compensator double pole on the line (STAGE5 25) |
| compensator double pole on the line | **diagnosed** (STAGE5 25): the pole-based merit has infinite directional derivatives there (steps refused down to 1e-10); the Hermite measure moves (option MERIT_CTRL='hermite') but its violation is quadratic in the pole excursion -- linear penalties (merit, QP slack) are not exact; constraint-qualification failure. Remedy = active-manifold SQP on {double pole on the line} -- **decision for the user** | **[2026-09-25: decided and implemented: face S (row 487)]**
| compensator pair at the line (double pole / both on the line) | **fixed** (STAGE5 26): the pair's face S = {equal real parts, split vertically or double, left of the line} as rows c_u <= 0, c_a <= 0, c_b = 0 from the pair's sum/product, trial points retracted onto S, released at an S-stationary point; Schur-based cluster block; pick_pair for three poles on the line |
| SQP line search from alpha = 1 every iteration | **fixed**: warm start at 2 x last accepted alpha (6-7 -> ~2 evaluations per iteration) |
| carried quasi-Newton matrix after a stalled solve | **fixed**: reset (code 6 / subproblem failures) | **[2026-09-25: superseded: --sqp-carry-h off, every round (row 494)]**
| witness invariant | **asserted every round** (witness_invariant_violations in the result json; tools/check_invariants.sh); the witness is the best, by full sigma_1, of the SQP's maximal row, the model's fine argmax and top peaks |
| ne=200 K+B+C | **6.438335** (from the frozen 7.964064, -19.2 %), KKT 3.2e-5, dense-validated, winding-certified; 69 rounds, ~2.2 h with two other runs on the two cores (STAGE5 26) | **[2026-09-25: a local minimum; another run ended at 6.485693; the residual is not a certificate]**
| ne=100 K+B+C | 6.627392 after 40 rounds (dense-validated), continuation running | **[2026-09-25: ~6.40-6.41 later]**
| per-evaluation cost grows with |Omega| (~1.4 s at 400 rows, ne=200) | **next**: batched reduced-model rows (one call for all frequencies); the flat optimum has 136 near-active peaks | **[2026-09-25: done (row 495)]**
| plateaus in every run | **diagnosed + fixed** (STAGE5 28): stale quasi-Newton memory carried across rounds; `--sqp-carry-h off` (default). Replays: 7.509 -> 7.090 in 40 its (run: 7.451); 6.865 -> 6.806 in 25 (run: 280 its to 6.826). Pruning Omega to the local maxima does NOT help (stalls) |
| batched reduced rows + batched peak following | **done**: reduced.gain_batch, peakfollow.follow_batch; agreement 4e-13 (sigma), 1e-11 (gradients); 1.9 -> 0.8 s per evaluation at the flat ne=200 optimum |
| progress stop | `--stop-progress k:tol` (3:0.003 in the scripts), not on stalled solves |
| line search on the pair's face | alpha from 1 (the radius is the memory there); warm alpha elsewhere |
| KKT residual on a flat optimum | active set 2e-3 gives ~5e-5 long before the minimum; 1e-4 discriminates better but not monotonically (STAGE5 28) |
| ne=200 K+B+C, both fixes | 6.485693 (a local minimum 0.7 % above 6.438335), within 1 % after 44 min, progress stop at 56 min (single process) |
| ne=100 K+B+C, final code | 6.4164 after 57 min, within 1 % of the best known 6.400 after 45 min; running | **[2026-09-26: stopped at 6.410499; v6 6.404234 in 23 min, STAGE5 29]**
| cavity extrapolation | 2-9 h (serial reduced rows) / 1-4 h (rows on 8 workers) in the low/mid scenarios, 22-45 h high; needs the user's LU/solve times and the reduced dimension (STAGE5 28) |
| jobs | one at a time from now on (2 cores; each run already uses a 2-worker pool) |

**2026-09-26 (user: fix the creep at 7.93; fields; open items; no dead code, no stale conclusions) -- STAGE5 29**

| item | status |
|---|---|
| creep at 7.93 (ne=100), cause 1 | tau governor capped steps after SUCCESSFUL homotopy splits; `--tau-split ignore` (default, SQP arm): 6.410212 in 27.5 min (v5) |
| creep at 7.93, cause 2 | compensator pair let onto S only at a solve's start, separation <= 0.05 (0.078 -> 0.047 over rounds 12-15); now 0.1 and checked after every accepted step |
| followed members on one peak | members collected on the top at -0.5250 while -0.5095 stood higher: smax under-stated, iterate accepted on a false value; now members sharing a top restart from their own base (FOLLOW_DEDUP), +8 % wall |
| tested and removed | sigma_2 blocks inside the solve (none formed); trust radius shrink off the face (no gain, collapses with stale tops) |
| ne=100 K+B+C v6 | **6.404234 in 22 rounds, 23 min**, progress stop (+0.066 % over the best known 6.400015); within 1 % after 14.5 min (v4: 45 min) |
| ne=200 K+B+C, new code (v4) | **6.486694 in 24 rounds, 42 min** (v3: 6.4955 at its 56-min stop, 6.485693 after 77 min; same local minimum); within 1 % of 6.438335 after 33.7 min |
| progress stop, two holes | fired at round 3-4 of ne=200 (a) on an accelerating solve, (b) after a refused better candidate (7.4964 vs 7.9490); both excluded now (STAGE5 29.7) |
| FB/FC ripple | NOT cosmetic: the 80-mode run gained 0.9 % by growing rippled tails 11 % -> 36 % of the field norm; smoothing only the tails +34 %; the smooth frozen design is 0.6 % worse. Unregularised field problem exploiting the mesh. Remedy proposed: two scale-invariant shape rows (mean wavenumber, footprint) in modal coordinates; atoms as the cavity parametrisation. **open** |
| dead code | removed options listed in STAGE5 29.5; replay md5 identity before/after, frozen path, selftest, check_invocations OK |
| replay tool | tools/replay_round.py (was a scratch script): one round's SQP from a dump, DEBUG_LS row diagnostics |
| invariants after the changes | frozen 10.23128516441147 PASS; witness invariant 0 violations PASS; selftest 0/200 PASS |

**2026-09-26 evening (user: close every loose end that does not need the cluster; never regress) -- STAGE5 30**

| item | status |
|---|---|
| triage of every open row | done: implemented (below) or closed by recorded decision (STAGE5 30.10) |
| where the factorisations go | tools/lu_breakdown.py (new): per component and round |
| reference sweep 61 points -> reduced-model peaks | **done** (--cert-sweep 0) |
| gate re-swept the line the abscissa certificate had just proved | **fixed** (stability_gate skip) |
| plant spectrum recomputed at every abscissa certificate | **fixed** (cached) |
| end-of-run plot curve on the full model (2,481 LU) | **fixed** (reduced model, gain_model in the record) |
| sweep compared with another design after a compaction (spurious anchors) | **fixed** |
| factorisations per round (no winding event) | 192 -> **104**; ~69 with --cert-every 4 |
| winding count inconclusive with the pair on face S (half of the refused continuations) | **fixed** (--wind-ctrl exact: det(sI - A_c) f, closed-form compensator phase; validated on 27 dense counts) |
| winding tail constants | **made rigorous** (exact ||M^-1 A|| at GL; compensator resolvent via ||A_c||) |
| partial ARPACK sets in the sparse certificate | **fixed** (--cert-disk strict: retry k = 4, bisect, else INCONCLUSIVE); frozen hole found (empty set at ne=200, w = 278, interval dropped unproven) |
| analytic tail bound | **done at GL** (tools/tail_certificate.py): every record of the session certified over all frequencies; cavity: its own argument |
| progress stop | armed after 1 % gain or 20 rounds; exceptions for accelerating solves and refused better candidates; scripts at 3:0.0015 |
| KKT residual with the stop | run scripts write kkt_1e-4.json (ne=100 final: 2.5e-5 x gradient scale) |
| closed-loop pair at the line | detector (closed_loop_pair_rounds), 0 everywhere |
| reduced-model certification every 4th round | **validated** on the current code: 6/6 chains confirmed, factorisations halved, slower at GL -- option for the cavity |
| CLEANUP 5.1-5.4 | **closed** (reference tables, bounds file, positions, archive files) |
| K+FB+FC on the current code | 3.898938 in 20 rounds / 69 min (was 3.901205 over ~36 rounds) |
| regression vs 29 | curves dominate at every wall time; confirmation runs of the final scripts: **ne=100 6.399134 in 22.6 min** (was 6.404234 in 22.9), **ne=200 6.486505 in 31.4 min** (was 6.486694 in 41.6); both certified over all frequencies |
| 27 frozen records | re-certified globally (dense level set + tail bound): recorded values within 6e-7 of the supremum (results/reference_table_ne200_global.csv) |
| best known ne=100 | **6.389368** (32 min, stop 3:0.001) |

**2026-09-28 (user: do the FB+FC oscillations go away under refinement; why is viscosity not enough; bring Sobolev back if needed, results must be mesh-independent; compare atoms; algebra item 11) -- STAGE5 32-33**

| item | status |
|---|---|
| field-basis ordering (`_m_orth` scrambled the modes by an SVD) | **fixed**; only the post-hoc mode-band analysis of 29.4 was affected (ERRATA there; corrected numbers STAGE5 32.1) |
| why viscosity does not regularise | measured: response to a forcing mode ~0.72/k^2 = 1/(abs(gamma) k^2) (tools/mode_response.py) -- the forward map is compact, so rough content is ineffective, not costly, and the gauge removes the norm cost; the report's Sobolev was a change of variables (32.2) |
| refinement, unconstrained (R1: ne=200, 160 modes) | 3.888738 and falling; energy spread to the new cutoff, flat band 19 -> 28 (the resonance of the cutoff wavenumber), abs(K) 321 -> 451: the ripple moves to smaller scales (32.3) |
| plant convergence | open-loop peak 901.57 / 882.88 / 881.72 / 881.65 at ne = 100 / 200 / 400 / 800 (32.4) |
| scale-invariant Sobolev rows | `--field-kmax`, `--field-order 1|2` in the SQP; H^1 at k <= 1 too weak (3.895803, still rippled), H^2 at k <= 1: 3.909258 / 3.913414 (ne=100, 80 / 160 modes), 3.911740 (ne=200), 3.914071 (ne=400; 3.915962 on ne=800) (32.5) |
| atoms | 4 Gabor atoms per field (width >= 1, wavenumber <= 0.707): 3.912595 (ne=200, one basin), **3.903930 (ne=400)**; mirror-symmetric actuator/sensor sets; A2' (A3's design re-optimised at ne=200): 3.888855 in 12 rounds, continued to **3.876296** (3.879283 on ne=800); the progress stop is too tight for field cases (32.6) |
| cross-evaluation (tools/cross_eval.py, P2 evaluation / exact atoms, certified over all w) | unconstrained +41.6 % (80 modes, 100 -> 200) and +18.4 % (160 modes, 200 -> 400); H^1 +40.4 %; H^2 +5.8 % (100 -> 200), +0.31 % (200 -> 400); atoms +0.03 % (200 -> 400), +0.001 % / +0.002 % (400 -> 200 / 800) (32.7) |
| K+FB+FC 3.898938 | **withdrawn** (5.52 as a function on ne=200) |
| transfer_ne.py | P2 evaluation instead of linear interpolation; output named after the target ne |
| algebra item 11 | `--cont-track reduced`; replay against dense truth (tools/gate_replay.py): 42/42 same decisions, 0 false accepts, 1830 -> 166 factorisations; loop at ne=100: 6.405826 in 16.7 min vs 6.399134 in 22.6 (stop-rule difference): **option**, not the GL default (33) |
| invariants after the changes | frozen 10.23128516441147 PASS; witness invariant 0 violations PASS; Hurwitz selftest 0/200 PASS |

**2026-09-29 (user: does the static gain diverge; M products only; MIMO decentralized and centralized at ne=100; ne=400 runs belong on the cluster) -- STAGE5 34**

| item | status |
|---|---|
| abs(K) at convergence | finite for given fields: with A2'' fields fixed, gamma 8.93 / 5.41 / 4.21 at 0.25 / 0.5 / 0.8 abs(K*), 3.87629 at K*, then 3.87646 as abs(K) -> infinity; control-cost weight kept for later |
| M^(1/2) / Cholesky audit | loop's gain, gradients, certificates: M products only (unchanged); fixed: _m_orth (dense M), laplace_basis (dense eigh -> sparse shift-invert), fit_atoms (dense Cholesky -> r^H M r), tail_certificate nu (dense Cholesky + SVD -> Lanczos with sparse LU of M, equal to 1e-15), winding tail nu (dense SVD / power x 1.3 -> Lanczos, equal to 1e-15) |
| MIMO prototype (tools/mimo_proto.py, mimo_plots.py) | rank-4 Woodbury resolvent, gradients FD-checked, dense global certificate; ne=100: K+B+C SISO 6.399134 -> decentralized 5.594745 -> centralized 4.277652 (160 rounds, still falling); atoms SISO 3.877016 -> decentralized 2.387150 -> centralized 2.386009 (floor 2.383533) |
| invariants after the changes | frozen 10.23128516441147 PASS; witness invariant 0 PASS; Hurwitz 0/200 PASS |

**2026-09-29 (user: how much did sparse buy on the SISO production case; back the SISO code up; then MIMO in the production loop) -- STAGE6**

| item | status |
|---|---|
| sparse vs dense, SISO production | answered in the conversation: the per-round path was already sparse (resolvent, certificates); the STAGE5 34.2 changes were one-off costs (basis, atom fit, tail constants); the dense band spectrum per round remains to replace before the cavity |
| SISO backup | tag `siso-v1` (fdae0e7), `codesign_siso_v1.bundle`, `codesign_siso_v1_20260929.tar.gz`; MIMO on branch `mimo` (worktree) |
| MIMO production code | resolvent (rank m+p), gradients, pencil (rank 2(m+p)), winding det(I - K G), reduced oracle, PackerMIMO, driver, starts, scripts (6.1) |
| validation | SISO identity 8.754365110002194 in both trees and invariants PASS (after 6.4, 6.6, final commit 0e7f1c1); `tools/check_mimo.py` on every record (dense sigma 1e-12..1e-14, m=p=1 vs SISO 1e-14, FD, dense level set over all w, winding vs dense, abscissa vs dense); prototype designs 1e-13..1e-15 (6.2) |
| unstable acceptance (+17.13 - 111.77i, outside the band) | `--wind-accept` (MIMO default) (6.3) |
| stall 2.918: high-gain branch unseen by tracked rows, accepted unstable iterate (+38.19 - 164.05i) | no tracking for MIMO; certificate region from eig(D_c C^H M^-1 B) (6.4) |
| gate INCONCLUSIVE on band +-1269 | winding-count fallback (MIMO); the three winding calls share one helper (6.5) |
| false best 2.587706 (true 12.251419 at w = -700.5) | `--band-first` on for MIMO; dense all-w gamma check at save (all cases, n + r <= 1000) (6.6) |
| progress stop on atoms | `--stop-progress 8:0.0015` for the MIMO atoms cases (6.7) |
| non-deterministic field runs | ARPACK start vectors fixed (laplace_basis, tail constants, svds fallback); replicas identical (6.8) |
| results ne=100 | atoms: central 2.383533 and decentral (from the prototype's start) 2.383533 = the two-pair floor (1e-10 / 5e-9); decentral from SISO 2.964887 (stall, 6.10); K+B+C: decentral 5.565958 (prototype 5.594745), central 3.792962 (prototype 4.277652) (6.9) |
| relative stationarity with one very sensitive row | measured (6.10), open |

**2026-09-29 (second pass; user: everything sparse and validated; the atoms stall; why K+B+C centralized stopped; MIMO symmetry; static gain; MIMO code for SISO) -- STAGE6 6.12-6.23**

| item | status |
|---|---|
| dense audit | band (closed-loop spectrum at every acceptance), plant pole data (winding), dumped spectrum -> sparse; dense left only as checks gated at low n (6.12) |
| sparse plant enclosure (numerical-range rectangle + tiles proved empty / exhausted) | counts equal to dense at ne = 100/200/400, eigenvalues to 1e-11..1e-4 (6.13) |
| sparse closed-loop set right of -band_recut (candidates + tiling + winding count) | identical to dense on 9 records incl. the far branch -0.54 - 700.5i; in the loop candidates only (tiling cost 30 % of a round), whole band + count at save (6.13) |
| sparse all-w certificate (numerical-range tail bound + sparse level set) | equal to the dense all-w value on 6 records (SISO ne = 100-400, MIMO) (6.19) |
| atoms stall | path-dependent; the stalled design improvable along the gain (x10: 2.95996); intermediate code reaches the floor from the same start; SQP exits caused by the relative stationarity test (atom amplitude rows) -> `--sqp-stat pred` (MIMO default) (6.14) |
| SISO A/B `--sqp-stat pred` | ne=100: 6.404012 / 22.2 min vs 6.399134 / 22.6 min (ahead until the stop) -> SISO default unchanged (6.14) |
| K+B+C central stop | progress rule (0.126 %/3 rounds); continued: 3.717885 after 24 more rounds; equalised optimum (28 peaks within 1 %); MIMO scripts stop on 8 rounds now (6.15) |
| MIMO on SISO (`--force-mimo`) | same trajectory (2.5e-9) and cost as the SISO path; batched MIMO rows (5-6x); gauge + scale rule aligned (6.16) |
| duality | GL: A^T = P A P; gamma invariant under the dual (1e-14); self-duality measured (6.17) |
| static gain / non-diagonal central gain | high-gain limit T_inf controller-free; zeroing the off-diagonal gains leaves 2.3835333 (6.17) |
| MIMO packer scale rule (SISO rule slowed K+B+C central) | per-group floor for m + p > 2, SISO rule at 1x1 (11c4073) (6.22) |
| false best 41.64 at w = 409.65 inside the final band (at_dec_p, c4cf908) | all-w certificate of every accepted design (tail bound + level set on [-W, W]), band widened to the peak (528be56); fired again in kbc_cen (6.21) |
| final MIMO runs | atoms 2.3835333 x3 (the floor); K+B+C dec 5.339317 (11c4073), cen 3.715450 (11c4073, stopped by hand); all checks pass (6.20) |
| `check_mimo` FD step too coarse for one huge entry | per-entry step; all final records re-checked (6.20) |

**2026-09-30 (user: parallelise and test on 2 cores; speed up the QP; FD?; order-5 compensator) -- STAGE6 6.25-6.27**

| item | status |
|---|---|
| profile of a K+B+C central round | reduced rows 44 %, QP 24 %, gradient assembly 14 %; the equalised plateau (788 rows within 1 %) (6.25) |
| sigma_2 blocks batched; lu_counter frame walk; placement derivative memoised | done, bit-identical / same trajectory (6.25, 6.26) |
| frequency-parallel rows (`--row-workers`) | done; big round 1 core 718 s, 2 cores 555 s (1.29x; Amdahl limit 1.29 with 45 % parallel); machine scales 1.9x (6.26) |
| QP | one thread: -28 %, bitwise same; constraint generation tried, not a win on the plateau, removed; second QP per iteration: to A/B (6.26) |
| finite differences | only the peak curvature d2 sigma/d w2 (5.7 %); gradients analytic, Hessian BFGS (6.26) |
| order-5 central compensator | closed-loop weighted truncation start (plain BT destabilises); 3.967890 in 88 min, within 1 % in 23 min; order 16: 3.715450 in 5.3 h (6.27) |

**2026-09-30 13:33 (user: review the code for anything silly before the cluster) -- STAGE6 6.28**

| item | status |
|---|---|
| four read-only reviews, claims checked against code and the LU table | done (6.28) |
| sigma_2 / k=1 warm-subspace thrash | fixed (-3 % inner; SISO value changes 3.6e-9; identity check -> agreement 1e-7) |
| dense SISO warm-start search on the MIMO path (160 GB at 1e5) | fixed (skipped for MIMO) |
| mass LU refactorised 5x/round, s_far per call, unused pencil M LU, Sobolev LU at construction | fixed (bit-identical) |
| stability_region silent fallback ev=[0] | fixed (proven plant enclosure, printed) |
| n-length lifting in every row/gradient (6.25's "n does not enter the SQP" was wrong) | **open, must precede 2-D** |
| winding ~700 LUs/accepted round (83 %), certificate, enrichment, LU ordering | **open, must precede 2-D** (6.28 list) |
| second QP per iteration | dropped (n-independent, algorithm change) |


**2026-09-30 15:31 (user: fix and validate the 6.28 items before parallelising; M once at warm-up; BFS code and cluster nodes attached) -- STAGE6 6.29**

| item | status |
|---|---|
| mass matrix factorised once (massops, warm-up) | done, bit-identical (b6123df) |
| 1 reduced-space rows and gradients (no n-vector in the SQP) | done (7e2c3d2); unit checks 1e-13..1e-14 |
| 2 winding tail from the numerical range (mesh-independent) | done: W_T 960 -> 480 / 122880 -> 30720, same counts; segments in parallel -> parallel phase |
| 3 LU ordering in one place (`sparse_lu`, `--lu-order`) | done; MUMPS backend -> cavity |
| 4 certificate: all-w outside the band only, near-k, slope climb, pinned LUs | done; atoms A/B certificate LUs 144 -> 36 |
| 5 enrichment: block `_orth`, multi-RHS input solves | done; 4x at n = 1e5; column version kept for the SISO check |
| moderate items (no-progress certificate, band memo, start sweep, pool pickling, save repeats, gain_batch sweeps, 9 -> 6 G calls, Ts stack, gate on unacceptable candidates, enclosures, eigen-enrichment repeat) | done |
| save-time winding implied by a complete final band check | done |
| validation | invariants PASS (frozen 10.23128516441147); SISO agreement 4.3e-9 (column orth); atoms A/B same trajectory, -34 % wall; K+B+C order 5 full run 3.985055 (check_mimo PASS) vs 3.967890; old code from a 1e-12-perturbed start 3.996517: path spread 0.7 % |

**2026-09-30 21:10 (user: parallelise the full-model tasks, validate on GL on 2 cores, speed-up vs n, 1 task = before) -- STAGE6 6.30**

| item | status |
|---|---|
| one full-model pool (`fullpar`), forked at warm-up and before the rounds, auto re-fork on a changed mesh object | done |
| winding segments, certificate generations + location shifts + candidate samples, full sweeps, enrichment anchors / eigen-anchors, band shifts and tiles, plant heights | done; bitwise = serial except the winding point count (count equal) |
| 1 task = previous commit | done: component check ALL IDENTICAL; A/B round values and LU counts identical; invariants and SISO check pass |
| speed-up | GL ne=100: pooled work 7-10 % of a run, x1.56-1.74 on it, ~3 % end to end; 2-D operator n = 1.6k-25.6k: sweeps 1.46 -> 1.81, winding ~1.7, certificate ~1.55, enrichment ~1.1; grows with n towards structural bounds (6.30) |
| `--cert-split`, `--oracle-union new` | options for the cluster (default off, bit-identical defaults) |
