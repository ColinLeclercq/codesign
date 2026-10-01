> **ERRATA (2026-09-25) -- read first.** Every K+B+C value recorded as "converged", "a KKT
> point" or "stationary" between 7.6 and 7.96 (sections 11, 14.2, 18, 23, 24's 7.657552, and the
> basin/correctness remarks attached to them) was a STALL: the compensator pair near a double pole on
> the line (fixed by its face S, 26) and quasi-Newton memory carried across rounds (fixed, 28). The
> KKT residual with a 2e-3 active set does not certify optimality on flat optima (28). Best now:
> ne=200 6.438335, ne=100 ~6.40 (29). Section 28's "cosmetic" field ripple is not cosmetic (29).
> The negative options of 24-25 (CTRL_HESS, CTRL_CUSHION, TR_SHRINK, TR_BALL, SOC_CTRL, H_SCALE,
> MERIT_CTRL, merit weight 100) were REMOVED from the code (29); their measurements stay here.
>
> **ERRATA (2026-09-23) -- read before the text below.**
> - Section 3.1, "the exchange is CHAOTIC in its surrogate": measured with the two-sided model (1e-6
>   errors across designs); path sensitivity remains real, but that magnitude came from the model and
>   the working set, both since fixed (sections 5, 11).
> - Section 5.2 "In-solve refinement closes the Omega-gap": the gap was largely the working-set defects
>   (section 9, 11); refinement compensated them. 5.2.1's 7.9379 is a real certified value (rtol 1e-4) of a
>   run whose long steps passed the nearby KKT point: a basin, not a better loop (section 11).
> - Section 7's "~207 per round": on the corrected loop, with the abscissa certificate memoised and the
>   dump on the reduced model, ~150 full-model factorisations per round were measured (TRACKER K.5, L).
> - Section 8's reading "most of the earlier gain came from the refinement rather than the solver": the
>   refusals were working-set defects (sections 9, 11); on the corrected loop the SQP beats pyGRANSO for
>   a structural reason (STAGE1B 8.3).
> - Section 6.3 (slow K+FB+FC rounds) is resolved by section 13.
> - Certified values at rtol 1e-4 compare only above ~1e-4 relative.

# Stage 5 -- the projected oracle for the inner loop (algebra note, projection
# sections; note item 12)

Pulled forward (user, 2026-09-18): it answers the cavity memory question and
makes the inner loop -- and Stage 2's multistart -- essentially free, on GL
too. Design target from STAGE3 section 8: an A-N-style inner loop on the
reduced model inside an exchange certified on the full model.

# 1. What is built: `reduced.py`

Two-sided Petrov-Galerkin reduction of the PLANT (ABMSV 2017, the note's
variant). V, W in C^{n x k} orthonormal; A~ = W^H A V and M~ = W^H M V once
per basis; per design two matvecs b~ = W^H b, c~ = V^H c. The closed loop keeps
its Woodbury structure with n -> k: the frozen `ClosedLoopResolvent` class is
instantiated on the k x k dense plant, so setup / solve / solveH /
design_grad_tri are the same code. The design enters only through the scalar
k(s) and the two small vectors.

Singular values in the full oracle's conventions (its gain() is the Euclidean
eigenproblem of M T^H M T, eigenvalues ||M^{1/2} T M^{1/2}||^2): with
T~ = V R~ W^H,

    sigma_j = svd( G_V^{1/2} R~ G_W^{1/2} ),   G_V = V^H M V,  G_W = W^H M W,

a k x k dense SVD; optimal load u = M W G_W^{-1/2} s, reduced right-hand side
G_W^{1/2} s. (First written with M^{-1} in G_W: 12 % off at the anchor, where
the interpolation theorem demands exactness; the sign of the metric is the
whole convention.) The (x, z) pairs D.grads expects are lifted, x_p = V x~_p,
z_p = W z~_p, normalised as gain() does; a proxy carries the full-length b, c
and the reduced controller state, so D.grads is unchanged.

Enrichment at an anchor: the plant-state parts of the top-n_vec singular pairs
of the FULL closed loop there (they span R M^{1/2} v_j and R b), their adjoint
parts, and the two Woodbury columns R b, R^H c from the plant LU the full
oracle already holds; Gram-Schmidt twice; equal dimensions.

# 2. Correctness anchor (ne=100, K+B+C baseline design, fixed)

Interpolation at every anchor: 1e-14 (sigma_1, all anchors, all runs).

ABMSV greedy -- each new anchor at the REDUCED maximiser, so a spurious peak
becomes the next anchor and is interpolated away -- n_vec = 3, band [-30, 30]:

| anchor | at | dim | reduced max (true 9.71293) | rel. error, top half of the response |
|---|---|---|---|---|
| 1 | -0.50 | 4 | 9.71293 | 9.2e-1 |
| 2 | -0.75 | 8 | 12.78 (spurious) | 1.5 |
| 3 | +0.25 | 12 | 15.24 (spurious) | 1.9 |
| 4 | +2.00 | 16 | 9.72575 | 3.4e-3 |
| 5 | -1.50 | 20 | 9.71293 | **1.7e-4** |
| 7 | +2.25 | 28 | 9.71293 | 7.2e-9 |
| 10 | -1.25 | 40 | 9.71293 | 2.1e-11 |

This is the note's table (plant sweep: 4 anchors 7.8e-2, 5 anchors 1.7e-4)
reproduced on the closed loop. Residual absolute errors of 0.1-0.5 sit at
frequencies where the true sigma is 0.1-0.5: tail bumps below the maximum.
Anchors at the true local maxima only (what a certifier alone would supply)
leave spurious peaks ABOVE the true maximum at n_vec = 2 (15.5 at -0.75
against 7.8): the greedy rule is not optional. One sweep: 0.03 s reduced
against 0.9 s full at ne=100; the ratio grows with n.

# 3. Next: into the exchange

Phase 1: the inner solve's sigma rows (and sigma_2 rows) on the reduced
oracle, abscissa still on the full model; anchors = the certified active peaks
of every round plus the reduced maximiser when it disagrees with the
certificate; Woodbury columns refreshed at each round's anchors for the
position cases. Measure: certified gamma identical, LU count per inner
iteration (must reach 0 in phase 2), reduced-vs-certified at the peaks (1e-5).
Phase 2: the abscissa on the reduced pencil (Mengi 2022/2023 -- Ritz predictor
with the full-model residual filter; certificate outside).

# 3. Phase 1: the sigma rows in the exchange (`--oracle reduced`)

The inner solve's sigma_1 and sigma_2 rows on the projected oracle; the
abscissa row still the full 1-D certificate. Anchoring per round: the
accepted design's certified peaks (Woodbury columns refreshed when the fields
move); the ABMV rule -- the returned design's certified witness, when the
reduced value there differs from the certified one by more than
`--oracle-gate` (1e-5); and a band-wide check against the certifier's own
241-point sweep (already paid for), anchoring the largest disagreements above
`--oracle-sweep-tol` where the response exceeds `--oracle-band-frac` of its
maximum, also as a warm-up before round 0 from the start certification.

Three defects on the way, each caught by a measurement rather than by
reasoning: the load metric's sign (12 % at an anchor where the theorem
demands zero); the controller gradient's proxy slicing lifted n-vectors at k
(gradient cosine 0.27 to the finite difference, inner solve stuck at its
start; fixed by handing it V^H x_p, W^H z_p, now 2e-13); and `lu_counter`
counting the k x k splu's (68 per evaluation reported as factorisations; the
reduced loop now takes a dense LU through the frozen class's cache). Plus one
staleness bug: a per-design reduced loop reading the oracle's projection after
an enrichment; it now snapshots its basis.

## 3.1 Measurement (ne=100 K+B+C from K+B, stage-1b flags, 6 rounds)

| | LU per inner evaluation | of which sigma rows | gamma |
|---|---|---|---|
| full oracle | 90-102 | 68 (= |Omega|) | 9.410 |
| **reduced** | **24-29** | **0** | 9.65-9.89 (see below) |

A 3.7x reduction in full-model factorisations; the remainder is the abscissa
certificate (14 at the start design, more as the design moves) and the
anchoring (one per anchor). Reduced against certified at the witness: 1e-8 to
1e-14 from round 1 on, every variant -- the 1e-5 gate is met.

Three anchoring variants:

| anchoring | dim (end) | rounds identical to the full oracle | gamma |
|---|---|---|---|
| certified peaks + ABMV witness | 63 | none | 9.864 |
| + sweep, 20 % mask, 1e-4 | 69 | none, close | 9.654 |
| + sweep, 2 % mask, 1e-6 | 104 | rounds 0-1 to the last digit | 9.890 |

The tightest reproduces the full oracle's first two rounds exactly and ends
worse than the loosest: the exchange is CHAOTIC in its surrogate. Once the
reduced and full models differ by 1e-6 anywhere on Omega the line searches
diverge, and small differences at round 2 compound either way. "Identical
certified gamma" is therefore not a property a 1e-6-accurate surrogate can
deliver on this loop; the gate should read: witness agreement to 1e-5 (met),
zero factorisations per inner iteration (phase 2), certified gamma within the
exchange's path variability (a few per cent across budgets, STAGE1B/STAGE3).
The certification is untouched: every gamma above is a full-model certified
value.

Basis size is set by the response's complexity, not by n: ~70-100 here at
n = 199, so the reduction is modest at ne=100 (an evaluation is |Omega| dense
k x k solves against |Omega| LUs), grows with n, and is what makes the cavity
affordable. A cap on dim with restart from the current anchors is still to be
written.

# 4. Phase 2: the abscissa on the reduced pencil

`_ReducedCL.abscissa`: dense generalised eigensolve of the (k + r) reduced
closed-loop pencil (same block form as `alpha_dense` with A~, M~, b~ = W^H b,
c~ = V^H c), Ritz pairs lifted, and the FULL-model residual filter
||(A_cl - lam M_cl) v|| / (|lam| ||M_cl v|| + ||A_cl v||) by sparse matvecs
(`Acl_apply`, `Mcl_apply`) -- no factorisation. The rightmost Ritz value that
passes `--oracle-res-tol` (1e-4) is the abscissa, its lifted pair and eta feed
D.grads unchanged; when none passes, the full certificate is used for that
evaluation and counted. Anchoring adds the near-line EIGENVECTORS each round
(`enrich_eig`: shift-invert at the located values, one factorisation each --
the continuation watch list plus the gate's rightmost, and the start
certificate's for round 0): with frequency-response anchors alone the reduced
pencil does not contain the rightmost eigenvalue and the filter correctly
refuses every Ritz value; with the eigenpair in, alpha is exact (residual
4e-15, gradient 1.7e-12).

The residual filter at 1e-6 refused round 0's Ritz values (the first steps
from the parent design move the eigenvectors); at 1e-4 round 0 takes 35 of 50
evaluations from the reduced pencil and every later round all of them, with
the trajectory unchanged to the last digit against the 1e-6 run.

## 4.1 Measurement (ne=100 K+B+C from K+B, 6 rounds)

| | LU per round | per inner evaluation | gamma |
|---|---|---|---|
| full oracle | 3,547-4,273 | 90-102 | 9.410 |
| reduced, phase 1 | 987-1,428 | 24-29 (abscissa certificate) | 9.65-9.89 |
| **reduced, phase 2** | **450-611** | **~1** | 9.789 |

What remains per round is the OUTER certification -- measured directly:
`certified_norm` (241-point sweep + peak search) 317 factorisations, the gate
certificate 14 -- plus anchoring (6-12) and, until this commit, the sigma_2
selection sweep on the full oracle (~|Omega|, now routed through the reduced
one). **The inner iteration is factorisation-free** (gate condition 2);
witness agreement 1e-8..1e-14 (condition 3); gamma within the exchange's path
variability (condition 1, restated in 3.1). Per round the count fell ~8x at
ne=100, where the basis (dim ~100) is half of n; the ratio is set by n over
the response's complexity, hence grows with the mesh.

## 4.2 What Stage 5 leaves open

- The outer certification is now the cost. Its 241-point sweep is a design
  choice from the frozen code; the handoff's coarse-then-fine tolerance and
  the analytic tail bound (band derived, not assumed) both belong here.
- A cap on the basis dimension with restart from the current anchors (dim
  grows ~10 per round).
- The one-sided certified lower bound of the note (||T V||), not built.
- The A-N-style inner loop on the reduced model (STAGE3 section 8): now
  affordable, not written.
- ne=200 and a second case; the multistart of Stage 2 on the reduced model.

# 5. The reduced model made position-robust (2026-09-22)

Why the model failed after SQP steps: not the number of modes. The
controller enters only through k(s), so the model is exact for any controller
at its anchored frequencies; the inputs b(x_a), c(x_s) move with the
positions, and R(iw) b(x_new) is not in the span of R(iw) b(x_old).

Fix (user's proposal of a linear basis for the fields, applied to the
position family): Phi_b, Psi_c = orthonormal bases of the actuator/sensor
vectors over the position ranges (SVD of 121 snapshots; 20 modes each at
1e-8 for sigma = 0.4 over a range of 3 -- the translating Gaussian's n-width
is small here); each anchor adds R(iw) Phi_b and R(iw)^H Psi_c -- triangular
solves with the anchor's LU, no factorisation. Two construction defects found
and fixed: (i) two-sided projection with unrelated input/output spaces loses
rank (reduced pencil condition 1e18: actuator images live downstream, sensor
adjoints upstream) -> Galerkin on the union space U = orth(V u W), which keeps
Hermite interpolation (U contains R b and R^H c); (ii) the orthogonalisation's
deflation threshold was absolute (1e-10) against resolvent images of norm
~1e6 -> relative (1e-8); a 738-column "basis" for a 399-dimensional plant was
the symptom.

| ne=200 K+B+C frozen design, 6 frequency anchors | dim | worst relative error over [-6, 6], positions moved by 0 / 0.05 / 0.3 / 1.0 |
|---|---|---|
| two-sided (in use until now) | 24 | 7.6e-3 / 6.3e-3 / 7.6e-2 / 9.2e-2 |
| union Galerkin | 48 | 1.0e-5 / 2.5e-4 / 1.5e-3 / 2.1e-3 |
| **union Galerkin + input basis** | **98** | **2.6e-9 / 2.6e-9 / 2.7e-9 / 4.9e-8** |

Exact at the true peak (1e-13) in every case; pencil condition 1e2-1e4.
Loewner was considered and not pursued (TRACKER I.1). Consequences to test:
the "validity horizon" of STAGE2 3.6 was this defect -> provisional
certification (`--cert-every`) should now hold; the SQP's steps should no
longer be refused for model error. Cavity: dim ~100 dense, +2 x (n-width)
triangular solves per anchor.

## 5.1 In the loop: ne=200 polish with the position-robust model (running) **[2026-09-25: finished, see 14]**

From the frozen K+B+C design (7.9637), SQP + block constraint + adaptive
budget + Re <= -eps + cert-sweep 61, `--oracle-inputs on` (union Galerkin,
input bases b 20 / c 20 modes, dim ~100):
round 0 -> 7.9622; round 1 t = 7.9562 certified 8.4892, REFUSED -- but the
reduced model agreed with that certificate AT THE WITNESS to 2.3e-15: the
refusal is no longer model error; the peak sat at a frequency outside Omega
and the model knew it. Round 2 -> 7.9594. So the SQP's refused steps are now
the exchange's Omega-gap, which is what the in-solve refinement
(`--sqp-refine`, STAGE1B 8.2) closes -- ineffective before only because the
old model could not see such peaks. Second arm switched from certify-every-4
to `--sqp-refine 0.01`; both arms running. **[2026-09-25: finished]**

## 5.2 In-solve refinement closes the Omega-gap (2026-09-22)

The first refinement arm (`--sqp-refine 0.01`) added nothing: round 0 matched
the plain arm to the last digit. A reduced peak is added only if it exceeds t
by the tolerance, and polish gains are ~0.1 %. At `--sqp-refine 1e-4`:

| ne=200 K+B+C polish from 7.9637 | plain SQP, certify every round | SQP + in-solve refinement (1e-4) |
|---|---|---|
| round 0 | 7.9622 | 7.9621 (+2 frequencies mid-solve) |
| round 1 | t 7.956, certified **8.489**, refused | t 7.9539, certified **7.9530**, accepted (+2 mid-solve) |
| round 2 | 7.9594 | (resumed) |

certified/t = 1.000 every round with refinement. Mechanism: after each
accepted step the SQP asks the position-robust reduced model for the
design's peaks (the reduced level-set certificate) and adds the missed ones;
with the old model such peaks were invisible, with the new one the witness
agrees to 2e-15. Tolerance rule: at the scale of the gains, not of the
level. Reference: the sigma_2 pyGRANSO polish, 7.9499 in 5 rounds.

### 5.2.1 Result and cost

The refinement arm continued (resumed from checkpoints): 7.9637 -> 7.9621 ->
7.9530 -> 7.9435 -> **7.9379**, certified/t = 1.000 every round -- below the
sigma_2 pyGRANSO polish's 7.9499, the best certified value at ne=200 so far
(frozen 7.9641).

Cost: the dumps reported 496 and 1,217 "factorisations" per round, against
176-220 for the plain arm. The counter counted every splu, including the
reduced level-set certificate's ~100-dimensional pencils. `lu_counter` now
separates full-model factorisations (`n_lu_full`, size >= n). Measured per
component at the current design:

| component | full-model LU | reduced-size LU |
|---|---|---|
| one anchor (input basis included) | 1 | 0 |
| one in-solve refinement call | 0 | 48 |
| one SQP evaluation (|Omega| = 212) | 0 | 0 |
| one full certification (61-pt reference + level-set) | 143 | 0 |
| one closed-loop abscissa certificate | 26 | 0 |

A refinement round costs the same in full-model factorisations as a plain
round (~200: certification, gate abscissa, the SQP's certified validation,
a few anchors); the refinement is free at the cavity's scale.

## 5.3 Certifying every 4th round, with refinement and the position-robust model (in progress) **[2026-09-25: finished, see 14.1-14.2]**

ne=200 polish from the frozen design: full 7.9622 -> provisional 7.9619 ->
provisional 7.9590 (spot checks 7.9559 and 7.9506, both below the claims) ->
resume: the provisional best FULLY certified at **7.958468** against the
claim 7.959037 -> ... provisional 7.956441 -> resume: fully certified
**7.955888**. Twice checked, twice valid, and both times the reduced claim
sat ABOVE the truth by ~7e-5 (inside the certificate's 1e-4) -- the
conservative side. (The last attempt, with the old model, escalated 7 of 9
provisional rounds.) Provisional checkpoints are now labelled as such; a
resume always re-certifies in full.

Per-round full-model factorisations, by component (`lu_counter` now records
the driver/SQP frame that asked, and full-size factorisations separately):
outer certification ~80 (61-pt sigma reference ~54, D.hinf lower-bound seed
~12, level-set certificate ~15); gate stability_gate 25; gate
alpha_certified 24; the SQP's certified validation 24 (the SAME design as
the gate's); anchoring ~14. Two redundancies removed (reduced path only,
frozen path bit-identical): alpha_certified memoised per design (-24/round);
the lower-bound seed from the position-robust reduced model at 0.99 x its
maximum instead of D.hinf (-12 when used). Also: missed_peaks returns after
a successful reduced certificate (wall time, -~700 s/round at ne=200).

### 5.3.1 The stall, and the certificate's tolerance

The arm stalled (three rounds, SQP returning nothing admissible) from three
interacting causes, none of them the reduced model's accuracy:
1. the carried SQP trust radius had collapsed (7.8e-5) and was carried into
   the next round -- now floored at 0.1 x the fresh radius;
2. the SQP's "never worse than its start" guard compared its reduced-model
   values with the loop's CERTIFIED t0; with tiny steps every iterate sat at
   the start and looked worse -- the guard now uses the SQP's own starting
   value on its own model;
3. the level-set certificate's tolerance, rtol = 1e-4: re-certified at 1e-6
   the start is 7.956453, the reduced model said 7.956441 (1.5e-6), and the
   1e-4 certificate had said 7.955888. The "reduced claim ~7e-5 above truth"
   of both provisional re-certifications was the full certificate
   UNDER-reporting within its tolerance. `--cert-rtol` (frozen 1e-4) is now
   set before the start certification, so a run uses one tolerance for every
   certified value.

**Reading the session's numbers:** every certified gamma reported at ne=200
this week was at rtol = 1e-4 and may lie up to ~1e-4 relative below the true
maximum -- including 7.9379 (refinement arm) and 7.9499 (sigma_2 pyGRANSO).
Differences of ~1e-3 between methods stand; below that they are inside the
certificate's tolerance. A terminal polish gaining ~1e-3 per round needs
rtol ~1e-6.

## 5.4 Why the SQP's validation rejected its best iterates (2026-09-22)

A three-link chain, each link verified at the design itself (ne=200 K+B+C):
1. **alpha_certified returned the wrong eigenvalue.** When the stability line
   is clear, certify_sparse.stability_and_abscissa returned alpha_seen and
   witness None; alpha_certified then recovered eigenvectors at the REAL point
   (alpha, 0), and shift-invert converged to the mode nearest the real axis
   (-0.2538-0.573i) instead of the one attaining alpha (-0.0545-0.937i).
   The alpha VALUE was right; the returned eigenvalue and eigenvectors -- and
   hence the abscissa row's GRADIENT -- were another mode's. This affects
   every path, the frozen pyGRANSO one included, whenever the abscissa
   constraint is active and its rightmost eigenvalue has large |Im|.
   Fix: the serial certificate keeps the attaining eigenvalue
   (certify_sparse.LAST_ARGMAX); `--alpha-argmax on` makes alpha_certified
   use it. Default OFF (changes frozen results; default flips wait for the
   cluster regression, which should include it).
2. **The wrong mode was anchored** into the reduced basis each round (the
   eigen-anchor used that returned eigenvalue). Fixed by 1.
3. **The binding row was filtered out.** With the right mode anchored, its
   Ritz value at the SQP's next iterate was accurate to 2e-6 but its
   normalised residual was 1.0e-4 -- exactly the filter threshold (the
   normalisation by ||A|| ~ 1/h^2 inflates it); the SQP dropped the row, its
   abscissa rows described only damped modes, its steps pushed the real
   rightmost eigenvalue out, and the full-certificate validation rejected the
   result. Fix: `--sqp-row-res-tol 1e-3` (spurious Ritz values sit ~1e-2);
   the validation remains the safety net.
Before/after at the same start: best iterate 7.954292 NOT returned (7.954386
returned) -> returned. `ReducedOracle.enrich_shift` (resolvent anchored at a
complex eigenvalue with the input basis: the closed-loop eigenvector's plant
part lies in span R(lam) Phi_b for any controller and position) was added and
kept; here it added 2 columns only, the frequency anchors near w = -0.94
already spanning the mode.

# 6. Free fields on a truncated basis (FB/FC; user 2026-09-22) -- first measurements

`fieldbasis.py`: Hermite functions (centred on a region) and Dirichlet
Laplacian eigenfunctions of a region (K phi = lambda M phi on its nodes, zero
outside: the geometry-general choice), both M-orthonormal with a
rank-revealing orthonormalisation (high-order Hermite functions are not
resolved by the mesh and become dependent).

The frozen optimised free fields are NOT localised: above 1 % of the maximum
they span x in [-24, 24] (peaks at +-1.12). Projection error of the frozen
FB+FC fields (ne=200): Laplacian-Dirichlet on [-25, 25]: 20 modes 9 %,
40 -> 3.4 %, 80 -> 0.4 %; Hermite: 40 -> 19 %, 70 -> 7 % (a localised basis
is the wrong one for these fields).

What the truncation costs in gamma, without re-optimisation (certified at
rtol 1e-6; frozen 4.341229): 40 modes 4.347234 (+0.14 %), 20 modes 4.437735
(+2.2 %). A 40-mode field (80 real parameters instead of 798) keeps gamma
within 0.14 %, a pessimistic bound. Next: the packer's basis mode
(va = Phi beta, gradient Phi^T g) and an optimisation in the basis from the
projected design.

## 6.1 Basis mode in the packer and the driver

`packer.py`: with `D.field_basis_a` / `D.field_basis_s` set, z holds the
coefficients beta (beta0 = Phi^T M v0), unpack builds v = Phi beta, the
gradient is Phi^T g (same [Re, Im] convention as the nodal path). Gradient
checked against central differences of sigma_1 at w = 1.12 on the frozen
FB+FC design with 40 Laplacian modes per field: agreement 7e-10 .. 3e-9 in
three random directions. Design vector 160 real parameters instead of 1,596.
Driver: `--field-basis {none,laplace,hermite} --field-n N [--field-lo/hi]`
(region defaults to the start field's support padded by 1); with
`--oracle-inputs on` the reduced model's input bases become orth(M Phi_a),
orth(M Phi_s) -- b = M va/||va||_M lies in span(M Phi_a) -- so the model is
exact in beta. Frozen path bit-identical (no basis = nodal path).

## 7. Exact block gradient; parallel footprint (2026-09-22)

**Block-Hermite gradient, exact.** The compensator block's Hermite matrix
depends on the design only through t = tr B and q = det B of the 2x2 block
B = U^H (A_c + eps I) V / rho: dt = tr(U^H dA V)/rho, dq = tr(adj(B) U^H dA V)
/rho, and dH/d(Re t, Im t, Re q, Im q) by a 5-point stencil (exact for H's
per-variable degree <= 4). A_c is affine in z, so the directions dA_c/dz_j
are fixed (cached once) and zero for every non-compensator parameter (20 of
64 at K+B+C; all field coefficients in basis mode). `hurwitz.
block_hermite_and_grad`, used by the SQP evaluator. Against the previous
central differences: agreement 7e-10 of the largest gradient (the apparent
5e-2 disagreement was finite-difference noise on directions whose true
derivative is 1e-9..1e-12); 3 ms vs 24 ms per evaluation at K+B+C, and the
gap grows with the number of non-compensator parameters.

**Parallel footprint at the cavity (estimate, to be MEASURED).** 170,260 dofs,
P2-P2-P1, 2-D: nnz(A) ~ 10-14 M; nested-dissection LU factors ~15-30 x nnz ->
~3-10 GB per plant factorisation (i w M - A); the level-set certificate's
2n-dimensional pencil ~2.5-4 x that. Per 512 GB node: memory allows ~45-150
plant factorisations at once -- more than the cores -- so throughput
(bandwidth) binds: ~24-48 frequencies at a time per node is the realistic
range; ~10-20 certificate pencils. Symbolic analysis once per sparsity
pattern (all frequencies share it); anchors' extra solves batched right after
their factorisation. Measurement: `tools/lu_footprint.py` (MUMPS INFOG(9),
(16), (17), times; plant and 2n pencil) and `tools/lu_concurrency.sh`
(aggregate factorisations/minute at k = 6, 12, 24, 48 concurrent jobs); the
parallel width is the smaller of 0.85 x 512 GB / INFOG(17) and the knee of
the throughput curve.

**Per-round cost, corrected (2026-09-22).** A round of the every-round arm
measured 374 full-model factorisations, of which 167 were the diagnostic
dump (`--dump` recomputed sigma_1 and gradients on all |Omega| = 167
frequencies with the full oracle). The method's own round: ~207 --
certification 144 (sigma checks 105 incl. the 61-point reference, certificate
39), stability gate 25, anchoring ~22, other ~16. The dump now uses the
reduced model when one is attached (frozen path unchanged).

## 6.2 K+FB+FC: truncation and the mode count

The frozen K+FB+FC fields (gamma 3.925456, Sobolev length 0.8) are wider
still: support [-38, 38] (1 % of max), peaks at +-3.08. Projection onto
Dirichlet-Laplacian modes on [-38.9, 39.2], no re-optimisation (rtol 1e-6):
40 modes -> field error 11 %, gamma 4.592973 (+17 %); 80 modes -> 1.7 %,
3.927660 (+0.06 %). The mode count must follow the fields' extent: ~1 mode
per unit length of support in both cases (FB+FC: 40 over ~50; K+FB+FC: 80
over ~78). Still 160 real parameters per field against 798 nodal.

## 8. The controlled solver comparison, and refinement for pyGRANSO

Refinement explained (user question): the exchange keeps a finite working set
Omega; the inner solver minimises the largest sigma_1 over Omega. A long
step can raise a new peak at a frequency outside Omega; the inner solver
cannot see it, reports a small t, and the full certificate refuses the round
(fair SQP arm, round 1: t 7.9599 on Omega, 8.927 in truth). In-solve
refinement asks the (position-robust, accurate) reduced model after each
accepted step where the design's true peak is over the band and adds it to
Omega mid-solve. Any long-stepping inner solver needs it; pyGRANSO's earlier
polish avoided it by short steps. pyGRANSO cannot ADD rows mid-call (its
constraint vector has a fixed length, its BFGS and QP state are built on
it), so its form of refinement is ONE EXTRA ROW: the reduced model's
certified band maximum with the gradient at the maximising frequency -- a
nonsmooth max-type constraint, which pyGRANSO is built for; it also judges
the iterate's cost on the band. `--pyg-bandmax on` (frozen path unchanged).

Four arms, identical start (frozen ne=200 K+B+C), reduced model, flags,
tolerances (rtol 1e-6), 6 rounds, certification every round:
tools/fair_solver.sh -- SQP, pyGRANSO, refinement OFF for both;
tools/fair_solver_refine.sh -- pyGRANSO + band-max row, SQP + in-solve
refinement.
SQP without refinement after 4 rounds: 7.9622, refused (8.927), refused
(8.003), 7.9542.
Resumes restart the round counter (an interrupted arm runs >6 rounds in
total); the arms are compared ROUND BY ROUND (best certified value after
round k, counting across launches), not by final values. SQP without
refinement, resumed: round 4 refused (t 7.9520, certified 8.580).

## 9. Why long steps are refused: Omega membership, not the linear model (2026-09-22)

The two refused rounds of the fair SQP arm (no refinement), reconstructed from
the dumps (design before the step vs candidate after it, full response):
- round 1: missed peak at w = +1.890, sigma 7.7244 BEFORE the step = 97.0 % of
  the maximum -- near-maximal, not in Omega (nearest member +2.000, 0.11 away,
  where the candidate reads 7.4175 while the peak reached 8.925);
- round 2: missed peak at w = -1.390, 96.4 % of the maximum before the step;
  nearest member -1.35 (0.04 away, on the peak's flank).
Steps were tiny (relative 6.7e-4, 1.3e-3): narrow resonances are very
sensitive. Each Omega member's linearised row was right AT its frequency; no
row sat on these peaks' tops. Two defects:
1. **The enrolment cap**: the certifier added its witness + at most 4 local
   maxima; near this optimum many near-equal peaks exist (6-18 within 0.2 %),
   so near-maximal peaks were left out. Fix: `--enroll-rel d` enrols EVERY
   local maximum within d of the max, located by golden-section on the
   reduced model (free); frozen default unchanged. At the round-0 design with
   d = 5 %: 6 peaks enrolled (frozen rule 4) -- the +1.890 peak is enrolled at
   its top 1.909 (0.019 from where the step moved it; frozen rule: nearest
   enrolled 1.46 away).
2. **Drift of fixed-frequency members**: the -1.39 peak's top was at -1.339 at
   the start and moved 0.05 in one step, onto the member's flank. Fix (next):
   PEAK-FOLLOWING rows -- at each evaluation re-locate the local maximum near
   each peak member on the reduced model (a few reduced gains) and use, by
   the envelope theorem, dsigma/dz at the relocated argmax.
With both, in-solve refinement becomes the safety net for genuinely new
resonances, not the main mechanism.

## 10. Proposal: certify on the reduced model, full model as a failing check (user, 2026-09-22)

Every round: the reduced level-set certificate + the one factorisation the
round already spends anchoring the model at its own witness, which gives the
FULL sigma_1 at the claimed peak for free. What that cannot see: a peak where
the basis does not reach, a missing eigenvalue. So: full certificate (sigma
and stability) on an ADAPTIVE schedule (interval lengthens while full checks
agree, shortens on disagreement) and always at the end; a failing full check
adds the missed frequency/eigenvector as anchors and Omega members and the
run CONTINUES (only the final design needs the guarantee). Evidence: both
provisional chains checked this week agreed with the full certificate to
1e-6. Expected: a handful of full factorisations per round + ~200 per full
check. To build after the solver comparison.

## 11. The working set, corrected (2026-09-22; details in TRACKER K-K.4)

Audit against the frozen snapshot (26 Aug): the enrolment weakness was always
there -- each round added the witness + the TOP 4 local maxima, members were
fixed frequencies, nothing retired -- with one degradation of ours:
`--cert-sweep 61` made the full sweep spacing 1.0, and the START peak list
(reduced model not yet anchored) came from it.

**Enrolment** (`--enroll-rel d`): every local maximum within d of the maximum,
located at its TOP by golden section on the reduced model (no factorisation);
the start's list is re-derived on the model once round 0 has anchored it.

**Peak following** (`--peak-follow on`, peakfollow.py): a member enrolled as a
peak re-locates its peak's top within +-0.1 at EVERY evaluation of the inner
solve (golden section on the reduced model, ~16 reduced evaluations) and takes
sigma_1 and its gradient there; exact by the envelope theorem
(d sigma(w*(z); z)/dz = d sigma/dz at w*, since d sigma/dw = 0 at an interior
maximum). After each solve members are moved to their tops at the new design
and members closer than 1e-3 are merged. The sigma_2-enrolled rows (2x2 Clarke
block in the SQP, sigma_2 rows in pyGRANSO) follow too: their bypassing
following was the cause of the last refusal (reproduced: fixed block rows ->
refused at 7.965; following -> accepted, t = certified = 7.955734).

**Other defects fixed on the way**: the SQP's start level was the run's
start certified value, never updated (a stationary SQP reported 7.964 at a
design certified 7.9518); resumes reset the adaptive budget, the trust
radius and the peak members (now in the checkpoint); dumps stored z relative to
the process's packer (now the design itself); a refusal diagnosis is logged.

**Result**: the corrected SQP loop takes certified steps (inner t vs the full
certificate within ~5e-6) and converges to a KKT point: K+B+C ne=200,
7.9641 -> 7.955184 -> 7.951836 (KKT residual 1.9e-4 x gradient scale) ->
7.950855 (2.0e-4); ~150 full-model factorisations per round. Longer-stepping
variants that reached lower values (7.9485 unfixed, 7.9379 with in-solve
refinement) passed this nearby stationary point: basin, not correctness. **[2026-09-25: every value in this section was a STALL of the compensator pair and stale curvature memory, not a KKT minimum: sections 25-28]**

## 12. Reduced-model certification: status (2026-09-22)

Paused by the user, then reconsidered. The case for it is stronger than when
proposed: every refusal investigated was a working-set defect, not a model
error (model vs full certificate at each refused witness: 1e-14); both
provisional chains were confirmed to 1e-6; with the corrected working set the
inner t agrees with the certificate to ~5e-6. Remaining risks: a peak where
the basis does not reach, an eigenvalue missing from the reduced pencil
(neither observed since the input basis). Proposed as a MEASURED experiment
on K+B+C against the corrected loop (gamma, KKT residual, full-model
factorisations): reduced certificate every round + the free full check at its
witness; full certificate (sigma and stability) every k-th round and at the
end; a failing full check becomes anchors and Omega members and the run
continues. After K+FB+FC, if the user agrees.

## 6.3 K+FB+FC in the basis: wall-time cost of the model's size (2026-09-22)

With 80 modes per field, each anchor adds the input images R(iw) M Phi (80 + 80
directions): after 7 anchors the reduced model has dimension 229, growing with
the warm-up. A reduced sigma_1 evaluation builds and decomposes a dense matrix
of that size (O(dim^3), ~30 ms at dim 230-400 in Python); ~160 Omega rows plus
peak searches make an inner evaluation several seconds and a round tens of
minutes. Full-model factorisations are unaffected -- this is wall time, and a
consequence of requiring exactness in 160 field coefficients: the dimension
scales with the basis size, not the response's complexity. Remedies (not
done): sigma_1 only by a few iterations on the operator instead of a full
decomposition; compress the input images across anchors (nearly dependent at
nearby frequencies).

## 13. Fast reduced sigma_1, sigma_2 (2026-09-23)

The reduced gain did three O(k^3) things per frequency -- a dense LU of the
k x k pencil, k solves forming the whole reduced resolvent, a full SVD --
although only sigma_1 (and sigma_2 at enrolled rows) is needed. Now:
- QZ (Hessenberg-triangular) form of the reduced pencil ONCE per basis update;
  a plant solve at any frequency is one triangular solve, O(k^2), no
  factorisation (Laub's method; backward-stable, unlike diagonalising);
- the closed loop by the frozen class's rank-2 Woodbury with plant-only
  right-hand sides done as BLOCK solves (all columns in one triangular solve);
- the leading k singular triplets by subspace iteration on S^H S with a
  Rayleigh-Ritz step (block k + 4), WARM-STARTED from the frequency's last
  subspace, or the nearest cached frequency within 0.05 (peak following
  evaluates at new frequencies); converged when every residual
  ||S^H u_j - s_j v_j|| <= tol s_1: 1e-10 when vectors are needed, 1e-6 for
  value-only calls (sigma's error goes as the residual squared); ARPACK
  fallback.
Checks against the dense path (K+FB+FC, 80 modes per field, dim 214):
sigma_1, sigma_2 to <= 7e-15, gradients to <= 9e-12. Per frequency: warm
17 ms vs dense 64-68 ms (3.9x); peak following over 7 peaks 0.79 s vs 8.36 s
(10.6x, identical tops); K+B+C (dim 92): warm 3.5 ms vs dense 8.4 ms.
An inner K+FB+FC evaluation: ~3.5 s instead of ~19 s. Remaining lever (not
done): looser tolerance for rows far below the maximum.

## 13.1 One SQP evaluation along a real path (2026-09-23)

Profiling on a PATH (design changing each evaluation, not a repeated design)
at dim 232, K+FB+FC: rows ~0.6 s, peak following 0.54 s, a FULL dense eig of
the reduced closed-loop pencil for the abscissa rows 0.35 s, per-frequency
setup 0.3 s -- plus overheads. Changes, results identical (rows to 1e-14,
abscissa rows to every digit):
- abscissa rows by shift-invert Arnoldi (one dense LU per shift) at the
  previous evaluation's near-line eigenvalues; full eig every 10th call or
  when nothing is cached (new modes); residual filter and gate unchanged;
- peak following warm: +-0.02 around the previous top, 11 steps, full-window
  search if the top reaches the narrow window's edge;
- QZ conjugate transposes and b~, c~ conjugates computed once, triangular
  solves without finiteness re-checks (these had been ~25 % of an evaluation);
- row screening (`--row-screen 0.05`): values for all rows at the loose
  tolerance, vectors and gradients only within 5 % of the max (36 of 127);
  the rest enter the QP with their true value and a zero gradient.
Per evaluation along the path: 2.7 s -> 2.15 s (overheads) -> 1.90 s
(screening). What remains is many small triangular solves inside the subspace
iterations, independent across rows: the next factor is parallelism.
**Correction (2026-09-23):** the shift-invert path for the near-line
eigenvalues produced WRONG ABSCISSA GRADIENTS at K+B+C: left eigenvectors from
a separate adjoint Arnoldi run, matched to the right ones by nearest
eigenvalue, can pair the wrong vectors near close eigenvalues (eta ~ 0), and
the SQP's first subproblem failed (Clarabel NumericalError; round 0 of the
reduced-certification experiment returned nothing). The equality check had
compared abscissa VALUES only. The path is OFF (full eig every call) until
the left vectors are computed by inverse iteration at each eigenvalue. The
ARPACK non-convergence that ended a run is also guarded (dense fallback).
**Shift-invert near-line eigenvalues -- fix attempted, NOT adopted (2026-09-23).**
Left eigenvectors by inverse iteration at each eigenvalue (instead of matching
a separate adjoint Arnoldi run) removed the pairing defect, but a direct
comparison of the abscissa ROWS with the full eig (values AND gradients, 4
designs at K+B+C) still disagrees: a truncated shift list missed 3 of 7
near-line eigenvalues (dropped constraints); with every near-line eigenvalue
as a shift the path returns 8 rows against 7 and one gradient differs by a
factor of order one. Not resolved; the path stays OFF (full eig every call,
verified correct). Benefit if fixed: ~0.7 s per evaluation at dim 118-232.

## 14. Reduced-model certification on K+B+C (user's scheme, 2026-09-23) -- first results, and scaling

Run: corrected SQP loop + `--cert-every 4 --prov-check 0 --cert-fail continue`
(provisional rounds: reduced level-set certificate + full spot check at the
witness and the reduced peaks, ~5 factorisations; full certificate every 4th
round; convergence only on full rounds). Resumes (sandbox) restart the
schedule, so the observed full:provisional ratio is higher than 1:3.
| round | kind | certified | full spot check |
|---|---|---|---|
| 0 | full | 7.961644 | -- |
| 1 | provisional | 7.957661 | 7.9577 |
| 2 (resume) | full | 7.953239 | -- |
| 3 | provisional | 7.952906 | 7.9529 |
| 4 | provisional | 7.950096 | 7.9501 |
| 5 | provisional | 7.922910 | 7.9229 |
| 6 | full: provisional best re-certified | **7.922909** (claim 7.922910) | -- |
Every provisional claim matched; the full re-certification of the chain
agreed to 1e-7 across a 0.35 % jump. First model-accuracy discrepancy since
the input basis: the next candidate's full certificate 7.906420 vs the SQP's
reduced t 7.891397 (0.19 %), witness 0.0002 from a followed peak member -- the
model itself, after a large step (caught by the full round; a provisional
round's spot check at the witness would catch it too).

Cost (full-model factorisations; the counter now tags reduced-model work
explicitly -- 19 reduced-certificate factorisations were first counted as
full, the doubled pencil 2(k+r) ~ 250-450 exceeding GL's n = 399): provisional
round ~76 (stability gate 25, abscissa certificate 20, anchoring 17,
continuation 8, spot check 5), full round ~150-170.

**Scaling to the cavity (n = 170k).** Full-model factorisations scale with n,
reduced work with the model dimension k (independent of n). On GL a
factorisation costs ~1 ms and the saving is invisible in wall time; at the
cavity (tens of seconds each; the certificate's 2n pencil ~3x an n
factorisation) total cost ~ sum of factorisations x their cost, and the
speed-up tends to the COST-WEIGHTED ratio: full round ~ 61 + 3 x 25 + 50 + 25
~ 210 units, provisional ~76 (or ~30 with the stability gate on the reduced
pencil in provisional rounds). Over a 4-round cycle: ~1.9x as implemented,
~2.8x with the gate reduced; larger on the critical path (provisional rounds
skip the certificate's sequential iterations). Worth it -- 2-3x -- if the full
checks keep agreeing.

**Shift-invert near-line eigenvalues at the cavity.** The saved cost is a
dense eig of the reduced closed-loop pencil, O(k^3) per SQP evaluation,
independent of n. k ~ anchors x (input basis + a few), doubled by the union
Galerkin: ~100 (GL K+B+C), ~230 (GL K+FB+FC), plausibly 500-2000 at the cavity
(15-30 anchors, 20-80 actuator modes) -> ~10 s to ~10 min per evaluation
(0.35 s at 232), x ~80 evaluations a round: dominant. Worth doing properly:
the plant QZ makes a solve at ANY complex shift an O(k^2) triangular solve,
the controller by the same Woodbury; track each near-line eigenpair (right and
left) by Rayleigh-quotient iteration in O(k^2) per step; periodic full eig
for new modes; the gate's full certificate as safety net. Complementary: cap
k by compressing the basis.

## 15. Near-line eigenpairs by Rayleigh-quotient tracking, O(k^2) (2026-09-23)

The reduced closed-loop pencil is A_cl - s M_cl = S0(s) + U V^H: S0 =
blkdiag(At - s Mt, Ac - s I) with the plant block TRIANGULAR in the QZ form
(computed once per basis) for ANY complex s, and an exact rank-2 controller
coupling U = [[b~, 0], [0, Bc]], V^H = [[K c~^H, n^T], [c~^H, 0]] (checked:
6e-17). A solve at any shift is O(k^2) (checked against dense solves: 1e-13,
forward and adjoint). Each near-line eigenpair of the previous evaluation is
refined by two-sided Rayleigh-quotient iteration -- right and left vectors of
the SAME eigenvalue by construction, no matching -- converging in 1-2 steps to
a mode-dependent floor (1e-14 .. 4e-11; a fixed 1e-12 tolerance had rejected
every set: now converged when < 1e-9 and no longer decreasing). Full eig at
the first call, every 10th, or when a track fails or two merge; the full-model
residual filter and the gate's certificate unchanged.
Checks: 6 designs at K+B+C -- same rows as the full eig, values 1e-10,
GRADIENTS 1.2e-8; K+FB+FC dim 232 -- rows 9e-11. Time of the eigen part:
0.70 s (full) -> 0.046 s (15x) at dim 232; whole evaluation 2.60 -> 2.21 s.
Scaling (full ~k^3, tracking ~k^2): k = 1000 ~56 s -> ~0.9 s, k = 2000
~7.5 min -> ~3.5 s per evaluation. Crossover below 232 (at 86 tracking is
slower: per-step overhead): enabled above dim 150 (TRACK_MIN_DIM).

**Defect (2026-09-24): provisional rounds still ran the FULL certificate.** The
provisional early return in `certified_norm` sat AFTER the full level-set
certificate call, so provisional rounds paid for the certificate (its
heaviest, most sequential part: 19 factorisations of the 2n pencil in the
measured round) and discarded its result; only the 61-point reference sweep
was skipped. The reliability evidence stands (the reduced value was the one
used); the measured provisional cost (~76-90) overstated it. Moved before the
full certificate, with its own positive lower bound from the reduced model.
Continuing: full round 7.873742; provisional 7.864519 (spot check 7.8645).

## 14.1 The stability gate on the reduced model in provisional rounds (2026-09-24)

`--prov-gate reduced` (default full): in provisional rounds the SQP skips its
full validation of candidates (reduced admissibility kept); the gate takes the
closed-loop abscissa from the reduced pencil's near-line Ritz pairs, each
residual-checked against the FULL operator (products, no factorisation), and
skips the rectangle certificate and the continuation gate (falls back to the
full gate if no pair passes the filter). The continuation anchor moves only
after a full gate. At the next full round the provisional best's STABILITY is
re-certified on the full model (abscissa certificate + rectangle certificate,
same arguments as the gate); sigma is meaningless for an unstable loop, so a
failure ROLLS BACK to the last fully certified design; success re-anchors the
continuation there. Frozen path bit-identical. The K+B+C experiment continues
from its checkpoint with this setting.
First provisional round with the reduced gate: **22 full-model factorisations**
(frequency anchors 9, eigenvector anchors 8, spot check 5) against ~165 for a
full round; claim 7.862627, spot check 7.8626. At the cavity, cost-weighted
(the certificate's 2n factorisations ~3 units each): full round ~210 units,
provisional ~22, ALL independent (one parallel wave) -> ~3x in total work over
a 4-round cycle (840 vs ~276), more on the critical path (three rounds in four
without any sequential certificate or gate iterations). Pending: the full
round's re-certification of the chain (sigma AND stability).
**Chain checked (reduced gate):** provisional rounds 7.863637 (full) ->
7.862627 -> 7.849748 (spot checks 7.8626, 7.8497); the full check (the
resume's start certification + the start's abscissa certificate and verified
continuation anchor): sigma **7.849747** (claim 7.849748, 1e-7), stability
alpha **-0.0542** (margin -0.05), 8 near-line eigenvalues watched. Three
provisional chains confirmed on the full model so far (1e-7, 5e-7, 1e-7),
the latest with stability decided on the reduced model throughout.

## 14.2 Reduced-model certification on K+B+C: conclusion (2026-09-24)

**Converged**: KKT residual 3.1e-4 x gradient scale at **7.848591** (30
near-active peaks near -2.5 and -1.42), certified on the full model. The
every-round loop converged at 7.9509 from the same start: both are stationary
points reached by different paths -- one pair of runs does not show that
provisional certification finds better optima. **[2026-09-25: 7.848591 and 7.9509 were stalls; 6.438335 at ne=200 later (26)]**
**Reliability**: every provisional claim matched its full spot check; three
provisional chains confirmed by the full certificate (1e-7, 5e-7, 1e-7), the
last with STABILITY decided on the reduced model throughout (full check
alpha -0.0542); one small model discrepancy after a large step (0.19 %),
caught. No failure of the scheme observed.
**Cost**: the run's total (12 rounds, 1453 full-model factorisations) is not
a clean measurement -- 7 resumes (each a full start certification + a full
first round), early provisional rounds still paying the full certificate (the
defect fixed in 14), the reduced gate only in the last rounds. Clean per-round
figures: full ~165, provisional 22-36 with the reduced gate. Uninterrupted
(the cluster): a 4-round cycle ~250 vs 660 -> **2.6x fewer factorisations,
~3x less work cost-weighted at the cavity**, more on the critical path.
**Verdict**: reliable on the evidence so far and worth ~3x at the cavity;
recommended as an OPTION to validate on the cluster (27-case regression with
--cert-every 4 --prov-gate reduced --cert-fail continue), not yet a default.

## 16. Newton peak following; debugging at ne=100 (user, 2026-09-24)

**Golden section was a regression.** The frozen code has always located peaks
by a safeguarded Newton method (ClosedLoopResolvent.peak: analytic dsigma/dw
'gprime' from the singular vectors, finite-difference second derivative,
trust radius, step halving). Peak following had used golden section instead,
ignoring the available derivative (11-16 evaluations per peak, accuracy
limited by the window). `peakfollow.follow` now uses the frozen Newton method,
starting from the previous top, with bounded halving (a 20-halving safeguard
cost 80 calls on a flat top), and returns the evaluation at the top (vectors
included) so the row needs no further gain call. Checked at ne=100 from
members displaced by up to 0.03: tops EXACT against a 4001-point scan of each
window, 7-15 gain calls per peak from a cold start (1-2 iterations from the
previous top). A run-like SQP evaluation at ne=100 (dim 165, 125 rows, 4
followed peaks, 3 sigma_2 blocks): ~0.75 s.
**The earlier "1.9 s per evaluation" benchmark was unrepresentative**: under
the run's own conditions at ne=200 (175 rows, 15 followed peaks, 4 blocks) an
evaluation took 6.5-7.8 s -- ~700 singular-value computations, ~200 of them
golden-section peak searches. The warm-start cache (keyed at 1e-8, scanned
linearly, growing with every search) is now keyed at 1e-3 with direct
neighbour lookup (bounded; not itself the cause).
**ne=100 for debugging**: tools/transfer_ne.py carries a record to another
mesh (controller and positions unchanged, fields interpolated, xi regenerated,
gamma re-certified): K+B+C 7.963743 (ne=200) -> 10.990169 (ne=100: a coarser,
different problem -- earlier ne=100 runs sat at 9.7-9.9); K+FB+FC 3.925389 ->
3.935632 (smooth fields, little mesh sensitivity). tools/ne100_nominal.sh runs
the corrected loop to convergence on both.

**Reduced certification solves the same problem.** The inner SQP's model --
sigma_1 rows on the reduced model, the 2x2 Clarke blocks, one row per
near-line closed-loop eigenvalue, the compensator's double-pole Hermite block
(computed exactly from A_c, never from the reduced model) -- is identical in
full and provisional rounds; only the outer ACCEPTANCE test changes
(provisional: reduced sigma_max, and with --prov-gate reduced reduced
abscissa; the full model re-certifies both every 4th round and before
convergence). Known gap in BOTH regimes: coalescing CLOSED-LOOP eigenvalues at
the margin have only per-eigenvalue rows, ill-conditioned there (eta -> 0; the
Burke-Overton situation the Hermite block fixed for the compensator) -- not
hit on these cases.

## 17. ne=100 K+B+C: rows below the truth -- not the anchors (2026-09-24)

Run (tools/ne100_nominal.sh, corrected loop): 10.990 -> 10.205 -> 9.742 ->
9.172 -> 8.920 -> 8.359 -> 8.265 -> (8.229 not accepted) -> 8.166. From round 4
on the SQP's t fell 0.4-4 % below the full certificate (e.g. round 5: t 7.960,
certified 8.265 at -2.4314; nearest followed member 0.055 away).
- The loop's model matched the full value AT each witness to 1e-14 (rounds 4-7);
  only round 2 had a genuine model error (16 %, early, far from the optimum,
  dim 89).
- A model rebuilt at round 5's start design, anchored at its 5 peaks, is exact
  at the candidate to 1e-13 within 0.2 of an anchor and 7e-9 anywhere in the band;
  flanking anchors change nothing. Frequency coverage is NOT the cause: the plant's
  resolvent images at an anchor are design-independent (positions/fields via the
  input basis, controller via the exact Woodbury), so frequency is the only
  interpolation, and a few anchors cover this band.
- Two narrow peaks sharing a following window was the next hypothesis (Newton
  climbs to the nearest local maximum): peakfollow now scans the window
  (21 value-only reduced evaluations) before Newton. But in the reconstruction,
  plain Newton from the member (-2.4861) already reaches the true peak
  (-2.4314, 8.2648) -- NOT confirmed.
Open: why rows reported less during the solve. Candidates: the member's state
during the solve (hints carried from earlier iterates), whether the member was
in the followed set then, how the SQP reports t for its returned iterate.
Next: replay round 5's SQP in isolation at ne=100 with per-row logging.

Second derivative in peak Newton: an exact sigma'' needs the whole singular
spectrum (sum over the other singular triplets), i.e. the full decomposition
removed in 13; the finite difference (2 warm calls) affects Newton's RATE
only -- the top is fixed by the exact dsigma/dw. Cheaper still, to do: a
secant step on successive dsigma/dw (1 evaluation per iteration).

## 18. Stability certificates in place; closed-loop Hermite block; replay (2026-09-24)

**The eigenvalue-COUNT certificate** (user question): per round the gate uses
layer 1 (compensator poles, exact) + layer 2 (the LINE Re = -eps proved
eigenvalue-free) + continuation from a certified-stable anchor; the 2-D
rectangle covering was replaced by this in Stage 1 (the flag `--rect_cert`
still names the layered gate). The UNCONDITIONAL count is the winding-number
certificate (argument principle on f = 1 - k(s)G(s), from the exact
factorisation det(sM_cl - A_cl) = det(sM - A) det(sI - A_c) f(s)), run at SAVE
time (`--wind 1`, default). None of the recent runs reached a final save, so it
was run now at ne=200: both K+B+C stationary points (7.848591 and 7.950855): **[2026-09-25: the winding results stand; both points were stalls]**
**0 closed-loop eigenvalues right of Re = -0.025** (plant alone: 1), ~620
contour points each -- unconditionally stable. At the cavity the plant's count
is supplied once (`--wind_pplant`, design-independent, by shift-invert).

**Closed-loop Hermite block by default** (user): when two near-line closed-loop
eigenvalues approach each other near the margin, compress the reduced pencil
onto their right/left eigenvectors (available from the O(k^2) tracking) and
impose the same Hermite condition as for the compensator; the new cost is the
block gradient U^H (dA_cl/dz) V -- ~8 calls of the existing gradient routine,
negligible. To build, test at ne=100, and make default.

**Replay of ne=100 round 5** (rows 0.4-4 % below the certificate in the loop):
in isolation the SQP returned nothing admissible, and over 244 evaluations the
rows near -2.45 never fell > 0.2 % below the true local maximum -- NOT
reproduced. Differences from the loop: carried trust radius and Hessian,
the evolved followed set, and ROW SCREENING (--row-screen 0.05: rows below
95 % of the max get a ZERO gradient, so the QP treats them as constant while a
step can raise them) -- on in the ne=100 runs only, where the gaps appeared.
Next: controlled A/B at ne=100, screening off vs on.

## 19. The ne=100 gaps diagnosed: new resonances and narrow twins -> refinement ON (2026-09-24)

A/B at ne=100 (K+B+C from the transferred start): the gaps between the SQP's t
and the full certificate occur with row screening OFF too (round 0: t 8.146,
certified 9.861; round 1: 8.614 vs 9.540) -- screening is NOT the cause. With
the rows printed at the returned design (CODESIGN_DEBUG_ROWS):
- round 0: only ONE peak was within 5 % at the start (3.315); the SQP lowered
  it (8.117, reduced = full) while a resonance at w = -1.357, below 95 % of the
  max at the start and so never enrolled, rose to 9.861 -- no row near it
  (nearest member 0.107 away). No enrolment rule anticipates a peak that
  starts under 95 % and rises > 20 % in one inner solve;
- round 1: that peak was enrolled (member -1.3571), but at the returned design
  following reported its top at -1.3625 (8.6156, reduced = full) while the
  certificate found 9.540 at -1.3570: two local maxima 0.0055 apart, the higher
  one narrower than the window scan's spacing (0.01).
The model was exact at these witnesses (1e-9 .. 1e-15, measured with the model
the SQP used: anchoring at the previous round's peaks happens at round start,
the witness check after certification). Both are what IN-SOLVE REFINEMENT is
for: after each accepted step the reduced level-set certificate gives the
model's TRUE band maximum (rigorous for the model; the model is exact here)
and adds it to Omega mid-solve. Same round with `--sqp-refine 1e-4`:
t 9.003048 = certified 9.003048, and a LOWER result (9.003 vs 9.861: the step
no longer overshoots into the new peak).
**Decision: refinement ON in the nominal loop.** Section 11 had reclassified
it as compensation for working-set defects; it is also the only mechanism for
resonances no working set can anticipate. With the reduced model it costs no
full factorisations.

On the closed-loop Hermite block (user): a modelling device for the SQP's
subproblem -- it belongs wherever the SQP linearises (the reduced model in the
nominal loop; the full model only if the inner loop runs on it). Certification
does not need it: stability is a statement about eigenvalue locations; the line
certificate refuses a coalesced pair ON the line (safe) and the winding count
counts with multiplicity.

## 20. Closed-loop Hermite block, ON by default in the SQP (user, 2026-09-24)

Environment reset on 2026-09-24 (working tree restored from
codesign_20260924.tar.gz; torch 2.14, pyGRANSO + patch, Clarabel/OSQP/DAQP
reinstalled; frozen path bit-identical, 10.23128516441147; ne=100 starts
regenerated by tools/transfer_ne.py: K+B+C 10.990169, K+FB+FC 3.935632 --
identical to before).

**The block** (`--sqp-clblock on`, default for --inner sqp; the frozen pyGRANSO
path is unaffected): the two rightmost near-line closed-loop eigenpairs of the
reduced pencil (residual-checked against the full operator) -> orthonormal
bases of their right and left eigenvector spans, biorthonormalised with M_cl ->
2x2 block B = Yt^H A_cl Xq; Hermite's criterion on B + (eps + mg) I (the
admissibility line of the per-eigenvalue rows) as a 4x4 PSD cone with its own
elastic slack; block-entry derivatives Yt_a^H (dA_cl/dz) Xq_b through D.grads
exactly as the sigma block's; H's derivative via trace and determinant
(`hurwitz.hermite_from_block`, now shared with the compensator block). The two
eigenvalues leave the per-eigenvalue rows. The subproblem takes a LIST of
Hermite blocks (compensator, closed loop).
Checks (ne=100 K+B+C transferred start, reduced basis fixed): block
eigenvalues = the pair's to all digits; d(trace), d(det) vs finite differences
4e-8, 2e-8; dH at fixed scale 2.7e-8 (the per-evaluation scale rho is a
congruence of H -- positive definiteness, i.e. the constraint, unchanged; the
compensator block uses the same convention); lambda_min(H0) > 0 with both
eigenvalues left of the line; an SQP solve with the block on/off: same step
(t 10.280156 vs 10.280154).
Where it belongs (user question, section 19): wherever the SQP linearises --
the reduced model here; certification needs no block (line certificate +
winding count, with multiplicity).

**hurwitz.py selftest** had reported 51/200 inertia mismatches, in the
delivered code too: its 1e-9 relative zero threshold misclassified eigenvalues
of degree-16 Bezoutians (spreads >= 1e9). Counted by sign: 0/200. The
property the SQP uses -- H positive definite iff Hurwitz -- held in all 400
full-size and 20,000 2x2 draws. Test fixed.

## 21. First ne=100 runs with refinement + closed-loop block; three defects (2026-09-25)

**K+B+C** (12 rounds): 10.990 -> 9.725 -> 9.507 -> 9.164 -> 8.665 -> 8.665 ->
8.583 -> 8.291 -> **7.956567** (t = certified in every round so far) -> refused
x3 (rounds 8-10: 0.8 %, 0.4 %, 1.0 %) -> **7.944454**, winding-certified (0
eigenvalues right of -0.025). Both Hermite blocks active at the round-7 design
(closed loop: -0.0572-0.911i, -0.181-1.312i, lambda_min(H) 2.3e-3; compensator
rightmost pole -0.050828). Not converged (round budget).
Round 8 dissected (dumps): a followed member (-0.913273) ended on a LOWER TWIN
top (-0.9060, 7.973) while the true top 0.0048 away reached 7.993; the model
was exact there (4e-15) and refinement added nothing. A direct test at the
candidate with the run's in-solve working set shows following can land on
either twin, and that refinement at the right level answers correctly; the
returned iterate's level (7.931) and refinement's silence are only consistent
if refinement's certificate was INCONCLUSIVE there -- not observable from the
old records. Refinement now logs every decision (t, certified max and where,
concluded or not, above or not, members within 1e-3) into the dumps.
**Defect 1 -- the certificate could return 'certified' after FINDING a higher
level set** (certify_sparse.hinf_certified_sparse): when certify_greater found a
level set above gam(1+rtol) but the climb from its sampled candidates did not
locate it, it returned (gam, certified=True). Now inconclusive (False) with the
witness; counter N_CLIMB_FAILED. Not reproduced at the round-8 candidate (full
model, 6 lower bounds: all correct), so not shown to be round 8's cause.
**Defect 2 -- the outer loop ignored the certificate's ok flag**: an
inconclusive value (a lower bound) was accepted as certified. Now an
inconclusive certificate never accepts a round; its witness is enrolled.
Frozen path bit-identical after both (case B, 10.23128516441147); the
27-case regression should confirm the branch never fires in the frozen suite.
**Defect 3 -- tools/transfer_ne.py and the Sobolev map**: it built the target
mesh with sob_len 0 (driver: 0.8), saving xi = field; the run then smoothed the
field again (18 % error) and K+FB+FC started at 6.18 instead of 3.94 -- its
12 rounds went 6.18 -> 3.930616 (winding-certified, below the transferred
3.935632, but not a clean comparison). Fixed: sob_len matches the driver, the
Packer gets xi (its contract), and the round trip smooth_map(xi) = field is
checked (1.6e-16). K+B+C has no fields: unaffected.
Both cases relaunched (tools/n100_kbc.sh, tools/n100_fb80.sh, 20 rounds,
--converge); first runs kept in runs_old/.

## 22. The remaining ne=100 refusals: a post-hoc step clip, and certificates on reduced pencils (2026-09-25)

**Refinement saw the right model, at the wrong design.** Round 8 (first K+B+C
run) and round 7 (a later run) were refused although refinement ran at every
iterate. The refinement log (now in the dumps) showed its certificate
concluding at every call with a model maximum = t, while the model saved right
after the round (debug dump) held the missed peak exactly. Cause: the frozen
**tau step governor** -- set after a continuation refusal -- CLIPPED the SQP's
returned step after the solve (round 7: 0.0629 -> 0.0117). The certified design
was a point the SQP had never evaluated; its peak (8.231 at +1.788) existed at
neither endpoint; t was the unclipped design's. Fix: the cap is imposed INSIDE
the SQP as a total-displacement cone ||z - z0||_2 <= rho in every subproblem
(`max_disp`; checked: largest displacement evaluated 0.005000 with a 0.005 cap,
0.66 without), and the post-hoc clip is skipped for the SQP (pyGRANSO path
unchanged).

**Certificates on reduced pencils.** On the run's saved model at the round-7
candidate, the sparse decision procedure took 60 iterations and 137 s and
could not locate the 8.231 peak (now reported inconclusive, section 21); the
dense level-set algorithm (Boyd-Balakrishnan / Bruinsma-Steinbuch, all
eigenvalues of the ~200-dimensional even pencil, `hinf_dense_levelset`) found
it in 0.26 s. Every certificate on a REDUCED pencil (refinement, provisional
rounds, pyGRANSO band row) now uses it (`--reduced-cert dense`, default);
refinement starts it at t -- the question it asks. Checked against the sparse
procedure and a 12001-point sweep on the full ne=100 model (agreement; 5-12x
faster even at full size), with a final local climb to the top.

**The sparse procedure and ARPACK.** `eigs_near` returned PARTIAL eigenvalue
sets on ArpackNoConvergence, and `certify_greater` sized eigenvalue-free disks
from them -- a possible false 'no crossing'. Bisecting such intervals instead
made the frozen certificates explode (the regression stalled >15 min): partial
convergence is frequent in the frozen suite. The frozen behaviour is KEPT and
the events COUNTED (N_DISK_UNTRUSTED); an open item for the certification
stack, to measure in the cluster regression. Frozen path bit-identical.

**Winding acceptance at the margin.** A continuation refusal of a stable
candidate (round 6: path midpoint alpha -0.0479) is now accepted when the
winding count right of Re = -eps (the margin itself; was -eps/2, leaving the
(-eps, -eps/2) gap) is 0 -- an unconditional proof of alpha <= -eps, the line
certificate having proved nothing lies ON the line. `--cont-accept-winding`
added to the ne=100 scripts.

**BLAS threads.** OMP_NUM_THREADS was unset in the sandbox; with two runs on 2
cores a 178x178 eigenvalue solve took 1.9 s instead of 0.03 s (60x). Every
wall-time figure measured while two processes ran is inflated (factorisation
counts are not). The run scripts now pin 1 thread per process, as env.sh does.
**Winding acceptance at the margin, made conclusive.** On the round-6 candidate
(alpha -0.050123, pinned 1.2e-4 from the margin) the count right of Re=-eps was
INCONCLUSIVE (the tracker's step floor ~6e-5 on this band is as coarse as the
feature the near-line eigenvalue makes); right of -eps/2 it was 0. With the
gate's near-line eigenvalues as zeros_hint and hmin_frac 1e-8: count right of
Re=-eps = 0, conclusive (957 points). The acceptance path now passes both; an
inconclusive count is logged and never accepts.

## 23. ne=100 K+B+C on the final code (2026-09-25)

Refinement (dense reduced certificate) + closed-loop Hermite block + step cap
inside the SQP + winding acceptance at the margin (with hints):
10.990 -> 9.725 -> 9.508 -> 9.164 -> 8.882 -> 8.594 -> 8.216 ->
**7.929704** (round 6: continuation refused, winding count right of Re=-eps
= 0, 957 pts, accepted) -> 7.927101 -> **7.926300**; t = certified in every
round (<= 1e-6 relative). KKT residual at 7.926300: 4.1e-4 x gradient scale
(8 active peaks; compensator pair on the margin -0.05002 / -0.05016; closed-
loop abscissa -0.0502, row active). Every refusal of the earlier runs is
explained and removed: the twin-peak / post-hoc-clip mechanism (22), the
sparse certificate on reduced pencils (22), the continuation refusal of a
stable margin-pinned design (winding at the margin, 22).
**Converged** (run kept in runs_old/results_n100_kbc_6_converged): fixed point
under --converge at **7.925385** (rounds 10-13), winding-certified (0
eigenvalues right of -0.025); independent KKT residual 1.9e-4 x gradient scale **[2026-09-25: a stall; later runs reached 6.40 (26-29)]**
(9 active peaks; compensator pair -0.05002 / -0.05005, Hermite lambda_min 3.5e-5;
closed-loop abscissa -0.0503, active). The loop reported "stalled, not
converged": the SQP's first subproblem FAILED in rounds 11-13 (code 12), after
rounds 9-10 ended on budget with trust radii ~1e-4. At the final design the
same subproblem SOLVES with a fresh Hessian (all trust radii, block on/off):
the carried quasi-Newton matrix (--sqp-carry) is the suspect. Fix: on a
subproblem failure, reset H to a scaled identity and retry once; statuses
recorded (sub_failures). Also: active frequencies enrol a sigma_2 block only if
the sigma_1-sigma_2 gap <= 0.15 (`--sigma2-active-gap`, default off = frozen
policy): K+FB+FC's flat optimum had enrolled 43 blocks with gaps ~0.87 (no
kink), and its rounds grew to 930 s. Clean reruns of both cases on this code.

## 24. Clean ne=100 rerun of K+B+C; a fake witness, the compensator tail, a scaling defect (2026-09-25)

**Fake witness (bug, fixed).** Every round of the first clean rerun logged
"reduced at the witness 7.10 vs certified 7.94 (rel 1e-1)", and anchoring there
never grew the model (dim fixed at 95). The model was exact (1e-13 in an offline
rebuild at the logged frequency). The witness was not: when the certificate
proves at its FIRST question that nothing exceeds gam_lb (1 + rtol) -- the
normal case once the SQP's own maximum is the norm -- it returns gam_lb and no
frequency, and `certified_norm` fell back to the argmax of the COARSE full
sweep (61 points over [-30, 30] with --cert-sweep 61), e.g. w = 2.0 where sigma
was 8.08 against 10.91 certified. Consequences: the 10 % "model error" was
spurious; one full factorisation per round anchored a non-peak; the grid point
was enrolled in Omega and kept by compaction. Fix: with a reduced model
attached, the witness is the model's fine-sweep argmax polished on the model
(free); the frozen path (no model) is unchanged. After the fix the logged
errors are 1e-8 .. 1e-15. Nothing in these notes had drawn a conclusion from
the spurious numbers.

**The rerun, with the fix (tools/n100_kbc.sh):** 10.908 -> 9.720 -> 8.730 ->
8.290 -> 7.985 -> 7.961 -> 7.951 -> ... -> 7.935 -> 7.931 -> **7.929388**
(20 rounds, 3,005 s, 3,807 evaluations), certified <= t (1 + 2e-6) in every round,
winding-certified (0 eigenvalues right of -0.025). KKT residual (the tool, with
the scaling fix below): **8.3e-4 x gradient scale**, 10 active peaks, the
compensator pair BOTH exactly on the line (-0.0500000, Im -0.420 / -0.376).
The earlier converged run (23) ended at 7.925385 (KKT 1.9e-4) with the pair
nearly coalescent (-0.05002 / -0.05005, Im 0.007 apart): a different local
point 0.05 % higher, same active-peak structure.

**Block-normalisation defect (fixed).** Rounds 18-19 of the rerun ended at
their first subproblem (Clarabel NumericalError / InsufficientProgress, also
after the Hessian reset), and the KKT tool failed on the end point
(NumericalError). With both poles ON the line the block's Hermite matrix
vanishes (4 eigenvalues ~1e-13: each boundary root costs rank 2 because the
construction squares p into a real polynomial), and the block was normalised
by its trace -- the gradients were multiplied by ~1e13. Now the normalisation
is max(|tr H0|, 1e-3 max_j ||dH_j||) in the SQP (compensator and closed-loop
blocks) and scale-free (block gradients on the peaks' scale) in
tools/kkt_residual.py; the tool reproduces 1.94e-4 on the 23 end point and
gives 8.3e-4 on the new one. (The subproblem failures in 23 happened at a
design whose H0 was NOT degenerate -- lambda_min 3e-7, trace 1.3e-3 -- and the
carried Hessian remains the suspect there.) Continuation of the rerun with the
fix: tools/n100_kbc_cont.sh (below).

**Why the tail is slow: the compensator pair on the line.** Rounds 7-10 of the
first rerun moved t by 1e-3 per round of 40 iterations. Diagnosis at a tail
design (offline, same Omega, followed tops, sigma_2 blocks):
- the line search accepted alpha = 1/64 each iteration; rejected steps had
  halved the box radius to 1.3e-4, and alpha < 1 never re-grows it;
- the peak rows are NOT what failed (their model error was 1e-6 at the accepted
  steps); the COMPENSATOR was: the pair at -0.0500738 +- 0.015i, eigenvalue
  condition 62, and a step the block's linearisation kept on the line put a
  pole at +0.011 (alpha = 1), -0.046 (1/16). The block's Hermite matrix is
  exact to first order (gradient vs FD 1e-10) and smooth; its two small
  eigenvalues are 1.7e-4 against 0.5 for the others, and a second-order residual
  of 0.5 % of the step's linear change lands in the small ones;
- remedies tried offline from the same design (15-25 iterations each): merit
  weight 1 on the epigraph rows instead of 100 (identical path: the pole line
  decides), Oren-Luenberger scaling of the identity start (worse: tiny steps),
  trust radius shrunk to the accepted step (collapses to 2e-6 without curvature),
  EXACT curvature of the compensator block in the QP Hessian
  (-<Lambda, d2H>, built from the fixed-basis trace/determinant, verified vs FD
  to 1e-7) + a cushion of mg on the block's line + radius shrink (steadier:
  radius 3e-5 .. 2.6e-4, alpha 0.25-1, 62 evaluations for 25 iterations), a
  2-norm trust region, a second-order correction (oscillates: the corrected
  steps flip). All reach 7.9366-7.9394 from 7.9403; the plain SQP 7.9351 in 15
  iterations (85 evaluations). None is a decisive improvement; none is made
  default. Kept as options (sqp_inner globals CTRL_HESS, CTRL_CUSHION,
  TR_SHRINK, TR_BALL, SOC_CTRL, H_SCALE; all off) and the merit now 'max'
  (`--sqp-merit`, weight 1 on the rows: the exact threshold, the duals of the
  rows sum to 1).
- the run itself got through: rounds 11-15 moved 7.945 -> 7.929 once the pair
  separated and settled on the line.

**Evaluation cost (user: make it fast before continuing).** Profile of one
K+FB+FC SQP evaluation at ne=100 (382 design variables, |W| = 210, 48 followed
members, 8 sigma_2 blocks, model dim 161): 4.6 s, of which peak following 3.7 s
and its 21-point window scan alone 2.6 s (1,008 of 1,564 reduced solves). The
scan was added (16/17) to catch a twin rising inside a member's window BEFORE
in-solve refinement was on; refinement now catches any peak above t (1 + 1e-4)
more than 1e-3 from a member at every accepted iterate, so the scan is
redundant there: `--follow-scan 0` in the ne=100 scripts (rows identical at the
profiled design to 2e-14; an offline K+B+C SQP solve of 12 iterations followed
the identical path). Also: Newton starts at the previous top without a value
comparison, the curvature is carried per member and updated by secant (one
evaluation per Newton step instead of three; a finite difference only when none
is known or a carried one fails), and the SQP's top tolerance is
|dsigma/dw| <= 1e-7 sigma (value error g1^2 / 2|g2| ~ 1e-15). Result:
4.6 -> 2.2 s per evaluation (following 3.7 -> ~1.2 s). The remainder: 160
value-only rows (screened), ~50 rows with vectors, the blocks.

**Runs relaunched on this code** (merit 'max', witness and scaling fixes,
follow-scan 0): K+B+C from the ne=100 start (tools/n100_kbc.sh); K+FB+FC from
its round-7 checkpoint 3.913182 (tools/n100_fb80.sh; the earlier rounds 0-7 in
runs_old/results_n100_fb80_5_oldsqp_r0-7: 3.935632 -> 3.933947 -> 3.922682 ->
3.918911 -> 3.917770 -> 3.916066 -> 3.914052 -> 3.913670 -> 3.913182, every
round accepted, ~15-20 min per round). K+FB+FC's optimum is flat: sigma_1 =
3.9130 at every member from -0.78 to +0.51 (23 sigma_2 rows active), an
equal-ripple response over a band -- the rich actuation flattens it, which is
why |Omega| keeps growing (156 at the start; 279 restored from the round-7 checkpoint).

**ne=100 K+B+C on the new code: converged at 7.657552** (tools/n100_kbc.sh, 20 **[2026-09-25: 'converged' = a fixed point of that code; a stall -- later runs 6.40 (29)]**
rounds, 1,149 s, then tools/n100_kbc_cont.sh, 15 rounds, 1,468 s):
10.908 -> 9.805 -> ... -> 8.093 (round 12: continuation refused, winding count
right of -eps = 0, accepted) -> 7.983 -> 7.907 -> 7.886 -> 7.850 -> 7.846865
[KKT 1.5e-4 here, yet not a minimum: see below] -> 7.843 -> 7.837 (round 5 of
the continuation: continuation refused, winding 0, accepted, chain restarted)
-> 7.792 -> 7.697 -> 7.657742 -> **7.657552** (rounds 10-14 identical to 1e-9).
Checks, independent of the loop (tools/dense_validate.py, new: dense
eigenvalues and the dense level-set certificate on the FULL model):
- H-infinity: dense level-set 7.65755699, certified, vs recorded 7.65755248
  (5.9e-7, inside cert-rtol 1e-6); 20,001-point sweep max 7.657543;
- closed-loop abscissa (dense) -0.050283: margin 2.8e-4, the row active;
- the compensator pair has COALESCED on the line: a double pole at
  -0.0500000 - 0.37344i (computed as -0.04999999006 / -0.05000017465, the
  sqrt(eps) splitting of a defective eigenvalue; Hermite block PSD to 1e-17 at
  the line, positive definite for the line moved by 1e-7). This is the
  Burke-Overton structure the block was built for (nonderogatory double
  eigenvalue at the constraint), reached and held by the SQP;
- KKT residual 2.1e-4 x gradient scale (23 near-active peaks, block multiplier
  |Lambda| 8e3: the compensator constraint is strongly active);
- winding-certified.
Lower than both earlier ne=100 end points (7.925385, 7.929388) and than the
ne=200 reduced-certification result (7.848591): a different basin, entered
through two winding-accepted rounds where the continuation chain could not
track an eigenvalue pair.
**A KKT residual of 1.5e-4 did not mean a minimum**: at 7.846865 the tool's
active set (peaks within 2e-3 of the max: 33 of them, clustered on flat tops)
balanced the gradients, and the loop then descended 2.4 % further. The residual
with a loose active set is a necessary check, not a certificate of local
optimality; the fixed point of the loop plus the residual is what we report.
**--converge fix**: the fixed-point test counted gains of 1e-9 as progress and
never fired (five identical rounds); gains below cert-rtol no longer count
(acceptance of a better design is unchanged).

## 25. ne=200 K+B+C; the double compensator pole on the line is a degenerate constraint (2026-09-25)

**ne=200 K+B+C on the ne=100-validated code** (tools/n200_kbc.sh, from the
frozen 7.964064): 7.955 -> 7.954 -> ... -> 7.940 -> 7.926 -> 7.914 -> 7.893 ->
... -> 7.824183 -> **7.824127** (rounds 14-17 identical), FIXED POINT under
--converge (the fix of 24 fired), 18 rounds, 1,914 s, 3,467 evaluations,
winding-certified. Dense validation on the full model: level-set 7.82412656
(recorded 7.82412656, 8.8e-12), closed-loop abscissa -0.055775 (margin 5.8e-3).
Lower than the earlier ne=200 results (7.848591 reduced certification, 7.950855
every-round SQP). But **not stationary**: KKT residual 3.55e-3 x gradient scale
(6 active peaks, block multiplier |Lambda| 812), and the loop itself said
"stalled, not converged": trust radii collapsed to 1e-10 with every step
refused (rounds 15-17).

**Cause: the compensator pair is a double pole ON the line** (-0.0500000 -
0.39796i, Im parts 1e-5 apart) -- the same structure the ne=100 optimum has
(24). At a double pole the poles move like sqrt(step) in every direction that
changes the discriminant: the pole-based merit (compensator real parts against
the line, rounding tolerance 1e-8) sees an infinite directional derivative, so
no step of any size passes the line search. Offline, from the stalled design:
- exact-line HERMITE MEASURE in the merit and admissibility (new option
  MERIT_CTRL = 'hermite': the pair enters through lambda_min of its Hermite
  matrix at the exact line, built from the pair's sum and product, which stay
  accurate at a defective pair; H >= 0 <=> both poles on or left of the line,
  exactly): the solve moves again (radius 2e-3, alpha 1/32 .. 1/8), best
  admissible 7.824127 -> 7.823697 in 12 iterations;
- but its iterates wander: the last one had the pair split to -0.0409 / -0.0591
  (0.009 PAST the line) at a Hermite violation of only 1.6e-7 in the block's
  units. At a double boundary root the Hermite eigenvalue is QUADRATIC in the
  pole's excursion, so a linear penalty on it -- the merit's, and the QP's
  elastic slack at price 1e3 -- is not exact. This is a constraint-
  qualification failure, not a tuning problem: lambda_min(H) has zero gradient
  in the splitting direction, the multipliers are large (|Lambda| 8e3 at
  ne=100, 812 at ne=200), and the SQP can only creep;
- a cushion on the block's line alone makes it worse (the merit gives no credit
  for the restoration the QP then asks for).
Burke-Overton: optimisation of spectral quantities typically ends at such
nonderogatory multiple eigenvalues; the smooth reformulation (Hermite, trace/
determinant) removes the non-Lipschitz pole map but not this degeneracy. The
standard remedy is to IDENTIFY the manifold -- "the pair is a double pole on the
line": three real equality conditions (discriminant = 0, complex; Re of the
double root = -eps), smooth with nonzero gradients -- and run the SQP on it,
checking the multiplier signs to leave it when that pays (VU / active-manifold
SQP). Not implemented; a design decision for the user (below). Options kept,
all off by default: MERIT_CTRL, NU_H, HMIN_REL_TOL, CTRL_CUSHION, CTRL_HESS,
SOC_CTRL, TR_SHRINK, TR_BALL, H_SCALE (sqp_inner globals).

**Where the three K+B+C runs ended, compensator-wise:** ne=100 7.925385 (pair
nearly coalescent, sep 0.007), ne=100 7.929388 (pair on the line, separate,
sep 0.044), ne=100 **7.657552** (double pole on the line, KKT 2.1e-4),
ne=200 **7.824127** (double pole on the line, KKT 3.6e-3, stalled). The best
designs have the double pole; reaching and converging at it is the tail
problem.

## 26. The pair on its face of the line; warm line search; invariants (2026-09-25, user: apply the fix)

**Why the two defects of 24 survived so long (user's question).** Both sat in
branches that were rarely taken until the inner solve became accurate:
- the witness fallback (`w` = argmax of the coarse full sweep) runs only when
  the certificate proves the lower bound at its first question. Before the SQP
  with in-solve refinement, the inner t was below the true norm (rows missed
  peaks), so the certificate always CLIMBED and returned a genuine witness. From
  2026-09-24 on, t = certified to 1e-6 made the fallback the normal path; the
  sweep had been cut from 241 to 61 points (coarse-then-fine, 4.2) and the
  fallback's grid argmax became a point up to 0.5 away from the peak. The log
  printed "rel 1e-1" every round; it was read as a model question, never checked
  against the full model. Now asserted every round (WITNESS INVARIANT,
  witness_invariant_violations in the result json; tools/check_invariants.sh);
- the fixed-point test predates the SQP: with pyGRANSO a stalled round did not
  improve at all; the SQP improves by 1e-9 per round at a fixed point, which the
  strict test counted as progress. Fixed in 24 (gains below cert-rtol).

**The compensator pair on its face of the line (the fix proposed in 25).**
The two rightmost compensator poles enter through their sum s and product p
(Schur-based invariant subspace: hurwitz.cluster_block now uses the ordered
Schur form; the eigenvector construction breaks down at a double pole --
|U| 7e7 and trace error 2e-8 on a Jordan test matrix, against 1e-15 now).
With Delta = s^2/4 - p (the pair is s/2 +- sqrt(Delta)):
  c_u = Re(s)/2 - line + kappa,  c_a = Re(Delta)/sc,  c_b = Im(Delta)/sc.
The feasible set is exactly F = {c_u <= 0, a <= u^2, b^2 <= 4 u^2 (u^2 - a)}.
Its face on the line is S = {c_u <= 0, c_a <= 0, c_b = 0}: "equal real parts,
split vertically or double, left of the line" -- S is inside F, equal to F on
the line, and at a double pole ON the line the tangent cone of F is exactly
{dc_u <= 0, dc_a <= 0, dc_b = 0}. All three rows are smooth with non-vanishing
gradients (the Hermite matrix's small eigenvalue is quadratic in the poles'
excursion there, the poles themselves move like sqrt(step)).
- ENTRY (per SQP solve): the pair nearly defective (|l1 - l2| <= 0.05) with its
  rightmost pole within 1e-3 of the line, OR both poles on the line (1e-6), any
  separation. The start is projected onto S (Gauss-Newton, compensator
  coordinates only, moves ~1e-7).
- QP: c_u <= 0 and c_a <= 0 as linear rows, c_b = 0 as an equality; the
  Hermite block is not built.
- every trial point is RETRACTED onto S (Gauss-Newton on c_b = 0, plus c_u = 0 /
  c_a = 0 if the step left them positive): only A_c (8 x 8) is evaluated, no
  plant work, residual 1e-16. The pole-based merit then sees smooth real parts.
- EXIT when the pair is strictly inside (c_u < -10 kappa) and the equality's
  multiplier |mu_b| > 1e-6 (inside, F allows unequal real parts).
- on S: 2-norm trust region shrunk to twice the accepted step (PAIR_BALL);
  elsewhere the box as before.
- KKT tool: the S-rows residual (mu_u, mu_a >= 0, mu_b free) is reported next to
  the Hermite-block one; they agree (2.08e-4 at the ne=100 7.657552 point,
  3.55e-3 at the ne=200 7.824127 point) with multipliers 0.3-0.8 instead of
  |Lambda| 8e3 -- a well-posed stationarity measure at the double pole.
Offline, same starts, same budgets:
| start | before | S |
|---|---|---|
| ne=200 stall 7.824127 (double pole) | 7.823964, 125 evaluations / 15 it | **7.792931**, 63 evaluations / 25 it |
| ne=200 stall 7.902673 (both on line, 0.057 apart) | stuck (alpha 1e-4) | **7.799084**, 61 evaluations / 25 it |
| ne=100 mid-path 7.940301 (pair 7e-5 inside, sep 0.015) | 7.935197, 154 evaluations | 7.934350, 113 evaluations (exits S at once) |

**Warm line search.** Every search started at alpha = 1 and walked down to the
same 1/64 .. 1/128 each iteration: 6-7 evaluations per iteration, 44 % of all
evaluations in the tails. It now starts at twice the last accepted alpha
(ALPHA_WARM). Rounds went from 150-300 evaluations to 60-90 at the same number
of iterations.

**Where the time went (user: 14 / 32 / 44 min unacceptable).** Measured on the
three finished runs: 90 % of the wall time is inside the SQP (evaluations), 10 %
certification, gate and anchoring. Per evaluation 0.42 s at ne=100, 0.46 s at
ne=200 (after the follow-scan cut). The evaluations were wasted, not slow:
~6 per iteration (line searches from alpha = 1), 44 % of the iterations refused
outright at the double pole, and the last 5-10 rounds of every run spent at a
stall the loop could not leave. Both are defects of the implementation (line
search, the pair's constraint), not of the method; the fixes above remove them.
K+FB+FC's rounds are longer for a different reason: 280-320 frequencies in
Omega (an equal-ripple optimum), 23-25 sigma_2 blocks, 382 variables.

**S in the runs (2026-09-25, same day).** Refinements found on the way, each
from a replay of a stalled round:
- entry also when both poles sit on the line with any separation (ne=200 stuck
  at 7.902673 with the pair at -0.05 +- 1e-9, 0.057 apart; S: 7.799 in 25
  iterations offline);
- release S only when S is STATIONARY (active-set rule), the pair strictly
  inside and |mu_b| > 1e-2: releasing on the first QP lost the face at a
  near-double pole (7.901729 in 62 evaluations, against 7.878932 in 27 on S);
- the witness invariant fired in the first S runs (sigma_1 7.929682 at the
  chosen witness, 7.929851 certified: a DIFFERENT near-equal peak): the witness
  is now the best, by FULL sigma_1, of the SQP's maximal row (passed by the
  SQP), the model's polished fine-sweep argmax and its top peaks. No violation
  since;
- STALL RECOVERY: a solve ending on a collapsed radius (code 6) or with failed
  subproblems no longer hands its quasi-Newton matrix and radius to the next
  round. The ne=100 run sat at 7.770563 four rounds; the same design with H = I
  went to 7.665533 in 25 iterations offline.

**ne=200 K+B+C on this code (tools/n200_kbc.sh, 40 rounds, 3,685 s with another
run sharing the two cores): 7.964064 (frozen) -> 6.825715**, still descending
at the cap (continuation running). Dense validation on the full model:
level-set 6.82571466 (recorded 6.82571466, 6e-14); closed-loop abscissa
-0.1676 (the abscissa constraint is no longer active: every earlier end point
had it at -0.050); compensator pair on the line (-0.050001, the S face, sep
5e-4). KKT residual 2.7e-4 (42 near-active peaks: the response is flattening,
|Omega| 156 -> 407). All earlier "converged" K+B+C values (7.85 .. 7.96 at
ne=200, 7.66 .. 7.93 at ne=100) were stalls of the compensator pair, not
minima.

**Three poles on the line.** At the ne=200 end point the compensator has a
double pole at -0.05 - 0.506i AND a simple pole at -0.05 + 6.598i. The pair was
chosen as "the two rightmost", i.e. one pole of the double pole plus the simple
one, and the double pole was split between the block and the per-pole rows.
hurwitz.pick_pair: when three or more poles are within 1e-3 of the rightmost
real part, the pair is the closest two of them (if within 0.05); used by the
SQP, the KKT tool and tools/dense_validate.py. The KKT tool now also carries a
row per OTHER compensator pole on the line.

**ne=200 K+B+C, continuation (tools/n200_kbc_cont.sh): 6.825715 -> 6.438335**,
FIXED POINT after 29 rounds (4,094 s, three runs on two cores). Dense validation:
level-set 6.43833485 (recorded 6.43833487); closed-loop abscissa -0.168071;
compensator: double pole at -0.050001 - 0.50638i (from its sum and product),
simple pole at -0.050001 + 6.598i. KKT residual **3.2e-5 x gradient scale**
(136 near-active peaks, the double pole's face rows and the third pole's row):
a stationary point. From the frozen 7.964064: -19.2 %.
**ne=100 K+B+C** (tools/n100_kbc.sh, 40 rounds, 2,817 s, three runs on two
cores): 10.990 -> **6.627392**, dense-validated (6.62739155), closed-loop
abscissa -0.158, pair at -0.050001 (sep 2.8e-3), not yet stationary (2.2e-3);
continuation running.

## 27. Curves, bounds, and where the time goes (2026-09-25, user questions)

tools/curves.py + tools/plot_curves.py -> figs/sigma12.png, sigma12_zoom.png,
controller_bode.png, gamma_vs_time.png (records: frozen K+B+C ne=200, new
K+B+C ne=200 6.438335 and ne=100 6.401849, K+FB+FC ne=100 3.902989).
- Floor (v6 4.3): sigma_1^cl(w) >= sigma_2^ol(w) pointwise (rank-one feedback,
  interlacing), sup = 3.509066 (3.509084 at ne=100). K+B+C new: +83 % (frozen
  +127 %); K+FB+FC: +11 %. The sensor-/actuator-limited bounds (9.832 / 9.660)
  hold only with a position fixed and do not apply to K+B+C.
- The new K+B+C responses are FLAT at gamma over roughly [-7, 4] plus a second
  plateau-level peak at w ~ +11.5 .. +12 (819 grid points within 0.2 % of the
  max at ne=200); K+FB+FC is flat over [-15, 18] (2548 points): the H-infinity
  optimum is (nearly) all-pass, which is why Omega grows to 300-400 members.
- Positions moved from -1.61 / +1.60 (frozen) to -2.335 / +2.335. The K+FB+FC
  compensator is essentially a static complex gain (|k| ~ 270, phase 108 deg):
  the fields do the shaping.
- gamma(t): within 1 % of the end value after 52 % (ne=100) and 60 % (ne=200) of
  the wall time. A progress-based stop (gain over 3-5 rounds < 0.1-1 %) would
  have stopped both runs on the 7.7-7.93 plateau, 20-24 % above the end value:
  the slow phases are plateaus BEFORE large descents, not a final tail. The
  ne=200 plateau at 6.83-6.80 (45-61 min) ended at the stage restart (fresh
  quasi-Newton matrix and re-derived Omega): stale SQP state without a code-6
  failure is an efficiency defect still open. **[2026-09-25: fixed in 28: no quasi-Newton memory across rounds]**

## 28. Plateaus = stale quasi-Newton memory; batched rows and following (2026-09-25, user: implement)

**Fields** (figs/fields.png): K+FB+FC ne=100 (3.9012) against the frozen ne=200
nodal design (3.9254): the same main lobes (actuator centred near x = -2.5 with
a quadrature component at -5..-1, sensor near +2.5 with its quadrature at 0..5).
The ne=100 basis design carries a grid-scale ripple (+-0.04) in both tails,
where the gain is insensitive to the field: 80 Laplacian modes with no
smoothness penalty leave those directions free. [2026-09-25: NOT cosmetic -- the ripple is doing work in the ne=100 discrete model (section 29).] A Sobolev
weight on the modal coefficients (the nodal path's smoothing) would remove it.

**What the plateaus were.** Replays of plateau rounds of the ne=200 run, same
design, same Omega, fresh quasi-Newton matrix (H = I) instead of the one the
run carried across rounds (--sqp-carry):
| start | run (carried H) | replay, H = I |
|---|---|---|
| round 20, 7.509409 | 7.450646 after 40 iterations | **7.090410** after 40 (106 evaluations) |
| round 33, 6.864869 | 6.825715 after ~280 iterations | **6.806314** after 25 (64 evaluations) |
The damped BFGS matrix accumulated over hundreds of iterations and many
active-set changes (peaks entering and leaving, the compensator pair changing
face) encodes curvature of Lagrangians that no longer apply; the QP then takes
steps a fraction of the size the current structure allows. The frequency set is
NOT the cause: pruning Omega to the current local maxima (401 -> 7) stalled the
same replay (6.8635 after 25 iterations) -- on a flat optimum the rows between
the ripples are what let a step lower the whole plateau. `--sqp-carry-h off`
(default): each round starts from H = I; the trust radius is still carried.
(pyGRANSO/hinfstruct-type methods keep no long curvature memory: their bundle
or BFGS is rebuilt each call, which is the robust part; the fix keeps the SQP's
second-order model within a round, where it pays.)

**Batched rows** (reduced.gain_batch): all pending frequencies in one subspace
iteration -- per-frequency QZ triangular solves by LAPACK ztrtrs, the rank-2
controller coupling as T~ = R - g R b c^H R with its exact adjoint, batched SVD /
QR on the (frequencies x k x 5) blocks, singular pairs recovered in gain()'s
conventions. Agreement with the per-row path at the flat ne=200 design (401
rows): sigma 4e-13, singular vectors 2e-11, gradients 8e-15. **Batched peak
following** (peakfollow.follow_batch): the same safeguarded Newton, every
member's evaluation of an iteration in one call; tops agree to 4e-12, rows to
2e-14, gradients to 1e-11. One SQP evaluation at that design: 1.9 s -> **0.80 s**
(single process). Remaining: the sigma_2 blocks and the near-line eigenpairs per
row, Python per-row gradient assembly.

**ne=200 K+B+C with both fixes (single process, no other job):** frozen 7.964 ->
6.485693 after 44 rounds (4,623 s; stopped by hand in the tail). Within 1 % of
that after 30 rounds (2,626 s = 44 min). No plateau: the progress rule "gain
over 3 rounds < 0.3 %" (`--stop-progress 3:0.003`, new) stops at round 34
(3,346 s = 56 min), 0.15 % above the value 10 rounds later; on the old runs the
same rule stopped 20-24 % high, on plateaus that were stale curvature. A
different local minimum from the 6.438335 of 26 (+0.7 %).
**Stationarity measures on a flat optimum are not decisive.** The KKT residual
with the tool's active set (peaks within 2e-3 of the max) was ~5e-5 x gradient
scale from round 24 (6.74) on: with 87-193 near-active gradients the hull
contains ~0 long before the minimum. With 1e-4: round 24 -> 2e-3 (not
stationary), round 28 -> 9.6e-5 at 6.572 (then -1.3 %), round 37 -> 4.8e-5;
with 1e-5 too few peaks are equalised to count. Report the progress stop and
the residual at 1e-4 together; neither alone.
**Measured per-round costs (ne=200, 35 rounds to the stop):** 79 % of the wall
time in the SQP (83 evaluations per round, 0.91 s each at |Omega| 163 -> 500,
reduced dimension ~120); outer work 20 s per round; 172 full factorisations per
round (sweep + witness 70, level-set certificate 22 on the 2n pencil, abscissa
certificate 27, stability gate 21, anchors and eigen-anchors 18, compaction 7),
one winding count (521 factorisations) in the run; 17,600 triangular solves per
round with those factors (ARPACK shift-invert, anchors' input bases).

**The pair's face and the line search.** The first ne=100 run with both fixes
was stopped by the progress rule at 7.9285 (round 11): on the pair's face the
radius is shrunk to the accepted step, and the warm line search (alpha from 2 x
the last) never tried alpha = 1 again, so radius and alpha ratcheted down
together (radius 2.5e-7). On that face the search now starts at alpha = 1 (the
radius is the memory there); replay of round 11: 7.9285 -> 7.8677 in 40
iterations. The progress stop no longer fires on a stalled solve (code 6/12):
the second attempt still crept at 7.93 for rounds 8-18 (the pair nearly
defective on the line, S stationary in small steps) and broke out afterwards. **[2026-09-26: the explanation in brackets was wrong -- the causes were the
governor's cap after successful homotopy splits and the pair let onto S only
at a solve's start with separation <= 0.05; section 29.1-29.2.]**
**ne=100 K+B+C, final code, single process, no progress stop:** 10.99 -> 6.4164
after 53 rounds (57 min, 4,586 evaluations at 0.60 s); within 5 / 2 / 1 % of the
best known ne=100 value (6.400015) after 31 / 40 / 45 min. **[Stopped by hand at 6.410499. Two more causes of the creep at 7.93 found and fixed in 29: the governor cap after homotopy splits, the late face entry; v6 (29.3) reaches 6.404234 in 23 min.]**

**Cavity extrapolation (Dergham, Sipp & Robinet 2013 backward-facing step, 170,260
dofs P2-P2-P1, one 96-core / 512 GB node).** Measured per round (ne=200): 150
plant-shifted factorisations + 22 on the 2n Hamiltonian-type pencil, 17,600
triangular solves with those factors, 83 reduced evaluations. Assumed: 24
factorisations concurrently (4 threads each; ~3-8 GB each); 8 sequential stages
per round (level-set levels, gate, certification order); reduced cost scaling
~ k^2 with the reduced dimension (GL ne=200: k ~ 120, 0.9 s per evaluation).
| scenario | LU (4 thr) | solve | red. eval | rounds | per round | total | rows on 8 workers |
|---|---|---|---|---|---|---|---|
| low  | 2 s  | 0.05 s | 1.9 s (k~170) | 35  | 1.2 + 2.6 min  | 2.2 h  | 0.9 h |
| mid  | 5 s  | 0.15 s | 3.9 s (k~250) | 60  | 3.2 + 5.4 min  | 8.6 h  | 3.9 h |
| high | 15 s | 0.4 s  | 9.7 s (k~400) | 120 | 9.1 + 13.4 min | 45 h   | 21.6 h |
The three unknowns decide it: the factorisation/solve times on that mesh
(tools/lu_footprint.py, tools/lu_concurrency.sh), the reduced dimension the
cavity needs, the number of rounds. Prerequisite not counted: the loop runs on
GL's own FE assembly and SciPy's SuperLU; the cavity needs the operators (A, M,
b(x), c(x)) from the user's solver and MUMPS through PETSc.

## 29. The creep at 7.93; the FB/FC ripple; dead code (2026-09-25/26, user: fix the creep, fields, open items)

**29.1 First cause: the tau governor capped steps after successful homotopy
splits.** On the run of 28 (v4) rounds 10-16 ended with code 6 on step caps
0.0021 -> 0.034: every round whose continuation chain needed a homotopy split
(and then PASSED the gate) had its next step capped as if the gate had failed.
`--tau-split ignore` (default; SQP arm only): no cap after successful splits,
and the cap relaxed x2 after every clean round. Run v5 (this fix only):
6.410212 after 31 rounds, 1,651 s, progress stop (v4: 6.4164 after 57 min).
But v5 still crept 7.938 -> 7.899 over rounds 12-15 (170 s), until round 16.

**29.2 Second cause (replay of round 14, dump outer_013, with per-trial row
diagnostics).** `DEBUG_LS` now prints, per line-search trial, the row attaining
the trial's maximum, its value against its linear model, and the retraction's
share of the step; on a rejected search it re-evaluates the current iterate and
lists the rows that changed.
- **The compensator pair was let onto its face S too late.** Rounds 12-15:
  two compensator poles at -0.0534-0.412i / -0.0586-0.334i (0.078 apart),
  then 0.054, 0.052, 0.047 apart, the rightmost within 4e-4 of the line. The
  entry test (separation <= 0.05) ran only at a solve's start, so the pair
  entered at the start of round 16 -- and round 16 went 7.909 -> 7.293. Off
  the face every search ended at alpha 1/128: the trial errors were quadratic
  in alpha, on the narrow peaks at w = -1.39 / -1.37 / -0.93 (curvature ~4e4
  along the step at -0.93, a closed-loop pole 0.003 inside the margin),
  which the identity-started quasi-Newton matrix does not carry.
  Remedy: **entry separation 0.05 -> 0.1, and the entry test after every
  accepted step** (no re-entry after a release in the same solve). Replay of
  round 14: 7.9087 -> 7.36 (per-member restarts) / 7.58 (batched restarts) in
  40 iterations, against 7.9053 -- replay paths are chaotic, the direction is
  not.
- **Followed members collected on one peak and under-stated smax.** Members
  based at -0.5280, -0.5105 and -0.5094 all sat on the top at -0.524950
  (7.9057032) while the neighbouring top at -0.509496 stood at 7.9063061. An
  iterate accepted on that value makes every later trial worse than an
  impossible reference (with a shrinking radius: 1e-10, code 6). Remedy
  (`FOLLOW_DEDUP`): **members sharing a top (within 1e-4) all restart cold from
  their own base, except the one whose base is nearest** (batched Newton); a
  member whose restart returns to the same top is marked merged and left
  there. Cost in the replay: +8 % wall time.
- **Tried and removed** (no code left): (i) sigma_2 blocks formed inside the
  solve at every near-active row with a gap < 5 % -- zero blocks formed in the
  faithful replay (the per-round enrolment already covered the band
  w in [-0.53, -0.50]), at the price of a k = 2 evaluation per near-active row
  per evaluation; (ii) the trust radius shrunk to the accepted step off the
  face, as on it: 7.9056 vs 7.9053 plain, and with stale tops it collapsed to
  1e-10 -- the step length is set by the narrow peaks' curvature, which only
  the quasi-Newton matrix can carry.

**29.3 ne=100 K+B+C on the new code (v6; single process, nothing else
running):** 9.805 -> **6.404234 after 22 rounds, 1,375 s (23 min)**, progress
stop; the best known ne=100 value is 6.400015 (+0.066 %). Round 9 7.186 (v5
8.378), round 16 6.501 (v5 7.293). Within 5 / 2 / 1 % of the best known value
after 9 / 13 / 14.5 min (v4: 31 / 40 / 45 min). The pair entered S in every
round but one (mid-solve in rounds 0, 6 and 7); dedupe restarted 79-763
members per round (summed over the round's evaluations). Invariants: frozen path 10.23128516441147, witness invariant 0
violations, Hurwitz selftest 0/200. Figure: figs/gamma_vs_time_n100_creep.png
(tools/plot_creep.py).

**29.4 The FB/FC fields (tools/field_truncation.py; logs
figs/field_truncation_fbfc_*.log).** **[ERRATA 2026-09-28, 32.1: the mode-band
statements and the 'lowest 60 / 20 modes' column below used a scrambled basis
(fieldbasis._m_orth); corrected numbers in 32.1. The smooth designs live in the
lowest ~50 modes; the optimised ones carry ~9 % at the basis cutoff. 32.2-32.3:
the ripple is band extension tied to the cutoff, not mesh-scale noise.]** Post-hoc tests with everything else fixed
(controller, gains), dense level-set certificate:

| design | gamma | tail fraction (abs(x) > 12) | smooth tails (s = 0.5) | window abs(x) < 10 | lowest 60 / 20 modes |
|---|---|---|---|---|---|
| ne=100 start (frozen ne=200 transferred, 80 modes) | 3.935632 | 0.111 | +0.06 % | +4.9 % | +88 % / +1738 % |
| ne=100 after 17 rounds, 80 free modes | 3.901205 | 0.364 | **+34 %** | **+36 %** | +110 % / +2466 % |
| ne=200 frozen (nodal, Sobolev map) | 3.925457 | 0.111 | +0.13 % | +7.4 % | +58 % / +573 % |

- The 0.9 % the 80-mode optimisation gained (3.9356 -> 3.9012) was bought by
  growing the tails from 11 % to 36 % of the field's norm, with grid-scale
  ripple (80 modes on 78 length units resolve a wavelength of ~2, the ne=100
  mesh ~1.6) that does real work: smoothing only the tails costs +34 %. The
  smooth frozen design is 0.6 % worse. This is the signature of an
  unregularised field problem exploiting the discretisation: the gain will not
  survive mesh refinement and is not physical.
- It is NOT "too many modes" in the sense that fewer would do: the energy is
  spread evenly over the 80 modes even for the smooth start (23/20/34/23 %),
  because a field localised over ~2 units needs the high modes of a GLOBAL
  basis, and truncating the modes of a fixed design is ruinous for the smooth
  designs too (+58 % .. +1738 %). What is missing is a statement of which
  fields are admissible.
- Normalisation does not regularise: the fields enter through their M-norm and
  the gauge K absorbs the scale (gamma is scale-invariant), so an H^1
  normalisation changes nothing. The regularisation must be a SHAPE condition,
  scale-invariant, e.g. two smooth rows in the SQP:
  mean wavenumber (v* K v)/(v* M v) <= k_max^2 (in modal coordinates
  sum lambda_k abs(beta_k)^2 <= k_max^2 sum abs(beta_k)^2: diagonal, cheap),
  and localisation (v* X^2 v)/(v* M v) <= L^2 about the field's centre.
  Both are physical (smallest feature size, footprint), mesh-independent.
- Efficiency: nodal parametrisation + Sobolev map (the frozen code) costs
  ~2 x (2 ne + 1) complex parameters per field (ne=200: ~1,600 real) in a
  dense quasi-Newton SQP whose QP and matrix grow with the square of the
  count; a modal basis of 80 costs 320, with the Sobolev weight diagonal. At
  the cavity (P2 fields: 10^4-10^5 nodes each) the nodal route is not an
  option at all. Projected (reduced) fields with scale-invariant rows are the
  efficient choice.
- Atoms (wavelets / Gaussians of unknown centre, width, amplitude, phase):
  natural generalisation of K+B+C (one fixed-width Gaussian per field) toward
  K+FB+FC; a handful of atoms represent localised fields with few parameters,
  smooth by construction, mesh-independent if the width is bounded below
  (a few cells). Costs: non-convex in centres and widths (atoms merge, widths
  collapse without a floor), the number of atoms is a discrete choice (add
  greedily, keep while gamma drops), overlapping atoms make the Jacobian
  ill-conditioned. Extended fields need wide atoms or many; a fixed dyadic
  wavelet dictionary (linear coefficients, multiscale, sparsity through
  epigraph rows) avoids the non-convexity at the price of more coefficients.
  For the 2-D cavity: atoms with (x0, y0, sx, sy, orientation) per atom, tens
  of parameters, is the scale the SQP handles well. Not yet tested here;
  proposed order: (1) 80 modes + the two shape rows at ne=100 and ne=200
  (mesh independence of gamma and fields); (2) K + n Gaussian atoms,
  n = 1, 2, 4, against (1).

**29.5 Dead code (user: careful with dead code after negative tests).** Removed
from sqp_inner / hurwitz / run_case_epi on 2026-09-25 (all negative or
superseded, all off by default): MERIT_CTRL, HMIN_REL_TOL, NU_H, TR_BALL,
CTRL_CUSHION, CTRL_HESS, SOC_CTRL, SOC_MAXIT, TR_SHRINK, H_SCALE, MERIT and
`--sqp-merit`, `_smat`, `comp_hermite_at`, the exact-hmin helpers, the
compensator-Hermite branches of the merit/admissibility, the SOC/curvature
blocks, `hurwitz.hermite_block_hessian`, their stats keys. Check: a replay of
round 11 gave the identical iterate history before and after (md5
3e5b9dfcc3286a50268b0fcabeb68ece), frozen path 10.23128516441147, selftest
0/200; tools/check_invocations.py: no script passes a removed flag. The two
tests of 29.2 were removed after testing. What remains flag-selectable is in
use or is the frozen path.

**29.6 Stale conclusions.** ERRATA blocks (2026-09-25) at the top of STAGE1,
STAGE1B, STAGE2, STAGE3 and STAGE5, inline annotations in STAGE5, ~50 TRACKER
rows annotated: every K+B+C value between 7.6 and 7.96 recorded as converged
or as a KKT point was a stall (pair, stale curvature, the governor cap of
29.1); the KKT residual with a 2e-3 active set is not an optimality
certificate on flat optima; "cosmetic ripple" (28) was wrong (29.4).

**29.7 ne=200 K+B+C on the new code, and two holes in the progress stop.**
The first two attempts stopped at 7.949041 after 4 and 5 rounds: from the
frozen design (the compensator pair 3e-4 from a double pole, entered on S at
once) rounds 0-3 crept 7.960 -> 7.949 with a face radius ~1e-5, and the rule
"< 0.3 % over 3 rounds" fired (a) at the end of round 3, whose solve had just
taken 7.9553 -> 7.9493 in its last iterations, and on the retry (b) at the end
of round 4, whose candidate 7.4964 had been REFUSED by the gate (the best stayed
7.9490). The rule now also requires (a) that the last solve was not
accelerating when its budget ran out (half or more of its decrease in its last
quarter of iterations; the tails of the runs that went on: 0-0.48; round 3
here: 0.78), and (b) that the round's candidate was not better than the best by
more than tol and refused. Neither changes the stops of v6 (ne=100) or of the
earlier ne=200 run.
**Result (v4, single process):** 7.960 -> **6.486694 after 24 rounds, 2,493 s
(42 min)**, progress stop; within 1 % of the best known ne=200 value
(6.438335) after 33.7 min (v3: 44 min); v3 reached 6.4955 at its stop (56 min)
and 6.485693 after 77 min -- the same local minimum, now in 42 min. Invariants
PASS after the change. Figure: figs/gamma_vs_time_n200.png.
For the cavity extrapolation of 28: the low scenario assumed 35 rounds; the
new code stops after 22 (ne=100) and 24 (ne=200) rounds, so the low / mid
figures drop by roughly a third, other things equal.

## 30. Loose ends before the cluster (2026-09-26, user: close what does not need the cluster; never regress)

**30.1 Triage.** Every open row of TRACKER.md was classified: implemented here
(30.2-30.8), closed by decision without code (30.9, with the reason), or left
for the next phases (cluster regression and studies, parallelisation, cavity,
forcing parametrisation -- the last at the user's request).

**30.2 Where the factorisations go (tools/lu_breakdown.py, new).** It wraps
splu before the driver's counter, attributes every full-size factorisation to
the driver frame (and certificate routine) that asked for it, and splits the
counts by round. ne=100 K+B+C, current loop, 12 rounds, per round:
reference sweep + certificate evaluations 76 (61 of them the fixed 61-point
full sweep), abscissa certificates 53 (SQP validation 15, continuation
midpoints 20, candidate 18), stability gate 22, level-set certificate pencils
19, eigen-anchors 8, anchors 4, start-value climb 3 -- ~160 in all -- plus,
when a continuation is refused, one winding count (400-960 points = LUs), and
at the end of the run 2,481 factorisations for the stored plot curve and 675
for the save-time winding certificate.

**30.3 Cuts (all measured, frozen path bit-identical).**
- `--cert-sweep 0`: the full model is sampled only at the reduced model's
  peaks (enrolled peaks, the fine-sweep argmax, the 4 highest local maxima;
  ~10-17 points) instead of a 61-point grid. The certificate is the proof; the
  sample checks the model where the norm lives and drives the disagreement
  anchoring. 76 -> 17 a round.
- The stability gate re-swept the whole line Re = -eps over the band right
  after the (memoised) abscissa certificate of the same design had swept its
  own region of it: it now sweeps only the part of [wlo, whi] outside that
  region (`stability_gate(skip=...)`, reduced-oracle loop). 22 -> 2 a round.
- `stability_region` recomputed the plant's extreme eigenvalues (ARPACK on the
  generalised problem, M factorised) at EVERY abscissa certificate: cached per
  plant (identical values).
- The stored gain curve at the end of a run is evaluated on the reduced model
  when one is attached (`gain_model` in the record): -2,481 factorisations per
  run at ne=100 (hours at the cavity).
- Defect found on the way: after a basis compaction the model at the BEST
  design was compared with the last sweep, which belonged to the last
  CERTIFIED CANDIDATE -- a refused candidate is a different design (ne=200
  rounds 4 and 6: "disagreements" of 70 %, three spurious anchors each). The
  sweep now carries its design and is used only for that design.
Result (ne=100, complete run): **104 full factorisations per round without a
winding event, against 192 before (-46 %)**; ne=200 below.

**30.4 The winding count was inconclusive whenever the compensator pair sat on
its face S.** A refused continuation is accepted if the argument-principle
count right of Re = -eps is 0. That count failed (INCONCLUSIVE, candidate
refused, governor cap) in half the attempts (v6 1 of 2, v5 1 of 4, first
new-code run 3 of 3). Cause: the compensator's poles are poles of
f = 1 - k G, and on S two of them sit 1e-6 left of the counting line; f swings
through ~2 pi twice over a width ~1e-6 and the tracker reached its step floor.
The refused candidates checked were stable (dense abscissa -0.089 and -0.080;
count 0 right of the margin). Remedy (`--wind-ctrl exact`, winding_cert `ctrl_exact`): track
g = det(sI - A_c) f = det(sI - A_c) - [(K - 1) det(sI - A_c) + det(sI - A_c + B_c n^T)] G
(matrix determinant lemma; no poles at the compensator's eigenvalues) and add
the phase of det(sI - A_c) along the line in closed form from its r roots.
Validated against the dense count on 3 designs x 9 lines (counts 0 to 6, lines
through, 5e-7 and 1.5e-6 from the compensator poles): every exact count
correct and conclusive; the plain count inconclusive on every line within
1e-6 of the pair. Two tail constants made rigorous on the way: ||M^{-1} A||
computed exactly at GL sizes (was a power-iteration estimate x 1.3), and the
compensator's resolvent bounded by 1/(|s| - ||A_c||) (was 1/(distance to its
poles), which holds only for a normal A_c; ||A_c|| = 66 against pole moduli
<= 17).

**30.5 Partial ARPACK sets in the sparse level-set certificate** (TRACKER row
511). `--cert-disk strict`: when ARPACK returns a partial set at a shift, the
nearest eigenvalues are asked for again (k = 4 -- the even pencil's
eigenvalues come in pairs mirrored across the imaginary axis, at equal
distance from every shift on it, which k = 1 cannot separate -- larger
subspace, budget 300 restarts, on the factors already held: the pencil now
keeps its factors per shift); if that fails, the interval is bisected (16 per
decision); if the budget is spent, the decision is INCONCLUSIVE (never
accepted by the loop). Counters in every result json (`cert_disks`: untrusted
/ retried / oversized / bisected / uncovered / climb_failed). Frozen case B: 4
partial sets, all resolved by the retry, none of which would have given a
too-large disk; gamma 10.23128516441147 in both modes. Loop runs: 0-2 events
per run, all resolved by the retry.
**A real hole in the frozen behaviour, found by the tail certificate
(30.6):** at ne=200 and w = 278 (= nu, far outside the loop's band) ARPACK
converges NO eigenvalue for any k (1-16), subspace (30-60) or budget (up to
3,000 restarts). The frozen code then saw an EMPTY set and dropped the whole
interval unproven -- and reported "certified". Strict mode reports it
inconclusive (86 s, 16 bisections); the dense level-set algorithm certifies
the same band in 1 s. Inside the loop's band [-30, 30] this has not
happened; at the cavity, where only the sparse procedure exists, it is a
certification item: the cure is a residual-based disk (the smallest singular
value of the shifted pencil, on the factors held), not more ARPACK.

**30.6 The certified value is now global in frequency** (STAGE1 8, "analytic
tail bound"). tools/tail_certificate.py: with nu = ||M^{-1/2} A M^{-1/2}||
(exact, dense), beta = ||M^{-1/2} b||, gamma_c = ||M^{-1/2} c|| and
||A_c||, the rank-one closed loop gives for |w| >= W
sigma_1 <= 1/(W - nu) + k_max beta gamma_c / ((W - nu)^2 (1 - k_max G_max)),
with k_max = |K| + ||n|| ||B_c|| / (W - ||A_c||) (no normality assumed); the
band [-W, W] is certified by the dense level-set algorithm (all eigenvalues of
the even pencil). Every record of this session: GLOBAL.

| record | nu | ||A_c|| | W | tail bound | band [-W, W] |
|---|---|---|---|---|---|
| ne=100 K+B+C 6.389368 | 75.8 | 66 | 303 | 0.018 | certified, 0.2 s |
| ne=100 K+B+C 6.412162 / 6.403690 | 75.8 | 73 / 75 | 303 | 0.016 | certified |
| ne=200 K+B+C 6.484857 | 278 | 74 | 556 | 0.005 | certified, 0.9 s |
| ne=100 K+FB+FC 3.898938 | 75.8 | 4,200 | 8,399 | 1.3e-4 | certified, 0.2 s |

GL only (dense norm; nu grows like 1/h^2); at the cavity the tail must come
from the operator's structure -- a cavity-readiness item.

**30.7 Detectors and reporting.** A closed-loop pair at the line (two
eigenvalues within 1e-3 of Re = -eps and 1e-2 of each other: the closed-loop
analogue of the compensator's double pole, without a face-S treatment) is
counted (`closed_loop_pair_rounds`) and logged; 0 on every run so far. The
run scripts compute the constraint-aware KKT residual at active-rtol 1e-4 on
the saved record (`kkt_1e-4.json`), reported with the progress stop.

**30.8 Regression against the results of 29 (single process, nothing else
running; figs/gamma_vs_time_loose_ends.png).** The loop is chaotic at the
1e-7 level: the new reference sample changed the certified value of round 6
by 1e-7 (within the certificate's tolerance), and the paths separate from
there -- one run per configuration is a sample, not a measurement, so the
comparison is made on whole curves at equal wall time, and each new run was
continued with a loose stop (3:0.001) to see its tail (paths are
deterministic, so a tighter stop is a prefix of the same run).

| | previous (29) | current (30), at the previous run's wall time | current, 3:0.0015 stop | current, 3:0.001 stop |
|---|---|---|---|---|
| ne=100 | 6.404234 at 22.9 min (v6) | 6.401 at 21.5 min, 6.3991 at 23.3 min | **6.399134 at 23.3 min** | 6.389368 at 32.0 min (best known ne=100) |
| ne=200 | 6.486694 at 41.6 min (v4) | 6.4849 at 36.4 min (finished) | **6.486505 at 32.2 min** | 6.484857 at 36.4 min |

At every wall time after the first minutes the current code is at or below
the previous one on both meshes. With the old stop (3:0.003) the current code
stops EARLIER and slightly higher (ne=100 6.412162 at 16.0 min, ne=200
6.488448 at 29.8 min: it takes larger steps earlier, so the 3-round gain
falls under 0.3 % sooner); the scripts' stop is therefore 3:0.0015, the
loosest value at which both reference runs dominate the previous results in
value, at equal or lower time. Both ne=200 runs end in the same local minimum
(6.485, the other known one is 6.438335).

**The progress stop needed arming.** From the frozen ne=200 design the
first rounds crawl on the pair's face (radius ~1e-5): 7.9603 -> 7.9567 over
three rounds stopped a run at round 3 (0.045 % < 0.1 %), with a KKT residual
of 2.4e-4 x the gradient scale (a stationary-looking start: the KKT test would
not have prevented it). The rule is now armed only once the run has gained
1 % over its start or after 20 rounds (`--stop-progress k:tol[:drop[:rounds]]`,
defaults 0.01 and 20); none of the stops above changes.

**30.9 Reduced-model certification every 4th round, on the current code**
(TRACKER row 492: "recommended as an option to validate"). ne=100 K+B+C,
same flags + `--cert-every 4 --prov-check 0 --cert-fail continue --prov-gate
reduced`: 6 of 6 provisional chains confirmed by the full certificate (claims
above the truth by ~1e-7), no rollback, no escalation, witness invariant 0;
6.403690 after 25 rounds (1,617 s); **2,674 full factorisations in the run
against 5,415** for the every-round run over 26 rounds (69 against 100 per
round without a winding event; 1 winding event against 3). At GL the wall
time is LONGER (a full factorisation costs ~1 ms here, the provisional
round's reduced certificate does not), and the per-round progress slightly
slower (6.4037 at round 24 against 6.3905). Verdict: holds, halves the
factorisations; the choice is the cavity's (factorisation-dominated) -- an
option to switch on there, not a GL default.

**30.10 Open rows closed by decision, without code.**

| row | decision | reason |
|---|---|---|
| basis-dimension cap "exercise it" | closed | exercised in every run since 23: compaction 122 -> 114 at the cap 120, dims 94-142 |
| global optimisation of positions on the reduced model (note item 12) | not done | the two ne=200 local minima (6.438335, 6.485) have nearly the same positions (+-2.335 / +-2.312, against a Gaussian width of 0.4); they differ in the compensator -- the lever is multistart over controllers (cluster, Stage 2) |
| random-start study on the reduced model (item 12), Stage-2 96 starts, HIFOO arm rerun | cluster | parametric studies are cluster jobs (decision 2026-09-20) |
| Newton polish of the balance system (item 9) | dropped | tried (STAGE2 3.4: unsafeguarded at a corner); the stalls it was meant for were the compensator pair and stale memory (29); the SQP with face S reaches KKT residuals 4e-5 x scale |
| bundle / gradient-sampling Plan B (item 8) | dropped | same: not a solver-class limit |
| IPOPT arm (item 8) | dropped | a third inner solver on a loop whose inner solve is no longer the bottleneck; needs conda |
| controlled SQP vs pyGRANSO comparison; pyGRANSO with the (tr, det) block | dropped | pyGRANSO cannot represent the compensator double pole (K.5); it stays only for the frozen path |
| A-N satellites and retirement (items 4-5), level-ladder reading | OFF, final | with the reduced oracle |Omega| costs no factorisation; peak following + in-solve refinement do what satellites were for; the cap was measured risky (L) |
| one-sided lower bound ||T V|| | dropped | the level-set certificate is two-sided and exact; a lower bound is any sigma_1 evaluation |
| LU-weighted accounting (Stage 3) | superseded | measured per component (tools/lu_breakdown.py, 30.2) |
| adaptive inner budget at ne=200 | done | the nominal scripts run --inner-maxit-adapt at ne=200 |
| adaptive input enrichment for free fields | left | forcing parametrisation untouched (user, 2026-09-26) |
| `--rigor` winding segment bound | stage 6 | it is pass 2 of the parallel winding scheme (CLEANUP 3) |
| winding cost (400-960 factorisations per refused continuation) | stage 6 | the samples are independent between knots: the parallel winding count |
| 27-case regression, default flip | cluster | unchanged |
| cavity port, tail bound at the cavity, P_plant recipe | cavity | next phase |

**30.11 CLEANUP section 5 closed.** Completed reference table
`results/reference_table_ne200.csv` (winding counts of the 27 frozen records at
the frozen line -eps/2: all 0, consistent with the dense abscissae) and
`results/reference_table_ne200_margin.csv` (at the margin -eps: all 0 with the
exact-compensator count; the plain count is inconclusive on 4 of them). The
bounds are written by `tools/bounds.py --json` and read by
`tools/plot_results.py`, with the positions they assume (1401-point grid:
sensor-limited 9.831865, actuator-limited 9.660137, floor 3.509193; the typed
literals were from coarser grids). FC's 9.660135 sits on the actuator bound
within the frozen certificate's tolerance (1e-4).

**30.12 K+FB+FC on the current code** (fields parametrised as before, 80
Laplacian modes). From the same start (3.935632), tools/n100_fb80.sh: 3.898938
after 20 rounds (its --outer), 69 min; the previous 3.901205 took ~36 rounds
over four launches of older code. Certified over all frequencies (30.6).
||A_c|| = 4,200: the field case's compensator carries a fast mode.

**30.13 Confirmation with the final scripts** (every change of this section
in, single process): tools/n100_kbc.sh **6.399134 in 1,356 s (22.6 min)**, 21
rounds, KKT residual 2.5e-5 x gradient scale (40 active peaks); tools/n200_kbc.sh
**6.486505 in 1,884 s (31.4 min)**, 18 rounds, KKT 5.1e-5 (53 active) -- the
same paths as the development runs of 30.8 (deterministic), against 6.404234
in 22.9 min and 6.486694 in 41.6 min before. Full factorisations: 3,803 in
the ne=200 run (6,698 before), 110 per round without a winding event (179).
Partial ARPACK sets 0 / 2 (both resolved by the retry), closed-loop pairs at
the line 0, witness invariant 0; both records certified over all
frequencies. The 27 frozen records re-certified the same way (dense level set
+ tail bound): every recorded value within 6e-7 of the global supremum
(`results/reference_table_ne200_global.csv`) -- the frozen certificate's
empty-set hole (30.5) did not affect them. Invariants: frozen path
10.23128516441147, witness invariant, Hurwitz selftest 0/200 -- PASS.

## 31. STATUS.md; the final curves and fields (2026-09-27, user: a short status file, the full item list, sigma_1 without control vs sigma_1, sigma_2 with, the final fields)

**STATUS.md** (new, kept short): current state of the handoff's stages 0-7, of
every item of the two notes' to-do lists (optimisation note section 9, algebra
note section 12) and of the items added along the way, with the reason for each
dropped item and the phase each open one belongs to. The stages are those of
HANDOFF_implementation.md section 5 (mirrored in TRACKER section A). The history
stays here and in TRACKER.md.

**tools/final_curves.py** (new): full-model sigma_1 of the plant alone (no
control) and sigma_1, sigma_2 of the closed loop, on a grid of ~2,000
frequencies over [-30, 30], for the nominal scripts' final designs; grid maxima
6.399134 / 6.486505 / 3.898881 against the certified 6.399134 / 6.486505 /
3.898938. Uncontrolled peak ~880 (ne=100) / 858 (ne=200) near w = -0.6. The
controlled sigma_1 is FLAT over wide bands -- K+B+C over w in [-7, 4] plus a
second plateau-peak at w ~ 12 (the compensator's resonance), K+FB+FC over
[-15, 19] -- the signature of an equalised (many-peak) optimum; sigma_2 of the
closed loop reaches 6.13 / 6.21 / 3.87 near w = -0.5, where it is pinned by
the uncontrolled sigma_2 (interlacing: sigma_1^cl >= sigma_2^ol pointwise).
figs/final_sigma.png, figs/final_fields.png (K+B+C: the two Gaussians at
+-2.313; K+FB+FC: the 80-mode fields with their mesh-scale ripple).

## 32. FB/FC fields: refinement, why viscosity does not regularise, Sobolev shape rows, atoms (2026-09-28, user: do the oscillations go away under refinement or move to smaller scales; bring Sobolev back if needed; results must be mesh-independent; compare with atoms)

**32.1 A defect in the analysis of 29.4 (not in the runs).** `fieldbasis._m_orth`
M-orthonormalised the Laplacian eigenvectors through an SVD, which returns an
orthonormal basis of the same SPAN in a scrambled order. The optimisation is
unaffected (same span; the quasi-Newton start is rotation-invariant), but every
post-hoc statement of 29.4 that took "the lowest k modes" as the first k
columns was wrong: the "energy spread evenly over the 80 modes (23/20/34/23 %)"
and the truncation costs "+58 % .. +1738 %" for the smooth designs. Fixed:
`_m_orth` is order-preserving (classical Gram-Schmidt twice, M q precomputed),
`laplace_basis` returns columns in eigenvalue order (and clamps N to the
interior nodes). Corrected analysis (tools/field_truncation.py, now with
eigenvalue bands and the RMS wavenumber; figs/field_truncation_v2_*.log):

| design | gamma | energy in modes 0-20 / 20-40 / 40-60 / 60-80 | RMS wavenumber | lowest 60 modes | lowest 40 modes |
|---|---|---|---|---|---|
| smooth start (frozen ne=200 -> ne=100) | 3.935632 | 0.921 / 0.068 / 0.011 / 0.0008 | 0.53 | +0.007 % | +12.5 % |
| frozen ne=200 (nodal) | 3.925457 | 0.922 / 0.066 / 0.012 / 0.0010 | 0.53 | +0.14 % | +17.0 % |
| 80 modes, 17 rounds (old run) | 3.901205 | 0.825 / 0.066 / 0.007 / 0.102 | 1.15 | +31.0 % | +39.5 % |
| 80 modes, nominal (3.898938) | 3.898938 | 0.844 / 0.067 / 0.004 / 0.086 | 1.10 | +27.8 % | +31.4 % |

The corrected picture is sharper than 29.4's: the smooth designs live in the
lowest ~50 modes (dropping modes 60-80 costs nothing); the optimised designs
put 8.6-10 % of their energy in the TOP 20 modes (a spike at the basis cutoff,
k ~ 3.1-3.5, with nothing in modes 40-60), and that 9 % carries a +28 %
penalty if removed: it does real work. What 29.4 said about normalisation and
shape conditions stands; its mode-band sentences are corrected here (ERRATA
note added in 29.4).

**32.2 Why the viscosity of the linearised equation does not do the smoothing.**
The plant is diffusive (gamma d_xx, gamma = 1 - i): the response to forcing
of wavenumber k is damped like 1/(abs(gamma) k^2). Measured (tools/mode_response.py,
ne=200, figs/mode_response.png): the energy of the response to a unit Laplacian
mode used as the actuator field is 197-262 for k < 0.5, 0.17 at k = 2, 0.080 at
3, 0.045 at 4, 0.019 at 6.2, 0.0116 at 8, 0.0052 at 12 -- k^2 x energy = 0.69 ..
0.74 = 1/abs(gamma) = 0.71 from k = 2 on, at w = -0.646, 0 and 0.75 alike; what
the smooth sensor sees falls faster (5e-4 at k = 3, 5e-6 at k = 6). So the
viscosity is doing its job on the RESPONSE: the map field -> response is
compact. It does nothing to the DESIGN:
- rough content is made INEFFECTIVE, not COSTLY: nothing in gamma penalises it,
  and the gauge (the compensator gain absorbs the field's scale, gamma depends on
  the field's direction only) removes even the norm budget as a cost. A
  compact forward map is precisely the situation in which the inverse
  (design) problem is ill-posed -- the directions it damps are near-null
  directions of the objective, along which the optimiser moves slowly and the
  minimiser, if any, is not determined by the data;
- and here rough content is not useless either. The H-infinity optimum is
  equalised: sigma_1 is flat at gamma over a band, and the wider the band the
  lower the plateau (the gain removed at the resonance, 880 -> 3.9, is spread
  over it). A mode e^{ikx} of the GL resonates at w = k^2 - 2k (k < 0: the
  upstream branch reaches w = 19 at abs(k) = 3.48), so the band the loop can
  flatten is set by the wavenumbers the fields carry. Measured
  (tools/band_study.py, figs/band_R1_partial.png): the smooth start is flat
  (>= 0.95 gamma) on w in [-7.0, +4.8] with a shoulder to +9; the 80-mode optimum
  on [-15.3, +19.1] -- and w = 19.1 is the resonance of abs(k) = 3.48, the
  80-mode basis cutoff at ne=100 (lambda_80 = 12.2). The ripple buys band
  extension; |K| grows with it (187 -> 321) and the closed loop's roll-off moves
  out (sigma_1 at w = 40: 0.135 -> 0.308). That mechanism is physical (the H^2
  designs of 32.5 show it at a converged k ~ 2.7 on every mesh); what the
  unconstrained designs do with it at their cutoff is not (32.7): there the
  coarse plant's response is wrong, and the controller is tuned to the error.
The report's "Sobolev" (fe_codesign sob_len, the frozen code) was a gradient
PRECONDITIONER P = (M + l^2 K)^{-1} M, i.e. a change of variables v = S xi: the
optimum in xi is the optimum in v; the map only steers the iterates towards
smooth ones first, which with a finite iteration budget acts as a hidden
early-stopping regulariser. That is why the frozen fields are smooth (RMS
wavenumber 0.53) and the modal runs, whose metric carries no such bias, are not.
The report (report_Claude_v2 3.3 and 7) states both halves correctly -- "the
regularisation changes only the search metric, the objective and the optimum
are untouched" -- but then concludes "the optimum is mesh-robust" and "the
grid-scale content was never buying performance". The second is true of
nodal-scale noise (k ~ 8-12 at ne=200: no measurable effect), false of the
k ~ 3-3.5 content above (+28 % if removed); the first does not follow from a
change of variable.

**32.3 Refinement: does the ripple go away or move to smaller scales?**
Run R1 (tools/fields_mesh.sh R1; results_fb_refine_n200_N160): no
constraint, ne=200 and 160 modes (cutoff k = 7.0), from the 80-mode ne=100
optimum transferred to ne=200 (4.024146 there, by LINEAR interpolation -- as a
P2 function it is 5.52 there, 32.7),
15 rounds, 145 min: 3.913146 after round 0, 3.896955 after 5, **3.888738**
after 15, still falling 0.03-0.05 % a round. Against the 80-mode optimum
(tools/field_study.py, tools/band_study.py; figs/fields_refine_*.png,
figs/band_refine.png; figs/fields_R1_partial_*, figs/band_R1_partial.* at
round 5):

| design | gamma | RMS wavenumber (act / sens) | energy at k > 3 | flat band (sigma_1 >= 0.95 gamma) | abs(K) | sigma_1 at w = +40 |
|---|---|---|---|---|---|---|
| smooth start, ne=100 | 3.935632 | 0.53 / 0.53 | 0.0005 | [-7.0, +4.8] | 187 | 0.118 |
| 80 modes, ne=100 | 3.898938 | 1.10 / 1.11 | 0.081 | [-15.3, +19.1] | 321 | 0.231 |
| 160 modes, ne=200, round 5 | 3.896955 | 1.00 / 0.99 | 0.067 | [-16.4, +19.3] | 379 | 0.271 |
| 160 modes, ne=200, round 15 | 3.888738 | 1.82 / 1.88 | 0.13 | [-21.0, +27.7] | 451 | 0.753 |

**The ripple does not go away; it moves to the new cutoff, as the user
suspected.** The spike at the 80-mode cutoff (k ~ 3.1-3.4) dissolved and the
energy spread over k = 3-7 with a spectrum RISING towards the new cutoff
(1e-2 per 0.25-bin at k = 5.5 and 6.5, zero only above 7.0); the flat band
widened from w = 19.1 to 27.7 (the resonance of abs(k) = 4.4), abs(K) grew
321 -> 451, the roll-off moved out. The gain is small and slow (0.26 % in 15
rounds, the optimiser crawling along the near-null directions of 32.2) but it
is not converging: gamma and the fields of the unconstrained problem are set
by the cutoff and the iteration budget. A FIXED mode count (a fixed physical
cutoff) is a regularisation in the limit, but an implicit one whose optimum
piles its energy at an arbitrary cutoff -- exactly where the plant
discretisation is least accurate, so that on a finer mesh the design falls
apart (32.7: +18 % to +42 %).

**32.4 The plant itself is not mesh-converged at ne=100.** Open-loop sup
sigma_1 (b, c irrelevant; sup over w, Brent-refined):

| ne | 100 | 200 | 400 | 800 |
|---|---|---|---|---|
| open-loop sup sigma_1 | 901.5654 | 882.8808 | 881.7226 | 881.6500 |
| relative to ne=800 | +2.26 % | +0.14 % | +0.008 % | -- |

So at ne=100 the PLANT carries a 2 % discretisation error -- more than the
whole 0.9 % the free fields gain over the smooth start -- and any statement
about mesh independence of gamma at the 1 % level must be made between ne=200
and ne=400 (0.13 % apart). The K+B+C difference ne=100 vs 200 (6.399 vs 6.487,
1.4 %) is consistent with this.

**32.5 The regularisation: a scale-invariant Sobolev shape row.**
The admissible fields must be stated; the statement must be SCALE-invariant
(the gauge), so a penalty or a normalisation cannot do it. Implemented: one
row per free field in the SQP (sqp_inner.Evaluator._field_rows, driver
`--field-kmax`, `--field-order`), c = R_p(v)/kmax^(2p) - 1 <= 0 with
  p = 1 (H^1): R_1 = (v^H K v)/(v^H M v), the squared RMS wavenumber;
  p = 2 (H^2): R_2 = (v^H K M^-1 K v)/(v^H M v) (M factored once per mesh);
gradients through the packer (any parametrisation; FD-checked 1e-9), elastic
in the QP, part of the violation and of admissibility. On the unit L2 sphere
either is a Sobolev ball: compact in L2, so a minimiser exists and the
discretisations converge to it. The difference is the RATE: the energy a
field can hold above wavenumber k is <= (kmax/k)^2 (H^1) or (kmax/k)^4 (H^2).
kmax = 1 (smallest feature ~ 3 length units; the smooth start has R_1^(1/2) =
0.53, R_2^(1/4) = 0.80 -- feasible for both).

- **H^1 barely constrains** (S1: ne=100, 80 modes, from the smooth start,
  results_fb_sob_n100): 3.895803 in 20 rounds / 68 min -- LOWER than the
  unconstrained 80-mode run (3.898938; the path differs), the row active at
  R_1^(1/2) = 1.000 and 6 % of the energy still above k = 3 with the spike at
  the cutoff (figs/fields_refine_spectra.png). (kmax/k)^2 at the cutoff is 8 %:
  the ball contains the rippled designs. H^1 is a correct regularisation in the
  limit and a useless one at these resolutions.
- **H^2 (tools/fields_mesh.sh S1/S2/S3 with ORD=2; results_fb_sob_n*_h2):**

| run | mesh / modes | start | gamma | rounds / min | energy k > 3 (act) | flat band | abs(K) |
|---|---|---|---|---|---|---|---|
| S1 H^2 | ne=100 / 80 | smooth start (3.935632) | 3.909258 | 20 / 63 | 3.2e-4 | [-11.0, +13.3] | 249 |
| S1 H^2, N=160 | ne=100 / 160 | same | 3.913414 | 20 / 79 | 3.0e-4 | [-10.3, +12.5] | 223 |
| S2 H^2 | ne=200 / 160 | frozen ne=200 (3.925457) | 3.911740 | 20 / 195 | 2.4e-5 | [-10.6, +13.0] | 229 |
| S3 H^2 | ne=400 / 160 | frozen -> ne=400 (3.948040) | **3.914071** | 20 / 281 | 2.9e-5 | [-10.3, +12.6] | 219 |

  The spectra of the three finished H^2 designs coincide up to k ~ 3 (a
  peak at k ~ 2.6-2.75 where the H^2 budget is spent, then a fall of 4-5
  decades) and differ only below 1e-4 beyond it (figs/fields_reg_spectra.png);
  the fields overlap (figs/fields_reg_fields.png); the flat band ends at
  w = 12.5-13.3 in all three -- the resonance of abs(k) = 2.7, where the
  constraint put the energy -- and abs(K) = 223-249. DOUBLING THE MODES (80 ->
  160 at ne=100) and DOUBLING THE MESH (100 -> 200) change gamma by +0.11 %
  and +0.06 %, within the runs' own convergence (each still creeping 0.003 %
  a round at round 20), where the unconstrained problem moved the band edge
  from 19 to 28 and kept going. (Equal optimal VALUES on two meshes do not
  make equal DESIGNS: the ne=100 H^2 design, carried to ne=200, is 5.8 % worse
  -- tuned to the ne=100 plant; ne=200 -> 400 is the meaningful pair, 32.7.) Gauge-invariant field distances
  (tools/field_distance.py, figs/field_distance_ne200.log; 0 = same shape,
  sqrt 2 = orthogonal): between H^2 designs 0.107-0.191 (the two ne=100 runs,
  same mesh, differ by 0.154: the objective is flat, the fields are determined
  to ~1-2 % in energy by the optimisation, not by the mesh), unconstrained 80
  vs 160 modes 0.345, unconstrained vs H^2 0.32-0.40. S3 (ne=400) repeats
  the picture: band [-10.3, +12.6], abs(K) 219, energy above k = 3 3e-5,
  gamma 3.914071 (+0.06 % over ne=200), fields 0.15 from S2's.

**32.6 Atoms.**
4 Gabor atoms per field (fieldbasis.atom_*, packer atoms mode, driver
`--field-atoms 4 --atom-smin 1 --atom-kmax 0.7071`, plus the H^1 row kmax = 1
so the sets are nested): v = sum_j a_j exp(-(x - c_j)^2/(2 s_j^2) + i kappa_j
(x - c_j)), s_j = smin (1 + rho_j^2) >= 1, kappa_j = kmax tanh(tau_j), so each
atom has RMS wavenumber sqrt(kappa^2 + 1/(2 s^2)) <= 1 by construction and a
Gaussian spectrum (smooth by construction: no cutoff to reach). 20 real
parameters per field (vs 320 for 160 modes); the reduced model keeps the
160-mode input basis. Start: the frozen fields fitted by 4 atoms (M-norm error
0.181 at ne=200, 0.179 at ne=400), which costs a lot with the controller held
(gamma 6.83 / 6.68 at the start: the optimum is very sensitive to the fields
with K fixed) -- recovered within 7 rounds.

| run | mesh | start | gamma | rounds / min | energy k > 3 | flat band | abs(K) |
|---|---|---|---|---|---|---|---|
| A2 | ne=200 | frozen fitted (6.829589) | 3.912595 | 15 / 62 (progress stop) | 6e-6 | [-10.2, +8.5] | 451 |
| A3 | ne=400 | frozen -> ne=400 fitted (6.676984) | **3.903930** | 11 / 82 (progress stop) | 6e-6 | [-10.9, +10.7] | 666 |
| A2' | ne=200 | A3's design carried to ne=200 (3.903963) | 3.888855 | 12 / 58 (round cap) | 4e-6 | | 1566 |
| A2'' | ne=200 | A2' continued | **3.876296** | 21 / 132 (progress stop) | | [-29.6, +19.2] | 4468 |

- A2 and A3 differ by optimisation, not by mesh: A3's design evaluated on
  ne=200 is 3.903963 and on ne=800 3.904026 (32.7), the atoms themselves
  nearly coincide (centres within 0.12, widths within 0.13, wavenumbers within
  0.02, both mirror-symmetric), and the fields of the two separately optimised
  designs are 0.027 apart (gauge-invariant M-distance,
  figs/field_distance_atoms.log; H^2 vs atoms 0.24, frozen vs atoms 0.16).
  Both runs were stopped by the progress stop (< 0.15 % over 3 rounds), which
  is too tight for the field problems: A2' -- A3's design carried to ne=200
  and re-optimised, 12 rounds / 58 min -- went on 3.903963 -> **3.888855**
  (-0.39 %), still falling ~0.03 % a round at the end; its design is 3.890959
  on ne=400 (+0.05 %, certified). A continuation of 40 more rounds is
  results_fb_atoms_n200_long: **3.876296** after 21 more rounds / 132 min
  (progress stop, 0.006 % over 3 rounds), the same design 3.879101 on ne=400
  and 3.879283 on ne=800 (+0.07 %; 400 -> 800 +0.005 %). So the optimum VALUE of the atom
  problem is below 3.889 and not yet pinned; every design along the way is
  mesh-independent to <= 5e-4.
- The K+FB+FC controller is a static gain K (nvec = 0). It drifts up as the
  fields improve (abs(K) 187 frozen, 451 A2, 666 A3, 1566 A2') but the designs
  sit in the high-gain regime where gamma depends on the fields only: with
  A2''s round-8 fields fixed, abs(K) x 2 / 5 / 20 / 100 moves gamma 3.8931 ->
  3.8935 / 3.8937 / 3.8938 / 3.8939 at an unchanged abscissa -0.2194. A flat
  direction, but it is where the equalised band now grows: A2'' (abs(K) 4468)
  is flat at gamma on w in [-29.6, +19.2] with sigma_1 still 2-3 at w = +-40
  and 0.6-0.8 at +-80 (figs/band_atoms_long.png) -- the smooth fields do not
  extend the band, the static feedthrough does. The objective has no
  control-effort term; a bound on abs(K) (or an effort channel in the
  performance output) would be the modelling choice if a finite gain is
  wanted. gamma is certified over ALL w for every record (tail bound), so the
  band leaving the loop's [-30, 30] window loses nothing.
- Atoms beat H^2 at kmax = 1 (3.889 vs 3.912 at ne=200) with 1/16 of
  the parameters and a third of the wall time per round (A2 alone equalled H^2 at
  the same mesh, 3.912595 vs 3.911740, before its early stop), fields smooth by construction
  (spectrum falls like a Gaussian: 1e-9 at k = 5); their shape is the H^2
  fields' without the k ~ 2.6 wiggles (distance to smooth start 0.16, to H^2
  0.21-0.24).
- Final atoms (A2): the actuator and sensor sets are MIRROR IMAGES (actuator
  centres -2.67, 0.29, 3.66, 8.84, sensor 2.65, -0.31, -3.66, -8.87: c -> -c,
  with the same widths 2.84 / 1.00 / 1.78 / 1.56 and wavenumbers 0.40 / -0.707
  / 0.28 / 0.44 atom for atom, and amplitudes within 0.03) -- the adjoint symmetry of the GL problem, found without being
  imposed. One atom per field sits on BOTH bounds (s = 1, kappa = -0.707): the
  atoms, too, want the finest content they are allowed; the bounds decide.
- Costs and caveats: non-convex in (c, s, kappa) (atoms can merge; here they
  did not); the count is a discrete choice; a poor fit at the start is
  expensive (+75 % with the controller held); the width and wavenumber floors
  are active -- the result depends on them as H^2's on kmax.

**32.7 The clean test: the same design on a finer mesh, nothing
re-optimised (tools/cross_eval.py -> tools/tail_certificate.py, dense level
set + tail bound, certified over all w; results_cross/, cross.log).** Two runs
on two meshes also differ by path and local minimum (32.5-32.6: ~0.1 % noise). Here
the design -- controller, and the fields as the FUNCTIONS they are (P2
evaluation of the finite-element field: the coarse P2 space is nested in the
fine one, so the forcing is the same function; atoms re-sampled exactly) -- is
carried to the finer mesh and certified there:

| design | on its own mesh | on the finer mesh | change |
|---|---|---|---|
| smooth start (ne=100) | 3.935632 | 3.938071 (ne=200) | +0.06 % |
| frozen report design, nodal + Sobolev map (ne=200) | 3.925389 | 3.930890 (ne=400) | +0.14 % |
| unconstrained, 80 modes (ne=100) -- the value in STATUS | 3.898938 | **5.520963** (ne=200) | **+41.6 %** |
| H^1 row kmax = 1, 80 modes (ne=100) | 3.895803 | **5.470988** (ne=200) | **+40.4 %** |
| unconstrained, 160 modes (ne=200, R1) | 3.888738 | **4.605269** (ne=400) | **+18.4 %** |
| H^2 row kmax = 1, 80 modes (ne=100) | 3.909258 | 4.135140 (ne=200) | +5.8 % |
| H^2 row kmax = 1, 160 modes (ne=200) | 3.911740 | 3.923768 (ne=400) | +0.31 % |
| 4 atoms (ne=200) | 3.912595 | 3.913894 (ne=400) | **+0.03 %** |
| 4 atoms (ne=400, A3) | 3.903930 | 3.903963 (ne=200) / 3.904026 (ne=800) | +0.001 % / +0.002 % |
| 4 atoms (ne=200, A2', re-optimised from A3) | 3.888855 | 3.890959 (ne=400) | +0.05 % |
| 4 atoms (ne=200, A2'', best K+FB+FC) | 3.876296 | 3.879101 (ne=400) / 3.879283 (ne=800) | +0.07 % / +0.08 % |
| H^2 row kmax = 1, 160 modes (ne=400, S3) | 3.914071 | 3.928785 (ne=200) / 3.915962 (ne=800) | +0.38 % / **+0.05 %** |

- The unconstrained designs are DISCRETE artefacts: their gain lives in the
  coarse plant's (wrong) response to the high-k content -- P2 at ne=100 has 3.3
  nodes per wavelength at the 80-mode cutoff, and the controller is tuned to
  that dispersion error. On the finer mesh they lose 18-42 %. The earlier
  transfer of the 80-mode design to ne=200 (tools/transfer_ne.py: 4.024146,
  "the transfer alone costs 3.2 %") interpolated LINEARLY, which smoothed away
  most of that content: the design as a function is 5.52 at ne=200.
- H^1 at kmax = 1 behaves like no constraint (+40 %).
- H^2 is mesh-consistent where the mesh resolves the plant at the constraint's
  wavenumbers (ne=200 -> 400: +0.3 %, ne=400 -> 800: +0.05 %), not at ne=100 (+5.8 %: the ne=100 plant
  is already 2 % off at the resonance, 32.4, and worse at k ~ 2.7).
- The atoms are mesh-independent: 3e-4 from ne=200 to 400 (A2), 1e-5 and 2e-5
  from ne=400 to 200 and 800 (A3) -- the plant's own discretisation error at
  these wavenumbers; so is the smooth start (6e-4 from ne=100).

**32.8 Conclusion.**

> **ERRATUM (STAGE6 6.36, 2026-10-01).** "Atoms give a LOWER gamma" below compared unequal starts and effort. From the same start on the same mesh, A2 equalled H^2 (3.9126 vs 3.9117). The 4-atom design is admissible for the H^2 problem (160-mode projection 5e-6, H^2 measure 0.60 < 1), and S2's problem started from it reaches 3.868466 in 5 rounds, below the atoms. The atoms' advantages are smoothness, mesh independence and cost, not the optimum.
- Viscosity regularises the response, not the design. The unconstrained
  field problem is ill-posed in the precise sense the user suspected: its
  optimum moves with the discretisation (energy at the cutoff, band edge at
  the cutoff's resonance), slowly, because the directions it uses are near-null;
  and its apparent gains are discretisation artefacts (+18 % to +42 % on the
  next finer mesh).
- A regularisation must state the admissible fields, scale-invariantly. H^1
  (the report's norm) is correct in the limit but too weak here; H^2 with
  kmax = 1 gives mesh- and basis-independent designs (gamma within 0.1 %,
  same spectra, same band) (optimised: 3.911740 at ne=200, 3.914071 at ne=400; the ne=400 design is 3.915962 on ne=800, +0.05 %; H^2 fields at ne=200 and 400 are 0.15 apart, figs/field_distance_final.log); 4 Gabor atoms with width >= 1 and
  wavenumber <= 0.707 give a LOWER gamma with 20 parameters per field and
  smooth fields, and are the most mesh-independent (3.903930 at ne=400, the
  same design 3.904026 on ne=800; continued at ne=200 to 3.888855, 3.890959 on
  ne=400; continued further to 3.876296, 3.879283 on ne=800; separately optimised atom fields at ne=200 and ne=400
  0.03 apart).
- What free fields really buy, measured on the finest mesh: the frozen report
  design 3.930890 (ne=400); H^2 3.915962 (-0.4 %) and the atoms 3.879283
  (-1.3 %) on ne=800 -- the atoms, smooth and mesh-independent, beat every
  unconstrained value, which were artefacts (3.8989 at ne=100 is 5.52 as a
  function on ne=200).
- kmax (and the atoms' width and wavenumber bounds) are MODELLING choices --
  the smallest feature an actuator or sensor can have -- to be set from the
  application (for the cavity: the size of a plasma actuator or a pressure
  tap), not numerical parameters; the values here (features >= ~3 length
  units) are illustrative.
- The field runs stop too early with the K+B+C progress stop (3:0.0015): the
  atoms went on from 3.9039 (stopped) to 3.8763 in 33 more rounds. Use a
  looser stop (or none) for field cases.
- Recommendation: atoms for the cavity (tens of parameters per field in 2-D,
  mesh-free, smooth; the input basis of the reduced model stays modal), with
  an H^2 row available when a modal parametrisation is wanted; the 80-mode
  unconstrained K+FB+FC value (3.898938) is WITHDRAWN (5.52 on ne=200).
- Plant discretisation: judge any gamma at ne >= 200 (0.14 % from converged
  open loop); ne=100 is for debugging.

## 33. Algebra item 11: the continuation gate on the reduced pencil (2026-09-28, user: address the open item that needs neither cluster nor cavity)

**Why it did not need the cluster.** STATUS listed item 11 as "validation
needs the 27-chain replay (cluster)". The validation that matters is not the
27 frozen chains but the gate's DECISIONS against the truth, and at GL size the
truth is cheap: dense eigenvalues of the closed loop along each step's
segment. So the replay was done here, on the two nominal runs' own steps.

**What changed (run_case_epi.continuation_gate_reduced, `--cont-track
reduced`, option).** Same decisions and splitting as continuation_gate (the
full gate); only the source of the eigenvalues changes:
- at every point of the segment, the reduced closed-loop pencil's Ritz pairs
  near the line (the oracle's `near_line_pairs`, kmax 16), each ACCEPTED only
  if its full-model residual (sparse matvecs, no factorisation) is <= 1e-3;
- each watched eigenvalue is tracked by the nearest accepted Ritz value; if
  none matches within the motion allowance it is tracked on the full model
  exactly as before (one shift-invert) -- the fallback;
- a midpoint line certificate (10-15 factorisations in the full gate) is
  replaced by the rightmost accepted Ritz value; with none accepted, the full
  certificate runs.
So every watched eigenvalue is followed at least as reliably as by the full
gate; what the reduced gate can miss is only a NEW eigenvalue entering the
band between watched points without appearing among the accepted Ritz pairs,
which is what the replay tests.

**tools/gate_replay.py** (new): for every step a run took (accepted design ->
next candidate, from the dumps), the closed-loop abscissa along the segment on
101 points by DENSE eigensolves (the truth: crosses Re = -eps or not, end
stable or not), then the full gate and the reduced gate (reduced model
anchored at the step's start as the loop anchors it), with full-model
factorisation counts.

| run | steps | truth crosses -eps | full gate refuses | reduced gate refuses | agreement | false accepts (full / reduced) | full-model LU (full / reduced) |
|---|---|---|---|---|---|---|---|
| ne=100 K+B+C (nominal, 6.389368 run) | 25 | 2 (rounds 8, 13) | 3 (8, 12, 13) | 3 (same) | 25/25 | 0 / 0 | 1172 / 106 |
| ne=200 K+B+C (nominal, 6.486505) | 17 | 2 (rounds 4, 5) | 2 (4, 5) | 2 (same) | 17/17 | 0 / 0 | 658 / 60 |

- Identical decisions on all 42 steps, including the one conservative refusal
  (ne=100 round 12: path clear by 0.09, both gates split 8 times and refuse)
  and all four true crossings (max abscissa on the path up to +0.256).
- Every factorisation the reduced gate made was a fallback (a watched
  eigenvalue with no accepted Ritz match, or a midpoint with no accepted pair);
  the splits are the same (27 / 27 at ne=100, 16 / 15 at ne=200).
- Factorisations: 11x fewer on both meshes.
- Refused steps are still resolved by the exact winding count (as before);
  the gate is a continuity check, the stability of every saved design remains
  the winding certificate's.

**In the loop (ne=100 K+B+C, tools/n100_kbc.sh with `EXTRA="--cont-track
reduced"`, results_n100_kbc_ctr):** 6.405826 in 16.7 min (18 rounds, progress
stop; 43 full-model fallbacks in all; 3 refused continuations resolved by the
winding count, as in the nominal run; witness invariant 0 violations) against
the nominal 6.399134 in 22.6 min (21 rounds). The two runs are identical to 7
digits through round 7 and part at round 8 (the watch list handed to the next
round now holds Ritz values instead of full eigenvalues, so the anchors differ
in the last digits and the non-convex trajectory separates). From round 10 on
the reduced-gate run is AHEAD at every round (round 17: 6.405826 vs 6.408609;
at equal wall time, 16.6 min, the nominal run was at 6.4122, round 16; the
wall time per round is the same, ~1.75 min: at GL the gate's factorisations are
a small share of a round, and 16.7 vs 22.6 min is the round count), but the
progress stop (< 0.15 % over 3 rounds) fired at round 17 (6.4115 -> 6.4058,
0.09 %) where the nominal run crept on to 6.3991. The final difference (+0.10
%) is the stop rule meeting a different trajectory, not a gate decision (the
replay shows identical decisions).

**Decision:** item 11 **done** as an OPTION (`--cont-track reduced`),
not the default. At GL the gate's factorisations are cheap (~20 per round of
~100) and the rule is never to regress a final value; the full gate stays the
default. The option is validated (42/42 decisions, 0 false accepts, 11x fewer
factorisations in the gate) and is the one to switch on at the cavity, where a
factorisation dominates and a refused continuation costs 400-960 of them.

## 34. The static gain at convergence; no square root of M anywhere; 2 actuators / 2 sensors at ne=100 (2026-09-28/29, user: does abs(K) diverge; M products only; decentralized and centralized MIMO "just to see if it works"; ne=400 belongs on the cluster)

**34.1 Does the K+FB+FC static gain diverge?** No sign of it. With the fields of
the best atoms design (A2'', 3.876296) held fixed, gamma against the gain
(grid over w in [-120, 120]):

| abs(K) / abs(K*) | 0.25 | 0.5 | 0.8 | 1 | 1.25 | 2 | 5 | 20 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| gamma | 8.927 | 5.411 | 4.206 | **3.87629** | 3.87630 | 3.87636 | 3.87642 | 3.87645 | 3.87646 |

(arg K +-0.1 rad: 3.9035 / 3.9607; abscissa -0.2188 throughout.) gamma falls
steeply up to abs(K*) = 4468 and then RISES slightly to the high-gain limit
3.87646 (+4e-5 relative): for given fields the optimal gain is finite, at the
knee where the loop becomes high-gain. It grew over the runs (187 -> 4468)
because the fields changed: better fields put the knee at a larger gain. The
value at convergence is large but finite and weakly determined (a flat valley:
gamma moves 1e-5 between K* and infinity), while its phase is sharply
determined. A control-cost weight (the user's H-infinity norm of W K S) is the
principled fix; kept for later (user, 2026-09-28).

**34.2 M products only (user: "surely there is a way to use products by the
sparse M").** Audit of every place M enters:
- the loop's gain: sigma^2 = lambda_max(M T^H M T) by Arnoldi, T applied
  through ONE sparse LU of i w M - A and a Woodbury correction (rank 2 SISO,
  rank 4 in 34.3): M only as a sparse mat-vec (closedloop_sparse, header).
  Gradients and dsigma/dw likewise;
- the level-set certificate: the even pencil [[A_cl, M/g^2], [M, A_cl^H]] -
  s diag(M_cl, -M_cl) with M as a block, shift-invert with sparse LU
  (certify_sparse.SparsePencil); the dense variant (GL reference) builds the
  same pencil densely -- no square root;
- the reduced model: square roots of the k x k Gram matrices V^H M V (k ~ 100),
  never of the full M;
- **fixed today** (they built a dense copy of M, or a dense Cholesky factor):
  `fieldbasis._m_orth` (dense M -> sparse products), `fieldbasis.laplace_basis`
  (dense generalised eigh -> sparse shift-invert Lanczos; eigenvalues equal to
  2e-12, same modes up to sign), `fieldbasis.fit_atoms` (least squares on a
  dense Cholesky factor -> r^H M r with its gradient 2 Re(J^H M r); better fit,
  0.155 vs 0.181 on the frozen field, 90x faster), `tools/tail_certificate.py`
  (nu = ||M^-1/2 A M^-1/2|| from the dense SVD of L^-1 A L^-H -> Lanczos on the
  pencil (A^H M^-1 A, M) with a sparse LU of M; equal to 1e-15),
  `winding_cert._plant_tail_constants` (dense SVD of M^-1 A at n <= 3000,
  power iteration x 1.3 above -> Lanczos with the sparse LU of M; equal to
  1e-15);
- left, as GL-only reference tools by design (dense everything, never in the
  loop's H-infinity path): `closedloop_sparse.dense_reference` (selftest
  validation, labelled), `FECoDesign.alpha_dense` (the spectrum stored in the
  records), `winding_cert.plant_pole_data` (dense plant spectrum once per mesh,
  refuses n > 3000 with a message), fe_pipeline / prepare.py (dense LQG seeds),
  stabilise.py, warmstart.py (legacy starts). A sparse LU of M (winding tail,
  smooth_unmap, the H^2 field row) is a factorisation of the same sparsity class
  as the resolvent's, not a square root; lumping M would remove even that.
Invariants after the changes: frozen path 10.23128516441147 PASS, witness invariant 0 violations PASS, Hurwitz selftest 0/200 PASS.

**34.3 Two actuators, two sensors at ne=100 (tools/mimo_proto.py,
tools/mimo_plots.py; results_mimo/).** A standalone prototype, not the
production loop: the design u = K(s) y with y = C^H q, C = M [psi_s1, psi_s2],
B = M [psi_a1, psi_a2]; the closed-loop resolvent by the same sparse LU and a
RANK-4 Woodbury correction; sigma from M T^H M T (M products only; gradients
FD-checked to 7 digits for sigma_1, sigma_2, the abscissa, the controller
entries, the positions and the atom parameters); an epigraph over a working set
of frequencies (sigma_1, sigma_2 rows) plus the rightmost closed-loop and
compensator eigenvalues, solved by SLSQP in a trust box; every candidate
CERTIFIED by the dense level set on the even pencil (all eigenvalues: global in
w -- it caught candidates whose peak sat at w = 300-4600, far outside any band)
and the dense closed-loop spectrum, refused and the box halved otherwise.
Floors: with one pair sigma_1^cl >= sigma_2^ol pointwise, with two pairs
sigma_1^cl >= sigma_3^ol (rank-2 loop, interlacing): sup_w sigma_2^ol =
3.509145, sup_w sigma_3^ol = 2.383533 at ne=100.

Starts: the SISO optimum as loop 1 (loop 2 switched off) -- the prototype
reproduces it exactly (K+B+C 6.399134, atoms 3.877016 = the ne=200 atoms design
evaluated at ne=100).

| family | design | gamma (ne=100, certified, all w) | vs SISO | rounds / min | notes |
|---|---|---|---|---|---|
| K+B+C (Gaussians at free positions, dynamic compensator, total order 16) | SISO optimum | 6.399134 | -- | -- | results_n100_kbc |
| | decentralized, pair 2 started at -+6 | 6.281979 | -1.8 % | 25 / 20 | stalled: pair 2 stays mirror-symmetric at -+5.84 |
| | decentralized, pair 2 started at -+9 | 6.286431 | -1.8 % | 26 / 21 | same stall (-+8.84) |
| | **decentralized, pair 2 started at -+4** | **5.594745** | **-12.6 %** | 40 + 26 / 44 | converged (progress stop); pairs at -+2.23 and -+3.34 |
| | central, from the SISO start | 5.821752 -> 4.920807 -> 4.591682 -> **4.277652** | **-33.2 %** | 4 x 40 / 122 | still falling ~0.1 % a round; pairs at -+1.71, -+4.77 |
| | central, from the best decentralized | 5.090384 | -20.5 % | 40 / 27 | below its start 5.594745 (as it must be); not continued |
| K+FB+FC (4 Gabor atoms per field, static gain) | SISO optimum (the ne=200 atoms design at ne=100) | 3.877016 | -- | -- | |
| | **decentralized** | **2.387150** | **-38.4 %** | 29 / 31 | 0.15 % above the 2-pair floor 2.383533 |
| | central, from the decentralized | 2.386009 | -38.5 % | 10 / 7 | 0.10 % above the floor; off-diagonal gains <= 0.03 |

- It works: both structures optimise, every accepted design is certified
  stable (loop and compensator) and certified in gamma over all frequencies.
  Figures: figs/mimo_sigma.png (sigma_1 of the three designs per family against
  the uncontrolled sigma_1 and the two floors; sigma_1..3 of the centralized
  design), figs/mimo_fields.png (actuator and sensor fields), figs/mimo_controller.png
  (|K_ij(i w)| of the K+B+C compensators).
- With free fields the second pair is worth almost everything it can be: the
  floor for two pairs is sup_w sigma_3 of the open loop, 2.383533, and the
  decentralized design is within 0.15 % of it (sigma_1 of the closed loop sits
  on the uncontrolled sigma_3 near w = -0.7). Centralizing then buys only 0.05 %
  and the optimiser leaves the cross gains near zero (|D_12|, |D_21| <= 0.03
  against |D_11| = 4.2e4, |D_22| = 12): with shaped fields each sensor can be
  matched to its own actuator.
- With Gaussians (fixed shape) the structure matters: decentralized loops gain
  12.6 %, the centralized compensator 33 % and still falling; its cross
  couplings K_12, K_21 are 10-60 in magnitude over the band -- a few times
  below K_11 but well above K_22 (figs/mimo_controller.png). With fixed-shape actuators the controller has to
  do the mixing that shaped fields do by themselves.
- The decentralized problem is multimodal in the second pair's position:
  started at -+6 or -+9 it stalls at 6.28 (the pair stays mirror-symmetric,
  x_a2 = -x_s2 to 1e-9 -- the GL adjoint symmetry again -- and never moves
  inward); started at -+4 it reaches 5.59. The centralized result depends on
  its start too (4.28 from the SISO start after 160 rounds, 5.09 from the best
  decentralized after 40). These are local results of a prototype.
- Cost at ne=100: 30-60 s a round (up to 290 working-set frequencies x sigma_1,2
  with gradients, plus the dense certificate and spectrum). The prototype
  converges slowly (SLSQP restarted every round without curvature memory; no
  peak following, no sigma_2 Clarke blocks, no face S for the near-double
  compensator pair; candidates refused for sharp new peaks near w ~ 4-7 where a
  closed-loop eigenvalue sits on Re = -eps). Everything the production loop has
  for these (STAGE5 9-30) carries over; the rank-2 algebra is what changes.
- For the production code (algebra item 7): the resolvent (rank 2 -> 4
  Woodbury), the gradients, the level-set pencil (rank 4 -> 8 coupling) and
  the reduced model generalise directly (done here in the prototype); what needs
  real work is the winding count (the scalar 1 - k G becomes det(I - K(s) G(s)),
  a 2x2 determinant -- still one number per frequency), the compensator's face S
  for MIMO pole pairs, the packer for (B_c, C_c, D_c) blocks, and the driver's
  SISO bookkeeping (~20 files touch b, c, nvec, K as scalars/vectors).
- Plant note: ne=100 carries the 2.3 % plant error of 32.4; the values above
  compare structures on the same (coarse) plant, as asked.
