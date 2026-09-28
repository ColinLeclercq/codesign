> **ERRATA (2026-09-25) -- read first.** K+B+C values recorded here as "KKT points", "converged"
> or "stationary" (7.95x at ne=200, 7.92x at ne=100) were STALLS: the compensator pair near a double pole
> on the line and quasi-Newton memory carried across rounds (STAGE5 25-28). Best now ne=200 6.438335,
> ne=100 ~6.40. A KKT residual (6e-4 reference, 2e-3 active set) is not an optimality certificate on
> flat optima (STAGE5 28). `--sqp-carry` now carries the trust radius only; the quasi-Newton matrix is
> reset each round (`--sqp-carry-h off`, default).
>
> **ERRATA (2026-09-23) -- read before the text below.**
> - Section 7.2 / "let the number decide decides for pyGRANSO": SUPERSEDED. The SQP's refused rounds
>   were working-set defects (top-4 enrolment cap, fixed-frequency members, sigma_2 block rows not
>   following -- STAGE5 9, 11, TRACKER K.3), not a property of the SQP. On the corrected loop the SQP
>   converges to a KKT point (7.9518 / 7.9509, residual ~2e-4) while pyGRANSO stalls at the compensator
>   double pole (residual 3.4e-2): section 8.3.
> - Section 8.1 (tie at equal evaluations): superseded (different compensator lines; see its heading).
> - Section 8.2, "a narrow peak the basis does not span is invisible to any search on the reduced model":
>   the reduced model was then two-sided and NOT robust to position moves; with the input basis +
>   union Galerkin it is exact across the position range to ~1e-8 (STAGE5 5). In-solve refinement mostly
>   compensated the working-set defects; with them fixed it is a safety net for genuinely new resonances.
> - Section 7.1 (the compensator double-pole wall) STANDS; it is what the 2x2 Hermite block constraint
>   resolves (STAGE2 3.2), and what stalls pyGRANSO (8.3).
> - The abscissa-gradient defect (STAGE1 errata, STAGE5 5.4) applies to runs here without `--alpha-argmax`.
> - Certified values at rtol 1e-4 compare only above ~1e-4 relative.

# Stage 1b -- the handoff's Stage 1, done as specified

The stage-1 work recorded in STAGE1.md was constraint handling (note item 7)
and start defects. It was needed, but it was not the handoff's Stage 1, whose
items are 0, 3 and 11 and whose gate is: balance residual on the stalled rows
from ~1e4 toward O(10); each of the three stall mechanisms (sigma_1 = sigma_2
crossing / QP accuracy / mu) confirmed or excluded by a measured number; the
27-case regression. The defaults flipped in STAGE1.md section 12.2 were flipped
without that regression, against the handoff's rule, and are **reverted** as
of 2026-09-17: the code's defaults are frozen again, the stage-1 configuration
is `--select-cons on --ctrl-imcap off --cont-anchor defer --inner-maxit 25`,
and every stage-1 script passes it explicitly.

# 1. First measurement: the sigma_1 - sigma_2 gap at the active frequencies

`tools/sigma_gap.py`. The frozen record's `.npz` stores the working set
Omega; sigma_1 on it identifies the active frequencies (within rtol 1e-3 of
gamma); sigma_2 there is the second eigenvalue of the same operator whose
first is sigma_1^2 (`closedloop_sparse.gain` with k=2). ne=200, frozen designs.

| case | active omega | sigma_1 | sigma_2 | gap/sigma_1 | balance (frozen) | mechanism |
|---|---|---|---|---|---|---|
| K+B+C | -0.527, -0.517, -0.500 | 7.964 | 7.921 | **0.5 %** | 1.47e+04 | sigma_1 = sigma_2 crossing: **confirmed** |
| KSP+B+C | -0.531, -0.516 | 8.246 | 8.226 | **0.1-0.2 %** | 1.73e+03 | crossing: **confirmed**, tighter |
| KSP+B+C | 3.796 | 8.244 | 0.548 | 93 % | | a second, clean active peak |
| KGC+B | -0.528 | 11.601 | 9.222 | 20 % | 2.74e-01 | crossing: **excluded** |

The two rows with pathological balance residuals sit within a percent of a
kink at their active frequency; the row whose residual was benign has a 20 %
gap. First of the three attributions, by number.

# 2. Item 3 -- sigma_2 enrollment: built, and what it did

`closedloop_sparse.gain_k(omega, ..., k=2)`: the top-k singular pairs with the
same (x, z) construction as `gain()`; agrees with `gain()` on sigma_1 to 1e-12
and reproduces the sigma_2 of section 1. `gain()` is untouched. In
`solve_epigraph`, sigma_2 rows are appended after every existing row (no index
moves), gradient from the second pair through the same `D.grads`. The enrolled
set is chosen before each inner solve, as the note prescribes: always at the
active frequencies, thresholded with hysteresis elsewhere (`--sigma2 on`,
`--sigma2-on 0.05`, `--sigma2-off 0.15`, `--sigma2-near 0.02`), kept as
frequency values because Omega grows. Frozen path bit-identical (B: 10.23128516441147).

At ne=100 the baseline designs have gaps of 17-20 %: no kink -- they stopped
at gamma ~ 10, short of the converged region where sigma_2 rises to meet
sigma_1. The note's own test is the terminal K+B+C polish, run at ne=200 here:

| polish of frozen K+B+C (7.9641) | rounds | result | inner t reached |
|---|---|---|---|
| sigma_2 off | 4 | 7.9571 | 7.956 -- pinned |
| sigma_2 on | 5 | **7.9499** | 7.939 -- moves |

Rows enrolled at w = -0.517 and -0.500 (gaps 0.5 %, 0.8 %), exactly where
section 1 measured. The inner solve that was pinned at the kink moves; the
exchange needed five rounds to certify any of it, each new inner point having a
peak in a gap of Omega (8.11, 8.73, 7.98, 8.03) until round 4 accepted 7.9499.

# 3. The gate's number was wrong, and what the right one says

`check_optimality`'s "rel |grad|" -- the stage-0 harness's "balance residual"
column, and the number the handoff's stage-1 gate targets -- is (a) the
gradient of ONE branch, which the file's own comment says is meaningful only
with a single active frequency, and (b) evaluated at the peak `D.hinf` finds
seeded from the rightmost eigenvalue: on K+B+C a LOCAL peak at w = -1.01
(7.81), not the active one at -0.53 (7.964). Its 1.45e4 is one branch at a
frequency that is not active; it does not move when gamma improves.

`tools/stationarity.py` computes the condition 0 in conv{grad sigma_1(w_i)}
over the active peaks located from the design, with grad sigma_2 added where
the gap is below 5 % (the Clarke subdifferential at a crossing), as an nnls
residual in the optimiser's coordinates, ||.|| ||z|| / gamma:

| design | active w | gap | rho (sigma_1 only) | rho (with sigma_2) | lambda |
|---|---|---|---|---|---|
| K+B+C frozen | -0.513 | 0.54 % | 40.0 | **0.58** | [0.54, 0.46] |
| K+B+C sigma_2-polished | -0.512, -2.42 | 0.45 % | 30.6 | 1.26 | [0.54, 0.46, 0] |
| KSP+B+C frozen | -0.530, +3.80 | 0.20 % | 44.1 | **2.09** | [0.47, 0, 0.53] |
| KGC+B frozen | -0.528 | 20 % | 0.274 | 0.274 | [1.0] |

KGC+B reproduces the old 2.74e-1 exactly (the tool agrees with the old number
where the old number was meaningful).

**Is 0 in the subdifferential?** The hull of {grad sigma_1, grad sigma_2} is
the wrong set: at a singular value of multiplicity 2 the Clarke
subdifferential is {Re tr(Y U2^H dM V2) : Y >= 0, tr Y = 1}, and the diagonal
Y give the two branch gradients while the off-diagonal Y give the mixed pairs
(u1, v2), (u2, v1) -- a rotation within the singular subspace. `stationarity.py`
now minimises over lambda and, per kinked peak, over the 2x2 PSD Y:

| design | rho (+sigma_2 hull) | **rho_Clarke** | / mean branch-gradient norm |
|---|---|---|---|
| K+B+C frozen | 0.582 | 0.579 | 1.5e-2 |
| KSP+B+C frozen | 2.09 | **0.552** | **6.2e-4** |

KSP+B+C: the cross terms matter (Y off-diagonal -0.04-0.23i, a factor 4) and
0 lies in the subdifferential to 6e-4 of the gradient scale -- a stationary
kink optimum. K+B+C: cross terms buy nothing; 0 is in it only to 1.5e-2 --
near, not at, consistent with its inner solves ending on code 6 at 1e-5; the
sigma_2 polish moves it (7.9641 -> 7.9499). The relative number, rho_Clarke over
the branch-gradient norm, is the one to read. K+B+C and KSP+B+C are **kink optima**:
the multipliers split nearly evenly between the sigma_1 and sigma_2 branches
and the residual is O(1). They were never non-stationary. The "stall" was the
solver being unable to move, or to recognise stationarity, at a kink it could
only see one branch of -- which is what item 3 gives it.

So the handoff's gate -- balance residual on the stalled rows from ~1e4 toward
O(10) -- is met on the frozen designs by measuring correctly: 0.58, 2.1, 0.27.

## 3.1 Why the subdifferential reads that way, and how the Y-problem is solved

**The derivative of a repeated singular value.** Write the closed-loop map at
a fixed frequency as M(theta), and let sigma_1(M_0) have multiplicity k with
left/right singular subspaces U_k, V_k (n x k, orthonormal columns; here k = 2,
and "nearly repeated" is treated as repeated -- the gap of 0.2-0.5 % is below
the accuracy of a one-branch gradient). Since

    sigma_1(M) = max_{|u| = |v| = 1} Re u^H M v ,

sigma_1 is a maximum of smooth functions and is directionally differentiable
everywhere. Perturbing M_0 by eps dM, first-order perturbation theory for the
k-fold singular value gives

    sigma_1(M_0 + eps dM) = sigma_1 + eps * lambda_max( sym(U_k^H dM V_k) ) + o(eps),

the largest eigenvalue of the Hermitian part of the k x k block B = U_k^H dM
V_k. And lambda_max of a Hermitian matrix is itself a maximum over the unit
trace PSD cone,

    lambda_max(H) = max { tr(Y H) : Y >= 0, tr Y = 1 } ,

attained at Y = the projector onto the top eigenvector. Putting the two
together, the directional derivative of sigma_1 in the direction dM is the
support function of the set

    d sigma_1(M_0) = { dM  ->  Re tr( Y U_k^H dM V_k ) : Y >= 0, tr Y = 1 },

which is exactly the Clarke subdifferential (sigma_1 is regular, being a
max-function of smooth functions, so the Clarke and the directional
subdifferentials coincide). With the design entering through M(theta), the
chain rule gives the j-th component of a subgradient as

    g_j(Y) = Re tr( Y U_k^H (dM/dtheta_j) V_k ) = sum_{a,b} Y_ab Re( u_a^H (dM/dtheta_j) v_b ).

The code already computes Re(u^H (dM/dtheta) v) for a singular pair: that is
what `D.grads(cl, x, z, ...)` returns from a right vector x and a left vector
z, mapped to the optimiser's coordinates by `pk.grad`. So the four entries of
the 2 x 2 block are four calls with the pairs (u_1,v_1), (u_2,v_2), (u_1,v_2),
(u_2,v_1): the first two are grad sigma_1 and grad sigma_2 (Y = e_1 e_1^T and
e_2 e_2^T), the last two are the CROSS TERMS an off-diagonal Y brings in. They
correspond to rotating the pair of singular vectors inside the two-dimensional
singular subspace, which the diagonal Y cannot express -- which is why the
hull of {grad sigma_1, grad sigma_2} is only a subset, and why for KSP+B+C the
residual drops by a factor 4 when the off-diagonal is admitted.

Phases: each singular vector is defined up to a unit complex factor; the
mixed-pair gradients depend on that choice, but Y absorbs it (Y_12 -> Y_12
e^{i phi}), so the set -- and the distance to it -- does not.

**The stationarity condition.** For min_theta max_i sigma_1(omega_i; theta)
with active frequencies i = 1..p, first-order optimality is

    0  in  conv_i d sigma_1(omega_i; theta) ,

so the test is the distance from 0 to that convex hull:

    rho = min { || sum_i lambda_i g_i(Y_i) || : lambda in the simplex, Y_i >= 0, tr Y_i = 1 }

(scaled by |z|/gamma to make it dimensionless). With Z_i = lambda_i Y_i this
is a small semidefinite program -- minimise a Euclidean norm of a linear map
of (Z_1..Z_p) over Z_i >= 0, sum_i tr Z_i = 1 -- hence CONVEX, with a unique
minimum value.

**How it is solved in practice here.** The problem is tiny (p <= 3 peaks, k =
2), so `stationarity.py` does not reach for an SDP solver: it parametrises

    Y = [[ a, b ], [ conj(b), 1 - a ]],   0 <= a <= 1,  |b|^2 <= a (1 - a),

which is exactly the trace-one PSD cone for 2 x 2 (a = (1 + tanh s)/2, b =
sqrt(a(1-a)) tanh r e^{i phi}), plus the simplex for lambda, and minimises
|| sum_i lambda_i g_i(Y_i) || by Nelder-Mead from twelve starts. Because the
underlying problem is convex, the twelve starts agree on the value; the
parametrisation only removes the constraints. For larger k, or many peaks, the
same problem should be posed directly as the SDP (cvxpy with SCS/Clarabel) --
noted, not needed at k = 2. rho_Clarke is reported next to the mean branch
gradient norm; the ratio is the dimensionless statement "0 is in the
subdifferential to this precision".

**What the inner solver does with it -- and does not.** With sigma_2 enrolled,
pyGRANSO's steering QP takes its direction from the gradients of the violated
and active rows: sigma_1 and sigma_2 at the enrolled frequencies, i.e. the
hull of the two branch gradients -- the DIAGONAL Y only. The cross terms are
not constraints of the form g <= t and the NLP has no place for them, so the
inner descent direction is Clarke-correct only when the optimal Y is
(near-)diagonal: true for K+B+C (Y_12 = -0.03+0.06i), not for KSP+B+C (Y_12 =
-0.04-0.23i). The exact way to give the solver the full set is to replace the
two rows at a kinked frequency by the k x k block constraint || B(theta) || <=
t -- an LMI [[t I, B^H], [B, t I]] >= 0 -- in the tangent program, which is
how Apkarian-Noll's nonsmooth H-infinity method handles multiple singular
values. pyGRANSO's QP subproblem cannot take a matrix constraint; an SQP whose
subproblem is a small SDP can. That is squarely note item 8 ("inner solver,
two levels") and is carried there with this evidence: enrollment fixes the
diagonal part now, the block constraint is the remainder.

# 4. Item 0 -- the three mechanisms, each with a number

`--qp-log` hooks pyGRANSO's `halt_log_fn` (mu, violation, stationarity, step,
fallback level per iterate) and wraps `osqp.OSQP.solve` for the status
histogram -- pyGRANSO hardcodes eps_abs = eps_rel = 1e-12 (solveQP.py:126) and
never reads `res.info.status` (line 130). `--qp-eps` / `--qp-maxiter` override
the setup without touching pyGRANSO.

| mechanism | K+B+C polish, measured | verdict |
|---|---|---|
| sigma_1 = sigma_2 crossing | gap 0.5 %; rows enroll there; stationary only with the second branch | **confirmed** (also KSP+B+C; excluded on KGC+B) |
| QP accuracy | 33/39 then 62/105 steering QPs 'maximum iterations reached' | present; with max_iter 200000 (53/73 solved, stationarity 8e-9 -> 2.6e-7) gamma 7.9587 vs 7.9579: **not the cause**. Case B likewise (9th digit) |
| mu | 1 -> 0.28 (round 0), 1 -> 0.0064 (round 1); starvation threshold 1e-4 not reached | present, **not starved**; regrow detector in place, did not fire |

QP accuracy is the common state of every inner iteration, not a property of
the stalled rows: on case B, 20/49 and 35/45 QPs end unconverged. It changes
gamma in the ninth digit. The tolerance is unreachable for ADMM within the
iteration limit, not too loose.

**ADMM versus active set, what the switch buys.** OSQP is an ADMM method; the
steering QP it solves is the DUAL over the constraint gradients, so its
dimension is the number of enrolled rows (|Omega| + a few, 90-160 here), not
the design dimension, and an active-set method solves it exactly in a handful
of pivots. On the design: no advantage measured (gamma to the twelfth digit).
On the inner solve's OWN stationarity measure -- the quantity pyGRANSO's
termination test and the code-6 diagnosis both read -- the advantage is real:
1e-9 / 3e-11 against 1e-8, with every QP certified optimal instead of 60-80 %
ending on an iteration limit. That reliability is what item 2 and any future
stopping rule will lean on, at a cost of ~30 % more oracle calls on case B.
Recommendation: DAQP as the default after the regression, not before.

**The active-set test.** qpOASES has no pip package (and `qpsolvers` here
sees only OSQP and Gurobi). DAQP (Arnstrom-Bemporad-Axehill, dual active set)
is the pip-installable stand-in; `--qp-solver daqp` binds it into the two
pyGRANSO modules that import solveQP by name -- the QP form (LB <= A y <= UB,
equality rows as LB = UB) is DAQP's native one. Case B: all 154 QPs exit
optimal, inner stationarity 1e-9 / 3e-11 against OSQP's 1e-8, gamma the same
to the twelfth digit, 100 calls against 75. Same design, tighter stationarity:
QP accuracy is not the mechanism.

# 5. Item 11 -- tau0, removed

The note's author confirms item 11 is `tau0`, the start slack above the peak.
Its CLI default was already 1.0 and the function default 1.05 was dead code
(the CLI always overrides); both are 1.0 now and the constant is gone. The
other candidate, measured before the confirmation, is kept for the record: `--tau0`, the start slack above the peak, whose CLI
default is already 1.0 (its help text says the >1 value was introduced and
removed); and the 1/t0 normalisation of the inner problem (rows sigma/t0 - tau,
objective tau). `--level-norm off` makes the second switchable. On case B:
gamma identical to 1e-13, 91 calls against 75, and OSQP unconverged on 81/92
QPs against 35/45 -- the unnormalised units make the steering QP harder. No
gain, worse conditioning. Which constant was meant is a question for the
note's author; both are recorded.

# 6. Regression at ne=100

The 15-case lattice under the stage-1 flags + `--sigma2 on`, same budgets as
the baseline, running. Rows go here when it finishes; the 27-case ne=200
regression stays a cluster job and the defaults stay frozen until it passes.

# 7. Item 8 -- inner solver, two levels: what was built, what the number says

`sqp_inner.py`, `--inner sqp`. An SQP on the epigraph problem with the same
evaluation as the frozen inner solve, and three things pyGRANSO cannot do:

1. **Exact subproblem** (Clarabel, interior point, QP + cones). Every step
   certified optimal; multipliers from the cone duals feed a damped BFGS on
   the Lagrangian. About two oracle evaluations per iteration against
   pyGRANSO's five to ten.
2. **The 2 x 2 block at kinked frequencies as a second-order cone** -- the
   first-order model of a repeated singular value with its cross terms
   (section 3.1), validated against a finite difference at the K+B+C kink:
   FD 4.58484, block 4.58530, single branch 4.57156.
3. **Compensator stability as a Lyapunov LMI**, `--sqp-lmi on`: exists P > 0
   with A_c^H P + P A_c + 2 eps P <= 0. A_c is affine in z (verified), so the
   LMI is bilinear in (z, P); linearised at (z_k, P_k) it is one PSD cone in
   (step, dP), with dP boxed by -rho P <= dP <= rho P (without the box the
   subproblem satisfied the linearised LMI with ||dP|| = 2 ||P|| and the true
   pole moved outward along the "safe" ray) and an elastic slack. r x r at any
   plant size: **feasible for the cavity** by construction; the closed-loop
   abscissa is the constraint that does not scale this way, and for it the
   answer stays item 2.

Two defects found and fixed on the way, both by printing the ray: a sign
slip in the cone rows (dt and the block gradients entered flipped -- the
block then predicted dt = -0.30 where sigma_1 allowed -0.07, and every step was
rejected), and a merit that penalised the start's own 5e-5 excursion at nu =
100 (now a deadband of the admissibility margin, as pyGRANSO's tolerance does).

## 7.1 The finding the prototype produced

At the frozen K+B+C (and KSP+B+C, and the ne=100 K+B+C) the optimiser has
parked **two compensator poles on the margin**: -0.05000 and -0.05004, 0.04
apart in imaginary part, in a non-normal 8 x 8. Their real parts split like a
square root under perturbation (measured: one pole -0.050 -> -0.024 for a
step of 5e-4 |z|, against a linearised prediction of -0.050). Every
formulation of the compensator constraint fails at that point: a single
max-Re row by nonsmoothness, per-pole rows by the splitting, the LMI by
conditioning (cond P = 1e7, Clarabel NumericalError). This is Burke-Lewis-
Overton's situation -- an abscissa optimum at a near-multiple eigenvalue -- on
the compensator constraint, and it is at least as responsible for the code-6
stops on those rows as the sigma_1 = sigma_2 crossing. pyGRANSO hit the same
wall; the exact SQP shows it because it honours every row.

Consequence: the LMI's job is to keep the search from ARRIVING at such points
(it does: from a healthy start the poles stay strictly inside, -0.05001 and
-0.05003), not to rescue one. A design already there needs a start elsewhere.

## 7.2 The number

ne=100 K+B+C from the baseline design, sigma_2 enrolled, ten inner iterations
per round, identical certificate and exchange:

| | inner iterations | evaluations | sigma_max on Omega | certified, 1 round | certified, 4 rounds |
|---|---|---|---|---|---|
| pyGRANSO | 10 | 22 / round | 9.534 | **9.790** accepted | **9.375** |
| SQP (block + LMI) | 10 | 23 / round | **9.437** | 11.49 REFUSED (peak in a gap) | 9.684 |

The SQP finds the better restricted optimum at equal cost and the certificate
refuses it: it exploits the gaps between the sampled frequencies harder,
because it solves the restricted problem better. pyGRANSO's less-converged
iterate is safer by accident. Over four rounds pyGRANSO wins, 9.375 to 9.684.

So "let the number decide" decides for pyGRANSO **as long as the exchange adds
one frequency per round** -- the same conclusion section 7 reached about
opt_tol from the other side. The SQP is the better inner solver and the worse
partner for this outer loop; its value is unlocked by the exchange work (note
items 3-5: enrichment, neighbour seeding, retirement) or by a trust region
tied to the surrogate's validity, and it should be re-measured there, not
tuned here. It stays in the tree, opt-in, with the LMI as the compensator
formulation for the cavity.

Not done from item 8: IPOPT (no pip wheel; conda on the cluster), the
restart-overhead measurement, the bundle Plan B. QPALM installs and is
available as a second QP backend for pyGRANSO beside DAQP; not measured.

# 8. The SQP made complete (2026-09-19, the user's two questions)

Q1 (the 2x2 Clarke block rather than the diagonal) and Q2 (the stability
constraint in the gradient for EVERY near-line eigenvalue) are both closed in
the SQP, and only there:

- Q1: the block as a second-order cone (section 7; FD-validated). pyGRANSO
  has two scalar rows (diagonal Y) and no place for the cross terms.
- Q2: `reduced.near_line_pairs` -- every residual-filtered Ritz pair of the
  reduced closed-loop pencil with Re >= -eps - reach gets its own row with its
  eigenvector gradient (no factorisation); the full certificate (rightmost
  only) is the fallback. Compensator: per-pole rows or the Lyapunov LMI.
  pyGRANSO keeps the frozen single rightmost row and the frozen max-Re
  compensator row.

Three consistency defects found by running the SQP on the reduced oracle:
1. the K+B design's compensator sits at -0.034 -- inside the GATE's line
   (Re <= 0, the decision) but outside the frozen inner's -eps; hard rows at
   -eps were PrimalInfeasible from the start. The SQP's compensator rows and
   its LMI now certify the gate's line (allowance ctrl_max, plus the
   admissibility cushion), and the abscissa/compensator rows are elastic
   (S-l1-QP), so a violated start is walked in rather than refused. With
   the compensator line at Re <= 0 its poles may sit closer to the axis than
   the frozen suite's (-0.034 -> -0.023 seen); that is the decision as stated
   and one constant (`ctrl_max`) to change if a margin is wanted.
2. the trust radius and BFGS matrix restarted every round (item 8's
   "restart overhead", now measurable): `--sqp-carry` carries both.
3. the reduced pencil's abscissa is a predictor that goes stale after a
   large step: a design read as admissible certified at -0.0485 and was
   refused at the gate twice. The SQP now validates its returned iterate
   with the full certificate (one per candidate, best first, ~14 LUs) before
   handing it over -- the stage-1 rule, applied to the SQP.

| K+B+C from K+B, ne=100, reduced oracle, adaptive budget, 8 rounds | evals | certified gamma |
|---|---|---|
| pyGRANSO (diagonal rows, rightmost only, frozen compensator row) | 248 | **9.894** |
| SQP (block, per-eigenvalue rows, LMI, carry, certified validation) | 151 | 10.234 |

Correct on both questions, cheaper per round, and 3 % worse at equal rounds:
its larger steps (inner t 8.5-9.1 in rounds 5-7) certify at 10.3-12.3 -- the
exchange's over-exploitation, unchanged by the surrogate being accurate. At
equal EVALUATIONS the SQP has ~14 rounds; that comparison is the fair one and
is still to run.

## 8.1 At equal evaluations (SUPERSEDED as a solver comparison: the SQP was on the Re <= 0 line, pyGRANSO on -eps; see TRACKER G)

| K+B+C from K+B, ne=100, reduced oracle, adaptive budget | rounds | evals | certified gamma |
|---|---|---|---|
| pyGRANSO | 8 | 248 | 9.894 |
| **SQP** (block, per-eigenvalue rows, LMI, carry, certified validation) | 14 | 221 | **9.848** |

A tie inside the path noise. The SQP's inner t is 8.7-9.0 in every late round
while the certified value stays 10.0-10.7: the gap is the surrogate's (one
frequency per round), not the solver's, and the adaptive budget reads it and
keeps the SQP at five iterations a round -- which is why fourteen rounds cost
221 evaluations. **Decision: the solver that is right on both questions and
costs the same is the inner solver from here** (multistart driver default;
`--inner sqp --sqp-carry --sqp-trust 0.01`); the code's default stays frozen
until the cluster regression. The exchange is now the bottleneck, and the
next item after Stage 2.

## 8.2 Exchange refinement inside the SQP (`--sqp-refine`): built, measured, opt-in

On the reduced oracle, after each accepted step, find the peaks of the reduced
sigma_1 the working set misses and add them mid-solve (the A-N rebuild on the
reduced model); the exchange's Omega absorbs them; candidates are re-evaluated
on the final set before the best is chosen; the full certificate validates the
returned iterate. Peak finder: the same level-set certificate the outer loop
uses, on the reduced matrices (~0.6 s at dim 20; the 1201-point sweep with
golden-section refinement missed a 0.02-wide peak the model held to 1e-15).

Three loop defects found on the way, all fixed and all independent of the
refinement: (i) the abscissa rows enforced alpha + eps <= 0 while
admissibility wanted <= -mg, so every iterate sat on the line and was
rejected (rows, merit and admissibility now share one line); (ii) the frozen
convergence rule declared the exchange converged whenever the certificate
agreed with t and added nothing -- under an iteration budget it now also
requires the inner solve to be stationary; (iii) frozen path re-verified
bit-identical after each.

What the refinement can and cannot do, measured (K+B+C from K+B, ne=100):

| configuration | rounds | evals | certified gamma | certified / t per round |
|---|---|---|---|---|
| SQP, no refinement | 14 | 221 | **9.848** | 1.10-1.25 |
| SQP, refinement (sweep) | 5 | ~360 | 10.32 | 1.06-1.28 |
| SQP, refinement (reduced certificate) | 3 | 306 | 10.15 | 1.00-1.01 |

Honest rounds, slower progress. The limit is structural, not a defect: the
full certificate found the round's peak at w = +2.3355 (10.3165) where the
reduced model, anchored by the sweep, certified 10.269 at w = -0.59 -- it did
not contain the peak until the ABMV rule anchored it afterwards. **A narrow
peak the basis does not span is invisible to any search on the reduced
model; the per-round full certificate is the discovery mechanism, and the
lever is many cheap rounds** (the reduced oracle: ~330 factorisations a
round; the adaptive budget: short solves when the certificate disagrees).
Refinement stays opt-in. The pilot resumes on the SQP without it.

## 8.3 SQP vs pyGRANSO on the corrected loop (2026-09-22; TRACKER K.5)

Identical start (frozen K+B+C ne=200), reduced model, flags, tolerances,
working-set fixes, no refinement:
| | round 1 | round 2 | latest | KKT residual | full LU / round |
|---|---|---|---|---|---|
| SQP | 7.955184 | 7.951836 | 7.950855 (5 rounds) | 2.0e-4, stationary | ~150 |
| pyGRANSO | 7.962768 | 7.961807 | 7.961104 (4 rounds) | 3.4e-2, not stationary | ~150 |
pyGRANSO is stuck at the compensator DOUBLE POLE on the margin (-0.0500 /
-0.05001): its single row max Re(pole) <= -eps is not Lipschitz there
(Burke-Overton); every sigma-improving step splits the pair. The SQP's 2x2
block Hermite constraint is smooth there. The SQP is better on this case for a
structural reason, at equal cost per round. pyGRANSO's constraints are scalar
functions with one gradient each; it uses the Clarke set implicitly (BFGS;
gradient-history stationarity test), not in its step; a (trace, det) smooth
reformulation of a 2x2 block is possible but degenerates at exact coalescence.
