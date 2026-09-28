> **ERRATA (2026-09-25) -- read first.** K+B+C values recorded here as "KKT points", "converged"
> or "stationary" (7.95x at ne=200, 7.92x at ne=100) were STALLS: the compensator pair near a double pole
> on the line and quasi-Newton memory carried across rounds (STAGE5 25-28). Best now ne=200 6.438335,
> ne=100 ~6.40. A KKT residual (6e-4 reference, 2e-3 active set) is not an optimality certificate on
> flat optima (STAGE5 28). `--sqp-carry` now carries the trust radius only; the quasi-Newton matrix is
> reset each round (`--sqp-carry-h off`, default).
>
> **ERRATA (2026-09-23) -- read before the text below.**
> - Section 3.1, the 9.748986 "fixed point": an artefact of the admissibility cushion applied to the
>   exactly-computed compensator poles (TRACKER H); without it the SQP moved to 9.7467 and, with the
>   2x2 Hermite block constraint (3.2), onwards.
> - Section 3.6, "certification frequency is set by the reduced model's validity horizon, which one SQP
>   step exceeds": SUPERSEDED. It was measured with the two-sided model (not robust to positions) and the
>   deficient working set; with the position-robust model provisional rounds held 2 of 2, confirmed to
>   1e-6 (STAGE5 5.3). Reduced-model certification is reconsidered in STAGE5 12.
> - Section 3.7, "7.9499 came from short steps": SUPERSEDED -- the SQP's refused steps were model error
>   (two-sided model) and then Omega-membership defects (STAGE5 9, 11), not step length as such.
> - Section 2 (eight-round pilot) is an eight-round record, not optima (correction in that section).
> - Certified values at rtol 1e-4 compare only above ~1e-4 relative.

# Stage 2 -- initialisation: random starts with a stabilisation phase (note item 1)

Run on the reduced oracle (Stage 5), which is why Stage 5 was taken first:
a start costs minutes on one core instead of hours.

# 1. What is built

**`tools/randstart.py`** -- a catalogue of N random controllers in the seed
format (`--seed NAME --seed_file FILE`), carrying also K and the start
positions (`run_case_epi` now reads `__K`, `__xa`, `__xs` from a catalogue
that has them). Two arms:
- `scaled` (note section 5.1 with the reviewer's refinement): omega_* per
  start, log-uniform between the plant's least-damped mode (0.648 at GL) and
  the band half-width (30) -- 1.7 decades, so the arm covers the scale we do
  not know rather than pinning it; K_* = 1 / max_w |c^H R(iw) b| over the
  band (the SISO loop gain the controller sees: 3.685 at w = -0.75, so K_* =
  0.271 -- not the performance channel's 882); A_c tridiagonal with diagonal
  -omega_* |xi| and off-diagonals omega_* xi, redrawn while unstable; n ~
  K_* N(0,1); K ~ 0.1 K_* N(0,1); positions uniform in their ranges.
- `hifoo`: every entry N(0,1), positions uniform.
  At ne=100, 16 draws each: scaled 16/16 controllers stable (by construction),
  hifoo 0/16.

**`stabilise.py`** -- the abscissa-descent phase, HIFOO's convention:
optimisation on the spectral abscissa, stopping at the first stabilising
controller. Objective max(alpha_cl, alpha_ctrl - allowance), both dense at GL
(n + r ~ 200-400; the reduced pencil is not used because its basis is built
from a design and the phase moves the design a lot), eigenvector gradients
through the same D.grads, pyGRANSO unconstrained as the BFGS engine (the
analytic gradient enters through an autograd Function -- set by hand it is
ignored and pyGRANSO reports "converged" at iteration 0, which cost one
run). Stop: alpha <= -eps - delta, delta = 3e-3 (the stage-1 on-the-line
finding makes the cushion mandatory). Budget: `--stab-maxit` (150); a start
not inside by then is NOT STABILISED, written as a marker and counted.
Positions free in the phase.

Six draws at ne=100: the three HIFOO starts (alpha +1.79, +1.69, +1.78)
stabilised in 14-19 evaluations (~5 s), the three scaled in 3-7; all six
inside the margin with the compensator stable.

**`--stabilise`** in `run_case_epi`: phase before the start certification
when the start is outside the margin plus the cushion; the record carries the
phase's alpha before/after, evaluations, and the verdict.

**`tools/multistart.py run`** -- isolated starts (own OUTDIR, --cold from its
catalogue entry, no resume, no donors, no shared checkpoints), resumable per
start. **`report`** -- fraction stabilised, best certified gamma against a
reference, distinct optima clustered by gamma (rtol 1e-3) AND by the
controller transfer function k(i w) on the band (relative 1e-2) AND
positions -- not by (d, e, f), whose gauge freedom would count copies -- and
the rarefaction curve (distinct optima against starts).

First start end to end (hifoo_000, K+B+C, ne=100, reduced oracle, adaptive
budget): phase 17 evaluations, anchor alpha -0.091, exchange 19.99 -> 11.86 in
4 rounds (~4 min); 10.317 in 8 rounds (181 s).

# 2. Pilot (ne=100, K+B+C, 16 starts per arm) -- running

Rows and the report go here.

# 2. Pilot (ne=100, K+B+C, 16 starts per arm) -- EIGHT-ROUND results

**Correction (user, 2026-09-20).** The values below are eight-round iterates,
not optima: every start ran a fixed budget and stopped. "Distinct optima"
overstates it -- read "distinct eight-round endpoints". Two solvers taking
different steps from the same start are at different points of different
paths after eight rounds (the same draws: hifoo_000 10.32 / 11.43, 005 26.10
/ 12.56), which says nothing about where either converges. A second flaw:
the SQP's compensator row sits on the gate's line (Re <= 0) while pyGRANSO's
frozen inner row asks Re <= -eps -- different feasible sets, hence different
optima legitimately; the decision is now Re <= -eps EVERYWHERE, `--ctrl-line eps`
(frozen default unchanged, bit-identical). The study that answers the question runs
each start to CONVERGENCE (`--converge`: stationary inner solve, certificate
agrees, no new frequency; the stall rule off) on both solvers with the same
constraints, and cross-polishes each solver's endpoint with the other: two
correct local solvers may reach different local optima by path in a rugged
landscape, but must not disagree at a point both call stationary. That is
the cluster study; the table below stays as the eight-round record.


Protocol per start: phase -> certified anchor -> 8 rounds on the reduced
oracle with the adaptive inner budget; isolated; 2.5-4.5 min each on one
core. Two defects the pilot itself exposed and that were fixed before the
scaled arm ran (STAGE1B 8.2 and the tracker): a certified design refused for
its homotopy path from a random start (now accepted when the unconditional
winding certificate passes, `--cont-accept-winding`), and the SQP handing back
a design worse than its start when the reduced abscissa was stale at the
round's start (start-of-round check against the certified value; never worse
than the start). The HIFOO arm ran on pyGRANSO before the SQP became the
inner solver; the scaled arm on the SQP. Both arms will be on the SQP before
any cross-arm conclusion is drawn; the numbers below carry that caveat.

| arm (inner) | stabilised | best gamma | median | quartiles | worst | within 25 % of the warm start (9.98) | distinct optima / starts |
|---|---|---|---|---|---|---|---|
| hifoo (pyGRANSO) | 16/16 | **9.519** | 15.4 | 12.5 / 24.8 | 50.6 | 4 | 16 / 16 |
| scaled (SQP) | 16/16 | 10.821 | 33.5 | 27.7 / 70.6 | 143.8 | 1 | 16 / 16 |

Reading (provisional):
- **Fraction stabilised: 100 % on both arms.** Every draw that needed the
  phase reached the margin within budget (13-38 evaluations for the HIFOO
  draws from alpha ~ +1.5, 2-23 for the scaled ones); seven HIFOO and eight
  scaled draws needed no phase at all.
- **Best gamma.** The best random start (hifoo_010, 9.519) beats the
  warm-started baseline (9.984 at four fixed rounds) and sits between it and
  the adaptive warm start (9.279 at eight). A cold random controller
  landing there is the headline.
- **Distinct optima: every start is its own optimum on both arms**; the
  rarefaction curve is the identity. The landscape is as multimodal as the
  note feared; 16 starts exhaust nothing, and the reviewer's "extend if the
  curve has not flattened" would say extend -- but the better inference is
  the note's own item 12: global optimisation of the position variables on
  the reduced model, not more random starts.
- **Scaled vs hifoo: the scaled draw is WORSE here** (median 33 vs 15, one
  start within 25 % against four). If it survives the HIFOO rerun on the
  SQP, the physics-scaled construction (omega_* log-uniform, K_* from the
  loop gain) gives worse basins than an unscaled N(0,1) draw on this case,
  and the note's hypothesis about it is refuted at ne=100.
- Random starts arrive with tiny working sets (9-16 members from their own
  response) and the exchange then discovers the band one frequency per
  round: the stage-3 bottleneck, at its most visible.

Per-start rows: `tools/multistart.py report` on `ms_hifoo_ne100`,
`ms_sqp_scaled_ne100`; JSON in the record.

# 3. One start to convergence (K+B+C, ne=100, hifoo_000), and what running it found

Protocol: `tools/converge_pair.sh` -- phase, certified anchor, `--converge`
(stationary inner solve + certificate agrees + no new frequency; stall rule
off), Re <= -eps everywhere, reduced oracle, adaptive budget; SQP then
pyGRANSO; run in five-minute chunks with checkpoint resumes (the sandbox
kills detached processes at turn boundaries).

Three defects the chunked run exposed, all in the frozen loop, all fixed:
1. **The checkpoint did not store the working set.** A resume rebuilt Omega
   from the design's own peaks and lost the rounds spent discovering the band
   (87 -> 76 members, gamma 8.19 -> 8.43). The checkpoint now carries
   `active_freqs` and a checkpoint resume restores it (a frozen --resume from
   a source record keeps the frozen construction; bit-identity kept).
2. **The band was widened one candidate late.** The gate certified a
   candidate over the current band and widened the band from the candidate's
   closed-loop spectrum only after accepting it; a design whose loop puts a
   near-axis eigenvalue outside the current band was certified over too
   narrow a band and accepted with a too-low gamma. `--band-first` widens
   from the candidate before its certificate; frozen order kept as default.
3. **The band was not persisted** in the checkpoint; a resume restarted at
   +-30. Now stored and restored.

4. **The stabilisation phase re-fired on every resume** -- the actual cause
   of the discrepancy. Its trigger used the phase's TARGET (inside the margin
   by the cushion delta) instead of the CONSTRAINT: a feasible checkpointed
   design with a compensator pole at, say, -0.052 was "re-stabilised", moved,
   re-certified worse (9.93 -> 11.09), and the run then overwrote its better
   checkpoint with the regressed one. The phase now fires only on
   infeasibility (alpha + eps > 0 or a compensator pole beyond -eps) and
   never on a checkpoint resume; the cushion governs only where it stops.

Observed before the fixes (now attributed to 4): a checkpoint certified at
7.994 was re-certified at 13.32 on resume over the printed band [-30, 30]
(the certificate itself is sound: a later checkpoint, 8.2899, agrees with a
2401-point sweep, a local refinement and the certificate at any gam_lb to
1e-4). The design was overwritten before it could be examined. The
prospective test: with the three fixes, every resume's start certification
must reproduce the checkpoint's gamma; the pair is restarted under them.

Interim numbers (under the frozen band order, to be superseded): the SQP arm
reached certified 7.994 and later 8.29 from a random unstable controller --
below every warm-started value at this mesh (8.95 adaptive, 9.41 fixed).

## 3.1 Result of the one-start convergence test (under all four fixes)

Resume consistency verified on every resume (checkpoint gamma reproduced to
the last digit, working set restored). The SQP arm: 19.24 -> 10.72 -> 10.29
-> 10.18 -> 9.86 -> 9.755 -> **9.748986**, a FIXED POINT: certificate = t, no
new frequency, no admissible step for three rounds (`--converge` now
terminates there and labels it `exchange_fixed_point`, not converged). From
a random unstable controller, below the warm-started 9.98 (4 fixed rounds)
and near the adaptive warm start's 9.28.

The endpoint (stationarity.py): a SIX-way tie -- six peaks within
9.7455-9.7490 at w = -1.49, -1.08, -0.83, -0.53, +0.16, +1.58 -- and both
compensator poles at -0.0501, a near-double pole on the margin (the wall of
STAGE1B 7.1, reached from a random start). rho_Clarke on the sigma hull =
1.56 = 3.4e-2 x the branch-gradient scale: not stationary on the hull alone,
but the compensator constraint is active and the test has no normal cone for
it, so this is not a KKT verdict.

**Cross-polish:** pyGRANSO started from the SQP's endpoint reproduces
9.748986 and makes no progress (104, 45, 45 evaluations, nothing admissible).
The two solvers agree on the point. So the answer to "the optimum should not
depend on the solver" on this case: both correct first-order solvers stop at
the same point; what neither can say is whether the point is a KKT point of
the nonsmooth problem or a stall at a high-multiplicity kink with a
double-pole constraint -- that needs a method with a subgradient model
(bundle / gradient sampling, item 8's Plan B) or the Newton polish of the
balance system with all six peaks and the constraint active (item 9), which
now has a concrete target.

## 3.2 The compensator in Burke-Overton's coordinates (2026-09-20, TRACKER H)

`hurwitz.py`: Hermite's criterion -- the Bezoutian of Re and Im of P(i w) is
definite iff P is Hurwitz -- verified exactly (0 failures in 400 random
complex spectra). On the WHOLE compensator it is numerically useless: poles
spanning 0.05-116 make a degree-16 polynomial whose Hermite matrix has
lambda_min/tr ~ 1e-17 whatever eps is. The construction that works is the
eigenvalue analogue of the sigma block: compress A_c onto the invariant
subspace of the two rightmost poles with biorthogonal bases (a smooth 2x2
block, the pair being far from the rest of the spectrum) and apply Hermite to
the block's shifted polynomial (degree 4, no spread). At the endpoint:
positive definite with the poles inside (eps <= 0.0500), negative just beyond
(0.0501); the first-order formula for the cluster's symmetric functions
matches a finite difference to four digits. In the SQP (`--sqp-ctrl block`):
a 4x4 PSD cone with an elastic slack, gradients by central differences on the
r x r compensator with the bases held fixed, the cluster's poles removed from
the per-pole rows. **Cavity cost: none that scales with n** -- everything is
on the r x r compensator.

Two rounding tolerances (1e-8) were needed on the exact compensator line --
admissibility, merit, and the gate (`certify_sparse`: "pole Re = -0.05 >
allowance -0.05" refused a design certified at 9.7436) -- because the
optimum sits on that line by design. Frozen path bit-identical throughout.

Effect: from the double-pole endpoint, where per-pole rows and the LMI could
not take a step, the SQP now walks the corner: certified 9.7473 -> 9.7406 ->
9.7380, poles held on the margin, ~1e-3 per round at 100-190 evaluations.

`tools/kkt_residual.py` (constraint-aware stationarity, Clarabel SOCP + PSD):
at the current endpoint rho_KKT = 1.65e-2 x the gradient scale with the
block's PSD multiplier (2.87e-2 on the sigma hull alone), 18 peaks within
2e-3 of the maximum, hull weight concentrated on one. Not a KKT point (the
stationary kink of frozen KSP+B+C reads 6e-4): the corner is walkable and
being walked, slowly -- item 9's Newton solve of the KKT system on the active
set is what finishes it. Gradient-sampling certificate: not yet run.

## 3.3 Gradient-sampling certificate at the endpoint (`tools/gradsample.py`)

On the reduced oracle (one reduced evaluation per sample plus the r x r
block's finite differences: independent of n), 96-128 samples per radius:

| radius (x |z|) | nu = 0 (objective only) | nu = 100 (penalty) | samples violating the block |
|---|---|---|---|
| 1e-3 | -- | 5.5e-7 x scale (phi spread 0.3: penalty-dominated) | 125/128 |
| 1e-5 | rho_GS = 0.92 x scale | 6.7e-7 x scale | 96/96 |
| 1e-6 | 1.00 x scale | 9.0e-7 x scale | 96/96 |

Reading. Objective only: the sampled gradients all point one way -- an
unconstrained descent direction exists, as it must where a constraint is
active. With the penalty: EVERY sample at EVERY radius is infeasible -- any
perturbation of a double pole on the margin splits it and pushes one pole
out. **The feasible set at this point is a cusp** (Burke-Overton's geometry
of a nonderogatory multiple eigenvalue on the boundary), which is exactly why
the SQP crawls: the only feasible directions keep the pair coalesced.
Gradient sampling with a penalty is dominated by the cusp and is not a usable
certificate here; the constraint-aware KKT residual (3.2, 1.65e-2) is, and it
says "not yet".

Consequence for item 9 (the Newton polish): the endpoint's structure is a
MANIFOLD -- the pair coalesced (discriminant of the 2x2 block = 0) and on the
margin (lambda_min of its Hermite matrix = 0) -- together with the active
peaks equalised. The polish is a Newton method on that balance system (active
peaks equal, the two block conditions, stationarity through the multipliers),
which moves along the cusp instead of across it. That is the design; it is
not written.

## 3.4 Newton polish on the manifold (item 9), first attempt (`tools/newton_polish.py`)

Unknowns z, t, the active-peak multipliers from the KKT residual (7 peaks
with weight > 1e-3), and three multipliers for the compensator pair written
through smooth symmetric functions of its two rightmost eigenvalues
(coalesced: discriminant = 0; on the margin: Re of the sum = 0). Square
system; residual on the reduced oracle; Levenberg-Marquardt.

Result: the seven peaks equalise (reduced max 9.735 -> 9.635, spread 4e-4)
while a peak NOT in the active set rises -- full certificate **10.119** at
w = -1.435 against 9.738 at the start -- one compensator pole leaves the
margin (-0.0492), a multiplier goes negative, and LM stops at |F| = 0.1.
An unsafeguarded equality solve at an active-set corner: the guessed active
set is not the optimum's, and nothing holds the inactive peaks or the
inequality signs. The safeguards it needs (step control on the true merit,
monitoring of every working-set peak, lambda >= 0) are the SQP with the
block cone, i.e. the method already running. Conclusion for item 9 on this
case: not a shortcut past the cusp; the crawl is the price, and it is cheap
per round on the reduced oracle (100-190 reduced evaluations + one
certification of ~330 factorisations).

## 3.5 Bookkeeping of the K+B+C programme (2026-09-20, late)

- A warm start that violates Re <= -eps needs the stabilisation phase like a
  random start (K+B's compensator pole at -0.034: without the phase the SQP
  returns nothing admissible, t = inf every round, the gate refusing the
  pole). The phase brings it inside at a price: start 10.41 -> 12.19.
- Two invalid runs were discarded (a phase-less cert-sweep pair; a script
  the tool cut before writing). Lesson kept: `pkill -f` with a pattern that
  matches one's own command line kills the shell.

## 3.6 Certification frequency (`--cert-every k`) -- measured, negative

Provisional rounds take the reduced model's own level-set certificate as level
and witness, anchor the witness (one factorisation), and accept
provisionally; a coarse full spot check (16 grid points + the reduced peaks,
~20 factorisations) escalates to a full certificate when the truth exceeds
the claim; at full rounds the provisional best is certified first, then the
candidate, with rollback to the last certified best if the truth disagrees.

| K+B+C from K+B (phase), 12 rounds | gamma | factorisations | provisional kept / escalated |
|---|---|---|---|
| certify every round | **8.885** | **2,484** | -- |
| every 4th round, provisional otherwise | 9.290 | 3,255 | 1 / 7 |

Less frequent certification costs MORE and ends WORSE: the reduced model's
claim fell 5-15 % below the truth in seven of nine provisional rounds (8.94
vs 10.40, 8.65 vs 10.02, 8.63 vs 9.64, 8.47 vs 9.37) -- the SQP's step lands
the peaks where the basis does not reach -- and every escalation paid the
spot check, a full certificate and the provisional best's certification.
The one provisional round that held agreed with its later full certificate to
3e-5. Without the spot check the chain claimed 8.55 where the truth was
17.33 (first run).

**Conclusion:** once per round is what this inner solver's step size
requires; the certification frequency is set by the reduced model's validity
horizon, which one SQP step exceeds. The lever is the cost per certificate
(reference sweep as a parameter -- halved already; the analytic tail bound;
parallel shifts on the cluster), not its frequency. `--cert-every` stays,
opt-in, with the spot check, for an inner solver with shorter steps.

## 3.7 The ne=200 terminal polish with the assembled method

From the frozen K+B+C ne=200 design (7.9637): SQP + block constraint +
reduced oracle + adaptive budget + Re <= -eps + coarse-then-fine
certification, four rounds: 7.9620, at 170-274 factorisations per round
(against thousands frozen). The SQP's larger steps were refused by the
certificate every round (t 7.955 certified 8.46; t 7.960 certified 8.93):
the sigma_2 pyGRANSO polish's 7.9499 came from short steps. Defect fixed on
the way: the SQP's relative stationarity test used the largest row gradient
(7e3 at ne=200) and fired after one iteration; it now uses the mean active
row gradient as stationarity.py does.
