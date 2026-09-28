> **ERRATA (2026-09-25) -- read first.** K+B+C values recorded here as "KKT points", "converged"
> or "stationary" (7.95x at ne=200, 7.92x at ne=100) were STALLS: the compensator pair near a double pole
> on the line and quasi-Newton memory carried across rounds (STAGE5 25-28). Best now ne=200 6.438335,
> ne=100 ~6.40. A KKT residual (6e-4 reference, 2e-3 active set) is not an optimality certificate on
> flat optima (STAGE5 28). `--sqp-carry` now carries the trust radius only; the quasi-Newton matrix is
> reset each round (`--sqp-carry-h off`, default).
>
> **ERRATA (2026-09-23) -- read before the text below.**
> - Every abscissa-row GRADIENT in this stage's runs came from `alpha_certified`, which, when
>   the stability line was clear, returned the eigenvalue and eigenvectors nearest the REAL point
>   (alpha, 0) instead of the eigenvalue attaining alpha (STAGE5 5.4). The alpha VALUES are right;
>   the gradients were another mode's whenever the rightmost eigenvalue had large |Im|. Fixed behind
>   `--alpha-argmax on` (frozen default kept for bit-identity; proposed for the cluster regression).
> - "Early stopping regularises against overfitting the sampled frequency set" (opt_tol, ~line 170):
>   the over-exploitation was largely a WORKING-SET defect -- each round enrolled only the top 4
>   local maxima, members sat at fixed frequencies while narrow peaks drifted (STAGE5 9, 11). With
>   near-maximal peaks enrolled at their tops and following rows, the loop takes certified steps
>   (inner t vs certificate within ~5e-6) and converges to KKT points.
> - Certified values here are at rtol 1e-4: each may lie up to ~1e-4 relative below the true maximum.

# Stage 1 -- solver settings and constraint handling

Development at ne=100, sequential, in the assistant's sandbox (1 core). The
cluster is used once, at the stage gate, to confirm at ne=200 on the user's
catalogue. Everything here is reproducible from the records in
`results_ne100_*` and the logs beside them.

# 1. The ne=100 development baseline

Built with `tools/lattice.py`, which implements `case.slurm`'s start protocol
portably: roots multistart over the seed catalogue, children resume from each
parent, no-regress across attempts. Settings: eps = 0.05, `--inner-budget iters
--inner-maxit 20`, four outer rounds, first parent only.

| case | gamma | start |
|---|---|---|
| KSP | 11.098 | best of 5 seeds (hinf_3); LQG seeds never moved (160-219) |
| B | 16.482 | best of 5 seeds (hinf_3) |
| C | 16.358 | best of 5 seeds |
| K | 10.597 | from KSP |
| KSP+B | 10.304 | from B |
| K+B | 10.414 | from K |
| KSP+C | 10.151 | from C |
| K+C | done | from K |
| B+C | 9.662 | from B; close to the frozen ne=200 9.674 |
| KSP+B+C | 9.662 | from B+C -- never improved on its parent |
| K+B+C | 9.984 | from K+B |
| KGC, KGC+B, KGC+C, KGC+B+C | -- | KGC refuses to start (section 4) |

Nesting holds on every checkable pair.

# 2. What the baseline showed: refusals at the gate

Every child that stalled had found a better design and been refused:

| case | kept | best found, refused | gap | refused for |
|---|---|---|---|---|
| KSP+B | 10.304 | 9.917 | 3.8 % | compensator |Im| = 22.3 > 20 |
| KSP+C | 10.151 | 9.687 | 4.6 % | compensator Re = +0.059; |Im| = 22.6 |
| KSP+B+C | 9.662 | 8.910 | 7.8 % | compensator Re = +0.025; |Im| = 22.4 |
| KSP+B+C (ne=200, cluster) | 9.797 | 9.620 | 1.8 % | compensator Re > 0 |

Every refused design was closed-loop feasible with margin. Both compensator
constraints were ALREADY in the inner problem (`run_case_epi.py:370-378`); the
defect was at line 412, where the inner solve's returned "best" was chosen on
closed-loop feasibility alone, so the iterate handed to the gate could be one
the inner problem itself considered infeasible.

# 3. The change (all opt-in, frozen defaults)

| flag | default | effect |
|---|---|---|
| `--select-cons on` | off | the inner solve returns its best iterate that the GATE will accept: closed loop at margin, compensator abscissa within `max(-eps, ctrl_max)`, |Im| under the cap when the cap is on |
| `--ctrl-imcap off` | on | drops the compensator |Im| cap from the inner problem and the gate, AND sets the band from the closed-loop spectrum at each accepted design (`band_from_closedloop`) -- one switch on purpose |
| `--ctrl-stab off` | on | drops compensator stability everywhere (`ctrl_max = 1e9`). Not a candidate; measures the cost of keeping it |
| `--band-recut` | 1.0 | closed-loop eigenvalues with Re >= -recut set the band |

Decisions (user, 2026-09-16): keep compensator stability -- practitioners want
it -- and drop the |Im| cap: controller poles are only a proxy for closed-loop
poles, and what certification needs is the band covering the closed loop.

Two defects in the first version, both caught from logs:
- `band_from_closedloop` was called with names not in scope at the gate and
  never ran (UnboundLocalError, caught and printed). Fixed; every SI/N row was
  rerun with it active. It never needed to widen on these designs.
- The selection first tested the INNER value `al_ctrl <= -eps`, eps stricter
  than the gate's `Re <= 0`, and discarded acceptable iterates: K+B+C, 8 rounds,
  9.367 against the frozen 9.046. Aligned to the gate's own criterion; K+B+C
  then reproduces the frozen 9.984257 to every digit at 4 rounds.

# 4. The A/B

Regression set from FIXED baseline parents, same eps/budget/rounds as the
baseline. Frozen rows are the baseline. Labels: **S** selection fix alone;
**SI** fix + cap off; **N** fix + cap off + stability off.

| case | frozen | SI | N | SI vs frozen | N vs SI | calls frozen / SI |
|---|---|---|---|---|---|---|
| KSP+B+C | 9.662 | 9.617 | **9.211** | -0.5 % | -4.2 % | 2298 / 272 |
| KSP+C | 10.151 | **9.836** | 9.688 | -3.1 % | -1.5 % | 2019 / 855 |
| K+B+C | 9.984 | 9.984 | 10.414 | 0 (identical) | +4.3 % | 155 / 155 |
| K+B+C, 8 rounds | 9.046 | 9.193 | see below | +1.6 % | | 340 / 345 |
| KSP (seed hinf_3) | 11.098 | 11.098 | 11.391 | 0 (same path) | +2.6 % (noise) | |

S on KSP+B+C: `t = inf` at outer 1 -- with the cap kept and the selection
honest, NO iterate satisfied the cap in 20 iterations; the run stalls cleanly
instead of handing over an infeasible point. 510 calls a round. Not a
candidate; it establishes that the cap was unsatisfiable from these starts.

## 4.1 What survives the noise, and what does not

The noise floor of a four-round comparison is 1-2 % either way (KSP: N worse
than SI despite fewer constraints; K+B+C 8 rounds: frozen and SI equal through
round 5, then 1.6 % apart). Claims at that level are not claims.

Survives:
- The |Im| cap is unsatisfiable from these starts (S) and removing it never
  hurts. On KSP+C it recovers 3.1 %.
- The selection fix is exactly neutral where the gate never refuses (K+B+C:
  identical to every digit) and is what makes SI possible where it does.
- The call count: where the frozen inner solve burned line searches on
  unreturnable iterates, SI costs 2-8x fewer oracle calls per round (KSP+B+C
  2298 -> 272). Where it did not, nothing changes (K+B+C 155 / 155). The
  "5x faster" is real but case-dependent.
- The compensator-stability constraint costs 4.2 % on KSP+B+C (SI 9.617 ->
  N 9.211, converged at round 1 and stable thereafter).

Does not survive: a single number for the cost of the stability constraint.
On K+B+C at four rounds, N is WORSE than SI by 4.3 %; the eight-round N run
(section 4.2) says whether that is path luck.

## 4.2 The cost of the compensator-stability constraint, corrected

The N column above was measured at `maxit 20`, and section 7 shows that budget
is binding: SI on KSP+B+C reaches 8.9997 at `maxit 100` against 9.617 at
`maxit 20`. So the four-round, 20-iteration N-vs-SI gaps mix the constraint's
effect with the budget's. Budget-matched at `maxit 100`, KSP+B+C from B+C:

| config | gamma | calls |
|---|---|---|
| SI (stability kept) | **8.9997** | 904 |
| N (stability dropped) | 9.0587 | 1130 |

At an adequate budget the constraint costs NOTHING on the case where it
appeared to cost most. On K+B+C at eight rounds (`maxit 20`) dropping it was
3 % WORSE (SI 9.193, N 9.466). For the record, as requested: keeping compensator
stability is the right default, and within the budgets measured here it has no
demonstrated cost -- the 4.4 % reported earlier was a 20-iteration artefact and
is withdrawn.

# 5. Recommendation

SI as the new default: `--select-cons on --ctrl-imcap off`, stability kept,
`--cont-anchor defer`, `opt_tol` unchanged at 1e-10. The stability constraint
has no demonstrated cost at an adequate inner budget (section 4.2). Confirm at ne=200 on the cluster on the
three stage-1 cases from the frozen parents, one job.

# 6. Item 3 -- KGC start: closed

The continuation gate (`--cont`, default on) postdates the frozen campaign: no
frozen record carries it and `case.slurm` never passed it. The frozen KGC
record's round 0 shows `gamma_best = 882.88` -- it started from the open loop,
which the gate now refuses as an infeasible anchor. The chain argument needs a
feasible ANCHOR, not a feasible start. `--cont-anchor defer` (default
`strict`, unchanged) runs without the gate until the first feasible accepted
design, anchors there, and enforces the chain from then on; the final winding
certificate is unconditional either way.

KGC at ne=100 from the open loop: 901.6 -> 11.684 in four rounds, anchor set at
round 0 (alpha -0.091), certified. `strict` still refuses. The lattice driver no
longer counts a refused attempt as completed. Baseline rows added: KGC 12.801,
KGC+B 12.801, KGC+C 12.260, KGC+B+C 12.250 (frozen configuration + defer).

# 7. Item 2 -- opt_tol: answered, recommendation withdrawn

Premise: inner solves end on code 6 because `opt_tol = 1e-10` is below the
oracle's precision. Under SI the premise is gone -- code 6 was the unsatisfiable
|Im| cap breaking the line search. With the cap off, solves end on code 4
(budget) at `maxit 20` and on code 0 (converged) when given room.

Sweep: SI, KSP+B+C from B+C, `maxit 100`, four rounds.

| opt_tol | gamma | calls | codes | |
|---|---|---|---|---|
| 1e-10 (frozen) | **8.9997** | 904 | 4 4 6 6 | never converges; non-stationary iterate best in the TRUE norm |
| 1e-6 | 9.6618 | 454 | 4 0 0 0 | converges each round; every restricted optimum has a peak outside Omega (certified 12.3 vs t 9.1); no progress on the parent |
| 1e-4 | 9.2525 | 391 | 0 0 0 0 | one good round, then stall |

The tight tolerance is not a precision setting: it is EARLY STOPPING, and early
stopping regularises against overfitting the sampled frequency set. Converging
the Omega-restricted problem exactly yields a design that exploits the gaps
between samples; the exchange adds one frequency, and the next optimum moves
its peak to a new gap. Codes 6 in rounds 3-4 of the 1e-10 row confirm a noise
floor exists; reaching it is harmless.

**Caveat from the ne=200 gate (section 11):** all three ne=200 rows end EVERY
round on code 6 even under SI at `maxit 100`, with balance residuals of
1e3-1e4. At ne=100 code 6 was the cap; at ne=200 the Wolfe bracket failure is
real and the inner solve is not reaching stationarity. The early-stopping
argument above still says a tighter tolerance is harmless, but the
oracle-precision question is open at scale and belongs with note item 0 and
stage 5.

`opt_tol = 1e-10` stays. The lever is the OUTER loop -- more than one frequency
per round, or a trust region on warm starts (the log shows `warm start -> trust
radius 0`) -- an exchange-design item, carried to the list below rather than
tried here.

# 8. Open, carried
- Quadprog UnboundLocalError (STAGE0.md section 14). Not seen at eps = 0.05.
- Analytic tail bound for the peak certificate, so the band is derived rather
  than assumed: stage 5.
- Parallelism tests: cluster only, by nature. Kept on a separate list.


# 9. Stage-1 gate, first attempt (job 16139009, ne=200): not passed, three causes

| case | frozen | stage-1 | refusals | start |
|---|---|---|---|---|
| K+B+C | 7.9641 | 9.5911 | 1 continuation | 9.8319 (parent K+B) |
| KSP+B+C | 8.2465 | **13.3306** | 1 continuation | **935.9** -- not the parent |
| KGC+B | 11.6009 | 12.2443 | 2 on-the-line | 12.2443 (parent KGC), no progress |

All three certified (dense, winding, peak at rtol 1e-6) and feasible at
eps=0.05. Zero compensator refusals -- the |Im| cap is gone and nothing was
refused for a compensator pole. But zero refusals overall was the claim, and
three occurred, of two kinds neither of which was seen at ne=100:

**1. KSP+B+C did not start from its parent** (frozen code). Resuming a
controller case from a controller-free parent, `run_case_epi` borrows a
controller from the lowest-gamma record on file -- here K+FB+FC (3.93, tuned
for FREE fields) -- and puts it on B+C's fixed fields: start 935.9. The parent's
own design (K = 0, 9.67) was never a candidate, so the "cannot finish above its
parent" guarantee was void before the first iteration. At ne=100 there was
nothing on file to borrow, which is why it worked there. The suite survived
this by running every parent and keeping the best; the gate ran only the first
(alphabetically B+C -- the bad one).

Fixed: both starts are evaluated and the lower is kept. Reproduced in the
sandbox on the frozen records (935.927917) and rescued (9.674473, the parent's
value to the digit).

**2. KGC+B refused for designs ON the margin line** -- `eigenvalue at
-0.05+5.89j on Re=-0.05`. With `select_cons` the returned iterate has the
stability constraint exactly active; the certificate's line sweep then finds an
eigenvalue on the line and refuses. The frozen selection sat at -0.051 by
accident; this one at -0.0500 by construction.

Fixed: `--sel-margin` (default 1e-3 of eps): the admitted iterate must be
inside the line by that much. Only under `--select-cons on`.

**3. The continuation gate refused the larger steps `maxit 100` produces** --
"split depth exhausted with motion still above the allowance", K+B+C and
KSP+B+C. The suite's clock budget gave five-iteration inner solves and small
steps. `--cont_depth 8` (256 sub-steps, from 16) is passed by the gate job.
This is the trust-region item of section 7 showing up from the other side; a
proper fix belongs with the exchange-design work.

The gate job now runs every parent (as the suite), reports refusals by kind,
and prints each attempt's start gamma. Default path re-verified bit-identical
after the changes (B, ne=100: 10.23128516441147).


# 10. Stage-1 gate, second attempt (job 16152993): the inner solve ignores
#     the certificate's verdict

Starts were right this time (KSP+B+C from B+C at 9.674473, the parent). But
that attempt made no progress in eight rounds, and its log says why:

    outer 0: t=7.898471  certified=29.395037   best 9.674473
    outer 2: t=8.137084  certified=9.502619  ->  gate REJECTS: eigenvalue at
             0.754294-1.6779j on Re=-0.05  (certificate alpha +0.7543)

The inner solve returned iterates it believed feasible -- t down to 5.46 --
and the certificate found the closed loop UNSTABLE, alpha +0.71 to +0.75.

`alpha_certified` (fe_codesign.py:309) returns a VALUE, `a_seen` -- the
abscissa of the eigenvalues its line sweep saw -- and a VERDICT, `clear` --
whether the line Re = -eps was proven free. The gate uses the verdict
(`_clr_c`, line 1714). The inner solve, line 341, takes the value and discards
the verdict. An eigenvalue that has crossed far to the right is not near the
line, so it is not "seen"; the value reads feasible; the iterate is admitted.
Small steps kept this from mattering; `maxit 100` at ne=200 does not.

Fixed in the selection: under `--select-cons on` an iterate is admissible only
if the certificate cleared it (`bool(_clr)`), in addition to the value tests.
Verified: same case, parent and flags at ne=200, two rounds, no refusals,
9.674 -> 9.041. The ne=100 SI row moves from 9.617 to 9.607 -- some admitted
iterates there had an uncleared verdict too.

Also from this job: the "rc=0, no summary line" on KGC+B's donor-K+B attempt
was the no-regress guard keeping the earlier 12.23 over the donor's 12.91
("keeping previous better result"); the driver now labels it as such.

The gate job is unchanged; run it again.

# 11. Stage-1 gate, second attempt (job 16152993) -- complete

Code: before the verdict fix of section 10 (that tarball had not been
launched). Every parent and donor, maxit 100, 8 rounds, cont_depth 8.

**Seventeen attempts, zero compensator refusals.** The stage-1 claim holds at
ne=200. The only refusals -- on-the-line (4+4+1) and continuation (1+1) -- are
on the two controller-free-parent starts and one donor, exactly the mechanism
section 10 fixed.

| case | frozen | stage-1 | | best attempt from | calls | codes |
|---|---|---|---|---|---|---|
| KSP+B+C | 8.2465 | **7.9515** | -3.6 % | donor K+B+C (7.964) | 858 | 6 6 6 6 |
| K+B+C | 7.9641 | 8.2448 | +3.5 % | parent KSP+B+C, FROZEN 8.2465 | 2785 | 6 6 6 6 |
| KGC+B | 11.6009 | 12.2006 | +5.2 % | parent KGC | 1150 | 6x7, 0 |

All three certified: dense, winding (count 0), peak at rtol 1e-6 (ratio 1.000).

Three things this exposed, the first two fixed before the next launch:

1. **Parents from the job's own output first.** K+B+C's best attempt started
   from the FROZEN KSP+B+C (8.2465) while the same job had just produced one
   at 7.9515 -- the stage-1 K+B+C sits above its own sibling, a nesting
   violation the suite's protocol would not produce. `lattice.py` now searches
   `--out` before `--parent-dir`; the gate job prints the nesting check.
2. **Degenerate donor starts refused in seconds.** Three donor attempts began
   at 882.88 (open loop: a controller-carrying donor into a static-gain case
   loses the design) and ran their full budget, ~25 min for nothing.
   `run_case_epi` now refuses a resumed start more than 10x its source
   record's gamma. Verified: KGC+B from KSP+B refused (90x); KSP+B+C from KSP+B
   proceeds.
3. **Code 6 at ne=200 is real** (section 7 caveat). Not fixed; recorded.

The reading of the stage-1 change stands: the constraint work did what it was
meant to, and the remaining gaps to frozen on K+B+C and KGC+B are
start/budget/nesting effects of the gate protocol, two of which are now
addressed. The next launch is the first with sections 10 and 11 in.

# 12. Stage-1 gate, third attempt (job 16205236): passed. Defaults flipped.

Code: verdict, margin, degenerate-start guard and own-output-first parents
all in. Every parent and donor, maxit 100, 8 rounds, cont_depth 8.

| case | frozen | stage-1 | | best attempt from |
|---|---|---|---|---|
| KSP+B+C | 8.2465 | **7.9515** | -3.6 % | donor K+B+C |
| K+B+C | 7.9641 | **7.9520** | -0.2 % | the job's own KSP+B+C (nesting: equal, as it must) |
| KGC+B | 11.6009 | 12.2310 | +5.4 % | parent KGC; 8 rounds vs the frozen 28 |

All three certified (dense; winding, count 0; peak at rtol 1e-6). Seventeen
attempts, **zero compensator refusals**. The two degenerate donor starts were
refused in seconds by the new guard.

The "nesting violation" the job printed -- K+B+C 7.9520 > KSP+B+C 7.9515 -- is
the same design twice: K+B+C never improved on the KSP+B+C design it started
from, and its transplant re-certified the peak 5.7e-5 higher ("refined
+5.7e-05, within prior rtol"). The check's tolerance was 1e-9, stricter than
the certificate that produced the numbers; it is now 2e-5.

## 12.1 The refusals that remained, explained by the certificate's own contract

The B+C -> KSP+B+C and K+C -> K+B+C starts again produced four on-the-line
and one continuation refusal each, and the identical bad iterate
(0.754294-1.6779j, alpha +0.7543) was admitted despite the verdict fix of
section 10. The reason is in `certify_sparse.py:433`: the 1-D sweep proves
**no eigenvalue ON the line** and returns the rightmost eigenvalue it
*encountered*; an eigenvalue already across the line is not on it, so both the
inner call and the gate call can report "clear" -- which one happens to
encounter it depends on the shift sequence. Stability follows only by
CONTINUATION, and the docstring names what keeps that premise true: small
steps, "supported by the trust region bounding ||dtheta||".

The trust region was removed on purpose in the frozen design (`--trust_warm`
help: 20x the calls for no gain) and its safety role given to the continuation
gate and the tau governor -- both measured under the CLOCK budget, where an
inner solve ran five to fifteen iterations. That short inner solve was the
step control. `maxit 100` removed it without replacing it.

So the machinery behaved as designed: the continuation gate and the line test
refused steps that had carried an eigenvalue across unseen. From a poor start
(a controller-free parent, default controller far from anything good) a
100-iteration inner solve spends its whole budget on such a step, four times
over. From good parents the steps stay small and there are no refusals, which
is where every design result above came from.

**Measured, same start (KSP+B+C from B+C, ne=200):**

| inner maxit | rounds | refusals | result |
|---|---|---|---|
| 100 (job) | 8 | 4 on-the-line + 1 continuation | none: stuck at 9.674 |
| 25 (sandbox) | 3 | **0** | 9.674 -> 8.958, 241 calls |

The verdict fix of section 10 stays: it is correct, and it costs nothing; it
just cannot see what is not on the line. The proper fix -- permanent rows for
tracked near-boundary eigenvalues -- is note item 2 and is carried with this
evidence attached.

## 12.2 Defaults, as of 2026-09-17

| flag | was | now |
|---|---|---|
| `--select-cons` | off | **on** |
| `--ctrl-imcap` | on | **off** (band from the closed loop) |
| `--cont-anchor` | strict | **defer** |
| `--inner-maxit` | 200 | **25** (the certificate's small-step regime) |
| `--ctrl-stab`, `--opt-tol`, `--inner-budget` | on, 1e-10, clock | unchanged |

The frozen configuration remains one command line:
`--select-cons off --ctrl-imcap on --cont-anchor strict --inner-maxit 200`.
Verified both ways on case B (ne=100): explicit frozen flags reproduce
10.23128516441147; the new defaults give the same value, since on a case the
gate never refuses the selection change is a no-op by construction. The gate
job's MAXIT default is 25.

## 12.3 Stage 1 closed. Carried forward

- Note item 2 (per-eigenvalue stability rows): the refusal mechanism of 12.1.
- Note items 3-5 (exchange enrichment): the one-frequency-per-round effect of
  section 7.
- Note item 0 and the oracle noise floor at ne=200: code 6 on every ne=200
  round, balance residuals 1e3-1e4 (section 7 caveat).
- Analytic tail bound for the peak certificate (stage 5).
- Parallelism: cluster only; separate list.
