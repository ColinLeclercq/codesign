> **ERRATA (2026-09-25) -- read first.** K+B+C values recorded here as "KKT points", "converged"
> or "stationary" (7.95x at ne=200, 7.92x at ne=100) were STALLS: the compensator pair near a double pole
> on the line and quasi-Newton memory carried across rounds (STAGE5 25-28). Best now ne=200 6.438335,
> ne=100 ~6.40. A KKT residual (6e-4 reference, 2e-3 active set) is not an optimality certificate on
> flat optima (STAGE5 28). `--sqp-carry` now carries the trust radius only; the quasi-Newton matrix is
> reset each round (`--sqp-carry-h off`, default).
>
> **ERRATA (2026-09-23) -- read before the text below.**
> - The "surrogate overfitting between rounds" (sections 3-8) and "|Omega| <~ 50 is incompatible with
>   once-per-round certification" (section 6) were measured with the two-sided reduced model and the
>   deficient working set (top-4 enrolment, fixed-frequency members). Their mechanism was largely those
>   defects (STAGE5 9, 11). Re-measured on the corrected loop (section 9): items 4-5 bring no gain
>   (same cost, slightly worse stationary point) and add a gap risk -- recommendation OFF in the nominal
>   algorithm, which stands on the new evidence rather than on the old mechanism.
> - Section 7 (adaptive inner budget beats the frozen policy at equal cost, ne=100) is not contradicted.
> - Certified values at rtol 1e-4 compare only above ~1e-4 relative.

# Stage 3 -- working-set policy (note items 4, 5)

Pulled forward from the handoff's order because the item-8 number (STAGE1B
section 7.2) says the exchange, not the inner solver, is the limiting part:
both the tightly converged inner solve and the exact SQP over-exploit the gaps
between the sampled frequencies, and one added frequency per round does not
close them.

# 1. What the note prescribes

Item 4 (A-N seeding): around each new peak w_p add satellites w_p +- delta0
rho^k, keep those with sigma_1 >= (1 - eps_w) gamma (eps_w in [0.05, 0.5]);
re-seed only the currently active peaks. Item 5 (retirement): retire the
starter grid after the first certified round and any member below the level
for two outer rounds; target |Omega| <~ 50; certification makes it safe -- a
wrongly retired frequency is re-located and re-added. Gate: gamma unchanged to
rtol, outer rounds and factorisation counts fall, 27-case regression.

# 2. Implementation: `--omega-policy an`

At the point where the frozen loop appends the certified peaks:
- satellites at delta0 rho^k (defaults 0.005, 2, six per side) around every
  certified peak of the accepted design that sits within the level -- all of
  them, not only the new ones;
- sigma_1 on Omega at the accepted design; dormancy counters; the starter grid
  retires by ONE round of dormancy once a design has been accepted, other
  members by two; the active and the young never retire; cap `--omega-cap`
  (50) dropping the lowest first; floor `--omega-floor` (30);
- per-round accounting (`omega_log` in the record).

Two rules were first written too literally and corrected from the run: the
grid retired on a round count while only 9 satellites existed, and the inner
solve at |Omega| = 19 collapsed to t = 1.5 (certified 13); and seeding only
the NEW peak left the set without structure. Frozen path unchanged
(B: 10.23128516441147).

# 3. First measurement (ne=100, K+B+C from the baseline K+B, 6 rounds)

| policy | gamma | |Omega| end | design evals | ~oracle gains (evals x |Omega|) |
|---|---|---|---|---|
| frozen | 9.410 | 70 | 253 | ~17,200 |
| an | **9.371** | 50 | 395 | ~23,200 |

gamma 0.4 % better; |Omega| held at the cap; but round 5 spent 175 evaluations
and the run did MORE oracle work overall. The gate's "factorisation counts
fall" is not met on this run. The certified values of rounds 0-3 (20.6, 14.1,
21.2, 29.8 against t ~ 9.7) show the surrogate still overfitting between
rounds: satellites around the ACCEPTED design's peaks do not cover where the
inner solve moves them. That is the half of the A-N recipe the exchange cannot
copy -- they rebuild the set at every trial design from a full norm evaluation
(the Hamiltonian level-set method), the exchange does it once per round.

# 4. What follows from that

The lever that remains is a trust region on the inner step tied to the
surrogate's validity (the frozen design measured one as harmful under the
clock budget, STAGE1.md 12.1 -- under an iteration budget it is a different
question), or an intermediate certification inside the inner solve at a coarse
tolerance (the handoff's "coarse-then-fine certification tolerance"). Both are
Stage-3 items still; neither is done. The policy stays opt-in.

Not done: the LU-weighted accounting per round (the dump has the LU counter;
the table above uses evals x |Omega| as its proxy), the level-ladder reading of
item 4, the ne=200 measurement on the terminal polish, the 27-case regression.

# 5. Second pass: exact accounting, three defects, and what |Omega| <~ 50 costs here

The 35 % was not the policy's bookkeeping (cheap: sigma_1 on Omega once per
round, a dozen gains per seeded peak). Exact per-round cost, evaluations x
rows per evaluation, from the records: **rounds 2-4 under the policy cost
2,120-2,544 gains against the frozen 2,948-3,080 -- 25 % less per round**, as
expected from the smaller set. The excess sat in two rounds: one on 111
members (seeding fired before the grid was eligible to retire; the cap
exempted the young), and one pyGRANSO thrash (175 evaluations, code 7) on a
17-way tie of near-level satellites -- a high-multiplicity kink, which sigma_2
rows do not cover and an exact-subproblem inner would.

Three defects found by running, all fixed:
1. the cap was soft (age-exempt) -- now hard, excluding only the active,
   this round's peaks, the skeleton and the refuters;
2. with the uniform grid gone the surrogate had no coverage: cap 50 and no
   skeleton made NO progress in six rounds (certified 29-39 against t ~ 9.7)
   -- a coarse uniform skeleton (`--omega-skeleton 32`) never retires;
3. a frequency added by REFUTATION is dormant at the accepted design by
   construction; retiring it re-derived the refuted design and the loop
   cycled (rounds 3-5 identical) -- refuters are protected for four rounds.

| ne=100 K+B+C from K+B, 6 rounds | gamma | |Omega| end | evals | gains |
|---|---|---|---|---|
| frozen policy | **9.410** | 70 | 253 | 17,137 |
| an, cap 50, no skeleton | 10.414 (no progress) | 50 | -- | -- |
| an, cap 50 + skeleton 32 | 10.161 | 50 | 267 | ~14,100 |
| an, cap 80 + skeleton + refuter guard | 9.523 | 80 | 259 | ~21,000 |

**Reading.** Under a once-per-round exchange, |Omega| <~ 50 costs quality:
the surrogate needs coverage the A-N method gets from rebuilding the set at
every trial design, which this loop cannot afford. With the cap loosened to
where it barely binds, gamma is within 1.2 % of the frozen policy at
comparable cost -- the retirement is then removing little and the seeding is
adding satellites the inner solve does not exploit within six rounds. The
gate ("gamma unchanged to rtol, outer rounds and factorisations fall") is not
met by any variant on this case at this budget.

**Where the value is.** Not in the working-set size. The overfitting between
rounds (certified 20-31 in the early rounds of every variant) is the exchange's
cost, and the two things that address it are a trust region on the inner step
tied to the surrogate's validity and a coarse intermediate certification --
both listed for Stage 3, neither done. The policy stays opt-in, with its
defects fixed, for re-measurement once one of those exists; the SQP inner is
the natural partner for the many-way ties the seeding creates.

# 6. Delayed cap (user's suggestion): grid first, cap at 50 later

`--omega-cap-after N`. Seeding from the start, retirement and cap from round N.

| ne=100 K+B+C, 6 rounds | gamma | |Omega| | gains |
|---|---|---|---|
| frozen policy | **9.410** | 64 -> 69 | 17,137 |
| cap 50 + skeleton from round 0 | 10.161 | 64 -> 50 | 15,256 |
| cap 50 + skeleton from round 3 | 10.184 | 64 -> 149 -> 50 | ~25,500 |

Negative, for one reason seen twice. Before the cap, seeding without
retirement inflates the set (satellites accumulate around every peak of every
accepted design; 149 members at round 2), so the early rounds cost MORE than
the frozen policy. When the cap lands, the next inner solve overfits exactly
as it did from round 0 (t = 7.15 against certified 11.58): being nearer an
optimum does not protect a 50-member surrogate from having a gap to walk into.

**Conclusion for item 5 on GL:** |Omega| <~ 50 is incompatible with
once-per-round certification on this problem; the surrogate needs ~70
well-spread members and the frozen uniform grid was providing them. The
retirement machinery is correct and stays (opt-in) for the projected oracle of
Stage 5, where a member costs a reduced-model evaluation rather than a
factorisation and the accounting changes; on the full-model oracle the
working-set size is not where the cost is. The loss in every variant is the
inner step outrunning the surrogate's validity -- the trust region / coarse
intermediate certification item, next.

# 7. The lever: an inner budget that follows the surrogate's validity

Three measurements, same case, ~250 evaluations:

| rule | overfitting early (certified/t) | gamma |
|---|---|---|
| 6 x 20 iterations | 2.1, 1.2, 1.06 | 9.410 |
| 24 x 5 iterations | 1.18, 1.11, 1.09 | 9.989 |
| trust region 0.05 (frozen ball) | 1.63, 1.17, 1.09 | ~10.2 at round 4 |

Short inner solves stop the gaps being exploited and move too little; long
ones overfit early and get further; the ball trust region slows progress more
than it prevents overfitting (the frozen ablation's finding, confirmed under
the iteration budget). So the budget should be ADAPTIVE: the certificate
measures certified/t after every round; `--inner-maxit-adapt` doubles the
next inner budget when that ratio is <= 1 + tol (0.05) and halves it when
> 1 + 3 tol, within [5, 40].

| K+B+C ne=100 from K+B | gamma at ~250 evals | later |
|---|---|---|
| fixed 6 x 20 | 9.410 (253 evals) | -- |
| **adaptive, start 10** | **9.279** (259 evals, round 7) | **8.953** at 443 evals (round 9), still improving |

Trace: budget 5 while certified/t read 1.63 / 1.12 / 1.09; 10, 20, 40 as it
read 1.015 / 1.016 / 1.04; the long solves then converted, 9.99 -> 9.28 ->
9.12 -> 8.95 with certified/t <= 1.07 throughout.

Second case, KSP+B+C from B+C: **9.449 at 263 evaluations** against 9.617 /
9.607 at ~272 with the fixed budget of 20 (-1.6 %), certified/t <= 1.07 from
round 1. Same trace shape (5 -> 10 -> 20 -> 40).

First stage-3 mechanism to beat the frozen policy at equal cost, on both
kink cases at ne=100. Opt-in; ne=200 (the terminal polish) before it is
proposed as a default.

# 8. Clarifications (from the review of sections 5-7)

**What "the surrogate" is.** The inner solve minimises t = max over the
FINITE working set Omega of sigma_1(omega; theta) -- the restriction of the
true objective max over the band. "Overfitting between rounds" is the inner
solve finding a theta whose maximum on Omega is small while the true maximum,
at an omega not in Omega, is large (certified/t = 2.1 in round 0 means the
true norm was twice what the restricted problem believed). It is the exchange
method's intrinsic error -- finite sampling of omega -- and one certification
round corrects it by adding the offending frequency.

**How many frequencies per round.** `certified_norm` returns the refuting
witness PLUS up to four local maxima of a 241-point sweep (run_case_epi.py
~1024); the exchange adds those not already in Omega. The mechanism is "up to
five"; the logs read "+1" because the sweep's maxima are almost always present
already. Item 4's satellites are the "add more" variant, and they did not help
at ne=100 because the inner solve moves the peaks to where the surrogate is
not looking. What would cover that is evaluating trial designs on the true
norm DURING the inner solve; on the full model that is the cost the exchange
exists to avoid; the adaptive budget (section 7) is its cheap approximation.

**Memory at the cavity, and note item 11.** On the full-model oracle a member
of Omega costs a factorisation per inner evaluation, and the pool holds |Omega|
of them at once: the concern is right for the frozen architecture. With the
projected oracle (item 11) the inner problem evaluates sigma_1(omega) on the
reduced model -- one basis V per mesh, a small dense solve per frequency -- so
|Omega| stops being a memory cost in the inner loop; the full-model
factorisations move to the certification, at a handful of peaks per round.
Item 5's "|Omega| <~ 50" was motivated by the full-model cost; the reduction
removes that motivation for the inner loop. The retirement machinery stays,
opt-in, in case it matters after all.

**What beat the frozen policy.** The adaptive inner budget (section 7), not
the working-set policy. One case, one mesh; KSP+B+C and ne=200 before any
default.

**Could A-N be better; why is warm-starting not simply more efficient?**
Warm-starting the working set IS the more efficient thing per evaluation, and
it is why the exchange exists; A-N do not warm-start -- at every trial design
they evaluate the true norm and rebuild the active set. They buy "never
overfit" (every trial judged on the truth) at a norm evaluation per trial,
affordable on their small problems, not on ours on the full model. The
exchange checks the truth once per round and pays with overfitting between
checks. **Once item 11 exists, the natural architecture is both at once: an
A-N-style inner loop on the reduced model, where a norm evaluation per trial
is cheap, inside an exchange whose outer certification is on the full
model.** That is the design Stage 5 should be built toward.

# 9. Items 4-5 re-measured on the corrected loop (2026-09-22)

Fixed before measuring: the policy evaluated sigma_1 on the FULL model for
every member and candidate satellite (|Omega| + ~70 factorisations a round);
it re-certified the accepted design every round (~150) only to get its peak
list; retirement and the cap could remove followed peak members. With an
unprotected cap, a 49-member Omega let the SQP return a design certified at
40.99 against t 7.93 (the capped-away peak member near 3.36).
With peaks protected (cap 50, skeleton 32), K+B+C ne=200, corrected SQP loop:
| | converged gamma | KKT residual | full-model LU / round |
|---|---|---|---|
| no policy | 7.951836 -> 7.950855 | 1.9e-4 / 2.0e-4 | 128-171 |
| satellites + retirement | 7.953104 | 3.6e-4 | 149-169 |
**Recommendation: OFF in the nominal algorithm.** With the reduced model the
inner loop costs no factorisations whatever |Omega| (retirement's purpose is
met), and following handles drifting peaks directly (the satellites' purpose);
the policy lands on a slightly worse stationary point and adds the gap risk.
Kept behind `--omega-policy an` for a full-model inner loop, where |Omega|
costs factorisations. Caveat: the policy's bookkeeping is not checkpointed.
