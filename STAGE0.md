# Stage 0 — harness and dump

Date: 7 September 2026. Deliverables of HANDOFF_implementation.md §5, Stage 0,
built on the cleaned archive (see `CLEANUP.md`).

## 1. What was built

| file | role |
|---|---|
| `harness.py` | run or replay any pipeline variant on a case list; emits the §7 reporting row; compares against the frozen suite |
| `lu_counter.py` | counts sparse factorisations and triangular solves without touching the pipeline |
| `run_case_epi.py --dump DIR` | per-outer-round diagnostic dump (optimisation note item 10) |
| `plot_omega.py` | reads the dump; four panels |
| `--json` on `check_kkt.py` and `tools/check_optimality.py` | machine-readable diagnostics for the harness |

### `lu_counter.py`

Every factorisation in the codebase reaches `scipy.sparse.linalg.splu` by
attribute access, at five sites (`closedloop_sparse:145`, `certify_sparse:112`
and `:314`, `fe_codesign:108` and `:133`). Replacing the module attribute
intercepts all of them and returns a proxy that counts `.solve()` and forwards
everything else, so **no line of production or certification code changes** —
the handoff's standing constraint. Counts are attributed to the caller's
`file:line`, which keeps the once-per-mesh Sobolev and mass factorisations
separate from the per-frequency ones.

Measured on the real oracle: **one σ₁ evaluation = 1 LU + 47 triangular
solves** (22 forward + 22 adjoint power iterations, plus the 3 Woodbury
columns).

Counts are per-process. When the worker pool is on, the workers factorise in
their own processes and this counter cannot see them, so the dump records
`pool_active` and the harness prints a warning rather than reporting a
flattering number.

### The dump patch

`--dump DIR` (opt-in; costs one σ₁-and-gradient pass over Ω per outer round,
on the same pool). Per round it writes `outer_NNN.npz` + `.json` with Ω, σ₁ on
Ω, the active set, the multipliers, the **full** closed-loop spectrum, the LU
counts, the gate decisions, and the inner solver statistics.

Two design decisions worth recording.

**The multipliers are not pyGRANSO's QP multipliers.** Those belong to the
restricted problem in the optimiser's scaled coordinates and are gone by the
time the outer loop sees anything. What the dump computes is the
exchange-level condition: nonnegative weights summing to one minimising
‖Σᵢ λᵢ ∇σ₁(ωᵢ)‖ over the active set. It is normalised **exactly** as
`check_optimality`'s rel |grad| (‖g‖·‖θ‖/γ), so the dump and the diagnostic
report the same quantity and Stage 1's gate reads off either. An earlier
version normalised by the mean gradient norm; that returns 1.0 for every
single-active round and says nothing.

**`solve_epigraph` was discarding the whole pyGRANSO solution object** except
`H_final`. It carries `iters` (optimiser steps, which is *not* `len(cache)`,
the distinct design evaluations), `fn_evals`, `termination_code`,
`stat_value`, and `quadprog_failure_rate` — the last being the direct read-out
for optimisation-note item 0, which Stage 1 opens with. These now go into the
saved `history`, so replay mode gets them at no cost.

## 2. The Stage 0 gate cannot be met as written

The gate asks that the frozen code reproduce all 27 γ **and all gate chains bit
for bit**.

For the archived **records** that is achievable and is verified: the
certification stack re-derives all 27 certified γ at rtol 1e-6 with ratio
1.000 and zero missed peaks, and the harness reproduces all 27 rows with zero
disagreements.

For a **re-run** it is not, and the reason is structural rather than
incidental. The inner solve is budgeted in wall-clock time:

```
solve_epigraph:   o.maxclocktime = max(wall, 10.)
main:             per_outer = (a.wall - elapsed - 45.) / max(a.outer, 1)
```

so the number of BFGS-SQP iterations the inner completes depends on machine
speed and load. Three identical invocations of case B at ne = 100 in this
sandbox returned γ = 10.259071, 10.355313, 10.263187. A harness asserting
bit-for-bit equality of a re-run would be asserting a property of the hardware.

The harness therefore implements the honest gate:

* certified γ agrees with the frozen value to `--rtol` (default 1e-6);
* the gate chain agrees in its **decisions** — accept/reject counts, located vs
  inconclusive, stalled vs converged;
* everything else is reported as measured spread, not asserted.
  `--mode determinism` measures it.

**Recommendation, before Stage 1 starts.** Make the inner budget an iteration
count rather than a clock. Stage 1 changes solver settings and reads the result
off γ and the balance residual; every one of those measurements otherwise
carries this variance, and a 10⁻³-level effect cannot be separated from it.
The change is small and it converts the gate from unachievable to checkable.

## 3. The 27-case replay

`python harness.py --mode replay --cases all --ne 200` — 1024.6 s on one core,
**27 cases, 0 disagreeing with the frozen suite**. γ matches every §3 value to
between 1.6e-07 and 1.1e-05 relative (the table is quoted to 4 dp, so this is
the rounding), all `certified: true`, all stall flags as recorded.

Full table in `harness_out/harness_replay_all27.json`.

## 4. Finding: the balance residual and the stall flag disagree

The balance residual is now baselined across all 27. Sorted:

| case | balance | #active | 2nd peak | termination | §3 stall flag |
|---|---|---|---|---|---|
| K+B+C | 1.47e+04 | 2 | 0.18 % | TERMINATED: inner at limit | yes |
| **KSP+B+C** | **1.73e+03** | 3 | 0.01 % | **CONVERGED** | **no** |
| K+FB | 4.18e+01 | 1 | 0.23 % | CONVERGED | no |
| K+FB+FC | 3.99e+01 | 9 | 0.00 % | CONVERGED | no |
| KSP+FB | 3.48e+01 | 1 | 0.20 % | CONVERGED | no |
| B+C | 1.54e+01 | 2 | 0.00 % | CONVERGED | no |
| KGC | 1.09e+01 | 3 | 0.01 % | CONVERGED | no |
| KSP+FB+FC | 8.72e+00 | 8 | 0.00 % | CONVERGED | no |
| *remaining 19* | median 1.19e-01, max 2.66e+00 | | | | |

**KSP+B+C carries a balance residual of 1.73e+03 — four orders above the median
— with three peaks tied to 0.01 %, while the exchange reported CONVERGED and
the §3 table records it as not stalled.** That is precisely the signature
report_v7 §7 (P5) says was refuted for K+B+C: a multi-active tie with a
gradient 10³ times the converged cases is a stall's way-station, not an
optimum. The design is certified and the γ is real; what is in doubt is whether
it is stationary.

Meanwhile the two other cases the §3 table flags as stalled, KGC+B and
KGC+B+C, have benign balance residuals (2.74e-01, 4.51e-01).

So the exchange's own stall flag and the balance residual disagree **in both
directions**. The flag records that the inner solver hit its limit; the balance
residual records whether the point is equalised. They are different questions
and the campaign has been reading the first as if it answered the second.

**Consequence for Stage 1.** Its case pair is K+B+C and KGC+B. On this
evidence KGC+B is a poor second case — it is not badly unequalised — and
**KSP+B+C is the better one**: same B+C structure, same pathology, one order
less extreme, and it arrives labelled converged, so a fix that works on it is
testable against a case the current diagnostics call healthy. I would run Stage
1 on K+B+C, KSP+B+C and KGC+B rather than the stated pair.

## 5. Smaller findings

1. **The handoff's cost constant is wrong at GL.** "At GL an LU costs ~5
   triangular solves." Measured: **25× at ne = 200, 12× at ne = 400 and 800**
   (LU 1.38/0.96/1.87 ms; solve 0.055/0.082/0.164 ms). The LU is 35 % of a σ₁
   evaluation at ne = 200 and 20 % beyond. This ratio feeds the projection
   through C ≈ 50–100 in both notes' cost tables, so those estimates understate
   what a factorisation-free oracle buys. Re-derive them from the measured
   ratio.
2. **`check_kkt.py` reports only α-active designs**, so it returned a row for
   15 of 27. The harness records its fields as absent rather than zero. The
   balance residual, which covers all 27, comes from `check_optimality`.
3. **`check_optimality` still stamps the refuted verdict.** K+B+C prints
   "equalised (2 active) -- classic H-inf optimum" beside a relative gradient
   of 1.47e+04. Report §7 P5 retired that reading. The `--json` output carries
   the string, but the harness acts only on the numbers, and the JSON writer
   says so in a comment.

## 6. Usage

```bash
# replay: no optimiser, minutes on one core.  Proves a diagnostic or
# certificate change altered nothing.
python harness.py --mode replay --cases all --ne 200

# run: needs the cluster (the §3 timings are 96 workers)
python harness.py --mode run --cases stage1 --ne 200 --wall 1800 \
                  --variant osqp_tight --dumpdir dumps

# how reproducible is the frozen code, really
python harness.py --mode determinism --cases K+B+C --repeat 5

# draw a run
python plot_omega.py --dump dumps/K_B_C --tag K+B+C --outdir figs
```

Case sets: `smoke` (B, K), `stall` (K+B+C), `fields` (K+FB+FC), `stage1`,
`stage2`, `stage3`, `stage5`, `all`.

## 7. Not done here

* `--mode run` and `--mode determinism` are exercised only at ne = 100 on this
  single-core box. Both need a run on the cluster before Stage 1 depends on
  them.
* The dump has been validated on case B at ne = 100 (2 outer rounds, 207
  eigenvalues, multipliers, LU counts). No ne = 200 dump exists yet, because
  producing one means running the optimiser, which on one core with a
  wall-clock budget produces a different answer each time.
* `plot_omega.py` is validated against that same ne = 100 dump.

---

# Stage 0 addendum — the inner budget, and the revised stage-1 cases

Both actions recommended in §2 and §4 above, carried out and measured.

## 8. `--inner-budget iters`

`run_case_epi.py` gains `--inner-budget {clock,iters}`, `--inner-maxit` and
`--clock-guard`. Under `iters`, `maxit` binds and the clock is demoted to a
guard at `clock_guard x` the per-outer budget — present only so a pathological
inner cannot overrun a SLURM allocation, and **loud when it binds**, because a
bound guard means the clock decided after all. pyGRANSO termination code 5 is
exactly that condition, so a run detects its own failure and records
`clock_guard_bound` in the result.

**The default is unchanged (`clock`).** Switching it would silently move every
number the frozen suite is compared against. `harness.py` defaults to `iters`
instead, because a measurement harness must not report numbers the wall clock
decided; it flags rows produced under a clock budget, and marks the frozen
records `reproducible: None` rather than failed, since they are the reference
and cannot be re-run into existence.

### Measured, case B at ne = 100, three identical invocations each

| budget | γ | calls | termination codes |
|---|---|---|---|
| `clock` | 10.23069918141667 / …296660177 / …181378927 | 47 / 49 / 61 | [5,6,5] [5,6,0] [5,6,6] |
| `iters`, maxit 25 | 10.230699296622497 ×3 | 116 / 116 / 116 | [6,6,6] |
| `iters`, maxit 6 | 10.230699296660347 ×3 | 53 / 53 / 53 | [4,4,6] |

Bit-for-bit reproducibility is restored in both `iters` rows.

### Why the calls doubled at maxit 25, and why that is not a regression

Iterations per outer round:

```
clock      [5, 8, 8]  [6, 8, 9]  [5, 8, 8]      codes [5,6,*]
iters(25)  [15, 8, 8] x3                        codes [6,6,6]
```

The difference is **outer round 0 alone**: the clock cut it at 5–6 iterations
when the solver wanted 15 to reach its own stopping criterion. Rounds 1 and 2
ran 8 iterations under both budgets — they were already terminating
algorithmically. Cost per oracle call is unchanged and nothing in the pipeline
became slower; the extra calls are extra work **requested**, plus a downstream
effect (a longer round 0 lands round 1 at a different point, whose line search
costs more evaluations).

The extra work is **optional**: `--inner-maxit 6` truncates at the frozen work
level (53 calls vs the frozen 47–61) and is still bit-identical, terminating on
code 4 (maxit) rather than code 5 (clock). What changed is not the cost but the
nature of the stop — truncation became a stated decision instead of an artefact
of machine load. Which cap is right is an empirical question for stage 1
(optimisation note item 8).

**Caveat on these numbers.** `--wall 40 --outer 3` makes
`per_outer = (40 − elapsed − 45)/3` negative, so `max(wall, 10.)` yields a
10-second inner limit. This is a deliberately stressed setting and the 2× is an
upper bound; at production settings (`--wall 1800`, `per_outer ≈ 44 s`, 96
workers) the clock binds less often. **How often code 5 fires at production
settings is now measurable and has never been measured** — the frozen records
predate `termination_codes`. Worth doing before stage 1 opens, since it sizes
the variance every stage-1 measurement carries.

## 9. Revised stage-1 case set

`CASE_SETS["stage1"]` is now **K+B+C, KSP+B+C, KGC+B**; the handoff's stated
pair is kept as `stage1_handoff`. Replayed clean, 0 disagreements:

```
K+B+C     7.964064  balance 1.47e+04  TERMINATED: inner at limit
KSP+B+C   8.246535  balance 1.73e+03  CONVERGED
KGC+B    11.600868  balance 2.74e-01  TERMINATED: inner at limit
```

KSP+B+C is the better second case (§4): same B+C structure, same pathology one
order less extreme, and the current diagnostics call it healthy, so a fix that
works on it is falsifiable. KGC+B is retained as a **control** — a fix for the
stall should not move it. The rationale sits in a comment beside the set so it
survives without this document.

## 10. State of validation after these changes

* `selftest.py`: ALL CHECKS PASSED, rc 0 (default path untouched).
* `verify_dense` 27/27, 0 invalid; `winding_cert` 27/27 count 0; `check_kkt`
  K+B+C unchanged; frozen table 0 discrepancies.
* `harness.py --mode replay --cases all`: 27 cases, 0 disagreeing.
* `harness.py --mode replay --cases stage1`: 3 cases, 0 disagreeing.
* Still only ever exercised at ne = 100 on one core: `--mode run`,
  `--mode determinism`, and the dump at ne = 200. Run
  `bash tools/validate_cleanup.sh` on the cluster before stage 1 depends on any
  of this.

---

# 11. Cluster validation, 11 September 2026 (host f02)

`tools/validate_cleanup.sh`, 16:00:13 → 16:24:02, 24 minutes.

| step | result |
|---|---|
| 1. `selftest.py` | ALL CHECKS PASSED |
| 2. controller order | r = 8, from the design records |
| 3. dense cross-check | 27/27, 0 invalid |
| 4. winding certificate | 27/27, count 0 |
| 5. peak certification 1e-6 | 27/27 certified |
| 6. handoff §3 table | 0 discrepancies, 19 notes |
| 7. harness 27-case replay | **27 cases, 0 disagreeing** (757.7 s) |
| 7a/7b | nesting lattice and collected summary build |
| 8. frozen records | RECORDS UNCHANGED |

Cross-check worth noting: the cluster runs **numpy 2.5.1 / scipy 1.18.0**
against this sandbox's 2.4.4 / 1.17.1, and every certified γ and every balance
residual agrees — including KSP+B+C at 1.73e+03. The §4 finding is not a
library artefact.

## 11.1 Step 0 did not actually run, and the script hid it

The environment gate aborted:

```
Error obtaining unique transport key from PMIX
  (OMPI_MCA_orte_precondition_transports not present in the environment)
*** An error occurred in MPI_Init_thread
*** MPI_ERRORS_ARE_FATAL (processes in this communicator will now abort...)
```

`check_env.py`'s PETSc probe called `MPI_Init` outside a job allocation; Open
MPI called `MPI_Abort`; the process died. Because the probes ran at import time
in one process and the report printed only at the end, **numpy, PyTorch and
pyGRANSO were never reported at all** — and `validate_cleanup.sh`'s
`|| echo "NOTE: ... not needed for GL"` turned that into a reassuring note. The
run continued as though the gate had passed. It had not run.

Nothing downstream was affected — steps 1–8 exercise the same libraries far
harder than the gate does, and they all passed — so the validation stands. But
the gate was decorative, which is worse than absent.

**Fixed, both halves.**

`check_env.py` now registers probes instead of executing them at decoration
time, and runs each in its own subprocess (`--in-process` restores the old
behaviour, `--only KEY` runs one). `MPI_Abort`, a segfault and a C-level exit
are not catchable in Python, so a process boundary is the only containment. A
probe that dies without reporting yields `probe died without reporting (exit N)`
plus, when MPI or PMIX appear in its stderr, the hint to run inside an
allocation. Verified with a probe that calls `os._exit(17)` after writing an
MPI-style message: one row fails, the other five report normally.

It also now distinguishes three outcomes rather than one boolean — all pass;
the three GL requires pass while cavity-only libraries do not (exit 0, since
that is the normal state for a GL node); a GL requirement fails (exit 1).

`tools/validate_cleanup.sh` step 0 no longer discards the exit status: it fails
the run if `check_env.py` exits nonzero, and separately greps for an explicit
`[PASS]` line for each of numpy, PyTorch and pyGRANSO, so "aborted before
reporting" can no longer look like "PETSc is absent, fine for GL".

**Operational note:** run the validation inside an allocation (`srun`/`sbatch`),
not on a login node. `env.sh` records check_env passing 6/6 on this cluster,
which is consistent — that was inside a job, where PMIX supplies the transport
key.

## 11.2 What the log does not yet tell us

Replay mode reads the frozen records, which predate `inner_budget`, so every
row came back `reproducible: None` and the wall-clock warning correctly did not
fire. The cluster has therefore still not exercised `--mode run`,
`--mode determinism`, or the ne = 200 dump. Those remain the open items before
stage 1, together with measuring how often pyGRANSO termination code 5 fires at
production wall settings (§8).

---

# 12. Environment findings

Superseded twice. The final picture, and what each intermediate conclusion got
wrong, because the wrong ones were acted on.

## 12.1 What is actually true

| | login node f02 | compute node n046, in a job |
|---|---|---|
| numpy | PASS | PASS 2.5.1 |
| PETSc complex | FAIL, MPI_Init | **PASS 3.25.3 complex128** |
| SLEPc | FAIL, MPI_Init | **PASS** |
| dolfinx | FAIL | **PASS 0.11.0** |
| torch | PASS | **FAIL, GLIBCXX_3.4.29** |
| pyGRANSO | PASS | **FAIL, GLIBCXX_3.4.29** |

Both columns with `env.sh` sourced from a clean shell. The compute-node column
is job 15966182.

So `env.sh` is right about MPI and about the Intel module -- PETSc, SLEPc and
dolfinx all pass inside an allocation with `impi/.22.2.0` loaded. Its one gap is
`libstdc++`: numpy's C extension needs `GLIBCXX_3.4.29`, the login node's
default environment happens to supply it, and a compute node does not.

**The fix is one line**, added to `env.sh` after the conda activation:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

The conda env ships a new enough libstdc++ in its own lib directory. Expected
result on a compute node: 6/6.

**Caveat that must not be skipped.** This also puts the conda BLAS ahead of the
module's MKL, which changes what runs underneath every number. Re-run
`tools/validate.slurm` after adding it and confirm all 27 certified gammas still
reproduce before trusting anything measured afterwards. That is precisely what
the harness replay in step 7 is for.

## 12.2 Two wrong conclusions, and why

**"Two MPI runtimes in one process."** Drawn from the login-node observation
that `module purge` made PETSc and SLEPc pass. It is refuted by the compute-node
run, where they pass WITH the Intel module loaded. Open MPI simply needs an
allocation; `module purge` on a login node fixes MPI for an unrelated reason.
The original hint -- "run inside an allocation" -- was right, and replacing it
was an over-correction on one data point.

**"dolfinx is broken."** It needs `libimf.so` from the Intel runtime, which
`env.sh` supplies and which only went missing because the suggested `module
purge` removed it. dolfinx 0.11.0 passes under `env.sh`. `env_cavity.sh`, built
to work around this, has been deleted: it solved a problem created by its own
recipe, and keeping a second environment file that diverges from the one every
validated number was produced under is worse than useless.

## 12.2b Confirmed on a compute node

Job of 2026-09-14 09:27 on **n005** (a different node from 15966182, so the fix
is not node-specific), with the one line added to `env.sh` and nothing else:

    [PASS] numpy 2.5.1   [PASS] PETSc 3.25.3 complex128   [PASS] SLEPc
    [PASS] torch 2.6.0+cpu   [PASS] pyGRANSO f*=0.5000   [PASS] dolfinx 0.11.0
    ALL GOOD

and the full chain behind it: selftest 16/16; dense cross-check 27/27 valid,
0 invalid; winding certificate 27/27, count 0; peak certification at
**rtol 1e-6, ratio 1.000 on all 27**; handoff table 0 discrepancies, 19 notes;
harness replay 27/27, 0 disagreeing; `results/` sha256 unchanged
(6727a83f...898eed before and after).

**The BLAS caveat in 12.1 is discharged for everything the chain recomputes.**
Steps 3-5 re-derive the closed-loop spectrum and re-find every peak with the
conda BLAS ahead of the module MKL, and no certified number moved. The
remaining gap is narrow but real: step 7 replays stored records, it does not
re-solve, so the first FRESH ne=200 solve under the new library order is
`tools/stage1_baseline.slurm`. The ne=100 solves inside selftest did run fresh
and passed.

## 12.3 The hazard that is real

`env.sh` loads the site python module and then activates conda, relying on the
activation to put the conda python first on PATH. From a clean shell that works
-- job 15966182 confirms `python = .../envs/codesign/bin/python`, 3.12.13. From
a shell where `codesign` is ALREADY active, conda treats the activation as a
no-op, makes no PATH change, and the module python 3.8 wins (numpy 1.23.4, no
torch).

`sbatch` defaults to `--export=ALL`, so a job submitted from an activated shell
inherits this. All three job scripts check it up front and stop with the cause
named. Durable fix for `env.sh` itself: `conda deactivate 2>/dev/null || true`
before the activate.

## 12.4 Incidental: two torch installations

Interactively without `PYTHONNOUSERSITE`, the gate reported torch 2.6.0+cu124;
with it -- as `env.sh` and all three job scripts set -- the conda env's
2.6.0+cpu is used. An interactive test that omits it is not testing what the
jobs run.


# 13. Missing input files

The frozen archive shipped without three files the pipeline needs:
`seeds_ne200.npz`, `case.slurm` and `env.sh`. All three were supplied by hand.
A fourth, `init_ne200.npz`, was missing and only surfaced when a job first tried
to cold start at ne=200:

    FileNotFoundError: [Errno 2] No such file or directory: 'init_ne200.npz'

`init_ne{ne}.npz` holds the cold-start controller (d, e, f, nvec at the
catalogue order) and the field positions xa, xs that `run_case_epi` starts every
case from. `prepare.py` writes it.

## 13.1 Why not just run prepare.py

`prepare.py` writes **both** `init_ne{ne}.npz` and `seeds_ne{ne}.npz`, and it
builds the catalogue first because the catalogue decides r. Running it now would
overwrite the supplied catalogue -- the one every certified gamma was warm
started from, that `case.slurm` selects seeds by name from, and that
`ctrl_order.py` reads r from. A rebuild that differed at all (a different BLAS
moving the Hankel singular values, the shoulder landing on r=7) would silently
make the frozen suite incomparable, and nothing downstream would check.

## 13.2 tools/make_init.py

Builds init alone. Takes r from the existing catalogue via `ctrl_order.infer_r`
rather than re-deriving it, refuses to overwrite an existing init without
`--force`, and never writes the seeds file. It raises rather than continuing if
the canonical form comes back at an order the catalogue disagrees with --
`run_case_epi` would otherwise quietly substitute a seed controller mid-run.

Verified at ne=100 against a `prepare.py` reference built in a scratch tree:
**all 10 fields bit-identical, dtypes included**, catalogue md5 unchanged.
3.9 s versus 120.8 s, since the catalogue was the expensive part.

`prepare.py` remains correct for a mesh built from scratch, where there is no
catalogue to protect.

## 13.3 Wired into the jobs

`determinism.slurm` and `stage1_baseline.slurm` build the init if it is absent
and stop with the cause named if the seed catalogue is missing too -- that case
needs a human decision, since the only way forward is a fresh catalogue.
`validate.slurm` does not need init: it replays and certifies stored designs.


# 14. pyGRANSO aborts on quadprog failure (code 12)

First seen in the determinism job (15969703, n044, K+B+C, ne=200), outer round 2:

    ERROR in LDL_factor: ... The problem seems to be non-convex
    ERROR in osqp_setup: KKT matrix factorization.
    PyGRANSO steeringQuadprogFailure: Steering aborted due to a quadprog failure.
    ...
    UnboundLocalError: cannot access local variable 'y' where it is not
    associated with a value
    PyGRANSO:steeringQuadprogFailure

## 14.1 Mechanism

Two defects stacked, both in pyGRANSO 1.2.0, neither ours:

1. `private/qpSteeringStrategy.py:250` -- the `except` around the OSQP call
   prints the failure and then falls through to
   `d = -self.mu_Hinv_f_grad - (self.Hinv_c_grads @ y)` with `y` never
   assigned. A commented-out `sys.exit()` shows the intent was to abort.
2. `private/bfgssqp.py:330` -- catches the resulting UnboundLocalError, calls
   `prepareTermination(12)` and returns immediately.

Step 2 is the damaging one. pyGRANSO's own fallback ladder is a dozen lines
below that handler: `fallback_level` 1 degraded steering (identity H), 2 plain
BFGS, 3 steepest descent. A steering QP failure is supposed to step down it.
The early return skips all of it, so the inner solve does not degrade, it stops.

The trigger is OSQP error 4 (`OSQP_NONCVX_ERROR`): the steering KKT matrix is
not numerically PSD when LDL-factored. `solveQP` sets `eps_abs = eps_rel =
1e-12` with `polish=True`, which is very tight.

## 14.2 What it costs

Round 2 of that run used 115 inner calls against 284 and 268 for rounds 0 and 1,
and returned a certified 9.679 above the incumbent 9.601 -- the round
contributed nothing. The point returned is the best iterate seen BEFORE the
failure, at an arbitrary place in the inner solve. The round's call count is a
floor, not the work the round needed.

## 14.3 Why the frozen suite could not see this

`solve_epigraph` discarded the pyGRANSO solution object's stats, so
`termination_code` never reached the record. `collect.py:222` then reports every
stall as "exchange TERMINATED: frequency set stationary, inner solver at its
limit" -- a hardcoded string, printed on the `stalled` flag alone. It is a guess
about the cause, and on code 12 it is the wrong one.

K+B+C is the case to suspect: balance residual 1.47e+04, two orders above every
other case, carrying exactly that label. "Ran out of budget" and "aborted on a
QP factorization failure" call for different fixes.

## 14.4 Recorded, not fixed

`run_case_epi` now names code 12 when it fires (mirroring the code-5 clock-guard
warning) and writes `qp_failure` into each round's dump. Both are print/record
only; no numerical path changed.

The fix belongs in stage 1, and there are three candidates, in increasing order
of intervention:

  a. Symmetrise and ridge the steering Hessian before OSQP sees it
     (H <- (H + H^T)/2 + delta*I). Addresses the trigger. Changes the steering
     subproblem slightly on EVERY step, including ones that currently succeed.
  b. Loosen `eps_abs`/`eps_rel` from 1e-12, or drop `polish`. Same caveat.
  c. Repair the fallback: on a steering failure, increment `fallback_level` and
     continue instead of returning. This changes behaviour ONLY on steps that
     currently abort, which makes it the one that cannot damage a healthy run.

(c) is the conservative choice and should be measured first, against the
determinism job's failure counts.

## 14.5 Determinism implication

A quadprog failure is a numerical-tolerance event. If the worker pool perturbs
sigma_1 in its last bits, the failure can fire on one repeat and not another,
which would break `--inner-budget iters` reproducibility at ne=200 by a route
that has nothing to do with the clock. Watch for `qp_failure` differing across
the three repeats.


# 15. Determinism, measured at production settings

Job 15975493, n011, K+B+C at ne=200, WALL=2250 OUTER=40 (per_outer ~ 55 s),
96-worker pool, exclusive node. Timed out at 3 h during the sixth run; the five
that completed are conclusive.

## 15.1 Result

| budget | repeats | gamma | calls | wall s |
|---|---|---|---|---|
| clock | 3 of 3 **bit-identical** | 9.872014121750077 | 2710 | 678.0 / 677.7 / 678.8 |
| iters (maxit 200) | 2 of 2 **bit-identical** | 9.827659514562182 | 7200 | 4222.5 / 4241.9 |

Identical on every outer round, not just at the end.

**The clock is not a source of nondeterminism at production settings.** It bound
-- 68 s per outer against a 55 s allowance -- and the runs still reproduced
exactly. The ne=100 spread reported earlier (10.259071 / 10.355313 / 10.263187)
was an artefact of a stressed 10 s inner limit, where the cut lands mid
iteration. At 55 s on a warm pool it lands on the same iteration every time.

Four `iters` rounds tripped the clock guard, whose warning claimed the round was
"not reproducible", and those rounds then reproduced bit for bit. The wording
has been corrected: a clock cut is reproducible when it lands on the same
iteration, and the guard cannot tell.

## 15.2 Consequence for --inner-budget

`iters` is NOT needed for reproducibility. It buys 0.45 % on gamma
(9.8720 -> 9.8277) for 2.7x the calls and 6.2x the wall clock. Its value is that
truncation becomes a decision rather than an accident; that is a stage-1 tuning
question, not a correctness fix. The default stays `clock`.

## 15.3 Across nodes: identical.  The variable was the input

Probe runs, K+B+C ne=200, OUTER=2 WALL=155 (per_outer 55 s), one repeat each:

| | n044 (15996816) | n011 (16003306) |
|---|---|---|
| clock gamma | 10.758914546915708 | 10.758914546915708 |
| clock calls | 564 | 564 |
| iters gamma | 10.758914546915708 | 10.758914546915708 |
| iters calls | 1979 | 1979 |
| outer 0 certified | 17.101121 | 17.101121 |
| outer 1 certified | 61.189677 | 61.189677 |

Bit-identical on both budgets. Both nodes report `ActiveFeatures=spr`, same
microarchitecture. **Determinism holds across nodes, and the hardware suspicion
of the previous draft was wrong.**

What actually differed: job 15969703 is the only run of this case that did NOT
build `init_ne200.npz` fresh -- it used a pre-existing file, and it is the only
run that diverged. Every run since has used an init built by `make_init` under
the fixed `env.sh`, and they all agree.

`init_ne{ne}.npz` comes out of a dense balanced truncation, the most
BLAS-sensitive step in the pipeline. If that older file was produced before the
`LD_LIBRARY_PATH` change it came from the module MKL rather than the conda BLAS.
A last-bit difference in d, e, f is invisible in a start gamma printed to six
digits -- both runs printed `start gamma=10.758915`, |Omega|=92, anchor
alpha=-0.2261 -- and still sends the inner solve down a different path from
outer 0, tau governor included. Unproven unless that file survives; the check is
`ls -la init_ne200.npz` against the mtime of the `env.sh` edit.

A smaller difference, and a reassuring one: 15975493 took 290 calls at outer 0
where the probes took 277, with identical t and certified. per_outer was 55.125 s
there against 55.0 s here, so the clock cut a few iterations later and those
iterations improved nothing.

**Fixed in the record.** The summary now carries an `inputs` field with the
sha256 prefixes of the init and seeds files actually used:

    "inputs": {"init": "b334f39d...", "init_file": "init_ne200.npz",
               "seeds": "85a22624...", "seeds_file": "seeds_ne200.npz"}

Neither file is under version control and both have been rebuilt by hand at
least once. Hashing costs microseconds and makes "same inputs?" answerable from
the record rather than from memory -- which here cost two jobs and a wrong
hypothesis.

The file hash is a valid content fingerprint despite .npz being a zip: numpy
pins every entry's timestamp to 1980-01-01, so savez output is byte
reproducible. Verified -- two make_init runs five seconds apart produced the
same sha256.

**The two-init question, resolved -- and it was the package.** Smoke job
16033711 reported `init=ef2de129`; the file hashed by hand the day before was
`4e88c231`. Only one exists now, `ef2de129`, and it is the `init_ne200.npz`
from the assistant's sandbox, built there on 2026-09-15 09:51 and **shipped in
every tarball since**. Each extraction over the working tree overwrote the init
that job 15996816 had built on n044.

Hankel singular values, cluster build (n044, from the job log) against the
sandbox build:

| k | n044 | sandbox | rel diff |
|---|---|---|---|
| 0 | 25.212178263756122 | 25.212178271518692 | 3.1e-10 |
| 3 | 2.765572036898369 | 2.765572038075117 | 4.3e-10 |
| 6 | 0.109555371961675 | 0.109555372308267 | 3.2e-09 |
| 7 | 0.021376444337181 | 0.021376443997991 | 1.6e-08 |

Different numpy (2.4.4 vs 2.5.1), different BLAS, different CPU. 1e-8 relative
in the cold-start controller is far beyond last-bit and easily enough to change
the inner solve's path from outer 0. This is the mechanism hypothesised for the
job 15969703 divergence, now seen directly on two files that can be compared.
Which init 15969703 itself used is still not established; that file is gone.

**Consequences.**
- Tarballs no longer carry `init_ne*.npz` or `seeds_ne*.npz`. Input files that
  are not under version control must not travel inside a code package: the
  package overwrote a cluster-built input with a sandbox-built one, silently,
  three times.
- A run started from `ef2de129` is not wrong -- the cold start is a starting
  point and the design is judged by certification -- but it did not start from
  a cluster-built init, and its trajectory will not match one that does.
  Delete the file before the next job and the preflight rebuilds it on the
  cluster in about 25 s; the record then carries a cluster-built hash.
- Cross-node bit-identity (section 15.3) was measured on ONE shared init file.
  It says the solver is reproducible across nodes; it says nothing about whether
  `make_init` is, and the sandbox/cluster gap suggests it should be measured
  rather than assumed: build on n044 and n011 and compare hashes.

The pre-fix init that job 15969703 used is not in the
working tree -- it is 14:51, after the 10:28 env.sh edit -- so the hypothesis in
this section stays unproven unless a copy survives in another tree.

## 15.4 The gate rejection loop wastes 29 % of the run

Outer rounds 6-9 are the same rejected design four times:

    gate REJECTS: controller pole Re=+0.0988276 > allowance +0  (alpha -0.0603)
    outer 7/8/9: |Omega|=98  t=10.251006  certified=9.773436  (+0 freq, 264 calls)

264 calls each, identical every time. The rejection never reaches the inner
problem, so it re-derives the same infeasible controller and is rejected again;
the exchange then stalls on "no progress for 3 outers". 792 of 2710 calls
produce nothing.

Stage-1 item: feed the controller-pole constraint into the inner problem instead
of filtering after the fact.

## 15.5 gamma here is not comparable to the frozen table

This run gives 9.872 for K+B+C against the frozen 7.964. `determinism.slurm`
uses ONE cold start; `case.slurm` runs 8 seeds and keeps the best. The two
answer different questions and the numbers must not be put in the same column.


# 16. Stage-1 baseline, job 16016215 -- VOID, and why

All three baseline runs are invalid. The cause is a defect in `harness.py`, not
in the pipeline.

## 16.1 The harness never passed --eps

`case.slurm` passes `--eps 0.05` explicitly. `harness.py` passed no `--eps` at
all, so `run_case_epi` fell back to its own default,
`packer.EPS_STAB_DEFAULT = 1e-3`. The stage-1 baseline therefore solved a
problem with a stability margin **50x weaker than the specification**, and so
did the determinism job, which had the same omission.

It is visible in the logs and was read past several times:

    winding certificate: ... right of Re=-0.0005      <- baseline / determinism
    counting line Re = -0.0250                        <- frozen validation

The counting line is -eps/2. -0.0005 is eps=0.001; -0.0250 is eps=0.05.

## 16.2 What it produced

| case | gamma | frozen gamma | alpha | alpha + 0.05 | at the spec eps |
|---|---|---|---|---|---|
| K+B+C | 9.8679 | 7.9641 | -0.043705 | **+0.0063** | INFEASIBLE |
| KSP+B+C | 9.7261 | 8.2465 | -0.222603 | -0.1726 | feasible |
| KGC+B | **882.8808** | 11.6009 | -0.017658 | **+0.0323** | INFEASIBLE |

KGC+B's 882.8808 is the **open-loop** gain (`run_case_epi.py:1131` names
that number). The run did not exploit a weak margin, as a first reading here
claimed; it never left the open loop in 15 outer rounds and 7332 calls. Why a
cold start at start 0 fails to move for this case, when the frozen KGC+B record
is also `_s0_` and reached 11.6009, is open -- see section 17.

## 16.3 The certification chain did not catch it

`verify_dense` reported `0 design(s) are INVALID`. `winding_cert` flagged KGC+B
but passed K+B+C. Both are behaving as written:

| tool | --eps default | what it tests |
|---|---|---|
| run_case_epi | **1e-3** (packer.EPS_STAB_DEFAULT) | alpha + eps <= 0 |
| verify_dense | **1e-3** | alpha + eps <= 0, i.e. "is it stable" |
| winding_cert | 0.05 | no eigenvalue right of **-eps/2** |
| check_optimality | 0.05 | -- |
| check_kkt | 0.05 | -- |

Two problems compound. The defaults disagree, which is how the two tools
returned opposite verdicts on the same design in the same log. And
`winding_cert` counts across a line at **-eps/2**, so a design between -eps/2
and -eps passes the winding certificate while violating the specification.
K+B+C at alpha = -0.0437 sits exactly in that gap: it cleared every certificate
in the chain and is infeasible.

**Nothing in the chain tested alpha + eps <= 0 at the suite's eps.**

## 16.4 Fixed

- `harness.py` passes `--eps` on both run paths; new `--eps` flag defaults to
  0.05, with `SUITE_EPS` as a module constant documenting why.
- `determinism.slurm` and `stage1_baseline.slurm` take `EPS=${EPS:-0.05}` and
  pass it to the harness, to `run_case_epi`, and to `verify_dense` and
  `winding_cert` in the certification step.
- The harness now checks feasibility **first and independently of the frozen
  comparison**, since an infeasible design can post any gamma it likes. A
  violation is reported as `INFEASIBLE alpha=... > -eps=...`, and a run at an
  eps other than the suite's as `EPS=... (suite uses 0.05)`.
- Records predating the `eps` field assume the suite value rather than skipping
  the check, flagged as `eps_assumed`. Verified against the tightest frozen
  design, KSP+C at alpha = -0.051179: margin -0.001179, feasible, no false
  alarm.

Retro-applied to job 16016215, the new check flags K+B+C and KGC+B as
INFEASIBLE and all three as EPS=0.001.

## 16.5 What this does NOT invalidate

The determinism conclusions of section 15 are about whether identical inputs
give identical outputs, which does not depend on eps. Bit-identity across three
repeats, two budgets and two nodes stands. **The gamma values quoted there are
at eps = 0.001 and must not be read as design results** -- which also explains
the K+B+C discrepancy that section 15.5 attributed to seed multistart alone.
The section 15.4 observation about the gate rejection loop stands, at eps=0.001.

Sections 1-14 are unaffected: they concern the frozen suite, which was produced
by `case.slurm` at eps = 0.05.

## 16.6 Two cheap guards, so this cannot recur silently

A 6 h job is a bad place to discover a missing command-line flag. Two tiers now
sit in front of it.

### tools/check_invocations.py -- under one second, no allocation

A textual audit: every invocation of `run_case_epi`, `verify_dense` and
`winding_cert` in the tree must carry `--eps`. Run it on a login node with no
environment at all.

On first run it found **nine further omissions** beyond the harness:

| file | calls | at |
|---|---|---|
| tools/validate_cleanup.sh | verify_dense, winding_cert | steps 3 and 4 of the validation chain |
| tools/case.slurm | verify_dense | the frozen campaign's own dense check |
| tools/sweep.sh | verify_dense, winding_cert | |
| selftest.py | run_case_epi | the ne=100 gate cases |
| check_kkt.py, tools/check_symmetry.py | run_case_epi | printed re-run advice |
| tools/collect.py | verify_dense | printed re-run advice |

So the frozen campaign's post-hoc dense check also ran at 1e-3, and the
validation chain's steps 3 and 4 were testing "is it stable" rather than "does
it meet the spec". No frozen result changes -- every one of the 27 has
alpha <= -0.0512 and passes at 0.05 -- but the check was weaker than it claimed
for the entire campaign.

The three advice strings mattered too: they told a user to re-run a case with a
command line that omits `--eps`, reproducing the bug by hand.

All fixed; the audit is clean. `selftest.py` now runs its gate cases at 0.05 and
still passes (case B, alpha = -0.1842).

### SMOKE=1 on stage1_baseline.slurm -- about five minutes

    SMOKE=1 sbatch --time=00:20:00 tools/stage1_baseline.slurm

The short `--time` is deliberate: in smoke mode the job should finish in about
five minutes, so a 20-minute limit makes a smoke run that is silently NOT in
smoke mode fail fast instead of consuming the full allocation. That happened
once -- job 16029721 ran the real baseline for 30 minutes because the SMOKE
block was missing from the script (a patch that mutated the file and then threw
before writing, "verified" by counting grep hits from the half that did land).
The banner now prints `smoke=$SMOKE` and `eps=$EPS`, so the log says which
configuration actually ran on its first line.

One case, one outer round, `MAXIT=25`, writing to `results_smoke/`. It exercises
the whole chain -- run, dump, plot, certify, harness verdict, exit codes -- and
prints a checklist: is eps in the record, did the feasibility check run, are the
inputs fingerprinted. It exits nonzero if any of those fail.

It proves the plumbing and nothing about the optimisation. **Gammas from a smoke
run mean nothing and the job says so twice.** A nonempty `problems` list is
expected there, since one outer round will not reach the frozen gamma.

### Smoke result, job 16032855 (n021, 5 minutes)

    cases=K+B+C ne=200 eps=0.05 wall=155 outer=1 maxit=25 smoke=1
    eps in record      : 0.05  OK
    alpha              : -0.0536   margin -0.0036      <- feasible at the real eps
    feasibility check  : ran
    inputs fingerprint : init=None seeds=None          <- FAIL

The one failure was a reporting omission: `run_case_epi` writes `inputs` into
the record and the harness never copied it into its row. Fixed (one line),
verified locally against a record that carries the field. Everything the smoke
run was for -- eps reaching the record, feasibility checked at the right margin,
certification, plots, exit codes -- worked.

Re-smoke, job 16033711: `PLUMBING OK`, fingerprint populated
(`init=ef2de129 seeds=b6c5f4f2`), everything else identical to 16032855.

## 16.7 Still to do

The real baseline: `sbatch tools/stage1_baseline.slurm`. Nothing else in the
job needs changing. Separately, resolve the two-init question above.


# 17. Stage-1 baseline, second attempt (job 16034591): one clean row of three

## 17.1 What happened

| case | ran for | row wall | row eps | source of the row |
|---|---|---|---|---|
| K+B+C | 4449 s | 4449 s | 0.05 | **this job** |
| KSP+B+C | 1698 s | 2761 s | 0.001 | job 16016215 |
| KGC+B | 11 s | 2078 s | 0.001 | job 16016215 |

`OUTDIR=results_stage1_baseline` was the same directory the void job wrote to.
`run_case_epi` writes `<tag>_ckpt.npz` on every accepted iterate and at startup
resumes from it whenever its gamma beats the chosen start. The void job's
eps=0.001 checkpoints were still there. KSP+B+C resumed from the void solution
at 9.7261, ran 28 minutes at eps=0.05 without beating it, and kept it. KGC+B
exited in 11 seconds. The harness flagged both rows `EPS=0.001 (suite uses
0.05)` -- the check added in section 16.4 did its job.

**Fixed:** `OUTDIR` and `DUMPDIR` default to `..._${SLURM_JOB_ID}`, smoke mode
included, and the job refuses to start on a directory that already holds
`*_ckpt.npz` unless `RESUME=1`. Verified by executing the guard on a directory
seeded with a stale checkpoint: refuses; with RESUME=1: proceeds.

## 17.2 The one valid row: K+B+C at eps = 0.05, cold start, iters budget

    gamma 9.816523   alpha -0.068384 (margin -0.018)   balance 1.08e+01
    7 outer, 747 inner it, 7038 calls, 4449 s, CONVERGED

Against the frozen K+B+C: gamma 7.964064, balance 1.47e+04, TERMINATED.

The two are different points. The frozen one has the lower gamma and is not
stationary (balance four orders higher, flagged stalled). This one is
stationary to within 1.08e+01 -- still not small, but a thousand times closer --
at a gamma 23 % higher. The frozen record is `_s0_` too, so start index alone
does not explain it; budget (clock vs iters), the pre-fix environment, and the
init file are all candidates and the fingerprint now records the last. This is
the first genuine stage-1 baseline row and the comparison it sets up -- a
non-stationary 7.96 against a stationary 9.82 -- is exactly the kind of thing
stage 1 has to resolve.

No gate-rejection loop (section 15.4) appears in this row's termination; the
per-round log will say whether it occurred at all at eps=0.05.

## 17.3 Still to do

    CASES=KSP+B+C,KGC+B sbatch tools/stage1_baseline.slurm

About 1.5-2 h. KGC+B may again fail to leave the open loop; if it does, the
start-gamma line and the first outer rounds of its log are what to look at,
since the frozen `_s0_` record shows a cold start CAN reach 11.6 for this case.


## 17.4 Rerun, job 16047382: KSP+B+C clean, KGC+B missing

**`make_init` is byte-reproducible across nodes.** The KSP+B+C record carries
`init=4e88c231` -- the hash of the n044 build, now rebuilt on n021. The
cross-node bit-identity of section 15.3 therefore holds for the input as well
as the solver. The 1e-8 gap of section 15.3 was sandbox-vs-cluster only.

**KSP+B+C at eps=0.05, cold start, iters budget:**

    gamma 9.797363   alpha -0.060043 (margin -0.010)   balance 3.98e+04
    12 outer, 1134 inner it, 4654 calls, 1715 s, TERMINATED: inner at limit
    termination codes [6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]   qp_failure_rate 0

Per-round history, the part that matters:

| round | certified | best kept | |
|---|---|---|---|
| 3 | 9.797363 | 9.797363 | accepted; **the final answer** |
| 4 | 9.619574 | 9.797363 | 1.8 % better, refused |
| 7 | 9.650219 | 9.797363 | 1.5 % better, refused |
| 9-11 | 10.046031 | 9.797363 | same point three times, then stall |

`gate_located = 4`: the controller-pole gate refused four designs, two of them
better than what was reported. The final design is a round-3 iterate that the
rest of the run improved on and was not allowed to keep. Its balance residual,
3.98e+04, is the same non-stationary signature as the frozen KSP+B+C (1.73e+03)
and K+B+C (1.47e+04) -- and section 15.4's "wasted 29 %" understated the
problem: the gate is not merely slow, it is selecting the wrong point, because
the pole constraint is applied after the inner solve instead of inside it.

Eleven of twelve inner solves ended on code 6. **Code 6 is not "maxit
reached"** (that is code 4; corrected in 17.5 below). It is a Wolfe line-search
bracket failure at a feasible point, and it ends the inner solve by pyGRANSO
option default. See 17.5.

Priority for stage 1, on this evidence: (1) the pole constraint into the inner
problem -- it is actively discarding better designs; (2) the quadprog fallback
of section 14, which did not fire here; (3) budget tuning.

**KGC+B: NO RECORD.** Not an open-loop result; no record was written at all.
The harness reported the row as missing. The job's `.out` is needed to tell a
crash from a skip; it has not been seen.


## 17.5 Job 16047382 dumps: what the gate refused, what code 6 means, KGC+B

### The refused rounds were closed-loop feasible

`outer_004.json` and `outer_007.json` of KSP+B+C:

| round | certified | gate alpha | alpha + eps | gate | refused for |
|---|---|---|---|---|---|
| 4 | 9.619574 | -0.0853 | -0.035 | feasible: false, located: 1 | controller pole |
| 7 | 9.650219 | -0.1037 | -0.054 | feasible: false, located: 4 | controller pole |

Both designs satisfy the stability margin comfortably. What the gate located
is a **compensator** pole in the right half-plane -- an unstable controller
with a stable closed loop. The rule "no unstable compensator" refused two
designs 1.8 % and 1.5 % better than the reported answer. Whether that rule is a
requirement of the problem or a convenience of the implementation is a stage-1
decision; either way it is that rule, not the eps constraint, that is costing
gamma on KSP+B+C, and it must be inside the inner problem if it is to be kept.

### Termination code 6, corrected

From pyGRANSO's `bfgssqp.py`: code **4** is max iterations; code **6** is "line
search bracketed a minimizer but failed to satisfy the Wolfe conditions, at a
feasible point". Round 4 above ended on code 6 after 39 iterations of a
200-iteration budget. Sections 15 and 17.4 read code 6 as maxit; that was wrong.

pyGRANSO's default `halt_on_linesearch_bracket = True` ends the inner solve on
the first such failure without consulting its fallback ladder (degraded
steering, plain BFGS, steepest descent). `run_case_epi` does not override the
default. With `opt_tol = 1e-10` against an eigenvalue oracle accurate to
perhaps 1e-8, a Wolfe bracket failure near the noise floor is close to
guaranteed -- KSP+B+C's rounds ended at `stat_value` 1e-5 to 2e-5, never near
1e-10. This is why the inner problem is never solved to stationarity and, very
likely, why balance residuals are large across the whole suite.

`--ls-fallback` added to `run_case_epi` (opt-in, default off, recorded in the
summary as `ls_fallback`): sets `halt_on_linesearch_bracket = False`.

**Measured at ne=100, case B, iters/maxit 25:** the ladder engages (191 calls
against 75) and returns the identical gamma to fourteen digits. So at that scale
the fallbacks spend 2.5x the work confirming that the bracket-failure point was
already the best reachable at the oracle's precision. This points away from
the halt option and toward the opt_tol / oracle-precision mismatch as stage-1
item 0's actual target. Whether ne=200 KSP+B+C behaves the same is the
experiment the flag exists for.

### KGC+B exits in 11 s with rc=0 and no record

Not the checkpoint (clean directory this time). `run_case_epi.main` has three
setup-phase refusals that print one `FATAL: ...` line and `return` -- exit
code 0. The likeliest is line 1476, "start design is infeasible ... the
continuation chain has no anchor": the cold start for KGC+B does not meet
eps = 0.05, so the run correctly refuses to begin and incorrectly reports
success. The harness was capturing `run_case_epi`'s output and discarding it,
so the one line that said which refusal fired was lost.

**Fixed in the harness:** each case's stdout/stderr is written to
`<workdir>/<tag>.log`; any `FATAL` line is echoed into the job output as
`*** run_case_epi REFUSED to run`; and an rc=0 finish in under a minute on a
multi-round run is called out as suspicious with its last five lines. Verified
by forcing the same refusal at ne=100 with `--eps 0.5`.

The refusal itself takes ten seconds and needs no allocation -- GL does not
touch MPI -- so it can be read on the login node:

    python run_case_epi.py --case KGC+B --start 0 --ne 200 --eps 0.05 \
        --cold --outer 1 --wall 60 --outdir /tmp/kgcb | grep -m1 FATAL

If it is the infeasible-start refusal: the frozen KGC+B record is `_s0_` and
reached 11.6009, so either its start was not the current init, or it was run
with a seed the cold path does not use. `case.slurm`'s start loop is where to
look.


# 18. Stage 0 closed

## 18.1 What Stage 0 was for, and its status

The handoff asked for a harness and a dump so later stages could be measured.

| deliverable | status |
|---|---|
| `harness.py` replay / run / determinism, section-7 row | done; 27/27 replay validated on the cluster |
| `run_case_epi.py --dump` per-round Omega, sigma_1, active set, multipliers, spectrum, LU counts, gate, PyGRANSO stats | done |
| `lu_counter.py` | done; 1 LU + 47 solves per sigma_1 evaluation |
| `plot_omega.py` | done |
| environment on compute nodes | done; 6/6 with one line in `env.sh` |
| determinism | settled: bit-identical across repeats, budgets, nodes and input builds |
| baseline for stage 1 | two rows at eps=0.05 (K+B+C, KSP+B+C) -- but see 18.2 |

## 18.2 What the baseline rows actually are

Both were produced with `--cold`. The suite protocol (`case.slurm`) cold-starts
only the four roots; every other case resumes from each of its parents, then its
field donors, then its own best. KGC+B's cold start is the open loop, infeasible
at eps=0.05 (reproduced in the sandbox in fifteen seconds; the frozen `_s0_`
record was reached from its parent KGC). And K+B+C cold reaches 9.82
(stationary) where the frozen K+B+C via its parents reached 7.96.

So the two rows are valid designs but not suite-protocol designs, and stage-1
comparisons must be made against a baseline built the suite's way.
`tools/lattice.py` now implements that protocol portably: the same script gives
an ne=100 development baseline on a sandbox and an ne=200 confirmation on a
cluster node.

## 18.3 Where the time went

Roughly half the cluster allocations since 2026-09-14 were spent on the
assistant's errors: two environment misdiagnoses, a tarball that overwrote a
cluster-built input, a half-applied patch, run logs captured and discarded, a
results directory reused with live checkpoints, and the `--eps` omission. The
last was necessary work once found -- the audit it produced also found the
same gap in the frozen `case.slurm` -- the rest was not. The common cause was
verifying by grep and assertion instead of by execution.

## 18.4 Handed to stage 1, in the order the evidence suggests

1. **The compensator-pole rule.** The gate refused two KSP+B+C designs 1.5-1.8 %
   better than the answer, both closed-loop feasible with margin, for an
   unstable compensator (section 17.5). Decide whether the rule is required;
   if so, it must live inside the inner problem.
2. **`opt_tol = 1e-10` against oracle precision.** Eleven of twelve inner
   solves end on code 6 (Wolfe bracket failure) at `stat_value` 1e-5; the
   inner problem is never solved to stationarity. `--ls-fallback` engages
   pyGRANSO's ladder and, at ne=100, spends 2.5x the calls for the same gamma
   to fourteen digits -- so the tolerance, not the halt option, is the target.
3. **Starts.** `lattice.py` restores the suite protocol; the ne=100 baseline is
   being built with it.
4. **Quadprog failure** (section 14): the UnboundLocalError skips the same
   fallback ladder by a different door. Not seen at eps=0.05 so far.
5. **Nesting gaps** (section 13-adjacent): KSP+FB and KSP+FC exceed their
   parents by 1e-4, masked by a stale 1e-3 tolerance in `collect.py`.
6. Budget tuning last; determinism is not a reason to change the default.

Stage 5's cost model still needs the measured 12-25x LU ratio, and a
factorisation-free oracle with different precision could make item 2 worse if
it is not fixed first. That ordering is the reason stage 1 precedes stage 5.

## 18.5 Working mode from here

Develop and measure at ne=100 in the sandbox; the pathologies above are
properties of the algorithm, not the mesh. Return to the cluster once per
stage, to confirm the winning settings at ne=200 on the user's catalogue. Every
change is executed before it is shipped, the invocation audit runs before any
job, and no new investigation is opened on a log unless it blocks the stage.
