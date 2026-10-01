# Archive cleanup — HANDOFF_implementation.md §4

Date: 7 September 2026. Applied to `frozen_20260826.tar.gz` plus the three
files that were missing from it (`seeds_ne200.npz`, `case.slurm`, `env.sh`).

`selftest.py` passes after the move — that was the stated condition.

---

## 1. Layout

| location | contents |
|---|---|
| root (17 `.py`) | production closure of `run_case_epi.py`; the certification stack; the gates; `ctrl_order.py`; `fe_pipeline.py` |
| `tools/` (16) | seeding, sweeps, plots, symmetry, diagnostics, launchers, `migration/` |
| `attic/` (11) | reached by nothing: benchmarks, one-off tests, abandoned tools. Moved, not deleted |

Root: `run_case_epi.py, certify_sparse.py, closedloop_sparse.py, controller.py,
fe_codesign.py, fe_gl.py, packer.py, parallel.py, warmstart.py,
winding_cert.py` (production) — `verify_dense.py, certify_results.py,
check_kkt.py` (certification) — `selftest.py, check_env.py, patch_pygranso.py`
(gates) — `ctrl_order.py, fe_pipeline.py`.

`tools/`: `cases.py, prepare.py, seeds.py, nesting.py, sweep.sh, case.slurm,
collect.py, check_band.py, check_symmetry.py, check_sym_pair.py,
plot_symmetry.py, plot_results.py, plot_bode.py, gauge_fields.py,
check_optimality.py, bounds.py`, plus `check_frozen_table.py`,
`validate_cleanup.sh` and `migration/`.

`attic/`: `ablate.py, bench_invit.py, bench_oracle_now.py, bench_par2.py,
bench_segcert.py, check_incomplete.py, eigcount.py, selftest_cert.py,
test_continuation.py, test_eigcount.py, test_substitution.py`.

## 2. Departures from the handoff's §4 lists

The import-graph classification in the handoff is correct as far as imports go
— an independent AST closure reproduced it exactly, and `attic ∩ kept-closure`
is empty. Four placements still needed changing, three of them because an
import graph does not see subprocess invocation.

**`fe_pipeline.py`: tools/ → root.** `selftest.py` imports it directly. A
top-level module cannot import from a subdirectory, so the handoff's placement
breaks the first gate. It belongs with the rest of `selftest`'s import closure.

**`check_optimality.py`: attic → tools/.** Reached by no import, but it is the
last step of `sweep.sh`'s post-processing chain and of report_v7 §9's
reproducibility chain (`collect → check_optimality --gbounds`). Filed beside
`collect.py` and `check_band.py`, which report §9 groups it with.

**`bounds.py`: attic → tools/.** It is the source of the three certified floors
(9.831865 sensor-limited / 9.660005 actuator-limited / 3.509066 absolute) that
report_v7 §3 quotes. Not deleted, because those numbers have to be
re-derivable.

**`winding_cert.py`'s `plant_pole_data`: no change.** The handoff lists the
`n > 3000` refusal as a discrepancy to fix. It is not one: the function raises
a `RuntimeError` that names `--p_plant` and the near-line plant modes as the
remedy, and the module docstring already says P_plant is dense at GL scale and
supplied by other means at cavity scale. Honest documented limit. The cavity
recipe stays algebra note §7 / to-do 3 → stage 7.

## 3. Changes to code

**`winding_cert.py` — the `--rigor` claim (algebra note to-do 5).** The
docstring advertised a rigorous per-sample Neumann segment bound "enabled with
`--rigor`". No such option existed in the parser and no such bound was ever
computed. Rewritten to say the bound is not implemented, that the flag never
existed, and that it is scheduled as the per-interval Neumann bound of pass 2
of the parallel winding scheme — algebra note item 5, merged into item 4,
**stage 6** — where it doubles as the test that closes quiet intervals without
further samples. Implementing it now would jump ahead of the stage that
validates it by replay.

**`ctrl_order.py` (new) — the r = 7 fallback, in eleven tools.** Every one of
`certify_results, check_kkt, verify_dense, winding_cert, check_band,
check_optimality, check_sym_pair, check_symmetry, gauge_fields, plot_results,
plot_symmetry` read the controller order like this:

```python
try:
    a.r = int(np.load(f'seeds_ne{a.ne}.npz', allow_pickle=True)['r'])
except Exception:
    a.r = 7
```

The intent is report_v7 §7's P4 fix, "derive from where it is decided". The
defect is the fallback. `seeds_ne200.npz` was not in the frozen archive, so the
`except` branch fired and all eleven silently adopted r = 7 while every one of
the 27 designs carries r = 8 — the exact hardcoded constant P4 was written
about. `verify_dense.py` happens to raise a shape error; the tools that index
rather than reshape would have produced quietly wrong numbers instead.

Replaced by `ctrl_order.infer_r(ne, results)`, which tries, in order:

1. **the design records** — `len(nvec)` per `.npz`. The most direct source: it
   is the order of the controller that was actually optimised and certified, it
   is per design, and it cannot drift from the record it describes.
   Disagreement between records is an error, not something to average over.
2. **the seed catalogue**, resolved against the repository root rather than the
   current directory.
3. **nothing** — raise. A missing order is a stop, never a default.

A records-vs-catalogue disagreement is also an error. Note that `prepare.py` at
`ne = 100` produces r = 7 seeds, so r really is a property of the design set
rather than a global constant — which is why source 1 is preferred.

**Root shim in 13 relocated tools.** Thirteen of the campaign tools import root
modules (`fe_codesign`, `controller`, ...). Several already carried
`sys.path.insert(0, dirname(abspath(__file__)))`, which inserted the root while
they sat in it and inserts `tools/` afterwards. A canonical root shim was
inserted above that line rather than rewriting it; the old line then merely
puts `tools/` on the path as well, which is harmless.

**Launchers.** `sweep.sh` now cds to the repository root (`dirname $0/..`) and
calls `python tools/nesting.py`, `sbatch ... tools/case.slurm`, and
`tools/{check_band,collect,check_optimality}.py` in its post-processing
message. `case.slurm` still cds to `$SLURM_SUBMIT_DIR` (the root) and calls
`python tools/nesting.py`; its `run_case_epi.py`, `verify_dense.py` and
`source ./env.sh` calls are unchanged because those are at the root.

**`selftest.py`.** Its sandbox copied only top-level `*.py`, so the relocated
`prepare.py` and `collect.py` never arrived. It now copies `tools/` into the
sandbox and invokes `tools/prepare.py` and `tools/collect.py`. The root shim in
those files resolves to the sandbox, because the top-level modules are copied
there too.

## 4. Verification

Every command below was run with **no `--r` flag anywhere**; the order was
inferred as 8 from the design records each time.

| check | result |
|---|---|
| `check_env.py` | numpy 2.4.4 / torch 2.14.0 / pyGRANSO 1.2.0 **PASS**. PETSc, SLEPc, dolfinx fail — not needed for GL |
| `selftest.py` | **ALL CHECKS PASSED — ready to sbatch** (rc 0). Kernels, 4 gradients vs FD, tridiagonal round-trip, Sobolev, certification decision, `prepare.py`, `run_case_epi` on B/K/FC, `collect.py` |
| `verify_dense.py` | 27/27 STABLE-valid, **0 INVALID**; dense abscissa matches every stored value. 24.0 s |
| `winding_cert.py` | 27/27 **CERTIFIED, count 0** (winding −1, P_plant 1, P_ctrl 0), 477–609 points/design. 6.4 s |
| `certify_results.py --rtol 1e-6 --band=poles` | **27/27 certified, ratio 1.000, 0 missed peaks** |
| `check_kkt.py --case K+B+C` | constraint INACTIVE (α 44.0 % of ε inside), ‖∇γ‖ = 3.826e+01 — report §5.4's terminal state |
| `tools/check_frozen_table.py` | 27 cases, **0 discrepancies** against the handoff §3 table (γ, s, calls, outer, \|Ω\|, stall flags), 19 notes |
| `tools/nesting.py --waves` | 27 cases in 5 waves |
| `tools/collect.py` | builds; K+B+C shows the P6 `TERMINATED: inner solver at its limit` message |
| frozen records | `diff -r` byte-identical. Nothing ran with `--update` |

`selftest.py` values at `ne = 100`, wall 25 s, for comparison against the
handoff §6 sandbox record: B γ = 10.230699181378927, K γ = 10.830551610038565,
FC γ = 9.687035779658093.

Re-run any time with `bash tools/validate_cleanup.sh`, which repeats all of the
above and checksums `results/` before and after to prove nothing was written.

## 5. Open items this cleanup surfaced but did not close

1. **`wind_count` is absent from 19 of the 27 JSON records.** `run_case_epi.py`
   writes it only under `--wind`; for those 19 the certificate came from the
   separate `sweep.sh` pass, which reports to `logs/` and does not write back.
   Report §4.3's claim is true — re-running `winding_cert.py` confirms count 0
   on all 27 — but it is not recoverable from the records. **Stage 0's harness
   needs a completed reference table, not the raw records.**
2. **Hand-typed bounds in `tools/plot_results.py:83`.** `BOUNDS` carries the
   three floors as literals with a comment naming `bounds.py` as their source.
   A live instance of P4's genus. Left alone deliberately — changing it changes
   a figure — but it should be derived from `bounds.py`.
3. **`bounds.py`'s constants are tied to `xa = −1.03, xs = 0.98`** in that same
   comment, with no check that the records still use those positions.
4. **The frozen archive was not self-sufficient.** `seeds_ne200.npz`,
   `case.slurm` and `env.sh` were all absent, and `sweep.sh`/`case.slurm` could
   not run without them. Whatever produces the next frozen tarball should
   include them; report §9 claims the archive contains the complete toolchain.
5. **`--rigor`** — implement in stage 6, per §3 above.

**Status 2026-09-26 (STAGE5 30.11):** 1 closed -- `results/reference_table_ne200.csv`
(winding counts of all 27 records at the frozen line -eps/2, all 0, written by
`winding_cert.py --csv`) and `results/reference_table_ne200_margin.csv` (at the
margin -eps itself: all 0 with the exact-compensator count; the plain count was
inconclusive on 4); 2 closed -- `tools/plot_results.py` reads the bounds from
`results/bounds_ne200.json`, written by `tools/bounds.py --json` (grid of 1401
points: sensor-limited 9.831865, actuator-limited 9.660137, floor 3.509193 --
the literals were 9.831865 / 9.660005 / 3.509066 from coarser grids); 3 closed --
the file carries the positions and the plot labels each bound with them (they
hold for designs with the sensor at xs = 0.98, resp. the actuator at
xa = -1.03; FC's 9.660135 sits on the actuator bound within the frozen
certificate's tolerance 1e-4); 4 closed -- seeds_ne200.npz, tools/case.slurm,
env.sh are in the repository and the packages; 5 -- stage 6.

## 6. Migration scripts

`tools/migration/cleanup_archive.py` and `tools/migration/patch_ctrl_order.py`
are the record of what was done. Both are idempotent and re-running them from
the repository root is a no-op.

`patch_ctrl_order.py` is line-wise and bounded, not a regex. The first version
used a multi-line regex whose non-greedy body matched an unrelated `try:` 110
lines earlier in `plot_results.py` and deleted `CASE_STYLE`, `BOUNDS` and the
whole argparse block. It was caught by diffing every patched file against its
backup, and every patch is now a −2 to −4 line edit.

## 7. Dead code after negative tests (2026-09-25/26)

Removed from `sqp_inner.py`, `hurwitz.py`, `run_case_epi.py` (all off by default,
all negative or superseded; STAGE5 24-29): MERIT_CTRL, HMIN_REL_TOL, NU_H,
TR_BALL, CTRL_CUSHION, CTRL_HESS, SOC_CTRL, SOC_MAXIT, TR_SHRINK, H_SCALE, MERIT
and the driver's `--sqp-merit`; `_smat`, `comp_hermite_at`, the exact-hmin
helpers, the compensator-Hermite branches of the merit and admissibility tests,
the SOC and exact-curvature blocks, `hurwitz.hermite_block_hessian`, the stats
keys n_soc / n_ctrl_hess / e_last_hermite. Verification: replay of one round
with identical iterate history before/after (md5), frozen path
10.23128516441147, Hurwitz selftest 0/200, `tools/check_invocations.py` (no
script passes a removed flag). 2026-09-26: two further tests (sigma_2 blocks
inside the solve, trust-radius shrink off the face) were removed after testing
(STAGE5 29.2); the diagnostics that found them stay behind `DEBUG_LS`.

## 8. 2026-09-28: field regularisation, item 11 (STAGE5 32-33)

Added (all in use or validated options): `sqp_inner` field shape rows
(`FIELD_KMAX2`, `FIELD_ORDER`; driver `--field-kmax`, `--field-order`);
atoms mode in `packer` / `fieldbasis` (`--field-atoms`, `--atom-smin`,
`--atom-kmax`); `fieldbasis.p2_eval`; `run_case_epi.continuation_gate_reduced`
(`--cont-track reduced`); a per-round RMS-wavenumber line for field cases.
Tools: `gate_replay.py`, `field_study.py`, `band_study.py`, `mode_response.py`,
`field_distance.py`, `cross_eval.py`, `fields_mesh.sh` (the runs of STAGE5 32).
Changed: `fieldbasis._m_orth` order-preserving (the SVD version scrambled the
modes), `laplace_basis` clamps N; `transfer_ne.py` evaluates the P2 field
instead of interpolating linearly and names its output after the target ne;
`n100_kbc.sh` takes OUT / EXTRA; `final_curves.py` shows the atoms design
(ne=400) instead of the withdrawn 80-mode one, and reuses computed curves.
Kept after a negative result, with the reason: `--field-order 1` (H^1) is too
weak at k <= 1 as the only regulariser but is the nesting row of the atoms
runs and the natural form for other kmax; the unconstrained modal fields
(`--field-basis laplace` without `--field-kmax`) remain the input basis of the
reduced model in every field run. Nothing removed. Checks: frozen path
10.23128516441147, witness invariant 0, Hurwitz selftest 0/200.

## 9. 2026-09-29: M products only; MIMO prototype (STAGE5 34)

Changed: `fieldbasis._m_orth` (sparse M), `fieldbasis.laplace_basis` (sparse
shift-invert Lanczos; dense only when every mode of a tiny mesh is asked for),
`fieldbasis.fit_atoms` (M-weighted objective and gradient, no Cholesky),
`tools/tail_certificate.py` and `winding_cert._plant_tail_constants` (Lanczos
with a sparse LU of M instead of dense Cholesky/SVD; the power-iteration x 1.3
branch removed). Added: `tools/mimo_proto.py`, `tools/mimo_plots.py` (prototype,
standalone; the production loop is untouched). Remaining dense uses are GL
reference tools, listed in STAGE5 34.2. Nothing else removed.

## 10. 2026-09-29: MIMO in the production loop (STAGE6, branch `mimo`)

Added: `closedloop_mimo.py`, `PackerMIMO`, `tools/mimo_start.py`, `tools/mimo_n100.sh`,
`tools/check_mimo.py`, `tools/check_siso_identity.sh`, `tools/mimo_prod_plots.py`,
`FECoDesign.cert_region_mimo`. Changed (SISO paths untouched, identity checked):
`certify_sparse` (`_cl_arow` dispatch, rank-2(m+p) pencil), `winding_cert` (`_FM`, MIMO
tail, deterministic Lanczos start), `reduced` (MIMO proxy and pencil, no eigenvalue
tracking for MIMO, deterministic `svds` fallback start), `fieldbasis.laplace_basis`
(deterministic start and sign), `tools/tail_certificate.py` (deterministic start),
`sqp_inner` (`exit_info`), `run_case_epi` (MIMO flags and records, `--wind-accept`,
gate winding fallback, `--band-first` for MIMO, dense all-w save check).
Removed: the two inline copies of the winding-count call in the driver (continuation
refusal, acceptance), now one helper `_wind_right_of_margin`. All temporary debug
prints of the diagnosis (line-search trace, pair residuals, candidate dumps) were
removed before the commit; none remain.
Runs kept as evidence under `results_mimo_prod/*_try*` (each named after what it
showed; STAGE6 6.3-6.8).

## 11. 2026-09-29 (second pass): dense out of the loop (STAGE6 6.12-6.19)

Added: `sparse_spectrum.py` (plant enclosure, closed-loop located set, tail bound),
`tools/dual_design.py`, `run_case_epi.py --force-mimo --sqp-stat`, `tools/mimo_start.py --one`,
`PackerMIMO.blocks`, `reduced._batch_setup_mimo/_Tt_mimo/_TtH_mimo`.
Replaced: the dense `band_from_closedloop` (now sparse), the dense default of
`winding_cert.plant_pole_data` (now `method='sparse'`, dense kept as the reference
`method='dense'`), the dense dumped spectrum, the `D.n <= 4 x dense_nmax` gates of the
winding plant data (removed: sparse at any n), the MIMO per-row fallback of
`reduced.gain_batch` (removed: batched), the MIMO packer's per-group scale rule.
Kept on purpose (validation at low dimension): `FECoDesign.alpha_dense` (anchor check,
save veto), the dense all-w level set at save (n + r <= 1000), `dense_reference`.
The temporary profiling and debug hooks of the diagnosis (py-spy runs, the
line-search traces) lived in scratch files only.
Later in the pass: the MIMO packer's per-group scale floor restored for m + p > 2
(11c4073, 6.22); `tools/check_mimo.py` check 3 per-entry FD step (replaces the
per-block step); the run directories superseded by the final records kept as
named diagnostics (`*_c4cf908_*`, `*_diag*`, `prev_0e7f1c1`), no unnamed leftovers.

## 12. 2026-09-30: parallel rows, QP, order-5 start (STAGE6 6.25-6.27)

Added: `reduced._RowPool` / `_row_pool` / `_row_worker` and `--row-workers`
(forked workers per basis version), `reduced.gain_batch(..., k)` +
`_batch_pairs`, `FECoDesign.Mdpsi_dx` (memoised), `tools/reduce_compensator.py`
(closed-loop weighted balanced truncation), `mimo_n100.sh kbc_cenR`.
Replaced: `lu_counter._caller` (frame walk; same site strings), the per-row
sigma_2 blocks and their per-row peak following (batched; the per-row path stays
for unbatched evaluators).  QP: `max_threads = 1`.
Tried and removed (6.26): constraint generation in the QP subproblem
(`_solve_generated`), a `--qp-linsolve` flag (Clarabel's default is already
faer).  Removed from the first version of reduce_compensator: plain balanced
truncation (destabilised every order <= 8).  Scratch only: the QP dump hook,
the synthetic checkpoint builder (`mkck.py`), benchmarks.

## 13. 2026-09-30 review fixes (STAGE6 6.28)

Added: `reduced._fit_subspace`, `sparse_spectrum._m_lu` / `_MLU_CACHE` /
`_SFAR_CACHE`, `FECoDesign._Hlu` (lazy property; `_Hlu_f`).  Replaced: the
eager Sobolev and pencil-M factorisations, the per-call M LUs, the silent
`ev = [0]` fallback of `stability_region`.  `tools/check_siso_identity.sh`:
exact identity -> agreement to 1e-7 (identity still reported).  Gated: the SISO
warm-start search (`not MIMO`).


## 14. 2026-09-30: the waste that grows with n (STAGE6 6.29)

**Added.**
- `massops.py`: `lu`, `lu_real`, `block_solve`, `warmup`.
- `sparse_lu.py` (`factor`, `ORDER`; driver `--lu-order`).
- Reduced-space rows (`_ProxyCL` / `_ProxyCLMIMO`):
  - `_reduce`, `lift`, `placement`, `grads_reduced`, `_fields_reducible`,
    `_field_row`;
  - `ReducedOracle._full_residual` and the Gram matrices in `_project`;
  - `Packer.field_jacobians`, `fieldbasis.atom_jacobian`.
- `reduced._TriStack`, `reduced.ORTH` with `_orth_block` / `_orth_column`
  (driver `--oracle-orth`).
- `certify_sparse`:
  - `_climb_slope`, `NEAR_K`, `CLIMB` (driver `--cert-near-k`, `--cert-climb`);
  - `register_plant` / `_ship` / `_land`.
- `sparse_spectrum`: `numrange_distance(..., line)`, the `memo` argument of
  `closed_loop_right_of` / `closed_loop_set`.
- Driver:
  - `_sigmas`, `_pin_lus` / `PIN_MAX` / `D._lu_pin`, `_plant_warmup`;
  - `_band_memo`, `_wind_best`, `_allw_best`, `_eig_done`;
  - the gate guard `_acceptable`;
  - the save-time winding implied by the acceptance count or the final band
    check.
- Tools:
  - `tools/ab_check.sh` (A/B with LU counts by callee file; `AB_CASES`);
  - `tools/check_reduced_pairs.py`, `tools/check_reduced_fields.py`.

**Replaced.**
- `sparse_spectrum._m_lu` / `_MLU_CACHE` (6.28) -> `massops`.
- The per-site M LUs in `winding_cert`, `sqp_inner` and `fe_codesign`
  (smooth_unmap, cert_region_mimo).
- Every `spla.splu(...)` of a full-model matrix -> `sparse_lu.factor`.
- The lifted row pairs -> reduced ones.
- The full `_Tstack` array -> `_TriStack`.
- Brent-only witness climb -> selectable.
- The all-w level set on [-W, W] -> on the two outer intervals.
- The 9-call sigma_2 block -> 6 calls.
- The column-by-column `_orth` -> the block version by default (the column
  version is kept as the reference for `check_siso_identity`).
- In the save section, the final band check now runs before the winding
  certificate.

**Removed.**
- The no-progress round's full certificate of the best (oracle path).
- `D.hinf` at save (oracle path).
- The start's second 241-point sweep.
- The duplicate eigen-enrichment after a compaction.
- The per-call `_Tstack` construction.
- A and M in the pool task tuples.
- The first blocked `_orth` with a pivoted QR (it decided differently at the
  1e-8 deflation margin).

**Fixed in the tools.**
- `tools/check_siso_identity.sh`: the reference tree's loop drops the flags it
  predates.

## 15. 2026-09-30: parallel full-model tasks (STAGE6 6.30)

**Added.**
- `fullpar.py`: `start`, `stop`, `pmap`, `sigmas`, `report`, the stamp check, and the lu_counter merge.
- `winding_cert`: `_march` (the old in-line march, moved into a function unchanged), `_h_hot`, `_make_F`, `_seg_task`, `SEG_PER_WORKER`.
- `certify_sparse`:
  - `certify_greater` rewritten around `merge()`. Its serial branch is the old loop body; the generation branch is new.
  - `_shift_eval`, `_shift_task`, `_par_ok`, `_pencil_key`, `CERT_SPLIT` (driver `--cert-split`);
  - `hinf_certified_sparse(sigma_many=...)`.
- `reduced`:
  - `enrich_columns`, `enrich_many`, `enrich(cols=, project=)`, `enrich_eig(bc=)`, `compact(bc=)`;
  - `_enrich_task`, `_eigvec_task`;
  - `UNION` / `_union_k0` (driver `--oracle-union`).
- `sparse_spectrum`: `_near_eval`, `_near_task`, `_pencil_of`, `_tile_task`, `_rect_task`, `eigs_in_rectangle(spec=)`.
- Tools:
  - `tools/check_parallel.py`, `tools/bench_parallel_2d.py`;
  - `tools/ab_check.sh` `AB_EXTRA`;
  - `results_parallel/bench_parallel_2d.json`.

**Replaced.**
- The driver's own `multiprocessing` pool. It is now fullpar's, forked at warm-up and again before the rounds. The continuation gate and the abscissa certificate segments use it through `fullpar.pmap`, so their LUs are counted.
- `_enrich`'s per-anchor `enrich` calls -> `enrich_many`.

**Removed.**
- `ReducedOracle.enrich_shift`: never called (dead since the input-basis enrichment).
- The per-anchor projection inside `enrich_many`: once per batch now.

**Tried and not kept as a default.**
- The union step on W's new columns only: not bitwise (per-column GEMM results depend on the column count). It is the `--oracle-union new` option instead.
- More winding segments per worker: 4 is the best of 4 / 8 / 16 (516 / 524 / 550 points, 4.9 / 5.0 / 5.5 s at n = 6400).

## 16. 2026-10-01: the lattice check (STAGE6 6.31)

**Added.**
- `tools/lattice_check.py`.
- `results_lattice_check_ne200/` (records, report, state, compressed logs).
- `certify_sparse`:
  - `SparsePencil.free_radius`, `N_DISK_FREE`;
  - `_shift_eval(free=)`;
  - the strict first-Arnoldi cap (300).
- `fieldbasis._rho_from_halfwidth`; `fit_atoms` now tries both starts (`_fit_atoms`).
- `reduced`: `_pencil_solve0`, `BATCH_BYTES` chunking, `SVCACHE_MAX`.

**Changed.**
- `run_case_epi`:
  - the certificate settings come before the start certificate;
  - a newly freed field starts from the parent's field;
  - a fixed parent controller is the default start;
  - `_sigmas` uses maxit 200.
- `sparse_spectrum._near_eval`, `closedloop_sparse.eigvecs_at` and `reduced._pencil_solve`: shifts on a compensator pole are moved.
- `reduced.gain_batch`'s cache stores copies, not views.

**Removed.** Nothing.
