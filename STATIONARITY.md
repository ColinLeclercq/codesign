# BFS Clarke stationarity: the measure, its review and the readings of six records (2026-10-17)

> Written by the review of `tools/bfs_stationarity.py` (commit after 3b1ec04). Paths marked (session scratch) were working files of that session and are not kept; the numbers are reproducible by running the tool (tools/stationarity.slurm on the cluster). LESSONS.md 21 holds the lessons.

## 0. Bottom line

None of the six records is Clarke-stationary at the certificate's resolution. For each one the tool found a design with a strictly lower certified gamma one linear step away. It is admissible on the line test that the SQP's line search and the gate use; the winding count over all w was not run. The six certified bounds from the testers' runs were reproduced exactly with the reviewed code.

The records differ in how close they are to ε-stationarity:

- **Central c151:** near-equiripple, 13 tops within 4e-4 of gamma.
- **Decentral 369.87:** close behind, 9 tops within 6.1e-4.
- **Decentral 412.28:** further off.
- **SISO 598.11, SISO 554.97 and central c177:** no cancellation at all until ε ≈ 1.4e-2.

| case | record (stop) | ρ₂/Σλ\|g\| at ε = 1e-6 / 1e-3 / 1e-2 | worst group-wise condition (scale-free) | ratio at smallest Δ | certified γ(z+d) | decrease |
|---|---|---|---|---|---|---|
| SISO | 598.106063, no strip (progress rule; code 0 under `rel`, stat 6.1) | 1.0 / 1.0 / 0.98 | K 1.0 at 1e-2 | 1.00 at 1e-7 | ≤ 593.161757 (Δ 1e-5) | 8.27e-3 |
| SISO | 554.969636, strip (progress rule; code 0, stat 12.65, 1 iteration) | 1.0 / 1.0 / 5.7e-2 | n 0.093 | 1.00 at 1e-7 | ≤ 549.905505 (Δ 1e-4) | 9.13e-3 |
| decentral | 369.868896 (deadline, code 4) | 1.0 / 1.1e-3 / 4.5e-4 | Cc 2.3e-5 | 1.00 at 1e-6 | ≤ 369.744507 (Δ 1e-5) | 3.36e-4 |
| decentral | 412.284543, strip (deadline, code 4) | 1.0 / 0.38 / 5.2e-3 | Cc 2.7e-4 (1.0 at ε = 1e-3) | 1.00 at 1e-6 | ≤ 411.945579 (Δ 1e-4) | 8.22e-4 |
| central | c151 123.904247 (progress rule 3:0.0015, code 4) | 1.0 / 6.0e-5 / 6.0e-5 | Cc 2.5e-4 | 1.00 at 1e-6 | ≤ 123.887235 (Δ 1e-5) | 1.37e-4, below the current rule's per-round scale |
| central | c177 153.615619 (deadline, code 4) | 1.0 / 1.0 / 4.4e-2 | Bc 0.66 | 1.00 at 1e-6 | ≤ 152.378057 (Δ 1e-3) | 8.06e-3 |

How to read the table:
- **Ratio at the smallest Δ** is measured over predicted decrease. A ratio of 1.00 with the predicted decrease linear in Δ means the directional derivative is negative. That, not the finite step, is what proves non-stationarity.
- **ε = 1e-6** is the certificate's resolution. On every record only the top piece is active there, with |g| between 5.6e3 and 3.1e5, so ρ = 1 means no cancellation.
- **Worst group-wise condition:** a value near 1 proves the record is not ε-stationary at that ε, whatever the packer's scales.

## 1. What the tool measures

**The problem.** It minimises f(z) = max over w in [0, whi] of σ₁(T(z, iw)), subject to:
- every closed-loop eigenvalue Re λ ≤ −ε_s;
- the modal compensator rows v ≤ L = −0.05 (per 2×2 block: tr/2 and L − det(M − LI)/s; per 1×1 block: its pole);
- the port position rows.

z is the SQP's own packed vector: the record's packer, real modal parametrisation, positions included. The trust box, displacement cone and elastic slacks are solver devices and are left out.

**Pieces.** These are the local maxima in w within ε_pred = 0.05 of gamma, ends included. Seeds come from the record's certified 2^-6 curve and its peakset. Newton (`peak_tops`) locates each top, edge stops are retried, and a top beyond whi becomes the end point. Rows come from one `sqp_inner.Evaluator` call at the tops.

**Blocks.** A top whose σ₂ is within ε_pred gets the 2×2 Clarke block, an SOC with PSD Y and the cross terms. If σ₃ also qualifies, an r×r PSD block is built on the driver.

**Constraints.** All closed-loop rows the line exhibits (the Hermite block for a near-defective pair), all modal rows, and all position rows. Any eigenvalue the line saw without a row is flagged.

**Residual.** ρ_q(ε) = min ‖Σλg + Σ⟨Y,dG⟩ + Σμh‖ over the ε-active pieces and the ε-active normal cones, for q = 2 and q = 1 (the dual of the box). It is reported:
- over the mean |g_i| (the GL convention);
- over min |g_i|;
- over Σλ|g_i| (the minimiser's own weights; the verdict's number);
- as the cancellation ratio ρ/(Σλ|g| + Σμ|h|);
- per variable group;
- as the group-wise necessary condition (scale-free, see §3.5);
- with the MFCQ margin.

The ε-sweep is the `--eps` list plus a profile just past every piece's and every constraint's entry level.

**Predicted decrease.** pred(Δ) is the SQP's box LP at H = 0 (Polak's optimality function with a box) over P(0.05), in gamma units, with and without constraints, plus a ball variant and a value at the record's trust radius.

**Descent test.** The step z + d* from the LP:
- holds the closed-loop rows on the QP's line;
- applies the SQP's own modal correction;
- re-locates the tops;
- tracks every closed-loop row that could reach −ε_s − mg;
- checks the compensator and the positions.

Δ_v is the largest Δ with measured decrease ≥ pred/2 that is also admissible.

**`--descent certify`.** It adds:
- the stability line clear, with α ≤ −ε_s − mg on [0, whi_t + wpad];
- the level set at the step's max × (1 + 1e-6) on [0, whi_t], where whi_t is the band the run would use at the step;
- regions found by the level set are located and the level set asked again;
- the certified design is saved.

**Verdict.**
- With a certificate: "NOT STATIONARY (certified …)" with the size of the decrease.
- Otherwise the class comes from the measured decrease at Δ_v: ≤ 1e-6γ "stationary at the certificate's resolution"; ≤ tol/k of `--progress` "near ε-stationary"; above that, "NOT STATIONARY (measured)".
- Every verdict also gives "stationary to ρ at ε", the ε = 1e-6 (Clarke) reading, the profile, the completeness status and the stop reason. It never says converged, optimal or minimum.

## 2. How to run it

**Sandbox** (at most 2 ranks):
```
export PATH=(session scratch) mamba/fx/bin:$PATH OMP_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd /home/user/codesign
mpiexec -n 2 python -u tools/bfs_stationarity.py RECORD.npz --ops bfs/data/coarse-4892cd08ee68e9d2/ops_coarse.npz \
   --eps 1e-4,1e-3,1e-2 --fd 2 --descent certify --descent-deltas 1e-6,1e-5,1e-4,1e-3 --json OUT.json
```
Options:
- `--strip on` for records of strip runs (20261010g `..._lat_s0_r300`).
- `--complete on` to certify the piece set.
- `--progress 3:0.0015` to judge a 20261010g/n run by its own stop rule.
- `--selftest` runs the convex-core toys (27/27).

Cost on 2 ranks:
- 14–31 min per record with `--descent certify`.
- About 20–30 min with `--complete on --descent off` (SISO 598: 1162 s; decentral 369.87: 1775 s).

**Cluster.** There is a template at `(session scratch) wf_stat/final/stationarity.slurm`; copy it to `bfs/slurm/` if the tool is adopted. It sources `env.sh`, runs `version_check`, and defaults to 24 ranks and the test plant's operators via `bfs/plant_id.py --dir coarse`. Submit it from the per-version worktree `BFS/v/<label>` that `bfs/stage.sh` prints:
```
REC=<run dir>/K_B_C_s0_bfs_mimo2x2_central.npz sbatch bfs/slurm/stationarity.slurm
```
Use `STRIP=on` for strip records, `COMPLETE=on` to certify the piece set, and `EXTRA="..."` for other options. Check the log's "gamma at the tops … agree" line: that is the plant-identity check, and the sandbox reproduced every record to within 4e-14.

## 3. Mathematics checked against the literature

1. **Maximum over w** (Danskin; Clarke 1983, Thm 2.8.2; Apkarian & Noll 2006, IEEE TAC 51:71–86).
   - With finitely many active frequencies, f is Clarke-regular and ∂f = co ∪ over active w of ∂_z σ₁(T(z, iw)).
   - Danskin needs no smoothness of the top's location, so the gradient at fixed w is correct for interior tops and at the hard band end whi.
   - At w = 0 it is also correct: for a real plant σ₁ is even in w, so w = 0 is an interior critical point of the symmetric extension.
   - Gap: when whi itself depends on z (c151: whi = |Im λ_c| + wpad), an end piece would also need σ′(whi)·∂whi/∂z, and that term is nonsmooth because of the ratchet. This is not implemented. No record has an end piece.
2. **Multiple singular value** (Overton 1992; Lewis & Sendov 2005; Boyd & Balakrishnan 1990).
   - ∂σ₁ = {U_r Y V_rᴴ : Y ⪰ 0, tr Y = 1}, and the directional derivative is λ_max(Herm(U_rᴴ T′ V_r)).
   - Checked algebraically: `Gc[a,b] = z_aᴴ dT x_b`, `hsym`, the SOC(4) parametrisation (Y ⪰ 0 ⟺ y₀ ≥ |(y₁, y₂, y₃)|), the r×r PSD cone, and the block row of the LP.
   - Toys 4 and 5 show the cross terms change the answer.
   - The ε-block (σ_k ≥ γ(1 − ε)) is the spectral-bundle ε-subdifferential (Oustry 2000; Apkarian–Noll's enriched multiplicity).
3. **The ε-active set.** It uses tops only, as Apkarian–Noll's extended frequency set does, not Polak's whole level set {w : σ₁ ≥ f − ε}.
   - So ρ_tops(ε) is an upper bound of the level-set version; both tend to the Clarke set as ε → 0.
   - A flat plateau (a continuum of maximisers) is outside this model. None was seen: every top converged with |σ′| ≤ 1e-7σ.
4. **Constraints** (Clarke 1983, Thm 6.1.1; Rockafellar–Wets 1998; Curtis & Overton 2012).
   - The Fritz–John condition reduces to the KKT form under MFCQ. For inequality rows MFCQ means 0 ∉ conv{∇c_j} (Gordan), which is exactly the tool's CQ margin. It is 0.51–1.0 on all six records, so the multipliers are bounded.
   - Closed-loop rows are smooth only at simple eigenvalues; the abscissa is non-Lipschitz at defective ones (Burke & Overton 2001). The Hermite block covers coalescence.
   - The modal rows are an exact smooth description of Re λ ≤ L for a real 2×2 block (Routh–Hurwitz). MFCQ fails only when the block equals L·I. v2's frozen scale is exact where v2 is active.
5. **Metric.**
   - 0 ∈ ∂f + N is invariant under linear changes of coordinates.
   - The magnitude of ρ, pred(Δ) and Δ_v are not: they live in packed z, with the record's scales and the 2-dimensional similarity gauge left in each modal 2×2 block.
   - The group-wise condition (the inclusion projected onto one group's coordinates) is invariant under any per-group rescaling.
   - The measured decrease of a given step, and the existence of a certified lower design, are coordinate-free.
6. **What a certificate proves.** A lower admissible design at a finite step does not by itself contradict stationarity. Non-stationarity follows from f′(z; d) < 0: the ratio ≈ 1 at the smallest Δ, equivalently ρ₁(ε → 0) > 0, since pred(Δ) = Δρ₁ for small Δ by LP duality. Both hold on all six records, and the docstring now says so.

## 4. Code against the oracle

- **The Evaluator gets the SQP's own rows:**
  - σ₁ at the followed tops;
  - blocks at the row point;
  - `al_rows_all` (every eigenvalue within ROW_REACH that the line exhibits);
  - `modal_rows` at max(−eps, ctrl_max) = −0.05, with `ctrl_block` forced off for the modal compensator as `_solve_epigraph_sqp` does;
  - `pos_rows`;
  - the line band `D.band_of = whi + wpad`, as at `run_case_epi` 3563;
  - `LAT_STRIP` only for strip records.
- **The packer is the run's packer.** The modal flag comes from the zero cut entries, and the real parametrisation and positions are included. NZ equals the record's NZ − 1 on all six, because the record also counts t. The central B_c gauge entry is the record's largest per row and may differ from the run's part start: the stationarity property is the same, the slice and metric differ.
- **Finite differences** are rerun every time. In the final runs:
  - pieces: Richardson relative error ≤ 1.3e-10 (SISO 598) and ≤ 5.9e-7 (c151);
  - constraint rows: all pass (38/38 on SISO 598);
  - the testers' step studies: truncation scales as h² on the stiff coordinates; block cross terms checked by `block_fd.py`.
- **Convex core:** an independent NNLS and HiGHS re-solve of the final SISO 598 JSON agrees to ≤ 1e-12 for ρ and ≤ 1e-7 for pred.
- **All active constraints are included** within the line's band. Eigenvalues above whi + wpad are not seen; neither the SQP nor the tool sees them.

## 5. Defects found, by severity

| # | severity | defect | status |
|---|---|---|---|
| 1 | high (test coverage) | The committed tool (63f6fd9) contains a first review pass (the ρ₂/Σλ\|g\| verdict pick, group-wise condition, MFCQ margin, band clipping, walked-top σ₂ block, warning for eigenvalues without a row) that had never run end to end; it was killed at 04:01 before the certify step. The SISO results came from a version without four later fixes (closed-loop tracking in the descent test, the SQP's modal correction, σ₂ blocks over P(ε_pred), one row per top). | Re-ran all six records with the final code: no crash, all six certified bounds reproduced to the printed digit. |
| 2 | medium | `--complete on` watched regions by value at tol 1e-9. A region around a known top whose estimate overshot the top by more than 1e-9, or was seen more than 1e-3 from it, came back as "region", re-located onto the known top, and the loop stopped there, with the rest of the band unexamined and the piece set never certified. On 369.87 the first call reported 1.91489 (3.3e-3 from the top at 1.918226) with 2.45 of the band uncovered. | Fixed: watched by location, and a re-located region is watched at its estimate in the next call. 369.87 is now certified at γ(1 − 1e-2) and γ(1 − 1e-3) after 5 calls; SISO 598 after 3. Neither found a new top. |
| 3 | low–medium | The certificate's level set used the record's band [0, whi], not the band the run would use at the step, max(whi, max\|Im λ(A_c)\| + wpad). c151's band end is its compensator pole (21.924i + 3). | Fixed. It triggered on c151: band 24.923680 → 24.923811, certified on the wider band; the bound is unchanged. |
| 4 | low | The class boundary was 5e-4, from the superseded 3-round stop rule. The BFS rule since 20261010r is 8:0.0015, 1.9e-4 per round. Decentral 369.87's 3.4e-4 read "near ε-stationary" although the current rule would keep going, and its 20261010n continuations did go 0.7 % lower. | Fixed: `--progress k:tol`, default 8:0.0015, printed in the verdict. 369.87 is now "above"; c151 stays "near". |
| 5 | low | The certified design was written to no file. | Fixed: it goes in the JSON and in `OUT_certified.npz`. Saved for 369.87 and 412.28; the earlier four runs predate this change and need a re-run to save theirs. |
| 6 | diagnostic | The ε-profile lacked the scale-free group-wise condition. | Added. |
| 7 | low (not fixed) | Seeds are the grid maxima and the peakset only; the record's active frequencies (Ω) are not used. The 20261010g records have no peakset. | Documented. Use `--complete on`, which now works. |
| 8 | low (not fixed) | No ∂whi/∂z term for end pieces; tops walked past whi by the Evaluator are not clipped. | Not triggered on any record. |
| 9 | info | The closed-loop Hermite block and blocks with r ≥ 3 are exercised only on toys. | Not triggered. |

## 6. Corrections to the three test reports

- **SISO 554.97 (strip).** "The 1e-5 and 3e-5 steps crossed the line and were rejected": no longer true. With the SQP's modal correction applied (a 3.8e-6 move in z) both are admissible. The certified bound is unchanged.
- **Decentral 369.87.** "Near ε-stationary at the progress rule's scale" holds only under the old 3-round rule. Under 8:0.0015 the certified decrease of 3.4e-4 is above the per-round scale.
- **Central c151.** "5.1e-5" is the mean-normalised residual. The other normalisations: 6.0e-5 by the λ-weights, 1.2e-3 by the smallest row, and 2.5e-4 for the scale-free group-wise condition. The weights are spread over all 13 tops (0.0035–0.23), so the mean is not misleading here, unlike the GL KSP+B+C case.
- **"ρ is an upper bound (complete off)".** For 369.87 and SISO 598 the piece set is now certified at ε = 1e-2 and 1e-3, so their ρ at those ε are the actual values, except for a possible second top inside a known top's region.

## 7. Results by case

### SISO
**598.11, no strip** (stop: progress rule; last round code 0 under `--sqp-stat rel`, stat 6.1, 6 iterations).
- Agreement: gamma at the tops within −3.7e-14 of the record; α −0.0501307 equal.
- Tops: 5 within 5 %, at 1.7012 (top), 0.9441 (−6.5e-3), 0.5221 (−8.0e-3), 0.2229 (−1.0e-2), 1.3732 (−1.4e-2). No σ₂ blocks.
- Active constraints: the pair −0.05 ± 2.593i (v1), a real pole at −0.05 (v2), and the closed-loop row −0.050131 + 4.2007i (enters at ε = 2.6e-3). MFCQ margin 0.54.
- Residual: ρ/Σλ|g| = 1.0 for ε ≤ 1e-3 and 0.98 at 1e-2. The mean-normalised 0.19 there is misleading. Groups K and e are at 1.0 at ε = 1e-2: the e and K coordinates alone show it is not ε-stationary there, whatever the scales. ρ/Σλ|g| reaches 2.0e-3 only at ε = 1.4e-2 (5 tops).
- Descent: ratios 1.00, 1.00, 0.99, 0.94, 0.90, 0.88 for Δ = 1e-7 to 1e-4. Closed-loop rows tracked at −0.05015 to −0.05010, all admissible.
- Certificate: ≤ 593.161757 at Δ = 1e-5. `--complete on`: no missed top.

**554.97, strip** (stop: progress rule; code 0, stat 12.65, 1 iteration).
- Tops: 5 within 5 %, at 0.1377 (top), 1.0991, 1.3355, 0.8124, 0.3288.
- Active constraints: the pair −0.05 ± 1.534i, and a double real pole at −0.05 (v1 and v2 both active). MFCQ margin 0.51.
- Residual: ρ/Σλ|g| = 1.0 for ε ≤ 1e-3, 5.7e-2 at 1e-2, and 1.7e-3 at ε = 1.41e-2.
- Descent: ratios 1.00 down to 0.99.
- Certificate: ≤ 549.905505 at Δ = 1e-4.

### MIMO decentral
**369.87** (stop: deadline, code 4 in every round).
- Tops: 9, all within 6.1e-4. No blocks.
- Active constraints: pole (9) at −0.05, the pair (5,6) at −0.05 ± 2.293i, and the closed-loop row −0.050104 + 2.285i (at ε = 2.1e-3).
- Residual: ρ/Σλ|g| = 1.1e-3 at 1e-3 and 4.5e-4 at 1e-2. The group-wise condition first falls below 1e-2 at ε = 5.4e-4 (7 tops).
- Predicted decrease: 3.4e-4 for Δ from 1e-5 to 1e-4.
- Certificate: ≤ 369.744507 at Δ = 1e-5, with α −0.0501000 on the QP's line, which is admissible. `--complete on`: piece set certified, no new top.

**412.28, strip** (stop: deadline, code 4).
- Tops: 7 within 2.5e-3.
- Active constraints: the pair (0,1) at −0.05 ± 4.256i (v1), and real poles at −0.05 in blocks (2,3) and (5,6) (v2).
- Residual: at ε = 1e-3 the C_c group cannot cancel (1.0), so it is not ε-stationary there. ρ/Σλ|g| = 5.2e-3 at 1e-2.
- Certificate: ≤ 411.945579 at Δ = 1e-4, after the SQP's modal correction of 2e-8.

### MIMO central
**c151** (stop: progress rule 3:0.0015; code 4, 40 iterations, all on the trust boundary).
- Tops: 13, all within 4.05e-4. The top at 0.926 has σ₂ 1.34 % below and enters the decrease LP as a 2×2 block.
- No constraint is active within ε = 1e-2.
- Residual: ρ/Σλ|g| = 6.0e-5 at ε = 1e-3. Predicted decrease levels off at about 1.6e-4.
- The first-order model holds up to about Δ = 5e-5. At the SQP's trust radius 0.0428 the LP predicts 2.7e-3, but the model is not valid there.
- Certificate: ≤ 123.887235 at Δ = 1e-5, on the widened band. It is near ε-stationary: the decrease is below the current rule's 1.9e-4 per round.

**c177** (stop: deadline, code 4).
- Tops: 6 within 1e-2. The 2.913 top dominates (|g| 1.17e5); the next is 1.9e-3 below. The top at 0.767 has σ₂ 3.8 % below and gets a block.
- Active constraint: actuator 1 clearance at the wall margin (y = 0.05). MFCQ margin 1.0.
- Residual: B_c group at 0.66 at ε = 1e-2. No group-wise cancellation below 1e-2 up to ε = 2.8e-2.
- Descent: ratios 1.00, 1.00, 0.99 and 0.72 for Δ = 1e-6, 1e-5, 1e-4 and 1e-3.
- Certificate: ≤ 152.378057 at Δ = 1e-3.

## 8. Limits

1. **No winding count.** The certificates use the line test (Im in [0, whi_t + 3]) and the level set on [0, whi_t]. They do not run the winding count over all w or the continuation gate along the segment.
2. **Completeness.** Without `--complete on`, ρ is an upper bound and pred is optimistic. Even with it, a second top inside a known top's region is not separated.
3. **Metric dependence.** ρ, pred(Δ) and Δ_v depend on the metric: the record's scales, not the run's part-start scales, plus the modal gauge.
4. **Activity scales differ.** Pieces are ε-active relative to gamma; constraints relative to ε_s or the margin.
5. **Untested paths.** End pieces with a z-dependent band end, the Hermite closed-loop block and blocks with r ≥ 3 are untested on plant data.

## 9. Files

`tools/bfs_stationarity.py` is modified but not committed: +97/−23 lines against 63f6fd9, final md5 2be195a7. The final edits after the last runs are docstring-only.

Everything else is in `(session scratch) wf_stat/final/`:
- Patch: `review_changes.diff`
- Main runs: `siso598.log`, `siso598.json`, `siso555.log`, `siso555.json`, `c151.log`, `c151.json`, `c177.log`, `c177.json`, `d370.log`, `d370.json`, `d412.log`, `d412.json`
- Completeness runs: `d370_complete.log`, `d370_complete.json`, `siso598_complete.log`, `siso598_complete.json`
- Certified designs: `d370_certified.npz`, `d412_certified.npz`
- Cluster template: `stationarity.slurm`
- Queue scripts and log: `queue.sh`, `queue_rest.sh`, `queue_complete.sh`, `queue.log`
- The exact tool version each run used: `tool_<tag>.py`