# STATUS -- where every planned and added item stands (2026-09-29)

Short by design: one line per item, current state only. The history -- what was
tried, measured, reverted and why -- is in STAGE0-6.md (by date and section), the
row-by-row log in TRACKER.md, the code removals in CLEANUP.md.

**Where the plan comes from.** The stages 0-7 are in `HANDOFF_implementation.md`
section 5 (the upload of 2026-09-07; TRACKER.md section A mirrors them). The items
are the to-do lists of the two notes: `docs/note_optimization_clean.pdf` section 9
(items 0, 0b, 1-11 and the unnumbered "projected oracle" item) and
`docs/note_algebra_clean.pdf` section 12 (items 1-13); the handoff's section 8 maps
each item to a stage. Items added along the way are in the last table.

Status words: **done**; **option** (built and validated, not the default);
**next: parallel / cavity / cluster** (the coming phases); **dropped** (no longer
relevant -- reason given); **left** (deliberately not touched, by decision).

## Current results (K+B+C unless stated; single process; certified over all frequencies)

| case | value | wall time | notes |
|---|---|---|---|
| ne=100 | 6.399134 | 22.6 min | nominal scripts (`tools/n100_kbc.sh`); best known 6.389368 (32 min, looser stop) |
| ne=200 | 6.486505 | 31.4 min | `tools/n200_kbc.sh`; the other known local minimum is 6.438335 |
| ne=200 frozen (reference) | 7.964064 | -- | the 27-case frozen suite, re-certified globally (`results/reference_table_ne200_global.csv`) |
| K+FB+FC, 4 Gabor atoms per field (best) | 3.876296 (ne=200) | 58 + 132 min from the ne=400 atoms design | same design 3.879101 on ne=400, 3.879283 on ne=800; static gain abs(K) 4468, high-gain regime (STAGE5 32.6) |
| K+FB+FC ne=400, 4 Gabor atoms per field | 3.903930 | 82 min, 11 rounds | stopped early by the progress stop; same design 3.903963 on ne=200, 3.904026 on ne=800 |
| K+FB+FC ne=400, 160 modes + H^2 shape row (k <= 1) | 3.914071 | 281 min, 20 rounds | same design 3.915962 on ne=800; optimised at ne=200: 3.911740 |
| K+FB+FC, unconstrained fields | **withdrawn** | | 3.898938 (ne=100, 80 modes) is 5.52 on ne=200: a discretisation artefact (STAGE5 32.7) |
| **MIMO, 2 actuators + 2 sensors, ne=100** (branch `mimo`, STAGE6) | atoms (static gains): **2.383533 centralized and decentralized = the two-pair floor** (1e-10 / 5e-9 above sup_w sigma_3^ol); K+B+C: decentralized 5.565958, centralized **3.792962** | 8-44 min | SISO at ne=100: 3.877016 (atoms), 6.399134 (K+B+C); every record re-checked over all w by `tools/check_mimo.py` |
| lower bounds | 3.509193 (any single actuator/sensor pair), 9.831865 / 9.660137 (sensor at 0.98 / actuator at -1.03) | | `results/bounds_ne200.json` |

Figures: `figs/final_sigma.png` (uncontrolled sigma_1 vs controlled sigma_1, sigma_2),
`figs/final_fields.png`, `figs/gamma_vs_time_loose_ends.png`; fields: `figs/fields_refine_*.png`
(refinement, unconstrained), `figs/fields_reg_*.png` (H^2, atoms), `figs/band_*.png`, `figs/mode_response.png`.

**Mesh.** The plant is 2.3 % off at ne=100 (open-loop peak 901.6 vs 881.7 at ne=400), 0.14 % at
ne=200: judge any value at ne >= 200; ne=100 is for debugging (STAGE5 32.4).

## Nominal algorithm (what the scripts run)

Epigraph SQP (Clarabel QP) on a reduced (projected) model of the closed loop,
inside certify-and-exchange rounds on the full model. Inner rows: sigma_1 at the
working-set frequencies following their peaks, 2x2 Clarke blocks where sigma_1 ~
sigma_2, one row per near-line closed-loop eigenvalue from the reduced pencil
(residual-filtered) plus a 2x2 Hermite block for the two rightmost, the compensator
Hermite block and its pair held on the double-pole face S. Outer: sparse level-set
certificate (strict disks), full-model samples at the reduced model's peaks,
continuation gate with an exact winding count as fallback, progress stop.
Frozen path bit-identical (case B ne=100: 10.23128516441147). ~105 full
factorisations per round, ~70 with reduced-model certification every 4th round.

## Stages (HANDOFF section 5)

| stage | status | gate |
|---|---|---|
| 0 harness and dump | **done** | 27 gamma and gate chains bit for bit (cluster replay) |
| 1 solver settings (note items 0, 3, 11) | **done**, except the 27-case regression -> **next: cluster** | balance residual on stalled rows: met once measured correctly (Clarke residual); stall mechanisms attributed |
| 2 initialisation (item 1) | pilot done (16+16 starts, old code); the 96-start study -> **next: cluster** | multimodality confirmed (two local minima at ne=200 on the current code) |
| 3 working-set policy (items 4, 5) | built and measured; **dropped** as default (see items 4-5) | not met, by design: the reduced oracle removed the cost it targeted |
| 4 explicit constraints + inner solver (items 2, 8, 9) | **done**, on the reduced oracle rather than the full one (full-model rows were unaffordable at ne=200; every round is certified on the full model instead) | K+B+C rides the boundary unaided (face S); KKT residual 2.5e-5 / 5.1e-5 x gradient scale at the final designs; IPM-vs-SQP decided by structure, not overhead |
| 5 projection oracle (algebra items 8, 11-13) | **done** for K, K+B+C, K+FB+FC at ne=100/200/400; 27-case validation -> **next: cluster**; item 11 done as an option (below) | 0 factorisations per inner evaluation (1 when the Ritz filter refuses); reduced vs certified 1e-13..1e-14 at the witness every round (asserted); trajectory test: model refreshed by anchoring each round, compacted at dim 120 |
| 6 parallel certificates (algebra 4-6, 10; note item 6) | **next: parallel** | -- |
| 7 cavity readiness (algebra 1-3, 9) | **next: cavity** (GL-side parts done: tail certificate, exact tail constants) | -- |

## Optimisation note, section 9

| item | status | reason / where |
|---|---|---|
| 0 QP accuracy, mu log, regrow detector | **done** | QP accuracy is not the stall mechanism (9th-12th digit) |
| 0b direct nonsmooth alternative (bundle / HANSO) | **dropped** | the stalls it targeted were the compensator double pole and stale quasi-Newton memory, both fixed (STAGE5 25-28) |
| 1 random-start multistart + stabilisation | pilot **done**; study **next: cluster** | the lever is multistart over controllers (the two ne=200 minima share the positions) |
| 2 explicit stability rows (+ cheap reduced form) | **done** | per-eigenvalue rows from the reduced pencil, closed-loop Hermite block, compensator face S; winding count outside |
| 3 sigma_2 enrolment | **done** | full 2x2 Clarke block (cross terms), not only the diagonal |
| 4 neighbour seeding (A-N satellites) | **dropped** | peak following + in-solve refinement track drifting peaks directly; with the reduced oracle the working set costs no factorisation |
| 5 retirement, abs(Omega) <~ 50 | **dropped** | same; capping measured risky (certified 40.99 vs t 7.93 once) |
| 6 parallel gate, replay-validated | **next: parallel** (stage 6) | |
| 7 controller-pole cap -> regulariser | **done** | band from the closed loop; certified over all frequencies at GL |
| 8 inner solver, two levels (IPOPT vs SQP, restart overhead) | **done**: SQP (Clarabel, cones) | IPOPT **dropped**: the inner solve is no longer the bottleneck and the SQP represents the kinks exactly |
| 9 Newton polish of the balance system | **dropped** | tried: unsafeguarded at a corner; the SQP reaches KKT residuals ~3e-5 |
| 10 visualisation (dump, plot_omega.py) | **done** | |
| projected oracle in the inner loop | **done** | = stage 5 |
| (a) random starts free on the reduced model | **next: cluster** (with item 1) | |
| (b) global optimisation of the position cases | **dropped** | positions are not the lever (nearly equal in both ne=200 minima) |
| 11 remove the level-scaling constant | **done** | |

## Algebra note, section 12

| item | status | reason / where |
|---|---|---|
| 1 tail bound for a singular mass matrix | **next: cavity** | GL version done (`tools/tail_certificate.py`: every record certified over all frequencies) |
| 2 once-per-mesh constants, tight and certified | GL: **done** (exact ||M^-1 A||, rigorous compensator resolvent bound); cavity Lanczos version **next: cavity** | |
| 3 P_plant at cavity scale (Beyn) | **next: cavity** | |
| 4 parallel winding certificate | **next: parallel** | now the largest single cost: 400-960 factorisations per refused continuation |
| 5 `--rigor` Neumann segment bound | **next: parallel** (inside item 4) | |
| 6 cost table, parallel column | **next: parallel** | |
| 7 MIMO | **done** in the production loop (branch `mimo`, STAGE6): rank-(m+p) resolvent, certificates, det(I - K G) winding count, reduced oracle, packer (central / decentral / static), driver; SISO bit-identical | ne=100: atoms 3.877 -> 2.383533 (the two-pair floor, central and decentral); K+B+C 6.399 -> 5.566 (decentral) / 3.793 (central); prototype STAGE5 34.3 |
| 8 projection oracle, GL prototype | **done** | 27-case validation -> cluster |
| 9 subspace dimension in 2-D | **next: cavity** | GL: dim 95-142 with the cap at 120 |
| 10 anchor-preconditioned winding tail | **next: parallel** | |
| 11 watch-list tracking by interpolation (Mengi) | **option** (`--cont-track reduced`) | validated against dense truth on 42 real steps (no cluster needed): same decisions 42/42, 0 false accepts, 11x fewer factorisations in the gate; not the GL default (ne=100 loop: 6.405826 vs 6.399134, a stop-rule difference, not a gate one); switch on at the cavity (STAGE5 33) |
| 12 margin on the reduced model | **done** | = note item 2's cheap form |
| 13 global optimisation, position cases | **dropped** | as note item (b) |

## Items added along the way

| item | status | where |
|---|---|---|
| constraint handling at the start (compensator cap, band, selection, KGC start, donor guard) | **done** | STAGE1 |
| winding-number (unconditional) stability certificate at save and for refused continuations | **done**; exact in the compensator since 2026-09-26 | STAGE5 18, 30.4 |
| compensator double pole on the line: Hermite block, then the face S | **done** | STAGE5 25-26, 29 |
| wrong-eigenvector abscissa gradient (`--alpha-argmax`) | **done** | STAGE5 5.4 |
| position-robust reduced model (input-basis enrichment) | **done** | STAGE5 5 |
| peak following, enrolment of near-maximal peaks, in-solve refinement | **done** | STAGE5 9-19 |
| closed-loop Hermite block; Rayleigh-quotient near-line eigenpairs | **done** | STAGE5 15, 20 |
| fake witness / witness invariant asserted every round | **done** | STAGE5 24 |
| stale quasi-Newton memory across rounds (plateaus) | **done** | STAGE5 28 |
| batched reduced rows and batched peak following | **done** | STAGE5 28 |
| tau governor after homotopy splits; late face entry; members sharing a top | **done** | STAGE5 29 |
| progress stop (with arming and exceptions) + KKT residual report | **done** | STAGE5 28-30 |
| factorisation cuts (peak reference, gate skip, caches, end curve) | **done** | STAGE5 30.2-30.3 |
| sparse certificate on partial ARPACK sets (`--cert-disk strict`) | **done** at GL; residual-based disk -> **next: cavity** (the far-band hole of STAGE5 30.5) | STAGE5 30.5 |
| certification over all frequencies (tail bound + dense band) | **done** at GL; **sparse at any n** (numerical-range tail bound + sparse level set, at save) | STAGE5 30.6, STAGE6 6.19 |
| no dense n x n operation in the loop (band, plant pole data, dumped spectrum) | **done**: sparse enclosures and located sets, validated against dense at ne = 100-400; dense left only as checks gated at low n | STAGE6 6.12-6.13 |
| reduced-model certification every k-th round | **option** (halves factorisations; decide at the cavity) | STAGE5 14, 30.9 |
| K+FB+FC in a Laplacian basis | **done**; unconstrained use **withdrawn** (cutoff-driven, not mesh-independent) | STAGE5 6, 29, 32 |
| FB/FC field regularisation | **done**: 4 Gabor atoms (width >= 1, wavenumber <= 0.707; recommended) and a scale-invariant H^2 shape row (`--field-kmax 1 --field-order 2`); H^1 row too weak at k <= 1 | STAGE5 32 |
| why viscosity does not regularise the fields; refinement moves the ripple to the new cutoff | **answered** | STAGE5 32.2-32.3 |
| static gain abs(K) in K+FB+FC | finite optimum for given fields (gamma rises 4e-5 towards the high-gain limit); control-cost weight (H-infinity norm of W K S) kept for later (user) | STAGE5 34.1 |
| no square root / factor of M in the H-infinity path; sparse-only basis, atom fit, tail constants | **done** | STAGE5 34.2 |
| mesh-independence test by cross-evaluation (tools/cross_eval.py) | **done** | STAGE5 32.7 |
| closed-loop pair on the line (face-S analogue) | detector only; never hit | STAGE5 30.7 |
| debugging at ne=100 (transfer tool) | **done** | STAGE5 21 |
| MIMO: unstable acceptance far outside the band (`--wind-accept`, winding count at every acceptance) | **done** (MIMO default) | STAGE6 6.3 |
| MIMO: high-gain branches (no eigenvalue tracking in the reduced pencil; certificate region from eig(D_c C^H M^-1 B)) | **done** | STAGE6 6.4 |
| MIMO: gate INCONCLUSIVE on a very wide band -> winding count | **done** | STAGE6 6.5 |
| MIMO: false best, band one candidate late (`--band-first` on for MIMO); dense all-w gamma check at save (all cases, n + r <= 1000) | **done** | STAGE6 6.6 |
| run-to-run determinism of field runs (ARPACK start vectors) | **done** | STAGE6 6.8 |
| SQP stationarity test fooled by one very sensitive row (atom amplitudes) | **done** for MIMO (`--sqp-stat pred`: predicted decrease too); SISO: faster early, 0.08 % higher at the stop on ne=100 -> decide on the cluster | STAGE6 6.14 |
| MIMO code on SISO (`--force-mimo`) | **done**: same trajectory as the SISO path (2.5e-9 over 5 rounds), same cost; batched MIMO rows; packer gauge/scales = SISO | STAGE6 6.16 |
| actuator/sensor duality (B' = P conj C, C' = P conj B, K' = K^T) | **done**: exact for MIMO (gamma equal to 1e-14); optima self-dual in SISO and decentralized K+B+C, not for atoms (non-unique optimum at the floor) | STAGE6 6.17 |
| static gain in K+FB+FC; non-diagonal central gain | **answered**: high-gain regime, the closed loop tends to the controller-free T_inf = R - R B (C^H R B)^-1 C^H R | STAGE6 6.17 |
| dead code after negative tests | **done** (removed, recorded) | CLEANUP 7 |
| CLEANUP section 5 (reference tables, bounds file, archive) | **done** | CLEANUP 5 |

## Next

0. MIMO: done at ne=100 (STAGE6); open: multistart (the decentralized problems are multimodal),
   `--sqp-stat pred` for SISO (cluster regression), control-cost weight W K S (later, user). The compensator's
   face S carries over (kbc_dec entered it at round 0: the SISO optimum's pair on the line). Runs at ne >= 400 go to the cluster (user, 2026-09-28).
1. Parallelisation (stage 6): winding count, gate and certificate shifts, reduced rows over workers.
2. Cavity (stage 7): operators from the user's solver, MUMPS/PETSc, subspace dimension, residual-based disks; the tail bound and P_plant are sparse now (STAGE6 6.13, 6.19) -- to be measured at 3-D sizes.
3. Cluster: 27-case regression, multistart over controllers.
