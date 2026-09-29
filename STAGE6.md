# STAGE 6 -- m actuators and p sensors in the production loop (algebra note item 7)

Started 2026-09-29 (user: "if you are confident in the siso code and make sure it
is safely stored somewhere as backup, now proceed to create a new version with
mimo in the production loop"). Branch `mimo` of the repository, worktree
`/home/claude/mimo`; the SISO code is frozen as tag `siso-v1` (section 6.0).
The prototype that preceded this is STAGE5 34.3 (`tools/mimo_proto.py`).

## 6.0 The SISO backup

- git tag `siso-v1` on commit fdae0e7 of the main tree (`/home/claude/work`, branch
  as before): the SISO production code, records and notes as of STAGE5 34.
- in the delivered outputs: `codesign_siso_v1.bundle` (a git bundle of the whole
  repository, history and tag included: `git clone codesign_siso_v1.bundle`) and
  `codesign_siso_v1_20260929.tar.gz` (the working tree at the tag, records
  included).
- the MIMO work lives on branch `mimo` in a separate worktree; the SISO tree is
  not touched.

## 6.1 Architecture

The SISO names are kept for the design arguments so that the driver's plumbing
carries both cases; MIMO is detected from the shape of the load/sensor vectors
(`b`, `c` two-dimensional, n x m and n x p).

| piece | SISO (unchanged) | MIMO |
|---|---|---|
| closed-loop resolvent | `closedloop_sparse.ClosedLoopResolvent`, rank-2 Woodbury | `closedloop_mimo.ClosedLoopResolventMIMO`: rank-(m+p) Woodbury, U = [[B, 0], [0, B_c]], V = [[-D_c C^H, -C_c], [-C^H, 0]]; ONE sparse LU of i w M - A per frequency; sigma^2 = lambda_max(M T^H M T) (M only as a sparse product) |
| design arguments | arow = (d, e, f), nvec, K scalar | arow = (d, e, f, B_c), nvec = C_c (m x r), K = D_c (m x p); `controller.Bc_of` dispatches |
| gradients | `design_grad_tri`, `port_sides` | D_c[i,j]: (z_p^H B_i)(C_j^H x_p); C_c[i,k]: (z_p^H B_i) x_c[k]; B_c[k,j]: conj(z_c[k]) (C_j^H x_p); port sides act = D_c C^H x_p + C_c x_c (m), sen = z_p^H B D_c + z_c^H B_c (p) |
| level-set certificate | `certify_sparse.SparsePencil`, coupling rank 4 | same class, coupling rank 2(m+p); `_cl_arow` dispatches everywhere a closed loop is built |
| winding count | f = 1 - K(s) G(s) | f = det(I - K(s) G(s)) (class `_FM`); with `--wind-ctrl exact`, g = det([[sI - A_c, -B_c G], [-C_c, I - D_c G]]), g'/g = tr(Blk^-1 Blk'); tail bound q = 1/(2m); step floor referenced to min(WT, 4000) (a high static gain made WT ~ 1.2e5 and the floor 0.12) |
| reduced oracle | `_ProxyCL`, batched gains | `_ProxyCLMIMO` (port sides with the full-length B, C), `_reduced_pencil_mimo` (exact coupling U_c V_c^H for the near-line pairs and the abscissa); gains per row (no batch yet) |
| packer | `Packer` | `PackerMIMO`: free mask over (d, e, f, B_c, C_c, D_c) -- central: all; decentral: block-tridiagonal A_c, B_c fixed to the SISO gauge per block, C_c and D_c block-diagonal; static gain ('gain'): D_c only; scales per entry max(abs(theta0), a floor per group) |
| fields | one field per side | one field per port (m actuator, p sensor), atoms or basis; one field row per port |
| driver | -- | `--n-act m --n-sens p --ctrl-struct {central,decentral} --loop-orders --mimo-start REC`; MIMO-aware record, checkpoint and dumps; `--wind-accept {auto,on,off}` (6.3) |
| starts | -- | `tools/mimo_start.py`: from a SISO record (loop 2 switched off, pair 2 at chosen positions; or the atoms design with shifted copies of its fields and a passive compensator), or from a prototype record |
| scripts | -- | `tools/mimo_n100.sh {kbc_dec, kbc_cen, at_dec, at_cen}` |

The SISO machinery on the compensator carries over unchanged through the packer
(A_c is tridiagonal in both): the double-pole face S was entered by kbc_dec at
round 0 (the SISO optimum's pair sits on the line), per-pole rows and the
Hermite block act on the whole A_c.

Instrumentation: the SQP's inner statistics carry `exit_info` (stat, gradient
scale, violation, active rows) when it declares stationarity (6.10).

Not supported, by decision: a MIMO design with no compensator states (r = 0).
The static-gain MIMO starts keep the SISO design's compensator states with
C_c = 0 (passive: unobservable from u, its poles are closed-loop poles at
Re = -0.2188).

## 6.2 Validation

- **SISO bit-identity** (the SISO path must not move): invariants in the MIMO tree
  (`tools/check_invariants.sh`: frozen path 10.23128516441147, witness invariant 0
  violations, Hurwitz selftest 0/200) PASS; `tools/check_siso_identity.sh` (new):
  the same 5-round K+B+C ne=100 run in this tree and in the `siso-v1` tree,
  8.754365110002194 in both. Re-run after each change of this stage: after 6.4,
  after 6.6 and on the final commit 0e7f1c1 -- PASS each time.
- **`tools/check_mimo.py REC`** (new; writes REC_check.json), every check against
  an independent dense reference (explicit Cholesky of M -- validation only):
  1. sigma_1 of `ClosedLoopResolventMIMO` vs sigma_max(L^H [S_cl^-1]_11 L) at the
     record's active frequencies and a grid: 1e-12..1e-13;
  2. the MIMO class at m = p = 1 vs the SISO class on the SISO K+B+C record:
     sigma 1.1e-14, controller gradients 3.8e-14;
  3. central finite differences of sigma_1 along random directions in D_c, C_c,
     B_c, the A_c diagonal and each load/sensor vector (port sides), at a
     generic frequency (w = 2): error <= 2e-6 of the largest derivative (FD
     truncation on entries of size 2e4; h -> 1e-4 there gives 8e-9); at the
     witness the same on every record except the two floor designs, where the
     gradient is only 2e-5 .. 3e-4 (sigma_1 pinned to sigma_3^ol) and the
     central difference is dominated by its truncation error;
  4. the record's gamma vs the dense level set on the full even pencil over ALL w
     (every eigenvalue at each level, no band);
  5. the winding count right of Re = -eps and Re = 0 vs the dense spectrum;
  6. the abscissa certificate vs the dense spectrum.
- **prototype vs production**: sigma_1 of the four prototype designs
  (`results_mimo/{dec_p4b,cen3,atoms_dec,atoms_cen}.npz`, independent code) with
  the production class: 1.0e-13, 8.3e-15, 2.2e-15, 4.5e-15.
- **reduced oracle**: reduced vs certified at the witness 1e-11..1e-16 every round
  (the existing assertion, now on MIMO designs).

## 6.3 First production runs: an unstable design accepted, `--wind-accept`

at_dec try 1 (atoms, static gains, decentral) accepted a design with a
closed-loop eigenvalue at +17.13 - 111.77i, far outside the band [-30, 30]: the
continuation gate watches the near-line eigenvalues of the certified band, and a
high loop gain moved an eigenvalue from far left to Re > 0 in one step. Only the
dense save veto caught it (records: `results_mimo_prod/at_dec_try1_unstable`).
Fix: `--wind-accept` (default `auto` = on for MIMO): every candidate the gate
would accept must also have a winding count of 0 right of Re = -eps (global in
w). SISO default unchanged (never observed there).

## 6.4 The stall at 2.918: high-gain branches the SQP could not see

at_dec try 2 stalled at 2.918085 from round 4 (three rounds without progress,
`FIXED POINT ... stalled, not converged`), with the inner SQP collapsing its
radius (termination 6) after one accepted step. Diagnosis (line-search trace with
the full-model residual of every near-line pair; `results_mimo_prod/at_dec_try2_stall`):

- with two loops at high static gain (abs(D_c,11) ~ 4900) the closed loop has
  HIGH-GAIN branches near the eigenvalues of D_c C^H M^-1 B, at |Im| ~ 150-170;
- the SQP's abscissa rows come from the reduced pencil, whose near-line
  eigenpairs are TRACKED (Rayleigh quotients, O(k^2)) between full dense eigs every
  10 calls; tracking follows the eigenvalues already near the line and finds
  newcomers only at the next full eig;
- one accepted step moved a high-gain branch from Re = -23 to Re = +38.19
  (Im = -164.05): the tracked rows did not contain it, the step was accepted, and
  the iterate was UNSTABLE (dense check: +38.19 - 164.05i; the pair has full-model
  residual 1e-14 once seen, so it is in the basis);
- at the next evaluations the full eig found it: every trial point then carried a
  violation ~ 38 x nu in the merit against a current iterate recorded as
  admissible, so every step was refused down to radius 0; the returned candidate
  failed the full certificate and the round returned its start. Same step next
  round: stall.
- the full abscissa certificate itself swept only the region of
  `certify_sparse.stability_region` (plant's rightmost modes and compensator poles,
  about [-30, 30]) and missed a high-gain branch at -0.0502 - 156.87i (the dense
  save check flagged it: "the line sweep missed the rightmost eigenvalue").

Fixes (MIMO only; SISO bit-identical, 6.2):
1. `reduced._near_line_eigs`: no tracking for MIMO -- the dense eig of the
   reduced pencil (dim ~ 170) at every evaluation sees every mode of the basis.
2. `FECoDesign.cert_region_mimo`: the MIMO abscissa certificate sweeps the union
   of `stability_region` and, for each eigenvalue l of D_c C^H M^-1 B (one solve
   with the cached sparse LU of M), [Im l - (abs(l)/4 + 2), Im l + (abs(l)/4 + 2)].
   Estimates against the dense truth: -1.7-144.7i vs -0.05-156.9i; 38.8-153.0i vs
   38.19-164.05i; 58.5-167.4i vs 57.92-177.13i. On those three designs the
   certificate now returns -0.05023 (was -0.2188, wrong), +38.19 and +57.92
   (clear = False), in 0.05-0.25 s.

Effect: at_dec from the same start passes 2.918 at round 4 (2.752) and ends at
2.627326 (progress stop, round 9; `results_mimo_prod/at_dec_try3_premfix`).

## 6.5 Gate INCONCLUSIVE on a very wide band

With the fixes of 6.4, at_dec round 8 was refused with "gate REJECTS:
INCONCLUSIVE: could not prove the line Re=-0.05 clear (85 uncovered)": the band
follows the closed-loop spectrum (`band_from_closedloop`: 1.5 x the largest |Im|
of the eigenvalues with Re >= -band_recut) and a high-gain branch near the margin
at |Im| ~ 844 widened it to [-1268.9, 1268.9]; the line sweep ran out of shifts on
it. Fix (MIMO, with `--wind-accept`): an INCONCLUSIVE line sweep at the gate falls
back to the exact winding count right of Re = -eps; count 0 clears it (counted in
the record, `wind_accept.gate_inconclusive_cleared`). The three winding calls of
the driver (continuation refusal, acceptance, this fallback) now share one helper
`_wind_right_of_margin` and the count is computed once per candidate.

## 6.6 A false best: the band one candidate late

at_cen (from the at_dec record, code of 6.4-6.5) accepted in its round 0 a design
certified at 2.587706, then refused every candidate for 4 rounds (the SQP handed
back t ~ 6.85) and ended on a FIXED POINT with gamma 2.587706 in the record.
`tools/check_mimo.py` on that record: the dense level set over all w gives
**12.251419 at w = -700.501** -- a closed-loop eigenvalue at -0.54 - 700.50i (a
high-gain branch) and its resonance. The loop certified the candidate over the
band of the PREVIOUS design and widened the band from the candidate's spectrum
only after accepting it (the frozen order, `--band-first` off: "one candidate
late", harmless for SISO where the band moves slowly). The next round saw the
peak (greedy anchor: reduced 12.2514 > certified 2.5877) but the false best
stayed the reference, and nothing could beat it. Record kept as evidence:
`results_mimo_prod/at_cen_try2_falsebest`.

Fixes:
1. MIMO turns `--band-first` on (the candidate's band from its own closed-loop
   spectrum before its certificate).
2. A dense all-frequency check at save, for every case with n + r <= 1000 (SISO
   too -- check only): the dense level set on the full even pencil with no band
   (`certify_sparse.hinf_dense_levelset`, wlo = -inf, whi = inf); a higher value is
   reported ("the band missed a peak") and stored; the record carries
   `gamma_dense_all_w`. Like the dense alpha veto, it does not exist at cavity
   scale; there the tail certificate (algebra item 1) has to carry it.

The at_dec record of the same code (2.627326) passes the check (dense all-w
2.627326 at w = -796.078, rel 9e-13): its binding peak IS the high-gain branch's
resonance, which the band followed in time.

## 6.7 The progress stop on the atoms cases

With 6.6, at_dec (from the SISO atoms design) stopped on the 3-round progress
rule at 3.299918 after 6 rounds: rounds 3 and 5 returned candidates whose NEW far
resonance was caught by the candidate's band (certified 580.8 and 23.78, refused)
while their witnesses joined the working set -- the mechanism working, not a
plateau (`results_mimo_prod/at_dec_try4_stop3`). The SISO field cases showed the
same (STAGE5 32.6). `tools/mimo_n100.sh` runs the atoms cases with
`--stop-progress 8:0.0015`; K+B+C keeps 3.

## 6.8 Run-to-run determinism

Two at_dec runs of the same code and flags differed from round 0 (t 3.602146 vs
3.601816, same certified value 3.589338). Cause: `fieldbasis.laplace_basis` (made
sparse in STAGE5 34.2) called ARPACK (`eigsh`, shift-invert) without a start
vector, so the Laplacian basis -- and with it the reduced model's input basis of
every field run -- changed from run to run in signs and rounding. Fixed: a fixed
start vector and a sign convention (largest entry positive); same for the
Lanczos calls of the tail constants (`winding_cert._plant_tail_constants`,
`tools/tail_certificate.py`) and the `svds` fallback of the reduced model. Two
concurrent one-round replicas of at_dec now agree to the last digit (3.589316032515165
both). Scope: field cases run after STAGE5 34.2 (2026-09-29); none of the SISO
results reported in STATUS (the atoms and H^2 records predate 34.2), and the SISO
K+B+C identity test was never affected (no fields).

## 6.9 Results at ne=100 (2 actuators, 2 sensors) -- first pass (superseded by 6.19)

All on commit 0e7f1c1 (the code of 6.3-6.8), `tools/mimo_n100.sh <case>`, one
process each, two at a time on the 2 cores; records now under
`results_mimo_prod/prev_0e7f1c1/`. Every record passes
`tools/check_mimo.py` (REC_check.log / .json next to it): gamma equal to the dense
level set over ALL w (rel <= 3e-14), winding count 0 right of Re = -eps and of
Re = 0 equal to the dense count, abscissa certificate equal to the dense one.
Figures (`tools/mimo_prod_plots.py`): `figs/mimo_prod_sigma.png` (sigma_1 of every
design over |w| <= 3000 against no control and the floors; zoom: sigma_1..3 of
the best design, the prototype's sigma_1 and the open-loop sigma_3),
`figs/mimo_prod_fields.png` (actuator and sensor fields per port),
`figs/mimo_prod_controller.png` (|K_ij(i w)| of the K+B+C compensators).
Mesh: ne=100 carries the 2.3 % plant error of STAGE5 32.4; the structures are
compared on the same plant, as for the prototype.

| family | design | gamma (certified, all w) | vs SISO | rounds / wall | stop | notes |
|---|---|---|---|---|---|---|
| K+B+C | SISO optimum (loop 2 off) | 6.399134 | -- | -- | -- | `results_n100_kbc` |
| | **decentralized** (pair 2 started at -+4) | **5.565958** | **-13.0 %** | 12 / 34 min | progress | pairs at -+2.205, -+3.948 (x_a = -x_s to 1e-8); |D_c| = 0.607, 0.0055; prototype 5.594745 |
| | **centralized**, from the decentralized | **3.792962** | **-40.7 %** | 21 / 144 min | progress | pairs at x_a = -1.374, -4.605, x_s = 1.316, 4.571 (the mirror symmetry broken); cross couplings of K(i w) of the order of the diagonal in the band (figs/mimo_prod_controller.png); prototype 4.277652 after 160 rounds |
| atoms (4 Gabor atoms per field, static gains) | SISO optimum (the ne=200 design at ne=100) | 3.877016 | -- | -- | -- | |
| | decentralized, from the SISO optimum | 2.964887 | -23.5 % | 19 / 18 min | fixed point | |D_c| = 4826, 27: stalled (6.10) |
| | **centralized**, from that decentralized | **2.383533** | **-38.5 %** | 22 / 44 min | fixed point | = the two-pair floor sup_w sigma_3^ol = 2.3835333277 to **1.0e-10**; |D_c| = [[5060, 40], [33, 130]] |
| | **decentralized, from the prototype's decentralized optimum** | **2.383533** | **-38.5 %** | 6 / 8 min | fixed point | 2.38353334, 5e-9 above the floor; |D_c| = 42044, 32; prototype 2.387150 |

- **The atoms designs reach the floor.** With two actuator/sensor pairs the
  closed-loop sigma_1 can never go below sigma_3 of the open loop (a rank-2
  perturbation of the resolvent, Weyl); its supremum 2.3835333277 at w = -0.68245
  is the best any two-pair controller can do on this plant. The centralized design
  sits on it to 1e-10 and the decentralized one (from the prototype's start) to
  5e-9: globally optimal, certified over all w, and the decentralized structure
  loses nothing. At the witness the gradient of sigma_1 w.r.t. the design is
  2e-5 .. 3e-4 (sigma_1 is pinned to sigma_3^ol, which no design moves; the runs end
  on the FIXED POINT rule).
- **K+B+C**: the production decentralized design (5.565958) is below the
  prototype's (5.594745) from the same start, pairs mirror-symmetric (x_a = -x_s:
  the GL adjoint symmetry, STAGE5 34.3). The centralized one reaches 3.792962
  (-40.7 %) in 21 rounds where the prototype was at 4.277652 after 160; its static
  cross gains are ~0 but the dynamic cross couplings K_12(i w), K_21(i w) are of
  the order of the diagonal ones in the band: with fixed-shape actuators the
  mixing is done by the compensator's dynamics. It ends on the progress stop
  (0.126 % over 3 rounds), still falling slowly.
- **the decentralized atoms run from the SISO optimum stalls at 2.964887**
  (6.10), 24 % above the floor that the centralized run then reaches from it, and
  that the decentralized structure reaches from the prototype's start. Like the
  K+B+C decentralized problem (STAGE5 34.3), it is multimodal.

## 6.10 Why at_dec stalls (measured; the attribution is open)

- From round 4 on (except round 15) the inner SQP of at_dec declared stationarity
  after 1-3 iterations every round (`exit_info` in the dumps, new): ONE active
  row with a gradient norm of 1.4e3 - 1.7e6 against stat 0.009-0.16, so the
  relative test stat < 1e-4 x (mean active-row gradient norm) passed. Each round
  then gained <= 1 % through the working set's growth, not through the SQP.
- The binding peaks were in the plant's band (w ~ 0.62, later -1.08); far
  resonances at w = -53 .. -57 were binding in rounds 5-8.
- |D_c,11| stayed at 4816-4826 over 19 rounds (K22 crept 0.2 -> 27), while the
  decentralized design ON the floor has |D_c,11| = 42044 (and |D_c,22| = 32): the
  run never travelled along the gain.
- So the relative stationarity test (a SISO calibration, STAGE5 28) is not
  scale-safe when one very sensitive row is active; which coordinates carry the
  large gradient (scaled gains, atom parameters) is not yet measured. The
  static-gain problem is badly scaled in the high-gain regime of STAGE5 34.1; a
  control-cost weight (the user's H-infinity norm of W K S) would regularise
  exactly this, kept for later as agreed.

## 6.11 Cost

Per round at ne=100: K+B+C decentralized 170 s, centralized ~300 s (|Omega| up to
450, 114 free complex entries); atoms 55-120 s. Per-row reduced gains (no MIMO
batch yet) and a dense eig of the reduced pencil at every evaluation (6.4) are
the MIMO-specific costs. The prototype took 30-60 s a round with far less per
round (no peak following, no Clarke blocks, SLSQP restarted).

# Second pass (2026-09-29, user: "remove dense calls used to set the frequency band, replace with sparse. Everything needs to be sparse and validated ... because we are about to increase dimension a lot"; investigate the stalled atoms run and why K+B+C centralized stopped while still improving; actuator/sensor symmetry in MIMO; why a static gain in SISO K+FB+FC and why the centralized gain is not diagonal; is the MIMO code safe for SISO)

## 6.12 Dense operations: the audit

Every O(n^3) or dense-n operation reachable from the driver:

| where | what | kind | now |
|---|---|---|---|
| `band_from_closedloop` (every acceptance; every candidate for MIMO) | full closed-loop spectrum (`alpha_dense`) | **algorithm** | **sparse** (6.13) |
| `winding_cert.plant_pole_data` (winding count at acceptance, continuation fallback, final certificate) | full plant spectrum; refused n > 3000 | **algorithm** | **sparse** (6.13) |
| dumped spectrum (every round with `--dump`, all production scripts) | full closed-loop spectrum up to n + r = 4000 | diagnostic | **sparse** (the located set + the certificate's harvest) |
| anchor check, dense save veto (alpha) | full closed-loop spectrum | check only, n + r <= `--dense_nmax` | kept: validation at low dimension |
| dense all-w gamma check at save (6.6) | dense even pencil | check only, n + r <= 1000 | kept: validation at low dimension; the claim itself is now carried by the sparse all-w certificate at any n (6.19) |
| `closedloop_sparse.dense_reference`, `tools/check_mimo.py`, `tools/dual_design.py`, `tools/tail_certificate.py --method dense` | references | validation tools | kept |
| reduced model (`reduced.py`), compensator (`A_c`, r x r), QP matrices | dense, but k ~ 100-200 / r / #variables | not n | unchanged |
| `prepare.py`, `fe_pipeline` (LQG seeds), `stabilise.py`, `warmstart.py` | dense Riccati | starts only, not in the loop | unchanged; at scale the start comes from a coarse design transferred (`tools/transfer_ne.py`) |

## 6.13 Sparse spectra: the plant enclosure, the closed-loop located set

New module `sparse_spectrum.py` (sparse LU, M products, Lanczos/Arnoldi only).

**Plant: every eigenvalue of (A, M) with Re >= s0, certified complete.** With
H = (A + A^H)/2, S = (A - A^H)/(2i): Re lambda <= s_far = lambda_max(H, M), and for
Re lambda >= s0 the S-lemma gives +-Im lambda <= lambda_max(+-S + tau (H - s0 M), M)
for every tau >= 0 (minimised over a few tau) -- a rectangle, mesh-independent
because the diffusion that makes the high modes stiff makes H very negative on
them (at ne = 100/200/400: Re <= 0.309, Im in [-1.00, 3.93] for s0 = -1). The
rectangle is cut into tiles; a tile is either PROVED empty -- an eigenpair gives
|lambda - sigma| >= 1 / ||(A - sigma M)^-1 M||_M, the norm by Lanczos -- or
exhausted by the k eigenvalues nearest its centre (shift-invert Arnoldi, residuals
checked, k doubled until the k-th lies outside the tile), or split. An Arnoldi
failure doubles k instead of splitting (it comes from several eigenvalues at
the same distance; a smaller tile does not separate them).

| ne | s0 | sparse vs dense count | max distance | shifts | time |
|---|---|---|---|---|---|
| 100 | -0.65 / -1 / -3 | 5 / 7 / 20, equal | 1.7e-11 / 1.6e-10 / 4.1e-6 | 2 / 3 / 12 | 1.2 / 1.4 / 3.3 s |
| 200 | same | 5 / 7 / 20, equal | 3.2e-11 / 4.9e-10 / 6.0e-5 | same | 2.5 / 2.8 / 6.1 s |
| 400 | same | 5 / 7 / 20, equal | 1.4e-10 / 1.3e-9 / 8.5e-5 | same | 8.3 / 9.3 / 14.7 s |

(-3 reaches deep into GL's non-normal spectrum: eigenvalue condition numbers make
1e-5 the attainable accuracy there; irrelevant for the band or the count.)
`winding_cert.plant_pole_data` now uses it (dense only as `method='dense'`, a
reference): counts and hot lists equal to dense at ne = 100/200/400 for the lines
-0.05, -0.025, -1 (hot lists to 1.8e-11 .. 4e-8). Once per mesh.

**Closed loop: the eigenvalues with Re >= -band_recut.** The low-rank coupling
with |D_c| ~ 1e4 defeats numerical-range bounds, so: LOCATE from candidates --
the plant eigenvalues right of s0 - 0.6, the compensator poles, the high-gain
estimates eig(D_c C^H M^-1 B) (6.4) and the near-line eigenvalues the certificate
harvested -- each refined by the eigenvalues of the closed-loop pencil nearest it
(shift-invert through the resolvent's sparse LU + Woodbury), plus the band tiled
as above; then COUNT them with the winding number right of s0 (global in w) and
compare. Validation against the dense spectrum on every record (SISO K+B+C
ne=100/200, SISO atoms ne=200, the five MIMO designs, the false-best design of
6.6): located sets identical to the dense ones (0 missed, 7-18 eigenvalues each,
including the high-gain branch at -0.54 - 700.50i), band "need" identical, and
located = winding count on all (COMPLETE).

**In the loop (cost).** Tiling the whole band at every acceptance cost 30-35 % of
a SISO round at ne = 100 (Arnoldi far from a non-normal spectrum converges
slowly); tiling only the band's edge strips still 27 %. The loop therefore
LOCATES from candidates only (a few shifts per acceptance): only eigenvalues
beyond |Im| = (band - pad)/1.5 can widen the band, and those come from the
compensator poles, the high-gain branches and the certificate's harvest. At SAVE
the whole band is tiled and the set COUNTED (`final band check`, record key
`band_check`); a resonance the loop missed widens the band and the design is
certified again over it. SISO K+B+C ne=100, 5 rounds: 49.9 s against 43.4 s for
the dense code (siso-v1), same result 8.754365110002194; the difference is the
one-off plant enclosure and the final check.

## 6.14 The atoms stall, and the SQP's stationarity test

**On the intermediate code of this pass (sparse band, batched MIMO rows, the
packer's new scale rule -- before `--sqp-stat pred`) at_dec from the SISO atoms
optimum reaches the floor** (2.3835333281, 13 rounds, 32 min;
`results_mimo_prod/at_dec_diag_newcode`) where the first-pass run stalled at
2.964887. Nothing in those changes targets a stall: the problem is multimodal and
path-dependent (6.9), rounding differences sent the run elsewhere, and the
stalled design was simply improvable:
multiplying its D_c by 10 gives 2.95996, its fields' high-gain limit 2.95996
(6.17) -- a flat but descending direction the solve did not take.

**The mechanism of the short solves** (new `exit_info`: the gradient of the
active row by block of the design vector): the SQP declared stationarity after
1-6 iterations in many rounds because its test was stat < 1e-4 x (mean norm of
the active rows' gradients), and that norm is dominated by the atom AMPLITUDE
parameters (|dsigma/d amp| 100-840 against ~1 elsewhere; 1.7e6 once, at a far
resonance) -- so stat 0.004-0.16 passed. The same premature exits exist in SISO:
the nominal ne=100 K+B+C run spends rounds 0-6 at 4-8 evaluations each (with
the fix below: 33-122).
Fix, `--sqp-stat pred`: stationarity also requires the QP's predicted decrease at
the iterate to be <= 1e-7 t -- in the objective's own units, independent of the
parametrisation. Measured:
- SISO K+B+C ne=100, 5 rounds: 7.618183 instead of 8.754365 (-13 %);
- SISO K+B+C ne=100 to convergence (A/B, `results_n100_kbc_statpred`): 6.404012
  in 16 rounds / 22.2 min against 6.399134 in 21 rounds / 22.6 min -- ahead all
  the way (6.639 at round 7 against 7.907), but the 3-round progress stop ended it
  0.08 % higher. Not a regression of the method, a stop-rule interaction; still,
  "never regress": **default `auto` = pred for MIMO, rel for SISO**; the SISO
  switch belongs to the 27-case cluster regression.

## 6.15 Why K+B+C centralized stopped while still improving

It stopped on the progress rule (3 rounds < 0.15 %: 0.126 %). Continued from its
record with the rule relaxed (6:0.0005; `results_mimo_prod/kbc_cen_cont_diag`,
code 99dbb35): 3.792962 -> 3.717885 in 24 more rounds (-2.0 %), still ~0.1 % a
round. Why so slow: the optimum is EQUALISED -- sigma_1 within 1 % of gamma over
|w| <~ 16 with 28 local maxima within 1 % (SISO optimum: 12; decentralized: 21),
the flat "all-pass" shape of H-infinity optima. Every round must lower dozens of
near-active peaks together: the SQP runs its full budget (40 iterations) with
line-search steps of 1/32-1/128 (the linearisation of a continuum of peaks by a
finite working set), and 4 rounds ended early on the stationarity test of 6.14.
Fixes in place: all MIMO cases now stop on 8 rounds < 0.15 % (was 3), and MIMO
uses `--sqp-stat pred`. The tail itself is the known limit of epigraph SQP on
equalised optima (STAGE5 29-30); multistart / cluster budget are the levers.

## 6.16 The MIMO code on SISO (m = p = 1)

`run_case_epi.py --force-mimo` runs a SISO case through the MIMO path
(`tools/mimo_start.py --siso REC --one` converts the record). Two differences
found and removed:
- the MIMO packer's scale rule (a floor per group) differed from the SISO rule;
  now the SISO rule, with the group floor only for entries that START AT ZERO
  (switched-off loops, cross couplings) -- identical at m = p = 1;
- the central structure left B_c entirely free: a diagonal similarity leaves K(s)
  unchanged and keeps A_c tridiagonal, an r-dimensional flat direction; now one
  entry per row of B_c is fixed (its largest at the start) -- the SISO gauge
  B_c = mc ones at m = p = 1, and 16 fewer free complex entries for K+B+C central.
- batched reduced rows for MIMO (`reduced._batch_setup_mimo`: T~ = R + R B~ Gam C~^H R,
  Gam = K(iw)(I - G K(iw))^-1): values equal to the per-row path to 2e-15,
  gradients 1e-12, 5-6x faster.

Same start, same flags, 5 rounds (`results_mimo_prod/siso_vs_1x1`):

| stationarity | SISO path | MIMO path, m = p = 1 |
|---|---|---|
| rel (siso-v1) | 8.754365110 in 51.3 s | 8.754365088 in 54.3 s (same trajectory; 2.5e-9) |
| pred | 7.618183 in 212 s | 7.499712 in 213 s |

With 'pred' the solves run longer and the two paths' rounding differences (dense
reduced eig vs eigenvalue tracking, batched rows) grow into different iterates;
neither is systematically better. **Safe to use for SISO**: same trajectory where
the SQP is short, same cost; the MIMO path keeps its extra protections
(winding count at acceptance, band from the candidate, no eigenvalue tracking).
What remains SISO-only is code, not results: the symmetric-subspace mode
(`flags['sym']`), the legacy solvers and the SISO record format.

## 6.17 Actuator/sensor symmetry, the static gain, and why the centralized gain is not diagonal

**The symmetry.** The GL operator satisfies A^T = P A P and M = P M P (P: x -> -x;
5e-15), so R(s)^T = P R(s) P and the DUAL design
    B' = P conj(C),  C' = P conj(B),  K'(s) = K(s)^T
has the transposed closed-loop map up to P: the same singular values at every w,
the same gamma (`tools/dual_design.py`: equal to 1e-14..1e-16 on all five MIMO
records). A design equal to its dual (up to port order and a phase per port) is
self-dual: SISO x_a = -x_s, psi_a = conj(P psi_s). Measured distance to
self-duality ||C - P conj(B) Pi|| / ||C||:

| design | SISO K+B+C | SISO atoms | MIMO K+B+C dec | MIMO K+B+C cen | atoms dec | atoms cen | atoms dec (proto start) |
|---|---|---|---|---|---|---|---|
| distance | 0 | 0.022 | 0 (x_a = -x_s to 1e-8) | 0.083 | 0.31 | 0.84 | 0.64 |

So the symmetry of the PROBLEM holds exactly in MIMO; the optima need not be
symmetric. Expected where the optimum is not unique: the atoms designs sit on the
two-pair floor, reached by a whole family (next point), so nothing selects the
self-dual member; the centralized K+B+C optimum is 8 % from its dual -- a
symmetry-breaking local optimum or an unconverged one (its dual has the same
gamma; a symmetric design is a stationary point candidate but need not be the
minimum). Decentralized K+B+C, like SISO, lands exactly on the self-dual set.

**Why the SISO field design has a static gain.** K+FB+FC is the FULL dynamic
compensator (d, e, f, C_c, D_c free); the optimiser made |C_c| <= 0.09 against
|D_c| = 4468, and setting C_c = 0 leaves gamma unchanged (3.8762964). Reason: with
shaped fields the loop sits in the HIGH-GAIN regime, where the closed loop tends to
    T_inf = R - R B (C^H R B)^-1 C^H R
(Woodbury with K -> infinity), which does not depend on the controller at all
(3.8764825 for those fields, 5e-5 above the design). Once the gain is high at all
relevant frequencies, a dynamic compensator has nothing left to shape: the
fields do the work (they choose what is actuated and sensed), and a constant gain
reaches the limit at every frequency at once.

**Why the centralized MIMO gain is not diagonal.** The same limit holds for m = p
= 2: T_inf is controller-independent, so in the high-gain regime ANY sufficiently
large K gives the same gamma -- diagonal or not. Measured on the centralized atoms
design: zeroing the off-diagonal gains: 2.3835333 (unchanged to 7 digits); D_c x 10:
unchanged; the controller-free limit T_inf: 2.3835327 = the floor. The
off-diagonal entries (40, 33 against 5060, 130) are leftovers in a flat direction,
not a feature -- the observation "centralizing buys nothing" DOES generalise;
what it predicts is not "diagonal" but "irrelevant". (D_c x 0.5 gives 4.377: the
design sits near the knee of the high-gain regime.) The decentralized design from
the prototype's start reaches the floor differently: one loop at very high gain
(|D_c,11| = 42044) and one at a finite gain (|D_c,22| = 32) -- its T_inf is 7.94,
so it is NOT in the both-loops high-gain regime; the floor is reached by more
than one mechanism. With fixed Gaussian actuators (K+B+C) the high-gain limit is
poor and the controller must mix: there centralizing buys 31.9 % (5.566 -> 3.793)
and K_12(iw), K_21(iw) are of the order of the diagonal (6.9).

## 6.18 Cost after this pass

SISO K+B+C ne=100, 5 rounds: siso-v1 43.4 s; this code 49.9 s (the one-off plant
enclosure, ~2-3 s at ne=100, 9 s at ne=400, and the final band check with its
winding count); per round the band costs a few shifts instead of a dense
eigensolve. MIMO rows are batched (5-6x on the row evaluations). What scales with
n now: sparse LUs (resolvent, certificate shifts, band candidates, winding count),
Lanczos for the plant rectangle once per mesh -- no dense n x n operation in the
loop; the dense ones left are checks gated at n + r <= 1000 (all-w gamma) and
<= --dense_nmax (alpha). At save the sparse all-w certificate adds the plant
heights (once per mesh: ~5 s at ne = 100, ~100 s at ne = 400) and one level-set
call on [-W, W].

## 6.19 A sparse all-frequency certificate

The band is a heuristic (eigenvalues right of -band_recut); "certified over all w"
at low dimension came from the dense level set on the even pencil (check only).
At any n, now (`sparse_spectrum.closed_loop_tail`, run at save, record key
`all_w`; `tools/check_mimo.py` check 7):
1. a TAIL FREQUENCY W beyond which sigma_1 < gamma is PROVED:
   sigma_1(w) <= 1 / (dist(i w, W(A, M)) - rho_K(w)), where W(A, M) is the
   plant's numerical range -- bounded by the S-lemma heights H(d) of 6.13 at a
   ladder of depths d (dist >= max_d min(d, |w| - H(d)); mesh-independent: at
   ne = 400 H(d) ~ d + 24, so dist grows like |w|/2) -- and rho_K = ||M^-1/2 B||
   ||M^-1/2 C|| sup_{|w'| >= |w|} ||K(i w')|| (the closed-loop plant block's
   numerical range lies in W(A) + a disk of that radius; ||K|| bounded by
   ||D_c|| + min(Neumann, modal) terms of the r x r compensator);
2. the sparse level set (`hinf_certified_sparse`) proves no crossing above gamma
   on [-W, W].
Validation against the dense level set over all w:

| record | tail W | sparse all-w | dense all-w | time (tail + level set) |
|---|---|---|---|---|
| SISO K+B+C ne=100 | 257.5 | certified, 6.399134 | 6.399134 | 6.0 s |
| SISO K+B+C ne=200 | 335.3 | certified, 6.486505 | 6.486505 | 36.6 s |
| SISO atoms ne=200 | 4660.1 | certified, 3.876296 | 3.876296 | 39.8 s |
| SISO atoms ne=400 | 1431.4 | certified, 3.903930 | 3.903930 | 107 s |
| MIMO K+B+C central (first pass) | 658.0 | certified, 3.792962 | 3.792962 | 0.5 s (level set) |
| MIMO atoms central (first pass) | 6682.2 | certified, 2.383533 | 2.383533 | 7.4 s (level set) |

The time is mostly the plant heights (Lanczos, 17 depths; once per mesh and
process). Started from a value BELOW gamma the sparse level set climbs and can
exhaust its budget on a wide band (MIMO K+B+C from 0.999 gamma: 60 calls, not
certified); started at the loop's gamma it needs one call. At high gain W is
large (the high-gain branches sit at |w| ~ |D_c| |C^H M^-1 B|), which is where it
has to be.

## 6.20 Results at ne=100 on the final code

(filled in below when the runs finish)

## 6.21 Open items

- the plant heights of the tail certificate: ~100 s at ne = 400 per process
  (17 depths); a per-mesh disk cache when the dimension grows;
- the sparse tiling is slow far from a non-normal spectrum (Arnoldi convergence):
  used only once per mesh (plant) and once per run (final band check);
- `--sqp-stat pred` for SISO: faster early, to be decided on the 27-case cluster
  regression;
- multistart for the decentralized problems (multimodal) -> cluster;
- the high-gain regime of the static gains; control-cost weight (W K S) kept for
  later (user, 2026-09-28);
- winding count at cavity scale: P_plant from the sparse enclosure now (6.13);
  its cost at 3-D sizes still to measure;
- ne >= 400 MIMO runs: on the cluster (user, 2026-09-28).
