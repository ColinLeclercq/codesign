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

## 6.9 Results at ne=100 (2 actuators, 2 sensors) -- first pass (superseded by 6.20)

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
  now the SISO rule at m = p = 1 (for m + p > 2 the per-group floor stays: 6.22);
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
records of the first pass and to 4e-15..7e-15 on the final ones of 6.20). A
design equal to its dual (up to port order and a phase per port) is
self-dual: SISO x_a = -x_s, psi_a = conj(P psi_s). Measured distance to
self-duality ||C - P conj(B) Pi|| / ||C||:

| design | SISO K+B+C | SISO atoms | MIMO K+B+C dec | MIMO K+B+C cen | atoms dec | atoms cen | atoms dec (proto start) |
|---|---|---|---|---|---|---|---|
| first pass (0e7f1c1) | 0 | 0.022 | 0 (x_a = -x_s to 1e-8) | 0.083 | 0.31 | 0.84 | 0.64 |
| final records (6.20) | | | 1.2e-7 | 0.158 | 0.89 | 0.89 | 0.64 |

(All the final atoms designs are on the floor with different distances: the
self-dual member of the family is not selected. ||D_c - D_c^T|| / ||D_c||: 0 for
the decentralized designs, 1.3e-9 for the final centralized atoms design.)

So the symmetry of the PROBLEM holds exactly in MIMO; the optima need not be
symmetric. Expected where the optimum is not unique: the atoms designs sit on the
two-pair floor, reached by a whole family (next point), so nothing selects the
self-dual member; the centralized K+B+C optimum is 8 % from its dual (final record: 16 %) -- a
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
design sits near the knee of the high-gain regime.) That first-pass centralized
design started from the stalled decentralized one (2.965), off the floor, and the
off-diagonals grew on the way. The final centralized run (6.20) starts from a
decentralized design already ON the floor and keeps it diagonal: |D_c| =
[[5028, 2.0e-6], [2.7e-6, 171]] after 3 rounds, gamma unchanged -- with nothing
to gain, the SQP does not move the cross gains. The final decentralized design
(|D_c| = 5028, 171): gamma on a grid |w| <= 20 is unchanged by D_c x 10, by D_c x
1e4 and by either loop alone x 1e4; D_c x 0.5 gives 5.050 -- both loops in the
high-gain regime, T_inf = the floor.
The decentralized design from the prototype's start reaches the floor differently:
one loop at very high gain (|D_c,11| = 42052) and one at a finite gain (|D_c,22| =
17.5); on the same grid D_c x 10 gives 5.980, D_c x 1e4 gives 8.640, and loop 2
alone x 1e4 gives 8.645 while loop 1 alone x 1e4 leaves the floor unchanged. So
it is NOT in the both-loops high-gain regime: loop 2 needs its finite gain. The
floor is reached by more than one mechanism (first-pass record: T_inf 7.94). With fixed Gaussian actuators (K+B+C) the high-gain limit is
poor and the controller must mix: there centralizing buys 31.9 % (5.566 -> 3.793;
final records 30.4 %, 5.339 -> 3.715, 6.20) and K_12(iw), K_21(iw) are of the order
of the diagonal (6.9, 6.20).

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

`tools/mimo_n100.sh <case>` (8-round progress stop for every case, `--sqp-stat
auto` = pred for MIMO), one process each, at most two at a time; records in
`results_mimo_prod/<case>/`, each with `tools/check_mimo.py` next to it
(REC_check.log / .json): the certified gamma equal to the dense level set over ALL
w, winding counts equal to the dense counts, abscissa certificate equal to the dense
one, FD gradient at the witness, and check 7 (sparse all-w certificate, 6.19).
Every final band check located = counted (no incomplete band in any round).

| family | design | gamma (certified, all w) | code | rounds / wall | stop | dense all-w (rel) | check 7: tail W, level set |
|---|---|---|---|---|---|---|---|
| K+B+C | SISO optimum (loop 2 off) | 6.399134 | | | | | |
| | decentralized (from pair 2 at -+4) | **5.339317** (-16.6 %; first pass 5.565958) | 11c4073 | 41 / 186 min | progress (0.124 % / 8) | 3e-14 | 423.3, 1 call, 6.6 s |
| | decentralized, same start, SISO scale rule | 5.473942 | c4cf908 | 35 / 139 min | progress (0.133 % / 8) | 6e-14 | 285.9, 1 call, 5.9 s |
| | centralized, from the c4cf908 decentralized (5.473942) | **3.715450** (-41.9 %; first pass 3.792962) | 11c4073 | 36 / 5.3 h | stopped by hand (still -0.25 %/round), + 1 finalising round | 7e-14 | 590.0, 1 call, 5.4 s |
| atoms | SISO optimum at ne=100 | 3.877016 | | | | | |
| | decentralized, from the SISO optimum | **2.3835333278** (first pass 2.964887, stalled) | c4cf908 | 17 / 48 min | fixed point | 7e-15 | 6742.0, 1 call, 8.8 s |
| | centralized, from that decentralized | **2.3835333277** | c4cf908 | 3 / 5 min | fixed point | 1e-15 | 6742.0, 1 call, 9.0 s |
| | decentralized, from the prototype's optimum | **2.3835333438** | 528be56 | 14 / 33 min | fixed point | 1e-14 | 49272, 1 call, 378 s |

- **All three atoms designs are on the two-pair floor** sup_w sigma_3^ol =
  2.3835333277 (6.9): the decentralized one from the SISO optimum, which stalled at
  2.965 in the first pass, reaches it to 6e-11; the centralized one from it to
  <1e-12 in 3 rounds WITHOUT creating cross gains (|D_c,12|, |D_c,21| ~ 2e-6 against
  5028, 171; 6.17); the one from the prototype's start to 1.6e-8 with a different
  mechanism (one loop at finite gain, 6.17). The atoms problem is solved on this
  mesh: centralization buys nothing, decentralization loses nothing.
- **K+B+C decentralized**: 5.339317 on 11c4073, 4.1 % below the first pass and
  2.5 % below the c4cf908 run from the same start (the packer scale rule of 6.22:
  close to the first pass's trajectory for 9 rounds, then further), mirror symmetric
  (self-dual to 1.2e-7; |D_c| = 0.611, 0.0064), stopped on the 8-round progress rule.
  It became the record (`kbc_dec`); the c4cf908 one is kept as
  `kbc_dec_c4cf908_sisoscale` -- it is the start of the centralized run below.
- **K+B+C centralized**: 3.715450, -32.1 % from its start and -41.9 % against SISO,
  below the first pass (3.792962 after 21 rounds) and below the first pass continued
  for 24 more rounds (3.717885, 6.15). Timeline: rounds 0-5 (5.474 -> 4.549, 15 min),
  container restart, resumed from the checkpoint, rounds 6-34 (-> 3.717430, 4.8 h;
  rounds grow from 2.5 to 15 min as |Omega| grows to 1085 with 340 sigma_2 rows), then
  stopped by hand (killed after a checkpoint) and finalised by one more round from it
  (`EXTRA="--outer 1"`: 3.715450, 15 min, then the save sequence). It was still
  descending ~0.25 %/round: the equalised tail of 6.15; `--wall` is a per-round
  budget, not a hard stop, so the 8-round progress rule is the only automatic stop
  and at this rate it needs many hours -- a cluster matter. The in-loop all-w
  certificate (6.21) fired once here: round 18 found a peak 4.055491 at w = +42.455
  outside the band, the candidate was refused and the band widened to +-53.95.
  Cross couplings: |K_12(iw)|, |K_21(iw)| of the order of the diagonal in the band
  (figs/mimo_prod_controller.png); self-dual distance 0.158, ||D_c - D_c^T||/||D_c||
  = 0.028 (6.17).
- `tools/check_mimo.py` check 3 (FD) had a step 1e-7 max|theta| per block: on
  kbc_cen (max |C_c| = 95834, h = 9.6e-3) the n entry differed by 0.8 %
  (-1.5138/-1.5016; with h = 1e-3 .. 1e-8 the FD converges to the analytic
  -1.51384992: truncation of the check, not a gradient error), and on the c4cf908
  kbc_dec record the error was 2.5e-2 of the largest derivative. Now a per-entry
  step 1e-7 max(1, |theta_i|); all six final records re-checked: K+B+C 1e-9 .. 2.6e-6
  of the largest derivative at the witness, <= 2e-8 at w = 2; atoms: at w = 2
  <= 4e-8, at the witness the derivatives are 3e-6 .. 5e-5 (sigma_1 pinned to the
  open-loop sigma_3, which no design moves), so the relative FD error there (1e-3 ..
  4e-2) is an absolute 1e-7 -- the FD noise floor. All other checks unchanged
  (gamma, winding counts, abscissa, all-w certificate).
- Centralization: K+B+C 5.339317 -> 3.715450 (-30.4 % from the best decentralized);
  atoms 0 (both on the floor).
- Provenance. at_dec and at_cen ran on c4cf908 (sparse band; the SISO scale rule
  in the packer, which 6.22 replaced for m + p > 2 -- identical scales for static
  gains; the all-w certificate only at save as a check, dense -- the in-loop
  certificate came with 528be56): their dense all-w values equal the loop's, so no
  false best, and check 7 certifies them sparsely. kbc_dec and kbc_cen ran on
  11c4073 (tree snapshots of the commit, launched with `tools/mimo_n100.sh`).
  at_dec_p is the 528be56 rerun of the c4cf908 false best (6.21). A container restart (exit 137) killed the 528be56 rerun of at_dec (not
  needed: the c4cf908 record passes every check) and the K+B+C centralized run of
  11c4073 at round 5 (4.548733); the latter was resumed from its checkpoint (same
  snapshot, same flags; `=== container restart` in its run.log).
- Diagnostic runs kept: `at_dec_diag_newcode` (6.14), `kbc_cen_cont_diag` (6.15),
  `kbc_dec_c4cf908_sisoscale` (above),
  `siso_vs_1x1` (6.16), `at_dec_p_c4cf908_falsebest` (6.21),
  `kbc_cen_c4cf908_sisoscale` (6.22); first pass in `prev_0e7f1c1`.
- Figures regenerated on these records (`tools/mimo_prod_plots.py`):
  `figs/mimo_prod_sigma.png`, `figs/mimo_prod_fields.png`,
  `figs/mimo_prod_controller.png`.

## 6.21 A false best in the final runs, and the all-w certificate in the loop

The final at_dec_p run (code c4cf908) ended with the loop certifying 2.383534 over
[-616.57, 616.57] while the dense all-w check at save found **41.640080 at
w = +409.65 -- inside the final band** (the record stores the dense value). The
design had been accepted when the band was narrower; the band grew later (from
other candidates' high-gain branches) and the accepted design was never certified
again over it -- the "one candidate late" defect of 6.6 in another form, and the
resonance came from an eigenvalue left of -band_recut (a far branch with a large
residue), which the band heuristic does not see at all.

Fix (commit 528be56, `--cert-all-w on`, default): every full certificate of a
candidate that would be ACCEPTED (value below the current best) also proves its
value over ALL w -- the tail bound W of 6.19 (sigma_1 < g beyond W) and one more
level-set question on [-W, W] at g. A norm found outside the band is returned
(the candidate refused if it exceeds the best) and the band widened to cover it.
The rerun of at_dec_p caught the same peak at round 2 ("all-w certificate: peak
41.640080 at w = +409.650 outside the band"), refused that candidate and went on.
Cost: one extra level-set call per accepted round (SISO K+B+C ne=100: +6 s a
round, 5 rounds 79.6 s against 49.9); SISO results unchanged (identity 8.754365110002194).
With it the band only steers the working set; the certified value is global at
every accepted design, not only at save.
It fired again in the final K+B+C centralized run (round 18: 4.055491 at
w = +42.455 outside the band; 6.20).

## 6.22 The MIMO scale rule, measured

6.16 made the MIMO packer use the SISO scale rule (max(|theta0|, 1e-3 x max |theta0|))
so that m = p = 1 reproduces the SISO path. On K+B+C central that rule is
harmful: a weak loop (kbc_dec: |D_c| 0.6 and 0.0055) has small nonzero entries
that the rule scales by their own size. From the SAME start (the first-pass
decentralized design, 5.566), three rounds:

| code | round 0 | 1 | 2 |
|---|---|---|---|
| first pass (0e7f1c1, per-group floor, no gauge) | 5.507 | 5.236 | 4.941 |
| SISO rule + gauge (c4cf908) | 5.500 | 5.449 | 5.393 |
| SISO rule, no gauge | 5.525 | 5.443 | 5.418 |
| per-group floor + gauge (11c4073) | 5.493 | 5.261 | 5.042 |
| per-group floor, no gauge | 5.501 | 5.289 | 4.971 |

The line-search steps told the same story (1/128-1/256 against 1/4-1/2). Fix
(11c4073): the SISO rule at m = p = 1 only; for m + p > 2 every entry gets the
floor 0.1 x its group's median nonzero magnitude (at least 1e-3 x the group's
max). The gauge is neutral here and kept. The atoms cases (static gains) are
unaffected (identical scales). The final K+B+C central run (6.20) is on 11c4073,
from the c4cf908 decentralized record; the c4cf908 central run (SISO rule) is kept
as `results_mimo_prod/kbc_cen_c4cf908_sisoscale` (5.070 after 20 rounds, stopped).

## 6.24 Discussion (user, 2026-09-29 23:00): symmetry, the set of optima, a K S bound, run time

- Symmetric designs early on: symmetry was never imposed (records: `sym` False).
  The starts were self-dual (the Gaussian pairs at x_a = -x_s; decentralized, so
  K = K^T), and gamma is invariant under the duality, so at a self-dual point the
  gradient, the working set and the SQP step are self-dual too: the iteration stays
  on the symmetric set up to rounding. It leaves only where the symmetric set is
  unstable for the iteration (K+B+C centralized: 0.083 first pass, 0.158 final).
  Symmetry gives no convexity. Restricting to the symmetric set halves the
  parameters, and its critical points are critical for the full problem
  (symmetric criticality). It gives no uniqueness: an asymmetric optimum always
  comes with its dual as a twin of equal gamma, and a UNIQUE optimum would have to
  be self-dual.
- The set of optima at the two-pair floor is a continuum. Several different fields
  reach it (at_dec vs at_dec_p). For given fields every gain beyond the knee gives
  the same gamma (x10, x1e4 unchanged; 6.17). Nothing selects a bounded gain; the
  gains found (5028, 42052) are where the iteration happened to stop.
- A bound on || K S ||_inf (|| W K S ||):
  - it removes that degeneracy directly: at w -> inf, G -> 0 and K S -> D_c, so
    || K S ||_inf >= || D_c ||;
  - it also bounds the high-gain branches that set the certificate's tail W and
    the band (W = 49272, level set 378 s for at_dec_p against W = 590 for K+B+C
    central) and the bad scaling of the stall of 6.10.
- Its cost:
  - rows: K S (i w) = Gam(i w) = K (I - G K)^-1 is already formed in the batched
    MIMO rows (6.16); G = C^H R B comes from the same solves; it is an m x p
    matrix: a sigma_1 of a 2 x 2 per frequency, with Clarke blocks like the
    performance rows;
  - no separate reduced model of G is needed: the reduced oracle's union-Galerkin
    basis (resolvent solves against b and c at the working-set frequencies) is
    rebuilt every round, so it follows B and C; G~(i w) = c~^H R~ b~;
  - certification over all w: the same closed-loop pencil, the same sparse level
    set, with m inputs and p outputs; one more call per accepted design;
  - the new part: its peaks lie elsewhere (roll-off, high w), so more rows in the
    working set.
  - Cheaper proxy: || K ||_inf of the r-dimensional compensator (dense, trivial),
    which does not see the closed-loop amplification near resonances.
  - Not needed if the goal is the best achievable gamma (a performance limit): the
    flat directions do not change gamma. Needed if the designs must be realisable,
    or if the high-gain branches drive the certificate cost at scale.
- Run time: the 5.3 h run is ne = 100, the hardest case (2 x 2, centralized
  order-16 compensator, 98 free complex entries). The dimension is not the driver.
  The driver is the working set (|Omega| up to 1085, ~340 sigma_2 rows, ~100
  certificate calls a round) on an equalised optimum. Rounds grew from 2.5 to
  15 min with |Omega|; the other cases took 5-186 min, SISO ne = 200 31 min. At
  backward-facing-step sizes each certificate call carries a 2-D/3-D sparse LU, so
  per-round cost is what to cut first: row pruning, frequency-parallel solves,
  a hard wall. No ne >= 400 production runs in the sandbox (user); the ne = 400
  runs of this pass were validation only (plant counts, the 107 s certificate).

## 6.25 Where a K+B+C centralized round goes (user, 2026-09-29 23:13: "something is taking too much time at such low dimensions")

**Pruning was already tested (my 6.24 suggestion was stale).**
- STAGE3 5-6: a cap of ~50 frequencies cost gamma (10.16 against 9.41). Its
  "the working-set size is not where the cost is" was stated for the full-model
  oracle.
- STAGE5 28: pruning to the local maxima (401 -> 7) stalled on the flat optimum:
  the rows between the ripples let a step lower the whole plateau.

**Profile (py-spy).**
- Reproduction: one round from the final record with a SYNTHETIC checkpoint
  holding the round-28 working set (1225 frequencies, 255 followed members, 347
  sigma_2 blocks; scratch `mkck.py`).
- Round length: 742 s. The SQP takes 86 % of it: 99 evaluations and 40
  iterations, ~6.5 s per evaluation.
- Split of the round (after the batching below):
  - reduced-model rows (gain_batch): 44 %; that is 1225 frequencies x (reduced
    dimension 128)^2 triangular solves x ~3 subspace iterations per evaluation --
    arithmetic, not overhead;
  - QP (Clarabel; ~1225 linear rows, 347 SOC blocks, the LMIs): 24 %;
  - gradient assembly (grads_mimo, 5 per sigma_2 block): 14 %.
- The dimension n does not enter the SQP (reduced model). The certificates,
  winding counts and the band are ~10 %.
- Why 1225 rows: the optimum is EQUALISED. 788 rows lie within 1 % of gamma and
  909 within 5 %; 1036 rows are in |w| <= 16; median spacing 0.008. Every row
  near the level is a constraint the step must respect, and STAGE5 28 shows the
  step needs them.
- Anytime behaviour:
  - kbc_cen: within 10 % of its final value after 99 min (round 17); 5 %: 142 min;
    2 %: 212 min; 1 %: 236 min.
  - kbc_dec: 5 % after 11 min, 2 % after 76 min, 1 % after 119 min.
  - Early kbc_cen rounds are short (2.5 min at |Omega| ~ 180) but gain only a
    few % each.

**Changes (same mathematics; checked against the per-row path).**
- `lu_counter._caller`: a frame walk (sys._getframe) instead of
  `traceback.extract_stack`, which read source lines on every counted solve.
  Site strings are identical. It cost 6 % of a round at |Omega| ~ 180.
- sigma_2 blocks batched:
  - `reduced.gain_batch(..., k=2)`: the leading k pairs in one subspace
    iteration, block size k + 4, all k residuals converged; `_batch_pairs`
    assembles the pairs;
  - the enrolled rows' peak following goes into `follow_batch`.
  - Per-row path kept when rows are not batched.
- Measured, same start, same flags:
  - 2 rounds at |Omega| ~ 175: same t to 5e-9 (3.7147118 / 3.7135816 against
    3.7147118 / 3.7135819), 29.2 + 47.9 s -> 25.1 + 39.0 s;
  - 1 round at |Omega| = 1225: t 3.713568 (old) / 3.713573 (new), both certified,
    742 s -> 642 s (-14 %).
- SISO identity (8.754365110002194), frozen path (10.23128516441147), witness
  invariant: PASS.

**Reading.**
- The time of the centralized case is the equalised plateau: rows x
  evaluations x rounds.
- A || K ||_inf bound does not address it: the rows are in |w| <= 16, where
  |K(iw)| ~ 10-300. The peaks of |K| (~5000) are at w ~ 60, where 40 of the 1225
  rows lie.
- Levers, by expected effect:
  1. frequency-parallel row evaluation (rows are independent; linear on the
     cluster);
  2. rows below the level evaluated lazily, with a margin re-checked at
     acceptance;
  3. QP warm start / row aggregation;
  4. a time budget with the anytime numbers above;
  5. fewer design parameters (a lower-order compensator flattens less).

## 6.26 Parallel rows, the QP, finite differences (user, 2026-09-30 10:07)

**Terms.**
- A ROW is one frequency of the working set Omega: sigma_1 there, a constraint
  sigma_1(w) <= t of the epigraph QP.
- The working set is everything the loop keeps (1225 frequencies late in the
  K+B+C central run). The ACTIVE set is the rows that bind at the QP solution:
  4 linear rows and 57 sigma_2 blocks (dual > 1e-3 max) of the dumped QP below.
- On the plateau 909 working-set rows lie within 5 % of the maximum and 316
  below. "Well below" (6.25) meant those 316, which still cost an evaluation
  each.
- "Batched" = all rows of one kind computed together in one vectorised subspace
  iteration (numpy / LAPACK over a stack of frequencies), instead of a Python
  loop that calls the reduced model once per frequency.

**Finite differences.**
- Design gradients are analytic: the adjoint formula from the singular pair,
  checked against FD by check_mimo check 3. The SQP Hessian is BFGS. No FD in
  either.
- The only FD is the CURVATURE d2 sigma / d w2 in peak following: two
  evaluations at w +- 1e-5 when no secant value is carried. It costs 5.7 % of a
  big round (profile lines peakfollow.py:118-119). An analytic second derivative
  in w needs the full singular spectrum of the reduced operator; not worth it at
  5.7 %.

**Parallel rows (`--row-workers N`, reduced._RowPool).**
- N processes are FORKED from the driver when the reduced basis changes (a
  version counter in `_project`), so each holds the oracle without pickling it.
- A gain_batch call sends each worker the design (ports, compensator,
  frequencies); the worker rebuilds the per-design reduced loop (make_cl) and runs
  the serial gain_batch on its share while the driver does its own.
- Frequencies are assigned by w-block (0.064), so each process keeps warm
  subspace caches for its frequencies. Batches under 32 rows stay in the driver.
- Validation: same trajectory as serial (2 rounds: 3.71471179 / 3.71358191
  against 3.71471178 / 3.71358190; big round 3.7135748 against 3.7135729, both
  certified).

**Timing: the big synthetic round (1225 rows), each run ALONE on the 2-core
container.**
- Machine check: two independent single-thread LAPACK loops run in 2.68 s side
  by side against 2.56 s alone, so the two cores scale 1.9x.

| code | 1 core (inner / round) | 2 cores, `--row-workers 1` | speed-up |
|---|---|---|---|
| d2e77e2 minus cache/QP thread (first test) | 761.7 / 855 s | 604.0 / 694 s | 1.26 / 1.23 |
| d2e77e2 (placement derivative memoised, QP one thread) | 626.1 / 718 s | 467.8 / 555 s | **1.34 / 1.29** |

- Amdahl: the parallel part is the reduced rows (~45 % of the round); the QP
  (~22 %), the gradient assembly (~8 %), peak-following logic and the
  certificates stay serial. Ideal with that split: 1.29. So the 2-core figure
  is what the split allows, not a scaling defect.
- To approach 2x the serial part must shrink or go parallel:
  - gradient assembly in the workers (they would need the packer's state);
  - the second QP per SQP iteration: it is solved at the new iterate only for
    the BFGS multipliers; the textbook update uses the old QP's multipliers, which
    would halve the QP time. That is an algorithm change, to be A/B tested.

**QP (Clarabel, the dumped big-round QPs).**
- Size: 214 variables, 2716 rows (1308 linear, 347 SOC(4), 2 small PSD); A 70 %
  dense (411k non-zeros). 26-30 interior-point iterations, 2.2-2.7 s.
- Levers tried:
  - `max_threads = 1`: Clarabel's default uses all cores and was 28 % SLOWER
    than one thread on these QPs (median 2.44 against 1.76 s), with bitwise the
    same solution. Adopted everywhere (bit-identical; the SISO identity and the
    frozen path pass).
  - KKT solver: the default 'auto' is already faer; qdldl was 45 % slower.
  - Constraint generation (solve with the previously active and the 40 nearest
    rows, add violated rows, repeat): exact, but on the plateau a subset's step
    overshoots hundreds of rows that are slack at the full solution. It ended
    with ~1500 of 2716 rows and was slower cold (10-12 s) and only 25 % faster
    warm. Removed.
  - Looser gap tolerance: the solve stops on AlmostSolved at a gap ~1e-7 either
    way; no gain.

## 6.27 K+B+C centralized with an order-5 compensator (user, 2026-09-30)

**Start: reduced order-5 compensators.** `tools/reduce_compensator.py`.
- Plain balanced truncation of the compensator destabilised the loop at every
  order <= 8. From kbc_dec 5.339 the closed-loop max Re was +1.52 at order 5,
  +0.33 at order 8; from kbc_cen, +0.38 at order 5. Its error sits in the band,
  where the loop gain matters.
- CLOSED-LOOP WEIGHTED truncation (Enns) instead:
  - the error K - K_r enters the loop through M = (I - G K)^-1 G;
  - reachability gramian of the compensator states taken in the cascade M -> K,
    observability gramian from K;
  - A_r written in modal (diagonal) form, D_c kept.
- Order-5 closed-loop max Re from four sources:

| source | max Re at order 5 |
|---|---|
| kbc_dec 11c4073 (5.339) | +0.265 |
| kbc_cen (3.715) | +0.381 |
| first-pass kbc_cen | +0.078 |
| first-pass kbc_dec (5.566) | **-0.161**, grid gamma 14.38 |
| c4cf908 kbc_dec | -0.109, grid gamma 20.58 |

- The start is the first-pass decentralized design reduced to order 5
  (`results_mimo_prod/start_kbc_cen5.npz`; `tools/mimo_n100.sh kbc_cenR` builds
  order R the same way). Certified start 14.432880, anchor alpha -0.161.
- 5 compensator states, 32 free complex controller entries (the order-16 run
  had 98).

**Run.**
- Code d2e77e2, `EXTRA="--row-workers 1" tools/mimo_n100.sh kbc_cen5`, alone on
  the 2 cores.
- **3.967890** after 46 rounds, 88 min, progress stop (0.145 % over 8 rounds).
- Checks (check_mimo):
  - dense all-w level set 3.967890 (rel 2.9e-8: the certificate's rtol is 1e-6;
    the earlier records' witnesses sat on the exact top);
  - winding counts equal to dense; abscissa -0.4681 equal to dense;
  - FD at the witness 1.5e-9;
  - sparse all-w certificate W = 249.2, 5.2 s.

| | order 5 (d2e77e2, 2 cores) | order 16 (11c4073, 1 core) |
|---|---|---|
| final gamma | 3.967890 (stopped by the rule) | 3.715450 (stopped by hand, still -0.25 %/round) |
| vs SISO 6.399134 | -38.0 % | -41.9 % |
| vs decentralized 5.339317 | -25.7 % | -30.4 % |
| within 10 % of its final | 10 min (round 7) | 99 min (round 17) |
| within 2 % | 13.5 min (round 9) | 212 min (round 27) |
| within 1 % | 23 min (round 14) | 236 min (round 29) |
| per round, late | ~2 min at |Omega| ~ 500 | 13-15 min at |Omega| ~ 1000 |
| rows within 1 % of gamma (late) | 202 of 516 | 788 of 1225 |
| sigma_2 blocks (late) | 92 | 347 |

- The order-5 optimum is less equalised: a quarter of the rows are near the
  level, against two thirds at order 16. Rounds are cheaper (half the rows,
  a quarter of the blocks) and more productive.
- The parallel rows and the single-thread QP are worth ~1.5x of the ~7x per
  round.
- Cost of the lower order: 6.8 % in gamma (3.968 against 3.715; 3.793 in the
  first pass at order 16).
- Result record: `results_mimo_prod/kbc_cen5/`.

## 6.28 Code review before the cluster (user, 2026-09-30 13:33)

Four independent read-only reviews:
- the driver, split in two halves;
- the SQP / reduced model / packer / gradients;
- certificates / spectra / winding / closed-loop classes.

Their claims were checked against the code and against the lu_counter table
of the order-5 run (38,421 full-size LUs over 46 rounds). Their scratch
benchmarks at n = 1e5 and on a 2-D model problem (n = 14,400) are theirs.

**Correction to 6.25.** "The dimension n does not enter the SQP" was WRONG.
- Every row's singular pair is lifted to full length (x = V x~, z = W z~,
  M x in `_batch_pairs` / `gain_k`).
- Every gradient (`G` -> `grads_mimo` -> `_ProxyCL._reduce`) projects it
  straight back (V^H x; V = W orthonormal, so the round trip is the identity)
  and forms b^H z_p, c^H x_p and the placement terms in full length.
- ~3700 `G` calls per evaluation at 1225 rows. At n = 199 this is invisible.
  At n = 1e5, kd = 130 it is ~0.1 s per call (reviewer benchmark), i.e.
  minutes per evaluation.
- The row pool also pickles those n-vectors.
- All of it has an exact reduced form: port sides with rcl.b / rcl.c, placement
  terms with W^H Mdpsi once per design, gprime = Im(z~^H Mt x~ + z_c^H x_c).
- **Must be done before any 2-D case.**

**LU count by call site (order-5 run, 46 rounds).**

| call site | full-size LUs | share |
|---|---|---|
| winding count at acceptance (`winding_cert.py:234`) | 31,770 | 82.7 % |
| abscissa certificate | 1,714 | 4.5 % |
| Hamiltonian pencil, band | 1,184 | 3.1 % |
| Hamiltonian pencil, all-w | 1,240 | 3.2 % |
| gains in the certificate | 1,063 | 2.8 % |
| enrichment / compaction / eigen-enrichment | 927 | 2.4 % |
| band candidates, M factors | ~450 | 1.2 % |

At n = 199 each LU is ~0.1 ms. On the 2-D model the reviewer measured a
resolvent LU at 0.2-0.5 s and a Hamiltonian LU at 3.4-35 s. **At that size
the full-model work is the round**, and today all of it is serial.

**Fixed now.**
- **Warm subspace across block sizes** (`reduced._fit_subspace`). k = 1 rows
  cached 5 columns and k = 2 blocks 6 at the same keys, and each discarded the
  other's entry as the wrong shape.
  - Now the leading columns are used, or completed.
  - Big round: 626 -> 608.5 s inner (-3 %; the reviewer's 4x iteration count was
    on its own test model).
  - The SISO 5-round value changes by 3.6e-9 (8.754365078411837 against
    8.754365110002194): the first non-identical SISO result. Attributed by
    reverting this one change: identical again. `tools/check_siso_identity.sh`
    now requires agreement to 1e-7 and reports identity.
- **SISO warm-start search skipped for MIMO.** Its stability filter is a
  DENSE plant eig (`warmstart.py:57`, ungated): 160 GB at n = 1e5.
  - Dormant only because no matching SISO result files existed in `results/`.
  - Its borrowed-controller branch could also have put a SISO packer on a MIMO
    run.
- **Mass-matrix LU once per matrix** (`sparse_spectrum._m_lu`).
  - It was refactorised by highgain_estimates, closed_loop_tail, PlantPencil and
    every plant_rectangle call (5 per round, 20 per plant_heights).
  - s_far = lambda_max(H, M) cached per plant (it was recomputed by each of the
    17 plant_rectangle calls of plant_heights).
  - ClosedLoopPencil's M LU is now built on first use: the loop never tiles, so
    it was never used.
  - All bit-identical.
- **Sobolev LU (M + l^2 K) on first use.** The default --sob_len 0.8 factorised
  it for every case, K+B+C included, and every forked worker inherited it.
- **`stability_region`: the silent ev = [0] fallback on an ARPACK failure**
  (ARPACK 'LR' without shift-invert, fragile on stiff meshes) now falls back to
  the proven plant enclosure right of -1 and prints it.
- Checks: frozen path 10.23128516441147, witness invariant, hurwitz selftest,
  SISO agreement 3.6e-9: PASS.

**Must be done before 2-D scale (exact changes; A/B at ne = 100 to rounding):**
1. **Reduced-space gradients and pairs** (above): no n-length vector in the SQP
   and no n-vectors through the row pipes.
2. **Winding count: ~700 LUs per accepted round.**
   - Its knots (121 + pole windows) split the march into independent segments
     whose phase changes add: farm them to the process pool.
   - Tail frequency WT from ||M^-1 A|| ~ h^-2 (960 at ne = 100): replace by the
     mesh-independent numerical-range bound of 6.19.
   - Real 2-D plants: S(conj s) = conj S(s) halves the points.
3. **splu defaults everywhere** (no permc_spec). On the 2-D model the
   Hamiltonian LU went 35.1 s -> 3.4 s with MMD_AT_PLUS_A (+ symmetric mode) and
   the resolvent LU 0.51 -> 0.21 s. For cavity/step sizes, MUMPS/PETSc (stage 7
   plan).
4. **Certificate.**
   - The all-w question re-proves the band it just certified. Ask it only on
     [-W, wlo] U [whi, W]: exact, about half the Hamiltonian LUs.
   - `eigs_near` asks k = 8, ncv = 30, tol 1e-9 where the nearest pair sets
     the disk.
   - The witness climb is derivative-free Brent on the FULL model to xatol
     1e-11 (~23 LUs per round) although gain() returns g'(w): a few Newton
     steps, or the reduced model plus 1-2 full checks.
   - The 9 location shifts after a witness are independent: parallel.
5. **Enrichment.**
   - `_orth` is column-by-column BLAS2 with conj copies of the basis (~47 s
     per union step at n = 1e5, kd = 130; 1.8 s blocked).
   - The same frequency is factorised 2-4 times per round (peak sweep,
     witness, next round's enrichment) through an 8-entry FIFO shared with
     complex shifts, which must be 0 at cavity scale: build the enrichment
     vectors inside the peak sweep.

**Moderate waste (per round or per run), to do with 1-5.**
- A full certificate of the unchanged best design in no-progress rounds, only
  to get its peak list.
- band_from_closedloop twice per round on the same candidate.
- The start computes the same 241-point full sweep twice, plus D.hinf.
- The pool ships A and M with every continuation task.
- Save-time repeats of acceptance-time work (winding at -eps/2, all-w at the
  same level, alpha).
- Reduced sweeps as per-frequency loops instead of gain_batch.
- 9 G calls per sigma_2 block where 4 suffice.
- The Ts stack rebuilt per gain_batch call.
- Gate work on candidates that cannot be accepted.
- Plant enclosures at three lines where one suffices.

**Parallel structure for the cluster.**
- Rows (done, `--row-workers`): parallel over frequencies; the cost depends on
  kd and |Omega|, not n.
- Full-model tasks: independent LUs at different points/shifts. Winding
  segments, certificate intervals and shifts, enrichment anchors, the peak
  sweep, band tiles, plant heights per depth. Process-parallel; memory per LU
  is the limit at n ~ 1e6.
- QP: serial, size independent of n.
- The second QP per iteration (6.26): NOT to be pursued. It saves ~12 % of a
  round at ne = 100, its size does not grow with n, and it is an algorithm
  change.

## 6.29 The waste that grows with n: fixed and validated (user, 2026-09-30 15:31)

The user: fix and validate every item raised in 6.28 before any parallelisation,
in particular the five items that grow with n; the mass matrix never changes, so
factorise it once at warm-up and cache it. The backward-facing step (BFS)
FEniCSx code was attached so the end goal's cost is known; the target stays
Ginzburg-Landau until it is validated cleanly.

### What was done

Three commits and one batch:
- `b6123df` massops;
- `7e2c3d2` reduced-space rows;
- this section's batch: every other item below.

**Mass matrix (`massops.py`).**
- It owns M's LU: one complex LU (plus a real one on first use for
  smooth_unmap), keyed on the matrix object.
- It is factorised at warm-up, right after the plant is assembled.
- Seven call sites used to factorise M themselves, several on every call:
  - the certificate region;
  - the high-gain estimates;
  - the all-w tail;
  - the plant enclosure and pencils;
  - the winding tail;
  - the field rows;
  - smooth_unmap.
- Bit-identical (A/B `ab_m`: every round value equal).

**1. Reduced-space rows (`7e2c3d2`).**
- Rows carry the reduced pair x~ = [x~_p; x_c], z~ = [z~_p; z_c].
- `gprime` = Im(z~^H Mt x~ + z_c^H x_c).
- Port sides use W^H B and V^H C.
- Placement terms use W^H M dpsi(x_a) and V^H M dpsi(x_s), once per design.
- Field gradients (atoms, bases) use projected Jacobians:
  - X = B^H M J, h = B^H M psi, y = Re(psi^H M J), once per design;
  - packed row = [Re(conj(s) u^H X) - t y] / nv.
- Near-line residuals use Gram matrices A V, M V (kd x kd products, no lift).
- The closed-loop Hermite block runs on the reduced pencil.
- The row pool sends each design once.
- No length-n vector remains in the SQP.

**2. Winding count.**
- The tail frequency now comes from the smaller of two valid bounds:
  - the old one, which uses ||M^-1 A|| ~ h^-2;
  - the mesh-independent ||M^-1/2 B|| ||M^-1/2 C|| / dist(s, W(A, M)), using
    the numerical-range heights of 6.19 at the counting line
    (`numrange_distance(..., line)`).
- K+B+C order 5: W_T 960 -> 480, 673 -> 633 points, same count.
- Atoms at_dec: W_T 122880 -> 30720.
- Still open:
  - The segment farm is the parallel phase.
  - The conjugate symmetry S(conj s) = conj S(s) does not apply at GL (complex
    operator and complex controller). At the BFS it applies only with real
    ports AND a real-coefficient controller (not the current packer).

**3. LU ordering (`sparse_lu.py`).**
- Every full-model factorisation calls `sparse_lu.factor`.
- `--lu-order colamd` is scipy's default: the frozen path, bit-identical.
- `--lu-order mmd_at_plus_a` is symmetric minimum degree with SymmetricMode, for
  structurally symmetric FE matrices.
  - Reviewer's 2-D model (n = 14,400): Hamiltonian LU 35.1 -> 3.4 s, resolvent
    LU 0.51 -> 0.21 s.
- MUMPS/PETSc goes in as another backend of `factor` (the callers only use
  `.solve(b, trans)`).

**4. Certificate.**
- The all-w question is asked only on [-W, wlo] U [whi, W]: the band was just
  proved at the same level.
- `--cert-near-k`: eigenvalues per decision shift. The nominal scripts use 2
  (only the nearest sizes the disk); 8 is the frozen value.
- `--cert-climb slope`: the witness climb uses the analytic g'(w) (doubling steps
  to a bracket, then a safeguarded secant on the slope). It takes ~10 full
  evaluations where Brent to 1e-11 took 30-40, and reaches the top of the dense
  sigma_1 exactly on the records tested.
- The peak sweep's LUs are pinned (`D._lu_pin`, at most 24) until the next
  round's enrichment consumes them. That enrichment now receives the exact
  frequency, not a rounded one, so the cache can hit.

**5. Enrichment.**
- `_orth` (`--oracle-orth block`):
  - two BLAS3 passes against the basis;
  - the new columns against each other only;
  - one final block pass.
- The deflation decisions are those of the column version: the same basis
  dimension on every test.
- At n = 1e5, kd = 130, +12 columns: 0.72 s against 2.89 s.
- A first blocked version with a pivoted QR decided differently at the 1e-8
  margin (basis 43 instead of 45) and was replaced.
- The input-basis images are one multi-RHS solve (bitwise the per-column
  solves).

**Moderate items.**

| item | fix | exact? |
|---|---|---|
| a no-progress round re-certified the unchanged best only to list its peaks | the best's peak list is kept from its acceptance (`_peaks_of_best`) | yes |
| band_from_closedloop twice per round on the same design | shift results memoised per design | yes |
| the start's 241-point full sweep computed twice | reused from the start certificate | yes |
| the pool pickled A and M into every task | `csp.register_plant` before the fork; tasks carry a tag | yes |
| save repeated acceptance work | winding: 0 right of -eps at acceptance implies 0 right of -eps/2; all-w reused for the same design and level; `D.hinf` (tens of LUs) skipped on the oracle path; the active count comes from the reduced gain curve | yes |
| reduced sweeps as per-frequency loops | `_sigmas`: one `gain_batch` call | rounding (2.4e-13) |
| 9 G calls per sigma_2 block | 6: both consumers read only Re Gc[a, a] and Gc[0, 1] + conj Gc[1, 0], and Gc[0, 0] is sigma_1's row | bitwise |
| the (i w BB - AA) stack rebuilt per call (331 MB at 1225 x 130^2) | `_TriStack`, built on access, Fortran order | bitwise |
| the gate ran on candidates that cannot be accepted | skipped when the certified value is not below the best (or inconclusive) | yes (same decisions) |
| plant enclosures tiled at three lines | the widest once at warm-up, the others filtered from it | same sets |
| eigen-enrichment repeated after a compaction | the compaction's eigenvalues are not factorised again | yes |
| the save-time winding count right of -eps/2 after a final band check that located every eigenvalue right of -band_recut | the band check runs first; complete with none right of -eps/2 implies the count 0 | yes |

**Not done, with reasons.**
- 4 G calls per sigma_2 block (6.28 said 4 suffice). With a real-valued
  gradient, h12 needs two calls. The polarisation form
  G(x0 + x1, z0 + z1) - G00 - G11 does give 4, but it is not bitwise. G no
  longer depends on n.
- The closed-loop Hermite block keeps 8 calls: it needs all four complex
  entries of dB.

### Validation

- **Unit checks.**
  - `tools/check_reduced_pairs.py` (MIMO): gradients reduced vs lifted 1.3e-14,
    g' 1.6e-13, batch vs per-row sigma 2.4e-13, Gram residual = direct residual.
  - `tools/check_reduced_fields.py` (atoms): 9.2e-13.
  - SISO proxy (scratch): 5.8e-14, g' 6.9e-15, batch 1.7e-14.
  - `_orth_column` is bitwise the old `_orth`. `_orth_block` spans the same
    subspace (smallest principal cosine 1 - 1e-15) with the same dimensions.
- **`tools/check_invariants.sh`: PASS.**
  - Frozen path 10.23128516441147.
  - Witness invariant 0 violations.
  - hurwitz 0/200.
- **`tools/check_siso_identity.sh`: PASS**, 8.754365072650645 vs
  8.754365110002194 (4.3e-9), with `--oracle-orth column`.
  - The script was fixed: the reference tree predates `--cert-near-k` and
    `--cert-climb` and failed on them, so its loop is stripped.
  - With the block orthogonalisation the same run gives 8.755548 (1.4e-4).
  - Attribution by elimination: the committed 7e2c3d2 gives 3.4e-9. This batch
    with the old `_orth` gives 4.3e-9, which holds every other change here,
    including the gate skip in the no-progress round 3.
- **A/B (`tools/ab_check.sh`, against 7e2c3d2), atoms case** (same trajectory,
  3.2e-12):

  | | 7e2c3d2 | this batch |
  |---|---|---|
  | full-model LUs | 2316 | 2105 |
  | certificate | 144 | 36 |
  | closed loop | 215 | 158 |
  | spectrum | 8 | 0 |
  | winding | 1949 | 1911 |
  | wall | 193 s | 128 s (-34 %) |

- **A/B, K+B+C order 5, 4 rounds:** the trajectory differs from round 0 at
  2e-6 and by 1-3 % in rounds 1-3 (5.296 vs 5.342 after 4 rounds; LUs
  5387 -> 4471).
  - The same happens with the old `_orth` and the old certificate flags
    (1.9e-7 in round 0). The remaining sources are rounding: the batched reduced
    sweeps and the enclosure from a wider ARPACK run.
  - 7e2c3d2 itself left f86d802 the same way: 9.5e-7 in round 0 from
    1e-14 gradient differences, 3 % by round 1.
  - This path (equalised plateau, degenerate QPs, homotopy splits) is not
    reproducible below the percent level under rounding changes. Each round's
    value is certified either way.
  - The meaningful MIMO test is therefore the complete run: see below.
- **Full order-5 run** (`tools/mimo_n100.sh kbc_cen5`, nominal flags, nothing
  else running):
  - Result: **3.985055** in 51 rounds, 7126 s. The 6.27 run gave 3.967890 in 46
    rounds, 5293 s, so the new value is 0.43 % higher.
  - The wall times are not comparable. The 6.27 run had `--row-workers 1`; this
    one ran the rows in one process by mistake (6.26: 1.29x on the big round).
    The paths differ too: |Omega| 659 vs 518 at the end (75 vs 92 sigma_2 blocks).
  - `tools/check_mimo.py` passes on the record:
    - sigma vs dense 7.1e-14;
    - m = p = 1 vs SISO 1.1e-14;
    - FD gradients 9.4e-6 of the largest;
    - gamma vs the dense level set 4.7e-15;
    - winding = dense;
    - abscissa = dense;
    - sparse all-w certified.
  - **Path sensitivity, measured on the OLD code.** e34171e, which produced
    3.967890, rerun from the same start with the static gain multiplied by
    (1 + 1e-12): **3.996517** in 44 rounds, 4855 s (`--row-workers 1`, like 6.27).
    - Round 1 is already 8.722 against 9.133.
    - At round 21: original 3.993, perturbed 3.990, new code 4.080. At the stop:
      3.967890 / 3.996517 / 3.985055 (spread 0.7 %). The new code's value lies
      inside the old code's own spread.
    - Per round: old 115 and 110 s with parallel rows, new 140 s with serial
      rows (x 1/1.29 = ~108 s).
    - Records: `results_mimo_prod/kbc_cen5_v629` (new code) and
      `results_mimo_prod/kbc_cen5_e34171e_pert12` (old code, perturbed start),
      without the dumps.
  - The order-5 central problem's end value depends on the path at the
    0.5 %-level, whatever the code. Comparing codes by one run each is
    meaningless at this level: the tests above are the exact ones. Starts and
    repeats belong on the cluster.
- **The save-time winding implied by the final band check:** atoms case,
  bitwise the same values, 2105 -> 1532 full-model LUs (the 573-point count at
  -eps/2 is gone).

### Where the LUs still go (for the parallel phase)

At ne = 100 the winding counts are 90 % of the full-model LUs (atoms case, above):
- at acceptance: 627;
- in the save-time band check: 711;
- at save: 573.

Each count is a march over independent segments. That, the certificate's
decision shifts and intervals, the enrichment anchors, the peak sweep, the band
tiles and the plant heights are the process-parallel tasks. The reduced rows are
already parallel (`--row-workers`) and do not grow with n.

### The backward-facing step, for scale (attachment dsr_validation.zip)

- Taylor-Hood P2-P2-P1 on ~60k triangles, the Dergham-Sipp-Robinet case.
  - About 192k complex unknowns.
  - J = -dF/dW is real.
  - M is the VELOCITY mass: singular on the pressure block.
- One MUMPS LU plus the gain eigenproblem: ~40 s per omega on 2 cores, ~1.5 GB.
- Consequences:
  1. The M-norm bounds (tail constants, beta of the winding tail, the
     numerical range) must be taken on the velocity subspace. `massops`'s LU
     does not exist for a singular M: open item, stage 7.
  2. A winding count of ~600 points is ~600 such LUs: hours serial, minutes on
     one 96-core node (a few GB each, 512 GB per node).
  3. The real operators halve the frequencies only for real ports with a
     real-coefficient controller.
- On the cluster (44 x 96-core / 512 GB, 2 x 2 TB, 4 x Genoa 384 GB, OPA
  100 Gb/s), the full-model tasks are independent LUs at different shifts: one
  task per core (or a few MUMPS threads each), no communication but the
  results.
- GPUs:
  - they do not help the dominant cost: sparse complex FE LUs at many
    independent shifts, memory-bound;
  - the reduced model's dense work (batched kd x kd triangular solves, QZ) is
    small;
  - a GPU sparse direct solver (cuDSS) could be timed on one BFS matrix later,
    against 96 CPU cores doing 96 shifts at once.

## 6.30 Parallel full-model tasks (user, 2026-09-30 21:10)

The user asked for three things:
- parallelise the parts that matter at the backward-facing step (BFS);
- validate on GL on 2 cores;
- measure the speed-up and say how it changes with the dimension.

Running on 1 task must give the same result as before.

### Design

**`fullpar.py`** is one process pool for every full-model task.
- `start(nproc, D=..., Phi_b=..., Psi_c=...)` forks the workers.
  - Every task gets the plant, its mass LU and the reduced oracle's input bases by fork inheritance. Nothing large goes through a pipe except the ports.
  - The driver forks the pool at warm-up, right after the mass factorisation, so the plant enclosure and heights can use it.
  - It re-forks before the rounds, once the counters, the certificate settings and the input bases exist.
  - `pmap` re-forks by itself when the mesh object changed. The MIMO setup sets the compensator order after the warm-up fork: the first version failed on exactly that (`reshape (8,2)`).
- `pmap(fn, args)` returns `[fn(a) for a in args]` in order, one task at a time, with dynamic scheduling.
  - With `--nproc 1` there is no pool, and every call site runs its OLD serial code.
  - Each task is a pure function of its arguments. There are no warm starts, and ARPACK start vectors are fixed. So a result does not depend on the process that computed it.
- LU accounting: each worker returns its lu_counter delta, which is merged under the submitting driver line. The continuation gate's tracked queries now go through `pmap` too, so they are counted; before they were not.
- `fullpar.report()` prints, at the end of a run, the pool's task time against its wall time.

**What runs in the pool (all bitwise the serial result unless stated).**

| task | where | parallel unit | exact? |
|---|---|---|---|
| winding count | `winding_cert.winding_count` -> `_march`, `_seg_task` | segments between knots (4 per worker, at most knots/4) | count and ok: yes. Points: +-1 %, since each segment restarts its step rule |
| certificate decision | `certify_sparse.certify_greater` | one GENERATION of critical intervals at a time (the FIFO queue is generation by generation); results merged in queue order | yes: same shifts, disks, counters, witness |
| location shifts | `hinf_certified_sparse` | the 9 shifts | yes |
| candidate samples | `hinf_certified_sparse(sigma_many=...)` | frequencies | yes |
| full sigma sweeps | `certified_norm` (peak reference, 241-point sweep), start seeding (`fullpar.sigmas`) | frequencies | yes |
| enrichment | `ReducedOracle.enrich_many` / `enrich_columns` / `_enrich_task` | anchors: columns (LU, pairs, Woodbury, input-basis images) in the workers, basis updates in order here | yes |
| eigen-anchors | `enrich_eig(bc=...)`, `compact(bc=...)` | eigenvalues | yes |
| band refinement shifts | `sparse_spectrum.closed_loop_right_of` | shifts | yes |
| band / plant tiles | `eigs_in_rectangle(spec=...)` | top-level tiles | yes |
| plant heights | `plant_heights` | depths (s_far computed first, then handed to the workers) | yes |

Other points:
- **Projection once per anchor batch.** `enrich_many` runs the basis updates in order and computes the projection (A V, M V, the Gram matrices, the QZ form) once, after the last anchor. It depends on the final basis only, so the result is bitwise the same.
- **Not parallel, and why.**
  - The witness climb is sequential (about 10 evaluations).
  - The abscissa certificate keeps its own `--cert-nseg` segments on the same pool.
  - The SQP is n-independent; its rows have their own pool (`--row-workers`). The two pools never run at the same time: rows during the SQP, full-model tasks during certification.

**Options for the cluster.** Both default to off; the defaults are bit-identical.
- `--cert-split N` starts the certificate's queue with N intervals. With the default 1 the certificate has only about 4 shifts per generation on average (measured below), so wider pools idle. It uses other shifts but gives the same rigorous decision.
- `--oracle-union new`: the union Galerkin step projects only W's new columns. When V = W, W's leading columns are V's and deflate to nothing anyway.
  - Measured at n = 25,600, 8 anchors: the basis updates take 4.9 s instead of 7.8 s.
  - Dimension 272 either way, subspace cosine 1 - 2e-15, reduced sigma equal to 4e-15.
  - The vectors differ at rounding level, so it is not the default.

### Validation

- **`tools/check_parallel.py`** (new) runs every pooled task on the order-5 K+B+C, atoms and SISO K+B+C records:
  - sweeps;
  - certificate;
  - enrichment basis;
  - closed-loop set right of -1 (refinement shifts, tiles, count);
  - plant enclosure and heights;
  - winding counts.
- **Previous commit vs 1 task:** ALL IDENTICAL, including the winding points.
- **1 task vs 2 tasks:** ALL IDENTICAL. The winding points differ by at most 7 of 600-800; the counts are equal.
- **A/B runs** (`tools/ab_check.sh`, new `AB_EXTRA`):
  - Previous commit `--nproc 1` vs new `--nproc 1`: every round value bitwise equal, full-model LUs 4759 = 4759 and 1538 = 1538.
  - New `--nproc 1` vs `--nproc 2`: every round value bitwise equal. LU counts 4741 and 1555, because the winding segment boundaries and the gate queries are now counted.
- **`check_invariants.sh`** with the pool on: PASS (frozen 10.23128516441147, witness invariant, hurwitz).
- **`check_siso_identity.sh`** with the pool on: 8.754365072650645, as before (4.3e-9 to siso-v1).
- **Full order-5 run with the pool:** same flags as the 6.29 run (`tools/mimo_n100.sh kbc_cen5`; `--nproc` defaults to 2, rows serial).
  - The sandbox restarted at round 22, which killed the run.
  - Every one of rounds 0-22 is bitwise equal to the 6.29 run (t and certified value from the dumps).
  - Wall at round 22: 2431 s against 2581 s (-5.8 %). The 6.29 run used the pool only for the gate's tracked queries.

### Speed-up on 2 cores, and how it changes with n

**GL ne = 100 (n = 199), whole runs.** Order 5, 23 rounds: 2581 -> 2431 s (x1.06). A/B, 4 and 2 rounds:
- K+B+C: 202 -> 197 s. The pool did 14.0 s of task work in 8.0 s of wall (x1.74).
- Atoms: 127 -> 124 s. The pool did 12.8 s of task work in 8.2 s (x1.56).
- At n = 199 a factorisation costs 0.1 ms, so the pooled work is only 7-10 % of a run. The rest is the reduced SQP, which does not depend on n.

**The same task code on a 2-D operator** (`tools/bench_parallel_2d.py`: 5-point convection-diffusion-reaction on N x N, 2 actuators, 2 sensors, the order-5 compensator, `--lu-order mmd_at_plus_a`). The results are bitwise identical serial vs pool at every size. Speed-ups are serial time / pool time, from two runs; the second run is the final code:

| n | one gain | sweep | winding | certificate | enrichment |
|---|---|---|---|---|---|
| 1,600 | 0.013-0.021 s | 1.50 / 1.46 | 1.57 / 1.86 | 1.54 / 2.01 | 1.03 / 1.30 |
| 6,400 | 0.053 s | 1.69 / 1.62 | 1.69 / 1.61 | 1.54 / 1.46 | 1.05 / 1.07 |
| 25,600 | 0.26 s | 1.81 / 1.80 | 1.65 / 1.71 | 1.64 / 1.55 | 1.07 / 1.10 |

(The 1,600 row is noisy: tasks of 10-20 ms.)

**Direction: the speed-up grows with n, towards a bound set by each task's structure, not by overhead.**
1. **Overhead fades.** A task's overhead is the dispatch (~1 ms) plus pickling the ports, which is O(n). Its work is a 2-D LU, O(n^1.5) with nested dissection (measured ~n^1.15 from 6.4k to 25.6k). So the overhead fraction falls with n. Pure LU tasks (the sweeps) go 1.46 -> 1.81 of 2.
2. **Structural bounds remain.**
   - Winding: the boundary points (1 extra LU per segment) and the imbalance of 8 segments. ~1.7 of 2; on N workers at most min(4N, knots/4) segments, ~40-60 on GL.
   - Certificate: the width of the generations. Measured on the ne=100 records, including the 9 location shifts: widths [9,1,9,1,2,4,4,4,6,4,2] and [9,1,2,4,4,4]. That is 46 and 24 shifts in 11 and 6 generations: at most ~4x on any number of workers. `--cert-split 32` widens them to 15x and 12x, for 2.3x and 2x the shifts.
   - Enrichment: the basis updates stay serial (O(n kd^2) per anchor). Only the LUs (20-30 % of the enrichment at these sizes) are parallel: ~1.1 on 2 cores. `--oracle-union new` removes ~40 % of the serial part.
3. **The run becomes the full-model tasks.** The SQP does not grow with n, and the full-model work goes from 7-10 % of a run at n = 199 to essentially all of it at the BFS. The end-to-end speed-up therefore rises from ~1.03-1.06 at GL ne=100 to the task-level bounds above.

**At the BFS** (n ~ 192k, one LU ~ 10 s, ~1200 full LUs per round on the order-5 run's mix):

| component | share of a round's LUs | bound with 96 workers |
|---|---|---|
| winding counts | ~80 % | ~40-60x (segments) |
| gains (peak reference, enrichment anchors, climb) | ~20 % | ~12-20x (frequencies per call) |
| certificate (Hamiltonian LUs of size 2n, ~3-10x a resolvent LU) | small | ~4x, or 12-15x with `--cert-split` |

A round would take ~3 h serial and roughly 10-15 min on one 96-core node. Memory is one LU per worker: 96 x 1.5 GB fits in 512 GB.

Levers beyond this pass, if needed:
- MUMPS inside `sparse_lu` (fewer, faster LUs per worker);
- fewer points per winding count;
- the widened certificate queue.

## 6.31 Children against parents: the lattice check at ne = 200 (user, 2026-09-30 23:59)

The user asked:
- run the 3-round check of every child against its parents in the sandbox, at ne = 200;
- fix the minimum atom width to 0.4.

### The check (`tools/lattice_check.py`, new)

**Protocol.** A child resumed from a parent's record starts at that parent's design and its best value never rises, so it ends at or below the parent. Convergence is not needed to test this, so each run is short:
- **Roots** (B, C, KGC, KSP): one cold run each. KSP uses the first usable seed (`lqgbt`).
- **Children:** one run from EVERY parent's record, each in its own directory. The case's record is its best attempt.
- No donors, no polish, 3 rounds.
- 2 attempts run at a time, each with `--nproc 1`.

**Checks, per (child, parent):**
- *inclusion*: the start value equals the parent's value;
- *descent*: the end value is at most the start value;
- *nesting*: the child's best is at most the parent's value.

**Flags.**
- The nominal loop (`tools/n100_kbc.sh`) at ne = 200.
- Field cases add 4 Gabor atoms of width >= 0.4 and the H^1 row at 1.8. The Laplacian modes (160) are only the reduced model's input basis.
- Width 0.4 is the actuator's own width, so a parent's Gaussian is an admissible field: fit error 1e-12, against 0.17 at width 1.0.
- The row has to go from 1.0 to 1.8 because the Gaussian's RMS wavenumber is 1.759; at 1.0 the row would still exclude the parent.

### What it found, and the fixes (each found by the check, fixed, the check rerun)

1. **The certificate's far band.** The roots B and C never moved: end = start = 10.7589, every improved candidate (10.14-10.18) refused as INCONCLUSIVE.
   - Cause: the all-frequency question on [30, W = 608]. Far from the spectrum, shift-invert Arnoldi on the even pencil does not converge (18 of the shifts), and strict disks left the interval uncovered. This is the far-band hole of STAGE5 30.5. SISO designs at ne = 200 have W ~ 600, the MIMO ones ~200.
   - Fix: `SparsePencil.free_radius`. When no eigenvalue converges, the disk comes from a norm: |lam - s| >= 1/||(A - sE)^-1 E||_W with W = |E_H| = blockdiag(M_cl, M_cl).
     - It is a Hermitian Lanczos on the factors already held: 0.02 s, and it always converges.
     - It is provably eigenvalue-free, though smaller than the true distance.
     - It is used before the k = 4 retry, which cost ~1.5 s and mostly failed out there.
   - In strict mode the first Arnoldi is capped at 300 iterations instead of 800. A failing shift cost 3 s.
   - The far band now certifies in 30 s (it never did before). Serial and 2 tasks agree bit for bit.
2. **Shifts on a compensator pole.** A static gain (KGC) leaves A_c's poles in the closed loop, decoupled. Three solves hit them exactly and crashed with "Singular matrix":
   - the save-time band refinement (`_near_eval`);
   - the eigen-anchor (`eigvecs_at`);
   - the reduced Rayleigh-quotient tracking (`_pencil_solve`).
   
   Each now moves the shift by 1e-7 / 1e-10 relative. The solve is then a large multiple of the eigenvector, which is what the iterations need.
3. **The child did not start from the parent's design.**
   - Fields: the resume rule inherits only groups both cases free. A child freeing a field its parent did not free (B -> FB, FB -> FB+FC for the sensor, K+B -> K+FB, ...) started from the DEFAULT field. FB from B started at 9.9949 against B's 10.1374; FB+FC from FB at 8.6106 against 9.8694. Now the child starts from the parent's field: its Gaussian at the parent's optimised or fixed position.
   - Controller: a child freeing a controller its parent kept fixed now takes the parent's controller as the default start. It is still compared with the borrowed one, and the lower is kept.
4. **The start was certified less rigorously than the rounds.**
   - The certificate's settings (strict disks, near-k, climb, cert-rtol 1e-6, all-w) were applied AFTER the start certificate. Every run's start was therefore certified with the frozen disks, rtol 1e-4 and no all-w question.
   - Effect: K+FB+FC from KSP+FB+FC started at 4.154741 against the parent's 4.156543, the same design measured less rigorously.
   - The settings now come first. Frozen path, SISO check and the ne=100 A/B are unchanged bit for bit; their starts were not affected.
5. **Atom fit.** The greedy fit started each atom at width 2 smin and stalled at a 2e-3 error on a Gaussian of the floor width, because ds/drho vanishes at the floor.
   - The width is now also tried from the residual's half-maximum width, and the better of the two fits is kept.
   - The parent's Gaussian is fitted to 1e-12. The old start is kept wherever it is better: width-1.0 fits are unchanged (0.17).
6. **Memory.**
   - The batched rows' warm-start cache stored a VIEW of each batch array, so every entry kept the whole batch alive. K+FB+FC at kd ~ 300 reached 5.6 GB at save and was killed (SIGKILL) twice. The cache now stores copies, bounded at 20,000 frequencies.
   - Batches above 512 MB are split into chunks: 4001 frequencies at kd = 307 now peak at 430 MB. GL ne=100 batches are below the threshold and unchanged.
   - Rerun: under 1 GB throughout.
7. **Slow sweeps at a large basis.** With two optimised fields at high gain, sigma_1 ~ sigma_2 and the batched subspace iteration needed more than 40 iterations. Each unconverged row fell back to a per-row ARPACK svds (~1 s at kd ~ 300): over 20 min for the save curve and ~10 min per round. The value sweeps (`_sigmas`) now allow 200 batched iterations; rows that converge within 40 are unaffected.

**Validation of the fixes.**
- Frozen path 10.23128516441147, witness invariant, hurwitz: PASS.
- SISO check: 8.754365072650645, unchanged.
- ne=100 A/B at 1 task: kbc5 and atoms bitwise equal to before, same LU counts.
- The free disk, serial vs 2 tasks: bitwise equal.

### Result

**All 54 (child, parent) pairs pass inclusion, descent and nesting** (`results_lattice_check_ne200/check_report.md`; 0 failed checks).
- Every child started exactly at each parent's certified value, to the 6 printed decimals.
- No attempt ended above its start.
- Every child's best is at or below every parent.

Four attempts could not move: they started at the parent's value and ended there. They sit at the single-pair floors 9.831876 / 9.660144 (the bounds file's 9.831865 / 9.660137 at the ne=200 bound positions).

The first report flagged these 4 attempts, plus the one killed at save. The flags came from the start being printed to 6 decimals (now the test's resolution) and from the kill (fix 6).

The frozen ne=200 table itself breaks nesting in three places:
- K+FB 9.831944 > K+B 9.831877;
- KSP+FB 9.832102 > KSP+B 9.831877;
- KSP+FC 9.660272 > KSP+C 9.660158.

The protocol now excludes this.

| case | check (3 rounds) | from | frozen ne=200 (reference) |
|---|---|---|---|
| B | 10.137367 | cold seed lqgbt | 9.944136 |
| C | 10.010604 | cold seed lqgbt | 9.917965 |
| KGC | 42.948630 | cold | 12.244295 |
| KSP | 10.009940 | cold seed lqgbt | 9.831883 |
| B+C | 9.674451 | from B | 9.674473 |
| FB | 9.869400 | from B | 9.831877 |
| FC | 9.660144 | from C | 9.660135 |
| KGC+B | 13.250065 | from KGC | 11.600868 |
| KGC+C | 12.872117 | from KGC | 11.719055 |
| KSP+B | 9.835678 | from B | 9.831877 |
| KSP+C | 9.684162 | from C | 9.660158 |
| K | 9.861398 | from KSP | 9.831878 |
| FB+FC | 4.566533 | from FC | 4.341229 |
| KGC+B+C | 11.547840 | from KGC+C | 11.411560 |
| KGC+FB | 9.831876 | from KGC+B | 9.831876 |
| KGC+FC | 9.660144 | from KGC+C | 9.660144 |
| KSP+B+C | 9.372717 | from B+C | 8.246535 |
| KSP+FB | 9.831876 | from KSP+B | 9.832102 |
| KSP+FC | 9.660144 | from FC | 9.660272 |
| K+B | 9.832954 | from KSP+B | 9.831877 |
| K+C | 9.666270 | from KSP+C | 9.660159 |
| KGC+FB+FC | 4.142350 | from KGC+FC | 3.925812 |
| KSP+FB+FC | 4.156543 | from FB+FC | 4.140174 |
| K+B+C | 8.838092 | from KSP+B+C | 7.964064 |
| K+FB | 9.831876 | from KGC+FB | 9.831944 |
| K+FC | 9.660144 | from KGC+FC | 9.660153 |
| K+FB+FC | 3.964539 | from KGC+FB+FC | 3.925456 |

These are 3-round values: a check of the guarantee, not optima. The roots are short cold runs: KGC is still at 42.9 from the open loop's 882.9.

**Cost.**
- 58 attempts, 6.7 h of single-core work, about 4.5 h of wall time on 2 cores, including the reruns after each fix.
- Median attempt: 2.5 min non-field, 8.4 min field.
- Longest: 91 min, K+FB+FC from KSP+FB+FC. Its reduced basis reached kd ~ 300 (two field input bases of 160 modes each, above the compaction cap of 120), and its band widened to +-320.
- Most of the 91 min was the record's 4001-point gain curve on the reduced model. At kd ~ 300 the batched subspace iteration costs ~7 GFlop per iteration over all frequencies.

**Open (for the cluster estimate).** Field cases at ne=200 have a reduced basis of ~300. That makes their rounds ~3x and their saves far slower than the 120-dimensional cases. Options:
- fewer input modes (80, as at ne=100);
- input bases restricted to the fields' support;
- a coarser stored curve.

The full lattice's field runs should be budgeted at 1-2 h each, not 1 h.

## 6.32 How many atoms do the earlier high-dimensional fields need? (user, 2026-10-01 09:33)

The question: instead of asking how well Laplacian modes cover an atom, ask how many atoms are needed to represent the optimal FB/FC fields found earlier with high-dimensional parametrisations.

Two roles, kept apart:
- **Design parametrisation.** The atoms (5 real parameters each, 20 per field at 4 atoms) are what the optimiser moves. The 160 Laplacian modes are not design parameters.
- **Reduced model's input basis.** Every reduced-model gamma needs the plant's responses to the fields being tried. The 160 modes make it exact for any field in their span; that is the kd ~ 300 of 6.31. A smaller alternative is the current atom fields plus their 20 parameter derivatives (21 columns per field), which covers the fields within a round's steps to first order. This changes cost only, not what the optimiser can reach.

**Fit study** (`tools/atom_count.py`, new; `results_atom_count/atom_count.md`, `.json`).
- `fieldbasis.fit_atoms`, N = 1..8 atoms, width >= 0.4 or >= 1.0, |wavenumber| <= 0.7071.
- Relative M-norm error, complex amplitudes free (phase and scale invariant).
- The fit is greedy and non-convex, so the errors are upper bounds.
- 16 minutes on 1 core; a first try with N up to 10 hit the 10-minute shell limit.

| reference (gamma) | width >= | N=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| H^2 kmax=1, 160 modes, ne=400 (3.914071), actuator | 0.4 | 0.294 | 0.244 | 0.217 | 0.199 | 0.162 | 0.146 | 0.130 | 0.124 |
| same | 1.0 | 0.294 | 0.244 | 0.218 | 0.201 | 0.164 | 0.157 | 0.148 | 0.130 |
| H^2 kmax=1, 160 modes, ne=200 (3.911740), actuator | 0.4 | 0.311 | 0.286 | 0.266 | 0.210 | 0.191 | 0.151 | 0.130 | 0.120 |
| unconstrained, 160 modes, ne=200 (3.888738; artefact, 4.605 on ne=400), actuator | 0.4 | 0.425 | 0.420 | 0.388 | 0.372 | 0.300 | 0.265 | 0.250 | 0.195 |
| same | 1.0 | 0.425 | 0.425 | 0.418 | 0.417 | 0.414 | 0.406 | 0.401 | 0.395 |
| frozen report design, ne=200 (3.925389), actuator | 0.4 | 0.260 | 0.199 | 0.178 | 0.140 | 0.123 | 0.106 | 0.092 | 0.064 |
| 4 atoms width >= 1 (A2'', 3.876296), actuator (check) | 0.4 | 0.193 | 0.121 | 0.035 | 0.000 | 0 | 0 | 0 | 0 |

The sensor fields give the same numbers to within 0.03; all rows are in the .md.

**Why the H^2 fields need many atoms.** The fields' content, measured with an FFT of the nodal values (actuator):

| field | energy at abs(k) > 1 | energy at abs(k) > 2 | spatial spread (RMS) |
|---|---|---|---|
| H^2 ne=400 | 0.036 | 0.019 | 6.3 |
| H^2 ne=200 | 0.049 | 0.019 | 7.8 |
| unconstrained ne=200 | 0.144 | 0.133 | 13.8 |
| frozen | 0.022 | 0.006 | 3.8 |
| 4 atoms (A2'') | 0.012 | 1e-4 | 2.4 |

- The H^2 fields have long, low-amplitude tails, with a spread of 6-8 against 2.4 for the atom design.
- About 2 % of their energy sits in the k ~ 2-2.6 wiggles (32.6). This alone is an error of sqrt(0.019) = 0.14.
- 8 atoms reach 0.12: the main part, and some of the wiggles through the width-0.4 atoms' broad spectra.
- The unconstrained fields carry 13 % of their energy at abs(k) > 2. They are not representable at width 1 (0.40 at 8 atoms), which is consistent with their being mesh artefacts (STAGE5 32.7).

**The fit error is not the yardstick for gamma.**
- The 4-atom optimum (3.876296; 3.879283 on ne=800) is below the H^2 optimum (3.914071 at ne=400; 3.915962 on ne=800). It is 0.16-0.24 away from the H^2 fields (STAGE5 32.6).
- So the high-dimensional fields are not an optimum for the atoms to copy.
- Gamma is very sensitive to the fields with the controller held (a 0.18 fit error took gamma 3.93 -> 6.83, 32.6). The value of an N-atom fit is therefore known only after re-optimising.
- The number of atoms is set by the optimised gamma against N: one K+FB+FC run per N at ne=200, width >= 0.4. Starts:
  - for N <= 4, the fit to A2'';
  - for N > 4, extra atoms with nonzero amplitude. An atom of zero amplitude has zero gradient in its centre, width and wavenumber.

## 6.33 Atoms as the reduced model's input basis: `--field-input tangent` (user, 2026-10-01 10:10)

The user's question: why the Laplacian modes at all, if the fields are atoms? There is no reason. The modes came with the field-on-modes parametrisation (STAGE5 32), where they were the design basis and so the natural input basis. When atoms replaced them as the design parametrisation (32.6), the 160 modes stayed as the model's input basis. That is the kd ~ 300 of 6.31.

**What the input basis has to contain.** Within a round the SQP evaluates gamma for fields b(theta + delta) near the round's design b(theta). To first order these lie in span(M J), where J is the atoms' Jacobian (`fieldbasis.atom_jacobian`, already used for the gradients since 6.29).
- In the complex span, the two amplitude columns are one column: the atom phi.
- Centre and wavenumber give the same direction, (x - c) phi.
- The width gives (x - c)^2 phi.
- So the span is 3 columns per atom: 12 per field at 4 atoms, against 160 modes.

**Implementation.**
- `--field-input {modes,tangent}`, default `modes`, so every existing path is unchanged.
- With `tangent`, the field side's input basis is orth(M J) at the round's design (SVD, relative cutoff 1e-12; it drops exactly the 2 dependent columns per atom). It is rebuilt at the start of every round.
- The position side (B/C cases) keeps its range basis.
- `ReducedOracle.set_input_basis(..., per_design=True)`: with the full-model pool up, the enrichment tasks carry the bases in their arguments, since the workers' forked copy would be stale.
- `tools/lattice_check.py` now passes `--field-input tangent` for field cases. The 6.31 check ran with the modes.

**Measured** on K+FB+FC at ne=200, from the lattice check's KGC+FB+FC record, 3 rounds, 1 core:

| input basis | kd | gamma after rounds 1 / 2 / 3 | model vs certificate at the witness | wall |
|---|---|---|---|---|
| 160 + 160 modes (6.31 check) | 299-307 | 4.076221 / 4.003424 / 3.964539 | 1e-15 .. 5e-14 | 809 s |
| none (`--oracle-inputs off`), 2 rounds | 16-82 | 4.072165 (refused, model error 2e-4) / 4.072165 (refused) | 2e-4 | 78 s |
| atom tangent, 12 + 12 | 94-126 (compacted at 120) | 4.076221 / 4.003428 / **3.963170** | 3e-15 .. 3e-13 | **191 s** |

- The tangent basis follows the modes' path: the same first two rounds to 4e-6, and a slightly lower third round. The model is exact at the witness to 1e-13.
- 4.2x faster per 3 rounds. Without any input basis the model is 2e-4 off and both rounds are refused.
- 2 workers give the same record bit for bit (2 rounds: 4.003428128817293 on 1 and on 2 workers; pool x1.63).
- Frozen path 10.23128516441147, witness invariant 0, Hurwitz selftest 0/200 (`tools/check_invariants.sh`).

**Where the remaining time goes** (profile, 3 rounds, 225 s with the profiler):
- inner SQP on the reduced model: 125 s, of which the batched singular values take 96 s;
- full-model certificates: 58 s;
- plant heights: 24 s, once per mesh;
- closed-loop band: 22 s.

That is about 1 min per round. Earlier atom runs converged in 12-21 rounds (STAGE5 32.6), so about 15-25 min per run at ne=200 on 1 core, against the 1-2 h budgeted in 6.31-6.32 with the modes.

**Well-posedness of the atom family** (for the user's point on constraints):
- Width collapse: excluded by construction (s = smin (1 + rho^2)). But ds/drho = 0 at the floor, so an atom AT the floor has a zero width gradient: the floor is stationary for any objective. In the KGC+FB+FC start design, 3 of 8 atoms sit there (rho ~ 6e-4, width-column norm 2e-3 against ~3). An explicit bound s >= smin in the SQP would remove this.
- Wavenumber: bounded by tanh, so it has the same flatness at the bound.
- Amplitude: a vanishing atom loses its centre, width and wavenumber gradients (dead atom). This is not prevented.
- Overlap: not prevented. Two atoms with the same centre, width and wavenumber are one atom: the Jacobian loses rank and the parametrisation is not identifiable. A separation penalty or merge-and-reseed rule would be the fix. So far it has not occurred (STAGE5 32.6).

## 6.34 Optimised gamma against the number of atoms, no extra constraints (user, 2026-10-01 11:59)

User: try without constraints on the atoms first and see what happens.

**Protocol** (`tools/atom_count_runs.sh`, new; `results_atom_count/runs/`):
- K+FB+FC at ne=200, the nominal loop, N Gabor atoms per field (width >= 0.4, abs(wavenumber) <= 0.7071), `--field-input tangent` (6.33).
- The same start for every N: the H^2 ne=200 record (3.911740), its controller held and its fields fitted by N atoms (the 6.32 fits).
- Progress stop 0.05 % over 3 rounds (the nominal 0.15 % stopped atom runs early, STAGE5 32.6), at most 60 rounds, 2 runs at a time on 1 core each.
- `--resume-guard 0`: the start guard (a resumed start 10x above its record is a failed transplant) refused N = 4, whose fitted start is 81.86 with the controller held. New option, default 10 as before.

**A stability escape, found by the first set of runs** (`results_atom_count/runs_v1_escape/`).
- The N = 2 run reached a certified 3.823106, but the dense save veto refused it: an eigenvalue at +1.134 - 0.656i.
- A replay with `--dense-audit` (new: a dense eigensolve of every round's candidate, printed with the gate's decision; check only) located it at round 0. That round accepted a design with an eigenvalue at +51.94 - 20.14i, while the chain's abscissa read -0.209.
- The continuation gate watches the near-line eigenvalues. An eigenvalue arriving from far left in one step is not among them. This is the mechanism of the MIMO escape (STAGE6, `--wind-accept`), whose global winding count at acceptance was on for MIMO only.
- Fix: `--wind-accept auto` now also covers every case with a free field (FB/FC). The replay refuses that round ("winding check at acceptance: 1 eigenvalue(s) right of Re=-eps: REFUSED").
- Frozen path 10.23128516441147, witness invariant 0, Hurwitz selftest 0/200 (`tools/check_invariants.sh`).
- The lattice check's field cases (6.31) and the STAGE5 field records all passed the dense save veto, so their final designs are stable.

**Results** (rerun with the fix and `--dense-audit`: no candidate the chain accepted had alpha_dense > -0.158):

| N atoms per field | start (fields fitted, controller held) | final gamma | rounds | wall |
|---|---|---|---|---|
| 2 | 28.918332 | 4.914215 | 24 (progress stop) | 10 min |
| 4 | 81.858469 | 3.969141 | 41 (progress stop) | 26 min |
| 6 | 5.137328 | 3.892286 | 13 (progress stop) | 22 min |
| 8 | 6.196054 | 3.887178 | 15 (progress stop) | 25 min |
| references | H^2 160 modes, ne=200: 3.911740 | 4 atoms width >= 1 (A2''): **3.876296** | | |

- N = 6 is the v1 run bit for bit (3.8922862813224963): it had no escape. N = 8 differs from v1 (3.888885) at the 1e-4 level.
- 6 and 8 atoms, started from the H^2 fields, end below the H^2 optimum.
- No run reaches the 4-atom A2'' value. A2'' is admissible here: width >= 1 atoms are width >= 0.4 atoms. So the N = 4 run (3.969) ended in a worse local minimum from a worse start.
- From a common start, the final value measures the start and the path more than the capacity of N atoms. The low-N starts are poor with the controller held (81.9 at N = 4), and those runs spend most rounds recovering in small steps.

**What the atoms did without constraints.**
- **Edge atoms.** The H^2 fields carry 4 % of their energy at abs(x) > 30, the edges of the 160-mode region [-38.9, 39.2]. The greedy fit places atoms there first (N = 2: one per field, at -36.9 / +37.3), and they stay: in every run they carry about 2 % of the field energy. At N = 8 four of the eight sensor atoms sit at abs(c) > 34.
- **Dead atoms.** Amplitudes 0.004-0.008 at the ends (N = 8). Their centre, width and wavenumber gradients scale with the amplitude.
- **Overlap.** At N = 8 the actuator has two nearly identical atoms (c 1.59 / 1.95, s 4.21 / 3.99, kappa 0.237 / 0.242) with amplitudes 21.6 and 20.7 of opposite sign; they cancel to a difference shape. The start fit already had it (amplitudes 118 / 113 / 73). The sum of the atoms' energies is 7700x the field's.
- **Floors.** Widths and wavenumbers on their bounds: many edge atoms at abs(kappa) = 0.707 and s ~ 0.5.
- **Adjoint symmetry** persists: at N = 2 and 4 the sensor atoms mirror the actuator atoms (c -> -c, same widths and wavenumbers), as in STAGE5 32.6.

**Latent issue (not fixed).** Records store the atom parameters (rho, tau), not smin and kmax. A record resumed with a different `--atom-smin` / `--atom-kmax` and the same atom count is read with the new bounds, so the fields change silently. A2'' (smin 1) resumed at smin 0.4 would be wrong; the start guard catches it only if gamma rises 10x.

**Next (proposed).** Nested in N from the best design, so that a child starts at its parent:
- 4 atoms: A2'' converted to smin 0.4 (exact: rho' = sqrt((1 + rho^2)/0.4 - 1));
- 5, 6: A2'' plus new atoms at small amplitude;
- 3, 2: A2'' minus its weakest atoms.

## 6.35 Atom records carry their bounds; which atoms a mesh resolves (user, 2026-10-01 14:43)

**The record issue (fixed).** A record stored only the optimiser's atom parameters (rho, tau), meaning a width smin (1 + rho^2) and a wavenumber kmax tanh(tau). The same vector is a different field under other bounds.
- Read at width >= 0.4, the A2'' record (made at width >= 1) was a different design: a gauge-free field distance of 0.56 / 0.57.
- Nothing flagged it; the start guard fires only at 10x gamma.

The fix:
- **Writing.** Records (the final record and the checkpoint) now also store the atoms in natural form, `atoms_{a,s}_nat` = (Re a, Im a, c, width, wavenumber), and the bounds `atoms_{a,s}_bounds` = (smin, kmax).
- **Reading** (`fieldbasis.atoms_from_record`, used by the driver at both resume sites, SISO and MIMO, and by `tools/cross_eval.py`, `mimo_start.py`, `mimo_proto.py`, `check_reduced_fields.py`):
  - from the natural form when present;
  - else (legacy record) the stored (rho, tau) under whichever known bound pair, (1, 0.7071) or (0.4, 0.7071), reproduces the record's own nodal field to 1e-9;
  - then expressed in the reader's bounds: exact while the atoms satisfy them; an atom outside is moved onto the bound and the message says the field changed;
  - a record made with the reader's own bounds returns its stored vector bit for bit, so earlier resumes are unchanged;
  - an unidentifiable record, or one with another atom count, returns nothing, and the driver fits atoms to its field and says why.
- **Checked:**
  - A2'' read at width >= 0.4 starts at 3.876296, its own value (K+FB+FC, ne=200, 1 round: 3.876187); the written record carries the natural form and the bounds (0.4, 0.7071);
  - the identification is right on A2'' (1), the 6.34 runs (0.4) and the MIMO at_dec record (1, 2 ports);
  - an 8-atom record read for 4 atoms is refused;
  - `cross_eval` reproduces its earlier A2'' -> ne=400 value 3.879101;
  - `check_reduced_fields` unchanged (8.3e-13);
  - invariants pass (frozen 10.23128516441147).

**Which atoms a mesh resolves** (`tools/atom_resolution.py`, new; `results_atom_count/atom_resolution.{json,log}`).
- What the optimiser sees of a field is the plant's response to it.
- The test: the plant transfer between an actuator atom at -2 and a sensor atom at +2 with the same width s and wavenumber kappa, ne = 100 / 200 / 400 against ne = 1600, max over w in [-4, 4], relative to its peak.
- Selected values:

| s, kappa | ne=100 | ne=200 | ne=400 |
|---|---|---|---|
| 2, 0 (the plant's own discretisation error) | 5.1e-3 | 3.3e-4 | 2.1e-5 |
| 1, 0.7071 (STAGE5 bounds) | 1.1e-2 | 7.5e-4 | 4.8e-5 |
| 0.4, 0.7071 (current bounds) | 5.1e-2 | 3.7e-3 | 4.7e-4 |
| 0.4, 2 | 2.0e-1 | 1.5e-2 | 1.4e-3 |
| 0.2, 0.7071 | 5.4e-1 | 7.1e-3 | 7.3e-3 |
| 1, 4 | 8.3e+1 | 9.8e-2 | 7.3e-3 |

- At ne=100 the width-0.4 atoms are not resolved (5 %).
- At ne=200 they are resolved to 4e-3, against the plant's own 3e-4.

**Atom coherence** (normalised M-inner product of two atoms):
- the 6.34 N = 8 pair: 0.9972;
- same centre and wavenumber, width ratio 1.2 / 1.5 / 2 / 3: 0.992 / 0.961 / 0.894 / 0.775;
- same width 1, centres 0.5 / 1 / 2 apart: 0.939 / 0.779 / 0.368;
- same centre and width 1, wavenumbers 0.25 / 0.5 / 1 apart: 0.985 / 0.940 / 0.779.

## 6.36 Why did 4 atoms beat 160 Laplacian modes? Optimisation, not representation (user, 2026-10-01 16:07)

The user's question: if the modes can reconstruct the atoms faithfully, the atoms cannot do better unless the mode runs were unconverged or stuck in a local minimum. Which was it?

**The atom design is admissible in both mode problems** (ne=200, 160 modes on [-38.9, 39.2]):

| fields | 160-mode projection error | H^1 RMS wavenumber | H^2 measure R_2^(1/4) (S2's bound 1) |
|---|---|---|---|
| A2'', 4 atoms (3.876296) | 4.9e-6 / 3.9e-6 | 0.49 | 0.60 / 0.59 |
| S2, H^2 (3.911740) | 1e-15 | 0.63 | 1.000 (on its bound) |
| R1, unconstrained (3.888738) | 2e-15 | 1.82 / 1.88 | 3.18 / 3.30 |

So R1 (3.8887) and S2 (3.9117) both ended above a point of their own admissible set. Representation is excluded.

**Same start, same mesh, same value.**
- A2 (atoms) and S2 (H^2) both started from the frozen ne=200 design (`tools/fields_mesh.sh`). They ended at 3.912595 (progress stop, 15 rounds) and 3.911740 (round cap, 20 rounds), both still descending. STAGE5 32.6 noted that A2 "equalled H^2".
- The atoms' 3.876 came from another start and about twice the effort: A3 at ne=400 from the frozen design carried to ne=400 (3.9039, 11 rounds), carried to ne=200, then A2' (12 rounds) and A2'' (21 rounds).
- STAGE5 32.6 / 32.8's "atoms beat H^2 with 1/16 of the parameters" compared unequal starts and effort. **Withdrawn**: the atoms gave no better optimum. Their advantages are smoothness by construction, mesh independence and fewer parameters, with a lower cost per round. The STAGE5 conclusions about mesh independence stand.

**The decisive test** (`tools/modes_vs_atoms.sh`, new; `results_modes_vs_atoms/`): S2's problem (ne=200, 160 modes, H^2 row kmax = 1, the STAGE5 loop with progress stop 0.05 %), stopped by hand after 5 / 9 rounds:

| run | start | after rounds 1 .. | stopped at |
|---|---|---|---|
| from_atoms (A2'' design, fields projected onto the modes) | 3.876296 | 3.875400, 3.871456, 3.868796, (refused), **3.868466** | 5 rounds, still -0.01 .. -0.1 % a round |
| s2_cont (S2 continued) | 3.911740 | 3.911534, 3.911534, 3.911482, 3.911218, 3.910895, 3.910778, 3.910778, 3.910760, 3.910707 | 9 rounds, -0.003 % a round on average |

- **The modes do better than the atoms when started well.** From the atom design the mode problem goes below it at once: 3.868466 after 5 rounds, 0.20 % below 3.876296.
  - The design is stable (alpha_dense -0.219).
  - Its fields moved 0.10 from the atom fields.
  - Their H^2 measure rose from 0.60 to 0.98 / 0.997, onto the bound the atoms never reached.
  - The modes use the extra freedom: finer content inside the H^2 ball, which 4 atoms of width >= 1 cannot carry.
- **S2's end is a stall in a flat valley.**
  - 9 more rounds gained 0.026 %.
  - At that rate the 1.1 % to the new value would take hundreds of rounds.
  - Formally unconverged (it still descends), practically a different basin reached from the frozen start. A local minimum cannot be excluded; nothing here shows a barrier.
- So both: the mode runs were **not converged**, and they crawled in a **different, flat region** reached from the frozen start. The modes' representation was never the limit.

The runs used `--nproc` default (2 pool workers each, mostly idle); the rounds took 5-8 min with kd ~300 (two 160-mode input bases). The checkpoints are kept.

## 6.23 Open items

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
- orders 10 and 15 of the central K+B+C compensator (user, 2026-09-30):
  `tools/mimo_n100.sh kbc_cen10` / `kbc_cen15` (weighted-truncation starts, 6.27);
- field cases at ne=200: reduced basis ~300 with two 160-mode input bases (6.31) --
  `--field-input tangent` (6.33): 12 + 12 columns, kd ~120, 4.2x faster, same path;
- number of atoms (6.32): optimised K+FB+FC gamma against N at ne=200, width >= 0.4
  (with `--field-input tangent`: ~15-25 min per run);
- atom well-posedness (6.33, 6.34): explicit width/wavenumber bounds instead of the flat
  reparametrisation at the floor; dead, edge and overlapping atoms (all seen in 6.34);
- atom count: nested study from A2'' (6.34);
- field parametrisation (6.36): the modes + H^2 beat the atoms when started from them;
  the choice is between parametrisations on cost and conditioning, not optimum;
- before 2-D scale: the five items of 6.28 are done (6.29), the parallel
  full-model tasks too (6.30); left: MUMPS as a `sparse_lu` backend, the
  singular-M (velocity mass) versions of the M-norm bounds at the BFS.
- K+B+C centralized: still -0.25 %/round when stopped (6.20); its start was the
  c4cf908 decentralized record (5.474), not the better 11c4073 one (5.339) -- both
  (the tail and the start) on the cluster; `--wall` as a hard stop would bound such
  runs.
