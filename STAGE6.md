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

## 6.9 Results at ne=100 (2 actuators, 2 sensors)

All on commit 0e7f1c1 (the code of 6.3-6.8), `tools/mimo_n100.sh <case>`, one
process each, two at a time on the 2 cores. Every record passes
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

## 6.12 Open items

- the relative stationarity test with one very sensitive active row (6.10): which
  coordinates carry the gradient, and a scale-safe test (e.g. per-block scales);
- multimodality: decentralized atoms from the SISO optimum stall at 2.964887
  while the structure reaches the floor from another start; K+B+C decentralized
  depends on pair 2's start (STAGE5 34.3) -> multistart on the cluster;
- batched MIMO reduced gains (speed; rows are evaluated one by one);
- the static gains grow into the high-gain regime (|D_c,11| up to 4.2e4); a
  control-cost weight (H-infinity norm of W K S) is kept for later (user,
  2026-09-28);
- `band_from_closedloop`, the dump spectrum and the dense all-w save check are
  dense (fine at ne <= 200; at the cavity the tail certificate, algebra item 1,
  has to carry the all-w claim);
- the winding count needs the plant pole data (dense below n = 4 x dense_nmax);
  at cavity scale P_plant comes from algebra item 3;
- ne >= 400 MIMO runs: on the cluster (user, 2026-09-28).
