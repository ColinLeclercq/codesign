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

Not supported, by decision: a MIMO design with no compensator states (r = 0).
The static-gain MIMO starts keep the SISO design's compensator states with
C_c = 0 (passive: unobservable from u, its poles are closed-loop poles at
Re = -0.2188).

## 6.2 Validation

- **SISO bit-identity** (the SISO path must not move): invariants in the MIMO tree
  (`tools/check_invariants.sh`: frozen path 10.23128516441147, witness invariant 0
  violations, Hurwitz selftest 0/200) PASS; `tools/check_siso_identity.sh` (new):
  the same 5-round K+B+C ne=100 run in this tree and in the `siso-v1` tree,
  8.754365110002194 in both. Re-run after each change of this stage (6.4: PASS;
  final code: 6.9).
- **`tools/check_mimo.py REC`** (new; writes REC_check.json), every check against
  an independent dense reference (explicit Cholesky of M -- validation only):
  1. sigma_1 of `ClosedLoopResolventMIMO` vs sigma_max(L^H [S_cl^-1]_11 L) at the
     record's active frequencies and a grid: 1e-12..1e-13;
  2. the MIMO class at m = p = 1 vs the SISO class on the SISO K+B+C record:
     sigma 1.1e-14, controller gradients 3.8e-14;
  3. finite differences of sigma_1 at the active frequency of largest sigma, along
     random directions in D_c, C_c, B_c, the A_c diagonal and each load/sensor
     vector (port sides): 7.5e-5..1.7e-4 relative (the largest on entries whose
     derivative is 1e-5 of the others);
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

## 6.7 Results at ne=100 (2 actuators, 2 sensors)

(filled in below as the runs finish)

## 6.9 Open items

- batched MIMO reduced gains (speed; rows are evaluated one by one);
- the high-gain branch sits on the margin at |Im| up to ~850 while the static gain
  grows: the regime STAGE5 34.1 described for SISO (large, weakly determined
  gain) with two loops; a control-cost weight (H-infinity norm of W K S) is kept
  for later (user, 2026-09-28);
- `band_from_closedloop` and the dump spectrum are dense (fine at ne <= 200;
  replace before the cavity -- already on the list);
- the winding count needs the plant pole data (dense below n = 4 x dense_nmax);
  at cavity scale P_plant comes from algebra item 3.
