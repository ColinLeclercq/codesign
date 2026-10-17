# Literature report: certified fixed-order H∞ / port-position co-design on the BFS (Re = 600)

Prepared 2026-10-16. Sources:
- the 22 distinct documents in the user's archive (23 PDFs, because one is a duplicate), read section by section from the pdftotext files, with pages checked on images where the equations were garbled;
- web searches, which returned bibliographic hits only: no journal or arXiv page could be opened, but four software sources were fetched;
- the project notes, the code at HEAD, the 20261010n batch logs and the stationarity smoke runs.
- a literature refresh with network access on 2026-10-17 (arXiv, HAL record pages, package metadata); its corrections and tag changes are applied below, and its addendum lists what was read with sections and pages. Texts fetched and read are [F]; abstracts or record pages only are [A].

Nothing in /home/user/codesign was edited.

Evidence tags:
- **[R]** read in the archive, with section or page;
- **[F]** fetched and read outside the archive;
- **[A]** abstract or record page fetched and read, full text not read (added 2026-10-17);
- **[B]** reference found in the bibliography of a paper read here (archive or fetched), content not read;
- **[S]** search snippet only, content not read;
- **[C]** checked in this project's code, logs or notes;
- **[I]** my inference.

Scratchpad files are cited as [E1]-[E12]; the full paths are listed at the end.

> Paths: `results:20261010n/...` is the folder 20261010n/ of the branch `results` (the batch logs and records). Paths marked (session scratch) were working files of the 2026-10-16 session and are not kept; their conclusions are in this report and in LESSONS.md. The papers are the user's BIBLIO archive.

---

## 1. Verdict

**Is there an obviously better way? No, among everything read or found.** The tool belongs to the Apkarian-Noll family of nonsmooth, frequency-domain structured H∞ methods: grow a finite frequency set by a worst-case search, solve the finite max with a local nonsmooth method, certify afterwards (AN 2006 TAC §VI [R]; Apkarian-Dao-Noll TAC 2015 Alg. 1 [R]; Apkarian-Noll IJRNC 2018 Alg. 3 [R]).
- Our inner QP is the Variant II subproblem of AN 2006 (the optimality function θ, eqs. 19-20), with a BFGS Lagrangian Hessian, a trust region and a merit function added.
- The 2x2 Clarke SOC block is their Y ∈ B_2 (eq. 13).

Each alternative family fails on a property this problem needs:
- **A dense level set or Hamiltonian per evaluation** (BBBS, Bruinsma-Steinbuch, HIFOO, hinfstruct/systune) costs O(n³). The Apkarian-Noll survey calls it costly above about 500 states (1905.11511 [R]).
- **Large-scale norm estimators** (GGO 2013, Benner-Voigt 2014, ABMSV 2017 [R]; HEC [B]) give local lower bounds only. GGO is wrong exactly on HIFOO-optimised loops with many near-equal peaks (Table 5.1, CM3/CM4).
- **Subspace or ROM minimisation** (ABMV 2019, Mengi 2023 [R]) assumes no stability constraint, n ≫ m, p, few affine parameters and a smooth optimum. None of these holds here. BMO 2018 [F] is different: a ROM objective, but a stability constraint α(A_f) < 0 on the full-order closed loop (arXiv:1803.06549 §3 eqs. 5a-5c), computed by eigs and uncertified (§4 p. 6), a nonsmooth GRANSO solve and order-10 controllers with a full-state performance channel (Table 1); without that constraint 5 of 12 designs left the full-order loop unstable (Table 2). It is the nearest published case of the full model in the loop for stability. Our own ROM failed on measured position-dependence errors of 3.6e-3 / 3.9e-2 / 22 % for port moves of 0.1 / 0.3 / 0.8 (LESSONS §7 [C]).
- **First-order nonsmooth solvers.** Bundle methods (Kiwiel 1985, Lukšan-Vlček 1998 [R]) need about 1-2 evaluations per iteration (L-V Table 3 p. 390) but many iterations (null steps, first-order or approximate curvature). Basic gradient sampling needs at least n + 1 gradients per iteration, adaptive variants a constant number (gradient-sampling survey Alg. GS p. 4, §4.3 p. 9 [F]). GRANSO [R] needs 1 plus its line-search trials (5-10 for pyGRANSO, LESSONS l.117). Our SQP needs about 2. The project's pyGRANSO A/B is GL-only and 0.13 % apart (7.950855 against 7.961104), inside the 0.7 % path spread. The case against switching therefore rests on structure (explicit pieces, Clarke block, rows added mid-solve) and iteration count, not on that A/B; cost per iteration decides only against gradient sampling.

**Where we stand.**
- **Ahead in scope.**
  - The full FE descriptor model sits in the inner loop, with moving ports and a full-field performance channel.
  - Positions are continuous variables, and the same tool handles SISO, decentral and central.
  - γ is certified on the full test-plant model.
  - No paper read in full co-designs continuous ports and a fixed-order H∞ compensator with a certified γ on a full flow model (2026-10-17 refresh [F]: Jussiau 2024, Nibourel 2023, Werner-Overton-Peherstorfer 2023, Benner-Heiland-Werner 2022, Jin 2022, Martini 2022, Manohar 2022, Gumussoy-Michiels 2011). The nearest are WOP 2023 (fixed-order H∞, n_K = 2, by gradient sampling on full descriptor models up to 22,060 states, fixed ports, a norm without a global certificate; Table 1 p. 13, p. 15) and Nibourel 2023 (structured mixed H2/H∞ by systune on ERA ROMs of order 115-130, sensor chosen by a 1-D sweep; §5, App. A). The novelty is the combination, not the size (33,593 states is 1.5x WOP's largest model). It remains a negative claim: Leclercq 2019, Jussiau 2022 (L-CSS), Chen-Rowley 2011 and Kasinathan-Morris 2013 could not be read.
- **On par in method, ahead in deployment, for the in-band γ certificate.**
  - It is the Schwerdtner-Mengi-Voigt decision procedure (SMV Alg. 2 [R]), run in parallel and inside the solve.
  - At 2n ≈ 67k it is 1.6x SMV's largest test (q = 42,264). That is a modest step in size, not a new class.
  - Its rigour carries SMV's proviso (Arnoldi returns the nearest eigenvalues). It also has one weakness SMV avoid: crossings are classified with an unstructured tolerance (§2, row 5). SMV's exactness holds for simple eigenvalues of a real pencil (the BFS, not GL), and an off-axis pair near the axis can still come out imaginary, so σ sampling stays necessary (SMV p. 5 [R]).
- **The all-frequency stability count is currently failing.** In 20261010n the save veto rejected the final design of all five decentral arms: "winding_count: tail bound never certifies" [E1]. Each record fell back to its job start, so the certified decentral values are still **369.87** and **412.28**; the vetoed finals were 366.49-367.40 and 388.44-390.82. A new tail bound is in progress. It gives W_T = 7,680 against 7.86e6 for the old bound on 369.87 [E2].
- **Behind in measuring and stopping.**
  - Production stops on ‖∇L‖ ("rel", SISO) or the QP's predicted decrease ("pred", MIMO), plus a 3-round progress rule. The project's own rule says MIMO needs 8 rounds (LESSONS l.132-134), but bfs/run_bfs.sh:131 passes 3:0.0015 for every case [C].
  - Preliminary readings, not certified: SISO 598.11 is not stationary. A step of ‖d‖∞ = 1e-6 lowers the max over its relocated tops by 0.10 %, with model ratio 1.00 [E3, E6].
  - Decentral 369.87 read "near ε-stationary" [E5], yet the next batch went 0.7-0.9 % lower [E1].
- **Behind in coordinates (new finding).**
  - Each modal 2x2 block keeps a 2-dimensional similarity gauge. The iterates drifted to ill-conditioned representatives: block cond(V) is 21.6 for SISO 598.11 block (2,3) (rechecked here), and up to 509 for SISO 554.97 as measured by a reviewer.
  - Gradient magnitudes span about 200x across variable groups.
  - On all three records, the first-order model along the H = 0 box direction holds only up to ‖d‖∞ ≈ 1e-5 [E6].
  - This is the cheapest untried lever on the code-4 crawl.
- **Behind in lower bounds and validation.**
  - Nothing yet says how far a BFS design is from the optimum. The port-independent open-loop floor is weak: γ ≥ 32.66 for SISO and ≥ 25.00 for 2x2, at the open-loop peak [E7].
  - The per-port (Parrott) bounds that tools/bounds.py computes for GL have not been computed on the BFS.
  - There is no dense end-to-end BFS audit, no full-mesh run, no time-domain check and no domain-sensitivity check. BFS.md l.965 notes that gains differ by orders of magnitude between the truncated domain and the paper's.

**The one plausible algorithmic gain** is a safeguarded second-order tail on the identified manifold (Bompart-Noll-Apkarian 2007 [B], still not read: no open copy on 2026-10-17). Every fast-rate theorem read since needs an isolated, nondegenerate minimiser and a fixed identified piece set (Bareilles-Iutzeler-Malick 2023 Thm 4.5 and Assumption 4.2 p. 18; Lewis-Wylie 2019 Thm 5.11 p. 18 [F]). A similarity gauge rules out isolation, so a gauge-free compensator form is a prerequisite of this tail, not only an earlier step. It comes after the gauge fix and after measuring where the linear model breaks down. At c151 it breaks down first at the σ1/σ2 block near ω = 0.926, so multiplicity curvature matters as much as U-space curvature [E6].

**Suggested order** (§5; nothing to implement now):
1. Fix the tail bound and re-count the vetoed designs.
2. Use 8 rounds for MIMO.
3. A/B a gauge rebalance.
4. Run the stationarity tool as a diagnostic (it is not a licence to stop).
5. Compute per-port bounds.
6. Audit the certificates and validate (α-sweep, dense mini-mesh, full-mesh open-loop sweep, time domain).
7. Measure, then apply, the parallel levers.
8. Only then make second-order changes, through clean A/Bs.

**Decisions that remain the user's:**
- the band ([0, 10] for SISO and decentral);
- no control-effort weight (W K S deferred; LESSONS l.604);
- whether −ε on the compensator is a requirement;
- whether to align with the H2/LQG measure of the physical baseline (Barbagallo et al. 2012 [A]; LQG per Nibourel 2023 p. 3-4 [F]; its channel is upstream noise to a wall-shear window, BFS.md l.84, l.91-95 [C]).

---

## 2. Component-by-component comparison

Gap ratings: ahead, on par, behind, different trade-off, unknown.

| # | Component | Ours | Literature | Gap | Evidence |
|---|---|---|---|---|---|
| 1 | Architecture | Outer rounds: SQP on the epigraph over Ω, then the round-end γ certificate, enrolment of new peaks, the continuation gate, and a winding count at save. In-solve level set, asynchronous as an option. | AN 2006 Variant II: step 1 picks a new Ω_k ⊇ Ω(x_k) at each iterate; step 4 refines at a fixed iterate after a rejected step (Lemma 7) [R]. TAC 2015 Alg. 1: Δ_a grows, a-posteriori μ certificate [R]. IJRNC 2018 Alg. 3: Ω_opt fixed per solve, Lipschitz verification grid, restart [R]. | on par | Never retiring Ω rests on our own A/Bs (LESSONS §2: the cap of 50 and retirement lost every A/B), not on AN's theory, which rebuilds Ω at every iterate (§VI-G, capped at 50). What their theory does need is that Ω_k contain every active frequency at x_k; job 770632's missed peak violated exactly that. |
| 2 | Inner solver | Clarabel conic QP (σ rows, 2x2 SOC, elastic stability rows, modal rows, positions); damped BFGS fed by cone duals and reset every round; trust region; merit; about 2 evaluations per iteration. | AN 2006 first-order θ-QP with Armijo, "may be slow in the neighborhood of a local solution" (p. 82) [R]. HANSO/GRANSO, BFGS on the max itself, R-linear on model problems (Lewis-Overton 2013) [R]. Kiwiel and L-V bundle methods [R]. Bompart et al. 2007 [B]. | on par; unknown against Bompart | pyGRANSO A/B: GL, 0.13 % apart, inside the 0.7 % spread (LESSONS §1, §14). Direct gradient descent on a nonsmooth L∞ objective took 37 iterations and 624 evaluations (HinfMOR §2 p. 6, iterates in Table 2.1 [R]); for BFGS the paper states only a linear rate at best, with no count. Resetting BFGS each round won a clean A/B (LESSONS §1). |
| 3 | Multiplicity / Clarke set | Full 2x2 Clarke block (SOC) where σ1 ≈ σ2 (gap 0.15); no k ≥ 3. | AN 2006 eq. 13, TAC 2015 Prop. 1: Y PSD of the multiplicity's size over every active frequency, an SDP [R]. Noll-Apkarian 2005: identify the multiplicity, then Newton [B]; the identification is a heuristic and the fast rate is not guaranteed (Bareilles et al. 2023 p. 3 [F]). | on par; now live on central | 20261010n: the c151 arms enrol σ2 at 8-14 of about 180 frequencies, c177 at 1-2, decentral at 0 [E1]. Smoke c151: a 2x2 piece at ω = 0.926 with gap 0.0134 [E4]. The descent probe fails first at that block (top 1.00137 γ at ‖d‖∞ = 1e-4) [E6]. No SDP for k ≥ 3 (LESSONS §11). |
| 4 | Peak location | Ω accumulated and never retired; followed members by safeguarded Newton on the exact slope (stop when the slope magnitude ≤ 1e-7 σ); in-solve level set at max(t, smax)(1+tol). | AN 2006 §VI-G: Hamiltonian peaks plus log-spaced points, cap 50, rebuilt each iterate [R]. Benner-Mitchell 2018 Alg. 2: Newton on the promising intervals, then one global check [R]. TAC 2015 §III: a worst-case search is a min-min (upper-C1) program and can be run like a smooth problem on any active branch [R]. | ahead in exactness | Tops match the serial Newton to 2e-14 (BFS.md §3). A cap of 50 is too small on our plateaus (164 of 214 members within 5 %). Our ω-search at a fixed design is the TAC upper-C1 case, and Newton is a second-order instance of it, which TAC does not prescribe. The design-level min-max is lower-C1, the class they say needs "a tailored solution" (p. 1859), here our SQP plus exchange. |
| 5 | γ certificate in the band | Round end: SMV Alg. 2 (certify_greater) on a factored 2n pencil, 334 s serial. In-solve: the same decision through plant factors, CG in the M semi-inner product, an M-free norm disk, residual-checked disks, an attained witness, rtol 1e-6. | SMV 2020: serial, q ≤ 42,264. The symmetry-preserving K(ξ) (eq. 5) returns axis eigenvalues with exactly zero real part. Intervals shorter than ε_eigs are removed (§3) [R]. Benner-Lowe-Voigt, NACO 8(1):119-133, 2018 [S] (not cited by SMV). Dense BBBS/BS [R]. | on par in method, ahead in deployment, behind in structure | Benner-Mitchell 2018 §7: no scalable global method existed in 2018 [R]. The round certificate (certify_greater) tests crossings by "abs(Re λ) < 1e-8·max(1, max_j abs λ_j)" over the returned set (/home/user/codesign/certify_sparse.py l.348-350). Its disk radius min_j abs(λ_j − iθ) is never smaller than abs(Re λ) (l.357), and intervals shorter than ε_eigs are dropped without a σ evaluation (l.361-364, l.370), so a true crossing computed off the axis is cleared or left in a dropped sliver. The in-solve level set differs: a per-eigenvalue test "≤ 1e-8·max(1, abs λ)" (lattice.py l.115, l.2050), disks shrunk to 0.9 r (l.92), so Im λ falls inside a cleared interval only if abs(Re λ) > 0.48·abs(Im λ − θ), σ1 at every centre, and slivers given a finest-lattice centre instead of being dropped (l.2343-2350). SMV justify their sliver removal empirically ("only occur due to errors in the eigenvalue computation", p. 5-6); that their exact classification makes it safe is our inference [I]; for certify_greater ours is not safe. The contour count promised at l.36-45 is not implemented (l.218 is a header only), and the Hutchinson count it describes would be a stochastic estimate, not a certificate (Di Napoli-Polizzi-Saad 2014 Lemma 2.1 p. 4, §6 [F]). |
| 6 | Band coverage of γ | [0, 10] for SISO and decentral; central widens to the compensator poles ([0, 24.92] at c151) [E8]. | IJRNC 2018: γ on a finite verification grid with L from finite differences (Lemma 3, Thm 3, Remark 12), so neither rigorous nor up to ∞; its decay bound (Remark 6) serves the Nyquist count [R]. Bruinsma-Steinbuch includes σmax(D), i.e. ω → ∞ [R]. | behind for SISO and decentral (a user decision) | LESSONS §4-5. On GL, out-of-band peaks gave false bests (STAGE6 §6.6, 6.21). |
| 7 | Stability inside the optimisation | Elastic per-eigenvalue rows from the line cover (plus the strip in strip runs), a line certificate at every evaluation, the continuation gate, merit-weighted crossings. | AN 2006/2007: shifted-norm barrier or distance-to-instability homotopy (Lipschitz) [R]. IJRNC 2018: modulus-margin barrier plus a Nyquist test at every primary step; Remark 9 prefers abscissa constraints when A(K) is available [R]. HIFOO: an abscissa phase [R]. | different trade-off | AN 2007 Table 1 shows that Model I (the homotopy) solved directly at the target fails, so a homotopy is required. It is not a barrier result. Our rows are not Lipschitz where eigenvalues coalesce (LESSONS §5, open). |
| 8 | All-frequency stability certificate | Argument-principle count of det(I − KG), M-free; plant poles by an inertia-certified enclosure; tail bound min(H^1, L2) since commit 9bfdee0 (bfs/plant.py l.587-625). Run at the start, at save (with veto) and as the gate fallback. | IJRNC 2018 Thm 1-2 and Lemma 2: a rigorous sampled Nyquist test when L bounds the derivative of f per segment, run at every primary step [R]. Kamgnia-Philippe 2013 [F] (read as INRIA RR-7770 v2): the rigorous segment test is Condition (B), eq. (9) p. 7, but their own step control, (C) plus (B'), is "a heuristic procedure" (§4.3), and Ex. 3.2-3.3 show both tests failing. Ying-Katz 1988, Johnson-Tucker 2009 [B] (described in Batra-Sharma 2019 p. 10-11 [F]). | ahead in scope; behind in rigour; currently failing on decentral | 20261010n: 5 of 5 decentral finals vetoed, "tail bound never certifies" (e.g. 774361 no strip l.269-276) [E1]. The veto counts at most 3 designs, so every arm fell back to its job start. /home/user/codesign/winding_cert.py l.48-55: the checks "validate but do not prove"; the segment bound is "NOT IMPLEMENTED"; the march (l.277-334: step 0.35·abs(f)/abs(f'), chord at most 0.75·min abs(f), phase increments at most π/2) is of Kamgnia-Philippe's (C)+(B') class. `--wind-accept off` overrides the MIMO auto-on rule (LESSONS l.313-314). |
| 9 | Compensator parametrisation | Real modal 2x2 blocks; exact Routh-Hurwitz rows (tr, det); second-order correction back onto the line. | HIFOO: unstructured [R]. HinfMOR: tridiagonal A_red and diagonal E_red, justified by a Kronecker form (p. 5); its abscissa row assumes a simple rightmost eigenvalue (p. 16) and was never imposed (p. 19) [R]. IJRNC: conical damping on the controller modes [R]. Tridiagonal controllers (after McKelvey-Helmersson 1996 [B]) in TAC 2015 (85 parameters, §VI p. 1866 [R]), Apkarian-Noll 2022 (54, §5.2 p. 14 [F]) and Nibourel 2023 (§5.2 p. 20 [F]) keep a diagonal-scaling gauge [I]; their controller-stability constraints are abscissa-type (IJRNC §6; Apkarian-Noll 2022 §5.1 p. 13, radius and damping bounds [F]). The matrix abscissa is Clarke-regular only at nonderogatory active eigenvalues, the coefficient-space abscissa everywhere (Burke-Eaton 2016 §1 [F]). | rows ahead; coordinates behind | The rows are smooth, with FD agreement to 7.6e-11 (LESSONS §6). "5 of 6 arms improved" from 20261010d to 20261010e is confounded (several changes at once), and central no strip regressed (387.06 against 364.64). The gauge fix fixes one B_c entry per row (/home/user/codesign/packer.py l.349-357), which removes diagonal scaling only and leaves 2 dimensions per block: 10 of the 60 central variables (10 of SISO's 34) are gauge. Block cond(V) per a reviewer: SISO 554.97 up to 509; 598.11 13.8 / 21.6 / 9.2; central 123.9 34; decentral 412.28 61. det's gradient vanishes at M = L·I, a constraint-qualification failure that only the second-order correction handles. |
| 10 | Constraint handling | Elastic S-ℓ1-QP; Byrd-Nocedal-Waltz steering of ν; second-order correction on the compensator line. | GRANSO Procedure 1 (Curtis-Mitchell-Overton 2017) [R]. systune Φ_η = max{f, ηg} with bisection (ECC 2014) [R]. Kiwiel's improvement function [R]. | on par | The fixed-ν = 100 failure (jobs 760989, 761011, 761012) is GRANSO's SFPP failure mode (CMO 2017 §4.2). |
| 11 | Stationarity and stopping | Code 0 under "rel" (SISO) or "pred" (MIMO); progress rule 3:0.0015 for every case; deadline. A tool is in progress. | AN 2006 θ (continuous, zero iff Clarke-critical), stop (23)&(24) or (25) [R]. Kiwiel's w and the constrained Goldstein set M(x; ε) (Ch. 6 §7) [R]. L-V w_k [R]. CMO (12)-(13) [R]. Lewis-Overton §6.3 hull test and HIFOO dnorm/evaldist: heuristics [R]. TAC 2015 stops on relative step and decrease (Remark 9), with no criticality test [R]. | behind | SISO 598.11 stopped at stat 6.1 while 0.10 % was available at ‖d‖∞ = 1e-6 [E3, E6]. c151 async stopped at 0.124 % over 3 rounds [E1]. LESSONS l.132-136: use 8 rounds for MIMO; 20261010e central stopped at 387.06 and was later 61 % lower. |
| 12 | Factor reuse across designs | Design-independent plant factors on a dyadic lattice; ports and controller enter by Woodbury. | IJRNC precomputes small p x m samples offline, impossible with moving ports and a full-field channel [R]. ROM methods amortise by projection [R]. | ahead | BFS.md §3. Off-lattice exact-Newton points still need fresh factors: c151 no strip did 162,851 factorisations (79,104 s), 156,225 of them at exact points with 791 side-cache hits [E1]. |
| 13 | Sparse kernels | Sequential MUMPS per rank via petsc4py: a new KSP and analysis for each factor, solves looped over columns; σ1 by ARPACK as chains of single-vector solves. | MUMPS 5.3.5 guide [F]: NRHS solves, JOB=2 reusing an analysis, ICNTL(7)=1/PERM_IN, ICNTL(16), BLR. SHIRA, EVEN_IRA [B]: reliability levers, not speed levers (two shifted solves per application and J-orthogonalisation, Appeltans-Michiels 2021 §2.2 [F]; structured solves only as a fallback in Mitchell 2021 §2.3 [F]). RSVD-LU [F]. | behind, modest payoff | /home/user/codesign/bfs/mumps_lu.py l.82-117. GL's 4x came from the blocked _orth; its multi-RHS solve was bitwise the column solves (STAGE6 l.1128-1139). BFS multi-column solves carry 1-2 columns. The analysis share of the 0.54 s factorisation was never measured. |
| 14 | Parallel scheduling | MPI farm, synchronous waves, LPT opt-in, asynchronous level set, two 48-rank arms per node. | Benner-Mitchell §8: intervals and starts are embarrassingly parallel [R]. FEAST/PFEAST, PEXSI, Gobbato et al. 2011, Mounié-Rapine-Trystram 2007, Dean-Barroso 2013, Berljafa-Güttel 2017 [S]. SLEPc CISS partitions [F]. | behind | 48-rank 20261010n: evaluation waves 32-61 % busy, synchronous level waves 11-27 %, background level set 27 % (c151 async). Async joins waited 437-2,335 s per solve of 3,260-5,035 s on central [E1]. Slurm's 99 % is busy-polling (LESSONS l.973). |
| 15 | Model fidelity | Full FE test plant in every evaluation; the full mesh has never been run with the lattice; domain [-1.5, 30] x [0, 3.5]. | ABMV 2019 [R]; BMO 2018 [F] (ROM objective, full-order stability constraint); Werner-Overton-Peherstorfer 2023 [F] (multi-fidelity restarts over n = 6,618 / 10,645 / 22,060; RMF-GS reached the lowest high-fidelity value, 1.137206 in 82.4 h, Table 6 p. 20, but its high-fidelity value can rise while a coarse level is optimised, p. 18); Kasinathan-Morris 2013 [B] (optimal locations computed on approximations converge, per Edalatzadeh-Morris 2018 p. 1 [F]); Morris 1994 unverified. | different trade-off; sensitivity unchecked | LESSONS §7, §8, §14. BFS.md l.965: the paper's domain [-20, 100] x [0, 20] gives gains orders of magnitude different. |
| 16 | Lower bounds / distance to the optimum | None on the BFS. | AN 2006 full-order column (Table II) [R]. HIFOO full-order hinfsyn (AC2) [R]. HinfMOR Hankel bound [R]. Parrott / Davis-Kahan-Weinberger [S]. | behind | Open-loop σ = [1464.36, 32.66, 25.00] at ω = 0.7318 [E7], so γ ≥ 32.66 for any SISO design and ≥ 25.00 for any 2x2 design. Per-port bounds exist on GL only (tools/bounds.py; LESSONS l.622). |
| 17 | Validation practice | FD gradients per structure (6e-11 to 1e-8); check_lattice against serial; SISO = MIMO at m = p = 1 (1.1e-14); GL frozen identity; mfree on 1,279 dofs. | SMV: 264 decisions over α-scaled levels [R]. Benner-Mitchell: digits against hinfnorm(1e-14) [R]. HIFOO: order continuation, restarts [R]. IJRNC: cross-check with systune [R]. | behind | No BFS dense end-to-end reference, no full mesh, no time-domain check, no order ladder (LESSONS §11, §14, §17 item 31). |
| 18 | Position co-design | Continuous positions as SQP variables, exact port jets, joint with the fixed-order compensator, certified γ. | Jin et al. 2022 [F]: H2, full-order Riccati controllers on terminal-reduced channels (3 resolvent modes up to frequency 9), brute-force maps plus gradient search, N = 1.2e5 (§2.2 p. 4, §3.3 p. 8-9, §4). Nibourel 2023 [F]: sensor by a 1-D sweep on ROMs (App. A). Martini 2022 [F]: causal H2 kernels with placement scans (§4.1.1). Manohar et al. 2022 [F]: an open-loop QR-pivoting heuristic on balancing modes (§IV, §VI). Kasinathan-Morris 2013 [B]: H∞ actuator location, existence and approximation results (Edalatzadeh-Morris p. 1 [F]); "state feedback, derivative-free" only from a thesis-abstract snippet [S]. Chen-Rowley 2011 (H2) [B]. | ahead | Supported by the full texts listed; Chen-Rowley and Kasinathan-Morris unread. |
| 19 | Physical baseline | Open loop 1464 → central ≈ 124, decentral 369.87 certified, SISO 554.97 / 598.11 (test plant). | Barbagallo et al. JFM 2012 [A]: LQG on a ROM (Nibourel 2023 p. 3-4 [F]), same step and devices; the "inherent amplification" reduced by about an order of magnitude in a metric the abstract does not name; some LQG controllers were unstable on the full linearised equations. Juillet et al. 2013 [A]: feed-forward on a BFS at Re = 350, an order of magnitude on an averaged perturbation norm. | unknown; not comparable | Same step and devices (BFS.md header), but the paper's channel is upstream noise B1 to the wall-shear window Cp = [11, 11.6] on [-20, 100] x [0, 20] (BFS.md l.84, l.91-95), against our full-field channel on a truncated domain where gains differ by orders of magnitude (BFS.md l.1014). Our ratios (2.6x SISO, 4.0x decentral, 11.8x central) cannot be set against it; compare only by evaluating both designs on one plant and one channel. |
| 20 | Formulation | Forcing over the full velocity field and full-state output (B Bᵀ = Cᵀ C = Q); no control-effort output. | Benner-Heiland-Werner 2022 [F]: Riccati H∞ for incompressible flow in the normalised form B1 = B2 = B, C1 = C2 = C (§2.2 eqs. 9-10), 2 controls and 3-4 averaged outputs at n_v = 41,718 / 45,528 (Tables 1-2 p. 13), controllers reduced to orders 12-48. Jin et al. 2022 [F] make a full-field channel tractable for H2 by keeping 3 resolvent modes (§3.3). | user decision | With a full-rank Q and no effort channel, low-rank Riccati/ADI and the regular full-order H∞ problem do not apply [I]. GL drifted to high gain without a control-cost weight (abs K 187 → 4468, LESSONS l.395). |

---

## 3. Measuring Clarke stationarity for our constrained max-type problem

### 3.1 What is measured

- **Problem.** f(z) = max over ω ∈ [0, ω_hi] of σ1(T(z, iω)), subject to c_j(z) ≤ 0. The constraints are the closed-loop Re λ_j + ε for eigenvalues near the line, the modal tr/det rows and the position rows.
- **Regularity.** f is a max of smooth pieces over a compact set. It is therefore locally Lipschitz and Clarke regular (Kiwiel 1985 Lemma 1.2.5 [R]), and lower-C1 in TAC 2015's terms (§III [R]).
- **Clarke set.** It is the hull, over the active tops ω_i, of Φ_{Y_i}: each Y_i is Hermitian PSD of the size of σ1's multiplicity at ω_i, and Σ tr Y_i = 1 (AN 2006 eq. 13; TAC 2015 Prop. 1 [R]).
- **Eigenvalue rows.** They have a valid subdifferential only where the active eigenvalues are semi-simple (TAC 2015 Prop. 3 [R]). Where they coalesce, flag the row instead of measuring it.
- **First-order condition.** The F. John form 0 ∈ μ0 ∂f + Σ μ_j ∂c_j becomes KKT when a constraint qualification holds (Kiwiel Lemma 1.2.15 and (1.2.66), p. 16-17 [R]).

### 3.2 Two prerequisites

1. **Fix the coordinates first.**
   - Whether 0 lies in the hull does not depend on the coordinates; the size of ρ and pred_inf(Δ) does.
   - Every piece's gradient is orthogonal to the gauge orbits, but the representative sets the metric. The pred box LP moves every coordinate by Δ, so groups whose gradients are 1e5 (d/e/f) dominate [E4].
   - Rebalance the gauge (idea 3), then report results per group.
2. **The piece set.** Without a complete set, ρ is an upper bound and pred is optimistic. "Stationary to x" then stands; "not stationary" does not.

### 3.3 The recipe

**(A) Dual residual.**

ρ_q(ε) = min ‖Σ λ_i g_i + Σ ⟨Y_b, ∇G_b⟩ + Σ μ_j ∇c_j‖_q, taken over:
- every top with s_i ≥ (1−ε)γ, with k x k PSD blocks where σ_k lies within ε (an SDP when k ≥ 3);
- every constraint with c_j ≥ −ε_c;
- λ, μ ≥ 0 and Σλ + Σ tr Y = 1.

Report:
- the profile over ε ∈ {1e-6, …, 1e-2}, normalised three ways: by the mean ‖g_i‖ (the current choice), by min_i ‖g_i‖ (the hull's min-norm point is at most min ‖g_i‖, so this is the honest scale), and per variable group;
- Kiwiel's linearisation-error weighting w = ½‖p‖²/σ + Σ λ_i (γ − s_i) + Σ μ_j (−c_j), in f-units, with the proximal weight σ and the metric stated (Ch. 2 Alg. 3.1 (3.3) and Lemma 4.4, Ch. 3, 6 [R]); dividing only the offsets by γ, as written before, mixes units. It needs no hard ε but trades it for σ, so it is metric-dependent like ρ, and −w is θ_σ of (B): report the two once;
- the F. John weight ν_f = Σλ + Σ tr Y. A small ν_f means the constraint qualification fails, not that the point is stationary (Kiwiel p. 17, p. 296-297 [R]);
- the multipliers against the penalty ν. A residual that needs μ > ν signals a penalty or CQ failure;
- the smallest separation among active closed-loop eigenvalues; the eigenvalue rows are invalid when it is small;
- the multipliers as first-order prices of the user's constraints (compensator line, closed-loop cushion, wall clearance). At SISO 598.11 the active modal rows carry μ‖h‖ below 1e-6 of the gradient scale, so the compensator line does not hold γ there [E3].

**(B) Primal side, without tolerances.**
- θ_σ(z) = min_d max_i [s_i − γ + g_i·d] + (σ/2)‖d‖² = −min_τ [Σ τ_i (γ − s_i) + ‖Σ τ_i g_i‖²/(2σ)] (AN 2006 eqs. 19-20 p. 79; θ ≤ 0, continuous, zero iff critical, Lemmas 3-4 [R]). AN's (19) is unconstrained (inf over h ∈ R^n); the constrained form is our extension: linear rows in the convex term (Drusvyatskiy-Lewis 2018 (5.1) [F]), closed-loop rows by an exact penalty (Drusvyatskiy-Paquette 2019 Ex. 3.3 [F]). The aggregate ‖p*‖ is the prox-linear step measure ‖G_t‖ at t = 1/σ [I], with ½‖G_t‖ ≤ dist(0, ∂φ) and a nearby nearly stationary point (DL 2018 Thm 5.6, Thm 5.3 [F]).
- Alternatively, the fixed-radius H = 0 box-LP decrease pred_inf(Δ) that the tool already computes. It is the first-order composite criticality measure φ^δ_{w,1} of Cartis-Gould-Toint (arXiv:2001.10802 eq. 3.11 p. 6 [F]), whose termination test is normalised by the radius (eqs. 2.4, 4.1): report pred_inf(Δ)/Δ, which is nonincreasing in Δ and tends to ρ_1 over the exactly active set as Δ → 0 [I]. (CGT 2011, cited here before, is [B].)
- The two are not equivalent; both vanish exactly at critical points.
- Report the profile over Δ and the validity radius Δ_v, the largest Δ at which the measured decrease is at least half the predicted one.
- Neither may use the BFGS matrix or the current trust radius. That dependence is the defect of "rel", and partly of "pred".

**(C) Completeness.**
- Run the level set at γ(1 − ε) to a full cover, excluding watched brackets (LESSONS §19.1).
- This is the expensive case: started at 0.999 γ, a level set ran 60 calls without certifying, against 1 call at γ (LESSONS l.250); central has 64 members within 1 %.
- Build the watched-bracket exclusion first. The tool's 40-90 min estimate has not been checked.

**(D) Certified descent.**
- Step to z + t·d*, relocate the tops, run the level set at the new max(1 + 1e-6), the line, and also the continuation gate or a count. A clear line is not stability (LESSONS §5).
- Only then report "NOT STATIONARY (certified)". The tool's `--descent certify` lacks the gate today.

**(E) Floor.** Claim nothing below the oracle's resolution:
- the Newton stop (slope ≤ 1e-7 σ);
- the certificate's rtol of 1e-6;
- the lag of t;
- rows reading a lower neighbouring top (LESSONS l.948-950).

Hare-Sagastizábal-Solodov 2016 [B] treat bundle methods with inexact information (title only; not read). The figure of "1.2e-7 to 9.6e-7 σ" applied to the superseded bracket-and-refine path (LESSONS l.214), not to the current code.

**(F) Report** for each final design, in the same form for SISO, decentral and central:
- the (ε, ρ) profile with three normalisations;
- the (Δ, θ or pred) profile and Δ_v;
- whether the piece set is certified;
- ν_f, μ/ν and the eigenvalue separation;
- active pieces and constraints against the **effective** dimension (NZ minus gauge: central 60 − 10 = 50; SISO 34 − 10 = 24);
- the constraint prices;
- the stop code.

Wording: "stationary to x at ε", never "local minimum" (LESSONS §4, §18.8).

**(G) How to use it.** As a diagnostic, not a licence to stop.
- On GL, residuals of 6e-4, 1.5e-4 and 9.6e-5 were followed by 19 %, 2.4 % and 1.3 % more descent (LESSONS l.243).
- Decentral 369.87, read "near ε-stationary", went 0.67-0.91 % lower in 20261010n.
- The stop rule should require criticality **and** progress, and never step length alone. The last part is the project's rule, not AN's: AN 2006 stop on (23)&(24) or on (25), the relative step, alone (p. 82 [R]), and their (23) is the exact Clarke min-norm test inf‖g‖ < 1e-5, which need not fire near a kink (Drusvyatskiy-Paquette 2019 p. 11 and fn. 5 [F]). Use θ_σ or ‖G_t‖ of (B) as the criticality half.
- Argue minimality separately: perturbation restarts and 1-D slices (HIFOO §5-6; Lewis-Overton §6.2; ABMV Fig. 4.1 [R]), later a projected second-order test on the identified manifold (Bareilles-Iutzeler-Malick 2023 Alg. 4.1, Thm 4.5; Lewis-Wylie 2021 Thm 5.5 [F]; Bonnans-Shapiro, Hettich-Kortanek [S] remain pointers for the semi-infinite reduction). With a gauge-free compensator form there is no gauge to project out.

**What can honestly be claimed.** ε-active stationarity at one point, with the ε-active set certified complete if (C) ran. Goldstein (δ, ε)-stationarity (Zhang et al. 2020 Def. 4 p. 4 [F]) needs gradients over a whole δ-ball, including moving tops; nothing here measures it. One direction transfers [I]: with L_f-Lipschitz pieces and L_g-Lipschitz gradients, the level-set residual at absolute ε is at most dist(0, ∂_δ f) + L_g δ for δ = ε/(2 L_f), so a large ρ certifies non-Goldstein-stationarity there and a small ρ certifies nothing. Zhang's Thm 5 (no finite certificate of Clarke ε-stationarity) is a black-box result and does not apply to an oracle that exposes its pieces. With a growing Ω, claim only "δ-stationary for the problem over Ω_k; Ω_k ε-complete yes/no" (consistent approximations, Royset 2023 Def. 2.2, Prop. 2.3 [F]).

### 3.4 Readings so far

Preliminary: completeness off, no gate. Smoke runs [E3-E5] and descent probes [E6].

| Record (stop) | Pieces at ε = 1e-3 | ρ2 / mean ‖g‖ | ρ2 / min ‖g‖ | Measured decrease | Note |
|---|---|---|---|---|---|
| SISO 598.11, 20261010g (progress rule; last QP code 0, stat 6.1) | 1 | 1.0 | 1.0 | −1.02e-3 γ at ‖d‖∞ = 1e-6 (ratio 1.00); −8.96e-3 at 1e-5 over 3 tops, −7.0e-3 once a fourth top at 0.986 γ is included (tool docstring) | One smooth top; the next tops lie 0.65 % lower, so the 1e-6 verdict is robust. The size beyond 1e-6 is not established. |
| Central 123.90, 20261010n c151 async (progress 0.124 % over 3 rounds) | 13 (12 tops plus 1 edge stop, "NOT A TOP") | 5.1e-5 | 1.2e-3 (3.39 / 2845) | −1.38e-4 at 1e-5; +1.37e-3 at 1e-4 | The 2x2 block top near 0.92 rises first. pred_inf at the record's trust radius 4.3e-2 is 2.7e-3 γ. |
| Decentral 369.87, 20261010g (deadline, code 4) | 9 | 9.6e-4 | 1.6e-3 (8.09 / 5113) | −3.38e-4 at 1e-5; +4.7e-4 at 1e-3 | Later 366.49-367.40 in 20261010n (stability vetoed). Its sensor windows start 0.0012 apart (LESSONS l.631). |

STATUS.md l.177 and BFS.md l.818 still call both SISO records "stationary (two local minima)". LESSONS §14 and these readings contradict that wording; it should be withdrawn.

---

## 4. Parallel efficiency: techniques from the literature, mapped to measurements

**Where the cores idle (20261010n, 48 ranks) [E1]:**
- Evaluation waves are 32-61 % busy: 51-61 % in synchronous no-strip arms, 32-44 % in strip arms, 39-40 % in async arms.
- Synchronous level waves are 11-27 % busy.
- The background level set in c151 async is 27 % busy: 38,800 s of wall.
- **The async level set still sits on the critical path at its joins.** It waited 437-2,335 s per solve of 3,260-5,035 s in central (typically about 30 %) and 488-812 s per solve of 2,228-3,838 s in decentral (14-28 %). The "about 60 s a round" in BFS.md l.954 counts only the foreground level calls.
- Level-pencil tasks are heavy-tailed. Mean against longest: 29.8 s against 102 s (c151), 25.6 s against 213 s (c177), 25.3 s against 154 s (d412).
- Every synchronous wave lasts as long as its slowest task.
- The driver's serial work is 0.4 % (LESSONS §19.4): leave it alone.

**Levers, with literature and expected payoff:**

1. **Split or hedge the heavy level-set tasks.** This is the main lever for central async.
   - Contour-filter disks: SLEPc CISS with `-eps_ciss_partitions` splits the shifted solves over sub-communicators (ciss.c l.862-864) [F]; FEAST L1/L2/L3 parallelism [S].
   - CISS needs off-lattice factors and a shell operator for the CG-eliminated pencil. Its count estimate is stochastic, so a disk is still cleared only by exhibited, residual-checked eigenpairs.
   - Hedged ncv-100 attempts on idle ranks (Dean-Barroso 2013 [S]) stay result-identical if the ncv-24 result is always taken when it converges. LESSONS §19.4 rates the hedge "doubtful until measured" (15-20 % of central).
   - Grep finds no CISS, FEAST or Sakurai in the notes.
2. **Moldable tasks.** Give the longest tasks of a wave more cores: MUMPS ICNTL(16) sets threads per call and restores the previous value afterwards (guide §5.17 [F]); shelf allotment as in Mounié-Rapine-Trystram 2007 [S]. Gains are limited because triangular solves are memory-bound. Not found in the notes.
3. **Fill idle ranks with independent work.**
   - An asynchronous winding count of each accepted design, once the tail bound is fixed (idea 10).
   - Several multistart drivers sharing one pool and its design-independent factors.
   - A third arm, or oversubscription with yielding waits: the farm busy-polls today (LESSONS l.973), and the limit is memory (133 MB per factor, cap 24).
   - Portfolio and oversubscription are not found in the notes.
4. **Shorten the critical path inside a task.**
   - Reuse the symbolic analysis across the 156k exact-point factorisations: MUMPS JOB=2 or ICNTL(7)=1/PERM_IN (guide p. 24 [F]), or petsc4py factorSymbolicLU/factorNumericLU. Measure METIS's share of the 0.54 s first.
   - A drift predictor for each top's new frequency, to cut the 6 cold Newton waves (BFS.md §3). The implicit-determinant Newton of Freitag-Spence-Van Dooren 2014 [B] is the published joint (ω, γ) form.
   - A block or warm-started subspace iteration for σ1 (Ribeiro-Yeh-Taira 2020 [S]; RSVD-LU [F]), finished by Lanczos steps to reach the 1e-11 the Newton stop needs.
5. **Structured parallel eigensolvers along the axis.** Gobbato-Chinea-Grivet-Talocia 2011 run Arnoldi processes concurrently at many shifts [S]. Parallel rational Arnoldi with poles at the cached lattice shifts (Berljafa-Güttel 2017 [S]). Not found in the notes.
6. **Full mesh (memory-bound).**
   - PEXSI-style groups of 2-4 ranks per frequency [S].
   - BLR factors with refinement (guide §3.18 [F]; Amestoy et al. 2023 [S]). LESSONS §9 bans perturbed-pivot solvers on the saddle pencil, so only with residual-checked refinement and a validation against full-rank certificates.
   - cuDSS batch solves (README [F]), only after the LESSONS §10 timing test.

**Low payoff, do not prioritise:**
- multi-RHS MUMPS solves for 1-2 columns;
- speculative line-search trials, which are not result-identical and gain little (LESSONS §8, §19).

**Measure first:**
- per-task cost logs per wave type;
- the METIS share;
- join waits per round, already in the ledger.

---

## 5. Ranked ideas

None is to be implemented now. Concerns: (1) validation, (2) stationarity, (3) parallel efficiency, (4) a better algorithm.

**1. Fix the winding tail bound, then re-count the vetoed designs.** (1)
- *Source:* IJRNC 2018 Remark 6 (a derivative decay bound for the Nyquist march) and Thm 1(iii) [R], as analogues; the bound itself is ours.
- *Payoff:* very high. It unblocks every decentral record. If the counts come out 0, the records become 366.49 (from 369.87) and 388.44 (from 412.28, −5.8 %).
- *Cost:* in progress; then 5 counts of about 400 s each, provided the round designs are retained.
- *Risk:* the new bound must be proven, not only tighter. The vetoed designs may really count ≥ 1: the probe reproduces the old failure when D_c is scaled by 1.3 (|k|·g_b(W) = 0.26 ≥ q = 0.25) [E2], so their high-frequency gain probably grew [I].
- *Tried:* the old H^1 tail failed 5 of 5. The new bound gives W_T = 7,680 / 3,840 / 1,920 against 7.86e6 / 1.97e6 / 4.9e5 for 369.87 / 412.28 / c151 [E2].

**2. Use 8 rounds for the MIMO progress stop.** (2)
- *Source:* LESSONS §1 l.132-136; AN 2006 (23)&(24) [R].
- *Payoff:* high. It removes a documented premature stop (c151 async at 0.124 % over 3 rounds; 20261010e at 387.06, later 61 % lower).
- *Cost:* trivial (bfs/run_bfs.sh:131).
- *Risk:* more wall per run; no pure stop-rule A/B exists (LESSONS l.136).
- *Tried:* the rule is written down but not applied.

**3. Rebalance the gauge of the modal blocks.** (2), (4)
- *Source:* no paper read prescribes it. HinfMOR p. 5 picks a structured (tridiagonal/diagonal) realisation, not a canonical one, and with B, C, D free keeps a diagonal-scaling gauge [R]. The identification literature treats the similarity gauge (data-driven local coordinates, McKelvey-Helmersson-Ribarits 2004 [S]; rank-deficient Jacobians, Wills-Ninness-Gibson 2008 [S]; a penalty on the similarity factor, Monti 2026 §5.4 eq. 39 [F]); none is a gauge-free chart. Superseded on 2026-10-17 by the per-block normal form being implemented, which removes the gauge instead of rebalancing it. This is otherwise an inference from /home/user/codesign/packer.py l.349-357 and the measured cond(V).
- *What:* at round start, where BFGS resets anyway, apply to each block the 2x2 similarity that keeps the fixed B_c entries and minimises cond(V), giving a normal form where it is reachable; transform C_c accordingly.
- *Payoff:* unknown, possibly large. It is result-neutral at the point (transfer function, rows and certificates unchanged). It removes 10 flat directions and may lift the 1e-5 validity radius and the code-4 crawl.
- *Cost:* low.
- *Risk:* it changes trajectories, so it needs a clean A/B judged against the 0.7 % spread.
- *Tried:* LESSONS l.379 says "check the modal compensator's gauge the same way (inference)"; never done. STAGE6 §6.16 fixed only the tridiagonal gauge.

**4. Use the BFS stationarity tool as a diagnostic.** (2), (1)
- *Source:* §3: AN 2006 §VI-VII, Kiwiel Ch. 3-6, TAC 2015 Props 1 and 3, SMV [R].
- *Add:*
  - normalisation by min ‖g‖ and per group;
  - a continuation gate in `--descent certify`;
  - Kiwiel's w, ν_f, μ/ν, the eigenvalue-separation flag and the constraint prices;
  - completeness only after the watched-bracket exclusion.
  Run it after idea 3, and correct STATUS.md l.177 and BFS.md l.818.
- *Payoff:* medium. It gives the first defensible BFS statements, but cannot license a stop (§3.4).
- *Cost:* low: 5-10 min per record with completeness off.
- *Risk:* the conventions on ε, Δ and scaling; expensive completeness on plateaus.
- *Tried:* GL (/home/user/codesign/tools/stationarity.py, tools/kkt_residual.py); BFS smoke runs only.

**5. Compute per-port (Parrott) lower bounds, then position maps.** (4), (1)
- *Source:* Davis-Kahan-Weinberger 1982 [S]; GL /home/user/codesign/tools/bounds.py [C]; the HinfMOR Hankel-bound practice and the AN 2006 full-order column [R].
- *What:* write G_cl = T + L(x_a) Q R(x_s), where L is fixed by the actuators and R by the sensors. Frequency by frequency, the minimum over all Q is max(‖P_L^⊥ T‖, ‖T P_R^⊥‖). That is the exact non-causal optimum at those ports, and a lower bound for any controller of any order.
  - Any sampled frequency already gives a valid bound.
  - Since the two terms depend only on the actuators and only on the sensors, max(min over x_a of A*, min over x_s of S*) bounds every placement and every controller. It is rigorous with a Lipschitz bound in position, otherwise an estimate.
- *Payoff:* high for "how close to optimal", and it screens positions globally.
- *Cost:* low to moderate: a rank-m deflation and one σ1 per frequency per bound, on the lattice factors.
- *Risk:* convective delay may put the non-causal bound far below the causal optimum (Martini et al. 2022 [F]: on a BFS at Re = 500 the non-causal optimum bounds any linear control and is solvable frequency by frequency, and "actuator placement has a negligible influence" on it, while causal control loses 50 % at about 4.5 step heights between actuator and sensor; §4.1.1 p. 22, an H2/PSD setting). Whether our H∞ full-field per-port bound is equally flat in the actuator position is untested: measure the map before screening with it. The port-free open-loop floor σ_{m+1}(T_zw) (≥ 32.66 SISO, ≥ 25.00 2x2) is already weak.
- *Tried:* GL yes (9.832 / 9.660 / 3.509; STAGE5 l.1868; LESSONS l.622), never at optimised ports and never on the BFS.

**6. Audit the γ certificate (on-axis classification and slivers).** (1)
- *Source:* SMV §2.3 (eq. 5) and §3; Bruinsma-Steinbuch Remark 3.4 [R].
- *What:*
  - log the real part and the mirror-pair asymmetry of every converged Ritz value near the axis;
  - σ-confirm every Ritz value whose real part is below a threshold tied to ε_eigs (exactly the set whose disks can swallow or isolate a crossing), not only below 1e-8·max(1, abs λ);
  - count and evaluate the dropped slivers;
  - cross-check the round certificate against the lattice level set every round;
  - correct the docstring at /home/user/codesign/certify_sparse.py l.36-45.
  Move to SMV's K(ξ) only if the logs show real parts in the 1e-8 to 1e-6 range.
- *Payoff:* medium. It closes the one route by which a tolerance can lose a true crossing.
- *Cost:* low.
- *Risk:* none to rigour.
- *Tried:* no structure-preserving solver in the notes; the two certificates were cross-checked once (job 767386).

**7. A validation stack.** (1)
- *Source:* SMV §3, Bruinsma-Steinbuch, IJRNC (cross-check with systune) [R]; Benner-Sima-Voigt 2012 [B] (dense structured factorisations are practical only up to q of a few thousand, SMV p. 2 [R]: the n ≈ 1-3k mini-mesh, q = 2n + m + p ≈ 2-6k, fits; a 6-8k mesh, q ≈ 12-16k, does not [I]); Barbagallo et al. 2012 (nonlinear simulation) [A].
- *What:*
  - (a) SMV's α-sweep per structure, through both certificates;
  - (b) a dense mini-mesh (n ≈ 1-3k, plus one 6-8k), re-deciding every verdict by dense structured QZ and dense eig;
  - (c) the open-loop σ1 sweep on the full mesh, which takes minutes;
  - (d) a time-domain check with one factor of M/dt − A: harmonic forcing at the peak with the optimal forcing should give σ1, and a random initial condition should decay at a rate of at least ε.
- *Payoff:* high.
- *Cost:* a few node-hours plus scripting.
- *Risk:* a coarse mesh may not exercise the hard regimes; at margin 0.05, transients decay slowly.
- *Tried:* mfree on 1,279 dofs; check_lattice once; GL dense audits. None of (a)-(d) on the BFS; "time-domain" does not appear in the notes.

**8. A convex Youla benchmark at fixed ports (central, SISO).** (1), (4)
- *Source:* Boyd-Barratt 1991 [S]; Lessard-Lall 2016 for quadratic invariance [S].
- *What:* the plant is stable with margin (rightmost pole −0.1148, BFS.md l.239), so T_cl = T_zw + T_zu Q T_yw is affine in Q. A Laguerre or Kautz Q with poles left of −ε keeps the margin by construction, and dropping fixed order and compensator stability makes the problem convex.
  - It is a known-answer test of SQP, Ω, certificate and stationarity tool: any stop above the convex optimum is a defect.
  - It gives an upper bound, for the chosen basis, on the full-order value.
- *Payoff:* medium-high.
- *Cost:* moderate (a new parametrisation on the same lattice).
- *Risk:* one-sided bounds only; decentral is not quadratically invariant, so it has no convex form.
- *Tried:* "Youla" does not appear in the notes.

**9. Cross-structure identity tests.** (1)
- *Source:* AN 2006 Ex. 6, the decentral subgradient as a Hadamard mask of the central one [R].
- *What:* (a) embed a decentral record into central and check identical rows, γ and masked gradients; (b) a doubled plant diag(A, A).
- *Payoff:* medium-high.
- *Cost:* low for (a), moderate for (b).
- *Risk:* use a record without coincident sensors (369.87's are 0.0012 apart); bitwise agreement is limited to about 7e-13.
- *Tried:* SISO = MIMO at 1.1e-14 and `--force-mimo` at 2.5e-9; (a) and (b) not found.

**10. An asynchronous winding count of every accepted design, after idea 1.** (1), (3)
- *Source:* IJRNC 2018 Alg. 2, a Nyquist test at every primary step [R].
- *Payoff:* medium-high. It would have caught the tail failure at round 1 and kept a counted fallback near the end instead of the job start.
- *Cost:* moderate (snapshot and rollback already exist).
- *Risk:* a rollback discards rounds; transient memory per count.
- *Tried:* `--wind-accept off` overrides LESSONS §5's MIMO rule; an asynchronous count is not found.

**11. Re-certify on the full mesh, after ideas 1 and 7(c).** (1)
- *Source:* Kasinathan-Morris 2013 [B] (optimal locations computed on approximations converge, per Edalatzadeh-Morris 2018 p. 1 [F]); Morris 1994 (unverified). For "optimise coarse, finish fine": Werner-Overton-Peherstorfer 2023 Alg. 1 (RMF-GS) [F], whose high-fidelity value is not monotone across levels (p. 18).
- *Payoff:* medium-high for credibility.
- *Cost:* about 2 h for the plant certificate on 48 ranks, then 5-10 min per design for line plus γ, and 20-40 min per count.
- *Risk:* peaks may move. Domain truncation is a separate axis (BFS.md l.965).
- *Tried:* not done (LESSONS §8, §14).

**12. A beyond-band screen for SISO and decentral.** (1)
- *Source:* IJRNC, Bruinsma-Steinbuch σmax(D), ABMSV greedy, Mengi 2022 dominant poles [R].
- *Payoff:* medium. *Cost:* low. *Risk:* not a certificate.
- *Tried:* recommended in LESSONS §5 open questions; GL used `--cert-all-w on`.

**13. Endpoint-perturbation restarts and SISO multistart, after idea 3.** (2), (4)
- *Source:* HIFOO §5-6, Lewis-Overton §6.2, AGB ECC 2014 [R].
- *Payoff:* medium-high. It also answers whether 554.97 and 598.11 are two basins.
- *Cost:* k copies x 2-3 rounds.
- *Risk:* size the perturbations per variable group (the model holds to about 1e-5) and budget for gate refusals.
- *Tried:* GL 1e-12 path-spread test; SISO multistart planned.

**14. A criticality-gated stop and "pred" for SISO, after idea 2.** (2)
- *Source:* AN 2006 (23)&(24); AN 2007 §6.2 [R].
- *Payoff:* medium. It avoids premature stops; it does not speed convergence.
- *Cost:* small.
- *Risk:* SISO 598.11's single top has a tiny validity radius, so the SQP may keep taking steps the merit rejects. A/B it together with idea 3. "pred" cost 0.08 % on GL.
- *Tried:* `--sqp-stat pred` on GL (STAGE6 §6.10, 6.14). A gated stop is not found. LESSONS §11 says the central-plateau levers are multistart, budget and lower order, not stop-rule tweaks.

**15. Parallel levers (§4).** (3)
- *Source:* §4.
- *Payoff:* medium for central async, where the joins take 9-50 % of the solve; low elsewhere.
- *Cost:* low (hedge, measurements) to high (CISS, rational Krylov).
- *Risk:* result identity; follow the LESSONS §19.3 protocol.
- *Tried:* the hedge was proposed; the rest not found.

**16. A small-gain (structured stability radius) gate for each step.** (1)
- *Source:* Benner-Voigt 2014 §2-3: 1/‖G‖∞ equals the structured complex stability radius [R].
- *What:* bound the low-rank change of each step through the ports and controller against the anchor's H∞ norm on Re s = −ε. That covers far-left jumps and eigenvalues beyond the band in Im, without tracking.
- *Payoff:* low-medium. *Cost:* moderate.
- *Risk:* very conservative on this non-normal pencil; measure first.
- *Tried:* not found.

**17. Make the winding count a proof.** (1)
- *Source:* IJRNC Lemma 2, Thm 2 [R]; Kamgnia-Philippe 2013 [F] (the criterion is Condition (B), eq. (9) p. 7; their own step control is heuristic, §4.3); Ying-Katz, Johnson-Tucker [B] (exclusion with a certified bound on f'', or Krawczyk-validated enclosures, per Batra-Sharma 2019 p. 10-11 [F]). The literature gives a criterion, not a ready method.
- *Payoff:* medium. *Cost:* the "about 8 extra mat-vecs per point" of the winding_cert docstring cannot give a proof if it is a Krylov norm estimate, which bounds from below. A proof needs a certified upper bound on the resolvent along each segment, e.g. an inertia (LDLᵀ) test of a Hermitian dilation at the knots, extended along the segment by Weyl's inequality [I]. Expect more points: Kamgnia-Philippe's safe control needed 1519 intervals where 825 already gave the right count, and 7,986 intervals (54 h 42 min, two LU factorisations per point, on 2012 hardware) at n = 17,281 (§5, Tables 2 and 4). Apply it only where the validated march has thin margins.
- *Risk:* slow or inconclusive counts near the 282 hot plant poles.
- *Tried:* "NOT IMPLEMENTED" (winding_cert.py l.48-55).

**18. A second-order tail on the identified manifold, after ideas 3 and 4.** (4)
- *Source:* Bompart et al. 2007 [B] (still unread); Noll-Apkarian 2005 [B] (heuristic multiplicity identification, then an SQP step, per Bareilles et al. 2023 p. 3 [F]); Overton 1988 [S]; Yuan 1985a/b [B] (Math. Program. 31:220-228 and 269-285, Lewis-Wright refs [52], [53]; LW §3 p. 9 credits the Cauchy-point analogue, a line search on a piecewise quadratic used to accept a higher-order step, to the second and to Wright); Fletcher 1982 [S]; Fletcher-Sainz de la Maza 1989 [B]. Multiplier-weighted per-piece Hessians: L-V eq. 3 and p. 375 (finite-difference Hessians gave the same iteration counts) [R]; Lewis-Wylie 2019 Alg. 2.2 p. 4 [F] (its k-step quadratic rate, Thm 5.11 p. 18, needs strongly convex pieces with positive definite Hessians and the active-set size k given). Dao et al. 2016 §8.2 [F] sum the unweighted Hessians of the attaining pieces. Bareilles et al. 2023 Alg. 4.1, Thm 4.5 [F]: SQP on the prox-identified manifold, locally quadratic at a strong minimiser, no Maratos effect assumed. None covers moving tops or a piece set that changes mid-solve.
- *Diagnostics first:*
  - active pieces against the effective dimension (central: 13 against 50; SISO: 1 against 24, i.e. a smooth point);
  - strict complementarity;
  - where the linear model breaks down (the c151 block first).
  Then A/B two options: a second-order correction on the active σ rows, and multiplier-weighted per-piece Hessians with the gauge projected out.
- *Payoff:* uncertain.
- *Risk:* the GL negatives (CTRL_HESS; the unsafeguarded polish of STAGE2 §3.4) arose at a compensator double pole, which the modal form removes, so they are weaker evidence than they look.
- *Tried:* CTRL_HESS and SOC_CTRL removed on GL; second-order correction on the BFS compensator line works; σ-row correction and per-piece Hessians not found.

**19. An order ladder.** (1), (4)
- *Source:* HIFOO §5, §6.3 [R].
- *Payoff:* medium. *Cost:* about 4 runs per structure.
- *Risk:* monotonicity holds only for global optima.
- *Tried:* GL orders 5 against 16 only.

**20. Robustness, fragility and multi-model reporting.** (1)
- *Source:* AGB ECC 2014 [R]; Keel-Bhattacharyya 1997 [S].
- *Why:* a step of 1e-4 in packed coordinates raises SISO γ by 0.5 %, and a step of 1e-3 raises central by 1.1 % [E6]. The compensator poles sit at −0.05 and are lightly damped.
- *Cost:* low. *Tried:* not found.

**21. Pole-targeted interpolatory rows.** (4)
- *Source:* Mengi 2022 Alg. 3.1; Benner-Voigt dominance [R].
- *Payoff:* low.
- *Risk:* high. Near-line eigenvalues missing from the basis was the failure that mattered (LESSONS §7-8: −0.1444 hidden behind −0.0723), and with the strip removed a miss costs what the strip guarded against.
- *Tried:* `--cont-track reduced` on GL (42 of 42 decisions); the reduced oracle was rejected on the BFS.

**22. A Lipschitz stability function (shifted norm or pseudospectral margin).** (4)
- A specification change, so a user decision.
- *Source:* AN 2007 Model II; erratum eq. (5) [R]; Burke-Lewis-Overton 2003 [S] (unread). The property used is stated in Aliyev-Mengi 2024 §1 p. 1-2 [F]: the abscissa need not be Lipschitz, even locally, at an optimiser with a nonsimple rightmost eigenvalue, while the ε-pseudospectral abscissa is locally Lipschitz (Lewis-Pang 2008 Cor. 7.2; Gürbüzbalaban-Overton 2012 Cor. 3.4 [B]). With the normal-form compensator the compensator rows are exactly linear, so the idea now concerns only the closed-loop rows.
- *Risk:* the optima park eigenvalues at the cushion, where the shifted norm blows up.
- *Tried:* not found.

**23. Frequency-domain convex-concave inner approximations, to assess only.** (4)
- *Source:* Karimi-Kammer 2017; Hast et al. 2013; Boyd-Hast-Åström 2016 [S].
- *Weakness at our scale:* a dense frequency grid with factorisations, and no certified sup.
- *Tried:* not found.

**Not recommended (listed so they are not retried):** (Note added after the re-read of 2026-10-16, REREAD_2026-10-16.md: the LESSONS rule "do not add another inner solver" cited below lost its justification on the BFS, because its premise, that the inner solve is not the bottleneck, is false there. These items are not recommended on structural and iteration-count grounds, with cost per iteration deciding only against gradient sampling, not on a BFS measurement.)
- bundle or proximity-control methods, gradient sampling, or GRANSO/HANSO as the inner solver. Gradient sampling costs at least n + 1 gradients per iteration (basic GS); bundle and proximity methods cost about 1-2 evaluations per iteration but are first-order with many null steps (L-V Table 3 p. 390 [R]); none represents the 2x2 Clarke block or rows added mid-solve (PyGRANSO has no cones, NCVX 2022 §5 [F]); SQP-GS's better minimisers in CMO §6 (78 % against 22 % of the best on the spectral-radius set) cost 26.6x the run time [R]. The Apkarian-Noll bundle trust region is not another solver class: with a box it solves our QP (ANR 2018 Remark 4, eq. 8, p. 7 [F]) and differs only in its radius, null-step and Q management, which are candidates for our own rule;
- IPOPT, SNOPT, KNITRO, WORHP, SLSQP or Uno as a drop-in inner solver (2026-10-17 refresh): C2 and fixed-problem assumptions (Ipopt main.dox l.43-44 [F]), weak interior-point warm starts after row changes (Ipopt special.dox; WARP 2026 Table 2 p. 4 [F]), a scaled L-BFGS with history 6 by default, which often breaks down on effectively nonsmooth problems (Asl-Overton 2020 [F]), no trust region, and no saving in evaluations on small minimax NLPs (Uno 2025 Table 5 [F]). IPOPT is installable, so one bounded GL diagnostic is possible (protocol in the refresh's addendum);
- a reduced or multi-fidelity inner model;
- spectral-value-set or HEC norm oracles;
- SOS/moment global bounds (Henrion-Lasserre 2006 [S]): they work only at tiny state dimensions and would bound a ROM;
- sparsity-promoting actuator selection (Zare et al.; Münz et al. [S]): H2 or state feedback; the project's nearest analogue is the GL field/atom work.

Evidence: LESSONS §4 (penalty gradient sampling is not a certificate at a cusp); §7. LESSONS §1 "Do not add another inner solver or solver class" has lost both its premises: the inner solve is the bottleneck on the BFS (LESSONS l.1082), and IPOPT is installable (the PyPI casadi 3.8.1 wheel bundles Ipopt 3.14.19 with sequential MUMPS; conda-forge has cyipopt 1.7.0 and Ipopt 3.14.20 on mumps-seq 5.8.2 and SPRAL), with an untested risk from sequential MUMPS's fake MPI inside the mpi4py farm (Ipopt install.dox l.303-315 [F]). Werner-Overton-Peherstorfer 2023 [F]: gradient sampling solved fixed-order H∞ (n_K = 2) on cylinder models up to 22,060 states in 37.9-86.4 h per run (Table 6 p. 20), with a norm without a global certificate (p. 15), and was "more consistent and reliable" than BFGS on the cylinder (p. 19). That supports feasibility and confirms the cost argument.

---

## 6. The papers in the user's archive

**Apkarian and Noll, "Nonsmooth H∞ synthesis", IEEE TAC 51(1):71-86, 2006 [R].** The foundation for us.
- It gives the Clarke subdifferential of a structured closed loop's H∞ norm (eq. 13), plant augmentation for dynamic K, the Hadamard mask for decentralised K (Ex. 6), and a chain rule for any smooth parametrisation.
- It rejects steepest descent as discontinuous. Variant I uses the ε-enlarged subdifferential.
- Variant II uses the optimality function θ (eqs. 19-21): θ ≤ 0, continuous, zero iff Clarke-critical.
- The semi-infinite case refines Ω at a fixed iterate (§VI-E, Lemma 7). In practice Ω is rebuilt at every iterate from Hamiltonian peaks and log-spaced points, capped at 50 (§VI-G).
- Stopping rules are (23)-(25). Table II: COMPleib up to 240 states (AC8 2.005 against FW 2.612; HF1 at the full-order bound).
- The authors call the method first-order and slow near a solution (p. 82).
- For us: our QP is Variant II plus BFGS and a trust region; our certificate replaces the Hamiltonian; our stop lacks θ.

**Apkarian and Noll, "Erratum to 'Nonsmooth H∞ synthesis'", IEEE TAC 51(2):382, 2006 [R].** It corrects four typeset formulas: the shifted-norm stabilisation (5), the abscissa (7), simultaneous stabilisation, and (8). There is no new content. It is needed to read Examples 1-4 and confirms that their stabilisation uses a shifted H∞ norm, not the abscissa.

**Apkarian and Noll, "Nonsmooth optimization for multiband frequency domain control design", Automatica 43:724-731, 2007 [R].**
- It minimises a max of band-restricted norms and treats stability as a constraint, in two ways:
  - Model I, a distance-to-instability homotopy;
  - Model II, a shifted-norm barrier with α < 0, which is our Re λ ≤ −ε specification made Lipschitz.
- The subdifferential is Thm 7; θ_e is computed on an enriched frequency set (eqs. 16-18).
- Table 1 shows that solving Model I directly at the target fails, so a homotopy is required. Both models then reach 0.84 / 0.84 / 0.31 / 0.84 on a double integrator.
- For us: the enriched set is our accumulated Ω, and Model II is the Lipschitz alternative to our eigenvalue rows. The example is a toy.

**Apkarian and Noll, "Optimization-based control design techniques and tools", arXiv:1905.11511, 2019 [R].**
- An essay on structured multi-model synthesis: soft objectives, hard constraints, proximity-control bundle methods with a trust-region extension, hinfstruct/systune.
- The Hamiltonian is costly above about 500 states; outer iterations scale with dim x.
- For us: it confirms the paradigm, and that the tools themselves do not reach our n. It offers no parallel scheme.

**Apkarian, Dao and Noll, "Parametric robust structured control design", IEEE TAC 60(7):1857-1869, 2015 [R].**
- Algorithm 1 (dynamic inner approximation): a multi-model synthesis on a growing scenario set Δ_a, local worst-case searches for destabilisation and performance, and an a-posteriori mixed-μ certificate.
- §III separates upper-C1 min-min worst-case searches (treat them as smooth on an active branch) from lower-C1 min-max synthesis (needs a tailored method).
- Props 1-4 give the subdifferentials; the abscissa needs semi-simple eigenvalues. Convergence is via Łojasiewicz. The stop (Remark 9) is relative step and decrease.
- For us: the structure of our outer loop, with ω as the scenario.

**Apkarian, Gahinet and Buhr, "Multi-model, multi-objective tuning of fixed-structure controllers", ECC 2014, pp. 856-861 [R].**
- The core of systune: minimise the soft requirements subject to hard ones ≤ 1 over a model family.
- Requirements that are not nearly active are pruned (κ = 0.2). Hard constraints are handled by Φ_η = max{f, ηg} with bisection on η, after the paper rejects barriers and notes that an exact penalty needs c ≥ λ*. It advises several starts.
- Examples: 9-model fault-tolerant flight control, 27 parameters, 30 s; a 28-state building, 11 s.
- For us: our steered exact penalty is a direct answer to their concern about choosing c. Pruning whole requirements is not pruning frequency samples. Multi-model design is the natural robustness extension.

**Apkarian and Noll, "Structured H∞-control of infinite-dimensional systems", IJRNC 28:3212-3238, 2018 [R].** The closest analogue to our pipeline.
- Synthesis on frequency samples of the unreduced model.
- An infinite-dimensional Nyquist theorem (Thm 1) and a sampled Nyquist certificate (Lemma 2, Thm 2).
- A bundle trust region with a modulus-margin barrier and a Nyquist test at every primary step (Alg. 2). Remark 9 prefers abscissa constraints when A(K) is available; Remark 11 says Cauchy points fail. Precisely (ANR 2016 [F]): the Cauchy point fails for the one-plane standard model of a nonsmooth function (counterexample §5.7 p. 17-19, a convex piecewise-affine f on R²); a classical ratio rule is valid with the strict natural model of a convex composite (Remark 2 p. 3, §5.1-5.2 p. 13) and for upper-C1 f (Thm 2 p. 16). For a ratio rule the danger is an incomplete model, not the ratio test.
- Performance verification on a Lipschitz grid (Lemma 3) with a finite-difference L (Remark 12), so not rigorous; Ω_opt fixed per solve (Alg. 3); conical damping on the controller modes.
- 28 COMPleib plants and 12 infinite-dimensional studies; a dense grid gives the wrong winding number on DLR1.
- For us: our in-band certificate is stronger (relative 1e-6, by exhibition) and our band coverage weaker. Their Lemma 2 is the rigorous rule our march lacks. Their p x m samples are precomputed offline; ours cannot be.

**Burke, Henrion, Lewis and Overton, "HIFOO – a MATLAB package for fixed-order controller design and H∞ optimization", ROCOND 2006 [R].**
- Fixed-order output feedback: an abscissa phase, then H∞ by HANSO (BFGS, then local bundle, then gradient sampling). A dense norm is computed at every evaluation.
- Random restarts; order augmentation without increasing f; the local-optimality pair (dnorm, evaldist).
- Examples: a multiplicity-six optimum on the two-mass spring (the analytic pole placement at −√15/5; HIFOO reached −0.7572 with clustered, not coincident, poles, §6.1); AC2 static output feedback reaches the full-order value; AC4 orders 0-3.
- For us: the same problem class at small scale. Restarts, order continuation and a reported local-optimality pair are its practices we lack. Its implicit handling of nonsmoothness, through an ill-conditioned BFGS matrix, is what our explicit epigraph avoids.

**Lewis and Overton, "Nonsmooth optimization via quasi-Newton methods", Math. Program. 141:135-163, 2013 [R].**
- Plain BFGS with a weak-Wolfe search on nonsmooth functions. Q- and R-linear rates on model problems; the eigenvalues of H_k that go to zero span the V-space (44 of them on a multiplicity-9 example).
- A practical hull-of-recent-gradients stop (§6.3); open conjectures (Challenge 7.1).
- For us: it supports the explicit lifting and gives the U/V diagnostic. A history-based stop cannot work with our changing Ω.

**Curtis, Mitchell and Overton, "A BFGS-SQP method for nonsmooth, nonconvex, constrained optimization…", OMS 32(1):148-181, 2017 [R].**
- GRANSO: penalty SQP with BFGS; steering (Procedure 1) after Byrd-Nocedal-Waltz; a stationarity QP over recent gradients ((12)-(13)).
- Relative minimisation profiles for benchmarking. A fixed penalty with restarts (SFPP, μ lowered every 20 iterations) is slow to become feasible: well over 100 iterations in the §4.2 example (p. 12), and fewer feasible points than SNOPT over §6, neither finding one on every problem.
- For us: the closest relative of our inner solver. Our "pred" is their measure with l = 1, but truncated by the trust region. RMPs would be a good format for A/B comparisons across arms.

**Guglielmi, Gürbüzbalaban and Overton, "Fast approximation of the H∞ norm via optimization over spectral value sets", SIMAX 34(2):709-737, 2013 [R].**
- Two copies are in the archive: guglielmi2013.pdf and FAST_APPROXIMATION_OF_THE_H_NORM_VIA_OPT.pdf are the same 29-page article from two downloads (different md5; the text differs only in the download stamp and glyphs). Keep one.
- Method: rank-one perturbation iterations for the spectral value set abscissa, plus Newton on ε. It gives lower bounds only, with E = I.
- It fails on CM3/CM4, HIFOO-optimised loops with many near-equal peaks, and on tolosa.
- For us: no use as an oracle. Read with Benner-Voigt, it says our γ is the reciprocal of a complex stability radius in the energy norm.

**Benner and Mitchell, "Faster and more accurate computation of the H∞ norm via optimization", SISC 40(5):A3609-A3635, 2018 [R].**
- A hybrid: local Newton or secant on the level-set intervals, then one global eigen-check.
- 1.72x faster than BBBS and 11x faster than hinfnorm at tolerance 1e-14. At large scale, direct LU-based optimisation is 3.7x faster than hinorm.
- §7: no scalable global method was known. §8: the work is embarrassingly parallel.
- For us: our pipeline is their Alg. 2 at scale. We call the global step after every accepted step, which is our cost. Their exact g'' needs a full SVD, so our secant is the right choice for a full-rank channel.

**Bruinsma and Steinbuch, "A fast algorithm to compute the H∞-norm of a transfer function matrix", SCL 14:287-293, 1990 [R].**
- The Hamiltonian characterisation and the two-step, quadratically convergent level-set algorithm.
- A starting bound that includes σmax(D), i.e. ω → ∞. Remark 3.4 warns that inaccurate eigenvalues break the algorithm.
- For us: the ancestor of our certificate, and the reminder of coverage at infinity.

**Aliyev, Benner, Mengi, Schwerdtner and Voigt, "Large-scale computation of L∞-norms by a greedy subspace method", arXiv:1705.10086 [R]** (SIMAX 38(4), 2017, per a later citation [B]).
- Greedy two-sided Hermite interpolation for large descriptor and delay systems; local superlinear convergence (Thm 3.3); a lower bound with a documented miss (p. 20). It assumes n ≫ m, p.
- For us: the basis of the GL reduced oracle (STAGE5 §1-2). Not applicable to a full-field channel without losing its guarantees.

**Aliyev, Benner, Mengi and Voigt, "A subspace framework for H∞-norm minimization", arXiv:1905.04208 [R]** (SIMAX 41, 2020, per [B]).
- Reduced minimisation, then a full-model global maximiser, then subspace expansion.
- It assumes stability, a smooth optimum, few affine parameters and n ≫ m, p. Only Alg. 2.1 was run.
- The authors note that no large-scale method exists to compare with, and leave stability to future work.
- For us: implemented as the GL reduced loop (8x fewer LUs); it failed on the BFS, and each failure maps to one of its assumptions.

**Mengi, "A subspace framework for L∞ model reduction", arXiv:2309.08011, 2023 [R].**
- L∞ model reduction with a canonical tridiagonal/diagonal parametrisation (p. 5).
- Direct gradient descent or BFGS on the nonsmooth objective is slow (Table 2.1). The Hankel lower bound is used to call results "nearly optimal globally".
- Equal-ripple optima lie outside its quadratic theory. Stability constraints were not imposed in the experiments.
- For us: it supports the epigraph form and the lower-bound column we lack.

**Schwerdtner, Mengi and Voigt, "Certifying global optimality for the L∞-norm computation of large-scale descriptor systems" [R]** (preprint; IFAC-PapersOnLine 53(2), 2020, per search [S]).
- A decision procedure with eigenvalue-free disks on the even pencil.
- The symmetry-preserving K(ξ) (eq. 5) puts axis eigenvalues exactly on the axis. Intervals shorter than ε_eigs are removed; without the removal, eigs failed on 7 of 264 cases.
- 264 decisions, all correct; serial, up to q = 42,264.
- For us: our γ certificate is this algorithm, deployed in parallel and inside the solve, without the structure-preserving transform.

**Benner and Voigt, "A structured pseudospectral method for H∞-norm computation of large-scale descriptor systems", MCSS 26:303-338, 2014 [R].**
- ‖G‖∞ = 1/r_C, the structured complex stability radius (Prop. 3.2).
- Rank-one pseudopole iterations started from dominant poles, then Newton; 33 examples correct; a thin peak needs 40 starting poles.
- For us: the robust-stability reading of our γ, the dominance and perturbability measures, and the basis of the small-gain gate idea. Not an oracle.

**Mengi, "Large-scale estimation of dominant poles of a transfer function by an interpolatory framework", SISC 44(4):A2412-A2438, 2022 [R].**
- Two-sided interpolatory projection aimed at dominant poles, with at least quadratic local convergence.
- More dominant poles than SAMDP on 9 of 11 benchmarks. Started at dominant poles, a local L∞ search is accurate 99.43 % of the time, against 100 % for a level set.
- For us: not a cover, so it cannot certify. Its index-one theory does not cover our index-two pencil; at most a row supplier.

**Kiwiel, "Methods of Descent for Nondifferentiable Optimization", LNM 1133, Springer 1985 [R].**
- Lemma 1.2.5 (the Clarke set of a max over a compact set), the improvement function, F. John conditions and the CQ (1.2.65)-(1.2.66).
- Linearisation-error measures w, and the Goldstein ε-subdifferential with its constrained form M(x; ε) (Ch. 6 §7). The phase-I caveat (p. 296-297).
- Numerical results in Ch. 8, including a controller design on a finite frequency set with one top-eigenvector subgradient.
- For us: the theory behind our Clarke construction, principled stopping measures we lack, and null steps (enriching the model from rejected trials) that we do not use.

**Lukšan and Vlček, "A bundle-Newton method for nonsmooth unconstrained minimization", Math. Program. 83:373-391, 1998 [R].**
- The epigraph QP with multiplier-weighted per-piece Hessians (eq. 3), the stopping measure w_k, and Clarke stationarity of accumulation points (Thm 3.8).
- Superlinear only at a smooth minimiser. Finite-difference Hessians gave the same iteration counts (p. 375). About 4x fewer evaluations than a first-order bundle method at small N.
- For us: our QP structure. What transfers is per-piece curvature and w_k.

**Kantharaju, Courtier, Leclaire and Jacquin, "Interactions of large-scale structures in the near field of round jets at high Reynolds numbers", JFM 888, 2020 [R].**
- Experimental: PIV and hot wire, SPOD, open-loop acoustic forcing. No model, optimisation or feedback, and no method content for us.
- Likely a candidate next application [I]. A jet would change several things:
  - a mean-flow resolvent instead of a laminar base flow;
  - complex per-azimuthal-mode problems, where the ±m symmetry makes σ1 structurally double and puts the Clarke block at the centre;
  - inlet actuators and velocity sensors;
  - a large convective delay;
  - factor memory multiplied by the number of m.

---

## 7. References the user may have missed

### 7a. Verified (fetched and read)

- **MUMPS 5.3.5 Users' Guide**, MUMPS team, 2020: §3.12 (MPI plus OpenMP), §3.18 (BLR), §5.17 ICNTL(16), p. 24 JOB=2, ICNTL(7)=1/PERM_IN, NRHS. These are the levers in §4.
- **SLEPc EPSCISS source** (ciss.c l.862-864: partitions; l.1420-1445: EPSCISSGetKSPs), citing Maeda-Sakurai-Roman, SLEPc STR-11, 2016, and Sakurai-Sugiura, JCAM 159(1), 2003. It is installed (SLEPc 3.25.1); a candidate for splitting heavy disks.
- **Farghadan, Martini, Towne, RSVD-LU and RSVD-Δt READMEs** (the JCP 2024 paper, arXiv:2309.04617, was not opened). A PETSc/SLEPc/MUMPS implementation of randomised resolvent SVD.
- **NVIDIA cuDSS samples README** (0.7.0): batched complex sparse direct solves. Only after the LESSONS §10 timing test.
- **Read in the 2026-10-17 refresh [F]** (arXiv or HAL versions; section and page numbers are those of the versions read; published versions not compared). Previously cited, now read:
  - Apkarian-Noll-Ravanbod, SVVA 24(1):115-148, 2016 (arXiv:1504.00648v1, titled "Nonsmooth trust-region algorithm with applications to robust stability of uncertain systems"): Alg. 1 p. 6 (secondary test ρ̃; memory radius doubled at ρ ≥ Γ), Remark 2 p. 3, §4 p. 12-13 (γ = 1e-4, Γ = 0.1; stopping), §5.1-5.2 p. 13, Thm 2 p. 16, §5.7 p. 17-19 (Cauchy-point counterexample).
  - Apkarian-Noll-Ravanbod, SVVA 26(2):405-429, 2018 (arXiv:1804.11320v1): Alg. 2 p. 6 (0 ⪯ Q, ‖Q‖ ≤ q, updated at serious steps), Remark 4 eq. (8) p. 7 (the box QP is ours), Remark 8 p. 11, §4.2 p. 12 (anticipated cutting planes), Remark 9 p. 12, §5.3 p. 22 (tridiagonal A_K).
  - Benner-Mitchell-Overton, CDC 2018 (arXiv:1803.06549v1): §3 eqs. 5a-5c, §4 p. 6, Tables 1-2.
  - Kamgnia-Philippe, ETNA 40:1-16, 2013, read as INRIA RR-7770 v2 (arXiv:1110.4797v2): eq. (9) p. 7, §4.3, Ex. 3.2-3.3, §5 Tables 2-4.
  - Bindel-Hood, SIMAX 34(4):1728-1749, 2013 (arXiv:1303.4668v2): Lemma 2.1 and Remark 1 p. 2, Prop. 4.3, Thm 4.4.
  - Mitchell, Kreiss-constant and distance-to-uncontrollability certificates, SIMAX 2021 (arXiv:1910.01069v3): §1.3, Thm 2.7, Alg. 2.1, §2.3 p. 13-14.
  - Zhang-Lin-Jegelka-Sra-Jadbabaie, ICML 2020 (arXiv:2002.04130v3): Thm 5 p. 3, Def. 4 p. 4.
  - Burke-Curtis-Lewis-Overton-Simões, gradient-sampling survey (arXiv:1804.11003v1): Alg. GS p. 4, Thm 3.1 p. 5, §4.3-4.4 p. 9, §5.2 p. 10.
  - Drusvyatskiy-Lewis, MOR 2018 (arXiv:1602.06661v2): §5 (5.1) p. 13, Alg. 1 p. 14, Thm 5.3 p. 16, Thm 5.6 p. 18.
  - Lewis-Wright, Math. Program. 158:501-546, 2016 (arXiv:0812.0423): §3 p. 9, Alg. 2 p. 25, Thm 5.4 p. 27, Thm 5.5 p. 30.
  - Lewis-Wylie, Active-set Newton methods and partial smoothness, MOR 46(2), 2021 (venue not printed in arXiv:1902.00724v1): Thm 5.1 p. 9, Thm 5.5 p. 12 (skimmed).
  - Bareilles-Iutzeler-Malick, SIOPT 33(3):2222-2247, 2023 (arXiv:2206.15053v3): Ex. 2.3-2.4 p. 5, Thm 3.2 p. 9, Alg. 4.1 p. 17, Thm 4.5 p. 18, §5.2 p. 20.
  - Han-Lewis, SIOPT 33(1), 2023 (arXiv:2111.15645v5): Def. 3 p. 10, Thm 17 p. 16, Thm 19 p. 17.
  - Davis-Jiang, FoCM 2024 (arXiv:2205.00064v3): Thm 1.1 p. 15, Assumption A p. 23 (statements only).
  - Martini-Jung-Cavalieri-Jordan-Towne, JFM 937 A19, 2022 (arXiv:2201.00361v1): p. 10, §4.1.1 p. 22.
  - Jin-Illingworth-Sandberg, JFM 2022, doi 10.1017/jfm.2021.948, volume 932 not verified (arXiv:2103.16802v1): §2.2 p. 4, §3.3 p. 8-9, §4.
  - Manohar-Kutz-Brunton, IEEE TAC 67(4), 2022 (arXiv:1812.01574v2): §IV, §VI.
  - Benner-Heiland-Werner, COAP 82(1):225-249, 2022 (arXiv:2103.01608v2): §2.2 eqs. 9-10, Tables 1-2 p. 13.
  - Werner-Overton-Peherstorfer, SISC 45(2), 2023 (arXiv:2205.15050v2): Alg. 1, Table 1 p. 13, p. 15, p. 18-19, Table 6 p. 20.
  - Jussiau-Leclercq-Demourant-Apkarian, JFM 999 A86, 2024 (arXiv:2404.08487v1): a data-driven LQG paper, not structured H∞ (Table 1 p. 6, §2.5): cylinder at Re = 100, a 113,000-state FE model used for simulation only, a fixed sensor, no certificate. The group's structured H∞ flow paper is Jussiau et al., IEEE L-CSS 6:3212-3217, 2022 [B].
  - Gumussoy-Michiels, SICON 49, 2011 (arXiv:2003.09252v1): fixed-order strong H∞ for delay DAEs, plants of order 2-5 (Table 7.4); not a flow or co-design paper.
- **New, read in the 2026-10-17 refresh [F]:** Dao-Gwinner-Noll-Ovcharova, COAP 65(1), 2016 (arXiv:1401.6807, proximity-control Alg. 1 p. 8, §4.5, §8.2); Burke-Engle (arXiv:1806.05218v2, ratio trust region for convex composites, Alg. 4 p. 19, Thm 6.1 p. 20); Lewis-Wylie, A simple Newton method for local nonsmooth optimization (arXiv:1907.11742v1, Alg. 2.2 p. 4, Thm 5.11 p. 18); Curtis-Robinson-Zhou (arXiv:1708.02552v5, Thm 3.1 p. 10, partial); Kiessling-Leyffer-Vanaret, funnel SQP (arXiv:2409.09208v1, §2.2 p. 6); Vanaret-Leyffer, Uno (arXiv:2406.13454v2, §4.6.2, Tables 5-6); Luo-Wächter (arXiv:2207.03082v2, SOCP SQP with warm starts); Seidel-Küfer (arXiv:1910.13798v1, adaptive discretisation for SIP); Goulart-Chen, Clarabel (arXiv:2405.12762v1, §2.4, §2.6, §3.3); WARP (arXiv:2605.05728v1, Table 2); Asl-Overton (arXiv:2006.11336v1); the Ipopt 3.14 documentation sources; Schwerdtner-Voigt, SOBSYN (arXiv:2209.07558v2); NCVX/PyGRANSO (arXiv:2111.13984v2, arXiv:2210.00973v2); Drusvyatskiy-Paquette (arXiv:1605.00125v6); Mehlitz (arXiv:2006.09734); Royset (arXiv:2201.05250v2); Cartis-Gould-Toint (arXiv:2001.10802v1, arXiv:1705.07285v2); Burke-Eaton (arXiv:1511.03687v4); Peeters-Hanzon-Olivi, LAA 425, 2007 (arXiv:1012.3281); Henrion-Overton (arXiv:math/0603681); Apkarian-Noll, hyperbolic PDEs (arXiv:2211.08083); Apkarian-Noll, Aerospace Lab 13, 2017 (hal-01653161); Aliyev-Mengi (arXiv:2208.07540v4, §1); Monti (arXiv:2608.16401); Ait Rami-Henrion (arXiv:1006.5876); Henrion-Louembet (arXiv:1104.2679); Hilaire-Chevrel-Whidborne, IJC 83(2), 2010 (hal-01146512); Gillis-Sharma (arXiv:1611.00595, Lemma 1); Appeltans-Michiels (arXiv:2110.02665, §2.2); Mitchell, Kreiss constant (arXiv:1907.06537, §5.3, Key Remark 6.3); Di Napoli-Polizzi-Saad (arXiv:1308.4275v2); Batra-Sharma (arXiv:1912.02820, p. 10-11); Nibourel-Leclercq-Demourant-Garnier-Sipp, JFM 954 A20, 2023 (arXiv:2206.06224v1); Edalatzadeh-Morris (arXiv:1802.05807, p. 1); Dadfar-Fabbiane-Bagheri-Henningson (arXiv:1402.0769, partial); Deshpande-Bhattacharya (arXiv:2409.09596, abstract and introduction); Fendl-Schichl II (arXiv:1506.08021, set-up only).
- **Abstract or record page only [A]:** Barbagallo et al. 2012 (hal-00996503); Hervé et al. 2012 (hal-00994468); Juillet-Schmid-Huerre 2013 (hal-00995144); Noll-Prot-Rondepierre 2008 (hal-00464695); Philippe et al. 2013 (hal-00903743).

### 7b. Reference found in an archive paper's bibliography (content not read)

- **Bompart, Noll, Apkarian, "Second-order nonsmooth optimization for H∞ synthesis", Numer. Math. 107(3):433-454, 2007** (cited by AN 2006 as [51]). Read it before any second-order change. Bibliography confirmed by HAL record hal-01868767 (doi 10.1007/s00211-007-0095-9), which has no file; Noll's copy and Springer were refused on 2026-10-17, so it needs the user's copy.
- **Noll, Apkarian, "Spectral bundle methods for non-convex maximum eigenvalue functions: second-order methods", Math. Program. B 104:729-747, 2005** (AN 2006 [38]), doi 10.1007/s10107-005-0635-y. Identify the multiplicity, then Newton; relevant for k ≥ 3 [I]. The identification is a heuristic and the step an SQP iteration (Bareilles et al. 2023 p. 3 [F]).
- **Polak, "Optimization: Algorithms and Consistent Approximations", Springer 1997** (AN 2006 [36]; also Polak, SIAM Rev. 1987, [35]). Optimality functions and consistent approximations.
- **Oustry, "A second-order bundle method to minimize the maximum eigenvalue function", Math. Program. 89:1-33, 2000** (AN 2006 [25]), doi 10.1007/PL00011388 (INRIA RR-3738, 1999). Per its abstract (snippet [S]): quadratic convergence needs strict complementarity and nondegeneracy; convex affine setting.
- **Burke, Lewis, Overton, gradient sampling, SIOPT 15(3):751-779, 2005** (AN 2006 [32]). Described in the gradient-sampling survey [F] (Alg. GS p. 4; its density assumption corrected, p. 5 fn. 3).
- **Boyd, Balakrishnan, Kabamba, MCSS 2(3):207-219, 1989** (AN 2006 [10]). The level-set bisection, itself inspired by Byers 1988 (Benner-Mitchell 2018 §2 [R]); the statement our certificate uses is its descriptor even-pencil form (SMV eq. 2, after Benner-Sima-Voigt 2012 [R]).
- **Mehrmann, Watkins, SISC 22(6), 2001 (SHIRA)**, and **Mehrmann, Schröder, Simoncini, LAA 436(10), 2012 (EVEN_IRA)** (SMV references). Structure-preserving even-pencil Krylov; it is EVEN_IRA (2012), not SHIRA, that handles a singular N (SMV p. 5 [R]); SISC 22(6):1905-1925 and LAA 436(10):4070-4087. Two shifted solves per application and J-orthogonalisation (Appeltans-Michiels 2021 §2.2 [F]).
- **Benner, Sima, Voigt, IEEE TAC 57(1):233-238, 2012** (SMV references; 1905.11511 [30]). A dense structured reference for the mini-mesh audit.
- **Mitchell, Overton, HIFOOS, ROCOND 2015** (1705.10086 [30]). The main earlier large-scale fixed-order competitor; uncertified norms.
- **Mitchell, Overton, HEC, IMA J. Numer. Anal. 36(3):985-1014, 2016** (17m1137966 [28]), doi 10.1093/imanum/drv046. Inside HIFOOS its approximations were inconsistent between nearby controllers and stopped line searches early (BMO 2018 §2 p. 3 [F]).
- **Apkarian, Noll, boundary control of PDEs, arXiv:1905.06786 / SCL 2020** (1905.11511 [44]).
- **Gahinet, Apkarian, "Structured H∞ synthesis in MATLAB", IFAC 2011, pp. 1435-1440** (1905.11511 [33]). Benchmark values for validating the controller-only path, if it holds any: no open copy was found on 2026-10-17.
- **Freitag, Spence, Van Dooren, SIMAX 35(2):619-635, 2014** (17m1137966 references). Implicit-determinant Newton in (ω, γ) per its abstract (snippet [S]): a two-parameter eigenvalue problem whose critical point is typically a 2-dimensional Jordan block; local only (SMV p. 2 [R]).

### 7c. Search snippets only: bibliographic data from search results, content not read (unverified; entries marked [B] or [A] had their data checked in a read paper's bibliography or on a record page on 2026-10-17)

*Second order and stationarity:*
- Hettich-Kortanek, SIAM Rev. 35(3):380-429, 1993, doi 10.1137/1035089;
- Overton, SIMAX 9(2):256-268, 1988;
- Yuan, "Conditions for convergence of a trust-region method for nonsmooth optimization", Math. Program. 31:220-228, 1985, and "On the superlinear convergence of a trust region algorithm for nonsmooth optimization", Math. Program. 31:269-285, 1985 [B] (Lewis-Wright refs [52], [53]; the second also Davis-Jiang ref [54]); a Yuan paper in IMA J. Numer. Anal. 4:327-335, 1984 (a file name in a snippet; title unconfirmed); Fletcher, LNM 912, 1982 (unverified). Yuan's Cambridge report "Global convergence of trust region algorithms for nonsmooth optimization" (Burke-Engle ref [22]) is a separate item;
- Cartis-Gould-Toint, SIOPT 21(4):1721-1739, 2011 [B] (for our use superseded by CGT arXiv:2001.10802, now read);
- Hare-Sagastizábal-Solodov, COAP 63(1):1-28, 2016, doi 10.1007/s10589-015-9762-4 [B];
- Lewis, SIOPT 13(3):702-725, 2002 [B] (partial smoothness);
- Han, Math. Program. 20, 1981;
- Bonnans-Shapiro 2000 (second-order conditions; content not verified).

*Certificates and counts:*
- Benner-Lowe-Voigt (R. Lowe, no umlaut), "L∞-norm computation for large-scale descriptor systems using structured iterative eigensolvers", NACO 8(1):119-133, 2018, doi 10.3934/naco.2018007 (preprint MPIMD/13-20, 2013);
- Ying-Katz, Numer. Math. 53(1-2):143-163, 1988 [B] (secondary account: Batra-Sharma 2019 p. 10);
- Johnson-Tucker, JCAM 228(1):418-423, 2009 [B] (secondary account: Batra-Sharma 2019 p. 11);
- Yin, J. Sci. Comput. 78, 2019;
- Meerbergen-Spence, Math. Comp. 66(218):667-689, 1997 (implicitly restarted Arnoldi in the B semi-inner product with purification, for a semidefinite B such as our singular E_H; abstract per snippet);
- Grivet-Talocia-Ubolli, IEEE TADVP 29(1), 2006;
- Gobbato-Chinea-Grivet-Talocia, DATE 2011;
- Rump, Acta Numerica 19:287-449, 2010;

*Lower bounds and convex benchmarks:*
- Davis-Kahan-Weinberger, SIAM J. Numer. Anal. 19:445-469, 1982 (Parrott-type completion);
- Boyd-Barratt 1991; Boyd-Barratt-Norman, Proc. IEEE 78(3), 1990;
- Lessard-Lall, IEEE TAC 61(10), 2016;
- Mengi-Yildirim-Kilic, SIMAX 35(2), 2014;
- Henrion-Lasserre, IEEE TAC 51(2), 2006;

*Co-design and flow control:*
- Kasinathan-Morris, IEEE TAC 58:2522-2535, 2013 [B] (issue number not confirmed);
- Morris, IEEE TAC 56:113-124, 2011 [B], and JMSEC 4(2), 1994 (unverified);
- Chen-Rowley, JFM 681:241-260, 2011 [B];
- Oehler-Illingworth, JFM 854:34-55, 2018 [B];
- Barbagallo-Dergham-Sipp-Schmid-Robinet, JFM 703:326-362, 2012, doi 10.1017/jfm.2012.223 (the physical baseline) [A] (HAL hal-00996503 abstract; full text not accessible);
- Dergham-Sipp-Robinet, JFM 719:406-430, 2013 [B];
- Hervé et al., JFM 702:26-58, 2012 [A] (ARMAX identification on the BFS);
- Barbagallo-Sipp-Schmid, JFM 641:1-50, 2009 [B];
- Sipp-Schmid, Appl. Mech. Rev. 68(2):020801, 2016 [B];
- Belson et al., Phys. Fluids 25:054106, 2013 [B];
- Leclercq-Demourant-Poussot-Vassal-Sipp, JFM 868, 2019 (pp. 26-65 per Jussiau 2024, 22-65 per Nibourel 2023) [B]: multi-objective structured H∞ synthesis on the mean-flow linearisation with full-order stacking, per Jussiau 2024 Table 1, p. 6 and p. 30, and Nibourel 2023 p. 3 [F]; model size, tool, certification and any placement unknown;
- TDS-CONTROL 2022;
- Karimi-Kammer, Automatica 85, 2017;
- Hast et al., ECC 2013; Boyd-Hast-Åström, IJRNC 2016;
- Keel-Bhattacharyya, IEEE TAC 42(8), 1997;
- Zare et al. (arXiv:1807.01739); Münz-Pfister-Wolfrum, IEEE TAC 59(11), 2014.

*Parallel and linear algebra:*
- Berljafa-Güttel, SISC 39(5), 2017;
- Güttel-Polizzi-Tang-Viaud, SISC 37(4), 2015;
- Polizzi FEAST v4 (arXiv:2002.04807) and PFEAST, SC16;
- Lin-García-Huhs-Yang (PEXSI), JPCM 26, 2014;
- Mounié-Rapine-Trystram, SICOMP 37(2), 2007;
- Dean-Barroso, CACM 56(2), 2013;
- Graham, SIAP 17(2), 1969;
- Amestoy-Buttari-Higham-L'Excellent-Mary-Vieublé, ACM TOMS 49(1), 2023;
- Ribeiro-Yeh-Taira, PRF 5, 2020;
- Soodhalter-de Sturler-Kilmer, GAMM 2020;
- Byrd-Schnabel-Shultz, Math. Program. 42(2):273-306, 1988 (parallel quasi-Newton with speculative evaluations, not a trust-region paper);
- Iutzeler-Malick-de Oliveira, Math. Program. 184, 2020;
- Elman-Meerbergen-Spence-Wu, SISC 34(3), 2012;
- Kressner-Vandereycken, SIMAX 35(1), 2014;
- Curtis-Overton, SIOPT 22(2):474-500, 2012 [B] (SQP-GS; its convergence result needs upper and lower bounds on the BFGS eigenvalues, CMO 2017 p. 3 [R]);
- Noll-Prot-Rondepierre, Pacific J. Optim. 4(3):571-604, 2008 [A] (HAL hal-00464695, ANR 2016 ref [33], 1905.11511 ref [23], TAC 2015 ref [21]; DGNO 2016 and IJRNC 2018 print 569-602); full text on a refused portal; the τ management is stated in full in Dao et al. 2016 Alg. 1 [F];
- Apkarian-Noll-Prot, SIOPT 19(1):281-306, 2008, doi 10.1137/060665191 [B] (1905.11511 ref [27]);
- Mifflin-Sagastizábal, Math. Program. 104(2-3):583-608, 2005 [B] (convex VU; the superlinear rate hinges on identification, Bareilles et al. 2023 p. 2 [F]);
- Sagastizábal, Math. Program. 140, 2013;

### 7d. Added in the 2026-10-17 refresh, not read (bibliography [B], abstract [A] or snippet [S])

- Jussiau-Leclercq-Demourant-Apkarian, IEEE L-CSS 6:3212-3217, 2022 [B] (Jussiau 2024 ref list): the ONERA group's structured H∞ flow-control paper; read before claiming novelty against structured H∞ in flows;
- Juillet-Schmid-Huerre, JFM 725:522-565, 2013 [A] (hal-00995144): a second BFS baseline (Re = 350, feed-forward, sensor "visibility length");
- Womersley-Fletcher, JOTA 48:493-523, 1986 [B] and Fletcher-Sainz de la Maza, Math. Program. 43:235-256, 1989 [B]: active-set second-order methods for max-of-smooth and ℓ∞ trust-region LP plus EQP (Bareilles et al. p. 3; Lewis-Wright §3);
- Zhou-Tits, SIOPT 6(2):461-487, 1996 [S]: SQP for finely discretised minimax with a small working set updated each iteration, the closest classical analogue of our Ω policy;
- McKelvey-Helmersson-Ribarits, Automatica 40(9):1629-1635, 2004 (DDLC) [S]; Wills-Ninness-Gibson, IEEE TAC 53(1):298-306, 2008 [S]; McKelvey-Helmersson, CDC 1996 (tridiagonal forms) [B]; Hanzon-Ober, LAA 281, 1998 [B];
- Grivet-Talocia, IEEE TADVP 30(2):226-237, 2007; Kressner, CACSD 2006; Chiprout-Nakhla 1995 [B] (SMV p. 3: the precursors of the eigenvalue-free-disk scan);
- Philippe et al., 2013 (hal-00903743) [A] and Nguenang-Kamgnia-Philippe, ARIMA 19, 2015 [S]: argument-principle counting and σ_min path following from one LU per point;
- Lewis-Pang, SIOPT 19, 2008 and Gürbüzbalaban-Overton, SIOPT 22, 2012 [B] (Aliyev-Mengi §1): local Lipschitz continuity of the pseudospectral abscissa;
- Wächter-Biegler, Math. Program. 106(1):25-57, 2006 [S]; Gill-Murray-Saunders, SNOPT, SIAM Rev. 47(1):99-131, 2005 [S]; Fletcher-Leyffer, Math. Program. 91:239-269, 2002 [B];
- Charisopoulos-Davis (arXiv:2201.04611) [A]: needs sharp growth, which our records lack; not applicable.

---

## Appendix A. Corrections applied to the draft

- **20261010n veto.** All five decentral final designs were vetoed by the winding tail bound; the certified decentral values are 369.87 and 412.28, not 366-368. "Clearly ahead at scale" and "no published analogue at n = 191,720" are toned down: the full mesh was never run, and our γ size is 1.6x SMV's.
- **AN 2006 nesting.** Variant II picks a new Ω_k at each iterate and nests only at a fixed iterate. Never retiring Ω rests on the project's A/Bs.
- **TAC 2015.** Newton is not prescribed, and the design-level objective is lower-C1. TAC 2015's stop is relative step and decrease, not a criticality test.
- **IJRNC Remark 6** serves the Nyquist count. IJRNC's γ is certified on a finite grid with a finite-difference L.
- **AN 2007 Table 1** concerns the Model I homotopy solved directly, not a barrier.
- **Slivers.** SMV also drop slivers, and the round certificate (2n pencil) is now kept distinct from the in-solve level set.
- **Parallel figures** are updated to 48 ranks. The level-set "60 s a round" is foreground only; join waits are reported.
- **σ2 enrolment** is live on central (8-14 frequencies); "13 tops" is 12 tops plus an edge stop; ρ is also normalised by min ‖g‖ (1.2e-3 central, 1.6e-3 decentral).
- **Central band** is [0, 24.92].
- **pyGRANSO evidence** is GL-only and within noise.
- **The GL 4x** came from the blocked _orth, not from multi-RHS solves.
- **Kantharaju** dropped as a source for the doubled-plant test.
- **The modal "5 of 6"** is confounded, and the CQ failure at M = L·I is noted.
- **Floor (E):** the 1e-7 to 1e-6 figure applied to the superseded bracket-and-refine path.
- **"(ε, δ)-stationarity"** claim removed.
- **SISO headline** restated at ‖d‖∞ = 1e-6, inside Δ_set, with the fourth-top correction at 1e-5.
- **The tool cannot license a stop:** GL history plus decentral 369.87 then −0.9 %.
- **θ and pred_inf** are not equivalent; HIFOO's pair is a heuristic.
- **Ranking:** the tail bound and 8 rounds for MIMO come first; the "certify" branch needs a gate; completeness is expensive on plateaus; the coincident sensors are noted; busy-polling and oversubscription are added; `--wind-accept off` contradicts a LESSONS rule; STATUS.md and BFS.md wording on SISO is to be corrected.
- **Ideas:** linear-algebra payoff lowered; async-count payoff raised; pole-targeted rows risk raised; full-mesh work conditional on the tail fix; perturbations sized per group.
- **Open-loop floor:** port-independent and rigorous by sampling, but weak (≥ 32.66 / ≥ 25.00).
- **Gauge** finding added (rechecked: cond 21.6 for 598.11 block (2,3)); NZ counts include 10 gauge directions.
- **New ideas added:** Parrott per-port bounds and position maps; the Youla benchmark; time-domain, mesh and domain checks; the small-gain gate; constraint prices; robustness reporting; the drift predictor.
- **Formulation and baseline issues** stated.
- **Search-only papers** are no longer cited for content (Hare et al., Cartis et al.). l.147 and l.167 still did until the 2026-10-17 refresh, which replaced CGT 2011 by CGT arXiv:2001.10802 [F] and reduced Hare et al. to a title.

## Appendix B. Critic points rejected or modified

- **"A wrong on-axis call changes crossing detection and cost, not disk soundness" (adversarial). Rejected as stated.** A disk's radius is |λ − iθ|, which is at least |Re λ|, so the cleared axis interval always contains Im λ. A true crossing computed with |Re λ| above the 1e-8 tolerance is therefore lost, either inside the disk or in a dropped sliver (certify_sparse.py l.349-378). Accepted parts: it is not the only tolerance-dependent step, and the sliver drop is SMV's own practice.
- **"The sliver drop is quantitatively harmless, about 1e-10" (a reader). Rejected for the combined case.** The bound holds only if the region above the level is itself a sliver. When both crossings of a wide region are lost, nothing samples σ at the shifts in between: only the start and the near-axis eigenvalues are sampled (l.255-262, l.349-355).
- **"TAC 2015 misattribution" (adversarial). Modified.** Accepted that Newton is not prescribed and that the design objective is lower-C1. Kept the narrower point: our ω-search at a fixed design is the upper-C1 min-min case that §III says may be run like a smooth problem on an active branch.
- **"The old tail gives W_T = 1.97e6 against 7680" for 369.87 (adversarial). Number corrected.** For 369.87 it is 7.86e6 against 7,680; 1.97e6 against 3,840 is for 412.28 [E2].
- **"Youla appears in BFS.md and STATUS.md in other contexts" (completeness critic). Minor correction.** A case-insensitive grep finds no "Youla" in LESSONS, STAGE*, BFS, STATUS or TRACKER. The conclusion that it was never tried is unchanged.
- **"The model is valid only within about 1e-5, while the SQP's radius is about 1e-2" (completeness critic, SISO). Modified.** The SISO record stores no trust radius (sqp_trust None), and the 1e-5 radius is measured along the H = 0 box direction, which moves every coordinate by Δ, not along the SQP's curvature-shaped step. The scaling inference stands; the direct comparison is dropped.
- **"Linear-algebra levers: payoff probably low" (adversarial). Mostly accepted, with one exception.** Analysis reuse stays "unknown until measured", because c151 no strip performed 156,225 off-lattice exact-point factorisations (79,104 s of factorisation work) [E1].

## Evidence files

- [E1] Batch logs:
  - results:20261010n/logs/bfs_testplant_774361_pr_d370_nostrip.log
  - results:20261010n/logs/bfs_testplant_774361_pr_d370_strip.log
  - results:20261010n/logs/bfs_testplant_774363_pr_d412_nostrip.log
  - results:20261010n/logs/bfs_testplant_774363_pr_d412_strip.log
  - results:20261010n/logs/bfs_testplant_774364_pr_d370_nostrip_async.log
  - results:20261010n/logs/bfs_testplant_774364_pr_c151_nostrip_async.log
  - results:20261010n/logs/bfs_testplant_774360_pr_c151_nostrip.log
  - results:20261010n/logs/bfs_testplant_774362_pr_c177_nostrip.log
- [E2] (session scratch) wf_fix/tail_probe_new.log
- [E3] (session scratch) wf_stat/smoke_siso598.log
- [E4] (session scratch) wf_stat/smoke_c151.log
- [E5] (session scratch) wf_stat/smoke_d370.log
- [E6] Descent probes:
  - (session scratch) wf_stat/descent_siso598.log
  - (session scratch) wf_stat/descent_c151.log
  - (session scratch) wf_stat/descent_d370.log
- [E7] results:20261010n/runs/testplant_bfs_pr_c151_nostrip/post.json
- [E8] results:20261010n/runs/testplant_bfs_pr_c151_nostrip_async/K_B_C_s0_bfs_mimo2x2_central.json
- Paper texts: (session scratch) biblio/txt/
- Stationarity tool: /home/user/codesign/tools/bfs_stationarity.py