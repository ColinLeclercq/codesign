# Literature refresh (network access, 2026-10-17)

This is an addendum to /home/user/codesign/LITERATURE.md at HEAD 15cdfcb. It merges six reading groups and a verifier pass that re-checked every proposed correction and impact claim against the downloaded texts and the code:
- A, second-order methods and trust regions;
- B, compensator parametrisation;
- C, γ and winding certificates;
- D, stationarity;
- E, flow control and co-design;
- F, solver libraries.

Its edits to LITERATURE.md are applied there (2026-10-17). The per-group reading notes are in lit/refresh_2026-10-17/notes/<group>.md and the small numerical checks in tools/lit_refresh_2026-10-17/; the downloaded texts were kept out of the repository (copyright), and the notes say where each was fetched.

**Tags.** These follow LITERATURE.md:
- **[R]** read in the user's archive.
- **[F]** full text fetched and read in this refresh. The brief calls these "[R]"; the verifier required [F], because LITERATURE.md reserves [R] for the archive.
- **[A]** abstract or record page only. This tag is new (EDITS E2).
- **[B]** reference found in the bibliography of a paper that was read.
- **[S]** search snippet only.
- **[C]** project code or notes.
- **[I]** our inference.
- "Secondary" means the paper is described in a fetched full text, but the original was not read.

Page numbers are PDF pages of the version read (arXiv unless stated). Published versions were not compared.

---

## 1. Outcome

- **Read.** 62 items were read in full or in the relevant part [F]: papers, the Ipopt documentation sources and package metadata. Six more were read as abstracts or record pages [A]. Of these, 21 full texts and 3 abstracts are references LITERATURE.md had tagged [S] or [B] (§3). Only arXiv, HAL record pages and package metadata were reachable.
  - Still unread: Bompart-Noll-Apkarian 2007, the paper LITERATURE.md asks to read before any second-order change. Also unread: Leclercq et al. 2019, Jussiau et al. 2022 (L-CSS) and Barbagallo et al. 2012 beyond its abstract (§8).
- **Corrections.** 25 corrections of LITERATURE.md went to the verifier. It upheld 22 and weakened 3; the weakened ones are entered in their weakened form (§4.1).
  - The largest: BMO 2018 does impose a full-order stability constraint (l.33).
  - Jussiau 2024 is an LQG paper (l.657), while Leclercq 2019 did use structured H∞ (l.656).
  - AN 2006 stop on step length alone, and their θ is unconstrained (l.146, l.183).
  - The γ-certificate threshold text is wrong in its details (l.91).
  - Kamgnia-Philippe's own step control is heuristic (l.94, l.388).
  - The case against bundle and GS solvers rests on iteration count and structure, not on evaluations per iteration (l.34, l.433).
  - Both premises of "do not add another inner solver" are gone (l.439).
- **(a) Normal form.** This is the best-supported item in the refresh. It is the controllable companion form under a fixed similarity. Its compensator rows are exactly linear and need no constraint qualification, and it removes the gauge that every local fast-rate theorem excludes. The literature tools (systune/tunableSS, the Apkarian-Noll papers, Nibourel 2023) keep a tridiagonal form with a gauge and use abscissa-type controller constraints. Four cautions apply (§5.1):
  - keep the rows elastic;
  - blocks sharing a root are non-identifiable;
  - the pivot chart is only local in central MIMO;
  - read admissibility from the rows, not from eig.
- **(b) Ratio rule.** It is supported by the ℓ∞-box SQP literature and by Burke-Engle for our convex-composite class. The literature contradicts tying the radius to 2α|d| after an accepted step: no source read does it, and the project rejected it three times under earlier defects. The Apkarian-Noll methods shrink only when the failure persists with the trial's own pieces in the model. Two additions are needed (§5.2):
  - a ρ̃-type gate on the shrink legs;
  - a radius-free exit, because the "pred" exit becomes radius-dependent under the new rule.
- **(c) Paired Arnoldi tiers.** No paper read addresses them. The design is consistent with the principle of never letting a decision hinge on one fragile eigensolve. Bitwise identity requires keeping "tier 0 whenever it converges", whatever the finish order (§5.3).
- **Stationarity tool.** Report the prox-linear step ‖G_t‖ (equal to −θ and Kiwiel's w up to scaling, one quantity) and pred(Δ)/Δ, which is the Cartis-Gould-Toint measure. Relabel the "Polak with a box" LP and demote the ε = 1e-6 "Clarke reading" (§5.4).
- **Ranked ideas.** Ideas 1 and 2 are done at HEAD and idea 3 is superseded by (a). After (a) and (b), the γ-certificate audit (idea 6) and the stationarity measures (ideas 4 and 14) move up; robustness reporting (idea 20) moves up. Ideas 5 and 17 keep their rank with changed content (§6).
- **IPOPT and other libraries.** Do not adopt one as the inner solver. IPOPT is now installable, so at most one bounded GL diagnostic is worth running; buy no licences (§7).

---

## 2. Access

**Reachable:**
- arxiv.org (abs, pdf);
- hal.science record pages (files only when the record is not on a collection portal);
- doi.org redirects;
- raw.githubusercontent.com, pypi.org and conda.anaconda.org (package metadata and documentation sources).

**Refused (403 at CONNECT or DNS failure):**
- publishers: Springer non-OA (idp.springer.com), SIAM, IEEE, Elsevier, Cambridge, OUP, AMS, AIMS and ETNA;
- HAL collection portals (unilim, inria, polytechnique, pastel), theses.hal.science, api.archives-ouvertes.fr and the HAL search page (robot block);
- author and preprint hosts: math.univ-toulouse.fr (Noll's papers), cs.nyu.edu, people.orie.cornell.edu, numerical.rl.ac.uk, optimization-online.org, apps.dtic.mil, export.arxiv.org;
- documentation sites: the MathWorks documentation and its mirrors, coin-or.github.io, docs.mosek.com.

WebFetch had no DNS. Nothing from the downloads was executed and nothing was installed.

---

## 3. Newly read

### A. Second order and trust regions
- **Apkarian-Noll-Ravanbod 2016** [F] (SVVA 24(1):115-148, read as arXiv:1504.00648v1):
  - Alg. 1 p. 6: the inner radius starts from a memory radius. A null step adds a cut. R is halved only if ρ̃ ≥ γ̃, and the memory radius doubles after a serious step with ρ ≥ Γ.
  - Remark 2 p. 3 and §5.1-5.2 p. 13: with the strict natural model the method is the classical trust region.
  - Thm 2 p. 16: the classical method converges for upper-C1 functions.
  - §5.7 p. 17-19: the Cauchy point fails on a convex piecewise-affine function in R².
  - §4 p. 12-13: three stopping tests together; γ = 1e-4, γ̃ = 2e-4, Γ = 0.1.
- **Apkarian-Noll-Ravanbod 2018** [F] (SVVA 26(2):405-429, read as arXiv:1804.11320v1):
  - Alg. 2 p. 6: 0 ⪯ Q with ‖Q‖ ≤ q, updated at serious steps only.
  - Remark 4, eq. (8) p. 7: with a box, the tangent program is exactly our epigraph QP.
  - Remark 8 p. 11: reducing the radius at every failure "must fail".
  - §4.2 p. 12: anticipated cutting planes. Remark 9 p. 12: backtracking trial points are allowed.
  - §5.3 p. 22: a tridiagonal A_K with x ∈ R¹⁴.
- **Dao-Gwinner-Noll-Ovcharova 2016** [F] (arXiv:1401.6807): Alg. 1 p. 8, the open statement of the proximity-control τ rules; §4.5 p. 6-7, when to enrich the model rather than shorten the step; §8.2, Q built from the Hessians of the attaining pieces, unweighted.
- **Burke-Engle** [F] (arXiv:1806.05218v2): Alg. 4 p. 19, a ratio trust region for convex composites that accepts at r ≥ β1 and shrinks whenever r < β2; Thm 6.1 p. 20, convergence with bounded H_k.
- **Lewis-Wright 2016** [F] (arXiv:0812.0423): ProxDescent, Alg. 2 p. 25 (correction step bounded by |d|/2; μ ← μ/τ after every acceptance); Thm 5.4 p. 27; identification, Thm 5.5 p. 30; §3 p. 9.
- **Lewis-Wylie 2019, "A simple Newton method for local nonsmooth optimization"** [F] (arXiv:1907.11742v1): the k-bundle Newton method, Alg. 2.2 p. 4; Thm 5.11 p. 18, which needs strongly convex pieces and k given.
- **Lewis-Wylie 2021** [F] (arXiv:1902.00724v1; venue not printed): Thm 5.1 p. 9 and Thm 5.5 p. 12, skimmed.
- **Bareilles-Iutzeler-Malick 2023** [F] (arXiv:2206.15053v3): Ex. 2.3-2.4 p. 5; Thm 3.2 p. 9; Alg. 4.1 p. 17; Thm 4.5 and Assumptions 4.2-4.4 p. 18; §5.2 p. 20; secondary accounts of Womersley-Fletcher, Mifflin-Sagastizábal and Noll-Apkarian 2005 on p. 2-3.
- **Han-Lewis 2023** [F] (arXiv:2111.15645v5): Def. 3 p. 10, Thm 17 p. 16, Thm 19 p. 17.
- **Davis-Jiang 2024** [F] (arXiv:2205.00064v3): Thm 1.1 p. 15 and Assumption A p. 23, statements only.
- **Curtis-Robinson-Zhou** [F, partial] (arXiv:1708.02552v5): Thm 3.1 p. 10.
- **Noll-Prot-Rondepierre 2008** [A]: HAL record hal-00464695.

### B. Parametrisation
- **Burke-Eaton** [F] (arXiv:1511.03687v4) §1 p. 1-3: the matrix abscissa is Clarke-regular iff the active eigenvalues are nonderogatory; polynomial root-max functions are regular.
- **Peeters-Hanzon-Olivi 2007** [F] (arXiv:1012.3281) §1 and §3 p. 5: pivot-structure charts are global for SISO and local for MIMO (a finite atlas).
- **Henrion-Overton 2006** [F] (arXiv:math/0603681): Thm in §5, a sharp minimiser with all roots coalesced; §6 p. 8, rounding sensitivity.
- **Apkarian-Noll 2022** [F] (arXiv:2211.08083): §2.3 p. 4; §5.1 p. 13, controller radius and damping bounds; §5.2 p. 14, 54 tridiagonal parameters.
- **Apkarian-Noll 2017** [F] (Aerospace Lab 13, hal-01653161): p. AL13-01 4, the full parametrisation; Fig. 3 p. 5, proximity control; p. 6, Lyapunov variables give ill-conditioning.
- **Aliyev-Mengi** [F] (arXiv:2208.07540v4) §1 p. 1-2.
- **Monti** [F] (arXiv:2608.16401): §5.4 eq. (39) p. 21; Supp. §S2.2.
- **Ait Rami-Henrion** [F] (arXiv:1006.5876) §4.1 p. 6.
- **Henrion-Louembet** [F] (arXiv:1104.2679) §1.
- **Hilaire-Chevrel-Whidborne 2010** [F] (hal-01146512): Table 1 p. 12.
- **Schwerdtner-Voigt 2022** [F] (arXiv:2209.07558v2): Lemma 1 §5; eq. (10) p. 5; Alg. 1 p. 6.
- **Gillis-Sharma** [F] (arXiv:1611.00595): abstract and Lemma 1.

### C. Certificates
- **Kamgnia-Philippe** [F], read as INRIA RR-7770 v2 (arXiv:1110.4797v2): Condition (B), eq. (9) p. 7; §4.3, "a heuristic procedure"; Ex. 3.2-3.3; §5, Tables 2-4.
- **Bindel-Hood 2013** [F] (arXiv:1303.4668v2): Lemma 2.1 and Remark 1 p. 2; Prop. 4.3; Thm 4.4.
- **Benner-Mitchell-Overton 2018** [F] (arXiv:1803.06549v1): §3 eqs. 5a-5c; §4 p. 6; Tables 1-2.
- **Mitchell 2021** [F] (arXiv:1910.01069v3): §1.3; Thm 2.7; Alg. 2.1; §2.3 p. 13-14.
- **Mitchell 2020** [F] (arXiv:1907.06537): §5.3 p. 14; Key Remark 6.3 p. 17.
- **Appeltans-Michiels** [F] (arXiv:2110.02665) §2.2 p. 5-7, a secondary account of SHIRA.
- **Di Napoli-Polizzi-Saad** [F] (arXiv:1308.4275v2): Lemma 2.1 p. 4; §5.7; §6.
- **Batra-Sharma** [F] (arXiv:1912.02820) p. 10-11, a secondary account of Ying-Katz and Johnson-Tucker.
- **Philippe et al. 2013** [A]: hal-00903743.

### D. Stationarity
- **Zhang et al. 2020** [F] (arXiv:2002.04130v3): Thm 5 p. 3; Def. 4 p. 4.
- **Gradient-sampling survey** [F] (arXiv:1804.11003v1): Alg. GS p. 4; Thm 3.1 p. 5 with fn. 3; §4.3-4.4 p. 9.
- **Drusvyatskiy-Lewis 2018** [F] (arXiv:1602.06661v2): §5 (5.1) p. 13; Alg. 1 p. 14; Thm 5.3 p. 16; Thm 5.6 (5.7) p. 18.
- **Drusvyatskiy-Paquette 2019** [F] (arXiv:1605.00125v6): Ex. 3.3 p. 7; p. 11 with fn. 5; Thm 4.5 and (4.9) p. 14-15.
- **Cartis-Gould-Toint** [F]: arXiv:2001.10802v1, (2.4) p. 3, (3.11) p. 6, Lemma 3.2, (4.1) p. 8; and arXiv:1705.07285v2, fn. 3 p. 10.
- **Mehlitz 2020** [F] (arXiv:2006.09734): Def. 3.1 and Thm 3.2 p. 7; Lemma 3.4 p. 9; Thm 3.10 p. 12.
- **Royset 2023** [F] (arXiv:2201.05250v2): p. 3; Def. 2.2 p. 7; Prop. 2.3 p. 8; Ex. 3.3 p. 15.
- Archive page images were re-read for AN 2006 p. 79 (eq. 19 is unconstrained) and p. 82 (the stopping rules), and the Kiwiel Ch. 2 text for (3.3).

### E. Flow control and co-design
- **Jussiau et al. 2024** [F] (arXiv:2404.08487v1): abstract; Table 1 p. 6; §2.3-2.6; p. 30.
- **Nibourel et al. 2023** [F] (arXiv:2206.06224v1): p. 2-4; §5.1-5.2 p. 18-21; §6.2; App. A p. 30-31.
- **Werner-Overton-Peherstorfer 2023** [F] (arXiv:2205.15050v2): Alg. 1 and 3; Table 1 p. 13; p. 15; p. 18-19; Table 6 p. 20; App. B p. 23.
- **Benner-Heiland-Werner 2022** [F] (arXiv:2103.01608v2): §2.2 eqs. 9-10; Tables 1-2 p. 13; Fig. 4.
- **Jin-Illingworth-Sandberg 2022** [F] (arXiv:2103.16802v1): §2.2 p. 4; §3.3 p. 8-9; §4; App. B.
- **Martini et al. 2022** [F] (arXiv:2201.00361v1): p. 10; §4 p. 20-22.
- **Manohar-Kutz-Brunton 2022** [F] (arXiv:1812.01574v2): §IV; §VI.
- **Gumussoy-Michiels** [F] (arXiv:2003.09252v1): §4-6; Table 7.4.
- **Edalatzadeh-Morris** [F] (arXiv:1802.05807) p. 1, secondary.
- **Dadfar et al. 2014** [F, partial] (arXiv:1402.0769) §3.2 p. 15.
- **Deshpande-Bhattacharya** [F, partial] (arXiv:2409.09596): abstract and introduction.
- Abstract only [A]: Barbagallo et al. 2012 (hal-00996503), Hervé et al. 2012 (hal-00994468) and Juillet et al. 2013 (hal-00995144).

### F. Solvers
- **Ipopt 3.14 documentation sources** [F]:
  - main.dox l.43-44;
  - special.dox, the quasi-Newton and warm-start sections, l.96-201;
  - options.dox: l.305-307, l.393-411, l.994-998, l.1123-1137, l.1406-1433;
  - install.dox l.303-315.
- **Clarabel** [F] (arXiv:2405.12762v1): §2.4, §2.6 and §3.3, plus settings.rs.
- **WARP** [F] (arXiv:2605.05728v1): Table 2 p. 4.
- **Uno** [F] (arXiv:2406.13454v2): §4.6.2 p. 13; §5.3 p. 18; Table 5 pp. 29-34; Table 6 p. 35.
- **Kiessling-Leyffer-Vanaret** [F] (arXiv:2409.09208v1) §2.2 p. 6.
- **Luo-Wächter** [F] (arXiv:2207.03082v2): abstract; §5.
- **Seidel-Küfer** [F] (arXiv:1910.13798v1): §1-2; Alg. 1 p. 7; §4.
- **Asl-Overton** [F] (arXiv:2006.11336v1): abstract; §2-3.
- **NCVX** [F] (arXiv:2111.13984v2 and arXiv:2210.00973v2) §5 p. 6.
- **Fendl-Schichl II** [F, set-up only] (arXiv:1506.08021) §4 p. 17.
- Package metadata [F]: the conda-forge ipopt, cyipopt and mumps recipes; PyPI cyipopt; the casadi 3.8.1 wheel's file list.

---

## 4. Verdicts on LITERATURE.md claims

### 4.1 Corrections

Each correction is entered as EDITS.md enters it. The verifier's verdict is in brackets.

| Line | Correction | Basis |
|---|---|---|
| 33 | BMO 2018 does have a stability constraint: α(A_f) < 0 on the full-order closed loop (eqs. 5a-5c), by eigs and uncertified (§4 p. 6). It is also nonsmooth (GRANSO), uses order-10 controllers and a full-state channel (Table 1). ROM-only variants left the FOM unstable on 5 of 12 problems (Table 2). | [upheld] BMO [F] |
| 34 | Bundle methods take about 1-2 evaluations per iteration (L-V Table 3 p. 390). Basic GS takes at least n + 1 gradients, adaptive GS a constant number (survey §4.3). GRANSO takes 1 plus line-search trials. The case rests on iteration count and structure; cost per iteration decides only against GS. The [R] tag did not cover GS. | [upheld ×2] D, F |
| 88 | HinfMOR: gradient descent took 37 iterations and 624 evaluations (§2 p. 6, Table 2.1). For BFGS the paper gives a linear rate at best, no count. | [upheld] archive |
| 91 | certify_sparse.py l.348-350 uses abs(Re λ) < 1e-8·max(1, max_j abs λ_j) over the returned set, with a strict "<". The radius is unshrunk (l.357). Slivers are dropped without σ (l.361-364, l.370). The per-eigenvalue "≤" form is lattice.py LEVEL_IMAG (l.115). The in-solve level set shrinks disks to 0.9 r: a misclassified crossing is cleared only if abs(Re λ) > 0.48·abs(Im λ − θ). It evaluates σ1 at every centre and gives slivers a finest-lattice centre (l.2343-2350). | [upheld] code at HEAD |
| 91 | SMV's sliver removal is justified empirically (p. 5-6). "Safe because of exact classification" is our inference. "Ours is not safe" stands for certify_greater. The promised Hutchinson count would be stochastic, not a certificate (DPS Lemma 2.1, §6). | [upheld] SMV [R], DPS [F] |
| 183 | AN 2006 (23) is the exact Clarke min-norm test. AN stop on (23)&(24) **or (25) alone** (p. 82). "Never step length alone" is the project's rule. The criticality half should be θ_σ or ‖G_t‖, because a test of type (23) need not fire near a kink (DP 2019 p. 11, fn. 5). | [upheld] |
| 139 | Kiwiel's w is ½‖p‖²/σ + Σλ_i(γ − s_i) + Σμ_j(−c_j), in f-units, with σ and the metric stated. The old formula mixed units. | [upheld] |
| 146 | AN 2006 (19) is unconstrained (inf over h ∈ Rⁿ, p. 79). θ_σ = −min_τ[Σ τ_i(γ − s_i) + ‖p_τ‖²/(2σ)] = −w. Constrained forms: DL 2018 (5.1) or the DP 2019 Ex. 3.3 penalty. | [upheld] |
| 147, 167, 717 | Appendix A said search-only papers were no longer cited for content, yet l.147 (CGT 2011) and l.167 (Hare et al.) still were. l.147 now cites CGT arXiv:2001.10802 eq. 3.11 [F], with radius normalisation by (2.4)/(4.1); Hare et al. is title only. | [upheld] |
| 266 | HinfMOR p. 5 picks a structured realisation, not a canonical one, and keeps a diagonal-scaling gauge. Gauge handling exists in identification: DDLC [S], Wills-Ninness-Gibson [S], Monti eq. 39 [F]. None of these is a gauge-free chart. | [upheld] |
| 388-389 | Idea 17: the literature gives a criterion (KP eq. 9), not a method; KP's own scheme is heuristic. "About 8 mat-vecs" cannot give a proof if it is a Krylov estimate. A certified resolvent upper bound is needed, and more points are to be expected (KP: 1519 against 825 intervals; 7,986 intervals and 54 h 42 min at n = 17,281). | [upheld] |
| 104, 41 | Position co-design "ahead" is now supported by full texts: Jin, Nibourel, WOP, MKB, Martini. Chen-Rowley and Kasinathan-Morris remain unread. | [upheld] |
| 439 | WOP 2023 read: GS solved fixed-order H∞ (n_K = 2) on cylinder models up to 22,060 states in 37.9-86.4 h per run (Table 6 p. 20), with an uncertified norm. GS was "more consistent and reliable" than BFGS there (p. 19). | [upheld] |
| 439 | Both premises of LESSONS §1 "do not add another inner solver" are gone. The solve is the bottleneck (LESSONS l.1082), and IPOPT installs via the casadi wheel or conda-forge cyipopt/ipopt. | [upheld] |
| 433 | Keep "not recommended", on corrected grounds (l.34 above). The Apkarian-Noll bundle trust region is a globalisation policy for the QP we already solve (ANR 2018 Remark 4, eq. 8), not another solver class. | [upheld ×2] A, F |
| 502 | SFPP is slow to become feasible (>100 iterations in §4.2) and finds fewer feasible points than SNOPT over §6. It is not permanently infeasible. | [upheld] |
| 607 | Yuan: MP 31:220-228 and 269-285, 1985 [B]. The IMA JNA 4:327-335, 1984 item is [S] with its title unconfirmed. Fletcher 1982 is unverified. Do not merge Yuan's Cambridge report (Burke-Engle ref [22]). | [upheld] |
| 656 | Leclercq 2019 did use multi-objective structured H∞ synthesis, per Jussiau 2024 Table 1, p. 6 and p. 30, and Nibourel p. 3 [F]. Pages are 26-65 or 22-65. The paper itself is unread. | [upheld] |
| 657 | Jussiau 2024 is a data-driven LQG paper (cylinder at Re 100). Its 113,000-state model is used for simulation only, its sensor is fixed, and it has no certificate. The group's structured H∞ flow paper is Jussiau 2022, IEEE L-CSS 6:3212-3217 [B]. | [upheld] |
| 682 | Noll-Prot-Rondepierre is Pacific J. Optim. 4(3):571-604, 2008, as given by four sources. DGNO and IJRNC print 569-602. | [upheld] |
| 394 | Yuan's two papers are [B]. LW prints the title of [52] as "...of a trust-region method..." and credits the line-search Cauchy-point test to [53] and Wright. For multiplier-weighted per-piece Hessians the clean read sources are L-V eq. 3 [R] and Lewis-Wylie 2019 Alg. 2.2 [F]; Thm 5.11 needs strongly convex pieces and k given. DGNO §8.2 is unweighted. Add Fletcher-Sainz de la Maza 1989 [B]. | **[weakened]** |
| 91, 620 | Benner-Lowe-Voigt, NACO 8(1):119-133, 2018, doi 10.3934/naco.2018007. The author is "R. Lowe", without an umlaut; the first name is unconfirmed. Content not read; SMV do not cite it. | **[weakened]** |
| 94 | Kamgnia-Philippe, read as INRIA RR-7770 v2, is tagged **[F]**, not [R]. Its rigorous segment test is Condition (B), eq. (9) p. 7; its own step control, (C) + (B'), is "a heuristic procedure" (§4.3); Ex. 3.2-3.3 show both tests failing. Our _march (winding_cert.py l.277-334) is of the same class. The tail bound is min(H¹, L2) since 9bfdee0. Ying-Katz and Johnson-Tucker are [B], with secondary accounts in Batra-Sharma p. 10-11. | **[weakened]** (tag only) |

### 4.2 Confirmed

Bibliography or content was confirmed. Additions entered in EDITS.md are marked "+".
- l.31: dense level sets cost O(n³); HIFOO §4 needs a dense norm [R]. l.32: HEC is local-only (BMO §2 p. 3 [F]).
- l.45 +: SMV's exactness applies to simple eigenvalues of a real pencil; near-axis pairs can still be computed as imaginary.
- l.61: Bompart is unverifiable. + Every fast-rate theorem needs an isolated minimiser, so (a) is a prerequisite.
- l.77 +: Barbagallo is [A], with secondary accounts.
- l.89 +: Noll-Apkarian 2005's identification is heuristic (BIM p. 3).
- l.93: HIFOO's abscissa phase is closed-loop.
- l.95: HIFOO is unstructured; HinfMOR p. 5/16/19; IJRNC conical damping; the CQ failure at M = L·I (also Burke-Eaton) + tridiagonal literature.
- l.96: SFPP is the failure mode of CMO §4.2.
- l.99 +: SHIRA and EVEN_IRA are reliability levers, not speed levers.
- l.101: tags.
- l.102, l.103, l.184 (1-D slices are not in HIFOO), l.360, l.405: HIFOO.
- l.105-106: tags and additions.
- l.148, l.150.
- l.186 +: Zhang Def. 4 [F], with a one-sided Goldstein transfer and the Royset wording.
- l.293 +: Martini [F]; its screening conclusion is weakened (§5.6).
- l.311: BSV is a dense reference only up to q of a few thousand.
- l.435: no HEC oracle (BMO §2).
- l.465, l.484 +: Remark 11 made precise.
- l.489-492 + (HIFOO §6.1), l.496, l.501.
- §7 bibliography: l.587-588, 590-593, 596, 600, 605-606, 608-615, 621-630, 638-655, 658, 676, 680-687.

### 4.3 Unverifiable

- l.61, l.88 (Bompart 2007).
- l.101 and l.348 (Morris 1994).
- l.104 (Kasinathan-Morris: "state feedback, derivative-free").
- l.184 (Bonnans-Shapiro, Hettich-Kortanek).
- l.228 and l.600 (Freitag-Spence-Van Dooren: content).
- l.423 (BLO 2003; the property itself is confirmed via Aliyev-Mengi).
- l.592 (BBK: content).
- l.599 (Gahinet-Apkarian: content).
- l.605, l.606 (content).
- l.612 (MOR venue).
- l.616 (Han 1981).
- l.617 (Bonnans-Shapiro).

---

## 5. Findings that bear on the current work

### 5.1 (a) Gauge-free normal form: supported, with four cautions

**What it is** [I; checked numerically in tools/lit_refresh_2026-10-17/tied_check.py; the verifier's re-run was not kept]:
- M = [[a, ω+g],[a, g]] with b = β[1; 1] gives tr = a + g and det = −aω.
- det(M − L·I) = L² − L(a+g) − aω. Both Routh-Hurwitz rows are linear in (a, g). Their gradients, (1, 1) and (−L−ω, −L), are independent iff ω ≠ 0.
- Every (tr, det) is reachable (a = −det/ω, g = tr − a), and det[b, Mb] = −β²ω ≠ 0.
- Under the fixed T = [[1, −1],[0, ω]]/(βω): T M T⁻¹ = [[0, 1],[−det, tr]] and T b = e₂. The form is the controllable companion form under a parameter-independent similarity.
- M = L·I is not representable. At a double root (s + ε)² the rows keep independent gradients (1, 1) and (ε − ω, ε), where the current det row's gradient vanishes at M = L·I.

**Support:**
- The matrix abscissa is Clarke-regular only at nonderogatory active eigenvalues; in coefficient space it is regular everywhere (Burke-Eaton §1 [F], restating Burke-Overton 2001 [B]). The normal form represents exactly the nonderogatory 2x2 blocks.
- Charts built on a pivot input are global for single-input pairs (Peeters-Hanzon-Olivi §3 p. 5 [F]).
- Degree-2 stability regions are polyhedral, while a single high-degree companion is nonconvex (Ait Rami-Henrion §4.1; Henrion-Louembet §1 [F]).
- Linear constraints need no constraint qualification (Mehlitz Thm 3.10 [F]) and enter the prox-linear model exactly (DL 2018 (5.1)).
- The tools keep a gauge and use abscissa-type controller constraints, which are non-Lipschitz at multiple eigenvalues:
  - tunableSS defaults to "tridiag" with B and C free [S];
  - TAC 2015 has 85 parameters for a 12th-order controller [R];
  - Apkarian-Noll 2022 has 54 [F];
  - Nibourel 2023 uses a tridiagonal A_K after McKelvey-Helmersson [F];
  - their controller constraints: IJRNC §6 [R]; Apkarian-Noll 2022 §5.1 [F].
- Every local fast-rate theorem needs an isolated minimiser, which a gauge orbit excludes (BIM Thm 4.5 and Assumption 4.2; Lewis-Wylie 2019 Thm 5.11; Han-Lewis Def. 3; Davis-Jiang A1 [F]). The normal form is therefore a prerequisite of idea 18.

**Correction to the framing** [verifier]: the current modal blocks [[d1, e],[f, d2]] already pass smoothly through real/complex transitions. A real-Schur or rotation form would not (Monti Supp. §S2.2 [F]). What (a) adds is gauge removal, exactly linear rows and LICQ at a double root.

**Cautions:**
1. **Keep the rows elastic.** Group A proposed hard rows. The verifier found this conflicts with the decision in LESSONS l.119: elastic rows let a start outside the line walk back instead of making the QP infeasible. [I]: with exactly linear rows and a feasible iterate, d = 0 satisfies them, so an exact ℓ1 elastic row whose penalty exceeds its multiplier has zero slack at the QP solution. Elastic rows then act as hard rows, and the second-order correction is not needed for these rows.
2. **Blocks sharing a root are non-identifiable** [I, numerical]. In a parallel sum of tied blocks, the smallest singular value of the Jacobian to transfer-function coefficients is 4e-17 at one shared real root and 1.1e-3 at a separation of 0.02. Rank drops by 2 for equal denominators. Cross-block Jordan structures are unreachable. The current form has the same limitation, so this is not a regression [verifier]. Optimisers park 3-4 poles at −0.0500 (LESSONS l.1012). Log the minimum cross-block root distance per accepted step, and count the effective dimension as NZ minus shared roots in the stationarity tool. Fallback: merge two blocks into a 4x4 companion, or allow the superdiagonal coupling of the McKelvey-Helmersson form (Monti Supp. §S2.2 [F]).
3. **Central MIMO: the pivot chart is local** (PHO §3 p. 5 [F]). A minimal block with M = λI and rank-2 B is unrepresentable. Approaching it makes the non-pivot column grow like 3.9/d (3.9e3 at d = 1e-3) [I]. The WIP already picks the pivot by _pivot_score and refuses cond([b, Mb]) > 1e12 (controller.py l.237, l.294-375 in the 20261010s work in progress). Per-round chart switching at the BFGS reset (an exact linear map) and splitting into two 1x1 blocks would be new.
4. **Read admissibility from the rows, not from eig.** A block parked at the exact double root −ε, perturbed by 1e-16, gives a computed rightmost real part up to −ε + 1.4e-9 [verifier; B measured 5.9e-10]; Henrion-Overton §6 [F] and HIFOO §6.1 [R] document the same sensitivity. So:
   - count P_ctrl and test the line on (tr, det);
   - build winding_cert's ctrl_exact polynomials from the block coefficients rather than from roots (winding_cert.py l.150-157);
   - log ‖A_c‖₂/max abs(λ_c), because the Neumann tail needs W_T > ‖A_c‖₂ (l.436-452) and the off-diagonal ω + g sets ‖A_c‖.

**ω.** Each ω gives an exactly equivalent chart via a fixed linear map, so re-centring is result-neutral at the point. ω ≈ sqrt(abs(det)) cut cond(V) at a near-double real pair from 1.8e4 to 1.0e3 in B's test (tools/lit_refresh_2026-10-17/tied_cond.py [I]). The WIP re-centres at every job start. Re-centring at every round start, where BFGS resets anyway, is an untested extension.

**Optional specification rows** (the user's decision). The rationale: controller resonances can invalidate the sampled Ω (IJRNC §6 p. 3222 [R]), and the compensator-pole count in the Nyquist count should stay fixed (Apkarian-Noll 2022 §5.1 [F]).
- **Radius.** All roots lie in abs(s) < δ iff abs(det) < δ² and abs(tr) < δ + det/δ (Schur-Cohn on the scaled polynomial). This is linear in (a, g) [I; verifier]. B's "det ≤ δ²" alone is incomplete.
- **Damping ζ ≥ μ.** The condition −tr ≥ 2μ sqrt(det) is **not convex**: (−1, 1/(4μ²)) and (−3, 9/(4μ²)) satisfy it, their midpoint does not [verifier]. B's "second-order cone" claim is wrong. Since g = −tr − 2μ·sqrt(det) is convex, its linearisation at the current point is a conservative linear row [I].

### 5.2 (b) Ratio-based trust radius

**The WIP rule** (the trust-rule work in progress, 20261010s: sqp_inner.py l.51-66, l.1817-1836, l.2154-2160, l.2196-2198):
- after a step accepted at α < 1, Δ = 2α|d|;
- after a full step: ×2 (capped at max(1.0, start radius)) at ρ ≥ 0.75 on the boundary; |d|/2 at ρ < 0.25;
- after a rejected search: min(Δ/2, 4α|d|);
- the line search starts at α = 1;
- a failed QP is retried with Clarabel's equilibration bounded to [1e-2, 1e2].

ρ is computed after full steps only, so the α·pred bias raised by group A does not arise [verifier].

**Supported:**
- The standard ℓ∞-box SQP rule grows Δ when the region is active at an accepted step and decreases it below min(Δ, ‖d‖∞) on rejection (Kiessling-Leyffer-Vanaret §2.2.1 p. 6; Uno §4.6.2 p. 13 [F]).
- Burke-Engle Alg. 4 p. 19-20 [F] accepts at r ≥ β1 but shrinks δ whenever r < β2 (β1 ≤ β2), i.e. also after an accepted poor step, with no boundary condition. It covers our finite-Ω class, convex h of smooth c (Thm 6.1).
- KLV §2.2.2 [F]: αd is "usually not optimal (or even feasible)" in the equivalent region. This is indirect support for re-solving the QP in a smaller region after a partial step.

**Not supported or contradicted:**
- No source read sets Δ to 2α|d| after an **accepted** step.
- ANR 2016/2018 keep or double the memory radius after every serious step (Γ = 0.1, ANR 2016 p. 13).
- Lewis-Wright lengthens the allowed step after every acceptance (Alg. 2 p. 25).
- ANR 2018 Remark 8 p. 11: shrinking at every failure "must fail" for nonsmooth models. ANR 2016/2018 Alg. Step 7 and DGNO Alg. 1 Step 7 p. 8 (also the 1905.11511 Fig. 3 flowchart [R]) halve R (double τ) only if ρ̃ ≥ γ̃, i.e. if the failure persists once the trial's own cut is in the model.
- The project rejected the 2α|d| leg three times (LESSONS l.69-70), under defects since fixed; LESSONS l.1084 asks for a re-test.
- The 0.25/0.75 thresholds are the common textbook pair, but they come from no source read here.

**Recommendations:**
1. **Gate the shrink legs on a ρ̃ analogue** [verifier upheld]. After a rejected or partial trial, find the row that attains the trial's max, or the violated stability row.
   - If it is a piece the QP did not contain (a new top; a σ2 block not enrolled because the gap was ≥ 0.15; an eigenvalue without a row), enrol it as an anticipated plane (ANR 2018 §4.2 p. 12) and keep Δ.
   - Shrink only if the QP's own rows failed.
   - This costs one extra QP and evaluation per gated failure (a null step). Log which leg fired.
2. **A/B the partial-step leg on its own:** the ratio rule with and without 2α|d|, at equal evaluations, judged against A/A noise. Burke-Engle supports shrinking on a poor ratio after acceptance; nothing supports tying Δ to α|d|.
3. **Give the exit a radius-free measure.** The "pred" exit is stat < 1e-4·gscale AND pred ≤ 1e-7·max(abs(t), 1) (sqp_inner.py l.2252-2253; run_case_epi.py l.4695).
   - pred is measured in the current box. Under the rule, Δ can fall to 2α|d| with α ≈ 2⁻⁹, so pred becomes small because Δ is small.
   - The stat leg is "rel", which LESSONS l.1073 found unreliable on the BFS.
   - Add pred(Δ_ref)/Δ_ref at a fixed Δ_ref in a declared metric (CGT (2.4)/(4.1) [F]), or ‖G_t‖ at a fixed t. At minimum, log the QP pred, Δ and the radius-free value at each exit [verifier weakened only DL 2018's "fixed t" detail].
4. **Bounded H is optional** [verifier weakened].
   - Every ratio-rule convergence theorem read assumes bounded model Hessians: Burke-Engle Thm 6.1; ANR 2018 Alg. 2 (‖Q‖ ≤ q); DGNO (−qI ⪯ Q ⪯ qI); SQP-GS per CMO p. 3.
   - But the 5e11-2e12 eigenvalues come from the two failing QPs only (LESSONS l.1083), and BFS steps are box-limited (37-40 of 40 on the boundary, l.1084).
   - Clipping is licensed but its benefit is unmeasured; CMO p. 3 also records Lewis-Overton's argument that ill-conditioning can help. Log λ_max(H) and whether the box or H limits each step before trying it.
5. **U-space curvature** [weakened, as a hypothesis]. With H = I after each reset, flat U-directions carry no problem curvature, so the step is a box corner cut by α. Seeding the reset with multiplier-weighted per-piece Hessians (L-V eq. 3 [R]; Lewis-Wylie 2019 Alg. 2.2 p. 4 [F]) is the structural fix to A/B after (a). DGNO §8.2 sums unweighted Hessians.
6. **Oracle floor** [C; low confidence, not checked by the verifier]. When abs(pred) is below about 10·rtol·γ (certificate rtol 1e-6; Newton stop 1e-7σ), ρ is noise: neither shrink nor grow on it. BMO §2 p. 3 documents line-search halts caused by an inconsistent norm oracle.
7. **Re-measure the model's validity radius after (a).** It was about 1e-5 along the H = 0 box direction, and it bounds what any radius rule can achieve.

### 5.3 (c) Paired ARPACK tiers

- **No paper read addresses concurrent Krylov tiers.** No group was assigned parallel eigensolvers, so this is "not searched", not "absent from the literature". LITERATURE §4 lever 1 (Dean-Barroso [S]) remains the only hedging reference.
- **Consistent principles:** a decision must not rest on a computation that can fail silently (Mitchell 2020 Key Remark 6.3 p. 17 [F]); many cheap independent tests are preferable (Mitchell 2021 §1.3 [F]).
- **Identity** [I]. lattice.py l.579-595 tries (24, 2) then (100, 3) with a deterministic v0 (_v0(2N)). It sizes a disk only when all LEVEL_K eigenvalues converged, which is SMV's practice.
  - Running the tiers concurrently is bitwise identical to the sequential tiers iff the rule "tier 0 whenever it converges, else tier 1" is kept regardless of which finishes first.
  - Taking the first to finish would break identity, because converged Ritz values differ between ncv at the tolerance level.
  - Log the deciding tier and both tiers' times per disk, so the hedge's payoff (LESSONS §19.4, "doubtful until measured") is measured.
- **Structured solvers are no substitute for the ncv-100 tier** (SHIRA, EVEN_IRA, K(ξ); verifier upheld):
  - they cost two shifted solves per application and need a custom J-orthogonalising Arnoldi (Appeltans-Michiels §2.2 eq. 2.4 [F]);
  - they leave the mirrored pair at k = 2 and do not cure the clustering that makes ncv 24 fail between peaks;
  - Mitchell 2021 §2.3 uses them only as a fallback.
- **Unverified lever:** Meerbergen-Spence 1997 (snippet [S]). Shift-invert Arnoldi with a semidefinite B can be unstable; their remedy is Arnoldi in the B semi-inner product with purification. Our E_H = diag(M_cl, −M_cl) is singular and ARPACK runs in the Euclidean product. Read the paper before trying it.

### 5.4 Stationarity tool (tools/bfs_stationarity.py, STATIONARITY.md)

1. **One measure, not three** [D; verifier upheld].
   - On a finite Ω the epigraph problem is a convex composite.
   - The prox-linear step ‖G_t‖ = ‖p*‖, the aggregate of AN's θ-QP with σ = 1/t [I], is continuous and zero iff stationary.
   - ½‖G_t‖ ≤ dist(0, ∂φ) (DL 2018 Thm 5.6 (5.7) p. 18).
   - A small ‖G_t‖ certifies a nearby nearly stationary point, though one that cannot be computed (DL Thm 5.3 p. 16; DP 2019 (4.9) p. 15: ‖x̂ − x‖ ≤ (2/μ)‖G‖, dist(0, ∂F(x̂)) ≤ 4‖G‖).
   - Kiwiel's w, AN's θ and ‖G_t‖ are one quantity: −θ_σ = w ≥ (t/2)‖G_t‖².
   - Compute it with one round-end Clarabel solve of the existing QP: H = I/t, no box, all Ω rows with their offsets, compensator and position rows linear, closed-loop rows as an exact penalty (DP 2019 Ex. 3.3). Record t, ω per block and the metric.
2. **The box LP** (STATIONARITY.md l.56) is the Cartis-Gould-Toint measure φ^δ_{w,1} (eq. 3.11 [F]), not "Polak's optimality function with a box": Polak's and AN's θ carry a quadratic term (AN p. 79 [R]).
   - Report pred(Δ)/Δ, which is nonincreasing in Δ [I].
   - The verdict classes at l.75 compare the absolute decrease at Δ_v with 1e-6γ and with tol/k, so they depend on the radius.
   - Rename them "no certifiable decrease within Δ_v (oracle-limited)", "certified decrease below the progress rule's per-round scale" and "certified decrease".
3. **The ε = 1e-6 "Clarke reading"** (l.27, l.76) is an exact-active-set residual. Such residuals need not tend to zero near a kink even as G_t → 0 (DP 2019 p. 11 and fn. 5 [F]).
   - It still certifies f′(z; d) < 0 (l.127), so it is not empty, but it says nothing about the distance to stationarity.
   - Label it so, and lead with the certified decrease and ‖G_t‖.
4. **Wording with a growing Ω:** "δ-stationary for the problem over Ω_k at metric M and step t; Ω_k ε-complete yes/no" (Royset Def. 2.2, Prop. 2.3 [F]).
   - A large ρ at ε certifies non-Goldstein-stationarity at δ = ε/(2L_f) [I]; a small ρ certifies nothing.
   - Zhang Thm 5 is a black-box result and does not apply to an oracle that exposes its pieces [F].
5. **Constraint qualification:** ρ_q(ε) with ε-active normal cones is an AKKT residual that needs no constraint qualification (Mehlitz Def. 3.1, Thm 3.2 [F]).
   - The tool's MFCQ margin is the bounded-multiplier test (Lemma 3.4).
   - Flag records whose minimising μ grows with ε (Fritz-John type).
   - The linear rows of (a) are polyhedral, so they need no constraint qualification (Thm 3.10).
6. **Optional identification diagnostic:** the water-filling prox of the max (BIM Ex. 2.3, Thm 3.2 [F]) gives a data-driven active set. Its admissible γ range depends on the unknown ‖x − x*‖, so it is a diagnostic only.
7. **Rerun the six-record table after (a)** as the before/after metric check. The measures remain metric-dependent, so keep the scale-free group-wise condition.

### 5.5 Certificates

- **γ certificate (idea 6)** [verifier upheld]:
  - sigma-confirm every near-axis Ritz value within an accuracy-tied band, e.g. tol·ε_mach·‖A₂⁻¹A₁‖ in Mitchell 2020 §5.3 [F], instead of the 1e-8·max(1, abs λ) floor;
  - add a free mirror-pair test: for a shift on iR, an off-axis eigenvalue's mirror −λ̄ is equidistant, so a near-axis nearest eigenvalue without its mirror among the converged set is probably on the axis. This is a heuristic, since it misfires on ties or a missed mirror;
  - evaluate σ1 at every certify_greater shift and at each sliver midpoint before dropping it. Each costs an iterative SVD (9-17 applications cold, lattice.py l.100-101), not one solve pair;
  - apply the in-solve path's 0.9 shrink;
  - use a structure-preserving recompute only for undecided near-axis cases (Mitchell 2021 §2.3).
- **Winding count:** the tail closure is a Gohberg-Sigal/Rouché argument (Bindel-Hood Lemma 2.1, Remark 1 [F]). For a half-plane count it must hold on the closing arc {Re s ≥ line, abs(s) ≥ W_T}, not only on the line. The gbound docstring states it for the line only (bfs/plant.py l.587-600) [verifier upheld]. State and check the arc case.
- **Proof route for idea 17** [I]: certify σ_min(s_k M − A) at the march knots by an inertia (LDLᵀ) test of a Hermitian dilation; bfs/plant.py l.374-379 already does inertia certificates. Then extend along the segment by Weyl's inequality and check KP Condition (B). Apply it only where the margins are thin.

### 5.6 Flow and specification items (the user's decisions)

- **Reporting:** report the modulus margin 1/‖S‖∞ (SISO) and ‖KS‖∞ for every certified record (Nibourel eq. 5.3 [F]). Optionally impose them as hard σ-row families.
- **D_c = 0:** a strictly proper option (Nibourel §5.2 item 5 [F]) is a robustness argument only. Its tail rationale is outdated: the veto failure came from the H¹ tail, fixed with W_T = 7,680 [verifier weakened].
- **Barbagallo 2012 baseline:** not comparable, since it uses LQG, a B1 → Cp channel and a larger domain (BFS.md l.84, l.91-95, l.1014). Compare only by evaluating both designs on one plant and one channel. That needs the full paper.
- **Comparators:**
  - at our optimised ports, an ERA/BT ROM with fixed-order synthesis (Nibourel-style), certified by our full-model certificate and winding count;
  - a Jin-style H2 position map;
  - WOP-style gradient sampling only as a small reference.
- **Per-port Parrott maps (idea 5):** Martini's "actuator placement has a negligible influence" holds for non-causal H2/PSD with one sensor. Whether our H∞ full-field bound is equally flat is untested, so measure the map before deciding to screen with it [verifier weakened].
- **Sensor placement:** literature physics puts the feedback sensor within about one wavelength downstream of the actuator (Nibourel App. A [F]). Our records are of that type. A multistart with the sensor upstream would test for a feed-forward basin [E, low confidence].

---

## 6. The 23 ranked ideas, re-ranked

Status was checked at HEAD and in the WIP worktrees [C].

| New | Idea (old #) | Status | What the refresh changes |
|---|---|---|---|
| — | 1 tail bound | done (9bfdee0, min(H¹, L2)) | The five vetoed decentral designs were never saved (LESSONS l.1089), so they cannot be re-counted. Remaining: the closing-arc statement (§5.5). |
| — | 2 MIMO 8 rounds | done (bfs/run_bfs.sh l.148: 8:0.0015, `--sqp-stat pred`) | none |
| — | 3 gauge rebalance | superseded by (a) (WIP 5999e64) | The literature favours removing the gauge over rebalancing it (§5.1). |
| 1 | (a) normal form A/B | WIP | Strongest support. Log the cross-block root distance, pivot score, ‖A_c‖/max abs(λ_c) and the validity radius; keep the rows elastic. |
| 2 | (b) ratio rule A/B + ρ̃ gate + radius-free exit (absorbs 14) | WIP | The partial-step leg is unsupported, so A/B it separately; a ρ̃ gate; the exit is otherwise radius-dependent (§5.2). |
| 3 | 6 γ-certificate audit | open | Concrete, cheap, literature-backed fixes (§5.5); raised from 6. |
| 4 | 4 stationarity diagnostic | tool at HEAD | Add ‖G_t‖ and pred/Δ; relabel; rerun after (a) (§5.4). |
| 5 | 5 Parrott bounds | open | Still the only "distance to optimum"; measure the map's flatness before screening actuators. |
| 6 | 7 validation stack + comparators | open | Add the ROM + fixed-order comparator at our ports and a Jin-style H2 map; the BSV dense reference only to q of a few thousand. |
| 7 | 10 async winding count | open | Now possible with the tail fixed. |
| 8 | 20 robustness reporting + ‖S‖∞, ‖KS‖∞ | open | Raised from 20: cheap, and standard practice in the nearest flow paper. |
| 9 | 13 restarts / multistart | open | Add a sensor-upstream start. |
| 10 | 8 Youla benchmark | open | none |
| 11 | 9 cross-structure identity | open | none |
| 12 | 11 full mesh | open | Kasinathan-Morris only secondary; WOP's multi-fidelity is non-monotone (p. 18). |
| 13 | 12 beyond-band screen | open | none |
| 14 | 17 winding proof | open | A concrete route (inertia + Weyl + KP (B)); cost above "8 mat-vecs". |
| 15 | 18 second-order tail | after (a), 4 | Templates now read (BIM Alg. 4.1; Lewis-Wylie 2019 Alg. 2.2); Bompart still unread; gate it within a round (no new piece for several steps, strict complementarity, active pieces < effective dimension). |
| 16 | 15 parallel levers | paired tiers WIP | §5.3; Meerbergen-Spence to read. |
| 17 | new: bounded-H option | — | Theory-licensed, benefit unmeasured; log first. |
| 18 | 19 order ladder | open | none |
| 19 | 16 small-gain gate | open | none |
| 20 | new: controller radius and damping rows, D_c = 0 | — | Specification options (§5.1, §5.6); the user's decision. |
| 21 | 22 Lipschitz stability function | open | It now concerns the closed-loop rows only (the compensator rows are linear). |
| 22 | 21 pole-targeted rows | open | none |
| 23 | 23 convex-concave | open | none |

---

## 7. Solver libraries: IPOPT and others

**Verdict: do not adopt any NLP library as the inner solver; IPOPT is worth at most one bounded diagnostic.** The verifier upheld every point below.
- **Assumptions.** Ipopt assumes twice continuously differentiable functions and a fixed problem (main.dox l.43-44). Our rows appear and disappear mid-solve, and σ1 at multiplicity needs a surrogate.
  - The 2x2 Clarke block can be written as three smooth inequalities, but LICQ fails at exact coalescence, where the iterates go [I].
  - PyGRANSO has no cones either (NCVX §5).
- **Warm starts.** Every row change means a restart, and IPM warm starts are weak. The docs call the option "far from solving" warm starting and say it may increase iterations when the active set changes (special.dox l.178-190). WARP Table 2 (IPOPT, exact Hessian) gives:
  - cold: 22.6 iterations;
  - x* only: 23.7;
  - x* with cold duals: diverges;
  - full primal-dual-barrier state: 3.3.
- **Hessian.** Without one, Ipopt uses scaled L-BFGS with history 6 (options.dox l.1406-1433). Scaled L-BFGS "often breaks down" on effectively nonsmooth problems, unlike full BFGS (Asl-Overton abstract [F]); our SQP already uses full BFGS at NZ = 34-60.
- **No evaluation saving.** On 21 small CUTE minimax epigraph NLPs, IPOPT with exact Hessians needed a median of 11.5 evaluations (max 660, 1 failure), against 9 for filterSQP and 13 for SNOPT (Uno Table 5). SNOPT was "completely uncompetitive" on nonsmooth spectral control problems (CMO §6 [R]).
- **No trust region.** Ipopt cannot help with (b).
- **Others** (Uno Table 6 and package metadata):
  - SNOPT, KNITRO and WORHP are licensed and not on conda-forge; KNITRO-ASM (SLQP, ℓ1 trust region) is our structural relative.
  - filterSQP needs an exact Hessian.
  - Uno has no quasi-Newton yet (p. 18).
  - SLSQP is dense, without a trust region or cones.
  - blockSQP and the QP-only solvers (DAQP, qpOASES, OSQP, PIQP, ProxSuite, HPIPM) cannot hold the SOC block.
  - MPBNGC is first-order (Fendl-Schichl p. 17).
- **Installability** (metadata only; nothing installed). PyPI casadi 3.8.1 (manylinux wheel) bundles libipopt.so.3.14.19 and libcoinmumps. conda-forge has cyipopt 1.7.0 and ipopt 3.14.20 linked against mumps-seq 5.8.2 and libspral; mumps-seq's _seq-suffixed libraries coexist with PETSc's MUMPS. PyPI cyipopt is an sdist only. No new repository access is needed.
- **Open risk.** "The non-MPI version of MUMPS uses internally a fake implementation of MPI, which can prevent using Ipopt together with MUMPS within an MPI program" (install.dox l.303-315). Smoke-test on a compute node before any experiment: `mpirun -n 2 python -c 'from mpi4py import MPI; import petsc4py; import cyipopt'`, then HS071 on rank 0. On a clash, try linear_solver=spral or a separate non-MPI process.

**If the user wants the evidence anyway, one GL A/B:**
1. Freeze a recorded round: z0, Ω and the row set, with the 2x2 blocks excluded or in the 3-inequality form.
2. Run cyipopt with exact Jacobians from the evaluator, limited_memory_max_history ≥ NZ and mu_strategy adaptive. Warm-start restarts with the full primal-dual state and a small mu_init.
3. Run the current SQP from the same start with the same evaluation budget.
4. Decision rule: IPOPT must reach the SQP's final certified t with no more evaluations in at least 2 of 3 starts, and lose by at most 1 % in any.
5. Only then run one SISO and one central BFS pair at equal evaluations, judged against A/A noise (0.7 % on GL MIMO).

Expected outcome: no gain.

**QP solver.** Keep Clarabel.
- It has no warm start (paper §2.4; settings.rs), and its default equilibration bounds are 1e-4 to 1e4.
- The WIP already retries a failed QP with bounds [1e-2, 1e2] (sqp_inner.py l.61-66, l.1198-1208 in the 20261010s work in progress). Log how often that retry is used.
- An active-set SOCP-SQP with polyhedral cone cuts (Luo-Wächter [F]) is the only warm-startable route that keeps the SOC block. It is worth it only if QP time exceeds about 15 % of the iteration wall; QPs take about 2 s against 10-45 s per evaluation.
- MOSEK serves only as a cross-check on dumped QPs; its documentation was not accessible.

**If a library route is ever wanted for seeding or diagnostics:** the published pattern is a smoothed squared-hinge objective over sampled singular values with bisection on γ and off-the-shelf BFGS (SOBSYN, Schwerdtner-Voigt eq. 10, Alg. 1 [F]). It is uncertified, with stability coming from structure only. No published use of IPOPT for frequency-grid fixed-order H∞ was found (search only).

---

## 8. Not accessed; PDFs the user should supply

**In priority order:**
1. **Bompart, Noll, Apkarian**, Numer. Math. 107(3):433-454, 2007. Needed before idea 18. HAL hal-01868767 has no file; Toulouse, Springer and ISTEX were refused.
2. **Leclercq, Demourant, Poussot-Vassal, Sipp**, JFM 868, 2019. Needed for the novelty claim and structured H∞ in flows (ONERA group; the user may hold it).
3. **Jussiau, Leclercq, Demourant, Apkarian**, IEEE L-CSS 6:3212-3217, 2022. The same group's structured H∞ flow paper.
4. **Barbagallo et al.**, JFM 703:326-362, 2012, and the Barbagallo thesis (pastel-00654930). Needed for the baseline's metric and channel.
5. **Noll, Apkarian**, Math. Program. B 104:729-747, 2005, and **Apkarian, Noll, Prot**, SIOPT 19(1):281-306, 2008. Needed for multiplicity identification and the radius management of the spectral bundle method.
6. **Mehrmann, Schröder, Simoncini**, LAA 436(10), 2012 (EVEN_IRA), and **Meerbergen, Spence**, Math. Comp. 66, 1997. Needed for on-axis classification and the singular-B Arnoldi lever.
7. **Zhou, Tits**, SIOPT 6(2):461-487, 1996. The working-set SQP for discretised minimax; read before any redesign of Ω or the stop.
8. **Kasinathan, Morris**, IEEE TAC 58, 2013, and **Chen, Rowley**, JFM 681, 2011. Co-design priors.
9. **Benner, Lowe, Voigt**, NACO 8(1), 2018, and **Freitag, Spence, Van Dooren**, SIMAX 35(2), 2014.
10. **Ying, Katz**, Numer. Math. 53, 1988. Needed for the winding proof.
11. **Gahinet, Apkarian**, IFAC 2011, and the MathWorks tunableSS, systune and TuningGoal pages. These are snippet-only.

**Also not accessed:**
- **Group A:** Yuan 1985a/b and 1984; Fletcher 1982; Oustry 2000; Mifflin-Sagastizábal 2005; Noll-Prot-Rondepierre 2008 (full text); Apkarian-Noll-Prot J. Convex Anal. 2009; Curtis-Overton 2012; Byrd-Schnabel-Shultz 1988; Womersley-Fletcher 1986; Fletcher-Sainz de la Maza 1989; Helmberg-Overton-Rendl 2014; the Bompart, Prot and Ravanbod theses.
- **Group B:** Burke-Overton 2001 (both papers); Burke-Lewis-Overton 2001-2003; Lewis-Pang 2008; Gürbüzbalaban-Overton 2012; DDLC 2004; Ribarits-Deistler-McKelvey 2004; Wills-Ninness-Gibson 2008; McKelvey-Helmersson 1996; Hanzon-Ober 1998; Henrion-Šebek-Kučera 2003; the DSP texts.
- **Group C:** Mehrmann-Watkins 2001; Johnson-Tucker 2009; HEC 2016; Rump 2010; Boyd-Balakrishnan-Kabamba 1989; Benner-Sima-Voigt 2012; ARIMA 2015.
- **Group D:** Cartis-Gould-Toint 2011; Hare-Sagastizábal-Solodov 2016; Lewis 2002; Burke-Lewis-Overton 2005; Hettich-Kortanek 1993; Polak 1997 and 1987; Overton 1988; Han 1981; Bonnans-Shapiro 2000; Helou-Santos-Simões 2020; Kiwiel 2007.
- **Group E:** Dergham 2013; Hervé 2012 (full text); Barbagallo 2009; Sipp-Schmid 2016; Juillet 2013 (full text); Morris 2011 and 1994; Oehler-Illingworth 2018; Belson 2013; the published JFM version of Jussiau 2024; the ONERA thesis proposal.
- **Group F:** Wächter-Biegler 2006 and 2005; Fletcher-Leyffer 2002; SNOPT 2005; KNITRO 2006; WORHP 2013; Kraft 1988; the MOSEK documentation; MPBNGC; Stein 2012; Djelassi-Mitsos-Stein 2021; Reemtsen; Lukšan-Matonoha-Vlček 2009/2010; Grimholt-Skogestad 2018; PENBMI; SSDP; Panier-Tits 1989; the Polak reports.
- **Every arXiv paper read:** its published journal version was not compared.

---

## 9. Missed references (not in LITERATURE.md before)

**Read [F]:**
- **Dao-Gwinner-Noll-Ovcharova 2016** (arXiv:1401.6807): the only open, full statement of the proximity-control radius rules (Alg. 1 p. 8, §4.5).
- **Burke-Engle** (arXiv:1806.05218): a ratio trust region proved for convex composites with bounded H (Alg. 4, Thm 6.1).
- **Kiessling-Leyffer-Vanaret 2024** (arXiv:2409.09208): the standard ℓ∞-box radius rule and why αd is not the region's optimum (§2.2).
- **Lewis-Wylie 2019** (arXiv:1907.11742): k-bundle Newton, the template for multiplier-weighted per-piece Hessians.
- **Curtis-Robinson-Zhou** (arXiv:1708.02552): self-correcting BFGS keeps eigenvalues bounded on a fraction of iterations (Thm 3.1).
- **Drusvyatskiy-Paquette 2019** (arXiv:1605.00125), **Mehlitz 2020** (arXiv:2006.09734), **Royset 2023** (arXiv:2201.05250) and **Cartis-Gould-Toint 2020/2019** (arXiv:2001.10802, arXiv:1705.07285): the stationarity measures and wording of §5.4.
- **Burke-Eaton** (arXiv:1511.03687), **Peeters-Hanzon-Olivi 2007** (arXiv:1012.3281), **Henrion-Overton 2006** (arXiv:math/0603681) and **Apkarian-Noll 2022** (arXiv:2211.08083): the theory behind (a) and the controller-constraint practice.
- **Nibourel et al. 2023** (arXiv:2206.06224): the closest structured fixed-order flow design; the comparator and margin practice.
- **Appeltans-Michiels** (arXiv:2110.02665), **Mitchell 2020** (arXiv:1907.06537) and **Di Napoli-Polizzi-Saad** (arXiv:1308.4275): the certificate items of §5.5.
- **Uno** (arXiv:2406.13454), **Luo-Wächter** (arXiv:2207.03082), **Seidel-Küfer** (arXiv:1910.13798), **WARP** (arXiv:2605.05728), **Asl-Overton** (arXiv:2006.11336), **SOBSYN** (arXiv:2209.07558) and **Clarabel** (arXiv:2405.12762): the solver assessment of §7.
  - Seidel-Küfer: plain Blankenship-Falk exchange is not quadratically convergent in general. Their quADAPT adds derivative information about the lower-level solution. Our peak following re-solves the tops, and at a maximiser the fixed-ω row gradient already equals the envelope gradient, so the second-order top-motion term is what quADAPT-type information would add. The verifier weakened the claim that "following = quADAPT".
- **Dadfar et al. 2014** (arXiv:1402.0769): a semi-decentralised LQG design lost only 11 % against centralised control. Our decentral/central gap is about 3x; test crossed pairings.
- **Deshpande-Bhattacharya 2024** (arXiv:2409.09596): convex sparse actuation and sensing co-design, at LMI scale only.

**Not read** (listed in the EDITS E-insert "7d"):
- Jussiau et al. 2022 L-CSS [B];
- Juillet et al. 2013 [A];
- Womersley-Fletcher 1986 [B];
- Fletcher-Sainz de la Maza 1989 [B];
- Zhou-Tits 1996 [S];
- DDLC 2004 [S] and Wills-Ninness-Gibson 2008 [S];
- McKelvey-Helmersson 1996 [B] and Hanzon-Ober 1998 [B];
- Grivet-Talocia 2007, Kressner 2006 and Chiprout-Nakhla 1995 [B, via SMV p. 3];
- Philippe et al. 2013 [A];
- Lewis-Pang 2008 and Gürbüzbalaban-Overton 2012 [B];
- Wächter-Biegler 2006, SNOPT 2005 and Fletcher-Leyffer 2002;
- Charisopoulos-Davis [A], which is not applicable (it needs sharp growth).

---

## 10. Edits outside LITERATURE.md (suggested, not applied)

- **LESSONS.md**
  - l.78: "IPOPT and qpOASES cannot be installed with pip" is outdated (casadi wheel; conda-forge cyipopt).
  - l.1088: the threshold is a strict "<" with max_j abs λ_j over the returned set in certify_sparse.py; the per-eigenvalue "≤" form belongs to lattice.py.
- **STATIONARITY.md**
  - l.56: relabel the LP as "CGT-type box criticality measure".
  - l.75-76: the verdict classes and the ε = 1e-6 label (§5.4).
- **certify_sparse.py l.36-45:** the docstring presents the Hutchinson count as a certificate; it would be a stochastic estimate.
- **winding_cert.py l.48-55:** "about 8 extra matvecs" is not a proof route by itself.
- **bfs/plant.py l.587-600:** state the closing-arc case of the gbound docstring.
- **LITERATURE.md status lines:** l.46, l.48, l.64-65 and ideas 1-3 are out of date at HEAD (tail bound and 8 rounds done; gauge superseded). These are status, not literature, so EDITS.md leaves them to the user.
