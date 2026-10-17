# Group C: certificates, counts, structured eigensolvers -- raw notes (2026-10-17)

Network: arxiv.org abs/pdf OK; hal.science record pages OK (no PDF for KP2013);
REFUSED (403/CONNECT): csc.mpi-magdeburg.mpg.de, etna.ricam.oeaw.ac.at, www2.math.uu.se,
opus4.kobv.de, perso.uclouvain.be, eudml.org, cs.nyu.edu, aimsciences.org, ams.org,
linkinghub.elsevier.com, idp.springer.com (Springer non-OA), link.springer.com/article (403),
export.arxiv.org (API). WebFetch: getaddrinfo ENOTFOUND for MPI host.

## Kamgnia-Philippe -- FULL TEXT of INRIA RR-7770 v2 (arXiv:1110.4797v2, 2 Mar 2012, 18 pp),
NOT the ETNA 40:1-16 (2013) final (ETNA host refused; HAL hal-00795730 has metadata only,
citation_volume 40, firstpage 1-16, online 2013/02/28).
- §2.1 eq (1)-(2): N_Gamma = (1/2 pi i) oint f'/f, f = det(zI - A).
- §3.1 eq (5): f(z+h) = f(z) det(I + h R(z)), Phi_z(h) = det(I + hR(z)).
- Lemma 3.1 Condition (A), eq (7): |Arg Phi_z(s)| < pi for all s in [0,h] => arg0 f(z+h) = arg0 f(z) + Arg Phi_z(h) (8).
- §3.2 Condition (B) eq (9): |Phi_z(s) - 1| < 1 for all s in [0,h] (sufficient); (B') eq (10) only at s = h
  ("the condition used in [5]" = Bertrand-Philippe 2001); "(B) implies (A) whereas this is not the case for (B')".
  "Since it is very difficult to check (9), we apply the condition on the linear approximation" -> Condition (C) eq (12):
  |h| < 1/|Phi_z'(0)|; Phi_z'(0) = trace R(z) (eq 14) = f'/f.
- Ex 3.2 (A = lambda I_n): (B') satisfied for values that VIOLATE (A) -> (B') unreliable.
- Ex 3.3 (critical): Phi'(0) = 0, (C) satisfied for all h while (A),(B),(B') need |h| < 1/2.
- §4.2: Phi'(0) estimated by finite difference with extra LU (s = alpha h, alpha = min(1e-6 mu/|h|, 1)).
- §4.3: "we describe a HEURISTIC procedure for including new points": check (C) AND (B') simultaneously;
  (C) also at the right end ("backwardly"); eq (15) M points inserted.
- §4.4 Table 1 Algorithm Eigencnt; cost 2 LU per point.
- §5 Ex 5.1 ODEP400A: box with aspect ratio: 1519 intervals; correct already at 825 if (8) applied blindly.
  Ex 5.2 TOLS2000: 2611 / 15669 intervals. Ex 5.3 E40R5000 (n=17281, driven cavity Re 5000): 116 eigenvalues,
  7986 intervals, 54 h 42 min (eigs with p=200 then found all 116 in 10.2 s).
- §6: "reliable but it involves a high level of computation"; parallel version future work.
=> For us: winding_cert._march = (C)-type step (0.35|f|/|f'|) + (B')-type chord test at midpoint and end
   (|f2-fm| <= 0.75 min|f|) + phase increments <= pi/2. Same class as Eigencnt: validated, not proven
   (KP's own word: heuristic). KP's Condition (B) on the WHOLE segment is the rigorous criterion; it needs a
   bound on |Phi_z(s) - 1| over the segment.

## Bindel-Hood -- FULL TEXT arXiv:1303.4668v2 (4 Aug 2013); journal SIMAX 34(4):1728-1749 per Bindel page (search).
- §2 Lemma 2.1 (p2): T, E analytic, Gamma simple closed; if T(z)+sE(z) nonsingular for all s in [0,1], z in Gamma,
  then T and T+E have the same number of eigenvalues inside Gamma (homotopy of winding numbers).
- Remark 1: almost a special case of Gohberg-Sigal operator Rouche: ||T(z)^-1 E(z)|| < 1 on Gamma.
- §3 Thm 3.1: nonlinear Gershgorin with counting for disjoint unions.
- §4 Prop 4.1 (4.6)-(4.8): Lambda_eps(T) = {z: ||T(z)^-1|| > 1/eps} = union of spectra of analytic perturbations.
  Prop 4.2: bounded component of Lambda_eps contains an eigenvalue (subharmonicity).
  Prop 4.3: T(lam^)x = r, ||r||/||x|| < eps => lam^ in Lambda_eps(T)  (residual -> pseudospectral inclusion).
  Thm 4.4: Lambda(T+E) cap Omega_eps subset Lambda_eps(T) cap Omega_eps; a bounded component U of Lambda_eps(T)
  with closure in Omega_eps contains the same number of eigenvalues of T and T+E.
- §5 Cor 5.1 (pencil pseudospectra), Cor 5.2 (Gershgorin), Thm 5.3 (Bauer-Fike, nonlinear).
=> For us: (i) the tail closure (|kG| < q on |s| >= W_T) is a Rouche/Gohberg-Sigal argument with T = I, E = -KG;
   it must hold on the WHOLE closing arc in Re s >= line, not only on the line. (ii) Prop 4.3 + Thm 4.4 + the
   even pencil's mirror symmetry give a rigorous on-axis classification: a component of the eps-pseudospectrum
   that is symmetric about iR and contains exactly one eigenvalue forces that eigenvalue onto iR. [I]

## Benner-Mitchell-Overton CDC 2018 -- FULL TEXT arXiv:1803.06549v1 (17 Mar 2018), 10 pp.
- §3 eq (5a)-(5c): min ||G_r||_Linf s.t. alpha(A_r) < 0 AND alpha(A_f) < 0 (FULL-ORDER closed-loop stability).
- §3: spectral abscissa not locally Lipschitz at multiple eigenvalues; GRANSO (BFGS-SQP).
- p4-5: GRANSO infeasible iterates "worked poorly"; Alg 1 (stabilize then optimize, F(K)=inf outside),
  Alg 2 (loop: restabilize (A), constrained (B) until infeasible).
- §4 p6: alpha(A_f) by eigs: "not absolutely guaranteed to return the eigenvalues of Af with largest real part,
  it is generally quite reliable in practice"; eigs twice (A_f and A_f^T) for gradients.
- Table 1: HF2D FOM nx 2025-4489, ROM 256-484, nw = nx+2, nz = nx (full-state performance channel!), order-10 K.
- Table 2: ROM-only (hifoo, Alg1/2 R only) give F = inf (FOM unstable) on 5 of 12; R+F always finite;
  Alg 2 R+F best on 7 of 12; Alg 2 R+F on average 8.35x smaller F than Alg 1 R+F.
- §2 p3: hifoos via HEC "inconsistencies from one controller K to another, effectively implying that the function
  being optimized was discontinuous in K ... optimization would not infrequently halt prematurely".
=> LITERATURE.md l.33 groups BMO 2018 with methods that "assume no stability constraint": WRONG for BMO 2018.

## Mitchell 2021 (globality certificates) -- FULL TEXT arXiv:1910.01069v3 (22 Dec 2020), read §1.3, §2.2-2.3, §5 (part).
- Abstract: interpolation-based globality certificates, O(k n^3) work, O(n^2) memory; "embarrassingly parallel".
- §1.3 p5-6: 2D level-set tests "assume exact computation, but rounding errors in computed eigenvalues may cause the
  methods to fail"; new approach "does not crucially hinge upon any single computation"; trade-off: "we assume that
  approximation via interpolation is reliable enough".
- Thm 2.7 (p9): g_gamma(theta) continuous, = 0 iff an imaginary eigenvalue; mu(G(gamma)) > 0 if gamma above optimum.
- Alg 2.1 (p12): Chebfun interpolant; final check at argmins / midpoints between roots. Not a proof (assumes
  interpolant to machine precision).
- §2.3 p13-14: "By using an sHH structure-preserving eigensolver ... imaginary eigenvalues will have exactly zero real
  part"; but structure costs a generalized problem; hybrid: unstructured eig first; if the attaining eigenvalue is
  within tol of the axis, check level-set points (sigma) via Thm 2.1; if NOT detected, RECOMPUTE with a structure-
  preserving eigensolver "to either overcome any rounding errors or verify". "numerical reliability remains
  uncompromised" (their claim).
- §5 p21-24: dense, n <= 100 (orrsommerfeld) for K(A); MATLAB eig; prototype resorts to structured pencils only on
  inf/nan/errors; >1000x faster than one 2D level-set test at n = 55,100.
## Mitchell 2020 "Computing the Kreiss constant of a matrix" -- arXiv:1907.06537 (read §5.3 p13-14, Key Remark 6.3 p17).
- §5.3 p14: "how far away from the real axis can a computed eigenvalue be allowed to be while still being deemed
  a real eigenvalue? ... a reliable tolerance is tol * eps_mach * ||A2^-1 A1||_inf" (tol > 1 user-given).
- Key Remark 6.3 p17: "the lower bound is only true if no such pair of level-set points exist, not if the procedure
  fails to detect them due to numerical problems!"; redesign: assert only when the procedure generates no level-set
  points whatsoever (on the gamma-level set or lower), which removes the fragile matching step.

## Appeltans-Michiels -- FULL TEXT arXiv:2110.02665 (read abstract, §1, §2.2 p5-7). Secondary account of SHIRA [16]=MW 2001.
- §2.2 p5: standard shift-invert Arnoldi: "the obtained approximation for mu typically does not lie on the imaginary axis";
  "computed approximations for purely imaginary eigenvalues of H typically have a small but non-zero real part ...
  additional processing to determine whether an eigenvalue is purely imaginary, is necessary."
- eq (2.1): R_sigma^-1 = (H + sigma I)^-1 (H - sigma I)^-1, real skew-Hamiltonian for real/imaginary sigma; lambda, -lambda
  -> mu = 1/(lambda^2 - sigma^2) (double); Krylov space J-neutral ([16, Prop 3.3]); Prop 2.3: at most one vector of
  span{v+, v-} in K_m -> no doubled Ritz values; simple imaginary eigenvalue -> real simple Ritz value -> "typically strictly imaginary".
- p6-7: J-neutrality lost in finite precision; [16, §5] J-orthogonalisation eq (2.4) (orthogonalise against Q_m AND J Q_m).
=> cost: two shifted solves per application; needs a custom Arnoldi (ARPACK cannot J-orthogonalise) [I].

## Di Napoli-Polizzi-Saad -- FULL TEXT arXiv:1308.4275v2 (5 Aug 2014); read §1-2 (p1-4), §4.1 (p11), §5.7 (p20), §6 (p21).
- §1 p2: exact count via Sylvester inertia of LDL^T at two shifts (Hermitian).
- §2 eq (3) Hutchinson; eq (4) Pr(|T - tr| >= eps tr) <= delta; Lemma 2.1 (p4): nv >= 16 tr(P) ln(2/delta) -> Pr(round(T) != tr P) <= delta;
  p4: "the lemma gives a rather loose bound"; bias from approximating the projector "quite often" dominates.
- §5.7 p20: non-symmetric: "preliminary results" (qc324, n = 324).
- §6: exact count -> Sylvester inertia + direct solver; otherwise stochastic estimates.
=> certify_sparse.py docstring l.40-45 ("Hutchinson ... Equality certifies") would NOT be a certificate.

## Batra-Sharma, arXiv:1912.02820v1 -- read p10-11 only (secondary on YK and JT).
- p10: Ying-Katz [38] (Numer. Math. 53(1-2):143-163, 1988): adaptive quadrature on a polygonal chain; "reliable but may
  return 'Failure' (viz. [38], p.157)"; complexity not analysed. YK [39] (Computing 41(4):317-333, 1989): exclusion predicate
  with a second-derivative bound M(B) > |f''(B)|; Real-RAM analysis, finite precision not discussed.
- p11: Johnson-Tucker [22] (JCAM 228(1):418-423, 2009): winding-number enclosure, Newton on f^(k-1), Krawczyk validation, subdivision.

## Meerbergen-Wang arXiv:2411.02895v2 -- fetched, abstract/§1 read: singular pencils det == 0 (normal rank). Not our case (regular pencil). Dropped.

## HAL hal-00903743 (Philippe et al., Workshop 2013) -- record + abstract only (no PDF: /document 404):
  combines PAT (path following of sigma_min level curve) with EIGENCNT; one LU gives sigma_min(zI-A) and det(zI-A); multilevel parallelism.
## Nguenang-Kamgnia-Philippe, ARIMA 19 (2015), DOI 10.46298/arima.1983 -- search snippet only (episciences refused).

## Archive cross-checks (already-read papers, cross-check only)
- SMV (main-LinfVal) p1: BB/BS extended to E != I "in Benner et al. [2012]" via even pencil (2); p2: dense structured
  factorisations limited to q "a few thousand at most"; p2: "Freitag et al. [2014]" Newton -> local maximiser only;
  p3: scan inspired by complex frequency hopping (Chiprout-Nakhla 1995) and Grivet-Talocia [2007] (passivity), Gu et al. 2006,
  Kressner 2006; novelty = structure -> reliable distinction imaginary vs small real part;
  p5: K(xi) eq (5) from MW 2001 and MSS 2012 (singular N); K(xi) real for imaginary/real xi (MSS); exact zero real part;
  near-axis pair may be computed as imaginary pair -> sampling (Alg 2 l.11-19); p5-6: slivers < eps_eigs removed: "only occur
  due to errors in the eigenvalue computation"; 7 of 264 eigs exceptions without removal.
  SMV reference list does NOT contain Benner-Lowe-Voigt NACO 2018.
- Benner-Mitchell 2018 (17m1137966) §2: bisection "first proposed in [12] (BBK) ... inspired by ... [15] (Byers 1988)";
  §7: [20 GGO, 10 BV, 16 FSVD, 28 HEC, 1 ABMSV] converge to local maximisers only. Refs [12] MCSS 2:207-219; [16] SIMAX 35:619-635; [28] HEC.
- AN 2006 ref [10] = BBK MCSS 2(3):207-219 (1989).

## Code facts checked (HEAD 15cdfcb)
- certify_sparse.py l.348-350: scale = max(1, max_j |lam_j|) over the returned set; test abs(Re l) < 1e-8*scale (strict).
  l.357-364: r = min |lam - i theta| (no shrink); intervals < eps_eigs dropped (l.361-364, l.370). sigma not evaluated per shift.
- lattice.py l.92 LINE_SHRINK 0.9; l.115 LEVEL_IMAG 1e-8 per-eigenvalue (<=), l.2050; centres with sigma_1 >= gam flagged;
  slivers < hmin get finest lattice point (l.2343-2350), not dropped; l.592-595 disk only when all LEVEL_K converged.
- certify_sparse.py l.218: count header only (docstring l.40-45 promises Hutchinson count).
- winding_cert.py l.48-55 "validate but do not prove"; _march l.276-334 (0.35|f|/|f'|, phase <= pi/2, chord <= 0.75 min|f|).
  ctrl_exact roots via _poly_and_deriv (l.150-157); kmax_at Neumann with ||A_c||_2 (l.436-452).
- bfs/plant.py l.374-379 heights "certified upper bounds (inertia)"; gbound l.587-625 = min(H1, L2) (since 9bfdee0).
