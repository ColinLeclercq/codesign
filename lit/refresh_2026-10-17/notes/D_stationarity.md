# Group D_stationarity -- raw notes (2026-10-17)

Downloads: (session scratchpad, not kept) pdf/D_stationarity/<key>/doc.pdf + doc.txt (pdftotext -layout).
Helper scripts (outside download dirs): ../scripts_D/get.sh ; page numbers = PDF page via ../pageof.sh (form-feed count).
Archive cross-checks: biblio/txt + page images rendered to ../scripts_D/img/ (AN 2006 p.79 and p.82).

## Network (this session)
- OK: arxiv.org (abs, pdf), doi.org (redirect only).
- BLOCKED (403 CONNECT or DNS): numerical.rl.ac.uk, pure.unamur.be, people.orie.cornell.edu, uwaterloo.ca, cs.nyu.edu,
  rcin.org.pl, drum.lib.umd.edu, link.springer.com/idp.springer.com (non-OA), coin-or.github.io, github.com (403), export.arxiv.org (DNS).
- WebFetch: DNS failure on people.orie.cornell.edu and numerical.rl.ac.uk.

## Accessed in FULL TEXT
### zhang2020 = arXiv:2002.04130v3 (ICML 2020, PMLR 119) -- 19 pp.
- Def 1-2 p.2: Clarke directional derivative / generalized gradient.
- Thm 5 p.3: for any algorithm using f-values + one generalized gradient per query, any eps in [0,1), any finite T, there is f in F(Delta,L)
  whose iterates contain no eps-stationary point (min ||g||, g in df(x), <= eps) with prob > 1/2.  (Black-box; zigzag functions.)
- Def 4 p.4: x is (delta,eps)-stationary if d(0, df(x+delta B)) <= eps, df(x+delta B) := conv(U_{y in x+delta B} df(y)) (Goldstein 1977).
- Prop 6 p.4: eps-stationary => (delta,eps)-stationary; for L-smooth f, (eps/3L, eps/3)-stationary => eps-stationary.
- Lemma 7 p.4: lim_{delta->0} df(x+delta B) = df(x).
- Thm 8 p.5: INGD finds (delta,eps)-stationary point with prob 1-gamma in O(Delta L^2/(eps^3 delta) log(4Delta/(gamma delta))) calls.
- Thm 11 p.7: Omega(Delta/(8 delta)) lower bound with gradient-only oracle.
- p.4: cites Shamir 2020: finding a point delta-close to an eps-stationary point needs exponential dependence on dimension.

### gs_survey2020 = arXiv:1804.11003v1 (Burke, Curtis, Lewis, Overton, Simoes; "Submitted to Special Methods for Nonsmooth Optimization, Springer, 2018") -- 18 pp.
- Alg GS p.4: sample m >= n+1 points in B(x_k, eps_k), g_k = min-norm element of conv of gradients (QP (2.3)); terminate if ||g_k|| <= nu_opt and eps_k <= eps_opt
  (Step 4); if ||g_k|| <= nu_k reduce nu, eps (Steps 5-6); Armijo line search (2.1).
- p.3: differences vs BLO05: Kiwiel's min{t_k,eps_k} (drops compact level set), termination tolerances (used in BLO05 experiments but not in its statement), non-normalised direction.
- (3.2) p.5: Clarke eps-subdifferential = conv d f(B(x,eps)) (Goldstein); Clarke eps-stationary if 0 in it.
- Thm 3.1 p.5: nu_opt=eps_opt=0, theta<1 => w.p.1 f->-inf or every cluster point Clarke stationary (= Kiwiel 2007 Thm 3.3; needs D open with FULL MEASURE, p.5 footnote 3 oversight in BLO05/Kiw07).
- Cor 3.1 p.6: positive tolerances => terminates.  Thm 3.2 p.6: fixed eps => cluster points Clarke eps-stationary; finite termination open.
- §3.2 p.6-7: linear rate for finite-max functions via VU (HSS17), "good" samples.
- p.6 and §4.3 p.9: main weakness m >= n+1 gradients/iteration; Curtis-Que 2013 adaptive sampling keeps guarantees with a CONSTANT number (independent of n).
- §4.4 p.9: second-order variant q_k(d) = f(x_k) + max_{g in G_k} g'd + 1/2 d'H_k d  (= our epigraph QP with sampled instead of explicit pieces).
- §5.2 p.10: SQP-GS (Curtis-Overton 2012) penalty-SQP, steering; limit points stationary for penalty or for infeasibility.
- §6 p.11: HANSO = BFGS then GS "verifying a loose measure of local optimality".

### drusv_lewis2018 = arXiv:1602.06661v2 (Drusvyatskiy-Lewis, "Error bounds, quadratic growth, and linear convergence of proximal methods") -- 35 pp.
- Abstract p.1: "short step-lengths in the algorithm indicate near-stationarity, suggesting a reliable termination criterion."
- §5 (5.1) p.13: min phi = g + h(c(x)), g closed convex, h finite convex, c C1.  dphi = dg + grad c^T dh(c(x)).
- p.14: phi_t(x;y) = g(y) + h(c(x)+grad c(x)(y-x)) + ||x-y||^2/(2t); G_t(x) = (x - x_t)/t; G_t = 0 iff stationary. (5.2): |phi(y)-phi(x;y)| <= (L beta/2)||x-y||^2.
- Alg 1 p.14: prox-linear, backtracking t <- qt until phi(x_t) <= phi(x) - (sigma/2)||G_t||^2; stop when ||G_t|| <= eps.
- (5.6) p.15: decrease >= ||G_t||^2/(2 L beta); min_i ||G_t(x_i)||^2 <= 2 L beta (phi(x_1)-phi*)/k.
- Thm 5.3 p.16: exists xhat: ||x_t - xhat|| <= ||x_t - x||; phi(xhat)-phi(x_t) <= (t/2)(L beta t + 1)||G_t||^2; dist(0, dphi(xhat)) <= (3 L beta t + 2)||G_t(x)||.  (Ekeland, Thm 5.2 p.15)
- Def 5.4 p.16: error bound dist(x, (dphi)^-1(0)) <= gamma ||G_t(x)|| near xbar.  Thm 5.5 p.17 linear convergence.
- Thm 5.6 (5.7) p.18: (1/2)||G_t(x)|| <= dist(0; dphi(x)).

### drusv_paquette2019 = arXiv:1605.00125v6 (Drusvyatskiy-Paquette, "Efficiency of minimizing compositions of convex functions and smooth maps") -- 58 pp.
- (1.1) p.1: F = g + h o c.  p.2: "The prox-linear method was recently investigated in [13 = CGT 2011, ...]"; complexity of ||G|| <= eps "see e.g. [13,23]".
- Example 3.3 p.7: exact penalty formulation min f + lambda theta_K(G(x)) is of the form (3.1).  Example 3.5: grey-box (h known, c expensive).
- Thm 3.1 p.9 chain rule dF = dg + grad c* dh(c(x)); (3.3) p.9: dist(0; dF(x)) = -inf_{||v||<=1} F'(x; v).
- §3.3 p.9-10: Alg 1 prox-linear; optimality of subproblem G_t(x) in dg(S_t(x)) + grad c(x)* dh(c(x)+grad c(x)(S_t(x)-x)); Prop 3.3 p.10: min_j ||G_t(x_j)||^2 <= 2 t^-1 (F(x0)-F*)/N.
- p.11 (+ footnote 5): for nonsmooth h, dist(0; dF(x_{k+1})) "will typically not even tend to zero" although G -> 0: f(x) = |x^2 - 1|, x0 > 1, x_k -> 1, f'(x_k) -> 2.
- Thm 4.5 p.14 + (4.8)-(4.9) p.15 (t = 1/mu, mu = L beta): xhat = prox_{F/2mu}(x): ||xhat - x|| <= (2/mu)||G_{1/mu}(x)||, F(xhat) <= F(x), dist(0; dF(xhat)) <= 4||G_{1/mu}(x)||.
  "Thus if ||G|| is small, the point x is near some point xhat that is nearly-stationary"; xhat not computable, only certifies.
- §5 p.15-16: inexact subproblems; "If primal-dual interior-point methods are applicable, then termination based on near-optimality in function value is most appropriate."

### cgt2020_composite = arXiv:2001.10802 (Cartis-Gould-Toint, "Strong evaluation complexity bounds for arbitrary-order optimization of nonconvex nonsmooth composite functions") -- 32 pp.
- (2.2) p.3: phi^delta_{f,j}(x) = f(x) - min_{||d||<=delta} T_{f,j}(x,d); (2.4) strong approx minimizer phi^delta_j <= eps_j delta^j/j!.
- (2.6) p.4: with inexpensive constraints F: min over x+d in F, ||d|| <= delta.
- AS.3 p.5: h Lipschitz, subadditive, h(0)=0 (norms, max(0,.) ...).
- (3.11) p.6: composite phi^delta_{w,j}(x) = w(x) - min_{x+d in F, ||d||<=delta} [T_{f,j}(x,s) + h(T_{c,j}(x,s))]  ("h left unchanged" in the measure).
- Lemma 3.2 p.6-7: h convex, x* (global) minimizer => phi^delta_{w,1}(x*) = 0 for all delta > 0 (proof is local).
- p.7: does not extend to q = 2 (counterexample).  (4.1) p.8 termination phi^delta_{w,j}(x_k) <= eps_j (scaled by delta^j/j! in (2.4)).
- Ref [6] p.31 = CGT 2011 SIOPT 21(4):1721-1739 (cited p.2 for penalty methods).

### cgt2019_orders = arXiv:1705.07285v2 (Cartis-Gould-Toint, J. Complexity 53, per search) -- 32 pp.
- footnote 3 p.10: "phi^Delta_{psi,1}(x)/Delta = chi_{psi,1}(x) as defined in [17, eq. (2.4)], irrespective of Delta in (0,1]" ([17] = CGT arXiv:1705.04895, the chi criticality measure).
- p.2: scaled KKT conditions (multiplier size) vs unscaled.

### lewis_wylie2021 = arXiv:1902.00724v1 (Lewis-Wylie, "Active-set Newton methods and partial smoothness") -- 21 pp.
- §1 p.1-2: partial smoothness introduced in [17] (= Lewis 2002); two phases: identification then Newton-type local phase; partial smoothness is a constant-rank property [18].
- Thm 3.1 p.4 semi-linearization error; Thm 5.1 p.9 active-set method for manifold intersection (transversal intersection);
  Def 5.3 p.11 partly smooth mappings; Thm 5.5 p.12 active set method for generalized equations; §6 p.15-17 partly smooth functions (symmetric second derivatives).
- Abstract only general; no max-function numerics. Venue (MOR 46(2), 2021) NOT stated in arXiv v1.

### bareilles2023 = arXiv:2206.15053v3 (Bareilles-Iutzeler-Malick, "Harnessing structure in composite nonsmooth minimization"; SIOPT 33(3):2222-2247 per dblp snippet) -- 26 pp.
- (1.1) p.1: F = g o c, g nonsmooth with explicit prox; running examples: max of smooth functions, lambda_max of parametrized matrix.
- Ex 2.3 p.5: prox_{gamma max}(y)_i = s if y_i > s else y_i, with s: sum_{y_i>s}(y_i - s) = gamma (water-filling).  Ex 2.4 p.5: prox of lambda_max via eigenvalues.
- Thm 3.2 p.9: near a qualified point, prox_{gamma g}(c(x)) in M_g for gamma in [phi(dist_M(x)), Gamma]; in particular L||x-x*|| <= gamma <= Gamma => correct structure.
- Alg 4.1 p.17: gamma halved each iter; structure from prox; SQP step (4.1) on smooth extension with Lagrangian Hessian, multipliers by least squares (4.2); accept if F decreases.
- Ass 4.2-4.4 p.18 (C2 extension, full-rank manifold Jacobian; stay in sublevel component; no Maratos); Thm 4.5 p.18: x* STRONG minimizer (footnote 3: F(x) >= F(x*) + eta||x-x*||^2) => M_k = M* and quadratic convergence.
- §5 p.20-21: maxquad and eigenvalue (multiplicity 3) examples, machine precision in 3 iterations.

### mehlitz2020 = arXiv:2006.09734 (Mehlitz, "Asymptotic stationarity and regularity for nonsmooth optimization problems", J. Nonsmooth Anal. Optim. 1 (2020) 6575 per page header) -- 30 pp.
- Def 3.1 p.7: AM-stationary: x_k -> x, eps_k -> 0, y_k -> 0 with eps_k in df(x_k) + D*Phi(x_k,y_k)(lambda_k); no bound on lambda_k.
- Thm 3.2 p.7: every local minimizer is AM-stationary, WITHOUT any CQ.
- Lemma 3.4 p.9: bounded lambda_k => M-stationary (KKT-type); unbounded => FJ-type (0 in D*Phi(x,0)(lambda), lambda != 0).
- Thm 3.10 p.12: Phi polyhedral => every feasible point AM-regular (so M-stationarity is necessary; linear constraints need no CQ).

### royset2023 = arXiv:2201.05250 (Royset, "Consistent approximations in composite optimization"; Math. Program. 2023 per snippet) -- 31 pp.
- p.3: term after Polak [42, Chap. 3-4]: epi-convergence of approximating objectives + outer epi-limit of approximating optimality functions (after sign change);
  applications to stochastic optimization, SEMI-INFINITE programming, nonsmooth optimization, optimal control [51,54,27,41,55]; Polak's concept ~ weak consistency.
- p.5: S(x,y,z) multiplier form of optimality; Prop 2.1 p.6 necessary under qualification (2.2).
- Def 2.2 p.7: consistent approximations: phi^nu ->e phi and S^nu ->g S; weakly consistent: LimOut gph S^nu subset gph S.
- Prop 2.3 p.8: (a) cluster points of eps_nu-argmin are argmin; (c) LimOut (S^nu)^{-1}(B(0,delta_nu)) subset S^{-1}(0): cluster points of delta_nu-stationary points (delta_nu -> 0) are stationary.
- Thm 2.4 p.9: sufficient conditions.  Example 3.3 p.15: min_x max_{p in A} sum p_i f_i(x) with A^nu ->s A: consistent (finite m).

## Archive cross-checks (permitted)
- AN 2006 TAC p.79 (image): §VI-D "follows the line of [36, Th. 2.1.6]" ([36] = Polak 1997 book). (19): theta(x) = inf_h sup_i sup_{Y_i} {-f(x) + mu_i(X) + Tr(Y_i Q_i^H [F'(x)h] Q_i) + (sigma/2)||h||^2};
  Q_i spans the invariant subspace of the FIRST k_{i+1}-1 eigenvalues (nested, all eigenvalues >= mu_i). theta <= 0. Lemma 3: theta = 0 iff 0 in df.
  (20) dual: sup_{tau>=0, sum tau=1} sup_Y {-f + sum tau_i mu_i - (1/2sigma)||sum tau_i F'* Q_i Y_i Q_i^H||^2}. (21) h(x) = -(1/sigma) sum tau_i F'* Q_i Y_i Q_i^H. Lemma 4 continuity.
  NO constraints in (19).
- AN 2006 p.82 (image): (23) inf{||g|| : g in df(x)} < eps1 (exact Clarke set, not theta); (24) relative H-inf progress < eps2(1+||T||); (25) ||K+ - K|| < eps3(1+||K||);
  eps1 = 1e-5, eps2 = eps3 = 1e-3; "we required that either the first two tests or the third one are satisfied" => (25) ALONE stops; enriched set capped at 50.
- Kiwiel 1985 Ch. 2 §3 Alg 3.1 (text p.64): dual subproblem (3.1) min 1/2|sum lambda_j g^j|^2 + sum lambda_j alpha_j (linearization errors); (3.2) v_k = -{|p_k|^2 + alpha_p};
  (3.3) w_k = 1/2|p_k|^2 + alpha_p, stop if w_k <= eps_s; Lemma 4.4 (4.7a-c) v_k <= -w_k <= 0; Remark 4.5 "w_k may be termed a stationarity measure".
  => AN's (20) with sigma = 1 and pieces evaluated at x (alpha_i = f - mu_i) is EXACTLY -w (theta = -w).

## Derivations [I] (mine)
1. theta_sigma(x) = -min_tau [sum tau_i (f - s_i) + (1/2sigma)||p_tau||^2], p_tau = sum tau_i g_i;  h* = -p*/sigma;  prox-linear G_t (t = 1/sigma) = p* for the finite-piece model.
   So ||G_t|| = ||p*|| (aggregate subgradient of the theta-QP) and -theta >= (t/2)||G_t||^2. Kiwiel w, AN theta, DL/DP G_t: one object.
2. Box LP: pred(D) = max_i(s_i) - min_{||d||_inf <= D} max_i (s_i + g_i d) [+ linearized rows]. Convex model => pred(D)/D nonincreasing in D; for D < (min offset)/(2 max||g_i||_1), pred(D)/D = min_{lambda in simplex(A0)} ||sum lambda g + N||_1 (exactly active set).
3. Goldstein vs eps-active: phi(.,w) L_f-Lipschitz, grad L_g-Lipschitz in z: d_delta f(z) subset conv{grad phi(z,w): phi(z,w) >= f(z) - 2 L_f delta} + L_g delta B.
   => rho_level(eps_abs = 2 L_f delta) <= dist(0, d_delta f(z)) + L_g delta. rho large at eps certifies NOT (eps/(2L_f), rho - L_g eps/(2L_f))-Goldstein; small rho certifies nothing.
4. Tied companion form M = [[a, om+g],[a, g]]: tr = a+g, det = -a om, det(M - L I) = L^2 - a(L+om) - g L: both RH rows LINEAR in (a,g); gradients (1,1), (-(L+om), -L) independent iff om != 0.
   M = L I needs a = L and a = 0: impossible for L = -0.05 => the MFCQ-failure point of the modal form is not representable. Every (tr, det) reachable: a = -det/om, g = tr - a.
   Controllability of (M, [1;1]): det[b, Mb] = -om != 0.
5. sigma_1 multiplicity block restricted to top-r singular subspace: lambda_max(X+E) = mu_1 + lambda_max(Q'EQ) + O(||E||^2/gap) => compressed model is first-order exact with curvature ~ ||dT||^2/gap(sigma_r - sigma_{r+1}): small gap to sigma_3 => small validity radius.

## NOT accessed
- Cartis-Gould-Toint SIOPT 21(4) 2011 (numerical.rl.ac.uk and pure.unamur.be 403). Bibliography confirmed in 3 full texts. Measure type confirmed via CGT 2019/2020 full texts.
- Hare-Sagastizabal-Solodov COAP 63(1):1-28, 2016, doi 10.1007/s10589-015-9762-4 (Springer paywall; no preprint found). Bibliography confirmed (citing arXiv 2303.14896 ref [13]).
- Lewis SIOPT 13(3):702-725, 2002 (Cornell, Waterloo 403). Only as cited by Lewis-Wylie [17].
- Burke-Lewis-Overton SIOPT 15(3) 2005 (Cornell, NYU, optimization-online 403). Content only via the GS survey.
- Hettich-Kortanek SIAM Rev 35(3):380-429, 1993 (SIAM). Search snippet only.
- Polak 1997 book, Polak SIAM Rev 1987: not accessed; via AN 2006 p.79 and Royset 2023 p.3 only.
- Overton SIMAX 9(2):256-268, 1988 (cs.nyu.edu 403). Snippet only.
- Han Math. Program. 20 (1981): not accessed.  Bonnans-Shapiro 2000: not accessed.
- Zhou-Tits SIOPT 6(2):461-487, 1996 (+ erratum SIOPT 8(1) 1998) and UMD TR 93-83: DRUM 403; snippet only.
- Helou-Santos-Simoes SIOPT 30(2):1610-1637, 2020: optimization-online 403; snippet only.
- Kiwiel 2007 (GS convergence) report at rcin.org.pl: 403.
- IPOPT docs (coin-or.github.io): 403.
