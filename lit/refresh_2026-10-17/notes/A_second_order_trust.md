# Group A_second_order_trust -- raw notes (2026-10-17)

Downloads: (session scratchpad, not kept) pdf/A_second_order_trust/<key>/
Page numbers below = PDF page (form-feed count, script ../pageof.sh).

## Network findings
- Reachable: arxiv.org, hal.science (main portal only), doi.org.
- BLOCKED (403 at CONNECT or DNS): www.math.univ-toulouse.fr (Noll's PAPERS/ where nummath.pdf, 066519.pdf, pjo.pdf, MP2.pdf, jca.pdf, svva.pdf live),
  HAL portal subdomains (unilim.hal.science, inria.hal.science, hal-univ-tlse3...), hal.archives-ouvertes.fr, api.archives-ouvertes.fr, api.istex.fr,
  link.springer.com (non-OA redirect to idp.springer.com -> 403), cs.nyu.edu, ftp.cc.ac.cn / lsec.cc.ac.cn (Yuan), apps.dtic.mil, optimization-online.org,
  par.nsf.gov, core.ac.uk, semanticscholar, openalex, web.archive.org, theses.hal.science. HAL search page = robot block. WebFetch: DNS failure on Toulouse, unilim.hal, Springer.
- hal.science/hal-01868767 (Bompart 2007): record exists, NO file (/document 404); only metadata (Numer. Math. 107(3):433-454, 2007/08/15, doi 10.1007/s00211-007-0095-9); only link is an ISTEX pdf (blocked).
- hal.science/hal-00464695 (Noll-Prot-Rondepierre): record page fetched (metadata + abstract); file on unilim.hal.science (blocked). Metadata: Pacific J. Optim. 4(3):571-604, 2008. Abstract: "new way to manage the proximity control parameter, which allows us to handle nonconvex objectives"; "every accumulation point of the sequence of serious steps is critical"; tested on H-inf synthesis. [ABSTRACT only]

## ANR 2016 = arXiv:1504.00648v1 (SVVA 24(1):115-148, 2016 per search; arXiv title "Nonsmooth trust-region algorithm with applications to robust stability of uncertain systems") -- FULL TEXT
- Alg. 1 p.6: params 0<gamma<gamma~<1, 0<gamma<Gamma<=1, 0<theta<<1, M>=1.
  Step 3: inner radius R_1 = memory radius R#_j. Step 4: tangent program min phi_k(y,x) s.t. y in C, ||y-x||<=R_k; ANY trial z in C cap B(x, M||x-y||) with f(x)-phi_k(z) >= theta (f(x)-phi_k(y)).
  Step 5: rho_k = (f(x)-f(z))/(f(x)-phi_k(z)) >= gamma -> serious step.
  Step 6 null step: add cutting plane at z_k.
  Step 7: secondary rho~_k = (f(x)-phi(z_k,x))/(f(x)-phi_k(z_k,x)) [phi = ideal model, phi_k = working model];
          R_{k+1} = R_k if rho~ < gamma~ (model was the problem -> only add cut), = R_k/2 if rho~ >= gamma~.
  Step 8: memory radius R#_{j+1} = R_k if rho_k < Gamma, = 2 R_k if rho_k >= Gamma.  (NO "on the boundary" condition; radius never increased inside the inner loop.)
- Remark 7 p.5: y_k plays role of Cauchy point; Remark 9-10 p.5: polyhedral norm -> LP; cut limit n+1 by Caratheodory, Kiwiel aggregation not known for TR (later solved in ANR 2018).
- Thm 1 p.8: f with STRICT first-order model, bounded level set -> every accumulation point critical. (Model axioms M1-M3, strict M2~ p.2-3.)
- Sec 4 p.12-13 stopping: ||x_j-x_{j+1}||/(1+||x_j||) < tol1 and (f_j - f_{j+1})/(1+|f_j|) < tol2 and min(||g_j||,||g_j'||)/(1+|f_j|) < tol3 (aggregate subgradient); inner fallback kmax=50, nu_max=5; tol1=tol2=1e-5, tol3=1e-6. "Typical values ... gamma=0.0001, gamma~=0.0002, Gamma=0.1" (p.13).
- 5.1 p.13: full model phi_k=phi -> rho~=1 always, only radius reduction: "close to the rationale of the classical trust-region method".
- 5.2 p.13: natural model for g o F (g convex, F C1) is strict; full-model case = Ruszczynski [38, Ch 7.5] composite TR.
- 5.3 p.13-14: spectral model, cuts = whole subdifferential {Q Y Q^T}, tangent program is an SDP with polyhedral/conic norm.
- 5.4/Remark 11 p.15: standard model phi#(y,x)=f(x)+f°(x,y-x) -> y_k = nonsmooth steepest descent of length R; "Can we use ... as a nonsmooth Cauchy point? ... no we can't" unless phi# strict; Lemma 4/5 p.15-16 upper-C1 -> phi# strict; Thm 2 p.16: for upper-C1 f the CLASSICAL TR (one exactness plane, shrink at every null step) converges (subsequence; KL -> single limit).
- 5.7 p.17-19 counterexample: convex polyhedral f=max{-100, +-2x1+3x2, +-5x1+2x2} in R2; classical TR with standard model (Cauchy point, radius doubled at rho>=Gamma, kept for gamma<=rho<Gamma, halved on rejection) -> serious iterates converge to the origin, NOT critical.
- Sec 6 p.19-21: worst-case Hinf, robust stability, distance to instability; TR norm = |delta|_inf (box) to match Delta=[-1,1]^m; Thm 3: for the upper-C1 worst-case Hinf program, Alg. 1 coincides with classical first-order TR.
- Sec 7: 87 benchmark problems; vs WCGAIN 121x faster on average (Table 1).

## ANR 2018 = arXiv:1804.11320v1 (SVVA 26(2), 2018 per 1905.11511 [25]) -- FULL TEXT
- Alg. 1 p.4: Omega_opt chosen from initial K so that ||T||inf <= f(x0)+theta; Step 6 refine grid & re-solve.
- Alg. 2 p.6 "Non-smooth trust-region algorithm": params as ANR2016 + q>0. Second-order working model Phi_k = phi_k + 1/2 (y-x)^T Q_j (y-x), Q_j >= 0, ||Q_j|| <= q, Q updated ONLY at serious steps (Step 8). Step 7: rho~_k = (f(x)-phi_{k+1}(z_k,x))/(f(x)-Phi_k(z_k,x)), halve R iff rho~>=gamma~. Step 8: memory radius 2R_k if rho_k>=Gamma else R_k.
- Remark 4 p.6-7: max-norm TR -> tangent program is a convex QP (8): min t+1/2 (y-x)'Q(y-x), a_i+g_i'(y-x)<=t, -R<=y_i-x_i<=R.  (== our box-TR epigraph QP)
- Lemma 3 p.8-10: aggregation (3 planes) justified if Q > 0 (positive definite) and z_k=y_k; Remark 6-7, Lemma 4 p.11: fallback to y_k if z_k fails and rho~<gamma~.
- Remark 8 p.11: "In the classical trust-region method failure of the trial step always leads to reduction of the trust-region radius. One occasionally sees non-smooth versions where authors do the same. As we have already shown in [5, section 5.5], that must fail."
- Thm 1 p.11: bounded level set -> accumulation points critical (proof "adapted from [5]", skipped).
- 4.1 p.12: barrier max{||Twz||, c||S||}.
- 4.2 p.12: "anticipated cutting planes" = secondary peaks included in the initial model so "in the vast majority of cases the first trial step will then be successful".
- Remark 9 p.12: backtracking z_k = x + t(y_k-x), 0<t<1; every t>=theta automatically a trial point (convexity), smaller t may still qualify.
- 4.3 p.13-14: Lemma 5 grid bound with first-order bound M[.,.]; Lemma 6: phi is C2 near every peak; Thm 2 certificate gamma* >= ||T||inf - theta; finite-difference M in practice.
- 5.3 p.22: 3rd-order controller, x in R^14, A_K TRIDIAGONAL (redundant parametrisation, i.e. gauge directions retained).
- NO second-order convergence result; NO Q choice beyond "Q>=0, ||Q||<=q" in Alg. 2.

## Lewis-Wright = arXiv:0812.0423 (Math. Program. 158, 2016 per LITERATURE [S]; arXiv version dated Oct 25, 2018 on p.1 footnote) -- FULL TEXT
- Alg. 1 p.1 / Alg. 2 p.25 ProxDescent: tau>1, sigma in (0,1), mu_min>0. Subproblem min h(c(x)+grad c d)+mu/2|d|^2. Accept if h(c(x))-h(c(x+)) >= sigma [h(c(x)) - h(c(x)+grad c d)] AND |x+ - (x+d)| <= |d|/2 (correction step allowed but bounded); on accept mu <- max(mu_min, mu/tau) (i.e. allow longer step next), on reject mu <- tau mu.
- Thm 5.4 p.27 global convergence (transversality (4.3), prox-bounded h).
- Thm 5.5 p.30: identification: h convex continuous partly smooth relative to M at c(xbar), CQ (4.11) + strict criticality (4.20), x_r -> xbar => c(x_r)+grad c(x_r) d_r in M for all large r; and {mu_r} bounded.
- p.30: second-order tracking of M only suggested, not analysed ("We could then try to 'track' M using second-order information").
- Sec 3 p.9: Fletcher-Sainz de la Maza (Math Prog 43, 1989) l_inf-TR LP, identification Thm 2.3, then EQP on identified constraints; Byrd et al SLQP (KNITRO-ACTIVE); Yuan [52] Math Prog 31 (1985) 220-228 "Conditions for convergence of a trust-region method for nonsmooth optimization", [53] Math Prog 31 (1985) 269-285 "On the superlinear convergence ...": Cauchy-point analog by line search on piecewise quadratic, acceptability test for higher-order step. [B for Yuan]
- Sec 6: tau=1.25, sigma=0.01, mu_min=1e-4 (p.31 l.2101); tau=1.5, sigma=1e-3, mu_min=1e-3 (l.2189). "making the strategy to increase and decrease mu more adaptive" (p.30-31).

## Han-Lewis = arXiv:2111.15645v5 (SIOPT 33(1):36-62, 2023 per search) -- FULL TEXT (main results read)
- Def. 1 p.2 Survey Descent: k points, one per active piece; k-th subproblem = projected GD step with constraints l_sj + L/2|x-sj|^2 <= l_si.
- Def. 3 p.10 strong C2 max function: C2 pieces, AFFINELY INDEPENDENT gradients, POSITIVE DEFINITE Hessians (delta I <= Hess <= L I); (10) 0 in relint subdiff (nondegenerate).
- Thm 17 p.16 Q-linear survey convergence; Thm 19 p.17 R-linear values. Needs k = degree known; Remark 20 p.17 heuristic: estimate subdifferential dimension d by SVD of recent gradients, k=d+1. Unconstrained.

## Davis-Jiang = arXiv:2205.00064v3 (FoCM 2024, doi 10.1007/s10208-024-09653-y per search) -- FULL TEXT (statements read, proofs not)
- Thm 1.1 p.15: NTDescent (Alg. 4 p.13), randomized, local nearly-linear: O(log^3(1/eps)) oracle calls; Assumption A p.23: (A1) quadratic growth, (A2) C4 active manifold, (A3) strong-(a), (A4) (b<=)-regularity. Unconstrained; covers max-of-smooth, decomposable, generic semialgebraic (Thm 3.1, 3.3 p.24).

## Bareilles-Iutzeler-Malick = arXiv:2206.15053v3 (SIOPT 2023 per LITERATURE [S]; SIAM template "Submitted June 27, 2022") -- FULL TEXT
- Alg. 4.1 p.17: gamma_k halved each iteration; manifold from prox_{gamma g}(c(x_k)); SQP step on manifold (4.1) with exact Lagrangian Hessian; accept iff F decreases else stay.
- Thm 4.5 p.18: exact identification + local QUADRATIC convergence; needs strong minimizer (quadratic growth, fn.3), qualification, Assumption 4.2 (C2 with Lipschitz 2nd derivs, FULL-RANK Jacobian), 4.3 (stays in sublevel component), 4.4 (NO Maratos, assumed); local only, no globalisation; p.19-20: holds with any superlinear update (quasi-Newton) in place of SQP.
- Sec 5.2 p.20: second-order correction added to every SQP step in practice (Maratos).
- p.2-3 (secondary descriptions, full text): Mifflin-Sagastizabal 2005 [23] = proximal bundle (heuristic identification) + U-Newton (an SQP step), superlinear hinges on identification; Womersley-Fletcher 1986 JOTA 48 [33] pioneered max-of-smooth manifold + second-order steps, identification heuristic = indices of maximal pieces along a descent direction; Noll-Apkarian 2005 second-order [26] = U-Newton for max eigenvalue, "identification mechanism is a heuristic determining the multiplicity ... optimization step is an SQP iteration"; "None of these methods guarantee identification".

## Lewis-Wylie 2019 "A simple Newton method for local nonsmooth optimization" arXiv:1907.11742v1 -- FULL TEXT (algorithm + main thm)
- Alg. 2.2 p.4 k-bundle Newton: x^ = argmin sum lambda_s q_s(x) s.t. all linearisations l_s equal (U-space Newton with multiplier-weighted piece Hessians); k = cardinality of optimal active set GIVEN.
- Thm 5.11 p.18: strong second-order conditions (Def 5.2) + PD piece Hessians -> k-step quadratic. Alg. 7.1 p.23 weakly convex version.

## Lewis-Wylie 2021 "Active-set Newton methods and partial smoothness" arXiv:1902.00724v1 (MOR 46(2) per LITERATURE [S]) -- skimmed: Thm 5.1 p.9 active-set method for manifold intersection, quadratic (abstract framework, no algorithmic globalisation).

## Dao-Gwinner-Noll-Ovcharova = arXiv:1401.6807v1 (COAP 65(1):173-203, 2016 per search) -- FULL TEXT of Sec. 4 + Alg. 1
- Alg. 1 p.8 proximity control: tangent program min Phi_k + tau/2|y-x|^2 s.t. Ay<=b; Q_j may be INDEFINITE with -qI<=Q<=qI, Q+tau I > 0; Step 7: rho~ = (f(x)-Phi_{k+1}(y_k))/(f(x)-Phi_k(y_k)); tau_{k+1}=2tau if rho~>=gamma~ ("too bad"), else tau ("bad"); Step 8: memory tau# = tau/2 if rho>=Gamma ("good"), tau if gamma<=rho<Gamma; Step 9 large-multiplier safeguard tau<=T.
- Sec 4.5 p.6-7 rationale: rho~ ~ 1 => "adding the cutting plane" gives no progress => force smaller steps; rho~ << 1 => planes can still help => do not increase tau.
- Sec 8.2 p.16: Q = stiffness + Hessians of the ACTIVE nonsmooth pieces (piece-Hessian second-order model).
- Ref list: NPR PJO 4(3) 569-602 (vs 571-604 in ANR/1905.11511: page inconsistency); ANP SIOPT printed "10 (1)" (others: 19(1)).

## Burke-Engle = arXiv:1806.05218v2 (2019) -- FULL TEXT Sec 6
- Alg. 4 p.19 Global Trust Region for convex-composite h(c(x))+g(x): 0<g1<=g2<1<=g3, 0<b1<=b2<b3<1; r>b3 -> delta in [delta, g3 delta]; b2<=r<=b3 keep; else delta in [g1 delta, g2 delta]; accept iff r>=b1. No "on boundary" condition. Thm 6.1 p.20: {H_k} BOUNDED + Cauchy sufficient decrease (Def 6.2) -> |Delta^C_1 f(x_k)| -> 0.

## Curtis-Robinson-Zhou = arXiv:1708.02552v5 (2019) -- abstract + Thm 3.1 statement
- damped BFGS-type updates with self-correcting property (Thm 3.1 p.10): eigenvalues bounded on a fraction of iterations; p.8 l.490-495 standard damping alone does not give uniform eigenvalue bounds.

## Charisopoulos-Davis arXiv:2201.04611 -- abstract + Assumption A1 p.7 (sharp growth) only: NOT applicable (our records are not sharp: 13 tops vs 50 effective dims central, 1 vs 24 SISO).

## Archive cross-checks (already [R] in LITERATURE)
- 1905.11511 p.5 l.334-345 + Fig. 3: proximity-control flowchart (tau+ = 2tau / tau / tau/2 tests rho>=gamma, rho~>=gamma~, rho>=Gamma); refs p.10-11: [23] NPR PJO 4(3) 2008 571-604; [24] ANR SVVA 24(1) 2016 115-148; [25] ANR SVVA 26(2) 2018 405-429; [27] ANP SIOPT 19(1) 2008 281-306.
- IJRNC 2018 p.10 Remark 8 (primary step = standard step of nonsmooth TR [4,5]); Remark 11 (Cauchy point fails; success hinges on polyhedral model from cutting planes).
- CMO 2017 p.3 l.150-158: SQP-GS (Curtis-Overton [13], SIOPT 22 (2012) 474-500) uses BFGS in the QP; its convergence result REQUIRES upper and lower bounds on BFGS eigenvalues.
- AN 2006 TAC p.82 (l.865): "a second-order version ... is currently under investigation [51]" = Bompart et al.

## Search-snippet-only facts ([S], not read)
- Bompart 2007: snippet = abstract opening (Riccati alternative; nonsmooth mathematical programming; avoids Lyapunov variables). Nothing on the second-order algorithm. HAL metadata (read): Numer. Math. 107(3):433-454, 2007/08/15, doi 10.1007/s00211-007-0095-9.
- NA 2005 MP B 104:729-747: snippet: second-order techniques once multiplicity at limit point reliably estimated; standard and projected Newton steps combined with first-order nonsmooth method; "reasonable chance" of local superlinear/quadratic convergence.
- ANP 2008 SIOPT 19(1):281-306 doi 10.1137/060665191: snippet abstract: nonconvex max-eigenvalue + infinite maxima; accumulation points critical; feedback control experiments.
- Oustry 2000 MP 89:1-33 doi 10.1007/PL00011388 (INRIA RR-3738, 1999, inria-00072927): snippet: bundle (approximate eigenvalue method) globalises Overton's second-order method; quadratic convergence needs strict complementarity + non-degeneracy.
- MS 2005 MP 104(2-3):583-608: snippet: VU decomposition, bundle subroutine for approximate proximal points, superlinear if track points and U-Hessian approximated well enough; globally convergent with line search.
- BSS 1988 MP 42(2):273-306: "Parallel quasi-Newton methods for unconstrained optimization" (CU-CS-396-88): parallel BFGS / expensive function evaluations.
- Curtis-Overton 2012: line-search SQP with gradient sampling; convergence w.p.1.
- Yuan IMA JNA 4 (1984) 327-335: only a file name on Yuan's ftp; title unconfirmed.
