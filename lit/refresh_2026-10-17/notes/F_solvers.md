# Group F_solvers -- raw notes (2026-10-17)

Downloads: (session scratchpad, not kept) pdf/F_solvers/<key>/
Access tags: FULL = full text read; ABS = abstract/landing page only; NONE = not accessed; META = package metadata.

## Installability (checked 2026-10-17, package metadata only; nothing executed)
- conda-forge ipopt-feedstock recipe/meta.yaml: Ipopt 3.14.20, host deps libblas, liblapack, mumps-seq, libspral [linux], ampl-asl. META.
- conda-forge cyipopt-feedstock meta.yaml: cyipopt 1.7.0, run deps ipopt>=3.13, numpy>=1.26.4, python; run_constrained scipy>=1.11.4. META.
- conda-forge mumps-feedstock recipe/recipe.yaml (5.8.2): mumps-seq builds only for mpi=='nompi'; its libs are suffixed _seq
  (dmumps_seq, zmumps_seq, mumps_common_seq, pord_seq, mpiseq_seq; headers under include/mumps_seq/) -> designed to coexist
  with mumps-mpi (PETSc's) in one env. Both carry run_constraint libopenblas * *openmp*. META.
- conda-forge channeldata (current versions): ipopt 3.14.20, cyipopt 1.7.0, pyoptsparse 2.16.1, casadi 3.7.2, mumps-seq/mpi 5.8.2,
  petsc 3.26.0, clarabel 0.11.1, osqp 1.1.3, daqp 0.10.3, piqp 0.6.4, proxsuite 0.7.3, highs 1.15.1, nlopt 2.11.0.
  NOT on conda-forge: knitro, snopt, worhp, pygranso, paropt, coinhsl. META.
- PyPI cyipopt 1.7.0: sdist only (no wheel) -> pip needs a system Ipopt + pkg-config. META.
- PyPI casadi 3.8.1: manylinux_2_28 x86_64 abi3 wheel (89 MB). Listed (zip namelist only, not executed): bundles libipopt.so.3.14.19,
  libsipopt, libcoinmumps.so.3.0.12 (sequential MUMPS from COIN ThirdParty), libcoinmetis, libhighs, libosqp, libblasfeo.
  NLP plugins: alpaqa, ampl, blocksqp, bonmin, ccopt, conopt, fatrop, feasiblesqpmethod, ipopt, knitro, madnlp, qrsqp, scpgen,
  sleqp, snopt, sqpmethod, uno, worhp (knitro/snopt/worhp/conopt need the licensed shared library at run time).
  Conic plugins (no clarabel): cbc, clp, cplex, daqp, fatrop, gurobi, highs, hpipm, ipqp, mosek, nlpsol, osqp, piqp, proxqp, qpoases, qrqp, superscs, xpress. (old line:) daqp, highs, hpipm, ipqp, mosek, osqp, piqp, proxqp, qpoases, qrqp, superscs, gurobi, cplex.
  => LESSONS l.78 "IPOPT ... cannot be installed with pip" is outdated: pip install casadi gives Ipopt+MUMPS; conda-forge gives cyipopt.

## Network (2026-10-17)
- Reachable: arxiv.org (abs/pdf), hal.science record pages (search page = robot block), raw.githubusercontent.com, pypi.org,
  files.pythonhosted.org, conda.anaconda.org.
- BLOCKED (403 CONNECT): link.springer.com (idp redirect), dominoweb.draco.res.ibm.com (Ipopt RC23149), users.iems.northwestern.edu
  (Waechter thesis), ccom.ucsd.edu (SNOPT SIGEST pdf), www.kybernetika.cz (Luksan IPM), drum.lib.umd.edu (Zhou-Tits TR 93-83),
  kop.ior.kit.edu (Stein 2012), www2.eecs.berkeley.edu (Polak tech reports), foreman.virt.dacs.utwente.nl (Still SIP survey),
  skoge.folk.ntnu.no (Grimholt), optimization-online.org, coin-or.github.io, cyipopt.readthedocs.io, artelys.com, docs.mosek.com,
  worhp.de, docs.scipy.org, api.anaconda.org, export.arxiv.org. gh api: repo access not enabled. WebFetch: DNS failure (ccom.ucsd.edu).

## Ipopt 3.14 documentation (GitHub coin-or/Ipopt stable/3.14 doc/*.dox) -- FULL (relevant sections)
- main.dox l.43-44: objective and constraints "should be twice continuously differentiable".
- special.dox QUASI_NEWTON (l.96-130): hessian_approximation=limited-memory (L-BFGS); exact 2nd derivatives "usually a good idea"
  ("fewer iterations and more robust").
- special.dox AMPL_WARMSTART (l.132-201): warm_start_init_point, warm_start_bound_push, warm_start_mult_bound_push, mu_init ~1e-6;
  "the use of this feature is far from solving the complicated issue of warm-starting interior-point algorithms";
  useful only if active set preserved; "If active-set changes are observed ... might not decrease the number of iterations (in some
  cases, it might even tend to increase the number of iterations)".
- options.dox: limited_memory_max_history default 6 (l.1406-1410); update bfgs (with skipping), sr1 "not working well" (l.1412-1420);
  limited_memory_initialization scalar1 sigma = s'y/s's (l.1422-1433) (= scaled L-BFGS); hessian_approximation default exact
  (l.1467-1475); max_soc default 4 (second-order corrections, Step A-5.9 of Alg. A of implementation paper) (l.994-998);
  linear_solver default ma27; options ma27/57/77/86/97, pardiso, pardisomkl, spral, wsmp, mumps (l.1123-1137);
  jac_d_constant / jac_c_constant / hessian_constant (l.393-411); dependency_detector "experimental and does not work well" (l.305-307);
  warm_start_same_structure (l.543-550); mu_strategy default monotone (l.722-724); accept_every_trial_step disables line search (l.878-884).
- install.dox l.283-315 (DOWNLOAD_MUMPS): "The non-MPI version of MUMPS uses internally a fake implementation of MPI, which can
  prevent using Ipopt together with MUMPS within an MPI program." -> must smoke-test inside the mpi4py farm, or linear_solver=spral
  (conda-forge ipopt links libspral on linux), or run Ipopt in a separate non-MPI process.

## Clarabel: Goulart & Chen, arXiv:2405.12762v1 (21 May 2024) -- FULL (sections 2.4, 2.6, 3.3 read; benchmarks skimmed)
- 2.4 p.7: Ruiz equilibration on all matrix data before solving. Initialisation 2.4.1-2.4.2 pp.7-8 (no user warm start; grep "warm": none).
- 2.6 p.9: termination on unscaled data; eps_f = 1e-8 default; weaker 1e-5 "near optimality" on early termination
  (lack of progress, timeout, iterations) -> AlmostSolved status.
- 3.3 p.12: static regularisation eps_s, dynamic pivot regularisation; QDLDL or faer supernodal LDL (p.13 l.722-733).
- Clarabel.rs settings.rs (GitHub, read): max_iter 200, tol_gap_abs/rel 1e-8, tol_feas 1e-8, reduced_tol_* 5e-5/1e-4,
  equilibrate_enable true, equilibrate_max_iter 10, equilibrate_min_scaling 1e-4, equilibrate_max_scaling 1e4,
  static_regularization_constant 1e-8. No warm-start setting.
  -> LESSONS l.1083 (failing QPs solve with equilibration off or bounded to [1e-2,1e2]) = narrowing the default [1e-4,1e4] bounds.

## WARP (Suri, Hilmarsson, Bose), arXiv:2605.05728v1 (May 2026; NeurIPS 2026 D&B) -- FULL (relevant parts)
- p.1-2 (abstract, contributions): primal-only warm starts do not reduce IPOPT iterations vs the default bound-midpoint start; x* without
  duals "causes the solver to diverge"; full state (x*, lambda*, z*, mu*) reduces IPOPT iterations 23 -> 3.
- Table 2 p.4 (IPOPT via cyipopt, exact Hessian, case118, 50 instances): cold 22.6 mean iters; x* only 23.7 (-4.9%); x* with cold
  dual init >50 (diverges); ...; full state 3.3 (+85%). p.2 l.106-107 warm-start config warm_start_bound_push 1e-20, mu_strategy monotone.

## Uno: Vanaret & Leyffer, "Implementing a unified solver for nonlinearly constrained optimization", arXiv:2406.13454v2 (Nov 2025) -- FULL (relevant parts)
- Abstract p.1: framework of 8 building blocks; competitive vs filterSQP, IPOPT, SNOPT, MINOS, LANCELOT, LOQO, CONOPT on 429 small CUTE problems; MIT license.
- 4.6.2 p.13: TR rule: increase Delta if TR active at accepted step; else decrease to a value smaller than min(Delta, ||d||).
- 5.3 p.18 l.1189-1192: if Hessian model "exact" but no Hessian provided -> set to "zero" "(as long as quasi-Newton approximations are not implemented in Uno)".
- Table 6 p.35: solver taxonomy: filterSQP (feas. restoration, filter, l_inf TR, exact Hessian, SQP); IPOPT (feas. restoration, primal-dual IPM,
  filter, line search, exact/QN, primal-dual inertia correction); KNITRO-ASM (l1 relaxation, EC, l1 merit, TR, exact, SLQP); KNITRO-IPM
  (step decomposition, IPM, l2 merit, LS+TR, exact); SLSQP (RHS relaxation, IC, l1 merit, LS, QN, SQP); SNOPT (l1 relaxation, IC, AL merit, LS, QN, SQP);
  WORHP (RHS relaxation, IC, l1 merit/filter, LS, QN, SQP); SQuID (l1 relaxation, l1 merit, LS, exact, Sl1QP).
- p.21 l.1373-1376: comparison set-up: filterSQP 20010817 (BQPD), IPOPT 3.14.20 with MA57 (exact Hessians, AMPL), SNOPT 7.5-1.2.
- Table 5 pp.29-34: number of OBJECTIVE EVALUATIONS on CUTE NLPs. Minimax-type epigraph problems (21 rows parsed by
  scripts/uno_minimax.py: polak1-6, rosenmmx, minmaxbd, minmaxrb, madsen, makela1-3, mifflin1-2, chaconn1-2, cb2-3, demymalo, womflet,
  kiwcresc ...; n <= 10, m <= 20 per Table 1 p.22): median evals filterSQP 9 (max 45, 0 fail); IPOPT 11.5 (max 660 polak6, 1 fail polak3);
  SNOPT 13 (max 288 makela3); LOQO 14; CONOPT 14; MINOS 30; LANCELOT 28; Uno-ipopt preset 24 (max 24451 polak1).

## Kiessling, Leyffer, Vanaret, "A unified funnel restoration SQP algorithm", arXiv:2409.09208v1 (Sep 2024) -- FULL (sections 1-2)
- 2.2.1 p.6 l.250-276: l_inf TR "most easily implemented within a QP subproblem"; radius reduced until trial accepted; "Usually, Delta is
  increased in successful iterations if the trust region is active at d, and it is decreased to a value smaller than min(Delta, ||d||_inf)
  when the trial iterate is rejected."; PD Hessian not required with TR.
- 2.2.2 p.6 l.278-288: line search: "no additional QPs need be solved if a trial iterate is rejected. On the other hand, the step
  alpha d is usually not optimal (or even feasible) within the equivalent trust region ||d||_inf <= alpha ||d||_inf."

## Luo & Waechter, "A quadratically convergent sequential programming method for second-order cone programs capable of warm starts",
## arXiv:2207.03082v2 (Jul 2023) -- FULL (abstract, sec 1, sec 3 skim, sec 5 skim)
- Abstract p.1: SQP framework for linear SOCP; active-set QP subproblems -> warm starts; cones approximated by polyhedral outer
  approximations refined over iterations; nondegenerate instances: extremal-active cones identified, final steps = regular SQP -> local quadratic.
- sec 5 pp.25-29: MATLAB + CPLEX QPs (p.26 l.1507-1513); warm starts from MOSEK solutions / perturbed instances (Tables 2-4 p.28-29).

## Seidel & Kuefer, "An adaptive discretization method solving semi-infinite optimization problems with quadratic rate of convergence",
## arXiv:1910.13798v1 (Oct 2019; Optimization 2022 per search) -- FULL (sec 1-2, sec 4 statement)
- sec 1 p.2: Blankenship-Falk [2] (JOTA 19(2), 1976): optimisation step on Y^k, refinement adds the most violated index; p.3: in general no
  quadratic rate, "For the 'final' steps towards a higher accuracy many iterations are needed. Not only the number of iterations grows, but also
  the number of discretization points added." Example 2.1 p.4: linear convergence.
- Algorithm 1 quADAPT p.7: add global solution y^{i,k} AND derivative information Dy^i(x^k), Dmu^i(x^k) to build an extra constraint
  (local-reduction style) -> quadratic rate (sec 4, p.12-14). p.14 l.738-746: multiple active indices per constraint -> compute ALL local
  solutions or no quadratic rate.
  -> supports the project's followed tops (peak following = lower-level solution tracking) and never-retired Omega.

## Asl & Overton, "Behavior of limited memory BFGS when applied to nonsmooth functions and their Nesterov smoothings",
## arXiv:2006.11336v1 (Jun 2020) -- FULL (abstract, sec 2-3 skim)
- Abstract p.1: scaled L-BFGS "often breaks down with a poor approximation to an optimal solution, in sharp contrast to full BFGS";
  unscaled L-BFGS far more function evaluations per iteration; "when the condition number is so large that the function is effectively
  nonsmooth, L-BFGS frequently fails"; better to apply L-BFGS to a smooth approximation. sec 3 p.7-11: experiments with hanso (Les Houches n=500, m=1, 20).
  -> Ipopt's default L-BFGS (history 6, scalar1 scaling) is exactly "scaled L-BFGS".

## Schwerdtner & Voigt, "Fixed-order H-infinity controller design for port-Hamiltonian systems", arXiv:2209.07558v2 (Sep 2022) -- FULL
- SOBSYN (Alg. 1 p.6): bisection on gamma; objective eq. (10) p.5 L = (1/gamma) sum_{s_i in S} sum_j [sigma_j(P*K(s_i)) - gamma]_+^2
  (differentiable, Prop 3.1 of S&V 2020); minimised with the BFGS solver of Optim.jl (Mogensen-Riseth 2018); sample set S updated each
  bisection step by the adaptive sampling of S&V 2021 Alg. 3.1, adapted from Apkarian-Noll 2018. Stability by pH structure (no constraints).
  Compared with HIFOO and hinfstruct on COMPleib passive plants (Fig. 2) and MSD plants (Table 1 p.9 runtimes). No certificate.

## Gradient sampling survey: Burke, Curtis, Lewis, Overton, Simoes, arXiv:1804.11003v1 (Apr 2018; "Submitted to: Special Methods for
## Nonsmooth Optimization, Springer, 2018, eds Bagirov, Gaudioso, Karmitsa, Maekelae" p.1) -- FULL (sec 2-5)
- Alg. GS p.4: sample size m >= n+1; sec 3.1 p.6 l.344-346 main weakness per-iteration cost (m >= n+1 gradients);
  sec 4.3 p.9 adaptive sampling; sec 4.4 p.9-10 second-order variants; sec 5.2 p.10 SQP-GS [CO12].
- Refs p.16: [BLO05] SIOPT 15(3):751-779, 2005; [CO12] SIOPT 22(2):474-500, 2012.

## NCVX/PyGRANSO: Liang, Mitchell, Sun arXiv:2111.13984v2 (Jan 2022) and arXiv:2210.00973v2 (OPT2022) -- FULL (short papers)
- 2111.13984 sec 2 p.2-3: PyGRANSO = PyTorch port of GRANSO; autodiff; GPU; OSQP and other QP solvers for search directions and the
  stationarity measure. 2210.00973 sec 5 p.6 (future work): SR1, stochastic, conic programming (SDP/SOCP), minimax -> no native cones.
- PyGRANSO README (GitHub): AGPL v3; deps PyTorch >= 2.8, OSQP >= 1.0.4, optional gurobipy.

## Archive cross-checks (already [R] in LITERATURE.md)
- CMO 2017 (OMS 32(1)): penalty phi = mu f + v (eq. (2),(4) sec 2 p.5-6); infeasible stationary points if mu "set too high" (p.6 l.296-299);
  SFPP sec 2 p.6 l.300-313; Procedure 1 p.8 l.389; (12)-(13) p.9 l.455-460; BFGS-SQP uses Armijo-weak Wolfe line search (p.9 l.470-475);
  GS "requires that O(n) gradients be evaluated per iteration" (sec 1 p.3 l.132-135).
  SNOPT included as comparator (p.9 l.487-495). sec 4.2 p.12 l.630-645: SNOPT least minimisation; SFPP "struggled for well over 100
  iterations before any real progress towards feasibility"; SQP-GS CPU per iteration > 10x BFGS-SQP. sec 6 PDF p.25 l.1275-1283 (journal p.171):
  inf-RMP SQP-GS 78% of best vs BFGS-SQP 22%, SQP-GS 26.6x longer; "SNOPT and SFPP appear to be completely uncompetitive";
  pseudospectral set p.26 l.1327-1335: SNOPT ran in 30% of BFGS-SQP time but BFGS-SQP still better for beta >= 1.
- Lukšan-Vlček 1998 Table 3 p.390 (image, rendered): BT Ni/Nf 79/88, 24/27, 13/16, 13/21, 9/13, 12/17, 10/11, 49/74, 6/13, 22/32, 29/30,
  45/56, 125/128, 74/84; EB Ni/Nf 20/20, 45/45, 58/58, 45/45, 191/600, 15/15, 16/16 -> bundle methods ~1-2 evaluations per iteration.
- HinfMOR (Mengi 2023) sec 2 p.6 l.344-356: gradient descent 37 iterations, 624 evaluations (8-28 per iteration), Table 2.1 p.7.
- AN 2006 journal p.82 (PDF p.12 l.746-747): "first-order method ... may be slow in the neighborhood of a local solution".
- HIFOO sec 3 p.3 l.145-158 (BFGS, local bundle, gradient sampling); sec 4 p.4 l.211-231 (linorm / Control System Toolbox norm).
- Lewis-Overton 2013: abstract p.1 l.32 (R-linear); sec 6.2 p.23 l.1239-1246 (44 eigenvalues of H_k -> 0; codimension 9(10)/2-1).

## Fendl & Schichl II, arXiv:1506.08021v1 -- partial (sec 4 set-up)
- p.17 l.774-788: MPBNGC (Maekelae) and SolvOpt chosen as comparators: compiled, publicly available, support nonconvex constraints;
  MPBNGC supports multiple nonsmooth constraints; "very fast w.r.t. pure solving time for the low dimensional examples".
- p.13 l.625-633: MOSEK / IPOPT considered as search-direction (QCQP) solvers.

## SciPy SLSQP source (scipy/optimize/_slsqp_py.py, main) -- read
- Kraft 1988 DFVLR-FB 88-28; LSQ subproblem (Lawson-Hanson ch. 25); maxiter 100, ftol 1e-6; exit codes for LSQ failures; no TR, no cones.

## Not accessed (content): Waechter-Biegler 2006 (paper), Waechter-Biegler SIOPT 2005 filter, Fletcher-Leyffer 2002, SNOPT SIGEST 2005,
## Byrd-Nocedal-Waltz KNITRO 2006, Buskens-Wassel WORHP 2013, Kraft 1988, MOSEK docs, Maekelae MPBNGC 2003 report, Hettich-Kortanek 1993,
## Reemtsen-Goerner 1998, Stein 2012, Djelassi-Mitsos-Stein 2021, Lopez-Still 2007, Zhou-Tits 1996 / CFSQP, Panier-Tits 1989,
## Polak-Wardi 1982, Polak-Mayne 1980 reports, Lukšan-Matonoha-Vlček 2009/2010 (Kybernetika), Fares-Noll-Apkarian SSDP 2002,
## Henrion-Löfberg-Kočvara-Stingl PENBMI 2005, Grimholt-Skogestad 2018, Han 1981, Janka et al. blockSQP 2016 (README only).
