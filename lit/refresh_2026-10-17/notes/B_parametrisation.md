# B_parametrisation raw notes (2026-10-17)
Network: only arxiv.org and hal.science reachable (curl). mathworks.com, cs.nyu.edu, chalmers, diva, ntnu, newcastle, optimization-online, springer, doi targets: 403/ENOTFOUND. HAL search page robot-blocked; record pages OK.
Downloads: scratchpad/lit2/pdf/B_parametrisation/<key>/ (pdf + pdftotext). Scripts: scratchpad/lit2/scripts/tied_check.py, tied_mimo.py (own, run with python3 -I).

## Full text read (F)
- AN2017_solved (hal-01653161, Aerospace Lab 13): p.AL13-01 3 reduced-order structure kappa=(vecA_K,vecB_K,vecC_K,vecD_K), dim k^2+k(ny+nu)+nynu (full, gauge not discussed); p.5 Lyapunov variables -> "serious ill-conditioning"; Fig.3 proximity control tau+=2tau if rho<gamma, tau/2 if rho>=Gamma.
- arXiv 2211.08083 (Apkarian-Noll, hyperbolic PDE): §2.3 p.4 alpha(Acl)<=-eps in finite dim; §5.1 p.12-13 rho(K)<=delta (spectral radius) and zeta(K)>=mu (min damping of controller eigenvalues) -> stable controllers so winding number 0 stays correct; §5.2 p.14 "54 tunable parameters ... tridiagonal state-space realizations. Those realizations do not limit generality and are less costly than full state-space forms"; delta=100, mu=1e-2.
- arXiv 1511.03687 Burke-Eaton: §1 p.1 spectral max functions continuous, generally not Lipschitz; Burke-Overton 2001 [11]: spectral abscissa regular iff active eigenvalues nonderogatory (§6 of this paper extends); polynomial root max functions regular (Burke-Eaton 2012 [2]); [12] Burke-Overton SICON 39(6) 2001 polynomial abscissa.
- arXiv math/0603681 Henrion-Overton: abscissa minimiser for 2-mass-spring with 2nd-order controller: all 6 closed-loop roots at -sqrt(15)/5 (§4-5), sharp local min (Thm §5 p.5); 5-digit rounding of controller moves poles -0.7746 -> -0.94...-0.66 (§6 p.8); m>=3 unbounded below (p.4).
- arXiv 1012.3281 Peeters-Hanzon-Olivi LAA 2007: §1 p.1 balanced parametrisations: stability built in; §3 p.5 local balanced canonical form (first n columns of controllability matrix positive upper triangular) is GLOBAL in SISO, LOCAL in MIMO (non-generic Kronecker structure) -> finite atlas of overlapping forms (abstract: minimal atlas).
- arXiv 1006.5876 Ait Rami-Henrion: §1 stability region in coefficient space generally nonconvex; §4.1 p.6 2nd-order DT region exact triangle.
- arXiv 1104.2679 Henrion-Louembet: §1 p.1 Routh-Hurwitz -> stabilising set basic semialgebraic, nonconvex in general.
- arXiv 2208.07540v4 Aliyev-Mengi: §1 p.1 abscissa not Lipschitz even locally at optimiser with nonsimple rightmost eigenvalue [BLO FoCM 2001, PAMS 2001]; p.2 pseudospectral abscissa locally Lipschitz [Lewis-Pang 2008 Cor 7.2; Gurbuzbalaban-Overton 2012 Cor 3.4].
- hal-01146512 Hilaire-Chevrel-Whidborne IJC 2010: Def 3.6 (56) closed-loop pole sensitivity; §5 Table 1 p.12 canonical (DFII) Psi=3.3562e7, balanced 6.5007e5, Psi-optimal 2742.5.
- arXiv 2209.07558 Schwerdtner-Voigt: §5 Lemma 1 pH parametrisation J skew, W=vtu^T vtu, Q=vtu^T vtu ... "no constraints have to be enforced"; p.5 "for higher orders, the number of parameters can be excessive".
- arXiv 1611.00595 Gillis-Sharma: abstract/Lemma 1 A stable iff A=(J-R)Q, J skew, R psd, Q pd.
- arXiv 2608.16401 Monti 2026: §5.4 p.20-21 similarity param S=RBR^-1 non-identifiable; penalty (39) lambda log|det R| - lambda p/2 ||R||_F^2 for conditioning/scale gauge; Supp §S2.2 (S13)-(S14) general 2x2 coupled constraints tr<0, det>0, redundant in similarity param; (S15)-(S17) equal-diagonal; McKelvey-Helmersson 1996 tridiagonal = equal-diagonal 2x2 substructures + superdiagonal couplings; real Schur block partition changes when real pair -> complex.
## Archive cross-checks (R)
- HIFOO: §1 p.1 abscissa nonsmooth at ties, non-Lipschitz at multiplicity>1, nonsmooth at optimisers (Burke et al 2003 ROCOND); §2 full Ahat Bhat Chat Dhat, closed loop affine in K; stabilisation = abscissa of A_K (closed loop) until negative, also before H-inf; no controller-stability option in this paper; §3 BFGS ill-conditioned Hessian; §4 linorm/norm dense; §5 random restarts, order augmentation; §6.1 multiplicity six, 8 digits shown because clustered eigenvalues sensitive; §6.2 AC2; §6.3 AC4.
- HinfMOR p.5 (§2) Kronecker canonical form under index-1 + semisimple poles -> tridiagonal A, diagonal E, 4r-2+rm+pr+pm variables (B,C,D free: diagonal-scaling redundancy remains); p.16 §6 simple rightmost eigenvalue; p.19 §8 stability constraints not imposed.
- IJRNC 2018 §6 p.3222: stable controllers alpha(A_K)<=-eps and conical alpha_cone(A)=max Re/|lambda| <= -r; subgradients per BLO LAA 2002 and Bompart et al.; Remark 9 p.3221.
- TAC 2015 §VI p.1866: 85 parameters, tridiagonal 12th-order controller (34 + 12x3 B + 12 C + 3 D: diag-scaling gauge kept).
## Snippet only (S)
- tunableSS: Astruct 'tridiag' (default) | 'full' | 'companion'; tridiag: free on three central diagonals; companion: char poly in rightmost column (see canon); B,C,D free by default; .Free fields; no conditioning rationale found. Third-party Go port (PlantControl/controlsys, GitHub code search fragment) says tridiag from sys uses canon modal form; its PR #297 (snippet) reports doc contradictions on Free defaults.
- TuningGoal.ControllerPoles(blockID,mindecay,mindamping,maxfreq); TuningGoal.Poles: Re(s)<-mindecay, Re(s)<-mindamping|s|; systuneOptions MinDecay/MaxRadius for implicitly constrained dynamics.
- DDLC (Automatica 40(9):1629-1635, 2004): orthogonal complement of tangent space of equivalence class; Ribarits-Deistler IFAC 2002 abstract; Wills-Ninness-Gibson TAC 53(1) 2008 abstract: DDLC special case of rank-deficient-Jacobian techniques; dynamic Jacobian rank determination.
- BLO FoCM 2001 abstract: active eigenvalues at critical matrices nonderogatory, multiplicity persistent.
- canon 'modal' condt: bound on cond of block-diagonalising transformation for close eigenvalues; canon deprecated -> modalreal/compreal.
## Own checks (I, scripts)
- tied M=[[a,om+g],[a,g]], b=beta[1;1]: tr=a+g, det=-a om, det[b,Mb]=-beta^2 om; T=[[1,-1],[0,om]]/(beta om) maps to controllable companion [[0,1],[-det,tr]], b->e2 (parameter-independent T).
- det(M+eps I)=eps^2+eps(a+g)-a om (linear). Double root (s+eps)^2: row gradients (1,1),(eps-om,eps) independent. Old (d1,e,f,d2) det-row gradient = 0 at M=-eps I.
- eig of exact double root at -eps after 1e-16 rel perturbation: max Re+eps = +5.9e-10.
- SISO parallel 2 tied blocks: TF-coefficient Jacobian sv_min 4e-17 when blocks share one real root; 1.07e-3 at 0.02 separation; rank -2 when denominators equal.
- MIMO 2-input block -> M=lam I (minimal): tied-chart |B_t[:,2]| ~ 3.9/d (3.9e3 at d=1e-3, 3.9e6 at 1e-6), cond(S) ~ 3.5/d.
