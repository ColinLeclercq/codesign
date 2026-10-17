# Group E: flow control, actuator/sensor co-design, physical baselines (raw notes, 2026-10-17)

Downloads: (session scratchpad, not kept) pdf/E_flow_codesign/<key>/ (pdf + pdftotext; page-tagged pp.txt / jp.txt / np.txt / bp.txt / wp.txt / mp.txt / gp.txt).
Access: arxiv.org OK; hal.science main portal OK (metadata, and files only when the record is NOT in a collection portal);
polytechnique.hal.science, pastel.hal.science, hal.archives-ouvertes.fr, *.archives-ouvertes.fr, cambridge.org, resolve.cambridge.org,
uwspace.uwaterloo.ca, folk.ntnu.no, cwrowley.princeton.edu, w3.onera.fr, sam.ensam.eu, spiral.imperial.ac.uk, core/semanticscholar/openalex: BLOCKED (403/000).
WebFetch: no DNS (useless). HAL API (api.archives-ouvertes.fr) blocked; HAL web search rejects robots.

Legend: FULL = full text read (arXiv/preprint version unless stated); ABS = abstract only (HAL metadata page); SEC = secondary account in a FULL paper; S = search snippet; NONE.

## Jussiau, Leclercq, Demourant, Apkarian, "Data-driven stabilization of an oscillating flow with LTI controllers", arXiv 2404.08487v1 (JFM 999 A86, 2024 per search) -- FULL (arXiv v1, 38 pp; only v1 exists; JFM version not accessible)
- Abstract p1: data-based; input-output identified LTI *mean transfer function*; **LQG synthesis** ("practical to automate online"); stacking + balanced truncation; cylinder Re = 100.
- Table 1 p6: Leclercq 2019 = "Linearization of the dynamics around the time-averaged flow" / "Structured H-inf" / "Full-order stacking"; this paper = "Identification of mean transfer function from input-output data (§2.4)" / "Automated LQG (§2.5)" / "Stacking, balanced truncation and state initialization (§2.6)".
- p6: Leclercq 2019 "used multi-objective structured H-inf synthesis (Apkarian & Noll 2006; Apkarian et al. 2014) for the design of low-order controllers ... can enforce controller structure, roll-off, or location of poles in the closed-loop ... not easy to automate"; controller order grows linearly with iterations.
- §2.3 p8-10: SISO, actuators = blowing/suction at cylinder poles (10 deg), sensor v2(x1=3, x2=0) FIXED (no placement study); FEniCS TH P2-P2-P1, ~25,000 triangles, **113,000 states** descriptor, 24 MPI cores, MUMPS -- used for simulation only.
- §2.4.3, §3.1.2 p13, p18: FORSE subspace ID, ROM order n_G = 8 per iteration, poles constrained Re <= 0 by LMI (Demourant & Poussot-Vassal 2017).
- §2.5 p13-14: LQG with Q = I, W = I, parameters R, V only; remark (iii) p14: LQG margins "may be arbitrarily small (Doyle 1978) but are checked a posteriori".
- §2.6.2 p15: BT error bound (2.16); unstable K modes split off before BT.
- §3.1.3 p19-20, fig. 10: R, V chosen by gridding closed-loop nonlinear simulations (PKE reduction, max|u|); R = 1e4, V = 1e-3 then V = 1.
- §3.2 p20-24: 8 iterations to the base flow; final controller order 22-23 vs 64 without BT (fig. 16).
- §4.1.2 p26-27: gap metric between implicit and mean-flow models.
- p4 §1.2.3: "structured H-inf control (Jussiau et al. 2022)" = design "by direct interaction with the full-size nonlinearity ... with the help of optimization"; ref list p36: Jussiau, Leclercq, Demourant, Apkarian 2022, "Learning linear feedback controllers for suppressing the vortex-shedding flow past a cylinder", IEEE Control Systems Letters 6, 3212-3217 (NOT accessed; no open copy found).
- Stability "certification": none on a full linear model; convergence judged by PKE in nonlinear DNS (fig. 11). No H-inf norm certified.

## Leclercq, Demourant, Poussot-Vassal, Sipp, JFM 868 (2019) -- NONE (Cambridge blocked; no arXiv/HAL copy found). SEC from Jussiau 2024 (Table 1 p6; p6; p27; p30) and Nibourel 2023 (p3).
- SEC: mean-flow linearisation; multi-objective structured H-inf (Apkarian-Noll 2006 + Apkarian-Gahinet-Buhr 2014 = systune); full-order stacking; open cavity, quasiperiodic (Jussiau p27, p30: "demonstrated its capability to fully stabilize the quasiperiodic flow over an open cavity").
- S: Re = 7500 open cavity; resolvent about mean flow; base flow reached after five iterations. Model size, ROM type, certification: unknown.
- Pages differ between ref lists: 26-65 (Jussiau p36) vs 22-65 (Nibourel p37).

## Nibourel, Leclercq, Demourant, Garnier, Sipp, arXiv 2206.06224v1 (JFM 954 A20, 2023 per search) -- FULL (arXiv v1)
- Abstract p1: M = 4.5 boundary layer; ERA ROMs; "multi-criteria structured mixed H2/H-inf synthesis"; feedforward vs feedback.
- p2 §1.1: "Controller synthesis is only feasible for models of small dimensions, of the order of 10^2 degrees of freedom at most".
- p3-4: Barbagallo et al. 2012 = feedback + LQG on BFS; "some of their feedback controllers turned out to be unstable on the real plant (the full linearized Navier-Stokes equations), because of the poor stability robustness of LQG to tiny erros in the ROM". Robust synthesis used so far mainly for oscillator flows (Flinois & Morgans 2016; Leclercq et al. 2019; Shaqarin et al. 2021).
- §5.1 p18: ERA ROMs with dead-time removal; total ROM order 130 (feedback) / 115 (feedforward); 220 for one transfer without dead-time removal. Results reported on the full DNS (elsA).
- §5.2 p19-21: systune; K state-space, **K stable**, order 5, **tridiagonal A_K (McKelvey & Helmersson 1996: any real square matrix similar to tridiagonal)**, strictly proper; problem (5.3): min max_i ||T_ziw||_2 s.t. ||W_S S||inf < 1 (|S| < 2, i.e. 6 dB), ||W_KS KS||inf < 1; "several tens of minutes".
- §6.2 p23-25: Nyquist plots, GM/PM/modulus margin under +-5% inflow velocity.
- App. A p30-31, fig. 14: sensor position by 1-D sweep of the achievable H2 reduction on ROMs; feedback performance drops fast with actuator-sensor distance (delay); sensor <= one wavelength 2pi/alpha_r downstream (~20 delta*); chosen 18.5 delta* downstream (feedback), 66 delta* upstream (feedforward). Placement and synthesis are NOT joint.

## Barbagallo, Dergham, Sipp, Schmid, Robinet, JFM 703 (2012) 326-362 -- ABS (HAL hal-00996503, metadata only; no file). SEC Nibourel p3-4, p25.
- ABS: Re = 600 rounded BFS, globally stable, upstream random noise; wall-mounted shear-stress sensor -> controller -> actuator; studies sensor placement, cost functional, ROM accuracy, compensator stability, performance; "inherent amplification behaviour ... reduced by an order of magnitude" if constraints observed; estimation sensor upstream near the actuator; very accurate ROM needed, especially actuator-to-estimation-sensor; "very minute errors even at low energies and high frequencies may render the large-scale compensated linearized simulation unstable"; nonlinear simulations: gradual deterioration with noise amplitude.
- SEC: LQG, feedback structure (Nibourel p3-4).
- Project notes [C] (BFS.md l.84, 91-95): paper's performance sensor Cp = [11, 11.6]; noise B1; actuator B2; sensors C1..C4; domain [-20,100]x[0,20], n = 400,775; BFS.md l.1014: gains differ by orders of magnitude between the paper's domain and ours.
- The "order of magnitude" metric could not be identified (norm/channel not in the abstract). Full text and the thesis (pastel-00654930) blocked.

## Dergham, Sipp, Robinet JFM 719 (2013) 406-430 -- NONE (biblio confirmed by Boujo-Gallaire ref list p30).
## Hervé, Sipp, Schmid, Samuelides JFM 702 (2012) 26-58 -- ABS (HAL hal-00994468 metadata). ARMAX system identification, BFS, robust to noise level and off-design. Biblio confirmed (Nibourel p37).
## Barbagallo, Sipp, Schmid JFM 641 (2009) 1-50 -- NONE (HAL hal-01021129 file on blocked portal). Biblio confirmed (Jussiau p35, Nibourel).
## Sipp & Schmid AMR 68(2) 020801 (2016) -- NONE. Biblio confirmed (Nibourel p38: 68, 020801). Nibourel p3: LTR used by Sipp & Schmid 2016 on open cavity; LTR unsuitable with delays (RHP zeros).
## Juillet, Schmid, Huerre JFM 725 (2013) 522-565 -- ABS (HAL hal-00995144 metadata; file on blocked portal). BFS Re = 350, three localised noise sources, subspace ID, feed-forward; "one-order-of-magnitude reduction in the averaged perturbation norm"; sensor upstream for convective flows; visibility length.

## Jin, Illingworth, Sandberg, arXiv 2103.16802v1 (JFM 932 per LITERATURE; DOI 10.1017/jfm.2021.948 per search) -- FULL
- Cylinder Re 50-110; FEniCS TH, 2.7e4 triangles, N = 1.2e5 dof (§2.2 p4); MUMPS + ARPACK.
- §3.3 p8-9: H2-optimal (Riccati, full order) estimators/controllers; disturbances "applied everywhere" and full-state performance (OE/FIC); Riccati intractable N > 1e3 -> "terminal reduction": B_d = P M F_m, C_z = U_m M P^T from POD of first k resolvent modes over [-w_n, w_n]; k = 3, w_n = 9 (App. B convergence); cost gamma^2(k, w_n) eq. 3.4.
- §4.x: brute-force sampling maps + "gradient minimisation method (e.g. Newton's method)" from sensible guesses (p11, p14, p18); SISO; time-lag trade-off; optimum splits into two local minima at Re ~ 91-92 (fig. 11).
- p2-3: Chen & Rowley (2011, 2014) H2 placement on CGL with extended gradient minimisation (Hiramoto et al. 2000); modal/Gramian heuristics fail; Oehler & Illingworth 2018: wavemaker has no special significance; ref list p26: Oehler & Illingworth JFM 854, 34-55; p24: Chen & Rowley JFM 681, 241-260.
- No H-inf, no fixed order, no certificate.

## Benner, Heiland, Werner, arXiv 2103.01608v2 (COAP 82(1) 225-249, 2022; confirmed by Jussiau ref list p35) -- FULL
- §2.1 eq. (6)-(7): standard H-inf Riccati pair; §2.2 eq. (9)-(10): **normalized** problem B1 = B2 = B, C1 = C2 = C (so B1B1^T = BB^T, C1^TC1 = C^TC); HINFBT (Alg. 1) low-rank square-root balancing; Prop. 2-3, Thm 1: a-priori/a-posteriori robustness bounds (coprime factor errors).
- §3: projected Riccati (Leray projector via saddle-point solves), M-M.E.S.S. low-rank Newton-ADI.
- §4 Table 1-2 p13: cylinder (2 outlet controls; 3 observation subdomains), n_v = 41,718 (+5,418 p); double cylinder (2 rotations; 4 subdomains), n_v = 45,528 (+5,809 p); gamma = 313.0176 / 12.5418. Fig. 4 p16: reduced controller orders 30-48 / 12-29; stabilisation checked in nonlinear simulation.
- Fixed ports, full-order theory then BT; not fixed-order optimisation.

## Werner, Overton, Peherstorfer, arXiv 2205.15050v2 (SISC 45(2) 2023 per search) -- FULL
- Fixed-order H-inf (n_K = 2, D_K = 0, §4.1-4.2 p12-13) by gradient sampling (HANSO base), multi-fidelity: RMF-GS (Alg. 1, restart level by level), AMF-GS (Alg. 3, low-fidelity sampled gradients).
- Table 1 p13: cylinder (Re 90, TH, descriptor, singular E) n = 6,618 / 10,645 / 22,060; m1/m2/p1/p2 = 14/6/14/8 (14) or 3/3/8/8 (15).
- p15: H-inf by ROSTAPACK normTfMaxPeak / normTfPeak; "normTfPeak does not do a stability check" -> eigs; no global certificate claimed.
- Table 6 p20: HF-GS 86.4 h -> 1.174923; RMF-GS 82.4 h -> 1.137206; AMF-GS 37.9 h -> 1.181741. Fig. 4: speedups 4 / 2 (formulation 15). Rail: speedup 452 (RMF) and 30 (AMF).
- p19: BFGS tried; "particularly for the cylinder example, the behavior of gradient sampling was more consistent and reliable".
- p18: RMF-GS high-fidelity objective not monotone (low-/high-fidelity mismatch).
- App. B p23: spectral-abscissa gradient (simple rightmost eigenvalue); shift updated from previous computations.

## Manohar, Kutz, Brunton, arXiv 1812.01574v2 (IEEE TAC 67(4) 2022 per search) -- FULL (8 pp)
- Greedy QR pivoting on dominant direct/adjoint balancing modes; runtime linear in n; GL demo; evaluated with H2 / LQG; approximates gradient-descent placements of Chen & Rowley. Open-loop observability/controllability heuristic, not closed-loop co-design.

## Martini, Jung, Cavalieri, Jordan, Towne, arXiv 2201.00361v1 (JFM 937 A19, 2022) -- FULL
- Wiener-Hopf causal optimal estimation/control from full-rank globally stable systems; p10: non-causal kernel "provide[s] upper bounds for the effectiveness of causal control"; truncating it is sub-optimal.
- §4 p20-22: BFS Re = 500 (step height), Nek5000 600 elements p = 5, Gaussian sensors/actuators (sigma_x 0.2, sigma_y 0.1), high-rank upstream forcing.
- §4.1.1 p22: non-causal control "provides upper bounds for the performance of any linear control strategy"; can be solved frequency by frequency; **"actuator placement has a negligible influence on the non-causal control"**; causal control degrades with placement (50 % loss at ~4.5 step heights upstream of the sensor).
- H2/PSD (stochastic) setting, not H-inf.

## Gumussoy & Michiels, arXiv 2003.09252v1 (SICON 49 2011 per LITERATURE) -- FULL
- Fixed-order strong H-inf for DDAEs (delays), predictor-corrector level set (§5), HANSO (§6); benchmarks with n = 2-5 (Table 7.4, p21). Not flows, small scale; relevant only as delay modelling.

## Kasinathan & Morris, IEEE TAC 58 (2013) 2522-2535 -- NONE. SEC Edalatzadeh & Morris arXiv 1802.05807 p1: existence and approximation-convergence results for H-inf actuator location ([27]). S: thesis abstract (UWSpace 10012/6732): H-inf state-feedback attenuation vs actuator location; derivative-free optimisation. CDC 2011 "Convergence of H-inf optimal actuator locations" (title, S).
## Morris, IEEE TAC 56 (2011) 113-124 -- NONE. SEC Edalatzadeh-Morris ref [38] (biblio) and p1 (existence of LQ optimal location + approximation conditions). S: library record abstract.
## Chen & Rowley JFM 681 (2011) 241-260 -- NONE (Princeton blocked). SEC Jin 2022 p2-3. S: snippet "actuator position and controller simultaneously optimised" (H2).
## Oehler & Illingworth JFM 854 (2018) 34-55 -- NONE. SEC Jin 2022 p3, p12, p15, p21.

## Extra (candidates for new_refs), FULL unless noted
- Dadfar, Fabbiane, Bagheri, Henningson arXiv 1402.0769 (centralised vs decentralised LQG, 3-D boundary layer): semi-decentralised 3x3 units x 6 -> only 11 % performance loss (abstract p1, §3.2, p15).
- Deshpande & Bhattacharya arXiv 2409.09596 (2024): convex (LMI) H2/H-inf with weighted-l1 sparse actuation/sensing, simultaneous design; structural-dynamics demo.
- Nugroho, Taha, Summers, Gatsis arXiv 1806.05481 (ACC 2018): simultaneous SaA selection via static output feedback, MINLP / binary search.
- Boujo & Gallaire arXiv 1411.1863: BFS (Gamma = 0.5, Re 500) sensitivity of harmonic/stochastic gains to steady control.
- Yao, Sun, Hemati arXiv 2009.03469: sensor selection for performance recovery (sparse synthesis / BT) in channel flow.
- S only: ONERA thesis proposal mfe-daaa-2026-09 (w3.onera.fr, blocked): joint optimisation of actuators, sensors and compensator by nonsmooth optimisation on the full-order resolvent; BFS and open cavity; "H2 or robust H-inf synthesis cannot easily manage systems of order greater than O(100)".
