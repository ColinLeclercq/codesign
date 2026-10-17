#!/usr/bin/env python3
"""bfs_stationarity.py -- how far a BFS co-design record is from Clarke
stationarity (SISO, MIMO decentral, MIMO central), with the active
constraints' normal cones, the first-order decrease still available, and a
measured check of that promise.

    mpiexec -n 2 python -u tools/bfs_stationarity.py RECORD.npz --ops OPS
        [--eps 1e-6,1e-5,1e-4,1e-3,1e-2] [--eps-max 1e-2] [--eps-pred 0.05] [--json OUT.json] [--fd K]
        [--deltas 1e-8,...,1e-1] [--trust T] [--complete off|on] [--complete-eps 1e-2,1e-3]
        [--strip off|on] [--cl-line problem|qp] [--fd-cl on|off]
        [--descent off|on|certify] [--descent-deltas 1e-7,...,1e-3] [--descent-ts 1]
        [--scale-from FILE.npz] [--wpad 3] [--margin 0.05] [--progress 8:0.0015] [--lu-cap 8]
    python tools/bfs_stationarity.py --selftest      # the convex core on designed toys (no plant, no MPI)

Run from the repository (the sandbox: PATH=<mamba>/fx/bin:$PATH OMP_NUM_THREADS=1
HDF5_USE_FILE_LOCKING=FALSE).  The record's JSON (same basename) supplies the
stop fields; the npz the design, its gamma, its certified curve (omega_grid /
gain, the full model at 2^-6), peakset, whi and sqp_trust (20261010n on).

THE PROBLEM.  min over z of f(z) = max_{w in [0, whi]} sigma_1(T(z, iw)) s.t.
the closed loop Re lambda_j <= -eps_s, the modal compensator rows v <= L
(controller.modal_rows, L = -eps_s), the port positions (BFSPlant.
position_rows).  z is the SQP's own packed vector (packer.Packer /
PackerMIMO, built with ctrl_modal on when the record is modal -- tools/
check_lattice.py omits it, and with it off the cut e/f entries become
variables: the measure would test a larger space than the SQP's).  The trust
box, the displacement cone, the elastic slacks are solver devices, not
constraints, and are left out.

WHAT IS MEASURED.  (i) The pieces: the local maxima w_i of sigma_1 within
relative eps of gamma (ends included), located by LatticeOracle.peak_tops
from the record's own 2^-6 curve and peakset (edge stops retried, tops within
1e-4 merged, a top beyond whi replaced by the end point); optionally made
complete by the level set at gamma (1 - eps) (LatticeOracle.level with the
known tops watched, --complete on).  (ii) Their rows, in the SQP's own form,
from ONE sqp_inner.Evaluator call at the exact tops (followed, hint = the
top): sigma_1 rows, the 2 x 2 Clarke blocks of the tops whose sigma_2 is
within eps_pred of gamma (the residual at eps counts the sigma_k within eps;
the decrease LP all of them) (cross terms: Y PSD, never the two diagonal branches
alone; r x r with r >= 3 built on the driver from gain_k + D.grads), every
closed-loop eigenvalue row the line exhibits, the modal rows, the position
rows.  (iii) The residual

    rho_q(eps) = min || sum_i lam_i g_i + sum_b <Y_b, dG_b> + sum_j mu_j h_j ||_q,
                 lam >= 0, Y_b PSD, sum lam + sum tr Y_b = 1, mu >= 0,

over the eps-active pieces and the eps-active constraints (c / 0.05 >= -eps;
0.05 = eps_s for eigenvalue and compensator rows, the margin for positions),
with and without the constraints, q = 2 (Euclidean) and q = 1 (the dual of
the trust box's l_inf), normalised by the mean ||g_i||_q of the active pieces
(the SQP's own _gscale, the GL convention of tools/stationarity.py): 1 = no
cancellation at all, 0 = exact (eps-)stationarity.  The mean misleads when the
|g_i| spread (GL KSP+B+C: rows 44.1 and 1734.1, the minimiser puts no weight on
the large one: a false "stationary kink"), so the residual is also printed over
the smallest active |g_i|, over the minimiser's own weights sum lam_i |g_i|
(blocks: sum_a Y_aa |Hs_aa|), and over that plus sum mu_j |h_j| (the
cancellation ratio in [0, 1]); per variable group the same cancellation ratio
on the group's coordinates, invariant under a uniform rescaling of the group
(the packed coordinates are badly scaled: sc, sx, the modal 2 x 2 blocks'
2-dimensional similarity gauge).  None of these is coordinate-free; the
measured decrease below is the number the verdict rests on.  The constraint
qualification: the distance from 0 to the hull of the unit active constraint
gradients (MFCQ when > 0; the residual's normalisation sum lam + sum tr Y = 1
is the KKT form, lambda_0 = 1 of Fritz John, meaningful only under MFCQ).
Every eigenvalue the line saw within eps_pred of it without a row is flagged
(a missing normal cone).  The decomposition: the
weights, each block's tr Y and eigenvalues, each mu_j and mu_j ||h_j||, the
residual by variable group.  (iv) The predicted first-order decrease

    pred_inf(Delta) = - min tau  s.t.  (s_i - gamma) + g_i.d <= tau,  lambda_max(diag(s_b - gamma)
                      + sym(sum_j d_j G_b,j)) <= tau,  c_j + h_j.d <= 0 (all rows),  |d|_inf <= Delta,

the SQP's own box QP with H = 0 (Polak's optimality function with the box in
place of |d|^2 / 2), over the pieces of P(eps_pred) (every top within
--eps-pred, default 0.05, of gamma; lower pieces and inactive rows are
discounted by their gaps).  Also the ball (|d|_2 <= Delta) and the box
without constraints; at the record's sqp_trust too.  Delta_set = gamma
eps_pred / max ||g_i||_1: beyond it a piece below gamma (1 - eps_pred) could
reach the max within the box, and pred is optimistic.  (v) The descent test:
d*(Delta) from the box LP with its closed-loop rows on the QP's line -eps_s -
2 mg (as the SQP), z + t d* re-evaluated (the tops of P(eps_pred) re-located
by peak_tops, the compensator, the positions, and the closed-loop eigenvalues
of every row that can reach -eps_s - mg within the step, tracked by
lattice._nearest; a compensator over its line is first put back by the SQP's
own second-order correction, sqp_inner _ctrl_restore_modal, as the run's line
search does): Delta_v = the largest Delta at which the actual decrease
is at least half the predicted one and the step is admissible on those
checks, and pred_inf(Delta_v) / gamma -- "the first-order decrease available
where the first-order model holds", the decision-relevant number.  (With
P(eps_max = 0.01) alone, SISO 598.11 at Delta = 1e-5 read -8.96e-3 over its 3
tops while the top at 1.373, 0.986 gamma, rose to 0.993: -7.04e-3.)
--descent certify adds, at the good steps (most decrease first, at most 3),
the stability line (clear, alpha <= -eps_s - mg) and the level set at the
step's max (1 + 1e-6); a region it finds is located, the max raised to its
top and the level set asked again (at most 3 times).  A clear outcome below
gamma is a certified strictly better design, admissible on the line test the
SQP's line search and the gate use (NOT STATIONARY (certified)); the run's
acceptance also counts windings over all w with a tail bound (winding_cert),
which is not run here, so an eigenvalue crossing above whi + wpad is not
excluded.  Otherwise the level set's tops correct the measured decrease
and Delta_v.

THE VERDICT (LESSONS 4, 18.8, 21): "stationary to rho_2/(sum lam_i |g_i|) = X
at eps = Y" (with rho_2/gscale, rho_2/min|g_i|, the worst group-wise necessary
condition and the MFCQ margin beside it) with the Clarke reading at the
certificate's eps = 1e-6, the measured pred_inf(Delta_v)/gamma, whether the
piece set is certified, and the stop reason -- never "optimal", "converged" or
"local minimum".  (eps, rho) is the smallest eps of --eps whose rho_2/(sum
lam_i |g_i|) (with constraints) is within a factor 2 of the smallest over the
list (rho itself is non-increasing in eps: more pieces and cones; its
normalisations are not).  Its class (design.md 4.2) comes from the measured decrease at
Delta_v: <= 1e-6 gamma "stationary at the certificate's resolution"; <= tol / k
of --progress k:tol (the run's progress rule; default 8:0.0015, the BFS rule
since 20261010r: 1.9e-4 per round; the 20261010g/n runs used 3:0.0015, 5e-4)
"near eps-stationary"; above, "NOT STATIONARY (measured, uncertified)"; a
level-set-certified lower admissible design, "NOT STATIONARY (certified)", the
level set on the band the run would use at the step (max(whi, max|Im
lambda(A_c)| + wpad): central 123.90's band end is its compensator pole) and
the design saved (--json OUT.json -> OUT_certified.npz).  A certified lower
design at a finite step is not by itself a proof of non-stationarity; that is
the measured/predicted ratio ~1 at the smallest step (pred linear in Delta:
the directional derivative is negative).  --complete on watches the known
tops by location (a region within 1e-3 of one is that top's), re-asking with
a re-located region watched at its estimate (at most 6 calls per level).  Without --complete on the piece set is the seeds' and the peakset's:
rho is then an UPPER bound of the true rho(eps) (a missed piece can only add
to the hull) and pred an optimistic estimate; the "stationary to" statement
stands, a large rho is not proof.

WHY THE RECORD'S stat_value IS NOT THIS.  The SQP's stat is ||grad_L||, the
QP duals times the Omega rows active at the QP's step inside its box -- by the
QP's own stationarity ||H d + box duals||, an eps-stationarity at an
effective eps(Delta) ~ Delta max||g||_1 / gamma (about 1 at Delta = 1e-3 on
SISO 598.11: stat 6.1 against a min-norm residual of 2.7e5).  That is how
code 0 under --sqp-stat rel passed a point with no cancellation (LESSONS 14).

CHECKS PRINTED EVERY RUN.  The convex core on designed toys (two opposite
pieces; one piece; five stationary by construction and perturbed against
nnls; the 2 x 2 cross-term block; a 3 x 3 block; an active constraint; the
eps discount; a PSD normal cone); the packer round trip; gamma recomputed at
the tops against the record's (the witness: a top above gamma means the
certificate is wrong, the max more than 1e-6 below a missing top); the line's
alpha against the record's; every top |sigma'| <= 1e-7 sigma; min_norm
without constraints <= min ||g_i||; pred primal = dual; the real-data
stationary point by construction (every g_i - r*: rho~ = 0, pred~ = 0, then
rho~ non-decreasing along t r*/|r*|).  FD (--fd K directions: the 3 largest
coordinates of the top piece, a random direction, the positions, ...): every
piece's sigma_1 at its fixed top by the Richardson pair (h, h/10), h = 1e-5
(central differences alone are 5e-4 off on the stiff coordinates, BFS
records 123.90 / 598.11), pass at 1e-6 relative to max(|g.u|, 1e-3 ||g||);
the blocks one-sided against lambda_max of the block's linearisation; the
modal rows (v2's scale s frozen at the record, as controller.modal_rows'
gradient has it: exact where v2 is active) and the position rows; with
--fd-cl on, the two most active closed-loop rows by lattice._nearest at
z +- h u (two factorisations each).

COST on 2 ranks (test plant, n = 33,593): 5-10 minutes per record with
--complete off --descent on (the Evaluator call is mostly the line), 14-31
minutes with --descent certify (the level sets at the step's max: 6-22
minutes), about 30 minutes with --complete on (5 level-set calls on
decentral 369.87).  At most mpiexec -n 2 in the
shared sandbox.

DESIGN NOTES (wf_stat/design.md, fixed where the code says otherwise).  The
line of the Evaluator covers Im in [0, whi + wpad] as in the run
(run_case_epi 3386: D.band_of = a.whi + a.wpad, wpad 3), not [0, whi].  The
LPs are solved in u = d / Delta with gamma-normalised pieces and rows scaled
by max(Delta |h|_1, |c|) (Clarabel returned AlmostSolved on the raw data,
|g| up to 3e5, |h| 0.03 .. 79).  The closed-loop Hermite block replaces the
two rightmost eigenvalue rows only when they are within 1e-2 of each other
and both active (near-defective); its normal cone is restricted to the
eigenvectors of H0 with eigenvalue <= eps tr H0 (complementarity), a
heuristic, and the block sits on the QP's line (-eps_s - 2 mg).  The
--sqp-view of the design (the QP's own -dt on its Omega) is not implemented.
The descent test (2026-10-17, decentral 369.87 / 412.28): its LP holds the
closed-loop rows on the QP's line -eps_s - 2 mg, its steps are checked against
the tracked closed-loop eigenvalues (-eps_s - mg) and corrected back onto the
compensator line like the SQP's trials; before, a step binding a closed-loop
row was quoted though it sat on -eps_s, and a step binding an active v2 row
was thrown out for a 1.2e-8 second-order crossing the SQP corrects.
"""
from __future__ import annotations
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(_os_root.path.dirname(_os_root.path.abspath(__file__))))
import argparse, json, os, sys, time
import numpy as np

CERT_RTOL = 1e-6       # the certificate's relative tolerance: the finest eps (the Clarke reading)
RAD = 1.0 / 32         # the followed members' window radius (the run's)
MERGE = 1e-4           # tops closer than this are one (Newton's tops agree to ~1e-6 on a flat peak)
GTOL_FD = 1e-6         # FD pass threshold, relative to max(|g.u|, 1e-3 |g|)
FD_H = 1e-5            # the Richardson pair (h, h/10)
BIND = 1e-9            # a row binds when mu ||h|| > BIND x gscale (not relative to the largest mu)
BIND_MN = 1e-6         # ... and, in the min-norm decomposition, when mu ||h|| > BIND_MN x gscale (below: the solver's slack)
PROGRESS = '8:0.0015'  # the BFS progress rule k:tol (20261010r: 0.15 % over 8 rounds; 3:0.0015 before): its per-round
                       # scale tol / k (1.9e-4; 5e-4 under the 3-round rule) is the verdict's class boundary (--progress)


# ============================================================== convex core
def _cl():
    import clarabel
    return clarabel


def _settings():
    clarabel = _cl()
    st = clarabel.DefaultSettings(); st.verbose = False; st.max_threads = 1
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = 1e-12; st.max_iter = 400
    return st


def _solve(n, c, A, b, cones):
    import scipy.sparse as sp
    clarabel = _cl()
    A = np.array(A, float).reshape(-1, n)
    sol = clarabel.DefaultSolver(sp.csc_matrix((n, n)), np.asarray(c, float), sp.csc_matrix(A),
                                 np.asarray(b, float), cones, _settings()).solve()
    return sol, A


def _hb(k):
    from sqp_inner import _herm_basis
    return _herm_basis(k)


def _sv(S):
    from sqp_inner import _svec
    return _svec(S)


def _em(H):
    from sqp_inner import _emb
    return _emb(H)


def hsym(Gc):
    """The Hermitian part, entry by entry: Hs[a, b] = (Gc[a, b] + conj Gc[b, a]) / 2 (k x k x NZ)."""
    return 0.5 * (Gc + np.conj(np.transpose(Gc, (1, 0, 2))))


def block_Y(y, k):
    """The block multiplier as a k x k Hermitian matrix from its coordinates (k = 2: the subproblem's SOC
    parametrisation; k >= 3: sqp_inner._herm_basis)."""
    if k == 2:
        return np.array([[(y[0] + y[1]) / 2, (y[2] + 1j * y[3]) / 2], [(y[2] - 1j * y[3]) / 2, (y[0] - y[1]) / 2]])
    return sum(c_ * B_ for c_, B_ in zip(y, _hb(k)))


def min_norm(NZ, G=None, blocks=(), H=None, psd=(), q=2):
    """rho_q = min || sum_i lam_i g_i + sum_b Re<Y_b, Hs_b> + sum_j mu_j h_j - sum_p DH_p svec(Lam_p) ||_q
    over lam >= 0, Y_b Hermitian PSD, mu >= 0, Lam_p PSD, sum lam + sum tr Y_b = 1.
    G (ns, NZ) scalar pieces; blocks: Hs (k, k, NZ) Hermitian in (a, b) -- k = 2 as sqp_inner.subproblem /
    grad_L (y in SOC(4), Y = [[(y0+y1)/2, (y2+iy3)/2], [., (y0-y1)/2]], tr Y = y0), k >= 3 in _herm_basis
    coordinates with svec(_emb(Y)) PSD; H (nc, NZ) constraint gradients; psd: (NZ, nsv) arrays svec(dH_j)
    of a PSD constraint H(z) >= 0 (projected on its near-null space): the normal cone -sum_j <Lam, dH_j> e_j.
    Returns dict(rho, r, lam, Y (matrices), mu, Lam, status)."""
    clarabel = _cl()
    G = np.zeros((0, NZ)) if G is None or not len(G) else np.asarray(G, float).reshape(-1, NZ)
    H = np.zeros((0, NZ)) if H is None or not len(H) else np.asarray(H, float).reshape(-1, NZ)
    ns, nc = len(G), len(H)
    cols, tr, bspec, pspec = [G.T], [np.ones(ns)], [], []
    off = ns
    for Hs in blocks:
        k = Hs.shape[0]
        if k == 2:
            h00, h11, h10 = np.real(Hs[0, 0]), np.real(Hs[1, 1]), Hs[1, 0]
            C = np.column_stack([0.5 * (h00 + h11), 0.5 * (h00 - h11), np.real(h10), -np.imag(h10)])
            t_ = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            basis = _hb(k)
            C = np.column_stack([np.real(np.einsum('ab,baj->j', B_, Hs)) for B_ in basis])
            t_ = np.array([float(np.real(np.trace(B_))) for B_ in basis])
        cols.append(C); tr.append(t_); bspec.append((off, k, C.shape[1])); off += C.shape[1]
    imu = off; cols.append(H.T); tr.append(np.zeros(nc)); off += nc
    for DH in psd:
        DH = np.asarray(DH, float).reshape(NZ, -1)
        cols.append(-DH); tr.append(np.zeros(DH.shape[1])); pspec.append((off, DH.shape[1])); off += DH.shape[1]
    nv = off
    Rm = np.hstack(cols)
    n = nv + (1 if q == 2 else NZ)
    A, b, cones = [], [], []

    def unit(i, v=-1.0):
        a_ = np.zeros(n); a_[i] = v; return a_
    a_ = np.zeros(n); a_[:nv] = np.concatenate(tr); A.append(a_); b.append(1.0); cones.append(clarabel.ZeroConeT(1))
    nn = 0
    for i in list(range(ns)) + list(range(imu, imu + nc)):
        A.append(unit(i)); b.append(0.0); nn += 1
    if q == 1:
        for k in range(NZ):                              # r_k - u_k <= 0, -r_k - u_k <= 0
            for sg in (1.0, -1.0):
                a_ = np.zeros(n); a_[:nv] = sg * Rm[k]; a_[nv + k] = -1.0; A.append(a_); b.append(0.0); nn += 1
    if nn:
        cones.append(clarabel.NonnegativeConeT(nn))
    for o, k, m_ in bspec:
        if k == 2:
            for j in range(4):
                A.append(unit(o + j)); b.append(0.0)
            cones.append(clarabel.SecondOrderConeT(4))
        else:
            S = np.array([_sv(_em(B_)) for B_ in _hb(k)]).T          # (nsv(2k), k^2)
            for r_ in range(S.shape[0]):
                a_ = np.zeros(n); a_[o:o + m_] = -S[r_]; A.append(a_); b.append(0.0)
            cones.append(clarabel.PSDTriangleConeT(2 * k))
    for o, m_ in pspec:
        qk = int(round((np.sqrt(8 * m_ + 1) - 1) / 2))
        for r_ in range(m_):
            A.append(unit(o + r_)); b.append(0.0)
        cones.append(clarabel.PSDTriangleConeT(qk))
    if q == 2:
        A.append(unit(nv)); b.append(0.0)
        for k in range(NZ):
            a_ = np.zeros(n); a_[:nv] = -Rm[k]; A.append(a_); b.append(0.0)
        cones.append(clarabel.SecondOrderConeT(NZ + 1))
    c = np.zeros(n)
    if q == 2:
        c[nv] = 1.0
    else:
        c[nv:] = 1.0
    sol, _ = _solve(n, c, A, b, cones)
    x = np.asarray(sol.x); r = Rm @ x[:nv]
    Ys = [block_Y(x[o:o + m_], k) for o, k, m_ in bspec]
    Lams = []
    for o, m_ in pspec:
        qk = int(round((np.sqrt(8 * m_ + 1) - 1) / 2)); L_ = np.zeros((qk, qk)); kk = 0
        for j in range(qk):
            for i in range(j + 1):
                v = x[o + kk] / (1.0 if i == j else np.sqrt(2.0)); L_[i, j] = L_[j, i] = v; kk += 1
        Lams.append(L_)
    return dict(rho=float(np.linalg.norm(r, 2 if q == 2 else 1)), r=r, lam=x[:ns], Y=Ys, mu=x[imu:imu + nc],
                Lam=Lams, status=str(sol.status))


def pred(NZ, gam, a=None, G=None, blocks=(), cons=(), psd=(), Delta=1e-2, norm='inf'):
    """pred(Delta) = - min tau over d, |d| <= Delta (box 'inf' or ball '2'), of
        a_i + g_i.d <= tau (scalar pieces, a_i = s_i - gamma <= 0),
        lambda_max(diag(a_b) + sym(sum_j d_j G_b,j)) <= tau (blocks: (a_b (k,), Hs (k, k, NZ))),
        c_j + h_j.d <= 0 (cons: (c_j <= 0, h_j)),  H0 + sum_j d_j dH_j >= 0 (psd: (H0, dH (NZ, q, q))).
    Solved in u = d / Delta, tau' = tau / gamma, each row scaled to O(1).  Returns dict(pred (in gamma
    units: pred / gamma), d, lam, Ytr (block weights tr Y), mu (multipliers in the residual's units: r =
    sum lam g + sum_b <Y_b, dG_b> + sum mu h), gap (primal - dual, relative), status)."""
    clarabel = _cl()
    G = np.zeros((0, NZ)) if G is None or not len(G) else np.asarray(G, float).reshape(-1, NZ)
    a = np.zeros(0) if a is None else np.asarray(a, float).ravel()
    ns, nc = len(G), len(cons)
    n = NZ + 1; it = NZ
    # tau' = tau / (gamma sp), sp = Delta max |g|_1 / gamma: the largest decrease the box allows, so the
    # objective is O(1) at every Delta (unscaled, the LP at Delta = 1e-8 ended AlmostSolved 4e-5 off)
    gm = [float(np.sum(np.abs(g_))) for g_ in G] + [float(np.max(np.sum(np.abs(np.real(np.einsum('aaj->aj', Hs))), axis=1)))
                                                   for _, Hs in blocks]
    sp = max(Delta * max(gm, default=0.0) / gam, 1e-14)
    gs_ = gam * sp
    A, b, cones = [], [], []
    for i in range(ns):                                   # (Delta / gamma sp) g.u - tau' <= -a / gamma sp
        r_ = np.zeros(n); r_[:NZ] = Delta * G[i] / gs_; r_[it] = -1.0; A.append(r_); b.append(-a[i] / gs_)
    nsc = []
    for c_, h_ in cons:                                   # (Delta h.u) / n_j <= -c / n_j
        nj = max(Delta * float(np.sum(np.abs(h_))), abs(float(c_)), 1e-300); nsc.append(nj)
        r_ = np.zeros(n); r_[:NZ] = Delta * np.asarray(h_, float) / nj; A.append(r_); b.append(-float(c_) / nj)
    lin = ns + nc
    if norm == 'inf':
        for j in range(NZ):
            for sg in (1.0, -1.0):
                r_ = np.zeros(n); r_[j] = sg; A.append(r_); b.append(1.0); lin += 1
    cones.append(clarabel.NonnegativeConeT(lin))
    bk = []
    for ab, Hs in blocks:
        k = Hs.shape[0]; ab = np.asarray(ab, float) / gs_
        if k == 2:
            # p = tau' - a1 - h11.d', r = tau' - a2 - h22.d', q = -h12.d' (d' = Delta u / gamma sp): (p+r, p-r, 2Re q, 2Im q)
            h11, h22, h12 = np.real(Hs[0, 0]), np.real(Hs[1, 1]), Hs[0, 1]
            s_ = Delta / gs_
            for cd, ct, k0 in ((-(h11 + h22) * s_, 2.0, -ab[0] - ab[1]), (-(h11 - h22) * s_, 0.0, -ab[0] + ab[1]),
                               (-2 * np.real(h12) * s_, 0.0, 0.0), (-2 * np.imag(h12) * s_, 0.0, 0.0)):
                r_ = np.zeros(n); r_[:NZ] = cd; r_[it] = ct; A.append(-r_); b.append(k0)
            cones.append(clarabel.SecondOrderConeT(4)); bk.append((k, 4))
        else:
            # svec(_emb(tau' I - diag(a) - (Delta / gamma) sum u_j Hs_j)) PSD
            b0 = _sv(_em(-np.diag(ab).astype(complex))); Mt = _sv(_em(np.eye(k, dtype=complex)))
            Mj = np.array([_sv(_em(Hs[:, :, j])) for j in range(NZ)]).T * (Delta / gs_)
            for r_i in range(len(b0)):
                r_ = np.zeros(n); r_[:NZ] = Mj[r_i]; r_[it] = -Mt[r_i]; A.append(r_); b.append(b0[r_i])
            cones.append(clarabel.PSDTriangleConeT(2 * k)); bk.append((k, len(b0)))
    pk_ = []
    for H0, dH in psd:
        H0 = np.real(np.asarray(H0)); dH = np.real(np.asarray(dH))
        lm = float(np.linalg.eigvalsh(H0).min())
        H0 = H0 + max(0.0, -lm) * np.eye(len(H0))              # numerically feasible at d = 0
        nh = max(float(np.linalg.norm(H0)), Delta * float(np.max(np.linalg.norm(dH.reshape(NZ, -1), axis=1))), 1e-300)
        b0 = _sv(H0) / nh; Mj = np.array([_sv(dH[j]) for j in range(NZ)]).T * (Delta / nh)
        for r_i in range(len(b0)):
            r_ = np.zeros(n); r_[:NZ] = -Mj[r_i]; A.append(r_); b.append(b0[r_i])
        cones.append(clarabel.PSDTriangleConeT(len(H0))); pk_.append(len(b0))
    if norm == '2':
        r_ = np.zeros(n); A.append(r_); b.append(1.0)
        for j in range(NZ):
            r_ = np.zeros(n); r_[j] = -1.0; A.append(r_); b.append(0.0)
        cones.append(clarabel.SecondOrderConeT(NZ + 1))
    c = np.zeros(n); c[it] = 1.0
    sol, Am = _solve(n, c, A, b, cones)
    x = np.asarray(sol.x); zd = np.asarray(sol.z); bv = np.asarray(b, float)
    primal = float(x[it]); dual = float(-bv @ zd)
    lam = zd[:ns]; mu = np.array([zd[ns + j] * gs_ / nsc[j] for j in range(nc)])
    Ytr, o = [], lin
    for k, m_ in bk:
        Ytr.append(float(2 * zd[o]) if k == 2 else float(np.sum(zd[o:o + m_] * _sv(_em(np.eye(k, dtype=complex)))))); o += m_
    return dict(pred=-primal * sp, d=Delta * x[:NZ], lam=lam, Ytr=np.array(Ytr), mu=mu,
                gap=abs(primal - dual) / max(1e-12, abs(primal), abs(dual)), status=str(sol.status))


def run_toys(verbose=True):
    """The designed toys of wf_stat/toys.py (design.md 6.2) plus a 3 x 3 block and a PSD normal cone.
    Returns (n_ok, n_total)."""
    from scipy.optimize import nnls
    res = []

    def chk(name, got, want, tol=1e-7):
        good = abs(got - want) <= tol * max(1.0, abs(want))
        res.append(good)
        if verbose:
            print(f"    {'ok ' if good else 'BAD'} {name}: {got:.3e} (want {want:.3e})")
    rng = np.random.default_rng(1)
    NZ = 40
    g = rng.standard_normal(NZ)
    r = min_norm(NZ, np.array([g, -g])); chk('1. +-g: rho', r['rho'], 0.0); chk('1. +-g: lam_1', r['lam'][0], 0.5)
    p = pred(NZ, 1.0, np.zeros(2), np.array([g, -g]), Delta=0.1); chk('1. +-g: pred box', p['pred'], 0.0)
    r = min_norm(NZ, np.array([g])); chk('2. one piece: rho = |g|', r['rho'], np.linalg.norm(g))
    p = pred(NZ, 1.0, np.zeros(1), np.array([g]), Delta=0.1); chk('2. pred box = Delta |g|_1', p['pred'], 0.1 * np.sum(np.abs(g)))
    p = pred(NZ, 1.0, np.zeros(1), np.array([g]), Delta=0.1, norm='2'); chk('2. pred ball = Delta |g|_2', p['pred'], 0.1 * np.linalg.norm(g))
    r = min_norm(NZ, np.array([g]), q=1); chk('2. rho_1 = |g|_1', r['rho'], np.sum(np.abs(g)))
    Gs = rng.standard_normal((4, NZ)); lam = np.array([0.1, 0.2, 0.3, 0.15]); g5 = -(lam @ Gs) / 0.25
    Gm = np.vstack([Gs, g5]); r = min_norm(NZ, Gm); chk('3. stationary by construction: rho', r['rho'], 0.0, 1e-6)
    Gp = Gm + 0.05 * rng.standard_normal(Gm.shape)
    s = 1e3 * np.abs(Gp).max(); x, _ = nnls(np.vstack([Gp.T, s * np.ones((1, 5))]), np.r_[np.zeros(NZ), s], maxiter=10000)
    x /= x.sum(); r = min_norm(NZ, Gp); chk('3. perturbed: rho vs nnls', r['rho'], float(np.linalg.norm(Gp.T @ x)), 1e-5)
    # 4. lambda_max([[1 + x1, x2], [x2, 1 - x1]]) + (0, 0.5).x: stationary only with the cross terms
    Gc = np.zeros((2, 2, 2), complex); Gc[0, 0] = [1, 0.5]; Gc[1, 1] = [-1, 0.5]; Gc[0, 1] = [0, 1]; Gc[1, 0] = [0, 1]
    r = min_norm(2, blocks=[hsym(Gc)]); chk('4. 2x2 block: rho with the block', r['rho'], 0.0, 1e-6)
    r = min_norm(2, np.array([[1.0, 0.5], [-1.0, 0.5]])); chk('4. 2x2 block: diagonal branches only', r['rho'], 0.5)
    p = pred(2, 1.0, blocks=[(np.zeros(2), hsym(Gc))], Delta=0.1, norm='2'); chk('4. pred ball, block', p['pred'], 0.0, 1e-6)
    p = pred(2, 1.0, np.zeros(2), np.array([[1.0, 0.5], [-1.0, 0.5]]), Delta=0.1, norm='2'); chk('4. pred ball, branches', p['pred'], 0.05)
    # 5. a 3 x 3 block: lambda_max(diag(x1, -x1, 0) + x2 offdiag(0, 1)) + 0.5 x2: Y = [[.5, -.25, 0], [-.25, .5, 0], 0]
    G3 = np.zeros((3, 3, 2), complex); G3[0, 0] = [1, 0.5]; G3[1, 1] = [-1, 0.5]; G3[2, 2] = [0, 0.5]
    G3[0, 1] = [0, 1]; G3[1, 0] = [0, 1]
    r = min_norm(2, blocks=[hsym(G3)]); chk('5. 3x3 block (PSD cone): rho', r['rho'], 0.0, 1e-6)
    r = min_norm(2, np.array([[1.0, 0.5], [-1.0, 0.5], [0.0, 0.5]])); chk('5. 3x3: diagonal branches only', r['rho'], 0.5)
    p = pred(2, 1.0, blocks=[(np.zeros(3), hsym(G3))], Delta=0.1, norm='2'); chk('5. pred ball, 3x3 block', p['pred'], 0.0, 1e-6)
    p = pred(2, 1.0, blocks=[(np.array([0.0, 0.0, -1.0]), hsym(G3))], Delta=0.1, norm='inf')
    chk('5. pred box, 3x3 block, sigma_3 1 below', p['pred'], 0.0, 1e-6)
    # 6. f = x1 with -x1 <= 0 active (mu = 1)
    e1 = np.zeros(NZ); e1[0] = 1.0
    r = min_norm(NZ, np.array([e1]), H=np.array([-e1])); chk('6. active constraint: rho', r['rho'], 0.0); chk('6. mu', r['mu'][0], 1.0)
    r = min_norm(NZ, np.array([e1])); chk('6. without the constraint', r['rho'], 1.0)
    p = pred(NZ, 1.0, np.zeros(1), np.array([e1]), cons=[(0.0, -e1)], Delta=0.1); chk('6. pred, active', p['pred'], 0.0)
    p = pred(NZ, 1.0, np.zeros(1), np.array([e1]), cons=[(-0.01, -e1)], Delta=0.1); chk('6. pred, 0.01 inside', p['pred'], 0.01)
    # 7. the eps discount: g1 = a at gamma, g2 = -a at gamma (1 - delta): pred = min(Delta |a|_1, delta gamma / 2)
    a_ = np.zeros(NZ); a_[0] = 2.0; gamma, delta = 100.0, 1e-3
    for D_ in (1e-3, 1e-2, 0.1):
        p = pred(NZ, gamma, np.array([0.0, -delta * gamma]), np.array([a_, -a_]), Delta=D_)
        chk(f'7. eps discount, Delta {D_:g}', p['pred'] * gamma, min(D_ * 2.0, delta * gamma / 2))
    # 8. a PSD constraint H = diag(x1, 1) >= 0 at x1 = 0, f = x1: stationary with Lam = e1 e1^T
    dH = np.zeros((NZ, 2, 2)); dH[0, 0, 0] = 1.0
    Q = np.array([[1.0], [0.0]]); DHq = np.array([_sv(Q.T @ dH[j] @ Q) for j in range(NZ)])
    r = min_norm(NZ, np.array([e1]), psd=[DHq]); chk('8. PSD normal cone: rho', r['rho'], 0.0, 1e-6)
    p = pred(NZ, 1.0, np.zeros(1), np.array([e1]), psd=[(np.diag([0.0, 1.0]), dH)], Delta=0.1); chk('8. pred, PSD row', p['pred'], 0.0, 1e-6)
    return int(sum(res)), len(res)


# ============================================================== the record and the plant
def load_record(path):
    z = np.load(path, allow_pickle=True)
    js = os.path.splitext(path)[0] + '.json'
    J = {}
    if os.path.exists(js):
        try:
            J = json.load(open(js))
        except Exception:
            J = {}
    return z, J, js


def _pk_from(D, flags, z, mimo):
    from packer import Packer, PackerMIMO
    R = len(np.atleast_1d(z['d']))
    if mimo:
        m, p = int(z['m']), int(z['p'])
        arow = (np.asarray(z['d'], complex), np.asarray(z['e'], complex), np.asarray(z['f'], complex),
                np.asarray(z['Bc'], complex).reshape(R, p))
        nvec = np.asarray(z['nvec'], complex).reshape(m, R); K = np.asarray(z['K'], complex).reshape(m, p)
        xa = np.asarray(z['xa'], float).reshape(m, 2); xs = np.asarray(z['xs'], float).reshape(p, 1)
        return PackerMIMO(D, flags, arow, nvec, K, xa, xs, None, None), (arow, nvec, K, xa, xs)
    arow = tuple(np.asarray(z[k], complex) for k in ('d', 'e', 'f'))
    nvec = np.asarray(z['nvec'], complex); K = complex(np.asarray(z['K']).ravel()[0])
    xa = np.asarray(z['xa'], float).ravel(); xs = np.atleast_1d(np.asarray(z['xs'], float))
    return Packer(D, flags, arow, nvec, K, xa, xs, None, None), (arow, nvec, K, xa, xs)


def build(z, J, ops, margin, scale_from=None):
    """BFSPlant, flags (real controller, ctrl_modal from the record's modal cut, MIMO), the packer from the
    record (its scales, or --scale-from's), and the round-trip error."""
    from bfs.plant import BFSPlant
    from fe_codesign import parse_case
    import controller as ctl
    R = len(np.atleast_1d(z['d']))
    D = BFSPlant(ops, r=R); D.pos_margin = float(margin)
    flags = parse_case(str(J.get('case', 'K+B+C'))); flags['real_ctrl'] = True
    mimo = 'struct' in z.files
    struct = str(z['struct']) if mimo else 'siso'
    orders = [int(v) for v in np.atleast_1d(z['loop_orders'])] if (mimo and struct == 'decentral') else [R]
    cut = ctl.modal_cut(orders)
    flags['ctrl_modal'] = bool(all(np.asarray(z['e'])[i] == 0 and np.asarray(z['f'])[i] == 0 for i in cut))
    if mimo:
        flags.update(m=int(z['m']), p=int(z['p']), struct=struct,
                     loop_orders=[int(v) for v in np.atleast_1d(z['loop_orders'])])
    pk, parts = _pk_from(D, flags, z, mimo)
    note = None
    if scale_from:
        z2 = np.load(scale_from, allow_pickle=True)
        pk2, _ = _pk_from(D, flags, z2, mimo)
        same = (np.array_equal(pk2.free, pk.free) if mimo else
                (pk2.cmask is None and pk.cmask is None) or np.array_equal(pk2.cmask, pk.cmask))
        if not same or np.shape(pk2.sc) != np.shape(pk.sc):
            raise SystemExit(f"--scale-from {scale_from}: its free mask (gauge / modal cut) differs from the record's")
        pk.sc = np.array(pk2.sc, float); note = f"scales from {scale_from}"
    z0 = pk.z0()
    u = pk.unpack(z0)
    arow, nvec, K, xa, xs = parts
    err = max(float(np.max(np.abs(np.asarray(u[0][k]) - np.asarray(arow[k])))) for k in range(len(arow)))
    err = max(err, float(np.max(np.abs(np.asarray(u[1]) - np.asarray(nvec)))), float(np.max(np.abs(np.asarray(u[2]) - np.asarray(K)))),
              float(np.max(np.abs(np.asarray(u[3], float).ravel() - np.asarray(xa, float).ravel()))),
              float(np.max(np.abs(np.asarray(u[4], float).ravel() - np.asarray(xs, float).ravel()))))
    return D, flags, pk, z0, err, struct, note


def coord_names(pk, flags, NZ):
    """A name per packed coordinate: d3, e2, Bc[1,0], Cc[0,4], Dc[1,1], xa1.y, xs0 (SISO: n3, K, xa.x, xs)."""
    names = []
    if getattr(pk, 'mimo', False):
        r, m, p = pk.r, pk.m, pk.p
        th = []
        for nm_, (a_, b_) in pk.off.items():
            for q in range(b_ - a_):
                th.append({'d': f'd{q}', 'e': f'e{q}', 'f': f'f{q}', 'Bc': f'Bc[{q // p},{q % p}]',
                           'n': f'Cc[{q // r},{q % r}]', 'K': f'Dc[{q // p},{q % p}]'}[nm_])
        sel = [th[q] for q in np.flatnonzero(pk.free)]
        names += sel + ([] if pk.real_ctrl else [s_ + '.im' for s_ in sel])
        if flags.get('B'):
            names += [f'xa{i}.{"xy"[j]}' for i in range(m) for j in range(pk.pd_a)]
        if flags.get('C'):
            names += [f'xs{i}' for i in range(p) for j in range(pk.pd_s)]
    else:
        r = len(pk.d0)
        th = [f'd{i}' for i in range(r)] + [f'e{i}' for i in range(r - 1)] + [f'f{i}' for i in range(r - 1)] + \
             [f'n{i}' for i in range(r)] + ['K']
        k = pk.nth
        sel = [th[i] for i in range(k) if (pk.cmask is None or pk.cmask[i])]
        names += sel + ([] if pk.real_ctrl else [s_ + '.im' for s_ in sel])
        if flags.get('B'):
            names += ['xa.x', 'xa.y'][:pk.pd_a]
        if flags.get('C'):
            names += ['xs']
    if len(names) != NZ:
        names = [f'z{j}' for j in range(NZ)]
    return names


def groups_of(names):
    """[(group, indices)]: the controller kinds (d, e, f, Bc, Cc / n, Dc / K), xa, xs."""
    out = {}
    for j, nm in enumerate(names):
        key = nm.split('[')[0].split('.')[0].rstrip('0123456789') if not nm.startswith(('xa', 'xs')) else nm[:2]
        out.setdefault(key, []).append(j)
    return [(k, np.array(v)) for k, v in out.items()]


# ============================================================== the plant side
def grid_seeds(z, gam, eps_max, whi):
    ws, gs = np.asarray(z['omega_grid'], float), np.asarray(z['gain'], float)
    ix = [k for k in range(len(gs)) if (k == 0 or gs[k] >= gs[k - 1]) and (k == len(gs) - 1 or gs[k] >= gs[k + 1])
          and gs[k] >= gam * (1 - eps_max)]
    seeds = [float(ws[k]) for k in ix]
    nps = 0
    if 'peakset' in z.files:
        for w in np.atleast_1d(z['peakset']):
            if 0.0 <= float(w) <= whi and np.interp(float(w), ws, gs) >= gam * (1 - 2 * eps_max):
                seeds.append(float(w)); nps += 1
    out = []
    for w in sorted(seeds):
        if all(abs(w - q) > 1e-6 for q in out):
            out.append(min(max(w, 0.0), whi))
    return out, len(ix), nps


def clip_band(lat, z, res, whi):
    """peak_tops results (w, s) inside the band: a Newton top beyond whi (the curve still rising at the band's end)
    is the end point whi with its own value -- the problem's max is over [0, whi] (the descent test and the
    completeness and certificate loops re-locate from points at or near whi; locate() does the same for the seeds)"""
    res = [(float(w_), float(s_)) for w_, s_ in res]
    if any(w_ > whi for w_, _ in res):
        sw = float(lat.sigmas(z, [whi])[0])
        res = [((whi, sw) if w_ > whi else (w_, s_)) for w_, s_ in res]
    return res


def merge_tops(tops):
    """Tops closer than MERGE are one: the higher kept."""
    kept = []
    for t in sorted(tops, key=lambda q: -q['s']):
        if all(abs(t['w'] - k['w']) > MERGE for k in kept):
            kept.append(t)
    return sorted(kept, key=lambda q: q['w'])


def locate(lat, z0, seeds, whi, log):
    """peak_tops from the seeds; edge stops retried from their stop point (at most 3 times, kept when the
    value rose); a top beyond whi becomes the end point; merged."""
    sts = [{} for _ in seeds]
    res = lat.peak_tops(z0, seeds, RAD, states=sts)
    tops = [dict(w=float(w), s=float(s), top=bool(st.get('top')), edge=bool(st.get('edge')), seed=float(sd), end=False)
            for (w, s), st, sd in zip(res, sts, seeds)]
    for it in range(3):
        ed = [t for t in tops if t['edge'] and not t['top'] and t['w'] < whi]
        if not ed:
            break
        sts2 = [{} for _ in ed]
        res2 = lat.peak_tops(z0, [t['w'] for t in ed], RAD, states=sts2)
        for t, (w, s), st in zip(ed, res2, sts2):
            log.append(f"edge stop at {t['w']:.6f} ({t['s']:.6f}) retried -> {w:.6f} ({s:.6f}, top {bool(st.get('top'))})")
            if s >= t['s']:
                t.update(w=float(w), s=float(s), top=bool(st.get('top')), edge=bool(st.get('edge')))
    beyond = [t for t in tops if t['w'] > whi]
    if beyond:
        sw = float(lat.sigmas(z0, [whi])[0])
        for t in beyond:
            log.append(f"top {t['w']:.6f} beyond whi {whi:.6f}: the end point whi ({sw:.6f}) instead")
            t.update(w=float(whi), s=sw, top=True, edge=False, end=True)
    return merge_tops(tops)


def driver_block(lat, z, w, k):
    """The k leading singular values at w and the entry gradients Gc[a, b] = G(x_b, z_a) + i G(x_b, i z_a)
    (k x k x NZ; lattice._block's construction for any k), on the driver (gain_k + D.grads): the r x r
    Clarke block of a multiplicity >= 3 top, the end point's row."""
    import lattice
    dz = lat.design(z)
    d = lattice._design(dz)
    P = d['cl'].gain_k(float(w), dz['arow'], dz['nvec'], dz['K'], k=int(k))
    grad = lambda g_: np.asarray(lat.pk.grad(g_), float)
    G = lambda x_, z_: grad(lattice._grads(d, dz, x_, z_))
    NZ = len(z)
    Gc = np.zeros((k, k, NZ), complex)
    for a_ in range(k):
        Gc[a_, a_] = G(P[a_]['x'], P[a_]['z'])
        for b_ in range(k):
            if a_ != b_:
                Gc[a_, b_] = G(P[b_]['x'], P[a_]['z']) + 1j * G(P[b_]['x'], 1j * P[a_]['z'])
    return np.array([float(P[a_]['sigma']) for a_ in range(k)]), Gc


def cl_eig_near(lat, z, lam):
    """The closed-loop eigenvalue nearest lam at design z (lattice._nearest on the driver, shift 1e-3 off
    the eigenvalue so the factor is regular)."""
    import lattice
    dz = lat.design(z)
    d = lattice._design(dz)
    lams, _, _ = lattice._nearest(d['cl'], complex(lam) + 1e-3, dz, 1)
    return min(lams, key=lambda l_: abs(l_ - lam)) if lams else None


# ============================================================== main
def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('record', nargs='?')
    ap.add_argument('--ops')
    ap.add_argument('--eps', default='1e-6,1e-5,1e-4,1e-3,1e-2', help='the activity ladder (relative to gamma)')
    ap.add_argument('--eps-max', dest='eps_max', type=float, default=None,
                    help='the residual\'s largest eps (default max(1e-2, max --eps)); seeds, the sigma_2 blocks and '
                         'the pred piece set use --eps-pred')
    ap.add_argument('--eps-pred', dest='eps_pred', type=float, default=0.05,
                    help='the pieces of the decrease LP and of the descent test: every top within this of gamma '
                         '(default 0.05, at least eps_max; a peak just below eps_max rises under the step)')
    ap.add_argument('--deltas', default='1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1')
    ap.add_argument('--trust', type=float, default=None, help="the SQP's radius (default the record's sqp_trust)")
    ap.add_argument('--complete', choices=['off', 'on'], default='off')
    ap.add_argument('--complete-eps', dest='complete_eps', default=None, help='default eps_max,1e-3')
    ap.add_argument('--strip', choices=['off', 'on'], default='off')
    ap.add_argument('--cl-line', dest='cl_line', choices=['problem', 'qp'], default='problem')
    ap.add_argument('--fd', type=int, default=6, help='FD directions (0: off)')
    ap.add_argument('--fd-cl', dest='fd_cl', choices=['off', 'on'], default='on')
    ap.add_argument('--descent', choices=['off', 'on', 'certify'], default='on')
    ap.add_argument('--descent-deltas', dest='descent_deltas', default='1e-7,1e-6,1e-5,1e-4,1e-3')
    ap.add_argument('--descent-ts', dest='descent_ts', default='1')
    ap.add_argument('--scale-from', dest='scale_from', default=None)
    ap.add_argument('--wpad', type=float, default=3.0, help="the line's band beyond whi (the run's --wpad)")
    ap.add_argument('--margin', type=float, default=0.05, help="the ports' margin (the run's --bfs-margin)")
    ap.add_argument('--progress', default=PROGRESS,
                    help="the run's progress rule k:tol (--stop-progress; default 8:0.0015, the BFS rule since 20261010r; "
                         "the 20261010g/n runs used 3:0.0015): the class boundary of the verdict is its per-round scale tol / k")
    ap.add_argument('--eps-s', dest='eps_s', type=float, default=None, help="stability margin (default the JSON's eps)")
    ap.add_argument('--lu-cap', dest='lu_cap', type=int, default=8)
    ap.add_argument('--json', default=None)
    ap.add_argument('--selftest', action='store_true')
    return ap.parse_args(argv)


def _f(v, p=3):
    return 'n/a' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}e}"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def stop_reason(J):
    codes = {0: 'stationarity test (--sqp-stat)', 4: 'iteration budget', 6: 'trust radius below 1e-10', 12: 'QP failure'}
    h = (J.get('history') or [{}])[-1] if J.get('history') else {}
    why = []
    for k_, txt in (('exchange_progress_stop', 'progress rule'), ('deadline_stop', 'deadline'),
                    ('exchange_converged', 'exchange converged'), ('exchange_fixed_point', 'exchange fixed point'),
                    ('exchange_stalled', 'exchange stalled')):
        if J.get(k_):
            why.append(txt)
    tc = h.get('termination_code')
    s = (', '.join(why) or 'unknown') + (f" (last round code {tc}: {codes.get(tc, '?')}" if tc is not None else '') + \
        (f", stat_value {h['stat_value']:.4g}, {h.get('inner_iters')} inner iterations)" if 'stat_value' in h else (')' if tc is not None else ''))
    return s, dict(why=why, termination_code=tc, stat_value=h.get('stat_value'), inner_iters=h.get('inner_iters'),
                   termination_codes=J.get('termination_codes'))


def main(argv=None):
    import sparse_lu; sparse_lu.ORDER = 'mumps'
    from bfs import mpipool
    if mpipool.active() and mpipool.comm().Get_rank() > 0:
        mpipool.serve()                       # never returns
    a = _args(argv)
    P = lambda *s: print(*s, flush=True)
    T0 = time.time(); tm = {}
    P("convex core self-test (designed toys, design.md 6.2):")
    nok, ntot = run_toys(verbose=bool(a.selftest))
    P(f"  {nok}/{ntot} ok" + ('' if nok == ntot else '  <-- THE CONVEX CORE FAILS ITS TOYS: no reading below is trustworthy'))
    if a.selftest:
        return 0 if nok == ntot else 1
    if not a.record or not a.ops:
        raise SystemExit("RECORD.npz and --ops OPS are required (or --selftest)")
    import fullpar, lattice, sqp_inner
    import controller as ctl
    z, J, jpath = load_record(a.record)
    gam = float(z['gamma'])
    eps_s = float(a.eps_s if a.eps_s is not None else J.get('eps', 0.05) or 0.05)
    mg = 1e-3 * eps_s
    Lc = -eps_s                                       # the compensator line (--ctrl-line eps: ctrl_max = -eps_s)
    whi = float(z['whi']) if 'whi' in z.files else float(J.get('whi', 10.0) or 10.0)
    eps_list = sorted(set(float(v) for v in a.eps.split(',')))
    eps_max = max(float(a.eps_max) if a.eps_max is not None else 1e-2, max(eps_list))   # every listed eps sees all its pieces
    eps_all = sorted(set(eps_list + [CERT_RTOL]))
    deltas = [float(v) for v in a.deltas.split(',')]
    trust = a.trust if a.trust is not None else (float(z['sqp_trust']) if 'sqp_trust' in z.files else None)
    D, flags, pk, z0, rt_err, struct, sc_note = build(z, J, a.ops, a.margin, a.scale_from)
    NZ = len(z0)
    names = coord_names(pk, flags, NZ); groups = groups_of(names)
    stop_txt, stop_rec = stop_reason(J)
    out = dict(record=os.path.abspath(a.record), json=jpath if J else None, struct=struct, NZ=NZ, gamma_record=gam,
               eps_s=eps_s, whi=whi, line_band=[0.0, whi + a.wpad], eps_max=eps_max, trust=trust, stop=stop_rec,
               args=vars(a), scale_note=sc_note, coord_names=names)
    P(f"record {a.record}")
    P(f"  {struct}, gamma {gam:.6f}, NZ {NZ} (modal {flags['ctrl_modal']}), band [0, {whi:.4f}], line [0, {whi + a.wpad:.4f}], "
      f"eps_s {eps_s:g}, margin {a.margin:g}, sqp_trust {trust}; packer round trip {rt_err:.1e}"
      + (f"; {sc_note}" if sc_note else '') + f"; stop: {stop_txt}")
    P(f"  groups: " + ', '.join(f"{k} {len(v)}" for k, v in groups))
    # ---------------------------------------------------------------- the oracle
    lattice.LU_CAP = int(a.lu_cap)
    fullpar.start(1, D=D)
    pool = fullpar.POOL if mpipool.active() else None
    lat = lattice.LatticeOracle(D, pk, flags, pool=pool, real=True)
    D.band_of = lambda: whi + a.wpad                  # the run's line band (run_case_epi: a.whi + a.wpad)
    P(f"  lattice oracle on {lat.size} rank(s), LU_CAP {lattice.LU_CAP}")
    try:
        _run(a, P, out, tm, z, J, gam, eps_s, mg, Lc, whi, eps_list, eps_max, eps_all, deltas, trust, D, flags, pk, z0,
             NZ, names, groups, lat, lattice, sqp_inner, ctl, stop_txt, rt_err, nok, ntot)
    finally:
        fullpar.stop()
    tm['total'] = time.time() - T0
    out['timing'] = tm
    P(f"timing: " + ', '.join(f"{k} {v:.0f} s" for k, v in tm.items()))
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(_jsonable(out), fh, indent=1)
        P(f"JSON: {a.json}")
    return 0


def _run(a, P, out, tm, z, J, gam, eps_s, mg, Lc, whi, eps_list, eps_max, eps_all, deltas, trust, D, flags, pk, z0,
         NZ, names, groups, lat, lattice, sqp_inner, ctl, stop_txt, rt_err, nok, ntot):
    log = []
    # the pieces of the decrease LP and of the descent test: every top within eps_p (>= eps_max) of gamma.  With
    # P(eps_max) alone the SISO 598.11 step at Delta = 1e-5 read -8.96e-3 over its 3 tops while the top at
    # 1.373 (0.9862 gamma, outside P(0.01)) rose to 0.9930: the true max was -7.04e-3 and the level set found it
    eps_p = max(float(a.eps_pred), eps_max)
    out['eps_pred'] = eps_p
    # ---------------------------------------------------------------- 1. tops
    t0 = time.time()
    # seeded from the grid maxima within eps_p: a top within eps_max can have its grid point below gamma (1 - eps_max)
    seeds, n_grid, n_ps = grid_seeds(z, gam, eps_p, whi)
    tops = locate(lat, z0, seeds, whi, log)
    tm['tops'] = time.time() - t0
    P(f"tops: {len(seeds)} seeds ({n_grid} grid maxima within {eps_p:g} of gamma, {n_ps} peakset members) -> "
      f"{len(tops)} tops in {tm['tops']:.0f} s")
    for s_ in log:
        P("  " + s_)
    # ---------------------------------------------------------------- 2. completeness (level set)
    comp = []
    if a.complete == 'on':
        t0 = time.time()
        ce = sorted(set(float(v) for v in (a.complete_eps.split(',') if a.complete_eps else [eps_max, 1e-3])), reverse=True)
        # WATCHED BY LOCATION (the review of 2026-10-17): a region within 1e-3 of a known top is that top's (tol 1:
        # lattice._level then compares the row's value with 0).  With tol 1e-9 a region around a known top whose
        # top estimate (the cubic Hermite fit on its crossings, wide at gamma (1 - 1e-2)) overshot the Newton top by
        # 1e-9 was reported, re-located onto the known top, and the loop stopped there -- the rest of the band
        # unexamined and the piece set never certified.  A located region that lands on a known top is watched at
        # its estimate from the next call on.  (A second top inside a known top's region is not separated: the
        # finer levels of --complete-eps are what can separate it.)
        for ec in ce:
            extra = []
            for rep in range(6):
                lev = lat.level(z0, gam * (1 - ec), 0.0, whi, watch=(1.0, [(t['w'], t['s']) for t in tops] + extra))
                comp.append(dict(eps=ec, outcome=lev['outcome'], uncovered=float(lev['uncovered']), cross=lev['cross'],
                                 shifts=int(lev['shifts']), waves=int(lev['waves'])))
                P(f"  level set at gamma (1 - {ec:g}): {lev['outcome']}, uncovered {lev['uncovered']:.2e}, "
                  f"{lev['shifts']} shifts, regions {[(round(w_, 5), round(v_ / gam, 7)) for w_, v_ in lev['cross']]}")
                if not lev['cross']:
                    break
                sts = [{} for _ in lev['cross']]
                new0 = lat.peak_tops(z0, [float(w_) for w_, _ in lev['cross']], RAD, states=sts)
                new = clip_band(lat, z0, new0, whi)
                n0 = len(tops)
                tops = merge_tops(tops + [dict(w=float(w_), s=float(s_), top=bool(st.get('top')) or w0_ > whi, edge=bool(st.get('edge')) and w0_ <= whi,
                                               seed=float(c_[0]), end=bool(w0_ > whi))
                                          for (w_, s_), (w0_, _), st, c_ in zip(new, new0, sts, lev['cross'])])
                extra += [(float(c_[0]), float(s_)) for (w_, s_), c_ in zip(new, lev['cross'])]
                if len(tops) == n0:
                    P("    the regions' tops are known tops (a region seen more than 1e-3 from its top): asked again "
                      "with the regions watched at their estimates")
                    comp[-1]['outcome'] += ' (re-located onto known tops)'
        tm['complete'] = time.time() - t0
    out['completeness'] = comp
    final = {}
    for c_ in comp:
        final[c_['eps']] = c_['outcome']                 # the last call at each level decides
    certified = bool(final) and all(o_ in ('clear', 'watched') for o_ in final.values())
    tops = [t for t in tops if t['s'] >= gam * (1 - eps_p)]
    if not any(t['s'] >= gam * (1 - eps_max) for t in tops):
        raise SystemExit("no top within eps_max of gamma: the record's curve and the design disagree")
    # ---------------------------------------------------------------- 3. sigma_2, enrolment, the Evaluator
    t0 = time.time()
    inner = [t for t in tops if not t['end']]
    S2 = lat.sigmas(z0, [t['w'] for t in inner], k=2) if inner else np.zeros((0, 2))
    for t, (s1, s2) in zip(inner, S2):
        t['s2'] = float(s2)
    # the blocks are enrolled over the decrease LP's piece set P(eps_p), not P(eps_max): a sigma_2 within eps_p of
    # gamma is a lower branch at the same top and rises under the step like a lower piece (central 123.90: the
    # sigma_2 of the top at 0.926, 1.34e-2 below, left out at eps_max = 1e-2, crossed gamma at Delta = 1e-4,
    # actual +1.37e-3 against the predicted -1.70e-4).  The residual at eps still counts only sigma_k >= gamma (1 - eps)
    enroll = [i for i, t in enumerate(inner) if t['s2'] >= gam * (1 - eps_p)]
    tm['sigma2'] = time.time() - t0
    t0 = time.time()
    sqp_inner.LAT_STRIP = (a.strip == 'on')
    W = np.array([t['w'] for t in inner], float)
    ev = sqp_inner.Evaluator(D, pk, flags, W, eps_s, -eps_s, enroll, kink_block=True, mg=mg, ctrl_block=False,
                             follow={round(float(w), 6): RAD for w in W}, cl_block=True, lattice=lat)
    ev.w_star = {round(float(w), 6): float(w) for w in W}
    e = ev(z0)
    tm['evaluator'] = time.time() - t0
    P(f"evaluation (Evaluator: {len(W)} followed rows, {len(enroll)} block(s), the line on [0, {whi + a.wpad:.4f}]"
      f"{', the strip' if a.strip == 'on' else ''}): {tm['evaluator']:.0f} s")
    pieces = []
    for i, t in enumerate(inner):
        k6 = round(float(W[i]), 6)
        wf = float(ev.w_star.get(k6, W[i]))
        fs = ev.follow_state.get(k6, {})                  # the Evaluator's own Newton state at this row (lattice._newton)
        pc = dict(w=wf, s=float(e['s'][i]), g=np.asarray(e['Gs'][i], float), top=bool(fs.get('top', t['top'])),
                  edge=bool(fs.get('edge', t['edge'])), end=False,
                  seed=t['seed'], moved=abs(wf - t['w']), s2=t.get('s2'), sig=np.array([e['s'][i]]), Hs=None)
        if i in e['blocks']:
            B = e['blocks'][i]
            pc['sig'] = np.asarray(B['sig'], float); pc['Hs'] = hsym(np.asarray(B['Gc'])); pc['s2'] = float(B['sig'][1])
        pieces.append(pc)
    for t in tops:
        if t['end']:
            sig, Gc = driver_block(lat, z0, t['w'], 2)
            pieces.append(dict(w=t['w'], s=float(sig[0]), g=np.real(Gc[0, 0]), top=True, edge=False, end=True, seed=t['seed'],
                               moved=0.0, s2=float(sig[1]), sig=sig if sig[1] >= gam * (1 - eps_p) else sig[:1],
                               Hs=hsym(Gc) if sig[1] >= gam * (1 - eps_p) else None))
    # the Evaluator's own Newton can walk two members onto one top (decentral 369.87: 1.125 -> 1.2172)
    kept = []
    for pc in sorted(pieces, key=lambda q: -q['s']):
        j = next((n for n, k in enumerate(kept) if abs(pc['w'] - k['w']) <= MERGE), None)
        if j is None:
            kept.append(pc); continue
        k = kept[j]
        # one top reached by two rows: their values agree to Newton's accuracy, not exactly, so "the higher" alone
        # kept the walked edge-stop row on central 123.90 (4.388246 flagged NOT A TOP).  Within 1e-9 relative: the
        # converged row, then the one with a block, then the one the Evaluator walked least
        if abs(pc['s'] - k['s']) <= 1e-9 * abs(k['s']) and \
                (not pc['top'], pc['Hs'] is None, pc['moved']) < (not k['top'], k['Hs'] is None, k['moved']):
            kept[j] = pc; pc, k = k, pc
        log.append(f"row at {pc['w']:.6f} ({pc['s']:.9f}, {'top' if pc['top'] else 'not a top'}, walked {pc['moved']:.1e}) "
                   f"merged into the top at {k['w']:.6f} ({k['s']:.9f}) within {MERGE:g}")
    pieces = sorted(kept, key=lambda q: -q['s'])
    moved = [pc for pc in pieces if pc['moved'] > 1e-6 and pc['Hs'] is None]
    if moved:
        S2m = lat.sigmas(z0, [pc['w'] for pc in moved], k=2)
        for pc, (s1, s2) in zip(moved, S2m):
            pc['s2'] = float(s2)
            if pc['s2'] >= gam * (1 - eps_p):
                # its sigma_2 now qualifies at the walked top (enrolment was decided at the seed's top): the 2 x 2 block
                # on the driver (driver_block, FD-checked against sigma_1, sigma_2 at +-h in central/block_fd.py)
                sigb, Gcb = driver_block(lat, z0, pc['w'], 2)
                pc['sig'] = sigb; pc['Hs'] = hsym(Gcb)
                log.append(f"top {pc['w']:.6f}: walked by the Evaluator, its sigma_2 {sigb[1]:.6f} within {eps_p:g}: "
                           f"2x2 block built on the driver")
    # multiplicity >= 3: the r x r block on the driver
    for pc in pieces:
        if pc['Hs'] is not None and not pc['end']:
            sig3, Gc3 = driver_block(lat, z0, pc['w'], 3)
            pc['s3'] = float(sig3[2])
            if sig3[2] >= gam * (1 - eps_p):
                k = 3
                while k < 6:
                    sigk, Gck = driver_block(lat, z0, pc['w'], k + 1)
                    if sigk[k] < gam * (1 - eps_p):
                        break
                    k += 1; sig3, Gc3 = sigk, Gck
                pc['sig'] = sig3[:k]; pc['Hs'] = hsym(Gc3[:k, :k])
                log.append(f"top {pc['w']:.6f}: multiplicity {k} within {eps_p:g} (sigma_{k} {sig3[k - 1]:.6f}): {k}x{k} block (PSD cone)")
    gref = max(pc['s'] for pc in pieces)
    for pc in pieces:
        pc['gn2'] = float(np.linalg.norm(pc['g'])); pc['gn1'] = float(np.sum(np.abs(pc['g'])))
        pc['entry'] = 1.0 - pc['s'] / gref
    # ---------------------------------------------------------------- 4. constraints
    cons = []
    lat_eigs = e.get('lat_eigs')
    lams = list(lat_eigs[0]) if lat_eigs is not None else []
    shift = eps_s + (2 * mg if a.cl_line == 'qp' else 0.0)
    for j, (re_, g_) in enumerate(e.get('al_rows_all', [])):
        g_ = np.asarray(g_, float)
        if not np.any(g_):
            continue                                       # the placeholder (no eigenvalue within reach)
        lam = complex(lams[j]) if j < len(lams) else complex(re_, np.nan)
        cons.append(dict(name=f"closed loop {lam.real:+.6f}{lam.imag:+.4f}i", kind='cl', c=float(re_) + shift, h=g_,
                         scale=eps_s, lam=lam))
    if getattr(pk, 'modal', None):
        arow = pk.unpack(z0)[0]
        Ac = ctl.A_of(*arow)
        rown = []
        for bl in pk.modal:
            pb = np.linalg.eigvals(Ac[np.ix_(list(bl), list(bl))])
            pd_ = ' '.join(f"{p_.real:+.4f}{p_.imag:+.3f}i" for p_ in sorted(pb, key=lambda q: -q.imag))
            if len(bl) == 1:
                rown.append(f"modal ({bl[0]}) pole [{pd_}]")
            else:
                rown += [f"modal ({bl[0]},{bl[1]}) v1 (trace) [{pd_}]", f"modal ({bl[0]},{bl[1]}) v2 (det) [{pd_}]"]
    else:
        rown = []
    for j, (v_, g_) in enumerate(e.get('ctrl_rows', [])):
        nm = rown[j] if j < len(rown) else f"compensator row {j}"
        cons.append(dict(name=nm, kind='modal', c=min(float(v_) - Lc, 0.0), c_raw=float(v_) - Lc, h=np.asarray(g_, float), scale=eps_s))
    pn = []
    xa_, xs_ = pk.unpack(z0)[3], pk.unpack(z0)[4]
    na_ = len(np.asarray(xa_, float).reshape(-1, 2)); ns_ = len(np.asarray(xs_, float).reshape(-1, 1))
    if flags.get('B'):
        for i in range(na_):
            tag = f"actuator {i}" if na_ > 1 or struct_is_mimo(pk) else "actuator"
            pn += [f"{tag} x >= x_lo + m", f"{tag} x <= x_hi - m", f"{tag} y <= y_hi - m", f"{tag} clearance >= m"]
    if flags.get('C'):
        for i in range(ns_):
            tag = f"sensor {i}" if ns_ > 1 or struct_is_mimo(pk) else "sensor"
            pn += [f"{tag} x_s >= floor start", f"{tag} x_s + L <= x_hi - m"]
    for j, (c_, g_) in enumerate(e.get('pos_rows', [])):
        cons.append(dict(name=pn[j] if j < len(pn) else f"position row {j}", kind='pos', c=float(c_), h=np.asarray(g_, float),
                         scale=float(a.margin)))
    for cc in cons:
        cc['ct'] = cc['c'] / cc['scale']; cc['entry'] = max(-cc['ct'], 0.0); cc['hn'] = float(np.linalg.norm(cc['h']))
    # every eigenvalue the line SAW within eps_pred of it must have a row: one without (its left vector failed,
    # sqp_inner's "the rightmost seen has no row") is an active constraint missing from the normal cone, and the
    # residual would read non-stationarity the cone removes (the SQP lacks the row as well)
    seen_ = [complex(l_) for l_ in (lat_eigs[1] if lat_eigs is not None else [])]
    no_row = []
    for l_ in seen_:
        ent_ = -(l_.real + eps_s) / eps_s
        if ent_ <= eps_p and all(min(abs(l_ - q_), abs(l_ - np.conj(q_))) > 1e-6 * max(1.0, abs(l_)) for q_ in lams) \
                and all(min(abs(l_ - q_), abs(l_ - np.conj(q_))) > 1e-6 * max(1.0, abs(l_)) for q_, _ in no_row):
            no_row.append((l_, max(ent_, 0.0)))
    # the closed-loop Hermite block: only for a near-defective active pair (design.md 2.5)
    herm = None
    cl_j = [j for j, cc in enumerate(cons) if cc['kind'] == 'cl']
    cls = [cons[j] for j in cl_j]
    if e.get('hermite_cl') is not None and len(cls) >= 2 and abs(cls[0]['lam'] - cls[1]['lam']) <= 1e-2:
        Hc = e['hermite_cl']
        herm = dict(H0=np.real(np.asarray(Hc['H0'])), dH=np.real(np.array(Hc['dH'])).reshape(NZ, *np.shape(Hc['H0'])),
                    rows=(0, 1), entry=max(cls[0]['entry'], cls[1]['entry']))
        log.append(f"closed-loop pair {cls[0]['lam']:.5f}, {cls[1]['lam']:.5f} within 1e-2: the Hermite block replaces their rows")
    # ---------------------------------------------------------------- 5. the report so far
    P(f"gamma at the tops {gref:.9f} against the record's {gam:.9f}: relative {gref / gam - 1:+.2e} "
      f"({'agree' if abs(gref / gam - 1) <= CERT_RTOL else 'DISAGREE beyond the certificate rtol 1e-6'}"
      + (': a top ABOVE the certified gamma -- the record is wrong' if gref > gam * (1 + CERT_RTOL) else
         (': a top is MISSING (run --complete on)' if gref < gam * (1 - CERT_RTOL) else '')) + ')')
    P(f"line: clear {e['clr']}, alpha {e['al']:+.7f} (record {float(z['alpha']) if 'alpha' in z.files else np.nan:+.7f}); "
      f"{len(cls)} closed-loop row(s), {sum(c_['kind'] == 'modal' for c_ in cons)} modal row(s), "
      f"{sum(c_['kind'] == 'pos' for c_ in cons)} position row(s)")
    if no_row:
        P(f"  WARNING: {len(no_row)} eigenvalue(s) the line saw within eps_pred of it have NO row (no gradient): "
          + ', '.join(f"{l_.real:+.6f}{l_.imag:+.4f}i (entry eps {en_:.1e})" for l_, en_ in no_row)
          + " -- their normal cones are missing from the residual and their rows from the decrease LP")
    for s_ in log:
        if s_.startswith(('row at', 'top ', 'closed-loop pair')):
            P("  " + s_)
    P(f"pieces (tops within eps_pred = {eps_p:g} of gamma; the residual uses those within eps <= eps_max = {eps_max:g}), by value:")
    P(f"  {'w':>10s} {'s/gamma - 1':>12s} {'sigma2 gap':>10s} {'mult@eps':>16s} {'|g|_2':>9s} {'|g|_1':>9s}  flags")
    for pc in pieces:
        gap2 = 1 - pc['s2'] / pc['s'] if pc.get('s2') is not None else np.nan
        mult = ','.join(str(int(np.sum(pc['sig'] >= gref * (1 - ep)))) for ep in eps_all if pc['s'] >= gref * (1 - ep)) or '-'
        fl = ('top' if pc['top'] else 'NOT A TOP') + (' edge' if pc['edge'] else '') + (' end' if pc['end'] else '') + \
             (f" block {len(pc['sig'])}x{len(pc['sig'])}" if pc['Hs'] is not None else '') + \
             (f" (walked {pc['moved']:.1e} by the Evaluator)" if pc['moved'] > 1e-6 else '')
        P(f"  {pc['w']:10.6f} {pc['s'] / gref - 1:+12.3e} {gap2:10.4f} {mult:>16s} {pc['gn2']:9.3e} {pc['gn1']:9.3e}  {fl}")
    P(f"  (mult@eps: the number of singular values within eps of gamma, for eps in {[f'{v:g}' for v in eps_all]} where the top is active)")
    it0 = 0
    jtop = np.argsort(-np.abs(pieces[it0]['g']))[:6]
    P(f"  largest |g| coordinates of the top piece: " + ', '.join(f"{names[j]} {pieces[it0]['g'][j]:+.3e}" for j in jtop))
    P(f"constraints (c <= 0; c~ = c / scale; active at eps when c~ >= -eps):")
    for cc in sorted(cons, key=lambda q: q['entry']):
        if cc['entry'] <= eps_max or cc['kind'] != 'pos':
            P(f"  {cc['name']:46s} c {cc['c']:+.3e} (c~ {cc['ct']:+.2e}, entry eps {cc['entry']:.2e}) |h| {cc['hn']:.3e}"
              + (f"  [raw v - L {cc['c_raw']:+.1e}, clipped]" if cc.get('c_raw', 0.0) > 0 else ''))
    nin = sum(1 for cc in cons if cc['kind'] == 'pos' and cc['entry'] > eps_max)
    if nin:
        P(f"  ({nin} position row(s) further than {eps_max:g} inside, not listed)")
    out['pieces'] = [dict(w=pc['w'], s=pc['s'], rel=pc['s'] / gref - 1, s2=pc.get('s2'), s3=pc.get('s3'), sig=pc['sig'],
                          block=(len(pc['sig']) if pc['Hs'] is not None else 1), top=pc['top'], edge=pc['edge'], end=pc['end'],
                          seed=pc['seed'], walked=pc['moved'], gn2=pc['gn2'], gn1=pc['gn1'], g=pc['g']) for pc in pieces]
    out['constraints'] = [dict(name=cc['name'], kind=cc['kind'], c=cc['c'], ct=cc['ct'], entry=cc['entry'], hn=cc['hn'],
                               lam=cc.get('lam'), h=cc['h']) for cc in cons]
    out['gamma_tops'] = gref; out['gamma_rel'] = gref / gam - 1
    out['line'] = dict(clear=bool(e['clr']), alpha=float(e['al']), alpha_record=float(z['alpha']) if 'alpha' in z.files else None,
                       seen_without_row=[dict(lam=l_, entry=en_) for l_, en_ in no_row])
    out['log'] = log
    # ---------------------------------------------------------------- 6. the measure per eps
    t0 = time.time()

    def active_set(ep):
        sc_, bl_ = [], []
        for i, pc in enumerate(pieces):
            if pc['s'] < gref * (1 - ep):
                continue
            k = int(np.sum(pc['sig'] >= gref * (1 - ep))) if pc['Hs'] is not None else 1
            (bl_ if k >= 2 else sc_).append((i, k))
        return sc_, bl_

    def cons_at(ep):
        ac = [j for j, cc in enumerate(cons) if cc['entry'] <= ep]
        hp = []
        if herm is not None and herm['entry'] <= ep:
            ac = [j for j in ac if j not in cl_j[:2]]
            H0 = herm['H0']; lmv, Q = np.linalg.eigh(H0)
            Q = Q[:, lmv <= max(ep, 1e-12) * max(float(np.trace(H0)), 1e-300)]
            if Q.shape[1]:
                hp.append(np.array([_sv(Q.T @ herm['dH'][j] @ Q) for j in range(NZ)]))
        return ac, hp

    def measure(ep, sc_, bl_, ac, hp, Gmod=None):
        gs2 = float(np.mean([pieces[i]['gn2'] for i, _ in sc_ + bl_]))
        gs1 = float(np.mean([pieces[i]['gn1'] for i, _ in sc_ + bl_]))
        gfun = (lambda i: pieces[i]['g']) if Gmod is None else (lambda i: Gmod(i, 'g'))
        hfun = (lambda i, k: pieces[i]['Hs'][:k, :k]) if Gmod is None else (lambda i, k: Gmod(i, 'Hs', k))
        G = np.array([gfun(i) for i, _ in sc_]).reshape(-1, NZ) / gs2
        Bl = [hfun(i, k) / gs2 for i, k in bl_]
        Hn = np.array([cons[j]['h'] / max(cons[j]['hn'], 1e-300) for j in ac]).reshape(-1, NZ)
        hp_n = [h_ / max(float(np.linalg.norm(h_)), 1e-300) for h_ in hp]
        res = {}
        for key, H_, ps_, q in (('r2', Hn, hp_n, 2), ('r2_nc', None, (), 2), ('r1', Hn, hp_n, 1), ('r1_nc', None, (), 1)):
            r = min_norm(NZ, G, Bl, H_, ps_, q=q)
            r['r'] = r['r'] * gs2; r['rho'] = float(np.linalg.norm(r['r'], q))
            r['mu'] = np.array([m_ * gs2 / max(cons[j]['hn'], 1e-300) for m_, j in zip(r['mu'], ac)]) if H_ is not None else np.zeros(0)
            # the multiplier-weighted scales (LESSONS 21: the mean |g_i| read the GL KSP+B+C residual 6.2e-4 as a
            # stationary kink while the min-norm element put no weight on its large row, 1734 against 44): wgt =
            # sum lam_i |g_i| + sum_b sum_a Y_b,aa |Hs_b,aa| (the objective's weights only), and the total mass
            # mass = wgt + sum mu_j |h_j|, so that rho / mass in [0, 1] is the cancellation ratio of the minimiser
            # (1: its terms all aligned, 0: exact cancellation).  Per variable group the same ratio on the group's
            # coordinates: invariant under a uniform rescaling of the group (the packer's sc, sx = 0.30, a gauge
            # representative changes the metric between groups much more than within one)
            def _wsum(ix=None):
                sel = (lambda v: v) if ix is None else (lambda v: v[ix])
                w_ = sum(float(l_) * float(np.linalg.norm(sel(gfun(i)), q)) for (i, _), l_ in zip(sc_, r['lam']))
                w_ += sum(float(np.real(Y[a_, a_])) * float(np.linalg.norm(sel(np.real(hfun(i, k)[a_, a_])), q))
                          for (i, k), Y in zip(bl_, r['Y']) for a_ in range(k))
                m_ = sum(float(mm) * float(np.linalg.norm(sel(cons[j]['h']), q)) for mm, j in zip(r['mu'], ac)) if H_ is not None else 0.0
                return w_, m_
            wg, ms = _wsum()
            r['wgt'] = wg; r['mass'] = wg + ms
            r['by_group_cancel'] = {}
            for k_, ix in groups:
                wgG, msG = _wsum(ix)
                r['by_group_cancel'][k_] = (float(np.linalg.norm(r['r'][ix], q)) / (wgG + msG) if wgG + msG > 0 else None)
            res[key] = r
        # the constraint qualification (MFCQ: the active gradients positively linearly independent): the distance
        # from 0 to the convex hull of the unit active gradients.  0 means a Fritz-John point with lambda_0 = 0 can
        # exist, the KKT residual's multipliers are unbounded and its value means little; ~1 is orthogonal rows
        res['cq'] = float(min_norm(NZ, Hn)['rho']) if len(ac) else None
        # the group-wise NECESSARY condition: at a stationary point the projection of the eps-subdifferential plus
        # the cones on each group's coordinates contains 0 as well, so each group's own min-norm problem (its own
        # weights) must reach 0.  rho_G over that problem's own mass (sum lam |g_iG| + sum Y_aa |Hs_aaG| + sum mu,
        # the rows unit in the group) is invariant under ANY rescaling of the groups against each other and under a
        # uniform one within the group, unlike every joint number above (their minimiser moves with the metric):
        # a large value proves non-stationarity at this eps whatever the packer's scales (an upper bound only on
        # the joint residual: small in every group is not stationarity)
        sep = {}
        for k_, ix in groups:
            Gg = G[:, ix]
            Blg = [B_[:, :, ix] for B_ in Bl]
            hsel = [jj for jj in range(len(Hn)) if float(np.linalg.norm(Hn[jj, ix])) > 1e-12]
            Hg = np.array([Hn[jj, ix] / float(np.linalg.norm(Hn[jj, ix])) for jj in hsel]).reshape(-1, len(ix))
            psg = [h_[ix] for h_ in hp_n if float(np.linalg.norm(h_[ix])) > 1e-12]
            rg = min_norm(len(ix), Gg, Blg, Hg if len(hsel) else None, psg, q=2)
            mass_g = sum(float(l_) * float(np.linalg.norm(Gg[ii])) for ii, l_ in enumerate(rg['lam']))
            mass_g += sum(float(np.real(Y[a_, a_])) * float(np.linalg.norm(np.real(Blg[bi][a_, a_])))
                          for bi, Y in enumerate(rg['Y']) for a_ in range(Y.shape[0]))
            mass_g += float(np.sum(rg['mu'])) if len(hsel) else 0.0
            typ = float(np.mean([np.linalg.norm(Gg[ii]) for ii in range(len(Gg))] +
                                [np.linalg.norm(np.real(B_[0, 0])) for B_ in Blg]))
            sep[k_] = (float(rg['rho']) / mass_g) if mass_g > 1e-9 * max(typ, 1e-300) else 0.0
        res['group_sep'] = sep
        return gs2, gs1, res

    meas = []
    for ep in eps_all:
        sc_, bl_ = active_set(ep)
        ac, hp = cons_at(ep)
        gs2, gs1, res = measure(ep, sc_, bl_, ac, hp)
        r2, r1 = res['r2'], res['r1']
        gmax = max(pieces[i]['gn2'] for i, _ in sc_ + bl_)
        gmin = min(pieces[i]['gn2'] for i, _ in sc_ + bl_)
        ok_min = res['r2_nc']['rho'] <= gmin * (1 + 1e-6)
        wts = [dict(w=pieces[i]['w'], lam=float(l_), gn2=pieces[i]['gn2']) for (i, _), l_ in zip(sc_, r2['lam'])]
        wts += [dict(w=pieces[i]['w'], block=k, trY=float(np.real(np.trace(Y))), eigY=np.linalg.eigvalsh(Y).tolist())
                for (i, k), Y in zip(bl_, r2['Y'])]
        mus = [dict(name=cons[j]['name'], mu=float(m_), mu_h=float(m_) * cons[j]['hn'], entry=cons[j]['entry'])
               for j, m_ in zip(ac, r2['mu'])]
        byg = {k_: float(np.linalg.norm(r2['r'][ix])) / gs2 for k_, ix in groups}
        jr = np.argsort(-np.abs(r2['r']))[:4]
        st_ = sorted(set(res[k_]['status'] for k_ in res if k_ not in ('cq', 'group_sep')))
        meas.append(dict(eps=ep, n_scalar=len(sc_), n_block=len(bl_), n_cons=len(ac), n_psd=len(hp), gscale2=gs2, gscale1=gs1,
                         rho2=r2['rho'], rho2_rel=r2['rho'] / gs2, rho2_nc_rel=res['r2_nc']['rho'] / gs2,
                         rho1=r1['rho'], rho1_rel=r1['rho'] / gs1, rho1_nc_rel=res['r1_nc']['rho'] / gs1,
                         rho2_rel_max=r2['rho'] / gmax, rho2_rel_min=r2['rho'] / gmin,
                         rho2_rel_w=r2['rho'] / max(r2['wgt'], 1e-300), rho1_rel_w=r1['rho'] / max(r1['wgt'], 1e-300),
                         rho2_cancel=r2['rho'] / max(r2['mass'], 1e-300), cq_margin=res['cq'],
                         by_group_cancel=r2['by_group_cancel'], group_sep=res['group_sep'], weights=wts, mu=mus, by_group=byg,
                         r_top=[(names[j], float(r2['r'][j])) for j in jr], status=st_, nc_le_min=bool(ok_min),
                         pieces=[pieces[i]['w'] for i, _ in sc_ + bl_], cons=[cons[j]['name'] for j in ac]))
    tm['measure'] = time.time() - t0
    out['measure'] = meas
    P(f"Clarke residual (min-norm element of the eps-subdifferential + the active normal cones; / mean |g_i|_q):")
    P(f"  {'eps':>7s} {'pieces':>9s} {'cons':>4s} {'rho2/gs':>9s} {'(no cons)':>9s} {'rho1/gs':>9s} {'(no cons)':>9s} "
      f"{'rho2/max|g|':>11s} {'rho2/min|g|':>11s} {'rho2/wgt':>9s} {'rho2/mass':>9s} {'rho2 abs':>9s} {'gscale2':>9s} {'CQ':>8s}  status")
    for m_ in meas:
        P(f"  {m_['eps']:7.0e} {m_['n_scalar']:5d}+{m_['n_block']:<3d} {m_['n_cons'] + m_['n_psd']:4d} {m_['rho2_rel']:9.2e} "
          f"{m_['rho2_nc_rel']:9.2e} {m_['rho1_rel']:9.2e} {m_['rho1_nc_rel']:9.2e} {m_['rho2_rel_max']:11.2e} "
          f"{m_['rho2_rel_min']:11.2e} {m_['rho2_rel_w']:9.2e} {m_['rho2_cancel']:9.2e} {m_['rho2']:9.2e} "
          f"{m_['gscale2']:9.2e} {(_f(m_['cq_margin'], 1) if m_['cq_margin'] is not None else '-'):>8s}  {','.join(m_['status'])}"
          + ('' if m_['nc_le_min'] else '  <-- min_norm > min |g_i|')
          + ('  <-- MFCQ nearly fails: the KKT multipliers are unbounded, read the residual with care'
             if m_['cq_margin'] is not None and m_['cq_margin'] < 1e-6 else ''))
    P(f"  (rho2/min|g| near 1: no cancellation -- the residual is the smallest active piece's gradient, and rho2/gs < 1 "
      f"then only reflects the spread of |g_i|, not progress towards stationarity.  rho2/wgt: over the minimiser's own "
      f"weights, sum lam_i |g_i| (blocks: sum_a Y_aa |Hs_aa|), the normalisation that ignores rows the minimiser does not "
      f"use; rho2/mass: over wgt + sum mu_j |h_j|, the cancellation ratio in [0, 1].  CQ: the distance from 0 to the hull "
      f"of the unit active constraint gradients (MFCQ holds when > 0; 1 = one row or orthogonal rows))")
    # the profile: rho_2 just past each entry eps of the pieces and rows of P(eps_pred) -- where the peaks start to
    # cancel (SISO 598.11: 1.0 up to eps = 1e-2, 3.9e-3 once the top at -1.03e-2 enters, 1.1e-3 with all five
    # tops within 1.4e-2: eps-stationary only at the scale of the ripple between its peaks)
    t1 = time.time()
    ent = sorted(set([pc['entry'] for pc in pieces if pc['entry'] <= eps_p] + [cc['entry'] for cc in cons if cc['entry'] <= eps_p]))
    prof = []
    for e_ in ent:
        ep = e_ * (1 + 1e-9) + 1e-12
        sc_, bl_ = active_set(ep); ac, hp = cons_at(ep)
        gs2, gs1, res = measure(ep, sc_, bl_, ac, hp)
        gmin = min(pieces[i]['gn2'] for i, _ in sc_ + bl_)
        gsw = max(res['group_sep'].items(), key=lambda kv: kv[1]) if res['group_sep'] else (None, None)
        prof.append(dict(eps=ep, n_pieces=len(sc_) + len(bl_), n_cons=len(ac) + len(hp), rho2_rel=res['r2']['rho'] / gs2,
                         rho2_nc_rel=res['r2_nc']['rho'] / gs2, rho2_rel_min=res['r2']['rho'] / gmin,
                         rho2_rel_w=res['r2']['rho'] / max(res['r2']['wgt'], 1e-300), group_sep_worst=gsw))
    out['profile'] = prof
    tm['measure'] += time.time() - t1
    if prof:
        P(f"  profile (rho_2 just past each entry eps of P({eps_p:g})): " + '; '.join(
            f"{q['eps']:.2e}: {q['rho2_rel']:.2e} ({q['n_pieces']}p {q['n_cons']}c, /min|g| {q['rho2_rel_min']:.2g}, "
            f"/wgt {q['rho2_rel_w']:.2g}, worst group {q['group_sep_worst'][0]} {q['group_sep_worst'][1]:.2g})" for q in prof))
        P(f"  (worst group: the group-wise necessary condition, scale-free; 1 = no convex combination of the active pieces "
          f"and cones cancels in that group's coordinates: not eps-stationary at that eps whatever the packer's scales)")
    hs = (J.get('history') or [{}])[-1] if J.get('history') else {}
    if 'stat_value' in hs:
        P(f"  (the record's stat_value {hs['stat_value']:.4g} is ||grad_L|| of the last QP, not this residual)")
    P("decomposition (rho_2, with the constraints):")
    for m_ in meas:
        P(f"  eps {m_['eps']:g}: weights " + ', '.join(
            (f"{q['w']:.4f}:{q['lam']:.3g}" if 'lam' in q else f"{q['w']:.4f}:block{q['block']} trY {q['trY']:.3g} eigY {np.round(q['eigY'], 4).tolist()}")
            for q in m_['weights'] if q.get('lam', q.get('trY', 0.0)) > 1e-4))
        if m_['mu']:
            mt_ = '; '.join(f"{q['name']}: mu {q['mu']:.3g} (mu|h| {q['mu_h'] / m_['gscale2']:.2e} gscale)" for q in m_['mu']
                            if q['mu_h'] > BIND_MN * m_['gscale2'])
            P(f"    multipliers of the {len(m_['mu'])} active row(s): " + (mt_ or f"all mu|h| below {BIND_MN:g} gscale (the cones remove nothing)"))
        P(f"    residual by group (/ gscale2): " + ', '.join(f"{k_} {v_:.2e}" for k_, v_ in m_['by_group'].items() if v_ > 1e-12)
          + f"; largest coordinates " + ', '.join(f"{n_} {v_:+.2e}" for n_, v_ in m_['r_top']))
        P(f"    cancellation by group at the joint minimiser (|r_G| / (sum lam |g_iG| + sum mu |h_jG|), 1 = none): "
          + ', '.join(f"{k_} {v_:.2e}" for k_, v_ in m_['by_group_cancel'].items() if v_ is not None))
        P(f"    group-wise necessary condition (each group's own min-norm over its own mass; scale-free, 0 at a "
          f"stationary point): " + ', '.join(f"{k_} {v_:.2e}" for k_, v_ in m_['group_sep'].items()))
    # ---------------------------------------------------------------- 7. predicted decrease
    t0 = time.time()
    sc_m, bl_m = active_set(eps_p)
    a_s = np.array([pieces[i]['s'] - gref for i, _ in sc_m]); G_s = np.array([pieces[i]['g'] for i, _ in sc_m]).reshape(-1, NZ)
    B_s = [(pieces[i]['sig'][:k] - gref, pieces[i]['Hs'][:k, :k]) for i, k in bl_m]
    cl_idx = cl_j[:2] if herm is not None else []
    C_all = [(cc['c'], cc['h']) for j, cc in enumerate(cons) if j not in cl_idx]
    C_names = [cc['name'] for j, cc in enumerate(cons) if j not in cl_idx]
    C_hn = [cc['hn'] for j, cc in enumerate(cons) if j not in cl_idx]
    Ps = [(herm['H0'], herm['dH'])] if herm is not None else []
    gmax1 = max(pieces[i]['gn1'] for i, _ in sc_m + bl_m)
    gs2m = float(np.mean([pieces[i]['gn2'] for i, _ in sc_m + bl_m]))
    D_set = gref * eps_p / gmax1
    lad = sorted(set(deltas + ([trust] if trust else [])))

    def pred_box(Dl, with_cons=True, norm='inf'):
        return pred(NZ, gref, a_s, G_s, B_s, C_all if with_cons else (), Ps if with_cons else (), Delta=Dl, norm=norm)
    # the descent test's LP: the same box LP with its closed-loop rows on the QP's line -eps_s - 2 mg (the run's
    # LAT_CUSHION), so that a step binding a row lands inside the admissibility line -eps_s - mg the certificate asks.
    # On the problem line (the pred table) such a step sits on -eps_s to first order: inadmissible by mg (decentral
    # 369.87, the row -0.050104 + 2.285i binds from Delta = 1e-4).  A row already between the two lines keeps h.d <= 0
    sh_cl = 2 * mg if a.cl_line == 'problem' else 0.0
    C_adm = [(min(cc['c'] + sh_cl, max(cc['c'], 0.0)) if cc['kind'] == 'cl' else cc['c'], cc['h'])
             for j, cc in enumerate(cons) if j not in cl_idx]

    def pred_desc(Dl):
        return pred(NZ, gref, a_s, G_s, B_s, C_adm, Ps, Delta=Dl, norm='inf')
    prs = []
    for Dl in lad:
        pb = pred_box(Dl); p0 = pred_box(Dl, with_cons=False); p2 = pred_box(Dl, norm='2')
        bind_p = [round(pieces[i]['w'], 4) for (i, _), l_ in zip(sc_m, pb['lam']) if l_ > 1e-6] + \
                 [f"block {round(pieces[i]['w'], 4)} trY {y_:.3f}" for (i, _), y_ in zip(bl_m, pb['Ytr']) if y_ > 1e-6]
        bind_c = [C_names[j] for j, m_ in enumerate(pb['mu']) if m_ * C_hn[j] > BIND * gs2m]
        dg = {k_: float(pieces[0]['g'][ix] @ pb['d'][ix]) / gref for k_, ix in groups}   # the top piece's change by group
        prs.append(dict(Delta=Dl, at_trust=bool(trust and Dl == trust), pred=pb['pred'], pred_nc=p0['pred'], pred_ball=p2['pred'],
                        bind_pieces=bind_p, bind_cons=bind_c, d_by_group=dg, d=pb['d'], gap=max(pb['gap'], p0['gap'], p2['gap']),
                        status=sorted(set([pb['status'], p0['status'], p2['status']]))))
    tm['pred'] = time.time() - t0
    out['pred'] = prs; out['Delta_set'] = D_set
    P(f"predicted first-order decrease (the SQP's box LP at H = 0 over the {len(sc_m) + len(bl_m)} pieces of P({eps_p:g}) "
      f"and all {len(C_all) + len(Ps)} rows; / gamma):  Delta_set = gamma eps_pred / max|g|_1 = {D_set:.2e}")
    P(f"  {'Delta':>9s} {'pred_inf':>9s} {'(no cons)':>9s} {'pred_2':>9s}  binding (pieces; rows)  the top piece's g.d*/gamma by group")
    for q in prs:
        P(f"  {q['Delta']:9.2e} {q['pred']:9.2e} {q['pred_nc']:9.2e} {q['pred_ball']:9.2e}  {q['bind_pieces'][:8]}"
          f"{' ...' if len(q['bind_pieces']) > 8 else ''}; {q['bind_cons'][:4]}{' ...' if len(q['bind_cons']) > 4 else ''}  "
          + ' '.join(f"{k_} {v_:+.1e}" for k_, v_ in sorted(q['d_by_group'].items(), key=lambda kv: kv[1])[:4])
          + ('  <- the record\'s sqp_trust' if q['at_trust'] else '') + (f"  [gap {q['gap']:.1e}]" if q['gap'] > 1e-6 else '')
          + ('' if q['status'] == ['Solved'] else f"  [{','.join(q['status'])}]"))
    # ---------------------------------------------------------------- 8. the real-data stationary point by construction
    t0 = time.time()
    sh = []
    for ep in sorted(set([max(eps_list), eps_max])):
        sc_, bl_ = active_set(ep); ac, hp = cons_at(ep)
        gs2, gs1, res = measure(ep, sc_, bl_, ac, hp)
        rstar = res['r2']['r']
        def Gmod(i, what, k=None, v=rstar):
            if what == 'g':
                return pieces[i]['g'] - v
            Hs = pieces[i]['Hs'][:k, :k].copy()
            for a_ in range(k):
                Hs[a_, a_] = Hs[a_, a_] - v
            return Hs
        _, _, res0 = measure(ep, sc_, bl_, ac, hp, Gmod=Gmod)
        Gt = np.array([pieces[i]['g'] - rstar for i, _ in sc_]).reshape(-1, NZ)
        Bt = [(np.zeros(k), Gmod(i, 'Hs', k)) for i, k in bl_]
        pt = pred(NZ, gref, np.zeros(len(sc_)), Gt, Bt, [(0.0, cons[j]['h']) for j in ac], [], Delta=1e-4) if not hp else None
        u_ = rstar / max(np.linalg.norm(rstar), 1e-300)
        grow = []
        for tt in (1e-3, 1e-2, 1e-1, 1.0):
            v2 = rstar - tt * gs2 * u_
            _, _, rr = measure(ep, sc_, bl_, ac, hp, Gmod=lambda i, what, k=None, v=v2: Gmod(i, what, k, v))
            grow.append(rr['r2']['rho'] / gs2)
        mono = all(grow[k + 1] >= grow[k] * (1 - 1e-6) - 1e-12 for k in range(len(grow) - 1)) and all(g_ <= tt * (1 + 1e-6) + 1e-12
                                                                                                      for g_, tt in zip(grow, (1e-3, 1e-2, 1e-1, 1.0)))
        ok = res0['r2']['rho'] / gs2 <= 1e-9 and (pt is None or pt['pred'] <= 1e-9) and mono
        sh.append(dict(eps=ep, rho_shift=res0['r2']['rho'] / gs2, pred_shift=(pt['pred'] if pt else None), growth=grow,
                       monotone=mono, ok=ok))
    tm['shifted'] = time.time() - t0
    out['shifted_check'] = sh
    # ---------------------------------------------------------------- 9. FD checks
    fd = []; fdc = []
    if a.fd > 0:
        t0 = time.time()
        rng = np.random.default_rng(12345)
        g0 = pieces[0]['g']
        dirs = []
        for j in np.argsort(-np.abs(g0))[:3]:
            u_ = np.zeros(NZ); u_[j] = 1.0; dirs.append((f"coordinate {names[j]}", u_))
        v_ = rng.standard_normal(NZ); dirs.append(("random unit", v_ / np.linalg.norm(v_)))
        used = {names[int(np.argmax(np.abs(u_)))] for nm, u_ in dirs if nm.startswith('coordinate')}
        for k_, ix in sorted(groups, key=lambda q: (q[0] not in ('xa', 'xs'), q[0])):
            j = int(ix[np.argmax(np.abs(g0[ix]))])
            if names[j] in used:
                continue
            u_ = np.zeros(NZ); u_[j] = 1.0; dirs.append((f"coordinate {names[j]} (group {k_})", u_)); used.add(names[j])
        v_ = rng.standard_normal(NZ); dirs.append(("random unit 2", v_ / np.linalg.norm(v_)))
        dirs = dirs[:a.fd]
        wl = [pc['w'] for pc in pieces]
        h = FD_H
        for nm, u_ in dirs:
            sp1 = lat.sigmas(z0 + h * u_, wl); sm1 = lat.sigmas(z0 - h * u_, wl)
            sp2 = lat.sigmas(z0 + 0.1 * h * u_, wl); sm2 = lat.sigmas(z0 - 0.1 * h * u_, wl)
            for k, pc in enumerate(pieces):
                Dh = (sp1[k] - sm1[k]) / (2 * h); Dh10 = (sp2[k] - sm2[k]) / (0.2 * h)
                rich = (100 * Dh10 - Dh) / 99; an = float(pc['g'] @ u_)
                den = max(abs(an), 1e-3 * pc['gn2'] * float(np.linalg.norm(u_)))
                row = dict(dir=nm, w=pc['w'], analytic=an, fd_h=Dh, fd_h10=Dh10, richardson=rich, rel=abs(rich - an) / den,
                           rel_central=abs(Dh10 - an) / den, ok=abs(rich - an) / den <= GTOL_FD)
                if pc['Hs'] is not None:
                    k2 = len(pc['sig'])
                    Hu = np.einsum('abj,j->ab', pc['Hs'], u_)
                    lin = float(np.linalg.eigvalsh(np.diag(pc['sig'] - pc['s']) + 0.1 * h * Hu).max()) / (0.1 * h)
                    row.update(block_onesided=(sp2[k] - pc['s']) / (0.1 * h), block_lin=lin, branch=an)
                fd.append(row)
        # the constraint rows (exact, cheap): modal and position rows
        hh = 1e-6
        for nm, u_ in dirs:
            for kind in ('modal', 'pos'):
                idx = [j for j, cc in enumerate(cons) if cc['kind'] == kind]
                if not idx:
                    continue
                if kind == 'modal':
                    vals = lambda zz: modal_frozen(ctl, pk, zz, z0, Lc)
                else:
                    vals = lambda zz: np.array([c_ for c_, _ in ev._pos_rows(zz)])
                vp, vm = vals(z0 + hh * u_), vals(z0 - hh * u_)
                for q_, j in enumerate(idx):
                    an = float(cons[j]['h'] @ u_); fdv = (vp[q_] - vm[q_]) / (2 * hh)
                    den = max(abs(an), 1e-3 * cons[j]['hn'], 1e-12)
                    fdc.append(dict(dir=nm, name=cons[j]['name'], analytic=an, fd=fdv, rel=abs(fdv - an) / den,
                                    ok=abs(fdv - an) / den <= 1e-5 or abs(fdv - an) <= 1e-9))
        if a.fd_cl == 'on':
            cl_act = sorted([cc for cc in cons if cc['kind'] == 'cl'], key=lambda q: q['entry'])[:2]
            for cc in cl_act:
                for nm, u_ in dirs[:2] + ([("row's steepest coordinate", np.eye(NZ)[int(np.argmax(np.abs(cc['h'])))])]):
                    lp = cl_eig_near(lat, z0 + hh * u_, cc['lam']); lm_ = cl_eig_near(lat, z0 - hh * u_, cc['lam'])
                    an = float(cc['h'] @ u_)
                    if lp is None or lm_ is None:
                        fdc.append(dict(dir=nm, name=cc['name'], analytic=an, fd=None, rel=None, ok=False)); continue
                    fdv = (lp.real - lm_.real) / (2 * hh); den = max(abs(an), 1e-3 * cc['hn'], 1e-12)
                    fdc.append(dict(dir=nm, name=cc['name'], analytic=an, fd=fdv, rel=abs(fdv - an) / den, ok=abs(fdv - an) / den <= 1e-4))
        tm['fd'] = time.time() - t0
    out['fd'] = fd; out['fd_cons'] = fdc
    # ---------------------------------------------------------------- 10. descent test
    dsc = []; Dv = None; cert = None; zsteps = {}

    def band_at(zz):
        """The band's upper end the run would use at zz (run_case_epi band_from_controller, BFS: it only grows,
        to cover every compensator pole: max(whi, max|Im lambda(A_c)| + wpad))."""
        im_ = np.abs(np.linalg.eigvals(ctl.A_of(*pk.unpack(zz)[0])).imag)
        return max(whi, (float(im_.max()) + a.wpad) if im_.size else 0.0)
    wb0 = band_at(z0)
    out['band_rule_at_record'] = wb0
    if wb0 > whi * (1 + 1e-9):
        P(f"  WARNING: the record's band [0, {whi:.6f}] is narrower than the run's rule at its own design "
          f"(max|Im lambda(A_c)| + wpad = {wb0:.6f}): its gamma, the tops and this measure are over too narrow a band")

    def ctrl_check(zz):
        """The compensator's abscissa and the modal rows' max v - L at zz (exact, r x r)."""
        ar_ = pk.unpack(zz)[0]
        al_ = float(np.max(np.linalg.eigvals(ctl.A_of(*ar_)).real))
        mv_ = max((v_ - Lc for v_, _ in ctl.modal_rows(pk.modal, *[np.asarray(x_) for x_ in ar_[:3]], Lc)), default=np.nan) \
            if getattr(pk, 'modal', None) else np.nan
        return al_, mv_

    def modal_h(zz):
        """The modal rows at zz as h <= 0 and their packed gradients (sqp_inner _modal_h, the Evaluator's rows)."""
        ar_ = pk.unpack(zz)[0]; rr_ = len(ar_[0])
        H_, G_ = [], []
        for v_, gd_ in ctl.modal_rows(pk.modal, ar_[0], ar_[1], ar_[2], Lc):
            gc = pk.zero_grad() if (getattr(pk, 'mimo', False) or getattr(pk, 'vec_pos', False)) \
                else dict(n=np.zeros(rr_, complex), K=0j, xa=0.0, xs=0.0)
            gd = np.zeros(rr_, complex); ge = np.zeros(max(rr_ - 1, 0), complex); gf = ge.copy()
            for i_, x_ in gd_['d'].items():
                gd[i_] = x_
            for i_, x_ in gd_['e'].items():
                ge[i_] = x_
            for i_, x_ in gd_['f'].items():
                gf[i_] = x_
            gc.update(d=gd, e=ge, f=gf)
            H_.append(v_ - Lc); G_.append(np.real(np.asarray(pk.grad(gc), float)))
        return np.array(H_), np.array(G_)

    def modal_restore(zn_, zbase):
        """sqp_inner _ctrl_restore_modal: Newton's least-norm steps on the violated modal rows (a trace is linear in
        z, a determinant quadratic) to 1e-7 inside the line; None when that is not a small correction of the step
        (more than half of it) or the abscissa stays over the line."""
        zc_ = np.asarray(zn_, float).copy()
        for _ in range(4):
            H_, G_ = modal_h(zc_)
            bad = np.flatnonzero(H_ > -1e-9)
            if not len(bad):
                break
            Gb = G_[bad]
            try:
                zc_ = zc_ + Gb.T @ np.linalg.solve(Gb @ Gb.T, -1e-7 - H_[bad])
            except np.linalg.LinAlgError:
                return None
        if not np.all(np.isfinite(zc_)) or np.linalg.norm(zc_ - zn_) > 0.5 * max(np.linalg.norm(zn_ - zbase), 1e-12) \
                or ctrl_check(zc_)[0] > Lc + 1e-8:
            return None
        return zc_
    if a.descent != 'off':
        t0 = time.time()
        ts = [float(v) for v in a.descent_ts.split(',')]
        tmax = max(ts)
        wl = [pieces[i]['w'] for i, _ in sc_m + bl_m]
        for Dl in [float(v) for v in a.descent_deltas.split(',')]:
            pb = pred_desc(Dl); d = pb['d']
            if pb['pred'] <= 1e-10:
                dsc.append(dict(Delta=Dl, t=None, pred=pb['pred'], note='no first-order decrease predicted')); break
            fail = False
            for t in ts:
                zt = z0 + t * d
                tt0 = time.time()
                # the compensator first (exact, r x r): an active v2 (det) row is quadratic in z, so a step that binds it
                # at first order crosses the line at second order (decentral 412.28, Delta = 1e-4: the pole at -0.05 to
                # -0.04999999, 1.2e-8 over the 1e-8 tolerance, and the step that measured -8.2e-4 was thrown out).  The
                # SQP puts such a trial back onto the line (sqp_inner _ctrl_restore_modal) before it evaluates the
                # plant; the test does the same, so it judges the step the SQP would take
                alc, mv = ctrl_check(zt)
                soc = 0.0
                if getattr(pk, 'modal', None) and (alc > Lc + 1e-8 or (np.isfinite(mv) and mv > 1e-8)):
                    zc = modal_restore(zt, z0)
                    if zc is not None:
                        soc = float(np.max(np.abs(zc - zt))); zt = zc; alc, mv = ctrl_check(zt)
                    else:
                        soc = np.nan
                zsteps[(Dl, t)] = zt
                tt = clip_band(lat, zt, lat.peak_tops(zt, wl, RAD), whi)
                fmax = max(v_ for _, v_ in tt)
                mod = max([pieces[i]['s'] + t * float(pieces[i]['g'] @ d) for i, _ in sc_m] +
                          [float(np.linalg.eigvalsh(np.diag(pieces[i]['sig'][:k]) + t * np.einsum('abj,j->ab', pieces[i]['Hs'][:k, :k], d)).max())
                           for i, k in bl_m])
                u_ = pk.unpack(zt)
                inside = all(D.inside(x_, 'a') for x_ in np.asarray(u_[3], float).reshape(-1, 2)) and \
                    all(D.inside(x_, 's') for x_ in np.asarray(u_[4], float).reshape(-1, 1))
                # the closed loop at the step: every row the first-order bound (twice it, for curvature) lets reach
                # the admissibility line -eps_s - mg, tracked by lattice._nearest from its predicted position
                # (Re lambda + t h.d).  Before, a step was "good" on the compensator, modal and position checks
                # alone, and the verdict quoted the decrease of a step the LP had put on the line.  Measured, not a
                # certificate (a hidden eigenvalue is the line's business: --descent certify)
                dinf_t = float(np.max(np.abs(t * d)))
                trk = [cc for cc in cons if cc['kind'] == 'cl' and np.isfinite(cc['lam'].imag)
                       and cc['c'] + 2 * dinf_t * float(np.sum(np.abs(cc['h']))) >= -mg]
                clv = []
                for cc in trk:
                    lt = cl_eig_near(lat, zt, complex(cc['lam'].real + t * float(cc['h'] @ d), cc['lam'].imag))
                    clv.append(float(lt.real) if lt is not None else np.nan)
                clmax = max(clv) if clv else -np.inf
                cl_ok = bool(all(np.isfinite(v_) for v_ in clv) and clmax <= -eps_s - mg)
                pr_ = (mod - gref) / gref; ac_ = (fmax - gref) / gref
                ratio = ac_ / pr_ if pr_ < 0 else np.nan
                kmax = int(np.argmax([v_ for _, v_ in tt]))
                ip_max = (sc_m + bl_m)[kmax][0]
                row = dict(Delta=Dl, t=t, dinf=float(np.max(np.abs(t * d))), pred=pr_, actual=ac_, ratio=ratio, alc=alc, modal_max=mv,
                           soc=soc, inside=bool(inside), cl_max=clmax, cl_tracked=len(trk), cl_ok=cl_ok,
                           tops=[(float(w_), float(v_ / gref)) for w_, v_ in tt], seconds=time.time() - tt0,
                           max_at=float(tt[kmax][0]), max_from=float(pieces[ip_max]['w']), max_from_rel=float(pieces[ip_max]['s'] / gref - 1))
                dsc.append(row)
                P(f"  descent Delta {Dl:.0e} t {t:g}: |t d*|_inf {row['dinf']:.1e}: predicted {pr_:+.3e}, actual {ac_:+.3e} "
                  f"(ratio {ratio:.2f}; max at {tt[kmax][0]:.5f}, the top of {pieces[ip_max]['w']:.5f} at "
                  f"{row['max_from_rel']:+.2e}); compensator abscissa {alc:+.7f}, modal max v - L {mv:+.1e}"
                  + ('' if soc == 0.0 else (f" (after the SQP's second-order correction, |dz|_inf {soc:.1e})" if np.isfinite(soc)
                                            else ' (the second-order correction FAILED)'))
                  + f", positions inside {inside}, "
                  f"closed loop max Re {clmax:+.7f} over {len(trk)} tracked row(s) ({'admissible' if cl_ok else 'NOT <= %.7f' % (-eps_s - mg)}) "
                  f"[{row['seconds']:.0f} s]")
                if t == tmax and not (ratio >= 0.5):
                    fail = True
            if fail:
                break
        good = [q for q in dsc if q.get('t') == tmax and q.get('ratio', 0) >= 0.5 and q.get('alc', 0) <= Lc + 1e-8
                and (not np.isfinite(q.get('modal_max', np.nan)) or q['modal_max'] <= 1e-8) and q.get('inside') and q.get('cl_ok')]
        if good:
            Dv = max(good, key=lambda q: q['Delta'])
        tm['descent'] = time.time() - t0
        if a.descent == 'certify' and good:
            # the good steps, most measured decrease first (at most 3).  A region the level set finds is a top the
            # re-located pieces did not see: it is located (peak_tops at its estimate), the step's max raised to it,
            # and the level set asked again (at most 3 times); the step is certified when the level set is clear at
            # the max of every known top and that max is below gamma.  Uncertified, the next step is tried.
            t0 = time.time()
            cert_tries = []
            for best in sorted(good, key=lambda q: q['actual'])[:3]:
                zt = zsteps[(best['Delta'], best['t'])]        # the step as the descent test measured it (corrected)
                # the band the run would certify z + d on: it only grows, to max|Im lambda(A_c)| + wpad
                # (run_case_epi band_from_controller, BFS; central 123.90's whi 24.9237 IS its pole at 21.924i + 3):
                # the step's own band, not the record's, or a step that moves that pole up is judged on too narrow a band
                whi_t = band_at(zt)
                ln = lat.line(zt, eps_s, 0.0, whi_t + a.wpad)
                mt = gref * (1 + best['actual'])
                lv = None; found = []
                ok_line = bool(ln['clear']) and float(ln['alpha']) <= -eps_s - mg
                if whi_t > whi * (1 + 1e-12):
                    # (a peak in the new part of the band is the level set's to find: it covers [0, whi_t])
                    P(f"    the step widens the band: whi {whi:.6f} -> {whi_t:.6f} (compensator |Im| + wpad)")
                if ok_line:
                    for rep in range(3):
                        lv = lat.level(zt, mt * (1 + CERT_RTOL), 0.0, whi_t)
                        if lv['outcome'] != 'region' or not lv['cross']:
                            break
                        new = clip_band(lat, zt, lat.peak_tops(zt, [float(w_) for w_, _ in lv['cross']], RAD), whi_t)
                        found += [(float(w_), float(s_)) for w_, s_ in new]
                        m2 = max([mt] + [float(s_) for _, s_ in new])
                        P(f"    level set at {mt * (1 + CERT_RTOL):.6f} (Delta {best['Delta']:.0e}): region(s) "
                          f"{[(round(float(w_), 5), round(float(v_), 6)) for w_, v_ in lv['cross']]} -> tops "
                          f"{[(round(w_, 5), round(s_, 6)) for w_, s_ in new]}; the step's max {mt:.6f} -> {m2:.6f}")
                        if m2 <= mt * (1 + CERT_RTOL):
                            break                          # the region's top is not above: left as found (not certified)
                        mt = m2
                ok = (ok_line and lv is not None and lv['outcome'] == 'clear' and mt * (1 + CERT_RTOL) <= gam * (1 - CERT_RTOL))
                c_ = dict(Delta=best['Delta'], t=best['t'], max_top=mt, actual=mt / gref - 1, pred=best['pred'],
                          measured=best['actual'], line_clear=bool(ln['clear']), alpha=float(ln['alpha']),
                          level=(lv['outcome'] if lv else None), uncovered=(float(lv['uncovered']) if lv else None),
                          found=found, certified=bool(ok), gamma_bound=mt * (1 + CERT_RTOL), whi_t=whi_t)
                cert_tries.append(c_)
                P(f"  certificate at Delta {best['Delta']:.0e}: line clear {ln['clear']} alpha {ln['alpha']:+.7f} "
                  f"(needs <= {-eps_s - mg:+.7f}, Im in [0, {whi_t + a.wpad:.4f}]); level set on [0, {whi_t:.4f}] at "
                  f"{mt * (1 + CERT_RTOL):.6f}: {lv['outcome'] if lv else 'not run'} -> "
                  + (f"CERTIFIED: gamma(z + d) <= {mt * (1 + CERT_RTOL):.6f} < {gam:.6f} ({mt / gref - 1:+.3e} relative)" if ok else
                     'not certified'))
                if ok:
                    break
            cert = next((c_ for c_ in cert_tries if c_['certified']), cert_tries[-1] if cert_tries else None)
            if cert is not None:
                cert['tries'] = [dict((k_, v_) for k_, v_ in c_.items() if k_ != 'tries') for c_ in cert_tries]
            if cert is not None and cert['certified']:
                # the certified design itself (the review of 2026-10-17: it was in no file): packed and unpacked in the
                # JSON, and as an npz with the record's design keys beside it (--json OUT.json -> OUT_certified.npz)
                try:
                    zc_ = zsteps[(cert['Delta'], cert['t'])]
                    u_ = pk.unpack(zc_)
                    des = dict(d=np.asarray(u_[0][0]), e=np.asarray(u_[0][1]), f=np.asarray(u_[0][2]), nvec=np.asarray(u_[1]),
                               K=np.asarray(u_[2]), xa=np.asarray(u_[3], float), xs=np.asarray(u_[4], float))
                    if len(u_[0]) > 3:
                        des['Bc'] = np.asarray(u_[0][3])
                    for k_ in ('m', 'p', 'struct', 'loop_orders'):
                        if k_ in z.files:
                            des[k_] = z[k_]
                    cert['design'] = dict(z=np.asarray(zc_, float), **{k_: v_ for k_, v_ in des.items() if k_ not in ('struct',)})
                    if a.json:
                        fn_ = os.path.splitext(a.json)[0] + '_certified.npz'
                        np.savez(fn_, **des, gamma_bound=cert['gamma_bound'], alpha=cert['alpha'], whi=cert['whi_t'],
                                 parent=os.path.abspath(a.record), parent_gamma=gam, Delta=cert['Delta'])
                        cert['design_file'] = fn_
                        P(f"  the certified design: {fn_}")
                except Exception as _e:
                    P(f"  (the certified design was not saved: {type(_e).__name__}: {_e})")
            # the level set's tops correct the measured decrease (a lower bound of the step's max): Delta_v again
            corr = {c_['Delta']: c_['actual'] for c_ in cert_tries}
            for q in good:
                if q['Delta'] in corr:
                    q['actual_level'] = corr[q['Delta']]
            good = [q for q in good if q.get('actual_level', q['actual']) <= 0.5 * q['pred']]
            Dv = max(good, key=lambda q: q['Delta']) if good else None
            tm['certify'] = time.time() - t0
    av_of = lambda q: q.get('actual_level', q['actual'])    # the measured change, with the level set's tops when known
    out['descent'] = dict(rows=dsc, Delta_v=(Dv['Delta'] if Dv else None), pred_at_Dv=(-Dv['pred'] if Dv else None),
                          actual_at_Dv=(-av_of(Dv) if Dv else None), certificate=cert)
    # ---------------------------------------------------------------- 11. checks and verdict
    P("checks:")
    P(f"  convex core toys {nok}/{ntot}; packer round trip {rt_err:.1e}; gamma agreement {gref / gam - 1:+.1e}; "
      f"line clear {e['clr']}, alpha {e['al']:+.7f} vs record {float(z['alpha']) if 'alpha' in z.files else np.nan:+.7f}")
    P(f"  tops: {sum(pc['top'] for pc in pieces)}/{len(pieces)} converged (|sigma'| <= 1e-7 sigma or Newton's distance <= 1e-4)"
      + (f"; NOT tops: {[round(pc['w'], 5) for pc in pieces if not pc['top']]}" if not all(pc['top'] for pc in pieces) else ''))
    P(f"  completeness: " + ('; '.join(f"eps {c_['eps']:g}: {c_['outcome']} (uncovered {c_['uncovered']:.1e})" for c_ in comp)
                             if comp else 'not run (--complete off): the piece set is the seeds\' (rho an upper bound, pred optimistic)'))
    P(f"  min_norm without constraints <= min |g_i| at every eps: {all(m_['nc_le_min'] for m_ in meas)}; "
      f"pred primal = dual: {all(q['gap'] <= 1e-6 for q in prs)} (max gap {max(q['gap'] for q in prs):.1e})")
    for q in sh:
        P(f"  stationary by construction at eps {q['eps']:g} (every g_i - r*): rho~/gs {q['rho_shift']:.1e}, pred~/gamma {_f(q['pred_shift'], 1)}; "
          f"rho~ along t r*/|r*| (t = 1e-3 .. 1) {np.round(q['growth'], 6).tolist()}, monotone {q['monotone']} -> {'ok' if q['ok'] else 'FAILED'}")
    if fd:
        nbad = sum(not r_['ok'] for r_ in fd)
        P(f"  FD of the pieces' sigma_1 at fixed w (Richardson h = {FD_H:g}, h/10; pass {GTOL_FD:g}): {len(fd) - nbad}/{len(fd)} pass, "
          f"max rel {max(r_['rel'] for r_ in fd):.1e} (central h/10 alone: max {max(r_['rel_central'] for r_ in fd):.1e})")
        for r_ in fd:
            if not r_['ok'] or r_['w'] == pieces[0]['w']:
                P(f"    {r_['dir']:34s} w {r_['w']:.6f}: analytic {r_['analytic']:+.8e} FD {r_['richardson']:+.8e} rel {r_['rel']:.1e}"
                  + (f"  block: one-sided {r_['block_onesided']:+.6e}, block linearisation {r_['block_lin']:+.6e}, branch {r_['branch']:+.6e}"
                     if 'block_lin' in r_ else '') + ('' if r_['ok'] else '  <-- FAIL'))
        for r_ in fd:
            if 'block_lin' in r_ and r_['w'] != pieces[0]['w']:
                P(f"    {r_['dir']:34s} w {r_['w']:.6f} (block): one-sided {r_['block_onesided']:+.6e}, block linearisation "
                  f"{r_['block_lin']:+.6e}, single branch {r_['branch']:+.6e}")
    if fdc:
        nbad = sum(not r_['ok'] for r_ in fdc)
        P(f"  FD of the constraint rows (central h = 1e-6): {len(fdc) - nbad}/{len(fdc)} pass")
        for r_ in fdc:
            if not r_['ok'] or 'closed loop' in r_['name']:
                P(f"    {r_['dir']:34s} {r_['name']:40s}: analytic {r_['analytic']:+.6e} FD "
                  f"{(r_['fd'] if r_['fd'] is not None else np.nan):+.6e} rel {(r_['rel'] if r_['rel'] is not None else np.nan):.1e}"
                  + ('' if r_['ok'] else '  <-- FAIL'))
    # the verdict
    ml = [m_ for m_ in meas if m_['eps'] in eps_list]
    # the pick and the head number: rho_2 over the minimiser's own weights sum lam_i |g_i| (LESSONS 21), not over the
    # mean |g_i| -- the mean is not monotone in eps (its piece set changes) and reads a spread of |g_i| as progress:
    # SISO 598.11 at eps = 1e-2 is 0.19 over the mean and 0.98 over the weights (no cancellation at all)
    rmin = min(m_['rho2_rel_w'] for m_ in ml)
    pick = min([m_ for m_ in ml if m_['rho2_rel_w'] <= 2 * rmin], key=lambda q: q['eps'])
    clk = [m_ for m_ in meas if m_['eps'] == CERT_RTOL][0]
    # the class (design.md 4.2): pred(Delta_v)/gamma -- measured where the first-order model holds -- against the
    # certificate's rtol and the progress rule's per-round scale tol / k (--progress; 8:0.0015 since 20261010r: 1.9e-4.
    # The constant was 5e-4, the 3-round rule's, so a 3.4e-4 decrease read "near eps-stationary" where the current
    # rule would go on)
    pk_, pt_ = a.progress.split(':')[:2]
    prog = float(pt_) / max(int(pk_), 1)
    prog_txt = f"~{prog:.1e} per round ({a.progress})"
    if cert and cert['certified']:
        # the size of the certified decrease belongs in the head: a lower admissible design 1.4e-4 below (central
        # 123.90, 13 tops within 4e-4: an equiripple plateau) and one 8e-3 below (central 153.62) are not the same reading
        dec = 1.0 - cert['gamma_bound'] / gam               # the certified bound against the record's gamma
        # "admissible" as the SQP's line search and the gate's line test have it (the level set on [0, whi], the
        # stability line on [0, whi + wpad], the compensator and the positions); the run's acceptance also counts
        # windings over ALL w (the tail bound), which this tool does not run
        head = (f"NOT STATIONARY (certified: a design {dec:.1e} gamma lower at Delta {cert['Delta']:.0e}, admissible on the "
                f"line test Im in [0, {cert.get('whi_t', whi) + a.wpad:g}] -- the winding count over all w not run"
                + (f"; below the progress rule's {prog_txt}: near eps-stationary at that scale" if dec <= prog else
                   f"; above the progress rule's {prog_txt}")
                + "): ")
    elif Dv:
        av = -av_of(Dv)
        head = ('stationary at the certificate\'s resolution (measured): ' if av <= CERT_RTOL else
                (f'near eps-stationary (measured decrease below the progress rule\'s {prog_txt}): ' if av <= prog
                 else f'NOT STATIONARY (measured, uncertified: a first-order decrease above the progress rule\'s {prog_txt}): '))
    else:
        head = ''
    bgc = dict(pick.get('group_sep') or {})
    kw_ = max(bgc, key=bgc.get) if bgc else None
    v = (f"{head}stationary to rho_2/(sum lam_i |g_i|) = {pick['rho2_rel_w']:.1e} at eps = {pick['eps']:.0e} "
         f"({pick['n_scalar'] + pick['n_block']} piece(s){', %d block(s)' % pick['n_block'] if pick['n_block'] else ''}, "
         f"{pick['n_cons'] + pick['n_psd']} active constraint(s); rho_2/gscale (mean |g_i|) {pick['rho2_rel']:.1e}; "
         f"rho_2/min|g_i| {pick['rho2_rel_min']:.2g}{' = no cancellation' if pick['rho2_rel_min'] >= 0.9 else ''}; "
         f"rho_1/(sum lam_i |g_i|_1) {pick['rho1_rel_w']:.1e}"
         + (f"; worst group-wise necessary condition {kw_} {bgc[kw_]:.2g} (scale-free)" if kw_ else '')
         + (f"; MFCQ margin {pick['cq_margin']:.1e}" + (' (NEARLY FAILS)' if pick['cq_margin'] < 1e-6 else '')
            if pick.get('cq_margin') is not None else '') + "); "
         f"Clarke at the certificate's eps = 1e-6: rho_2/(sum lam_i |g_i|) = {clk['rho2_rel_w']:.1e}; ")
    if any(en_ <= eps_max for _, en_ in no_row):
        v += (f"WARNING {sum(en_ <= eps_max for _, en_ in no_row)} eigenvalue(s) within eps_max of the line without a "
              f"row (missing normal cones); ")
    prof = out.get('profile') or []
    if prof:
        pm = min(q['rho2_rel_w'] for q in prof)
        pq = min([q for q in prof if q['rho2_rel_w'] <= 2 * pm], key=lambda q: q['eps'])
        if pq['eps'] > pick['eps'] * (1 + 1e-6) and pq['rho2_rel_w'] < 0.5 * pick['rho2_rel_w']:
            v += (f"over P({eps_p:g}) the residual falls to {pq['rho2_rel_w']:.1e} only at eps = {pq['eps']:.1e} "
                  f"({pq['n_pieces']} pieces: eps-stationary at that scale only); ")
    if a.descent == 'off':
        pv = [q for q in prs if q['Delta'] <= D_set]
        v += f"first-order decrease (model only, --descent off): pred_inf/gamma = {pv[-1]['pred']:.1e} at Delta = {pv[-1]['Delta']:.0e}; " if pv else ''
    elif Dv:
        v += (f"first-order decrease available pred_inf(Delta_v)/gamma = {-Dv['pred']:.1e} at Delta_v = {Dv['Delta']:.0e} "
              f"(measured {-av_of(Dv):.1e}" + (' with the level set\'s tops' if 'actual_level' in Dv else ' at the re-located tops of P(%g)' % eps_p)
              + (f"; certified gamma <= {cert['gamma_bound']:.6f} at Delta {cert['Delta']:.0e}" if cert and cert['certified'] else '') + "); ")
    elif dsc and dsc[0].get('t') is None:
        v += "no first-order decrease predicted (pred_inf <= 1e-10 gamma); "
    else:
        v += ("no step of the descent ladder both confirmed the first-order model (actual >= pred / 2) and stayed admissible "
              f"(Delta_v below {float(a.descent_deltas.split(',')[0]):.0e}); ")
    if cert and not cert['certified']:
        v += (f"certificate NOT obtained ({len(cert.get('tries', []))} step(s) tried; last at Delta {cert['Delta']:.0e}: line clear "
              f"{cert['line_clear']}, alpha {cert['alpha']:+.6f}, level set {cert['level']}, the step's max over the known tops "
              f"{cert['actual']:+.2e} relative); ")
    v += ((f"piece set certified by the level set at gamma (1 - eps) for eps in {sorted(final)} (a second top inside a "
           f"known top's region is not separated)") if certified else
          ("piece set uncertified (" + ('--complete off' if not comp else 'the level set did not clear') + ": rho an upper bound)")) + \
         f"; stop: {stop_txt}"
    out['verdict'] = v
    out['verdict_pick'] = dict(eps=pick['eps'], rho2_rel_w=pick['rho2_rel_w'], rho2_rel=pick['rho2_rel'],
                               rho2_rel_min=pick['rho2_rel_min'], clarke_rho2_rel_w=clk['rho2_rel_w'], clarke_rho2_rel=clk['rho2_rel'],
                               group_sep_worst=((kw_, bgc[kw_]) if kw_ else None), cq_margin=pick.get('cq_margin'))
    P("VERDICT: " + v)


def modal_frozen(ctl, pk, z, zref, L):
    """The modal rows at z with each 2 x 2 block's v2 scale s = |a - L| + |g - L| frozen at zref, as
    controller.modal_rows' gradient has it (exact where v2 is active, det = 0): the rows' FD reference."""
    ar = pk.unpack(z)[0]; ar0 = pk.unpack(zref)[0]
    v = np.array([v_ for v_, _ in ctl.modal_rows(pk.modal, *[np.asarray(x_) for x_ in ar[:3]], L)])
    d, d0 = np.real(np.asarray(ar[0])), np.real(np.asarray(ar0[0]))
    k = 0
    for bl in pk.modal:
        if len(bl) == 1:
            k += 1; continue
        i, j = bl
        s_, s0 = (max(abs(x_[i] - L) + abs(x_[j] - L), 1e-3) for x_ in (d, d0))
        v[k + 1] = L - (L - v[k + 1]) * s_ / s0              # det(M - L I) / s0
        k += 2
    return v


def struct_is_mimo(pk):
    return bool(getattr(pk, 'mimo', False))


if __name__ == '__main__':
    raise SystemExit(main())
