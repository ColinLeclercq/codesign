#!/usr/bin/env python3
"""replay_round.py -- replay one round's SQP solve from a dump (K+B+C), for diagnosis (2026-09-26).

    DUMPDIR=results_n100_kbc/dump_<id> OUTER=013 FOLLOW_SCAN=0 ENR=-0.528,-0.5256,-0.5095,-0.5 \
        python tools/replay_round.py <label> <maxit>

Starts from the design stored in outer_<OUTER>.npz (the end of that round) on the
working set W_final of its json, with the sigma_2 rows at the frequencies ENR (copy
them from the run log's "sigma2 row at" lines of the next round -- the replay is
faithful only then), FOLLOW_SCAN as the run's --follow-scan.  Env switches:
DEBUG_LS=1 (per-trial rows, rejected-search re-evaluation), ENTER_SEP (pair entry
separation), NODEDUP=1 (no shared-top restarts), NE (default 100).  The solver's
other settings are the n100_kbc.sh ones.  STAGE5 28-29."""
import sys, os, json, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import numpy as np
from fe_codesign import FECoDesign, parse_case
from packer import Packer
import sqp_inner as S, reduced as RD, run_case_epi as R
mode = sys.argv[1]; S.FOLLOW_DEDUP = not os.environ.get("NODEDUP"); S.PAIR_ENTER_SEP = float(os.environ.get("ENTER_SEP", S.PAIR_ENTER_SEP)); S.FOLLOW_SCAN = int(os.environ.get("FOLLOW_SCAN", "0")); S.DEBUG_LS = bool(os.environ.get("DEBUG_LS")); maxit = int(sys.argv[2]) if len(sys.argv) > 2 else 15
D = FECoDesign(ne=int(os.environ.get('NE', '100')), r=8); D.alpha_argmax = True; fl = parse_case('K+B+C')
d = os.environ['DUMPDIR']; ko = os.environ.get('OUTER', '013')
z8 = np.load(f'{d}/outer_{ko}.npz'); j = json.load(open(f'{d}/outer_{ko}.json'))
aC = (z8['design_d'], z8['design_e'], z8['design_f']); nC = z8['design_nvec']; KC = complex(z8['design_K'])
xaC, xsC = float(z8['design_xa']), float(z8['design_xs'])
W = np.sort(np.asarray(j['inner']['W_final'], float)); ws_star = j['inner'].get('w_star') or {}
pk = Packer(D, fl, aC, nC, KC, xaC, xsC, D.gauss(xaC), D.gauss(xsC)); zC = pk.z0()
bC, cC, *_ = D.bc(xaC, xsC, None, None); clC = D.cl(bC, cC)
ws = np.linspace(-6, 6, 1201); sC = np.array([clC.gain(float(w), aC, nC, KC, want_vectors=False)['sigma'] for w in ws])
ro = RD.ReducedOracle(D); ro.set_input_basis(RD.input_basis(D, 'b', -3, 0, other=0.98)[0], RD.input_basis(D, 'c', 0, 3, other=-1.03)[0]); ro.galerkin = True
for w in R._enrol_peaks(ws, sC, None, rel=0.3):
    ro.enrich(clC, float(w), aC, nC, KC, nvec_dirs=3)
al, lam, *_ = D.alpha_certified(aC, nC, KC, bC, cC, eps=0.05); ro.enrich_eig(clC, [lam], aC, nC, KC)
fol = {round(float(k), 6): 0.1 for k in ws_star}
if os.environ.get('PRUNE'):
    # Omega = the current design's local maxima within PRUNE of the max (full model, fine sweep), followed
    wf = np.linspace(-30, 30, 4801); gf = np.array([clC.gain(float(w), aC, nC, KC, want_vectors=False)['sigma'] for w in wf])
    pk_ = R._enrol_peaks(wf, gf, None, rel=float(os.environ['PRUNE']))
    print('pruned Omega:', len(W), '->', len(pk_), 'local maxima within', os.environ['PRUNE'])
    W = np.sort(np.asarray(pk_, float)); fol = {round(float(k), 6): 0.1 for k in W}
enr = [int(np.argmin(np.abs(W - w))) for w in [float(x) for x in os.environ['ENR'].split(',')]]
t0 = time.time()
best, ncalls, H, st = S.solve_epigraph_sqp(D, pk, fl, zC, list(W), float(j['gamma_certified']), 0.05, maxit=maxit, ctrl_max=-0.05,
    enroll=enr, kink_block=True, trust0=0.00835, ctrl_lmi=False, ctrl_block=True, follow=fol, follow_radius=0.1,
    row_screen=0.05, cl_block=True, oracle=ro, res_tol=1e-3, refine_tol=1e-4, refine_grid=np.linspace(-30, 30, 1201), refine_max=3, verbose=False)
print('ENTER_SEP', S.PAIR_ENTER_SEP, 'dedup', S.FOLLOW_DEDUP, st.get('dedup')); print(mode, '|W|', len(W), 'start', j['gamma_certified'], '-> best', best['g'], ncalls, 'calls', round(time.time()-t0), 's code', st.get('termination_code'))
for h in st['history']:
    print(f"  it {h.get('it')}: t {h.get('t', float('nan')):.6f} smax {h.get('smax', float('nan')):.6f} stat {h.get('stat', float('nan')):.2e} "
          f"step {h.get('step', float('nan')):.2e} a {h.get('alpha', float('nan')):g} D {h.get('Delta', float('nan')):.2e} {h.get('accepted')}")
print("last iterate:", "al_ctrl", st.get('e_last_al_ctrl'), "al", st.get('e_last_al'))
zl = np.array(st['z_last']); import controller as ctl
pz = np.linalg.eigvals(ctl.A_of(*pk.unpack(zl)[0])); ii = np.argsort(-pz.real)[:2]; print("last iterate poles", [f"{x.real:+.10f} {x.imag:+.6f}j" for x in pz[ii]])
print("pair log", st.get('pair_log'), "mode at end", st.get('pair_mode_end'))
