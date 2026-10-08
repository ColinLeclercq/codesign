"""check_lattice.py -- the lattice oracle (lattice.py) against the serial full
model, on a record of the backward-facing step (BFS.md 8).

    python tools/check_lattice.py RECORD.npz [--mimo m p struct] [--out OUT.npz] [--ref REF.npz]
    mpiexec -n 3 python tools/check_lattice.py ...       (the same on MPI ranks)

Checks, at the record's design:
  rows    every fixed row (sigma_1 and its packed gradient) against cl.gain +
          D.grads on a fresh closed loop: identical;
  follow  each local maximum of a sweep, placed by bracket and refine on the
          lattice, against the exact top (peakfollow.follow, safeguarded Newton
          on the full model): the row's deficit below the top <= FOLLOW_TOL;
  line    the stability line in waves against FECoDesign.alpha_certified (the
          serial certificate, stability_and_abscissa): the same clear / not clear,
          the same rightmost eigenvalue, and its gradient row against the one
          from the serial eigenvectors (eigvecs_at);
  reuse   the same calls at a design moved by 1e-4 (the followed rows and the
          line in one evaluation, as the SQP calls them): waves, shifts,
          factorisations; the line again alone and from scratch.
--out saves the numbers; --ref compares them with a saved run (MPI vs serial).
"""
import sys, time, argparse
sys.path.insert(0, '.')
import numpy as np
import sparse_lu; sparse_lu.ORDER = 'mumps'
from bfs import mpipool
if mpipool.active() and mpipool.comm().Get_rank() > 0:
    mpipool.serve()
import fullpar, lattice
from bfs.plant import BFSPlant
from run_case_epi import parse_case
from packer import Packer, PackerMIMO

ap = argparse.ArgumentParser()
ap.add_argument('record')
ap.add_argument('--ops', default='bfs/data/ops_coarse.npz')
ap.add_argument('--mimo', nargs=3, default=None)
ap.add_argument('--cap', type=int, default=10)
ap.add_argument('--out', default=None)
ap.add_argument('--ref', default=None)
ap.add_argument('--eps', type=float, default=0.05)
ap.add_argument('--whi', type=float, default=10.0)
a = ap.parse_args()

z = np.load(a.record, allow_pickle=True)
R = len(np.atleast_1d(z['d']))
D = BFSPlant(a.ops, r=R)
flags = parse_case('K+B+C'); flags['real_ctrl'] = True
if a.mimo:
    m, p, st = int(a.mimo[0]), int(a.mimo[1]), a.mimo[2]
    flags.update(m=m, p=p, struct=st, loop_orders=[int(v) for v in np.atleast_1d(z['loop_orders'])]
                 if 'loop_orders' in z.files else [R // m] * m)
    arow = (np.asarray(z['d'], complex), np.asarray(z['e'], complex), np.asarray(z['f'], complex),
            np.asarray(z['Bc'], complex).reshape(R, p))
    nvec = np.asarray(z['nvec'], complex).reshape(m, R); K = np.asarray(z['K'], complex).reshape(m, p)
    xa = np.asarray(z['xa'], float).reshape(m, 2); xs = np.asarray(z['xs'], float).reshape(p, 1)
    pk = PackerMIMO(D, flags, arow, nvec, K, xa, xs, None, None)
else:
    arow = tuple(np.asarray(z[k], complex) for k in ('d', 'e', 'f'))
    nvec = np.asarray(z['nvec'], complex); K = complex(z['K'])
    xa = np.asarray(z['xa'], float).ravel(); xs = np.atleast_1d(np.asarray(z['xs'], float))
    pk = Packer(D, flags, arow, nvec, K, xa, xs, None, None)
z0 = pk.z0()
lattice.LU_CAP = a.cap
fullpar.start(1, D=D)                      # the context (the plant); under MPI the ranks get it too
pool = fullpar.POOL if mpipool.active() else None
lat = lattice.LatticeOracle(D, pk, flags, pool=pool)
lat.verbose = True
print(f"record {a.record}: order {R}, {'MIMO ' + str(a.mimo) if a.mimo else 'SISO'}; ranks {lat.size}, cap {a.cap}", flush=True)
out = {}


def direct(zz, w):
    arow_, nvec_, K_, xa_, xs_, _, _ = pk.unpack(zz)
    b_, c_, pa_, ps_, na_, ns_ = D.bc(xa_, xs_)
    cl = D.cl(b_, c_)
    r = cl.gain(float(w), arow_, nvec_, K_)
    g = np.asarray(pk.grad(D.grads(cl, r['x'], r['z'], arow_, nvec_, K_, xa_, xs_, pa_, ps_, na_, ns_, flags)), float)
    return r['sigma'], g, cl


# ---- rows
W = np.array([0.0, 0.1, 0.25, 0.42, 0.79, 1.13, 1.31, 1.79, 2.35, 3.0, 5.0, 8.0])
t0 = time.perf_counter()
s, Gs, blocks, tops, info, _ = lat.evaluate(z0, W)
t1 = time.perf_counter()
dmax = gmax = 0.0
for i, w in enumerate(W):
    sd, gd, _ = direct(z0, lattice.snap(w, lattice.ROW_LEVEL))
    dmax = max(dmax, abs(s[i] - sd) / sd); gmax = max(gmax, np.max(np.abs(Gs[i] - gd)) / max(np.max(np.abs(gd)), 1e-300))
print(f"rows: {len(W)} fixed rows in {t1 - t0:.1f} s; vs the direct evaluation: sigma rel {dmax:.1e}, gradient rel {gmax:.1e}",
      flush=True)
out['rows_s'] = s; out['rows_G'] = Gs
# sigma_2 blocks (the Clarke 2x2 rows) at two members
_, _, blk, _, _, _ = lat.evaluate(z0, W[:3], enroll=(0, 2))
arow_, nvec_, K_, xa_, xs_, _, _ = pk.unpack(z0)
b_, c_, pa_, ps_, na_, ns_ = D.bc(xa_, xs_)
clb = D.cl(b_, c_)
Gf = lambda x_, z_: np.asarray(pk.grad(D.grads(clb, x_, z_, arow_, nvec_, K_, xa_, xs_, pa_, ps_, na_, ns_, flags)), float)
bmax = 0.0
for i in (0, 2):
    P = clb.gain_k(lattice.snap(W[i], lattice.ROW_LEVEL), arow_, nvec_, K_, k=2)
    Gc = np.zeros_like(blk[i]['Gc'])
    Gc[0, 0] = Gf(P[0]['x'], P[0]['z']); Gc[1, 1] = Gf(P[1]['x'], P[1]['z'])
    Gc[0, 1] = Gf(P[1]['x'], P[0]['z']) + 1j * Gf(P[1]['x'], 1j * P[0]['z'])
    Gc[1, 0] = Gf(P[0]['x'], P[1]['z']) + 1j * Gf(P[0]['x'], 1j * P[1]['z'])
    bmax = max(bmax, np.max(np.abs(Gc - blk[i]['Gc'])) / np.max(np.abs(Gc)),
               np.max(np.abs(blk[i]['sig'] - [P[0]['sigma'], P[1]['sigma']])) / P[0]['sigma'])
print(f"sigma_2 blocks at 2 members: vs the serial construction (sqp_inner._block_set) rel {bmax:.1e}", flush=True)

# ---- sweep and follow
ws = lat.grid(0.0, 3.0, 4)
t0 = time.perf_counter(); sv = lat.sigmas(z0, ws); t1 = time.perf_counter()
loc = [i for i in range(1, len(ws) - 1) if sv[i] >= sv[i - 1] and sv[i] >= sv[i + 1]]
loc = sorted(loc, key=lambda i: -sv[i])[:4]
print(f"sweep: {len(ws)} points on [0, 3] in {t1 - t0:.1f} s; local maxima {[round(float(ws[i]), 4) for i in loc]}", flush=True)
from peakfollow import follow as newton_follow
Wf = np.array([float(ws[i]) for i in loc])
fol = {round(float(w), 6): 0.1 for w in Wf}
w_star, state = {}, {}
t0 = time.perf_counter(); s2, G2, _, tops, info, _ = lat.evaluate(z0, Wf, follow=fol, w_star=w_star, state=state); t1 = time.perf_counter()
_, _, cl = direct(z0, 0.5)
for i, w in enumerate(Wf):
    wt, rt = newton_follow(cl, float(w), *pk.unpack(z0)[:3], 0.1, nscan=21)
    k6 = round(float(w), 6)
    print(f"  follow {w:.4f}: lattice row at {w_star[k6]:.6f} (level {state[k6]['lvl']}) sigma {s2[i]:.9f}; "
          f"Newton top {wt:.6f} sigma {rt['sigma']:.9f}; deficit {(rt['sigma'] - s2[i]) / rt['sigma']:.1e}", flush=True)
print(f"follow: {len(Wf)} members, {info['nev']} evaluations in {info['waves']} waves, {t1 - t0:.1f} s; largest Hermite "
      f"deficit {info['deficit']:.1e}",
      flush=True)
t0 = time.perf_counter(); s3, _, _, _, info3, _ = lat.evaluate(z0, Wf, follow=fol, w_star=w_star, state=state); t1 = time.perf_counter()
print(f"follow again (same design, warm starts): {info3['nev']} evaluations in {info3['waves']} waves, {t1 - t0:.1f} s, rows' largest relative "
      f"difference {np.max(np.abs(s3 - s2) / s2):.1e}", flush=True)
out['follow_s'] = s2

# ---- line
t0 = time.perf_counter(); L = lat.line(z0, a.eps, 0.0, a.whi); t1 = time.perf_counter()
print(f"line (lattice): clear {L['clear']}, alpha {L['alpha']:+.10f}, {L['nshift']} shifts in {L['waves']} waves "
      f"(cover kept: {len(lat.line_prev.get(round(a.eps, 12), []))} centres), "
      f"{t1 - t0:.1f} s; {len(L['rows'])} rows within {lattice.ROW_REACH} of the line, rightmost "
      f"{L['rows'][0]['lam'] if L['rows'] else None}", flush=True)
arow_, nvec_, K_, xa_, xs_, _, _ = pk.unpack(z0)
b_, c_, pa_, ps_, na_, ns_ = D.bc(xa_, xs_)
t0 = time.perf_counter()
al, lam, v, u, cla, clr = D.alpha_certified(arow_, nvec_, K_, b_, c_, eps=a.eps, wlo=0.0, whi=a.whi)
t1 = time.perf_counter()
print(f"line (serial certificate): clear {clr}, alpha {al:+.10f} at {lam}, {t1 - t0:.1f} s", flush=True)
if L['rows'] and v is not None:
    eta = np.vdot(u, cla.Mcl_apply(v))
    gser = np.asarray(pk.grad(D.grads(cla, v, u, arow_, nvec_, K_, xa_, xs_, pa_, ps_, na_, ns_, flags, eta=eta)), float)
    j = int(np.argmin([abs(r_['lam'] - lam) for r_ in L['rows']]))
    glat = np.asarray(pk.grad(L['rows'][j]['g']), float)
    print(f"  rightmost: lattice {L['rows'][j]['lam']:.10f} vs serial {lam:.10f}; gradient rel diff "
          f"{np.max(np.abs(glat - gser)) / np.max(np.abs(gser)):.1e}", flush=True)
vec = lat.vectors(z0, L['rows'][:2])
print(f"  vectors of the two rightmost: {[None if q is None else q[0].shape for q in vec]}", flush=True)
out['line_alpha'] = L['alpha']; out['line_clear'] = L['clear']

# ---- reuse at a moved design
rng = np.random.default_rng(0)
z1 = z0 + 1e-4 * rng.standard_normal(z0.shape) * np.maximum(np.abs(z0), 1e-2)
n0, h0 = lat.counts()
t0 = time.perf_counter(); s4, _, _, _, info4, _ = lat.evaluate(z1, W); t1 = time.perf_counter()
n1, h1 = lat.counts()
print(f"moved design: {len(W)} fixed rows {t1 - t0:.1f} s, {n1 - n0} factorisations, {h1 - h0} cached", flush=True)
# the SQP's evaluation: followed rows and the line in the same waves
t0 = time.perf_counter()
s5, _, _, _, info5, L1 = lat.evaluate(z1, Wf, follow=fol, w_star=w_star, state=state, line=(a.eps, 0.0, a.whi))
t1 = time.perf_counter()
n2, h2 = lat.counts()
print(f"moved design, rows and line together: {len(Wf)} followed rows ({info5['nev']} evaluations), line clear "
      f"{L1['clear']}, alpha {L1['alpha']:+.10f}, {L1['nshift']} shifts; {info5['waves']} waves, {t1 - t0:.1f} s, "
      f"{n2 - n1} factorisations, {h2 - h1} cached", flush=True)
lat.line_prev.clear()
L2 = lat.line(lat.design_raw(*pk.unpack(z1)[:5]), a.eps, 0.0, a.whi, eps_int=1e-5)
print(f"moved design, line alone from scratch: clear {L2['clear']}, alpha {L2['alpha']:+.10f}, {L2['nshift']} shifts in "
      f"{L2['waves']} waves", flush=True)
print(lat.report(), flush=True)
out['moved_alpha'] = L1['alpha']
if a.out:
    np.savez(a.out, **out)
if a.ref:
    ref = np.load(a.ref)
    for k in out:
        print(f"  vs ref {k}: max abs diff {np.max(np.abs(np.asarray(out[k], float) - np.asarray(ref[k], float))):.1e}", flush=True)
