#!/usr/bin/env python3
"""curves.py -- sigma_1, sigma_2 of the closed loop, the open-loop floor, and the
controller's frequency response for a set of records (2026-09-25).
    python tools/curves.py out.npz  label:ne:case:record ...
Pointwise floor: for rank-one feedback through (b, c), sigma_1 of the closed
loop >= sigma_2 of the OPEN loop at every omega (interlacing), hence
gamma >= sup_w sigma_2^ol (the v6 4.3 'absolute floor')."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import numpy as np
from fe_codesign import FECoDesign, parse_case
import controller as ctl

ws = np.unique(np.concatenate([np.arange(-30, -8, 0.05), np.arange(-8, 8.0001, 0.005), np.arange(8, 30.0001, 0.05)]))
out = dict(omega=ws)
Ds = {}
for spec in _s.argv[2:]:
    lab, ne, case, rec = spec.split(':', 3); ne = int(ne)
    d = np.load(rec, allow_pickle=True)
    r = len(d['d'])
    if (ne, r) not in Ds:
        Ds[(ne, r)] = FECoDesign(ne=ne, r=r)
    D = Ds[(ne, r)]
    aa = (d['d'], d['e'], d['f']); nn = d['nvec']; KK = complex(d['K'])
    va = d['va'] if np.size(d['va']) else None; vs = d['vs'] if np.size(d['vs']) else None
    b, c, *_ = D.bc(float(d['xa']), float(d['xs']), va, vs); cl = D.cl(b, c)
    s = np.array([[p['sigma'] for p in cl.gain_k(float(w), aa, nn, KK, k=2)] for w in ws])
    # open loop, same actuator/sensor (controller disconnected)
    so = np.array([[p['sigma'] for p in cl.gain_k(float(w), aa, np.zeros_like(nn), 0j, k=2)] for w in ws])
    Ac = ctl.A_of(*aa); Bc = np.full(r, D.mc, complex); I = np.eye(r)
    Kw = np.array([KK + nn @ np.linalg.solve(1j * w * I - Ac, Bc) for w in ws])
    out[lab + '_s1'], out[lab + '_s2'] = s[:, 0], s[:, 1]
    out[lab + '_ol1'], out[lab + '_ol2'] = so[:, 0], so[:, 1]
    out[lab + '_K'] = Kw; out[lab + '_poles'] = np.linalg.eigvals(Ac)
    out[lab + '_gamma'] = float(d['gamma']); out[lab + '_xa'] = float(d['xa']); out[lab + '_xs'] = float(d['xs'])
    print(f"{lab}: max sigma_1 {s[:, 0].max():.6f} (record {float(d['gamma']):.6f}); sup sigma_2 open loop {so[:, 1].max():.6f}; "
          f"closed-loop sigma_2 max {s[:, 1].max():.6f}", flush=True)
np.savez(_s.argv[1], **out)
