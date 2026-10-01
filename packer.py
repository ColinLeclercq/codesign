"""
packer.py — the design-vector packing/unpacking, extracted.

`Packer` maps between the optimiser's real vector z and the physical design
(tridiagonal controller, actuator/sensor centres, Sobolev field variables),
carrying the per-group scalings.  It is pure numpy.

It used to live in run_case.py, the LEGACY max-formulation driver, so importing
it dragged in that whole file -- including the adaptive ladder and, through
pygranso, torch.  Nothing in the epigraph pipeline needs any of that.
"""
from __future__ import annotations
import numpy as np

EPS_STAB_DEFAULT = 1e-3


class Packer:
    """Pack/unpack only the ACTIVE design groups; per-group scaling."""
    def __init__(self, D, flags, arow0, nvec0, K0, xa0, xs0, va0, vs0):
        self.D, self.f = D, flags
        # SYMMETRIC-SUBSPACE mode (flags['sym']): the co-design problem is
        # exactly invariant under S: psi_a -> R conj(psi_s), psi_s ->
        # R conj(psi_a), xa -> -xs, xs -> -xa (mu even, channels R-invariant,
        # SISO compensator).  This mode restricts the optimisation to the
        # FIXED-POINT subspace of S: only the actuator side is packed, and the
        # sensor side is DERIVED as psi_s = s * conj(R psi_a) (xs = -xa),
        # where s = flags['sym_branch'] in {+1, -1} selects the branch --
        # both are fixed points, and averaging across branches annihilates
        # the field.  Halves the field dofs as a bonus.  The optimiser then
        # answers the question the projection test cannot: what is the BEST
        # symmetric design, not merely the midpoint of an asymmetric pair.
        self.sym = bool(flags.get('sym'))
        self.sbr = float(flags.get('sym_branch', 1.0)) or 1.0
        self.sym_fld = self.sym and bool(flags.get('FB')) \
            and bool(flags.get('FC'))
        self.sym_pos = self.sym and bool(flags.get('B')) \
            and bool(flags.get('C'))
        # arow0 is the TRIDIAGONAL triple (d, e, f)
        self.d0, self.e0, self.f0 = (np.asarray(v, complex) for v in arow0)
        self.arow0, self.nvec0, self.K0 = arow0, nvec0, K0
        self.xa0, self.xs0, self.va0, self.vs0 = xa0, xs0, va0, vs0
        th = np.concatenate([self.d0, self.e0, self.f0, nvec0, [K0]])
        mag = np.abs(th)
        self.sc = np.maximum(mag, 1e-3 * max(mag.max(), 1.0))
        self.sK = max(abs(self.K0), 1.0)   # static-gain scale
        self.sx = 0.30          # placement scale
        self.nth = len(th)
        # backward-facing step (BFS.md 2): a REAL controller (only the real
        # parts are design variables) and 2-D placements (pos_dim entries each)
        self.real_ctrl = bool(flags.get('real_ctrl'))
        self.pd = int(getattr(D, 'pos_dim', 1))

    def z0(self):
        z = []
        m = self.f.get('ctrl_mode', 'full')
        if self.f['K'] and m in ('full', 'sp'):
            th = np.concatenate([self.d0, self.e0, self.f0,
                                 self.nvec0, [self.K0]])
            k = self.nth if m == 'full' else self.nth - 1   # drop Dc if s.proper
            z += [th[:k].real / self.sc[:k]] + ([] if self.real_ctrl else [th[:k].imag / self.sc[:k]])
        elif self.f['K'] and m == 'gain':          # static complex gain
            z += [np.array([self.K0.real / self.sK, self.K0.imag / self.sK])]
        elif self.f['K'] and m == 'gainr':         # static real gain
            z += [np.array([self.K0.real / self.sK])]
        if self.f['B']:  z += [np.atleast_1d(np.asarray(self.xa0, float)) / self.sx]
        if self.f['C'] and not self.sym_pos:
            z += [np.atleast_1d(np.asarray(self.xs0, float)) / self.sx]
        Pa, Ps = getattr(self.D, 'field_basis_a', None), getattr(self.D, 'field_basis_s', None)
        Aa, As = getattr(self.D, 'atoms_a', None), getattr(self.D, 'atoms_s', None)
        if self.f['FB']:
            if Aa is not None:                     # Gabor atoms (fieldbasis, STAGE5 32): the parameter vector itself
                z += [np.asarray(Aa['p0'], float)]
            else:
                xa_ = (Pa.T @ (self.D.M @ self.va0)) if Pa is not None else self.va0   # beta0 = Phi^T M v0
                z += [xa_.real] if self.f.get('real_fb') else [xa_.real, xa_.imag]
        if self.f['FC'] and not self.sym_fld:
            if As is not None:
                z += [np.asarray(As['p0'], float)]
            else:
                xs_ = (Ps.T @ (self.D.M @ self.vs0)) if Ps is not None else self.vs0
                z += [xs_.real] if self.f.get('real_fc') else [xs_.real, xs_.imag]
        return np.concatenate(z)

    def unpack(self, z):
        i = 0; r = self.D.r; n = self.D.n
        m = self.f.get('ctrl_mode', 'full')
        if self.f['K'] and m in ('full', 'sp'):
            k = self.nth if m == 'full' else self.nth - 1
            if self.real_ctrl:
                th = z[i:i+k] * self.sc[:k] + 0j; i += k
            else:
                th = (z[i:i+k] + 1j*z[i+k:i+2*k]) * self.sc[:k]; i += 2*k
            j = 0
            dd = th[j:j+r]; j += r
            ee = th[j:j+r-1]; j += r - 1
            ff = th[j:j+r-1]; j += r - 1
            nvec = th[j:j+r]; j += r
            K = th[j] if m == 'full' else 0j       # strictly proper: Dc = 0
            arow = (dd, ee, ff)
        elif self.f['K'] and m in ('gain', 'gainr'):
            # STATIC gain: the controller states are decoupled (n = 0), so the
            # loop is u = Dc*y and A_cl is block triangular; the (stable) seed
            # poles remain as passive closed-loop eigenvalues.
            arow = self.arow0
            nvec = np.zeros(r, complex)
            if m == 'gain':
                K = (z[i] + 1j * z[i+1]) * self.sK; i += 2
            else:
                K = complex(z[i] * self.sK); i += 1
        else:
            arow, nvec, K = self.arow0, self.nvec0, self.K0
        xa = self.xa0; xs = self.xs0
        pd = self.pd
        if self.f['B']:
            xa = float(z[i] * self.sx) if pd == 1 else np.asarray(z[i:i + pd], float) * self.sx; i += pd
        if self.f['C']:
            if self.sym_pos:
                xs = -xa
            else:
                xs = float(z[i] * self.sx) if pd == 1 else np.asarray(z[i:i + pd], float) * self.sx; i += pd
        va = vs = None
        Pa, Ps = getattr(self.D, 'field_basis_a', None), getattr(self.D, 'field_basis_s', None)
        Aa, As = getattr(self.D, 'atoms_a', None), getattr(self.D, 'atoms_s', None)
        if Aa is not None or As is not None:
            import fieldbasis as _fb
            x_ = self.D.xn
            if self.f['FB']:
                m_ = _fb.NATOM_PAR * Aa['n']
                va = _fb.atom_field(x_, *_fb.split_atoms(z[i:i + m_], Aa['n']), Aa['smin'], Aa['kmax']); i += m_
            if self.f['FC']:
                m_ = _fb.NATOM_PAR * As['n']
                vs = _fb.atom_field(x_, *_fb.split_atoms(z[i:i + m_], As['n']), As['smin'], As['kmax']); i += m_
            self._atom_z = (z.copy(), i)
            return arow, nvec, K, xa, xs, va, vs
        if self.f['FB']:
            na_ = Pa.shape[1] if Pa is not None else n
            if self.f.get('real_fb'):
                xi = z[i:i+na_].astype(complex); i += na_
            else:
                xi = z[i:i+na_] + 1j*z[i+na_:i+2*na_]; i += 2*na_
            va = (Pa @ xi) if Pa is not None else self.D.smooth_map(xi)
        if self.f['FC']:
            if self.sym_fld:
                # DERIVED, not free: the fixed point of the mirror symmetry.
                # smooth_map is real and reflection-commuting, so applying the
                # constraint after smoothing equals applying it to xi.
                vs = self.sbr * np.conj(va[::-1])
            elif self.f.get('real_fc'):
                ns_ = Ps.shape[1] if Ps is not None else n
                xi = z[i:i+ns_].astype(complex); i += ns_
                vs = (Ps @ xi) if Ps is not None else self.D.smooth_map(xi)
            else:
                ns_ = Ps.shape[1] if Ps is not None else n
                xi = z[i:i+ns_] + 1j*z[i+ns_:i+2*ns_]; i += 2*ns_
                vs = (Ps @ xi) if Ps is not None else self.D.smooth_map(xi)
        return arow, nvec, K, xa, xs, va, vs

    def grad(self, g, z=None):
        out = []
        m = self.f.get('ctrl_mode', 'full')
        if self.f['K'] and m in ('full', 'sp'):
            gc = np.concatenate([g['d'], g['e'], g['f'], g['n'], [g['K']]])
            k = self.nth if m == 'full' else self.nth - 1
            out += [np.real(gc[:k])*self.sc[:k]] + ([] if self.real_ctrl else [-np.imag(gc[:k])*self.sc[:k]])
        elif self.f['K'] and m == 'gain':
            out += [np.array([np.real(g['K'])*self.sK, -np.imag(g['K'])*self.sK])]
        elif self.f['K'] and m == 'gainr':
            out += [np.array([np.real(g['K'])*self.sK])]
        if self.f['B']:
            gxa = g['xa'] - g['xs'] if self.sym_pos else g['xa']
            out += [np.atleast_1d(np.asarray(gxa, float)) * self.sx]
        if self.f['C'] and not self.sym_pos:
            out += [np.atleast_1d(np.asarray(g['xs'], float)) * self.sx]
        Pa, Ps = getattr(self.D, 'field_basis_a', None), getattr(self.D, 'field_basis_s', None)
        Aa, As = getattr(self.D, 'atoms_a', None), getattr(self.D, 'atoms_s', None)
        if Aa is not None or As is not None:
            # atoms: the gradient depends on the parameters (nonlinear map); the
            # design is the one last unpacked unless z is given
            import fieldbasis as _fb
            zz = np.asarray(z if z is not None else self._atom_z[0], float)
            i = sum(len(o) for o in out)
            if self.f['FB']:
                m_ = _fb.NATOM_PAR * Aa['n']
                out += [np.asarray(g['va_packed'], float) if 'va_packed' in g else
                        _fb.atom_grad(self.D.xn, *_fb.split_atoms(zz[i:i + m_], Aa['n']), Aa['smin'], Aa['kmax'],
                                      np.asarray(g['va'], complex))]; i += m_
            if self.f['FC']:
                m_ = _fb.NATOM_PAR * As['n']
                out += [np.asarray(g['vs_packed'], float) if 'vs_packed' in g else
                        _fb.atom_grad(self.D.xn, *_fb.split_atoms(zz[i:i + m_], As['n']), As['smin'], As['kmax'],
                                      np.asarray(g['vs'], complex))]; i += m_
            return np.concatenate(out)
        if self.f['FB'] and 'va_packed' in g:
            out += [np.asarray(g['va_packed'], float)]
        elif self.f['FB']:
            if Pa is not None and self.sym_fld:
                raise NotImplementedError('field basis with the mirror-symmetric field constraint')
            gx = (Pa.T @ np.asarray(g['va'])) if Pa is not None else self.D.smooth_map_adj(g['va'])
            if self.sym_fld:
                # PULLBACK of the derived sensor side.  vs = s * conj(R va) is
                # ANTILINEAR in va but real-linear in (Re, Im), and with the
                # pairing dF = Re<conj(g), dv> the chain rule gives, for
                # h = P^T (R g_vs):  d/dRe += s Re(h), d/dIm -= s Im(h) --
                # i.e. the contribution is s * conj(h) in the [Re, Im]
                # convention used below.  (P real symmetric and
                # reflection-commuting lets R pass inside smooth_map_adj.)
                h = self.D.smooth_map_adj(np.asarray(g['vs'])[::-1])
                gx = gx + self.sbr * np.conj(h)
            out += [gx.real] if self.f.get('real_fb') else [gx.real, gx.imag]
        if self.f['FC'] and not self.sym_fld:
            if 'vs_packed' in g:
                out += [np.asarray(g['vs_packed'], float)]
            else:
                gx = (Ps.T @ np.asarray(g['vs'])) if Ps is not None else self.D.smooth_map_adj(g['vs'])
                out += [gx.real] if self.f.get('real_fc') else [gx.real, gx.imag]
        return np.concatenate(out)

    def field_jacobians(self, z=None):
        """{'va': [J], 'vs': [J]}: for each free field the complex n x P matrix J
        with packed field gradient = Re(G^H J) for a nodal gradient G (STAGE6
        6.29: the reduced rows project J once per design).  None where the map is
        not of that form here (smoothed raw fields, the mirror-symmetric or
        real-field variants): those rows are lifted."""
        import fieldbasis as _fb
        out = {}
        Aa, As = getattr(self.D, 'atoms_a', None), getattr(self.D, 'atoms_s', None)
        Pa, Ps = getattr(self.D, 'field_basis_a', None), getattr(self.D, 'field_basis_s', None)
        if Aa is not None or As is not None:
            zz = np.asarray(z if z is not None else self._atom_z[0], float)
            ia, is_ = self.atom_params(zz)
            if self.f['FB']:
                out['va'] = [_fb.atom_jacobian(self.D.xn, *_fb.split_atoms(ia, Aa['n']), Aa['smin'], Aa['kmax'])]
            if self.f['FC']:
                out['vs'] = [_fb.atom_jacobian(self.D.xn, *_fb.split_atoms(is_, As['n']), As['smin'], As['kmax'])]
            return out
        if self.f['FB']:
            out['va'] = [np.hstack([np.conj(Pa), 1j * np.conj(Pa)]).astype(complex) if (Pa is not None and not self.sym_fld and not self.f.get('real_fb')) else None]
        if self.f['FC'] and not self.sym_fld:
            out['vs'] = [np.hstack([np.conj(Ps), 1j * np.conj(Ps)]).astype(complex) if (Ps is not None and not self.f.get('real_fc')) else None]
        return out

    def zero_grad(self):
        """The gradient dict with every entry zero (rows that touch only some groups)."""
        r, n = self.D.r, self.D.n
        return dict(d=np.zeros(r, complex), e=np.zeros(max(r - 1, 0), complex), f=np.zeros(max(r - 1, 0), complex),
                    n=np.zeros(r, complex), K=0j,
                    xa=(0.0 if self.pd == 1 else np.zeros(self.pd)), xs=(0.0 if self.pd == 1 else np.zeros(self.pd)),
                    va=np.zeros(n, complex), vs=np.zeros(n, complex))

    def atom_params(self, z):
        """The atom parameter vectors (FB, FC) inside z, or (None, None)."""
        import fieldbasis as _fb
        Aa, As = getattr(self.D, 'atoms_a', None), getattr(self.D, 'atoms_s', None)
        if Aa is None and As is None:
            return None, None
        m_a = _fb.NATOM_PAR * Aa['n'] if (self.f['FB'] and Aa is not None) else 0
        m_s = _fb.NATOM_PAR * As['n'] if (self.f['FC'] and As is not None) else 0
        z = np.asarray(z, float); i0 = len(z) - m_a - m_s          # the field blocks are last
        return (z[i0:i0 + m_a] if m_a else None), (z[i0 + m_a:] if m_s else None)



class PackerMIMO:
    """MIMO design vector (2026-09-29, STAGE6): m actuators, p sensors.

    Physical design: arow = (d, e, f, B_c) (A_c tridiagonal r x r, B_c r x p),
    nvec = C_c (m x r), K = D_c (m x p), xa (m,), xs (p,), va / vs lists of m / p
    fields (or None).  flags: 'm', 'p', 'struct' in {'central', 'decentral'},
    'loop_orders' (decentral: the order of each loop, summing to r), the SISO
    keys K / ctrl_mode / B / C / FB / FC.

    Controller entries are free under a mask:
      central    -- every entry of d, e, f, B_c, C_c, D_c;
      decentral  -- loop k (sensor k -> its own compensator block -> actuator k,
                    m = p): its tridiagonal block (e, f zero across blocks), its
                    row of C_c on its block, D_c[k, k]; B_c FIXED to the SISO
                    gauge mc * ones on each block (as the SISO code fixes it);
      ctrl_mode 'sp': D_c fixed 0;  'gain' (static): only D_c free (masked as above).
    Scales per complex entry: at m = p = 1 the SISO rule max(|theta0|, 1e-3 x
    max |theta0|) (entries starting at zero: 0.1 x their group's median); for
    m + p > 2 max(|theta0|, 0.1 x the median nonzero magnitude of its group, at
    least 1e-3 x the group's max), so a weak loop and the cross couplings step
    with their group.  Central: one entry per row of B_c fixed (the similarity
    gauge; the SISO gauge at m = p = 1)."""

    def __init__(self, D, flags, arow0, nvec0, K0, xa0, xs0, va0, vs0):
        self.D, self.f = D, flags
        self.m, self.p, self.r = int(flags['m']), int(flags['p']), int(D.r)
        m, p, r = self.m, self.p, self.r
        self.mimo = True
        d0, e0, f0 = (np.asarray(v, complex) for v in arow0[:3])
        Bc0 = np.asarray(arow0[3], complex).reshape(r, p) if len(arow0) > 3 else np.full((r, p), D.mc, complex)
        self.th0 = np.concatenate([d0, e0, f0, Bc0.ravel(), np.asarray(nvec0, complex).reshape(m, r).ravel(),
                                   np.asarray(K0, complex).reshape(m, p).ravel()])
        self.sizes = [('d', r), ('e', max(r - 1, 0)), ('f', max(r - 1, 0)), ('Bc', r * p), ('n', m * r), ('K', m * p)]
        self.off = {}; o = 0
        for nm_, sz in self.sizes:
            self.off[nm_] = (o, o + sz); o += sz
        self.xa0 = np.atleast_1d(np.asarray(xa0, float)); self.xs0 = np.atleast_1d(np.asarray(xs0, float))
        self.va0, self.vs0 = va0, vs0
        # free mask
        mode = flags.get('ctrl_mode', 'full')
        free = np.zeros(len(self.th0), bool)
        if flags['K']:
            if mode in ('full', 'sp'):
                free[:] = True
                # GAUGE (STAGE6 6.16): a diagonal similarity S (A_c -> S A_c S^-1 stays
                # tridiagonal, B_c -> S B_c, C_c -> C_c S^-1) leaves K(s) unchanged, so
                # one entry per row of B_c is fixed -- its largest at the start -- as the
                # SISO code fixes B_c = mc * ones.  At m = p = 1 this IS the SISO gauge.
                Bc0m = np.abs(Bc0)
                for k_ in range(r):
                    if Bc0m[k_].max() > 0:
                        free[self.off['Bc'][0] + k_ * p + int(np.argmax(Bc0m[k_]))] = False
                if flags.get('struct', 'central') == 'decentral':
                    orders = list(flags.get('loop_orders') or [r // m] * m)
                    assert sum(orders) == r and m == p == len(orders), 'decentral: m = p loops whose orders sum to r'
                    bnd = np.cumsum(orders)[:-1]
                    for k_ in bnd:                                   # e, f across block boundaries
                        free[self.off['e'][0] + k_ - 1] = False; free[self.off['f'][0] + k_ - 1] = False
                    free[slice(*self.off['Bc'])] = False             # B_c: the SISO gauge per block
                    Cm = np.zeros((m, r), bool); Km = np.zeros((m, p), bool); s0 = 0
                    for k_, ok_ in enumerate(orders):
                        Cm[k_, s0:s0 + ok_] = True; Km[k_, k_] = True; s0 += ok_
                    free[slice(*self.off['n'])] = Cm.ravel(); free[slice(*self.off['K'])] = Km.ravel()
                if mode == 'sp':
                    free[slice(*self.off['K'])] = False
            elif mode in ('gain',):
                Km = np.ones((m, p), bool) if flags.get('struct', 'central') == 'central' else np.eye(m, p, dtype=bool)
                free[slice(*self.off['K'])] = Km.ravel()
            else:
                raise NotImplementedError(f"MIMO ctrl_mode {mode}")
        self.free = free
        # scales: the SISO rule, max(|theta0|, 1e-3 x the largest free |theta0|)
        # (identical at m = p = 1), except for entries that START AT ZERO (a
        # switched-off loop, the cross couplings of a centralised start): those
        # step with 0.1 x the median nonzero magnitude of their group (d, e, f,
        # B_c, C_c, D_c), not with the global floor
        sc = np.ones(len(self.th0))
        mag = np.abs(self.th0)
        gfloor = 1e-3 * max(float(mag[free].max()) if free.any() else 1.0, 1.0)
        for nm_, (a_, b_) in self.off.items():
            g = mag[a_:b_]
            if g.size == 0:
                continue
            nz = g[g > 0]
            if m == 1 and p == 1:
                # the SISO rule: identical trajectories at m = p = 1 (STAGE6 6.16)
                zfloor = max(gfloor, 0.1 * float(np.median(nz)) if nz.size else 1.0)
                sc[a_:b_] = np.where(g > 0, np.maximum(g, gfloor), zfloor)
            else:
                # MIMO: a floor per group for every entry, 0.1 x the group's median
                # nonzero magnitude (at least 1e-3 x its max).  The SISO rule scales a
                # small entry by its own size; with one loop orders of magnitude weaker
                # than the other it slowed K+B+C central from the same start to
                # 5.500 / 5.449 / 5.393 in three rounds against 5.493 / 5.261 / 5.042
                # with this rule (STAGE6 6.22)
                floor_ = max(1e-3 * max(g.max(), 1.0), 0.1 * float(np.median(nz)) if nz.size else 1.0)
                sc[a_:b_] = np.maximum(g, floor_)
        self.sc = sc
        self.sx = 0.30
        self.nth = int(free.sum())

    # ------------------------------------------------------------ helpers
    def _fields_cfg(self):
        D = self.D
        return (getattr(D, 'field_basis_a', None), getattr(D, 'field_basis_s', None),
                getattr(D, 'atoms_a', None), getattr(D, 'atoms_s', None))

    def z0(self):
        z = []
        th = self.th0[self.free]; sc = self.sc[self.free]
        if self.nth:
            z += [th.real / sc, th.imag / sc]
        if self.f['B']:
            z += [self.xa0 / self.sx]
        if self.f['C']:
            z += [self.xs0 / self.sx]
        Pa, Ps, Aa, As = self._fields_cfg()
        for key, flag, P_, A_, v0, cnt in (('va', 'FB', Pa, Aa, self.va0, self.m), ('vs', 'FC', Ps, As, self.vs0, self.p)):
            if not self.f[flag]:
                continue
            for i in range(cnt):
                if A_ is not None:
                    z += [np.asarray(A_['p0'][i], float)]
                else:
                    v = np.asarray(v0[i], complex)
                    x_ = (P_.T @ (self.D.M @ v)) if P_ is not None else v
                    z += [x_.real, x_.imag]
        return np.concatenate(z) if z else np.zeros(0)

    def unpack(self, z):
        z = np.asarray(z, float)
        i = 0; m, p, r, n = self.m, self.p, self.r, self.D.n
        th = self.th0.copy()
        if self.nth:
            k = self.nth; sc = self.sc[self.free]
            th[self.free] = (z[i:i + k] + 1j * z[i + k:i + 2 * k]) * sc; i += 2 * k
        g = lambda nm_: th[slice(*self.off[nm_])]
        arow = (g('d'), g('e'), g('f'), g('Bc').reshape(r, p))
        nvec = g('n').reshape(m, r); K = g('K').reshape(m, p)
        xa, xs = self.xa0.copy(), self.xs0.copy()
        if self.f['B']:
            xa = z[i:i + m] * self.sx; i += m
        if self.f['C']:
            xs = z[i:i + p] * self.sx; i += p
        va = vs = None
        Pa, Ps, Aa, As = self._fields_cfg()
        import fieldbasis as _fb
        out = {}
        for key, flag, P_, A_, cnt in (('va', 'FB', Pa, Aa, m), ('vs', 'FC', Ps, As, p)):
            if not self.f[flag]:
                continue
            L = []
            for _ in range(cnt):
                if A_ is not None:
                    mm_ = _fb.NATOM_PAR * A_['n']
                    L.append(_fb.atom_field(self.D.xn, *_fb.split_atoms(z[i:i + mm_], A_['n']), A_['smin'], A_['kmax']))
                    i += mm_
                else:
                    nb = P_.shape[1] if P_ is not None else n
                    xi = z[i:i + nb] + 1j * z[i + nb:i + 2 * nb]; i += 2 * nb
                    L.append((P_ @ xi) if P_ is not None else self.D.smooth_map(xi))
            out[key] = L
        va, vs = out.get('va'), out.get('vs')
        self._atom_z = (z.copy(), i)
        return arow, nvec, K, xa, xs, va, vs

    def zero_grad(self):
        m, p, r, n = self.m, self.p, self.r, self.D.n
        return dict(d=np.zeros(r, complex), e=np.zeros(max(r - 1, 0), complex), f=np.zeros(max(r - 1, 0), complex),
                    Bc=np.zeros((r, p), complex), n=np.zeros((m, r), complex), K=np.zeros((m, p), complex),
                    xa=np.zeros(m), xs=np.zeros(p),
                    va=[np.zeros(n, complex)] * m, vs=[np.zeros(n, complex)] * p)

    def grad(self, g, z=None):
        out = []
        m, p, r = self.m, self.p, self.r
        if self.nth:
            gz = self.zero_grad()
            gc = np.concatenate([np.asarray(g.get(k_, gz[k_]), complex).ravel() for k_ in ('d', 'e', 'f', 'Bc', 'n', 'K')])
            gc = gc[self.free]; sc = self.sc[self.free]
            out += [np.real(gc) * sc, -np.imag(gc) * sc]
        if self.f['B']:
            out += [np.asarray(g.get('xa', np.zeros(m)), float) * self.sx]
        if self.f['C']:
            out += [np.asarray(g.get('xs', np.zeros(p)), float) * self.sx]
        Pa, Ps, Aa, As = self._fields_cfg()
        import fieldbasis as _fb
        zz = np.asarray(z if z is not None else self._atom_z[0], float)
        i = sum(len(o) for o in out)
        for key, flag, P_, A_, cnt in (('va', 'FB', Pa, Aa, m), ('vs', 'FC', Ps, As, p)):
            if not self.f[flag]:
                continue
            pkd = g.get(key + '_packed')
            gl = (g.get(key) or [np.zeros(self.D.n, complex)] * cnt) if pkd is None else None
            for j in range(cnt):
                if pkd is not None:                    # reduced rows (STAGE6 6.29)
                    out += [np.asarray(pkd[j], float)]; i += len(pkd[j])
                    continue
                gv = np.asarray(gl[j], complex)
                if A_ is not None:
                    mm_ = _fb.NATOM_PAR * A_['n']
                    out += [_fb.atom_grad(self.D.xn, *_fb.split_atoms(zz[i:i + mm_], A_['n']), A_['smin'], A_['kmax'], gv)]
                    i += mm_
                else:
                    gx = (P_.T @ gv) if P_ is not None else self.D.smooth_map_adj(gv)
                    out += [gx.real, gx.imag]; i += 2 * len(gx)
        return np.concatenate(out) if out else np.zeros(0)

    def field_jacobians(self, z=None):
        """{'va': [J per actuator], 'vs': [J per sensor]}: complex n x P matrices
        with packed field gradient = Re(G^H J) for a nodal gradient G (STAGE6
        6.29); None for smoothed raw fields (the rows are then lifted)."""
        import fieldbasis as _fb
        Pa, Ps, Aa, As = self._fields_cfg()
        zz = np.asarray(z if z is not None else self._atom_z[0], float)
        i = (2 * self.nth if self.nth else 0) + (self.m if self.f['B'] else 0) + (self.p if self.f['C'] else 0)
        out = {}
        for key, flag, P_, A_, cnt in (('va', 'FB', Pa, Aa, self.m), ('vs', 'FC', Ps, As, self.p)):
            if not self.f[flag]:
                continue
            L = []
            for _ in range(cnt):
                if A_ is not None:
                    mm_ = _fb.NATOM_PAR * A_['n']
                    L.append(_fb.atom_jacobian(self.D.xn, *_fb.split_atoms(zz[i:i + mm_], A_['n']), A_['smin'], A_['kmax']))
                    i += mm_
                elif P_ is not None:
                    L.append(np.hstack([np.conj(P_), 1j * np.conj(P_)]).astype(complex)); i += 2 * P_.shape[1]
                else:
                    L.append(None); i += 2 * self.D.n
            out[key] = L
        return out

    def blocks(self):
        """[(name, slice)] of the design vector: controller entries by kind
        (real and imaginary parts), positions, the fields per port -- for
        diagnostics (sqp_inner exit_info)."""
        out = []; i = 0
        if self.nth:
            fr = np.flatnonzero(self.free); k = self.nth
            for part in ('re', 'im'):
                for nm_, (a_, b_) in self.off.items():
                    idx = [j for j, q in enumerate(fr) if a_ <= q < b_]
                    if idx:
                        out.append((f'{nm_}.{part}', np.array(idx) + i))
                i += k
        if self.f['B']:
            out.append(('xa', np.arange(i, i + self.m))); i += self.m
        if self.f['C']:
            out.append(('xs', np.arange(i, i + self.p))); i += self.p
        Pa, Ps, Aa, As = self._fields_cfg()
        import fieldbasis as _fb
        for key, flag, P_, A_, cnt in (('va', 'FB', Pa, Aa, self.m), ('vs', 'FC', Ps, As, self.p)):
            if not self.f[flag]:
                continue
            for j in range(cnt):
                if A_ is not None:
                    na_ = A_['n']; mm_ = _fb.NATOM_PAR * na_
                    for q, pn in enumerate(('amp.re', 'amp.im', 'centre', 'width', 'wavenumber')):   # split_atoms order
                        out.append((f'{key}{j + 1}.{pn}', np.arange(i + q * na_, i + (q + 1) * na_)))
                    i += mm_
                else:
                    nb = 2 * (P_.shape[1] if P_ is not None else self.D.n)
                    out.append((f'{key}{j + 1}', np.arange(i, i + nb))); i += nb
        return out

    def atom_params(self, z):
        """Lists of the atom parameter vectors (m actuator, p sensor fields) inside z."""
        import fieldbasis as _fb
        Pa, Ps, Aa, As = self._fields_cfg()
        if Aa is None and As is None:
            return None, None
        ma = _fb.NATOM_PAR * Aa['n'] if (self.f['FB'] and Aa is not None) else 0
        ms = _fb.NATOM_PAR * As['n'] if (self.f['FC'] and As is not None) else 0
        z = np.asarray(z, float); i0 = len(z) - self.m * ma - self.p * ms
        A_list = [z[i0 + j * ma:i0 + (j + 1) * ma] for j in range(self.m)] if ma else None
        i1 = i0 + self.m * ma
        S_list = [z[i1 + j * ms:i1 + (j + 1) * ms] for j in range(self.p)] if ms else None
        return A_list, S_list
