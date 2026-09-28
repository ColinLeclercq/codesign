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

    def z0(self):
        z = []
        m = self.f.get('ctrl_mode', 'full')
        if self.f['K'] and m in ('full', 'sp'):
            th = np.concatenate([self.d0, self.e0, self.f0,
                                 self.nvec0, [self.K0]])
            k = self.nth if m == 'full' else self.nth - 1   # drop Dc if s.proper
            z += [th[:k].real / self.sc[:k], th[:k].imag / self.sc[:k]]
        elif self.f['K'] and m == 'gain':          # static complex gain
            z += [np.array([self.K0.real / self.sK, self.K0.imag / self.sK])]
        elif self.f['K'] and m == 'gainr':         # static real gain
            z += [np.array([self.K0.real / self.sK])]
        if self.f['B']:  z += [np.array([self.xa0 / self.sx])]
        if self.f['C'] and not self.sym_pos:
            z += [np.array([self.xs0 / self.sx])]
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
        if self.f['B']: xa = float(z[i] * self.sx); i += 1
        if self.f['C']:
            if self.sym_pos:
                xs = -xa
            else:
                xs = float(z[i] * self.sx); i += 1
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
            out += [np.real(gc[:k])*self.sc[:k], -np.imag(gc[:k])*self.sc[:k]]
        elif self.f['K'] and m == 'gain':
            out += [np.array([np.real(g['K'])*self.sK, -np.imag(g['K'])*self.sK])]
        elif self.f['K'] and m == 'gainr':
            out += [np.array([np.real(g['K'])*self.sK])]
        if self.f['B']:
            gxa = g['xa'] - g['xs'] if self.sym_pos else g['xa']
            out += [np.array([gxa * self.sx])]
        if self.f['C'] and not self.sym_pos:
            out += [np.array([g['xs']*self.sx])]
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
                out += [_fb.atom_grad(self.D.xn, *_fb.split_atoms(zz[i:i + m_], Aa['n']), Aa['smin'], Aa['kmax'],
                                      np.asarray(g['va'], complex))]; i += m_
            if self.f['FC']:
                m_ = _fb.NATOM_PAR * As['n']
                out += [_fb.atom_grad(self.D.xn, *_fb.split_atoms(zz[i:i + m_], As['n']), As['smin'], As['kmax'],
                                      np.asarray(g['vs'], complex))]; i += m_
            return np.concatenate(out)
        if self.f['FB']:
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
            gx = (Ps.T @ np.asarray(g['vs'])) if Ps is not None else self.D.smooth_map_adj(g['vs'])
            out += [gx.real] if self.f.get('real_fc') else [gx.real, gx.imag]
        return np.concatenate(out)

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

