"""
cases.py — the FULL design lattice, generated rather than listed.

The hand-written list held 19 of the 80 available designs, and the gaps showed
up as missing nesting parents: KG+FBR+FCR ought to be bounded by KG+FBR and
KG+FCR, and neither existed.  A partial lattice gives partial guarantees.

Three independent axes:

    compensator   base   fixed LQG-BT reference               -- no K token
                  KGC    static COMPLEX gain      (n = 0)
                  KSP    strictly proper, complex (D_c = 0)
                  K      proper, complex D_c
    actuator      -      fixed real Gaussian, nominal position
                  B      real Gaussian, POSITION free
                  FB     free COMPLEX field
    sensor        -   /  C  /  FC              (likewise)

4 x 3 x 3 = 35 designs.  The real-restricted variants (KG, FBR, FCR) are gone:
the plant is complex, so a real compensator or field is an artificial
constraint.  `standard()` keeps the original 19-case list so older result
directories still tabulate.
"""
from __future__ import annotations

# The plant is COMPLEX (nu = 2+0.4i, gamma = 1-1i), so the LQG and H-infinity
# seeds are complex -- their poles come in no conjugate pairs.  A real-restricted
# compensator or field is therefore an artificial constraint with nothing to
# explain, and the real variants (KG, FBR, FCR) have been dropped.
#
# The Gaussians are the exception: they stay REAL and only their POSITION is
# optimised (B, C).  That is a physically meaningful design -- a single actuator
# somewhere on the domain -- rather than a numerical restriction.
CTRL = ['',      # fixed LQG-BT reference compensator
        'KGC',   # static COMPLEX gain
        'KSP',   # strictly proper (D_c = 0), complex
        'K']     # proper with complex D_c
ACT = ['',       # fixed real Gaussian at the nominal position
       'B',      # real Gaussian, POSITION free
       'FB']     # free COMPLEX field
SEN = ['', 'C', 'FC']


# Actuator and sensor must be treated ALIKE: both fixed, both free positions,
# or both free fields.  Mixing them (B+FC, FB+C) compares a design that is
# richer on one channel than the other, which answers no question we are
# asking -- the study is about POSITION versus FIELD, not about asymmetric
# hardware.
_LEVEL = {'': 0, 'B': 1, 'FB': 2, 'C': 1, 'FC': 2}


def full():
    out = []
    for k in CTRL:
        for a in ACT:
            for s in SEN:
                if _LEVEL[a] != _LEVEL[s] and a and s:
                    continue                 # no B+FC, no FB+C
                toks = [t for t in (k, a, s) if t]
                if toks:
                    out.append('+'.join(toks))
    return out


def validation():
    """A small nested chain for validating the pipeline end to end.

    Seven cases spanning every axis once, with a complete parent chain, so
    `collect.py` exercises the nesting check meaningfully.  Roughly a quarter of
    the full lattice's cost.

        base -> KSP -> K              (compensator: fixed, strictly proper, proper)
        fixed -> B+C -> FB+FC         (hardware: fixed, positions, fields)

    KGC is included because it is the one compensator mode that does NOT
    contain the base case (it zeroes n), and that asymmetry is worth exercising.
    """
    return ['B+C', 'FB+FC', 'KGC', 'KSP', 'KSP+FB+FC', 'K', 'K+FB+FC']


def standard():
    return ['B', 'C', 'B+C',
            'KG', 'KGC', 'KSP', 'K', 'K+B', 'K+C', 'K+B+C',
            'FBR', 'FCR', 'FBR+FCR', 'K+FBR+FCR', 'KG+FBR+FCR',
            'FB', 'FC', 'FB+FC', 'K+FB+FC']


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'full'
    cs = full() if which == 'full' else standard()
    print(f"{len(cs)} cases")
    for c in cs:
        print(' ', c)
