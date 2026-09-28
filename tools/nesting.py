"""
nesting.py — which designs must dominate which, derived rather than listed.

The hand-written table asserted relations that do not hold, and each cost real
time to chase.  The rules are subtle enough to be worth deriving from the case
flags:

HARDWARE.  For each of the actuator and the sensor, the freedom is ordered

    fixed  <  position (B/C)  <  real field (FBR/FCR)  <  complex field (FB/FC)

A field is initialised as a Gaussian and varies freely, so it can represent a
Gaussian centred anywhere: a free field therefore CONTAINS a free position.

CONTROLLER.  This is where the old table was wrong.  A case WITHOUT K keeps the
FIXED LQG-BT compensator; a case WITH K REPLACES it, and the replacements are
not all comparable:

    base   fixed LQG-BT: dynamic, D_c = 0
    gainr  static REAL gain: n = 0, D_c real      <- has a feedthrough LQG-BT lacks
    gain   static COMPLEX gain: n = 0, D_c complex
    sp     strictly proper: d,e,f,n free, D_c = 0
    full   d,e,f,n,D_c all free

so  full > {base, gainr, gain, sp},  sp > base,  gain > gainr,  and

    * gain and sp are INCOMPARABLE (one has D_c but no dynamics, the other the
      reverse);
    * gainr/gain do NOT contain base -- they zero n, so D_c = 0 removes the
      compensator entirely rather than recovering the parent.  That is why
      "warm starting KG+FBR+FCR from FBR+FCR" began at the OPEN LOOP, and why
      a static gain scoring worse than a fixed 7th-order compensator is not a
      convergence failure.
"""
from __future__ import annotations
# --- repository root on sys.path (moved into tools/ by cleanup_archive.py)
import os as _os_root, sys as _sys_root
_sys_root.path.insert(0, _os_root.path.dirname(
    _os_root.path.dirname(_os_root.path.abspath(__file__))))
# --- end root shim

# controller mode -> the modes whose designs it can reproduce
CTRL_CONTAINS = {
    'base':  {'base'},
    'gainr': {'gainr'},
    'gain':  {'gainr', 'gain'},
    'sp':    {'base', 'sp'},
    'full':  {'base', 'gainr', 'gain', 'sp', 'full'},
}


def _mode(f):
    return f.get('ctrl_mode', 'full') if f.get('K') else 'base'


def _hw(f, pos_key, real_key, cplx_key):
    """Freedom level of one channel: 0 fixed, 1 position, 2 real field, 3 complex."""
    if f.get(cplx_key) and not f.get(real_key):
        return 3
    if f.get(cplx_key):
        return 2
    if f.get(pos_key):
        return 1
    return 0


def contains(parse_case, child, parent):
    """Can `child` reproduce every design `parent` can?  (child <= parent in gamma)"""
    try:
        fc, fp = parse_case(child), parse_case(parent)
    except Exception:
        return False
    if _mode(fp) not in CTRL_CONTAINS[_mode(fc)]:
        return False
    if _hw(fc, 'B', 'real_fb', 'FB') < _hw(fp, 'B', 'real_fb', 'FB'):
        return False
    if _hw(fc, 'C', 'real_fc', 'FC') < _hw(fp, 'C', 'real_fc', 'FC'):
        return False
    return True


def is_field_donor(parse_case, donor, case):
    """May `case` usefully START from `donor`'s FIELDS?

    This is WEAKER than `contains` and serves a different purpose.  `contains`
    answers "is gamma_case <= gamma_donor guaranteed", which needs the whole
    design to be reproducible.  This asks only "are the donor's hardware
    profiles a good starting point", which needs only the hardware to be
    compatible -- the compensator is re-seeded separately.

    The distinction matters: KG+FBR+FCR cannot reproduce FBR+FCR's design,
    because KG forces n = 0 while FBR+FCR carries the dynamic LQG-BT
    compensator.  So there is no nesting guarantee.  But FBR+FCR's optimised
    FIELDS are exactly the right place to start from -- optimise the fields
    first, then swap the compensator and re-optimise.  Treating the strict test
    as if it governed warm starting threw that away.
    """
    try:
        fc, fd = parse_case(case), parse_case(donor)
    except Exception:
        return False
    if _hw(fc, 'B', 'real_fb', 'FB') < _hw(fd, 'B', 'real_fb', 'FB'):
        return False
    if _hw(fc, 'C', 'real_fc', 'FC') < _hw(fd, 'C', 'real_fc', 'FC'):
        return False
    return donor != case


def donors(parse_case, case, cases):
    """Field donors that are NOT already nesting parents, richest first."""
    par = set(build(parse_case, cases).get(case, []))
    ds = [d for d in cases
          if d not in par and d != case
          and is_field_donor(parse_case, d, case)
          and not contains(parse_case, case, d)]
    # prefer a donor whose hardware matches exactly: nothing is discarded
    def rank(d):
        fd, fc = parse_case(d), parse_case(case)
        exact = (_hw(fd, 'B', 'real_fb', 'FB') == _hw(fc, 'B', 'real_fb', 'FB')
                 and _hw(fd, 'C', 'real_fc', 'FC') == _hw(fc, 'C', 'real_fc', 'FC'))
        return (0 if exact else 1, d)
    return sorted(ds, key=rank)


def build(parse_case, cases):
    """{child: [immediate parents]} -- transitive edges dropped for readability."""
    rel = {c: [p for p in cases
               if p != c and contains(parse_case, c, p)] for c in cases}
    out = {}
    for c, ps in rel.items():
        imm = [p for p in ps
               if not any(q != p and contains(parse_case, q, p) for q in ps)]
        if imm:
            out[c] = sorted(imm)
    return out


def waves(parse_case, cases):
    """Cases grouped so every parent is finished before its children start."""
    N = build(parse_case, cases)
    done, out = set(), []
    remaining = list(cases)
    while remaining:
        layer = [c for c in remaining
                 if all(p in done for p in N.get(c, []))]
        if not layer:                       # cycle: should not happen
            layer = list(remaining)
        out.append(layer)
        done.update(layer)
        remaining = [c for c in remaining if c not in done]
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fe_codesign import parse_case
    from cases import full, standard, validation
    import os
    _w = os.environ.get('GL_CASES', 'full')
    ORDER = (validation() if _w == 'validation' else
             standard() if _w == 'standard' else full())
    if len(sys.argv) > 2 and sys.argv[1] == '--donors':
        # field donors: a good start, but NO nesting guarantee
        print(' '.join(donors(parse_case, sys.argv[2], ORDER)[:3]))
    elif len(sys.argv) > 2 and sys.argv[1] == '--parents':
        # every parent of one case, for the scheduler to warm start from
        print(' '.join(build(parse_case, ORDER).get(sys.argv[2], [])))
    elif len(sys.argv) > 1 and sys.argv[1] == '--waves':
        for w in waves(parse_case, ORDER):
            print(' '.join(w))
    else:
        N = build(parse_case, ORDER)
        for i, w in enumerate(waves(parse_case, ORDER), 1):
            print(f"wave {i}:")
            for c in w:
                ps = N.get(c, [])
                print(f"  {c:12s} " + (f"<= {', '.join(ps)}" if ps
                                       else "(root -- cold start)"))
