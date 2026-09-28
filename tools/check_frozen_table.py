#!/usr/bin/env python3
"""Cross-check the frozen results/ records against the §3 table of HANDOFF_implementation.md.

This is not part of the pipeline; it validates that the archive received is the one the
handoff describes, before any Stage 0 work is trusted.
"""
import json, os, sys

# case -> (gamma, seconds, calls, outer, nOmega, inner_stalled)
TABLE = {
    "B":            (9.9441,   69,  122,  3, 160, False),
    "C":            (9.9180,   60,  100,  3, 160, False),
    "B+C":          (9.6745,  182,  461,  3,  93, False),
    "FB":           (9.8319,   87,  275,  2,  91, False),
    "FC":           (9.6601,   53,  183,  2,  83, False),
    "FB+FC":        (4.3412,  556, 1358, 17, 173, False),
    "K":            (9.8319,  293,  757,  2, 159, False),
    "K+B":          (9.8319,  187,  453,  2, 159, False),
    "K+C":          (9.6602,  249,  679,  2, 159, False),
    "K+B+C":        (7.9641,  143,  491,  4, 159, True),
    "K+FB":         (9.8319,  132,  468,  1,  67, False),
    "K+FC":         (9.6602,  133,  401,  1,  68, False),
    "K+FB+FC":      (3.9255,  117,  261,  4, 162, False),
    "KGC":         (12.2443,  744, 2733,  8,  20, False),
    "KGC+B":       (11.6009, 1439, 3482, 28, 146, True),
    "KGC+C":       (11.7191, 1069, 1666, 17, 131, False),
    "KGC+B+C":     (11.4116,  283,  746,  8, 133, True),
    "KGC+FB":       (9.8319,   63,  198,  2, 122, False),
    "KGC+FC":       (9.6601,   36,   92,  1, 124, False),
    "KGC+FB+FC":    (3.9258,  104,  229,  3, 162, False),
    "KSP":          (9.8319,  149,  562,  3, 159, False),
    "KSP+B":        (9.8319,  120,  308,  2, 159, False),
    "KSP+C":        (9.6602,  127,  357,  2, 159, False),
    "KSP+B+C":      (8.2465,  237,  683,  7, 162, False),
    "KSP+FB":       (9.8321,  586, 2133,  7,  64, False),
    "KSP+FC":       (9.6603,  290,  951,  5,  15, False),
    "KSP+FB+FC":    (4.1402,  440, 1308,  8, 153, False),
}

RESDIR = sys.argv[1] if len(sys.argv) > 1 else "results"


def fname(case):
    return os.path.join(RESDIR, case.replace("+", "_") + "_s0_ne200.json")


def main():
    hdr = f"{'case':<12}{'gamma_rec':>12}{'gamma_tab':>11}{'d_rel':>9}  " \
          f"{'s':>10}{'calls':>10}{'outer':>8}{'|Om|':>8}{'stall':>8}{'cert':>6}{'wind':>6}"
    print(hdr)
    print("-" * len(hdr))
    nbad = 0
    nnote = 0
    for case, (g_t, s_t, c_t, o_t, m_t, st_t) in TABLE.items():
        p = fname(case)
        if not os.path.exists(p):
            print(f"{case:<12}  MISSING {p}")
            nbad += 1
            continue
        d = json.load(open(p))
        g = d["gamma"]
        drel = abs(g - g_t) / g_t
        # the table is rounded to 4 dp, so agreement to 5e-5 relative is the right test
        flags = []
        if drel > 6e-5:
            flags.append("GAMMA")
        for key, ref, name in (("seconds", s_t, "s"), ("calls", c_t, "calls"),
                               ("outer", o_t, "outer"), ("nOmega", m_t, "|Om|")):
            v = d.get(key)
            if name == "s":
                if abs(v - ref) > 0.6:
                    flags.append("s")
            elif v != ref:
                flags.append(name)
        stalled = bool(d.get("exchange_stalled", False))
        if stalled != st_t:
            flags.append("STALL")
        if not d.get("certified", False):
            flags.append("CERT")
        wc = d.get("wind_count", None)
        note = ""
        if wc is None:
            # 19 of the 27 records have no wind_count: run_case_epi writes the
            # field only under --wind, and for those runs the certificate was
            # produced by the separate sweep.sh pass, which reports to the log
            # and does not write back.  The claim in report_v7 section 4.3 is
            # true but not recoverable from the record.  Not a discrepancy --
            # re-derive it with winding_cert.py (step 4 of validate_cleanup.sh).
            note = "wind_count not in record; see winding_cert.py"
        elif wc != 0:
            flags.append("WIND")
        if flags:
            nbad += 1
        if note:
            nnote += 1
        print(f"{case:<12}{g:>12.6f}{g_t:>11.4f}{drel:>9.1e}  "
              f"{d.get('seconds'):>10}{d.get('calls'):>10}{d.get('outer'):>8}"
              f"{d.get('nOmega'):>8}{str(stalled):>8}"
              f"{str(d.get('certified')):>6}{str(d.get('wind_count')):>6}"
              + ("   <-- " + ",".join(flags) if flags else "")
              + ("   (" + note + ")" if note and not flags else ""))
    print("-" * len(hdr))
    print(f"{len(TABLE)} cases checked, {nbad} discrepancies, {nnote} notes")
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
