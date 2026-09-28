"""
patch_pygranso.py — make pyGRANSO 1.2.0 importable under numpy >= 2.

pyGRANSO was written for numpy 1.x and has two import-time statements that
numpy 2 removed:
    from numpy.core.numeric import Inf          -> from numpy import inf as Inf
    from numpy.core.defchararray import isspace -> from numpy.char import isspace
Both replacements are valid on numpy 1.x and 2.x, so this patch is safe either
way and is idempotent (re-running it does nothing).

Run once, in the environment where pyGRANSO is installed:
    python patch_pygranso.py
"""
import importlib.util
import pathlib

# Replacements chosen to be valid on BOTH numpy 1.x and 2.x.
_ISSPACE = "from numpy import char as _npchar; isspace = _npchar.isspace"
REPLACEMENTS = [
    ("from numpy.core.numeric import Inf", "from numpy import inf as Inf"),
    ("from numpy.core.defchararray import isspace", _ISSPACE),
    # also normalise a half-patched state (numpy-2-only form) to the safe one:
    ("from numpy.char import isspace", _ISSPACE),
]


def main():
    spec = importlib.util.find_spec("pygranso")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("pygranso not found in this environment.")
    pkg = pathlib.Path(spec.submodule_search_locations[0])
    print(f"pygranso package dir: {pkg}")

    changed = []
    for py in pkg.rglob("*.py"):
        text = py.read_text()
        new = text
        for old, rep in REPLACEMENTS:
            new = new.replace(old, rep)
        if new != text:
            py.write_text(new)
            changed.append(str(py.relative_to(pkg)))

    # drop stale bytecode so the edited source is used
    for pyc in pkg.rglob("*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass

    if changed:
        print("patched files:")
        for c in changed:
            print("  -", c)
    else:
        print("nothing to patch (already numpy-2 compatible).")


if __name__ == "__main__":
    main()


# 2026-09-22: pyGRANSO's private/tensor2vec.py imports an unused GUI constant
# (`from tkinter.messagebox import NO`); sandboxes without tkinter fail at
# import.  Replace it with the constant it provides.
def _patch_tkinter():
    import os, pygranso
    f = os.path.join(os.path.dirname(pygranso.__file__), 'private', 'tensor2vec.py')
    t = open(f).read()
    if 'from tkinter.messagebox import NO' in t:
        open(f, 'w').write(t.replace('from tkinter.messagebox import NO', 'NO = "no"   # was a tkinter import'))
        print("  - private/tensor2vec.py (tkinter import removed)")

try:
    _patch_tkinter()
except Exception as _e:
    print("  tkinter patch skipped:", _e)
