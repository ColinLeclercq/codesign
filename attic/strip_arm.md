# The strip arm of the lattice SQP (removed 2026-10-16)

From 2026-10-09 (20261009a) to 20261010n the lattice oracle's SQP covered the strip
[-eps - 0.25, -eps] at every iterate whose rows the QP read (`--lattice-strip on`, the
default; `STRIP=1` in the scripts), every eigenvalue in it a stability row. The user
removed it on 2026-10-16, after the 20261010n A/B (BFS.md 8, LESSONS.md 19-20): it bought
no stability (no refusal, veto or FATAL in either arm of any pair), cost about 45 % more
evaluations a round (123-135 calls against 83-95), and the level set beside the next
iteration (`--lattice-async`) runs only without it. The code left is the no-strip path,
bit for bit (the e2e test of the removal, BFS.md 8).

Kept: the strip SEARCH, as a diagnostic (`LatticeOracle._strip`, `evaluate(strip=True)`:
`tools/check_lattice.py`, and `sqp_inner.LAT_STRIP` for the Evaluator in tools; the SQP
refuses to run with it). The full code is at ac20964 (the last commit with the arm).

What was removed (small, so recorded here rather than moved as a file):

- `run_case_epi.py`: `--lattice-strip on|off` (`sqp_inner.LAT_STRIP = (a.lattice_strip == 'on')`)
  and its lines in the oracle and round printouts (`strip up to N shifts, M incomplete`).
  The driver now refuses the flag with a pointer to BFS.md 8.
- `sqp_inner.py`:
  - `Evaluator.lat_strip`, toggled off around every line-search trial (`_ctrl_restore` and
    `ev_fn(zn)`: a trial's admissibility is the line's certificate) and the output key
    `lat_strip` the solve read;
  - the accepted trial re-evaluated with its strip and, in the same waves, the level set:
    ```
    if not en.get('lat_strip', True):
        _lev_at = (zn.copy(), max(tn, en['smax']))
        ev_fn.level_req = ((_lev_at[1] * (1.0 + refine_tol), float(refine_grid[0]), float(refine_grid[-1]))
                           if refine_tol is not None else None)
        ev_fn.level_watch = (refine_tol, en.get('follow_rows') or []) if refine_tol is not None else None
        try:
            en = ev_fn(zn)
        finally:
            ev_fn.level_req = None; ev_fn.level_watch = None
    ```
    and, at the refine, the level read back from it
    (`if _lev_at is not None and np.array_equal(_lev_at[0], z) and _lev <= _lev_at[1] * (1.0 + 1e-9): _lev = _lev_at[1]`);
    `_lev_at` in the look-ahead's saved state;
  - `not LAT_STRIP` in the condition of the asynchronous level set;
  - `strip`, `strip_shifts_max`, `strip_incomplete` in `stats['lattice']`.
- `lattice.py`: `evaluate(level=..., watch=...)` (the level set in the evaluation's waves,
  kept in `_level_memo` for the `level()` / `level_start()` that asked for it, and in the
  look-ahead state); `evaluate`'s `strip` now defaults to False.
- Scripts: `STRIP` (`--lattice-strip`), the `_nostrip` directory suffix and `ASYNC=1`'s
  `_async` (`testplant.slurm`, `codesign.slurm`, `run_bfs.sh`; `ASYNC=0` and `_sync` now).
