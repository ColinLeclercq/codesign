#!/usr/bin/env python3
"""slurm_remaining.py -- the driver's --deadline from the job's own end time
(BFS.md 7): the seconds from NOW to the job's EndTime (scontrol), minus
--reserve for what runs after the driver (the figures).  The driver then
starts no round that would not end, with its own reserve for the final
certificate and count (--deadline-final), before that.  Prints 0 (no
deadline) outside a SLURM job or when the end time cannot be read.

    DEADLINE=$(python bfs/slurm_remaining.py --reserve 600)
"""
import os, sys, subprocess, argparse, datetime, re


def remaining(job):
    out = subprocess.run(['scontrol', 'show', 'job', str(job)], capture_output=True, text=True, timeout=30).stdout
    m = re.search(r'\bEndTime=(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)', out)
    if not m:
        return None
    end = datetime.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%S')
    return (end - datetime.datetime.now()).total_seconds()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reserve', type=float, default=600.0)
    a = ap.parse_args()
    job = os.environ.get('SLURM_JOB_ID')
    try:
        r = remaining(job) if job else None
    except Exception:
        r = None
    print(int(max(r - a.reserve, 1.0)) if r is not None else 0)
    if r is not None:
        print(f"slurm_remaining: {r / 3600:.2f} h left in job {job}; driver deadline {max(r - a.reserve, 1.0) / 3600:.2f} h",
              file=sys.stderr)


if __name__ == '__main__':
    main()
