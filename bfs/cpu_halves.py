#!/usr/bin/env python3
"""cpu_halves.py -- split this job's CPUs into two halves, one per NUMA side
(bfs/slurm/pair.slurm, 20261010n).

    python bfs/cpu_halves.py                 two lines: CPU list of half a, of half b
    python bfs/cpu_halves.py --describe      what the halves are (NUMA nodes, sizes)
    mpiexec -n N python bfs/cpu_halves.py --check LIST
                                             every rank runs inside LIST, N ranks

Why: sator cancels (as root, exit 0:54) a second job of the same user started
on a node (774342, 774343, 774346, 2026-10-15), so two 48-rank arms share a
96-core node only inside one batch job.  The two launches must then not run
on the same cores: each is started under taskset with one half, and MPICH's
Hydra, with the fork launcher (no srun step), passes the mask on to its ranks.
The halves follow the NUMA nodes (sysfs), so each arm's memory is local.
"""
import glob, os, re, sys


def _expand(s):
    out = []
    for part in s.strip().split(','):
        if not part:
            continue
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def _numa():
    nodes = []
    for d in glob.glob('/sys/devices/system/node/node[0-9]*'):
        try:
            nodes.append((int(re.sub(r'\D', '', os.path.basename(d))), _expand(open(d + '/cpulist').read())))
        except (OSError, ValueError):
            pass
    return sorted(nodes)


def halves():
    allowed = sorted(os.sched_getaffinity(0))
    order, where = [], {}
    for k, cpus in _numa():
        for c in cpus:
            if c in allowed and c not in where:
                order.append(c); where[c] = k
    order += [c for c in allowed if c not in where]
    h = len(order) // 2
    return order[:h], order[h:2 * h], where


def _fmt(cpus):
    cpus = sorted(cpus); out = []; i = 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}"); i = j + 1
    return ','.join(out)


def main(args):
    if args[:1] == ['--check']:
        want = set(_expand(args[1]))
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mine = sorted(os.sched_getaffinity(0))
        allm = comm.gather(mine, root=0)
        if comm.rank == 0:
            bad = [r for r, m in enumerate(allm) if not set(m) <= want]
            used = sorted(set().union(*map(set, allm)))
            print(f"cpu check: {comm.size} ranks, CPUs {_fmt(used)}"
                  + (f"; *** ranks {bad[:8]} run outside {args[1]}" if bad else f" inside {_fmt(want)}"), flush=True)
            return 1 if bad else 0
        return 0
    a, b, where = halves()
    if args[:1] == ['--describe']:
        for tag, h in (('a', a), ('b', b)):
            print(f"half {tag}: {len(h)} CPUs {_fmt(h)} on NUMA node(s) {sorted(set(where.get(c, -1) for c in h))}")
        return 0
    print(_fmt(a)); print(_fmt(b))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
