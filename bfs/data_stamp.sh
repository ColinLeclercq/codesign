#!/bin/bash
# data_stamp.sh -- are the plant files in bfs/data built by THIS code? (BFS.md 7)
#
#   bash bfs/data_stamp.sh check coarse|fx    exit 0 if bfs/data/STAMP_<which> matches the generators
#   bash bfs/data_stamp.sh write coarse|fx    record the generators' hashes (after a build)
#
# Since bfs/data is shared by every code version (bfs/home.sh), a file built by
# an older version could otherwise be used silently.  The stamp is the sha256 of
# the scripts that make the files: bfs/mesh_coarse.py (coarse only),
# bfs/ops_dolfinx.py (the operators) and bfs/plant_cert.py (the certified range
# and poles).  A cosmetic edit of one of them costs one rebuild.
set -eo pipefail
cd "$(dirname "$0")/.."
act=$1; which=$2
case "$which" in
    coarse) gens=(bfs/mesh_coarse.py bfs/ops_dolfinx.py bfs/plant_cert.py) ;;
    fx)     gens=(bfs/ops_dolfinx.py bfs/plant_cert.py) ;;
    *) echo "usage: data_stamp.sh check|write coarse|fx"; exit 2 ;;
esac
want=$(sha256sum "${gens[@]}" | awk '{print $1}' | sha256sum | cut -c1-16)
f=bfs/data/STAMP_$which
case "$act" in
    check) [ -f "$f" ] && [ "$(cat "$f")" = "$want" ] ;;
    write) mkdir -p bfs/data; echo "$want" > "$f.$$" && mv "$f.$$" "$f" ;;
    *) echo "usage: data_stamp.sh check|write coarse|fx"; exit 2 ;;
esac
