#!/bin/bash
# check-research-integration: the mechanical minimum of call/0005.
# 1. every run directory under research/ is referenced from a plan/ or
#    call/ record (no orphan research);
# 2. research/CONSOLIDATION.md's material map names every run.
# Exit non-zero on any failure. Run with the verify battery.
set -u
cd "$(dirname "$0")/.."
fail=0
for d in research/*/; do
    run=$(basename "$d")
    if ! grep -rq "research/$run" plan/ call/ AGENTS.md 2>/dev/null; then
        echo "ORPHAN RESEARCH: research/$run referenced by no plan or call record"
        fail=1
    fi
    if ! grep -q "$run" research/CONSOLIDATION.md 2>/dev/null; then
        echo "UNMAPPED RUN: research/$run missing from CONSOLIDATION.md material map"
        fail=1
    fi
done
[ -f research/CONSOLIDATION.md ] || {
    echo "MISSING: research/CONSOLIDATION.md"; fail=1; }
[ "$fail" -eq 0 ] && echo "research integration: clean"
exit $fail
