#!/usr/bin/env python
"""sim/eval/signature.py - Task 9 behavioral instrumentation from ROW lines.
Usage: python sim/eval/signature.py <file> [file2 ...]
Prints per variant (avg over seeds): drones killed, crates broken, cubes@90s,
specials@90s, drops captured, median death time.
"""
import json, sys


def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf8", errors="replace"):
            if line.startswith("ROW "):
                rows.append(json.loads(line[4:]))
    return rows


def report(paths):
    rows = load(paths)
    n = len(rows)
    if n == 0:
        print("no ROW lines")
        return
    avg = lambda k: sum(r.get(k, 0) for r in rows) / n
    dths = sorted(r["deathT"] for r in rows if r.get("deathT") is not None)
    med = dths[len(dths) // 2] if dths else -1
    surv = sum(1 for r in rows if r.get("survived"))
    print(f"n={n} drones={avg('sigDrones'):.1f} crates={avg('sigCrates'):.1f} "
          f"cubes@90={avg('sigCubesAt90'):.1f} specials@90={avg('sigSpecialsAt90'):.1f} "
          f"drops={avg('sigDrops'):.1f} medianDeath={med}s surv={surv}")


if __name__ == "__main__":
    report(sys.argv[1:])
