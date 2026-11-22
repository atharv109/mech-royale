#!/usr/bin/env python
"""sim/eval/signature.py - Task 14 kill-rate signatures from ROW lines."""
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
    visSec = sum(r.get("sigVisTicks", 0) for r in rows) * 0.1
    shots = sum(r.get("sigShots", 0) for r in rows)
    hits = sum(r.get("sigHits", 0) for r in rows)
    rate = shots / visSec if visSec > 0 else 0
    hitpct = 100 * hits / shots if shots > 0 else 0
    dths = sorted(r["deathT"] for r in rows if r.get("deathT") is not None)
    med = dths[len(dths) // 2] if dths else -1
    surv = sum(1 for r in rows if r.get("survived"))
    print(f"n={n} kills={avg('kills'):.2f} shots/sVis={rate:.2f} hitRate={hitpct:.0f}% "
          f"cubes@90={avg('sigCubesAt90'):.1f} maxAct={max(r.get('botMaxActMs', 0) for r in rows):.1f}ms")
    print(f"   shots={shots:.0f} drops={avg('sigDrops'):.1f} drones={avg('sigDrones'):.1f} crates={avg('sigCrates'):.1f} "
          f"cubes@150={avg('sigCubesAt150'):.1f} upperMedianDeath={med}s surv={surv}")


if __name__ == "__main__":
    report(sys.argv[1:])
