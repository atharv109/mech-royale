#!/usr/bin/env python
"""sim/eval/signature.py - Task 9/10 behavioral instrumentation from ROW lines.
Usage: python sim/eval/signature.py <file> [file2 ...]
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
    took = {k: round(sum((r.get("botTook") or {}).get(k, 0) for r in rows) / n, 2) for k in ("laser", "lightning", "mortar")}
    fail = {k: round(sum((r.get("botFail") or {}).get(k, 0) for r in rows) / n, 2) for k in ("gone", "range", "have", "full", "busy")}
    seen, sd = {}, {}
    for k in ("laser", "lightning", "mortar"):
        cnt = sum((r.get("botSeen") or {}).get(k, 0) for r in rows)
        seen[k] = round(cnt / n, 1)
        sd[k] = round(sum((r.get("botSeenDist") or {}).get(k, 0) for r in rows) / cnt) if cnt else 0
    print(f"n={n} dodges={avg('botDodges'):.1f} retreats={avg('botRetreats'):.1f} "
          f"cover%={100 * avg('sigCoverFrac'):.0f} maxAct={max(r.get('botMaxActMs', 0) for r in rows):.1f}ms")
    print(f"   seen={seen} seenDist={sd} walkedTicks={avg('botWalked'):.0f} took={took} fail={fail}")
    print(f"   drones={avg('sigDrones'):.1f} crates={avg('sigCrates'):.1f} "
          f"cubes@90={avg('sigCubesAt90'):.1f} specials@90={avg('sigSpecialsAt90'):.1f} "
          f"cubesColl={avg('sigCubesCollected'):.1f} drops={avg('sigDrops'):.1f} "
          f"medianDeath={med}s surv={surv}")


if __name__ == "__main__":
    report(sys.argv[1:])
