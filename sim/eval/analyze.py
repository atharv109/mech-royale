#!/usr/bin/env python
"""sim/eval/analyze.py - Task 5 eval stats from bun sim/run.ts per-match text lines.
Usage: python sim/eval/analyze.py <file> [file2 ...]
Parses lines like:
  seed 12: score 4 place 12 died@98s gun(by 14) kills 3 alive@death 5 hp-left 210
  seed 12: score 4 place 12 died 98s gun(by 14) kills 3   (legacy: no alive/hp)
Falls back to ROW {json} lines when present.
"""
import json, math, re, sys, collections

LINE = re.compile(
    r"seed (\d+): score (\S+) place (\d+) (?:died@(\S+)s|died (\S+)s)?\s*"
    r"([\w]+)(?:\(by (\S+)\))?.*?(?:alive@death (\d+))?.*?$")
ROWL = re.compile(r"^ROW (.*)$")


def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf8", errors="replace"):
            m = ROWL.match(line.strip())
            if m:
                rows.append(json.loads(m.group(1)))
                continue
            m2 = LINE.search(line)
            if m2:
                seed, score, place = int(m2.group(1)), float(m2.group(2)), int(m2.group(3))
                death = m2.group(4) or m2.group(5)
                deathT = None
                if death is not None and death not in ("-", "survived"):
                    try:
                        deathT = float(death)
                    except ValueError:
                        pass
                cause = m2.group(6)
                by = m2.group(7)
                if by == "survived":
                    cause, by = "survived", None
                alive = int(m2.group(8)) if m2.group(8) else None
                if cause == "survived":
                    deathT = None
                    alive = 1
                rows.append({"seed": seed, "score": score, "place": place, "deathT": deathT,
                             "cause": {"weapon": cause, "by": by}, "aliveAtDeath": alive})
    return rows


def report(paths):
    rows = load(paths)
    n = len(rows)
    if n == 0:
        print("no per-match lines found")
        return 0.0
    scores = [r["score"] for r in rows]
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    hist = collections.Counter(r["place"] for r in rows)
    hist_s = " ".join(f"p{p}:{hist[p]}" for p in range(1, 17) if hist[p])
    causes = collections.Counter((r.get("cause") or {}).get("weapon", "survived") for r in rows)
    survivors = causes.get("survived", 0)
    dths = [r["deathT"] for r in rows if r.get("deathT") is not None]
    med = sorted(dths)[len(dths) // 2] if dths else -1
    alive = [r["aliveAtDeath"] for r in rows if r.get("aliveAtDeath")]
    alive_avg = sum(alive) / len(alive) if alive else float("nan")
    print(f"n={n} mean={mean:.2f} SE={se:.2f} medianDeath={med}s "
          f"zoneDeaths={causes.get('zone', 0)} surv={survivors}")
    print(f"places: {hist_s}")
    print(f"causes: {dict(causes)}")
    print(f"alive@death avg: {alive_avg:.1f}")
    return mean


def paired(path_a, path_b):
    """PAIRED difference on shared seeds: mean of per-seed deltas + SE."""
    a = {r["seed"]: r["score"] for r in load([path_a])}
    b = {r["seed"]: r["score"] for r in load([path_b])}
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        print(f"paired: only {len(shared)} shared seeds — nothing to compare")
        return 0.0
    deltas = [a[s] - b[s] for s in shared]
    mean = sum(deltas) / len(deltas)
    var = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
    se = (var / len(deltas)) ** 0.5
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    print(f"paired n={len(shared)}: delta={mean:+.2f} ± {se:.2f} (A-B) wins={wins} losses={losses} ties={len(shared)-wins-losses}")
    return mean


if __name__ == "__main__":
    if len(sys.argv) == 4 and "--pair" == sys.argv[1]:
        paired(sys.argv[2], sys.argv[3])
    else:
        report(sys.argv[1:])
