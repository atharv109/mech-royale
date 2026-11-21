/**
 * nav.test.ts — Task 2 navigation tests. Uses only synthetic GameMap-shaped plain objects
 * (map.ts may not exist yet; the contract pins {nx,ny,step,minX,minY,terrain,cellSurf,
 * cellFeasible,cellPrisms}). Run with: bun test tests/nav.test.ts
 */
import { expect, test } from "bun:test"
import { findPath, localSteer, Navigator, StuckMonitor, type Vec2 } from "../nav.ts"

// minimal local GameMap-shaped stub (contract-shaped; cast where nav expects GameMap)
type SynthMap = {
  nx: number
  ny: number
  step: number
  minX: number
  minY: number
  terrain: Int16Array
  cellSurf: Float64Array
  cellFeasible: Uint8Array
  cellPrisms: unknown
}

function makeMap(
  nx: number,
  ny: number,
  opts?: {
    blocked?: (cx: number, cy: number) => boolean
    surf?: (cx: number, cy: number) => number
    minX?: number
    minY?: number
  },
): SynthMap {
  const cap = nx * ny
  const cellFeasible = new Uint8Array(cap).fill(1)
  const cellSurf = new Float64Array(cap)
  for (let cy = 0; cy < ny; cy++) {
    for (let cx = 0; cx < nx; cx++) {
      const i = cy * nx + cx
      if (opts?.surf) cellSurf[i] = opts.surf(cx, cy)
      if (opts?.blocked?.(cx, cy)) cellFeasible[i] = 0
    }
  }
  return {
    nx,
    ny,
    step: 1,
    minX: opts?.minX ?? 0,
    minY: opts?.minY ?? 0,
    terrain: new Int16Array(0),
    cellSurf,
    cellFeasible,
    cellPrisms: [],
  }
}

const asMap = (m: SynthMap) => m as unknown as Parameters<typeof findPath>[0]

const cellOf = (m: SynthMap, p: Vec2): [number, number] => [
  Math.round(p[0] - m.minX - 0.5),
  Math.round(p[1] - m.minY - 0.5),
]

/** structural checks: connectivity, 8-connectivity, feasibility, no corner cutting */
function checkStructure(m: SynthMap, path: Vec2[] | null): Vec2[] {
  expect(path).not.toBeNull()
  const p = path!
  expect(p.length).toBeGreaterThan(0)
  for (let i = 0; i < p.length; i++) {
    const [cx, cy] = cellOf(m, p[i])
    expect(cx).toBeGreaterThanOrEqual(0)
    expect(cy).toBeGreaterThanOrEqual(0)
    expect(cx).toBeLessThan(m.nx)
    expect(cy).toBeLessThan(m.ny)
    expect(m.cellFeasible[cy * m.nx + cx]).toBe(1)
    if (i > 0) {
      const [px, py] = cellOf(m, p[i - 1])
      const dx = cx - px
      const dy = cy - py
      expect(Math.abs(dx) <= 1).toBe(true)
      expect(Math.abs(dy) <= 1).toBe(true)
      expect(Math.abs(dx) + Math.abs(dy)).toBeGreaterThan(0) // no self-loops
      if (dx !== 0 && dy !== 0) {
        // no corner cutting: both orthogonals feasible
        expect(m.cellFeasible[py * m.nx + cx]).toBe(1)
        expect(m.cellFeasible[cy * m.nx + px]).toBe(1)
      }
    }
  }
  return p
}

/** structural checks + exact endpoints (full path found) */
function checkPath(m: SynthMap, path: Vec2[] | null, from: Vec2, to: Vec2): Vec2[] {
  const p = checkStructure(m, path)
  expect(p[0][0]).toBeCloseTo(from[0], 5)
  expect(p[0][1]).toBeCloseTo(from[1], 5)
  const last = p[p.length - 1]
  expect(last[0]).toBeCloseTo(to[0], 5)
  expect(last[1]).toBeCloseTo(to[1], 5)
  return p
}

/** structural checks + start endpoint + partial-path progress: end cell closer to goal than start cell */
function checkPartial(m: SynthMap, path: Vec2[] | null, from: Vec2, to: Vec2): Vec2[] {
  const p = checkStructure(m, path)
  expect(p[0][0]).toBeCloseTo(from[0], 5)
  expect(p[0][1]).toBeCloseTo(from[1], 5)
  const s = cellOf(m, from)
  const g = cellOf(m, to)
  const e = cellOf(m, p[p.length - 1])
  const d = (c: [number, number]) => Math.hypot(c[0] - g[0], c[1] - g[1])
  expect(d(e)).toBeLessThan(d(s))
  return p
}

function pathLength(path: Vec2[]): number {
  let len = 0
  for (let i = 1; i < path.length; i++) len += Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
  return len
}

// --- 1. straight empty line on a flat map -----------------------------------
test("straight line: path length ~= distance, 8-connected", () => {
  const m = makeMap(40, 40)
  // diagonal
  const from: Vec2 = [2.5, 2.5]
  const to: Vec2 = [30.5, 30.5]
  const p = findPath(asMap(m), from, to)!
  checkPath(m, p, from, to)
  const straight = Math.hypot(to[0] - from[0], to[1] - from[1])
  expect(pathLength(p)).toBeGreaterThan(straight * 0.99)
  expect(pathLength(p)).toBeLessThan(straight * 1.05)
  // pure horizontal
  const p2 = findPath(asMap(m), [5.5, 10.5], [30.5, 10.5])!
  checkPath(m, p2, [5.5, 10.5], [30.5, 10.5])
  expect(pathLength(p2)).toBeCloseTo(25, 5)
  // shortest in cells: max(|dx|,|dy|) + 1 waypoints for a monotone path
  expect(p2.length).toBe(26)
})

// --- 2. wall row: route around, never cross, never cut corners --------------
test("wall: path routes around without crossing or corner cutting", () => {
  // vertical wall segment at x = 20, y in [12..27], raised surf (5 m) to double-encode the wall
  // (surf jump only on the wall segment, so the gaps above/below stay walkable)
  const m = makeMap(40, 40, {
    blocked: (cx, cy) => cx === 20 && cy >= 12 && cy <= 27,
    surf: (cx, cy) => (cx === 20 && cy >= 12 && cy <= 27 ? 5 : 0),
  })
  const from: Vec2 = [10.5, 19.5]
  const to: Vec2 = [30.5, 19.5]
  const p = findPath(asMap(m), from, to)
  checkPath(m, p, from, to)
  // detour is longer than the straight line
  expect(pathLength(p!)).toBeGreaterThan(20.5)
  // no wall cell is ever touched (checkPath already asserts cellFeasible == 1 per step)

  // explicit corner-cut case: both orthogonals around a diagonal blocked
  // open (4,4) and (5,5); block (5,4) and (4,5): direct diagonal is forbidden
  const m2 = makeMap(10, 10, {
    blocked: (cx, cy) => (cx === 5 && cy === 4) || (cx === 4 && cy === 5),
  })
  const p2 = findPath(asMap(m2), [4.5, 4.5], [5.5, 5.5])
  checkPath(m2, p2, [4.5, 4.5], [5.5, 5.5])
  // must not be the direct diagonal move — it goes around
  expect(p2!.length).toBeGreaterThan(2)

  // blocked goal -> snapped to an adjacent feasible cell (see the endpoint-snapping test)
  const ps = checkStructure(m, findPath(asMap(m), [10.5, 19.5], [20.5, 19.5]))
  const se = cellOf(m, ps[ps.length - 1])
  expect(Math.max(Math.abs(se[0] - 20), Math.abs(se[1] - 19))).toBe(1)
  // unreachable region (fully sealed) -> null
  const m3 = makeMap(20, 10, { blocked: (cx, cy) => cx === 10 })
  expect(findPath(asMap(m3), [5.5, 5.5], [15.5, 5.5])).toBeNull()
})

// --- 3. long detour: connected, no wall crossing, makes progress -------------
// A detour this long can exceed the 15 ms cap on the left half of the map; the contract says
// cap-out returns the best partial path (non-null), so we assert partial semantics: connected
// steps, feasible cells, and the last cell strictly closer to the goal than the start cell.
test("long detour returns a valid connected path (full or cap-out partial)", () => {
  const N = 400
  const m = makeMap(N, N, {
    blocked: (cx, cy) => cx === 200 && cy < 380, // wall with a gap at the top
    surf: (cx) => (cx === 200 ? 4 : 0),
  })
  const from: Vec2 = [10.5, 10.5]
  const to: Vec2 = [389.5, 389.5]
  const p = checkPartial(m, findPath(asMap(m), from, to), from, to)
  const end = cellOf(m, p[p.length - 1])
  if (end[0] === 389 && end[1] === 389) {
    // full path: it must actually detour (straight line is ~536 m, the detour ~740 m)
    expect(pathLength(p)).toBeGreaterThan(Math.hypot(379, 379) * 1.3)
  }
})

test("huge map: path (cap-out partial) is connected, progresses toward goal, honors the cap", () => {
  const N = 1200
  const m = makeMap(N, N, {
    blocked: (cx, cy) => cx === 600 && cy < N - 3, // full wall, tiny gap at the top
    surf: (cx) => (cx === 600 ? 4 : 0),
  })
  new Navigator(asMap(m)) // warm scratch buffers so the timed call pays no allocation
  const from: Vec2 = [300.5, 300.5]
  const to: Vec2 = [1190.5, 1190.5]
  const t0 = performance.now()
  const p = findPath(asMap(m), from, to)
  const ms = performance.now() - t0
  checkPartial(m, p, from, to)
  expect(ms).toBeLessThan(40) // 14 ms deadline + generous CI slack
})

// --- 4. localSteer ----------------------------------------------------------
test("localSteer returns dir when the 3 m probe is clear", () => {
  const m = makeMap(30, 30)
  const d = localSteer(asMap(m), [10.5, 15.5], [1, 0])
  expect(d[0]).toBeCloseTo(1, 6)
  expect(d[1]).toBeCloseTo(0, 6)
})

test("localSteer rotates around a blocked cell (+30 blocked, -30 open)", () => {
  // pos (10.5,15.5), east. straight probe cell ~(13,15); +30 probe = pos + 3*m (0.866, 0.5) -> cell (13,17)
  const m = makeMap(30, 30, {
    blocked: (cx, cy) => (cx === 13 && cy === 15) || (cx === 13 && cy >= 17 && cy <= 19),
  })
  const d = localSteer(asMap(m), [10.5, 15.5], [1, 0])
  const ang = (Math.atan2(d[1], d[0]) * 180) / Math.PI
  expect(ang).toBeCloseTo(-30, 0)
})

test("localSteer passes a 30-deg-open case (straight blocked, +30 clear)", () => {
  const m = makeMap(30, 30, {
    blocked: (cx, cy) => cy === 15 && (cx === 13 || cx === 14),
  })
  const d = localSteer(asMap(m), [10.5, 15.5], [1, 0])
  const ang = (Math.atan2(d[1], d[0]) * 180) / Math.PI
  expect(ang).toBeCloseTo(30, 0)
})

test("localSteer falls back to dir when nothing within +/-150 deg is feasible", () => {
  // solid block covering the pos and every 3 m probe (all 10 rotated directions + straight):
  // cells x in [6..14], y in [10..20]
  const m = makeMap(30, 30, {
    blocked: (cx, cy) => cx >= 6 && cx <= 14 && cy >= 10 && cy <= 20,
  })
  const d = localSteer(asMap(m), [10.5, 15.5], [1, 0])
  expect(d[0]).toBeCloseTo(1, 6)
  expect(d[1]).toBeCloseTo(0, 6)
  // and the unit length is preserved
  expect(Math.hypot(d[0], d[1])).toBeCloseTo(1, 6)
})

// --- 5. StuckMonitor --------------------------------------------------------
// Update times use 0.25 s steps: exactly representable in binary, so accumulated stuck time
// crosses the 1 s and 0.5 s thresholds exactly (no float drift).
test("StuckMonitor: stuck -> burst, then perp once, then none, window resets", () => {
  const mon = new StuckMonitor(0)
  const seq: string[] = []
  for (let i = 1; i <= 11; i++) seq.push(mon.update(i * 0.25, 0.2, true))
  // t = 1.0 (idx 3): 1 s of stuck -> burst; t = 1.5 (idx 5): 0.5 s after burst -> perp
  expect(seq.indexOf("burst")).toBe(3)
  expect(seq.filter((s) => s === "perp").length).toBe(1)
  expect(seq.indexOf("perp")).toBe(5)
  expect(seq.slice(0, 3).every((s) => s === "none")).toBe(true)
  expect(seq[4]).toBe("none")
  // window was reset at the perp: another full stuck second -> new burst at t = 2.5 (idx 9)
  expect(seq.filter((s) => s === "burst").length).toBe(2)
  expect(seq[9]).toBe("burst")
  expect(seq[6]).toBe("none")
  expect(seq[7]).toBe("none")
  expect(seq[8]).toBe("none")
})

test("StuckMonitor: moving fast is always none", () => {
  const mon = new StuckMonitor(0)
  for (let i = 1; i <= 40; i++) expect(mon.update(i * 0.25, 6, true)).toBe("none")
})

test("StuckMonitor: not commanding a move never triggers, and recovery cancels the perp", () => {
  const mon = new StuckMonitor(0)
  for (let i = 1; i <= 12; i++) expect(mon.update(i * 0.25, 0.2, false)).toBe("none")
  // burst at t = 1.0, then the jump+boost burst recovers before the 0.5 s window ends:
  // no perp, window reset, no further triggers
  const mon2 = new StuckMonitor(0)
  let r = "none"
  for (let i = 1; i <= 4; i++) r = mon2.update(i * 0.25, 0.2, true)
  expect(r).toBe("burst")
  for (let i = 5; i <= 12; i++) expect(mon2.update(i * 0.25, 8, true)).toBe("none")
})

// --- Navigator (contract behavior checks) -----------------------------------
test("Navigator follows a path to the goal and reports atGoal", () => {
  const m = makeMap(40, 40)
  const nav = new Navigator(asMap(m))
  const goal: Vec2 = [20.5, 20.5]
  let pos: Vec2 = [2.5, 2.5]
  let steps = 0
  for (;;) {
    const { dir, atGoal } = nav.update(pos, goal, steps * 0.1)
    if (atGoal) break
    expect(Math.hypot(dir[0], dir[1])).toBeCloseTo(1, 6)
    pos = [pos[0] + dir[0] * 5 * 0.1, pos[1] + dir[1] * 5 * 0.1] // 5 m/s * 0.1 s
    if (++steps > 200) throw new Error("never reached the goal")
  }
  expect(Math.hypot(pos[0] - goal[0], pos[1] - goal[1])).toBeLessThanOrEqual(1.5)
})

test("Navigator replans immediately when the goal moves > 8 m", () => {
  const m = makeMap(40, 40)
  const nav = new Navigator(asMap(m))
  // first goal due north: dir north
  const first = nav.update([5.5, 5.5], [5.5, 30.5], 0)
  expect(first.atGoal).toBe(false)
  expect(first.dir[0]).toBeCloseTo(0, 5)
  expect(first.dir[1]).toBeCloseTo(1, 5)
  // t = 0.1: cadence (1.5 s) not elapsed, but the goal jumped ~35 m to due east -> only a real
  // replan can turn the dir east (the old north path would keep it north)
  const second = nav.update([5.5, 5.6], [30.5, 5.5], 0.1)
  expect(second.dir[0]).toBeGreaterThan(0.99)
  expect(Math.abs(second.dir[1])).toBeLessThan(0.1)
})

test("Navigator walks around a wall: every visited pos is feasible, goal reached", () => {
  const m = makeMap(40, 40, {
    blocked: (cx, cy) => cx === 20 && cy >= 12 && cy <= 27,
    surf: (cx, cy) => (cx === 20 && cy >= 12 && cy <= 27 ? 5 : 0),
  })
  const nav = new Navigator(asMap(m))
  const goal: Vec2 = [30.5, 19.5]
  let pos: Vec2 = [10.5, 19.5]
  let steps = 0
  for (;;) {
    // every commanded stance is feasible: check pos at the START of each tick (post-move pos can
    // land a fraction into the wall's inflation, then localSteer corrects on the next tick)
    const [scx, scy] = [Math.floor(pos[0]), Math.floor(pos[1])]
    expect(m.cellFeasible[scy * m.nx + scx]).toBe(1)
    const { dir, atGoal } = nav.update(pos, goal, steps * 0.1)
    if (atGoal) break
    expect(Math.hypot(dir[0], dir[1])).toBeCloseTo(1, 6)
    pos = [pos[0] + dir[0] * 0.5, pos[1] + dir[1] * 0.5] // 5 m/s * 0.1 s
    if (++steps > 300) throw new Error("never reached the goal")
  }
  expect(Math.hypot(pos[0] - goal[0], pos[1] - goal[1])).toBeLessThanOrEqual(2.5)
})

test("Navigator with no path returns a locally steered dir (not [0,0]) when straight is blocked", () => {
  // thick sealed wall x in [10..12]: goal on the far side is unreachable -> findPath null
  const m = makeMap(30, 20, { blocked: (cx) => cx >= 10 && cx <= 12 })
  expect(findPath(asMap(m), [8.5, 5.5], [20.5, 5.5])).toBeNull()
  const nav = new Navigator(asMap(m))
  // pos (8.5,5.5), goal due east: straight probe (11.5,5.5) and +/-30, +/-60 probes land in the
  // wall; +90 (north, probe (8.5,8.5)) is the first clear rotation
  const { dir, atGoal } = nav.update([8.5, 5.5], [20.5, 5.5], 0)
  expect(atGoal).toBe(false)
  expect(Math.hypot(dir[0], dir[1])).toBeCloseTo(1, 6)
  expect(dir[0]).toBeCloseTo(0, 5)
  expect(dir[1]).toBeCloseTo(1, 5)
})

test("findPath snaps infeasible endpoints to the nearest feasible cell", () => {
  // building block x in [15..24], y in [15..24]; goal cell (20,20): first feasible Chebyshev
  // ring is r = 5 (x or y = 25; <= 15) -> path ends on a feasible cell next to the block
  const m = makeMap(40, 40, { blocked: (cx, cy) => cx >= 15 && cx <= 24 && cy >= 15 && cy <= 24 })
  const p = checkStructure(m, findPath(asMap(m), [2.5, 2.5], [20.5, 20.5]))
  expect(p[0][0]).toBeCloseTo(2.5, 5)
  expect(p[0][1]).toBeCloseTo(2.5, 5)
  const e = cellOf(m, p[p.length - 1])
  expect(Math.max(Math.abs(e[0] - 20), Math.abs(e[1] - 20))).toBe(5)
  // infeasible start (mech slid onto the block's corner cell (15,15)): snapped, path starts adjacent
  const p2 = checkStructure(m, findPath(asMap(m), [15.5, 15.5], [2.5, 2.5]))
  const s2 = cellOf(m, p2[0])
  expect(Math.max(Math.abs(s2[0] - 15), Math.abs(s2[1] - 15))).toBe(1)
  // start deeper than 6 cells inside: no snap -> null
  const big = makeMap(60, 60, { blocked: (cx, cy) => cx >= 10 && cx < 50 && cy >= 10 && cy < 50 })
  expect(findPath(asMap(big), [30.5, 30.5], [2.5, 2.5])).toBeNull()
  // goal deeper than 15 cells inside: no snap -> null
  expect(findPath(asMap(big), [2.5, 2.5], [30.5, 30.5])).toBeNull()
})

test("Navigator heads for the snapped goal when the goal cell is infeasible", () => {
  const m = makeMap(40, 40, { blocked: (cx, cy) => cy === 20 && cx >= 30 }) // goal row blocked
  const nav = new Navigator(asMap(m))
  const { dir, atGoal } = nav.update([10.5, 20.5], [35.5, 20.5], 0)
  expect(atGoal).toBe(false)
  expect(dir[0]).toBeGreaterThan(0.95) // eastward along the path to the snapped goal
})

test("Navigator: a null plan respects the 1.5 s cadence (no per-tick replans)", () => {
  // sealed wall at x = 10: no path. pos (5.5,1.5), goal (15.5,1.5) -> no-path dir due east.
  const m = makeMap(20, 12, { blocked: (cx) => cx === 10 })
  const nav = new Navigator(asMap(m))
  const goal: Vec2 = [15.5, 1.5]
  const d0 = nav.update([5.5, 1.5], goal, 0).dir
  expect(d0[0]).toBeCloseTo(1, 5)
  // open a gap at the top (10,11): a replan would now find a path that heads north-east
  m.cellFeasible[11 * m.nx + 10] = 1
  // t = 0.1 .. 1.4: within cadence -> still the no-path dir (a per-tick replan would turn north)
  for (let i = 1; i <= 14; i++) {
    const d = nav.update([5.5, 1.5], goal, i * 0.1).dir
    expect(d[0]).toBeCloseTo(1, 5)
    expect(d[1]).toBeCloseTo(0, 5)
  }
  // t = 1.5: cadence replan finds the gap path -> dir gains a clear north component
  const d1 = nav.update([5.5, 1.5], goal, 1.5).dir
  expect(d1[1]).toBeGreaterThan(0.3)
})

test("Navigator rejects a non-1 m grid", () => {
  const m = makeMap(10, 10)
  m.step = 2
  expect(() => new Navigator(asMap(m))).toThrow()
})

test("StuckMonitor: a long observation gap does not trigger an instant burst (dt clamp)", () => {
  const mon = new StuckMonitor(0)
  // 5 s gap: unclamped this would accumulate 5 s stuck -> burst; clamped it counts 0.25 s
  expect(mon.update(5, 0.2, true)).toBe("none")
  expect(mon.update(5.25, 0.2, true)).toBe("none")
  expect(mon.update(5.5, 0.2, true)).toBe("none")
  expect(mon.update(5.75, 0.2, true)).toBe("burst") // 4 x 0.25 s = 1 s of real stuck time
})

test("StuckMonitor: a zero Vec2 passed as commandingMove counts as not commanding", () => {
  const mon = new StuckMonitor(0)
  const zero = [0, 0] as unknown as boolean
  for (let i = 1; i <= 12; i++) expect(mon.update(i * 0.25, 0, zero)).toBe("none")
})
