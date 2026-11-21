/** Tests for map.ts against the real kit arena. Run: bun test tests/map.test.ts
 *  Arena dir: ARENA_DIR env override, else the kit path on this machine. */
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { expect, test } from "bun:test"
import { feasible, ground, intervals, loadMap, los, surface, type GameMap, type Interval, type Vec3 } from "../map.ts"

const ARENA = process.env.ARENA_DIR ?? "C:/Users/athar/OneDrive/Desktop/mech royale/kit/arena"
const world = JSON.parse(readFileSync(join(ARENA, "v2", "world.json"), "utf8"))
const PRISMS: { kind: string; base: number; top: number; ring: number[]; holes: number[][] }[] =
  world.prisms.map((p: any) => ({ ...p, holes: p.holes ?? [] }))

let cached: GameMap | undefined
const getMap = () => (cached ??= loadMap(ARENA))

// --- test-side geometry oracle, independent of map.ts's rasterization ---

/** Even-odd point in polygon (same rule the protocol implies: inside ring = within the footprint). */
function pip(x: number, y: number, flat: number[]): boolean {
  let inside = false
  const m = flat.length / 2
  for (let k = 0, l = m - 1; k < m; l = k++) {
    const x1 = flat[l * 2], y1 = flat[l * 2 + 1]
    const x2 = flat[k * 2], y2 = flat[k * 2 + 1]
    if (y1 > y !== y2 > y && x < ((x2 - x1) * (y - y1)) / (y2 - y1) + x1) inside = !inside
  }
  return inside
}
const insidePrism = (p: { ring: number[]; holes: number[][] }, x: number, y: number) =>
  pip(x, y, p.ring) && !p.holes.some((h) => pip(x, y, h))

/** Does a sample point hit terrain or any world prism, according to raw world.json + ground()? */
function sampleBlocked(map: GameMap, x: number, y: number, z: number): boolean {
  if (z < ground(map, x, y)) return true
  for (const p of PRISMS) if (p.base <= z && z <= p.top && insidePrism(p, x, y)) return true
  return false
}

/** Segment check for the "clear above the roof" assertion (1 m samples like los()). */
function segmentClear(map: GameMap, a: Vec3, b: Vec3): boolean {
  const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2]
  const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy, dz)))
  for (let k = 0; k <= steps; k++) {
    const t = k / steps
    if (sampleBlocked(map, a[0] + dx * t, a[1] + dy * t, a[2] + dz * t)) return false
  }
  return true
}

/**
 * Cell-center-row crossings: runs of >= 4 CONSECUTIVE covered cells (tested at the cell centers
 * themselves, the same points loadMap rasterizes) in one grid row. A horizontal segment through
 * y = minY + j + 0.5 spanning the run must cross those cells, and with sample spacing < 1 m it
 * must land a sample inside them -> guaranteed blocked at z between base and top.
 */
function buildingCrossings(map: GameMap, count: number): { p: (typeof PRISMS)[number]; j: number; iFirst: number; iLast: number; zmid: number }[] {
  const out: ReturnType<typeof buildingCrossings> = []
  for (const p of PRISMS) {
    if (p.kind !== "building" || p.top - p.base < 3) continue
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
    for (let k = 0; k < p.ring.length; k += 2) {
      x0 = Math.min(x0, p.ring[k])
      x1 = Math.max(x1, p.ring[k])
      y0 = Math.min(y0, p.ring[k + 1])
      y1 = Math.max(y1, p.ring[k + 1])
    }
    const iS = Math.ceil(x0 - map.minX - 0.5), iE = Math.floor(x1 - map.minX - 0.5)
    for (let j = Math.ceil(y0 - map.minY - 0.5); j <= Math.floor(y1 - map.minY - 0.5); j++) {
      const cy = map.minY + j + 0.5
      let i = iS
      while (i <= iE) {
        if (!insidePrism(p, map.minX + i + 0.5, cy)) {
          i++
          continue
        }
        let e = i
        while (e + 1 <= iE && insidePrism(p, map.minX + e + 1.5, cy)) e++
        if (e - i >= 3) {
          out.push({ p, j, iFirst: i, iLast: e, zmid: (p.base + p.top) / 2 })
          if (out.length >= count) return out
        }
        i = e + 1
      }
    }
  }
  return out
}

test("ground: exact at grid nodes, bilinear midpoint between nodes", () => {
  const map = getMap()
  const { nx, ny, minX, minY, terrain } = map
  // an interior node that exists on both axes' interpolation paths
  const i = Math.floor(nx / 2), j = Math.floor(ny / 2)
  const x = minX + i * map.step, y = minY + j * map.step
  expect(ground(map, x, y)).toBeCloseTo(terrain[j * nx + i] / 100, 9)
  // midpoint on the vertical edge between node (i, j) and (i, j+1): average of the two
  expect(ground(map, x, y + map.step / 2)).toBeCloseTo((terrain[j * nx + i] + terrain[(j + 1) * nx + i]) / 200, 9)
  // midpoint on the horizontal edge between node (i, j) and (i+1, j)
  expect(ground(map, x + map.step / 2, y)).toBeCloseTo((terrain[j * nx + i] + terrain[j * nx + i + 1]) / 200, 9)
  // an off-grid point extrapolates by edge clamping: matches the border node value
  expect(ground(map, minX - 50, minY - 50)).toBeCloseTo(terrain[0] / 100, 9)
})

test("surface: open-ground cell surface equals terrain ground at center", () => {
  const map = getMap()
  const { nx, ny, minX, minY } = map
  // find a prism-free cell so surface == ground there
  for (let j = 20; j < ny - 20; j++) {
    for (let i = 20; i < nx - 20; i++) {
      const x = minX + i + 0.5, y = minY + j + 0.5
      const ivs = intervals(map, x, y)
      if (ivs && ivs.length > 0) continue
      // cellSurf is Float32 by contract: toBeCloseTo at 1e-6 m (5 mm) rounding
      expect(surface(map, x, y)).toBeCloseTo(ground(map, x, y), 6)
      return
    }
  }
  throw new Error("no prism-free cell found")
})

test("los: blocked across a building body, clear above its roof", () => {
  const map = getMap()
  const cands = buildingCrossings(map, 60)
  expect(cands.length).toBeGreaterThan(0)
  const open = (x: number, y: number, z: number) => z > ground(map, x, y) + 0.3 && (intervals(map, x, y)?.length ?? 0) === 0
  const cy = (c: (typeof cands)[number]) => map.minY + c.j + 0.5

  // blocked: horizontal segment through the run of covered cells at mid height, endpoints open
  let blocked: (typeof cands)[number] | undefined
  for (const c of cands) {
    const a: Vec3 = [map.minX + c.iFirst + 0.5 - 0.8, cy(c), c.zmid]
    const b: Vec3 = [map.minX + c.iLast + 0.5 + 0.83, cy(c), c.zmid] // +0.83: sample spacing < 1 m, run cells must be sampled
    if (open(a[0], a[1], a[2]) && open(b[0], b[1], b[2])) {
      blocked = c
      break
    }
  }
  expect(blocked, "no building with open endpoints at mid height").toBeDefined()
  if (!blocked) return
  const a: Vec3 = [map.minX + blocked.iFirst + 0.5 - 0.8, cy(blocked), blocked.zmid]
  const b: Vec3 = [map.minX + blocked.iLast + 0.5 + 0.83, cy(blocked), blocked.zmid]
  expect(los(map, a, b)).toBe(false)

  // clear: same crossing raised above the roof, confirmed by the test-side oracle first
  let clear: (typeof cands)[number] | undefined
  for (const c of cands) {
    const z = c.p.top + 5
    const a2: Vec3 = [map.minX + c.iFirst + 0.5 - 1.5, cy(c), z]
    const b2: Vec3 = [map.minX + c.iLast + 0.5 + 1.5, cy(c), z]
    if (open(a2[0], a2[1], a2[2]) && open(b2[0], b2[1], b2[2]) && segmentClear(map, a2, b2)) {
      clear = c
      break
    }
  }
  expect(clear, "no building with a verified clear above-roof crossing").toBeDefined()
  if (!clear) return
  const z = clear.p.top + 5
  expect(los(map, [map.minX + clear.iFirst + 0.5 - 1.5, cy(clear), z], [map.minX + clear.iLast + 0.5 + 1.5, cy(clear), z])).toBe(true)
})

test("los: segment dipping below terrain is blocked", () => {
  const map = getMap()
  const { nx, ny, minX, minY, terrain } = map
  // open flat spot: no prisms within 7 cells, terrain within the 15x15 window nearly level,
  // and the above-ground variant verified clear so the blocked case is not vacuous
  for (let j = 15; j < ny - 15; j += 4) {
    for (let i = 15; i < nx - 15; i += 4) {
      let hmin = Infinity, hmax = -Infinity
      for (let dj = -7; dj <= 7; dj++)
        for (let di = -7; di <= 7; di++) {
          const h = terrain[(j + dj) * nx + (i + di)]
          hmin = Math.min(hmin, h)
          hmax = Math.max(hmax, h)
        }
      if (hmax - hmin > 100) continue // need >2 m dip under a level surface
      const x = minX + i + 0.5, y = minY + j + 0.5
      let prismFree = true
      for (let dj = -8; dj <= 8 && prismFree; dj++)
        for (let di = -8; di <= 8; di++) {
          const ivs = intervals(map, minX + i + di + 0.5, minY + j + dj + 0.5)
          if (ivs && ivs.length > 0) {
            prismFree = false
            break
          }
        }
      if (!prismFree) continue
      const gz = ground(map, x, y)
      expect(feasible(map, x, y)).toBe(true)
      const below: Vec3 = [x - 5, y, gz - 2]
      const below2: Vec3 = [x + 5, y, gz - 2]
      if (!segmentClear(map, [x - 5, y, gz + 4], [x + 5, y, gz + 4])) continue // only assert on verified spots
      expect(los(map, [x - 5, y, gz + 4], [x + 5, y, gz + 4])).toBe(true)
      expect(los(map, below, below2)).toBe(false)
      return
    }
  }
  throw new Error("no open flat spot found for the below-terrain test")
})

test("intervals: prism interior contains its base/top; open ground empty; off-grid null", () => {
  const map = getMap()
  const cands = buildingCrossings(map, 10)
  expect(cands.length).toBeGreaterThan(0)
  const c = cands[0]
  const px = map.minX + Math.floor((c.iFirst + c.iLast) / 2) + 0.5, py = map.minY + c.j + 0.5
  const ivs: Interval[] | null = intervals(map, px, py)
  expect(ivs, "covered cell must have intervals").not.toBeNull()
  expect(ivs!.some((v) => v.base === c.p.base && v.top === c.p.top)).toBe(true)

  // open ground: a cell center verified prism-free by the oracle has no intervals (null or empty)
  let open: [number, number] | undefined
  for (let j = Math.floor(map.ny / 2); j < map.ny && !open; j++)
    for (let i = Math.floor(map.nx / 2); i < map.nx; i++) {
      const x = map.minX + i + 0.5, y = map.minY + j + 0.5
      if (!PRISMS.some((p) => insidePrism(p, x, y))) {
        open = [x, y]
        break
      }
    }
  expect(open, "no prism-free cell found").toBeDefined()
  const ivs2 = intervals(map, open![0], open![1])
  expect(ivs2 === null || ivs2.length === 0).toBe(true)

  // off-grid
  expect(intervals(map, map.minX - 100, map.minY - 100)).toBeNull()
  expect(intervals(map, map.minX + map.nx + 50, map.minY + 10)).toBeNull()
})

/** Cells (i, j) whose centers the oracle puts inside prism p. */
function prismCells(map: GameMap, p: (typeof PRISMS)[number]): [number, number][] {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
  for (let k = 0; k < p.ring.length; k += 2) {
    x0 = Math.min(x0, p.ring[k])
    x1 = Math.max(x1, p.ring[k])
    y0 = Math.min(y0, p.ring[k + 1])
    y1 = Math.max(y1, p.ring[k + 1])
  }
  const out: [number, number][] = []
  for (let j = Math.ceil(y0 - map.minY - 0.5); j <= Math.floor(y1 - map.minY - 0.5); j++)
    for (let i = Math.ceil(x0 - map.minX - 0.5); i <= Math.floor(x1 - map.minX - 0.5); i++)
      if (insidePrism(p, map.minX + i + 0.5, map.minY + j + 0.5)) out.push([i, j])
  return out
}
const center = (map: GameMap, i: number, j: number): [number, number] => [map.minX + i + 0.5, map.minY + j + 0.5]
const coveredAt = (x: number, y: number) => PRISMS.some((p) => insidePrism(p, x, y))

test("surface: roof cell surface equals the building's top", () => {
  const map = getMap()
  for (const p of PRISMS) {
    if (p.kind !== "building") continue
    for (const [i, j] of prismCells(map, p)) {
      const [x, y] = center(map, i, j)
      // only this prism covers the cell, and it is not overhead (base < ground + 9)
      if (PRISMS.filter((q) => insidePrism(q, x, y)).length !== 1 || p.base >= ground(map, x, y) + 9) continue
      expect(surface(map, x, y)).toBeCloseTo(p.top, 4) // Float32 cellSurf
      return
    }
  }
  throw new Error("no single-prism roof cell found")
})

test("surface: bridge #0 deck stands on its top; bridge #4 is walked under (ground)", () => {
  const map = getMap()
  const deck = PRISMS[0], over = PRISMS[4]
  expect(deck.kind).toBe("bridge")
  expect(over.kind).toBe("bridge")
  const deckCells = prismCells(map, deck)
  expect(deckCells.length).toBeGreaterThan(0)
  for (const [i, j] of deckCells) expect(surface(map, ...center(map, i, j))).toBeCloseTo(deck.top, 3) // ~0.29
  const overCells = prismCells(map, over)
  expect(overCells.length).toBeGreaterThan(0)
  for (const [i, j] of overCells) {
    const [x, y] = center(map, i, j)
    expect(surface(map, x, y)).toBeCloseTo(ground(map, x, y), 4)
  }
})

test("feasible: wall face clearance is exactly the mech radius (offsets 1-2 blocked, 3 open)", () => {
  const map = getMap()
  for (const p of PRISMS) {
    if (p.kind !== "building") continue
    for (const [i, j] of prismCells(map, p)) {
      if (i < 8 || j < 8 || i >= map.nx - 8 || j >= map.ny - 8) continue
      const [wx, wy] = center(map, i, j)
      if (p.base >= ground(map, wx, wy) + 9 || p.top - ground(map, wx, wy) < 3) continue
      // straight west-facing face: wall cells at rows j-2..j+2 in column i, all open to the west
      let ok = true
      for (let dj = -2; dj <= 2 && ok; dj++) ok = insidePrism(p, ...center(map, i, j + dj))
      // open, smooth (no ledge) ground in columns i-6..i-1, rows j-3..j+3
      for (let dj = -3; dj <= 3 && ok; dj++)
        for (let di = 1; di <= 6 && ok; di++) {
          const [x, y] = center(map, i - di, j + dj)
          if (coveredAt(x, y)) ok = false
          else if (di > 1 && Math.abs(ground(map, x, y) - ground(map, x + 1, y)) > 1) ok = false
          else if (dj > -3 && Math.abs(ground(map, x, y) - ground(map, x, y - 1)) > 1) ok = false
        }
      if (!ok) continue
      const at = (di: number) => feasible(map, ...center(map, i - di, j))
      expect(at(1)).toBe(false)
      expect(at(2)).toBe(false)
      expect(at(3)).toBe(true)
      return
    }
  }
  throw new Error("no clean wall face found")
})

test("feasible: a 30-45 degree terrain slope with no ledge nearby is walkable", () => {
  const map = getMap()
  const g = (i: number, j: number) => ground(map, ...center(map, i, j))
  const lo = Math.tan((30 * Math.PI) / 180), hi = Math.tan((45 * Math.PI) / 180)
  for (let j = 10; j < map.ny - 10; j++)
    for (let i = 10; i < map.nx - 10; i++) {
      const s = g(i, j)
      const slope = Math.max(Math.abs(g(i + 1, j) - s), Math.abs(g(i - 1, j) - s), Math.abs(g(i, j + 1) - s), Math.abs(g(i, j - 1) - s))
      if (slope < lo || slope > hi) continue
      // no prism and no ledge (4-neighbour rise > 1.2) within Chebyshev 3 (covers the radius-2 window's neighbours)
      let ok = true, uphill = -Infinity
      for (let dj = -3; dj <= 3 && ok; dj++)
        for (let di = -3; di <= 3 && ok; di++) {
          const h = g(i + di, j + dj)
          uphill = Math.max(uphill, h)
          if (coveredAt(...center(map, i + di, j + dj))) ok = false
          else if (Math.abs(h - g(i + di + 1, j + dj)) > 1.2 || Math.abs(h - g(i + di, j + dj + 1)) > 1.2) ok = false
        }
      // pick a cell the old all-surface R=3 rule would have blocked (uphill > s + 1.2): a real regression case
      if (!ok || uphill <= s + 1.2) continue
      expect(feasible(map, ...center(map, i, j))).toBe(true)
      return
    }
  throw new Error("no ledge-free 30-45 degree slope found")
})

test("feasible: centers within 2.5 m of world.bounds are not walkable (map edge is a wall)", () => {
  const map = getMap()
  const b = world.bounds
  const y = (b.minY + b.maxY) / 2
  expect(feasible(map, b.minX + 1.0, y)).toBe(false)
  expect(feasible(map, b.maxX - 1.0, y)).toBe(false)
  expect(feasible(map, b.minX - 3, y)).toBe(false) // grid extends past bounds
  let edgeFeasible = 0
  for (let j = 0; j < map.ny; j++)
    for (let i = 0; i < map.nx; i++) {
      const [cx, cy] = center(map, i, j)
      if (Math.min(cx - b.minX, b.maxX - cx, cy - b.minY, b.maxY - cy) < 2.5 && map.cellFeasible[j * map.nx + i]) edgeFeasible++
    }
  expect(edgeFeasible).toBe(0)
})

test("loadMap on the real arena finishes well under 3000 ms", () => {
  const t0 = performance.now()
  const m = loadMap(ARENA)
  const ms = performance.now() - t0
  console.log(`loadMap: ${ms.toFixed(0)} ms (${world.prisms.length} prisms, ${m.nx}x${m.ny})`)
  expect(ms).toBeLessThan(3000)
  // sanity on grid shape and sizes
  expect(m.nx).toBe(world.terrain.nx)
  expect(m.ny).toBe(world.terrain.ny)
  expect(m.terrain.length).toBe(m.nx * m.ny)
  expect(m.cellSurf.length).toBe(m.nx * m.ny)
  expect(m.cellFeasible.length).toBe(m.nx * m.ny)
  expect(m.cellPrisms.length).toBe(m.nx * m.ny)
})
