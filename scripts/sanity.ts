/**
 * sanity.ts — Task 2 sanity check on the REAL arena map (protocol.md §7 movement model).
 *
 *   bun scripts/sanity.ts            (uses ../../kit/arena by default)
 *   ARENA_DIR=<dir> bun scripts/sanity.ts
 *
 * Prints: map loader time, walkable %, and findPath times over 20 random walkable pairs
 * (median / mean / max) to verify the 15 ms hard cap holds on the real 1070x724 grid.
 *
 * If map.ts exists and exports loadMap(dir), it is used (and timed) directly. Otherwise a
 * fallback loader builds a contract-shaped GameMap from v2/world.json + v2/terrain.i16:
 * terrain only for cellSurf (cm -> m), cellFeasible = 1 unless a prism (building/stadium/
 * bridge) covers the cell center and stands in the walk band. NOTE: the fallback's
 * feasibility is an approximation of map.ts's cell rasterization; timing (grid size and
 * obstacle density) is what this script verifies.
 */
import { readFileSync } from "node:fs"
import { join } from "node:path"

type GameMap = {
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

const ARENA_DIR_DEFAULT = join(import.meta.dir, "../../kit/arena")
const dir = process.env.ARENA_DIR || ARENA_DIR_DEFAULT

async function main(): Promise<void> {
  // Prefer map.ts's real loader if it exists already.
  let map: GameMap
  let loadMs: number
  let loader: string
  let t0 = performance.now()
  try {
    const mod: { loadMap?: (dir: string) => GameMap | Promise<GameMap> } = await import("../map.ts")
    if (typeof mod.loadMap === "function") {
      map = await mod.loadMap(dir)
      loadMs = performance.now() - t0
      loader = "map.ts loadMap()"
    } else throw new Error("no loadMap export")
  } catch (e) {
    console.error(`map.ts loadMap unavailable or failed, using fallback loader:`, e)
    t0 = performance.now()
    map = fallbackLoad(dir)
    loadMs = performance.now() - t0
    loader = "fallback loader (map.ts not ready or no loadMap)"
  }

  const { nx, ny, cellFeasible } = map
  console.log(`arena dir : ${dir}`)
  console.log(`loader    : ${loader} in ${loadMs.toFixed(1)} ms`)
  console.log(`grid      : ${nx} x ${ny} (step ${map.step} m, origin ${map.minX}, ${map.minY})`)

  let walkable = 0
  for (let i = 0; i < cellFeasible.length; i++) if (cellFeasible[i]) walkable++
  const walkPct = (100 * walkable) / cellFeasible.length
  console.log(`walkable  : ${walkPct.toFixed(1)} % (${walkable}/${cellFeasible.length} cells)`)

  // feasible cell list for pair picking
  const cells: number[] = []
  for (let i = 0; i < cellFeasible.length; i++) if (cellFeasible[i]) cells.push(i)
  const cx = (i: number): number => i % nx
  const cy = (i: number): number => (i - (i % nx)) / nx
  const worldX = (i: number): number => map.minX + cx(i) + 0.5
  const worldY = (i: number): number => map.minY + cy(i) + 0.5

  // deterministic RNG
  let seed = 0x9e3779b9
  const rand = (): number => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let z = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    z = (z + Math.imul(z ^ (z >>> 7), 61 | z)) ^ z
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296
  }

  // 20 random feasible pairs, >= 60 m apart
  const pairs: [number, number][] = []
  let guard = 0
  while (pairs.length < 20 && guard++ < 20000) {
    const a = cells[(rand() * cells.length) | 0]
    const b = cells[(rand() * cells.length) | 0]
    const d = Math.hypot(worldX(a) - worldX(b), worldY(a) - worldY(b))
    if (d >= 60) pairs.push([a, b])
  }

  console.log(`\nA* over ${pairs.length} random walkable pairs (>= 60 m apart, 15 ms cap):`)
  const { findPath } = await import("../nav.ts")
  // Cold first call absorbs the one-time scratch allocation; report it, then warm up before timing.
  const tc = performance.now()
  findPath(map, [worldX(pairs[0][0]), worldY(pairs[0][0])], [worldX(pairs[0][1]), worldY(pairs[0][1])])
  console.log(`  cold first call (scratch allocation): ${(performance.now() - tc).toFixed(1)} ms`)

  const times: number[] = []
  let capped = 0
  let nulls = 0
  const tA = performance.now()
  for (const [a, b] of pairs) {
    const t1 = performance.now()
    const path = findPath(map, [worldX(a), worldY(a)], [worldX(b), worldY(b)])
    const ms = performance.now() - t1
    times.push(ms)
    if (path === null) nulls++
    if (ms > 15) capped++
  }
  const tNav = performance.now() - tA
  times.sort((x, y) => x - y)
  const med = times[times.length >> 1]
  const mean = times.reduce((s, x) => s + x, 0) / times.length
  const max = times[times.length - 1]
  console.log(`  median ${med.toFixed(2)} ms | mean ${mean.toFixed(2)} ms | max ${max.toFixed(2)} ms`)
  console.log(`  calls over 15 ms: ${capped}/${times.length} | null results: ${nulls} (real map regions split by buildings/stadium, or goal inside obstacle)`)
  console.log(`  total nav time for all pairs: ${tNav.toFixed(1)} ms`)
  // nav fires its deadline at 14 ms (16-pop check interval) so measured time stays < 15 ms
  const verdict = mean < 15 && max < 15 ? "OK (mean strictly under 15 ms)" : "REVIEW (cap exceeded)"
  console.log(`  verdict: ${verdict}`)
}

/** fallback GameMap built straight from the arena files (approximate feasibility, see header) */
function fallbackLoad(dir: string): GameMap {
  const world = JSON.parse(readFileSync(join(dir, "v2/world.json"), "utf8"))
  const t = world.terrain as {
    nx: number
    ny: number
    step: number
    minX: number
    minY: number
    scale?: number
  }
  const cap = t.nx * t.ny
  const buf = readFileSync(join(dir, `v2/${(t as { file?: string }).file ?? "terrain.i16"}`))
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) // alignment copy
  const terrain = new Int16Array(ab)

  const cellSurf = new Float64Array(cap)
  for (let i = 0; i < cap; i++) cellSurf[i] = terrain[i] * (t.scale ?? 0.01)

  const cellFeasible = new Uint8Array(cap).fill(1)
  const prisms = (world.prisms ?? []) as {
    kind: string
    base: number
    top: number
    ring: number[]
    holes?: number[][]
  }[]

  // precompute prism bounding boxes; block a cell if a prism covers its center and stands
  // in the walk band (base under head height ~9 m and top above the surface)
  type P = { minX: number; maxX: number; minY: number; maxY: number; pr: (typeof prisms)[number] }
  const boxes: P[] = []
  for (const pr of prisms) {
    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity
    for (let k = 0; k < pr.ring.length; k += 2) {
      minX = Math.min(minX, pr.ring[k])
      maxX = Math.max(maxX, pr.ring[k])
      minY = Math.min(minY, pr.ring[k + 1])
      maxY = Math.max(maxY, pr.ring[k + 1])
    }
    boxes.push({ minX, maxX, minY, maxY, pr })
  }

  const inPoly = (ring: number[], x: number, y: number): boolean => {
    let inside = false
    const n = ring.length / 2
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const xi = ring[2 * i]
      const yi = ring[2 * i + 1]
      const xj = ring[2 * j]
      const yj = ring[2 * j + 1]
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
    }
    return inside
  }

  const headClear = 9.0 // mech height; a prism base above this can be walked under
  for (let cyy = 0; cyy < t.ny; cyy++) {
    for (let cxx = 0; cxx < t.nx; cxx++) {
      const i = cyy * t.nx + cxx
      const x = t.minX + cxx + 0.5
      const y = t.minY + cyy + 0.5
      const s = cellSurf[i]
      for (const b of boxes) {
        if (x < b.minX || x > b.maxX || y < b.minY || y > b.maxY) continue
        const pr = b.pr
        if (pr.top <= s + 0.05) continue // buried / below ground level
        if (pr.base > s + headClear) continue // walk under it
        if (!inPoly(pr.ring, x, y)) continue
        let inHole = false
        if (pr.holes) for (const h of pr.holes) if (inPoly(h, x, y)) inHole = true
        if (inHole) continue
        cellFeasible[i] = 0
        break
      }
    }
  }

  return {
    nx: t.nx,
    ny: t.ny,
    step: t.step,
    minX: t.minX,
    minY: t.minY,
    terrain,
    cellSurf,
    cellFeasible,
    cellPrisms: world.prisms,
  }
}

await main()
