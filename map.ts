/**
 * Map module for the Mech Royale bot. Loads the static map (protocol.md §11) into a 1 m grid
 * once at startup, then answers queries fast enough for the 50 ms tick budget.
 *
 *   import { loadMap, ground, surface, feasible, intervals, los } from "./map.ts"
 *   const map = loadMap(hello.map.dir)          // synchronous, well under 3 s on the kit map
 *   los(map, self.pos, enemy.pos)               // buildings/bridges/terrain block, trees do not (§8)
 *
 * Grid conventions (§11): cell (i, j) covers [minX + i, minX + i + 1) x [minY + j, minY + j + 1),
 * row-major index j * nx + i. Terrain vertices sit on the 1 m lattice at (minX + i, minY + j)
 * and are sampled bilinearly, clamped to the grid edge outside it.
 */

import { readFileSync } from "node:fs"
import { join } from "node:path"

export type Vec3 = [number, number, number] // x east, y north, z up (m). Yaw: 0=+x, CCW+. Pitch: up+
export interface Interval {
  base: number
  top: number
}
export interface GameMap {
  readonly nx: number
  readonly ny: number
  readonly step: number
  readonly minX: number
  readonly minY: number
  /** raw cm heights, row-major index j*nx+i */
  terrain: Int16Array
  /** standing surface s per cell (m) */
  cellSurf: Float32Array
  /** 1 = walkable */
  cellFeasible: Uint8Array
  /** prism intervals whose footprint (ring minus holes) covers the cell; null where none */
  cellPrisms: (Interval[] | null)[]
}

export const CELL_SIZE: number = 1

/** A prism with base >= terrain + 9 m clears the mech (9 m tall): walk under it; it still blocks shots. */
const OVERHEAD_CLEARANCE = 9
/** Mech radius 2.5 m on 1 m center-rasterized cells: a ledge cell at Chebyshev offset <= 2 has a
 *  face < 2.5 m from this cell's center, so the center cannot stand here. */
const FEASIBILITY_RADIUS = 2
/** Mech radius (§5): a center closer than this to the world edge (a wall, §7) cannot stand. */
const MECH_RADIUS = 2.5

type WorldPrism = { kind: string; base: number; top: number; ring: number[]; holes?: number[][] }

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v)

export function loadMap(dir: string, stepUp = 1.2): GameMap {
  const world = JSON.parse(readFileSync(join(dir, "v2", "world.json"), "utf8"))
  const t = world.terrain
  const nx: number = t.nx
  const ny: number = t.ny
  const step: number = t.step
  const minX: number = t.minX
  const minY: number = t.minY
  const n = nx * ny

  const buf = readFileSync(join(dir, "v2", t.file ?? "terrain.i16"))
  if (buf.length < n * 2) throw new Error(`terrain.i16 too small: ${buf.length} bytes, expected ${n * 2}`)
  const terrain = decodeInt16LE(buf, n)

  // Standing surface starts at the bilinear terrain height (m) at each cell center.
  const cellSurf = new Float32Array(n)
  for (let j = 0; j < ny; j++) {
    const cy = minY + (j + 0.5) * step
    for (let i = 0; i < nx; i++) {
      cellSurf[j * nx + i] = terrainAtCM(terrain, nx, ny, minX, minY, step, minX + (i + 0.5) * step, cy) * 0.01
    }
  }

  // Rasterize prisms onto the grid: a cell is covered iff its CENTER is inside the ring and
  // outside every hole. Covered prisms raise the surface unless they are overhead (walk under),
  // and are always stored in cellPrisms because they block LOS/shots.
  const cellPrisms: (Interval[] | null)[] = new Array(n).fill(null)
  const ground0 = cellSurf.slice() // terrain-only ground per cell, for the overhead test
  for (const p of world.prisms as WorldPrism[]) {
    if (p.kind !== "building" && p.kind !== "stadium" && p.kind !== "bridge") continue
    const ring = p.ring
    const holes = p.holes ?? []
    let px0 = Infinity, px1 = -Infinity, py0 = Infinity, py1 = -Infinity
    for (let k = 0; k < ring.length; k += 2) {
      if (ring[k] < px0) px0 = ring[k]
      if (ring[k] > px1) px1 = ring[k]
      if (ring[k + 1] < py0) py0 = ring[k + 1]
      if (ring[k + 1] > py1) py1 = ring[k + 1]
    }
    const i0 = clamp(Math.floor((px0 - minX) / step - 0.5), 0, nx - 1)
    const i1 = clamp(Math.ceil((px1 - minX) / step - 0.5), 0, nx - 1)
    const j0 = clamp(Math.floor((py0 - minY) / step - 0.5), 0, ny - 1)
    const j1 = clamp(Math.ceil((py1 - minY) / step - 0.5), 0, ny - 1)
    for (let j = j0; j <= j1; j++) {
      const cy = minY + (j + 0.5) * step
      for (let i = i0; i <= i1; i++) {
        const cx = minX + (i + 0.5) * step
        if (!pointInPoly(cx, cy, ring)) continue
        if (holes.some((h) => pointInPoly(cx, cy, h))) continue
        const k = j * nx + i
        ;(cellPrisms[k] ??= []).push({ base: p.base, top: p.top })
        if (p.base < ground0[k] + OVERHEAD_CLEARANCE && p.top > cellSurf[k]) cellSurf[k] = p.top
      }
    }
  }

  // Feasibility = mech-radius clearance from LEDGES (a cell whose surface rises > stepUp above a
  // 4-neighbour: building walls, deck edges, terrain cliffs). Smooth slopes make no ledges; the
  // per-step slope/step-up limit is applied by nav on each A* edge (rise <= stepUp per 1 m cell ~ 50°).
  // Note: cellSurf is Float32 while prism tops are float64; any future equality-style comparison
  // of the two (e.g. top > cellSurf + eps) needs eps >= 1e-3 to avoid rounding false hits.
  const ledge = new Float32Array(n).fill(-Infinity)
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i
      const s = cellSurf[k]
      if ((i > 0 && s - cellSurf[k - 1] > stepUp) || (i < nx - 1 && s - cellSurf[k + 1] > stepUp) ||
          (j > 0 && s - cellSurf[k - nx] > stepUp) || (j < ny - 1 && s - cellSurf[k + nx] > stepUp)) ledge[k] = s
    }
  }
  const cellFeasible = new Uint8Array(n)
  const rowMax = new Float32Array(n)
  const R = FEASIBILITY_RADIUS
  for (let j = 0; j < ny; j++) {
    const off = j * nx
    for (let i = 0; i < nx; i++) {
      let m = -Infinity
      for (let k = Math.max(0, i - R), hi = Math.min(nx - 1, i + R); k <= hi; k++) {
        const v = ledge[off + k]
        if (v > m) m = v
      }
      rowMax[off + i] = m
    }
  }
  for (let j = 0; j < ny; j++) {
    const off = j * nx
    for (let i = 0; i < nx; i++) {
      let m = -Infinity
      for (let k = Math.max(0, j - R), hi = Math.min(ny - 1, j + R); k <= hi; k++) {
        const v = rowMax[k * nx + i]
        if (v > m) m = v
      }
      cellFeasible[off + i] = m <= cellSurf[off + i] + stepUp ? 1 : 0
    }
  }

  // The map edge is a wall (§7): a center within MECH_RADIUS of world.bounds cannot stand there.
  const b = world.bounds
  for (let j = 0; j < ny; j++) {
    const cy = minY + (j + 0.5) * step
    for (let i = 0; i < nx; i++) {
      const cx = minX + (i + 0.5) * step
      if (Math.min(cx - b.minX, b.maxX - cx, cy - b.minY, b.maxY - cy) < MECH_RADIUS) cellFeasible[j * nx + i] = 0
    }
  }

  return { nx, ny, step, minX, minY, terrain, cellSurf, cellFeasible, cellPrisms }
}

/** terrain.i16 is little-endian int16 cm. Typed arrays are platform-endian; swap bytes if not. */
function decodeInt16LE(buf: Buffer, count: number): Int16Array {
  if (new Uint8Array(new Int16Array([1]).buffer)[0] === 1) return new Int16Array(buf.buffer, buf.byteOffset, count)
  const u = new Uint8Array(count * 2)
  for (let k = 0; k < u.length; k += 2) {
    u[k] = buf[k + 1]
    u[k + 1] = buf[k]
  }
  return new Int16Array(u.buffer, 0, count)
}

/** Bilinear terrain height at (x, y) in centimeters; clamps to the grid edge outside it. */
function terrainAtCM(terrain: Int16Array, nx: number, ny: number, minX: number, minY: number, step: number, x: number, y: number): number {
  const fx = clamp((x - minX) / step, 0, nx - 1)
  const fy = clamp((y - minY) / step, 0, ny - 1)
  const i = Math.max(0, Math.min(Math.floor(fx), nx - 2))
  const j = Math.max(0, Math.min(Math.floor(fy), ny - 2))
  const tx = fx - i
  const ty = fy - j
  const r0 = j * nx + i
  const r1 = r0 + nx
  const h00 = terrain[r0], h10 = terrain[r0 + 1], h01 = terrain[r1], h11 = terrain[r1 + 1]
  return (1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10 + (1 - tx) * ty * h01 + tx * ty * h11
}

/** Even-odd point-in-polygon over a flat [x0, y0, x1, y1, ...] ring. */
export function pointInPoly(x: number, y: number, flat: number[]): boolean {
  let inside = false
  const m = flat.length / 2
  for (let k = 0, l = m - 1; k < m; l = k++) {
    const x1 = flat[l * 2], y1 = flat[l * 2 + 1]
    const x2 = flat[k * 2], y2 = flat[k * 2 + 1]
    if (y1 > y !== y2 > y && x < ((x2 - x1) * (y - y1)) / (y2 - y1) + x1) inside = !inside
  }
  return inside
}

function cellAt(map: GameMap, x: number, y: number): number {
  const i = Math.floor((x - map.minX) / map.step)
  const j = Math.floor((y - map.minY) / map.step)
  return i >= 0 && i < map.nx && j >= 0 && j < map.ny ? j * map.nx + i : -1
}

/** Terrain height (m) at (x, y); clamped to the grid edge outside the grid. */
export function ground(map: GameMap, x: number, y: number): number {
  return terrainAtCM(map.terrain, map.nx, map.ny, map.minX, map.minY, map.step, x, y) * 0.01
}

/** Standing surface (m) at (x, y); terrain ground when off-grid. */
export function surface(map: GameMap, x: number, y: number): number {
  const k = cellAt(map, x, y)
  return k < 0 ? ground(map, x, y) : map.cellSurf[k]
}

/** True when the containing cell is walkable; false off-grid. */
export function feasible(map: GameMap, x: number, y: number): boolean {
  const k = cellAt(map, x, y)
  return k >= 0 && map.cellFeasible[k] === 1
}

/** Prism intervals covering (x, y); null off-grid, null/[] where no prism covers. */
export function intervals(map: GameMap, x: number, y: number): Interval[] | null {
  const k = cellAt(map, x, y)
  return k < 0 ? null : map.cellPrisms[k]
}

/**
 * Line of sight from a to b: sample every 1 m (endpoints included). Blocked when a sample is
 * below terrain or inside any prism interval [base, top] (closed) at its cell. Trees ignored.
 */
export function los(map: GameMap, a: Vec3, b: Vec3): boolean {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  const dz = b[2] - a[2]
  const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy, dz)))
  for (let k = 0; k <= steps; k++) {
    const t = k / steps
    const x = a[0] + dx * t
    const y = a[1] + dy * t
    const z = a[2] + dz * t
    if (z < ground(map, x, y)) return false
    const ivs = intervals(map, x, y)
    if (ivs) for (const iv of ivs) if (iv.base <= z && z <= iv.top) return false
  }
  return true
}
