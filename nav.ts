/**
 * nav.ts — Task 2 navigation for Mech Royale v1: grid A*, path following, local steering,
 * stuck recovery. Feeds real movement (protocol.md §7): step_up 1.2 m is walked up, anything
 * taller is a wall, any drop is allowed (falls).
 *
 * Conventions (shared with map.ts's GameMap):
 *   cell (cx, cy) = the 1 m square [minX+cx, minX+cx+1) x [minY+cy, minY+cy+1),
 *   center at (minX+cx+0.5, minY+cy+0.5), index cy*nx + cx (row-major, like terrain.i16).
 * Feasibility lives in the map (map.ts builds cellFeasible / cellSurf); nav applies the climb
 * rule on top: a move is allowed if cellFeasible is set AND the rise in cellSurf <= 1.2 m.
 *
 * Signatures are pinned — brain.ts (Task 3) builds against them.
 */
import { type GameMap } from "./map.ts"

export type Vec2 = [number, number]

const STEP_UP = 1.2 + 1e-4 // cellSurf rise tolerance (m)
// Deadlines fire a hair under the contract's 15 ms cap: the open-set clock is sampled every
// SEARCH_CHECK_POPS pops, so the measured wall time of any call (deadline slack + final heap /
// trace work) stays strictly below 15 ms.
const CAP_MS = 14
const SEARCH_CHECK_POPS = 16
const SQRT2 = Math.SQRT2

// ---------------------------------------------------------------------------
// A* scratch buffers, cached per map object (stamped generations, no per-call clears).
// ---------------------------------------------------------------------------
type Scratch = {
  cap: number
  nx: number
  g: Float64Array
  gGen: Int32Array // generation stamp: g[i] is valid iff gGen[i] === gen
  closedGen: Int32Array
  cameFrom: Int32Array
  heapCell: Int32Array
  heapF: Float64Array
  heapGz: Float64Array // g score, for tie-breaking toward goal (larger g wins)
  heapN: number
  gen: number
}

const scratchCache = new WeakMap<object, Scratch>()

function getScratch(map: GameMap): Scratch {
  const nx = map.nx
  const cap = nx * map.ny
  let s = scratchCache.get(map)
  if (!s || s.cap !== cap || s.nx !== nx || cap === 0) {
    // heap pre-sized for the worst push count (re-relaxations can near-double the pop count);
    // grows on demand if a pathological map ever exceeds it
    const heapCap = Math.max(4096, cap * 2)
    s = {
      cap,
      nx,
      g: new Float64Array(cap),
      gGen: new Int32Array(cap),
      closedGen: new Int32Array(cap),
      cameFrom: new Int32Array(cap),
      heapCell: new Int32Array(heapCap),
      heapF: new Float64Array(heapCap),
      heapGz: new Float64Array(heapCap),
      heapN: 0,
      gen: 0,
    }
    scratchCache.set(map, s)
  }
  return s
}

function heapPush(s: Scratch, cell: number, f: number, g: number): void {
  let i = s.heapN
  if (i >= s.heapCell.length) {
    const nCap = s.heapCell.length * 2
    const hc = new Int32Array(nCap)
    hc.set(s.heapCell)
    s.heapCell = hc
    const hf = new Float64Array(nCap)
    hf.set(s.heapF)
    s.heapF = hf
    const hz = new Float64Array(nCap)
    hz.set(s.heapGz)
    s.heapGz = hz
  }
  s.heapN++
  // sift up
  while (i > 0) {
    const p = (i - 1) >> 1
    const pf = s.heapF[p]
    if (pf < f || (pf === f && s.heapGz[p] >= g)) break
    s.heapCell[i] = s.heapCell[p]
    s.heapF[i] = pf
    s.heapGz[i] = s.heapGz[p]
    i = p
  }
  s.heapCell[i] = cell
  s.heapF[i] = f
  s.heapGz[i] = g
}

function heapPop(s: Scratch): number {
  const top = s.heapCell[0]
  s.heapN--
  if (s.heapN === 0) return top
  const cell = s.heapCell[s.heapN]
  const f = s.heapF[s.heapN]
  const g = s.heapGz[s.heapN]
  let i = 0
  for (;;) {
    const l = 2 * i + 1
    if (l >= s.heapN) break
    let b = l
    const lf = s.heapF[l]
    const r = l + 1
    if (r < s.heapN) {
      const rf = s.heapF[r]
      // pick the smaller child (ties: the one with larger g, matching the push ordering)
      if (rf < lf || (rf === lf && s.heapGz[r] > s.heapGz[l])) b = r
    }
    // keep sifting down while the min child sorts strictly BEFORE the moving element
    if (s.heapF[b] < f || (s.heapF[b] === f && s.heapGz[b] > g)) {
      s.heapCell[i] = s.heapCell[b]
      s.heapF[i] = s.heapF[b]
      s.heapGz[i] = s.heapGz[b]
      i = b
    } else break
  }
  s.heapCell[i] = cell
  s.heapF[i] = f
  s.heapGz[i] = g
  return top
}

const octile = (dx: number, dy: number): number =>
  dx > dy ? dx + (SQRT2 - 1) * dy : dy + (SQRT2 - 1) * dx

/** cell index of a world point, or -1 when outside the grid */
function cellAt(map: GameMap, x: number, y: number): number {
  const cx = Math.floor(x - map.minX)
  const cy = Math.floor(y - map.minY)
  if (cx < 0 || cy < 0 || cx >= map.nx || cy >= map.ny) return -1
  return cy * map.nx + cx
}

/** world point of a cell index (must be in bounds) */
function cellCenter(map: GameMap, idx: number): Vec2 {
  const cx = idx % map.nx
  const cy = (idx - cx) / map.nx
  return [map.minX + cx + 0.5, map.minY + cy + 0.5]
}

function tracePath(map: GameMap, s: Scratch, endIdx: number): Vec2[] {
  const out: Vec2[] = []
  for (let c = endIdx; c !== -1; c = s.cameFrom[c]) out.push(cellCenter(map, c))
  out.reverse()
  return out
}

// 8 neighbor offsets, ordered so straight steps come before diagonals
const NB_DX = [1, -1, 0, 0, 1, 1, -1, -1]
const NB_DY = [0, 0, 1, -1, 1, -1, 1, -1]
const NB_DIAG = [0, 0, 0, 0, 1, 1, 1, 1]

const SNAP_START_R = 6 // max Chebyshev ring (cells) to snap an infeasible start
const SNAP_GOAL_R = 15 // max Chebyshev ring (cells) to snap an infeasible goal

/**
 * idx itself if feasible, else the first feasible cell on Chebyshev rings 1..maxR around it
 * (ring order, not true Euclidean nearest — good enough), else -1.
 */
function nearestFeasible(map: GameMap, idx: number, maxR: number): number {
  if (map.cellFeasible[idx]) return idx
  const nx = map.nx
  const cx0 = idx % nx
  const cy0 = (idx - cx0) / nx
  for (let r = 1; r <= maxR; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue
        const x = cx0 + dx
        const y = cy0 + dy
        if (x < 0 || y < 0 || x >= nx || y >= map.ny) continue
        const k = y * nx + x
        if (map.cellFeasible[k]) return k
      }
    }
  }
  return -1
}

/**
 * A* on the 1 m grid, 8-connected, no corner cutting (a diagonal needs both orthogonal
 * neighbors feasible), octile heuristic, binary-heap open set. Hard cap 15 ms: on timeout
 * returns the best partial path (frontier cell closest to goal). An infeasible start/goal cell
 * is first snapped to the nearest feasible cell (start <= 6 cells, goal <= 15 cells); the path
 * then starts/ends at the snapped cell. null when an endpoint is out of bounds or has no
 * feasible cell nearby, or when the search exhausts the start's region (unreachable).
 * Cap-out with a partial path is non-null. `capMs` overrides the 15 ms cap downward (the act()
 * deadline drops it to 10 ms); it is a hard ceiling on search wall time. Waypoints are world
 * coords of cell centers.
 */
export function findPath(map: GameMap, from: Vec2, to: Vec2, capMs: number = CAP_MS): Vec2[] | null {
  const nx = map.nx
  const feas = map.cellFeasible
  const surf = map.cellSurf
  const sRaw = cellAt(map, from[0], from[1])
  const gRaw = cellAt(map, to[0], to[1])
  if (sRaw < 0 || gRaw < 0) return null // out of bounds
  const sIdx = nearestFeasible(map, sRaw, SNAP_START_R)
  const gIdx = nearestFeasible(map, gRaw, SNAP_GOAL_R)
  if (sIdx < 0 || gIdx < 0) return null // no feasible cell near an endpoint
  if (sIdx === gIdx) return [cellCenter(map, sIdx)]

  const t0 = performance.now()
  const s = getScratch(map)
  const gen = ++s.gen
  s.cameFrom[sIdx] = -1
  s.g[sIdx] = 0
  s.gGen[sIdx] = gen

  const gx = gIdx % nx
  const gy = (gIdx - gx) / nx
  let bestCell = sIdx
  let bestH = octile(Math.abs((sIdx % nx) - gx), Math.abs((((sIdx - (sIdx % nx)) / nx) | 0) - gy))
  s.heapN = 0
  heapPush(s, sIdx, bestH, 0)

  let pops = 0
  while (s.heapN > 0) {
    if ((++pops & (SEARCH_CHECK_POPS - 1)) === 0 && performance.now() - t0 > capMs) {
      return tracePath(map, s, bestCell) // cap-out: partial toward the frontier cell closest to goal
    }
    const c = heapPop(s)
    if (s.closedGen[c] === gen) continue // stale heap entry
    s.closedGen[c] = gen

    const cx = c % nx
    const cy = (c - cx) / nx
    const ch = octile(Math.abs(cx - gx), Math.abs(cy - gy))
    if (ch < bestH) {
      bestH = ch
      bestCell = c
    }
    if (c === gIdx) return tracePath(map, s, gIdx)

    const cs = surf[c]
    const cg = s.g[c]
    for (let d = 0; d < 8; d++) {
      const px = cx + NB_DX[d]
      if (px < 0 || px >= nx) continue
      const py = cy + NB_DY[d]
      if (py < 0 || py >= map.ny) continue
      const n = py * nx + px
      if (s.closedGen[n] === gen) continue
      if (!feas[n]) continue
      if (surf[n] - cs > STEP_UP) continue // taller than step_up: a wall
      if (NB_DIAG[d]) {
        // no corner cutting: both orthogonal neighbors must be feasible cells
        if (!feas[cy * nx + px] || !feas[py * nx + cx]) continue
      }
      const ng = cg + (NB_DIAG[d] ? SQRT2 : 1)
      if (s.gGen[n] !== gen || ng < s.g[n]) {
        s.g[n] = ng
        s.gGen[n] = gen
        s.cameFrom[n] = c
        heapPush(s, n, ng + octile(Math.abs(px - gx), Math.abs(py - gy)), ng)
      }
    }
  }
  return null // search exhausted: the (snapped) goal is unreachable from the start's region
}

// ---------------------------------------------------------------------------
// Local geometry helpers (shared by localSteer / Navigator)
// ---------------------------------------------------------------------------

/** true if the cell containing (x, y) is feasible and not more than step_up above `baseSurf` */
function pointFeasible(map: GameMap, x: number, y: number, baseSurf: number): boolean {
  const c = cellAt(map, x, y)
  if (c < 0 || !map.cellFeasible[c]) return false
  return map.cellSurf[c] - baseSurf <= STEP_UP
}

/**
 * Is the straight segment pos -> target traversable? Walks EVERY grid cell the segment touches
 * (supercover / Amanatides-Woo, so a clipped corner cell is never skipped; at an exact corner
 * crossing both orthogonal cells are checked too). Each cell after pos's own must be feasible
 * and must not rise more than step_up above the previous cell.
 */
function lineFeasible(map: GameMap, pos: Vec2, target: Vec2, baseSurf: number): boolean {
  const x0 = pos[0] - map.minX
  const y0 = pos[1] - map.minY
  const dx = target[0] - pos[0]
  const dy = target[1] - pos[1]
  let cx = Math.floor(x0)
  let cy = Math.floor(y0)
  const ex = Math.floor(x0 + dx)
  const ey = Math.floor(y0 + dy)
  const stepX = dx > 0 ? 1 : dx < 0 ? -1 : 0
  const stepY = dy > 0 ? 1 : dy < 0 ? -1 : 0
  let tMaxX = stepX !== 0 ? ((stepX > 0 ? cx + 1 : cx) - x0) / dx : Infinity
  let tMaxY = stepY !== 0 ? ((stepY > 0 ? cy + 1 : cy) - y0) / dy : Infinity
  const tDX = stepX !== 0 ? stepX / dx : Infinity
  const tDY = stepY !== 0 ? stepY / dy : Infinity
  let last = baseSurf
  const ok = (x: number, y: number): boolean => {
    if (x < 0 || y < 0 || x >= map.nx || y >= map.ny) return false
    const c = y * map.nx + x
    if (!map.cellFeasible[c]) return false
    if (map.cellSurf[c] - last > STEP_UP) return false
    last = map.cellSurf[c]
    return true
  }
  // exactly |ex-cx| + |ey-cy| unit moves reach the target cell (a corner move counts as 2)
  let remaining = Math.abs(ex - cx) + Math.abs(ey - cy)
  while (remaining > 0) {
    if (tMaxX < tMaxY) {
      cx += stepX
      tMaxX += tDX
      remaining--
    } else if (tMaxY < tMaxX) {
      cy += stepY
      tMaxY += tDY
      remaining--
    } else {
      // passes exactly through a grid corner: be conservative, both orthogonals must be clear
      const base = last
      if (!ok(cx + stepX, cy)) return false
      last = base
      if (!ok(cx, cy + stepY)) return false
      last = base
      cx += stepX
      cy += stepY
      tMaxX += tDX
      tMaxY += tDY
      remaining -= 2
    }
    if (!ok(cx, cy)) return false
  }
  return true
}

// ---------------------------------------------------------------------------
// localSteer
// ---------------------------------------------------------------------------

/**
 * If the point 3 m ahead along `dir` is feasible, return dir. Otherwise try dir rotated
 * +30deg, -30deg, +60deg, -60deg, ... (alternating, increasing, first match wins, up to
 * +/-150deg). If nothing is feasible, return dir unchanged (the caller decides).
 */
export function localSteer(map: GameMap, pos: Vec2, dir: Vec2): Vec2 {
  const len = Math.hypot(dir[0], dir[1])
  if (!(len > 1e-9)) return [dir[0], dir[1]]
  const ux = dir[0] / len
  const uy = dir[1] / len
  const posCell = cellAt(map, pos[0], pos[1])
  if (posCell < 0) return [ux, uy]
  const base = map.cellSurf[posCell]
  const clear3 = (a: number, b: number): boolean =>
    pointFeasible(map, pos[0] + a * 3, pos[1] + b * 3, base)
  if (clear3(ux, uy)) return [ux, uy]
  for (let k = 1; k <= 5; k++) {
    const ang = (k * 30 * Math.PI) / 180
    const c = Math.cos(ang)
    const sn = Math.sin(ang)
    // +k*30deg then -k*30deg (rotation in world frame, y = north)
    const px = ux * c - uy * sn
    const py = ux * sn + uy * c
    if (clear3(px, py)) return [px, py]
    const mx = ux * c + uy * sn
    const my = -ux * sn + uy * c
    if (clear3(mx, my)) return [mx, my]
  }
  return [ux, uy]
}

// ---------------------------------------------------------------------------
// Navigator
// ---------------------------------------------------------------------------

const AT_GOAL_M = 1.5 // close enough to stop
const REPLAN_S = 1.5 // replan cadence
const GOAL_MOVED_M = 8 // replan immediately when the goal moves this far
const WP_REACHED_M = 1.5 // waypoint consumed when pos is this close
const RANGE_M = 25 // look-ahead window for straight-line reachable waypoints

/** Optional per-call control for Navigator.update (all members optional). */
export type NavUpdateOpts = {
  /** false: never run A* this call — reuse the last path as-is (act() deadline) */
  replan?: boolean
  /** true: ignore the 1.5 s cadence and replan right now (stress tests) */
  force?: boolean
  /** A* cap in ms for this call, below the contracted 15 (act() deadline drops it to 10) */
  capMs?: number
}

/**
 * Replans at most every 1.5 s (or immediately when the goal moved > 8 m from the goal the
 * last plan targeted) and follows the path: aims at the FARTHEST waypoint within 25 m of pos
 * reachable in a straight line whose every touched cell is feasible; if none, the
 * nearest waypoint; with no path at all, the direct direction toward the goal. Every non-zero
 * dir passes through localSteer before return. A null plan obeys the same 1.5 s cadence (no
 * per-tick replans); a fully consumed partial (cap-out) path forces a replan next tick.
 * opts (NavUpdateOpts) can allow or forbid the replan for this call, force one despite the
 * cadence, or lower the A* cap. dir is a unit move direction; [0,0] stands still.
 * Requires map.step === 1.
 */
export class Navigator {
  private map: GameMap
  private path: Vec2[] | null = null
  private pathPartial = false // last waypoint is not the (snapped) goal cell: cap-out partial
  private planned = false // a plan was attempted (path may still be null)
  private planGoal: Vec2 | null = null
  private lastPlanSec = -1e9
  private wpIdx = 0

  constructor(map: GameMap) {
    if (map.step !== 1) throw new Error(`nav.ts assumes a 1 m grid (map.step = ${map.step})`)
    this.map = map
    getScratch(map) // warm the A* scratch buffers now, so no findPath call pays allocation
  }

  update(pos: Vec2, goal: Vec2, nowSec: number, opts?: NavUpdateOpts): { dir: Vec2; atGoal: boolean } {
    const movedFar =
      !this.planGoal || Math.hypot(goal[0] - this.planGoal[0], goal[1] - this.planGoal[1]) > GOAL_MOVED_M
    if (
      (opts?.replan ?? true) &&
      (opts?.force === true || !this.planned || movedFar || nowSec - this.lastPlanSec >= REPLAN_S)
    ) {
      const p = findPath(this.map, pos, goal, opts?.capMs ?? CAP_MS)
      this.path = p
      this.pathPartial = false
      if (p && p.length > 0) {
        const last = p[p.length - 1]
        const gRaw = cellAt(this.map, goal[0], goal[1])
        const gSnap = gRaw >= 0 ? nearestFeasible(this.map, gRaw, SNAP_GOAL_R) : -1
        this.pathPartial = cellAt(this.map, last[0], last[1]) !== gSnap
      }
      this.planned = true
      this.planGoal = [goal[0], goal[1]]
      this.lastPlanSec = nowSec
      this.wpIdx = 0
    }

    const dGoal = Math.hypot(goal[0] - pos[0], goal[1] - pos[1])
    if (dGoal <= AT_GOAL_M) return { dir: [0, 0], atGoal: true }

    if (this.path && this.path.length > 0) {
      const p = this.path
      while (this.wpIdx < p.length - 1 && Math.hypot(p[this.wpIdx][0] - pos[0], p[this.wpIdx][1] - pos[1]) <= WP_REACHED_M) {
        this.wpIdx++
      }
      // partial path consumed (mech is at its end, short of the goal): replan next tick instead
      // of dithering around the end cell until the 1.5 s cadence
      const lastWp = p[p.length - 1]
      if (
        this.pathPartial &&
        this.wpIdx === p.length - 1 &&
        Math.hypot(lastWp[0] - pos[0], lastWp[1] - pos[1]) <= WP_REACHED_M
      ) {
        this.lastPlanSec = -1e9
      }
      // farthest waypoint index within RANGE_M of pos
      let hi = -1
      for (let j = p.length - 1; j >= this.wpIdx; j--) {
        if (Math.hypot(p[j][0] - pos[0], p[j][1] - pos[1]) <= RANGE_M) {
          hi = j
          break
        }
      }
      // aim at the farthest such waypoint with a feasible straight line; else nearest waypoint
      if (hi >= 0) {
        let target: Vec2 | null = null
        const posCell = cellAt(this.map, pos[0], pos[1])
        const baseSurf = posCell >= 0 ? this.map.cellSurf[posCell] : 0
        for (let j = hi; j >= this.wpIdx; j--) {
          if (lineFeasible(this.map, pos, p[j], baseSurf)) {
            target = p[j]
            this.wpIdx = j
            break
          }
        }
        // nothing ahead is clear: a waypoint consumed early (within WP_REACHED_M) may be the
        // corner still to round — aim back at the nearest clear one (wpIdx stays monotone)
        if (!target) {
          for (let j = this.wpIdx - 1; j >= Math.max(0, this.wpIdx - 3); j--) {
            if (lineFeasible(this.map, pos, p[j], baseSurf)) {
              target = p[j]
              break
            }
          }
        }
        if (target) {
          const ddx = target[0] - pos[0]
          const ddy = target[1] - pos[1]
          const dl = Math.hypot(ddx, ddy)
          if (dl > 1e-9) return { dir: localSteer(this.map, pos, [ddx / dl, ddy / dl]), atGoal: false }
        }
      }
      // crawl toward the NEAREST remaining waypoint whose straight segment is wall-safe (the
      // nearest-without-a-line can aim across a wall corner and localSteer's 3 m point probe
      // won't catch it); plain nearest only if no remaining waypoint has a clear segment
      let posCell = cellAt(this.map, pos[0], pos[1])
      let baseSurf = posCell >= 0 ? this.map.cellSurf[posCell] : 0
      let best = -1
      let bestD = Infinity
      let plainBest = this.wpIdx
      let plainBestD = Infinity
      for (let j = this.wpIdx; j < p.length; j++) {
        const d = Math.hypot(p[j][0] - pos[0], p[j][1] - pos[1])
        if (d < plainBestD) {
          plainBestD = d
          plainBest = j
        }
        if (d >= bestD) continue
        if (lineFeasible(this.map, pos, p[j], baseSurf)) {
          bestD = d
          best = j
        }
      }
      const pick = best >= 0 ? best : plainBest
      const ddx = p[pick][0] - pos[0]
      const ddy = p[pick][1] - pos[1]
      const dl = Math.hypot(ddx, ddy)
      if (dl > 1e-9) return { dir: localSteer(this.map, pos, [ddx / dl, ddy / dl]), atGoal: false }
    }

    // no path: direct dir toward the goal, locally steered (straight if the 3 m probe is clear,
    // else the first clear rotation; unchanged if nothing within +/-150deg is clear)
    const dl = Math.hypot(goal[0] - pos[0], goal[1] - pos[1])
    if (dl > 1e-9) {
      return { dir: localSteer(this.map, pos, [(goal[0] - pos[0]) / dl, (goal[1] - pos[1]) / dl]), atGoal: false }
    }
    return { dir: [0, 0], atGoal: false }
  }
}

// ---------------------------------------------------------------------------
// StuckMonitor
// ---------------------------------------------------------------------------

const STUCK_SPEED = 1 // m/s
const STUCK_TIME_S = 1
const BURST_S = 0.5

/**
 * Detects pushing against a wall. "burst": commanding a move but speed < 1 m/s for >= 1 s;
 * the caller then jumps+boosts for 0.5 s. "perp": exactly once, 0.5 s after the burst, if the
 * burst did not get the mech moving again; the 1 s stuck window then resets. Moving fast or
 * not commanding a move resets the window.
 */
export class StuckMonitor {
  private lastSec: number
  private stuckSec = 0
  private burstAt: number | null = null

  constructor(nowSec: number) {
    this.lastSec = nowSec
  }

  update(nowSec: number, speed: number, commandingMove: boolean): "none" | "burst" | "perp" {
    // clamp: an observation gap (respawn, missed ticks) must not jump straight to "burst"
    const dt = Math.min(0.25, Math.max(0, nowSec - this.lastSec))
    this.lastSec = nowSec
    // contract is boolean; defensively treat a Vec2 move as "commanding" only when non-zero
    const c: unknown = commandingMove
    if (Array.isArray(c)) commandingMove = c[0] !== 0 || c[1] !== 0
    else commandingMove = !!c

    if (this.burstAt !== null) {
      // burst window is running (caller jumps+boosts for 0.5 s)
      if (speed >= STUCK_SPEED || !commandingMove) {
        // burst worked (or caller let off): no perpendicular nudge needed
        this.burstAt = null
        this.stuckSec = 0
        return "none"
      }
      if (nowSec - this.burstAt >= BURST_S) {
        this.burstAt = null
        this.stuckSec = 0 // reset the 1 s window after the perp nudge
        return "perp"
      }
      return "none"
    }

    if (commandingMove && speed < STUCK_SPEED) {
      this.stuckSec += dt
      if (this.stuckSec >= STUCK_TIME_S) {
        this.burstAt = nowSec
        return "burst"
      }
    } else {
      this.stuckSec = 0
    }
    return "none"
  }
}
