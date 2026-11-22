/**
 * sim/engine.ts — local Mech Royale match engine (protocol.md §2–§11).
 *
 * v1 simplifications (documented in the sim report):
 * - canopy ignored for vision (user-directed); trunks/crowns block nothing (matches map.los())
 * - projectile/beam visibility: distance+wedge+LOS approximations
 * - overhang/eave slide-out + roof-edge hold approximated by plain slide; NPC wander simplified
 *
 * Lockstep: bots decide every cfg.match.decide_every ticks; commands persist across the
 * interval like the server's coast.
 */
import { loadMap, los, ground, intervals, feasible, type GameMap, type Vec3 } from "../map.ts"
import { type Cfg, angDiff, gauss, rng } from "./config.ts"

export type Vec2 = [number, number]

export type WeaponSlot = {
  kind: string
  cooldown: number
  ammo: number
  reload: number
  charge: number
  release: boolean
  shortUntil: number
  equipUntil: number
} | null

export type MemEntry = {
  pos: Vec3
  vel: Vec3
  aim: [number, number]
  hp: number
  maxHp: number
  cubes: number
  weapons: string[]
  charging: boolean
  lastT: number
  kind: "mech" | "npc"
}

export type Mech = {
  id: number
  isNpc: boolean
  pos: Vec3
  vel: Vec3
  aim: [number, number]
  cmdAim: [number, number]
  hp: number
  maxHp: number
  energy: number
  maxEnergy: number
  grounded: boolean
  cubes: number
  radius: number
  height: number
  inZone: boolean
  alive: boolean
  place: number
  deathT: number
  deathCause: { by: number | null; weapon: string } | null
  deathPos: Vec3
  weapons: [WeaponSlot, WeaponSlot]
  move: Vec2
  boost: boolean
  fire: [boolean, boolean]
  reloadReq: boolean
  upgrades: Record<string, number>
  stats: Record<string, number>
  weaponStats: Record<string, Record<string, number>>
  lastCombatT: number
  lastEnergySpendT: number
  stepTimer: number
  mem: Map<number, MemEntry>
  events: Record<string, unknown>[]
  lastObsT: number
  offer: { id: number; expires: number; choices: { id: string; name: string; desc: string }[] } | null
  offerQueue: number
  dmgTaken: Record<string, number>
  dmgTakenFront: number
  dmgTakenRear: number
  dmgDealt: number
  kills: number
  lastHitBy: { id: number | null; weapon: string } | null
}

export type Npc = Mech & { isNpc: true; home: Vec2; wanderT: number; wanderTarget: [number, number]; lastAttacker: number; lastAttackerT: number }

export type Crate = { id: number; pos: Vec3; hp: number }
export type Cube = { id: number; pos: Vec3 }
export type Pickup = { id: number; kind: string; pos: Vec3 }
export type Proj = {
  id: number
  kind: "gun" | "mortar" | "npc"
  ownerId: number
  pos: Vec3
  vel: Vec3
  bornT: number
  maxT: number
}
export type Drop = { id: number; pos: Vec3; radius: number; landsIn: number; captureS: number; progress: number; holder: number | null; called: boolean }
export type SoundEv = { srcId: number; kind: string; x: number; y: number; t: number }

const dist2d = (a: Vec3 | [number, number], b: Vec3 | [number, number]): number => Math.hypot(a[0] - b[0], a[1] - b[1])
const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v)

export type DeathStat = { mechId: number; place: number; t: number; cause: { by: number | null; weapon: string } | null; inZoneAtDeath: boolean }
export type ReplayFrame = {
  tick: number
  t: number
  zone: { x: number; y: number; r: number }
  mechs: { id: number; x: number; y: number; z: number; yaw: number; hp: number; cubes: number; alive: boolean }[]
  npcs: { id: number; x: number; y: number; hp: number }[]
  shots: { kind: string; a: [number, number, number]; b: [number, number, number] | null; by: number }[]
  deaths: number[]
}
export type MatchResult = {
  seed: number
  shotsByHouse: number
  duration: number
  placements: { id: number; place: number; name: string; hp: number; cubes: number; kills: number; alive: boolean }[]
  deaths: DeathStat[]
  focus: {
    id: number
    place: number
    score: number
    deathT: number | null
    cause: { by: number | null; weapon: string } | null
    dmgTaken: Record<string, number>
    dmgTakenFront: number
    dmgTakenRear: number
    dmgDealt: number
    kills: number
    cubes: number
    inZoneAtDeath: boolean
    survived: boolean
  } | null
  replay: ReplayFrame[]
}

export type BotHandle = {
  name?: string
  act?: (obs: Record<string, unknown>) => Record<string, unknown>
  cmd?: string[]
  cwd?: string
}

export class Engine {
  cfg: Cfg
  map: GameMap
  rng: () => number
  t = 0
  tick = 0
  dt = 0.05
  mechs: Mech[] = []
  npcs: Npc[] = []
  crates: Crate[] = []
  cubes: Cube[] = []
  pickups: Pickup[] = []
  projs: Proj[] = []
  drops: Drop[] = []
  beams: { by: number; kind: string; path: Vec3[]; bornT: number }[] = []
  sounds: SoundEv[] = []
  frameShots: { kind: string; a: Vec3; b: Vec3 | null; by: number }[] = []
  zonePhase = 0
  zone = { cx: 0, cy: 0, r: 560 }
  next: { cx: number; cy: number; r: number; dps: number; shrinkStart: number; shrinkEnd: number } = { cx: 0, cy: 0, r: 0, dps: 0, shrinkStart: 0, shrinkEnd: 0 }
  curDps = 0
  aliveCount = 0
  deaths: DeathStat[] = []
  nextId = 8000
  replay: ReplayFrame[] = []
  replayEvery = 5
  names: string[] = []
  shotsByHouse = 0
  sigCrates = 0
  sigDrones = 0
  sigDrops = 0
  shotsByFocus = 0
  projHits = 0
  hitVictims: Record<number, number> = {}
  visRows: Record<number, number> = {}
  projWalls = 0
  bounds = { minX: 0, maxX: 0, minY: 0, maxY: 0 }
  phaseStart = new Map<number, { cx: number; cy: number; r: number }>()
  private offerSeq = 1

  constructor(cfg: Cfg, map: GameMap, matchSeed: number) {
    this.cfg = cfg
    this.map = map
    this.dt = 1 / cfg.match.tick_hz
    this.rng = rng(matchSeed)
    this.aliveCount = cfg.match.mechs
    this.bounds = { minX: map.minX, maxX: map.minX + map.nx * map.step, minY: map.minY, maxY: map.minY + map.ny * map.step }
    const z = cfg.zone
    this.zone = { cx: z.start_center[0], cy: z.start_center[1], r: z.start_radius }
    this.phaseStart.set(0, { ...this.zone })
    this.setupNextPhase(0, 0)
    this.curDps = z.phases[0]!.dps
    this.spawnWorld()
  }

  // ---------- world setup ----------

  private setupNextPhase(k: number, now: number) {
    const p = this.cfg.zone.phases[k]!
    const dMax = Math.max(0, this.zone.r - p.radius)
    const d = Math.sqrt(this.rng()) * dMax
    const a = this.rng() * Math.PI * 2
    this.next = {
      cx: this.zone.cx + Math.cos(a) * d,
      cy: this.zone.cy + Math.sin(a) * d,
      r: p.radius,
      dps: p.dps,
      shrinkStart: now + p.hold_s,
      shrinkEnd: now + p.hold_s + p.shrink_s,
    }
    this.zonePhase = k
  }

  private openPoint(cx: number, cy: number, rMax: number, wantRoof: boolean): { pos: Vec3; roofTop: number | null } | null {
    for (let i = 0; i < 400; i++) {
      const d = Math.sqrt(this.rng()) * rMax
      const a = this.rng() * Math.PI * 2
      const x = cx + Math.cos(a) * d
      const y = cy + Math.sin(a) * d
      if (x < this.bounds.minX + 4 || x > this.bounds.maxX - 4 || y < this.bounds.minY + 4 || y > this.bounds.maxY - 4) continue
      const z = ground(this.map, x, y)
      const ivs = intervals(this.map, x, y)
      const solid = ivs?.some((iv) => iv.base < z + 9 && iv.top > z) ?? false
      if (wantRoof) {
        if (!ivs) continue
        let topZ = -Infinity
        let roofed = false
        for (const iv of ivs) if (iv.base <= z + 2.5 && iv.top > z + 1) {
          roofed = true
          if (iv.top > topZ) topZ = iv.top
        }
        if (!roofed) continue
        return { pos: [x, y, topZ], roofTop: topZ }
      }
      if (solid) continue
      return { pos: [x, y, z], roofTop: null }
    }
    return null
  }

  private spawnWorld() {
    const c = this.cfg
    const R = c.zone.start_radius * c.match.spawn_zone_fraction
    const pts: [number, number][] = []
    for (let i = 0; i < c.match.mechs; i++) {
      let ok = false
      for (let k = 0; k < 600; k++) {
        const d = Math.sqrt(this.rng()) * R
        const a = this.rng() * Math.PI * 2
        const x = c.zone.start_center[0] + Math.cos(a) * d
        const y = c.zone.start_center[1] + Math.sin(a) * d
        if (x < this.bounds.minX + 5 || x > this.bounds.maxX - 5 || y < this.bounds.minY + 5 || y > this.bounds.maxY - 5) continue
        if (pts.some((p) => Math.hypot(p[0] - x, p[1] - y) < c.match.spawn_min_separation)) continue
        const z = ground(this.map, x, y)
        const ivs = intervals(this.map, x, y)
        if (ivs?.some((iv) => iv.base < z + 9 && iv.top > z)) continue
        pts.push([x, y])
        ok = true
        break
      }
      if (!ok) pts.push([c.zone.start_center[0] + (i - 8) * 12, c.zone.start_center[1]])
    }
    for (let i = 0; i < c.match.mechs; i++) {
      const [x, y] = pts[i]!
      const m = this.freshMech(i, [x, y, ground(this.map, x, y)])
      m.aim = [this.rng() * Math.PI * 2, 0]
      m.cmdAim = [...m.aim] as [number, number]
      this.mechs.push(m)
      this.names[i] = `bot-${i}`
    }
    for (let i = 0; i < c.npc.count; i++) {
      const p = this.openPoint(this.zone.cx, this.zone.cy, this.zone.r * 0.95, false)
      const pos = p?.pos ?? [this.zone.cx, this.zone.cy, 0]
      const n = this.freshMech(100 + i, pos, true) as Npc
      n.home = [pos[0], pos[1]]
      n.wanderT = this.rng() * 3
      n.wanderTarget = [pos[0], pos[1]]
      n.lastAttacker = -1
      n.lastAttackerT = -1
      this.npcs.push(n)
    }
    for (let i = 0; i < c.crates.count; i++) {
      const wantRoof = this.rng() < c.crates.roof_fraction
      const p = this.openPoint(this.zone.cx, this.zone.cy, this.zone.r * 0.98, wantRoof)
      if (p) this.crates.push({ id: 1000 + i, pos: p.roofTop !== null && wantRoof ? [p.pos[0], p.pos[1], p.roofTop] : p.pos, hp: c.crates.hp })
    }
    for (let i = 0; i < c.loot.floor_count; i++) {
      const wantRoof = this.rng() < c.loot.floor_roof_fraction
      const p = this.openPoint(this.zone.cx, this.zone.cy, this.zone.r * 0.98, wantRoof)
      const pos: Vec3 = p ? (p.roofTop !== null ? [p.pos[0], p.pos[1], p.roofTop] : p.pos) : [this.zone.cx, this.zone.cy, 0]
      this.pickups.push({ id: this.nextId++, kind: this.randomWeaponKind(), pos })
    }
    this.drops = c.drops.times.map((tm, i) => ({ id: 20000 + i, pos: [0, 0, 0], radius: c.drops.radius, landsIn: tm + c.drops.fall_s, captureS: c.drops.capture_s, progress: 0, holder: null, called: false }))
  }

  private freshMech(id: number, pos: Vec3, isNpc = false): Mech {
    const c = this.cfg
    return {
      id,
      isNpc,
      pos: [...pos] as Vec3,
      vel: [0, 0, 0],
      aim: [0, 0],
      cmdAim: [0, 0],
      hp: c.mech.hp,
      maxHp: c.mech.hp,
      energy: c.mech.energy,
      maxEnergy: c.mech.energy,
      grounded: true,
      cubes: 0,
      radius: c.mech.radius,
      height: c.mech.height,
      inZone: true,
      alive: true,
      place: 0,
      deathT: 0,
      deathCause: null,
      deathPos: [0, 0, 0],
      weapons: [
        isNpc ? null : { kind: "gun", cooldown: 0, ammo: Math.floor(c.weapons.gun!.mag ?? 12), reload: 0, charge: 0, release: false, shortUntil: 0, equipUntil: 0 },
        null,
      ],
      move: [0, 0],
      boost: false,
      fire: [false, false],
      reloadReq: false,
      upgrades: {},
      stats: this.baseStats(),
      weaponStats: {},
      lastCombatT: -999,
      lastEnergySpendT: -999,
      stepTimer: 0,
      mem: new Map(),
      events: [],
      lastObsT: 0,
      offer: null,
      offerQueue: 0,
      dmgTaken: {},
      dmgTakenFront: 0
      ,dmgTakenRear: 0,
      dmgDealt: 0,
      kills: 0,
      lastHitBy: null,
    }
  }

  private baseStats(): Record<string, number> {
    const m = this.cfg.mech
    return {
      walk_speed: m.walk_speed, boost_speed: m.boost_speed, turn_rate: m.turn_rate, pitch_rate: m.pitch_rate,
      eye_height: m.eye_height, muzzle_height: m.muzzle_height, gravity: m.gravity, jump_speed: m.jump_speed,
      air_accel: m.air_accel, ground_accel: m.ground_accel, armor: 0, near_range_bonus: 0, vision_bonus: 0,
      hearing: m.hearing, regen_delay: m.regen_delay, regen_per_s: m.regen_per_s, max_rise_speed: m.max_rise_speed,
      jet_accel: m.jet_accel, boost_drain: m.boost_drain, jet_drain: m.jet_drain, hp: m.hp, energy: m.energy,
      energy_regen: m.energy_regen, energy_regen_delay: m.energy_regen_delay, rear_mult: m.rear_mult,
      front_arc_deg: m.front_arc_deg, self_damage_mult: m.self_damage_mult,
    }
  }

  randomWeaponKind(): string {
    const w = this.cfg.loot.weights
    const kinds = Object.keys(w).filter((k) => (w[k] ?? 0) > 0)
    let x = this.rng() * kinds.reduce((s, k) => s + (w[k] ?? 0), 0)
    for (const k of kinds) {
      x -= w[k] ?? 0
      if (x <= 0) return k
    }
    return kinds[0] ?? "gun"
  }

  // ---------- geometry ----------

  /** standing surface at (x,y) given feet height z: terrain or prism tops <= z + step_up */
  surfAt(x: number, y: number, z: number): number {
    let s = ground(this.map, x, y)
    const ivs = intervals(this.map, x, y)
    if (ivs) for (const iv of ivs) if (iv.top > s && iv.top <= z + this.cfg.mech.step_up + 0.05) s = iv.top
    return s
  }

  /** cylinder horizontal collision at candidate (x,y), feet z */
  blockedXY(x: number, y: number, z: number, r: number, h: number, groundedStep: number): boolean {
    if (x < this.bounds.minX + r || x > this.bounds.maxX - r || y < this.bounds.minY + r || y > this.bounds.maxY - r) return true
    const stepCap = groundedStep
    // footprint: center + 8 ring points
    for (let k = 0; k < 9; k++) {
      const px = k === 0 ? x : x + Math.cos((k * Math.PI) / 4) * r
      const py = k === 0 ? y : y + Math.sin((k * Math.PI) / 4) * r
      const gh = ground(this.map, px, py)
      if (gh > z + stepCap + 0.01) return true
      const ivs = intervals(this.map, px, py)
      if (ivs) for (const iv of ivs) if (iv.base < z + h && iv.top > z + stepCap) return true
    }
    return false
  }

  /** first t in [0,1] where segment a->b is blocked by terrain/prism, or null */
  segBlockedT(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number | null {
    const d = Math.hypot(bx - ax, by - ay, bz - az)
    const n = Math.max(1, Math.ceil(d / 0.4))
    for (let k = 1; k <= n; k++) {
      const t = k / n
      const x = ax + (bx - ax) * t
      const y = ay + (by - ay) * t
      const z = az + (bz - az) * t
      if (z < ground(this.map, x, y)) return t
      const ivs = intervals(this.map, x, y)
      if (ivs) for (const iv of ivs) if (iv.base <= z && z <= iv.top) return t
    }
    return null
  }

  /** first t where segment hits a vertical cylinder (feet, r, h) or null */
  segHitT(px: number, py: number, pz: number, qx: number, qy: number, qz: number, cx: number, cy: number, cz: number, r: number, h: number): number | null {
    const dx = qx - px
    const dy = qy - py
    const a = dx * dx + dy * dy
    if (a < 1e-9) {
      const inside = Math.hypot(px - cx, py - cy) <= r
      if (!inside) return null
      const t = (cz + h - pz) / (qz - pz || 1e-9)
      const tz = pz + (qz - pz) * clamp(t, 0, 1)
      return tz >= cz && tz <= cz + h ? clamp(t, 0, 1) : null
    }
    const b = 2 * ((px - cx) * dx + (py - cy) * dy)
    const cc = (px - cx) ** 2 + (py - cy) ** 2 - r * r
    const disc = b * b - 4 * a * cc
    if (disc < 0) return null
    const s = Math.sqrt(disc)
    for (const t of [(-b - s) / (2 * a), (-b + s) / (2 * a)]) {
      if (t < 0 || t > 1) continue
      const z = pz + (qz - pz) * t
      if (z >= cz && z <= cz + h) return t
    }
    return null
  }

  private pointClear(x: number, y: number, z: number): boolean {
    if (z < ground(this.map, x, y)) return false
    const ivs = intervals(this.map, x, y)
    return !(ivs?.some((iv) => iv.base <= z && z <= iv.top) ?? false)
  }

  // ---------- weapons ----------

  private dir3(aim: [number, number], out: Vec3): Vec3 {
    const cp = Math.cos(aim[1])
    out[0] = cp * Math.cos(aim[0])
    out[1] = cp * Math.sin(aim[0])
    out[2] = Math.sin(aim[1])
    return out
  }

  private mergedWeaponStats(m: Mech, kind: string): Record<string, number> {
    return { ...(this.cfg.weapons[kind] ?? {}), ...(m.weaponStats[kind] ?? {}) }
  }

  private muzzleOf(m: Mech): Vec3 {
    const kind = m.weapons.find((w) => w?.kind)?.kind ?? "gun"
    const h = kind === "mortar" ? m.height + 0.5 : m.stats.muzzle_height!
    return [m.pos[0], m.pos[1], m.pos[2] + h]
  }

  private shotOrigin(m: Mech): Vec3 {
    const mz = this.muzzleOf(m)
    const d = this.dir3(m.aim, [0, 0, 0] as Vec3)
    let dist = 3.0
    let p: Vec3 = [mz[0] + d[0] * dist, mz[1] + d[1] * dist, mz[2] + d[2] * dist]
    for (let i = 0; i < 12 && !this.pointClear(p[0], p[1], p[2]) && dist > 0.5; i++) {
      dist -= 0.25
      p = [mz[0] + d[0] * dist, mz[1] + d[1] * dist, mz[2] + d[2] * dist]
    }
    return p
  }

  private fireMech(m: Mech, dt: number) {
    const now = this.t
    for (let i = 0; i < 2; i++) {
      const w = m.weapons[i]
      if (!w) continue
      w.cooldown = Math.max(0, w.cooldown - dt)
      if (w.reload > 0) {
        w.reload -= dt
        if (w.reload <= 0) {
          w.reload = 0
          w.ammo = Math.floor(this.mergedWeaponStats(m, "gun").mag ?? 12)
        }
      }
      const firing = m.fire[i]
      const st = this.mergedWeaponStats(m, w.kind)
      const equipOk = now >= w.equipUntil
      if (w.kind === "gun") {
        if (firing && w.cooldown <= 0 && w.reload <= 0 && w.ammo > 0 && equipOk) {
          w.ammo--
          w.cooldown = st.cooldown ?? 0.24
          this.spawnBullet(m, "gun", st)
        }
        if (w.reload <= 0 && w.ammo <= 0) w.reload = st.reload ?? 2.4
        if (m.reloadReq && w.reload <= 0 && w.ammo < (st.mag ?? 12)) w.reload = st.reload ?? 2.4
      } else if (w.kind === "mortar") {
        if (firing && w.cooldown <= 0 && equipOk && m.energy >= (st.drain ?? 0)) {
          w.cooldown = st.cooldown ?? 2.6
          m.energy -= st.drain ?? 0
          m.lastEnergySpendT = now
          this.spawnMortar(m, st)
        }
      } else if (w.kind === "laser") {
        const maxCharge = st.charge ?? 0.75
        const canCharge = now >= w.shortUntil && equipOk
        if (firing) {
          if (w.charge === 0 && canCharge) this.sounds.push({ srcId: m.id, kind: "laser_charge", x: m.pos[0]!, y: m.pos[1]!, t: now })
          if (canCharge) {
            w.charge = Math.min(maxCharge + (st.hold_s ?? 2), w.charge + dt)
            if (w.charge >= maxCharge + (st.hold_s ?? 2)) this.fireLaser(m, w, st)
          }
        } else if (w.charge > 0 && canCharge) {
          if (w.charge < (st.min_charge ?? 0.15)) w.charge = Math.min(maxCharge, w.charge + dt)
          if (w.charge >= (st.min_charge ?? 0.15)) this.fireLaser(m, w, st)
        }
        if (w.charge > 0 && !firing) w.release = true
        if (firing) w.release = false
      } else if (w.kind === "lightning") {
        if (firing && w.cooldown <= 0 && equipOk && m.energy >= (st.energy ?? 3)) {
          m.energy -= st.energy ?? 3
          m.lastEnergySpendT = now
          w.cooldown = st.cooldown ?? 0.25
          this.fireLightning(m, st)
        }
      }
    }
  }

  private spawnBullet(m: Mech, kind: "gun" | "npc", st: Record<string, number>) {
    const o = this.shotOrigin(m)
    const d = this.dir3(m.aim, [0, 0, 0] as Vec3)
    const s = (st.spread_deg ?? 0) * (Math.PI / 180)
    const yaw = m.aim[0] + gauss(this.rng) * s
    const pitch = m.aim[1] + gauss(this.rng) * s
    const cp = Math.cos(pitch)
    const v: Vec3 = [cp * Math.cos(yaw) * st.speed!, cp * Math.sin(yaw) * st.speed!, Math.sin(pitch) * st.speed!]
    if (m.id === 0) this.shotsByFocus++
    else this.shotsByHouse++
    this.projs.push({ id: this.nextId++, kind, ownerId: m.id, pos: o, vel: v, bornT: this.t, maxT: this.t + (st.range ?? 230) / st.speed! })
    this.sounds.push({ srcId: m.id, kind, x: o[0], y: o[1], t: this.t })
  }

  private spawnMortar(m: Mech, st: Record<string, number>) {
    const mz = [m.pos[0], m.pos[1], m.pos[2] + m.height + 0.5] as Vec3
    const d = this.dir3(m.aim, [0, 0, 0] as Vec3)
    const v: Vec3 = [d[0] * st.speed!, d[1] * st.speed!, d[2] * st.speed!]
    this.projs.push({ id: this.nextId++, kind: "mortar", ownerId: m.id, pos: mz, vel: v, bornT: this.t, maxT: this.t + (st.ttl ?? 7) })
    this.sounds.push({ srcId: m.id, kind: "mortar", x: mz[0], y: mz[1], t: this.t })
  }

  private fireLaser(m: Mech, w: NonNullable<WeaponSlot>, st: Record<string, number>) {
    const f = clamp(w.charge / (st.charge ?? 0.75), 0, 1)
    const frac = (st.min_frac ?? 0.35) + (1 - (st.min_frac ?? 0.35)) * f
    w.charge = 0
    w.release = false
    w.cooldown = st.cooldown ?? 1.65
    this.hitscan(m, "laser", (st.damage ?? 315) * frac, st.range ?? 280, { cookoffMul: 1 + (st.cookoff ?? 0.6) })
    this.sounds.push({ srcId: m.id, kind: "laser", x: m.pos[0], y: m.pos[1], t: this.t })
  }

  private fireLightning(m: Mech, st: Record<string, number>) {
    const mz = this.shotOrigin(m)
    const range = st.range ?? 45
    const cone = ((st.cone_deg ?? 30) * Math.PI) / 180
    const path: Vec3[] = [mz]
    let from: Vec3 = mz
    let dmg = st.damage ?? 60
    const used = new Set<number>()
    for (let jump = 0; jump <= (st.chain ?? 2); jump++) {
      if (jump > 0) dmg *= st.chain_falloff ?? 0.7
      let best: { e: Mech | Crate; d2: number } | null = null
      if (jump === 0) {
        for (const e of [...this.mechs.filter((x) => x.alive && x.id !== m.id), ...this.npcs.filter((n) => n.alive), ...this.crates]) {
          const d = dist2d(e.pos, from)
          if (jump === 0 && !(d <= range && Math.abs(angDiff(Math.atan2(e.pos[1]! - mz[1], e.pos[0]! - mz[0]), m.aim[0])) <= cone)) continue
          if (jump > 0 && d > (st.chain_range ?? 14)) continue
          if (!los(this.map, from, [e.pos[0], e.pos[1], e.pos[2]! + (e.height ?? 3) / 2])) continue
          if (!best || d < best.d2) best = { e, d2: d }
        }
      } else {
        for (const e of [...this.mechs.filter((x) => x.alive && x.id !== m.id), ...this.npcs.filter((n) => n.alive), ...this.crates]) {
          if (used.has(e.id)) continue
          const d = dist2d(e.pos, from)
          if (d > (st.chain_range ?? 14)) continue
          if (!best || d < best.d2) best = { e, d2: d }
        }
      }
      if (!best) break
      const e = best.e
      used.add(e.id)
      path.push([e.pos[0], e.pos[1], e.pos[2]! + (e.height ?? 3) / 2])
      const com: Vec3 = [e.pos[0], e.pos[1], e.pos[2]! + (e.height ?? 3) / 2]
      if (e.height === undefined) {
        // crate
        this.hurtCrate(e as Crate, dmg, m.id, "lightning", m.pos, m)
      } else {
        this.damage(e as Mech, dmg, { weapon: "lightning", byId: m.id, srcPos: m.pos, src: m })
        const held = (e as Mech).weapons?.some((w) => w?.kind === "laser")
        if (held) {
          const em = e as Mech
          for (const w of em.weapons) if (w?.kind === "laser") {
            w.charge = 0
            w.shortUntil = this.t + (st.short_s ?? 2)
          }
          em.events.push({ type: "effect", effect: "short", by: m.id })
        }
      }
      from = com
      if (jump === 0 && !best) break
    }
    if (path.length === 1) {
      // no target: arc out to first wall or range
      const d = this.dir3(m.aim, [0, 0, 0] as Vec3)
      let tEnd = range
      for (let k = 1; k <= 45; k++) {
        const t = (range * k) / 45
        const p: Vec3 = [mz[0] + d[0] * t, mz[1] + d[1] * t, mz[2] + d[2] * t]
        if (!this.pointClear(p[0], p[1], p[2])) {
          tEnd = t
          break
        }
      }
      path.push([mz[0] + d[0] * tEnd, mz[1] + d[1] * tEnd, mz[2] + d[2] * tEnd])
    }
    this.beams.push({ by: m.id, kind: "lightning", path, bornT: this.t })
    this.frameShots.push({ kind: "lightning", a: path[0]!, b: path[path.length - 1]!, by: m.id })
    this.sounds.push({ srcId: m.id, kind: "lightning", x: m.pos[0], y: m.pos[1], t: this.t })
  }

  private hitscan(m: Mech, kind: string, dmg: number, range: number, opts: { cookoffMul?: number }) {
    const o = this.shotOrigin(m)
    const d = this.dir3(m.aim, [0, 0, 0] as Vec3)
    let tBest: number | null = null
    let hitE: Mech | Npc | Crate | null = null
    for (const e of [...this.mechs.filter((x) => x.alive && x.id !== m.id), ...this.npcs.filter((n) => n.alive), ...this.crates]) {
      const t = this.segHitT(o[0], o[1], o[2], o[0] + d[0] * range, o[1] + d[1] * range, o[2] + d[2] * range, e.pos[0], e.pos[1], e.pos[2], e.radius, e.height ?? 3)
      if (t !== null && (tBest === null || t < tBest)) {
        tBest = t
        hitE = e
      }
    }
    const wallT: number | null = this.segBlockedT(o[0], o[1], o[2], o[0] + d[0] * range, o[1] + d[1] * range, o[2] + d[2] * range)
    const tEnd = clamp(Math.min(tBest ?? range, wallT ?? range), 0, range)
    const end: Vec3 = [o[0] + d[0] * tEnd, o[1] + d[1] * tEnd, o[2] + d[2] * tEnd]
    this.beams.push({ by: m.id, kind, path: [o, end], bornT: this.t })
    this.frameShots.push({ kind, a: o, b: end, by: m.id })
    if (hitE && tBest !== null && tBest <= tEnd + 1e-9) {
      let amount = dmg
      let effect: string | undefined
      if (hitE.height === undefined) {
        this.hurtCrate(hitE as Crate, amount, m.id, kind, m.pos, m)
      } else {
        const em = hitE as Mech
        if (opts.cookoffMul && em.weapons?.some((w) => w?.kind === "mortar")) {
          amount *= opts.cookoffMul
          effect = "cookoff"
          em.events.push({ type: "effect", effect: "cookoff", by: m.id })
        }
        this.damage(em, amount, { weapon: kind, byId: m.id, srcPos: m.pos, src: m, effect })
      }
    }
  }

  private hurtCrate(c: Crate, amount: number, byId: number, weapon: string, srcPos: Vec3, src: Mech | null) {
    const wasAlive = c.hp > 0
    if (wasAlive && c.hp - amount <= 0 && byId === 0) (this as unknown as { sigCrates: number }).sigCrates++
    c.hp -= amount
    if (src) {
      src.dmgDealt += amount
      src.events.push({ type: "hit", target: c.id, amount: Math.round(amount), weapon })
    }
    void byId
    void srcPos
  }

  // ---------- damage & death ----------

  damage(victim: Mech, amount: number, info: { weapon: string; byId: number | null; srcPos: Vec3; src?: Mech | null; effect?: string }) {
    if (!victim.alive || amount <= 0) return
    let amt = amount
    let rear = false
    const isZone = info.weapon === "zone"
    if (!isZone) {
      const src = info.src
      if (src) amt *= 1 + this.cfg.cubes.damage_per_cube * Math.min(src.cubes, this.cfg.cubes.stat_cap)
      const armor = victim.stats.armor ?? 0
      amt *= Math.max(0.1, 1 - armor)
      if (info.byId !== null && info.byId === victim.id) amt *= victim.stats.self_damage_mult ?? 0.5
      const bearing = Math.atan2(info.srcPos[1]! - victim.pos[1]!, info.srcPos[0]! - victim.pos[0]!)
      const frontArc = ((victim.stats.front_arc_deg ?? 180) * Math.PI) / 180
      rear = Math.abs(angDiff(bearing, victim.aim[0])) > frontArc / 2
      if (rear && info.byId !== victim.id) amt *= victim.stats.rear_mult ?? 1.5
    }
    victim.hp = Math.max(0, victim.hp - amt)
    victim.lastCombatT = this.t
    victim.dmgTaken[info.weapon] = (victim.dmgTaken[info.weapon] ?? 0) + amt
    if (!isZone) {
      if (rear) victim.dmgTakenRear += amt
      else victim.dmgTakenFront += amt
    }
    if (victim.isNpc) {
      victim.lastAttacker = info.byId ?? -1
      victim.lastAttackerT = this.t
    }
    if (info.src) {
      info.src.dmgDealt += amt
      info.src.lastCombatT = this.t
      const he: Record<string, unknown> = { type: "hit", target: victim.id, amount: Math.round(amt), weapon: info.weapon }
      if (!victim.isNpc) he["rear"] = rear
      if (info.effect) he["effect"] = info.effect
      info.src.events.push(he)
    }
    const ve: Record<string, unknown> = { type: "damaged", amount: Math.round(amt), weapon: isZone ? "zone" : info.weapon, bearing: isZone ? null : Math.atan2(info.srcPos[1]! - victim.pos[1]!, info.srcPos[0]! - victim.pos[0]!), by: isZone ? null : info.byId }
    if (rear) ve["rear"] = true
    victim.events.push(ve)
    victim.lastHitBy = { id: info.byId, weapon: isZone ? "zone" : info.weapon }
    if (victim.hp <= 0 && victim.alive) {
      victim.alive = false
      if (victim.isNpc && info.byId === 0) (this as unknown as { sigDrones: number }).sigDrones++
      ;(victim as unknown as { aliveAtDeath: number }).aliveAtDeath = this.mechs.filter((z) => z.alive).length + 1
      victim.deathT = this.t
      victim.deathCause = { by: info.byId, weapon: info.weapon }
      victim.deathPos = [...victim.pos] as Vec3
      if (info.src && !victim.isNpc) {
        info.src.kills++
        info.src.events.push({ type: "kill", target: victim.id })
      }
    }
  }

  private explode(pos: Vec3, ownerId: number, src: Mech) {
    const st = this.mergedWeaponStats(src, "mortar")
    const R = st.splash_radius ?? 7
    for (const e of [...this.mechs.filter((x) => x.alive), ...this.npcs.filter((n) => n.alive), ...this.crates]) {
      const d = dist2d(pos, e.pos)
      if (d > R) continue
      const com: Vec3 = [e.pos[0], e.pos[1], e.pos[2]! + (e.height ?? 3) / 2]
      if (d > 0.5 && !los(this.map, pos, com)) continue
      const f = e === src ? 1 : 1 - (1 - (st.splash_edge ?? 0.3)) * (d / R)
      const dmg = (st.splash ?? 160) * f
      if (e.height === undefined) this.hurtCrate(e as Crate, dmg, ownerId, "mortar", pos, src)
      else {
        this.damage(e as Mech, dmg, { weapon: "mortar", byId: ownerId, srcPos: pos, src: e.id === ownerId ? null : src })
        const em = e as Mech
        const ux = (em.pos[0]! - pos[0]!) / Math.max(0.5, d)
        const uy = (em.pos[1]! - pos[1]!) / Math.max(0.5, d)
        em.vel[0]! += ux * (st.knock ?? 16) * f
        em.vel[1]! += uy * (st.knock ?? 16) * f
        em.vel[2]! += (st.knock_up ?? 9) * f
        em.energy = Math.max(0, em.energy - (st.drain ?? 45) * f)
        em.lastEnergySpendT = this.t
        if (em.id !== ownerId) em.events.push({ type: "effect", effect: "concuss", by: ownerId })
      }
    }
    this.sounds.push({ srcId: ownerId, kind: "explosion", x: pos[0], y: pos[1], t: this.t })
    this.frameShots.push({ kind: "explosion", a: pos, b: null, by: ownerId })
  }

  // ---------- movement ----------

  private stepMech(m: Mech, dt: number) {
    // torso slew
    const turnRate = m.stats.turn_rate!
    const dy = angDiff(m.cmdAim[0], m.aim[0])
    m.aim[0] += clamp(dy, -turnRate * dt, turnRate * dt)
    const pitchRate = m.stats.pitch_rate!
    m.aim[1] += clamp(m.cmdAim[1] - m.aim[1], -pitchRate * dt, pitchRate * dt)
    // movement
    const charging = m.weapons.some((w) => w?.kind === "laser" && w.charge > 0)
    let speed = m.stats.walk_speed!
    const wantBoost = m.boost && m.energy > 0
    if (wantBoost && m.grounded) {
      speed = m.stats.boost_speed!
      m.energy = Math.max(0, m.energy - m.stats.boost_drain! * dt)
      m.lastEnergySpendT = this.t
    }
    if (charging) speed *= 0.5
    const ml = Math.hypot(m.move[0], m.move[1])
    const len = Math.min(1, ml)
    const ux = ml > 1e-9 ? m.move[0]! / ml : 0
    const uy = ml > 1e-9 ? m.move[1]! / ml : 0
    const cmdVx = ux * len * speed
    const cmdVy = uy * len * speed
    const accel = m.grounded ? m.stats.ground_accel! : m.stats.air_accel!
    let dvx = cmdVx - m.vel[0]!
    let dvy = cmdVy - m.vel[1]!
    const dl = Math.hypot(dvx, dvy)
    if (dl > accel * dt && dl > 0) {
      dvx = (dvx / dl) * accel * dt
      dvy = (dvy / dl) * accel * dt
    }
    m.vel[0]! += dvx
    m.vel[1]! += dvy
    // horizontal integrate with slide
    const stepCap = m.grounded ? this.cfg.mech.step_up : 0.02
    let nx = m.pos[0]! + m.vel[0]! * dt
    let ny = m.pos[1]!
    if (!this.blockedXY(nx, ny, m.pos[2]!, m.radius, m.height, stepCap)) m.pos[0]! = nx
    else m.vel[0]! = 0
    ny = m.pos[1]! + m.vel[1]! * dt
    const nny = ny
    if (!this.blockedXY(m.pos[0]!, nny, m.pos[2]!, m.radius, m.height, stepCap)) m.pos[1]! = nny
    else m.vel[1]! = 0
    // vertical
    const surf = this.surfAt(m.pos[0]!, m.pos[1]!, m.pos[2]!)
    if (m.grounded && m.jumpReq) {
      m.vel[2] = m.stats.jump_speed!
      m.grounded = false
    }
    m.jumpReq = false
    if (!m.grounded) {
      if (m.boost && m.energy > 0) {
        m.vel[2] = Math.min(m.vel[2]! + m.stats.jet_accel! * dt, Math.max(m.vel[2]!, m.stats.max_rise_speed!))
        m.energy = Math.max(0, m.energy - m.stats.jet_drain! * dt)
        m.lastEnergySpendT = this.t
      }
      const prevZ = m.pos[2]!
      m.vel[2] -= m.stats.gravity! * dt
      let nz = m.pos[2]! + m.vel[2]! * dt
      // head bump (bridges/overhangs)
      if (m.vel[2]! > 0) {
        const ivs = intervals(this.map, m.pos[0]!, m.pos[1]!)
        const h = m.height
        if (ivs) for (const iv of ivs) if (iv.base >= prevZ + h && iv.base < nz + h) {
          nz = iv.base - h
          m.vel[2] = 0
          break
        }
      }
      m.pos[2] = nz
      if (m.pos[2]! <= surf) {
        m.pos[2] = surf
        m.vel[2] = 0
        m.grounded = true
      }
    } else {
      if (surf > m.pos[2]! + 0.01) {
        m.pos[2] = Math.min(surf, m.pos[2]! + this.cfg.mech.step_up)
      } else if (surf < m.pos[2]! - 0.3) {
        m.grounded = false
      } else {
        m.pos[2] = surf
        m.vel[2] = 0
      }
    }
    // step / boost / jet sounds
    const hSpeed = Math.hypot(m.vel[0]!, m.vel[1]!)
    const ss = this.cfg.sound
    if ((m.grounded && hSpeed > ss.step_speed) || (!m.grounded && m.boost && m.energy > 0)) {
      m.stepTimer += dt
      if (m.stepTimer >= ss.step_every) {
        m.stepTimer = 0
        const kind = !m.grounded ? "jet" : m.boost ? "boost" : "step"
        this.sounds.push({ srcId: m.id, kind, x: m.pos[0]!, y: m.pos[1]!, t: this.t })
      }
    } else m.stepTimer = 0
    this.fireMech(m, dt)
  }

  private stepNpc(n: Npc, dt: number) {
    if (!n.alive) return
    // target acquisition
    let target: Mech | null = null
    const attacker = this.mechs.find((m) => m.id === n.lastAttacker && m.alive)
    if (attacker && this.t - n.lastAttackerT < 8 && dist2d(n.pos, n.home) <= this.cfg.npc.chase_leash && dist2d(n.pos, attacker.pos) <= this.cfg.npc.deaggro_range) target = attacker
    if (!target) {
      if (dist2d(n.pos, n.home) <= 30) {
        let bestD = Infinity
        for (const m of this.mechs) {
          if (!m.alive) continue
          const d = dist2d(m.pos, n.pos)
          if (d <= this.cfg.npc.aggro_range && d < bestD && los(this.map, [n.pos[0], n.pos[1], n.pos[2]! + 2], [m.pos[0], m.pos[1], m.pos[2]! + 4.5])) {
            bestD = d
            target = m
          }
        }
      }
    }
    // wander
    n.wanderT -= dt
    if (n.wanderT <= 0) {
      n.wanderT = 2 + this.rng() * 3
      const a = this.rng() * Math.PI * 2
      const r = this.rng() * this.cfg.npc.leash
      n.wanderTarget = [n.home[0] + Math.cos(a) * r, n.home[1] + Math.sin(a) * r]
    }
    const goal = target ? target.pos : (n.wanderTarget as unknown as Vec3)
    const dx = goal[0] - n.pos[0]!
    const dy = goal[1] - n.pos[1]!
    const dd = Math.hypot(dx, dy)
    if (dd > 1.5) {
      const ux = dx / dd
      const uy = dy / dd
      n.vel[0]! += clamp(ux * this.cfg.npc.speed - n.vel[0]!, -30 * dt, 30 * dt)
      n.vel[1]! += clamp(uy * this.cfg.npc.speed - n.vel[1]!, -30 * dt, 30 * dt)
    } else {
      n.vel[0]! *= 0.8
      n.vel[1]! *= 0.8
    }
    const stepCap = this.cfg.mech.step_up
    const nx = n.pos[0]! + n.vel[0]! * dt
    if (!this.blockedXY(nx, n.pos[1]!, n.pos[2]!, this.cfg.npc.radius, this.cfg.npc.height, stepCap)) n.pos[0]! = nx
    else n.vel[0]! = 0
    const ny = n.pos[1]! + n.vel[1]! * dt
    if (!this.blockedXY(n.pos[0]!, ny, n.pos[2]!, this.cfg.npc.radius, this.cfg.npc.height, stepCap)) n.pos[1]! = ny
    else n.vel[1]! = 0
    const surf = this.surfAt(n.pos[0]!, n.pos[1]!, n.pos[2]!)
    n.grounded = true
    n.pos[2] = surf
    // shooting
    if (target) {
      n.aim = [Math.atan2(target.pos[1]! - n.pos[1]!, target.pos[0]! - n.pos[0]!), 0]
      const w = n.weapons[0]
      if (w) {
        w.cooldown = Math.max(0, w.cooldown - dt)
        if (w.cooldown <= 0) {
          w.cooldown = this.cfg.npc.cooldown
          const o: Vec3 = [n.pos[0], n.pos[1], n.pos[2]! + 2.5]
          const s = (this.cfg.npc.spread_deg * Math.PI) / 180
          const yaw = n.aim[0] + gauss(this.rng) * s
          const cp = Math.cos(0)
          const v: Vec3 = [cp * Math.cos(yaw) * this.cfg.npc.projectile_speed, cp * Math.sin(yaw) * this.cfg.npc.projectile_speed, 0]
          this.projs.push({ id: this.nextId++, kind: "npc", ownerId: n.id, pos: o, vel: v, bornT: this.t, maxT: this.t + 200 / this.cfg.npc.projectile_speed })
          this.sounds.push({ srcId: n.id, kind: "npc_gun", x: o[0], y: o[1], t: this.t })
        }
      }
    }
  }

  private stepProjectiles(dt: number) {
    for (let i = this.projs.length - 1; i >= 0; i--) {
      const p = this.projs[i]!
      const prev = [...p.pos] as Vec3
      if (p.kind === "mortar") p.vel[2]! -= this.cfg.mech.gravity * dt
      p.pos = [p.pos[0]! + p.vel[0]! * dt, p.pos[1]! + p.vel[1]! * dt, p.pos[2]! + p.vel[2]! * dt] as Vec3
      const wallT = this.segBlockedT(prev[0], prev[1], prev[2], p.pos[0]!, p.pos[1]!, p.pos[2]!)
      // entity hits along the segment
      let hitT: number | null = null
      let hitE: Mech | Crate | null = null
      const targets: (Mech | Crate)[] = [...this.mechs.filter((x) => x.alive && x.id !== p.ownerId), ...this.npcs.filter((n) => n.alive), ...this.crates]
      for (const e of targets) {
        const r = e.height === undefined ? this.cfg.crates.radius : e.radius
        const h = e.height === undefined ? this.cfg.crates.height : e.height
        const t = this.segHitT(prev[0], prev[1], prev[2], p.pos[0]!, p.pos[1]!, p.pos[2]!, e.pos[0], e.pos[1], e.pos[2], r, h)
        if (t !== null && (wallT === null || t <= wallT + 0.05) && (hitT === null || t < hitT)) {
          hitT = t
          hitE = e
        }
      }
      const shooter = p.ownerId < 16 ? this.mechs.find((x) => x.id === p.ownerId) : undefined
      if (p.kind === "mortar") {
        let boom: Vec3 | null = null
        if (hitE !== null && hitT !== null) {
          boom = [prev[0] + (p.pos[0]! - prev[0]) * hitT, prev[1] + (p.pos[1]! - prev[1]) * hitT, prev[2] + (p.pos[2]! - prev[2]) * hitT]
          if (hitE.height !== undefined) this.damage(hitE as Mech, this.mergedWeaponStats(shooter ?? (this.mechs[0] as Mech), "mortar").damage ?? 200, { weapon: "mortar", byId: p.ownerId, srcPos: prev, src: shooter ?? null })
          else this.hurtCrate(hitE as Crate, this.cfg.weapons.mortar!.damage ?? 200, p.ownerId, "mortar", prev, shooter ?? null)
        } else if (wallT !== null) {
          boom = [prev[0] + (p.pos[0]! - prev[0]) * wallT, prev[1] + (p.pos[1]! - prev[1]) * wallT, prev[2] + (p.pos[2]! - prev[2]) * wallT]
        } else if (p.pos[2]! <= this.surfAt(p.pos[0]!, p.pos[1]!, p.pos[2]! + 0.6)) {
          boom = [p.pos[0]!, p.pos[1]!, this.surfAt(p.pos[0]!, p.pos[1]!, p.pos[2]! + 0.6)]
        } else if (this.t >= p.maxT) boom = [...p.pos] as Vec3
        if (boom && shooter) {
          this.explode(boom, p.ownerId, shooter)
          this.projs.splice(i, 1)
        } else if (boom && !shooter) this.projs.splice(i, 1)
        continue
      }
      // straight bullets (gun / npc)
      const st = p.kind === "gun" ? this.cfg.weapons.gun! : { damage: this.cfg.npc.damage }
      if (hitE !== null && hitT !== null) {
        const hp: Vec3 = [prev[0] + (p.pos[0]! - prev[0]) * hitT, prev[1] + (p.pos[1]! - prev[1]) * hitT, prev[2] + (p.pos[2]! - prev[2]) * hitT]
        this.projHits++
        if (hitE.height !== undefined) this.hitVictims[hitE.id] = (this.hitVictims[hitE.id] ?? 0) + 1
        if (hitE.height === undefined) this.hurtCrate(hitE as Crate, st.damage ?? 45, p.ownerId, p.kind, prev, shooter ?? null)
        else this.damage(hitE as Mech, st.damage ?? 45, { weapon: p.kind, byId: p.ownerId, srcPos: prev, src: shooter ?? null })
        void hp
        this.projs.splice(i, 1)
      } else if (wallT !== null || this.t >= p.maxT) {
        if (wallT !== null) this.projWalls++
        this.projs.splice(i, 1)
      }
    }
  }

  // ---------- per-tick systems + step ----------

  step(dt: number) {
    // Task 9 instrumentation: t=90s signature snapshot per mech
    if (this.t < 90 && this.t + dt >= 90) {
      for (const m of this.mechs) {
        const mx = m as unknown as { sigCubesAt90?: number; sigSpecialsAt90?: number }
        mx.sigCubesAt90 = m.cubes
        mx.sigSpecialsAt90 = m.weapons.filter((w) => w && w.kind !== "gun").length
      }
    }
    // ec: t=150s cube snapshot
    if (this.t < 150 && this.t + dt >= 150) {
      for (const m of this.mechs) {
        const mx = m as unknown as { sigCubesAt150?: number }
        mx.sigCubesAt150 = m.cubes
      }
    }
    // Task 10 instrumentation: fraction of steps where no enemy mech sees the focus
    // canSee includes range+fov+LOS from the enemy's side == "enemy can see us"
    if (this.mechs[0]?.alive) {
      const me = this.mechs[0]!
      const self: { sigSteps: number; sigCoverSteps: number } = this as unknown as { sigSteps: number; sigCoverSteps: number }
      self.sigSteps = (self.sigSteps ?? 0) + 1
      let seen = false
      for (let i = 1; i < this.mechs.length; i++) {
        const e = this.mechs[i]!
        if (!e.alive) continue
        if (dist2d(e.pos, me.pos) > this.cfg.vision.range + 2) continue
        if (this.canSee(e, me.pos, me.pos[2]! + 4.5)) { seen = true; break }
      }
      if (!seen) self.sigCoverSteps = (self.sigCoverSteps ?? 0) + 1
    }
    this.t += dt
    this.tick++
    // zone timeline
    if (this.t >= this.next.shrinkEnd) {
      this.zone = { cx: this.next.cx, cy: this.next.cy, r: this.next.r }
      const nk = this.zonePhase + 1
      if (nk < this.cfg.zone.phases.length) {
        this.phaseStart.set(nk, { ...this.zone })
        this.setupNextPhase(nk, this.t)
        this.curDps = this.cfg.zone.phases[nk]!.dps
      } else {
        this.next = { ...this.next, shrinkStart: Infinity, shrinkEnd: Infinity }
      }
    } else if (this.t >= this.next.shrinkStart) {
      const start = this.phaseStart.get(this.zonePhase) ?? this.zone
      const f = (this.t - this.next.shrinkStart) / (this.next.shrinkEnd - this.next.shrinkStart)
      this.zone = { cx: start.cx + f * (this.next.cx - start.cx), cy: start.cy + f * (this.next.cy - start.cy), r: start.r + f * (this.next.r - start.r) }
    }
    // supply drops
    for (const d of this.drops) {
      if (!d.called && this.t >= -d.landsIn) {
        d.called = true
        const off = this.cfg.drops.spread
        const p = this.openPoint(this.next.cx, this.next.cy, Math.max(5, this.next.r * 0.9 - off), false)
        d.pos = [p?.pos[0] ?? this.next.cx, p?.pos[1] ?? this.next.cy, ground(this.map, p?.pos[0] ?? this.next.cx, p?.pos[1] ?? this.next.cy)]
        d.landsIn = this.cfg.drops.fall_s
      }
      if (d.called && d.landsIn > 0) {
        d.landsIn -= dt
        if (d.landsIn <= 0) {
          d.landsIn = 0
          d.pos[2] = ground(this.map, d.pos[0]!, d.pos[1]!)
          this.sounds.push({ srcId: -2, kind: "drop", x: d.pos[0]!, y: d.pos[1]!, t: this.t })
          this.frameShots.push({ kind: "drop", a: [...d.pos] as Vec3, b: null, by: -2 })
        }
      }
      if (d.called && d.landsIn <= 0.001) {
        const inside = this.mechs.filter((m) => m.alive && m.pos[2]! - d.pos[2]! < 4 && dist2d(m.pos, d.pos) <= d.radius)
        const sole = inside.length === 1 ? inside[0]! : null
        if (sole && d.holder !== sole.id) {
          d.holder = sole.id
          d.progress = 0
        }
        if (sole) {
          d.progress += dt
          if (d.progress >= d.captureS) {
            d.progress = -1
            d.holder = null
            if (sole.id === 0) (this as unknown as { sigDrops: number }).sigDrops++
            sole.hp = sole.maxHp
            sole.energy = sole.maxEnergy
            this.grantCubes(sole, this.cfg.drops.cubes)
            sole.events.push({ type: "drop", id: d.id, cubes: sole.cubes })
          }
        } else if (d.progress > 0) {
          d.holder = null
          d.progress = 0
        }
      }
    }
    this.drops = this.drops.filter((d) => d.progress >= 0)
    // mechs
    for (const m of this.mechs) if (m.alive) this.stepMech(m, dt)
    // separation
    for (let i = 0; i < this.mechs.length; i++) {
      const a = this.mechs[i]!
      if (!a.alive) continue
      for (let j = i + 1; j < this.mechs.length; j++) {
        const b = this.mechs[j]!
        if (!b.alive) continue
        const dx = b.pos[0]! - a.pos[0]!
        const dy = b.pos[1]! - a.pos[1]!
        const dd = Math.hypot(dx, dy)
        const min = a.radius + b.radius
        if (dd < min && dd > 1e-6) {
          const push = (min - dd) / 2
          const ux = dx / dd
          const uy = dy / dd
          if (!this.blockedXY(a.pos[0]! - ux * push, a.pos[1]! - uy * push, a.pos[2]!, a.radius, a.height, this.cfg.mech.step_up)) {
            a.pos[0]! -= ux * push
            a.pos[1]! -= uy * push
          }
          if (!this.blockedXY(b.pos[0]! + ux * push, b.pos[1]! + uy * push, b.pos[2]!, b.radius, b.height, this.cfg.mech.step_up)) {
            b.pos[0]! += ux * push
            b.pos[1]! += uy * push
          }
        }
      }
    }
    for (const n of this.npcs) this.stepNpc(n, dt)
    this.stepProjectiles(dt)
    // per-mech systems
    for (const m of this.mechs) {
      if (!m.alive) continue
      m.inZone = dist2d(m.pos, [this.zone.cx, this.zone.cy]) <= this.zone.r
      if (!m.inZone) this.damage(m, this.curDps * dt, { weapon: "zone", byId: null, srcPos: [...m.pos] as Vec3 })
      if (this.t - m.lastCombatT > (m.stats.regen_delay ?? 5) && m.inZone && m.hp < m.maxHp) m.hp = Math.min(m.maxHp, m.hp + (m.stats.regen_per_s ?? 45) * dt)
      if (this.t - m.lastEnergySpendT > this.cfg.mech.energy_regen_delay && m.energy < m.maxEnergy) m.energy = Math.min(m.maxEnergy, m.energy + this.cfg.mech.energy_regen * dt)
      for (let k = this.cubes.length - 1; k >= 0; k--) {
        const cb = this.cubes[k]!
        if (dist2d(cb.pos, m.pos) <= this.cfg.cubes.pickup_radius && Math.abs(cb.pos[2]! - m.pos[2]!) < 4) {
          this.cubes.splice(k, 1)
          this.grantCubes(m, 1)
          m.events.push({ type: "pickup", cubes: m.cubes })
        }
      }
      if (m.offer && this.t >= m.offer.expires) this.applyPick(m, 0)
    }
    // deaths (shared place per tick)
    const diedNow = this.mechs.filter((m) => !m.alive && m.place === 0)
    if (diedNow.length > 0) {
      for (const m of diedNow) {
        m.place = this.aliveCount
        this.deaths.push({ mechId: m.id, place: m.place, t: this.t, cause: m.deathCause, inZoneAtDeath: dist2d(m.deathPos, [this.zone.cx, this.zone.cy]) <= this.zone.r })
        this.sounds.push({ srcId: m.id, kind: "death", x: m.pos[0]!, y: m.pos[1]!, t: this.t })
        for (const w of m.weapons) if (w && w.kind !== "gun") this.pickups.push({ id: this.nextId++, kind: w.kind, pos: [m.pos[0]!, m.pos[1]!, m.pos[2]!] })
        for (let i = 0; i < m.cubes + 2; i++) {
          const a = this.rng() * Math.PI * 2
          const r = this.rng() * 2
          this.cubes.push({ id: this.nextId++, pos: [m.pos[0]! + Math.cos(a) * r, m.pos[1]! + Math.sin(a) * r, m.pos[2]!] })
        }
      }
      this.aliveCount -= diedNow.length
    }
    for (const n of this.npcs) {
      if (!n.alive && n.place === 0) {
        n.place = -1
        for (let i = 0; i < this.cfg.npc.cubes; i++) this.cubes.push({ id: this.nextId++, pos: [n.pos[0]!, n.pos[1]!, n.pos[2]!] })
      }
    }
    for (let i = this.crates.length - 1; i >= 0; i--) {
      if (this.crates[i]!.hp <= 0) {
        const c = this.crates.splice(i, 1)[0]!
        this.cubes.push({ id: this.nextId++, pos: [...c.pos] as Vec3 })
        if (this.rng() < this.cfg.loot.crate_weapon_chance) this.pickups.push({ id: this.nextId++, kind: this.randomWeaponKind(), pos: [...c.pos] as Vec3 })
        this.sounds.push({ srcId: -1, kind: "crate", x: c.pos[0]!, y: c.pos[1]!, t: this.t })
      }
    }
    this.beams = this.beams.filter((b) => this.t - b.bornT < 2)
    this.sounds = this.sounds.filter((s) => this.t - s.t < 2)
    if (this.tick % this.replayEvery === 0) {
      const deaths = this.mechs.filter((m) => !m.alive && this.t - m.deathT < this.dt * this.replayEvery).map((m) => m.id)
      this.replay.push({
        tick: this.tick,
        t: Math.round(this.t * 100) / 100,
        zone: { x: Math.round(this.zone.cx * 10) / 10, y: Math.round(this.zone.cy * 10) / 10, r: Math.round(this.zone.r * 10) / 10 },
        mechs: this.mechs.map((m) => ({ id: m.id, x: Math.round(m.pos[0]! * 10) / 10, y: Math.round(m.pos[1]! * 10) / 10, z: Math.round(m.pos[2]! * 10) / 10, yaw: Math.round(m.aim[0] * 100) / 100, hp: Math.round(m.hp), cubes: m.cubes, alive: m.alive })),
        npcs: this.npcs.filter((n) => n.alive).map((n) => ({ id: n.id, x: Math.round(n.pos[0]! * 10) / 10, y: Math.round(n.pos[1]! * 10) / 10, hp: Math.round(n.hp) })),
        shots: this.frameShots.splice(0, this.frameShots.length).map((s) => ({ kind: s.kind, a: s.a.map((v) => Math.round(v * 100) / 100) as [number, number, number], b: s.b ? (s.b.map((v) => Math.round(v * 100) / 100) as [number, number, number]) : null, by: s.by })),
        deaths,
      })
    }
  }

  result(focusId: number): MatchResult {
    const survivors = this.mechs.filter((m) => m.alive).sort((a, b) => b.hp - a.hp)
    survivors.forEach((m, i) => {
      m.place = i + 1
    })
    const placements = [...this.mechs]
      .sort((a, b) => a.place - b.place)
      .map((m) => ({ id: m.id, place: m.place, name: this.names[m.id] ?? `bot-${m.id}`, hp: Math.round(m.hp), cubes: m.cubes, kills: m.kills, alive: m.alive }))
    const f = this.mechs.find((m) => m.id === focusId)
    return {
      seed: 0,
      shotsByHouse: this.shotsByHouse,
      shotsByFocus: this.shotsByFocus,
      sigCrates: this.sigCrates,
      sigDrones: this.sigDrones,
      sigDrops: this.sigDrops,
      sigCoverFrac: this.sigSteps ? this.sigCoverSteps / this.sigSteps : 1,
      projHits: this.projHits,
      projWalls: this.projWalls,
      hitVictims: this.hitVictims,
      visRows: this.visRows,
      duration: Math.round(this.t * 10) / 10,
      placements,
      deaths: this.deaths,
      focus: f
        ? {
            id: f.id, place: f.place, score: 16 - f.place, deathT: f.alive ? null : Math.round(f.deathT * 10) / 10,
            cause: f.alive ? null : f.deathCause,
            dmgTaken: Object.fromEntries(Object.entries(f.dmgTaken).map(([k, v]) => [k, Math.round(v)])),
            dmgTakenFront: Math.round(f.dmgTakenFront), dmgTakenRear: Math.round(f.dmgTakenRear), dmgDealt: Math.round(f.dmgDealt),
            kills: f.kills, cubes: f.cubes, inZoneAtDeath: f.alive ? true : this.deaths.find((d) => d.mechId === f.id)?.inZoneAtDeath ?? false,
            aliveAtDeath: (f as unknown as { aliveAtDeath?: number }).aliveAtDeath ?? 1,
            hp: Math.round(f.hp),
            survived: f.alive,
            sigCubesAt90: (f as unknown as { sigCubesAt90?: number }).sigCubesAt90 ?? 0,
            sigCubesAt150: (f as unknown as { sigCubesAt150?: number }).sigCubesAt150 ?? 0,
            sigSpecialsAt90: (f as unknown as { sigSpecialsAt90?: number }).sigSpecialsAt90 ?? 0,
          }
        : null,
      replay: this.replay,
    }
  }

  finished(): boolean {
    return this.aliveCount <= 1 || this.t >= this.cfg.match.duration_s
  }

  grantCubes(m: Mech, n: number) {
    m.cubes += n
    const scale = 1 + this.cfg.cubes.size_per_cube * m.cubes
    m.radius = this.cfg.mech.radius * scale
    m.height = this.cfg.mech.height * scale
    m.stats.eye_height = this.cfg.mech.eye_height * scale
    m.stats.muzzle_height = this.cfg.mech.muzzle_height * scale
    const oldMax = m.maxHp
    m.maxHp = this.cfg.mech.hp * (1 + this.cfg.cubes.hp_per_cube * Math.min(m.cubes, this.cfg.cubes.stat_cap)) * this.upgMul(m, "mech.hp")
    m.hp += Math.max(0, m.maxHp - oldMax)
    for (const th of this.cfg.cubes.upgrade_thresholds) if (m.cubes - n < th && m.cubes >= th) m.offerQueue++
    if (m.offerQueue > 0 && !m.offer) this.makeOffer(m)
  }

  private upgMul(m: Mech, path: string): number {
    let mul = 1
    for (const u of this.cfg.upgrades) {
      if ((m.upgrades[u.id] ?? 0) > 0) for (const e of u.effects) if (e.path === path && e.mul) mul *= e.mul
    }
    return mul
  }

  private makeOffer(m: Mech) {
    m.offerQueue--
    const avail = this.cfg.upgrades.filter((u) => {
      if ((m.upgrades[u.id] ?? 0) >= ((u as unknown as { max: number }).max ?? 3)) return false
      const km = u.id.match(/^(gun|laser|lightning|mortar)_/)
      if (km) return m.weapons.some((w) => w?.kind === km[1])
      return true
    })
    const choices: { id: string; name: string; desc: string }[] = []
    for (let i = 0; i < 3 && avail.length > 0; i++) {
      const idx = Math.floor(this.rng() * avail.length)
      const u = avail.splice(idx, 1)[0]!
      choices.push({ id: u.id, name: (u as unknown as { name: string }).name, desc: (u as unknown as { desc: string }).desc })
    }
    m.offer = choices.length > 0 ? { id: this.offerSeq++, expires: this.t + this.cfg.bot.offer_timeout_s, choices } : null
  }

  applyPick(m: Mech, idx: number) {
    if (!m.offer) return
    const ch = m.offer.choices[clamp(idx, 0, m.offer.choices.length - 1)]!
    m.offer = null
    const u = this.cfg.upgrades.find((x) => x.id === ch.id)
    if (!u) return
    m.upgrades[u.id] = (m.upgrades[u.id] ?? 0) + 1
    for (const e of u.effects) {
      if (e.path.startsWith("weapons.")) {
        const [, kind, stat] = e.path.split(".")
        const w = (m.weaponStats[kind!] ??= { ...this.cfg.weapons[kind!] })
        if (e.mul) w[stat!] = (w[stat!] ?? 1) * e.mul
        else if (e.add !== undefined) w[stat!] = (w[stat!] ?? 0) + e.add
      } else if (e.path === "mech.hp") {
        const prev = m.maxHp
        m.maxHp *= e.mul ?? 1
        m.hp += Math.max(0, m.maxHp - prev)
      } else {
        const stat = e.path.replace("mech.", "")
        if (e.mul) m.stats[stat] = (m.stats[stat] ?? 1) * e.mul
        else if (e.add !== undefined) m.stats[stat] = (m.stats[stat] ?? 0) + e.add
      }
    }
    m.events.push({ type: "upgrade", id: u.id })
    if (m.offerQueue > 0 && !m.offer) this.makeOffer(m)
  }

  // ---------- actions ----------

  applyAction(m: Mech, a: Record<string, unknown>) {
    const mv = (a.move as [number, number] | undefined) ?? [0, 0]
    const ml = Math.hypot(mv[0] ?? 0, mv[1] ?? 0)
    m.move = ml > 1 ? [(mv[0] ?? 0) / ml, (mv[1] ?? 0) / ml] : [(mv[0] ?? 0), (mv[1] ?? 0)]
    const aim = (a.aim as [number, number] | undefined) ?? m.cmdAim
    m.cmdAim = [aim[0] ?? 0, clamp(aim[1] ?? 0, this.cfg.mech.min_pitch, this.cfg.mech.max_pitch)]
    m.boost = !!a.boost
    const f = (a.fire as [boolean, boolean] | undefined) ?? [false, false]
    m.fire = [!!f[0], !!f[1]]
    m.reloadReq = !!a.reload
    if (a.pick !== undefined && a.pick !== null && m.offer) this.applyPick(m, a.pick as number)
    const slotFor = (): number => (m.weapons[0] === null ? 0 : m.weapons[1] === null ? 1 : -1)
    if (typeof a.take === "number") {
      const pu = this.pickups.find((p) => p.id === a.take)
      const reason = !pu ? "gone" : dist2d(pu.pos, m.pos) > this.cfg.loot.pickup_radius ? "range" : (m.weapons.some((w) => w?.kind === pu.kind) ? "have" : null)
      if (reason) m.events.push({ type: "take_failed", id: a.take, reason })
      else {
        let slot = slotFor()
        if (slot === -1 && typeof a.drop === "number") slot = a.drop as number
        if (slot === -1) m.events.push({ type: "take_failed", id: a.take, reason: "full" })
        else {
          const charging = m.weapons[slot]?.kind === "laser" && m.weapons[slot]!.charge > 0
          if (charging) m.events.push({ type: "take_failed", id: a.take, reason: "busy" })
          else {
            const old = m.weapons[slot]
            if (old) {
              const pid = this.nextId++
              this.pickups.push({ id: pid, kind: old.kind, pos: [m.pos[0], m.pos[1], m.pos[2]] })
              m.events.push({ type: "dropped", id: pid, kind: old.kind, slot })
            }
            const st = this.mergedWeaponStats(m, pu.kind)
            m.weapons[slot] = { kind: pu.kind, cooldown: Math.max(this.cfg.loot.equip_s, st.cooldown ?? 0), ammo: Math.floor(st.mag ?? 0), reload: 0, charge: 0, release: false, shortUntil: 0, equipUntil: this.t + Math.max(this.cfg.loot.equip_s, st.cooldown ?? 0), }
            this.pickups = this.pickups.filter((p) => p.id !== a.take)
            m.events.push({ type: "took", id: a.take, kind: pu.kind, slot })
          }
        }
      }
    }
    if (typeof a.drop === "number" && typeof a.take !== "number") {
      const slot = a.drop as number
      if (slot === 0 || slot === 1) {
        const old = m.weapons[slot]
        const charging = old?.kind === "laser" && old.charge > 0
        if (old && !charging) {
          const pid = this.nextId++
          this.pickups.push({ id: pid, kind: old.kind, pos: [m.pos[0], m.pos[1], m.pos[2]] })
          m.weapons[slot] = null
          m.events.push({ type: "dropped", id: pid, kind: old.kind, slot })
        }
      }
    }
  }

  // ---------- observation ----------

  canSee(v: Mech, target: Vec3, topZ: number): boolean {
    const eye: Vec3 = [v.pos[0], v.pos[1], v.pos[2]! + v.stats.eye_height!]
    const d = dist2d(v.pos, target)
    const inNear = d <= this.cfg.vision.near_range + (v.stats.near_range_bonus ?? 0)
    const fov = ((this.cfg.vision.fov_deg * Math.PI) / 180) / 2
    const inWedge = d <= this.cfg.vision.range + (v.stats.vision_bonus ?? 0) && Math.abs(angDiff(Math.atan2(target[1]! - v.pos[1]!, target[0]! - v.pos[0]!), v.aim[0])) <= fov
    if (!inNear && !inWedge) return false
    const vHit = los(this.map, eye, [target[0], target[1], target[2] ?? 0]) || los(this.map, eye, [target[0], target[1], topZ])
    if (vHit) this.visRows[v.id!] = (this.visRows[v.id!] ?? 0) + 1
    return vHit
  }

  buildObs(m: Mech): Record<string, unknown> {
    const c = this.cfg
    // visible actors + memory
    for (const e of [...this.mechs.filter((x) => x.alive && x.id !== m.id), ...this.npcs.filter((n) => n.alive)]) {
      const com: Vec3 = [e.pos[0], e.pos[1], e.pos[2]! + e.height / 2]
      const top: Vec3 = [e.pos[0], e.pos[1], e.pos[2]! + e.height]
      if (this.canSee(m, com, top[2])) {
        if (e.id === 0 && m.id !== 0) (this as unknown as { visHouse0?: Record<number, number> }).visHouse0 = ((this as unknown as { visHouse0?: Record<number, number> }).visHouse0 ?? {})
        if (e.id === 0 && m.id !== 0) {
          const vh = (this as unknown as { visHouse0?: Record<number, number> }).visHouse0!
          vh[m.id] = (vh[m.id] ?? 0) + 1
          const bucket = Math.floor(this.t / 60)
          ;(this as unknown as { visHouse0T?: Record<string, number> }).visHouse0T = ((this as unknown as { visHouse0T?: Record<string, number> }).visHouse0T ?? {})
          const vt = (this as unknown as { visHouse0T?: Record<string, number> }).visHouse0T!
          vt["m" + bucket] = (vt["m" + bucket] ?? 0) + 1
        }
        m.mem.set(e.id, { pos: [...e.pos] as Vec3, vel: [...e.vel] as Vec3, aim: [...e.aim] as [number, number], hp: e.hp, maxHp: e.maxHp, cubes: e.cubes, weapons: e.weapons.filter((w) => w).map((w) => w!.kind), charging: e.weapons.some((w) => w?.kind === "laser" && w.charge > 0), lastT: this.t, kind: e.isNpc ? "npc" : "mech" })
      }
    }
    const seen: Record<string, unknown>[] = []
    for (const [id, ent] of m.mem) {
      if (this.t - ent.lastT > c.vision.memory_s) {
        m.mem.delete(id)
        continue
      }
      seen.push({
        id, kind: ent.kind, pos: ent.pos, vel: ent.vel, aim: ent.aim,
        ...(ent.kind === "mech" && c.vision.show_enemy_hp ? { hp: Math.round(ent.hp), max_hp: Math.round(ent.maxHp), cubes: ent.cubes } : {}),
        ...(ent.kind === "mech" ? { weapons: ent.weapons, charging: ent.charging } : { hp: Math.round(ent.hp), max_hp: Math.round(ent.maxHp) }),
      })
    }
    // projectiles / crates / cubes / pickups
    const eye: Vec3 = [m.pos[0], m.pos[1], m.pos[2]! + m.stats.eye_height!]
    const fov = ((c.vision.fov_deg * Math.PI) / 180) / 2
    const inView = (p: Vec3, topZ: number): boolean => this.canSee(m, p, topZ)
    void eye
    void fov
    const projs = this.projs.filter((p) => p.ownerId !== m.id && inView(p.pos, p.pos[2]!)).map((p) => ({ id: p.id, kind: p.kind, pos: p.pos, vel: p.vel }))
    const crates = this.crates.filter((cr) => inView(cr.pos, cr.pos[2]! + c.crates.height)).map((cr) => ({ id: cr.id, pos: cr.pos, hp: Math.round(cr.hp) }))
    const cubes = this.cubes.filter((cb) => inView(cb.pos, cb.pos[2]!)).map((cb) => ({ id: cb.id, pos: cb.pos }))
    const pickups = this.pickups.filter((p) => inView(p.pos, p.pos[2]!)).map((p) => ({ id: p.id, kind: p.kind, pos: p.pos }))
    // sounds heard since last obs
    const heard: Record<string, unknown>[] = []
    for (const s of this.sounds) {
      if (s.srcId === m.id || s.t < m.lastObsT) continue
      const d = Math.hypot(s.x - m.pos[0]!, s.y - m.pos[1]!)
      const carry = c.sound.range[s.kind === "drop_called" ? "drop" : s.kind]
      if (carry === undefined || d > carry) continue
      const sigma = ((c.sound.bearing_noise_deg * Math.PI) / 180) * (m.stats.hearing ?? 1)
      const bearing = Math.atan2(s.y - m.pos[1]!, s.x - m.pos[0]!) + gauss(this.rng) * sigma
      heard.push({ kind: s.kind, bearing, dist: d < c.sound.near ? "near" : d < c.sound.mid ? "mid" : "far", age: Math.round((this.t - s.t) * 100) / 100 })
    }
    const beams = this.beams
      .filter((b) => b.bornT > m.lastObsT)
      .filter((b) => {
        const ends: Vec3[] = b.path.length >= 2 ? [b.path[0]!, b.path[b.path.length - 1]!] : [b.path[0]!]
        return ends.some((e) => dist2d(e, m.pos) <= c.vision.range && los(this.map, eye as Vec3, e))
      })
      .map((b) => ({ by: b.by, kind: b.kind, path: b.path, age: Math.round((this.t - b.bornT) * 100) / 100 }))
    const events = m.events.splice(0, m.events.length)
    m.lastObsT = this.t
    return {
      type: "obs",
      tick: this.tick,
      t: Math.round(this.t * 1000) / 1000,
      alive: this.aliveCount,
      self: {
        id: m.id,
        pos: m.pos,
        vel: m.vel,
        aim: m.aim,
        hp: Math.round(m.hp),
        max_hp: Math.round(m.maxHp),
        energy: Math.round(m.energy),
        max_energy: m.maxEnergy,
        grounded: m.grounded,
        cubes: m.cubes,
        upgrades: Object.keys(m.upgrades),
        in_zone: m.inZone,
        stats: m.stats,
        weapons: m.weapons.map((w) =>
          w ? { kind: w.kind, cooldown: Math.round(w.cooldown * 100) / 100, ammo: w.ammo, reload: Math.round(w.reload * 100) / 100, ...(w.kind === "laser" ? { charge: Math.round(w.charge * 100) / 100 } : {}), ...(w.kind === "gun" ? {} : {}), stats: this.mergedWeaponStats(m, w.kind) } : null
        ),
      },
      mechs: seen,
      projectiles: projs,
      crates,
      cubes,
      pickups,
      sounds: heard,
      beams,
      events,
      drops: this.drops.filter((d) => d.called).map((d) => ({ id: d.id, pos: d.pos, radius: d.radius, lands_in: Math.max(0, Math.round(d.landsIn * 10) / 10), capture_s: d.captureS, progress: Math.max(0, Math.round(d.progress * 10) / 10), holder: d.holder })),
      zone: {
        center: [Math.round(this.zone.cx * 10) / 10, Math.round(this.zone.cy * 10) / 10],
        radius: Math.round(this.zone.r * 10) / 10,
        next_center: [Math.round(this.next.cx * 10) / 10, Math.round(this.next.cy * 10) / 10],
        next_radius: this.next.r,
        shrink_start: Number.isFinite(this.next.shrinkStart) ? Math.round(this.next.shrinkStart * 10) / 10 : 1e9,
        shrink_end: Number.isFinite(this.next.shrinkEnd) ? Math.round(this.next.shrinkEnd * 10) / 10 : 1e9,
        dps: this.curDps,
      },
      offer: m.offer,
    }
  }
}

