/**
 * brain.ts — per-tick priority logic (Task 4): zone > fight > supply > heal > loot > cubes > idle,
 * all composed into one action per tick.
 *
 *   export function zoneCircle(zone, t)         live zone circle — protocol §9 (linear lerp)
 *   export function pickUpgrade(choices)        offer choice index by priority — protocol §10
 *   new Brain(hello, map, los).act(obs)         → Omit<Action, "tick">
 *
 * nav.ts / combat.ts are imported statically; the optional 4th constructor arg overrides them
 * for tests.
 */
import { decide } from "./combat.ts"
import type { CombatState } from "./combat.ts"
import { feasible } from "./map.ts"
import type { GameMap } from "./map.ts"
import type {
  Action,
  ActorObs,
  Hello,
  Observation,
  PickupObs,
  SoundObs,
  Vec2,
  Vec3,
  ZoneObs,
} from "./mechfight.ts"
import { Navigator, StuckMonitor, type NavUpdateOpts } from "./nav.ts"

type LosFn = (a: Vec3, b: Vec3) => boolean
type NavLike = Pick<Navigator, "update">
type StuckLike = Pick<StuckMonitor, "update">
type DecideFn = typeof decide
type BrainCombatState = CombatState & { disengaging: boolean }

/** Test injection points; production uses nav.ts / combat.ts directly. */
export type BrainDeps = {
  navigator?: (map: GameMap) => NavLike
  /** contract: new StuckMonitor(now) */
  stuck?: (now: number) => StuckLike
  decide?: DecideFn
  /** shared counters (bot.ts owns the BOTSIG emission) */
  stats?: BrainStats
}

const LOOT_WINDOW_S = 150
/** M3: never cross the map for a pickup. */
const PICKUP_MAX_DIST = 80
/** I5: crates/NPCs worth walking to (loot window, no mech visible), and where to stop short of them. */
const PVE_APPROACH_DIST = 60
const PVE_STANDOFF = 10
/** M2: hold the stuck "perp" rotation this long (same as the burst). */
const PERP_HOLD_S = 0.5
/** Pre-position point fractions of next_radius tried until one sits on feasible ground. */
const PREP_FRACTIONS = [0.6, 0.45, 0.3, 0.2]
/** Preferred range band per fighting weapon (b. strafe radial term). */
const RANGE_BANDS: Record<string, [number, number]> = {
  gun: [50, 90],
  laser: [70, 160],
  lightning: [20, 35],
  mortar: [60, 150],
}
/** §10 priority order — first offer choice whose id CONTAINS the token wins. */
const UPGRADE_PRIORITY = [
  "plating",
  "armor",
  "nanites",
  "laser_damage",
  "laser_charge",
  "gun_rate",
  "gun_damage",
  "lightning_efficiency",
  "gyros",
  "servos",
  "reactor",
  "capacitor",
] as const
/** Special-weapon rank: laser > lightning > mortar (gun is never taken or dropped). */
const SPECIAL_RANK: Record<string, number> = { laser: 3, lightning: 2, mortar: 1 }
/** sv/duelist: fighting ranges per weapon (lightning closes to ~30, laser ~85 m). */
const DUEL_BANDS: Record<string, [number, number]> = {
  gun: [50, 90],
  laser: [75, 95],
  lightning: [25, 35],
  mortar: [60, 150],
}
/** sv/duelist upgrade order: armor > plating > nanites > gun rate/dmg > held special > rest. */
const DUEL_UPGRADE_PRIORITY = [
  "armor",
  "plating",
  "nanites",
  "gun_rate",
  "gun_damage",
  "laser_damage",
  "laser_charge",
  "lightning_efficiency",
  "gyros",
  "servos",
  "reactor",
  "capacitor",
] as const
/** sv/duelist: index of the first offer choice whose id contains a priority token. */
export function pickUpgradeDuelist(choices: { id: string }[]): number {
  for (const token of DUEL_UPGRADE_PRIORITY) {
    const idx = choices.findIndex((c) => c.id.includes(token))
    if (idx >= 0) return idx
  }
  return 0
}
/** sv/duelist: mortar is never picked up; otherwise identical to takePlan. */
function takePlanDuelist(kind: string, weapons: Observation["self"]["weapons"]): { drop?: number } | null {
  if (kind === "mortar") return null
  return takePlan(kind, weapons)
}
/** §f: idle scan is interrupted only by these heard sounds. */
const TURN_SOUND_KINDS = new Set(["gun", "laser", "lightning", "step"])

/** Live zone circle (§9): current circle, linearly lerped inside [shrink_start, shrink_end]. */
export function zoneCircle(zone: ZoneObs, t: number): { c: Vec2; r: number } {
  if (t >= zone.shrink_end) return { c: [zone.next_center[0], zone.next_center[1]], r: zone.next_radius }
  if (t <= zone.shrink_start || zone.shrink_end <= zone.shrink_start)
    return { c: [zone.center[0], zone.center[1]], r: zone.radius }
  const f = (t - zone.shrink_start) / (zone.shrink_end - zone.shrink_start)
  return {
    c: [
      zone.center[0] + f * (zone.next_center[0] - zone.center[0]),
      zone.center[1] + f * (zone.next_center[1] - zone.center[1]),
    ],
    r: zone.radius + f * (zone.next_radius - zone.radius),
  }
}

/** Index of the first offer choice whose id contains the highest-priority token; fallback 0. */
export function pickUpgrade(choices: { id: string }[]): number {
  for (const token of UPGRADE_PRIORITY) {
    const idx = choices.findIndex((c) => c.id.includes(token))
    if (idx >= 0) return idx
  }
  return 0
}

/**
 * I4: the single pickup predicate. Take `kind` iff there's an empty slot and we don't already
 * hold it, or it ranks strictly above the worst held special (then drop that special's slot).
 * Returns null when not worth taking; `drop` undefined means "goes into the empty slot".
 * A charging laser is never dropped: nothing ranks above laser.
 */
function takePlan(kind: string, weapons: Observation["self"]["weapons"], neverMortar = false): { drop?: number } | null {
  if (neverMortar && kind === "mortar") return null
  const rank = SPECIAL_RANK[kind]
  if (!rank) return null
  if (weapons.some((w) => w?.kind === kind)) return null
  if (weapons.some((w) => w === null)) return {}
  let worstSlot = -1
  let worstRank = Infinity
  weapons.forEach((w, i) => {
    const r = w ? SPECIAL_RANK[w.kind] : undefined
    if (r !== undefined && r < worstRank) {
      worstRank = r
      worstSlot = i
    }
  })
  return worstSlot >= 0 && rank > worstRank ? { drop: worstSlot } : null
}

/** mulberry32 — deterministic per-slot RNG seeded from hello.seed. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const dist2d = (a: Vec3 | Vec2, b: Vec3 | Vec2) => Math.hypot(a[0] - b[0], a[1] - b[1])

/** Behavior counters (Task 10) shared with bot.ts, emitted as BOTSIG on stderr at exit. */
export type BrainStats = {
  dodges: number
  retreats: number
  coverTicks: number
  seen: Record<string, number>
  seenDist: Record<string, number>
  walked: number
  took: Record<string, number>
  fail: Record<string, number>
}

export const newBrainStats = (): BrainStats => ({
  dodges: 0,
  retreats: 0,
  coverTicks: 0,
  seen: {},
  seenDist: {},
  walked: 0,
  took: {},
  fail: {},
})

export class Brain {
  private readonly hello: Hello
  private readonly map: GameMap
  private readonly los: LosFn
  private readonly rng: () => number
  private readonly deps?: BrainDeps
  /** Shared combat state handed to decide() every tick; combat owns lastDamaged entirely. */
  private readonly combat: BrainCombatState
  private readonly nav: NavLike
  private readonly decideFn: DecideFn
  /** created on the first act() so its clock starts at the first observation's t */
  private stuck: StuckLike | null = null
  private strafeSign: number
  private strafeFlipAt: number
  /** Task 10 counters (filled by variant brains; diagnostics by every brain) */
  readonly stats: BrainStats
  private pkIds = new Set<number>()
  private pkBlack: Map<number, number> = new Map()
  private lootWatch: { id: number; t: number; d0: number } | null = null
  /** sv/duelist state */
  private pkMem = new Map<number, { pos: Vec3; kind: string; t: number }>()
  private duelId = -1
  private duelLostT = -10
  private lastSeenEnemyT = -10
  private lastDodgeT = -10
  private rushT = -10
  private rushGoalB: { pos: Vec3; kind: string } | null = null
  private killLoot: { pos: Vec3; t: number } | null = null
  private duelVictimPos: Vec3 | null = null
  private coverMode = false
  private coverCell: Vec2 | null = null
  private coverReseekT = -10
  private dmgLog: { t: number; amt: number }[] = []
  private lastAttacker: Vec3 | null = null
  private lastAttackerT = -10

  private burstUntil = -1
  private perpUntil = -1
  private perpSign = 1

  constructor(hello: Hello, map: GameMap, los: LosFn, deps?: BrainDeps) {
    this.hello = hello
    this.map = map
    this.los = los
    this.deps = deps
    this.rng = mulberry32(hello.seed)
    this.strafeSign = this.rng() < 0.5 ? -1 : 1
    this.strafeFlipAt = 0.6 + this.rng() * 0.8
    this.nav = deps?.navigator?.(map) ?? new Navigator(map)
    this.decideFn = deps?.decide ?? decide
    this.combat = {
      gravity: hello.config.weapons?.mortar?.gravity ?? 22,
      lastDamaged: null,
      disengaging: false,
    }
    this.stats = deps?.stats ?? newBrainStats()
  }

  /** act() deadline context, passed by bot.ts: elapsed wall time and a force-replan switch */
  actCtx: { elapsedMs?: () => number; forceReplan?: boolean } | undefined = undefined

  act(obs: Observation, ctx?: { elapsedMs?: () => number; forceReplan?: boolean }): Omit<Action, "tick"> {
    this.actCtx = ctx
    this.stuck ??= this.deps?.stuck?.(obs.t) ?? new StuckMonitor(obs.t)
    const me = obs.self
    const pos = me.pos
    const pos2: Vec2 = [pos[0], pos[1]]
    const live = zoneCircle(obs.zone, obs.t)
    const walkSpeed = me.stats.walk_speed || 11.7
    const pickupRadius = this.hello.config.loot.pickup_radius || 4

    // Task 10 diagnostics: pickups seen once by id (with first-sight distance), take results
    for (const p of obs.pickups) {
      if (!this.pkIds.has(p.id)) {
        this.pkIds.add(p.id)
        this.stats.seen[p.kind] = (this.stats.seen[p.kind] ?? 0) + 1
        this.stats.seenDist[p.kind] = (this.stats.seenDist[p.kind] ?? 0) + Math.round(dist2d(p.pos, pos2))
      }
    }
    for (const ev of obs.events) {
      if (ev.type === "take_failed") this.stats.fail[ev.reason] = (this.stats.fail[ev.reason] ?? 0) + 1
      else if (ev.type === "took") this.stats.took[ev.kind] = (this.stats.took[ev.kind] ?? 0) + 1
    }

    // sv/duelist: remember every laser/lightning pickup ever seen; go back for it later
    const seenIds = new Set<number>()
    for (const p of obs.pickups) {
      seenIds.add(p.id)
      if (p.kind === "laser" || p.kind === "lightning") {
        if (!this.pkMem.has(p.id)) {
          this.pkMem.set(p.id, { pos: [...p.pos] as Vec3, kind: p.kind, t: obs.t })
        }
      }
    }
    for (const [id, ent] of this.pkMem) {
      // not visible now: drop the memory once we are close enough that we would certainly see it
      // (someone looted it) — otherwise the rush walks to a ghost for the whole early game
      if (!seenIds.has(id) && dist2d(ent.pos, pos2) < 20) {
        this.pkMem.delete(id)
      }
    }
    for (const ev of obs.events) {
      if (ev.type === "kill" && this.duelVictimPos) this.killLoot = { pos: [...this.duelVictimPos] as Vec3, t: obs.t }
    }
    // sv/cover bookkeeping: damage in the last 3 s, last attacker, enemies with LOS to us
    this.dmgLog = this.dmgLog.filter((e) => obs.t - e.t <= 3)
    for (const ev of obs.events) {
      if (ev.type === "damaged" && ev.bearing !== null) {
        if (ev.weapon !== "npc") this.dmgLog.push({ t: obs.t, amt: ev.amount })
        if (ev.by != null) {
          const src = obs.mechs.find((m) => m.id === ev.by)
          this.lastAttacker = src
            ? ([...src.pos] as Vec3)
            : ([pos[0] + Math.cos(ev.bearing) * 30, pos[1] + Math.sin(ev.bearing) * 30, pos[2]] as Vec3)
          this.lastAttackerT = obs.t
        }
      }
    }
    const dmg3s = this.dmgLog.reduce((sum, e) => sum + e.amt, 0)
    const eye = [pos[0], pos[1], pos[2] + (this.hello.config.mech.eye_height ?? 8)] as Vec3
    const watchers: Vec3[] = []
    for (const m of obs.mechs) {
      if (m.kind !== "mech") continue
      if (this.los(eye, m.pos)) watchers.push([...m.pos] as Vec3)
    }
    if (watchers.length > 0) this.lastSeenEnemyT = obs.t

    let enemy: ActorObs | null = null
    let enemyDist = Infinity
    for (const m of obs.mechs) {
      if (m.kind !== "mech") continue
      const d = dist2d(m.pos, pos2)
      if (d < enemyDist) {
        enemy = m
        enemyDist = d
      }
    }

    const primary = me.weapons[1]?.kind ?? me.weapons[0]?.kind ?? "gun"
    const _decision = this.decideFn(obs, this.combat, { los: this.los })
    // sv/duelist: stick to the enemy we engaged (duelId) while it stays visible — combat's
    // weakest-first re-pick must not peel us onto the freshly wounded neighbor
    let decision = _decision
    const duelMech =
      this.duelId >= 0 ? (obs.mechs.find((m) => m.id === this.duelId && m.kind === "mech") ?? null) : null
    if (duelMech) {
      this.lastSeenEnemyT = obs.t
      this.duelLostT = obs.t
      this.duelVictimPos = [...duelMech.pos] as Vec3
      if (_decision?.target && _decision.target.kind === "mech" && _decision.target.id !== this.duelId) {
        const only = { ...obs, mechs: [duelMech] }
        decision = this.decideFn(only, this.combat, { los: this.los })
      }
    } else if (this.duelId >= 0) {
      if (obs.t - this.duelLostT > 3) this.duelId = -1
    }
    if (_decision?.target && _decision.target.kind === "mech" && this.duelId < 0) {
      this.duelId = _decision.target.id
      this.duelLostT = obs.t
    }
    const target = decision?.target ?? null

    // C1: decide's target carries only {id, kind} — resolve the actor from obs by id
    const targetMech = target?.kind === "mech" ? (obs.mechs.find((m) => m.id === target.id) ?? null) : null

    const a: Omit<Action, "tick"> = {}
    if (obs.offer) a.pick = pickUpgradeDuelist(obs.offer.choices)
    if (decision) {
      a.fire = decision.fire
      if (decision.reload) a.reload = true
    }

    // ----- movement goal: ladder, first match wins -----
    let goal: Vec2 | null = null
    let farming = false
    let boost = false
    let silent = false
    let strafeOf: ActorObs | null = null
    let healing = false

    // a. zone — never die to it
    if (dist2d(pos, live.c) > live.r) {
      goal = [live.c[0], live.c[1]]
      boost = true
    } else {
      const dNext = Math.hypot(pos[0] - obs.zone.next_center[0], pos[1] - obs.zone.next_center[1]) - obs.zone.next_radius
      const tToStart = obs.zone.shrink_start - obs.t
      if (dNext > 0 && tToStart <= dNext / walkSpeed + 10) goal = this.prePositionPoint(pos2, obs.zone)
    }

    // a2. sv/duelist — arm up first while nobody is shooting at us: rush the nearest remembered
    // laser/lightning (any distance — pickups leave vision, memory does not)
    const holdsL = me.weapons.some((w) => w?.kind === "laser")
    const holdsLg = me.weapons.some((w) => w?.kind === "lightning")
    const quiet = dmg3s === 0 && obs.tick - (this.combat.lastDamaged?.tObsTick ?? -999) > 60
    if (goal === null && obs.t < 120 && quiet && (!holdsL || !holdsLg)) {
      const wantKind = !holdsL ? "laser" : "lightning"
      let bp: { pos: Vec3; kind: string } | null = null
      let bd = Infinity
      for (const ent of this.pkMem.values()) {
        if (me.weapons.some((w) => w?.kind === ent.kind)) continue
        const d = dist2d(ent.pos, pos2)
        if (d > 150) continue
        if (ent.kind === wantKind ? d < bd : d + 40 < bd) {
          bd = d
          bp = ent
        }
      }
      if (!bp) {
        // fall back to a currently visible special pickup even beyond the 80 m loot cap
        for (const p of obs.pickups) {
          if (p.kind !== "laser" && p.kind !== "lightning") continue
          const d = dist2d(p.pos, pos2)
          if (d > 150 || (me.weapons.some((w) => w?.kind === p.kind) && this.pkMem.size === 0)) continue
          if (this.pkMem.has(p.id) === false && me.weapons.some((w) => w?.kind === p.kind)) continue
          if (d < bd) {
            bd = d
            bp = { pos: [...p.pos] as Vec3, kind: p.kind }
          }
        }
      }
      if (bp) {
        if (this.rushT < 0 || this.rushGoalB !== bp) {
          this.rushT = obs.t
          this.rushGoalB = bp
        } else if (obs.t - this.rushT > 12) {
          // did not converge in 12 s: ghost or unreachable — forget it and go back to farming
          for (const [id, ent] of this.pkMem) if (ent === bp) this.pkMem.delete(id)
          this.rushT = -10
          this.rushGoalB = null
          bp = null
        }
        if (bp) {
          goal = [bp.pos[0], bp.pos[1]]
          if (dist2d(bp.pos, pos2) > 40 && me.energy > 30) boost = true
          this.stats.walked++
          ;(this.stats as Record<string, number>).rushFired = ((this.stats as Record<string, number>).rushFired ?? 0) + 1
        }
      } else {
        bp = null
      }
    }
    // a1. sv/cover — refuse bad fights (2+ watchers / hp<45% / >300 dmg in 3 s) until hp >= 85%
    if (!this.coverMode) {
      if (watchers.length >= 2 || me.hp < 0.45 * me.max_hp || dmg3s > 300) {
        this.coverMode = true
        this.stats.retreats++
        this.coverCell = null
        this.coverReseekT = -10
      }
    } else {
      // exit when healed OR when nobody has watched us for 8 s — the bad fight broke up and
      // keeping cover mode would starve farming/cubes forever (staying out is also what makes
      // the regen the brief asks for actually happen)
      if (me.hp >= 0.85 * me.max_hp || (watchers.length === 0 && obs.t - this.lastAttackerT > 8))
        this.coverMode = false
    }
    if (this.coverMode) this.stats.coverTicks++
    if (this.coverMode && goal === null) {
      const badCell =
        this.coverCell === null ||
        !feasible(this.map, this.coverCell[0]!, this.coverCell[1]!) ||
        dist2d(this.coverCell, live.c) > live.r - 2
      if (badCell || obs.t - this.coverReseekT > 2) {
        this.coverCell = this.findCoverCell(pos2, watchers, live)
        this.coverReseekT = obs.t
      }
      goal = this.coverCell ?? ([live.c[0], live.c[1]] as Vec2)
    }

    // a2.5 sv/duelist — loot the corpse: a killed mech drops its weapon (loot.death_drop), and
    // expert house bots carry the very lasers/lightnings we want — walk onto the kill spot
    if (goal === null && this.killLoot && obs.t - this.killLoot.t < 10 && (!holdsL || !holdsLg)) {
      const dk = dist2d(this.killLoot.pos, pos2)
      if (dk < 80) {
        goal = [this.killLoot.pos[0], this.killLoot.pos[1]]
        farming = true
        if (dk > 50 && me.energy > 40) boost = true
        this.stats.walked++
      }
    }

    // b. fight — decide() owns aim/fire; strafe the mech it targets (else the nearest) within 200 m
    if (goal === null) {
      if (targetMech && dist2d(targetMech.pos, pos2) <= 200) strafeOf = targetMech
      else if (enemy && enemyDist <= 200) strafeOf = enemy
    }
    // M5: disengage against the same mech we strafe
    const disengaging = strafeOf !== null && me.hp < 0.3 * me.max_hp && (strafeOf.hp ?? 0) > me.hp
    this.combat.disengaging = disengaging

    // c. heal — stand still with no mech visible at all (M4); 5 s without dealing damage starts regen
    if (goal === null && strafeOf === null && enemy === null && me.hp < 0.7 * me.max_hp) {
      goal = pos2
      healing = true
    }

    // d. loot — walk to the best takeable weapon pickup within 80 m
    if (goal === null && strafeOf === null && obs.t < LOOT_WINDOW_S) {
      const want = this.preferredPickup(obs.pickups, me.weapons, pos2, PICKUP_MAX_DIST, true)
      if (want && !this.stallGuard(want.id, dist2d(want.pos, pos2), obs)) {
        if (dist2d(want.pos, pos2) > pickupRadius) goal = [want.pos[0], want.pos[1]]
        else {
          goal = pos2
          farming = true
        }
      }
      if (goal !== null) {
        farming = true
        this.stats.walked++
      }
    }

    // d0. sv/duelist — arm-up window: with a special still missing and nobody shooting, go break
    // the nearest crate first: crates roll weapon drops (crate_weapon_chance), drones only give cubes.
    // Stands down when a takeable special pickup is visible — walking to it outranks breaking crates.
    const armUp = obs.t < 120 && quiet && (!holdsL || !holdsLg)
    const specialVisible = obs.pickups.some(
      (p) => (p.kind === "laser" || p.kind === "lightning") && takePlanDuelist(p.kind, me.weapons),
    )
    if (goal === null && strafeOf === null && enemy === null && obs.t < LOOT_WINDOW_S && armUp && !specialVisible) {
      let crate: Observation["crates"][number] | null = null
      let crateD = Infinity
      for (const c of obs.crates) {
        const d = dist2d(c.pos, pos2)
        if (d < crateD) {
          crate = c
          crateD = d
        }
      }
      // but a visible drop (a broken crate just rolled a weapon) outranks the next crate
      let drop: PickupObs | null = null
      let dropD = Infinity
      for (const p of obs.pickups) {
        if (p.kind !== "laser" && p.kind !== "lightning") continue
        const d = dist2d(p.pos, pos2)
        if (d < dropD && (d < 0.9 * crateD || crateD === Infinity)) {
          dropD = d
          drop = p
        }
      }
      if (drop) {
        goal = [drop.pos[0], drop.pos[1]]
        farming = true
      } else if (crate) {
        goal = [crate.pos[0], crate.pos[1]]
        farming = true
        if (obs.t < 90 && crateD > 60 && me.energy > 40) boost = true
      }
    }

    // d. crates/NPCs — loot window, no mech visible: approach decide's crate/NPC target within 60 m
    if (goal === null && strafeOf === null && enemy === null && obs.t < LOOT_WINDOW_S && target) {
      const tpos =
        target.kind === "crate"
          ? obs.crates.find((c) => c.id === target.id)?.pos
          : target.kind === "npc"
            ? obs.mechs.find((m) => m.id === target.id)?.pos
            : undefined
      if (tpos && dist2d(tpos, pos2) <= PVE_APPROACH_DIST) {
        farming = true
        if (target.kind === "npc") {
          // sv/duelist: shoot drones from ~60 m, never close into their gun range
          const d = Math.max(0.001, dist2d(tpos, pos2))
          if (d > 60) goal = [tpos[0], tpos[1]]
          else if (d < 55) {
            const ux = (pos2[0] - tpos[0]) / d
            const uy = (pos2[1] - tpos[1]) / d
            goal = [tpos[0] + ux * 60, tpos[1] + uy * 60]
          } else goal = pos2
        } else goal = dist2d(tpos, pos2) > PVE_STANDOFF ? [tpos[0], tpos[1]] : pos2
      }
    }

    // d+. cubes — nearest visible cube inside the live zone (never trade zone safety for it)
    if (goal === null && strafeOf === null) {
      let cube: Observation["cubes"][number] | null = null
      let cubeDist = Infinity
      for (const c of obs.cubes) {
        if (dist2d(c.pos, live.c) > live.r - 5) continue
        const d = dist2d(c.pos, pos2)
        if (d < cubeDist) {
          cube = c
          cubeDist = d
        }
      }
      if (cube) {
        goal = [cube.pos[0], cube.pos[1]]
        farming = true
      }
    }

    // d2. sv/duelist — uncontested supply drops, after cubes (capture restores full HP)
    if (goal === null && strafeOf === null) {
      for (const dr of obs.drops) {
        if (dr.lands_in > 0) continue
        if (dist2d(dr.pos, pos2) > 150) continue
        if (dist2d(dr.pos, live.c) > live.r) continue
        if (dr.holder !== null && dr.holder !== me.id) continue
        if (obs.mechs.some((m) => m.kind === "mech" && dist2d(m.pos, dr.pos) < 60)) continue
        goal = dist2d(dr.pos, pos2) <= (dr.radius || 8) ? pos2 : [dr.pos[0], dr.pos[1]]
        break
      }
    }


    // f. idle — walk silently to the next circle and scan
    if (goal === null && strafeOf === null) {
      goal = [obs.zone.next_center[0], obs.zone.next_center[1]]
      silent = true
    }

    // I5: a visible crate/NPC must not break the 5 s no-damage regen
    if (healing && target?.kind !== "mech") a.fire = [false, false]

    // ----- movement through the Navigator (single call per tick) -----
    let navGoal: Vec2
    if (strafeOf) {
      let band = DUEL_BANDS[primary] ?? DUEL_BANDS.gun!
      // sv/duelist: an enemy holding lightning outranges us unless we hold it too — stay > 45 m
      if (strafeOf.kind === "mech" && strafeOf.weapons?.includes("lightning") && primary !== "lightning")
        band = [46, Math.max(90, band[1])]
      if (obs.t >= this.strafeFlipAt) {
        this.strafeSign *= -1
        this.strafeFlipAt = obs.t + 0.6 + this.rng() * 0.8
      }
      const d = Math.max(0.001, dist2d(strafeOf.pos, pos2))
      const ux = (strafeOf.pos[0] - pos[0]) / d
      const uy = (strafeOf.pos[1] - pos[1]) / d
      let vx = -uy * this.strafeSign
      let vy = ux * this.strafeSign
      if (disengaging) {
        vx = -ux
        vy = -uy
      } else if (d < band[0]) {
        vx -= ux
        vy -= uy
      } else if (d > band[1]) {
        vx += ux
        vy += uy
      }
      const l = Math.hypot(vx, vy) || 1
      navGoal = [pos[0] + (vx / l) * 20, pos[1] + (vy / l) * 20]
      // sv/duelist: peek — a shot ready means keep LOS (the band strafe does); while
      // reloading or on laser cooldown, step behind cover instead
      const wGun = me.weapons[0]
      const wSpec = me.weapons[1]
      const peekDown = (wGun?.reload ?? 0) > 0.2 || (primary === "laser" && (wSpec?.cooldown ?? 0) > 0.6)
      if (peekDown) {
        for (let k = 0; k < 8; k++) {
          const ang = (k / 8) * Math.PI * 2
          const px = pos[0] + Math.cos(ang) * 7
          const py = pos[1] + Math.sin(ang) * 7
          if (!feasible(this.map, px, py)) continue
          if (this.los([px, py, pos[2]! + 6], [strafeOf.pos[0], strafeOf.pos[1], strafeOf.pos[2]! + 6])) continue
          navGoal = [px, py]
          break
        }
      }
      boost = me.energy > 50 && primary !== "lightning"
    } else {
      navGoal = goal ?? pos2
    }
    // act() deadline (Task 3): A* only when act is still under 10 ms of wall time, and then
    // with the cap dropped to 10 ms; otherwise reuse the last path (replan: false)
    const elapsedMs = this.actCtx?.elapsedMs?.()
    let navOpts: NavUpdateOpts | undefined
    if (elapsedMs !== undefined) {
      navOpts = elapsedMs < 10 ? { capMs: 10, ...(this.actCtx?.forceReplan ? { force: true } : {}) } : { replan: false }
    } else if (this.actCtx?.forceReplan) {
      navOpts = { force: true }
    }
    const nav = this.nav.update(pos2, navGoal, obs.t, navOpts)
    let move: Vec2 = nav.atGoal ? [0, 0] : [nav.dir[0], nav.dir[1]]

    // Stuck handling: burst = hop+boost 0.5 s, perp = rotate the command 0.5 s (I1, M2)
    const commanding = move[0] !== 0 || move[1] !== 0
    const verdict = this.stuck.update(obs.t, Math.hypot(me.vel[0], me.vel[1]), commanding)
    if (verdict === "burst") this.burstUntil = obs.t + 0.5
    if (verdict === "perp") {
      this.perpUntil = obs.t + PERP_HOLD_S
      this.perpSign = this.strafeSign
    }
    if (commanding && obs.t <= this.perpUntil) move = [-move[1] * this.perpSign, move[0] * this.perpSign]
    if (obs.t <= this.burstUntil) {
      a.jump = true
      boost = true
    }
    if (silent && commanding) {
      // I3: stay under sound.step_speed (servos raise walk_speed, so derive it per tick)
      const frac = Math.min(1, (0.9 * (this.hello.config.sound?.step_speed ?? 5)) / walkSpeed)
      const l = Math.hypot(move[0], move[1])
      if (l > frac) move = [(move[0] / l) * frac, (move[1] / l) * frac]
    }
    // sv/duelist: sneak at <= 4.6 m/s when enemies are known but nobody is shooting at us
    if (!strafeOf && !farming && obs.t - this.lastSeenEnemyT < 10 && obs.t - this.lastAttackerT > 5 && this.dmgLog.length === 0) {
      const frac = Math.min(1, 4.6 / walkSpeed)
      const l = Math.hypot(move[0], move[1])
      if (l > frac) move = [(move[0] / l) * frac, (move[1] / l) * frac]
    }
    // Task 10 diag: activity split (temp counters)
    {
      const _st2 = this.stats as Record<string, number>
      if (strafeOf) _st2.fightTicks = (_st2.fightTicks ?? 0) + 1
      else if (farming) _st2.farmTicks = (_st2.farmTicks ?? 0) + 1
      else if (goal !== null) _st2.otherGoalTicks = (_st2.otherGoalTicks ?? 0) + 1
      else _st2.idleTicks = (_st2.idleTicks ?? 0) + 1
    }
    // sv/dodge: sidestep perpendicular to any threat about to land on us (movement only)
    const dodge = this.dodgeDir(obs, pos)
    if (dodge) {
      move = dodge
      if (obs.t - this.lastDodgeT > 1) {
        this.lastDodgeT = obs.t
        this.stats.dodges++
      }
    }

    a.move = move
    if (boost && me.energy > 0) a.boost = true

    // I2: decide's aim counts only when it has a target, or (no target) it differs from our actual
    // aim — that's combat's turn toward an unseen attacker. Otherwise strafe-aim / scan / sound-turn.
    const decideHasOpinion =
      decision !== null &&
      (target !== null ||
        Math.abs(decision.aim[0] - me.aim[0]) > 1e-9 ||
        Math.abs(decision.aim[1] - me.aim[1]) > 1e-9)
    if (decideHasOpinion) a.aim = decision!.aim
    else a.aim = strafeOf ? this.aimAt(pos, strafeOf.pos) : this.scanAim(obs)

    // sv/cover: while retreating, keep facing the last attacker
    if (this.coverMode && this.lastAttacker && obs.t - this.lastAttackerT < 6)
      a.aim = this.aimAt(pos, this.lastAttacker)

    // opportunistic takes in the loot window (independent of the movement branch)
    if (obs.t < LOOT_WINDOW_S) {
      const p = this.preferredPickup(obs.pickups, me.weapons, pos2, pickupRadius, true)
      const plan = p ? takePlanDuelist(p.kind, me.weapons) : null

      if (p && plan) {
        a.take = p.id
        if (plan.drop !== undefined) a.drop = plan.drop
      }
    }
    return a
  }

  // ---------- internals ----------

  /** A point at 0.6 × next_radius from next_center toward us, backed off until feasible. */
  private prePositionPoint(pos2: Vec2, zone: ZoneObs): Vec2 {
    const dx = pos2[0] - zone.next_center[0]
    const dy = pos2[1] - zone.next_center[1]
    const d = Math.hypot(dx, dy) || 1
    const ux = dx / d
    const uy = dy / d
    for (const f of PREP_FRACTIONS) {
      const r = f * zone.next_radius
      const x = zone.next_center[0] + ux * r
      const y = zone.next_center[1] + uy * r
      if (feasible(this.map, x, y)) return [x, y]
    }
    return [zone.next_center[0], zone.next_center[1]]
  }

  /**
   * sv/dodge: unit sidestep direction when a threat is about to hit us, else null.
   * Priority: mortar shell landing within 9 m (radial out) > visible projectile path within
   * 3 m in the next 0.6 s (perpendicular) > a charging laser whose aim line passes within
   * 4 m of us (perpendicular to the line).
   */
  private dodgeDir(obs: Observation, pos: Vec3): Vec2 | null {
    const g = this.hello.config.weapons?.mortar?.gravity ?? 22
    for (const p of obs.projectiles) {
      if (p.kind === "mortar" && p.vel[2] > 0.5) {
        const T = (2 * p.vel[2]) / g // shell returns to launch height
        const lx = p.pos[0] + p.vel[0] * T
        const ly = p.pos[1] + p.vel[1] * T
        const d = Math.hypot(pos[0] - lx, pos[1] - ly)
        if (d < 9) {
          const l = Math.max(0.001, d)
          return [(pos[0] - lx) / l, (pos[1] - ly) / l]
        }
      }
    }
    for (const p of obs.projectiles) {
      const sp = Math.hypot(p.vel[0], p.vel[1])
      if (sp < 1) continue
      const t = Math.max(0, Math.min(0.6, ((pos[0] - p.pos[0]) * p.vel[0] + (pos[1] - p.pos[1]) * p.vel[1]) / (sp * sp)))
      const cx = p.pos[0] + p.vel[0] * t
      const cy = p.pos[1] + p.vel[1] * t
      const d = Math.hypot(pos[0] - cx, pos[1] - cy)
      if (d < 3) {
        const side = Math.sign((pos[0] - cx) * (p.vel[1] / sp) - (pos[1] - cy) * (p.vel[0] / sp)) || 1
        return [-(p.vel[1] / sp) * side, (p.vel[0] / sp) * side]
      }
    }
    for (const m of obs.mechs) {
      if (m.kind !== "mech" || !m.charging) continue
      const dx = pos[0] - m.pos[0]
      const dy = pos[1] - m.pos[1]
      const d = Math.hypot(dx, dy)
      if (d > 200 || d < 1) continue
      const rx = Math.cos(m.aim[0])
      const ry = Math.sin(m.aim[0])
      const t = dx * rx + dy * ry
      if (t <= 0) continue
      if (Math.abs(dx * ry - dy * rx) < 4) {
        const side = Math.sign(dx * ry - dy * rx) || 1
        return [-ry * side, rx * side]
      }
    }
    return null
  }

  /**
   * sv/cover: nearest walkable cell around us that no tracked enemy has LOS to, inside the
   * live circle. Ring search outward (6..36 m), hard 3 ms cap; null -> zone-center fallback.
   */
  private findCoverCell(pos2: Vec2, enemies: Vec3[], live: { c: Vec2; r: number }): Vec2 | null {
    const t0 = performance.now()
    for (let ring = 6; ring <= 36; ring += 6) {
      for (let k = 0; k < 12; k++) {
        if (k % 3 === 0 && performance.now() - t0 > 3) return null
        const ang = (k / 12) * Math.PI * 2
        const x = pos2[0] + Math.cos(ang) * ring
        const y = pos2[1] + Math.sin(ang) * ring
        const cd: Vec3 = [x, y, 6]
        if (dist2d(cd, live.c) > live.r - 5) continue
        if (!feasible(this.map, x, y)) continue
        let blocked = false
        for (const e of enemies) {
          const ez: Vec3 = [e[0], e[1], e[2] + (this.hello.config.mech.eye_height ?? 8)]
          if (this.los(cd, ez)) {
            blocked = true
            break
          }
        }
        if (!blocked) return [x, y]
      }
    }
    return null
  }

  /** sv/duelist: blacklist a pickup whose walk-to goal stalls 3 s without closing ~5 m. */
  private stallGuard(id: number, d: number, obs: Observation): boolean {
    if ((this.pkBlack.get(id) ?? 0) > obs.t) return true
    const w = this.lootWatch
    if (w === null || w.id !== id) {
      this.lootWatch = { id, t: obs.t, d0: d }
    } else if (obs.t - w.t > 3) {
      if (d > w.d0 - 5) {
        this.pkBlack.set(id, Date.now() + 20000)
      }
      this.lootWatch = null
    }
    return false
  }

  /** Highest-ranked pickup within maxDist that passes takePlan; nearest wins a rank tie. */
  private preferredPickup(
    pickups: PickupObs[],
    weapons: Observation["self"]["weapons"],
    pos2: Vec2,
    maxDist: number,
    neverMortar = false,
  ): PickupObs | null {
    let best: PickupObs | null = null
    let bestRank = 0
    let bestD = Infinity
    for (const p of pickups) {
      const d = dist2d(p.pos, pos2)
      if ((this.pkBlack.get(p.id) ?? 0) > Date.now()) continue
      if (d > maxDist || !takePlan(p.kind, weapons, neverMortar)) continue
      const r = SPECIAL_RANK[p.kind]!
      if (r > bestRank || (r === bestRank && d < bestD)) {
        best = p
        bestRank = r
        bestD = d
      }
    }
    return best
  }

  private aimAt(from: Vec3, to: Vec3): Vec2 {
    const dx = to[0] - from[0]
    const dy = to[1] - from[1]
    const d = Math.max(0.001, Math.hypot(dx, dy))
    const eye = from[2] + (this.hello.config.mech.eye_height ?? 8)
    return [Math.atan2(dy, dx), Math.atan2(to[2] + 4.5 - eye, d)]
  }

  /** Newest heard weapon/step sound → turn to it; otherwise sweep ±90° around the zone-center bearing at ~1.5 rad/s. */
  private scanAim(obs: Observation): Vec2 {
    let heard: SoundObs | null = null
    for (const s of obs.sounds) {
      if (!TURN_SOUND_KINDS.has(s.kind)) continue
      if (!heard || s.age < heard.age) heard = s
    }
    if (heard) return [heard.bearing, 0]
    const c = obs.zone.center
    const base = Math.atan2(c[1] - obs.self.pos[1], c[0] - obs.self.pos[0])
    return [base + (Math.PI / 2) * Math.sin(obs.t * 1.5), 0]
  }
}
