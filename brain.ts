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
function takePlan(kind: string, weapons: Observation["self"]["weapons"]): { drop?: number } | null {
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
    const decision = this.decideFn(obs, this.combat, { los: this.los })
    const target = decision?.target ?? null

    // C1: decide's target carries only {id, kind} — resolve the actor from obs by id
    const targetMech = target?.kind === "mech" ? (obs.mechs.find((m) => m.id === target.id) ?? null) : null

    const a: Omit<Action, "tick"> = {}
    if (obs.offer) a.pick = pickUpgrade(obs.offer.choices)
    if (decision) {
      a.fire = decision.fire
      if (decision.reload) a.reload = true
    }

    // ----- movement goal: ladder, first match wins -----
    let goal: Vec2 | null = null
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

    // b. fight — decide() owns aim/fire. var/hunt: chase the lowest-HP tracked mech within
    // 250 m that is inside the live zone; decide's target wins when it is that mech.
    let hunted: ActorObs | null = null
    {
      let bestHp = Infinity
      for (const m of obs.mechs) {
        if (m.kind !== "mech") continue
        if (dist2d(m.pos, pos2) > 250) continue
        if (dist2d(m.pos, live.c) > live.r) continue // not inside the zone: let it rot
        const hp = m.hp ?? 1e9
        if (hp < bestHp) {
          bestHp = hp
          hunted = m
        }
      }
    }
    if (goal === null) {
      if (targetMech && dist2d(targetMech.pos, pos2) <= 200) strafeOf = targetMech
      else if (hunted && dist2d(hunted.pos, pos2) <= 250) strafeOf = hunted
      else if (hunted) {
        // hunted is inside the zone and tracked but out of the 250 ring: close the gap
        goal = [hunted.pos[0], hunted.pos[1]]
      } else if (enemy && enemyDist <= 200) strafeOf = enemy
    }
    // M5: disengage against the same mech we strafe
    const disengaging = strafeOf !== null && me.hp < 0.25 * me.max_hp && (strafeOf.hp ?? 0) > me.hp
    this.combat.disengaging = disengaging

    // e. supply drops (M6: before heal — capture restores full HP) — landed, in the live zone, uncontested
    if (goal === null && strafeOf === null) {
      for (const dr of obs.drops) {
        if (dr.lands_in > 0) continue
        if (dist2d(dr.pos, pos2) > 150) continue
        if (dist2d(dr.pos, live.c) > live.r) continue
        if (obs.mechs.some((m) => dist2d(m.pos, dr.pos) < 60)) continue
        goal = dist2d(dr.pos, pos2) <= (dr.radius || 8) ? pos2 : [dr.pos[0], dr.pos[1]]
        break
      }
    }

    // c. heal — stand still with no mech visible at all (M4); 5 s without dealing damage starts regen
    if (goal === null && strafeOf === null && enemy === null && me.hp < 0.7 * me.max_hp) {
      goal = pos2
      healing = true
    }

    // d. loot — walk to the best takeable weapon pickup within 80 m.
    // var/hunt: in the first 40 s rush the nearest laser/lightning within 250 m.
    if (goal === null && strafeOf === null && obs.t < LOOT_WINDOW_S) {
      if (obs.t < 40) {
        let rush: PickupObs | null = null
        let rushD = Infinity
        for (const p of obs.pickups) {
          if (p.kind !== "laser" && p.kind !== "lightning") continue
          if (!takePlan(p.kind, me.weapons)) continue // do not rush what we already hold
          if (dist2d(p.pos, live.c) > live.r - 5) continue // stay inside the live circle
          const d = dist2d(p.pos, pos2)
          if (d <= pickupRadius || d > 250) continue
          if (d < rushD) {
            rushD = d
            rush = p
          }
        }
        if (rush) {
          goal = [rush.pos[0], rush.pos[1]]
          boost = me.energy > 25
        }
      }
      if (goal === null) {
        const want = this.preferredPickup(obs.pickups, me.weapons, pos2, PICKUP_MAX_DIST)
        if (want && dist2d(want.pos, pos2) > pickupRadius) goal = [want.pos[0], want.pos[1]]
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
      if (tpos && dist2d(tpos, pos2) <= PVE_APPROACH_DIST)
        goal = dist2d(tpos, pos2) > PVE_STANDOFF ? [tpos[0], tpos[1]] : pos2
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
      if (cube) goal = [cube.pos[0], cube.pos[1]]
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
      const band = RANGE_BANDS[primary] ?? RANGE_BANDS.gun!
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

    // opportunistic takes in the loot window (independent of the movement branch)
    if (obs.t < LOOT_WINDOW_S) {
      const p = this.preferredPickup(obs.pickups, me.weapons, pos2, pickupRadius)
      const plan = p ? takePlan(p.kind, me.weapons) : null
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

  /** Highest-ranked pickup within maxDist that passes takePlan; nearest wins a rank tie. */
  private preferredPickup(
    pickups: PickupObs[],
    weapons: Observation["self"]["weapons"],
    pos2: Vec2,
    maxDist: number,
  ): PickupObs | null {
    let best: PickupObs | null = null
    let bestRank = 0
    let bestD = Infinity
    for (const p of pickups) {
      const d = dist2d(p.pos, pos2)
      if (d > maxDist || !takePlan(p.kind, weapons)) continue
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
