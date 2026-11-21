/**
 * brain.test.ts — brain module tests (Task 4). Unit tests stub nav/stuck/decide through BrainDeps
 * using the REAL contract shapes (decide → {aim, fire, reload, target: {id, kind} | null});
 * the integration test at the bottom runs the real decide + Navigator + StuckMonitor.
 */
import { expect, test } from "bun:test"

import { Brain, pickUpgrade, zoneCircle, type BrainDeps } from "../brain.ts"
import type { CombatDecision, CombatState } from "../combat.ts"
import type { GameMap } from "../map.ts"
import type { ActorObs, Hello, Observation, Vec2 } from "../mechfight.ts"

// ---------- fixtures ----------

const mkHello = (seed = 42): Hello =>
  ({
    type: "hello",
    protocol: "mechfight/1",
    slot: 0,
    seed,
    map: { dir: ".", files: [] },
    config: {
      weapons: { mortar: { gravity: 22 } },
      sound: { step_speed: 5 },
      loot: { pickup_radius: 4 },
      mech: { eye_height: 8 },
    },
  }) as unknown as Hello

const mkEnemy = (pos: [number, number, number], hp = 900, id = 5): ActorObs => ({
  id,
  kind: "mech",
  pos,
  vel: [0, 0, 0],
  aim: [0, 0],
  hp,
  max_hp: 1400,
  weapons: ["gun"],
})

const GUN = { kind: "gun" as const, cooldown: 0, ammo: 12, reload: 0, stats: {} }

const mkObs = (over: {
  t?: number
  self?: Partial<Observation["self"]>
  mechs?: ActorObs[]
  zone?: Partial<Observation["zone"]>
  pickups?: Observation["pickups"]
  sounds?: Observation["sounds"]
  offer?: Observation["offer"]
  crates?: Observation["crates"]
  drops?: Observation["drops"]
  events?: Observation["events"]
}): Observation => ({
  type: "obs",
  tick: 1000,
  t: over.t ?? 10,
  alive: 16,
  self: {
    id: 0,
    pos: [0, 0, 0],
    vel: [0, 0, 0],
    aim: [0, 0],
    hp: 1400,
    max_hp: 1400,
    energy: 100,
    max_energy: 100,
    grounded: true,
    cubes: 0,
    upgrades: [],
    weapons: [GUN, null],
    in_zone: true,
    stats: { walk_speed: 11.7, boost_speed: 19.5, eye_height: 8 },
    ...over.self,
  },
  mechs: over.mechs ?? [],
  projectiles: [],
  crates: over.crates ?? [],
  cubes: [],
  pickups: over.pickups ?? [],
  sounds: over.sounds ?? [],
  beams: [],
  events: over.events ?? [],
  zone: {
    center: [0, 0],
    radius: 560,
    next_center: [0, 0],
    next_radius: 400,
    shrink_start: 600,
    shrink_end: 650,
    dps: 20,
    ...over.zone,
  },
  offer: over.offer ?? null,
  drops: over.drops ?? [],
})

type StubDecide = (obs: Observation, state: CombatState, opts: unknown) => CombatDecision | null

/** Real no-target shape: aim = self.aim, nothing to fire. */
const noTarget: StubDecide = (obs) => ({
  aim: [obs.self.aim[0], obs.self.aim[1]],
  fire: [false, false],
  reload: false,
  target: null,
})

/** Real targeted shape: target carries only {id, kind} — no pos. */
const targeting =
  (id: number, kind: "mech" | "npc" | "crate", aim: Vec2 = [1.1, -0.2], fire: [boolean, boolean] = [true, false]): StubDecide =>
  () => ({ aim, fire, reload: false, target: { id, kind } })

type StuckVerdict = "none" | "burst" | "perp"

/** Straight-line stub Navigator: dir = unit(goal − pos), atGoal inside 0.5 m. */
const stubDeps = (
  decideFn: StubDecide = noTarget,
  stuck: (now: number, speed: number, commanding: boolean) => StuckVerdict = () => "none",
): BrainDeps => ({
  navigator: (_map: GameMap) => ({
    update(pos: Vec2, goal: Vec2, _now: number) {
      const dx = goal[0] - pos[0]
      const dy = goal[1] - pos[1]
      const d = Math.hypot(dx, dy)
      if (d < 0.5) return { dir: [0, 0] as Vec2, atGoal: true }
      return { dir: [dx / d, dy / d] as Vec2, atGoal: false }
    },
  }),
  stuck: (_now: number) => ({ update: stuck }),
  decide: decideFn as BrainDeps["decide"],
})

// ---------- zoneCircle ----------

const LERP_ZONE: Observation["zone"] = {
  center: [0, 0],
  radius: 560,
  next_center: [10, 0],
  next_radius: 380,
  shrink_start: 60,
  shrink_end: 110,
  dps: 20,
}

test("zoneCircle: before shrink_start the circle holds", () => {
  expect(zoneCircle(LERP_ZONE, 0)).toEqual({ c: [0, 0], r: 560 })
  expect(zoneCircle(LERP_ZONE, 59.9).r).toBe(560)
})

test("zoneCircle: mid-window lerps linearly", () => {
  const mid = zoneCircle(LERP_ZONE, 85)
  expect(mid.r).toBeCloseTo(470, 6)
  expect(mid.c[0]).toBeCloseTo(5, 6)
  expect(mid.c[1]).toBeCloseTo(0, 6)
})

test("zoneCircle: after shrink_end it is the next circle", () => {
  expect(zoneCircle(LERP_ZONE, 200)).toEqual({ c: [10, 0], r: 380 })
  expect(zoneCircle(LERP_ZONE, 110).r).toBe(380)
})

test("zoneCircle: degenerate shrink_start == shrink_end jumps whole-circle", () => {
  const deg = { ...LERP_ZONE, shrink_start: 60, shrink_end: 60 }
  expect(zoneCircle(deg, 59.999)).toEqual({ c: [0, 0], r: 560 })
  expect(zoneCircle(deg, 60)).toEqual({ c: [10, 0], r: 380 })
})

// ---------- pickUpgrade ----------

test("pickUpgrade: prefers highest-priority token wherever it appears (contains)", () => {
  expect(pickUpgrade([{ id: "servos" }, { id: "plating_plus" }])).toBe(1)
  expect(pickUpgrade([{ id: "gyros" }, { id: "laser_charge" }, { id: "armor_x" }])).toBe(2)
  expect(pickUpgrade([{ id: "gun_rate" }, { id: "laser_damage_mk2" }])).toBe(1)
})

test("pickUpgrade: resolves ties in priority order and falls back to 0", () => {
  expect(pickUpgrade([{ id: "lightning_efficiency" }, { id: "armor" }, { id: "nanites" }])).toBe(1)
  expect(pickUpgrade([{ id: "sensors" }, { id: "whatever" }])).toBe(0)
  expect(pickUpgrade([{ id: "capacitor" }, { id: "reactor_small" }])).toBe(1)
})

// ---------- Brain with stubbed deps ----------

const MAP = {} as GameMap

test("brain: mech outside the live zone moves toward the center and boosts", () => {
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps())
  const obs = mkObs({
    self: { pos: [600, 0, 0] },
    zone: { radius: 500, next_radius: 300 },
  })
  const act = brain.act(obs)
  expect(act.move![0]).toBeLessThan(-0.99)
  expect(Math.abs(act.move![1])).toBeLessThan(0.01)
  expect(act.boost).toBe(true)
})

test("brain (C1): decide target {id, kind} without pos → no throw, aim/fire pass through, strafes perpendicular", () => {
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps(targeting(5, "mech")))
  const obs = mkObs({ mechs: [mkEnemy([80, 0, 0])] })
  const act = brain.act(obs)
  expect(act.aim).toEqual([1.1, -0.2])
  expect(act.fire).toEqual([true, false])
  // gun band 50–90 m: at 80 m the move is pure perpendicular → [0, ±1]
  expect(Math.abs(act.move![1])).toBeGreaterThan(0.99)
  expect(Math.abs(act.move![0])).toBeLessThan(0.01)
  expect(act.boost).toBe(true) // energy 100 > 50, not planning lightning
})

test("brain (C1/M5): strafes the mech decide targets, not merely the nearest", () => {
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps(targeting(7, "mech")))
  // nearest (id 5) due east at 80 m; decide's target (id 7) due north at 150 m → band pushes toward +y
  const obs = mkObs({ mechs: [mkEnemy([80, 0, 0], 900, 5), mkEnemy([0, 150, 0], 400, 7)] })
  const act = brain.act(obs)
  expect(act.move![1]).toBeGreaterThan(0.5)
})

test("brain: primary lightning holds energy — no strafe boost", () => {
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps(targeting(5, "mech")))
  const obs = mkObs({
    self: { weapons: [GUN, { kind: "lightning", cooldown: 0, stats: {} }] },
    mechs: [mkEnemy([30, 0, 0])],
  })
  const act = brain.act(obs)
  expect(act.boost).toBeFalsy() // within lightning band 20–35 m, no boost
})

test("brain: offer present → pick follows the priority order", () => {
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps())
  const obs = mkObs({ offer: { id: 2, expires: 61, choices: [{ id: "sensors", name: "", desc: "" }, { id: "armor", name: "", desc: "" }] } })
  const act = brain.act(obs)
  expect(act.pick).toBe(1)
})

test("brain (I1): hp low and no enemy → stands still, stuck monitor told not commanding", () => {
  const seen: boolean[] = []
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps(noTarget, (_n, _s, c) => (seen.push(c), "none")))
  const act = brain.act(mkObs({ self: { hp: 300 } }))
  expect(act.move).toEqual([0, 0])
  expect(act.boost).toBeFalsy()
  expect(act.jump).toBeFalsy()
  expect(seen).toEqual([false])
})

test("brain (I5): heal with a crate targeted masks fire", () => {
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps(targeting(90, "crate", [0, -0.2], [true, false])))
  const act = brain.act(mkObs({ self: { hp: 300 }, crates: [{ id: 90, pos: [30, 0, 0], hp: 50 }] }))
  expect(act.move).toEqual([0, 0])
  expect(act.fire).toEqual([false, false])
})

test("brain (I5): loot window, no mech → walks toward decide's crate target within 60 m", () => {
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps(targeting(90, "crate")))
  const act = brain.act(mkObs({ crates: [{ id: 90, pos: [0, 40, 0], hp: 50 }] }))
  expect(act.move![1]).toBeGreaterThan(0.2)
  expect(act.fire).toEqual([true, false])
})

test("brain (M4): low hp with a mech visible beyond 200 m does NOT stand to heal", () => {
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps())
  const act = brain.act(mkObs({ self: { hp: 300, pos: [100, 0, 0] }, mechs: [mkEnemy([350, 0, 0])] }))
  expect(Math.hypot(act.move![0], act.move![1])).toBeGreaterThan(0)
})

test("brain (M6): low hp, landed uncontested drop → goes to capture instead of standing", () => {
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps())
  const drop = { id: 1, pos: [0, 50, 0] as [number, number, number], radius: 8, lands_in: 0, capture_s: 5, progress: 0, holder: null }
  const act = brain.act(mkObs({ self: { hp: 300 }, drops: [drop] }))
  expect(act.move![1]).toBeGreaterThan(0.99)
})

test("brain: disengaging (low hp, enemy healthier) moves away while keeping aim/fire", () => {
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps(targeting(5, "mech", [0, 0], [true, true])))
  const obs = mkObs({ self: { hp: 300 }, mechs: [mkEnemy([80, 0, 0], 500)] })
  const act = brain.act(obs)
  expect(act.move![0]).toBeLessThan(-0.99)
  expect(Math.abs(act.move![1])).toBeLessThan(0.01)
  expect(act.fire).toEqual([true, true])
})

test("brain (I4): pickup within radius is taken; laser swaps out mortar, lightning swaps out mortar", () => {
  const laser = { id: 8013, kind: "laser" as const, pos: [2, 0, 0] as [number, number, number] }
  const act = new Brain(mkHello(), MAP, () => true, stubDeps()).act(mkObs({ pickups: [laser] }))
  expect(act.take).toBe(8013)
  expect(act.drop).toBeUndefined() // slot 1 empty → goes straight in

  const mortarHeld = { weapons: [GUN, { kind: "mortar" as const, cooldown: 0, stats: {} }] as Observation["self"]["weapons"] }
  const swap = new Brain(mkHello(), MAP, () => true, stubDeps()).act(mkObs({ pickups: [laser], self: mortarHeld }))
  expect(swap.take).toBe(8013)
  expect(swap.drop).toBe(1)

  const lightning = { id: 8014, kind: "lightning" as const, pos: [2, 0, 0] as [number, number, number] }
  const swap2 = new Brain(mkHello(), MAP, () => true, stubDeps()).act(mkObs({ pickups: [lightning], self: mortarHeld }))
  expect(swap2.take).toBe(8014)
  expect(swap2.drop).toBe(1)
})

test("brain (I4): holding laser → never takes or walks to a weaker special (no circling)", () => {
  const laserHeld = { weapons: [GUN, { kind: "laser" as const, cooldown: 0, charge: 0.4, stats: {} }] as Observation["self"]["weapons"] }
  const near = { id: 1, kind: "mortar" as const, pos: [2, 0, 0] as [number, number, number] }
  const act = new Brain(mkHello(), MAP, () => true, stubDeps()).act(mkObs({ pickups: [near], self: laserHeld }))
  expect(act.take).toBeUndefined()
  expect(act.drop).toBeUndefined()

  // lightning 30 m north; idle goal is next_center due west → must not head north
  const far = { id: 2, kind: "lightning" as const, pos: [100, 30, 0] as [number, number, number] }
  const walk = new Brain(mkHello(), MAP, () => true, stubDeps()).act(
    mkObs({ pickups: [far], self: { ...laserHeld, pos: [100, 0, 0] } }),
  )
  expect(walk.move![0]).toBeLessThan(0)
  expect(Math.abs(walk.move![1])).toBeLessThan(1e-9)
})

test("brain (M3): pickup beyond 80 m is not walked to", () => {
  const far = { id: 3, kind: "laser" as const, pos: [100, 200, 0] as [number, number, number] }
  const act = new Brain(mkHello(), MAP, () => true, stubDeps()).act(mkObs({ pickups: [far], self: { pos: [100, 0, 0] } }))
  expect(Math.abs(act.move![1])).toBeLessThan(1e-9)
})

test("brain (I2): decide returning self.aim with no target falls through to sound-turn/scan", () => {
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps(noTarget))
  const obs = mkObs({
    self: { aim: [0.3, 0] },
    sounds: [
      { kind: "gun", bearing: -2.1, dist: "near", age: 0.4 },
      { kind: "lightning", bearing: 1.3, dist: "near", age: 0.1 },
    ],
  })
  expect(brain.act(obs).aim![0]).toBeCloseTo(1.3, 6)
})

test("brain (I2): decide's no-target bearing turn (aim ≠ self.aim) is kept", () => {
  const bearing: StubDecide = () => ({ aim: [2.0, 0], fire: [false, false], reload: false, target: null })
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps(bearing))
  const obs = mkObs({ sounds: [{ kind: "gun", bearing: -2.1, dist: "near", age: 0.1 }] })
  expect(brain.act(obs).aim).toEqual([2.0, 0])
})

test("brain (I6): combat state starts lastDamaged=null and brain never writes it", () => {
  let seen: CombatState | null = null
  const spy: StubDecide = (obs, state) => ((seen = state), noTarget(obs, state, null))
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps(spy))
  brain.act(mkObs({ events: [{ type: "damaged", amount: 20, weapon: "zone", bearing: null, by: null }] }))
  expect(seen!.lastDamaged).toBeNull()
  expect(seen!.gravity).toBe(22)
})

test("brain (I3): silent idle stays under step_speed, also with upgraded walk_speed", () => {
  for (const walk of [11.7, 20]) {
    const brain = new Brain(mkHello(), MAP, () => true, stubDeps())
    const act = brain.act(mkObs({ self: { pos: [100, 0, 0], stats: { walk_speed: walk, boost_speed: 19.5, eye_height: 8 } } }))
    const speed = Math.hypot(act.move![0], act.move![1]) * walk
    expect(speed).toBeGreaterThan(0)
    expect(speed).toBeLessThanOrEqual(5)
    expect(act.boost).toBeFalsy()
  }
})

test("brain (M2): stuck 'perp' rotates the move for ~0.5 s, not one tick", () => {
  const verdicts: StuckVerdict[] = ["perp", "none", "none", "none", "none", "none", "none", "none"]
  let i = 0
  const brain = new Brain(mkHello(), MAP, () => true, stubDeps(noTarget, () => verdicts[i++] ?? "none"))
  const moves: Vec2[] = []
  for (let k = 0; k < 8; k++) moves.push(brain.act(mkObs({ t: 10 + k * 0.1, self: { pos: [100, 0, 0] } })).move!)
  // idle goal due west → rotated move is ±y for ticks t=10.0..10.4, then back to −x
  for (let k = 0; k <= 4; k++) expect(Math.abs(moves[k]![0])).toBeLessThan(1e-9)
  expect(moves[7]![0]).toBeLessThan(0)
})

// ---------- seeded strafe ----------

test("brain: same seed → identical strafe flip schedule across instances", () => {
  const run = () => {
    const brain = new Brain(mkHello(42), MAP, () => true, stubDeps(targeting(5, "mech")))
    const moves: Vec2[] = []
    for (let i = 0; i < 101; i++) {
      const obs = mkObs({ t: 100 + i * 0.1, mechs: [mkEnemy([80, 0, 0])] })
      moves.push(brain.act(obs).move!)
    }
    return moves
  }
  const a = run()
  const b = run()
  expect(b).toEqual(a)

  // pure perpendicular at 80 m → the y-component flips over time (at least once in 10 s)
  let flips = 0
  for (let i = 1; i < a.length; i++) if (a[i]![1] !== a[i - 1]![1]) flips++
  expect(flips).toBeGreaterThanOrEqual(1)
})

// ---------- integration: real decide + Navigator + StuckMonitor on a synthetic map ----------

/** 200×200 m all-walkable flat grid centered on the origin (GameMap contract shape). */
function flatMap(): GameMap {
  const nx = 200
  const ny = 200
  return {
    nx,
    ny,
    step: 1,
    minX: -100,
    minY: -100,
    terrain: new Int16Array(nx * ny),
    cellSurf: new Float32Array(nx * ny),
    cellFeasible: new Uint8Array(nx * ny).fill(1),
    cellPrisms: new Array(nx * ny).fill(null),
  }
}

test("brain integration: real decide/nav/stuck — fight fires, heal-with-crate holds fire, idle stays silent", () => {
  const map = flatMap()

  // fight tick: enemy 80 m east, our aim already on the lead → must not throw and must fire the gun
  const fight = new Brain(mkHello(), map, () => true)
  const fa = fight.act(mkObs({ mechs: [mkEnemy([80, 0, 0])] }))
  expect(fa.fire![0]).toBe(true)
  expect(Math.hypot(fa.move![0], fa.move![1])).toBeGreaterThan(0.5)

  // heal with a crate visible and aimed: real decide wants to shoot it; brain masks fire
  const crate = { id: 90, pos: [30, 0, 0] as [number, number, number], hp: 50 }
  const pitch = Math.atan2(1 - 7, 30) // crate center 1 m vs muzzle 7 m
  const heal = new Brain(mkHello(), map, () => true)
  const ha = heal.act(mkObs({ self: { hp: 300, aim: [0, pitch] }, crates: [crate] }))
  expect(ha.fire).toEqual([false, false])
  expect(ha.move).toEqual([0, 0])

  // silent idle: walking toward next_center at ≤ step_speed, no boost
  const idle = new Brain(mkHello(), map, () => true)
  const ia = idle.act(mkObs({ self: { pos: [50, 0, 0] } }))
  const speed = Math.hypot(ia.move![0], ia.move![1]) * 11.7
  expect(speed).toBeGreaterThan(0)
  expect(speed).toBeLessThanOrEqual(5)
  expect(ia.boost).toBeFalsy()
})
